"""Assembly cycle with two robots running concurrently in one simulation.

Each robot runs a *program*: a Python generator that sets its actuator targets and
yields once per simulation step. A scheduler advances both programs every step, so
the robots move at the same time. Coordination uses two mechanisms:

- `ZoneLock`: only one robot may be inside the fixture zone at a time (requirement S2).
- precedence: a Pos 2 is placed only when both Pos 1 are in the fixture.

Every action is logged with its start and end time, which gives the cycle timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .frames import tool_down, yaw_axis
from .ik import FLIP_Z, solve_ik

HOVER = 0.10          # m above the pick / place surface
GRASP_H = 0.016       # TCP height above the part bottom when grasping
CLEARANCE = 0.0005    # part bottom above its support when released
SETTLE = 0.3          # s, wait for the servos to settle before measuring / releasing


@dataclass
class Event:
    robot: str
    action: str
    t0: float
    t1: float = float("nan")


@dataclass
class ZoneLock:
    owner: str | None = None
    history: list = field(default_factory=list)

    def request(self, who: str, t: float) -> bool:
        if self.owner in (None, who):
            if self.owner is None:
                self.history.append((t, who, "enter"))
            self.owner = who
            return True
        return False

    def release(self, who: str, t: float) -> None:
        if self.owner == who:
            self.owner = None
            self.history.append((t, who, "leave"))


class RobotProgram:
    """Motion primitives for one robot, written as generators (one yield = one sim step)."""

    def __init__(self, model, data, prefix: str, cfg: dict, log: list, tcp_noise=None):
        self.m, self.d, self.p, self.name = model, data, prefix, prefix.rstrip("_")
        self.cfg, self.log = cfg, log
        self.joints = [f"{prefix}joint{i}" for i in range(1, 7)]
        self.act = np.array([model.actuator(j).id for j in self.joints])
        self.fingers = np.array([model.actuator(f"{prefix}finger_{k}").id for k in "lr"])
        self.site = f"{prefix}tcp"
        self.dt = model.opt.timestep
        self.tcp_noise = tcp_noise          # callable -> 3-vector: robot positioning error at a placing location
        self.bias = None                    # error currently applied (one draw per placement = systematic bias)

    # --- helpers -----------------------------------------------------------------
    def _steps(self, T):
        return max(1, int(round(T / self.dt)))

    def _ik(self, pos, R, try_flip=True, offset=None):
        """IK from the current joint targets; for a symmetric gripper try R and R flipped.
        `offset`: part position in the TCP frame, so that the *part* lands on `pos`."""
        seed = self.d.ctrl[self.act].copy()
        best = None
        for Rc in ((R, R @ FLIP_Z) if try_flip else (R,)):
            target = np.asarray(pos, float) - (Rc @ offset if offset is not None else 0.0)
            if self.bias is not None:
                target = target + self.bias
            q, ep, er, ok = solve_ik(self.m, self.d, self.site, self.joints, target, Rc, q_init=seed)
            cost = (not ok, np.abs(q - seed).sum())
            if best is None or cost < best[0]:
                best = (cost, q, ok, ep)
        if not best[2]:
            raise RuntimeError(f"{self.name}: IK failed for {np.round(pos, 3)} ({best[3]:.1e} m)")
        return best[1]

    def _event(self, action):
        ev = Event(self.name, action, self.d.time)
        self.log.append(ev)
        return ev

    # --- primitives ----------------------------------------------------------------
    def move_joints(self, q, T, action=None):
        ev = self._event(action) if action else None
        q0, n = self.d.ctrl[self.act].copy(), self._steps(T)
        for k in range(1, n + 1):
            s = k / n
            s = 3 * s * s - 2 * s ** 3
            self.d.ctrl[self.act] = q0 + s * (q - q0)
            yield
        if ev:
            ev.t1 = self.d.time

    def move_tcp(self, pos, R, T, action=None, offset=None):
        yield from self.move_joints(self._ik(pos, R, offset=offset), T, action)

    def grip(self, opening, T, action=None):
        ev = self._event(action) if action else None
        self.d.ctrl[self.fingers] = max(opening / 2, 0.002)
        for _ in range(self._steps(T)):
            yield
        if ev:
            ev.t1 = self.d.time

    def settle(self, T):
        for _ in range(self._steps(T)):
            yield

    def wait_until(self, cond, action):
        ev = None
        while not cond():
            ev = ev or self._event(action)
            yield
        if ev:
            ev.t1 = self.d.time

    def part_in_tcp(self, part: str):
        """Measured position of the part (bottom centre) in the TCP frame. In the real cell this is
        what a camera or a laser check after grasping would give; here it is read from the simulation."""
        R = self.d.site(self.site).xmat.reshape(3, 3)
        return R.T @ (self.d.body(part).xpos - self.d.site(self.site).xpos)


def transfer(rp: RobotProgram, part: str, supply, place_xy, place_yaw: float, support_z: float,
             lock: ZoneLock, ready, placed: list, next_hover=None):
    """Pick `part` from its magazine and place it in the fixture."""
    cfg = rp.cfg
    x, y, yaw = supply
    R_pick = tool_down(yaw_axis(yaw))
    R_place = tool_down(yaw_axis(place_yaw))
    open_w = cfg["gripper"]["max_opening"]
    yield from rp.grip(open_w, 0.2)
    yield from rp.move_tcp([x, y, HOVER], R_pick, 1.4, f"approach {part}")
    # wait above the part, gripper open, for the precedence condition and the fixture zone:
    # the part is gripped only when it can be placed, so it is never held during a wait
    yield from rp.wait_until(ready, "wait (Pos 1 not placed)")
    yield from rp.wait_until(lambda: lock.request(rp.name, rp.d.time), "wait (zone busy)")
    yield from rp.move_tcp([x, y, GRASP_H], R_pick, 0.8, f"pick {part}")
    yield from rp.grip(0.0, 0.5, f"grip {part}")
    yield from rp.move_tcp([x, y, HOVER], R_pick, 0.7, f"lift {part}")
    px, py = place_xy
    # robot inaccuracy at the fixture: one systematic error for this placing location
    rp.bias = rp.tcp_noise() if rp.tcp_noise is not None else None
    yield from rp.move_tcp([px, py, support_z + HOVER], R_place, 1.4, f"transfer {part}")
    yield from rp.settle(SETTLE)
    # check where the part sits in the gripper (camera / laser in the real cell) and place the part, not the TCP
    offset = rp.part_in_tcp(part)
    target = np.array([px, py, support_z + CLEARANCE])
    yield from rp.move_tcp(target, R_place, 0.9, f"place {part}", offset=offset)
    yield from rp.settle(SETTLE)
    yield from rp.grip(open_w, 0.4, f"release {part}")
    placed.append(part)
    yield from rp.move_tcp(target + [0, 0, 0.012], R_place, 0.5, "retract")
    yield from rp.move_tcp([px, py, support_z + HOVER], R_place, 0.5, "retract")
    rp.bias = None
    if next_hover is not None:
        nx, ny, nyaw = next_hover
        yield from rp.move_tcp([nx, ny, HOVER], tool_down(yaw_axis(nyaw)), 1.2, "leave zone")
    else:
        q_home = np.array(cfg["robots"][rp.name]["home_q"])
        yield from rp.move_joints(q_home, 1.4, "leave zone")
    lock.release(rp.name, rp.d.time)


def build_programs(model, data, cfg, log, lock, tcp_noise=None):
    """The two robot programs of one assembly cycle (A: Pos 1 at y<0 and Pos 2 at x<0; B mirrored)."""
    plate = cfg["fixture"]["plate"][2]
    h1 = cfg["profiles"]["pos1"]["height"]
    d1 = cfg["assembly"]["pos1_centre_distance"] / 2
    d2 = cfg["assembly"]["pos2_centre_distance"] / 2
    sup = cfg["supply"]
    placed: list[str] = []
    pos1_done = lambda: sum(p.startswith("pos1") for p in placed) == 2
    always = lambda: True
    progs = {}
    for name, sgn, p1, p2 in (("A", -1, "pos1_a", "pos2_a"), ("B", 1, "pos1_b", "pos2_b")):
        rp = RobotProgram(model, data, f"{name}_", cfg, log, tcp_noise)

        def program(rp=rp, sgn=sgn, p1=p1, p2=p2):
            yield from transfer(rp, p1, sup[p1], (0.0, sgn * d1), 0.0, plate, lock, always, placed,
                                next_hover=sup[p2])
            yield from transfer(rp, p2, sup[p2], (sgn * d2, 0.0), np.pi / 2, plate + h1, lock, pos1_done,
                                placed)

        progs[name] = program()
    return progs, placed


def run(model, data, programs: dict, on_step=None, t_max: float = 60.0):
    """Advance all programs together until they are all finished."""
    active = dict(programs)
    while active and data.time < t_max:
        for k in list(active):
            try:
                next(active[k])
            except StopIteration:
                del active[k]
        mujoco.mj_step(model, data)
        if on_step:
            on_step()
    # let the parts settle
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if on_step:
            on_step()
    return not active


def placement_errors(model, data, cfg) -> dict[str, dict[str, float]]:
    """Final pose of each part against its nominal assembled pose (mm, deg)."""
    plate = cfg["fixture"]["plate"][2]
    h1 = cfg["profiles"]["pos1"]["height"]
    d1 = cfg["assembly"]["pos1_centre_distance"] / 2
    d2 = cfg["assembly"]["pos2_centre_distance"] / 2
    nominal = {"pos1_a": (0, -d1, plate, 0.0), "pos1_b": (0, d1, plate, 0.0),
               "pos2_a": (-d2, 0, plate + h1, 90.0), "pos2_b": (d2, 0, plate + h1, 90.0)}
    out = {}
    for part, (x, y, z, yaw) in nominal.items():
        p = data.body(part).xpos
        R = data.body(part).xmat.reshape(3, 3)
        ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        dyaw = (ang - yaw + 90) % 180 - 90          # profiles are symmetric: 180 deg is the same pose
        out[part] = {"dx": 1e3 * (p[0] - x), "dy": 1e3 * (p[1] - y), "dz": 1e3 * (p[2] - z), "dyaw": dyaw}
    return out
