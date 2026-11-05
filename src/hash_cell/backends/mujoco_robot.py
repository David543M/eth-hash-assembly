"""MuJoCo backend of the Robot interface. Both robots share one MjModel/MjData;
`step_fn` advances the simulation (and syncs the viewer if there is one)."""
from __future__ import annotations

import mujoco
import numpy as np

from ..ik import solve_grasp_ik
from ..robot_interface import Robot


class MujocoRobot(Robot):
    def __init__(self, model, data, prefix: str, step_fn):
        self.name = prefix.rstrip("_")
        self.m, self.d, self.p, self.step = model, data, prefix, step_fn
        self.joints = [f"{prefix}joint{i}" for i in range(1, 7)]
        self.act = [model.actuator(j).id for j in self.joints]
        self.grip_act = [model.actuator(f"{prefix}finger_{k}").id for k in "lr"]
        self.qadr = [model.joint(j).qposadr[0] for j in self.joints]

    def joint_positions(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def move_joints(self, q, duration: float = 1.0) -> None:
        q0 = self.joint_positions()
        n = max(1, int(duration / self.m.opt.timestep))
        for k in range(1, n + 1):
            s = k / n
            s = 3 * s**2 - 2 * s**3          # smooth start and stop
            self.d.ctrl[self.act] = q0 + s * (np.asarray(q) - q0)
            self.step()

    def move_tcp(self, pos, R, duration: float = 1.0) -> None:
        q, ep, er, ok = solve_grasp_ik(self.m, self.d, f"{self.p}tcp", self.joints, pos, R,
                                 q_init=self.joint_positions())
        if not ok:
            raise RuntimeError(f"{self.name}: IK failed for {np.round(pos, 3)} (err {ep:.2e} m)")
        self.move_joints(q, duration)

    def set_gripper(self, opening: float, duration: float = 0.4) -> None:
        self.d.ctrl[self.grip_act] = max(opening / 2, 0.002)
        for _ in range(int(duration / self.m.opt.timestep)):
            self.step()
