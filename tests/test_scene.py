"""Scene sanity and reachability: every pick and place pose must be reachable
with the tool pointing down, for the robot that handles that part."""
import mujoco
import numpy as np
import pytest

from hash_cell.frames import tool_down, yaw_axis
from hash_cell.ik import solve_grasp_ik
from hash_cell.scene import build_scene, load_config

CFG = load_config()


@pytest.fixture(scope="module")
def sim():
    from hash_cell.scene import MODELS
    path = MODELS / "scene.xml"
    path.write_text(build_scene(CFG), encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def test_loads_without_initial_contacts(sim):
    m, d = sim
    assert m.nu == 16 and d.ncon == 0


def test_home_is_stable(sim):
    m, _ = sim
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = d.qpos[:16].copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.abs(d.qpos[:16] - q0).max() < 1e-3


def _poses():
    a, p1, p2 = CFG["assembly"], CFG["profiles"]["pos1"], CFG["profiles"]["pos2"]
    plate = CFG["fixture"]["plate"][2]
    g = 0.016  # TCP height above the part bottom when grasping
    # (robot, part, pick xy+yaw, place centre, place closing axis)
    d1, d2 = a["pos1_centre_distance"] / 2, a["pos2_centre_distance"] / 2
    return [
        ("A", CFG["supply"]["pos1_a"], [0, -d1, plate + g], [0, 1, 0]),
        ("B", CFG["supply"]["pos1_b"], [0, d1, plate + g], [0, 1, 0]),
        ("A", CFG["supply"]["pos2_a"], [-d2, 0, plate + p1["height"] + g], [1, 0, 0]),
        ("B", CFG["supply"]["pos2_b"], [d2, 0, plate + p1["height"] + g], [1, 0, 0]),
    ]


@pytest.mark.parametrize("robot,supply,place,axis", _poses())
@pytest.mark.parametrize("approach", [0.0, 0.08])
def test_pick_and_place_reachable(sim, robot, supply, place, axis, approach):
    m, d = sim
    joints = [f"{robot}_joint{i}" for i in range(1, 7)]
    home = CFG["robots"][robot]["home_q"]
    x, y, yaw = supply
    for pos, R in (([x, y, 0.016 + approach], tool_down(yaw_axis(yaw))),
                   (np.add(place, [0, 0, approach]), tool_down(axis))):
        _, ep, er, ok = solve_grasp_ik(m, d, f"{robot}_tcp", joints, pos, R, q_init=home)
        assert ok, f"{robot} cannot reach {np.round(pos, 3)}: {ep:.1e} m, {er:.1e} rad"
