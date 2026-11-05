"""Build the scene from config/cell.yaml and open it in the MuJoCo viewer.

    python scripts/view_scene.py            # interactive viewer, robots at home
    python scripts/view_scene.py --demo     # robot A moves above its Pos 1 magazine and back

On macOS use `mjpython` instead of `python`.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
import mujoco.viewer
import numpy as np

from hash_cell.backends.mujoco_robot import MujocoRobot
from hash_cell.frames import tool_down
from hash_cell.scene import load_config, write_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    path = write_scene()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    cfg = load_config()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[2] = 1          # show the green target ghosts

        def step():
            t0 = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - t0)))

        if args.demo:
            a = MujocoRobot(model, data, "A_", step)
            x, y, _ = cfg["supply"]["pos1_a"]
            home = a.joint_positions()
            a.move_tcp(np.array([x, y, 0.12]), tool_down([0, 1, 0]), 2.0)
            a.move_tcp(np.array([x, y, 0.016]), tool_down([0, 1, 0]), 1.5)
            a.set_gripper(0.0)            # close: force-limited squeeze on the flanges
            a.move_tcp(np.array([x, y, 0.12]), tool_down([0, 1, 0]), 1.5)
            a.move_joints(home, 2.0)
        while viewer.is_running():
            step()


if __name__ == "__main__":
    main()
