"""Build the MuJoCo scene (MJCF) of the hash assembly cell from config/cell.yaml.

The xArm6 kinematics and inertias are taken from the official xarm_ros2
description (xarm_description/config, BSD licence, see models/xarm6/LICENSE_xarm_ros2).
Everything else (table, fixture, gripper, profiles) is built from primitives.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "cell.yaml"
MODELS = ROOT / "models"

# xArm6 joint frames, from xarm6_default_kinematics.yaml (xyz in m, roll in rad)
XARM6_JOINTS = [
    ((0.0, 0.0, 0.267), 0.0),
    ((0.0, 0.0, 0.0), -1.5708),
    ((0.0535, -0.2845, 0.0), 0.0),
    ((0.0775, 0.3425, 0.0), -1.5708),
    ((0.0, 0.0, 0.0), 1.5708),
    ((0.076, 0.097, 0.0), -1.5708),
]
# joint limits, from xarm6_robot_macro.xacro
XARM6_LIMITS = [
    (-math.pi * 0.99, math.pi * 0.99),
    (-2.059, 2.0944),
    (-math.pi * 0.99, 0.19198),
    (-math.pi * 0.99, math.pi * 0.99),
    (-1.69297, math.pi * 0.99),
    (-math.pi * 0.99, math.pi * 0.99),
]
# link inertias, from xarm6_type6_HT_BR2.yaml: mass, com, (ixx, iyy, izz, ixy, ixz, iyz)
XARM6_INERTIA = [
    (2.28, (-0.00033, 0.02745, -0.014), (0.005768, 0.005052, 0.003212, -1.6654e-05, 2.528e-06, -0.0008984)),
    (2.545, (0.04025, -0.3603, 0.0102), (0.076034, 0.0131983, 0.068486, 0.0086817, 0.00344615, -0.01936147)),
    (1.836, (0.06865, -0.1133, 0.0132), (0.0068547, 0.00408725, 0.0064999, 0.0018093, 0.001098, -0.00212676)),
    (1.41, (0.00033, 0.0148, -0.0307), (0.004887, 0.0044147, 0.0013373, -6.996e-05, -1.334e-05, -0.0002229)),
    (1.354, (0.0652, 0.03179, 0.00311), (0.001473, 0.00185, 0.0023493, -0.00046355, 0.00029315, 0.000135)),
    (0.17, (0.0, -0.00677, -0.01098), (9.3e-05, 5.87e-05, 0.000132, 0.0, 0.0, -3.6e-06)),
]
XARM6_KP = [1500, 1500, 1000, 500, 300, 200]   # position-servo gains (sim only)


def load_config(path: Path = CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _v(xs) -> str:
    return " ".join(f"{x:.6g}" for x in xs)


def _robot(p: str, cfg: dict) -> str:
    """One xArm6 + parallel gripper, every name prefixed with p (e.g. 'A_')."""
    r = cfg["robots"][p.rstrip("_")]
    g = cfg["gripper"]
    open_half = g["max_opening"] / 2
    palm, fing = g["palm"], g["finger"]
    finger_z = 2 * palm[2] + fing[2]
    s = [f'<body name="{p}link_base" pos="{_v(r["base_pos"])}" euler="0 0 {r["base_yaw"]:.6g}">',
         '<geom class="xarm_visual" mesh="link_base"/>']
    for i, ((xyz, roll), (lo, hi), (m, com, I)) in enumerate(zip(XARM6_JOINTS, XARM6_LIMITS, XARM6_INERTIA), 1):
        s.append(f'<body name="{p}link{i}" pos="{_v(xyz)}" euler="{roll:.6g} 0 0" gravcomp="1">')
        s.append(f'<inertial pos="{_v(com)}" mass="{m}" fullinertia="{_v(I)}"/>')
        s.append(f'<joint name="{p}joint{i}" class="xarm_joint" range="{lo:.5f} {hi:.5f}"/>')
        s.append(f'<geom class="xarm_visual" mesh="link{i}"/>')
    # gripper on the flange (link6 frame, tool z out of the flange)
    s += [
        f'<body name="{p}gripper" gravcomp="1">',
        f'<geom class="gripper" type="box" size="{_v(palm)}" pos="0 0 {palm[2]:.6g}" mass="0.4"/>',
        f'<site name="{p}tcp" pos="0 0 {g["tcp_offset"]:.6g}" size="0.004" rgba="1 0 0 1"/>',
        f'<body name="{p}finger_l" pos="0 0 {finger_z:.6g}" gravcomp="1">',
        f'<joint name="{p}finger_l" type="slide" axis="0 1 0" range="0.002 {open_half:.6g}" damping="400"/>',
        f'<geom class="finger" type="box" size="{_v(fing)}" pos="0 {fing[1]:.6g} 0" mass="0.05"/>',
        '</body>',
        f'<body name="{p}finger_r" pos="0 0 {finger_z:.6g}" gravcomp="1">',
        f'<joint name="{p}finger_r" type="slide" axis="0 -1 0" range="0.002 {open_half:.6g}" damping="400"/>',
        f'<geom class="finger" type="box" size="{_v(fing)}" pos="0 {-fing[1]:.6g} 0" mass="0.05"/>',
        '</body>',
        '</body>',
    ]
    s.append("</body>" * 7)  # link6 ... link1, link_base
    return "\n".join(s)


def _profile_geoms(prof: dict, rgba: str, cls: str) -> str:
    """U-profile, flanges down, body frame at the bottom centre, length along local x."""
    w, h, t, L = prof["width"], prof["height"], prof["thickness"], prof["length"]
    web = f'<geom class="{cls}" type="box" size="{L/2:.6g} {w/2:.6g} {t/2:.6g}" pos="0 0 {h - t/2:.6g}" rgba="{rgba}"/>'
    fl = (h - t) / 2
    flanges = "".join(
        f'<geom class="{cls}" type="box" size="{L/2:.6g} {t/2:.6g} {fl:.6g}" pos="0 {sgn*(w/2 - t/2):.6g} {fl:.6g}" rgba="{rgba}"/>'
        for sgn in (1, -1))
    return web + flanges


def _profiles(cfg: dict) -> tuple[str, list[tuple[float, ...]]]:
    """Loose parts in their magazines (free bodies). Returns xml and initial qpos (7 per part)."""
    xml, qpos = [], []
    for name, (x, y, yaw) in cfg["supply"].items():
        kind = name.split("_")[0]
        rgba = "0.70 0.72 0.75 1" if kind == "pos1" else "0.55 0.62 0.72 1"
        xml.append(f'<body name="{name}" pos="{x} {y} 0.0005" euler="0 0 {yaw}">'
                   f'<freejoint name="{name}"/>{_profile_geoms(cfg["profiles"][kind], rgba, "part")}</body>')
        qpos.append((x, y, 0.0005, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)))
    return "\n".join(xml), qpos


def _fixture(cfg: dict) -> str:
    f, p1, p2 = cfg["fixture"], cfg["profiles"]["pos1"], cfg["profiles"]["pos2"]
    a = cfg["assembly"]
    plate_h = f["plate"][2]
    sh = f["stop_height"] / 2
    d1, d2 = a["pos1_centre_distance"] / 2, a["pos2_centre_distance"] / 2
    s = [f'<body name="fixture" pos="0 0 0">',
         f'<geom name="plate" class="fixture" type="box" size="{f["plate"][0]/2} {f["plate"][1]/2} {plate_h/2}" pos="0 0 {plate_h/2}"/>']
    # Pos 1: outer side rails (datum B) and end stops on -x (datum C)
    for sgn, tag in ((1, "l"), (-1, "r")):
        y_rail = sgn * (d1 + p1["width"] / 2 + 0.005)
        s.append(f'<geom name="rail_pos1_{tag}" class="fixture" type="box" size="0.12 0.005 {sh}" pos="0 {y_rail:.6g} {plate_h + sh:.6g}"/>')
        s.append(f'<geom name="stop_pos1_{tag}" class="fixture" type="box" size="0.005 0.02 {sh}" pos="{-(p1["length"]/2 + 0.005):.6g} {sgn*d1:.6g} {plate_h + sh:.6g}"/>')
    # Pos 2: end stops on -y, posts standing on the plate
    post_h = (p1["height"] + p2["height"] / 2) / 2
    for sgn, tag in ((1, "l"), (-1, "r")):
        s.append(f'<geom name="stop_pos2_{tag}" class="fixture" type="box" size="0.015 0.005 {post_h:.6g}" '
                 f'pos="{sgn*d2:.6g} {-(p2["length"]/2 + 0.005):.6g} {plate_h + post_h:.6g}"/>')
    s.append("</body>")
    # nominal assembled positions, visual only (ghosts)
    z2 = plate_h + p1["height"]
    for sgn, tag in ((1, "l"), (-1, "r")):
        s.append(f'<body name="target_pos1_{tag}" pos="0 {sgn*d1:.6g} {plate_h}">{_profile_geoms(p1, "0.2 0.8 0.3 0.25", "ghost")}</body>')
        s.append(f'<body name="target_pos2_{tag}" pos="{sgn*d2:.6g} 0 {z2:.6g}" euler="0 0 1.5708">{_profile_geoms(p2, "0.2 0.8 0.3 0.25", "ghost")}</body>')
    return "\n".join(s)


def build_scene(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    t = cfg["table"]
    rho = cfg["steel_density"]
    parts_xml, parts_qpos = _profiles(cfg)
    home = []
    for name in ("A", "B"):
        half = cfg["gripper"]["max_opening"] / 2
        home += list(cfg["robots"][name]["home_q"]) + [half, half]
    key_qpos = _v(home + [v for q in parts_qpos for v in q])
    actuators = []
    for p in ("A_", "B_"):
        for i, kp in enumerate(XARM6_KP, 1):
            lo, hi = XARM6_LIMITS[i - 1]
            actuators.append(f'<position name="{p}joint{i}" joint="{p}joint{i}" kp="{kp}" ctrlrange="{lo:.5f} {hi:.5f}"/>')
        f = cfg["gripper"]["grip_force"]
        for side in ("l", "r"):
            actuators.append(f'<position name="{p}finger_{side}" joint="{p}finger_{side}" kp="5000" '
                             f'forcerange="-{f} {f}" ctrlrange="0.002 {cfg["gripper"]["max_opening"]/2:.6g}"/>')
    ctrl0 = []
    for name in ("A", "B"):
        ctrl0 += list(cfg["robots"][name]["home_q"]) + [cfg["gripper"]["max_opening"] / 2] * 2

    return f"""<mujoco model="hash_assembly_cell">
<compiler angle="radian" meshdir="xarm6/meshes" autolimits="true"/>
<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
<default>
  <default class="xarm_joint"><joint axis="0 0 1" damping="10" armature="0.1"/></default>
  <default class="xarm_visual"><geom type="mesh" rgba="0.92 0.92 0.92 1" contype="1" conaffinity="1" group="0"/></default>
  <default class="gripper"><geom rgba="0.15 0.15 0.15 1"/></default>
  <default class="finger"><geom rgba="0.25 0.25 0.3 1" friction="1.2 0.02 0.001" condim="4"/></default>
  <default class="fixture"><geom rgba="0.45 0.35 0.25 1"/></default>
  <default class="part"><geom density="{rho}" friction="0.6 0.01 0.001" condim="4" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <default class="ghost"><geom contype="0" conaffinity="0" group="2" mass="0"/></default>
</default>
<asset>
  <mesh name="link_base" file="link_base.stl"/>
  {"".join(f'<mesh name="link{i}" file="link{i}.stl"/>' for i in range(1, 7))}
  <texture name="grid" type="2d" builtin="checker" rgb1=".25 .26 .28" rgb2=".3 .31 .33" width="512" height="512"/>
  <material name="floor" texture="grid" texrepeat="8 8"/>
</asset>
<worldbody>
  <light pos="0 -1 2.5" dir="0 0.4 -1" directional="true"/>
  <light pos="0 1 2.5" dir="0 -0.4 -1" diffuse="0.4 0.4 0.4"/>
  <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" material="floor"/>
  <geom name="table" type="box" size="{t['size'][0]/2} {t['size'][1]/2} {t['size'][2]/2}" pos="0 0 {-t['size'][2]/2}" rgba="0.82 0.82 0.8 1"/>
  <camera name="overview" pos="0 -1.35 0.95" xyaxes="1 0 0 0 0.58 0.81"/>
  <camera name="top" pos="0 0 1.3" xyaxes="1 0 0 0 1 0"/>
  {_fixture(cfg)}
  {_robot("A_", cfg)}
  {_robot("B_", cfg)}
  {parts_xml}
</worldbody>
<actuator>
{chr(10).join(actuators)}
</actuator>
<keyframe>
  <key name="home" qpos="{key_qpos}" ctrl="{_v(ctrl0)}"/>
</keyframe>
</mujoco>
"""


def write_scene(path: Path = MODELS / "scene.xml") -> Path:
    path.write_text(build_scene(), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_scene())
