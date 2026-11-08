"""Kinematic analysis of the xArm6 in this cell, from the MuJoCo model built from xarm_ros2.

    python scripts/analyze_xarm6.py

1. Margins of every pick and place pose of the cycle to the three singularity families
   (elbow, shoulder, wrist alignment), dexterity, and how joint errors map to TCP error.
2. A dexterity map of robot A over the table, tool pointing down, at placing height.

Outputs: docs/figures/fig_xarm6_dexterity.png, docs/figures/xarm6_poses.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

from hash_cell.cycle import GRASP_H, HOVER
from hash_cell.frames import tool_down, yaw_axis
from hash_cell.ik import solve_grasp_ik
from hash_cell.scene import load_config, write_scene

L_CHAR = 0.30            # m, characteristic length to mix linear and angular Jacobian rows
ELBOW_J3 = np.degrees(-(np.pi - np.arctan2(0.0535, 0.2845) - np.arctan2(0.0775, 0.3425)))   # stretched arm
JOINT_ERR_DEG = 0.01     # joint error used to illustrate error amplification
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class Arm:
    def __init__(self, model, prefix):
        self.m, self.d, self.p = model, mujoco.MjData(model), prefix
        self.joints = [f"{prefix}joint{i}" for i in range(1, 7)]
        self.qa = [model.joint(j).qposadr[0] for j in self.joints]
        self.da = [model.joint(j).dofadr[0] for j in self.joints]
        self.site = model.site(f"{prefix}tcp").id
        self.base = model.body(f"{prefix}link_base").id

    def fk(self, q):
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def metrics(self, q):
        self.fk(q)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        Jp, Jr = jp[:, self.da], jr[:, self.da]
        s = np.linalg.svd(np.vstack([Jp, L_CHAR * Jr]), compute_uv=False)
        # shoulder: horizontal distance of the wrist (J5 anchor) to the J1 axis
        j1 = self.m.joint(self.joints[0]).id
        w = self.d.xanchor[self.m.joint(self.joints[4]).id] - self.d.xanchor[j1]
        a1 = self.d.xaxis[j1]
        shoulder = np.linalg.norm(w - (w @ a1) * a1)
        dq = np.radians(JOINT_ERR_DEG)
        return {
            "inv_cond": s[-1] / s[0],                                       # 1 = isotropic, 0 = singular
            "elbow_margin_deg": abs(np.degrees(q[2]) - ELBOW_J3),
            "shoulder_mm": 1e3 * shoulder,
            "wrist_J5_deg": np.degrees(q[4]),
            # worst-case TCP error if every joint is off by JOINT_ERR_DEG (sum of column norms)
            "tcp_err_um": 1e6 * dq * np.linalg.norm(Jp, axis=0).sum(),
        }


def cycle_poses(cfg):
    plate = cfg["fixture"]["plate"][2]
    h1 = cfg["profiles"]["pos1"]["height"]
    d1 = cfg["assembly"]["pos1_centre_distance"] / 2
    d2 = cfg["assembly"]["pos2_centre_distance"] / 2
    s = cfg["supply"]
    out = []
    for r, sgn, p1, p2 in (("A", -1, "pos1_a", "pos2_a"), ("B", 1, "pos1_b", "pos2_b")):
        for part, (x, y, yaw), place, axis, z in (
                (p1, s[p1], (0.0, sgn * d1), yaw_axis(0.0), plate),
                (p2, s[p2], (sgn * d2, 0.0), yaw_axis(np.pi / 2), plate + h1)):
            out.append((r, f"pick {part}", [x, y, GRASP_H], tool_down(yaw_axis(yaw))))
            out.append((r, f"place {part}", [place[0], place[1], z + GRASP_H], tool_down(axis)))
            out.append((r, f"above place {part}", [place[0], place[1], z + HOVER], tool_down(axis)))
    return out


def main():
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(write_scene()))
    arms = {"A": Arm(model, "A_"), "B": Arm(model, "B_")}
    data0 = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data0, 0)

    rows = []
    print(f"{'robot':5} {'pose':22} {'1/cond':>7} {'elbow':>7} {'shoulder':>9} {'J5':>7} {'TCP err':>8}")
    for r, name, pos, R in cycle_poses(cfg):
        arm = arms[r]
        q, ep, er, ok = solve_grasp_ik(model, data0, f"{r}_tcp", arm.joints, pos, R,
                                       q_init=cfg["robots"][r]["home_q"])
        mt = arm.metrics(q)
        rows.append({"robot": r, "pose": name, "ik_ok": ok, **{k: round(v, 3) for k, v in mt.items()},
                     **{f"q{i+1}_deg": round(np.degrees(v), 1) for i, v in enumerate(q)}})
        print(f"{r:5} {name:22} {mt['inv_cond']:7.3f} {mt['elbow_margin_deg']:6.1f}° {mt['shoulder_mm']:7.0f}mm "
              f"{mt['wrist_J5_deg']:6.1f}° {mt['tcp_err_um']:6.0f}µm")
    out = ROOT / "docs" / "figures"
    with open(out / "xarm6_poses.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nelbow singularity at J3 = {ELBOW_J3:.1f} deg (stretched arm); "
          f"TCP error for {JOINT_ERR_DEG} deg on every joint = worst case")

    # --- dexterity map of robot A, tool down, fingers closing along y, at Pos 1 placing height ---
    arm = arms["A"]
    z = cfg["fixture"]["plate"][2] + GRASP_H
    R = tool_down([0, 1, 0])
    xs, ys = np.linspace(-0.85, 0.30, 70), np.linspace(-0.55, 0.55, 66)
    M = np.full((len(ys), len(xs)), np.nan)
    seed_row = np.array(cfg["robots"]["A"]["home_q"])
    for i, y in enumerate(ys):
        seed = seed_row.copy()
        for j, x in enumerate(xs):
            q, ep, er, ok = solve_grasp_ik(model, data0, "A_tcp", arm.joints, [x, y, z], R, q_init=seed, iters=200)
            if ok:
                M[i, j] = arm.metrics(q)["inv_cond"]
                seed = q
                if j == 0 or np.isnan(M[i, j - 1]):
                    seed_row = q
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "font.size": 10, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(M, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=cmap, vmin=0, vmax=0.25,
                   aspect="equal", interpolation="nearest")
    base = cfg["robots"]["A"]["base_pos"]
    ax.add_patch(Circle(base[:2], 0.70, fill=False, ls=(0, (4, 3)), color=INK2, lw=1))
    ax.text(-0.08, 0.52, "700 mm nominal reach", fontsize=8.5, color=INK2, ha="right")
    ax.plot(*base[:2], marker="o", color=INK, ms=7)
    ax.text(base[0], base[1] - 0.05, "robot A base", ha="center", va="top", fontsize=8.5)
    pl = cfg["fixture"]["plate"]
    ax.add_patch(Rectangle((-pl[0] / 2, -pl[1] / 2), pl[0], pl[1], fill=False, ec=INK, lw=1.4))
    ax.text(0.0, -pl[1] / 2 - 0.02, "fixture", ha="center", va="top", fontsize=8.5)
    for key in ("pos1_a", "pos2_a"):
        x, y, _ = cfg["supply"][key]
        L = cfg["profiles"][key[:4]]["length"]
        ax.add_patch(Rectangle((x - L / 2, y - 0.02), L, 0.04, fill=False, ec=INK, lw=1.2))
        ax.text(x, y + 0.035, f"magazine {key[:4].replace('pos', 'Pos ')}", ha="center", fontsize=8.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("dexterity (inverse condition number)")
    cb.outline.set_visible(False)
    ax.set_title("Robot A, tool vertical at placing height: dexterity drops on a ring ~300 mm\n"
                 "from the base (J5 -> 0); the magazines were moved out to ~450 mm",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_xarm6_dexterity.png", dpi=200)
    reach = np.nanmax(np.hypot(*np.meshgrid(xs - base[0], ys - base[1]))[~np.isnan(M)])
    print(f"max horizontal reach with tool vertical at z = {1e3 * z:.0f} mm: {1e3 * reach:.0f} mm")


if __name__ == "__main__":
    main()
