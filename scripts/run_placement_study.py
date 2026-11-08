"""Free placement study: repeat the full cycle with and without robot inaccuracy and record
where the parts end up (no active fixture: parts are released by the robot only).

    python scripts/run_placement_study.py --runs 30

Output: docs/figures/fig_free_placement.png, docs/figures/free_placement.csv
"""
import argparse
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

from hash_cell.cycle import ZoneLock, build_programs, placement_errors, run
from hash_cell.scene import load_config, write_scene

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {0.0: "#2a78d6", 1.0: "#eb6834"}


def one_cycle(model, cfg, noise_mm, seed):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    rng = np.random.default_rng(seed)
    noise = (lambda: rng.normal(0, noise_mm / 3e3, 3)) if noise_mm > 0 else None
    progs, _ = build_programs(model, data, cfg, [], ZoneLock(), tcp_noise=noise)
    ok = run(model, data, progs)
    return ok, placement_errors(model, data, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(write_scene()))
    rows = []
    for noise in (0.0, 1.0):
        # without noise every run is identical (deterministic simulation): one run is enough
        for seed in range(1 if noise == 0 else args.runs):
            ok, errs = one_cycle(model, cfg, noise, seed)
            for part, e in errs.items():
                rows.append((noise, seed, part, ok, e["dx"], e["dy"], e["dz"], e["dyaw"]))
    out = ROOT / "docs" / "figures"
    with open(out / "free_placement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_mm", "seed", "part", "cycle_ok", "dx_mm", "dy_mm", "dz_mm", "dyaw_deg"])
        w.writerows(rows)
    a = np.array([(r[0], r[4], r[5]) for r in rows])
    for noise in (0.0, 1.0):
        sel = a[a[:, 0] == noise]
        r = np.hypot(sel[:, 1], sel[:, 2])
        inside = np.mean((np.abs(sel[:, 1]) <= 0.05) & (np.abs(sel[:, 2]) <= 0.05)) * 100
        print(f"robot error ±{noise:.1f} mm: {len(sel)} parts, planar error mean {r.mean():.2f} mm, "
              f"max {r.max():.2f} mm, within ±0.05 mm: {inside:.0f} %")
    print("cycles completed:", sum(r[3] for r in rows) // 4, "of", len(rows) // 4)

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "font.size": 10, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.add_patch(plt.Rectangle((-0.05, -0.05), 0.1, 0.1, facecolor="none", edgecolor=INK, lw=1.2, zorder=3))
    ax.annotate("±0.05 mm\ntolerance box", (0.05, 0.05), xytext=(0.6, 1.6), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    for noise, label in ((1.0, "robot error ±1 mm (3σ), 30 cycles"), (0.0, "ideal robot (release and contact only)")):
        sel = a[a[:, 0] == noise]
        ax.scatter(sel[:, 1], sel[:, 2], s=26 if noise else 46, color=COLORS[noise], edgecolor=SURF,
                   linewidth=0.8, label=label, zorder=4 if noise == 0 else 2)
    lim = max(2.5, np.abs(a[:, 1:]).max() * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x error of the placed part (mm)")
    ax.set_ylabel("y error of the placed part (mm)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=9)
    ax.set_title("Released by the robot, parts land about 1 mm from nominal,\n"
                 "bounded by the rail clearance: 20 times the ±0.05 mm box",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig_free_placement.png", dpi=200)


if __name__ == "__main__":
    main()
