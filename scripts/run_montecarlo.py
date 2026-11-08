"""Monte-Carlo tolerance analysis of the hash assembly.

    python scripts/run_montecarlo.py            # 100 000 assemblies per scenario
    python scripts/run_montecarlo.py --n 20000  # faster

Writes docs/figures/*.png and docs/figures/yield_table.csv.
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from hash_cell.tolerance.model import CHARACTERISTICS, DESCRIPTION, conforming, simulate, worst_case_length_allowance
from hash_cell.tolerance.params import PAIRED_H, Nominal, PartTol, Scenario, scenarios

OUT = ROOT / "docs" / "figures"
# reference palette (light): surface, text, categorical slots 1-3, blue sequential ramp
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1C, S2C, S3C = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def yield_table(scns, n, seed):
    rows = {}
    for s in scns:
        ok = conforming(simulate(s, n, np.random.default_rng(seed)))
        rows[s.key] = {k: float(v.mean()) for k, v in ok.items()}
    return rows


def fig_heatmap(scns, table):
    keys = CHARACTERISTICS + ["All"]
    M = np.array([[table[s.key][k] for s in scns] for k in keys]) * 100
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "100" if v >= 99.995 else ("<0.1" if 0 < v < 0.05 else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if v > 55 else INK, fontweight="bold" if i == len(keys) - 1 else "normal")
    ax.set_xticks(range(len(scns)), [f"{s.key}\n{s.label}" for s in scns], fontsize=8.5)
    ax.set_yticks(range(len(keys)), [f"{k}  {DESCRIPTION[k]}" if k in DESCRIPTION else "All  whole assembly" for k in keys],
                  fontsize=8.5)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axhline(len(keys) - 1.5, color=SURF, lw=3)
    fig.suptitle("Conforming assemblies (%): the robot alone never conforms; the fixture\n"
                 "secures A1, A2, A5, A6, A8, and the part dimensions decide A3, A4, A7",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_yield_heatmap.png", dpi=200)
    plt.close(fig)


def fig_a3_hist(scns, n, seed):
    pick = [(scns[0], S1C), (scns[2], S2C), (scns[3], S3C)]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (s, c) in zip(axes, pick):
        a3 = simulate(s, n, np.random.default_rng(seed))["A3"].ravel()
        lim = max(0.12, np.percentile(np.abs(a3), 99.5))
        ax.axvspan(-0.05, 0.05, color=GRID, zorder=0)
        ax.hist(a3, bins=np.linspace(-lim, lim, 70), color=c, edgecolor=SURF, linewidth=0.6, zorder=2)
        inside = np.mean(np.abs(a3) <= 0.05) * 100
        ax.set_title(f"{s.key} {s.label.replace(chr(10), ' ')}", fontsize=10)
        ax.text(0.02, 0.95, f"{inside:.1f} % of A3 values\nwithin ±0.05 mm", transform=ax.transAxes,
                va="top", fontsize=8.5, color=INK2)
        ax.set_xlabel("A3 deviation (mm)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle("A3 (Pos 1 end to Pos 2 axis): the closed chain 100 + 120 + 100 = 320 passes the length error through",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_a3_hist.png", dpi=200)
    plt.close(fig)


def fig_cut_tolerance(n, seed):
    tols = np.array([0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    base = scenarios()[3]                    # S3: centring, paired heights
    res = {"A3": [], "A4": [], "All": []}
    for t in tols:
        s = replace(base, parts=replace(base.parts, L1=t, L2=t))
        ok = conforming(simulate(s, n, np.random.default_rng(seed)))
        for k in res:
            res[k].append(ok[k].mean() * 100)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.grid(axis="y", color=GRID, lw=0.8)
    for k, c, lab, dy in (("A3", S1C, "A3  end to axis", 0), ("A4", S2C, "A4  edge to end", 7),
                          ("All", S3C, "All characteristics", -5)):
        ax.plot(tols, res[k], color=c, lw=2, marker="o", ms=4, zorder=3)
        ax.annotate(lab, (tols[-1], res[k][-1]), xytext=(8, dy), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.49, 103, "ISO 2768-m (drawing)", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 0.68)
    ax.set_ylim(-6, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Part length tolerance, ± mm (Pos 1 and Pos 2)")
    ax.set_ylabel("Conforming assemblies (%)")
    ax.set_title("A3 needs part lengths within about ±0.1 mm;\nA4 stays limited by the Pos 1 width (±0.3 mm)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_yield_vs_cut_tolerance.png", dpi=200)
    plt.close(fig)
    return tols, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    scns = scenarios()
    table = yield_table(scns, args.n, args.seed)

    with open(OUT / "yield_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["characteristic"] + [s.key for s in scns])
        for k in CHARACTERISTICS + ["All"]:
            w.writerow([k] + [f"{table[s.key][k]:.4f}" for s in scns])

    print(f"Conforming assemblies, n = {args.n} per scenario\n")
    print(f"{'':6}" + "".join(f"{s.key:>8}" for s in scns))
    for k in CHARACTERISTICS + ["All"]:
        print(f"{k:6}" + "".join(f"{100 * table[s.key][k]:8.1f}" for s in scns))
    print("\nWorst-case length allowance for A3/A4 (mm):", worst_case_length_allowance())

    fig_heatmap(scns, table)
    fig_a3_hist(scns, min(args.n, 50_000), args.seed)
    tols, res = fig_cut_tolerance(min(args.n, 50_000), args.seed)
    print("\nYield vs length tolerance (S3):")
    for t, a, b, c in zip(tols, res["A3"], res["A4"], res["All"]):
        print(f"  ±{t:.3f} mm  A3 {a:5.1f} %  A4 {b:5.1f} %  All {c:5.1f} %")
    print(f"\nFigures and table written to {OUT}")


if __name__ == "__main__":
    main()
