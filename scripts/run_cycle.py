"""Run one full assembly cycle with both robots.

    python scripts/run_cycle.py              # headless: timeline figure + placement errors
    python scripts/run_cycle.py --viewer     # watch it in the MuJoCo viewer (mjpython on macOS)
    python scripts/run_cycle.py --video      # also render docs/cycle.mp4 (needs ffmpeg on PATH)
    python scripts/run_cycle.py --noise 1.0  # robot inaccuracy, +/-1 mm (3 sigma) on every TCP target

Outputs: docs/figures/fig_cycle_timeline.png, docs/figures/cycle_placement.csv
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from hash_cell.cycle import ZoneLock, build_programs, placement_errors, run
from hash_cell.scene import load_config, write_scene

FIG = ROOT / "docs" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MOTION, GRIPPER, ZONE, WAIT = "#2a78d6", "#1baf7a", "#eb6834", "#c9c8c3"


def timeline_figure(log, lock, t_end, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "font.size": 10, "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(10, 2.9))
    rows = {"A": 1.0, "B": 0.0}
    for e in log:
        if not np.isfinite(e.t1) or e.t1 <= e.t0:
            continue
        y = rows[e.robot]
        if e.action.startswith("wait"):
            c = WAIT
        elif e.action.startswith(("grip", "release")):
            c = GRIPPER
        else:
            c = MOTION
        ax.barh(y + 0.08, e.t1 - e.t0, left=e.t0, height=0.42, color=c, edgecolor=SURF, linewidth=1.2)
    enter = {}
    for t, who, what in lock.history:
        if what == "enter":
            enter[who] = t
        else:
            t_in = enter.pop(who)
            ax.barh(rows[who] - 0.24, t - t_in, left=t_in, height=0.1, color=ZONE)
    for e in log:   # label the part transfers
        if e.action.startswith("place"):
            ax.text(e.t0, rows[e.robot] + 0.36, e.action.split()[1].replace("pos", "Pos ").replace("_", " "),
                    fontsize=8.5, color=INK2, ha="center")
    ax.set_yticks([1.0, 0.0], ["Robot A", "Robot B"])
    ax.set_xlim(0, t_end)
    ax.set_ylim(-0.45, 1.55)
    ax.set_xlabel("Simulated time (s)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=MOTION, label="robot motion"), Patch(color=GRIPPER, label="gripper"),
                       Patch(color=WAIT, label="waiting"), Patch(color=ZONE, label="inside fixture zone")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=9)
    ax.set_title(f"Full cycle in {t_end:.1f} s: the robots take turns in the fixture zone, which sets the pace",
                 loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="robot TCP error, +/- mm at 3 sigma")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    model = mujoco.MjModel.from_xml_path(str(write_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    rng = np.random.default_rng(args.seed)
    noise = (lambda: rng.normal(0, args.noise / 3e3, 3)) if args.noise > 0 else None
    log, lock = [], ZoneLock()
    programs, _ = build_programs(model, data, cfg, log, lock, tcp_noise=noise)

    on_step, finish = None, None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
        viewer.opt.geomgroup[2] = 1

        def on_step():
            viewer.sync()
            time.sleep(model.opt.timestep)
        finish = viewer.close
    elif args.video:
        frames_dir = ROOT / "outputs" / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for f in frames_dir.glob("*.png"):
            f.unlink()
        renderer = mujoco.Renderer(model, 720, 1280)
        opt = mujoco.MjvOption()
        opt.geomgroup[2] = 1
        every, k = int(round(1 / 30 / model.opt.timestep)), [0]
        from PIL import Image

        def on_step():
            if k[0] % every == 0:
                renderer.update_scene(data, camera="overview", scene_option=opt)
                Image.fromarray(renderer.render()).save(frames_dir / f"{k[0] // every:05d}.png")
            k[0] += 1

        def finish():
            out = ROOT / "docs" / "cycle.mp4"
            if shutil.which("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                                str(frames_dir / "%05d.png"), "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", "23", "-pix_fmt", "yuv420p", str(out)], check=True)
                print("video:", out)
            else:
                print("ffmpeg not found; frames are in", frames_dir)

    ok = run(model, data, programs, on_step=on_step)
    if finish:
        finish()
    t_end = max(e.t1 for e in log if np.isfinite(e.t1))
    print(f"cycle {'complete' if ok else 'INCOMPLETE'} in {t_end:.1f} s (simulated)")
    print("zone:", ", ".join(f"{who} {what} {t:.1f}s" for t, who, what in lock.history))

    errs = placement_errors(model, data, cfg)
    FIG.mkdir(parents=True, exist_ok=True)
    with open(FIG / "cycle_placement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part", "dx_mm", "dy_mm", "dz_mm", "dyaw_deg"])
        for part, e in errs.items():
            w.writerow([part] + [f"{e[k]:.3f}" for k in ("dx", "dy", "dz", "dyaw")])
            print(f"{part}: dx {e['dx']:+.2f} mm  dy {e['dy']:+.2f} mm  dz {e['dz']:+.2f} mm  dyaw {e['dyaw']:+.3f} deg")
    timeline_figure(log, lock, t_end, FIG / "fig_cycle_timeline.png")


if __name__ == "__main__":
    main()
