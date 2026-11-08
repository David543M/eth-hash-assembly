"""The full two-robot cycle completes, respects the zone lock and puts every part near its nominal pose."""
import mujoco
import numpy as np

from hash_cell.cycle import ZoneLock, build_programs, placement_errors, run
from hash_cell.scene import MODELS, build_scene, load_config


def test_full_cycle():
    cfg = load_config()
    path = MODELS / "scene.xml"
    path.write_text(build_scene(cfg), encoding="utf-8")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    log, lock = [], ZoneLock()
    progs, placed = build_programs(m, d, cfg, log, lock)
    assert run(m, d, progs)
    assert sorted(placed) == ["pos1_a", "pos1_b", "pos2_a", "pos2_b"]
    # zone lock: strictly alternating enter / leave, never two robots inside
    inside = None
    for _, who, what in lock.history:
        if what == "enter":
            assert inside is None
            inside = who
        else:
            assert inside == who
            inside = None
    # both Pos 1 are released before any Pos 2 is placed
    t_rel = {e.action.split()[1]: e.t1 for e in log if e.action.startswith("release")}
    t_place = {e.action.split()[1]: e.t0 for e in log if e.action.startswith("place")}
    assert max(t_rel["pos1_a"], t_rel["pos1_b"]) < min(t_place["pos2_a"], t_place["pos2_b"])
    # every part seated within a few millimetres of nominal (free placement, no active fixture)
    for e in placement_errors(m, d, cfg).values():
        assert np.hypot(e["dx"], e["dy"]) < 3.0 and abs(e["dz"]) < 0.5
