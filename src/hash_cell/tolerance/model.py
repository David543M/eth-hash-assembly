"""Vectorised Monte-Carlo of the hash assembly characteristics A1-A8.

Frame: x along Pos 1, y along Pos 2, origin at the assembly centre (mm).
Index 0/1: Pos 1 "a" at y = +d/2 (robot A) and "b" at y = -d/2 (robot B);
           Pos 2 "a" at x = -d/2 (robot A) and "b" at x = +d/2 (robot B).
A yaw psi is the small rotation of a part about z; a Pos 2 axis is x(y) = x2 - psi2 * (y - y2).
"""
from __future__ import annotations

import numpy as np

from .params import FormErrors, Nominal, Process, Scenario

CHARACTERISTICS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
DESCRIPTION = {
    "A1": "Centre distance Pos 2-Pos 2",
    "A2": "Centre distance Pos 1-Pos 1",
    "A3": "Pos 1 end to Pos 2 axis (x4)",
    "A4": "Pos 1 edge to Pos 2 end (x4)",
    "A5": "Parallelism Pos 1 / Pos 1",
    "A6": "Perpendicularity Pos 2 / Pos 1",
    "A7": "Parallelism Pos 2 top / plate",
    "A8": "Gap at contacts (x4)",
}
SIGNED = {"A1", "A2", "A3", "A4"}   # deviation from nominal, else a positive zone width / gap


def _draw(rng, tol, shape, dist):
    if dist == "uniform":
        return rng.uniform(-tol, tol, shape)
    return rng.normal(0.0, tol / 3.0, shape)


def simulate(scn: Scenario, n: int, rng: np.random.Generator, nom: Nominal = Nominal(),
             form: FormErrors = FormErrors(), proc: Process = Process()) -> dict[str, np.ndarray]:
    pt, h = scn.parts, nom.d / 2
    N = rng.normal
    L1 = nom.L1 + _draw(rng, pt.L1, (n, 2), pt.dist)
    W1 = nom.W1 + _draw(rng, pt.W1, (n, 2), pt.dist)
    H1 = nom.H1 + _draw(rng, pt.H1, (n, 2), pt.dist)
    if pt.paired_height_sigma is not None:
        H1[:, 1] = H1[:, 0] + N(0, pt.paired_height_sigma, n)
    L2 = nom.L2 + _draw(rng, pt.L2, (n, 2), pt.dist)
    W2 = nom.W2 + _draw(rng, pt.W2, (n, 2), pt.dist)
    y1_nom, x2_nom = np.array([h, -h]), np.array([-h, h])
    f = lambda shape: N(0, proc.fixture_sigma, shape)

    if scn.placement == "robot":
        # robot A places index 0, robot B index 1; B carries a common base-calibration offset
        cal = np.stack([np.zeros((n, 2)), N(0, proc.base_cal_sigma, (n, 2))], axis=1)   # (n, part, xy)
        e1, e2 = N(0, proc.robot_sigma, (n, 2, 2)) + cal, N(0, proc.robot_sigma, (n, 2, 2)) + cal
        x1 = e1[..., 0] + N(0, proc.grasp_sigma, (n, 2))
        y1 = y1_nom + e1[..., 1]
        x2 = x2_nom + e2[..., 0]
        y2 = e2[..., 1] + N(0, proc.grasp_sigma, (n, 2))
        psi1, psi2 = N(0, proc.robot_yaw_sigma, (n, 2)), N(0, proc.robot_yaw_sigma, (n, 2))
    elif scn.placement == "datum":
        # Pos 1 pushed against an end stop at x = -L1/2 (datum C) and an outer side rail (datum B)
        x1 = -nom.L1 / 2 + f((n, 2)) + L1 / 2
        y1 = np.sign(y1_nom) * (h + nom.W1 / 2 + f((n, 2)) - W1 / 2)
        # Pos 2 pushed against an end stop at y = -L2/2 and a side reference face
        y2 = -nom.L2 / 2 + f((n, 2)) + L2 / 2
        x2 = x2_nom - nom.W2 / 2 + f((n, 2)) + W2 / 2
        psi1 = N(0, 2 * proc.fixture_sigma / nom.L1, (n, 2))
        psi2 = N(0, 2 * proc.fixture_sigma / nom.L2, (n, 2))
    elif scn.placement == "centring":
        # symmetric jaws centre each part in length and width: dimensions split equally
        c = lambda shape: N(0, proc.centring_sigma, shape) + f(shape)
        x1, y1 = c((n, 2)), y1_nom + c((n, 2))
        x2, y2 = x2_nom + c((n, 2)), c((n, 2))
        psi1 = N(0, 2 * proc.fixture_sigma / nom.L1, (n, 2))
        psi2 = N(0, 2 * proc.fixture_sigma / nom.L2, (n, 2))
    else:
        raise ValueError(scn.placement)

    bow_s = proc.bow_residual_sigma if scn.clamped else form.bow_sigma
    bow1, bow2 = np.abs(N(0, bow_s, (n, 2))), np.abs(N(0, bow_s, (n, 2)))

    out = {}
    out["A1"] = (x2[:, 1] - x2[:, 0] - nom.d)[:, None]
    out["A2"] = (y1[:, 0] - y1[:, 1] - nom.d)[:, None]
    # A3: for each Pos 1, distance from its ends to the Pos 2 axes, taken at the Pos 1 axis
    x2_at = x2[:, None, :] - psi2[:, None, :] * (y1[:, :, None] - y2[:, None, :])   # (n, pos1, pos2)
    left = x2_at[:, :, 0] - (x1 - L1 / 2) - nom.a3
    right = (x1 + L1 / 2) - x2_at[:, :, 1] - nom.a3
    out["A3"] = np.concatenate([left, right], axis=1)
    # A4: for each Pos 2, distance from the outer edge of each Pos 1 to the Pos 2 end
    edge_b = (y1[:, 1] - W1[:, 1] / 2)[:, None] + psi1[:, 1:2] * (x2 - x1[:, 1:2])
    edge_a = (y1[:, 0] + W1[:, 0] / 2)[:, None] + psi1[:, 0:1] * (x2 - x1[:, 0:1])
    lower = edge_b - (y2 - L2 / 2) - nom.a4
    upper = (y2 + L2 / 2) - edge_a - nom.a4
    out["A4"] = np.concatenate([lower, upper], axis=1)
    if scn.machine_ends:
        # ends cut after joining, referenced on the Pos 2 axes and Pos 1 edges
        out["A3"] = N(0, proc.machining_sigma, out["A3"].shape)
        out["A4"] = N(0, proc.machining_sigma, out["A4"].shape)
    out["A5"] = (np.abs(psi1[:, 0] - psi1[:, 1]) * nom.L1 + bow1[:, 0])[:, None]
    out["A6"] = np.abs(psi2 - psi1[:, 1:2]) * nom.L2 + bow2
    out["A7"] = (np.abs(H1[:, 0] - H1[:, 1]) * nom.L2 / nom.d)[:, None] + np.abs(N(0, form.h2_var_sigma, (n, 2)))
    gap = np.abs(N(0, form.gap_sigma, (n, 4)))
    if scn.clamped:
        gap = np.where(gap <= proc.gap_closable, np.abs(N(0, proc.gap_residual_sigma, (n, 4))),
                       gap - proc.gap_closable)
    out["A8"] = gap
    return out


def conforming(chars: dict[str, np.ndarray], nom: Nominal = Nominal()) -> dict[str, np.ndarray]:
    """Boolean per assembly and characteristic (all instances of it conform), plus 'All'."""
    ok = {}
    for k, v in chars.items():
        lim = nom.tol if k in SIGNED else nom.geo
        ok[k] = np.all(np.abs(v) <= lim + 1e-12, axis=1)
    ok["All"] = np.all(np.stack([ok[k] for k in CHARACTERISTICS]), axis=0)
    return ok


def yields(scn: Scenario, n: int = 100_000, seed: int = 0, **kw) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {k: float(v.mean()) for k, v in conforming(simulate(scn, n, rng, **kw)).items()}


def worst_case_length_allowance(nom: Nominal = Nominal()) -> dict[str, float]:
    """Largest part-length deviation that can still meet A3/A4, with perfect positioning (worst case)."""
    return {
        # datum stop on one end: the far-end A3 takes the whole length deviation
        "L1_datum": nom.tol,
        # self-centring: each end takes half of it
        "L1_centring": 2 * nom.tol,
        # best possible split of the closed chain 100 + 120 + 100 = 320 (A3, A1, A3 all at their limits)
        "L1_chain_limit": 3 * nom.tol,
        # Pos 2: 60 + (20 + 120 + 20) + 60 = 280, centring; Pos 1 width enters by half on each side
        "L2_centring_given_W1": 2 * nom.tol,
    }
