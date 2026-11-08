"""Tolerance model: perfect inputs must conform, and the closed chains must behave as derived by hand."""
from dataclasses import replace

import numpy as np

from hash_cell.tolerance.model import conforming, simulate, yields
from hash_cell.tolerance.params import FormErrors, PartTol, Process, scenarios

ZERO_PARTS = PartTol(L1=0, W1=0, H1=0, L2=0, W2=0, H2=0)
ZERO_FORM = FormErrors(bow_sigma=0, h2_var_sigma=0, gap_sigma=0)
ZERO_PROC = Process(robot_sigma=0, robot_yaw_sigma=0, grasp_sigma=0, base_cal_sigma=0, fixture_sigma=0,
                    centring_sigma=0, bow_residual_sigma=0, gap_residual_sigma=0, machining_sigma=0)


def test_perfect_inputs_conform_in_every_scenario():
    for s in scenarios():
        s = replace(s, parts=ZERO_PARTS)
        chars = simulate(s, 10, np.random.default_rng(1), form=ZERO_FORM, proc=ZERO_PROC)
        for v in chars.values():
            assert np.allclose(v, 0)


def test_centring_splits_length_error_on_both_ends():
    s = replace(scenarios()[2], parts=replace(ZERO_PARTS, L1=0.3, dist="uniform"))
    c = simulate(s, 2000, np.random.default_rng(2), form=ZERO_FORM, proc=ZERO_PROC)
    # A3 left and right of the same Pos 1 are equal: each end takes half of the length error
    assert np.allclose(c["A3"][:, 0], c["A3"][:, 2]) and np.abs(c["A3"]).max() <= 0.15 + 1e-9


def test_datum_stop_puts_all_length_error_on_the_far_end():
    s = replace(scenarios()[1], parts=replace(ZERO_PARTS, L1=0.3, dist="uniform"))
    c = simulate(s, 2000, np.random.default_rng(3), form=ZERO_FORM, proc=ZERO_PROC)
    assert np.allclose(c["A3"][:, :2], 0) and np.abs(c["A3"][:, 2:]).max() > 0.25


def test_robot_only_never_conforms_and_machining_fixes_a3_a4():
    y0, y4 = yields(scenarios()[0], 20_000), yields(scenarios()[4], 20_000)
    assert y0["All"] < 0.01
    assert y4["A3"] > 0.999 and y4["A4"] > 0.999
