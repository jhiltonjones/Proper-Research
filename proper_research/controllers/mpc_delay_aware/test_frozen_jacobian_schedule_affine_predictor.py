"""2026-11-19: offline verification for Condition B (FROZEN) -- tiling the
real schedule's row 0 across all samples and checking the resulting affine
predictor's documented property: constant_error's Jbar-dependent correction
term is IDENTICALLY ZERO at zero state deviation (so the nominal anchor at
each reference point is preserved) regardless of whether the Jacobian is
frozen or scheduled, but DIFFERS under nonzero deviation (so freezing is
NOT merely a slope change -- see the audit correction in this session: both
the slope (G) and this offset term are affected away from zero deviation).

Formula reproduced faithfully from delay_aware_mpc.py's
_beam_prediction_terms_exec (the j_nominal6/jacobians/Jbar/constant_error
block, g=0 i.e. jacobian_adaptation OFF, matching every FROZEN/BASE
condition in this protocol):

    Jbar = block_diag(jacobians[indices])
    constant_error = (nominal_position - desired_position)
                      + Jbar @ (free_state - state_reference_vector)
                      + beta_d * tile(estimated_residual, N)

This test does not instantiate the full MPC class (too much unrelated
construction machinery for what this property needs); it reproduces the
documented formula directly against REAL schedule data, which is the
property under test, not a simulation of unrelated behaviour.
"""
import numpy as np
import scipy.sparse as sp

SCHED_PATH = (
    "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_"
    "L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
)
FROZEN_PATH = (
    "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_"
    "L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor_FROZEN_J0.npy"
)


def _constant_error(jacobians, nominal_position, desired_position, free_state,
                     state_reference_vector, beta_d, residual, N):
    Jbar = sp.block_diag(list(jacobians), format="csc").toarray()
    return (
        nominal_position.reshape(3 * N) - desired_position.reshape(3 * N)
        + Jbar @ (free_state - state_reference_vector)
        + beta_d * np.tile(residual, N)
    ), Jbar


def test_files_exist_and_are_real_schedule_data():
    sched = np.load(SCHED_PATH)
    frozen = np.load(FROZEN_PATH)
    assert sched.shape == frozen.shape
    assert np.allclose(frozen[0], sched[0])
    assert not np.allclose(frozen[100], sched[100])  # frozen genuinely differs away from index 0
    assert np.allclose(frozen[100], frozen[0])  # ...because every row of frozen equals row 0


def test_zero_deviation_constant_error_identical_regardless_of_freezing():
    sched = np.load(SCHED_PATH)
    frozen = np.load(FROZEN_PATH)
    N = 5
    rng = np.random.default_rng(0)
    indices = np.arange(50, 50 + N)
    nominal_position = rng.normal(size=(N, 3))
    desired_position = rng.normal(size=(N, 3))
    residual = rng.normal(size=3) * 1e-4
    beta_d = 1.0
    # ZERO deviation: free_state == state_reference_vector
    state_reference_vector = rng.normal(size=7 * N)
    free_state = state_reference_vector.copy()

    ce_sched, _ = _constant_error(
        sched[indices], nominal_position, desired_position, free_state,
        state_reference_vector, beta_d, residual, N,
    )
    ce_frozen, G_frozen = _constant_error(
        frozen[indices], nominal_position, desired_position, free_state,
        state_reference_vector, beta_d, residual, N,
    )
    ce_sched_G, G_sched = _constant_error(
        sched[indices], nominal_position, desired_position, free_state,
        state_reference_vector, beta_d, residual, N,
    )
    assert np.allclose(ce_sched, ce_frozen), (
        "constant_error must be identical between scheduled and frozen Jacobians "
        "at zero state deviation (anchor preserved)"
    )
    # but the slope (G = Jbar, here identity test via direct jacobians) DOES differ
    assert not np.allclose(G_frozen, G_sched), (
        "the Jacobian (slope) must still differ between frozen and scheduled "
        "away from index 0 -- FROZEN is not identical to BASE, only its "
        "zero-deviation anchor is preserved"
    )


def test_nonzero_deviation_constant_error_differs_between_frozen_and_scheduled():
    sched = np.load(SCHED_PATH)
    frozen = np.load(FROZEN_PATH)
    N = 5
    rng = np.random.default_rng(1)
    indices = np.arange(80, 80 + N)
    nominal_position = rng.normal(size=(N, 3))
    desired_position = rng.normal(size=(N, 3))
    residual = rng.normal(size=3) * 1e-4
    beta_d = 1.0
    state_reference_vector = rng.normal(size=7 * N)
    # NONZERO deviation -- realistic magnitude (a few mrad / mm scale)
    free_state = state_reference_vector + rng.normal(size=7 * N) * 1e-2

    ce_sched, _ = _constant_error(
        sched[indices], nominal_position, desired_position, free_state,
        state_reference_vector, beta_d, residual, N,
    )
    ce_frozen, _ = _constant_error(
        frozen[indices], nominal_position, desired_position, free_state,
        state_reference_vector, beta_d, residual, N,
    )
    assert not np.allclose(ce_sched, ce_frozen), (
        "under nonzero deviation (the normal operating condition under delay "
        "and tracking error) constant_error MUST differ between frozen and "
        "scheduled -- this is the corrected finding: freezing affects both "
        "the slope AND this offset term away from zero deviation, not slope-"
        "only as an earlier, now-corrected claim implied"
    )


def test_estimate_output_residual_site_reads_the_same_shared_schedule_array():
    # Confirms the OTHER consumer (base class's _estimate_output_residual,
    # simulate_time_parameterized_beam_output_mpc.py line ~891) reads the
    # SAME self.reference_position_jacobians attribute as the QP cost
    # builder -- so feeding a frozen array at construction automatically
    # propagates to BOTH sites; no separate wiring is needed for the
    # residual estimator. Verified by source inspection, asserted here as
    # a living regression check against the two call sites' line content.
    import inspect
    from proper_research.controllers.mpc_delay_aware import delay_aware_mpc
    from proper_research.simulation.simulations import simulate_time_parameterized_beam_output_mpc as base

    qp_src = inspect.getsource(delay_aware_mpc)
    assert "self.reference_position_jacobians[indices]" in qp_src
    base_src = inspect.getsource(base)
    assert "self.reference_position_jacobians[index]" in base_src
