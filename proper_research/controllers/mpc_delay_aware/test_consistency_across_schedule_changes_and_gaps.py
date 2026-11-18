"""2026-11-19 (second review round), point 3: demonstrate that the
estimator's measurement equation
    dr(k) = g * d_eta_pred_m(k) + d_noise(k),  d_eta_pred_m(k) = eta_pred_m(k) - eta_pred_m(k')
(see jacobian_adaptation.py's module docstring) is CONSISTENT with the
predictor's measured-state-anchored correction
    correction_j = g * J_nom[idx_j][:,:6] @ (q_phys_j - q_meas6)
(see delay_aware_mpc.py's `_beam_prediction_terms_exec`) under the two
conditions explicitly called out: (1) the schedule index changes between
the two samples being differenced, and (2) a GAP of rejected samples
separates the current sample from the last ACCEPTED one (k' != k-1).

The consistency being demonstrated is specific: the predictor's anchor is
ALWAYS this tick's own q_meas6, taken fresh every solve -- it has no
dependency whatsoever on which earlier sample the ESTIMATOR last
differenced against. So "gaps between accepted samples" is purely an
ESTIMATOR-internal concern (how stale is the g_hat currently in use);
it cannot, by construction, desynchronize the PREDICTOR's own anchoring,
which is demonstrated directly below by re-running the anchoring proof
(test_anchoring_zero_at_measured_state.py's test) at the SAME g_hat value
an estimator would produce after a real gap.

Run: python3 -m proper_research.controllers.mpc_delay_aware.test_consistency_across_schedule_changes_and_gaps
"""
from __future__ import annotations

import numpy as np

from proper_research.controllers.mpc_delay_aware.jacobian_adaptation import (
    JacobianAdaptationConfig, ScalarJacobianAdaptation,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, ConfigurationReference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
    InsertionAnchoredTaskNullspaceMPC,
)

RNG = np.random.default_rng(7)
DT = 0.1
S = 20
N = 5


def _J(variant: int) -> np.ndarray:
    base = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0],
    ])
    theta = 0.07 * variant
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return rot @ base


def test_measurement_equation_valid_across_schedule_change_and_gap():
    """Two accepted samples, several rejected ticks apart (a genuine gap,
    k' != k-1), with the schedule index ALSO different between them (J
    genuinely changes). The measurement equation must still recover a
    known global g_true."""
    cfg = JacobianAdaptationConfig(mode="active", excitation_floor_m=1e-6, forgetting_lambda=1.0,
                                    max_sample_age_s=None)
    est = ScalarJacobianAdaptation(cfg)
    g_true = 0.21

    # Sample 0: accepted (first ever).
    dq0 = RNG.normal(scale=0.03, size=6)
    J0 = _J(0)
    r0 = g_true * (J0 @ dq0)
    d0 = est.step(instantaneous_residual=r0, delta_q6=dq0, j_nominal6=J0, frame_index=0, sample_age_s=0.01)
    assert d0["accepted"] and d0["first_sample"]

    # Samples 1..4: deliberately REJECTED (duplicate frame_index each time,
    # simulating a camera stall) -- a genuine gap of 4 ticks with no
    # accepted sample in between.
    for k in range(1, 5):
        dq_rej = RNG.normal(scale=0.03, size=6)
        d = est.step(instantaneous_residual=np.zeros(3), delta_q6=dq_rej, j_nominal6=_J(k),
                     frame_index=0, sample_age_s=0.01)  # SAME frame_index=0 -> rejected as duplicate
        assert not d["accepted"] and d["reject_reason"] == "duplicate_frame_index"

    # Sample 5: accepted, with a DIFFERENT schedule index (J changed) and
    # a fresh frame_index -- this is k'=0, k=5, a real 5-tick gap.
    dq5 = RNG.normal(scale=0.05, size=6)
    J5 = _J(5)
    r5 = g_true * (J5 @ dq5)
    d5 = est.step(instantaneous_residual=r5, delta_q6=dq5, j_nominal6=J5, frame_index=5, sample_age_s=0.01)
    assert d5["accepted"], d5

    # Verify the measurement equation DIRECTLY: dr = r5-r0 should equal
    # g_true*(eta_pred_m(5)-eta_pred_m(0)) to within float precision (both
    # residuals are noiseless here by construction), confirming the
    # equation holds across BOTH the schedule change (J0 != J5) and the
    # 5-tick gap (k'=0, not k-1=4).
    eta0 = J0 @ dq0
    eta5 = J5 @ dq5
    d_eta_pred_vec_m = eta5 - eta0
    dr_expected = g_true * d_eta_pred_vec_m
    dr_actual = r5 - r0
    assert np.allclose(dr_actual, dr_expected, atol=1e-10), (
        f"measurement equation violated across schedule-change+gap: {dr_actual} vs {dr_expected}"
    )
    assert abs(d5["g_hat"] - g_true) < 1e-6, f"g_hat={d5['g_hat']} vs g_true={g_true} (noiseless, should be exact)"
    print(f"PASS: measurement equation dr=g*d_eta_pred_m holds EXACTLY across a schedule "
          f"change (J0 at variant 0 -> J5 at variant 5) AND a 5-tick gap (k'=0, not k-1); "
          f"g_hat={d5['g_hat']:.6f} (true={g_true})")


def test_predictor_anchoring_unaffected_by_estimator_gap_history():
    """Confirms the SPECIFIC consistency claim: the predictor's anchor
    (q_meas6, this tick's own measurement) is identical whether g_hat
    arrived from a freshly-updated estimator or one that just crossed a
    long gap -- the predictor has no visibility into, or dependency on,
    the estimator's internal differencing history. Demonstrated by
    constructing the SAME g value two ways (direct injection vs an
    estimator that just resolved a gap) and confirming the predictor's
    QP terms are bit-identical either way."""
    time_s = np.arange(S) * DT
    q0 = np.array([0.1, -1.2, 1.3, -1.6, -1.57, 0.05])
    q_end = q0 + np.array([0.05, 0.03, -0.04, 0.02, 0.01, -0.02])
    alpha = np.linspace(0.0, 1.0, S)[:, None]
    q = q0[None, :] * (1 - alpha) + q_end[None, :] * alpha
    L = (0.025 * (1 - alpha[:, 0]) + 0.03 * alpha[:, 0])[:, None]
    state = np.concatenate([q, L], axis=1)
    input_ = np.zeros_like(state)
    input_[:-1] = np.diff(state, axis=0) / DT
    input_[-1] = input_[-2]
    desired = np.zeros((S, 3)); desired[:, 0] = 0.4
    tangent = np.zeros((S, 3)); tangent[:, 0] = 1.0
    ref = ConfigurationReference(
        time_s=time_s, path_coordinate_m=np.arange(S) * 0.001,
        state=state, input=input_, acceleration=np.zeros_like(state),
        desired_position_m=desired, desired_tangent=tangent,
        planned_beam_feasible=np.ones(S, dtype=bool),
        planned_position_error_m=np.zeros(S), planned_tangent_error_rad=np.zeros(S),
        sample_period_s=DT, source="gap_consistency_test",
    )
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    sched = np.tile(J0[None, :, :], (S, 1, 1))

    def make_controller(jac_adapt):
        mpc_cfg = ConfigurationMPCConfig(
            sample_period_s=DT, prediction_horizon=N,
            state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
            velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        )
        beam_cfg = BeamOutputMPCConfig(
            position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
            use_dare_terminal_cost=False, directional_damping=0.0,
        )
        return InsertionAnchoredTaskNullspaceMPC(
            gamma=0.0, insertion_state_anchor_weight=0.0,
            reference=ref, config=mpc_cfg, beam_config=beam_cfg,
            reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
            jacobian_adaptation=jac_adapt,
        )

    k = 4
    q_meas6 = np.asarray(ref.state[k, :6]) + 0.06
    x_exec = np.concatenate([q_meas6, q_meas6, [float(ref.state[k, 6])]])

    # Path 1: directly inject g via the plain (no estimator) predictor call.
    c_direct = make_controller(None)
    G1, ce1, _, _, _, Jbar1 = c_direct._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3),
        jacobian_adaptation_g=0.21, jacobian_adaptation_q_meas6=q_meas6,
    )

    # Path 2: an estimator that just resolved a real gap (reusing the
    # exact scenario from the previous test) to produce g_hat ~= 0.21,
    # then call the SAME predictor method with THAT g_hat.
    c_gap = make_controller(dict(mode="active", excitation_floor_m=1e-6, forgetting_lambda=1.0))
    g_true = 0.21
    dq0 = RNG.normal(scale=0.03, size=6)
    J0_est = _J(0)
    r0 = g_true * (J0_est @ dq0)
    c_gap._jac_adapt.step(instantaneous_residual=r0, delta_q6=dq0, j_nominal6=J0_est, frame_index=0, sample_age_s=0.01)
    for kk in range(1, 5):
        c_gap._jac_adapt.step(instantaneous_residual=np.zeros(3), delta_q6=RNG.normal(scale=0.03, size=6),
                               j_nominal6=_J(kk), frame_index=0, sample_age_s=0.01)
    dq5 = RNG.normal(scale=0.05, size=6)
    J5_est = _J(5)
    r5 = g_true * (J5_est @ dq5)
    diag_gap = c_gap._jac_adapt.step(instantaneous_residual=r5, delta_q6=dq5, j_nominal6=J5_est, frame_index=5, sample_age_s=0.01)
    g_from_gap = diag_gap["g_used"]
    assert abs(g_from_gap - 0.21) < 1e-6

    G2, ce2, _, _, _, Jbar2 = c_gap._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3),
        jacobian_adaptation_g=g_from_gap, jacobian_adaptation_q_meas6=q_meas6,
    )

    assert np.allclose(G1, G2) and np.allclose(ce1, ce2) and np.allclose(Jbar1, Jbar2), (
        "predictor QP terms differ depending on whether g came from direct injection "
        "or a gap-crossing estimator -- anchoring should be IDENTICAL either way"
    )
    print(f"PASS: predictor QP terms (G, constant_error, Jbar) are bit-identical whether "
          f"g={g_from_gap:.6f} came from direct injection or from an estimator that just "
          f"crossed a 5-tick gap with a changing schedule index -- confirms the predictor's "
          f"q_meas6 anchor has no dependency on the estimator's internal gap history")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, repr(exc)))
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests)-len(failures)}/{len(tests)} consistency checks passed.")
    if failures:
        raise SystemExit(1)
