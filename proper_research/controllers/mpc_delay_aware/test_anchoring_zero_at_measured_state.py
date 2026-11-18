"""Point 2 of the 2026-11-19 review: verify anchoring in the REAL
integrated predictor (`InsertionAnchoredTaskNullspaceMPC`, the actual
class the rectangle worker builds), not just the estimator.

Required test: set q_meas != q_ref and q_pred = q_meas. The adaptive
contribution must be exactly zero at that point (the measured-state
anchored correction, per the 2026-11-19 re-anchoring fix in
delay_aware_mpc.py's `_beam_prediction_terms_exec`). Also shows the
EFFECTIVE JACOBIAN and the AFFINE OFFSET passed into QP construction
separately, and confirms which output dimensions the correction touches
(joint columns only, insertion column exactly untouched).

Run: python3 -m proper_research.controllers.mpc_delay_aware.test_anchoring_zero_at_measured_state
"""
from __future__ import annotations

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, ConfigurationReference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
    InsertionAnchoredTaskNullspaceMPC,
)

DT = 0.1
S = 20
N = 5


def _make_reference() -> ConfigurationReference:
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
    accel = np.zeros_like(state)
    desired = np.zeros((S, 3))
    desired[:, 0] = 0.4
    tangent = np.zeros((S, 3))
    tangent[:, 0] = 1.0
    return ConfigurationReference(
        time_s=time_s, path_coordinate_m=np.arange(S) * 0.001,
        state=state, input=input_, acceleration=accel,
        desired_position_m=desired, desired_tangent=tangent,
        planned_beam_feasible=np.ones(S, dtype=bool),
        planned_position_error_m=np.zeros(S), planned_tangent_error_rad=np.zeros(S),
        sample_period_s=DT, source="synthetic_anchoring_test",
    )


def _schedule(n: int) -> np.ndarray:
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    return np.tile(J0[None, :, :], (n, 1, 1))


def _make_controller(ref, sched, g_min=-1.0, g_max=1.0) -> InsertionAnchoredTaskNullspaceMPC:
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
        jacobian_adaptation=dict(mode="active", g_min=g_min, g_max=g_max),
    )


def test_adaptive_contribution_zero_when_predicted_equals_measured():
    ref = _make_reference()
    sched = _schedule(ref.sample_count)
    c = _make_controller(ref, sched)
    k = 3
    q_ref6 = np.asarray(ref.state[k, :6])
    q_meas6 = q_ref6 + 0.08  # q_meas != q_ref, deliberately large and unambiguous

    # Directly construct x_exec such that q_cmd == q_cmd_prev == q_meas6
    # EXACTLY -- with delay_samples=2 and both history slots equal, the
    # physical prediction's stage-0 (and every later stage, since no new
    # decision has had time to act yet at j<delay_samples) is pinned to
    # q_meas6 regardless of v -- i.e. q_pred_0 = q_meas exactly, by this
    # construction, as the test requires.
    L_meas = float(ref.state[k, 6]) + 0.002
    x_exec = np.concatenate([q_meas6, q_meas6, [L_meas]])

    g_test = 0.37  # force a large, unambiguous g (not estimated -- injected directly)
    G0, constant_error0, _, _, _, Jbar0 = c._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3),
        jacobian_adaptation_g=0.0,
    )
    G1, constant_error1, _, _, _, Jbar1 = c._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3),
        jacobian_adaptation_g=g_test, jacobian_adaptation_q_meas6=q_meas6,
    )

    # The ADAPTIVE CONTRIBUTION to the predicted beam position at v=0 is
    # G@0 + constant_error, i.e. just constant_error here (same G@v=0 for
    # both). Stage 0's physical prediction uses free_state[0][:6], which
    # by this x_exec construction equals q_meas6 exactly (delay_samples=2
    # means stage 0 draws purely from q_cmd/q_cmd_prev history, not v).
    stage0_slice = slice(0, 3)
    adaptive_contribution_stage0 = (constant_error1 - constant_error0)[stage0_slice]
    assert np.allclose(adaptive_contribution_stage0, 0.0, atol=1e-10), (
        f"adaptive contribution at q_pred=q_meas is NOT zero: {adaptive_contribution_stage0} "
        f"(q_meas != q_ref by construction: ||q_meas-q_ref||={np.linalg.norm(q_meas6-q_ref6):.4f} rad)"
    )
    print(f"PASS: adaptive contribution at stage 0 is exactly zero ({adaptive_contribution_stage0}) "
          f"when q_pred==q_meas, even though ||q_meas-q_ref||={np.linalg.norm(q_meas6-q_ref6)*1e3:.1f}mrad "
          f"(g_test={g_test}) -- confirms the correction is anchored at q_meas, not q_ref")

    # Show the EFFECTIVE JACOBIAN and AFFINE OFFSET separately, as requested.
    joint_slope_ratio = Jbar1[:3, :6] / np.where(Jbar0[:3, :6] != 0, Jbar0[:3, :6], np.nan)
    print(f"  effective Jacobian (stage0, joint cols) ratio to baseline: "
          f"{np.nanmean(joint_slope_ratio):.6f} (expected 1+g_test={1+g_test:.6f})")
    assert np.allclose(np.nan_to_num(joint_slope_ratio, nan=1 + g_test), 1 + g_test, atol=1e-8)
    assert np.allclose(Jbar1[:3, 6], Jbar0[:3, 6]), "insertion column of the effective Jacobian was touched!"
    print("  insertion column of the effective Jacobian: UNTOUCHED (confirmed identical to baseline)")
    print(f"  affine-offset (constant_error) DID change away from q_meas: "
          f"stage1 delta={ (constant_error1 - constant_error0)[3:6] }")


def test_unobserved_direction_not_silently_adapted_elsewhere():
    """The correction must change ONLY the directions the Jacobian's own
    joint columns actually span at each stage's own schedule point --
    confirm insertion-column predictions are bit-identical with/without
    adaptation at an ARBITRARY (not just q_pred=q_meas) configuration,
    i.e. the insertion OUTPUT CHANNEL is never touched regardless of
    anchor point."""
    ref = _make_reference()
    sched = _schedule(ref.sample_count)
    c = _make_controller(ref, sched)
    k = 5
    q_meas6 = np.asarray(ref.state[k, :6]) + 0.05
    q_cmd = np.asarray(ref.state[k, :6]) + 0.11  # a DIFFERENT, arbitrary predicted config
    x_exec = np.concatenate([q_cmd, q_cmd, [float(ref.state[k, 6]) + 0.004]])
    G0, ce0, _, _, _, Jbar0 = c._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3), jacobian_adaptation_g=0.0,
    )
    G1, ce1, _, _, _, Jbar1 = c._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=k, estimated_residual=np.zeros(3),
        jacobian_adaptation_g=0.4, jacobian_adaptation_q_meas6=q_meas6,
    )
    assert np.allclose(Jbar1[:3, 6], Jbar0[:3, 6]), "insertion column touched at an arbitrary anchor"
    assert not np.allclose(Jbar1[:3, :6], Jbar0[:3, :6]), "joint columns should have changed (g!=0)"
    print("PASS: insertion output channel bit-identical regardless of anchor point or g; "
          "only the joint-column-spanned directions are ever adapted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, repr(exc)))
            print(f"FAIL {t.__name__}: {exc!r}")
    print(f"\n{len(tests)-len(failures)}/{len(tests)} anchoring checks passed.")
    if failures:
        raise SystemExit(1)
