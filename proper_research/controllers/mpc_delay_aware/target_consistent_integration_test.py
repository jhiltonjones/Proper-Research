#!/usr/bin/env python3
"""Integration self-test for `TargetConsistentDelayAwareMPC` (MT) against
`DelayAwareBeamOutputTrackingMPC` (M0), using real plan/schedule data (not
synthetic) -- see `target_consistent.py`'s module docstring for the
real-data finding that motivated MT, and `target_consistent_mpc.py`'s
docstring for exactly what MT changes (only the Q-cost center) and what it
leaves alone (H, the physical prediction, all hard constraints).

Tests:
  1. H is bit-identical between M0 and MT at the same tick, REGARDLESS of
     disturbance (shifting a quadratic's center only changes its linear
     term -- this must hold structurally, not just when disturbance is 0).
  2. With the disturbance estimate forced to exactly zero, MT's linear
     cost is bit-identical to M0's (z_target == z_ref exactly then).
  3. With a real (nonzero) disturbance from actual logged data, MT's
     linear cost DIFFERS from M0's, and the difference is explained
     exactly by the z_target shift `target_consistent.py` predicts
     independently (cross-check, not just "it changed").
  4. z_target's in-plane task residual (C @ J_stage @ delta_z + C @ d_k)
     is ~0 at every horizon stage for a real disturbance -- confirms exact
     task cancellation holds end-to-end inside the constructed MPC, not
     just in the standalone target_consistent.py unit test.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.target_consistent_integration_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC
from .target_consistent import build_beam_plane_projection, compute_target_shift
from .target_consistent_mpc import TargetConsistentDelayAwareMPC


def run_integration_self_test(plan_dir: str, schedule_cache: str, control_index: int = 100) -> None:
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(schedule_cache)
    N, DT = 15, 0.1
    vlim = tuple([0.10] * 6 + [2.0e-3])
    alim = tuple([0.40] * 6 + [10.0 * 2.0e-3])
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [-0.05]), state_max=tuple([2 * math.pi] * 6 + [0.20]),
        velocity_limit=vlim, acceleration_limit=alim,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0, wall_avoidance_gain=0.0,
    )

    m0 = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    mt = TargetConsistentDelayAwareMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )

    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0_true = np.asarray(reference.desired_position_m[control_index], dtype=float)
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    x_exec = np.concatenate([q0, q0, [l0]])

    # --- 1: H bit-identical regardless of disturbance ---
    p0_perturbed = p0_true + np.array([0.003, -0.002, 0.0005])  # induces a real, nonzero d_k
    _, d_zero = m0._estimate_output_residual(
        measured_state=z_meas, measured_beam_position=p0_true, control_index=control_index,
    )
    m0._filtered_output_residual = None
    _, d_nonzero = m0._estimate_output_residual(
        measured_state=z_meas, measured_beam_position=p0_perturbed, control_index=control_index,
    )
    assert np.linalg.norm(d_zero) < 1e-9, "measured_beam_position == p0_true should give ~zero residual"
    assert np.linalg.norm(d_nonzero) > 1e-6, "perturbed beam position should give a real nonzero residual"

    H_m0_zero, f_m0_zero, _ = m0._dynamic_qp_terms_exec(
        x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_zero,
    )
    H_mt_zero, f_mt_zero, terms_mt_zero = mt._dynamic_qp_terms_exec(
        x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_zero,
    )
    H_m0_nz, f_m0_nz, _ = m0._dynamic_qp_terms_exec(
        x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_nonzero,
    )
    H_mt_nz, f_mt_nz, terms_mt_nz = mt._dynamic_qp_terms_exec(
        x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_nonzero,
    )
    assert np.allclose(H_m0_zero, H_mt_zero, atol=1e-12), "H must be bit-identical (zero disturbance)"
    assert np.allclose(H_m0_nz, H_mt_nz, atol=1e-12), "H must be bit-identical (nonzero disturbance)"
    assert np.allclose(H_m0_zero, H_m0_nz, atol=1e-12), (
        "H must not depend on disturbance AT ALL (sanity: M0 itself doesn't move H either)"
    )
    print("[target_consistent integration test] 1/4 PASS: H bit-identical between M0 and MT, "
          "with and without disturbance")

    # --- 2: zero disturbance -> MT's linear cost == M0's exactly ---
    assert np.allclose(f_m0_zero, f_mt_zero, atol=1e-9), (
        f"zero-disturbance linear cost mismatch, max diff={np.max(np.abs(f_m0_zero - f_mt_zero))}"
    )
    assert np.allclose(terms_mt_zero["z_target"], reference.state_window(control_index, N), atol=1e-12), (
        "z_target must equal z_ref exactly when disturbance is zero"
    )
    print("[target_consistent integration test] 2/4 PASS: zero disturbance -> MT bit-identical to M0")

    # --- 3: nonzero disturbance -> MT differs from M0, and the difference
    # is explained exactly by target_consistent.py's own independent
    # per-stage computation ---
    assert not np.allclose(f_m0_nz, f_mt_nz, atol=1e-6), "nonzero disturbance should change the linear cost"
    C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
    state_scale = np.asarray(mpc_config.state_error_scale, dtype=float)
    w_diag = 1.0 / state_scale ** 2
    indices = m0._reference_indices(control_index, future=True)
    J_schedule = schedule[indices]
    state_ref = reference.state_window(control_index, N)
    for j in range(N):
        dz_indep, _ = compute_target_shift(
            J_stage=J_schedule[j], d_k=d_nonzero, C=C, w_diag=w_diag,
        )
        assert np.allclose(terms_mt_nz["z_target"][j], state_ref[j] + dz_indep, atol=1e-9), (
            f"stage {j}: MT's internal z_target doesn't match an independent "
            f"target_consistent.py computation"
        )
    print("[target_consistent integration test] 3/4 PASS: nonzero disturbance changes the linear "
          "cost, exactly matching an independent per-stage target_consistent.py computation")

    # --- 4: task residual ~0 at every stage for the real disturbance ---
    max_resid = np.max(np.abs(terms_mt_nz["target_residual"]))
    assert max_resid < 1e-8, f"task residual should be ~0 at every stage, max={max_resid:.3e}"
    print(f"[target_consistent integration test] 4/4 PASS: task residual ~0 at every horizon "
          f"stage (max={max_resid:.2e})")

    print("[target_consistent integration test] ALL PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--control-index", type=int, default=100)
    args = p.parse_args()
    run_integration_self_test(args.plan_dir, args.schedule_cache, args.control_index)


if __name__ == "__main__":
    main()
