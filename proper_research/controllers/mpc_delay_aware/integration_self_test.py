#!/usr/bin/env python3
"""Integration-level regression tests for `DelayAwareBeamOutputTrackingMPC`,
run against the real rectangle plan + genuine LTV schedule (not synthetic
data -- these need an actual reference/Jacobian schedule to compare against
the real `BeamOutputTrackingMPC`). See `prediction.py`'s `run_self_test()`
for the kinematics-only tests that use synthetic data and need no plan.

These are the six tests specified alongside the delay-aware MPC design:

  1. delay_samples=0 reproduces the existing MPC's H and u0 exactly.
  2. Perturbing z_meas changes d_k but never the (Ep, Sp) prediction
     matrices, which are built once in __init__ and never depend on it.
  3. Perturbing q_cmd (x_exec) changes the physical prediction but never
     d_k, which is a function of z_meas only.
  4. Finite-differencing the complete beam prediction w.r.t. u_{q,0} gives
     zero at stages 1-2 and exactly dt*J_{q,k+3} at stage 3.
  5. Finite-differencing w.r.t. u_{L,0} gives dt*J_{L,k+1} at stage 1 --
     insertion has immediate authority (undelayed).
  6. The first rate constraint depends only on the explicit previous_input
     argument (u_applied,k-1), never on q_cmd_prev / the delay queue.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.integration_self_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig, BeamOutputTrackingMPC,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC

Array = np.ndarray


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

    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float)
    z_meas = np.concatenate([q0, [l0]])
    prev = np.array([0.001, -0.002, 0.0005, 0.001, -0.0007, 0.0003, 0.0002])

    # --- 1: delay_samples=0 reproduces the existing MPC exactly ---
    old = BeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule,
    )
    new0 = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=0, beta_d=1.0,
    )
    assert np.allclose(old.H, new0.H, atol=1e-10), f"H mismatch, max diff={np.max(np.abs(old.H - new0.H))}"
    old_step = old.solve(measured_state=z_meas, measured_beam_position=p0,
                          control_index=control_index, previous_input=prev)
    new0_step = new0.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert old_step.success and new0_step.success
    diff_u0 = float(np.max(np.abs(old_step.command - new0_step.command)))
    assert diff_u0 < 1.0e-6, f"u0 mismatch at delay_samples=0: {diff_u0:.3e}"
    print(f"[integration self-test] 1/6 PASS: H identical, u0 identical to {diff_u0:.1e} "
          f"at delay_samples=0")

    # --- 2 & 3: z_meas/x_exec role separation ---
    new2 = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    Ep_before, Sp_before = new2.Ep.copy(), new2.Sp.copy()
    z_meas_pert = z_meas.copy()
    z_meas_pert[:3] += 0.003
    new2._filtered_output_residual = None
    _, d_before = new2._estimate_output_residual(
        measured_state=z_meas, measured_beam_position=p0, control_index=control_index,
    )
    new2._filtered_output_residual = None
    _, d_after = new2._estimate_output_residual(
        measured_state=z_meas_pert, measured_beam_position=p0, control_index=control_index,
    )
    assert np.allclose(new2.Ep, Ep_before) and np.allclose(new2.Sp, Sp_before), "Ep/Sp must never depend on z_meas"
    assert not np.allclose(d_before, d_after, atol=1.0e-9), "d_k should change when z_meas changes"
    print("[integration self-test] 2/6 PASS: Ep/Sp unaffected by z_meas; d_k changed "
          f"({np.linalg.norm(d_before)*1e3:.4f}mm -> {np.linalg.norm(d_after)*1e3:.4f}mm)")

    new2._filtered_output_residual = None
    _, d_a = new2._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    q_cmd_pert = q0.copy()
    q_cmd_pert[:3] += 0.003
    new2._filtered_output_residual = None
    _, d_b = new2._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    assert np.allclose(d_a, d_b, atol=1.0e-12), "d_k must NOT depend on q_cmd"
    x_exec_a = np.concatenate([q0, q0, [l0]])
    x_exec_b = np.concatenate([q_cmd_pert, q0, [l0]])
    free_a, free_b = new2.Ep @ x_exec_a, new2.Ep @ x_exec_b
    assert not np.allclose(free_a, free_b), "predicted physical states should change with q_cmd"
    print("[integration self-test] 3/6 PASS: d_k identical when only q_cmd changes; "
          f"physical prediction changed by {np.max(np.abs(free_a-free_b))*1e3:.4f}mm")

    # --- 4 & 5: finite-difference the complete beam prediction ---
    x_exec = np.concatenate([q0, q0, [l0]])
    new2._filtered_output_residual = None
    _, d_k = new2._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    G, constant_error, _, _, _, _ = new2._beam_prediction_terms_exec(
        x_exec=x_exec, control_index=control_index, estimated_residual=d_k,
    )

    def p_stack(v: Array) -> Array:
        return (G @ v + constant_error).reshape(N, 3)

    v0 = np.zeros(N * 7)
    p_base = p_stack(v0)
    eps = 1.0e-4
    for i in range(6):
        v_pert = v0.copy()
        v_pert[i] = eps
        dp = (p_stack(v_pert) - p_base) / eps
        assert np.allclose(dp[0], 0, atol=1.0e-9), f"d(p_1)/d(u_q0[{i}]) should be 0"
        assert np.allclose(dp[1], 0, atol=1.0e-9), f"d(p_2)/d(u_q0[{i}]) should be 0"
    indices = new2._reference_indices(control_index, future=True)
    expected_dp3 = DT * schedule[indices[2]][:, :6]
    dp3_cols = []
    for i in range(6):
        v_pert = v0.copy()
        v_pert[i] = eps
        dp3_cols.append((p_stack(v_pert)[2] - p_base[2]) / eps)
    dp3 = np.array(dp3_cols).T
    assert np.allclose(dp3, expected_dp3, atol=1.0e-9), "d(p_3)/d(u_q0) mismatch"
    print("[integration self-test] 4/6 PASS: d(p_1)/du_q0=0, d(p_2)/du_q0=0, "
          "d(p_3)/du_q0 = dt*J_q,k+3 exactly")

    v_pert = v0.copy()
    v_pert[6] = eps
    dp1_dL0 = (p_stack(v_pert)[0] - p_base[0]) / eps
    expected = DT * schedule[indices[0]][:, 6]
    assert np.allclose(dp1_dL0, expected, atol=1.0e-9), "d(p_1)/d(u_L0) mismatch"
    print("[integration self-test] 5/6 PASS: d(p_1)/d(u_L,0) = dt*J_L,k+1 exactly "
          "(insertion has immediate authority)")

    # --- 6: rate constraint uses previous_input only ---
    lower_a, upper_a = new2._constraint_bounds_exec(x_exec=x_exec, previous_input=prev)
    x_exec_diff_history = np.concatenate([q0, q0 + 0.05, [l0]])
    lower_b, upper_b = new2._constraint_bounds_exec(x_exec=x_exec_diff_history, previous_input=prev)
    sl = slice(2 * new2.nu, 2 * new2.nu + 7)
    assert np.allclose(lower_a[sl], lower_b[sl]) and np.allclose(upper_a[sl], upper_b[sl]), (
        "first rate-constraint bounds must not depend on q_cmd_prev"
    )
    expected_lower0 = prev - DT * np.asarray(alim)
    expected_upper0 = prev + DT * np.asarray(alim)
    assert np.allclose(lower_a[sl], expected_lower0) and np.allclose(upper_a[sl], expected_upper0)
    print("[integration self-test] 6/6 PASS: first rate constraint uses u_appl,k-1 only, "
          "unaffected by the delay queue")

    print("[integration self-test] ALL PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--control-index", type=int, default=100)
    args = p.parse_args()
    run_integration_self_test(args.plan_dir, args.schedule_cache, args.control_index)


if __name__ == "__main__":
    main()
