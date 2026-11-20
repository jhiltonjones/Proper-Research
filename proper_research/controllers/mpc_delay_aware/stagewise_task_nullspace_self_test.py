#!/usr/bin/env python3
"""Unit + integration tests for `stagewise_task_nullspace.py`, required
before trusting any gamma=0 replay (per the design discussion this module
implements). Uses real plan/schedule data.

Tests:
  1. Projector identities: P_N^T=P_N, P_N^2=P_N, P_R@P_N~=0, P_R+P_N~=I,
     at every horizon stage.
  2. Pure row-space vector x_R=P_R@x satisfies P_N@x_R=0; pure null vector
     x_N=P_N@x satisfies J_s@x_N=0 (J_s = the SAME scaled, task-projected
     Jacobian the projector was built from).
  3. Rank is taken numerically, not hard-coded -- confirm it equals 2 for
     this system (2D beam-plane task), not 3.
  4. gamma=1 reproduces M0 (DelayAwareBeamOutputTrackingMPC) EXACTLY: H,
     f, and the solved command must match to numerical precision (since
     P_N+P_R=I everywhere).

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.stagewise_task_nullspace_self_test \\
        --plan-dir plans/... --schedule-cache /tmp/....npy
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC
from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC, build_stagewise_projectors
from .target_consistent import build_beam_plane_projection


def run_self_test(plan_dir: str, schedule_cache: str, control_index: int = 100) -> None:
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

    C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
    state_scale = np.asarray(mpc_config.state_error_scale, dtype=float)
    indices = list(range(control_index + 1, control_index + 1 + N))
    J_schedule = schedule[indices]
    P_R, P_N, ranks = build_stagewise_projectors(
        J_schedule=J_schedule, C=C, state_error_scale=state_scale,
    )

    # --- 1: projector identities, every stage ---
    for j in range(N):
        assert np.allclose(P_N[j], P_N[j].T, atol=1e-10), f"stage {j}: P_N not symmetric"
        assert np.allclose(P_N[j] @ P_N[j], P_N[j], atol=1e-8), f"stage {j}: P_N not idempotent"
        assert np.allclose(P_R[j] @ P_N[j], 0.0, atol=1e-8), f"stage {j}: P_R@P_N != 0"
        assert np.allclose(P_R[j] + P_N[j], np.eye(7), atol=1e-8), f"stage {j}: P_R+P_N != I"
    print("[stagewise_task_nullspace self-test] 1/4 PASS: projector identities hold at every "
          "horizon stage (symmetric, idempotent, P_R@P_N~=0, P_R+P_N~=I)")

    # --- 2: pure row/null vectors behave correctly ---
    rng = np.random.default_rng(0)
    S_z = state_scale
    for j in (0, 5, 14):
        Jt = C @ J_schedule[j]
        J_s = Jt * S_z[None, :]
        x = rng.normal(size=7)
        x_R = P_R[j] @ x
        x_N = P_N[j] @ x
        assert np.max(np.abs(P_N[j] @ x_R)) < 1e-8, f"stage {j}: P_N @ (P_R@x) should be 0"
        assert np.max(np.abs(J_s @ x_N)) < 1e-6, f"stage {j}: J_s @ (P_N@x) should be 0"
    print("[stagewise_task_nullspace self-test] 2/4 PASS: pure row-space vectors are annihilated "
          "by P_N; pure null-space vectors are annihilated by the scaled task Jacobian")

    # --- 3: numerical rank is 2 (2D beam-plane task), not hard-coded 3 ---
    assert np.all(ranks == 2), f"expected rank=2 (2D beam-plane task) at every stage, got {ranks}"
    print(f"[stagewise_task_nullspace self-test] 3/4 PASS: numerical rank = 2 at every stage "
          f"(computed, not assumed) -- nullspace is 5D")

    # --- 4: gamma=1 reproduces M0 exactly ---
    m0 = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    g1 = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=1.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float) + np.array([0.002, -0.001, 0.0])
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    x_exec = np.concatenate([q0, q0, [l0]])

    _, d_k = m0._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    H_m0, f_m0, _ = m0._dynamic_qp_terms_exec(x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_k)
    g1._filtered_output_residual = None
    _, d_k_g1 = g1._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    H_g1, f_g1, _ = g1._dynamic_qp_terms_exec(x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_k_g1)

    assert np.allclose(H_m0, H_g1, atol=1e-7), f"gamma=1: H mismatch, max diff={np.max(np.abs(H_m0-H_g1))}"
    assert np.allclose(f_m0, f_g1, atol=1e-6), f"gamma=1: f mismatch, max diff={np.max(np.abs(f_m0-f_g1))}"

    step_m0 = m0.solve_delay_aware(z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
                                    measured_beam_position=p0, control_index=control_index, previous_input=prev)
    step_g1 = g1.solve_delay_aware(z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
                                    measured_beam_position=p0, control_index=control_index, previous_input=prev)
    diff_u0 = float(np.max(np.abs(step_m0.command - step_g1.command)))
    assert diff_u0 < 1e-5, f"gamma=1: command mismatch, max diff={diff_u0:.3e}"
    print(f"[stagewise_task_nullspace self-test] 4/4 PASS: gamma=1 reproduces M0 exactly "
          f"(H diff={np.max(np.abs(H_m0-H_g1)):.2e}, f diff={np.max(np.abs(f_m0-f_g1)):.2e}, "
          f"u0 diff={diff_u0:.2e})")

    print("[stagewise_task_nullspace self-test] ALL PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--control-index", type=int, default=100)
    args = p.parse_args()
    run_self_test(args.plan_dir, args.schedule_cache, args.control_index)


if __name__ == "__main__":
    main()
