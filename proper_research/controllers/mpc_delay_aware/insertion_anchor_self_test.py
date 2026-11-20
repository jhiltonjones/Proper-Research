#!/usr/bin/env python3
"""w_L=0 <-> R700/gamma=0 exact-equivalence regression guard (2026-09-21),
same spirit as stagewise_task_nullspace_self_test.py's gamma=1<->M0 check.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.insertion_anchor_self_test \\
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

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC
from .insertion_anchor_mpc import InsertionAnchoredTaskNullspaceMPC


def run_self_test(plan_dir: str, schedule_cache: str, control_index: int = 100) -> None:
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(schedule_cache)
    N, DT = 15, 0.1
    RD_SCALE = tuple([DT * 0.40] * 6 + [DT * 10.0 * 2.0e-3])
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [-0.05]), state_max=tuple([2 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([0.10] * 6 + [2.0e-3]), acceleration_limit=tuple([0.40] * 6 + [0.02]),
        input_increment_scale=RD_SCALE, input_tracking_weight=1.0e-2 * 700.0,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0, wall_avoidance_gain=0.0,
    )

    r700 = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    wl0 = InsertionAnchoredTaskNullspaceMPC(
        insertion_state_anchor_weight=0.0, gamma=0.0,
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )

    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float) + np.array([0.002, -0.001, 0.0])
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    x_exec = np.concatenate([q0, q0, [l0]])

    _, d_r700 = r700._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    H_r700, f_r700, _ = r700._dynamic_qp_terms_exec(x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_r700)
    wl0._filtered_output_residual = None
    _, d_wl0 = wl0._estimate_output_residual(measured_state=z_meas, measured_beam_position=p0, control_index=control_index)
    H_wl0, f_wl0, _ = wl0._dynamic_qp_terms_exec(x_exec=x_exec, previous_input=prev, control_index=control_index, estimated_residual=d_wl0)

    h_diff = float(np.max(np.abs(H_r700 - H_wl0)))
    f_diff = float(np.max(np.abs(f_r700 - f_wl0)))
    assert h_diff < 1e-9, f"w_L=0: H mismatch, max diff={h_diff:.3e}"
    assert f_diff < 1e-9, f"w_L=0: f mismatch, max diff={f_diff:.3e}"

    step_r700 = r700.solve_delay_aware(z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
                                        measured_beam_position=p0, control_index=control_index, previous_input=prev)
    step_wl0 = wl0.solve_delay_aware(z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
                                      measured_beam_position=p0, control_index=control_index, previous_input=prev)
    diff_u0 = float(np.max(np.abs(step_r700.command - step_wl0.command)))
    assert diff_u0 < 1e-9, f"w_L=0: command mismatch, max diff={diff_u0:.3e}"

    print(f"[insertion_anchor self-test] w_L=0 <-> R700/gamma=0 EXACT: "
          f"H diff={h_diff:.2e}  f diff={f_diff:.2e}  u0 diff={diff_u0:.2e}")

    # sanity: w_L>0 actually changes the solution (not a silent no-op)
    wl_nonzero = InsertionAnchoredTaskNullspaceMPC(
        insertion_state_anchor_weight=0.1, gamma=0.0,
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    step_wl_nz = wl_nonzero.solve_delay_aware(z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
                                               measured_beam_position=p0, control_index=control_index, previous_input=prev)
    diff_nz = float(np.max(np.abs(step_r700.command - step_wl_nz.command)))
    assert diff_nz > 1e-6, "w_L=0.1 should change the solution vs R700 baseline, but didn't"
    print(f"[insertion_anchor self-test] w_L=0.1 DOES change the solution (diff={diff_nz:.3e}) -- not a no-op")
    print("[insertion_anchor self-test] ALL PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--control-index", type=int, default=100)
    args = p.parse_args()
    run_self_test(args.plan_dir, args.schedule_cache, args.control_index)


if __name__ == "__main__":
    main()
