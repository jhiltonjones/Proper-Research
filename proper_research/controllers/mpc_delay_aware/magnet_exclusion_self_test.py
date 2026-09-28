#!/usr/bin/env python3
"""Regression guard for the in-QP magnet-to-vessel-wall exclusion
constraint (2026-09-28) -- see `delay_aware_mpc.py`'s
`DelayAwareBeamOutputTrackingMPC._configure_magnet_exclusion` docstring for
why this was added: the frozen-Jacobian run
(close_loop_logs/myrun/vessel_live_trimmed6mm_mpc_frozen_20260928T140358Z)
aborted at 143/525 ticks on a magnet-exclusion violation while every single
logged QP solve reports success -- the optimizer had no idea the wall
existed. This test proves the new constraint (a) is a genuine no-op when
disabled (bit-identical to the pre-existing controller), (b) actually
changes the solution once configured with a binding radius, (c) the
resulting COMMANDED magnet position genuinely respects (or nearly respects,
within linearization error) the true nonlinear distance-to-lumen radius,
and (d) the OSQP and scipy backends agree, confirming the OSQP `Ax` runtime
update is wired correctly.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.magnet_exclusion_self_test
"""
from __future__ import annotations

import json
import math

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)
from proper_research.planning.planning_context import make_robot_config

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC

PLAN_DIR = "plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path"
SCHEDULE_CACHE = "/tmp/vessel_live_trimmed6mm_schedule_frozen.npy"
LUMEN_FILE = "vessel_lumen_robot_frame_raised3cm_trimmed6mm_2026-09-28.json"


def _build_magnet_schedule(reference, robot_kin):
    q = np.asarray(reference.state[:, :6], dtype=float)
    jacobians = np.empty((q.shape[0], 3, 6), dtype=float)
    positions = np.empty((q.shape[0], 3), dtype=float)
    for i in range(q.shape[0]):
        fk = urik.forward_kinematics(q[i], robot_kin.dh, robot_kin.T_F_M)
        positions[i] = fk.T_R_target[:3, 3]
        jacobians[i] = urik.geometric_jacobian(q[i], robot_kin.dh, robot_kin.T_F_M)[:3, :]
    return jacobians, positions


def run_self_test(control_index: int = 60) -> None:
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    schedule = np.load(SCHEDULE_CACHE)
    with open(LUMEN_FILE) as f:
        lumen = json.load(f)
    lumen_C_m = np.asarray(lumen["lumen_C_m"], dtype=float)

    robot_kin = _resolve_robot_kinematics(make_robot_config())
    mag_jacobians, mag_positions = _build_magnet_schedule(reference, robot_kin)

    N, DT = 15, 0.1
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [0.0]), state_max=tuple([2 * math.pi] * 6 + [0.065]),
        velocity_limit=tuple([0.10] * 6 + [2.0e-3]), acceleration_limit=tuple([0.40] * 6 + [0.02]),
        input_tracking_weight=1.0e-2 * 700.0, input_increment_weight=1.0e-3,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0, wall_avoidance_gain=0.0,
    )

    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float)
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    common = dict(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )

    # (a) disabled -> bit-identical to the pre-existing controller
    baseline = DelayAwareBeamOutputTrackingMPC(**common)
    disabled = DelayAwareBeamOutputTrackingMPC(magnet_exclusion=None, **common)
    step_base = baseline.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    step_disabled = disabled.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    diff_disabled = float(np.max(np.abs(step_base.command - step_disabled.command)))
    assert diff_disabled < 1e-9, f"magnet_exclusion=None should be a no-op, diff={diff_disabled:.3e}"
    print(f"[magnet_exclusion self-test] disabled == baseline EXACT (diff={diff_disabled:.2e})")

    # deliberately aggressive radius: comfortably ABOVE the schedule's own
    # nominal minimum margin at this stage, so the constraint MUST bind.
    idx = np.clip(control_index + 1 + np.arange(N), 0, reference.sample_count - 1)
    diffs = mag_positions[idx][:, None, :] - lumen_C_m[None, :, :]
    d_nom = np.linalg.norm(diffs, axis=2).min(axis=1)
    radius = float(d_nom.min() + 0.010)  # 10mm inside the nominal margin -> must bind
    print(f"[magnet_exclusion self-test] nominal min margin over horizon={d_nom.min()*1e3:.1f}mm, "
          f"testing radius={radius*1e3:.1f}mm (constraint must bind)")

    magnet_exclusion = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        lumen_C_m=lumen_C_m, radius_m=radius,
    )
    constrained_osqp = DelayAwareBeamOutputTrackingMPC(magnet_exclusion=magnet_exclusion, **common)
    step_c = constrained_osqp.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_c.success, "constrained OSQP solve failed to converge"
    diff_c = float(np.max(np.abs(step_base.command - step_c.command)))
    assert diff_c > 1e-6, "binding magnet-exclusion radius should change the command, but didn't"
    print(f"[magnet_exclusion self-test] binding constraint DOES change the command (diff={diff_c:.3e})")

    # (c) verify the TRUE nonlinear commanded magnet position respects the
    # radius (within linearization error -- the constraint is linear, the
    # real FK/distance is not, so allow a small margin).
    q_cmd0 = step_c.predicted_commands[0, :6]
    fk = urik.forward_kinematics(q_cmd0, robot_kin.dh, robot_kin.T_F_M)
    true_gap = float(np.linalg.norm(lumen_C_m - fk.T_R_target[:3, 3][None, :], axis=1).min())
    print(f"[magnet_exclusion self-test] true nonlinear gap at commanded q_cmd[0]="
          f"{true_gap*1e3:.2f}mm vs radius={radius*1e3:.2f}mm")
    assert true_gap > radius - 0.003, (
        f"true gap {true_gap*1e3:.2f}mm violates radius {radius*1e3:.2f}mm by more than "
        "3mm linearization slack"
    )

    # (d) OSQP vs scipy backend agreement
    from dataclasses import replace as dc_replace
    scipy_config = dc_replace(mpc_config, solver_backend="scipy")
    constrained_scipy = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=scipy_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        magnet_exclusion=magnet_exclusion,
    )
    step_s = constrained_scipy.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_s.success, "constrained scipy solve failed to converge"
    diff_backend = float(np.max(np.abs(step_c.command - step_s.command)))
    print(f"[magnet_exclusion self-test] OSQP vs scipy command diff={diff_backend:.3e}")
    assert diff_backend < 1e-4, f"OSQP/scipy backends disagree by {diff_backend:.3e} -- Ax wiring is wrong"

    print("[magnet_exclusion self-test] ALL PASS")


if __name__ == "__main__":
    run_self_test()
