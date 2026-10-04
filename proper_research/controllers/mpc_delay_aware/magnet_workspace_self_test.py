#!/usr/bin/env python3
"""Regression guard for the in-QP magnet z-workspace constraint (2026-09-28)
-- see `delay_aware_mpc.py`'s `DelayAwareBeamOutputTrackingMPC._configure_
magnet_workspace` docstring for why this was added: a live frozen-Jacobian
run (Q_N=0, no relinearization) let the redundant joint DOF drift the arm's
flange to z=0.262m while every QP solve that run reported success -- none of
the QP's existing constraints (per-joint box, rate limits, point exclusion)
cover a general Cartesian/TCP-workspace bound, so this was only catchable
after the fact by the external `tcp_out_of_workspace` monitor.

Also covers the SAME-DAY rework of the point-exclusion constraint from a
many-point vessel centreline to a single fixed beam-base point (210.43mm) --
confirms the shared `_configure_magnet_exclusion` machinery still works
correctly at K=1, and that BOTH new constraints (point exclusion + z
workspace) compose correctly in the OSQP row bookkeeping when active
simultaneously (the highest-risk part of this change: two separate row
blocks, two separate `_row_start` trackers, `self.A` grown twice).

This test proves:
  (a) magnet_workspace=None is a genuine no-op (bit-identical to baseline).
  (b) a binding z-bound actually changes the command.
  (c) the resulting COMMANDED magnet z genuinely respects (or nearly
      respects, within linearization error) the true nonlinear z bound.
  (d) OSQP and scipy backends agree (confirms the OSQP `Ax` runtime update
      is wired correctly for the NEW row block, not just the old one).
  (e) the single-point (K=1) beam-base exclusion still works via the SAME
      `_configure_magnet_exclusion` method the many-point vessel version
      used (repurposed, not rewritten).
  (f) BOTH constraints active simultaneously (point exclusion K=1 + z
      workspace) solve correctly with the expected total row count, and
      the workspace block still binds correctly even with the exclusion
      block also present ahead of it in `self.A`.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.magnet_workspace_self_test
"""
from __future__ import annotations

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
# Same fixed beam-base point/radius run_mpc_delay_aware_vessel.py now uses
# (see that module's _BEAM_BASE_PIVOT_XYZ_R / _BEAM_BASE_EXCLUSION_RADIUS_M).
BEAM_BASE_PIVOT_XYZ_R = np.array([[0.525575, -0.719727, 0.013433]])
BEAM_BASE_EXCLUSION_RADIUS_M = 0.21043


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
    disabled = DelayAwareBeamOutputTrackingMPC(magnet_workspace=None, **common)
    step_base = baseline.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    step_disabled = disabled.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    diff_disabled = float(np.max(np.abs(step_base.command - step_disabled.command)))
    assert diff_disabled < 1e-9, f"magnet_workspace=None should be a no-op, diff={diff_disabled:.3e}"
    print(f"[magnet_workspace self-test] disabled == baseline EXACT (diff={diff_disabled:.2e})")

    # (b)/(c) deliberately tight z_max: comfortably BELOW the schedule's own
    # nominal z over the horizon, so the constraint MUST bind. NOTE: at this
    # control_index the magnet's nominal z barely moves across the horizon
    # (~0.03mm total) -- a large forced z_max perturbation (e.g. 10mm) turns
    # genuinely primal-infeasible (confirmed empirically: joint-rate limits
    # can't move the magnet that far within N=15 steps at 0.1s each), which
    # is a real property of the constraint, not a test bug, but not what
    # this test is checking -- 3mm is comfortably solvable while still
    # forcing a real, measurable correction.
    idx = np.clip(control_index + 1 + np.arange(N), 0, reference.sample_count - 1)
    z_nom_horizon = mag_positions[idx][:, 2]
    z_max = float(z_nom_horizon.max() - 0.003)  # 3mm below nominal max -> must bind, stays feasible
    z_min = float(z_nom_horizon.min() - 0.100)  # comfortably slack, not under test
    print(f"[magnet_workspace self-test] nominal z over horizon=[{z_nom_horizon.min()*1e3:.1f},"
          f"{z_nom_horizon.max()*1e3:.1f}]mm, testing z_max={z_max*1e3:.1f}mm (constraint must bind)")

    magnet_workspace = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        z_min_m=z_min, z_max_m=z_max,
    )
    constrained_osqp = DelayAwareBeamOutputTrackingMPC(magnet_workspace=magnet_workspace, **common)
    step_c = constrained_osqp.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_c.success, "constrained OSQP solve failed to converge"
    diff_c = float(np.max(np.abs(step_base.command - step_c.command)))
    assert diff_c > 1e-6, "binding z-workspace bound should change the command, but didn't"
    print(f"[magnet_workspace self-test] binding constraint DOES change the command (diff={diff_c:.3e})")

    # nonzero dual on the workspace row block confirms it's genuinely
    # binding, not just coincidentally satisfied.
    assert step_c.dual_y is not None, "dual_y should be populated on the osqp backend"
    ws_dual_block = step_c.dual_y[-N:]  # workspace rows are always the LAST block appended
    assert float(np.max(np.abs(ws_dual_block))) > 1e-6, (
        "workspace constraint row block has all-zero duals -- not actually binding"
    )
    print(f"[magnet_workspace self-test] workspace row dual block max|y|="
          f"{float(np.max(np.abs(ws_dual_block))):.3e} (nonzero -> genuinely binding)")

    q_cmd0 = step_c.predicted_commands[0, :6]
    fk = urik.forward_kinematics(q_cmd0, robot_kin.dh, robot_kin.T_F_M)
    true_z = float(fk.T_R_target[2, 3])
    print(f"[magnet_workspace self-test] true nonlinear z at commanded q_cmd[0]="
          f"{true_z*1e3:.2f}mm vs z_max={z_max*1e3:.2f}mm")
    assert true_z < z_max + 0.003, (
        f"true z {true_z*1e3:.2f}mm violates z_max {z_max*1e3:.2f}mm by more than "
        "3mm linearization slack"
    )

    # (d) OSQP vs scipy backend agreement
    from dataclasses import replace as dc_replace
    scipy_config = dc_replace(mpc_config, solver_backend="scipy")
    constrained_scipy = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=scipy_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        magnet_workspace=magnet_workspace,
    )
    step_s = constrained_scipy.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_s.success, "constrained scipy solve failed to converge"
    diff_backend = float(np.max(np.abs(step_c.command - step_s.command)))
    print(f"[magnet_workspace self-test] OSQP vs scipy command diff={diff_backend:.3e}")
    assert diff_backend < 1e-4, f"OSQP/scipy backends disagree by {diff_backend:.3e} -- Ax wiring is wrong"

    # (e) K=1 beam-base point exclusion ALONE -- proves the shared
    # `_configure_magnet_exclusion` machinery (previously only exercised
    # with a many-point vessel centreline in magnet_exclusion_self_test.py)
    # binds correctly at K=1 too.
    idx_excl = np.clip(control_index + 1 + np.arange(N), 0, reference.sample_count - 1)
    nominal_gap = np.linalg.norm(
        mag_positions[idx_excl] - BEAM_BASE_PIVOT_XYZ_R[0][None, :], axis=1,
    )
    radius_k1 = float(nominal_gap.min() + 0.005)  # 5mm inside nominal -> must bind
    magnet_exclusion_k1 = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        lumen_C_m=BEAM_BASE_PIVOT_XYZ_R, radius_m=radius_k1,
    )
    excl_only = DelayAwareBeamOutputTrackingMPC(magnet_exclusion=magnet_exclusion_k1, **common)
    step_excl = excl_only.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_excl.success, "K=1 beam-base exclusion solve failed to converge"
    diff_excl = float(np.max(np.abs(step_base.command - step_excl.command)))
    assert diff_excl > 1e-6, "binding K=1 exclusion should change the command, but didn't"
    excl_dual = step_excl.dual_y[-N:]
    assert float(np.max(np.abs(excl_dual))) > 1e-6, "K=1 exclusion row block has all-zero duals"
    q_cmd0_excl = step_excl.predicted_commands[0, :6]
    fk_excl = urik.forward_kinematics(q_cmd0_excl, robot_kin.dh, robot_kin.T_F_M)
    true_gap = float(np.linalg.norm(BEAM_BASE_PIVOT_XYZ_R[0] - fk_excl.T_R_target[:3, 3]))
    print(f"[magnet_workspace self-test] K=1 beam-base exclusion ALONE: binds "
          f"(diff={diff_excl:.3e}, dual max={float(np.max(np.abs(excl_dual))):.3e}), "
          f"true gap={true_gap*1e3:.3f}mm vs radius={radius_k1*1e3:.3f}mm")
    assert true_gap > radius_k1 - 0.003, "K=1 exclusion violated by more than 3mm linearization slack"

    # (f) K=1 beam-base point exclusion + z workspace TOGETHER -- the
    # highest-risk part of this change (two extra row blocks, both appended
    # to self.A, both needing correctly-sized dummy bounds at OSQP setup).
    # Uses the REAL production radius (210.43mm) here, not the artificially
    # tightened radius_k1 above -- at this control_index the real radius is
    # comfortably loose (see nominal_gap ~240mm printed above), so this
    # block is checking composition/row-bookkeeping correctness, not
    # exclusion bindingness (that's already proven standalone above).
    magnet_exclusion = dict(
        position_jacobians=mag_jacobians, nominal_positions_m=mag_positions,
        lumen_C_m=BEAM_BASE_PIVOT_XYZ_R, radius_m=BEAM_BASE_EXCLUSION_RADIUS_M,
    )
    both = DelayAwareBeamOutputTrackingMPC(
        magnet_exclusion=magnet_exclusion, magnet_workspace=magnet_workspace, **common,
    )
    expected_extra_rows = 2 * N  # one N-row block per constraint
    actual_extra_rows = both.A.shape[0] - baseline.A.shape[0]
    assert actual_extra_rows == expected_extra_rows, (
        f"expected {expected_extra_rows} extra rows (2 blocks x N={N}), got {actual_extra_rows} "
        f"-- row bookkeeping is wrong when both constraints are active"
    )
    print(f"[magnet_workspace self-test] combined K=1 exclusion + workspace: "
          f"self.A grew by exactly {actual_extra_rows} rows (2 x N={N}), as expected")

    step_both = both.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_both.success, "combined exclusion+workspace OSQP solve failed to converge"
    print(f"[magnet_workspace self-test] combined solve: status={step_both.status} "
          f"objective={step_both.objective:.4f}")

    # nominal margin for the K=1 beam-base exclusion at this stage/horizon is
    # WAY beyond 210.43mm (the plan operates near the vessel, ~metres from
    # the physical beam base in absolute terms is wrong framing -- check the
    # actual nominal gap instead of assuming it binds; this constraint is a
    # loose safety floor here, not expected to bind on this trajectory. Just
    # confirm it doesn't corrupt the workspace block's own binding behaviour.)
    ws_dual_block_both = step_both.dual_y[-N:]
    assert float(np.max(np.abs(ws_dual_block_both))) > 1e-6, (
        "workspace constraint stopped binding once the exclusion block was ALSO active -- "
        "row-block interaction bug"
    )
    print(f"[magnet_workspace self-test] workspace block still binds with exclusion block also "
          f"active (max|y|={float(np.max(np.abs(ws_dual_block_both))):.3e})")

    print("[magnet_workspace self-test] ALL PASS")


if __name__ == "__main__":
    run_self_test()
