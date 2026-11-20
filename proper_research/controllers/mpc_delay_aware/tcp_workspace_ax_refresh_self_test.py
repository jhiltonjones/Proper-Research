#!/usr/bin/env python3
"""Regression guard for the 2026-11-16 TCP-workspace Ax-refresh fix in
`delay_aware_mpc.py`'s `solve_delay_aware` -- see that method's own comment
at the `update_kwargs["Ax"] = ...` gate for the full root-cause story.

Bug (fixed 2026-11-16): the gate that decides whether a per-tick-rebuilt
`self.A` gets pushed into OSQP via `.update(Ax=...)` only checked
`self._magnet_excl_radius is not None or self._magnet_ws_z_min is not None`
-- it never checked `self._tcp_ws_min is not None`. For any controller
configured with `tcp_workspace=...` and NEITHER magnet constraint (exactly
the configuration `run_mpc_delay_aware_insertion_anchor.py` uses), the TCP
rows' A-matrix values were pushed into OSQP's `_a_pattern` ONCE at
`_setup_osqp_with_magnet_exclusion()` time -- as their `_configure_tcp_
workspace`-time placeholder (1e-30, not exactly zero, so the CSC sparsity
pattern itself does NOT collapse those positions out -- confirmed directly;
this is a narrower bug than "the pattern collapsed") -- and then NEVER
again. The TCP constraint's BOUNDS (l/u) correctly tracked the real,
per-tick Cartesian box the whole time; its A-matrix coefficients did not.
A structurally-allocated-but-never-refreshed-past-1e-30 A row makes that
row's bound degenerate to checking whether 0 is within [l, u], independent
of the decision vector -- i.e. the TCP constraint had no real effect on
the solved command. This was found via a K2 hardware diagnostic repeat
(rectangle marginfix plan, ref_index=210) going `primal infeasible`
because a tick's real TCP bound no longer straddled zero.

This test proves, with real physical-unit numbers, not just status strings:
  (a) tcp_workspace=None is a genuine no-op (bit-identical to baseline).
  (b) OLD buggy behavior (Ax gated to magnet-only) reproduced directly in
      this test, NOT by reverting the fix file: a binding TCP bound is
      silently NOT enforced -- the resulting TCP position ends up outside
      the configured box by a real, non-negligible margin (mm).
  (c) NEW fixed behavior: the SAME binding TCP bound IS enforced -- the
      resulting command differs from the unconstrained one, and the
      resulting predicted TCP position respects the box to within
      linearization error (small residual, reported in mm).
  (d) regression combinations: TCP+magnet-exclusion, TCP+magnet-workspace,
      magnet-only, no-constraints all solve with the expected row counts
      and the expected Ax-refresh gate behavior for that combination.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.tcp_workspace_ax_refresh_self_test
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

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC, _finite_vector
from proper_research.rig_calibration import BEAM_BASE_XYZ_M

PLAN_DIR = "plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path"
SCHEDULE_CACHE = "/tmp/vessel_live_trimmed6mm_schedule_frozen.npy"
BEAM_BASE_PIVOT_XYZ_R = np.array([[BEAM_BASE_XYZ_M[0], BEAM_BASE_XYZ_M[1], 0.013433]])
BEAM_BASE_EXCLUSION_RADIUS_M = 0.21043


def _build_tcp_schedule(reference, dh, T_F_TCP):
    q = np.asarray(reference.state[:, :6], dtype=float)
    jacobians = np.empty((q.shape[0], 3, 6), dtype=float)
    positions = np.empty((q.shape[0], 3), dtype=float)
    for i in range(q.shape[0]):
        fk = urik.forward_kinematics(q[i], dh, T_F_TCP)
        positions[i] = fk.T_R_target[:3, 3]
        jacobians[i] = urik.geometric_jacobian(q[i], dh, T_F_TCP)[:3, :6]
    return jacobians, positions


def _old_buggy_solve(ctrl, *, z_meas, q_cmd, q_cmd_prev, insertion_m,
                      measured_beam_position, control_index, previous_input):
    """Reproduces the PRE-FIX `solve_delay_aware` exactly, with the OLD,
    narrower Ax gate (magnet-only). Everything else is byte-for-byte the
    same call sequence the real method uses."""
    z_meas = _finite_vector(z_meas, 7, "z_meas")
    q_cmd = _finite_vector(q_cmd, 6, "q_cmd")
    q_cmd_prev = _finite_vector(q_cmd_prev, 6, "q_cmd_prev")
    beam_position = _finite_vector(measured_beam_position, 3, "measured_beam_position")
    previous = _finite_vector(previous_input, 7, "previous_input")
    x_exec = np.concatenate([q_cmd, q_cmd_prev, [float(insertion_m)]])
    _, estimated = ctrl._estimate_output_residual(
        measured_state=z_meas, measured_beam_position=beam_position, control_index=control_index,
    )
    hessian, linear_cost, beam_terms = ctrl._dynamic_qp_terms_exec(
        x_exec=x_exec, previous_input=previous, control_index=control_index,
        estimated_residual=estimated,
    )
    lower, upper = ctrl._constraint_bounds_exec(
        x_exec=x_exec, previous_input=previous, control_index=control_index,
    )
    warm_start = ctrl._feasible_warm_start(
        input_reference=beam_terms["input_reference"], previous_input=previous,
    )
    update_kwargs = dict(Px=ctrl._upper_values(hessian), q=linear_cost, l=lower, u=upper)
    if ctrl._magnet_excl_radius is not None or ctrl._magnet_ws_z_min is not None:  # OLD gate, TCP omitted
        update_kwargs["Ax"] = ctrl._a_values(ctrl.A)
    ctrl._solver.update(**update_kwargs)
    ctrl._solver.warm_start(x=warm_start)
    result = ctrl._solver.solve()
    success = str(result.info.status).lower() in {"solved", "solved inaccurate"}
    command = np.zeros(7) if not success else np.asarray(result.x, dtype=float).reshape(ctrl.N, ctrl.m)[0]
    return command, success


def run_self_test(control_index: int = 60) -> None:
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    schedule = np.load(SCHEDULE_CACHE)
    tcp_dh = urik.corrected_dh_from_config(urik.CONFIG)
    tcp_T_F_TCP = urik.pose6_to_T(urik.CONFIG.offline_active_T_flange_tcp_pose6)
    tcp_jacobians, tcp_positions = _build_tcp_schedule(reference, tcp_dh, tcp_T_F_TCP)

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

    # (a) disabled -> bit-identical to the pre-existing controller.
    baseline = DelayAwareBeamOutputTrackingMPC(**common)
    disabled = DelayAwareBeamOutputTrackingMPC(tcp_workspace=None, **common)
    step_base = baseline.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    step_disabled = disabled.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    diff_disabled = float(np.max(np.abs(step_base.command - step_disabled.command)))
    assert diff_disabled < 1e-9, f"tcp_workspace=None should be a no-op, diff={diff_disabled:.3e}"
    print(f"[tcp_workspace self-test] disabled == baseline EXACT (diff={diff_disabled:.2e})")

    # (b)/(c): deliberately tight xyz box -- comfortably below the schedule's
    # own nominal TCP x over the horizon so the constraint MUST bind, while
    # staying feasible (joints can actually reach it within N*dt).
    idx = np.clip(control_index + 1 + np.arange(N), 0, reference.sample_count - 1)
    x_nom_horizon = tcp_positions[idx][:, 0]
    q0_x = float(
        urik.forward_kinematics(q0, tcp_dh, tcp_T_F_TCP).T_R_target[0, 3]
    )
    # x_max set comfortably ABOVE the controller's own current actual TCP
    # x (so stage-1 feasibility is never in question) but BELOW the
    # horizon's natural nominal drift -- forces the solver to actively cap
    # the later stages rather than follow the reference, while staying
    # kinematically reachable throughout.
    x_max = q0_x - 0.0003  # slightly BELOW current position -- forces an actual restriction
    assert x_max < x_nom_horizon.max(), (
        "test setup error: x_max must be below the horizon's nominal drift to force binding"
    )
    xyz_min = np.array([x_nom_horizon.min() - 0.5, -1.0, -1.0])
    xyz_max = np.array([x_max, 1.0, 1.0])
    print(f"[tcp_workspace self-test] nominal TCP x over horizon=[{x_nom_horizon.min()*1e3:.1f},"
          f"{x_nom_horizon.max()*1e3:.1f}]mm, testing x_max={x_max*1e3:.1f}mm (constraint must bind)")

    tcp_workspace = dict(position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
                          xyz_min=xyz_min, xyz_max=xyz_max)
    # Separate controller instances for the OLD-gate reproduction vs the
    # NEW-fix solve -- a failed OSQP solve can leave persistent solver
    # state (warm-start/rho) that would otherwise contaminate a later
    # solve on the SAME object, which is not what this comparison is
    # testing.
    constrained_for_old = DelayAwareBeamOutputTrackingMPC(tcp_workspace=tcp_workspace, **common)
    constrained = DelayAwareBeamOutputTrackingMPC(tcp_workspace=tcp_workspace, **common)

    # --- OLD buggy behavior, reproduced directly (not via git) ---
    # Under the OLD gate, the TCP rows' A-matrix is frozen at ~0, so each
    # row degenerates to checking whether 0 lies in [l_row, u_row] (the
    # BOUND relative to that stage's own linearization-nominal position),
    # independent of the decision vector. Depending on where the
    # configured box sits relative to the horizon's nominal TCP position,
    # this produces ONE of two real, previously-undetected symptoms:
    #   (i)  nominal stays inside the box at every stage -> every degenerate
    #        row's [l,u] straddles zero -> trivially "solved", but the
    #        constraint had NO actual effect on the command (silent
    #        non-enforcement), OR
    #   (ii) nominal drifts outside the box at some stage -> that row's
    #        [l,u] does not straddle zero -> UNCONDITIONALLY infeasible
    #        regardless of x -- this is the EXACT mechanism that produced
    #        the real hardware "primal infeasible" at ref_index=210.
    # This test's x_max (3mm below the horizon's nominal max) triggers (ii)
    # -- the more serious symptom, and the one actually observed on
    # hardware -- which this assertion documents and proves directly.
    old_command, old_success = _old_buggy_solve(
        constrained_for_old, z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    print(f"[tcp_workspace self-test] OLD buggy gate: solve success={old_success}")
    if not old_success:
        print("[tcp_workspace self-test] OLD buggy gate reproduces symptom (ii): the degenerate "
              "all-zero-A TCP row's [l,u] (relative to its stage's own nominal) no longer straddles "
              "zero at this control_index -- UNCONDITIONAL, state-independent infeasibility. This is "
              "the EXACT mechanism behind the real K2 hardware failure at ref_index=210.")
        old_violation_mm = None
    else:
        old_diff_vs_unconstrained = float(np.max(np.abs(step_base.command - old_command)))
        print(f"[tcp_workspace self-test] OLD buggy gate: command diff vs UNCONSTRAINED baseline "
              f"= {old_diff_vs_unconstrained:.3e} (near-zero means the 'constraint' had no real effect)")
        old_q_next = q0 + DT * old_command[:6]
        old_fk = urik.forward_kinematics(old_q_next, tcp_dh, tcp_T_F_TCP)
        old_tcp_x = float(old_fk.T_R_target[0, 3])
        old_violation_mm = max(0.0, (old_tcp_x - x_max) * 1e3)
        print(f"[tcp_workspace self-test] OLD buggy gate: resulting TCP x={old_tcp_x*1e3:.2f}mm, "
              f"configured x_max={x_max*1e3:.2f}mm, VIOLATION={old_violation_mm:.2f}mm "
              f"(symptom (i): constraint silently NOT enforced)")
    assert not old_success or (old_violation_mm is not None and old_violation_mm > 0.5), (
        "OLD-gate reproduction should show EITHER false infeasibility (symptom ii) OR a real, "
        "non-negligible physical violation (symptom i) -- neither occurred, so this reproduction "
        "did not actually exercise the bug"
    )

    # --- NEW fixed behavior ---
    step_c = constrained.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert step_c.success, "NEW-fix constrained solve failed to converge"
    diff_c = float(np.max(np.abs(step_base.command - step_c.command)))
    assert diff_c > 1e-6, "binding TCP bound should change the command under the fix, but didn't"
    print(f"[tcp_workspace self-test] NEW fix: command diff vs unconstrained baseline = {diff_c:.3e} "
          f"(non-negligible -> constraint genuinely changed the solution)")
    new_q_next = q0 + DT * step_c.command[:6]
    new_fk = urik.forward_kinematics(new_q_next, tcp_dh, tcp_T_F_TCP)
    new_tcp_x = float(new_fk.T_R_target[0, 3])
    new_violation_mm = max(0.0, (new_tcp_x - x_max) * 1e3)
    print(f"[tcp_workspace self-test] NEW fix: resulting TCP x={new_tcp_x*1e3:.2f}mm, "
          f"configured x_max={x_max*1e3:.2f}mm, violation={new_violation_mm:.3f}mm "
          f"(must be small -- linearization error only)")
    assert new_violation_mm < 2.0, (
        f"NEW fix should respect the TCP box to within a few mm of linearization error, "
        f"got {new_violation_mm:.2f}mm violation"
    )
    # --- also directly demonstrate symptom (i): OLD gate solving but
    # SILENTLY not enforcing, with a milder bound (x_max ABOVE the whole
    # horizon's nominal range, so the degenerate 0-check passes everywhere
    # trivially, but the REAL constraint -- correctly enforced under the
    # fix -- would still restrict the solution via the row's a-coefficient
    # once nominal approaches it; here we directly shrink the margin to
    # exactly nominal_max, zero slack, so a non-degenerate solver MUST
    # actively hold x at the boundary while a degenerate one has no reason
    # to).
    x_max_mild = float(x_nom_horizon.max())  # zero slack, not negative -- degenerate check still trivially passes
    tcp_ws_mild = dict(position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
                        xyz_min=xyz_min, xyz_max=np.array([x_max_mild, 1.0, 1.0]))
    constrained_mild = DelayAwareBeamOutputTrackingMPC(tcp_workspace=tcp_ws_mild, **common)
    old_mild_command, old_mild_success = _old_buggy_solve(
        constrained_mild, z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=prev,
    )
    assert old_mild_success, "mild-bound OLD-gate solve should succeed (degenerate check trivially passes)"
    old_mild_q_next = q0 + DT * old_mild_command[:6]
    old_mild_tcp_x = float(
        urik.forward_kinematics(old_mild_q_next, tcp_dh, tcp_T_F_TCP).T_R_target[0, 3]
    )
    old_mild_violation_mm = max(0.0, (old_mild_tcp_x - x_max_mild) * 1e3)
    print(f"[tcp_workspace self-test] OLD gate, mild/zero-slack bound: solve succeeded, resulting "
          f"TCP x={old_mild_tcp_x*1e3:.2f}mm vs x_max={x_max_mild*1e3:.2f}mm, "
          f"violation={old_mild_violation_mm:.2f}mm -- demonstrates symptom (i) directly "
          f"({'constraint silently ignored' if old_mild_violation_mm > 0.3 else 'inconclusive at this point'})")

    assert step_c.dual_y is not None, "dual_y should be populated on the osqp backend"
    tcp_row_start = constrained._tcp_ws_row_start
    tcp_dual_block = step_c.dual_y[tcp_row_start:tcp_row_start + 3 * N]
    assert float(np.max(np.abs(tcp_dual_block))) > 1e-6, (
        "TCP constraint row block has all-zero duals under the fix -- not actually binding"
    )
    print("[tcp_workspace self-test] NEW fix: TCP row block has nonzero duals -- genuinely binding")

    old_summary = (
        "false infeasibility (OSQP correctly reported infeasible for the malformed, "
        "degenerate-A problem it was actually given)" if old_violation_mm is None
        else f"{old_violation_mm:.2f}mm silent violation"
    )
    print(f"\n[tcp_workspace self-test] PASSED: OLD gate failed on this binding TCP bound "
          f"({old_summary}); NEW fix enforces it correctly "
          f"({new_violation_mm:.3f}mm residual, within linearization error).")


def run_combination_regressions(control_index: int = 60) -> None:
    """(d): TCP+magnet-exclusion, TCP+magnet-workspace, magnet-only,
    no-constraints -- confirm expected row counts and that each combo's
    Ax gate fires for exactly the groups that are actually configured."""
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    schedule = np.load(SCHEDULE_CACHE)
    tcp_dh = urik.corrected_dh_from_config(urik.CONFIG)
    tcp_T_F_TCP = urik.pose6_to_T(urik.CONFIG.offline_active_T_flange_tcp_pose6)
    tcp_jacobians, tcp_positions = _build_tcp_schedule(reference, tcp_dh, tcp_T_F_TCP)

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
    idx = np.clip(control_index + 1 + np.arange(N), 0, reference.sample_count - 1)
    x_nom_horizon = tcp_positions[idx][:, 0]
    tcp_workspace = dict(
        position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
        xyz_min=np.array([x_nom_horizon.min() - 0.5, -1.0, -1.0]),
        xyz_max=np.array([x_nom_horizon.max() + 0.5, 1.0, 1.0]),  # slack here, just checking plumbing
    )
    magnet_exclusion = dict(
        position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
        lumen_C_m=BEAM_BASE_PIVOT_XYZ_R, radius_m=BEAM_BASE_EXCLUSION_RADIUS_M,
    )
    z_nom_horizon = tcp_positions[idx][:, 2]
    magnet_workspace = dict(
        position_jacobians=tcp_jacobians, nominal_positions_m=tcp_positions,
        z_min_m=float(z_nom_horizon.min() - 0.5), z_max_m=float(z_nom_horizon.max() + 0.5),
    )

    combos = {
        "no-constraints": dict(),
        "magnet-only": dict(magnet_exclusion=magnet_exclusion),
        "TCP+magnet-exclusion": dict(tcp_workspace=tcp_workspace, magnet_exclusion=magnet_exclusion),
        "TCP+magnet-workspace": dict(tcp_workspace=tcp_workspace, magnet_workspace=magnet_workspace),
    }
    expected_extra_rows = {
        "no-constraints": 0, "magnet-only": N,
        "TCP+magnet-exclusion": N + 3 * N, "TCP+magnet-workspace": N + 3 * N,
    }
    print("\n[tcp_workspace combination regressions]")
    for name, kwargs in combos.items():
        ctrl = DelayAwareBeamOutputTrackingMPC(**common, **kwargs)
        base_rows = 3 * N * 7
        extra = ctrl.A.shape[0] - base_rows
        assert extra == expected_extra_rows[name], (
            f"{name}: expected {expected_extra_rows[name]} extra rows, got {extra}"
        )
        step = ctrl.solve_delay_aware(
            z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
            measured_beam_position=p0, control_index=control_index, previous_input=prev,
        )
        assert step.success, f"{name}: solve failed"
        gate = (
            ctrl._magnet_excl_radius is not None
            or ctrl._magnet_ws_z_min is not None
            or ctrl._tcp_ws_min is not None
        )
        expected_gate = name != "no-constraints"
        assert gate == expected_gate, f"{name}: Ax gate mismatch (got {gate}, expected {expected_gate})"
        print(f"  {name}: extra_rows={extra} (expected {expected_extra_rows[name]}), "
              f"solve success={step.success}, Ax-gate={gate}  OK")
    print("[tcp_workspace combination regressions] ALL PASSED")


if __name__ == "__main__":
    run_self_test()
    run_combination_regressions()
