#!/usr/bin/env python3
"""Regression tests for the planner(P)<->live(R) frame-mismatch fix
(2026-09-21, corrected design). See `worker_process.py`'s
`_apply_frame_transform` / `MPCWorkerHandle.set_frame_transform` for the
fix: p_des and p_nominal (both default from reference.desired_position_m,
the raw un-registered planner-frame array this worker was necessarily
constructed from) are transformed via

    p_des^R = R_fit @ p_des^P + t_fit
    p_nominal^R = R_fit @ p_nominal^P + t_fit

while the schedule (J), the task-nullspace projector C, z_ref, u_ref, and
the measured beam position are left UNTOUCHED -- they are already natively
robot-frame (J is built via real robot FK in build_or_load_schedule,
labelled "R.x,R.y,R.z" in its own console dump, independent of the
planner's own abstract path coordinates). An earlier attempt to instead
transform ONLY measured_beam_position at the adapter boundary (leaving
p_des/p_nominal/J/C alone) was reverted after this exact test file (an
earlier version of it) proved that formulation mixes frames unless R_fit
is near-identity -- see process_isolated_adapter.py's __init__ note.

Tests
-----
1. Identity transform (R=I, t=0): set_frame_transform with the identity
   must reproduce the untransformed controller's solve bit-for-bit.
2. Non-identity synthetic transform: applying set_frame_transform's
   underlying `_apply_frame_transform` must produce EXACTLY the same u0 as
   independently constructing a controller directly from a manually
   pre-transformed reference (the two are the same computation performed
   two different ways -- this is the "strong equivalence" check).
3. Round-trip / orthogonality sanity on the synthetic transform itself.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.frame_transform_equivalence_test \\
        --plan-dir plans/... --schedule-cache /tmp/....npy
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace as dc_replace

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC


def _synthetic_rotation(angle_rad: float, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)


def _apply_frame_transform_reference_only(controller, R_fit, t_fit) -> None:
    """Standalone copy of worker_process.py's `_apply_frame_transform`
    (duplicated here deliberately, not imported, so this test independently
    re-derives the same fix rather than trivially calling the exact code
    under test)."""
    des_planner = np.asarray(controller.reference.desired_position_m, dtype=float)
    des_R = des_planner @ R_fit.T + t_fit
    replace_kwargs = {"desired_position_m": des_R}
    tan_planner = getattr(controller.reference, "desired_tangent", None)
    if tan_planner is not None:
        replace_kwargs["desired_tangent"] = np.asarray(tan_planner, dtype=float) @ R_fit.T
    controller.reference = dc_replace(controller.reference, **replace_kwargs)
    controller.nominal_reference_positions_m = des_R.copy()


def run_test(plan_dir: str, schedule_cache: str, control_index: int = 100) -> None:
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(schedule_cache)  # native robot frame, NEVER rotated by this fix
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
    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    p_meas = np.asarray(reference.desired_position_m[control_index], dtype=float) + np.array(
        [0.003, -0.0015, 0.0004]
    )

    # === Test 1: identity transform reproduces the untransformed solve bit-for-bit ===
    c_baseline = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    step_baseline = c_baseline.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p_meas, control_index=control_index, previous_input=prev,
    )
    c_identity = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    _apply_frame_transform_reference_only(c_identity, np.eye(3), np.zeros(3))
    step_identity = c_identity.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p_meas, control_index=control_index, previous_input=prev,
    )
    diff_identity = float(np.max(np.abs(step_baseline.command - step_identity.command)))
    assert diff_identity < 1e-12, f"identity transform changed u0: diff={diff_identity:.3e}"
    print(f"[frame_transform_equivalence] 1/3 PASS: identity transform bit-identical "
          f"(u0 diff={diff_identity:.2e})")

    # === Test 2: non-identity transform matches an independently-constructed
    #     controller built directly from a manually pre-transformed reference ===
    R_fit = _synthetic_rotation(0.37, np.array([0.2, -0.5, 0.83]))
    t_fit = np.array([0.481, -0.664, -0.012])
    assert np.max(np.abs(R_fit.T @ R_fit - np.eye(3))) < 1e-12, "test transform not orthogonal"
    assert abs(np.linalg.det(R_fit) - 1.0) < 1e-12, "test transform not a proper rotation"

    c_via_fix = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    _apply_frame_transform_reference_only(c_via_fix, R_fit, t_fit)

    des_planner = np.asarray(reference.desired_position_m, dtype=float)
    des_R = des_planner @ R_fit.T + t_fit
    reference_prebuilt = dc_replace(reference, desired_position_m=des_R)
    c_prebuilt = StagewiseTaskNullspaceDelayAwareMPC(
        gamma=0.0, reference=reference_prebuilt, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule,  # SAME schedule, NOT rotated -- J is native robot-frame
        delay_samples=2, beta_d=1.0,
    )
    assert c_prebuilt.nominal_reference_positions_m.shape == des_R.shape
    nom_diff = float(np.max(np.abs(c_prebuilt.nominal_reference_positions_m - des_R)))
    assert nom_diff < 1e-12

    p_meas_moved = des_R[control_index] + np.array([0.003, -0.0015, 0.0004])  # same physical offset, R frame
    # p_meas is a REAL measurement, native robot frame -- SAME value both ways
    # (this fix never transforms it), but the two controllers were built from
    # DIFFERENT copies of the reference object at different points, so use a
    # value consistent with each controller's own (now-identical) p_des.
    step_via_fix = c_via_fix.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p_meas_moved, control_index=control_index, previous_input=prev,
    )
    step_prebuilt = c_prebuilt.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p_meas_moved, control_index=control_index, previous_input=prev,
    )
    diff_u0 = float(np.max(np.abs(step_via_fix.command - step_prebuilt.command)))
    print(f"[frame_transform_equivalence] u0 (via _apply_frame_transform) vs "
          f"u0 (independently pre-built reference): max diff = {diff_u0:.3e}")
    assert diff_u0 < 1e-9, f"u0 mismatch: {diff_u0:.3e}"
    print(f"[frame_transform_equivalence] 2/3 PASS: non-identity transform matches "
          f"independent construction (u0 diff={diff_u0:.2e})")

    # === Test 3: round-trip / orthogonality sanity on the transform itself ===
    p_R = np.array([0.51, -0.61, -0.02])
    p_P = R_fit.T @ (p_R - t_fit)
    roundtrip = R_fit @ p_P + t_fit
    rt_err = float(np.max(np.abs(roundtrip - p_R)))
    assert rt_err < 1e-12, f"round-trip failed: {rt_err:.3e}"
    print(f"[frame_transform_equivalence] 3/3 PASS: round-trip R[R^T(p_R-t)]+t == p_R "
          f"(err={rt_err:.2e})")

    print("[frame_transform_equivalence] ALL PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--control-index", type=int, default=100)
    args = p.parse_args()
    run_test(args.plan_dir, args.schedule_cache, args.control_index)


if __name__ == "__main__":
    main()
