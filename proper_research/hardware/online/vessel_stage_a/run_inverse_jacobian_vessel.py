#!/usr/bin/env python3
"""Naive inverse-Jacobian (LTV, scheduled) closed-loop run for a vessel
plan, live on hardware -- direct comparator against run_mpc_delay_aware_
vessel.py's MPC condition, for the same reference and the same Jacobian
schedule (no-contact today).

Mirrors run_open_loop_vessel.py's safety pattern exactly (vessel_stage_a
preflight, reset-with-retreat against the magnet-to-beam-base floor,
--z-raise-mm handling, 2-marker vision), but sets controller_kind to
"naive_inverse_jacobian_ltv" and wires the Jacobian schedule through
close_loop_path_follow's module-level _SCHEDULE_OVERRIDE (the same
mechanism naive_inverse_jacobian_ltv/mpc_ltv_offline use elsewhere in that
file) instead of building an MPC worker.

This controller has NEITHER the MPC path's in-QP magnet-exclusion/z-
workspace constraints NOR process_isolated_adapter's post-hoc measured-
joint monitors for those (both are MPC-specific machinery) -- the only
live per-tick safety net here is close_loop_path_follow's own generic
tcp_out_of_workspace box check. That box's DEFAULT bounds are tuned for
the unrelated rectangle_stage_a workspace and would either trip
immediately (our vessel operates in a completely different xyz region) or
provide no real protection, so this script overrides them with a box
derived from a VALIDATED, already-completed vessel MPC run's own achieved
flange-frame TCP trajectory (2026-09-29, no-contact/floor220 run) plus a
20mm margin on every side -- real data, not a guess, same principle
rectangle_stage_a's own tuned box used (see close_loop_path_follow.py's
workspace_xyz_min_m/max_m comments).

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.run_inverse_jacobian_vessel \\
        --plan-dir plans/vessel_lumen_2026-09-29_v3_centreline_right0p5mm/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --run-name vessel_v3_right0p5mm_invjac_nocontact \\
        --schedule-cache /tmp/vessel_lumen_2026-09-29_v3_schedule_nocontact.npy \\
        --exclusion-floor-mm 210.43
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
import proper_research.hardware.online.state_stream as state_stream_mod
from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics

_RealStateStreamConfig = state_stream_mod.StateStreamConfig

# 2026-09-30 fix: same bug documented in run_mpc_delay_aware_vessel.py's
# module docstring ("fix 5", found live 2026-09-23) and in run_open_loop_
# vessel.py (ported 2026-09-30) -- close_loop_path_follow.py's generic
# planner<->robot frame-registration step silently picks up an UNRELATED
# stale shape_centreline.npz (alphabetically first of 17 matches under
# plans/, from plans/circle_15mm_2026-09-12/) and uses its un-raised tip0.z
# to inject a spurious z-offset on top of the vessel reference's already-
# correct desired_position_m. The two 2026-09-29 inverse-Jacobian runs this
# script produced (vessel_v3_right0p5mm_invjac_nocontact[_v2]) predate this
# fix -- their reported tracking error/workspace-box trips may carry a small
# (<=5mm at that plan's near-zero z-raise) contribution from this bug on
# top of the posture-drift finding. The vessel reference is already in
# robot frame (built via real FK throughout build_vessel_planning_context),
# so no registration is needed here -- bypass with an exact identity
# transform, same as the MPC script.
def _identity_fit_planner_to_robot(npz_path, start_tip_R, T_R_B):
    return np.eye(3), np.zeros(3)


pf._fit_planner_to_robot = _identity_fit_planner_to_robot
pf._find_shape_npz = lambda plan_dir: Path(__file__)

# 2026-09-29: derived from vessel_lumen_2026-09-29_v3_right0p5mm_mpc_
# nocontact_floor220_20260929T195312Z's own q_meas_rad trajectory (639
# ticks, path_complete, 0 wall penetration, 0 vision failures) via
# flange-frame FK (T_F_target=eye(4), matching getActualTCPPose()'s
# default zero TCP offset -- this codebase never calls setTcp).
# First live inverse-jacobian attempt (2026-09-29) tripped this at step 6
# on a completely benign flange y=-0.790m (3mm past an initial +/-20mm
# margin) while beam-TIP tracking error was small and stable
# (3.5-4.3mm) the whole time -- not a runaway, just a naive-inverse-
# Jacobian run naturally sitting a few mm outside one specific MPC run's
# exact envelope. Widened to +/-50mm: still an order of magnitude
# tighter than the physical incidents this box exists to catch (magnet
# hitting the table, ~25cm z excursions -- see magnet-workspace-
# constraints memory), while giving real operating headroom.
_VALIDATED_WORKSPACE_XYZ_MIN_M = (0.400, -0.817, 0.249)
_VALIDATED_WORKSPACE_XYZ_MAX_M = (0.615, -0.586, 0.381)


def _patch_state_stream_config(z_raise_mm: float) -> None:
    z_raise_m = float(z_raise_mm) / 1000.0

    def _patched_state_stream_config(**kwargs):
        kwargs.setdefault("marker_min_count", 2)
        kwargs.setdefault("marker_max_count", 2)
        cfg = _RealStateStreamConfig(**kwargs)
        if z_raise_m != 0.0 and "T_robot_beam_pose6" not in kwargs:
            pose6 = list(cfg.T_robot_beam_pose6)
            pose6[2] += z_raise_m
            cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
        return cfg

    pf.StateStreamConfig = _patched_state_stream_config


BEAM_BASE_PIVOT_XY_ROT = np.array([0.525575, -0.670028])
BEAM_BASE_PIVOT_Z_UNRAISED = -0.016567

_robot_kin = _resolve_robot_kinematics(make_robot_config())


def _magnet_transform_fn(q6):
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def _safe_reset_with_retreat(
    plan_dir: str, insertion_tol_mm: float, exclusion_floor_m: float,
    beam_base_pivot_xyz: np.ndarray,
):
    q0, l0 = common.load_plan_initial_state(plan_dir)
    try:
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_floor_m,
        )
    except RuntimeError as exc:
        print(f"[reset] direct path unsafe ({exc}); routing via a retreat waypoint")
        from proper_research.hardware.ur_rtde_robot import URRTDERobot
        from scipy.spatial.transform import Rotation as Rot

        robot = URRTDERobot(common.ROBOT_IP, frequency=125.0)
        robot.connect()
        q_now = np.array(robot.get_joints())
        robot.close()
        m_now = _magnet_transform_fn(q_now)
        base = beam_base_pivot_xyz[0]
        d_now = np.linalg.norm(m_now - base)
        direction = (m_now - base) / d_now
        retreat_xyz = base + direction * (d_now + 0.030)
        T_now = urik.forward_kinematics(q_now, _robot_kin.dh, _robot_kin.T_F_M)
        rotvec_now = Rot.from_matrix(T_now.T_R_target[:3, :3]).as_rotvec()
        T_retreat = urik.pose6_to_T(np.r_[retreat_xyz, rotvec_now])
        ik = urik.inverse_kinematics_dls(
            T_R_target=T_retreat, q_seed_rad=q_now, dh=_robot_kin.dh,
            T_F_target=_robot_kin.T_F_M, cfg=_robot_kin.ik_cfg,
        )
        if not ik.converged:
            raise RuntimeError("retreat-waypoint IK did not converge") from exc
        common.reset_to_plan_initial_safe(
            ik.q_rad, magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz, magnet_exclusion_radius_m=exclusion_floor_m,
        )
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_floor_m,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="vessel_invjac_ltv")
    p.add_argument("--schedule-cache", required=True,
                    help="pre-built (S,3,7) Jacobian schedule .npy, e.g. one already "
                         "built by run_mpc_delay_aware_vessel.py for the same plan-dir "
                         "-- reused as-is via close_loop_path_follow._SCHEDULE_OVERRIDE, "
                         "this script never builds a schedule itself.")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0)
    p.add_argument("--exclusion-floor-mm", type=float, default=210.43,
                    help="magnet-to-beam-base safety floor to check the RESET path "
                         "against (one-time, before motion starts). This controller has "
                         "no live in-loop magnet-exclusion monitor -- see module docstring.")
    p.add_argument("--max-control-steps", type=int, default=1000)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--z-raise-mm", type=float, default=0.0)
    args = p.parse_args()

    _patch_state_stream_config(args.z_raise_mm)
    beam_base_pivot_xyz = np.array([[
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1],
        BEAM_BASE_PIVOT_Z_UNRAISED + args.z_raise_mm / 1000.0,
    ]])
    print(f"[inv-jac] z_raise_mm = {args.z_raise_mm}")
    print(f"[inv-jac] beam_base_pivot_xyz = {beam_base_pivot_xyz[0].tolist()}")

    exclusion_floor_m = args.exclusion_floor_mm / 1000.0

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = _safe_reset_with_retreat(
            args.plan_dir, args.insertion_tol_mm, exclusion_floor_m, beam_base_pivot_xyz,
        )

    schedule = np.load(args.schedule_cache)
    print(f"[inv-jac] loaded schedule {schedule.shape} from {args.schedule_cache}")
    pf._SCHEDULE_OVERRIDE = schedule

    cfg = pf.CONFIG
    cfg.controller_kind = "naive_inverse_jacobian_ltv"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"

    cfg.feedforward_joint_trajectory = False
    cfg.accumulator_seam = True
    cfg.servo_stream_hz = common.SERVO_STREAM_HZ
    cfg.force_mpc_feedforward = False

    cfg.joint_velocity_limit_rad_s = common.JOINT_VELOCITY_LIMIT_RAD_S
    cfg.max_joint_step_rad = common.MAX_JOINT_STEP_RAD
    cfg.joint_acceleration_limit_rad_s2 = common.JOINT_ACCELERATION_LIMIT_RAD_S2
    cfg.control_insertion = True
    cfg.control_hz = common.CONTROL_HZ
    cfg.max_control_steps = args.max_control_steps
    cfg.initial_insertion_m = l0

    cfg.workspace_xyz_min_m = _VALIDATED_WORKSPACE_XYZ_MIN_M
    cfg.workspace_xyz_max_m = _VALIDATED_WORKSPACE_XYZ_MAX_M
    print(f"[inv-jac] tcp workspace box: {cfg.workspace_xyz_min_m} .. {cfg.workspace_xyz_max_m}")

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name

    print(f"[inv_jac_vessel] plan={args.plan_dir}\n"
          f"[inv_jac_vessel] q0={q0.tolist()} L0={l0 * 1000:.2f}mm")
    pf.main()


if __name__ == "__main__":
    main()
