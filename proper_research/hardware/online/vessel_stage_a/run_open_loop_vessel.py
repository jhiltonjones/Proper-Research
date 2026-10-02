#!/usr/bin/env python3
"""Open-loop (feedforward-only) sanity check for a vessel plan, live on
hardware. Run this FIRST on any new vessel plan, before any closed-loop
controller -- it's the cheap gate that confirms the plan tracks reasonably
before committing to anything more expensive.

Mirrors rectangle_stage_a/run_open_loop_c.py's execution-layer-C config
exactly, but uses vessel_stage_a.common for preflight -- NOT
rectangle_stage_a.common -- so the reset:
  (a) reads q0/L0 directly from the plan's own saved state (not from
      make_initial_poses(), which vessel plans override), and
  (b) validates the straight-line joint-space reset path against the
      magnet-to-beam-base safety floor before any motion, routing through
      a retreat waypoint automatically if the direct path would violate it
      (routine given how tight that floor is, not a fault).

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel \\
        --plan-dir plans/vessel_lumen_2026-10-02_left1p5mm_insmax80/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --run-name vessel_lumen_2026-10-02_left1p5mm_openloop
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
import proper_research.hardware.online.state_stream as state_stream_mod
from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
    BEAM_BASE_PIVOT_XY_ROT, BEAM_BASE_PIVOT_Z,
)
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics

_RealStateStreamConfig = state_stream_mod.StateStreamConfig

# 2026-09-30 fix: same bug run_mpc_delay_aware_vessel.py's module docstring
# documents as "fix 5" (found live 2026-09-23) -- close_loop_path_follow.py's
# `_load_plan_reference` calls `_find_shape_npz(plan_dir)`, which searches
# plan_dir's ancestry for ANY file named shape_centreline.npz (a mechanism
# built for plan_shape_path.py's free-space shapes, which genuinely need a
# live rigid-registration fit). Vessel plans never produce their own such
# file, so the search silently picks up an UNRELATED stale file from
# plans/circle_15mm_2026-09-12/ (alphabetically first of 17 matches under
# plans/) and uses ITS un-raised tip0.z to compute a spurious z-registration
# offset on top of the vessel reference's already-correct desired_position_m
# -- invisible-small (<=5mm) for earlier near-zero-raise plans, but exactly
# equal to the z-raise amount (42mm, confirmed live 2026-09-30) once a real
# raise is in play. The vessel reference is already in robot frame (built
# via real FK throughout build_vessel_planning_context), so no registration
# is needed here -- bypass with an exact identity transform, same as the
# MPC script.
def _identity_fit_planner_to_robot(npz_path, start_tip_R, T_R_B):
    return np.eye(3), np.zeros(3)


pf._fit_planner_to_robot = _identity_fit_planner_to_robot
pf._find_shape_npz = lambda plan_dir: Path(__file__)


def _patch_state_stream_config() -> None:
    """Patch pf.StateStreamConfig with (a) T_robot_beam_pose6's z set to the
    recalibrated beam-base height (BEAM_BASE_PIVOT_Z, imported from
    build_vessel_plan.py -- same fixed constant the plan was built against)
    and (b) marker_min_count/marker_max_count defaulted to 2 -- see
    run_mpc_delay_aware_vessel.py's identical 2026-09-29 fix for why (the
    physical rig's middle marker was permanently removed; tip position
    tracking is unaffected, only the now-unused chord tangent is). Only the
    DEFAULT changes; explicit kwargs still win.

    2026-10-02 fix: removed the --z-raise-mm offset-from-a-moving-target
    scheme entirely (same change as build_vessel_plan.py's and
    run_mpc_delay_aware_vessel.py's own 2026-10-02 fixes) -- sets the pose
    directly to the fixed, recalibrated absolute height instead."""

    def _patched_state_stream_config(**kwargs):
        kwargs.setdefault("marker_min_count", 2)
        kwargs.setdefault("marker_max_count", 2)
        cfg = _RealStateStreamConfig(**kwargs)
        if "T_robot_beam_pose6" not in kwargs:
            pose6 = list(cfg.T_robot_beam_pose6)
            pose6[2] = BEAM_BASE_PIVOT_Z
            cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
        return cfg

    pf.StateStreamConfig = _patched_state_stream_config


_robot_kin = _resolve_robot_kinematics(make_robot_config())


def _magnet_transform_fn(q6):
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def _safe_reset_with_retreat(
    plan_dir: str, insertion_tol_mm: float, exclusion_floor_m: float,
    beam_base_pivot_xyz: np.ndarray, path_check_noise_tol_m: float = 0.0005,
):
    """common.preflight, but if the direct straight-line reset path is
    unsafe, retry via a retreat waypoint (further from the beam base)
    before giving up -- the tight floor makes a direct-path refusal
    routine, not exceptional.

    2026-10-02 fix: a start position whose exclusion_floor_m is set EXACTLY
    equal to its own plan-initial magnet-to-beam-base distance (as the
    2026-10-02 recalibrated position deliberately is -- "no closer than
    where it started") puts the plan's initial joints precisely ON the
    path-safety check's boundary. Retreating first does not help in this
    case: the final leg of the retry still resets INTO that same
    exactly-on-the-boundary target, so the identical gap<floor check trips
    again on pure FK/float noise (confirmed live: "101.8mm < 101.8mm"),
    propagating out of the `except` block uncaught. `path_check_noise_tol_m`
    subtracts a small noise tolerance from the radius used for ALL path
    checks in this function -- NOT a real safety margin, just enough to
    absorb recomputation noise at an intentionally-zero-slack target; the
    actual exclusion_floor_m passed in is otherwise untouched."""
    path_check_radius_m = exclusion_floor_m - path_check_noise_tol_m
    q0, l0 = common.load_plan_initial_state(plan_dir)
    try:
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=path_check_radius_m,
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
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz, magnet_exclusion_radius_m=path_check_radius_m,
        )
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=path_check_radius_m,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="vessel_openloop")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0)
    p.add_argument("--exclusion-floor-mm", type=float, default=101.80230875991425,
                    help="magnet-to-beam-base safety floor to check the reset path against -- "
                         "default matches vessel_magnet_initial_position_2026-10-02_"
                         "recalibrated.json's exclusion_floor_mm; override to match whatever "
                         "start-position JSON the plan you're running was actually built from.")
    p.add_argument("--max-control-steps", type=int, default=1000)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--path-check-noise-tol-mm", type=float, default=0.5,
                    help="see _safe_reset_with_retreat's docstring -- a start position whose "
                         "exclusion floor equals its own plan-initial distance (as the "
                         "2026-10-02 recalibrated position is) sits exactly on the path-safety "
                         "check's boundary, tripping on pure FK/float noise. Subtracts this "
                         "from the radius used for the path check ONLY -- not a real safety "
                         "margin.")
    args = p.parse_args()

    _patch_state_stream_config()
    beam_base_pivot_xyz = np.array([[
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z,
    ]])
    print(f"[open-loop] beam_base_pivot_xyz = {beam_base_pivot_xyz[0].tolist()} (recalibrated, fixed)")

    exclusion_floor_m = args.exclusion_floor_mm / 1000.0

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = _safe_reset_with_retreat(
            args.plan_dir, args.insertion_tol_mm, exclusion_floor_m, beam_base_pivot_xyz,
            path_check_noise_tol_m=args.path_check_noise_tol_mm * 1e-3,
        )

    cfg = pf.CONFIG
    cfg.controller_kind = "open_loop_ff"
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

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name

    print(f"[open_loop_vessel] plan={args.plan_dir}\n"
          f"[open_loop_vessel] q0={q0.tolist()} L0={l0 * 1000:.2f}mm\n"
          f"[open_loop_vessel] servo_stream_hz={cfg.servo_stream_hz} "
          f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
          f"max_joint_step_rad={cfg.max_joint_step_rad}")
    pf.main()


if __name__ == "__main__":
    main()
