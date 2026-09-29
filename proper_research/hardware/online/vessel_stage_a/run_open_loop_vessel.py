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
        --plan-dir plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --run-name vessel_live_trimmed6mm_openloop \\
        --exclusion-floor-mm 210.43
"""
from __future__ import annotations

import argparse

import numpy as np

from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics

# 2026-09-29 fix: this used to hardcode the +30mm raised-workspace pivot
# (z=0.013433) and unconditionally call zraise_patch.apply(30.0) at import
# time, regardless of which plan was passed in -- silently checking the
# reset path (and, more seriously, interpreting live camera detections)
# against a raised pivot/vision-plane even for an UNRAISED plan, a 30mm
# frame mismatch. Now parameterized via --z-raise-mm (default 0.0,
# matching build_vessel_plan.py's own default and this project's own
# zraise_patch docstring: "0.0 / omit for an unraised (original-height)
# plan"); pass 30 explicitly to reproduce the old raised-workspace
# behaviour used by e.g. plans/vessel_live_trimmed6mm_2026-09-28.
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
    """common.preflight, but if the direct straight-line reset path is
    unsafe, retry via a retreat waypoint (further from the beam base)
    before giving up -- the tight floor makes a direct-path refusal
    routine, not exceptional."""
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
    p.add_argument("--run-name", default="vessel_openloop")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0)
    p.add_argument("--exclusion-floor-mm", type=float, default=210.43,
                    help="magnet-to-beam-base safety floor to check the reset path against")
    p.add_argument("--max-control-steps", type=int, default=1000)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--z-raise-mm", type=float, default=0.0,
                    help="rigid z-shift applied to the beam-base pivot AND the live "
                         "vision reconstruction plane -- 0.0 (default) for an "
                         "unraised/original-height workspace, 30.0 to reproduce the "
                         "2026-09-27/28 raised-workspace runs. MUST match whether the "
                         "plan (and the lumen it was built from) was itself built "
                         "raised or unraised -- a mismatch here silently misreads "
                         "real camera detections by the z-raise amount.")
    args = p.parse_args()

    zraise_patch.apply(args.z_raise_mm)
    beam_base_pivot_xyz = np.array([[
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1],
        BEAM_BASE_PIVOT_Z_UNRAISED + args.z_raise_mm / 1000.0,
    ]])
    print(f"[open-loop] z_raise_mm = {args.z_raise_mm}")
    print(f"[open-loop] beam_base_pivot_xyz = {beam_base_pivot_xyz[0].tolist()}")

    exclusion_floor_m = args.exclusion_floor_mm / 1000.0

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = _safe_reset_with_retreat(
            args.plan_dir, args.insertion_tol_mm, exclusion_floor_m, beam_base_pivot_xyz,
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
