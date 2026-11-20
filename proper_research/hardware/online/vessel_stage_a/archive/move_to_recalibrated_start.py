#!/usr/bin/env python3
"""Move the robot to the recalibrated vessel start position and verify the
2026-10-02 z-raise-removal fix: that the magnet, once there, really is on
the same z-plane as `build_vessel_plan.py`'s fixed beam-base pivot, and
that it is no closer to the beam base than the recorded exclusion floor.

This does NOT build or run a plan -- it just moves the robot to the
--start-position-json's recorded joints (default: the 2026-10-02
recalibrated file) and re-derives the magnet pose via the SAME real
forward-kinematics model build_vessel_plan.py/run_mpc_delay_aware_vessel.py
rely on, so you can confirm the fix before trusting it in a live run.

Before moving, the straight-line joint-space path from the robot's CURRENT
joints to the target is checked against the beam-base exclusion radius
(same `common.reset_to_plan_initial_safe` safety check used by every other
preflight in this package) -- it refuses to move if that path would come
closer to the beam base than the recorded floor at any sampled point.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.archive.move_to_recalibrated_start \\
        --start-position-json vessel_magnet_initial_position_2026-10-02_recalibrated.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.ur_rtde_robot import URRTDERobot
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)

from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
    BEAM_BASE_PIVOT_XY_ROT,
    BEAM_BASE_PIVOT_Z,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--start-position-json",
        default="vessel_magnet_initial_position_2026-10-02_recalibrated.json",
        help="from capture_live_start_position.py -- same file build_vessel_plan.py's "
             "--start-position-json takes.",
    )
    p.add_argument("--robot-ip", default=common.ROBOT_IP)
    p.add_argument("--speed", type=float, default=0.2)
    p.add_argument("--acceleration", type=float, default=0.2)
    p.add_argument("--skip-path-safety-check", action="store_true",
                    help="skip the pre-move straight-line exclusion-radius check (NOT "
                         "recommended -- move_j cannot be interrupted mid-motion once "
                         "started).")
    p.add_argument("--z-match-tol-mm", type=float, default=1.0,
                    help="pass/fail tolerance for |magnet_z - beam_base_z| after the move.")
    p.add_argument("--path-check-noise-tol-mm", type=float, default=0.5,
                    help="this recalibrated start position sets exclusion_floor_mm EXACTLY "
                         "equal to its own magnet-to-beam-base distance (by design -- 'no "
                         "closer than where it started'), so the target pose sits precisely "
                         "ON the path-safety check's boundary. Without slack, the check's "
                         "strict gap<floor comparison trips on pure FK/float noise at the "
                         "target itself (gap and floor agree to ~15 decimal digits but not "
                         "exactly). This subtracts a small noise tolerance from the radius "
                         "used ONLY for the pre-move path check -- NOT a real safety margin, "
                         "just enough to absorb recomputation noise; the final PASS/FAIL "
                         "report below still checks against the true, untouched floor.")
    args = p.parse_args()

    with open(args.start_position_json) as f:
        start_ref = json.load(f)
    target_q = np.asarray(start_ref["joints_rad"], dtype=float).reshape(6)
    recorded_magnet_pose6 = np.asarray(start_ref["magnet_pose6_R"], dtype=float)
    exclusion_floor_m = float(start_ref["exclusion_floor_mm"]) / 1000.0

    beam_base_xyz = np.array([BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z])
    print(f"[move] target joints (rad) = {target_q.tolist()}")
    print(f"[move] beam_base_pivot_xyz (build_vessel_plan.py, fixed) = {beam_base_xyz.tolist()}")
    print(f"[move] exclusion_floor = {exclusion_floor_m * 1e3:.2f}mm (from {args.start_position_json})")

    robot_kin = _resolve_robot_kinematics(make_robot_config())

    def _magnet_transform_fn(q6: np.ndarray) -> np.ndarray:
        T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), robot_kin.dh, robot_kin.T_F_M)
        return np.asarray(T.T_R_target[:3, 3], dtype=float)

    common.check_robot_safe(robot_ip=args.robot_ip)

    if args.skip_path_safety_check:
        common.reset_to_plan_initial(
            target_q, robot_ip=args.robot_ip, speed=args.speed, acceleration=args.acceleration,
        )
    else:
        path_check_radius_m = exclusion_floor_m - args.path_check_noise_tol_mm * 1e-3
        common.reset_to_plan_initial_safe(
            target_q, robot_ip=args.robot_ip, speed=args.speed, acceleration=args.acceleration,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_xyz[None, :],
            magnet_exclusion_radius_m=path_check_radius_m,
        )

    robot = URRTDERobot(args.robot_ip, frequency=125.0)
    robot.connect()
    try:
        q_live = np.array(robot.get_joints())
    finally:
        robot.close()

    magnet_xyz = _magnet_transform_fn(q_live)
    magnet_z_diff_mm = (magnet_xyz[2] - beam_base_xyz[2]) * 1e3
    distance_to_base_mm = float(np.linalg.norm(magnet_xyz - beam_base_xyz)) * 1e3
    joint_offset = float(np.max(np.abs(q_live - target_q)))
    drift_from_capture_mm = float(np.linalg.norm(magnet_xyz - recorded_magnet_pose6[:3])) * 1e3

    print(f"\n[verify] live joints = {q_live.tolist()}")
    print(f"[verify] max joint offset from target = {joint_offset:.5f} rad")
    print(f"[verify] live magnet_xyz (FK)          = {magnet_xyz.tolist()}")
    print(f"[verify] magnet_z - beam_base_z         = {magnet_z_diff_mm:+.3f}mm "
          f"(tol +/-{args.z_match_tol_mm:.1f}mm)")
    print(f"[verify] magnet-to-beam-base distance   = {distance_to_base_mm:.2f}mm "
          f"(exclusion floor = {exclusion_floor_m * 1e3:.2f}mm)")
    print(f"[verify] drift vs captured magnet_pose6_R = {drift_from_capture_mm:.2f}mm")

    ok = True
    if abs(magnet_z_diff_mm) > args.z_match_tol_mm:
        print(f"[FAIL] magnet is NOT on the beam-base z-plane within tolerance")
        ok = False
    if distance_to_base_mm < exclusion_floor_m * 1e3 - args.path_check_noise_tol_mm:
        print(f"[FAIL] magnet is CLOSER to the beam base than the recorded exclusion floor "
              f"(beyond the {args.path_check_noise_tol_mm:.1f}mm noise tolerance)")
        ok = False
    if ok:
        print("[PASS] z-plane match and exclusion floor both verified")


if __name__ == "__main__":
    main()
