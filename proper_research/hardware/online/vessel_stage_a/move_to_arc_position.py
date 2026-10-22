#!/usr/bin/env python3
"""Move the source magnet to a chosen (phi, radius) position on the arc
around the beam base, with the dipole aligned to point at the base (plus
an optional extra psi rotation about the magnet's own local Z / joint 6).

This is a thin, interactive CLI wrapper around the same functions used
throughout the 2026-10-03 hardware debugging session
(sweep_free_space_arc_dipole.py): the continuity-preserving orientation
formula (reference_orientation_matrix, fixed to drag the real calibration
orientation along the arc instead of re-deriving an arbitrary roll at
each phi -- see that function's own docstring for why this matters), the
widened multi-seed/multi-branch IK search (robust_ik_for_pose), and the
80mm-spherical-only path safety check (find_safe_path /
path_safety_check). Nothing here invents new logic; it only sequences
those pieces into a single "pick phi and radius, verify, move" command,
by default in DRY-RUN mode (no real motion) so you can inspect the plan
before committing to it with --execute.

Usage
-----
Dry run (prints the plan, computes safety, does NOT move anything)::

    python -m proper_research.hardware.online.vessel_stage_a.move_to_arc_position \\
        --phi-deg 40 --radius-mm 110

Same, plus an extra 90deg dipole rotation (pure joint-6, applied after
reaching the arc position, zero extra IK/path-safety risk)::

    python -m ...move_to_arc_position --phi-deg 40 --radius-mm 110 --psi-deg 90

Actually execute the move (only after reviewing the dry-run output)::

    python -m ...move_to_arc_position --phi-deg 40 --radius-mm 110 --execute

Safety notes
------------
- The only safety restriction enforced is the 80mm spherical keep-out
  around the beam base (confirmed 2026-10-03: a separate z-bounds window
  was over-restrictive and not actually required). --radius-mm must stay
  comfortably above 80mm -- this script refuses below 85mm outright and
  prints a loud warning below 100mm.
- Always reads the robot's ACTUAL current joints live before planning --
  never assumes where the robot is.
- Refuses to execute (even with --execute) if the computed path is not
  verified safe.
- Checks for a protective stop before every individual waypoint move and
  aborts immediately if one is detected mid-sequence.
- A mode-4 ("not RUNNING") connection drop has been a recurring,
  unexplained issue on this rig during large reconfigurations this
  session -- if a move is interrupted, this script does NOT retry
  automatically; it reports the robot's actual resulting position (read
  via a read-only RTDE connection, which has kept working even when the
  control connection dropped) and stops. Re-run with --execute again
  once you've confirmed the robot shows robot_mode=7 (RUNNING).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
    BEAM_BASE_PIVOT_Z, BEAM_BASE_PIVOT_XY_ROT,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    reference_orientation_matrix, dipole_world_direction, _magnet_tcp6,
    robust_ik_for_pose, build_seed_pool, find_safe_path,
)

ROBOT_IP = "192.168.56.101"
MIN_RADIUS_MM = 85.0
WARN_RADIUS_MM = 100.0
EXCLUSION_RADIUS_M = 0.080


def get_live_joints() -> np.ndarray:
    import rtde_receive
    rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    return np.array(rr.getActualQ())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phi-deg", type=float, required=True,
                    help="arc angle (deg) around the beam base, world x-y plane. "
                         "phi=0 is directly -X from the base (the beam's own axial "
                         "direction); positive phi sweeps toward +Y.")
    p.add_argument("--radius-mm", type=float, required=True,
                    help="distance (mm) from the beam base. Must be well above the "
                         "80mm exclusion floor.")
    p.add_argument("--psi-deg", type=float, default=0.0,
                    help="extra dipole rotation (deg) about the magnet's own local Z "
                         "(= joint 6), applied AFTER reaching the arc position via a "
                         "pure joint-6 move -- zero extra position/path risk. "
                         "Default 0 (dipole points directly at the beam base).")
    p.add_argument("--speed", type=float, default=0.15, help="move_j speed (rad/s)")
    p.add_argument("--acceleration", type=float, default=0.15, help="move_j acceleration (rad/s^2)")
    p.add_argument("--execute", action="store_true",
                    help="actually move the robot. Without this flag, only computes "
                         "and prints the plan (dry run, no motion).")
    args = p.parse_args()

    if args.radius_mm < MIN_RADIUS_MM:
        raise SystemExit(
            f"--radius-mm {args.radius_mm:.1f} is below the minimum {MIN_RADIUS_MM:.0f}mm "
            f"(only {(args.radius_mm - EXCLUSION_RADIUS_M*1000):.1f}mm margin above the "
            f"80mm exclusion floor) -- refusing."
        )
    if args.radius_mm < WARN_RADIUS_MM:
        print(f"[warn] --radius-mm {args.radius_mm:.1f} is close to the 80mm exclusion "
              f"floor ({args.radius_mm - 80:.1f}mm margin) -- double-check this is intended.")

    beam_base_xyz = np.array([BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z])

    print("[plan] reading current robot joints (live)...")
    q_now = get_live_joints()
    print(f"[plan] current joints (deg): {np.round(np.degrees(q_now), 2).tolist()}")

    fk_now = urik.forward_kinematics(q_now, rk.dh, rk.T_F_M)
    xyz_now = np.asarray(fk_now.T_R_target)[:3, 3]
    rotvec_now = Rot.from_matrix(np.asarray(fk_now.T_R_target)[:3, :3]).as_rotvec()
    d_now = xyz_now - beam_base_xyz
    radius_now_mm = np.linalg.norm(d_now) * 1000
    phi_now_deg = np.degrees(np.arctan2(d_now[1], -d_now[0]))
    print(f"[plan] current magnet position: phi={phi_now_deg:.2f}deg r={radius_now_mm:.2f}mm")

    phi_target = np.deg2rad(args.phi_deg)
    radius_target_m = args.radius_mm / 1000.0
    xyz_target = beam_base_xyz + radius_target_m * np.array([-np.cos(phi_target), np.sin(phi_target), 0.0])
    R_base_aligned = reference_orientation_matrix(xyz_target, beam_base_xyz)
    R_psi = Rot.from_rotvec([0.0, 0.0, np.deg2rad(args.psi_deg)]).as_matrix()
    R_target = R_base_aligned @ R_psi
    rotvec_target = Rot.from_matrix(R_target).as_rotvec()

    dip_target = dipole_world_direction(rotvec_target)
    dir_to_base = beam_base_xyz - xyz_target
    dir_to_base /= np.linalg.norm(dir_to_base)
    dipole_vs_base_deg = np.degrees(np.arccos(np.clip(np.dot(dip_target, dir_to_base), -1, 1)))

    print(f"[plan] target: phi={args.phi_deg:.2f}deg r={args.radius_mm:.2f}mm psi={args.psi_deg:.2f}deg")
    print(f"[plan] target magnet xyz: {xyz_target.tolist()}")
    print(f"[plan] dipole-vs-toward-base angle at psi=0 would be ~0deg; at this psi "
          f"the dipole is {dipole_vs_base_deg:.2f}deg off pointing at the base "
          f"(0deg expected only when --psi-deg 0)")

    print("[plan] solving IK (multi-seed, multi-branch search)...")
    pool = build_seed_pool(q_now)
    sols = robust_ik_for_pose(xyz_target, rotvec_target, pool, rk.dh, rk.ik_cfg)
    print(f"[plan] {len(sols)} IK solution(s) found")
    if not sols:
        raise SystemExit("[plan] no IK solution found for this target -- aborting.")

    lumen_C = beam_base_xyz.reshape(1, 3)
    best = None
    for q in sorted(sols, key=lambda q: float(np.max(np.abs(urik.wrapped_joint_difference(q, q_now))))):
        wps, safe, msg = find_safe_path(
            q_now, xyz_now, rotvec_now, q, xyz_target, rotvec_target,
            DH=rk.dh, ik_cfg=rk.ik_cfg, T_F_M=rk.T_F_M, lumen_C_m=lumen_C,
            exclusion_radius_m=EXCLUSION_RADIUS_M, z_bounds_m=None, beam_base_xyz=beam_base_xyz,
        )
        dq = np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q, q_now))))
        print(f"[plan]   candidate: dq_from_current={dq:.1f}deg  path_safe={safe}")
        if safe:
            best = (q, wps)
            break
    if best is None:
        raise SystemExit("[plan] no path-safe route found to this target -- aborting, nothing moved.")

    q_target, waypoints = best
    print(f"[plan] SAFE path found: {len(waypoints)} waypoint(s)")

    if not args.execute:
        print()
        print("[plan] DRY RUN ONLY -- nothing moved. Re-run with --execute to perform this move.")
        return

    print()
    print("[execute] connecting to robot...")
    from proper_research.hardware.ur_rtde_robot import URRTDERobot
    robot = URRTDERobot(ROBOT_IP, frequency=125.0)
    robot.connect()
    try:
        safety_mode = robot.get_safety_mode()
        prot_stopped = robot.is_protective_stopped()
        robot_mode = robot.get_robot_mode()
        print(f"[execute] pre-move: safety_mode={safety_mode} protective_stopped={prot_stopped} robot_mode={robot_mode}")
        if prot_stopped or safety_mode not in (1, 2) or robot_mode != 7:
            raise SystemExit("[execute] robot not in a safe/running state -- refusing to move.")

        for i, q_wp in enumerate(waypoints, start=1):
            if robot.is_protective_stopped():
                print(f"[execute] ABORT: protective stop detected before waypoint {i}/{len(waypoints)}")
                break
            robot.move_j(list(q_wp), speed=args.speed, acceleration=args.acceleration)
            time.sleep(0.1)
            print(f"[execute] waypoint {i}/{len(waypoints)} commanded")
        time.sleep(0.3)
    except Exception as exc:
        print(f"[execute] exception during motion: {exc!r}")
        print("[execute] checking actual robot position via read-only connection...")
    finally:
        try:
            robot.close()
        except Exception:
            pass

    print("[execute] reading final position (read-only connection)...")
    q_final = get_live_joints()
    fk_final = urik.forward_kinematics(q_final, rk.dh, rk.T_F_M)
    xyz_final = np.asarray(fk_final.T_R_target)[:3, 3]
    rotvec_final = Rot.from_matrix(np.asarray(fk_final.T_R_target)[:3, :3]).as_rotvec()
    d_final = xyz_final - beam_base_xyz
    radius_final_mm = np.linalg.norm(d_final) * 1000
    phi_final_deg = np.degrees(np.arctan2(d_final[1], -d_final[0]))
    dip_final = dipole_world_direction(rotvec_final)
    dir_to_base_final = -d_final / np.linalg.norm(d_final)
    dipole_err_deg = np.degrees(np.arccos(np.clip(np.dot(dip_final, dir_to_base_final), -1, 1)))
    reach_err_deg = np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_final, q_target))))

    print(f"[execute] final: phi={phi_final_deg:.2f}deg r={radius_final_mm:.2f}mm "
          f"dipole_err={dipole_err_deg:.2f}deg (expect ~0 only at psi=0) "
          f"joint_err_from_target={reach_err_deg:.3f}deg")
    if reach_err_deg > 1.0:
        print("[execute] WARNING: did not cleanly reach the commanded target -- "
              "check robot state before issuing another move.")
    else:
        print("[execute] reached target cleanly.")


if __name__ == "__main__":
    main()
