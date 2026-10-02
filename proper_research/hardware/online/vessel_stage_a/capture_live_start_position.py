#!/usr/bin/env python3
"""Capture the robot's CURRENT live joints/TCP-pose/source-magnet-pose and
save them as a start-position JSON usable by build_vessel_plan.py's
--start-position-json.

This is the first step of "plan from wherever the robot happens to be right
now": jog the robot (manually, or via any other script) to the pose you
want the offline plan to start from, then run this to record it.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.capture_live_start_position \\
        --out vessel_magnet_initial_position_live_2026-09-28.json \\
        --exclusion-floor-mm 210.43
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.ur_rtde_robot import URRTDERobot
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
    BEAM_BASE_PIVOT_XY_ROT, BEAM_BASE_PIVOT_Z,
)

# 2026-10-02 fix: removed the --z-raise-mm/zraise_patch indirection
# entirely (same change as build_vessel_plan.py/run_mpc_delay_aware_
# vessel.py/run_open_loop_vessel.py's own 2026-10-02 fixes) -- the rig has
# been recalibrated via real forward kinematics to a single fixed
# beam-base height, so there is no longer a "raised vs unraised" choice to
# make here, and no flag to keep in sync across scripts.
ROBOT_IP = "192.168.56.101"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--exclusion-floor-mm", type=float, default=None,
                   help="magnet-to-beam-base safety floor to record alongside this position. "
                        "If omitted, uses this position's own measured distance to the beam base "
                        "(i.e. treats THIS position as the new floor -- matches the "
                        "2026-09-27 workflow where the floor was set from a deliberately "
                        "jogged-to, verified-safe position).")
    p.add_argument("--robot-ip", default=ROBOT_IP)
    args = p.parse_args()

    beam_base_xyz = np.array([
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z,
    ])

    robot = URRTDERobot(args.robot_ip, frequency=125.0)
    robot.connect()
    q_live = np.array(robot.get_joints())
    safety_mode = robot.get_safety_mode()
    protective_stopped = robot.is_protective_stopped()
    robot.close()

    if protective_stopped or safety_mode not in (1, 2):
        raise RuntimeError(
            f"robot not in a safe/normal state (safety_mode={safety_mode}, "
            f"protective_stopped={protective_stopped}) -- refusing to capture."
        )

    robot_kin = _resolve_robot_kinematics(make_robot_config())
    T_magnet = urik.forward_kinematics(q_live, robot_kin.dh, robot_kin.T_F_M)
    magnet_xyz = np.asarray(T_magnet.T_R_target[:3, 3])
    magnet_rotvec = Rot.from_matrix(T_magnet.T_R_target[:3, :3]).as_rotvec()
    magnet_pose6 = np.r_[magnet_xyz, magnet_rotvec]

    T_tcp = urik.forward_kinematics(q_live, robot_kin.dh)
    tcp_xyz = np.asarray(T_tcp.T_R_F[:3, 3])
    tcp_rotvec = Rot.from_matrix(T_tcp.T_R_F[:3, :3]).as_rotvec()
    tcp_pose6 = np.r_[tcp_xyz, tcp_rotvec]

    distance_to_base_mm = float(np.linalg.norm(magnet_xyz - beam_base_xyz)) * 1000.0
    exclusion_floor_mm = args.exclusion_floor_mm if args.exclusion_floor_mm is not None else distance_to_base_mm

    if distance_to_base_mm < exclusion_floor_mm - 1e-6:
        print(f"[WARNING] this position ({distance_to_base_mm:.2f}mm from beam base) is CLOSER than "
              f"the recorded exclusion floor ({exclusion_floor_mm:.2f}mm) -- double check this is intended.")

    record = {
        "description": "Live-captured robot start position (recalibrated, fixed beam-base height).",
        "joints_rad": q_live.tolist(),
        "magnet_pose6_R": magnet_pose6.tolist(),
        "tcp_pose6_R": tcp_pose6.tolist(),
        "beam_base_pivot_xyz_R": beam_base_xyz.tolist(),
        "distance_magnet_to_beam_base_mm": distance_to_base_mm,
        "exclusion_floor_mm": exclusion_floor_mm,
        "safety_mode_at_read": safety_mode,
        "protective_stopped_at_read": protective_stopped,
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)

    print(f"[capture] q_live = {q_live.tolist()}")
    print(f"[capture] magnet_pose6_R = {magnet_pose6.tolist()}")
    print(f"[capture] distance to beam base = {distance_to_base_mm:.2f}mm")
    print(f"[capture] exclusion_floor_mm = {exclusion_floor_mm:.2f}mm")
    print(f"[capture] saved -> {args.out}")


if __name__ == "__main__":
    main()
