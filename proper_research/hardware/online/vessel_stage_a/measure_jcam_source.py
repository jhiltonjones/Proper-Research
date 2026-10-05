"""Measure J_cam^source at the phi=40deg/r=225mm/L=25mm nominal state via
6 Cartesian (source-magnet-pose) finite-difference perturbations.

Isolates magnet-to-tip beam physics from robot kinematics/IK-branch
effects, per the user's explicit decomposition:
    J_q->tip = J_source->tip @ J_q->source
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose, find_safe_path, build_seed_pool, robust_ik_for_pose,
)
from proper_research.hardware.online.vessel_stage_a import common as vsa_common

ROBOT_IP = "192.168.56.101"
EXCLUSION_RADIUS_M = 0.080
PHI_DEG = 40.0
RADIUS_MM = 225.0
L_M = 0.025
RESULTS_DIR = Path(__file__).resolve().parents[4] / "calibration_2026-10-05"
RESULTS_DIR.mkdir(exist_ok=True)

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_left1p5mm_zcorrected.json"
placeholder_magnet_pose6 = np.array(
    [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
     -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
)
beam_base_xyz6 = np.concatenate([beam_base_xyz, [3.14159265, 0.0, 0.0]])

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R0 = reference_orientation_matrix(xyz0, beam_base_xyz)
rotvec0 = Rot.from_matrix(R0).as_rotvec()

EPS = {
    0: 0.0164, 1: 0.0200, 2: 0.0157,  # x,y,z translation (m)
    3: np.radians(15.0), 4: np.radians(15.0), 5: np.radians(14.1851),  # rx,ry,rz (rad)
}
LABELS = ["x", "y", "z", "rx", "ry", "rz"]


def perturbed_pose(axis_idx, sign):
    eps = EPS[axis_idx]
    if axis_idx < 3:
        d = np.zeros(3)
        d[axis_idx] = sign * eps
        return xyz0 + d, rotvec0, eps
    else:
        ax = axis_idx - 3
        d = np.zeros(3)
        d[ax] = sign * eps
        R_pert = Rot.from_rotvec(d).as_matrix() @ R0
        return xyz0, Rot.from_matrix(R_pert).as_rotvec(), eps


def get_live_joints():
    import rtde_receive
    rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    return np.array(rr.getActualQ())


def ik_for_target(xyz, rotvec, q_seed):
    pool = build_seed_pool(q_seed)
    sols = robust_ik_for_pose(xyz, rotvec, pool, rk.dh, rk.ik_cfg)
    if not sols:
        raise RuntimeError(f"no IK solution for xyz={xyz} rotvec={rotvec}")
    q = min(sols, key=lambda qq: float(np.max(np.abs(urik.wrapped_joint_difference(qq, q_seed)))))
    fk_check = urik.forward_kinematics(q, rk.dh, rk.T_F_M)
    xyz_check = np.asarray(fk_check.T_R_target)[:3, 3]
    err_mm = float(np.linalg.norm(xyz_check - xyz) * 1000.0)
    if err_mm > 2.0:
        raise RuntimeError(f"IK sanity check FAILED: {err_mm:.1f}mm off target")
    return q


def safe_move_j(q_target, *, speed=0.15, acceleration=0.15):
    q_now = get_live_joints()
    fk_now = urik.forward_kinematics(q_now, rk.dh, rk.T_F_M)
    xyz_now = np.asarray(fk_now.T_R_target)[:3, 3]
    rotvec_now = Rot.from_matrix(np.asarray(fk_now.T_R_target)[:3, :3]).as_rotvec()
    fk_target = urik.forward_kinematics(q_target, rk.dh, rk.T_F_M)
    xyz_target = np.asarray(fk_target.T_R_target)[:3, 3]
    rotvec_target = Rot.from_matrix(np.asarray(fk_target.T_R_target)[:3, :3]).as_rotvec()

    lumen_C = beam_base_xyz.reshape(1, 3)
    waypoints, safe, msg = find_safe_path(
        q_now, xyz_now, rotvec_now, q_target, xyz_target, rotvec_target,
        DH=rk.dh, ik_cfg=rk.ik_cfg, T_F_M=rk.T_F_M, lumen_C_m=lumen_C,
        exclusion_radius_m=EXCLUSION_RADIUS_M, z_bounds_m=None, beam_base_xyz=beam_base_xyz,
    )
    if not safe:
        raise RuntimeError(f"no safe path found: {msg[:300]}")

    from proper_research.hardware.ur_rtde_robot import URRTDERobot
    robot = URRTDERobot(ROBOT_IP, frequency=125.0)
    robot.connect()
    try:
        if robot.is_protective_stopped() or robot.get_robot_mode() != 7:
            raise RuntimeError("robot not in a safe/running state")
        for q_wp in waypoints:
            if robot.is_protective_stopped():
                raise RuntimeError("protective stop mid-sequence")
            robot.move_j(list(q_wp), speed=speed, acceleration=acceleration)
            time.sleep(0.1)
        time.sleep(0.2)
    finally:
        try:
            robot.close()
        except Exception:
            pass
    q_final = get_live_joints()
    err_deg = np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_final, q_target))))
    return q_final, err_deg


def build_camera():
    scfg = vsa_common._raised_stream_stream_config()
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/measure_jcam_source.png",
            roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: L_M, frame_processor=mapper,
    )
    return camera


def measure_tip_mean(camera, n=25, timeout_s=10.0):
    tips = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s and len(tips) < n:
        est, _age = camera.latest(0.5)
        if est is not None:
            tip = np.asarray(est.tip_position_m, dtype=float)
            if np.all(np.isfinite(tip)):
                tips.append(tip)
        time.sleep(0.1)
    if not tips:
        return None, None
    arr = np.vstack(tips)
    return np.mean(arr, axis=0), np.std(arr, axis=0)


def main():
    print(f"nominal xyz0: {xyz0}  rotvec0: {rotvec0}")
    q_now = get_live_joints()
    q0 = ik_for_target(xyz0, rotvec0, q_now)
    print(f"q0 (deg): {np.round(np.degrees(q0), 3).tolist()}")

    camera = build_camera()
    camera.start()
    results = {"xyz0": xyz0.tolist(), "rotvec0": rotvec0.tolist(), "eps": {str(k): v for k, v in EPS.items()},
               "columns": [], "returns": []}
    try:
        time.sleep(1.0)
        print("\nMoving to nominal q0...")
        q_cur, err = safe_move_j(q0)
        print(f"  reached, err={err:.4f}deg")
        tip0, std0 = measure_tip_mean(camera)
        print(f"  tip0={tip0}  std_mm={std0*1000 if std0 is not None else None}")
        results["tip0"] = tip0.tolist() if tip0 is not None else None

        for i in range(6):
            label = LABELS[i]
            print(f"\n--- axis {label}, eps={EPS[i]:.5f} ---")

            xyz_p, rv_p, eps = perturbed_pose(i, +1)
            q_p = ik_for_target(xyz_p, rv_p, q0)
            safe_move_j(q_p)
            tip_plus, std_plus = measure_tip_mean(camera)
            print(f"  +eps: tip={tip_plus}  std_mm={std_plus*1000 if std_plus is not None else None}")

            safe_move_j(q0)
            tip_ret1, _ = measure_tip_mean(camera)
            drift1 = np.linalg.norm(tip_ret1 - tip0) * 1000 if tip_ret1 is not None and tip0 is not None else None
            print(f"  return1: drift_mm={drift1}")

            xyz_m, rv_m, _ = perturbed_pose(i, -1)
            q_m = ik_for_target(xyz_m, rv_m, q0)
            safe_move_j(q_m)
            tip_minus, std_minus = measure_tip_mean(camera)
            print(f"  -eps: tip={tip_minus}  std_mm={std_minus*1000 if std_minus is not None else None}")

            safe_move_j(q0)
            tip_ret2, _ = measure_tip_mean(camera)
            drift2 = np.linalg.norm(tip_ret2 - tip0) * 1000 if tip_ret2 is not None and tip0 is not None else None
            print(f"  return2: drift_mm={drift2}")

            J_col = (tip_plus - tip_minus) / (2 * eps) if tip_plus is not None and tip_minus is not None else None
            print(f"  J_cam^source col {label}: {J_col}")

            results["columns"].append({
                "axis": label, "eps": eps,
                "tip_plus": tip_plus.tolist() if tip_plus is not None else None,
                "tip_minus": tip_minus.tolist() if tip_minus is not None else None,
                "J_col": J_col.tolist() if J_col is not None else None,
            })
            results["returns"].append({"axis": label, "drift1_mm": drift1, "drift2_mm": drift2})
            with open(RESULTS_DIR / f"jcam_source_phi{PHI_DEG:.0f}_r{RADIUS_MM:.0f}_L{int(L_M*1000)}mm.json", "w") as f:
                json.dump(results, f, indent=2, default=float)
    finally:
        camera.stop()

    print("\n\nsaved jcam_source_results.json")


if __name__ == "__main__":
    main()
