"""Measure J_cam at one nominal state (phi=40deg, r=225mm, a given L) via
6 column finite-difference perturbations + 3 held-out mixed probes +
return-to-nominal repeatability checks.

Usage: python measure_jcam_state.py <L_mm>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import (
    BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, MAGNET_EXCLUSION_RADIUS_BASE_M, REFERENCE_MAGNET_POSE6,
    ROBOT_IP, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, path_safety_check, find_safe_path,
)
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.hardware.online.vessel_stage_a import common as vsa_common

EXCLUSION_RADIUS_M = MAGNET_EXCLUSION_RADIUS_BASE_M
PHI_DEG = 40.0
RADIUS_MM = 225.0
MAX_DQ_RAD = np.radians(5.0)
MIN_DQ_RAD = np.radians(0.05)
TARGET_MM = 1.5
RESULTS_DIR = Path(__file__).resolve().parents[5] / "calibration_2026-10-05"  # archive/ adds one level
RESULTS_DIR.mkdir(exist_ok=True)

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = beam_base_pose6()
LUMEN_FILE = CURRENT_LUMEN_FILE
placeholder_magnet_pose6 = REFERENCE_MAGNET_POSE6
controller_pack = {"robot_dh": rk.dh, "T_F_M": rk.T_F_M}


def q0_for(phi_deg, radius_mm):
    """Bug fixed 2026-10-05: this used to manually subtract the 0.43m
    flange-to-magnet offset to build a TCP-only target AND pass
    T_F_target=rk.T_F_M to the solver, which composes that same offset
    again internally (forward_kinematics(q, dh, T_F_target) already
    returns the MAGNET pose) -- double-counting it by 0.43m. Confirmed
    live: this put the magnet ~0.43m above the beam base in Z (-0.0396 +
    0.43 = 0.39, exactly the bad reading measured on the robot) instead
    of at beam-base height. Fix: pass the MAGNET pose (xyz, rotvec)
    directly as the target; the solver's own T_F_target composition
    handles the offset -- same convention every working script from the
    prior session uses (see robust_ik_for_pose)."""
    phi = np.radians(phi_deg)
    xyz = beam_base_xyz + (radius_mm / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    R = reference_orientation_matrix(xyz, beam_base_xyz)
    rotvec = Rot.from_matrix(R).as_rotvec()
    magnet_pose6 = np.concatenate([xyz, rotvec])
    T = urik.pose6_to_T(magnet_pose6)
    res = urik.inverse_kinematics_dls(
        T_R_target=T, q_seed_rad=rk.q_seed_rad, dh=rk.dh, T_F_target=rk.T_F_M, cfg=rk.ik_cfg,
    )
    if not res.converged:
        raise RuntimeError(f"IK failed for phi={phi_deg} r={radius_mm}")
    q = np.asarray(res.q_rad, dtype=float)
    # Independent verification -- never trust IK convergence alone again.
    fk_check = urik.forward_kinematics(q, rk.dh, rk.T_F_M)
    xyz_check = np.asarray(fk_check.T_R_target)[:3, 3]
    err_mm = float(np.linalg.norm(xyz_check - xyz) * 1000.0)
    if err_mm > 2.0:
        raise RuntimeError(f"q0_for sanity check FAILED: FK(q0) is {err_mm:.1f}mm from intended xyz={xyz}, got {xyz_check}")
    print(f"  [q0_for] verified: FK(q0) matches intended target to {err_mm:.3f}mm")
    return q


def get_live_joints():
    import rtde_receive
    rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    return np.array(rr.getActualQ())


def safe_move_j(q_target, *, speed=0.15, acceleration=0.15):
    """Arc-interpolated, bulge-fallback path-safe move (same machinery as
    move_to_arc_position.py) -- NOT a direct-only exclusion-radius check.

    The direct-only check used earlier only verifies the path never gets
    too CLOSE to the beam base; it never bounds how far the arm swings
    getting there. For a large reconfiguration (e.g. the initial move to
    a new nominal q0 from wherever the robot currently sits), a direct
    joint-space interpolation can pass through a huge, uncontrolled
    Cartesian excursion while staying arbitrarily far from the exclusion
    sphere the whole time -- confirmed live: a massive X excursion this
    session came from exactly this gap. find_safe_path's arc-interpolated
    fallback keeps every waypoint on (or near) the actual physical arc
    instead of letting the straight joint-space line do whatever it wants.
    """
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
            cam_index=0, exposure=29.0, image_filename="/dev/shm/measure_jcam.png",
            roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: 0.025, frame_processor=mapper,
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
    L_mm = float(sys.argv[1])
    L_m = L_mm / 1000.0

    print(f"Building model bundle + J_fast provider for L={L_mm}mm...")
    bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_m)
    provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="fast")
    q0 = q0_for(PHI_DEG, RADIUS_MM)
    state0 = np.concatenate([q0, [L_m]])
    J_fast = provider(state0)[:, :6]  # 3x6, drop the insertion column
    print(f"q0 (deg): {np.round(np.degrees(q0), 3).tolist()}")

    dqs = np.zeros(6)
    for i in range(6):
        col_norm = np.linalg.norm(J_fast[:, i])
        dq = MAX_DQ_RAD if col_norm < 1e-9 else (TARGET_MM / 1000.0) / col_norm
        dqs[i] = float(np.clip(dq, MIN_DQ_RAD, MAX_DQ_RAD))
    print(f"delta_q (deg): {np.round(np.degrees(dqs), 3).tolist()}")

    # 3 held-out mixed probes, normalized, scaled to ~1.5mm predicted via J_fast
    rng = np.random.default_rng(42)
    probes = []
    for _ in range(3):
        u = rng.standard_normal(6)
        u /= np.linalg.norm(u)
        pred_norm = np.linalg.norm(J_fast @ u)
        alpha = (TARGET_MM / 1000.0) / pred_norm if pred_norm > 1e-9 else MAX_DQ_RAD
        alpha = float(np.clip(alpha, MIN_DQ_RAD, MAX_DQ_RAD))
        probes.append(u * alpha)

    camera = build_camera()
    camera.start()
    results = {"L_mm": L_mm, "q0": q0.tolist(), "dqs_rad": dqs.tolist(), "J_fast": J_fast.tolist(),
               "columns": [], "returns": [], "probes": []}
    try:
        print("\nMoving to nominal q0...")
        q_cur, err = safe_move_j(q0)
        print(f"  reached, err={err:.4f}deg")
        tip0, std0 = measure_tip_mean(camera)
        print(f"  tip0={tip0}  std_mm={std0*1000 if std0 is not None else None}")
        results["tip0"] = tip0.tolist() if tip0 is not None else None
        results["tip0_std_mm"] = (std0 * 1000).tolist() if std0 is not None else None

        for i in range(6):
            print(f"\n--- joint q{i+1}, delta_q={np.degrees(dqs[i]):.3f}deg ---")
            dq_vec = np.zeros(6); dq_vec[i] = dqs[i]

            q_plus = q0 + dq_vec
            safe_move_j(q_plus)
            tip_plus, std_plus = measure_tip_mean(camera)
            print(f"  q0+dq: tip={tip_plus}  std_mm={std_plus*1000 if std_plus is not None else None}")

            q_ret1, err1 = safe_move_j(q0)
            tip_ret1, std_ret1 = measure_tip_mean(camera)
            print(f"  return1: tip={tip_ret1}  drift_from_tip0_mm={np.linalg.norm(tip_ret1-tip0)*1000 if tip_ret1 is not None and tip0 is not None else None}")

            q_minus = q0 - dq_vec
            safe_move_j(q_minus)
            tip_minus, std_minus = measure_tip_mean(camera)
            print(f"  q0-dq: tip={tip_minus}  std_mm={std_minus*1000 if std_minus is not None else None}")

            q_ret2, err2 = safe_move_j(q0)
            tip_ret2, std_ret2 = measure_tip_mean(camera)
            print(f"  return2: tip={tip_ret2}  drift_from_tip0_mm={np.linalg.norm(tip_ret2-tip0)*1000 if tip_ret2 is not None and tip0 is not None else None}")

            J_cam_col = (tip_plus - tip_minus) / (2 * dqs[i]) if tip_plus is not None and tip_minus is not None else None
            print(f"  J_cam col {i+1}: {J_cam_col}")

            results["columns"].append({
                "joint": i, "dq_rad": float(dqs[i]),
                "tip_plus": tip_plus.tolist() if tip_plus is not None else None,
                "tip_plus_std_mm": (std_plus * 1000).tolist() if std_plus is not None else None,
                "tip_minus": tip_minus.tolist() if tip_minus is not None else None,
                "tip_minus_std_mm": (std_minus * 1000).tolist() if std_minus is not None else None,
                "J_cam_col": J_cam_col.tolist() if J_cam_col is not None else None,
            })
            results["returns"].append({
                "joint": i,
                "tip_ret1": tip_ret1.tolist() if tip_ret1 is not None else None,
                "tip_ret2": tip_ret2.tolist() if tip_ret2 is not None else None,
            })
            with open(RESULTS_DIR / f"jcam_joint_space_phi{PHI_DEG:.0f}_r{RADIUS_MM:.0f}_L{L_mm:.0f}mm.json", "w") as f:
                json.dump(results, f, indent=2, default=float)

        for pi, u_scaled in enumerate(probes):
            print(f"\n--- held-out probe {pi+1} ---")
            q_p = q0 + u_scaled
            safe_move_j(q_p)
            tip_p, std_p = measure_tip_mean(camera)
            print(f"  probe+: tip={tip_p}  std_mm={std_p*1000 if std_p is not None else None}")
            safe_move_j(q0)
            measure_tip_mean(camera)  # settle, not recorded

            delta_x_real = (tip_p - tip0) if tip_p is not None and tip0 is not None else None
            pred_fast = J_fast @ u_scaled
            results["probes"].append({
                "u_scaled_rad": u_scaled.tolist(),
                "tip_p": tip_p.tolist() if tip_p is not None else None,
                "delta_x_real": delta_x_real.tolist() if delta_x_real is not None else None,
                "pred_fast": pred_fast.tolist(),
            })
            with open(RESULTS_DIR / f"jcam_joint_space_phi{PHI_DEG:.0f}_r{RADIUS_MM:.0f}_L{L_mm:.0f}mm.json", "w") as f:
                json.dump(results, f, indent=2, default=float)
    finally:
        camera.stop()

    print(f"\n\nsaved jcam_L{L_mm:.0f}mm_results.json")


if __name__ == "__main__":
    main()
