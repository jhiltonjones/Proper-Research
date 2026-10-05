"""Stage-2: measure J_cam^contact at phi=35deg, r=225mm, psi=+-30deg, L,
against the REAL digitized vessel wall (vessel_lumen_robot_frame_zcorrected.json)
-- the matched-pair contact counterpart to Stage 1's free-space
measure_jcam_source_psi.py at the SAME phi/psi/L, so any J_cam^free vs
J_cam^contact difference is attributable to contact, not insertion or
source geometry (per the user's explicit matched-pair design).

Same x, y, rz columns only (z, rx, ry excluded for the same reason as
Stage 1). Same eps values as Stage 1's free-space campaign at this
state -- already verified (verify_contact_jacobian_implementation.py's
Check 2) to stay on the same side of the contact/free boundary at both
+eps and -eps, so this is not measuring a mode-switch.

Usage: python measure_jcam_contact.py <psi_deg> <L_mm>
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
    ADVANCER_PORT, BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, MAGNET_EXCLUSION_RADIUS_BASE_M,
    REFERENCE_MAGNET_POSE6, ROBOT_IP, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose, find_safe_path, build_seed_pool, robust_ik_for_pose,
)
from proper_research.hardware.online.vessel_stage_a import common as vsa_common
from proper_research.hardware.online.vessel_stage_a.insertion_control import chord_mm, closed_loop_insertion
from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit

EXCLUSION_RADIUS_M = MAGNET_EXCLUSION_RADIUS_BASE_M
PHI_DEG = 35.0
RADIUS_MM = 225.0
LUMEN_FILE = CURRENT_LUMEN_FILE
CONTACT_BAND_M = 0.5e-3

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = beam_base_pose6()
placeholder_magnet_pose6 = REFERENCE_MAGNET_POSE6

PSI_DEG = float(sys.argv[1])
L_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
L_M = L_MM / 1000.0

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
R0 = R_aligned @ R_psi
rotvec0 = Rot.from_matrix(R0).as_rotvec()

EPS = {"x": 0.015, "y": {30.0: 0.0169, -30.0: 0.020}.get(PSI_DEG, 0.017), "rz": np.radians({30.0: 8.54, -30.0: 10.53}.get(PSI_DEG, 9.5))}
AXIS_IDX = {"x": 0, "y": 1, "rz": 5}


def perturbed_pose(axis, sign):
    eps = EPS[axis]
    idx = AXIS_IDX[axis]
    if idx < 3:
        d = np.zeros(3)
        d[idx] = sign * eps
        return xyz0 + d, rotvec0, eps
    else:
        ax = idx - 3
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
            cam_index=0, exposure=29.0, image_filename="/dev/shm/measure_jcam_contact.png",
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


def model_c_min(model_c, xyz, rotvec, insertion_m):
    out = solve_pose(model_c, xyz, rotvec, insertion_m)
    centerline = np.asarray(out["centerline"])
    r_beam = float(model_c.contact_cfg.params.r_beam)
    delta, Rloc, _, grad_delta, _ = model_c.lumen_query.closest_many_with_gradients(centerline, window=None)
    c = Rloc - delta - r_beam
    M = centerline.shape[0]
    i_min = int(np.argmin(c[1:])) + 1 if M > 1 else 0
    return float(c[i_min])


def main():
    print(f"PHI_DEG={PHI_DEG} L_MM={L_MM} PSI_DEG={PSI_DEG}  (CONTACT campaign)")
    print(f"nominal xyz0: {xyz0}  rotvec0: {rotvec0}")

    bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_M)
    model_c = bundle.models["contact"]
    c_min0 = model_c_min(model_c, xyz0, rotvec0, L_M)
    print(f"model-predicted nominal c_min = {c_min0*1000:.3f}mm  (in_contact={c_min0 <= CONTACT_BAND_M})")

    q_now = get_live_joints()
    q0 = ik_for_target(xyz0, rotvec0, q_now)
    print(f"q0 (deg): {np.round(np.degrees(q0), 3).tolist()}")

    camera = build_camera()
    camera.start()
    advancer = AdvancerUnit(port=ADVANCER_PORT)
    results = {"phi_deg": PHI_DEG, "L_mm_target": L_MM, "psi_deg": PSI_DEG, "campaign": "contact",
               "xyz0": xyz0.tolist(), "rotvec0": rotvec0.tolist(),
               "eps": {k: (v if k != "rz" else float(v)) for k, v in EPS.items()},
               "model_c_min0_mm": c_min0 * 1000.0,
               "columns": [], "returns": []}
    try:
        time.sleep(1.0)
        print("\nMoving to nominal q0...")
        q_cur, err = safe_move_j(q0)
        print(f"  reached, err={err:.4f}deg")

        print(f"\nConverging insertion length to L={L_MM:.2f}mm (real advancer command)...")
        conv_L, _ratio = closed_loop_insertion(camera, advancer, L_MM, beam_base_xyz)
        print(f"  converged_L={conv_L}")
        results["converged_L_mm"] = conv_L
        if conv_L is None or abs(conv_L - L_MM) > 2.0:
            raise RuntimeError(f"insertion convergence FAILED: got {conv_L}mm, wanted {L_MM}mm")

        tip0, std0 = measure_tip_mean(camera)
        print(f"  tip0={tip0}  std_mm={std0*1000 if std0 is not None else None}")
        results["tip0"] = tip0.tolist() if tip0 is not None else None

        for axis in ("x", "y", "rz"):
            print(f"\n--- axis {axis}, eps={EPS[axis]:.5f} ---")

            xyz_p, rv_p, eps = perturbed_pose(axis, +1)
            c_min_p = model_c_min(model_c, xyz_p, rv_p, L_M)
            q_p = ik_for_target(xyz_p, rv_p, q0)
            safe_move_j(q_p)
            tip_plus, std_plus = measure_tip_mean(camera)
            print(f"  +eps: tip={tip_plus}  std_mm={std_plus*1000 if std_plus is not None else None}  model_c_min={c_min_p*1000:.3f}mm")

            safe_move_j(q0)
            tip_ret1, _ = measure_tip_mean(camera)
            drift1 = np.linalg.norm(tip_ret1 - tip0) * 1000 if tip_ret1 is not None and tip0 is not None else None
            print(f"  return1: drift_mm={drift1}")

            xyz_m, rv_m, _ = perturbed_pose(axis, -1)
            c_min_m = model_c_min(model_c, xyz_m, rv_m, L_M)
            q_m = ik_for_target(xyz_m, rv_m, q0)
            safe_move_j(q_m)
            tip_minus, std_minus = measure_tip_mean(camera)
            print(f"  -eps: tip={tip_minus}  std_mm={std_minus*1000 if std_minus is not None else None}  model_c_min={c_min_m*1000:.3f}mm")

            safe_move_j(q0)
            tip_ret2, _ = measure_tip_mean(camera)
            drift2 = np.linalg.norm(tip_ret2 - tip0) * 1000 if tip_ret2 is not None and tip0 is not None else None
            print(f"  return2: drift_mm={drift2}")

            J_col = (tip_plus - tip_minus) / (2 * eps) if tip_plus is not None and tip_minus is not None else None
            print(f"  J_cam^contact col {axis}: {J_col}")

            results["columns"].append({
                "axis": axis, "eps": eps,
                "tip_plus": tip_plus.tolist() if tip_plus is not None else None,
                "tip_minus": tip_minus.tolist() if tip_minus is not None else None,
                "J_col": J_col.tolist() if J_col is not None else None,
                "model_c_min_plus_mm": c_min_p * 1000.0, "model_c_min_minus_mm": c_min_m * 1000.0,
            })
            results["returns"].append({"axis": axis, "drift1_mm": drift1, "drift2_mm": drift2})
            out_path = Path(__file__).resolve().parents[4] / "calibration_2026-10-05" / f"jcam_contact_phi35_L{L_MM:.0f}_psi{PSI_DEG:+.0f}.json"
            out_path.parent.mkdir(exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=float)
    finally:
        advancer.shutdown()
        camera.stop()

    print(f"\n\nsaved {out_path}")


if __name__ == "__main__":
    main()
