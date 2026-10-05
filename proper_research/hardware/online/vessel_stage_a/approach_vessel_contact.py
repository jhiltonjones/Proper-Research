"""Stage-2 hardware: incrementally approach the real vessel wall at
phi=35deg, r=225mm, for a given psi, converging insertion to a target L
one step at a time (30 -> 35 -> 40mm), reporting tip position and
model-predicted contact state after each step -- per the user's explicit
"incremental insertion with a pause to check" instruction, since this is
the first time this session actually drives the beam into contact with
a real wall (not just free space).

Usage: python approach_vessel_contact.py <psi_deg> <L_mm_target>
"""
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose, find_safe_path, build_seed_pool, robust_ik_for_pose,
)
from proper_research.hardware.online.vessel_stage_a import common as vsa_common
from proper_research.hardware.online.vessel_stage_a.insertion_control import chord_mm, closed_loop_insertion
from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit

ROBOT_IP = "192.168.56.101"
ADVANCER_PORT = "/dev/ttyACM0"
EXCLUSION_RADIUS_M = 0.080
PHI_DEG = 35.0
RADIUS_MM = 225.0
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_zcorrected.json"
CONTACT_BAND_M = 0.5e-3

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = np.concatenate([beam_base_xyz, [3.14159265, 0.0, 0.0]])
placeholder_magnet_pose6 = np.array(
    [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
     -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
)

PSI_DEG = float(sys.argv[1])
L_MM_TARGET = float(sys.argv[2])

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
R0 = R_aligned @ R_psi
rotvec0 = Rot.from_matrix(R0).as_rotvec()


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
            cam_index=0, exposure=29.0, image_filename="/dev/shm/approach_vessel_contact.png",
            roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: L_MM_TARGET / 1000.0, frame_processor=mapper,
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


def model_contact_profile(L_m):
    bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_m)
    model_c = bundle.models["contact"]
    out = solve_pose(model_c, xyz0, rotvec0, L_m)
    centerline = np.asarray(out["centerline"])
    r_beam = float(model_c.contact_cfg.params.r_beam)
    delta, Rloc, _, grad_delta, _ = model_c.lumen_query.closest_many_with_gradients(centerline, window=None)
    c = Rloc - delta - r_beam
    M = centerline.shape[0]
    i_min = int(np.argmin(c[1:])) + 1 if M > 1 else 0
    return float(c[i_min]), out["tip"]


def main():
    print(f"PHI_DEG={PHI_DEG} RADIUS_MM={RADIUS_MM} PSI_DEG={PSI_DEG} L_MM_TARGET={L_MM_TARGET}")
    print(f"nominal xyz0: {xyz0}  rotvec0: {rotvec0}")

    c_min_model, tip_model = model_contact_profile(L_MM_TARGET / 1000.0)
    print(f"model-predicted c_min at this L = {c_min_model*1000:.3f}mm  "
          f"(in_contact={c_min_model <= CONTACT_BAND_M})  model tip={np.round(tip_model,5)}")

    q_now = get_live_joints()
    q0 = ik_for_target(xyz0, rotvec0, q_now)
    print(f"q0 (deg): {np.round(np.degrees(q0), 3).tolist()}")

    camera = build_camera()
    camera.start()
    advancer = AdvancerUnit(port=ADVANCER_PORT)
    try:
        time.sleep(1.0)
        print("\nMoving to nominal q0...")
        q_cur, err = safe_move_j(q0)
        print(f"  reached, err={err:.4f}deg")

        print(f"\nConverging insertion to L={L_MM_TARGET:.2f}mm...")
        conv_L, ratio = closed_loop_insertion(camera, advancer, L_MM_TARGET, beam_base_xyz)
        print(f"  converged_L={conv_L}  assumed_ratio={ratio:.2f}")

        tip, std = measure_tip_mean(camera)
        print(f"\ntip(camera) = {tip}  std_mm={std*1000 if std is not None else None}")
        print(f"tip(model)  = {np.round(tip_model, 5)}")
        if tip is not None:
            err_mm = float(np.linalg.norm(tip - np.asarray(tip_model)) * 1000.0)
            print(f"camera-vs-model tip discrepancy = {err_mm:.3f}mm")
    finally:
        advancer.shutdown()
        camera.stop()


if __name__ == "__main__":
    main()
