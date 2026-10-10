"""Revised pilot: converge insertion AT r=225mm (not far away), then sweep
phi=-60..60 with the dipole orientation held fixed (computed once at
phi=0). No separate far/weak-field baseline -- that confounded gravity
sag (invisible to the overhead camera) with insertion control.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    reference_orientation_matrix, robust_ik_for_pose, build_seed_pool, find_safe_path,
)
from proper_research.hardware.online.vessel_stage_a import common as vsa_common
from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit

ROBOT_IP = "192.168.56.101"
EXCLUSION_RADIUS_M = 0.080
RADIUS_MM = 225.0
L_TARGET_MM = 40.0
PHI_SWEEP_DEG = [-60.0, -30.0, 0.0, 30.0, 60.0]

beam_base_xyz = BEAM_BASE_XYZ_M.copy()

xyz_phi0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-1.0, 0.0, 0.0])
R_FIXED = reference_orientation_matrix(xyz_phi0, beam_base_xyz)
ROTVEC_FIXED = Rot.from_matrix(R_FIXED).as_rotvec()
print("FIXED dipole orientation rotvec (held constant for the whole arc):", ROTVEC_FIXED)


def get_live_joints():
    import rtde_receive
    rr = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    return np.array(rr.getActualQ())


def move_fixed_orientation(phi_deg: float, radius_mm: float, *, speed=0.2, acceleration=0.2):
    q_now = get_live_joints()
    fk_now = urik.forward_kinematics(q_now, rk.dh, rk.T_F_M)
    xyz_now = np.asarray(fk_now.T_R_target)[:3, 3]
    rotvec_now = Rot.from_matrix(np.asarray(fk_now.T_R_target)[:3, :3]).as_rotvec()

    phi = np.deg2rad(phi_deg)
    xyz_target = beam_base_xyz + (radius_mm / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    rotvec_target = ROTVEC_FIXED

    pool = build_seed_pool(q_now)
    sols = robust_ik_for_pose(xyz_target, rotvec_target, pool, rk.dh, rk.ik_cfg)
    if not sols:
        raise RuntimeError(f"no IK solution for phi={phi_deg} r={radius_mm}")
    lumen_C = beam_base_xyz.reshape(1, 3)
    best = None
    for q in sorted(sols, key=lambda q: float(np.max(np.abs(urik.wrapped_joint_difference(q, q_now))))):
        wps, safe, msg = find_safe_path(
            q_now, xyz_now, rotvec_now, q, xyz_target, rotvec_target,
            DH=rk.dh, ik_cfg=rk.ik_cfg, T_F_M=rk.T_F_M, lumen_C_m=lumen_C,
            exclusion_radius_m=EXCLUSION_RADIUS_M, z_bounds_m=None, beam_base_xyz=beam_base_xyz,
        )
        if safe:
            best = (q, wps)
            break
    if best is None:
        raise RuntimeError(f"no safe path to phi={phi_deg} r={radius_mm}")
    q_target, waypoints = best
    dq = np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_target, q_now))))
    print(f"  [move] phi={phi_deg:+.1f}deg r={radius_mm:.1f}mm  dq={dq:.1f}deg  {len(waypoints)} waypoint(s)")

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
    fk_final = urik.forward_kinematics(q_final, rk.dh, rk.T_F_M)
    xyz_final = np.asarray(fk_final.T_R_target)[:3, 3]
    err_deg = np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_final, q_target))))
    print(f"  [move] reached: xyz={xyz_final}  joint_err={err_deg:.3f}deg")
    return q_final, xyz_final


def build_camera():
    scfg = vsa_common._raised_stream_stream_config()
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/fixed_dipole_arc_pilot_v2.png",
            roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: 0.025, frame_processor=mapper,
    )
    return camera


def measure_tip_median(camera, n=25, timeout_s=12.0):
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
    return np.median(arr, axis=0), np.std(arr, axis=0)


def chord_mm(camera, retries=3):
    for attempt in range(retries):
        tip, _std = measure_tip_median(camera, n=15, timeout_s=8.0)
        if tip is not None:
            return float(np.linalg.norm(tip - beam_base_xyz) * 1000.0)
        print(f"    [chord_mm] no detections, retry {attempt+1}/{retries}")
        time.sleep(1.0)
    return None


def closed_loop_insertion(camera, advancer, target_mm, *, tol_mm=0.5, max_iters=10, max_step_mm=10.0):
    """Adaptive-gain closed loop. The advancer's commanded-vs-actual
    realization ratio varies (confirmed live: ~50% far from the magnet at
    r=450mm, ~90-100% at r=225mm with the magnet's field present) -- a
    fixed assumed ratio caused a divergent, growing oscillation here
    earlier. Track the ratio empirically from each step's actual-vs-
    commanded delta and use that estimate for the next step, clamped to a
    sane range, plus a hard per-step command-size clamp as a backstop."""
    current = chord_mm(camera)
    if current is None:
        print("    [insertion] WARNING: no camera detection at start -- skipping convergence")
        return None
    print(f"    [insertion] start L~{current:.2f}mm target={target_mm:.2f}mm")
    assumed_ratio = 0.9  # confirmed close to this near the magnet at r=225mm
    for it in range(max_iters):
        delta = target_mm - current
        if abs(delta) <= tol_mm:
            print(f"    [insertion] within tolerance after {it} correction(s)")
            break
        cmd_mm = float(np.clip(delta / assumed_ratio, -max_step_mm, max_step_mm))
        if cmd_mm > 0:
            advancer.forward(abs(cmd_mm), delay_us=25)
        else:
            advancer.backward(abs(cmd_mm), delay_us=25)
        time.sleep(1.0)
        new_current = chord_mm(camera)
        if new_current is None:
            print("    [insertion] WARNING: lost camera detection mid-correction -- stopping here")
            break
        actual_delta = new_current - current
        if abs(cmd_mm) > 0.5:
            observed_ratio = actual_delta / cmd_mm
            assumed_ratio = float(np.clip(0.5 * assumed_ratio + 0.5 * np.clip(observed_ratio, 0.2, 1.5), 0.2, 1.5))
        current = new_current
        print(f"    [insertion] iter {it+1}: L~{current:.2f}mm (target {target_mm:.2f}mm, assumed_ratio={assumed_ratio:.2f})")
    else:
        print(f"    [insertion] WARNING: did not converge, final L~{current:.2f}mm")
    return current


if __name__ == "__main__":
    camera = build_camera()
    camera.start()
    advancer = AdvancerUnit(port="/dev/ttyACM0")
    results = {}
    try:
        time.sleep(1.0)

        print(f"\n=== phi=0, r={RADIUS_MM}mm, L~{L_TARGET_MM}mm (already converged, measuring reference) ===")
        tip_med, tip_std = measure_tip_median(camera, n=25, timeout_s=12.0)
        print(f"phi=0 reference tip: {tip_med}  std_mm={tip_std*1000 if tip_std is not None else None}")
        results["phi0_reference"] = {
            "tip": tip_med.tolist() if tip_med is not None else None,
            "std_mm": (tip_std * 1000).tolist() if tip_std is not None else None,
            "converged_L_mm": 39.91,
        }

        print(f"\n=== Sweep phi = {PHI_SWEEP_DEG} at r={RADIUS_MM}mm, L~{L_TARGET_MM}mm ===")
        for phi_deg in PHI_SWEEP_DEG:
            if phi_deg == 0.0:
                # already there/measured above
                results[phi_deg] = results["phi0_reference"]
                continue
            print(f"\n  -- phi={phi_deg}deg --")
            move_fixed_orientation(phi_deg, RADIUS_MM)
            conv_L = closed_loop_insertion(camera, advancer, L_TARGET_MM)
            tip_med, tip_std = measure_tip_median(camera, n=25, timeout_s=12.0)
            if tip_med is None:
                print(f"  phi={phi_deg}: NO CAMERA DETECTION")
                results[phi_deg] = {"tip": None, "std_mm": None, "converged_L_mm": conv_L}
                continue
            print(f"  phi={phi_deg}: tip={tip_med}  std_mm={tip_std*1000}  converged_L={conv_L}")
            results[phi_deg] = {
                "tip": tip_med.tolist(), "std_mm": (tip_std * 1000).tolist(),
                "converged_L_mm": conv_L,
            }
    finally:
        advancer.shutdown()
        camera.stop()

    print("\n\n=== SUMMARY ===")
    ref = np.asarray(results["phi0_reference"]["tip"]) if results["phi0_reference"]["tip"] is not None else None
    print(f"{'phi':>8s} {'tip':>40s} {'std_mm':>25s} {'disp_from_phi0_mm':>20s}")
    for phi_deg in PHI_SWEEP_DEG:
        r = results[phi_deg]
        if r["tip"] is None:
            print(f"{phi_deg:8.1f} {'NO DETECTION':>40s}")
            continue
        tip = np.asarray(r["tip"])
        disp = np.linalg.norm(tip - ref) * 1000.0 if ref is not None else float("nan")
        print(f"{phi_deg:8.1f} {str(np.round(tip,5)):>40s} {str(np.round(r['std_mm'],3)):>25s} {disp:20.3f}")

    import json
    with open("/home/jack/.claude/jobs/043c0b8b/tmp/fixed_dipole_arc_pilot_v2_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=float)
    print("\nsaved results JSON")
