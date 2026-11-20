"""
sweep_forward_vs_vision.py

Sweep get_point(0, j) for j = 0..70 and record:
- commanded pose
- actual robot pose
- predicted tip
- measured tip
- errors

Outputs
-------
results_sweep_forward_validation/
    sweep_results.json
    sweep_results.csv
"""

import os
import json
import csv
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel
from proper_research.experiments.charactisations import rotate_body_xy ,compute_mm_per_pixel_from_green, rod_section_stiffness, make_Kbt_inv_profile,effective_lengths
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.robot.transformations import get_point

from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz

from proper_research.simulation.boundary_forward_model import (
    EnergyMinForwardWithLumen,
    DeterministicForward6D,WarmForwardP8TipTangent,
    effective_lengths,
)
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel
from proper_research.vision.measure_length import new_capture
from proper_research.parameters import default_magnet_params, default_beam_params

from proper_research.vision.bounds_beam import (
    detect_2_green_calibration_points,
    measure_tip_state_4markers, load_polygon
)

mag_params = default_magnet_params()
beam_params = default_beam_params()

# ============================================================
# Config
# ============================================================

@dataclass
class SweepEvalConfig:
    pivot_pose6: np.ndarray
    beam_base_point_robot_m: np.ndarray
    L_m: float

    i_fixed: int = 0
    j_start: int = 0
    j_end: int = 70
    image_filename: str = "focused_image.jpg"
    reference_image_filename: str = "focused_image_straight.jpg"
    use_reference_frame: bool = True

    red_roi_path: str = "red_roi_box.json"
    green_roi_path: str = "green_roi_box.json"

    known_green_distance_mm: float = 30.0
    pivot_hint: Tuple[float, float] | None = None

    results_dir: str = "results_sweep_forward_validation"

    show_debug_vision: bool = False
    show_debug_model: bool = False

    capture_each_step: bool = True
    settle_time_s: float = 0.3


# ============================================================
# Helpers copied from your working script
# ============================================================

def pivot_rotation_matrix(pivot_pose6: np.ndarray) -> np.ndarray:
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    return T[:3, :3]


def robot_point_to_pivot_local(p_robot, pivot_pose6):
    p_robot = np.asarray(p_robot, dtype=float).reshape(3,)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)
    return R_pivot.T @ (p_robot - p_pivot)


def pivot_local_point_to_robot(local_xyz, pivot_pose6):
    local_xyz = np.asarray(local_xyz, dtype=float).reshape(3,)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)
    return p_pivot + R_pivot @ local_xyz


def robot_dir_to_pivot_local(v_robot, pivot_pose6):
    v_robot = np.asarray(v_robot, dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)
    return R_pivot.T @ v_robot


def pivot_local_dir_to_robot(v_local, pivot_pose6):
    v_local = np.asarray(v_local, dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)
    return R_pivot @ v_local
# ============================================================
# Basic helpers
# ============================================================
# def robot_point_to_pivot_local(p_robot, pivot_pose6):
#     p_robot = np.asarray(p_robot, dtype=float).reshape(3,)
#     p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
#     return p_robot - p_pivot

def robot_pose_to_pivot_local_pose(pose6_robot, pivot_pose6):
    pose6_robot = np.asarray(pose6_robot, dtype=float).reshape(6,)
    return robot_point_to_pivot_local(pose6_robot[:3], pivot_pose6)

# def pivot_local_point_to_robot(local_xyz, pivot_pose6):
#     p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
#     local_xyz = np.asarray(local_xyz, dtype=float).reshape(3,)
#     return p_pivot + local_xyz

# def robot_dir_to_pivot_local(v_robot, pivot_pose6):
#     v_robot = np.asarray(v_robot, dtype=float).reshape(3,)
#     return v_robot.copy()
def rotvec_to_quat_wxyz(rvec: np.ndarray) -> np.ndarray:
    r = Rot.from_rotvec(np.asarray(rvec, dtype=float).reshape(3,))
    q_xyzw = r.as_quat()  # [x, y, z, w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
    return q_wxyz

def pose6_and_L_to_pose8_quat(ur_pose6: np.ndarray, L_m: float) -> np.ndarray:
    ur_pose6 = np.asarray(ur_pose6, dtype=float).reshape(6,)
    q_wxyz = rotvec_to_quat_wxyz(ur_pose6[3:6])

    p8 = np.array([
        ur_pose6[0], ur_pose6[1], ur_pose6[2],
        q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3],
        float(L_m)
    ], dtype=float)

    if not np.all(np.isfinite(p8)):
        raise ValueError(f"Non-finite pose8 constructed: {p8}")

    return p8
def compute_position_errors_mm(pred_local_m: np.ndarray, meas_local_m: np.ndarray) -> Dict[str, float]:
    e = np.asarray(meas_local_m, dtype=float).reshape(3,) - np.asarray(pred_local_m, dtype=float).reshape(3,)
    return {
        "err_x_mm": float(1e3 * e[0]),
        "err_y_mm": float(1e3 * e[1]),
        "err_z_mm": float(1e3 * e[2]),
        "err_xy_mm": float(1e3 * np.linalg.norm(e[:2])),
        "err_xyz_mm": float(1e3 * np.linalg.norm(e)),
    }

def load_roi_box(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))

def compute_tip_angle_from_base_local_deg(p_base_local_m: np.ndarray) -> float:
    v = np.asarray(p_base_local_m, dtype=float).reshape(3,)
    vx = -v[0]
    vy = v[1]
    return float(np.degrees(np.arctan2(vy, vx)))


# ============================================================
# Forward model
# ============================================================

def build_initial_lumen_from_vision(
    pivot_point,
    image_filename="focused_image.jpg",
    red_roi_path="red_roi_box.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=False,
):
    new_capture()

    vision_result = reconstruct_beam_within_vessel(
        image_filename=image_filename,
        red_roi_path=red_roi_path,
        blue_roi_path=blue_roi_path,
        green_roi_path=green_roi_path,
        pivot_hint=pivot_hint,
        show=show,
    )

    lumen_C_robot_m = transform_local_points_to_robot(
        vision_result["lumen_C_m"],
        pivot_point,
    )
    lumen_R_robot_m = np.asarray(vision_result["lumen_R_m"], float)

    vision_result["lumen_C_robot_m"] = lumen_C_robot_m
    vision_result["lumen_R_robot_m"] = lumen_R_robot_m

    return lumen_C_robot_m, lumen_R_robot_m
def build_dummy_straight_lumen_from_pivot(
    pivot_pose6: np.ndarray,
    length_m: float = 0.12,
    radius_m: float = 0.05,
    n_pts: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    p0_ur = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)

    # Straight lumen along pivot local -x, expressed in robot frame
    t0 = -(R_pivot[:, 0])
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)

    lumen_C = np.array(
        [p0_ur + s * t0 for s in np.linspace(0.0, length_m, n_pts)],
        dtype=float,
    )
    lumen_R = np.full(len(lumen_C), float(radius_m), dtype=float)
    return lumen_C, lumen_R

def transform_local_points_to_robot(
    points_local_m,
    pivot_pose6,
    flip_y: bool = False,
):
    """
    Convert lumen points from the vision/local frame into robot/world frame.

    Assumes points_local_m are expressed in a pivot-attached local frame,
    except for the historical image convention where +y must be flipped.
    """
    P_local = np.asarray(points_local_m, dtype=float).copy()
    if P_local.ndim == 1:
        P_local = P_local.reshape(1, 3)

    # Preserve your existing vision sign convention.
    if flip_y:
        P_local[:, 1] *= -1.0

    R_pivot = pivot_rotation_matrix(pivot_pose6)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(1, 3)

    P_robot = (R_pivot @ P_local.T).T + p_pivot
    return P_robot
def build_forward_model_no_lumen_effect(pivot_pose6: np.ndarray, L0: float, image_filename, red_roi_path, blue_roi_path, green_roi_path, pivot_hint, lumen = False) -> DeterministicForward6D:
    """
    Build the same solver stack as the controller uses, but with a dummy
    straight/wide lumen so the comparison is effectively 'beam theory only'.
    """
    T_ur_pivot = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )
    MAG_YAW_CAL_DEG = -5.0  # try -5 first because physically subtracting joint 5 fixed it

    m_body_nominal = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)
    m_body = rotate_body_xy(m_body_nominal, MAG_YAW_CAL_DEG)

    print("[MAG CAL]")
    print("m_body_nominal =", m_body_nominal)
    print("m_body_calibrated =", m_body)
    print("mag yaw calibration deg =", MAG_YAW_CAL_DEG)
    # m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)
    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    # tip = rod_section_stiffness(
    #     r=2e-3,
    #     E=3e6,
    #     nu=0.49,
    # )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )
    EA_wire = wire["EA"]
    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]

    EA_tip = tip["EA"]
    EI_tip = tip["EI"]
    GJ_tip = tip["GJ"]

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )
    if lumen == False:
        lumen_C, lumen_R =  build_initial_lumen_from_vision(
            pivot_point = pivot_pose6,
            image_filename="focused_image.jpg",
            blue_roi_path="blue_roi_box.json",
            green_roi_path="green_roi_box.json",
            pivot_hint=pivot_hint,
            show=False,
        )
        print(f"LUMEN FALSE")
        

        forward_model = EnergyMinForwardWithLumen(
            p0_ur=p0_ur,
            q0_ur=q0_ur,
            Kinv_fun=Kinv_fun,
            u_star=np.zeros(3),
            m_body=m_body,
            lumen_C=np.asarray(lumen_C, float),
            lumen_R=np.asarray(lumen_R, float),
            N_nodes=5,
            maxiter=1e7,
            L0_init=0.01,
            dL_internal=0.002,
            use_lumen_jac=False,
            L_tip_full=tip_len_model,
            L_tip_min=0.01,
        )
    else:
        lumen_C, lumen_R =  build_initial_lumen_from_vision(
            pivot_point = pivot_pose6,
            image_filename="focused_image.jpg",
            blue_roi_path="blue_roi_box.json",
            green_roi_path="green_roi_box.json",
            pivot_hint=pivot_hint,
            show=False,
        )
        forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        N_nodes=7,
        maxiter=30,
        L0_init=0.01,
        dL_internal=0.004,
        L_tip_full=0.04,
        L_tip_min=0.01,
        use_lumen_jac=True,
    )




    return DeterministicForward6D(forward_model)
def robot_points_to_base_local(points_robot_m, pivot_pose6, beam_base_point_robot_m):
    """
    Convert Nx3 robot-frame points into base-local coordinates.

    base-local = pivot-local coordinates shifted so that the beam base is at the origin.
    """
    P_robot = np.asarray(points_robot_m, dtype=float)
    if P_robot.ndim == 1:
        P_robot = P_robot.reshape(1, 3)

    P_local = np.array(
        [robot_point_to_pivot_local(p, pivot_pose6) for p in P_robot],
        dtype=float,
    )

    base_local = robot_point_to_pivot_local(beam_base_point_robot_m, pivot_pose6)
    return P_local - base_local.reshape(1, 3)
def predict_tip_local_from_model(
    forward6d: DeterministicForward6D,
    pose6: np.ndarray,
    L_m: float,
    pivot_pose6: np.ndarray,
) -> Dict:
    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)
    print(f"Inside prediction pose: {pose6}")
    if hasattr(forward6d, "start_step"):
        forward6d.start_step()

    y_pred_robot = np.asarray(forward6d(p8, commit=False), dtype=float).reshape(-1)
    p_tip_robot = y_pred_robot[:3]
    p_tip_local = robot_point_to_pivot_local(p_tip_robot, pivot_pose6)

    centerline_robot = None
    centerline_pivot_local = None

    if getattr(forward6d, "last_p_centerline", None) is not None:
        C_robot = np.asarray(forward6d.last_p_centerline, dtype=float)   # (3, N)
        centerline_robot = C_robot.copy()

        centerline_pivot_local = np.column_stack([
            robot_point_to_pivot_local(C_robot[:, i], pivot_pose6)
            for i in range(C_robot.shape[1])
        ])  # (3, N)

    out = {
        "pose8": p8.copy(),
        "y_pred_robot": y_pred_robot.copy(),
        "tip_robot_m": p_tip_robot.copy(),
        "tip_local_m": p_tip_local.copy(),
        "centerline_robot_m": centerline_robot,
        "centerline_pivot_local_m": centerline_pivot_local,
    }

    return out
def predicted_tip_to_base_local_from_base_point_robot(
    pred_tip_robot_m: np.ndarray,
    beam_base_point_robot_m: np.ndarray,
    pivot_pose6: np.ndarray,
) -> np.ndarray:
    pred_tip_pivot_local = robot_point_to_pivot_local(pred_tip_robot_m, pivot_pose6)
    beam_base_pivot_local = robot_point_to_pivot_local(beam_base_point_robot_m, pivot_pose6)
    return pred_tip_pivot_local - beam_base_pivot_local


# ============================================================
# Vision
# ============================================================

def build_reference_beam_frame_from_image(
    image_filename: str,
    red_roi_path: str,
    pivot_hint=None,
    show: bool = False,
    show_debug_markers: bool = False,
):
    roi_polygon = load_polygon("/home/jack/Proper-Research/custom_area.json")
    ref_result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=None,
        roi_polygon=roi_polygon,
        show=show,
        show_debug_markers=show_debug_markers,
        unwrap_angle=True,
        pivot_hint=pivot_hint,
    )
    base_px_ref = ref_result["markers"]["base_px"]
    ex_ref = np.asarray(ref_result["beam_frame_fit"]["ex"], dtype=float)
    ey_ref = np.asarray(ref_result["beam_frame_fit"]["ey"], dtype=float)

    return {
        "base_px_ref": base_px_ref,
        "ex_ref": ex_ref,
        "ey_ref": ey_ref,
        "raw_result": ref_result,
    }


def measure_tip_from_vision_base_local(
    cfg: SweepEvalConfig,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
    ) -> Dict:
    import cv2

    image_bgr = cv2.imread(cfg.image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {cfg.image_filename}")

    red_roi_box = load_roi_box(cfg.red_roi_path)
    green_roi_box = load_roi_box(cfg.green_roi_path)

    mm_per_pixel = compute_mm_per_pixel_from_green(
        image_bgr=image_bgr,
        green_roi_box=green_roi_box,
        known_distance_mm=cfg.known_green_distance_mm,
    )
    pivot_hint = (300, 391)
    roi_polygon = load_polygon("/home/jack/Proper-Research/custom_area.json")
    tip_result = measure_tip_state_4markers(
        image_filename=cfg.image_filename,
        roi_box=None,
        roi_polygon=roi_polygon,
        unwrap_angle=True,
        pivot_hint=cfg.pivot_hint,
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
    )

    tip_xy_px = np.asarray(tip_result["tip_xy_from_base"], dtype=float).reshape(2,)
    tip_xy_m = (tip_xy_px * mm_per_pixel) / 1000.0

    tip_base_local_m = np.array([
        -tip_xy_m[0],
        tip_xy_m[1],
        0.0,
    ], dtype=float)

    return {
        "mm_per_pixel": float(mm_per_pixel),
        "tip_result": tip_result,
        "tip_base_local_m": tip_base_local_m.copy(),
    }

# ============================================================
# Motion helper
# ============================================================

def move_robot_to_get_point(hw: LiveHardwareController, i_idx: int, j_idx: int, L0: float) -> np.ndarray:
    
    pose6 = np.asarray(get_point(i_idx, j_idx,base_point, pivot_point), dtype=float).reshape(6,)
    pose6[2] = -0.1

    T = ur_pose6_to_T(pose6)
    p, q_wxyz = T_to_p_quat_wxyz(T)
    p_now = np.concatenate([p, q_wxyz, [L0]])

    u0 = np.zeros(7, dtype=float)
    hw.send_step(p_now=p_now, u0=u0, dt=0.1)

    robot_pose6 = hw.get_robot_pose_once()
    robot_pose6 = np.asarray(robot_pose6, dtype=float).reshape(6,)
    robot_pose6[2] = -0.1
    return robot_pose6


# ============================================================
# Sweep
# ============================================================

def save_results_json(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

def save_results_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return

    flat_rows = []
    for r in rows:
        flat_rows.append({
            "i_idx": r["i_idx"],
            "j_idx": r["j_idx"],
            "L_m": r["L_m"],

            # "cmd_x": r["cmd_pose6"][0],
            # "cmd_y": r["cmd_pose6"][1],
            # "cmd_z": r["cmd_pose6"][2],
            # "cmd_rx": r["cmd_pose6"][3],
            # "cmd_ry": r["cmd_pose6"][4],
            # "cmd_rz": r["cmd_pose6"][5],

            "act_x": r["act_pose6"][0],
            "act_y": r["act_pose6"][1],
            "act_z": r["act_pose6"][2],
            "act_rx": r["act_pose6"][3],
            "act_ry": r["act_pose6"][4],
            "act_rz": r["act_pose6"][5],

            "pred_base_x_mm": 1e3 * r["pred_tip_base_local_m"][0],
            "pred_base_y_mm": 1e3 * r["pred_tip_base_local_m"][1],
            "pred_base_z_mm": 1e3 * r["pred_tip_base_local_m"][2],

            "meas_base_x_mm": 1e3 * r["meas_tip_base_local_m"][0],
            "meas_base_y_mm": 1e3 * r["meas_tip_base_local_m"][1],
            "meas_base_z_mm": 1e3 * r["meas_tip_base_local_m"][2],

            "pred_tip_angle_deg": r["pred_tip_angle_from_ref_deg"],
            "meas_tip_angle_deg": r["meas_tip_angle_from_ref_deg"],

            "err_x_mm": r["err_x_mm"],
            "err_y_mm": r["err_y_mm"],
            "err_z_mm": r["err_z_mm"],
            "err_xy_mm": r["err_xy_mm"],
            "err_xyz_mm": r["err_xyz_mm"],
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
def order_measured_curve_base_to_tip(curve_mm, meas_tip_mm):
    curve_mm = np.asarray(curve_mm, dtype=float).reshape(-1, 3)
    meas_tip_mm = np.asarray(meas_tip_mm, dtype=float).reshape(3,)

    d0 = np.linalg.norm(curve_mm[0] - meas_tip_mm)
    d1 = np.linalg.norm(curve_mm[-1] - meas_tip_mm)

    # If the first point is closer to the tip than the last point,
    # then the curve is tip->base, so reverse it.
    if d0 < d1:
        curve_mm = curve_mm[::-1].copy()

    return curve_mm
def sweep_joint_5_relative(
    hw,
    delta_start_deg=0.0,
    delta_end_deg=90.0,
    step_deg=5.0,
    settle_s=1.0,
    out_json="joint5_sweep_results.json",
):
    q0 = hw.get_robot_joints_once()
    if q0 is None:
        raise RuntimeError("Could not read current joints.")

    q5_start = q0[5]
    results = []

    deltas_deg = np.arange(delta_start_deg, delta_end_deg + 1e-9, step_deg)

    for ddeg in deltas_deg:
        q_cmd = q0.copy()
        q_cmd[5] = q5_start + np.deg2rad(ddeg)

        hw.send_joints(q_cmd, speed=0.3, accel=0.5)
        time.sleep(settle_s)

        q_act = hw.get_robot_joints_once()

        results.append({
            "joint5_delta_cmd_deg": float(ddeg),
            "joint5_cmd_deg": float(np.degrees(q_cmd[5])),
            "joint5_act_deg": float(np.degrees(q_act[5])),
            "all_cmd_deg": np.degrees(q_cmd).tolist(),
            "all_act_deg": np.degrees(q_act).tolist(),
        })

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    return results
def evaluate_sweep(cfg: SweepEvalConfig, hw: LiveHardwareController) -> List[Dict]:
    os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.use_reference_frame:
        ref_frame = build_reference_beam_frame_from_image(
            image_filename=cfg.reference_image_filename,
            red_roi_path=cfg.red_roi_path,
            pivot_hint=cfg.pivot_hint,
            show=False,
            show_debug_markers=False,
        )
        print("\n--- REFERENCE FRAME ---")
        print("base_px_ref =", ref_frame["base_px_ref"])
        print("ex_ref      =", ref_frame["ex_ref"])
        print("ey_ref      =", ref_frame["ey_ref"])
    else:
        ref_frame = None

    forward6d = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),image_filename="focused_image.jpg",
            red_roi_path="red_roi_box.json",
            blue_roi_path="blue_roi_box.json",
            green_roi_path="green_roi_box.json",
            pivot_hint = (318.200927734375, 369.6798095703125), lumen = False
    )

    results = []
    # for j_idx in range(cfg.j_start, cfg.j_end +1, 5):
    #     print("\n====================================")
    #     print(f"SWEEP STEP i={cfg.i_fixed}, j={j_idx}")
    #     print("====================================")

    #     # ------------------------------------------------------------
    #     # Command + move
    #     # ------------------------------------------------------------
    #     q0 = np.array([-0.6708334128009241, -2.5134555302061976, -1.0908429622650146, -1.107914463882782, 1.573858618736267, 2.9142839908599854])
    #     q_cmd = q0.copy()
    #     q5_start = q0[5]
    #     q_cmd[5] = q5_start + np.deg2rad(j_idx)

    #     hw.send_joints(q_cmd, speed=0.3, accel=0.5)
    #     time.sleep(1.0)

    #     q_act = hw.get_robot_joints_once()
    #     act_pose6 = hw.get_robot_pose_once()

    #     print("Commanded joint[5] [deg]:", float(np.degrees(q_cmd[5])))
    #     print("Actual    joint[5] [deg]:", float(np.degrees(q_act[5])))
    for j_idx in range(cfg.j_start, cfg.j_end - 1, -5):
        print("\n====================================")
        print(f"SWEEP STEP i={cfg.i_fixed}, j={j_idx}")
        print("====================================")

        # ------------------------------------------------------------
        # # Command + move
        # ------------------------------------------------------------
        cmd_pose6 = np.asarray(get_point(cfg.i_fixed, j_idx, base_point, pivot_point), dtype=float).reshape(6,)
        cmd_pose6[2] = -0.1

        act_pose6 = move_robot_to_get_point(hw, cfg.i_fixed, j_idx, cfg.L_m)
        
        if cfg.settle_time_s > 0:
            time.sleep(cfg.settle_time_s)

        if cfg.capture_each_step:
            new_capture()
        roi_polygon_path="/home/jack/Proper-Research/custom_area.json"
        roi_polygon = load_polygon(roi_polygon_path)
        # ------------------------------------------------------------
        # Vision reconstruction from image
        # ------------------------------------------------------------
        recon = reconstruct_beam_within_vessel(
            image_filename=cfg.image_filename,
            red_roi_polygon=roi_polygon,
            blue_roi_path="blue_roi_box.json",
            green_roi_path=cfg.green_roi_path,
            pivot_hint=cfg.pivot_hint,
            show=False,
            save_overlay_path=os.path.join(
                cfg.results_dir, f"overlay_j_{j_idx:03d}.png"
            ),
        )

        # Use ONE consistent frame for both measured tip and measured beam
        if ref_frame is not None:
            meas_base_px = ref_frame["base_px_ref"]
            meas_ex = ref_frame["ex_ref"]
            meas_ey = ref_frame["ey_ref"]

            meas = measure_tip_from_vision_base_local(
                cfg,
                base_px_ref=meas_base_px,
                ex_ref=meas_ex,
                ey_ref=meas_ey,
            )
        else:
            meas_base_px = recon["tip_result"]["markers"]["base_px"]
            meas_ex = np.asarray(recon["tip_result"]["beam_frame_fit"]["ex"], dtype=float)
            meas_ey = np.asarray(recon["tip_result"]["beam_frame_fit"]["ey"], dtype=float)

            meas = measure_tip_from_vision_base_local(cfg)

        meas_beam_local_mm = beam_points_px_to_base_local_mm(
            beam_points_px=recon["beam_centerline_px"],
            base_px_ref=meas_base_px,
            ex_ref=meas_ex,
            ey_ref=meas_ey,
            mm_per_pixel=recon["mm_per_pixel"],
        )

        # Reorder measured beam so it is base -> tip
        meas_beam_local_mm = order_measured_curve_base_to_tip(
            meas_beam_local_mm,
            1e3 * np.asarray(meas["tip_base_local_m"], dtype=float),
        )

        # Optional: force exact base anchor to zero for shape plotting
        # meas_beam_local_mm = meas_beam_local_mm - meas_beam_local_mm[0]

        # ------------------------------------------------------------
        # Prediction
        # ------------------------------------------------------------
        pred = predict_tip_local_from_model(
            forward6d=forward6d,
            pose6=act_pose6,
            L_m=float(cfg.L_m),
            pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        )

        pred_centerline_robot_m = extract_predicted_centerline_robot_from_forward(forward6d)

        pred_beam_local_mm = predicted_centerline_robot_to_base_local_mm(
            centerline_robot_m=pred_centerline_robot_m,
            beam_base_point_robot_m=np.asarray(cfg.beam_base_point_robot_m, dtype=float),
            pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        )

        pred_base_local_m = predicted_tip_to_base_local_from_base_point_robot(
            pred_tip_robot_m=pred["tip_robot_m"],
            beam_base_point_robot_m=np.asarray(cfg.beam_base_point_robot_m, dtype=float),
            pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        )

        # ------------------------------------------------------------
        # Errors
        # ------------------------------------------------------------
        pred_tip_angle_deg = compute_tip_angle_from_base_local_deg(pred_base_local_m)

        err = compute_position_errors_mm(
            pred_local_m=pred_base_local_m,
            meas_local_m=meas["tip_base_local_m"],
        )

        # ------------------------------------------------------------
        # Save row
        # ------------------------------------------------------------
        row = {
            "i_idx": int(cfg.i_fixed),
            "j_idx": int(j_idx),
            "L_m": float(cfg.L_m),

            # "cmd_pose6": cmd_pose6.tolist(),
            # "act_pose6": act_pose6.tolist(),
            "act_pose6": None if act_pose6 is None else np.asarray(act_pose6, dtype=float).tolist(),
            "pred_tip_robot_m": pred["tip_robot_m"].tolist(),
            "pred_tip_pivot_local_m": pred["tip_local_m"].tolist(),
            "pred_tip_base_local_m": pred_base_local_m.tolist(),

            "meas_tip_base_local_m": meas["tip_base_local_m"].tolist(),
            "mm_per_pixel": float(meas["mm_per_pixel"]),

            "pred_tip_angle_from_ref_deg": float(pred_tip_angle_deg),
            "meas_tip_angle_from_ref_deg": float(
                meas["tip_result"]["tip_base_angle_from_ref_deg"]
            ),

            # full beam curves for later plotting
            "pred_beam_local_mm": pred_beam_local_mm.tolist(),
            "meas_beam_local_mm": meas_beam_local_mm.tolist(),

            # save the exact frame used for measured beam conversion
            "meas_frame_base_px": np.asarray(meas_base_px, dtype=float).tolist(),
            "meas_frame_ex": np.asarray(meas_ex, dtype=float).tolist(),
            "meas_frame_ey": np.asarray(meas_ey, dtype=float).tolist(),

            # optional raw reconstruction data
            "meas_beam_centerline_px": np.asarray(
                recon["beam_centerline_px"], dtype=float
            ).tolist(),
            "meas_markers": recon["tip_result"]["markers"],
            "beam_poly_coeffs": np.asarray(
                recon["beam_poly_coeffs"], dtype=float
            ).tolist(),

            **err,
        }

        print(
            f"pred [mm] = ({1e3 * pred_base_local_m[0]:.3f}, {1e3 * pred_base_local_m[1]:.3f}) | "
            f"meas [mm] = ({1e3 * meas['tip_base_local_m'][0]:.3f}, {1e3 * meas['tip_base_local_m'][1]:.3f}) | "
            f"err_xy = {err['err_xy_mm']:.3f} mm"
        )

        results.append(row)

    save_results_json(os.path.join(cfg.results_dir, "sweep_results.json"), results)
    save_results_csv(os.path.join(cfg.results_dir, "sweep_results.csv"), results)
    print(f"\nSaved sweep results to: {cfg.results_dir}")
    return results

def beam_points_px_to_base_local_mm(
    beam_points_px,
    base_px_ref,
    ex_ref,
    ey_ref,
    mm_per_pixel,
):
    pts = np.asarray(beam_points_px, dtype=float).reshape(-1, 2)
    base = np.asarray(base_px_ref, dtype=float).reshape(2,)
    ex = np.asarray(ex_ref, dtype=float).reshape(2,)
    ey = np.asarray(ey_ref, dtype=float).reshape(2,)

    ex = ex / (np.linalg.norm(ex) + 1e-12)
    ey = ey / (np.linalg.norm(ey) + 1e-12)

    out = []
    for px, py in pts:
        v = np.array([px - base[0], -(py - base[1])], dtype=float)

        x_mm = -(v @ ex) * mm_per_pixel
        y_mm = -(v @ ey) * mm_per_pixel   # keep same convention as your current script
        out.append([x_mm, y_mm, 0.0])

    return np.asarray(out, dtype=float)
def extract_predicted_centerline_robot_from_forward(forward6d):
    candidates = [getattr(forward6d, "last_p_centerline", None)]

    for subname in ["model", "forward_model", "_model", "_forward_model", "fm"]:
        sub = getattr(forward6d, subname, None)
        if sub is not None:
            candidates.append(getattr(sub, "last_p_centerline", None))
            candidates.append(getattr(sub, "_last", {}).get("p_centerline", None) if hasattr(sub, "_last") else None)

    for c in candidates:
        if c is None:
            continue
        arr = np.asarray(c, dtype=float)

        if arr.ndim == 2 and arr.shape[0] == 3:
            return arr.T.copy()   # convert (3,N) -> (N,3)

        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.copy()

    raise RuntimeError("Could not find predicted full centerline.")
def predicted_centerline_robot_to_base_local_mm(
    centerline_robot_m,
    beam_base_point_robot_m,
    pivot_pose6,
):
    pts = np.asarray(centerline_robot_m, dtype=float).reshape(-1, 3)
    base_local = robot_point_to_pivot_local(
        np.asarray(beam_base_point_robot_m, dtype=float),
        np.asarray(pivot_pose6, dtype=float),
    )

    out = []
    for p_robot in pts:
        p_local = robot_point_to_pivot_local(p_robot, pivot_pose6)
        p_base_local = p_local - base_local
        out.append(1e3 * p_base_local)

    return np.asarray(out, dtype=float)
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    hw = LiveHardwareController(
        robot_ip="192.168.56.101",
        dry_run=False,
        use_advancer=False,
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=0.27,
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )
    L0 = 0.035

    pivot_point = np.array([0.8281328220229531, -0.6812560048066458, -0.1, 3.1374639959012303, 0.13796054585074355, 0.0009377017734513666]
    , float)
    base_point = np.array([
        pivot_point[0] - (L0 + 0.17),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    beam_base_point_robot_m = pivot_point[:3].copy()
 
    cfg = SweepEvalConfig(
        pivot_pose6=pivot_point,
        beam_base_point_robot_m=beam_base_point_robot_m,
        L_m=L0,
        i_fixed=0,
        j_start=0,
        j_end=-90,
        image_filename="focused_image.jpg",
        reference_image_filename="focused_image_straight.jpg",
        use_reference_frame=True,
        red_roi_path="red_roi_box.json",
        green_roi_path="green_roi_box.json",
        known_green_distance_mm=30.0,
        pivot_hint=(321.200927734375, 331.6798095703125),
        results_dir="results_with_lumen_no_drawing_29",
        show_debug_vision=False,
        show_debug_model=False,
        capture_each_step=True,
        settle_time_s=1,
    )

    evaluate_sweep(cfg, hw)