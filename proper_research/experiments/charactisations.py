"""
single_pose_forward_vs_vision.py

Purpose
-------
Validate the static tip-position prediction of the beam theory at ONE pose and ONE insertion length.

Workflow
--------
1. Build a "no-lumen-effect" forward model (straight, wide dummy lumen).
2. Define one test robot pose and one insertion length.
3. Run the same forward model used by the controller.
4. Capture / read one image.
5. Measure the actual tip from vision.
6. Convert both predicted and measured tip positions into the pivot-local frame.
7. Print and save the comparison.

Notes
-----
- This is NOT a sweep script.
- This is NOT a controller script.
- This is a single static model-vs-measurement validation.
- We use a dummy lumen only because your current EnergyMinForwardWithLumen API requires lumen_C and lumen_R.
  The radius is made very large so lumen constraints do not matter.

Outputs
-------
Creates a folder:
    results_single_pose_forward_validation/

Files:
    result_summary.json
    comparison_overlay.png   (if your vision code saves it)
"""
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
import copy
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, WarmForwardP8TipTangent
import time
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple
from proper_research.robot.live_hardware_control import LiveHardwareController, p8_to_ur_pose6_and_L
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
# =========================
# Project imports
# =========================
from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
from proper_research.robot.transformations import get_point

from proper_research.simulation.boundary_forward_model import (
    EnergyMinForwardWithLumen,
    DeterministicForward6D
)
from proper_research.control.lab_ready_mpc import integrate_pose8_body
from proper_research.vision.measure_length import new_capture
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel, get_saved_2_point_calibration

from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from proper_research.parameters import default_magnet_params, default_beam_params

# ---- vision side ----
# Update this import path if your vision utilities live elsewhere.
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
class SinglePoseEvalConfig:
    pivot_pose6: np.ndarray
    test_pose6: np.ndarray
    beam_base_point_robot_m: np.ndarray
    L_m: float

    image_filename: str = "focused_image.jpg"
    reference_image_filename: str = "focused_image_straight.jpg"
    use_reference_frame: bool = True

    red_roi_path: str = "/home/jack/Proper-Research/red_roi_box.json"
    green_roi_path: str = "green_roi_box.json"

    known_green_distance_mm: float = 40.0
    pivot_hint: Tuple[float, float] | None = None

    results_dir: str = "results_single_pose_forward_validation"
    save_overlay_path: str = "results_single_pose_forward_validation/comparison_overlay.png"

    show_debug_vision: bool = True
    show_debug_model: bool = True


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
# ============================================================
# Build "no-lumen-effect" forward model
# ============================================================
def build_initial_lumen_from_vision(
    pivot_point,
    image_filename="focused_image.jpg",
    roi_polygon_path="/home/jack/Proper-Research/custom_area.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=True,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
):
    new_capture()

    roi_polygon = load_polygon(roi_polygon_path)
    print("[DBG lumen frame actually passed through]")
    print("  base_px_ref =", base_px_ref)
    print("  ex_ref =", ex_ref)
    print("  ey_ref =", ey_ref)
    vision_result = reconstruct_beam_within_vessel(
        image_filename=image_filename,
        red_roi_polygon=roi_polygon,
        blue_roi_path=blue_roi_path,
        green_roi_path=green_roi_path,
        pivot_hint=pivot_hint,
        show=show,
        save_overlay_path="debug_outputs/reconstruction_overlay.png",
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
    )


    lumen_C_robot_m = transform_local_points_to_robot(
        vision_result["lumen_C_m"],
        pivot_point,
    )
    lumen_R_robot_m = np.asarray(vision_result["lumen_R_m"], float)

    vision_result["lumen_C_robot_m"] = lumen_C_robot_m
    vision_result["lumen_R_robot_m"] = lumen_R_robot_m

    return lumen_C_robot_m, lumen_R_robot_m
def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
    """
    L_ins      : commanded insertion (what MPC tracks)
    L_tip_full : physical magnetic tip length (4 cm)
    L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

    Returns (L_model, wire_len, tip_len)
    """
    L_ins = float(L_ins)

    # Magnetised tip inside grows with insertion until full tip is inside
    tip_len = min(L_ins, L_tip_full)

    # Wire is everything beyond the physical tip length
    wire_len = max(L_ins - L_tip_full, 0.0)

    # Total model length is the inserted length, but don't go below minimum model length
    L_model = max(L_ins, L_tip_min)

    # If we are below L_tip_min, we still model a minimum rod,
    # but magnetisation should NOT exceed what's actually inserted:
    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len
def make_Kbt_inv_profile(EI_wire, EI_tip, GJ_wire, GJ_tip, bend_soft=1.0, tors_soft=1.0):
    def Kbt_inv_profile(s, len_wire):
        s = np.asarray(s, float)
        mask_tip = (s >= len_wire)

        EI_s = np.where(mask_tip, EI_tip, EI_wire)
        GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)

        Kinv = np.zeros((3, 3, s.size), float)
        Kinv[0, 0, :] = tors_soft / GJ_s
        Kinv[1, 1, :] = bend_soft / EI_s
        Kinv[2, 2, :] = bend_soft / EI_s
        return Kinv

    return Kbt_inv_profile
def rod_section_stiffness(r, E, nu):
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    EA = E * A
    EI = E * I
    GJ = G * J

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": EA,
        "EI": EI,
        "GJ": GJ,
    }
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
def build_forward_model_no_lumen_effect(
    pivot_pose6: np.ndarray,
    L0: float,
    image_filename,
    red_roi_path,
    blue_roi_path,
    green_roi_path,
    pivot_hint,
    lumen=False,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
) -> DeterministicForward6D:
    """
    Build the same solver stack as the controller uses.

    If lumen=True, use lumen reconstructed from vision.
    If lumen=False, still uses the same forward class here, but you can later
    swap in a dummy lumen if you want a true no-lumen comparison.

    IMPORTANT:
    base_px_ref / ex_ref / ey_ref are the fixed straight-reference frame used
    for both measured tip and reconstructed lumen, so all vision-derived points
    stay in one consistent local frame.
    """
    T_ur_pivot = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)

    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )

    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )

    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]
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

    if lumen:
        lumen_C, lumen_R = build_initial_lumen_from_vision(
            pivot_point=pivot_pose6,
            image_filename=image_filename,
            blue_roi_path=blue_roi_path,
            green_roi_path=green_roi_path,
            pivot_hint=pivot_hint,
            show=False,
            base_px_ref=base_px_ref,
            ex_ref=ex_ref,
            ey_ref=ey_ref,
        )
        print("[DBG] using vision lumen in fixed reference frame")
    else:
        lumen_C, lumen_R = build_initial_lumen_from_vision(
            pivot_point=pivot_pose6,
            image_filename=image_filename,
            blue_roi_path=blue_roi_path,
            green_roi_path=green_roi_path,
            pivot_hint=pivot_hint,
            show=False,
            base_px_ref=base_px_ref,
            ex_ref=ex_ref,
            ey_ref=ey_ref,
        )
        print("[DBG] lumen=False branch still using vision lumen")
        # later you can replace this with build_dummy_straight_lumen_from_pivot(...)

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, float),
        lumen_R=np.asarray(lumen_R, float),
        N_nodes=50,
        maxiter=1e7,
        L0_init=0.01,
        dL_internal=0.002,
        use_lumen_jac=False,
        L_tip_full=tip_len_model,
        L_tip_min=0.01,
    )

    return DeterministicForward6D(forward_model)


def numerical_B_y_wrt_u(p8, forward_y_fn, dt, eps_u, n_out):

    p8 = np.asarray(p8, float).ravel()
    B = np.zeros((n_out, 7), float)

    tint = 0.0
    tfwd = 0.0

    for i in range(6):
        du = np.zeros(7); du[i] = eps_u[i]

        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)

        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

        B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

    delta_L = 1e-3
    p_plus = p8.copy();  p_plus[7]  += delta_L
    p_minus = p8.copy(); p_minus[7] -= delta_L

    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
    B[:, 6] = dt * dy_dL



    return B
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

# ============================================================
# Model prediction
# ============================================================

def predict_tip_local_from_model(
    forward6d: WarmForwardP8TipTangent,
    pose6: np.ndarray,
    L_m: float,
    pivot_pose6: np.ndarray,
) -> Dict:
    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)
    print(f"length: {L_m}")
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

# ============================================================
# Vision measurement
# ============================================================

def load_roi_box(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        import json as _json
        data = _json.load(f)
    return (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))


def compute_mm_per_pixel_from_green(image_bgr, green_roi_box, known_distance_mm: float) -> float:
    # green_result = detect_2_green_calibration_points(
    #     image_bgr=image_bgr,
    #     roi_box=green_roi_box,
    #     green_h_low=35,
    #     green_h_high=95,
    #     sat_min=40,
    #     val_min=40,
    #     min_area=3,
    #     max_area=50000,
    #     show_debug=False,
    # )
    green_result = get_saved_2_point_calibration("/home/jack/Proper-Research/calibration_points.json")

    p1, p2 = green_result["points_px"]
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dist_px = np.linalg.norm(p2 - p1)

    if dist_px < 1e-12:
        raise ValueError("Green calibration points are too close.")

    mm_per_pixel = float(known_distance_mm / dist_px)
    return mm_per_pixel
def predicted_tip_to_base_local_from_base_point_robot(
    pred_tip_robot_m: np.ndarray,
    beam_base_point_robot_m: np.ndarray,
    pivot_pose6: np.ndarray,
) -> np.ndarray:
    pred_tip_pivot_local = robot_point_to_pivot_local(pred_tip_robot_m, pivot_pose6)
    beam_base_pivot_local = robot_point_to_pivot_local(beam_base_point_robot_m, pivot_pose6)
    return pred_tip_pivot_local - beam_base_pivot_local
def build_reference_beam_frame_from_image(
    image_filename: str,
    red_roi_path: str,
    pivot_hint=None,
    show: bool = False,
    roi_polygon=None,
    show_debug_markers: bool = False,
    ):
    red_roi_box = load_roi_box(red_roi_path)
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
    cfg: SinglePoseEvalConfig,
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
    print(f"MM pixel: {mm_per_pixel} and known distance {cfg.known_green_distance_mm}")
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
    base_px = np.asarray(tip_result["markers"]["base_px"], dtype=float)
    tip_px = np.asarray(tip_result["markers"]["tip_px"], dtype=float)
    raw_vec_px = tip_px - base_px
    raw_dist_px = np.linalg.norm(raw_vec_px)
    raw_dist_px_x = tip_px[0] - base_px[0]
    raw_dist_px_y = tip_px[1] - base_px[1]
    tip_xy_px = np.asarray(tip_result["tip_xy_from_base"], dtype=float)
    proj_dist_px = np.linalg.norm(tip_xy_px)


    print("\n--- VISION SANITY CHECK ---")
    print("raw base->tip vector [px] =", raw_vec_px)
    print("raw base->tip distance [px] =", raw_dist_px)
    print("projected tip_xy_from_base [px] =", tip_xy_px)
    print("projected tip_xy_from_base [px] in x =", raw_dist_px_x)
    print("projected tip_xy_from_base [px] in y =", raw_dist_px_y)
    print("projected distance [px] =", proj_dist_px)
    print("axial distance [mm] =", tip_xy_px[0] * mm_per_pixel)
    print("lateral distance [mm] =", tip_xy_px[1] * mm_per_pixel)
    print("euclidean distance [mm] =", np.linalg.norm(tip_xy_px) * mm_per_pixel)
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
def plot_jacobian_comparison(J_no, J_yes, results_dir,show=False, filename="jacobian_comparison.png"):
    os.makedirs(results_dir, exist_ok=True)

    row_labels = ["tip_x", "tip_y", "tip_z", "tan_x", "tan_y", "tan_z"]
    col_labels = ["vx", "vy", "vz", "wx", "wy", "wz", "vL"]

    J_diff = J_yes - J_no

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    mats = [J_no, J_yes, J_diff]
    titles = ["Jacobian without lumen", "Jacobian with lumen", "Difference (with - without)"]

    vmax = max(np.max(np.abs(J_no)), np.max(np.abs(J_yes)), 1e-12)
    vmax_diff = max(np.max(np.abs(J_diff)), 1e-12)

    for ax, M, title in zip(axes[:2], mats[:2], titles[:2]):
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2e}", ha="center", va="center", fontsize=8)

    ax = axes[2]
    im = ax.imshow(J_diff, aspect="auto", cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_title(titles[2])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(J_diff.shape[0]):
        for j in range(J_diff.shape[1]):
            ax.text(j, i, f"{J_diff[i, j]:.2e}", ha="center", va="center", fontsize=8)

    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Saved Jacobian comparison to: {out_path}")
def print_jacobian_summary(name, J):
    print(f"\n--- {name} ---")
    print("shape:", J.shape)
    print("max abs entry:", np.max(np.abs(J)))
    print("fro norm:", np.linalg.norm(J, ord="fro"))
    print(J)
# ============================================================
# Single-shot comparison
# ============================================================
def beam_tangent_in_robot_from_pose(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()
    t_body = np.array([-1.0, 0.0, 0.0], float)
    t_robot = R @ t_body
    return t_robot / (np.linalg.norm(t_robot) + 1e-12)
def evaluate_single_pose(cfg: SinglePoseEvalConfig) -> Dict:
    os.makedirs(cfg.results_dir, exist_ok=True)
    print("\n==============================")
    print(" SINGLE-POSE MODEL VALIDATION ")
    print("==============================")
    print("pivot_pose6 =", np.asarray(cfg.pivot_pose6, dtype=float))
    print("test_pose6  =", np.asarray(cfg.test_pose6, dtype=float))
    print("base_point_robot_m =", np.asarray(cfg.beam_base_point_robot_m, dtype=float))
    print("L_m         =", float(cfg.L_m))
    ref_frame = None
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
    fwd_no_lumen = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),
        image_filename=cfg.image_filename,
        red_roi_path=cfg.red_roi_path,
        blue_roi_path="blue_roi_box.json",
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint,
        lumen=False,
        base_px_ref=ref_frame["base_px_ref"] if ref_frame is not None else None,
        ex_ref=ref_frame["ex_ref"] if ref_frame is not None else None,
        ey_ref=ref_frame["ey_ref"] if ref_frame is not None else None,
    )
    dp_robot = np.asarray(cfg.test_pose6[:3], float) - np.asarray(cfg.pivot_pose6[:3], float)
    R_pivot = ur_pose6_to_T(cfg.pivot_pose6)[:3, :3]
    dp_local = R_pivot.T @ dp_robot

    print("\n--- FRAME DEBUG ---")
    print("dp_robot [mm] =", 1e3 * dp_robot)
    print("dp_local [mm] =", 1e3 * dp_local)


    pred = predict_tip_local_from_model(
        forward6d=fwd_no_lumen,
        pose6=np.asarray(cfg.test_pose6, dtype=float),
        L_m=float(cfg.L_m),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )

    if ref_frame is not None:
        meas = measure_tip_from_vision_base_local(
            cfg,
            base_px_ref=ref_frame["base_px_ref"],
            ex_ref=ref_frame["ex_ref"],
            ey_ref=ref_frame["ey_ref"],
        )
    else:
        meas = measure_tip_from_vision_base_local(cfg)

    pred_base_local_m = predicted_tip_to_base_local_from_base_point_robot(
        pred_tip_robot_m=pred["tip_robot_m"],
        beam_base_point_robot_m=np.asarray(cfg.beam_base_point_robot_m, dtype=float),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )
    pred_centerline_base_local_m = None
    if pred.get("centerline_robot_m", None) is not None:
        pred_centerline_base_local_m = robot_points_to_base_local(
            pred["centerline_robot_m"].T,   # robot_points_to_base_local expects (N,3)
            pivot_pose6=cfg.pivot_pose6,
            beam_base_point_robot_m=cfg.beam_base_point_robot_m,
        )
    pred_tip_angle_deg = compute_tip_angle_from_base_local_deg(pred_base_local_m)

    err = compute_position_errors_mm(
        pred_local_m=pred_base_local_m,
        meas_local_m=meas["tip_base_local_m"],
    )

    pred_x_mm = 1e3 * pred_base_local_m[0]
    pred_y_mm = 1e3 * pred_base_local_m[1]
    meas_x_mm = 1e3 * meas["tip_base_local_m"][0]
    meas_y_mm = 1e3 * meas["tip_base_local_m"][1]

    print("\n--- PLANAR DEFLECTION ---")
    print(f"pred chord [mm] = ({pred_x_mm:.3f}, {pred_y_mm:.3f})")
    print(f"meas chord [mm] = ({meas_x_mm:.3f}, {meas_y_mm:.3f})")
    print("pred_tip_angle_from_ref_deg =", pred_tip_angle_deg)
    print("meas_tip_angle_from_ref_deg =", meas["tip_result"]["tip_base_angle_from_ref_deg"])
    
    
    
    result = {
        "pivot_pose6": np.asarray(cfg.pivot_pose6, dtype=float).tolist(),
        "test_pose6": np.asarray(cfg.test_pose6, dtype=float).tolist(),
        "base_point_robot_m": np.asarray(cfg.beam_base_point_robot_m, dtype=float).tolist(),
        "L_m": float(cfg.L_m),
        "pred_tip_angle_from_ref_deg": float(pred_tip_angle_deg),
        "meas_tip_angle_from_ref_deg": float(meas["tip_result"]["tip_base_angle_from_ref_deg"]),
        "pred_tip_robot_m": pred["tip_robot_m"].tolist(),
        "pred_tip_pivot_local_m": pred["tip_local_m"].tolist(),
        "pred_tip_base_local_m": pred_base_local_m.tolist(),

        "meas_tip_base_local_m": meas["tip_base_local_m"].tolist(),
        "mm_per_pixel": float(meas["mm_per_pixel"]),

        **err,
    }

    out_json = os.path.join(cfg.results_dir, "result_summary.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    def unit(v):
        v = np.asarray(v, float)
        return v / (np.linalg.norm(v) + 1e-12)

    m_robot = source_dipole_in_robot(cfg.test_pose6)
    t_robot = beam_tangent_in_robot_from_pose(cfg.test_pose6)

    print("source dipole dir in robot =", m_robot)
    print("beam tangent dir in robot  =", t_robot)
    print("alignment cos(theta)       =", float(np.dot(m_robot, t_robot)))

    m_local = unit(m_robot)
    t_local = unit(t_robot)

    print("cos robot =", np.dot(m_robot, t_robot))
    print("cos local =", np.dot(m_local, t_local))
    print("\n--- PREDICTED TIP ---")
    print("robot [m]:", pred["tip_robot_m"])
    print("pivot-local [m]:", pred["tip_local_m"])
    print("base-local [m]:", pred_base_local_m)
    print("base-local [mm]:", 1e3 * pred_base_local_m)

    print("\n--- MEASURED TIP ---")
    print("base-local [m]:", meas["tip_base_local_m"])
    print("base-local [mm]:", 1e3 * meas["tip_base_local_m"])

    print("\n--- ERROR ---")
    for k, v in err.items():
        print(f"{k}: {v:.4f}")

    src_base_local_m = robot_point_to_pivot_local(
        np.asarray(cfg.test_pose6[:3], dtype=float),
        np.asarray(cfg.pivot_pose6, dtype=float),
    )

    # if beam base is exactly the pivot, this is already the correct base-local point
    # otherwise subtract the true base point expressed in the same frame:
    base_local_m = robot_point_to_pivot_local(
        np.asarray(cfg.beam_base_point_robot_m, dtype=float),
        np.asarray(cfg.pivot_pose6, dtype=float),
    )
    src_base_local_m = src_base_local_m - base_local_m

    src_dir_robot = source_dipole_in_robot(cfg.test_pose6)
    src_dir_local = robot_dir_to_pivot_local(src_dir_robot, cfg.pivot_pose6)

    pred_base_local_m_plot = pred_base_local_m.copy()
    meas_base_local_m_plot = meas["tip_base_local_m"].copy()
    src_base_local_plot_m = src_base_local_m.copy()
    print(f"TEST POSE {cfg.test_pose6}")
    src_local_check = robot_pose_to_pivot_local_pose(cfg.test_pose6, cfg.pivot_pose6)



    r_test = Rot.from_rotvec(np.asarray(cfg.test_pose6[3:6], float))
    r_pivot = Rot.from_rotvec(np.asarray(cfg.pivot_pose6[3:6], float))

    r_rel = r_pivot.inv() * r_test
    rel_rotvec = r_rel.as_rotvec()
    rel_angle_deg = np.degrees(np.linalg.norm(rel_rotvec))

    print("relative rotvec =", rel_rotvec)
    print("relative angle [deg] =", rel_angle_deg)
    print(f"Pivot POSE {cfg.pivot_pose6}")
    print("source position from robot pose in pivot local [mm] =", 1e3 * src_local_check)
    R_pivot = ur_pose6_to_T(cfg.pivot_pose6)[:3, :3]

    src_pos_robot = np.asarray(cfg.test_pose6[:3], float)
    src_pos_local = robot_point_to_pivot_local(src_pos_robot, cfg.pivot_pose6)

    m_robot = source_dipole_in_robot(cfg.test_pose6)
    m_local = robot_dir_to_pivot_local(m_robot, cfg.pivot_pose6)

    pivot_x_local = np.array([1.0, 0.0, 0.0])
    pivot_neg_x_local = np.array([-1.0, 0.0, 0.0])
    T_pivot = ur_pose6_to_T(cfg.pivot_pose6)
    R_pivot = T_pivot[:3, :3]

    dp_robot = np.asarray(cfg.test_pose6[:3]) - np.asarray(cfg.pivot_pose6[:3])
    x_pivot_robot = R_pivot[:, 0]

    dp_robot_u = dp_robot / (np.linalg.norm(dp_robot) + 1e-12)
    x_pivot_robot_u = x_pivot_robot / (np.linalg.norm(x_pivot_robot) + 1e-12)

    print("dp_robot unit =", dp_robot_u)
    print("pivot x-axis  =", x_pivot_robot_u)
    print("dot(dp_robot_u, pivot x-axis) =", np.dot(dp_robot_u, x_pivot_robot_u))
    print("\n--- CONSISTENT LOCAL DEBUG ---")
    print("src_pos_local [mm] =", 1e3 * src_pos_local)
    print("m_local =", m_local / (np.linalg.norm(m_local) + 1e-12))
    print("dot(m_local, +x_local) =", float(np.dot(m_local, pivot_x_local)))
    print("dot(m_local, -x_local) =", float(np.dot(m_local, pivot_neg_x_local)))
    print("\n--- DIPOLE DEBUG ---")
    print("m_robot =", m_robot)
    print("src_dir_local =", src_dir_local)
    print("difference =", src_dir_local - m_robot)
    print("xy angle in robot/local [deg] =",
        np.degrees(np.arctan2(m_robot[1], m_robot[0])),
        np.degrees(np.arctan2(src_dir_local[1], src_dir_local[0])))
    pivot_m_robot = source_dipole_in_robot(cfg.pivot_pose6)
    test_m_robot = source_dipole_in_robot(cfg.test_pose6)

    print("pivot dipole xy angle [deg] =", np.degrees(np.arctan2(pivot_m_robot[1], pivot_m_robot[0])))
    print("test  dipole xy angle [deg] =", np.degrees(np.arctan2(test_m_robot[1], test_m_robot[0])))
    print("pivot dipole =", pivot_m_robot)
    print("test  dipole =", test_m_robot)
    # Rebuild the same lumen used by the forward model, for plotting
    if True:  # set to False if you want dummy lumen instead
        lumen_C_robot_m, lumen_R_robot_m = build_initial_lumen_from_vision(
            pivot_point=cfg.pivot_pose6,
            image_filename=cfg.image_filename,
            blue_roi_path="blue_roi_box.json",
            green_roi_path=cfg.green_roi_path,
            pivot_hint=cfg.pivot_hint,
            show=False,
            base_px_ref=ref_frame["base_px_ref"] if ref_frame is not None else None,
            ex_ref=ref_frame["ex_ref"] if ref_frame is not None else None,
            ey_ref=ref_frame["ey_ref"] if ref_frame is not None else None,
        )
    else:
        lumen_C_robot_m, lumen_R_robot_m = build_dummy_straight_lumen_from_pivot(
            pivot_pose6=cfg.pivot_pose6,
            length_m=0.16,
            radius_m=0.05,
            n_pts=160,
        )

    lumen_C_base_local_m = robot_points_to_base_local(
        lumen_C_robot_m,
        pivot_pose6=cfg.pivot_pose6,
        beam_base_point_robot_m=cfg.beam_base_point_robot_m,
    )
    import time

    t = time.perf_counter()
    plot_single_tip_comparison_local(
        pred_local_m=pred_base_local_m_plot,
        meas_local_m=meas_base_local_m_plot,
        results_dir=cfg.results_dir,
        filename="tip_comparison_base_local.png",
        src_local_m=src_base_local_plot_m,
        src_dir_local=src_dir_local,
        lumen_C_local_m=lumen_C_base_local_m,
        lumen_R_m=lumen_R_robot_m,
        beam_centerline_local_m=pred_centerline_base_local_m,
    )
    print(f"[TIME] tip plot: {(time.perf_counter()-t):.2f} s")

    t = time.perf_counter()
    fwd_no_lumen = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),
        image_filename=cfg.image_filename,
        red_roi_path=cfg.red_roi_path,
        blue_roi_path="blue_roi_box.json",
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint,
        lumen=False,
    )
    print(f"[TIME] build no-lumen model: {(time.perf_counter()-t):.2f} s")

    t = time.perf_counter()
    fwd_with_lumen = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),
        image_filename=cfg.image_filename,
        red_roi_path=cfg.red_roi_path,
        blue_roi_path="blue_roi_box.json",
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint,
        lumen=False,
    )
    print(f"[TIME] build with-lumen model: {(time.perf_counter()-t):.2f} s")

    # t = time.perf_counter()
    # J_no, p8_eval, eps_u = compute_model_jacobian(
    #     fwd_no_lumen, cfg.test_pose6, cfg.L_m, dt=0.01
    # )
    # print(f"[TIME] J_no: {(time.perf_counter()-t):.2f} s")

    # t = time.perf_counter()
    # J_yes, _, _ = compute_model_jacobian(
    #     fwd_with_lumen, cfg.test_pose6, cfg.L_m, dt=0.01
    # )
    # print(f"[TIME] J_yes: {(time.perf_counter()-t):.2f} s")

    # t = time.perf_counter()
    # plot_jacobian_comparison(J_no, J_yes, cfg.results_dir)
    # print(f"[TIME] jacobian plot: {(time.perf_counter()-t):.2f} s")
    # print_jacobian_summary("Jacobian without lumen", J_no)
    # print_jacobian_summary("Jacobian with lumen", J_yes)
    # print_jacobian_summary("Jacobian difference", J_yes - J_no)

    # save_jacobian_tables(J_no, J_yes, cfg.results_dir)
    # plot_jacobian_comparison(J_no, J_yes, cfg.results_dir)
    print(f"\nSaved summary to: {out_json}")
    return result

def save_jacobian_tables(J_no, J_yes, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "J_no_lumen.npy"), J_no)
    np.save(os.path.join(results_dir, "J_with_lumen.npy"), J_yes)
    np.save(os.path.join(results_dir, "J_diff.npy"), J_yes - J_no)

    np.savetxt(os.path.join(results_dir, "J_no_lumen.csv"), J_no, delimiter=",")
    np.savetxt(os.path.join(results_dir, "J_with_lumen.csv"), J_yes, delimiter=",")
    np.savetxt(os.path.join(results_dir, "J_diff.csv"), J_yes - J_no, delimiter=",")
# ============================================================
# Main
# ============================================================

def make_initial_poses_single_use(hw) -> tuple[np.ndarray, np.ndarray, float, float]:
    L0 = 0.04
    pivot_point = np.array([
    0.8281328220229531, -0.6812731669220016, -0.1,  np.pi, 0.001,0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L0 + 0.14),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    # start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    # start_point[2] = -0.1

    # pose6=start_point
    pose6 = hw.get_robot_pose_once()
    # pose6 = np.asarray(get_point(0, 20), dtype=float)
    pose6[2] = -0.1
    # pose6[0] = 0.3
    print(f"POSE6 is {pose6}")
    T = ur_pose6_to_T(pose6)
    p, q_wxyz = T_to_p_quat_wxyz(T)

    p_now = np.concatenate([p, q_wxyz, [L0]])
    # ur_pose6_next, _ = p8_to_ur_pose6_and_L(p_now)

    # print("original pose6:", pose6)
    # print("recovered pose6:", ur_pose6_next)
    # print("difference:", ur_pose6_next - pose6)
    # print("z_offset:", hw.z_offset)
    # print("final sent pose:", np.array([*ur_pose6_next[:2], ur_pose6_next[2] + hw.z_offset, *ur_pose6_next[3:]]))
    u0 = np.zeros(7, dtype=float)
    hw.send_step(p_now=p_now, u0=u0, dt=0.01)
    # start_point = np.array([
    # 0.665894307606053, -0.7112810117612073, -0.1, np.pi, 0,0
    # ], float)
    robot_pose6 = hw.get_robot_pose_once()
    # robot_pose6[2] = -0.1
    # # # pose6[2] = -0.1
    start_point = robot_pose6
    print(f"START POINT: {start_point}")
    # L0 = 0.065
    dt =0.01


    return pivot_point, start_point, L0, dt
def compute_tip_angle_from_base_local_deg(p_base_local_m: np.ndarray) -> float:
    """
    Angle of base->tip chord relative to straight local -x axis.
    Returns signed angle in degrees.

    Convention:
      straight beam along local -x gives 0 deg
      negative local y gives negative angle
      positive local y gives positive angle
    """
    v = np.asarray(p_base_local_m, dtype=float).reshape(3,)
    vx = -v[0]   # straight reference magnitude along local -x
    vy = v[1]

    return float(np.degrees(np.arctan2(vy, vx)))
def plot_single_tip_comparison_local(
    pred_local_m: np.ndarray,
    meas_local_m: np.ndarray,
    results_dir: str,
    filename: str = "tip_comparison_local.png",
    src_local_m: np.ndarray | None = None,
    src_dir_local: np.ndarray | None = None,
    lumen_C_local_m: np.ndarray | None = None,
    lumen_R_m: np.ndarray | None = None,
    beam_centerline_local_m: np.ndarray | None = None,
):
    pred_local_m = np.asarray(pred_local_m, dtype=float).reshape(3,)
    meas_local_m = np.asarray(meas_local_m, dtype=float).reshape(3,)

    pred_mm = 1e3 * pred_local_m
    meas_mm = 1e3 * meas_local_m

    plt.figure(figsize=(7, 7))

    # pivot / base-local origin
    plt.plot(0.0, 0.0, "ko", label="Base / pivot-local origin")

    # predicted / measured tip
    plt.plot(pred_mm[0], pred_mm[1], "bs", markersize=2, label="Predicted tip")
    plt.plot(meas_mm[0], meas_mm[1], "ro", markersize=2, label="Measured tip")

    # connect predicted and measured
    plt.plot(
        [pred_mm[0], meas_mm[0]],
        [pred_mm[1], meas_mm[1]],
        "k--",
        alpha=0.7,
        label="Error vector",
    )

    # annotate tips
    plt.annotate(
        f"Pred\n({pred_mm[0]:.2f}, {pred_mm[1]:.2f}) mm",
        (pred_mm[0], pred_mm[1]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    plt.annotate(
        f"Meas\n({meas_mm[0]:.2f}, {meas_mm[1]:.2f}) mm",
        (meas_mm[0], meas_mm[1]),
        textcoords="offset points",
        xytext=(8, -18),
    )
    # lumen centerline
    if lumen_C_local_m is not None:
        lumen_C_local_m = np.asarray(lumen_C_local_m, dtype=float)
        if lumen_C_local_m.ndim == 1:
            lumen_C_local_m = lumen_C_local_m.reshape(1, 3)

        lumen_mm = 1e3 * lumen_C_local_m
        plt.plot(
            lumen_mm[:, 0],
            lumen_mm[:, 1],
            "-",
            linewidth=2.0,
            label="Lumen centerline",
        )

        # Optional: show start/end markers
        plt.plot(
            lumen_mm[0, 0], lumen_mm[0, 1],
            "o", markersize=6, label="Lumen start"
        )
        plt.plot(
            lumen_mm[-1, 0], lumen_mm[-1, 1],
            "x", markersize=8, label="Lumen end"
        )
        if lumen_R_m is not None and len(lumen_R_m) == len(lumen_C_local_m):
            lumen_R_m = np.asarray(lumen_R_m, dtype=float).reshape(-1)

            # Build approximate normal in the plot plane
            tangents = np.zeros_like(lumen_C_local_m)
            tangents[1:-1] = lumen_C_local_m[2:] - lumen_C_local_m[:-2]
            tangents[0] = lumen_C_local_m[1] - lumen_C_local_m[0]
            tangents[-1] = lumen_C_local_m[-1] - lumen_C_local_m[-2]

            normals = np.zeros_like(tangents)
            for i, t in enumerate(tangents):
                tx, ty = t[0], t[1]
                n = np.array([-ty, tx, 0.0], dtype=float)
                nn = np.linalg.norm(n[:2])
                if nn > 1e-12:
                    normals[i] = n / nn

            upper = lumen_C_local_m + normals * lumen_R_m[:, None]
            lower = lumen_C_local_m - normals * lumen_R_m[:, None]

            upper_mm = 1e3 * upper
            lower_mm = 1e3 * lower

            plt.plot(upper_mm[:, 0], upper_mm[:, 1], "--", alpha=0.6, label="Lumen wall")
            plt.plot(lower_mm[:, 0], lower_mm[:, 1], "--", alpha=0.6)
        # # predicted beam centerline
        # if beam_centerline_local_m is not None:
        #     beam_centerline_local_m = np.asarray(beam_centerline_local_m, dtype=float)
        #     if beam_centerline_local_m.ndim == 2 and beam_centerline_local_m.shape[1] == 3:
        #         beam_mm = 1e3 * beam_centerline_local_m
        #         plt.plot(
        #             beam_mm[:, 0],
        #             beam_mm[:, 1],
        #             "-b",
        #             linewidth=2.5,
        #             label="Predicted beam centerline",
        #         )

    # if src_local_m is not None:
    #     src_local_m = np.asarray(src_local_m, dtype=float).reshape(3,)
    #     src_mm = 1e3 * src_local_m

    #     plt.plot(src_mm[0], src_mm[1], "md", markersize=10, label="External magnet")

    #     plt.annotate(
    #         f"Mag\n({src_mm[0]:.1f}, {src_mm[1]:.1f}) mm",
    #         (src_mm[0], src_mm[1]),
    #         textcoords="offset points",
    #         xytext=(8, 8),
    #     )

    #     # optional dipole direction arrow
    #     if src_dir_local is not None:
    #         src_dir_local = np.asarray(src_dir_local, dtype=float).reshape(3,)
    #         dxy = src_dir_local[:2]
    #         n = np.linalg.norm(dxy)
    #         if n > 1e-12:
    #             dxy = dxy / n
    #             arrow_len_mm = 25.0
    #             plt.arrow(
    #                 src_mm[0],
    #                 src_mm[1],
    #                 arrow_len_mm * dxy[0],
    #                 arrow_len_mm * dxy[1],
    #                 head_width=3.0,
    #                 head_length=5.0,
    #                 length_includes_head=True,
    #             )

    plt.xlabel("Local x [mm]")
    plt.ylabel("Local y [mm]")
    plt.title("Predicted vs measured tip with external magnet")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved local tip plot to: {out_path}")

def snapshot_forward_cache(fwd_model):
    snap = {}
    if hasattr(fwd_model, "_last"):
        snap["_last"] = copy.deepcopy(fwd_model._last)
    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if hasattr(fwd_model, name):
            snap[name] = copy.deepcopy(getattr(fwd_model, name))
    return snap

def restore_forward_cache(fwd_model, snap):
    if "_last" in snap and hasattr(fwd_model, "_last"):
        fwd_model._last = copy.deepcopy(snap["_last"])
    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if name in snap and hasattr(fwd_model, name):
            setattr(fwd_model, name, copy.deepcopy(snap[name]))

def quat_from_yaw_wxyz(dpsi):
    c = np.cos(0.5 * dpsi)
    s = np.sin(0.5 * dpsi)
    return np.array([c, 0.0, 0.0, s], dtype=float)
def J_full_from_robot_reduced_tip_tangent(J_red, n_out_full=5):
    """
    J_red shape: (5,4)
        columns = [x, y, yaw_z, L]

    Returns full robot Jacobian in 7 controls:
        [vx, vy, vz, wx, wy, wz, dL]
    """
    J_full = np.zeros((n_out_full, 7), float)
    J_full[:, 0] = J_red[:, 0]   # x translation
    J_full[:, 1] = J_red[:, 1]   # y translation
    J_full[:, 5] = J_red[:, 2]   # yaw about z
    J_full[:, 6] = J_red[:, 3]   # insertion
    return J_full
def numerical_J_robot_xy_yaw_dL_warm_branch(
    p8,
    forward_model,   # the actual warm wrapper object, not just a plain fn
    dx=5e-3,
    dy=5e-3,
    dyaw=np.deg2rad(5.0),
    dL=1e-3,
    n_out=5,         # e.g. [tip_x, tip_y, tip_z, tx, ty]
):
    t0 = time.perf_counter()

    p8 = np.asarray(p8, float).ravel().copy()
    J = np.zeros((n_out, 4), float)

    def eval_y_from_p8(p):
        y = np.asarray(forward_model(p), float).reshape(-1)
        return y[:n_out]

    # First solve nominal point ONCE to establish current branch
    y0 = eval_y_from_p8(p8)

    # Snapshot warm branch state at nominal point
    snap0 = snapshot_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model)

    # 1) x
    p_plus = p8.copy();  p_plus[0] += dx
    p_minus = p8.copy(); p_minus[0] -= dx

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 0] = (y_plus - y_minus) / (2.0 * dx)

    # 2) y
    p_plus = p8.copy();  p_plus[1] += dy
    p_minus = p8.copy(); p_minus[1] -= dy

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 1] = (y_plus - y_minus) / (2.0 * dy)

    # 3) yaw
    p_plus = p8.copy()
    p_minus = p8.copy()

    q = quat_wxyz_normalize(p8[3:7])
    dqz_plus = quat_from_yaw_wxyz(+dyaw)
    dqz_minus = quat_from_yaw_wxyz(-dyaw)

    p_plus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_plus, q))
    p_minus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_minus, q))

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 2] = (y_plus - y_minus) / (2.0 * dyaw)

    # 4) L
    p_plus = p8.copy();  p_plus[7] += dL
    p_minus = p8.copy(); p_minus[7] -= dL

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 3] = (y_plus - y_minus) / (2.0 * dL)

    t1 = time.perf_counter()
    print(f"[TIME] numerical_J_robot_warm_branch total: {(t1-t0)*1e3:.2f} ms")
    return J
def compute_model_jacobian(forward6d, pose6, L_m, dt=0.01,
                           dr=5e-3, dtheta_deg=50.0, dL=1e-3, n_out=6):
    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        np.deg2rad(dtheta_deg) / dt,
        np.deg2rad(dtheta_deg) / dt,
        np.deg2rad(dtheta_deg) / dt,
        dL / dt
    ], dtype=float)

    if hasattr(forward6d, "start_step"):
        forward6d.start_step()

    J = numerical_J_robot_xy_yaw_dL_warm_branch(
        p8=p8,
        forward_y_fn=lambda pp: forward6d(pp, commit=False),
        dt=dt,
        eps_u=eps_u,
        n_out=n_out,
    )
    return J, p8, eps_u
def build_forward_model(
    pivot_pose6,
    L0,
    pivot_hint,
    use_lumen,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
):
    return build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(pivot_pose6, dtype=float),
        L0=float(L0),
        image_filename="focused_image.jpg",
        red_roi_path="red_roi_box.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",
        pivot_hint=pivot_hint,
        lumen=use_lumen,
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
    )
def source_dipole_in_robot(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()
    m_body_dir = np.array([1.0, 0.0, 0.0], float)
    m_robot = R @ m_body_dir
    return m_robot / (np.linalg.norm(m_robot) + 1e-12)

def beam_tangent_in_robot_from_pose(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()

    # straight beam direction in tool/body frame
    t_body = np.array([-1.0, 0.0, 0.0], float)

    t_robot = R @ t_body
    return t_robot / (np.linalg.norm(t_robot) + 1e-12)
if __name__ == "__main__":


    hw = LiveHardwareController(
        robot_ip="192.168.56.101",
        dry_run=False,                 # True first
        use_advancer=True,
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
    pivot_point2, start_point2, L0_default, dt = make_initial_poses_single_use(hw)

    # pivot_point2, start_point2, L0_default, dt = make_initial_poses_single_use()
    new_capture()

    # src_local = np.array([-0.183, 0.0, 0.0], dtype=float)
    # src_robot = pivot_local_point_to_robot(src_local, pivot_point2)
    # src_local_check = robot_point_to_pivot_local(src_robot, pivot_point2)

    # print("desired source in pivot-local [mm] =", 1e3 * src_local)
    # print("converted source in robot frame [mm] =", 1e3 * src_robot)
    # print("back-converted source in pivot-local [mm] =", 1e3 * src_local_check)
    test_pose6 = np.asarray(start_point2, dtype=float).reshape(6,)
    # test_pose6[:3] = src_robot
    src_local_check = robot_pose_to_pivot_local_pose(test_pose6, pivot_point2)
    print("source position from robot pose in pivot local [mm] =", 1e3 * src_local_check)

    L_test = float(L0_default)
    
    m_robot = source_dipole_in_robot(test_pose6)
    t_robot = beam_tangent_in_robot_from_pose(test_pose6)
    print("source dipole dir in robot =", m_robot)
    print("beam axis dir in robot     =", t_robot)
    print("alignment cos(theta)       =", float(np.dot(m_robot, t_robot)))

    # IMPORTANT:
    # Replace this with the real beam base point in robot coordinates.
    # This is a 3D point, not a pose6.
    beam_base_point_robot_m = pivot_point2[:3]
    pivot_hint = (325, 371)
    cfg = SinglePoseEvalConfig(
        pivot_pose6=np.asarray(pivot_point2, dtype=float),
        test_pose6=test_pose6,
        beam_base_point_robot_m=beam_base_point_robot_m,
        L_m=L_test,
        image_filename="focused_image.jpg",
        reference_image_filename="focused_image_straight.jpg",
        use_reference_frame=True,
        red_roi_path="/home/jack/Proper-Research/red_roi_box.json",
        green_roi_path="green_roi_box.json",
        known_green_distance_mm=19.00,
        pivot_hint=pivot_hint,
        results_dir="results_single_pose_forward_validation_back",
        save_overlay_path="results_single_pose_forward_validation/comparison_back.png",
        show_debug_vision=False,
        show_debug_model=False,
    )

    evaluate_single_pose(cfg)