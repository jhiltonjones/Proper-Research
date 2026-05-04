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

from proper_research.simulation.boundary_forward_model import ContactParams
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
import copy
from proper_research.simulation.boundary_forward_model import (
    EnergyMinForwardWithAnalyticJac,
    effective_lengths,
    EnergyMinForwardWithLumen,
    DeterministicForward6D,
    WarmForwardP8TipTangent,)
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

from proper_research.control.lab_ready_mpc import pose8_quat_to_pose7_rotvec ,integrate_pose8_body, analytic_J_robot_xy_yaw_dL
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
from proper_research.vision.detect_blue import load_manual_vessel_boundaries_with_frame, build_lumen_from_manual_boundaries_with_frame

MANUAL_VESSEL_BOUNDARY_FILE = "/home/jack/Proper-Research/manual_vessel_boundaries.json"

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

def debug_raw_forward_position_sensitivity(warm_fwd, p8_nominal):
    raw_fwd = warm_fwd.fwd

    p7 = pose8_quat_to_pose7_rotvec(p8_nominal)

    p7_x = p7.copy()
    p7_y = p7.copy()
    p7_z = p7.copy()

    p7_x[0] += 0.05
    p7_y[1] += 0.05
    p7_z[2] += 0.05

    # Reset caches between calls so warm-start does not obscure anything.
    if hasattr(warm_fwd, "reset_cache"):
        warm_fwd.reset_cache()

    y0 = np.asarray(raw_fwd(p7), float).reshape(-1)

    if hasattr(warm_fwd, "reset_cache"):
        warm_fwd.reset_cache()

    yx = np.asarray(raw_fwd(p7_x), float).reshape(-1)

    if hasattr(warm_fwd, "reset_cache"):
        warm_fwd.reset_cache()

    yy = np.asarray(raw_fwd(p7_y), float).reshape(-1)

    if hasattr(warm_fwd, "reset_cache"):
        warm_fwd.reset_cache()

    yz = np.asarray(raw_fwd(p7_z), float).reshape(-1)

    # print("\n==============================")
    # print(" RAW FORWARD POSITION TEST ")
    # print("==============================")
    # print("p7 nominal =", p7)
    # print("p7_x       =", p7_x)
    # print("p7_y       =", p7_y)
    # print("p7_z       =", p7_z)

    # print("\ny0 =", y0)
    # print("yx =", yx)
    # print("yx - y0 =", yx - y0)
    # print("yy =", yy)
    # print("yy - y0 =", yy - y0)
    # print("yz =", yz)
    # print("yz - y0 =", yz - y0)
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
    manual_boundary_path=MANUAL_VESSEL_BOUNDARY_FILE,
    pivot_hint=None,
    show=False,
):
    new_capture()

    manual = load_manual_vessel_boundaries_with_frame(manual_boundary_path)

    # Use your clicked base/origin and +x axis
    base_px_ref = manual["base_px"]
    ex_ref = manual["ex_img"]
    ey_ref = manual["ey_img"]

    print("[DBG lumen manual frame used]")
    print("  base_px_ref =", base_px_ref)
    print("  ex_ref =", ex_ref)
    print("  ey_ref =", ey_ref)

    roi_polygon = load_polygon(roi_polygon_path)

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

    # The lumen_C_m should now already be in the clicked local frame:
    # base click = (0,0), clicked reference direction = +x
    lumen_C_m = np.asarray(vision_result["lumen_C_m"], float).copy()

    # Force centerline start to local origin
    lumen_C_m -= lumen_C_m[0:1, :]

    lumen_C_robot_m = transform_local_points_to_robot(
        lumen_C_m,
        pivot_point,
    )

    lumen_R_robot_m = np.asarray(vision_result["lumen_R_m"], float)

    lumen_base_robot_m = transform_local_points_to_robot(
        np.array([[0.0, 0.0, 0.0]], dtype=float),
        pivot_point,
    )[0]

    return lumen_C_robot_m, lumen_R_robot_m, lumen_base_robot_m
# def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
#     """
#     L_ins      : commanded insertion (what MPC tracks)
#     L_tip_full : physical magnetic tip length (4 cm)
#     L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

#     Returns (L_model, wire_len, tip_len)
#     """
#     L_ins = float(L_ins)

#     # Magnetised tip inside grows with insertion until full tip is inside
#     tip_len = min(L_ins, L_tip_full)

#     # Wire is everything beyond the physical tip length
#     wire_len = max(L_ins - L_tip_full, 0.0)

#     # Total model length is the inserted length, but don't go below minimum model length
#     L_model = max(L_ins, L_tip_min)

#     # If we are below L_tip_min, we still model a minimum rod,
#     # but magnetisation should NOT exceed what's actually inserted:
#     tip_len = min(tip_len, L_model)

#     return L_model, wire_len, tip_len
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
    MAG_YAW_CAL_DEG = 5  # try -5 first because physically subtracting joint 5 fixed it

    m_body_nominal = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)
    m_body = rotate_body_xy(m_body_nominal, MAG_YAW_CAL_DEG)

    print("[MAG CAL]")
    print("m_body_nominal =", m_body_nominal)
    print("m_body_calibrated =", m_body)
    print("mag yaw calibration deg =", MAG_YAW_CAL_DEG)
    # m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)

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
        lumen_C, lumen_R ,_= build_initial_lumen_from_vision(
            pivot_point=pivot_pose6,
            image_filename=image_filename,
            blue_roi_path=blue_roi_path,
            green_roi_path=green_roi_path,
            pivot_hint=pivot_hint,
            show=False,

        )
        # print("[DBG] using vision lumen in fixed reference frame")
        # print("[LUMEN LOCAL DEBUG]")
        # print("lumen_C_m[0] [mm] =", 1e3 * lumen_C[0])
        # print("min/max x [mm] =", 1e3 * np.min(lumen_C[:, 0]), 1e3 * np.max(lumen_C[:, 0]))
        # print("min/max y [mm] =", 1e3 * np.min(lumen_C[:, 1]), 1e3 * np.max(lumen_C[:, 1]))
    else:
        lumen_C, lumen_R,_ = build_initial_lumen_from_vision(
            pivot_point=pivot_pose6,
            image_filename=image_filename,
            blue_roi_path=blue_roi_path,
            green_roi_path=green_roi_path,
            pivot_hint=pivot_hint,
            show=False,

        )
        print("[DBG] lumen=False branch still using vision lumen")
        # later you can replace this with build_dummy_straight_lumen_from_pivot(...)

    # forward_model = EnergyMinForwardWithLumen(
    #     p0_ur=p0_ur,
    #     q0_ur=q0_ur,
    #     Kinv_fun=Kinv_fun,
    #     u_star=np.zeros(3),
    #     m_body=m_body,
    #     lumen_C=np.asarray(lumen_C, float),
    #     lumen_R=np.asarray(lumen_R, float),
    #     N_nodes=5,
    #     maxiter=1e7,
    #     L0_init=0.01,
    #     dL_internal=0.002,
    #     use_lumen_jac=True,
    #     L_tip_full=tip_len_model,
    #     L_tip_min=0.01,
    # )
    contact = ContactParams(
        r_beam=beam_params.r,   # or 0.001, but use one value everywhere
        k=1e8,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=3,
    )

    forward_model = EnergyMinForwardWithAnalyticJac(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, float),
        lumen_R=np.asarray(lumen_R, float),
        N_nodes=10,
        maxiter=30,
        L0_init=0.01,
        dL_internal=0.04,
        use_lumen_jac=True,

        # Use physical full tip length, not current inserted tip length
        L_tip_full=0.04,
        L_tip_min=0.01,

        # New canonical contact parameter object
        contact_params=contact,
        use_fast_contact_grad=False,
    )
    return WarmForwardP8TipTangent(forward_model)


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


    # print("\n--- VISION SANITY CHECK ---")
    # print("raw base->tip vector [px] =", raw_vec_px)
    # print("raw base->tip distance [px] =", raw_dist_px)
    # print("projected tip_xy_from_base [px] =", tip_xy_px)
    # print("projected tip_xy_from_base [px] in x =", raw_dist_px_x)
    # print("projected tip_xy_from_base [px] in y =", raw_dist_px_y)
    # print("projected distance [px] =", proj_dist_px)
    # print("axial distance [mm] =", tip_xy_px[0] * mm_per_pixel)
    # print("lateral distance [mm] =", tip_xy_px[1] * mm_per_pixel)
    # print("euclidean distance [mm] =", np.linalg.norm(tip_xy_px) * mm_per_pixel)
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
            show=True,
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
        lumen=True,
        base_px_ref=ref_frame["base_px_ref"] if ref_frame is not None else None,
        ex_ref=ref_frame["ex_ref"] if ref_frame is not None else None,
        ey_ref=ref_frame["ey_ref"] if ref_frame is not None else None,
    )
    p8_test = pose6_and_L_to_pose8_quat(cfg.test_pose6, cfg.L_m)
    debug_raw_forward_position_sensitivity(fwd_no_lumen, p8_test)
    p8_test = pose6_and_L_to_pose8_quat(cfg.test_pose6, cfg.L_m)

    fwd_debug = copy.deepcopy(fwd_no_lumen)

    y0 = np.asarray(fwd_debug(p8_test, commit=True), float).reshape(-1)[:6]

    p8_x = p8_test.copy()
    p8_x[0] += 0.05  # 50 mm, deliberately huge

    y_x = np.asarray(fwd_debug(p8_x, commit=False), float).reshape(-1)[:6]

    p8_y = p8_test.copy()
    p8_y[1] += 0.05  # 50 mm, deliberately huge

    y_y = np.asarray(fwd_debug(p8_y, commit=False), float).reshape(-1)[:6]

    p8_z = p8_test.copy()
    p8_z[2] += 0.05  # 50 mm, deliberately huge

    y_z = np.asarray(fwd_debug(p8_z, commit=False), float).reshape(-1)[:6]

    print("\n==============================")
    print(" DIRECT SOURCE POSITION TEST ")
    print("==============================")
    print("p8 nominal[:3] =", p8_test[:3])
    print("p8_x[:3]       =", p8_x[:3])
    print("p8_y[:3]       =", p8_y[:3])
    print("p8_z[:3]       =", p8_z[:3])

    print("\ny0 =", y0)
    print("y_x =", y_x)
    print("y_x - y0 =", y_x - y0)
    print("y_y =", y_y)
    print("y_y - y0 =", y_y - y0)
    print("y_z =", y_z)
    print("y_z - y0 =", y_z - y0)
    dp_robot = np.asarray(cfg.test_pose6[:3], float) - np.asarray(cfg.pivot_pose6[:3], float)
    R_pivot = ur_pose6_to_T(cfg.pivot_pose6)[:3, :3]
    dp_local = R_pivot.T @ dp_robot

    # print("\n--- FRAME DEBUG ---")
    # print("dp_robot [mm] =", 1e3 * dp_robot)
    # print("dp_local [mm] =", 1e3 * dp_local)


    pred = predict_tip_local_from_model(
        forward6d=fwd_no_lumen,
        pose6=np.asarray(cfg.test_pose6, dtype=float),
        L_m=float(cfg.L_m),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )
    jac_check = compare_analytic_vs_fd_jacobian(
        forward6d=fwd_no_lumen,
        pose6=np.asarray(cfg.test_pose6, dtype=float),
        L_m=float(cfg.L_m),
        dt=0.01,
        dx=5e-3,
        dy=5e-3,
        dyaw=np.deg2rad(5.0),
        dL=1e-3,
        n_out=3,
        results_dir=cfg.results_dir,
        name="with_lumen_after_forward_solution",
    )
    # if ref_frame is not None:
    #     meas = measure_tip_from_vision_base_local(
    #         cfg,
    #         base_px_ref=ref_frame["base_px_ref"],
    #         ex_ref=ref_frame["ex_ref"],
    #         ey_ref=ref_frame["ey_ref"],
    #     )
    # else:
    #     meas = measure_tip_from_vision_base_local(cfg)
    manual = load_manual_vessel_boundaries_with_frame(MANUAL_VESSEL_BOUNDARY_FILE)

    meas = measure_tip_from_vision_base_local(
        cfg,
        base_px_ref=manual["base_px"],
        ex_ref=manual["ex_img"],
        ey_ref=manual["ey_img"],
    )
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

        "jacobian_max_abs_err_state": jac_check["max_abs_err_state"],
        "jacobian_mean_abs_err_state": jac_check["mean_abs_err_state"],
        "jacobian_max_rel_err_state": jac_check["max_rel_err_state"],
        "jacobian_mean_rel_err_state": jac_check["mean_rel_err_state"],

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

    # print("dp_robot unit =", dp_robot_u)
    # print("pivot x-axis  =", x_pivot_robot_u)
    # print("dot(dp_robot_u, pivot x-axis) =", np.dot(dp_robot_u, x_pivot_robot_u))
    # print("\n--- CONSISTENT LOCAL DEBUG ---")
    # print("src_pos_local [mm] =", 1e3 * src_pos_local)
    # print("m_local =", m_local / (np.linalg.norm(m_local) + 1e-12))
    # print("dot(m_local, +x_local) =", float(np.dot(m_local, pivot_x_local)))
    # print("dot(m_local, -x_local) =", float(np.dot(m_local, pivot_neg_x_local)))
    # print("\n--- DIPOLE DEBUG ---")
    # print("m_robot =", m_robot)
    # print("src_dir_local =", src_dir_local)
    # print("difference =", src_dir_local - m_robot)
    # print("xy angle in robot/local [deg] =",
    #     np.degrees(np.arctan2(m_robot[1], m_robot[0])),
    #     np.degrees(np.arctan2(src_dir_local[1], src_dir_local[0])))
    pivot_m_robot = source_dipole_in_robot(cfg.pivot_pose6)
    test_m_robot = source_dipole_in_robot(cfg.test_pose6)

    print("pivot dipole xy angle [deg] =", np.degrees(np.arctan2(pivot_m_robot[1], pivot_m_robot[0])))
    print("test  dipole xy angle [deg] =", np.degrees(np.arctan2(test_m_robot[1], test_m_robot[0])))
    print("pivot dipole =", pivot_m_robot)
    print("test  dipole =", test_m_robot)
    # Rebuild the same lumen used by the forward model, for plotting
    if True:  # set to False if you want dummy lumen instead
        lumen_C_robot_m, lumen_R_robot_m, lumen_base_robot_m = build_initial_lumen_from_vision(
            pivot_point=cfg.pivot_pose6,
            image_filename=cfg.image_filename,
            blue_roi_path="blue_roi_box.json",
            green_roi_path=cfg.green_roi_path,
            pivot_hint=cfg.pivot_hint,
            show=False,

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
        beam_base_point_robot_m=lumen_base_robot_m,
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
        beam_radius_m= beam_params.r
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
        lumen=True,
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
def smooth_lumen_local(lumen_C_m, lumen_R_m, n=300, smooth=1e-7):
    from scipy.interpolate import splprep, splev

    C = np.asarray(lumen_C_m, float)
    R = np.asarray(lumen_R_m, float).reshape(-1)

    keep = np.r_[True, np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1) > 1e-9]
    C = C[keep]
    R = R[keep]

    if len(C) < 4:
        return C, R

    ds = np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1)
    s_old = np.r_[0.0, np.cumsum(ds)]
    s_old /= s_old[-1]

    tck, _ = splprep([C[:, 0], C[:, 1]], u=s_old, s=smooth, k=min(3, len(C) - 1))

    s_new = np.linspace(0.0, 1.0, n)
    x_new, y_new = splev(s_new, tck)

    z_new = np.interp(s_new, s_old, C[:, 2])
    R_new = np.interp(s_new, s_old, R)

    return np.column_stack([x_new, y_new, z_new]), R_new


def offset_walls_from_centerline(C_m, R_m):
    C = np.asarray(C_m, float)
    R = np.asarray(R_m, float).reshape(-1)

    dC = np.gradient(C[:, :2], axis=0)
    tangent = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-12)

    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    upper = C.copy()
    lower = C.copy()

    upper[:, :2] = C[:, :2] + normal * R[:, None]
    lower[:, :2] = C[:, :2] - normal * R[:, None]

    return upper, lower
def make_initial_poses_single_use(hw) -> tuple[np.ndarray, np.ndarray, float, float]:
    L0 = 0.0301
    pivot_point = np.array([
    0.8281328220229531, -0.6812731669220016, -0.1,  np.pi, 0.001,0.001
    ], float)
    base_point = np.array([
        pivot_point[0] - (L0 + 0.13),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)
    beam_base_point_robot_m = pivot_point[:3].copy()

    start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    start_point[2] = -0.1

    pose6=start_point
    # pose6 = hw.get_robot_pose_once()
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
    # hw.send_step(p_now=p_now, u0=u0, dt=0.01)
    # start_point = np.array([
    # 0.665894307606053, -0.7112810117612073, -0.1, np.pi, 0,0
    # ], float)
    robot_pose6 = hw.get_robot_pose_once()
    robot_pose6[2] = -0.1
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
    beam_radius_m: float = 1e-3,   # 2 mm diameter beam
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
        lumen_R_m = np.asarray(lumen_R_m, dtype=float).reshape(-1)

        if lumen_C_local_m.ndim == 1:
            lumen_C_local_m = lumen_C_local_m.reshape(1, 3)

        lumen_C_plot_m, lumen_R_plot_m = smooth_lumen_local(
            lumen_C_local_m,
            lumen_R_m,
            n=500,
            smooth=0,
        )

        lumen_mm = 1e3 * lumen_C_plot_m
        plt.plot(lumen_mm[:, 0], lumen_mm[:, 1], "-", linewidth=2.0, label="Lumen centerline")

        upper, lower = offset_walls_from_centerline(lumen_C_plot_m, lumen_R_plot_m)

        upper_mm = 1e3 * upper
        lower_mm = 1e3 * lower

        plt.plot(upper_mm[:, 0], upper_mm[:, 1], "--", alpha=0.6, label="Lumen wall")
        plt.plot(lower_mm[:, 0], lower_mm[:, 1], "--", alpha=0.6)

        plt.plot(lumen_mm[0, 0], lumen_mm[0, 1], "o", markersize=6, label="Lumen start")
        plt.plot(lumen_mm[-1, 0], lumen_mm[-1, 1], "x", markersize=8, label="Lumen end")
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

            # plt.plot(upper_mm[:, 0], upper_mm[:, 1], "--", alpha=0.6, label="Lumen wall")
            # plt.plot(lower_mm[:, 0], lower_mm[:, 1], "--", alpha=0.6)
        # predicted beam centerline
    # predicted beam centerline + physical beam envelope
        if beam_centerline_local_m is not None:
            beam_centerline_local_m = np.asarray(beam_centerline_local_m, dtype=float)

            if beam_centerline_local_m.ndim != 2:
                raise ValueError("beam_centerline_local_m must be 2D")

            if beam_centerline_local_m.shape[1] == 3:
                beam_xyz_m = beam_centerline_local_m
            elif beam_centerline_local_m.shape[0] == 3:
                beam_xyz_m = beam_centerline_local_m.T
            else:
                raise ValueError(
                    f"Expected beam_centerline_local_m shape (N,3) or (3,N), "
                    f"got {beam_centerline_local_m.shape}"
                )

            # Build normals in local x-y plot plane
            tangents = np.zeros_like(beam_xyz_m)
            tangents[1:-1] = beam_xyz_m[2:] - beam_xyz_m[:-2]
            tangents[0] = beam_xyz_m[1] - beam_xyz_m[0]
            tangents[-1] = beam_xyz_m[-1] - beam_xyz_m[-2]

            normals = np.zeros_like(beam_xyz_m)
            for i, tvec in enumerate(tangents):
                tx, ty = tvec[0], tvec[1]
                n = np.array([-ty, tx, 0.0], dtype=float)
                nn = np.linalg.norm(n[:2])
                if nn > 1e-12:
                    normals[i] = n / nn

            beam_upper_m = beam_xyz_m + beam_radius_m * normals
            beam_lower_m = beam_xyz_m - beam_radius_m * normals

            beam_mm = 1e3 * beam_xyz_m
            beam_upper_mm = 1e3 * beam_upper_m
            beam_lower_mm = 1e3 * beam_lower_m

            # Filled 2D beam body
            beam_poly_x = np.r_[beam_upper_mm[:, 0], beam_lower_mm[::-1, 0]]
            beam_poly_y = np.r_[beam_upper_mm[:, 1], beam_lower_mm[::-1, 1]]

            plt.fill(
                beam_poly_x,
                beam_poly_y,
                alpha=0.25,
                label=f"Predicted beam body, diameter={2e3 * beam_radius_m:.1f} mm",
            )

            # Beam edges
            plt.plot(
                beam_upper_mm[:, 0],
                beam_upper_mm[:, 1],
                "-b",
                linewidth=1.0,
                alpha=0.8,
                label="Predicted beam edge",
            )
            plt.plot(
                beam_lower_mm[:, 0],
                beam_lower_mm[:, 1],
                "-b",
                linewidth=1.0,
                alpha=0.8,
            )

            # Centerline
            plt.plot(
                beam_mm[:, 0],
                beam_mm[:, 1],
                "-b",
                linewidth=2.0,
                label="Predicted beam centerline",
            )

            plt.plot(
                beam_mm[0, 0],
                beam_mm[0, 1],
                "bo",
                markersize=4,
                label="Predicted beam base",
            )

            plt.plot(
                beam_mm[-1, 0],
                beam_mm[-1, 1],
                "bs",
                markersize=5,
                label="Predicted beam tip",
            )
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

    if src_local_m is not None:
        src_local_m = np.asarray(src_local_m, dtype=float).reshape(3,)
        src_mm = 1e3 * src_local_m

        plt.plot(src_mm[0], src_mm[1], "md", markersize=10, label="External magnet")

        plt.annotate(
            f"Mag\n({src_mm[0]:.1f}, {src_mm[1]:.1f}) mm",
            (src_mm[0], src_mm[1]),
            textcoords="offset points",
            xytext=(8, 8),
        )

        # optional dipole direction arrow
        if src_dir_local is not None:
            src_dir_local = np.asarray(src_dir_local, dtype=float).reshape(3,)
            dxy = src_dir_local[:2]
            n = np.linalg.norm(dxy)
            if n > 1e-12:
                dxy = dxy / n
                arrow_len_mm = 25.0
                plt.arrow(
                    src_mm[0],
                    src_mm[1],
                    arrow_len_mm * dxy[0],
                    arrow_len_mm * dxy[1],
                    head_width=3.0,
                    head_length=5.0,
                    length_includes_head=True,
                )

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
def reset_energy_forward_cache(raw_fwd):
    if hasattr(raw_fwd, "_last"):
        raw_fwd._last = dict(
            p=None,
            L=None,
            u_init=None,
            tip=None,
            p_centerline=None,
            info=None,
            hist=None,
        )

    for name in ["last_info", "last_p_centerline", "last_tip", "last_hist"]:
        if hasattr(raw_fwd, name):
            setattr(raw_fwd, name, None)


def numerical_J_raw_pose7_tip(
    raw_fwd,
    p7,
    dx=5e-3,
    dy=5e-3,
    dz=5e-3,
    drz=np.deg2rad(5.0),
    dL=1e-3,
    n_out=3,
):
    """
    Finite-difference Jacobian of the raw full-coordinate forward model.

    p7 = [x, y, z, rx, ry, rz, L]

    Columns:
        [x, y, z, rz, L]
    """
    p7 = np.asarray(p7, float).reshape(7,)
    J = np.zeros((n_out, 5), float)

    def eval_y(p7_eval, label=""):
        reset_energy_forward_cache(raw_fwd)

        y = np.asarray(raw_fwd(p7_eval), float).reshape(-1)
        y = y[:n_out]

        # print(f"\n[RAW FD {label}]")
        # print("p7 =", p7_eval)
        # print("y  =", y)

        return y

    # x
    pp = p7.copy()
    pm = p7.copy()
    pp[0] += dx
    pm[0] -= dx
    yp = eval_y(pp, "x plus")
    ym = eval_y(pm, "x minus")
    print("x diff =", yp - ym)
    J[:, 0] = (yp - ym) / (2.0 * dx)

    # y
    pp = p7.copy()
    pm = p7.copy()
    pp[1] += dy
    pm[1] -= dy
    yp = eval_y(pp, "y plus")
    ym = eval_y(pm, "y minus")
    print("y diff =", yp - ym)
    J[:, 1] = (yp - ym) / (2.0 * dy)

    # z
    pp = p7.copy()
    pm = p7.copy()
    pp[2] += dz
    pm[2] -= dz
    yp = eval_y(pp, "z plus")
    ym = eval_y(pm, "z minus")
    print("z diff =", yp - ym)
    J[:, 2] = (yp - ym) / (2.0 * dz)

    # simple rotvec-z perturbation
    pp = p7.copy()
    pm = p7.copy()
    pp[5] += drz
    pm[5] -= drz
    yp = eval_y(pp, "rz plus")
    ym = eval_y(pm, "rz minus")
    print("rz diff =", yp - ym)
    J[:, 3] = (yp - ym) / (2.0 * drz)

    # L
    pp = p7.copy()
    pm = p7.copy()
    pp[6] += dL
    pm[6] -= dL
    yp = eval_y(pp, "L plus")
    ym = eval_y(pm, "L minus")
    print("L diff =", yp - ym)
    J[:, 4] = (yp - ym) / (2.0 * dL)

    print("\n--- RAW POSE7 FD JACOBIAN ---")
    print("columns = [x, y, z, rz, L]")
    print(J)

    return J
def numerical_J_robot_xy_yaw_dL_warm_branch(
    p8,
    forward_model,   # the actual warm wrapper object, not just a plain fn
    dx=1e-2,
    dy=1e-2,
    dyaw=np.deg2rad(5.0),
    dL=1e-3,
    n_out=5,         # e.g. [tip_x, tip_y, tip_z, tx, ty]
):
    


    def eval_y_from_p8(p, label=""):
        y = np.asarray(forward_model(p), float).reshape(-1)

        print(f"\n[FD EVAL {label}]")
        print("input p[:3] =", p[:3])
        print("input q     =", p[3:7])
        print("input L     =", p[7])
        print("output y    =", y[:n_out])

        fwd_obj = forward_model.fwd if hasattr(forward_model, "fwd") else forward_model
        info = getattr(fwd_obj, "last_info", None)

        if info is not None:
            print("last_info keys =", list(info.keys()))
            for k, v in info.items():
                kl = k.lower()
                if any(s in kl for s in ["src", "source", "mag", "pose", "p_", "x", "field", "dipole"]):
                    print(f"last_info[{k}] =", v)

        return y[:n_out]
    t0 = time.perf_counter()

    p8 = np.asarray(p8, float).ravel().copy()
    J = np.zeros((n_out, 4), float)

    # def eval_y_from_p8(p):
    #     y = np.asarray(forward_model(p), float).reshape(-1)
    #     return y[:n_out]

    # First solve nominal point ONCE to establish current branch
    y0 = eval_y_from_p8(p8)

    # Snapshot warm branch state at nominal point
    snap0 = snapshot_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model)

    # 1) x
    # 1) x
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[0] += dx
    p_minus[0] -= dx

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus, label="x plus")
    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus, label="x minus")

    # print("\n[FD DEBUG x]")
    # print("p8[0] nominal:", p8[0])
    # print("p_plus[0]:", p_plus[0])
    # print("p_minus[0]:", p_minus[0])
    # print("y_plus:", y_plus)
    # print("y_minus:", y_minus)
    # print("y_plus - y_minus:", y_plus - y_minus)

    J[:, 0] = (y_plus - y_minus) / (2.0 * dx)


    # 2) y
    p_plus = p8.copy();  p_plus[1] += dy
    p_minus = p8.copy(); p_minus[1] -= dy

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)
    # print("\n[FD DEBUG y]")
    # print("p8[1] nominal:", p8[1])
    # print("p_plus[1]:", p_plus[1])
    # print("p_minus[1]:", p_minus[1])
    # print("y_plus:", y_plus)
    # print("y_minus:", y_minus)
    # print("y_plus - y_minus:", y_plus - y_minus)
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
MAG_YAW_CAL_DEG = 5
def clone_warm_forward_model(forward6d):
    """
    Make an independent warm forward model for Jacobian tests.

    This avoids contaminating the main prediction object's warm cache.
    """
    return copy.deepcopy(forward6d)

def compare_analytic_vs_fd_jacobian(
    forward6d,
    pose6,
    L_m,
    dt=0.01,
    dx=1e-2,
    dy=1e-2,
    dyaw=3e-1,
    dL=1e-3,
    n_out=6,
    results_dir=None,
    name="jacobian_check",
):
    """
    Compare analytic Jacobian and central finite-difference Jacobian.

    Reduced state Jacobian:
        columns = [x, y, yaw_z, L]
        shape   = (n_out, 4)

    Full control Jacobian:
        columns = [vx, vy, vz, wx, wy, wz, vL]
        shape   = (n_out, 7)

    Mapping:
        delta_x   = vx * dt
        delta_y   = vy * dt
        delta_yaw = wz * dt
        delta_L   = vL * dt

    Therefore:
        J_control = dt * J_state
    """
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)

    # Use separate copies so the analytic and FD evaluations do not contaminate
    # each other's warm-start cache.


    # ============================================================
    # Analytic / hybrid Jacobian
    # ============================================================
    t0 = time.perf_counter()

    fwd_analytic = copy.deepcopy(forward6d)
    fwd_fd = copy.deepcopy(forward6d)

    if hasattr(fwd_analytic, "start_step"):
        fwd_analytic.start_step()

    # Commit nominal solve once so analytic_J_robot_xy_yaw_dL can use last_info.
    y_nom = np.asarray(fwd_analytic(p8, commit=True), float).reshape(-1)[:n_out]

    if hasattr(fwd_analytic, "fwd") and getattr(fwd_analytic.fwd, "last_info", None) is not None:
        print("[ANALYTIC] last_info keys =", fwd_analytic.fwd.last_info.keys())
    else:
        print("[WARNING] fwd_analytic.fwd.last_info is None before analytic Jacobian")

    J_an_state = hybrid_J_robot_xy_yaw_dL(
        p8=p8,
        fwd_analytic=fwd_analytic,
        fwd_fd=fwd_fd,
        n_out=n_out,
        dL_fd=max(abs(dL), 3e-4),  # or simply 1e-3
    )

    print(f"[TIME] analytic/hybrid reduced Jacobian: {(time.perf_counter() - t0) * 1e3:.2f} ms")

    # ============================================================
    # Finite-difference Jacobian
    # ============================================================
    # ============================================================
    # Finite-difference Jacobian from raw pose7 forward model
    # ============================================================
    t0 = time.perf_counter()

    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)
    p7 = pose8_quat_to_pose7_rotvec(p8)

    J_fd_pose7 = numerical_J_raw_pose7_tip(
        raw_fwd=fwd_fd.fwd,
        p7=p7,
        dx=dx,
        dy=dy,
        dz=5e-3,
        drz=dyaw,
        dL=dL,
        n_out=3,
    )

    # J_fd_pose7 columns are [x, y, z, rz, L].
    # Convert to reduced columns [x, y, yaw_z, L].
    J_fd_state = np.column_stack([
        J_fd_pose7[:, 0],
        J_fd_pose7[:, 1],
        J_fd_pose7[:, 3],
        J_fd_pose7[:, 4],
    ])

    # For now compare tip position only.
    J_an_state = np.asarray(J_an_state, float)[:3, :]
    J_fd_state = np.asarray(J_fd_state, float)
    n_out_cmp = 3

    print(f"[TIME] finite-difference reduced Jacobian: {(time.perf_counter() - t0) * 1e3:.2f} ms")
    # ============================================================
    # Shape checks
    # ============================================================
    expected_shape = (3, 4)

    if J_an_state.shape != expected_shape:
        raise ValueError(
            f"Analytic reduced Jacobian has shape {J_an_state.shape}, "
            f"expected {expected_shape}"
        )

    if J_fd_state.shape != expected_shape:
        raise ValueError(
            f"FD reduced Jacobian has shape {J_fd_state.shape}, "
            f"expected {expected_shape}"
        )

    # ============================================================
    # Reduced-state comparison
    # ============================================================
    J_err_state = J_an_state - J_fd_state
    abs_err_state = np.abs(J_err_state)

    rel_denom_state = np.maximum(np.abs(J_fd_state), 1e-12)
    rel_err_state = abs_err_state / rel_denom_state

    print("\n==============================")
    print(" ANALYTIC VS FD JACOBIAN CHECK ")
    print("==============================")

    print("\n--- nominal output y ---")
    print(y_nom)

    print("\n--- J_fd_state: dy/d[x, y, yaw_z, L] ---")
    print(J_fd_state)

    print("\n--- J_an_state: dy/d[x, y, yaw_z, L] ---")
    print(J_an_state)

    print("\n--- J_an_state - J_fd_state ---")
    print(J_err_state)

    print("\n--- reduced-state error summary ---")
    print("max abs error:", float(np.max(abs_err_state)))
    print("mean abs error:", float(np.mean(abs_err_state)))
    print("fro abs error:", float(np.linalg.norm(J_err_state, ord="fro")))
    print("max relative error:", float(np.max(rel_err_state)))
    print("mean relative error:", float(np.mean(rel_err_state)))

    row_labels = ["tip_x", "tip_y", "tip_z"]
    col_labels_red = ["x", "y", "yaw_z", "L"]

    print("\n--- per-entry reduced comparison ---")
    for i in range(3):
        for j in range(4):
            print(
                f"{row_labels[i]:>8s} wrt {col_labels_red[j]:>5s}: "
                f"analytic={J_an_state[i, j]: .6e}, "
                f"fd={J_fd_state[i, j]: .6e}, "
                f"abs_err={J_err_state[i, j]: .6e}, "
                f"rel_err={rel_err_state[i, j]: .6e}"
            )

    # ============================================================
    # Full control Jacobians
    # ============================================================
    J_an_full = J_full_from_robot_reduced_tip_tangent(
        dt * J_an_state,
        n_out_full=n_out,
    )

    J_fd_full = J_full_from_robot_reduced_tip_tangent(
        dt * J_fd_state,
        n_out_full=n_out,
    )

    J_err_full = J_an_full - J_fd_full

    print("\n--- full control Jacobian comparison ---")
    print("columns = [vx, vy, vz, wx, wy, wz, vL]")

    print("\nJ_fd_full:")
    print(J_fd_full)

    print("\nJ_an_full:")
    print(J_an_full)

    print("\nJ_an_full - J_fd_full:")
    print(J_err_full)

    print("\n--- full-control error summary ---")
    print("max abs error:", float(np.max(np.abs(J_err_full))))
    print("mean abs error:", float(np.mean(np.abs(J_err_full))))
    print("fro abs error:", float(np.linalg.norm(J_err_full, ord="fro")))

    # ============================================================
    # Save files
    # ============================================================
    if results_dir is not None:
        np.save(os.path.join(results_dir, f"{name}_J_fd_state.npy"), J_fd_state)
        np.save(os.path.join(results_dir, f"{name}_J_an_state.npy"), J_an_state)
        np.save(os.path.join(results_dir, f"{name}_J_err_state.npy"), J_err_state)

        np.save(os.path.join(results_dir, f"{name}_J_fd_full.npy"), J_fd_full)
        np.save(os.path.join(results_dir, f"{name}_J_an_full.npy"), J_an_full)
        np.save(os.path.join(results_dir, f"{name}_J_err_full.npy"), J_err_full)

        np.savetxt(os.path.join(results_dir, f"{name}_J_fd_state.csv"), J_fd_state, delimiter=",")
        np.savetxt(os.path.join(results_dir, f"{name}_J_an_state.csv"), J_an_state, delimiter=",")
        np.savetxt(os.path.join(results_dir, f"{name}_J_err_state.csv"), J_err_state, delimiter=",")

        np.savetxt(os.path.join(results_dir, f"{name}_J_fd_full.csv"), J_fd_full, delimiter=",")
        np.savetxt(os.path.join(results_dir, f"{name}_J_an_full.csv"), J_an_full, delimiter=",")
        np.savetxt(os.path.join(results_dir, f"{name}_J_err_full.csv"), J_err_full, delimiter=",")

        summary = {
            "name": name,
            "dt": float(dt),
            "dx": float(dx),
            "dy": float(dy),
            "dyaw": float(dyaw),
            "dL": float(dL),
            "n_out": int(n_out),
            "p8": p8.tolist(),
            "y_nom": y_nom.tolist(),
            "max_abs_err_state": float(np.max(abs_err_state)),
            "mean_abs_err_state": float(np.mean(abs_err_state)),
            "fro_abs_err_state": float(np.linalg.norm(J_err_state, ord="fro")),
            "max_rel_err_state": float(np.max(rel_err_state)),
            "mean_rel_err_state": float(np.mean(rel_err_state)),
            "max_abs_err_full": float(np.max(np.abs(J_err_full))),
            "mean_abs_err_full": float(np.mean(np.abs(J_err_full))),
            "fro_abs_err_full": float(np.linalg.norm(J_err_full, ord="fro")),
        }

        with open(os.path.join(results_dir, f"{name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        plot_analytic_vs_fd_jacobian(
            J_an=J_an_state,
            J_fd=J_fd_state,
            results_dir=results_dir,
            filename=f"{name}_reduced_state.png",
            title_prefix="Reduced dy/d[x,y,yaw,L]",
        )

        plot_analytic_vs_fd_jacobian(
            J_an=J_an_full,
            J_fd=J_fd_full,
            results_dir=results_dir,
            filename=f"{name}_full_control.png",
            title_prefix="Full dy/d[vx,vy,vz,wx,wy,wz,vL]",
        )

        print(f"\nSaved Jacobian comparison files to: {results_dir}")

    return {
        "p8": p8,
        "y_nom": y_nom,
        "J_fd_state": J_fd_state,
        "J_an_state": J_an_state,
        "J_err_state": J_err_state,
        "J_fd_full": J_fd_full,
        "J_an_full": J_an_full,
        "J_err_full": J_err_full,
        "max_abs_err_state": float(np.max(abs_err_state)),
        "mean_abs_err_state": float(np.mean(abs_err_state)),
        "max_rel_err_state": float(np.max(rel_err_state)),
        "mean_rel_err_state": float(np.mean(rel_err_state)),
    }

def source_dipole_in_robot(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()

    m_body_dir = np.array([1.0, 0.0, 0.0], float)
    m_body_dir = rotate_body_xy(m_body_dir, MAG_YAW_CAL_DEG)

    m_robot = R @ m_body_dir
    return m_robot / (np.linalg.norm(m_robot) + 1e-12)

# def source_dipole_in_robot(pose6_robot):
#     rvec = np.asarray(pose6_robot[3:6], float)
#     R = Rot.from_rotvec(rvec).as_matrix()
#     m_body_dir = np.array([1.0, 0.0, 0.0], float)
#     m_robot = R @ m_body_dir
#     return m_robot / (np.linalg.norm(m_robot) + 1e-12)
def rotate_body_xy(v, yaw_deg):
    """
    Rotate a body-frame vector about the local body z-axis.
    Positive yaw uses right-hand rule.
    """
    a = np.deg2rad(yaw_deg)
    Rz = np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a),  np.cos(a), 0.0],
        [0.0,        0.0,       1.0],
    ], dtype=float)
    return Rz @ np.asarray(v, dtype=float).reshape(3,)
def beam_tangent_in_robot_from_pose(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()

    # straight beam direction in tool/body frame
    t_body = np.array([-1.0, 0.0, 0.0], float)

    t_robot = R @ t_body
    return t_robot / (np.linalg.norm(t_robot) + 1e-12)

def plot_analytic_vs_fd_jacobian(
    J_an,
    J_fd,
    results_dir,
    filename="analytic_vs_fd_jacobian.png",
    title_prefix="Reduced state Jacobian",
):
    os.makedirs(results_dir, exist_ok=True)

    J_an = np.asarray(J_an, float)
    J_fd = np.asarray(J_fd, float)
    J_err = J_an - J_fd

    row_labels = ["tip_x", "tip_y", "tip_z", "tan_x", "tan_y", "tan_z"][:J_an.shape[0]]
    col_labels = ["x", "y", "yaw_z", "L"][:J_an.shape[1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    vmax = max(np.max(np.abs(J_an)), np.max(np.abs(J_fd)), 1e-12)
    vmax_err = max(np.max(np.abs(J_err)), 1e-12)

    mats = [J_fd, J_an, J_err]
    titles = [
        f"{title_prefix}: finite difference",
        f"{title_prefix}: analytic",
        "analytic - finite difference",
    ]
    vlims = [vmax, vmax, vmax_err]

    for ax, M, title, lim in zip(axes, mats, titles, vlims):
        im = ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(title)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2e}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()

    print(f"Saved analytic-vs-FD Jacobian plot to: {out_path}")
def apply_reduced_step_to_p8(
    p8: np.ndarray,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    dyaw: float = 0.0,
    dL: float = 0.0,
    flip_yaw: bool = True,
) -> np.ndarray:
    """
    Reduced state perturbation:
        [x, y, yaw_z, L]

    dx, dy in metres.
    dyaw in radians.
    dL in metres.

    If flip_yaw=True, the commanded yaw sign is inverted before applying
    the quaternion update.
    """
    p8_next = np.asarray(p8, float).copy()

    p8_next[0] += float(dx)
    p8_next[1] += float(dy)

    q = quat_wxyz_normalize(p8_next[3:7])

    dyaw_apply = -float(dyaw) if flip_yaw else float(dyaw)
    dqz = quat_from_yaw_wxyz(dyaw_apply)

    p8_next[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz, q))

    p8_next[7] += float(dL)

    if p8_next[7] <= 0.0:
        raise ValueError(f"Invalid insertion length after step: {p8_next[7]}")

    return p8_next
def capture_and_measure_tip_base_local(cfg: SinglePoseEvalConfig) -> Dict:
    """
    Capture image from camera, then measure physical tip in base-local coordinates.
    """
    new_capture()

    manual = load_manual_vessel_boundaries_with_frame(MANUAL_VESSEL_BOUNDARY_FILE)

    meas = measure_tip_from_vision_base_local(
        cfg,
        base_px_ref=manual["base_px"],
        ex_ref=manual["ex_img"],
        ey_ref=manual["ey_img"],
    )

    return meas
def validate_one_physical_jacobian_step(
    *,
    hw,
    cfg: SinglePoseEvalConfig,
    dx: float = 0.0,
    dy: float = 0.0,
    dyaw: float = 0.0,
    dL: float = 0.0,
    dt: float = 0.01,
) -> Dict:
    """
    Physically move the robot a small amount and compare:

        measured delta tip
        nonlinear model delta tip
        linear Jacobian delta tip

    Reduced step:
        dq = [dx, dy, dyaw, dL]

    Units:
        dx, dy, dL in metres
        dyaw in radians
    """

    os.makedirs(cfg.results_dir, exist_ok=True)

    print("\n==============================")
    print(" PHYSICAL ONE-STEP JACOBIAN TEST ")
    print("==============================")
    print("dx [mm]    =", 1e3 * dx)
    print("dy [mm]    =", 1e3 * dy)
    print("dyaw [deg] =", np.degrees(dyaw))
    print("dL [mm]    =", 1e3 * dL)

    # ------------------------------------------------------------
    # Current nominal pose/state
    # ------------------------------------------------------------
    pose6_0 = np.asarray(cfg.test_pose6, float).reshape(6,)
    L0 = float(cfg.L_m)
    p8_0 = pose6_and_L_to_pose8_quat(pose6_0, L0)

    # ------------------------------------------------------------
    # Measure physical initial tip
    # ------------------------------------------------------------
    print("\n--- CAPTURE INITIAL PHYSICAL TIP ---")
    meas0 = capture_and_measure_tip_base_local(cfg)
    y_meas0_base = np.asarray(meas0["tip_base_local_m"], float).reshape(3,)

    # ------------------------------------------------------------
    # Build model at initial image/lumen
    # ------------------------------------------------------------
    print("\n--- BUILD FORWARD MODEL ---")
    fwd = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=L0,
        image_filename=cfg.image_filename,
        red_roi_path=cfg.red_roi_path,
        blue_roi_path="blue_roi_box.json",
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint,
        lumen=True,
    )

    # ------------------------------------------------------------
    # Nonlinear model nominal prediction
    # ------------------------------------------------------------
    print("\n--- MODEL NOMINAL PREDICTION ---")
    pred0 = predict_tip_local_from_model(
        forward6d=fwd,
        pose6=pose6_0,
        L_m=L0,
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )

    y_model0_robot = np.asarray(pred0["tip_robot_m"], float).reshape(3)
    y_model0_base = predicted_tip_to_base_local_from_base_point_robot(
        pred_tip_robot_m=y_model0_robot,
        beam_base_point_robot_m=np.asarray(cfg.beam_base_point_robot_m, dtype=float),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )

    # ------------------------------------------------------------
    # Compute analytic Jacobian at nominal pose
    # ------------------------------------------------------------
    print("\n--- ANALYTIC JACOBIAN AT NOMINAL ---")

    fwd_jac = copy.deepcopy(fwd)

    # Make sure nominal state is committed before analytic Jacobian.
    _ = np.asarray(fwd_jac(p8_0, commit=True), float).reshape(-1)[:3]

    fwd_L_fd = copy.deepcopy(fwd)

    J_state_robot = hybrid_J_robot_xy_yaw_dL(
        p8=p8_0,
        fwd_analytic=fwd_jac,
        fwd_fd=fwd_L_fd,
        n_out=3,
        dL_fd=1e-3,
    )

    # Convert robot-frame tip delta Jacobian to base-local coordinates.
    R_pivot = pivot_rotation_matrix(cfg.pivot_pose6)
    J_state_base = R_pivot.T @ J_state_robot

    dq = np.array([dx, dy, dyaw, dL], dtype=float)

    dy_lin_base = J_state_base @ dq
    y_lin1_base = y_model0_base + dy_lin_base

    # ------------------------------------------------------------
    # Nonlinear model prediction at commanded next pose
    # ------------------------------------------------------------
    p8_1 = apply_reduced_step_to_p8(
        p8_0,
        dx=dx,
        dy=dy,
        dyaw=dyaw,
        dL=dL,
    )

    pose6_1, L1 = p8_to_ur_pose6_and_L(p8_1)

    pred1 = predict_tip_local_from_model(
        forward6d=fwd,
        pose6=np.asarray(pose6_1, float),
        L_m=float(L1),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )

    y_model1_robot = np.asarray(pred1["tip_robot_m"], float).reshape(3)
    y_model1_base = predicted_tip_to_base_local_from_base_point_robot(
        pred_tip_robot_m=y_model1_robot,
        beam_base_point_robot_m=np.asarray(cfg.beam_base_point_robot_m, dtype=float),
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
    )

    dy_nonlinear_base = y_model1_base - y_model0_base

    # ------------------------------------------------------------
    # Hardware move
    # ------------------------------------------------------------
    print("\n--- SEND SMALL HARDWARE STEP ---")
    print("pose6_0 =", pose6_0)
    print("pose6_1 =", pose6_1)
    print("L0 =", L0)
    print("L1 =", L1)

    # send_step expects velocity-like control over dt.
    u_step = np.zeros(7, dtype=float)
    u_step[0] = dx / dt
    u_step[1] = dy / dt
    u_step[5] = dyaw / dt
    u_step[6] = dL / dt

    print("u_step =", u_step)
    print("dt =", dt)

    hw.send_step(
        p_now=p8_0,
        u0=u_step,
        dt=dt,
    )

    time.sleep(0.5)

    # ------------------------------------------------------------
    # Measure physical final tip
    # ------------------------------------------------------------
    print("\n--- CAPTURE FINAL PHYSICAL TIP ---")
    cfg_after = copy.deepcopy(cfg)
    cfg_after.test_pose6 = np.asarray(pose6_1, float)
    cfg_after.L_m = float(L1)

    meas1 = capture_and_measure_tip_base_local(cfg_after)
    y_meas1_base = np.asarray(meas1["tip_base_local_m"], float).reshape(3)

    dy_meas_base = y_meas1_base - y_meas0_base

    # ------------------------------------------------------------
    # Compare deltas
    # ------------------------------------------------------------
    err_lin_delta = dy_meas_base - dy_lin_base
    err_nonlin_delta = dy_meas_base - dy_nonlinear_base

    print("\n==============================")
    print(" ONE-STEP RESULT ")
    print("==============================")

    print("\n--- base-local initial/final measured tip [mm] ---")
    print("meas0 [mm] =", 1e3 * y_meas0_base)
    print("meas1 [mm] =", 1e3 * y_meas1_base)

    print("\n--- base-local model nominal/final tip [mm] ---")
    print("model0 [mm] =", 1e3 * y_model0_base)
    print("model1 nonlinear [mm] =", 1e3 * y_model1_base)
    print("model1 linear [mm]    =", 1e3 * y_lin1_base)

    print("\n--- delta tip [mm] ---")
    print("measured delta      =", 1e3 * dy_meas_base)
    print("nonlinear model     =", 1e3 * dy_nonlinear_base)
    print("linear J prediction =", 1e3 * dy_lin_base)

    print("\n--- delta errors [mm] ---")
    print("measured - nonlinear =", 1e3 * err_nonlin_delta)
    print("measured - linear    =", 1e3 * err_lin_delta)

    print("\n--- norms [mm] ---")
    print("||measured delta||      =", 1e3 * np.linalg.norm(dy_meas_base))
    print("||nonlinear delta||     =", 1e3 * np.linalg.norm(dy_nonlinear_base))
    print("||linear delta||        =", 1e3 * np.linalg.norm(dy_lin_base))
    print("||meas - nonlinear||    =", 1e3 * np.linalg.norm(err_nonlin_delta))
    print("||meas - linear||       =", 1e3 * np.linalg.norm(err_lin_delta))

    result = {
        "dx_m": float(dx),
        "dy_m": float(dy),
        "dyaw_rad": float(dyaw),
        "dL_m": float(dL),
        "dt": float(dt),

        "pose6_0": pose6_0.tolist(),
        "pose6_1": np.asarray(pose6_1, float).tolist(),
        "L0": float(L0),
        "L1": float(L1),

        "y_meas0_base_m": y_meas0_base.tolist(),
        "y_meas1_base_m": y_meas1_base.tolist(),
        "dy_meas_base_m": dy_meas_base.tolist(),

        "y_model0_base_m": y_model0_base.tolist(),
        "y_model1_nonlinear_base_m": y_model1_base.tolist(),
        "dy_nonlinear_base_m": dy_nonlinear_base.tolist(),

        "dy_linear_base_m": dy_lin_base.tolist(),
        "y_model1_linear_base_m": y_lin1_base.tolist(),

        "err_linear_delta_m": err_lin_delta.tolist(),
        "err_nonlinear_delta_m": err_nonlin_delta.tolist(),

        "err_linear_delta_norm_mm": float(1e3 * np.linalg.norm(err_lin_delta)),
        "err_nonlinear_delta_norm_mm": float(1e3 * np.linalg.norm(err_nonlin_delta)),

        "J_state_robot": J_state_robot.tolist(),
        "J_state_base": J_state_base.tolist(),
    }

    out_path = os.path.join(cfg.results_dir, "physical_one_step_jacobian_test.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved physical one-step result to:", out_path)

    return result
def hybrid_J_robot_xy_yaw_dL(
    *,
    p8,
    fwd_analytic,
    fwd_fd,
    n_out=3,
    dL_fd=1e-3,
):
    """
    Hybrid reduced Jacobian wrt [x, y, yaw_z, L].

    Columns:
      0: analytic dx
      1: analytic dy
      2: analytic yaw_z
      3: finite-difference dL

    Important:
      fwd_analytic should already have had the nominal pose committed
      if analytic_J_robot_xy_yaw_dL expects cached last_info.

      fwd_fd should be an independent deepcopy/wrapper so FD calls do not
      contaminate the analytic cache.
    """
    p8 = np.asarray(p8, float).reshape(8,)

    # Analytic x, y, yaw, and provisional L.
    J = analytic_J_robot_xy_yaw_dL(
        p8,
        fwd_analytic,
        n_out=n_out,
    )

    J = np.asarray(J, float)[:n_out, :4]

    # Replace only the L column using finite difference.
    p8_Lp = p8.copy()
    p8_Lm = p8.copy()
    p8_Lp[7] += dL_fd
    p8_Lm[7] -= dL_fd

    y_Lp = np.asarray(fwd_fd(p8_Lp, commit=False), float).reshape(-1)[:n_out]
    y_Lm = np.asarray(fwd_fd(p8_Lm, commit=False), float).reshape(-1)[:n_out]

    J[:, 3] = (y_Lp - y_Lm) / (2.0 * dL_fd)

    return J
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
        z_offset=0.28,
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )
    pivot_point2, start_point2, L0_default, dt = make_initial_poses_single_use(hw)

    # pivot_point2, start_point2, L0_default, dt = make_initial_poses_single_use()
    # new_capture()

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
    pivot_hint = (321.200927734375, 331.6798095703125)
    cfg = SinglePoseEvalConfig(
        pivot_pose6=np.asarray(pivot_point2, dtype=float),
        test_pose6=test_pose6,
        beam_base_point_robot_m=beam_base_point_robot_m,
        L_m=L_test,
        image_filename="/home/jack/Proper-Research/focused_image.jpg",
        reference_image_filename="/home/jack/Proper-Research/focused_image.jpg",
        use_reference_frame=True,
        red_roi_path="/home/jack/Proper-Research/red_roi_box.json",
        green_roi_path="green_roi_box.json",
        known_green_distance_mm=15,
        pivot_hint=pivot_hint,
        results_dir="results_single_pose_forward_validation_back",
        save_overlay_path="results_single_pose_forward_validation/comparison_back.png",
        show_debug_vision=False,
        show_debug_model=False,
    )
    validate_one_physical_jacobian_step(
        hw=hw,
        cfg=cfg,
        dx=0.0,
        dy=1.0e-3,          # 1 mm
        dyaw=0.0,
        dL=0.0,
        dt=0.05,
    )
    validate_one_physical_jacobian_step(
        hw=hw,
        cfg=cfg,
        dx=1.0e-3,          # 1 mm
        dy=0.0,
        dyaw=0.0,
        dL=0.0,
        dt=0.05,
    )
    validate_one_physical_jacobian_step(
        hw=hw,
        cfg=cfg,
        dx=0.0,
        dy=0.0,
        dyaw=np.deg2rad(1.0),   # 1 degree
        dL=0.0,
        dt=0.05,
    )
    validate_one_physical_jacobian_step(
        hw=hw,
        cfg=cfg,
        dx=0.0,
        dy=0.0,
        dyaw=0.0,
        dL=0.3e-3,          # 0.3 mm
        dt=0.05,
    )
    evaluate_single_pose(cfg)