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

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple
from proper_research.robot.live_hardware_control import LiveHardwareController
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
    DeterministicForward6D,
    effective_lengths,
)
from proper_research.vision.measure_length import new_capture
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel

from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from proper_research.parameters import default_magnet_params

# ---- vision side ----
# Update this import path if your vision utilities live elsewhere.
from proper_research.vision.bounds_beam import (
    detect_2_green_calibration_points,
    measure_tip_state_4markers,
)



mag_params = default_magnet_params()


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

    red_roi_path: str = "red_roi_box.json"
    green_roi_path: str = "green_roi_box.json"

    known_green_distance_mm: float = 40.0
    pivot_hint: Tuple[float, float] | None = None

    results_dir: str = "results_single_pose_forward_validation"
    save_overlay_path: str = "results_single_pose_forward_validation/comparison_overlay.png"

    show_debug_vision: bool = True
    show_debug_model: bool = True

# ============================================================
# Basic helpers
# ============================================================
def robot_point_to_pivot_local(p_robot, pivot_pose6):
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    R = T[:3, :3]
    t = T[:3, 3]
    p_robot = np.asarray(p_robot, dtype=float).reshape(3,)
    return R.T @ (p_robot - t)

def pivot_local_point_to_robot(local_xyz, pivot_pose6):
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    R = T[:3, :3]
    t = T[:3, 3]
    local_xyz = np.asarray(local_xyz, dtype=float).reshape(3,)
    return t + R @ local_xyz

def robot_dir_to_pivot_local(v_robot, pivot_pose6):
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    R = T[:3, :3]
    v_robot = np.asarray(v_robot, dtype=float).reshape(3,)
    return R.T @ v_robot
# def robot_dir_to_pivot_local(v_robot, pivot_pose6):
#     T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
#     R = T[:3, :3]
#     v_robot = np.asarray(v_robot, dtype=float).reshape(3,)
#     return R.T @ v_robot
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

def transform_local_points_to_robot(points_local_m, pivot_pose6, flip_y=True):
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    R = T[:3, :3]
    t = T[:3, 3]

    P_local = np.asarray(points_local_m, dtype=float).copy()
    if P_local.ndim == 1:
        P_local = P_local.reshape(1, 3)

    if flip_y:
        P_local[:, 1] *= -1.0

    return (R @ P_local.T).T + t
# ============================================================
# Build "no-lumen-effect" forward model
# ============================================================
def mirror_x_plane_point(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy().reshape(3,)
    p[1] *= -1.0
    return p

def mirror_x_plane_dir(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy().reshape(3,)
    v[1] *= -1.0
    return v

def mirror_x_plane_rotmat(R: np.ndarray) -> np.ndarray:
    """
    Mirror a rotation basis across the x-z plane, i.e. y -> -y.
    """
    M = np.diag([1.0, -1.0, 1.0])
    return M @ R @ M

def mirror_pose6_across_x_plane(pose6: np.ndarray) -> np.ndarray:
    """
    Mirror a UR pose [x,y,z,rx,ry,rz] across the x-z plane (flip y).
    This mirrors both translation and orientation consistently.
    """
    pose6 = np.asarray(pose6, dtype=float).copy().reshape(6,)

    # mirror translation
    pose6[:3] = mirror_x_plane_point(pose6[:3])

    # mirror orientation
    R = Rot.from_rotvec(pose6[3:6]).as_matrix()
    Rm = mirror_x_plane_rotmat(R)
    pose6[3:6] = Rot.from_matrix(Rm).as_rotvec()

    return pose6
def build_initial_lumen_from_vision(
    pivot_point,
    image_filename="focused_image.jpg",
    red_roi_path="red_roi_box.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=True,
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

    lumen_C_vision_m = np.asarray(vision_result["lumen_C_m"], dtype=float).copy()
    lumen_R_m = np.asarray(vision_result["lumen_R_m"], dtype=float)

    lumen_C_robot_m = transform_local_points_to_robot(
        lumen_C_vision_m,
        pivot_point,
    )

    return lumen_C_vision_m, lumen_C_robot_m, lumen_R_m
def build_dummy_straight_lumen_from_pivot(
    pivot_pose6: np.ndarray,
    length_m: float = 0.12,
    radius_m: float = 0.05,
    n_pts: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    p0_ur = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    t0 = np.array([-1.0, 0.0, 0.0], dtype=float)

    lumen_C = np.array(
        [p0_ur + s * t0 for s in np.linspace(0.0, length_m, n_pts)],
        dtype=float,
    )
    lumen_R = np.full(len(lumen_C), float(radius_m), dtype=float)
    return lumen_C, lumen_R

def build_forward_model_no_lumen_effect(
    pivot_pose6: np.ndarray,
    L0: float,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    ) -> DeterministicForward6D:    """
    Build the same solver stack as the controller uses, but with a dummy
    straight/wide lumen so the comparison is effectively 'beam theory only'.
    """
def build_forward_model_no_lumen_effect(
    pivot_pose6: np.ndarray,
    L0: float,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
) -> DeterministicForward6D:
    T_ur_pivot = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )

    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, dtype=float),
        lumen_R=np.asarray(lumen_R, dtype=float),
        N_nodes=10,
        maxiter=70,
        L0_init=0.01,
        dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01,
        use_lumen_jac=False,
    )

    return DeterministicForward6D(forward_model)


# ============================================================
# Model prediction
# ============================================================

def predict_tip_local_from_model(
    forward6d: DeterministicForward6D,
    pose6: np.ndarray,
    L_m: float,
    pivot_pose6: np.ndarray,
) -> Dict:
    p8 = pose6_and_L_to_pose8_quat(pose6, L_m)

    if hasattr(forward6d, "start_step"):
        forward6d.start_step()

    y_pred_robot = np.asarray(forward6d(p8, commit=False), dtype=float).reshape(-1)
    p_tip_robot = y_pred_robot[:3]
    p_tip_local = robot_point_to_pivot_local(p_tip_robot, pivot_pose6)

    out = {
        "pose8": p8.copy(),
        "y_pred_robot": y_pred_robot.copy(),
        "tip_robot_m": p_tip_robot.copy(),
        "tip_local_m": p_tip_local.copy(),
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
    green_result = detect_2_green_calibration_points(
        image_bgr=image_bgr,
        roi_box=green_roi_box,
        green_h_low=35,
        green_h_high=95,
        sat_min=40,
        val_min=40,
        min_area=3,
        max_area=50000,
        show_debug=False,
    )

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
    show_debug_markers: bool = False,
):
    red_roi_box = load_roi_box(red_roi_path)

    ref_result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=red_roi_box,
        show=show,
        show_debug_markers=show_debug_markers,
        unwrap_angle=False,
        pivot_hint=pivot_hint,
        base_px_ref=None,
        ex_ref=None,
        ey_ref=None,
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

    tip_result = measure_tip_state_4markers(
        image_filename=cfg.image_filename,
        roi_box=red_roi_box,
        show=cfg.show_debug_vision,
        show_debug_markers=cfg.show_debug_vision,
        unwrap_angle=False,
        pivot_hint=cfg.pivot_hint,
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
    )

    tip_xy_px = np.asarray(tip_result["tip_xy_from_base"], dtype=float).reshape(2,)
    tip_xy_m = (tip_xy_px * mm_per_pixel) / 1000.0

    tip_base_local_m = np.array([
        -tip_xy_m[0],
        -tip_xy_m[1],
        0.0,
    ], dtype=float)

    return {
        "mm_per_pixel": float(mm_per_pixel),
        "tip_result": tip_result,
        "tip_base_local_m": tip_base_local_m.copy(),
    }

# ============================================================
# Single-shot comparison
# ============================================================
def beam_tangent_in_robot_from_pose(pose6_robot):
    rvec = np.asarray(pose6_robot[3:6], float)
    R = Rot.from_rotvec(rvec).as_matrix()
    t_body = np.array([-1.0, 0.0, 0.0], float)
    t_robot = R @ t_body
    return t_robot / (np.linalg.norm(t_robot) + 1e-12)
def mirror_pose6_across_x_plane_about_pivot(pose6_robot: np.ndarray, pivot_pose6: np.ndarray) -> np.ndarray:
    pose6_robot = np.asarray(pose6_robot, dtype=float).copy().reshape(6,)

    # position: robot -> pivot-local
    p_local = robot_point_to_pivot_local(pose6_robot[:3], pivot_pose6)
    p_local[1] *= -1.0
    pose6_robot[:3] = pivot_local_point_to_robot(p_local, pivot_pose6)

    # orientation: robot dir -> pivot-local dir -> mirror -> back
    R_robot = Rot.from_rotvec(pose6_robot[3:6]).as_matrix()
    R_pivot = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))[:3, :3]

    # express orientation in pivot-local basis
    R_local = R_pivot.T @ R_robot

    M = np.diag([1.0, -1.0, 1.0])
    R_local_m = M @ R_local @ M

    # back to robot frame
    R_robot_m = R_pivot @ R_local_m
    pose6_robot[3:6] = Rot.from_matrix(R_robot_m).as_rotvec()

    return pose6_robot
def evaluate_single_pose(
    cfg: SinglePoseEvalConfig,
    lumen_C_plot_m: np.ndarray,
    lumen_R_plot_m: np.ndarray,
    lumen_C_solver_m: np.ndarray,
    lumen_R_solver_m: np.ndarray,
) -> Dict:
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

    # Build forward model ONCE using the fixed solver lumen
    forward6d = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),
        lumen_C=np.asarray(lumen_C_solver_m, dtype=float),
        lumen_R=np.asarray(lumen_R_solver_m, dtype=float),
    )

    dp_robot = np.asarray(cfg.test_pose6[:3], float) - np.asarray(cfg.pivot_pose6[:3], float)
    R_pivot = ur_pose6_to_T(cfg.pivot_pose6)[:3, :3]
    dp_local = R_pivot.T @ dp_robot

    print("\n--- FRAME DEBUG ---")
    print("dp_robot [mm] =", 1e3 * dp_robot)
    print("dp_local [mm] =", 1e3 * dp_local)

        # Solver pose: mirrored if that is your agreed solver convention
    test_pose6_solver = mirror_pose6_across_x_plane_about_pivot(
        np.asarray(cfg.test_pose6, dtype=float),
        np.asarray(cfg.pivot_pose6, dtype=float),
    )
    pred = predict_tip_local_from_model(
        forward6d=forward6d,
        pose6=test_pose6_solver,
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
        "test_pose6_solver": np.asarray(test_pose6_solver, dtype=float).tolist(),
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

    # Debug using the actual solver pose
    m_robot_solver = source_dipole_in_robot(test_pose6_solver)
    t_robot_solver = beam_tangent_in_robot_from_pose(test_pose6_solver)

    print("source dipole dir in robot =", m_robot_solver)
    print("beam tangent dir in robot  =", t_robot_solver)
    print("alignment cos(theta)       =", float(np.dot(m_robot_solver, t_robot_solver)))

    m_local_solver = unit(m_robot_solver)
    t_local_solver = unit(t_robot_solver)

    print("cos robot =", np.dot(m_robot_solver, t_robot_solver))
    print("cos local =", np.dot(m_local_solver, t_local_solver))

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

    # External magnet for plotting/debug should match the solver pose
    src_base_local_m = robot_point_to_pivot_local(
        np.asarray(test_pose6_solver[:3], dtype=float),
        np.asarray(cfg.pivot_pose6, dtype=float),
    )

    base_local_m = robot_point_to_pivot_local(
        np.asarray(cfg.beam_base_point_robot_m, dtype=float),
        np.asarray(cfg.pivot_pose6, dtype=float),
    )
    src_base_local_m = src_base_local_m - base_local_m

    src_dir_robot = source_dipole_in_robot(test_pose6_solver)
    src_dir_local = robot_dir_to_pivot_local(src_dir_robot, cfg.pivot_pose6)

    pred_base_local_m_plot = pred_base_local_m.copy()
    meas_base_local_m_plot = meas["tip_base_local_m"].copy()

    # If your compare frame mirrors y for robot-derived quantities, keep this:
    src_base_compare_m = pivot_local_to_compare_frame_point(src_base_local_m)
    src_dir_compare = pivot_local_to_compare_frame_dir(src_dir_local)

    print(f"TEST POSE RAW    {cfg.test_pose6}")
    print(f"TEST POSE SOLVER {test_pose6_solver}")

    plot_single_tip_comparison_local(
        pred_local_m=pred_base_local_m_plot,
        meas_local_m=meas_base_local_m_plot,
        results_dir=cfg.results_dir,
        filename="tip_comparison_base_local.png",
        src_local_m=src_base_compare_m,
        src_dir_local=src_dir_compare,
        lumen_C_local_m=lumen_C_plot_m,
        lumen_R_m=lumen_R_plot_m,
    )

    print(f"\nSaved summary to: {out_json}")
    return result
def robot_points_to_pivot_local(points_robot, pivot_pose6):
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    R = T[:3, :3]
    t = T[:3, 3]

    P = np.asarray(points_robot, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, 3)

    return (R.T @ (P - t).T).T
def pivot_local_to_compare_frame_point(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).copy().reshape(3,)
    p[1] *= -1.0
    return p

def pivot_local_to_compare_frame_dir(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy().reshape(3,)
    v[1] *= -1.0
    return v

def pivot_local_to_compare_frame_points(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()
    if P.ndim == 1:
        P = P.reshape(1, 3)
    P[:, 1] *= -1.0
    return P
# ============================================================
# Main
# ============================================================
def make_initial_poses_single_use(hw) -> tuple[np.ndarray, np.ndarray, float, float]:
    pivot_point = np.array([
    0.8681328220229531, -0.7112731669220016, -0.1,  np.pi, 0.001,0.001
    ], float)


    
    L0 = 0.078

    pose6 = np.asarray(get_point(0, 30), dtype=float)
    pose6[2] = -0.1
    # pose6[0] = 0.3
    print(f"POSE6 is {pose6}")
    T = ur_pose6_to_T(pose6)
    p, q_wxyz = T_to_p_quat_wxyz(T)

    p_now = np.concatenate([p, q_wxyz, [L0]])
    u0 = np.zeros(7, dtype=float)
    hw.send_step(p_now=p_now, u0=u0, dt=0.1)
    # start_point = np.array([
    # 0.665894307606053, -0.7112810117612073, -0.1, np.pi, 0,0
    # ], float)
    robot_pose6 = hw.get_robot_pose_once()
    robot_pose6[2] = -0.1

    start_point = robot_pose6
    print(f"START POINT: {start_point}")
    # L0 = 0.065
    dt = 0.01
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
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    pred_local_m = np.asarray(pred_local_m, dtype=float).reshape(3,)
    meas_local_m = np.asarray(meas_local_m, dtype=float).reshape(3,)

    pred_mm = 1e3 * pred_local_m
    meas_mm = 1e3 * meas_local_m

    plt.figure(figsize=(7, 7))

    # pivot / base-local origin
    plt.plot(0.0, 0.0, "ko", label="Base / pivot-local origin")

    # -----------------------------
    # lumen plotting
    # -----------------------------
    if lumen_C_local_m is not None:
        lumen_C_local_m = np.asarray(lumen_C_local_m, dtype=float)
        if lumen_C_local_m.ndim != 2 or lumen_C_local_m.shape[1] != 3:
            raise ValueError(f"lumen_C_local_m must have shape (N,3), got {lumen_C_local_m.shape}")

        lumen_xy_mm = 1e3 * lumen_C_local_m[:, :2]

        # plot lumen centerline
        plt.plot(
            lumen_xy_mm[:, 0],
            lumen_xy_mm[:, 1],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="Lumen centerline",
        )

        if lumen_R_m is not None:
            lumen_R_m = np.asarray(lumen_R_m, dtype=float).reshape(-1)
            if lumen_R_m.shape[0] != lumen_C_local_m.shape[0]:
                raise ValueError(
                    f"lumen_R_m must have same length as lumen_C_local_m, "
                    f"got {lumen_R_m.shape[0]} vs {lumen_C_local_m.shape[0]}"
                )

            lumen_R_mm = 1e3 * lumen_R_m

            # build a 2D normal from the centerline tangent
            xy = lumen_xy_mm
            dx = np.gradient(xy[:, 0])
            dy = np.gradient(xy[:, 1])
            tang = np.stack([dx, dy], axis=1)

            tang_n = np.linalg.norm(tang, axis=1, keepdims=True)
            tang_n[tang_n < 1e-12] = 1.0
            tang = tang / tang_n

            # left normal in plane
            normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)

            upper = xy + normal * lumen_R_mm[:, None]
            lower = xy - normal * lumen_R_mm[:, None]

            plt.plot(
                upper[:, 0],
                upper[:, 1],
                color="green",
                linewidth=1.5,
                label="Lumen wall",
            )
            plt.plot(
                lower[:, 0],
                lower[:, 1],
                color="green",
                linewidth=1.5,
            )

            plt.fill_betweenx(
                upper[:, 1],
                lower[:, 0],
                upper[:, 0],
                color="green",
                alpha=0.08,
            )

    # predicted / measured tip
    plt.plot(pred_mm[0], pred_mm[1], "bs", markersize=10, label="Predicted tip")
    plt.plot(meas_mm[0], meas_mm[1], "ro", markersize=10, label="Measured tip")

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

    # external magnet position
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
    use_advancer=False,
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
    pivot_point2, start_point2, L0_default, dt = make_initial_poses_single_use(hw=hw)


    new_capture()
    pivot_hint = (300, 391)
    # lumen_C_vision_m, lumen_C_robot_m, lumen_R_m = build_initial_lumen_from_vision(
    #     pivot_point=pivot_point2,
    #     image_filename="focused_image.jpg",
    #     red_roi_path="red_roi_box.json",
    #     blue_roi_path="blue_roi_box.json",
    #     green_roi_path="green_roi_box.json",
    #     pivot_hint=pivot_hint,
    #     show=False,
    # )

    # lumen_C_local_m = robot_points_to_pivot_local(lumen_C_robot_m, pivot_point2)
    # lumen_R_local_m = np.asarray(lumen_R_m, dtype=float).copy()
    src_local = np.array([-0.183, 0.0, 0.0], dtype=float)
    src_robot = pivot_local_point_to_robot(src_local, pivot_point2)
    # src_local_check = robot_point_to_pivot_local(src_robot, pivot_point2)

    # print("desired source in pivot-local [mm] =", 1e3 * src_local)
    # print("converted source in robot frame [mm] =", 1e3 * src_robot)
    # print("back-converted source in pivot-local [mm] =", 1e3 * src_local_check)
    test_pose6 = np.asarray(start_point2, dtype=float).reshape(6,)
    test_pose6[:3] = src_robot
    src_local_check = robot_pose_to_pivot_local_pose(test_pose6, pivot_point2)
    print("source position from robot pose in pivot local [mm] =", 1e3 * src_local_check)

    L_test = float(L0_default)
    
    m_robot = source_dipole_in_robot(test_pose6)
    t_robot = beam_tangent_in_robot_from_pose(test_pose6)
    print("source dipole dir in robot =", m_robot)
    print("beam axis dir in robot     =", t_robot)
    print("alignment cos(theta)       =", float(np.dot(m_robot, t_robot)))

    # # IMPORTANT:
    # # Replace this with the real beam base point in robot coordinates.
    # # This is a 3D point, not a pose6.
    beam_base_point_robot_m = pivot_point2[:3]
    pivot_hint = (318.200927734375, 369.6798095703125)
    cfg = SinglePoseEvalConfig(
        pivot_pose6=np.asarray(pivot_point2, dtype=float),
        test_pose6=test_pose6,
        beam_base_point_robot_m=beam_base_point_robot_m,
        L_m=L_test,
        image_filename="focused_image.jpg",
        reference_image_filename="focused_image_straight.jpg",
        use_reference_frame=True,
        red_roi_path="red_roi_box.json",
        green_roi_path="green_roi_box.json",
        known_green_distance_mm=40.0,
        pivot_hint=pivot_hint,
        results_dir="results_single_pose_forward_validation",
        save_overlay_path="results_single_pose_forward_validation/comparison_overlay.png",
        show_debug_vision=False,
        show_debug_model=False,
    )

    lumen_C_vision_m, lumen_C_robot_m, lumen_R_m = build_initial_lumen_from_vision(
        pivot_point=pivot_point2,
        image_filename="focused_image.jpg",
        red_roi_path="red_roi_box.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",
        pivot_hint=pivot_hint,
        show=False,
    )

    # Solver lumen: fixed in robot/world frame
    lumen_C_solver_m = lumen_C_robot_m
    lumen_R_solver_m = np.asarray(lumen_R_m, dtype=float).copy()

    # Plot lumen: whatever comparison frame you are using
    lumen_C_plot_m = robot_points_to_pivot_local(lumen_C_robot_m, pivot_point2)
    lumen_R_plot_m = np.asarray(lumen_R_m, dtype=float).copy()

    evaluate_single_pose(
        cfg,
        lumen_C_plot_m=lumen_C_plot_m,
        lumen_R_plot_m=lumen_R_plot_m,
        lumen_C_solver_m=lumen_C_solver_m,
        lumen_R_solver_m=lumen_R_solver_m,
    )