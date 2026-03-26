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
    p_robot = np.asarray(p_robot, dtype=float).reshape(3,)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    return p_robot - p_pivot

def robot_pose_to_pivot_local_pose(pose6_robot, pivot_pose6):
    pose6_robot = np.asarray(pose6_robot, dtype=float).reshape(6,)
    return robot_point_to_pivot_local(pose6_robot[:3], pivot_pose6)

def pivot_local_point_to_robot(local_xyz, pivot_pose6):
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    local_xyz = np.asarray(local_xyz, dtype=float).reshape(3,)
    return p_pivot + local_xyz

def robot_dir_to_pivot_local(v_robot, pivot_pose6):
    v_robot = np.asarray(v_robot, dtype=float).reshape(3,)
    return v_robot.copy()
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


# ============================================================
# Build "no-lumen-effect" forward model
# ============================================================

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

def build_forward_model_no_lumen_effect(pivot_pose6: np.ndarray, L0: float) -> DeterministicForward6D:
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

    lumen_C, lumen_R = build_dummy_straight_lumen_from_pivot(
        pivot_pose6=pivot_pose6,
        length_m=0.16,
        radius_m=0.05,   # huge radius -> essentially unconstrained
        n_pts=160,
    )

    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        N_nodes=35,
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
    forward6d = build_forward_model_no_lumen_effect(
        pivot_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
        L0=float(cfg.L_m),
    )
    dp_robot = np.asarray(cfg.test_pose6[:3], float) - np.asarray(cfg.pivot_pose6[:3], float)
    R_pivot = ur_pose6_to_T(cfg.pivot_pose6)[:3, :3]
    dp_local = R_pivot.T @ dp_robot

    print("\n--- FRAME DEBUG ---")
    print("dp_robot [mm] =", 1e3 * dp_robot)
    print("dp_local [mm] =", 1e3 * dp_local)


    pred = predict_tip_local_from_model(
        forward6d=forward6d,
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
    plot_single_tip_comparison_local(
        pred_local_m=pred_base_local_m_plot,
        meas_local_m=meas_base_local_m_plot,
        results_dir=cfg.results_dir,
        filename="tip_comparison_base_local.png",
        src_local_m=src_base_local_plot_m,
        src_dir_local=src_dir_local,
    )

    print(f"\nSaved summary to: {out_json}")
    return result


# ============================================================
# Main
# ============================================================
def make_initial_poses_single_use(hw) -> tuple[np.ndarray, np.ndarray, float, float]:
    pivot_point = np.array([
    0.8781328220229531, -0.7112731669220016, -0.1,  np.pi, 0.001,0.001
    ], float)


    
    L0 = 0.049

    pose6 = np.asarray(get_point(0, 0), dtype=float)
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
):
    pred_local_m = np.asarray(pred_local_m, dtype=float).reshape(3,)
    meas_local_m = np.asarray(meas_local_m, dtype=float).reshape(3,)

    pred_mm = 1e3 * pred_local_m
    meas_mm = 1e3 * meas_local_m

    plt.figure(figsize=(7, 7))

    # pivot / base-local origin
    plt.plot(0.0, 0.0, "ko", label="Base / pivot-local origin")

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

    evaluate_single_pose(cfg)