"""Read-only frame and measurement validation for the composite beam model.

Run this file from the Proper-Research repository root with no command-line
arguments.  It imports the exact experiment configuration and direct model
adapter from ``beam_hardware_experiment_v2.py`` so the diagnostic cannot drift
silently onto different beam, pivot, camera or source-magnet parameters.

This script never calls ``send_step``.  It only:

1. reads the current UR pose;
2. captures the image required to initialize the vision-derived lumen;
3. captures one dedicated validation image and reconstructs the measured beam;
4. commits the nominal forward-model solution at that measured source pose;
5. compares measured and predicted centerlines/tips in camera, robot and
   pivot-local frames; and
6. plots the source magnet position, body axes and magnetic dipole direction.

Outputs are written to a timestamped directory:

* ``frame_validation.png`` -- four-panel visual frame check;
* ``frame_validation_summary.json`` -- poses, transforms, metrics and checks;
* ``frame_validation_points.csv`` -- corresponding measured/model curve points;
* ``camera_frame_with_geometry.png`` -- camera overlay at original resolution.

Hardware execution has not been tested in this workspace.  Review the saved
pose values and axes before using any live-motion script.
"""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import beam_hardware_experiment_v2 as experiment_v2


# =============================================================================
# USER CONFIGURATION -- no terminal arguments
# =============================================================================


# Reuse the exact v2 experiment configuration, including its composite beam.
# Edit beam/composite/robot/camera values in beam_hardware_experiment_v2.py.
EXPERIMENT_CONFIG = copy.deepcopy(experiment_v2.CONFIG)


@dataclass
class FrameValidationConfig:
    output_root: str = "beam_frame_validation"
    run_name: str = "nominal_frame_check"
    show_interactively: bool = False
    save_camera_overlay: bool = True
    comparison_points: int = 100

    # Display lengths; these do not alter any model or measured values.
    pivot_axis_display_length_mm: float = 8.0
    source_axis_display_length_mm: float = 8.0
    dipole_arrow_display_length_mm: float = 12.0
    camera_axis_display_length_px: float = 80.0

    # Diagnostic thresholds. A failed check is reported; it does not move or
    # stop hardware because this script is read-only.
    predicted_base_to_pivot_tolerance_mm: float = 1.0
    measured_base_to_pivot_tolerance_mm: float = 3.0
    measured_tip_pipeline_tolerance_mm: float = 2.0
    model_tip_pipeline_tolerance_mm: float = 0.5
    measured_model_tip_tolerance_mm: float = 10.0
    centerline_rmse_tolerance_mm: float = 10.0
    beam_length_difference_tolerance_mm: float = 10.0
    quaternion_norm_tolerance: float = 1.0e-6
    rotation_determinant_tolerance: float = 1.0e-6
    strict_checks: bool = False


CONFIG = FrameValidationConfig()


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}; got {array}.")
    return array


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"Cannot normalize zero {name}.")
    return vector / norm


def _polyline_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _orient_polyline_from_reference(
    points: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float).reshape(points.shape[1])
    if np.linalg.norm(points[-1] - reference) < np.linalg.norm(points[0] - reference):
        return points[::-1].copy()
    return points.copy()


def resample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        raise ValueError("A polyline needs at least two points.")
    if count < 2:
        raise ValueError("comparison_points must be at least 2.")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    if cumulative[-1] < 1.0e-12:
        raise ValueError("Cannot resample a zero-length polyline.")
    sample_s = np.linspace(0.0, cumulative[-1], count)
    output = np.column_stack(
        [np.interp(sample_s, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )
    return output, sample_s / cumulative[-1]


def pivot_transform(pivot_pose6: np.ndarray) -> np.ndarray:
    pivot_pose6 = _finite_array(pivot_pose6, (6,), "pivot_pose6")
    transform = np.eye(4)
    transform[:3, :3] = Rot.from_rotvec(pivot_pose6[3:6]).as_matrix()
    transform[:3, 3] = pivot_pose6[:3]
    return transform


def local_to_robot(points_local_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_local_m, dtype=float).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def robot_to_local(points_robot_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_robot_m, dtype=float).reshape(-1, 3)
    return (points - transform[:3, 3]) @ transform[:3, :3]


def camera_pixels_to_local_m(
    points_px: np.ndarray,
    base_px: np.ndarray,
    ex_img: np.ndarray,
    ey_img: np.ndarray,
    mm_per_pixel: float,
) -> np.ndarray:
    """Apply bounds_beam.image_to_fixed_local_frame to a full polyline."""

    points_px = np.asarray(points_px, dtype=float).reshape(-1, 2)
    base_px = np.asarray(base_px, dtype=float).reshape(2)
    ex_img = np.asarray(ex_img, dtype=float).reshape(2)
    ey_img = np.asarray(ey_img, dtype=float).reshape(2)
    ex_img /= np.linalg.norm(ex_img)
    ey_img /= np.linalg.norm(ey_img)
    image_cartesian = np.column_stack(
        [points_px[:, 0] - base_px[0], -(points_px[:, 1] - base_px[1])]
    )
    x_local_px = -(image_cartesian @ ex_img)
    y_local_px = image_cartesian @ ey_img
    local_mm = np.column_stack(
        [x_local_px, y_local_px, np.zeros(points_px.shape[0])]
    ) * float(mm_per_pixel)
    return local_mm / 1.0e3


def local_m_to_camera_pixels(
    points_local_m: np.ndarray,
    base_px: np.ndarray,
    ex_img: np.ndarray,
    ey_img: np.ndarray,
    mm_per_pixel: float,
) -> np.ndarray:
    """Inverse of camera_pixels_to_local_m for the camera-plane projection."""

    points_local_mm = np.asarray(points_local_m, dtype=float).reshape(-1, 3) * 1.0e3
    base_px = np.asarray(base_px, dtype=float).reshape(2)
    ex_img = np.asarray(ex_img, dtype=float).reshape(2)
    ey_img = np.asarray(ey_img, dtype=float).reshape(2)
    ex_img /= np.linalg.norm(ex_img)
    ey_img /= np.linalg.norm(ey_img)
    image_cartesian = (
        (-points_local_mm[:, 0] / mm_per_pixel)[:, None] * ex_img[None, :]
        + (points_local_mm[:, 1] / mm_per_pixel)[:, None] * ey_img[None, :]
    )
    return np.column_stack(
        [
            base_px[0] + image_cartesian[:, 0],
            base_px[1] - image_cartesian[:, 1],
        ]
    )


def source_rotation_from_p8(p8: np.ndarray) -> np.ndarray:
    p8 = experiment_v2._finite_vector(p8, 8, "source p8")
    quaternion_wxyz = p8[3:7] / np.linalg.norm(p8[3:7])
    quaternion_xyzw = np.r_[quaternion_wxyz[1:4], quaternion_wxyz[0]]
    return Rot.from_quat(quaternion_xyzw).as_matrix()


def extract_predicted_centerline(adapter: Any, pivot_xyz: np.ndarray) -> np.ndarray:
    cache = getattr(adapter.raw_model, "cache", None)
    centerline = None if cache is None else getattr(cache, "centerline", None)
    if centerline is None:
        raise RuntimeError(
            "The committed forward solve did not expose cache.centerline. "
            "Check the installed ForwardCache/SolveResult interface."
        )
    centerline = np.asarray(centerline, dtype=float)
    if centerline.ndim != 2 or centerline.shape[0] < 2 or centerline.shape[1] < 3:
        raise RuntimeError(f"Invalid predicted centerline shape {centerline.shape}.")
    return _orient_polyline_from_reference(centerline[:, :3], pivot_xyz)


# =============================================================================
# READ-ONLY CAPTURE AND MODEL EVALUATION
# =============================================================================


def capture_validation_state(
    experiment_cfg: experiment_v2.ExperimentConfig,
    validation_cfg: FrameValidationConfig,
) -> dict[str, Any]:
    import cv2

    base = experiment_v2._base_module()
    base.configure_bounds_beam_paths(experiment_cfg)
    hw = base.connect_hardware(experiment_cfg)
    try:
        robot_pose6 = hw.get_robot_pose_once()
        if robot_pose6 is None:
            raise RuntimeError("Could not read the UR robot pose.")
        robot_pose6 = experiment_v2._finite_vector(robot_pose6, 6, "robot pose6")

        adapter, composite_metadata = experiment_v2.build_direct_model(
            experiment_cfg,
            robot_pose6,
        )
        manual_frame = base.load_manual_vessel_boundaries_with_frame(
            experiment_cfg.manual_boundary_path
        )
        roi_polygon = base.load_polygon(experiment_cfg.roi_polygon_path)

        # One new frame is used for all measured-curve, tip and overlay outputs.
        base.new_capture(filename=experiment_cfg.image_filename)
        vision_result = base.bounds_beam.reconstruct_beam_within_vessel(
            image_filename=experiment_cfg.image_filename,
            red_roi_path=experiment_cfg.roi_polygon_path,
            red_roi_polygon=roi_polygon,
            blue_roi_path=experiment_cfg.blue_roi_path,
            green_roi_path=experiment_cfg.green_roi_path,
            pivot_hint=experiment_cfg.pivot_hint_px,
            show=False,
            save_overlay_path=None,
            base_px_ref=manual_frame["base_px"],
            ex_ref=manual_frame["ex_img"],
            ey_ref=manual_frame["ey_img"],
        )
        vision_result["base_px_ref"] = manual_frame["base_px"]
        vision_result["ex_ref"] = manual_frame["ex_img"]
        vision_result["ey_ref"] = manual_frame["ey_img"]
        measured_output = np.asarray(
            base.vision_result_to_x_meas_robot(
                vision_result,
                pivot_point_pose6=np.asarray(experiment_cfg.pivot_pose6),
            ),
            dtype=float,
        ).reshape(-1)
        measured_tip, measured_tangent = experiment_v2._output_parts(
            measured_output,
            np.asarray(experiment_cfg.pivot_pose6[:3]),
        )

        # Read the pose again after capture so model/source pose corresponds as
        # closely as possible to the saved camera frame.
        robot_pose6 = hw.get_robot_pose_once()
        if robot_pose6 is None:
            raise RuntimeError("Could not re-read the UR pose after capture.")
        robot_pose6 = experiment_v2._finite_vector(robot_pose6, 6, "robot pose6")
        beam_length_mm = float(vision_result.get("beam_length_mm", np.nan))
        model_length_m = base._model_length_from_vision(
            experiment_cfg,
            beam_length_mm,
        )
        p8 = experiment_v2.robot_pose6_to_p8(
            robot_pose6,
            model_length_m,
            experiment_cfg.z_offset_m,
        )
        p7, model_output = adapter.commit_nominal(p8)
        model_tip, model_tangent = experiment_v2._output_parts(
            model_output,
            np.asarray(experiment_cfg.pivot_pose6[:3]),
        )

        pivot_pose6 = np.asarray(experiment_cfg.pivot_pose6, dtype=float)
        transform = pivot_transform(pivot_pose6)
        predicted_robot_m = extract_predicted_centerline(adapter, pivot_pose6[:3])
        predicted_local_m = robot_to_local(predicted_robot_m, transform)

        measured_px = np.asarray(vision_result["beam_centerline_px"], dtype=float)
        measured_px = _orient_polyline_from_reference(
            measured_px,
            np.asarray(manual_frame["base_px"], dtype=float),
        )
        measured_local_m = camera_pixels_to_local_m(
            measured_px,
            manual_frame["base_px"],
            manual_frame["ex_img"],
            manual_frame["ey_img"],
            float(vision_result["mm_per_pixel"]),
        )
        measured_robot_m = local_to_robot(measured_local_m, transform)

        predicted_rs, normalized_s = resample_polyline(
            predicted_robot_m,
            validation_cfg.comparison_points,
        )
        measured_rs, _ = resample_polyline(
            measured_robot_m,
            validation_cfg.comparison_points,
        )
        residual_robot_m = measured_rs - predicted_rs

        source_rotation = source_rotation_from_p8(p8)
        source_axes_robot = source_rotation.copy()
        m_body = np.asarray(adapter.raw_model.m_body, dtype=float).reshape(3)
        dipole_robot = _normalize(source_rotation @ m_body, "source dipole")
        dipole_local = transform[:3, :3].T @ dipole_robot
        source_local_m = robot_to_local(p8[:3][None, :], transform)[0]
        source_axes_local = transform[:3, :3].T @ source_axes_robot

        image_bgr = cv2.imread(experiment_cfg.image_filename)
        if image_bgr is None:
            raise FileNotFoundError(experiment_cfg.image_filename)

        return {
            "hardware": hw,
            "adapter": adapter,
            "composite_metadata": composite_metadata,
            "manual_frame": manual_frame,
            "vision_result": vision_result,
            "image_bgr": image_bgr,
            "robot_pose6": robot_pose6,
            "p8": p8,
            "p7": p7,
            "pivot_pose6": pivot_pose6,
            "pivot_transform": transform,
            "measured_output": measured_output,
            "model_output": model_output,
            "measured_tip_robot_m": measured_tip,
            "model_tip_robot_m": model_tip,
            "measured_tangent_robot": measured_tangent,
            "model_tangent_robot": model_tangent,
            "measured_centerline_px": measured_px,
            "measured_centerline_local_m": measured_local_m,
            "measured_centerline_robot_m": measured_robot_m,
            "predicted_centerline_local_m": predicted_local_m,
            "predicted_centerline_robot_m": predicted_robot_m,
            "measured_resampled_robot_m": measured_rs,
            "predicted_resampled_robot_m": predicted_rs,
            "residual_robot_m": residual_robot_m,
            "normalized_s": normalized_s,
            "source_rotation_robot": source_rotation,
            "source_axes_robot": source_axes_robot,
            "source_position_local_m": source_local_m,
            "source_axes_local": source_axes_local,
            "dipole_robot": dipole_robot,
            "dipole_local": dipole_local,
            "model_length_m": model_length_m,
        }
    except Exception:
        base._safe_stop(hw)
        raise


# =============================================================================
# METRICS AND CHECKS
# =============================================================================


def build_metrics_and_checks(
    state: dict[str, Any],
    cfg: FrameValidationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pivot_xyz = state["pivot_pose6"][:3]
    predicted = state["predicted_centerline_robot_m"]
    measured = state["measured_centerline_robot_m"]
    residual = state["residual_robot_m"]
    predicted_length_mm = 1.0e3 * _polyline_length(predicted)
    measured_length_mm = 1.0e3 * _polyline_length(measured)

    measured_tip_pipeline_mm = 1.0e3 * float(
        np.linalg.norm(state["measured_tip_robot_m"] - measured[-1])
    )
    model_tip_pipeline_mm = 1.0e3 * float(
        np.linalg.norm(state["model_tip_robot_m"] - predicted[-1])
    )
    tip_error_mm = 1.0e3 * float(
        np.linalg.norm(state["measured_tip_robot_m"] - state["model_tip_robot_m"])
    )
    centerline_norm_mm = 1.0e3 * np.linalg.norm(residual, axis=1)
    quaternion_norm = float(np.linalg.norm(state["p8"][3:7]))
    pivot_det = float(np.linalg.det(state["pivot_transform"][:3, :3]))
    source_det = float(np.linalg.det(state["source_rotation_robot"]))

    metrics = {
        "predicted_base_to_pivot_mm": 1.0e3
        * float(np.linalg.norm(predicted[0] - pivot_xyz)),
        "measured_base_to_pivot_mm": 1.0e3
        * float(np.linalg.norm(measured[0] - pivot_xyz)),
        "measured_tip_vs_centerline_tip_mm": measured_tip_pipeline_mm,
        "model_tip_vs_centerline_tip_mm": model_tip_pipeline_mm,
        "measured_model_tip_error_mm": tip_error_mm,
        "centerline_rmse_mm": float(np.sqrt(np.mean(centerline_norm_mm**2))),
        "centerline_max_error_mm": float(np.max(centerline_norm_mm)),
        "predicted_centerline_length_mm": predicted_length_mm,
        "measured_centerline_length_mm": measured_length_mm,
        "beam_length_difference_mm": abs(predicted_length_mm - measured_length_mm),
        "vision_reported_beam_length_mm": float(
            state["vision_result"].get("beam_length_mm", np.nan)
        ),
        "source_quaternion_norm": quaternion_norm,
        "pivot_rotation_determinant": pivot_det,
        "source_rotation_determinant": source_det,
        "measured_tangent_angle_xy_deg": experiment_v2._plane_angle_deg(
            state["measured_tangent_robot"], (0, 1)
        ),
        "model_tangent_angle_xy_deg": experiment_v2._plane_angle_deg(
            state["model_tangent_robot"], (0, 1)
        ),
        "source_distance_from_pivot_mm": 1.0e3
        * float(np.linalg.norm(state["p8"][:3] - pivot_xyz)),
        "source_out_of_camera_plane_local_mm": 1.0e3
        * float(state["source_position_local_m"][2]),
        "predicted_max_out_of_plane_local_mm": 1.0e3
        * float(np.max(np.abs(state["predicted_centerline_local_m"][:, 2]))),
    }

    checks = {
        "predicted_base_matches_pivot": {
            "passed": metrics["predicted_base_to_pivot_mm"]
            <= cfg.predicted_base_to_pivot_tolerance_mm,
            "value": metrics["predicted_base_to_pivot_mm"],
            "limit": cfg.predicted_base_to_pivot_tolerance_mm,
            "units": "mm",
        },
        "measured_base_matches_pivot": {
            "passed": metrics["measured_base_to_pivot_mm"]
            <= cfg.measured_base_to_pivot_tolerance_mm,
            "value": metrics["measured_base_to_pivot_mm"],
            "limit": cfg.measured_base_to_pivot_tolerance_mm,
            "units": "mm",
        },
        "measured_tip_pipeline_consistent": {
            "passed": measured_tip_pipeline_mm
            <= cfg.measured_tip_pipeline_tolerance_mm,
            "value": measured_tip_pipeline_mm,
            "limit": cfg.measured_tip_pipeline_tolerance_mm,
            "units": "mm",
        },
        "model_tip_pipeline_consistent": {
            "passed": model_tip_pipeline_mm <= cfg.model_tip_pipeline_tolerance_mm,
            "value": model_tip_pipeline_mm,
            "limit": cfg.model_tip_pipeline_tolerance_mm,
            "units": "mm",
        },
        "measured_model_tip_close": {
            "passed": tip_error_mm <= cfg.measured_model_tip_tolerance_mm,
            "value": tip_error_mm,
            "limit": cfg.measured_model_tip_tolerance_mm,
            "units": "mm",
        },
        "centerlines_close": {
            "passed": metrics["centerline_rmse_mm"]
            <= cfg.centerline_rmse_tolerance_mm,
            "value": metrics["centerline_rmse_mm"],
            "limit": cfg.centerline_rmse_tolerance_mm,
            "units": "mm",
        },
        "beam_lengths_close": {
            "passed": metrics["beam_length_difference_mm"]
            <= cfg.beam_length_difference_tolerance_mm,
            "value": metrics["beam_length_difference_mm"],
            "limit": cfg.beam_length_difference_tolerance_mm,
            "units": "mm",
        },
        "source_quaternion_normalized": {
            "passed": abs(quaternion_norm - 1.0) <= cfg.quaternion_norm_tolerance,
            "value": abs(quaternion_norm - 1.0),
            "limit": cfg.quaternion_norm_tolerance,
            "units": "absolute error",
        },
        "pivot_rotation_right_handed": {
            "passed": abs(pivot_det - 1.0) <= cfg.rotation_determinant_tolerance,
            "value": abs(pivot_det - 1.0),
            "limit": cfg.rotation_determinant_tolerance,
            "units": "determinant error",
        },
        "source_rotation_right_handed": {
            "passed": abs(source_det - 1.0) <= cfg.rotation_determinant_tolerance,
            "value": abs(source_det - 1.0),
            "limit": cfg.rotation_determinant_tolerance,
            "units": "determinant error",
        },
    }
    checks["all_passed"] = bool(all(item["passed"] for item in checks.values()))
    return metrics, checks


# =============================================================================
# PLOTS AND FILE OUTPUTS
# =============================================================================


AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_LABELS = ("x", "y", "z")


def _draw_triad_3d(
    axis: Any,
    origin_mm: np.ndarray,
    rotation: np.ndarray,
    length_mm: float,
    label_prefix: str,
) -> None:
    for index, (color, name) in enumerate(zip(AXIS_COLORS, AXIS_LABELS)):
        vector = rotation[:, index] * length_mm
        axis.quiver(
            *origin_mm,
            *vector,
            color=color,
            linewidth=1.7,
            arrow_length_ratio=0.18,
        )
        end = origin_mm + vector
        axis.text(*end, f"{label_prefix}{name}", color=color, fontsize=8)


def _set_equal_3d_limits(axis: Any, points_mm: np.ndarray) -> None:
    """Use one physical scale on all axes so pose geometry is not distorted."""

    points = np.asarray(points_mm, dtype=float).reshape(-1, 3)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = 0.5 * (low + high)
    half_range = max(0.5 * float(np.max(high - low)), 1.0)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def make_camera_overlay(
    state: dict[str, Any],
    cfg: FrameValidationConfig,
) -> np.ndarray:
    import cv2

    image = state["image_bgr"].copy()
    manual = state["manual_frame"]
    base_px = np.asarray(manual["base_px"], dtype=float)
    ex_img = np.asarray(manual["ex_img"], dtype=float)
    ey_img = np.asarray(manual["ey_img"], dtype=float)
    scale = float(state["vision_result"]["mm_per_pixel"])
    predicted_px = local_m_to_camera_pixels(
        state["predicted_centerline_local_m"],
        base_px,
        ex_img,
        ey_img,
        scale,
    )
    source_px = local_m_to_camera_pixels(
        state["source_position_local_m"][None, :],
        base_px,
        ex_img,
        ey_img,
        scale,
    )[0]
    measured_px = state["measured_centerline_px"]

    def polyline(points: np.ndarray, color: tuple[int, int, int], thickness: int):
        integer_points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [integer_points], False, color, thickness, cv2.LINE_AA)

    polyline(measured_px, (255, 255, 0), 3)  # cyan in BGR
    polyline(predicted_px, (255, 0, 255), 2)  # magenta
    cv2.circle(image, tuple(np.round(base_px).astype(int)), 7, (0, 255, 0), -1)
    cv2.putText(
        image,
        "pivot/base",
        tuple((base_px + [8, -8]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    height, width = image.shape[:2]
    if 0 <= source_px[0] < width and 0 <= source_px[1] < height:
        cv2.drawMarker(
            image,
            tuple(np.round(source_px).astype(int)),
            (0, 165, 255),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            image,
            f"source magnet (local z={1e3*state['source_position_local_m'][2]:+.1f} mm)",
            tuple((source_px + [8, -8]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    ex = ex_img / np.linalg.norm(ex_img)
    ey = ey_img / np.linalg.norm(ey_img)
    for vector, color, label in (
        (ex, (0, 0, 255), "camera ex"),
        (ey, (0, 255, 0), "camera ey"),
    ):
        image_vector = np.array([vector[0], -vector[1]])
        endpoint = base_px + cfg.camera_axis_display_length_px * image_vector
        cv2.arrowedLine(
            image,
            tuple(np.round(base_px).astype(int)),
            tuple(np.round(endpoint).astype(int)),
            color,
            2,
            cv2.LINE_AA,
            tipLength=0.15,
        )
        cv2.putText(
            image,
            label,
            tuple(np.round(endpoint).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "cyan: measured   magenta: model projection",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def make_validation_figure(
    state: dict[str, Any],
    metrics: dict[str, Any],
    checks: dict[str, Any],
    cfg: FrameValidationConfig,
    output_path: Path,
) -> None:
    import cv2

    camera_overlay = make_camera_overlay(state, cfg)
    predicted_relative_mm = (
        state["predicted_centerline_robot_m"] - state["pivot_pose6"][:3]
    ) * 1.0e3
    measured_relative_mm = (
        state["measured_centerline_robot_m"] - state["pivot_pose6"][:3]
    ) * 1.0e3
    source_relative_mm = (state["p8"][:3] - state["pivot_pose6"][:3]) * 1.0e3
    predicted_local_mm = state["predicted_centerline_local_m"] * 1.0e3
    measured_local_mm = state["measured_centerline_local_m"] * 1.0e3
    source_local_mm = state["source_position_local_m"] * 1.0e3
    residual_mm = state["residual_robot_m"] * 1.0e3

    figure = plt.figure(figsize=(16, 12))
    grid = figure.add_gridspec(2, 2)
    camera_axis = figure.add_subplot(grid[0, 0])
    robot_axis = figure.add_subplot(grid[0, 1], projection="3d")
    local_axis = figure.add_subplot(grid[1, 0])
    residual_axis = figure.add_subplot(grid[1, 1])

    camera_axis.imshow(cv2.cvtColor(camera_overlay, cv2.COLOR_BGR2RGB))
    camera_axis.set_title("Camera frame and projected model")
    camera_axis.axis("off")

    robot_axis.plot(
        *predicted_relative_mm.T,
        color="magenta",
        linewidth=2.5,
        label="model solution",
    )
    robot_axis.plot(
        *measured_relative_mm.T,
        color="cyan",
        linewidth=2.5,
        label="camera measurement",
    )
    robot_axis.scatter(0.0, 0.0, 0.0, color="black", s=35, label="pivot/base")
    robot_axis.scatter(
        *source_relative_mm,
        color="orange",
        marker="D",
        s=55,
        label="source magnet",
    )
    _draw_triad_3d(
        robot_axis,
        np.zeros(3),
        state["pivot_transform"][:3, :3],
        cfg.pivot_axis_display_length_mm,
        "base ",
    )
    _draw_triad_3d(
        robot_axis,
        source_relative_mm,
        state["source_axes_robot"],
        cfg.source_axis_display_length_mm,
        "src ",
    )
    robot_axis.quiver(
        *source_relative_mm,
        *(state["dipole_robot"] * cfg.dipole_arrow_display_length_mm),
        color="orange",
        linewidth=2.5,
        arrow_length_ratio=0.2,
        label="source dipole",
    )
    robot_axis.set_xlabel("robot X relative to pivot (mm)")
    robot_axis.set_ylabel("robot Y relative to pivot (mm)")
    robot_axis.set_zlabel("robot Z relative to pivot (mm)")
    robot_axis.set_title("Robot-frame geometry and source pose")
    _set_equal_3d_limits(
        robot_axis,
        np.vstack([predicted_relative_mm, measured_relative_mm, source_relative_mm]),
    )
    robot_axis.legend(loc="best", fontsize=8)

    local_axis.plot(
        predicted_local_mm[:, 0],
        predicted_local_mm[:, 1],
        color="magenta",
        linewidth=2.5,
        label="model solution",
    )
    local_axis.plot(
        measured_local_mm[:, 0],
        measured_local_mm[:, 1],
        color="tab:blue",
        linewidth=2.5,
        label="camera measurement",
    )
    local_axis.scatter(0.0, 0.0, color="black", s=35, label="pivot/base")
    local_axis.scatter(
        source_local_mm[0],
        source_local_mm[1],
        color="orange",
        marker="D",
        s=55,
        label="source projection",
    )
    dipole_xy = state["dipole_local"][:2]
    dipole_xy_norm = float(np.linalg.norm(dipole_xy))
    if dipole_xy_norm > 1.0e-12:
        dipole_xy = dipole_xy / dipole_xy_norm
        local_axis.arrow(
            source_local_mm[0],
            source_local_mm[1],
            cfg.dipole_arrow_display_length_mm * dipole_xy[0],
            cfg.dipole_arrow_display_length_mm * dipole_xy[1],
            width=0.25,
            head_width=1.5,
            color="orange",
            length_includes_head=True,
        )
    for axis_index, (color, label) in enumerate(zip(AXIS_COLORS, AXIS_LABELS)):
        body_axis_xy = state["source_axes_local"][:2, axis_index].copy()
        projected_norm = float(np.linalg.norm(body_axis_xy))
        if projected_norm < 1.0e-12:
            continue
        body_axis_xy /= projected_norm
        endpoint = source_local_mm[:2] + (
            cfg.source_axis_display_length_mm * body_axis_xy
        )
        local_axis.annotate(
            "",
            xy=endpoint,
            xytext=source_local_mm[:2],
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.6},
        )
        local_axis.text(endpoint[0], endpoint[1], f"src {label}", color=color, fontsize=8)
    local_axis.axhline(0.0, color="0.8", linewidth=1)
    local_axis.axvline(0.0, color="0.8", linewidth=1)
    local_axis.set_aspect("equal", adjustable="datalim")
    local_axis.set_xlabel("pivot-local X (mm)")
    local_axis.set_ylabel("pivot-local Y (mm)")
    local_axis.set_title(
        "Pivot-local camera plane\n"
        f"source local z = {source_local_mm[2]:+.2f} mm"
    )
    local_axis.grid(True, alpha=0.3)
    local_axis.legend(loc="best", fontsize=8)

    residual_axis.plot(
        state["normalized_s"],
        residual_mm[:, 0],
        label="measured-model robot X",
    )
    residual_axis.plot(
        state["normalized_s"],
        residual_mm[:, 1],
        label="measured-model robot Y",
    )
    residual_axis.plot(
        state["normalized_s"],
        residual_mm[:, 2],
        label="measured-model robot Z",
    )
    residual_axis.plot(
        state["normalized_s"],
        np.linalg.norm(residual_mm, axis=1),
        color="black",
        linewidth=2,
        label="error norm",
    )
    residual_axis.axhline(0.0, color="0.7", linewidth=1)
    residual_axis.set_xlabel("normalized beam arc length")
    residual_axis.set_ylabel("centerline residual (mm)")
    residual_axis.set_title(
        f"Centerline RMSE {metrics['centerline_rmse_mm']:.2f} mm; "
        f"tip error {metrics['measured_model_tip_error_mm']:.2f} mm"
    )
    residual_axis.grid(True, alpha=0.3)
    residual_axis.legend(loc="best", fontsize=8)

    failed = [name for name, item in checks.items() if name != "all_passed" and not item["passed"]]
    status = "ALL CHECKS PASSED" if checks["all_passed"] else "CHECK: " + ", ".join(failed)
    figure.suptitle(f"Nominal beam frame validation — {status}", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    if cfg.show_interactively:
        plt.show()
    plt.close(figure)


def save_points_csv(state: dict[str, Any], path: Path) -> None:
    measured = state["measured_resampled_robot_m"]
    predicted = state["predicted_resampled_robot_m"]
    residual = state["residual_robot_m"]
    measured_local = robot_to_local(measured, state["pivot_transform"])
    predicted_local = robot_to_local(predicted, state["pivot_transform"])
    fields = [
        "normalized_s",
        "measured_robot_x_m",
        "measured_robot_y_m",
        "measured_robot_z_m",
        "model_robot_x_m",
        "model_robot_y_m",
        "model_robot_z_m",
        "measured_local_x_m",
        "measured_local_y_m",
        "measured_local_z_m",
        "model_local_x_m",
        "model_local_y_m",
        "model_local_z_m",
        "residual_robot_x_mm",
        "residual_robot_y_mm",
        "residual_robot_z_mm",
        "residual_norm_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, normalized_s in enumerate(state["normalized_s"]):
            writer.writerow(
                {
                    "normalized_s": normalized_s,
                    "measured_robot_x_m": measured[index, 0],
                    "measured_robot_y_m": measured[index, 1],
                    "measured_robot_z_m": measured[index, 2],
                    "model_robot_x_m": predicted[index, 0],
                    "model_robot_y_m": predicted[index, 1],
                    "model_robot_z_m": predicted[index, 2],
                    "measured_local_x_m": measured_local[index, 0],
                    "measured_local_y_m": measured_local[index, 1],
                    "measured_local_z_m": measured_local[index, 2],
                    "model_local_x_m": predicted_local[index, 0],
                    "model_local_y_m": predicted_local[index, 1],
                    "model_local_z_m": predicted_local[index, 2],
                    "residual_robot_x_mm": 1.0e3 * residual[index, 0],
                    "residual_robot_y_mm": 1.0e3 * residual[index, 1],
                    "residual_robot_z_mm": 1.0e3 * residual[index, 2],
                    "residual_norm_mm": 1.0e3 * np.linalg.norm(residual[index]),
                }
            )


def save_summary(
    state: dict[str, Any],
    metrics: dict[str, Any],
    checks: dict[str, Any],
    validation_cfg: FrameValidationConfig,
    path: Path,
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "experiment_config": asdict(EXPERIMENT_CONFIG),
        "validation_config": asdict(validation_cfg),
        "composite": state["composite_metadata"],
        "frame_convention": {
            "p8": "[source_xyz_robot_model, quaternion_wxyz, insertion_length]",
            "p7": "[source_xyz_robot_model, rotation_vector, insertion_length]",
            "z_offset_application": (
                "Applied exactly once inside "
                "build_measured_p8_from_pose6_and_length"
            ),
            "camera_local_x": "negative projection on saved ex_img",
            "camera_local_y": "positive projection on saved ey_img",
            "camera_local_z": "zero by single-camera planar assumption",
            "local_to_robot": "pivot_translation + pivot_rotation @ local",
        },
        "robot_pose6_raw": state["robot_pose6"],
        "pivot_pose6": state["pivot_pose6"],
        "pivot_transform": state["pivot_transform"],
        "source_p8": state["p8"],
        "source_p7": state["p7"],
        "source_rotation_robot": state["source_rotation_robot"],
        "source_axes_robot_columns": state["source_axes_robot"],
        "source_position_local_m": state["source_position_local_m"],
        "source_axes_local_columns": state["source_axes_local"],
        "source_dipole_direction_robot": state["dipole_robot"],
        "source_dipole_direction_local": state["dipole_local"],
        "measured_tip_robot_m": state["measured_tip_robot_m"],
        "model_tip_robot_m": state["model_tip_robot_m"],
        "measured_tangent_robot": state["measured_tangent_robot"],
        "model_tangent_robot": state["model_tangent_robot"],
        "metrics": metrics,
        "checks": checks,
        "physical_assumptions_to_verify": [
            "The saved image-frame ex_img/ey_img vectors use the intended signs.",
            "The camera is treated as planar; measured local z is set to zero.",
            "The pivot pose maps that camera-local plane into robot coordinates.",
            "The existing p8 helper applies the correct z-offset sign and magnitude.",
            "The source quaternion is wxyz and the raw model uses a rotation vector in p7.",
            "raw_model.m_body is expressed in the source-magnet body frame.",
            "The predicted cache.centerline is expressed in robot/model coordinates.",
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def make_output_directory(cfg: FrameValidationConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> None:
    import cv2

    experiment_v2.validate_config(EXPERIMENT_CONFIG)
    cfg = CONFIG
    output_dir = make_output_directory(cfg)
    state: dict[str, Any] | None = None
    base = experiment_v2._base_module()
    try:
        state = capture_validation_state(EXPERIMENT_CONFIG, cfg)
        metrics, checks = build_metrics_and_checks(state, cfg)
        camera_overlay = make_camera_overlay(state, cfg)
        if cfg.save_camera_overlay:
            cv2.imwrite(
                str(output_dir / "camera_frame_with_geometry.png"),
                camera_overlay,
            )
        make_validation_figure(
            state,
            metrics,
            checks,
            cfg,
            output_dir / "frame_validation.png",
        )
        save_points_csv(state, output_dir / "frame_validation_points.csv")
        save_summary(
            state,
            metrics,
            checks,
            cfg,
            output_dir / "frame_validation_summary.json",
        )

        print("\nFRAME VALIDATION RESULTS")
        for name, item in checks.items():
            if name == "all_passed":
                continue
            marker = "PASS" if item["passed"] else "CHECK"
            print(
                f"[{marker:5s}] {name}: {item['value']:.4g} "
                f"{item['units']} (limit {item['limit']:.4g})"
            )
        print(f"Overall: {'PASS' if checks['all_passed'] else 'REVIEW FAILED CHECKS'}")
        print(f"Figure:  {output_dir / 'frame_validation.png'}")
        print(f"Summary: {output_dir / 'frame_validation_summary.json'}")
        print(f"Points:  {output_dir / 'frame_validation_points.csv'}")

        if cfg.strict_checks and not checks["all_passed"]:
            raise RuntimeError("One or more strict frame-validation checks failed.")
    finally:
        if state is not None and state.get("hardware") is not None:
            base._safe_stop(state["hardware"])


if __name__ == "__main__":
    main()
