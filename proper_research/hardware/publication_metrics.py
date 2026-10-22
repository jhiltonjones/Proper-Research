from __future__ import annotations

import csv
import json
import math
import os
import platform
import socket
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_EPS = 1.0e-12


def _as_vector(value: Any, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"Expected vector length {size}, got {array.size}.")
    return array


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _norm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return float("nan")
    return float(np.linalg.norm(vector))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    denom = _norm(a) * _norm(b)
    if not np.isfinite(denom) or denom <= _EPS:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _angle_deg_from_cosine(cosine: float) -> float:
    if not np.isfinite(cosine):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _path_error_components(
    position: np.ndarray,
    reference: np.ndarray,
    tangent: np.ndarray,
) -> tuple[float, float]:
    """Return local 3D contour magnitude and signed lag error in metres."""
    position = _as_vector(position)[:3]
    reference = _as_vector(reference)[:3]
    tangent = _as_vector(tangent)[:3]
    tangent_norm = _norm(tangent)
    if (
        position.size < 3
        or reference.size < 3
        or tangent.size < 3
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(reference))
        or not np.isfinite(tangent_norm)
        or tangent_norm <= _EPS
    ):
        return float("nan"), float("nan")
    tangent = tangent / tangent_norm
    error = position - reference
    lag = float(np.dot(tangent, error))
    contour = error - lag * tangent
    return _norm(contour), lag


def vector_components(prefix: str, vector: np.ndarray, *, scale: float = 1.0) -> dict[str, float]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    names = ("x", "y", "z")
    out: dict[str, float] = {}
    for index, name in enumerate(names):
        out[f"{prefix}_{name}"] = (
            float(scale * vector[index]) if index < vector.size else float("nan")
        )
    return out


@dataclass
class PreviousHardwarePlan:
    """Information required to validate one command on the next camera frame."""

    step: int
    x_meas: np.ndarray
    u0: np.ndarray
    B_first: np.ndarray | None
    x_pred0: np.ndarray | None
    reference0: np.ndarray | None
    robot_pose6_before: np.ndarray | None
    command_executed: bool
    monotonic_s: float
    reference_tangent0: np.ndarray | None = None

    def copy(self) -> "PreviousHardwarePlan":
        return PreviousHardwarePlan(
            step=int(self.step),
            x_meas=np.asarray(self.x_meas, dtype=float).copy(),
            u0=np.asarray(self.u0, dtype=float).copy(),
            B_first=(
                None
                if self.B_first is None
                else np.asarray(self.B_first, dtype=float).copy()
            ),
            x_pred0=(
                None
                if self.x_pred0 is None
                else np.asarray(self.x_pred0, dtype=float).copy()
            ),
            reference0=(
                None
                if self.reference0 is None
                else np.asarray(self.reference0, dtype=float).copy()
            ),
            robot_pose6_before=(
                None
                if self.robot_pose6_before is None
                else np.asarray(self.robot_pose6_before, dtype=float).copy()
            ),
            command_executed=bool(self.command_executed),
            monotonic_s=float(self.monotonic_s),
            reference_tangent0=(
                None
                if self.reference_tangent0 is None
                else np.asarray(self.reference_tangent0, dtype=float).copy()
            ),
        )


def empty_hardware_transition_metrics() -> dict[str, Any]:
    keys = {
        "jac_hw_valid": 0,
        "jac_hw_prev_step": -1,
        "jac_hw_elapsed_s": np.nan,
        "jac_hw_gain_xy": np.nan,
        "jac_hw_gain_xyz": np.nan,
        "jac_hw_cos_xy": np.nan,
        "jac_hw_cos_xyz": np.nan,
        "jac_hw_angle_xy_deg": np.nan,
        "jac_hw_angle_xyz_deg": np.nan,
        "jac_hw_pred_norm_xy_mm": np.nan,
        "jac_hw_pred_norm_xyz_mm": np.nan,
        "jac_hw_real_norm_xy_mm": np.nan,
        "jac_hw_real_norm_xyz_mm": np.nan,
        "jac_hw_err_norm_xy_mm": np.nan,
        "jac_hw_err_norm_xyz_mm": np.nan,
        "jac_hw_relative_err_xy": np.nan,
        "jac_hw_relative_err_xyz": np.nan,
        "tracking_prev_before_xy_mm": np.nan,
        "tracking_prev_after_xy_mm": np.nan,
        "tracking_prev_improvement_xy_mm": np.nan,
        "tracking_prev_predicted_xy_mm": np.nan,
        "tracking_prev_before_contour_mm": np.nan,
        "tracking_prev_after_contour_mm": np.nan,
        "tracking_prev_improvement_contour_mm": np.nan,
        "tracking_prev_before_lag_mm": np.nan,
        "tracking_prev_after_lag_mm": np.nan,
        "tracking_prev_predicted_contour_mm": np.nan,
        "tracking_prev_predicted_lag_mm": np.nan,
        "prediction_prev_actual_error_xy_mm": np.nan,
        "prediction_prev_actual_error_xyz_mm": np.nan,
        "robot_actual_translation_mm": np.nan,
        "robot_actual_rotation_rad": np.nan,
    }
    for prefix in (
        "jac_hw_dx_pred_mm",
        "jac_hw_dx_real_mm",
        "jac_hw_error_mm",
        "prediction_prev_actual_error_mm",
    ):
        keys.update(vector_components(prefix, np.full(3, np.nan)))
    return keys


def compute_hardware_transition_metrics(
    previous: PreviousHardwarePlan | None,
    *,
    x_meas_now: np.ndarray,
    robot_pose6_now: np.ndarray | None,
    monotonic_now_s: float,
) -> dict[str, Any]:
    """
    Compare the previously commanded one-step Jacobian/MPC prediction with the
    next camera measurement.

    The controller's B_first matrix is already the discrete one-step input
    matrix used in x[k+1] = x[k] + B_first u[k]. Therefore no additional dt
    multiplier is applied here.
    """
    out = empty_hardware_transition_metrics()
    if previous is None or not previous.command_executed:
        return out
    if previous.B_first is None:
        return out

    x_previous = _as_vector(previous.x_meas)
    x_current = _as_vector(x_meas_now)
    B = np.asarray(previous.B_first, dtype=float)
    u = _as_vector(previous.u0)
    if B.ndim != 2 or B.shape[1] != u.size or B.shape[0] < 3:
        return out
    if x_previous.size < 3 or x_current.size < 3:
        return out
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(u))):
        return out

    dx_pred = (B @ u)[:3]
    dx_real = x_current[:3] - x_previous[:3]
    error = dx_real - dx_pred

    pred_xy = _norm(dx_pred[:2])
    pred_xyz = _norm(dx_pred)
    real_xy = _norm(dx_real[:2])
    real_xyz = _norm(dx_real)
    err_xy = _norm(error[:2])
    err_xyz = _norm(error)
    cos_xy = _cosine(dx_pred[:2], dx_real[:2])
    cos_xyz = _cosine(dx_pred, dx_real)

    out.update(
        {
            "jac_hw_valid": 1,
            "jac_hw_prev_step": int(previous.step),
            "jac_hw_elapsed_s": float(monotonic_now_s - previous.monotonic_s),
            "jac_hw_gain_xy": float(real_xy / max(pred_xy, _EPS)),
            "jac_hw_gain_xyz": float(real_xyz / max(pred_xyz, _EPS)),
            "jac_hw_cos_xy": cos_xy,
            "jac_hw_cos_xyz": cos_xyz,
            "jac_hw_angle_xy_deg": _angle_deg_from_cosine(cos_xy),
            "jac_hw_angle_xyz_deg": _angle_deg_from_cosine(cos_xyz),
            "jac_hw_pred_norm_xy_mm": 1.0e3 * pred_xy,
            "jac_hw_pred_norm_xyz_mm": 1.0e3 * pred_xyz,
            "jac_hw_real_norm_xy_mm": 1.0e3 * real_xy,
            "jac_hw_real_norm_xyz_mm": 1.0e3 * real_xyz,
            "jac_hw_err_norm_xy_mm": 1.0e3 * err_xy,
            "jac_hw_err_norm_xyz_mm": 1.0e3 * err_xyz,
            "jac_hw_relative_err_xy": float(err_xy / max(pred_xy, _EPS)),
            "jac_hw_relative_err_xyz": float(err_xyz / max(pred_xyz, _EPS)),
        }
    )
    out.update(vector_components("jac_hw_dx_pred_mm", dx_pred, scale=1.0e3))
    out.update(vector_components("jac_hw_dx_real_mm", dx_real, scale=1.0e3))
    out.update(vector_components("jac_hw_error_mm", error, scale=1.0e3))

    if previous.reference0 is not None:
        reference = _as_vector(previous.reference0)
        if reference.size >= 3 and np.all(np.isfinite(reference[:3])):
            before = _norm(x_previous[:2] - reference[:2])
            after = _norm(x_current[:2] - reference[:2])
            out["tracking_prev_before_xy_mm"] = 1.0e3 * before
            out["tracking_prev_after_xy_mm"] = 1.0e3 * after
            out["tracking_prev_improvement_xy_mm"] = 1.0e3 * (before - after)
            if previous.x_pred0 is not None:
                predicted = _as_vector(previous.x_pred0)
                if predicted.size >= 3:
                    out["tracking_prev_predicted_xy_mm"] = 1.0e3 * _norm(
                        predicted[:2] - reference[:2]
                    )

    if previous.reference0 is not None and previous.reference_tangent0 is not None:
        reference = _as_vector(previous.reference0)
        tangent = _as_vector(previous.reference_tangent0)
        if reference.size >= 3 and tangent.size >= 3:
            before_contour, before_lag = _path_error_components(
                x_previous[:3], reference[:3], tangent[:3]
            )
            after_contour, after_lag = _path_error_components(
                x_current[:3], reference[:3], tangent[:3]
            )
            out["tracking_prev_before_contour_mm"] = 1.0e3 * before_contour
            out["tracking_prev_after_contour_mm"] = 1.0e3 * after_contour
            out["tracking_prev_improvement_contour_mm"] = 1.0e3 * (
                before_contour - after_contour
            )
            out["tracking_prev_before_lag_mm"] = 1.0e3 * before_lag
            out["tracking_prev_after_lag_mm"] = 1.0e3 * after_lag
            if previous.x_pred0 is not None:
                predicted = _as_vector(previous.x_pred0)
                if predicted.size >= 3:
                    pred_contour, pred_lag = _path_error_components(
                        predicted[:3], reference[:3], tangent[:3]
                    )
                    out["tracking_prev_predicted_contour_mm"] = (
                        1.0e3 * pred_contour
                    )
                    out["tracking_prev_predicted_lag_mm"] = 1.0e3 * pred_lag

    if previous.x_pred0 is not None:
        predicted = _as_vector(previous.x_pred0)
        if predicted.size >= 3 and np.all(np.isfinite(predicted[:3])):
            prediction_error = x_current[:3] - predicted[:3]
            out["prediction_prev_actual_error_xy_mm"] = 1.0e3 * _norm(
                prediction_error[:2]
            )
            out["prediction_prev_actual_error_xyz_mm"] = 1.0e3 * _norm(
                prediction_error
            )
            out.update(
                vector_components(
                    "prediction_prev_actual_error_mm",
                    prediction_error,
                    scale=1.0e3,
                )
            )

    if previous.robot_pose6_before is not None and robot_pose6_now is not None:
        old_pose = _as_vector(previous.robot_pose6_before, 6)
        new_pose = _as_vector(robot_pose6_now, 6)
        if np.all(np.isfinite(old_pose)) and np.all(np.isfinite(new_pose)):
            out["robot_actual_translation_mm"] = 1.0e3 * _norm(
                new_pose[:3] - old_pose[:3]
            )
            out["robot_actual_rotation_rad"] = _norm(
                new_pose[3:6] - old_pose[3:6]
            )

    return out


def jacobian_matrix_metrics(B_first: np.ndarray | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "jac_B_available": 0,
        "jac_B_fro_norm": np.nan,
        "jac_B_spectral_norm": np.nan,
        "jac_B_sigma_min": np.nan,
        "jac_B_condition": np.nan,
        "jac_B_rank": -1,
    }
    for name in ("vx", "vy", "vz", "wx", "wy", "wz", "dL"):
        out[f"jac_B_col_norm_{name}"] = np.nan
    if B_first is None:
        return out
    B = np.asarray(B_first, dtype=float)
    if B.ndim != 2 or B.shape[0] < 2 or not np.all(np.isfinite(B)):
        return out
    try:
        singular_values = np.linalg.svd(B, compute_uv=False)
    except np.linalg.LinAlgError:
        return out
    if singular_values.size == 0:
        return out
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    out.update(
        {
            "jac_B_available": 1,
            "jac_B_fro_norm": float(np.linalg.norm(B)),
            "jac_B_spectral_norm": sigma_max,
            "jac_B_sigma_min": sigma_min,
            "jac_B_condition": float(sigma_max / max(sigma_min, _EPS)),
            "jac_B_rank": int(np.linalg.matrix_rank(B)),
        }
    )
    channel_names = ("vx", "vy", "vz", "wx", "wy", "wz", "dL")
    for index, name in enumerate(channel_names):
        if index < B.shape[1]:
            out[f"jac_B_col_norm_{name}"] = float(np.linalg.norm(B[:, index]))
    return out


def current_tracking_metrics(
    *,
    x_meas: np.ndarray,
    reference0: np.ndarray | None,
    x_pred0: np.ndarray | None,
    reference_tangent0: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Current reference errors.

    The legacy Euclidean XY/XYZ errors are retained for backwards comparison.
    When a continuous path tangent is supplied, contour and signed lag errors
    are also reported using exactly the local decomposition used by the new
    contouring objective.
    """
    out = {
        "tracking_ref_available": 0,
        "tracking_error_xy_mm": np.nan,
        "tracking_error_xyz_mm": np.nan,
        "tracking_predicted_error_xy_mm": np.nan,
        "tracking_predicted_error_xyz_mm": np.nan,
        "tracking_contour_error_mm": np.nan,
        "tracking_lag_error_mm": np.nan,
        "tracking_predicted_contour_error_mm": np.nan,
        "tracking_predicted_lag_error_mm": np.nan,
    }
    if reference0 is None:
        return out
    measurement = _as_vector(x_meas)
    reference = _as_vector(reference0)
    if measurement.size < 3 or reference.size < 3:
        return out
    out["tracking_ref_available"] = 1
    out["tracking_error_xy_mm"] = 1.0e3 * _norm(
        measurement[:2] - reference[:2]
    )
    out["tracking_error_xyz_mm"] = 1.0e3 * _norm(
        measurement[:3] - reference[:3]
    )

    tangent = None
    if reference_tangent0 is not None:
        tangent_candidate = _as_vector(reference_tangent0)
        if tangent_candidate.size >= 3:
            tangent = tangent_candidate[:3]
            contour, lag = _path_error_components(
                measurement[:3], reference[:3], tangent
            )
            out["tracking_contour_error_mm"] = 1.0e3 * contour
            out["tracking_lag_error_mm"] = 1.0e3 * lag

    if x_pred0 is not None:
        predicted = _as_vector(x_pred0)
        if predicted.size >= 3:
            out["tracking_predicted_error_xy_mm"] = 1.0e3 * _norm(
                predicted[:2] - reference[:2]
            )
            out["tracking_predicted_error_xyz_mm"] = 1.0e3 * _norm(
                predicted[:3] - reference[:3]
            )
            if tangent is not None:
                contour, lag = _path_error_components(
                    predicted[:3], reference[:3], tangent
                )
                out["tracking_predicted_contour_error_mm"] = 1.0e3 * contour
                out["tracking_predicted_lag_error_mm"] = 1.0e3 * lag
    return out


def model_linearisation_metrics(
    *,
    x_meas: np.ndarray,
    x_model0: np.ndarray | None,
    x_nonlinear_next: np.ndarray | None,
    x_pred0: np.ndarray | None,
    B_first: np.ndarray | None,
    u0: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_bias_xy_mm": np.nan,
        "model_bias_xyz_mm": np.nan,
        "jac_model_valid": 0,
        "jac_model_err_xy_mm": np.nan,
        "jac_model_err_xyz_mm": np.nan,
        "jac_model_gain_xy": np.nan,
        "jac_model_gain_xyz": np.nan,
        "jac_model_cos_xy": np.nan,
        "jac_model_cos_xyz": np.nan,
        "mpc_pred_vs_nonlinear_xy_mm": np.nan,
        "mpc_pred_vs_nonlinear_xyz_mm": np.nan,
    }
    for prefix in (
        "model_bias_mm",
        "jac_model_dx_pred_mm",
        "jac_model_dx_real_mm",
        "jac_model_error_mm",
        "mpc_pred_vs_nonlinear_mm",
    ):
        out.update(vector_components(prefix, np.full(3, np.nan)))

    measurement = _as_vector(x_meas)
    if x_model0 is not None:
        model0 = _as_vector(x_model0)
        if model0.size >= 3 and measurement.size >= 3:
            bias = measurement[:3] - model0[:3]
            out["model_bias_xy_mm"] = 1.0e3 * _norm(bias[:2])
            out["model_bias_xyz_mm"] = 1.0e3 * _norm(bias)
            out.update(vector_components("model_bias_mm", bias, scale=1.0e3))
    else:
        model0 = np.empty(0)

    if (
        x_model0 is None
        or x_nonlinear_next is None
        or B_first is None
        or model0.size < 3
    ):
        return out

    nonlinear = _as_vector(x_nonlinear_next)
    B = np.asarray(B_first, dtype=float)
    command = _as_vector(u0)
    if nonlinear.size < 3 or B.ndim != 2 or B.shape[1] != command.size:
        return out
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(command))):
        return out

    dx_pred = (B @ command)[:3]
    dx_real = nonlinear[:3] - model0[:3]
    error = dx_real - dx_pred
    pred_xy = _norm(dx_pred[:2])
    pred_xyz = _norm(dx_pred)
    real_xy = _norm(dx_real[:2])
    real_xyz = _norm(dx_real)

    out.update(
        {
            "jac_model_valid": 1,
            "jac_model_err_xy_mm": 1.0e3 * _norm(error[:2]),
            "jac_model_err_xyz_mm": 1.0e3 * _norm(error),
            "jac_model_gain_xy": float(real_xy / max(pred_xy, _EPS)),
            "jac_model_gain_xyz": float(real_xyz / max(pred_xyz, _EPS)),
            "jac_model_cos_xy": _cosine(dx_pred[:2], dx_real[:2]),
            "jac_model_cos_xyz": _cosine(dx_pred, dx_real),
        }
    )
    out.update(vector_components("jac_model_dx_pred_mm", dx_pred, scale=1.0e3))
    out.update(vector_components("jac_model_dx_real_mm", dx_real, scale=1.0e3))
    out.update(vector_components("jac_model_error_mm", error, scale=1.0e3))

    if x_pred0 is not None:
        predicted = _as_vector(x_pred0)
        if predicted.size >= 3:
            pred_nonlinear_error = predicted[:3] - nonlinear[:3]
            out["mpc_pred_vs_nonlinear_xy_mm"] = 1.0e3 * _norm(
                pred_nonlinear_error[:2]
            )
            out["mpc_pred_vs_nonlinear_xyz_mm"] = 1.0e3 * _norm(
                pred_nonlinear_error
            )
            out.update(
                vector_components(
                    "mpc_pred_vs_nonlinear_mm",
                    pred_nonlinear_error,
                    scale=1.0e3,
                )
            )
    return out


def default_run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pid": os.getpid(),
        "metric_definitions": {
            "jac_hw_dx_pred": "B_first @ u0 using the controller's discrete one-step B matrix",
            "jac_hw_dx_real": "camera_tip[k] - camera_tip[k-1] after an executed command",
            "jac_hw_gain": "norm(real displacement) / norm(predicted displacement)",
            "jac_hw_prediction_error": "real displacement - Jacobian-predicted displacement",
            "jac_model_prediction_error": "nonlinear model displacement - Jacobian-predicted displacement",
            "tracking_error": "camera tip to the first continuous MPC reference C(s); legacy Euclidean XY/XYZ metrics are retained",
            "tracking_contour_error": "norm((I - t t^T) (camera_tip - C(s))) at the first frozen path reference",
            "tracking_lag_error": "signed t^T (camera_tip - C(s)) at the first frozen path reference",
            "prediction_prev_actual_error": "next camera tip - previous MPC first-stage prediction",
        },
    }
    if extra:
        metadata.update(extra)
    return metadata


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = _finite_or_nan(row.get(key, np.nan))
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "rmse": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


class PublicationRunLogger:
    """Append-only per-step CSV plus metadata and publication summary JSON."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_name: str,
        metadata: dict[str, Any] | None = None,
        fsync_each_step: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = str(run_name)
        self.csv_path = self.output_dir / f"{self.run_name}_steps.csv"
        self.metadata_path = self.output_dir / f"{self.run_name}_metadata.json"
        self.summary_path = self.output_dir / f"{self.run_name}_summary.json"
        self.fsync_each_step = bool(fsync_each_step)
        self.rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] | None = None
        self.metadata = default_run_metadata(metadata)
        self.metadata_path.write_text(
            json.dumps(_json_value(self.metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def append(self, row: dict[str, Any]) -> None:
        clean = {str(key): _json_value(value) for key, value in row.items()}
        if self._fieldnames is None:
            self._fieldnames = list(clean.keys())
        else:
            missing = [key for key in self._fieldnames if key not in clean]
            extra = [key for key in clean if key not in self._fieldnames]
            if missing or extra:
                raise ValueError(
                    "Publication log schema changed between steps. "
                    f"Missing={missing}; extra={extra}."
                )
        exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow(clean)
            stream.flush()
            if self.fsync_each_step:
                os.fsync(stream.fileno())
        self.rows.append(clean)

    def finalize(self, *, stop_reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics = (
            "tracking_error_xy_mm",
            "tracking_error_xyz_mm",
            "tracking_contour_error_mm",
            "tracking_lag_error_mm",
            "tracking_predicted_contour_error_mm",
            "tracking_predicted_lag_error_mm",
            "tracking_prev_after_xy_mm",
            "tracking_prev_improvement_xy_mm",
            "tracking_prev_after_contour_mm",
            "tracking_prev_improvement_contour_mm",
            "tracking_prev_after_lag_mm",
            "prediction_prev_actual_error_xy_mm",
            "prediction_prev_actual_error_xyz_mm",
            "jac_hw_gain_xy",
            "jac_hw_gain_xyz",
            "jac_hw_cos_xy",
            "jac_hw_cos_xyz",
            "jac_hw_err_norm_xy_mm",
            "jac_hw_err_norm_xyz_mm",
            "jac_model_err_xy_mm",
            "jac_model_err_xyz_mm",
            "model_bias_xy_mm",
            "model_bias_xyz_mm",
            "clearance_mm",
            "tip_vessel_angle_deg",
            "path_projection_distance_mm",
            "progress_nominal_shortfall_mm",
            "progress_slack_solution_fraction",
            "controller_step_ms",
            "controller_solve_ms",
            "qp_build_ms",
            "osqp_ms",
            "forward_time_ms",
            "jacobian_time_ms",
            "loop_wall_ms",
        )
        summary: dict[str, Any] = {
            "run_name": self.run_name,
            "stop_reason": str(stop_reason),
            "steps": len(self.rows),
            "executed_commands": int(
                sum(int(bool(row.get("command_executed", 0))) for row in self.rows)
            ),
            "valid_hardware_jacobian_transitions": int(
                sum(int(bool(row.get("jac_hw_valid", 0))) for row in self.rows)
            ),
            "controller_success_rate": (
                float(
                    np.mean(
                        [
                            1.0 - float(bool(row.get("infeasible", 0)))
                            for row in self.rows
                        ]
                    )
                )
                if self.rows
                else None
            ),
            "metrics": {
                key: _summary_stats(_numeric_values(self.rows, key))
                for key in metrics
            },
        }
        if extra:
            summary.update(extra)
        self.summary_path.write_text(
            json.dumps(_json_value(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary
