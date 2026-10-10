#!/usr/bin/env python3
"""Analyse how global constrained configuration optimisation changes an inverse path.

This script is deliberately *output-only*: it does not import the robot/beam model,
run inverse kinematics, or re-run the global optimiser.  It compares the files saved
by:

  inverse_configuration_path.csv
  inverse_configuration_summary.json

against:

  global_configuration_path.csv
  global_configuration_summary.json
  global_configuration_dense_validation.csv   (optional)

The most direct before/after comparison uses ``seed_*`` columns in the global CSV,
because these are the states actually presented to the global optimisation layer on
its own node grid.  The original inverse CSV is also used for tracking-error and
head-exclusion diagnostics.

Outputs written to --output-dir:
  analysis_report.md
  analysis_metrics.json
  summary_metrics.csv
  per_coordinate_metrics.csv
  configuration_before_after.png
  smoothness_before_after.png
  tracking_constraints.png
  tip_path_comparison.png
  global_change_from_seed.png

Typical use:

  python analyse_global_configuration_improvement.py \
      --inverse-dir results/offline_inverse_configuration_60 \
      --global-dir results/global_configuration_full_debug

Interpretation for time parameterisation
-----------------------------------------
For a geometric path q(s), a time parameteriser sees

    q_dot  = q'(s) * s_dot
    q_ddot = q''(s) * s_dot**2 + q'(s) * s_ddot

so reducing large q'(s) and especially q''(s) excursions generally makes the path
less demanding to time-parameterise.  This script therefore reports first- and
second-derivative metrics with respect to lumen arc length s.  These are useful
TOPP proxies, not a substitute for running the same time-parameterisation algorithm
on both paths and comparing final traversal time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


COORD_NAMES = ("q1", "q2", "q3", "q4", "q5", "q6", "L")
COORD_UNITS = ("rad", "rad", "rad", "rad", "rad", "rad", "m")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved inverse and global configuration paths without "
            "re-running either planner."
        )
    )
    parser.add_argument(
        "--inverse-dir",
        required=True,
        type=Path,
        help="Directory containing inverse_configuration_path.csv/json.",
    )
    parser.add_argument(
        "--global-dir",
        required=True,
        type=Path,
        help="Directory containing global_configuration_path.csv/json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Analysis output directory. Default: <global-dir>/comparison_analysis"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Saved plot DPI (default: 180).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Compute tables/report only.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_float(row, key) for row in rows], dtype=float)


def _matrix(rows: list[dict[str, str]], keys: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(keys)), dtype=float)
    return np.asarray([[_float(row, key) for key in keys] for row in rows], dtype=float)


def _state_matrix(rows: list[dict[str, str]], prefix: str = "") -> np.ndarray:
    if prefix:
        keys = [f"{prefix}_q{i}" for i in range(1, 7)] + [f"{prefix}_L"]
    else:
        keys = [f"q{i}_rad" for i in range(1, 7)] + ["insertion_m"]
    return _matrix(rows, keys)


def _xyz(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    return _matrix(rows, [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"])


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _safe_percent_improvement(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after) or abs(before) <= 1e-30:
        return np.nan
    return 100.0 * (before - after) / abs(before)


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values * values)))


def _p95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, 95.0))


def _max_abs(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        if axis is None:
            return float("nan")
        out_size = arr.shape[1] if arr.ndim == 2 and axis == 0 else 0
        return np.full(out_size, np.nan)
    return np.nanmax(np.abs(arr), axis=axis)


def _path_derivatives(
    s: np.ndarray,
    states: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ds, scaled dchi/ds, second derivative, raw delta states).

    The formulas match the first/second difference matrices in the global
    optimiser, including the non-uniform-grid second derivative.
    """
    s = np.asarray(s, dtype=float).reshape(-1)
    states = np.asarray(states, dtype=float).reshape(-1, 7)
    scale = np.asarray(scale, dtype=float).reshape(7)
    if s.size != states.shape[0]:
        raise ValueError("s and state node counts differ")
    if s.size < 2:
        return (
            np.empty(0),
            np.empty((0, 7)),
            np.empty((0, 7)),
            np.empty((0, 7)),
        )
    ds = np.diff(s)
    if np.any(~np.isfinite(ds)) or np.any(ds <= 0.0):
        raise ValueError("Path coordinates s_m must be finite and strictly increasing")
    delta = np.diff(states, axis=0)
    first = delta / ds[:, None] / scale[None, :]
    if first.shape[0] >= 2:
        second = 2.0 * np.diff(first, axis=0) / (ds[:-1] + ds[1:])[:, None]
    else:
        second = np.empty((0, 7), dtype=float)
    return ds, first, second, delta


def _smoothness_metrics(
    s: np.ndarray,
    states: np.ndarray,
    scale: np.ndarray,
    first_weight: np.ndarray,
    second_weight: np.ndarray,
) -> dict[str, Any]:
    ds, first, second, delta = _path_derivatives(s, states, scale)
    first_norm = np.linalg.norm(first, axis=1) if first.size else np.empty(0)
    second_norm = np.linalg.norm(second, axis=1) if second.size else np.empty(0)
    scaled_delta = delta / scale[None, :] if delta.size else np.empty((0, 7))

    # These reproduce the two smoothing contributions to the optimiser's
    # quadratic objective: 0.5 * (D x)^T W (D x).
    first_cost = (
        0.5 * float(np.sum(first * first * first_weight[None, :]))
        if first.size
        else 0.0
    )
    second_cost = (
        0.5 * float(np.sum(second * second * second_weight[None, :]))
        if second.size
        else 0.0
    )

    return {
        "ds": ds,
        "first": first,
        "second": second,
        "delta": delta,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "scaled_path_length": (
            float(np.sum(np.linalg.norm(scaled_delta, axis=1)))
            if scaled_delta.size
            else 0.0
        ),
        "first_norm_max": float(np.max(first_norm)) if first_norm.size else np.nan,
        "first_norm_rms": _rms(first_norm),
        "second_norm_max": (
            float(np.max(second_norm)) if second_norm.size else np.nan
        ),
        "second_norm_rms": _rms(second_norm),
        "weighted_first_difference_cost": first_cost,
        "weighted_second_difference_cost": second_cost,
        "weighted_smoothing_cost": first_cost + second_cost,
        "max_abs_delta": (
            np.max(np.abs(delta), axis=0) if delta.size else np.zeros(7)
        ),
        "max_abs_first": (
            np.max(np.abs(first), axis=0) if first.size else np.zeros(7)
        ),
        "rms_first": (
            np.sqrt(np.mean(first * first, axis=0)) if first.size else np.zeros(7)
        ),
        "max_abs_second": (
            np.max(np.abs(second), axis=0) if second.size else np.zeros(7)
        ),
        "rms_second": (
            np.sqrt(np.mean(second * second, axis=0)) if second.size else np.zeros(7)
        ),
    }


def _tracking_metrics(
    position_error_m: np.ndarray,
    tangent_error_rad: np.ndarray,
    position_tolerance_m: float,
    tangent_tolerance_rad: float,
) -> dict[str, float]:
    position_error_m = np.asarray(position_error_m, dtype=float)
    tangent_error_rad = np.asarray(tangent_error_rad, dtype=float)
    pos_score = position_error_m / position_tolerance_m
    tan_score = tangent_error_rad / tangent_tolerance_rad
    score = np.maximum(pos_score, tan_score)
    return {
        "position_error_max_m": float(np.nanmax(position_error_m)),
        "position_error_rms_m": _rms(position_error_m),
        "position_error_p95_m": _p95(position_error_m),
        "tangent_error_max_rad": float(np.nanmax(tangent_error_rad)),
        "tangent_error_rms_rad": _rms(tangent_error_rad),
        "tangent_error_p95_rad": _p95(tangent_error_rad),
        "normalized_task_score_max": float(np.nanmax(score)),
        "normalized_task_score_rms": _rms(score),
        "feasible_fraction_by_errors": float(np.mean(score <= 1.0 + 1e-12)),
    }


def _nearest_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Distance from each point to a 3-D polyline using segment projection."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    polyline = np.asarray(polyline, dtype=float).reshape(-1, 3)
    if polyline.shape[0] < 2:
        return np.full(points.shape[0], np.nan)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    length_sq = np.sum(vectors * vectors, axis=1)
    valid = length_sq > 1e-24
    starts = starts[valid]
    vectors = vectors[valid]
    length_sq = length_sq[valid]
    if starts.shape[0] == 0:
        return np.full(points.shape[0], np.nan)

    result = np.empty(points.shape[0], dtype=float)
    for index, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            result[index] = np.nan
            continue
        fraction = np.sum((point - starts) * vectors, axis=1) / length_sq
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = starts + fraction[:, None] * vectors
        result[index] = float(np.min(np.linalg.norm(closest - point, axis=1)))
    return result


def _reconstructed_centreline(
    inverse_s: np.ndarray,
    inverse_desired: np.ndarray,
    global_s: np.ndarray,
    global_desired: np.ndarray,
) -> np.ndarray:
    s = np.concatenate((inverse_s, global_s))
    points = np.vstack((inverse_desired, global_desired))
    finite = np.isfinite(s) & np.all(np.isfinite(points), axis=1)
    s = s[finite]
    points = points[finite]
    order = np.argsort(s)
    s = s[order]
    points = points[order]

    # Collapse repeated arc-length coordinates to one point.
    unique_points: list[np.ndarray] = []
    last_s: float | None = None
    for s_i, point in zip(s, points):
        if last_s is not None and abs(s_i - last_s) <= 1e-12:
            continue
        unique_points.append(point)
        last_s = float(s_i)
    return np.asarray(unique_points, dtype=float)


def _match_s_values(
    s_a: np.ndarray,
    s_b: np.ndarray,
    atol: float = 1e-9,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    j = 0
    for i, value in enumerate(s_a):
        while j + 1 < s_b.size and s_b[j + 1] <= value:
            j += 1
        candidates = [j]
        if j + 1 < s_b.size:
            candidates.append(j + 1)
        best = min(candidates, key=lambda k: abs(s_b[k] - value))
        if abs(s_b[best] - value) <= atol:
            pairs.append((i, best))
    return pairs


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["metric", "before", "after", "improvement_percent", "unit", "interpretation"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_coordinate_csv(
    path: Path,
    before: dict[str, Any],
    after: dict[str, Any],
    adjacent_limits: np.ndarray,
) -> None:
    fields = [
        "coordinate",
        "unit",
        "max_abs_step_before",
        "max_abs_step_after",
        "step_improvement_percent",
        "adjacent_limit",
        "limit_utilisation_before",
        "limit_utilisation_after",
        "max_abs_dstate_ds_scaled_before",
        "max_abs_dstate_ds_scaled_after",
        "dstate_ds_improvement_percent",
        "rms_dstate_ds_scaled_before",
        "rms_dstate_ds_scaled_after",
        "max_abs_d2state_ds2_scaled_before",
        "max_abs_d2state_ds2_scaled_after",
        "d2state_ds2_improvement_percent",
        "rms_d2state_ds2_scaled_before",
        "rms_d2state_ds2_scaled_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (name, unit) in enumerate(zip(COORD_NAMES, COORD_UNITS)):
            b_step = float(before["max_abs_delta"][i])
            a_step = float(after["max_abs_delta"][i])
            b_first = float(before["max_abs_first"][i])
            a_first = float(after["max_abs_first"][i])
            b_second = float(before["max_abs_second"][i])
            a_second = float(after["max_abs_second"][i])
            limit = float(adjacent_limits[i])
            writer.writerow(
                {
                    "coordinate": name,
                    "unit": unit,
                    "max_abs_step_before": b_step,
                    "max_abs_step_after": a_step,
                    "step_improvement_percent": _safe_percent_improvement(b_step, a_step),
                    "adjacent_limit": limit,
                    "limit_utilisation_before": b_step / limit if limit > 0 else np.nan,
                    "limit_utilisation_after": a_step / limit if limit > 0 else np.nan,
                    "max_abs_dstate_ds_scaled_before": b_first,
                    "max_abs_dstate_ds_scaled_after": a_first,
                    "dstate_ds_improvement_percent": _safe_percent_improvement(b_first, a_first),
                    "rms_dstate_ds_scaled_before": float(before["rms_first"][i]),
                    "rms_dstate_ds_scaled_after": float(after["rms_first"][i]),
                    "max_abs_d2state_ds2_scaled_before": b_second,
                    "max_abs_d2state_ds2_scaled_after": a_second,
                    "d2state_ds2_improvement_percent": _safe_percent_improvement(b_second, a_second),
                    "rms_d2state_ds2_scaled_before": float(before["rms_second"][i]),
                    "rms_d2state_ds2_scaled_after": float(after["rms_second"][i]),
                }
            )


def _format_metric(value: float, scale: float = 1.0, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{scale * value:.{digits}f}"


def _plot_configuration(
    output: Path,
    s: np.ndarray,
    seed: np.ndarray,
    optimized: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    axes = axes.ravel()
    for i in range(7):
        factor = 1e3 if i == 6 else 1.0
        unit = "mm" if i == 6 else "rad"
        axes[i].plot(s, factor * seed[:, i], "--", label="global seed / before")
        axes[i].plot(s, factor * optimized[:, i], label="global optimized / after")
        axes[i].set_xlabel("lumen arc length s [m]")
        axes[i].set_ylabel(f"{COORD_NAMES[i]} [{unit}]")
        axes[i].grid(True, alpha=0.25)
    axes[0].legend()
    change = (optimized - seed)
    change[:, 6] *= 1e3
    axes[7].plot(s, np.linalg.norm(change, axis=1))
    axes[7].set_xlabel("lumen arc length s [m]")
    axes[7].set_ylabel("state change norm [mixed; L in mm]")
    axes[7].set_title("Magnitude of global change from seed")
    axes[7].grid(True, alpha=0.25)
    fig.suptitle("Configuration path: input seed vs global optimisation")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_smoothness(
    output: Path,
    s: np.ndarray,
    before: dict[str, Any],
    after: dict[str, Any],
    scale: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    if before["first_norm"].size:
        axes[0].plot(s[:-1], before["first_norm"], "--", label="seed")
        axes[0].plot(s[:-1], after["first_norm"], label="global")
    axes[0].set_ylabel(r"$||d(\chi/scale)/ds||$")
    axes[0].set_title("Scaled first derivative — velocity-demand proxy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    if before["second_norm"].size:
        axes[1].plot(s[1:-1], before["second_norm"], "--", label="seed")
        axes[1].plot(s[1:-1], after["second_norm"], label="global")
    axes[1].set_ylabel(r"$||d^2(\chi/scale)/ds^2||$")
    axes[1].set_title("Scaled second derivative — acceleration-demand proxy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    if before["delta"].size:
        before_step = np.linalg.norm(before["delta"] / scale[None, :], axis=1)
        after_step = np.linalg.norm(after["delta"] / scale[None, :], axis=1)
        axes[2].plot(s[:-1], before_step, "--", label="seed")
        axes[2].plot(s[:-1], after_step, label="global")
    axes[2].set_xlabel("lumen arc length s [m]")
    axes[2].set_ylabel("scaled adjacent state change")
    axes[2].set_title("Discrete node-to-node motion")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Smoothness effect of global configuration optimisation")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_tracking(
    output: Path,
    inverse_s: np.ndarray,
    inverse_pos: np.ndarray,
    inverse_tan: np.ndarray,
    global_s: np.ndarray,
    global_pos: np.ndarray,
    global_tan: np.ndarray,
    pos_tol: float,
    tan_tol: float,
    dense_rows: list[dict[str, str]],
    inverse_margin: np.ndarray,
    global_margin_approx: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)

    axes[0].plot(inverse_s, 1e3 * inverse_pos, "--", label="inverse")
    axes[0].plot(global_s, 1e3 * global_pos, label="global")
    axes[0].axhline(1e3 * pos_tol, linestyle=":", label="position tolerance")
    axes[0].set_ylabel("position error [mm]")
    axes[0].set_title("Tip position tracking")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(inverse_s, np.degrees(inverse_tan), "--", label="inverse")
    axes[1].plot(global_s, np.degrees(global_tan), label="global")
    axes[1].axhline(np.degrees(tan_tol), linestyle=":", label="tangent tolerance")
    if dense_rows:
        dense_s = _column(dense_rows, "s_m")
        dense_tan = _column(dense_rows, "tangent_error_rad")
        axes[1].scatter(dense_s, np.degrees(dense_tan), s=8, alpha=0.45, label="dense global samples")
    axes[1].set_ylabel("tangent error [deg]")
    axes[1].set_title("Tip tangent tracking")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    if np.any(np.isfinite(inverse_margin)):
        axes[2].plot(inverse_s, 1e3 * inverse_margin, "--", label="inverse exact saved margin")
    if np.any(np.isfinite(global_margin_approx)):
        axes[2].plot(global_s, 1e3 * global_margin_approx, label="global approximate output-only margin")
    axes[2].axhline(0.0, linestyle=":")
    axes[2].set_xlabel("lumen arc length s [m]")
    axes[2].set_ylabel("source-magnet exclusion margin [mm]")
    axes[2].set_title("Head-exclusion check (global value is reconstructed/approximate)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Tracking and constraint preservation")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_tip_path(
    output: Path,
    inverse_desired: np.ndarray,
    inverse_tip: np.ndarray,
    global_desired: np.ndarray,
    global_tip: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*inverse_desired.T, "--", label="desired centreline")
    ax.plot(*inverse_tip.T, label="inverse achieved")
    ax.plot(*global_tip.T, label="global achieved")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Beam-tip geometric path")
    ax.legend()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def _plot_change_heatmap(
    output: Path,
    s: np.ndarray,
    seed: np.ndarray,
    optimized: np.ndarray,
    scale: np.ndarray,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized_change = np.abs((optimized - seed) / scale[None, :]).T
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    image = ax.imshow(
        normalized_change,
        aspect="auto",
        origin="lower",
        extent=[float(s[0]), float(s[-1]), -0.5, 6.5],
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(7), COORD_NAMES)
    ax.set_xlabel("lumen arc length s [m]")
    ax.set_ylabel("configuration coordinate")
    ax.set_title("Absolute global change from seed, normalized by configuration_scale")
    fig.colorbar(image, ax=ax, label="|change| / scale")
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = _args()
    inverse_dir = args.inverse_dir.expanduser().resolve()
    global_dir = args.global_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else global_dir / "comparison_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inverse_csv = inverse_dir / "inverse_configuration_path.csv"
    inverse_json = inverse_dir / "inverse_configuration_summary.json"
    global_csv = global_dir / "global_configuration_path.csv"
    global_json = global_dir / "global_configuration_summary.json"
    dense_csv = global_dir / "global_configuration_dense_validation.csv"

    inverse_rows = _read_csv(inverse_csv)
    global_rows = _read_csv(global_csv)
    dense_rows = _read_csv(dense_csv) if dense_csv.exists() else []
    inverse_payload = _read_json(inverse_json)
    global_payload = _read_json(global_json)

    if len(inverse_rows) < 2:
        raise ValueError("Need at least two inverse path nodes for path analysis")
    if len(global_rows) < 2:
        raise ValueError("Need at least two global path nodes for path analysis")

    inverse_config = dict(inverse_payload.get("configuration", {}))
    global_config = dict(global_payload.get("configuration", {}))
    inverse_summary = dict(inverse_payload.get("summary", {}))
    global_summary = dict(global_payload.get("summary", {}))

    pos_tol = _finite_or_none(global_config.get("position_tolerance_m"))
    tan_tol = _finite_or_none(global_config.get("tangent_tolerance_rad"))
    if pos_tol is None:
        pos_tol = _finite_or_none(inverse_config.get("position_tolerance_m"))
    if tan_tol is None:
        tan_tol = _finite_or_none(inverse_config.get("tangent_tolerance_rad"))
    if pos_tol is None or tan_tol is None or pos_tol <= 0 or tan_tol <= 0:
        raise ValueError("Could not recover positive position/tangent tolerances from JSON")

    scale = np.asarray(
        global_config.get("configuration_scale", [1, 1, 1, 1, 1, 1, 1e-2]),
        dtype=float,
    ).reshape(7)
    first_weight = np.asarray(
        global_config.get("first_difference_weight", [1e-5] * 7), dtype=float
    ).reshape(7)
    second_weight = np.asarray(
        global_config.get("second_difference_weight", [1e-8] * 7), dtype=float
    ).reshape(7)

    joint_limits = np.asarray(
        global_config.get("maximum_adjacent_joint_change_rad", [np.nan] * 6),
        dtype=float,
    ).reshape(6)
    insertion_limit = _finite_or_none(
        global_config.get("maximum_adjacent_insertion_change_m")
    )
    adjacent_limits = np.concatenate(
        (joint_limits, [np.nan if insertion_limit is None else insertion_limit])
    )

    inverse_s = _column(inverse_rows, "s_m")
    global_s = _column(global_rows, "s_m")
    inverse_state = _state_matrix(inverse_rows)
    global_state = _state_matrix(global_rows)
    global_seed = _state_matrix(global_rows, prefix="seed")
    if not np.all(np.isfinite(global_seed)):
        raise ValueError(
            "Global CSV is missing finite seed_q1..seed_q6/seed_L columns. "
            "These are required for a direct before/after global-layer comparison."
        )

    before_smooth = _smoothness_metrics(
        global_s, global_seed, scale, first_weight, second_weight
    )
    after_smooth = _smoothness_metrics(
        global_s, global_state, scale, first_weight, second_weight
    )

    inverse_pos_error = _column(inverse_rows, "position_error_m")
    inverse_tan_error = _column(inverse_rows, "tangent_error_rad")
    global_pos_error = _column(global_rows, "position_error_m")
    global_tan_error = _column(global_rows, "tangent_error_rad")
    inverse_tracking = _tracking_metrics(
        inverse_pos_error, inverse_tan_error, pos_tol, tan_tol
    )
    global_tracking = _tracking_metrics(
        global_pos_error, global_tan_error, pos_tol, tan_tol
    )

    inverse_desired = _xyz(inverse_rows, "desired")
    inverse_tip = _xyz(inverse_rows, "tip")
    global_desired = _xyz(global_rows, "desired")
    global_tip = _xyz(global_rows, "achieved")

    # Verify that the global seed corresponds to the inverse path wherever the
    # path coordinates coincide. This is especially useful in refine_complete.
    matched_pairs = _match_s_values(global_s, inverse_s)
    seed_inverse_differences = []
    for g_i, i_i in matched_pairs:
        seed_inverse_differences.append((global_seed[g_i] - inverse_state[i_i]) / scale)
    seed_inverse_differences_arr = (
        np.asarray(seed_inverse_differences, dtype=float)
        if seed_inverse_differences
        else np.empty((0, 7), dtype=float)
    )
    seed_match_max = (
        float(np.max(np.abs(seed_inverse_differences_arr)))
        if seed_inverse_differences_arr.size
        else np.nan
    )
    seed_match_rms = _rms(seed_inverse_differences_arr)

    # Direct state-change statistics on global grid.
    normalized_change = (global_state - global_seed) / scale[None, :]
    change_norm = np.linalg.norm(normalized_change, axis=1)
    seed_deviation_weight = np.asarray(
        global_config.get("seed_deviation_weight", [0.0] * 7), dtype=float
    ).reshape(7)
    nominal_seed_deviation_cost = 0.5 * float(
        np.sum(normalized_change * normalized_change * seed_deviation_weight[None, :])
    )

    # Dense nonlinear validation metrics, if present.
    dense_metrics: dict[str, Any] = {
        "available": bool(dense_rows),
        "number_samples": len(dense_rows),
    }
    if dense_rows:
        dense_pos = _column(dense_rows, "position_error_m")
        dense_tan = _column(dense_rows, "tangent_error_rad")
        dense_score = np.maximum(dense_pos / pos_tol, dense_tan / tan_tol)
        dense_feasible_col = _column(dense_rows, "feasible")
        dense_metrics.update(
            {
                "maximum_normalized_task_score": float(np.nanmax(dense_score)),
                "rms_normalized_task_score": _rms(dense_score),
                "minimum_position_margin_m": float(
                    np.nanmin(_column(dense_rows, "position_margin_m"))
                ),
                "minimum_tangent_margin_rad": float(
                    np.nanmin(_column(dense_rows, "tangent_margin_rad"))
                ),
                "feasible_fraction": float(np.nanmean(dense_feasible_col > 0.5)),
            }
        )

    # Head-exclusion analysis. The inverse output contains authoritative saved
    # distances. The global output does not contain an equivalent field, so we
    # reconstruct a centreline from saved desired points and use magnet XYZ to
    # make an output-only approximation.
    inverse_exclusion_radius = _column(
        inverse_rows, "source_magnet_lumen_exclusion_radius_m"
    )
    finite_radius = inverse_exclusion_radius[np.isfinite(inverse_exclusion_radius)]
    exclusion_radius = float(np.nanmedian(finite_radius)) if finite_radius.size else np.nan
    inverse_saved_distance = _column(inverse_rows, "source_magnet_lumen_distance_m")
    inverse_saved_margin = _column(inverse_rows, "source_magnet_lumen_margin_m")

    reconstructed_polyline = _reconstructed_centreline(
        inverse_s, inverse_desired, global_s, global_desired
    )
    inverse_magnet = _xyz(inverse_rows, "magnet_pose")
    global_magnet = _xyz(global_rows, "magnet_pose")
    inverse_reconstructed_distance = _nearest_polyline_distance(
        inverse_magnet, reconstructed_polyline
    )
    global_reconstructed_distance = _nearest_polyline_distance(
        global_magnet, reconstructed_polyline
    )
    inverse_reconstruction_error = inverse_reconstructed_distance - inverse_saved_distance
    if np.isfinite(exclusion_radius):
        global_approx_margin = global_reconstructed_distance - exclusion_radius
    else:
        global_approx_margin = np.full(global_s.shape, np.nan)

    exclusion_metrics: dict[str, Any] = {
        "inverse_constraint_radius_m": (
            exclusion_radius if np.isfinite(exclusion_radius) else None
        ),
        "inverse_saved_minimum_distance_m": (
            float(np.nanmin(inverse_saved_distance))
            if np.any(np.isfinite(inverse_saved_distance))
            else None
        ),
        "inverse_saved_minimum_margin_m": (
            float(np.nanmin(inverse_saved_margin))
            if np.any(np.isfinite(inverse_saved_margin))
            else None
        ),
        "reconstructed_centreline_point_count": int(reconstructed_polyline.shape[0]),
        "inverse_reconstruction_distance_error_max_abs_m": (
            float(np.nanmax(np.abs(inverse_reconstruction_error)))
            if np.any(np.isfinite(inverse_reconstruction_error))
            else None
        ),
        "inverse_reconstruction_distance_error_rms_m": _rms(
            inverse_reconstruction_error
        ),
        "global_output_only_approx_minimum_distance_m": (
            float(np.nanmin(global_reconstructed_distance))
            if np.any(np.isfinite(global_reconstructed_distance))
            else None
        ),
        "global_output_only_approx_minimum_margin_m": (
            float(np.nanmin(global_approx_margin))
            if np.any(np.isfinite(global_approx_margin))
            else None
        ),
        "global_output_only_approx_constraint_satisfied": (
            bool(np.nanmin(global_approx_margin) >= 0.0)
            if np.any(np.isfinite(global_approx_margin))
            else None
        ),
        "authoritative_for_global": False,
    }

    # Summary rows: lower is better for all quantities in this table.
    summary_rows = [
        {
            "metric": "scaled configuration path length",
            "before": before_smooth["scaled_path_length"],
            "after": after_smooth["scaled_path_length"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["scaled_path_length"], after_smooth["scaled_path_length"]
            ),
            "unit": "dimensionless",
            "interpretation": "overall movement through scaled configuration space",
        },
        {
            "metric": "max scaled first-derivative norm",
            "before": before_smooth["first_norm_max"],
            "after": after_smooth["first_norm_max"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["first_norm_max"], after_smooth["first_norm_max"]
            ),
            "unit": "1/m",
            "interpretation": "peak q'(s) demand proxy for time parameterisation",
        },
        {
            "metric": "RMS scaled first-derivative norm",
            "before": before_smooth["first_norm_rms"],
            "after": after_smooth["first_norm_rms"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["first_norm_rms"], after_smooth["first_norm_rms"]
            ),
            "unit": "1/m",
            "interpretation": "typical q'(s) demand",
        },
        {
            "metric": "max scaled second-derivative norm",
            "before": before_smooth["second_norm_max"],
            "after": after_smooth["second_norm_max"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["second_norm_max"], after_smooth["second_norm_max"]
            ),
            "unit": "1/m^2",
            "interpretation": "peak q''(s) demand proxy; often most relevant smoothing gain",
        },
        {
            "metric": "RMS scaled second-derivative norm",
            "before": before_smooth["second_norm_rms"],
            "after": after_smooth["second_norm_rms"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["second_norm_rms"], after_smooth["second_norm_rms"]
            ),
            "unit": "1/m^2",
            "interpretation": "typical q''(s) demand",
        },
        {
            "metric": "weighted first-difference objective term",
            "before": before_smooth["weighted_first_difference_cost"],
            "after": after_smooth["weighted_first_difference_cost"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["weighted_first_difference_cost"],
                after_smooth["weighted_first_difference_cost"],
            ),
            "unit": "objective units",
            "interpretation": "exact first-difference smoothing term used by global objective",
        },
        {
            "metric": "weighted second-difference objective term",
            "before": before_smooth["weighted_second_difference_cost"],
            "after": after_smooth["weighted_second_difference_cost"],
            "improvement_percent": _safe_percent_improvement(
                before_smooth["weighted_second_difference_cost"],
                after_smooth["weighted_second_difference_cost"],
            ),
            "unit": "objective units",
            "interpretation": "exact second-difference smoothing term used by global objective",
        },
        {
            "metric": "maximum tip position error",
            "before": inverse_tracking["position_error_max_m"],
            "after": global_tracking["position_error_max_m"],
            "improvement_percent": _safe_percent_improvement(
                inverse_tracking["position_error_max_m"],
                global_tracking["position_error_max_m"],
            ),
            "unit": "m",
            "interpretation": "tracking is a hard constraint; increase can be acceptable if within tolerance",
        },
        {
            "metric": "maximum tip tangent error",
            "before": inverse_tracking["tangent_error_max_rad"],
            "after": global_tracking["tangent_error_max_rad"],
            "improvement_percent": _safe_percent_improvement(
                inverse_tracking["tangent_error_max_rad"],
                global_tracking["tangent_error_max_rad"],
            ),
            "unit": "rad",
            "interpretation": "tracking is a hard constraint; increase can be acceptable if within tolerance",
        },
    ]

    _write_summary_csv(output_dir / "summary_metrics.csv", summary_rows)
    _write_coordinate_csv(
        output_dir / "per_coordinate_metrics.csv",
        before_smooth,
        after_smooth,
        adjacent_limits,
    )

    metrics = {
        "inputs": {
            "inverse_dir": str(inverse_dir),
            "global_dir": str(global_dir),
            "global_mode": global_config.get("mode"),
            "inverse_nodes": len(inverse_rows),
            "global_nodes": len(global_rows),
            "dense_samples": len(dense_rows),
            "position_tolerance_m": pos_tol,
            "tangent_tolerance_rad": tan_tol,
            "configuration_scale": scale,
        },
        "global_status": {
            "globally_feasible": global_summary.get("globally_feasible"),
            "dense_validation_feasible": global_summary.get("dense_validation_feasible"),
            "optimizer_success": global_summary.get("optimizer_success"),
            "termination_reason": global_summary.get("termination_reason"),
            "optimizer_iterations": global_summary.get("optimizer_iterations"),
            "normalized_constraint_violation": global_summary.get("normalized_constraint_violation"),
        },
        "seed_consistency": {
            "matched_nodes": len(matched_pairs),
            "global_nodes": len(global_rows),
            "max_abs_scaled_seed_minus_inverse": seed_match_max,
            "rms_scaled_seed_minus_inverse": seed_match_rms,
        },
        "global_change_from_seed": {
            "maximum_scaled_state_change_norm": float(np.max(change_norm)),
            "rms_scaled_state_change_norm": _rms(change_norm),
            "per_coordinate_max_abs_change": np.max(np.abs(global_state - global_seed), axis=0),
            "nominal_seed_deviation_objective_term": nominal_seed_deviation_cost,
        },
        "smoothness_before_seed": {
            key: value
            for key, value in before_smooth.items()
            if key not in {"ds", "first", "second", "delta", "first_norm", "second_norm"}
        },
        "smoothness_after_global": {
            key: value
            for key, value in after_smooth.items()
            if key not in {"ds", "first", "second", "delta", "first_norm", "second_norm"}
        },
        "tracking_inverse": inverse_tracking,
        "tracking_global": global_tracking,
        "dense_validation": dense_metrics,
        "head_exclusion": exclusion_metrics,
        "important_interpretation": {
            "tracking_error_note": (
                "Global optimisation is allowed to trade tracking error within the hard "
                "position/tangent tolerances in exchange for a smoother configuration path."
            ),
            "time_parameterisation_note": (
                "First/second derivatives with respect to s are TOPP-relevant proxies only. "
                "Actual time improvement requires running the same time parameteriser on both paths."
            ),
            "head_exclusion_note": (
                "The global output does not contain an authoritative source-magnet/lumen "
                "clearance constraint diagnostic. The reported post-global clearance is reconstructed "
                "from saved magnet XYZ and desired path points and is therefore approximate."
            ),
        },
    }
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2), encoding="utf-8"
    )

    smooth_first_improvement = _safe_percent_improvement(
        before_smooth["first_norm_max"], after_smooth["first_norm_max"]
    )
    smooth_second_improvement = _safe_percent_improvement(
        before_smooth["second_norm_max"], after_smooth["second_norm_max"]
    )
    smoothing_cost_improvement = _safe_percent_improvement(
        before_smooth["weighted_smoothing_cost"],
        after_smooth["weighted_smoothing_cost"],
    )

    report_lines = [
        "# Global configuration optimisation analysis",
        "",
        "## What is being compared",
        "",
        f"- Inverse path nodes: **{len(inverse_rows)}**",
        f"- Global path nodes: **{len(global_rows)}**",
        f"- Global mode: **{global_config.get('mode', 'unknown')}**",
        f"- Dense validation samples: **{len(dense_rows)}**",
        f"- Position tolerance: **{1e3 * pos_tol:.4f} mm**",
        f"- Tangent tolerance: **{math.degrees(tan_tol):.4f} deg**",
        "- The primary before/after smoothness comparison is **global seed -> global optimized** on the same node grid.",
        "",
        "## Main result: geometric path quality",
        "",
        f"- Peak scaled first-derivative norm: `{before_smooth['first_norm_max']:.6g}` -> `{after_smooth['first_norm_max']:.6g}`  "
        f"(**{smooth_first_improvement:+.2f}% lower-is-better improvement**).",
        f"- Peak scaled second-derivative norm: `{before_smooth['second_norm_max']:.6g}` -> `{after_smooth['second_norm_max']:.6g}`  "
        f"(**{smooth_second_improvement:+.2f}% lower-is-better improvement**).",
        f"- Weighted first+second smoothing objective: `{before_smooth['weighted_smoothing_cost']:.6g}` -> `{after_smooth['weighted_smoothing_cost']:.6g}`  "
        f"(**{smoothing_cost_improvement:+.2f}% improvement**).",
        f"- Scaled configuration path length: `{before_smooth['scaled_path_length']:.6g}` -> `{after_smooth['scaled_path_length']:.6g}`.",
        "",
        "For time parameterisation, the second-derivative reduction is particularly important because "
        "`q_ddot = q''(s) s_dot^2 + q'(s) s_ddot`. A smoother geometric path can therefore permit a less restrictive "
        "speed profile, but this report does **not** claim a traversal-time improvement until the same time parameteriser is run on both paths.",
        "",
        "## Tracking preservation",
        "",
        f"- Inverse max position error: **{1e3 * inverse_tracking['position_error_max_m']:.4f} mm**",
        f"- Global max position error: **{1e3 * global_tracking['position_error_max_m']:.4f} mm**",
        f"- Inverse max tangent error: **{math.degrees(inverse_tracking['tangent_error_max_rad']):.4f} deg**",
        f"- Global max tangent error: **{math.degrees(global_tracking['tangent_error_max_rad']):.4f} deg**",
        f"- Global node error-based feasible fraction: **{100 * global_tracking['feasible_fraction_by_errors']:.2f}%**",
        "",
        "Tracking error does not need to decrease for the global layer to be beneficial. Its purpose is to improve the "
        "configuration path while keeping the task inside the hard position/tangent tolerances.",
        "",
        "## Global-solver status",
        "",
        f"- Optimizer success: **{global_summary.get('optimizer_success')}**",
        f"- Globally feasible: **{global_summary.get('globally_feasible')}**",
        f"- Dense validation feasible: **{global_summary.get('dense_validation_feasible')}**",
        f"- Termination: `{global_summary.get('termination_reason')}`",
        f"- Reported normalized hard-constraint violation: `{global_summary.get('normalized_constraint_violation')}`",
        "",
        "## Seed consistency",
        "",
        f"- Matching inverse/global-seed coordinates: **{len(matched_pairs)} / {len(global_rows)}** global nodes",
        f"- Maximum scaled seed-vs-inverse mismatch on matching coordinates: **{_format_metric(seed_match_max, digits=3)}**",
        "",
        "A near-zero mismatch confirms that the plotted 'before' path is the inverse path actually consumed by the global layer. "
        "In recovery/refinement cases with added nodes, not every global coordinate necessarily has an exact inverse counterpart.",
        "",
        "## Head-exclusion warning",
        "",
        "The saved inverse output includes source-magnet/lumen exclusion distances and margins. The global output does not "
        "contain an equivalent authoritative constraint field. This analysis therefore reconstructs a polyline from saved desired "
        "positions and computes an **approximate output-only** global magnet-to-centreline distance.",
    ]

    if np.isfinite(exclusion_radius):
        report_lines.extend(
            [
                f"- Exclusion radius from inverse output: **{1e3 * exclusion_radius:.3f} mm**",
                f"- Inverse saved minimum margin: **{_format_metric(float(np.nanmin(inverse_saved_margin)), 1e3, 3)} mm**",
                f"- Approximate global minimum margin: **{_format_metric(float(np.nanmin(global_approx_margin)), 1e3, 3)} mm**",
                f"- Reconstruction self-check on inverse distances, max absolute error: **{_format_metric(float(np.nanmax(np.abs(inverse_reconstruction_error))), 1e3, 3)} mm**",
                "",
                "Treat the approximate global margin as a screening diagnostic only. If exclusion must remain a hard physical "
                "constraint after smoothing, the authoritative solution is to enforce/re-evaluate that constraint inside the global layer using the original lumen geometry.",
            ]
        )
    else:
        report_lines.extend(
            [
                "- No finite exclusion radius was found in the inverse CSV, so no post-global clearance margin could be estimated.",
            ]
        )

    if dense_rows:
        report_lines.extend(
            [
                "",
                "## Dense between-node validation",
                "",
                f"- Samples: **{dense_metrics['number_samples']}**",
                f"- Maximum normalized task score: **{dense_metrics['maximum_normalized_task_score']:.5f}** (<= 1 is within both tolerances)",
                f"- Dense feasible fraction: **{100 * dense_metrics['feasible_fraction']:.2f}%**",
                f"- Minimum position margin: **{1e3 * dense_metrics['minimum_position_margin_m']:.4f} mm**",
                f"- Minimum tangent margin: **{math.degrees(dense_metrics['minimum_tangent_margin_rad']):.4f} deg**",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary_metrics.csv`: headline before/after metrics and percentage changes.",
            "- `per_coordinate_metrics.csv`: q1..q6 and insertion step/derivative metrics.",
            "- `analysis_metrics.json`: full machine-readable results.",
            "- `configuration_before_after.png`: direct seed/global configuration comparison.",
            "- `smoothness_before_after.png`: first/second derivative and adjacent-step comparison.",
            "- `tracking_constraints.png`: position/tangent errors plus head-exclusion diagnostic.",
            "- `tip_path_comparison.png`: desired, inverse achieved, and global achieved tip paths.",
            "- `global_change_from_seed.png`: where the global optimiser moved each configuration coordinate.",
        ]
    )

    (output_dir / "analysis_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    if not args.no_plots:
        _plot_configuration(
            output_dir / "configuration_before_after.png",
            global_s,
            global_seed,
            global_state,
            args.dpi,
        )
        _plot_smoothness(
            output_dir / "smoothness_before_after.png",
            global_s,
            before_smooth,
            after_smooth,
            scale,
            args.dpi,
        )
        _plot_tracking(
            output_dir / "tracking_constraints.png",
            inverse_s,
            inverse_pos_error,
            inverse_tan_error,
            global_s,
            global_pos_error,
            global_tan_error,
            pos_tol,
            tan_tol,
            dense_rows,
            inverse_saved_margin,
            global_approx_margin,
            args.dpi,
        )
        _plot_tip_path(
            output_dir / "tip_path_comparison.png",
            inverse_desired,
            inverse_tip,
            global_desired,
            global_tip,
            args.dpi,
        )
        _plot_change_heatmap(
            output_dir / "global_change_from_seed.png",
            global_s,
            global_seed,
            global_state,
            scale,
            args.dpi,
        )

    print("Global configuration analysis complete")
    print(f"  report: {output_dir / 'analysis_report.md'}")
    print(f"  metrics: {output_dir / 'analysis_metrics.json'}")
    print(f"  summary: {output_dir / 'summary_metrics.csv'}")
    print(f"  per-coordinate: {output_dir / 'per_coordinate_metrics.csv'}")
    if not args.no_plots:
        print(f"  plots: {output_dir}")


if __name__ == "__main__":
    main()
