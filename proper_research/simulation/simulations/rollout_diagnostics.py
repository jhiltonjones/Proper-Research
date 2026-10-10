from __future__ import annotations

"""Reference-spacing and rollout-diagnostics utilities.

This module is designed for the magnetic continuum robot controller code used
in the accompanying conversation. It provides:

1. Arc-length-based reference diagnostics and selection.
2. Long-format per-stage logging for each feedback frame.
3. Post-processing summaries and plots comparing:
   - affine/linear controller prediction,
   - nonlinear controller-model rollout,
   - applied plant rollout,
   - centreline reference.

Expected controller ``info`` entries
------------------------------------
Required for full diagnostics:
    idx_ref, U_seq, X_pred, X_nl_plan, x_rollout, U_applied, x_start

The functions degrade gracefully when some arrays are unavailable.
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Any

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class ReferenceSpacingReport:
    number_of_points: int
    total_arclength_m: float
    mean_spacing_m: float
    median_spacing_m: float
    min_spacing_m: float
    max_spacing_m: float
    lookahead_m: float
    stage_spacing_m: float
    horizon_span_m: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "number_of_points": self.number_of_points,
            "total_arclength_m": self.total_arclength_m,
            "mean_spacing_m": self.mean_spacing_m,
            "median_spacing_m": self.median_spacing_m,
            "min_spacing_m": self.min_spacing_m,
            "max_spacing_m": self.max_spacing_m,
            "lookahead_m": self.lookahead_m,
            "stage_spacing_m": self.stage_spacing_m,
            "horizon_span_m": self.horizon_span_m,
        }


def _as_xyz(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected shape (M, >=3), got {points.shape}.")
    if points.shape[0] < 1:
        raise ValueError("Point array is empty.")
    return points[:, :3].copy()


def cumulative_arclength(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative arc length ``s`` and adjacent segment lengths ``ds``."""
    xyz = _as_xyz(points)
    if xyz.shape[0] == 1:
        return np.zeros(1, dtype=float), np.zeros(0, dtype=float)
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s, ds


def reference_spacing_report(
    lumen_C: np.ndarray,
    *,
    ref_stride_pts: int,
    ref_lookahead_pts: int,
    Np: int,
) -> ReferenceSpacingReport:
    """Quantify the physical spacing implied by point-index settings."""
    if ref_stride_pts <= 0:
        raise ValueError("ref_stride_pts must be positive.")
    if ref_lookahead_pts < 0:
        raise ValueError("ref_lookahead_pts must be non-negative.")
    if Np <= 0:
        raise ValueError("Np must be positive.")

    xyz = _as_xyz(lumen_C)
    s, ds = cumulative_arclength(xyz)

    median_ds = float(np.median(ds)) if ds.size else 0.0
    return ReferenceSpacingReport(
        number_of_points=int(xyz.shape[0]),
        total_arclength_m=float(s[-1]),
        mean_spacing_m=float(np.mean(ds)) if ds.size else 0.0,
        median_spacing_m=median_ds,
        min_spacing_m=float(np.min(ds)) if ds.size else 0.0,
        max_spacing_m=float(np.max(ds)) if ds.size else 0.0,
        lookahead_m=float(ref_lookahead_pts * median_ds),
        stage_spacing_m=float(ref_stride_pts * median_ds),
        horizon_span_m=float(max(Np - 1, 0) * ref_stride_pts * median_ds),
    )


def print_reference_spacing_report(report: ReferenceSpacingReport) -> None:
    """Print a compact millimetre-based reference-spacing report."""
    d = report.to_dict()
    print("[REFERENCE SPACING]")
    print(f"  points              : {d['number_of_points']}")
    print(f"  total arclength     : {1e3*d['total_arclength_m']:.3f} mm")
    print(f"  adjacent mean       : {1e3*d['mean_spacing_m']:.3f} mm")
    print(f"  adjacent median     : {1e3*d['median_spacing_m']:.3f} mm")
    print(f"  adjacent min/max    : {1e3*d['min_spacing_m']:.3f} / {1e3*d['max_spacing_m']:.3f} mm")
    print(f"  effective lookahead : {1e3*d['lookahead_m']:.3f} mm")
    print(f"  stage spacing       : {1e3*d['stage_spacing_m']:.3f} mm")
    print(f"  horizon span        : {1e3*d['horizon_span_m']:.3f} mm")


def _nearest_s_indices(s: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return centreline indices nearest to specified arc-length targets."""
    s = np.asarray(s, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if s.size == 0:
        raise ValueError("s is empty.")

    right = np.searchsorted(s, targets, side="left")
    right = np.clip(right, 0, s.size - 1)
    left = np.clip(right - 1, 0, s.size - 1)
    use_left = np.abs(targets - s[left]) <= np.abs(s[right] - targets)
    return np.where(use_left, left, right).astype(int)


def select_reference_indices_metric(
    *,
    lumen_C: np.ndarray,
    x_now: np.ndarray,
    Np: int,
    lookahead_m: float,
    stride_m: float,
    use_xy: bool = True,
    allow_backward: bool = False,
    i_ref_last: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select horizon references using physical arc-length spacing.

    The nearest centreline sample is still found geometrically, but lookahead
    and horizon spacing are specified in metres rather than raw sample counts.
    """
    if Np <= 0:
        raise ValueError("Np must be positive.")
    if lookahead_m < 0.0:
        raise ValueError("lookahead_m must be non-negative.")
    if stride_m <= 0.0:
        raise ValueError("stride_m must be positive.")

    C = _as_xyz(lumen_C)
    x_now = np.asarray(x_now, dtype=float).reshape(-1)
    if x_now.size < 3:
        raise ValueError("x_now must have at least three entries.")

    tip = x_now[:3]
    if use_xy:
        d_all = np.linalg.norm(C[:, :2] - tip[:2], axis=1)
    else:
        d_all = np.linalg.norm(C - tip, axis=1)

    i_closest = int(np.argmin(d_all))
    s, _ = cumulative_arclength(C)

    start_target_s = min(float(s[-1]), float(s[i_closest] + lookahead_m))
    i_ref = int(_nearest_s_indices(s, np.array([start_target_s]))[0])

    if not allow_backward:
        i_ref = max(i_ref, int(i_ref_last))

    i_ref = int(np.clip(i_ref, 0, C.shape[0] - 1))
    target_s = np.clip(s[i_ref] + stride_m * np.arange(Np), 0.0, s[-1])
    idx_ref = _nearest_s_indices(s, target_s)

    if not allow_backward:
        idx_ref = np.maximum.accumulate(idx_ref)
        idx_ref = np.maximum(idx_ref, i_ref)

    actual_s = s[idx_ref]
    diagnostics = {
        "i_closest": i_closest,
        "i_ref": i_ref,
        "idx_ref": idx_ref.copy(),
        "target_s_m": target_s.copy(),
        "actual_s_m": actual_s.copy(),
        "actual_stage_spacing_m": np.diff(actual_s),
        "horizon_span_m": float(actual_s[-1] - actual_s[0]) if actual_s.size else 0.0,
        "dist_to_closest_m": float(d_all[i_closest]),
        "dist_to_first_ref_m": float(d_all[i_ref]),
    }
    return idx_ref.astype(int), diagnostics


def _array(info: dict[str, Any], key: str, *, cols: int | None = None) -> np.ndarray:
    value = info.get(key, None)
    if value is None:
        return np.empty((0, 0 if cols is None else cols), dtype=float)
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.empty((0, 0 if cols is None else cols), dtype=float)
    if arr.ndim == 1:
        if cols is not None and arr.size % cols == 0:
            arr = arr.reshape(-1, cols)
        else:
            arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.empty((0, 0 if cols is None else cols), dtype=float)
    return arr


def _vec_or_nan(arr: np.ndarray, i: int, n: int = 3) -> np.ndarray:
    if arr.ndim == 2 and 0 <= i < arr.shape[0]:
        out = np.full(n, np.nan, dtype=float)
        m = min(n, arr.shape[1])
        out[:m] = arr[i, :m]
        return out
    return np.full(n, np.nan, dtype=float)


def _norm_xy(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a[:2])) or not np.all(np.isfinite(b[:2])):
        return np.nan
    return float(np.linalg.norm(a[:2] - b[:2]))


def _norm_xyz(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a[:3])) or not np.all(np.isfinite(b[:3])):
        return np.nan
    return float(np.linalg.norm(a[:3] - b[:3]))


def _gain_and_angle(pred_delta: np.ndarray, actual_delta: np.ndarray) -> tuple[float, float]:
    if not np.all(np.isfinite(pred_delta[:3])) or not np.all(np.isfinite(actual_delta[:3])):
        return np.nan, np.nan
    denom = float(pred_delta[:3] @ pred_delta[:3])
    if denom <= EPS:
        return np.nan, np.nan
    gain = float((actual_delta[:3] @ pred_delta[:3]) / denom)
    npred = float(np.linalg.norm(pred_delta[:3]))
    nact = float(np.linalg.norm(actual_delta[:3]))
    if npred <= EPS or nact <= EPS:
        angle = np.nan
    else:
        c = float(np.clip((pred_delta[:3] @ actual_delta[:3]) / (npred * nact), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(c)))
    return gain, angle


def build_rollout_stage_rows(
    *,
    k: int,
    info: dict[str, Any],
    lumen_C: np.ndarray,
    controller_name: str = "",
) -> list[dict[str, Any]]:
    """Build one long-format row per planned horizon stage.

    Error definitions
    -----------------
    linearisation error:
        X_nl_plan - X_pred
    execution prediction error:
        x_rollout - X_pred
    plant-versus-nonlinear-model error:
        x_rollout - X_nl_plan
    tracking error:
        state - X_ref
    """
    C = _as_xyz(lumen_C)
    s, _ = cumulative_arclength(C)

    idx_ref = np.asarray(info.get("idx_ref", []), dtype=int).reshape(-1)
    U_seq = _array(info, "U_seq", cols=7)
    U_applied = _array(info, "U_applied", cols=7)
    X_pred = _array(info, "X_pred")
    X_nl = _array(info, "X_nl_plan")
    X_roll = _array(info, "x_rollout")

    x_start = np.asarray(info.get("x_start", np.full(3, np.nan)), dtype=float).reshape(-1)
    if x_start.size < 3:
        x_start = np.pad(x_start, (0, 3 - x_start.size), constant_values=np.nan)

    N = max(idx_ref.size, U_seq.shape[0], X_pred.shape[0], X_nl.shape[0], X_roll.shape[0])
    rows: list[dict[str, Any]] = []

    previous_pred = x_start[:3].copy()
    previous_actual = x_start[:3].copy()

    for stage in range(N):
        ref_idx = int(idx_ref[stage]) if stage < idx_ref.size else -1
        ref = C[ref_idx] if 0 <= ref_idx < C.shape[0] else np.full(3, np.nan)
        ref_s = float(s[ref_idx]) if 0 <= ref_idx < s.size else np.nan

        pred = _vec_or_nan(X_pred, stage, 3)
        nl = _vec_or_nan(X_nl, stage, 3)
        actual = _vec_or_nan(X_roll, stage, 3)
        u_plan = _vec_or_nan(U_seq, stage, 7)
        u_applied = _vec_or_nan(U_applied, stage, 7)

        pred_delta = pred - previous_pred if np.all(np.isfinite(previous_pred)) else np.full(3, np.nan)
        actual_delta = actual - previous_actual if np.all(np.isfinite(previous_actual)) else np.full(3, np.nan)
        gain, direction_error_deg = _gain_and_angle(pred_delta, actual_delta)

        row = {
            "k": int(k),
            "stage": int(stage),
            "controller": str(controller_name or info.get("controller_type", info.get("solver_mode", ""))),
            "solver_mode": str(info.get("solver_mode_used", info.get("solver_mode", ""))),
            "inverse_sequence_mode": str(info.get("inverse_sequence_mode", "")),
            "rollout_steps_used": int(info.get("rollout_steps_used", info.get("rollout_steps", 0))),
            "was_applied": int(stage < U_applied.shape[0] and stage < X_roll.shape[0]),
            "ref_idx": ref_idx,
            "ref_s_mm": 1e3 * ref_s if np.isfinite(ref_s) else np.nan,
            "ref_x_mm": 1e3 * float(ref[0]),
            "ref_y_mm": 1e3 * float(ref[1]),
            "ref_z_mm": 1e3 * float(ref[2]),
            "pred_x_mm": 1e3 * float(pred[0]),
            "pred_y_mm": 1e3 * float(pred[1]),
            "pred_z_mm": 1e3 * float(pred[2]),
            "nl_x_mm": 1e3 * float(nl[0]),
            "nl_y_mm": 1e3 * float(nl[1]),
            "nl_z_mm": 1e3 * float(nl[2]),
            "actual_x_mm": 1e3 * float(actual[0]),
            "actual_y_mm": 1e3 * float(actual[1]),
            "actual_z_mm": 1e3 * float(actual[2]),
            "linearisation_err_xy_mm": 1e3 * _norm_xy(nl, pred),
            "linearisation_err_xyz_mm": 1e3 * _norm_xyz(nl, pred),
            "execution_err_xy_mm": 1e3 * _norm_xy(actual, pred),
            "execution_err_xyz_mm": 1e3 * _norm_xyz(actual, pred),
            "plant_vs_nl_err_xy_mm": 1e3 * _norm_xy(actual, nl),
            "plant_vs_nl_err_xyz_mm": 1e3 * _norm_xyz(actual, nl),
            "tracking_pred_xy_mm": 1e3 * _norm_xy(pred, ref),
            "tracking_nl_xy_mm": 1e3 * _norm_xy(nl, ref),
            "tracking_actual_xy_mm": 1e3 * _norm_xy(actual, ref),
            "pred_step_mm": 1e3 * float(np.linalg.norm(pred_delta)) if np.all(np.isfinite(pred_delta)) else np.nan,
            "actual_step_mm": 1e3 * float(np.linalg.norm(actual_delta)) if np.all(np.isfinite(actual_delta)) else np.nan,
            "local_gain": gain,
            "direction_error_deg": direction_error_deg,
        }

        channel_names = ("vx", "vy", "vz", "wx", "wy", "wz", "dL")
        for j, name in enumerate(channel_names):
            row[f"u_plan_{name}"] = float(u_plan[j])
            row[f"u_applied_{name}"] = float(u_applied[j])

        rows.append(row)

        if np.all(np.isfinite(pred)):
            previous_pred = pred.copy()
        if np.all(np.isfinite(actual)):
            previous_actual = actual.copy()

    return rows


def append_rollout_stage_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    """Append long-format rollout rows to CSV."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def save_rollout_npz(
    output_path,
    *,
    k,
    info,
    lumen_C=None,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "k": np.asarray([int(k)]),
    }

    array_keys = (
        # State and controls.
        "p_start",
        "x_start",
        "U_seq",
        "U_applied",
        "u_applied",

        # References and predictions.
        "idx_ref",
        "X_ref",
        "X_ref_plan",
        "reference_s_plan_m",
        "X_pred",
        "X_lin_candidate",
        "X_nl_candidate",
        "X_nl_plan",
        "x_rollout",

        # Jacobian used when building the controller model.
        "B_sequence",
        "B_sequence_linearisation",
        "B_pose_nodes_linearisation",

        # Jacobian evaluated along the selected solution.
        "B_sequence_solution",
        "B_pose_nodes_solution",
        "B_drift_relative_stage",

        # Characteristic command scales.
        "jacobian_diag_input_scale",

        # Optional precomputed metrics.
        "jac_solution_scaled_authority",
        "jac_solution_weighted_scaled_authority",
        "jac_solution_contribution_vectors",
        "jac_solution_contribution_norm",
        "jac_solution_predicted_increment",
        "jac_solution_matrix_change_relative",
        "jac_solution_column_gain_ratio",
        "jac_solution_column_direction_change_deg",

        "jac_linearisation_scaled_authority",
        "jac_linearisation_weighted_scaled_authority",
        "jac_linearisation_contribution_vectors",
        "jac_linearisation_contribution_norm",
    )

    for key in array_keys:
        value = info.get(key, None)

        if value is None:
            continue

        try:
            array = np.asarray(value)
        except Exception:
            continue

        if array.dtype == object:
            continue

        payload[key] = array.copy()

    if lumen_C is not None:
        payload["lumen_C"] = np.asarray(
            lumen_C,
            float,
        ).copy()

    np.savez_compressed(
        output_path,
        **payload,
    )


def summarise_stage_csv(path: str | Path) -> dict[str, Any]:
    """Return compact aggregate metrics from a long-format stage CSV."""
    import pandas as pd

    df = pd.read_csv(path)
    applied = df[df["was_applied"] == 1].copy()

    def mean_or_nan(name: str) -> float:
        if name not in applied or applied.empty:
            return np.nan
        return float(pd.to_numeric(applied[name], errors="coerce").mean())

    def max_or_nan(name: str) -> float:
        if name not in applied or applied.empty:
            return np.nan
        return float(pd.to_numeric(applied[name], errors="coerce").max())

    return {
        "rows": int(len(df)),
        "applied_rows": int(len(applied)),
        "mean_linearisation_err_xy_mm": mean_or_nan("linearisation_err_xy_mm"),
        "max_linearisation_err_xy_mm": max_or_nan("linearisation_err_xy_mm"),
        "mean_execution_err_xy_mm": mean_or_nan("execution_err_xy_mm"),
        "max_execution_err_xy_mm": max_or_nan("execution_err_xy_mm"),
        "mean_plant_vs_nl_err_xy_mm": mean_or_nan("plant_vs_nl_err_xy_mm"),
        "mean_tracking_actual_xy_mm": mean_or_nan("tracking_actual_xy_mm"),
        "mean_local_gain": mean_or_nan("local_gain"),
        "mean_direction_error_deg": mean_or_nan("direction_error_deg"),
    }


def plot_frame_rollout(path: str | Path, *, k: int):
    """Plot reference, linear prediction, nonlinear plan and applied rollout."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(path)
    frame = df[df["k"] == k].sort_values("stage")
    if frame.empty:
        raise ValueError(f"No rows found for k={k}.")

    fig, ax = plt.subplots()
    ax.plot(frame["ref_x_mm"], frame["ref_y_mm"], "o-", label="reference")
    ax.plot(frame["pred_x_mm"], frame["pred_y_mm"], "o-", label="linear/affine prediction")
    ax.plot(frame["nl_x_mm"], frame["nl_y_mm"], "o-", label="nonlinear plan")

    applied = frame[frame["was_applied"] == 1]
    if not applied.empty:
        ax.plot(applied["actual_x_mm"], applied["actual_y_mm"], "o-", label="applied plant rollout")

    for _, row in frame.iterrows():
        if np.isfinite(row["pred_x_mm"]) and np.isfinite(row["pred_y_mm"]):
            ax.annotate(str(int(row["stage"])), (row["pred_x_mm"], row["pred_y_mm"]))

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Rollout comparison at feedback frame k={k}")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_error_by_stage(path: str | Path):
    """Plot median rollout errors as a function of horizon stage."""
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(path)
    applied = df[df["was_applied"] == 1].copy()
    metrics = [
        "linearisation_err_xy_mm",
        "execution_err_xy_mm",
        "plant_vs_nl_err_xy_mm",
        "tracking_actual_xy_mm",
    ]
    grouped = applied.groupby("stage")[metrics].median(numeric_only=True)

    fig, ax = plt.subplots()
    for name in metrics:
        if name in grouped:
            ax.plot(grouped.index, grouped[name], marker="o", label=name)
    ax.set_xlabel("rollout stage")
    ax.set_ylabel("median error [mm]")
    ax.set_title("Error accumulation over the applied rollout")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_summary_json(stage_csv: str | Path, output_json: str | Path) -> Path:
    summary = summarise_stage_csv(stage_csv)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2))
    return output_json
