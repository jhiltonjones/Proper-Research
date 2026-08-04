#!/usr/bin/env python3
"""
Analyse how contact-vs-no-contact LTI effects change as rollout length increases.

Expected run tree, for example:

contact_vs_no_contact_rollout_sim2/
    bend_m30_jac_contact_plant_contact_1_lti_rollout3_Np3_sqp4/log.csv
    bend_m30_jac_contact_plant_contact_1_lti_rollout5_Np5_sqp4/log.csv
    bend_m30_jac_contact_plant_contact_1_lti_rollout7_Np7_sqp4/log.csv
    bend_m30_jac_no_contact_plant_contact_1_lti_rollout3_Np3_sqp4/log.csv
    ...

This script asks three questions:

1. For each bend and rollout, is the contact Jacobian better than no-contact?
2. As rollout increases from 3 -> 5 -> 7, do the measured effects increase,
   decrease, or behave non-monotonically?
3. Does the contact advantage itself grow or shrink with rollout length?

Outputs:
    RUN_ROOT/rollout_effect_analysis/
        rollout_run_summary.csv
        rollout_pairwise_contact_vs_no_contact.csv
        rollout_trend_by_run.csv
        contact_advantage_trends.csv
        rollout_horizon_error_detail.csv
        enriched_logs/*.csv
        summary_plots/*.png
        contact_advantage_plots/*.png
        horizon_error_plots/*.png
        heatmaps/*.png

Author: generated for Jack Hilton-Jones
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# User configuration
# =============================================================================

# Run the script from the directory containing this folder, or change this path.
RUN_ROOT = Path("contact_vs_no_contact_rollout_sim2")
OUT_DIR = RUN_ROOT / "rollout_effect_analysis"

# Set True if you want a per-row CSV with derived cosine/gain metrics for each run.
SAVE_ENRICHED_LOGS = True

# Tolerance used when deciding whether a trend is monotonic.
TREND_TOL = 1e-9


# =============================================================================
# Basic helpers
# =============================================================================


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric column, or NaNs if it is missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def safe_str(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a string column, or empty strings if it is missing."""
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index, dtype=str)


def nanmean(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def nanmedian(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan


def nanmax(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan


def nanmin(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanmin(a)) if np.isfinite(a).any() else np.nan


def nanpercentile(x: Sequence[float], q: float) -> float:
    a = np.asarray(x, dtype=float)
    return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else np.nan


def finite_slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Simple least-squares slope dy/dx, ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.nanmax(x[mask]) - np.nanmin(x[mask]) < 1e-12:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def signed_pct_change(first: float, last: float) -> float:
    """Percentage change from first to last, robust to zero."""
    if not np.isfinite(first) or not np.isfinite(last):
        return np.nan
    denom = abs(first)
    if denom < 1e-12:
        return np.nan
    return float(100.0 * (last - first) / denom)


def monotonic_label(values: Sequence[float], tol: float = TREND_TOL) -> str:
    """Classify raw numeric sequence as increasing, decreasing, flat, or non-monotonic."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return "insufficient_data"
    diffs = np.diff(vals)
    if np.all(np.abs(diffs) <= tol):
        return "flat"
    if np.all(diffs >= -tol):
        return "increasing"
    if np.all(diffs <= tol):
        return "decreasing"
    return "non_monotonic"


def performance_trend_label(values: Sequence[float], lower_is_better: bool) -> str:
    """
    Classify whether the metric improves/worsens as rollout increases.

    For lower-is-better metrics, decreasing is improvement.
    For higher-is-better metrics, increasing is improvement.
    """
    raw = monotonic_label(values)
    if raw in {"insufficient_data", "flat", "non_monotonic"}:
        return raw
    if lower_is_better:
        return "improves_with_rollout" if raw == "decreasing" else "worsens_with_rollout"
    return "improves_with_rollout" if raw == "increasing" else "worsens_with_rollout"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Run-name parsing
# =============================================================================


def parse_bend_angle_from_run_name(run_name: str) -> float:
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def parse_jacobian_variant(run_name: str) -> str:
    if "jac_no_contact" in run_name:
        return "no_contact"
    if "jac_contact" in run_name:
        return "contact"
    return "unknown"


def parse_int_after(pattern: str, run_name: str) -> float:
    m = re.search(pattern, run_name)
    if not m:
        return np.nan
    return float(m.group(1))


def parse_bool_after(pattern: str, run_name: str) -> Optional[bool]:
    m = re.search(pattern, run_name)
    if not m:
        return None
    return bool(int(m.group(1)))


def parse_run_metadata(run_dir: Path) -> Dict[str, object]:
    name = run_dir.name
    bend_angle = parse_bend_angle_from_run_name(name)
    return {
        "run_name": name,
        "run_dir": str(run_dir),
        "bend_angle_deg": bend_angle,
        "bend_abs_deg": abs(bend_angle) if np.isfinite(bend_angle) else np.nan,
        "jacobian_variant": parse_jacobian_variant(name),
        "plant_contact": parse_bool_after(r"plant_contact_(\d+)", name),
        "solver_mode_from_name": "lti" if "_lti_" in name else "unknown",
        "rollout": parse_int_after(r"rollout(\d+)", name),
        "Np": parse_int_after(r"Np(\d+)", name),
        "N_sqp": parse_int_after(r"sqp(\d+)", name),
    }


def read_runs(run_root: Path) -> List[Dict[str, object]]:
    runs = []
    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_dir = log_csv.parent
        try:
            df = pd.read_csv(log_csv)
        except Exception as exc:
            print(f"[WARN] Could not read {log_csv}: {exc}")
            continue

        meta = parse_run_metadata(run_dir)
        meta.update({"log_csv": str(log_csv), "df": df})
        runs.append(meta)

    return runs


# =============================================================================
# Per-row derived metrics: direction/cosine/gain
# =============================================================================


def available_horizon_indices(df: pd.DataFrame, kind: str = "rollout_err_xy") -> List[int]:
    """Return sorted horizon indices available in columns like rollout_err_xy_0_mm."""
    pat = re.compile(rf"{re.escape(kind)}_(\d+)_mm")
    out = []
    for c in df.columns:
        m = pat.fullmatch(c)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def vector_cols(df: pd.DataFrame, prefix: str, h: Optional[int], dims: Sequence[str]) -> Optional[pd.DataFrame]:
    """
    Return vector columns as a DataFrame, or None if any are missing.

    prefix examples:
        x_pred, x_rollout, u_applied
    h:
        horizon/control index, e.g. 0, 1, 2. If None, prefix is used directly.
    dims:
        e.g. ("x", "y", "z") or ("vx", "vy", "vz", "wx", "wy", "wz", "dL")
    """
    if h is None:
        cols = [f"{prefix}_{d}" for d in dims]
    else:
        cols = [f"{prefix}_{h}_{d}" for d in dims]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce")


def rowwise_norm(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(arr * arr, axis=1))


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.nansum(a * b, axis=1)
    na = rowwise_norm(a)
    nb = rowwise_norm(b)
    denom = na * nb
    out = np.full(len(dot), np.nan)
    mask = np.isfinite(dot) & np.isfinite(denom) & (denom > 1e-12)
    out[mask] = dot[mask] / denom[mask]
    out[mask] = np.clip(out[mask], -1.0, 1.0)
    return out


def add_direction_and_gain_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived predicted-vs-actual direction and gain metrics per horizon.

    Interpretation:
        baseline current tip position = tip_x, tip_y, tip_z if available,
        otherwise p_now_x, p_now_y, p_now_z.

        pred_delta_h   = x_pred_h    - baseline
        actual_delta_h = x_rollout_h - baseline

        cosine_h = cos(angle between predicted and actual displacement)

        cumulative control norm up to horizon h:
            ||[u_applied_0, ..., u_applied_h]||

        pred_gain_h   = ||pred_delta_h|| / cumulative_u_norm_h
        actual_gain_h = ||actual_delta_h|| / cumulative_u_norm_h
        gain_ratio_h  = pred_gain_h / actual_gain_h

    The gain is an effective input-output gain proxy. It is not a true logged
    singular value of the Jacobian unless you separately log J itself.
    """
    df = df.copy()

    base = vector_cols(df, "tip", None, ("x", "y", "z"))
    if base is None:
        base = vector_cols(df, "p_now", None, ("x", "y", "z"))
    if base is None:
        return df

    base_xyz = base.to_numpy(dtype=float)
    base_xy = base_xyz[:, :2]

    horizons = sorted(
        set(available_horizon_indices(df, "rollout_err_xy"))
        | set(available_horizon_indices(df, "rollout_err_xyz"))
    )

    # Precompute cumulative control vector norms by concatenating controls up to h.
    control_blocks: Dict[int, np.ndarray] = {}
    for h in horizons:
        blocks = []
        for j in range(h + 1):
            u = vector_cols(df, "u_applied", j, ("vx", "vy", "vz", "wx", "wy", "wz", "dL"))
            if u is not None:
                blocks.append(u.to_numpy(dtype=float))
        if blocks:
            U = np.concatenate(blocks, axis=1)
            control_blocks[h] = rowwise_norm(U)
            df[f"u_cumulative_norm_{h}"] = control_blocks[h]

    for h in horizons:
        pred = vector_cols(df, "x_pred", h, ("x", "y", "z"))
        actual = vector_cols(df, "x_rollout", h, ("x", "y", "z"))
        if pred is None or actual is None:
            continue

        pred_xyz = pred.to_numpy(dtype=float)
        actual_xyz = actual.to_numpy(dtype=float)

        d_pred_xyz = pred_xyz - base_xyz
        d_actual_xyz = actual_xyz - base_xyz
        d_pred_xy = d_pred_xyz[:, :2]
        d_actual_xy = d_actual_xyz[:, :2]

        pred_norm_xy = rowwise_norm(d_pred_xy)
        actual_norm_xy = rowwise_norm(d_actual_xy)
        pred_norm_xyz = rowwise_norm(d_pred_xyz)
        actual_norm_xyz = rowwise_norm(d_actual_xyz)

        cos_xy = rowwise_cosine(d_pred_xy, d_actual_xy)
        cos_xyz = rowwise_cosine(d_pred_xyz, d_actual_xyz)

        df[f"cos_pred_actual_xy_{h}"] = cos_xy
        df[f"cos_pred_actual_xyz_{h}"] = cos_xyz
        df[f"direction_error_xy_deg_{h}"] = np.degrees(np.arccos(np.clip(cos_xy, -1.0, 1.0)))
        df[f"direction_error_xyz_deg_{h}"] = np.degrees(np.arccos(np.clip(cos_xyz, -1.0, 1.0)))

        df[f"pred_disp_norm_xy_{h}"] = pred_norm_xy
        df[f"actual_disp_norm_xy_{h}"] = actual_norm_xy
        df[f"pred_disp_norm_xyz_{h}"] = pred_norm_xyz
        df[f"actual_disp_norm_xyz_{h}"] = actual_norm_xyz

        u_norm = control_blocks.get(h, np.full(len(df), np.nan))
        valid_u = np.isfinite(u_norm) & (u_norm > 1e-12)

        pred_gain_xy = np.full(len(df), np.nan)
        actual_gain_xy = np.full(len(df), np.nan)
        pred_gain_xyz = np.full(len(df), np.nan)
        actual_gain_xyz = np.full(len(df), np.nan)

        pred_gain_xy[valid_u] = pred_norm_xy[valid_u] / u_norm[valid_u]
        actual_gain_xy[valid_u] = actual_norm_xy[valid_u] / u_norm[valid_u]
        pred_gain_xyz[valid_u] = pred_norm_xyz[valid_u] / u_norm[valid_u]
        actual_gain_xyz[valid_u] = actual_norm_xyz[valid_u] / u_norm[valid_u]

        df[f"pred_gain_xy_{h}"] = pred_gain_xy
        df[f"actual_gain_xy_{h}"] = actual_gain_xy
        df[f"pred_gain_xyz_{h}"] = pred_gain_xyz
        df[f"actual_gain_xyz_{h}"] = actual_gain_xyz

        ratio_xy = np.full(len(df), np.nan)
        ratio_xyz = np.full(len(df), np.nan)
        mask_xy = np.isfinite(actual_gain_xy) & (np.abs(actual_gain_xy) > 1e-12)
        mask_xyz = np.isfinite(actual_gain_xyz) & (np.abs(actual_gain_xyz) > 1e-12)
        ratio_xy[mask_xy] = pred_gain_xy[mask_xy] / actual_gain_xy[mask_xy]
        ratio_xyz[mask_xyz] = pred_gain_xyz[mask_xyz] / actual_gain_xyz[mask_xyz]
        df[f"gain_ratio_pred_over_actual_xy_{h}"] = ratio_xy
        df[f"gain_ratio_pred_over_actual_xyz_{h}"] = ratio_xyz

        # Ideal gain ratio is 1. Use log error so over/under prediction are symmetric:
        # ratio 2 and 0.5 both produce abs(log10(ratio)) = 0.301.
        gain_ratio_error_xy = np.full(len(df), np.nan)
        gain_ratio_error_xyz = np.full(len(df), np.nan)
        mask_xy_pos = np.isfinite(ratio_xy) & (ratio_xy > 0)
        mask_xyz_pos = np.isfinite(ratio_xyz) & (ratio_xyz > 0)
        gain_ratio_error_xy[mask_xy_pos] = np.abs(np.log10(ratio_xy[mask_xy_pos]))
        gain_ratio_error_xyz[mask_xyz_pos] = np.abs(np.log10(ratio_xyz[mask_xyz_pos]))
        df[f"gain_ratio_log_error_xy_{h}"] = gain_ratio_error_xy
        df[f"gain_ratio_log_error_xyz_{h}"] = gain_ratio_error_xyz

    return df


# =============================================================================
# Summary metrics
# =============================================================================


def status_failure_mask(df: pd.DataFrame) -> pd.Series:
    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    good_status = status.isin(["solved", "solved_inaccurate", "solved inaccurate"])
    return (infeasible > 0) | (~good_status)


def first_existing_series(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return safe_num(df, c)
    return pd.Series(np.nan, index=df.index, dtype=float)


def summarise_horizon_errors(run: Dict[str, object], df: pd.DataFrame) -> List[Dict[str, object]]:
    """Return one row per available rollout horizon."""
    rows = []
    hs = sorted(available_horizon_indices(df, "rollout_err_xy"))
    for h in hs:
        xy = safe_num(df, f"rollout_err_xy_{h}_mm")
        xyz = safe_num(df, f"rollout_err_xyz_{h}_mm")
        cos = safe_num(df, f"cos_pred_actual_xy_{h}")
        direction = safe_num(df, f"direction_error_xy_deg_{h}")
        gain_ratio = safe_num(df, f"gain_ratio_pred_over_actual_xy_{h}")
        gain_ratio_error = safe_num(df, f"gain_ratio_log_error_xy_{h}")

        rows.append(
            {
                "run_name": run["run_name"],
                "bend_abs_deg": run["bend_abs_deg"],
                "bend_angle_deg": run["bend_angle_deg"],
                "jacobian_variant": run["jacobian_variant"],
                "rollout": run["rollout"],
                "Np": run["Np"],
                "N_sqp": run["N_sqp"],
                "horizon_index": h,
                "horizon_step": h + 1,
                "mean_rollout_err_xy_mm": float(xy.mean()),
                "median_rollout_err_xy_mm": float(xy.median()),
                "p95_rollout_err_xy_mm": float(xy.quantile(0.95)),
                "max_rollout_err_xy_mm": float(xy.max()),
                "mean_rollout_err_xyz_mm": float(xyz.mean()),
                "p95_rollout_err_xyz_mm": float(xyz.quantile(0.95)),
                "max_rollout_err_xyz_mm": float(xyz.max()),
                "mean_cos_pred_actual_xy": float(cos.mean()),
                "min_cos_pred_actual_xy": float(cos.min()),
                "frac_negative_cos_xy": float((cos < 0).mean()) if len(cos) else np.nan,
                "mean_direction_error_xy_deg": float(direction.mean()),
                "p95_direction_error_xy_deg": float(direction.quantile(0.95)),
                "median_gain_ratio_pred_over_actual_xy": float(gain_ratio.median()),
                "mean_gain_ratio_log_error_xy": float(gain_ratio_error.mean()),
                "p95_gain_ratio_log_error_xy": float(gain_ratio_error.quantile(0.95)),
            }
        )
    return rows


def summarise_run(run: Dict[str, object]) -> Tuple[Dict[str, object], pd.DataFrame, List[Dict[str, object]]]:
    """Summarise one run and return summary row, enriched dataframe, horizon detail rows."""
    df_raw = run["df"]
    assert isinstance(df_raw, pd.DataFrame)
    df = add_direction_and_gain_metrics(df_raw)

    failure_mask = status_failure_mask(df)
    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)

    # Main one-step/raw/adaptive errors.
    # For rollout > 1, pred1_err_xy_mm is the immediate one-step prediction error.
    one_step_xy = first_existing_series(df, ["pred1_err_xy_mm", "rollout_err_xy_0_mm"])
    one_step_xyz = first_existing_series(df, ["pred1_err_xyz_mm", "rollout_err_xyz_0_mm"])
    raw_xy = safe_num(df, "rollout_err_xy_0_mm")
    raw_xyz = safe_num(df, "rollout_err_xyz_0_mm")
    adapt_xy = safe_num(df, "adapt_pred_err_xy_mm")
    adapt_xyz = safe_num(df, "adapt_pred_err_xyz_mm")

    # Horizon-stack errors.
    h_xy_cols = [f"rollout_err_xy_{h}_mm" for h in available_horizon_indices(df, "rollout_err_xy")]
    h_xyz_cols = [f"rollout_err_xyz_{h}_mm" for h in available_horizon_indices(df, "rollout_err_xyz")]

    if h_xy_cols:
        E_xy = df[h_xy_cols].apply(pd.to_numeric, errors="coerce")
        horizon_mean_per_step_xy = E_xy.mean(axis=1)
        terminal_xy = E_xy.iloc[:, -1]
    else:
        E_xy = pd.DataFrame(index=df.index)
        horizon_mean_per_step_xy = pd.Series(np.nan, index=df.index)
        terminal_xy = pd.Series(np.nan, index=df.index)

    if h_xyz_cols:
        E_xyz = df[h_xyz_cols].apply(pd.to_numeric, errors="coerce")
        horizon_mean_per_step_xyz = E_xyz.mean(axis=1)
        terminal_xyz = E_xyz.iloc[:, -1]
    else:
        E_xyz = pd.DataFrame(index=df.index)
        horizon_mean_per_step_xyz = pd.Series(np.nan, index=df.index)
        terminal_xyz = pd.Series(np.nan, index=df.index)

    # Direction/gain at first horizon and terminal horizon.
    available_h = available_horizon_indices(df, "rollout_err_xy")
    h0 = available_h[0] if available_h else 0
    ht = available_h[-1] if available_h else 0

    cos0 = safe_num(df, f"cos_pred_actual_xy_{h0}")
    cost = safe_num(df, f"cos_pred_actual_xy_{ht}")
    dir0 = safe_num(df, f"direction_error_xy_deg_{h0}")
    dirt = safe_num(df, f"direction_error_xy_deg_{ht}")
    gain0 = safe_num(df, f"gain_ratio_pred_over_actual_xy_{h0}")
    gaint = safe_num(df, f"gain_ratio_pred_over_actual_xy_{ht}")
    gainerr0 = safe_num(df, f"gain_ratio_log_error_xy_{h0}")
    gainerrt = safe_num(df, f"gain_ratio_log_error_xy_{ht}")

    u0_cols = [
        "u_applied_0_vx", "u_applied_0_vy", "u_applied_0_vz",
        "u_applied_0_wx", "u_applied_0_wy", "u_applied_0_wz", "u_applied_0_dL",
    ]
    if all(c in df.columns for c in u0_cols):
        U0 = df[u0_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        u0_norm = pd.Series(rowwise_norm(U0), index=df.index)
    else:
        u_plan_cols = [
            "u_plan_0_vx", "u_plan_0_vy", "u_plan_0_vz",
            "u_plan_0_wx", "u_plan_0_wy", "u_plan_0_wz", "u_plan_0_dL",
        ]
        if all(c in df.columns for c in u_plan_cols):
            U0 = df[u_plan_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            u0_norm = pd.Series(rowwise_norm(U0), index=df.index)
        else:
            u0_norm = pd.Series(np.nan, index=df.index)

    row = {
        **{k: v for k, v in run.items() if k not in {"df"}},
        "n_steps": int(len(df)),
        "success": bool((~failure_mask).all()) if len(df) else False,
        "num_failed_steps": int(failure_mask.sum()) if len(df) else 0,
        "num_infeasible_steps": int(infeasible.sum()) if len(df) else 0,
        "final_status": str(status.iloc[-1]) if len(df) else "",
        "final_i_ref": float(safe_num(df, "i_ref").iloc[-1]) if len(df) else np.nan,
        "max_i_ref": float(safe_num(df, "i_ref").max()) if len(df) else np.nan,
        # One-step/raw/adaptive errors.
        "mean_one_step_xy_mm": float(one_step_xy.mean()),
        "median_one_step_xy_mm": float(one_step_xy.median()),
        "p95_one_step_xy_mm": float(one_step_xy.quantile(0.95)),
        "max_one_step_xy_mm": float(one_step_xy.max()),
        "mean_one_step_xyz_mm": float(one_step_xyz.mean()),
        "p95_one_step_xyz_mm": float(one_step_xyz.quantile(0.95)),
        "max_one_step_xyz_mm": float(one_step_xyz.max()),
        "mean_raw_one_step_xy_mm": float(raw_xy.mean()),
        "p95_raw_one_step_xy_mm": float(raw_xy.quantile(0.95)),
        "max_raw_one_step_xy_mm": float(raw_xy.max()),
        "mean_adapt_err_xy_mm": float(adapt_xy.mean()),
        "p95_adapt_err_xy_mm": float(adapt_xy.quantile(0.95)),
        "max_adapt_err_xy_mm": float(adapt_xy.max()),
        "mean_adapt_minus_raw_xy_mm": float((adapt_xy - raw_xy).mean()),
        "mean_adapt_over_raw_xy": float((adapt_xy / raw_xy.replace(0, np.nan)).mean()),
        # Multi-step rollout errors.
        "num_logged_horizons_xy": int(len(h_xy_cols)),
        "mean_rollout_all_xy_mm": float(E_xy.to_numpy(dtype=float).mean()) if h_xy_cols else np.nan,
        "median_rollout_all_xy_mm": float(np.nanmedian(E_xy.to_numpy(dtype=float))) if h_xy_cols else np.nan,
        "p95_rollout_all_xy_mm": float(np.nanpercentile(E_xy.to_numpy(dtype=float), 95)) if h_xy_cols else np.nan,
        "max_rollout_all_xy_mm": float(np.nanmax(E_xy.to_numpy(dtype=float))) if h_xy_cols else np.nan,
        "mean_rollout_mean_per_step_xy_mm": float(horizon_mean_per_step_xy.mean()),
        "p95_rollout_mean_per_step_xy_mm": float(horizon_mean_per_step_xy.quantile(0.95)),
        "mean_rollout_terminal_xy_mm": float(terminal_xy.mean()),
        "median_rollout_terminal_xy_mm": float(terminal_xy.median()),
        "p95_rollout_terminal_xy_mm": float(terminal_xy.quantile(0.95)),
        "max_rollout_terminal_xy_mm": float(terminal_xy.max()),
        "mean_rollout_all_xyz_mm": float(E_xyz.to_numpy(dtype=float).mean()) if h_xyz_cols else np.nan,
        "p95_rollout_all_xyz_mm": float(np.nanpercentile(E_xyz.to_numpy(dtype=float), 95)) if h_xyz_cols else np.nan,
        "max_rollout_all_xyz_mm": float(np.nanmax(E_xyz.to_numpy(dtype=float))) if h_xyz_cols else np.nan,
        "mean_rollout_terminal_xyz_mm": float(terminal_xyz.mean()),
        "p95_rollout_terminal_xyz_mm": float(terminal_xyz.quantile(0.95)),
        "max_rollout_terminal_xyz_mm": float(terminal_xyz.max()),
        # Direction quality.
        "mean_cos_one_step_xy": float(cos0.mean()),
        "min_cos_one_step_xy": float(cos0.min()),
        "frac_negative_cos_one_step_xy": float((cos0 < 0).mean()) if len(cos0) else np.nan,
        "mean_direction_error_one_step_xy_deg": float(dir0.mean()),
        "p95_direction_error_one_step_xy_deg": float(dir0.quantile(0.95)),
        "mean_cos_terminal_xy": float(cost.mean()),
        "min_cos_terminal_xy": float(cost.min()),
        "frac_negative_cos_terminal_xy": float((cost < 0).mean()) if len(cost) else np.nan,
        "mean_direction_error_terminal_xy_deg": float(dirt.mean()),
        "p95_direction_error_terminal_xy_deg": float(dirt.quantile(0.95)),
        # Effective gain proxy.
        "median_gain_ratio_one_step_xy": float(gain0.median()),
        "mean_gain_ratio_one_step_xy": float(gain0.mean()),
        "mean_gain_ratio_log_error_one_step_xy": float(gainerr0.mean()),
        "p95_gain_ratio_log_error_one_step_xy": float(gainerr0.quantile(0.95)),
        "median_gain_ratio_terminal_xy": float(gaint.median()),
        "mean_gain_ratio_terminal_xy": float(gaint.mean()),
        "mean_gain_ratio_log_error_terminal_xy": float(gainerrt.mean()),
        "p95_gain_ratio_log_error_terminal_xy": float(gainerrt.quantile(0.95)),
        # Control and safety/conditioning.
        "mean_u_norm": float(u0_norm.mean()),
        "p95_u_norm": float(u0_norm.quantile(0.95)),
        "max_u_norm": float(u0_norm.max()),
        "mean_clearance_mm": float(first_existing_series(df, ["adapt_clearance_mm", "clearance_mm"]).mean()),
        "min_clearance_mm": float(first_existing_series(df, ["adapt_clearance_mm", "clearance_mm"]).min()),
        "mean_tip_vessel_angle_deg": float(first_existing_series(df, ["adapt_tip_vessel_angle_deg", "tip_vessel_angle_deg"]).mean()),
        "max_tip_vessel_angle_deg": float(first_existing_series(df, ["adapt_tip_vessel_angle_deg", "tip_vessel_angle_deg"]).max()),
        "mean_cond_H_beam": float(safe_num(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_num(df, "cond_H_beam").max()),
        "mean_cond_H_mpc": float(first_existing_series(df, ["cond_H_mpc", "mpc_eig_cond"]).mean()),
        "max_cond_H_mpc": float(first_existing_series(df, ["cond_H_mpc", "mpc_eig_cond"]).max()),
        "mean_sqp_iters_done": float(safe_num(df, "sqp_iters_done").mean()),
        "sum_sqp_iters_done": float(safe_num(df, "sqp_iters_done").sum()),
        "mean_N_sqp_used": float(safe_num(df, "N_sqp_used").mean()),
        "sum_N_sqp_used": float(safe_num(df, "N_sqp_used").sum()),
    }

    horizon_rows = summarise_horizon_errors(run, df)
    return row, df, horizon_rows


# =============================================================================
# Contact/no-contact pairwise comparisons and rollout trend analysis
# =============================================================================


METRIC_INFO: Dict[str, Dict[str, object]] = {
    # lower is better
    "mean_one_step_xy_mm": {"label": "Mean one-step XY error [mm]", "lower_is_better": True},
    "p95_one_step_xy_mm": {"label": "P95 one-step XY error [mm]", "lower_is_better": True},
    "max_one_step_xy_mm": {"label": "Max one-step XY error [mm]", "lower_is_better": True},
    "mean_raw_one_step_xy_mm": {"label": "Mean raw one-step XY error [mm]", "lower_is_better": True},
    "mean_adapt_err_xy_mm": {"label": "Mean adaptive XY error [mm]", "lower_is_better": True},
    "mean_rollout_all_xy_mm": {"label": "Mean all-horizon rollout XY error [mm]", "lower_is_better": True},
    "p95_rollout_all_xy_mm": {"label": "P95 all-horizon rollout XY error [mm]", "lower_is_better": True},
    "mean_rollout_terminal_xy_mm": {"label": "Mean terminal rollout XY error [mm]", "lower_is_better": True},
    "p95_rollout_terminal_xy_mm": {"label": "P95 terminal rollout XY error [mm]", "lower_is_better": True},
    "max_rollout_terminal_xy_mm": {"label": "Max terminal rollout XY error [mm]", "lower_is_better": True},
    "mean_direction_error_one_step_xy_deg": {"label": "Mean one-step direction error [deg]", "lower_is_better": True},
    "p95_direction_error_one_step_xy_deg": {"label": "P95 one-step direction error [deg]", "lower_is_better": True},
    "mean_direction_error_terminal_xy_deg": {"label": "Mean terminal direction error [deg]", "lower_is_better": True},
    "p95_direction_error_terminal_xy_deg": {"label": "P95 terminal direction error [deg]", "lower_is_better": True},
    "mean_gain_ratio_log_error_one_step_xy": {"label": "Mean one-step gain-ratio log error", "lower_is_better": True},
    "p95_gain_ratio_log_error_one_step_xy": {"label": "P95 one-step gain-ratio log error", "lower_is_better": True},
    "mean_gain_ratio_log_error_terminal_xy": {"label": "Mean terminal gain-ratio log error", "lower_is_better": True},
    "p95_gain_ratio_log_error_terminal_xy": {"label": "P95 terminal gain-ratio log error", "lower_is_better": True},
    "mean_u_norm": {"label": "Mean control norm", "lower_is_better": True},
    "p95_u_norm": {"label": "P95 control norm", "lower_is_better": True},
    "max_u_norm": {"label": "Max control norm", "lower_is_better": True},
    "n_steps": {"label": "Number of MPC updates", "lower_is_better": True},
    "mean_cond_H_beam": {"label": "Mean beam Hessian condition", "lower_is_better": True},
    "mean_cond_H_mpc": {"label": "Mean MPC Hessian condition", "lower_is_better": True},
    # higher is better
    "mean_cos_one_step_xy": {"label": "Mean one-step cosine direction", "lower_is_better": False},
    "mean_cos_terminal_xy": {"label": "Mean terminal cosine direction", "lower_is_better": False},
    "mean_clearance_mm": {"label": "Mean clearance [mm]", "lower_is_better": False},
    "min_clearance_mm": {"label": "Minimum clearance [mm]", "lower_is_better": False},
}


def contact_effect(contact: float, no_contact: float, lower_is_better: bool) -> Tuple[float, float]:
    """
    Return signed contact effect and percentage improvement.

    Positive signed effect means contact is better.
    Percentage improvement is relative to |no_contact|.
    """
    if not np.isfinite(contact) or not np.isfinite(no_contact):
        return np.nan, np.nan
    if lower_is_better:
        diff = no_contact - contact
    else:
        diff = contact - no_contact
    denom = abs(no_contact)
    pct = np.nan if denom < 1e-12 else 100.0 * diff / denom
    return float(diff), float(pct)


def build_pairwise_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["bend_abs_deg", "rollout"]
    for (bend, rollout), g in summary.groupby(keys):
        c = g[g["jacobian_variant"] == "contact"]
        n = g[g["jacobian_variant"] == "no_contact"]
        if c.empty or n.empty:
            continue
        c = c.iloc[0]
        n = n.iloc[0]
        for metric, info in METRIC_INFO.items():
            if metric not in summary.columns:
                continue
            lower = bool(info["lower_is_better"])
            cv = float(c[metric]) if pd.notna(c[metric]) else np.nan
            nv = float(n[metric]) if pd.notna(n[metric]) else np.nan
            effect, pct = contact_effect(cv, nv, lower)
            rows.append(
                {
                    "bend_abs_deg": bend,
                    "rollout": rollout,
                    "metric": metric,
                    "metric_label": info["label"],
                    "lower_is_better": lower,
                    "contact": cv,
                    "no_contact": nv,
                    "contact_minus_no_contact_raw": cv - nv if np.isfinite(cv) and np.isfinite(nv) else np.nan,
                    "signed_contact_effect_positive_is_better": effect,
                    "contact_improvement_pct": pct,
                    "contact_better": bool(effect > 0) if np.isfinite(effect) else False,
                }
            )
    return pd.DataFrame(rows)


def build_run_trends(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, info in METRIC_INFO.items():
        if metric not in summary.columns:
            continue
        lower = bool(info["lower_is_better"])
        for (bend, jac), g in summary.groupby(["bend_abs_deg", "jacobian_variant"]):
            g = g.sort_values("rollout")
            xs = g["rollout"].to_numpy(dtype=float)
            ys = g[metric].to_numpy(dtype=float)
            if len(g) < 2:
                continue
            rows.append(
                {
                    "bend_abs_deg": bend,
                    "jacobian_variant": jac,
                    "metric": metric,
                    "metric_label": info["label"],
                    "lower_is_better": lower,
                    "rollout_values": ",".join(str(int(v)) for v in xs if np.isfinite(v)),
                    "values": ",".join(f"{v:.6g}" for v in ys),
                    "value_at_min_rollout": float(ys[0]),
                    "value_at_max_rollout": float(ys[-1]),
                    "raw_pct_change_min_to_max": signed_pct_change(float(ys[0]), float(ys[-1])),
                    "slope_per_rollout_step": finite_slope(xs, ys),
                    "raw_monotonicity": monotonic_label(ys),
                    "performance_trend": performance_trend_label(ys, lower),
                }
            )
    return pd.DataFrame(rows)


def build_contact_advantage_trends(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairwise.empty:
        return pd.DataFrame()
    for (bend, metric), g in pairwise.groupby(["bend_abs_deg", "metric"]):
        g = g.sort_values("rollout")
        xs = g["rollout"].to_numpy(dtype=float)
        ys = g["contact_improvement_pct"].to_numpy(dtype=float)
        eff = g["signed_contact_effect_positive_is_better"].to_numpy(dtype=float)
        if len(g) < 2:
            continue
        rows.append(
            {
                "bend_abs_deg": bend,
                "metric": metric,
                "metric_label": g["metric_label"].iloc[0],
                "lower_is_better": bool(g["lower_is_better"].iloc[0]),
                "rollout_values": ",".join(str(int(v)) for v in xs if np.isfinite(v)),
                "contact_improvement_pct_values": ",".join(f"{v:.6g}" for v in ys),
                "signed_contact_effect_values": ",".join(f"{v:.6g}" for v in eff),
                "improvement_pct_at_min_rollout": float(ys[0]),
                "improvement_pct_at_max_rollout": float(ys[-1]),
                "improvement_pct_change_min_to_max": float(ys[-1] - ys[0]) if np.isfinite(ys[0]) and np.isfinite(ys[-1]) else np.nan,
                "slope_improvement_pct_per_rollout_step": finite_slope(xs, ys),
                "improvement_monotonicity": monotonic_label(ys),
                "contact_advantage_trend": performance_trend_label(ys, lower_is_better=False),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================


def style_axes(ax):
    ax.grid(True, alpha=0.3)
    return ax


def plot_metric_vs_rollout(summary: pd.DataFrame, out_dir: Path, metric: str):
    if metric not in summary.columns or summary.empty:
        return
    info = METRIC_INFO.get(metric, {"label": metric})
    label = str(info.get("label", metric))
    bends = sorted(summary["bend_abs_deg"].dropna().unique())
    if not bends:
        return

    ncols = 3
    nrows = int(math.ceil(len(bends) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, bend in zip(axes, bends):
        g = summary[summary["bend_abs_deg"] == bend]
        for jac in ["contact", "no_contact"]:
            gj = g[g["jacobian_variant"] == jac].sort_values("rollout")
            if gj.empty:
                continue
            ax.plot(gj["rollout"], gj[metric], marker="o", linewidth=2.0, label=jac)
        ax.set_title(f"{int(bend)}° bend")
        ax.set_xlabel("Rollout / Np")
        ax.set_ylabel(label)
        style_axes(ax)
        ax.legend(fontsize=8)

    for ax in axes[len(bends):]:
        ax.axis("off")

    fig.suptitle(f"{label} vs rollout length", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}_vs_rollout_by_bend.png", dpi=300)
    plt.close(fig)


def plot_contact_advantage(pairwise: pd.DataFrame, out_dir: Path, metric: str):
    if pairwise.empty:
        return
    g = pairwise[pairwise["metric"] == metric]
    if g.empty:
        return

    label = g["metric_label"].iloc[0]
    bends = sorted(g["bend_abs_deg"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for bend in bends:
        gb = g[g["bend_abs_deg"] == bend].sort_values("rollout")
        ax.plot(
            gb["rollout"],
            gb["contact_improvement_pct"],
            marker="o",
            linewidth=2.0,
            label=f"{int(bend)}°",
        )
    ax.axhline(0.0, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Rollout / Np")
    ax.set_ylabel("Contact improvement [%]\npositive = contact better")
    ax.set_title(f"Contact advantage vs rollout: {label}")
    style_axes(ax)
    ax.legend(title="Bend", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"contact_improvement_{metric}_vs_rollout.png", dpi=300)
    plt.close(fig)


def plot_metric_heatmap(pairwise: pd.DataFrame, out_dir: Path, metric: str):
    if pairwise.empty:
        return
    g = pairwise[pairwise["metric"] == metric]
    if g.empty:
        return
    piv = g.pivot(index="bend_abs_deg", columns="rollout", values="contact_improvement_pct").sort_index()
    if piv.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    im = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([str(int(c)) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{int(b)}°" for b in piv.index])
    ax.set_xlabel("Rollout / Np")
    ax.set_ylabel("Bend")
    ax.set_title(f"Contact improvement [%]: {g['metric_label'].iloc[0]}")

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Contact improvement [%], positive = contact better")
    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_contact_improvement_{metric}.png", dpi=300)
    plt.close(fig)


def plot_horizon_error_curves(horizon: pd.DataFrame, out_dir: Path):
    if horizon.empty:
        return
    for bend in sorted(horizon["bend_abs_deg"].dropna().unique()):
        for rollout in sorted(horizon["rollout"].dropna().unique()):
            g = horizon[(horizon["bend_abs_deg"] == bend) & (horizon["rollout"] == rollout)]
            if g.empty:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.4))
            for jac in ["contact", "no_contact"]:
                gj = g[g["jacobian_variant"] == jac].sort_values("horizon_step")
                if gj.empty:
                    continue
                ax.plot(
                    gj["horizon_step"],
                    gj["mean_rollout_err_xy_mm"],
                    marker="o",
                    linewidth=2.0,
                    label=f"{jac}: mean",
                )
                ax.plot(
                    gj["horizon_step"],
                    gj["p95_rollout_err_xy_mm"],
                    marker="s",
                    linestyle="--",
                    linewidth=1.6,
                    label=f"{jac}: p95",
                )
            ax.set_xlabel("Prediction step inside rollout")
            ax.set_ylabel("Rollout prediction error [mm]")
            ax.set_title(f"Error growth along rollout, {int(bend)}° bend, rollout {int(rollout)}")
            style_axes(ax)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"horizon_error_bend_{int(bend)}_rollout_{int(rollout)}.png", dpi=300)
            plt.close(fig)


def make_all_plots(summary: pd.DataFrame, pairwise: pd.DataFrame, horizon: pd.DataFrame, out_dir: Path):
    summary_dir = ensure_dir(out_dir / "summary_plots")
    advantage_dir = ensure_dir(out_dir / "contact_advantage_plots")
    heatmap_dir = ensure_dir(out_dir / "heatmaps")
    horizon_dir = ensure_dir(out_dir / "horizon_error_plots")

    key_metrics = [
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_adapt_err_xy_mm",
        "mean_rollout_all_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_rollout_terminal_xy_mm",
        "p95_rollout_terminal_xy_mm",
        "mean_cos_one_step_xy",
        "mean_direction_error_one_step_xy_deg",
        "mean_cos_terminal_xy",
        "mean_direction_error_terminal_xy_deg",
        "mean_gain_ratio_log_error_one_step_xy",
        "mean_gain_ratio_log_error_terminal_xy",
        "mean_u_norm",
        "n_steps",
        "mean_clearance_mm",
        "min_clearance_mm",
        "mean_cond_H_mpc",
    ]

    for metric in key_metrics:
        plot_metric_vs_rollout(summary, summary_dir, metric)
        plot_contact_advantage(pairwise, advantage_dir, metric)
        plot_metric_heatmap(pairwise, heatmap_dir, metric)

    plot_horizon_error_curves(horizon, horizon_dir)


# =============================================================================
# Console reporting
# =============================================================================


def print_metric_table(summary: pd.DataFrame, metric: str):
    if metric not in summary.columns or summary.empty:
        return
    print(f"\n[{metric}]")
    cols = ["bend_abs_deg", "rollout", "jacobian_variant", metric]
    table = summary[cols].sort_values(["bend_abs_deg", "rollout", "jacobian_variant"])
    print(table.to_string(index=False))


def print_headline_report(summary: pd.DataFrame, pairwise: pd.DataFrame, trends: pd.DataFrame, advantage_trends: pd.DataFrame):
    print("\n" + "=" * 88)
    print("ROLLOUT EFFECT ANALYSIS")
    print("=" * 88)
    print(f"Runs analysed: {len(summary)}")
    if len(summary):
        print(f"Bends: {sorted(summary['bend_abs_deg'].dropna().unique())}")
        print(f"Rollouts: {sorted(summary['rollout'].dropna().unique())}")
        print(f"Jacobian variants: {sorted(summary['jacobian_variant'].dropna().unique())}")

    for metric in [
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_rollout_terminal_xy_mm",
        "p95_rollout_terminal_xy_mm",
        "mean_cos_one_step_xy",
        "mean_direction_error_one_step_xy_deg",
        "mean_gain_ratio_log_error_one_step_xy",
        "mean_u_norm",
        "n_steps",
    ]:
        print_metric_table(summary, metric)

    if not advantage_trends.empty:
        print("\n" + "=" * 88)
        print("CONTACT ADVANTAGE TREND SUMMARY")
        print("Positive improvement means contact is better than no-contact.")
        print("Trend says whether that contact advantage increases/decreases as rollout grows.")
        print("=" * 88)
        show_metrics = [
            "mean_one_step_xy_mm",
            "p95_one_step_xy_mm",
            "mean_rollout_terminal_xy_mm",
            "p95_rollout_terminal_xy_mm",
            "mean_direction_error_one_step_xy_deg",
            "mean_gain_ratio_log_error_one_step_xy",
            "mean_u_norm",
            "n_steps",
        ]
        show = advantage_trends[advantage_trends["metric"].isin(show_metrics)].copy()
        show = show.sort_values(["metric", "bend_abs_deg"])
        cols = [
            "metric",
            "bend_abs_deg",
            "contact_improvement_pct_values",
            "improvement_pct_change_min_to_max",
            "improvement_monotonicity",
            "contact_advantage_trend",
        ]
        with pd.option_context("display.max_colwidth", 80):
            print(show[cols].to_string(index=False))

    print("\nInterpretation notes:")
    print("  - lower-is-better metrics: errors, direction error, gain-ratio log error, control norm, n_steps")
    print("  - higher-is-better metrics: cosine direction and clearance")
    print("  - gain-ratio log error is |log10(predicted_gain / actual_gain)|, so 0 is ideal")
    print("  - terminal rollout error is the last logged horizon step for that rollout")
    print("  - all-horizon rollout error pools every logged horizon step in the run")


# =============================================================================
# Main
# =============================================================================


def main():
    out_dir = ensure_dir(OUT_DIR)
    ensure_dir(out_dir / "enriched_logs")

    runs = read_runs(RUN_ROOT)
    if not runs:
        raise RuntimeError(f"No log.csv files found under {RUN_ROOT.resolve()}")

    summary_rows = []
    horizon_rows_all = []

    for run in runs:
        row, enriched, horizon_rows = summarise_run(run)
        summary_rows.append(row)
        horizon_rows_all.extend(horizon_rows)

        if SAVE_ENRICHED_LOGS:
            name = str(run["run_name"])
            enriched.to_csv(out_dir / "enriched_logs" / f"{name}_enriched.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["bend_abs_deg", "rollout", "jacobian_variant"]).reset_index(drop=True)

    horizon = pd.DataFrame(horizon_rows_all)
    if not horizon.empty:
        horizon = horizon.sort_values(["bend_abs_deg", "rollout", "jacobian_variant", "horizon_index"])

    pairwise = build_pairwise_comparison(summary)
    trends = build_run_trends(summary)
    advantage_trends = build_contact_advantage_trends(pairwise)

    # Aggregate summaries across bends for each rollout and Jacobian.
    numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
    aggregate = (
        summary.groupby(["rollout", "jacobian_variant"], as_index=False)[numeric_cols]
        .mean(numeric_only=True)
        .sort_values(["rollout", "jacobian_variant"])
    )

    # Save outputs.
    summary.to_csv(out_dir / "rollout_run_summary.csv", index=False)
    horizon.to_csv(out_dir / "rollout_horizon_error_detail.csv", index=False)
    pairwise.to_csv(out_dir / "rollout_pairwise_contact_vs_no_contact.csv", index=False)
    trends.to_csv(out_dir / "rollout_trend_by_run.csv", index=False)
    advantage_trends.to_csv(out_dir / "contact_advantage_trends.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_by_rollout_and_jacobian.csv", index=False)

    # JSON versions for easy inspection.
    for df, fname in [
        (summary, "rollout_run_summary.json"),
        (pairwise, "rollout_pairwise_contact_vs_no_contact.json"),
        (trends, "rollout_trend_by_run.json"),
        (advantage_trends, "contact_advantage_trends.json"),
    ]:
        with open(out_dir / fname, "w") as f:
            json.dump(df.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    make_all_plots(summary, pairwise, horizon, out_dir)
    print_headline_report(summary, pairwise, trends, advantage_trends)

    print("\nSaved analysis to:")
    print(f"  {out_dir.resolve()}")
    print("\nMost useful files:")
    print("  rollout_run_summary.csv")
    print("  rollout_pairwise_contact_vs_no_contact.csv")
    print("  contact_advantage_trends.csv")
    print("  rollout_horizon_error_detail.csv")
    print("  summary_plots/*.png")
    print("  contact_advantage_plots/*.png")
    print("  horizon_error_plots/*.png")


if __name__ == "__main__":
    main()
