#!/usr/bin/env python3
"""
Analyse geometry / numerical-conditioning features as online measures for choosing a candidate rollout.

Goal
----
Find features such that increasing feature value makes longer rollout increasingly risky.
This is intended for hierarchical MPC/SQP controller design:

    geometry / conditioning features -> candidate rollout
    initial SQP probe                -> revise rollout downward
    final SQP tolerance              -> accept/reject solve

Inputs
------
Either:
  1) a run root containing subdirectories with log.csv files, e.g.
       run_figure4_simulation_sqp_investigation2/

  2) a precomputed event CSV that already contains per-step rows and the relevant columns.

The script will prefer raw log.csv files because they contain the richest conditioning columns:
    cond_H_beam, lambda_min_H_beam, lambda_max_H_beam,
    num_negative_H_beam, num_near_zero_H_beam,
    cond_H_mpc, lambda_min_H_mpc, lambda_max_H_mpc,
    mpc_eig_cond, mpc_weak_lambda, mpc_strong_lambda,
    clearance_mm, adapt_clearance_mm,
    curvature_local_1pm, curvature_max_ahead_1pm, curvature_mean_ahead_1pm,
    sqp_du_0, sqp_du_rel_0, sqp_du_final, sqp_du_rel_final.

Outputs
-------
  geometry_rollout_measure_analysis/
    geometry_event_table.csv
    feature_predictiveness_summary.csv
    feature_threshold_sweep.csv
    feature_bin_rollout_risk.csv
    candidate_rollout_by_feature_bin.csv
    composite_risk_rollout_policy.csv
    composite_score_events.csv
    geometry_rollout_report.md
    plots/*.png

Usage
-----
  python analyse_geometry_candidate_rollout.py run_figure4_simulation_sqp_investigation2 \
      --target-spike-rate 0.05 \
      --spike-error-mm 1.0

  python analyse_geometry_candidate_rollout.py events_with_tolerance_bins.csv \
      --out-dir geometry_rollout_measure_analysis
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


# -----------------------------
# Utilities
# -----------------------------

RUN_RE = re.compile(
    r"bend_m(?P<bend>\d+)_jac_(?P<jac>contact|no_contact)_plant_contact_(?P<plant>[01])_"
    r"(?P<solver>[a-zA-Z0-9_]+?)_rollout(?P<rollout>\d+)_Np(?P<Np>\d+)_sqp(?P<sqp>\d+)"
)


def parse_run_name(name: str) -> Dict[str, object]:
    m = RUN_RE.search(name)
    if not m:
        # Return NaNs rather than failing; script can still process arbitrary logs.
        return {
            "bend_angle_deg": np.nan,
            "bend_magnitude_deg": np.nan,
            "jacobian_variant": np.nan,
            "solver_mode": np.nan,
            "rollout_steps": np.nan,
            "Np": np.nan,
            "sqp_budget": np.nan,
        }
    bend_mag = float(m.group("bend"))
    return {
        "bend_angle_deg": -bend_mag,
        "bend_magnitude_deg": bend_mag,
        "jacobian_variant": m.group("jac"),
        "solver_mode": m.group("solver"),
        "rollout_steps": int(m.group("rollout")),
        "Np": int(m.group("Np")),
        "sqp_budget": int(m.group("sqp")),
    }


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def get_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return to_num(df[col])
    return pd.Series(default, index=df.index, dtype="float64")


def safe_log10(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    x = to_num(x)
    out = pd.Series(np.nan, index=x.index, dtype="float64")
    mask = x > 0
    out.loc[mask] = np.log10(np.maximum(x.loc[mask], eps))
    return out


def signed_low_risk_to_high_risk(x: pd.Series) -> pd.Series:
    # Converts a quantity where low values are risky into a high-is-risk score.
    return -to_num(x)


def qcut_labels(s: pd.Series, labels: Sequence[str]) -> pd.Series:
    s = to_num(s)
    out = pd.Series("missing", index=s.index, dtype="object")
    valid = s.dropna()
    if len(valid) < len(labels):
        return out
    try:
        binned = pd.qcut(valid, q=len(labels), labels=labels, duplicates="drop")
        out.loc[valid.index] = binned.astype(str)
    except Exception:
        # Fallback: use rank to break ties.
        ranks = valid.rank(method="first")
        try:
            binned = pd.qcut(ranks, q=len(labels), labels=labels, duplicates="drop")
            out.loc[valid.index] = binned.astype(str)
        except Exception:
            pass
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    x = to_num(x)
    y = to_num(y)
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 5 or x.loc[mask].nunique(dropna=True) < 2 or y.loc[mask].nunique(dropna=True) < 2:
        return np.nan, np.nan, n
    if stats is None:
        return x.loc[mask].corr(y.loc[mask], method="spearman"), np.nan, n
    r, p = stats.spearmanr(x.loc[mask], y.loc[mask], nan_policy="omit")
    return float(r), float(p), n


def slope_ols_simple(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    x = to_num(x)
    y = to_num(y)
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 5 or x.loc[mask].nunique(dropna=True) < 2:
        return np.nan, np.nan, n
    if stats is None:
        return np.nan, np.nan, n
    lr = stats.linregress(x.loc[mask].astype(float), y.loc[mask].astype(float))
    return float(lr.slope), float(lr.pvalue), n


# -----------------------------
# Load/build event table
# -----------------------------

def read_input(input_path: Path, spike_error_mm: float) -> pd.DataFrame:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        df = add_derived_event_columns(df, spike_error_mm=spike_error_mm)
        return df

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    logs = sorted(input_path.glob("**/log.csv"))
    if not logs:
        raise FileNotFoundError(f"No log.csv files found under {input_path}")

    rows: List[pd.DataFrame] = []
    for log_path in logs:
        run_dir = log_path.parent
        run_name = run_dir.name
        meta = parse_run_name(run_name)
        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            warnings.warn(f"Failed to read {log_path}: {e}")
            continue
        if df.empty:
            continue

        ev = pd.DataFrame(index=df.index)
        ev["run_name"] = run_name
        ev["run_dir"] = str(run_dir)
        ev["log_csv"] = str(log_path)
        for k, v in meta.items():
            ev[k] = v

        # Basic indices and status.
        for col in [
            "k", "i_ref", "i_ref_mpc", "idx_ref_terminal", "path_idx_nearest",
            "status", "infeasible", "solver_mode", "solver_mode_used",
            "rollout_steps", "N_sqp_used", "rollout_steps_used", "sqp_iters_done",
        ]:
            if col in df.columns:
                ev[col] = df[col]

        # Error metrics.
        for col in [
            "adapt_pred_err_xy_mm", "adapt_pred_err_xyz_mm",
            "pred1_err_xy_mm", "pred1_err_xyz_mm",
            "rollout_err_xy_0_mm", "rollout_err_xy_1_mm", "rollout_err_xy_2_mm",
            "rollout_err_xy_3_mm", "rollout_err_xy_4_mm",
        ]:
            if col in df.columns:
                ev[col] = to_num(df[col])

        # Geometry/contact/path features.
        for col in [
            "rho_mm", "clearance_mm", "adapt_clearance_mm",
            "tip_vessel_angle_deg", "adapt_tip_vessel_angle_deg",
            "curvature_local_1pm", "curvature_max_ahead_1pm", "curvature_mean_ahead_1pm",
            "curvature_lookahead_points",
        ]:
            if col in df.columns:
                ev[col] = to_num(df[col])

        # Beam Hessian and optimisation conditioning.
        for col in [
            "cond_H_beam", "lambda_min_H_beam", "lambda_max_H_beam",
            "num_negative_H_beam", "num_near_zero_H_beam", "stationarity_norm",
            "cond_H_mpc", "lambda_min_H_mpc", "lambda_max_H_mpc",
            "num_negative_H_mpc", "num_near_zero_H_mpc", "valid_H_mpc",
            "norm_H_mpc", "norm_H_track", "norm_H_effort", "norm_H_smooth", "qp_reg",
            "mpc_eig_logged", "mpc_eig_failed", "mpc_eig_cond",
            "mpc_weak_lambda", "mpc_strong_lambda",
            "mpc_weak_channel_energy", "mpc_strong_channel_energy",
            "beam_stiff_direction_absmax",
        ]:
            if col in df.columns:
                ev[col] = to_num(df[col])

        # SQP diagnostics.
        for col in ["sqp_du_final", "sqp_du_rel_final"]:
            if col in df.columns:
                ev[col] = to_num(df[col])
        # First SQP update is iteration 0.
        if "sqp_du_0" in df.columns:
            ev["sqp_du_abs_first"] = to_num(df["sqp_du_0"])
        if "sqp_du_rel_0" in df.columns:
            ev["sqp_du_rel_first"] = to_num(df["sqp_du_rel_0"])

        # Carry all individual SQP updates if present.
        for c in df.columns:
            if re.match(r"sqp_du(_rel)?_\d+$", c):
                ev[c] = to_num(df[c])

        rows.append(ev)

    if not rows:
        raise RuntimeError("No logs were successfully read.")

    out = pd.concat(rows, ignore_index=True)
    out = add_derived_event_columns(out, spike_error_mm=spike_error_mm)
    return out


def add_derived_event_columns(df: pd.DataFrame, spike_error_mm: float) -> pd.DataFrame:
    df = df.copy()

    # Ensure metadata numeric.
    for col in ["bend_angle_deg", "bend_magnitude_deg", "rollout_steps", "Np", "sqp_budget", "k", "i_ref"]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # Error deltas inside each run.
    if "adapt_pred_err_xy_mm" in df.columns:
        df["adapt_pred_err_xy_mm"] = to_num(df["adapt_pred_err_xy_mm"])
        if "run_name" in df.columns:
            df = df.sort_values(["run_name", "k" if "k" in df.columns else df.index.name or "run_name"]).reset_index(drop=True)
            df["previous_adapt_pred_err_xy_mm"] = df.groupby("run_name")["adapt_pred_err_xy_mm"].shift(1)
            df["next_adapt_pred_err_xy_mm"] = df.groupby("run_name")["adapt_pred_err_xy_mm"].shift(-1)
        else:
            df["previous_adapt_pred_err_xy_mm"] = df["adapt_pred_err_xy_mm"].shift(1)
            df["next_adapt_pred_err_xy_mm"] = df["adapt_pred_err_xy_mm"].shift(-1)
        df["error_delta_from_prev"] = df["adapt_pred_err_xy_mm"] - df["previous_adapt_pred_err_xy_mm"]
        df["next_error_delta"] = df["next_adapt_pred_err_xy_mm"] - df["adapt_pred_err_xy_mm"]
        df["is_next_error_spike"] = (df["next_error_delta"] >= spike_error_mm).astype(float)

    # Normalise existing spike column if present.
    if "is_next_error_spike" in df.columns:
        df["is_next_error_spike_int"] = to_num(df["is_next_error_spike"]).fillna(0).astype(int)

    # Final and first SQP absolute/relative names.
    if "sqp_du_abs_first" not in df.columns and "sqp_du_0" in df.columns:
        df["sqp_du_abs_first"] = to_num(df["sqp_du_0"])
    if "sqp_du_rel_first" not in df.columns and "sqp_du_rel_0" in df.columns:
        df["sqp_du_rel_first"] = to_num(df["sqp_du_rel_0"])
    if "sqp_du_abs_final" not in df.columns and "sqp_du_final" in df.columns:
        df["sqp_du_abs_final"] = to_num(df["sqp_du_final"])
    if "sqp_du_rel_final" not in df.columns and "sqp_du_rel_final" in df.columns:
        df["sqp_du_rel_final"] = to_num(df["sqp_du_rel_final"])

    # Transitions between optimisation problems.
    if "run_name" in df.columns:
        group = df.groupby("run_name", sort=False)
        if "sqp_du_abs_final" in df.columns and "sqp_du_abs_first" in df.columns:
            prev_final = group["sqp_du_abs_final"].shift(1)
            df["prev_final_to_current_first_abs_delta"] = df["sqp_du_abs_first"] - prev_final
            df["prev_final_to_current_first_abs_ratio"] = df["sqp_du_abs_first"] / prev_final.replace(0, np.nan)
        if "sqp_du_rel_final" in df.columns and "sqp_du_rel_first" in df.columns:
            prev_final = group["sqp_du_rel_final"].shift(1)
            df["prev_final_to_current_first_rel_delta"] = df["sqp_du_rel_first"] - prev_final
            df["prev_final_to_current_first_rel_ratio"] = df["sqp_du_rel_first"] / prev_final.replace(0, np.nan)

    # Derived risk-oriented columns: higher means more risky.
    if "clearance_mm" in df.columns:
        df["risk_low_clearance_mm"] = signed_low_risk_to_high_risk(df["clearance_mm"])
        c = to_num(df["clearance_mm"])
        df["risk_inv_clearance"] = np.where(c > 1e-9, 1.0 / c, np.nan)
    if "adapt_clearance_mm" in df.columns:
        df["risk_low_adapt_clearance_mm"] = signed_low_risk_to_high_risk(df["adapt_clearance_mm"])

    for col in ["cond_H_beam", "cond_H_mpc", "mpc_eig_cond"]:
        if col in df.columns:
            df[f"log10_{col}"] = safe_log10(df[col])

    for col in ["lambda_min_H_beam", "lambda_min_H_mpc", "mpc_weak_lambda"]:
        if col in df.columns:
            # Low/near-zero eigenvalues are risky. This score increases as lambda decreases.
            x = to_num(df[col])
            df[f"risk_low_{col}"] = -np.log10(np.maximum(x.where(x > 0), 1e-12))
            # Negative eigenvalues are very risky; store separately.
            df[f"is_negative_{col}"] = (x < 0).astype(float)

    if "lambda_max_H_beam" in df.columns and "lambda_min_H_beam" in df.columns:
        lmax = get_col(df, "lambda_max_H_beam")
        lmin = get_col(df, "lambda_min_H_beam")
        df["beam_lambda_spread"] = lmax - lmin
        df["log10_beam_lambda_spread"] = safe_log10(df["beam_lambda_spread"])

    if "lambda_max_H_mpc" in df.columns and "lambda_min_H_mpc" in df.columns:
        lmax = get_col(df, "lambda_max_H_mpc")
        lmin = get_col(df, "lambda_min_H_mpc")
        df["mpc_lambda_spread"] = lmax - lmin
        df["log10_mpc_lambda_spread"] = safe_log10(df["mpc_lambda_spread"])

    return df


# -----------------------------
# Feature screening
# -----------------------------

def candidate_feature_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        # Primary proposed measures.
        "log10_cond_H_beam", "cond_H_beam",
        "risk_low_lambda_min_H_beam", "lambda_min_H_beam",
        "num_near_zero_H_beam", "num_negative_H_beam",
        "beam_lambda_spread", "log10_beam_lambda_spread",
        "beam_stiff_direction_absmax",
        # MPC Hessian / numerical measures.
        "log10_cond_H_mpc", "cond_H_mpc",
        "log10_mpc_eig_cond", "mpc_eig_cond",
        "risk_low_lambda_min_H_mpc", "lambda_min_H_mpc",
        "num_near_zero_H_mpc", "num_negative_H_mpc",
        "log10_mpc_lambda_spread", "norm_H_mpc", "qp_reg", "stationarity_norm",
        "risk_low_mpc_weak_lambda", "mpc_weak_lambda", "mpc_weak_channel_energy",
        # Geometry/contact.
        "risk_low_clearance_mm", "risk_inv_clearance", "risk_low_adapt_clearance_mm",
        "rho_mm", "tip_vessel_angle_deg", "adapt_tip_vessel_angle_deg",
        "curvature_local_1pm", "curvature_max_ahead_1pm", "curvature_mean_ahead_1pm",
        # Previous behaviour and SQP-probe-related features.
        "previous_adapt_pred_err_xy_mm", "error_delta_from_prev",
        "sqp_du_abs_first", "sqp_du_rel_first",
        "prev_final_to_current_first_abs_delta", "prev_final_to_current_first_abs_ratio",
        "prev_final_to_current_first_rel_delta", "prev_final_to_current_first_rel_ratio",
        # Final tolerance is not a geometry feature, but useful as benchmark.
        "sqp_du_abs_final", "sqp_du_rel_final",
    ]
    return [c for c in preferred if c in df.columns and to_num(df[c]).notna().sum() >= 10]


def feature_summary(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    y_delta = df.get("next_error_delta", pd.Series(np.nan, index=df.index))
    y_spike = df.get("is_next_error_spike_int", pd.Series(np.nan, index=df.index))
    y_err = df.get("adapt_pred_err_xy_mm", pd.Series(np.nan, index=df.index))
    y_rollout = df.get("rollout_steps", pd.Series(np.nan, index=df.index))

    rows = []
    for f in features:
        x = to_num(df[f])
        r_delta, p_delta, n_delta = spearman(x, y_delta)
        r_spike, p_spike, n_spike = spearman(x, y_spike)
        r_err, p_err, n_err = spearman(x, y_err)
        r_rollout, p_rollout, n_rollout = spearman(x, y_rollout)
        slope_delta, p_slope_delta, n_slope_delta = slope_ols_simple(x, y_delta)
        rows.append({
            "feature": f,
            "n_valid": int(x.notna().sum()),
            "median": x.median(skipna=True),
            "q25": x.quantile(0.25),
            "q75": x.quantile(0.75),
            "q90": x.quantile(0.90),
            "spearman_next_error_delta": r_delta,
            "p_next_error_delta": p_delta,
            "spearman_next_spike": r_spike,
            "p_next_spike": p_spike,
            "spearman_current_error": r_err,
            "p_current_error": p_err,
            "spearman_rollout": r_rollout,
            "p_rollout": p_rollout,
            "simple_slope_next_delta_per_feature_unit": slope_delta,
            "p_slope_next_delta": p_slope_delta,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # High ranking: feature correlates positively with spike/delta/error and is not merely rollout proxy.
    # Use absolute contribution but prefer positive risk orientation.
    out["screen_score"] = (
        out["spearman_next_spike"].fillna(0).clip(lower=0) * 2.0
        + out["spearman_next_error_delta"].fillna(0).clip(lower=0)
        + out["spearman_current_error"].fillna(0).clip(lower=0)
    )
    out = out.sort_values("screen_score", ascending=False).reset_index(drop=True)
    return out


def threshold_sweep(df: pd.DataFrame, features: List[str], min_events: int) -> pd.DataFrame:
    if "is_next_error_spike_int" not in df.columns:
        raise ValueError("Missing is_next_error_spike_int")
    y = to_num(df["is_next_error_spike_int"]).fillna(0).astype(int)
    y_delta = to_num(df.get("next_error_delta", pd.Series(np.nan, index=df.index)))
    y_err = to_num(df.get("adapt_pred_err_xy_mm", pd.Series(np.nan, index=df.index)))
    total_spikes = int(y.sum())
    total_nonspikes = int((1-y).sum())
    base_rate = float(y.mean()) if len(y) else np.nan

    rows = []
    quantiles = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for f in features:
        x = to_num(df[f])
        valid = x.notna() & y.notna()
        if valid.sum() < max(min_events, 20):
            continue
        thresholds = sorted(set(float(x.loc[valid].quantile(q)) for q in quantiles if pd.notna(x.loc[valid].quantile(q))))
        for thr in thresholds:
            flag = valid & (x >= thr)
            n_flag = int(flag.sum())
            n_not = int((valid & ~flag).sum())
            if n_flag < min_events or n_not < min_events:
                continue
            spikes_flag = int(y.loc[flag].sum())
            spikes_not = int(y.loc[valid & ~flag].sum())
            nonspikes_flag = n_flag - spikes_flag
            rate_flag = spikes_flag / n_flag if n_flag else np.nan
            rate_not = spikes_not / n_not if n_not else np.nan
            rr = rate_flag / rate_not if rate_not > 0 else np.inf
            recall = spikes_flag / total_spikes if total_spikes else np.nan
            fpr = nonspikes_flag / total_nonspikes if total_nonspikes else np.nan
            ci_lo, ci_hi = wilson_ci(spikes_flag, n_flag)
            rows.append({
                "feature": f,
                "threshold_high_risk_if_feature_ge": thr,
                "n_flagged": n_flag,
                "frac_flagged": n_flag / int(valid.sum()),
                "spike_rate_flagged": rate_flag,
                "spike_rate_unflagged": rate_not,
                "spike_rate_flagged_ci95_low": ci_lo,
                "spike_rate_flagged_ci95_high": ci_hi,
                "risk_ratio_flagged_vs_unflagged": rr,
                "precision": rate_flag,
                "recall": recall,
                "false_positive_rate": fpr,
                "mean_next_error_delta_flagged": y_delta.loc[flag].mean(skipna=True),
                "mean_next_error_delta_unflagged": y_delta.loc[valid & ~flag].mean(skipna=True),
                "mean_current_error_flagged": y_err.loc[flag].mean(skipna=True),
                "mean_current_error_unflagged": y_err.loc[valid & ~flag].mean(skipna=True),
                "base_spike_rate": base_rate,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["threshold_score"] = (
            out["risk_ratio_flagged_vs_unflagged"].replace(np.inf, np.nan).fillna(out["risk_ratio_flagged_vs_unflagged"].replace(np.inf, np.nan).max())
            * out["recall"].fillna(0)
            * np.sqrt(out["frac_flagged"].fillna(0))
        )
        out = out.sort_values(["threshold_score", "risk_ratio_flagged_vs_unflagged"], ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# Rollout-risk analysis by feature bin
# -----------------------------

def feature_bin_rollout_risk(df: pd.DataFrame, features: List[str], min_events: int) -> pd.DataFrame:
    rows = []
    if "rollout_steps" not in df.columns or "is_next_error_spike_int" not in df.columns:
        return pd.DataFrame()
    y = to_num(df["is_next_error_spike_int"]).fillna(0).astype(int)
    for f in features:
        x = to_num(df[f])
        valid = x.notna() & df["rollout_steps"].notna()
        if valid.sum() < min_events * 3:
            continue
        bins = qcut_labels(x.loc[valid], labels=["low", "medium", "high"])
        temp = df.loc[valid, ["rollout_steps"]].copy()
        temp["feature"] = f
        temp["feature_bin"] = bins.loc[valid].values
        temp["feature_value_mean"] = x.loc[valid].values
        temp["is_next_error_spike_int"] = y.loc[valid].values
        for col in ["adapt_pred_err_xy_mm", "next_error_delta", "pred1_err_xy_mm", "sqp_du_rel_final", "sqp_du_abs_final"]:
            if col in df.columns:
                temp[col] = to_num(df.loc[valid, col]).values
        g = temp.groupby(["feature", "feature_bin", "rollout_steps"], dropna=False)
        for keys, sub in g:
            feature, feature_bin, rollout = keys
            n = len(sub)
            if n < min_events:
                continue
            spikes = int(sub["is_next_error_spike_int"].sum())
            ci_lo, ci_hi = wilson_ci(spikes, n)
            row = {
                "feature": feature,
                "feature_bin": feature_bin,
                "rollout_steps": int(rollout) if pd.notna(rollout) else np.nan,
                "n_events": n,
                "spike_count": spikes,
                "spike_rate": spikes / n,
                "spike_rate_ci95_low": ci_lo,
                "spike_rate_ci95_high": ci_hi,
                "feature_value_mean": sub["feature_value_mean"].mean(skipna=True),
                "feature_value_median": sub["feature_value_mean"].median(skipna=True),
            }
            for col in ["adapt_pred_err_xy_mm", "next_error_delta", "pred1_err_xy_mm", "sqp_du_rel_final", "sqp_du_abs_final"]:
                if col in sub.columns:
                    row[f"mean_{col}"] = sub[col].mean(skipna=True)
                    row[f"median_{col}"] = sub[col].median(skipna=True)
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["feature", "feature_bin", "rollout_steps"]).reset_index(drop=True)
    return out


def candidate_rollout_by_feature_bin(risk_table: pd.DataFrame, target_spike_rate: float, min_events: int) -> pd.DataFrame:
    if risk_table.empty:
        return risk_table
    rows = []
    for (feature, fbin), sub in risk_table.groupby(["feature", "feature_bin"], dropna=False):
        sub = sub[sub["n_events"] >= min_events].copy()
        if sub.empty:
            continue
        safe = sub[sub["spike_rate"] <= target_spike_rate].sort_values("rollout_steps")
        if safe.empty:
            candidate = 1
            reason = f"no rollout bin met target_spike_rate<={target_spike_rate:.3f}; choose rollout 1 or reduce/damp"
            evidence_rollout = np.nan
            evidence_rate = np.nan
            evidence_n = 0
        else:
            best = safe.sort_values("rollout_steps", ascending=False).iloc[0]
            candidate = int(best["rollout_steps"])
            evidence_rollout = int(best["rollout_steps"])
            evidence_rate = float(best["spike_rate"])
            evidence_n = int(best["n_events"])
            reason = "largest rollout satisfying target spike rate"
        rows.append({
            "feature": feature,
            "feature_bin": fbin,
            "recommended_candidate_rollout": candidate,
            "evidence_rollout": evidence_rollout,
            "evidence_spike_rate": evidence_rate,
            "evidence_n_events": evidence_n,
            "target_spike_rate": target_spike_rate,
            "reason": reason,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Sort low/medium/high meaningfully.
        bin_order = {"low": 0, "medium": 1, "high": 2, "missing": 3}
        out["_bin_order"] = out["feature_bin"].map(bin_order).fillna(99)
        out = out.sort_values(["feature", "_bin_order"]).drop(columns=["_bin_order"]).reset_index(drop=True)
    return out


# -----------------------------
# Composite risk score
# -----------------------------

def percentile_rank(s: pd.Series) -> pd.Series:
    s = to_num(s)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    valid = s.notna()
    if valid.sum() == 0:
        return out
    out.loc[valid] = s.loc[valid].rank(pct=True)
    return out


def build_composite_score(df: pd.DataFrame, top_features: List[str], summary: pd.DataFrame, max_features: int = 5) -> Tuple[pd.DataFrame, List[str]]:
    # Only use features with positive association with spike/current error and avoid direct outcome/final-tolerance benchmark columns.
    disallow = {
        "sqp_du_abs_final", "sqp_du_rel_final", "adapt_pred_err_xy_mm", "next_error_delta",
        "previous_adapt_pred_err_xy_mm", "error_delta_from_prev",
    }
    chosen = []
    for f in top_features:
        if f in disallow:
            continue
        if f.startswith("sqp_du_") or f.startswith("prev_final"):
            # SQP-probe features are not geometry-only. Keep out of geometry composite.
            continue
        chosen.append(f)
        if len(chosen) >= max_features:
            break
    out = df.copy()
    if not chosen:
        out["geometry_conditioning_risk_score"] = np.nan
        return out, chosen

    score_parts = []
    weights = []
    for f in chosen:
        xrank = percentile_rank(out[f])
        score_parts.append(xrank)
        # Use screen_score as weight, fallback to 1.
        w = 1.0
        if summary is not None and not summary.empty and f in set(summary["feature"]):
            w = float(summary.loc[summary["feature"] == f, "screen_score"].iloc[0])
            if not np.isfinite(w) or w <= 0:
                w = 1.0
        weights.append(w)
    M = pd.concat(score_parts, axis=1)
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    out["geometry_conditioning_risk_score"] = M.mul(weights_arr, axis=1).sum(axis=1, skipna=True)
    out["geometry_conditioning_risk_bin"] = qcut_labels(out["geometry_conditioning_risk_score"], ["low", "medium", "high"])
    return out, chosen


def composite_policy(df: pd.DataFrame, target_spike_rate: float, min_events: int) -> pd.DataFrame:
    if "geometry_conditioning_risk_bin" not in df.columns or "rollout_steps" not in df.columns:
        return pd.DataFrame()
    rows = []
    g = df.groupby(["geometry_conditioning_risk_bin", "rollout_steps"], dropna=False)
    for (risk_bin, rollout), sub in g:
        n = len(sub)
        if n < min_events:
            continue
        y = to_num(sub["is_next_error_spike_int"]).fillna(0).astype(int)
        spikes = int(y.sum())
        ci_lo, ci_hi = wilson_ci(spikes, n)
        rows.append({
            "risk_bin": risk_bin,
            "rollout_steps": int(rollout),
            "n_events": n,
            "spike_rate": spikes/n,
            "spike_rate_ci95_low": ci_lo,
            "spike_rate_ci95_high": ci_hi,
            "mean_adapt_pred_err_xy_mm": to_num(sub.get("adapt_pred_err_xy_mm", pd.Series(np.nan, index=sub.index))).mean(skipna=True),
            "mean_next_error_delta": to_num(sub.get("next_error_delta", pd.Series(np.nan, index=sub.index))).mean(skipna=True),
        })
    risk_table = pd.DataFrame(rows)
    if risk_table.empty:
        return risk_table

    recs = []
    for risk_bin, sub in risk_table.groupby("risk_bin"):
        safe = sub[sub["spike_rate"] <= target_spike_rate]
        if safe.empty:
            rec = 1
            evidence = None
        else:
            evidence = safe.sort_values("rollout_steps", ascending=False).iloc[0]
            rec = int(evidence["rollout_steps"])
        recs.append({
            "risk_bin": risk_bin,
            "recommended_candidate_rollout": rec,
            "target_spike_rate": target_spike_rate,
            "evidence_rollout": np.nan if evidence is None else int(evidence["rollout_steps"]),
            "evidence_spike_rate": np.nan if evidence is None else float(evidence["spike_rate"]),
            "evidence_n_events": 0 if evidence is None else int(evidence["n_events"]),
        })
    return pd.merge(risk_table, pd.DataFrame(recs), on="risk_bin", how="left")


# -----------------------------
# Plotting/report
# -----------------------------

def save_plots(out_dir: Path, feature_summary_df: pd.DataFrame, risk_table: pd.DataFrame, threshold_df: pd.DataFrame, composite_df: pd.DataFrame, top_n: int = 8) -> None:
    if plt is None:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Bar plot of top feature screen scores.
    if not feature_summary_df.empty:
        top = feature_summary_df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(top))))
        ax.barh(top["feature"], top["screen_score"])
        ax.set_xlabel("Feature screening score")
        ax.set_title("Top candidate geometry/conditioning risk measures")
        fig.tight_layout()
        fig.savefig(plot_dir / "top_feature_screen_scores.png", dpi=180)
        plt.close(fig)

    # Heatmaps for top features: spike rate by feature bin x rollout.
    if not risk_table.empty and not feature_summary_df.empty:
        top_features = [f for f in feature_summary_df["feature"].head(min(top_n, len(feature_summary_df))).tolist() if f in set(risk_table["feature"])]
        for f in top_features:
            sub = risk_table[risk_table["feature"] == f].copy()
            if sub.empty:
                continue
            pivot = sub.pivot_table(index="feature_bin", columns="rollout_steps", values="spike_rate", aggfunc="mean")
            # Order bins if present.
            order = [b for b in ["low", "medium", "high"] if b in pivot.index]
            pivot = pivot.loc[order] if order else pivot
            fig, ax = plt.subplots(figsize=(7, 3.2))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(c)) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel("Rollout")
            ax.set_ylabel(f"{f} bin")
            ax.set_title(f"Next-spike rate by rollout and {f}")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f"{100*val:.1f}%", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="Spike rate")
            fig.tight_layout()
            safe_f = re.sub(r"[^A-Za-z0-9_\-]+", "_", f)
            fig.savefig(plot_dir / f"spike_rate_heatmap_{safe_f}.png", dpi=180)
            plt.close(fig)

    # Composite risk heatmap.
    if not composite_df.empty:
        piv = composite_df.pivot_table(index="risk_bin", columns="rollout_steps", values="spike_rate", aggfunc="mean")
        order = [b for b in ["low", "medium", "high"] if b in piv.index]
        piv = piv.loc[order] if order else piv
        fig, ax = plt.subplots(figsize=(7, 3.2))
        im = ax.imshow(piv.values, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(int(c)) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Composite risk bin")
        ax.set_title("Next-spike rate by rollout and composite risk")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{100*val:.1f}%", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Spike rate")
        fig.tight_layout()
        fig.savefig(plot_dir / "spike_rate_heatmap_composite_risk.png", dpi=180)
        plt.close(fig)

    # Threshold sweeps for top features.
    if not threshold_df.empty and not feature_summary_df.empty:
        top_features = [f for f in feature_summary_df["feature"].head(min(6, len(feature_summary_df))).tolist() if f in set(threshold_df["feature"])]
        for f in top_features:
            sub = threshold_df[threshold_df["feature"] == f].sort_values("threshold_high_risk_if_feature_ge")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sub["threshold_high_risk_if_feature_ge"], sub["spike_rate_flagged"], marker="o", label="flagged")
            ax.plot(sub["threshold_high_risk_if_feature_ge"], sub["spike_rate_unflagged"], marker="o", label="unflagged")
            ax.set_xlabel(f"{f} threshold")
            ax.set_ylabel("Next-spike rate")
            ax.set_title(f"Threshold sweep for {f}")
            ax.legend()
            fig.tight_layout()
            safe_f = re.sub(r"[^A-Za-z0-9_\-]+", "_", f)
            fig.savefig(plot_dir / f"threshold_sweep_{safe_f}.png", dpi=180)
            plt.close(fig)


def write_report(
    out_dir: Path,
    df: pd.DataFrame,
    feature_summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    risk_table: pd.DataFrame,
    candidate_df: pd.DataFrame,
    composite_features: List[str],
    composite_policy_df: pd.DataFrame,
    target_spike_rate: float,
) -> None:
    n = len(df)
    spikes = int(to_num(df.get("is_next_error_spike_int", pd.Series(0, index=df.index))).fillna(0).sum())
    spike_rate = spikes / n if n else np.nan
    lines = []
    lines.append("# Geometry/conditioning candidate-rollout analysis")
    lines.append("")
    lines.append(f"Events analysed: **{n}**")
    lines.append(f"Next-error spikes: **{spikes}** ({100*spike_rate:.2f}%)")
    lines.append(f"Target spike rate for candidate rollout: **{100*target_spike_rate:.1f}%**")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This analysis searches for online measures where larger values indicate that longer rollout becomes increasingly risky. The intended controller use is:")
    lines.append("")
    lines.append("```text")
    lines.append("geometry / conditioning measure -> candidate maximum rollout")
    lines.append("initial SQP probe              -> reduce rollout if mismatch is large")
    lines.append("final SQP tolerance            -> accept/reject the selected rollout")
    lines.append("```")
    lines.append("")

    lines.append("## Top candidate measures")
    lines.append("")
    if feature_summary_df.empty:
        lines.append("No usable feature summary was produced. Check that the input CSV contains conditioning/geometry columns.")
    else:
        cols = [
            "feature", "n_valid", "screen_score", "spearman_next_spike", "p_next_spike",
            "spearman_next_error_delta", "p_next_error_delta", "spearman_current_error", "p_current_error",
            "median", "q75", "q90",
        ]
        top = feature_summary_df[cols].head(15).copy()
        lines.append(top.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Best single-feature thresholds")
    lines.append("")
    if threshold_df.empty:
        lines.append("No threshold sweep was produced.")
    else:
        cols = [
            "feature", "threshold_high_risk_if_feature_ge", "n_flagged", "frac_flagged",
            "spike_rate_flagged", "spike_rate_unflagged", "risk_ratio_flagged_vs_unflagged",
            "recall", "false_positive_rate", "mean_current_error_flagged", "mean_current_error_unflagged",
        ]
        lines.append(threshold_df[cols].head(20).to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Candidate rollout by feature bin")
    lines.append("")
    if candidate_df.empty:
        lines.append("No candidate-rollout table was produced.")
    else:
        # Show only top-ranked features if possible.
        top_features = feature_summary_df["feature"].head(8).tolist() if not feature_summary_df.empty else candidate_df["feature"].unique().tolist()[:8]
        show = candidate_df[candidate_df["feature"].isin(top_features)].copy()
        lines.append(show.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Composite geometry/conditioning score")
    lines.append("")
    if composite_features:
        lines.append("Composite score used these features:")
        for f in composite_features:
            lines.append(f"- `{f}`")
    else:
        lines.append("No composite score was generated because no suitable geometry/conditioning features were found.")
    lines.append("")
    if not composite_policy_df.empty:
        lines.append(composite_policy_df.to_markdown(index=False, floatfmt=".4g"))
    lines.append("")

    lines.append("## Controller implication")
    lines.append("")
    lines.append("A good candidate-rollout feature should have: positive association with next-error spikes, clear high-vs-low threshold separation, and a heatmap where high feature bins become unsafe at lower rollout values. Beam and MPC conditioning features should be used as numerical/geometry trust measures, not final convergence measures. Final SQP residuals should still be used separately as acceptance tolerances.")
    lines.append("")
    lines.append("Recommended online structure:")
    lines.append("")
    lines.append("```python")
    lines.append("risk = geometry_conditioning_score")
    lines.append("candidate_rollout = rollout_policy_from_risk_bin(risk)")
    lines.append("sqp_probe = run_one_sqp_iteration(candidate_rollout)")
    lines.append("if sqp_probe.initial_abs_change is large:")
    lines.append("    candidate_rollout = min(candidate_rollout, 1 or 2)")
    lines.append("rel_tol, abs_tol = tolerance_for_rollout(candidate_rollout, risk)")
    lines.append("solve_until_tolerance_or_cap(rel_tol, abs_tol)")
    lines.append("if tolerance_not_met:")
    lines.append("    reduce_rollout_or_damp")
    lines.append("```")

    (out_dir / "geometry_rollout_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input", type=Path, help="Run root containing log.csv files, or an event CSV.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: <input>/geometry_rollout_measure_analysis or ./geometry_rollout_measure_analysis")
    ap.add_argument("--spike-error-mm", type=float, default=1.0, help="Next error delta threshold for spike label. Default 1.0 mm.")
    ap.add_argument("--target-spike-rate", type=float, default=0.05, help="Safe rollout target spike rate. Default 0.05.")
    ap.add_argument("--min-events", type=int, default=10, help="Minimum events per bin/threshold. Default 10.")
    ap.add_argument("--top-n-plots", type=int, default=8, help="Number of top features to plot. Default 8.")
    args = ap.parse_args()

    input_path = args.input
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif input_path.is_dir():
        out_dir = input_path / "geometry_rollout_measure_analysis"
    else:
        out_dir = Path("geometry_rollout_measure_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_input(input_path, spike_error_mm=args.spike_error_mm)
    df.to_csv(out_dir / "geometry_event_table.csv", index=False)

    features = candidate_feature_columns(df)
    if not features:
        raise RuntimeError("No candidate features found. Use raw log.csv root or an event CSV containing conditioning/geometry columns.")

    fs = feature_summary(df, features)
    fs.to_csv(out_dir / "feature_predictiveness_summary.csv", index=False)

    th = threshold_sweep(df, features, min_events=args.min_events)
    th.to_csv(out_dir / "feature_threshold_sweep.csv", index=False)

    rt = feature_bin_rollout_risk(df, features, min_events=args.min_events)
    rt.to_csv(out_dir / "feature_bin_rollout_risk.csv", index=False)

    cand = candidate_rollout_by_feature_bin(rt, target_spike_rate=args.target_spike_rate, min_events=args.min_events)
    cand.to_csv(out_dir / "candidate_rollout_by_feature_bin.csv", index=False)

    # Composite risk: use top-ranked non-SQP/non-final features.
    top_features = fs["feature"].tolist() if not fs.empty else []
    df_score, composite_features = build_composite_score(df, top_features, fs, max_features=5)
    df_score.to_csv(out_dir / "composite_score_events.csv", index=False)
    comp = composite_policy(df_score, target_spike_rate=args.target_spike_rate, min_events=args.min_events)
    comp.to_csv(out_dir / "composite_risk_rollout_policy.csv", index=False)

    save_plots(out_dir, fs, rt, th, comp, top_n=args.top_n_plots)
    write_report(out_dir, df_score, fs, th, rt, cand, composite_features, comp, target_spike_rate=args.target_spike_rate)

    print(f"Wrote analysis to: {out_dir}")
    print("Key outputs:")
    for name in [
        "geometry_rollout_report.md",
        "feature_predictiveness_summary.csv",
        "feature_threshold_sweep.csv",
        "feature_bin_rollout_risk.csv",
        "candidate_rollout_by_feature_bin.csv",
        "composite_risk_rollout_policy.csv",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
