#!/usr/bin/env python3
"""
Analyse how rollout length, local curvature, and SQP update diagnostics interact.

Designed for experiment folders such as:

    run_figure4_simulation_sqp_investigation2/
      bend_m90_jac_contact_plant_contact_1_sqp_full_rollout5_Np5_sqp6/
        log.csv

Core questions:
    1. Does local curvature increase next-step prediction-error spikes?
    2. Is the initial SQP update more predictive of the next error spike than
       the final SQP update?
    3. Does the jump from the previous optimisation problem's final SQP update
       to the current problem's first SQP update predict error spikes?
    4. For each curvature / SQP-risk regime, which rollout and SQP budget look
       empirically safest?

Important terminology used by this script:
    - sqp_*_first:
        first logged SQP update norm in the current optimisation problem,
        usually sqp_du_0 or sqp_du_rel_0.
    - sqp_*_final:
        final logged SQP update norm for the current optimisation problem,
        using sqp_du_final / sqp_du_rel_final.
    - final_inner_contraction:
        final update divided by the previous logged inner update. This tells
        whether the last SQP iteration still changed the solution materially.
    - prev_final_to_current_first:
        current first SQP update minus previous optimisation problem's final
        SQP update. This is the measured jump from the previous problem into
        the current problem.
    - current_final_to_next_first:
        next optimisation problem's first SQP update minus current final SQP
        update. This is a diagnostic for transitions but is not available until
        the next optimisation begins.

Outputs:
    <run_root>/rollout_sqp_curvature_spike_analysis/
      transition_events.csv
      signal_predictiveness_summary.csv
      spike_predictor_thresholds.csv
      curvature_rollout_sqp_summary.csv
      rollout_sqp_policy_evidence.csv
      recommended_rollout_sqp_by_curvature_risk.csv
      ols_next_error_delta_predictors.csv
      plots/
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Basic helpers
# ----------------------------

def parse_bend_angle_from_run_name(run_name: str) -> float:
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    return (-1.0 if m.group(1) == "m" else 1.0) * float(m.group(2))


def infer_jacobian_variant(run_name: str) -> str:
    if "jac_no_contact" in run_name:
        return "no_contact"
    if "jac_contact" in run_name:
        return "contact"
    return ""


def infer_solver_mode(run_name: str) -> str:
    if "sqp_full" in run_name:
        return "sqp_full"
    if "ltv_oneshot" in run_name:
        return "ltv_oneshot"
    if "lti" in run_name:
        return "lti"
    return ""


def parse_rollout_steps_from_run_name(run_name: str) -> float:
    m = re.search(r"rollout(\d+)", run_name)
    return float(m.group(1)) if m else np.nan


def parse_Np_from_run_name(run_name: str) -> float:
    m = re.search(r"_Np(\d+)", run_name)
    return float(m.group(1)) if m else np.nan


def parse_sqp_budget_from_run_name(run_name: str) -> float:
    m = re.search(r"_sqp(\d+)(?:$|_)", run_name)
    return float(m.group(1)) if m else np.nan


def bend_tag(bend: float) -> str:
    if not np.isfinite(bend):
        return "bend_unknown"
    return f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"


def safe_filename(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def safe_mean(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan


def safe_median(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(s.median()) if s.notna().any() else np.nan


def safe_max(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(s.max()) if s.notna().any() else np.nan


def safe_min(x) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(s.min()) if s.notna().any() else np.nan


def ci95(values: Iterable[float]) -> tuple[float, float, float, int]:
    s = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(s.mean())
    if n == 1:
        return mean, np.nan, np.nan, n
    se = float(s.std(ddof=1) / math.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


# ----------------------------
# SQP update extraction
# ----------------------------

def get_sqp_iter_cols(df: pd.DataFrame, *, rel: bool) -> list[str]:
    pat = r"sqp_du_rel_\d+" if rel else r"sqp_du_\d+"
    return sorted(
        [c for c in df.columns if re.fullmatch(pat, c)],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )


def sqp_iter_df(df: pd.DataFrame, *, rel: bool) -> pd.DataFrame:
    cols = get_sqp_iter_cols(df, rel=rel)
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce")


def sqp_update_diagnostics(df: pd.DataFrame, *, rel: bool) -> dict[str, pd.Series]:
    """Return row-wise diagnostics for absolute or relative SQP update norms."""
    kind = "rel" if rel else "abs"
    it = sqp_iter_df(df, rel=rel)

    if it.shape[1] > 0:
        first = it.iloc[:, 0]
        max_v = it.max(axis=1)
        min_v = it.min(axis=1)
        last_logged_inner = it.ffill(axis=1).iloc[:, -1]

        # Previous inner update before the final logged update. If there is only
        # one column, no previous inner update exists.
        if it.shape[1] >= 2:
            prev_logged_inner = it.ffill(axis=1).iloc[:, -2]
        else:
            prev_logged_inner = pd.Series(np.nan, index=df.index)
    else:
        first = pd.Series(np.nan, index=df.index)
        max_v = pd.Series(np.nan, index=df.index)
        min_v = pd.Series(np.nan, index=df.index)
        last_logged_inner = pd.Series(np.nan, index=df.index)
        prev_logged_inner = pd.Series(np.nan, index=df.index)

    final_col = "sqp_du_rel_final" if rel else "sqp_du_final"
    final_logged = to_num(df, final_col)
    final = final_logged.where(final_logged.notna(), last_logged_inner)

    # If final_logged is a true extra final value beyond the last per-iteration
    # column, compare last_logged_inner -> final. Otherwise compare previous -> final.
    prev_for_final = prev_logged_inner.copy()
    has_final_beyond_columns = final_logged.notna() & last_logged_inner.notna() & (abs(final_logged - last_logged_inner) > 1e-14)
    prev_for_final.loc[has_final_beyond_columns] = last_logged_inner.loc[has_final_beyond_columns]

    final_delta_from_prev = final - prev_for_final
    final_drop_from_prev = prev_for_final - final
    final_contraction_ratio = final / prev_for_final.replace(0.0, np.nan)

    first_to_final_drop = first - final
    first_to_final_ratio = final / first.replace(0.0, np.nan)

    return {
        f"sqp_du_{kind}_first": first,
        f"sqp_du_{kind}_final": final,
        f"sqp_du_{kind}_max": max_v,
        f"sqp_du_{kind}_min": min_v,
        f"sqp_du_{kind}_prev_inner_for_final": prev_for_final,
        f"sqp_du_{kind}_final_delta_from_prev_inner": final_delta_from_prev,
        f"sqp_du_{kind}_final_drop_from_prev_inner": final_drop_from_prev,
        f"sqp_du_{kind}_final_contraction_ratio_over_prev_inner": final_contraction_ratio,
        f"sqp_du_{kind}_drop_first_to_final": first_to_final_drop,
        f"sqp_du_{kind}_contraction_ratio_final_over_first": first_to_final_ratio,
    }


# ----------------------------
# Run collection and event table
# ----------------------------

def collect_runs(run_root: Path) -> list[dict]:
    run_root = Path(run_root)
    runs = []
    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name
        try:
            df = pd.read_csv(log_csv)
        except Exception as exc:
            print(f"[WARN] Could not read {log_csv}: {exc}")
            continue
        runs.append(
            {
                "run_root": run_root,
                "run_dir": run_dir,
                "run_name": run_name,
                "log_csv": log_csv,
                "df": df,
                "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
                "bend_magnitude_deg": abs(parse_bend_angle_from_run_name(run_name)),
                "jacobian_variant": infer_jacobian_variant(run_name),
                "solver_mode": infer_solver_mode(run_name),
                "rollout_steps": parse_rollout_steps_from_run_name(run_name),
                "Np": parse_Np_from_run_name(run_name),
                "sqp_budget": parse_sqp_budget_from_run_name(run_name),
            }
        )
    return runs


def build_transition_events(run_root: Path, *, out_root: Path, spike_error_mm: float) -> pd.DataFrame:
    rows = []

    for run in collect_runs(run_root):
        df = run["df"].copy().reset_index(drop=True)
        n = len(df)
        if n == 0:
            continue

        k_series = to_num(df, "k") if "k" in df.columns else pd.Series(np.arange(n), index=df.index)
        fallback_order = pd.Series(np.arange(n), index=df.index, dtype=float)
        order = np.argsort(k_series.where(k_series.notna(), fallback_order).to_numpy())
        df = df.iloc[order].reset_index(drop=True)

        rel = sqp_update_diagnostics(df, rel=True)
        absd = sqp_update_diagnostics(df, rel=False)
        diag = {**rel, **absd}

        err = to_num(df, "adapt_pred_err_xy_mm")
        pred1 = to_num(df, "pred1_err_xy_mm")
        err_delta = err.diff()
        next_err = err.shift(-1)
        next_err_delta = next_err - err

        rel_first = diag["sqp_du_rel_first"]
        rel_final = diag["sqp_du_rel_final"]
        abs_first = diag["sqp_du_abs_first"]
        abs_final = diag["sqp_du_abs_final"]

        # Transition from previous optimisation problem's final SQP update to
        # current optimisation problem's first SQP update. This is available at
        # the current solve after the first SQP iteration has occurred.
        prev_rel_final = rel_final.shift(1)
        prev_abs_final = abs_final.shift(1)
        prev_to_curr_rel_delta = rel_first - prev_rel_final
        prev_to_curr_abs_delta = abs_first - prev_abs_final
        prev_to_curr_rel_ratio = rel_first / prev_rel_final.replace(0.0, np.nan)
        prev_to_curr_abs_ratio = abs_first / prev_abs_final.replace(0.0, np.nan)

        # Diagnostic transition from current final SQP update to next problem's
        # first update. This is not known until the next optimisation begins,
        # but is useful for understanding why the next solve became difficult.
        next_rel_first = rel_first.shift(-1)
        next_abs_first = abs_first.shift(-1)
        curr_to_next_rel_delta = next_rel_first - rel_final
        curr_to_next_abs_delta = next_abs_first - abs_final
        curr_to_next_rel_ratio = next_rel_first / rel_final.replace(0.0, np.nan)
        curr_to_next_abs_ratio = next_abs_first / abs_final.replace(0.0, np.nan)

        for i in range(len(df)):
            row = {
                "run_name": run["run_name"],
                "log_csv": str(run["log_csv"]),
                "bend_angle_deg": run["bend_angle_deg"],
                "bend_magnitude_deg": run["bend_magnitude_deg"],
                "jacobian_variant": run["jacobian_variant"],
                "solver_mode": run["solver_mode"],
                "rollout_steps": run["rollout_steps"],
                "Np": run["Np"],
                "sqp_budget": run["sqp_budget"],
                "k": int(to_num(df, "k").iloc[i]) if "k" in df.columns and pd.notna(to_num(df, "k").iloc[i]) else i,
                "i_ref": float(to_num(df, "i_ref").iloc[i]) if pd.notna(to_num(df, "i_ref").iloc[i]) else np.nan,
                "idx_ref_terminal": float(to_num(df, "idx_ref_terminal").iloc[i]) if pd.notna(to_num(df, "idx_ref_terminal").iloc[i]) else np.nan,
                "path_idx_nearest": float(to_num(df, "path_idx_nearest").iloc[i]) if pd.notna(to_num(df, "path_idx_nearest").iloc[i]) else np.nan,
                "curvature_local_1pm": float(to_num(df, "curvature_local_1pm").iloc[i]) if pd.notna(to_num(df, "curvature_local_1pm").iloc[i]) else np.nan,
                "curvature_max_ahead_1pm": float(to_num(df, "curvature_max_ahead_1pm").iloc[i]) if pd.notna(to_num(df, "curvature_max_ahead_1pm").iloc[i]) else np.nan,
                "curvature_mean_ahead_1pm": float(to_num(df, "curvature_mean_ahead_1pm").iloc[i]) if pd.notna(to_num(df, "curvature_mean_ahead_1pm").iloc[i]) else np.nan,
                "curvature_lookahead_points": float(to_num(df, "curvature_lookahead_points").iloc[i]) if pd.notna(to_num(df, "curvature_lookahead_points").iloc[i]) else np.nan,
                "adapt_pred_err_xy_mm": float(err.iloc[i]) if pd.notna(err.iloc[i]) else np.nan,
                "pred1_err_xy_mm": float(pred1.iloc[i]) if pd.notna(pred1.iloc[i]) else np.nan,
                "error_delta_from_prev": float(err_delta.iloc[i]) if pd.notna(err_delta.iloc[i]) else np.nan,
                "next_adapt_pred_err_xy_mm": float(next_err.iloc[i]) if pd.notna(next_err.iloc[i]) else np.nan,
                "next_error_delta": float(next_err_delta.iloc[i]) if pd.notna(next_err_delta.iloc[i]) else np.nan,
                "is_error_spike": bool(err_delta.iloc[i] >= spike_error_mm) if pd.notna(err_delta.iloc[i]) else False,
                "is_next_error_spike": bool(next_err_delta.iloc[i] >= spike_error_mm) if pd.notna(next_err_delta.iloc[i]) else False,
                "sqp_iters_done": float(to_num(df, "sqp_iters_done").iloc[i]) if pd.notna(to_num(df, "sqp_iters_done").iloc[i]) else np.nan,
                "N_sqp_used": float(to_num(df, "N_sqp_used").iloc[i]) if pd.notna(to_num(df, "N_sqp_used").iloc[i]) else np.nan,
                "rollout_steps_used": float(to_num(df, "rollout_steps_used").iloc[i]) if pd.notna(to_num(df, "rollout_steps_used").iloc[i]) else np.nan,
                "infeasible": int(to_num(df, "infeasible").fillna(0).iloc[i]) if "infeasible" in df.columns else 0,
                "prev_final_to_current_first_rel_delta": float(prev_to_curr_rel_delta.iloc[i]) if pd.notna(prev_to_curr_rel_delta.iloc[i]) else np.nan,
                "prev_final_to_current_first_abs_delta": float(prev_to_curr_abs_delta.iloc[i]) if pd.notna(prev_to_curr_abs_delta.iloc[i]) else np.nan,
                "prev_final_to_current_first_rel_ratio": float(prev_to_curr_rel_ratio.iloc[i]) if pd.notna(prev_to_curr_rel_ratio.iloc[i]) else np.nan,
                "prev_final_to_current_first_abs_ratio": float(prev_to_curr_abs_ratio.iloc[i]) if pd.notna(prev_to_curr_abs_ratio.iloc[i]) else np.nan,
                "current_final_to_next_first_rel_delta": float(curr_to_next_rel_delta.iloc[i]) if pd.notna(curr_to_next_rel_delta.iloc[i]) else np.nan,
                "current_final_to_next_first_abs_delta": float(curr_to_next_abs_delta.iloc[i]) if pd.notna(curr_to_next_abs_delta.iloc[i]) else np.nan,
                "current_final_to_next_first_rel_ratio": float(curr_to_next_rel_ratio.iloc[i]) if pd.notna(curr_to_next_rel_ratio.iloc[i]) else np.nan,
                "current_final_to_next_first_abs_ratio": float(curr_to_next_abs_ratio.iloc[i]) if pd.notna(curr_to_next_abs_ratio.iloc[i]) else np.nan,
            }
            for name, series in diag.items():
                row[name] = float(series.iloc[i]) if pd.notna(series.iloc[i]) else np.nan
            rows.append(row)

    events = pd.DataFrame(rows)
    if len(events):
        events = events.sort_values(["bend_angle_deg", "rollout_steps", "sqp_budget", "run_name", "k"])
    events.to_csv(out_root / "transition_events.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'transition_events.csv'}")
    return events


# ----------------------------
# Risk classes and statistics
# ----------------------------

def add_risk_classes(
    events: pd.DataFrame,
    *,
    curvature_col: str,
    rel_warn: float,
    rel_high: float,
    abs_warn_quantile: float,
    abs_high_quantile: float,
) -> pd.DataFrame:
    e = events.copy()

    curv = pd.to_numeric(e[curvature_col], errors="coerce") if curvature_col in e.columns else pd.Series(np.nan, index=e.index)
    q1 = float(curv.quantile(1 / 3)) if curv.notna().any() else np.nan
    q2 = float(curv.quantile(2 / 3)) if curv.notna().any() else np.nan

    def curv_bin(v):
        if not np.isfinite(v):
            return "unknown_curvature"
        if v <= q1:
            return "low_curvature"
        if v <= q2:
            return "medium_curvature"
        return "high_curvature"

    e["curvature_bin"] = curv.map(curv_bin)
    e["curvature_bin_source"] = curvature_col
    e["curvature_low_tercile"] = q1
    e["curvature_high_tercile"] = q2

    abs_first = pd.to_numeric(e["sqp_du_abs_first"], errors="coerce") if "sqp_du_abs_first" in e.columns else pd.Series(np.nan, index=e.index)
    abs_warn = float(abs_first.quantile(abs_warn_quantile)) if abs_first.notna().any() else np.nan
    abs_high = float(abs_first.quantile(abs_high_quantile)) if abs_first.notna().any() else np.nan
    e["abs_first_warn_threshold"] = abs_warn
    e["abs_first_high_threshold"] = abs_high

    rel_first = pd.to_numeric(e["sqp_du_rel_first"], errors="coerce")

    def risk(row):
        rf = row.get("sqp_du_rel_first", np.nan)
        af = row.get("sqp_du_abs_first", np.nan)
        if not np.isfinite(rf) and not np.isfinite(af):
            return "unknown_initial_update"
        high_rel = np.isfinite(rf) and rf >= rel_high
        warn_rel = np.isfinite(rf) and rf >= rel_warn
        high_abs = np.isfinite(af) and np.isfinite(abs_high) and af >= abs_high
        warn_abs = np.isfinite(af) and np.isfinite(abs_warn) and af >= abs_warn
        if high_rel and high_abs:
            return "high_rel_and_abs_initial_change"
        if high_rel and not high_abs:
            return "high_rel_small_abs_initial_change"
        if warn_rel or warn_abs:
            return "moderate_initial_change"
        return "low_initial_change"

    e["initial_sqp_risk_class"] = e.apply(risk, axis=1)

    # Final convergence class.
    final_rel = pd.to_numeric(e.get("sqp_du_rel_final", np.nan), errors="coerce")
    e["final_sqp_convergence_class"] = np.where(
        final_rel <= rel_warn,
        "final_rel_low",
        np.where(final_rel <= rel_high, "final_rel_moderate", "final_rel_high"),
    )
    e.loc[final_rel.isna(), "final_sqp_convergence_class"] = "final_rel_unknown"

    return e


def summarize_group(g: pd.DataFrame) -> dict:
    next_delta = pd.to_numeric(g["next_error_delta"], errors="coerce")
    spike = g["is_next_error_spike"].astype(bool) if "is_next_error_spike" in g.columns else pd.Series(False, index=g.index)
    return {
        "n_events": int(len(g)),
        "mean_next_error_delta": safe_mean(next_delta),
        "median_next_error_delta": safe_median(next_delta),
        "max_next_error_delta": safe_max(next_delta),
        "next_spike_rate": float(spike.mean()) if len(spike) else np.nan,
        "mean_error": safe_mean(g.get("adapt_pred_err_xy_mm", np.nan)),
        "mean_pred1_error": safe_mean(g.get("pred1_err_xy_mm", np.nan)),
        "mean_curvature_local_1pm": safe_mean(g.get("curvature_local_1pm", np.nan)),
        "mean_curvature_max_ahead_1pm": safe_mean(g.get("curvature_max_ahead_1pm", np.nan)),
        "mean_sqp_du_rel_first": safe_mean(g.get("sqp_du_rel_first", np.nan)),
        "mean_sqp_du_rel_final": safe_mean(g.get("sqp_du_rel_final", np.nan)),
        "mean_sqp_du_abs_first": safe_mean(g.get("sqp_du_abs_first", np.nan)),
        "mean_sqp_du_abs_final": safe_mean(g.get("sqp_du_abs_final", np.nan)),
        "mean_prev_final_to_current_first_rel_delta": safe_mean(g.get("prev_final_to_current_first_rel_delta", np.nan)),
        "mean_prev_final_to_current_first_abs_delta": safe_mean(g.get("prev_final_to_current_first_abs_delta", np.nan)),
        "mean_current_final_to_next_first_rel_delta": safe_mean(g.get("current_final_to_next_first_rel_delta", np.nan)),
        "mean_current_final_to_next_first_abs_delta": safe_mean(g.get("current_final_to_next_first_abs_delta", np.nan)),
        "mean_final_rel_contraction_over_prev_inner": safe_mean(g.get("sqp_du_rel_final_contraction_ratio_over_prev_inner", np.nan)),
        "mean_final_abs_contraction_over_prev_inner": safe_mean(g.get("sqp_du_abs_final_contraction_ratio_over_prev_inner", np.nan)),
    }


def build_curvature_rollout_sqp_summary(events: pd.DataFrame, *, out_root: Path) -> pd.DataFrame:
    group_cols = [
        "bend_angle_deg",
        "curvature_bin",
        "initial_sqp_risk_class",
        "rollout_steps",
        "sqp_budget",
        "Np",
        "jacobian_variant",
        "solver_mode",
    ]
    rows = []
    for key, g in events.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, key))
        out.update(summarize_group(g))
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "curvature_rollout_sqp_summary.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'curvature_rollout_sqp_summary.csv'}")
    return df


def build_rollout_sqp_policy_evidence(
    events: pd.DataFrame,
    *,
    out_root: Path,
    min_events: int,
    spike_penalty_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each curvature/risk regime, compare rollout and SQP budget choices.
    Lower score is better:
        score = mean_next_error_delta + spike_penalty_mm * next_spike_rate
    """
    group_cols = ["curvature_bin", "initial_sqp_risk_class", "rollout_steps", "sqp_budget"]
    rows = []
    for key, g in events.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, key))
        out.update(summarize_group(g))
        out["policy_score_lower_better"] = out["mean_next_error_delta"] + spike_penalty_mm * out["next_spike_rate"]
        rows.append(out)
    evidence = pd.DataFrame(rows)
    evidence = evidence.sort_values(["curvature_bin", "initial_sqp_risk_class", "policy_score_lower_better"])
    evidence.to_csv(out_root / "rollout_sqp_policy_evidence.csv", index=False)

    rec_rows = []
    for key, g in evidence.groupby(["curvature_bin", "initial_sqp_risk_class"], dropna=False):
        gg = g[(g["n_events"] >= min_events) & pd.to_numeric(g["policy_score_lower_better"], errors="coerce").notna()].copy()
        if gg.empty:
            gg = g[pd.to_numeric(g["policy_score_lower_better"], errors="coerce").notna()].copy()
        if gg.empty:
            continue
        best = gg.sort_values(["policy_score_lower_better", "rollout_steps", "sqp_budget"]).iloc[0]
        out = dict(zip(["curvature_bin", "initial_sqp_risk_class"], key))
        out.update({
            "recommended_rollout_steps": float(best["rollout_steps"]),
            "recommended_sqp_budget": float(best["sqp_budget"]),
            "n_events_for_recommendation": int(best["n_events"]),
            "mean_next_error_delta": float(best["mean_next_error_delta"]),
            "next_spike_rate": float(best["next_spike_rate"]),
            "policy_score_lower_better": float(best["policy_score_lower_better"]),
            "mean_sqp_du_rel_first": float(best["mean_sqp_du_rel_first"]),
            "mean_sqp_du_rel_final": float(best["mean_sqp_du_rel_final"]),
            "reason": "lowest_mean_next_error_plus_spike_penalty_for_regime",
        })
        rec_rows.append(out)
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(out_root / "recommended_rollout_sqp_by_curvature_risk.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'rollout_sqp_policy_evidence.csv'}")
    print(f"[ANALYSIS] Wrote {out_root / 'recommended_rollout_sqp_by_curvature_risk.csv'}")
    return evidence, rec


# ----------------------------
# Predictiveness and thresholds
# ----------------------------

DEFAULT_SIGNALS = [
    "curvature_local_1pm",
    "curvature_max_ahead_1pm",
    "curvature_mean_ahead_1pm",
    "sqp_du_rel_first",
    "sqp_du_abs_first",
    "sqp_du_rel_final",
    "sqp_du_abs_final",
    "sqp_du_rel_final_contraction_ratio_over_prev_inner",
    "sqp_du_abs_final_contraction_ratio_over_prev_inner",
    "prev_final_to_current_first_rel_delta",
    "prev_final_to_current_first_abs_delta",
    "prev_final_to_current_first_rel_ratio",
    "prev_final_to_current_first_abs_ratio",
    "current_final_to_next_first_rel_delta",
    "current_final_to_next_first_abs_delta",
    "current_final_to_next_first_rel_ratio",
    "current_final_to_next_first_abs_ratio",
]


def build_signal_predictiveness(events: pd.DataFrame, *, out_root: Path) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(events["next_error_delta"], errors="coerce")
    spike = events["is_next_error_spike"].astype(bool)
    for signal in DEFAULT_SIGNALS:
        if signal not in events.columns:
            continue
        x = pd.to_numeric(events[signal], errors="coerce")
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        xv = x[valid]
        yv = y[valid]
        sv = spike[valid]
        pearson = float(xv.corr(yv, method="pearson")) if xv.nunique() > 1 else np.nan
        spearman = float(xv.corr(yv, method="spearman")) if xv.nunique() > 1 else np.nan
        mean_spike = safe_mean(xv[sv])
        mean_no_spike = safe_mean(xv[~sv])
        median_spike = safe_median(xv[sv])
        median_no_spike = safe_median(xv[~sv])
        rows.append({
            "signal": signal,
            "n_valid": int(valid.sum()),
            "pearson_corr_with_next_error_delta": pearson,
            "spearman_corr_with_next_error_delta": spearman,
            "mean_signal_when_next_spike": mean_spike,
            "mean_signal_when_no_next_spike": mean_no_spike,
            "median_signal_when_next_spike": median_spike,
            "median_signal_when_no_next_spike": median_no_spike,
            "mean_difference_spike_minus_no_spike": mean_spike - mean_no_spike if np.isfinite(mean_spike) and np.isfinite(mean_no_spike) else np.nan,
            "median_difference_spike_minus_no_spike": median_spike - median_no_spike if np.isfinite(median_spike) and np.isfinite(median_no_spike) else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("spearman_corr_with_next_error_delta", ascending=False, na_position="last")
    out.to_csv(out_root / "signal_predictiveness_summary.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'signal_predictiveness_summary.csv'}")
    return out


def build_spike_threshold_table(events: pd.DataFrame, *, out_root: Path, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    spike = events["is_next_error_spike"].astype(bool)
    base_spike_rate = float(spike.mean()) if len(spike) else np.nan
    for signal in DEFAULT_SIGNALS:
        if signal not in events.columns:
            continue
        x = pd.to_numeric(events[signal], errors="coerce")
        vals = x.dropna()
        if len(vals) < 5 or vals.nunique() < 2:
            continue
        thresholds = sorted(set(float(vals.quantile(q)) for q in quantiles if np.isfinite(vals.quantile(q))))
        for thr in thresholds:
            high = x >= thr
            valid = x.notna()
            if valid.sum() == 0:
                continue
            high = high & valid
            low = (~high) & valid
            tp = int((high & spike).sum())
            fp = int((high & ~spike).sum())
            fn = int((low & spike).sum())
            tn = int((low & ~spike).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            fpr = fp / (fp + tn) if (fp + tn) else np.nan
            f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
            rows.append({
                "signal": signal,
                "threshold": thr,
                "n_valid": int(valid.sum()),
                "n_high": int(high.sum()),
                "baseline_spike_rate": base_spike_rate,
                "precision_p_spike_given_high": precision,
                "recall_p_high_given_spike": recall,
                "false_positive_rate": fpr,
                "f1": f1,
                "mean_next_error_delta_high": safe_mean(events.loc[high, "next_error_delta"]),
                "mean_next_error_delta_low": safe_mean(events.loc[low, "next_error_delta"]),
                "mean_next_error_delta_difference_high_minus_low": safe_mean(events.loc[high, "next_error_delta"]) - safe_mean(events.loc[low, "next_error_delta"]),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["f1", "precision_p_spike_given_high"], ascending=False, na_position="last")
    out.to_csv(out_root / "spike_predictor_thresholds.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'spike_predictor_thresholds.csv'}")
    return out


# ----------------------------
# OLS predictor table
# ----------------------------

def build_ols_predictor_table(events: pd.DataFrame, *, out_root: Path) -> pd.DataFrame:
    """
    Standardized OLS for next_error_delta. This is not a causal model, but it
    ranks which signals explain next error changes after controlling for others.
    """
    candidate_features = [
        "bend_magnitude_deg",
        "rollout_steps",
        "sqp_budget",
        "curvature_local_1pm",
        "curvature_max_ahead_1pm",
        "sqp_du_rel_first",
        "sqp_du_abs_first",
        "sqp_du_rel_final",
        "sqp_du_abs_final",
        "sqp_du_rel_final_contraction_ratio_over_prev_inner",
        "sqp_du_abs_final_contraction_ratio_over_prev_inner",
        "prev_final_to_current_first_rel_delta",
        "prev_final_to_current_first_abs_delta",
    ]
    features = [f for f in candidate_features if f in events.columns]
    df = events[["next_error_delta"] + features].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    rows = []
    if len(df) <= len(features) + 3:
        out = pd.DataFrame(rows)
        out.to_csv(out_root / "ols_next_error_delta_predictors.csv", index=False)
        return out

    y = df["next_error_delta"].to_numpy(float)
    y_mean = y.mean()
    y_std = y.std(ddof=1)
    yz = (y - y_mean) / y_std if y_std > 1e-12 else y - y_mean

    X_cols = []
    X_parts = []
    for f in features:
        x = df[f].to_numpy(float)
        xs = x.std(ddof=1)
        if xs <= 1e-12:
            continue
        X_cols.append(f)
        X_parts.append((x - x.mean()) / xs)

    # Add interaction terms to test the hypothesis that curvature amplifies the
    # effect of initial SQP change.
    def add_interaction(a: str, b: str, name: str):
        if a in df.columns and b in df.columns:
            x = pd.to_numeric(df[a], errors="coerce").to_numpy(float)
            z = pd.to_numeric(df[b], errors="coerce").to_numpy(float)
            prod = x * z
            ps = prod.std(ddof=1)
            if ps > 1e-12:
                X_cols.append(name)
                X_parts.append((prod - prod.mean()) / ps)

    add_interaction("curvature_max_ahead_1pm", "sqp_du_rel_first", "curvature_max_x_initial_rel")
    add_interaction("curvature_max_ahead_1pm", "sqp_du_abs_first", "curvature_max_x_initial_abs")

    if not X_parts:
        out = pd.DataFrame(rows)
        out.to_csv(out_root / "ols_next_error_delta_predictors.csv", index=False)
        return out

    X = np.column_stack([np.ones(len(df))] + X_parts)
    beta, *_ = np.linalg.lstsq(X, yz, rcond=None)
    resid = yz - X @ beta
    dof = max(len(yz) - X.shape[1], 1)
    sigma2 = float((resid @ resid) / dof)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    for idx, name in enumerate(["intercept"] + X_cols):
        b = float(beta[idx])
        s = float(se[idx]) if idx < len(se) else np.nan
        t = b / s if np.isfinite(s) and s > 1e-12 else np.nan
        # Normal approximation to two-sided p-value.
        p_approx = math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan
        rows.append({
            "predictor": name,
            "standardized_beta": b,
            "standard_error": s,
            "t_stat_approx": t,
            "p_value_normal_approx": p_approx,
            "n_events": int(len(df)),
            "r_squared": float(1.0 - (resid @ resid) / np.sum((yz - yz.mean()) ** 2)) if len(yz) > 1 else np.nan,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("standardized_beta", ascending=False)
    out.to_csv(out_root / "ols_next_error_delta_predictors.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'ols_next_error_delta_predictors.csv'}")
    return out


# ----------------------------
# Plots
# ----------------------------

def plot_scatter_by_rollout(events: pd.DataFrame, *, x: str, y: str, out_dir: Path) -> None:
    if x not in events.columns or y not in events.columns:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = events[[x, y, "rollout_steps", "bend_angle_deg", "is_next_error_spike"]].copy()
    for c in [x, y, "rollout_steps", "bend_angle_deg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, y])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for rollout, g in df.groupby("rollout_steps"):
        ax.scatter(g[x], g[y], s=24, alpha=0.75, label=f"rollout={int(rollout)}")
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"scatter_{safe_filename(y)}_vs_{safe_filename(x)}.png", dpi=220)
    plt.close(fig)


def plot_metric_by_sqp_and_rollout(summary: pd.DataFrame, *, metric: str, out_dir: Path) -> None:
    if metric not in summary.columns:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    bends = sorted(summary["bend_angle_deg"].dropna().unique())
    for bend in bends:
        g_b = summary[summary["bend_angle_deg"] == bend]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        for rollout, g in g_b.groupby("rollout_steps"):
            gg = g.groupby("sqp_budget", as_index=False)[metric].mean().sort_values("sqp_budget")
            ax.plot(gg["sqp_budget"], gg[metric], marker="o", linewidth=2.0, label=f"rollout={int(rollout)}")
        ax.set_xlabel("SQP budget")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs SQP budget, bend {bend:.0f}°")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{safe_filename(metric)}_by_sqp_rollout_bend_{bend_tag(bend)}.png", dpi=220)
        plt.close(fig)


def make_plots(events: pd.DataFrame, summary: pd.DataFrame, *, out_root: Path) -> None:
    plot_root = out_root / "plots"
    for x in [
        "curvature_max_ahead_1pm",
        "sqp_du_rel_first",
        "sqp_du_abs_first",
        "sqp_du_rel_final",
        "sqp_du_abs_final",
        "prev_final_to_current_first_rel_delta",
        "prev_final_to_current_first_abs_delta",
        "current_final_to_next_first_rel_delta",
        "current_final_to_next_first_abs_delta",
    ]:
        plot_scatter_by_rollout(events, x=x, y="next_error_delta", out_dir=plot_root / "scatter_next_error")

    for metric in [
        "mean_next_error_delta",
        "next_spike_rate",
        "mean_sqp_du_rel_first",
        "mean_sqp_du_rel_final",
        "mean_prev_final_to_current_first_rel_delta",
        "mean_sqp_du_rel_final_contraction_ratio_over_prev_inner",
    ]:
        plot_metric_by_sqp_and_rollout(summary, metric=metric, out_dir=plot_root / "summary_by_sqp_rollout")
    print(f"[ANALYSIS] Wrote plots to {plot_root}")


# ----------------------------
# Report
# ----------------------------

def write_report(out_root: Path, events: pd.DataFrame, predictiveness: pd.DataFrame, thresholds: pd.DataFrame, rec: pd.DataFrame) -> None:
    path = out_root / "curvature_sqp_spike_report.md"
    lines = []
    lines.append("# Curvature × SQP update × rollout spike analysis\n\n")
    lines.append("## Interpretation of the new signals\n\n")
    lines.append("- `sqp_du_rel_first` / `sqp_du_abs_first`: first SQP update in the current optimisation problem; this is an initial mismatch/warm-start signal.\n")
    lines.append("- `sqp_du_rel_final` / `sqp_du_abs_final`: final SQP update; this is a convergence residual.\n")
    lines.append("- `sqp_du_*_final_contraction_ratio_over_prev_inner`: how much the final inner iteration contracted relative to the previous logged inner iteration. Values near 1 mean the last SQP iteration was still making a similar-sized update; small values mean it settled.\n")
    lines.append("- `prev_final_to_current_first_*`: jump from the previous optimisation problem's final update to the current optimisation problem's first update. This is the key transition signal you described.\n")
    lines.append("- `current_final_to_next_first_*`: jump from the current problem's final update to the next problem's first update. This is useful for post-hoc diagnosis, but it is only known when the next optimisation starts.\n\n")

    lines.append("## Controller-design reading\n\n")
    lines.append("Use the initial update and previous-to-current jump to choose rollout. Use the final update and final contraction to decide whether more SQP iterations are needed. If initial update predicts spikes more strongly than final update, then rollout should be controlled by the initial update rather than final residual.\n\n")

    if len(events):
        lines.append("## Dataset coverage\n\n")
        lines.append(f"- Events: **{len(events)}**\n")
        lines.append(f"- Next-spike rate: **{events['is_next_error_spike'].mean():.3f}**\n")
        lines.append(f"- Rollouts: `{sorted(events['rollout_steps'].dropna().unique())}`\n")
        lines.append(f"- SQP budgets: `{sorted(events['sqp_budget'].dropna().unique())}`\n\n")

    if len(predictiveness):
        lines.append("## Most positively associated signals with next error change\n\n")
        view = predictiveness.sort_values("spearman_corr_with_next_error_delta", ascending=False).head(10)
        cols = ["signal", "n_valid", "spearman_corr_with_next_error_delta", "mean_difference_spike_minus_no_spike"]
        lines.append(view[cols].to_markdown(index=False))
        lines.append("\n\n")

    if len(thresholds):
        lines.append("## Best threshold rules for next error spikes\n\n")
        view = thresholds.sort_values(["f1", "precision_p_spike_given_high"], ascending=False).head(10)
        cols = ["signal", "threshold", "precision_p_spike_given_high", "recall_p_high_given_spike", "f1", "mean_next_error_delta_difference_high_minus_low"]
        lines.append(view[cols].to_markdown(index=False))
        lines.append("\n\n")

    if len(rec):
        lines.append("## Recommended rollout/SQP by curvature and initial SQP risk\n\n")
        lines.append(rec.to_markdown(index=False))
        lines.append("\n\n")

    lines.append("## Main files\n\n")
    for f in [
        "transition_events.csv",
        "signal_predictiveness_summary.csv",
        "spike_predictor_thresholds.csv",
        "curvature_rollout_sqp_summary.csv",
        "rollout_sqp_policy_evidence.csv",
        "recommended_rollout_sqp_by_curvature_risk.csv",
        "ols_next_error_delta_predictors.csv",
        "plots/",
    ]:
        lines.append(f"- `{f}`\n")

    path.write_text("".join(lines))
    print(f"[ANALYSIS] Wrote {path}")


# ----------------------------
# Main
# ----------------------------

def run_analysis(
    run_root: Path,
    *,
    out_root: Path,
    spike_error_mm: float,
    curvature_col: str,
    rel_warn: float,
    rel_high: float,
    abs_warn_quantile: float,
    abs_high_quantile: float,
    min_events: int,
    spike_penalty_mm: float,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    events = build_transition_events(run_root, out_root=out_root, spike_error_mm=spike_error_mm)
    if events.empty:
        print("[ANALYSIS] No events found. Expected subdirectories containing log.csv")
        return

    events = add_risk_classes(
        events,
        curvature_col=curvature_col,
        rel_warn=rel_warn,
        rel_high=rel_high,
        abs_warn_quantile=abs_warn_quantile,
        abs_high_quantile=abs_high_quantile,
    )
    events.to_csv(out_root / "transition_events.csv", index=False)

    curv_summary = build_curvature_rollout_sqp_summary(events, out_root=out_root)
    evidence, rec = build_rollout_sqp_policy_evidence(
        events,
        out_root=out_root,
        min_events=min_events,
        spike_penalty_mm=spike_penalty_mm,
    )
    predictiveness = build_signal_predictiveness(events, out_root=out_root)
    thresholds = build_spike_threshold_table(events, out_root=out_root, quantiles=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95])
    ols = build_ols_predictor_table(events, out_root=out_root)

    make_plots(events, curv_summary, out_root=out_root)
    write_report(out_root, events, predictiveness, thresholds, rec)

    print("\n[ANALYSIS] Main outputs:")
    for name in [
        "transition_events.csv",
        "signal_predictiveness_summary.csv",
        "spike_predictor_thresholds.csv",
        "curvature_rollout_sqp_summary.csv",
        "rollout_sqp_policy_evidence.csv",
        "recommended_rollout_sqp_by_curvature_risk.csv",
        "ols_next_error_delta_predictors.csv",
        "curvature_sqp_spike_report.md",
        "plots",
    ]:
        print(f"  {out_root / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse curvature, SQP update transitions, rollout choice, and next error spikes.")
    parser.add_argument(
        "run_root",
        type=Path,
        nargs="?",
        default=Path("run_figure4_simulation_sqp_investigation2"),
        help="Root directory containing run subdirectories with log.csv files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_root>/rollout_sqp_curvature_spike_analysis.",
    )
    parser.add_argument("--spike-error-mm", type=float, default=1.0, help="Next-step error increase counted as a spike.")
    parser.add_argument(
        "--curvature-col",
        type=str,
        default="curvature_max_ahead_1pm",
        choices=["curvature_local_1pm", "curvature_max_ahead_1pm", "curvature_mean_ahead_1pm"],
        help="Curvature signal used to form low/medium/high curvature bins.",
    )
    parser.add_argument("--rel-warn", type=float, default=1.0, help="Moderate initial relative SQP update threshold.")
    parser.add_argument("--rel-high", type=float, default=2.0, help="High initial relative SQP update threshold.")
    parser.add_argument("--abs-warn-quantile", type=float, default=0.75, help="Quantile for moderate absolute initial update threshold.")
    parser.add_argument("--abs-high-quantile", type=float, default=0.90, help="Quantile for high absolute initial update threshold.")
    parser.add_argument("--min-events", type=int, default=5, help="Minimum events for a policy recommendation row.")
    parser.add_argument(
        "--spike-penalty-mm",
        type=float,
        default=2.0,
        help="Penalty added to policy score for spike rate. Score = mean_next_error_delta + penalty * spike_rate.",
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve() if args.out_root is not None else run_root / "rollout_sqp_curvature_spike_analysis"

    run_analysis(
        run_root,
        out_root=out_root,
        spike_error_mm=args.spike_error_mm,
        curvature_col=args.curvature_col,
        rel_warn=args.rel_warn,
        rel_high=args.rel_high,
        abs_warn_quantile=args.abs_warn_quantile,
        abs_high_quantile=args.abs_high_quantile,
        min_events=args.min_events,
        spike_penalty_mm=args.spike_penalty_mm,
    )


if __name__ == "__main__":
    main()
