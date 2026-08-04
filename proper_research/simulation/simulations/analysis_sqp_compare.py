#!/usr/bin/env python3
"""
Analyse the effect of increasing the SQP iteration budget inside the controller.

Expected run directory layout, e.g.

    run_figure4_simulation_sqp_investigation/
      bend_m90_jac_contact_plant_contact_1_sqp_full_rollout5_Np12_sqp1/
        log.csv
        experiment_config.json
        run_summary.json
      ...

The central experimental variable is parsed from the run directory suffix:

    _sqp1, _sqp2, ..., _sqp5

The script stratifies by:
    bend_angle_deg, jacobian_variant, solver_mode, rollout_steps, Np

and compares every SQP budget against sqp1 within each stratum.

Additional diagnostics test whether prediction-error spikes are associated with
large SQP relative updates (`sqp_du_rel_*`) and simulate a possible controller
change where the SQP loop stops on a relative-change tolerance instead of only
running to the configured max-iteration budget.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Parsing helpers
# ----------------------------

def parse_bend_angle_from_run_name(run_name: str) -> float:
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


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
    """
    Parse configured SQP iteration budget from the run name.

    This is intentionally different from the per-row log key `N_sqp_used`
    and from `sqp_iters_done`.
    """
    m = re.search(r"_sqp(\d+)(?:$|_)", run_name)
    return float(m.group(1)) if m else np.nan


def bend_tag(bend: float) -> str:
    if not np.isfinite(bend):
        return "bend_unknown"
    return f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


# ----------------------------
# Collection and summarisation
# ----------------------------

def collect_run_logs(run_root: Path) -> list[dict]:
    run_root = Path(run_root)
    runs = []

    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name
        try:
            df = pd.read_csv(log_csv)
        except Exception as e:
            print(f"[WARN] Failed to read {log_csv}: {e}")
            continue

        runs.append(
            {
                "run_root": run_root,
                "run_dir": run_dir,
                "run_name": run_name,
                "log_csv": log_csv,
                "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
                "jacobian_variant": infer_jacobian_variant(run_name),
                "solver_mode": infer_solver_mode(run_name),
                "rollout_steps": parse_rollout_steps_from_run_name(run_name),
                "Np": parse_Np_from_run_name(run_name),
                "sqp_budget": parse_sqp_budget_from_run_name(run_name),
                "df": df,
            }
        )

    return runs


def summarise_run(run: dict) -> dict:
    df = run["df"]

    infeasible = to_num(df, "infeasible").fillna(0).astype(int)
    reasons = df.get("rollout_adaptation_reason", pd.Series("", index=df.index)).fillna("")

    u_cols = [
        "u0_vx", "u0_vy", "u0_vz",
        "u0_wx", "u0_wy", "u0_wz",
        "u0_dL",
    ]
    existing_u_cols = [c for c in u_cols if c in df.columns]
    if existing_u_cols:
        U = df[existing_u_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        control_rms = float(np.sqrt(np.nanmean(U ** 2)))
        control_max_abs = float(np.nanmax(np.abs(U)))
    else:
        control_rms = np.nan
        control_max_abs = np.nan

    # Per-row SQP convergence diagnostics.
    sqp_du_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"sqp_du_\d+", c)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    sqp_du_rel_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"sqp_du_rel_\d+", c)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )

    final_du = to_num(df, "sqp_du_final")
    final_du_rel = to_num(df, "sqp_du_rel_final")

    row = {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),
        "run_root": str(run["run_root"]),

        "bend_angle_deg": run["bend_angle_deg"],
        "jacobian_variant": run["jacobian_variant"],
        "solver_mode": run["solver_mode"],
        "rollout_steps": run["rollout_steps"],
        "Np": run["Np"],
        "sqp_budget": run["sqp_budget"],

        "n_steps": int(len(df)),
        "success": bool((infeasible == 0).all()),
        "num_infeasible_steps": int(infeasible.sum()),
        "final_status": str(df["status"].iloc[-1]) if "status" in df.columns and len(df) else "",

        "final_i_ref": float(to_num(df, "i_ref").iloc[-1]) if len(df) else np.nan,
        "max_i_ref": float(to_num(df, "i_ref").max()),
        "final_p_now_L_mm": 1e3 * float(to_num(df, "p_now_L").iloc[-1]) if len(df) else np.nan,

        "mean_pred1_err_xy_mm": float(to_num(df, "pred1_err_xy_mm").mean()),
        "max_pred1_err_xy_mm": float(to_num(df, "pred1_err_xy_mm").max()),
        "mean_pred1_err_xyz_mm": float(to_num(df, "pred1_err_xyz_mm").mean()),
        "max_pred1_err_xyz_mm": float(to_num(df, "pred1_err_xyz_mm").max()),

        "mean_adapt_pred_err_xy_mm": float(to_num(df, "adapt_pred_err_xy_mm").mean()),
        "median_adapt_pred_err_xy_mm": float(to_num(df, "adapt_pred_err_xy_mm").median()),
        "max_adapt_pred_err_xy_mm": float(to_num(df, "adapt_pred_err_xy_mm").max()),

        "mean_adapt_pred_err_xyz_mm": float(to_num(df, "adapt_pred_err_xyz_mm").mean()),
        "max_adapt_pred_err_xyz_mm": float(to_num(df, "adapt_pred_err_xyz_mm").max()),

        "mean_clearance_mm": float(to_num(df, "adapt_clearance_mm").mean()),
        "min_clearance_mm": float(to_num(df, "adapt_clearance_mm").min()),

        "mean_tip_vessel_angle_deg": float(to_num(df, "adapt_tip_vessel_angle_deg").mean()),
        "max_tip_vessel_angle_deg": float(to_num(df, "adapt_tip_vessel_angle_deg").max()),

        "mean_cond_H_beam": float(to_num(df, "cond_H_beam").mean()),
        "median_cond_H_beam": float(to_num(df, "cond_H_beam").median()),
        "max_cond_H_beam": float(to_num(df, "cond_H_beam").max()),

        "mean_cond_H_mpc": float(to_num(df, "cond_H_mpc").mean()),
        "median_cond_H_mpc": float(to_num(df, "cond_H_mpc").median()),
        "max_cond_H_mpc": float(to_num(df, "cond_H_mpc").max()),

        "mean_mpc_eig_cond": float(to_num(df, "mpc_eig_cond").mean()),
        "max_mpc_eig_cond": float(to_num(df, "mpc_eig_cond").max()),

        "control_rms": control_rms,
        "control_max_abs": control_max_abs,
        "max_abs_u0_vx": float(to_num(df, "u0_vx").abs().max()),
        "max_abs_u0_vy": float(to_num(df, "u0_vy").abs().max()),
        "max_abs_u0_wz_rad_s": float(to_num(df, "u0_wz").abs().max()),
        "max_u0_dL": float(to_num(df, "u0_dL").max()),

        "mean_N_sqp_used": float(to_num(df, "N_sqp_used").mean()),
        "max_N_sqp_used": float(to_num(df, "N_sqp_used").max()),
        "mean_sqp_iters_done": float(to_num(df, "sqp_iters_done").mean()),
        "max_sqp_iters_done": float(to_num(df, "sqp_iters_done").max()),

        "mean_sqp_du_final": float(final_du.mean()),
        "median_sqp_du_final": float(final_du.median()),
        "max_sqp_du_final": float(final_du.max()),
        "mean_sqp_du_rel_final": float(final_du_rel.mean()),
        "median_sqp_du_rel_final": float(final_du_rel.median()),
        "max_sqp_du_rel_final": float(final_du_rel.max()),

        "num_good_increase": int(reasons.str.contains("good:increase", regex=False).sum()),
        "num_bad_prediction_error": int(reasons.str.contains("bad:prediction_error", regex=False).sum()),
        "num_bad_tip_angle": int(reasons.str.contains("bad:tip_angle", regex=False).sum()),
        "num_bad_low_clearance": int(reasons.str.contains("bad:low_clearance", regex=False).sum()),
    }

    # Mean per-SQP-iteration step sizes: useful to see diminishing returns.
    for col in sqp_du_cols:
        row[f"mean_{col}"] = float(to_num(df, col).mean())
        row[f"median_{col}"] = float(to_num(df, col).median())
    for col in sqp_du_rel_cols:
        row[f"mean_{col}"] = float(to_num(df, col).mean())
        row[f"median_{col}"] = float(to_num(df, col).median())

    # Relative-change diagnostics used to investigate replacing a fixed SQP
    # iteration budget with a convergence/stopping criterion.
    if sqp_du_rel_cols:
        rel_mat = df[sqp_du_rel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        row["mean_sqp_du_rel_any"] = float(np.nanmean(rel_mat))
        row["median_sqp_du_rel_any"] = float(np.nanmedian(rel_mat))
        row["max_sqp_du_rel_any"] = float(np.nanmax(rel_mat))
        row["p90_sqp_du_rel_any"] = float(np.nanpercentile(rel_mat, 90))

        first_rel = pd.to_numeric(df[sqp_du_rel_cols[0]], errors="coerce")
        last_rel = final_du_rel
        row["mean_sqp_rel_drop_first_to_final"] = float((first_rel - last_rel).mean())
        row["median_sqp_rel_drop_first_to_final"] = float((first_rel - last_rel).median())
        row["mean_sqp_rel_ratio_final_over_first"] = float((last_rel / first_rel.replace(0, np.nan)).mean())
    else:
        row["mean_sqp_du_rel_any"] = np.nan
        row["median_sqp_du_rel_any"] = np.nan
        row["max_sqp_du_rel_any"] = np.nan
        row["p90_sqp_du_rel_any"] = np.nan
        row["mean_sqp_rel_drop_first_to_final"] = np.nan
        row["median_sqp_rel_drop_first_to_final"] = np.nan
        row["mean_sqp_rel_ratio_final_over_first"] = np.nan

    err = to_num(df, "adapt_pred_err_xy_mm")
    err_delta = err.diff()
    row["max_positive_adapt_pred_err_xy_delta_mm"] = float(err_delta.clip(lower=0).max())
    row["mean_positive_adapt_pred_err_xy_delta_mm"] = float(err_delta.clip(lower=0).mean())

    return row


def build_sqp_summary(run_root: Path, out_root: Path) -> pd.DataFrame:
    runs = collect_run_logs(run_root)
    rows = []

    for run in runs:
        try:
            rows.append(summarise_run(run))
        except Exception as e:
            rows.append(
                {
                    "run_name": run["run_name"],
                    "run_dir": str(run["run_dir"]),
                    "bend_angle_deg": run["bend_angle_deg"],
                    "jacobian_variant": run["jacobian_variant"],
                    "solver_mode": run["solver_mode"],
                    "rollout_steps": run["rollout_steps"],
                    "Np": run["Np"],
                    "sqp_budget": run["sqp_budget"],
                    "analysis_success": False,
                    "analysis_error": repr(e),
                }
            )

    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(
            ["bend_angle_deg", "rollout_steps", "Np", "jacobian_variant", "solver_mode", "sqp_budget"]
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "sqp_summary.csv", index=False)
    with open(out_root / "sqp_summary.json", "w") as f:
        json.dump(summary.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    print(f"[ANALYSIS] Found {len(summary)} runs")
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_summary.csv'}")
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_summary.json'}")
    return summary


# ----------------------------
# Effect tables
# ----------------------------

DEFAULT_EFFECT_METRICS = [
    "success",
    "final_i_ref",
    "max_i_ref",
    "n_steps",
    "num_infeasible_steps",

    "mean_adapt_pred_err_xy_mm",
    "max_adapt_pred_err_xy_mm",
    "mean_adapt_pred_err_xyz_mm",
    "max_adapt_pred_err_xyz_mm",

    "min_clearance_mm",
    "mean_clearance_mm",
    "mean_tip_vessel_angle_deg",
    "max_tip_vessel_angle_deg",

    "mean_cond_H_beam",
    "max_cond_H_beam",
    "mean_cond_H_mpc",
    "max_cond_H_mpc",
    "mean_mpc_eig_cond",
    "max_mpc_eig_cond",

    "control_rms",
    "control_max_abs",

    "mean_sqp_iters_done",
    "max_sqp_iters_done",
    "mean_sqp_du_final",
    "median_sqp_du_final",
    "mean_sqp_du_rel_final",
    "median_sqp_du_rel_final",
    "mean_sqp_du_rel_any",
    "median_sqp_du_rel_any",
    "max_sqp_du_rel_any",
    "p90_sqp_du_rel_any",
    "mean_sqp_rel_drop_first_to_final",
    "median_sqp_rel_drop_first_to_final",
    "mean_sqp_rel_ratio_final_over_first",
    "max_positive_adapt_pred_err_xy_delta_mm",
    "mean_positive_adapt_pred_err_xy_delta_mm",

    "num_good_increase",
    "num_bad_prediction_error",
    "num_bad_tip_angle",
    "num_bad_low_clearance",
]


def build_relative_to_sqp1(
    summary: pd.DataFrame,
    *,
    out_root: Path,
    baseline_sqp: int = 1,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if metrics is None:
        metrics = DEFAULT_EFFECT_METRICS

    group_cols = ["bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np"]
    rows = []

    for key, g in summary.groupby(group_cols, dropna=False):
        g = g.sort_values("sqp_budget")
        base = g[g["sqp_budget"] == baseline_sqp]
        if base.empty:
            continue
        base_row = base.iloc[0]

        for _, row in g.iterrows():
            out = {col: row[col] for col in group_cols}
            out["run_name"] = row.get("run_name", "")
            out["sqp_budget"] = row["sqp_budget"]
            out["baseline_sqp_budget"] = baseline_sqp

            for metric in metrics:
                if metric not in summary.columns:
                    continue

                a = row[metric]
                b = base_row[metric]

                # Booleans are handled as ints in deltas.
                try:
                    a_num = float(a)
                    b_num = float(b)
                except Exception:
                    continue

                out[f"{metric}__value"] = a_num
                out[f"{metric}__baseline_sqp{baseline_sqp}"] = b_num
                out[f"{metric}__abs_delta_vs_sqp{baseline_sqp}"] = a_num - b_num
                out[f"{metric}__pct_delta_vs_sqp{baseline_sqp}"] = (
                    100.0 * (a_num - b_num) / abs(b_num)
                    if np.isfinite(b_num) and abs(b_num) > 1e-12
                    else np.nan
                )

            rows.append(out)

    rel = pd.DataFrame(rows)
    rel.to_csv(out_root / f"sqp_effect_relative_to_sqp{baseline_sqp}.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / f'sqp_effect_relative_to_sqp{baseline_sqp}.csv'}")
    return rel


def build_trend_summary(
    summary: pd.DataFrame,
    *,
    out_root: Path,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    if metrics is None:
        metrics = DEFAULT_EFFECT_METRICS

    group_cols = ["bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np"]
    rows = []

    for key, g in summary.groupby(group_cols, dropna=False):
        g = g.sort_values("sqp_budget")
        x = pd.to_numeric(g["sqp_budget"], errors="coerce")

        for metric in metrics:
            if metric not in g.columns:
                continue

            y = pd.to_numeric(g[metric], errors="coerce")
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

            if valid.sum() < 2:
                continue

            xv = x[valid].to_numpy(float)
            yv = y[valid].to_numpy(float)

            try:
                slope = float(np.polyfit(xv, yv, 1)[0])
            except Exception:
                slope = np.nan

            spearman = pd.Series(xv).corr(pd.Series(yv), method="spearman")

            out = dict(zip(group_cols, key))
            out.update(
                {
                    "metric": metric,
                    "n_points": int(valid.sum()),
                    "slope_per_sqp_iter": slope,
                    "spearman_r": float(spearman) if pd.notna(spearman) else np.nan,
                    "value_at_min_sqp": float(yv[np.argmin(xv)]),
                    "value_at_max_sqp": float(yv[np.argmax(xv)]),
                    "delta_max_minus_min_sqp": float(yv[np.argmax(xv)] - yv[np.argmin(xv)]),
                }
            )
            rows.append(out)

    trends = pd.DataFrame(rows)
    trends.to_csv(out_root / "sqp_trend_summary.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_trend_summary.csv'}")
    return trends


def build_incremental_gain_table(
    summary: pd.DataFrame,
    *,
    out_root: Path,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Computes adjacent changes sqp1->2, 2->3, ... within each matched stratum.
    This is the most direct table for diminishing returns.
    """
    if metrics is None:
        metrics = DEFAULT_EFFECT_METRICS

    group_cols = ["bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np"]
    rows = []

    for key, g in summary.groupby(group_cols, dropna=False):
        g = g.sort_values("sqp_budget").reset_index(drop=True)
        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            curr = g.iloc[i]
            out = dict(zip(group_cols, key))
            out["from_sqp"] = prev["sqp_budget"]
            out["to_sqp"] = curr["sqp_budget"]

            for metric in metrics:
                if metric not in g.columns:
                    continue
                try:
                    a = float(curr[metric])
                    b = float(prev[metric])
                except Exception:
                    continue
                out[f"{metric}__adjacent_abs_delta"] = a - b
                out[f"{metric}__adjacent_pct_delta"] = (
                    100.0 * (a - b) / abs(b)
                    if np.isfinite(b) and abs(b) > 1e-12
                    else np.nan
                )

            rows.append(out)

    inc = pd.DataFrame(rows)
    inc.to_csv(out_root / "sqp_incremental_gains.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_incremental_gains.csv'}")
    return inc


# ----------------------------
# Plotting
# ----------------------------

PRETTY = {
    "contact": "Contact-aware Jacobian",
    "no_contact": "No-contact Jacobian",
    "sqp_full": "SQP full",

    "final_i_ref": "Final reference index",
    "max_i_ref": "Maximum reference index",
    "n_steps": "Number of control steps",
    "num_infeasible_steps": "Infeasible steps",

    "mean_adapt_pred_err_xy_mm": "Mean prediction error XY [mm]",
    "max_adapt_pred_err_xy_mm": "Max prediction error XY [mm]",
    "mean_adapt_pred_err_xyz_mm": "Mean prediction error XYZ [mm]",
    "max_adapt_pred_err_xyz_mm": "Max prediction error XYZ [mm]",

    "min_clearance_mm": "Minimum clearance [mm]",
    "mean_clearance_mm": "Mean clearance [mm]",
    "mean_tip_vessel_angle_deg": "Mean tip-vessel angle [deg]",
    "max_tip_vessel_angle_deg": "Max tip-vessel angle [deg]",

    "mean_cond_H_beam": "Mean beam Hessian condition",
    "max_cond_H_beam": "Max beam Hessian condition",
    "mean_cond_H_mpc": "Mean MPC Hessian condition",
    "max_cond_H_mpc": "Max MPC Hessian condition",
    "mean_mpc_eig_cond": "Mean MPC eig. condition",
    "max_mpc_eig_cond": "Max MPC eig. condition",

    "control_rms": "Control RMS",
    "control_max_abs": "Max absolute control",

    "mean_sqp_iters_done": "Mean SQP iterations done",
    "max_sqp_iters_done": "Max SQP iterations done",
    "mean_sqp_du_final": "Mean final SQP update norm",
    "median_sqp_du_final": "Median final SQP update norm",
    "mean_sqp_du_rel_final": "Mean final relative SQP update",
    "median_sqp_du_rel_final": "Median final relative SQP update",
    "mean_sqp_du_rel_any": "Mean relative SQP update, all inner iterations",
    "median_sqp_du_rel_any": "Median relative SQP update, all inner iterations",
    "max_sqp_du_rel_any": "Max relative SQP update, all inner iterations",
    "p90_sqp_du_rel_any": "90th percentile relative SQP update",
    "mean_sqp_rel_drop_first_to_final": "Mean relative-update drop, first to final",
    "median_sqp_rel_drop_first_to_final": "Median relative-update drop, first to final",
    "mean_sqp_rel_ratio_final_over_first": "Mean final/first relative-update ratio",
    "max_positive_adapt_pred_err_xy_delta_mm": "Max positive error jump XY [mm]",
    "mean_positive_adapt_pred_err_xy_delta_mm": "Mean positive error jump XY [mm]",
}


def metric_label(metric: str) -> str:
    return PRETTY.get(metric, metric)


def should_log_y(metric: str) -> bool:
    return (
        "cond" in metric
        or "du_final" in metric
        or "du_rel_final" in metric
        or metric in {"mean_sqp_du_final", "median_sqp_du_final"}
    )


def plot_metric_vs_sqp_by_rollout(
    summary: pd.DataFrame,
    metric: str,
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if metric not in summary.columns:
        return

    bends = sorted(summary["bend_angle_deg"].dropna().unique())
    rollouts = sorted(summary["rollout_steps"].dropna().unique())

    for bend in bends:
        fig, axes = plt.subplots(
            1,
            len(rollouts),
            figsize=(5.2 * len(rollouts), 4.2),
            sharey=True,
        )
        if len(rollouts) == 1:
            axes = [axes]

        for ax, rollout in zip(axes, rollouts):
            g = summary[
                (summary["bend_angle_deg"] == bend)
                & (summary["rollout_steps"] == rollout)
            ].copy()

            if g.empty:
                continue

            # Usually one jacobian/solver/Np, but keep this robust.
            for label_key, gg in g.groupby(["jacobian_variant", "solver_mode", "Np"], dropna=False):
                gg = gg.sort_values("sqp_budget")
                label = (
                    f"{PRETTY.get(label_key[0], label_key[0])}, "
                    f"{PRETTY.get(label_key[1], label_key[1])}, "
                    f"Np={int(label_key[2]) if np.isfinite(label_key[2]) else label_key[2]}"
                )
                ax.plot(
                    gg["sqp_budget"],
                    pd.to_numeric(gg[metric], errors="coerce"),
                    marker="o",
                    linewidth=2.2,
                    label=label,
                )

                failed = ~gg["success"].astype(bool) if "success" in gg.columns else pd.Series(False, index=gg.index)
                if failed.any():
                    ax.scatter(
                        gg.loc[failed, "sqp_budget"],
                        gg.loc[failed, metric],
                        marker="x",
                        s=90,
                        linewidths=2,
                        zorder=10,
                        label="failed" if ax.get_legend_handles_labels()[1].count("failed") == 0 else None,
                    )

            ax.set_title(f"rollout={int(rollout)}")
            ax.set_xlabel("Configured SQP iteration budget")
            ax.set_xticks(sorted(summary["sqp_budget"].dropna().unique()))
            ax.grid(True, alpha=0.3)

            if should_log_y(metric):
                ax.set_yscale("log")

        axes[0].set_ylabel(metric_label(metric))
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"{metric_label(metric)} vs SQP budget, bend {bend:.0f}°")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fig.savefig(out_dir / f"{safe_filename(metric)}_vs_sqp_bend_{bend_tag(bend)}.png", dpi=220)
        plt.close(fig)


def plot_relative_metric_vs_sqp(
    rel: pd.DataFrame,
    metric: str,
    *,
    out_dir: Path,
    baseline_sqp: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    col = f"{metric}__pct_delta_vs_sqp{baseline_sqp}"
    if col not in rel.columns:
        return

    bends = sorted(rel["bend_angle_deg"].dropna().unique())
    rollouts = sorted(rel["rollout_steps"].dropna().unique())

    for bend in bends:
        fig, axes = plt.subplots(
            1,
            len(rollouts),
            figsize=(5.2 * len(rollouts), 4.2),
            sharey=True,
        )
        if len(rollouts) == 1:
            axes = [axes]

        for ax, rollout in zip(axes, rollouts):
            g = rel[
                (rel["bend_angle_deg"] == bend)
                & (rel["rollout_steps"] == rollout)
            ].copy()

            if g.empty:
                continue

            for label_key, gg in g.groupby(["jacobian_variant", "solver_mode", "Np"], dropna=False):
                gg = gg.sort_values("sqp_budget")
                label = (
                    f"{PRETTY.get(label_key[0], label_key[0])}, "
                    f"{PRETTY.get(label_key[1], label_key[1])}, "
                    f"Np={int(label_key[2]) if np.isfinite(label_key[2]) else label_key[2]}"
                )
                ax.plot(
                    gg["sqp_budget"],
                    gg[col],
                    marker="o",
                    linewidth=2.2,
                    label=label,
                )

            ax.axhline(0.0, linestyle=":", linewidth=1.2)
            ax.set_title(f"rollout={int(rollout)}")
            ax.set_xlabel("Configured SQP iteration budget")
            ax.set_xticks(sorted(rel["sqp_budget"].dropna().unique()))
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(f"% change vs sqp{baseline_sqp}")
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"Relative change in {metric_label(metric)}, bend {bend:.0f}°")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fig.savefig(
            out_dir / f"{safe_filename(metric)}_pct_delta_vs_sqp{baseline_sqp}_bend_{bend_tag(bend)}.png",
            dpi=220,
        )
        plt.close(fig)


def plot_reference_traces_by_sqp(
    run_root: Path,
    metric: str,
    *,
    out_dir: Path,
    x_col: str = "i_ref",
) -> None:
    runs = collect_run_logs(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted({r["bend_angle_deg"] for r in runs if np.isfinite(r["bend_angle_deg"])})
    rollouts = sorted({r["rollout_steps"] for r in runs if np.isfinite(r["rollout_steps"])})

    for bend in bends:
        for rollout in rollouts:
            matching = [
                r for r in runs
                if r["bend_angle_deg"] == bend
                and r["rollout_steps"] == rollout
            ]
            if not matching:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 4.8))

            for r in sorted(matching, key=lambda z: z["sqp_budget"]):
                df = r["df"]
                if x_col not in df.columns or metric not in df.columns:
                    continue

                x = to_num(df, x_col)
                y = to_num(df, metric)
                valid = x.notna() & y.notna() & np.isfinite(y)

                ax.plot(
                    x[valid],
                    y[valid],
                    marker="o",
                    markersize=3.4,
                    linewidth=1.8,
                    label=f"sqp{int(r['sqp_budget'])}",
                )

                if "infeasible" in df.columns:
                    infeas = to_num(df, "infeasible").fillna(0).astype(int) > 0
                    if infeas.any():
                        ax.scatter(
                            x[infeas],
                            y[infeas],
                            marker="x",
                            s=80,
                            linewidths=2.0,
                            zorder=10,
                        )

            if should_log_y(metric):
                ax.set_yscale("log")

            ax.set_title(f"{metric_label(metric)} trace, bend {bend:.0f}°, rollout={int(rollout)}")
            ax.set_xlabel("Reference index along trajectory")
            ax.set_ylabel(metric_label(metric))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()

            fig.savefig(
                out_dir / f"{safe_filename(metric)}_trace_bend_{bend_tag(bend)}_rollout_{int(rollout)}.png",
                dpi=220,
            )
            plt.close(fig)


def plot_sqp_iteration_decay(run_root: Path, *, out_dir: Path) -> None:
    """
    Plot mean update norm per SQP inner iteration for each configured SQP budget.
    This directly visualises diminishing returns inside the SQP loop.
    """
    runs = collect_run_logs(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted({r["bend_angle_deg"] for r in runs if np.isfinite(r["bend_angle_deg"])})
    rollouts = sorted({r["rollout_steps"] for r in runs if np.isfinite(r["rollout_steps"])})

    for bend in bends:
        for rollout in rollouts:
            matching = [
                r for r in runs
                if r["bend_angle_deg"] == bend
                and r["rollout_steps"] == rollout
            ]
            if not matching:
                continue

            fig, ax = plt.subplots(figsize=(7.2, 4.4))

            for r in sorted(matching, key=lambda z: z["sqp_budget"]):
                df = r["df"]
                du_cols = sorted(
                    [c for c in df.columns if re.fullmatch(r"sqp_du_\d+", c)],
                    key=lambda x: int(x.rsplit("_", 1)[1]),
                )
                if not du_cols:
                    continue

                xs = []
                ys = []
                for col in du_cols:
                    idx = int(col.rsplit("_", 1)[1]) + 1
                    if idx > int(r["sqp_budget"]):
                        continue
                    xs.append(idx)
                    ys.append(float(to_num(df, col).median()))

                if xs:
                    ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"sqp{int(r['sqp_budget'])}")

            ax.set_yscale("log")
            ax.set_xlabel("SQP inner iteration")
            ax.set_ylabel("Median SQP update norm")
            ax.set_title(f"SQP update decay, bend {bend:.0f}°, rollout={int(rollout)}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()

            fig.savefig(
                out_dir / f"sqp_update_decay_bend_{bend_tag(bend)}_rollout_{int(rollout)}.png",
                dpi=220,
            )
            plt.close(fig)



# ----------------------------
# Relative-change stopping and spike diagnostics
# ----------------------------

def parse_float_list(s: str | Iterable[float]) -> list[float]:
    if isinstance(s, str):
        vals = []
        for part in s.split(','):
            part = part.strip()
            if part:
                vals.append(float(part))
        return vals
    return [float(x) for x in s]


def get_sqp_du_rel_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if re.fullmatch(r"sqp_du_rel_\d+", c)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )


def predict_stop_iter_from_relative_change(row: pd.Series, rel_cols: list[str], *, rel_tol: float) -> float:
    """Return 1-based SQP iteration count for earliest sqp_du_rel_i <= rel_tol.

    If no logged inner iteration satisfies the tolerance, return NaN. The caller can
    fill NaN with the configured/current budget if they want a max-iteration cap.
    """
    for col in rel_cols:
        val = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
        if pd.notna(val) and np.isfinite(val) and val <= rel_tol:
            return float(int(col.rsplit("_", 1)[1]) + 1)
    return np.nan


def build_relative_change_event_table(
    run_root: Path,
    *,
    out_root: Path,
    error_metric: str = "adapt_pred_err_xy_mm",
    spike_delta_mm: float = 1.0,
    spike_factor: float = 2.0,
) -> pd.DataFrame:
    """One row per control step, with current/next error spikes and SQP relative-change features.

    This is the main table for asking whether large relative SQP changes are
    associated with immediate or next-step prediction-error spikes.
    """
    rows = []
    for run in collect_run_logs(run_root):
        df = run["df"].copy()
        if df.empty:
            continue

        rel_cols = get_sqp_du_rel_cols(df)
        err = to_num(df, error_metric)
        i_ref = to_num(df, "i_ref")
        step = np.arange(len(df))

        current_delta = err.diff()
        next_err = err.shift(-1)
        next_delta = err.shift(-1) - err
        prev_safe = err.shift(1).replace(0, np.nan)
        current_factor = err / prev_safe
        next_factor = err.shift(-1) / err.replace(0, np.nan)

        if rel_cols:
            rel_df = df[rel_cols].apply(pd.to_numeric, errors="coerce")
            rel_max = rel_df.max(axis=1)
            rel_min = rel_df.min(axis=1)
            rel_first = rel_df.iloc[:, 0]
            rel_final = to_num(df, "sqp_du_rel_final")
            rel_drop = rel_first - rel_final
            rel_ratio = rel_final / rel_first.replace(0, np.nan)
            rel_df = df[rel_cols].apply(pd.to_numeric, errors="coerce")

            # pandas idxmax(axis=1) raises ValueError when a row is all-NaN.
            # Make it safe by only applying idxmax to rows with at least one finite value.
            has_any_rel = rel_df.notna().any(axis=1)

            dominant_idx = pd.Series(np.nan, index=df.index, dtype=float)

            if has_any_rel.any():
                dominant_cols = rel_df.loc[has_any_rel].idxmax(axis=1)
                dominant_idx.loc[has_any_rel] = dominant_cols.map(
                    lambda c: int(c.rsplit("_", 1)[1]) + 1 if isinstance(c, str) else np.nan
                ).astype(float)
        else:
            rel_max = rel_min = rel_first = rel_final = rel_drop = rel_ratio = pd.Series(np.nan, index=df.index)
            dominant_idx = pd.Series(np.nan, index=df.index)
        du_cols = sorted(
            [c for c in df.columns if re.fullmatch(r"sqp_du_\d+", c)],
            key=lambda x: int(x.rsplit("_", 1)[1]),
        )

        du_df = df[du_cols].apply(pd.to_numeric, errors="coerce") if du_cols else pd.DataFrame(index=df.index)

        if len(du_cols):
            du_first = du_df.iloc[:, 0]
            du_max = du_df.max(axis=1)
            du_min = du_df.min(axis=1)
            du_final_logged = to_num(df, "sqp_du_final")
            du_final_from_cols = du_df.ffill(axis=1).iloc[:, -1]

            du_final = du_final_logged.where(du_final_logged.notna(), du_final_from_cols)
        else:
            du_first = pd.Series(np.nan, index=df.index)
            du_max = pd.Series(np.nan, index=df.index)
            du_min = pd.Series(np.nan, index=df.index)
            du_final = pd.Series(np.nan, index=df.index)
        for j in range(len(df)):
            rows.append({
                "run_name": run["run_name"],
                "run_dir": str(run["run_dir"]),
                "bend_angle_deg": run["bend_angle_deg"],
                "jacobian_variant": run["jacobian_variant"],
                "solver_mode": run["solver_mode"],
                "rollout_steps": run["rollout_steps"],
                "Np": run["Np"],
                "sqp_budget": run["sqp_budget"],
                "step_idx": int(step[j]),
                "i_ref": float(i_ref.iloc[j]) if pd.notna(i_ref.iloc[j]) else np.nan,
                "error_metric": error_metric,
                "error_value": float(err.iloc[j]) if pd.notna(err.iloc[j]) else np.nan,
                "error_delta_from_prev": float(current_delta.iloc[j]) if pd.notna(current_delta.iloc[j]) else np.nan,
                "error_factor_from_prev": float(current_factor.iloc[j]) if pd.notna(current_factor.iloc[j]) else np.nan,
                "next_error_value": float(next_err.iloc[j]) if pd.notna(next_err.iloc[j]) else np.nan,
                "next_error_delta": float(next_delta.iloc[j]) if pd.notna(next_delta.iloc[j]) else np.nan,
                "next_error_factor": float(next_factor.iloc[j]) if pd.notna(next_factor.iloc[j]) else np.nan,
                "is_current_error_spike_delta": bool(pd.notna(current_delta.iloc[j]) and current_delta.iloc[j] >= spike_delta_mm),
                "is_next_error_spike_delta": bool(pd.notna(next_delta.iloc[j]) and next_delta.iloc[j] >= spike_delta_mm),
                "is_current_error_spike_factor": bool(pd.notna(current_factor.iloc[j]) and current_factor.iloc[j] >= spike_factor),
                "is_next_error_spike_factor": bool(pd.notna(next_factor.iloc[j]) and next_factor.iloc[j] >= spike_factor),
                "N_sqp_used": float(to_num(df, "N_sqp_used").iloc[j]) if "N_sqp_used" in df.columns else np.nan,
                "sqp_iters_done": float(to_num(df, "sqp_iters_done").iloc[j]) if "sqp_iters_done" in df.columns else np.nan,
                "sqp_du_rel_first": float(rel_first.iloc[j]) if pd.notna(rel_first.iloc[j]) else np.nan,
                "sqp_du_rel_final": float(rel_final.iloc[j]) if pd.notna(rel_final.iloc[j]) else np.nan,
                "sqp_du_rel_max": float(rel_max.iloc[j]) if pd.notna(rel_max.iloc[j]) else np.nan,
                "sqp_du_rel_min": float(rel_min.iloc[j]) if pd.notna(rel_min.iloc[j]) else np.nan,
                "sqp_du_rel_drop_first_to_final": float(rel_drop.iloc[j]) if pd.notna(rel_drop.iloc[j]) else np.nan,
                "sqp_du_rel_ratio_final_over_first": float(rel_ratio.iloc[j]) if pd.notna(rel_ratio.iloc[j]) else np.nan,
                "sqp_du_rel_dominant_iter_1based": float(dominant_idx.iloc[j]) if pd.notna(dominant_idx.iloc[j]) else np.nan,
                "sqp_du_first": float(du_first.iloc[j]),
                "sqp_du_final": float(du_final.iloc[j]),
                "sqp_du_max": float(du_max.iloc[j]),
                "sqp_du_min": float(du_min.iloc[j]),
            })

    events = pd.DataFrame(rows)
    events.to_csv(out_root / "sqp_relative_change_events.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_relative_change_events.csv'}")
    return events


def build_relative_change_threshold_sweep(
    run_root: Path,
    *,
    out_root: Path,
    rel_tolerances: list[float],
) -> pd.DataFrame:
    """Simulate a controller that stops SQP when sqp_du_rel_i <= tolerance.

    This does not re-run the controller. It answers: given the logged inner-loop
    sequence, how many iterations would a relative-change stopping rule have used?
    """
    rows = []
    group_cols = ["bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np", "sqp_budget"]

    for run in collect_run_logs(run_root):
        df = run["df"]
        rel_cols = get_sqp_du_rel_cols(df)
        if not rel_cols:
            continue

        actual_iters = to_num(df, "sqp_iters_done")
        budget = run["sqp_budget"]
        available_iters = min(len(rel_cols), int(budget) if np.isfinite(budget) else len(rel_cols))

        for tol in rel_tolerances:
            predicted = df.apply(lambda row: predict_stop_iter_from_relative_change(row, rel_cols[:available_iters], rel_tol=tol), axis=1)
            predicted_capped = predicted.fillna(available_iters)
            stopped_by_tol = predicted.notna()
            saved_vs_budget = available_iters - predicted_capped
            saved_vs_actual = actual_iters - predicted_capped

            row = {c: run[c] for c in group_cols}
            row.update({
                "rel_tol": float(tol),
                "n_steps": int(len(df)),
                "available_logged_iters": int(available_iters),
                "mean_predicted_iters_by_rel_tol": float(predicted_capped.mean()),
                "median_predicted_iters_by_rel_tol": float(predicted_capped.median()),
                "max_predicted_iters_by_rel_tol": float(predicted_capped.max()),
                "pct_steps_stopped_by_rel_tol": 100.0 * float(stopped_by_tol.mean()),
                "pct_steps_hit_max_budget": 100.0 * float((~stopped_by_tol).mean()),
                "mean_iters_saved_vs_budget": float(saved_vs_budget.mean()),
                "median_iters_saved_vs_budget": float(saved_vs_budget.median()),
                "mean_iters_saved_vs_actual": float(saved_vs_actual.mean()),
                "median_iters_saved_vs_actual": float(saved_vs_actual.median()),
                "mean_sqp_du_rel_final": float(to_num(df, "sqp_du_rel_final").mean()),
                "median_sqp_du_rel_final": float(to_num(df, "sqp_du_rel_final").median()),
                "mean_adapt_pred_err_xy_mm": float(to_num(df, "adapt_pred_err_xy_mm").mean()),
                "max_adapt_pred_err_xy_mm": float(to_num(df, "adapt_pred_err_xy_mm").max()),
                "min_clearance_mm": float(to_num(df, "adapt_clearance_mm").min()),
            })
            rows.append(row)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_root / "sqp_relative_change_stop_threshold_sweep.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_relative_change_stop_threshold_sweep.csv'}")
    return sweep


def build_spike_predictor_threshold_table(
    events: pd.DataFrame,
    *,
    out_root: Path,
    rel_thresholds: list[float],
    spike_col: str = "is_next_error_spike_delta",
) -> pd.DataFrame:
    """Precision/recall style table: does high relative change flag next-step error spikes?"""
    rows = []
    if events.empty:
        out = pd.DataFrame()
        out.to_csv(out_root / "sqp_rel_change_spike_predictor_thresholds.csv", index=False)
        return out

    group_cols = ["bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np", "sqp_budget"]
    for key, g in events.groupby(group_cols, dropna=False):
        spike = g[spike_col].astype(bool)
        for thr in rel_thresholds:
            high = pd.to_numeric(g["sqp_du_rel_final"], errors="coerce") >= float(thr)
            tp = int((high & spike).sum())
            fp = int((high & ~spike).sum())
            fn = int((~high & spike).sum())
            tn = int((~high & ~spike).sum())
            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / (tp + fn) if (tp + fn) else np.nan
            false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan
            out = dict(zip(group_cols, key))
            out.update({
                "rel_threshold": float(thr),
                "spike_col": spike_col,
                "n_events": int(len(g)),
                "num_high_rel_change": int(high.sum()),
                "num_spikes": int(spike.sum()),
                "tp_high_rel_and_spike": tp,
                "fp_high_rel_no_spike": fp,
                "fn_low_rel_but_spike": fn,
                "tn_low_rel_no_spike": tn,
                "precision_p_spike_given_high_rel": float(precision) if np.isfinite(precision) else np.nan,
                "recall_p_high_rel_given_spike": float(recall) if np.isfinite(recall) else np.nan,
                "false_positive_rate": float(false_positive_rate) if np.isfinite(false_positive_rate) else np.nan,
                "mean_next_error_delta_when_high_rel": float(pd.to_numeric(g.loc[high, "next_error_delta"], errors="coerce").mean()),
                "mean_next_error_delta_when_low_rel": float(pd.to_numeric(g.loc[~high, "next_error_delta"], errors="coerce").mean()),
            })
            rows.append(out)

    table = pd.DataFrame(rows)
    table.to_csv(out_root / "sqp_rel_change_spike_predictor_thresholds.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'sqp_rel_change_spike_predictor_thresholds.csv'}")
    return table


def plot_error_vs_relative_change_scatter(events: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if events.empty:
        return
    for (bend, rollout), g in events.groupby(["bend_angle_deg", "rollout_steps"], dropna=False):
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for sqp_budget, gg in g.groupby("sqp_budget", dropna=False):
            x = pd.to_numeric(gg["sqp_du_rel_final"], errors="coerce")
            y = pd.to_numeric(gg["next_error_delta"], errors="coerce")
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            if valid.any():
                ax.scatter(x[valid], y[valid], s=42, alpha=0.75, label=f"sqp{int(sqp_budget)}")
        ax.axhline(0.0, linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("Final relative SQP update, sqp_du_rel_final")
        ax.set_ylabel("Next-step prediction-error change [mm]")
        ax.set_title(f"Relative SQP change vs next error jump, bend {bend:.0f}°, rollout={int(rollout)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"rel_change_vs_next_error_delta_bend_{bend_tag(bend)}_rollout_{int(rollout)}.png", dpi=220)
        plt.close(fig)


def plot_relative_stop_threshold_sweep(sweep: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if sweep.empty:
        return
    for (bend, rollout), g in sweep.groupby(["bend_angle_deg", "rollout_steps"], dropna=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for sqp_budget, gg in g.groupby("sqp_budget", dropna=False):
            gg = gg.sort_values("rel_tol")
            ax.plot(
                gg["rel_tol"],
                gg["mean_predicted_iters_by_rel_tol"],
                marker="o",
                linewidth=2.0,
                label=f"sqp{int(sqp_budget)}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Relative-change stopping tolerance")
        ax.set_ylabel("Mean predicted SQP iterations used")
        ax.set_title(f"Simulated relative-change stopping, bend {bend:.0f}°, rollout={int(rollout)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"relative_stop_threshold_sweep_bend_{bend_tag(bend)}_rollout_{int(rollout)}.png", dpi=220)
        plt.close(fig)


def plot_error_and_rel_change_traces(run_root: Path, *, out_dir: Path, error_metric: str = "adapt_pred_err_xy_mm") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_run_logs(run_root)
    for run in runs:
        df = run["df"]
        if "i_ref" not in df.columns or error_metric not in df.columns or "sqp_du_rel_final" not in df.columns:
            continue
        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        x = to_num(df, "i_ref")
        err = to_num(df, error_metric)
        rel = to_num(df, "sqp_du_rel_final")
        valid_err = x.notna() & err.notna()
        valid_rel = x.notna() & rel.notna()
        ax1.plot(x[valid_err], err[valid_err], marker="o", linewidth=2.0, label=error_metric)
        ax1.set_xlabel("Reference index along trajectory")
        ax1.set_ylabel("Prediction error [mm]")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(x[valid_rel], rel[valid_rel], marker="s", linestyle="--", linewidth=1.8, label="sqp_du_rel_final")
        ax2.set_yscale("log")
        ax2.set_ylabel("Final relative SQP update")
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        ax1.set_title(run["run_name"])
        fig.tight_layout()
        fig.savefig(out_dir / f"error_and_rel_change_{safe_filename(run['run_name'])}.png", dpi=220)
        plt.close(fig)

# ----------------------------
# Reporting helpers
# ----------------------------

def print_compact_recommendation_tables(summary: pd.DataFrame, trends: pd.DataFrame) -> None:
    key_metrics = [
        "final_i_ref",
        "mean_adapt_pred_err_xy_mm",
        "min_clearance_mm",
        "max_tip_vessel_angle_deg",
        "mean_cond_H_mpc",
        "control_rms",
        "median_sqp_du_final",
    ]

    cols = [
        "bend_angle_deg", "rollout_steps", "sqp_budget",
        "success", "final_i_ref",
        "mean_adapt_pred_err_xy_mm", "min_clearance_mm",
        "max_tip_vessel_angle_deg",
        "mean_cond_H_mpc", "control_rms", "median_sqp_du_final",
    ]
    cols = [c for c in cols if c in summary.columns]

    print("\n[SUMMARY] Compact run table:")
    if cols:
        print(summary[cols].to_string(index=False))

    print("\n[TRENDS] Slope per additional configured SQP iteration:")
    if trends is None or trends.empty or "metric" not in trends.columns:
        print("[TRENDS] Not enough SQP-budget points per stratum to estimate trends.")
        return
    trend_view = trends[trends["metric"].isin(key_metrics)].copy()
    trend_cols = [
        "bend_angle_deg", "rollout_steps", "metric",
        "slope_per_sqp_iter", "spearman_r",
        "value_at_min_sqp", "value_at_max_sqp",
        "delta_max_minus_min_sqp",
    ]
    trend_cols = [c for c in trend_cols if c in trend_view.columns]
    if len(trend_view) and trend_cols:
        print(trend_view[trend_cols].to_string(index=False))


def run_analysis(
    run_root: Path,
    *,
    out_root: Path,
    baseline_sqp: int = 1,
    rel_tolerances: list[float] | None = None,
    rel_thresholds: list[float] | None = None,
    error_metric: str = "adapt_pred_err_xy_mm",
    spike_delta_mm: float = 1.0,
    spike_factor: float = 2.0,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    if rel_tolerances is None:
        rel_tolerances = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    if rel_thresholds is None:
        rel_thresholds = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    summary = build_sqp_summary(run_root, out_root)
    if summary.empty:
        print("[ANALYSIS] No runs found. Expected subdirectories containing log.csv")
        return

    rel = build_relative_to_sqp1(
        summary,
        out_root=out_root,
        baseline_sqp=baseline_sqp,
    )
    trends = build_trend_summary(summary, out_root=out_root)
    inc = build_incremental_gain_table(summary, out_root=out_root)

    events = build_relative_change_event_table(
        run_root,
        out_root=out_root,
        error_metric=error_metric,
        spike_delta_mm=spike_delta_mm,
        spike_factor=spike_factor,
    )
    threshold_sweep = build_relative_change_threshold_sweep(
        run_root,
        out_root=out_root,
        rel_tolerances=rel_tolerances,
    )
    spike_thresholds = build_spike_predictor_threshold_table(
        events,
        out_root=out_root,
        rel_thresholds=rel_thresholds,
        spike_col="is_next_error_spike_delta",
    )

    plot_metrics = [
        "final_i_ref",
        "max_i_ref",
        "n_steps",
        "num_infeasible_steps",

        "mean_adapt_pred_err_xy_mm",
        "max_adapt_pred_err_xy_mm",
        "mean_adapt_pred_err_xyz_mm",
        "max_adapt_pred_err_xyz_mm",

        "min_clearance_mm",
        "mean_clearance_mm",
        "mean_tip_vessel_angle_deg",
        "max_tip_vessel_angle_deg",

        "mean_cond_H_beam",
        "max_cond_H_beam",
        "mean_cond_H_mpc",
        "max_cond_H_mpc",
        "mean_mpc_eig_cond",
        "max_mpc_eig_cond",

        "control_rms",
        "control_max_abs",
        "mean_sqp_iters_done",
        "median_sqp_du_final",
        "mean_sqp_du_rel_final",
        "max_sqp_du_rel_any",
        "p90_sqp_du_rel_any",
        "mean_sqp_rel_drop_first_to_final",
        "mean_sqp_rel_ratio_final_over_first",
        "max_positive_adapt_pred_err_xy_delta_mm",
    ]

    for metric in plot_metrics:
        plot_metric_vs_sqp_by_rollout(
            summary,
            metric,
            out_dir=out_root / "plots" / "absolute_vs_sqp",
        )
        plot_relative_metric_vs_sqp(
            rel,
            metric,
            out_dir=out_root / "plots" / f"relative_to_sqp{baseline_sqp}",
            baseline_sqp=baseline_sqp,
        )

    trace_metrics = [
        "adapt_pred_err_xy_mm",
        "adapt_pred_err_xyz_mm",
        "adapt_clearance_mm",
        "adapt_tip_vessel_angle_deg",
        "cond_H_beam",
        "cond_H_mpc",
        "mpc_eig_cond",
        "sqp_iters_done",
        "sqp_du_final",
        "sqp_du_rel_final",
        "u0_dL",
    ]

    for metric in trace_metrics:
        plot_reference_traces_by_sqp(
            run_root,
            metric,
            out_dir=out_root / "plots" / "reference_traces_by_sqp",
        )

    plot_sqp_iteration_decay(
        run_root,
        out_dir=out_root / "plots" / "sqp_inner_iteration_decay",
    )
    plot_error_vs_relative_change_scatter(
        events,
        out_dir=out_root / "plots" / "relative_change_vs_error_spikes",
    )
    plot_relative_stop_threshold_sweep(
        threshold_sweep,
        out_dir=out_root / "plots" / "relative_stop_threshold_sweep",
    )
    plot_error_and_rel_change_traces(
        run_root,
        out_dir=out_root / "plots" / "error_and_relative_change_traces",
        error_metric=error_metric,
    )

    print_compact_recommendation_tables(summary, trends)

    print("\n[ANALYSIS] Main files to inspect:")
    print(f"  {out_root / 'sqp_summary.csv'}")
    print(f"  {out_root / f'sqp_effect_relative_to_sqp{baseline_sqp}.csv'}")
    print(f"  {out_root / 'sqp_incremental_gains.csv'}")
    print(f"  {out_root / 'sqp_relative_change_events.csv'}")
    print(f"  {out_root / 'sqp_relative_change_stop_threshold_sweep.csv'}")
    print(f"  {out_root / 'sqp_rel_change_spike_predictor_thresholds.csv'}")
    print(f"  {out_root / 'sqp_trend_summary.csv'}")
    print(f"  {out_root / 'plots'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse effect of increasing SQP iterations inside the controller."
    )
    parser.add_argument(
        "run_root",
        type=Path,
        nargs="?",
        default=Path("run_figure4_simulation_sqp_investigation"),
        help="Root directory containing run subdirectories with log.csv files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_root>/sqp_analysis.",
    )
    parser.add_argument(
        "--baseline-sqp",
        type=int,
        default=1,
        help="Baseline SQP budget for relative comparisons.",
    )
    parser.add_argument(
        "--rel-tolerances",
        type=str,
        default="0.05,0.1,0.2,0.5,1,2,5",
        help="Comma-separated sqp_du_rel_i tolerances for simulated relative-change stopping.",
    )
    parser.add_argument(
        "--rel-thresholds",
        type=str,
        default="0.2,0.5,1,2,5,10",
        help="Comma-separated sqp_du_rel_final thresholds for spike-predictor diagnostics.",
    )
    parser.add_argument(
        "--error-metric",
        type=str,
        default="adapt_pred_err_xy_mm",
        help="Prediction-error column used for spike diagnostics.",
    )
    parser.add_argument(
        "--spike-delta-mm",
        type=float,
        default=1.0,
        help="Absolute next-step error increase [mm] counted as a spike.",
    )
    parser.add_argument(
        "--spike-factor",
        type=float,
        default=2.0,
        help="Multiplicative error increase counted as a spike in the event table.",
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    out_root = args.out_root
    if out_root is None:
        out_root = run_root / "sqp_analysis"
    else:
        out_root = out_root.expanduser().resolve()

    run_analysis(
        run_root,
        out_root=out_root,
        baseline_sqp=args.baseline_sqp,
        rel_tolerances=parse_float_list(args.rel_tolerances),
        rel_thresholds=parse_float_list(args.rel_thresholds),
        error_metric=args.error_metric,
        spike_delta_mm=args.spike_delta_mm,
        spike_factor=args.spike_factor,
    )


if __name__ == "__main__":
    main()
