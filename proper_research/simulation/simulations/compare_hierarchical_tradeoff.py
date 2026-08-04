#!/usr/bin/env python3
"""
compare_hierarchical_tradeoff.py

Purpose
-------
Compare six single-run controller logs:

    30 deg:
        1) low rollout + high SQP
        2) high rollout + low SQP
        3) hierarchical controller

    90 deg:
        4) low rollout + high SQP
        5) high rollout + low SQP
        6) hierarchical controller

The output is designed to support the claim:

    The hierarchical controller achieves accuracy comparable to the expensive
    low-rollout/high-SQP baseline, but at substantially lower computational cost.

Main outputs
------------
    tradeoff_summary_by_run.csv
    paired_hierarchical_comparison.csv
    plots/tradeoff_pareto_p90_adaptive_error_mm.png
    plots/tradeoff_pareto_mean_adaptive_error_mm.png
    plots/cost_by_condition.png
    plots/error_by_condition_p90_adaptive_error_mm.png
    plots/hierarchical_cost_saving_vs_error_ratio.png

How to use
----------
1. Edit the RUNS list below so each path points to your local log.csv.
2. Run:

    python compare_hierarchical_tradeoff.py --out-dir hierarchical_tradeoff_analysis

Optional:
    python compare_hierarchical_tradeoff.py \
        --out-dir hierarchical_tradeoff_analysis \
        --primary-error p90_adaptive_error_mm \
        --primary-cost realised_sqp_rollout_units

Notes
-----
- The script uses a robust CSV reader because some of your logs contain trailing
  vector/list fields with unquoted commas.
- The primary cost proxy is:

      realised_sqp_rollout_units = sum(sqp_iters_done * rollout_steps_used)

  This is more useful than raw SQP iteration count because high rollout means
  each SQP iteration solves a larger horizon problem.
- If you later log wall-clock solve time, the script will also summarise those
  columns automatically.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# EDIT THIS BLOCK
# =====================================================================
# Replace each "path" with your real local path to the corresponding log.csv.
# Keep condition_key exactly as one of:
#     low_rollout_high_sqp
#     high_rollout_low_sqp
#     hierarchical
#
# The bend_deg values are used for paired comparison and grouping.
RUNS = [
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp_budget_sweep/sqp6/bend_m30_jac_contact_plant_contact_1_sqp_full_rollout1_Np5_sqp6/log.csv",
        "bend_deg": 30,
        "condition_key": "low_rollout_high_sqp",
        "display_name": "30° low rollout + high SQP",
    },
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp_budget_sweep/sqp1/bend_m30_jac_contact_plant_contact_1_sqp_full_rollout5_Np5_sqp1/log.csv",
        "bend_deg": 30,
        "condition_key": "high_rollout_low_sqp",
        "display_name": "30° high rollout + low SQP",
    },
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp4_hierarchical_controller/bend_m30_jac_contact_plant_contact_1_sqp_full_rollout5_Np5_sqp4/log.csv",
        "bend_deg": 30,
        "condition_key": "hierarchical",
        "display_name": "30° hierarchical",
    },
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp_budget_sweep/sqp4/bend_m90_jac_contact_plant_contact_1_sqp_full_rollout1_Np5_sqp4/log.csv",
        "bend_deg": 90,
        "condition_key": "low_rollout_high_sqp",
        "display_name": "90° low rollout + high SQP",
    },
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp_budget_sweep/sqp1/bend_m90_jac_contact_plant_contact_1_sqp_full_rollout5_Np5_sqp1/log.csv",
        "bend_deg": 90,
        "condition_key": "high_rollout_low_sqp",
        "display_name": "90° high rollout + low SQP",
    },
    {
        "path": "/Users/jackhilton-jones/Proper-Research/evidence_contact_sqp4_hierarchical_controller/bend_m90_jac_contact_plant_contact_1_sqp_full_rollout5_Np5_sqp4/log.csv",
        "bend_deg": 90,
        "condition_key": "hierarchical",
        "display_name": "90° hierarchical",
    },
]
# =====================================================================


CONDITION_ORDER = [
    "low_rollout_high_sqp",
    "high_rollout_low_sqp",
    "hierarchical",
]

CONDITION_LABELS = {
    "low_rollout_high_sqp": "Low rollout + high SQP",
    "high_rollout_low_sqp": "High rollout + low SQP",
    "hierarchical": "Hierarchical",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Read experiment CSVs robustly.

    Some raw log files contain unquoted vector/list fields with commas near the
    end of each row. If pandas parsing fails, this fallback truncates/pads each
    row to the header width. The early scalar diagnostic columns are preserved.
    """
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        pass

    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [str(c).strip() for c in header]
        width = len(header)

        for row in reader:
            if len(row) > width:
                row = row[:width]
            elif len(row) < width:
                row = row + [None] * (width - len(row))
            rows.append(row)

    return pd.DataFrame(rows, columns=header)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def numeric(df: pd.DataFrame, col: Optional[str], default: float = np.nan) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def finite_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def finite_sum(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(x.sum()) if x.notna().any() else np.nan


def finite_quantile(x: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.quantile(q)) if len(x) else np.nan


def finite_last(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.iloc[-1]) if len(x) else np.nan


def finite_first(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(x.iloc[0]) if len(x) else np.nan


def bool_rate(x: pd.Series) -> float:
    if x.empty:
        return np.nan
    if x.dtype == bool:
        return float(x.mean())
    s = x.astype(str).str.strip().str.lower()
    mapped = s.map({
        "true": 1.0,
        "t": 1.0,
        "yes": 1.0,
        "y": 1.0,
        "1": 1.0,
        "false": 0.0,
        "f": 0.0,
        "no": 0.0,
        "n": 0.0,
        "0": 0.0,
        "nan": np.nan,
        "none": np.nan,
        "": np.nan,
    })
    return float(mapped.mean()) if mapped.notna().any() else np.nan


def maybe_seconds(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    low = col.lower()
    if low.endswith("_ms") or "millisecond" in low:
        return x / 1000.0
    return x


def summarise_run(run_cfg: dict, vars_per_rollout_step: float) -> dict:
    path = Path(run_cfg["path"]).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    df = read_csv_safely(path)
    if df.empty:
        raise RuntimeError(f"Empty CSV: {path}")

    # Core columns.
    step_col = first_existing(df, ["k", "step", "timestep", "i_ref"])
    sqp_iter_col = first_existing(df, [
        "sqp_iters_done",
        "sqp_iterations_done",
        "sqp_iters",
        "sqp_iterations",
        "nlp_iters",
        "solver_iters",
    ])
    sqp_budget_col = first_existing(df, [
        "N_sqp_used",
        "hierarchy_N_sqp",
        "N_sqp",
        "sqp_budget",
        "sqp_budget_for_filter",
    ])
    rollout_col = first_existing(df, [
        "rollout_steps_used",
        "hierarchy_rollout_steps",
        "runtime_rollout_steps",
        "selected_rollout_steps",
        "rollout_steps",
    ])

    one_step_col = first_existing(df, [
        "pred1_err_xy_mm",
        "one_step_error_mm",
        "pred1_error_xy_mm",
        "pred1_err_xyz_mm",
    ])
    adaptive_col = first_existing(df, [
        "adapt_pred_err_xy_mm",
        "adaptive_error_mm",
        "adapt_error_xy_mm",
        "adapt_pred_err_xyz_mm",
    ])

    infeasible_col = first_existing(df, ["infeasible"])
    hierarchy_col = first_existing(df, ["hierarchy_enabled"])
    qp_dim_col = first_existing(df, ["mpc_H_shape_0", "H_mpc_shape_0", "qp_dim", "n_qp_vars", "n_decision_vars"])

    # Dynamic per-step values.
    n_rows = int(len(df))
    n_steps = int(pd.to_numeric(df[step_col], errors="coerce").nunique()) if step_col else n_rows

    rollout_used = numeric(df, rollout_col)
    if rollout_used.notna().sum() == 0:
        rollout_used = pd.Series(np.nan, index=df.index)
    rollout_median = float(rollout_used.median()) if rollout_used.notna().any() else np.nan
    rollout_mean = float(rollout_used.mean()) if rollout_used.notna().any() else np.nan
    rollout_max = float(rollout_used.max()) if rollout_used.notna().any() else np.nan

    sqp_iters = numeric(df, sqp_iter_col)
    sqp_budget = numeric(df, sqp_budget_col)

    # If actual iterations are missing, use budget as a fallback.
    iteration_source = sqp_iter_col or "estimated_from_budget"
    if sqp_iters.notna().sum() == 0 and sqp_budget.notna().sum() > 0:
        sqp_iters = sqp_budget.copy()

    # If budget is missing, use the realised iterations as a fallback for configured budget.
    if sqp_budget.notna().sum() == 0 and sqp_iters.notna().sum() > 0:
        sqp_budget = sqp_iters.copy()

    # If rollout is missing, use 1 as a conservative fallback.
    if rollout_used.notna().sum() == 0:
        rollout_used = pd.Series(1.0, index=df.index)

    total_sqp_iters = finite_sum(sqp_iters)
    mean_sqp_iters_per_step = finite_mean(sqp_iters)
    max_sqp_iters_per_step = float(sqp_iters.max()) if sqp_iters.notna().any() else np.nan

    # Primary cost metrics.
    realised_sqp_rollout_units = finite_sum(sqp_iters * rollout_used)
    configured_sqp_rollout_budget_units = finite_sum(sqp_budget * rollout_used)

    # Dense proxy. Use actual QP dimension if it looks plausible; otherwise
    # approximate dimension as vars_per_rollout_step * rollout_steps_used.
    qp_dim = numeric(df, qp_dim_col)
    approx_qp_dim = vars_per_rollout_step * rollout_used
    plausible_actual_dim = (
        qp_dim.notna().sum() >= max(3, int(0.5 * len(qp_dim)))
        and float(qp_dim.median(skipna=True)) >= 1.0
    )
    qp_dim_used = qp_dim if plausible_actual_dim else approx_qp_dim
    dense_qp_work_proxy = finite_sum(sqp_iters * (qp_dim_used ** 3))

    one_step = numeric(df, one_step_col)
    adaptive = numeric(df, adaptive_col)

    # Optional timing columns.
    time_candidates = [
        "solve_time_s",
        "mpc_solve_time_s",
        "solver_time_s",
        "step_time_s",
        "wall_time_s",
        "elapsed_s",
        "runtime_s",
        "total_time_s",
        "solve_time_ms",
        "mpc_solve_time_ms",
        "solver_time_ms",
        "step_time_ms",
        "wall_time_ms",
        "elapsed_ms",
        "runtime_ms",
    ]
    time_cols = [c for c in time_candidates if c in df.columns]

    row = {
        "bend_deg": float(run_cfg["bend_deg"]),
        "condition_key": run_cfg["condition_key"],
        "condition_label": CONDITION_LABELS.get(run_cfg["condition_key"], run_cfg["condition_key"]),
        "display_name": run_cfg.get("display_name", ""),
        "source_csv": str(path),
        "n_rows": n_rows,
        "n_steps": n_steps,
        "iteration_source": iteration_source,
        "rollout_median": rollout_median,
        "rollout_mean": rollout_mean,
        "rollout_max": rollout_max,
        "sqp_budget_median": float(sqp_budget.median()) if sqp_budget.notna().any() else np.nan,
        "sqp_budget_mean": finite_mean(sqp_budget),
        "total_sqp_iters": total_sqp_iters,
        "mean_sqp_iters_per_step": mean_sqp_iters_per_step,
        "max_sqp_iters_per_step": max_sqp_iters_per_step,
        "realised_sqp_rollout_units": realised_sqp_rollout_units,
        "configured_sqp_rollout_budget_units": configured_sqp_rollout_budget_units,
        "dense_qp_work_proxy": dense_qp_work_proxy,
        "qp_dim_source": qp_dim_col if plausible_actual_dim else f"approx_{vars_per_rollout_step:g}x_rollout",
        "mean_one_step_error_mm": finite_mean(one_step),
        "median_one_step_error_mm": finite_quantile(one_step, 0.50),
        "p90_one_step_error_mm": finite_quantile(one_step, 0.90),
        "final_one_step_error_mm": finite_last(one_step),
        "mean_adaptive_error_mm": finite_mean(adaptive),
        "median_adaptive_error_mm": finite_quantile(adaptive, 0.50),
        "p90_adaptive_error_mm": finite_quantile(adaptive, 0.90),
        "final_adaptive_error_mm": finite_last(adaptive),
        "infeasible_rate": bool_rate(df[infeasible_col]) if infeasible_col else np.nan,
        "hierarchy_enabled_rate": bool_rate(df[hierarchy_col]) if hierarchy_col else np.nan,
    }

    for tcol in time_cols:
        t = maybe_seconds(df, tcol)
        row[f"total_{safe_filename(tcol)}_seconds"] = finite_sum(t)
        row[f"mean_{safe_filename(tcol)}_seconds"] = finite_mean(t)

    # Sanity warnings encoded into CSV as well.
    warnings_out = []
    if run_cfg["condition_key"] == "hierarchical":
        if np.isfinite(row["hierarchy_enabled_rate"]) and row["hierarchy_enabled_rate"] < 0.5:
            warnings_out.append("labelled hierarchical but hierarchy_enabled is mostly false/missing")
    else:
        if np.isfinite(row["hierarchy_enabled_rate"]) and row["hierarchy_enabled_rate"] > 0.5:
            warnings_out.append("labelled baseline but hierarchy_enabled is mostly true")
    if n_steps < 3:
        warnings_out.append("very few steps")
    if row["mean_adaptive_error_mm"] is np.nan and row["mean_one_step_error_mm"] is np.nan:
        warnings_out.append("no recognised error columns")

    row["warnings"] = "; ".join(warnings_out)

    return row


def build_paired_comparison(summary: pd.DataFrame, primary_error: str, primary_cost: str) -> pd.DataFrame:
    rows = []

    for bend, g in summary.groupby("bend_deg"):
        by_condition = {row["condition_key"]: row for _, row in g.iterrows()}

        if "hierarchical" not in by_condition:
            continue

        h = by_condition["hierarchical"]
        for baseline_key in ["low_rollout_high_sqp", "high_rollout_low_sqp"]:
            if baseline_key not in by_condition:
                continue

            b = by_condition[baseline_key]
            h_error = float(h[primary_error])
            b_error = float(b[primary_error])
            h_cost = float(h[primary_cost])
            b_cost = float(b[primary_cost])

            error_ratio = h_error / b_error if np.isfinite(h_error) and np.isfinite(b_error) and b_error > 0 else np.nan
            cost_ratio = h_cost / b_cost if np.isfinite(h_cost) and np.isfinite(b_cost) and b_cost > 0 else np.nan

            rows.append({
                "bend_deg": bend,
                "baseline_condition": baseline_key,
                "baseline_label": CONDITION_LABELS[baseline_key],
                "primary_error": primary_error,
                "primary_cost": primary_cost,
                "hierarchical_error": h_error,
                "baseline_error": b_error,
                "error_ratio_hierarchical_over_baseline": error_ratio,
                "error_percent_change": 100.0 * (error_ratio - 1.0) if np.isfinite(error_ratio) else np.nan,
                "hierarchical_cost": h_cost,
                "baseline_cost": b_cost,
                "cost_ratio_hierarchical_over_baseline": cost_ratio,
                "cost_percent_saving": 100.0 * (1.0 - cost_ratio) if np.isfinite(cost_ratio) else np.nan,
                "hierarchical_display_name": h["display_name"],
                "baseline_display_name": b["display_name"],
            })

    return pd.DataFrame(rows)


def annotate_points(ax, d: pd.DataFrame, xcol: str, ycol: str) -> None:
    for _, r in d.iterrows():
        label = f"{int(r['bend_deg'])}°\n{r['condition_label']}"
        ax.annotate(
            label,
            (r[xcol], r[ycol]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def plot_tradeoff(summary: pd.DataFrame, out_dir: Path, error_col: str, cost_col: str) -> None:
    d = summary.dropna(subset=[error_col, cost_col]).copy()
    d = d[(d[error_col] > 0) & (d[cost_col] > 0)]
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))

    marker_map = {
        "low_rollout_high_sqp": "o",
        "high_rollout_low_sqp": "s",
        "hierarchical": "^",
    }

    for condition_key in CONDITION_ORDER:
        sub = d[d["condition_key"] == condition_key]
        if sub.empty:
            continue
        ax.scatter(
            sub[cost_col],
            sub[error_col],
            marker=marker_map.get(condition_key, "o"),
            s=90,
            label=CONDITION_LABELS.get(condition_key, condition_key),
            alpha=0.85,
        )

    annotate_points(ax, d, cost_col, error_col)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(cost_col.replace("_", " "))
    ax.set_ylabel(error_col.replace("_", " "))
    ax.set_title("Accuracy/cost trade-off: lower-left is better")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"tradeoff_pareto_{safe_filename(error_col)}.png", dpi=240)
    plt.close(fig)


def plot_bar_by_condition(summary: pd.DataFrame, out_dir: Path, value_col: str, ylabel: str, fname: str) -> None:
    d = summary.dropna(subset=[value_col]).copy()
    if d.empty:
        return

    d["condition_label"] = pd.Categorical(
        d["condition_label"],
        categories=[CONDITION_LABELS[k] for k in CONDITION_ORDER],
        ordered=True,
    )

    pivot = (
        d.pivot_table(
            index="bend_deg",
            columns="condition_label",
            values=value_col,
            aggfunc="mean",
            observed=False,
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Bend angle [deg]")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by bend and controller")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=240)
    plt.close(fig)


def plot_cost_saving_vs_error_ratio(paired: pd.DataFrame, out_dir: Path) -> None:
    if paired.empty:
        return

    # The main scientific claim compares hierarchical against the expensive
    # low-rollout/high-SQP baseline.
    d = paired[paired["baseline_condition"] == "low_rollout_high_sqp"].copy()
    d = d.dropna(subset=[
        "cost_percent_saving",
        "error_ratio_hierarchical_over_baseline",
    ])
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.scatter(
        d["cost_percent_saving"],
        d["error_ratio_hierarchical_over_baseline"],
        s=100,
        alpha=0.85,
    )

    for _, r in d.iterrows():
        ax.annotate(
            f"{int(r['bend_deg'])}°",
            (r["cost_percent_saving"], r["error_ratio_hierarchical_over_baseline"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Hierarchical cost saving vs low-rollout/high-SQP baseline [%]")
    ax.set_ylabel("Hierarchical error / baseline error")
    ax.set_title("Claim check: similar error with lower cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hierarchical_cost_saving_vs_error_ratio.png", dpi=240)
    plt.close(fig)


def write_markdown_report(
    out_dir: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    primary_error: str,
    primary_cost: str,
) -> None:
    lines = []
    lines.append("# Hierarchical controller accuracy/cost trade-off\n")
    lines.append("## Primary metrics\n")
    lines.append(f"- Primary error: `{primary_error}`")
    lines.append(f"- Primary cost: `{primary_cost}`")
    lines.append("- Lower error is better. Lower cost is better.\n")

    lines.append("## Run summary\n")
    cols = [
        "bend_deg",
        "condition_label",
        "n_steps",
        "rollout_mean",
        "sqp_budget_mean",
        "total_sqp_iters",
        "realised_sqp_rollout_units",
        "configured_sqp_rollout_budget_units",
        "mean_adaptive_error_mm",
        "p90_adaptive_error_mm",
        "mean_one_step_error_mm",
        "p90_one_step_error_mm",
        "infeasible_rate",
        "warnings",
    ]
    cols = [c for c in cols if c in summary.columns]
    lines.append(summary[cols].to_markdown(index=False))

    lines.append("\n## Paired hierarchical comparisons\n")
    if paired.empty:
        lines.append("No paired comparisons were possible. Check that each bend has a hierarchical run and at least one baseline.")
    else:
        cols = [
            "bend_deg",
            "baseline_label",
            "error_ratio_hierarchical_over_baseline",
            "error_percent_change",
            "cost_ratio_hierarchical_over_baseline",
            "cost_percent_saving",
        ]
        lines.append(paired[cols].to_markdown(index=False))

    lines.append("\n## Interpretation\n")
    lines.append(
        "- The strongest evidence for your intended claim is: "
        "`cost_percent_saving > 0` while `error_ratio_hierarchical_over_baseline` is close to 1 "
        "when the baseline is `Low rollout + high SQP`."
    )
    lines.append(
        "- If the hierarchical controller has a much lower error than `High rollout + low SQP`, "
        "that supports the argument that naive long rollout with insufficient SQP budget is inaccurate."
    )
    lines.append(
        "- With one run per condition, this is descriptive evidence, not a formal statistical test. "
        "For statistics, repeat each condition across seeds/initialisations and apply paired bend-level or run-level models."
    )

    (out_dir / "tradeoff_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("hierarchical_tradeoff_analysis"))
    p.add_argument(
        "--primary-error",
        type=str,
        default="p90_adaptive_error_mm",
        choices=[
            "mean_adaptive_error_mm",
            "median_adaptive_error_mm",
            "p90_adaptive_error_mm",
            "final_adaptive_error_mm",
            "mean_one_step_error_mm",
            "median_one_step_error_mm",
            "p90_one_step_error_mm",
            "final_one_step_error_mm",
        ],
    )
    p.add_argument(
        "--primary-cost",
        type=str,
        default="realised_sqp_rollout_units",
        choices=[
            "total_sqp_iters",
            "realised_sqp_rollout_units",
            "configured_sqp_rollout_budget_units",
            "dense_qp_work_proxy",
        ],
    )
    p.add_argument(
        "--vars-per-rollout-step",
        type=float,
        default=7.0,
        help=(
            "Used only for dense_qp_work_proxy when mpc_H_shape_0 is not reliable. "
            "Your logs suggest roughly 7 decision variables per rollout step."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)
    plot_dir = args.out_dir / "plots"
    safe_mkdir(plot_dir)

    rows = []
    for run_cfg in RUNS:
        rows.append(summarise_run(run_cfg, vars_per_rollout_step=args.vars_per_rollout_step))

    summary = pd.DataFrame(rows)

    # Stable ordering.
    condition_rank = {k: i for i, k in enumerate(CONDITION_ORDER)}
    summary["_condition_rank"] = summary["condition_key"].map(condition_rank)
    summary = summary.sort_values(["bend_deg", "_condition_rank"]).drop(columns=["_condition_rank"])

    summary.to_csv(args.out_dir / "tradeoff_summary_by_run.csv", index=False)

    paired = build_paired_comparison(
        summary,
        primary_error=args.primary_error,
        primary_cost=args.primary_cost,
    )
    paired.to_csv(args.out_dir / "paired_hierarchical_comparison.csv", index=False)

    # Plots.
    plot_tradeoff(summary, plot_dir, "p90_adaptive_error_mm", args.primary_cost)
    plot_tradeoff(summary, plot_dir, "mean_adaptive_error_mm", args.primary_cost)
    plot_tradeoff(summary, plot_dir, "p90_one_step_error_mm", args.primary_cost)
    plot_tradeoff(summary, plot_dir, args.primary_error, args.primary_cost)

    plot_bar_by_condition(
        summary,
        plot_dir,
        args.primary_cost,
        args.primary_cost.replace("_", " "),
        "cost_by_condition.png",
    )
    plot_bar_by_condition(
        summary,
        plot_dir,
        args.primary_error,
        args.primary_error.replace("_", " "),
        f"error_by_condition_{safe_filename(args.primary_error)}.png",
    )
    plot_cost_saving_vs_error_ratio(paired, plot_dir)

    write_markdown_report(args.out_dir, summary, paired, args.primary_error, args.primary_cost)

    print(f"Done. Wrote outputs to: {args.out_dir}")
    print("Key outputs:")
    print(f"  {args.out_dir / 'tradeoff_summary_by_run.csv'}")
    print(f"  {args.out_dir / 'paired_hierarchical_comparison.csv'}")
    print(f"  {args.out_dir / 'tradeoff_report.md'}")
    print(f"  {plot_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
