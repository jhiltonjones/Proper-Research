#!/usr/bin/env python3
"""
Analyse rollout length × prediction horizon sweeps for contact SQP MPC experiments.

Expected local layout examples supported:

    evidence_contact_sqp_budget_sweep/roll_4/pred_6/<run_folder>/log.csv
    evidence_contact_sqp_budget_sweep/sqp5/<run_folder>/log.csv
    evidence_contact_sqp_budget_sweep/<run_folder>/log.csv

The script builds a run-level table, tests whether longer prediction horizons improve
error metrics as rollout changes, and produces diagnostic plots including compute-cost
trade-offs.

Usage:
    python analyse_rollout_prediction_sweep.py --root evidence_contact_sqp_budget_sweep
    python analyse_rollout_prediction_sweep.py --root evidence_contact_sqp_budget_sweep --out analysis_sweep

Main outputs:
    <out>/run_level_summary.csv
    <out>/group_summary.csv
    <out>/paired_horizon_tests.csv
    <out>/ols_results.txt
    <out>/plots/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
PRIMARY_METRICS = [
    "mean_pred1_err_xy_mm",          # one-step prediction error
    "mean_adapt_pred_err_xy_mm",     # adaptive prediction error
    "mean_rollout_err_xy_mm",        # mean over available rollout error columns
    "terminal_rollout_err_xy_mm",    # error at rollout terminal index, when available
    "max_rollout_err_xy_mm",         # worst rollout error across rollout columns/steps
]

SECONDARY_METRICS = [
    "mean_pred1_err_xyz_mm",
    "mean_adapt_pred_err_xyz_mm",
    "min_clearance_mm",
    "mean_clearance_mm",
    "mean_sqp_du_final",
    "p95_cond_H_mpc",
    "failure_rate",
]

EPS = 1e-12


# -----------------------------
# Small utilities
# -----------------------------
def safe_float(x: Any) -> float | np.nan:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def first_existing_numeric(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    return None


def numeric_summary(s: pd.Series, prefix: str) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            f"mean_{prefix}": np.nan,
            f"median_{prefix}": np.nan,
            f"p95_{prefix}": np.nan,
            f"max_{prefix}": np.nan,
            f"min_{prefix}": np.nan,
        }
    return {
        f"mean_{prefix}": float(s.mean()),
        f"median_{prefix}": float(s.median()),
        f"p95_{prefix}": float(s.quantile(0.95)),
        f"max_{prefix}": float(s.max()),
        f"min_{prefix}": float(s.min()),
    }


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        warnings.warn(f"Could not read JSON file {path}: {exc}")
        return {}


def scalar_values(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten scalar JSON values only. Avoid dumping nested arrays into the summary."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(scalar_values(v, prefix=f"{key}_"))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
    return out


def parse_params_from_path(path: Path) -> dict[str, Any]:
    """Extract bend, rollout, horizon, SQP count and mode from directory/file names."""
    text = str(path)
    name_text = " / ".join(path.parts)
    out: dict[str, Any] = {}

    # Folder styles: roll_4 / pred_6, roll1_and_pred6, pred6
    m = re.search(r"(?:^|[/\\])roll[_-]?(\d+)(?:$|[/\\])", text)
    if m:
        out["rollout"] = int(m.group(1))
    m = re.search(r"(?:^|[/\\])pred[_-]?(\d+)(?:$|[/\\])", text)
    if m:
        out["Np"] = int(m.group(1))
    m = re.search(r"roll[_-]?(\d+).*?pred[_-]?(\d+)", name_text)
    if m:
        out.setdefault("rollout", int(m.group(1)))
        out.setdefault("Np", int(m.group(2)))

    # Run-folder style: rollout4_Np6_sqp4
    m = re.search(r"rollout(\d+)", name_text)
    if m:
        out.setdefault("rollout", int(m.group(1)))
    m = re.search(r"Np(\d+)", name_text)
    if m:
        out.setdefault("Np", int(m.group(1)))
    m = re.search(r"sqp(\d+)", name_text)
    if m:
        out["N_sqp_from_path"] = int(m.group(1))

    # Bend style: bend_m30, bend_m90
    m = re.search(r"bend_?m(\d+)", name_text)
    if m:
        out["bend_deg"] = -int(m.group(1))
    m = re.search(r"bend_?p(\d+)", name_text)
    if m:
        out["bend_deg"] = int(m.group(1))

    for mode in ["sqp_full", "sqp", "contact"]:
        if mode in name_text and mode == "sqp_full":
            out["solver_mode_from_path"] = mode

    return out


def extract_config_params(run_dir: Path) -> dict[str, Any]:
    cfg = read_json_if_exists(run_dir / "experiment_config.json")
    meta = read_json_if_exists(run_dir / "meta.json")
    summary = read_json_if_exists(run_dir / "run_summary.json")

    out: dict[str, Any] = {}
    for source_name, source in [("config", cfg), ("meta", meta), ("summary", summary)]:
        flat = scalar_values(source, prefix=f"{source_name}_")
        out.update(flat)

    # Promote common config fields to canonical names when present.
    candidates = {
        "Np": ["config_Np", "meta_Np", "summary_Np"],
        "rollout": [
            "config_rollout_steps",
            "config_rollout_steps_values",
            "meta_rollout_steps",
            "summary_rollout_steps",
        ],
        "N_sqp": ["config_N_sqp", "meta_N_sqp", "summary_N_sqp"],
        "bend_deg": ["config_bend_angle_deg", "meta_bend_angle_deg", "summary_bend_angle_deg"],
    }
    for canonical, keys in candidates.items():
        for key in keys:
            if key in out:
                val = out[key]
                # Some configs store a singleton tuple/list as string; path parsing usually handles this.
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    out.setdefault(canonical, int(val) if canonical in {"Np", "rollout", "N_sqp"} else float(val))
                    break
    return out


def infer_missing_params_from_log(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "rollout_steps" in df.columns:
        vals = pd.to_numeric(df["rollout_steps"], errors="coerce").dropna().unique()
        if len(vals) == 1:
            out["rollout"] = int(vals[0])
    if "rollout_steps_used" in df.columns:
        vals = pd.to_numeric(df["rollout_steps_used"], errors="coerce").dropna().unique()
        if len(vals) == 1:
            out.setdefault("rollout", int(vals[0]))
    if "N_sqp_used" in df.columns:
        vals = pd.to_numeric(df["N_sqp_used"], errors="coerce").dropna().unique()
        if len(vals) == 1:
            out["N_sqp"] = int(vals[0])
    return out


def get_runtime_like_columns(df: pd.DataFrame) -> list[str]:
    """Find numeric wall-clock/cpu/elapsed columns, if your logger later includes them."""
    patterns = re.compile(r"(time|runtime|elapsed|wall|cpu|solve|solver).*(_s|sec|seconds|ms)?$", re.I)
    reject = re.compile(r"(status|mode|used|steps|iters|iteration)", re.I)
    cols: list[str] = []
    for c in df.columns:
        if patterns.search(c) and not reject.search(c) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def rollout_error_columns(df: pd.DataFrame, kind: str = "xy") -> list[tuple[int, str]]:
    pat = re.compile(rf"rollout_err_{kind}_(\d+)_mm$")
    cols: list[tuple[int, str]] = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    return sorted(cols)


def summarise_run(log_path: Path) -> dict[str, Any]:
    run_dir = log_path.parent
    try:
        df = pd.read_csv(log_path)
    except Exception as exc:
        warnings.warn(f"Skipping unreadable CSV {log_path}: {exc}")
        return {"log_path": str(log_path), "read_error": str(exc)}

    row: dict[str, Any] = {
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "n_steps_logged": int(len(df)),
    }

    # Parameters: path first, then JSON/config, then log fallbacks.
    row.update(parse_params_from_path(log_path))
    cfg_params = extract_config_params(run_dir)
    for k, v in cfg_params.items():
        row.setdefault(k, v)
    log_params = infer_missing_params_from_log(df)
    for k, v in log_params.items():
        row.setdefault(k, v)

    # Canonicalise N_sqp.
    if "N_sqp" not in row and "N_sqp_from_path" in row:
        row["N_sqp"] = row["N_sqp_from_path"]

    # Status / feasibility.
    if "status" in df.columns:
        status = df["status"].astype(str).str.lower()
        row["n_status_solved"] = int(status.str.contains("solved|success|optimal", regex=True).sum())
        row["failure_rate"] = float(1.0 - row["n_status_solved"] / max(len(df), 1))
        row["all_steps_solved"] = bool(row["n_status_solved"] == len(df))
    if "infeasible" in df.columns:
        infeas = pd.to_numeric(df["infeasible"], errors="coerce").fillna(0)
        row["n_infeasible"] = int((infeas != 0).sum())
        row["any_infeasible"] = bool(row["n_infeasible"] > 0)

    # Core prediction / adaptive errors.
    direct_metrics = {
        "pred1_err_xy_mm": "pred1_err_xy_mm",
        "pred1_err_xyz_mm": "pred1_err_xyz_mm",
        "adapt_pred_err_xy_mm": "adapt_pred_err_xy_mm",
        "adapt_pred_err_xyz_mm": "adapt_pred_err_xyz_mm",
        "clearance_mm": "clearance_mm",
        "adapt_clearance_mm": "adapt_clearance_mm",
        "adapt_tip_vessel_angle_deg": "adapt_tip_vessel_angle_deg",
        "sqp_iters_done": "sqp_iters_done",
        "sqp_du_final": "sqp_du_final",
        "sqp_du_rel_final": "sqp_du_rel_final",
        "cond_H_mpc": "cond_H_mpc",
        "mpc_eig_cond": "mpc_eig_cond",
        "cond_H_beam": "cond_H_beam",
        "curvature_local_1pm": "curvature_local_1pm",
        "curvature_max_ahead_1pm": "curvature_max_ahead_1pm",
        "curvature_mean_ahead_1pm": "curvature_mean_ahead_1pm",
        "qp_reg": "qp_reg",
    }
    for col, prefix in direct_metrics.items():
        if col in df.columns:
            row.update(numeric_summary(df[col], prefix))

    # Rollout error summaries across all available rollout indices.
    for kind in ["xy", "xyz"]:
        cols = rollout_error_columns(df, kind=kind)
        if cols:
            values = pd.concat([pd.to_numeric(df[c], errors="coerce") for _, c in cols], axis=0)
            row.update(numeric_summary(values, f"rollout_err_{kind}_mm"))

            available_indices = [i for i, _ in cols]
            row[f"max_available_rollout_error_index_{kind}"] = int(max(available_indices))
            # Terminal rollout error: prefer rollout-1, otherwise max available index.
            rollout = row.get("rollout", np.nan)
            target_i = None
            if pd.notna(rollout):
                requested = int(rollout) - 1
                if requested in available_indices:
                    target_i = requested
            if target_i is None:
                target_i = max(available_indices)
            terminal_col = dict(cols)[target_i]
            terminal = pd.to_numeric(df[terminal_col], errors="coerce")
            row.update(numeric_summary(terminal, f"terminal_rollout_err_{kind}_mm"))
            row[f"terminal_rollout_error_index_{kind}"] = int(target_i)

            # Per-index means are useful to see where rollout predictions degrade.
            for i, c in cols:
                s = pd.to_numeric(df[c], errors="coerce")
                row[f"mean_rollout_err_{kind}_{i}_mm"] = float(s.mean()) if s.notna().any() else np.nan
                row[f"p95_rollout_err_{kind}_{i}_mm"] = float(s.quantile(0.95)) if s.notna().any() else np.nan

    # Compute cost: prefer real timing columns if present; always include proxies.
    runtime_cols = get_runtime_like_columns(df)
    for c in runtime_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        row[f"sum_{c}"] = float(s.sum())
        row[f"mean_{c}"] = float(s.mean())

    sqp_iters_sum = safe_float(row.get("mean_sqp_iters_done")) * max(len(df), 1)
    Np = safe_float(row.get("Np"))
    N_sqp = safe_float(row.get("N_sqp"))
    row["total_sqp_iters"] = sqp_iters_sum if not math.isnan(sqp_iters_sum) else np.nan
    row["sqp_work_units"] = sqp_iters_sum * Np if not (math.isnan(sqp_iters_sum) or math.isnan(Np)) else np.nan
    row["configured_budget_units"] = max(len(df), 1) * Np * N_sqp if not (math.isnan(Np) or math.isnan(N_sqp)) else np.nan

    if "mpc_H_shape_0" in df.columns:
        h_dim = pd.to_numeric(df["mpc_H_shape_0"], errors="coerce").median()
        row["median_mpc_H_dim"] = float(h_dim) if pd.notna(h_dim) else np.nan
        if pd.notna(h_dim) and not math.isnan(sqp_iters_sum):
            # Rough dense-solve scaling proxy; use relatively, not as seconds.
            row["dense_qp_work_proxy"] = float(sqp_iters_sum * (h_dim ** 3))

    # Some run_summary.json files may contain useful actual timing. Bring scalar timing-like JSON keys forward.
    for key, val in list(row.items()):
        if re.search(r"(time|runtime|elapsed|wall|cpu|duration)", str(key), re.I):
            num = safe_float(val)
            if not math.isnan(num):
                row[f"numeric_{key}"] = num

    # Derived design variables.
    if pd.notna(row.get("Np", np.nan)) and pd.notna(row.get("rollout", np.nan)):
        row["extra_horizon"] = int(row["Np"]) - int(row["rollout"])
        row["valid_pred_ge_rollout"] = bool(row["extra_horizon"] >= 0)

    return row


def collect_runs(root: Path) -> pd.DataFrame:
    logs = sorted(root.rglob("log.csv"))
    # Allow testing on a single uploaded CSV not named exactly log.csv.
    if root.is_file() and root.suffix.lower() == ".csv":
        logs = [root]
    if not logs:
        raise FileNotFoundError(f"No log.csv files found under {root}")
    rows = [summarise_run(p) for p in logs]
    runs = pd.DataFrame(rows)

    # Fill canonical Np/rollout from path/config/log variants if possible.
    for c in ["rollout", "Np", "N_sqp", "bend_deg"]:
        if c in runs.columns:
            runs[c] = pd.to_numeric(runs[c], errors="coerce")
    if "rollout" in runs.columns and "Np" in runs.columns:
        runs["extra_horizon"] = runs["Np"] - runs["rollout"]
    return runs


# -----------------------------
# Statistical testing
# -----------------------------
def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    p = p_values.astype(float).copy()
    valid = p.dropna().sort_values()
    m = len(valid)
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    running_max = 0.0
    for rank, (idx, pv) in enumerate(valid.items(), start=1):
        adj = min(1.0, (m - rank + 1) * pv)
        running_max = max(running_max, adj)
        adjusted.loc[idx] = running_max
    return adjusted


def paired_tests(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Within each rollout, compare longer Np against the baseline Np = rollout.
    Pairs are matched by bend angle if bend_deg is available.
    """
    needed = {"rollout", "Np", metric}
    if not needed.issubset(runs.columns):
        return pd.DataFrame()

    try:
        from scipy.stats import ttest_rel, wilcoxon
    except Exception:
        ttest_rel = wilcoxon = None

    group_cols = ["rollout", "Np"]
    id_col = "bend_deg" if "bend_deg" in runs.columns and runs["bend_deg"].notna().any() else None

    rows: list[dict[str, Any]] = []
    for roll, sub in runs.dropna(subset=["rollout", "Np", metric]).groupby("rollout"):
        baseline_np = int(roll)
        base = sub[sub["Np"] == baseline_np]
        if base.empty:
            # Fall back to smallest available horizon for that rollout.
            baseline_np = int(sub["Np"].min())
            base = sub[sub["Np"] == baseline_np]
        for np_val, comp in sub.groupby("Np"):
            if int(np_val) == baseline_np:
                continue
            row: dict[str, Any] = {
                "metric": metric,
                "rollout": int(roll),
                "baseline_Np": int(baseline_np),
                "comparison_Np": int(np_val),
                "extra_horizon_vs_baseline": int(np_val - baseline_np),
            }
            if id_col:
                merged = base[[id_col, metric]].rename(columns={metric: "baseline"}).merge(
                    comp[[id_col, metric]].rename(columns={metric: "comparison"}),
                    on=id_col,
                    how="inner",
                )
            else:
                # Last-resort unpaired-by-row alignment. Prefer bend_deg in your experiment design.
                n = min(len(base), len(comp))
                merged = pd.DataFrame({
                    "baseline": base[metric].to_numpy()[:n],
                    "comparison": comp[metric].to_numpy()[:n],
                })
            merged = merged.dropna()
            row["n_pairs"] = int(len(merged))
            if len(merged) >= 2:
                diff = merged["comparison"] - merged["baseline"]
                row["mean_baseline"] = float(merged["baseline"].mean())
                row["mean_comparison"] = float(merged["comparison"].mean())
                row["mean_delta_comparison_minus_baseline"] = float(diff.mean())
                row["median_delta_comparison_minus_baseline"] = float(diff.median())
                row["pct_change_vs_baseline"] = float(100.0 * diff.mean() / (abs(merged["baseline"].mean()) + EPS))
                row["n_improved"] = int((diff < 0).sum())  # lower error is better
                row["n_worse"] = int((diff > 0).sum())
                row["cohens_dz"] = float(diff.mean() / (diff.std(ddof=1) + EPS)) if len(diff) > 1 else np.nan
                if ttest_rel is not None:
                    try:
                        row["paired_t_p"] = float(ttest_rel(merged["comparison"], merged["baseline"], nan_policy="omit").pvalue)
                    except Exception:
                        row["paired_t_p"] = np.nan
                if wilcoxon is not None and len(diff) >= 3 and np.any(np.abs(diff) > EPS):
                    try:
                        row["wilcoxon_p"] = float(wilcoxon(merged["comparison"], merged["baseline"]).pvalue)
                    except Exception:
                        row["wilcoxon_p"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        for pcol in ["paired_t_p", "wilcoxon_p"]:
            if pcol in out.columns:
                out[f"{pcol}_holm"] = holm_adjust(out[pcol])
    return out


def fit_ols_models(runs: pd.DataFrame, metrics: list[str]) -> str:
    """OLS models with bend fixed effects if statsmodels is installed."""
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return f"statsmodels is not installed, so OLS models were skipped: {exc}\n"

    lines: list[str] = []
    for metric in metrics:
        if metric not in runs.columns:
            continue
        cols = ["rollout", "Np", "extra_horizon", metric]
        data = runs.dropna(subset=[c for c in cols if c in runs.columns]).copy()
        if data.empty or data[metric].nunique(dropna=True) < 2:
            lines.append(f"\n--- {metric}: skipped; insufficient variation.\n")
            continue
        data["log_metric"] = np.log1p(pd.to_numeric(data[metric], errors="coerce"))
        data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_metric", "rollout", "extra_horizon"])
        if len(data) < 10:
            lines.append(f"\n--- {metric}: skipped; fewer than 10 complete runs.\n")
            continue

        # Numeric interaction is better behaved than saturated C(rollout)*C(Np) for triangular designs.
        formula = "log_metric ~ extra_horizon * rollout"
        if "bend_deg" in data.columns and data["bend_deg"].notna().nunique() >= 2:
            formula += " + C(bend_deg)"
        if "curvature_mean_ahead_1pm" in data.columns and data["curvature_mean_ahead_1pm"].notna().nunique() > 1:
            formula += " + mean_curvature_mean_ahead_1pm"

        try:
            model = smf.ols(formula, data=data).fit(cov_type="HC3")
            lines.append(f"\n=== OLS for {metric} ===\n")
            lines.append("Outcome is log1p(metric), so negative extra_horizon coefficients mean lower error for longer horizons.\n")
            lines.append(model.summary().as_text())
            lines.append("\n")
        except Exception as exc:
            lines.append(f"\n--- {metric}: OLS failed: {exc}\n")

    return "\n".join(lines)


def build_group_summary(runs: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = [c for c in ["rollout", "Np", "extra_horizon"] if c in runs.columns]
    if not keys:
        return pd.DataFrame()
    agg: dict[str, list[str]] = {}
    for m in metrics:
        if m in runs.columns:
            agg[m] = ["count", "mean", "median", "std", "min", "max"]
    for m in ["sqp_work_units", "configured_budget_units", "dense_qp_work_proxy", "total_sqp_iters"]:
        if m in runs.columns:
            agg[m] = ["mean", "median", "std"]
    if not agg:
        return pd.DataFrame()
    grouped = runs.groupby(keys, dropna=False).agg(agg)
    grouped.columns = ["_".join([str(x) for x in col if x]) for col in grouped.columns]
    return grouped.reset_index()


# -----------------------------
# Plotting
# -----------------------------
def ensure_plot_dir(out: Path) -> Path:
    p = out / "plots"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_heatmap(runs: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if not {"rollout", "Np", metric}.issubset(runs.columns):
        return
    pivot = runs.pivot_table(index="rollout", columns="Np", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([int(x) if pd.notna(x) else "NA" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([int(x) if pd.notna(x) else "NA" for x in pivot.index])
    ax.set_xlabel("Prediction horizon Np")
    ax.set_ylabel("Rollout steps")
    ax.set_title(f"Mean {metric} by rollout and horizon")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    # Annotate values for small grids.
    if pivot.shape[0] <= 10 and pivot.shape[1] <= 12:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"heatmap_{metric}.png", dpi=200)
    plt.close(fig)


def save_interaction_plot(runs: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if not {"rollout", "Np", metric}.issubset(runs.columns):
        return
    data = runs.dropna(subset=["rollout", "Np", metric])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for roll, sub in sorted(data.groupby("rollout"), key=lambda x: x[0]):
        g = sub.groupby("Np")[metric].agg(["mean", "std", "count"]).reset_index()
        yerr = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.errorbar(g["Np"], g["mean"], yerr=yerr, marker="o", capsize=3, label=f"rollout={int(roll)}")
    ax.set_xlabel("Prediction horizon Np")
    ax.set_ylabel(metric)
    ax.set_title(f"Prediction horizon effect by rollout: {metric}")
    ax.legend(title="Rollout", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"interaction_{metric}.png", dpi=200)
    plt.close(fig)


def save_extra_horizon_plot(runs: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if not {"rollout", "extra_horizon", metric}.issubset(runs.columns):
        return
    data = runs.dropna(subset=["rollout", "extra_horizon", metric])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for roll, sub in sorted(data.groupby("rollout"), key=lambda x: x[0]):
        g = sub.groupby("extra_horizon")[metric].agg(["mean", "std", "count"]).reset_index()
        yerr = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.errorbar(g["extra_horizon"], g["mean"], yerr=yerr, marker="o", capsize=3, label=f"rollout={int(roll)}")
    ax.set_xlabel("Extra horizon beyond rollout: Np - rollout")
    ax.set_ylabel(metric)
    ax.set_title(f"Does extra prediction horizon help? {metric}")
    ax.legend(title="Rollout", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"extra_horizon_{metric}.png", dpi=200)
    plt.close(fig)


def save_cost_tradeoff_plot(runs: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if metric not in runs.columns:
        return
    cost_candidates = [
        "numeric_summary_total_runtime_s",
        "sum_runtime_s",
        "sum_elapsed_s",
        "dense_qp_work_proxy",
        "sqp_work_units",
        "configured_budget_units",
        "total_sqp_iters",
    ]
    cost = next((c for c in cost_candidates if c in runs.columns and runs[c].notna().any()), None)
    if cost is None:
        return
    data = runs.dropna(subset=[cost, metric])
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    # Plot per rollout without manually setting colours.
    if "rollout" in data.columns:
        for roll, sub in sorted(data.groupby("rollout"), key=lambda x: x[0]):
            ax.scatter(sub[cost], sub[metric], label=f"rollout={int(roll)}", alpha=0.8)
    else:
        ax.scatter(data[cost], data[metric], alpha=0.8)
    ax.set_xlabel(cost)
    ax.set_ylabel(metric)
    ax.set_title(f"Accuracy vs computational cost: {metric}")
    if data[cost].min() > 0 and data[cost].max() / max(data[cost].min(), EPS) > 100:
        ax.set_xscale("log")
    if "rollout" in data.columns:
        ax.legend(title="Rollout", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"cost_tradeoff_{metric}_vs_{cost}.png", dpi=200)
    plt.close(fig)


def save_delta_from_baseline_plot(runs: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if not {"rollout", "Np", metric}.issubset(runs.columns):
        return
    id_cols = ["rollout"]
    if "bend_deg" in runs.columns:
        id_cols.append("bend_deg")
    data = runs.dropna(subset=["rollout", "Np", metric]).copy()
    if data.empty:
        return
    rows: list[pd.DataFrame] = []
    for roll, sub in data.groupby("rollout"):
        base_np = int(roll) if (sub["Np"] == int(roll)).any() else int(sub["Np"].min())
        base = sub[sub["Np"] == base_np][id_cols + [metric]].rename(columns={metric: "baseline_metric"})
        merged = sub.merge(base, on=id_cols, how="inner")
        merged["baseline_Np"] = base_np
        merged["delta_vs_baseline"] = merged[metric] - merged["baseline_metric"]
        merged["pct_delta_vs_baseline"] = 100 * merged["delta_vs_baseline"] / (merged["baseline_metric"].abs() + EPS)
        rows.append(merged)
    if not rows:
        return
    delta = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for roll, sub in sorted(delta.groupby("rollout"), key=lambda x: x[0]):
        g = sub.groupby("Np")["delta_vs_baseline"].agg(["mean", "std", "count"]).reset_index()
        yerr = g["std"] / np.sqrt(g["count"].clip(lower=1))
        ax.errorbar(g["Np"], g["mean"], yerr=yerr, marker="o", capsize=3, label=f"rollout={int(roll)}")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Prediction horizon Np")
    ax.set_ylabel(f"Delta {metric} vs baseline horizon")
    ax.set_title(f"Improvement from longer horizon; negative is better: {metric}")
    ax.legend(title="Rollout", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"delta_vs_baseline_{metric}.png", dpi=200)
    plt.close(fig)


def make_plots(runs: pd.DataFrame, out: Path, metrics: list[str]) -> None:
    plot_dir = ensure_plot_dir(out)
    for metric in metrics:
        if metric not in runs.columns:
            continue
        save_heatmap(runs, metric, plot_dir)
        save_interaction_plot(runs, metric, plot_dir)
        save_extra_horizon_plot(runs, metric, plot_dir)
        save_delta_from_baseline_plot(runs, metric, plot_dir)
        save_cost_tradeoff_plot(runs, metric, plot_dir)

    # Compute-only diagnostic.
    cost_cols = [c for c in ["sqp_work_units", "configured_budget_units", "dense_qp_work_proxy", "total_sqp_iters"] if c in runs.columns]
    for c in cost_cols:
        save_heatmap(runs, c, plot_dir)
        save_interaction_plot(runs, c, plot_dir)


# -----------------------------
# Reporting
# -----------------------------
def write_design_diagnostics(runs: pd.DataFrame, out: Path) -> None:
    lines: list[str] = []
    lines.append("# Sweep design diagnostics\n")
    lines.append(f"Runs loaded: {len(runs)}\n")

    if {"rollout", "Np"}.issubset(runs.columns):
        cell_counts = runs.pivot_table(index="rollout", columns="Np", values="log_path", aggfunc="count", fill_value=0)
        lines.append("\nCell counts by rollout × Np:\n")
        lines.append(cell_counts.to_string())
        lines.append("\n\n")
        extra_counts = runs.groupby("rollout")["extra_horizon"].nunique(dropna=True) if "extra_horizon" in runs.columns else None
        if extra_counts is not None:
            weak = extra_counts[extra_counts < 2]
            if not weak.empty:
                lines.append(
                    "Warning: some rollout values have fewer than two prediction-horizon levels. "
                    "For those rollouts, you cannot estimate within-rollout evidence that a longer horizon helps.\n"
                )
                lines.append(weak.to_string())
                lines.append("\n")

    available = [m for m in PRIMARY_METRICS + SECONDARY_METRICS if m in runs.columns]
    lines.append("\nMetrics available in the run-level summary:\n")
    lines.extend([f"- {m}\n" for m in available])

    cost_cols = [c for c in runs.columns if re.search(r"(work|cost|runtime|elapsed|wall|cpu|iters)", c, re.I)]
    lines.append("\nCompute-cost columns/proxies available:\n")
    lines.extend([f"- {c}\n" for c in sorted(cost_cols)[:100]])

    (out / "design_diagnostics.txt").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse rollout × prediction-horizon MPC sweep logs.")
    parser.add_argument("--root", type=Path, required=True, help="Root folder containing experiment log.csv files, or a single CSV for smoke testing.")
    parser.add_argument("--out", type=Path, default=Path("analysis_rollout_prediction"), help="Output directory.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=PRIMARY_METRICS,
        help="Run-level metric columns to plot/test. Defaults to the main error metrics.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.root)

    # Standardise ordering.
    sort_cols = [c for c in ["rollout", "Np", "bend_deg", "log_path"] if c in runs.columns]
    if sort_cols:
        runs = runs.sort_values(sort_cols)

    runs.to_csv(args.out / "run_level_summary.csv", index=False)

    all_metrics = list(dict.fromkeys(args.metrics + SECONDARY_METRICS))
    group = build_group_summary(runs, all_metrics)
    if not group.empty:
        group.to_csv(args.out / "group_summary.csv", index=False)

    test_frames = []
    for metric in args.metrics:
        tests = paired_tests(runs, metric)
        if not tests.empty:
            test_frames.append(tests)
    if test_frames:
        tests_all = pd.concat(test_frames, ignore_index=True)
        tests_all.to_csv(args.out / "paired_horizon_tests.csv", index=False)

    ols_text = fit_ols_models(runs, args.metrics)
    (args.out / "ols_results.txt").write_text(ols_text)

    write_design_diagnostics(runs, args.out)
    make_plots(runs, args.out, args.metrics)

    print(f"Loaded {len(runs)} runs")
    print(f"Wrote outputs to: {args.out.resolve()}")
    print("Key files:")
    print(f"  - {args.out / 'run_level_summary.csv'}")
    if (args.out / "group_summary.csv").exists():
        print(f"  - {args.out / 'group_summary.csv'}")
    if (args.out / "paired_horizon_tests.csv").exists():
        print(f"  - {args.out / 'paired_horizon_tests.csv'}")
    print(f"  - {args.out / 'ols_results.txt'}")
    print(f"  - {args.out / 'plots'}")


if __name__ == "__main__":
    main()
