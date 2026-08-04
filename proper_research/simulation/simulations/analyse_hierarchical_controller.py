"""
analyse_contact_sqp4_hierarchical_controller.py

Analysis script for the deployed contact-aware SQP-full SQP4 hierarchical controller run.

Typical use:
    python analyse_contact_sqp4_hierarchical_controller.py \
        --run-root evidence_contact_sqp4_hierarchical_controller \
        --fixed-root evidence_contact_sqp4_fixed_rollouts \
        --out-dir evidence_contact_sqp4_hierarchical_analysis

What it does:
    1. Recursively finds event-level CSVs under the hierarchy run root.
    2. Filters to contact + sqp_full + SQP4 where those columns exist.
    3. Detects the runtime/selected rollout column.
    4. Reconstructs beam-Hessian risk bins using supplied thresholds.
    5. Summarises:
         - overall spike/error performance
         - performance by bend
         - performance by risk bin
         - performance by selected rollout
         - selected-rollout distribution by risk bin
         - final tolerance accepted/rejected performance if SQP final columns exist
    6. Checks whether selected rollout matches the expected policy mapping.
    7. Optionally compares hierarchy against the fixed-rollout evidence data.
    8. Saves plots and a markdown report.

This script is intentionally focused on the actual deployed setting:
    contact + sqp_full + SQP4 + hierarchical MPC enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None


EVENT_HINT_COLUMNS = {
    "adapt_pred_err_xy_mm",
    "pred1_err_xy_mm",
    "is_next_error_spike",
    "is_next_error_spike_int",
    "next_spike",
    "rollout_steps",
    "risk_low_lambda_min_H_beam",
    "lambda_min_H_beam",
    "sqp_du_abs_first",
    "solver_mode",
    "jacobian_variant",
}


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safely(path: Path, nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, nrows=nrows)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None


def looks_like_event_csv(path: Path) -> bool:
    sample = read_csv_safely(path, nrows=5)
    if sample is None or sample.empty:
        return False
    cols = set(sample.columns)
    return len(cols & EVENT_HINT_COLUMNS) >= 2


def discover_event_csvs(run_root: Path) -> list[Path]:
    if run_root.is_file() and run_root.suffix.lower() == ".csv":
        return [run_root]

    all_csvs = sorted(run_root.rglob("*.csv"))
    preferred_patterns = [
        "composite_score_events",
        "geometry_event_table",
        "events_with",
        "event_table",
        "per_step_events",
    ]
    preferred = [
        p for p in all_csvs
        if any(pattern in p.name.lower() for pattern in preferred_patterns)
        and looks_like_event_csv(p)
    ]
    if preferred:
        return preferred
    return [p for p in all_csvs if looks_like_event_csv(p)]


def load_events(root: Path) -> pd.DataFrame:
    csvs = discover_event_csvs(root)
    if not csvs:
        raise FileNotFoundError(f"No event-like CSV files found under {root}")

    frames = []
    for p in csvs:
        df = read_csv_safely(p)
        if df is None or df.empty:
            continue
        df["source_csv"] = str(p)
        if "run_name" not in df.columns:
            df["run_name"] = p.parent.name
        frames.append(df)

    if not frames:
        raise RuntimeError("No CSVs could be read.")

    out = pd.concat(frames, ignore_index=True, sort=False)
    likely_keys = [c for c in ["run_name", "step", "rollout_steps", "bend_angle_deg", "solver_mode", "jacobian_variant"] if c in out.columns]
    if likely_keys:
        out = out.drop_duplicates(subset=likely_keys + [c for c in ["adapt_pred_err_xy_mm", "pred1_err_xy_mm"] if c in out.columns])
    return out.reset_index(drop=True)


def find_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        low = c.lower()
        if any(token in low for token in ["err", "risk", "lambda", "cond", "clearance", "curvature", "angle", "rollout", "sqp", "budget", "step", "bend", "score", "norm"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_columns(df: pd.DataFrame, spike_threshold: Optional[float]) -> pd.DataFrame:
    df = coerce_numeric(df.copy())

    target_col = find_first_existing(df, [
        "is_next_error_spike_int",
        "is_next_error_spike",
        "next_spike",
        "next_error_spike",
        "is_spike_next",
    ])
    if target_col is not None:
        raw = df[target_col]
        if raw.dtype == bool:
            df["target_next_spike"] = raw.astype(int)
        elif raw.dtype == object:
            mapped = raw.astype(str).str.lower().map({
                "true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0
            })
            df["target_next_spike"] = pd.to_numeric(mapped.fillna(raw), errors="coerce")
        else:
            df["target_next_spike"] = pd.to_numeric(raw, errors="coerce")
        df["target_next_spike"] = (df["target_next_spike"] > 0).astype(float)
    else:
        err_col = find_first_existing(df, ["next_error_delta_mm", "pred1_err_xy_mm", "adapt_pred_err_xy_mm"])
        if err_col is None:
            raise ValueError("No spike label or usable error column found.")
        if spike_threshold is None:
            spike_threshold = float(df[err_col].quantile(0.90))
            print(f"Warning: no spike label found. Using {err_col} >= 90th percentile = {spike_threshold:.6g}")
        df["target_next_spike"] = (df[err_col] >= spike_threshold).astype(float)

    adapt_col = find_first_existing(df, ["adapt_pred_err_xy_mm", "adaptive_error_mm", "adapt_error_xy_mm"])
    pred1_col = find_first_existing(df, ["pred1_err_xy_mm", "one_step_error_mm", "pred1_error_xy_mm"])
    if adapt_col:
        df["adaptive_error"] = pd.to_numeric(df[adapt_col], errors="coerce")
    if pred1_col:
        df["one_step_error"] = pd.to_numeric(df[pred1_col], errors="coerce")

    rollout_col = find_first_existing(df, [
        # Preferred hierarchy/runtime rollout columns
        "runtime_rollout_steps",
        "selected_rollout_steps",
        "selected_rollout",
        "hierarchical_rollout_steps",
        "hier_rollout_steps",
        "effective_rollout_steps",
        "rollout_steps_used",
        "rollout_used",
        # Fallback fixed-rollout column
        "rollout_steps",
    ])
    if rollout_col:
        df["selected_rollout_for_analysis"] = pd.to_numeric(df[rollout_col], errors="coerce")
        df["selected_rollout_source_column"] = rollout_col
    else:
        df["selected_rollout_for_analysis"] = np.nan
        df["selected_rollout_source_column"] = ""

    budget_col = find_first_existing(df, ["sqp_budget", "N_sqp", "n_sqp", "sqp_iters", "sqp_iterations"])
    if budget_col:
        df["sqp_budget_for_filter"] = pd.to_numeric(df[budget_col], errors="coerce")

    bend_col = find_first_existing(df, ["bend_angle_deg", "bend_deg", "angle_deg"])
    if bend_col:
        df["bend_for_analysis"] = pd.to_numeric(df[bend_col], errors="coerce")

    if "cond_H_beam" in df.columns and "log10_cond_H_beam" not in df.columns:
        df["log10_cond_H_beam"] = np.log10(pd.to_numeric(df["cond_H_beam"], errors="coerce").clip(lower=1e-300))
    if "lambda_min_H_beam" in df.columns and "risk_low_lambda_min_H_beam" not in df.columns:
        df["risk_low_lambda_min_H_beam"] = -np.log10(pd.to_numeric(df["lambda_min_H_beam"], errors="coerce").clip(lower=1e-12))

    return df


def filter_contact_sqp4(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.filter_contact and "jacobian_variant" in out.columns:
        out = out[out["jacobian_variant"].astype(str).str.lower().eq("contact")]
    if args.filter_sqp_full and "solver_mode" in out.columns:
        out = out[out["solver_mode"].astype(str).str.lower().isin(["sqp_full", "sqp", "full_sqp"])]
    if args.filter_sqp_budget is not None and "sqp_budget_for_filter" in out.columns:
        out = out[np.isclose(out["sqp_budget_for_filter"], args.filter_sqp_budget)]
    return out.reset_index(drop=True)


def add_policy_bins(df: pd.DataFrame, low_medium: float, medium_high: float) -> pd.DataFrame:
    out = df.copy()
    if "risk_low_lambda_min_H_beam" not in out.columns:
        out["lambda_risk_bin"] = pd.NA
        out["expected_policy_rollout"] = np.nan
        return out

    x = pd.to_numeric(out["risk_low_lambda_min_H_beam"], errors="coerce")
    out["lambda_risk_bin"] = pd.cut(
        x,
        bins=[-np.inf, low_medium, medium_high, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype("object")

    mapping = {"low": 4, "medium": 2, "high": 1}
    out["expected_policy_rollout"] = out["lambda_risk_bin"].map(mapping).astype(float)
    out["matches_expected_policy"] = np.isclose(
        out["selected_rollout_for_analysis"],
        out["expected_policy_rollout"],
        equal_nan=False,
    )
    return out


def add_final_tolerance_flags(
    df: pd.DataFrame,
    low_medium_abs_tol: float,
    low_medium_rel_tol: float,
    high_abs_tol: float,
    high_rel_tol: float,
) -> pd.DataFrame:
    out = df.copy()
    if "sqp_du_abs_final" not in out.columns and "sqp_du_rel_final" not in out.columns:
        out["final_tolerance_available"] = False
        out["accepted_by_final_tolerance"] = pd.NA
        return out

    abs_final = pd.to_numeric(out.get("sqp_du_abs_final", pd.Series(np.nan, index=out.index)), errors="coerce")
    rel_final = pd.to_numeric(out.get("sqp_du_rel_final", pd.Series(np.nan, index=out.index)), errors="coerce")

    out["final_tolerance_available"] = abs_final.notna() | rel_final.notna()

    # Zone-dependent rule used in prior controller evidence.
    low_med = out["lambda_risk_bin"].isin(["low", "medium"])
    high = out["lambda_risk_bin"].eq("high")

    accepted = pd.Series(False, index=out.index)
    accepted.loc[low_med] = (
        (abs_final.loc[low_med] <= low_medium_abs_tol)
        & (rel_final.loc[low_med] <= low_medium_rel_tol)
    )
    accepted.loc[high] = (
        (abs_final.loc[high] <= high_abs_tol)
        & (rel_final.loc[high] <= high_rel_tol)
    )

    out["accepted_by_final_tolerance"] = accepted.where(out["final_tolerance_available"], pd.NA)
    return out


def summarise_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols:
        d = df.copy()
        d["_all"] = "all"
        group_cols = ["_all"]
    else:
        d = df.copy()

    agg_kwargs = dict(
        events=("target_next_spike", "size"),
        spikes=("target_next_spike", "sum"),
        spike_rate=("target_next_spike", "mean"),
    )
    if "adaptive_error" in d.columns:
        agg_kwargs.update(
            mean_adaptive_error=("adaptive_error", "mean"),
            median_adaptive_error=("adaptive_error", "median"),
            p90_adaptive_error=("adaptive_error", lambda x: np.nanquantile(x, 0.90)),
        )
    if "one_step_error" in d.columns:
        agg_kwargs.update(
            mean_one_step_error=("one_step_error", "mean"),
            median_one_step_error=("one_step_error", "median"),
            p90_one_step_error=("one_step_error", lambda x: np.nanquantile(x, 0.90)),
        )

    out = d.groupby(group_cols, dropna=False).agg(**agg_kwargs).reset_index()
    return out


def compare_against_fixed(hier: pd.DataFrame, fixed_root: Optional[Path], args: argparse.Namespace) -> pd.DataFrame:
    if fixed_root is None:
        return pd.DataFrame()

    fixed = load_events(fixed_root)
    fixed = ensure_columns(fixed, args.spike_threshold)
    fixed = filter_contact_sqp4(fixed, args)
    fixed = add_policy_bins(fixed, args.low_medium_threshold, args.medium_high_threshold)

    # Select counterfactual fixed rows matching expected rollout from original policy.
    fixed_match = fixed[np.isclose(fixed["selected_rollout_for_analysis"], fixed["expected_policy_rollout"], equal_nan=False)].copy()

    rows = []
    for label, data in [
        ("hierarchical_actual", hier),
        ("fixed_rows_matching_policy_rollout", fixed_match),
        ("fixed_all_rollouts", fixed),
    ]:
        row = {
            "subset": label,
            "events": len(data),
            "runs": data["run_name"].nunique() if "run_name" in data.columns else np.nan,
            "spikes": int(data["target_next_spike"].sum()) if len(data) else 0,
            "spike_rate": float(data["target_next_spike"].mean()) if len(data) else np.nan,
        }
        if "adaptive_error" in data.columns:
            row["mean_adaptive_error"] = float(data["adaptive_error"].mean())
            row["median_adaptive_error"] = float(data["adaptive_error"].median())
        if "one_step_error" in data.columns:
            row["mean_one_step_error"] = float(data["one_step_error"].mean())
            row["median_one_step_error"] = float(data["one_step_error"].median())
        rows.append(row)

    return pd.DataFrame(rows)


def plot_overall_by_bend(df: pd.DataFrame, out_dir: Path) -> None:
    if "bend_for_analysis" not in df.columns:
        return
    d = summarise_group(df, ["bend_for_analysis"]).sort_values("bend_for_analysis")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d["bend_for_analysis"], d["spike_rate"], marker="o")
    ax.set_xlabel("Bend angle [deg]")
    ax.set_ylabel("Spike rate")
    ax.set_title("Hierarchical controller spike rate by bend")
    fig.tight_layout()
    fig.savefig(out_dir / "hierarchical_spike_rate_by_bend.png", dpi=220)
    plt.close(fig)

    if "mean_adaptive_error" in d.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(d["bend_for_analysis"], d["mean_adaptive_error"], marker="o")
        ax.set_xlabel("Bend angle [deg]")
        ax.set_ylabel("Mean adaptive error [mm]")
        ax.set_title("Hierarchical controller mean adaptive error by bend")
        fig.tight_layout()
        fig.savefig(out_dir / "hierarchical_mean_adaptive_error_by_bend.png", dpi=220)
        plt.close(fig)


def plot_rollout_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "lambda_risk_bin" not in df.columns or "selected_rollout_for_analysis" not in df.columns:
        return
    d = df.dropna(subset=["lambda_risk_bin", "selected_rollout_for_analysis"]).copy()
    if d.empty:
        return

    piv = pd.crosstab(d["lambda_risk_bin"], d["selected_rollout_for_analysis"], normalize="index")
    # Stable risk-bin order if present.
    piv = piv.reindex([x for x in ["low", "medium", "high"] if x in piv.index])
    fig, ax = plt.subplots(figsize=(8, 5))
    piv.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Fraction of events")
    ax.set_xlabel("Beam-Hessian risk bin")
    ax.set_title("Selected rollout distribution by risk bin")
    ax.legend(title="Selected rollout", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "selected_rollout_distribution_by_risk_bin.png", dpi=220)
    plt.close(fig)


def plot_performance_by_rollout(df: pd.DataFrame, out_dir: Path) -> None:
    if "selected_rollout_for_analysis" not in df.columns:
        return
    d = summarise_group(df, ["selected_rollout_for_analysis"]).sort_values("selected_rollout_for_analysis")
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(d["selected_rollout_for_analysis"].astype(str), d["spike_rate"])
    ax.set_xlabel("Selected rollout")
    ax.set_ylabel("Spike rate")
    ax.set_title("Spike rate by selected rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "spike_rate_by_selected_rollout.png", dpi=220)
    plt.close(fig)

    if "mean_adaptive_error" in d.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(d["selected_rollout_for_analysis"].astype(str), d["mean_adaptive_error"])
        ax.set_xlabel("Selected rollout")
        ax.set_ylabel("Mean adaptive error [mm]")
        ax.set_title("Mean adaptive error by selected rollout")
        fig.tight_layout()
        fig.savefig(out_dir / "mean_adaptive_error_by_selected_rollout.png", dpi=220)
        plt.close(fig)


def plot_tolerance_summary(df: pd.DataFrame, out_dir: Path) -> None:
    if "accepted_by_final_tolerance" not in df.columns or df["accepted_by_final_tolerance"].isna().all():
        return
    d = df.dropna(subset=["accepted_by_final_tolerance"]).copy()
    d["tolerance_status"] = np.where(d["accepted_by_final_tolerance"].astype(bool), "accepted", "rejected")
    s = summarise_group(d, ["tolerance_status"])
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(s["tolerance_status"], s["spike_rate"])
    ax.set_xlabel("Final tolerance status")
    ax.set_ylabel("Spike rate")
    ax.set_title("Spike rate by final tolerance status")
    fig.tight_layout()
    fig.savefig(out_dir / "spike_rate_by_final_tolerance_status.png", dpi=220)
    plt.close(fig)

    if "mean_adaptive_error" in s.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(s["tolerance_status"], s["mean_adaptive_error"])
        ax.set_xlabel("Final tolerance status")
        ax.set_ylabel("Mean adaptive error [mm]")
        ax.set_title("Mean adaptive error by final tolerance status")
        fig.tight_layout()
        fig.savefig(out_dir / "mean_adaptive_error_by_final_tolerance_status.png", dpi=220)
        plt.close(fig)


def write_report(out_dir: Path, df: pd.DataFrame, summaries: dict[str, pd.DataFrame], comparison: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = []
    lines.append("# Contact SQP4 hierarchical-controller analysis\n")
    lines.append("## Dataset\n")
    lines.append(f"- Events after filtering: **{len(df)}**")
    lines.append(f"- Runs/clusters: **{df['run_name'].nunique() if 'run_name' in df.columns else 'unknown'}**")
    lines.append(f"- Next-spike events: **{int(df['target_next_spike'].sum())}**")
    lines.append(f"- Next-spike rate: **{df['target_next_spike'].mean():.4%}**")
    if "adaptive_error" in df.columns:
        lines.append(f"- Mean adaptive error: **{df['adaptive_error'].mean():.6g} mm**")
    if "one_step_error" in df.columns:
        lines.append(f"- Mean one-step error: **{df['one_step_error'].mean():.6g} mm**")

    src_col = df["selected_rollout_source_column"].dropna().astype(str).unique()
    lines.append(f"- Selected-rollout source column detected: **{src_col[0] if len(src_col) else 'not found'}**")
    lines.append(f"- Beam-Hessian thresholds used: **{args.low_medium_threshold:.6g}**, **{args.medium_high_threshold:.6g}**")

    for name, table in summaries.items():
        lines.append(f"\n## {name.replace('_', ' ').title()}\n")
        if table.empty:
            lines.append("No rows available.")
        else:
            lines.append(table.to_markdown(index=False))

    if not comparison.empty:
        lines.append("\n## Hierarchy vs fixed-rollout comparison\n")
        lines.append(comparison.to_markdown(index=False))

    if "matches_expected_policy" in df.columns:
        valid = df["matches_expected_policy"].dropna()
        if len(valid):
            lines.append("\n## Policy-match check\n")
            lines.append(f"- Fraction matching expected low→4, medium→2, high→1 rule: **{valid.mean():.4%}**")
            lines.append(
                "- If this is far below 100%, check whether the hierarchy includes additional vetoes "
                "such as SQP-probe, condition-number, geometry-score, or final-tolerance reductions."
            )

    lines.append("\n## Interpretation notes\n")
    lines.append(
        "- This script analyses the deployed hierarchy, not feature calibration.\n"
        "- Use the fixed-rollout feature script to decide whether the best rollout-selection feature is "
        "`risk_low_lambda_min_H_beam`, `log10_cond_H_beam`, or a composite geometry score.\n"
        "- The hierarchy should be judged by final spike rate, adaptive error, selected rollout distribution, "
        "and tolerance rejection behaviour.\n"
    )

    (out_dir / "hierarchical_controller_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, required=True, help="Root directory of hierarchical-controller run.")
    p.add_argument("--fixed-root", type=Path, default=None, help="Optional fixed-rollout evidence root for comparison.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    p.add_argument("--spike-threshold", type=float, default=None)
    p.add_argument("--low-medium-threshold", type=float, default=8.86645513357)
    p.add_argument("--medium-high-threshold", type=float, default=8.9859359566)
    p.add_argument("--filter-contact", action="store_true", default=True)
    p.add_argument("--no-filter-contact", dest="filter_contact", action="store_false")
    p.add_argument("--filter-sqp-full", action="store_true", default=True)
    p.add_argument("--no-filter-sqp-full", dest="filter_sqp_full", action="store_false")
    p.add_argument("--filter-sqp-budget", type=float, default=4.0)

    # Final tolerance thresholds from previous controller evidence.
    p.add_argument("--low-medium-abs-tol", type=float, default=2.4)
    p.add_argument("--low-medium-rel-tol", type=float, default=2.0)
    p.add_argument("--high-abs-tol", type=float, default=7.5)
    p.add_argument("--high-rel-tol", type=float, default=0.60)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    safe_mkdir(args.out_dir)
    safe_mkdir(args.out_dir / "plots")

    df = load_events(args.run_root)
    df = ensure_columns(df, args.spike_threshold)
    df = filter_contact_sqp4(df, args)
    if df.empty:
        raise RuntimeError("No rows left after filtering. Try disabling contact/SQP/SQP-budget filters.")

    df = add_policy_bins(df, args.low_medium_threshold, args.medium_high_threshold)
    df = add_final_tolerance_flags(
        df,
        low_medium_abs_tol=args.low_medium_abs_tol,
        low_medium_rel_tol=args.low_medium_rel_tol,
        high_abs_tol=args.high_abs_tol,
        high_rel_tol=args.high_rel_tol,
    )

    df.to_csv(args.out_dir / "filtered_hierarchical_events.csv", index=False)

    summaries = {
        "overall": summarise_group(df, []),
        "by_bend": summarise_group(df, ["bend_for_analysis"]) if "bend_for_analysis" in df.columns else pd.DataFrame(),
        "by_lambda_risk_bin": summarise_group(df, ["lambda_risk_bin"]) if "lambda_risk_bin" in df.columns else pd.DataFrame(),
        "by_selected_rollout": summarise_group(df, ["selected_rollout_for_analysis"]) if "selected_rollout_for_analysis" in df.columns else pd.DataFrame(),
        "by_risk_bin_and_selected_rollout": summarise_group(df, ["lambda_risk_bin", "selected_rollout_for_analysis"]) if "lambda_risk_bin" in df.columns else pd.DataFrame(),
    }

    if "accepted_by_final_tolerance" in df.columns and not df["accepted_by_final_tolerance"].isna().all():
        tmp = df.dropna(subset=["accepted_by_final_tolerance"]).copy()
        tmp["tolerance_status"] = np.where(tmp["accepted_by_final_tolerance"].astype(bool), "accepted", "rejected")
        summaries["by_final_tolerance_status"] = summarise_group(tmp, ["tolerance_status"])

    for name, table in summaries.items():
        table.to_csv(args.out_dir / f"{name}.csv", index=False)

    # Decision table: expected vs selected.
    if "lambda_risk_bin" in df.columns:
        decision = df.groupby(["lambda_risk_bin", "expected_policy_rollout", "selected_rollout_for_analysis"], dropna=False).agg(
            events=("target_next_spike", "size"),
            spikes=("target_next_spike", "sum"),
            spike_rate=("target_next_spike", "mean"),
            mean_adaptive_error=("adaptive_error", "mean") if "adaptive_error" in df.columns else ("target_next_spike", "mean"),
        ).reset_index()
        decision.to_csv(args.out_dir / "policy_decision_check.csv", index=False)

    comparison = compare_against_fixed(df, args.fixed_root, args)
    if not comparison.empty:
        comparison.to_csv(args.out_dir / "hierarchy_vs_fixed_comparison.csv", index=False)

    plot_overall_by_bend(df, args.out_dir / "plots")
    plot_rollout_distribution(df, args.out_dir / "plots")
    plot_performance_by_rollout(df, args.out_dir / "plots")
    plot_tolerance_summary(df, args.out_dir / "plots")

    write_report(args.out_dir, df, summaries, comparison, args)

    print(f"Done. Wrote hierarchical controller analysis to: {args.out_dir}")
    print(f"Key files:")
    print(f"  {args.out_dir / 'hierarchical_controller_report.md'}")
    print(f"  {args.out_dir / 'filtered_hierarchical_events.csv'}")
    print(f"  {args.out_dir / 'plots'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()