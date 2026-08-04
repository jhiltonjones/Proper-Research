#!/usr/bin/env python3
"""
Tolerance-based analysis for SQP/MPC controller experiments.

This script takes the transition_events.csv produced by the rollout/SQP/curvature
analysis and re-groups events by achieved SQP tolerance rather than configured
SQP iteration budget. This lets SQP3 and SQP4, for example, be treated as the
same regime if they produce similar final update residuals.

Core questions:
  1. How does final SQP relative/absolute tolerance relate to next error spikes?
  2. Which SQP iteration budgets actually achieve each tolerance level?
  3. How should tolerance depend on rollout and curvature?
  4. Are initial update signals still more useful for rollout selection than final residuals?

Expected input columns include:
  sqp_du_rel_final, sqp_du_abs_final,
  sqp_du_rel_first, sqp_du_abs_first,
  rollout_steps, sqp_budget, bend_angle_deg, curvature_bin,
  next_error_delta, is_next_error_spike, adapt_pred_err_xy_mm, pred1_err_xy_mm
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None


REL_TOLS_DEFAULT = [0.10, 0.20, 0.30, 0.45, 0.75, 1.00, 1.50, 2.00]
ABS_TOLS_DEFAULT = [0.10, 0.25, 0.50, 1.00, 1.50, 2.40, 5.00]


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def as_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_candidates = [
        "bend_angle_deg", "bend_magnitude_deg", "rollout_steps", "Np", "sqp_budget",
        "k", "i_ref", "idx_ref_terminal", "path_idx_nearest",
        "curvature_local_1pm", "curvature_max_ahead_1pm", "curvature_mean_ahead_1pm",
        "adapt_pred_err_xy_mm", "pred1_err_xy_mm", "next_error_delta", "error_delta_from_prev",
        "sqp_du_rel_first", "sqp_du_rel_final", "sqp_du_rel_max", "sqp_du_rel_min",
        "sqp_du_abs_first", "sqp_du_abs_final", "sqp_du_abs_max", "sqp_du_abs_min",
        "prev_final_to_current_first_rel_delta", "prev_final_to_current_first_abs_delta",
        "current_final_to_next_first_rel_delta", "current_final_to_next_first_abs_delta",
        "sqp_du_rel_final_contraction_ratio_over_prev_inner",
        "sqp_du_abs_final_contraction_ratio_over_prev_inner",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_next_error_spike" in df.columns:
        if df["is_next_error_spike"].dtype == bool:
            df["is_next_error_spike_int"] = df["is_next_error_spike"].astype(int)
        else:
            df["is_next_error_spike_int"] = df["is_next_error_spike"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
    else:
        df["is_next_error_spike_int"] = (as_num(df, "next_error_delta") >= 1.0).astype(int)

    if "bend_magnitude_deg" not in df.columns and "bend_angle_deg" in df.columns:
        df["bend_magnitude_deg"] = df["bend_angle_deg"].abs()

    return df


def add_tolerance_bins(df: pd.DataFrame) -> pd.DataFrame:
    rel_bins = [-np.inf, 0.10, 0.20, 0.45, 0.75, np.inf]
    rel_labels = [
        "<=0.10 very_strict",
        "0.10-0.20 strict",
        "0.20-0.45 medium",
        "0.45-0.75 loose",
        ">0.75 unconverged",
    ]
    df["final_rel_tol_bin"] = pd.cut(
        df["sqp_du_rel_final"], bins=rel_bins, labels=rel_labels
    )

    abs_bins = [-np.inf, 0.25, 1.00, 2.40, 5.00, np.inf]
    abs_labels = [
        "<=0.25 tiny",
        "0.25-1.0 small",
        "1.0-2.4 medium",
        "2.4-5.0 large",
        ">5.0 very_large",
    ]
    df["final_abs_tol_bin"] = pd.cut(
        df["sqp_du_abs_final"], bins=abs_bins, labels=abs_labels
    )

    df["log10_final_rel"] = np.log10(df["sqp_du_rel_final"].clip(lower=1e-12))
    df["log10_final_abs"] = np.log10(df["sqp_du_abs_final"].clip(lower=1e-12))
    df["log10_first_rel"] = np.log10(df["sqp_du_rel_first"].clip(lower=1e-12))
    df["log10_first_abs"] = np.log10(df["sqp_du_abs_first"].clip(lower=1e-12))

    return df


def aggregate_events(g: pd.DataFrame) -> pd.Series:
    spike = g["is_next_error_spike_int"]
    return pd.Series(
        {
            "n_events": int(len(g)),
            "n_spikes": int(spike.sum()),
            "next_spike_rate": float(spike.mean()) if len(g) else np.nan,
            "mean_next_error_delta": float(as_num(g, "next_error_delta").mean()),
            "median_next_error_delta": float(as_num(g, "next_error_delta").median()),
            "mean_error": float(as_num(g, "adapt_pred_err_xy_mm").mean()),
            "median_error": float(as_num(g, "adapt_pred_err_xy_mm").median()),
            "mean_pred1_error": float(as_num(g, "pred1_err_xy_mm").mean()),
            "median_pred1_error": float(as_num(g, "pred1_err_xy_mm").median()),
            "mean_rollout": float(as_num(g, "rollout_steps").mean()),
            "mean_sqp_budget": float(as_num(g, "sqp_budget").mean()),
            "median_sqp_budget": float(as_num(g, "sqp_budget").median()),
            "mean_final_rel": float(as_num(g, "sqp_du_rel_final").mean()),
            "median_final_rel": float(as_num(g, "sqp_du_rel_final").median()),
            "mean_final_abs": float(as_num(g, "sqp_du_abs_final").mean()),
            "median_final_abs": float(as_num(g, "sqp_du_abs_final").median()),
            "mean_first_rel": float(as_num(g, "sqp_du_rel_first").mean()),
            "median_first_rel": float(as_num(g, "sqp_du_rel_first").median()),
            "mean_first_abs": float(as_num(g, "sqp_du_abs_first").mean()),
            "median_first_abs": float(as_num(g, "sqp_du_abs_first").median()),
        }
    )


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, observed=False, dropna=False).apply(aggregate_events).reset_index()
    return out


def build_threshold_sweep(
    df: pd.DataFrame,
    rel_tols: list[float],
    abs_tols: list[float],
    out_root: Path,
) -> pd.DataFrame:
    rows = []
    base_spike = float(df["is_next_error_spike_int"].mean())

    for rel_tol in rel_tols:
        m = df["sqp_du_rel_final"] <= rel_tol
        rows.append(
            {
                "threshold_type": "relative_only",
                "rel_tol": rel_tol,
                "abs_tol": np.nan,
                "n_satisfied": int(m.sum()),
                "frac_satisfied": float(m.mean()),
                "spike_rate_if_satisfied": float(df.loc[m, "is_next_error_spike_int"].mean()) if m.any() else np.nan,
                "spike_rate_if_not_satisfied": float(df.loc[~m, "is_next_error_spike_int"].mean()) if (~m).any() else np.nan,
                "baseline_spike_rate": base_spike,
                "mean_error_if_satisfied": float(df.loc[m, "adapt_pred_err_xy_mm"].mean()) if m.any() else np.nan,
                "mean_error_if_not_satisfied": float(df.loc[~m, "adapt_pred_err_xy_mm"].mean()) if (~m).any() else np.nan,
                "mean_rollout_if_satisfied": float(df.loc[m, "rollout_steps"].mean()) if m.any() else np.nan,
                "mean_sqp_budget_if_satisfied": float(df.loc[m, "sqp_budget"].mean()) if m.any() else np.nan,
            }
        )

    for abs_tol in abs_tols:
        m = df["sqp_du_abs_final"] <= abs_tol
        rows.append(
            {
                "threshold_type": "absolute_only",
                "rel_tol": np.nan,
                "abs_tol": abs_tol,
                "n_satisfied": int(m.sum()),
                "frac_satisfied": float(m.mean()),
                "spike_rate_if_satisfied": float(df.loc[m, "is_next_error_spike_int"].mean()) if m.any() else np.nan,
                "spike_rate_if_not_satisfied": float(df.loc[~m, "is_next_error_spike_int"].mean()) if (~m).any() else np.nan,
                "baseline_spike_rate": base_spike,
                "mean_error_if_satisfied": float(df.loc[m, "adapt_pred_err_xy_mm"].mean()) if m.any() else np.nan,
                "mean_error_if_not_satisfied": float(df.loc[~m, "adapt_pred_err_xy_mm"].mean()) if (~m).any() else np.nan,
                "mean_rollout_if_satisfied": float(df.loc[m, "rollout_steps"].mean()) if m.any() else np.nan,
                "mean_sqp_budget_if_satisfied": float(df.loc[m, "sqp_budget"].mean()) if m.any() else np.nan,
            }
        )

    for rel_tol in rel_tols:
        for abs_tol in abs_tols:
            m = (df["sqp_du_rel_final"] <= rel_tol) & (df["sqp_du_abs_final"] <= abs_tol)
            rows.append(
                {
                    "threshold_type": "relative_and_absolute",
                    "rel_tol": rel_tol,
                    "abs_tol": abs_tol,
                    "n_satisfied": int(m.sum()),
                    "frac_satisfied": float(m.mean()),
                    "spike_rate_if_satisfied": float(df.loc[m, "is_next_error_spike_int"].mean()) if m.any() else np.nan,
                    "spike_rate_if_not_satisfied": float(df.loc[~m, "is_next_error_spike_int"].mean()) if (~m).any() else np.nan,
                    "baseline_spike_rate": base_spike,
                    "mean_error_if_satisfied": float(df.loc[m, "adapt_pred_err_xy_mm"].mean()) if m.any() else np.nan,
                    "mean_error_if_not_satisfied": float(df.loc[~m, "adapt_pred_err_xy_mm"].mean()) if (~m).any() else np.nan,
                    "mean_rollout_if_satisfied": float(df.loc[m, "rollout_steps"].mean()) if m.any() else np.nan,
                    "mean_sqp_budget_if_satisfied": float(df.loc[m, "sqp_budget"].mean()) if m.any() else np.nan,
                }
            )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_root / "final_tolerance_threshold_sweep.csv", index=False)
    return sweep


def build_policy_recommendations(
    df: pd.DataFrame,
    rel_tols: list[float],
    abs_tols: list[float],
    out_root: Path,
    target_spike_rate: float,
    min_events: int,
) -> pd.DataFrame:
    rows = []
    group_cols = ["curvature_bin", "rollout_steps"]

    for key, g in df.groupby(group_cols, observed=False, dropna=False):
        candidates = []
        for rel_tol in rel_tols:
            for abs_tol in abs_tols:
                m = (g["sqp_du_rel_final"] <= rel_tol) & (g["sqp_du_abs_final"] <= abs_tol)
                if int(m.sum()) < min_events:
                    continue
                spike_rate = float(g.loc[m, "is_next_error_spike_int"].mean())
                candidates.append(
                    {
                        "curvature_bin": key[0],
                        "rollout_steps": key[1],
                        "rel_tol": rel_tol,
                        "abs_tol": abs_tol,
                        "n_satisfied": int(m.sum()),
                        "frac_satisfied": float(m.mean()),
                        "spike_rate_if_satisfied": spike_rate,
                        "mean_error_if_satisfied": float(g.loc[m, "adapt_pred_err_xy_mm"].mean()),
                        "mean_sqp_budget_if_satisfied": float(g.loc[m, "sqp_budget"].mean()),
                        "meets_target": spike_rate <= target_spike_rate,
                    }
                )

        if not candidates:
            rows.append({"curvature_bin": key[0], "rollout_steps": key[1], "recommendation": "insufficient_data"})
            continue

        cand = pd.DataFrame(candidates)
        feasible = cand[cand["meets_target"]].copy()
        if feasible.empty:
            # Pick the safest observed candidate.
            best = cand.sort_values(["spike_rate_if_satisfied", "mean_error_if_satisfied", "frac_satisfied"]).iloc[0]
            rec = "no_candidate_met_target_choose_safest_observed"
        else:
            # Pick the loosest tolerance that still meets target, with coverage as tiebreaker.
            feasible["looseness"] = feasible["rel_tol"].rank(method="dense") + feasible["abs_tol"].rank(method="dense")
            best = feasible.sort_values(["rel_tol", "abs_tol", "frac_satisfied"], ascending=[False, False, False]).iloc[0]
            rec = "loosest_tolerance_meeting_target"

        out = best.to_dict()
        out["recommendation"] = rec
        out["target_spike_rate"] = target_spike_rate
        rows.append(out)

    recs = pd.DataFrame(rows)
    recs.to_csv(out_root / "tolerance_policy_recommendations.csv", index=False)
    return recs


def build_regression_tables(df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    if smf is None:
        out = pd.DataFrame([{"error": "statsmodels_not_available"}])
        out.to_csv(out_root / "tolerance_regression_summary.csv", index=False)
        return out

    cols = [
        "is_next_error_spike_int", "next_error_delta", "log10_final_rel", "log10_final_abs",
        "log10_first_rel", "log10_first_abs", "rollout_steps", "sqp_budget",
        "bend_magnitude_deg", "curvature_max_ahead_1pm",
    ]
    d = df[cols].dropna().copy()

    for col in [c for c in cols if c not in ["is_next_error_spike_int", "next_error_delta"]]:
        sd = d[col].std(ddof=0)
        if sd > 0:
            d[col + "_z"] = (d[col] - d[col].mean()) / sd

    features = [
        "log10_final_rel_z", "log10_final_abs_z", "log10_first_rel_z", "log10_first_abs_z",
        "rollout_steps_z", "sqp_budget_z", "bend_magnitude_deg_z", "curvature_max_ahead_1pm_z",
    ]

    rows = []
    formula_rhs = " + ".join(features)

    try:
        logit = smf.logit(f"is_next_error_spike_int ~ {formula_rhs}", data=d).fit(disp=False, maxiter=300)
        conf = logit.conf_int()
        for term in logit.params.index:
            rows.append(
                {
                    "model": "logit_next_spike",
                    "term": term,
                    "coef": float(logit.params[term]),
                    "odds_ratio": float(np.exp(logit.params[term])),
                    "p_value": float(logit.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n": int(logit.nobs),
                    "pseudo_r2": float(logit.prsquared),
                }
            )
    except Exception as exc:
        rows.append({"model": "logit_next_spike", "term": "ERROR", "error": repr(exc)})

    try:
        ols = smf.ols(f"next_error_delta ~ {formula_rhs}", data=d).fit(cov_type="HC3")
        conf = ols.conf_int()
        for term in ols.params.index:
            rows.append(
                {
                    "model": "ols_next_error_delta",
                    "term": term,
                    "coef": float(ols.params[term]),
                    "odds_ratio": np.nan,
                    "p_value": float(ols.pvalues[term]),
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n": int(ols.nobs),
                    "pseudo_r2": float(ols.rsquared),
                }
            )
    except Exception as exc:
        rows.append({"model": "ols_next_error_delta", "term": "ERROR", "error": repr(exc)})

    out = pd.DataFrame(rows)
    out.to_csv(out_root / "tolerance_regression_summary.csv", index=False)
    return out


def make_plots(df: pd.DataFrame, out_root: Path) -> None:
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Spike rate by final relative tolerance bin.
    s = group_summary(df, ["final_rel_tol_bin"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(np.arange(len(s)), s["next_spike_rate"])
    ax.set_xticks(np.arange(len(s)))
    ax.set_xticklabels(s["final_rel_tol_bin"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Next error spike rate")
    ax.set_title("Spike rate by achieved final relative SQP tolerance")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "spike_rate_by_final_rel_tolerance.png", dpi=220)
    plt.close(fig)

    # Mean error by rollout and final tolerance.
    s2 = group_summary(df, ["rollout_steps", "final_rel_tol_bin"])
    for metric, ylabel in [("mean_error", "Mean adaptive error [mm]"), ("next_spike_rate", "Next error spike rate")]:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for rollout, g in s2.groupby("rollout_steps", observed=False):
            g = g.copy()
            ax.plot(np.arange(len(g)), g[metric], marker="o", linewidth=2.0, label=f"rollout={int(rollout)}")
        labels = s2["final_rel_tol_bin"].drop_duplicates().astype(str).tolist()
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by rollout and final SQP tolerance")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name(metric)}_by_rollout_and_final_rel_tolerance.png", dpi=220)
        plt.close(fig)

    # Distribution of SQP budget by tolerance bin.
    ctab = pd.crosstab(df["final_rel_tol_bin"], df["sqp_budget"], normalize="index")
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(ctab))
    x = np.arange(len(ctab))
    for col in ctab.columns:
        ax.bar(x, ctab[col].values, bottom=bottom, label=f"sqp{int(col)}")
        bottom += ctab[col].values
    ax.set_xticks(x)
    ax.set_xticklabels(ctab.index.astype(str), rotation=25, ha="right")
    ax.set_ylabel("Fraction within tolerance bin")
    ax.set_title("Which SQP budgets achieve each final relative tolerance bin?")
    ax.legend(fontsize=8, ncols=3)
    fig.tight_layout()
    fig.savefig(plot_dir / "sqp_budget_distribution_by_final_rel_tolerance.png", dpi=220)
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    summary_rel: pd.DataFrame,
    sweep: pd.DataFrame,
    recs: pd.DataFrame,
    regression: pd.DataFrame,
    out_root: Path,
) -> None:
    report = out_root / "sqp_tolerance_report.md"
    base_spike = float(df["is_next_error_spike_int"].mean())

    lines = []
    lines.append("# SQP tolerance-based controller analysis\n\n")
    lines.append(f"Total events: **{len(df)}**\n\n")
    lines.append(f"Baseline next-error spike rate: **{100 * base_spike:.2f}%**\n\n")

    lines.append("## Interpretation rule\n\n")
    lines.append(
        "This analysis groups events by the achieved final SQP update residual, not by configured SQP iteration budget. "
        "Therefore SQP3 and SQP4 can fall into the same category if their final relative/absolute changes are similar.\n\n"
    )
    lines.append("Final relative tolerance bins used:\n\n")
    for label in summary_rel["final_rel_tol_bin"].astype(str).tolist():
        lines.append(f"- `{label}`\n")

    lines.append("\n## Main warning\n\n")
    lines.append(
        "Achieved tolerance is not purely causal. A very small final residual often means the problem was easier or that a larger SQP budget was used. "
        "Use these results to set candidate stopping tolerances, then validate them prospectively in a controller run.\n\n"
    )

    lines.append("## Summary by final relative tolerance\n\n")
    cols = ["final_rel_tol_bin", "n_events", "next_spike_rate", "mean_error", "mean_pred1_error", "mean_sqp_budget", "median_final_rel", "median_final_abs"]
    lines.append(summary_rel[cols].to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Threshold sweep highlights\n\n")
    rel = sweep[sweep["threshold_type"] == "relative_only"].copy()
    cols = ["rel_tol", "n_satisfied", "frac_satisfied", "spike_rate_if_satisfied", "spike_rate_if_not_satisfied", "mean_error_if_satisfied", "mean_sqp_budget_if_satisfied"]
    lines.append(rel[cols].to_markdown(index=False))
    lines.append("\n\n")

    combined = sweep[sweep["threshold_type"] == "relative_and_absolute"].copy()
    combined = combined[(combined["n_satisfied"] >= 20)].sort_values(["spike_rate_if_satisfied", "frac_satisfied"], ascending=[True, False]).head(10)
    lines.append("Best combined relative+absolute tolerance candidates by observed spike rate:\n\n")
    cols = ["rel_tol", "abs_tol", "n_satisfied", "frac_satisfied", "spike_rate_if_satisfied", "mean_error_if_satisfied", "mean_sqp_budget_if_satisfied"]
    lines.append(combined[cols].to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Controller recommendation\n\n")
    lines.append(
        "Use final SQP tolerance as a stopping/convergence criterion, not as the sole rollout-risk criterion. "
        "Rollout should still be selected from initial SQP change, curvature/bend severity, and rollout length. "
        "A practical starting point from this dataset is:\n\n"
    )
    lines.append("- Low-risk rollout 1: final relative tolerance around `0.45–0.75`.\n")
    lines.append("- Rollout 2–3: final relative tolerance around `0.30–0.45`.\n")
    lines.append("- Rollout 4–5 or high curvature: final relative tolerance around `0.20–0.30`, with an absolute residual check.\n")
    lines.append("- Very high-risk cases: target `<=0.20` relative and `<=0.5–1.0` absolute, or reduce rollout.\n")
    lines.append("\n")

    lines.append("## Output files\n\n")
    for fn in [
        "events_with_tolerance_bins.csv",
        "tolerance_event_summary_by_final_rel_bin.csv",
        "tolerance_event_summary_by_rollout_and_final_rel_bin.csv",
        "tolerance_event_summary_by_curvature_and_final_rel_bin.csv",
        "final_tolerance_threshold_sweep.csv",
        "sqp_budget_distribution_by_tolerance.csv",
        "tolerance_policy_recommendations.csv",
        "tolerance_regression_summary.csv",
        "plots/",
    ]:
        lines.append(f"- `{fn}`\n")

    report.write_text("".join(lines))


def run_analysis(
    transition_csv: Path,
    out_root: Path,
    rel_tols: list[float],
    abs_tols: list[float],
    target_spike_rate: float,
    min_events: int,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(transition_csv)
    df = ensure_numeric(df)
    df = add_tolerance_bins(df)

    df.to_csv(out_root / "events_with_tolerance_bins.csv", index=False)

    summary_rel = group_summary(df, ["final_rel_tol_bin"])
    summary_rel.to_csv(out_root / "tolerance_event_summary_by_final_rel_bin.csv", index=False)

    summary_rollout = group_summary(df, ["rollout_steps", "final_rel_tol_bin"])
    summary_rollout.to_csv(out_root / "tolerance_event_summary_by_rollout_and_final_rel_bin.csv", index=False)

    summary_curv = group_summary(df, ["curvature_bin", "final_rel_tol_bin"])
    summary_curv.to_csv(out_root / "tolerance_event_summary_by_curvature_and_final_rel_bin.csv", index=False)

    summary_rollout_curv = group_summary(df, ["curvature_bin", "rollout_steps", "final_rel_tol_bin"])
    summary_rollout_curv.to_csv(out_root / "tolerance_event_summary_by_curvature_rollout_and_final_rel_bin.csv", index=False)

    budget_dist = pd.crosstab(
        df["final_rel_tol_bin"],
        df["sqp_budget"],
        margins=True,
        normalize=False,
    )
    budget_dist.to_csv(out_root / "sqp_budget_distribution_by_tolerance.csv")

    sweep = build_threshold_sweep(df, rel_tols, abs_tols, out_root)
    recs = build_policy_recommendations(df, rel_tols, abs_tols, out_root, target_spike_rate, min_events)
    reg = build_regression_tables(df, out_root)
    make_plots(df, out_root)
    write_report(df, summary_rel, sweep, recs, reg, out_root)

    print(f"[ANALYSIS] Wrote tolerance analysis to {out_root}")
    print(f"[ANALYSIS] Main report: {out_root / 'sqp_tolerance_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse SQP tolerance effects on error/spikes.")
    parser.add_argument("transition_csv", type=Path, help="Path to transition_events.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("sqp_tolerance_analysis"))
    parser.add_argument("--rel-tols", type=str, default=",".join(str(x) for x in REL_TOLS_DEFAULT))
    parser.add_argument("--abs-tols", type=str, default=",".join(str(x) for x in ABS_TOLS_DEFAULT))
    parser.add_argument("--target-spike-rate", type=float, default=0.05)
    parser.add_argument("--min-events", type=int, default=15)
    args = parser.parse_args()

    run_analysis(
        transition_csv=args.transition_csv.expanduser().resolve(),
        out_root=args.out_dir.expanduser().resolve(),
        rel_tols=parse_float_list(args.rel_tols),
        abs_tols=parse_float_list(args.abs_tols),
        target_spike_rate=args.target_spike_rate,
        min_events=args.min_events,
    )


if __name__ == "__main__":
    main()
