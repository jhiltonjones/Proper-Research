#!/usr/bin/env python3
"""
Policy-oriented analysis for hierarchical rollout + SQP iteration decisions.

Input: sqp_relative_change_events.csv, including both relative and absolute SQP
change diagnostics, e.g.

    sqp_du_rel_first, sqp_du_rel_final, sqp_du_first, sqp_du_final

Goal:
    1. Test whether initial relative/absolute SQP change predicts next-step
       prediction-error spikes.
    2. Quantify when extra SQP iterations are useful.
    3. Produce tables that justify controller choices:
          - choose rollout from initial SQP mismatch/risk
          - choose max_sqp_iters from convergence benefit vs compute cost

Typical usage:

    python analyse_sqp_policy_with_absolute_change.py \
        sqp_analysis/sqp_relative_change_events.csv \
        --out-dir sqp_analysis/controller_policy_absolute_change \
        --spike-col is_next_error_spike_delta \
        --error-delta-col next_error_delta \
        --rel-accept-tol 1.0 \
        --min-n 3

The script writes CSVs, PNGs, and a Markdown report.
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


# -------------------------
# Generic utilities
# -------------------------

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def coerce_bool(s: pd.Series) -> pd.Series:
    """Robust bool conversion for bool/int/string CSV columns."""
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def finite_median(x: pd.Series) -> float:
    x = to_num(x)
    x = x[np.isfinite(x)]
    return float(np.nanmedian(x)) if len(x) else np.nan


def finite_mean(x: pd.Series) -> float:
    x = to_num(x)
    x = x[np.isfinite(x)]
    return float(np.nanmean(x)) if len(x) else np.nan


def finite_percentile(x: pd.Series, q: float) -> float:
    x = to_num(x)
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, q)) if len(x) else np.nan


def positive_p90(x: pd.Series) -> float:
    """90th percentile of positive error increases. Negative deltas clipped to zero."""
    x = to_num(x).clip(lower=0)
    x = x[np.isfinite(x)]
    return float(np.nanpercentile(x, 90)) if len(x) else np.nan


def safe_div(a: float, b: float) -> float:
    if b is None or not np.isfinite(b) or abs(b) < 1e-12:
        return np.nan
    return float(a / b)


def ensure_cols(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Input event table is missing required columns:\n  "
            + "\n  ".join(missing)
        )


def normalise_series_value(value: float, scale: float) -> float:
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 1e-12:
        return np.nan
    return float(value / scale)


# -------------------------
# Preprocessing and risk bins
# -------------------------

def load_events(
    csv_path: Path,
    *,
    spike_col: str,
    error_delta_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "rollout_steps",
        "sqp_budget",
        "sqp_du_rel_first",
        "sqp_du_rel_final",
        "sqp_du_first",
        "sqp_du_final",
        spike_col,
        error_delta_col,
    ]
    ensure_cols(df, required)

    numeric_cols = [
        "bend_angle_deg",
        "rollout_steps",
        "Np",
        "sqp_budget",
        "step_idx",
        "i_ref",
        "error_value",
        "next_error_value",
        error_delta_col,
        "N_sqp_used",
        "sqp_iters_done",
        "sqp_du_rel_first",
        "sqp_du_rel_final",
        "sqp_du_rel_max",
        "sqp_du_rel_min",
        "sqp_du_rel_drop_first_to_final",
        "sqp_du_rel_ratio_final_over_first",
        "sqp_du_first",
        "sqp_du_final",
        "sqp_du_max",
        "sqp_du_min",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_num(df[col])

    df["spike"] = coerce_bool(df[spike_col])
    df["next_error_delta_used"] = to_num(df[error_delta_col])

    # Keep rows with the quantities needed for prediction/decision analysis.
    df = df[
        df["sqp_du_rel_first"].notna()
        & df["sqp_du_first"].notna()
        & df["next_error_delta_used"].notna()
    ].copy()

    return df


def add_risk_classes(
    df: pd.DataFrame,
    *,
    rel_low: float = 1.0,
    rel_high: float = 2.0,
    rel_extreme: float = 5.0,
    abs_small: float = 5.0,
    abs_large: float = 10.0,
) -> pd.DataFrame:
    """
    Adds human-interpretable risk classes.

    Interpretation:
      - relative change detects scale-normalised mismatch/correction
      - absolute change protects against denominator artefacts in the relative norm
    """
    df = df.copy()

    rel = df["sqp_du_rel_first"]
    absu = df["sqp_du_first"]

    rel_bin = np.select(
        [
            rel < rel_low,
            (rel >= rel_low) & (rel < rel_high),
            (rel >= rel_high) & (rel < rel_extreme),
            rel >= rel_extreme,
        ],
        [
            f"rel < {rel_low:g}",
            f"{rel_low:g} <= rel < {rel_high:g}",
            f"{rel_high:g} <= rel < {rel_extreme:g}",
            f"rel >= {rel_extreme:g}",
        ],
        default="rel unknown",
    )

    abs_bin = np.select(
        [
            absu < abs_small,
            (absu >= abs_small) & (absu < abs_large),
            absu >= abs_large,
        ],
        [
            f"abs < {abs_small:g}",
            f"{abs_small:g} <= abs < {abs_large:g}",
            f"abs >= {abs_large:g}",
        ],
        default="abs unknown",
    )

    risk_class = []
    for r, a in zip(rel, absu):
        if not np.isfinite(r) or not np.isfinite(a):
            risk_class.append("unknown")
        elif r < rel_low and a < abs_large:
            risk_class.append("low_mismatch")
        elif r < rel_low and a >= abs_large:
            risk_class.append("large_absolute_only")
        elif rel_low <= r < rel_high:
            risk_class.append("moderate_relative")
        elif r >= rel_high and a < abs_small:
            risk_class.append("high_relative_small_absolute")
        elif rel_high <= r < rel_extreme and a >= abs_small:
            risk_class.append("high_relative_and_absolute")
        elif r >= rel_extreme and a >= abs_small:
            risk_class.append("extreme_relative_and_absolute")
        else:
            risk_class.append("other")

    df["rel_first_bin"] = rel_bin
    df["abs_first_bin"] = abs_bin
    df["initial_update_risk_class"] = risk_class

    return df


# -------------------------
# Main analysis tables
# -------------------------

def build_rollout_sqp_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["rollout_steps", "sqp_budget"]
    rows = []

    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n": int(len(g)),
                "spike_count": int(g["spike"].sum()),
                "spike_rate": float(g["spike"].mean()),
                "mean_error_value": finite_mean(g.get("error_value", pd.Series(np.nan, index=g.index))),
                "median_error_value": finite_median(g.get("error_value", pd.Series(np.nan, index=g.index))),
                "p90_positive_next_error_delta": positive_p90(g["next_error_delta_used"]),
                "mean_next_error_delta": finite_mean(g["next_error_delta_used"]),
                "median_rel_first": finite_median(g["sqp_du_rel_first"]),
                "median_abs_first": finite_median(g["sqp_du_first"]),
                "median_rel_final": finite_median(g["sqp_du_rel_final"]),
                "median_abs_final": finite_median(g["sqp_du_final"]),
                "median_rel_contraction_ratio": finite_median(
                    g["sqp_du_rel_final"] / g["sqp_du_rel_first"].replace(0, np.nan)
                ),
                "median_abs_contraction_ratio": finite_median(
                    g["sqp_du_final"] / g["sqp_du_first"].replace(0, np.nan)
                ),
                "mean_sqp_iters_done": finite_mean(g.get("sqp_iters_done", pd.Series(np.nan, index=g.index))),
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    return out


def build_risk_threshold_grid(
    df: pd.DataFrame,
    *,
    rel_thresholds: Iterable[float],
    abs_thresholds: Iterable[float],
) -> pd.DataFrame:
    """
    Tests flags of form:
        sqp_du_rel_first >= rel_threshold
        sqp_du_first >= abs_threshold
        both together
    for predicting next-step error spikes.
    """
    base_rate = float(df["spike"].mean())
    rows = []

    def add_row(flag_name: str, flag: pd.Series, rel_thr: float | None, abs_thr: float | None):
        flag = flag.fillna(False).astype(bool)
        spike = df["spike"].astype(bool)

        tp = int((flag & spike).sum())
        fp = int((flag & ~spike).sum())
        fn = int((~flag & spike).sum())
        tn = int((~flag & ~spike).sum())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        fpr = safe_div(fp, fp + tn)
        specificity = safe_div(tn, tn + fp)
        f1 = safe_div(2 * precision * recall, precision + recall) if np.isfinite(precision) and np.isfinite(recall) else np.nan

        rows.append(
            {
                "flag_name": flag_name,
                "rel_threshold": rel_thr,
                "abs_threshold": abs_thr,
                "n": int(len(df)),
                "base_spike_rate": base_rate,
                "flagged_count": int(flag.sum()),
                "flagged_fraction": float(flag.mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision_p_spike_given_flag": precision,
                "recall_p_flag_given_spike": recall,
                "false_positive_rate": fpr,
                "specificity": specificity,
                "f1": f1,
                "lift_vs_base_spike_rate": safe_div(precision, base_rate),
                "mean_next_error_delta_flagged": finite_mean(df.loc[flag, "next_error_delta_used"]),
                "mean_next_error_delta_unflagged": finite_mean(df.loc[~flag, "next_error_delta_used"]),
                "p90_positive_next_error_delta_flagged": positive_p90(df.loc[flag, "next_error_delta_used"]),
                "p90_positive_next_error_delta_unflagged": positive_p90(df.loc[~flag, "next_error_delta_used"]),
                "median_rel_first_flagged": finite_median(df.loc[flag, "sqp_du_rel_first"]),
                "median_abs_first_flagged": finite_median(df.loc[flag, "sqp_du_first"]),
            }
        )

    for r in rel_thresholds:
        add_row(
            f"rel_first >= {r:g}",
            df["sqp_du_rel_first"] >= r,
            r,
            None,
        )

    for a in abs_thresholds:
        add_row(
            f"abs_first >= {a:g}",
            df["sqp_du_first"] >= a,
            None,
            a,
        )

    for r in rel_thresholds:
        for a in abs_thresholds:
            add_row(
                f"rel_first >= {r:g} AND abs_first >= {a:g}",
                (df["sqp_du_rel_first"] >= r) & (df["sqp_du_first"] >= a),
                r,
                a,
            )

    return pd.DataFrame(rows).sort_values(
        ["f1", "precision_p_spike_given_flag", "recall_p_flag_given_spike"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_conditional_budget_table(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["rollout_steps", "initial_update_risk_class", "sqp_budget"]
    rows = []

    for key, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n": int(len(g)),
                "spike_count": int(g["spike"].sum()),
                "spike_rate": float(g["spike"].mean()),
                "median_error_value": finite_median(g.get("error_value", pd.Series(np.nan, index=g.index))),
                "mean_error_value": finite_mean(g.get("error_value", pd.Series(np.nan, index=g.index))),
                "p90_positive_next_error_delta": positive_p90(g["next_error_delta_used"]),
                "mean_next_error_delta": finite_mean(g["next_error_delta_used"]),
                "median_rel_first": finite_median(g["sqp_du_rel_first"]),
                "median_abs_first": finite_median(g["sqp_du_first"]),
                "median_rel_final": finite_median(g["sqp_du_rel_final"]),
                "median_abs_final": finite_median(g["sqp_du_final"]),
                "median_rel_contraction_ratio": finite_median(
                    g["sqp_du_rel_final"] / g["sqp_du_rel_first"].replace(0, np.nan)
                ),
                "median_abs_contraction_ratio": finite_median(
                    g["sqp_du_final"] / g["sqp_du_first"].replace(0, np.nan)
                ),
                "mean_sqp_iters_done": finite_mean(g.get("sqp_iters_done", pd.Series(np.nan, index=g.index))),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def add_decision_score(
    table: pd.DataFrame,
    *,
    spike_weight: float,
    error_weight: float,
    rel_final_weight: float,
    abs_final_weight: float,
    compute_penalty_per_iter: float,
) -> pd.DataFrame:
    """
    Adds a transparent scalar score for ranking SQP budgets.

    The individual components are also written out so the recommendation can be
    justified without blindly trusting the weighted score.
    """
    out = table.copy()

    err_scale = np.nanpercentile(out["p90_positive_next_error_delta"], 90)
    rel_scale = np.nanpercentile(out["median_rel_final"], 90)
    abs_scale = np.nanpercentile(out["median_abs_final"], 90)

    if not np.isfinite(err_scale) or err_scale <= 1e-12:
        err_scale = 1.0
    if not np.isfinite(rel_scale) or rel_scale <= 1e-12:
        rel_scale = 1.0
    if not np.isfinite(abs_scale) or abs_scale <= 1e-12:
        abs_scale = 1.0

    out["score_spike_component"] = spike_weight * out["spike_rate"]
    out["score_error_component"] = error_weight * out["p90_positive_next_error_delta"] / err_scale
    out["score_rel_final_component"] = rel_final_weight * out["median_rel_final"] / rel_scale
    out["score_abs_final_component"] = abs_final_weight * out["median_abs_final"] / abs_scale
    out["score_compute_component"] = compute_penalty_per_iter * (out["sqp_budget"] - 1.0)

    out["decision_score"] = out[
        [
            "score_spike_component",
            "score_error_component",
            "score_rel_final_component",
            "score_abs_final_component",
            "score_compute_component",
        ]
    ].sum(axis=1, skipna=False)

    return out


def mark_pareto_front(
    table: pd.DataFrame,
    *,
    group_cols: list[str],
    objective_cols: list[str],
) -> pd.DataFrame:
    """
    Marks non-dominated rows within each group.

    A row is dominated if another row has <= all objective values and < at least
    one objective. All objectives are minimised.
    """
    out = table.copy()
    out["pareto_nondominated"] = False

    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        values = out.loc[idx, objective_cols].apply(pd.to_numeric, errors="coerce")
        budgets = out.loc[idx, "sqp_budget"].astype(float)

        # Treat missing objectives as very bad for Pareto comparison.
        vals = values.fillna(np.inf).to_numpy(float)
        nondom = np.ones(len(idx), dtype=bool)

        for i in range(len(idx)):
            for j in range(len(idx)):
                if i == j:
                    continue
                if np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                    nondom[i] = False
                    break

        out.loc[np.array(idx)[nondom], "pareto_nondominated"] = True

    return out


def build_recommendations(
    conditional: pd.DataFrame,
    *,
    min_n: int,
    rel_accept_tol: float,
    spike_tolerance: float,
    score_tolerance: float,
) -> pd.DataFrame:
    """
    Produces an interpretable max_sqp recommendation per rollout/risk class.

    Policy preference:
      1. Ignore cells with too few samples.
      2. Prefer the lowest SQP budget that is close to the best spike rate and
         reaches acceptable median final relative change.
      3. If none meets the final-relative tolerance, choose the minimum weighted
         decision score.
    """
    group_cols = ["rollout_steps", "initial_update_risk_class"]
    rows = []

    usable = conditional[conditional["n"] >= min_n].copy()

    for key, g in usable.groupby(group_cols, dropna=False):
        g = g.sort_values("sqp_budget").copy()
        if g.empty:
            continue

        best_spike = float(g["spike_rate"].min())
        best_score = float(g["decision_score"].min())

        acceptable = g[
            (g["spike_rate"] <= best_spike + spike_tolerance)
            & (g["median_rel_final"] <= rel_accept_tol)
        ].copy()

        if len(acceptable):
            choice = acceptable.sort_values(["sqp_budget", "decision_score"]).iloc[0]
            rule = (
                f"lowest SQP budget within {spike_tolerance:g} spike-rate of best "
                f"and median final rel <= {rel_accept_tol:g}"
            )
        else:
            near_best = g[g["decision_score"] <= best_score + score_tolerance].copy()
            choice = near_best.sort_values(["sqp_budget", "decision_score"]).iloc[0]
            rule = (
                f"lowest SQP budget within {score_tolerance:g} score of best; "
                "no budget met final-relative acceptance rule"
            )

        out = dict(zip(group_cols, key))
        out.update(
            {
                "recommended_max_sqp": float(choice["sqp_budget"]),
                "recommendation_rule": rule,
                "n_for_choice": int(choice["n"]),
                "choice_spike_rate": float(choice["spike_rate"]),
                "best_spike_rate_in_group": best_spike,
                "choice_p90_positive_next_error_delta": float(choice["p90_positive_next_error_delta"]),
                "choice_median_rel_final": float(choice["median_rel_final"]),
                "choice_median_abs_final": float(choice["median_abs_final"]),
                "choice_decision_score": float(choice["decision_score"]),
                "best_decision_score_in_group": best_score,
                "available_sqp_budgets": ",".join(str(int(x)) for x in sorted(g["sqp_budget"].dropna().unique())),
            }
        )
        rows.append(out)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# -------------------------
# Controller policy synthesis
# -------------------------

def propose_controller_policy(
    recommendations: pd.DataFrame,
    *,
    desired_rollouts: Iterable[int] = (1, 5, 10),
) -> pd.DataFrame:
    """
    Produces a conservative deployment table.

    Important: this table does not pretend that an empirical recommendation for
    rollout=1 is direct evidence for rollout=10. If the exact risk_class/rollout
    cell is missing, it marks the row as a heuristic fallback.
    """
    rules = [
        # risk_class, rollout cap, fallback max_sqp, rationale
        ("low_mismatch", 10, 3, "small initial correction: long rollout can be attempted"),
        ("large_absolute_only", 5, 3, "absolute move is large despite small relative move: cap rollout"),
        ("moderate_relative", 5, 4, "moderate model mismatch: cap rollout and allow extra convergence"),
        ("high_relative_small_absolute", 1, 4, "relative warning may be denominator-driven; use short rollout"),
        ("high_relative_and_absolute", 1, 5, "true large correction: short rollout plus full convergence budget"),
        ("extreme_relative_and_absolute", 1, 5, "severe mismatch: shortest rollout and full convergence budget"),
    ]

    rows = []
    for risk_class, suggested_rollout, fallback_sqp, rationale in rules:
        exact = recommendations[
            (recommendations["initial_update_risk_class"] == risk_class)
            & (recommendations["rollout_steps"] == suggested_rollout)
        ]

        if len(exact):
            rec = exact.sort_values(["recommended_max_sqp", "choice_decision_score"]).iloc[0]
            max_sqp = rec["recommended_max_sqp"]
            evidence_level = "empirical_exact_cell"
            evidence = (
                f"rollout={rec['rollout_steps']}, n={rec['n_for_choice']}, "
                f"spike_rate={rec['choice_spike_rate']:.3f}, "
                f"median_final_rel={rec['choice_median_rel_final']:.3g}"
            )
        else:
            same_risk = recommendations[recommendations["initial_update_risk_class"] == risk_class].copy()
            max_sqp = fallback_sqp
            evidence_level = "heuristic_fallback"
            if len(same_risk):
                available = ", ".join(
                    f"rollout={int(r)}" for r in sorted(same_risk["rollout_steps"].dropna().unique())
                )
                evidence = (
                    f"no exact empirical cell for rollout={suggested_rollout}; "
                    f"available same-risk evidence: {available}; using conservative fallback"
                )
            else:
                evidence = "no empirical recommendation for this risk class; using conservative fallback"

        rows.append(
            {
                "initial_update_risk_class": risk_class,
                "suggested_rollout_steps": suggested_rollout,
                "suggested_max_sqp": float(max_sqp),
                "evidence_level": evidence_level,
                "rationale": rationale,
                "evidence": evidence,
            }
        )

    return pd.DataFrame(rows)


# -------------------------
# Plots
# -------------------------

def plot_scatter_initial_update(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    non = ~df["spike"]
    spk = df["spike"]

    ax.scatter(
        df.loc[non, "sqp_du_rel_first"],
        df.loc[non, "sqp_du_first"],
        s=24,
        alpha=0.5,
        label="no next error spike",
    )
    ax.scatter(
        df.loc[spk, "sqp_du_rel_first"],
        df.loc[spk, "sqp_du_first"],
        s=42,
        marker="x",
        linewidths=1.8,
        label="next error spike",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Initial relative SQP update, sqp_du_rel_first")
    ax.set_ylabel("Initial absolute SQP update, sqp_du_first")
    ax.set_title("Initial SQP update as a prediction-error spike signal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "initial_relative_vs_absolute_update_spikes.png", dpi=220)
    plt.close(fig)


def plot_rollout_sqp_lines(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("spike_rate", "Next-step error spike rate"),
        ("median_rel_final", "Median final relative SQP update"),
        ("median_abs_final", "Median final absolute SQP update"),
        ("p90_positive_next_error_delta", "P90 positive next error delta [mm]"),
    ]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for rollout, g in summary.groupby("rollout_steps"):
            g = g.sort_values("sqp_budget")
            ax.plot(
                g["sqp_budget"],
                g[metric],
                marker="o",
                linewidth=2.0,
                label=f"rollout={int(rollout)}",
            )
        ax.set_xlabel("Configured max SQP iterations")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs max SQP iterations")
        ax.set_xticks(sorted(summary["sqp_budget"].dropna().unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        if "rel" in metric or "abs_final" in metric:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_vs_sqp_by_rollout.png", dpi=220)
        plt.close(fig)


def plot_threshold_tradeoff(grid: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = grid[grid["flag_name"].str.contains("AND", regex=False)].copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(
        subset["recall_p_flag_given_spike"],
        subset["precision_p_spike_given_flag"],
        s=40 + 250 * subset["flagged_fraction"].fillna(0),
        alpha=0.7,
    )
    ax.set_xlabel("Recall: P(flag | spike)")
    ax.set_ylabel("Precision: P(spike | flag)")
    ax.set_title("Relative+absolute initial-update threshold trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_precision_recall_tradeoff.png", dpi=220)
    plt.close(fig)


def plot_conditional_scores(conditional: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for risk_class, g0 in conditional.groupby("initial_update_risk_class"):
        g0 = g0.sort_values(["rollout_steps", "sqp_budget"])
        if g0.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for rollout, g in g0.groupby("rollout_steps"):
            g = g.sort_values("sqp_budget")
            ax.plot(
                g["sqp_budget"],
                g["decision_score"],
                marker="o",
                linewidth=2.0,
                label=f"rollout={int(rollout)}",
            )
        ax.set_xlabel("Configured max SQP iterations")
        ax.set_ylabel("Decision score, lower is better")
        ax.set_title(f"SQP budget score for risk class: {risk_class}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(risk_class))
        fig.savefig(out_dir / f"decision_score_{name}.png", dpi=220)
        plt.close(fig)


# -------------------------
# Markdown report
# -------------------------

def write_report(
    out_path: Path,
    *,
    df: pd.DataFrame,
    rollout_summary: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    recommendations: pd.DataFrame,
    controller_policy: pd.DataFrame,
    rel_accept_tol: float,
    min_n: int,
) -> None:
    base_rate = float(df["spike"].mean())
    n = len(df)
    n_spikes = int(df["spike"].sum())

    corr_rows = []
    for col in ["sqp_du_rel_first", "sqp_du_first", "sqp_du_rel_final", "sqp_du_final"]:
        if col in df.columns:
            corr_rows.append(
                {
                    "signal": col,
                    "spearman_with_next_error_delta": df[col].corr(df["next_error_delta_used"], method="spearman"),
                    "median_if_spike": finite_median(df.loc[df["spike"], col]),
                    "median_if_no_spike": finite_median(df.loc[~df["spike"], col]),
                }
            )
    corr = pd.DataFrame(corr_rows)

    best_f1 = threshold_grid.sort_values("f1", ascending=False).head(5)
    best_precision = threshold_grid[threshold_grid["flagged_count"] >= max(3, int(0.02 * len(df)))].sort_values(
        "precision_p_spike_given_flag", ascending=False
    ).head(5)

    with open(out_path, "w") as f:
        f.write("# SQP rollout / max-iteration policy analysis\n\n")
        f.write(f"Analysed **{n}** valid control-step events with **{n_spikes}** next-step error spikes. ")
        f.write(f"Baseline spike rate: **{base_rate:.3f}**.\n\n")

        f.write("## 1. What each SQP signal means\n\n")
        f.write("- `sqp_du_rel_first`: scale-normalised initial correction. Treat this as a mismatch/risk signal.\n")
        f.write("- `sqp_du_first`: absolute initial correction. Use this to reject denominator artefacts in the relative norm.\n")
        f.write("- `sqp_du_rel_final`: final relative residual. Treat this as a convergence/acceptance signal.\n")
        f.write("- `sqp_budget`: configured maximum SQP iterations. Treat this as compute budget, not directly as a mismatch signal.\n\n")

        f.write("## 2. Signal separation: spike vs non-spike medians\n\n")
        f.write(corr.to_markdown(index=False, floatfmt=".4g"))
        f.write("\n\n")

        f.write("## 3. Rollout and SQP budget summary\n\n")
        cols = [
            "rollout_steps", "sqp_budget", "n", "spike_rate",
            "median_rel_first", "median_abs_first", "median_rel_final",
            "median_abs_final", "p90_positive_next_error_delta",
        ]
        f.write(rollout_summary[cols].to_markdown(index=False, floatfmt=".4g"))
        f.write("\n\n")

        f.write("## 4. Best initial-update risk thresholds\n\n")
        f.write("Best F1-score thresholds:\n\n")
        f.write(best_f1[[
            "flag_name", "flagged_count", "precision_p_spike_given_flag",
            "recall_p_flag_given_spike", "false_positive_rate", "f1",
            "lift_vs_base_spike_rate",
        ]].to_markdown(index=False, floatfmt=".4g"))
        f.write("\n\nHighest precision thresholds with non-trivial support:\n\n")
        f.write(best_precision[[
            "flag_name", "flagged_count", "precision_p_spike_given_flag",
            "recall_p_flag_given_spike", "false_positive_rate", "lift_vs_base_spike_rate",
        ]].to_markdown(index=False, floatfmt=".4g"))
        f.write("\n\n")

        f.write("## 5. Recommended max SQP budget by rollout and initial-update risk class\n\n")
        if recommendations.empty:
            f.write(f"No recommendation rows had at least min_n={min_n}. Lower `--min-n` or collect more runs.\n\n")
        else:
            f.write(recommendations.to_markdown(index=False, floatfmt=".4g"))
            f.write("\n\n")

        f.write("## 6. Proposed controller policy table\n\n")
        f.write(controller_policy.to_markdown(index=False, floatfmt=".4g"))
        f.write("\n\n")

        f.write("## 7. Controller interpretation\n\n")
        f.write(
            "Use the initial relative+absolute SQP update to choose rollout length. "
            "Use the final SQP relative update to decide whether the SQP solve has converged enough. "
            "Use higher max SQP iterations when the planned rollout is long or the initial update is risky, "
            "but do not expect more SQP iterations alone to remove local model mismatch.\n\n"
        )
        f.write(
            f"A practical acceptance rule to test is `sqp_du_rel_final <= {rel_accept_tol:g}` "
            "or a sufficient contraction ratio, combined with a separate initial-risk rule.\n"
        )


# -------------------------
# Main
# -------------------------

def run_analysis(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_events(
        args.events_csv.expanduser().resolve(),
        spike_col=args.spike_col,
        error_delta_col=args.error_delta_col,
    )
    df = add_risk_classes(
        df,
        rel_low=args.rel_low,
        rel_high=args.rel_high,
        rel_extreme=args.rel_extreme,
        abs_small=args.abs_small,
        abs_large=args.abs_large,
    )

    # Save cleaned event table with risk classes.
    df.to_csv(out_dir / "events_with_policy_risk_classes.csv", index=False)

    rollout_summary = build_rollout_sqp_summary(df)
    rollout_summary.to_csv(out_dir / "rollout_sqp_budget_summary.csv", index=False)

    rel_thresholds = [float(x) for x in args.rel_thresholds.split(",") if x.strip()]
    abs_thresholds = [float(x) for x in args.abs_thresholds.split(",") if x.strip()]
    threshold_grid = build_risk_threshold_grid(
        df,
        rel_thresholds=rel_thresholds,
        abs_thresholds=abs_thresholds,
    )
    threshold_grid.to_csv(out_dir / "initial_update_risk_threshold_grid.csv", index=False)

    conditional = build_conditional_budget_table(df)
    conditional = add_decision_score(
        conditional,
        spike_weight=args.spike_weight,
        error_weight=args.error_weight,
        rel_final_weight=args.rel_final_weight,
        abs_final_weight=args.abs_final_weight,
        compute_penalty_per_iter=args.compute_penalty_per_iter,
    )

    conditional = mark_pareto_front(
        conditional,
        group_cols=["rollout_steps", "initial_update_risk_class"],
        objective_cols=[
            "spike_rate",
            "p90_positive_next_error_delta",
            "median_rel_final",
            "median_abs_final",
            "sqp_budget",
        ],
    )
    conditional.to_csv(out_dir / "conditional_sqp_budget_by_risk_class.csv", index=False)

    pareto = conditional[conditional["pareto_nondominated"]].copy()
    pareto.to_csv(out_dir / "pareto_sqp_budgets_by_rollout_and_risk_class.csv", index=False)

    recommendations = build_recommendations(
        conditional,
        min_n=args.min_n,
        rel_accept_tol=args.rel_accept_tol,
        spike_tolerance=args.spike_tolerance,
        score_tolerance=args.score_tolerance,
    )
    recommendations.to_csv(out_dir / "recommended_max_sqp_by_rollout_and_risk_class.csv", index=False)

    controller_policy = propose_controller_policy(recommendations)
    controller_policy.to_csv(out_dir / "proposed_controller_policy.csv", index=False)

    # Plots.
    plot_scatter_initial_update(df, out_dir / "plots")
    plot_rollout_sqp_lines(rollout_summary, out_dir / "plots")
    plot_threshold_tradeoff(threshold_grid, out_dir / "plots")
    plot_conditional_scores(conditional, out_dir / "plots")

    write_report(
        out_dir / "controller_policy_report.md",
        df=df,
        rollout_summary=rollout_summary,
        threshold_grid=threshold_grid,
        recommendations=recommendations,
        controller_policy=controller_policy,
        rel_accept_tol=args.rel_accept_tol,
        min_n=args.min_n,
    )

    print("[ANALYSIS] Wrote:")
    for name in [
        "events_with_policy_risk_classes.csv",
        "rollout_sqp_budget_summary.csv",
        "initial_update_risk_threshold_grid.csv",
        "conditional_sqp_budget_by_risk_class.csv",
        "pareto_sqp_budgets_by_rollout_and_risk_class.csv",
        "recommended_max_sqp_by_rollout_and_risk_class.csv",
        "proposed_controller_policy.csv",
        "controller_policy_report.md",
        "plots/",
    ]:
        print(f"  {out_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse relative+absolute SQP update signals for hierarchical rollout / SQP policy design."
    )
    parser.add_argument("events_csv", type=Path, help="Path to sqp_relative_change_events.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("sqp_controller_policy_analysis"))
    parser.add_argument("--spike-col", default="is_next_error_spike_delta")
    parser.add_argument("--error-delta-col", default="next_error_delta")

    # Risk class definitions.
    parser.add_argument("--rel-low", type=float, default=1.0)
    parser.add_argument("--rel-high", type=float, default=2.0)
    parser.add_argument("--rel-extreme", type=float, default=5.0)
    parser.add_argument("--abs-small", type=float, default=5.0)
    parser.add_argument("--abs-large", type=float, default=10.0)

    # Threshold grid.
    parser.add_argument("--rel-thresholds", default="0.5,1,1.5,2,3,5,10,25")
    parser.add_argument("--abs-thresholds", default="0,2,5,8,10,12,15")

    # Recommendation parameters.
    parser.add_argument("--min-n", type=int, default=3, help="Minimum samples required in a rollout/risk/budget cell.")
    parser.add_argument("--rel-accept-tol", type=float, default=1.0)
    parser.add_argument("--spike-tolerance", type=float, default=0.03)
    parser.add_argument("--score-tolerance", type=float, default=0.02)

    # Decision score weights. These do not hide the raw metrics; they only provide a sortable default.
    parser.add_argument("--spike-weight", type=float, default=5.0)
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--rel-final-weight", type=float, default=1.0)
    parser.add_argument("--abs-final-weight", type=float, default=0.5)
    parser.add_argument("--compute-penalty-per-iter", type=float, default=0.05)

    args = parser.parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
