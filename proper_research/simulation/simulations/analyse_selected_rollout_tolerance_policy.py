#!/usr/bin/env python3
"""
Analyse a rollout-first / tolerance-second controller policy.

Purpose
-------
This script tests the policy structure:

    beam-Hessian / geometry risk feature -> candidate rollout
    first SQP update                     -> optional rollout veto/reduction
    final SQP tolerance                  -> accept/reject selected rollout

The key analysis here is conditional: once the risk feature chooses a rollout
(e.g. low risk -> rollout 4, medium risk -> rollout 2, high risk -> rollout 1),
we analyse SQP budget and final tolerance only inside those selected zones.
This avoids averaging over rollout/SQP combinations that the controller would
never actually execute.

Example
-------
python analyse_selected_rollout_tolerance_policy.py \
    geometry_rollout_measure_analysis/composite_score_events.csv \
    --feature risk_low_lambda_min_H_beam \
    --rollout-map low:4,medium:2,high:1 \
    --target-spike-rate 0.05

Inputs
------
A per-event CSV such as composite_score_events.csv containing columns:
    risk_low_lambda_min_H_beam
    rollout_steps
    sqp_budget
    is_next_error_spike_int or is_next_error_spike
    adapt_pred_err_xy_mm
    pred1_err_xy_mm
    sqp_du_abs_final
    sqp_du_rel_final
    sqp_du_abs_first / sqp_du_rel_first optional

Outputs
-------
CSV files and a markdown report in the output directory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_rollout_map(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in s.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        out[k.strip()] = int(v.strip())
    for required in ["low", "medium", "high"]:
        if required not in out:
            raise ValueError(f"rollout map must contain {required}:<int>")
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summary_row(g: pd.DataFrame) -> pd.Series:
    n = int(len(g))
    spikes = int(g["is_next_error_spike_int"].sum()) if n else 0
    lo, hi = wilson_ci(spikes, n)
    def safe_mean(col: str):
        return pd.to_numeric(g[col], errors="coerce").mean() if col in g.columns else np.nan
    def safe_median(col: str):
        return pd.to_numeric(g[col], errors="coerce").median() if col in g.columns else np.nan
    return pd.Series({
        "n": n,
        "spikes": spikes,
        "spike_rate": spikes / n if n else np.nan,
        "spike_ci95_low": lo,
        "spike_ci95_high": hi,
        "mean_adapt_error": safe_mean("adapt_pred_err_xy_mm"),
        "median_adapt_error": safe_median("adapt_pred_err_xy_mm"),
        "mean_pred1_error": safe_mean("pred1_err_xy_mm"),
        "median_pred1_error": safe_median("pred1_err_xy_mm"),
        "mean_next_error_delta": safe_mean("next_error_delta"),
        "median_final_abs": safe_median("sqp_du_abs_final"),
        "mean_final_abs": safe_mean("sqp_du_abs_final"),
        "median_final_rel": safe_median("sqp_du_rel_final"),
        "mean_final_rel": safe_mean("sqp_du_rel_final"),
        "median_first_abs": safe_median("sqp_du_abs_first"),
        "median_first_rel": safe_median("sqp_du_rel_first"),
        "mean_sqp_budget": safe_mean("sqp_budget"),
        "median_sqp_budget": safe_median("sqp_budget"),
    })


def ensure_spike_column(df: pd.DataFrame, spike_error_mm: float) -> pd.DataFrame:
    df = df.copy()
    if "is_next_error_spike_int" in df.columns:
        df["is_next_error_spike_int"] = pd.to_numeric(df["is_next_error_spike_int"], errors="coerce").fillna(0).astype(int)
    elif "is_next_error_spike" in df.columns:
        df["is_next_error_spike_int"] = pd.to_numeric(df["is_next_error_spike"], errors="coerce").fillna(0).astype(int)
    elif "next_error_delta" in df.columns:
        df["next_error_delta"] = pd.to_numeric(df["next_error_delta"], errors="coerce")
        df["is_next_error_spike_int"] = (df["next_error_delta"] >= spike_error_mm).astype(int)
    else:
        raise ValueError("CSV must contain is_next_error_spike_int, is_next_error_spike, or next_error_delta")
    return df


def add_risk_bins(df: pd.DataFrame, feature: str, thresholds: str | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    x = pd.to_numeric(df[feature], errors="coerce")
    valid = x.notna()
    if thresholds:
        t1, t2 = parse_float_list(thresholds)
        df.loc[valid, "selected_feature_risk_bin"] = pd.cut(
            x[valid], bins=[-np.inf, t1, t2, np.inf], labels=["low", "medium", "high"]
        ).astype(str)
    else:
        # Tertile bins create an online feature partition from the dataset.
        df.loc[valid, "selected_feature_risk_bin"] = pd.qcut(
            x[valid], q=3, labels=["low", "medium", "high"], duplicates="drop"
        ).astype(str)
        qs = x[valid].quantile([1 / 3, 2 / 3]).to_list()
        t1, t2 = float(qs[0]), float(qs[1])
    thresholds_df = pd.DataFrame([
        {"feature": feature, "bin": "low", "condition": f"{feature} <= {t1:.12g}", "upper": t1, "lower": -np.inf},
        {"feature": feature, "bin": "medium", "condition": f"{t1:.12g} < {feature} <= {t2:.12g}", "lower": t1, "upper": t2},
        {"feature": feature, "bin": "high", "condition": f"{feature} > {t2:.12g}", "lower": t2, "upper": np.inf},
    ])
    return df, thresholds_df


def tolerance_sweep(
    df: pd.DataFrame,
    group_cols: List[str],
    mode: str,
    abs_tols: List[float],
    rel_tols: List[float],
) -> pd.DataFrame:
    rows = []
    grouped = list(df.groupby(group_cols, observed=True, dropna=False))
    # Add policy-wide aggregate.
    grouped.append((tuple(["all_selected"] * len(group_cols)), df))
    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols, key)}
        n_total = len(g)
        if n_total == 0:
            continue
        if mode == "abs":
            for abs_tol in abs_tols:
                accepted = g[pd.to_numeric(g["sqp_du_abs_final"], errors="coerce") <= abs_tol]
                rows.append(make_tol_row(base, n_total, accepted, abs_tol=abs_tol, rel_tol=np.nan, tolerance_mode="absolute_only"))
        elif mode == "rel":
            for rel_tol in rel_tols:
                accepted = g[pd.to_numeric(g["sqp_du_rel_final"], errors="coerce") <= rel_tol]
                rows.append(make_tol_row(base, n_total, accepted, abs_tol=np.nan, rel_tol=rel_tol, tolerance_mode="relative_only"))
        elif mode == "combined":
            for rel_tol in rel_tols:
                for abs_tol in abs_tols:
                    accepted = g[
                        (pd.to_numeric(g["sqp_du_rel_final"], errors="coerce") <= rel_tol)
                        & (pd.to_numeric(g["sqp_du_abs_final"], errors="coerce") <= abs_tol)
                    ]
                    rows.append(make_tol_row(base, n_total, accepted, abs_tol=abs_tol, rel_tol=rel_tol, tolerance_mode="combined"))
        else:
            raise ValueError(mode)
    return pd.DataFrame(rows)


def make_tol_row(base: Dict[str, object], n_total: int, accepted: pd.DataFrame, abs_tol: float, rel_tol: float, tolerance_mode: str) -> Dict[str, object]:
    n = len(accepted)
    spikes = int(accepted["is_next_error_spike_int"].sum()) if n else 0
    lo, hi = wilson_ci(spikes, n)
    out = dict(base)
    out.update({
        "tolerance_mode": tolerance_mode,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "n_total_zone": n_total,
        "n_accept": n,
        "frac_accept": n / n_total if n_total else np.nan,
        "spikes_accept": spikes,
        "spike_rate_accept": spikes / n if n else np.nan,
        "spike_ci95_low_accept": lo,
        "spike_ci95_high_accept": hi,
        "mean_adapt_error_accept": accepted["adapt_pred_err_xy_mm"].mean() if n and "adapt_pred_err_xy_mm" in accepted else np.nan,
        "median_adapt_error_accept": accepted["adapt_pred_err_xy_mm"].median() if n and "adapt_pred_err_xy_mm" in accepted else np.nan,
        "mean_pred1_error_accept": accepted["pred1_err_xy_mm"].mean() if n and "pred1_err_xy_mm" in accepted else np.nan,
        "mean_sqp_budget_accept": accepted["sqp_budget"].mean() if n and "sqp_budget" in accepted else np.nan,
        "median_final_abs_accept": accepted["sqp_du_abs_final"].median() if n else np.nan,
        "median_final_rel_accept": accepted["sqp_du_rel_final"].median() if n else np.nan,
    })
    return out


def choose_recommendations(sweep: pd.DataFrame, group_cols: List[str], target: float, min_accept: int) -> pd.DataFrame:
    recs = []
    for key, sub in sweep.groupby(group_cols + ["tolerance_mode"], observed=True, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        base = {col: val for col, val in zip(group_cols + ["tolerance_mode"], key)}
        cand = sub[(sub["n_accept"] >= min_accept) & (sub["spike_rate_accept"] <= target)].copy()
        if len(cand):
            # Choose the most permissive threshold by max acceptance, then looser thresholds.
            cand["looseness"] = cand["rel_tol"].fillna(1.0).replace(0, 1.0) * cand["abs_tol"].fillna(1.0).replace(0, 1.0)
            row = cand.sort_values(["frac_accept", "looseness"], ascending=[False, False]).iloc[0].to_dict()
            row["recommendation_status"] = "meets_target"
        else:
            # Fall back to lowest observed spike rate, then highest acceptance.
            sub2 = sub[sub["n_accept"] >= min_accept].copy()
            if len(sub2) == 0:
                sub2 = sub.copy()
            sub2["looseness"] = sub2["rel_tol"].fillna(1.0).replace(0, 1.0) * sub2["abs_tol"].fillna(1.0).replace(0, 1.0)
            row = sub2.sort_values(["spike_rate_accept", "frac_accept"], ascending=[True, False]).iloc[0].to_dict()
            row["recommendation_status"] = "best_available_does_not_meet_target"
        recs.append(row)
    return pd.DataFrame(recs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--feature", default="risk_low_lambda_min_H_beam")
    p.add_argument("--feature-thresholds", default=None, help="Optional low_high thresholds: '<low_upper>,<medium_upper>'")
    p.add_argument("--rollout-map", default="low:4,medium:2,high:1")
    p.add_argument("--target-spike-rate", type=float, default=0.05)
    p.add_argument("--min-accept", type=int, default=10)
    p.add_argument("--spike-error-mm", type=float, default=1.0)
    p.add_argument("--abs-tols", default="0.05,0.1,0.25,0.5,0.75,1.0,1.5,2.4,3.5,5.0,7.5")
    p.add_argument("--rel-tols", default="0.1,0.2,0.3,0.45,0.6,0.75,1.0,1.5,2.0")
    args = p.parse_args()

    out_dir = args.out_dir or args.input_csv.parent / "selected_rollout_tolerance_policy"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = [args.feature, "rollout_steps", "sqp_budget", "sqp_du_abs_final", "sqp_du_rel_final"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = ensure_spike_column(df, args.spike_error_mm)
    for c in set(required + ["adapt_pred_err_xy_mm", "pred1_err_xy_mm", "next_error_delta", "sqp_du_abs_first", "sqp_du_rel_first"]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[args.feature, "rollout_steps", "sqp_budget", "sqp_du_abs_final", "sqp_du_rel_final", "is_next_error_spike_int"]).copy()

    rollout_map = parse_rollout_map(args.rollout_map)
    df, thresholds_df = add_risk_bins(df, args.feature, args.feature_thresholds)
    df["selected_candidate_rollout"] = df["selected_feature_risk_bin"].map(rollout_map).astype(float)
    df["is_selected_policy_zone"] = df["rollout_steps"].astype(float).eq(df["selected_candidate_rollout"])

    thresholds_df["candidate_rollout"] = thresholds_df["bin"].map(rollout_map)
    thresholds_df.to_csv(out_dir / "feature_risk_bin_thresholds.csv", index=False)
    df.to_csv(out_dir / "events_with_selected_policy_zone.csv", index=False)

    selected = df[df["is_selected_policy_zone"]].copy()
    group_cols = ["selected_feature_risk_bin", "selected_candidate_rollout"]

    zone_summary = selected.groupby(group_cols, observed=True).apply(summary_row).reset_index()
    all_row = summary_row(selected).to_frame().T
    all_row.insert(0, "selected_candidate_rollout", "policy")
    all_row.insert(0, "selected_feature_risk_bin", "all_selected")
    zone_summary = pd.concat([zone_summary, all_row], ignore_index=True)
    zone_summary.to_csv(out_dir / "selected_policy_zone_summary.csv", index=False)

    sqp_budget_summary = selected.groupby(group_cols + ["sqp_budget"], observed=True).apply(summary_row).reset_index()
    sqp_budget_summary.to_csv(out_dir / "selected_policy_sqp_budget_summary.csv", index=False)

    selected["sqp_class"] = pd.cut(selected["sqp_budget"], bins=[0, 2, 3, 6], labels=["sqp1_2", "sqp3", "sqp4_6"])
    sqp_class_summary = selected.groupby(group_cols + ["sqp_class"], observed=True).apply(summary_row).reset_index()
    sqp_class_summary.to_csv(out_dir / "selected_policy_sqp_class_summary.csv", index=False)

    abs_tols = parse_float_list(args.abs_tols)
    rel_tols = parse_float_list(args.rel_tols)
    abs_sweep = tolerance_sweep(selected, group_cols, "abs", abs_tols, rel_tols)
    rel_sweep = tolerance_sweep(selected, group_cols, "rel", abs_tols, rel_tols)
    combo_sweep = tolerance_sweep(selected, group_cols, "combined", abs_tols, rel_tols)
    abs_sweep.to_csv(out_dir / "selected_policy_abs_tolerance_sweep.csv", index=False)
    rel_sweep.to_csv(out_dir / "selected_policy_rel_tolerance_sweep.csv", index=False)
    combo_sweep.to_csv(out_dir / "selected_policy_combined_tolerance_sweep.csv", index=False)

    all_sweeps = pd.concat([abs_sweep, rel_sweep, combo_sweep], ignore_index=True)
    recs = choose_recommendations(all_sweeps, group_cols, args.target_spike_rate, args.min_accept)
    recs.to_csv(out_dir / "selected_policy_tolerance_recommendations.csv", index=False)

    # Fixed SQP summaries for the selected zones.
    fixed_rows = []
    for sqp in sorted(selected["sqp_budget"].dropna().unique()):
        g = selected[selected["sqp_budget"] == sqp]
        row = summary_row(g).to_dict()
        row["sqp_budget_fixed"] = int(sqp)
        fixed_rows.append(row)
    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(out_dir / "selected_policy_fixed_sqp_summary.csv", index=False)

    # Brief markdown report.
    md = []
    md.append("# Selected rollout / SQP tolerance policy analysis\n")
    md.append(f"Input CSV: `{args.input_csv}`\n")
    md.append(f"Risk feature: `{args.feature}`\n")
    md.append(f"Rollout map: `{args.rollout_map}`\n")
    md.append(f"Target spike rate: `{args.target_spike_rate:.3f}`\n")
    md.append("\n## Feature bin thresholds\n")
    md.append(thresholds_df.to_markdown(index=False))
    md.append("\n\n## Selected policy-zone summary\n")
    md.append(zone_summary.to_markdown(index=False, floatfmt=".4g"))
    md.append("\n\n## Recommended tolerances inside selected zones\n")
    keep_cols = [c for c in ["selected_feature_risk_bin", "selected_candidate_rollout", "tolerance_mode", "rel_tol", "abs_tol", "n_accept", "frac_accept", "spike_rate_accept", "spike_ci95_high_accept", "mean_adapt_error_accept", "mean_sqp_budget_accept", "recommendation_status"] if c in recs.columns]
    md.append(recs[keep_cols].to_markdown(index=False, floatfmt=".4g"))
    md.append("\n\n## Interpretation note\n")
    md.append("The selected-zone analysis is conditional on the controller's planned rollout rule. It should be used to choose the final SQP tolerance only for the rollout that would actually be selected by the risk feature.\n")
    (out_dir / "selected_rollout_tolerance_policy_report.md").write_text("\n".join(md))

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
