#!/usr/bin/env python3
"""
Per-bend fixed-effect analysis for LTI prediction-horizon effects.

Inputs are the reference-aligned CSVs produced by the previous analysis:
  - per_step_events.csv
  - pairwise_common_reference_advantage_per_ref.csv
  - pairwise_common_reference_advantage.csv
  - reference_coverage_by_run.csv

The analysis asks:
  1. Within each bend and Jacobian, does increasing Np increase error at the same references?
  2. Within each bend, does increasing Np change the contact advantage at the same references?
  3. How does reference coverage change with rollout and bend?

Outputs:
  - per_bend_within_ref_np_slopes.csv
  - per_bend_rollout_within_ref_np_slopes.csv
  - per_bend_common_reference_advantage_np_slopes.csv
  - per_bend_common_reference_contact_advantage_stats.csv
  - coverage_by_bend_rollout.csv
  - plots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


def within_slope(df: pd.DataFrame, metric: str, fe_cols: list[str], xcol: str = "Np") -> dict:
    """Within/fixed-effect slope of metric on xcol after demeaning by fe_cols.

    This is equivalent to fitting:
        metric ~ xcol + fixed_effect(fe_cols)
    but is numerically more stable than creating many dummy variables.
    """
    d = df[[metric, xcol] + fe_cols].dropna().copy()
    if len(d) < 5 or d[xcol].nunique() < 2:
        return {"n": len(d), "n_fe": np.nan, "beta_Np": np.nan, "note": "insufficient variation"}

    # Keep only fixed-effect groups where Np actually varies.
    keep = d.groupby(fe_cols)[xcol].transform("nunique") >= 2
    d = d[keep].copy()
    if len(d) < 5:
        return {"n": len(d), "n_fe": np.nan, "beta_Np": np.nan, "note": "no within-FE Np variation"}

    group = d.groupby(fe_cols)
    y = d[metric] - group[metric].transform("mean")
    x = d[xcol] - group[xcol].transform("mean")
    dd = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    dd = dd[np.abs(dd["x"]) > 1e-12]
    if len(dd) < 5:
        return {"n": len(dd), "n_fe": group.ngroups, "beta_Np": np.nan, "note": "zero residual x"}

    model = sm.OLS(dd["y"].to_numpy(), dd[["x"]].to_numpy()).fit(cov_type="HC1")
    beta = float(model.params[0])
    se = float(model.bse[0])
    z = beta / se if se > 0 else np.nan
    return {
        "n": int(len(dd)),
        "n_fe": int(group.ngroups),
        "beta_Np": beta,
        "se": se,
        "ci95_low": beta - 1.96 * se,
        "ci95_high": beta + 1.96 * se,
        "p_two_sided": float(model.pvalues[0]),
        "p_one_sided_positive": float(1 - stats.norm.cdf(z)) if np.isfinite(z) else np.nan,
        "p_one_sided_negative": float(stats.norm.cdf(z)) if np.isfinite(z) else np.nan,
        "r2_within": float(model.rsquared),
    }


def mean_ci_positive(x: pd.Series) -> dict:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = float(vals.mean())
    median = float(np.median(vals))
    if n > 1:
        se = float(vals.std(ddof=1) / np.sqrt(n))
        ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        p = float(stats.ttest_1samp(vals, 0.0, alternative="greater").pvalue)
    else:
        ci = (np.nan, np.nan)
        p = np.nan
    pos = int((vals > 0).sum())
    neg = int((vals < 0).sum())
    psign = float(stats.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue) if (pos + neg) else np.nan
    return {
        "n": int(n),
        "mean": mean,
        "median": median,
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "p_one_sided_positive": p,
        "positive_fraction": float(pos / n),
        "positive_count": pos,
        "negative_count": neg,
        "sign_test_p": psign,
    }


def analyse(input_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(input_dir / "per_step_events.csv")
    pair_ref = pd.read_csv(input_dir / "pairwise_common_reference_advantage_per_ref.csv")
    pair_common = pd.read_csv(input_dir / "pairwise_common_reference_advantage.csv")
    coverage = pd.read_csv(input_dir / "reference_coverage_by_run.csv")

    metrics = ["adapt_pred_err_xy_mm", "pred1_err_xy_mm"]

    # 1. Per-bend fixed-effect Np slopes within each Jacobian.
    rows = []
    for bend in sorted(events["bend_angle_deg"].dropna().unique()):
        for jac in ["contact", "no_contact"]:
            d = events[(events["bend_angle_deg"] == bend) & (events["jacobian_variant"] == jac)]
            for metric in metrics:
                res = within_slope(d, metric, ["rollout_steps", "i_ref_int"])
                res.update({"bend_angle_deg": bend, "jacobian_variant": jac, "metric": metric})
                rows.append(res)
    bend_slopes = pd.DataFrame(rows)
    bend_slopes.to_csv(out_dir / "per_bend_within_ref_np_slopes.csv", index=False)

    # 2. Per-bend, per-rollout slopes.
    rows = []
    for bend in sorted(events["bend_angle_deg"].dropna().unique()):
        for rollout in sorted(events["rollout_steps"].dropna().unique()):
            for jac in ["contact", "no_contact"]:
                d = events[(events["bend_angle_deg"] == bend) & (events["rollout_steps"] == rollout) & (events["jacobian_variant"] == jac)]
                for metric in metrics:
                    res = within_slope(d, metric, ["i_ref_int"])
                    res.update({"bend_angle_deg": bend, "rollout_steps": rollout, "jacobian_variant": jac, "metric": metric})
                    rows.append(res)
    rollout_slopes = pd.DataFrame(rows)
    rollout_slopes.to_csv(out_dir / "per_bend_rollout_within_ref_np_slopes.csv", index=False)

    # 3. Does Np change contact advantage within each bend?
    pr = pair_ref.copy()
    pr["bend_angle_deg"] = pr["bend_angle_deg__contact"]
    pr["rollout_steps"] = pr["rollout_steps__contact"]
    pr["Np"] = pr["Np__contact"]

    adv_metrics = [
        "adapt_pred_err_xy_mm__common_ref_advantage",
        "pred1_err_xy_mm__common_ref_advantage",
    ]
    rows = []
    for bend in sorted(pr["bend_angle_deg"].dropna().unique()):
        d = pr[pr["bend_angle_deg"] == bend]
        for metric in adv_metrics:
            res = within_slope(d, metric, ["rollout_steps", "i_ref_int"])
            res.update({"bend_angle_deg": bend, "metric": metric})
            rows.append(res)
    adv_slopes = pd.DataFrame(rows)
    adv_slopes.to_csv(out_dir / "per_bend_common_reference_advantage_np_slopes.csv", index=False)

    # 4. Static common-reference advantage by bend.
    rows = []
    for bend in sorted(pair_common["bend_angle_deg"].dropna().unique()):
        g = pair_common[pair_common["bend_angle_deg"] == bend]
        for metric in [
            "adapt_pred_err_xy_mm__common_ref_mean_advantage",
            "pred1_err_xy_mm__common_ref_mean_advantage",
        ]:
            res = mean_ci_positive(g[metric])
            res.update({"level": "pair_level", "bend_angle_deg": bend, "metric": metric})
            rows.append(res)

    for bend in sorted(pr["bend_angle_deg"].dropna().unique()):
        g = pr[pr["bend_angle_deg"] == bend]
        for metric in adv_metrics:
            res = mean_ci_positive(g[metric])
            res.update({"level": "per_reference", "bend_angle_deg": bend, "metric": metric})
            rows.append(res)

    adv_stats = pd.DataFrame(rows)
    adv_stats.to_csv(out_dir / "per_bend_common_reference_contact_advantage_stats.csv", index=False)

    # 5. Coverage summary by bend and rollout.
    cov = coverage.groupby(["bend_angle_deg", "rollout_steps"], dropna=False).agg(
        n_runs=("run_name", "size"),
        mean_unique_refs=("n_unique_refs_visited", "mean"),
        mean_coverage_union=("coverage_fraction_of_bend_union", "mean"),
        mean_ref_jump=("mean_ref_jump_unique_refs", "mean"),
        mean_frac_gaps_gt1=("frac_ref_gaps_gt1", "mean"),
    ).reset_index()
    cov.to_csv(out_dir / "coverage_by_bend_rollout.csv", index=False)

    # Plots: slopes and advantage by bend.
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for jac, marker in [("contact", "o"), ("no_contact", "s")]:
            g = bend_slopes[(bend_slopes["metric"] == metric) & (bend_slopes["jacobian_variant"] == jac)].sort_values("bend_angle_deg")
            ax.errorbar(g["bend_angle_deg"], g["beta_Np"],
                        yerr=[g["beta_Np"] - g["ci95_low"], g["ci95_high"] - g["beta_Np"]],
                        marker=marker, capsize=4, linewidth=2, label=jac)
        ax.axhline(0, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Bend angle [deg]")
        ax.set_ylabel("Within-reference Np slope [mm per Np]")
        ax.set_title(f"Per-bend fixed-effect Np slope: {metric}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"per_bend_np_slope_{metric}.png", dpi=220)
        plt.close(fig)

    for metric in adv_metrics:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        g = adv_slopes[adv_slopes["metric"] == metric].sort_values("bend_angle_deg")
        ax.errorbar(g["bend_angle_deg"], g["beta_Np"],
                    yerr=[g["beta_Np"] - g["ci95_low"], g["ci95_high"] - g["beta_Np"]],
                    marker="o", capsize=4, linewidth=2)
        ax.axhline(0, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Bend angle [deg]")
        ax.set_ylabel("Within-reference Np slope of contact advantage [mm per Np]")
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"per_bend_advantage_np_slope_{metric}.png", dpi=220)
        plt.close(fig)

    print(f"[ANALYSIS] Wrote outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Directory containing the reference-aligned CSV outputs")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else input_dir / "per_bend_fixed_effect_analysis"
    analyse(input_dir, out_dir)


if __name__ == "__main__":
    main()
