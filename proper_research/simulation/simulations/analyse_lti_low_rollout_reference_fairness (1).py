#!/usr/bin/env python3
"""
Reference-aligned analysis for LTI contact vs no-contact horizon/rollout sweeps.

Purpose
-------
1. Focus on low rollouts (default 1,2,3) while prediction horizon Np increases.
2. Compare contact vs no-contact using both run-level means and exact common-reference means.
3. Quantify whether larger rollout skips references and therefore makes run-level averages unfair.

Expected run layout:
    testing_lti_rollout_and_predicitions/
      bend_m90_jac_contact_plant_contact_1_lti_rollout1_Np1_sqp1/log.csv
      bend_m90_jac_no_contact_plant_contact_1_lti_rollout1_Np1_sqp1/log.csv
      ...

Outputs:
    <run_root>/lti_low_rollout_reference_fairness/
      per_step_events.csv
      reference_coverage_by_run.csv
      per_reference_summary.csv
      pairwise_common_reference_advantage.csv
      low_rollout_np_effect_run_level.csv
      low_rollout_np_effect_common_reference.csv
      lower_rollout_common_reference_set_summary.csv
      plots/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def bend_tag(b: float) -> str:
    return f"m{int(abs(b))}" if b < 0 else f"p{int(abs(b))}"


def collect_per_step_events(run_root: Path) -> pd.DataFrame:
    rows = []
    for log_csv in sorted(Path(run_root).glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name
        try:
            df = pd.read_csv(log_csv)
        except Exception as exc:
            print(f"[WARN] failed reading {log_csv}: {exc}")
            continue

        meta = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "log_csv": str(log_csv),
            "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
            "jacobian_variant": infer_jacobian_variant(run_name),
            "solver_mode": infer_solver_mode(run_name),
            "rollout_steps": parse_rollout_steps_from_run_name(run_name),
            "Np": parse_Np_from_run_name(run_name),
            "sqp_budget": parse_sqp_budget_from_run_name(run_name),
        }

        out = pd.DataFrame(index=df.index)
        for k, v in meta.items():
            out[k] = v
        out["step_idx"] = np.arange(len(df))
        out["i_ref"] = to_num(df, "i_ref")
        out["i_ref_int"] = out["i_ref"].round().astype("Int64")

        metric_cols = [
            "adapt_pred_err_xy_mm",
            "adapt_pred_err_xyz_mm",
            "pred1_err_xy_mm",
            "pred1_err_xyz_mm",
            "adapt_clearance_mm",
            "adapt_tip_vessel_angle_deg",
            "cond_H_mpc",
            "cond_H_beam",
            "u0_dL",
            "sqp_du_rel_final",
            "sqp_du_final",
        ]
        for col in metric_cols:
            out[col] = to_num(df, col)

        out["infeasible"] = to_num(df, "infeasible").fillna(0).astype(int)
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    return events


def build_reference_coverage(events: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    # Reference universe per bend: useful for seeing how much of the path each run touches.
    bend_ref_universe = (
        events.dropna(subset=["i_ref_int"])
        .groupby("bend_angle_deg")["i_ref_int"]
        .agg(lambda s: sorted(set(int(x) for x in s.dropna())))
        .to_dict()
    )

    rows = []
    group_cols = [
        "run_name", "bend_angle_deg", "jacobian_variant", "solver_mode",
        "rollout_steps", "Np", "sqp_budget",
    ]
    for key, g in events.groupby(group_cols, dropna=False):
        refs = sorted(set(int(x) for x in g["i_ref_int"].dropna()))
        if refs:
            diffs = np.diff(refs)
            span_count = max(refs) - min(refs) + 1
        else:
            diffs = np.array([])
            span_count = np.nan

        bend = key[1]
        universe = bend_ref_universe.get(bend, [])
        rows.append({
            **dict(zip(group_cols, key)),
            "n_control_steps": int(len(g)),
            "n_unique_refs_visited": int(len(refs)),
            "ref_min": min(refs) if refs else np.nan,
            "ref_max": max(refs) if refs else np.nan,
            "ref_span_count_min_to_max": span_count,
            "coverage_fraction_within_own_span": (len(refs) / span_count) if refs and span_count else np.nan,
            "n_possible_refs_in_bend_union": int(len(universe)),
            "coverage_fraction_of_bend_union": (len(refs) / len(universe)) if universe else np.nan,
            "mean_ref_jump_unique_refs": float(np.mean(diffs)) if len(diffs) else np.nan,
            "max_ref_jump_unique_refs": float(np.max(diffs)) if len(diffs) else np.nan,
            "num_ref_gaps_gt1": int(np.sum(diffs > 1)) if len(diffs) else 0,
            "frac_ref_gaps_gt1": float(np.mean(diffs > 1)) if len(diffs) else np.nan,
        })

    cov = pd.DataFrame(rows)
    cov.to_csv(out_dir / "reference_coverage_by_run.csv", index=False)
    return cov


def build_per_reference_summary(events: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    group_cols = [
        "bend_angle_deg", "jacobian_variant", "solver_mode", "rollout_steps", "Np",
        "sqp_budget", "run_name", "i_ref_int",
    ]
    metric_cols = [
        c for c in [
            "adapt_pred_err_xy_mm", "adapt_pred_err_xyz_mm",
            "pred1_err_xy_mm", "pred1_err_xyz_mm",
            "adapt_clearance_mm", "adapt_tip_vessel_angle_deg",
            "cond_H_mpc", "cond_H_beam",
        ] if c in events.columns
    ]

    agg = {c: ["mean", "median", "max", "count"] for c in metric_cols}
    ref = events.dropna(subset=["i_ref_int"]).groupby(group_cols, dropna=False).agg(agg)
    ref.columns = [f"{a}__{b}" for a, b in ref.columns]
    ref = ref.reset_index()
    ref.to_csv(out_dir / "per_reference_summary.csv", index=False)
    return ref


def build_pairwise_common_reference_advantage(ref: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Contact-vs-no-contact comparison only on exact common i_ref values.
    This removes bias caused by the two controllers visiting different references.
    """
    index_cols = ["bend_angle_deg", "solver_mode", "rollout_steps", "Np", "sqp_budget"]
    metric_bases = [
        "adapt_pred_err_xy_mm", "adapt_pred_err_xyz_mm",
        "pred1_err_xy_mm", "pred1_err_xyz_mm",
    ]

    pair_rows = []
    summary_rows = []

    for key, g in ref.groupby(index_cols, dropna=False):
        c = g[g["jacobian_variant"] == "contact"].copy()
        n = g[g["jacobian_variant"] == "no_contact"].copy()
        if c.empty or n.empty:
            continue

        c_refs = set(int(x) for x in c["i_ref_int"].dropna())
        n_refs = set(int(x) for x in n["i_ref_int"].dropna())
        common_refs = sorted(c_refs & n_refs)
        union_refs = sorted(c_refs | n_refs)
        if not common_refs:
            continue

        merged = c.merge(
            n,
            on="i_ref_int",
            suffixes=("__contact", "__no_contact"),
        )

        # Add clean metadata.
        meta = dict(zip(index_cols, key))
        for k, v in meta.items():
            merged[k] = v
        merged["n_contact_refs"] = len(c_refs)
        merged["n_no_contact_refs"] = len(n_refs)
        merged["n_common_refs"] = len(common_refs)
        merged["n_union_refs"] = len(union_refs)
        merged["ref_jaccard_contact_no_contact"] = len(common_refs) / len(union_refs) if union_refs else np.nan

        for base in metric_bases:
            c_col = f"{base}__mean__contact"
            n_col = f"{base}__mean__no_contact"
            if c_col in merged.columns and n_col in merged.columns:
                merged[f"{base}__common_ref_advantage"] = merged[n_col] - merged[c_col]

        pair_rows.append(merged)

        srow = {**meta}
        srow.update({
            "n_contact_refs": len(c_refs),
            "n_no_contact_refs": len(n_refs),
            "n_common_refs": len(common_refs),
            "n_union_refs": len(union_refs),
            "ref_jaccard_contact_no_contact": len(common_refs) / len(union_refs) if union_refs else np.nan,
            "contact_only_refs": len(c_refs - n_refs),
            "no_contact_only_refs": len(n_refs - c_refs),
        })
        for base in metric_bases:
            adv_col = f"{base}__common_ref_advantage"
            c_col = f"{base}__mean__contact"
            n_col = f"{base}__mean__no_contact"
            if adv_col in merged.columns:
                srow[f"{base}__common_ref_mean_advantage"] = float(merged[adv_col].mean())
                srow[f"{base}__common_ref_median_advantage"] = float(merged[adv_col].median())
                srow[f"{base}__common_ref_frac_contact_better"] = float((merged[adv_col] > 0).mean())
                srow[f"{base}__common_ref_contact_error"] = float(merged[c_col].mean())
                srow[f"{base}__common_ref_no_contact_error"] = float(merged[n_col].mean())
        summary_rows.append(srow)

    pair = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    summ = pd.DataFrame(summary_rows)
    pair.to_csv(out_dir / "pairwise_common_reference_advantage_per_ref.csv", index=False)
    summ.to_csv(out_dir / "pairwise_common_reference_advantage.csv", index=False)
    return pair, summ


def build_low_rollout_np_effect(pair_common: pd.DataFrame, events: pd.DataFrame, out_dir: Path, low_rollouts: list[int]) -> None:
    low = pair_common[pair_common["rollout_steps"].isin(low_rollouts)].copy()
    rows = []
    for metric in ["adapt_pred_err_xy_mm", "pred1_err_xy_mm"]:
        adv = f"{metric}__common_ref_mean_advantage"
        c = f"{metric}__common_ref_contact_error"
        n = f"{metric}__common_ref_no_contact_error"
        if adv not in low.columns:
            continue
        tab = low.groupby(["rollout_steps", "Np"], dropna=False).agg(
            n_pairs=(adv, "size"),
            mean_common_ref_advantage=(adv, "mean"),
            median_common_ref_advantage=(adv, "median"),
            frac_contact_better=(adv, lambda s: float((s > 0).mean())),
            mean_contact_error=(c, "mean"),
            mean_no_contact_error=(n, "mean"),
            mean_common_refs=("n_common_refs", "mean"),
            mean_ref_jaccard=("ref_jaccard_contact_no_contact", "mean"),
        ).reset_index()
        tab["metric"] = metric
        rows.append(tab)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out.to_csv(out_dir / "low_rollout_np_effect_common_reference.csv", index=False)


def build_common_reference_set_across_lower_rollouts(events: pd.DataFrame, out_dir: Path, low_rollouts: list[int]) -> pd.DataFrame:
    """
    For each bend and Np, find references common to all available lower-rollout runs
    and both Jacobians. This lets you compare rollout=1/2/3 on an identical reference set.
    """
    rows = []
    keys = ["bend_angle_deg", "solver_mode", "Np", "sqp_budget"]
    ev = events[events["rollout_steps"].isin(low_rollouts)].dropna(subset=["i_ref_int"])

    for key, g in ev.groupby(keys, dropna=False):
        # Need each rollout/jac run's visited refs.
        ref_sets = []
        labels = []
        for (rollout, jac), gg in g.groupby(["rollout_steps", "jacobian_variant"], dropna=False):
            refs = set(int(x) for x in gg["i_ref_int"].dropna())
            if refs:
                ref_sets.append(refs)
                labels.append((rollout, jac))
        if len(ref_sets) < 2:
            continue
        common = set.intersection(*ref_sets)
        if not common:
            continue
        common = sorted(common)

        for (rollout, jac), gg in g.groupby(["rollout_steps", "jacobian_variant"], dropna=False):
            gg_common = gg[gg["i_ref_int"].isin(common)]
            if gg_common.empty:
                continue
            rows.append({
                **dict(zip(keys, key)),
                "rollout_steps": rollout,
                "jacobian_variant": jac,
                "n_common_refs_across_lower_rollouts_and_jacs": len(common),
                "mean_adapt_pred_err_xy_mm_common_lower_rollout_refs": float(gg_common["adapt_pred_err_xy_mm"].mean()),
                "mean_pred1_err_xy_mm_common_lower_rollout_refs": float(gg_common["pred1_err_xy_mm"].mean()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "lower_rollout_common_reference_set_summary.csv", index=False)
    return out


def plot_low_rollout_lines(pair_common: pd.DataFrame, out_dir: Path, low_rollouts: list[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    low = pair_common[pair_common["rollout_steps"].isin(low_rollouts)].copy()
    bends = sorted(low["bend_angle_deg"].dropna().unique())

    for metric in ["adapt_pred_err_xy_mm", "pred1_err_xy_mm"]:
        adv = f"{metric}__common_ref_mean_advantage"
        if adv not in low.columns:
            continue
        for bend in bends:
            g_b = low[low["bend_angle_deg"] == bend]
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for rollout in low_rollouts:
                g = g_b[g_b["rollout_steps"] == rollout].sort_values("Np")
                if g.empty:
                    continue
                ax.plot(g["Np"], g[adv], marker="o", linewidth=2.0, label=f"rollout={rollout}")
            ax.axhline(0.0, linestyle=":", linewidth=1.2)
            ax.set_xlabel("Prediction horizon Np")
            ax.set_ylabel("Common-reference contact advantage\n(no-contact error - contact error)")
            ax.set_title(f"Low-rollout common-reference advantage, {metric}, bend {bend:.0f}°")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(plot_dir / f"low_rollout_common_ref_advantage_{safe_name(metric)}_bend_{bend_tag(bend)}.png", dpi=220)
            plt.close(fig)


def plot_coverage(cov: pd.DataFrame, out_dir: Path, low_rollouts: list[int]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    low = cov[cov["rollout_steps"].isin(low_rollouts)].copy()
    if low.empty:
        return
    bends = sorted(low["bend_angle_deg"].dropna().unique())
    for bend in bends:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for rollout in low_rollouts:
            g = low[(low["bend_angle_deg"] == bend) & (low["rollout_steps"] == rollout)]
            if g.empty:
                continue
            tab = g.groupby("Np")["coverage_fraction_within_own_span"].mean().reset_index()
            ax.plot(tab["Np"], tab["coverage_fraction_within_own_span"], marker="o", linewidth=2, label=f"rollout={rollout}")
        ax.set_xlabel("Prediction horizon Np")
        ax.set_ylabel("Reference coverage within own min-max span")
        ax.set_title(f"Reference coverage vs Np, bend {bend:.0f}°")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(plot_dir / f"reference_coverage_low_rollouts_bend_{bend_tag(bend)}.png", dpi=220)
        plt.close(fig)


def run_analysis(run_root: Path, out_dir: Path, low_rollouts: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = collect_per_step_events(run_root)
    if events.empty:
        print("[ANALYSIS] No log.csv files found.")
        return
    events.to_csv(out_dir / "per_step_events.csv", index=False)
    print(f"[ANALYSIS] wrote {out_dir / 'per_step_events.csv'}")

    cov = build_reference_coverage(events, out_dir)
    ref = build_per_reference_summary(events, out_dir)
    _, pair_common = build_pairwise_common_reference_advantage(ref, out_dir)
    build_low_rollout_np_effect(pair_common, events, out_dir, low_rollouts)
    build_common_reference_set_across_lower_rollouts(events, out_dir, low_rollouts)
    plot_low_rollout_lines(pair_common, out_dir, low_rollouts)
    plot_coverage(cov, out_dir, low_rollouts)

    print("[ANALYSIS] Main outputs:")
    for name in [
        "reference_coverage_by_run.csv",
        "per_reference_summary.csv",
        "pairwise_common_reference_advantage.csv",
        "pairwise_common_reference_advantage_per_ref.csv",
        "low_rollout_np_effect_common_reference.csv",
        "lower_rollout_common_reference_set_summary.csv",
        "plots",
    ]:
        print(f"  {out_dir / name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_root", type=Path, nargs="?", default=Path("testing_lti_rollout_and_predicitions"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--low-rollouts", type=str, default="1,2,3")
    args = p.parse_args()

    run_root = args.run_root.expanduser().resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = run_root / "lti_low_rollout_reference_fairness"
    else:
        out_dir = out_dir.expanduser().resolve()
    low_rollouts = [int(x.strip()) for x in args.low_rollouts.split(",") if x.strip()]
    run_analysis(run_root, out_dir, low_rollouts)


if __name__ == "__main__":
    main()
