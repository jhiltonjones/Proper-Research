#!/usr/bin/env python3
"""
Analyse when contact-aware vs no-contact LTI Jacobians are beneficial as
prediction horizon Np and rollout length increase.

Expected run layout:

    testing_lti_rollout_and_predicitions/
      bend_m90_jac_contact_plant_contact_1_lti_rollout1_Np1_sqp1/
        log.csv
      bend_m90_jac_no_contact_plant_contact_1_lti_rollout1_Np1_sqp1/
        log.csv
      ...

Core question:
    Is the contact-aware Jacobian better for short local horizons, but less
    distinguishable from the no-contact Jacobian as LTI prediction horizon /
    rollout length increase?

Outputs:
    <run_root>/lti_contact_horizon_analysis/
      lti_rollout_np_run_summary.csv
      lti_contact_pairwise_advantage.csv
      lti_contact_advantage_regime_summary.csv
      lti_equivalence_summary.csv
      lti_controller_interpretation_table.csv
      lti_contact_horizon_report.md
      plots/
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

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
    m = re.search(r"_sqp(\d+)(?:$|_)", run_name)
    return float(m.group(1)) if m else np.nan


def bend_tag(bend: float) -> str:
    if not np.isfinite(bend):
        return "bend_unknown"
    return f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def safe_mean(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    return float(v.mean()) if v.notna().any() else np.nan


def safe_median(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    return float(v.median()) if v.notna().any() else np.nan


def safe_max(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    return float(v.max()) if v.notna().any() else np.nan


def safe_min(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    return float(v.min()) if v.notna().any() else np.nan


# ----------------------------
# Run collection and summaries
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
                "jacobian_variant": infer_jacobian_variant(run_name),
                "solver_mode": infer_solver_mode(run_name),
                "rollout_steps": parse_rollout_steps_from_run_name(run_name),
                "Np": parse_Np_from_run_name(run_name),
                "sqp_budget": parse_sqp_budget_from_run_name(run_name),
            }
        )

    return runs


def summarise_run(run: dict, *, spike_error_mm: float) -> dict:
    df = run["df"]
    n = len(df)

    infeasible = to_num(df, "infeasible").fillna(0).astype(int)

    # Prediction-error candidates.
    adapt_xy = to_num(df, "adapt_pred_err_xy_mm")
    adapt_xyz = to_num(df, "adapt_pred_err_xyz_mm")
    pred1_xy = to_num(df, "pred1_err_xy_mm")
    pred1_xyz = to_num(df, "pred1_err_xyz_mm")

    # Spike diagnostics based on actual row-to-row change in adaptive prediction error.
    err = adapt_xy.copy()
    err_delta = err.diff()
    err_next_delta = err.shift(-1) - err
    spike_now = err_delta >= spike_error_mm
    spike_next = err_next_delta >= spike_error_mm

    # Controls.
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

    # SQP / update diagnostics, if present.
    sqp_rel_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"sqp_du_rel_\d+", c)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )
    sqp_abs_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"sqp_du_\d+", c)],
        key=lambda x: int(x.rsplit("_", 1)[1]),
    )

    if sqp_rel_cols:
        rel_df = df[sqp_rel_cols].apply(pd.to_numeric, errors="coerce")
        sqp_du_rel_first = rel_df.iloc[:, 0]
        sqp_du_rel_max = rel_df.max(axis=1)
    else:
        sqp_du_rel_first = pd.Series(np.nan, index=df.index)
        sqp_du_rel_max = pd.Series(np.nan, index=df.index)

    if sqp_abs_cols:
        abs_df = df[sqp_abs_cols].apply(pd.to_numeric, errors="coerce")
        sqp_du_first = abs_df.iloc[:, 0]
        sqp_du_max = abs_df.max(axis=1)
    else:
        sqp_du_first = pd.Series(np.nan, index=df.index)
        sqp_du_max = pd.Series(np.nan, index=df.index)

    reasons = df.get("rollout_adaptation_reason", pd.Series("", index=df.index)).fillna("")

    Np = run["Np"]
    rollout = run["rollout_steps"]
    horizon_gap = Np - rollout if np.isfinite(Np) and np.isfinite(rollout) else np.nan
    horizon_ratio = rollout / Np if np.isfinite(Np) and Np > 0 and np.isfinite(rollout) else np.nan

    return {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),
        "log_csv": str(run["log_csv"]),

        "bend_angle_deg": run["bend_angle_deg"],
        "jacobian_variant": run["jacobian_variant"],
        "solver_mode": run["solver_mode"],
        "rollout_steps": rollout,
        "Np": Np,
        "sqp_budget": run["sqp_budget"],

        "horizon_gap": horizon_gap,
        "horizon_ratio_rollout_over_Np": horizon_ratio,
        "matched_horizon": bool(np.isfinite(horizon_gap) and abs(horizon_gap) < 1e-12),
        "prediction_longer_than_rollout": bool(np.isfinite(horizon_gap) and horizon_gap > 0),

        "n_steps": int(n),
        "success": bool((infeasible == 0).all()) if n else False,
        "num_infeasible_steps": int(infeasible.sum()) if n else 0,
        "final_status": str(df["status"].iloc[-1]) if "status" in df.columns and n else "",

        "final_i_ref": float(to_num(df, "i_ref").iloc[-1]) if n else np.nan,
        "max_i_ref": safe_max(to_num(df, "i_ref")),
        "final_p_now_L_mm": 1e3 * float(to_num(df, "p_now_L").iloc[-1]) if n and "p_now_L" in df.columns else np.nan,

        "mean_adapt_pred_err_xy_mm": safe_mean(adapt_xy),
        "median_adapt_pred_err_xy_mm": safe_median(adapt_xy),
        "max_adapt_pred_err_xy_mm": safe_max(adapt_xy),
        "mean_adapt_pred_err_xyz_mm": safe_mean(adapt_xyz),
        "median_adapt_pred_err_xyz_mm": safe_median(adapt_xyz),
        "max_adapt_pred_err_xyz_mm": safe_max(adapt_xyz),

        "mean_pred1_err_xy_mm": safe_mean(pred1_xy),
        "median_pred1_err_xy_mm": safe_median(pred1_xy),
        "max_pred1_err_xy_mm": safe_max(pred1_xy),
        "mean_pred1_err_xyz_mm": safe_mean(pred1_xyz),
        "median_pred1_err_xyz_mm": safe_median(pred1_xyz),
        "max_pred1_err_xyz_mm": safe_max(pred1_xyz),

        "mean_clearance_mm": safe_mean(to_num(df, "adapt_clearance_mm")),
        "min_clearance_mm": safe_min(to_num(df, "adapt_clearance_mm")),
        "mean_tip_vessel_angle_deg": safe_mean(to_num(df, "adapt_tip_vessel_angle_deg")),
        "max_tip_vessel_angle_deg": safe_max(to_num(df, "adapt_tip_vessel_angle_deg")),

        "mean_cond_H_beam": safe_mean(to_num(df, "cond_H_beam")),
        "max_cond_H_beam": safe_max(to_num(df, "cond_H_beam")),
        "mean_cond_H_mpc": safe_mean(to_num(df, "cond_H_mpc")),
        "max_cond_H_mpc": safe_max(to_num(df, "cond_H_mpc")),
        "mean_mpc_eig_cond": safe_mean(to_num(df, "mpc_eig_cond")),
        "max_mpc_eig_cond": safe_max(to_num(df, "mpc_eig_cond")),

        "control_rms": control_rms,
        "control_max_abs": control_max_abs,

        "mean_sqp_du_rel_first": safe_mean(sqp_du_rel_first),
        "median_sqp_du_rel_first": safe_median(sqp_du_rel_first),
        "max_sqp_du_rel_first": safe_max(sqp_du_rel_first),
        "mean_sqp_du_rel_max": safe_mean(sqp_du_rel_max),
        "mean_sqp_du_rel_final": safe_mean(to_num(df, "sqp_du_rel_final")),
        "median_sqp_du_rel_final": safe_median(to_num(df, "sqp_du_rel_final")),
        "max_sqp_du_rel_final": safe_max(to_num(df, "sqp_du_rel_final")),

        "mean_sqp_du_first": safe_mean(sqp_du_first),
        "median_sqp_du_first": safe_median(sqp_du_first),
        "max_sqp_du_first": safe_max(sqp_du_first),
        "mean_sqp_du_max": safe_mean(sqp_du_max),
        "mean_sqp_du_final": safe_mean(to_num(df, "sqp_du_final")),
        "median_sqp_du_final": safe_median(to_num(df, "sqp_du_final")),
        "max_sqp_du_final": safe_max(to_num(df, "sqp_du_final")),

        "mean_sqp_iters_done": safe_mean(to_num(df, "sqp_iters_done")),
        "max_sqp_iters_done": safe_max(to_num(df, "sqp_iters_done")),

        "spike_error_mm": spike_error_mm,
        "num_error_spikes": int(spike_now.fillna(False).sum()),
        "error_spike_rate": float(spike_now.fillna(False).mean()) if n else np.nan,
        "num_next_error_spikes": int(spike_next.fillna(False).sum()),
        "next_error_spike_rate": float(spike_next.fillna(False).mean()) if n else np.nan,

        "num_good_increase": int(reasons.str.contains("good:increase", regex=False).sum()),
        "num_bad_prediction_error": int(reasons.str.contains("bad:prediction_error", regex=False).sum()),
        "num_bad_tip_angle": int(reasons.str.contains("bad:tip_angle", regex=False).sum()),
        "num_bad_low_clearance": int(reasons.str.contains("bad:low_clearance", regex=False).sum()),
    }


def build_run_summary(run_root: Path, *, out_root: Path, spike_error_mm: float) -> pd.DataFrame:
    runs = collect_runs(run_root)
    rows = []

    for run in runs:
        try:
            rows.append(summarise_run(run, spike_error_mm=spike_error_mm))
        except Exception as exc:
            print(f"[WARN] Failed to summarise {run['run_name']}: {exc}")
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
                    "analysis_error": repr(exc),
                }
            )

    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(
            [
                "bend_angle_deg",
                "rollout_steps",
                "Np",
                "jacobian_variant",
                "solver_mode",
                "sqp_budget",
            ]
        )

    out_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_root / "lti_rollout_np_run_summary.csv", index=False)

    print(f"[ANALYSIS] Found {len(summary)} runs")
    print(f"[ANALYSIS] Wrote {out_root / 'lti_rollout_np_run_summary.csv'}")

    return summary


# ----------------------------
# Pairwise contact-vs-no-contact analysis
# ----------------------------

DEFAULT_PAIR_METRICS = [
    "mean_adapt_pred_err_xy_mm",
    "median_adapt_pred_err_xy_mm",
    "max_adapt_pred_err_xy_mm",
    "mean_adapt_pred_err_xyz_mm",
    "max_adapt_pred_err_xyz_mm",
    "mean_pred1_err_xy_mm",
    "max_pred1_err_xy_mm",
    "error_spike_rate",
    "next_error_spike_rate",
    "final_i_ref",
    "max_i_ref",
    "n_steps",
    "num_infeasible_steps",
    "min_clearance_mm",
    "mean_clearance_mm",
    "max_tip_vessel_angle_deg",
    "mean_cond_H_mpc",
    "max_cond_H_mpc",
    "control_rms",
]


LOWER_IS_BETTER = {
    "mean_adapt_pred_err_xy_mm",
    "median_adapt_pred_err_xy_mm",
    "max_adapt_pred_err_xy_mm",
    "mean_adapt_pred_err_xyz_mm",
    "max_adapt_pred_err_xyz_mm",
    "mean_pred1_err_xy_mm",
    "max_pred1_err_xy_mm",
    "error_spike_rate",
    "next_error_spike_rate",
    "num_infeasible_steps",
    "max_tip_vessel_angle_deg",
    "mean_cond_H_mpc",
    "max_cond_H_mpc",
    "control_rms",
    "n_steps",
}

HIGHER_IS_BETTER = {
    "final_i_ref",
    "max_i_ref",
    "min_clearance_mm",
    "mean_clearance_mm",
}


def classify_horizon_regime(Np: float, rollout: float) -> str:
    if not np.isfinite(Np) or not np.isfinite(rollout):
        return "unknown"
    gap = int(Np - rollout)
    if gap == 0:
        return "matched_Np_eq_rollout"
    if rollout == 1 and Np > 1:
        return "long_prediction_short_rollout"
    if gap <= 2:
        return "mild_prediction_gap"
    return "large_prediction_gap"


def classify_winner(
    *,
    contact_value: float,
    no_contact_value: float,
    metric: str,
    tie_abs_mm: float,
    tie_rel_pct: float,
    high_error_mm: float,
) -> tuple[str, float, float, str]:
    """
    Returns:
        winner, advantage, advantage_pct, equivalence_class

    For lower-is-better metrics:
        advantage = no_contact - contact
        advantage > 0 means contact is better.

    For higher-is-better metrics:
        advantage = contact - no_contact
        advantage > 0 means contact is better.
    """
    if not np.isfinite(contact_value) or not np.isfinite(no_contact_value):
        return "missing", np.nan, np.nan, "missing"

    if metric in HIGHER_IS_BETTER:
        advantage = contact_value - no_contact_value
        denom = abs(no_contact_value)
    else:
        advantage = no_contact_value - contact_value
        denom = abs(no_contact_value)

    advantage_pct = 100.0 * advantage / denom if denom > 1e-12 else np.nan

    abs_close = abs(advantage) <= tie_abs_mm
    rel_close = np.isfinite(advantage_pct) and abs(advantage_pct) <= tie_rel_pct

    equivalent = abs_close or rel_close

    mean_error_like_value = 0.5 * (contact_value + no_contact_value)

    if equivalent:
        if metric in LOWER_IS_BETTER and np.isfinite(mean_error_like_value) and mean_error_like_value >= high_error_mm:
            equivalence_class = "equivalent_high_error"
        else:
            equivalence_class = "equivalent_low_or_moderate_error"
        return "equivalent", advantage, advantage_pct, equivalence_class

    if advantage > 0:
        return "contact_better", advantage, advantage_pct, "not_equivalent"
    return "no_contact_better", advantage, advantage_pct, "not_equivalent"


def build_pairwise_advantage(
    summary: pd.DataFrame,
    *,
    out_root: Path,
    primary_metric: str,
    pair_metrics: list[str],
    tie_abs_mm: float,
    tie_rel_pct: float,
    high_error_mm: float,
) -> pd.DataFrame:
    index_cols = [
        "bend_angle_deg",
        "solver_mode",
        "rollout_steps",
        "Np",
        "sqp_budget",
    ]

    rows = []

    for key, g in summary.groupby(index_cols, dropna=False):
        contact = g[g["jacobian_variant"] == "contact"]
        no_contact = g[g["jacobian_variant"] == "no_contact"]

        if contact.empty or no_contact.empty:
            continue

        c = contact.iloc[0]
        n = no_contact.iloc[0]

        out = dict(zip(index_cols, key))
        out["horizon_gap"] = float(out["Np"] - out["rollout_steps"])
        out["horizon_ratio_rollout_over_Np"] = (
            float(out["rollout_steps"] / out["Np"])
            if np.isfinite(out["Np"]) and out["Np"] > 0
            else np.nan
        )
        out["matched_horizon"] = bool(out["horizon_gap"] == 0)
        out["horizon_regime"] = classify_horizon_regime(out["Np"], out["rollout_steps"])

        out["contact_run_name"] = c.get("run_name", "")
        out["no_contact_run_name"] = n.get("run_name", "")

        for metric in pair_metrics:
            if metric not in summary.columns:
                continue

            c_val = float(c[metric]) if pd.notna(c[metric]) else np.nan
            n_val = float(n[metric]) if pd.notna(n[metric]) else np.nan

            winner, adv, adv_pct, eq_class = classify_winner(
                contact_value=c_val,
                no_contact_value=n_val,
                metric=metric,
                tie_abs_mm=tie_abs_mm,
                tie_rel_pct=tie_rel_pct,
                high_error_mm=high_error_mm,
            )

            out[f"{metric}__contact"] = c_val
            out[f"{metric}__no_contact"] = n_val
            out[f"{metric}__contact_advantage"] = adv
            out[f"{metric}__contact_advantage_pct"] = adv_pct
            out[f"{metric}__winner"] = winner
            out[f"{metric}__equivalence_class"] = eq_class

        # Primary metric convenience columns.
        p = primary_metric
        out["primary_metric"] = p
        out["contact_error"] = out.get(f"{p}__contact", np.nan)
        out["no_contact_error"] = out.get(f"{p}__no_contact", np.nan)
        out["contact_advantage"] = out.get(f"{p}__contact_advantage", np.nan)
        out["contact_advantage_pct"] = out.get(f"{p}__contact_advantage_pct", np.nan)
        out["winner"] = out.get(f"{p}__winner", "missing")
        out["equivalence_class"] = out.get(f"{p}__equivalence_class", "missing")

        rows.append(out)

    pairwise = pd.DataFrame(rows)

    if len(pairwise):
        pairwise = pairwise.sort_values(
            ["bend_angle_deg", "rollout_steps", "Np", "sqp_budget"]
        )

    pairwise.to_csv(out_root / "lti_contact_pairwise_advantage.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'lti_contact_pairwise_advantage.csv'}")
    return pairwise


# ----------------------------
# Regime summaries
# ----------------------------

def summarise_regimes(pairwise: pd.DataFrame, *, out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def frac(mask: pd.Series) -> float:
        return float(mask.mean()) if len(mask) else np.nan

    group_cols = [
        "bend_angle_deg",
        "horizon_regime",
        "matched_horizon",
    ]

    rows = []
    for key, g in pairwise.groupby(group_cols, dropna=False):
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "n_pairs": int(len(g)),
                "mean_contact_advantage": float(g["contact_advantage"].mean()),
                "median_contact_advantage": float(g["contact_advantage"].median()),
                "mean_contact_advantage_pct": float(g["contact_advantage_pct"].mean()),
                "frac_contact_better": frac(g["winner"] == "contact_better"),
                "frac_no_contact_better": frac(g["winner"] == "no_contact_better"),
                "frac_equivalent": frac(g["winner"] == "equivalent"),
                "frac_equivalent_high_error": frac(g["equivalence_class"] == "equivalent_high_error"),
                "mean_contact_error": float(g["contact_error"].mean()),
                "mean_no_contact_error": float(g["no_contact_error"].mean()),
            }
        )

    regime = pd.DataFrame(rows)
    regime.to_csv(out_root / "lti_contact_advantage_regime_summary.csv", index=False)

    # Equivalence by exact rollout/Np.
    eq_rows = []
    for key, g in pairwise.groupby(["bend_angle_deg", "rollout_steps", "Np"], dropna=False):
        eq_rows.append(
            {
                "bend_angle_deg": key[0],
                "rollout_steps": key[1],
                "Np": key[2],
                "horizon_gap": float(key[2] - key[1]),
                "n_pairs": int(len(g)),
                "mean_contact_advantage": float(g["contact_advantage"].mean()),
                "winner": str(g["winner"].iloc[0]) if len(g) else "",
                "equivalence_class": str(g["equivalence_class"].iloc[0]) if len(g) else "",
                "contact_error": float(g["contact_error"].mean()),
                "no_contact_error": float(g["no_contact_error"].mean()),
            }
        )

    equivalence = pd.DataFrame(eq_rows)
    equivalence.to_csv(out_root / "lti_equivalence_summary.csv", index=False)

    # Controller interpretation table.
    interp_rows = []
    for key, g in pairwise.groupby(["bend_angle_deg", "rollout_steps", "Np"], dropna=False):
        winner = str(g["winner"].iloc[0])
        eq_class = str(g["equivalence_class"].iloc[0])
        contact_adv = float(g["contact_advantage"].mean())
        c_err = float(g["contact_error"].mean())
        n_err = float(g["no_contact_error"].mean())
        horizon_gap = float(key[2] - key[1])
        regime_name = classify_horizon_regime(key[2], key[1])

        if winner == "contact_better":
            interpretation = "contact_model_advantage"
            controller_action = "Contact-aware LTI model is beneficial in this regime."
        elif winner == "no_contact_better":
            interpretation = "no_contact_model_lower_error"
            controller_action = "Contact-aware LTI propagation may be overconfident or biased here; inspect trajectories."
        elif eq_class == "equivalent_high_error":
            interpretation = "both_models_error_prone"
            controller_action = "Reduce rollout or replan sooner; model choice is less important than horizon reliability."
        elif winner == "equivalent":
            interpretation = "models_effectively_equivalent"
            controller_action = "Either Jacobian is similar; choose based on robustness or compute."
        else:
            interpretation = "missing_or_inconclusive"
            controller_action = "Missing matched pair or inconclusive metric."

        interp_rows.append(
            {
                "bend_angle_deg": key[0],
                "rollout_steps": key[1],
                "Np": key[2],
                "horizon_gap": horizon_gap,
                "horizon_regime": regime_name,
                "winner": winner,
                "equivalence_class": eq_class,
                "contact_error": c_err,
                "no_contact_error": n_err,
                "contact_advantage": contact_adv,
                "interpretation": interpretation,
                "controller_action": controller_action,
            }
        )

    interpretation = pd.DataFrame(interp_rows)
    interpretation.to_csv(out_root / "lti_controller_interpretation_table.csv", index=False)

    print(f"[ANALYSIS] Wrote {out_root / 'lti_contact_advantage_regime_summary.csv'}")
    print(f"[ANALYSIS] Wrote {out_root / 'lti_equivalence_summary.csv'}")
    print(f"[ANALYSIS] Wrote {out_root / 'lti_controller_interpretation_table.csv'}")

    return regime, equivalence, interpretation


# ----------------------------
# Plots
# ----------------------------

def make_triangular_grid(pairwise: pd.DataFrame, bend: float, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = pairwise[pairwise["bend_angle_deg"] == bend].copy()

    rollouts = sorted(g["rollout_steps"].dropna().unique())
    Nps = sorted(g["Np"].dropna().unique())

    grid = np.full((len(rollouts), len(Nps)), np.nan)

    for i, rollout in enumerate(rollouts):
        for j, Np in enumerate(Nps):
            gg = g[(g["rollout_steps"] == rollout) & (g["Np"] == Np)]
            if len(gg) and value_col in gg.columns:
                grid[i, j] = float(gg[value_col].iloc[0])

    return np.asarray(rollouts), np.asarray(Nps), grid


def plot_heatmap(
    pairwise: pd.DataFrame,
    *,
    value_col: str,
    title_prefix: str,
    cbar_label: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted(pairwise["bend_angle_deg"].dropna().unique())

    for bend in bends:
        rollouts, Nps, grid = make_triangular_grid(pairwise, bend, value_col)

        if len(rollouts) == 0 or len(Nps) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7.2, 5.4))

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[
                min(Nps) - 0.5,
                max(Nps) + 0.5,
                min(rollouts) - 0.5,
                max(rollouts) + 0.5,
            ],
        )

        ax.set_xticks(Nps)
        ax.set_yticks(rollouts)
        ax.set_xlabel("Prediction horizon Np")
        ax.set_ylabel("Rollout length")
        ax.set_title(f"{title_prefix}, bend {bend:.0f}°")

        # Annotate each cell.
        for i, rollout in enumerate(rollouts):
            for j, Np in enumerate(Nps):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        Np,
                        rollout,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

        fig.tight_layout()
        fig.savefig(
            out_dir / f"{safe_filename(value_col)}_heatmap_bend_{bend_tag(bend)}.png",
            dpi=220,
        )
        plt.close(fig)


def plot_winner_heatmap(pairwise: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    code_map = {
        "no_contact_better": -1,
        "equivalent": 0,
        "contact_better": 1,
    }

    temp = pairwise.copy()
    temp["winner_code"] = temp["winner"].map(code_map).astype(float)

    bends = sorted(temp["bend_angle_deg"].dropna().unique())

    for bend in bends:
        rollouts, Nps, grid = make_triangular_grid(temp, bend, "winner_code")

        fig, ax = plt.subplots(figsize=(7.2, 5.4))

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            vmin=-1,
            vmax=1,
            extent=[
                min(Nps) - 0.5,
                max(Nps) + 0.5,
                min(rollouts) - 0.5,
                max(rollouts) + 0.5,
            ],
        )

        ax.set_xticks(Nps)
        ax.set_yticks(rollouts)
        ax.set_xlabel("Prediction horizon Np")
        ax.set_ylabel("Rollout length")
        ax.set_title(f"Winner by primary metric, bend {bend:.0f}°")

        label_map = {
            -1: "NC",
            0: "=",
            1: "C",
        }

        for i, rollout in enumerate(rollouts):
            for j, Np in enumerate(Nps):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(
                        Np,
                        rollout,
                        label_map.get(int(val), "?"),
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Winner code: -1 no-contact, 0 equivalent, +1 contact")

        fig.tight_layout()
        fig.savefig(
            out_dir / f"winner_heatmap_bend_{bend_tag(bend)}.png",
            dpi=220,
        )
        plt.close(fig)


def plot_advantage_lines(pairwise: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted(pairwise["bend_angle_deg"].dropna().unique())

    for bend in bends:
        g_bend = pairwise[pairwise["bend_angle_deg"] == bend].copy()
        rollouts = sorted(g_bend["rollout_steps"].dropna().unique())

        fig, ax = plt.subplots(figsize=(8.2, 5.0))

        for rollout in rollouts:
            g = g_bend[g_bend["rollout_steps"] == rollout].sort_values("Np")
            if g.empty:
                continue

            ax.plot(
                g["Np"],
                g["contact_advantage"],
                marker="o",
                linewidth=2.0,
                label=f"rollout={int(rollout)}",
            )

        ax.axhline(0.0, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Prediction horizon Np")
        ax.set_ylabel("Contact advantage: no-contact error - contact error")
        ax.set_title(f"Contact advantage vs Np, bend {bend:.0f}°")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(
            out_dir / f"contact_advantage_lines_bend_{bend_tag(bend)}.png",
            dpi=220,
        )
        plt.close(fig)


def plot_error_lines(pairwise: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted(pairwise["bend_angle_deg"].dropna().unique())

    for bend in bends:
        g_bend = pairwise[pairwise["bend_angle_deg"] == bend].copy()
        rollouts = sorted(g_bend["rollout_steps"].dropna().unique())

        for rollout in rollouts:
            g = g_bend[g_bend["rollout_steps"] == rollout].sort_values("Np")
            if g.empty:
                continue

            fig, ax = plt.subplots(figsize=(7.4, 4.6))

            ax.plot(
                g["Np"],
                g["contact_error"],
                marker="o",
                linewidth=2.2,
                label="contact-aware",
            )
            ax.plot(
                g["Np"],
                g["no_contact_error"],
                marker="s",
                linewidth=2.2,
                linestyle="--",
                label="no-contact",
            )

            ax.set_xlabel("Prediction horizon Np")
            ax.set_ylabel("Primary error metric")
            ax.set_title(f"Contact vs no-contact error, bend {bend:.0f}°, rollout={int(rollout)}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            fig.tight_layout()
            fig.savefig(
                out_dir / f"contact_vs_no_contact_error_bend_{bend_tag(bend)}_rollout_{int(rollout)}.png",
                dpi=220,
            )
            plt.close(fig)


def plot_reference_traces(
    summary: pd.DataFrame,
    *,
    metric_col_in_log: str,
    out_dir: Path,
    max_plots: int = 60,
) -> None:
    """
    Generates matched contact/no-contact traces for the raw per-step log metric.
    To avoid too many files, stops after max_plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    index_cols = ["bend_angle_deg", "rollout_steps", "Np", "sqp_budget"]
    count = 0

    for key, g in summary.groupby(index_cols, dropna=False):
        if count >= max_plots:
            break

        contact = g[g["jacobian_variant"] == "contact"]
        no_contact = g[g["jacobian_variant"] == "no_contact"]

        if contact.empty or no_contact.empty:
            continue

        c_log = Path(contact.iloc[0]["log_csv"])
        n_log = Path(no_contact.iloc[0]["log_csv"])

        try:
            c_df = pd.read_csv(c_log)
            n_df = pd.read_csv(n_log)
        except Exception:
            continue

        if metric_col_in_log not in c_df.columns or metric_col_in_log not in n_df.columns:
            continue

        x_col = "i_ref" if "i_ref" in c_df.columns and "i_ref" in n_df.columns else None

        c_x = to_num(c_df, x_col) if x_col else pd.Series(np.arange(len(c_df)))
        n_x = to_num(n_df, x_col) if x_col else pd.Series(np.arange(len(n_df)))

        c_y = to_num(c_df, metric_col_in_log)
        n_y = to_num(n_df, metric_col_in_log)

        fig, ax = plt.subplots(figsize=(8.0, 4.6))

        ax.plot(c_x, c_y, marker="o", linewidth=1.8, markersize=3.2, label="contact-aware")
        ax.plot(n_x, n_y, marker="s", linewidth=1.8, markersize=3.2, linestyle="--", label="no-contact")

        bend, rollout, Np, sqp_budget = key
        ax.set_xlabel("Reference index" if x_col else "Step")
        ax.set_ylabel(metric_col_in_log)
        ax.set_title(
            f"{metric_col_in_log}, bend {bend:.0f}°, rollout={int(rollout)}, Np={int(Np)}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(
            out_dir / (
                f"{safe_filename(metric_col_in_log)}_trace_"
                f"bend_{bend_tag(bend)}_rollout_{int(rollout)}_Np_{int(Np)}.png"
            ),
            dpi=200,
        )
        plt.close(fig)

        count += 1


def make_plots(
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    out_root: Path,
    primary_metric: str,
) -> None:
    plot_root = out_root / "plots"

    plot_heatmap(
        pairwise,
        value_col="contact_advantage",
        title_prefix="Contact advantage",
        cbar_label="No-contact error - contact error",
        out_dir=plot_root / "heatmaps",
    )

    plot_heatmap(
        pairwise,
        value_col="contact_error",
        title_prefix="Contact-aware error",
        cbar_label=primary_metric,
        out_dir=plot_root / "heatmaps",
    )

    plot_heatmap(
        pairwise,
        value_col="no_contact_error",
        title_prefix="No-contact error",
        cbar_label=primary_metric,
        out_dir=plot_root / "heatmaps",
    )

    plot_winner_heatmap(pairwise, out_dir=plot_root / "heatmaps")
    plot_advantage_lines(pairwise, out_dir=plot_root / "lineplots")
    plot_error_lines(pairwise, out_dir=plot_root / "lineplots")

    # Raw traces use the underlying log key, not the summary metric name.
    if primary_metric.startswith("mean_"):
        raw_metric = primary_metric.replace("mean_", "")
    elif primary_metric.startswith("median_"):
        raw_metric = primary_metric.replace("median_", "")
    elif primary_metric.startswith("max_"):
        raw_metric = primary_metric.replace("max_", "")
    else:
        raw_metric = "adapt_pred_err_xy_mm"

    plot_reference_traces(
        summary,
        metric_col_in_log=raw_metric,
        out_dir=plot_root / "traces",
        max_plots=60,
    )

    print(f"[ANALYSIS] Wrote plots to {plot_root}")


# ----------------------------
# Report
# ----------------------------

def write_report(
    *,
    out_root: Path,
    pairwise: pd.DataFrame,
    regime: pd.DataFrame,
    equivalence: pd.DataFrame,
    primary_metric: str,
    tie_abs_mm: float,
    tie_rel_pct: float,
    high_error_mm: float,
) -> None:
    report_path = out_root / "lti_contact_horizon_report.md"

    total = len(pairwise)
    n_contact = int((pairwise["winner"] == "contact_better").sum()) if total else 0
    n_no_contact = int((pairwise["winner"] == "no_contact_better").sum()) if total else 0
    n_equiv = int((pairwise["winner"] == "equivalent").sum()) if total else 0
    n_equiv_high = int((pairwise["equivalence_class"] == "equivalent_high_error").sum()) if total else 0

    matched = pairwise[pairwise["matched_horizon"] == True]
    gapped = pairwise[pairwise["matched_horizon"] == False]

    def mean_adv(df: pd.DataFrame) -> float:
        return float(df["contact_advantage"].mean()) if len(df) else np.nan

    lines = []
    lines.append("# LTI contact-vs-no-contact horizon/rollout analysis\n")
    lines.append("## Configuration\n")
    lines.append(f"- Primary metric: `{primary_metric}`\n")
    lines.append(f"- Equivalence threshold: absolute <= `{tie_abs_mm}` or relative <= `{tie_rel_pct}%`\n")
    lines.append(f"- High-error equivalence threshold: mean pair error >= `{high_error_mm}`\n")
    lines.append("\n")

    lines.append("## Overall results\n")
    lines.append(f"- Matched contact/no-contact pairs: **{total}**\n")
    lines.append(f"- Contact better: **{n_contact}**\n")
    lines.append(f"- No-contact better: **{n_no_contact}**\n")
    lines.append(f"- Equivalent: **{n_equiv}**\n")
    lines.append(f"- Equivalent with high error: **{n_equiv_high}**\n")
    lines.append(f"- Mean contact advantage over all pairs: **{mean_adv(pairwise):.4g}**\n")
    lines.append(f"- Mean contact advantage for matched `Np == rollout`: **{mean_adv(matched):.4g}**\n")
    lines.append(f"- Mean contact advantage for `Np > rollout`: **{mean_adv(gapped):.4g}**\n")
    lines.append("\n")

    lines.append("## Interpretation\n")
    lines.append(
        "A positive contact advantage means the no-contact error is larger than the contact-aware error. "
        "Therefore positive values support the contact-aware Jacobian.\n\n"
    )
    lines.append(
        "The key hypothesis is that the contact-aware Jacobian should be most beneficial when the "
        "controller is making a short local prediction, especially when `Np` and rollout are small or matched. "
        "As the LTI prediction is propagated over larger horizons, both Jacobians may become error-prone because "
        "a time-invariant linearisation cannot capture the nonlinear/contact-rich evolution over the full horizon. "
        "In that regime, contact and no-contact models may become equivalent, or the no-contact model may sometimes "
        "appear better because it is smoother or less aggressive.\n\n"
    )

    lines.append("## Controller-design reading\n")
    lines.append(
        "- If `winner == contact_better`, the contact-aware local model is useful for that horizon/rollout regime.\n"
    )
    lines.append(
        "- If `winner == equivalent_low_or_moderate_error`, the two models are similar and the simpler model may be acceptable.\n"
    )
    lines.append(
        "- If `equivalent_high_error`, both models are inaccurate; this supports reducing rollout or replanning sooner.\n"
    )
    lines.append(
        "- If `winner == no_contact_better`, inspect the traces: the contact-aware LTI propagation may be overconfident or stale over that horizon.\n"
    )
    lines.append("\n")

    lines.append("## Files to inspect\n")
    lines.append("- `lti_contact_pairwise_advantage.csv`\n")
    lines.append("- `lti_contact_advantage_regime_summary.csv`\n")
    lines.append("- `lti_equivalence_summary.csv`\n")
    lines.append("- `lti_controller_interpretation_table.csv`\n")
    lines.append("- `plots/heatmaps/`\n")
    lines.append("- `plots/lineplots/`\n")
    lines.append("- `plots/traces/`\n")

    with open(report_path, "w") as f:
        f.write("".join(lines))

    print(f"[ANALYSIS] Wrote {report_path}")


# ----------------------------
# Main
# ----------------------------

def run_analysis(
    run_root: Path,
    *,
    out_root: Path,
    primary_metric: str,
    tie_abs_mm: float,
    tie_rel_pct: float,
    high_error_mm: float,
    spike_error_mm: float,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    summary = build_run_summary(
        run_root,
        out_root=out_root,
        spike_error_mm=spike_error_mm,
    )

    if summary.empty:
        print("[ANALYSIS] No runs found. Expected subdirectories containing log.csv")
        return

    pair_metrics = [m for m in DEFAULT_PAIR_METRICS if m in summary.columns]
    if primary_metric not in pair_metrics:
        pair_metrics.insert(0, primary_metric)

    pairwise = build_pairwise_advantage(
        summary,
        out_root=out_root,
        primary_metric=primary_metric,
        pair_metrics=pair_metrics,
        tie_abs_mm=tie_abs_mm,
        tie_rel_pct=tie_rel_pct,
        high_error_mm=high_error_mm,
    )

    if pairwise.empty:
        print("[ANALYSIS] No matched contact/no-contact pairs found.")
        return

    regime, equivalence, interpretation = summarise_regimes(
        pairwise,
        out_root=out_root,
    )

    make_plots(
        pairwise,
        summary,
        out_root=out_root,
        primary_metric=primary_metric,
    )

    write_report(
        out_root=out_root,
        pairwise=pairwise,
        regime=regime,
        equivalence=equivalence,
        primary_metric=primary_metric,
        tie_abs_mm=tie_abs_mm,
        tie_rel_pct=tie_rel_pct,
        high_error_mm=high_error_mm,
    )

    print("\n[ANALYSIS] Main outputs:")
    print(f"  {out_root / 'lti_rollout_np_run_summary.csv'}")
    print(f"  {out_root / 'lti_contact_pairwise_advantage.csv'}")
    print(f"  {out_root / 'lti_contact_advantage_regime_summary.csv'}")
    print(f"  {out_root / 'lti_equivalence_summary.csv'}")
    print(f"  {out_root / 'lti_controller_interpretation_table.csv'}")
    print(f"  {out_root / 'lti_contact_horizon_report.md'}")
    print(f"  {out_root / 'plots'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse contact-aware vs no-contact LTI Jacobian over Np/rollout sweep."
    )
    parser.add_argument(
        "run_root",
        type=Path,
        nargs="?",
        default=Path("testing_lti_rollout_and_predicitions"),
        help="Root directory containing run subdirectories with log.csv files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run_root>/lti_contact_horizon_analysis.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_adapt_pred_err_xy_mm",
        help="Primary summary metric used for contact-vs-no-contact comparison.",
    )
    parser.add_argument(
        "--tie-abs-mm",
        type=float,
        default=0.25,
        help="Absolute difference threshold for treating models as equivalent.",
    )
    parser.add_argument(
        "--tie-rel-pct",
        type=float,
        default=10.0,
        help="Relative percentage threshold for treating models as equivalent.",
    )
    parser.add_argument(
        "--high-error-mm",
        type=float,
        default=2.0,
        help="If models are equivalent and average error exceeds this, classify as equivalent_high_error.",
    )
    parser.add_argument(
        "--spike-error-mm",
        type=float,
        default=1.0,
        help="Step-to-step prediction-error increase counted as an error spike.",
    )

    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    out_root = args.out_root
    if out_root is None:
        out_root = run_root / "lti_contact_horizon_analysis"
    else:
        out_root = out_root.expanduser().resolve()

    run_analysis(
        run_root,
        out_root=out_root,
        primary_metric=args.metric,
        tie_abs_mm=args.tie_abs_mm,
        tie_rel_pct=args.tie_rel_pct,
        high_error_mm=args.high_error_mm,
        spike_error_mm=args.spike_error_mm,
    )


if __name__ == "__main__":
    main()