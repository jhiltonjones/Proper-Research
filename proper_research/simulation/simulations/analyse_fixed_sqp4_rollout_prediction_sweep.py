#!/usr/bin/env python3
"""
Analyse rollout length and prediction horizon with fixed SQP budget N_sqp = 4.

Expected experiment layout
--------------------------
evidence_contact_sqp_budget_sweep/
    roll_1/
        pred_1/
            bend_m30_jac_contact_plant_contact_1_sqp_full_rollout1_Np1_sqp4/log.csv
            ...
        pred_2/
        ...
        pred_6/
    roll_2/
        pred_2/
        ...
        pred_6/
    ...
    roll_6/
        pred_6/

This script intentionally ignores top-level sqp1...sqp6 folders and analyses only
the roll_*/pred_* triangular sweep, filtering to N_sqp = 4.

Outputs
-------
evidence_contact_sqp_budget_sweep/
    fixed_sqp4_rollout_prediction_analysis/
        sqp4_roll_pred_run_summary.csv
        sqp4_roll_pred_aggregate_by_roll_pred.csv
        sqp4_roll_pred_horizon_error_detail.csv
        sqp4_roll_pred_trends_by_roll.csv
        sqp4_roll_pred_trends_by_pred.csv
        sqp4_roll_pred_extra_horizon_summary.csv
        sqp4_roll_pred_best_settings.csv
        enriched_logs/
        heatmaps/
        line_plots/
        horizon_plots/
        bend_plots/
"""

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------
RUN_ROOT = Path("evidence_contact_sqp_budget_sweep")
OUT_DIR = RUN_ROOT / "fixed_sqp4_rollout_prediction_analysis"

FIXED_N_SQP = 4
EXPECTED_ROLLOUTS = list(range(1, 7))
EXPECTED_PREDS = list(range(1, 7))

# Keep this false unless you deliberately want to analyse the older duplicate
# top-level folder named "roll1_and_pred6".
INCLUDE_LEGACY_ROLL1_AND_PRED6 = False

# If true, only keep folders where pred >= rollout.
REQUIRE_PRED_GE_ROLL = True

# If true, only keep contact-Jacobian runs.
ONLY_CONTACT_JACOBIAN = True

# If true, only keep sqp_full solver runs.
ONLY_SQP_FULL = True


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def safe_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index)


def nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def nanmax(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else np.nan


def nanmin(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmin(a)) if np.isfinite(a).any() else np.nan


def nanp(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else np.nan


def finite_values(s):
    s = pd.to_numeric(s, errors="coerce")
    return s[np.isfinite(s)]


def monotonicity_from_values(values, atol=1e-12):
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) <= 1:
        return "insufficient"
    diffs = np.diff(vals)
    if np.all(diffs > atol):
        return "strictly_increasing"
    if np.all(diffs < -atol):
        return "strictly_decreasing"
    if np.all(diffs >= -atol):
        return "nondecreasing"
    if np.all(diffs <= atol):
        return "nonincreasing"
    return "non_monotonic"


def linear_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(x[m], y[m], deg=1)[0])


def parse_int(pattern: str, text: str, default=np.nan):
    m = re.search(pattern, text)
    if not m:
        return default
    return int(m.group(1))


def parse_bend_angle_from_run_name(run_name: str):
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def parse_jacobian_variant(run_name: str):
    if "jac_no_contact" in run_name:
        return "no_contact"
    if "jac_contact" in run_name:
        return "contact"
    return "unknown"


def parse_solver_mode(run_name: str):
    if "sqp_full" in run_name:
        return "sqp_full"
    if "ltv_oneshot" in run_name:
        return "ltv_oneshot"
    if "lti" in run_name:
        return "lti"
    return "unknown"


def parse_metadata_from_path(log_csv: Path):
    run_dir = log_csv.parent
    run_name = run_dir.name

    # Path-derived rollout and prediction horizon are the most reliable for this sweep.
    pred_from_path = np.nan
    roll_from_path = np.nan

    for part in log_csv.parts:
        m_roll = re.fullmatch(r"roll_(\d+)", part)
        m_pred = re.fullmatch(r"pred_(\d+)", part)
        if m_roll:
            roll_from_path = int(m_roll.group(1))
        if m_pred:
            pred_from_path = int(m_pred.group(1))

    roll_from_name = parse_int(r"rollout(\d+)", run_name)
    pred_from_name = parse_int(r"Np(\d+)", run_name)
    sqp_from_name = parse_int(r"sqp(\d+)", run_name)

    rollout = int(roll_from_path) if np.isfinite(roll_from_path) else roll_from_name
    pred = int(pred_from_path) if np.isfinite(pred_from_path) else pred_from_name

    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "log_csv": log_csv,
        "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
        "bend_abs_deg": abs(parse_bend_angle_from_run_name(run_name)),
        "jacobian_variant": parse_jacobian_variant(run_name),
        "solver_mode_from_name": parse_solver_mode(run_name),
        "rollout": rollout,
        "Np": pred,
        "extra_horizon": pred - rollout if np.isfinite(pred) and np.isfinite(rollout) else np.nan,
        "N_sqp": sqp_from_name,
    }


# ---------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------
def discover_logs(run_root: Path):
    run_root = Path(run_root)
    candidates = []

    for roll in EXPECTED_ROLLOUTS:
        roll_dir = run_root / f"roll_{roll}"
        if not roll_dir.exists():
            warnings.warn(f"Missing expected rollout folder: {roll_dir}")
            continue

        for pred in EXPECTED_PREDS:
            pred_dir = roll_dir / f"pred_{pred}"
            if REQUIRE_PRED_GE_ROLL and pred < roll:
                continue
            if not pred_dir.exists():
                continue

            candidates.extend(sorted(pred_dir.glob("*/log.csv")))

    if INCLUDE_LEGACY_ROLL1_AND_PRED6:
        candidates.extend(sorted((run_root / "roll1_and_pred6").glob("*/log.csv")))

    runs = []
    skipped = []

    for log_csv in sorted(set(candidates)):
        meta = parse_metadata_from_path(log_csv)

        if np.isfinite(meta["N_sqp"]) and int(meta["N_sqp"]) != FIXED_N_SQP:
            skipped.append((str(log_csv), f"N_sqp={meta['N_sqp']}"))
            continue

        if ONLY_CONTACT_JACOBIAN and meta["jacobian_variant"] != "contact":
            skipped.append((str(log_csv), f"jac={meta['jacobian_variant']}"))
            continue

        if ONLY_SQP_FULL and meta["solver_mode_from_name"] != "sqp_full":
            skipped.append((str(log_csv), f"solver={meta['solver_mode_from_name']}"))
            continue

        if REQUIRE_PRED_GE_ROLL and np.isfinite(meta["rollout"]) and np.isfinite(meta["Np"]):
            if int(meta["Np"]) < int(meta["rollout"]):
                skipped.append((str(log_csv), "Np < rollout"))
                continue

        try:
            df = pd.read_csv(log_csv)
        except Exception as e:
            skipped.append((str(log_csv), f"read_error={e}"))
            continue

        meta["df"] = df
        runs.append(meta)

    return runs, skipped


# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------
def choose_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def rollout_error_columns(df: pd.DataFrame, kind="xy"):
    pattern = re.compile(rf"rollout_err_{kind}_(\d+)_mm$")
    cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    return sorted(cols, key=lambda t: t[0])


def control_norm(df: pd.DataFrame, prefix="u_applied_0"):
    cols = [
        c for c in df.columns
        if c.startswith(prefix + "_")
        and not c.endswith("_norm")
    ]
    if not cols:
        # Fallback if a norm is already logged.
        for c in [f"{prefix}_norm", "u_norm", "control_norm"]:
            if c in df.columns:
                return safe_num(df, c)

        return pd.Series(np.nan, index=df.index)

    U = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.linalg.norm(U, axis=1), index=df.index)


def infer_position_scale_to_mm(df: pd.DataFrame):
    # State positions usually appear to be in metres. If typical magnitudes are < 1,
    # convert displacement norms to mm for readability.
    cols = [c for c in ["x_rollout_0_x", "x_rollout_0_y", "x_rollout_0_z"] if c in df.columns]
    if not cols:
        return 1.0

    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med_abs = np.nanmedian(np.abs(vals))
    if np.isfinite(med_abs) and med_abs < 1.0:
        return 1000.0
    return 1.0


def derive_one_step_direction_and_gain(df: pd.DataFrame):
    """
    Derive local one-step direction/gain diagnostics.

    We estimate:
        previous actual state = x_rollout_0 at previous log row
        predicted displacement = x_pred_0[current] - previous_actual
        actual displacement    = x_rollout_0[current] - previous_actual

    This is a useful diagnostic, but it is an effective realised gain proxy rather
    than a directly logged Jacobian singular value.
    """
    required = [
        "x_rollout_0_x", "x_rollout_0_y",
        "x_pred_0_x", "x_pred_0_y",
    ]
    if not all(c in df.columns for c in required):
        return pd.DataFrame(index=df.index)

    scale_mm = infer_position_scale_to_mm(df)

    actual_xy = df[["x_rollout_0_x", "x_rollout_0_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pred_xy = df[["x_pred_0_x", "x_pred_0_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    prev_actual_xy = np.vstack([np.full((1, 2), np.nan), actual_xy[:-1, :]])

    actual_delta = (actual_xy - prev_actual_xy) * scale_mm
    pred_delta = (pred_xy - prev_actual_xy) * scale_mm

    pred_norm = np.linalg.norm(pred_delta, axis=1)
    actual_norm = np.linalg.norm(actual_delta, axis=1)
    dot = np.sum(pred_delta * actual_delta, axis=1)
    denom = pred_norm * actual_norm

    cos = np.full(len(df), np.nan)
    valid = np.isfinite(denom) & (denom > 1e-12)
    cos[valid] = dot[valid] / denom[valid]
    cos = np.clip(cos, -1.0, 1.0)

    direction_error_deg = np.degrees(np.arccos(cos))

    u_norm = control_norm(df).to_numpy(dtype=float)
    pred_gain = np.full(len(df), np.nan)
    actual_gain = np.full(len(df), np.nan)
    gain_ratio = np.full(len(df), np.nan)

    valid_u = np.isfinite(u_norm) & (u_norm > 1e-12)
    pred_gain[valid_u] = pred_norm[valid_u] / u_norm[valid_u]
    actual_gain[valid_u] = actual_norm[valid_u] / u_norm[valid_u]

    valid_gain = np.isfinite(actual_gain) & (actual_gain > 1e-12) & np.isfinite(pred_gain)
    gain_ratio[valid_gain] = pred_gain[valid_gain] / actual_gain[valid_gain]

    gain_ratio_log_error = np.full(len(df), np.nan)
    valid_ratio = np.isfinite(gain_ratio) & (gain_ratio > 1e-12)
    gain_ratio_log_error[valid_ratio] = np.abs(np.log10(gain_ratio[valid_ratio]))

    return pd.DataFrame(
        {
            "derived_cos_one_step_xy": cos,
            "derived_direction_error_one_step_xy_deg": direction_error_deg,
            "derived_pred_gain_one_step_xy_mm_per_u": pred_gain,
            "derived_actual_gain_one_step_xy_mm_per_u": actual_gain,
            "derived_gain_ratio_pred_over_actual_one_step_xy": gain_ratio,
            "derived_gain_ratio_log_error_one_step_xy": gain_ratio_log_error,
            "derived_u_norm": u_norm,
        },
        index=df.index,
    )


def summarise_run(run):
    df = run["df"].copy()

    derived = derive_one_step_direction_and_gain(df)
    if len(derived.columns):
        df = pd.concat([df, derived], axis=1)

    # Save enriched log for later step-level inspection.
    enriched_dir = OUT_DIR / "enriched_logs"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = enriched_dir / f"{run['run_name']}_enriched.csv"
    df.to_csv(enriched_path, index=False)

    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    failure_mask = (
        (infeasible > 0)
        | (~status.isin(["solved", "solved_inaccurate", "solved inaccurate"]))
    )

    one_step_col = choose_first_existing(
        df,
        [
            "pred1_err_xy_mm",
            "rollout_err_xy_0_mm",
            "adapt_pred_err_xy_mm",
        ],
    )

    one_step_xyz_col = choose_first_existing(
        df,
        [
            "pred1_err_xyz_mm",
            "rollout_err_xyz_0_mm",
            "adapt_pred_err_xyz_mm",
        ],
    )

    raw_col = choose_first_existing(
        df,
        [
            "raw_one_step_err_xy_mm",
            "raw_pred_err_xy_mm",
            "raw_err_xy_mm",
            "pred1_raw_err_xy_mm",
        ],
    )

    adapt_col = choose_first_existing(
        df,
        [
            "adapt_pred_err_xy_mm",
            "adaptive_pred_err_xy_mm",
        ],
    )

    rollout_cols_xy = rollout_error_columns(df, kind="xy")
    rollout_cols_xyz = rollout_error_columns(df, kind="xyz")

    rollout_matrix_xy = None
    if rollout_cols_xy:
        rollout_matrix_xy = df[[c for _, c in rollout_cols_xy]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    rollout_matrix_xyz = None
    if rollout_cols_xyz:
        rollout_matrix_xyz = df[[c for _, c in rollout_cols_xyz]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    roll = int(run["rollout"])
    pred = int(run["Np"])

    def horizon_col_value(k, kind="xy"):
        cols = rollout_cols_xy if kind == "xy" else rollout_cols_xyz
        lookup = {idx: col for idx, col in cols}
        col = lookup.get(k)
        if col is None:
            return pd.Series(np.nan, index=df.index)
        return safe_num(df, col)

    # Index convention: rollout_err_xy_0_mm is first predicted step.
    # Therefore terminal index for rollout R is R-1, terminal index for Np is Np-1.
    rollout_terminal_xy = horizon_col_value(roll - 1, kind="xy")
    prediction_terminal_xy = horizon_col_value(pred - 1, kind="xy")

    # If prediction terminal is not logged, fall back to the deepest logged horizon.
    deepest_logged_xy = pd.Series(np.nan, index=df.index)
    if rollout_cols_xy:
        deepest_logged_xy = safe_num(df, rollout_cols_xy[-1][1])

    one = safe_num(df, one_step_col) if one_step_col else pd.Series(np.nan, index=df.index)
    one_xyz = safe_num(df, one_step_xyz_col) if one_step_xyz_col else pd.Series(np.nan, index=df.index)
    raw = safe_num(df, raw_col) if raw_col else pd.Series(np.nan, index=df.index)
    adapt = safe_num(df, adapt_col) if adapt_col else pd.Series(np.nan, index=df.index)

    u_norm = safe_num(df, "derived_u_norm")
    if u_norm.isna().all():
        u_norm = control_norm(df)

    i_ref = safe_num(df, "i_ref")
    final_i_ref = float(i_ref.iloc[-1]) if len(df) and np.isfinite(i_ref.iloc[-1]) else np.nan

    row = {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),
        "log_csv": str(run["log_csv"]),
        "enriched_log_csv": str(enriched_path),

        "bend_angle_deg": run["bend_angle_deg"],
        "bend_abs_deg": run["bend_abs_deg"],
        "jacobian_variant": run["jacobian_variant"],
        "solver_mode": run["solver_mode_from_name"],
        "rollout": roll,
        "Np": pred,
        "extra_horizon": pred - roll,
        "N_sqp": run["N_sqp"],

        "n_steps": int(len(df)),
        "success": bool((failure_mask == 0).all()) if len(df) else False,
        "num_failed_steps": int(failure_mask.sum()) if len(df) else 0,
        "num_infeasible_steps": int(infeasible.sum()) if len(df) else 0,
        "final_status": str(status.iloc[-1]) if len(df) else "",
        "final_i_ref": final_i_ref,
        "max_i_ref": float(i_ref.max()) if np.isfinite(i_ref).any() else np.nan,
        "progress_per_update": final_i_ref / len(df) if len(df) and np.isfinite(final_i_ref) else np.nan,

        # One-step local accuracy.
        "mean_one_step_xy_mm": float(one.mean()),
        "p50_one_step_xy_mm": nanp(one, 50),
        "p95_one_step_xy_mm": nanp(one, 95),
        "max_one_step_xy_mm": float(one.max()),

        "mean_one_step_xyz_mm": float(one_xyz.mean()),
        "p95_one_step_xyz_mm": nanp(one_xyz, 95),
        "max_one_step_xyz_mm": float(one_xyz.max()),

        # Raw/adaptive if present.
        "mean_raw_err_xy_mm": float(raw.mean()),
        "p95_raw_err_xy_mm": nanp(raw, 95),
        "max_raw_err_xy_mm": float(raw.max()),

        "mean_adapt_err_xy_mm": float(adapt.mean()),
        "p95_adapt_err_xy_mm": nanp(adapt, 95),
        "max_adapt_err_xy_mm": float(adapt.max()),
        "mean_adapt_minus_raw_xy_mm": float((adapt - raw).mean()) if raw.notna().any() and adapt.notna().any() else np.nan,

        # Multi-step rollout/prediction error.
        "num_logged_rollout_error_steps_xy": len(rollout_cols_xy),
        "deepest_logged_rollout_error_index_xy": int(rollout_cols_xy[-1][0]) if rollout_cols_xy else np.nan,

        "mean_rollout_all_xy_mm": nanmean(rollout_matrix_xy) if rollout_matrix_xy is not None else np.nan,
        "p95_rollout_all_xy_mm": nanp(rollout_matrix_xy, 95) if rollout_matrix_xy is not None else np.nan,
        "max_rollout_all_xy_mm": nanmax(rollout_matrix_xy) if rollout_matrix_xy is not None else np.nan,

        "mean_rollout_terminal_xy_mm": float(rollout_terminal_xy.mean()),
        "p95_rollout_terminal_xy_mm": nanp(rollout_terminal_xy, 95),
        "max_rollout_terminal_xy_mm": float(rollout_terminal_xy.max()),

        "mean_prediction_terminal_xy_mm": float(prediction_terminal_xy.mean()),
        "p95_prediction_terminal_xy_mm": nanp(prediction_terminal_xy, 95),
        "max_prediction_terminal_xy_mm": float(prediction_terminal_xy.max()),

        "mean_deepest_logged_xy_mm": float(deepest_logged_xy.mean()),
        "p95_deepest_logged_xy_mm": nanp(deepest_logged_xy, 95),
        "max_deepest_logged_xy_mm": float(deepest_logged_xy.max()),

        "mean_rollout_all_xyz_mm": nanmean(rollout_matrix_xyz) if rollout_matrix_xyz is not None else np.nan,
        "p95_rollout_all_xyz_mm": nanp(rollout_matrix_xyz, 95) if rollout_matrix_xyz is not None else np.nan,
        "max_rollout_all_xyz_mm": nanmax(rollout_matrix_xyz) if rollout_matrix_xyz is not None else np.nan,

        # Direction/gain diagnostics.
        "mean_cos_one_step_xy": float(safe_num(df, "derived_cos_one_step_xy").mean()),
        "min_cos_one_step_xy": float(safe_num(df, "derived_cos_one_step_xy").min()),
        "frac_negative_cos_one_step_xy": float((safe_num(df, "derived_cos_one_step_xy") < 0).mean()),
        "mean_direction_error_one_step_xy_deg": float(safe_num(df, "derived_direction_error_one_step_xy_deg").mean()),
        "p95_direction_error_one_step_xy_deg": nanp(safe_num(df, "derived_direction_error_one_step_xy_deg"), 95),
        "max_direction_error_one_step_xy_deg": float(safe_num(df, "derived_direction_error_one_step_xy_deg").max()),

        "mean_pred_gain_one_step_xy_mm_per_u": float(safe_num(df, "derived_pred_gain_one_step_xy_mm_per_u").mean()),
        "mean_actual_gain_one_step_xy_mm_per_u": float(safe_num(df, "derived_actual_gain_one_step_xy_mm_per_u").mean()),
        "median_gain_ratio_pred_over_actual_one_step_xy": nanp(safe_num(df, "derived_gain_ratio_pred_over_actual_one_step_xy"), 50),
        "mean_gain_ratio_log_error_one_step_xy": float(safe_num(df, "derived_gain_ratio_log_error_one_step_xy").mean()),
        "p95_gain_ratio_log_error_one_step_xy": nanp(safe_num(df, "derived_gain_ratio_log_error_one_step_xy"), 95),

        # Control/safety/conditioning.
        "mean_u_norm": float(u_norm.mean()),
        "p95_u_norm": nanp(u_norm, 95),
        "max_u_norm": float(u_norm.max()),

        "mean_clearance_mm": float(safe_num(df, "adapt_clearance_mm").mean()),
        "min_clearance_mm": float(safe_num(df, "adapt_clearance_mm").min()),
        "p05_clearance_mm": nanp(safe_num(df, "adapt_clearance_mm"), 5),
        "mean_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").mean()),
        "max_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").max()),

        "mean_cond_H_beam": float(safe_num(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_num(df, "cond_H_beam").max()),
        "mean_cond_H_mpc": float(safe_num(df, "cond_H_mpc").mean()),
        "max_cond_H_mpc": float(safe_num(df, "cond_H_mpc").max()),

        "mean_sqp_iters_done": float(safe_num(df, "sqp_iters_done").mean()),
        "max_sqp_iters_done": float(safe_num(df, "sqp_iters_done").max()),
        "sum_sqp_iters_done": float(safe_num(df, "sqp_iters_done").sum()),
        "mean_N_sqp_used": float(safe_num(df, "N_sqp_used").mean()),
        "max_N_sqp_used": float(safe_num(df, "N_sqp_used").max()),
        "sum_N_sqp_used": float(safe_num(df, "N_sqp_used").sum()),
        "mean_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").mean()),
        "max_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").max()),
    }

    return row


def build_horizon_error_detail(runs):
    rows = []

    for run in runs:
        df = run["df"]
        cols_xy = rollout_error_columns(df, kind="xy")
        cols_xyz = rollout_error_columns(df, kind="xyz")

        for kind, cols in [("xy", cols_xy), ("xyz", cols_xyz)]:
            for hidx, col in cols:
                s = safe_num(df, col)
                rows.append(
                    {
                        "run_name": run["run_name"],
                        "bend_abs_deg": run["bend_abs_deg"],
                        "jacobian_variant": run["jacobian_variant"],
                        "solver_mode": run["solver_mode_from_name"],
                        "rollout": int(run["rollout"]),
                        "Np": int(run["Np"]),
                        "extra_horizon": int(run["Np"]) - int(run["rollout"]),
                        "N_sqp": run["N_sqp"],
                        "kind": kind,
                        "horizon_index": int(hidx),
                        "horizon_step": int(hidx + 1),
                        "mean_err_mm": float(s.mean()),
                        "p50_err_mm": nanp(s, 50),
                        "p95_err_mm": nanp(s, 95),
                        "max_err_mm": float(s.max()),
                        "n": int(s.notna().sum()),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Aggregation/trend analysis
# ---------------------------------------------------------------------
LOWER_IS_BETTER = {
    "n_steps",
    "num_failed_steps",
    "num_infeasible_steps",
    "mean_one_step_xy_mm",
    "p95_one_step_xy_mm",
    "max_one_step_xy_mm",
    "mean_one_step_xyz_mm",
    "p95_one_step_xyz_mm",
    "max_one_step_xyz_mm",
    "mean_adapt_err_xy_mm",
    "p95_adapt_err_xy_mm",
    "max_adapt_err_xy_mm",
    "mean_rollout_all_xy_mm",
    "p95_rollout_all_xy_mm",
    "max_rollout_all_xy_mm",
    "mean_rollout_terminal_xy_mm",
    "p95_rollout_terminal_xy_mm",
    "max_rollout_terminal_xy_mm",
    "mean_prediction_terminal_xy_mm",
    "p95_prediction_terminal_xy_mm",
    "max_prediction_terminal_xy_mm",
    "mean_deepest_logged_xy_mm",
    "p95_deepest_logged_xy_mm",
    "max_deepest_logged_xy_mm",
    "mean_direction_error_one_step_xy_deg",
    "p95_direction_error_one_step_xy_deg",
    "max_direction_error_one_step_xy_deg",
    "frac_negative_cos_one_step_xy",
    "mean_gain_ratio_log_error_one_step_xy",
    "p95_gain_ratio_log_error_one_step_xy",
    "mean_u_norm",
    "p95_u_norm",
    "max_u_norm",
    "mean_tip_vessel_angle_deg",
    "max_tip_vessel_angle_deg",
    "mean_cond_H_beam",
    "max_cond_H_beam",
    "mean_cond_H_mpc",
    "max_cond_H_mpc",
    "mean_sqp_iters_done",
    "max_sqp_iters_done",
    "sum_sqp_iters_done",
    "mean_sqp_du_rel_final",
    "max_sqp_du_rel_final",
}

HIGHER_IS_BETTER = {
    "success",
    "final_i_ref",
    "max_i_ref",
    "progress_per_update",
    "mean_cos_one_step_xy",
    "min_cos_one_step_xy",
    "mean_clearance_mm",
    "min_clearance_mm",
    "p05_clearance_mm",
}


KEY_METRICS = [
    "n_steps",
    "progress_per_update",
    "mean_one_step_xy_mm",
    "p95_one_step_xy_mm",
    "max_one_step_xy_mm",
    "mean_adapt_err_xy_mm",
    "p95_adapt_err_xy_mm",
    "mean_rollout_all_xy_mm",
    "p95_rollout_all_xy_mm",
    "mean_rollout_terminal_xy_mm",
    "p95_rollout_terminal_xy_mm",
    "mean_prediction_terminal_xy_mm",
    "p95_prediction_terminal_xy_mm",
    "mean_cos_one_step_xy",
    "mean_direction_error_one_step_xy_deg",
    "mean_gain_ratio_log_error_one_step_xy",
    "mean_u_norm",
    "mean_clearance_mm",
    "min_clearance_mm",
    "mean_cond_H_mpc",
    "mean_sqp_iters_done",
    "mean_sqp_du_rel_final",
]


def aggregate_by_roll_pred(summary):
    group_cols = ["rollout", "Np", "extra_horizon"]
    available_metrics = [m for m in KEY_METRICS if m in summary.columns]
    agg = summary.groupby(group_cols, as_index=False)[available_metrics].mean(numeric_only=True)

    # Also record number of bend runs and success fraction.
    aux = summary.groupby(group_cols, as_index=False).agg(
        n_bend_runs=("run_name", "count"),
        success_fraction=("success", "mean"),
        mean_final_i_ref=("final_i_ref", "mean"),
    )
    return aux.merge(agg, on=group_cols, how="left")


def trend_by_roll(summary):
    rows = []
    metrics = [m for m in KEY_METRICS if m in summary.columns]

    # Aggregate over bends first so each rollout/Np point represents the sweep average.
    agg = aggregate_by_roll_pred(summary)

    for roll, g in agg.groupby("rollout"):
        g = g.sort_values("Np")
        for metric in metrics:
            vals = g[["Np", metric]].dropna()
            if len(vals) < 2:
                continue

            best_idx = vals[metric].idxmin() if metric in LOWER_IS_BETTER else vals[metric].idxmax()
            worst_idx = vals[metric].idxmax() if metric in LOWER_IS_BETTER else vals[metric].idxmin()

            first = vals.iloc[0][metric]
            last = vals.iloc[-1][metric]
            change = last - first

            if metric in LOWER_IS_BETTER:
                improvement_pct = (first - last) / abs(first) * 100.0 if abs(first) > 1e-12 else np.nan
            else:
                improvement_pct = (last - first) / abs(first) * 100.0 if abs(first) > 1e-12 else np.nan

            rows.append(
                {
                    "rollout": int(roll),
                    "metric": metric,
                    "Np_values": ",".join(str(int(v)) for v in vals["Np"].tolist()),
                    "metric_values": ",".join(f"{v:.6g}" for v in vals[metric].tolist()),
                    "slope_per_prediction_horizon": linear_slope(vals["Np"], vals[metric]),
                    "monotonicity": monotonicity_from_values(vals[metric]),
                    "first_Np": int(vals.iloc[0]["Np"]),
                    "last_Np": int(vals.iloc[-1]["Np"]),
                    "first_value": float(first),
                    "last_value": float(last),
                    "raw_change_last_minus_first": float(change),
                    "improvement_pct_positive_is_better": float(improvement_pct),
                    "best_Np": int(g.loc[best_idx, "Np"]),
                    "best_value": float(g.loc[best_idx, metric]),
                    "worst_Np": int(g.loc[worst_idx, "Np"]),
                    "worst_value": float(g.loc[worst_idx, metric]),
                }
            )

    return pd.DataFrame(rows)


def trend_by_pred(summary):
    rows = []
    metrics = [m for m in KEY_METRICS if m in summary.columns]
    agg = aggregate_by_roll_pred(summary)

    for pred, g in agg.groupby("Np"):
        g = g.sort_values("rollout")
        for metric in metrics:
            vals = g[["rollout", metric]].dropna()
            if len(vals) < 2:
                continue

            best_idx = vals[metric].idxmin() if metric in LOWER_IS_BETTER else vals[metric].idxmax()
            worst_idx = vals[metric].idxmax() if metric in LOWER_IS_BETTER else vals[metric].idxmin()

            first = vals.iloc[0][metric]
            last = vals.iloc[-1][metric]
            change = last - first

            if metric in LOWER_IS_BETTER:
                improvement_pct = (first - last) / abs(first) * 100.0 if abs(first) > 1e-12 else np.nan
            else:
                improvement_pct = (last - first) / abs(first) * 100.0 if abs(first) > 1e-12 else np.nan

            rows.append(
                {
                    "Np": int(pred),
                    "metric": metric,
                    "rollout_values": ",".join(str(int(v)) for v in vals["rollout"].tolist()),
                    "metric_values": ",".join(f"{v:.6g}" for v in vals[metric].tolist()),
                    "slope_per_rollout_step": linear_slope(vals["rollout"], vals[metric]),
                    "monotonicity": monotonicity_from_values(vals[metric]),
                    "first_rollout": int(vals.iloc[0]["rollout"]),
                    "last_rollout": int(vals.iloc[-1]["rollout"]),
                    "first_value": float(first),
                    "last_value": float(last),
                    "raw_change_last_minus_first": float(change),
                    "improvement_pct_positive_is_better": float(improvement_pct),
                    "best_rollout": int(g.loc[best_idx, "rollout"]),
                    "best_value": float(g.loc[best_idx, metric]),
                    "worst_rollout": int(g.loc[worst_idx, "rollout"]),
                    "worst_value": float(g.loc[worst_idx, metric]),
                }
            )

    return pd.DataFrame(rows)


def extra_horizon_summary(summary):
    metrics = [m for m in KEY_METRICS if m in summary.columns]
    g = summary.groupby(["extra_horizon"], as_index=False)[metrics].mean(numeric_only=True)
    aux = summary.groupby(["extra_horizon"], as_index=False).agg(
        n_runs=("run_name", "count"),
        mean_rollout=("rollout", "mean"),
        mean_Np=("Np", "mean"),
        success_fraction=("success", "mean"),
    )
    return aux.merge(g, on="extra_horizon", how="left")


def best_settings_table(agg):
    rows = []
    metrics = [m for m in KEY_METRICS if m in agg.columns]

    for metric in metrics:
        vals = agg[["rollout", "Np", "extra_horizon", metric, "n_bend_runs", "success_fraction"]].dropna()
        if vals.empty:
            continue

        ascending = metric in LOWER_IS_BETTER
        ranked = vals.sort_values(metric, ascending=ascending).head(6)

        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "rollout": int(row["rollout"]),
                    "Np": int(row["Np"]),
                    "extra_horizon": int(row["extra_horizon"]),
                    "value": float(row[metric]),
                    "n_bend_runs": int(row["n_bend_runs"]),
                    "success_fraction": float(row["success_fraction"]),
                    "optimisation_direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
METRIC_LABELS = {
    "n_steps": "MPC update count",
    "progress_per_update": "Progress per update",
    "mean_one_step_xy_mm": "Mean one-step XY error [mm]",
    "p95_one_step_xy_mm": "P95 one-step XY error [mm]",
    "max_one_step_xy_mm": "Max one-step XY error [mm]",
    "mean_adapt_err_xy_mm": "Mean adaptive error [mm]",
    "p95_adapt_err_xy_mm": "P95 adaptive error [mm]",
    "mean_rollout_all_xy_mm": "Mean all-horizon rollout XY error [mm]",
    "p95_rollout_all_xy_mm": "P95 all-horizon rollout XY error [mm]",
    "mean_rollout_terminal_xy_mm": "Mean rollout-terminal XY error [mm]",
    "p95_rollout_terminal_xy_mm": "P95 rollout-terminal XY error [mm]",
    "mean_prediction_terminal_xy_mm": "Mean Np-terminal XY error [mm]",
    "p95_prediction_terminal_xy_mm": "P95 Np-terminal XY error [mm]",
    "mean_cos_one_step_xy": "Mean cosine direction",
    "mean_direction_error_one_step_xy_deg": "Mean direction error [deg]",
    "mean_gain_ratio_log_error_one_step_xy": "Mean gain-ratio log error",
    "mean_u_norm": "Mean control norm",
    "mean_clearance_mm": "Mean clearance [mm]",
    "min_clearance_mm": "Minimum clearance [mm]",
    "mean_cond_H_mpc": "Mean MPC Hessian condition",
    "mean_sqp_iters_done": "Mean SQP iterations done",
    "mean_sqp_du_rel_final": "Mean final SQP relative input change",
}


def plot_heatmap(agg, metric, out_dir):
    if metric not in agg.columns:
        return

    mat = np.full((len(EXPECTED_ROLLOUTS), len(EXPECTED_PREDS)), np.nan)

    for _, row in agg.iterrows():
        r = int(row["rollout"])
        p = int(row["Np"])
        if r in EXPECTED_ROLLOUTS and p in EXPECTED_PREDS:
            mat[EXPECTED_ROLLOUTS.index(r), EXPECTED_PREDS.index(p)] = row[metric]

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    im = ax.imshow(mat, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(METRIC_LABELS.get(metric, metric))

    ax.set_xticks(range(len(EXPECTED_PREDS)))
    ax.set_xticklabels(EXPECTED_PREDS)
    ax.set_yticks(range(len(EXPECTED_ROLLOUTS)))
    ax.set_yticklabels(EXPECTED_ROLLOUTS)

    ax.set_xlabel("Prediction horizon $N_p$")
    ax.set_ylabel("Executed rollout")
    ax.set_title(METRIC_LABELS.get(metric, metric) + "\naveraged over bends, SQP budget fixed at 4")

    for i, r in enumerate(EXPECTED_ROLLOUTS):
        for j, p in enumerate(EXPECTED_PREDS):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=300)
    plt.close(fig)


def plot_metric_vs_np_by_roll(agg, metric, out_dir):
    if metric not in agg.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for roll, g in agg.groupby("rollout"):
        g = g.sort_values("Np")
        ax.plot(
            g["Np"],
            g[metric],
            marker="o",
            linewidth=1.8,
            label=f"rollout {int(roll)}",
        )

    ax.set_xlabel("Prediction horizon $N_p$")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric) + " vs prediction horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{metric}_vs_Np_by_rollout.png", dpi=300)
    plt.close(fig)


def plot_metric_vs_roll_by_np(agg, metric, out_dir):
    if metric not in agg.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for pred, g in agg.groupby("Np"):
        g = g.sort_values("rollout")
        ax.plot(
            g["rollout"],
            g[metric],
            marker="o",
            linewidth=1.8,
            label=f"$N_p$ {int(pred)}",
        )

    ax.set_xlabel("Executed rollout")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric) + " vs rollout")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{metric}_vs_rollout_by_Np.png", dpi=300)
    plt.close(fig)


def plot_metric_vs_extra_horizon(summary, metric, out_dir):
    if metric not in summary.columns:
        return

    g = summary.groupby(["extra_horizon", "rollout"], as_index=False)[metric].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for roll, gg in g.groupby("rollout"):
        gg = gg.sort_values("extra_horizon")
        ax.plot(
            gg["extra_horizon"],
            gg[metric],
            marker="o",
            linewidth=1.8,
            label=f"rollout {int(roll)}",
        )

    ax.set_xlabel("Extra prediction horizon beyond rollout, $N_p - rollout$")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric) + " vs extra horizon")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{metric}_vs_extra_horizon_by_rollout.png", dpi=300)
    plt.close(fig)


def plot_horizon_error_growth(horizon, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    hxy = horizon[horizon["kind"] == "xy"].copy()
    if hxy.empty:
        return

    # Averaged over bends for each rollout/Np setting.
    for (roll, pred), g0 in hxy.groupby(["rollout", "Np"]):
        g = g0.groupby("horizon_step", as_index=False).agg(
            mean_err_mm=("mean_err_mm", "mean"),
            p95_err_mm=("p95_err_mm", "mean"),
        ).sort_values("horizon_step")

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(g["horizon_step"], g["mean_err_mm"], marker="o", linewidth=1.8, label="mean")
        ax.plot(g["horizon_step"], g["p95_err_mm"], marker="s", linewidth=1.6, linestyle="--", label="P95")
        ax.set_xlabel("Prediction step inside horizon")
        ax.set_ylabel("Rollout XY error [mm]")
        ax.set_title(f"Horizon error growth, rollout {int(roll)}, $N_p$ {int(pred)}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(out_dir / f"horizon_error_growth_roll{int(roll)}_Np{int(pred)}.png", dpi=300)
        plt.close(fig)

    # A compact plot for each Np showing rollout lines.
    for pred, g0 in hxy.groupby("Np"):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))

        for roll, gr in g0.groupby("rollout"):
            g = gr.groupby("horizon_step", as_index=False)["mean_err_mm"].mean().sort_values("horizon_step")
            ax.plot(g["horizon_step"], g["mean_err_mm"], marker="o", linewidth=1.8, label=f"rollout {int(roll)}")

        ax.set_xlabel("Prediction step inside horizon")
        ax.set_ylabel("Mean rollout XY error [mm]")
        ax.set_title(f"Mean horizon error growth for $N_p$ {int(pred)}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"horizon_error_growth_by_rollout_Np{int(pred)}.png", dpi=300)
        plt.close(fig)


def plot_bend_specific_heatmaps(summary, metric, out_dir):
    if metric not in summary.columns:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for bend, g in summary.groupby("bend_abs_deg"):
        mat = np.full((len(EXPECTED_ROLLOUTS), len(EXPECTED_PREDS)), np.nan)
        for _, row in g.iterrows():
            r = int(row["rollout"])
            p = int(row["Np"])
            if r in EXPECTED_ROLLOUTS and p in EXPECTED_PREDS:
                mat[EXPECTED_ROLLOUTS.index(r), EXPECTED_PREDS.index(p)] = row[metric]

        fig, ax = plt.subplots(figsize=(7.0, 5.4))
        im = ax.imshow(mat, aspect="auto", origin="lower")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(METRIC_LABELS.get(metric, metric))

        ax.set_xticks(range(len(EXPECTED_PREDS)))
        ax.set_xticklabels(EXPECTED_PREDS)
        ax.set_yticks(range(len(EXPECTED_ROLLOUTS)))
        ax.set_yticklabels(EXPECTED_ROLLOUTS)

        ax.set_xlabel("Prediction horizon $N_p$")
        ax.set_ylabel("Executed rollout")
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}, {int(bend)} deg bend")

        for i, r in enumerate(EXPECTED_ROLLOUTS):
            for j, p in enumerate(EXPECTED_PREDS):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8)

        fig.tight_layout()
        fig.savefig(out_dir / f"bend_{int(bend)}_{metric}_heatmap.png", dpi=300)
        plt.close(fig)


def make_plots(summary, agg, horizon, out_dir):
    heatmap_metrics = [
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_rollout_all_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_rollout_terminal_xy_mm",
        "p95_rollout_terminal_xy_mm",
        "mean_prediction_terminal_xy_mm",
        "p95_prediction_terminal_xy_mm",
        "mean_direction_error_one_step_xy_deg",
        "mean_gain_ratio_log_error_one_step_xy",
        "mean_u_norm",
        "n_steps",
        "progress_per_update",
        "mean_clearance_mm",
        "min_clearance_mm",
        "mean_sqp_du_rel_final",
    ]

    for m in heatmap_metrics:
        plot_heatmap(agg, m, out_dir / "heatmaps")
        plot_metric_vs_np_by_roll(agg, m, out_dir / "line_plots")
        plot_metric_vs_roll_by_np(agg, m, out_dir / "line_plots")
        plot_metric_vs_extra_horizon(summary, m, out_dir / "line_plots")

    for m in [
        "p95_one_step_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_direction_error_one_step_xy_deg",
        "mean_u_norm",
        "n_steps",
        "min_clearance_mm",
    ]:
        plot_bend_specific_heatmaps(summary, m, out_dir / "bend_plots")

    plot_horizon_error_growth(horizon, out_dir / "horizon_plots")


# ---------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------
def write_readme(summary, agg, trends_roll, trends_pred, out_dir, skipped):
    readme = out_dir / "README_analysis_notes.txt"

    expected_valid = sum(1 for r in EXPECTED_ROLLOUTS for p in EXPECTED_PREDS if p >= r)
    expected_runs = expected_valid * len(sorted(summary["bend_abs_deg"].dropna().unique())) if len(summary) else np.nan

    with open(readme, "w") as f:
        f.write("Fixed SQP-4 rollout/prediction-horizon analysis\n")
        f.write("================================================\n\n")
        f.write(f"RUN_ROOT: {RUN_ROOT}\n")
        f.write(f"Fixed N_sqp: {FIXED_N_SQP}\n")
        f.write(f"Analysed runs: {len(summary)}\n")
        f.write(f"Expected triangular rollout/prediction settings: {expected_valid}\n")
        f.write(f"Expected runs if 5 bends per setting: {expected_runs}\n\n")

        f.write("Design interpretation\n")
        f.write("---------------------\n")
        f.write("The sweep is triangular: valid settings satisfy rollout <= Np. Therefore, higher rollout values have fewer accessible prediction horizons. The script separates three effects:\n")
        f.write("1. Increasing Np while holding rollout fixed.\n")
        f.write("2. Increasing rollout while holding Np fixed.\n")
        f.write("3. Increasing extra horizon Np - rollout.\n\n")

        f.write("Primary output files\n")
        f.write("--------------------\n")
        f.write("- sqp4_roll_pred_run_summary.csv: one row per bend/run.\n")
        f.write("- sqp4_roll_pred_aggregate_by_roll_pred.csv: averaged over bends for each rollout/Np setting.\n")
        f.write("- sqp4_roll_pred_trends_by_roll.csv: whether metrics improve/worsen as Np increases for fixed rollout.\n")
        f.write("- sqp4_roll_pred_trends_by_pred.csv: whether metrics improve/worsen as rollout increases for fixed Np.\n")
        f.write("- sqp4_roll_pred_extra_horizon_summary.csv: effect of Np - rollout.\n")
        f.write("- sqp4_roll_pred_horizon_error_detail.csv: error at each predicted step inside the horizon.\n")
        f.write("- sqp4_roll_pred_best_settings.csv: ranked best settings for each metric.\n\n")

        f.write("Suggested first plots\n")
        f.write("---------------------\n")
        f.write("- heatmaps/heatmap_p95_one_step_xy_mm.png\n")
        f.write("- heatmaps/heatmap_p95_rollout_all_xy_mm.png\n")
        f.write("- heatmaps/heatmap_mean_direction_error_one_step_xy_deg.png\n")
        f.write("- heatmaps/heatmap_mean_u_norm.png\n")
        f.write("- heatmaps/heatmap_n_steps.png\n")
        f.write("- heatmaps/heatmap_min_clearance_mm.png\n")
        f.write("- line_plots/p95_one_step_xy_mm_vs_Np_by_rollout.png\n")
        f.write("- line_plots/p95_rollout_all_xy_mm_vs_extra_horizon_by_rollout.png\n")
        f.write("- horizon_plots/horizon_error_growth_by_rollout_Np6.png\n\n")

        f.write("Skipped logs\n")
        f.write("------------\n")
        if skipped:
            for path, reason in skipped[:100]:
                f.write(f"{reason}: {path}\n")
            if len(skipped) > 100:
                f.write(f"... {len(skipped)-100} more skipped logs\n")
        else:
            f.write("None.\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs, skipped = discover_logs(RUN_ROOT)
    if not runs:
        raise RuntimeError(
            f"No matching SQP-{FIXED_N_SQP} roll_*/pred_* log.csv files found under {RUN_ROOT}"
        )

    rows = [summarise_run(run) for run in runs]
    summary = pd.DataFrame(rows).sort_values(["rollout", "Np", "bend_abs_deg"])
    summary.to_csv(OUT_DIR / "sqp4_roll_pred_run_summary.csv", index=False)

    with open(OUT_DIR / "sqp4_roll_pred_run_summary.json", "w") as f:
        json.dump(summary.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    horizon = build_horizon_error_detail(runs)
    horizon.to_csv(OUT_DIR / "sqp4_roll_pred_horizon_error_detail.csv", index=False)

    agg = aggregate_by_roll_pred(summary)
    agg.to_csv(OUT_DIR / "sqp4_roll_pred_aggregate_by_roll_pred.csv", index=False)

    trends_roll = trend_by_roll(summary)
    trends_roll.to_csv(OUT_DIR / "sqp4_roll_pred_trends_by_roll.csv", index=False)

    trends_pred = trend_by_pred(summary)
    trends_pred.to_csv(OUT_DIR / "sqp4_roll_pred_trends_by_pred.csv", index=False)

    extra = extra_horizon_summary(summary)
    extra.to_csv(OUT_DIR / "sqp4_roll_pred_extra_horizon_summary.csv", index=False)

    best = best_settings_table(agg)
    best.to_csv(OUT_DIR / "sqp4_roll_pred_best_settings.csv", index=False)

    make_plots(summary, agg, horizon, OUT_DIR)
    write_readme(summary, agg, trends_roll, trends_pred, OUT_DIR, skipped)

    print("\n[FIXED SQP-4 ROLLOUT/PREDICTION-HORIZON SWEEP]")
    print(f"Analysed runs: {len(summary)}")
    print(f"Unique rollouts: {sorted(summary['rollout'].dropna().unique().astype(int).tolist())}")
    print(f"Unique Np:       {sorted(summary['Np'].dropna().unique().astype(int).tolist())}")
    print(f"Unique bends:    {sorted(summary['bend_abs_deg'].dropna().unique().astype(int).tolist())}")
    print(f"Skipped logs:    {len(skipped)}")

    display_cols = [
        "rollout",
        "Np",
        "extra_horizon",
        "n_bend_runs",
        "success_fraction",
        "mean_final_i_ref",
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_rollout_all_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_direction_error_one_step_xy_deg",
        "mean_u_norm",
        "n_steps",
        "mean_clearance_mm",
        "min_clearance_mm",
    ]

    display_cols = [c for c in display_cols if c in agg.columns]
    print("\n[AGGREGATE BY ROLLOUT AND PREDICTION HORIZON]")
    print(agg[display_cols].to_string(index=False))

    print("\n[RECOMMENDED FIRST OUTPUTS]")
    print(OUT_DIR / "sqp4_roll_pred_aggregate_by_roll_pred.csv")
    print(OUT_DIR / "sqp4_roll_pred_trends_by_roll.csv")
    print(OUT_DIR / "sqp4_roll_pred_trends_by_pred.csv")
    print(OUT_DIR / "sqp4_roll_pred_extra_horizon_summary.csv")
    print(OUT_DIR / "heatmaps" / "heatmap_p95_one_step_xy_mm.png")
    print(OUT_DIR / "heatmaps" / "heatmap_p95_rollout_all_xy_mm.png")
    print(OUT_DIR / "line_plots" / "p95_one_step_xy_mm_vs_Np_by_rollout.png")
    print(OUT_DIR / "horizon_plots" / "horizon_error_growth_by_rollout_Np6.png")

    print(f"\nSaved analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()
