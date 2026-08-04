"""
Reference-matched analysis for the fixed SQP-4 rollout / prediction-horizon sweep.

Why this exists
---------------
The ordinary analysis averages over log rows. That can be unfair when rollout is
larger, because larger rollout can skip intermediate reference indices. A method
can look better simply because it did not sample a difficult reference region.

This script therefore compares settings only on shared reference points (`i_ref`).

Expected layout
---------------
evidence_contact_sqp_budget_sweep/
    roll_1/pred_1/.../log.csv
    roll_1/pred_2/.../log.csv
    ...
    roll_6/pred_6/.../log.csv

It analyses only:
    - roll_*/pred_* triangular sweep
    - N_sqp = 4
    - sqp_full
    - jac_contact
    - pred >= rollout

Outputs
-------
evidence_contact_sqp_budget_sweep/
    fixed_sqp4_reference_matched_analysis/
        reference_step_metrics.csv
        reference_point_summary.csv
        reference_coverage_by_run.csv

        matched_all_settings_by_bend.csv
        matched_all_settings_aggregate.csv

        matched_fixed_rollout_by_bend.csv
        matched_fixed_rollout_aggregate.csv

        matched_fixed_Np_by_bend.csv
        matched_fixed_Np_aggregate.csv

        pairwise_reference_matched_comparisons.csv

        heatmaps/
        line_plots/
        coverage_plots/
"""

from pathlib import Path
import re
import itertools
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------
RUN_ROOT = Path("evidence_contact_sqp_budget_sweep")
OUT_DIR = RUN_ROOT / "fixed_sqp4_reference_matched_analysis"

FIXED_N_SQP = 4
EXPECTED_ROLLOUTS = list(range(1, 7))
EXPECTED_PREDS = list(range(1, 7))

REQUIRE_PRED_GE_ROLL = True
ONLY_CONTACT_JACOBIAN = True
ONLY_SQP_FULL = True

# If i_ref is integer-like, the script uses integer reference keys. If not,
# it rounds to this many decimal places.
I_REF_ROUND_DECIMALS = 6

# Minimum number of common reference points needed for a matched comparison.
MIN_COMMON_REFS = 2


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def safe_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index)


def nanp(values, q):
    a = np.asarray(values, dtype=float)
    if np.isfinite(a).any():
        return float(np.nanpercentile(a, q))
    return np.nan


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

    roll_from_path = np.nan
    pred_from_path = np.nan

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
        "solver_mode": parse_solver_mode(run_name),
        "rollout": rollout,
        "Np": pred,
        "extra_horizon": pred - rollout if np.isfinite(pred) and np.isfinite(rollout) else np.nan,
        "N_sqp": sqp_from_name,
    }


def discover_logs(run_root: Path):
    run_root = Path(run_root)
    candidates = []

    for roll in EXPECTED_ROLLOUTS:
        roll_dir = run_root / f"roll_{roll}"
        if not roll_dir.exists():
            warnings.warn(f"Missing expected rollout folder: {roll_dir}")
            continue

        for pred in EXPECTED_PREDS:
            if REQUIRE_PRED_GE_ROLL and pred < roll:
                continue

            pred_dir = roll_dir / f"pred_{pred}"
            if pred_dir.exists():
                candidates.extend(sorted(pred_dir.glob("*/log.csv")))

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

        if ONLY_SQP_FULL and meta["solver_mode"] != "sqp_full":
            skipped.append((str(log_csv), f"solver={meta['solver_mode']}"))
            continue

        if REQUIRE_PRED_GE_ROLL and int(meta["Np"]) < int(meta["rollout"]):
            skipped.append((str(log_csv), "Np < rollout"))
            continue

        try:
            meta["df"] = pd.read_csv(log_csv)
        except Exception as e:
            skipped.append((str(log_csv), f"read_error={e}"))
            continue

        runs.append(meta)

    return runs, skipped


# ---------------------------------------------------------------------
# Per-row metric extraction
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

    if cols:
        U = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return pd.Series(np.linalg.norm(U, axis=1), index=df.index)

    for c in [f"{prefix}_norm", "u_norm", "control_norm"]:
        if c in df.columns:
            return safe_num(df, c)

    return pd.Series(np.nan, index=df.index)


def infer_position_scale_to_mm(df: pd.DataFrame):
    cols = [c for c in ["x_rollout_0_x", "x_rollout_0_y", "x_rollout_0_z"] if c in df.columns]
    if not cols:
        return 1.0

    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med_abs = np.nanmedian(np.abs(vals))

    # Most of your state coordinates appear to be metres, while logged errors are mm.
    if np.isfinite(med_abs) and med_abs < 1.0:
        return 1000.0

    return 1.0


def derive_one_step_direction_and_gain(df: pd.DataFrame):
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

    gain_log_error = np.full(len(df), np.nan)
    valid_ratio = np.isfinite(gain_ratio) & (gain_ratio > 1e-12)
    gain_log_error[valid_ratio] = np.abs(np.log10(gain_ratio[valid_ratio]))

    return pd.DataFrame(
        {
            "cos_one_step_xy": cos,
            "direction_error_one_step_xy_deg": direction_error_deg,
            "pred_gain_one_step_xy_mm_per_u": pred_gain,
            "actual_gain_one_step_xy_mm_per_u": actual_gain,
            "gain_ratio_pred_over_actual_one_step_xy": gain_ratio,
            "gain_ratio_log_error_one_step_xy": gain_log_error,
            "u_norm": u_norm,
        },
        index=df.index,
    )


def make_ref_key(i_ref_series: pd.Series):
    i_ref = pd.to_numeric(i_ref_series, errors="coerce")

    rounded_int = np.round(i_ref)
    near_integer = np.nanmax(np.abs(i_ref - rounded_int)) < 1e-8 if i_ref.notna().any() else False

    if near_integer:
        return rounded_int.astype("Int64")

    return i_ref.round(I_REF_ROUND_DECIMALS)


def build_step_metrics(runs):
    rows = []

    for run in runs:
        df = run["df"].copy()
        derived = derive_one_step_direction_and_gain(df)
        if len(derived.columns):
            df = pd.concat([df, derived], axis=1)

        roll = int(run["rollout"])
        pred = int(run["Np"])

        one_step_col = choose_first_existing(
            df,
            ["pred1_err_xy_mm", "rollout_err_xy_0_mm", "adapt_pred_err_xy_mm"],
        )
        one_step_xyz_col = choose_first_existing(
            df,
            ["pred1_err_xyz_mm", "rollout_err_xyz_0_mm", "adapt_pred_err_xyz_mm"],
        )
        raw_col = choose_first_existing(
            df,
            ["raw_one_step_err_xy_mm", "raw_pred_err_xy_mm", "raw_err_xy_mm", "pred1_raw_err_xy_mm"],
        )
        adapt_col = choose_first_existing(
            df,
            ["adapt_pred_err_xy_mm", "adaptive_pred_err_xy_mm"],
        )

        rollout_cols_xy = rollout_error_columns(df, kind="xy")
        rollout_lookup_xy = {idx: col for idx, col in rollout_cols_xy}

        rollout_matrix_xy = None
        if rollout_cols_xy:
            rollout_matrix_xy = df[[c for _, c in rollout_cols_xy]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        ref_key = make_ref_key(safe_num(df, "i_ref"))

        local = pd.DataFrame(
            {
                "run_name": run["run_name"],
                "run_dir": str(run["run_dir"]),
                "log_csv": str(run["log_csv"]),
                "bend_abs_deg": run["bend_abs_deg"],
                "bend_angle_deg": run["bend_angle_deg"],
                "jacobian_variant": run["jacobian_variant"],
                "solver_mode": run["solver_mode"],
                "rollout": roll,
                "Np": pred,
                "extra_horizon": pred - roll,
                "N_sqp": run["N_sqp"],
                "step_index": np.arange(len(df)),
                "i_ref": safe_num(df, "i_ref"),
                "ref_key": ref_key,
                "status": safe_str(df, "status"),
                "infeasible": safe_num(df, "infeasible"),
                "one_step_xy_mm": safe_num(df, one_step_col) if one_step_col else np.nan,
                "one_step_xyz_mm": safe_num(df, one_step_xyz_col) if one_step_xyz_col else np.nan,
                "raw_err_xy_mm": safe_num(df, raw_col) if raw_col else np.nan,
                "adapt_err_xy_mm": safe_num(df, adapt_col) if adapt_col else np.nan,
                "rollout_terminal_xy_mm": safe_num(df, rollout_lookup_xy.get(roll - 1, "")) if (roll - 1) in rollout_lookup_xy else np.nan,
                "prediction_terminal_xy_mm": safe_num(df, rollout_lookup_xy.get(pred - 1, "")) if (pred - 1) in rollout_lookup_xy else np.nan,
                "deepest_logged_xy_mm": safe_num(df, rollout_cols_xy[-1][1]) if rollout_cols_xy else np.nan,
                "cos_one_step_xy": safe_num(df, "cos_one_step_xy"),
                "direction_error_one_step_xy_deg": safe_num(df, "direction_error_one_step_xy_deg"),
                "gain_ratio_log_error_one_step_xy": safe_num(df, "gain_ratio_log_error_one_step_xy"),
                "u_norm": safe_num(df, "u_norm"),
                "clearance_mm": safe_num(df, "adapt_clearance_mm"),
                "tip_vessel_angle_deg": safe_num(df, "adapt_tip_vessel_angle_deg"),
                "cond_H_mpc": safe_num(df, "cond_H_mpc"),
                "cond_H_beam": safe_num(df, "cond_H_beam"),
                "sqp_iters_done": safe_num(df, "sqp_iters_done"),
                "sqp_du_rel_final": safe_num(df, "sqp_du_rel_final"),
            }
        )

        if rollout_matrix_xy is not None:
            local["rollout_all_xy_mm"] = np.nanmean(rollout_matrix_xy, axis=1)
            local["rollout_all_xy_mm_max_over_horizon"] = np.nanmax(rollout_matrix_xy, axis=1)
        else:
            local["rollout_all_xy_mm"] = np.nan
            local["rollout_all_xy_mm_max_over_horizon"] = np.nan

        rows.append(local)

    if not rows:
        return pd.DataFrame()

    step = pd.concat(rows, ignore_index=True)
    step = step[pd.notna(step["ref_key"])].copy()

    return step


# ---------------------------------------------------------------------
# Reference summarisation
# ---------------------------------------------------------------------
SETTING_COLS = ["rollout", "Np", "extra_horizon"]
ID_COLS = [
    "run_name",
    "run_dir",
    "log_csv",
    "bend_abs_deg",
    "bend_angle_deg",
    "jacobian_variant",
    "solver_mode",
    "N_sqp",
] + SETTING_COLS

BASE_METRICS = [
    "one_step_xy_mm",
    "one_step_xyz_mm",
    "raw_err_xy_mm",
    "adapt_err_xy_mm",
    "rollout_all_xy_mm",
    "rollout_all_xy_mm_max_over_horizon",
    "rollout_terminal_xy_mm",
    "prediction_terminal_xy_mm",
    "deepest_logged_xy_mm",
    "cos_one_step_xy",
    "direction_error_one_step_xy_deg",
    "gain_ratio_log_error_one_step_xy",
    "u_norm",
    "clearance_mm",
    "tip_vessel_angle_deg",
    "cond_H_mpc",
    "cond_H_beam",
    "sqp_iters_done",
    "sqp_du_rel_final",
]


def build_reference_point_summary(step):
    """
    Collapse duplicate visits to the same reference index within each run.

    This prevents a setting from being weighted more heavily just because it
    oscillates or spends multiple MPC updates at one reference point.
    """
    metric_cols = [m for m in BASE_METRICS if m in step.columns]

    grouped = (
        step.groupby(ID_COLS + ["ref_key"], dropna=False)
        .agg(
            **{m: (m, "mean") for m in metric_cols},
            n_updates_at_ref=("step_index", "count"),
            first_step_index=("step_index", "min"),
            last_step_index=("step_index", "max"),
            mean_i_ref=("i_ref", "mean"),
        )
        .reset_index()
    )

    return grouped


def build_reference_coverage(ref_summary):
    rows = []

    for keys, g in ref_summary.groupby(ID_COLS, dropna=False):
        d = dict(zip(ID_COLS, keys))
        refs = sorted(pd.to_numeric(g["ref_key"], errors="coerce").dropna().unique())

        if len(refs) > 1:
            gaps = np.diff(refs)
            mean_gap = float(np.mean(gaps))
            max_gap = float(np.max(gaps))
        else:
            mean_gap = np.nan
            max_gap = np.nan

        d.update(
            {
                "n_unique_refs": int(len(refs)),
                "ref_min": float(np.min(refs)) if refs else np.nan,
                "ref_max": float(np.max(refs)) if refs else np.nan,
                "mean_ref_gap": mean_gap,
                "max_ref_gap": max_gap,
                "total_updates": int(g["n_updates_at_ref"].sum()),
                "mean_updates_per_ref": float(g["n_updates_at_ref"].mean()),
                "ref_list": ",".join(str(int(r)) if abs(r - round(r)) < 1e-8 else f"{r:g}" for r in refs),
            }
        )
        rows.append(d)

    return pd.DataFrame(rows)


def metric_summary(series, prefix):
    s = pd.to_numeric(series, errors="coerce")
    return {
        f"mean_{prefix}": float(s.mean()),
        f"p50_{prefix}": nanp(s, 50),
        f"p95_{prefix}": nanp(s, 95),
        f"max_{prefix}": float(s.max()) if s.notna().any() else np.nan,
    }


def clearance_summary(series):
    s = pd.to_numeric(series, errors="coerce")
    return {
        "mean_clearance_mm": float(s.mean()),
        "p05_clearance_mm": nanp(s, 5),
        "min_clearance_mm": float(s.min()) if s.notna().any() else np.nan,
    }


def summarise_metrics_over_refs(g):
    out = {}

    # Errors / lower-is-better metrics.
    for col, prefix in [
        ("one_step_xy_mm", "one_step_xy_mm"),
        ("one_step_xyz_mm", "one_step_xyz_mm"),
        ("raw_err_xy_mm", "raw_err_xy_mm"),
        ("adapt_err_xy_mm", "adapt_err_xy_mm"),
        ("rollout_all_xy_mm", "rollout_all_xy_mm"),
        ("rollout_all_xy_mm_max_over_horizon", "rollout_all_xy_mm_max_over_horizon"),
        ("rollout_terminal_xy_mm", "rollout_terminal_xy_mm"),
        ("prediction_terminal_xy_mm", "prediction_terminal_xy_mm"),
        ("deepest_logged_xy_mm", "deepest_logged_xy_mm"),
        ("direction_error_one_step_xy_deg", "direction_error_one_step_xy_deg"),
        ("gain_ratio_log_error_one_step_xy", "gain_ratio_log_error_one_step_xy"),
        ("u_norm", "u_norm"),
        ("tip_vessel_angle_deg", "tip_vessel_angle_deg"),
        ("cond_H_mpc", "cond_H_mpc"),
        ("cond_H_beam", "cond_H_beam"),
        ("sqp_du_rel_final", "sqp_du_rel_final"),
    ]:
        if col in g.columns:
            out.update(metric_summary(g[col], prefix))

    # Cosine: higher is better, so include mean and minimum rather than p95.
    if "cos_one_step_xy" in g.columns:
        s = pd.to_numeric(g["cos_one_step_xy"], errors="coerce")
        out["mean_cos_one_step_xy"] = float(s.mean())
        out["min_cos_one_step_xy"] = float(s.min()) if s.notna().any() else np.nan
        out["frac_negative_cos_one_step_xy"] = float((s < 0).mean()) if s.notna().any() else np.nan

    # Clearance: higher is better, so include mean, 5th percentile and minimum.
    if "clearance_mm" in g.columns:
        out.update(clearance_summary(g["clearance_mm"]))

    if "sqp_iters_done" in g.columns:
        s = pd.to_numeric(g["sqp_iters_done"], errors="coerce")
        out["mean_sqp_iters_done"] = float(s.mean())
        out["max_sqp_iters_done"] = float(s.max()) if s.notna().any() else np.nan

    out["n_reference_points_used"] = int(g["ref_key"].nunique())
    out["n_updates_represented"] = int(g["n_updates_at_ref"].sum()) if "n_updates_at_ref" in g.columns else np.nan
    out["mean_updates_per_ref"] = float(g["n_updates_at_ref"].mean()) if "n_updates_at_ref" in g.columns else np.nan
    out["ref_min_used"] = float(pd.to_numeric(g["ref_key"], errors="coerce").min())
    out["ref_max_used"] = float(pd.to_numeric(g["ref_key"], errors="coerce").max())

    return out


def compute_matched_table(ref_summary, scope_cols, setting_cols, label):
    """
    For each scope, intersect reference keys across all settings in the scope,
    then summarise every setting using only those common reference keys.

    Examples
    --------
    scope_cols = ["bend_abs_deg"]
        strictest comparison: all rollout/Np settings within a bend share refs.

    scope_cols = ["bend_abs_deg", "rollout"], setting_cols = ["Np"]
        compare prediction horizons fairly for each fixed rollout.

    scope_cols = ["bend_abs_deg", "Np"], setting_cols = ["rollout"]
        compare rollouts fairly for each fixed prediction horizon.
    """
    rows = []

    all_cols = scope_cols + setting_cols
    for scope_key, gscope in ref_summary.groupby(scope_cols, dropna=False):
        if not isinstance(scope_key, tuple):
            scope_key = (scope_key,)
        scope_dict = dict(zip(scope_cols, scope_key))

        setting_groups = []
        setting_ref_sets = []

        for setting_key, gset in gscope.groupby(setting_cols, dropna=False):
            if not isinstance(setting_key, tuple):
                setting_key = (setting_key,)
            setting_dict = dict(zip(setting_cols, setting_key))
            refs = set(pd.to_numeric(gset["ref_key"], errors="coerce").dropna().tolist())

            if refs:
                setting_groups.append((setting_dict, gset))
                setting_ref_sets.append(refs)

        if len(setting_ref_sets) < 2:
            continue

        common_refs = set.intersection(*setting_ref_sets)
        if len(common_refs) < MIN_COMMON_REFS:
            continue

        common_refs_sorted = sorted(common_refs)

        for setting_dict, gset in setting_groups:
            gmatched = gset[pd.to_numeric(gset["ref_key"], errors="coerce").isin(common_refs)].copy()

            row = {}
            row.update(scope_dict)
            row.update(setting_dict)

            # Fill missing setting columns from gset if they are constant but not explicit.
            for c in SETTING_COLS:
                if c not in row and c in gset.columns and gset[c].nunique(dropna=True) == 1:
                    row[c] = gset[c].dropna().iloc[0]

            row["match_scope"] = label
            row["n_common_refs_in_scope"] = int(len(common_refs_sorted))
            row["common_ref_min"] = float(min(common_refs_sorted))
            row["common_ref_max"] = float(max(common_refs_sorted))
            row["common_ref_list"] = ",".join(
                str(int(r)) if abs(r - round(r)) < 1e-8 else f"{r:g}"
                for r in common_refs_sorted
            )

            row.update(summarise_metrics_over_refs(gmatched))
            rows.append(row)

    return pd.DataFrame(rows)


def aggregate_matched_over_bends(matched, group_cols):
    if matched.empty:
        return matched

    metric_cols = [
        c for c in matched.columns
        if c not in (
            group_cols
            + [
                "bend_abs_deg",
                "bend_angle_deg",
                "match_scope",
                "common_ref_list",
                "ref_min_used",
                "ref_max_used",
            ]
        )
        and pd.api.types.is_numeric_dtype(matched[c])
    ]

    agg = matched.groupby(group_cols, as_index=False)[metric_cols].mean(numeric_only=True)
    aux = matched.groupby(group_cols, as_index=False).agg(
        n_bends=("bend_abs_deg", "nunique"),
        mean_common_refs=("n_common_refs_in_scope", "mean"),
        min_common_refs=("n_common_refs_in_scope", "min"),
    )

    return aux.merge(agg, on=group_cols, how="left")


# ---------------------------------------------------------------------
# Pairwise matched comparisons
# ---------------------------------------------------------------------
LOWER_IS_BETTER = {
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
    "mean_direction_error_one_step_xy_deg",
    "p95_direction_error_one_step_xy_deg",
    "mean_gain_ratio_log_error_one_step_xy",
    "p95_gain_ratio_log_error_one_step_xy",
    "mean_u_norm",
    "p95_u_norm",
    "mean_tip_vessel_angle_deg",
    "mean_cond_H_mpc",
    "mean_sqp_du_rel_final",
}
HIGHER_IS_BETTER = {
    "mean_cos_one_step_xy",
    "min_cos_one_step_xy",
    "mean_clearance_mm",
    "p05_clearance_mm",
    "min_clearance_mm",
}


def pairwise_reference_matched(ref_summary):
    rows = []
    metric_base_cols = [m for m in BASE_METRICS if m in ref_summary.columns]

    for bend, gb in ref_summary.groupby("bend_abs_deg"):
        settings = []
        for (roll, pred), gs in gb.groupby(["rollout", "Np"]):
            refs = set(pd.to_numeric(gs["ref_key"], errors="coerce").dropna().tolist())
            settings.append(((int(roll), int(pred)), refs, gs))

        for (s1, refs1, g1), (s2, refs2, g2) in itertools.combinations(settings, 2):
            common = sorted(refs1 & refs2)
            if len(common) < MIN_COMMON_REFS:
                continue

            g1m = g1[pd.to_numeric(g1["ref_key"], errors="coerce").isin(common)]
            g2m = g2[pd.to_numeric(g2["ref_key"], errors="coerce").isin(common)]

            # Summarise means over common refs for each base metric.
            for metric in metric_base_cols:
                v1 = pd.to_numeric(g1m[metric], errors="coerce").mean()
                v2 = pd.to_numeric(g2m[metric], errors="coerce").mean()

                if not np.isfinite(v1) or not np.isfinite(v2):
                    continue

                # Positive means second setting improves over first.
                if metric in ["cos_one_step_xy", "clearance_mm"]:
                    improvement = (v2 - v1) / abs(v1) * 100.0 if abs(v1) > 1e-12 else np.nan
                    signed_delta_positive_is_better = v2 - v1
                else:
                    improvement = (v1 - v2) / abs(v1) * 100.0 if abs(v1) > 1e-12 else np.nan
                    signed_delta_positive_is_better = v1 - v2

                rows.append(
                    {
                        "bend_abs_deg": bend,
                        "metric": metric,
                        "rollout_A": s1[0],
                        "Np_A": s1[1],
                        "rollout_B": s2[0],
                        "Np_B": s2[1],
                        "A_value": float(v1),
                        "B_value": float(v2),
                        "B_minus_A_raw": float(v2 - v1),
                        "signed_B_improvement_positive_is_better": float(signed_delta_positive_is_better),
                        "B_improvement_pct_positive_is_better": float(improvement),
                        "n_common_refs": int(len(common)),
                        "common_ref_min": float(min(common)),
                        "common_ref_max": float(max(common)),
                        "common_ref_list": ",".join(
                            str(int(r)) if abs(r - round(r)) < 1e-8 else f"{r:g}"
                            for r in common
                        ),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
METRIC_LABELS = {
    "n_unique_refs": "Number of unique reference points sampled",
    "mean_ref_gap": "Mean reference-index gap",
    "mean_one_step_xy_mm": "Matched mean one-step XY error [mm]",
    "p95_one_step_xy_mm": "Matched P95 one-step XY error [mm]",
    "mean_adapt_err_xy_mm": "Matched mean adaptive error [mm]",
    "p95_adapt_err_xy_mm": "Matched P95 adaptive error [mm]",
    "mean_rollout_all_xy_mm": "Matched mean all-horizon rollout error [mm]",
    "p95_rollout_all_xy_mm": "Matched P95 all-horizon rollout error [mm]",
    "mean_rollout_terminal_xy_mm": "Matched mean rollout-terminal error [mm]",
    "p95_rollout_terminal_xy_mm": "Matched P95 rollout-terminal error [mm]",
    "mean_prediction_terminal_xy_mm": "Matched mean prediction-terminal error [mm]",
    "p95_prediction_terminal_xy_mm": "Matched P95 prediction-terminal error [mm]",
    "mean_direction_error_one_step_xy_deg": "Matched mean direction error [deg]",
    "mean_cos_one_step_xy": "Matched mean cosine direction",
    "mean_gain_ratio_log_error_one_step_xy": "Matched mean gain-ratio log error",
    "mean_u_norm": "Matched mean control norm",
    "mean_clearance_mm": "Matched mean clearance [mm]",
    "min_clearance_mm": "Matched minimum clearance [mm]",
    "mean_sqp_du_rel_final": "Matched mean final SQP input change",
    "n_common_refs_in_scope": "Number of common reference points",
    "mean_common_refs": "Mean common reference count",
}


def heatmap_from_table(table, metric, out_dir, title_suffix="", prefix="heatmap"):
    if table.empty or metric not in table.columns:
        return

    mat = np.full((len(EXPECTED_ROLLOUTS), len(EXPECTED_PREDS)), np.nan)

    for _, row in table.iterrows():
        if "rollout" not in row or "Np" not in row:
            continue

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
    ax.set_title(METRIC_LABELS.get(metric, metric) + title_suffix)

    for i, r in enumerate(EXPECTED_ROLLOUTS):
        for j, p in enumerate(EXPECTED_PREDS):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{prefix}_{metric}.png", dpi=300)
    plt.close(fig)


def plot_line_vs_np_by_roll(table, metric, out_dir, title_suffix="", prefix="line"):
    if table.empty or metric not in table.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for roll, g in table.groupby("rollout"):
        g = g.sort_values("Np")
        ax.plot(g["Np"], g[metric], marker="o", linewidth=1.8, label=f"rollout {int(roll)}")

    ax.set_xlabel("Prediction horizon $N_p$")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric) + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{prefix}_{metric}_vs_Np_by_rollout.png", dpi=300)
    plt.close(fig)


def plot_line_vs_roll_by_np(table, metric, out_dir, title_suffix="", prefix="line"):
    if table.empty or metric not in table.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for pred, g in table.groupby("Np"):
        g = g.sort_values("rollout")
        ax.plot(g["rollout"], g[metric], marker="o", linewidth=1.8, label=f"$N_p$ {int(pred)}")

    ax.set_xlabel("Executed rollout")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(METRIC_LABELS.get(metric, metric) + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{prefix}_{metric}_vs_rollout_by_Np.png", dpi=300)
    plt.close(fig)


def plot_coverage(coverage, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_agg = (
        coverage.groupby(["rollout", "Np", "extra_horizon"], as_index=False)
        .agg(
            n_unique_refs=("n_unique_refs", "mean"),
            mean_ref_gap=("mean_ref_gap", "mean"),
            max_ref_gap=("max_ref_gap", "mean"),
            total_updates=("total_updates", "mean"),
            mean_updates_per_ref=("mean_updates_per_ref", "mean"),
        )
    )

    for metric in ["n_unique_refs", "mean_ref_gap", "max_ref_gap", "total_updates", "mean_updates_per_ref"]:
        heatmap_from_table(
            coverage_agg,
            metric,
            out_dir,
            title_suffix="\ncoverage averaged over bends",
            prefix="coverage_heatmap",
        )


def make_plots(coverage, matched_all_agg, matched_fixed_roll_agg, matched_fixed_np_agg, out_dir):
    plot_coverage(coverage, out_dir / "coverage_plots")

    heatmap_metrics = [
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_adapt_err_xy_mm",
        "p95_adapt_err_xy_mm",
        "mean_rollout_all_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_rollout_terminal_xy_mm",
        "p95_rollout_terminal_xy_mm",
        "mean_prediction_terminal_xy_mm",
        "p95_prediction_terminal_xy_mm",
        "mean_direction_error_one_step_xy_deg",
        "mean_cos_one_step_xy",
        "mean_gain_ratio_log_error_one_step_xy",
        "mean_u_norm",
        "mean_clearance_mm",
        "min_clearance_mm",
        "mean_common_refs",
    ]

    for metric in heatmap_metrics:
        heatmap_from_table(
            matched_all_agg,
            metric,
            out_dir / "heatmaps",
            title_suffix="\nstrict reference-matched aggregate",
            prefix="strict_matched_heatmap",
        )

    for metric in [
        "mean_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_rollout_all_xy_mm",
        "p95_rollout_all_xy_mm",
        "mean_direction_error_one_step_xy_deg",
        "mean_u_norm",
        "mean_clearance_mm",
        "min_clearance_mm",
    ]:
        plot_line_vs_np_by_roll(
            matched_fixed_roll_agg,
            metric,
            out_dir / "line_plots",
            title_suffix="\nreference-matched within each fixed rollout",
            prefix="fixed_rollout_matched",
        )
        plot_line_vs_roll_by_np(
            matched_fixed_np_agg,
            metric,
            out_dir / "line_plots",
            title_suffix="\nreference-matched within each fixed $N_p$",
            prefix="fixed_Np_matched",
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs, skipped = discover_logs(RUN_ROOT)

    if not runs:
        raise RuntimeError(f"No matching fixed SQP-{FIXED_N_SQP} logs found under {RUN_ROOT}")

    step = build_step_metrics(runs)
    step.to_csv(OUT_DIR / "reference_step_metrics.csv", index=False)

    ref_summary = build_reference_point_summary(step)
    ref_summary.to_csv(OUT_DIR / "reference_point_summary.csv", index=False)

    coverage = build_reference_coverage(ref_summary)
    coverage.to_csv(OUT_DIR / "reference_coverage_by_run.csv", index=False)

    # Strictest comparison: within each bend, compare all rollout/Np settings on
    # the intersection of reference points shared by every setting.
    matched_all = compute_matched_table(
        ref_summary,
        scope_cols=["bend_abs_deg"],
        setting_cols=["rollout", "Np", "extra_horizon"],
        label="all_settings_within_bend",
    )
    matched_all.to_csv(OUT_DIR / "matched_all_settings_by_bend.csv", index=False)

    matched_all_agg = aggregate_matched_over_bends(
        matched_all,
        group_cols=["rollout", "Np", "extra_horizon"],
    )
    matched_all_agg.to_csv(OUT_DIR / "matched_all_settings_aggregate.csv", index=False)

    # Compare Np values fairly for each fixed rollout.
    matched_fixed_roll = compute_matched_table(
        ref_summary,
        scope_cols=["bend_abs_deg", "rollout"],
        setting_cols=["Np", "extra_horizon"],
        label="fixed_rollout_vary_Np",
    )
    matched_fixed_roll.to_csv(OUT_DIR / "matched_fixed_rollout_by_bend.csv", index=False)

    matched_fixed_roll_agg = aggregate_matched_over_bends(
        matched_fixed_roll,
        group_cols=["rollout", "Np", "extra_horizon"],
    )
    matched_fixed_roll_agg.to_csv(OUT_DIR / "matched_fixed_rollout_aggregate.csv", index=False)

    # Compare rollout values fairly for each fixed Np.
    matched_fixed_np = compute_matched_table(
        ref_summary,
        scope_cols=["bend_abs_deg", "Np"],
        setting_cols=["rollout", "extra_horizon"],
        label="fixed_Np_vary_rollout",
    )
    matched_fixed_np.to_csv(OUT_DIR / "matched_fixed_Np_by_bend.csv", index=False)

    matched_fixed_np_agg = aggregate_matched_over_bends(
        matched_fixed_np,
        group_cols=["Np", "rollout", "extra_horizon"],
    )
    # Save in rollout/Np order for easier reading.
    if not matched_fixed_np_agg.empty:
        matched_fixed_np_agg = matched_fixed_np_agg.sort_values(["rollout", "Np"])
    matched_fixed_np_agg.to_csv(OUT_DIR / "matched_fixed_Np_aggregate.csv", index=False)

    pairwise = pairwise_reference_matched(ref_summary)
    pairwise.to_csv(OUT_DIR / "pairwise_reference_matched_comparisons.csv", index=False)

    make_plots(
        coverage=coverage,
        matched_all_agg=matched_all_agg,
        matched_fixed_roll_agg=matched_fixed_roll_agg,
        matched_fixed_np_agg=matched_fixed_np_agg,
        out_dir=OUT_DIR,
    )

    notes = {
        "analysed_runs": int(len(runs)),
        "analysed_rows": int(len(step)),
        "reference_point_rows": int(len(ref_summary)),
        "coverage_rows": int(len(coverage)),
        "matched_all_settings_rows": int(len(matched_all)),
        "matched_fixed_rollout_rows": int(len(matched_fixed_roll)),
        "matched_fixed_Np_rows": int(len(matched_fixed_np)),
        "pairwise_rows": int(len(pairwise)),
        "skipped_logs": skipped,
        "important_note": (
            "Matched tables first collapse duplicate visits to the same i_ref within a run. "
            "They then compare settings only over common i_ref values, preventing larger rollouts "
            "from looking artificially better because they skipped difficult reference points."
        ),
    }

    with open(OUT_DIR / "reference_matched_analysis_notes.json", "w") as f:
        json.dump(notes, f, indent=2)

    print("\n[REFERENCE-MATCHED FIXED SQP-4 ANALYSIS]")
    print(f"Analysed runs:                {len(runs)}")
    print(f"Step rows:                    {len(step)}")
    print(f"Run/reference rows:           {len(ref_summary)}")
    print(f"Matched all-settings rows:    {len(matched_all)}")
    print(f"Matched fixed-rollout rows:   {len(matched_fixed_roll)}")
    print(f"Matched fixed-Np rows:        {len(matched_fixed_np)}")
    print(f"Pairwise comparison rows:     {len(pairwise)}")
    print(f"Skipped logs:                 {len(skipped)}")

    print("\n[KEY OUTPUTS]")
    print(OUT_DIR / "reference_coverage_by_run.csv")
    print(OUT_DIR / "matched_all_settings_aggregate.csv")
    print(OUT_DIR / "matched_fixed_rollout_aggregate.csv")
    print(OUT_DIR / "matched_fixed_Np_aggregate.csv")
    print(OUT_DIR / "pairwise_reference_matched_comparisons.csv")
    print(OUT_DIR / "coverage_plots" / "coverage_heatmap_n_unique_refs.png")
    print(OUT_DIR / "heatmaps" / "strict_matched_heatmap_p95_one_step_xy_mm.png")
    print(OUT_DIR / "line_plots" / "fixed_rollout_matched_p95_one_step_xy_mm_vs_Np_by_rollout.png")
    print(OUT_DIR / "line_plots" / "fixed_Np_matched_p95_one_step_xy_mm_vs_rollout_by_Np.png")
    print(f"\nSaved analysis to: {OUT_DIR}")


if __name__ == "__main__":
    main()