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
# Expected layout:
# contact_vs_no_contact_sim1/
#   bend_m30_jac_contact_plant_contact_1_lti_rollout1_Np1_sqp4/log.csv
#   bend_m30_jac_no_contact_plant_contact_1_lti_rollout1_Np1_sqp4/log.csv
#   ...
RUN_ROOT = Path("sim1_contact_vs_non_contact")
OUT_DIR = RUN_ROOT / "lti_contact_vs_no_contact_analysis"

# The first logged row usually has no previous actual state, so derived
# displacement/gain/cosine metrics are undefined for that row.
EPS = 1e-12


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


def parse_bend_angle_from_run_name(run_name: str) -> float:
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def parse_jacobian_variant(run_name: str) -> str:
    # Important: match no_contact before contact.
    m = re.search(r"_jac_(no_contact|contact)_", run_name)
    if m:
        return m.group(1)
    return "unknown"


def parse_plant_contact(run_name: str):
    m = re.search(r"_plant_contact_(\d+)_", run_name)
    if m:
        return bool(int(m.group(1)))
    return np.nan


def parse_int_token(run_name: str, token: str):
    m = re.search(rf"_{token}(\d+)(?:_|$)", run_name)
    if m:
        return int(m.group(1))
    return np.nan


def parse_solver_mode(run_name: str) -> str:
    # Your folders currently contain _lti_, but this keeps the parser reusable.
    for mode in ["sqp_full", "ltv_oneshot", "lti"]:
        if f"_{mode}_" in run_name:
            return mode
    return "unknown"


def bend_tag(bend_angle: float) -> str:
    if not np.isfinite(bend_angle):
        return "unknown"
    prefix = "m" if bend_angle < 0 else "p"
    return f"{prefix}{int(abs(bend_angle))}"


def count_switches(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) <= 1:
        return 0
    return int(s.ne(s.shift()).sum() - 1)


def finite_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def finite_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.median()) if x.notna().any() else np.nan


def finite_max(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.max()) if x.notna().any() else np.nan


def finite_min(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    return float(x.min()) if x.notna().any() else np.nan


def pct_improvement_lower_is_better(contact_value, no_contact_value):
    """Positive means contact has lower/better value than no_contact."""
    if not np.isfinite(contact_value) or not np.isfinite(no_contact_value) or abs(no_contact_value) < EPS:
        return np.nan
    return 100.0 * (no_contact_value - contact_value) / abs(no_contact_value)


def pct_change_contact_vs_no_contact(contact_value, no_contact_value):
    """Positive means contact is larger than no_contact."""
    if not np.isfinite(contact_value) or not np.isfinite(no_contact_value) or abs(no_contact_value) < EPS:
        return np.nan
    return 100.0 * (contact_value - no_contact_value) / abs(no_contact_value)


# ---------------------------------------------------------------------
# Derived per-step metrics
# ---------------------------------------------------------------------
def first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def vec_from_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def norm_rows(A):
    return np.linalg.norm(A, axis=1)


def cosine_rows(A, B):
    na = norm_rows(A)
    nb = norm_rows(B)
    den = na * nb
    out = np.full(len(A), np.nan, dtype=float)
    good = np.isfinite(den) & (den > EPS)
    out[good] = np.sum(A[good] * B[good], axis=1) / den[good]
    return np.clip(out, -1.0, 1.0)


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns useful for contact-vs-no-contact LTI comparison.

    Raw/one-step error:
      pred1_err_xy_mm if available, otherwise rollout_err_xy_0_mm.

    Adaptive error:
      adapt_pred_err_xy_mm / adapt_pred_err_xyz_mm.

    Direction cosine:
      compares predicted one-step displacement direction against actual one-step
      displacement direction. Since the logged x_rollout_0 equals the realised
      state for that row in your sample log, the actual displacement is computed
      relative to the previous logged actual state. The first row is NaN.

    Jacobian/effective gain proxy:
      predicted or actual displacement magnitude per applied control norm.
      This is not a true singular value of J unless you explicitly log J; it is
      the local realised/predicted displacement-per-command gain.
    """
    df = df.copy()

    # Error aliases.
    raw_xy_col = first_existing_col(df, ["pred1_err_xy_mm", "rollout_err_xy_0_mm"])
    raw_xyz_col = first_existing_col(df, ["pred1_err_xyz_mm", "rollout_err_xyz_0_mm"])
    adapt_xy_col = first_existing_col(df, ["adapt_pred_err_xy_mm", raw_xy_col])
    adapt_xyz_col = first_existing_col(df, ["adapt_pred_err_xyz_mm", raw_xyz_col])

    df["raw_one_step_err_xy_mm"] = safe_num(df, raw_xy_col) if raw_xy_col else np.nan
    df["raw_one_step_err_xyz_mm"] = safe_num(df, raw_xyz_col) if raw_xyz_col else np.nan
    df["adapt_err_xy_mm"] = safe_num(df, adapt_xy_col) if adapt_xy_col else np.nan
    df["adapt_err_xyz_mm"] = safe_num(df, adapt_xyz_col) if adapt_xyz_col else np.nan

    df["adapt_minus_raw_xy_mm"] = df["adapt_err_xy_mm"] - df["raw_one_step_err_xy_mm"]
    df["adapt_over_raw_xy"] = df["adapt_err_xy_mm"] / df["raw_one_step_err_xy_mm"].where(
        df["raw_one_step_err_xy_mm"].abs() > EPS
    )

    # State vectors: use x_rollout_0 as realised one-step state if available;
    # otherwise fall back to tip position.
    actual_xyz = vec_from_cols(df, ["x_rollout_0_x", "x_rollout_0_y", "x_rollout_0_z"])
    if actual_xyz is None:
        actual_xyz = vec_from_cols(df, ["tip_x", "tip_y", "tip_z"])

    pred_xyz = vec_from_cols(df, ["x_pred_0_x", "x_pred_0_y", "x_pred_0_z"])

    if actual_xyz is not None and pred_xyz is not None:
        prev_actual_xyz = np.vstack([np.full((1, 3), np.nan), actual_xyz[:-1]])
        actual_delta_xyz = actual_xyz - prev_actual_xyz
        pred_delta_xyz = pred_xyz - prev_actual_xyz

        actual_delta_xy = actual_delta_xyz[:, :2]
        pred_delta_xy = pred_delta_xyz[:, :2]

        df["actual_step_norm_xy_mm"] = 1000.0 * norm_rows(actual_delta_xy)
        df["pred_step_norm_xy_mm"] = 1000.0 * norm_rows(pred_delta_xy)
        df["actual_step_norm_xyz_mm"] = 1000.0 * norm_rows(actual_delta_xyz)
        df["pred_step_norm_xyz_mm"] = 1000.0 * norm_rows(pred_delta_xyz)

        df["cos_pred_actual_xy"] = cosine_rows(pred_delta_xy, actual_delta_xy)
        df["cos_pred_actual_xyz"] = cosine_rows(pred_delta_xyz, actual_delta_xyz)
        df["direction_error_xy_deg"] = np.degrees(np.arccos(df["cos_pred_actual_xy"]))
        df["direction_error_xyz_deg"] = np.degrees(np.arccos(df["cos_pred_actual_xyz"]))
    else:
        for c in [
            "actual_step_norm_xy_mm",
            "pred_step_norm_xy_mm",
            "actual_step_norm_xyz_mm",
            "pred_step_norm_xyz_mm",
            "cos_pred_actual_xy",
            "cos_pred_actual_xyz",
            "direction_error_xy_deg",
            "direction_error_xyz_deg",
        ]:
            df[c] = np.nan

    # Applied command norm. Include all 7 controls if present.
    u_cols = [
        "u_applied_0_vx", "u_applied_0_vy", "u_applied_0_vz",
        "u_applied_0_wx", "u_applied_0_wy", "u_applied_0_wz",
        "u_applied_0_dL",
    ]
    u = vec_from_cols(df, u_cols)
    if u is None:
        # Fall back to planned first control if applied controls were not logged.
        u_cols = [
            "u_plan_0_vx", "u_plan_0_vy", "u_plan_0_vz",
            "u_plan_0_wx", "u_plan_0_wy", "u_plan_0_wz",
            "u_plan_0_dL",
        ]
        u = vec_from_cols(df, u_cols)

    if u is not None:
        u_norm = norm_rows(u)
        u_norm_safe = np.where(u_norm > EPS, u_norm, np.nan)
        df["u_norm"] = u_norm
        df["pred_gain_xy_mm_per_u"] = df["pred_step_norm_xy_mm"] / u_norm_safe
        df["actual_gain_xy_mm_per_u"] = df["actual_step_norm_xy_mm"] / u_norm_safe
        df["gain_ratio_pred_over_actual_xy"] = df["pred_gain_xy_mm_per_u"] / df[
            "actual_gain_xy_mm_per_u"
        ].where(df["actual_gain_xy_mm_per_u"].abs() > EPS)
    else:
        df["u_norm"] = np.nan
        df["pred_gain_xy_mm_per_u"] = np.nan
        df["actual_gain_xy_mm_per_u"] = np.nan
        df["gain_ratio_pred_over_actual_xy"] = np.nan

    # Optional exact columns if your logger later adds them.
    explicit_gain_col = first_existing_col(df, [
        "jacobian_gain", "jacobian_gain_xy", "jac_gain", "J_gain",
    ])
    if explicit_gain_col:
        df["logged_jacobian_gain"] = safe_num(df, explicit_gain_col)
    else:
        df["logged_jacobian_gain"] = np.nan

    return df


# ---------------------------------------------------------------------
# Reading and summarising runs
# ---------------------------------------------------------------------
def read_runs(run_root: Path):
    runs = []
    for log_csv in sorted(Path(run_root).glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name
        df = pd.read_csv(log_csv)
        df = add_derived_metrics(df)

        bend_angle = parse_bend_angle_from_run_name(run_name)
        runs.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "log_csv": log_csv,
                "bend_angle_deg": bend_angle,
                "bend_abs_deg": abs(bend_angle) if np.isfinite(bend_angle) else np.nan,
                "jacobian_variant": parse_jacobian_variant(run_name),
                "plant_contact": parse_plant_contact(run_name),
                "solver_mode_from_name": parse_solver_mode(run_name),
                "rollout_from_name": parse_int_token(run_name, "rollout"),
                "Np_from_name": parse_int_token(run_name, "Np"),
                "sqp_from_name": parse_int_token(run_name, "sqp"),
                "df": df,
            }
        )
    return runs


def summarise_run(run) -> dict:
    df = run["df"]

    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    failure_mask = (
        (infeasible > 0)
        | (~status.isin(["solved", "solved_inaccurate", "solved inaccurate"]))
    )

    solver_used = safe_str(df, "solver_mode_used")
    if (solver_used == "").all():
        solver_used = safe_str(df, "solver_mode")

    # Count rollout error columns present/populated.
    rollout_xy_cols = [c for c in df.columns if re.match(r"rollout_err_xy_\d+_mm", c)]
    rollout_xyz_cols = [c for c in df.columns if re.match(r"rollout_err_xyz_\d+_mm", c)]

    def rollout_mean_max(cols):
        if not cols:
            return np.nan, np.nan
        E = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        if not np.isfinite(E).any():
            return np.nan, np.nan
        return float(np.nanmean(E)), float(np.nanmax(E))

    mean_rollout_xy, max_rollout_xy = rollout_mean_max(rollout_xy_cols)
    mean_rollout_xyz, max_rollout_xyz = rollout_mean_max(rollout_xyz_cols)

    row = {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),
        "bend_angle_deg": run["bend_angle_deg"],
        "bend_abs_deg": run["bend_abs_deg"],
        "bend_tag": bend_tag(run["bend_angle_deg"]),
        "jacobian_variant": run["jacobian_variant"],
        "plant_contact": run["plant_contact"],
        "solver_mode_from_name": run["solver_mode_from_name"],
        "rollout_from_name": run["rollout_from_name"],
        "Np_from_name": run["Np_from_name"],
        "sqp_from_name": run["sqp_from_name"],

        "n_steps": int(len(df)),
        "success": bool((failure_mask == 0).all()) if len(df) else False,
        "num_failed_steps": int(failure_mask.sum()) if len(df) else 0,
        "num_infeasible_steps": int(infeasible.sum()) if len(df) else 0,
        "final_status": str(status.iloc[-1]) if len(df) else "",
        "final_i_ref": float(safe_num(df, "i_ref").iloc[-1]) if len(df) else np.nan,
        "max_i_ref": finite_max(safe_num(df, "i_ref")),

        # Core requested error metrics.
        "mean_one_step_xy_mm": finite_mean(df["raw_one_step_err_xy_mm"]),
        "median_one_step_xy_mm": finite_median(df["raw_one_step_err_xy_mm"]),
        "max_one_step_xy_mm": finite_max(df["raw_one_step_err_xy_mm"]),
        "p95_one_step_xy_mm": float(df["raw_one_step_err_xy_mm"].quantile(0.95))
            if df["raw_one_step_err_xy_mm"].notna().any() else np.nan,
        "mean_one_step_xyz_mm": finite_mean(df["raw_one_step_err_xyz_mm"]),
        "max_one_step_xyz_mm": finite_max(df["raw_one_step_err_xyz_mm"]),

        "mean_rollout_err_xy_mm": mean_rollout_xy,
        "max_rollout_err_xy_mm": max_rollout_xy,
        "mean_rollout_err_xyz_mm": mean_rollout_xyz,
        "max_rollout_err_xyz_mm": max_rollout_xyz,

        "mean_adapt_err_xy_mm": finite_mean(df["adapt_err_xy_mm"]),
        "median_adapt_err_xy_mm": finite_median(df["adapt_err_xy_mm"]),
        "max_adapt_err_xy_mm": finite_max(df["adapt_err_xy_mm"]),
        "p95_adapt_err_xy_mm": float(df["adapt_err_xy_mm"].quantile(0.95))
            if df["adapt_err_xy_mm"].notna().any() else np.nan,
        "mean_adapt_err_xyz_mm": finite_mean(df["adapt_err_xyz_mm"]),
        "max_adapt_err_xyz_mm": finite_max(df["adapt_err_xyz_mm"]),
        "mean_adapt_minus_raw_xy_mm": finite_mean(df["adapt_minus_raw_xy_mm"]),
        "mean_adapt_over_raw_xy": finite_mean(df["adapt_over_raw_xy"]),

        # Direction metrics.
        "mean_cos_pred_actual_xy": finite_mean(df["cos_pred_actual_xy"]),
        "median_cos_pred_actual_xy": finite_median(df["cos_pred_actual_xy"]),
        "min_cos_pred_actual_xy": finite_min(df["cos_pred_actual_xy"]),
        "frac_negative_cos_xy": float((df["cos_pred_actual_xy"] < 0).mean())
            if df["cos_pred_actual_xy"].notna().any() else np.nan,
        "mean_direction_error_xy_deg": finite_mean(df["direction_error_xy_deg"]),
        "max_direction_error_xy_deg": finite_max(df["direction_error_xy_deg"]),

        # Jacobian/effective gain proxy.
        "mean_pred_gain_xy_mm_per_u": finite_mean(df["pred_gain_xy_mm_per_u"]),
        "median_pred_gain_xy_mm_per_u": finite_median(df["pred_gain_xy_mm_per_u"]),
        "max_pred_gain_xy_mm_per_u": finite_max(df["pred_gain_xy_mm_per_u"]),
        "mean_actual_gain_xy_mm_per_u": finite_mean(df["actual_gain_xy_mm_per_u"]),
        "median_actual_gain_xy_mm_per_u": finite_median(df["actual_gain_xy_mm_per_u"]),
        "max_actual_gain_xy_mm_per_u": finite_max(df["actual_gain_xy_mm_per_u"]),
        "mean_gain_ratio_pred_over_actual_xy": finite_mean(df["gain_ratio_pred_over_actual_xy"]),
        "median_gain_ratio_pred_over_actual_xy": finite_median(df["gain_ratio_pred_over_actual_xy"]),
        "mean_logged_jacobian_gain": finite_mean(df["logged_jacobian_gain"]),

        # Other useful safety/conditioning/control diagnostics.
        "mean_clearance_mm": finite_mean(safe_num(df, "clearance_mm")),
        "min_clearance_mm": finite_min(safe_num(df, "clearance_mm")),
        "mean_adapt_clearance_mm": finite_mean(safe_num(df, "adapt_clearance_mm")),
        "min_adapt_clearance_mm": finite_min(safe_num(df, "adapt_clearance_mm")),
        "mean_tip_vessel_angle_deg": finite_mean(safe_num(df, "tip_vessel_angle_deg")),
        "max_tip_vessel_angle_deg": finite_max(safe_num(df, "tip_vessel_angle_deg")),
        "mean_cond_H_beam": finite_mean(safe_num(df, "cond_H_beam")),
        "max_cond_H_beam": finite_max(safe_num(df, "cond_H_beam")),
        "mean_cond_H_mpc": finite_mean(safe_num(df, "cond_H_mpc")),
        "max_cond_H_mpc": finite_max(safe_num(df, "cond_H_mpc")),
        "mean_sqp_du_rel_final": finite_mean(safe_num(df, "sqp_du_rel_final")),
        "max_sqp_du_rel_final": finite_max(safe_num(df, "sqp_du_rel_final")),
        "mean_u_norm": finite_mean(df["u_norm"]),
        "max_u_norm": finite_max(df["u_norm"]),
        "mean_sqp_iters_done": finite_mean(safe_num(df, "sqp_iters_done")),
        "sum_sqp_iters_done": float(safe_num(df, "sqp_iters_done").sum(skipna=True)),
        "unique_solver_modes": ",".join(sorted(solver_used.dropna().unique())),
        "num_solver_switches": count_switches(solver_used),
    }

    return row


def build_summary(runs, out_dir: Path) -> pd.DataFrame:
    rows = [summarise_run(r) for r in runs]
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["bend_abs_deg", "jacobian_variant"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "lti_contact_vs_no_contact_run_summary.csv", index=False)
    with open(out_dir / "lti_contact_vs_no_contact_run_summary.json", "w") as f:
        json.dump(summary.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)
    return summary


def build_pairwise_comparison(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics_lower_better = [
        "mean_one_step_xy_mm",
        "median_one_step_xy_mm",
        "max_one_step_xy_mm",
        "p95_one_step_xy_mm",
        "mean_adapt_err_xy_mm",
        "median_adapt_err_xy_mm",
        "max_adapt_err_xy_mm",
        "p95_adapt_err_xy_mm",
        "mean_rollout_err_xy_mm",
        "max_rollout_err_xy_mm",
        "mean_direction_error_xy_deg",
        "max_direction_error_xy_deg",
        "mean_cond_H_beam",
        "max_cond_H_beam",
        "mean_cond_H_mpc",
        "max_cond_H_mpc",
        "mean_sqp_du_rel_final",
        "max_sqp_du_rel_final",
        "mean_u_norm",
        "sum_sqp_iters_done",
    ]

    metrics_higher_better = [
        "mean_cos_pred_actual_xy",
        "median_cos_pred_actual_xy",
        "min_cos_pred_actual_xy",
        "mean_clearance_mm",
        "min_clearance_mm",
        "mean_adapt_clearance_mm",
        "min_adapt_clearance_mm",
    ]

    all_metrics = metrics_lower_better + metrics_higher_better + [
        "mean_pred_gain_xy_mm_per_u",
        "mean_actual_gain_xy_mm_per_u",
        "median_gain_ratio_pred_over_actual_xy",
        "mean_gain_ratio_pred_over_actual_xy",
        "final_i_ref",
        "max_i_ref",
    ]

    rows = []
    group_cols = ["bend_abs_deg", "bend_angle_deg", "bend_tag"]
    for keys, g in summary.groupby(group_cols, dropna=False):
        if not {"contact", "no_contact"}.issubset(set(g["jacobian_variant"])):
            continue
        c = g[g["jacobian_variant"] == "contact"].iloc[0]
        n = g[g["jacobian_variant"] == "no_contact"].iloc[0]
        base = dict(zip(group_cols, keys))
        for m in all_metrics:
            cv = float(c[m]) if m in c and pd.notna(c[m]) else np.nan
            nv = float(n[m]) if m in n and pd.notna(n[m]) else np.nan
            row = {
                **base,
                "metric": m,
                "contact": cv,
                "no_contact": nv,
                "contact_minus_no_contact": cv - nv if np.isfinite(cv) and np.isfinite(nv) else np.nan,
                "pct_change_contact_vs_no_contact": pct_change_contact_vs_no_contact(cv, nv),
            }
            if m in metrics_lower_better:
                row["contact_improvement_pct"] = pct_improvement_lower_is_better(cv, nv)
                row["interpretation"] = "positive_improvement_means_contact_lower"
            elif m in metrics_higher_better:
                row["contact_improvement_pct"] = pct_change_contact_vs_no_contact(cv, nv)
                row["interpretation"] = "positive_improvement_means_contact_higher"
            else:
                row["contact_improvement_pct"] = np.nan
                row["interpretation"] = "diagnostic_ratio_or_progress_metric"
            rows.append(row)

    comparison = pd.DataFrame(rows)
    if len(comparison):
        comparison = comparison.sort_values(["bend_abs_deg", "metric"]).reset_index(drop=True)
    comparison.to_csv(out_dir / "lti_contact_vs_no_contact_pairwise_comparison.csv", index=False)
    with open(out_dir / "lti_contact_vs_no_contact_pairwise_comparison.json", "w") as f:
        json.dump(comparison.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)
    return comparison


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def prepare_plot_df(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    cols = ["bend_abs_deg", "bend_tag", "jacobian_variant", metric]
    g = summary[cols].dropna(subset=[metric]).copy()
    g = g[g["jacobian_variant"].isin(["contact", "no_contact"])]
    return g.sort_values(["bend_abs_deg", "jacobian_variant"])


def grouped_bar(summary: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path):
    g = prepare_plot_df(summary, metric)
    if g.empty:
        return

    bends = sorted(g["bend_abs_deg"].dropna().unique())
    variants = ["no_contact", "contact"]
    x = np.arange(len(bends))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, variant in enumerate(variants):
        vals = []
        for bend in bends:
            s = g[(g["bend_abs_deg"] == bend) & (g["jacobian_variant"] == variant)][metric]
            vals.append(float(s.iloc[0]) if len(s) else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=variant.replace("_", " "))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}°" for b in bends])
    ax.set_xlabel("Bend magnitude")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_summary_bars(summary: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("mean_one_step_xy_mm", "Mean one-step XY error [mm]", "Mean one-step prediction error"),
        ("max_one_step_xy_mm", "Max one-step XY error [mm]", "Worst one-step prediction error"),
        ("mean_adapt_err_xy_mm", "Mean adaptive XY error [mm]", "Mean adaptive prediction error"),
        ("max_adapt_err_xy_mm", "Max adaptive XY error [mm]", "Worst adaptive prediction error"),
        ("mean_cos_pred_actual_xy", "Mean cosine(pred, actual)", "Direction alignment"),
        ("min_cos_pred_actual_xy", "Minimum cosine(pred, actual)", "Worst direction alignment"),
        ("mean_direction_error_xy_deg", "Mean direction error [deg]", "Mean direction error"),
        ("mean_pred_gain_xy_mm_per_u", "Predicted gain [mm / ||u||]", "Predicted effective Jacobian gain"),
        ("mean_actual_gain_xy_mm_per_u", "Actual gain [mm / ||u||]", "Actual realised gain"),
        ("median_gain_ratio_pred_over_actual_xy", "Median gain ratio", "Predicted / actual gain"),
        ("min_clearance_mm", "Minimum clearance [mm]", "Minimum clearance"),
        ("mean_cond_H_beam", "Mean beam Hessian condition", "Beam conditioning"),
        ("mean_cond_H_mpc", "Mean MPC Hessian condition", "MPC conditioning"),
        ("mean_u_norm", "Mean ||u||", "Control effort"),
    ]
    for metric, ylabel, title in specs:
        if metric in summary.columns:
            grouped_bar(summary, metric, ylabel, title, out_dir / f"{metric}_by_bend.png")


def plot_timeline_per_bend(runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_bend = {}
    for run in runs:
        by_bend.setdefault(run["bend_abs_deg"], []).append(run)

    for bend, bend_runs in sorted(by_bend.items(), key=lambda kv: kv[0]):
        fig, axes = plt.subplots(4, 1, figsize=(9.0, 10.5), sharex=True)

        for run in sorted(bend_runs, key=lambda r: r["jacobian_variant"]):
            df = run["df"]
            label = run["jacobian_variant"].replace("_", " ")
            x = safe_num(df, "i_ref") if "i_ref" in df.columns else pd.Series(df.index, index=df.index)

            valid = x.notna() & df["raw_one_step_err_xy_mm"].notna()
            axes[0].plot(x[valid], df.loc[valid, "raw_one_step_err_xy_mm"], marker="o", markersize=3, label=label)
            axes[0].set_ylabel("Raw / one-step\nXY error [mm]")

            valid = x.notna() & df["adapt_err_xy_mm"].notna()
            axes[1].plot(x[valid], df.loc[valid, "adapt_err_xy_mm"], marker="o", markersize=3, label=label)
            axes[1].set_ylabel("Adaptive\nXY error [mm]")

            valid = x.notna() & df["cos_pred_actual_xy"].notna()
            axes[2].plot(x[valid], df.loc[valid, "cos_pred_actual_xy"], marker="o", markersize=3, label=label)
            axes[2].axhline(0.0, linewidth=1.0, linestyle="--")
            axes[2].set_ylabel("Cosine direction")

            valid = x.notna() & df["pred_gain_xy_mm_per_u"].notna()
            axes[3].plot(x[valid], df.loc[valid, "pred_gain_xy_mm_per_u"], marker="o", markersize=3, label=f"{label} pred")
            valid = x.notna() & df["actual_gain_xy_mm_per_u"].notna()
            axes[3].plot(x[valid], df.loc[valid, "actual_gain_xy_mm_per_u"], marker="x", markersize=3, linestyle="--", label=f"{label} actual")
            axes[3].set_ylabel("Gain [mm / ||u||]")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Reference index along vessel")
        fig.suptitle(f"LTI contact-vs-no-contact metrics, {int(bend)}° bend")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"timeline_dashboard_bend_{int(bend)}deg.png", dpi=300)
        plt.close(fig)


def plot_gain_error_scatter(runs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        df = run["df"]
        for _, r in df.iterrows():
            rows.append({
                "bend_abs_deg": run["bend_abs_deg"],
                "jacobian_variant": run["jacobian_variant"],
                "pred_gain_xy_mm_per_u": r.get("pred_gain_xy_mm_per_u", np.nan),
                "actual_gain_xy_mm_per_u": r.get("actual_gain_xy_mm_per_u", np.nan),
                "raw_one_step_err_xy_mm": r.get("raw_one_step_err_xy_mm", np.nan),
                "cos_pred_actual_xy": r.get("cos_pred_actual_xy", np.nan),
            })
    d = pd.DataFrame(rows)
    if d.empty:
        return

    for gain_col, title in [
        ("pred_gain_xy_mm_per_u", "Predicted gain vs one-step error"),
        ("actual_gain_xy_mm_per_u", "Actual gain vs one-step error"),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for variant, g in d.groupby("jacobian_variant"):
            valid = g[gain_col].notna() & g["raw_one_step_err_xy_mm"].notna()
            ax.scatter(g.loc[valid, gain_col], g.loc[valid, "raw_one_step_err_xy_mm"], label=variant.replace("_", " "), alpha=0.75)
        ax.set_xlabel("Gain [mm / ||u||]")
        ax.set_ylabel("One-step XY error [mm]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{gain_col}_vs_error.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for variant, g in d.groupby("jacobian_variant"):
        valid = g["cos_pred_actual_xy"].notna() & g["raw_one_step_err_xy_mm"].notna()
        ax.scatter(g.loc[valid, "cos_pred_actual_xy"], g.loc[valid, "raw_one_step_err_xy_mm"], label=variant.replace("_", " "), alpha=0.75)
    ax.set_xlabel("Cosine(predicted direction, actual direction)")
    ax.set_ylabel("One-step XY error [mm]")
    ax.set_title("Direction alignment vs one-step error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_direction_vs_error.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------
def print_key_tables(summary: pd.DataFrame, comparison: pd.DataFrame):
    key_cols = [
        "bend_abs_deg",
        "jacobian_variant",
        "success",
        "final_i_ref",
        "mean_one_step_xy_mm",
        "max_one_step_xy_mm",
        "mean_adapt_err_xy_mm",
        "max_adapt_err_xy_mm",
        "mean_cos_pred_actual_xy",
        "min_cos_pred_actual_xy",
        "mean_pred_gain_xy_mm_per_u",
        "mean_actual_gain_xy_mm_per_u",
        "median_gain_ratio_pred_over_actual_xy",
        "min_clearance_mm",
        "mean_cond_H_mpc",
    ]
    print("\n[LTI CONTACT VS NO-CONTACT RUN SUMMARY]")
    present = [c for c in key_cols if c in summary.columns]
    if len(summary):
        print(summary[present].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        print("No rows.")

    focus_metrics = [
        "mean_one_step_xy_mm",
        "max_one_step_xy_mm",
        "mean_adapt_err_xy_mm",
        "max_adapt_err_xy_mm",
        "mean_cos_pred_actual_xy",
        "min_cos_pred_actual_xy",
        "mean_pred_gain_xy_mm_per_u",
        "mean_actual_gain_xy_mm_per_u",
        "median_gain_ratio_pred_over_actual_xy",
        "min_clearance_mm",
        "mean_cond_H_mpc",
    ]
    if len(comparison):
        print("\n[PAIRWISE CONTACT EFFECT: positive improvement = better for errors/angles/conditioning; higher is better for cosine/clearance]")
        d = comparison[comparison["metric"].isin(focus_metrics)].copy()
        print(
            d[[
                "bend_abs_deg",
                "metric",
                "contact",
                "no_contact",
                "contact_minus_no_contact",
                "contact_improvement_pct",
            ]].to_string(index=False, float_format=lambda x: f"{x:.4g}")
        )
    else:
        print("\nNo complete contact/no_contact pairs found for pairwise comparison.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = read_runs(RUN_ROOT)

    if not runs:
        raise RuntimeError(f"No log.csv files found under {RUN_ROOT.resolve()}")

    variants = sorted({r["jacobian_variant"] for r in runs})
    if "contact" not in variants or "no_contact" not in variants:
        warnings.warn(f"Expected both contact and no_contact runs; found variants: {variants}")

    summary = build_summary(runs, OUT_DIR)
    comparison = build_pairwise_comparison(summary, OUT_DIR)

    # Also save enriched per-step logs so you can inspect cosine/gain/error row-by-row.
    enriched_dir = OUT_DIR / "enriched_logs"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        run["df"].to_csv(enriched_dir / f"{run['run_name']}__enriched.csv", index=False)

    plot_summary_bars(summary, OUT_DIR / "summary_plots")
    plot_timeline_per_bend(runs, OUT_DIR / "timeline_dashboards")
    plot_gain_error_scatter(runs, OUT_DIR / "scatter_plots")

    print_key_tables(summary, comparison)
    print(f"\nSaved analysis to: {OUT_DIR}")
    print("Key files:")
    print(f"  - {OUT_DIR / 'lti_contact_vs_no_contact_run_summary.csv'}")
    print(f"  - {OUT_DIR / 'lti_contact_vs_no_contact_pairwise_comparison.csv'}")
    print(f"  - {OUT_DIR / 'enriched_logs'}")
    print(f"  - {OUT_DIR / 'summary_plots'}")
    print(f"  - {OUT_DIR / 'timeline_dashboards'}")
    print(f"  - {OUT_DIR / 'scatter_plots'}")


if __name__ == "__main__":
    main()
