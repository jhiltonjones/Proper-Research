#!/usr/bin/env python3
"""
Compare forced LTI/SQP baselines against a hierarchical controller.

Designed for experiments such as:

1. Forced LTI, rollout 1, Np 1
2. Forced LTI, rollout 5, Np 5
3. Forced high-risk SQP, rollout 1, Np 1, N_sqp 3
4. Actual hierarchical controller, max rollout 5, Np 5

The point is to separate:
    - fewer MPC updates
    - prediction accuracy
    - SQP iteration count
    - horizon-aware computational burden

This script is deliberately robust to different run-root names. Edit RUN_GROUPS below.
"""

from pathlib import Path
import re
import csv
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User config: edit these paths after running the four experiments
# ---------------------------------------------------------------------
RUN_GROUPS = {
    # Example names; change paths to your actual folders.
    "LTI rollout 1": Path("/Users/jackhilton-jones/Proper-Research/evidence_forced_lti_rollout1_Np1"),
    "LTI rollout 5": Path("/Users/jackhilton-jones/Proper-Research/evidence_forced_lti_rollout5_Np5"),
    "SQP high-risk rollout 1": Path("/Users/jackhilton-jones/Proper-Research/evidence_forced_sqp_highrisk_rollout1_Np1_sqp3"),
    "Hierarchical": Path("/Users/jackhilton-jones/Proper-Research/evidence_hierarchical_lti_low_rollout5"),
}

OUT_DIR = Path("forced_lti_sqp_vs_hierarchy_analysis")

# Expected bends. The script will still work if a bend is missing.
EXPECTED_BENDS = [30, 45, 60, 75, 90]

# If true, focus the plots/tables on high-bend cases.
HIGH_BEND_ONLY = False
HIGH_BENDS = [60, 75, 90]


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def read_csv_robust(path: Path):
    """
    Reads normal or ragged CSVs.

    Some hierarchy logs can become ragged when longer rollout states are appended.
    We keep the header-width leading fields, which contain the stable control,
    error, risk, and cost columns.
    """
    path = Path(path)

    try:
        return pd.read_csv(path), {}
    except pd.errors.ParserError:
        pass

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        widths = []
        for row in reader:
            widths.append(len(row))
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            else:
                row = row[:len(header)]
            rows.append(row)

    df = pd.DataFrame(rows, columns=header)

    # Convert numeric-looking columns while preserving true string columns.
    # Do not use errors="ignore": recent pandas versions may reject it.
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        non_empty_original = df[c].astype(str).str.strip().ne("").sum()
        numeric_count = converted.notna().sum()

        # Convert if the column is empty or if nearly all non-empty entries are numeric.
        # This preserves columns such as status/run names while still handling ragged numeric logs.
        if non_empty_original == 0 or numeric_count >= max(1, int(0.8 * non_empty_original)):
            df[c] = converted

    return df, {"ragged_width_counts": dict(pd.Series(widths).value_counts().sort_index())}


def safe_num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def safe_str(df, col):
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index)


def pctl(series, q):
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanpercentile(s, q)) if np.isfinite(s).any() else np.nan


def parse_bend(path_or_name):
    text = str(path_or_name)
    m = re.search(r"bend_([mp])(\d+)", text)
    if not m:
        return np.nan
    return float(m.group(2))


def parse_rollout(path_or_name):
    text = str(path_or_name)
    m = re.search(r"rollout(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"roll[_-]?(\d+)", text)
    if m:
        return int(m.group(1))
    return np.nan


def parse_Np(path_or_name):
    text = str(path_or_name)
    m = re.search(r"Np(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"pred[_-]?(\d+)", text)
    if m:
        return int(m.group(1))
    return np.nan


def parse_sqp(path_or_name):
    text = str(path_or_name)
    # Avoid matching sqp_full as a number.
    m = re.search(r"sqp(\d+)", text)
    if m:
        return int(m.group(1))
    return np.nan


def infer_controller_from_run_name(run_name):
    if "lti" in run_name:
        return "lti"
    if "sqp_full" in run_name:
        return "sqp_full"
    return "unknown"


# ---------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------
def control_norm(df):
    cols = [f"u0_{x}" for x in ["vx", "vy", "vz", "wx", "wy", "wz", "dL"]]
    if all(c in df.columns for c in cols):
        U = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return pd.Series(np.linalg.norm(U, axis=1), index=df.index)

    # Some logs may use u_applied_0_*.
    cols = [c for c in df.columns if c.startswith("u_applied_0_") and not c.endswith("_norm")]
    if cols:
        U = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return pd.Series(np.linalg.norm(U, axis=1), index=df.index)

    for c in ["u_norm", "u0_norm", "control_norm"]:
        if c in df.columns:
            return safe_num(df, c)

    return pd.Series(np.nan, index=df.index)


def derive_direction_gain(df):
    required = ["x_rollout_0_x", "x_rollout_0_y", "x_pred_0_x", "x_pred_0_y"]
    if not all(c in df.columns for c in required):
        n = len(df)
        return (
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
            pd.Series(np.nan, index=df.index),
        )

    actual = df[["x_rollout_0_x", "x_rollout_0_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pred = df[["x_pred_0_x", "x_pred_0_y"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    prev = np.vstack([np.full((1, 2), np.nan), actual[:-1]])

    actual_delta = actual - prev
    pred_delta = pred - prev

    actual_norm = np.linalg.norm(actual_delta, axis=1)
    pred_norm = np.linalg.norm(pred_delta, axis=1)
    denom = actual_norm * pred_norm

    cos = np.full(len(df), np.nan)
    valid = np.isfinite(denom) & (denom > 1e-12)
    cos[valid] = np.clip(np.sum(actual_delta[valid] * pred_delta[valid], axis=1) / denom[valid], -1.0, 1.0)

    direction_error = np.degrees(np.arccos(cos))

    ratio = np.full(len(df), np.nan)
    valid_gain = np.isfinite(actual_norm) & (actual_norm > 1e-12) & np.isfinite(pred_norm)
    ratio[valid_gain] = pred_norm[valid_gain] / actual_norm[valid_gain]

    gain_log_error = np.full(len(df), np.nan)
    valid_ratio = np.isfinite(ratio) & (ratio > 1e-12)
    gain_log_error[valid_ratio] = np.abs(np.log10(ratio[valid_ratio]))

    return (
        pd.Series(cos, index=df.index),
        pd.Series(direction_error, index=df.index),
        pd.Series(gain_log_error, index=df.index),
    )


def rollout_error_columns(df, kind="xy"):
    pat = re.compile(rf"rollout_err_{kind}_(\d+)_mm$")
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    return sorted(cols)


def summarise_log(group_name, log_csv):
    log_csv = Path(log_csv)
    df, notes = read_csv_robust(log_csv)

    cos, direction_error, gain_log_error = derive_direction_gain(df)
    u_norm = control_norm(df)

    run_name = log_csv.parent.name
    bend = parse_bend(run_name)
    rollout_from_name = parse_rollout(run_name)
    Np_from_name = parse_Np(run_name)
    sqp_from_name = parse_sqp(run_name)

    # For hierarchy, the actual selected rollout / SQP budget should come from columns.
    rollout_used = safe_num(df, "rollout_steps_used")
    N_sqp_used = safe_num(df, "N_sqp_used")
    sqp_iters_done = safe_num(df, "sqp_iters_done")

    # Fallbacks for fixed controllers.
    if rollout_used.isna().all():
        fallback_roll = rollout_from_name if np.isfinite(rollout_from_name) else 1
        rollout_used = pd.Series(float(fallback_roll), index=df.index)

    if N_sqp_used.isna().all():
        fallback_sqp = sqp_from_name if np.isfinite(sqp_from_name) else np.nan
        N_sqp_used = pd.Series(float(fallback_sqp), index=df.index)

    if sqp_iters_done.isna().all():
        # For LTI this is not an SQP iteration count; it is only a solve-count proxy.
        if "lti" in group_name.lower() or "lti" in run_name.lower():
            sqp_iters_done = pd.Series(1.0, index=df.index)
        elif np.isfinite(sqp_from_name):
            sqp_iters_done = pd.Series(float(sqp_from_name), index=df.index)

    rollout_cols = rollout_error_columns(df, "xy")
    rollout_matrix = None
    if rollout_cols:
        rollout_matrix = df[[c for _, c in rollout_cols]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    i_ref = safe_num(df, "i_ref")
    refs = i_ref.dropna().astype(int).tolist()
    skipped = []
    if refs:
        skipped = sorted(set(range(min(refs), max(refs) + 1)) - set(refs))

    status = safe_str(df, "status")
    infeas = safe_num(df, "infeasible").fillna(0)
    success = bool(((infeas == 0) & status.isin(["solved", "solved_inaccurate", "solved inaccurate"])).all())

    one_step = safe_num(df, "pred1_err_xy_mm")
    if one_step.isna().all():
        one_step = safe_num(df, "rollout_err_xy_0_mm")

    adapt = safe_num(df, "adapt_pred_err_xy_mm")
    if adapt.isna().all():
        adapt = one_step

    clearance = safe_num(df, "adapt_clearance_mm")

    row = {
        "group": group_name,
        "run_name": run_name,
        "log_csv": str(log_csv),
        "bend_abs_deg": bend,
        "controller_mode_from_name": infer_controller_from_run_name(run_name),
        "rollout_from_name": rollout_from_name,
        "Np_from_name": Np_from_name,
        "N_sqp_from_name": sqp_from_name,

        "n_mpc_updates": int(len(df)),
        "success": success,
        "final_i_ref": float(i_ref.iloc[-1]) if len(df) and np.isfinite(i_ref.iloc[-1]) else np.nan,
        "max_i_ref": float(i_ref.max()) if np.isfinite(i_ref).any() else np.nan,
        "n_unique_i_refs": int(i_ref.nunique()),
        "mean_ref_gap": float(np.mean(np.diff(refs))) if len(refs) > 1 else np.nan,
        "max_ref_gap": float(np.max(np.diff(refs))) if len(refs) > 1 else np.nan,
        "n_skipped_refs_within_range": int(len(skipped)),
        "skipped_refs_within_range": ",".join(map(str, skipped)),

        "mean_one_step_xy_mm": float(one_step.mean()),
        "p95_one_step_xy_mm": pctl(one_step, 95),
        "max_one_step_xy_mm": float(one_step.max()) if np.isfinite(one_step).any() else np.nan,

        "mean_adapt_err_xy_mm": float(adapt.mean()),
        "p95_adapt_err_xy_mm": pctl(adapt, 95),
        "max_adapt_err_xy_mm": float(adapt.max()) if np.isfinite(adapt).any() else np.nan,

        "mean_rollout_all_xy_mm": float(np.nanmean(rollout_matrix)) if rollout_matrix is not None and np.isfinite(rollout_matrix).any() else np.nan,
        "p95_rollout_all_xy_mm": float(np.nanpercentile(rollout_matrix, 95)) if rollout_matrix is not None and np.isfinite(rollout_matrix).any() else np.nan,
        "max_rollout_all_xy_mm": float(np.nanmax(rollout_matrix)) if rollout_matrix is not None and np.isfinite(rollout_matrix).any() else np.nan,

        "mean_cos_one_step_xy": float(cos.mean()),
        "mean_direction_error_xy_deg": float(direction_error.mean()),
        "p95_direction_error_xy_deg": pctl(direction_error, 95),
        "mean_gain_log_error": float(gain_log_error.mean()),

        "mean_u_norm": float(u_norm.mean()),
        "p95_u_norm": pctl(u_norm, 95),

        "mean_clearance_mm": float(clearance.mean()),
        "p05_clearance_mm": pctl(clearance, 5),
        "min_clearance_mm": float(clearance.min()) if np.isfinite(clearance).any() else np.nan,

        "mean_rollout_steps_used": float(rollout_used.mean()),
        "sum_rollout_steps_used": float(rollout_used.sum()),
        "mean_N_sqp_used": float(N_sqp_used.mean()),
        "sum_N_sqp_used": float(N_sqp_used.sum()),
        "mean_sqp_iters_done": float(sqp_iters_done.mean()),
        "sum_sqp_iters_done": float(sqp_iters_done.sum()),

        # Horizon-aware cost proxies.
        "horizon_weighted_budget_sum_Nsqp_x_rollout": float((N_sqp_used * rollout_used).sum()),
        "iter_horizon_cost_sum_iters_x_rollout": float((sqp_iters_done * rollout_used).sum()),

        "rollout_steps_used_counts": str(rollout_used.value_counts(dropna=True).sort_index().to_dict()),
        "N_sqp_used_counts": str(N_sqp_used.value_counts(dropna=True).sort_index().to_dict()),
        "sqp_iters_done_counts": str(sqp_iters_done.value_counts(dropna=True).sort_index().to_dict()),
    }

    row.update({f"note_{k}": str(v) for k, v in notes.items()})

    # Save enriched log with derived metrics.
    enriched = df.copy()
    enriched["derived_u_norm"] = u_norm
    enriched["derived_cos_one_step_xy"] = cos
    enriched["derived_direction_error_xy_deg"] = direction_error
    enriched["derived_gain_log_error"] = gain_log_error
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", f"{group_name}_{run_name}")[:180]
    enriched.to_csv(OUT_DIR / "enriched_logs" / f"{safe_name}_enriched.csv", index=False)

    return row


def discover_logs_for_group(path):
    path = Path(path)
    if not path.exists():
        warnings.warn(f"Missing run group path: {path}")
        return []

    if path.is_file() and path.name.endswith(".csv"):
        return [path]

    return sorted(path.glob("**/log.csv"))


def build_summary():
    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "enriched_logs").mkdir(exist_ok=True)

    for group, root in RUN_GROUPS.items():
        logs = discover_logs_for_group(root)
        for log_csv in logs:
            try:
                row = summarise_log(group, log_csv)
                rows.append(row)
            except Exception as e:
                warnings.warn(f"Failed to summarise {log_csv}: {e}")

    if not rows:
        raise RuntimeError("No logs found. Edit RUN_GROUPS at the top of the script.")

    summary = pd.DataFrame(rows)

    if HIGH_BEND_ONLY:
        summary = summary[summary["bend_abs_deg"].isin(HIGH_BENDS)].copy()

    summary = summary.sort_values(["bend_abs_deg", "group"])
    return summary


def make_plots(summary):
    plots = [
        ("n_mpc_updates", "MPC updates", "mpc_updates_by_bend.png"),
        ("p95_one_step_xy_mm", "P95 one-step XY error [mm]", "p95_one_step_error_by_bend.png"),
        ("mean_adapt_err_xy_mm", "Mean adaptive error [mm]", "mean_adaptive_error_by_bend.png"),
        ("mean_direction_error_xy_deg", "Mean direction error [deg]", "direction_error_by_bend.png"),
        ("sum_sqp_iters_done", "Total SQP iterations / solve-count proxy", "total_sqp_iters_by_bend.png"),
        ("horizon_weighted_budget_sum_Nsqp_x_rollout", "Horizon-weighted budget Σ(Nsqp×rollout)", "horizon_weighted_budget_by_bend.png"),
        ("iter_horizon_cost_sum_iters_x_rollout", "Iteration-horizon cost Σ(iters×rollout)", "iter_horizon_cost_by_bend.png"),
        ("mean_clearance_mm", "Mean clearance [mm]", "mean_clearance_by_bend.png"),
        ("min_clearance_mm", "Minimum clearance [mm]", "min_clearance_by_bend.png"),
    ]

    for metric, ylabel, filename in plots:
        if metric not in summary.columns:
            continue
        pivot = summary.pivot_table(index="bend_abs_deg", columns="group", values=metric, aggfunc="mean").sort_index()

        ax = pivot.plot(kind="bar", figsize=(10, 5.4))
        ax.set_xlabel("Bend angle [deg]")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=240)
        plt.close(fig)

    # Cost/accuracy scatter.
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for group, g in summary.groupby("group"):
        ax.scatter(
            g["horizon_weighted_budget_sum_Nsqp_x_rollout"],
            g["p95_one_step_xy_mm"],
            label=group,
        )
        for _, r in g.iterrows():
            if np.isfinite(r["bend_abs_deg"]):
                ax.annotate(int(r["bend_abs_deg"]), (
                    r["horizon_weighted_budget_sum_Nsqp_x_rollout"],
                    r["p95_one_step_xy_mm"],
                ), fontsize=8)
    ax.set_xlabel("Horizon-weighted budget Σ(Nsqp×rollout)")
    ax.set_ylabel("P95 one-step XY error [mm]")
    ax.set_title("Cost–accuracy trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cost_accuracy_tradeoff.png", dpi=240)
    plt.close(fig)


def main():
    summary = build_summary()

    summary.to_csv(OUT_DIR / "forced_lti_sqp_vs_hierarchy_summary_by_bend.csv", index=False)

    aggregate = summary.groupby("group", as_index=False).agg(
        n_bends=("bend_abs_deg", "nunique"),
        success_fraction=("success", "mean"),
        mean_mpc_updates=("n_mpc_updates", "mean"),
        mean_unique_refs=("n_unique_i_refs", "mean"),
        mean_one_step_xy_mm=("mean_one_step_xy_mm", "mean"),
        p95_one_step_xy_mm_mean_across_bends=("p95_one_step_xy_mm", "mean"),
        mean_adapt_err_xy_mm=("mean_adapt_err_xy_mm", "mean"),
        p95_adapt_err_xy_mm_mean_across_bends=("p95_adapt_err_xy_mm", "mean"),
        mean_direction_error_xy_deg=("mean_direction_error_xy_deg", "mean"),
        mean_clearance_mm=("mean_clearance_mm", "mean"),
        min_clearance_mm_mean_across_bends=("min_clearance_mm", "mean"),
        mean_total_sqp_iters=("sum_sqp_iters_done", "mean"),
        mean_horizon_weighted_budget=("horizon_weighted_budget_sum_Nsqp_x_rollout", "mean"),
        mean_iter_horizon_cost=("iter_horizon_cost_sum_iters_x_rollout", "mean"),
    )
    aggregate.to_csv(OUT_DIR / "forced_lti_sqp_vs_hierarchy_aggregate.csv", index=False)

    make_plots(summary)

    with open(OUT_DIR / "README.txt", "w") as f:
        f.write("Forced LTI/SQP baselines vs hierarchical controller\n")
        f.write("===================================================\n\n")
        f.write("Edit RUN_GROUPS at the top of the script to point to your four experiment roots.\n\n")
        f.write("Recommended experiment groups:\n")
        f.write("1. LTI rollout 1: solver lti, rollout 1, Np 1.\n")
        f.write("2. LTI rollout 5: solver lti, rollout 5, Np 5.\n")
        f.write("3. SQP high-risk rollout 1: solver sqp_full, rollout 1, Np 1, N_sqp 3.\n")
        f.write("4. Hierarchical: actual hierarchy, max rollout 5, Np 5.\n\n")
        f.write("Main fairness caveat:\n")
        f.write("Counting raw SQP updates alone can be misleading. Also report horizon-weighted budget:\n")
        f.write("sum(N_sqp_used * rollout_steps_used), and iteration-horizon cost:\n")
        f.write("sum(sqp_iters_done * rollout_steps_used).\n")

    zip_path = OUT_DIR.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                z.write(p, arcname=p.relative_to(OUT_DIR))

    print(f"Analysed {len(summary)} runs.")
    print(f"Saved outputs to {OUT_DIR}")
    print(f"Created {zip_path}")
    print("\nAggregate:")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
