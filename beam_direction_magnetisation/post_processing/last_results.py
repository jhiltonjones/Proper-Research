#!/usr/bin/env python3
"""
Compare boundary-condition and no-boundary-condition MPC logs.

Expected CSV columns include:
    step
    meas_ref_err_xy
    one_step_pred_err_xy
    pred_meas_err_xy
    raw_model_err_xy
    nom_corr_resid_xy
    i_ref
    closest_distance_mm
    dist_left_edge_mm
    dist_right_edge_mm
    jac_pred_xy_mm
    jac_real_xy_mm
    jac_gain_real_over_pred
    jac_cos_xy
    jac_err_xy_mm
    jac_err_xyz_mm

Run:
    python compare_bc_vs_nobc_jacobian_log.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 18,          # Default text size
    "axes.labelsize": 22,     # X/Y axis label size
    "axes.titlesize": 24,     # Title size
    "xtick.labelsize": 18,    # X tick number size
    "ytick.labelsize": 18,    # Y tick number size
    "legend.fontsize": 16,    # Legend size
})
# ============================================================
# Edit these paths
# ============================================================

BC_CSV_PATHS = [
    Path("/Users/jackhilton-jones/Proper-Research/lti_1np_1s_bc_60_2.csv"),
    Path("/Users/jackhilton-jones/Proper-Research/lti_1np_1s_bc_60_3.csv"),
    # Path("/Users/jackhilton-jones/Proper-Research/lti_1np_1s_bc_60.csv"),
]

NOBC_CSV_PATHS = [
    Path("/Users/jackhilton-jones/Proper-Research/lti_1np_1s_nobc_60_2.csv"),
    Path("/Users/jackhilton-jones/Proper-Research/lti_1np_1s_nobc_60.csv"),
    # Path("/Users/jackhilton-jones/Proper-Research/lti_1np_3s_bc_60.csv"),
]

OUT_DIR = Path(
    "/Users/jackhilton-jones/Proper-Research/bc_vs_nobc_jacobian_comparison_1np"
)

K_MIN = 0
K_MAX = None


# ============================================================
# Loading
# ============================================================
def mean_std_by_x(df, x_col, y_col):
    """
    Average repeated runs at each x value.

    Returns columns:
        x_col, mean, std, n
    """
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=[x_col, "mean", "std", "n"])

    tmp = df[[x_col, y_col]].copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[x_col, y_col])

    if tmp.empty:
        return pd.DataFrame(columns=[x_col, "mean", "std", "n"])

    out = (
        tmp
        .groupby(x_col, as_index=False)[y_col]
        .agg(mean="mean", std="std", n="count")
        .sort_values(x_col)
    )

    return out
def load_log(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["case"] = label

    for c in df.columns:
        if c not in ("case", "jac_interpretation"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "step" not in df.columns:
        df["step"] = np.arange(len(df))

    return df

def load_group(paths, label):
    dfs = []

    for run_id, path in enumerate(paths):
        df = load_log(path, label)
        df["run_id"] = run_id
        df["run_name"] = Path(path).parent.name

        df = filter_by_step(df, K_MIN, K_MAX)
        df = add_derived_metrics(df)

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No runs supplied for {label}")

    return pd.concat(dfs, ignore_index=True)
def filter_by_step(df: pd.DataFrame, k_min=None, k_max=None) -> pd.DataFrame:
    out = df.copy()

    if k_min is not None:
        out = out[out["step"] >= k_min]

    if k_max is not None:
        out = out[out["step"] <= k_max]

    return out.reset_index(drop=True)


# ============================================================
# Derived metrics
# ============================================================

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Errors already appear to be in metres for these columns.
    metre_error_cols = [
        "meas_ref_err_xy",
        "raw_model_err_xy",
        "nom_corr_resid_xy",
        "pred0_current_resid_xy",
        "nom_ref_err_xy",
        "pred_ref_err_xy",
        "one_step_pred_err_xy",
        "pred_meas_err_xy",
        "err_to_center",
    ]

    for c in metre_error_cols:
        if c in df.columns:
            df[c + "_mm"] = 1e3 * df[c]

    # Useful absolute error from ideal Jacobian gain.
    if "jac_gain_real_over_pred" in df.columns:
        df["jac_gain_abs_err_from_1"] = np.abs(df["jac_gain_real_over_pred"] - 1.0)
    else:
        df["jac_gain_abs_err_from_1"] = np.nan

    # Direction angle from xy cosine.
    if "jac_cos_xy" in df.columns:
        cosv = np.clip(df["jac_cos_xy"].to_numpy(float), -1.0, 1.0)
        df["jac_angle_xy_deg"] = np.degrees(np.arccos(cosv))
    else:
        df["jac_angle_xy_deg"] = np.nan

    # A conservative wall distance column.
    # Your CSV already has closest_distance_mm, so use that first.
    if "closest_distance_mm" not in df.columns:
        if all(c in df.columns for c in ["dist_left_edge_mm", "dist_right_edge_mm"]):
            df["closest_distance_mm"] = np.minimum(
                df["dist_left_edge_mm"],
                df["dist_right_edge_mm"],
            )
        else:
            df["closest_distance_mm"] = np.nan

    return df


# ============================================================
# Summary
# ============================================================

def _nanmean(df, col):
    return float(np.nanmean(df[col])) if col in df.columns else np.nan


def _nanmedian(df, col):
    return float(np.nanmedian(df[col])) if col in df.columns else np.nan


def _nanmin(df, col):
    return float(np.nanmin(df[col])) if col in df.columns else np.nan


def _nanmax(df, col):
    return float(np.nanmax(df[col])) if col in df.columns else np.nan


def _nanp(df, col, p):
    return float(np.nanpercentile(df[col], p)) if col in df.columns else np.nan


def _final(df, col):
    if col not in df.columns:
        return np.nan
    x = df[col].dropna()
    if x.empty:
        return np.nan
    return float(x.iloc[-1])


def summarise_case(df: pd.DataFrame, label: str) -> dict:
    return {
        "case": label,
        "n_steps": int(len(df)),
        "final_step": _final(df, "step"),
        "final_i_ref": _final(df, "i_ref"),
        "final_meas_ref_idx": _final(df, "meas_ref_idx"),

        "mean_meas_ref_err_xy_mm": _nanmean(df, "meas_ref_err_xy_mm"),
        "median_meas_ref_err_xy_mm": _nanmedian(df, "meas_ref_err_xy_mm"),
        "p95_meas_ref_err_xy_mm": _nanp(df, "meas_ref_err_xy_mm", 95),
        "max_meas_ref_err_xy_mm": _nanmax(df, "meas_ref_err_xy_mm"),

        "mean_one_step_pred_err_xy_mm": _nanmean(df, "one_step_pred_err_xy_mm"),
        "median_one_step_pred_err_xy_mm": _nanmedian(df, "one_step_pred_err_xy_mm"),
        "p95_one_step_pred_err_xy_mm": _nanp(df, "one_step_pred_err_xy_mm", 95),
        "max_one_step_pred_err_xy_mm": _nanmax(df, "one_step_pred_err_xy_mm"),

        "mean_pred_meas_err_xy_mm": _nanmean(df, "pred_meas_err_xy_mm"),
        "median_pred_meas_err_xy_mm": _nanmedian(df, "pred_meas_err_xy_mm"),
        "max_pred_meas_err_xy_mm": _nanmax(df, "pred_meas_err_xy_mm"),
        "mean_raw_model_err_xy_mm": _nanmean(df, "raw_model_err_xy_mm"),
        "median_raw_model_err_xy_mm": _nanmedian(df, "raw_model_err_xy_mm"),
        "p95_raw_model_err_xy_mm": _nanp(df, "raw_model_err_xy_mm", 95),
        "max_raw_model_err_xy_mm": _nanmax(df, "raw_model_err_xy_mm"),

        "mean_nom_corr_resid_xy_mm": _nanmean(df, "nom_corr_resid_xy_mm"),

        "mean_err_to_center_mm": _nanmean(df, "err_to_center_mm"),
        "median_err_to_center_mm": _nanmedian(df, "err_to_center_mm"),
        "p95_err_to_center_mm": _nanp(df, "err_to_center_mm", 95),
        "max_err_to_center_mm": _nanmax(df, "err_to_center_mm"),

        "min_closest_distance_mm": _nanmin(df, "closest_distance_mm"),
        "mean_closest_distance_mm": _nanmean(df, "closest_distance_mm"),

        "mean_jac_pred_xy_mm": _nanmean(df, "jac_pred_xy_mm"),
        "mean_jac_real_xy_mm": _nanmean(df, "jac_real_xy_mm"),

        "mean_jac_gain_real_over_pred": _nanmean(df, "jac_gain_real_over_pred"),
        "median_jac_gain_real_over_pred": _nanmedian(df, "jac_gain_real_over_pred"),
        "p05_jac_gain_real_over_pred": _nanp(df, "jac_gain_real_over_pred", 5),
        "p95_jac_gain_real_over_pred": _nanp(df, "jac_gain_real_over_pred", 95),
        "mean_jac_gain_abs_err_from_1": _nanmean(df, "jac_gain_abs_err_from_1"),

        "mean_jac_cos_xy": _nanmean(df, "jac_cos_xy"),
        "mean_jac_angle_xy_deg": _nanmean(df, "jac_angle_xy_deg"),
        "median_jac_angle_xy_deg": _nanmedian(df, "jac_angle_xy_deg"),
        "p95_jac_angle_xy_deg": _nanp(df, "jac_angle_xy_deg", 95),
        "max_jac_angle_xy_deg": _nanmax(df, "jac_angle_xy_deg"),

        "mean_jac_err_xy_mm": _nanmean(df, "jac_err_xy_mm"),
        "median_jac_err_xy_mm": _nanmedian(df, "jac_err_xy_mm"),
        "max_jac_err_xy_mm": _nanmax(df, "jac_err_xy_mm"),

        "mean_jac_err_xyz_mm": _nanmean(df, "jac_err_xyz_mm"),
        "median_jac_err_xyz_mm": _nanmedian(df, "jac_err_xyz_mm"),
        "max_jac_err_xyz_mm": _nanmax(df, "jac_err_xyz_mm"),
    }


def make_summary(df_bc, df_nobc, out_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame([
        summarise_case(df_bc, "CC"),
        summarise_case(df_nobc, "No CC"),
    ])

    diff = {"case": "BC_minus_NoBC"}

    for c in summary.columns:
        if c == "case":
            continue

        try:
            diff[c] = float(summary.loc[0, c]) - float(summary.loc[1, c])
        except Exception:
            diff[c] = np.nan

    summary = pd.concat([summary, pd.DataFrame([diff])], ignore_index=True)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_series(
    df_bc,
    df_nobc,
    y_col,
    ylabel,
    title,
    out_path,
    ideal_line=None,
    show_std=True,
):
    plt.figure(figsize=(10, 5))

    for df, case_label in [(df_bc, "CC"), (df_nobc, "No CC")]:
        if y_col not in df.columns:
            print(f"[plot_series] Missing {y_col} for {case_label}")
            continue

        avg = mean_std_by_x(df, "step", y_col)

        if avg.empty:
            print(f"[plot_series] No valid data for {y_col} in {case_label}")
            continue

        x = avg["step"].to_numpy(float)
        y = avg["mean"].to_numpy(float)
        sd = avg["std"].to_numpy(float)
        n = avg["n"].to_numpy(float)

        plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"{case_label} mean",
        )

        if show_std:
            valid_std = np.isfinite(sd) & (n > 1)
            if np.any(valid_std):
                plt.fill_between(
                    x[valid_std],
                    y[valid_std] - sd[valid_std],
                    y[valid_std] + sd[valid_std],
                    alpha=0.18,
                    linewidth=0,
                )

    if ideal_line is not None:
        plt.axhline(
            ideal_line,
            linestyle="--",
            linewidth=1,
            label=f"ideal = {ideal_line:g}",
        )

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_series_vs_i_ref(
    df_bc,
    df_nobc,
    y_col,
    ylabel,
    title,
    out_path,
    ideal_line=None,
    show_std=True,
):
    plt.figure(figsize=(10, 5))

    plotted_any = False

    for df, case_label in [(df_bc, "CC"), (df_nobc, "No CC")]:
        if y_col not in df.columns or "i_ref" not in df.columns:
            continue

        tmp = df.copy()
        tmp = tmp[np.isfinite(tmp["i_ref"]) & (tmp["i_ref"] >= 0)]

        avg = mean_std_by_x(tmp, "i_ref", y_col)

        if avg.empty:
            continue

        x = avg["i_ref"].to_numpy(float)
        y = avg["mean"].to_numpy(float)
        sd = avg["std"].to_numpy(float)
        n = avg["n"].to_numpy(float)

        plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"{case_label} mean",
        )

        if show_std:
            valid_std = np.isfinite(sd) & (n > 1)
            if np.any(valid_std):
                plt.fill_between(
                    x[valid_std],
                    y[valid_std] - sd[valid_std],
                    y[valid_std] + sd[valid_std],
                    alpha=0.18,
                    linewidth=0,
                )

        plotted_any = True

    if not plotted_any:
        plt.close()
        print(f"[plot_series_vs_i_ref] Could not plot {y_col} vs i_ref.")
        return

    if ideal_line is not None:
        plt.axhline(
            ideal_line,
            linestyle="--",
            linewidth=1,
            label=f"ideal = {ideal_line:g}",
        )

    plt.xlabel("Reference index $i_{ref}$")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def plot_jac_pred_vs_real(df_bc, df_nobc, out_path):
    plt.figure(figsize=(6, 6))

    vals = []

    for df, label in [(df_bc, "CC"), (df_nobc, "No CC")]:
        if not all(c in df.columns for c in ["jac_pred_xy_mm", "jac_real_xy_mm"]):
            print(f"[plot_jac_pred_vs_real] Missing columns for {label}")
            continue

        x = df["jac_pred_xy_mm"].to_numpy(float)
        y = df["jac_real_xy_mm"].to_numpy(float)

        valid = np.isfinite(x) & np.isfinite(y)

        if np.any(valid):
            plt.scatter(x[valid], y[valid], s=20, alpha=0.75, label=label)
            vals.append(x[valid])
            vals.append(y[valid])

    if vals:
        vals = np.concatenate(vals)
        vals = vals[np.isfinite(vals)]

        if vals.size > 0:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            pad = 0.05 * max(hi - lo, 1e-9)
            plt.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                linestyle="--",
                linewidth=1,
                label="ideal",
            )

    plt.xlabel(r"Jacobian-predicted XY step $\|\Delta p_{Jac}\|$ [mm]")
    plt.ylabel(r"Measured XY step $\|\Delta p_{real}\|$ [mm]")
    plt.title("Jacobian predicted vs measured XY movement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_hist(df_bc, df_nobc, col, xlabel, title, out_path, bins=30):
    plt.figure(figsize=(8, 5))

    for df, label in [(df_bc, "CC"), (df_nobc, "No CC")]:
        if col not in df.columns:
            continue

        x = df[col].to_numpy(float)
        x = x[np.isfinite(x)]

        if x.size > 0:
            plt.hist(x, bins=bins, alpha=0.55, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_all_plots(df_bc, df_nobc, out_dir: Path):
    plot_series(
        df_bc,
        df_nobc,
        "meas_ref_err_xy_mm",
        "Measured reference error [mm]",
        "Measured tracking error",
        out_dir / "tracking_error_vs_step.png",
    )

    plot_series_vs_i_ref(
        df_bc,
        df_nobc,
        "meas_ref_err_xy_mm",
        "Measured reference error [mm]",
        "Measured tracking error vs reference index",
        out_dir / "tracking_error_vs_i_ref.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "one_step_pred_err_xy_mm",
        "One-step prediction error [mm]",
        "One-step prediction error",
        out_dir / "one_step_prediction_error_vs_step.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "pred_meas_err_xy_mm",
        "Predicted-measured error [mm]",
        "Predicted-measured XY error",
        out_dir / "pred_meas_error_vs_step.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "closest_distance_mm",
        "Closest wall distance [mm]",
        "Closest wall distance",
        out_dir / "closest_wall_distance_vs_step.png",
        ideal_line=0.0,
    )

    plot_series(
        df_bc,
        df_nobc,
        "jac_gain_real_over_pred",
        r"Jacobian gain $\|\Delta p_{real}\|/\|\Delta p_{Jac}\|$",
        "Jacobian gain",
        out_dir / "jacobian_gain_vs_step.png",
        ideal_line=1.0,
    )

    plot_series_vs_i_ref(
        df_bc,
        df_nobc,
        "jac_gain_real_over_pred",
        r"Jacobian gain $\|\Delta p_{real}\|/\|\Delta p_{Jac}\|$",
        "Jacobian gain vs reference index",
        out_dir / "jacobian_gain_vs_i_ref.png",
        ideal_line=1.0,
    )

    plot_series(
        df_bc,
        df_nobc,
        "jac_angle_xy_deg",
        "Jacobian direction error [deg]",
        "Jacobian XY direction error",
        out_dir / "jacobian_angle_vs_step.png",
        ideal_line=0.0,
    )

    plot_series(
        df_bc,
        df_nobc,
        "jac_err_xy_mm",
        "Jacobian XY vector error [mm]",
        "Jacobian XY vector error",
        out_dir / "jacobian_vector_error_xy_vs_step.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "jac_pred_xy_mm",
        "Predicted XY movement [mm]",
        "Jacobian-predicted XY movement",
        out_dir / "jacobian_predicted_xy_vs_step.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "jac_real_xy_mm",
        "Measured XY movement [mm]",
        "Measured XY movement",
        out_dir / "jacobian_real_xy_vs_step.png",
    )

    plot_jac_pred_vs_real(
        df_bc,
        df_nobc,
        out_dir / "jacobian_predicted_vs_real_xy_scatter.png",
    )

    plot_hist(
        df_bc,
        df_nobc,
        "jac_gain_real_over_pred",
        r"Jacobian gain $\|\Delta p_{real}\|/\|\Delta p_{Jac}\|$",
        "Distribution of Jacobian gain",
        out_dir / "jacobian_gain_histogram.png",
    )

    plot_hist(
        df_bc,
        df_nobc,
        "jac_angle_xy_deg",
        "Jacobian direction error [deg]",
        "Distribution of Jacobian XY direction error",
        out_dir / "jacobian_angle_histogram.png",
    )
    plot_series(
        df_bc,
        df_nobc,
        "raw_model_err_xy_mm",
        "Raw model error [mm]",
        "Raw model error",
        out_dir / "raw_model_error_vs_step.png",
    )

    plot_series_vs_i_ref(
        df_bc,
        df_nobc,
        "raw_model_err_xy_mm",
        "Raw model error [mm]",
        "Raw model error vs reference index",
        out_dir / "raw_model_error_vs_i_ref.png",
    )

    plot_series(
        df_bc,
        df_nobc,
        "err_to_center_mm",
        "Centreline error [mm]",
        "Error to vessel centreline",
        out_dir / "err_to_center_vs_step.png",
    )

    plot_series_vs_i_ref(
        df_bc,
        df_nobc,
        "err_to_center_mm",
        "Centreline error [mm]",
        "Error to vessel centreline vs reference index",
        out_dir / "err_to_center_vs_i_ref.png",
    )


# ============================================================
# Interpretation helper
# ============================================================

def print_key_interpretation(summary: pd.DataFrame):
    bc = summary[summary["case"] == "CC"].iloc[0]
    nb = summary[summary["case"] == "No CC"].iloc[0]

    print("\nKey comparison:")
    print("-" * 80)

    def show(metric, lower_is_better=True):
        a = bc.get(metric, np.nan)
        b = nb.get(metric, np.nan)

        if not np.isfinite(a) or not np.isfinite(b):
            return

        diff = a - b

        if lower_is_better:
            better = "CC" if a < b else "No CC"
        else:
            better = "CC" if a > b else "No CC"

        print(
            f"{metric}: "
            f"CC={a:.4g}, No CC={b:.4g}, "
            f"CC-NoBC={diff:.4g} -> {better} better"
        )

    show("mean_meas_ref_err_xy_mm", lower_is_better=True)
    show("median_meas_ref_err_xy_mm", lower_is_better=True)
    show("mean_one_step_pred_err_xy_mm", lower_is_better=True)
    show("mean_pred_meas_err_xy_mm", lower_is_better=True)
    show("mean_jac_gain_abs_err_from_1", lower_is_better=True)
    show("mean_jac_angle_xy_deg", lower_is_better=True)
    show("mean_jac_err_xy_mm", lower_is_better=True)
    show("min_closest_distance_mm", lower_is_better=False)

    print("-" * 80)

def make_per_run_summary(df_all, out_dir):
    summaries = []

    for (case, run_id, run_name), df_run in df_all.groupby(["case", "run_id", "run_name"]):
        s = summarise_case(df_run, case)
        s["run_id"] = run_id
        s["run_name"] = run_name
        summaries.append(s)

    per_run = pd.DataFrame(summaries)

    cols_front = ["case", "run_id", "run_name"]
    per_run = per_run[cols_front + [c for c in per_run.columns if c not in cols_front]]

    per_run.to_csv(out_dir / "per_run_summary_metrics.csv", index=False)

    group_summary = (
        per_run
        .drop(columns=["run_id", "run_name"], errors="ignore")
        .groupby("case")
        .agg(["mean", "std", "median", "min", "max"])
    )

    group_summary.to_csv(out_dir / "group_summary_metrics.csv")

    return per_run, group_summary
# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bc = load_group(BC_CSV_PATHS, "CC")
    df_nobc = load_group(NOBC_CSV_PATHS, "No CC")
    df_all = pd.concat([df_bc, df_nobc], ignore_index=True)

    df_bc.to_csv(OUT_DIR / "bc_all_runs_cleaned.csv", index=False)
    df_nobc.to_csv(OUT_DIR / "nobc_all_runs_cleaned.csv", index=False)
    df_all.to_csv(OUT_DIR / "all_runs_cleaned.csv", index=False)

    per_run_summary, group_summary = make_per_run_summary(df_all, OUT_DIR)

    summary = make_summary(df_bc, df_nobc, OUT_DIR)
    make_all_plots(df_bc, df_nobc, OUT_DIR)

    print("\nSaved outputs to:")
    print(OUT_DIR.resolve())

    print("\nPer-run summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(per_run_summary)

    print("\nGroup summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(group_summary)

    print("\nPooled timestep summary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary)

    print_key_interpretation(summary)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()