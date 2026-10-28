#!/usr/bin/env python3
"""
Compare MPC experiment logs with boundary contact (BC) and without boundary contact.

Edit the paths below, then run:

    python analyse_bc_vs_nonbc.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EDIT THESE PATHS
# ============================================================

BC_CSV_PATH = Path("/Users/jackhilton-jones/Proper-Research/mpc_1np1s_bc_60/log.csv")
NONBC_CSV_PATH = Path("/Users/jackhilton-jones/Proper-Research/mpc_3np1s_bc_60/log.csv")

OUT_DIR = Path("/Users/jackhilton-jones/Proper-Research/comparison_results_1np_3np_bc_60")


# ============================================================
# Loading and preprocessing
# ============================================================

def load_log(path: Path, label: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df["case"] = label

    for c in df.columns:
        if c not in ("status", "frame_path", "case"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tracking / one-step prediction error
    pred_col = first_existing(df, ["pred1_err_xy_mm", "pred1_err_mm", "pred1_err"])
    if pred_col is None:
        df["tracking_err_mm"] = np.nan
    elif pred_col == "pred1_err":
        df["tracking_err_mm"] = 1e3 * df[pred_col]
    else:
        df["tracking_err_mm"] = df[pred_col]

    # Clearance
    clr_col = first_existing(df, ["clearance_mm", "clearance"])
    if clr_col is None:
        df["clearance_plot_mm"] = np.nan
    elif clr_col == "clearance":
        df["clearance_plot_mm"] = 1e3 * df[clr_col]
    else:
        df["clearance_plot_mm"] = df[clr_col]

    # Vessel/tip angle
    angle_col = first_existing(
        df,
        ["vessel_tip_alignment_deg", "vessel_tip_alignment", "tip_vessel_angle_deg"]
    )
    df["vessel_tip_angle_deg"] = df[angle_col] if angle_col else np.nan

    # Command norms
    if all(c in df.columns for c in ["u0_vx", "u0_vy", "u0_vz"]):
        df["u_trans_norm"] = np.linalg.norm(
            df[["u0_vx", "u0_vy", "u0_vz"]].to_numpy(float),
            axis=1,
        )
    else:
        df["u_trans_norm"] = np.nan

    if all(c in df.columns for c in ["u0_wx", "u0_wy", "u0_wz"]):
        df["u_rot_norm"] = np.linalg.norm(
            df[["u0_wx", "u0_wy", "u0_wz"]].to_numpy(float),
            axis=1,
        )
    else:
        df["u_rot_norm"] = np.nan

    if "u0_dL" in df.columns:
        df["u_dL_abs"] = np.abs(df["u0_dL"])
    else:
        df["u_dL_abs"] = np.nan

    # Tip motion from logged tip positions
    if all(c in df.columns for c in ["y_x", "y_y", "y_z"]):
        xyz = df[["y_x", "y_y", "y_z"]].to_numpy(float)
        dxyz = np.vstack([np.full((1, 3), np.nan), np.diff(xyz, axis=0)])
        df["tip_step_mm"] = 1e3 * np.linalg.norm(dxyz, axis=1)
        df["tip_step_xy_mm"] = 1e3 * np.linalg.norm(dxyz[:, :2], axis=1)
    else:
        df["tip_step_mm"] = np.nan
        df["tip_step_xy_mm"] = np.nan

    # Jacobian diagnostics
    for col in [
        "jac_pred_mm",
        "jac_mpc_pred_mm",
        "jac_actual_mm",
        "jac_gain_actual_over_jac",
        "jac_gain_actual_over_mpc",
        "jac_angle_actual_deg",
        "jac_cos_actual",
        "jac_actual_along_pred_mm",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["jac_gain_error_abs"] = np.abs(df["jac_gain_actual_over_jac"] - 1.0)
    df["jac_angle_abs_deg"] = np.abs(df["jac_angle_actual_deg"])

    if "k" not in df.columns:
        df["k"] = np.arange(len(df))

    return df


# ============================================================
# Summary
# ============================================================

def summarise_case(df: pd.DataFrame, label: str) -> dict:
    def mean(name):
        return float(np.nanmean(df[name])) if name in df else np.nan

    def median(name):
        return float(np.nanmedian(df[name])) if name in df else np.nan

    def maxv(name):
        return float(np.nanmax(df[name])) if name in df else np.nan

    def minv(name):
        return float(np.nanmin(df[name])) if name in df else np.nan

    def final(name):
        if name not in df or df[name].dropna().empty:
            return np.nan
        return float(df[name].dropna().iloc[-1])

    return {
        "case": label,
        "n_steps": int(len(df)),

        "final_i_ref": final("i_ref"),
        "final_i_ref_mpc": final("i_ref_mpc"),

        "mean_tracking_err_mm": mean("tracking_err_mm"),
        "median_tracking_err_mm": median("tracking_err_mm"),
        "max_tracking_err_mm": maxv("tracking_err_mm"),

        "mean_tip_step_xy_mm": mean("tip_step_xy_mm"),
        "max_tip_step_xy_mm": maxv("tip_step_xy_mm"),

        "min_clearance_mm": minv("clearance_plot_mm"),
        "mean_clearance_mm": mean("clearance_plot_mm"),

        "mean_vessel_tip_angle_deg": mean("vessel_tip_angle_deg"),
        "max_vessel_tip_angle_deg": maxv("vessel_tip_angle_deg"),

        "mean_jac_pred_mm": mean("jac_pred_mm"),
        "mean_jac_actual_mm": mean("jac_actual_mm"),
        "mean_jac_gain_actual_over_jac": mean("jac_gain_actual_over_jac"),
        "median_jac_gain_actual_over_jac": median("jac_gain_actual_over_jac"),
        "mean_jac_gain_abs_error": mean("jac_gain_error_abs"),

        "mean_jac_angle_actual_deg": mean("jac_angle_abs_deg"),
        "median_jac_angle_actual_deg": median("jac_angle_abs_deg"),
        "max_jac_angle_actual_deg": maxv("jac_angle_abs_deg"),

        "mean_u_trans_norm": mean("u_trans_norm"),
        "mean_u_rot_norm": mean("u_rot_norm"),
        "mean_u_dL_abs": mean("u_dL_abs"),
    }


def save_summary(df_bc: pd.DataFrame, df_nonbc: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame([
        summarise_case(df_bc, "BC"),
        summarise_case(df_nonbc, "nonBC"),
    ])

    diff = {"case": "BC_minus_nonBC"}

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

def plot_series(df_bc, df_nonbc, y_col, ylabel, title, out_path):
    plt.figure(figsize=(9, 5))

    if y_col in df_bc.columns:
        plt.plot(df_bc["k"], df_bc[y_col], label="BC")

    if y_col in df_nonbc.columns:
        plt.plot(df_nonbc["k"], df_nonbc[y_col], label="non-BC")

    plt.xlabel("Step k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_pred_vs_actual(df_bc, df_nonbc, out_path):
    plt.figure(figsize=(6, 6))

    for df, label in [(df_bc, "BC"), (df_nonbc, "non-BC")]:
        if "jac_pred_mm" not in df.columns or "jac_actual_mm" not in df.columns:
            continue

        x = df["jac_pred_mm"].to_numpy(float)
        y = df["jac_actual_mm"].to_numpy(float)

        mask = np.isfinite(x) & np.isfinite(y)

        if np.any(mask):
            plt.scatter(x[mask], y[mask], s=18, alpha=0.7, label=label)

    vals = []

    for df in [df_bc, df_nonbc]:
        if "jac_pred_mm" in df.columns:
            vals.append(df["jac_pred_mm"].to_numpy(float))
        if "jac_actual_mm" in df.columns:
            vals.append(df["jac_actual_mm"].to_numpy(float))

    if vals:
        vals = np.concatenate(vals)
        vals = vals[np.isfinite(vals)]

        if vals.size > 0:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            pad = 0.05 * max(hi - lo, 1e-9)
            plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1)

    plt.xlabel("Jacobian-predicted movement |B₀u₀| [mm]")
    plt.ylabel("Actual movement |Δp| [mm]")
    plt.title("Jacobian predicted vs actual movement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_hist(df_bc, df_nonbc, col, xlabel, title, out_path, bins=30):
    plt.figure(figsize=(8, 5))

    for df, label in [(df_bc, "BC"), (df_nonbc, "non-BC")]:
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


def make_all_plots(df_bc, df_nonbc, out_dir: Path):
    plot_series(
        df_bc,
        df_nonbc,
        "tracking_err_mm",
        "Tracking / one-step prediction error [mm]",
        "Tracking error comparison",
        out_dir / "tracking_error_comparison.png",
    )

    plot_series(
        df_bc,
        df_nonbc,
        "jac_gain_actual_over_jac",
        "Actual / Jacobian-predicted gain",
        "Jacobian gain comparison",
        out_dir / "jacobian_gain_comparison.png",
    )

    plot_series(
        df_bc,
        df_nonbc,
        "jac_angle_actual_deg",
        "Angle between predicted and actual movement [deg]",
        "Jacobian direction error comparison",
        out_dir / "jacobian_angle_comparison.png",
    )

    plot_series(
        df_bc,
        df_nonbc,
        "clearance_plot_mm",
        "Tip clearance [mm]",
        "Clearance comparison",
        out_dir / "clearance_comparison.png",
    )

    plot_series(
        df_bc,
        df_nonbc,
        "u_trans_norm",
        "‖translational command‖",
        "Translational command norm comparison",
        out_dir / "command_norm_comparison.png",
    )

    plot_series(
        df_bc,
        df_nonbc,
        "i_ref",
        "Reference index",
        "Reference progress comparison",
        out_dir / "progress_comparison.png",
    )

    plot_pred_vs_actual(
        df_bc,
        df_nonbc,
        out_dir / "jacobian_pred_vs_actual.png",
    )

    plot_hist(
        df_bc,
        df_nonbc,
        "jac_angle_actual_deg",
        "Jacobian direction error [deg]",
        "Distribution of Jacobian direction error",
        out_dir / "jacobian_angle_histogram.png",
    )

    plot_hist(
        df_bc,
        df_nonbc,
        "jac_gain_actual_over_jac",
        "Actual / predicted gain",
        "Distribution of Jacobian gain",
        out_dir / "jacobian_gain_histogram.png",
    )


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bc = load_log(BC_CSV_PATH, "BC")
    df_nonbc = load_log(NONBC_CSV_PATH, "nonBC")

    df_bc = compute_derived_metrics(df_bc)
    df_nonbc = compute_derived_metrics(df_nonbc)

    df_bc.to_csv(OUT_DIR / "bc_cleaned.csv", index=False)
    df_nonbc.to_csv(OUT_DIR / "nonbc_cleaned.csv", index=False)

    summary = save_summary(df_bc, df_nonbc, OUT_DIR)
    make_all_plots(df_bc, df_nonbc, OUT_DIR)

    print("\nSaved results to:")
    print(OUT_DIR.resolve())

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()