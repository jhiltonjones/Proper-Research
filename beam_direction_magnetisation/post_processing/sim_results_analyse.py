#!/usr/bin/env python3
"""
Analyse 30-degree curved-vessel MPC results.

Edit the CSV paths in RUNS below, then run:

    python analyse_30deg_results.py

Outputs:
    - summary_30deg.csv
    - summary_30deg_latex_table.tex
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# EDIT THESE PATHS
# ============================================================

OUT_DIR = Path("analysis_120deg_results")

RUNS = [
    # {
    #     "vessel": "30$^\\circ$",
    #     "magnet_placement": "Above-patient",
    #     "case": "BC",
    #     "csv": Path("/Users/jackhilton-jones/Proper-Research/mpc_1np1s_above_30/log.csv"),
    # },
    {
        "vessel": "120$^\\circ$",
        "magnet_placement": "In-plane",
        "case": "BC",
        "csv": Path("/Users/jackhilton-jones/Proper-Research/mpc_3np1s_bc_120/log.csv"),
    },
    {
        "vessel": "120$^\\circ$",
        "magnet_placement": "In-plane",
        "case": "nonBC",
        "csv": Path("/Users/jackhilton-jones/Proper-Research/mpc_3np1s_nobc_120/log.csv"),
    },
    # {
    #     "vessel": "30$^\\circ$",
    #     "magnet_placement": "In-plane",
    #     "case": "Weighted advancement",
    #     "csv": Path("/Users/jackhilton-jones/Proper-Research/mpc_1np1s_wadv_30/log.csv"),
    # },
]


# ============================================================
# Loading utilities
# ============================================================

def load_log(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    for c in df.columns:
        if c not in ("status", "frame_path"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    x = df[col].to_numpy(float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan


def safe_median(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    x = df[col].to_numpy(float)
    return float(np.nanmedian(x)) if np.any(np.isfinite(x)) else np.nan


def safe_min(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    x = df[col].to_numpy(float)
    return float(np.nanmin(x)) if np.any(np.isfinite(x)) else np.nan


def safe_max(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    x = df[col].to_numpy(float)
    return float(np.nanmax(x)) if np.any(np.isfinite(x)) else np.nan


def safe_final(df: pd.DataFrame, col: str) -> float:
    if col not in df:
        return np.nan
    x = df[col].dropna()
    return float(x.iloc[-1]) if len(x) else np.nan


# ============================================================
# Derived metrics
# ============================================================

def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tracking error
    xy_col = first_existing(df, ["pred1_err_xy_mm", "pred1_err_mm", "pred1_err"])
    xyz_col = first_existing(df, ["pred1_err_xyz_mm", "pred1_err_xyz_m"])

    if xy_col is None:
        df["tracking_err_xy_mm"] = np.nan
    elif xy_col == "pred1_err":
        df["tracking_err_xy_mm"] = 1e3 * df[xy_col]
    else:
        df["tracking_err_xy_mm"] = df[xy_col]

    if xyz_col is None:
        df["tracking_err_xyz_mm"] = np.nan
    elif xyz_col == "pred1_err_xyz_m":
        df["tracking_err_xyz_mm"] = 1e3 * df[xyz_col]
    else:
        df["tracking_err_xyz_mm"] = df[xyz_col]

    # Clearance
    if "clearance_mm" in df.columns:
        df["clearance_plot_mm"] = df["clearance_mm"]
    elif "clearance_m" in df.columns:
        df["clearance_plot_mm"] = 1e3 * df["clearance_m"]
    else:
        df["clearance_plot_mm"] = np.nan

    # Vessel-tip angle
    angle_col = first_existing(
        df,
        ["vessel_tip_alignment_deg", "tip_vessel_angle_deg", "vessel_tip_alignment"],
    )
    df["vessel_tip_angle_deg"] = df[angle_col] if angle_col else np.nan

    # Tip motion from logged tip positions
    if all(c in df.columns for c in ["y_x", "y_y", "y_z"]):
        xyz = df[["y_x", "y_y", "y_z"]].to_numpy(float)
        dxyz = np.vstack([np.full((1, 3), np.nan), np.diff(xyz, axis=0)])
        df["tip_step_xyz_mm"] = 1e3 * np.linalg.norm(dxyz, axis=1)
        df["tip_step_xy_mm_from_pos"] = 1e3 * np.linalg.norm(dxyz[:, :2], axis=1)
    else:
        df["tip_step_xyz_mm"] = np.nan
        df["tip_step_xy_mm_from_pos"] = np.nan

    # Command magnitudes
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

    # Jacobian diagnostics
    required_jac_cols = [
        "jac_pred_mm",
        "jac_mpc_pred_mm",
        "jac_actual_mm",
        "jac_gain_actual_over_jac",
        "jac_gain_actual_over_mpc",
        "jac_angle_actual_deg",
        "jac_cos_actual",
        "jac_actual_along_pred_mm",
    ]

    for c in required_jac_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["jac_gain_abs_error"] = np.abs(df["jac_gain_actual_over_jac"] - 1.0)
    df["jac_angle_abs_deg"] = np.abs(df["jac_angle_actual_deg"])

    # Solver success
    if "status" in df.columns:
        df["solved_flag"] = df["status"].astype(str).str.lower().eq("solved").astype(float)
    else:
        df["solved_flag"] = np.nan

    if "infeasible" not in df.columns:
        df["infeasible"] = np.nan

    if "ref_advanced" not in df.columns:
        df["ref_advanced"] = np.nan

    if "k" not in df.columns:
        df["k"] = np.arange(len(df))

    return df


# ============================================================
# Summary
# ============================================================

def summarise_run(df: pd.DataFrame, meta: dict) -> dict:
    return {
        "vessel": meta["vessel"],
        "magnet_placement": meta["magnet_placement"],
        "case": meta["case"],

        "n_steps": int(len(df)),
        "final_i_ref": safe_final(df, "i_ref"),
        "final_i_ref_mpc": safe_final(df, "i_ref_mpc"),
        "total_ref_advancements": safe_mean(df, "ref_advanced") * len(df)
        if "ref_advanced" in df else np.nan,

        "mean_tracking_err_xy_mm": safe_mean(df, "tracking_err_xy_mm"),
        "median_tracking_err_xy_mm": safe_median(df, "tracking_err_xy_mm"),
        "max_tracking_err_xy_mm": safe_max(df, "tracking_err_xy_mm"),

        "mean_tracking_err_xyz_mm": safe_mean(df, "tracking_err_xyz_mm"),
        "median_tracking_err_xyz_mm": safe_median(df, "tracking_err_xyz_mm"),
        "max_tracking_err_xyz_mm": safe_max(df, "tracking_err_xyz_mm"),

        "mean_tip_step_xy_mm": safe_mean(df, "tip_step_xy_mm_from_pos"),
        "max_tip_step_xy_mm": safe_max(df, "tip_step_xy_mm_from_pos"),

        "min_clearance_mm": safe_min(df, "clearance_plot_mm"),
        "mean_clearance_mm": safe_mean(df, "clearance_plot_mm"),

        "mean_vessel_tip_angle_deg": safe_mean(df, "vessel_tip_angle_deg"),
        "max_vessel_tip_angle_deg": safe_max(df, "vessel_tip_angle_deg"),

        "mean_jac_pred_mm": safe_mean(df, "jac_pred_mm"),
        "mean_jac_mpc_pred_mm": safe_mean(df, "jac_mpc_pred_mm"),
        "mean_jac_actual_mm": safe_mean(df, "jac_actual_mm"),

        "mean_jac_gain_actual_over_jac": safe_mean(df, "jac_gain_actual_over_jac"),
        "median_jac_gain_actual_over_jac": safe_median(df, "jac_gain_actual_over_jac"),
        "mean_jac_gain_abs_error": safe_mean(df, "jac_gain_abs_error"),

        "mean_jac_angle_actual_deg": safe_mean(df, "jac_angle_abs_deg"),
        "median_jac_angle_actual_deg": safe_median(df, "jac_angle_abs_deg"),
        "max_jac_angle_actual_deg": safe_max(df, "jac_angle_abs_deg"),

        "mean_u_trans_norm": safe_mean(df, "u_trans_norm"),
        "mean_u_rot_norm": safe_mean(df, "u_rot_norm"),
        "mean_u_dL_abs": safe_mean(df, "u_dL_abs"),

        "solver_success_rate": safe_mean(df, "solved_flag"),
        "n_infeasible": np.nansum(df["infeasible"].to_numpy(float))
        if "infeasible" in df else np.nan,
    }


# ============================================================
# LaTeX output
# ============================================================

def fmt(x, ndp=3):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndp}f}"


def make_latex_table(summary: pd.DataFrame) -> str:
    lines = []

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Summary of 90$^\circ$ curved-vessel MPC experiments for different external-magnet placements and control formulations.}"
    )
    lines.append(r"\label{tab:results_30deg}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(
        r"\begin{tabular}{llrrrrrrr}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Magnet placement} & "
        r"\textbf{Case} & "
        r"\textbf{Steps} & "
        r"\textbf{Final $i_{\mathrm{ref}}$} & "
        r"\textbf{Med. err. [mm]} & "
        r"\textbf{Min clear. [mm]} & "
        r"\textbf{Mean clear. [mm]} & "
        r"\textbf{Mean Jac. gain} & "
        r"\textbf{Mean Jac. angle [deg]} & "
    )
    lines.append(r"\midrule")

    for _, r in summary.iterrows():
        line = (
            f"{r['magnet_placement']} & "
            f"{r['case']} & "
            f"{int(r['n_steps'])} & "
            f"{fmt(r['final_i_ref'], 0)} & "
            f"{fmt(r['median_tracking_err_xy_mm'], 4)} & "
            f"{fmt(r['min_clearance_mm'], 3)} & "
            f"{fmt(r['mean_clearance_mm'], 3)} & "
            f"{fmt(r['mean_jac_gain_actual_over_jac'], 3)} & "
            f"{fmt(r['mean_jac_angle_actual_deg'], 2)} & "
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    cleaned_frames = []

    for meta in RUNS:
        path = Path(meta["csv"])

        if not path.exists():
            print(f"[WARNING] Skipping missing file: {path}")
            continue

        print(f"[LOAD] {meta['magnet_placement']} | {meta['case']} | {path}")

        df = load_log(path)
        df = compute_derived_metrics(df)

        df["vessel"] = meta["vessel"]
        df["magnet_placement"] = meta["magnet_placement"]
        df["case"] = meta["case"]

        rows.append(summarise_run(df, meta))
        cleaned_frames.append(df)

    if not rows:
        raise RuntimeError("No valid CSV files were loaded. Check the paths in RUNS.")

    summary = pd.DataFrame(rows)

    # Sort table order
    placement_order = {
        "Above-patient": 0,
        "In-plane": 1,
    }
    case_order = {
        "BC": 0,
        "nonBC": 1,
        "Weighted advancement": 2,
    }

    summary["placement_order"] = summary["magnet_placement"].map(placement_order).fillna(99)
    summary["case_order"] = summary["case"].map(case_order).fillna(99)
    summary = summary.sort_values(["placement_order", "case_order"]).drop(
        columns=["placement_order", "case_order"]
    )

    summary_path = OUT_DIR / "summary_120deg.csv"
    latex_path = OUT_DIR / "summary_120deg_latex_table.tex"

    summary.to_csv(summary_path, index=False)

    latex = make_latex_table(summary)
    latex_path.write_text(latex)

    if cleaned_frames:
        all_cleaned = pd.concat(cleaned_frames, ignore_index=True)
        all_cleaned.to_csv(OUT_DIR / "all_120deg_cleaned.csv", index=False)

    print("\nSaved:")
    print(f"  {summary_path.resolve()}")
    print(f"  {latex_path.resolve()}")
    print(f"  {(OUT_DIR / 'all_120deg_cleaned.csv').resolve()}")

    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary)


if __name__ == "__main__":
    main()