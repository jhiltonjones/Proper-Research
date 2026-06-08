#!/usr/bin/env python3
"""
Compare MPC rollout/controller CSV logs.

Edit RUNS below, then run:

    python plot_mpc_comparison.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER INPUTS: EDIT THESE
# ============================================================

RUNS = {
    "LTI contact": r"/Users/jackhilton-jones/Proper-Research/rollout_10_90_lti/log.csv",
    "LTV full SQP": r"/Users/jackhilton-jones/Proper-Research/rollout_10_90_ltv_sqp/log.csv",
    "LTV one-shot SQP": r"/Users/jackhilton-jones/Proper-Research/rollout_10_90_ltv_onshot_sqp/log.csv",
    "LTI no contact": r"/Users/jackhilton-jones/Proper-Research/rollout_10_90_lti_nocc/log.csv",
}

OUTDIR = r"mpc_comparison_plots_rollout90_deg"

# Set True if errors span many orders of magnitude.
LOGSCALE_ERRORS = False


# ============================================================
# Loading / utilities
# ============================================================

def read_mpc_csv_robust(path: Path) -> pd.DataFrame:
    """
    Robust reader for MPC logs where the CSV header may have fewer columns
    than later rows because rollout_steps/Np changed or the file was appended
    after logging changes.

    Keeps all columns by expanding the header if needed.
    """
    import csv

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) == 0:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    data_rows = rows[1:]

    max_cols = max(len(header), *(len(r) for r in data_rows)) if data_rows else len(header)

    if len(header) < max_cols:
        header = header + [f"extra_col_{i}" for i in range(len(header), max_cols)]

    fixed_rows = []

    for r in data_rows:
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]

        fixed_rows.append(r)

    df = pd.DataFrame(fixed_rows, columns=header)

    # Convert numeric-looking columns.
    for c in df.columns:
        if c == "frame_path" or c.startswith("extra_col_"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_runs(runs_config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    runs = {}

    for label, path_str in runs_config.items():
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(f"CSV not found for '{label}': {path}")

        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError:
            print(f"[WARN] Standard pandas read failed for {label}. Using robust CSV reader.")
            df = read_mpc_csv_robust(path)

        if "k" not in df.columns:
            df["k"] = np.arange(len(df))

        runs[label] = df
        print(f"loaded {label}: {len(df)} rows, {len(df.columns)} columns")

    return runs

def get_col(df: pd.DataFrame, name: str, default=np.nan) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy()
    return np.full(len(df), default, dtype=float)


def mm(df: pd.DataFrame, name_m: str, name_mm: str | None = None) -> np.ndarray:
    if name_mm is not None and name_mm in df.columns:
        return get_col(df, name_mm)
    return 1e3 * get_col(df, name_m)


def finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def finite_max(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan


def finite_min(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else np.nan


def find_rollout_indices(df: pd.DataFrame, prefix: str = "rollout_err_xy") -> List[int]:
    idxs = []

    start = f"{prefix}_"
    end = "_mm"

    for c in df.columns:
        if c.startswith(start) and c.endswith(end):
            try:
                idxs.append(int(c[len(start):-len(end)]))
            except Exception:
                pass

    return sorted(set(idxs))


def rollout_matrix_mm(
    df: pd.DataFrame,
    prefix: str = "rollout_err_xy",
) -> Tuple[np.ndarray, List[int]]:
    idxs = find_rollout_indices(df, prefix=prefix)

    if not idxs:
        return np.empty((len(df), 0)), []

    cols = []

    for j in idxs:
        col_mm = f"{prefix}_{j}_mm"
        col_m = f"{prefix}_{j}_m"

        if col_mm in df.columns:
            vals = get_col(df, col_mm)
        elif col_m in df.columns:
            vals = 1e3 * get_col(df, col_m)
        else:
            vals = np.full(len(df), np.nan)

        cols.append(vals)

    return np.vstack(cols).T, idxs


def control_norms(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    vx = get_col(df, "u0_vx")
    vy = get_col(df, "u0_vy")
    vz = get_col(df, "u0_vz")
    wx = get_col(df, "u0_wx")
    wy = get_col(df, "u0_wy")
    wz = get_col(df, "u0_wz")
    dL = get_col(df, "u0_dL")

    return {
        "trans_norm": np.sqrt(vx**2 + vy**2 + vz**2),
        "omega_norm": np.sqrt(wx**2 + wy**2 + wz**2),
        "dL": dL,
    }


# ============================================================
# Plot helpers
# ============================================================

def setup_outdir(outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def savefig(outdir: Path, name: str):
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved: {path}")


def plot_time_series(
    runs: Dict[str, pd.DataFrame],
    y_getter,
    ylabel: str,
    title: str,
    outdir: Path,
    filename: str,
    yscale: str = "linear",
):
    plt.figure(figsize=(10, 5))

    for label, df in runs.items():
        k = get_col(df, "k")
        y = y_getter(df)
        plt.plot(k, y, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.xlabel("MPC outer step k")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale(yscale)
    plt.legend()
    savefig(outdir, filename)


def plot_rollout_stage_means(
    runs: Dict[str, pd.DataFrame],
    outdir: Path,
    prefix: str = "rollout_err_xy",
    filename: str = "rollout_stage_mean_xy_error.png",
):
    plt.figure(figsize=(10, 5))

    for label, df in runs.items():
        E, idxs = rollout_matrix_mm(df, prefix=prefix)
        if E.shape[1] == 0:
            continue

        mean_by_stage = np.nanmean(E, axis=0)
        plt.plot(idxs, mean_by_stage, marker="o", linewidth=1.8, label=label)

    plt.xlabel("Applied rollout substep")
    plt.ylabel("Mean rollout XY prediction error [mm]")
    plt.title("Mean rollout prediction error by applied substep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, filename)


def plot_rollout_stage_max(
    runs: Dict[str, pd.DataFrame],
    outdir: Path,
    prefix: str = "rollout_err_xy",
    filename: str = "rollout_stage_max_xy_error.png",
):
    plt.figure(figsize=(10, 5))

    for label, df in runs.items():
        E, idxs = rollout_matrix_mm(df, prefix=prefix)
        if E.shape[1] == 0:
            continue

        max_by_stage = np.nanmax(E, axis=0)
        plt.plot(idxs, max_by_stage, marker="o", linewidth=1.8, label=label)

    plt.xlabel("Applied rollout substep")
    plt.ylabel("Max rollout XY prediction error [mm]")
    plt.title("Max rollout prediction error by applied substep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, filename)


def plot_summary_bars(runs: Dict[str, pd.DataFrame], outdir: Path):
    labels = []
    pred1_mean = []
    rollout_mean = []
    rollout_final = []
    jac_angle_mean = []
    gain_mean = []
    dist_mean = []
    clearance_min = []

    for label, df in runs.items():
        labels.append(label)

        pred1 = mm(df, "pred1_err_xy_m", "pred1_err_xy_mm")
        pred1_mean.append(finite_mean(pred1))

        E, _ = rollout_matrix_mm(df)
        if E.shape[1] > 0:
            rollout_mean.append(finite_mean(E.ravel()))
            rollout_final.append(finite_mean(E[:, -1]))
        else:
            rollout_mean.append(np.nan)
            rollout_final.append(np.nan)

        jac_angle_mean.append(finite_mean(get_col(df, "jac_angle_actual_deg")))
        gain_mean.append(finite_mean(get_col(df, "jac_gain_actual_over_jac")))
        dist_mean.append(finite_mean(mm(df, "dist_to_ref_m", "dist_to_ref_mm")))
        clearance_min.append(finite_min(mm(df, "clearance_m", "clearance_mm")))

    metrics = [
        ("Mean one-step XY error [mm]", pred1_mean, "summary_mean_pred1_xy.png"),
        ("Mean rollout XY error [mm]", rollout_mean, "summary_mean_rollout_xy.png"),
        ("Mean final-stage rollout XY error [mm]", rollout_final, "summary_final_rollout_xy.png"),
        ("Mean Jacobian angle error [deg]", jac_angle_mean, "summary_jac_angle.png"),
        ("Mean gain actual / predicted", gain_mean, "summary_jac_gain.png"),
        ("Mean distance to reference [mm]", dist_mean, "summary_dist_to_ref.png"),
        ("Minimum clearance [mm]", clearance_min, "summary_min_clearance.png"),
    ]

    for title, values, filename in metrics:
        plt.figure(figsize=(10, 5))
        x = np.arange(len(labels))
        plt.bar(x, values)
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel(title)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        savefig(outdir, filename)


def plot_tip_xy(runs: Dict[str, pd.DataFrame], outdir: Path):
    plt.figure(figsize=(7, 7))

    for label, df in runs.items():
        x = get_col(df, "tip_x")
        y = get_col(df, "tip_y")
        plt.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=label)

    plt.xlabel("tip x [m]")
    plt.ylabel("tip y [m]")
    plt.title("Tip XY trajectory")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, "tip_xy_trajectory.png")


# ============================================================
# Summary table
# ============================================================

def build_summary_table(runs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for label, df in runs.items():
        pred1 = mm(df, "pred1_err_xy_m", "pred1_err_xy_mm")
        dist = mm(df, "dist_to_ref_m", "dist_to_ref_mm")
        clearance = mm(df, "clearance_m", "clearance_mm")
        jac_angle = get_col(df, "jac_angle_actual_deg")
        jac_gain = get_col(df, "jac_gain_actual_over_jac")
        infeas = get_col(df, "infeasible")
        sqp_iters = get_col(df, "sqp_iters_done")
        sqp_conv = get_col(df, "sqp_converged")
        sqp_du_rel = get_col(df, "sqp_du_rel_final")

        E, _ = rollout_matrix_mm(df)

        if E.shape[1] > 0:
            rollout_mean = finite_mean(E.ravel())
            rollout_max = finite_max(E.ravel())
            rollout_final_mean = finite_mean(E[:, -1])
            rollout_final_max = finite_max(E[:, -1])
        else:
            rollout_mean = np.nan
            rollout_max = np.nan
            rollout_final_mean = np.nan
            rollout_final_max = np.nan

        rows.append({
            "run": label,
            "num_rows": len(df),
            "infeasible_count": int(np.nansum(infeas == 1)),
            "mean_sqp_iters": finite_mean(sqp_iters),
            "sqp_converged_count": int(np.nansum(sqp_conv == 1)),
            "mean_sqp_du_rel_final": finite_mean(sqp_du_rel),

            "mean_pred1_xy_mm": finite_mean(pred1),
            "max_pred1_xy_mm": finite_max(pred1),

            "mean_rollout_xy_mm": rollout_mean,
            "max_rollout_xy_mm": rollout_max,
            "mean_final_rollout_xy_mm": rollout_final_mean,
            "max_final_rollout_xy_mm": rollout_final_max,

            "mean_dist_to_ref_mm": finite_mean(dist),
            "min_dist_to_ref_mm": finite_min(dist),
            "max_dist_to_ref_mm": finite_max(dist),

            "min_clearance_mm": finite_min(clearance),
            "mean_clearance_mm": finite_mean(clearance),

            "mean_jac_angle_deg": finite_mean(jac_angle),
            "max_jac_angle_deg": finite_max(jac_angle),
            "mean_jac_gain": finite_mean(jac_gain),
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    outdir = setup_outdir(OUTDIR)
    runs = load_runs(RUNS)

    yscale = "log" if LOGSCALE_ERRORS else "linear"

    plot_time_series(
        runs,
        y_getter=lambda df: mm(df, "pred1_err_xy_m", "pred1_err_xy_mm"),
        ylabel="One-step XY prediction error [mm]",
        title="One-step prediction error",
        outdir=outdir,
        filename="time_pred1_xy_error.png",
        yscale=yscale,
    )

    plot_time_series(
        runs,
        y_getter=lambda df: mm(df, "dist_to_ref_m", "dist_to_ref_mm"),
        ylabel="Distance to reference [mm]",
        title="Distance to reference",
        outdir=outdir,
        filename="time_dist_to_ref.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: mm(df, "clearance_m", "clearance_mm"),
        ylabel="Clearance [mm]",
        title="Clearance",
        outdir=outdir,
        filename="time_clearance.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "tip_vessel_angle_deg"),
        ylabel="Tip-vessel angle [deg]",
        title="Tip-vessel angle",
        outdir=outdir,
        filename="time_tip_vessel_angle.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "jac_angle_actual_deg"),
        ylabel="Jacobian predicted-vs-actual angle [deg]",
        title="Jacobian movement direction error",
        outdir=outdir,
        filename="time_jac_angle.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "jac_gain_actual_over_jac"),
        ylabel="Actual / Jacobian predicted motion gain",
        title="Jacobian gain actual / predicted",
        outdir=outdir,
        filename="time_jac_gain.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "sqp_du_rel_final"),
        ylabel="Final SQP relative update",
        title="SQP relative update",
        outdir=outdir,
        filename="time_sqp_du_rel_final.png",
        yscale=yscale,
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "sqp_iters_done"),
        ylabel="SQP iterations done",
        title="SQP iterations",
        outdir=outdir,
        filename="time_sqp_iters_done.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: get_col(df, "infeasible"),
        ylabel="Infeasible flag",
        title="Infeasibility",
        outdir=outdir,
        filename="time_infeasible.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: control_norms(df)["trans_norm"],
        ylabel="||u_trans||",
        title="Translational control magnitude",
        outdir=outdir,
        filename="time_control_trans_norm.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: control_norms(df)["omega_norm"],
        ylabel="||omega||",
        title="Angular control magnitude",
        outdir=outdir,
        filename="time_control_omega_norm.png",
    )

    plot_time_series(
        runs,
        y_getter=lambda df: control_norms(df)["dL"],
        ylabel="dL command",
        title="Insertion command dL",
        outdir=outdir,
        filename="time_control_dL.png",
    )

    plot_rollout_stage_means(runs, outdir)
    plot_rollout_stage_max(runs, outdir)
    plot_summary_bars(runs, outdir)
    plot_tip_xy(runs, outdir)

    summary = build_summary_table(runs)
    summary_path = outdir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print(f"saved: {summary_path}")
    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary)


if __name__ == "__main__":
    main()