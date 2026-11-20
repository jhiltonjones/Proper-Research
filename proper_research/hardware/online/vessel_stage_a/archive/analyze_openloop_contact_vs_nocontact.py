"""Mechanistic analysis of the four live open-loop runs (2026-10-07):
openloop_contact rep1/rep2 (plan built with the contact Jacobian/plant) vs
openloop_nocontact rep1/rep2 (plan built with no-contact Jacobian/plant).

Does the physical experiment behave as predicted when the planner
neglects contact? Analysis items, in the priority order requested:

1. Measured tip error e(s) = ||p_tip,meas(s) - p_ref(s)|| vs path progress
   s, contact onset and the 10mm abort threshold marked.
2. Common-interval (0 <= s <= s_common = min(s_end_C, s_end_NC)) RMSE,
   median, p95, max, and path-normalized integrated error E_AUC -- never
   a full-path mean for C against a truncated mean for NC.
3. (separate, slower script: validate_contact_model_against_hardware.py)
4. Onset of divergence: Delta_e(s) = e_NC(s) - e_C(s), compared against
   the offline-predicted contact onset (s=28.50mm, from
   plans/stage3_design/PC_vs_PNC_comparison.csv's e_NC_to_C>0.5mm onset)
   and the measured s_10mm.
5. Along-path/cross-path decomposition of e using the reference path's
   own tangent (finite-differenced from the contact plan's dense
   desired_position_m -- both plans share the identical target path, only
   resampled at different density).
6. Execution-fidelity check: joint tracking error (q_meas vs q_target,
   from path_follow.jsonl) and source-magnet FK execution error, to rule
   out "NC simply tracked its own command worse" as an alternative
   explanation for its larger tip error.
7. Planned insertion L(s) and source-magnet position vs s for the two
   plans' own state_reference, next to Delta_e(s) -- the causal story
   (wall contact -> different required configuration -> P_C/P_NC separate
   -> NC tip error accumulates).

Outputs (all under plans/stage3_design/openloop_hw_analysis/):
  fig1_tip_error_vs_s.png        -- headline figure (item 1)
  fig3_configuration_vs_s.png    -- causal-story figure (item 7)
  common_interval_stats.csv      -- item 2
  divergence_onset.csv           -- item 4
  per_tick_runs.csv              -- joined per-run per-tick table (items 1,4,5,6)

Usage: python analyze_openloop_contact_vs_nocontact.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)

RUN_DIRS = {
    "contact_rep1": "close_loop_logs/myrun/openloop_contact_2026-10-07_20261007T113710Z",
    "contact_rep2": "close_loop_logs/myrun/openloop_contact_2026-10-07_20261007T114131Z",
    "nocontact_rep1": "close_loop_logs/myrun/openloop_nocontact_2026-10-07_20261007T114533Z",
    "nocontact_rep2": "close_loop_logs/myrun/openloop_nocontact_2026-10-07_20261007T114830Z",
}
RUN_PLAN_ROOT = {
    "contact_rep1": "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06",
    "contact_rep2": "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06",
    "nocontact_rep1": "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06",
    "nocontact_rep2": "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06",
}
IS_CONTACT = {"contact_rep1": True, "contact_rep2": True, "nocontact_rep1": False, "nocontact_rep2": False}

OUT_DIR = Path("plans/stage3_design/openloop_hw_analysis")
# Predicted contact onset: first s where the REAL contact model's prediction
# for the no-contact plan's own path diverges from target by >0.5mm -- see
# plans/stage3_design/PC_vs_PNC_comparison.csv (compare_PC_vs_PNC.py).
S_CONTACT_ONSET_MM = 28.50
ABORT_THRESHOLD_MM = 10.0

_robot_kin = _resolve_robot_kinematics(make_robot_config())


def magnet_xyz(q6: np.ndarray) -> np.ndarray:
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def load_run(label: str) -> pd.DataFrame:
    run_dir = RUN_DIRS[label]
    plan_root = RUN_PLAN_ROOT[label]
    df = pd.read_csv(f"{run_dir}/tip_trajectory.csv")

    npz = np.load(f"{plan_root}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    path_s_m = npz["path_s_m"]
    ref_index = df["ref_index"].to_numpy()
    ref_index_clamped = np.clip(ref_index, 0, len(path_s_m) - 1)
    df["s_mm"] = path_s_m[ref_index_clamped] * 1e3

    recs = [json.loads(line) for line in open(f"{run_dir}/path_follow.jsonl")]
    q_meas = {r["step"]: np.asarray(r["q_meas_rad"], dtype=float) for r in recs}
    q_target = {r["step"]: np.asarray(r["q_target_rad"], dtype=float) for r in recs}

    df["joint_err_rad"] = df["step"].map(lambda s: float(np.linalg.norm(q_meas[s] - q_target[s])))
    df["magnet_exec_err_mm"] = df["step"].map(
        lambda s: float(np.linalg.norm(magnet_xyz(q_meas[s]) - magnet_xyz(q_target[s]))) * 1e3
    )
    df["magnet_target_x_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_target[s])[0]) * 1e3)
    df["magnet_target_y_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_target[s])[1]) * 1e3)
    df["magnet_target_z_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_target[s])[2]) * 1e3)
    df["magnet_meas_x_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_meas[s])[0]) * 1e3)
    df["magnet_meas_y_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_meas[s])[1]) * 1e3)
    df["magnet_meas_z_mm"] = df["step"].map(lambda s: float(magnet_xyz(q_meas[s])[2]) * 1e3)

    df["label"] = label
    df["is_contact"] = IS_CONTACT[label]
    df["plan_root"] = plan_root
    return df


def path_tangent_table():
    """Central-difference tangent of the (shared) reference path, built from
    the denser contact plan's own desired_position_m -- both plans track the
    identical target path, just resampled at different density."""
    npz = np.load(
        f"{RUN_PLAN_ROOT['contact_rep1']}/time_parameterized_configuration_path/"
        f"time_parameterized_configuration_path.npz"
    )
    s_m = npz["path_s_m"]
    p = npz["desired_position_m"]
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    t = t / norms
    return s_m, t


def tangent_at(s_mm: np.ndarray, s_grid_m: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    s_m = s_mm * 1e-3
    out = np.empty((len(s_mm), 3))
    for k in range(3):
        out[:, k] = np.interp(s_m, s_grid_m, t_grid[:, k])
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def decompose_error(df: pd.DataFrame, s_grid_m: np.ndarray, t_grid: np.ndarray) -> None:
    e = df[["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy() - df[["des_x_m", "des_y_m", "des_z_m"]].to_numpy()
    t = tangent_at(df["s_mm"].to_numpy(), s_grid_m, t_grid)
    e_par = np.einsum("ij,ij->i", e, t)
    e_perp_vec = e - e_par[:, None] * t
    df["e_parallel_mm"] = e_par * 1e3
    df["e_perp_mm"] = np.linalg.norm(e_perp_vec, axis=1) * 1e3


def interp_error_on_grid(df: pd.DataFrame, s_grid_mm: np.ndarray) -> np.ndarray:
    """Interpolate this run's own err_norm_mm onto s_grid_mm; NaN beyond the
    run's own measured s-range (so a truncated run cannot silently extend)."""
    s = df["s_mm"].to_numpy()
    e = df["err_norm_mm"].to_numpy()
    order = np.argsort(s)
    s, e = s[order], e[order]
    out = np.interp(s_grid_mm, s, e, left=np.nan, right=np.nan)
    out[s_grid_mm > s.max()] = np.nan
    return out


def summarize_common_interval(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s_end = {label: df["s_mm"].max() for label, df in runs.items()}
    s_common = min(s_end.values())
    grid = np.arange(0.0, s_common, 0.1)
    rows = []
    for label, df in runs.items():
        e_full = df["err_norm_mm"].to_numpy()
        auc_full = float(np.trapz(e_full, df["s_mm"].to_numpy())) / max(df["s_mm"].max(), 1e-9)
        e_on_common = interp_error_on_grid(df, grid)
        valid = ~np.isnan(e_on_common)
        auc_common = float(np.trapz(e_on_common[valid], grid[valid])) / s_common
        rows.append({
            "label": label, "is_contact": IS_CONTACT[label],
            "s_end_mm": s_end[label], "s_common_mm": s_common,
            "rmse_common_mm": float(np.sqrt(np.mean(e_on_common[valid] ** 2))),
            "median_common_mm": float(np.median(e_on_common[valid])),
            "p95_common_mm": float(np.percentile(e_on_common[valid], 95)),
            "max_common_mm": float(np.max(e_on_common[valid])),
            "auc_common_mm": auc_common,
            "rmse_full_mm": float(np.sqrt(np.mean(e_full ** 2))),
            "max_full_mm": float(np.max(e_full)),
            "auc_full_mm": auc_full,
        })
    return pd.DataFrame(rows), grid, s_common


def divergence_onset(runs: dict[str, pd.DataFrame], grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for c_label, nc_label in [("contact_rep1", "nocontact_rep1"), ("contact_rep2", "nocontact_rep2")]:
        e_c = interp_error_on_grid(runs[c_label], grid)
        e_nc = interp_error_on_grid(runs[nc_label], grid)
        valid = ~(np.isnan(e_c) | np.isnan(e_nc))
        d = np.where(valid, e_nc - e_c, np.nan)
        pre = grid < S_CONTACT_ONSET_MM
        pre_valid = pre & valid
        noise_floor = float(np.nanstd(d[pre_valid])) if pre_valid.any() else 0.0
        noise_mean = float(np.nanmean(d[pre_valid])) if pre_valid.any() else 0.0
        threshold = noise_mean + 3.0 * noise_floor
        post = (grid >= S_CONTACT_ONSET_MM) & valid
        s_divergence = float("nan")
        idx_post = np.where(post)[0]
        for i in range(len(idx_post) - 4):
            window = d[idx_post[i:i + 5]]
            if np.all(window > threshold):
                s_divergence = float(grid[idx_post[i]])
                break
        rows.append({
            "pair": f"{c_label}_vs_{nc_label}",
            "pre_contact_noise_mean_mm": noise_mean, "pre_contact_noise_std_mm": noise_floor,
            "threshold_mm": threshold, "s_contact_onset_mm": S_CONTACT_ONSET_MM,
            "s_measured_divergence_mm": s_divergence,
            "s_10mm_nc_mm": float(runs[nc_label]["s_mm"].max()),
        })
    return pd.DataFrame(rows)


def make_fig1(runs: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"contact_rep1": "#1b7f3b", "contact_rep2": "#5cb373",
              "nocontact_rep1": "#b3331d", "nocontact_rep2": "#e08070"}
    for label, df in runs.items():
        ax.plot(df["s_mm"].to_numpy(), df["err_norm_mm"].to_numpy(), color=colors[label], lw=1.3,
                label=f"{label} ({'completed' if IS_CONTACT[label] else 'ABORTED'})",
                linestyle="-" if "rep1" in label else "--")
    ax.axvline(S_CONTACT_ONSET_MM, color="k", linestyle=":", lw=1.2,
               label=f"predicted contact onset (s={S_CONTACT_ONSET_MM:.1f}mm)")
    ax.axhline(ABORT_THRESHOLD_MM, color="gray", linestyle="--", lw=1.2,
               label=f"abort threshold ({ABORT_THRESHOLD_MM:.0f}mm)")
    ax.set_xlabel("path progress s (mm)")
    ax.set_ylabel("measured tip tracking error e(s) (mm)")
    ax.set_title("Open-loop hardware: contact vs no-contact tip tracking", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_tip_error_vs_s.png", dpi=160)
    plt.close(fig)


def make_fig3(runs: dict[str, pd.DataFrame], grid: np.ndarray) -> None:
    npz_c = np.load(
        f"{RUN_PLAN_ROOT['contact_rep1']}/time_parameterized_configuration_path/"
        f"time_parameterized_configuration_path.npz"
    )
    npz_nc = np.load(
        f"{RUN_PLAN_ROOT['nocontact_rep1']}/time_parameterized_configuration_path/"
        f"time_parameterized_configuration_path.npz"
    )
    s_c_mm = npz_c["path_s_m"] * 1e3
    s_nc_mm = npz_nc["path_s_m"] * 1e3
    L_c_mm = npz_c["state_reference"][:, 6] * 1e3
    L_nc_mm = npz_nc["state_reference"][:, 6] * 1e3

    def magnet_xy_from_q(q_rows):
        return np.array([magnet_xyz(q) for q in q_rows])

    mag_c = magnet_xy_from_q(npz_c["state_reference"][:, :6])
    mag_nc = magnet_xy_from_q(npz_nc["state_reference"][:, :6])

    e_c = interp_error_on_grid(runs["contact_rep1"], grid)
    e_nc = interp_error_on_grid(runs["nocontact_rep1"], grid)
    valid = ~(np.isnan(e_c) | np.isnan(e_nc))
    delta_e = np.where(valid, e_nc - e_c, np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(s_c_mm, L_c_mm, color="#1b7f3b", label="planned insertion L(s), contact plan")
    axes[0].plot(s_nc_mm, L_nc_mm, color="#b3331d", label="planned insertion L(s), no-contact plan")
    axes[0].axvline(S_CONTACT_ONSET_MM, color="k", linestyle=":", lw=1.0)
    axes[0].set_ylabel("insertion L (mm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(s_c_mm, mag_c[:, 0] * 1e3, color="#1b7f3b", label="magnet x, contact plan")
    axes[1].plot(s_nc_mm, mag_nc[:, 0] * 1e3, color="#b3331d", label="magnet x, no-contact plan")
    axes[1].axvline(S_CONTACT_ONSET_MM, color="k", linestyle=":", lw=1.0)
    axes[1].set_ylabel("source-magnet x (mm, robot frame)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(grid, delta_e, color="#333333", label=r"$\Delta e(s)=e_{NC}(s)-e_C(s)$ (rep1 pair)")
    axes[2].axvline(S_CONTACT_ONSET_MM, color="k", linestyle=":", lw=1.0,
                    label=f"predicted contact onset ({S_CONTACT_ONSET_MM:.1f}mm)")
    axes[2].axhline(0.0, color="gray", lw=0.8)
    axes[2].set_ylabel(r"$\Delta e(s)$ (mm)")
    axes[2].set_xlabel("path progress s (mm)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Planner configuration divergence vs measured tip-error divergence")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_configuration_vs_s.png", dpi=160)
    plt.close(fig)


def make_fig5_magnet_xy_trajectory(runs: dict[str, pd.DataFrame]) -> None:
    """Measured source-magnet x vs y (top-down, robot frame) -- the actual
    spatial path the magnet traced, with no path-progress axis, so the
    pre-contact overlap and post-contact separation are visible directly
    as two trajectories diverging in space rather than two curves
    separating vs s."""
    colors = {"contact_rep1": "#1b7f3b", "contact_rep2": "#5cb373",
              "nocontact_rep1": "#b3331d", "nocontact_rep2": "#e08070"}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for label, df in runs.items():
        x = df["magnet_meas_x_mm"].to_numpy()
        y = df["magnet_meas_y_mm"].to_numpy()
        ax.plot(x, y, color=colors[label], lw=1.4,
                linestyle="-" if "rep1" in label else "--", label=label)
        ax.plot(x[0], y[0], marker="o", color=colors[label], ms=5)
        ax.plot(x[-1], y[-1], marker="s", color=colors[label], ms=6)
    ax.plot([], [], marker="o", color="k", ms=5, linestyle="none", label="start")
    ax.plot([], [], marker="s", color="k", ms=6, linestyle="none", label="end")
    ax.set_xlabel("magnet x (mm, robot frame)")
    ax.set_ylabel("magnet y (mm, robot frame)")
    ax.set_title("Measured source-magnet trajectory: contact vs no-contact", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_magnet_xy_trajectory.png", dpi=160)
    plt.close(fig)


def make_fig4_magnet_and_joints(runs: dict[str, pd.DataFrame]) -> None:
    """Measured (not planned) source-magnet position and all six joint
    angles vs path progress s, contact vs no-contact, both reps -- the
    actual executed hardware trajectory, from FK of the measured joints
    logged every tick (q1..q6 in tip_trajectory.csv == q_meas_rad)."""
    colors = {"contact_rep1": "#1b7f3b", "contact_rep2": "#5cb373",
              "nocontact_rep1": "#b3331d", "nocontact_rep2": "#e08070"}
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)

    magnet_axes = axes[0]
    for comp, ax, ylabel in zip(
        ["magnet_meas_x_mm", "magnet_meas_y_mm", "magnet_meas_z_mm"],
        magnet_axes, ["magnet x (mm)", "magnet y (mm)", "magnet z (mm)"],
    ):
        for label, df in runs.items():
            ax.plot(df["s_mm"].to_numpy(), df[comp].to_numpy(), color=colors[label], lw=1.1,
                    linestyle="-" if "rep1" in label else "--", label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    magnet_axes[1].set_title("source-magnet position (robot frame, measured)", fontsize=11)
    magnet_axes[0].legend(fontsize=6, loc="best")

    joint_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    for i, (col, ax) in enumerate(zip(joint_cols, list(axes[1]) + list(axes[2]))):
        for label, df in runs.items():
            ax.plot(df["s_mm"].to_numpy(), np.degrees(df[col].to_numpy()), color=colors[label],
                     lw=1.1, linestyle="-" if "rep1" in label else "--", label=label)
        ax.set_ylabel(f"{col} (deg)")
        ax.grid(alpha=0.3)
        if i >= 3:
            ax.set_xlabel("path progress s (mm)")

    fig.suptitle("Open-loop hardware: measured source-magnet position & joint angles", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "fig4_magnet_and_joints_vs_s.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s_grid_m, t_grid = path_tangent_table()

    runs = {label: load_run(label) for label in RUN_DIRS}
    for df in runs.values():
        decompose_error(df, s_grid_m, t_grid)

    print("=== per-run completion summary ===")
    for label, df in runs.items():
        print(f"  {label:16s} rows={len(df):4d} s_end={df['s_mm'].max():6.2f}mm "
              f"final_e={df['err_norm_mm'].iloc[-1]:6.2f}mm max_e={df['err_norm_mm'].max():6.2f}mm "
              f"{'COMPLETED' if IS_CONTACT[label] else 'ABORTED@10mm'}")

    stats, grid, s_common = summarize_common_interval(runs)
    stats.to_csv(OUT_DIR / "common_interval_stats.csv", index=False)
    print(f"\n=== common-interval (0..{s_common:.2f}mm) stats ===")
    print(stats[["label", "s_end_mm", "rmse_common_mm", "median_common_mm",
                  "p95_common_mm", "max_common_mm", "auc_common_mm"]].to_string(index=False))

    onset = divergence_onset(runs, grid)
    onset.to_csv(OUT_DIR / "divergence_onset.csv", index=False)
    print("\n=== divergence onset (item 4) ===")
    print(onset.to_string(index=False))

    print("\n=== execution fidelity (item 6): joint + magnet-FK tracking ===")
    for label, df in runs.items():
        print(f"  {label:16s} joint_err_rad: rms={np.sqrt(np.mean(df['joint_err_rad']**2)):.5f} "
              f"max={df['joint_err_rad'].max():.5f}  "
              f"magnet_exec_err_mm: rms={np.sqrt(np.mean(df['magnet_exec_err_mm']**2)):.4f} "
              f"max={df['magnet_exec_err_mm'].max():.4f}")

    print("\n=== along/cross-path decomposition (item 5), common interval means ===")
    for label, df in runs.items():
        mask = df["s_mm"] <= s_common
        print(f"  {label:16s} mean|e_par|={df.loc[mask,'e_parallel_mm'].abs().mean():.4f}mm "
              f"mean e_perp={df.loc[mask,'e_perp_mm'].mean():.4f}mm "
              f"(full-run mean e_perp={df['e_perp_mm'].mean():.4f}mm)")

    combined = pd.concat(runs.values(), ignore_index=True)
    combined.drop(columns=[c for c in combined.columns if combined[c].dtype == object and c not in
                            ("label", "plan_root")], errors="ignore")
    keep_cols = ["label", "is_contact", "step", "t_s", "ref_index", "s_mm", "err_norm_mm",
                 "e_parallel_mm", "e_perp_mm", "joint_err_rad", "magnet_exec_err_mm",
                 "insertion_m", "terminal_hold"]
    combined[keep_cols].to_csv(OUT_DIR / "per_tick_runs.csv", index=False)

    make_fig1(runs)
    make_fig3(runs, grid)
    make_fig4_magnet_and_joints(runs)
    make_fig5_magnet_xy_trajectory(runs)
    print(f"\nsaved figures + CSVs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
