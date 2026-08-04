from __future__ import annotations

from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_ROOT = Path("testing_full_sqp_np_predicitions")
OUT_DIR = RUN_ROOT / "analysis_prediction_horizon_sqp"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def parse_bend_angle(name: str) -> float:
    m = re.search(r"bend_([mp])(\d+)", name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def parse_bend_abs(name: str) -> float:
    val = parse_bend_angle(name)
    return abs(val) if np.isfinite(val) else np.nan


def parse_jacobian_variant(name: str) -> str:
    if "jac_no_contact" in name:
        return "no_contact"
    if "jac_contact" in name:
        return "contact"
    return "unknown"


def parse_Np(name: str) -> int:
    m = re.search(r"_Np(\d+)", name)
    return int(m.group(1)) if m else -1


def parse_rollout(name: str) -> int:
    m = re.search(r"_rollout(\d+)", name)
    return int(m.group(1)) if m else -1


def bend_tag(angle: float) -> str:
    if angle < 0:
        return f"m{int(abs(angle))}"
    return f"p{int(abs(angle))}"


def variant_label(v: str) -> str:
    return {
        "contact": "Contact-aware Jacobian",
        "no_contact": "Non-contact Jacobian",
    }.get(v, v)


def load_lumen(run_dir: Path):
    C_path = run_dir / "lumen_C.npy"
    R_path = run_dir / "lumen_R.npy"

    C = np.load(C_path) if C_path.exists() else None
    R = np.load(R_path) if R_path.exists() else None

    return C, R


def nearest_centreline_distance_mm(points_xyz: np.ndarray, lumen_C: np.ndarray) -> np.ndarray:
    """
    Compute nearest xy distance from each point to the centreline.

    This is used for predicted terminal trajectory diagnostics.
    """
    points_xyz = np.asarray(points_xyz, float)
    lumen_C = np.asarray(lumen_C, float)

    if points_xyz.ndim != 2 or points_xyz.shape[1] < 2:
        return np.full(points_xyz.shape[0], np.nan)

    if lumen_C.ndim != 2 or lumen_C.shape[1] < 2:
        return np.full(points_xyz.shape[0], np.nan)

    P = points_xyz[:, :2]
    C = lumen_C[:, :2]

    out = []
    for p in P:
        d = np.linalg.norm(C - p[None, :], axis=1)
        out.append(1e3 * float(np.nanmin(d)))

    return np.asarray(out, float)


def first_bad_step(df: pd.DataFrame, pred_err_limit_mm: float = 5.0):
    status = df["status"].astype(str) if "status" in df.columns else pd.Series("", index=df.index)
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    pred = safe_num(df, "adapt_pred_err_xy_mm")

    bad = (
        (infeasible > 0)
        | status.str.lower().str.contains("infeasible", na=False)
        | (pred > pred_err_limit_mm)
    )

    if not bad.any():
        return np.nan

    return int(np.where(bad.to_numpy())[0][0])


# ---------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------

def summarise_run(run_dir: Path) -> dict:
    log_path = run_dir / "log.csv"
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    df = pd.read_csv(log_path)

    run_name = run_dir.name
    bend_angle = parse_bend_angle(run_name)
    bend_abs = parse_bend_abs(run_name)
    jac = parse_jacobian_variant(run_name)
    Np = parse_Np(run_name)
    rollout = parse_rollout(run_name)

    C, R = load_lumen(run_dir)

    rho = safe_num(df, "rho_mm")
    clearance = safe_num(df, "clearance_mm")
    angle = safe_num(df, "tip_vessel_angle_deg")

    pred1 = safe_num(df, "pred1_err_xy_mm")
    adapt_pred = safe_num(df, "adapt_pred_err_xy_mm")

    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    status = df["status"].astype(str) if "status" in df.columns else pd.Series("", index=df.index)

    i_ref = safe_num(df, "i_ref")
    i_ref_mpc = safe_num(df, "i_ref_mpc")
    idx_terminal = safe_num(df, "idx_ref_terminal")

    cond_beam = safe_num(df, "cond_H_beam")
    cond_mpc = safe_num(df, "cond_H_mpc")

    sqp_du = safe_num(df, "sqp_du_rel_final")

    # Applied rollout errors. For rollout=1, this will mainly be stage 0.
    rollout_err_cols = [
        c for c in df.columns
        if re.match(r"rollout_err_xy_\d+_mm$", c)
    ]

    rollout_err_values = []
    for c in rollout_err_cols:
        vals = safe_num(df, c).to_numpy(float)
        rollout_err_values.append(vals)

    if rollout_err_values:
        rollout_err_mat = np.vstack(rollout_err_values).T
        max_rollout_err = np.nanmax(rollout_err_mat, axis=1)
        mean_rollout_err = np.nanmean(rollout_err_mat, axis=1)
    else:
        max_rollout_err = np.full(len(df), np.nan)
        mean_rollout_err = np.full(len(df), np.nan)

    # Prediction-horizon terminal plan diagnostic:
    # distance of x_pred_{Np-1} to the centreline, if available.
    terminal_plan_rho_mm = np.full(len(df), np.nan)
    terminal_x_col = f"x_pred_{Np - 1}_x"
    terminal_y_col = f"x_pred_{Np - 1}_y"
    terminal_z_col = f"x_pred_{Np - 1}_z"

    if C is not None and terminal_x_col in df.columns and terminal_y_col in df.columns:
        z = safe_num(df, terminal_z_col) if terminal_z_col in df.columns else pd.Series(0.0, index=df.index)
        pts = np.column_stack([
            safe_num(df, terminal_x_col),
            safe_num(df, terminal_y_col),
            z,
        ])
        terminal_plan_rho_mm = nearest_centreline_distance_mm(pts, C)

    # Control effort from first applied command.
    u_cols = [
        "u0_vx", "u0_vy", "u0_vz",
        "u0_wx", "u0_wy", "u0_wz",
        "u0_dL",
    ]
    available_u = [c for c in u_cols if c in df.columns]
    if available_u:
        U0 = df[available_u].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        u0_norm = np.linalg.norm(U0, axis=1)
    else:
        u0_norm = np.full(len(df), np.nan)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "bend_angle_deg": bend_angle,
        "bend_abs_deg": bend_abs,
        "jacobian_variant": jac,
        "Np": int(Np),
        "rollout": int(rollout),

        "n_steps": int(len(df)),
        "success": int(not (infeasible > 0).any()),
        "num_infeasible_steps": int(np.nansum(infeasible)),
        "first_bad_step": first_bad_step(df, pred_err_limit_mm=5.0),

        "final_i_ref": float(i_ref.dropna().iloc[-1]) if i_ref.notna().any() else np.nan,
        "max_i_ref": float(np.nanmax(i_ref)) if i_ref.notna().any() else np.nan,
        "final_i_ref_mpc": float(i_ref_mpc.dropna().iloc[-1]) if i_ref_mpc.notna().any() else np.nan,
        "final_idx_ref_terminal": float(idx_terminal.dropna().iloc[-1]) if idx_terminal.notna().any() else np.nan,

        "mean_rho_mm": float(np.nanmean(rho)),
        "max_rho_mm": float(np.nanmax(rho)),
        "median_rho_mm": float(np.nanmedian(rho)),

        "min_clearance_mm": float(np.nanmin(clearance)),
        "mean_clearance_mm": float(np.nanmean(clearance)),

        "mean_tip_vessel_angle_deg": float(np.nanmean(angle)),
        "max_tip_vessel_angle_deg": float(np.nanmax(angle)),

        "mean_pred1_err_xy_mm": float(np.nanmean(pred1)),
        "max_pred1_err_xy_mm": float(np.nanmax(pred1)),
        "mean_adapt_pred_err_xy_mm": float(np.nanmean(adapt_pred)),
        "max_adapt_pred_err_xy_mm": float(np.nanmax(adapt_pred)),

        "mean_rollout_err_xy_mm": float(np.nanmean(mean_rollout_err)),
        "max_rollout_err_xy_mm": float(np.nanmax(max_rollout_err)),

        "mean_terminal_plan_rho_mm": float(np.nanmean(terminal_plan_rho_mm)),
        "max_terminal_plan_rho_mm": float(np.nanmax(terminal_plan_rho_mm)),

        "mean_cond_H_beam": float(np.nanmean(cond_beam)),
        "max_cond_H_beam": float(np.nanmax(cond_beam)),
        "mean_cond_H_mpc": float(np.nanmean(cond_mpc)),
        "max_cond_H_mpc": float(np.nanmax(cond_mpc)),

        "mean_sqp_du_rel_final": float(np.nanmean(sqp_du)),
        "max_sqp_du_rel_final": float(np.nanmax(sqp_du)),

        "mean_u0_norm": float(np.nanmean(u0_norm)),
        "max_u0_norm": float(np.nanmax(u0_norm)),
    }

    return summary


def build_summary(run_root: Path) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "log.csv").exists():
            continue
        if "bend_" not in run_dir.name:
            continue

        try:
            rows.append(summarise_run(run_dir))
        except Exception as e:
            print(f"[WARN] failed to summarise {run_dir.name}: {e}")

    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(
            ["bend_abs_deg", "jacobian_variant", "Np"]
        ).reset_index(drop=True)

    return summary


# ---------------------------------------------------------------------
# Contact benefit
# ---------------------------------------------------------------------

def build_contact_benefit(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Positive delta for error metrics means contact-aware is better.

    delta = no_contact - contact
    ratio = no_contact / contact
    """
    metrics = [
        "mean_rho_mm",
        "max_rho_mm",
        "mean_pred1_err_xy_mm",
        "max_pred1_err_xy_mm",
        "mean_adapt_pred_err_xy_mm",
        "max_adapt_pred_err_xy_mm",
        "mean_rollout_err_xy_mm",
        "max_rollout_err_xy_mm",
        "mean_terminal_plan_rho_mm",
        "max_terminal_plan_rho_mm",
        "max_tip_vessel_angle_deg",
        "mean_u0_norm",
    ]

    rows = []

    keys = ["bend_abs_deg", "Np"]

    for key_vals, g in summary.groupby(keys):
        if isinstance(key_vals, tuple):
            bend_abs, Np = key_vals
        else:
            bend_abs, Np = key_vals, np.nan

        gc = g[g["jacobian_variant"] == "contact"]
        gn = g[g["jacobian_variant"] == "no_contact"]

        if len(gc) != 1 or len(gn) != 1:
            continue

        c = gc.iloc[0]
        n = gn.iloc[0]

        row = {
            "bend_abs_deg": bend_abs,
            "Np": int(Np),
            "contact_success": int(c["success"]),
            "no_contact_success": int(n["success"]),
            "contact_final_i_ref": c["final_i_ref"],
            "no_contact_final_i_ref": n["final_i_ref"],
            "delta_final_i_ref_contact_minus_no_contact": c["final_i_ref"] - n["final_i_ref"],
        }

        for metric in metrics:
            c_val = float(c.get(metric, np.nan))
            n_val = float(n.get(metric, np.nan))

            row[f"delta_{metric}_no_contact_minus_contact"] = n_val - c_val

            if np.isfinite(c_val) and abs(c_val) > 1e-12:
                row[f"ratio_{metric}_no_contact_over_contact"] = n_val / c_val
            else:
                row[f"ratio_{metric}_no_contact_over_contact"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(["bend_abs_deg", "Np"]).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_metric_vs_Np(summary: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(4.2 * len(bends), 3.6),
        sharey=False,
    )

    if len(bends) == 1:
        axes = [axes]

    for ax, bend in zip(axes, bends):
        g = summary[summary["bend_abs_deg"] == bend]

        for variant in ["no_contact", "contact"]:
            gv = g[g["jacobian_variant"] == variant].sort_values("Np")
            if len(gv) == 0:
                continue

            ax.plot(
                gv["Np"],
                gv[metric],
                marker="o",
                linewidth=2.0,
                label=variant_label(variant),
            )

        ax.set_title(f"{int(bend)}° bend")
        ax.set_xlabel("Prediction horizon $N_p$")
        ax.set_xticks(sorted(g["Np"].dropna().unique()))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_contact_benefit(benefit: pd.DataFrame, metric: str, ylabel: str, out_path: Path):
    bends = sorted(benefit["bend_abs_deg"].dropna().unique())

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for bend in bends:
        g = benefit[benefit["bend_abs_deg"] == bend].sort_values("Np")
        if len(g) == 0:
            continue

        ax.plot(
            g["Np"],
            g[metric],
            marker="o",
            linewidth=2.0,
            label=f"{int(bend)}° bend",
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Prediction horizon $N_p$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_for_Np(
    run_root: Path,
    out_dir: Path,
    *,
    bend_abs_deg: int = 90,
    Np: int = 5,
):
    """
    Representative trajectory comparison for contact vs no-contact.
    Uses logged tip_x, tip_y and lumen_C.npy.
    """
    candidates = []

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "log.csv").exists():
            continue

        if int(parse_bend_abs(run_dir.name)) == int(bend_abs_deg) and parse_Np(run_dir.name) == int(Np):
            candidates.append(run_dir)

    if not candidates:
        print(f"[WARN] no trajectory candidates for bend={bend_abs_deg}, Np={Np}")
        return

    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    lumen_plotted = False

    for run_dir in candidates:
        df = pd.read_csv(run_dir / "log.csv")
        variant = parse_jacobian_variant(run_dir.name)

        C, _ = load_lumen(run_dir)
        if C is not None and not lumen_plotted:
            ax.plot(
                1e3 * C[:, 0],
                1e3 * C[:, 1],
                linestyle="--",
                linewidth=1.5,
                label="Vessel centreline",
            )
            lumen_plotted = True

        if "tip_x" in df.columns and "tip_y" in df.columns:
            ax.plot(
                1e3 * safe_num(df, "tip_x"),
                1e3 * safe_num(df, "tip_y"),
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=variant_label(variant),
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Trajectory comparison, {bend_abs_deg}° bend, $N_p={Np}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"trajectory_bend_{bend_abs_deg}_Np{Np}_contact_vs_no_contact.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_prediction_horizon_summary(summary: pd.DataFrame, benefit: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("mean_rho_mm", "Mean tracking error $\\rho$ [mm]", "mean_rho_vs_Np.png"),
        ("max_rho_mm", "Maximum tracking error $\\rho$ [mm]", "max_rho_vs_Np.png"),
        ("mean_pred1_err_xy_mm", "Mean one-step prediction error [mm]", "mean_pred1_err_vs_Np.png"),
        ("max_pred1_err_xy_mm", "Maximum one-step prediction error [mm]", "max_pred1_err_vs_Np.png"),
        ("min_clearance_mm", "Minimum clearance [mm]", "min_clearance_vs_Np.png"),
        ("max_tip_vessel_angle_deg", "Maximum tip-vessel angle [deg]", "max_tip_angle_vs_Np.png"),
        ("final_i_ref", "Final reference index", "final_i_ref_vs_Np.png"),
        ("mean_cond_H_mpc", "Mean MPC Hessian condition number", "mean_cond_H_mpc_vs_Np.png"),
        ("mean_u0_norm", "Mean first control norm", "mean_u0_norm_vs_Np.png"),
    ]

    for metric, ylabel, filename in metric_specs:
        if metric in summary.columns:
            plot_metric_vs_Np(
                summary,
                metric,
                ylabel,
                out_dir / filename,
            )

    benefit_specs = [
        (
            "delta_mean_rho_mm_no_contact_minus_contact",
            "Contact benefit in mean tracking error [mm]\n(no-contact − contact)",
            "benefit_mean_rho_vs_Np.png",
        ),
        (
            "delta_max_rho_mm_no_contact_minus_contact",
            "Contact benefit in max tracking error [mm]\n(no-contact − contact)",
            "benefit_max_rho_vs_Np.png",
        ),
        (
            "delta_mean_pred1_err_xy_mm_no_contact_minus_contact",
            "Contact benefit in mean prediction error [mm]\n(no-contact − contact)",
            "benefit_mean_pred1_vs_Np.png",
        ),
        (
            "delta_max_pred1_err_xy_mm_no_contact_minus_contact",
            "Contact benefit in max prediction error [mm]\n(no-contact − contact)",
            "benefit_max_pred1_vs_Np.png",
        ),
        (
            "delta_final_i_ref_contact_minus_no_contact",
            "Contact benefit in progress\n(contact final index − no-contact final index)",
            "benefit_progress_vs_Np.png",
        ),
    ]

    for metric, ylabel, filename in benefit_specs:
        if metric in benefit.columns:
            plot_contact_benefit(
                benefit,
                metric,
                ylabel,
                out_dir / filename,
            )

    for bend in [30, 60, 90]:
        for Np in [1, 3, 5]:
            plot_trajectories_for_Np(
                RUN_ROOT,
                out_dir,
                bend_abs_deg=bend,
                Np=Np,
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary(RUN_ROOT)
    benefit = build_contact_benefit(summary)

    summary_path = OUT_DIR / "prediction_horizon_contact_summary.csv"
    benefit_path = OUT_DIR / "prediction_horizon_contact_benefit.csv"

    summary.to_csv(summary_path, index=False)
    benefit.to_csv(benefit_path, index=False)

    with open(OUT_DIR / "prediction_horizon_contact_summary.json", "w") as f:
        json.dump(summary.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    with open(OUT_DIR / "prediction_horizon_contact_benefit.json", "w") as f:
        json.dump(benefit.replace({np.nan: None}).to_dict(orient="records"), f, indent=2)

    plot_prediction_horizon_summary(summary, benefit, OUT_DIR)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {benefit_path}")
    print(f"  plots -> {OUT_DIR}")

    if len(summary):
        cols = [
            "bend_abs_deg",
            "jacobian_variant",
            "Np",
            "success",
            "n_steps",
            "final_i_ref",
            "mean_rho_mm",
            "max_rho_mm",
            "mean_pred1_err_xy_mm",
            "max_pred1_err_xy_mm",
            "min_clearance_mm",
            "max_tip_vessel_angle_deg",
        ]
        cols = [c for c in cols if c in summary.columns]

        print("\nSummary:")
        print(summary[cols].to_string(index=False))

    if len(benefit):
        cols = [
            "bend_abs_deg",
            "Np",
            "delta_mean_rho_mm_no_contact_minus_contact",
            "delta_max_rho_mm_no_contact_minus_contact",
            "delta_mean_pred1_err_xy_mm_no_contact_minus_contact",
            "delta_max_pred1_err_xy_mm_no_contact_minus_contact",
            "delta_final_i_ref_contact_minus_no_contact",
        ]
        cols = [c for c in cols if c in benefit.columns]

        print("\nContact benefit:")
        print(benefit[cols].to_string(index=False))


if __name__ == "__main__":
    main()