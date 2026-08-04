from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RUN_ROOT = Path("run_figure5_simulation3_rollout_larger")
OUT_DIR = RUN_ROOT / "figure5_model_fidelity_analysis3_rollout"


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def parse_bend_angle_from_run_name(run_name: str):
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def infer_solver_mode(run_name: str):
    if "sqp_full" in run_name:
        return "sqp_full"
    if "ltv_oneshot" in run_name:
        return "ltv_oneshot"
    if "lti" in run_name:
        return "lti"
    return ""


def parse_rollout_steps_from_run_name(run_name: str):
    m = re.search(r"rollout(\d+)", run_name)
    if not m:
        return np.nan
    return int(m.group(1))


def parse_Np_from_run_name(run_name: str):
    m = re.search(r"_Np(\d+)", run_name)
    if not m:
        return np.nan
    return int(m.group(1))


def bend_tag(bend):
    return f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"


def safe_num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def safe_str(df, col):
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series("", index=df.index)


def pretty_solver(s):
    return {
        "lti": "LTI",
        "ltv_oneshot": "One-shot SQP",
        "sqp_full": "Full SQP",
    }.get(s, s)


# ---------------------------------------------------------------------
# Load and summarise
# ---------------------------------------------------------------------
def read_runs(run_root: Path):
    runs = []

    for log_csv in sorted(Path(run_root).glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name

        df = pd.read_csv(log_csv)

        runs.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "log_csv": log_csv,
                "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
                "bend_abs_deg": abs(parse_bend_angle_from_run_name(run_name)),
                "solver_mode": infer_solver_mode(run_name),
                "rollout_steps": parse_rollout_steps_from_run_name(run_name),
                "Np": parse_Np_from_run_name(run_name),
                "df": df,
            }
        )

    return runs


def summarise_run(run):
    df = run["df"]
    n = len(df)

    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)

    failure_mask = (
        (infeasible > 0)
        | (~status.isin(["solved", "solved_inaccurate", "solved inaccurate"]))
    )

    # Rollout prediction error columns
    rollout_err_cols = [
        c for c in df.columns
        if re.match(r"rollout_err_xy_\d+_mm", c)
    ]

    if rollout_err_cols:
        E = df[rollout_err_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        mean_rollout_err_xy_mm = float(np.nanmean(E))
        max_rollout_err_xy_mm = float(np.nanmax(E))

        # Terminal available rollout error = last rollout error column present
        terminal_col = sorted(
            rollout_err_cols,
            key=lambda c: int(re.search(r"rollout_err_xy_(\d+)_mm", c).group(1))
        )[-1]
        terminal_rollout_err_xy_mm = float(
            pd.to_numeric(df[terminal_col], errors="coerce").mean()
        )
        max_terminal_rollout_err_xy_mm = float(
            pd.to_numeric(df[terminal_col], errors="coerce").max()
        )
    else:
        mean_rollout_err_xy_mm = np.nan
        max_rollout_err_xy_mm = np.nan
        terminal_rollout_err_xy_mm = np.nan
        max_terminal_rollout_err_xy_mm = np.nan

    sqp_iters = safe_num(df, "sqp_iters_done")
    N_sqp_used = safe_num(df, "N_sqp_used")

    # Fallback complexity proxy:
    # LTI = 1, one-shot = 1, full SQP = N_sqp_used or sqp iters.
    if N_sqp_used.notna().any():
        complexity_step = N_sqp_used.fillna(1.0)
    elif sqp_iters.notna().any():
        complexity_step = sqp_iters.fillna(0.0)
    else:
        if run["solver_mode"] == "sqp_full":
            complexity_step = pd.Series(5.0, index=df.index)
        else:
            complexity_step = pd.Series(1.0, index=df.index)

    # Control effort
    u_cols = [
        "u0_vx",
        "u0_vy",
        "u0_vz",
        "u0_wx",
        "u0_wy",
        "u0_wz",
        "u0_dL",
    ]

    existing_u_cols = [c for c in u_cols if c in df.columns]

    if existing_u_cols:
        U = df[existing_u_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        control_rms = float(np.sqrt(np.nanmean(U**2)))
        control_max_abs = float(np.nanmax(np.abs(U)))
    else:
        control_rms = np.nan
        control_max_abs = np.nan

    row = {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),

        "bend_angle_deg": run["bend_angle_deg"],
        "bend_abs_deg": run["bend_abs_deg"],
        "solver_mode": run["solver_mode"],
        "rollout_steps": run["rollout_steps"],
        "Np": run["Np"],

        "n_steps": int(n),
        "success": bool((failure_mask == 0).all()) if n else False,
        "num_failed_steps": int(failure_mask.sum()) if n else 0,
        "num_infeasible_steps": int(infeasible.sum()) if n else 0,
        "final_status": str(status.iloc[-1]) if n else "",

        "final_i_ref": float(safe_num(df, "i_ref").iloc[-1]) if n else np.nan,
        "max_i_ref": float(safe_num(df, "i_ref").max()),

        # Tracking / prediction
        "mean_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").mean()),
        "max_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").max()),
        "mean_rollout_err_xy_mm": mean_rollout_err_xy_mm,
        "max_rollout_err_xy_mm": max_rollout_err_xy_mm,
        "terminal_rollout_err_xy_mm": terminal_rollout_err_xy_mm,
        "max_terminal_rollout_err_xy_mm": max_terminal_rollout_err_xy_mm,

        # Safety/contact
        "mean_clearance_mm": float(safe_num(df, "adapt_clearance_mm").mean()),
        "min_clearance_mm": float(safe_num(df, "adapt_clearance_mm").min()),
        "mean_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").mean()),
        "max_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").max()),

        # Conditioning
        "mean_cond_H_beam": float(safe_num(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_num(df, "cond_H_beam").max()),
        "mean_cond_H_mpc": float(safe_num(df, "cond_H_mpc").mean()),
        "max_cond_H_mpc": float(safe_num(df, "cond_H_mpc").max()),

        # SQP/convergence
        "mean_sqp_iters_done": float(sqp_iters.mean()),
        "max_sqp_iters_done": float(sqp_iters.max()),
        "sum_sqp_iters_done": float(sqp_iters.sum()),
        "mean_N_sqp_used": float(N_sqp_used.mean()),
        "max_N_sqp_used": float(N_sqp_used.max()),
        "sum_N_sqp_used": float(N_sqp_used.sum()),

        # Complexity proxy
        "mean_complexity_proxy": float(complexity_step.mean()),
        "sum_complexity_proxy": float(complexity_step.sum()),

        # Control effort
        "control_rms": control_rms,
        "control_max_abs": control_max_abs,
    }

    return row


def build_summary(runs, out_dir):
    rows = [summarise_run(r) for r in runs]
    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(
            ["bend_abs_deg", "rollout_steps", "solver_mode"]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "figure5_model_fidelity_summary.csv", index=False)

    with open(out_dir / "figure5_model_fidelity_summary.json", "w") as f:
        json.dump(
            summary.replace({np.nan: None}).to_dict(orient="records"),
            f,
            indent=2,
        )

    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_metric_vs_rollout_by_solver(summary, metric, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_order = ["lti", "ltv_oneshot", "sqp_full"]
    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(5.0 * len(bends), 4.0),
        sharey=True,
    )

    if len(bends) == 1:
        axes = [axes]

    for ax, bend in zip(axes, bends):
        for solver in solver_order:
            g = summary[
                (summary["bend_abs_deg"] == bend)
                & (summary["solver_mode"] == solver)
            ].copy()

            if g.empty or metric not in g.columns:
                continue

            g = g.sort_values("rollout_steps")

            ax.plot(
                g["rollout_steps"],
                g[metric],
                marker="o",
                linewidth=2.2,
                label=pretty_solver(solver),
            )

            failed = ~g["success"].astype(bool)
            if failed.any():
                ax.scatter(
                    g.loc[failed, "rollout_steps"],
                    g.loc[failed, metric],
                    marker="x",
                    s=90,
                    linewidths=2.0,
                    color="black",
                    zorder=10,
                )

        ax.set_title(f"{int(bend)}$^\\circ$ bend")
        ax.set_xlabel("Execution rollout length")
        ax.set_xticks(sorted(summary["rollout_steps"].dropna().unique()))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(fontsize=8)

    fig.suptitle(metric.replace("_", " ") + " vs rollout and model fidelity")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(
        out_dir / f"{metric}_vs_rollout_by_solver.png",
        dpi=300,
    )
    plt.close(fig)


def plot_metric_by_solver_grouped(summary, metric, out_dir):
    """
    One figure per rollout. Bars are solver modes, panels are bends.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_order = ["lti", "ltv_oneshot", "sqp_full"]
    bends = sorted(summary["bend_abs_deg"].dropna().unique())
    rollouts = sorted(summary["rollout_steps"].dropna().unique())

    for rollout in rollouts:
        fig, axes = plt.subplots(
            1,
            len(bends),
            figsize=(5.0 * len(bends), 4.0),
            sharey=True,
        )

        if len(bends) == 1:
            axes = [axes]

        for ax, bend in zip(axes, bends):
            values = []
            success_flags = []

            for solver in solver_order:
                g = summary[
                    (summary["bend_abs_deg"] == bend)
                    & (summary["rollout_steps"] == rollout)
                    & (summary["solver_mode"] == solver)
                ]

                if g.empty or metric not in g.columns:
                    values.append(np.nan)
                    success_flags.append(False)
                else:
                    values.append(float(g[metric].iloc[0]))
                    success_flags.append(bool(g["success"].iloc[0]))

            x = np.arange(len(solver_order))
            bars = ax.bar(x, values)

            for bar, ok in zip(bars, success_flags):
                if not ok:
                    bar.set_hatch("//")
                    bar.set_edgecolor("black")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [pretty_solver(s) for s in solver_order],
                rotation=20,
                ha="right",
            )
            ax.set_title(f"{int(bend)}$^\\circ$ bend")
            ax.grid(True, axis="y", alpha=0.3)

        axes[0].set_ylabel(metric.replace("_", " "))
        fig.suptitle(
            f"{metric.replace('_', ' ')} by model fidelity, rollout={int(rollout)}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        fig.savefig(
            out_dir / f"{metric}_by_solver_rollout_{int(rollout)}.png",
            dpi=300,
        )
        plt.close(fig)


def plot_pareto_accuracy_vs_complexity(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_order = ["lti", "ltv_oneshot", "sqp_full"]
    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    marker_map = {
        1: "o",
        3: "s",
        5: "^",
    }

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(5.0 * len(bends), 4.2),
        sharey=True,
    )

    if len(bends) == 1:
        axes = [axes]

    for ax, bend in zip(axes, bends):
        g_bend = summary[summary["bend_abs_deg"] == bend].copy()

        for solver in solver_order:
            g_solver = g_bend[g_bend["solver_mode"] == solver].copy()

            for _, row in g_solver.iterrows():
                rollout = int(row["rollout_steps"])

                ax.scatter(
                    row["sum_complexity_proxy"],
                    row["max_pred_err_xy_mm"],
                    s=90,
                    marker=marker_map.get(rollout, "o"),
                    label=f"{pretty_solver(solver)}, r={rollout}",
                )

                ax.annotate(
                    f"r{rollout}",
                    (
                        row["sum_complexity_proxy"],
                        row["max_pred_err_xy_mm"],
                    ),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

        ax.set_title(f"{int(bend)}$^\\circ$ bend")
        ax.set_xlabel("Total solver complexity proxy")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Max prediction error [mm]")

    # Remove duplicate legend labels.
    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[-1].legend(unique.values(), unique.keys(), fontsize=7)

    fig.suptitle("Accuracy--computation trade-off")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(out_dir / "pareto_accuracy_vs_complexity.png", dpi=300)
    plt.close(fig)


def plot_reference_traces_for_rollout(summary, runs, metric, rollout, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_order = ["lti", "ltv_oneshot", "sqp_full"]
    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    for bend in bends:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))

        for solver in solver_order:
            matches = [
                r for r in runs
                if r["bend_abs_deg"] == bend
                and r["rollout_steps"] == rollout
                and r["solver_mode"] == solver
            ]

            if not matches:
                continue

            run = matches[0]
            df = run["df"]

            if metric not in df.columns or "i_ref" not in df.columns:
                continue

            x = safe_num(df, "i_ref")
            y = safe_num(df, metric)

            valid = x.notna() & y.notna() & np.isfinite(y)

            if metric in ("cond_H_beam", "cond_H_mpc"):
                ax.set_yscale("log")

            ax.plot(
                x[valid],
                y[valid],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=pretty_solver(solver),
            )

        ax.set_title(
            f"{metric} vs reference index, {int(bend)}$^\\circ$ bend, rollout={rollout}"
        )
        ax.set_xlabel("Reference index")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        fig.tight_layout()

        fig.savefig(
            out_dir / f"{metric}_bend_{int(bend)}_rollout_{rollout}.png",
            dpi=300,
        )
        plt.close(fig)


def plot_trajectory_if_available(runs, out_dir):
    """
    Optional: trajectory plot if p_now_x/p_now_y and lumen_C.npy exist.
    Adjust column names if your log uses different state names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    solver_order = ["lti", "ltv_oneshot", "sqp_full"]

    # Hardest representative case: 90 degree, rollout 5.
    target_bend = 90.0
    target_rollout = 1

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    plotted_any = False

    for solver in solver_order:
        matches = [
            r for r in runs
            if r["bend_abs_deg"] == target_bend
            and r["rollout_steps"] == target_rollout
            and r["solver_mode"] == solver
        ]

        if not matches:
            continue

        run = matches[0]
        df = run["df"]

        # Try common column names.
        x_col_candidates = ["p_tip_x", "p_now_x", "tip_x", "x_tip", "p_now_0"]
        y_col_candidates = ["p_tip_y", "p_now_y", "tip_y", "y_tip", "p_now_1"]

        x_col = next((c for c in x_col_candidates if c in df.columns), None)
        y_col = next((c for c in y_col_candidates if c in df.columns), None)

        if x_col is None or y_col is None:
            continue

        x = safe_num(df, x_col)
        y = safe_num(df, y_col)

        valid = x.notna() & y.notna()

        ax.plot(
            x[valid],
            y[valid],
            linewidth=2.0,
            label=pretty_solver(solver),
        )

        plotted_any = True

    # Plot lumen centreline if available.
    # Use the first matching run directory.
    candidate_dirs = [
        r["run_dir"] for r in runs
        if r["bend_abs_deg"] == target_bend and r["rollout_steps"] == target_rollout
    ]

    if candidate_dirs:
        lumen_path = candidate_dirs[0] / "lumen_C.npy"
        if lumen_path.exists():
            C = np.load(lumen_path)
            if C.ndim == 2 and C.shape[1] >= 2:
                ax.plot(
                    C[:, 0],
                    C[:, 1],
                    linestyle=":",
                    linewidth=2.5,
                    color="black",
                    label="Vessel centreline",
                )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Representative trajectory: 90$^\\circ$ bend, rollout=5")
    ax.grid(True, alpha=0.3)

    if plotted_any:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_90deg_rollout5.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = read_runs(RUN_ROOT)

    if not runs:
        raise RuntimeError(f"No log.csv files found under {RUN_ROOT}")

    summary = build_summary(runs, OUT_DIR)

    print("\n[FIGURE 5 MODEL FIDELITY SUMMARY]")
    cols = [
        "bend_abs_deg",
        "solver_mode",
        "rollout_steps",
        "success",
        "final_i_ref",
        "max_pred_err_xy_mm",
        "max_rollout_err_xy_mm",
        "min_clearance_mm",
        "mean_complexity_proxy",
        "sum_complexity_proxy",
        "mean_sqp_iters_done",
        "sum_sqp_iters_done",
    ]
    print(summary[cols].to_string(index=False))

    # Main rollout/model-fidelity plots
    metrics_vs_rollout = [
        "max_pred_err_xy_mm",
        "mean_pred_err_xy_mm",
        "max_rollout_err_xy_mm",
        "terminal_rollout_err_xy_mm",
        "min_clearance_mm",
        "mean_clearance_mm",
        "final_i_ref",
        "mean_complexity_proxy",
        "sum_complexity_proxy",
        "control_rms",
        "mean_cond_H_beam",
        "mean_cond_H_mpc",
    ]

    for metric in metrics_vs_rollout:
        if metric in summary.columns:
            plot_metric_vs_rollout_by_solver(
                summary,
                metric,
                OUT_DIR / "metric_vs_rollout",
            )

    # Bar comparisons at rollout = 1, 3, 5
    for metric in [
        "max_pred_err_xy_mm",
        "max_rollout_err_xy_mm",
        "min_clearance_mm",
        "sum_complexity_proxy",
        "mean_complexity_proxy",
    ]:
        if metric in summary.columns:
            plot_metric_by_solver_grouped(
                summary,
                metric,
                OUT_DIR / "bar_by_solver_each_rollout",
            )

    # Pareto
    plot_pareto_accuracy_vs_complexity(
        summary,
        OUT_DIR / "pareto",
    )

    # Reference traces for representative hardest rollout
    trace_metrics = [
        "adapt_pred_err_xy_mm",
        "adapt_clearance_mm",
        "cond_H_beam",
        "cond_H_mpc",
        "sqp_iters_done",
        "N_sqp_used",
    ]

    for rollout in [1, 3, 5]:
        for metric in trace_metrics:
            plot_reference_traces_for_rollout(
                summary,
                runs,
                metric,
                rollout,
                OUT_DIR / "reference_traces",
            )

    # Optional trajectory plot
    plot_trajectory_if_available(
        runs,
        OUT_DIR / "trajectory",
    )

    print(f"\nSaved Figure 5 analysis to: {OUT_DIR}")