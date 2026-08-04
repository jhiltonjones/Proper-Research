from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def parse_bend_angle_from_run_name(run_name: str):
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


def infer_jacobian_variant(run_name: str):
    if "jac_no_contact" in run_name:
        return "no_contact"
    if "jac_contact" in run_name:
        return "contact"
    return ""


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


def numeric_col(df, name):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------------
# Per-run summary
# ---------------------------------------------------------------------
def summarise_single_run(run_dir: Path):
    run_dir = Path(run_dir)
    run_name = run_dir.name
    log_csv = run_dir / "log.csv"

    df = pd.read_csv(log_csv)

    rollout_steps = parse_rollout_steps_from_run_name(run_name)

    infeasible = numeric_col(df, "infeasible").fillna(0).astype(int)

    # ------------------------------------------------------------
    # Rollout prediction error columns
    # ------------------------------------------------------------
    rollout_err_cols = [
        f"rollout_err_xy_{i}_mm"
        for i in range(int(rollout_steps))
        if f"rollout_err_xy_{i}_mm" in df.columns
    ]

    if rollout_err_cols:
        rollout_err_matrix = df[rollout_err_cols].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy()

        mean_rollout_pred_err_xy_mm = float(np.nanmean(rollout_err_matrix))
        max_rollout_pred_err_xy_mm = float(np.nanmax(rollout_err_matrix))

        terminal_col = rollout_err_cols[-1]
        terminal_rollout_pred_err_xy_mm = float(
            numeric_col(df, terminal_col).mean()
        )
        max_terminal_rollout_pred_err_xy_mm = float(
            numeric_col(df, terminal_col).max()
        )
    else:
        mean_rollout_pred_err_xy_mm = np.nan
        max_rollout_pred_err_xy_mm = np.nan
        terminal_rollout_pred_err_xy_mm = np.nan
        max_terminal_rollout_pred_err_xy_mm = np.nan

    # ------------------------------------------------------------
    # Control effort
    # ------------------------------------------------------------
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
        "run_name": run_name,
        "run_dir": str(run_dir),

        # Experiment factors
        "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
        "bend_abs_deg": abs(parse_bend_angle_from_run_name(run_name)),
        "jacobian_variant": infer_jacobian_variant(run_name),
        "solver_mode": infer_solver_mode(run_name),
        "rollout_steps": rollout_steps,
        "Np": parse_Np_from_run_name(run_name),

        # Status / progress
        "n_steps": int(len(df)),
        "success": bool((infeasible == 0).all()),
        "num_infeasible_steps": int(infeasible.sum()),
        "final_status": str(df["status"].iloc[-1]) if "status" in df.columns else "",
        "final_i_ref": float(numeric_col(df, "i_ref").iloc[-1]),
        "max_i_ref": float(numeric_col(df, "i_ref").max()),

        # Path tracking / safety
        "mean_rho_mm": float(numeric_col(df, "rho_mm").mean()),
        "max_rho_mm": float(numeric_col(df, "rho_mm").max()),
        "mean_clearance_mm": float(numeric_col(df, "clearance_mm").mean()),
        "min_clearance_mm": float(numeric_col(df, "clearance_mm").min()),
        "mean_tip_vessel_angle_deg": float(
            numeric_col(df, "tip_vessel_angle_deg").mean()
        ),
        "max_tip_vessel_angle_deg": float(
            numeric_col(df, "tip_vessel_angle_deg").max()
        ),

        # One-step / logged prediction error
        "mean_adapt_pred_err_xy_mm": float(
            numeric_col(df, "adapt_pred_err_xy_mm").mean()
        ),
        "max_adapt_pred_err_xy_mm": float(
            numeric_col(df, "adapt_pred_err_xy_mm").max()
        ),

        # Rollout prediction error
        "mean_rollout_pred_err_xy_mm": mean_rollout_pred_err_xy_mm,
        "max_rollout_pred_err_xy_mm": max_rollout_pred_err_xy_mm,
        "terminal_rollout_pred_err_xy_mm": terminal_rollout_pred_err_xy_mm,
        "max_terminal_rollout_pred_err_xy_mm": max_terminal_rollout_pred_err_xy_mm,

        # Conditioning
        "mean_cond_H_beam": float(numeric_col(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(numeric_col(df, "cond_H_beam").max()),
        "mean_cond_H_mpc": float(numeric_col(df, "cond_H_mpc").mean()),
        "max_cond_H_mpc": float(numeric_col(df, "cond_H_mpc").max()),

        # Control effort
        "control_rms": control_rms,
        "control_max_abs": control_max_abs,
    }

    return row


def build_figure4_summary(run_root: Path):
    run_root = Path(run_root)

    rows = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "log.csv").exists():
            continue

        try:
            rows.append(summarise_single_run(run_dir))
        except Exception as exc:
            rows.append({
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "analysis_error": repr(exc),
            })

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["bend_abs_deg", "jacobian_variant", "rollout_steps"],
        na_position="last",
    )

    summary.to_csv(run_root / "figure4_summary_analysis.csv", index=False)

    with open(run_root / "figure4_summary_analysis.json", "w") as f:
        json.dump(
            summary.replace({np.nan: None}).to_dict(orient="records"),
            f,
            indent=2,
        )

    return summary


# ---------------------------------------------------------------------
# Contact benefit table
# ---------------------------------------------------------------------
def build_contact_benefit_table(summary: pd.DataFrame, metric: str):
    """
    Builds a table comparing no-contact minus contact for each bend/rollout.
    Positive delta means contact-aware performed better for error-like metrics.
    """
    key_cols = ["bend_abs_deg", "rollout_steps"]

    contact = summary[
        summary["jacobian_variant"] == "contact"
    ][key_cols + [metric]].rename(columns={metric: f"{metric}_contact"})

    no_contact = summary[
        summary["jacobian_variant"] == "no_contact"
    ][key_cols + [metric]].rename(columns={metric: f"{metric}_no_contact"})

    merged = pd.merge(contact, no_contact, on=key_cols, how="inner")

    merged[f"{metric}_delta_no_contact_minus_contact"] = (
        merged[f"{metric}_no_contact"] - merged[f"{metric}_contact"]
    )

    merged[f"{metric}_ratio_no_contact_over_contact"] = (
        merged[f"{metric}_no_contact"]
        / (merged[f"{metric}_contact"] + 1e-9)
    )

    return merged.sort_values(key_cols)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_metric_vs_rollout(summary, metric, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        for jac, label, linestyle, marker in [
            ("contact", "Contact-aware Jacobian", "-", "o"),
            ("no_contact", "No-contact Jacobian", "--", "s"),
        ]:
            g = summary[
                (summary["bend_abs_deg"] == bend)
                & (summary["jacobian_variant"] == jac)
            ].copy()

            if g.empty:
                continue

            g = g.sort_values("rollout_steps")

            ax.plot(
                g["rollout_steps"],
                g[metric],
                linestyle=linestyle,
                marker=marker,
                linewidth=2.2,
                markersize=6,
                label=label,
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
                    label="Failed" if jac == "contact" else None,
                )

        ax.set_title(f"{int(bend)}$^\\circ$ bend")
        ax.set_xlabel("Execution rollout length")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(summary["rollout_steps"].dropna().unique()))

    axes[0].set_ylabel(metric.replace("_", " "))
    axes[-1].legend(fontsize=8)

    fig.suptitle(metric.replace("_", " ") + " vs execution rollout")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    save_path = out_dir / f"{metric}_vs_rollout_contact_vs_no_contact.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path


def plot_contact_benefit(summary, metric, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benefit = build_contact_benefit_table(summary, metric)

    bends = sorted(benefit["bend_abs_deg"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(5.0 * len(bends), 4.0),
        sharey=True,
    )

    if len(bends) == 1:
        axes = [axes]

    delta_col = f"{metric}_delta_no_contact_minus_contact"

    for ax, bend in zip(axes, bends):
        g = benefit[benefit["bend_abs_deg"] == bend].copy()
        g = g.sort_values("rollout_steps")

        ax.axhline(0.0, linestyle=":", linewidth=1.2, color="black")

        ax.bar(
            g["rollout_steps"].astype(str),
            g[delta_col],
        )

        ax.set_title(f"{int(bend)}$^\\circ$ bend")
        ax.set_xlabel("Execution rollout length")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("No-contact minus contact")

    fig.suptitle(
        f"Contact-aware benefit for {metric}\n"
        "Positive values mean contact-aware is better"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    save_path = out_dir / f"{metric}_contact_benefit.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return save_path


def plot_rollout_error_by_stage(run_root: Path, out_dir: Path):
    """
    Shows how prediction error accumulates within the rollout:
    h = 1, 2, ..., rollout_steps.
    """
    run_root = Path(run_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "log.csv").exists():
            continue

        run_name = run_dir.name
        df = pd.read_csv(run_dir / "log.csv")

        rollout_steps = parse_rollout_steps_from_run_name(run_name)

        for h in range(int(rollout_steps)):
            col = f"rollout_err_xy_{h}_mm"
            if col not in df.columns:
                continue

            values = pd.to_numeric(df[col], errors="coerce")

            rows.append({
                "bend_abs_deg": abs(parse_bend_angle_from_run_name(run_name)),
                "jacobian_variant": infer_jacobian_variant(run_name),
                "rollout_steps": rollout_steps,
                "stage": h + 1,
                "mean_err_mm": float(values.mean()),
                "max_err_mm": float(values.max()),
            })

    stage_df = pd.DataFrame(rows)
    stage_df.to_csv(out_dir / "rollout_error_by_stage.csv", index=False)

    for bend in sorted(stage_df["bend_abs_deg"].dropna().unique()):
        fig, axes = plt.subplots(
            1,
            5,
            figsize=(17, 3.6),
            sharey=True,
        )

        for ax, rollout in zip(axes, [1, 5,10]):
            for jac, label, linestyle, marker in [
                ("contact", "Contact-aware", "-", "o"),
                ("no_contact", "No-contact", "--", "s"),
            ]:
                g = stage_df[
                    (stage_df["bend_abs_deg"] == bend)
                    & (stage_df["rollout_steps"] == rollout)
                    & (stage_df["jacobian_variant"] == jac)
                ].copy()

                if g.empty:
                    continue

                g = g.sort_values("stage")

                ax.plot(
                    g["stage"],
                    g["mean_err_mm"],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.0,
                    label=label,
                )

            ax.set_title(f"r={rollout}")
            ax.set_xlabel("Step inside rollout")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Mean rollout prediction error [mm]")
        axes[-1].legend(fontsize=8)

        fig.suptitle(f"Prediction error accumulation, {int(bend)}$^\\circ$ bend")
        fig.tight_layout(rect=[0, 0, 1, 0.9])

        save_path = out_dir / f"rollout_error_by_stage_bend_{int(bend)}.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    return stage_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_root = Path("run_figure4_simulation_indices40")
    out_dir = run_root / "figure4_analysis_plots_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_figure4_summary(run_root)

    print("\nSummary:")
    print(
        summary[
            [
                "bend_abs_deg",
                "jacobian_variant",
                "rollout_steps",
                "success",
                "final_i_ref",
                "mean_rho_mm",
                "max_rho_mm",
                "min_clearance_mm",
                "terminal_rollout_pred_err_xy_mm",
                "max_rollout_pred_err_xy_mm",
            ]
        ].to_string(index=False)
    )

    # Main paper plots
    main_metrics = [
        "mean_rho_mm",
        "max_rho_mm",
        "min_clearance_mm",
        "terminal_rollout_pred_err_xy_mm",
        "max_rollout_pred_err_xy_mm",
        "mean_adapt_pred_err_xy_mm",
        "final_i_ref",
        "control_rms",
    ]

    for metric in main_metrics:
        if metric in summary.columns:
            plot_metric_vs_rollout(summary, metric, out_dir)
            plot_contact_benefit(summary, metric, out_dir)

            benefit = build_contact_benefit_table(summary, metric)
            benefit.to_csv(
                out_dir / f"{metric}_contact_benefit_table.csv",
                index=False,
            )

    # Plot prediction error accumulation inside each rollout
    plot_rollout_error_by_stage(run_root, out_dir / "rollout_error_by_stage")

    print(f"\nSaved plots to: {out_dir}")