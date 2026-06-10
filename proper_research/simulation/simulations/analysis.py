import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    path = Path(path)

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)

def parse_bend_angle_from_run_name(run_name: str):
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan

    sign = -1.0 if m.group(1) == "m" else 1.0
    value = float(m.group(2))
    return sign * value


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
def collect_run_logs(run_root: Path):
    run_root = Path(run_root)
    runs = []

    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name

        df = pd.read_csv(log_csv)

        runs.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "log_csv": log_csv,
                "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
                "jacobian_variant": infer_jacobian_variant(run_name),
                "solver_mode": infer_solver_mode(run_name),
                "df": df,
            }
        )

    return runs

def summarize_log_csv(log_csv_path, *, bend_angle_deg=None, run_name=None):
    log_csv_path = Path(log_csv_path)
    df = pd.read_csv(log_csv_path)

    def safe_col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    reasons = df.get(
        "rollout_adaptation_reason",
        pd.Series("", index=df.index),
    ).fillna("")

    infeasible = safe_col("infeasible").fillna(0).astype(int)

    summary = {
        "run_name": run_name or log_csv_path.parent.name,
        "bend_angle_deg": bend_angle_deg,

        "n_steps": int(len(df)),
        "success": bool((infeasible == 0).all()),
        "num_infeasible_steps": int(infeasible.sum()),

        "final_status": str(df["status"].iloc[-1]) if "status" in df.columns else "",
        "final_i_ref": int(safe_col("i_ref").iloc[-1]),
        "max_i_ref": int(safe_col("i_ref").max()),

        "final_p_now_L_mm": 1e3 * float(safe_col("p_now_L").iloc[-1]),

        "mean_pred_err_xy_mm": float(safe_col("adapt_pred_err_xy_mm").mean()),
        "max_pred_err_xy_mm": float(safe_col("adapt_pred_err_xy_mm").max()),

        "min_clearance_mm": float(safe_col("adapt_clearance_mm").min()),
        "mean_clearance_mm": float(safe_col("adapt_clearance_mm").mean()),

        "max_tip_vessel_angle_deg": float(
            safe_col("adapt_tip_vessel_angle_deg").max()
        ),
        "mean_tip_vessel_angle_deg": float(
            safe_col("adapt_tip_vessel_angle_deg").mean()
        ),

        "mean_cond_H_beam": float(safe_col("cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_col("cond_H_beam").max()),

        "mean_cond_H_mpc": float(safe_col("cond_H_mpc").mean()),
        "max_cond_H_mpc": float(safe_col("cond_H_mpc").max()),

        "num_good_increase": int(
            reasons.str.contains("good:increase", regex=False).sum()
        ),
        "num_bad_tip_angle": int(
            reasons.str.contains("bad:tip_angle", regex=False).sum()
        ),
        "num_bad_low_clearance": int(
            reasons.str.contains("bad:low_clearance", regex=False).sum()
        ),

        "max_abs_u0_vx": float(safe_col("u0_vx").abs().max()),
        "max_abs_u0_vy": float(safe_col("u0_vy").abs().max()),
        "max_abs_u0_wz_rad_s": float(safe_col("u0_wz").abs().max()),
        "max_u0_dL": float(safe_col("u0_dL").max()),
    }

    return summary


def summarize_run_dir(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run_name = run_dir.name

    meta = load_json(run_dir / "experiment_config.json")
    run_summary = load_json(run_dir / "run_summary.json")

    bend_angle_deg = parse_bend_angle_from_run_name(run_name)

    log_summary = summarize_log_csv(
        run_dir / "log.csv",
        bend_angle_deg=bend_angle_deg,
        run_name=run_name,
    )

    row = {}

    row.update(meta)

    row.update(
        {
            f"runner_{k}": v
            for k, v in run_summary.items()
            if k not in row
        }
    )

    row.update(log_summary)

    row["run_dir"] = str(run_dir)
    row["run_name_from_dir"] = run_name
    row["jacobian_variant"] = infer_jacobian_variant(run_name)
    row["solver_mode"] = infer_solver_mode(run_name)

    return row


def summarize_run_root(
    run_root: Path,
    *,
    output_csv_name: str = "summary.csv",
    output_json_name: str = "summary.json",
) -> pd.DataFrame:
    run_root = Path(run_root)

    rows = []

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue

        if not (run_dir / "log.csv").exists():
            continue

        try:
            row = summarize_run_dir(run_dir)
            row["analysis_success"] = True
            row["analysis_error"] = ""
        except Exception as e:
            row = {
                "run_dir": str(run_dir),
                "run_name_from_dir": run_dir.name,
                "analysis_success": False,
                "analysis_error": repr(e),
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df.to_csv(run_root / output_csv_name, index=False)

        with open(run_root / output_json_name, "w") as f:
            json.dump(rows, f, indent=2)

    return df


def print_summary_table(df: pd.DataFrame):
    if df is None or len(df) == 0:
        print("[ANALYSIS] No runs found.")
        return

    preferred_cols = [
        "run_name",
        "bend_angle_deg",
        "jacobian_variant",
        "solver_mode",
        "n_steps",
        "final_i_ref",
        "num_infeasible_steps",
        "mean_pred_err_xy_mm",
        "max_pred_err_xy_mm",
        "mean_clearance_mm",
        "min_clearance_mm",
        "mean_tip_vessel_angle_deg",
        "max_tip_vessel_angle_deg",
        "mean_cond_H_beam",
        "max_cond_H_beam",
        "mean_cond_H_mpc",
        "max_cond_H_mpc",
    ]

    cols = [c for c in preferred_cols if c in df.columns]

    print(df[cols].to_string(index=False))
def plot_metric_one_figure_per_curvature_split_controller(
    run_root: Path,
    metric: str,
    *,
    x_col: str = "i_ref",
    out_dir: Path | None = None,
    solver_order=("lti", "ltv_oneshot", "sqp_full"),
    jacobian_order=("contact", "no_contact"),
    highlight_cond_col: str | None = None,
    highlight_threshold: float | None = None,
):
    runs = collect_run_logs(run_root)

    if out_dir is None:
        out_dir = Path(run_root) / "analysis_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    linestyle_map = {
        "contact": "-",
        "no_contact": "--",
    }

    marker_map = {
        "contact": "o",
        "no_contact": "s",
    }

    color_map = {
        "contact": "tab:blue",
        "no_contact": "tab:red",
    }

    pretty_solver = {
        "lti": "LTI",
        "ltv_oneshot": "LTV one-shot",
        "sqp_full": "SQP full",
    }

    pretty_jac = {
        "contact": "Contact-aware Jacobian",
        "no_contact": "Model-mismatch baseline",
    }

    bends = sorted(
        {r["bend_angle_deg"] for r in runs if np.isfinite(r["bend_angle_deg"])}
    )

    figs = {}

    for bend in bends:
        fig, axes = plt.subplots(
            1,
            len(solver_order),
            figsize=(14, 4.5),
            sharex=True,
            sharey=True,
        )

        if len(solver_order) == 1:
            axes = [axes]

        for ax, solver_mode in zip(axes, solver_order):
            for jacobian_variant in jacobian_order:
                matching = [
                    r for r in runs
                    if r["bend_angle_deg"] == bend
                    and r["solver_mode"] == solver_mode
                    and r["jacobian_variant"] == jacobian_variant
                ]

                if not matching:
                    continue

                run = matching[0]
                df = run["df"]

                if x_col not in df.columns or metric not in df.columns:
                    continue

                x = pd.to_numeric(df[x_col], errors="coerce")
                y = pd.to_numeric(df[metric], errors="coerce")

                valid = x.notna() & y.notna()

                ax.plot(
                    x[valid],
                    y[valid],
                    color=color_map.get(jacobian_variant, None),
                    linestyle=linestyle_map.get(jacobian_variant, "-"),
                    marker=marker_map.get(jacobian_variant, "o"),
                    linewidth=2.2,
                    markersize=4.5,
                    markerfacecolor="white" if jacobian_variant == "no_contact" else None,
                    markeredgewidth=1.2,
                    label=pretty_jac.get(jacobian_variant, jacobian_variant),
                )
                if (
                    highlight_cond_col is not None
                    and highlight_threshold is not None
                    and highlight_cond_col in df.columns
                ):
                    cond = pd.to_numeric(df[highlight_cond_col], errors="coerce")
                    high = (
                        valid
                        & cond.notna()
                        & np.isfinite(cond)
                        & (cond >= float(highlight_threshold))
                    )

                    if high.any():
                        ax.scatter(
                            x[high],
                            y[high],
                            s=80,
                            facecolors="none",
                            edgecolors="black",
                            linewidths=1.6,
                            zorder=10,
                        )
            if highlight_cond_col is not None and highlight_threshold is not None:
                ax.text(
                    0.02,
                    0.96,
                    f"black circles: {highlight_cond_col} ≥ {highlight_threshold:.1e}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )
            ax.set_title(pretty_solver.get(solver_mode, solver_mode))
            ax.set_xlabel("Reference index along trajectory")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(metric)
        axes[-1].legend(fontsize=9)

        fig.suptitle(f"{metric} vs reference index, bend {bend:.0f}°")
        fig.tight_layout()

        safe_metric = metric.replace("/", "_")
        bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"

        fig.savefig(
            out_dir / f"{safe_metric}_vs_{x_col}_bend_{bend_tag}_split_controller.png",
            dpi=200,
        )

        figs[bend] = (fig, axes)

    return figs

def plot_metric_vs_reference(
    log_csv_path,
    metric,
    *,
    x_col="i_ref",
    ax=None,
    label=None,
):
    log_csv_path = Path(log_csv_path)
    df = pd.read_csv(log_csv_path)

    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found in {log_csv_path}")

    if x_col not in df.columns:
        raise ValueError(f"x_col {x_col!r} not found in {log_csv_path}")

    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[metric], errors="coerce")

    valid = x.notna() & y.notna()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(
        x[valid],
        y[valid],
        marker="o",
        label=label or log_csv_path.parent.name,
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)

    if label is not None:
        ax.legend()

    fig.tight_layout()
    return fig, ax
def plot_metric_vs_reference_for_runs(
    run_root,
    metric,
    *,
    x_col="i_ref",
    run_filter=None,
    group_label_fn=None,
    save_path=None,
):
    run_root = Path(run_root)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_name = log_csv.parent.name

        if run_filter is not None and not run_filter(run_name):
            continue

        df = pd.read_csv(log_csv)

        if metric not in df.columns or x_col not in df.columns:
            continue

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[metric], errors="coerce")

        valid = x.notna() & y.notna()

        label = (
            group_label_fn(run_name)
            if group_label_fn is not None
            else run_name
        )

        ax.plot(
            x[valid],
            y[valid],
            marker="o",
            linewidth=1.5,
            label=label,
        )

    ax.set_xlabel("Reference index along trajectory")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)

    return fig, ax
def get_high_conditioning_mask(df, cond_col, *, threshold):
    if cond_col not in df.columns:
        return pd.Series(False, index=df.index)

    cond = pd.to_numeric(df[cond_col], errors="coerce")
    return cond.notna() & np.isfinite(cond) & (cond >= threshold)
def plot_conditioning_diagnostic_per_curvature_controller(
    run_root: Path,
    *,
    x_col: str = "i_ref",
    out_dir: Path | None = None,
    solver_order=("lti", "ltv_oneshot", "sqp_full"),
    jacobian_order=("contact", "no_contact"),
    beam_high: float = 5e3,
    mpc_high: float = 5e7,
):
    """
    One figure per curvature and controller.
    Each figure compares contact vs no_contact and stacks:
        cond_H_beam
        cond_H_mpc
        adapt_pred_err_xy_mm
        adapt_clearance_mm
        adapt_tip_vessel_angle_deg
        rollout_steps_used

    High conditioning regions are shaded.
    """
    runs = collect_run_logs(run_root)

    if out_dir is None:
        out_dir = Path(run_root) / "analysis_plots" / "conditioning_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("cond_H_beam", "Beam Hessian condition", "log", beam_high),
        ("cond_H_mpc", "MPC Hessian condition", "log", mpc_high),
        ("adapt_pred_err_xy_mm", "Prediction error XY [mm]", "linear", None),
        ("adapt_clearance_mm", "Clearance [mm]", "linear", None),
        ("adapt_tip_vessel_angle_deg", "Tip-vessel angle [deg]", "linear", None),
        ("rollout_steps_used", "Rollout used", "linear", None),
    ]

    color_map = {
        "contact": "tab:blue",
        "no_contact": "tab:red",
    }

    linestyle_map = {
        "contact": "-",
        "no_contact": "--",
    }

    marker_map = {
        "contact": "o",
        "no_contact": "s",
    }

    pretty_solver = {
        "lti": "LTI",
        "ltv_oneshot": "LTV one-shot",
        "sqp_full": "SQP full",
    }

    pretty_jac = {
        "contact": "Contact-aware",
        "no_contact": "No-contact baseline",
    }

    bends = sorted(
        {r["bend_angle_deg"] for r in runs if np.isfinite(r["bend_angle_deg"])}
    )

    figs = {}

    for bend in bends:
        for solver_mode in solver_order:
            fig, axes = plt.subplots(
                len(metrics),
                1,
                figsize=(10, 12),
                sharex=True,
            )

            plotted_any = False

            for jacobian_variant in jacobian_order:
                matching = [
                    r for r in runs
                    if r["bend_angle_deg"] == bend
                    and r["solver_mode"] == solver_mode
                    and r["jacobian_variant"] == jacobian_variant
                ]

                if not matching:
                    continue

                run = matching[0]
                df = run["df"]

                if x_col not in df.columns:
                    continue

                x = pd.to_numeric(df[x_col], errors="coerce")

                # Masks for shaded high-conditioning regions.
                beam_high_mask = (
                    pd.to_numeric(df.get("cond_H_beam", np.nan), errors="coerce")
                    >= beam_high
                )
                mpc_high_mask = (
                    pd.to_numeric(df.get("cond_H_mpc", np.nan), errors="coerce")
                    >= mpc_high
                )

                for ax, (metric, ylabel, scale, threshold) in zip(axes, metrics):
                    if metric not in df.columns:
                        continue

                    y = pd.to_numeric(df[metric], errors="coerce")
                    valid = x.notna() & y.notna() & np.isfinite(y)

                    if valid.sum() == 0:
                        continue
                    if metric in (
                        "adapt_pred_err_xy_mm",
                        "adapt_clearance_mm",
                        "adapt_tip_vessel_angle_deg",
                        "rollout_steps_used",
                    ):
                        if "cond_H_beam" in df.columns:
                            cond_beam = pd.to_numeric(df["cond_H_beam"], errors="coerce")
                            high_beam = x.notna() & cond_beam.notna() & (cond_beam >= beam_high)

                            for xv in x[high_beam]:
                                ax.axvline(
                                    xv,
                                    color="black",
                                    linestyle=":",
                                    linewidth=0.8,
                                    alpha=0.25,
                                )
                    if metric == "cond_H_beam":
                        high = valid & (y >= beam_high)
                        if high.any():
                            ax.scatter(
                                x[high],
                                y[high],
                                s=70,
                                facecolors="none",
                                edgecolors="black",
                                linewidths=1.4,
                                zorder=10,
                            )

                    if metric == "cond_H_mpc":
                        high = valid & (y >= mpc_high)
                        if high.any():
                            ax.scatter(
                                x[high],
                                y[high],
                                s=70,
                                facecolors="none",
                                edgecolors="black",
                                linewidths=1.4,
                                zorder=10,
                            )
                    if metric in (
                        "adapt_pred_err_xy_mm",
                        "adapt_clearance_mm",
                        "adapt_tip_vessel_angle_deg",
                        "rollout_steps_used",
                    ):
                        if "cond_H_beam" in df.columns:
                            cond_beam = pd.to_numeric(df["cond_H_beam"], errors="coerce")
                            high_beam = x.notna() & cond_beam.notna() & (cond_beam >= beam_high)

                            for xv in x[high_beam]:
                                ax.axvline(
                                    xv,
                                    color="black",
                                    linestyle=":",
                                    linewidth=0.8,
                                    alpha=0.25,
                                )
                    if metric == "rollout_steps_used":
                        ax.step(
                            x[valid],
                            y[valid],
                            where="post",
                            color=color_map.get(jacobian_variant, None),
                            linestyle=linestyle_map.get(jacobian_variant, "-"),
                            linewidth=2.0,
                            label=pretty_jac.get(jacobian_variant, jacobian_variant),
                        )
                    else:
                        ax.plot(
                            x[valid],
                            y[valid],
                            color=color_map.get(jacobian_variant, None),
                            linestyle=linestyle_map.get(jacobian_variant, "-"),
                            marker=marker_map.get(jacobian_variant, "o"),
                            markersize=3.5,
                            linewidth=1.8,
                            markerfacecolor=(
                                "white" if jacobian_variant == "no_contact" else None
                            ),
                            label=pretty_jac.get(jacobian_variant, jacobian_variant),
                        )

                    if scale == "log":
                        ax.set_yscale("log")

                    if threshold is not None:
                        ax.axhline(
                            threshold,
                            color="black",
                            linestyle=":",
                            linewidth=1.2,
                        )

                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)

                plotted_any = True

            # Shade high-conditioning regions based on union over visible runs.
            # This is a simple second pass to avoid clutter.
            for ax in axes:
                ax.set_xlim(auto=True)

            axes[-1].set_xlabel("Reference index along trajectory")

            if plotted_any:
                axes[0].legend(fontsize=9, loc="best")

            fig.suptitle(
                f"Conditioning diagnostics, bend {bend:.0f}°, "
                f"{pretty_solver.get(solver_mode, solver_mode)}",
                fontsize=13,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.97])

            bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"
            solver_tag = solver_mode.replace("/", "_")
            save_path = out_dir / f"conditioning_diag_bend_{bend_tag}_{solver_tag}.png"
            fig.savefig(save_path, dpi=220)

            figs[(bend, solver_mode)] = (fig, axes)

    return figs
def parse_rollout_steps_from_run_name(run_name: str):
    m = re.search(r"rollout(\d+)", run_name)
    if not m:
        return np.nan
    return int(m.group(1))
def collect_run_logs(run_root: Path):
    run_root = Path(run_root)
    runs = []

    for log_csv in sorted(run_root.glob("*/log.csv")):
        run_dir = log_csv.parent
        run_name = run_dir.name

        df = pd.read_csv(log_csv)

        runs.append(
            {
                "run_name": run_name,
                "run_dir": run_dir,
                "log_csv": log_csv,
                "bend_angle_deg": parse_bend_angle_from_run_name(run_name),
                "jacobian_variant": infer_jacobian_variant(run_name),
                "solver_mode": infer_solver_mode(run_name),
                "rollout_steps": parse_rollout_steps_from_run_name(run_name),
                "df": df,
            }
        )

    return runs
def build_grid_summary(run_root: Path) -> pd.DataFrame:
    runs = collect_run_logs(run_root)
    rows = []

    for run in runs:
        df = run["df"]

        def col(name):
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
            return pd.Series(np.nan, index=df.index)

        infeasible = col("infeasible").fillna(0).astype(int)

        row = {
            "run_name": run["run_name"],
            "run_dir": str(run["run_dir"]),
            "bend_angle_deg": run["bend_angle_deg"],
            "jacobian_variant": run["jacobian_variant"],
            "solver_mode": run["solver_mode"],
            "rollout_steps": run["rollout_steps"],

            "n_steps": int(len(df)),
            "success": bool((infeasible == 0).all()),
            "num_infeasible_steps": int(infeasible.sum()),
            "final_status": str(df["status"].iloc[-1]) if "status" in df.columns and len(df) else "",

            "final_i_ref": float(col("i_ref").iloc[-1]) if len(df) else np.nan,
            "max_i_ref": float(col("i_ref").max()),
            "final_p_now_L_mm": 1e3 * float(col("p_now_L").iloc[-1]) if len(df) else np.nan,

            "mean_pred_err_xy_mm": float(col("adapt_pred_err_xy_mm").mean()),
            "max_pred_err_xy_mm": float(col("adapt_pred_err_xy_mm").max()),

            "mean_clearance_mm": float(col("adapt_clearance_mm").mean()),
            "min_clearance_mm": float(col("adapt_clearance_mm").min()),

            "mean_tip_vessel_angle_deg": float(col("adapt_tip_vessel_angle_deg").mean()),
            "max_tip_vessel_angle_deg": float(col("adapt_tip_vessel_angle_deg").max()),

            "mean_cond_H_beam": float(col("cond_H_beam").mean()),
            "max_cond_H_beam": float(col("cond_H_beam").max()),

            "mean_cond_H_mpc": float(col("cond_H_mpc").mean()),
            "max_cond_H_mpc": float(col("cond_H_mpc").max()),

            "max_abs_u0_vx": float(col("u0_vx").abs().max()),
            "max_abs_u0_vy": float(col("u0_vy").abs().max()),
            "max_abs_u0_wz_rad_s": float(col("u0_wz").abs().max()),
            "max_u0_dL": float(col("u0_dL").max()),
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    if len(summary) > 0:
        summary = summary.sort_values(
            ["bend_angle_deg", "solver_mode", "jacobian_variant", "rollout_steps"]
        )

    out_path = Path(run_root) / "grid_summary_analysis.csv"
    summary.to_csv(out_path, index=False)
    print(f"[ANALYSIS] Wrote {out_path}")

    return summary
def plot_metric_vs_rollout_by_controller(
    summary: pd.DataFrame,
    metric: str,
    *,
    out_dir: Path,
    solver_order=("lti", "ltv_oneshot", "sqp_full"),
    jacobian_order=("contact", "no_contact"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretty_solver = {
        "lti": "LTI",
        "ltv_oneshot": "LTV one-shot",
        "sqp_full": "SQP full",
    }

    pretty_jac = {
        "contact": "Contact-aware",
        "no_contact": "Model-mismatch baseline",
    }

    color_map = {
        "contact": "tab:blue",
        "no_contact": "tab:red",
    }

    linestyle_map = {
        "contact": "-",
        "no_contact": "--",
    }

    bends = sorted(summary["bend_angle_deg"].dropna().unique())

    for solver_mode in solver_order:
        fig, axes = plt.subplots(
            1,
            len(bends),
            figsize=(5 * len(bends), 4),
            sharey=True,
        )

        if len(bends) == 1:
            axes = [axes]

        for ax, bend in zip(axes, bends):
            for jac in jacobian_order:
                g = summary[
                    (summary["bend_angle_deg"] == bend)
                    & (summary["solver_mode"] == solver_mode)
                    & (summary["jacobian_variant"] == jac)
                ].copy()

                if g.empty or metric not in g.columns:
                    continue

                g = g.sort_values("rollout_steps")

                ax.plot(
                    g["rollout_steps"],
                    g[metric],
                    marker="o",
                    linewidth=2.2,
                    linestyle=linestyle_map.get(jac, "-"),
                    color=color_map.get(jac, None),
                    label=pretty_jac.get(jac, jac),
                )

                # Mark failed runs.
                failed = ~g["success"].astype(bool)
                if failed.any():
                    ax.scatter(
                        g.loc[failed, "rollout_steps"],
                        g.loc[failed, metric],
                        marker="x",
                        s=90,
                        color="black",
                        linewidths=2,
                        zorder=10,
                    )

            ax.set_title(f"Bend {bend:.0f}°")
            ax.set_xlabel("Fixed rollout length")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(metric)
        axes[-1].legend(fontsize=9)

        fig.suptitle(
            f"{metric} vs rollout length — {pretty_solver.get(solver_mode, solver_mode)}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = out_dir / f"{metric}_vs_rollout_{solver_mode}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
def plot_metric_by_controller_for_each_rollout(
    summary: pd.DataFrame,
    metric: str,
    *,
    out_dir: Path,
    solver_order=("lti", "ltv_oneshot", "sqp_full"),
    jacobian_order=("contact", "no_contact"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretty_solver = {
        "lti": "LTI",
        "ltv_oneshot": "LTV one-shot",
        "sqp_full": "SQP full",
    }

    pretty_jac = {
        "contact": "Contact-aware",
        "no_contact": "Model-mismatch baseline",
    }

    bends = sorted(summary["bend_angle_deg"].dropna().unique())
    rollouts = sorted(summary["rollout_steps"].dropna().unique())

    x = np.arange(len(solver_order))
    width = 0.35

    for rollout in rollouts:
        fig, axes = plt.subplots(
            1,
            len(bends),
            figsize=(5 * len(bends), 4),
            sharey=True,
        )

        if len(bends) == 1:
            axes = [axes]

        for ax, bend in zip(axes, bends):
            for j, jac in enumerate(jacobian_order):
                values = []
                success_flags = []

                for solver in solver_order:
                    g = summary[
                        (summary["bend_angle_deg"] == bend)
                        & (summary["rollout_steps"] == rollout)
                        & (summary["solver_mode"] == solver)
                        & (summary["jacobian_variant"] == jac)
                    ]

                    if g.empty or metric not in g.columns:
                        values.append(np.nan)
                        success_flags.append(False)
                    else:
                        values.append(float(g[metric].iloc[0]))
                        success_flags.append(bool(g["success"].iloc[0]))

                offset = (j - 0.5) * width

                bars = ax.bar(
                    x + offset,
                    values,
                    width=width,
                    label=pretty_jac.get(jac, jac),
                    alpha=0.8,
                )

                # Mark failed bars with hatch.
                for bar, ok in zip(bars, success_flags):
                    if not ok:
                        bar.set_hatch("//")
                        bar.set_edgecolor("black")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [pretty_solver.get(s, s) for s in solver_order],
                rotation=20,
                ha="right",
            )
            ax.set_title(f"Bend {bend:.0f}°")
            ax.grid(True, axis="y", alpha=0.3)

        axes[0].set_ylabel(metric)
        axes[-1].legend(fontsize=9)

        fig.suptitle(f"{metric} by controller, fixed rollout = {int(rollout)}")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        save_path = out_dir / f"{metric}_by_controller_rollout_{int(rollout)}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
def plot_metric_reference_traces_by_rollout(
    run_root: Path,
    metric: str,
    *,
    rollout_steps: int,
    x_col: str = "i_ref",
    out_dir: Path | None = None,
    solver_order=("lti", "ltv_oneshot", "sqp_full"),
    jacobian_order=("contact", "no_contact"),
):
    runs = collect_run_logs(run_root)

    if out_dir is None:
        out_dir = Path(run_root) / "analysis_plots" / f"reference_traces_rollout_{rollout_steps}"
    out_dir.mkdir(parents=True, exist_ok=True)

    linestyle_map = {
        "contact": "-",
        "no_contact": "--",
    }

    marker_map = {
        "contact": "o",
        "no_contact": "s",
    }

    color_map = {
        "contact": "tab:blue",
        "no_contact": "tab:red",
    }

    pretty_solver = {
        "lti": "LTI",
        "ltv_oneshot": "LTV one-shot",
        "sqp_full": "SQP full",
    }

    pretty_jac = {
        "contact": "Contact-aware",
        "no_contact": "Model-mismatch baseline",
    }

    bends = sorted(
        {r["bend_angle_deg"] for r in runs if np.isfinite(r["bend_angle_deg"])}
    )

    for bend in bends:
        fig, axes = plt.subplots(
            1,
            len(solver_order),
            figsize=(14, 4.5),
            sharex=True,
            sharey=True,
        )

        if len(solver_order) == 1:
            axes = [axes]

        for ax, solver_mode in zip(axes, solver_order):
            for jacobian_variant in jacobian_order:
                matching = [
                    r for r in runs
                    if r["bend_angle_deg"] == bend
                    and r["solver_mode"] == solver_mode
                    and r["jacobian_variant"] == jacobian_variant
                    and int(r["rollout_steps"]) == int(rollout_steps)
                ]

                if not matching:
                    continue

                run = matching[0]
                df = run["df"]

                if x_col not in df.columns or metric not in df.columns:
                    continue

                x = pd.to_numeric(df[x_col], errors="coerce")
                y = pd.to_numeric(df[metric], errors="coerce")

                valid = x.notna() & y.notna() & np.isfinite(y)

                if metric in ("rollout_steps_used", "rollout_steps_next"):
                    ax.step(
                        x[valid],
                        y[valid],
                        where="post",
                        color=color_map.get(jacobian_variant, None),
                        linestyle=linestyle_map.get(jacobian_variant, "-"),
                        linewidth=2.2,
                        label=pretty_jac.get(jacobian_variant, jacobian_variant),
                    )
                else:
                    ax.plot(
                        x[valid],
                        y[valid],
                        color=color_map.get(jacobian_variant, None),
                        linestyle=linestyle_map.get(jacobian_variant, "-"),
                        marker=marker_map.get(jacobian_variant, "o"),
                        linewidth=2.0,
                        markersize=3.8,
                        markerfacecolor="white" if jacobian_variant == "no_contact" else None,
                        label=pretty_jac.get(jacobian_variant, jacobian_variant),
                    )

                # Mark infeasible points.
                if "infeasible" in df.columns:
                    infeas = pd.to_numeric(df["infeasible"], errors="coerce").fillna(0).astype(int) > 0
                    if infeas.any():
                        ax.scatter(
                            x[infeas],
                            y[infeas],
                            marker="x",
                            s=90,
                            color="black",
                            linewidths=2.0,
                            zorder=10,
                        )

            ax.set_title(pretty_solver.get(solver_mode, solver_mode))
            ax.set_xlabel("Reference index along trajectory")
            ax.grid(True, alpha=0.3)

            if metric in ("cond_H_beam", "cond_H_mpc"):
                ax.set_yscale("log")

        axes[0].set_ylabel(metric)
        axes[-1].legend(fontsize=9)

        fig.suptitle(
            f"{metric} vs reference index, bend {bend:.0f}°, rollout={rollout_steps}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        safe_metric = metric.replace("/", "_")
        bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"

        save_path = out_dir / f"{safe_metric}_bend_{bend_tag}_rollout_{rollout_steps}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
if __name__ == "__main__":
    run_root = Path("run_experiment_4")

    metrics = [
        "max_pred_err_xy_mm",
        "mean_pred_err_xy_mm",
        "min_clearance_mm",
        "max_tip_vessel_angle_deg",
        "final_i_ref",
        "max_cond_H_beam",
        "max_cond_H_mpc",
        "num_infeasible_steps",
    ]
    summary = build_grid_summary(run_root)
    print(summary.to_string(index=False))
    for metric in metrics:
        plot_metric_vs_rollout_by_controller(
            summary,
            metric,
            out_dir=run_root / "analysis_plots" / "rollout_effects",
        )
    for metric in metrics:
        plot_metric_by_controller_for_each_rollout(
            summary,
            metric,
            out_dir=run_root / "analysis_plots" / "controller_comparison",
        )
    trace_metrics = [
        "adapt_pred_err_xy_mm",
        "adapt_clearance_mm",
        "adapt_tip_vessel_angle_deg",
        "cond_H_beam",
        "cond_H_mpc",
        "rollout_steps_used"
    ]

    for rollout in (3, 5, 10):
        for metric in trace_metrics:
            plot_metric_reference_traces_by_rollout(
                run_root,
                metric,
                rollout_steps=rollout,
                x_col="i_ref",
            )
    plt.show()