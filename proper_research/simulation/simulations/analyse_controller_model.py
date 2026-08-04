from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------
RUN_ROOT = Path("run_figure6_simulation2")
OUT_DIR = RUN_ROOT / "figure6_hierarchy_analysis2"


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------
def parse_bend_angle_from_run_name(run_name: str):
    m = re.search(r"bend_([mp])(\d+)", run_name)
    if not m:
        return np.nan
    sign = -1.0 if m.group(1) == "m" else 1.0
    return sign * float(m.group(2))


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
                "df": df,
            }
        )

    return runs


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------
def fraction_equal(series, value):
    s = series.fillna("").astype(str)
    if len(s) == 0:
        return np.nan
    return float((s == value).mean())


def count_contains(series, substring):
    s = series.fillna("").astype(str)
    if len(s) == 0:
        return 0
    return int(s.str.contains(substring, regex=False).sum())


def count_switches(series):
    s = series.dropna()
    if len(s) <= 1:
        return 0
    return int(s.ne(s.shift()).sum() - 1)


def summarise_run(run):
    df = run["df"]

    status = safe_str(df, "status")
    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)

    failure_mask = (
        (infeasible > 0)
        | (~status.isin(["solved", "solved_inaccurate", "solved inaccurate"]))
    )

    solver_used = safe_str(df, "solver_mode_used")
    if (solver_used == "").all():
        solver_used = safe_str(df, "solver_mode")

    hierarchy_reason = safe_str(df, "hierarchy_reason")
    rollout_reason = safe_str(df, "rollout_adaptation_reason")

    rollout_used = safe_num(df, "rollout_steps_used")
    rollout_next = safe_num(df, "rollout_steps_next")
    hierarchy_level = safe_num(df, "hierarchy_level")
    hierarchy_confidence = safe_num(df, "hierarchy_confidence")
    hierarchy_risk = safe_num(df, "hierarchy_risk")

    sqp_iters = safe_num(df, "sqp_iters_done")
    N_sqp_used = safe_num(df, "N_sqp_used")

    # Rollout prediction error columns.
    rollout_err_cols = [
        c for c in df.columns
        if re.match(r"rollout_err_xy_\d+_mm", c)
    ]

    if rollout_err_cols:
        E = df[rollout_err_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        mean_rollout_err = float(np.nanmean(E))
        max_rollout_err = float(np.nanmax(E))
    else:
        mean_rollout_err = np.nan
        max_rollout_err = np.nan

    row = {
        "run_name": run["run_name"],
        "run_dir": str(run["run_dir"]),
        "bend_angle_deg": run["bend_angle_deg"],
        "bend_abs_deg": run["bend_abs_deg"],

        "n_steps": int(len(df)),
        "success": bool((failure_mask == 0).all()) if len(df) else False,
        "num_failed_steps": int(failure_mask.sum()) if len(df) else 0,
        "num_infeasible_steps": int(infeasible.sum()) if len(df) else 0,
        "final_status": str(status.iloc[-1]) if len(df) else "",

        "final_i_ref": float(safe_num(df, "i_ref").iloc[-1]) if len(df) else np.nan,
        "max_i_ref": float(safe_num(df, "i_ref").max()),

        # Performance
        "mean_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").mean()),
        "max_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").max()),
        "mean_rollout_err_xy_mm": mean_rollout_err,
        "max_rollout_err_xy_mm": max_rollout_err,

        "mean_clearance_mm": float(safe_num(df, "adapt_clearance_mm").mean()),
        "min_clearance_mm": float(safe_num(df, "adapt_clearance_mm").min()),
        "mean_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").mean()),
        "max_tip_vessel_angle_deg": float(safe_num(df, "adapt_tip_vessel_angle_deg").max()),

        # Trigger metrics
        "mean_curvature_max_ahead_1pm": float(safe_num(df, "curvature_max_ahead_1pm").mean()),
        "max_curvature_max_ahead_1pm": float(safe_num(df, "curvature_max_ahead_1pm").max()),

        "mean_cond_H_beam": float(safe_num(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_num(df, "cond_H_beam").max()),
        "mean_cond_H_mpc": float(safe_num(df, "cond_H_mpc").mean()),
        "max_cond_H_mpc": float(safe_num(df, "cond_H_mpc").max()),

        "mean_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").mean()),
        "max_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").max()),

        # Hierarchical decisions
        "mean_rollout_steps_used": float(rollout_used.mean()),
        "min_rollout_steps_used": float(rollout_used.min()),
        "max_rollout_steps_used": float(rollout_used.max()),
        "unique_rollout_steps_used": ",".join(
            str(int(v)) for v in sorted(rollout_used.dropna().unique())
        ),

        "mean_hierarchy_level": float(hierarchy_level.mean()),
        "max_hierarchy_level": float(hierarchy_level.max()),
        "mean_hierarchy_confidence": float(hierarchy_confidence.mean()),
        "min_hierarchy_confidence": float(hierarchy_confidence.min()),
        "mean_hierarchy_risk": float(hierarchy_risk.mean()),
        "max_hierarchy_risk": float(hierarchy_risk.max()),

        "frac_lti": fraction_equal(solver_used, "lti"),
        "frac_ltv_oneshot": fraction_equal(solver_used, "ltv_oneshot"),
        "frac_sqp_full": fraction_equal(solver_used, "sqp_full"),

        "unique_solver_modes": ",".join(sorted(solver_used.dropna().unique())),
        "num_solver_switches": count_switches(solver_used),
        "num_rollout_switches": count_switches(rollout_used),
        "num_level_switches": count_switches(hierarchy_level),

        # Reason counts
        "num_reason_easy": count_contains(hierarchy_reason, "easy"),
        "num_reason_moderate": count_contains(hierarchy_reason, "moderate"),
        "num_reason_hard": count_contains(hierarchy_reason, "hard"),
        "num_reason_clearance": count_contains(hierarchy_reason, "clearance"),
        "num_reason_curvature": count_contains(hierarchy_reason, "curvature"),
        "num_reason_beam_cond": count_contains(hierarchy_reason, "beam_cond"),
        "num_reason_mpc_cond": count_contains(hierarchy_reason, "mpc_cond"),
        "num_reason_sqp_du_rel": count_contains(hierarchy_reason, "sqp_du_rel"),
        "num_reason_escalate": count_contains(hierarchy_reason, "escalate"),
        "num_reason_downgrade_held": count_contains(hierarchy_reason, "downgrade_held"),
        "num_reason_downgrade_allowed": count_contains(hierarchy_reason, "downgrade_allowed"),

        # Rollout adaptation reason counts
        "num_rollout_good_increase": count_contains(rollout_reason, "good:increase"),
        "num_rollout_bad_tip_angle": count_contains(rollout_reason, "bad:tip_angle"),
        "num_rollout_bad_low_clearance": count_contains(rollout_reason, "bad:low_clearance"),

        # Complexity
        "mean_sqp_iters_done": float(sqp_iters.mean()),
        "max_sqp_iters_done": float(sqp_iters.max()),
        "sum_sqp_iters_done": float(sqp_iters.sum()),
        "mean_N_sqp_used": float(N_sqp_used.mean()),
        "max_N_sqp_used": float(N_sqp_used.max()),
        "sum_N_sqp_used": float(N_sqp_used.sum()),
    }

    return row


def build_summary(runs, out_dir):
    rows = [summarise_run(r) for r in runs]
    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values("bend_abs_deg")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "figure6_hierarchy_summary.csv", index=False)

    with open(out_dir / "figure6_hierarchy_summary.json", "w") as f:
        json.dump(
            summary.replace({np.nan: None}).to_dict(orient="records"),
            f,
            indent=2,
        )

    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_solver_timeline(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_to_level = {
        "lti": 0,
        "ltv_oneshot": 1,
        "sqp_full": 2,
    }

    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(9.0, 2.8 * len(runs)),
        sharex=False,
    )

    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        df = run["df"]
        x = safe_num(df, "i_ref")

        solver = safe_str(df, "solver_mode_used")
        if (solver == "").all():
            solver = safe_str(df, "solver_mode")

        y = solver.map(mode_to_level)

        valid = x.notna() & y.notna()

        ax.step(
            x[valid],
            y[valid],
            where="post",
            linewidth=2.2,
        )

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["LTI", "1-shot", "full SQP"])
        ax.set_title(f"{int(run['bend_abs_deg'])}$^\\circ$ bend")
        ax.set_ylabel("Model")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Reference index along vessel")

    fig.suptitle("Hierarchical model-fidelity selection")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "solver_mode_timeline_all_bends.png", dpi=300)
    plt.close(fig)


def plot_rollout_timeline(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(9.0, 2.8 * len(runs)),
        sharex=False,
    )

    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        df = run["df"]
        x = safe_num(df, "i_ref")
        y = safe_num(df, "rollout_steps_used")
        y_next = safe_num(df, "rollout_steps_next")

        valid = x.notna() & y.notna()
        ax.step(
            x[valid],
            y[valid],
            where="post",
            linewidth=2.2,
            label="used",
        )

        valid_next = x.notna() & y_next.notna()
        if valid_next.any():
            ax.step(
                x[valid_next],
                y_next[valid_next],
                where="post",
                linewidth=1.6,
                linestyle="--",
                label="next",
            )

        ax.set_title(f"{int(run['bend_abs_deg'])}$^\\circ$ bend")
        ax.set_ylabel("Rollout")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Reference index along vessel")

    fig.suptitle("Self-triggered rollout selection")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "rollout_timeline_all_bends.png", dpi=300)
    plt.close(fig)


def normalise_series(s):
    s = pd.to_numeric(s, errors="coerce")
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return s * np.nan
    lo = finite.min()
    hi = finite.max()
    if abs(hi - lo) < 1e-12:
        return 0.0 * s
    return (s - lo) / (hi - lo)


def plot_trigger_dashboard_per_bend(runs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("adapt_clearance_mm", "Clearance", False, True),  # invert: lower clearance = higher risk
        ("curvature_max_ahead_1pm", "Curvature ahead", False, False),
        ("cond_H_beam", "Beam cond.", True, False),
        ("cond_H_mpc", "MPC cond.", True, False),
        ("sqp_du_rel_final", "SQP input change", True, False),
    ]

    for run in runs:
        df = run["df"]
        x = safe_num(df, "i_ref")

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(9.0, 9.0),
            sharex=True,
        )

        # Solver mode
        solver = safe_str(df, "solver_mode_used")
        if (solver == "").all():
            solver = safe_str(df, "solver_mode")

        mode_to_level = {"lti": 0, "ltv_oneshot": 1, "sqp_full": 2}
        solver_level = solver.map(mode_to_level)

        valid = x.notna() & solver_level.notna()
        axes[0].step(x[valid], solver_level[valid], where="post", linewidth=2.0)
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(["LTI", "1-shot", "SQP"])
        axes[0].set_ylabel("Model")
        axes[0].grid(True, alpha=0.3)

        # Rollout
        y_roll = safe_num(df, "rollout_steps_used")
        valid = x.notna() & y_roll.notna()
        axes[1].step(x[valid], y_roll[valid], where="post", linewidth=2.0)
        axes[1].set_ylabel("Rollout")
        axes[1].grid(True, alpha=0.3)

        # Normalised risk metrics
        for col, label, use_log, invert in metrics:
            if col not in df.columns:
                continue

            y = safe_num(df, col)
            if use_log:
                y = np.log10(np.maximum(y, 1e-12))

            yn = normalise_series(y)
            if invert:
                yn = 1.0 - yn

            valid = x.notna() & yn.notna() & np.isfinite(yn)
            axes[2].plot(
                x[valid],
                yn[valid],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=label,
            )

        axes[2].set_ylabel("Normalised risk")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8, ncol=2)

        # Prediction error / clearance actual
        pred = safe_num(df, "adapt_pred_err_xy_mm")
        clear = safe_num(df, "adapt_clearance_mm")

        valid_pred = x.notna() & pred.notna()
        axes[3].plot(
            x[valid_pred],
            pred[valid_pred],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label="Prediction error [mm]",
        )

        ax2 = axes[3].twinx()
        valid_clear = x.notna() & clear.notna()
        ax2.plot(
            x[valid_clear],
            clear[valid_clear],
            marker="s",
            markersize=3,
            linewidth=1.5,
            linestyle="--",
            label="Clearance [mm]",
        )

        axes[3].set_ylabel("Pred. err. [mm]")
        ax2.set_ylabel("Clearance [mm]")
        axes[3].grid(True, alpha=0.3)

        lines1, labels1 = axes[3].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[3].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        axes[-1].set_xlabel("Reference index along vessel")

        fig.suptitle(
            f"Hierarchical controller behaviour, {int(run['bend_abs_deg'])}$^\\circ$ bend"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(
            out_dir / f"trigger_dashboard_bend_{bend_tag(run['bend_angle_deg'])}.png",
            dpi=300,
        )
        plt.close(fig)


def plot_reason_counts(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    reason_cols = [
        ("num_reason_easy", "easy"),
        ("num_reason_moderate", "moderate"),
        ("num_reason_hard", "hard"),
        ("num_reason_clearance", "clearance"),
        ("num_reason_curvature", "curvature"),
        ("num_reason_beam_cond", "beam cond."),
        ("num_reason_mpc_cond", "MPC cond."),
        ("num_reason_sqp_du_rel", "input change"),
    ]

    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    x = np.arange(len(reason_cols))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, bend in enumerate(bends):
        g = summary[summary["bend_abs_deg"] == bend]
        if g.empty:
            continue

        vals = [float(g[col].iloc[0]) for col, _ in reason_cols]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=f"{int(bend)}$^\\circ$",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in reason_cols], rotation=25, ha="right")
    ax.set_ylabel("Number of MPC updates")
    ax.set_title("Hierarchy decision reasons")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "hierarchy_reason_counts.png", dpi=300)
    plt.close(fig)


def plot_solver_usage(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    usage_cols = [
        ("frac_lti", "LTI"),
        ("frac_ltv_oneshot", "1-shot"),
        ("frac_sqp_full", "full SQP"),
    ]

    x = np.arange(len(bends))
    bottom = np.zeros(len(bends))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for col, label in usage_cols:
        vals = []
        for bend in bends:
            g = summary[summary["bend_abs_deg"] == bend]
            vals.append(float(g[col].iloc[0]) if not g.empty else 0.0)

        vals = np.asarray(vals)
        ax.bar(x, vals, bottom=bottom, label=label)
        bottom += vals

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}$^\\circ$" for b in bends])
    ax.set_ylabel("Fraction of MPC updates")
    ax.set_title("Model-fidelity usage by vessel curvature")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "solver_usage_by_bend.png", dpi=300)
    plt.close(fig)


def plot_summary_bars(summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("max_pred_err_xy_mm", "Max prediction error [mm]"),
        ("mean_rollout_err_xy_mm", "Mean rollout error [mm]"),
        ("min_clearance_mm", "Minimum clearance [mm]"),
        ("mean_rollout_steps_used", "Mean rollout used"),
        ("sum_N_sqp_used", "Total SQP complexity proxy"),
        ("num_solver_switches", "Solver switches"),
        ("num_rollout_switches", "Rollout switches"),
    ]

    bends = sorted(summary["bend_abs_deg"].dropna().unique())

    for metric, ylabel in metrics:
        if metric not in summary.columns:
            continue

        vals = []
        for bend in bends:
            g = summary[summary["bend_abs_deg"] == bend]
            vals.append(float(g[metric].iloc[0]) if not g.empty else np.nan)

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.bar([f"{int(b)}$^\\circ$" for b in bends], vals)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by bend")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_by_bend.png", dpi=300)
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

    print("\n[FIGURE 6 HIERARCHY SUMMARY]")
    print(
        summary[
            [
                "bend_abs_deg",
                "success",
                "final_i_ref",
                "max_pred_err_xy_mm",
                "min_clearance_mm",
                "mean_rollout_steps_used",
                "unique_rollout_steps_used",
                "frac_lti",
                "frac_ltv_oneshot",
                "frac_sqp_full",
                "unique_solver_modes",
                "num_solver_switches",
                "num_rollout_switches",
                "num_reason_easy",
                "num_reason_curvature",
                "num_reason_clearance",
                "num_reason_beam_cond",
                "num_reason_mpc_cond",
                "num_reason_sqp_du_rel",
            ]
        ].to_string(index=False)
    )

    plot_solver_timeline(runs, OUT_DIR / "timelines")
    plot_rollout_timeline(runs, OUT_DIR / "timelines")
    plot_trigger_dashboard_per_bend(runs, OUT_DIR / "trigger_dashboards")
    plot_reason_counts(summary, OUT_DIR / "summary_plots")
    plot_solver_usage(summary, OUT_DIR / "summary_plots")
    plot_summary_bars(summary, OUT_DIR / "summary_plots")

    print(f"\nSaved analysis to: {OUT_DIR}")