from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collect_runs_from_roots(roots: dict[str, Path]) -> list[dict]:
    runs = []

    for controller, root in roots.items():
        root = Path(root)

        if not root.exists():
            print(f"[WARN] Missing root: {root}")
            continue

        log_paths = sorted(root.rglob("log.csv"))

        if not log_paths:
            print(f"[WARN] No log.csv files found under: {root}")
            continue

        for log_csv in log_paths:
            try:
                bend_angle_deg = parse_bend_angle_from_path(log_csv)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue

            runs.append(
                {
                    "controller": controller,
                    "bend_angle_deg": bend_angle_deg,
                    "log_csv": log_csv,
                }
            )

    return runs
# ============================================================
# USER CONFIGURATION
# ============================================================
# Put the exact log paths here.
# Each entry is one run.
BASELINE_CONTROLLERS = [
    "fixed_lti_rollout1_contact",
    "fixed_lti_rollout5_contact",
    "fixed_ltv_rollout1_contact",
    "fixed_ltv_rollout3_contact",
    "fixed_sqp_rollout1_contact",
    "hierarchy_full_contact",
]

baseline_metrics = [
    "final_i_ref",
    "max_pred_err_xy_mm",
    "min_clearance_mm",
    "num_failed_steps",
    "mean_rollout_steps_used",
    "frac_lti",
    "frac_ltv_oneshot",
    "frac_sqp_full",
    "mean_solver_complexity_proxy",
    "sum_solver_complexity_proxy",
]

ABLATION_CONTROLLERS = [
    "hierarchy_full_contact",
    "hierarchy_no_curvature_contact",
    "hierarchy_no_hessian_contact",
    "hierarchy_no_sqp_step_contact",
]

ablation_metrics = [
    "final_i_ref",
    "max_pred_err_xy_mm",
    "min_clearance_mm",
    "num_failed_steps",
    "frac_sqp_full",
    "mean_rollout_steps_used",
    "num_hier_curvature",
    "num_hier_sqp_du_rel",
    "num_hier_clearance",
    "num_hier_beam_cond",
    "num_hier_mpc_cond",
    "num_solver_switches",
    "num_rollout_switches",
    "mean_solver_complexity_proxy",
]

MISMATCH_CONTROLLERS = [
    "hierarchy_full_contact",
    "hierarchy_full_no_contact_model",
    "fixed_sqp_rollout1_contact",
    "fixed_sqp_rollout1_no_contact_model",
]

mismatch_metrics = [
    "final_i_ref",
    "max_pred_err_xy_mm",
    "min_clearance_mm",
    "max_cond_H_beam",
    "max_cond_H_mpc",
    "frac_sqp_full",
    "num_failed_steps",
    "mean_solver_complexity_proxy",
]

NP_CONTROLLERS = [
    "hierarchy_full_contact_Np5",
    "hierarchy_full_contact_Np8",
    "hierarchy_full_contact_Np10",
]

np_metrics = [
    "final_i_ref",
    "max_pred_err_xy_mm",
    "min_clearance_mm",
    "max_cond_H_mpc",
    "frac_sqp_full",
    "mean_rollout_steps_used",
    "num_failed_steps",
    "mean_solver_complexity_proxy",
]
ROOTS = {
    # ------------------------------------------------------------
    # A. Fixed-controller baselines
    # ------------------------------------------------------------
    "fixed_lti_rollout5_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_lti_rollout5_np5"
    ),
    "fixed_lti_rollout1_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_lti_rollout1_np5"
    ),
    "fixed_ltv_rollout3_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_ltv_rollout3_np5"
    ),
    "fixed_ltv_rollout1_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_ltv_rollout1_np5"
    ),
    "fixed_sqp_rollout1_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_sqp_rollout1_np5"
    ),

    # ------------------------------------------------------------
    # B. Hierarchy ablations
    # ------------------------------------------------------------
    "hierarchy_full_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_ablation_full"
    ),
    "hierarchy_no_curvature_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_ablation_no_curvature"
    ),
    "hierarchy_no_hessian_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_ablation_no_hessian"
    ),
    "hierarchy_no_sqp_step_contact": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_ablation_no_sqp_step"
    ),

    # ------------------------------------------------------------
    # C. Model-mismatch tests
    # ------------------------------------------------------------
    "hierarchy_full_no_contact_model": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_simple_v1_no_contact_model"
    ),
    "fixed_sqp_rollout1_no_contact_model": Path(
        "/Users/jackhilton-jones/Proper-Research/run_baseline_sqp_rollout1_np5_no_contact_model"
    ),

    # ------------------------------------------------------------
    # D. Prediction-horizon sensitivity
    # ------------------------------------------------------------
    "hierarchy_full_contact_Np5": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_simple_Np5"
    ),
    "hierarchy_full_contact_Np8": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_simple_Np8"
    ),
    "hierarchy_full_contact_Np10": Path(
        "/Users/jackhilton-jones/Proper-Research/run_hierarchy_simple_Np10"
    ),
}

OUT_ROOT = Path("analysis_hierarchy_full_study")

# ============================================================
# BASIC UTILITIES
# ============================================================
def parse_bend_angle_from_path(path: Path) -> float:
    """
    Extract bend angle from folder names like:
        bend_m30_jac_contact_...
        bend_m60_jac_contact_...
        bend_m90_jac_contact_...
    """
    text = str(path)

    if "bend_m30" in text:
        return -30.0
    if "bend_m60" in text:
        return -60.0
    if "bend_m90" in text:
        return -90.0

    if "bend_p30" in text:
        return 30.0
    if "bend_p60" in text:
        return 60.0
    if "bend_p90" in text:
        return 90.0

    raise ValueError(f"Could not infer bend angle from path: {path}")

def controller_metadata(controller: str) -> dict:
    c = str(controller)

    if c.startswith("fixed_"):
        family = "fixed_baseline"
    elif c.startswith("hierarchy_"):
        family = "hierarchy"
    else:
        family = "other"

    if "lti" in c:
        nominal_solver = "lti"
    elif "ltv" in c:
        nominal_solver = "ltv_oneshot"
    elif "sqp" in c:
        nominal_solver = "sqp_full"
    elif "hierarchy" in c:
        nominal_solver = "adaptive"
    else:
        nominal_solver = ""

    if "rollout5" in c:
        nominal_rollout = 5
    elif "rollout3" in c:
        nominal_rollout = 3
    elif "rollout1" in c:
        nominal_rollout = 1
    else:
        nominal_rollout = np.nan

    if "no_contact_model" in c:
        jacobian_model = "no_contact"
    else:
        jacobian_model = "contact"

    if "no_curvature" in c:
        ablation = "no_curvature"
    elif "no_hessian" in c:
        ablation = "no_hessian"
    elif "no_sqp_step" in c:
        ablation = "no_sqp_step"
    elif "hierarchy_full" in c or "hierarchy_simple" in c:
        ablation = "full"
    else:
        ablation = ""

    if "Np10" in c:
        Np = 10
    elif "Np8" in c:
        Np = 8
    elif "Np5" in c:
        Np = 5
    else:
        Np = np.nan

    return {
        "family": family,
        "nominal_solver": nominal_solver,
        "nominal_rollout": nominal_rollout,
        "jacobian_model": jacobian_model,
        "ablation": ablation,
        "Np": Np,
    }
def read_log_csv(path: Path) -> pd.DataFrame:
    path = Path(path)

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="warn")


def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def safe_str(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].astype(str)
    return pd.Series("", index=df.index)


def fraction_equal(series: pd.Series, value: str) -> float:
    if len(series) == 0:
        return np.nan

    s = series.astype(str)
    return float((s == value).mean())


def count_contains(series: pd.Series, substring: str) -> int:
    if len(series) == 0:
        return 0

    return int(
        series.fillna("")
        .astype(str)
        .str.contains(substring, regex=False)
        .sum()
    )


def savefig(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

def summarize_log(controller: str, bend_angle_deg: float, log_csv: Path) -> dict:
    df = read_log_csv(log_csv)
    n = len(df)

    infeasible = safe_num(df, "infeasible").fillna(0).astype(int)
    status = safe_str(df, "status")

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
    hierarchy_level = safe_num(df, "hierarchy_level")
    hierarchy_confidence = safe_num(df, "hierarchy_confidence")
    hierarchy_risk = safe_num(df, "hierarchy_risk")

    u0_vx = safe_num(df, "u0_vx")
    u0_vy = safe_num(df, "u0_vy")
    u0_wz = safe_num(df, "u0_wz")
    u0_dL = safe_num(df, "u0_dL")

    sat_vxy = np.maximum(u0_vx.abs(), u0_vy.abs()) >= 0.95
    sat_wz = u0_wz.abs() >= 0.95 * 2.0 * np.pi
    sat_dL = u0_dL.abs() >= 0.095
    level_series = safe_num(df, "hierarchy_level")
    rollout_series = safe_num(df, "rollout_steps_used")
    solver_series = solver_used.astype(str)

    num_level_switches = int(
        level_series.dropna().diff().fillna(0).ne(0).sum()
    )

    num_rollout_switches = int(
        rollout_series.dropna().diff().fillna(0).ne(0).sum()
    )

    num_solver_switches = int(
        solver_series.ne(solver_series.shift()).sum() - 1
    ) if len(solver_series) > 1 else 0
    first_failure_k = -1
    if failure_mask.any():
        first_failure_k = int(np.where(failure_mask.to_numpy())[0][0])
    sqp_iters = safe_num(df, "sqp_iters_done")
    N_sqp_used = safe_num(df, "N_sqp_used")


    row = {
        "controller": controller,
        "bend_angle_deg": float(bend_angle_deg),
        "log_csv": str(log_csv),

        "n_steps": int(n),
        "success": bool((failure_mask == 0).all()) if n else False,
        "num_failed_steps": int(failure_mask.sum()) if n else 0,
        "num_infeasible_steps": int(infeasible.sum()) if n else 0,
        "first_failure_k": first_failure_k,
        "final_status": str(status.iloc[-1]) if n else "",

        "final_i_ref": float(safe_num(df, "i_ref").iloc[-1]) if n else np.nan,
        "max_i_ref": float(safe_num(df, "i_ref").max()),

        "mean_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").mean()),
        "max_pred_err_xy_mm": float(safe_num(df, "adapt_pred_err_xy_mm").max()),

        "mean_clearance_mm": float(safe_num(df, "adapt_clearance_mm").mean()),
        "min_clearance_mm": float(safe_num(df, "adapt_clearance_mm").min()),

        "mean_tip_vessel_angle_deg": float(
            safe_num(df, "adapt_tip_vessel_angle_deg").mean()
        ),
        "max_tip_vessel_angle_deg": float(
            safe_num(df, "adapt_tip_vessel_angle_deg").max()
        ),

        "mean_cond_H_beam": float(safe_num(df, "cond_H_beam").mean()),
        "max_cond_H_beam": float(safe_num(df, "cond_H_beam").max()),

        "mean_cond_H_mpc": float(safe_num(df, "cond_H_mpc").mean()),
        "max_cond_H_mpc": float(safe_num(df, "cond_H_mpc").max()),

        "mean_sqp_du_final": float(safe_num(df, "sqp_du_final").mean()),
        "max_sqp_du_final": float(safe_num(df, "sqp_du_final").max()),
        "mean_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").mean()),
        "max_sqp_du_rel_final": float(safe_num(df, "sqp_du_rel_final").max()),

        "mean_curvature_ahead_1pm": float(
            safe_num(df, "curvature_max_ahead_1pm").mean()
        ),
        "max_curvature_ahead_1pm": float(
            safe_num(df, "curvature_max_ahead_1pm").max()
        ),

        "mean_rollout_steps_used": float(rollout_used.mean()),
        "max_rollout_steps_used": float(rollout_used.max()),

        "mean_hierarchy_level": float(hierarchy_level.mean()),
        "max_hierarchy_level": float(hierarchy_level.max()),
        "mean_hierarchy_confidence": float(hierarchy_confidence.mean()),
        "min_hierarchy_confidence": float(hierarchy_confidence.min()),
        "mean_hierarchy_risk": float(hierarchy_risk.mean()),
        "max_hierarchy_risk": float(hierarchy_risk.max()),

        "frac_lti": fraction_equal(solver_used, "lti"),
        "frac_ltv_oneshot": fraction_equal(solver_used, "ltv_oneshot"),
        "frac_sqp_full": fraction_equal(solver_used, "sqp_full"),

        "frac_vxy_saturated": float(sat_vxy.mean()) if n else np.nan,
        "frac_wz_saturated": float(sat_wz.mean()) if n else np.nan,
        "frac_dL_saturated": float(sat_dL.mean()) if n else np.nan,

        "num_hier_curvature": count_contains(hierarchy_reason, "curvature"),
        "num_hier_sqp_du_rel": count_contains(hierarchy_reason, "sqp_du_rel"),
        "num_hier_clearance": count_contains(hierarchy_reason, "clearance"),
        "num_hier_beam_cond": count_contains(hierarchy_reason, "beam_cond"),
        "num_hier_mpc_cond": count_contains(hierarchy_reason, "mpc_cond"),

        "num_hier_hard": count_contains(hierarchy_reason, "hard:"),
        "num_hier_moderate": count_contains(hierarchy_reason, "moderate:"),
        "num_hier_easy": count_contains(hierarchy_reason, "easy"),

        "num_hier_escalate": count_contains(hierarchy_reason, "escalate"),
        "num_hier_downgrade_held": count_contains(hierarchy_reason, "downgrade_held"),
        "num_hier_downgrade_allowed": count_contains(hierarchy_reason, "downgrade_allowed"),
        "num_level_switches": num_level_switches,
        "num_rollout_switches": num_rollout_switches,
        "num_solver_switches": num_solver_switches,
        "unique_rollout_steps": ",".join(
            str(int(v)) for v in sorted(rollout_series.dropna().unique())
        ),
        "unique_solver_modes": ",".join(
            sorted(solver_series.dropna().unique())
        ),
        "num_rollout_good_increase": count_contains(rollout_reason, "good:increase"),
        "num_rollout_bad_tip_angle": count_contains(rollout_reason, "bad:tip_angle"),
        "num_rollout_bad_low_clearance": count_contains(
            rollout_reason,
            "bad:low_clearance",
        ),
    }
    row.update(
        {
            "mean_sqp_iters_done": float(sqp_iters.mean()),
            "max_sqp_iters_done": float(sqp_iters.max()),
            "sum_sqp_iters_done": float(sqp_iters.sum()),

            "mean_N_sqp_used": float(N_sqp_used.mean()),
            "max_N_sqp_used": float(N_sqp_used.max()),
            "sum_N_sqp_used": float(N_sqp_used.sum()),

            # Simple normalized complexity proxy.
            # LTI and LTV one-shot cost 1; SQP full costs N_sqp_used.
            "mean_solver_complexity_proxy": float(N_sqp_used.fillna(1).mean()),
            "sum_solver_complexity_proxy": float(N_sqp_used.fillna(1).sum()),
        }
    )
    row.update(controller_metadata(controller))
    return row


def build_summary(runs_config: list[dict], out_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    loaded_runs = []
    rows = []

    for item in runs_config:
        controller = item["controller"]
        bend_angle_deg = float(item["bend_angle_deg"])
        log_csv = Path(item["log_csv"])

        if not log_csv.exists():
            print(f"[WARN] Missing log: {log_csv}")
            continue

        df = read_log_csv(log_csv)

        loaded_runs.append(
            {
                "controller": controller,
                "bend_angle_deg": bend_angle_deg,
                "log_csv": log_csv,
                "df": df,
            }
        )

        rows.append(
            summarize_log(
                controller=controller,
                bend_angle_deg=bend_angle_deg,
                log_csv=log_csv,
            )
        )

    summary = pd.DataFrame(rows)

    if len(summary):
        summary = summary.sort_values(["bend_angle_deg", "controller"])

    summary.to_csv(out_root / "hierarchy_summary.csv", index=False)
    print(f"[ANALYSIS] Wrote {out_root / 'hierarchy_summary.csv'}")

    return summary, loaded_runs


# ============================================================
# SCORING
# ============================================================

def add_score(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()

    clearance_penalty = np.maximum(
        0.0,
        1.0 - df["min_clearance_mm"].astype(float),
    )

    df["score"] = (
        1.0 * df["final_i_ref"].astype(float)
        - 10.0 * df["max_pred_err_xy_mm"].astype(float).fillna(0.0)
        - 25.0 * clearance_penalty.fillna(0.0)
        - 0.4 * df["max_tip_vessel_angle_deg"].astype(float).fillna(0.0)
        - 100.0 * df["num_failed_steps"].astype(float).fillna(0.0)
        - 3.0 * df["frac_sqp_full"].astype(float).fillna(0.0)
    )

    return df


# ============================================================
# PLOTS
# ============================================================

def controller_order(summary: pd.DataFrame):
    preferred = [
        "fixed_lti_rollout1_contact",
        "fixed_lti_rollout5_contact",
        "fixed_ltv_rollout1_contact",
        "fixed_ltv_rollout3_contact",
        "fixed_sqp_rollout1_contact",
        "hierarchy_full_contact",
        "hierarchy_no_curvature_contact",
        "hierarchy_no_hessian_contact",
        "hierarchy_no_sqp_step_contact",
        "hierarchy_full_no_contact_model",
        "fixed_sqp_rollout1_no_contact_model",
        "hierarchy_full_contact_Np5",
        "hierarchy_full_contact_Np8",
        "hierarchy_full_contact_Np10",
    ]

    present = list(summary["controller"].dropna().unique())

    ordered = [p for p in preferred if p in present]
    ordered += [p for p in present if p not in ordered]

    return ordered


def bend_order(summary: pd.DataFrame):
    return sorted(summary["bend_angle_deg"].dropna().unique())


def plot_summary_metric(summary: pd.DataFrame, metric: str, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = controller_order(summary)
    bends = bend_order(summary)

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(5 * len(bends), 4),
        sharey=True,
    )

    if len(bends) == 1:
        axes = [axes]

    for ax, bend in zip(axes, bends):
        values = []
        success_flags = []

        for policy in policies:
            g = summary[
                (summary["bend_angle_deg"] == bend)
                & (summary["controller"] == policy)
            ]

            if g.empty or metric not in g.columns:
                values.append(np.nan)
                success_flags.append(False)
            else:
                values.append(float(g[metric].iloc[0]))
                success_flags.append(bool(g["success"].iloc[0]))

        x = np.arange(len(policies))
        bars = ax.bar(x, values)

        for bar, ok in zip(bars, success_flags):
            if not ok:
                bar.set_hatch("//")
                bar.set_edgecolor("black")

        ax.set_title(f"Bend {bend:.0f}°")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"{metric} by hierarchy policy")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    savefig(fig, out_dir / f"{metric}_by_policy.png")


def plot_solver_usage(summary: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = controller_order(summary)
    bends = bend_order(summary)

    usage_cols = [
        ("frac_lti", "LTI"),
        ("frac_ltv_oneshot", "LTV"),
        ("frac_sqp_full", "SQP"),
    ]

    for bend in bends:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        x = np.arange(len(policies))
        bottom = np.zeros(len(policies), float)

        for col, label in usage_cols:
            vals = []

            for policy in policies:
                g = summary[
                    (summary["bend_angle_deg"] == bend)
                    & (summary["controller"] == policy)
                ]

                if g.empty:
                    vals.append(0.0)
                else:
                    vals.append(float(g[col].iloc[0]))

            vals = np.asarray(vals, float)

            ax.bar(x, vals, bottom=bottom, label=label)
            bottom += vals

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction of MPC updates")
        ax.set_title(f"Solver usage, bend {bend:.0f}°")
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        fig.tight_layout()

        bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"
        savefig(fig, out_dir / f"solver_usage_bend_{bend_tag}.png")


def plot_trace_metric(
    loaded_runs: list[dict],
    metric: str,
    out_dir: Path,
    *,
    x_col: str = "i_ref",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bends = sorted({r["bend_angle_deg"] for r in loaded_runs})
    policies = sorted({r["controller"] for r in loaded_runs})

    for bend in bends:
        fig, ax = plt.subplots(figsize=(8, 4.8))

        for policy in policies:
            matching = [
                r for r in loaded_runs
                if r["bend_angle_deg"] == bend
                and r["controller"] == policy
            ]

            if not matching:
                continue

            run = matching[0]
            df = run["df"]

            if metric not in df.columns or x_col not in df.columns:
                continue

            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[metric], errors="coerce")

            valid = x.notna() & y.notna() & np.isfinite(y)

            if valid.sum() == 0:
                continue

            if metric in ("rollout_steps_used", "hierarchy_level", "N_sqp_used"):
                ax.step(
                    x[valid],
                    y[valid],
                    where="post",
                    linewidth=2.0,
                    label=policy,
                )
            else:
                ax.plot(
                    x[valid],
                    y[valid],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.8,
                    label=policy,
                )

            if "infeasible" in df.columns:
                infeas = safe_num(df, "infeasible").fillna(0).astype(int) > 0
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

        if metric in ("cond_H_beam", "cond_H_mpc", "sqp_du_rel_final"):
            ax.set_yscale("log")

        ax.set_title(f"{metric} vs {x_col}, bend {bend:.0f}°")
        ax.set_xlabel("Reference index")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"
        savefig(fig, out_dir / f"{metric}_bend_{bend_tag}.png")


def plot_solver_timeline(loaded_runs: list[dict], out_dir: Path, *, x_col: str = "i_ref"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_to_level = {
        "lti": 0,
        "ltv_oneshot": 1,
        "sqp_full": 2,
    }

    bends = sorted({r["bend_angle_deg"] for r in loaded_runs})
    policies = sorted({r["controller"] for r in loaded_runs})

    for bend in bends:
        fig, ax = plt.subplots(figsize=(8, 4.8))

        for policy in policies:
            matching = [
                r for r in loaded_runs
                if r["bend_angle_deg"] == bend
                and r["controller"] == policy
            ]

            if not matching:
                continue

            run = matching[0]
            df = run["df"]

            if x_col not in df.columns:
                continue

            solver = safe_str(df, "solver_mode_used")
            if (solver == "").all():
                solver = safe_str(df, "solver_mode")

            y = solver.map(mode_to_level)
            x = pd.to_numeric(df[x_col], errors="coerce")

            valid = x.notna() & y.notna()

            ax.step(
                x[valid],
                y[valid],
                where="post",
                linewidth=2.0,
                label=policy,
            )

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["LTI", "LTV", "SQP"])
        ax.set_xlabel("Reference index")
        ax.set_ylabel("Solver mode used")
        ax.set_title(f"Solver mode timeline, bend {bend:.0f}°")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        bend_tag = f"m{int(abs(bend))}" if bend < 0 else f"p{int(abs(bend))}"
        savefig(fig, out_dir / f"solver_timeline_bend_{bend_tag}.png")


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_analysis():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    RUNS = collect_runs_from_roots(ROOTS)
    summary, loaded_runs = build_summary(RUNS, OUT_ROOT)

    if summary.empty:
        print("[ANALYSIS] No logs found.")
        return

    summary = add_score(summary)
    summary.to_csv(OUT_ROOT / "hierarchy_summary_with_score.csv", index=False)

    print("\n[SUMMARY]")
    print(
        summary[
            [
                "controller",
                "bend_angle_deg",
                "success",
                "score",
                "final_i_ref",
                "final_status",
                "max_pred_err_xy_mm",
                "min_clearance_mm",
                "mean_rollout_steps_used",
                "frac_lti",
                "frac_ltv_oneshot",
                "frac_sqp_full",
                "num_solver_switches",
                "num_rollout_switches",
                "unique_solver_modes",
                "unique_rollout_steps",
            ]
        ].to_string(index=False)
    )

    print("\n[MEAN SCORE BY POLICY]")
    print(
        summary.groupby("controller")["score"]
        .mean()
        .sort_values(ascending=False)
        .to_string()
    )

    plot_dir = OUT_ROOT / "plots"


    comparison_groups = {
        "baseline_comparison": (
            BASELINE_CONTROLLERS,
            baseline_metrics,
        ),
        "hierarchy_ablation": (
            ABLATION_CONTROLLERS,
            ablation_metrics,
        ),
        "model_mismatch": (
            MISMATCH_CONTROLLERS,
            mismatch_metrics,
        ),
        "prediction_horizon": (
            NP_CONTROLLERS,
            np_metrics,
        ),
    }

    for group_name, (controllers, metrics) in comparison_groups.items():
        for metric in metrics:
            if metric in summary.columns:
                plot_summary_metric_for_controllers(
                    summary,
                    controllers,
                    metric,
                    out_dir=plot_dir / group_name,
                    title_prefix=group_name,
                )
    plot_solver_usage(summary, out_dir=plot_dir / "solver_usage")

    trace_metrics = [
        # Performance
        "adapt_pred_err_xy_mm",
        "adapt_clearance_mm",

        # Hierarchy behaviour
        "rollout_steps_used",
        "hierarchy_level",
        "N_sqp_used",

        # Preventive switching metrics
        "curvature_max_ahead_1pm",
        "sqp_du_rel_final",
        "cond_H_beam",
        "cond_H_mpc",
    ]

    for metric in trace_metrics:
        plot_trace_metric(
            loaded_runs,
            metric,
            out_dir=plot_dir / "traces",
            x_col="i_ref",
        )

    plot_solver_timeline(
        loaded_runs,
        out_dir=plot_dir / "solver_timeline",
        x_col="i_ref",
    )

    print(f"\n[ANALYSIS] Finished. Outputs written to: {OUT_ROOT}")
    write_group_summary(
        summary,
        BASELINE_CONTROLLERS,
        OUT_ROOT / "summary_baseline_comparison.csv",
    )

    write_group_summary(
        summary,
        ABLATION_CONTROLLERS,
        OUT_ROOT / "summary_hierarchy_ablation.csv",
    )

    write_group_summary(
        summary,
        MISMATCH_CONTROLLERS,
        OUT_ROOT / "summary_model_mismatch.csv",
    )

    write_group_summary(
        summary,
        NP_CONTROLLERS,
        OUT_ROOT / "summary_prediction_horizon.csv",
    )
def write_group_summary(summary, controllers, out_path):
    sub = summary[summary["controller"].isin(controllers)].copy()
    sub.to_csv(out_path, index=False)
    return sub
def plot_summary_metric_for_controllers(
    summary: pd.DataFrame,
    controllers: list[str],
    metric: str,
    out_dir: Path,
    *,
    title_prefix: str = "",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = summary[summary["controller"].isin(controllers)].copy()
    bends = bend_order(sub)

    if sub.empty or metric not in sub.columns:
        return

    fig, axes = plt.subplots(
        1,
        len(bends),
        figsize=(5.5 * len(bends), 4.2),
        sharey=True,
    )

    if len(bends) == 1:
        axes = [axes]

    for ax, bend in zip(axes, bends):
        values = []
        success_flags = []

        for controller in controllers:
            g = sub[
                (sub["bend_angle_deg"] == bend)
                & (sub["controller"] == controller)
            ]

            if g.empty:
                values.append(np.nan)
                success_flags.append(False)
            else:
                values.append(float(g[metric].iloc[0]))
                success_flags.append(bool(g["success"].iloc[0]))

        x = np.arange(len(controllers))
        bars = ax.bar(x, values)

        for bar, ok in zip(bars, success_flags):
            if not ok:
                bar.set_hatch("//")
                bar.set_edgecolor("black")

        ax.set_title(f"Bend {bend:.0f}°")
        ax.set_xticks(x)
        ax.set_xticklabels(controllers, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(metric)

    if title_prefix:
        fig.suptitle(f"{title_prefix}: {metric}")
    else:
        fig.suptitle(metric)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    safe_metric = metric.replace("/", "_")
    safe_title = title_prefix.lower().replace(" ", "_").replace("/", "_")
    savefig(fig, out_dir / f"{safe_title}_{safe_metric}.png")
if __name__ == "__main__":

    run_analysis()
    plt.show()
