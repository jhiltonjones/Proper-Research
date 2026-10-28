"""Publication figure set for the 10x15mm rectangle controller comparison
(2026-09-12/13) -- the shape-generalization follow-up to the 20mm-triangle
comparison (make_publication_figures.py / day2_triangle20mm_2026-09-12/).

Run from the repo root:

    python3 -m proper_research.hardware.online.make_rectangle_figures

Writes 8 PNGs into close_loop_logs/day3_rectangle10x15mm_2026-09-13/figures/:
    fig1_tip_paths.png                  tip path vs reference, 4 conditions
    fig2_error_vs_time.png              tracking error vs time, 4 conditions
    fig3_summary_bars.png               full 7-condition RMS/max/hold/final bars
    fig4_jacobian_svd.png               frozen beam Jacobian singular-value spectrum
    fig5_jacobian_evolution.png         singular values / condition number along
                                         the path (mpc_ltv's per-step schedule)
    fig6_jacobian_mismatch_vs_error.png ||J(k)-J(0)||_F vs progress, overlaid with
                                         mpc_lti's vs mpc_ltv's tracking error --
                                         explains why LTV edges out LTI here too
    fig7_controller_fingerprint.png     saturation / jerk / nullspace-usage
                                         fingerprint, inv+FF vs mpc_lti+FF vs
                                         mpc_ltv+FF -- same honest diagnostic used
                                         on the triangle, checking whether the
                                         same explanation (or lack of one) holds
    fig8_correction_vs_error.png        |correction| vs tracking error at every
                                         hold-phase tick, inv+FF vs mpc_ltv+FF,
                                         all 3 reps pooled -- inv+FF is cleanly
                                         proportional (r=+0.97 to +1.00 every
                                         rep); mpc_ltv+FF is NEGATIVELY
                                         correlated in EVERY rep (r=-0.28 to
                                         -0.96) despite the state_tracking_weight
                                         fix being applied -- an unresolved,
                                         genuinely surprising finding: the fix
                                         that cleanly resolved this exact
                                         pathology on the triangle does not fully
                                         generalise to the rectangle's geometry.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
BASE = Path("close_loop_logs")
OUT_ROOT = BASE / "day3_rectangle10x15mm_2026-09-13"

RUN_GLOBS: dict[str, list[Path]] = {
    "open loop": [BASE / "openloop_rectangle10x15mm_2026-09-12" / "runs" / "*",
                  BASE / "openloop_rectangle10x15mm_2026-09-13" / "runs" / "*"],
    "inv (pure fb)": [BASE / "rectangle10x15mm_noFF_n3_2026-09-13" / "runs" / "inv_rep*"],
    "inv+FF": [BASE / "rectangle10x15mm_FF_n3_2026-09-13" / "runs" / "inv_rep*"],
    "mpc_lti (pure fb)": [BASE / "rectangle10x15mm_noFF_n3_2026-09-13" / "runs" / "lti_rep*"],
    "mpc_lti+FF (fixed)": [BASE / "rectangle10x15mm_FF_n3_2026-09-13" / "runs" / "lti_rep*"],
    "mpc_ltv (pure fb)": [BASE / "rectangle10x15mm_noFF_n3_2026-09-13" / "runs" / "ltv_rep*"],
    "mpc_ltv+FF (fixed)": [BASE / "rectangle10x15mm_FF_n3_2026-09-13" / "runs" / "ltv_rep*"],
}
LTV_SCHEDULE_NPZ = BASE / "rectangle10x15mm_FF_n3_2026-09-13" / "ltv_schedule.npz"
MAX_JOINT_STEP_RAD = 0.006

FF_CONDITIONS = ["open loop", "inv+FF", "mpc_lti+FF (fixed)", "mpc_ltv+FF (fixed)"]
ALL_CONDITIONS = [
    "open loop",
    "inv (pure fb)", "inv+FF",
    "mpc_lti (pure fb)", "mpc_lti+FF (fixed)",
    "mpc_ltv (pure fb)", "mpc_ltv+FF (fixed)",
]
COLORS = {
    "open loop": "#888888",
    "inv (pure fb)": "#1f77b4", "inv+FF": "#1f77b4",
    "mpc_lti (pure fb)": "#ff7f0e", "mpc_lti+FF (fixed)": "#ff7f0e",
    "mpc_ltv (pure fb)": "#2ca02c", "mpc_ltv+FF (fixed)": "#2ca02c",
}
U_AXIS = np.array([0.0, 1.0, 0.0])
V_AXIS = np.array([-1.0, 0.0, 0.0])

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 130, "savefig.dpi": 220,
    "axes.spines.top": False, "axes.spines.right": False,
})


# =============================================================================
def _glob(pattern: Path) -> list[Path]:
    parent = pattern.parent
    return sorted(parent.glob(pattern.name)) if parent.exists() else []


def valid_reps(patterns: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for pat in patterns:
        candidates += _glob(pat)
    out = []
    for d in candidates:
        s = d / "summary.json"
        if s.exists():
            try:
                if json.loads(s.read_text()).get("stop_reason") == "path_complete":
                    out.append(d)
            except Exception:
                pass
    return sorted(set(out))


def read_traj(run_dir: Path) -> np.ndarray:
    return np.genfromtxt(run_dir / "tip_trajectory.csv", delimiter=",", names=True)


def read_rows(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in open(run_dir / "path_follow.jsonl")]


def rms_metric(run_dir: Path) -> float:
    a = read_traj(run_dir)
    err = np.atleast_1d(a["err_norm_mm"])
    hold = np.atleast_1d(a["terminal_hold"]).astype(bool)
    return float(np.sqrt(np.mean(err[~hold] ** 2)))


def all_metrics(run_dir: Path) -> dict:
    a = read_traj(run_dir)
    err = np.atleast_1d(a["err_norm_mm"])
    hold = np.atleast_1d(a["terminal_hold"]).astype(bool)
    track = ~hold
    return {
        "rms": float(np.sqrt(np.mean(err[track] ** 2))),
        "max": float(np.max(err[track])),
        "hold": float(np.mean(err[hold])) if hold.any() else float("nan"),
        "final": float(err[-1]),
    }


def median_rep(reps: list[Path]) -> Path:
    vals = [(r, rms_metric(r)) for r in reps]
    med = float(np.median([v for _, v in vals]))
    return min(vals, key=lambda rv: abs(rv[1] - med))[0]


def weak_dir_fraction(run_dir: Path, v_weak: np.ndarray, hold_only: bool = True) -> float:
    rows = read_rows(run_dir)
    corr6 = np.array([r["u0_correction"][:6] for r in rows])
    total = np.linalg.norm(corr6, axis=1)
    weak_comp = np.abs(corr6 @ v_weak)
    frac = np.divide(weak_comp, total, out=np.zeros_like(total), where=total > 1e-9)
    if hold_only:
        hold = np.array([r["terminal_hold"] for r in rows])
        return float(frac[hold].mean()) if hold.any() else float(frac.mean())
    return float(frac.mean())


# =============================================================================
def fig1_tip_paths(conditions, out: Path) -> None:
    fig, axes = plt.subplots(1, len(FF_CONDITIONS), figsize=(19, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, FF_CONDITIONS):
        rep = median_rep(conditions[name])
        a = read_traj(rep)
        tip = np.c_[a["tip_x_m"], a["tip_y_m"], a["tip_z_m"]]
        des = np.c_[a["des_x_m"], a["des_y_m"], a["des_z_m"]]
        origin = des[0]
        tip_u = 1e3 * (tip - origin) @ U_AXIS
        tip_v = 1e3 * (tip - origin) @ V_AXIS
        des_u = 1e3 * (des - origin) @ U_AXIS
        des_v = 1e3 * (des - origin) @ V_AXIS
        ax.plot(des_u, des_v, "--", color="black", lw=1.6, alpha=0.8, label="reference")
        ax.plot(tip_u, tip_v, "-", color=COLORS[name], lw=1.8, label="measured tip")
        ax.plot(tip_u[0], tip_v[0], "ko", ms=5)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("B.y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("B.x (mm)")
    fig.suptitle("Tip path vs. reference — 10x15mm rectangle (representative rep per condition)", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "fig1_tip_paths.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_tip_paths.png")


def fig2_error_vs_time(conditions, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in FF_CONDITIONS:
        rep = median_rep(conditions[name])
        a = read_traj(rep)
        t_rel = a["t_s"] - a["t_s"][0]
        ax.plot(t_rel, a["err_norm_mm"], color=COLORS[name], lw=1.6, label=name, alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tracking error |e| (mm)")
    ax.set_title("Tip-tracking error vs. time — 10x15mm rectangle, feedforward conditions")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig2_error_vs_time.png")
    plt.close(fig)
    print("wrote fig2_error_vs_time.png")


def fig3_summary_bars(conditions, out: Path) -> None:
    agg = {}
    for name in ALL_CONDITIONS:
        ms = [all_metrics(r) for r in conditions[name]]
        agg[name] = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms]))
                     for k in ("rms", "max", "hold", "final")}
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.5))
    metric_labels = [("rms", "RMS tracking error (mm)"), ("max", "Max tracking error (mm)"),
                      ("hold", "Mean hold error (mm)"), ("final", "Final error (mm)")]
    x = np.arange(len(ALL_CONDITIONS))
    bar_colors = [COLORS[n] for n in ALL_CONDITIONS]
    hatches = ["" if "FF" in n or n == "open loop" else "//" for n in ALL_CONDITIONS]
    for ax, (key, label) in zip(axs, metric_labels):
        means = [agg[n][key][0] for n in ALL_CONDITIONS]
        stds = [agg[n][key][1] for n in ALL_CONDITIONS]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CONDITIONS, rotation=40, ha="right", fontsize=8.5)
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0])
    fig.suptitle("Full controller comparison, 10x15mm rectangle, n=3 per condition (n=1x2 days for "
                 "open loop) — hatched = pure feedback, solid = feedforward", y=1.04)
    fig.tight_layout()
    fig.savefig(out / "fig3_summary_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_summary_bars.png")


def fig4_jacobian_svd(conditions, out: Path):
    jac_path = median_rep(conditions["mpc_ltv+FF (fixed)"]) / "frozen_jacobian.json"
    jac = json.loads(jac_path.read_text())
    J6 = np.array(jac["j_full_3x7"])[:, :6]
    u, s, vt = np.linalg.svd(J6, full_matrices=False)
    cond = s[0] / s[-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"$\\sigma_{i+1}$" for i in range(3)], s,
                  color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("singular value (mm / rad)")
    ax.set_title(f"Beam Jacobian $J$ (3$\\times$6, joint block)\ncondition number $\\kappa$ = {cond:,.0f}")
    for bar, val in zip(bars, s):
        ax.annotate(f"{val:.2e}", (bar.get_x() + bar.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_jacobian_svd.png")
    plt.close(fig)
    print(f"wrote fig4_jacobian_svd.png (cond={cond:.0f}, sigma={s})")
    return s, vt


def fig5_jacobian_evolution(out: Path):
    if not LTV_SCHEDULE_NPZ.exists():
        print("skip fig5/fig6 (LTV_SCHEDULE_NPZ not found)")
        return None
    schedule = np.load(LTV_SCHEDULE_NPZ)["schedule"]
    n = schedule.shape[0]
    sv = np.array([np.linalg.svd(schedule[k, :, :6], compute_uv=False) for k in range(n)])
    cond = sv[:, 0] / sv[:, 2]
    progress = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(progress, sv[:, 0], label="$\\sigma_1$ (strongest)", color="#2ca02c")
    ax1.plot(progress, sv[:, 1], label="$\\sigma_2$", color="#ff7f0e")
    ax1.plot(progress, sv[:, 2], label="$\\sigma_3$ (near-null)", color="#d62728")
    ax1.axhline(sv[0, 0], color="#2ca02c", ls=":", lw=1, alpha=0.6, label="LTI frozen value")
    ax1.axhline(sv[0, 1], color="#ff7f0e", ls=":", lw=1, alpha=0.6)
    ax1.axhline(sv[0, 2], color="#d62728", ls=":", lw=1, alpha=0.6)
    ax1.set_yscale("log")
    ax1.set_xlabel("reference sample index (path progress)")
    ax1.set_ylabel("singular value (mm / rad)")
    ax1.set_title("Local Jacobian singular values along the path")
    ax1.legend(fontsize=8.5)

    ax2.plot(progress, cond, color="#9467bd")
    ax2.axhline(cond[0], color="#9467bd", ls=":", lw=1, alpha=0.6, label="LTI frozen value")
    ax2.set_yscale("log")
    ax2.set_xlabel("reference sample index (path progress)")
    ax2.set_ylabel("condition number $\\kappa$ (log scale)")
    ax2.set_title("Local conditioning along the path\n(spikes = near-singularities at the rectangle's corners)")
    ax2.legend(fontsize=9)
    fig.suptitle("The beam Jacobian changes substantially along the rectangle's path too — "
                 "same mechanism as the triangle", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig5_jacobian_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig5_jacobian_evolution.png (sigma3 range {sv[:,2].min():.2e}-{sv[:,2].max():.2e}, "
          f"cond range {cond.min():.0f}-{cond.max():.0f})")
    return schedule


def fig6_jacobian_mismatch_vs_error(schedule, conditions, out: Path) -> None:
    if schedule is None:
        return
    n = schedule.shape[0]
    J0 = schedule[0, :, :6]
    mismatch = np.array([np.linalg.norm(schedule[k, :, :6] - J0, ord="fro") for k in range(n)])
    mismatch_pct = 100 * mismatch / np.linalg.norm(J0, ord="fro")

    lti_rep = median_rep(conditions["mpc_lti+FF (fixed)"])
    ltv_rep = median_rep(conditions["mpc_ltv+FF (fixed)"])
    lti_a, ltv_a = read_traj(lti_rep), read_traj(ltv_rep)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})
    ax1.plot(np.arange(n), mismatch_pct, color="#9467bd")
    ax1.set_ylabel("$\\|J(k)-J(0)\\|_F$ / $\\|J(0)\\|_F$  (%)")
    ax1.set_title("Frozen-Jacobian mismatch grows along the path (this is what mpc_lti is blind to)")

    ax2.plot(lti_a["ref_index"], lti_a["err_norm_mm"], color=COLORS["mpc_lti+FF (fixed)"],
              lw=1.5, label="mpc_lti+FF (frozen Jacobian)")
    ax2.plot(ltv_a["ref_index"], ltv_a["err_norm_mm"], color=COLORS["mpc_ltv+FF (fixed)"],
              lw=1.5, label="mpc_ltv+FF (relinearised each step)")
    ax2.set_xlabel("reference sample index (path progress)")
    ax2.set_ylabel("tracking error |e| (mm)")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig6_jacobian_mismatch_vs_error.png")
    plt.close(fig)
    print("wrote fig6_jacobian_mismatch_vs_error.png "
          f"(mismatch grows to {mismatch_pct.max():.0f}% of ||J(0)||)")


def fig7_controller_fingerprint(conditions, out: Path) -> None:
    names = ["inv+FF", "mpc_lti+FF (fixed)", "mpc_ltv+FF (fixed)"]
    sat, jerk, nullfrac = [], [], []
    for name in names:
        rep = median_rep(conditions[name])
        rows = read_rows(rep)
        delta = np.array([r["q_target_delta_rad"] for r in rows])
        clipped = np.any(np.abs(delta) >= MAX_JOINT_STEP_RAD - 1e-6, axis=1)
        sat.append(100 * clipped.mean())

        corr6 = np.array([r["u0_correction"][:6] for r in rows])
        j = np.linalg.norm(np.diff(delta, axis=0), axis=1)
        jerk.append(1e3 * j.mean())

        jac = json.loads((rep / "frozen_jacobian.json").read_text())
        J6 = np.array(jac["j_full_3x7"])[:, :6]
        Jpinv = np.linalg.pinv(J6)
        P_null = np.eye(6) - Jpinv @ J6
        total = np.linalg.norm(corr6, axis=1)
        null_comp = np.linalg.norm((P_null @ corr6.T).T, axis=1)
        frac = np.divide(null_comp, total, out=np.zeros_like(total), where=total > 1e-9)
        nullfrac.append(100 * frac.mean())

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        (sat, "% ticks with >=1 joint\nat the step-size limit", "step-limit saturation"),
        (jerk, "mean |applied step(k) - step(k-1)|\n(mrad)", "tick-to-tick jerk"),
        (nullfrac, "% of command magnitude in the\nEXACT nullspace of J (zero tip effect)", "nullspace usage"),
    ]
    cols = [COLORS[n] for n in names]
    for ax, (vals, ylabel, title) in zip(axs, panels):
        ax.bar(names, vals, color=cols, edgecolor="black")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Controller behaviour fingerprint, 10x15mm rectangle — checking whether the same "
                 "explanation (or non-explanation) for MPC's edge holds on a second shape",
                 y=1.06, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out / "fig7_controller_fingerprint.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig7_controller_fingerprint.png:",
          {n: (s, j, nf) for n, s, j, nf in zip(names, sat, jerk, nullfrac)})


def fig8_correction_vs_error(conditions, out: Path) -> None:
    def hold_series(run_dir: Path):
        rows = read_rows(run_dir)
        corr6 = np.array([r["u0_correction"][:6] for r in rows])
        total = np.linalg.norm(corr6, axis=1)
        err = np.array([r["error_norm_mm"] for r in rows])
        hold = np.array([r["terminal_hold"] for r in rows])
        return total[hold], err[hold]

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    markers = ["o", "s", "^"]
    for ax, key, label in [(axs[0], "inv+FF", "inv+FF"),
                            (axs[1], "mpc_ltv+FF (fixed)", "mpc_ltv+FF (fixed)")]:
        reps = conditions[key]
        all_corr, all_err = [], []
        per_rep_r = []
        for i, rep in enumerate(reps):
            corr, err = hold_series(rep)
            all_corr.append(corr); all_err.append(err)
            per_rep_r.append(np.corrcoef(corr, err)[0, 1] if len(corr) > 1 else float("nan"))
            ax.scatter(err, corr, s=16, marker=markers[i % len(markers)],
                       color=COLORS[key], alpha=0.6, label=f"rep{i+1} (r={per_rep_r[-1]:+.2f})")
        pooled_r = np.corrcoef(np.concatenate(all_corr), np.concatenate(all_err))[0, 1]
        ax.set_title(f"{label}\npooled corr = {pooled_r:+.2f}  (per-rep: "
                     + ", ".join(f"{r:+.2f}" for r in per_rep_r) + ")", fontsize=10)
        ax.set_xlabel("tracking error at hold (mm)")
        ax.set_ylabel("|joint correction| (rad)")
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("inv+FF is cleanly proportional in every rep; mpc_ltv+FF is NEGATIVE in every rep — "
                 "the state_tracking_weight fix does not fully generalise to this shape", y=1.04)
    fig.tight_layout()
    fig.savefig(out / "fig8_correction_vs_error.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig8_correction_vs_error.png (all 3 reps pooled per controller)")


# =============================================================================
def main() -> None:
    out = OUT_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    conditions = {name: valid_reps(patterns) for name, patterns in RUN_GLOBS.items()}
    for name, reps in conditions.items():
        print(f"{name:22s}: {len(reps)} valid reps")
        if not reps:
            raise SystemExit(f"no valid (path_complete) runs found for {name!r}")

    fig1_tip_paths(conditions, out)
    fig2_error_vs_time(conditions, out)
    fig3_summary_bars(conditions, out)
    fig4_jacobian_svd(conditions, out)
    schedule = fig5_jacobian_evolution(out)
    fig6_jacobian_mismatch_vs_error(schedule, conditions, out)
    fig7_controller_fingerprint(conditions, out)
    fig8_correction_vs_error(conditions, out)

    print("\nALL FIGURES WRITTEN TO", out)


if __name__ == "__main__":
    main()
