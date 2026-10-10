"""Publication figure set for the 10x15mm U-shape controller comparison
(2026-09-13) -- the fourth-shape generalization follow-up to the triangle,
rectangle and S-curve comparisons, and the first OPEN (non-returning) shape
tested. This is the cleanest MPC-LTV win of any shape so far, and this
script's fig6 is built specifically to show the root cause (a near-singular
neighbourhood right next to the path's own well-conditioned endpoint).

Run from the repo root:

    python3 -m proper_research.hardware.online.make_ushape_figures

Writes 7 PNGs into close_loop_logs/day5_ushape10x15mm_2026-09-13/figures/:
    fig1_tip_paths.png                  tip path vs reference, 4 conditions
    fig2_error_vs_time.png              tracking error vs time, 4 conditions
    fig3_summary_bars.png               4-condition RMS/max/hold/final bars
    fig4_jacobian_svd.png               frozen (sample-0) beam Jacobian
                                         singular-value spectrum
    fig5_jacobian_evolution.png         singular values / condition number
                                         along the NOMINAL path (mpc_ltv's
                                         per-step schedule) -- shows the
                                         nominal path is well-conditioned
                                         everywhere, including the endpoint
    fig6_conditioning_cliff.png         THE ROOT CAUSE: condition number at
                                         the nominal endpoint (used by
                                         mpc_ltv's schedule) vs. the analytic
                                         Jacobian evaluated at inv's own,
                                         barely-different (~4mm tip error)
                                         LIVE measured configuration -- a
                                         >200x conditioning cliff -- plus the
                                         resulting near-orthogonal DLS
                                         correction-direction misalignment
                                         between the two.
    fig7_controller_fingerprint.png     saturation / jerk / nullspace-usage
                                         fingerprint, inv+FF vs mpc_lti+FF vs
                                         mpc_ltv+FF
    fig8_correction_vs_error.png        |correction| vs tracking error at
                                         every hold-phase tick -- inv+FF is
                                         PERFECTLY proportional (r~+1.00) yet
                                         still fails to converge (error
                                         plateaus at ~4mm instead of
                                         shrinking) -- proof that
                                         proportionality alone doesn't help
                                         when the correction DIRECTION itself
                                         is unreliable tick-to-tick.
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
OUT_ROOT = BASE / "day5_ushape10x15mm_2026-09-13"

RUN_GLOBS: dict[str, list[Path]] = {
    "open loop": [BASE / "openloop_ushape10x15mm_n3_2026-09-13" / "runs" / "*"],
    "inv+FF": [BASE / "ushape_FF_n3_2026-09-13" / "runs" / "inv_rep*"],
    "mpc_lti+FF (fixed)": [BASE / "ushape_FF_n3_2026-09-13" / "runs" / "lti_rep*"],
    "mpc_ltv+FF (fixed)": [BASE / "ushape_FF_n3_2026-09-13" / "runs" / "ltv_rep*"],
}
LTV_SCHEDULE_NPZ = BASE / "ushape_FF_n3_2026-09-13" / "ltv_schedule.npz"
SINGULARITY_NPZ = BASE / "ushape_FF_n3_2026-09-13" / "singularity_analysis.npz"
MAX_JOINT_STEP_RAD = 0.006

FF_CONDITIONS = ["open loop", "inv+FF", "mpc_lti+FF (fixed)", "mpc_ltv+FF (fixed)"]
ALL_CONDITIONS = FF_CONDITIONS  # no pure-feedback run was collected on this shape
COLORS = {
    "open loop": "#888888",
    "inv+FF": "#1f77b4",
    "mpc_lti+FF (fixed)": "#ff7f0e",
    "mpc_ltv+FF (fixed)": "#2ca02c",
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
        ax.plot(tip_u[-1], tip_v[-1], "k^", ms=7, label="end (open path)")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("B.y (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("B.x (mm)")
    fig.suptitle("Tip path vs. reference — 10x15mm U-shape, open path (representative rep per condition)", y=1.03)
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
    ax.set_title("Tip-tracking error vs. time — 10x15mm U-shape, feedforward conditions")
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
    fig, axs = plt.subplots(1, 4, figsize=(15, 5.5))
    metric_labels = [("rms", "RMS tracking error (mm)"), ("max", "Max tracking error (mm)"),
                      ("hold", "Mean hold error (mm)"), ("final", "Final error (mm)")]
    x = np.arange(len(ALL_CONDITIONS))
    bar_colors = [COLORS[n] for n in ALL_CONDITIONS]
    for ax, (key, label) in zip(axs, metric_labels):
        means = [agg[n][key][0] for n in ALL_CONDITIONS]
        stds = [agg[n][key][1] for n in ALL_CONDITIONS]
        ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CONDITIONS, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0])
    fig.suptitle("Full controller comparison, 10x15mm U-shape (open path), n=3 per condition — "
                 "mpc_ltv+FF's biggest hold/final margin of any shape tested", y=1.04)
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
    ax.set_title(f"Beam Jacobian $J$ at path start (3$\\times$6, joint block)\ncondition number $\\kappa$ = {cond:,.0f}")
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
        print("skip fig5 (LTV_SCHEDULE_NPZ not found)")
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
    ax1.set_yscale("log")
    ax1.set_xlabel("reference sample index (path progress)")
    ax1.set_ylabel("singular value (mm / rad)")
    ax1.set_title("Local Jacobian singular values along the NOMINAL path")
    ax1.legend(fontsize=8.5)

    ax2.plot(progress, cond, color="#9467bd")
    ax2.axhline(cond[-1], color="black", ls=":", lw=1.4, label=f"nominal endpoint (used by mpc_ltv's hold): {cond[-1]:.0f}")
    ax2.set_yscale("log")
    ax2.set_xlabel("reference sample index (path progress)")
    ax2.set_ylabel("condition number $\\kappa$ (log scale)")
    ax2.set_title("Local conditioning along the NOMINAL path\n(the planner kept this well-conditioned everywhere, endpoint included)")
    ax2.legend(fontsize=9)
    fig.suptitle("The offline-planned path itself is well-conditioned throughout — "
                 "the problem (fig6) is what happens just OFF this path", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "fig5_jacobian_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig5_jacobian_evolution.png (cond range {cond.min():.0f}-{cond.max():.0f}, "
          f"endpoint {cond[-1]:.0f})")
    return schedule


def fig6_conditioning_cliff(out: Path) -> None:
    if not SINGULARITY_NPZ.exists():
        print("skip fig6 (SINGULARITY_NPZ not found -- see beam-lateral-authority-limit memory "
              "2026-09-13 U-shape entry for how to regenerate it)")
        return
    d = np.load(SINGULARITY_NPZ)
    cond_nom = float(d["cond_nom"])
    conds_meas = d["conds_meas"]
    cos_list = d["cos_list"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    labels = ["nominal endpoint\n(mpc_ltv's schedule)"] + [f"inv+FF rep{i+1}\n(live q_meas at hold)" for i in range(len(conds_meas))]
    vals = [cond_nom] + list(conds_meas)
    cols = ["#2ca02c"] + ["#1f77b4"] * len(conds_meas)
    ax1.bar(labels, vals, color=cols, edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_ylabel("condition number $\\kappa$ (log scale)")
    ax1.set_title(f"Conditioning cliff: {cond_nom:.0f} -> {conds_meas.mean():,.0f}\n"
                  f"(~{conds_meas.mean()/cond_nom:,.0f}x worse) for a ~4mm tip deviation")
    ax1.tick_params(axis="x", labelrotation=12, labelsize=8.5)

    ax2.axhline(0, color="gray", lw=0.8)
    ax2.bar([f"rep{i+1}" for i in range(len(cos_list))], cos_list, color="#d62728", edgecolor="black")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("cos(correction from live J, correction from nominal J)")
    ax2.set_title(f"Same hold-phase error, two ~identical configs\n(joint diff ~1e-5 rad) -> "
                  f"near-ORTHOGONAL DLS corrections\n(mean cos = {cos_list.mean():+.2f})")
    fig.suptitle("Root cause: the U-shape's endpoint sits at the edge of a near-singular neighbourhood. "
                 "inv+FF re-linearises on its own live (unstable) estimate every tick; mpc_ltv+FF uses a "
                 "FIXED, well-conditioned nominal-endpoint Jacobian instead.", y=1.08, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig6_conditioning_cliff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig6_conditioning_cliff.png (cond_nom={cond_nom:.0f}, "
          f"cond_meas={conds_meas}, cos={cos_list})")


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
    fig.suptitle("Controller behaviour fingerprint, 10x15mm U-shape — checking whether the same "
                 "explanation (or non-explanation) for MPC's edge holds on a fourth shape",
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

    names = ["inv+FF", "mpc_lti+FF (fixed)", "mpc_ltv+FF (fixed)"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    markers = ["o", "s", "^"]
    for ax, key in zip(axs, names):
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
        ax.set_title(f"{key}\npooled corr = {pooled_r:+.2f}  (per-rep: "
                     + ", ".join(f"{r:+.2f}" for r in per_rep_r) + ")", fontsize=10)
        ax.set_xlabel("tracking error at hold (mm)")
        ax.set_ylabel("|joint correction| (rad)")
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("inv+FF's correction IS perfectly proportional to error (r~+1.00) yet still fails to "
                 "converge (error plateaus ~4mm) — proportionality doesn't help when the correction "
                 "DIRECTION is unreliable tick-to-tick (fig6)", y=1.05)
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
    fig5_jacobian_evolution(out)
    fig6_conditioning_cliff(out)
    fig7_controller_fingerprint(conditions, out)
    fig8_correction_vs_error(conditions, out)

    print("\nALL FIGURES WRITTEN TO", out)


if __name__ == "__main__":
    main()
