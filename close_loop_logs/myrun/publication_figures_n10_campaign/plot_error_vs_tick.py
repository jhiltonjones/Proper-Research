"""Error-vs-control-tick figures with 95% CI bands, companion to
plot_error_vs_arclength.py. x = tick/step index (time-aligned from the start
of each run, tick 1 = first logged control step for every rep -- unlike the
arc-length plots, this directly answers "do the two conditions start from
the same place" since tick 1 vs tick 1 is a true like-for-like comparison,
with no spatial-binning mixing).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/jack/.claude/jobs/043c0b8b/tmp/figures")
from publication_figures import GROUPS, BASE, COLOR_A, COLOR_B, COLOR_OL

FIGDIR = "/home/jack/.claude/jobs/043c0b8b/tmp/figures"
MAX_TICK = 700  # covers the longest completing runs (~690 ticks for open-loop)


def load_run_tick_curve(run_name):
    d = BASE + "/" + run_name
    curve = np.full(MAX_TICK, np.nan)
    with open(d + "/path_follow.jsonl") as f:
        for i, line in enumerate(f):
            if i >= MAX_TICK:
                break
            r = json.loads(line)
            e = r.get("error_norm_mm")
            if e is not None:
                curve[i] = e
    return curve


def group_mean_ci(gname):
    runs = GROUPS[gname]
    curves = np.array([load_run_tick_curve(r) for r in runs])  # (n_reps, MAX_TICK)
    n_valid = np.sum(~np.isnan(curves), axis=0)
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(np.maximum(n_valid, 1))
    mean = np.where(n_valid >= 3, mean, np.nan)
    ci = np.where(n_valid >= 3, ci, np.nan)
    return mean, ci, n_valid


print("=== computing tick-aligned curves for all groups ===")
TICKS = np.arange(1, MAX_TICK + 1)
CURVES = {}
for gname in GROUPS:
    mean, ci, n_valid = group_mean_ci(gname)
    CURVES[gname] = dict(mean=mean, ci=ci, n_valid=n_valid)
    n_covered = np.sum(n_valid >= 3)
    print(f"  {gname:20s} ticks with n>=3: {n_covered}/{MAX_TICK}")

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def plot_pair(ax, gname_a, gname_b, label_a, label_b, color_a, color_b, log=False, xmax=None):
    for gname, label, color in [(gname_a, label_a, color_a), (gname_b, label_b, color_b)]:
        c = CURVES[gname]
        mean, ci = c["mean"], c["ci"]
        valid = ~np.isnan(mean)
        ax.plot(TICKS[valid], mean[valid], color=color, linewidth=2.0, label=label)
        ax.fill_between(TICKS[valid], (mean - ci)[valid], (mean + ci)[valid],
                         color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("Control tick (from start of run)", fontsize=12)
    ax.set_ylabel("Tip error (mm)" + (" [log]" if log else ""), fontsize=12)
    if log:
        ax.set_yscale("log")
    if xmax:
        ax.set_xlim(0, xmax)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper right", fontsize=10.5, frameon=False)


# ---- Fig1c: open-loop vs closed-loop ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "OL_dipole", "MPC_N15_dipole",
          "Open-loop (feedforward only)", "Closed-loop MPC (N=15)",
          COLOR_OL, COLOR_A, log=True)
ax.set_title("Open-loop vs closed-loop, aligned by control tick", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig1c_openloop_vs_closedloop_tick.png", dpi=220)
print("saved fig1c")

# ---- Fig2c: MPC vs inverse-Jacobian ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "InvJac_dipole", "MPC_N15_dipole",
          "Inverse-Jacobian controller", "MPC (scheduled Jacobian)",
          COLOR_B, COLOR_A, log=True, xmax=350)
ax.set_title("MPC vs inverse-Jacobian, aligned by control tick", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig2c_mpc_vs_invjac_tick.png", dpi=220)
print("saved fig2c")

# ---- Fig3c: scheduled vs frozen Jacobian ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "Frozen_dipole", "MPC_N15_dipole",
          "Frozen Jacobian (fixed at t=0)", "Scheduled Jacobian (re-linearized)",
          COLOR_B, COLOR_A, log=False, xmax=400)
ax.set_title("Scheduled vs frozen Jacobian, aligned by control tick", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig3c_scheduled_vs_frozen_tick.png", dpi=220)
print("saved fig3c")

# ---- Fig4c: horizon N=15 vs N=1 ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "MPC_N1_dipole", "MPC_N15_dipole",
          "N = 1", "N = 15",
          COLOR_B, COLOR_A, log=False)
ax.set_title("Horizon N=1 vs N=15, aligned by control tick", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig4c_horizon_n15_vs_n1_tick.png", dpi=220)
print("saved fig4c")

# ---- Fig5c: dipole-align vs no-dipole, both horizons ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)
plot_pair(axes[0], "MPC_N15_nodipole", "MPC_N15_dipole",
          "No-dipole", "Dipole-aligned",
          "#5ba3d9", COLOR_A, log=False)
axes[0].set_title("N = 15", fontsize=12.5, weight="bold")
plot_pair(axes[1], "MPC_N1_nodipole", "MPC_N1_dipole",
          "No-dipole", "Dipole-aligned",
          "#e07b6b", COLOR_B, log=False)
axes[1].set_title("N = 1", fontsize=12.5, weight="bold")
axes[1].set_ylabel("")
fig.suptitle("Dipole-alignment cost by horizon, aligned by control tick", fontsize=13.5, weight="bold", y=0.99)
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.08)
fig.savefig(f"{FIGDIR}/fig5c_dipole_vs_nodipole_tick.png", dpi=220)
print("saved fig5c")

print("\nAll tick-aligned figures saved to", FIGDIR)
