"""Error-vs-arc-length figures with 95% CI bands, companion to publication_figures.py.
x = insertion_length_m (arc length along the lumen, mm). y = error_norm_mm.
Per run: bin error_norm_mm into 0.5mm arc-length cells, average within each
cell (handles small tick-to-tick non-monotonicity in insertion_length_m).
Per group: mean + 95% CI (1.96*std/sqrt(n_valid)) across the 10 reps per cell,
masking cells with n_valid < 3.
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
S_GRID = np.arange(25, 90, 0.5)  # mm
R_C = (55.0, 65.0)  # mm, contact-sensitive region used throughout this session


def load_run_curve(run_name):
    d = BASE + "/" + run_name
    s_vals, e_vals = [], []
    with open(d + "/path_follow.jsonl") as f:
        for line in f:
            r = json.loads(line)
            L = r.get("insertion_length_m")
            e = r.get("error_norm_mm")
            if L is not None and e is not None:
                s_vals.append(L * 1000.0)
                e_vals.append(e)
    s_vals = np.array(s_vals)
    e_vals = np.array(e_vals)
    # bin into S_GRID cells, average within each cell
    curve = np.full(len(S_GRID), np.nan)
    cell_idx = np.clip(np.round((s_vals - S_GRID[0]) / 0.5).astype(int), 0, len(S_GRID) - 1)
    for i in range(len(S_GRID)):
        mask = cell_idx == i
        if mask.any():
            curve[i] = np.mean(e_vals[mask])
    return curve


def group_mean_ci(gname):
    runs = GROUPS[gname]
    curves = np.array([load_run_curve(r) for r in runs])  # (n_reps, n_cells)
    n_valid = np.sum(~np.isnan(curves), axis=0)
    mean = np.nanmean(curves, axis=0)
    std = np.nanstd(curves, axis=0, ddof=1)
    ci = 1.96 * std / np.sqrt(np.maximum(n_valid, 1))
    mean = np.where(n_valid >= 3, mean, np.nan)
    ci = np.where(n_valid >= 3, ci, np.nan)
    return mean, ci, n_valid


print("=== computing arc-length curves for all groups ===")
CURVES = {}
for gname in GROUPS:
    mean, ci, n_valid = group_mean_ci(gname)
    CURVES[gname] = dict(mean=mean, ci=ci, n_valid=n_valid)
    n_covered = np.sum(n_valid >= 3)
    print(f"  {gname:20s} cells with n>=3: {n_covered}/{len(S_GRID)}  "
          f"s-range: [{S_GRID[n_valid>=1].min() if (n_valid>=1).any() else np.nan:.1f},"
          f"{S_GRID[n_valid>=1].max() if (n_valid>=1).any() else np.nan:.1f}]")

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def plot_pair(ax, gname_a, gname_b, label_a, label_b, color_a, color_b, log=False):
    for gname, label, color in [(gname_a, label_a, color_a), (gname_b, label_b, color_b)]:
        c = CURVES[gname]
        mean, ci = c["mean"], c["ci"]
        valid = ~np.isnan(mean)
        ax.plot(S_GRID[valid], mean[valid], color=color, linewidth=2.0, label=label)
        ax.fill_between(S_GRID[valid], (mean - ci)[valid], (mean + ci)[valid],
                         color=color, alpha=0.18, linewidth=0)
    ax.axvspan(R_C[0], R_C[1], color="#999999", alpha=0.12, zorder=0)
    ax.annotate("region C", xy=((R_C[0] + R_C[1]) / 2, 1), xycoords=("data", "axes fraction"),
                xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9.5,
                color="#777777", style="italic")
    ax.set_xlabel("Arc length along lumen (mm)", fontsize=12)
    ax.set_ylabel("Tip error (mm)" + (" [log]" if log else ""), fontsize=12)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left", fontsize=10.5, frameon=False)


# ---- Fig1b: open-loop vs closed-loop ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "OL_dipole", "MPC_N15_dipole",
          "Open-loop (feedforward only)", "Closed-loop MPC (N=15)",
          COLOR_OL, COLOR_A, log=True)
ax.set_title("Where closed-loop feedback corrects open-loop drift", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig1b_openloop_vs_closedloop_arclength.png", dpi=220)
print("saved fig1b")

# ---- Fig2b: MPC vs inverse-Jacobian ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "InvJac_dipole", "MPC_N15_dipole",
          "Inverse-Jacobian controller", "MPC (scheduled Jacobian)",
          COLOR_B, COLOR_A, log=True)
ax.set_title("Where MPC's constraint handling outperforms\nthe inverse-Jacobian controller", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig2b_mpc_vs_invjac_arclength.png", dpi=220)
print("saved fig2b")

# ---- Fig3b: scheduled vs frozen Jacobian ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "Frozen_dipole", "MPC_N15_dipole",
          "Frozen Jacobian (fixed at t=0)", "Scheduled Jacobian (re-linearized)",
          COLOR_B, COLOR_A, log=False)
ax.set_title("Where a frozen Jacobian loses tracking\nrelative to a continuously-updated schedule", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig3b_scheduled_vs_frozen_arclength.png", dpi=220)
print("saved fig3b")

# ---- Fig4b: horizon N=15 vs N=1 ----
fig, ax = plt.subplots(figsize=(8.0, 5.2))
plot_pair(ax, "MPC_N1_dipole", "MPC_N15_dipole",
          "N = 1", "N = 15",
          COLOR_B, COLOR_A, log=False)
ax.set_title("Where the longer prediction horizon\nmost reduces tracking error", fontsize=13, weight="bold", pad=12)
fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.14)
fig.savefig(f"{FIGDIR}/fig4b_horizon_n15_vs_n1_arclength.png", dpi=220)
print("saved fig4b")

# ---- Fig5b: dipole-align vs no-dipole, both horizons ----
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
fig.suptitle("Where dipole-alignment's tracking cost shows up, by horizon", fontsize=13.5, weight="bold", y=0.99)
fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.08)
fig.savefig(f"{FIGDIR}/fig5b_dipole_vs_nodipole_arclength.png", dpi=220)
print("saved fig5b")

print("\nAll spatial figures saved to", FIGDIR)
