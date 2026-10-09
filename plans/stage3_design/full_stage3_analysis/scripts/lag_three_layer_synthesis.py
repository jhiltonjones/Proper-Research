"""§5.4 synthesis figure: three-layer causal story in one plot.

Assembles data already computed and saved by lag_chronology_analysis.py and
effective_gain_and_baseline_analysis.py (no new live-model evaluations) into
a single three-row figure, one row per layer of the proposed mechanism:

  Layer 1 (top)    -- the inherent Kp=0.6 proportional-tracking lag floor,
                       shared by both Jacobian pairings (e_lag_measured vs
                       the v_s*dt/Kp baseline).
  Layer 2 (middle) -- the no-contact Jacobian's realized longitudinal gain
                       k_parallel dropping below its nominal Kp=0.6 value
                       specifically post-contact, while the contact pairing
                       stays near nominal.
  Layer 3 (bottom) -- the downstream redundant-configuration-drift chain:
                       null-space energy E_N rising as the x_max workspace
                       margin collapses to the tcp_out_of_workspace exit.

Contact onset (s=28.5mm, the contact model's own prediction, used
throughout this report) and each pairing's mean workspace-exit s are marked
as vertical lines in every row.

Read-only against existing tables; writes only this one new figure.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
COLOR = {"selective_contact": "#1b7f3b", "selective_nocontact": "#b3331d"}
LABEL = {"selective_contact": "contact Jacobian $J_C$", "selective_nocontact": "no-contact Jacobian $J_{NC}$"}
CONTACT_ONSET_MM = 28.5

excess = pd.read_csv(f"{OUT}/tables/excess_lag_baseline.csv")
kpar = pd.read_csv(f"{OUT}/tables/k_parallel_k_perp_selective.csv")
en = pd.read_csv(f"{OUT}/tables/lag_chronology_EN_strided.csv")
pertick = pd.read_csv(f"{OUT}/tables/lag_chronology_per_tick.csv")
onset = pd.read_csv(f"{OUT}/tables/lag_chronology_onset_summary.csv")

exit_s_mean = onset.groupby("pairing")["final_s_mm"].mean()

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

# --- Layer 1: inherent Kp=0.6 lag floor ---
ax = axes[0]
baseline_plotted = False
for pairing, color in COLOR.items():
    sub = excess[excess.pairing == pairing].sort_values("s_mm")
    for rep, d in sub.groupby("rep"):
        d = d.sort_values("s_mm")
        ax.plot(d.s_mm, d.e_lag_measured_mm, color=color, lw=0.9, alpha=0.75,
                label=f"{LABEL[pairing]}, measured" if rep == sub.rep.iloc[0] else None)
    if not baseline_plotted:
        # the v_s*dt/Kp baseline is a property of the shared reference path and Kp=0.6,
        # not of the Jacobian -- plot once using either pairing's own rep-1 series
        rep1 = sub[sub.rep == sub.rep.iloc[0]].sort_values("s_mm")
for pairing in COLOR:
    sub = excess[excess.pairing == pairing]
    rep1 = sub[sub.rep == sorted(sub.rep.unique())[0]].sort_values("s_mm")
    ax.plot(rep1.s_mm, rep1.e_lag_expected_dt_mm, color="black", lw=1.4, ls="--",
            label=r"theoretical baseline $v_s\Delta t/K_p$ ($K_p=0.6$, shared)" if pairing == "selective_contact" else None)
    break
ax.set_ylabel(r"$e_{lag}$ (mm)", fontsize=10)
ax.set_title("Layer 1 -- inherent $K_p=0.6$ tracking-lag floor (shared by both Jacobians)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_ylim(0, 6)

# --- Layer 2: effective longitudinal gain ---
ax = axes[1]
for pairing, color in COLOR.items():
    sub = kpar[kpar.pairing == pairing].sort_values("s_mm")
    for rep, d in sub.groupby("rep"):
        d = d.sort_values("s_mm")
        ax.plot(d.s_mm, d.k_parallel, color=color, lw=0.9, alpha=0.6, marker="o", ms=2,
                label=LABEL[pairing] if rep == sub.rep.iloc[0] else None)
ax.axhline(0.6, color="black", lw=1.2, ls="--", label=r"nominal $K_p=0.6$")
ax.set_ylabel(r"$k_\parallel$ (realized longitudinal gain)", fontsize=10)
ax.set_title("Layer 2 -- no-contact Jacobian reduces realized gain post-contact ($\\approx$30% drop)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_ylim(-0.5, 1.2)

# --- Layer 3: redundant drift -> workspace exit (twin axis) ---
ax = axes[2]
ax2 = ax.twinx()
for pairing, color in COLOR.items():
    sub = en[en.pairing == pairing].sort_values("s_mm")
    for rep, d in sub.groupby("rep"):
        d = d.sort_values("s_mm")
        ax.plot(d.s_mm, d.E_N, color=color, lw=1.0, alpha=0.7,
                label=f"{LABEL[pairing]}, $E_N$" if rep == sub.rep.iloc[0] else None)
    sub_m = pertick[pertick.pairing == pairing].sort_values("s_mm")
    for rep, d in sub_m.groupby("rep"):
        d = d.sort_values("s_mm")
        ax2.plot(d.s_mm, d.margin_x_max_mm, color=color, lw=1.0, alpha=0.35, ls=":")
ax.set_ylabel(r"$E_N$ (null-space energy fraction)", fontsize=10)
ax2.set_ylabel(r"$h_{x_{max}}$ margin (mm, dotted)", fontsize=10)
ax2.axhline(0, color="gray", lw=0.8, ls="-")
ax.set_title("Layer 3 -- redundant-configuration drift ($E_N$, solid) collapses the workspace margin ($h_{x_{max}}$, dotted) to exit", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlabel("path progress $s$ (mm)", fontsize=10)

for ax in axes:
    ax.axvline(CONTACT_ONSET_MM, color="gray", lw=1.0, ls="-.", alpha=0.7)
    for pairing, color in COLOR.items():
        ax.axvline(exit_s_mean[pairing], color=color, lw=1.0, ls=":", alpha=0.5)
axes[0].text(CONTACT_ONSET_MM + 0.5, 5.6, "contact onset", fontsize=7, color="gray", rotation=90, va="top")

fig.suptitle(
    "Three-layer mechanism: $K_p$=0.6 lag floor $\\to$ $J_{NC}$ gain reduction amplifies it $\\to$ redundant drift ends in workspace exit\n"
    "(selective-gate inverse-Jacobian control, 255mm floor; both pairings exit on the same $x_{max}$ face)",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
f = f"{OUT}/figures/lag_three_layer_synthesis.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"saved {f}")
