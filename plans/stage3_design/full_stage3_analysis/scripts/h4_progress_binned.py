"""Task 5 (H4 refinement): common-progress-matched bins.

Problem being fixed: cross-condition H4 averages in the final report (and in
h4_final_summary.csv) are whole-run means. Conditions that abort early
(e.g. 255mm MPC-J_NC, which never gets past s=64.25mm) do not sample the same
path-progress range as conditions that complete the full ~75mm path. A
whole-run mean therefore silently compares unlike portions of the trajectory.

Fix: bin every per-sample row (from the existing, already-computed
h4_final_mismatch_vs_Cstate.csv -- the corrected J_C^state-referenced table,
1636 guided-sampled ticks across all 8 closed-loop conditions) by path
progress s_ref_mm into shared bins, and report per-bin statistics per
condition. A condition with zero samples in a bin is reported as
"not observed" rather than silently excluded from a run-level average that
other conditions' numbers get compared against.

Read-only against the existing per-sample table; writes only a new table and
figure.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"

BINS = [
    (0.0, 28.5, "pre-contact"),
    (28.5, 45.0, "early contact"),
    (45.0, 55.0, "mid-contact"),
    (55.0, 60.0, "late common region"),
    (60.0, 64.0, "NC pre-failure region"),
]

df = pd.read_csv(f"{OUT}/tables/h4_final_mismatch_vs_Cstate.csv")

rows = []
for radius in (210, 255):
    for condition in ("mpc_C", "mpc_NC", "invjac_C", "invjac_NC"):
        sub_cond = df[(df.radius_intended_mm == radius) & (df.condition == condition)]
        max_s_observed = float(sub_cond["s_ref_mm"].max()) if len(sub_cond) else np.nan
        for lo, hi, label in BINS:
            sub = sub_cond[(sub_cond.s_ref_mm >= lo) & (sub_cond.s_ref_mm < hi)]
            n = len(sub)
            if n == 0:
                rows.append(dict(
                    radius_intended_mm=radius, condition=condition,
                    bin_lo_mm=lo, bin_hi_mm=hi, bin_label=label, n=0,
                    E_J_mean=np.nan, eps_u_mm_mean=np.nan, eps_u_mm_p95=np.nan,
                    gain_ratio_mean=np.nan, direction_angle_deg_mean=np.nan,
                    error_norm_mm_mean=np.nan,
                    not_observed=bool(max_s_observed < lo) if not np.isnan(max_s_observed) else True,
                    max_s_observed_mm=max_s_observed,
                ))
                continue
            rows.append(dict(
                radius_intended_mm=radius, condition=condition,
                bin_lo_mm=lo, bin_hi_mm=hi, bin_label=label, n=n,
                E_J_mean=float(sub["mismatch_sched_vs_Cstate"].mean()),
                eps_u_mm_mean=float(sub["eps_u_mm"].mean()),
                eps_u_mm_p95=float(sub["eps_u_mm"].quantile(0.95)),
                gain_ratio_mean=float(sub["gain_ratio"].mean()),
                direction_angle_deg_mean=float(sub["direction_angle_deg"].mean()),
                error_norm_mm_mean=float(sub["error_norm_mm"].mean()),
                not_observed=False,
                max_s_observed_mm=max_s_observed,
            ))

out_df = pd.DataFrame(rows)
out_df.to_csv(f"{OUT}/tables/h4_common_progress_bins.csv", index=False)
print(f"wrote {OUT}/tables/h4_common_progress_bins.csv ({len(out_df)} rows)")

# --- focus print: 255mm MPC-JC vs MPC-JNC, the primary H2-linked comparison ---
foc = out_df[(out_df.radius_intended_mm == 255) & (out_df.condition.isin(["mpc_C", "mpc_NC"]))]
print("\n255mm MPC-J_C vs MPC-J_NC, by common-progress bin:")
print(foc[["condition", "bin_label", "n", "E_J_mean", "eps_u_mm_mean", "gain_ratio_mean",
           "direction_angle_deg_mean", "error_norm_mm_mean", "not_observed"]].to_string(index=False))

# --- figure: 4 panels (E_J, eps_u, gain_ratio, direction_angle), bars grouped by bin, MPC only, both radii ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [
    ("E_J_mean", "relative Frobenius mismatch $E_J$"),
    ("eps_u_mm_mean", r"command-weighted error $\epsilon_u$ (mm)"),
    ("gain_ratio_mean", "prediction gain ratio"),
    ("direction_angle_deg_mean", "dominant-direction angle (deg)"),
]
bin_labels = [b[2] for b in BINS]
x = np.arange(len(bin_labels))
width = 0.19
conds = [(210, "mpc_C", "#1b7f3b", "210mm MPC-$J_C$"),
         (210, "mpc_NC", "#7fbf7f", "210mm MPC-$J_{NC}$"),
         (255, "mpc_C", "#1d4fb3", "255mm MPC-$J_C$"),
         (255, "mpc_NC", "#b3331d", "255mm MPC-$J_{NC}$")]

for ax, (col, ylabel) in zip(axes.flat, panels):
    for i, (radius, cond, color, label) in enumerate(conds):
        sub = out_df[(out_df.radius_intended_mm == radius) & (out_df.condition == cond)].set_index("bin_label").reindex(bin_labels)
        vals = sub[col].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1.5) * width, np.nan_to_num(vals, nan=0.0), width, color=color, label=label)
        for j, v in enumerate(vals):
            if np.isnan(v):
                ax.text(x[j] + (i - 1.5) * width, 0.02 * (np.nanmax(out_df[col]) if np.nanmax(out_df[col]) > 0 else 1),
                        "N/O", ha="center", va="bottom", fontsize=6, rotation=90, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[2]}\n({b[0]}-{b[1]}mm)" for b in BINS], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
axes.flat[0].legend(fontsize=7, loc="upper left")
fig.suptitle("H4: common-progress-matched schedule-vs-$J_C^{state}$ mismatch (N/O = not observed, run did not reach this bin)",
             fontsize=10)
fig.tight_layout()
f = f"{OUT}/figures/h4_progress_binned_accuracy.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"\nsaved {f}")
