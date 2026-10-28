"""Robustness check for H4's gain-ratio statistic (review issue #5).

gain_ratio = ||J_used_sched @ du|| / ||J_C^state @ du||  (per tick; see
h4_actual_recompute.py in the prior job's scripts for the original formula --
num = ||J_used_sched @ du||, den = ||J_C^state @ du||, du = u0*dt).

A MEAN of this ratio over samples can be dominated by a few ticks with a
near-zero denominator (near-zero commanded motion at that exact tick),
inflating the reported mean without reflecting a robust physical effect.
This script recomputes num/den directly (re-deriving them via live_jac,
since the existing per-sample table only stores the ratio, not num/den
separately) for the 255mm MPC-J_C / MPC-J_NC comparison, over the same
ticks/states h4_progress_binned.py already bins, and reports:
  - original mean gain ratio (for comparison)
  - median gain ratio (outlier-robust)
  - G_agg = sum(num)/sum(den) (aggregate ratio, insensitive to a few
    near-zero-denominator ticks blowing up an individual ratio)
  - the same three statistics after excluding samples whose denominator
    falls below a fixed tiny-motion threshold (0.01mm, chosen as a
    physically negligible one-tick predicted tip displacement -- roughly
    2 orders of magnitude below this dataset's typical du_norm-driven
    tip motions, which run ~0.1-1mm per tick)

Read-only against existing per-sample selection and live model evaluation;
writes only a new table/figure.
"""
import sys, time
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest
import loader
import live_jac

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
SCHED_C = f"{loader.REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy"
SCHED_NC = f"{loader.REPO}/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy"
DT = 0.1
TINY_MOTION_THRESHOLD_MM = 0.01

BINS = [
    (0.0, 28.5, "pre-contact"),
    (28.5, 45.0, "early contact"),
    (45.0, 55.0, "mid-contact"),
    (55.0, 60.0, "late common region"),
    (60.0, 64.0, "NC pre-failure region"),
]

t_start = time.time()
live_jac.get_context()
print(f"[robustness] context ready at {time.time()-t_start:.0f}s", flush=True)

selection = pd.read_csv(f"{OUT}/tables/h4_final_mismatch_vs_Cstate.csv")
selection = selection[(selection.radius_intended_mm == 255) & (selection.condition.isin(["mpc_C", "mpc_NC"]))].copy()
print(f"[robustness] {len(selection)} rows to recompute (255mm mpc_C+mpc_NC)", flush=True)

cache = {}
sched_cache = {}
rows = []
for i, row in selection.iterrows():
    dirname = row["dirname"]
    if dirname not in cache:
        rm = next(r for r in manifest.RUNS if r["dirname"] == dirname)
        cache[dirname] = (loader.enrich_run(rm), rm["jacobian"])
    (out, meta), jac_type = cache[dirname]

    if jac_type not in sched_cache:
        sched_cache[jac_type] = np.load(SCHED_C if jac_type == "contact" else SCHED_NC)
    sched = sched_cache[jac_type]

    k = int(row["tick"])
    state = np.concatenate([out["q_meas_rad"][k], [out["insertion_length_m"][k]]])
    ref_idx = int(np.clip(out["ref_index"][k] + 1, 0, sched.shape[0] - 1))
    J_used_sched = sched[ref_idx]
    J_C_state = live_jac.live_jacobian(state, True)

    du = out["u0"][k] * DT
    num_mm = float(np.linalg.norm(J_used_sched @ du) * 1000.0)
    den_mm = float(np.linalg.norm(J_C_state @ du) * 1000.0)
    gain_ratio_recomputed = num_mm / den_mm if den_mm > 1e-9 else np.nan

    rows.append(dict(
        dirname=dirname, condition=row["condition"], rep=row.get("rep"), tick=k,
        s_ref_mm=float(row["s_ref_mm"]), error_norm_mm=float(row["error_norm_mm"]),
        num_mm=num_mm, den_mm=den_mm, gain_ratio_recomputed=gain_ratio_recomputed,
        gain_ratio_original=float(row["gain_ratio"]),
    ))
    if (i + 1) % 50 == 0:
        print(f"[robustness] {i+1}/{len(selection)} ({time.time()-t_start:.0f}s elapsed)", flush=True)

per_sample = pd.DataFrame(rows)
per_sample.to_csv(f"{OUT}/tables/h4_gain_ratio_robustness_per_sample.csv", index=False)
print(f"\n[robustness] recompute done in {time.time()-t_start:.0f}s", flush=True)

# sanity: recomputed vs original gain_ratio should match closely (same formula, same du/state)
match_err = (per_sample["gain_ratio_recomputed"] - per_sample["gain_ratio_original"]).abs()
print(f"[robustness] sanity: |recomputed - original| gain_ratio, mean={match_err.mean():.4f}, "
      f"max={match_err.max():.4f} (should be ~0, confirms same formula/state reproduced)")

# --- bin and compute robustness statistics ---
summary_rows = []
for condition in ("mpc_C", "mpc_NC"):
    sub_cond = per_sample[per_sample.condition == condition]
    den_p10_whole = float(sub_cond["den_mm"].quantile(0.10))
    for lo, hi, label in BINS:
        sub = sub_cond[(sub_cond.s_ref_mm >= lo) & (sub_cond.s_ref_mm < hi)]
        n = len(sub)
        if n == 0:
            summary_rows.append(dict(condition=condition, bin_label=label, bin_lo_mm=lo, bin_hi_mm=hi,
                                      n=0, n_excluded=0))
            continue
        mean_orig = float(sub["gain_ratio_original"].mean())
        median_r = float(sub["gain_ratio_recomputed"].median())
        mean_r = float(sub["gain_ratio_recomputed"].mean())
        g_agg = float(sub["num_mm"].sum() / sub["den_mm"].sum())

        keep = sub[sub["den_mm"] >= TINY_MOTION_THRESHOLD_MM]
        n_excluded = n - len(keep)
        if len(keep) > 0:
            mean_r_ex = float(keep["gain_ratio_recomputed"].mean())
            median_r_ex = float(keep["gain_ratio_recomputed"].median())
            g_agg_ex = float(keep["num_mm"].sum() / keep["den_mm"].sum())
        else:
            mean_r_ex = median_r_ex = g_agg_ex = np.nan

        summary_rows.append(dict(
            condition=condition, bin_label=label, bin_lo_mm=lo, bin_hi_mm=hi, n=n,
            mean_gain_ratio_original=round(mean_orig, 3),
            mean_gain_ratio_recomputed=round(mean_r, 3),
            median_gain_ratio=round(median_r, 3),
            G_agg=round(g_agg, 3),
            n_excluded_tiny_motion=n_excluded,
            den_p10_whole_condition_mm=round(den_p10_whole, 5),
            mean_gain_ratio_excl_tiny=round(mean_r_ex, 3) if not np.isnan(mean_r_ex) else np.nan,
            median_gain_ratio_excl_tiny=round(median_r_ex, 3) if not np.isnan(median_r_ex) else np.nan,
            G_agg_excl_tiny=round(g_agg_ex, 3) if not np.isnan(g_agg_ex) else np.nan,
        ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(f"{OUT}/tables/h4_gain_ratio_robustness.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== H4 gain-ratio robustness, 255mm MPC-J_C vs MPC-J_NC, by bin ===")
print(summary.to_string(index=False))

# --- figure: original mean vs median vs G_agg vs excl-tiny mean, per bin, mpc_NC only (the condition of interest) ---
fig, ax = plt.subplots(figsize=(10, 5.5))
nc = summary[summary.condition == "mpc_NC"].reset_index(drop=True)
x = np.arange(len(nc))
width = 0.19
metrics = [
    ("mean_gain_ratio_original", "mean (original)", "#b3331d"),
    ("median_gain_ratio", "median", "#1b7f3b"),
    ("G_agg", "$G_{agg}$ (sum/sum)", "#1d4fb3"),
    ("mean_gain_ratio_excl_tiny", f"mean, den>={TINY_MOTION_THRESHOLD_MM}mm excl.", "#7a1fa2"),
]
for i, (col, label, color) in enumerate(metrics):
    ax.bar(x + (i - 1.5) * width, nc[col].to_numpy(dtype=float), width, color=color, label=label)
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([f"{r.bin_label}\n({r.bin_lo_mm:g}-{r.bin_hi_mm:g}mm, n={r.n})" for r in nc.itertuples()], fontsize=8)
ax.set_ylabel("gain ratio")
ax.set_title("H4 gain-ratio robustness: 255mm MPC-$J_{NC}$, by common-progress bin")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
f = f"{OUT}/figures/h4_gain_ratio_robustness.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"\nsaved {f}")
