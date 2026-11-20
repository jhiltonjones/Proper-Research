"""Section 8 plots: tracking/completion, clearance vs threshold,
forecast error by lead time (A vs B), adaptation accept/reject trace
for one representative E run. Minimal set covering the explicitly
requested categories; reads only the already-computed CSVs/JSON plus
one direct re-read of E's own predicted_beam_positions.jsonl for the
adaptation trace (g_hat/g_used aren't aggregated anywhere else).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jack/Proper-Research"
LOGS = os.path.join(ROOT, "close_loop_logs")
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

rows = json.load(open(os.path.join(OUT_DIR, "per_run_metrics.json")))

# --- Plot 1: per-run active-tracking RMSE by condition, all points shown ---
fig, ax = plt.subplots(figsize=(8, 5))
conds = ["A", "B", "C", "D", "E"]
colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red", "E": "tab:purple"}
for i, cond in enumerate(conds):
    vals = [r["active_tracking_rmse_mm"] for r in rows if r["condition"] == cond]
    completed = [r["completed"] for r in rows if r["condition"] == cond]
    xs = [i + np.random.uniform(-0.08, 0.08) for _ in vals]
    for x, v, c in zip(xs, vals, completed):
        marker = "o" if c else "x"
        ax.scatter(x, v, color=colors[cond], marker=marker, s=60, zorder=3)
    if vals:
        ax.hlines(np.mean(vals), i - 0.2, i + 0.2, color=colors[cond], lw=2, zorder=2)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds)
ax.set_ylabel("active-motion tracking RMSE (mm)")
ax.set_title("Per-run active-motion tracking RMSE by condition\n(each point = 1 run; x = aborted run; bar = condition mean)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot1_tracking_rmse_by_condition.png"), dpi=110)
plt.close(fig)

# --- Plot 2: measured clearance minimum per run vs the 230/233mm thresholds ---
fig, ax = plt.subplots(figsize=(8, 5))
for i, cond in enumerate(conds):
    vals = [r["min_measured_clearance_mm"] for r in rows if r["condition"] == cond]
    xs = [i + np.random.uniform(-0.08, 0.08) for _ in vals]
    ax.scatter(xs, vals, color=colors[cond], s=60, zorder=3)
ax.axhline(230.0, color="red", ls="--", label="external measured-clearance abort (230mm)")
ax.axhline(233.0, color="darkred", ls=":", label="in-QP magnet-exclusion boundary (233mm, A/B/D/E only)")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(conds)
ax.set_ylabel("min measured clearance this run (mm)")
ax.set_title("Minimum measured magnet-to-beam-base clearance per run\nNo run in this campaign came within ~40mm of either threshold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot2_clearance_vs_threshold.png"), dpi=110)
plt.close(fig)

# --- Plot 3: forecast error by lead time, A vs B ---
fc = json.load(open(os.path.join(OUT_DIR, "forecast_scheduled_vs_frozen.json")))
fig, ax = plt.subplots(figsize=(8, 5))
a_leads = fc["A_scheduled"]["leads"]
a_means = [fc["A_scheduled"]["mean_err_mm"][str(j)] for j in a_leads]
b_leads = fc["B_frozen"]["leads"]
b_means = [fc["B_frozen"]["mean_err_mm"][str(j)] for j in b_leads]
ax.plot(a_leads, a_means, "o-", label="A (scheduled Jacobian), n=4 runs pooled")
ax.plot(b_leads, b_means, "s-", label="B (frozen J[0]), n=4 completed runs pooled")
ax.set_xlabel("lead stage j (target time = t_k + j*dt, dt~0.1s)")
ax.set_ylabel("mean |forecast - measured tip| (mm)")
ax.set_title("Full-horizon forecast error by lead time: scheduled vs frozen Jacobian\n(pooled ticks across runs -- NOT independent samples; run is the real unit, n=4 per condition)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot3_forecast_error_by_leadtime.png"), dpi=110)
plt.close(fig)

# --- Plot 4: E's adaptation gain trace for one representative run ---
e_run = "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T130244Z"
pred = [json.loads(l) for l in open(os.path.join(LOGS, e_run, "predicted_beam_positions.jsonl"))]
pf = [json.loads(l) for l in open(os.path.join(LOGS, e_run, "path_follow.jsonl"))]
t_by_step = {r["step"]: r["t_s"] for r in pf}
ts, g_hat, g_used, accepted = [], [], [], []
for r in pred:
    diag = r.get("jacobian_adaptation_diagnostics")
    if diag is None:
        continue
    t = t_by_step.get(r["step"])
    if t is None:
        continue
    ts.append(t)
    g_hat.append(diag.get("g_hat"))
    g_used.append(diag.get("g_used"))
    accepted.append(1 if diag.get("accepted") else 0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax1.plot(ts, g_hat, ".", ms=3, alpha=0.5, label="g_hat (raw estimate)")
ax1.plot(ts, g_used, "-", lw=1.2, label="g_used (after accept/reject gate)")
ax1.axhline(-0.3, color="r", ls="--", lw=0.8)
ax1.axhline(0.5, color="r", ls="--", lw=0.8)
ax1.set_ylabel("g")
ax1.legend(fontsize=8)
ax1.set_title(f"Condition E adaptation trace, one representative run ({e_run[-20:]})")
ax2.step(ts, accepted, where="mid")
ax2.set_ylabel("update accepted (0/1)")
ax2.set_xlabel("t (s)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot4_adaptation_trace_E.png"), dpi=110)
plt.close(fig)

print("Saved 4 plots to", OUT_DIR)
