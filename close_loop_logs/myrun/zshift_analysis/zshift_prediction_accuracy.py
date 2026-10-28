"""MPC prediction-accuracy analysis for the z_shift_workspace live campaign
(2026-09-24) -- CORRECTED VERSION: the original single Q_N-on raised run
(rep1) was later found to have been run while a separate CPU-heavy
background job (789% CPU) was silently still active, degrading its
tracking (RMS 2.00mm) independent of the Q_N formulation -- confirmed by
4 clean reps (2-5, CPU contention removed) achieving RMS 0.393mm, nearly
identical to Q_N=0's 0.429mm. This version uses ONLY the 4 clean Q_N-on
reps, not the contaminated rep1 (whose raw log was separately lost --
see conversation history for its exact recorded numbers).

Question: how well does the MPC's own within-horizon prediction
(E_chosen -- the QP's belief about where the beam tip will be H ticks from
now) match what actually happens (E_actual -- the real measured error H
ticks later)? Per-run correlation (r), regression (E_actual = a + b*E_chosen),
RMSE, signed bias, and wrong-sign fraction, at horizon depths +3/+5/+10 --
run-level summaries (mean +/- sd across clean reps), not pooled ticks
treated as independent replicates.

INV-7 has NO internal multi-step prediction (predicted_beam_positions.jsonl
does not exist for it, horizon_len=0 in its own logs) -- included only as
an actual-tracking-performance comparison baseline, not prediction accuracy.

error convention throughout (matches the harness's own path_follow.jsonl
`error_mm` field): error = desired - achieved (target minus actual/predicted
position), consistent sign between E_chosen and E_actual.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HORIZONS = (3, 5, 10)

MPC_QN0_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_qn_zero_zraise30mm_20260924T105435Z",
    "close_loop_logs/myrun/mpc_delay_aware_qn_zero_zraise30mm_rep2_20260924T110947Z",
    "close_loop_logs/myrun/mpc_delay_aware_qn_zero_zraise30mm_rep3_20260924T111126Z",
    "close_loop_logs/myrun/mpc_delay_aware_qn_zero_zraise30mm_rep4_20260924T110621Z",
    "close_loop_logs/myrun/mpc_delay_aware_qn_zero_zraise30mm_rep5_20260924T110756Z",
]
MPC_QNON_RUNS = [  # clean reps only -- rep1 was contended, excluded (see module docstring)
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep4_20260924T113802Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep5_20260924T114043Z",
]
INV7_KP1_RUNS = [
    "close_loop_logs/myrun/inv_7dof_delay_aware_zraise30mm_20260924T103744Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep2_20260924T111308Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep3_20260924T111434Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep4_20260924T111548Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep5_20260924T111702Z",
]
INV7_KP06_RUN = "close_loop_logs/myrun/inv_7dof_delay_aware_kp06_zraise30mm_20260924T105832Z"
ORIGINAL_MPC_RUNS = [  # un-raised, frozen w_L=0 baseline, for context
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112405Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112843Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114154Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114458Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114842Z",
]


def load_run(rundir):
    rows = [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]
    pred_path = f"{rundir}/predicted_beam_positions.jsonl"
    pred_rows = None
    if os.path.exists(pred_path):
        pred_rows = [json.loads(l) for l in open(pred_path)]
    return rows, pred_rows


def run_rms(rundir):
    rows, _ = load_run(rundir)
    err = np.array([r["error_norm_mm"] for r in rows])
    return float(np.sqrt(np.mean(err ** 2))), len(rows)


def prediction_pairs(rundir, horizon):
    rows, pred_rows = load_run(rundir)
    if pred_rows is None:
        return None
    actual_by_step = {r["step"]: r for r in rows}
    chosen_vecs, actual_vecs = [], []
    for pr in pred_rows:
        step = pr["step"]
        preds = pr.get("predicted_beam_positions_m")
        if preds is None:
            continue
        preds = np.asarray(preds, dtype=float)
        stage = horizon - 1
        if stage >= preds.shape[0]:
            continue
        future_step = step + horizon
        future_row = actual_by_step.get(future_step)
        if future_row is None:
            continue
        desired_future_mm = np.asarray(future_row["desired_mm"], dtype=float)
        p_pred_mm = preds[stage] * 1000.0
        e_chosen_vec = desired_future_mm - p_pred_mm
        e_actual_vec = np.asarray(future_row["error_mm"], dtype=float)
        chosen_vecs.append(e_chosen_vec)
        actual_vecs.append(e_actual_vec)
    if not chosen_vecs:
        return None
    chosen_vecs = np.asarray(chosen_vecs)
    actual_vecs = np.asarray(actual_vecs)
    return dict(
        chosen_vec=chosen_vecs, actual_vec=actual_vecs,
        chosen_norm=np.linalg.norm(chosen_vecs, axis=1),
        actual_norm=np.linalg.norm(actual_vecs, axis=1),
    )


def run_level_stats(chosen, actual):
    n = len(chosen)
    if n < 3:
        return None
    r = float(np.corrcoef(chosen, actual)[0, 1]) if np.std(chosen) > 1e-9 and np.std(actual) > 1e-9 else np.nan
    b, a = np.polyfit(chosen, actual, 1)
    pred = a + b * chosen
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    bias = float(np.mean(actual - chosen))
    wrong_sign = float(np.mean(np.sign(chosen) != np.sign(actual)))
    return dict(r=r, slope=float(b), intercept=float(a), rmse=rmse, bias=bias,
                wrong_sign_frac=wrong_sign, n=n)


def aggregate_run_level(runs, horizon, component):
    out = []
    for rd in runs:
        pp = prediction_pairs(rd, horizon)
        if pp is None:
            continue
        if component == "norm":
            chosen, actual = pp["chosen_norm"], pp["actual_norm"]
        elif component == "x":
            chosen, actual = pp["chosen_vec"][:, 0], pp["actual_vec"][:, 0]
        elif component == "y":
            chosen, actual = pp["chosen_vec"][:, 1], pp["actual_vec"][:, 1]
        stats = run_level_stats(chosen, actual)
        if stats is not None:
            stats["run"] = rd.split("/")[-1]
            out.append(stats)
    return out


def mean_sd(vals):
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def print_summary(label, runs):
    print("=" * 100)
    print(f"{label} -- run-level prediction-accuracy summary (n={len(runs)} clean reps)")
    print("=" * 100)
    out = {}
    for h in HORIZONS:
        for comp in ("norm", "x", "y"):
            stats_list = aggregate_run_level(runs, h, comp)
            out[(h, comp)] = stats_list
            if not stats_list:
                continue
            r_m, r_s = mean_sd([s["r"] for s in stats_list])
            b_m, b_s = mean_sd([s["slope"] for s in stats_list])
            a_m, a_s = mean_sd([s["intercept"] for s in stats_list])
            rmse_m, rmse_s = mean_sd([s["rmse"] for s in stats_list])
            bias_m, bias_s = mean_sd([s["bias"] for s in stats_list])
            ws_m, ws_s = mean_sd([s["wrong_sign_frac"] for s in stats_list])
            print(f"  +{h:>2} [{comp:>4}] (n_runs={len(stats_list)}): "
                  f"r={r_m:+.3f}+/-{r_s:.3f}  slope={b_m:+.3f}+/-{b_s:.3f}  "
                  f"intercept={a_m:+.3f}+/-{a_s:.3f}mm  rmse={rmse_m:.3f}+/-{rmse_s:.3f}mm  "
                  f"bias={bias_m:+.3f}+/-{bias_s:.3f}mm  wrong_sign={ws_m:.3f}+/-{ws_s:.3f}")
    return out


qn0_stats = print_summary("MPC Q_N=0 (raised)", MPC_QN0_RUNS)
print()
qnon_stats = print_summary("MPC Q_N ON (raised, clean reps only)", MPC_QNON_RUNS)

print("\n" + "=" * 100)
print("INV-7 -- NO internal multi-step prediction exists. Actual-tracking-only comparison:")
print("=" * 100)
for label, runs in [("INV-7 kp=1.0", INV7_KP1_RUNS), ("INV-7 kp=0.6", [INV7_KP06_RUN]),
                     ("MPC Q_N=0 (raised)", MPC_QN0_RUNS), ("MPC Q_N ON (raised, clean)", MPC_QNON_RUNS),
                     ("MPC Q_N ON (original, historical)", ORIGINAL_MPC_RUNS)]:
    rmses = [run_rms(rd)[0] for rd in runs]
    m, s = mean_sd(rmses)
    print(f"  {label:35s}: RMS mean={m:.3f}mm sd={s:.3f}mm (n={len(rmses)})")

# ---------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/zshift_prediction_plots_v2"
os.makedirs(OUT, exist_ok=True)


def scatter_fig(runs, label, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    for ax, h in zip(axes, HORIZONS):
        all_chosen, all_actual = [], []
        for rd, c in zip(runs, colors):
            pp = prediction_pairs(rd, h)
            if pp is None:
                continue
            ax.scatter(pp["chosen_norm"], pp["actual_norm"], s=10, color=c, alpha=0.5,
                       label=rd.split("/")[-1][-15:])
            all_chosen.append(pp["chosen_norm"])
            all_actual.append(pp["actual_norm"])
        if all_chosen:
            all_chosen = np.concatenate(all_chosen)
            all_actual = np.concatenate(all_actual)
            lims = [0, max(all_chosen.max(), all_actual.max()) * 1.05]
            ax.plot(lims, lims, "k--", alpha=0.4, label="y=x (perfect prediction)")
            b, a = np.polyfit(all_chosen, all_actual, 1)
            xx = np.linspace(*lims, 50)
            ax.plot(xx, a + b * xx, "r-", alpha=0.7, label=f"fit: y={a:.2f}+{b:.2f}x")
        ax.set_xlabel("E_chosen (predicted error, mm)")
        ax.set_ylabel("E_actual (real error, mm)")
        ax.set_title(f"+{h} ticks ahead")
        ax.legend(fontsize=6)
    fig.suptitle(f"{label}: predicted vs actual beam-tip error")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=150)
    print(f"saved -> {OUT}/{fname}")


scatter_fig(MPC_QN0_RUNS, "MPC Q_N=0 (raised)", "fig1a_qn0_scatter.png")
scatter_fig(MPC_QNON_RUNS, "MPC Q_N ON (raised, clean)", "fig1b_qnon_scatter.png")

# Figure 2: run-level metric comparison, Q_N=0 vs Q_N-on, both clean n>=4
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
metric_names = ["r", "slope", "intercept", "rmse", "bias", "wrong_sign_frac"]
metric_labels = ["Pearson r", "regression slope", "regression intercept (mm)",
                  "RMSE (mm)", "signed bias (mm)", "wrong-sign fraction"]
for ax, mname, mlabel in zip(axes.flat, metric_names, metric_labels):
    for stats_dict, label, color, marker in [
        (qn0_stats, "MPC Q_N=0 (n=5)", "tab:blue", "o"),
        (qnon_stats, "MPC Q_N ON (n=4, clean)", "tab:red", "s"),
    ]:
        means, sds = [], []
        for h in HORIZONS:
            vals = [s[mname] for s in stats_dict[(h, "norm")]]
            m, s = mean_sd(vals)
            means.append(m)
            sds.append(s)
        ax.errorbar(HORIZONS, means, yerr=sds, marker=marker, capsize=4, label=label, color=color)
    ax.set_xlabel("horizon depth (ticks ahead)")
    ax.set_ylabel(mlabel)
    ax.set_title(mlabel)
    ax.legend(fontsize=7)
fig.suptitle("MPC prediction-accuracy metrics vs horizon depth (both CLEAN, run-level mean +/- sd)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_metrics_vs_horizon.png", dpi=150)
print(f"saved -> {OUT}/fig2_metrics_vs_horizon.png")

# Figure 3: overall RMS comparison, corrected
fig, ax = plt.subplots(figsize=(8, 5))
labels_runs = [
    ("Original MPC\n(un-raised)", ORIGINAL_MPC_RUNS),
    ("MPC Q_N ON\n(raised, clean)", MPC_QNON_RUNS),
    ("MPC Q_N=0\n(raised)", MPC_QN0_RUNS),
    ("INV-7 kp=1.0\n(raised)", INV7_KP1_RUNS),
    ("INV-7 kp=0.6\n(raised)", [INV7_KP06_RUN]),
]
means, sds = [], []
for _, runs in labels_runs:
    rmses = [run_rms(rd)[0] for rd in runs]
    m, s = mean_sd(rmses)
    means.append(m)
    sds.append(s)
ax.bar(range(len(labels_runs)), means, yerr=sds, capsize=5,
       color=["gray", "tab:red", "tab:blue", "tab:green", "tab:orange"])
ax.set_xticks(range(len(labels_runs)))
ax.set_xticklabels([l for l, _ in labels_runs])
ax.set_ylabel("tracking RMS (mm)")
ax.set_title("Actual tracking performance across all conditions (corrected, CPU-contention-free)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_rms_comparison.png", dpi=150)
print(f"saved -> {OUT}/fig3_rms_comparison.png")

print(f"\nAll plots saved under {OUT}/")
