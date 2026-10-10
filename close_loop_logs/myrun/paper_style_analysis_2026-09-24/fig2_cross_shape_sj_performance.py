"""Cross-shape SJ performance figure, in the style of
paper_package_2026-09-21/00_main/Figure2_cross_shape_SJ_performance.png:
3 panels (RMS tracking error, normalized feedback effort E_u, joint
command-increment constraint activity %), bars = INV-7 (green, hatched)
vs MPC (blue), one bar-pair per shape, individual rep dots overlaid.
Rectangle + triangle only (today's data; U-shape excluded, see
ANALYSIS.md). Uses the SJ (genuine schedule) runs -- this is a clean-
model performance comparison, not the FJ/SJ robustness question (that's
Figures 2/3 in this folder).

Metric definitions (reconstructed, see ANALYSIS.md's caveats):
  E_u: per-rep MEAN of the normalized combined feedback effort
       ||S_u^-1(u0-u_ref)|| across all ticks (same normalization as used
       throughout this project: S_u,q=0.05 rad/s, S_u,L=0.005 m/s).
  activity (%): per-rep MEAN over all (tick, joint) pairs of
       |dq_i|/max_joint_step_rad * 100 -- an average utilization level,
       not a saturation-threshold probability (which was already reported
       as P(active>=0.9) in the earlier effort_and_constraint_activity
       figures and is far smaller, <1%, for both controllers on every
       shape tested).

Run from the repo root:
    python3 close_loop_logs/myrun/paper_style_analysis_2026-09-24/fig2_cross_shape_sj_performance.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.path.dirname(os.path.abspath(__file__))
MAX_JOINT_STEP_RAD = 0.010
S_U_Q = 0.05
S_U_L = 0.005

RECT_DIR = "close_loop_logs/myrun"
TRI_DIR = "close_loop_logs/myrun"

SHAPES = {
    "Rectangle": dict(
        plan_dir="plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path",
        mpc_sj=[
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep4_20260924T113802Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep5_20260924T114043Z",
        ],
        inv_sj=[
            f"{RECT_DIR}/inv_7dof_delay_aware_zraise30mm_20260924T103744Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep2_20260924T111308Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep3_20260924T111434Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep4_20260924T111548Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep5_20260924T111702Z",
        ],
    ),
    "Triangle (25 mm)": dict(
        plan_dir="plans/triangle_base25mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path",
        mpc_sj=[
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep1_20260924T162512Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep2_20260924T162637Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep3_20260924T162805Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep4_20260924T162939Z",
        ],
        inv_sj=[
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep1_20260924T164854Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep2_20260924T163610Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep3_20260924T163730Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep4_20260924T163855Z",
        ],
    ),
}


def load_rows(rundir):
    rows = [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]
    rows.sort(key=lambda r: r["step"])
    return rows


def per_rep_metrics(rundir, plan_dir):
    import glob
    plan = np.load(glob.glob(plan_dir + "/*.npz")[0])
    input_reference = plan["input_reference"]
    N_REF = plan["state_reference"].shape[0]
    rows = load_rows(rundir)
    err = np.array([r["error_norm_mm"] for r in rows])
    rms = float(np.sqrt(np.mean(err ** 2)))

    effort_vals, activity_vals = [], []
    for r in rows:
        ri = min(r["ref_index"], N_REF - 1)
        u0 = np.asarray(r["u0"], dtype=float)
        u_ref = input_reference[ri]
        du_q = (u0[:6] - u_ref[:6]) / S_U_Q
        du_L = (u0[6] - u_ref[6]) / S_U_L
        effort_vals.append(float(np.sqrt(np.sum(du_q ** 2) + du_L ** 2)))
        dq = np.asarray(r["q_target_delta_rad"], dtype=float)
        activity_vals.extend((np.abs(dq) / MAX_JOINT_STEP_RAD * 100.0).tolist())
    return rms, float(np.mean(effort_vals)), float(np.mean(activity_vals))


results = {}
for shape_name, cfg in SHAPES.items():
    results[shape_name] = {}
    for label, runs in [("INV-7", cfg["inv_sj"]), ("MPC", cfg["mpc_sj"])]:
        rms_list, eu_list, act_list = [], [], []
        for rd in runs:
            rms, eu, act = per_rep_metrics(rd, cfg["plan_dir"])
            rms_list.append(rms); eu_list.append(eu); act_list.append(act)
        results[shape_name][label] = dict(rms=rms_list, eu=eu_list, act=act_list)
        print(f"{shape_name} {label}: RMS={np.mean(rms_list):.3f}+/-{np.std(rms_list):.3f}mm  "
              f"E_u={np.mean(eu_list):.3f}+/-{np.std(eu_list):.3f}  "
              f"activity={np.mean(act_list):.2f}+/-{np.std(act_list):.2f}%")

shape_names = list(SHAPES.keys())
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
bar_w = 0.35
colors = {"INV-7": "tab:green", "MPC": "tab:blue"}
hatches = {"INV-7": "//", "MPC": None}

for ax, metric, title, ylabel in [
    (axes[0], "rms", "RMS tracking error", "RMS error (mm)"),
    (axes[1], "eu", "Normalized feedback effort", r"$E_u$ (dimensionless)"),
    (axes[2], "act", "Joint command-increment constraint activity", "activity (%)"),
]:
    x = np.arange(len(shape_names))
    for i, label in enumerate(["INV-7", "MPC"]):
        means = [np.mean(results[s][label][metric]) for s in shape_names]
        sds = [np.std(results[s][label][metric]) for s in shape_names]
        offset = (i - 0.5) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=sds, capsize=4, color=colors[label],
               hatch=hatches[label], edgecolor="black", alpha=0.85, label=label)
        for j, s in enumerate(shape_names):
            vals = results[s][label][metric]
            ax.scatter(np.full(len(vals), x[j] + offset), vals, color="black", s=18, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(shape_names)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Cross-shape SJ performance, raised +30mm workspace, today's data", fontsize=13)
fig.tight_layout(rect=[0, 0.08, 1, 0.92])
OUT = f"{OUTDIR}/fig2_cross_shape_sj_performance.png"
fig.savefig(OUT, dpi=150)
print(f"\nsaved -> {OUT}")
