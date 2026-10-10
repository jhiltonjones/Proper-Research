"""Full FJ-vs-SJ report for the raised (+30mm) rectangle: separates the
evidence into the three claims requested --
  (1) FJ is worse than SJ
  (2) the reason is Jacobian staleness / model-prediction failure
  (3) MPC is more fragile to that staleness than INV-7
-- via 4 figures + 2 tables, using per-rep (never averaged) trajectories and
a common-path RMS so aborted FJ runs are not compared unfairly against full
SJ runs. Self-contained: run from the repo root with
    python3 close_loop_logs/myrun/zshift_analysis/fj_full_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/jack/Proper-Research")
import zraise_patches
zraise_patches.apply()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = os.path.dirname(os.path.abspath(__file__))
FJ_SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_FJ.npy"
SJ_SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_accurate.npy"
DELAY_SAMPLES = 2
MAX_JOINT_STEP_RAD = 0.010
S_U_Q = 0.05
S_U_L = 0.005

MPC_SJ_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep4_20260924T113802Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep5_20260924T114043Z",
]
MPC_FJ_RUNS = [
    ("rep1", "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep1_20260924T134324Z", "tcp_out_of_workspace"),
    ("rep2", "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep2_20260924T140357Z", "tcp_out_of_workspace"),
    ("rep3", "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep3_20260924T140528Z", None),
    ("rep4", "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep4_20260924T140647Z", "insertion_offset_exceeded"),
]
INV_SJ_RUNS = [
    "close_loop_logs/myrun/inv_7dof_delay_aware_zraise30mm_20260924T103744Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep2_20260924T111308Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep3_20260924T111434Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep4_20260924T111548Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep5_20260924T111702Z",
]
INV_FJ_RUNS = [
    ("rep1", "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep1_20260924T140105Z", None),
    ("rep2", "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep2_20260924T140829Z", None),
    ("rep3", "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep3_20260924T140945Z", None),
    ("rep4", "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep4_20260924T141059Z", None),
]

J_FJ_schedule = np.load(FJ_SCHEDULE_PATH)
J_SJ_schedule = np.load(SJ_SCHEDULE_PATH)
J_FJ = J_FJ_schedule[0]
N_REF = J_SJ_schedule.shape[0]


def load_rows(rundir):
    rows = [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]
    rows.sort(key=lambda r: r["step"])
    return rows


def load_pb(rundir):
    path = f"{rundir}/predicted_beam_positions.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    return {r["step"]: r for r in rows}


# =====================================================================
# FIGURE 1: error vs path progress, every rep individually, aborts marked
# =====================================================================
fig1, axes1 = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
fig1.suptitle("Figure 1 -- tracking error vs path progress, every replicate individually\n"
              "(raised +30mm rectangle, MPC Q_N-on vs INV-7 kp=1.0, SJ vs FJ)", fontsize=12)

ax = axes1[0]
for rd in MPC_SJ_RUNS:
    rows = load_rows(rd)
    ax.plot([r["ref_index"] for r in rows], [r["error_norm_mm"] for r in rows],
            color="tab:blue", alpha=0.5, lw=1.2)
ax.plot([], [], color="tab:blue", alpha=0.5, lw=1.2, label=f"SJ (n={len(MPC_SJ_RUNS)} reps)")
colors_fj = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
for (label, rd, abort_reason), c in zip(MPC_FJ_RUNS, colors_fj):
    rows = load_rows(rd)
    xs = [r["ref_index"] for r in rows]
    ys = [r["error_norm_mm"] for r in rows]
    tag = f"FJ {label}" + (f" [ABORT: {abort_reason}]" if abort_reason else " [completed]")
    ax.plot(xs, ys, color=c, lw=1.6, label=tag)
    if abort_reason:
        ax.scatter([xs[-1]], [ys[-1]], color=c, marker="x", s=140, zorder=6, linewidths=3)
ax.set_yscale("log")
ax.set_title("Panel A: MPC (Q_N on)", fontsize=11)
ax.set_ylabel("||e_k|| (mm)")
ax.legend(fontsize=8, ncol=2)

ax = axes1[1]
for rd in INV_SJ_RUNS:
    rows = load_rows(rd)
    ax.plot([r["ref_index"] for r in rows], [r["error_norm_mm"] for r in rows],
            color="tab:blue", alpha=0.5, lw=1.2)
ax.plot([], [], color="tab:blue", alpha=0.5, lw=1.2, label=f"SJ (n={len(INV_SJ_RUNS)} reps)")
for (label, rd, abort_reason), c in zip(INV_FJ_RUNS, colors_fj):
    rows = load_rows(rd)
    xs = [r["ref_index"] for r in rows]
    ys = [r["error_norm_mm"] for r in rows]
    tag = f"FJ {label}" + (f" [ABORT: {abort_reason}]" if abort_reason else " [completed]")
    ax.plot(xs, ys, color=c, lw=1.6, label=tag)
    if abort_reason:
        ax.scatter([xs[-1]], [ys[-1]], color=c, marker="x", s=140, zorder=6, linewidths=3)
ax.set_yscale("log")
ax.set_title("Panel B: INV-7 (kp=1.0)", fontsize=11)
ax.set_xlabel("ref_index"); ax.set_ylabel("||e_k|| (mm)")
ax.legend(fontsize=8, ncol=2)

fig1.tight_layout(rect=[0, 0, 1, 0.93])
fig1.savefig(f"{OUTDIR}/fig1_error_vs_progress_per_rep.png", dpi=150)
print(f"saved -> {OUTDIR}/fig1_error_vs_progress_per_rep.png")

# =====================================================================
# FIGURE 2: Jacobian staleness r_J(k) + principal angle, aligned above
# tracking error; relinearized markers for MPC-FJ rep1 AND one INV-FJ rep
# =====================================================================
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    build_planning_context,
)

_, bundle, controller_pack, _ = build_planning_context()
jac_provider_accurate = from_model_bundle(
    bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate"
)


def principal_angles_deg(A, B):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


r_J_curve = np.array([
    np.linalg.norm(J_SJ_schedule[ri] - J_FJ, "fro") / np.linalg.norm(J_SJ_schedule[ri], "fro")
    for ri in range(N_REF)
])
angle_curve = np.array([
    principal_angles_deg(J_FJ, J_SJ_schedule[ri]).max() for ri in range(N_REF)
])

mpc_fj_rep1_rows = load_rows(MPC_FJ_RUNS[0][1])
mpc_fj_rep1_pb = load_pb(MPC_FJ_RUNS[0][1])
inv_fj_rep1_rows = load_rows(INV_FJ_RUNS[0][1])

relin_ticks_mpc = [120, 140, 150, 160, 165, 170]
relin_mpc = []
for t in relin_ticks_mpc:
    if t not in mpc_fj_rep1_pb:
        continue
    z_k = np.asarray(mpc_fj_rep1_pb[t]["z_meas"], dtype=float)
    J_live = np.asarray(jac_provider_accurate(z_k), dtype=float).reshape(3, 7)
    r = float(np.linalg.norm(J_live - J_FJ, "fro") / np.linalg.norm(J_live, "fro"))
    relin_mpc.append((mpc_fj_rep1_pb[t]["ref_index"], r))
    print(f"[fig2] MPC-FJ rep1 relinearized tick={t} ref_idx={mpc_fj_rep1_pb[t]['ref_index']} r_J={r:.4f}")

target_ref_idxs = [ri for ri, _ in relin_mpc]
relin_inv = []
inv_by_ri = {r["ref_index"]: r for r in inv_fj_rep1_rows}
for target_ri in target_ref_idxs:
    closest = min(inv_by_ri.keys(), key=lambda ri: abs(ri - target_ri))
    row = inv_by_ri[closest]
    z_k = np.array(row["q_meas_rad"] + [row["insertion_length_m"]], dtype=float)
    J_live = np.asarray(jac_provider_accurate(z_k), dtype=float).reshape(3, 7)
    r = float(np.linalg.norm(J_live - J_FJ, "fro") / np.linalg.norm(J_live, "fro"))
    relin_inv.append((closest, r))
    print(f"[fig2] INV-FJ rep1 relinearized ref_idx={closest} (target {target_ri}) r_J={r:.4f}")

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
fig2.suptitle("Figure 2 -- Jacobian staleness precedes tracking divergence\n"
              "(raised +30mm rectangle)", fontsize=12)

ax = axes2[0]
ax.plot(range(N_REF), r_J_curve, color="tab:purple", label="r_J(ref_idx) vs genuine SJ schedule")
ax.scatter(*zip(*relin_mpc), color="black", zorder=5, label="relinearized at MPC-FJ rep1's measured z_k")
ax.scatter(*zip(*relin_inv), color="darkgreen", marker="^", zorder=5, label="relinearized at INV-FJ rep1's measured z_k")
ax.set_ylabel("relative Frobenius error r_J")
ax.set_title("Panel A: Jacobian staleness r_J(k)", fontsize=10)
ax.legend(fontsize=8)

ax = axes2[1]
ax.plot(range(N_REF), angle_curve, color="tab:purple")
ax.set_ylabel("degrees")
ax.set_title("Panel B: weak-direction principal angle\nrow(J_FJ) vs row(J_sched(ref_idx))", fontsize=10)

ax = axes2[2]
ax.plot([r["ref_index"] for r in mpc_fj_rep1_rows], [r["error_norm_mm"] for r in mpc_fj_rep1_rows],
        color="tab:red", label="MPC-FJ rep1 (failing)")
for rd in MPC_SJ_RUNS[:1]:
    rows = load_rows(rd)
    ax.plot([r["ref_index"] for r in rows], [r["error_norm_mm"] for r in rows],
            color="tab:blue", alpha=0.7, label="MPC-SJ rep2 (clean, for reference)")
ax.set_yscale("log")
ax.set_xlabel("ref_index"); ax.set_ylabel("||e_k|| (mm)")
ax.set_title("Panel C: tracking error, same x-axis\n(staleness leads, divergence follows)", fontsize=10)
ax.legend(fontsize=8)

fig2.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig(f"{OUTDIR}/fig2_jacobian_staleness.png", dpi=150)
print(f"saved -> {OUTDIR}/fig2_jacobian_staleness.png")

# =====================================================================
# FIGURE 3: predicted (c_k) vs actual motion direction + wrong-descent %
# =====================================================================
def diag_c_dV(rows, pb, J_model_fn, label):
    """J_model_fn(ref_index) -> 3x7 Jacobian to use at that tick."""
    out = dict(ref_idx=[], c_k=[], dV_pred=[], dV_real=[])
    by_step = {r["step"]: r for r in rows}
    steps = sorted(by_step.keys())
    for i in range(len(steps) - 1):
        s, s1 = steps[i], steps[i + 1]
        if s1 != s + 1 or pb is None or s not in pb or s1 not in pb:
            continue
        z_k = np.asarray(pb[s]["z_meas"], dtype=float)
        z_k1 = np.asarray(pb[s1]["z_meas"], dtype=float)
        dz = z_k1 - z_k
        p_a = np.asarray(pb[s]["measured_beam_position_m"], dtype=float)
        p_b = np.asarray(pb[s1]["measured_beam_position_m"], dtype=float)
        dp_meas = p_b - p_a
        J = J_model_fn(by_step[s]["ref_index"])
        dp_model = J @ dz
        nfj, nmeas = np.linalg.norm(dp_model), np.linalg.norm(dp_meas)
        if nfj < 1e-9 or nmeas < 1e-9:
            continue
        c_k = float(np.dot(dp_model, dp_meas) / (nfj * nmeas))
        e_k = -np.asarray(by_step[s]["error_mm"], dtype=float) / 1000.0
        e_k1 = -np.asarray(by_step[s1]["error_mm"], dtype=float) / 1000.0
        dV_pred = float(2.0 * e_k @ dp_model)
        dV_real = float(e_k1 @ e_k1 - e_k @ e_k)
        out["ref_idx"].append(by_step[s]["ref_index"])
        out["c_k"].append(c_k)
        out["dV_pred"].append(dV_pred)
        out["dV_real"].append(dV_real)
    return {k: np.array(v) for k, v in out.items()}


mpc_sj_rep2_rows = load_rows(MPC_SJ_RUNS[0])
mpc_sj_rep2_pb = load_pb(MPC_SJ_RUNS[0])

d3_fj = diag_c_dV(mpc_fj_rep1_rows, mpc_fj_rep1_pb, lambda ri: J_FJ, "MPC-FJ rep1 (model=J_FJ)")
d3_sj = diag_c_dV(mpc_sj_rep2_rows, mpc_sj_rep2_pb, lambda ri: J_SJ_schedule[min(ri, N_REF - 1)], "MPC-SJ rep2 (model=J_SJ,k)")

wrong_descent_fj = (d3_fj["dV_pred"] < 0) & (d3_fj["dV_real"] > 0)
wrong_descent_sj = (d3_sj["dV_pred"] < 0) & (d3_sj["dV_real"] > 0)
tail_fj = d3_fj["ref_idx"] >= 130
tail_sj = d3_sj["ref_idx"] >= 130
pct_fj_tail = 100 * wrong_descent_fj[tail_fj].mean() if tail_fj.any() else float("nan")
pct_sj_tail = 100 * wrong_descent_sj[tail_sj].mean() if tail_sj.any() else float("nan")
print(f"[fig3] MPC-FJ rep1 tail c_k mean={d3_fj['c_k'][tail_fj].mean():.3f}, "
      f"wrong-descent%={pct_fj_tail:.1f}%")
print(f"[fig3] MPC-SJ rep2 tail c_k mean={d3_sj['c_k'][tail_sj].mean():.3f}, "
      f"wrong-descent%={pct_sj_tail:.1f}%")

fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
fig3.suptitle("Figure 3 -- predicted vs actual motion direction (MPC-FJ rep1 vs MPC-SJ rep2)", fontsize=12)

ax = axes3[0]
ax.plot(d3_fj["ref_idx"], d3_fj["c_k"], color="tab:red", label="FJ rep1 (model=J_FJ)")
ax.plot(d3_sj["ref_idx"], d3_sj["c_k"], color="tab:blue", label="SJ rep2 (model=J_SJ,k)")
ax.axhline(0, color="black", lw=0.7, linestyle=":")
ax.axvspan(130, max(d3_fj["ref_idx"].max(), d3_sj["ref_idx"].max()), color="gray", alpha=0.1, label="tail (ref_idx>=130)")
ax.set_ylabel("c_k = cos<J*dz, dp_meas>")
ax.set_title(f"Panel A: directional agreement\n"
             f"(FJ tail mean={d3_fj['c_k'][tail_fj].mean():+.2f}, SJ tail mean={d3_sj['c_k'][tail_sj].mean():+.2f})", fontsize=10)
ax.legend(fontsize=8)

ax = axes3[1]
ax.scatter(d3_fj["ref_idx"][wrong_descent_fj], np.ones(wrong_descent_fj.sum()), color="tab:red", marker="|", s=200,
           label=f"FJ: dV_pred<0 & dV_real>0 (tail {pct_fj_tail:.0f}%)")
ax.scatter(d3_sj["ref_idx"][wrong_descent_sj], np.zeros(wrong_descent_sj.sum()), color="tab:blue", marker="|", s=200,
           label=f"SJ: dV_pred<0 & dV_real>0 (tail {pct_sj_tail:.0f}%)")
ax.set_yticks([0, 1]); ax.set_yticklabels(["SJ", "FJ"])
ax.set_ylim(-0.5, 1.5)
ax.axvspan(130, max(d3_fj["ref_idx"].max(), d3_sj["ref_idx"].max()), color="gray", alpha=0.1)
ax.set_xlabel("ref_index")
ax.set_title('Panel B: "confidently wrong" ticks\n(controller predicted error reduction, error actually grew)', fontsize=10)
ax.legend(fontsize=8, loc="center left")

fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(f"{OUTDIR}/fig3_predicted_vs_actual_motion.png", dpi=150)
print(f"saved -> {OUTDIR}/fig3_predicted_vs_actual_motion.png")

# =====================================================================
# FIGURE 4: why MPC diverges while INV degrades -- effort, joint-rate
# activity, K_eff(k) = ||delta_u_k|| / (||e_k||+eps)
# =====================================================================
plan_dir = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
import glob
plan = np.load(glob.glob(plan_dir + "/*.npz")[0])
input_reference = plan["input_reference"]


def effort_series(rows):
    out = dict(ref_idx=[], du_norm=[], max_joint_ratio=[], k_eff=[])
    for r in rows:
        ri = min(r["ref_index"], N_REF - 1)
        u0 = np.asarray(r["u0"], dtype=float)
        u_ref = input_reference[ri]
        du = u0 - u_ref
        du_q = du[:6] / S_U_Q
        du_L = du[6] / S_U_L
        du_norm = float(np.sqrt(np.sum(du_q ** 2) + du_L ** 2))
        dq = np.asarray(r["q_target_delta_rad"], dtype=float)
        max_ratio = float(np.max(np.abs(dq)) / MAX_JOINT_STEP_RAD)
        e_norm_m = r["error_norm_mm"] / 1000.0
        k_eff = du_norm / (e_norm_m + 1e-6)
        out["ref_idx"].append(r["ref_index"])
        out["du_norm"].append(du_norm)
        out["max_joint_ratio"].append(max_ratio)
        out["k_eff"].append(k_eff)
    return {k: np.array(v) for k, v in out.items()}


eff_mpc_fj = effort_series(mpc_fj_rep1_rows)
eff_inv_fj = effort_series(inv_fj_rep1_rows)
eff_mpc_sj = effort_series(mpc_sj_rep2_rows)
inv_sj_rep_rows = load_rows(INV_SJ_RUNS[0])
eff_inv_sj = effort_series(inv_sj_rep_rows)

sat_idx = np.where(eff_mpc_fj["max_joint_ratio"] >= 0.99)[0]
first_sat_step = mpc_fj_rep1_rows[sat_idx[0]]["step"] if len(sat_idx) else None
print(f"[fig4] MPC-FJ rep1 first joint-step saturation at step={first_sat_step}")

fig4, axes4 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
fig4.suptitle("Figure 4 -- why MPC diverges while INV-7 merely degrades (both under the same stale J_FJ)", fontsize=12)

ax = axes4[0]
ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["du_norm"], color="tab:red", label="MPC-FJ rep1")
ax.plot(eff_mpc_sj["ref_idx"], eff_mpc_sj["du_norm"], color="tab:red", alpha=0.4, linestyle="--", label="MPC-SJ rep2 (clean)")
ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["du_norm"], color="tab:orange", label="INV-FJ rep1")
ax.plot(eff_inv_sj["ref_idx"], eff_inv_sj["du_norm"], color="tab:orange", alpha=0.4, linestyle="--", label="INV-SJ (clean)")
ax.set_ylabel(r"$\|\delta u_k\| = \|S_u^{-1}(u_0-u_{ref})\|$")
ax.set_title("Panel A: normalized commanded correction magnitude", fontsize=10)
ax.legend(fontsize=8)

ax = axes4[1]
ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["max_joint_ratio"], color="tab:red", label="MPC-FJ rep1")
ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["max_joint_ratio"], color="tab:orange", label="INV-FJ rep1")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.7, label="hard limit")
if first_sat_step is not None:
    first_sat_ri = mpc_fj_rep1_rows[sat_idx[0]]["ref_index"]
    ax.axvline(first_sat_ri, color="black", linestyle="-.", alpha=0.6,
               label=f"first MPC-FJ saturation (ref_idx={first_sat_ri})")
ax.set_ylabel("max|dq_i|/max_step")
ax.set_title("Panel B: joint-increment constraint activity\n(saturation is a consequence, not the trigger)", fontsize=10)
ax.legend(fontsize=8)

ax = axes4[2]
ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["k_eff"], color="tab:red", label="MPC-FJ rep1")
ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["k_eff"], color="tab:orange", label="INV-FJ rep1")
ax.set_yscale("log")
ax.set_ylabel(r"$K_{eff}(k)=\|\delta u_k\|/(\|e_k\|+\epsilon)$")
ax.set_xlabel("ref_index")
ax.set_title("Panel C: effective feedback gain K_eff(k)\n(does correction strength grow with error, or stay bounded?)", fontsize=10)
ax.legend(fontsize=8)

fig4.tight_layout(rect=[0, 0, 1, 0.93])
fig4.savefig(f"{OUTDIR}/fig4_why_mpc_diverges.png", dpi=150)
print(f"saved -> {OUTDIR}/fig4_why_mpc_diverges.png")

# =====================================================================
# COMMON-PATH METRIC + TABLES
# =====================================================================
mpc_fj_last_ref = {}
for label, rd, _ in MPC_FJ_RUNS:
    rows = load_rows(rd)
    mpc_fj_last_ref[label] = rows[-1]["ref_index"]
K_MIN = min(mpc_fj_last_ref.values())
print(f"\n[common-path] k_min = {K_MIN} (limited by {min(mpc_fj_last_ref, key=mpc_fj_last_ref.get)})")


def rms_over_window(rows, lo, hi):
    vals = [r["error_norm_mm"] for r in rows if lo <= r["ref_index"] <= hi]
    if not vals:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


def full_rms(rows):
    vals = [r["error_norm_mm"] for r in rows]
    return float(np.sqrt(np.mean(np.square(vals))))


def sat_p_active(rows, threshold=0.9):
    pairs = []
    for r in rows:
        dq = np.asarray(r["q_target_delta_rad"], dtype=float)
        pairs.extend((np.abs(dq) / MAX_JOINT_STEP_RAD).tolist())
    pairs = np.array(pairs)
    return float(np.mean(pairs >= threshold))


regimes = [(0, K_MIN * 0.33, "0-33%"), (K_MIN * 0.33, K_MIN * 0.66, "33-66%"), (K_MIN * 0.66, K_MIN, "66-100%")]

print("\n" + "=" * 100)
print("TABLE A: primary live-performance results")
print("=" * 100)
header = f"{'Controller':10s} {'Jacobian':8s} {'Completed':10s} {'Aborts':7s} {'RMS(completed)':15s} {'CommonPathRMS(0-%d)':22s} {'Max/final err':15s} {'P(sat>=0.9)':12s}" % K_MIN if False else None
print(f"{'Controller':10s} {'Jac':4s} {'Completed':10s} {'Aborts':7s} {'RMS(completed)':16s} {'CommonPathRMS(0-'+str(K_MIN)+')':24s} {'Max/final err (mm)':20s} {'P(joint sat)':12s}")

rows_table_a = []


def summarize_arm(name, jac, runs, is_fj, abort_info=None):
    completed = 0
    aborted = 0
    rms_completed_vals = []
    common_rms_vals = []
    max_err_vals = []
    p_sat_vals = []
    for item in runs:
        if is_fj:
            label, rd, abort_reason = item
        else:
            rd, abort_reason = item
        rows = load_rows(rd)
        did_complete = abort_reason is None
        if did_complete:
            completed += 1
            rms_completed_vals.append(full_rms(rows))
        else:
            aborted += 1
        common_rms_vals.append(rms_over_window(rows, 0, K_MIN))
        max_err_vals.append(max(r["error_norm_mm"] for r in rows))
        p_sat_vals.append(sat_p_active(rows))
    n = len(runs)
    rms_c = f"{np.mean(rms_completed_vals):.3f}" + ("*" if is_fj and aborted > 0 else "") if rms_completed_vals else "n/a"
    print(f"{name:10s} {jac:4s} {completed}/{n:<8d} {aborted}/{n:<5d} {rms_c:16s} "
          f"{np.mean(common_rms_vals):.3f} +/- {np.std(common_rms_vals):.3f}      "
          f"{max(max_err_vals):.2f}                {np.mean(p_sat_vals):.4f}")
    return dict(name=name, jac=jac, completed=completed, n=n, rms_completed=rms_completed_vals,
                common_rms=common_rms_vals, max_err=max_err_vals, p_sat=p_sat_vals)


res_mpc_sj = summarize_arm("MPC", "SJ", [(rd, None) for rd in MPC_SJ_RUNS], is_fj=False)
res_mpc_fj = summarize_arm("MPC", "FJ", MPC_FJ_RUNS, is_fj=True)
res_inv_sj = summarize_arm("INV-7", "SJ", [(rd, None) for rd in INV_SJ_RUNS], is_fj=False)
res_inv_fj = summarize_arm("INV-7", "FJ", INV_FJ_RUNS, is_fj=True)

print("\n* MPC-FJ RMS(completed) is from the SOLE completed replicate (rep3) and must not be read as overall FJ performance.")
print(f"  MPC-FJ's primary result is completion rate: {res_mpc_fj['completed']}/{res_mpc_fj['n']} = "
      f"{100*res_mpc_fj['completed']/res_mpc_fj['n']:.0f}%.")

print("\n" + "=" * 100)
print(f"TABLE A2: common-path RMS by progress regime (window = ref_index 0-{K_MIN}, the MPC-FJ common path)")
print("=" * 100)
print(f"{'Controller':10s} {'Jac':4s} " + " ".join(f"{name:>10s}" for _, _, name in regimes))
for name, jac, runs, is_fj in [
    ("MPC", "SJ", [(rd, None) for rd in MPC_SJ_RUNS], False),
    ("MPC", "FJ", MPC_FJ_RUNS, True),
    ("INV-7", "SJ", [(rd, None) for rd in INV_SJ_RUNS], False),
    ("INV-7", "FJ", INV_FJ_RUNS, True),
]:
    per_regime = []
    for lo, hi, _ in regimes:
        vals = []
        for item in runs:
            rd = item[1] if is_fj else item[0]
            rows = load_rows(rd)
            vals.append(rms_over_window(rows, lo, hi))
        per_regime.append(np.nanmean(vals))
    print(f"{name:10s} {jac:4s} " + " ".join(f"{v:10.3f}" for v in per_regime))

print("\n" + "=" * 100)
print("TABLE B: MPC-FJ per-replicate breakdown (n=4 is too small to hide behind aggregate stats)")
print("=" * 100)
print(f"{'rep':6s} {'outcome':11s} {'termination':20s} {'final err (mm)':16s} {'max err (mm)':14s} {'RMS (mm)':10s}")
for label, rd, abort_reason in MPC_FJ_RUNS:
    rows = load_rows(rd)
    outcome = "aborted" if abort_reason else "completed"
    print(f"{label:6s} {outcome:11s} {str(abort_reason):20s} {rows[-1]['error_norm_mm']:16.2f} "
          f"{max(r['error_norm_mm'] for r in rows):14.2f} {full_rms(rows):10.3f}")

print("\n" + "=" * 100)
print("TABLE C: mechanism table -- MPC-SJ vs MPC-FJ (rules out alternative explanations)")
print("=" * 100)


def solver_health(rows, pb):
    n = len(rows)
    ok = sum(1 for r in rows if r["solver_success"])
    dm = sum(1 for r in pb.values() if r.get("deadline_miss")) if pb else None
    delay_errs = []
    if pb:
        for r in rows:
            s_prev = r["step"] - DELAY_SAMPLES
            if s_prev in pb and r["step"] in pb:
                q_now = np.asarray(pb[r["step"]]["z_meas"][:6])
                q_then = np.asarray(pb[s_prev]["q_cmd_k"][:6])
                delay_errs.append(np.max(np.abs(q_now - q_then)))
    tail_delay_mean = np.mean(delay_errs) if delay_errs else float("nan")
    return ok, n, dm, tail_delay_mean


ok_fj, n_fj, dm_fj, delay_fj = solver_health(mpc_fj_rep1_rows, mpc_fj_rep1_pb)
ok_sj, n_sj, dm_sj, delay_sj = solver_health(mpc_sj_rep2_rows, mpc_sj_rep2_pb)
r_J_late = np.mean([r for _, r in relin_mpc[-2:]])
angle_late = angle_curve[[ri for ri, _ in relin_mpc][-1]]

print(f"{'metric':30s} {'MPC-SJ (rep2)':18s} {'MPC-FJ (rep1)':18s}")
print(f"{'solver success':30s} {f'{ok_sj}/{n_sj}':18s} {f'{ok_fj}/{n_fj}':18s}")
print(f"{'deadline misses':30s} {str(dm_sj):18s} {str(dm_fj):18s}")
print(f"{'delay-consistency mean (rad)':30s} {delay_sj:<18.5f} {delay_fj:<18.5f}")
print(f"{'r_J late (relinearized)':30s} {'--':18s} {r_J_late:<18.3f}")
print(f"{'weak-axis principal angle (deg)':30s} {'--':18s} {angle_late:<18.1f}")
print(f"{'tail c_k (ref_idx>=130)':30s} {d3_sj['c_k'][tail_sj].mean():<18.3f} {d3_fj['c_k'][tail_fj].mean():<18.3f}")
print(f"{'wrong-descent ticks, tail %':30s} {pct_sj_tail:<18.1f} {pct_fj_tail:<18.1f}")
print(f"{'first joint-step saturation':30s} {'--':18s} {str(first_sat_step) + ' (step)':18s}")

print("\n" + "=" * 100)
print("INV-7 lightweight staleness check (does INV also see a stale J but stay bounded?)")
print("=" * 100)
d3_inv_fj = diag_c_dV(inv_fj_rep1_rows, {r["step"]: dict(z_meas=r["q_meas_rad"] + [r["insertion_length_m"]],
                                                          measured_beam_position_m=[m / 1000.0 for m in r["tip_mm"]])
                                          for r in inv_fj_rep1_rows},
                       lambda ri: J_FJ, "INV-FJ rep1")
tail_inv = d3_inv_fj["ref_idx"] >= 130
print(f"INV-FJ rep1: tail c_k mean={d3_inv_fj['c_k'][tail_inv].mean():.3f} "
      f"(vs MPC-FJ rep1 tail c_k mean={d3_fj['c_k'][tail_fj].mean():.3f})")
print(f"INV-FJ rep1 relinearized r_J at matched ref_idx: {[f'{r:.3f}' for _, r in relin_inv]} "
      f"(vs MPC-FJ rep1: {[f'{r:.3f}' for _, r in relin_mpc]})")
print(f"K_eff(k) late-path mean: MPC-FJ={eff_mpc_fj['k_eff'][-20:].mean():.3f}  "
      f"INV-FJ={eff_inv_fj['k_eff'][-20:].mean():.3f}  "
      f"(MPC's effective gain grows as error grows; INV's stays bounded if architecture story holds)")

print("\nDONE.")
