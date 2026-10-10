"""FJ rep1 divergence diagnostics -- per the user's 8-part specification.

Diagnostics computed purely from already-logged data (no live hardware,
no re-solving the QP):
  1. Frozen-model prediction vs actual beam response: c_k (cosine sim),
     g_k (gain ratio), using Delta_z_k from realized measured configuration
     increments (predicted_beam_positions.jsonl's own z_meas sequence).
  2. Predicted vs realized task descent: dV_pred vs dV_real (Lyapunov-style,
     squared tracking-error convention, error = achieved - desired).
  3. Jacobian staleness at selected ticks: relinearize J(z_k) online via the
     SAME from_model_bundle/build_diagnostic_adapter machinery (jacobian_mode
     ="accurate"), compare to J_FJ via relative Frobenius error and
     row-space principal angles.
  6. Solver/execution health: solver_status/success, solve_time_s,
     deadline_miss, and the delay-consistency check q_meas_k ~= q_cmd_{k-2}.
  7. Disturbance-smoothness proxy: immediate one-step prediction residual
     measured_beam_position - predicted_beam_positions_m[0], as a proxy for
     the controller's internal filtered disturbance (not the literal
     internal EMA state, which isn't recoverable from these logs alone).
  8. Reference/invariant continuity: ref_index monotonicity, and a direct
     check that the cached FJ schedule really is bitwise-constant.

All of this is run for BOTH the failing FJ rep1 run and a clean, successful
SJ run (MPC Q_N-on, same plan/controller), for direct overlay comparison.
"""
import glob
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

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
FJ_DIR = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep1_20260924T134324Z"
SJ_DIR = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z"
FJ_SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_FJ.npy"
SJ_SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_accurate.npy"
DELAY_SAMPLES = 2

# ---------------------------------------------------------------------
# 8. reference/invariant continuity: confirm the FJ schedule really is
#    constant (every row bitwise-equal to row 0).
# ---------------------------------------------------------------------
J_FJ_schedule = np.load(FJ_SCHEDULE_PATH)
J_SJ_schedule = np.load(SJ_SCHEDULE_PATH)
n_distinct = len(np.unique(J_FJ_schedule.reshape(J_FJ_schedule.shape[0], -1), axis=0))
print(f"[8] FJ schedule: {J_FJ_schedule.shape[0]} rows, {n_distinct} distinct rows "
      f"(should be 1 if truly frozen)")
J_FJ = J_FJ_schedule[0]  # (3,7), constant Jacobian used throughout the FJ run
print(f"[8] J_FJ (row0) vs J_SJ[0] identical: "
      f"{np.allclose(J_FJ, J_SJ_schedule[0])}  (FJ was built as repeat(SJ[0]))")


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def load_run(rundir):
    pf = {r["step"]: r for r in load_jsonl(f"{rundir}/path_follow.jsonl")}
    pb = {r["step"]: r for r in load_jsonl(f"{rundir}/predicted_beam_positions.jsonl")}
    steps = sorted(set(pf) & set(pb))
    return pf, pb, steps


pf_fj, pb_fj, steps_fj = load_run(FJ_DIR)
pf_sj, pb_sj, steps_sj = load_run(SJ_DIR)
print(f"loaded FJ rep1: {len(steps_fj)} aligned steps; SJ rep2: {len(steps_sj)} aligned steps")

# ---------------------------------------------------------------------
# [8] ref_index monotonicity check
# ---------------------------------------------------------------------
for label, pf, steps in [("FJ rep1", pf_fj, steps_fj), ("SJ rep2", pf_sj, steps_sj)]:
    ref_idx = np.array([pf[s]["ref_index"] for s in steps])
    backwards = np.sum(np.diff(ref_idx) < 0)
    big_jump = np.max(np.diff(ref_idx)) if len(ref_idx) > 1 else 0
    print(f"[8] {label}: ref_index range [{ref_idx.min()},{ref_idx.max()}], "
          f"backward steps={backwards}, max forward jump={big_jump}")

# ---------------------------------------------------------------------
# [6] solver/execution health + delay consistency
# ---------------------------------------------------------------------
def solver_health(pf, pb, steps, label):
    n = len(steps)
    ok = sum(1 for s in steps if pf[s]["solver_success"])
    deadline_miss = sum(1 for s in steps if pb[s]["deadline_miss"])
    solve_t = np.array([pf[s]["solver_time_s"] for s in steps]) * 1000.0
    # delay check: q_meas at step s should match q_cmd applied DELAY_SAMPLES steps earlier
    delay_errs = []
    for s in steps:
        s_prev = s - DELAY_SAMPLES
        if s_prev in pb:
            q_meas_now = np.asarray(pb[s]["z_meas"][:6])
            q_cmd_then = np.asarray(pb[s_prev]["q_cmd_k"][:6])
            delay_errs.append(np.max(np.abs(q_meas_now - q_cmd_then)))
    delay_errs = np.array(delay_errs)
    print(f"[6] {label}: solver_success={ok}/{n}, deadline_miss={deadline_miss}/{n}, "
          f"solve_time_ms mean={solve_t.mean():.3f} max={solve_t.max():.3f}, "
          f"delay-consistency |q_meas_k - q_cmd_(k-{DELAY_SAMPLES})| max_rad="
          f"{delay_errs.max():.5f} mean_rad={delay_errs.mean():.5f}")
    return dict(solve_t=solve_t, delay_errs=delay_errs)


health_fj = solver_health(pf_fj, pb_fj, steps_fj, "FJ rep1")
health_sj = solver_health(pf_sj, pb_sj, steps_sj, "SJ rep2")

# also specifically check the last 40 ticks of FJ (the failure window)
tail = [s for s in steps_fj if s >= 130]
health_tail = solver_health(pf_fj, pb_fj, tail, "FJ rep1 [tail, step>=130]")

# ---------------------------------------------------------------------
# [1] directional agreement c_k, gain ratio g_k  (+ [2] dV_pred vs dV_real)
# ---------------------------------------------------------------------
def diag_1_2(pf, pb, steps, J_const, label):
    out = dict(step=[], ref_idx=[], c_k=[], g_k=[], dV_pred=[], dV_real=[],
               e_norm=[], dp_meas_norm=[], dp_fj_norm=[])
    steps_sorted = steps
    for i in range(len(steps_sorted) - 1):
        s, s1 = steps_sorted[i], steps_sorted[i + 1]
        if s1 != s + 1:
            continue  # only use genuinely consecutive ticks
        z_k = np.asarray(pb[s]["z_meas"], dtype=float)
        z_k1 = np.asarray(pb[s1]["z_meas"], dtype=float)
        dz = z_k1 - z_k
        p_meas_k = np.asarray(pb[s]["measured_beam_position_m"], dtype=float)
        p_meas_k1 = np.asarray(pb[s1]["measured_beam_position_m"], dtype=float)
        dp_meas = p_meas_k1 - p_meas_k
        dp_fj = J_const @ dz

        nfj, nmeas = np.linalg.norm(dp_fj), np.linalg.norm(dp_meas)
        if nfj > 1e-9 and nmeas > 1e-9:
            c_k = float(np.dot(dp_fj, dp_meas) / (nfj * nmeas))
            g_k = float(nmeas / nfj)
        else:
            c_k, g_k = np.nan, np.nan

        # error convention: achieved - desired = -(error_mm) since logged
        # error_mm = desired - achieved
        e_k = -np.asarray(pf[s]["error_mm"], dtype=float) / 1000.0
        e_k1 = -np.asarray(pf[s1]["error_mm"], dtype=float) / 1000.0
        dV_pred = float(2.0 * e_k @ dp_fj)
        dV_real = float(e_k1 @ e_k1 - e_k @ e_k)

        out["step"].append(s)
        out["ref_idx"].append(pf[s]["ref_index"])
        out["c_k"].append(c_k)
        out["g_k"].append(g_k)
        out["dV_pred"].append(dV_pred)
        out["dV_real"].append(dV_real)
        out["e_norm"].append(float(np.linalg.norm(e_k)))
        out["dp_meas_norm"].append(nmeas)
        out["dp_fj_norm"].append(nfj)
    return {k: np.array(v) for k, v in out.items()}


d12_fj = diag_1_2(pf_fj, pb_fj, steps_fj, J_FJ, "FJ rep1 (vs J_FJ)")
d12_sj_vs_fj = diag_1_2(pf_sj, pb_sj, steps_sj, J_FJ, "SJ rep2 (vs J_FJ, counterfactual)")


def diag_1_2_sj_own(pf, pb, steps, schedule):
    """SJ run evaluated against ITS OWN genuine per-tick schedule value
    (the honest apples-to-apples comparison: SJ's own model vs SJ's own
    realized response)."""
    out = dict(step=[], ref_idx=[], c_k=[], g_k=[], dV_pred=[], dV_real=[], e_norm=[])
    for i in range(len(steps) - 1):
        s, s1 = steps[i], steps[i + 1]
        if s1 != s + 1:
            continue
        ref_idx = min(pf[s]["ref_index"], schedule.shape[0] - 1)
        J_k = schedule[ref_idx]
        z_k = np.asarray(pb[s]["z_meas"], dtype=float)
        z_k1 = np.asarray(pb[s1]["z_meas"], dtype=float)
        dz = z_k1 - z_k
        p_meas_k = np.asarray(pb[s]["measured_beam_position_m"], dtype=float)
        p_meas_k1 = np.asarray(pb[s1]["measured_beam_position_m"], dtype=float)
        dp_meas = p_meas_k1 - p_meas_k
        dp_sj = J_k @ dz
        nsj, nmeas = np.linalg.norm(dp_sj), np.linalg.norm(dp_meas)
        if nsj > 1e-9 and nmeas > 1e-9:
            c_k = float(np.dot(dp_sj, dp_meas) / (nsj * nmeas))
            g_k = float(nmeas / nsj)
        else:
            c_k, g_k = np.nan, np.nan
        e_k = -np.asarray(pf[s]["error_mm"], dtype=float) / 1000.0
        e_k1 = -np.asarray(pf[s1]["error_mm"], dtype=float) / 1000.0
        dV_pred = float(2.0 * e_k @ dp_sj)
        dV_real = float(e_k1 @ e_k1 - e_k @ e_k)
        out["step"].append(s); out["ref_idx"].append(pf[s]["ref_index"])
        out["c_k"].append(c_k); out["g_k"].append(g_k)
        out["dV_pred"].append(dV_pred); out["dV_real"].append(dV_real)
        out["e_norm"].append(float(np.linalg.norm(e_k)))
    return {k: np.array(v) for k, v in out.items()}


d12_sj_own = diag_1_2_sj_own(pf_sj, pb_sj, steps_sj, J_SJ_schedule)


def summarize_12(d, label):
    disagree = np.sum((d["dV_pred"] < 0) & (d["dV_real"] > 0))
    n = len(d["c_k"])
    print(f"[1,2] {label}: n={n}, mean c_k={np.nanmean(d['c_k']):.3f}, "
          f"mean g_k={np.nanmean(d['g_k']):.3f}, "
          f"frac(dV_pred<0 & dV_real>0)={disagree}/{n} = {disagree/n:.3f}")
    tail = d["step"] >= 130
    if tail.any():
        print(f"       [tail, step>=130] n={tail.sum()}, mean c_k={np.nanmean(d['c_k'][tail]):.3f}, "
              f"mean g_k={np.nanmean(d['g_k'][tail]):.3f}, "
              f"frac(dV_pred<0&dV_real>0)={np.sum((d['dV_pred'][tail]<0)&(d['dV_real'][tail]>0))}/{tail.sum()}")


summarize_12(d12_fj, "FJ rep1, model=J_FJ (the model it actually ran with)")
summarize_12(d12_sj_vs_fj, "SJ rep2, model=J_FJ (counterfactual: would FJ have predicted SJ's real trajectory well?)")
summarize_12(d12_sj_own, "SJ rep2, model=J_SJ,k (its own genuine per-tick schedule)")

# strongest cross-replay check per the user's spec: evaluate FJ rep1's OWN
# failing trajectory (its own realized states) against the GENUINE per-tick
# schedule value J_SJ,k -- does the honest model predict FJ's own realized
# motion well, even though FJ never had access to it live?
d12_fj_vs_sjsched = diag_1_2_sj_own(pf_fj, pb_fj, steps_fj, J_SJ_schedule)
summarize_12(d12_fj_vs_sjsched, "FJ rep1 (own failing states), model=J_SJ,k (genuine schedule, counterfactual)")

# ---------------------------------------------------------------------
# [3] Jacobian staleness at selected ticks -- relinearize J(z_k) online
# ---------------------------------------------------------------------
print("\n[3] Jacobian staleness: relinearizing J(z_k) at selected FJ rep1 ticks "
      "via the genuine from_model_bundle/accurate adapter...")
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    build_planning_context,
)

_, bundle, controller_pack, _ = build_planning_context()
jac_provider_accurate = from_model_bundle(
    bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate"
)


def principal_angles_deg(A, B):
    """Principal angles (degrees) between the row-spaces of A and B (3x7)."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


ticks = [120, 140, 150, 160, 165, 170]
staleness_rows = []
for t in ticks:
    if t not in pb_fj:
        continue
    z_k = np.asarray(pb_fj[t]["z_meas"], dtype=float)
    J_live = np.asarray(jac_provider_accurate(z_k), dtype=float).reshape(3, 7)
    r_J = float(np.linalg.norm(J_live - J_FJ, "fro") / np.linalg.norm(J_live, "fro"))
    angles = principal_angles_deg(J_FJ, J_live)
    sv_live = np.linalg.svd(J_live, compute_uv=False)
    sv_fj = np.linalg.svd(J_FJ, compute_uv=False)
    ref_idx = pf_fj[t]["ref_index"]
    print(f"  tick={t:4d} ref_idx={ref_idx:4d}  r_J={r_J:.4f}  "
          f"principal_angles_deg={np.round(angles,2)}  "
          f"sv_live={np.round(sv_live,5)}  sv_FJ={np.round(sv_fj,5)}")
    staleness_rows.append(dict(step=t, ref_idx=ref_idx, r_J=r_J,
                                max_angle_deg=float(angles.max())))

# also compute r_J between J_FJ and the GENUINE schedule's own value at each
# ref_idx (cheap, already computed offline in J_SJ_schedule) for the full
# trajectory -- this is the "how stale is FJ, continuously" curve.
r_J_curve_ref_idx = []
r_J_curve_vals = []
for ri in range(J_SJ_schedule.shape[0]):
    Jr = J_SJ_schedule[ri]
    r = float(np.linalg.norm(Jr - J_FJ, "fro") / np.linalg.norm(Jr, "fro"))
    r_J_curve_ref_idx.append(ri)
    r_J_curve_vals.append(r)
r_J_curve_ref_idx = np.array(r_J_curve_ref_idx)
r_J_curve_vals = np.array(r_J_curve_vals)
angle_curve_vals = np.array([
    principal_angles_deg(J_FJ, J_SJ_schedule[ri]).max() for ri in range(J_SJ_schedule.shape[0])
])
print(f"[3] r_J(ref_idx) vs schedule (whole path): min={r_J_curve_vals.min():.4f} "
      f"at ref_idx={r_J_curve_ref_idx[r_J_curve_vals.argmin()]}, "
      f"max={r_J_curve_vals.max():.4f} at ref_idx={r_J_curve_ref_idx[r_J_curve_vals.argmax()]}")

# ---------------------------------------------------------------------
# [7] disturbance-smoothness proxy: one-step prediction residual
# ---------------------------------------------------------------------
def disturbance_proxy(pb, steps, label):
    d = []
    for s in steps:
        pred0 = np.asarray(pb[s]["predicted_beam_positions_m"][0], dtype=float)
        meas = np.asarray(pb[s]["measured_beam_position_m"], dtype=float)
        d.append(meas - pred0)
    d = np.array(d)
    dnorm = np.linalg.norm(d, axis=1)
    ddelta = np.linalg.norm(np.diff(d, axis=0), axis=1)
    print(f"[7] {label}: ||resid|| mean={dnorm.mean()*1000:.4f}mm max={dnorm.max()*1000:.4f}mm, "
          f"||resid_k - resid_k-1|| mean={ddelta.mean()*1000:.4f}mm max={ddelta.max()*1000:.4f}mm")
    return dict(step=np.array(steps), dnorm=dnorm)


dist_fj = disturbance_proxy(pb_fj, steps_fj, "FJ rep1")
dist_sj = disturbance_proxy(pb_sj, steps_sj, "SJ rep2")

# ---------------------------------------------------------------------
# constraint activity (joint step vs hard limit) -- reuse established metric
# ---------------------------------------------------------------------
MAX_JOINT_STEP_RAD = 0.010


def joint_activity(pf, steps):
    out_step, out_ratio = [], []
    for s in steps:
        dq = np.asarray(pf[s]["q_target_delta_rad"], dtype=float)
        out_step.append(s)
        out_ratio.append(float(np.max(np.abs(dq)) / MAX_JOINT_STEP_RAD))
    return np.array(out_step), np.array(out_ratio)


ja_fj_step, ja_fj_ratio = joint_activity(pf_fj, steps_fj)
ja_sj_step, ja_sj_ratio = joint_activity(pf_sj, steps_sj)
print(f"\n[5] max joint-step/limit ratio: FJ rep1 max={ja_fj_ratio.max():.3f} "
      f"(at step {ja_fj_step[ja_fj_ratio.argmax()]}), SJ rep2 max={ja_sj_ratio.max():.3f}")

# ---------------------------------------------------------------------
# THE SINGLE PLOT: common x-axis (ref_idx), FJ vs SJ overlay
# ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=False)
fig.suptitle("FJ rep1 divergence diagnostics vs a clean SJ run (MPC, Q_N on), raised (+30mm) workspace", fontsize=13)

TITLE_KW = dict(fontsize=10)

ax = axes[0, 0]
ax.plot([pf_fj[s]["ref_index"] for s in steps_fj], [pf_fj[s]["error_norm_mm"] for s in steps_fj],
        color="tab:red", label="FJ rep1 (failed)")
ax.plot([pf_sj[s]["ref_index"] for s in steps_sj], [pf_sj[s]["error_norm_mm"] for s in steps_sj],
        color="tab:blue", label="SJ rep2 (clean)")
ax.set_title("||e_k|| (tracking error norm)", **TITLE_KW)
ax.set_xlabel("ref_index"); ax.set_ylabel("mm"); ax.legend(); ax.set_yscale("log")

ax = axes[0, 1]
ax.plot(d12_fj["ref_idx"], d12_fj["c_k"], color="tab:red", label="FJ rep1 (model=J_FJ)")
ax.plot(d12_sj_own["ref_idx"], d12_sj_own["c_k"], color="tab:blue", label="SJ rep2 (model=J_SJ,k)")
ax.axhline(0, color="black", lw=0.6, linestyle=":")
ax.set_title("c_k: directional agreement (cosine sim)\npredicted vs realized dp", **TITLE_KW)
ax.set_xlabel("ref_index"); ax.set_ylabel("cosine similarity"); ax.legend()

ax = axes[1, 0]
ax.plot(r_J_curve_ref_idx, r_J_curve_vals, color="tab:purple", label="r_J(ref_idx) vs SJ schedule")
ax.scatter([row["ref_idx"] for row in staleness_rows], [row["r_J"] for row in staleness_rows],
           color="black", zorder=5, label="relinearized at measured z_k")
ax.set_title("Jacobian staleness r_J\n(SJ schedule + relinearized-at-measured-state)", **TITLE_KW)
ax.set_xlabel("ref_index"); ax.set_ylabel("relative Frobenius error"); ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(r_J_curve_ref_idx, angle_curve_vals, color="tab:purple")
ax.set_title("max principal angle: row(J_FJ) vs row(J_sched(ref_idx))", **TITLE_KW)
ax.set_xlabel("ref_index"); ax.set_ylabel("degrees")

ax = axes[2, 0]
ax.plot(ja_fj_step, ja_fj_ratio, color="tab:red", label="FJ rep1")
ax.plot(ja_sj_step, ja_sj_ratio, color="tab:blue", label="SJ rep2")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.6)
ax.set_title("max-over-joints |dq|/max_step\n(joint-increment constraint activity)", **TITLE_KW)
ax.set_xlabel("step"); ax.set_ylabel("fraction of hard limit"); ax.legend()

ax = axes[2, 1]
dV_agree_fj = (d12_fj["dV_pred"] < 0) & (d12_fj["dV_real"] > 0)
ax.plot(d12_fj["ref_idx"], d12_fj["dV_pred"], color="tab:red", linestyle="-", label="FJ dV_pred (model=J_FJ)")
ax.plot(d12_fj["ref_idx"], d12_fj["dV_real"], color="tab:red", linestyle="--", label="FJ dV_real")
ax.plot(d12_sj_own["ref_idx"], d12_sj_own["dV_pred"], color="tab:blue", linestyle="-", alpha=0.6, label="SJ dV_pred (model=J_SJ,k)")
ax.plot(d12_sj_own["ref_idx"], d12_sj_own["dV_real"], color="tab:blue", linestyle="--", alpha=0.6, label="SJ dV_real")
ax.axhline(0, color="black", lw=0.6)
ax.set_title("Lyapunov-style descent: predicted vs realized\nd(|e|^2)/tick", **TITLE_KW)
ax.set_xlabel("ref_index"); ax.set_ylabel("d(|e|^2)/tick  [m^2]"); ax.legend(fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fj_rep1_diagnostics.png")
fig.savefig(OUT, dpi=150)
print(f"\nsaved -> {OUT}")
