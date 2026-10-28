"""Effective longitudinal gain (k_parallel/k_perp), gain-ablation lag
re-analysis, and theoretical lag baseline -- requested by an external
reviewer of STAGE3_FINAL_REPORT_REVISED.md to test whether the wrong
(no-contact) Jacobian reduces the inverse-Jacobian controller's effective
closed-loop gain in the direction of travel, as a mechanistic driver of
lag -> command growth -> redundant drift.

Part 1: k_parallel(s) = Kp * t^T J_C^state(x_k) Jhat_k^dagger t,
        k_perp(s)      = ||(I - t t^T) J_C^state(x_k) Jhat_k^dagger t||
  for the 6 selective-gate runs (invjac_selective = contact Jacobian,
  invjac_clip_nc = no-contact Jacobian), moderate stride across the full
  run. Jhat_k^dagger is the controller's own damped-least-squares pseudo-
  inverse of the SCHEDULED Jacobian at that tick (same formula as
  h3_replay.py's replay_raw_commands); J_C^state is the contact model
  recomputed at the exact measured state (the one physically-grounded
  reference used throughout this investigation) -- used for BOTH pairings
  (this is "what actually happens physically", not each pairing's own
  model), per the reviewer's own M_k = J_C^state(x_k) Jhat_k^dagger
  definition.

Part 2: for the 6 gain-ablation runs (Kp=0.6 vs 1.0, both no-contact
  Jacobian, hold gate, 210mm), per-tick lag/cross-track stats (mean/
  median/p95/final) plus u_raw reconstruction (reusing h3_replay.py's
  exact math) and its norm stats, plus early/late E_N at a coarse stride.

Part 3: theoretical scalar proportional-tracking lag baseline. NOTE: the
  reviewer's literal formula e_lag_expected ~= v_s/Kp is dimensionally
  inconsistent (v_s in m/s, Kp dimensionless -> result in m/s, not m).
  Deriving it properly from the controller's own discrete update law
  (u_k*DT = Kp*(s_ref_k - s_tip_k), i.e. the controller corrects a Kp
  fraction of the current error every tick) gives a steady-state balance
  v_s*DT = Kp*e_ss, i.e. e_lag_expected = v_s*DT/Kp (DT=0.1s here) -- this
  is what this script reports as the primary baseline, with the literal
  v_s/Kp (no DT) also reported alongside for direct comparison with the
  reviewer's request.

Read-only against existing run data, schedules, and source; writes only
new scripts/tables/figures.
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
REPO = "/home/jack/Proper-Research"
SCHED_C_REPAIRED = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy"
SCHED_NC = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy"
DT = 0.1


def damped_pseudo_inverse(J, damping):
    gram = J @ J.T + (damping ** 2) * np.eye(3)
    return J.T @ np.linalg.solve(gram, np.eye(3))


def raw_commands(out, meta, schedule, desired_position_m):
    """Exact reproduction of h3_replay.py's replay_raw_commands (damped
    least-squares, zero nullspace_gain, zero feedforward, same clip)."""
    N = len(out["step"])
    position_gain = float(meta["position_gain"])
    damping = float(meta["damping"])
    n_sched = schedule.shape[0]
    n_ref = desired_position_m.shape[0]
    VEL_LIMIT = np.array([0.1] * 6 + [0.002])
    ACC_LIMIT = np.array([0.4] * 6 + [10.0 * 0.002])
    STATE_MIN = np.array([-2 * np.pi] * 6 + [-0.05])
    STATE_MAX = np.array([2 * np.pi] * 6 + [0.20])

    def clip_command(command, state, previous):
        command = np.clip(command, -VEL_LIMIT, VEL_LIMIT)
        command = np.clip(command, previous - ACC_LIMIT * DT, previous + ACC_LIMIT * DT)
        command = np.clip(command, (STATE_MIN - state) / DT, (STATE_MAX - state) / DT)
        return np.clip(command, -VEL_LIMIT, VEL_LIMIT)

    raw_cmd = np.zeros((N, 7))
    prev_raw = np.zeros(7)
    q = out["q_meas_rad"]; L = out["insertion_length_m"]; tip_m = out["tip_mm"] / 1000.0
    ref_idx = out["ref_index"]
    for k in range(N):
        state = np.concatenate([q[k], [L[k]]])
        measured = tip_m[k]
        index = int(np.clip(ref_idx[k] + 1, 0, min(n_sched, n_ref) - 1))
        J = schedule[index]
        desired = desired_position_m[index]
        pseudo = damped_pseudo_inverse(J, damping)
        task_velocity = pseudo @ (position_gain * (desired - measured) / DT)
        command = clip_command(task_velocity, state, prev_raw)
        raw_cmd[k] = command
        prev_raw = command.copy()
    return raw_cmd


def get_runs(group):
    return sorted([rm for rm in manifest.RUNS if rm["group"] == group], key=lambda r: r["rep"])


t_start = time.time()
live_jac.get_context()
print(f"[gain] context ready at {time.time()-t_start:.0f}s", flush=True)

sel_c = get_runs("invjac_selective")
sel_nc = get_runs("invjac_clip_nc")
PAIRINGS = [("selective_contact", sel_c, SCHED_C_REPAIRED), ("selective_nocontact", sel_nc, SCHED_NC)]

# ===================== PART 1: k_parallel / k_perp =====================
STRIDE_TARGET = 120  # live evals per run
kp_rows = []
for pairing, runs, sched_path in PAIRINGS:
    schedule = np.load(sched_path)
    for rm in runs:
        out, meta = loader.enrich_run(rm)
        ref = loader.get_reference(meta["plan_dir"])
        desired_tangent = np.asarray(ref["desired_tangent"], dtype=float)
        Kp = float(meta["position_gain"]); damping = float(meta["damping"])
        N = len(out["step"])
        stride = max(1, N // STRIDE_TARGET)
        idxs = np.arange(0, N, stride)
        t0 = time.time()
        for k in idxs:
            q = out["q_meas_rad"][k]; L = out["insertion_length_m"][k]
            state7 = np.concatenate([q, [L]])
            ref_idx = int(np.clip(out["ref_index"][k] + 1, 0, schedule.shape[0] - 1))
            J_sched = schedule[ref_idx]
            pseudo = damped_pseudo_inverse(J_sched, damping)
            J_C_state = live_jac.live_jacobian(state7, True)
            M = J_C_state @ pseudo  # (3,3)
            t = desired_tangent[min(ref_idx, desired_tangent.shape[0] - 1)]
            t = t / max(np.linalg.norm(t), 1e-12)
            Mt = M @ t
            k_par = Kp * float(t @ Mt)
            k_perp = float(np.linalg.norm(Mt - t * float(t @ Mt)))
            kp_rows.append(dict(
                pairing=pairing, dirname=meta["dirname"], rep=rm["rep"],
                tick=int(k), s_mm=float(out["s_ref_mm"][k]),
                k_parallel=k_par, k_perp=k_perp, Kp=Kp,
            ))
        print(f"[gain] {pairing} rep{rm['rep']}: {len(idxs)} samples in {time.time()-t0:.0f}s "
              f"(stride={stride})", flush=True)

kp_df = pd.DataFrame(kp_rows)
kp_df.to_csv(f"{OUT}/tables/k_parallel_k_perp_selective.csv", index=False)
print(f"\n[gain] wrote k_parallel_k_perp_selective.csv ({len(kp_df)} rows)")
print(kp_df.groupby("pairing")[["k_parallel", "k_perp"]].describe().to_string())
late = kp_df[kp_df.s_mm > 45]
print("\n--- late path (s>45mm) k_parallel by pairing ---")
print(late.groupby("pairing")["k_parallel"].agg(["mean", "median", "min", "max", "count"]).to_string())

# figure
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
colors = {"selective_contact": "#1b7f3b", "selective_nocontact": "#b3331d"}
for pairing in ["selective_contact", "selective_nocontact"]:
    sub = kp_df[kp_df.pairing == pairing]
    for rep, d in sub.groupby("rep"):
        d = d.sort_values("s_mm")
        axes[0].plot(d.s_mm, d.k_parallel, color=colors[pairing], lw=1.0, alpha=0.75,
                     label=pairing if rep == 1 else None)
        axes[1].plot(d.s_mm, d.k_perp, color=colors[pairing], lw=1.0, alpha=0.75)
axes[0].axhline(0.6, color="gray", ls="--", lw=0.8, label="nominal $K_p$=0.6")
axes[0].axvline(28.5, color="k", ls=":", lw=0.8, label="contact onset (s=28.5mm)")
axes[1].axvline(28.5, color="k", ls=":", lw=0.8)
axes[0].set_ylabel(r"$k_\parallel$"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_ylabel(r"$k_\perp$"); axes[1].set_xlabel("path progress s (mm)"); axes[1].grid(alpha=0.3)
fig.suptitle("Effective longitudinal/transverse gain: selective contact vs no-contact Jacobian")
fig.tight_layout()
fig.savefig(f"{OUT}/figures/k_parallel_vs_s.png", dpi=160)
plt.close(fig)
print(f"[gain] saved figures/k_parallel_vs_s.png")

# ===================== PART 2: gain-ablation re-analysis =====================
g06 = get_runs("closedloop")
g06 = [r for r in g06 if r["condition"] == "invjac_NC" and r["radius_intended_mm"] == 210]
g10 = get_runs("gain_ablation")
ga_rows = []
_sched_nc_arr = np.load(SCHED_NC)
for gain_label, runs in [("0.6", g06), ("1.0", g10)]:
    for rm in runs:
        out, meta = loader.enrich_run(rm)
        ref = loader.get_reference(meta["plan_dir"])
        desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)
        lag = out["e_lag_mm"]; cross = out["cross_track_mm"]
        raw = raw_commands(out, meta, _sched_nc_arr, desired_position_m)
        norm_raw = np.linalg.norm(raw, axis=1)
        ga_rows.append(dict(
            gain=gain_label, dirname=meta["dirname"], rep=rm["rep"],
            lag_mean=float(np.mean(lag)), lag_median=float(np.median(lag)),
            lag_p95=float(np.percentile(lag, 95)), lag_final=float(lag[-1]),
            cross_mean=float(np.mean(cross)), cross_median=float(np.median(cross)),
            cross_p95=float(np.percentile(cross, 95)), cross_final=float(cross[-1]),
            u_raw_mean=float(np.mean(norm_raw)), u_raw_p95=float(np.percentile(norm_raw, 95)),
        ))
ga_df = pd.DataFrame(ga_rows)
ga_df.to_csv(f"{OUT}/tables/gain_ablation_lag_reanalysis.csv", index=False)
print(f"\n[gain] wrote gain_ablation_lag_reanalysis.csv")
print(ga_df.groupby("gain")[["lag_mean", "lag_median", "lag_p95", "lag_final",
                             "cross_mean", "cross_p95", "u_raw_mean"]].mean().to_string())

# ===================== PART 3: theoretical lag baseline =====================
bl_rows = []
for pairing, runs, _ in PAIRINGS:
    for rm in runs:
        out, meta = loader.enrich_run(rm)
        ref = loader.get_reference(meta["plan_dir"])
        v_s_arr = np.asarray(ref["path_speed_m_s"], dtype=float)
        Kp = float(meta["position_gain"])
        N = len(out["step"])
        ref_idx = np.clip(out["ref_index"], 0, v_s_arr.shape[0] - 1)
        v_s = v_s_arr[ref_idx]
        e_lag_m = out["e_lag_mm"] / 1000.0
        e_expected_dt = v_s * DT / Kp
        e_expected_literal = v_s / Kp
        for k in range(N):
            bl_rows.append(dict(
                pairing=pairing, dirname=meta["dirname"], rep=rm["rep"],
                tick=int(k), s_mm=float(out["s_ref_mm"][k]), v_s_m_s=float(v_s[k]), Kp=Kp,
                e_lag_measured_mm=float(out["e_lag_mm"][k]),
                e_lag_expected_dt_mm=float(e_expected_dt[k] * 1000.0),
                e_lag_expected_literal_mm=float(e_expected_literal[k] * 1000.0),
                e_excess_dt_mm=float((e_lag_m[k] - e_expected_dt[k]) * 1000.0),
            ))
bl_df = pd.DataFrame(bl_rows)
bl_df.to_csv(f"{OUT}/tables/excess_lag_baseline.csv", index=False)
print(f"\n[gain] wrote excess_lag_baseline.csv")
post_contact = bl_df[bl_df.s_mm > 28.5]
print("\n--- post-contact-onset (s>28.5mm) excess lag (DT-corrected baseline) by pairing ---")
print(post_contact.groupby("pairing")["e_excess_dt_mm"].agg(["mean", "median", "max", "count"]).to_string())
pre_contact = bl_df[bl_df.s_mm <= 28.5]
print("\n--- pre-contact (s<=28.5mm) excess lag by pairing ---")
print(pre_contact.groupby("pairing")["e_excess_dt_mm"].agg(["mean", "median", "max", "count"]).to_string())

print(f"\n[gain] DONE in {time.time()-t_start:.0f}s", flush=True)
