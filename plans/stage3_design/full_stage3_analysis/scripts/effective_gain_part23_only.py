"""Re-run of Parts 2-3 only (Part 1, k_parallel/k_perp, already completed
and saved by effective_gain_and_baseline_analysis.py -- this just fixes a
bug in Part 2 without repeating Part 1's ~45min of live-model calls).
Self-contained: does not import the Part-1 script, to avoid re-triggering
its module-level execution."""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import manifest
import loader

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
REPO = "/home/jack/Proper-Research"
SCHED_C_REPAIRED = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy"
SCHED_NC = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy"
DT = 0.1


def damped_pseudo_inverse(J, damping):
    gram = J @ J.T + (damping ** 2) * np.eye(3)
    return J.T @ np.linalg.solve(gram, np.eye(3))


def raw_commands(out, meta, schedule, desired_position_m):
    N = len(out["step"])
    position_gain = float(meta["position_gain"]); damping = float(meta["damping"])
    n_sched = schedule.shape[0]; n_ref = desired_position_m.shape[0]
    VEL_LIMIT = np.array([0.1] * 6 + [0.002])
    ACC_LIMIT = np.array([0.4] * 6 + [10.0 * 0.002])
    STATE_MIN = np.array([-2 * np.pi] * 6 + [-0.05])
    STATE_MAX = np.array([2 * np.pi] * 6 + [0.20])

    def clip_command(command, state, previous):
        command = np.clip(command, -VEL_LIMIT, VEL_LIMIT)
        command = np.clip(command, previous - ACC_LIMIT * DT, previous + ACC_LIMIT * DT)
        command = np.clip(command, (STATE_MIN - state) / DT, (STATE_MAX - state) / DT)
        return np.clip(command, -VEL_LIMIT, VEL_LIMIT)

    raw_cmd = np.zeros((N, 7)); prev_raw = np.zeros(7)
    q = out["q_meas_rad"]; L = out["insertion_length_m"]; tip_m = out["tip_mm"] / 1000.0
    ref_idx = out["ref_index"]
    for k in range(N):
        state = np.concatenate([q[k], [L[k]]])
        measured = tip_m[k]
        index = int(np.clip(ref_idx[k] + 1, 0, min(n_sched, n_ref) - 1))
        J = schedule[index]; desired = desired_position_m[index]
        pseudo = damped_pseudo_inverse(J, damping)
        task_velocity = pseudo @ (position_gain * (desired - measured) / DT)
        command = clip_command(task_velocity, state, prev_raw)
        raw_cmd[k] = command; prev_raw = command.copy()
    return raw_cmd


def get_runs(group):
    return sorted([rm for rm in manifest.RUNS if rm["group"] == group], key=lambda r: r["rep"])


sel_c = get_runs("invjac_selective")
sel_nc = get_runs("invjac_clip_nc")
PAIRINGS = [("selective_contact", sel_c, SCHED_C_REPAIRED), ("selective_nocontact", sel_nc, SCHED_NC)]

# ===================== PART 2: gain-ablation re-analysis =====================
g06 = [r for r in get_runs("closedloop") if r["condition"] == "invjac_NC" and r["radius_intended_mm"] == 210]
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
print(f"[gain-p23] wrote gain_ablation_lag_reanalysis.csv")
print(ga_df.to_string(index=False))
print()
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
print(f"\n[gain-p23] wrote excess_lag_baseline.csv")
post_contact = bl_df[bl_df.s_mm > 28.5]
print("\n--- post-contact-onset (s>28.5mm) excess lag (DT-corrected baseline) by pairing ---")
print(post_contact.groupby("pairing")["e_excess_dt_mm"].agg(["mean", "median", "max", "count"]).to_string())
pre_contact = bl_df[bl_df.s_mm <= 28.5]
print("\n--- pre-contact (s<=28.5mm) excess lag by pairing ---")
print(pre_contact.groupby("pairing")["e_excess_dt_mm"].agg(["mean", "median", "max", "count"]).to_string())
print("\n--- sanity: baseline magnitude check, one sample row ---")
print(bl_df[["v_s_m_s", "Kp", "e_lag_expected_dt_mm", "e_lag_expected_literal_mm"]].head(3).to_string())
