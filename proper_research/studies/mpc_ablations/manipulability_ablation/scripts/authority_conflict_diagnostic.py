"""Section 4: conflict diagnostic between the CURRENT reference-nullspace
anchor and the proposed authority-ascent direction, evaluated on REAL
MPC-SJ rectangle live states (not the bare reference trajectory -- at
z=z_ref exactly, dz_norm=0 trivially, so this needs genuine tracking
deviation, which only real replay data has). Reuses Phase 0's cached
P_N/g_N schedule (no new expensive state-dependent Jacobian evaluations).
"""
import sys
import json
import pickle
import numpy as np

S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
N_REF = 265

with open("../data/authority_phase0_schedule.pkl", "rb") as f:
    sched = pickle.load(f)

MPC_SJ_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112405Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112843Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114154Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114458Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114842Z",
]


def load_ref_state():
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )
    PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    return np.asarray(ref.state, dtype=float)


z_ref_all = load_ref_state()

cos_thetas = []
ref_indices_seen = []
norm_gN_seen = []
for run_dir in MPC_SJ_RUNS:
    rows = [json.loads(l) for l in open(f"/home/jack/Proper-Research/{run_dir}/path_follow.jsonl")]
    for r in rows:
        idx = min(int(r["ref_index"]), N_REF - 1)
        g_N = sched["g_N"][idx]
        norm_gN = np.linalg.norm(g_N)
        if norm_gN < 1e-9:
            continue  # no meaningful authority-ascent direction here -- skip, not a conflict
        q_meas = np.asarray(r["q_meas_rad"], dtype=float)
        L_meas = float(r["insertion_length_m"])
        z_actual = np.concatenate([q_meas, [L_meas]])
        z_ref_k = z_ref_all[idx]
        dz_norm = (z_actual - z_ref_k) / S_Z
        P_N = sched["P_N"][idx]
        v_anchor = -(P_N @ dz_norm)
        norm_va = np.linalg.norm(v_anchor)
        if norm_va < 1e-9:
            continue  # controller already exactly at the reference in the nullspace -- no anchor direction to compare
        cos_t = float(np.dot(v_anchor, g_N) / (norm_va * norm_gN))
        cos_thetas.append(cos_t)
        ref_indices_seen.append(idx)
        norm_gN_seen.append(norm_gN)

cos_thetas = np.array(cos_thetas)
ref_indices_seen = np.array(ref_indices_seen)
print(f"n_ticks_with_meaningful_comparison = {len(cos_thetas)}")
print(f"cos_theta: mean={cos_thetas.mean():.3f}  median={np.median(cos_thetas):.3f}  "
      f"sd={cos_thetas.std():.3f}")
print(f"  frac > +0.5 (anchor already authority-aligned)  = {np.mean(cos_thetas > 0.5):.3f}")
print(f"  frac in [-0.5, 0.5] (largely unrelated)          = {np.mean((cos_thetas>=-0.5)&(cos_thetas<=0.5)):.3f}")
print(f"  frac < -0.5 (anchor OPPOSES authority ascent)    = {np.mean(cos_thetas < -0.5):.3f}")

# where along the path do strongly negative values occur?
strong_neg = ref_indices_seen[cos_thetas < -0.5]
if strong_neg.size:
    print(f"\nref_index bins with cos_theta < -0.5 (n={strong_neg.size}):")
    hist, edges = np.histogram(strong_neg, bins=10, range=(0, N_REF))
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        if h > 0:
            print(f"  [{e0:.0f},{e1:.0f}): {h}")
else:
    print("\nno ticks with cos_theta < -0.5")

with open("../data/authority_conflict_diagnostic.pkl", "wb") as f:
    pickle.dump(dict(cos_thetas=cos_thetas, ref_indices=ref_indices_seen, norm_gN=np.array(norm_gN_seen)), f)
print("\nsaved authority_conflict_diagnostic.pkl")
