"""Cheap supplementary diagnostic: does the A/B/C replay's wrong-sign failure
correlate with gradient staleness -- i.e. cos angle between g_N(z_ref) (what
the QP actually used) and g_N(z_live) (the true local gradient at the real
measured state at that tick)?

Uses the SAME fd_gradient_normalized + nullspace_projector pipeline as
Phase 0, evaluated at z_live instead of z_ref, for a stratified sample of
replay ticks (half wrong-sign under lambda=10 vs B_none, half correct-sign).
"""
import sys
import json
import pickle
import numpy as np

from proper_research.planning.planning_context import build_planning_context
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.controllers.mpc_delay_aware.target_consistent import build_beam_plane_projection
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from authority_objective import build_Jbar, phi_authority, nullspace_projector, fd_gradient_normalized

PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
N_REF = 265
S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
S_P = np.array([0.5e-3, 0.5e-3])
D_U = np.array([0.10] * 6 + [2.0e-3])
DT = 0.1
C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
EPS = 1e-2

with open("../data/authority_phase0_schedule.pkl", "rb") as f:
    phase0 = pickle.load(f)
with open("../data/authority_abc_replay.pkl", "rb") as f:
    abc = pickle.load(f)

results = abc["results"]

# reconstruct wrong-sign flag per tick for lambda=10 vs B_none (same def as before)
wrong_sign_ticks, correct_sign_ticks = [], []
for r in results:
    c = r["per_mode"]["C_manip_lam10"]
    b = r["per_mode"]["B_none"]
    if c["phi_true1"] is None or b["phi_true1"] is None:
        continue
    dphi_true_rel = c["phi_true1"] - b["phi_true1"]
    dphi_lin_rel = c["dphi_lin"] - b["dphi_lin"]
    entry = dict(run=r["run"], tick=r["tick"], ref_index=r["ref_index"])
    if dphi_lin_rel > 0 and dphi_true_rel <= 0:
        wrong_sign_ticks.append(entry)
    elif dphi_lin_rel > 0 and dphi_true_rel > 0:
        correct_sign_ticks.append(entry)

rng = np.random.default_rng(0)
n_each = 10
wrong_sample = [wrong_sign_ticks[i] for i in rng.choice(len(wrong_sign_ticks), size=min(n_each, len(wrong_sign_ticks)), replace=False)]
correct_sample = [correct_sign_ticks[i] for i in rng.choice(len(correct_sign_ticks), size=min(n_each, len(correct_sign_ticks)), replace=False)]

print(f"wrong-sign pool={len(wrong_sign_ticks)}, correct-sign pool={len(correct_sign_ticks)}")
print(f"sampled {len(wrong_sample)} wrong-sign + {len(correct_sample)} correct-sign ticks\n")

MPC_SJ_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112405Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T112843Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114154Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114458Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T114842Z",
]
run_rows_cache = {}
def get_rows(run_idx):
    if run_idx not in run_rows_cache:
        run_rows_cache[run_idx] = [json.loads(l) for l in open(f"/home/jack/Proper-Research/{MPC_SJ_RUNS[run_idx]}/path_follow.jsonl")]
        pr = [json.loads(l) for l in open(f"/home/jack/Proper-Research/{MPC_SJ_RUNS[run_idx]}/predicted_beam_positions.jsonl")]
        run_rows_cache[run_idx] = (run_rows_cache[run_idx], pr)
    return run_rows_cache[run_idx]

print("[setup] building state-dependent Jacobian provider...")
_, bundle, controller_pack, _ = build_planning_context()
jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
print("[setup] done.\n")


def phi_fn_for(z_base):
    def phi(z):
        J = C @ jac_provider(z)
        Jbar = build_Jbar(J, S_P, D_U, DT)
        p, _ = phi_authority(Jbar)
        return p
    return phi


def g_N_at(z):
    phi_fn = phi_fn_for(z)
    g = fd_gradient_normalized(phi_fn, z, S_Z, EPS)
    J_task = C @ jac_provider(z)
    _, P_N, _ = nullspace_projector(J_task, S_Z)
    return P_N @ g


rows_out = []
for label, sample in (("WRONG-SIGN", wrong_sample), ("correct-sign", correct_sample)):
    print(f"--- {label} ---")
    for entry in sample:
        rows, _ = get_rows(entry["run"])
        r = rows[entry["tick"]]
        q_meas = np.asarray(r["q_meas_rad"], dtype=float)
        L_meas = float(r["insertion_length_m"])
        z_live = np.concatenate([q_meas, [L_meas]])
        idx = entry["ref_index"]
        g_ref = phase0["g_N"][idx]
        g_live = g_N_at(z_live)
        na, nb = np.linalg.norm(g_ref), np.linalg.norm(g_live)
        cos_a = float(np.dot(g_ref, g_live) / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")
        print(f"  run={entry['run']} tick={entry['tick']:3d} idx={idx:3d}: "
              f"||g_ref||={na:.3f} ||g_live||={nb:.3f} cos_angle={cos_a:+.3f}")
        rows_out.append(dict(label=label, **entry, cos_angle=cos_a, norm_g_ref=na, norm_g_live=nb))

wrong_cos = [r["cos_angle"] for r in rows_out if r["label"] == "WRONG-SIGN"]
correct_cos = [r["cos_angle"] for r in rows_out if r["label"] == "correct-sign"]
print(f"\nWRONG-SIGN  cos_angle: mean={np.nanmean(wrong_cos):+.3f}  sd={np.nanstd(wrong_cos):.3f}")
print(f"correct-sign cos_angle: mean={np.nanmean(correct_cos):+.3f}  sd={np.nanstd(correct_cos):.3f}")

with open("../data/authority_gradient_staleness.pkl", "wb") as f:
    pickle.dump(rows_out, f)
print("\nsaved authority_gradient_staleness.pkl")
