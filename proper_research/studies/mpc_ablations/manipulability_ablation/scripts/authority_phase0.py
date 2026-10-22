"""Phase 0: cheap decision-gate diagnostic (spec section 3), run along the
FULL reference trajectory (all 265 rectangle indices -- pre-checks showed
~0.22s per state-dependent Jacobian eval, ~15 total per index for phi+FD
gradient, so ~15min total; acceptable and gives full-path coverage for the
Section 4 conflict diagnostic that reuses this cache against real replay
logs).

Caches EVERYTHING per reference index (spec section 6): phi, singular
values, g, P_N, g_N -- the state-dependent beam model is evaluated ONCE per
index here, never again downstream.
"""
import sys
import time
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
DT = 0.1
S_Z = np.array([np.radians(0.5)] * 6 + [0.25e-3])
S_P = np.array([0.5e-3, 0.5e-3])
D_U = np.array([0.10] * 6 + [2.0e-3])
EPS = 1.0e-2
C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))

reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
_, bundle, controller_pack, _ = build_planning_context()
jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
z_ref_all = np.asarray(reference.state, dtype=float)
N_ref = z_ref_all.shape[0]


def J_task_at(z):
    return C @ jac_provider(z)


def phi_at(z):
    Jbar = build_Jbar(J_task_at(z), S_P, D_U, DT)
    phi, _ = phi_authority(Jbar)
    return phi


schedule = {
    "phi": np.full(N_ref, np.nan), "sv": [None] * N_ref,
    "g": np.full((N_ref, 7), np.nan), "P_N": [None] * N_ref, "g_N": np.full((N_ref, 7), np.nan),
    "rank": np.full(N_ref, -1, dtype=int),
    "dphi_true_a25": np.full(N_ref, np.nan), "dphi_true_a50": np.full(N_ref, np.nan),
    "sv_after_a50": [None] * N_ref,
}

t0 = time.perf_counter()
for k in range(N_ref):
    z0 = z_ref_all[k]
    J0 = J_task_at(z0)
    Jbar0 = build_Jbar(J0, S_P, D_U, DT)
    phi0, sv0 = phi_authority(Jbar0)
    _, P_N, rank = nullspace_projector(J0, S_Z)
    g = fd_gradient_normalized(phi_at, z0, S_Z, EPS)
    g_N = P_N @ g

    schedule["phi"][k] = phi0
    schedule["sv"][k] = sv0
    schedule["g"][k] = g
    schedule["P_N"][k] = P_N
    schedule["g_N"][k] = g_N
    schedule["rank"][k] = rank

    norm_gN = np.linalg.norm(g_N)
    if norm_gN > 1e-9:
        v_N = g_N / norm_gN
        for alpha, key in ((0.25, "dphi_true_a25"), (0.5, "dphi_true_a50")):
            z_test = z0 + S_Z * (alpha * v_N)
            phi_test = phi_at(z_test)
            schedule[key][k] = phi_test - phi0
            if alpha == 0.5:
                Jbar_test = build_Jbar(J_task_at(z_test), S_P, D_U, DT)
                _, sv_test = phi_authority(Jbar_test)
                schedule["sv_after_a50"][k] = sv_test
    if k % 20 == 0:
        elapsed = time.perf_counter() - t0
        print(f"[{k:3d}/{N_ref}] phi={phi0:.4f} rank={rank} ||g||={np.linalg.norm(g):.4f} "
              f"||g_N||={norm_gN:.4f}  elapsed={elapsed:.0f}s", flush=True)

with open("../data/authority_phase0_schedule.pkl", "wb") as f:
    pickle.dump(schedule, f)
print(f"\ndone in {time.perf_counter()-t0:.0f}s, saved authority_phase0_schedule.pkl")
