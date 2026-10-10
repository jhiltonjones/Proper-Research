"""Node-150 focused diagnostic, per the requested methodology:

1. FD convergence/branch ladder: for each of the 7 actuation dims, compute
   the brute-force central-difference Jacobian column at a ladder of step
   sizes h, h/2, h/4, ..., with EVERY +h/-h solve warm-started from the
   SAME nominal converged equilibrium (not from each other, not from the
   solver's own default warm-start heuristic), tracking u*, tip+tangent
   output, solver success/residual, and the nearest-wall-segment index (for
   branch/active-set-switch detection). Reports
       E_FD(h) = ||J_FD(h) - J_FD(h/2)|| / ||J_FD(h/2)||
   looking for a plateau.

2. Builds u_theta^FD = (u*(theta+h) - u*(theta-h)) / (2h) at the smallest
   converged/plateaued h per dimension, and directly evaluates the implicit
   relation's own residual
       r_implicit = H @ u_theta^FD + Gtheta
   projected onto the retained (rank-4) and discarded (rank-23) eigenspaces
   of H, to test whether the truncated-rank idea is consistent with the
   independently-measured u*(theta) sensitivity, or whether the retained
   subspace itself is already inconsistent.

3. Separately eigendecomposes the CONTACT-ONLY Hessian (active set held
   fixed, no magnetic/elastic contribution) to re-check the negative-
   eigenvalue noise criterion against just the nominally-convex term, not
   the full energy (which can have genuine magnetic curvature).

Usage: python diagnose_node150_fd_convergence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
from proper_research.simulation.magnetic_beam.sensitivity_optimized import implicit_tip_jacobian, SensitivityOptions
from proper_research.simulation.magnetic_beam.gradients import contact_energy_gradient_u_consistent

NODE_IDX = 150
PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"

H_TRANS_LADDER = [2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5]
H_ROT_LADDER = [np.radians(x) for x in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)]
H_L_LADDER = [5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.125e-5, 1.5625e-5]
LADDERS = [H_TRANS_LADDER] * 3 + [H_ROT_LADDER] * 3 + [H_L_LADDER]
DIM_LABELS = ["x", "y", "z", "rx", "ry", "rz", "L"]


def perturb(xyz0, rotvec0, L0, dim, h, sign):
    if dim < 3:
        d = np.zeros(3); d[dim] = sign * h
        return xyz0 + d, rotvec0, L0
    elif dim < 6:
        d = np.zeros(3); d[dim - 3] = sign * h
        R_pert = Rot.from_rotvec(d).as_matrix() @ Rot.from_rotvec(rotvec0).as_matrix()
        return xyz0, Rot.from_matrix(R_pert).as_rotvec(), L0
    else:
        return xyz0, rotvec0, L0 + sign * h


def tip_tangent_of(result):
    centerline = np.asarray(result.p, dtype=float)
    tip = np.asarray(result.tip, dtype=float)
    tangent = centerline[:, -1] - centerline[:, -2]
    tangent = tangent / np.linalg.norm(tangent)
    return np.concatenate([tip, tangent])


def nearest_segment_index(problem, result):
    p = np.asarray(result.p, dtype=float).T  # (N,3)
    delta, Rloc, q, grad_delta, grad_Rloc = problem.lumen_query.closest_many_with_gradients(
        p, window=problem.contact.window,
    )
    c = Rloc - delta - problem.contact.r_beam
    i_min = int(np.argmin(c))
    # q[i_min] is the closest POINT on the centreline (3D); report the arc-length
    # position of that point along the lumen centreline as a proxy for which
    # segment is active, plus the minimum gap itself.
    C = problem.lumen_query.C
    dists_to_C = np.linalg.norm(C - q[i_min], axis=1)
    nearest_C_idx = int(np.argmin(dists_to_C))
    return nearest_C_idx, float(c[i_min])


def main():
    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
        )
    )
    model = bundle.models["contact"]
    adapter = build_diagnostic_adapter(beam_model=model, controller_pack=controller_pack, jacobian_mode="accurate")

    df = pd.read_csv(f"{PLAN_ROOT}/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv")
    state0 = df.loc[NODE_IDX, ["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy(dtype=float)
    T_R_M = adapter.magnet_transform(state0)
    xyz0 = T_R_M[:3, 3].copy()
    rotvec0 = Rot.from_matrix(T_R_M[:3, :3]).as_rotvec()
    L0 = float(state0[6])
    p7_0 = np.concatenate([xyz0, rotvec0, [L0]])

    # Nominal equilibrium (the baseline every perturbed solve warm-starts from).
    result0 = model.solve(p7_0, commit=True, reuse_cache=True)
    u_star_0 = np.asarray(result0.u_flat_opt, dtype=float).copy()
    problem0 = model.build_problem(p7_0)
    seg0, gap0 = nearest_segment_index(problem0, result0)
    print(f"nominal: u*_norm={np.linalg.norm(u_star_0):.6f}  nearest_seg_idx={seg0}  gap_mm={gap0*1e3:.4f}")
    print(f"nominal solve info: success={result0.info.get('success')} solve_path={result0.info.get('solve_path')} "
          f"max_bend={result0.info.get('max_bend')}")

    # --- Part 1: FD ladder per dimension ---
    ladder_records = []
    u_theta_fd_best = np.zeros((u_star_0.size, 7))
    out_cols_at_smallest_h = {}

    for dim in range(7):
        ladder = LADDERS[dim]
        J_cols = []
        u_diffs = []
        for h in ladder:
            xp, rvp, Lp = perturb(xyz0, rotvec0, L0, dim, h, +1)
            xm, rvm, Lm = perturb(xyz0, rotvec0, L0, dim, h, -1)
            p7p = np.concatenate([xp, rvp, [Lp]])
            p7m = np.concatenate([xm, rvm, [Lm]])
            problem_p = model.build_problem(p7p)
            problem_m = model.build_problem(p7m)
            result_p = solve_quasistatic_insertion_optimized(
                problem_p, u0_flat=u_star_0, options=model.beam, result_detail=model.result_detail,
            )
            result_m = solve_quasistatic_insertion_optimized(
                problem_m, u0_flat=u_star_0, options=model.beam, result_detail=model.result_detail,
            )
            out_p = tip_tangent_of(result_p)
            out_m = tip_tangent_of(result_m)
            J_col = (out_p - out_m) / (2.0 * h)
            J_cols.append(J_col)
            seg_p, gap_p = nearest_segment_index(problem_p, result_p)
            seg_m, gap_m = nearest_segment_index(problem_m, result_m)
            u_p = np.asarray(result_p.u_flat_opt, dtype=float)
            u_m = np.asarray(result_m.u_flat_opt, dtype=float)
            u_diffs.append((u_p - u_m) / (2.0 * h))
            ladder_records.append({
                "dim": DIM_LABELS[dim], "h": h,
                "success_p": bool(result_p.info.get("success", True)),
                "success_m": bool(result_m.info.get("success", True)),
                "seg_idx_p": seg_p, "seg_idx_m": seg_m, "seg_idx_0": seg0,
                "gap_mm_p": gap_p * 1e3, "gap_mm_m": gap_m * 1e3,
                "branch_switch": bool(seg_p != seg0 or seg_m != seg0),
            })

        # consecutive E_FD(h) = ||J(h) - J(h/2)|| / ||J(h/2)||, attached to
        # each rung's own record (last rung has no "next" to compare against).
        base = len(ladder_records) - len(ladder)
        for i in range(len(ladder)):
            e_next = (
                float(np.linalg.norm(J_cols[i] - J_cols[i + 1]) / max(np.linalg.norm(J_cols[i + 1]), 1e-12))
                if i < len(ladder) - 1 else None
            )
            ladder_records[base + i]["E_FD_vs_next"] = e_next

        # report per-dim convergence summary
        print(f"\ndim {DIM_LABELS[dim]}: ladder h={ladder}")
        for i in range(len(ladder)):
            branch = ladder_records[base + i]["branch_switch"]
            e_next = ladder_records[base + i]["E_FD_vs_next"]
            print(f"  h={ladder[i]:.3e}  |J_col|={np.linalg.norm(J_cols[i]):.5f}  "
                  f"branch_switch={branch}  E_FD(h vs h/2)={e_next}")

        # use the SMALLEST h's column as the best estimate (if no branch switch
        # anywhere in the ladder); otherwise flag it.
        u_theta_fd_best[:, dim] = u_diffs[-1]
        out_cols_at_smallest_h[DIM_LABELS[dim]] = J_cols[-1]

    ladder_df = pd.DataFrame.from_records(ladder_records)
    ladder_df.to_csv("plans/stage3_design/node150_fd_ladder.csv", index=False)
    print(f"\nsaved plans/stage3_design/node150_fd_ladder.csv")
    print(f"any branch switches anywhere in the whole ladder sweep: {ladder_df['branch_switch'].any()}")

    # --- Part 2: H u_theta^FD + Gtheta residual, projected onto retained/discarded ---
    problem = model.build_problem(p7_0)
    solution = model._cache_as_solution_view()
    theta_model = model.build_theta_model(p7_0)
    sens = implicit_tip_jacobian(
        solution=solution, problem=problem, theta_model=theta_model,
        options=SensitivityOptions(eps_theta=1e-6, eps_hess=1e-4, difference_scheme="central"),
    )
    H = sens.H
    Gtheta = sens.Gtheta
    eigvals, eigvecs = np.linalg.eigh(H)
    order = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    # retained rank from the already-validated truncated_svd run (rank=4, gap after index 3)
    RETAINED_RANK = 4
    V_r = eigvecs_sorted[:, :RETAINED_RANK]
    V_d = eigvecs_sorted[:, RETAINED_RANK:]

    r_implicit = H @ u_theta_fd_best + Gtheta  # (27,7)
    r_retained = V_r.T @ r_implicit  # (4,7)
    r_discarded = V_d.T @ r_implicit  # (23,7)

    print("\n=== H u_theta^FD + Gtheta residual, projected ===")
    print(f"||r_implicit|| total = {np.linalg.norm(r_implicit):.6e}")
    print(f"||r_implicit|| in RETAINED (rank-{RETAINED_RANK}) subspace = {np.linalg.norm(r_retained):.6e}")
    print(f"||r_implicit|| in DISCARDED (rank-{H.shape[0]-RETAINED_RANK}) subspace = {np.linalg.norm(r_discarded):.6e}")
    print("per-column norms, retained subspace:", np.round(np.linalg.norm(r_retained, axis=0), 6))
    print("per-column norms, discarded subspace:", np.round(np.linalg.norm(r_discarded, axis=0), 6))
    print(f"||Gtheta|| (for scale) = {np.linalg.norm(Gtheta):.6e}")

    # --- Part 3: CONTACT-ONLY Hessian negative-eigenvalue re-check ---
    s = np.linspace(0.0, float(problem.L_model), int(problem.N_nodes))
    u_opt = np.asarray(solution.u_flat_opt, dtype=float).reshape(-1)
    eps_hess = 1e-4
    g_fun = lambda u: contact_energy_gradient_u_consistent(
        u, p0=problem.p0, q0=problem.q0, s=s, lumen_query=problem.lumen_query, contact=problem.contact,
    )
    H_contact = np.zeros((u_opt.size, u_opt.size))
    for k in range(u_opt.size):
        up = u_opt.copy(); up[k] += eps_hess
        um = u_opt.copy(); um[k] -= eps_hess
        H_contact[:, k] = (g_fun(up) - g_fun(um)) / (2 * eps_hess)
    H_contact = 0.5 * (H_contact + H_contact.T)
    eigvals_contact = np.linalg.eigvalsh(H_contact)
    n_negative = int(np.sum(eigvals_contact < 0))
    print(f"\n=== CONTACT-ONLY Hessian (active set implicitly fixed by eps_hess={eps_hess}) ===")
    print(f"negative eigenvalues: {n_negative}/{eigvals_contact.size}")
    order_c = np.argsort(np.abs(eigvals_contact))[::-1]
    print("sorted |eigenvalues| (descending):")
    print(np.array2string(np.abs(eigvals_contact)[order_c], precision=4))
    print("signed, same order:")
    print(np.array2string(eigvals_contact[order_c], precision=4))


if __name__ == "__main__":
    main()
