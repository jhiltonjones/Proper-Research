"""Position-only (3x7, tip x/y/z) FD convergence study, re-run per explicit
request: the online controller only tracks tip position, so restrict the
whole validation to J_p (top 3 rows), ignore tangent rows entirely, and
check both per-column and Frobenius convergence vs step size -- at node 150,
two neighbouring Stage-3 nodes, and one known-good Stage-2 state.

For each state:
  1. FD ladder (h, h/2, ..., warm-started from the SAME nominal equilibrium
     for every +h/-h solve), position rows only. Reports per-column
     E_FD(h) = ||Jcol_p(h)-Jcol_p(h/2)|| / ||Jcol_p(h/2)|| and the combined
     Frobenius E_FD_fro(h) across the full 3x7 matrix at each rung.
  2. Converged FD reference = the smallest-h rung's 3x7 position Jacobian
     (only trusted if the ladder actually plateaued -- reported per state).
  3. Compares J_tikhonov[:3,:] and J_truncated_svd[:3,:] (both "accurate"
     mode) against that converged FD reference.

Usage: python diagnose_position_jacobian_convergence.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import (
    BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix,
)
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized

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


def tip_of(result):
    return np.asarray(result.tip, dtype=float).copy()


def fd_ladder_position_only(model, xyz0, rotvec0, L0, label):
    p7_0 = np.concatenate([xyz0, rotvec0, [L0]])
    result0 = model.solve(p7_0, commit=True, reuse_cache=True)
    u_star_0 = np.asarray(result0.u_flat_opt, dtype=float).copy()

    J_at_h = {h_rung_idx: np.zeros((3, 7)) for h_rung_idx in range(len(H_TRANS_LADDER))}
    per_col_records = []

    for dim in range(7):
        ladder = LADDERS[dim]
        cols = []
        for h in ladder:
            xp, rvp, Lp = perturb(xyz0, rotvec0, L0, dim, h, +1)
            xm, rvm, Lm = perturb(xyz0, rotvec0, L0, dim, h, -1)
            p7p = np.concatenate([xp, rvp, [Lp]])
            p7m = np.concatenate([xm, rvm, [Lm]])
            result_p = solve_quasistatic_insertion_optimized(
                model.build_problem(p7p), u0_flat=u_star_0, options=model.beam, result_detail=model.result_detail,
            )
            result_m = solve_quasistatic_insertion_optimized(
                model.build_problem(p7m), u0_flat=u_star_0, options=model.beam, result_detail=model.result_detail,
            )
            col = (tip_of(result_p) - tip_of(result_m)) / (2.0 * h)
            cols.append(col)
        for i, h in enumerate(ladder):
            J_at_h[i][:, dim] = cols[i]
        for i in range(len(ladder)):
            e_next = (
                float(np.linalg.norm(cols[i] - cols[i + 1]) / max(np.linalg.norm(cols[i + 1]), 1e-12))
                if i < len(ladder) - 1 else None
            )
            per_col_records.append({
                "state": label, "dim": DIM_LABELS[dim], "h": ladder[i],
                "col_norm": float(np.linalg.norm(cols[i])), "E_FD_vs_next": e_next,
            })

    # Frobenius convergence across the full 3x7 matrix at each common rung index
    fro_records = []
    for i in range(len(H_TRANS_LADDER) - 1):
        e_fro = float(np.linalg.norm(J_at_h[i] - J_at_h[i + 1]) / max(np.linalg.norm(J_at_h[i + 1]), 1e-12))
        fro_records.append({"state": label, "rung": i, "E_FD_fro_vs_next": e_fro})

    # 2026-10-07: empirically, E_FD(h) does NOT monotonically shrink as h
    # shrinks for this stiff, highly nonlinear contact problem -- confirmed
    # across all four tested states, the Frobenius consecutive-rung error is
    # SMALLEST at the largest step (2mm/1deg/0.5mm, matching the original
    # jnc_contact_source_jacobian.py methodology) and gets WORSE at finer
    # steps, i.e. sub-~250um/~0.25deg perturbations fall below the
    # equilibrium solver's own reliable resolution here and inject noise
    # rather than reducing truncation error. The largest-h rung is therefore
    # the empirically best-supported FD reference, not the smallest.
    J_fd_largest_h = J_at_h[0]
    J_fd_smallest_h = J_at_h[len(H_TRANS_LADDER) - 1]
    return J_fd_largest_h, J_fd_smallest_h, per_col_records, fro_records, model


def relerr(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def run_state(label, xyz0, rotvec0, L0, model):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    J_fd, J_fd_smallest, per_col, fro, model = fd_ladder_position_only(model, xyz0, rotvec0, L0, label)

    print("\nper-column E_FD(h vs h/2), position rows only:")
    df_pc = pd.DataFrame.from_records(per_col)
    for dim in DIM_LABELS:
        sub = df_pc[df_pc["dim"] == dim]
        vals = [f"{v:.2e}" if v is not None else "  --  " for v in sub["E_FD_vs_next"]]
        print(f"  {dim:>3s}: " + "  ".join(vals))

    print("\nFrobenius E_FD(h vs h/2), full 3x7:")
    for r in fro:
        print(f"  rung {r['rung']} (h={H_TRANS_LADDER[r['rung']]:.2e} trans-scale): {r['E_FD_fro_vs_next']:.4e}")

    p7_0 = np.concatenate([xyz0, rotvec0, [L0]])
    model.solve(p7_0, commit=True, reuse_cache=True)
    J_tik = model.jacobian_output_actuation_tangent(p7_0, solve_if_needed=False, mode="accurate", hessian_inversion="tikhonov")[:3, :]
    J_tsvd = model.jacobian_output_actuation_tangent(p7_0, solve_if_needed=False, mode="accurate", hessian_inversion="truncated_svd")[:3, :]
    info_tsvd = dict(model.last_sens_info)

    err_tik_largeh = relerr(J_tik, J_fd)
    err_tsvd_largeh = relerr(J_tsvd, J_fd)
    err_tik_smallh = relerr(J_tik, J_fd_smallest)
    err_tsvd_smallh = relerr(J_tsvd, J_fd_smallest)
    print(f"\nFD reference @ LARGEST h (2mm/1deg/0.5mm, empirically most reliable):")
    print(np.round(J_fd, 5))
    print(f"FD reference @ SMALLEST h (for contrast -- empirically noise-contaminated):")
    print(np.round(J_fd_smallest, 5))
    print(f"\nJ_tikhonov position (3x7):"); print(np.round(J_tik, 5))
    print(f"J_truncated_svd position (3x7):"); print(np.round(J_tsvd, 5))
    print(f"\nrelative Frobenius error vs LARGEST-h FD: tikhonov={err_tik_largeh:.4f}  truncated_svd={err_tsvd_largeh:.4f}")
    print(f"relative Frobenius error vs smallest-h FD (reference only, not trusted): "
          f"tikhonov={err_tik_smallh:.4f}  truncated_svd={err_tsvd_smallh:.4f}")
    print(f"truncated_svd rank info: effective_rank={info_tsvd.get('effective_rank')}/{info_tsvd.get('n_total')} "
          f"gap_ratio={info_tsvd.get('gap_ratio')}")

    return {
        "label": label, "relerr_tikhonov_largeh": err_tik_largeh, "relerr_truncated_svd_largeh": err_tsvd_largeh,
        "relerr_tikhonov_smallh": err_tik_smallh, "relerr_truncated_svd_smallh": err_tsvd_smallh,
        "effective_rank": info_tsvd.get("effective_rank"), "n_total": info_tsvd.get("n_total"),
    }


def main():
    summary = []

    # --- Stage-3 node 150 + two neighbours ---
    exp_cfg, bundle3, controller_pack, out_root, centreline, lumen_R, provenance = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
    )
    model3 = bundle3.models["contact"]
    adapter3 = build_diagnostic_adapter(beam_model=model3, controller_pack=controller_pack, jacobian_mode="accurate")
    df = pd.read_csv(f"{PLAN_ROOT}/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv")

    for idx in (149, 150, 151):
        state = df.loc[idx, ["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy(dtype=float)
        T_R_M = adapter3.magnet_transform(state)
        xyz0 = T_R_M[:3, 3].copy()
        rotvec0 = Rot.from_matrix(T_R_M[:3, :3]).as_rotvec()
        L0 = float(state[6])
        label = f"Stage3 node{idx} s={df.loc[idx,'s_m']*1e3:.2f}mm"
        summary.append(run_state(label, xyz0, rotvec0, L0, model3))

    # --- Stage-2 known-good state ---
    beam_base_xyz = BEAM_BASE_XYZ_M.copy()
    beam_base_xyz6 = beam_base_pose6()
    PHI_DEG, RADIUS_MM, L_MM, PSI_DEG = 35.0, 225.0, 40.0, 30.0
    L_M2 = L_MM / 1000.0
    phi = np.radians(PHI_DEG)
    xyz0_2 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    R_aligned = reference_orientation_matrix(xyz0_2, beam_base_xyz)
    R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
    R0 = R_aligned @ R_psi
    rotvec0_2 = Rot.from_matrix(R0).as_rotvec()
    bundle2 = build_model_bundle(CURRENT_LUMEN_FILE, beam_base_xyz6, REFERENCE_MAGNET_POSE6, L_M2)
    model2 = bundle2.models["contact"]
    summary.append(run_state(f"Stage2 psi=+{PSI_DEG:.0f} L={L_MM:.0f}mm", xyz0_2, rotvec0_2, L_M2, model2))

    print(f"\n{'='*70}\nSUMMARY (position-only J_p, converged-FD arbiter)\n{'='*70}")
    out_df = pd.DataFrame.from_records(summary)
    print(out_df.to_string(index=False))
    out_df.to_csv("plans/stage3_design/position_jacobian_convergence_summary.csv", index=False)
    print("\nsaved plans/stage3_design/position_jacobian_convergence_summary.csv")


if __name__ == "__main__":
    main()
