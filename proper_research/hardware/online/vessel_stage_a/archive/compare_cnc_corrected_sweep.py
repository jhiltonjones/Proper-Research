"""Corrected, branch-consistent contact-vs-no-contact position-Jacobian
sweep at 5 representative locations, using the REAL one-control-tick
command from the time-parameterized trajectory.

Fixes applied relative to the earlier (invalidated) comparison:
  - build_vessel_planning_context is given the real plan's initial_poses
    (not the stale default beam-base pivot).
  - BOTH the contact and no-contact models are walked with sequential
    continuation (u0_flat=u_prev at every sample from k=0), not isolated
    solves -- each model gets its own independent warm-start chain over
    the identical state_reference trajectory.
  - The contact chain's tip is cross-checked against the plan's own
    recorded achieved_position_m at every target before trusting it.

At each of the 5 locations, reports:
  - J_p,C, J_p,NC (3x7, position-only, stabilized "truncated_svd")
  - singular values + dominant output direction for each, and the angle
    between the two models' dominant directions
  - ||J_p,C - J_p,NC||
  - linear prediction d_p_C = J_p,C @ d_chi vs the TRUE nonlinear contact
    response over the same real one-tick command (own-model accuracy)
  - linear prediction d_p_NC = J_p,NC @ d_chi vs the TRUE nonlinear
    no-contact response to the SAME command (own-model accuracy)
  - d_p_C_true vs d_p_NC_true directly: what contact actually does vs what
    no-contact would have predicted for the identical command

Usage: python compare_cnc_corrected_sweep.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE, beam_base_pose6
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs

PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"
REGIONS = [
    ("pre-contact", 2.0),
    ("contact entry", 17.5),
    ("s~37.5mm", 37.5),
    ("s~60mm", 60.0),
    ("near end", 73.0),
]


def walk_continuation(model, adapter, state_ref, n):
    """Sequential continuation: every sample warm-started from the previous
    one's own converged u*. Returns per-sample tip (true nonlinear output)."""
    u_prev = None
    tips = np.zeros((n, 3))
    u_stars = [None] * n
    for k in range(n):
        chi = state_ref[k]
        T_R_M = adapter.magnet_transform(chi)
        p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
        problem = model.build_problem(p7)
        result = solve_quasistatic_insertion_optimized(
            problem, u0_flat=u_prev, options=model.beam, result_detail=model.result_detail,
        )
        u_prev = np.asarray(result.u_flat_opt, dtype=float).copy()
        u_stars[k] = u_prev
        tips[k] = np.asarray(result.tip, dtype=float)
    return tips, u_stars


def jacobian_at(model, adapter, chi, u_star_prev):
    T_R_M = adapter.magnet_transform(chi)
    p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
    problem = model.build_problem(p7)
    result = solve_quasistatic_insertion_optimized(
        problem, u0_flat=u_star_prev, options=model.beam, result_detail=model.result_detail,
    )
    model._commit_result(p7, result, problem.L_model)
    model._invalidate_jacobian_values()
    J6 = model.jacobian_output_actuation_tangent(
        p7, solve_if_needed=False, mode="accurate", hessian_inversion="truncated_svd",
    )
    info = dict(model.last_sens_info or {})
    return np.asarray(J6, dtype=float).reshape(6, 7)[:3], info


def main():
    design = json.loads(open(f"plans/stage3_design/phi30_L30_left1mm_newwall_design.json").read())
    start_point = np.asarray(design["magnet_pose6_R"], dtype=float)
    initial_poses = (beam_base_pose6(), start_point.copy(), 0.030, 0.01)

    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
            initial_poses=initial_poses,
        )
    )
    model_c = bundle.models["contact"]
    model_nc = bundle.models["no_contact"]
    adapter_c = build_diagnostic_adapter(beam_model=model_c, controller_pack=controller_pack, jacobian_mode="accurate")
    adapter_nc = build_diagnostic_adapter(beam_model=model_nc, controller_pack=controller_pack, jacobian_mode="accurate")

    d = np.load(f"{PLAN_ROOT}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    s_mm = d["path_s_m"] * 1e3
    state_ref = d["state_reference"]
    achieved = d["achieved_position_m"]

    target_k = {label: int(np.argmin(np.abs(s_mm - s_target))) for label, s_target in REGIONS}
    n_needed = max(target_k.values()) + 4

    print(f"walking {n_needed} samples, CONTACT model...", flush=True)
    tips_c, u_c = walk_continuation(model_c, adapter_c, state_ref, n_needed)
    print(f"walking {n_needed} samples, NO-CONTACT model...", flush=True)
    tips_nc, u_nc = walk_continuation(model_nc, adapter_nc, state_ref, n_needed)

    print("\ncross-check CONTACT walk against recorded achieved_position_m:")
    for label, k0 in target_k.items():
        drift = np.linalg.norm(tips_c[k0] - achieved[k0]) * 1e3
        print(f"  {label:15s} k={k0:4d} s={s_mm[k0]:6.2f}mm  drift={drift:.4f}mm "
              f"{'OK' if drift < 0.5 else 'WARNING'}")

    records = []
    print(f"\n{'region':15s} {'s_mm':>7s} {'sig1_C':>7s} {'sig1_NC':>8s} {'dom_angle':>10s} "
          f"{'||Jc-Jnc||':>11s} {'e_C_mm':>8s} {'e_NC_mm':>8s} {'dpC_true':>9s} {'dpNC_true':>10s} {'C_vs_NC':>8s}")
    for label, k0 in target_k.items():
        chi = state_ref[k0]
        d_chi = state_ref[k0 + 1] - state_ref[k0]

        Jc, info_c = jacobian_at(model_c, adapter_c, chi, u_c[max(k0 - 1, 0)])
        Jnc, info_nc = jacobian_at(model_nc, adapter_nc, chi, u_nc[max(k0 - 1, 0)])

        Uc, Sc, _ = np.linalg.svd(Jc)
        Unc, Snc, _ = np.linalg.svd(Jnc)
        cos_dom = float(np.clip(abs(np.dot(Uc[:, 0], Unc[:, 0])), -1.0, 1.0))
        angle_dom = float(np.degrees(np.arccos(cos_dom)))
        fro_diff = float(np.linalg.norm(Jc - Jnc))

        dpC_pred = Jc @ d_chi
        dpNC_pred = Jnc @ d_chi
        dpC_true = tips_c[k0 + 1] - tips_c[k0]
        dpNC_true = tips_nc[k0 + 1] - tips_nc[k0]

        e_C = float(np.linalg.norm(dpC_true - dpC_pred))
        e_NC = float(np.linalg.norm(dpNC_true - dpNC_pred))
        c_vs_nc_true = float(np.linalg.norm(dpC_true - dpNC_true))

        rec = {
            "region": label, "s_mm": float(s_mm[k0]),
            "sigma1_C": float(Sc[0]), "sigma2_C": float(Sc[1]), "sigma3_C": float(Sc[2]),
            "sigma1_NC": float(Snc[0]), "sigma2_NC": float(Snc[1]), "sigma3_NC": float(Snc[2]),
            "dominant_dir_angle_deg": angle_dom, "fro_diff_JC_JNC": fro_diff,
            "d_chi_L_mm": float(d_chi[6] * 1e3), "d_chi_q_norm_rad": float(np.linalg.norm(d_chi[:6])),
            "dpC_pred_mm": (dpC_pred * 1e3).tolist(), "dpC_true_mm": (dpC_true * 1e3).tolist(),
            "dpNC_pred_mm": (dpNC_pred * 1e3).tolist(), "dpNC_true_mm": (dpNC_true * 1e3).tolist(),
            "e_C_mm": e_C * 1e3, "e_NC_mm": e_NC * 1e3,
            "dpC_true_norm_mm": float(np.linalg.norm(dpC_true) * 1e3),
            "dpNC_true_norm_mm": float(np.linalg.norm(dpNC_true) * 1e3),
            "C_vs_NC_true_diff_mm": c_vs_nc_true * 1e3,
            "rank_C": info_c.get("effective_rank"), "n_total_C": info_c.get("n_total"),
        }
        records.append(rec)
        print(f"{label:15s} {rec['s_mm']:7.2f} {rec['sigma1_C']:7.3f} {rec['sigma1_NC']:8.3f} "
              f"{angle_dom:10.1f} {fro_diff:11.4f} {e_C*1e3:8.4f} {e_NC*1e3:8.4f} "
              f"{rec['dpC_true_norm_mm']:9.4f} {rec['dpNC_true_norm_mm']:10.4f} {c_vs_nc_true*1e3:8.4f}")

    out_df = pd.DataFrame.from_records(records)
    out_csv = "plans/stage3_design/cnc_corrected_sweep.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    # --- Near-end tick deep dive ---
    print("\n=== near-end tick deep dive (component-level) ===")
    k0 = target_k["near end"]
    for dk in range(3):
        k = k0 + dk
        chi = state_ref[k]
        d_chi = state_ref[k + 1] - state_ref[k]
        Jc, info_c = jacobian_at(model_c, adapter_c, chi, u_c[max(k - 1, 0)])
        dpC_pred = Jc @ d_chi
        dpC_true = tips_c[k + 1] - tips_c[k]
        e_vec = dpC_true - dpC_pred
        gain_true = np.linalg.norm(dpC_true)
        gain_pred = np.linalg.norm(dpC_pred)
        cos_dir = float(np.clip(np.dot(dpC_true, dpC_pred) / max(gain_true * gain_pred, 1e-12), -1, 1))
        angle = float(np.degrees(np.arccos(cos_dir)))
        print(f"\nk={k} s={s_mm[k]:.3f}mm")
        print(f"  d_chi (q1..q6,L) = {np.round(d_chi[:6],6)}  dL_mm={d_chi[6]*1e3:.5f}")
        print(f"  dominant d_chi component: {'q'+str(int(np.argmax(np.abs(d_chi[:6])))+1) if np.max(np.abs(d_chi[:6]))*1 > abs(d_chi[6]) else 'L'}")
        print(f"  dpC_true (mm) = {np.round(dpC_true*1e3,5)}  |.|={gain_true*1e3:.4f}mm")
        print(f"  dpC_pred (mm) = {np.round(dpC_pred*1e3,5)}  |.|={gain_pred*1e3:.4f}mm")
        print(f"  gain ratio (true/pred) = {gain_true/max(gain_pred,1e-12):.3f}   direction angle = {angle:.1f}deg")
        print(f"  error vector (mm) = {np.round(e_vec*1e3,5)}  |e|={np.linalg.norm(e_vec)*1e3:.4f}mm")


if __name__ == "__main__":
    main()
