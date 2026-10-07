"""Corrected online-scale Jacobian validation: walks the time-parameterized
trajectory SEQUENTIALLY from the start, warm-starting each sample's
equilibrium from the immediately preceding one (u0_flat=u_prev) -- exactly
how the real system tracks the path, and exactly what the original
time-parameterization validation did. This matters: an isolated
forward-solve at a single far-along state (no continuation) was confirmed
to land on a DIFFERENT local equilibrium branch than the one the real
trajectory follows (verified against the plan's own recorded
achieved_position_m -- off by tens of mm), which silently invalidated a
naive version of this same test. Every point below is cross-checked
against that recorded achieved_position_m before being trusted.

At each requested region, once reached by the trusted continuation chain,
computes the one-control-tick online-scale test:

    e_one_step = || p_C(chi+d_chi) - p_C(chi) - J_p @ d_chi ||

with d_chi = state_reference[k+1] - state_reference[k] (the real MPC tick,
dt=0.1s), p_C(.) from the SAME trusted continuation (not an isolated
re-solve), and J_p the position-only (3x7) Jacobian, stabilized
(truncated_svd), evaluated at the trusted equilibrium.

Usage: python validate_online_scale_jacobian_v2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE, beam_base_pose6
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs
import json

PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"

REGIONS = [
    ("before contact", 2.0),
    ("entry into contacted region", 17.5),
    ("node 140/150/160 region", 37.5),
    ("highly nonlinear region (node 240)", 60.0),
    ("near end", 73.0),
]
TICKS_PER_REGION = 3


def main():
    d = np.load(f"{PLAN_ROOT}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    s_mm = d["path_s_m"] * 1e3
    state_ref = d["state_reference"]
    achieved = d["achieved_position_m"]
    n = state_ref.shape[0]

    # 2026-10-07: build_vessel_planning_context MUST be given the real plan's
    # initial_poses (beam-base pivot + start point), or it silently falls
    # back to initial_conditions.make_initial_poses()'s stale beam-base Z --
    # confirmed live: without this, a fresh re-solve at node 0 landed tens of
    # mm away from the plan's own recorded tip (misaligned contact wall
    # geometry), while with it, node 0/150/240/300 all reproduce the
    # recorded tip to within ~0.004-0.4mm.
    design = json.loads(open("plans/stage3_design/phi30_L30_left1mm_newwall_design.json").read())
    start_point = np.asarray(design["magnet_pose6_R"], dtype=float)
    initial_poses = (beam_base_pose6(), start_point.copy(), 0.030, 0.01)
    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
            initial_poses=initial_poses,
        )
    )
    model = bundle.models["contact"]
    adapter = build_diagnostic_adapter(
        beam_model=model, controller_pack=controller_pack,
        jacobian_mode="accurate", hessian_inversion="truncated_svd",
    )

    target_k = {label: int(np.argmin(np.abs(s_mm - s_target))) for label, s_target in REGIONS}
    max_k_needed = max(target_k.values()) + TICKS_PER_REGION + 1
    print(f"walking {max_k_needed} sequential samples with continuation warm-start...", flush=True)

    # Sequential continuation walk: solve every sample from 0 up to the
    # deepest requested region, each warm-started from the previous one.
    u_prev = None
    tips_true = np.zeros((max_k_needed, 3))
    u_stars = [None] * max_k_needed
    for k in range(max_k_needed):
        chi = state_ref[k]
        T_R_M = adapter.magnet_transform(chi)
        p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
        problem = model.build_problem(p7)
        result = solve_quasistatic_insertion_optimized(
            problem, u0_flat=u_prev, options=model.beam, result_detail=model.result_detail,
        )
        u_prev = np.asarray(result.u_flat_opt, dtype=float).copy()
        u_stars[k] = u_prev
        tips_true[k] = np.asarray(result.tip, dtype=float)
        if k % 100 == 0:
            drift = np.linalg.norm(tips_true[k] - achieved[k]) * 1e3
            print(f"  k={k}  s={s_mm[k]:.2f}mm  drift_vs_recorded={drift:.4f}mm", flush=True)

    print("\ncontinuation-walk cross-check against recorded achieved_position_m:")
    max_drift = 0.0
    for label, k0 in target_k.items():
        drift = np.linalg.norm(tips_true[k0] - achieved[k0]) * 1e3
        max_drift = max(max_drift, drift)
        print(f"  {label:35s} k={k0:4d}  s={s_mm[k0]:6.2f}mm  "
              f"my_tip={np.round(tips_true[k0],5)}  recorded={np.round(achieved[k0],5)}  "
              f"drift={drift:.4f}mm")
    print(f"  max drift across all region anchors: {max_drift:.4f}mm "
          f"({'TRUSTED -- branch-consistent' if max_drift < 0.5 else 'WARNING: still off-branch, do not trust'})")

    print(f"\n{'region':35s} {'s_mm':>7s} {'|d_chi_L|_mm':>12s} {'|d_chi_q|_rad':>12s} "
          f"{'|true_dtip|_mm':>14s} {'e_one_step_mm':>14s} {'e_rel':>8s}")
    records = []
    for label, k0 in target_k.items():
        for tick in range(TICKS_PER_REGION):
            k = min(k0 + tick, max_k_needed - 2)
            chi = state_ref[k]
            d_chi = state_ref[k + 1] - state_ref[k]

            # Jacobian anchored at the TRUSTED equilibrium: seed the model's
            # own cache with u_stars[k] before asking for the Jacobian, so
            # implicit_tip_jacobian differentiates around the same branch
            # the continuation walk actually found (not whatever the
            # adapter's own internal warm-start heuristic would pick).
            T_R_M = adapter.magnet_transform(chi)
            p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
            problem = model.build_problem(p7)
            result_k = solve_quasistatic_insertion_optimized(
                problem, u0_flat=u_stars[max(k - 1, 0)], options=model.beam, result_detail=model.result_detail,
            )
            model._commit_result(p7, result_k, problem.L_model)
            model._invalidate_jacobian_values()
            Jp6 = model.jacobian_output_actuation_tangent(
                p7, solve_if_needed=False, mode="accurate", hessian_inversion="truncated_svd",
            )
            info = dict(model.last_sens_info or {})
            Jp = np.asarray(Jp6, dtype=float).reshape(6, 7)[:3]

            tip0 = tips_true[k]
            tip1 = tips_true[k + 1]
            true_dtip = tip1 - tip0
            lin_pred = Jp @ d_chi
            e_one_step = np.linalg.norm(true_dtip - lin_pred)
            rel = e_one_step / max(np.linalg.norm(true_dtip), 1e-9)

            rec = {
                "label": label, "s_mm": float(s_mm[k]), "d_chi_L_mm": float(d_chi[6] * 1e3),
                "d_chi_q_norm_rad": float(np.linalg.norm(d_chi[:6])),
                "true_dtip_mm": float(np.linalg.norm(true_dtip) * 1e3),
                "e_one_step_mm": float(e_one_step * 1e3), "e_rel": float(rel),
                "effective_rank": info.get("effective_rank"), "n_total": info.get("n_total"),
            }
            records.append(rec)
            print(f"{label:35s} {rec['s_mm']:7.2f} {rec['d_chi_L_mm']:12.5f} "
                  f"{rec['d_chi_q_norm_rad']:12.6f} {rec['true_dtip_mm']:14.5f} "
                  f"{rec['e_one_step_mm']:14.6f} {rec['e_rel']*100:7.2f}%  "
                  f"rank={rec['effective_rank']}/{rec['n_total']}")

    out_df = pd.DataFrame.from_records(records)
    out_csv = "plans/stage3_design/online_scale_jacobian_validation_v2.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    print("\n=== SUMMARY BY REGION (mean over ticks) ===")
    print(out_df.groupby("label", sort=False)[["true_dtip_mm", "e_one_step_mm", "e_rel"]].mean().to_string())
    print(f"\noverall max e_one_step_mm = {out_df['e_one_step_mm'].max():.6f}")
    print(f"overall max e_rel = {out_df['e_rel'].max()*100:.2f}%")
    print("(context: this plan's position_tolerance was 0.5mm)")


if __name__ == "__main__":
    main()
