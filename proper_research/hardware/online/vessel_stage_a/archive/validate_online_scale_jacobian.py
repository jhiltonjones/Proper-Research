"""The deployment-relevant validation: does J_p (3x7, position-only,
stabilized) predict the tip's response to a REALISTIC ONE-CONTROL-TICK MPC
step, not the much larger offline node-to-node step?

For each requested region of the path, takes several consecutive real
commanded steps d_chi = state_reference[k+1] - state_reference[k] from the
time-parameterized (dt=0.1s, the actual MPC tick) trajectory -- NOT the
sparser offline inverse-configuration node spacing -- and computes

    e_one_step = || p_C(chi + d_chi) - p_C(chi) - J_p @ d_chi ||

via full forward solves for the true nonlinear tip change (no linearization
anywhere in the "truth" side), compared against the linear prediction from
J_p alone. Tangent rows are dropped entirely -- only the 3x7 position
Jacobian the online MPC actually uses.

Usage: python validate_online_scale_jacobian.py
"""
from __future__ import annotations

import numpy as np

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE

PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"

# Regions requested: (label, target path-s in mm)
REGIONS = [
    ("before contact", 2.0),
    ("entry into contacted region", 17.5),
    ("node 140/150/160 region", 37.5),
    ("highly nonlinear region (node 240)", 60.0),
    ("near end", 73.0),
]
TICKS_PER_REGION = 3  # consecutive one-control-tick steps sampled at/after each anchor


def tip_of(adapter, state):
    return np.asarray(adapter.forward_output(state, commit=False), dtype=float)[:3]


def main():
    d = np.load(f"{PLAN_ROOT}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    s_mm = d["path_s_m"] * 1e3
    state_ref = d["state_reference"]  # (534,7): q1..q6, L
    n = state_ref.shape[0]

    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
        )
    )
    adapter_c = build_diagnostic_adapter(
        beam_model=bundle.models["contact"], controller_pack=controller_pack,
        jacobian_mode="accurate", hessian_inversion="truncated_svd",
    )

    print(f"{'region':35s} {'s_mm':>7s} {'|d_chi_L|_mm':>12s} {'|d_chi|_rad':>11s} "
          f"{'|true_dtip|_mm':>14s} {'e_one_step_mm':>14s} {'e/|true_dtip|':>13s} rank")
    all_records = []
    for label, s_target in REGIONS:
        k0 = int(np.argmin(np.abs(s_mm - s_target)))
        for tick in range(TICKS_PER_REGION):
            k = min(k0 + tick, n - 2)
            chi = state_ref[k].copy()
            d_chi = state_ref[k + 1] - state_ref[k]

            Jp = np.asarray(adapter_c.continuous_output_jacobian(chi), dtype=float).reshape(6, 7)[:3]
            info = dict(bundle.models["contact"].last_sens_info or {})

            tip0 = tip_of(adapter_c, chi)
            tip1 = tip_of(adapter_c, chi + d_chi)
            true_dtip = tip1 - tip0
            lin_pred = Jp @ d_chi
            e_one_step = np.linalg.norm(true_dtip - lin_pred)

            rel = e_one_step / max(np.linalg.norm(true_dtip), 1e-9)
            rec = {
                "label": label, "s_mm": float(s_mm[k]), "d_chi_L_mm": float(d_chi[6] * 1e3),
                "d_chi_joint_norm_rad": float(np.linalg.norm(d_chi[:6])),
                "true_dtip_mm": float(np.linalg.norm(true_dtip) * 1e3),
                "e_one_step_mm": float(e_one_step * 1e3), "e_rel": float(rel),
                "effective_rank": info.get("effective_rank"), "n_total": info.get("n_total"),
            }
            all_records.append(rec)
            print(f"{label:35s} {rec['s_mm']:7.2f} {rec['d_chi_L_mm']:12.5f} "
                  f"{rec['d_chi_joint_norm_rad']:11.6f} {rec['true_dtip_mm']:14.5f} "
                  f"{rec['e_one_step_mm']:14.6f} {rec['e_rel']*100:11.2f}% "
                  f"{rec['effective_rank']}/{rec['n_total']}")

    import pandas as pd
    out_df = pd.DataFrame.from_records(all_records)
    out_csv = "plans/stage3_design/online_scale_jacobian_validation.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    print("\n=== SUMMARY BY REGION (mean over ticks) ===")
    summary = out_df.groupby("label", sort=False)[["true_dtip_mm", "e_one_step_mm", "e_rel"]].mean()
    print(summary.to_string())
    print(f"\noverall max e_one_step_mm = {out_df['e_one_step_mm'].max():.6f}")
    print(f"overall max e_rel = {out_df['e_rel'].max()*100:.2f}%")
    print(f"\n(context: this plan's position_tolerance was 0.5mm)")


if __name__ == "__main__":
    main()
