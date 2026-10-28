"""Contact vs no-contact POSITION Jacobian comparison at identical
configurations, using the actual commanded steps from the completed
contact-aware offline plan.

For each sampled node i along the solved contact path:
  - J_C, J_NC: joint-space position Jacobians (3x7, tip x/y/z only) at
    state[i], both via the rank-aware ("truncated_svd") stabilized
    inversion -- validated to correctly reconstruct the six robot-pose
    columns everywhere, and to use the stable forward insertion branch
    where the state is contact-adjacent (see the 2026-10-06 node140/150/160
    investigation).
  - sigma_1, dominant output direction (first left singular vector),
    ||J_C - J_NC||_F.
  - The REAL commanded step du = state[i+1] - state[i] from the contact
    plan's own CSV (insertion always advances in this plan -- confirmed
    separately -- so du's insertion component is always the validated
    stable forward direction).
  - Linearized prediction: J_C @ du vs J_NC @ du.
  - Ground truth: actual nonlinear d_tip = tip(state[i]+du) - tip(state[i])
    computed by a full forward solve under EACH model, independent of any
    linearization -- the most direct, assumption-free comparison of what
    each model says would actually happen under the real candidate command.

Usage: python compare_contact_nocontact_position_jacobian.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE

PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"
SAMPLE_NODES = list(range(0, 303, 30))  # 0,30,...,300


def tip_at(adapter, state):
    out = np.asarray(adapter.forward_output(state, commit=False), dtype=float)
    return out[:3]


def main():
    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate", plant_contact=True,
        )
    )
    adapter_c = build_diagnostic_adapter(
        beam_model=bundle.models["contact"], controller_pack=controller_pack,
        jacobian_mode="accurate", hessian_inversion="truncated_svd",
    )
    adapter_nc = build_diagnostic_adapter(
        beam_model=bundle.models["no_contact"], controller_pack=controller_pack,
        jacobian_mode="accurate", hessian_inversion="truncated_svd",
    )

    df = pd.read_csv(f"{PLAN_ROOT}/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv")
    states = df[["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad", "insertion_m"]].to_numpy()
    s_mm = df["s_m"].to_numpy() * 1e3

    records = []
    for idx in SAMPLE_NODES:
        if idx + 1 >= len(states):
            continue
        state_i = states[idx]
        du = states[idx + 1] - state_i

        Jc = np.asarray(adapter_c.continuous_output_jacobian(state_i), dtype=float).reshape(6, 7)[:3]
        Jnc = np.asarray(adapter_nc.continuous_output_jacobian(state_i), dtype=float).reshape(6, 7)[:3]
        info_c = dict(bundle.models["contact"].last_sens_info or {})

        Uc, Sc, _ = np.linalg.svd(Jc)
        Unc, Snc, _ = np.linalg.svd(Jnc)
        cos_dom = float(np.clip(abs(np.dot(Uc[:, 0], Unc[:, 0])), -1.0, 1.0))
        angle_dom = float(np.degrees(np.arccos(cos_dom)))

        fro_diff = float(np.linalg.norm(Jc - Jnc))

        pred_c_lin = Jc @ du
        pred_nc_lin = Jnc @ du

        tip0_c = tip_at(adapter_c, state_i)
        tip0_nc = tip_at(adapter_nc, state_i)
        tip1_c = tip_at(adapter_c, state_i + du)
        tip1_nc = tip_at(adapter_nc, state_i + du)
        d_tip_c_true = tip1_c - tip0_c
        d_tip_nc_true = tip1_nc - tip0_nc

        lin_error_c = float(np.linalg.norm(pred_c_lin - d_tip_c_true))
        lin_error_nc = float(np.linalg.norm(pred_nc_lin - d_tip_nc_true))
        true_response_diff = float(np.linalg.norm(d_tip_c_true - d_tip_nc_true))

        rec = {
            "node": idx, "s_mm": s_mm[idx],
            "sigma1_C": float(Sc[0]), "sigma1_NC": float(Snc[0]),
            "dominant_dir_angle_deg": angle_dom,
            "fro_diff_JC_JNC": fro_diff,
            "du_norm": float(np.linalg.norm(du)), "du_L_mm": float(du[6] * 1e3),
            "lin_pred_C_norm_mm": float(np.linalg.norm(pred_c_lin) * 1e3),
            "lin_pred_NC_norm_mm": float(np.linalg.norm(pred_nc_lin) * 1e3),
            "true_dtip_C_norm_mm": float(np.linalg.norm(d_tip_c_true) * 1e3),
            "true_dtip_NC_norm_mm": float(np.linalg.norm(d_tip_nc_true) * 1e3),
            "lin_error_C_mm": lin_error_c * 1e3,
            "lin_error_NC_mm": lin_error_nc * 1e3,
            "true_response_diff_C_vs_NC_mm": true_response_diff * 1e3,
            "effective_rank_C": info_c.get("effective_rank"),
            "n_total_C": info_c.get("n_total"),
        }
        records.append(rec)
        print(f"node={idx:3d} s={s_mm[idx]:6.2f}mm  "
              f"sigma1: C={rec['sigma1_C']:.3f} NC={rec['sigma1_NC']:.3f}  "
              f"dom_angle={angle_dom:5.1f}deg  fro_diff={fro_diff:.3f}  "
              f"du_L={rec['du_L_mm']:+.3f}mm  "
              f"true_dtip: C={rec['true_dtip_C_norm_mm']:.4f}mm NC={rec['true_dtip_NC_norm_mm']:.4f}mm  "
              f"true_C_vs_NC_diff={true_response_diff*1e3:.4f}mm  "
              f"lin_err: C={lin_error_c*1e3:.4f}mm NC={lin_error_nc*1e3:.4f}mm  "
              f"rank_C={rec['effective_rank_C']}/{rec['n_total_C']}", flush=True)

    out_df = pd.DataFrame.from_records(records)
    out_csv = "plans/stage3_design/contact_vs_nocontact_position_jacobian_comparison.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    print("\n=== SUMMARY ===")
    print(out_df[["node", "s_mm", "sigma1_C", "sigma1_NC", "dominant_dir_angle_deg",
                   "true_dtip_C_norm_mm", "true_dtip_NC_norm_mm",
                   "true_response_diff_C_vs_NC_mm", "lin_error_C_mm", "lin_error_NC_mm"]].to_string(index=False))


if __name__ == "__main__":
    main()
