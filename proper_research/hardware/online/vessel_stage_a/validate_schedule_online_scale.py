"""Final, decisive arbiter: does the saved schedule's raw-matrix disagreement
(15-41% relative error vs a branch-consistent FD-validated Jacobian, found by
validate_mpc_schedule_jacobians.py) actually matter at the scale the MPC uses
it at?

diagnose_schedule_baseline_mismatch.py already showed the forward (tip)
solution is branch-robust at the worst-offending state (0.0086mm apart
despite the raw u* differing), so the 15-41% figure is Hessian-conditioning
sensitivity, not a wrong-equilibrium-branch bug. The established Stage-3
"online-scale" methodology (not the offline node-to-node jump) is the
correct final arbiter: take the REAL one-control-tick command
Delta_chi = state_reference[idx+1] - state_reference[idx] and compare
J_sched[idx] @ Delta_chi against the TRUE nonlinear tip change
p(chi+Delta_chi) - p(chi), both evaluated on the SAME branch-consistent
sequential-continuation walk.

Usage: python validate_schedule_online_scale.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE

from proper_research.hardware.online.vessel_stage_a.compare_cnc_corrected_sweep import walk_continuation
from proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel import (
    _recalibrated_make_initial_poses,
)

OUT_DIR = "plans/stage3_design/openloop_hw_analysis"
REGIONS = [
    ("pre-contact", 2.0), ("contact entry", 17.5), ("s~37.5mm", 37.5),
    ("s~60mm", 60.0), ("near end", 73.0),
]
PLANS = {
    "CONTACT": dict(
        plan_root="plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06",
        schedule_path="/tmp/vessel_c_schedule.npy", model_key="contact",
    ),
    "NO-CONTACT": dict(
        plan_root="plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06",
        schedule_path="/tmp/vessel_nc_schedule.npy", model_key="no_contact",
    ),
}


def main():
    initial_poses = _recalibrated_make_initial_poses()
    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate",
        plant_contact=True, initial_poses=initial_poses,
    )

    rows = []
    for plan_name, cfg in PLANS.items():
        model = bundle.models[cfg["model_key"]]
        adapter = build_diagnostic_adapter(beam_model=model, controller_pack=controller_pack, jacobian_mode="accurate")

        npz = np.load(f"{cfg['plan_root']}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
        state_ref = npz["state_reference"]
        path_s_mm = npz["path_s_m"] * 1e3
        schedule = np.load(cfg["schedule_path"])

        target_k = {label: int(np.argmin(np.abs(path_s_mm - s_target))) for label, s_target in REGIONS}
        n_needed = max(target_k.values()) + 2
        print(f"\n[{plan_name}] walking {n_needed} samples via sequential continuation...", flush=True)
        tips, u_stars = walk_continuation(model, adapter, state_ref, n_needed)

        for label, idx in target_k.items():
            d_chi = state_ref[idx + 1] - state_ref[idx]
            dp_true = tips[idx + 1] - tips[idx]
            dp_pred = schedule[idx] @ d_chi
            e_mm = float(np.linalg.norm(dp_true - dp_pred)) * 1e3
            gain_true_mm = float(np.linalg.norm(dp_true)) * 1e3
            gain_pred_mm = float(np.linalg.norm(dp_pred)) * 1e3
            cos_dir = float(np.clip(
                np.dot(dp_true, dp_pred) / max(gain_true_mm * gain_pred_mm * 1e-6, 1e-15), -1, 1
            ))
            angle_deg = float(np.degrees(np.arccos(cos_dir)))
            print(f"  {label:15s} s={path_s_mm[idx]:6.2f}mm  e_onetick={e_mm:.4f}mm  "
                  f"|dp_true|={gain_true_mm:.4f}mm |dp_pred|={gain_pred_mm:.4f}mm  angle={angle_deg:.1f}deg")
            rows.append({
                "plan": plan_name, "region": label, "s_mm": float(path_s_mm[idx]),
                "e_onetick_mm": e_mm, "dp_true_norm_mm": gain_true_mm, "dp_pred_norm_mm": gain_pred_mm,
                "direction_angle_deg": angle_deg,
            })

    df = pd.DataFrame.from_records(rows)
    df.to_csv(f"{OUT_DIR}/schedule_online_scale_validation.csv", index=False)
    print(f"\n{'='*70}\nSUMMARY (one-tick prediction error, the deployment-relevant test)\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\nmax e_onetick_mm = {df['e_onetick_mm'].max():.4f}mm across all {len(df)} tested locations")
    print(f"saved {OUT_DIR}/schedule_online_scale_validation.csv")


if __name__ == "__main__":
    main()
