"""P_C vs P_NC planner comparison: same initial q/L, vessel centreline,
safety constraints, joint/insertion limits, path discretization,
continuation parameters, multistart settings and random seed (both plans
came from identical build_vessel_plan.py invocations differing only in
--no-contact-plant) -- the beam/contact model is the only changed
ingredient, confirmed by the shared 303-node s-grid.

1. Configuration separation ||chi_C - chi_NC||, source-magnet position
   separation relative to the beam base, and the raw q_i(s)/L(s) traces
   for both planners.

2. The 2x2 cross-evaluation. F_C(P_C) and F_NC(P_NC) are read directly
   from each plan's own recorded tip/position_error columns (each
   planner's own self-consistent forward evaluation). F_C(P_NC) and
   F_NC(P_C) require walking the OTHER plan's chosen [q,L] sequence
   through this model with proper sequential continuation (every node
   warm-started from the previous one's own converged state, exactly as
   validated for the online-scale/corrected C-vs-NC work) -- an isolated
   re-solve at a single far-along state was already shown to land on the
   wrong equilibrium branch.

Usage: python compare_PC_vs_PNC.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, beam_base_pose6
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter

PC_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"
PNC_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06"


def walk_model(model, adapter, states):
    """Sequential continuation walk: states is (N,7) [q1..q6,L]. Returns
    (N,3) tip array, each node warm-started from the previous node's own
    converged u*."""
    n = states.shape[0]
    u_prev = None
    tips = np.zeros((n, 3))
    for k in range(n):
        chi = states[k]
        T_R_M = adapter.magnet_transform(chi)
        p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
        problem = model.build_problem(p7)
        result = solve_quasistatic_insertion_optimized(
            problem, u0_flat=u_prev, options=model.beam, result_detail=model.result_detail,
        )
        u_prev = np.asarray(result.u_flat_opt, dtype=float).copy()
        tips[k] = np.asarray(result.tip, dtype=float)
        if k % 50 == 0:
            print(f"    k={k}/{n}", flush=True)
    return tips


def main():
    df_c = pd.read_csv(f"{PC_ROOT}/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv")
    df_nc = pd.read_csv(f"{PNC_ROOT}/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv")
    assert len(df_c) == len(df_nc) and np.allclose(df_c.s_m.to_numpy(), df_nc.s_m.to_numpy()), \
        "s-grids differ -- plans are not directly comparable node-by-node"
    n = len(df_c)
    s_mm = df_c["s_m"].to_numpy() * 1e3

    q_cols = ["q1_rad", "q2_rad", "q3_rad", "q4_rad", "q5_rad", "q6_rad"]
    chi_c = df_c[q_cols + ["insertion_m"]].to_numpy()
    chi_nc = df_nc[q_cols + ["insertion_m"]].to_numpy()

    chi_sep = np.linalg.norm(chi_c - chi_nc, axis=1)
    joint_sep = np.linalg.norm(chi_c[:, :6] - chi_nc[:, :6], axis=1)
    L_sep_mm = np.abs(chi_c[:, 6] - chi_nc[:, 6]) * 1e3

    magnet_c = df_c[["magnet_pose_x", "magnet_pose_y", "magnet_pose_z"]].to_numpy()
    magnet_nc = df_nc[["magnet_pose_x", "magnet_pose_y", "magnet_pose_z"]].to_numpy()
    base = BEAM_BASE_XYZ_M
    magnet_c_rel = magnet_c - base
    magnet_nc_rel = magnet_nc - base
    magnet_sep_mm = np.linalg.norm(magnet_c - magnet_nc, axis=1) * 1e3

    p_ref_c = df_c[["desired_x", "desired_y", "desired_z"]].to_numpy()
    p_ref_nc = df_nc[["desired_x", "desired_y", "desired_z"]].to_numpy()
    assert np.allclose(p_ref_c, p_ref_nc, atol=1e-9), "target paths differ between the two plans"
    p_ref = p_ref_c

    F_C_of_PC = df_c[["tip_x", "tip_y", "tip_z"]].to_numpy()
    F_NC_of_PNC = df_nc[["tip_x", "tip_y", "tip_z"]].to_numpy()
    e_C_to_C = np.linalg.norm(F_C_of_PC - p_ref, axis=1) * 1e3
    e_NC_to_NC = np.linalg.norm(F_NC_of_PNC - p_ref, axis=1) * 1e3

    # --- Build the (correctly-constructed) model bundle for cross-evaluation ---
    design = json.loads(open("plans/stage3_design/phi30_L30_left1mm_newwall_design.json").read())
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

    print(f"walking P_C's own {n} nodes through F_C (self-consistency check)...", flush=True)
    F_C_of_PC_walked = walk_model(model_c, adapter_c, chi_c)
    drift_self_c = np.linalg.norm(F_C_of_PC_walked - F_C_of_PC, axis=1) * 1e3
    print(f"  max drift vs recorded: {drift_self_c.max():.4f}mm")

    print(f"walking P_NC's {n} nodes through F_C (THE key cross-evaluation)...", flush=True)
    F_C_of_PNC = walk_model(model_c, adapter_c, chi_nc)

    print(f"walking P_C's {n} nodes through F_NC...", flush=True)
    F_NC_of_PC = walk_model(model_nc, adapter_nc, chi_c)

    e_NC_to_C = np.linalg.norm(F_C_of_PNC - p_ref, axis=1) * 1e3
    e_C_to_NC = np.linalg.norm(F_NC_of_PC - p_ref, axis=1) * 1e3

    out = pd.DataFrame({
        "s_mm": s_mm,
        "chi_sep": chi_sep, "joint_sep_rad": joint_sep, "L_sep_mm": L_sep_mm,
        "magnet_sep_mm": magnet_sep_mm,
        "magnet_c_rel_x_mm": magnet_c_rel[:, 0] * 1e3, "magnet_c_rel_y_mm": magnet_c_rel[:, 1] * 1e3,
        "magnet_c_rel_z_mm": magnet_c_rel[:, 2] * 1e3,
        "magnet_nc_rel_x_mm": magnet_nc_rel[:, 0] * 1e3, "magnet_nc_rel_y_mm": magnet_nc_rel[:, 1] * 1e3,
        "magnet_nc_rel_z_mm": magnet_nc_rel[:, 2] * 1e3,
        "L_C_mm": chi_c[:, 6] * 1e3, "L_NC_mm": chi_nc[:, 6] * 1e3,
        "e_C_to_C_mm": e_C_to_C, "e_NC_to_NC_mm": e_NC_to_NC,
        "e_NC_to_C_mm": e_NC_to_C, "e_C_to_NC_mm": e_C_to_NC,
        "self_consistency_drift_C_mm": drift_self_c,
    })
    for i, col in enumerate(["q1", "q2", "q3", "q4", "q5", "q6"]):
        out[f"{col}_C_deg"] = np.degrees(chi_c[:, i])
        out[f"{col}_NC_deg"] = np.degrees(chi_nc[:, i])

    out_csv = "plans/stage3_design/PC_vs_PNC_comparison.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")

    print("\n=== SUMMARY ===")
    print(f"max self-consistency drift (F_C(P_C) walked vs recorded): {drift_self_c.max():.4f}mm "
          f"({'TRUSTED' if drift_self_c.max() < 0.5 else 'WARNING'})")
    print(f"\nconfiguration separation ||chi_C-chi_NC||: min={chi_sep.min():.5f} max={chi_sep.max():.5f} "
          f"mean={chi_sep.mean():.5f}")
    print(f"magnet position separation: min={magnet_sep_mm.min():.4f}mm max={magnet_sep_mm.max():.4f}mm "
          f"mean={magnet_sep_mm.mean():.4f}mm")
    print(f"\ne_C_to_C (P_C under its own model):   min={e_C_to_C.min():.4f} max={e_C_to_C.max():.4f} "
          f"mean={e_C_to_C.mean():.4f}mm")
    print(f"e_NC_to_NC (P_NC under its own model): min={e_NC_to_NC.min():.4f} max={e_NC_to_NC.max():.4f} "
          f"mean={e_NC_to_NC.mean():.4f}mm")
    print(f"e_NC_to_C (P_NC's path, evaluated under the REAL contact model): "
          f"min={e_NC_to_C.min():.4f} max={e_NC_to_C.max():.4f} mean={e_NC_to_C.mean():.4f}mm")
    print(f"e_C_to_NC (P_C's path, evaluated under the no-contact model):    "
          f"min={e_C_to_NC.min():.4f} max={e_C_to_NC.max():.4f} mean={e_C_to_NC.mean():.4f}mm")

    # Where does the magnet-separation signal first become non-negligible?
    threshold_mm = 0.5
    onset_idx = np.argmax(magnet_sep_mm > threshold_mm) if np.any(magnet_sep_mm > threshold_mm) else -1
    if onset_idx >= 0:
        print(f"\nmagnet-position separation first exceeds {threshold_mm}mm at "
              f"node {onset_idx}, s={s_mm[onset_idx]:.2f}mm")
    threshold_e = 0.5
    onset_e = np.argmax(e_NC_to_C > threshold_e) if np.any(e_NC_to_C > threshold_e) else -1
    if onset_e >= 0:
        print(f"e_NC_to_C first exceeds {threshold_e}mm at node {onset_e}, s={s_mm[onset_e]:.2f}mm")


if __name__ == "__main__":
    main()
