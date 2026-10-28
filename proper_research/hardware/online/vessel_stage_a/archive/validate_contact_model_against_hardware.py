"""Item 3 (the highest-value mechanistic test): does the physical no-contact
replay follow the CONTACT model's prediction better than the no-contact
model's own prediction, once wall interaction starts?

Walks each plan's own ACTUAL commanded trajectory (the time-parameterized
state_reference each live open-loop run tracked, q,L at every control
tick -- not the offline inverse-configuration grid) through both forward
models via validated sequential continuation (every sample warm-started
from the previous sample's own converged u*, same methodology as
compare_PC_vs_PNC.py / compare_cnc_corrected_sweep.py), then joins against
the MEASURED hardware tip at the matching ref_index:

  e_model,C  = ||p_measured^{P_C}  - F_C(P_C)||    (contact run vs contact model)
  e_model,NC = ||p_measured^{P_NC} - F_C(P_NC)||   (no-contact run vs CONTACT model)
  e_self,NC  = ||p_measured^{P_NC} - F_NC(P_NC)||  (no-contact run vs its OWN model)

If e_model,NC < e_self,NC once contact starts, the no-contact run's
physical failure is explained BY the contact physics the no-contact
planner neglected -- not just "the no-contact plan failed."

Usage: python validate_contact_model_against_hardware.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE, beam_base_pose6
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs

PLAN_C = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"
PLAN_NC = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_nocontact_2026-10-06"
RUN_C = "close_loop_logs/myrun/openloop_contact_2026-10-07_20261007T113710Z"
RUN_NC = "close_loop_logs/myrun/openloop_nocontact_2026-10-07_20261007T114533Z"
OUT_DIR = Path("plans/stage3_design/openloop_hw_analysis")
S_CONTACT_ONSET_MM = 28.50


def walk_model(model, adapter, states: np.ndarray, n: int, label: str) -> np.ndarray:
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
            print(f"    [{label}] k={k}/{n}", flush=True)
    return tips


def measured_tip_by_ref_index(run_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{run_dir}/tip_trajectory.csv")
    # first occurrence of each ref_index -- the fresh sample at that config,
    # not repeated terminal-hold ticks.
    first = df.drop_duplicates(subset="ref_index", keep="first")
    return first[["ref_index", "tip_x_m", "tip_y_m", "tip_z_m"]].set_index("ref_index")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    design = json.loads(open("plans/stage3_design/phi30_L30_left1mm_newwall_design.json").read())
    start_point = np.asarray(design["magnet_pose6_R"], dtype=float)
    initial_poses = (beam_base_pose6(), start_point.copy(), 0.030, 0.01)

    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate",
        plant_contact=True, initial_poses=initial_poses,
    )
    model_c = bundle.models["contact"]
    model_nc = bundle.models["no_contact"]
    adapter_c = build_diagnostic_adapter(beam_model=model_c, controller_pack=controller_pack, jacobian_mode="accurate")
    adapter_nc = build_diagnostic_adapter(beam_model=model_nc, controller_pack=controller_pack, jacobian_mode="accurate")

    npz_c = np.load(f"{PLAN_C}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    npz_nc = np.load(f"{PLAN_NC}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    state_c = npz_c["state_reference"]
    state_nc = npz_nc["state_reference"]
    s_c_mm = npz_c["path_s_m"] * 1e3
    s_nc_mm = npz_nc["path_s_m"] * 1e3

    meas_c = measured_tip_by_ref_index(RUN_C)
    meas_nc = measured_tip_by_ref_index(RUN_NC)
    n_c = int(meas_c.index.max()) + 1          # contact run reached the full path
    n_nc = min(int(meas_nc.index.max()) + 4, len(state_nc))  # NC run aborted early -- only walk as far as needed

    print(f"walking P_C's own {n_c} commanded samples through F_C (self-consistency)...", flush=True)
    F_C_of_PC = walk_model(model_c, adapter_c, state_c, n_c, "F_C(P_C)")

    print(f"walking P_NC's own {n_nc} commanded samples through F_C (THE key cross-evaluation)...", flush=True)
    F_C_of_PNC = walk_model(model_c, adapter_c, state_nc, n_nc, "F_C(P_NC)")

    print(f"walking P_NC's own {n_nc} commanded samples through F_NC (self-consistency)...", flush=True)
    F_NC_of_PNC = walk_model(model_nc, adapter_nc, state_nc, n_nc, "F_NC(P_NC)")

    rows_c = []
    for k in range(n_c):
        if k not in meas_c.index:
            continue
        p_meas = meas_c.loc[k, ["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy(dtype=float)
        e_model_c = float(np.linalg.norm(p_meas - F_C_of_PC[k])) * 1e3
        rows_c.append({"ref_index": k, "s_mm": float(s_c_mm[k]), "e_model_C_mm": e_model_c})
    out_c = pd.DataFrame(rows_c)

    rows_nc = []
    for k in range(n_nc):
        if k not in meas_nc.index:
            continue
        p_meas = meas_nc.loc[k, ["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy(dtype=float)
        e_model_nc = float(np.linalg.norm(p_meas - F_C_of_PNC[k])) * 1e3
        e_self_nc = float(np.linalg.norm(p_meas - F_NC_of_PNC[k])) * 1e3
        rows_nc.append({
            "ref_index": k, "s_mm": float(s_nc_mm[k]),
            "e_model_NC_vs_FC_mm": e_model_nc, "e_model_NC_vs_FNC_mm": e_self_nc,
        })
    out_nc = pd.DataFrame(rows_nc)

    out_c.to_csv(OUT_DIR / "model_validation_contact_run.csv", index=False)
    out_nc.to_csv(OUT_DIR / "model_validation_nocontact_run.csv", index=False)

    print("\n=== e_model,C = ||measured^PC - F_C(P_C)|| ===")
    print(f"  mean={out_c['e_model_C_mm'].mean():.4f}mm max={out_c['e_model_C_mm'].max():.4f}mm "
          f"(n={len(out_c)})")

    print("\n=== P_NC run: measured vs F_C(P_NC) [contact model] vs F_NC(P_NC) [its own model] ===")
    pre = out_nc[out_nc["s_mm"] < S_CONTACT_ONSET_MM]
    post = out_nc[out_nc["s_mm"] >= S_CONTACT_ONSET_MM]
    for name, sub in [("pre-contact (s<28.5mm)", pre), ("post-contact (s>=28.5mm)", post), ("all", out_nc)]:
        if len(sub) == 0:
            continue
        print(f"  {name:28s} n={len(sub):4d}  "
              f"mean e_vs_F_C={sub['e_model_NC_vs_FC_mm'].mean():7.4f}mm  "
              f"mean e_vs_F_NC={sub['e_model_NC_vs_FNC_mm'].mean():7.4f}mm  "
              f"(F_C better on {100.0*np.mean(sub['e_model_NC_vs_FC_mm'] < sub['e_model_NC_vs_FNC_mm']):5.1f}% of ticks)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(out_nc["s_mm"].to_numpy(), out_nc["e_model_NC_vs_FC_mm"].to_numpy(),
            color="#1b7f3b", lw=1.5, label=r"$\|p_{meas}^{NC} - F_C(P_{NC})\|$ (contact model)")
    ax.plot(out_nc["s_mm"].to_numpy(), out_nc["e_model_NC_vs_FNC_mm"].to_numpy(),
            color="#b3331d", lw=1.5, label=r"$\|p_{meas}^{NC} - F_{NC}(P_{NC})\|$ (no-contact model)")
    ax.axvline(S_CONTACT_ONSET_MM, color="k", linestyle=":", lw=1.2,
               label=f"predicted contact onset (s={S_CONTACT_ONSET_MM:.1f}mm)")
    ax.set_xlabel("path progress s (mm)")
    ax.set_ylabel("model-prediction error (mm)")
    ax.set_title("No-contact hardware replay: which model predicts it better?")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_model_validation.png", dpi=160)
    plt.close(fig)
    print(f"\nsaved fig2_model_validation.png + model_validation_*.csv -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
