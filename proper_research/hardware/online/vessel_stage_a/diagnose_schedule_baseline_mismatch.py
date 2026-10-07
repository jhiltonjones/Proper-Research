"""Focused follow-up to validate_mpc_schedule_jacobians.py's finding: the
saved schedule's Jacobian at several locations differs from a branch-
consistent, FD-validated recomputation by 15-41% relative error. Root
cause located in _make_beam_jacobian_callback (controller_factory_joint_
space.py): every single call resets the model to
`forward_adapter.get_baseline_cache_copy()` (the adapter's CONSTRUCTION-
TIME equilibrium, never advanced during precompute_schedule's loop, since
that loop never calls the forward adapter itself) before solving -- i.e.
every schedule sample is warm-started from the SAME fixed baseline, not
from a properly continued trajectory.

This script asks the one remaining question: at the worst-offending test
state (CONTACT s~60mm, where even the branch-consistent analytical
Jacobian showed 22% error vs FD), does the BASELINE-RESET solve converge
to a meaningfully DIFFERENT TIP POSITION than the branch-consistent
continuation solve? If yes: a genuine different-equilibrium-branch bug.
If the tip positions agree closely but the Jacobians still differ by
15-40%: the forward solution itself is branch-robust, and the Jacobian
discrepancy is pure numerical sensitivity of the implicit-sensitivity
Hessian inversion to sub-mm/sub-percent differences in u* -- consistent
with (not a new instance of) this project's existing Stage-3 finding
that the Jacobian is far more fragile than the forward tip prediction.

Usage: python diagnose_schedule_baseline_mismatch.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import CURRENT_LUMEN_FILE
from proper_research.simulation.magnetic_beam.solver_optimized import solve_quasistatic_insertion_optimized
import proper_research.simulation.simulations.controller_factory_joint_space as cfjs

from proper_research.hardware.online.vessel_stage_a.compare_cnc_corrected_sweep import walk_continuation
from proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel import (
    _recalibrated_make_initial_poses,
)

PLAN_ROOT = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06"
TARGET_S_MM = 60.0


def main():
    initial_poses = _recalibrated_make_initial_poses()
    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=0.11, jacobian_mode="accurate",
        plant_contact=True, initial_poses=initial_poses,
    )
    model = bundle.models["contact"]
    adapter = build_diagnostic_adapter(beam_model=model, controller_pack=controller_pack, jacobian_mode="accurate")

    npz = np.load(f"{PLAN_ROOT}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    state_ref = npz["state_reference"]
    path_s_mm = npz["path_s_m"] * 1e3
    idx = int(np.argmin(np.abs(path_s_mm - TARGET_S_MM)))
    print(f"target: idx={idx} s={path_s_mm[idx]:.2f}mm")

    # --- (A) branch-consistent: sequential continuation from s=0 ---
    print(f"walking {idx+1} samples via sequential continuation...", flush=True)
    tips_cont, u_stars = walk_continuation(model, adapter, state_ref, idx + 1)
    tip_continuation = tips_cont[idx]
    u_continuation = u_stars[idx]
    print(f"[continuation] tip = {np.round(tip_continuation*1e3, 4)} mm")

    # --- (B) schedule-style: reset to the adapter's CONSTRUCTION-TIME
    # baseline cache (exactly what _make_beam_jacobian_callback's
    # `model.set_cache(forward_adapter.get_baseline_cache_copy())` does),
    # then solve directly at this one state -- no continuation at all.
    chi = state_ref[idx]
    T_R_M = adapter.magnet_transform(chi)
    p7 = cfjs._pose7_from_transform(T_R_M, float(chi[6]))
    # JointSpaceBeamMPCAdapter itself has no cache methods -- they live on
    # the ControllerForwardAdapter that _make_beam_forward_callback built
    # and attached to beam_output_fn (see controller_factory_joint_space.py
    # _make_beam_forward_callback: beam_output.get_baseline_cache_copy =
    # forward_adapter.get_baseline_cache_copy). This is the EXACT object
    # _make_beam_jacobian_callback's beam_jacobian closure reads from.
    baseline_cache = adapter.beam_output_fn.get_baseline_cache_copy()
    model.set_cache(baseline_cache)
    result_baseline = model.solve(p7, commit=True, reuse_cache=True)
    tip_baseline = np.asarray(result_baseline.tip, dtype=float)
    u_baseline = np.asarray(result_baseline.u_flat_opt, dtype=float)
    print(f"[baseline-reset, schedule-style] tip = {np.round(tip_baseline*1e3, 4)} mm")

    tip_diff_mm = float(np.linalg.norm(tip_continuation - tip_baseline)) * 1e3
    u_diff = float(np.linalg.norm(u_continuation - u_baseline))
    u_norm = float(np.linalg.norm(u_continuation))
    print(f"\ntip position difference (continuation vs baseline-reset): {tip_diff_mm:.4f}mm")
    print(f"u* difference: ||du||={u_diff:.6e}  ||u_continuation||={u_norm:.6e}  "
          f"relative={u_diff/max(u_norm,1e-12):.6e}")

    if tip_diff_mm > 0.5:
        print("\n==> DIFFERENT EQUILIBRIUM BRANCH: baseline-reset warm-start converges to a "
              "meaningfully different physical tip position than the trajectory's own "
              "properly-continued branch. This is a genuine branch-selection bug in "
              "precompute_schedule's use of _make_beam_jacobian_callback for OFFLINE "
              "schedule-building (the baseline-reset design is appropriate for the LIVE "
              "forward/Jacobian split, not for building a schedule with no forward "
              "adapter stepping through the trajectory in parallel).")
    else:
        print("\n==> SAME tip position (sub-mm) despite different warm starts: the forward "
              "solution is branch-robust here. The large schedule-vs-FD Jacobian error is "
              "pure numerical sensitivity of the implicit-sensitivity Hessian inversion to "
              "small, physically-insignificant differences in u* -- consistent with (not a "
              "new instance of) this project's existing Stage-3 finding that the Jacobian "
              "is far more fragile than the forward tip prediction.")


if __name__ == "__main__":
    main()
