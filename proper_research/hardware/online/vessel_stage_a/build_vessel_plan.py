#!/usr/bin/env python3
"""Build a vessel offline plan (Layer 1 inverse configuration -> Layer 2
global recovery if needed -> Layer 3 time-parameterization) from:
  - a detect_blue.py-format lumen file (robot-frame lumen_C_m/lumen_R_m),
  - a start-position JSON (from capture_live_start_position.py, or hand-
    built the same way), which REPLACES the library-default start_point
    entirely (not an offset from it),
  - a fixed magnet-to-beam-base safety floor (kept independent of the
    start position -- see the 2026-09-27 lesson: setting these equal
    pins the magnet on the constraint boundary with ~0 slack from node 0).

Also applies the finite-difference chain-rule validation fix found
2026-09-28: the library default FD step (1e-6 rad/m) is far smaller than
the contact-aware beam solver's own convergence precision
(optimizer_gtol=1e-5), so at 1e-6 the validation measures pure solver
noise (~100% "error") rather than the analytic Jacobian's real accuracy
(~5-10% at a properly-scaled step). This is NOT a bug in the analytic
Jacobian -- swept step sizes 1e-6..2.5e-2 directly and confirmed the error
falls to a genuine ~9% plateau around joint_step~1.5-2e-2 rad /
insertion~5e-3m before rising again from real curvature at larger steps.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.build_vessel_plan \\
        --lumen-file <your lumen file> \\
        --start-position-json vessel_magnet_initial_position_2026-10-02_recalibrated.json \\
        --insertion-max-mm 65 \\
        --output-root plans/vessel_recalibrated_2026-10-02
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

import proper_research.simulation.simulations.initial_conditions as initial_conditions_mod
import proper_research.planning.planning_context as planning_context_mod

# 2026-10-02 fix: removed the --z-raise-mm/zraise_patch indirection entirely.
# That mechanism required every script building or running a plan to pass a
# MATCHING offset by hand (build_vessel_plan.py / run_mpc_delay_aware_vessel.py
# / capture_live_start_position.py each had their own --z-raise-mm default,
# and a mismatch silently produced a plan whose beam-base pivot disagreed
# with the live vision/magnet frame by the z-raise amount -- see this
# script's own prior 2026-09-29 fix note for one such incident). The rig has
# since been recalibrated via real forward kinematics (see
# vessel_magnet_initial_position_2026-10-02_recalibrated.json,
# `capture_live_start_position.py`) to a single fixed beam-base height --
# BEAM_BASE_PIVOT_Z below -- so there is no longer a "raised vs unraised"
# choice to make here at all.
BEAM_BASE_PIVOT_XY_ROT = np.array([0.525575, -0.670028, 3.14159265, 0.0, 0.0])
BEAM_BASE_PIVOT_Z = -0.039627  # recalibrated 2026-10-02, matches beam_base_pivot_xyz_R in
                                # vessel_magnet_initial_position_2026-10-02_recalibrated.json
L_CMD = 0.03044
DT_INIT = 0.01


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True,
                    help="detect_blue.py-format lumen file (robot frame R, raised workspace)")
    p.add_argument("--start-position-json", required=True,
                    help="from capture_live_start_position.py -- replaces the default start_point")
    p.add_argument("--insertion-max-mm", type=float, default=65.0)
    p.add_argument("--position-tolerance-mm", type=float, default=3.5)
    p.add_argument("--tangent-tolerance-deg", type=float, default=179.9)
    p.add_argument("--maximum-function-evaluations", type=int, default=15,
                    help="per-node solver budget; the library default (300) let a hard node "
                         "take 60+ minutes 3x in a row on 2026-09-27. Diagnosed 2026-09-28: the "
                         "core physics solve (L-BFGS-B, maxiter=40) is properly bounded -- this "
                         "is NOT an infinite loop -- but the outer multistart search sometimes "
                         "lands on an alternate joint-configuration branch where each evaluation "
                         "costs ~1-2s instead of ~0.15-0.3s (confirmed by instrumenting "
                         "forward_output calls directly), and how many times that slow branch "
                         "gets hit is sensitive to CPU contention from other processes on the "
                         "machine -- the SAME settings got stuck 5+ min at one node in one run "
                         "and sailed through node 0-49/54 in ~6.5 min in another. A tight budget "
                         "(15/2) keeps worst-case per-node time bounded to roughly a minute "
                         "regardless; Layer 2 repairs anything Layer 1 still can't reach.")
    p.add_argument("--maximum-multistart-attempts", type=int, default=2,
                    help="candidate initial guesses per node (Jacobian-predicted, "
                         "extrapolated, previous-node, then random perturbations of "
                         "previous-node up to this many total). At the default (2), "
                         "the first 2-3 deterministic candidates (Jacobian-predicted/"
                         "extrapolated/previous) typically already fill this budget, "
                         "so a random perturbation is rarely or never tried -- raise "
                         "this (e.g. 6-8) to give the solver real alternate starting "
                         "points for a node the deterministic guesses can't reach, "
                         "e.g. a sharp local bend where linear extrapolation from a "
                         "smoother neighbouring node is a poor seed.")
    p.add_argument("--multistart-joint-perturbation-rad", type=float, default=None,
                    help="override InverseConfigurationPlannerConfig's default "
                         "multistart random-perturbation scale (0.05 rad, ~2.9deg) "
                         "applied around the previous node's state when generating "
                         "extra multistart guesses -- widen this to let multistart "
                         "actually jump far enough to escape a bad local seed near a "
                         "sharp bend, rather than only ever searching a ~3deg "
                         "neighbourhood of it.")
    p.add_argument("--node-timeout-s", type=float, default=0.0,
                    help="wall-clock deadline (seconds) for EACH per-guess node solve "
                         "attempt (0 = disabled, previous behaviour). Unlike "
                         "--maximum-function-evaluations (which bounds the outer "
                         "solver's iteration count), this bounds real time -- added "
                         "2026-09-29 after a node's SINGLE inner nonlinear beam-solve "
                         "call (not the iteration count) turned out to be the actual "
                         "bottleneck at an ill-conditioned/near-rank-deficient state, "
                         "something no iteration cap can see. A timed-out attempt is "
                         "treated exactly like any other failed numerical attempt -- "
                         "the node moves to its next multistart guess, or falls "
                         "through to Layer 2's recovery pass if all guesses fail. Try "
                         "60-120s.")
    p.add_argument("--finite-difference-joint-step-rad", type=float, default=1.5e-2,
                    help="see this script's docstring -- the library default (1e-6) is far "
                         "smaller than the contact solver's own precision and produces a "
                         "~100%% spurious validation failure.")
    p.add_argument("--finite-difference-insertion-step-m", type=float, default=5.0e-3)
    p.add_argument("--maximum-chain-rule-relative-error", type=float, default=0.15)
    p.add_argument("--skip-chain-rule-validation-at-start", action="store_true",
                    help="disable the one-time chain-rule finite-difference check at the "
                         "plan's start state (InverseConfigurationPlannerConfig."
                         "finite_difference_validation_at_start). Found 2026-10-01: at a "
                         "shallow start insertion (L0~25mm) the insertion-derivative "
                         "(d/dL) column can disagree hugely between the analytical and "
                         "finite-difference Jacobian (seen: ~80-98%% relative error) "
                         "purely because the +-5mm finite-difference window straddles a "
                         "genuine contact-model transition near the vessel mouth -- not "
                         "a sign the analytical Jacobian used during the actual solve is "
                         "wrong, just that THIS validation's step size is too coarse at "
                         "this exact state. Confirmed by direct inspection: the joint "
                         "(d/dq) columns always agreed closely; only d/dL disagreed. "
                         "Only skip this if you've independently confirmed (as above) "
                         "that the disagreement is concentrated in d/dL at a shallow "
                         "start insertion -- it does not disable any other validation, "
                         "and the solve itself still uses the analytical Jacobian "
                         "throughout, unaffected by this flag. "
                         "SECOND, DIFFERENT known cause (2026-10-02): the "
                         "plant_diagnostic_joint_adapter's analytical beam Jacobian used "
                         "to be built with jacobian_mode='fast' unconditionally (hardcoded "
                         "in controller_factory_joint_space.build_controller, never "
                         "threaded through to here) -- the SAME 'fast' mode already "
                         "documented elsewhere (build_or_load_schedule's own comment) to "
                         "have a single-tick numerical-estimator artifact, a spurious "
                         "spike while the true tip output changes smoothly. When this "
                         "fires, the mismatch is NOT confined to d/dL -- most/all "
                         "entries disagree by a large, roughly consistent factor "
                         "(analytical >> finite-difference), and confirmed live "
                         "2026-10-02 to make Layer 1's own SLSQP solve chase a bad "
                         "gradient: every node hit the iteration limit, jacobian_"
                         "condition swung from 1e5 to 1e8 between neighbouring nodes, "
                         "and the magnet position jumped tens of mm from joint steps of "
                         "only ~0.03-0.13 rad -- a recalibrated-geometry/locked-Z plan at "
                         "L0~30mm, relative error 178%% vs finite-difference. NOW FIXED: "
                         "jacobian_mode is threaded through build_controller -> "
                         "planning_context -> vessel_context -> here (see --jacobian-mode "
                         "below), defaulting to 'accurate' for this script specifically "
                         "(every other caller of those three functions still defaults to "
                         "'fast', unaffected). This flag's own role is unchanged by that "
                         "fix -- it only skips the validation CHECK, not the Jacobian "
                         "mode the real solve uses.")
    p.add_argument("--jacobian-mode", choices=("fast", "accurate"), default="accurate",
                    help="mode for the analytical beam Jacobian Layer 1's SLSQP solve "
                         "uses (threaded through to controller_factory_joint_space."
                         "build_controller -- see its docstring). Defaults to 'accurate' "
                         "HERE (unlike every other caller, which defaults to 'fast') "
                         "because 'fast' mode's numerical-estimator artifact was "
                         "confirmed 2026-10-02 to make Layer 1 chase a bad gradient at "
                         "every node for a recalibrated-geometry/L0~30mm plan -- see "
                         "--skip-chain-rule-validation-at-start's help above for the "
                         "full story. 'accurate' is slower per-evaluation; pass 'fast' "
                         "to restore the old (buggy-for-this-case) behaviour if needed.")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--insertion-start-mm", type=float, default=L_CMD * 1000.0,
                    help="initial inserted length the offline planner starts from AND the "
                         "length the live beam must be set to before running this plan -- "
                         f"defaults to the historical {L_CMD * 1000.0:.1f}mm constant. Pass "
                         "the live-measured insertion (see "
                         "proper_research.vision.insertion_check or "
                         "checkpoint_beam_shape_campaign.measure_current_insertion_mm) "
                         "when starting a plan from wherever the robot/beam actually is "
                         "right now, rather than assuming the old default.")
    p.add_argument("--lock-magnet-z", action="store_true",
                    help="pin the source magnet's world-Z to exactly its start value for "
                         "the whole plan (floor AND ceiling, not just the default "
                         "no-decrease floor) -- InverseConfigurationPlannerConfig."
                         "enforce_magnet_z_fixed. Use when the physical setup requires the "
                         "magnet to stay on a single Z plane throughout.")
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()

    BEAM_BASE_PIVOT = np.array([
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z,
        BEAM_BASE_PIVOT_XY_ROT[2], BEAM_BASE_PIVOT_XY_ROT[3], BEAM_BASE_PIVOT_XY_ROT[4],
    ])

    with open(args.start_position_json) as f:
        start_ref = json.load(f)
    start_point = np.array(start_ref["magnet_pose6_R"], dtype=float)
    exclusion_floor_m = float(start_ref["exclusion_floor_mm"]) / 1000.0
    insertion_max_m = args.insertion_max_mm / 1000.0

    insertion_start_m = args.insertion_start_mm / 1000.0

    def _make_initial_poses():
        return BEAM_BASE_PIVOT.copy(), start_point.copy(), insertion_start_m, DT_INIT

    initial_conditions_mod.make_initial_poses = _make_initial_poses
    planning_context_mod.make_initial_poses = _make_initial_poses

    print(f"[build] pivot_point (beam base) = {BEAM_BASE_PIVOT.tolist()}")
    print(f"[build] start_point (source magnet) = {start_point.tolist()}  "
          f"(from {args.start_position_json})")
    print(f"[build] magnet_beam_base_exclusion_radius_m (fixed floor) = {exclusion_floor_m:.6f} "
          f"({exclusion_floor_m * 1e3:.2f}mm)")
    print(f"[build] insertion_max_m = {insertion_max_m:.4f} ({args.insertion_max_mm:.1f}mm)")
    print(f"[build] insertion_start_m = {insertion_start_m:.4f} ({args.insertion_start_mm:.2f}mm)")

    from proper_research.planning.vessel_context import build_vessel_planning_context
    from proper_research.planning.planning_context import make_inverse_config
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        InverseConfigurationPlannerConfig,
        solve_from_controller_pack,
    )
    from proper_research.planning.global_constrained_configuration_path import (
        GlobalConfigurationOptimizerConfig,
        optimize_from_saved_inverse_result,
    )
    from proper_research.planning.time_parameterized_configuration_path import (
        TimeParameterizationConfig,
        time_parameterize_saved_global_path,
        time_parameterize_saved_inverse_path,
    )

    print(f"[build] jacobian_mode = {args.jacobian_mode!r}")
    exp_cfg, bundle, controller_pack, out_root, centreline, lumen_R, provenance = (
        build_vessel_planning_context(
            lumen_file=args.lumen_file, insertion_max_m=insertion_max_m,
            jacobian_mode=args.jacobian_mode,
        )
    )
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    perimeter = float(np.sum(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))
    print(f"[vessel] {args.lumen_file} -> {centreline.shape[0]} points, "
          f"arclength {1e3 * perimeter:.1f}mm, radius {1e3 * lumen_R.min():.2f}-{1e3 * lumen_R.max():.2f}mm")
    print(f"[vessel] centreline[0]  (world/R) = {np.round(centreline[0], 4).tolist()}")
    print(f"[vessel] centreline[-1] (world/R) = {np.round(centreline[-1], 4).tolist()}")

    vessel_dir = out_root / "vessel_lumen"
    vessel_dir.mkdir(parents=True, exist_ok=True)
    np.savez(vessel_dir / "vessel_centreline.npz", lumen_C=centreline, lumen_R=lumen_R)
    (vessel_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    planner_config = make_inverse_config()
    shared = {
        f.name: getattr(planner_config, f.name)
        for f in fields(InverseConfigurationPlannerConfig)
        if hasattr(planner_config, f.name)
    }
    shared["tangent_tolerance_rad"] = math.radians(min(args.tangent_tolerance_deg, 179.95))
    shared["position_tolerance_m"] = args.position_tolerance_mm * 1.0e-3
    if "solve_initial_node" in shared:
        shared["solve_initial_node"] = True
    shared["source_magnet_lumen_exclusion_radius_m"] = None
    shared["magnet_beam_base_exclusion_radius_m"] = exclusion_floor_m
    shared["maximum_function_evaluations"] = args.maximum_function_evaluations
    shared["maximum_multistart_attempts"] = args.maximum_multistart_attempts
    shared["finite_difference_joint_step_rad"] = args.finite_difference_joint_step_rad
    shared["finite_difference_insertion_step_m"] = args.finite_difference_insertion_step_m
    shared["maximum_chain_rule_relative_error"] = args.maximum_chain_rule_relative_error
    shared["node_timeout_s"] = args.node_timeout_s
    if args.skip_chain_rule_validation_at_start:
        shared["finite_difference_validation_at_start"] = False
    if args.lock_magnet_z:
        shared["enforce_magnet_z_fixed"] = True
    if args.multistart_joint_perturbation_rad is not None:
        shared["multistart_joint_perturbation_rad"] = args.multistart_joint_perturbation_rad

    planner_config = InverseConfigurationPlannerConfig(**shared)
    print(f"[layer1] tangent tolerance = {args.tangent_tolerance_deg:.0f}deg, "
          f"position tolerance = {1e3 * planner_config.position_tolerance_m:.2f}mm")
    print(f"[layer1] node_timeout_s = {planner_config.node_timeout_s} "
          f"(0 = disabled), maximum_multistart_attempts = {planner_config.maximum_multistart_attempts}, "
          f"multistart_joint_perturbation_rad = {planner_config.multistart_joint_perturbation_rad}")

    inverse_dir = vessel_dir / "offline_inverse_configuration"
    print(f"\n[layer1] inverse planning -> {inverse_dir}", flush=True)
    t0 = time.perf_counter()
    inverse_result = solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=planner_config,
        output_dir=inverse_dir,
        alternative_initial_states=[np.asarray(controller_pack["p0"], dtype=float)],
    )
    print(f"[layer1] done in {time.perf_counter() - t0:.1f}s  "
          f"all_nodes_feasible={inverse_result.all_nodes_feasible}", flush=True)

    tp_config = TimeParameterizationConfig(
        sample_period_s=float(args.dt),
        state_velocity_limit=tuple([0.10] * 6 + [2.0e-3]),
        state_acceleration_limit=tuple([0.5] * 6 + [0.02]),
        maximum_path_speed_m_s=3.0e-3,
        constraint_tolerance=1.0e-8,
        velocity_safety_factor=0.8,
        acceleration_safety_factor=0.8,
        require_nonlinear_beam_feasible=False,
        require_saved_global_feasible=False,
        require_saved_dense_feasible=False,
    )

    time_param_dir = out_root / "time_parameterized_configuration_path"

    if inverse_result.all_nodes_feasible:
        print(f"\n[layer3] time-parameterising the INVERSE path -> {time_param_dir}", flush=True)
        time_parameterize_saved_inverse_path(
            inverse_output_dir=inverse_dir, controller_pack=controller_pack,
            config=tp_config, output_dir=time_param_dir, lumen_C=centreline,
        )
    else:
        global_dir = vessel_dir / "global_configuration_converged"
        global_config = GlobalConfigurationOptimizerConfig(
            mode="recover_partial",
            position_tolerance_m=float(planner_config.position_tolerance_m),
            tangent_tolerance_rad=float(planner_config.tangent_tolerance_rad),
            path_step_m=3.0e-3, minimum_path_step_m=1.5e-3,
            maximum_refinement_rounds=1, maximum_nodes=200,
            maximum_iterations=40, maximum_wall_time_s=180.0,
            stagnation_function_evaluations=4000, dense_validation_enabled=True,
            compute_node_jacobian_diagnostics=False, require_contact_model=True,
            insertion_non_decreasing=False, fix_initial_state=True,
        )
        print(f"\n[layer2] Layer 1 incomplete -- global optimiser (recover_partial) -> {global_dir}", flush=True)
        t0 = time.perf_counter()
        global_result = optimize_from_saved_inverse_result(
            inverse_output_dir=inverse_dir, controller_pack=controller_pack,
            lumen_C=centreline, config=global_config, output_dir=global_dir,
        )
        print(f"[layer2] done in {time.perf_counter() - t0:.1f}s  "
              f"nodes={len(getattr(global_result, 'nodes', []))}", flush=True)

        print(f"\n[layer3] time-parameterising the GLOBAL path -> {time_param_dir}", flush=True)
        time_parameterize_saved_global_path(
            global_output_dir=global_dir, controller_pack=controller_pack,
            config=tp_config, output_dir=time_param_dir, lumen_C=centreline,
        )

    npz = time_param_dir / "time_parameterized_configuration_path.npz"
    data = np.load(npz)
    n = int(np.asarray(data["time_s"]).size)
    dur = float(np.asarray(data["time_s"])[-1])
    print(f"\n[done] {npz}\n"
          f"       {n} samples, {dur:.1f}s at dt={args.dt}s\n"
          f"       desired_position_m[0]  = {np.round(np.asarray(data['desired_position_m'])[0], 4).tolist()}\n"
          f"       desired_position_m[-1] = {np.round(np.asarray(data['desired_position_m'])[-1], 4).tolist()}")


if __name__ == "__main__":
    main()
