#!/usr/bin/env python3
"""Run the global configuration smoother with settings that can converge.

This replaces `debug_global_from_saved_inverse.py --full` for production runs.
The debug driver's defaults are diagnostic ones — 20 trust-constr iterations
and zero refinement rounds — and a run that stops at its iteration cap is an
intermediate trust-region iterate, not a solution.  Nothing downstream should
be judged from one.

What this driver adds, in the order the problems bite:

1. **A seed preflight.**  Before touching the optimiser it checks the saved
   inverse path against the tolerance the global layer is about to enforce.
   One infeasible node is enough to disable `preserve_feasible_seed_on_failure`
   for the whole path — the safety net that keeps a feasible input from being
   turned into an infeasible output — and that is worth knowing before a long
   solve rather than after.  `--auto-tolerance` sets the global tolerance from
   what layer 1 actually achieved instead of a number it never met.

2. **Bounded convergence settings.**  The nonlinear solve starts on a coarser
   exact subset of the inverse grid, retains keep-out boundary nodes, stops
   after a wall-time/stagnation limit, and preserves its best feasible iterate.

3. **Balanced smoothing weights.**  `--balance-smoothing-weights` sets the
   per-coordinate second-difference weights so every coordinate contributes
   equally to the objective at the seed.  With a uniform weight the coordinate
   whose scaled second derivative is already smallest is the cheapest to
   sacrifice, which is how insertion — the axis with the tightest acceleration
   limit — ends up rougher after a pass that improved every joint.

4. **The keep-out and the task-space controls.**  `--exclusion-radius` restores
   the source-magnet / lumen constraint that layer 1 enforces and layer 2 did
   not have.  `--tolerance-spend-fraction` bounds how much of the tip tolerance
   smoothing may consume.  `--tip-centring-weight` pays for a centred tip
   instead of forbidding an off-centre one.

Typical first run:

    python run_global_configuration.py \\
        --exclusion-radius 0.12 \\
        --auto-tolerance \\
        --balance-smoothing-weights \\
        --tolerance-spend-fraction 0.5
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from proper_research.planning.global_constrained_configuration_path import (
        GlobalConfigurationOptimizerConfig,
        load_inverse_path_result,
        optimize_from_saved_inverse_result,
    )
    from proper_research.planning.global_upgrades import (
        balanced_second_difference_weights,
    )
    from proper_research.planning.planning_context import (
        build_planning_context,
        make_inverse_config,
    )
except ModuleNotFoundError:  # standalone review
    from global_constrained_configuration_path import (  # type: ignore
        GlobalConfigurationOptimizerConfig,
        load_inverse_path_result,
        optimize_from_saved_inverse_result,
    )
    from global_upgrades import (  # type: ignore
        balanced_second_difference_weights,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inverse-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    convergence = parser.add_argument_group("convergence")
    convergence.add_argument("--maxiter", type=int, default=100)
    convergence.add_argument("--refinement-rounds", type=int, default=2)
    convergence.add_argument("--dense-samples", type=int, default=3)
    convergence.add_argument("--maximum-nodes", type=int, default=300)
    convergence.add_argument("--multistart-attempts", type=int, default=1)
    convergence.add_argument("--path-step-mm", type=float, default=1)
    convergence.add_argument("--max-runtime-minutes", type=float, default=45.0)
    convergence.add_argument("--stagnation-evaluations", type=int, default=8)
    convergence.add_argument(
        "--preserve-all-inverse-nodes",
        action="store_true",
        help="Disable the adaptive coarse initial grid (substantially slower).",
    )
    convergence.add_argument(
        "--node-jacobian-diagnostics",
        action="store_true",
        help="Compute expensive per-node condition numbers after optimization.",
    )

    seed = parser.add_argument_group("seed feasibility")
    seed.add_argument(
        "--auto-tolerance", action="store_true",
        help="Set the global tolerance from what the inverse path achieved, "
             "times --auto-tolerance-margin, instead of a tolerance it never met.",
    )
    seed.add_argument(
        "--auto-tolerance-margin",
        type=float,
        default=1.50,
        help="Effective tolerance multiplier around the achieved Layer-1 errors.",
    )
    seed.add_argument(
        "--allow-infeasible-seed", action="store_true",
        help="Continue even when the preflight finds the seed infeasible. "
             "The feasible-seed fallback will be inactive for the whole path.",
    )
    seed.add_argument(
        "--require-complete-inverse", action="store_true",
        help="Stop before the solve if Layer 1 did not reach the path end. "
             "By default a partial inverse path is refined over its feasible "
             "prefix and the run warns loudly.",
    )
    seed.add_argument(
        "--recover-partial", action="store_true",
        help="Run the global layer in recover_partial mode: seed with the "
             "feasible prefix, then attempt a slack-based feasibility "
             "restoration to extend toward the path end. Usually futile when "
             "Layer 1 stopped on a rank-deficient / ill-conditioned Jacobian.",
    )
    seed.add_argument(
        "--no-physical-tolerance-fallback", action="store_true",
        help="Disable retaining the inverse seed as the fallback when it meets "
             "the physical tolerance but not the tighter spent tolerance. "
             "Off by default: the seed fallback stays armed so a stalled solve "
             "returns the inverse path, not its last infeasible iterate.",
    )

    safety = parser.add_argument_group("safety and task space")
    safety.add_argument(
        "--exclusion-radius", type=float, default=None,
        help="Source-magnet / lumen keep-out radius [m]; use the same value "
             "the inverse planner used. Omitting it reproduces the old "
             "behaviour, which enforces no keep-out at all.",
    )
    safety.add_argument("--exclusion-tolerance", type=float, default=1.0e-6)
    safety.add_argument(
        "--dense-exclusion-acceptance-slack-mm", type=float, default=0.3,
        help="The keep-out is enforced as a hard constraint at the decision "
             "nodes only; dense validation between nodes can then fail by a "
             "hair even when every node clears it comfortably. A round whose "
             "ONLY dense-validation failure is a source-magnet keep-out "
             "margin within this slack (default 0.3 mm, negligible against "
             "a typically >=100 mm exclusion radius) is still accepted "
             "instead of triggering the all-or-nothing inverse-path "
             "fallback. Position/tangent tolerances and the box constraint "
             "are never loosened by this. Set to 0 to reproduce the old "
             "strict all-or-nothing behaviour exactly.",
    )
    safety.add_argument(
        "--dense-position-acceptance-slack-mm", type=float, default=0.0,
        help="Same idea as --dense-exclusion-acceptance-slack-mm, for the "
             "position task tolerance: a round whose only dense-validation "
             "position failure is within this many mm of the spent tolerance "
             "is still accepted. 0 (default) reproduces strict behaviour. "
             "Watch the '[GLOBAL DENSE GAP]' debug line first to see whether "
             "position is actually what's failing before reaching for this.",
    )
    safety.add_argument(
        "--dense-tangent-acceptance-slack-deg", type=float, default=0.0,
        help="Same idea, for the tangent task tolerance, in degrees.",
    )
    safety.add_argument("--require-analytical-magnet-jacobian", action="store_true")
    safety.add_argument("--tolerance-spend-fraction", type=float, default=1.0)
    safety.add_argument("--tip-centring-weight", type=float, default=0.0)
    safety.add_argument(
        "--tip-curvature-weight",
        type=float,
        default=0.0,
        help="Penalise second-difference curvature of the ACHIEVED tip path "
             "(not just of chi), linearised at each round's seed. This is the "
             "term that fixes a tip path that is jagged even though the joint "
             "path is smooth -- tip_centring_weight alone only pulls each node "
             "toward its own static target, it does not couple neighbours. "
             "Try something small relative to second-difference-weight and "
             "raise it until the tip trace is smooth without eating the whole "
             "iteration budget; 0.0 keeps the old behaviour.",
    )
    safety.add_argument(
        "--magnet-path-weight",
        type=float,
        default=0.0,
        help="Penalise the SOURCE-MAGNET path length (first difference of the "
             "magnet position, linearised at each round's seed). Nothing else "
             "in the objective touches the magnet -- it only appears as the "
             "keep-out inequality -- so without this the optimiser is free to "
             "swing the external magnet far out and sharply back as long as "
             "chi stays smooth. Raise it until the magnet path/straight-line "
             "ratio (printed as [GLOBAL MAGNET PATH]) drops; 0.0 keeps the old "
             "behaviour.",
    )
    safety.add_argument(
        "--magnet-curvature-weight",
        type=float,
        default=0.0,
        help="Penalise the source-magnet path CURVATURE (second difference). "
             "Use with or instead of --magnet-path-weight to kill an "
             "out-and-back swing without shortening the whole path.",
    )

    weights = parser.add_argument_group("smoothing weights")
    weights.add_argument("--balance-smoothing-weights", action="store_true")
    weights.add_argument("--first-difference-weight", type=float, default=1.0e-5)
    weights.add_argument("--second-difference-weight", type=float, default=1.0e-5)
    weights.add_argument("--seed-deviation-weight", type=float, default=1.0e-10)

    parser.add_argument(
        "--trace-after",
        type=int,
        default=0,
        help="Dump diagnostic stack traces at this interval [s]; 0 disables them.",
    )
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    if arguments.maxiter < 1:
        parser.error("--maxiter must be at least 1")
    if arguments.refinement_rounds < 0:
        parser.error("--refinement-rounds cannot be negative")
    if arguments.dense_samples < 1:
        parser.error("--dense-samples must be at least 1")
    if arguments.path_step_mm <= 0.0:
        parser.error("--path-step-mm must be positive")
    if arguments.max_runtime_minutes <= 0.0:
        parser.error("--max-runtime-minutes must be positive")
    if arguments.stagnation_evaluations < 1:
        parser.error("--stagnation-evaluations must be at least 1")
    if arguments.auto_tolerance_margin <= 1.0:
        parser.error("--auto-tolerance-margin must be greater than 1")
    if not 0.0 < arguments.tolerance_spend_fraction <= 1.0:
        parser.error("--tolerance-spend-fraction must lie in (0, 1]")
    if arguments.exclusion_radius is not None and arguments.exclusion_radius <= 0:
        parser.error("--exclusion-radius must be positive")
    if arguments.tip_curvature_weight < 0.0:
        parser.error("--tip-curvature-weight must be non-negative")
    if arguments.magnet_path_weight < 0.0:
        parser.error("--magnet-path-weight must be non-negative")
    if arguments.magnet_curvature_weight < 0.0:
        parser.error("--magnet-curvature-weight must be non-negative")
    return arguments


def seed_preflight(
    inverse_dir: Path,
    effective_position_tolerance_m: float,
    effective_tangent_tolerance_rad: float,
) -> dict:
    """Report feasibility against the tolerances the optimizer really sees.

    ``--auto-tolerance`` and the feasibility check both use only the FEASIBLE
    inverse nodes.  A partial inverse result (Layer 1 stopped before the path
    end) carries a large-error trailing node that the global layer discards in
    ``refine_complete`` mode anyway; letting it drive the tolerance would slack
    the whole solve to a number no real node needs.
    """
    result = load_inverse_path_result(inverse_dir)
    position = np.asarray(
        [node.position_error_m for node in result.nodes], dtype=float
    )
    tangent = np.asarray(
        [node.tangent_error_rad for node in result.nodes], dtype=float
    )
    node_flags = np.asarray([bool(node.feasible) for node in result.nodes])
    feasible_position = position[node_flags] if np.any(node_flags) else position
    feasible_tangent = tangent[node_flags] if np.any(node_flags) else tangent
    # Feasibility of the seed the optimizer actually gets: only the feasible
    # nodes, checked against the tolerance this run will enforce.
    over_position = np.flatnonzero(feasible_position > effective_position_tolerance_m)
    over_tangent = np.flatnonzero(feasible_tangent > effective_tangent_tolerance_rad)
    summary = result.summary if isinstance(result.summary, dict) else {}
    path_total = float(summary.get("path_total_length_m", 0.0) or 0.0)
    last_feasible_progress = float(
        getattr(result, "last_feasible_progress_m", float("nan"))
    )
    report = {
        "inverse_dir": str(inverse_dir),
        "nodes": int(position.size),
        "feasible_nodes": int(np.sum(node_flags)),
        "all_nodes_flagged_feasible": bool(np.all(node_flags)),
        "inverse_path_complete": bool(getattr(result, "all_nodes_feasible", np.all(node_flags))),
        "last_feasible_progress_m": last_feasible_progress,
        "path_total_length_m": path_total,
        "maximum_position_error_m": float(np.max(feasible_position)),
        "maximum_tangent_error_rad": float(np.max(feasible_tangent)),
        "effective_position_tolerance_m": float(effective_position_tolerance_m),
        "effective_tangent_tolerance_rad": float(effective_tangent_tolerance_rad),
        "nodes_over_position_tolerance": over_position.tolist(),
        "nodes_over_tangent_tolerance": over_tangent.tolist(),
        "seed_is_feasible": bool(
            over_position.size == 0 and over_tangent.size == 0
        ),
        "feasible_seed_fallback_available": bool(
            over_position.size == 0 and over_tangent.size == 0
        ),
        "suggested_position_tolerance_m": float(np.max(feasible_position)),
    }
    return report


def print_preflight(report: dict, margin: float) -> None:
    print("\n[SEED PREFLIGHT]", flush=True)
    print(
        f"  nodes                     {report['feasible_nodes']} feasible / {report['nodes']} total\n"
        f"  worst position error      {1e6 * report['maximum_position_error_m']:.3f} um  (feasible nodes)\n"
        f"  effective tolerance       {1e6 * report['effective_position_tolerance_m']:.3f} um\n"
        f"  worst tangent error       {math.degrees(report['maximum_tangent_error_rad']):.3f} deg  (feasible nodes)\n"
        f"  effective tolerance       {math.degrees(report['effective_tangent_tolerance_rad']):.3f} deg",
        flush=True,
    )
    if not report.get("inverse_path_complete", True):
        progress = report.get("last_feasible_progress_m", float("nan"))
        total = report.get("path_total_length_m", 0.0)
        pct = 100.0 * progress / total if total > 0 else float("nan")
        print(
            "\n  *** PARTIAL INVERSE PATH ***\n"
            f"  Layer 1 stopped at s = {1e3 * progress:.3f} mm of "
            f"{1e3 * total:.3f} mm  ({pct:.1f}% of the centreline).\n"
            "  In refine_complete mode the global layer refines ONLY this "
            "feasible prefix -- it will not extend the path.\n"
            "  The output will end where Layer 1 ended. To go further, fix "
            "Layer 1 (geometry / tolerances / actuation) or run --recover-partial\n"
            "  to attempt a slack-based extension (usually futile when the "
            "prefix ended on a rank-deficient / ill-conditioned Jacobian).",
            flush=True,
        )
    if report["seed_is_feasible"]:
        print(
            "  VERDICT  the feasible prefix meets this run's tolerance; the "
            "retained-seed fallback is armed, so a stalled or failed solve "
            "returns the inverse path, not a worse iterate.",
            flush=True,
        )
        return
    over_position = report["nodes_over_position_tolerance"]
    over_tangent = report["nodes_over_tangent_tolerance"]
    print(
        f"  VERDICT  feasible-prefix nodes over the *spent* tolerance: "
        f"position={len(over_position)} (first: {over_position[:5]}), "
        f"tangent={len(over_tangent)} (first: {over_tangent[:5]}).\n"
        "           The retained-seed fallback still arms whenever the seed "
        "meets the PHYSICAL tolerance (see\n"
        "           preserve_feasible_seed_against_physical_tolerance); it is "
        "disarmed only if the seed misses the physical tolerance too, in\n"
        "           which case feasibility restoration runs and may move the "
        "tip. To use what Layer 1 achieved as the envelope, add\n"
        f"           --auto-tolerance ({1e6 * report['maximum_position_error_m'] * margin:.3f} um).",
        flush=True,
    )


def main() -> None:
    args = _arguments()
    faulthandler.enable()
    if args.trace_after > 0:
        faulthandler.dump_traceback_later(
            args.trace_after, repeat=True, file=sys.stderr
        )

    _, bundle, controller_pack, out_root = build_planning_context()
    planner_config = make_inverse_config()
    inverse_dir = args.inverse_dir or (out_root / "offline_inverse_configuration_60")
    output_dir = Path(args.output_dir or (out_root / "global_configuration_converged"))

    position_tolerance = float(planner_config.position_tolerance_m)
    tangent_tolerance = float(planner_config.tangent_tolerance_rad)
    spend_fraction = float(args.tolerance_spend_fraction)

    initial_report = seed_preflight(
        inverse_dir,
        position_tolerance * spend_fraction,
        tangent_tolerance * spend_fraction,
    )
    if args.auto_tolerance:
        # Set the *effective* optimizer envelope to the achieved Layer-1 error
        # times the requested margin.  Divide by the spend fraction because the
        # global optimizer applies that fraction once more internally.
        position_tolerance = float(
            initial_report["maximum_position_error_m"]
            * args.auto_tolerance_margin
            / spend_fraction
        )
        tangent_tolerance = float(
            initial_report["maximum_tangent_error_rad"]
            * args.auto_tolerance_margin
            / spend_fraction
        )
        if tangent_tolerance >= math.pi:
            raise SystemExit(
                "--auto-tolerance with this spend fraction would require a "
                "nominal tangent tolerance >= 180 deg. Increase "
                "--tolerance-spend-fraction."
            )
        print(
            "\n[AUTO TOLERANCE] effective optimizer envelope set from Layer 1:\n"
            f"  position {1e6 * position_tolerance * spend_fraction:.3f} um\n"
            f"  tangent  {math.degrees(tangent_tolerance * spend_fraction):.3f} deg",
            flush=True,
        )
    report = seed_preflight(
        inverse_dir,
        position_tolerance * spend_fraction,
        tangent_tolerance * spend_fraction,
    )
    report["nominal_position_tolerance_m"] = position_tolerance
    report["nominal_tangent_tolerance_rad"] = tangent_tolerance
    report["tolerance_spend_fraction"] = spend_fraction
    print_preflight(report, float(args.auto_tolerance_margin))
    if not report.get("inverse_path_complete", True) and args.require_complete_inverse:
        raise SystemExit(
            "\nStopping: --require-complete-inverse is set and Layer 1 did not "
            "reach the path end (feasible to "
            f"{1e3 * report.get('last_feasible_progress_m', float('nan')):.3f} mm "
            f"of {1e3 * report.get('path_total_length_m', 0.0):.3f} mm)."
        )
    if not report["seed_is_feasible"] and not args.allow_infeasible_seed:
        raise SystemExit(
            "\nStopping: the seed is infeasible for the tolerance this run would "
            "enforce.\nUse --auto-tolerance to enforce what layer 1 achieved, or "
            "--allow-infeasible-seed to proceed anyway with the fallback disabled."
        )

    second_weight = (float(args.second_difference_weight),) * 7
    weight_report: dict = {"balanced": False}
    if args.balance_smoothing_weights:
        inverse_result = load_inverse_path_result(inverse_dir)
        s = np.asarray([node.s_m for node in inverse_result.nodes], dtype=float)
        states = np.asarray(
            [
                np.concatenate((node.q_rad, [node.insertion_m]))
                for node in inverse_result.nodes
            ],
            dtype=float,
        )
        scale = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-2)
        second_weight = balanced_second_difference_weights(
            s=s,
            states=states,
            configuration_scale=scale,
            reference_weight=float(args.second_difference_weight),
        )
        names = ("q1", "q2", "q3", "q4", "q5", "q6", "L")
        weight_report = {
            "balanced": True,
            "second_difference_weight": list(second_weight),
        }
        print("\n[SMOOTHING WEIGHTS] balanced so every coordinate pays equally:")
        for name, value in zip(names, second_weight):
            ratio = value / float(args.second_difference_weight)
            print(f"  {name:>2}  {value:.4e}   ({ratio:6.2f}x uniform)")

    config = GlobalConfigurationOptimizerConfig(
        mode="recover_partial" if args.recover_partial else "refine_complete",
        position_tolerance_m=position_tolerance,
        tangent_tolerance_rad=tangent_tolerance,
        path_step_m=3.0e-3 * float(args.path_step_mm),
        minimum_path_step_m=1.25e-4,
        dense_validation_samples_per_interval=int(args.dense_samples),
        maximum_refinement_rounds=int(args.refinement_rounds),
        maximum_nodes=int(args.maximum_nodes),
        configuration_scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0e-2),
        first_difference_weight=(float(args.first_difference_weight),) * 7,
        second_difference_weight=second_weight,
        seed_deviation_weight=(float(args.seed_deviation_weight),) * 7,
        joint_centre_weight=(1.0e-8,) * 6 + (0.0,),
        maximum_adjacent_joint_change_rad=(0.25,) * 6,
        maximum_adjacent_insertion_change_m=3.0e-3,
        insertion_non_decreasing=False,
        fix_initial_state=True,
        require_contact_model=True,
        feasibility_restoration_enabled=True,
        feasibility_restoration_iterations=200,
        maximum_iterations=int(args.maxiter),
        maximum_wall_time_s=60.0 * float(args.max_runtime_minutes),
        stagnation_function_evaluations=int(args.stagnation_evaluations),
        maximum_multistart_attempts=int(args.multistart_attempts),
        dense_validation_enabled=True,
        compute_node_jacobian_diagnostics=bool(args.node_jacobian_diagnostics),
        coarsen_complete_seed=not bool(args.preserve_all_inverse_nodes),
        keep_feasible_complete_seed=True,
        preserve_feasible_seed_on_failure=True,
        preserve_feasible_seed_against_physical_tolerance=(
            not bool(args.no_physical_tolerance_fallback)
        ),
        source_magnet_lumen_exclusion_radius_m=args.exclusion_radius,
        source_magnet_lumen_constraint_tolerance_m=float(args.exclusion_tolerance),
        dense_magnet_exclusion_acceptance_slack_m=(
            1.0e-3 * float(args.dense_exclusion_acceptance_slack_mm)
        ),
        dense_position_acceptance_slack_m=(
            1.0e-3 * float(args.dense_position_acceptance_slack_mm)
        ),
        dense_tangent_acceptance_slack_rad=(
            math.radians(float(args.dense_tangent_acceptance_slack_deg))
        ),
        require_analytical_magnet_position_jacobian=bool(
            args.require_analytical_magnet_jacobian
        ),
        tolerance_spend_fraction=spend_fraction,
        tip_centring_weight=float(args.tip_centring_weight),
        tip_curvature_weight=float(args.tip_curvature_weight),
        magnet_path_weight=float(args.magnet_path_weight),
        magnet_curvature_weight=float(args.magnet_curvature_weight),
        debug=not args.quiet,
        trust_constr_verbose=0 if args.quiet else 1,
    )

    print(
        "\n[GLOBAL RUN] "
        f"maxiter={config.maximum_iterations} "
        f"refinement_rounds={config.maximum_refinement_rounds} "
        f"dense_samples={config.dense_validation_samples_per_interval} "
        f"path_step={1e3 * config.path_step_m:.3f} mm\n"
        f"             keep_out={'off' if args.exclusion_radius is None else f'{args.exclusion_radius:.4f} m'} "
        f"spend_fraction={config.tolerance_spend_fraction:.3f} "
        f"tip_centring_weight={config.tip_centring_weight:g} "
        f"tip_curvature_weight={config.tip_curvature_weight:g}\n"
        f"             magnet_path_weight={config.magnet_path_weight:g} "
        f"magnet_curvature_weight={config.magnet_curvature_weight:g}\n"
        f"             wall_limit={args.max_runtime_minutes:.1f} min "
        f"stagnation_evals={config.stagnation_function_evaluations} "
        f"node_jacobian_diagnostics={config.compute_node_jacobian_diagnostics}\n"
        f"             output={output_dir}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        result = optimize_from_saved_inverse_result(
            inverse_output_dir=inverse_dir,
            controller_pack=controller_pack,
            lumen_C=bundle.lumen_C,
            config=config,
            output_dir=output_dir,
        )
    finally:
        if args.trace_after > 0:
            faulthandler.cancel_dump_traceback_later()
    elapsed = time.perf_counter() - started

    summary = result.summary
    exclusion = summary.get("source_magnet_lumen_exclusion", {})
    print(f"\nElapsed: {elapsed:.3f} s")
    print(f"Optimizer success:      {result.optimizer_success}")
    print(f"Globally feasible:      {result.globally_feasible}")
    print(f"Dense validation:       {result.dense_validation_feasible}")
    print(f"Termination:            {result.termination_reason}")
    print(f"Iterations / maximum:   {result.iterations} / {config.maximum_iterations}")
    print(f"Refinement rounds:      {result.refinement_rounds} / {config.maximum_refinement_rounds}")
    print(f"Constraint violation:   {summary.get('normalized_constraint_violation')}")
    print(f"Seed fallback selected: {summary.get('selected_seed_fallback')}")
    print(f"Fell back to inverse:   {summary.get('fallback_to_inverse_path')}")
    print(f"Any feasible round:     {summary.get('any_feasible_global_round')}")
    print(
        "Maximum normalized change: "
        f"{summary.get('maximum_normalized_state_change_from_seed')}"
    )
    print(f"Maximum position error: {1e3 * result.maximum_position_error_m:.6f} mm")
    print(f"Maximum tangent error:  {math.degrees(result.maximum_tangent_error_rad):.4f} deg")
    if exclusion.get("enabled"):
        print(
            f"Keep-out minimum margin: "
            f"{1e3 * exclusion.get('minimum_margin_m', float('nan')):.6f} mm "
            f"(satisfied={exclusion.get('all_nodes_satisfied')})"
        )

    (output_dir / "run_preflight.json").write_text(
        json.dumps(
            {
                "seed_preflight": report,
                "smoothing_weights": weight_report,
                "elapsed_s": elapsed,
                "hit_iteration_cap": bool(
                    result.iterations >= config.maximum_iterations
                ),
                "globally_feasible": bool(result.globally_feasible),
                "fallback_to_inverse_path": bool(
                    summary.get("fallback_to_inverse_path")
                ),
                "selected_seed_fallback": bool(
                    summary.get("selected_seed_fallback")
                ),
                "any_feasible_global_round": bool(
                    summary.get("any_feasible_global_round")
                ),
                "termination_reason": str(result.termination_reason),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if result.iterations >= config.maximum_iterations:
        print(
            "\nWARNING: the solve used every permitted iteration. What was "
            "written out is an intermediate iterate, not a converged solution. "
            "Raise --maxiter before drawing conclusions from it.",
            flush=True,
        )

    if summary.get("fallback_to_inverse_path"):
        print(
            "\nNOTE: no global refinement round produced a feasible path, so "
            "the written-out path IS the exact inverse configuration path, "
            "unchanged. Layer 2 never returns something worse than Layer 1. "
            "The tip may still be jagged (that is Layer 1's path); to actually "
            "smooth it, loosen what the optimiser must hit -- "
            "--tolerance-spend-fraction below 1, a larger --auto-tolerance-"
            "margin, fewer/looser hard constraints -- so a round can land "
            "feasible.",
            flush=True,
        )
    elif summary.get("selected_seed_fallback"):
        print(
            "\nNOTE: the global solve did not beat the inverse seed, so the "
            "written-out path IS the inverse configuration path (its best "
            "feasible prefix), not an optimised one. This is the intended "
            "fallback: Layer 2 never returns something worse than Layer 1.",
            flush=True,
        )
    elif not result.globally_feasible:
        print(
            "\nWARNING: the result is NOT globally feasible even though no "
            "fallback fired -- this should not happen; please report the "
            "run_preflight.json and global_configuration_summary.json.",
            flush=True,
        )

    if not report.get("inverse_path_complete", True):
        progress = report.get("last_feasible_progress_m", float("nan"))
        total = report.get("path_total_length_m", 0.0)
        print(
            f"\nREMINDER: Layer 1 was partial ({1e3 * progress:.3f} / "
            f"{1e3 * total:.3f} mm). This global output ends there by design.",
            flush=True,
        )


if __name__ == "__main__":
    main()
