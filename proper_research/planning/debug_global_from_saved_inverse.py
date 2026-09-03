"""Diagnose or run global optimization from a saved inverse path.

The default mode is deliberately a seed audit.  It does not enter the global
optimizer, compute beam Jacobians, or run dense between-node validation.  It
establishes whether the saved inverse states remain feasible when their beam
outputs are recomputed by the global-planner context.

After the seed audit reports node feasibility, pass ``--full`` to enable dense
validation and a multi-iteration global solve.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time

from proper_research.planning.global_constrained_configuration_path import (
    GlobalConfigurationOptimizerConfig,
    audit_inverse_seed_round_trip,
    load_inverse_path_result,
    optimize_from_saved_inverse_result,
)
from proper_research.planning.planning_context import (
    build_planning_context,
    make_inverse_config,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or optimize a saved inverse configuration path."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run global optimization with dense validation.",
    )
    parser.add_argument(
        "--commit-forward-audit",
        action="store_true",
        help=(
            "Audit saved nodes sequentially with commit=True. This diagnoses "
            "equilibrium continuation and cannot be combined with --full."
        ),
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum trust-constr iterations in --full mode (default: 20).",
    )
    parser.add_argument(
        "--dense-samples",
        type=int,
        default=1,
        help="Dense samples per interval in --full mode (default: 1).",
    )
    parser.add_argument(
        "--refinement-rounds",
        type=int,
        default=5,
        help="Adaptive refinement rounds in --full mode (default: 0).",
    )
    parser.add_argument(
        "--jacobian-diagnostics",
        action="store_true",
        help="Compute expensive final per-node Jacobian condition diagnostics.",
    )
    parser.add_argument(
        "--trace-after",
        type=int,
        default=120,
        help="Seconds before repeating faulthandler stack reports.",
    )
    arguments = parser.parse_args()
    if arguments.maxiter < 1:
        parser.error("--maxiter must be at least 1")
    if arguments.dense_samples < 1:
        parser.error("--dense-samples must be at least 1")
    if arguments.refinement_rounds < 0:
        parser.error("--refinement-rounds cannot be negative")
    if arguments.trace_after < 1:
        parser.error("--trace-after must be at least 1")
    if arguments.full and arguments.commit_forward_audit:
        parser.error("--commit-forward-audit cannot be combined with --full")
    return arguments


def _make_global_config(
    *,
    planner_config,
    full: bool,
    maximum_iterations: int,
    dense_samples: int,
    refinement_rounds: int,
    jacobian_diagnostics: bool,
) -> GlobalConfigurationOptimizerConfig:
    return GlobalConfigurationOptimizerConfig(
        mode="refine_complete",
        position_tolerance_m=planner_config.position_tolerance_m,
        tangent_tolerance_rad=planner_config.tangent_tolerance_rad,
        # A complete inverse seed retains every saved feasible coordinate.
        # This setting can add missing coordinates but does not subsample it.
        path_step_m=5.0e-4,
        minimum_path_step_m=2.5e-4,
        dense_validation_samples_per_interval=dense_samples,
        maximum_refinement_rounds=(refinement_rounds if full else 0),
        maximum_nodes=200,
        configuration_scale=(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0e-2,
        ),
        first_difference_weight=(1.0e-5,) * 7,
        second_difference_weight=(1.0e-8,) * 7,
        seed_deviation_weight=(1.0e-5,) * 7,
        joint_centre_weight=(
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            0.0,
        ),
        maximum_adjacent_joint_change_rad=(0.25,) * 6,
        maximum_adjacent_insertion_change_m=3.0e-3,
        insertion_non_decreasing=False,
        fix_initial_state=True,
        require_contact_model=True,
        feasibility_restoration_enabled=True,
        feasibility_restoration_iterations=50,
        feasibility_restoration_maximum_slack=1.0e3,
        feasibility_restoration_slack_weight=1.0e4,
        maximum_iterations=(maximum_iterations if full else 1),
        maximum_multistart_attempts=1,
        # New diagnostic/safety controls in the revised global module.
        dense_validation_enabled=full,
        compute_node_jacobian_diagnostics=(
            bool(jacobian_diagnostics) if full else False
        ),
        keep_feasible_complete_seed=True,
        preserve_feasible_seed_on_failure=True,
        debug=True,
        trust_constr_verbose=3,
    )


def main() -> None:
    arguments = _arguments()
    faulthandler.enable()
    faulthandler.dump_traceback_later(
        arguments.trace_after,
        repeat=True,
        file=sys.stderr,
    )

    _, bundle, controller_pack, out_root = build_planning_context()
    planner_config = make_inverse_config()
    global_config = _make_global_config(
        planner_config=planner_config,
        full=arguments.full,
        maximum_iterations=arguments.maxiter,
        dense_samples=arguments.dense_samples,
        refinement_rounds=arguments.refinement_rounds,
        jacobian_diagnostics=arguments.jacobian_diagnostics,
    )

    inverse_output_dir = out_root / "offline_inverse_configuration_60"
    output_name = (
        "global_configuration_full_debug"
        if arguments.full
        else "global_configuration_seed_audit"
    )
    driver_mode = (
        "full"
        if arguments.full
        else "committed_seed_audit"
        if arguments.commit_forward_audit
        else "seed_audit"
    )

    print(
        "[DEBUG DRIVER] "
        f"mode={driver_mode} "
        f"inverse_directory={inverse_output_dir}",
        flush=True,
    )
    started = time.perf_counter()
    try:
        if arguments.full:
            result = optimize_from_saved_inverse_result(
                inverse_output_dir=inverse_output_dir,
                controller_pack=controller_pack,
                lumen_C=bundle.lumen_C,
                config=global_config,
                output_dir=out_root / output_name,
            )
            audit_report = None
        else:
            inverse_result = load_inverse_path_result(inverse_output_dir)
            audit_report = audit_inverse_seed_round_trip(
                inverse_result=inverse_result,
                controller_pack=controller_pack,
                lumen_C=bundle.lumen_C,
                config=global_config,
                commit_forward=arguments.commit_forward_audit,
            )
            result = None
    finally:
        faulthandler.cancel_dump_traceback_later()

    elapsed = time.perf_counter() - started
    if audit_report is not None:
        print(f"Elapsed: {elapsed:.3f} seconds")
        print(f"Recomputed node feasibility: {audit_report['node_feasible']}")
        print(
            "Saved maximum position error: "
            f"{1.0e3 * audit_report['saved_maximum_position_error_m']:.4f} mm"
        )
        print(
            "Recomputed maximum position error: "
            f"{1.0e3 * audit_report['recomputed_maximum_position_error_m']:.4f} mm"
        )
        print(
            "Saved-to-recomputed tip drift: "
            f"{1.0e3 * audit_report['maximum_saved_tip_position_drift_m']:.4f} mm"
        )
        if audit_report["node_feasible"]:
            print("Seed round trip passed. Run again with --full.")
        else:
            print(
                "Seed round trip failed before global optimization. "
                "Do not tune trust-constr yet; inspect beam equilibrium "
                "initialization/state and reference reconstruction."
            )
        return

    assert result is not None
    nodes_feasible = bool(result.nodes) and all(
        node.feasible for node in result.nodes
    )
    print(f"Elapsed: {elapsed:.3f} seconds")
    print(f"Node feasibility: {nodes_feasible}")
    print(f"Optimizer success: {result.optimizer_success}")
    print(f"Dense validation: {result.dense_validation_feasible}")
    print(f"Global feasibility: {result.globally_feasible}")
    print(f"Termination: {result.termination_reason}")
    print(
        "Maximum position error: "
        f"{1.0e3 * result.maximum_position_error_m:.4f} mm"
    )
    print(
        "Maximum tangent error: "
        f"{result.maximum_tangent_error_rad * 180.0 / 3.141592653589793:.4f} deg"
    )


if __name__ == "__main__":
    main()
