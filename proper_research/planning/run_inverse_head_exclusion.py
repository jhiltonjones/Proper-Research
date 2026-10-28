# run_inverse_once.py

from __future__ import annotations

# Corrected runner: upgrades the legacy configuration object returned by
# planning_context before applying source-magnet head-exclusion settings.
RUNNER_IMPLEMENTATION_TAG = "head_exclusion_config_upgrade_v2"

import argparse
import json
import time
from dataclasses import fields, replace


def _upgrade_planner_config(
    legacy_config: object,
    target_config_type: type,
) -> object:
    """Copy shared fields from an older planner-config dataclass.

    ``planning_context.make_inverse_config()`` may still construct the legacy
    class from ``offline_inverse_configuration``.  Calling ``replace`` on that
    instance cannot add fields introduced by the head-exclusion planner, so we
    explicitly rebuild the configuration using the new class.
    """
    shared_values = {
        field.name: getattr(legacy_config, field.name)
        for field in fields(target_config_type)
        if hasattr(legacy_config, field.name)
    }
    return target_config_type(**shared_values)

def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline inverse configuration planner, optionally with "
            "a hard source-magnet exclusion region around the lumen centreline."
        )
    )
    parser.add_argument(
        "--source-magnet-lumen-exclusion-radius-m",
        type=float,
        default=None,
        help=(
            "minimum permitted source-magnet-centre distance from the closest "
            "lumen segment [m]. Use head radius + magnet collision radius + "
            "safety gap. Supplying this enables hard collision avoidance; no "
            "scientific default is assumed."
        ),
    )
    parser.add_argument(
        "--require-analytical-magnet-jacobian",
        action="store_true",
        help=(
            "fail if the adapter does not expose an analytical 3x7 (or 3x6) "
            "source-magnet position Jacobian; otherwise bound-safe central "
            "differences of magnet FK are used for this constraint only"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()

    # Keep repository imports after argument parsing so ``--help`` can be
    # checked even outside the research environment.  Import the deliberately
    # renamed exclusion implementation, not the superseded containment file.
    from proper_research.planning.global_constrained_configuration_path import (
        GlobalConfigurationOptimizerConfig,
        audit_inverse_seed_round_trip,
    )
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        IMPLEMENTATION_TAG,
        InverseConfigurationPlannerConfig,
        solve_from_controller_pack,
    )
    from proper_research.planning.planning_context import (
        build_planning_context,
        make_inverse_config,
    )

    if IMPLEMENTATION_TAG != "source_magnet_head_exclusion_v1":
        raise RuntimeError(
            "The runner imported an unexpected inverse-planner implementation."
        )
    _, bundle, controller_pack, out_root = (
        build_planning_context()
    )
    planner_config = _upgrade_planner_config(
        make_inverse_config(),
        InverseConfigurationPlannerConfig,
    )
    exclusion_radius = arguments.source_magnet_lumen_exclusion_radius_m
    if exclusion_radius is not None:
        if exclusion_radius <= 0.0:
            raise ValueError(
                "--source-magnet-lumen-exclusion-radius-m must be positive."
            )
        planner_config = replace(
            planner_config,
            source_magnet_lumen_exclusion_radius_m=exclusion_radius,
            require_analytical_magnet_position_jacobian=(
                arguments.require_analytical_magnet_jacobian
            ),
        )
        print(
            "[SAME-PROCESS TEST] Hard EPM/head exclusion enabled: "
            f"distance >= {exclusion_radius:.6f} m",
            flush=True,
        )
    elif arguments.require_analytical_magnet_jacobian:
        raise ValueError(
            "--require-analytical-magnet-jacobian only applies when "
            "--source-magnet-lumen-exclusion-radius-m is supplied."
        )
    else:
        print(
            "[SAME-PROCESS TEST] EPM/head exclusion constraint disabled; pass "
            "--source-magnet-lumen-exclusion-radius-m to enable it.",
            flush=True,
        )

    # Keep this diagnostic run separate from the original saved result.
    inverse_output_dir = (
        out_root
        / "offline_inverse_configuration_60"
    )

    print("[SAME-PROCESS TEST] Running inverse planner...", flush=True)
    inverse_started = time.perf_counter()

    inverse_result = solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=bundle.lumen_C,
        config=planner_config,
        output_dir=inverse_output_dir,
    )

    inverse_elapsed = time.perf_counter() - inverse_started

    print(
        "[SAME-PROCESS TEST] "
        f"Inverse completed in {inverse_elapsed:.3f} seconds",
        flush=True,
    )
    print(
        "[SAME-PROCESS TEST] "
        f"all_nodes_feasible={inverse_result.all_nodes_feasible}",
        flush=True,
    )

    # This configuration is only used to audit the inverse seed.
    # It does not run trust-constr or dense validation.
    global_config = GlobalConfigurationOptimizerConfig(
        mode=(
            "refine_complete"
            if inverse_result.all_nodes_feasible
            else "recover_partial"
        ),
        position_tolerance_m=(
            planner_config.position_tolerance_m
        ),
        tangent_tolerance_rad=(
            planner_config.tangent_tolerance_rad
        ),
        path_step_m=3.0e-3,
        minimum_path_step_m=2e-3,
        maximum_refinement_rounds=0,
        maximum_nodes=200,
        maximum_adjacent_joint_change_rad=(0.25,) * 6,
        maximum_adjacent_insertion_change_m=3.0e-3,
        insertion_non_decreasing=False,
        fix_initial_state=True,
        require_contact_model=True,
        maximum_iterations=1,
        maximum_multistart_attempts=1,
        feasibility_restoration_enabled=False,
        feasibility_restoration_iterations=1,
        dense_validation_enabled=False,
        compute_node_jacobian_diagnostics=False,
        debug=True,
    )

    print(
        "[SAME-PROCESS TEST] "
        "Auditing the in-memory inverse result immediately...",
        flush=True,
    )
    audit_started = time.perf_counter()

    # Important: test commit=False first. Do not perform another forward
    # evaluation between the inverse solve and this audit.
    audit = audit_inverse_seed_round_trip(
        inverse_result=inverse_result,
        controller_pack=controller_pack,
        lumen_C=bundle.lumen_C,
        config=global_config,
        commit_forward=False,
    )

    audit_elapsed = time.perf_counter() - audit_started

    print(
        f"[SAME-PROCESS TEST] Audit completed in "
        f"{audit_elapsed:.3f} seconds"
    )
    print(json.dumps(audit, indent=2))

    print(f"Inverse result saved to: {inverse_output_dir}")
    print(
        "Same-process round trip feasible: "
        f"{audit['node_feasible']}"
    )
    print(
        "Maximum saved-to-recomputed tip drift: "
        f"{1.0e3 * audit['maximum_saved_tip_position_drift_m']:.4f} mm"
    )

    if audit["node_feasible"]:
        print(
            "RESULT: The same-process result is reproducible. "
            "The failure after reconstruction indicates that internal "
            "beam equilibrium state is not being saved/restored."
        )
    else:
        print(
            "RESULT: The result is not reproducible even in the same "
            "process. Inspect how the inverse solver accepts a candidate "
            "and stores its configuration, output, and beam state."
        )


if __name__ == "__main__":
    main()
