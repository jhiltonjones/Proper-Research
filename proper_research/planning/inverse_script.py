# run_inverse_once.py

from __future__ import annotations

import json
import time

from proper_research.planning.planning_context import (
    build_planning_context,
    make_inverse_config,
)
from proper_research.planning.offline_inverse_configuration import (
    solve_from_controller_pack,
)
from proper_research.planning.global_constrained_configuration_path import (
    GlobalConfigurationOptimizerConfig,
    audit_inverse_seed_round_trip,
)


def main() -> None:
    _, bundle, controller_pack, out_root = (
        build_planning_context()
    )
    planner_config = make_inverse_config()

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
        path_step_m=5.0e-4,
        minimum_path_step_m=2.5e-4,
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