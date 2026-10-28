#!/usr/bin/env python3
"""Run one nominal Cosserat magnetic-beam solve."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cosserat_clean.config import ExperimentConfig
from cosserat_clean.experiment import build_context, make_run_directory, run_nominal
from cosserat_clean.external import MissingExternalDependency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=120, help="Initial BVP mesh nodes")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output root")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Run only the nonlinear forward solve",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig().with_nodes(args.nodes).with_output_root(args.output)
    try:
        context = build_context(config)
        result = run_nominal(context, compute_sensitivity=not args.skip_sensitivity)
    except MissingExternalDependency as exc:
        print(f"Dependency error: {exc}")
        return 2

    output = make_run_directory(args.output, "nominal")
    np.savez(
        output / "nominal_result.npz",
        tip_position=result["tip_position"],
        source_position=result["source_position"],
        source_dipole=result["source_dipole"],
        field_tip=result["field_tip"],
        net_force=result["net_force"],
        net_torque=result["net_torque"],
        tip_jacobian=result.get("tip_jacobian", np.empty((0, 0))),
    )

    print(f"Placement: {result['placement_label']}")
    print(f"Tip position [m]: {np.asarray(result['tip_position'])}")
    print(f"Tip field [T]: {np.asarray(result['field_tip'])}")
    print(f"Tip field magnitude [mT]: {1e3 * float(result['field_norm_t']):.6g}")
    print(
        "Tip bending [deg]: "
        f"y={float(result['theta_y_deg']):.6g}, "
        f"z={float(result['theta_z_deg']):.6g}, "
        f"total={float(result['theta_total_deg']):.6g}"
    )
    print(f"Net force [N]: {np.asarray(result['net_force'])}")
    print(f"Net torque [N m]: {np.asarray(result['net_torque'])}")
    print(f"Total energy [J]: {float(result['energy_j']):.6g}")
    if "tip_jacobian" in result:
        print("Tip Jacobian d p_tip / d [r_src, delta_phi, L]:")
        print(np.asarray(result["tip_jacobian"]))
        diagnostics = result["sensitivity_diagnostics"]
        print(
            "Sensitivity terminal condition number: "
            f"{float(diagnostics['terminal_condition_number']):.6g}"
        )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
