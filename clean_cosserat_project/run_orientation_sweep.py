#!/usr/bin/env python3
"""Run the placement-versus-dipole-world-z-rotation comparison sweep."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from cosserat_clean.config import ExperimentConfig
from cosserat_clean.experiment import (
    build_context,
    quick_config,
    run_orientation_sweep,
)
from cosserat_clean.external import MissingExternalDependency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=120, help="Initial BVP mesh nodes")
    parser.add_argument("--angles", type=int, default=101, help="Number of rotation angles")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output root")
    parser.add_argument("--quick", action="store_true", help="Use 9 angles and at most 30 nodes")
    parser.add_argument("--no-plots", action="store_true", help="Do not create PNG plots")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Skip the analytic shooting sensitivity at every sweep point",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots as well as saving")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExperimentConfig().with_nodes(args.nodes).with_output_root(args.output)
    config = replace(
        config,
        sweep=replace(
            config.sweep,
            angle_count=args.angles,
            make_plots=not args.no_plots,
            compute_sensitivity=not args.skip_sensitivity,
        ),
    )
    if args.quick:
        config = quick_config(config)

    try:
        context = build_context(config)
        result = run_orientation_sweep(context, show_plots=args.show_plots)
    except MissingExternalDependency as exc:
        print(f"Dependency error: {exc}")
        return 2

    successful = int((result.table["status"] == "ok").sum())
    failed = int((result.table["status"] != "ok").sum())
    print(f"Completed points: {successful}")
    print(f"Failed points: {failed}")
    print(f"Saved: {result.output_directory}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
