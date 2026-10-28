#!/usr/bin/env python3
"""
Rollout × prediction-horizon sweep launcher.

This script assumes your actual run_experiment_grid function already exists
somewhere in your project. Provide its module with --runner-module.

Example:
    python -m proper_research.simulation.simulations.run_rollout_prediction_sweep_fixed \
        --root evidence_contact_sqp_budget_sweep \
        --runner-module proper_research.simulation.simulations.YOUR_ORIGINAL_SWEEP_SCRIPT \
        --pred-max 12 \
        --skip-existing

Replace YOUR_ORIGINAL_SWEEP_SCRIPT with the Python file, without ".py", that defines:
    run_experiment_grid(...)
"""
from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=Path, default=Path("evidence_contact_sqp_budget_sweep"))

    parser.add_argument("--roll-min", type=int, default=1)
    parser.add_argument("--roll-max", type=int, default=6)

    parser.add_argument("--pred-min", type=int, default=1)
    parser.add_argument("--pred-max", type=int, default=12)

    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="Run all rollout×Np combinations. Use only if Np < rollout is meaningful/valid.",
    )

    parser.add_argument("--n-sqp", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=45)

    parser.add_argument(
        "--runner-module",
        type=str,
        required=True,
        help=(
            "Import path of the module containing run_experiment_grid, e.g. "
            "proper_research.simulation.simulations.contact_sqp_budget_sweep"
        ),
    )
    parser.add_argument("--runner-name", type=str, default="run_experiment_grid")

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a rollout/Np cell if its run_root already contains grid_summary.csv. "
            "This avoids overwriting completed cells."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing cells to be rerun. Use carefully.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest but do not run experiments.")

    return parser.parse_args()


def load_runner(module_name: str, function_name: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"\nCould not import runner module:\n"
            f"  {module_name}\n\n"
            f"Check that you are running from the repository root, and that the module path is correct.\n"
            f"For a file at proper_research/simulation/simulations/my_script.py, use:\n"
            f"  --runner-module proper_research.simulation.simulations.my_script\n"
        ) from exc

    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise SystemExit(
            f"\nImported module, but it does not define {function_name!r}:\n"
            f"  {module_name}\n\n"
            f"Open that file and confirm the function name, or pass --runner-name."
        ) from exc


def main() -> None:
    args = parse_args()

    if args.skip_existing and args.overwrite:
        raise SystemExit("Choose either --skip-existing or --overwrite, not both.")

    run_experiment_grid = load_runner(args.runner_module, args.runner_name)

    rollout_values = np.arange(args.roll_min, args.roll_max + 1)
    prediction_horizon_values = np.arange(args.pred_min, args.pred_max + 1)

    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "sweep_manifest.csv"

    planned_rows = []
    for roll in rollout_values:
        for pred in prediction_horizon_values:
            if (not args.full_factorial) and pred < roll:
                continue

            run_root = args.root / f"roll_{roll}" / f"pred_{pred}"

            planned_rows.append(
                {
                    "rollout_steps": int(roll),
                    "Np": int(pred),
                    "extra_horizon": int(pred - roll),
                    "N_sqp": int(args.n_sqp),
                    "run_root": str(run_root),
                    "grid_summary_exists": str((run_root / "grid_summary.csv").exists()),
                }
            )

    if not planned_rows:
        raise SystemExit("No cells planned. Check roll/pred ranges and --full-factorial.")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(planned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planned_rows)

    print(f"Planned {len(planned_rows)} rollout×prediction-horizon cells")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        print("Dry run only. No experiments were executed.")
        return

    for row in planned_rows:
        roll = row["rollout_steps"]
        pred = row["Np"]
        run_root = Path(row["run_root"])
        summary_path = run_root / "grid_summary.csv"

        if summary_path.exists() and args.skip_existing:
            print(f"Skipping existing rollout={roll}, Np={pred}: {summary_path}")
            continue

        if summary_path.exists() and not args.overwrite:
            raise SystemExit(
                f"\nRefusing to overwrite existing cell:\n"
                f"  {run_root}\n\n"
                f"Use --skip-existing to continue past completed cells, or --overwrite to rerun them."
            )

        print(f"Running rollout={roll}, Np={pred}, extra_horizon={pred-roll}: {run_root}")

        run_experiment_grid(
            run_root=run_root,
            bend_angles_deg=(-30.0, -45.0, -60.0, -75.0, -90.0),
            jacobian_variants=("contact",),
            solver_modes=("sqp_full",),
            rollout_steps_values=(roll,),
            Np=pred,
            N_sqp=args.n_sqp,
            max_steps=args.max_steps,
            plant_contact=True,
            save_plots=False,
            adaptive_rollout_enabled=False,
            hierarchical_mpc_enabled=False,
            hierarchical_policy_config=None,
            stop_on_failure=False,
        )


if __name__ == "__main__":
    main()
