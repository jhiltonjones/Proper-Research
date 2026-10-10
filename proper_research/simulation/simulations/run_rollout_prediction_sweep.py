#!/usr/bin/env python3
"""
Rollout × prediction-horizon sweep launcher.

This is your original grid loop rewritten to make the experimental design explicit
and easier to analyse later.

Important design point:
- If your MPC implementation requires prediction horizon >= rollout, keep --valid-only.
- To obtain stronger evidence for horizon effects at every rollout, extend Np beyond 6
  (for example --pred-max 12). With Np only up to 6, rollout=6 has only one horizon
  condition, so it cannot show a within-rollout horizon effect.

Usage examples:
    python run_rollout_prediction_sweep.py
    python run_rollout_prediction_sweep.py --pred-max 12
    python run_rollout_prediction_sweep.py --full-factorial  # only if Np < rollout is valid in your code
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Import your actual function here. Adjust this import to match your project.
from proper_research.simulation.simulations.run_experiment_grid import run_experiment_grid


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
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest but do not run experiments.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
                }
            )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(planned_rows[0].keys()))
        writer.writeheader()
        writer.writerows(planned_rows)

    print(f"Planned {len(planned_rows)} rollout×prediction-horizon cells")
    print(f"Manifest: {manifest_path}")

    if args.dry_run:
        return

    for row in planned_rows:
        roll = row["rollout_steps"]
        pred = row["Np"]
        run_root = Path(row["run_root"])
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
