"""Test the four controllers (naive resolved-rate, MPC LTI, MPC LTV-offline,
MPC LTV-SQP-online) against each other, closed loop, on the real
contact-enabled plant.

Reusable/geometry-pinned, like the other scripts in this module (see
``pinned_planning_context.py``). Unlike ``clip_ratio_study.py`` (which only
needs the naive controller, so it skips building the MPC variants entirely for
speed) this one needs all four, so it uses ``run_contact_study``'s normal
``build_cell_environment`` -- but only ONCE per (radius, Jacobian model) cell,
then reuses that one environment to run every requested controller, since the
expensive part (the offline Jacobian schedule + the three MPC variant builds)
does not depend on which controller you ask ``env["make"]`` for afterward.

Outputs (into --output-dir)
----------------------------
controller_comparison.json        one record per (radius, jacobian_model, controller)
controller_comparison_steps.csv   per-step long table: cell, radius_mm,
                                   jacobian_model, controller, k, arc_mm,
                                   position_error_mm, tangent_error_deg,
                                   solve_time_ms, qp_success

Usage
-----
    python -m proper_research.experiments.controller_comparison_study \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --radius-scales 1.0 0.70 0.40 --jacobian-models contact \\
        --controllers naive_inverse_jacobian mpc_lti mpc_ltv_offline mpc_ltv_sqp_online \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5

    python -m proper_research.experiments.plot_controller_comparison \\
        --study-dir results/contact_study_radius_p30_m70
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent, _HERE.parent / "controllers"):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from proper_research.experiments import run_contact_study as RCS  # noqa: E402
from proper_research.experiments.pinned_planning_context import (  # noqa: E402
    add_geometry_arguments,
    resolve_planning_context_from_args,
    resolve_reference_dir,
)

DEFAULT_CONTROLLERS = (
    "naive_inverse_jacobian", "mpc_lti", "mpc_ltv_offline", "mpc_ltv_sqp_online",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all four controllers closed-loop against each other, "
                    "reusable across vessels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--radius-scales", type=float, nargs="+", default=(1.0, 0.70, 0.40))
    parser.add_argument("--jacobian-models", nargs="+", default=("contact",),
                         choices=("contact", "no_contact"))
    parser.add_argument("--controllers", nargs="+", default=list(DEFAULT_CONTROLLERS),
                         choices=list(DEFAULT_CONTROLLERS))
    parser.add_argument("--max-steps", type=int, default=None,
                         help="Omit for the full reference length.")
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--tag", default="")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing controller_comparison{tag}.json "
                              "and recompute every cell. Default: resume -- keep "
                              "cells already in that file and only run the missing ones.")
    add_geometry_arguments(parser)
    return parser.parse_args()


def _load_existing(json_path: Path, csv_path: Path):
    """Return ``(results, step_rows, done_cells)`` from a prior (partial) run."""
    if not json_path.is_file():
        return [], [], set()
    try:
        results = json.loads(json_path.read_text())
    except (ValueError, OSError):
        return [], [], set()
    if not isinstance(results, list):
        return [], [], set()
    step_rows: list[dict[str, Any]] = []
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            step_rows = list(csv.DictReader(handle))
    done_cells = {r.get("cell") for r in results if r.get("cell")}
    return results, step_rows, done_cells


def main() -> int:
    args = _arguments()
    tag = f"_{args.tag}" if args.tag else ""

    harness = RCS._load("compare_controllers")
    providers_module = RCS._load("beam_jacobian_providers")
    variants_module = RCS._load("mpc_variants")
    inverse_module = RCS._load("inverse_jacobian_controller")
    base_module = harness.load_base_module()
    beam_module = harness.load_beam_output_module()

    exp_cfg, bundle0, pack0, out_root = resolve_planning_context_from_args(args)
    RCS._assert_plant_contact(bundle0, "the baseline planning context")
    reference_dir = resolve_reference_dir(out_root, args.reference)

    base_context = RCS.resolve_planning_context(exp_cfg, pack0)
    base_context["reference_dir"] = reference_dir
    base_context.pop("resolved_from", None)

    settings = RCS.StudySettings(
        horizon=int(args.horizon), verbose=False, max_control_steps=args.max_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"controller_comparison{tag}.json"
    csv_path = args.output_dir / f"controller_comparison{tag}_steps.csv"
    if args.fresh:
        results: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        done_cells: set = set()
    else:
        results, step_rows, done_cells = _load_existing(json_path, csv_path)
        if done_cells:
            print(f"[resume] {len(done_cells)} cell(s) already in {json_path.name}; "
                  f"skipping those, running only the rest. Pass --fresh to recompute all.",
                  flush=True)

    for scale in args.radius_scales:
        name = f"radius_x{scale:g}".replace(".", "p")
        spec = RCS.AnatomySpec(name=name, radius_scale=float(scale))
        ctx = RCS.build_anatomy_context(
            spec=spec, base_context=base_context, providers_module=providers_module,
            base_module=base_module, require_planned_beam_feasible=False,
            shared_controller_pack=None,
        )
        radius_mm = round(1000.0 * float(np.mean(np.asarray(ctx.bundle.lumen_R, dtype=float))), 4)
        ic0 = np.asarray(ctx.reference.state, dtype=float)[0]
        arc_m = np.asarray(ctx.reference.path_coordinate_m, dtype=float)
        arc_mm_rel = 1.0e3 * (arc_m - float(arc_m[0]))

        for jacobian_model in args.jacobian_models:
            pending = [
                c for c in args.controllers
                if f"{name}__{jacobian_model}__{c}{tag}" not in done_cells
            ]
            if not pending:
                print(f"[resume] {name} / {jacobian_model}: all controllers done, "
                      f"skipping environment build.", flush=True)
                continue
            build_started = time.perf_counter()
            env = RCS.build_cell_environment(
                context=ctx, jacobian_model=jacobian_model, settings=settings,
                disturbance_alpha=0.0, base_module=base_module, beam_module=beam_module,
                harness=harness, variants_module=variants_module, inverse_module=inverse_module,
            )
            build_elapsed = time.perf_counter() - build_started
            print(f"[build] {name} / {jacobian_model}: environment ready in "
                  f"{build_elapsed:.1f}s (schedule + 3 MPC variants)", flush=True)

            for controller_name in pending:
                started = time.perf_counter()
                controller = env["make"](controller_name)
                res = harness.run_closed_loop(
                    controller=controller, reference=ctx.reference,
                    beam_snapshot=env["beam_snapshot"], magnet_pose=env["magnet_pose"],
                    reset_plant=env["reset_plant"], config=env["loop_config"],
                    initial_state=ic0, verbose=False,
                )
                elapsed = time.perf_counter() - started
                summary = dict(res.summary)
                summary.pop("controller", None)  # collides with our own controller_name
                summary.pop("description", None)
                cell = f"{name}__{jacobian_model}__{controller_name}{tag}"
                rec = dict(
                    cell=cell, anatomy=name, radius_mm=radius_mm,
                    jacobian_model=jacobian_model, controller=controller_name,
                    controller_description=res.summary.get("description", ""),
                    environment_build_s=build_elapsed, run_elapsed_s=elapsed,
                    **summary,
                )
                results.append(rec)
                print(json.dumps({k: v for k, v in rec.items()
                                  if k in ("cell", "rms_beam_position_error_mm",
                                           "maximum_beam_position_error_mm",
                                           "mean_solve_time_ms", "all_solves_succeeded",
                                           "stopped_reason")}), flush=True)

                for row in res.records[1:]:
                    k = int(row["simulation_sample"]) - 1
                    step_rows.append(dict(
                        cell=cell, anatomy=name, radius_mm=radius_mm,
                        jacobian_model=jacobian_model, controller=controller_name,
                        k=k, arc_mm=float(arc_mm_rel[min(k, len(arc_mm_rel) - 1)]),
                        position_error_mm=row.get("beam_position_error_mm"),
                        tangent_error_deg=row.get("beam_tangent_error_deg"),
                        solve_time_ms=1.0e3 * row.get("qp_solve_time_s", 0.0),
                        qp_success=row.get("qp_success"),
                    ))

                (args.output_dir / f"controller_comparison{tag}.json").write_text(
                    json.dumps(results, indent=2)
                )
                if step_rows:
                    header = list(step_rows[0].keys())
                    with (args.output_dir / f"controller_comparison{tag}_steps.csv").open(
                        "w", newline=""
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=header)
                        writer.writeheader()
                        writer.writerows(step_rows)

    print("wrote", args.output_dir / f"controller_comparison{tag}.json")
    print("wrote", args.output_dir / f"controller_comparison{tag}_steps.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
