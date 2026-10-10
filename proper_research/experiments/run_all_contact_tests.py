"""Run the whole contact study -- Tests A, B, C, and the controller comparison
-- against one pinned vessel, with one command.

This is a thin orchestrator: every stage below is one of the standalone,
independently-runnable scripts in this package, invoked as a subprocess with
the SAME pinned geometry so nothing can silently drift onto a different
vessel mid-run (see pinned_planning_context.py). Nothing here duplicates
their logic -- read this file to see the recipe, or run any one script by
hand for a stage you want to iterate on alone.

Stages (see --stages)
----------------------
a              Test A: offline Jacobian divergence + explainability figures
jacobian       the free/contact Jacobian path-variation checks (Test B follow-up)
b              Test B: scheduling index S + contraction margin mu
c              Test C: naive-controller clip ratio (quick ablation + full run)
controllers    all four controllers head-to-head, closed loop

Cost: this is a real research pipeline, not a demo -- expect stage "a" alone
to take ~15-30 min (5 radii), "b" similarly, "c" ~20-90 min depending on
--controller-radius-scales, and "controllers" ~15-30 min PER (radius,
Jacobian-model) cell (schedule + 3 MPC variants dominate). Use --dry-run
first to see the exact commands and skip whatever you don't need with
--stages.

Usage
-----
    # see what would run, without running it
    python -m proper_research.experiments.run_all_contact_tests \\
        --output-dir results/contact_study_radius_p30_m70 \\
        --first-angle-deg 30 --second-angle-deg -70 --nominal-radius-mm 3.5 \\
        --title-suffix "+30/-70 double-bend vessel" --dry-run

    # the real thing, all stages
    python -m proper_research.experiments.run_all_contact_tests \\
        --output-dir results/contact_study_radius_pXX_mYY \\
        --first-angle-deg XX --second-angle-deg -YY --nominal-radius-mm R \\
        --title-suffix "+XX/-YY vessel"

    # just Test A and the jacobian checks, e.g. while iterating on a new geometry
    python -m proper_research.experiments.run_all_contact_tests \\
        --output-dir results/contact_study_radius_pXX_mYY \\
        --first-angle-deg XX --second-angle-deg -YY --nominal-radius-mm R \\
        --stages a jacobian

    # fill in the naive-vs-LTV-offline-under-no_contact comparison the
    # default controller run skips (see the module docstring above)
    python -m proper_research.experiments.run_all_contact_tests \\
        --output-dir results/contact_study_radius_pXX_mYY \\
        --first-angle-deg XX --second-angle-deg -YY --nominal-radius-mm R \\
        --stages controllers --include-no-contact-controllers
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

STAGE_CHOICES = ("a", "jacobian", "b", "c", "controllers")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tests A/B/C and the controller comparison, one command, "
                    "one pinned vessel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="",
                         help='e.g. "+30/-70 double-bend vessel" -- used in every figure title.')
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Explicit time-parameterised reference trajectory directory. If "
             "omitted, each stage auto-detects it under the vessel's out_root "
             "(the standard time_parameterized_configuration_path, then the "
             "--from-inverse / --profile both outputs).",
    )

    geometry = parser.add_argument_group("vessel geometry")
    geometry.add_argument("--first-angle-deg", type=float, default=None)
    geometry.add_argument("--second-angle-deg", type=float, default=None)
    geometry.add_argument("--nominal-radius-mm", type=float, default=None)
    geometry.add_argument("--run-root", type=Path, default=None)
    geometry.add_argument("--r-beam-mm", type=float, default=1.0)

    sweep = parser.add_argument_group("radius sweeps")
    sweep.add_argument("--radius-scales", type=float, nargs="+",
                        default=(1.15, 1.0, 0.70, 0.50),
                        help="Test A, Test B, and the Jacobian-variation checks.")
    sweep.add_argument("--controller-radius-scales", type=float, nargs="+",
                        default=(1.0, 0.70,0.5),
                        help="Test C and the controller comparison (centre/threshold/corner).")
    sweep.add_argument("--offline-stride", type=int, default=4)
    sweep.add_argument("--mu-stride", type=int, default=5)

    c_group = parser.add_argument_group("Test C")
    c_group.add_argument("--quick-max-steps", type=int, default=320)
    c_group.add_argument("--quick-jacobian-models", nargs="+", default=("contact", "no_contact"))

    ctrl_group = parser.add_argument_group("controller comparison")
    ctrl_group.add_argument(
        "--controllers", nargs="+",
        default=("naive_inverse_jacobian", "mpc_lti", "mpc_ltv_offline", "mpc_ltv_sqp_online"),
    )
    ctrl_group.add_argument(
        "--include-no-contact-controllers", action="store_true",
        help="Also run all four controllers under the no_contact Jacobian -- fills the "
             "naive-vs-LTV-offline-under-a-contact-blind-model comparison the default "
             "(contact-only) run skips. Doubles the cost of this stage.",
    )

    parser.add_argument("--stages", nargs="+", default=list(STAGE_CHOICES),
                         choices=STAGE_CHOICES)
    parser.add_argument("--dry-run", action="store_true",
                         help="Print every command that would run, and stop.")
    parser.add_argument("--keep-going", action="store_true",
                         help="Don't stop the whole run if one stage's subprocess fails.")
    return parser.parse_args()


def _geometry_flags(args: argparse.Namespace) -> list[str]:
    flags: list[str] = []
    if args.first_angle_deg is not None:
        flags += ["--first-angle-deg", str(args.first_angle_deg)]
    if args.second_angle_deg is not None:
        flags += ["--second-angle-deg", str(args.second_angle_deg)]
    if args.nominal_radius_mm is not None:
        flags += ["--nominal-radius-mm", str(args.nominal_radius_mm)]
    if args.run_root is not None:
        flags += ["--run-root", str(args.run_root)]
    return flags


def _reference_flags(args: argparse.Namespace) -> list[str]:
    return ["--reference", str(args.reference)] if args.reference is not None else []


def _cmd(module: str, *flags: str) -> list[str]:
    return [sys.executable, "-u", "-m", f"proper_research.experiments.{module}", *flags]


def build_plan(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """[(stage_label, argv), ...] in execution order, filtered by --stages."""
    geo = _geometry_flags(args)
    ref = _reference_flags(args)
    out = str(args.output_dir)
    radii = [f"{r:g}" for r in args.radius_scales]
    ctrl_radii = [f"{r:g}" for r in args.controller_radius_scales]
    plan: list[tuple[str, list[str]]] = []

    if "a" in args.stages:
        plan.append((
            "Test A: offline Jacobian divergence",
            _cmd("run_contact_study", "--stage", "offline", "--output-dir", out,
                 "--radius-scales", *radii, "--offline-stride", str(args.offline_stride), *geo, *ref),
        ))
        plan.append((
            "Test A: explainability + beam-shape figures",
            _cmd("test_a_analysis", "--output-dir", out, "--r-beam-mm", str(args.r_beam_mm),
                 "--title-suffix", args.title_suffix, *geo, *ref),
        ))

    if "jacobian" in args.stages:
        plan.append((
            "Jacobian check: does J_free vary along the path (and stay radius-blind)?",
            _cmd("check_free_jacobian_path_variation", "--output-dir", out,
                 "--radius-scales", *radii, "--stride", str(args.offline_stride),
                 "--title-suffix", args.title_suffix, *geo, *ref),
        ))
        plan.append((
            "Jacobian check: J_contact vs J_free path variation",
            _cmd("check_contact_jacobian_path_variation", "--output-dir", out,
                 "--radius-scales", *radii, "--stride", str(args.offline_stride),
                 "--title-suffix", args.title_suffix, *geo, *ref),
        ))

    if "b" in args.stages:
        plan.append((
            "Test B: scheduling index S + contraction margin mu",
            _cmd("scheduling_index_study", "--output-dir", out,
                 "--radius-scales", *radii, "--stride", str(args.offline_stride),
                 "--mu-stride", str(args.mu_stride), *geo, *ref),
        ))
        plan.append((
            "Test B: plot",
            _cmd("plot_scheduling_index", "--study-dir", out,
                 "--title-suffix", args.title_suffix),
        ))

    if "c" in args.stages:
        plan.append((
            "Test C: quick clip-ratio ablation (Jacobian model x radius)",
            _cmd("clip_ratio_study", "--output-dir", out,
                 "--radius-scales", ctrl_radii[0], ctrl_radii[-1],
                 "--jacobian-models", *args.quick_jacobian_models,
                 "--max-steps", str(args.quick_max_steps), "--tag", "quick", *geo, *ref),
        ))
        plan.append((
            "Test C: full-length clip-ratio run (contact Jacobian)",
            _cmd("clip_ratio_study", "--output-dir", out,
                 "--radius-scales", *ctrl_radii, "--jacobian-models", "contact",
                 "--tag", "full", *geo, *ref),
        ))
        plan.append((
            "Test C: plot (quick)",
            _cmd("plot_clip_ratio_study", "--study-dir", out, "--tags", "quick",
                 "--title-suffix", f"{args.title_suffix}, {args.quick_max_steps}-step truncation"),
        ))
        plan.append((
            "Test C: plot (full)",
            _cmd("plot_clip_ratio_study", "--study-dir", out, "--tags", "full",
                 "--title-suffix", f"{args.title_suffix}, full path"),
        ))

    if "controllers" in args.stages:
        jacobian_models = ["contact"]
        if args.include_no_contact_controllers:
            jacobian_models.append("no_contact")
        plan.append((
            "Controller comparison: all four controllers, closed loop"
            + (" (contact + no_contact)" if args.include_no_contact_controllers else " (contact)"),
            _cmd("controller_comparison_study", "--output-dir", out,
                 "--radius-scales", *ctrl_radii, "--jacobian-models", *jacobian_models,
                 "--controllers", *args.controllers, *geo, *ref),
        ))
        plan.append((
            "Controller comparison: plot",
            _cmd("plot_controller_comparison", "--study-dir", out,
                 "--title-suffix", f"{args.title_suffix}, full path"),
        ))

    return plan


def main() -> int:
    args = _arguments()
    plan = build_plan(args)

    print(f"[run-all] {len(plan)} steps across stages {args.stages}, output-dir={args.output_dir}")
    for index, (label, argv) in enumerate(plan, start=1):
        print(f"  {index}. {label}\n     $ {shlex.join(argv)}")
    if args.dry_run:
        print("[run-all] --dry-run: stopping without executing anything.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "run_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for index, (label, argv) in enumerate(plan, start=1):
        slug = label.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        log_path = log_dir / f"{index:02d}_{slug[:60]}.log"
        print(f"\n[run-all] ({index}/{len(plan)}) {label}", flush=True)
        print(f"[run-all] logging to {log_path}", flush=True)
        started = time.perf_counter()
        with log_path.open("w") as log_file:
            process = subprocess.run(argv, stdout=log_file, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        status = "OK" if process.returncode == 0 else f"FAILED (exit {process.returncode})"
        print(f"[run-all] {label}: {status} in {elapsed:.1f}s", flush=True)
        if process.returncode != 0:
            failures.append(label)
            print(f"[run-all] see {log_path} for the full output", flush=True)
            if not args.keep_going:
                print("[run-all] stopping (pass --keep-going to continue past a failed stage).",
                      flush=True)
                return 1

    if failures:
        print(f"\n[run-all] completed with {len(failures)} failed stage(s): {failures}")
        return 1
    print(f"\n[run-all] all {len(plan)} steps completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
