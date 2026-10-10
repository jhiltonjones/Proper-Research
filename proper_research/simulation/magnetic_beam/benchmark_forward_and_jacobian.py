from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_FACTORY = (
    "proper_research.simulation.magnetic_beam."
    "run_solver_smoke_test:build_benchmark_model"
)

# Keep the 68 mm x 62.4 mm source outside the undeformed beam.  The previous
# (30, 0, 30) mm centre placed the base inside the physical magnet while using
# a singular point-dipole field.
DEFAULT_P7 = (0.0, 0.12, 0.0, 0.2, -0.1, 0.15, 0.06)
DEFAULT_STEP_P7 = (1e-4, 0.0, 0.0, 0.0, 0.0, 1e-3, 1e-4)


def load_factory(spec: str) -> Callable:
    """Load ``module.submodule:function`` without coupling this script to a package path."""
    try:
        module_name, function_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Factory must use module:function syntax, got {spec!r}."
        ) from exc

    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{spec!r} resolved to a non-callable object.")
    return factory


def build_model(factory: Callable, contact_mode: str):
    if contact_mode == "factory-default":
        return factory()
    return factory(contact_enabled=(contact_mode == "on"))


def measure_ms(
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> tuple[dict[str, float | int], list[float]]:
    for _ in range(warmups):
        operation()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start_ns = time.perf_counter_ns()
        operation()
        elapsed_ns = time.perf_counter_ns() - start_ns
        samples_ms.append(elapsed_ns / 1.0e6)

    values = np.asarray(samples_ms, float)
    summary: dict[str, float | int] = {
        "count": int(values.size),
        "min_ms": float(np.min(values)),
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "stdev_ms": (
            float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
        ),
    }
    return summary, samples_ms


def print_summary(name: str, summary: dict[str, float | int]) -> None:
    print(
        f"{name:28s} "
        f"median={summary['median_ms']:10.3f} ms  "
        f"p95={summary['p95_ms']:10.3f} ms  "
        f"min={summary['min_ms']:10.3f} ms  "
        f"max={summary['max_ms']:10.3f} ms"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cold/warm magnetic-beam forward solves and the implicit "
            "tip Jacobian. Optionally append machine-readable results and fail "
            "when median timing budgets regress."
        )
    )
    parser.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        help="Model factory in module:function form.",
    )
    parser.add_argument(
        "--contact",
        choices=("factory-default", "on", "off"),
        default="factory-default",
        help="Pass contact_enabled to the factory, or retain its default.",
    )
    parser.add_argument(
        "--p7",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ", "L"),
        default=DEFAULT_P7,
        help="Magnet position, additive rotation vector, and inserted length.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--step-p7",
        type=float,
        nargs=7,
        metavar=("DX", "DY", "DZ", "DRX", "DRY", "DRZ", "DL"),
        default=DEFAULT_STEP_P7,
        help=(
            "Small alternating controller step used to benchmark a genuine "
            "changed-pose warm solve."
        ),
    )
    parser.add_argument(
        "--jacobian-coordinates",
        choices=("tangent", "pose7"),
        default="tangent",
        help=(
            "Benchmark robot-compatible world-tangent columns or additive "
            "rotation-vector pose7 columns."
        ),
    )
    parser.add_argument(
        "--jacobian-mode",
        choices=("fast", "accurate"),
        default="fast",
        help=(
            "fast uses forward differences and local Hessian reuse; accurate "
            "uses central differences and rebuilds the Hessian."
        ),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="Append one complete benchmark record to this JSON Lines file.",
    )
    parser.add_argument(
        "--max-forward-ms",
        type=float,
        help="Fail if the median warm forward solve exceeds this budget.",
    )
    parser.add_argument(
        "--max-jacobian-ms",
        type=float,
        help="Fail if the median Jacobian exceeds this budget.",
    )
    parser.add_argument(
        "--max-control-step-ms",
        type=float,
        help=(
            "Fail if the median changed-pose forward-plus-Jacobian control "
            "step exceeds this budget."
        ),
    )
    args = parser.parse_args()

    if args.warmups < 0:
        parser.error("--warmups must be non-negative.")
    if args.repeats < 1:
        parser.error("--repeats must be at least one.")
    if args.max_forward_ms is not None and args.max_forward_ms <= 0:
        parser.error("--max-forward-ms must be positive.")
    if args.max_jacobian_ms is not None and args.max_jacobian_ms <= 0:
        parser.error("--max-jacobian-ms must be positive.")
    if args.max_control_step_ms is not None and args.max_control_step_ms <= 0:
        parser.error("--max-control-step-ms must be positive.")

    return args


def main() -> int:
    args = parse_args()
    p7 = np.asarray(args.p7, float)
    step_p7 = np.asarray(args.step_p7, float)
    factory = load_factory(args.factory)
    model = build_model(factory, args.contact)

    def cold_forward():
        model.reset_cache()
        return model.solve(p7, commit=True)

    cold_summary, cold_samples = measure_ms(
        cold_forward,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    model.reset_cache()
    model.solve(p7, commit=True)

    def cached_forward():
        return model.solve(p7, commit=True)

    cached_forward_summary, cached_forward_samples = measure_ms(
        cached_forward,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    def alternating_poses():
        count = 0
        while True:
            sign = 1.0 if count % 2 == 0 else -1.0
            count += 1
            yield p7 + sign * step_p7

    model.reset_cache()
    model.solve(p7, commit=True)
    forward_poses = alternating_poses()

    def warm_changed_forward():
        return model.solve(next(forward_poses), commit=True)

    changed_forward_summary, changed_forward_samples = measure_ms(
        warm_changed_forward,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    # Establish one full Hessian outside the real-time loop.  With fast mode,
    # subsequent calls recompute G_theta but reuse this local Hessian.
    model.reset_cache()
    model.solve(p7, commit=True)
    if args.jacobian_coordinates == "tangent":
        jacobian_method = model.jacobian_tip_actuation_tangent
        jacobian_name = "jacobian_tangent"
    else:
        jacobian_method = model.jacobian_tip_pose7
        jacobian_name = "jacobian_pose7"

    jacobian_method(
        p7,
        solve_if_needed=False,
        mode=args.jacobian_mode,
        reuse_cached=False,
    )

    def jacobian_recomputed():
        return jacobian_method(
            p7,
            solve_if_needed=False,
            mode=args.jacobian_mode,
            reuse_cached=False,
        )

    jac_summary, jac_samples = measure_ms(
        jacobian_recomputed,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    def jacobian_cached():
        return jacobian_method(
            p7,
            solve_if_needed=False,
            mode=args.jacobian_mode,
            reuse_cached=True,
        )

    cached_jac_summary, cached_jac_samples = measure_ms(
        jacobian_cached,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    model.reset_cache()
    model.solve(p7, commit=True)
    jacobian_method(
        p7,
        solve_if_needed=False,
        mode=args.jacobian_mode,
        reuse_cached=False,
    )
    combined_poses = alternating_poses()

    def warm_changed_forward_and_jacobian():
        pose = next(combined_poses)
        model.solve(pose, commit=True)
        return jacobian_method(
            pose,
            solve_if_needed=False,
            mode=args.jacobian_mode,
            reuse_cached=False,
        )

    combined_summary, combined_samples = measure_ms(
        warm_changed_forward_and_jacobian,
        warmups=args.warmups,
        repeats=args.repeats,
    )

    summaries = {
        "cold_forward": cold_summary,
        "cached_forward": cached_forward_summary,
        "warm_changed_forward": changed_forward_summary,
        f"{jacobian_name}_recomputed": jac_summary,
        f"{jacobian_name}_cached": cached_jac_summary,
        "changed_forward_and_jacobian": combined_summary,
    }
    samples = {
        "cold_forward_ms": cold_samples,
        "cached_forward_ms": cached_forward_samples,
        "warm_changed_forward_ms": changed_forward_samples,
        f"{jacobian_name}_recomputed_ms": jac_samples,
        f"{jacobian_name}_cached_ms": cached_jac_samples,
        "changed_forward_and_jacobian_ms": combined_samples,
    }

    print("\nMagnetic-beam timing benchmark")
    print(f"factory: {args.factory}")
    print(f"contact: {args.contact}")
    print(f"p7:      {p7.tolist()}")
    print(f"step_p7: {step_p7.tolist()}")
    print(f"Jacobian mode: {args.jacobian_mode}")
    print(f"runs:    {args.repeats} measured, {args.warmups} warm-up(s)")
    print()
    for name, summary in summaries.items():
        print_summary(name, summary)

    forward_info = model.last_info or {}
    jacobian_diag = model.get_last_jacobian_diag()
    runtime_diagnostics = {
        "forward_solve_path": forward_info.get("solve_path"),
        "forward_hist_n": forward_info.get("hist_n"),
        "forward_total_nit": forward_info.get("total_nit"),
        "forward_total_objective_calls": forward_info.get(
            "total_objective_calls"
        ),
        "forward_total_gradient_calls": forward_info.get(
            "total_gradient_calls"
        ),
        "jacobian_gradient_evaluations": jacobian_diag.get(
            "gradient_evaluations"
        ),
        "jacobian_hessian_reused": jacobian_diag.get("hessian_reused"),
        "jacobian_difference_scheme": jacobian_diag.get("difference_scheme"),
    }
    print("\nLast changed control-step diagnostics")
    for key, value in runtime_diagnostics.items():
        print(f"{key}: {value}")

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "factory": args.factory,
        "contact": args.contact,
        "p7": p7.tolist(),
        "step_p7": step_p7.tolist(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "jacobian_coordinates": args.jacobian_coordinates,
        "jacobian_mode": args.jacobian_mode,
        "python": sys.version,
        "platform": platform.platform(),
        "summary": summaries,
        "samples": samples,
        "runtime_diagnostics": runtime_diagnostics,
    }

    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"\nAppended result to {args.jsonl}")

    failures: list[str] = []
    if (
        args.max_forward_ms is not None
        and changed_forward_summary["median_ms"] > args.max_forward_ms
    ):
        failures.append(
            "warm changed-pose forward median "
            f"{changed_forward_summary['median_ms']:.3f} ms > "
            f"{args.max_forward_ms:.3f} ms"
        )
    if (
        args.max_jacobian_ms is not None
        and jac_summary["median_ms"] > args.max_jacobian_ms
    ):
        failures.append(
            f"{jacobian_name} median "
            f"{jac_summary['median_ms']:.3f} ms > {args.max_jacobian_ms:.3f} ms"
        )
    if (
        args.max_control_step_ms is not None
        and combined_summary["median_ms"] > args.max_control_step_ms
    ):
        failures.append(
            "changed forward-plus-Jacobian median "
            f"{combined_summary['median_ms']:.3f} ms > "
            f"{args.max_control_step_ms:.3f} ms"
        )

    if failures:
        print("\nTiming budget failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())