#!/usr/bin/env python3
"""Time-parameterise a global path with the optimal or the standard profile.

    python run_time_parameterization.py --profile optimal
    python run_time_parameterization.py --profile standard
    python run_time_parameterization.py --profile both      # and compare

``both`` is the interesting one.  It runs the two profiles over the same
geometric path with identical limits and writes them to sibling directories,
then prints the traversal-time difference.  That difference is the only
defensible statement about whether an upstream smoothing pass helped: a
second-derivative proxy is not a substitute for running the same parameteriser
on both paths and comparing durations.

Pair it with the layer-3 analysis for a full picture:

    python analyse_layer3_time_parameterization.py \
        --reference-dir <out>/optimal --baseline-dir <out>/standard
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from proper_research.planning.time_parameterized_configuration_path import (
        TimeParameterizationConfig,
        time_parameterize_saved_global_path,
    )
    from proper_research.planning.time_parameterization_profiles import (
        PROFILES,
        describe,
        profile_scope,
    )
except ModuleNotFoundError:  # standalone review
    from time_parameterized_configuration_path import (  # type: ignore
        TimeParameterizationConfig,
        time_parameterize_saved_global_path,
    )
    from time_parameterization_profiles import (  # type: ignore
        PROFILES,
        describe,
        profile_scope,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        choices=(*PROFILES, "both"),
        default="optimal",
        help="Speed profile. 'both' runs each and reports the difference.",
    )
    parser.add_argument("--global-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--joint-velocity-limit", type=float, default=None)
    parser.add_argument("--insertion-velocity-limit", type=float, default=None)
    parser.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    parser.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    parser.add_argument("--maximum-path-speed", type=float, default=5.0e-3)
    parser.add_argument("--maximum-path-acceleration", type=float, default=2.0e-2)
    parser.add_argument("--velocity-safety-factor", type=float, default=0.8)
    parser.add_argument("--acceleration-safety-factor", type=float, default=0.8)
    parser.add_argument("--timing-subdivisions", type=int, default=4)
    parser.add_argument("--beam-validation-samples", type=int, default=3)
    parser.add_argument("--skip-beam-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    from proper_research.planning.planning_context import build_planning_context

    _, bundle, controller_pack, out_root = build_planning_context()
    global_dir = args.global_dir or (out_root / "global_configuration_full_debug")
    output_root = Path(
        args.output_dir or (out_root / "time_parameterized_configuration_path")
    )

    dt = args.dt
    if dt is None:
        for key in ("controller", "mpc"):
            value = getattr(controller_pack.get(key), "dt", None)
            if value is not None and np.isfinite(value) and float(value) > 0.0:
                dt = float(value)
                break
    if dt is None:
        raise SystemExit("Could not infer the sample period; pass --dt.")

    u_max = np.asarray(controller_pack["u_max"], dtype=float).reshape(7)
    joint_velocity = args.joint_velocity_limit or float(np.min(u_max[:6]))
    insertion_velocity = args.insertion_velocity_limit or float(u_max[6])

    config = TimeParameterizationConfig(
        sample_period_s=float(dt),
        state_velocity_limit=tuple([joint_velocity] * 6 + [insertion_velocity]),
        state_acceleration_limit=tuple(
            [float(args.joint_acceleration_limit)] * 6
            + [float(args.insertion_acceleration_limit)]
        ),
        velocity_safety_factor=float(args.velocity_safety_factor),
        acceleration_safety_factor=float(args.acceleration_safety_factor),
        maximum_path_speed_m_s=float(args.maximum_path_speed),
        maximum_path_acceleration_m_s2=float(args.maximum_path_acceleration),
        timing_subdivisions_per_interval=int(args.timing_subdivisions),
        beam_validation_samples_per_interval=int(args.beam_validation_samples),
        validate_nonlinear_beam=not args.skip_beam_validation,
    )

    profiles = list(PROFILES) if args.profile == "both" else [args.profile]
    results: dict[str, dict] = {}
    for profile in profiles:
        directory = output_root / profile if len(profiles) > 1 else output_root
        print(f"\n[TIME PARAM] profile={profile} -- {describe(profile)}", flush=True)
        started = time.perf_counter()
        with profile_scope(profile):
            result = time_parameterize_saved_global_path(
                global_output_dir=global_dir,
                controller_pack=controller_pack,
                config=config,
                output_dir=directory,
                lumen_C=bundle.lumen_C,
            )
        elapsed = time.perf_counter() - started
        peak_velocity = float(
            np.max(
                np.abs(result.state_rate)
                / np.asarray(config.effective_velocity_limit)
            )
        )
        peak_acceleration = float(
            np.max(
                np.abs(result.state_acceleration)
                / np.asarray(config.effective_acceleration_limit)
            )
        )
        results[profile] = {
            "output_dir": str(directory),
            "duration_s": float(result.duration_s),
            "samples": int(result.time_s.size),
            "peak_velocity_utilisation": peak_velocity,
            "peak_acceleration_utilisation": peak_acceleration,
            "beam_feasible_fraction": float(np.mean(result.beam_sample_feasible)),
            "maximum_position_error_mm": float(
                1e3 * np.max(result.position_error_m)
            ),
            "wall_time_s": float(elapsed),
        }
        print(
            f"[TIME PARAM] duration={result.duration_s:.4f} s "
            f"samples={result.time_s.size} "
            f"peak |v|/vmax={peak_velocity:.3f} "
            f"peak |a|/amax={peak_acceleration:.3f} "
            f"-> {directory}",
            flush=True,
        )

    if len(results) == 2:
        optimal = results["optimal"]["duration_s"]
        standard = results["standard"]["duration_s"]
        saved = standard - optimal
        percent = 100.0 * saved / standard if standard > 0 else math.nan
        comparison = {
            "standard_duration_s": standard,
            "optimal_duration_s": optimal,
            "seconds_saved": saved,
            "percent_faster": percent,
            "profiles": results,
            "note": (
                "Same geometric path, same limits, same grid, same beam "
                "validation. The only difference is the speed profile, so this "
                "difference is what the optimal parameteriser is worth on this "
                "path. Re-run both on a differently smoothed path to find out "
                "what the smoothing is worth."
            ),
        }
        (output_root / "profile_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8"
        )
        print(
            f"\n[TIME PARAM COMPARISON] standard {standard:.4f} s -> "
            f"optimal {optimal:.4f} s  ({percent:+.2f}% faster)\n"
            f"  saved to {output_root / 'profile_comparison.json'}"
        )


if __name__ == "__main__":
    main()
