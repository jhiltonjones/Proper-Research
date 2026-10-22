#!/usr/bin/env python3
"""Layer 3 analysis — time parameterisation.

What this layer does
--------------------
It converts the geometric path ``chi(s)`` into a timed trajectory ``chi(t)`` by
choosing a speed profile ``s_dot(s)`` that respects the actuator limits:

    chi_dot  = chi'(s) s_dot
    chi_ddot = chi''(s) s_dot^2 + chi'(s) s_ddot

This is the only layer whose output has units of time, and it is the layer that
converts geometric smoothness into a concrete benefit — or fails to. The two
questions worth asking of it are:

1. **Which axis binds?** A time-optimal profile is bang-bang: at nearly every
   instant some actuator is at a velocity or acceleration limit. Knowing *which*
   one tells you whose smoothness actually buys traversal time. Smoothing a
   coordinate that never binds buys nothing.
2. **Is the geometry or the actuator the bottleneck?** If saturation is low the
   profile is limited by something else (a jerk cap, a heuristic, or a
   conservative safety factor) and improving path smoothness will not help.

This script sniffs column names, so it tolerates schema drift between versions
of the time-parameterisation writer. Pass two directories to compare the timed
result of two different geometric paths — that is the only honest way to claim
a smoother path traverses faster.

Usage
-----
    python analyse_layer3_time_parameterization.py \
        --reference-dir results/time_parameterized_configuration_path
    python analyse_layer3_time_parameterization.py \
        --reference-dir results/timed_from_global \
        --baseline-dir  results/timed_from_inverse
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from . import _stack_common as C
except ImportError:
    import _stack_common as C


# Naming patterns tried in order, so the script survives schema drift.
TIME_KEYS = ("time_s", "t_s", "time", "reference_time_s")
PATH_KEYS = ("path_coordinate_m", "s_m", "reference_path_s_m", "path_s_m", "arc_length_m")
JOINT_PATTERNS = ("q{i}_rad", "q{i}", "q{i}_reference", "state_q{i}")
INSERTION_KEYS = ("insertion_m", "insertion", "L_m", "insertion_reference")
RATE_PATTERNS = ("qd{i}_rad_s", "qd{i}", "qd{i}_reference", "q{i}_dot", "velocity_q{i}")
INSERTION_RATE_KEYS = (
    "insertion_rate_m_s", "insertion_rate", "insertion_rate_reference", "L_dot",
)
ACCEL_PATTERNS = ("qdd{i}_rad_s2", "qdd{i}", "q{i}_ddot", "acceleration_q{i}")
INSERTION_ACCEL_KEYS = (
    "insertion_acceleration_m_s2", "insertion_acceleration", "L_ddot",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument(
        "--baseline-dir", type=Path, default=None,
        help="A second timed trajectory to compare against (e.g. timed from the "
             "un-smoothed path). Enables the traversal-time claim.",
    )
    parser.add_argument(
        "--limits-json", type=Path, default=None,
        help="JSON carrying velocity_limit / acceleration_limit, e.g. an MPC "
             "simulation summary. Without it, limits are inferred from the data.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def _find_table(directory: Path) -> C.Table:
    path = C.find_one(
        directory,
        "time_parameterized_configuration_path.csv",
        "*time_parameterized*.csv",
        "*trajectory*.csv",
        "*.csv",
    )
    if path is None:
        raise FileNotFoundError(f"No trajectory CSV found in {directory}")
    return C.read_table(path)


def _stack(table: C.Table, patterns, insertion_keys) -> tuple[np.ndarray | None, str]:
    for pattern in patterns:
        keys = [pattern.format(i=i) for i in range(1, 7)]
        if table.has(*keys):
            insertion = table.first_present(insertion_keys)
            columns = [table.columns[key] for key in keys]
            if insertion is not None:
                columns.append(table.columns[insertion])
            return np.stack(columns, axis=1), pattern
    return None, ""


def _limits(payload: dict | None, key: str, width: int) -> np.ndarray:
    if payload:
        for holder in (payload, payload.get("mpc_configuration", {})):
            if isinstance(holder, dict) and key in holder:
                values = np.asarray(holder[key], dtype=float).reshape(-1)
                if values.size >= width:
                    return values[:width]
    return np.full(width, math.nan)


def analyse_time_layer(
    reference_dir: Path,
    baseline_dir: Path | None = None,
    limits_json: Path | None = None,
) -> tuple[C.Report, dict]:
    reference_dir = Path(reference_dir).expanduser().resolve()
    table = _find_table(reference_dir)
    payload = None
    for candidate in (
        limits_json,
        reference_dir / "time_parameterized_configuration_summary.json",
    ):
        if candidate is not None and Path(candidate).exists():
            payload = C.read_json(Path(candidate))
            break

    report = C.Report("Layer 3 — time parameterisation")
    metrics: dict = {
        "layer": 3,
        "name": "time_parameterization",
        "source_dir": str(reference_dir),
        "source_csv": str(table.path),
        "sample_count": int(table.row_count),
    }

    time_key = table.first_present(TIME_KEYS)
    if time_key is None:
        raise ValueError(
            f"{table.path.name} has no recognisable time column; tried {TIME_KEYS}."
        )
    time = table.column(time_key)
    duration = float(time[-1] - time[0])
    dt = np.diff(time)
    metrics["timing"] = {
        "time_column": time_key,
        "duration_s": duration,
        "median_sample_period_s": float(np.median(dt)) if dt.size else math.nan,
        "sample_period_jitter_s": float(np.max(dt) - np.min(dt)) if dt.size else 0.0,
        "uniform_sampling": bool(dt.size and (np.max(dt) - np.min(dt)) < 1e-9),
    }

    path_key = table.first_present(PATH_KEYS)
    if path_key is not None:
        s = table.column(path_key)
        forward = np.all(np.diff(s) >= -1e-12)
        metrics["path"] = {
            "path_column": path_key,
            "total_length_m": float(s[-1] - s[0]),
            "monotone_non_decreasing": bool(forward),
            "mean_speed_m_s": float((s[-1] - s[0]) / duration) if duration > 0 else math.nan,
        }
        if not forward:
            report.finding(
                "warn", "Path coordinate is not monotone",
                "s(t) decreases somewhere, which a forward traversal should not do.",
            )
    else:
        s = None

    states, _ = _stack(table, JOINT_PATTERNS, INSERTION_KEYS)
    rates, rate_pattern = _stack(table, RATE_PATTERNS, INSERTION_RATE_KEYS)
    accelerations, accel_pattern = _stack(table, ACCEL_PATTERNS, INSERTION_ACCEL_KEYS)

    if rates is None and states is not None and dt.size:
        rates = np.vstack(
            [np.zeros((1, states.shape[1])), np.diff(states, axis=0) / dt[:, None]]
        )
        rate_pattern = "finite differences of the state"
        report.finding(
            "info", "Velocities were differenced from the state",
            "no velocity columns were found, so limit utilisation below is a "
            "reconstruction and will be noisier than the planner's own numbers.",
        )
    if accelerations is None and rates is not None and dt.size:
        accelerations = np.vstack(
            [np.zeros((1, rates.shape[1])), np.diff(rates, axis=0) / dt[:, None]]
        )
        accel_pattern = "finite differences of the velocity"

    width = states.shape[1] if states is not None else 7
    velocity_limit = _limits(payload, "velocity_limit", width)
    acceleration_limit = _limits(payload, "acceleration_limit", width)
    inferred = False
    if rates is not None and not np.all(np.isfinite(velocity_limit)):
        velocity_limit = C.max_abs(rates, axis=0)
        inferred = True
    if accelerations is not None and not np.all(np.isfinite(acceleration_limit)):
        acceleration_limit = C.max_abs(accelerations, axis=0)
        inferred = True
    if inferred:
        report.finding(
            "warn", "Actuator limits were inferred from the trajectory itself",
            "utilisation is then 1.0 by construction on the peak axis. Pass "
            "--limits-json (an MPC summary works) for a meaningful reading.",
        )

    report.head("Traversal")
    report.bullets(
        [
            f"Duration **{duration:.3f} s** over {table.row_count} samples "
            f"(dt = {metrics['timing']['median_sample_period_s'] * 1e3:.2f} ms).",
            (
                f"Path length {metrics['path']['total_length_m'] * 1e3:.2f} mm, "
                f"mean path speed {metrics['path']['mean_speed_m_s'] * 1e3:.3f} mm/s."
                if s is not None else "No path-coordinate column found."
            ),
        ]
    )

    # ------------------------------------------------------ binding axis
    if rates is not None:
        velocity_use = C.limit_utilisation(rates, velocity_limit)
        metrics["velocity"] = {
            "source": rate_pattern,
            "limit": velocity_limit.tolist(),
            "peak_utilisation_per_axis": C.max_abs(velocity_use, axis=0).tolist(),
            "saturated_fraction": C.fraction(
                np.nanmax(velocity_use, axis=1) >= 0.99
            ),
            "binding": C.binding_axis(velocity_use),
        }
    if accelerations is not None:
        acceleration_use = C.limit_utilisation(accelerations, acceleration_limit)
        metrics["acceleration"] = {
            "source": accel_pattern,
            "limit": acceleration_limit.tolist(),
            "peak_utilisation_per_axis": C.max_abs(acceleration_use, axis=0).tolist(),
            "saturated_fraction": C.fraction(
                np.nanmax(acceleration_use, axis=1) >= 0.99
            ),
            "binding": C.binding_axis(acceleration_use),
        }

    if "velocity" in metrics or "acceleration" in metrics:
        report.head("Which axis binds")
        report.text(
            "A time-optimal profile is bang-bang: some actuator should be at a "
            "limit almost everywhere. The binding axis is the one whose geometric "
            "smoothness actually buys traversal time — smoothing anything else is "
            "free of benefit here."
        )
        rows = []
        for i in range(width):
            rows.append(
                [
                    C.COORDINATE_NAMES[i] if i < len(C.COORDINATE_NAMES) else str(i),
                    velocity_limit[i] if i < velocity_limit.size else math.nan,
                    (metrics.get("velocity", {}).get("peak_utilisation_per_axis") or [math.nan] * width)[i],
                    acceleration_limit[i] if i < acceleration_limit.size else math.nan,
                    (metrics.get("acceleration", {}).get("peak_utilisation_per_axis") or [math.nan] * width)[i],
                ]
            )
        report.table(
            ["coordinate", "velocity limit", "peak v/vmax",
             "acceleration limit", "peak a/amax"],
            rows,
        )
        for name, block in (("velocity", metrics.get("velocity")),
                            ("acceleration", metrics.get("acceleration"))):
            if not block or not block["binding"].get("available"):
                continue
            binding = block["binding"]
            share = binding["share_of_samples_where_axis_is_worst"].get(
                binding["binding_axis"], 0.0
            )
            peak = binding["binding_peak"]
            if peak > 1.0 + 1e-6:
                report.finding(
                    "fail",
                    f"The trajectory exceeds its {name} limit on {binding['binding_axis']}",
                    f"peak {peak:.2f}x the limit. Either the limits passed here are not "
                    "the ones the parameteriser used, or the saved trajectory is not "
                    "actuator-feasible — check before trusting anything downstream of it.",
                )
            elif peak >= 0.5:
                report.finding(
                    "info",
                    f"{binding['binding_axis']} binds the {name} limit",
                    f"peak {peak:.2f} of its limit, and it is the closest-to-limit axis "
                    f"for {100 * share:.0f}% of samples. Geometric smoothness on this "
                    "axis is what buys traversal time; smoothing the others does not.",
                )
            else:
                report.finding(
                    "info",
                    f"No axis comes close to its {name} limit",
                    f"the worst is {binding['binding_axis']} at {peak:.2f} of its limit. "
                    "Whatever sets the duration, it is not this bound.",
                )
            saturated = block["saturated_fraction"]
            if np.isfinite(saturated) and saturated < 0.2:
                report.finding(
                    "warn",
                    f"{name.capitalize()} limits are rarely active ({100 * saturated:.0f}% of samples)",
                    "the traversal is not limited by this actuator bound, so a "
                    "smoother geometric path will not make it faster. Look for a jerk "
                    "cap, a safety factor, or a fixed duration instead.",
                )

    # ------------------------------- consistency with the geometric path
    if s is not None and states is not None and rates is not None and dt.size:
        ds = np.gradient(s, time)
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.where(np.abs(ds)[:, None] > 1e-12, rates / ds[:, None], np.nan)
        metrics["geometric_consistency"] = {
            "max_abs_implied_dchi_ds": C.max_abs(implied, axis=0).tolist(),
            "note": (
                "chi'(s) implied by chi_dot / s_dot. Compare with the L1/L2 "
                "path-quality tables — large disagreement means the timed "
                "trajectory is not tracking the geometric path it claims to."
            ),
        }

    # ----------------------------------------------- planned feasibility
    for key in ("planned_position_error_m", "position_error_m"):
        if table.has(key):
            metrics["planned_position_error"] = {
                "column": key,
                "max_m": C.maximum(table.column(key)),
                "rms_m": C.rms(table.column(key)),
            }
            break
    if table.has("planned_beam_feasible"):
        feasible_fraction = C.fraction(table.column("planned_beam_feasible") > 0.5)
        metrics["planned_beam_feasible_fraction"] = feasible_fraction
        if feasible_fraction < 1.0:
            report.finding(
                "fail", "The timed reference contains beam-infeasible samples",
                f"{100 * (1 - feasible_fraction):.1f}% of samples. The MPC layer "
                "refuses these unless --allow-planned-beam-failure is set.",
            )

    # --------------------------------------------------- baseline compare
    if baseline_dir is not None:
        baseline_table = _find_table(Path(baseline_dir).expanduser().resolve())
        baseline_time_key = baseline_table.first_present(TIME_KEYS)
        baseline_time = baseline_table.column(baseline_time_key)
        baseline_duration = float(baseline_time[-1] - baseline_time[0])
        change = C.improvement_percent(baseline_duration, duration)
        metrics["baseline_comparison"] = {
            "baseline_dir": str(baseline_dir),
            "baseline_duration_s": baseline_duration,
            "duration_s": duration,
            "duration_improvement_percent": change,
        }
        report.head("Traversal-time comparison")
        report.table(
            ["trajectory", "duration [s]", "samples"],
            [
                ["baseline", baseline_duration, baseline_table.row_count],
                ["this run", duration, table.row_count],
            ],
        )
        level = "ok" if change > 1.0 else ("warn" if change < -1.0 else "info")
        report.finding(
            level,
            f"Traversal time changed by {-change:+.1f}%",
            f"{baseline_duration:.3f} s -> {duration:.3f} s. This is the only "
            "defensible statement about whether upstream smoothing helped; "
            "derivative proxies are not a substitute for it.",
        )
    else:
        report.finding(
            "info",
            "No baseline trajectory supplied",
            "run this script with --baseline-dir pointing at the timed result of "
            "the un-smoothed path to turn the smoothness proxies into an actual "
            "traversal-time claim.",
        )

    report.metrics = metrics
    return report, metrics


def _plots(metrics: dict, table: C.Table, output_dir: Path, dpi: int) -> None:
    figure, axes = C.new_figure(2, 1, (12, 8))
    if figure is None:
        return
    time_key = metrics["timing"]["time_column"]
    time = table.column(time_key)
    rates, _ = _stack(table, RATE_PATTERNS, INSERTION_RATE_KEYS)
    if rates is not None and "velocity" in metrics:
        use = C.limit_utilisation(rates, np.asarray(metrics["velocity"]["limit"]))
        for i in range(use.shape[1]):
            name = C.COORDINATE_NAMES[i] if i < len(C.COORDINATE_NAMES) else str(i)
            axes[0].plot(time, use[:, i], label=name, linewidth=1.2)
        axes[0].axhline(1.0, color=C.PALETTE["limit"], linestyle=":")
        axes[0].set_ylabel("|v| / v_max")
        axes[0].set_title("Velocity-limit utilisation")
        axes[0].legend(ncol=7, fontsize=8)
    accelerations, _ = _stack(table, ACCEL_PATTERNS, INSERTION_ACCEL_KEYS)
    if accelerations is not None and "acceleration" in metrics:
        use = C.limit_utilisation(
            accelerations, np.asarray(metrics["acceleration"]["limit"])
        )
        for i in range(use.shape[1]):
            name = C.COORDINATE_NAMES[i] if i < len(C.COORDINATE_NAMES) else str(i)
            axes[1].plot(time, use[:, i], label=name, linewidth=1.2)
        axes[1].axhline(1.0, color=C.PALETTE["limit"], linestyle=":")
        axes[1].set_ylabel("|a| / a_max")
        axes[1].set_title("Acceleration-limit utilisation")
    axes[1].set_xlabel("time [s]")
    C.save_figure(figure, output_dir / "layer3_limit_utilisation.png", dpi)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir or (args.reference_dir / "stack_analysis")
    report, metrics = analyse_time_layer(
        args.reference_dir, args.baseline_dir, args.limits_json
    )
    paths = report.write(Path(output_dir), "layer3_time")
    if not args.no_plots:
        _plots(metrics, _find_table(Path(args.reference_dir)), Path(output_dir), args.dpi)
    print(f"Layer 3 analysis -> {paths['report']}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
