#!/usr/bin/env python3
"""Layer 4 analysis — MPC tracking of the timed trajectory.

What this layer does
--------------------
It closes the loop. The timed reference is a feedforward plan; the MPC solves a
condensed QP each sample to keep the *measured* beam tip on the desired
centreline despite the plan being built from a model that is not the plant.

The questions this script answers, in order of how often they are the real
problem:

1. **Whose error is it?** The tip error you measure at the end of the stack is
   part planning error (the plan's own configuration does not put the tip where
   the centreline wants it) and part control error (the controller does not
   reach the plan's configuration). Fixing the wrong one wastes weeks. If the
   timed reference is supplied, this script separates them.
2. **Is the affine output model good enough?** The controller predicts the next
   tip position from ``p_nom + J (z - z_ref) + d_hat``. Comparing that one-step
   prediction against what the plant actually did is the cleanest evidence for
   whether an offline Jacobian suffices or an online relinearisation is needed —
   the same question the controller comparison answers with more machinery.
3. **Would it run on hardware?** A QP that takes longer than the sample period
   is not a controller.

Usage
-----
    python analyse_layer4_mpc.py --mpc-dir results/..._mpc_simulation
    python analyse_layer4_mpc.py --mpc-dir results/..._mpc_simulation \
        --reference-dir results/time_parameterized_configuration_path
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


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mpc-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-dir", type=Path, default=None,
        help="The timed reference the MPC tracked; enables error attribution.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def _load(mpc_dir: Path) -> tuple[C.Table, dict]:
    csv_path = C.find_one(
        mpc_dir,
        "configuration_mpc_simulation.csv",
        "*mpc_simulation.csv",
        "*simulation.csv",
    )
    if csv_path is None:
        raise FileNotFoundError(f"No MPC simulation CSV in {mpc_dir}")
    json_path = C.find_one(
        mpc_dir,
        "configuration_mpc_simulation_summary.json",
        "*summary.json",
    )
    payload = C.read_json(json_path) if json_path is not None else {}
    return C.read_table(csv_path), payload


def analyse_mpc_layer(
    mpc_dir: Path, reference_dir: Path | None = None
) -> tuple[C.Report, dict]:
    mpc_dir = Path(mpc_dir).expanduser().resolve()
    table, summary = _load(mpc_dir)

    report = C.Report("Layer 4 — MPC tracking")
    mpc_configuration = dict(summary.get("mpc_configuration", {}))
    simulation_configuration = dict(summary.get("simulation_configuration", {}))
    position_tolerance = float(
        summary.get("position_tolerance_m")
        or simulation_configuration.get("position_tolerance_m")
        or 1.5e-3
    )
    tangent_tolerance = float(
        summary.get("tangent_tolerance_rad")
        or simulation_configuration.get("tangent_tolerance_rad")
        or math.pi / 2
    )
    sample_period = float(mpc_configuration.get("sample_period_s", math.nan))
    if not np.isfinite(sample_period) and table.has("time_s"):
        sample_period = float(np.median(np.diff(table.column("time_s"))))

    metrics: dict = {
        "layer": 4,
        "name": "mpc_tracking",
        "source_dir": str(mpc_dir),
        "source_csv": str(table.path),
        "sample_count": int(table.row_count),
        "sample_period_s": sample_period,
        "position_tolerance_m": position_tolerance,
        "tangent_tolerance_rad": tangent_tolerance,
        "summary_from_simulation": {
            key: summary.get(key)
            for key in (
                "success", "stopped_reason", "all_qp_solved",
                "terminal_beam_tracking_satisfied",
                "nominal_terminal_configuration_satisfied",
                "analytical_jacobian_source",
            )
            if key in summary
        },
    }

    # -------------------------------------------------------- tracking
    position_error = table.column("beam_position_error_m", math.nan)
    if not np.any(np.isfinite(position_error)) and table.has("beam_position_error_mm"):
        position_error = table.column("beam_position_error_mm") * 1e-3
    tangent_error = table.column("beam_tangent_error_rad", math.nan)
    position_budget = C.budget_report(
        "tip position", position_error, position_tolerance,
        unit="mm", display_scale=1e3,
    )
    tangent_budget = C.budget_report(
        "tip tangent", tangent_error, tangent_tolerance,
        unit="deg", display_scale=180.0 / math.pi,
    )
    metrics["position_budget"] = position_budget.as_dict()
    metrics["tangent_budget"] = tangent_budget.as_dict()

    report.head("Closed-loop tracking")
    report.table(
        ["quantity", "tolerance", "max", "p95", "rms", "budget p95"],
        [
            ["tip position", f"{1e3 * position_tolerance:.3f} mm",
             f"{1e3 * position_budget.maximum:.4f} mm",
             f"{1e3 * position_budget.p95:.4f} mm",
             f"{1e3 * position_budget.rms:.4f} mm",
             f"{position_budget.utilisation_p95:.3f}x"],
            ["tip tangent", f"{math.degrees(tangent_tolerance):.1f} deg",
             f"{math.degrees(tangent_budget.maximum):.3f} deg",
             f"{math.degrees(tangent_budget.p95):.3f} deg",
             f"{math.degrees(tangent_budget.rms):.3f} deg",
             f"{tangent_budget.utilisation_p95:.3f}x"],
        ],
    )
    if position_budget.utilisation_max < 0.25:
        report.finding(
            "ok", "Closed-loop tip tracking is comfortably inside tolerance",
            f"peak {1e3 * position_budget.maximum:.3f} mm against "
            f"{1e3 * position_tolerance:.2f} mm, i.e. "
            f"{position_budget.utilisation_max:.2f} of budget. Micron-scale roughness "
            "upstream is invisible at this scale — worth remembering before spending "
            "effort smoothing a planner path the controller cannot even resolve.",
        )
    elif position_budget.over_budget_fraction > 0:
        report.finding(
            "fail", "The controller leaves tolerance",
            f"{100 * position_budget.over_budget_fraction:.1f}% of samples exceed "
            f"{1e3 * position_tolerance:.2f} mm.",
        )

    # ------------------------------------------------ error attribution
    report.head("Whose error is it?")
    joint_error_keys = [f"q{i}_error" for i in range(1, 7)]
    if table.has(*joint_error_keys):
        joint_error = np.stack([table.columns[key] for key in joint_error_keys], axis=1)
        metrics["configuration_tracking"] = {
            "max_abs_joint_error_deg": float(np.degrees(C.max_abs(joint_error))),
            "rms_joint_error_deg": float(np.degrees(C.rms(joint_error))),
            "max_abs_insertion_error_mm": (
                1e3 * float(C.max_abs(table.column("insertion_error", math.nan)))
            ),
        }
    planning = None
    if reference_dir is not None:
        reference_csv = C.find_one(
            Path(reference_dir),
            "time_parameterized_configuration_path.csv",
            "*time_parameterized*.csv",
            "*.csv",
        )
        if reference_csv is not None:
            reference_table = C.read_table(reference_csv)
            key = reference_table.first_present(
                ["planned_position_error_m", "position_error_m"]
            )
            if key is not None:
                planned_error = reference_table.column(key)
                index = table.column("reference_index", math.nan)
                usable = np.isfinite(index)
                mapped = np.full(table.row_count, math.nan)
                if np.any(usable):
                    clipped = np.clip(
                        index[usable].astype(int), 0, planned_error.size - 1
                    )
                    mapped[usable] = planned_error[clipped]
                planning = mapped
                metrics["planning_error"] = {
                    "source_column": key,
                    "max_m": C.maximum(mapped),
                    "rms_m": C.rms(mapped),
                    "share_of_total_rms": (
                        C.rms(mapped) / C.rms(position_error)
                        if C.rms(position_error) > 0 else math.nan
                    ),
                }
    if planning is not None:
        report.text(
            "Total measured tip error contains an error the plan already had. "
            "The controller cannot remove it by tracking the plan better — only "
            "by using beam feedback, which is exactly what the beam-output "
            "variant does."
        )
        report.table(
            ["component", "rms [mm]", "max [mm]", "share of total rms"],
            [
                ["planning (plan's own tip error)",
                 1e3 * metrics["planning_error"]["rms_m"],
                 1e3 * metrics["planning_error"]["max_m"],
                 f"{100 * metrics['planning_error']['share_of_total_rms']:.1f}%"],
                ["total measured (closed loop)",
                 1e3 * position_budget.rms, 1e3 * position_budget.maximum, "100%"],
            ],
        )
        share = metrics["planning_error"]["share_of_total_rms"]
        if np.isfinite(share) and share > 0.5:
            report.finding(
                "info", "Most of the closed-loop error was inherited from the plan",
                f"{100 * share:.0f}% of the rms. Tuning controller weights cannot fix "
                "this; the fix is upstream or in the output feedback.",
            )
    else:
        report.finding(
            "info", "Error attribution unavailable",
            "pass --reference-dir to split the measured tip error into the part "
            "the plan already had and the part the controller added.",
        )

    # ------------------------------- how good is the affine output model?
    predicted = table.vector("predicted_next_beam")
    achieved = table.vector("achieved")
    if predicted is not None and achieved is not None and table.row_count > 2:
        prediction_error = np.linalg.norm(predicted[:-1] - achieved[1:], axis=1)
        step_motion = np.linalg.norm(np.diff(achieved, axis=0), axis=1)
        metrics["one_step_model_quality"] = {
            "prediction_error_max_m": C.maximum(prediction_error),
            "prediction_error_rms_m": C.rms(prediction_error),
            "tip_motion_per_step_rms_m": C.rms(step_motion),
            "relative_prediction_error": (
                C.rms(prediction_error) / C.rms(step_motion)
                if C.rms(step_motion) > 0 else math.nan
            ),
        }
        report.head("Quality of the affine beam model")
        report.text(
            "The controller predicts the next tip position from "
            "`p_nom + J (z - z_ref) + d_hat`, where J was linearised offline at the "
            "reference configuration. Comparing that prediction with what the plant "
            "actually did is the direct test of whether an offline Jacobian is "
            "sufficient."
        )
        report.bullets(
            [
                f"One-step prediction error: rms "
                f"{1e6 * metrics['one_step_model_quality']['prediction_error_rms_m']:.2f} um, "
                f"max {1e6 * metrics['one_step_model_quality']['prediction_error_max_m']:.2f} um.",
                f"Tip motion per step: rms "
                f"{1e6 * metrics['one_step_model_quality']['tip_motion_per_step_rms_m']:.2f} um.",
                f"Relative prediction error: "
                f"{metrics['one_step_model_quality']['relative_prediction_error']:.3f}.",
            ]
        )
        relative = metrics["one_step_model_quality"]["relative_prediction_error"]
        if np.isfinite(relative) and relative > 0.5:
            report.finding(
                "warn", "The offline Jacobian predicts poorly one step ahead",
                f"prediction error is {relative:.2f}x the tip motion it is trying to "
                "predict. That is the case for relinearising online (SQP) rather than "
                "trusting the reference-point Jacobian.",
            )
        elif np.isfinite(relative):
            report.finding(
                "ok", "The offline Jacobian is a good local model",
                f"one-step prediction error is only {relative:.2f}x the per-step tip "
                "motion, so online relinearisation has little headroom to improve on it.",
            )

    residual = table.column("beam_residual_norm_m", math.nan)
    if np.any(np.isfinite(residual)):
        metrics["estimated_residual"] = {
            "max_m": C.maximum(residual),
            "rms_m": C.rms(residual),
        }

    # ------------------------------------------------------- QP health
    report.head("Solver health and real-time feasibility")
    solve_ms = table.column("qp_solve_time_s", math.nan) * 1e3
    success = table.column("qp_success", 1.0) > 0.5
    iterations = table.column("qp_iterations", math.nan)
    metrics["qp"] = {
        "success_fraction": C.fraction(success),
        "failures": int(np.sum(~success)),
        "solve_ms_mean": float(np.nanmean(solve_ms)) if solve_ms.size else math.nan,
        "solve_ms_p95": C.percentile(solve_ms, 95.0),
        "solve_ms_max": C.maximum(solve_ms),
        "iterations_max": C.maximum(iterations),
        "sample_period_ms": 1e3 * sample_period,
        "overruns": (
            int(np.sum(C.finite(solve_ms) > 1e3 * sample_period))
            if np.isfinite(sample_period) else None
        ),
    }
    report.bullets(
        [
            f"QP success {100 * metrics['qp']['success_fraction']:.1f}% "
            f"({metrics['qp']['failures']} failures).",
            f"Solve time mean {metrics['qp']['solve_ms_mean']:.2f} ms, "
            f"p95 {metrics['qp']['solve_ms_p95']:.2f} ms, "
            f"max {metrics['qp']['solve_ms_max']:.2f} ms "
            f"against a {1e3 * sample_period:.1f} ms sample period.",
        ]
    )
    if metrics["qp"]["overruns"]:
        report.finding(
            "warn",
            f"{metrics['qp']['overruns']} QP solve(s) exceeded the sample period",
            f"worst {metrics['qp']['solve_ms_max']:.1f} ms against "
            f"{1e3 * sample_period:.1f} ms. In simulation this is invisible; on "
            "hardware it is a missed deadline. Report worst-case, not mean, solve "
            "time when comparing controllers.",
        )

    # ------------------------------------------------ limit utilisation
    velocity_limit = np.asarray(
        mpc_configuration.get("velocity_limit", [math.nan] * 7), dtype=float
    )
    acceleration_limit = np.asarray(
        mpc_configuration.get("acceleration_limit", [math.nan] * 7), dtype=float
    )
    command_keys = [f"qd{i}_command" for i in range(1, 7)]
    if table.has(*command_keys):
        commands = np.stack(
            [table.columns[key] for key in command_keys]
            + [table.column("insertion_rate_command", math.nan)],
            axis=1,
        )
        velocity_use = C.limit_utilisation(commands, velocity_limit)
        metrics["command_limits"] = {
            "peak_velocity_utilisation": C.max_abs(velocity_use, axis=0).tolist(),
            "velocity_binding": C.binding_axis(velocity_use),
        }
    acceleration_keys = [f"qd{i}_acceleration" for i in range(1, 7)]
    if table.has(*acceleration_keys):
        accelerations = np.stack(
            [table.columns[key] for key in acceleration_keys]
            + [table.column("insertion_rate_acceleration", math.nan)],
            axis=1,
        )
        acceleration_use = C.limit_utilisation(accelerations, acceleration_limit)
        metrics["acceleration_limits"] = {
            "peak_acceleration_utilisation": C.max_abs(acceleration_use, axis=0).tolist(),
            "acceleration_binding": C.binding_axis(acceleration_use),
        }
        peak = C.maximum(acceleration_use)
        if np.isfinite(peak) and peak >= 0.999:
            binding = metrics["acceleration_limits"]["acceleration_binding"]
            report.finding(
                "warn",
                f"The acceleration limit saturates on {binding.get('binding_axis')}",
                "the controller is running against a hard bound, so any extra "
                "acceleration demand created upstream translates directly into "
                "tracking error rather than faster motion.",
            )

    report.metrics = metrics
    return report, metrics


def _plots(metrics: dict, table: C.Table, output_dir: Path, dpi: int) -> None:
    figure, axes = C.new_figure(3, 1, (12, 10))
    if figure is None:
        return
    time = table.column("time_s", math.nan)
    if not np.any(np.isfinite(time)):
        time = np.arange(table.row_count, dtype=float)
    tolerance = metrics["position_tolerance_m"]
    error = table.column("beam_position_error_m", math.nan)
    axes[0].plot(time, 1e3 * error, color=C.PALETTE["achieved"], label="measured")
    axes[0].axhline(1e3 * tolerance, color=C.PALETTE["limit"], linestyle=":", label="tolerance")
    axes[0].set_ylabel("tip position error [mm]")
    axes[0].set_title("Closed-loop tip tracking")
    axes[0].legend()

    predicted = table.vector("predicted_next_beam")
    achieved = table.vector("achieved")
    if predicted is not None and achieved is not None and table.row_count > 2:
        prediction_error = np.linalg.norm(predicted[:-1] - achieved[1:], axis=1)
        axes[1].semilogy(
            time[:-1], 1e6 * prediction_error,
            color=C.PALETTE["desired"], label="one-step prediction error",
        )
        axes[1].semilogy(
            time[:-1], 1e6 * np.linalg.norm(np.diff(achieved, axis=0), axis=1),
            color=C.PALETTE["before"], label="tip motion per step",
        )
        axes[1].set_ylabel("[um]")
        axes[1].set_title("Affine beam-model quality")
        axes[1].legend()

    solve_ms = table.column("qp_solve_time_s", math.nan) * 1e3
    axes[2].semilogy(time, solve_ms, color=C.PALETTE["magnet"])
    if np.isfinite(metrics["sample_period_s"]):
        axes[2].axhline(
            1e3 * metrics["sample_period_s"], color=C.PALETTE["limit"],
            linestyle=":", label="sample period",
        )
        axes[2].legend()
    axes[2].set_ylabel("QP solve time [ms]")
    axes[2].set_xlabel("time [s]")
    C.save_figure(figure, output_dir / "layer4_tracking_and_solver.png", dpi)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir or (args.mpc_dir / "stack_analysis")
    report, metrics = analyse_mpc_layer(args.mpc_dir, args.reference_dir)
    paths = report.write(Path(output_dir), "layer4_mpc")
    if not args.no_plots:
        table, _ = _load(Path(args.mpc_dir))
        _plots(metrics, table, Path(output_dir), args.dpi)
    print(f"Layer 4 analysis -> {paths['report']}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
