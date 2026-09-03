#!/usr/bin/env python3
"""Fair comparison of four beam-tip controllers on one timed trajectory.

The four controllers
--------------------
========================  ================================================
naive_inverse_jacobian    Resolved-rate: one damped-least-squares inverse of
                          the beam Jacobian per step, no preview, no
                          constraints beyond clipping. Nullspace pulls the
                          configuration toward the reference so the redundant
                          4-dimensional nullspace does not drift.
mpc_lti                   The same condensed QP as your beam-output MPC, but
                          with a single Jacobian frozen at the first reference
                          sample and held for the whole run.
mpc_ltv_offline           Your current controller: one analytical Jacobian per
                          reference sample, all precomputed before the loop,
                          never relinearised online.
mpc_ltv_sqp_online        Relinearised at the *measured* state every control
                          step, optionally with inner SQP iterations that
                          re-linearise along the predicted trajectory.
========================  ================================================

They form a ladder, and the point of running all four is to price each rung:
preview and constraints (naive -> LTI), scheduling the model along the path
(LTI -> LTV), and relinearising on the measured state (LTV -> SQP). If a rung
costs solve time and buys no accuracy, you have evidence to stop climbing.

What makes the comparison fair
------------------------------
Every controller is run through the *same* closed-loop function, against the
same plant instance, from the same initial state, with the same reference,
sample period, actuator limits, tolerances, and terminal hold. Only the
controller object differs. Two further controls matter here:

* **Jacobians are built contact-free; the plant keeps contact enabled.** That is
  a deliberate, declared model mismatch, and it is the mismatch each controller
  is being asked to reject. Any controller allowed a contact-aware Jacobian
  would be solving an easier problem.
* **Worst-case solve time is reported, not the mean.** A controller that is
  more accurate but misses its deadline has not won.

Each controller writes a directory in the schema of your existing MPC
simulation, so `analyse_layer4_mpc.py` runs on each one unchanged.

Integration
-----------
This module needs three things from your project, resolved at import time with
explicit errors if they are missing:

    simulate_time_parameterized_configuration_mpc   (base MPC module)
    simulate_time_parameterized_beam_output_mpc     (your LTV controller)
    proper_research.planning.planning_context.build_planning_context

Run `--self-test` to exercise the whole harness against mock objects with no
project imports at all.

Usage
-----
    python compare_controllers.py --self-test
    python compare_controllers.py --output-dir results/controller_comparison
    python compare_controllers.py --controllers naive_inverse_jacobian mpc_ltv_sqp_online
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

try:
    from . import _stack_common as C
except ImportError:
    import _stack_common as C

Array = np.ndarray

CONTROLLER_NAMES = (
    "naive_inverse_jacobian",
    "mpc_lti",
    "mpc_ltv_offline",
    "mpc_ltv_sqp_online",
)


# ==========================================================================
# project bindings
# ==========================================================================
def _import_first(*candidates: str) -> ModuleType | None:
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    return None


def _import_sibling(file_stem: str) -> ModuleType | None:
    """Load a module from a file beside this one, or in ./controllers or ../controllers."""
    here = Path(__file__).parent
    roots = (here, here / "controllers", here.parent / "controllers")
    candidates = [
        root / f"{file_stem}{suffix}.py"
        for root in roots
        for suffix in ("", "(3)", "(2)")
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"_harness_{file_stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return None


def load_base_module() -> ModuleType:
    module = _import_first(
        "proper_research.simulation.simulations."
        "simulate_time_parameterized_configuration_mpc",
        "simulate_time_parameterized_configuration_mpc",
    ) or _import_sibling("simulate_time_parameterized_configuration_mpc")
    if module is None:
        raise ModuleNotFoundError(
            "Could not import simulate_time_parameterized_configuration_mpc. "
            "Put this script beside it, or make the project package importable."
        )
    return module


def load_beam_output_module() -> ModuleType:
    module = _import_first(
        "proper_research.simulation.simulations."
        "simulate_time_parameterized_beam_output_mpc",
        "simulate_time_parameterized_beam_output_mpc",
    ) or _import_sibling("simulate_time_parameterized_beam_output_mpc")
    if module is None:
        raise ModuleNotFoundError(
            "Could not import simulate_time_parameterized_beam_output_mpc."
        )
    return module


# ==========================================================================
# controllers  (implementations live in the controllers package)
# ==========================================================================
# The four control laws are deliberately NOT defined here.  This file is the
# harness: one plant, one reference, one closed loop, one record schema.  Any
# controller detail living in the harness is a place where the harness could
# favour one rung over another, so they live in their own modules and this file
# only wires them up.
def _controllers_module(name: str) -> ModuleType:
    module = _import_first(
        f"proper_research.simulation.controllers.{name}",
        f"controllers.{name}",
        name,
    ) or _import_sibling(name)
    if module is None:
        raise ModuleNotFoundError(
            f"Could not import {name}. Put the controllers package beside this "
            "file or make it importable."
        )
    return module


@dataclass
class ControlOutput:
    """The minimum a controller must return; mirrors ConfigurationMPCStep."""

    command: Array
    planned_input: Array
    status: str = "solved"
    success: bool = True
    iterations: int = 0
    solve_time_s: float = 0.0
    objective: float = math.nan
    primal_residual: float = math.nan
    dual_residual: float = math.nan
    predicted_states: Array | None = None
    predicted_inputs: Array | None = None
    measured_beam_position: Array | None = None
    instantaneous_output_residual: Array | None = None
    estimated_output_residual: Array | None = None
    predicted_beam_positions: Array | None = None
    predicted_beam_errors: Array | None = None
    first_predicted_beam_error_m: float = math.nan


class BaseController:
    name = "base"
    description = ""

    def reset(self) -> None:
        return None

    def solve(
        self,
        *,
        measured_state: Array,
        measured_beam_position: Array,
        control_index: int,
        previous_input: Array,
    ) -> ControlOutput:
        raise NotImplementedError


class ControllerAdapter(BaseController):
    """Wraps any controller with a ``solve(...)`` in the harness protocol."""

    def __init__(self, name: str, description: str, controller: Any) -> None:
        self.name = name
        self.description = description
        self.controller = controller

    def reset(self) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()

    def solve(
        self, *, measured_state, measured_beam_position, control_index, previous_input
    ) -> ControlOutput:
        step = self.controller.solve(
            measured_state=measured_state,
            measured_beam_position=measured_beam_position,
            control_index=control_index,
            previous_input=previous_input,
        )
        return ControlOutput(
            command=np.asarray(step.command, dtype=float).reshape(7),
            planned_input=np.asarray(step.planned_input, dtype=float).reshape(7),
            status=str(step.status),
            success=bool(step.success),
            iterations=int(step.iterations),
            solve_time_s=float(step.solve_time_s),
            objective=float(step.objective),
            primal_residual=float(getattr(step, "primal_residual", math.nan)),
            dual_residual=float(getattr(step, "dual_residual", math.nan)),
            predicted_states=getattr(step, "predicted_states", None),
            predicted_inputs=getattr(step, "predicted_inputs", None),
            measured_beam_position=getattr(step, "measured_beam_position", None),
            instantaneous_output_residual=getattr(
                step, "instantaneous_output_residual", None
            ),
            estimated_output_residual=getattr(step, "estimated_output_residual", None),
            predicted_beam_positions=getattr(step, "predicted_beam_positions", None),
            predicted_beam_errors=getattr(step, "predicted_beam_errors", None),
            first_predicted_beam_error_m=float(
                getattr(step, "first_predicted_beam_error_m", math.nan)
            ),
        )


# Kept as an alias so older call sites still resolve.
MPCControllerAdapter = ControllerAdapter


# ==========================================================================
# shared closed loop
# ==========================================================================
@dataclass
class LoopConfig:
    sample_period_s: float
    velocity_limit: Array
    acceleration_limit: Array
    state_min: Array
    state_max: Array
    position_tolerance_m: float
    tangent_tolerance_rad: float
    terminal_hold_steps: int = 15
    maximum_control_steps: int | None = None
    stop_on_qp_failure: bool = False
    progress_stride: int = 50


@dataclass
class LoopResult:
    name: str
    description: str
    records: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _unit(vector: Array) -> Array:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-15 else vector


def _record(
    *,
    sample: int,
    reference_index: int,
    reference: Any,
    state: Array,
    command: Array,
    previous_command: Array,
    output: ControlOutput | None,
    achieved_position: Array,
    achieved_tangent: Array,
    magnet_position: Array,
    magnet_quaternion: Array,
    config: LoopConfig,
) -> dict[str, Any]:
    """One CSV row in the schema of the existing MPC simulation output."""
    desired_position = np.asarray(reference.desired_position_m, dtype=float)[reference_index]
    desired_tangent = np.asarray(reference.desired_tangent, dtype=float)[reference_index]
    reference_state = np.asarray(reference.state, dtype=float)[reference_index]
    reference_input = np.asarray(reference.input, dtype=float)[reference_index]
    position_error = float(np.linalg.norm(achieved_position - desired_position))
    tangent_error = float(
        np.arccos(
            np.clip(float(_unit(achieved_tangent) @ _unit(desired_tangent)), -1.0, 1.0)
        )
    )
    row: dict[str, Any] = {
        "simulation_sample": int(sample),
        "time_s": float(sample) * config.sample_period_s,
        "reference_index": int(reference_index),
        "reference_time_s": float(np.asarray(reference.time_s, dtype=float)[reference_index]),
        "reference_path_s_m": float(
            np.asarray(reference.path_coordinate_m, dtype=float)[reference_index]
        ),
        "reference_is_terminal_hold": int(reference_index >= reference.sample_count - 1),
        "qp_status": "initial" if output is None else output.status,
        "qp_success": 1 if (output is None or output.success) else 0,
        "qp_iterations": 0 if output is None else int(output.iterations),
        "qp_solve_time_s": 0.0 if output is None else float(output.solve_time_s),
        "qp_objective": math.nan if output is None else float(output.objective),
        "qp_primal_residual": math.nan if output is None else float(output.primal_residual),
        "qp_dual_residual": math.nan if output is None else float(output.dual_residual),
        "beam_position_error_m": position_error,
        "beam_position_error_mm": 1e3 * position_error,
        "beam_tangent_error_rad": tangent_error,
        "beam_tangent_error_deg": math.degrees(tangent_error),
        "beam_feasible": int(
            position_error <= config.position_tolerance_m
            and tangent_error <= config.tangent_tolerance_rad
        ),
    }
    for prefix, vector in (
        ("desired", desired_position),
        ("achieved", achieved_position),
    ):
        for axis, value in zip("xyz", np.asarray(vector, dtype=float)):
            row[f"{prefix}_{axis}"] = float(value)
    for prefix, vector in (
        ("desired_tangent", desired_tangent),
        ("achieved_tangent", achieved_tangent),
    ):
        for axis, value in zip("xyz", np.asarray(vector, dtype=float)):
            row[f"{prefix}_{axis}"] = float(value)
    for axis, value in zip("xyz", np.asarray(magnet_position, dtype=float)):
        row[f"magnet_{axis}"] = float(value)
    for axis, value in zip(("qx", "qy", "qz", "qw"), np.asarray(magnet_quaternion, dtype=float)):
        row[f"magnet_{axis}"] = float(value)
    for index in range(6):
        row[f"q{index + 1}_actual"] = float(state[index])
        row[f"q{index + 1}_reference"] = float(reference_state[index])
        row[f"q{index + 1}_error"] = float(state[index] - reference_state[index])
    row["insertion_actual"] = float(state[6])
    row["insertion_reference"] = float(reference_state[6])
    row["insertion_error"] = float(state[6] - reference_state[6])
    acceleration = (command - previous_command) / config.sample_period_s
    for index in range(6):
        row[f"qd{index + 1}_command"] = float(command[index])
        row[f"qd{index + 1}_reference"] = float(reference_input[index])
        row[f"qd{index + 1}_error"] = float(command[index] - reference_input[index])
        row[f"qd{index + 1}_acceleration"] = float(acceleration[index])
    row["insertion_rate_command"] = float(command[6])
    row["insertion_rate_reference"] = float(reference_input[6])
    row["insertion_rate_error"] = float(command[6] - reference_input[6])
    row["insertion_rate_acceleration"] = float(acceleration[6])
    residual = (
        np.full(3, np.nan) if output is None or output.estimated_output_residual is None
        else np.asarray(output.estimated_output_residual, dtype=float)
    )
    instantaneous = (
        np.full(3, np.nan)
        if output is None or output.instantaneous_output_residual is None
        else np.asarray(output.instantaneous_output_residual, dtype=float)
    )
    predicted = (
        np.full(3, np.nan)
        if output is None or output.predicted_beam_positions is None
        else np.asarray(output.predicted_beam_positions, dtype=float)[0]
    )
    predicted_error = (
        np.full(3, np.nan)
        if output is None or output.predicted_beam_errors is None
        else np.asarray(output.predicted_beam_errors, dtype=float)[0]
    )
    for axis_index, axis in enumerate("xyz"):
        row[f"beam_residual_{axis}_m"] = float(residual[axis_index])
        row[f"beam_residual_instantaneous_{axis}_m"] = float(instantaneous[axis_index])
        row[f"predicted_next_beam_{axis}_m"] = float(predicted[axis_index])
        row[f"predicted_next_beam_error_{axis}_m"] = float(predicted_error[axis_index])
    row["beam_residual_norm_m"] = float(np.linalg.norm(residual))
    row["predicted_next_beam_error_m"] = (
        math.nan if output is None else float(output.first_predicted_beam_error_m)
    )
    return row


def run_closed_loop(
    *,
    controller: BaseController,
    reference: Any,
    beam_snapshot: Callable[[Array, bool], tuple[Array, Array]],
    magnet_pose: Callable[[Array], tuple[Array, Array]],
    reset_plant: Callable[[], None] | None,
    config: LoopConfig,
    initial_state: Array,
    verbose: bool = True,
) -> LoopResult:
    """One simulation loop, shared by every controller.

    Sharing this function is what makes the comparison fair: the plant is
    stepped and committed identically regardless of which control law produced
    the command.
    """
    if reset_plant is not None:
        reset_plant()
    controller.reset()

    state = np.asarray(initial_state, dtype=float).reshape(7).copy()
    previous_command = np.zeros(7, dtype=float)
    achieved_position, achieved_tangent = beam_snapshot(state, True)
    magnet_position, magnet_quaternion = magnet_pose(state)
    records = [
        _record(
            sample=0, reference_index=0, reference=reference, state=state,
            command=previous_command, previous_command=previous_command, output=None,
            achieved_position=achieved_position, achieved_tangent=achieved_tangent,
            magnet_position=magnet_position, magnet_quaternion=magnet_quaternion,
            config=config,
        )
    ]

    steps = reference.sample_count - 1 + int(config.terminal_hold_steps)
    if config.maximum_control_steps is not None:
        steps = min(steps, int(config.maximum_control_steps))

    stopped_reason = "completed"
    for control_index in range(steps):
        output = controller.solve(
            measured_state=state,
            measured_beam_position=achieved_position,
            control_index=control_index,
            previous_input=previous_command,
        )
        if not output.success:
            stopped_reason = f"controller_failure:{output.status}"
            if config.stop_on_qp_failure:
                break
        command = np.asarray(output.command, dtype=float).reshape(7)
        state = state + config.sample_period_s * command
        old_command = previous_command.copy()
        previous_command = command.copy()
        reference_index = min(control_index + 1, reference.sample_count - 1)
        achieved_position, achieved_tangent = beam_snapshot(state, True)
        magnet_position, magnet_quaternion = magnet_pose(state)
        records.append(
            _record(
                sample=control_index + 1, reference_index=reference_index,
                reference=reference, state=state, command=command,
                previous_command=old_command, output=output,
                achieved_position=achieved_position, achieved_tangent=achieved_tangent,
                magnet_position=magnet_position, magnet_quaternion=magnet_quaternion,
                config=config,
            )
        )
        if verbose and (
            control_index == 0
            or (control_index + 1) % config.progress_stride == 0
            or control_index + 1 == steps
        ):
            print(
                f"[{controller.name}] step {control_index + 1}/{steps} "
                f"tip={records[-1]['beam_position_error_mm']:.4f} mm "
                f"solve={1e3 * output.solve_time_s:.2f} ms",
                flush=True,
            )

    errors = np.asarray([row["beam_position_error_m"] for row in records], dtype=float)
    tangents = np.asarray([row["beam_tangent_error_rad"] for row in records], dtype=float)
    solve_ms = 1e3 * np.asarray([row["qp_solve_time_s"] for row in records], dtype=float)
    successes = np.asarray([row["qp_success"] for row in records], dtype=float) > 0.5
    joint_errors = np.asarray(
        [[row[f"q{i}_error"] for i in range(1, 7)] for row in records], dtype=float
    )
    summary = {
        "controller": controller.name,
        "description": controller.description,
        "completed_requested_steps": bool(len(records) == steps + 1),
        "stopped_reason": stopped_reason,
        "all_solves_succeeded": bool(np.all(successes)),
        "solve_failures": int(np.sum(~successes)),
        "samples": int(len(records)),
        "duration_s": float(records[-1]["time_s"]),
        "maximum_beam_position_error_mm": float(1e3 * np.nanmax(errors)),
        "rms_beam_position_error_mm": float(1e3 * C.rms(errors)),
        "p95_beam_position_error_mm": float(1e3 * C.percentile(errors, 95.0)),
        "final_beam_position_error_mm": float(1e3 * errors[-1]),
        "maximum_beam_tangent_error_deg": float(math.degrees(np.nanmax(tangents))),
        "maximum_absolute_joint_error_deg": float(math.degrees(C.max_abs(joint_errors))),
        "rms_joint_error_deg": float(math.degrees(C.rms(joint_errors))),
        "mean_solve_time_ms": float(np.nanmean(solve_ms[1:])) if len(solve_ms) > 1 else 0.0,
        "p95_solve_time_ms": float(C.percentile(solve_ms[1:], 95.0)) if len(solve_ms) > 1 else 0.0,
        "maximum_solve_time_ms": float(np.nanmax(solve_ms[1:])) if len(solve_ms) > 1 else 0.0,
        "sample_period_ms": float(1e3 * config.sample_period_s),
        "deadline_overruns": int(np.sum(solve_ms[1:] > 1e3 * config.sample_period_s)),
        "position_tolerance_mm": float(1e3 * config.position_tolerance_m),
        "beam_feasible_fraction": float(
            np.mean([row["beam_feasible"] for row in records])
        ),
    }
    return LoopResult(controller.name, controller.description, records, summary)


def save_loop_result(result: LoopResult, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    header = list(result.records[0].keys())
    C.write_csv(
        output_dir / "configuration_mpc_simulation.csv",
        header,
        [[row.get(key, "") for key in header] for row in result.records],
    )
    (output_dir / "configuration_mpc_simulation_summary.json").write_text(
        json.dumps(C.jsonable(result.summary), indent=2), encoding="utf-8"
    )
    return {
        "csv": output_dir / "configuration_mpc_simulation.csv",
        "json": output_dir / "configuration_mpc_simulation_summary.json",
    }


# ==========================================================================
# comparison report
# ==========================================================================
def build_comparison_report(
    results: Sequence[LoopResult], context: dict[str, Any]
) -> C.Report:
    report = C.Report("Controller comparison — beam-tip tracking")
    report.metrics = {
        "context": context,
        "controllers": {result.name: result.summary for result in results},
    }

    report.head("Experiment controls")
    report.text(
        "Everything below differs only in the control law. Same reference, same "
        "plant, same initial state, same limits, same tolerances, same terminal "
        "hold, and the same closed-loop function stepping the plant."
    )
    report.bullets(
        [
            f"Reference: `{context.get('reference_source')}` — "
            f"{context.get('reference_samples')} samples, "
            f"{context.get('reference_duration_s', float('nan')):.3f} s, "
            f"dt = {1e3 * context.get('sample_period_s', float('nan')):.2f} ms.",
            f"Jacobian model: **{context.get('jacobian_model')}** — "
            f"{context.get('jacobian_note')}",
            f"Plant model: **{context.get('plant_model')}**.",
            f"Prediction horizon (MPC controllers): {context.get('prediction_horizon')}.",
            f"Position tolerance {1e3 * context.get('position_tolerance_m', float('nan')):.2f} mm.",
        ]
    )

    report.head("Results")
    rows = []
    for result in results:
        s = result.summary
        rows.append(
            [
                result.name,
                s["rms_beam_position_error_mm"],
                s["p95_beam_position_error_mm"],
                s["maximum_beam_position_error_mm"],
                s["maximum_absolute_joint_error_deg"],
                s["mean_solve_time_ms"],
                s["maximum_solve_time_ms"],
                s["deadline_overruns"],
                "yes" if s["all_solves_succeeded"] else f"{s['solve_failures']} failed",
            ]
        )
    report.table(
        ["controller", "tip rms [mm]", "tip p95 [mm]", "tip max [mm]",
         "joint max [deg]", "solve mean [ms]", "solve max [ms]",
         "deadline overruns", "all solved"],
        rows,
    )

    finite = [r for r in results if np.isfinite(r.summary["rms_beam_position_error_mm"])]
    if finite:
        best = min(finite, key=lambda r: r.summary["rms_beam_position_error_mm"])
        fastest = min(finite, key=lambda r: r.summary["maximum_solve_time_ms"])
        report.finding(
            "info",
            f"Most accurate: {best.name}",
            f"tip rms {best.summary['rms_beam_position_error_mm']:.4f} mm, "
            f"worst-case solve {best.summary['maximum_solve_time_ms']:.2f} ms.",
        )
        report.finding(
            "info",
            f"Cheapest worst case: {fastest.name}",
            f"worst-case solve {fastest.summary['maximum_solve_time_ms']:.2f} ms, "
            f"tip rms {fastest.summary['rms_beam_position_error_mm']:.4f} mm.",
        )
        by_name = {r.name: r.summary for r in results}
        if "mpc_ltv_offline" in by_name and "mpc_ltv_sqp_online" in by_name:
            offline = by_name["mpc_ltv_offline"]["rms_beam_position_error_mm"]
            online = by_name["mpc_ltv_sqp_online"]["rms_beam_position_error_mm"]
            gain = C.improvement_percent(offline, online)
            cost = (
                by_name["mpc_ltv_sqp_online"]["maximum_solve_time_ms"]
                / max(1e-9, by_name["mpc_ltv_offline"]["maximum_solve_time_ms"])
            )
            level = "ok" if gain > 5 else "warn"
            report.finding(
                level,
                f"Online relinearisation changes tip rms by {gain:+.1f}% for "
                f"{cost:.1f}x the worst-case solve time",
                "If the gain is small, the offline schedule is already a good local "
                "model over one sample period and the extra machinery is not "
                "earning its deadline risk. The layer-4 one-step prediction check "
                "is the cheap way to predict this result before running it.",
            )
        for name, summary in by_name.items():
            if summary["deadline_overruns"]:
                report.finding(
                    "warn",
                    f"{name} misses its deadline",
                    f"{summary['deadline_overruns']} solve(s) exceeded the "
                    f"{summary['sample_period_ms']:.1f} ms sample period "
                    f"(worst {summary['maximum_solve_time_ms']:.1f} ms). "
                    "An accuracy win that cannot be computed in time is not a win.",
                )

    report.head("How to read this")
    report.bullets(
        [
            "**naive -> LTI** prices preview and constraint handling. If the naive "
            "controller is close, the horizon is not buying much on this trajectory.",
            "**LTI -> LTV offline** prices scheduling the model along the path. A "
            "large gap means the Jacobian varies materially over the run.",
            "**LTV offline -> SQP online** prices relinearising on the measured "
            "state. A large gap means the plant leaves the neighbourhood the "
            "offline linearisation is valid in — which is exactly what the "
            "contact-free Jacobian is designed to provoke.",
            "Read accuracy and worst-case solve time together, never separately.",
        ]
    )
    return report


# ==========================================================================
# self-test with mock objects
# ==========================================================================
class _MockReference:
    """A ConfigurationReference-shaped object, enough for the harness."""

    def __init__(self, samples: int = 120, dt: float = 0.01) -> None:
        self.sample_count = samples
        self.sample_period_s = dt
        self.duration_s = dt * (samples - 1)
        self.source = "self_test"
        t = np.arange(samples) * dt
        self.time_s = t
        self.path_coordinate_m = 0.01 * t
        base = np.zeros((samples, 7))
        base[:, 0] = 0.15 * np.sin(2 * np.pi * 0.25 * t)
        base[:, 1] = 0.10 * np.cos(2 * np.pi * 0.20 * t)
        base[:, 6] = 0.004 * t
        self.state = base
        self.input = np.vstack([np.zeros((1, 7)), np.diff(base, axis=0) / dt])
        self.acceleration = np.zeros_like(base)
        self.desired_position_m = np.stack(
            [0.30 * base[:, 0], 0.20 * base[:, 1], 5.0 * base[:, 6]], axis=1
        )
        self.desired_tangent = np.tile(np.array([1.0, 0.0, 0.0]), (samples, 1))
        self.planned_beam_feasible = np.ones(samples, dtype=bool)
        self.planned_position_error_m = np.zeros(samples)
        self.planned_tangent_error_rad = np.zeros(samples)

    def state_window(self, index: int, horizon: int) -> Array:
        indices = np.clip(index + 1 + np.arange(horizon), 0, self.sample_count - 1)
        return self.state[indices]

    def input_window(self, index: int, horizon: int) -> Array:
        indices = np.clip(index + 1 + np.arange(horizon), 0, self.sample_count - 1)
        return self.input[indices]


class _MockPlant:
    """A mildly nonlinear tip map with a 'contact' term the Jacobian can omit."""

    def __init__(self, contact: bool = True) -> None:
        self.contact = contact
        self.use_contact_in_jacobian = True

    def tip(self, state: Array) -> Array:
        q = np.asarray(state, dtype=float)
        base = np.array([0.30 * q[0], 0.20 * q[1], 5.0 * q[6]])
        stiffening = np.array([0.04 * q[0] ** 3, 0.02 * q[1] ** 3, 0.0])
        contact = np.array([0.010 * np.tanh(6.0 * q[0]), 0.0, 0.0]) if self.contact else 0.0
        return base + stiffening + contact

    def jacobian(self, state: Array) -> Array:
        q = np.asarray(state, dtype=float)
        matrix = np.zeros((3, 7))
        matrix[0, 0] = 0.30 + 0.12 * q[0] ** 2
        matrix[1, 1] = 0.20 + 0.06 * q[1] ** 2
        matrix[2, 6] = 5.0
        if self.contact and self.use_contact_in_jacobian:
            matrix[0, 0] += 0.060 / np.cosh(6.0 * q[0]) ** 2
        return matrix


def run_self_test() -> None:
    print("[harness self-test] building mock reference and plant")
    reference = _MockReference()
    plant = _MockPlant(contact=True)
    plant.use_contact_in_jacobian = False  # the declared model mismatch

    config = LoopConfig(
        sample_period_s=reference.sample_period_s,
        velocity_limit=np.array([2.0] * 6 + [0.5]),
        acceleration_limit=np.array([20.0] * 6 + [5.0]),
        state_min=np.array([-3.0] * 6 + [-1.0]),
        state_max=np.array([3.0] * 6 + [1.0]),
        position_tolerance_m=1.5e-3,
        tangent_tolerance_rad=math.pi / 2,
        terminal_hold_steps=5,
        progress_stride=1000,
    )
    inverse_module = _controllers_module("inverse_jacobian_controller")
    controller = ControllerAdapter(
        "naive_inverse_jacobian",
        "Damped resolved-rate inverse Jacobian, nullspace-regulated",
        inverse_module.InverseJacobianBeamController(
            reference=reference,
            jacobian_provider=plant.jacobian,
            sample_period_s=config.sample_period_s,
            velocity_limit=config.velocity_limit,
            acceleration_limit=config.acceleration_limit,
            state_min=config.state_min,
            state_max=config.state_max,
        ),
    )
    result = run_closed_loop(
        controller=controller,
        reference=reference,
        beam_snapshot=lambda state, commit: (
            plant.tip(state), np.array([1.0, 0.0, 0.0])
        ),
        magnet_pose=lambda state: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])),
        reset_plant=None,
        config=config,
        initial_state=reference.state[0] + np.array([0.02, -0.02, 0, 0, 0, 0, 1e-3]),
        verbose=False,
    )
    summary = result.summary
    assert summary["samples"] > 100, "loop did not run"
    assert np.isfinite(summary["rms_beam_position_error_mm"]), "no tracking metric"
    assert summary["all_solves_succeeded"], "resolved-rate control reported a failure"
    header = list(result.records[0].keys())
    for required in (
        "beam_position_error_mm", "q1_actual", "qd1_command",
        "achieved_x", "desired_x", "magnet_qw", "predicted_next_beam_error_m",
    ):
        assert required in header, f"record schema is missing {required}"
    report = build_comparison_report(
        [result],
        {
            "reference_source": "self_test", "reference_samples": reference.sample_count,
            "reference_duration_s": reference.duration_s,
            "sample_period_s": reference.sample_period_s,
            "jacobian_model": "contact-free (mock)",
            "jacobian_note": "contact term omitted from the mock Jacobian",
            "plant_model": "contact enabled (mock)",
            "prediction_horizon": "n/a", "position_tolerance_m": config.position_tolerance_m,
        },
    )
    assert "Controller comparison" in report.render()
    print(
        f"[harness self-test] PASS — {summary['samples']} samples, "
        f"tip rms {summary['rms_beam_position_error_mm']:.4f} mm, "
        f"final {summary['final_beam_position_error_mm']:.4f} mm, "
        f"schema {len(header)} columns"
    )


# ==========================================================================
# project entry point
# ==========================================================================
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--controllers", nargs="+", default=list(CONTROLLER_NAMES),
        choices=list(CONTROLLER_NAMES),
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--terminal-hold-steps", type=int, default=15)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=90.0)
    parser.add_argument("--beam-position-scale-mm", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--beam-position-weight", type=float, default=1.0)
    parser.add_argument("--beam-terminal-multiplier", type=float, default=20.0)
    parser.add_argument("--state-tracking-weight", type=float, default=1.0e-2)
    parser.add_argument("--input-tracking-weight", type=float, default=1.0e-3)
    parser.add_argument("--input-increment-weight", type=float, default=1.0e-3)
    parser.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    parser.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    parser.add_argument("--velocity-safety-factor", type=float, default=0.8)
    parser.add_argument("--acceleration-safety-factor", type=float, default=0.8)
    parser.add_argument("--sqp-inner-iterations", type=int, default=1)
    parser.add_argument("--sqp-relinearise-horizon", action="store_true")
    parser.add_argument("--lti-freeze-index", type=int, default=0)
    parser.add_argument(
        "--jacobian-model",
        choices=("no_contact", "contact", "flag", "pack"),
        default="no_contact",
        help="Where the controllers' Jacobian comes from. 'no_contact' (default) "
             "differentiates bundle.models['no_contact'] - a separate, "
             "structurally contact-free beam model. 'flag' instead forces "
             "contact_cfg.use_in_jacobian=False on the plant's own adapter, "
             "differentiating the contact-free energy at the contact "
             "equilibrium. 'contact' is the control condition; 'pack' uses "
             "whatever the pack was built with.",
    )
    parser.add_argument("--resolved-rate-gain", type=float, default=1.0)
    parser.add_argument("--resolved-rate-damping", type=float, default=1.0e-3)
    parser.add_argument("--nullspace-gain", type=float, default=1.0)
    parser.add_argument("--initial-joint-offset-deg", type=float, nargs=6, default=(0.0,) * 6)
    parser.add_argument("--initial-insertion-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument(
        "--allow-contact-jacobian", action="store_true",
        help="Proceed even if the contact-free Jacobian toggle is unavailable.",
    )
    parser.add_argument("--allow-planned-beam-failure", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.self_test:
        run_self_test()
        return

    base = load_base_module()
    beam_module = load_beam_output_module()
    from proper_research.planning.planning_context import build_planning_context

    _, bundle, controller_pack, out_root = build_planning_context()
    reference_source = args.reference or (out_root / "time_parameterized_configuration_path")
    reference = base.load_configuration_reference(
        reference_source,
        require_planned_beam_feasible=not args.allow_planned_beam_failure,
    )
    output_dir = Path(args.output_dir or (out_root / "controller_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = controller_pack.get("plant_diagnostic_joint_adapter")
    if adapter is None:
        raise KeyError("controller_pack has no 'plant_diagnostic_joint_adapter'.")
    forward6d = controller_pack.get("forward6d_plant")
    if forward6d is None:
        raise KeyError("controller_pack has no 'forward6d_plant'.")

    mpc_config = base.make_default_mpc_config(
        reference=reference,
        controller_pack=controller_pack,
        prediction_horizon=args.horizon,
        joint_acceleration_limit_rad_s2=args.joint_acceleration_limit,
        insertion_acceleration_limit_m_s2=args.insertion_acceleration_limit,
        velocity_safety_factor=args.velocity_safety_factor,
        acceleration_safety_factor=args.acceleration_safety_factor,
        solver_backend=args.solver,
        solver_verbose=False,
    )
    mpc_config = replace(
        mpc_config,
        state_tracking_weight=float(args.state_tracking_weight),
        input_tracking_weight=float(args.input_tracking_weight),
        input_increment_weight=float(args.input_increment_weight),
    )
    beam_config = beam_module.BeamOutputMPCConfig(
        position_error_scale_m=tuple(
            1e-3 * np.asarray(args.beam_position_scale_mm, dtype=float)
        ),
        position_tracking_weight=float(args.beam_position_weight),
        terminal_weight_multiplier=float(args.beam_terminal_multiplier),
    )
    loop_config = LoopConfig(
        sample_period_s=float(reference.sample_period_s),
        velocity_limit=np.asarray(mpc_config.velocity_limit, dtype=float),
        acceleration_limit=np.asarray(mpc_config.acceleration_limit, dtype=float),
        state_min=np.asarray(mpc_config.state_min, dtype=float),
        state_max=np.asarray(mpc_config.state_max, dtype=float),
        position_tolerance_m=1e-3 * float(args.position_tolerance_mm),
        tangent_tolerance_rad=math.radians(float(args.tangent_tolerance_deg)),
        terminal_hold_steps=int(args.terminal_hold_steps),
        maximum_control_steps=args.max_control_steps,
    )
    initial_state = np.asarray(reference.state[0], dtype=float) + np.concatenate(
        (
            np.radians(np.asarray(args.initial_joint_offset_deg, dtype=float)),
            [1e-3 * float(args.initial_insertion_offset_mm)],
        )
    )

    # ------------------------------------------------------------------
    # Jacobians: contact-free for every controller, contact plant throughout.
    # ------------------------------------------------------------------
    providers_module = _controllers_module("beam_jacobian_providers")
    inverse_module = _controllers_module("inverse_jacobian_controller")
    variants_module = _controllers_module("mpc_variants")

    jacobian_provider = None
    contact_scope = None
    provider_note = ""
    if args.jacobian_model in {"no_contact", "contact"} and bundle is not None:
        jacobian_provider = providers_module.from_model_bundle(
            bundle=bundle,
            controller_pack=controller_pack,
            contact=(args.jacobian_model == "contact"),
        )
        provider_note = (
            "a dedicated diagnostic adapter around "
            f"bundle.models[{args.jacobian_model!r}] - structurally "
            + ("contact-aware" if args.jacobian_model == "contact" else "contact-free")
        )
    elif args.jacobian_model == "pack":
        jacobian_provider = providers_module.from_controller_pack(controller_pack)
        provider_note = "the pack's own jacobian adapter, as built"
    else:
        # Flag route: differentiate the contact-free energy at the contact
        # equilibrium, on the plant's own adapter.
        jacobian_provider = providers_module.from_controller_pack(
            controller_pack, prefer="plant"
        )
        contact_scope = providers_module.contact_free_jacobian_scope(
            jacobian_provider.model, required=not args.allow_contact_jacobian
        )
        provider_note = (
            "the plant adapter with contact_cfg.use_in_jacobian forced False - "
            "the contact-free energy differentiated at the contact equilibrium"
        )

    results: list[LoopResult] = []
    scope = contact_scope if contact_scope is not None else contextlib.nullcontext(False)
    with scope as flag_applied:
        print(
            "[comparison] Jacobian model: "
            + jacobian_provider.summary_line()
            + (f"\n[comparison] route: {provider_note}" if provider_note else ""),
            flush=True,
        )
        if contact_scope is not None and not flag_applied:
            print(
                "[comparison] WARNING: the contact-free toggle was unavailable. "
                "This is NOT the declared contact-free experiment.",
                flush=True,
            )

        schedule = variants_module.precompute_schedule(
            reference=reference, jacobian_provider=jacobian_provider
        )

        # How different is the contact-free model from the contact one? If the
        # answer is 'barely', the comparison has little to reject and four
        # near-identical controllers would mean nothing.
        agreement = None
        if bundle is not None and args.jacobian_model != "pack":
            try:
                contact_provider = providers_module.from_model_bundle(
                    bundle=bundle, controller_pack=controller_pack, contact=True
                )
                agreement = providers_module.compare_providers(
                    contact_provider,
                    jacobian_provider,
                    np.asarray(reference.state, dtype=float)[::max(
                        1, reference.sample_count // 40
                    )],
                )
                print(
                    "[comparison] contact vs contact-free Jacobian: relative "
                    f"max {agreement['relative_difference_max']:.4f}, "
                    f"rms {agreement['relative_difference_rms']:.4f}",
                    flush=True,
                )
            except Exception as error:  # diagnostics must not stop the run
                agreement = {"error": f"{type(error).__name__}: {error}"}

        mpc_variants = variants_module.build_all_mpc_variants(
            reference=reference,
            config=mpc_config,
            beam_config=beam_config,
            jacobian_provider=jacobian_provider,
            schedule=schedule,
            freeze_index=int(args.lti_freeze_index),
            sqp_inner_iterations=int(args.sqp_inner_iterations),
            relinearise_horizon=bool(args.sqp_relinearise_horizon),
        )

        def build(name: str) -> BaseController:
            if name == "naive_inverse_jacobian":
                return ControllerAdapter(
                    name,
                    "Damped resolved-rate inverse Jacobian, nullspace-regulated",
                    inverse_module.build_inverse_jacobian_controller(
                        reference=reference,
                        jacobian_provider=jacobian_provider,
                        mpc_config=mpc_config,
                        position_gain=float(args.resolved_rate_gain),
                        damping=float(args.resolved_rate_damping),
                        nullspace_gain=float(args.nullspace_gain),
                    ),
                )
            controller = mpc_variants[name]
            return ControllerAdapter(
                name, str(controller.variant_description), controller
            )

        def beam_snapshot(state: Array, commit: bool) -> tuple[Array, Array]:
            return base._beam_snapshot(
                forward6d=forward6d, state=state, commit=commit
            )

        def magnet_pose(state: Array) -> tuple[Array, Array]:
            return base._magnet_pose(controller_pack=controller_pack, state=state)

        reset_plant = getattr(adapter, "reset_to_initial_baseline", None)
        if reset_plant is None:
            forward_adapter = getattr(
                getattr(adapter, "beam_output_fn", None), "forward_adapter", None
            )
            reset_plant = getattr(forward_adapter, "reset_to_initial_baseline", None)

        for name in args.controllers:
            print(f"[comparison] running {name}", flush=True)
            result = run_closed_loop(
                controller=build(name),
                reference=reference,
                beam_snapshot=beam_snapshot,
                magnet_pose=magnet_pose,
                reset_plant=reset_plant,
                config=loop_config,
                initial_state=initial_state,
                verbose=not args.quiet,
            )
            save_loop_result(result, output_dir / name)
            results.append(result)

    context = {
        "reference_source": str(reference_source),
        "reference_samples": int(reference.sample_count),
        "reference_duration_s": float(reference.duration_s),
        "sample_period_s": float(reference.sample_period_s),
        "prediction_horizon": int(mpc_config.prediction_horizon),
        "position_tolerance_m": loop_config.position_tolerance_m,
        "jacobian_model": args.jacobian_model,
        "jacobian_note": provider_note,
        "jacobian_provenance": jacobian_provider.describe(),
        "contact_free_jacobian_agreement": agreement,
        "plant_model": "contact enabled (the plant is never modified)",
        "lti_freeze_index": int(args.lti_freeze_index),
        "sqp_inner_iterations": int(args.sqp_inner_iterations),
    }
    report = build_comparison_report(results, context)
    report.write(output_dir, "comparison")
    C.write_csv(
        output_dir / "comparison_summary.csv",
        list(results[0].summary.keys()),
        [[result.summary.get(key, "") for key in results[0].summary] for result in results],
    )
    print(f"\nComparison -> {output_dir / 'comparison_report.md'}")
    for finding in report.findings:
        print("   " + finding.line().replace("- **", "").replace("**", ""))


if __name__ == "__main__":
    main()
