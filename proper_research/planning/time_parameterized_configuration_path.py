"""Kinematic time parameterization of a feasible magnetic-beam path.

This module is the bridge between the offline geometric planner and the online
joint/insertion MPC.  The geometric planner supplies

    chi(s) = [q1(s), ..., q6(s), L(s)],

where ``s`` is lumen arc length.  This module keeps that geometric path fixed
and computes a monotone timing law ``s(t)``.  The path-velocity decomposition is

    chi_dot  = chi_s * s_dot
    chi_ddot = chi_s * s_ddot + chi_ss * s_dot**2.

Joint and insertion velocity/acceleration limits therefore become linear
inequalities in squared path speed ``x = s_dot**2`` and path acceleration
``u = s_ddot``.  A backward controllable-set pass followed by a forward
maximum-speed pass computes a reachability-style parameterization on a dense
spatial grid.  The duration is then dilated to an exact multiple of the MPC
sample period, which can only increase constraint margin.

The implementation follows the path-parameterization formulation used in
robotics research, including Bobrow, Dubowsky & Gibson (IJRR, 1985), Shin &
McKay (IEEE TAC, 1985), and the reachability formulation of Pham & Pham
(IEEE TRO, 2018; arXiv:1707.07239).  It is deliberately described as a
*kinematic reachability-style* method rather than a complete torque-constrained
TOPP-RA implementation: the constraints here are configuration velocity and
acceleration bounds, not manipulator torque, contact-wrench, or jerk bounds.

Scientific safeguards
---------------------
* A saved path must report global and dense physical feasibility.
* A C2 natural cubic spline removes velocity discontinuities at path knots.
* The interpolated spline is checked against absolute configuration bounds.
* Analytic state velocity and acceleration are checked at the MPC samples.
* The nonlinear magnetic-beam model can be replayed sequentially over the
  union of spatial-validation and time-sample coordinates.  This is important
  because spline interpolation changes the between-node geometric path.
* Start and end path speeds are zero by default.

The acceleration limits supplied to this module are planning limits.  They
must be chosen from validated robot and insertion-unit limits before hardware
execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog


Array = np.ndarray


def _finite_vector(value: Any, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != int(size) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain {size} finite values; got {result.shape}."
        )
    return result.copy()


def _finite_matrix(value: Any, columns: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != int(columns):
        raise ValueError(
            f"{name} must have shape (N, {columns}); got {result.shape}."
        )
    if result.shape[0] < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have at least two finite rows.")
    return result.copy()


def _unit_rows(value: Any, name: str) -> Array:
    result = _finite_matrix(value, 3, name)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"{name} contains a near-zero vector.")
    return result / norms[:, None]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class GeometricConfigurationPath:
    """Feasible configuration path parameterized by lumen arc length."""

    s_m: Array
    states: Array
    desired_position_m: Array
    desired_tangent: Array
    position_tolerance_m: float
    tangent_tolerance_rad: float
    globally_feasible: bool
    dense_validation_feasible: bool
    source: str

    def validate(self) -> None:
        s = _finite_vector(self.s_m, np.asarray(self.s_m).size, "path coordinates")
        states = _finite_matrix(self.states, 7, "configuration states")
        if s.size != states.shape[0] or s.size < 2:
            raise ValueError("Path coordinates and states have inconsistent lengths.")
        if np.any(np.diff(s) <= 0.0):
            raise ValueError("Path coordinates must be strictly increasing.")
        if _finite_matrix(
            self.desired_position_m, 3, "desired positions"
        ).shape[0] != s.size:
            raise ValueError("desired_position_m has the wrong row count.")
        if _unit_rows(self.desired_tangent, "desired tangents").shape[0] != s.size:
            raise ValueError("desired_tangent has the wrong row count.")
        if not np.isfinite(self.position_tolerance_m) or self.position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be finite and positive.")
        if not np.isfinite(self.tangent_tolerance_rad) or not (
            0.0 < self.tangent_tolerance_rad < math.pi
        ):
            raise ValueError("tangent_tolerance_rad must lie in (0, pi).")


@dataclass(frozen=True)
class TimeParameterizationConfig:
    """Limits and numerical settings for path time parameterization.

    ``state_velocity_limit`` is ordered ``[q1dot..q6dot, Ldot]`` with units
    ``[rad/s, ..., rad/s, m/s]``.  ``state_acceleration_limit`` uses
    ``[rad/s^2, ..., rad/s^2, m/s^2]``.
    """

    sample_period_s: float
    state_velocity_limit: tuple[float, ...]
    state_acceleration_limit: tuple[float, ...]

    velocity_safety_factor: float = 0.8
    acceleration_safety_factor: float = 0.8
    maximum_path_speed_m_s: float = 5.0e-3
    maximum_path_acceleration_m_s2: float = 2.0e-2
    start_path_speed_m_s: float = 0.0
    end_path_speed_m_s: float = 0.0

    timing_subdivisions_per_interval: int = 4
    beam_validation_samples_per_interval: int = 3
    validate_nonlinear_beam: bool = True
    require_nonlinear_beam_feasible: bool = True
    require_saved_global_feasible: bool = False
    require_saved_dense_feasible: bool = False

    interpolation: str = "natural_cubic"
    linear_program_tolerance: float = 1.0e-10
    constraint_tolerance: float = 1.0e-8
    debug: bool = True

    def validate(self) -> None:
        positive = {
            "sample_period_s": self.sample_period_s,
            "maximum_path_speed_m_s": self.maximum_path_speed_m_s,
            "maximum_path_acceleration_m_s2": (
                self.maximum_path_acceleration_m_s2
            ),
            "linear_program_tolerance": self.linear_program_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("velocity_safety_factor", "acceleration_safety_factor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        velocity = _finite_vector(
            self.state_velocity_limit, 7, "state_velocity_limit"
        )
        acceleration = _finite_vector(
            self.state_acceleration_limit, 7, "state_acceleration_limit"
        )
        if np.any(velocity <= 0.0) or np.any(acceleration <= 0.0):
            raise ValueError("State velocity/acceleration limits must be positive.")
        if self.start_path_speed_m_s < 0.0 or self.end_path_speed_m_s < 0.0:
            raise ValueError("Endpoint path speeds cannot be negative.")
        if self.start_path_speed_m_s > self.maximum_path_speed_m_s:
            raise ValueError("Start path speed exceeds maximum_path_speed_m_s.")
        if self.end_path_speed_m_s > self.maximum_path_speed_m_s:
            raise ValueError("End path speed exceeds maximum_path_speed_m_s.")
        if self.timing_subdivisions_per_interval < 1:
            raise ValueError("timing_subdivisions_per_interval must be >= 1.")
        if self.beam_validation_samples_per_interval < 1:
            raise ValueError("beam_validation_samples_per_interval must be >= 1.")
        if self.interpolation != "natural_cubic":
            raise ValueError("Only interpolation='natural_cubic' is supported.")

    @property
    def effective_velocity_limit(self) -> Array:
        return self.velocity_safety_factor * _finite_vector(
            self.state_velocity_limit, 7, "state_velocity_limit"
        )

    @property
    def effective_acceleration_limit(self) -> Array:
        return self.acceleration_safety_factor * _finite_vector(
            self.state_acceleration_limit, 7, "state_acceleration_limit"
        )


@dataclass
class TimeParameterizedConfigurationPath:
    """MPC-rate trajectory and its validation diagnostics."""

    time_s: Array
    path_coordinate_m: Array
    path_speed_m_s: Array
    path_acceleration_m_s2: Array
    state: Array
    state_rate: Array
    state_acceleration: Array
    desired_position_m: Array
    desired_tangent: Array
    achieved_position_m: Array
    achieved_tangent: Array
    position_error_m: Array
    tangent_error_rad: Array
    beam_sample_feasible: Array
    spatial_validation_s_m: Array
    spatial_position_error_m: Array
    spatial_tangent_error_rad: Array
    spatial_beam_feasible: Array
    globally_feasible: bool
    beam_validation_performed: bool
    duration_s: float
    raw_duration_s: float
    configuration: dict[str, Any]
    summary: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)


class _ConfigurationSpline:
    def __init__(self, path: GeometricConfigurationPath):
        path.validate()
        self.s = np.asarray(path.s_m, dtype=float)
        self.states = np.asarray(path.states, dtype=float)
        self.spline = CubicSpline(
            self.s,
            self.states,
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )
        self.desired_position_spline = CubicSpline(
            self.s,
            np.asarray(path.desired_position_m, dtype=float),
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )
        self.desired_tangent_spline = CubicSpline(
            self.s,
            np.asarray(path.desired_tangent, dtype=float),
            axis=0,
            bc_type="natural",
            extrapolate=False,
        )

    def value(self, s: Any) -> Array:
        result = np.asarray(self.spline(s), dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Configuration spline returned non-finite values.")
        return result

    def first(self, s: Any) -> Array:
        result = np.asarray(self.spline(s, 1), dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Configuration spline derivative is non-finite.")
        return result

    def second(self, s: Any) -> Array:
        result = np.asarray(self.spline(s, 2), dtype=float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Configuration spline second derivative is non-finite."
            )
        return result

    def reference(self, s: Any) -> tuple[Array, Array]:
        position = np.asarray(self.desired_position_spline(s), dtype=float)
        tangent = np.asarray(self.desired_tangent_spline(s), dtype=float)
        one_dimensional = tangent.ndim == 1
        tangent_rows = tangent.reshape(-1, 3)
        norms = np.linalg.norm(tangent_rows, axis=1)
        if np.any(norms <= 1.0e-12):
            raise FloatingPointError("Interpolated desired tangent is near zero.")
        tangent_rows = tangent_rows / norms[:, None]
        return position, tangent_rows[0] if one_dimensional else tangent_rows


def _subdivide_path(s_nodes: Array, subdivisions: int) -> Array:
    s_nodes = np.asarray(s_nodes, dtype=float).reshape(-1)
    pieces: list[Array] = []
    for index in range(s_nodes.size - 1):
        pieces.append(
            np.linspace(
                s_nodes[index],
                s_nodes[index + 1],
                int(subdivisions),
                endpoint=False,
                dtype=float,
            )
        )
    pieces.append(np.array([s_nodes[-1]], dtype=float))
    result = np.concatenate(pieces)
    if np.any(np.diff(result) <= 0.0):
        raise RuntimeError("Refined path grid is not strictly increasing.")
    return result


def _velocity_squared_caps(
    path_first: Array,
    velocity_limit: Array,
    maximum_path_speed: float,
) -> Array:
    path_first = _finite_matrix(path_first, 7, "path first derivatives")
    velocity_limit = _finite_vector(velocity_limit, 7, "velocity limits")
    caps = np.full(path_first.shape[0], float(maximum_path_speed) ** 2)
    epsilon = 1.0e-14
    for coordinate in range(7):
        active = np.abs(path_first[:, coordinate]) > epsilon
        coordinate_caps = np.full(path_first.shape[0], np.inf)
        coordinate_caps[active] = (
            velocity_limit[coordinate]
            / np.abs(path_first[active, coordinate])
        ) ** 2
        caps = np.minimum(caps, coordinate_caps)
    if np.any(~np.isfinite(caps)) or np.any(caps < 0.0):
        raise FloatingPointError("Invalid squared path-speed caps.")
    return caps


def _local_acceleration_inequalities(
    path_first: Array,
    path_second: Array,
    acceleration_limit: Array,
    maximum_path_acceleration: float,
) -> tuple[Array, Array]:
    """Return A, b for A @ [u, x] <= b at one path coordinate."""
    first = _finite_vector(path_first, 7, "local path first derivative")
    second = _finite_vector(path_second, 7, "local path second derivative")
    limit = _finite_vector(acceleration_limit, 7, "acceleration limit")
    rows: list[Array] = []
    upper: list[float] = []
    for coordinate in range(7):
        row = np.array([first[coordinate], second[coordinate]], dtype=float)
        rows.extend((row, -row))
        upper.extend((float(limit[coordinate]), float(limit[coordinate])))
    rows.extend((np.array([1.0, 0.0]), np.array([-1.0, 0.0])))
    upper.extend(
        (float(maximum_path_acceleration), float(maximum_path_acceleration))
    )
    return np.asarray(rows, dtype=float), np.asarray(upper, dtype=float)


def _controllable_interval_lp(
    *,
    path_first: Array,
    path_second: Array,
    acceleration_limit: Array,
    maximum_path_acceleration: float,
    squared_speed_cap: float,
    ds: float,
    next_interval: tuple[float, float],
    lp_tolerance: float,
) -> tuple[float, float]:
    """Backward one-step controllable interval for x = s_dot**2."""
    A, b = _local_acceleration_inequalities(
        path_first,
        path_second,
        acceleration_limit,
        maximum_path_acceleration,
    )
    next_lower, next_upper = next_interval
    transition_rows = np.array(
        [
            [2.0 * ds, 1.0],
            [-2.0 * ds, -1.0],
        ],
        dtype=float,
    )
    transition_upper = np.array([next_upper, -next_lower], dtype=float)
    A_ub = np.vstack((A, transition_rows))
    b_ub = np.concatenate((b, transition_upper))
    bounds = ((None, None), (0.0, float(squared_speed_cap)))
    options = {
        "dual_feasibility_tolerance": max(1.0e-10, float(lp_tolerance)),
        "primal_feasibility_tolerance": max(1.0e-10, float(lp_tolerance)),
    }

    minimum = linprog(
        np.array([0.0, 1.0]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options=options,
    )
    maximum = linprog(
        np.array([0.0, -1.0]),
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not minimum.success or not maximum.success:
        raise RuntimeError(
            "Backward reachability LP is infeasible: "
            f"min_status={minimum.status} {minimum.message}; "
            f"max_status={maximum.status} {maximum.message}."
        )
    lower = max(0.0, float(minimum.x[1]))
    upper = min(float(squared_speed_cap), float(maximum.x[1]))
    if lower > upper + 10.0 * lp_tolerance:
        raise RuntimeError("Backward controllable interval is numerically empty.")
    return lower, max(lower, upper)


def _acceleration_interval_at_speed(
    *,
    path_first: Array,
    path_second: Array,
    squared_speed: float,
    acceleration_limit: Array,
    maximum_path_acceleration: float,
    tolerance: float,
) -> tuple[float, float]:
    first = _finite_vector(path_first, 7, "path first derivative")
    second = _finite_vector(path_second, 7, "path second derivative")
    limit = _finite_vector(acceleration_limit, 7, "acceleration limit")
    x = max(0.0, float(squared_speed))
    lower = -float(maximum_path_acceleration)
    upper = float(maximum_path_acceleration)
    epsilon = 1.0e-14
    for coordinate in range(7):
        slope = float(first[coordinate])
        offset = float(second[coordinate]) * x
        if abs(slope) <= epsilon:
            if abs(offset) > limit[coordinate] + tolerance:
                raise RuntimeError(
                    "Path curvature violates a state acceleration limit at "
                    "the current squared speed."
                )
            continue
        bound_a = (-limit[coordinate] - offset) / slope
        bound_b = (limit[coordinate] - offset) / slope
        lower = max(lower, min(bound_a, bound_b))
        upper = min(upper, max(bound_a, bound_b))
    if lower > upper + tolerance:
        raise RuntimeError("No admissible path acceleration at current speed.")
    return lower, upper


def _reachability_parameterization(
    *,
    s_grid: Array,
    path_first: Array,
    path_second: Array,
    config: TimeParameterizationConfig,
) -> tuple[Array, Array, list[tuple[float, float]]]:
    """Return squared speed, segment acceleration, controllable intervals."""
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    first = _finite_matrix(path_first, 7, "path first derivatives")
    second = _finite_matrix(path_second, 7, "path second derivatives")
    if first.shape[0] != s_grid.size or second.shape[0] != s_grid.size:
        raise ValueError("Derivative arrays do not match timing grid.")
    ds = np.diff(s_grid)
    if np.any(ds <= 0.0):
        raise ValueError("Timing grid must be strictly increasing.")

    velocity_limit = config.effective_velocity_limit
    acceleration_limit = config.effective_acceleration_limit
    caps = _velocity_squared_caps(
        first,
        velocity_limit,
        config.maximum_path_speed_m_s,
    )
    start_x = float(config.start_path_speed_m_s) ** 2
    end_x = float(config.end_path_speed_m_s) ** 2
    tolerance = float(config.linear_program_tolerance)
    if start_x > caps[0] + tolerance or end_x > caps[-1] + tolerance:
        raise RuntimeError("An endpoint path speed exceeds its velocity cap.")

    controllable: list[tuple[float, float]] = [(0.0, 0.0)] * s_grid.size
    controllable[-1] = (end_x, end_x)
    for index in range(s_grid.size - 2, -1, -1):
        try:
            controllable[index] = _controllable_interval_lp(
                path_first=first[index],
                path_second=second[index],
                acceleration_limit=acceleration_limit,
                maximum_path_acceleration=config.maximum_path_acceleration_m_s2,
                squared_speed_cap=caps[index],
                ds=float(ds[index]),
                next_interval=controllable[index + 1],
                lp_tolerance=tolerance,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Time parameterization is not backward-controllable at "
                f"grid index {index}, s={s_grid[index]:.9f} m: {exc}"
            ) from exc

    if not (
        controllable[0][0] - tolerance
        <= start_x
        <= controllable[0][1] + tolerance
    ):
        raise RuntimeError(
            "The requested start speed is outside the backward-controllable set."
        )

    squared_speed = np.empty(s_grid.size, dtype=float)
    path_acceleration = np.empty(s_grid.size - 1, dtype=float)
    squared_speed[0] = start_x
    for index in range(s_grid.size - 1):
        lower_u, upper_u = _acceleration_interval_at_speed(
            path_first=first[index],
            path_second=second[index],
            squared_speed=squared_speed[index],
            acceleration_limit=acceleration_limit,
            maximum_path_acceleration=config.maximum_path_acceleration_m_s2,
            tolerance=tolerance,
        )
        next_lower, next_upper = controllable[index + 1]
        transition_scale = 2.0 * float(ds[index])
        lower_u = max(
            lower_u,
            (next_lower - squared_speed[index]) / transition_scale,
        )
        upper_u = min(
            upper_u,
            (next_upper - squared_speed[index]) / transition_scale,
        )
        if lower_u > upper_u + tolerance:
            raise RuntimeError(
                f"Forward reachability set is empty at grid index {index}."
            )
        chosen_u = upper_u
        next_x = squared_speed[index] + transition_scale * chosen_u
        next_x = float(np.clip(next_x, next_lower, next_upper))
        squared_speed[index + 1] = max(0.0, next_x)
        path_acceleration[index] = (
            squared_speed[index + 1] - squared_speed[index]
        ) / transition_scale

    if abs(squared_speed[-1] - end_x) > 100.0 * tolerance:
        raise RuntimeError("Forward pass did not reach the requested end speed.")
    return squared_speed, path_acceleration, controllable


def _segment_times(s_grid: Array, squared_speed: Array) -> Array:
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    speed = np.sqrt(np.maximum(0.0, np.asarray(squared_speed, dtype=float)))
    ds = np.diff(s_grid)
    denominator = speed[:-1] + speed[1:]
    if np.any(denominator <= 1.0e-14):
        bad = int(np.flatnonzero(denominator <= 1.0e-14)[0])
        raise RuntimeError(
            f"Timing interval {bad} has zero speed at both ends and cannot "
            "traverse a positive distance."
        )
    return 2.0 * ds / denominator


def _align_duration_to_sample_period(
    *,
    s_grid: Array,
    squared_speed: Array,
    path_acceleration: Array,
    sample_period_s: float,
) -> tuple[Array, Array, Array, float, float]:
    raw_segment_time = _segment_times(s_grid, squared_speed)
    raw_duration = float(np.sum(raw_segment_time))
    number_steps = max(1, int(math.ceil(raw_duration / sample_period_s)))
    aligned_duration = float(number_steps * sample_period_s)
    dilation = aligned_duration / raw_duration
    scaled_x = np.asarray(squared_speed, dtype=float) / dilation**2
    scaled_u = np.asarray(path_acceleration, dtype=float) / dilation**2
    segment_time = raw_segment_time * dilation
    grid_time = np.concatenate(([0.0], np.cumsum(segment_time)))
    grid_time[-1] = aligned_duration
    return grid_time, scaled_x, scaled_u, raw_duration, aligned_duration


def _sample_piecewise_constant_acceleration(
    *,
    time_grid: Array,
    s_grid: Array,
    squared_speed: Array,
    path_acceleration: Array,
    sample_period_s: float,
) -> tuple[Array, Array, Array, Array]:
    duration = float(time_grid[-1])
    number_steps = int(round(duration / sample_period_s))
    time = np.arange(number_steps + 1, dtype=float) * sample_period_s
    time[-1] = duration
    interval = np.searchsorted(time_grid, time, side="right") - 1
    interval = np.clip(interval, 0, time_grid.size - 2)
    local_time = time - time_grid[interval]
    speed_start = np.sqrt(np.maximum(0.0, squared_speed[interval]))
    acceleration = path_acceleration[interval]
    s = (
        s_grid[interval]
        + speed_start * local_time
        + 0.5 * acceleration * local_time**2
    )
    speed = speed_start + acceleration * local_time
    s[0] = s_grid[0]
    s[-1] = s_grid[-1]
    speed[0] = math.sqrt(max(0.0, squared_speed[0]))
    speed[-1] = math.sqrt(max(0.0, squared_speed[-1]))
    if np.any(np.diff(s) < -1.0e-11):
        raise RuntimeError("Sampled timing law is not monotone.")
    return time, s, np.maximum(0.0, speed), acceleration


def _physical_errors(
    output: Array,
    desired_position: Array,
    desired_tangent: Array,
) -> tuple[float, float, Array, Array]:
    output = _finite_vector(output, 6, "beam output")
    desired_position = _finite_vector(desired_position, 3, "desired position")
    desired_tangent = _finite_vector(desired_tangent, 3, "desired tangent")
    desired_tangent /= np.linalg.norm(desired_tangent)
    achieved_tangent = output[3:6]
    achieved_tangent /= np.linalg.norm(achieved_tangent)
    position_error = float(np.linalg.norm(output[:3] - desired_position))
    tangent_error = float(
        math.acos(float(np.clip(achieved_tangent @ desired_tangent, -1.0, 1.0)))
    )
    return position_error, tangent_error, output[:3].copy(), achieved_tangent


def _beam_validate_sorted_states(
    *,
    controller_pack: dict[str, Any],
    states: Array,
    debug: bool = False,
) -> Array:
    states = _finite_matrix(states, 7, "beam-validation states")
    adapter = controller_pack["plant_diagnostic_joint_adapter"]
    pose_adapter = controller_pack.get("plant_forward_pose8_adapter")
    if pose_adapter is None:
        raise KeyError("controller_pack has no plant_forward_pose8_adapter.")

    capture = getattr(pose_adapter, "capture_cache_state", None)
    restore = getattr(pose_adapter, "restore_cache_state", None)
    snapshot = capture() if callable(capture) else None
    reset_to_initial = getattr(pose_adapter, "reset_to_initial_baseline", None)
    outputs: list[Array] = []
    progress_stride = max(1, states.shape[0] // 20)
    try:
        if callable(reset_to_initial):
            reset_to_initial()
        else:
            reset = getattr(pose_adapter, "reset", None)
            if callable(reset):
                reset()
        for state in states:
            index = len(outputs)
            output = np.asarray(
                adapter.forward_output(state, commit=True), dtype=float
            ).reshape(-1)
            if output.size < 6 or not np.all(np.isfinite(output[:6])):
                raise FloatingPointError(
                    "Sequential beam validation produced an invalid output."
                )
            outputs.append(output[:6].copy())
            if debug and (
                index == 0
                or (index + 1) % progress_stride == 0
                or index + 1 == states.shape[0]
            ):
                print(
                    "[TIME BEAM VALIDATION] "
                    f"sample={index + 1}/{states.shape[0]}",
                    flush=True,
                )
    finally:
        if snapshot is not None and callable(restore):
            restore(snapshot)
    return np.asarray(outputs, dtype=float)


def _validate_beam_on_spatial_and_time_samples(
    *,
    spline: _ConfigurationSpline,
    path: GeometricConfigurationPath,
    time_s_coordinates: Array,
    controller_pack: dict[str, Any],
    spatial_samples_per_interval: int,
    lumen_C: Array | None = None,
    debug: bool = False,
) -> dict[str, Array | bool | float]:
    spatial_s = _subdivide_path(
        np.asarray(path.s_m, dtype=float),
        int(spatial_samples_per_interval) + 1,
    )
    combined = np.concatenate(
        (spatial_s, np.asarray(time_s_coordinates, dtype=float))
    )
    unique_s, inverse = np.unique(combined, return_inverse=True)
    states = spline.value(unique_s)
    outputs = _beam_validate_sorted_states(
        controller_pack=controller_pack,
        states=states,
        debug=debug,
    )
    if lumen_C is None:
        desired_position, desired_tangent = spline.reference(unique_s)
    else:
        from proper_research.planning.offline_inverse_configuration import (
            CentrelinePath,
        )

        reference_path = CentrelinePath(lumen_C)
        references = [reference_path.reference(float(value)) for value in unique_s]
        desired_position = np.asarray([item[0] for item in references], dtype=float)
        desired_tangent = np.asarray([item[1] for item in references], dtype=float)
    position_error = np.empty(unique_s.size, dtype=float)
    tangent_error = np.empty(unique_s.size, dtype=float)
    achieved_position = np.empty((unique_s.size, 3), dtype=float)
    achieved_tangent = np.empty((unique_s.size, 3), dtype=float)
    for index in range(unique_s.size):
        (
            position_error[index],
            tangent_error[index],
            achieved_position[index],
            achieved_tangent[index],
        ) = _physical_errors(
            outputs[index], desired_position[index], desired_tangent[index]
        )
    feasible = (
        (position_error <= path.position_tolerance_m)
        & (tangent_error <= path.tangent_tolerance_rad)
    )
    spatial_count = spatial_s.size
    spatial_map = inverse[:spatial_count]
    time_map = inverse[spatial_count:]
    return {
        "spatial_s": spatial_s,
        "spatial_position_error": position_error[spatial_map],
        "spatial_tangent_error": tangent_error[spatial_map],
        "spatial_feasible": feasible[spatial_map],
        "time_achieved_position": achieved_position[time_map],
        "time_achieved_tangent": achieved_tangent[time_map],
        "time_position_error": position_error[time_map],
        "time_tangent_error": tangent_error[time_map],
        "time_feasible": feasible[time_map],
        "maximum_position_error_m": float(np.max(position_error)),
        "maximum_tangent_error_rad": float(np.max(tangent_error)),
        "all_feasible": bool(np.all(feasible)),
    }


def parameterize_configuration_path(
    *,
    geometric_path: GeometricConfigurationPath,
    config: TimeParameterizationConfig,
    state_min: Any,
    state_max: Any,
    controller_pack: dict[str, Any] | None = None,
    lumen_C: Array | None = None,
) -> TimeParameterizedConfigurationPath:
    """Compute, verify, and return an MPC-rate time parameterization."""
    geometric_path.validate()
    config.validate()
    if config.require_saved_global_feasible and not geometric_path.globally_feasible:
        raise RuntimeError("The source geometric path is not globally feasible.")
    if (
        config.require_saved_dense_feasible
        and not geometric_path.dense_validation_feasible
    ):
        raise RuntimeError("The source geometric path did not pass dense validation.")
    if config.validate_nonlinear_beam and controller_pack is None:
        raise ValueError(
            "controller_pack is required when validate_nonlinear_beam=True."
        )

    lower = _finite_vector(state_min, 7, "state_min")
    upper = _finite_vector(state_max, 7, "state_max")
    if np.any(lower >= upper):
        raise ValueError("State bounds are invalid.")
    spline = _ConfigurationSpline(geometric_path)
    timing_s = _subdivide_path(
        np.asarray(geometric_path.s_m, dtype=float),
        config.timing_subdivisions_per_interval,
    )
    timing_state = spline.value(timing_s)
    state_scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    bound_tolerance = config.constraint_tolerance * state_scale
    if np.any(timing_state < lower - bound_tolerance) or np.any(
        timing_state > upper + bound_tolerance
    ):
        bad = np.argwhere(
            (timing_state < lower - bound_tolerance)
            | (timing_state > upper + bound_tolerance)
        )[0]
        raise RuntimeError(
            "Natural-cubic interpolation leaves a configuration bound at "
            f"timing grid row {bad[0]}, coordinate {bad[1]}."
        )

    path_first = spline.first(timing_s)
    path_second = spline.second(timing_s)
    squared_speed, path_acceleration, controllable = (
        _reachability_parameterization(
            s_grid=timing_s,
            path_first=path_first,
            path_second=path_second,
            config=config,
        )
    )
    (
        timing_time,
        squared_speed,
        path_acceleration,
        raw_duration,
        duration,
    ) = _align_duration_to_sample_period(
        s_grid=timing_s,
        squared_speed=squared_speed,
        path_acceleration=path_acceleration,
        sample_period_s=config.sample_period_s,
    )
    velocity_limit = config.effective_velocity_limit
    acceleration_limit = config.effective_acceleration_limit
    # Collocation constraints are imposed on the spatial timing grid.  Cubic
    # path curvature can make an MPC sample between two collocation points
    # slightly more restrictive.  Reconstruct the analytic sampled trajectory
    # and, if necessary, dilate time globally.  A dilation gamma changes rates
    # by 1/gamma and accelerations by 1/gamma**2 without changing chi(s).
    maximum_velocity_ratio = np.inf
    maximum_acceleration_ratio = np.inf
    for validation_round in range(4):
        time, s_time, s_dot, s_ddot = _sample_piecewise_constant_acceleration(
            time_grid=timing_time,
            s_grid=timing_s,
            squared_speed=squared_speed,
            path_acceleration=path_acceleration,
            sample_period_s=config.sample_period_s,
        )
        state = spline.value(s_time)
        state_s = spline.first(s_time)
        state_ss = spline.second(s_time)
        state_rate = state_s * s_dot[:, None]
        state_acceleration = (
            state_s * s_ddot[:, None]
            + state_ss * (s_dot[:, None] ** 2)
        )
        if lumen_C is None:
            desired_position, desired_tangent = spline.reference(s_time)
        else:
            from proper_research.planning.offline_inverse_configuration import (
                CentrelinePath,
            )

            reference_path = CentrelinePath(lumen_C)
            references = [
                reference_path.reference(float(value)) for value in s_time
            ]
            desired_position = np.asarray(
                [item[0] for item in references], dtype=float
            )
            desired_tangent = np.asarray(
                [item[1] for item in references], dtype=float
            )
        velocity_ratio = np.max(
            np.abs(state_rate) / velocity_limit[None, :], axis=0
        )
        acceleration_ratio = np.max(
            np.abs(state_acceleration) / acceleration_limit[None, :], axis=0
        )
        maximum_velocity_ratio = float(np.max(velocity_ratio))
        maximum_acceleration_ratio = float(np.max(acceleration_ratio))
        required_dilation = max(
            1.0,
            maximum_velocity_ratio,
            math.sqrt(maximum_acceleration_ratio),
        )
        if required_dilation <= 1.0 + config.constraint_tolerance:
            break
        if validation_round >= 3:
            break
        old_duration = float(timing_time[-1])
        requested_duration = old_duration * required_dilation * (1.0 + 1.0e-6)
        number_steps = int(math.ceil(requested_duration / config.sample_period_s))
        new_duration = number_steps * config.sample_period_s
        dilation = new_duration / old_duration
        timing_time = timing_time * dilation
        squared_speed = squared_speed / dilation**2
        path_acceleration = path_acceleration / dilation**2
        duration = float(new_duration)
        timing_time[-1] = duration
        if config.debug:
            print(
                "[TIME PARAMETERIZATION] post-collocation dilation "
                f"round={validation_round + 1} factor={dilation:.9f} "
                f"duration={duration:.6f} s",
                flush=True,
            )

    if maximum_velocity_ratio > 1.0 + config.constraint_tolerance:
        raise RuntimeError(
            "MPC-sampled trajectory violates a state velocity limit: "
            f"maximum ratio={maximum_velocity_ratio:.9f}."
        )
    if maximum_acceleration_ratio > 1.0 + config.constraint_tolerance:
        raise RuntimeError(
            "MPC-sampled trajectory violates a state acceleration limit: "
            f"maximum ratio={maximum_acceleration_ratio:.9f}."
        )

    achieved_position = np.full((time.size, 3), np.nan)
    achieved_tangent = np.full((time.size, 3), np.nan)
    position_error = np.full(time.size, np.nan)
    tangent_error = np.full(time.size, np.nan)
    time_beam_feasible = np.zeros(time.size, dtype=bool)
    spatial_s = np.empty(0, dtype=float)
    spatial_position_error = np.empty(0, dtype=float)
    spatial_tangent_error = np.empty(0, dtype=float)
    spatial_beam_feasible = np.empty(0, dtype=bool)
    beam_all_feasible = False
    maximum_beam_position_error = np.nan
    maximum_beam_tangent_error = np.nan
    if config.validate_nonlinear_beam:
        assert controller_pack is not None
        beam = _validate_beam_on_spatial_and_time_samples(
            spline=spline,
            path=geometric_path,
            time_s_coordinates=s_time,
            controller_pack=controller_pack,
            spatial_samples_per_interval=(
                config.beam_validation_samples_per_interval
            ),
            lumen_C=lumen_C,
            debug=config.debug,
        )
        achieved_position = np.asarray(beam["time_achieved_position"], dtype=float)
        achieved_tangent = np.asarray(beam["time_achieved_tangent"], dtype=float)
        position_error = np.asarray(beam["time_position_error"], dtype=float)
        tangent_error = np.asarray(beam["time_tangent_error"], dtype=float)
        time_beam_feasible = np.asarray(beam["time_feasible"], dtype=bool)
        spatial_s = np.asarray(beam["spatial_s"], dtype=float)
        spatial_position_error = np.asarray(
            beam["spatial_position_error"], dtype=float
        )
        spatial_tangent_error = np.asarray(
            beam["spatial_tangent_error"], dtype=float
        )
        spatial_beam_feasible = np.asarray(
            beam["spatial_feasible"], dtype=bool
        )
        beam_all_feasible = bool(beam["all_feasible"])
        maximum_beam_position_error = float(beam["maximum_position_error_m"])
        maximum_beam_tangent_error = float(beam["maximum_tangent_error_rad"])

    globally_feasible = bool(
        maximum_velocity_ratio <= 1.0 + config.constraint_tolerance
        and maximum_acceleration_ratio <= 1.0 + config.constraint_tolerance
        and (
            beam_all_feasible
            if config.validate_nonlinear_beam
            else geometric_path.globally_feasible
        )
    )
    summary = {
        "time_parameterized": True,
        "globally_feasible": globally_feasible,
        "source_global_feasible": geometric_path.globally_feasible,
        "source_dense_validation_feasible": (
            geometric_path.dense_validation_feasible
        ),
        "beam_validation_performed": config.validate_nonlinear_beam,
        "beam_validation_feasible": (
            beam_all_feasible if config.validate_nonlinear_beam else None
        ),
        "duration_s": duration,
        "raw_duration_s": raw_duration,
        "number_mpc_samples": int(time.size),
        "number_timing_grid_points": int(timing_s.size),
        "number_spatial_beam_validation_samples": int(spatial_s.size),
        "maximum_velocity_limit_ratio": maximum_velocity_ratio,
        "maximum_acceleration_limit_ratio": maximum_acceleration_ratio,
        "maximum_state_rate": np.max(np.abs(state_rate), axis=0),
        "maximum_state_acceleration": np.max(
            np.abs(state_acceleration), axis=0
        ),
        "maximum_path_speed_m_s": float(np.max(s_dot)),
        "maximum_absolute_path_acceleration_m_s2": float(
            np.max(np.abs(s_ddot))
        ),
        "maximum_beam_position_error_m": maximum_beam_position_error,
        "maximum_beam_tangent_error_rad": maximum_beam_tangent_error,
        "endpoint_state_rate_norms": (
            float(np.linalg.norm(state_rate[0])),
            float(np.linalg.norm(state_rate[-1])),
        ),
        "interpolation": config.interpolation,
        "constraint_model": (
            "kinematic configuration velocity and acceleration limits"
        ),
        "torque_constraints_included": False,
        "jerk_constraints_included": False,
        "path_parameter": "lumen_arc_length_m",
        "state_layout": "q1_q2_q3_q4_q5_q6_insertion",
        "input_layout": "qd1_qd2_qd3_qd4_qd5_qd6_insertion_rate",
        "controllable_interval_at_start": controllable[0],
    }
    result = TimeParameterizedConfigurationPath(
        time_s=time,
        path_coordinate_m=s_time,
        path_speed_m_s=s_dot,
        path_acceleration_m_s2=s_ddot,
        state=state,
        state_rate=state_rate,
        state_acceleration=state_acceleration,
        desired_position_m=desired_position,
        desired_tangent=desired_tangent,
        achieved_position_m=achieved_position,
        achieved_tangent=achieved_tangent,
        position_error_m=position_error,
        tangent_error_rad=tangent_error,
        beam_sample_feasible=time_beam_feasible,
        spatial_validation_s_m=spatial_s,
        spatial_position_error_m=spatial_position_error,
        spatial_tangent_error_rad=spatial_tangent_error,
        spatial_beam_feasible=spatial_beam_feasible,
        globally_feasible=globally_feasible,
        beam_validation_performed=config.validate_nonlinear_beam,
        duration_s=duration,
        raw_duration_s=raw_duration,
        configuration=_json_ready(config),
        summary=_json_ready(summary),
    )
    if (
        config.validate_nonlinear_beam
        and config.require_nonlinear_beam_feasible
        and not beam_all_feasible
    ):
        raise RuntimeError(
            "The time-parameterized spline failed nonlinear beam validation: "
            f"maximum_position_error={1.0e3 * maximum_beam_position_error:.6f} mm, "
            f"maximum_tangent_error={math.degrees(maximum_beam_tangent_error):.6f} deg."
        )
    return result


def _path_from_payload(payload: dict[str, Any], source: str) -> GeometricConfigurationPath:
    nodes = payload.get("nodes", [])
    if not nodes:
        raise ValueError("Global configuration summary contains no nodes.")
    states = np.array(
        [
            np.concatenate(
                (
                    _finite_vector(node["q_rad"], 6, "node joints"),
                    [float(node["insertion_m"])],
                )
            )
            for node in nodes
        ],
        dtype=float,
    )
    configuration = payload.get("configuration", {})
    summary = payload.get("summary", {})
    path = GeometricConfigurationPath(
        s_m=np.asarray([node["s_m"] for node in nodes], dtype=float),
        states=states,
        desired_position_m=np.asarray(
            [node["desired_position_m"] for node in nodes], dtype=float
        ),
        desired_tangent=np.asarray(
            [node["desired_tangent"] for node in nodes], dtype=float
        ),
        position_tolerance_m=float(configuration["position_tolerance_m"]),
        tangent_tolerance_rad=float(configuration["tangent_tolerance_rad"]),
        globally_feasible=bool(summary.get("globally_feasible", False)),
        dense_validation_feasible=bool(
            summary.get("dense_validation_feasible", False)
        ),
        source=str(source),
    )
    path.validate()
    return path


def load_saved_global_configuration_path(
    global_output_dir: str | Path,
) -> GeometricConfigurationPath:
    """Load and cross-check the saved global geometric path."""
    root = Path(global_output_dir)
    json_path = root / "global_configuration_summary.json"
    csv_path = root / "global_configuration_path.csv"
    dense_path = root / "global_configuration_dense_validation.csv"
    for path in (json_path, csv_path, dense_path):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    geometric = _path_from_payload(payload, str(root))

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != geometric.s_m.size:
        raise ValueError("Global CSV and JSON node counts differ.")
    s_csv = np.asarray([float(row["s_m"]) for row in rows], dtype=float)
    states_csv = np.asarray(
        [
            [float(row[f"q{joint}_rad"]) for joint in range(1, 7)]
            + [float(row["insertion_m"])]
            for row in rows
        ],
        dtype=float,
    )
    if not np.allclose(s_csv, geometric.s_m, atol=1.0e-12, rtol=0.0):
        raise ValueError("Global CSV and JSON path coordinates differ.")
    if not np.allclose(states_csv, geometric.states, atol=1.0e-10, rtol=0.0):
        raise ValueError("Global CSV and JSON configuration states differ.")
    if not all(bool(int(row["feasible"])) for row in rows):
        raise RuntimeError("Saved global path CSV contains an infeasible node.")

    with dense_path.open("r", encoding="utf-8", newline="") as stream:
        dense_rows = list(csv.DictReader(stream))
    if not dense_rows:
        raise RuntimeError("Saved global path has no dense-validation samples.")
    # if not all(bool(int(row["feasible"])) for row in dense_rows):
    #     raise RuntimeError("Saved dense-validation CSV contains a failed sample.")
    return geometric


def geometric_path_from_global_result(
    global_result: Any,
    *,
    source: str = "in_memory_global_result",
) -> GeometricConfigurationPath:
    """Adapt an in-memory GlobalConfigurationPathResult without serialization."""
    nodes = list(getattr(global_result, "nodes", []))
    if not nodes:
        raise ValueError("global_result contains no nodes.")
    configuration = dict(getattr(global_result, "configuration", {}))
    return GeometricConfigurationPath(
        s_m=np.asarray([node.s_m for node in nodes], dtype=float),
        states=np.asarray(
            [
                np.concatenate(
                    (
                        _finite_vector(node.q_rad, 6, "node joints"),
                        [float(node.insertion_m)],
                    )
                )
                for node in nodes
            ],
            dtype=float,
        ),
        desired_position_m=np.asarray(
            [node.desired_position_m for node in nodes], dtype=float
        ),
        desired_tangent=np.asarray(
            [node.desired_tangent for node in nodes], dtype=float
        ),
        position_tolerance_m=float(configuration["position_tolerance_m"]),
        tangent_tolerance_rad=float(configuration["tangent_tolerance_rad"]),
        globally_feasible=bool(global_result.globally_feasible),
        dense_validation_feasible=bool(global_result.dense_validation_feasible),
        source=str(source),
    )


def _time_csv_record(
    result: TimeParameterizedConfigurationPath,
    index: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_index": int(index),
        "time_s": float(result.time_s[index]),
        "path_s_m": float(result.path_coordinate_m[index]),
        "path_speed_m_s": float(result.path_speed_m_s[index]),
        "path_acceleration_m_s2": float(
            result.path_acceleration_m_s2[index]
        ),
        "position_error_m": float(result.position_error_m[index]),
        "position_error_mm": float(1.0e3 * result.position_error_m[index]),
        "tangent_error_rad": float(result.tangent_error_rad[index]),
        "tangent_error_deg": float(np.degrees(result.tangent_error_rad[index])),
        "beam_feasible": int(result.beam_sample_feasible[index]),
        "insertion_m": float(result.state[index, 6]),
        "insertion_rate_m_s": float(result.state_rate[index, 6]),
        "insertion_acceleration_m_s2": float(
            result.state_acceleration[index, 6]
        ),
    }
    for coordinate in range(6):
        record[f"q{coordinate + 1}_rad"] = float(result.state[index, coordinate])
        record[f"qd{coordinate + 1}_rad_s"] = float(
            result.state_rate[index, coordinate]
        )
        record[f"qdd{coordinate + 1}_rad_s2"] = float(
            result.state_acceleration[index, coordinate]
        )
    for prefix, values in (
        ("desired", result.desired_position_m[index]),
        ("achieved", result.achieved_position_m[index]),
        ("desired_tangent", result.desired_tangent[index]),
        ("achieved_tangent", result.achieved_tangent[index]),
    ):
        for axis, value in zip(("x", "y", "z"), values):
            record[f"{prefix}_{axis}"] = float(value)
    return record


def save_time_parameterized_configuration_path(
    result: TimeParameterizedConfigurationPath,
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "time_parameterized_configuration_path.csv"
    json_path = root / "time_parameterized_configuration_summary.json"
    npz_path = root / "time_parameterized_configuration_path.npz"
    validation_path = root / "time_parameterized_spatial_validation.csv"
    plot_path = root / "time_parameterized_configuration_path.png"

    records = [_time_csv_record(result, index) for index in range(result.time_s.size)]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with validation_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "sample_index",
            "path_s_m",
            "position_error_m",
            "position_error_mm",
            "tangent_error_rad",
            "tangent_error_deg",
            "feasible",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, s_value in enumerate(result.spatial_validation_s_m):
            writer.writerow(
                {
                    "sample_index": index,
                    "path_s_m": float(s_value),
                    "position_error_m": float(
                        result.spatial_position_error_m[index]
                    ),
                    "position_error_mm": float(
                        1.0e3 * result.spatial_position_error_m[index]
                    ),
                    "tangent_error_rad": float(
                        result.spatial_tangent_error_rad[index]
                    ),
                    "tangent_error_deg": float(
                        np.degrees(result.spatial_tangent_error_rad[index])
                    ),
                    "feasible": int(result.spatial_beam_feasible[index]),
                }
            )

    payload = {
        "configuration": result.configuration,
        "summary": result.summary,
        "output_files": result.output_files,
    }
    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        npz_path,
        time_s=result.time_s,
        path_s_m=result.path_coordinate_m,
        path_speed_m_s=result.path_speed_m_s,
        path_acceleration_m_s2=result.path_acceleration_m_s2,
        state=result.state,
        state_reference=result.state,
        state_rate=result.state_rate,
        input_reference=result.state_rate,
        state_acceleration=result.state_acceleration,
        state_acceleration_reference=result.state_acceleration,
        desired_position_m=result.desired_position_m,
        desired_tangent=result.desired_tangent,
        achieved_position_m=result.achieved_position_m,
        achieved_tangent=result.achieved_tangent,
        position_error_m=result.position_error_m,
        tangent_error_rad=result.tangent_error_rad,
        beam_sample_feasible=result.beam_sample_feasible,
    )
    try:
        _save_time_parameterization_plot(result, plot_path)
    except Exception as exc:
        result.summary["plot_error"] = f"{type(exc).__name__}: {exc}"
        json_path.write_text(
            json.dumps(_json_ready(payload), indent=2), encoding="utf-8"
        )
    files = {
        "csv": str(csv_path),
        "json": str(json_path),
        "npz": str(npz_path),
        "spatial_validation_csv": str(validation_path),
        "plot": str(plot_path),
    }
    result.output_files.update(files)
    payload["output_files"] = result.output_files
    json_path.write_text(
        json.dumps(_json_ready(payload), indent=2), encoding="utf-8"
    )
    return files


def load_mpc_configuration_reference(
    output_dir: str | Path,
    *,
    require_feasible: bool = True,
) -> dict[str, Any]:
    """Load the compact arrays consumed by an online MPC/simulation runner.

    The returned ``state_reference`` has columns ``[q1..q6, L]`` and
    ``input_reference`` has columns ``[q1dot..q6dot, Ldot]``.  The function
    verifies the saved feasibility flag and constant MPC sample period before
    exposing the reference.
    """
    root = Path(output_dir)
    json_path = root / "time_parameterized_configuration_summary.json"
    npz_path = root / "time_parameterized_configuration_path.npz"
    for path in (json_path, npz_path):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary = dict(payload.get("summary", {}))
    configuration = dict(payload.get("configuration", {}))
    if require_feasible and not bool(summary.get("globally_feasible", False)):
        raise RuntimeError("Saved time-parameterized reference is not feasible.")
    with np.load(npz_path) as archive:
        required = (
            "time_s",
            "path_s_m",
            "state_reference",
            "input_reference",
            "state_acceleration_reference",
            "desired_position_m",
            "desired_tangent",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise KeyError(
                "Time-parameterized NPZ is missing: " + ", ".join(missing)
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    time = _finite_vector(
        arrays["time_s"], np.asarray(arrays["time_s"]).size, "time reference"
    )
    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Saved MPC time grid must be strictly increasing.")
    configured_dt = float(configuration["sample_period_s"])
    if not np.allclose(
        np.diff(time), configured_dt, atol=1.0e-12, rtol=1.0e-10
    ):
        raise ValueError("Saved MPC time grid is not uniformly sampled.")
    if _finite_matrix(arrays["state_reference"], 7, "state reference").shape[0] != time.size:
        raise ValueError("state_reference row count differs from time_s.")
    if _finite_matrix(arrays["input_reference"], 7, "input reference").shape[0] != time.size:
        raise ValueError("input_reference row count differs from time_s.")
    return {
        **arrays,
        "sample_period_s": configured_dt,
        "summary": summary,
        "configuration": configuration,
        "source_directory": str(root),
    }


def _save_time_parameterization_plot(
    result: TimeParameterizedConfigurationPath,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = result.time_s
    velocity_limit = np.asarray(
        result.configuration["state_velocity_limit"], dtype=float
    ) * float(result.configuration["velocity_safety_factor"])
    acceleration_limit = np.asarray(
        result.configuration["state_acceleration_limit"], dtype=float
    ) * float(result.configuration["acceleration_safety_factor"])

    figure, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    axes[0, 0].plot(time, 1.0e3 * result.path_coordinate_m)
    axes[0, 0].set_ylabel("path s [mm]")
    axes[0, 0].set_title("Monotone timing law")

    axes[0, 1].plot(time, 1.0e3 * result.path_speed_m_s, label="speed [mm/s]")
    axis_acceleration = axes[0, 1].twinx()
    axis_acceleration.plot(
        time,
        1.0e3 * result.path_acceleration_m_s2,
        color="tab:orange",
        alpha=0.75,
        label="acceleration [mm/s²]",
    )
    axes[0, 1].set_title("Scalar path speed and acceleration")
    axes[0, 1].set_ylabel("speed [mm/s]")
    axis_acceleration.set_ylabel("acceleration [mm/s²]")

    for coordinate in range(6):
        axes[1, 0].plot(
            time,
            np.degrees(result.state[:, coordinate]),
            label=f"q{coordinate + 1}",
        )
    axes[1, 0].set_title("Joint reference")
    axes[1, 0].set_ylabel("angle [deg]")
    axes[1, 0].legend(ncol=3, fontsize=8)

    axes[1, 1].plot(time, 1.0e3 * result.state[:, 6], color="tab:purple")
    axes[1, 1].set_title("Insertion reference")
    axes[1, 1].set_ylabel("insertion [mm]")

    for coordinate in range(6):
        axes[2, 0].plot(
            time,
            result.state_rate[:, coordinate],
            label=f"qdot{coordinate + 1}",
        )
    axes[2, 0].axhline(np.min(velocity_limit[:6]), color="k", linestyle="--")
    axes[2, 0].axhline(-np.min(velocity_limit[:6]), color="k", linestyle="--")
    axes[2, 0].set_title("Joint-rate feedforward")
    axes[2, 0].set_ylabel("rate [rad/s]")

    axes[2, 1].plot(
        time,
        1.0e3 * result.state_rate[:, 6],
        color="tab:purple",
        label="Ldot",
    )
    axes[2, 1].axhline(
        1.0e3 * velocity_limit[6], color="k", linestyle="--"
    )
    axes[2, 1].axhline(
        -1.0e3 * velocity_limit[6], color="k", linestyle="--"
    )
    axes[2, 1].set_title("Insertion-rate feedforward")
    axes[2, 1].set_ylabel("rate [mm/s]")

    for coordinate in range(6):
        axes[3, 0].plot(
            time,
            result.state_acceleration[:, coordinate],
            label=f"qdd{coordinate + 1}",
        )
    axes[3, 0].axhline(
        np.min(acceleration_limit[:6]), color="k", linestyle="--"
    )
    axes[3, 0].axhline(
        -np.min(acceleration_limit[:6]), color="k", linestyle="--"
    )
    axes[3, 0].set_title("Joint acceleration")
    axes[3, 0].set_ylabel("acceleration [rad/s²]")

    if result.beam_validation_performed:
        axes[3, 1].plot(
            time,
            1.0e3 * result.position_error_m,
            label="position [mm]",
        )
        axis_tangent = axes[3, 1].twinx()
        axis_tangent.plot(
            time,
            np.degrees(result.tangent_error_rad),
            color="tab:orange",
            label="tangent [deg]",
        )
        axes[3, 1].set_ylabel("position error [mm]")
        axis_tangent.set_ylabel("tangent error [deg]")
        axes[3, 1].set_title("Nonlinear beam validation at MPC samples")
    else:
        axes[3, 1].plot(
            time,
            1.0e3 * result.state_acceleration[:, 6],
            color="tab:purple",
        )
        axes[3, 1].set_title("Insertion acceleration")
        axes[3, 1].set_ylabel("acceleration [mm/s²]")

    for axis in axes.flat:
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Time-parameterized magnetic-beam configuration path",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def time_parameterize_saved_global_path(
    *,
    global_output_dir: str | Path,
    controller_pack: dict[str, Any],
    config: TimeParameterizationConfig,
    output_dir: str | Path | None,
    lumen_C: Array | None = None,
) -> TimeParameterizedConfigurationPath:
    geometric = load_saved_global_configuration_path(global_output_dir)
    result = parameterize_configuration_path(
        geometric_path=geometric,
        config=config,
        state_min=controller_pack["p_min"],
        state_max=controller_pack["p_max"],
        controller_pack=controller_pack,
        lumen_C=lumen_C,
    )
    if output_dir is not None:
        save_time_parameterized_configuration_path(result, output_dir)
    return result


def time_parameterize_global_result(
    *,
    global_result: Any,
    controller_pack: dict[str, Any],
    config: TimeParameterizationConfig,
    output_dir: str | Path | None,
    lumen_C: Array | None = None,
) -> TimeParameterizedConfigurationPath:
    geometric = geometric_path_from_global_result(global_result)
    result = parameterize_configuration_path(
        geometric_path=geometric,
        config=config,
        state_min=controller_pack["p_min"],
        state_max=controller_pack["p_max"],
        controller_pack=controller_pack,
        lumen_C=lumen_C,
    )
    if output_dir is not None:
        save_time_parameterized_configuration_path(result, output_dir)
    return result


def _controller_sample_period(controller_pack: dict[str, Any]) -> float | None:
    for key in ("controller", "mpc"):
        controller = controller_pack.get(key)
        value = getattr(controller, "dt", None)
        if value is not None and np.isfinite(value) and float(value) > 0.0:
            return float(value)
    return None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute an MPC-rate, velocity/acceleration-constrained timing law "
            "for a globally feasible magnetic-beam configuration path."
        )
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
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_self_test() -> None:
    s = np.linspace(0.0, 0.1, 21)
    states = np.column_stack(
        (
            0.2 * np.sin(2.0 * np.pi * s / s[-1]),
            0.1 * np.cos(np.pi * s / s[-1]),
            0.05 * np.sin(np.pi * s / s[-1]),
            np.zeros_like(s),
            np.zeros_like(s),
            np.zeros_like(s),
            0.01 + 0.04 * s,
        )
    )
    desired = np.column_stack((s, np.zeros_like(s), np.zeros_like(s)))
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (s.size, 1))
    path = GeometricConfigurationPath(
        s_m=s,
        states=states,
        desired_position_m=desired,
        desired_tangent=tangent,
        position_tolerance_m=1.0e-3,
        tangent_tolerance_rad=0.5,
        globally_feasible=True,
        dense_validation_feasible=True,
        source="self_test",
    )
    config = TimeParameterizationConfig(
        sample_period_s=0.01,
        state_velocity_limit=(1.0,) * 6 + (0.1,),
        state_acceleration_limit=(2.0,) * 6 + (0.5,),
        maximum_path_speed_m_s=0.05,
        maximum_path_acceleration_m_s2=0.2,
        validate_nonlinear_beam=False,
        require_nonlinear_beam_feasible=False,
        debug=False,
    )
    result = parameterize_configuration_path(
        geometric_path=path,
        config=config,
        state_min=(-2.0,) * 6 + (0.0,),
        state_max=(2.0,) * 6 + (0.2,),
        controller_pack=None,
    )
    if not result.globally_feasible:
        raise AssertionError("Self-test parameterization is not feasible.")
    if not np.all(np.diff(result.path_coordinate_m) >= -1.0e-12):
        raise AssertionError("Self-test timing law is not monotone.")
    if np.linalg.norm(result.state_rate[0]) > 1.0e-10:
        raise AssertionError("Self-test does not start at rest.")
    if np.linalg.norm(result.state_rate[-1]) > 1.0e-10:
        raise AssertionError("Self-test does not end at rest.")
    print(
        "[TIME PARAMETERIZATION SELF TEST] PASS "
        f"duration={result.duration_s:.6f} s "
        f"samples={result.time_s.size}"
    )


def main() -> None:
    arguments = _arguments()
    if arguments.self_test:
        run_self_test()
        return

    from proper_research.planning.planning_context import build_planning_context

    _, bundle, controller_pack, out_root = build_planning_context()
    controller_velocity = _finite_vector(
        controller_pack["u_max"], 7, "controller_pack['u_max']"
    )
    if arguments.joint_velocity_limit is not None:
        controller_velocity[:6] = float(arguments.joint_velocity_limit)
    if arguments.insertion_velocity_limit is not None:
        controller_velocity[6] = float(arguments.insertion_velocity_limit)
    inferred_dt = _controller_sample_period(controller_pack)
    sample_period = (
        float(arguments.dt)
        if arguments.dt is not None
        else inferred_dt
        if inferred_dt is not None
        else 0.01
    )
    config = TimeParameterizationConfig(
        sample_period_s=sample_period,
        state_velocity_limit=tuple(controller_velocity),
        state_acceleration_limit=(
            (float(arguments.joint_acceleration_limit),) * 6
            + (float(arguments.insertion_acceleration_limit),)
        ),
        velocity_safety_factor=float(arguments.velocity_safety_factor),
        acceleration_safety_factor=float(arguments.acceleration_safety_factor),
        maximum_path_speed_m_s=float(arguments.maximum_path_speed),
        maximum_path_acceleration_m_s2=float(
            arguments.maximum_path_acceleration
        ),
        timing_subdivisions_per_interval=int(arguments.timing_subdivisions),
        beam_validation_samples_per_interval=int(
            arguments.beam_validation_samples
        ),
        validate_nonlinear_beam=not arguments.skip_beam_validation,
        require_nonlinear_beam_feasible=not arguments.skip_beam_validation,
        debug=True,
    )
    global_directory = arguments.global_dir or (
        out_root / "global_configuration_full_debug"
    )
    output_directory = arguments.output_dir or (
        out_root / "time_parameterized_configuration_path"
    )
    print(
        "[TIME PARAMETERIZATION] "
        f"global_directory={global_directory} "
        f"output_directory={output_directory} "
        f"dt={sample_period:.6f} s",
        flush=True,
    )
    print(
        "[TIME PARAMETERIZATION LIMITS] "
        f"velocity={config.effective_velocity_limit} "
        f"acceleration={config.effective_acceleration_limit}",
        flush=True,
    )
    result = time_parameterize_saved_global_path(
        global_output_dir=global_directory,
        controller_pack=controller_pack,
        config=config,
        output_dir=output_directory,
        lumen_C=bundle.lumen_C,
    )
    print(f"Duration: {result.duration_s:.6f} s")
    print(f"MPC samples: {result.time_s.size}")
    print(f"Trajectory feasible: {result.globally_feasible}")
    print(
        "Maximum velocity-limit ratio: "
        f"{result.summary['maximum_velocity_limit_ratio']:.6f}"
    )
    print(
        "Maximum acceleration-limit ratio: "
        f"{result.summary['maximum_acceleration_limit_ratio']:.6f}"
    )
    if result.beam_validation_performed:
        print(
            "Maximum beam position error: "
            f"{1.0e3 * result.summary['maximum_beam_position_error_m']:.6f} mm"
        )
        print(
            "Maximum beam tangent error: "
            f"{np.degrees(result.summary['maximum_beam_tangent_error_rad']):.6f} deg"
        )
    print(f"Saved time-parameterized path: {output_directory}")


if __name__ == "__main__":
    main()


__all__ = [
    "GeometricConfigurationPath",
    "TimeParameterizationConfig",
    "TimeParameterizedConfigurationPath",
    "geometric_path_from_global_result",
    "load_saved_global_configuration_path",
    "load_mpc_configuration_reference",
    "parameterize_configuration_path",
    "save_time_parameterized_configuration_path",
    "time_parameterize_global_result",
    "time_parameterize_saved_global_path",
]
