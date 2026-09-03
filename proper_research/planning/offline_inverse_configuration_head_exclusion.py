"""Offline inverse-configuration planning for the magnetic-beam system.

This module computes a discrete configuration path

    chi_j = [q1, q2, q3, q4, q5, q6, insertion]

whose nonlinear beam output follows a sampled lumen centreline in tip position
and directed tip tangent.  It deliberately stops at a *discrete feasible path*;
it does not smooth the path, assign time, run MPC, or command hardware.

The intended integration point is the dictionary returned by
``controller_factory_joint_space.build_controller``.  In particular, this
module uses ``plant_diagnostic_joint_adapter`` because that adapter exposes:

* the complete contact-aware output ``[tip_xyz, tangent_xyz]``;
* the continuous 6-by-7 chained Jacobian with respect to ``[q1..q6, L]``;
* the calibrated robot forward kinematics and flange-to-magnet transform.

The MPC's discrete matrix ``B_k = dt * J_k`` is intentionally not used here.
The decision variable is configuration, not a one-step velocity command, so the
required derivative is the continuous configuration Jacobian ``J_k``.

Typical use from the existing experiment construction is::

    config = InverseConfigurationPlannerConfig(
        position_tolerance_m=1.0e-3,
        tangent_tolerance_rad=np.deg2rad(5.0),
        source_magnet_lumen_exclusion_radius_m=0.12,  # apparatus-specific
    )
    result = solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=bundle.lumen_C,
        config=config,
        output_dir=out_root / "offline_inverse_configuration",
    )

The two task tolerances are required scientific inputs.  Defaults are not
silently assigned for them.  The source-magnet/lumen exclusion radius is
likewise a scientific geometry choice; leaving it as ``None`` preserves the
original unconstrained behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize, least_squares
from scipy.spatial.transform import Rotation as Rot


Array = np.ndarray

# Deliberate delivery marker: this version enforces distance >= exclusion
# radius.  It is distinct from the superseded containment implementation.
IMPLEMENTATION_TAG = "source_magnet_head_exclusion_v1"


def _finite_vector(value: Any, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != int(size) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} must contain {size} finite values; got {result.shape}."
        )
    return result.copy()


def _unit_vector(value: Any, name: str, eps: float = 1.0e-12) -> Array:
    vector = _finite_vector(value, 3, name)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        raise ValueError(f"{name} has near-zero norm ({norm:.3e}).")
    return vector / norm


def _json_ready(value: Any) -> Any:
    """Convert NumPy/dataclass values into JSON-safe Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class InverseConfigurationPlannerConfig:
    """Numerical and physical settings for the offline inverse solve.

    ``position_tolerance_m`` and ``tangent_tolerance_rad`` are required because
    they are experiment-level scientific choices.  A node is reported feasible
    only when both physical tolerances are satisfied after nonlinear model
    evaluation, regardless of the SciPy termination flag.

    Continuity weights act on changes normalized by the configured maximum
    adjacent-node changes.  They are secondary objectives, not feasibility
    definitions.
    """

    position_tolerance_m: float
    tangent_tolerance_rad: float

    initial_path_step_m: float = 2e-3
    minimum_path_step_m: float = 2.5e-4
    maximum_path_step_m: float = 2.0e-3
    grow_step_after_successes: int = 3
    step_growth_factor: float = 1.5

    maximum_function_evaluations: int = 100
    maximum_multistart_attempts: int = 6
    maximum_joint_step_rad: tuple[float, ...] = (
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    )
    maximum_insertion_step_m: float = 3.0e-3

    # Residual weights.  Values are deliberately small relative to the
    # tolerance-normalized task residuals.
    continuity_weight: tuple[float, ...] = (
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
        1.0e-3,
    )
    joint_centre_weight: tuple[float, ...] = (
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        1.0e-6,
        0.0,
    )

    multistart_joint_perturbation_rad: float = 0.05
    multistart_insertion_perturbation_m: float = 5.0e-4
    random_seed: int = 7

    start_at_initial_tip_projection: bool = True
    solve_initial_node: bool = False
    use_extrapolated_guess: bool = True

    # Next-node initial guess. ``use_extrapolated_guess`` is the historical
    # secant predictor ``x_{k+1} ~ 2 x_k - x_{k-1}`` -- constant joint velocity,
    # blind to the analytical Jacobian, the local path curvature, and any
    # change in the path step.  ``use_jacobian_predictor_guess`` instead solves
    # a damped least-norm joint step from the analytical 6x7 chained output
    # Jacobian at the current node for the desired change in tip position (and,
    # optionally, tangent):  dx = J^T (J J^T + lam^2 I)^-1 d_task.  It is a
    # first-order continuation (Euler) predictor, works from the very first
    # node (the secant needs two), naturally projects onto the controllable
    # subspace when J is rank-deficient, and is tried BEFORE the secant.  The
    # secant and previous state remain as fallback multistart candidates.
    use_jacobian_predictor_guess: bool = True
    jacobian_predictor_damping: float = 1.0e-3
    jacobian_predictor_use_tangent: bool = False

    insertion_non_decreasing: bool = False
    require_contact_model: bool = True

    # Optional hard source-magnet collision-avoidance constraint.  When
    # supplied, the source-magnet centre must remain OUTSIDE the capsule-like
    # exclusion region formed by sweeping this radius around every segment of
    # ``lumen_C``.  This is a centre-to-centre minimum distance.  To prevent
    # contact with a finite-radius head, configure
    #
    #   exclusion radius = head radius + magnet collision radius + safety gap.
    source_magnet_lumen_exclusion_radius_m: float | None = None
    source_magnet_lumen_constraint_tolerance_m: float = 1.0e-6

    # Prefer an analytical translational magnet Jacobian supplied by the
    # adapter.  A central-difference FK fallback is available because the
    # existing diagnostic adapter is only guaranteed to expose
    # ``magnet_transform``.  This setting does not alter the analytical beam
    # output Jacobian used by the inverse objective.
    require_analytical_magnet_position_jacobian: bool = False
    magnet_jacobian_joint_step_rad: float = 1.0e-6
    magnet_jacobian_insertion_step_m: float = 1.0e-6

    finite_difference_validation_at_start: bool = True
    finite_difference_validation_stride: int = 0
    finite_difference_joint_step_rad: float = 1.0e-6
    finite_difference_insertion_step_m: float = 1.0e-6
    maximum_chain_rule_relative_error: float = 5.0e-2

    ftol: float = 1.0e-9
    xtol: float = 1.0e-9
    gtol: float = 1.0e-9
    debug: bool = True

    def validate(self) -> None:
        positive_scalars = {
            "position_tolerance_m": self.position_tolerance_m,
            "tangent_tolerance_rad": self.tangent_tolerance_rad,
            "initial_path_step_m": self.initial_path_step_m,
            "minimum_path_step_m": self.minimum_path_step_m,
            "maximum_path_step_m": self.maximum_path_step_m,
            "maximum_insertion_step_m": self.maximum_insertion_step_m,
            "finite_difference_joint_step_rad": self.finite_difference_joint_step_rad,
            "finite_difference_insertion_step_m": (
                self.finite_difference_insertion_step_m
            ),
            "maximum_chain_rule_relative_error": (
                self.maximum_chain_rule_relative_error
            ),
            "magnet_jacobian_joint_step_rad": (
                self.magnet_jacobian_joint_step_rad
            ),
            "magnet_jacobian_insertion_step_m": (
                self.magnet_jacobian_insertion_step_m
            ),
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
        }
        for name, value in positive_scalars.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")

        if not 0.0 < self.tangent_tolerance_rad < math.pi:
            raise ValueError("tangent_tolerance_rad must lie in (0, pi).")
        if self.minimum_path_step_m > self.initial_path_step_m:
            raise ValueError(
                "minimum_path_step_m cannot exceed initial_path_step_m."
            )
        if self.initial_path_step_m > self.maximum_path_step_m:
            raise ValueError(
                "initial_path_step_m cannot exceed maximum_path_step_m."
            )
        if self.maximum_function_evaluations < 1:
            raise ValueError("maximum_function_evaluations must be >= 1.")
        if self.maximum_multistart_attempts < 1:
            raise ValueError("maximum_multistart_attempts must be >= 1.")
        if self.grow_step_after_successes < 1:
            raise ValueError("grow_step_after_successes must be >= 1.")
        if not np.isfinite(self.step_growth_factor) or self.step_growth_factor < 1.0:
            raise ValueError("step_growth_factor must be finite and >= 1.")
        if self.finite_difference_validation_stride < 0:
            raise ValueError("finite_difference_validation_stride must be >= 0.")
        if (
            not np.isfinite(self.jacobian_predictor_damping)
            or float(self.jacobian_predictor_damping) <= 0.0
        ):
            raise ValueError("jacobian_predictor_damping must be finite and positive.")

        radius = self.source_magnet_lumen_exclusion_radius_m
        if radius is not None and (
            not np.isfinite(radius) or float(radius) <= 0.0
        ):
            raise ValueError(
                "source_magnet_lumen_exclusion_radius_m must be finite and "
                "positive when collision avoidance is enabled."
            )
        tolerance = float(self.source_magnet_lumen_constraint_tolerance_m)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "source_magnet_lumen_constraint_tolerance_m must be finite and "
                "non-negative."
            )

        joint_step = np.asarray(self.maximum_joint_step_rad, dtype=float).reshape(-1)
        if joint_step.size != 6 or np.any(~np.isfinite(joint_step)):
            raise ValueError("maximum_joint_step_rad must contain six finite values.")
        if np.any(joint_step <= 0.0):
            raise ValueError("maximum_joint_step_rad entries must be positive.")

        for name in ("continuity_weight", "joint_centre_weight"):
            weights = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if weights.size != 7 or np.any(~np.isfinite(weights)):
                raise ValueError(f"{name} must contain seven finite values.")
            if np.any(weights < 0.0):
                raise ValueError(f"{name} entries must be non-negative.")

        for name in (
            "multistart_joint_perturbation_rad",
            "multistart_insertion_perturbation_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass
class InverseNodeResult:
    node_index: int
    s_m: float
    q_rad: Array
    insertion_m: float
    magnet_pose6: Array
    tip_position_m: Array
    tip_tangent: Array
    desired_position_m: Array
    desired_tangent: Array
    position_error_m: float
    tangent_error_rad: float
    feasible: bool
    solver_success: bool
    termination_reason: str
    function_evaluations: int
    objective_value: float
    jacobian_condition: float
    jacobian_effective_rank: int
    minimum_joint_margin_rad: float
    minimum_state_margin: float
    minimum_clearance_m: float
    contact_active: bool
    source_magnet_lumen_distance_m: float
    source_magnet_lumen_exclusion_radius_m: float
    source_magnet_lumen_margin_m: float
    source_magnet_lumen_segment_index: int
    source_magnet_lumen_constraint_satisfied: bool
    delta_q_norm_rad: float
    delta_insertion_m: float
    attempt_count: int
    chain_rule_relative_error: float = np.nan


@dataclass
class InversePathResult:
    nodes: list[InverseNodeResult]
    all_nodes_feasible: bool
    last_feasible_index: int
    last_feasible_progress_m: float
    configuration: dict[str, Any]
    summary: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)


class CentrelinePath:
    """Arc-length parameterization of a non-degenerate 3-D polyline."""

    def __init__(self, centreline: Any):
        points = np.asarray(centreline, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            raise ValueError("lumen_C must have shape (N, >=3), with N >= 2.")
        points = points[:, :3].copy()
        if not np.all(np.isfinite(points)):
            raise ValueError("lumen_C contains non-finite values.")

        segments = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        if np.any(segment_lengths <= 1.0e-12):
            indices = np.flatnonzero(segment_lengths <= 1.0e-12).tolist()
            raise ValueError(
                "lumen_C contains degenerate consecutive segments at indices "
                f"{indices}."
            )

        self.C = points
        self.segment_lengths = segment_lengths
        self.s = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.total_length_m = float(self.s[-1])

        tangents = np.empty_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if points.shape[0] > 2:
            tangents[1:-1] = points[2:] - points[:-2]
        tangent_norms = np.linalg.norm(tangents, axis=1)
        if np.any(tangent_norms <= 1.0e-12):
            raise ValueError("Centreline tangent construction produced zero norm.")
        self.T = tangents / tangent_norms[:, None]

    def position(self, s_query: float) -> Array:
        s_value = float(np.clip(float(s_query), 0.0, self.total_length_m))
        return np.array(
            [np.interp(s_value, self.s, self.C[:, axis]) for axis in range(3)],
            dtype=float,
        )

    def tangent(self, s_query: float) -> Array:
        s_value = float(np.clip(float(s_query), 0.0, self.total_length_m))
        interpolated = np.array(
            [np.interp(s_value, self.s, self.T[:, axis]) for axis in range(3)],
            dtype=float,
        )
        return _unit_vector(interpolated, "interpolated reference tangent")

    def reference(self, s_query: float) -> tuple[Array, Array]:
        return self.position(s_query), self.tangent(s_query)

    def closest_geometry(
        self, point: Any
    ) -> tuple[float, Array, float, int, float, Array]:
        """Return continuous nearest-segment geometry for a query point.

        Returns ``(s, closest_point, distance, segment_index,
        segment_fraction, displacement)``.  The displacement points from the
        closest lumen point to the query point.  Using segment projection
        avoids the false distance jumps caused by nearest-node lookup.
        """
        query = _finite_vector(point, 3, "projection point")
        starts = self.C[:-1]
        vectors = self.C[1:] - starts
        length_sq = np.sum(vectors * vectors, axis=1)
        fractions = np.sum((query - starts) * vectors, axis=1) / length_sq
        fractions = np.clip(fractions, 0.0, 1.0)
        candidates = starts + fractions[:, None] * vectors
        distances = np.linalg.norm(candidates - query, axis=1)
        index = int(np.argmin(distances))
        s_value = float(self.s[index] + fractions[index] * self.segment_lengths[index])
        displacement = query - candidates[index]
        return (
            s_value,
            candidates[index].copy(),
            float(distances[index]),
            index,
            float(fractions[index]),
            displacement.copy(),
        )

    def project(self, point: Any) -> tuple[float, Array, float]:
        """Return the closest polyline coordinate, point and Euclidean distance."""
        s_value, closest, distance, _, _, _ = self.closest_geometry(point)
        return s_value, closest, distance


def _physical_task_errors(
    output6: Any,
    desired_position: Any,
    desired_tangent: Any,
) -> tuple[float, float, Array, Array]:
    output = _finite_vector(output6, 6, "beam output")
    desired_p = _finite_vector(desired_position, 3, "desired position")
    desired_t = _unit_vector(desired_tangent, "desired tangent")
    tangent = _unit_vector(output[3:6], "beam tip tangent")
    position_error = float(np.linalg.norm(output[:3] - desired_p))
    dot = float(np.clip(tangent @ desired_t, -1.0, 1.0))
    tangent_error = float(np.arccos(dot))
    return position_error, tangent_error, output[:3].copy(), tangent


def _normalized_tangent_jacobian(tangent_raw: Array, J_tangent_raw: Array) -> Array:
    tangent_raw = _finite_vector(tangent_raw, 3, "raw tangent")
    J_tangent_raw = np.asarray(J_tangent_raw, dtype=float).reshape(3, 7)
    norm = float(np.linalg.norm(tangent_raw))
    if norm <= 1.0e-12:
        raise FloatingPointError("Cannot normalize a near-zero beam tangent.")
    tangent = tangent_raw / norm
    projection = np.eye(3) - np.outer(tangent, tangent)
    return (projection / norm) @ J_tangent_raw


def _effective_condition(matrix: Any) -> tuple[float, int, Array]:
    J = np.asarray(matrix, dtype=float)
    if J.ndim != 2 or not np.all(np.isfinite(J)):
        return np.inf, 0, np.empty(0, dtype=float)
    singular_values = np.linalg.svd(J, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return np.inf, 0, singular_values
    tolerance = max(J.shape) * np.finfo(float).eps * singular_values[0]
    active = singular_values[singular_values > tolerance]
    rank = int(active.size)
    condition = np.inf if rank == 0 else float(active[0] / active[-1])
    return condition, rank, singular_values


def _contact_model_flags(adapter: Any) -> dict[str, bool | None]:
    model = getattr(adapter, "model", None)
    contact_cfg = getattr(model, "contact_cfg", None)
    if contact_cfg is None:
        return {"enabled": None, "use_in_jacobian": None}
    return {
        "enabled": bool(getattr(contact_cfg, "enabled", False)),
        "use_in_jacobian": bool(getattr(contact_cfg, "use_in_jacobian", False)),
    }


def _collect_diagnostics(adapter: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    getter = getattr(adapter, "get_last_diag", None)
    if callable(getter):
        try:
            diagnostics.update(dict(getter() or {}))
        except Exception:
            pass

    model = getattr(adapter, "model", None)
    model_getter = getattr(model, "get_last_jacobian_diag", None)
    if callable(model_getter):
        try:
            diagnostics.update(dict(model_getter() or {}))
        except Exception:
            pass
    model_diag = getattr(model, "last_jacobian_diag", None)
    if isinstance(model_diag, dict):
        diagnostics.update(model_diag)
    return diagnostics


def _first_finite_diagnostic(
    diagnostics: dict[str, Any], keys: Sequence[str], default: float = np.nan
) -> float:
    for key in keys:
        if key not in diagnostics:
            continue
        try:
            value = float(np.asarray(diagnostics[key], dtype=float).reshape(-1)[0])
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _contact_diagnostics(adapter: Any) -> tuple[bool, float]:
    diagnostics = _collect_diagnostics(adapter)
    active = False
    for key in (
        "contact_active",
        "has_contact",
        "contact_is_active",
    ):
        if key in diagnostics:
            try:
                active = bool(np.any(np.asarray(diagnostics[key])))
                break
            except Exception:
                pass
    if not active:
        for key in (
            "num_contact_nodes",
            "number_contact_nodes",
            "contact_count",
        ):
            if key in diagnostics:
                try:
                    active = float(np.asarray(diagnostics[key]).reshape(-1)[0]) > 0.0
                    break
                except Exception:
                    pass

    minimum_clearance = _first_finite_diagnostic(
        diagnostics,
        (
            "minimum_clearance_m",
            "min_clearance_m",
            "minimum_gap_m",
            "min_gap_m",
            "contact_minimum_gap_m",
        ),
    )
    return active, minimum_clearance


def _magnet_pose6(adapter: Any, state: Array) -> Array:
    transform_fn = getattr(adapter, "magnet_transform", None)
    if not callable(transform_fn):
        return np.full(6, np.nan, dtype=float)
    T = np.asarray(transform_fn(state), dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        return np.full(6, np.nan, dtype=float)
    return np.concatenate((T[:3, 3], Rot.from_matrix(T[:3, :3]).as_rotvec()))


class _MagnetLumenExclusionConstraint:
    """Hard EPM-centre exclusion from the head around the lumen polyline.

    The dimensionless SLSQP constraint is

        g(state) = distance(state)^2 / radius^2 - 1 >= 0.

    On a fixed closest segment its exact configuration gradient is

        dg/dstate = 2 displacement.T @ J_magnet / radius^2.

    The closest-segment distance is continuous and piecewise differentiable.
    At the measure-zero boundaries where two segments are equally close, the
    gradient of the selected active segment is used.
    """

    _ANALYTICAL_METHOD_NAMES = (
        "magnet_position_jacobian",
        "continuous_magnet_position_jacobian",
        "source_magnet_position_jacobian",
    )

    def __init__(
        self,
        *,
        adapter: Any,
        path: CentrelinePath,
        state_min: Array,
        state_max: Array,
        config: InverseConfigurationPlannerConfig,
    ):
        radius = config.source_magnet_lumen_exclusion_radius_m
        if radius is None:
            raise ValueError(
                "A lumen-head exclusion constraint requires a configured radius."
            )
        self.adapter = adapter
        self.path = path
        self.state_min = _finite_vector(state_min, 7, "state_min")
        self.state_max = _finite_vector(state_max, 7, "state_max")
        self.config = config
        self.radius_m = float(radius)
        self._analytical_method = self._find_analytical_method()
        if (
            self._analytical_method is None
            and config.require_analytical_magnet_position_jacobian
        ):
            expected = ", ".join(self._ANALYTICAL_METHOD_NAMES)
            raise TypeError(
                "The hard source-magnet/lumen exclusion constraint was configured "
                "to require an analytical magnet-position Jacobian, but the "
                f"adapter exposes none of: {expected}."
            )

        self.jacobian_source = (
            str(
                getattr(
                    self._analytical_method,
                    "__name__",
                    type(self._analytical_method).__name__,
                )
            )
            if self._analytical_method is not None
            else "central_difference_of_magnet_transform"
        )
        self._x_position: Array | None = None
        self._position: Array | None = None
        self._x_jacobian: Array | None = None
        self._position_jacobian: Array | None = None

    def _find_analytical_method(self) -> Any | None:
        for name in self._ANALYTICAL_METHOD_NAMES:
            method = getattr(self.adapter, name, None)
            if callable(method):
                return method
        return None

    @staticmethod
    def _same(left: Array | None, right: Array) -> bool:
        return left is not None and np.array_equal(left, right)

    def position(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "magnet-position state")
        if not self._same(self._x_position, state):
            pose = _magnet_pose6(self.adapter, state)
            if not np.all(np.isfinite(pose[:3])):
                raise FloatingPointError(
                    "magnet_transform returned a non-finite source-magnet "
                    "position."
                )
            self._x_position = state.copy()
            self._position = pose[:3].copy()
        return np.asarray(self._position, dtype=float).copy()

    def _coerce_analytical_jacobian(self, value: Any) -> Array:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape == (3, 6):
            matrix = np.column_stack((matrix, np.zeros(3, dtype=float)))
        if matrix.shape != (3, 7) or not np.all(np.isfinite(matrix)):
            raise FloatingPointError(
                "An analytical magnet-position Jacobian must be finite with "
                f"shape (3, 7) or (3, 6); got {matrix.shape}."
            )
        return matrix.copy()

    def _finite_difference_jacobian(self, state: Array) -> Array:
        steps = np.full(
            7, float(self.config.magnet_jacobian_joint_step_rad), dtype=float
        )
        steps[6] = float(self.config.magnet_jacobian_insertion_step_m)
        centre = self.position(state)
        matrix = np.zeros((3, 7), dtype=float)
        for coordinate, nominal_step in enumerate(steps):
            plus_step = min(
                nominal_step,
                max(0.0, self.state_max[coordinate] - state[coordinate]),
            )
            minus_step = min(
                nominal_step,
                max(0.0, state[coordinate] - self.state_min[coordinate]),
            )
            if plus_step > 1.0e-15 and minus_step > 1.0e-15:
                plus = state.copy()
                minus = state.copy()
                plus[coordinate] += plus_step
                minus[coordinate] -= minus_step
                # The configured steps are normally equal.  This secant form
                # remains bound-safe when an iterate is very near one limit.
                matrix[:, coordinate] = (
                    self.position(plus) - self.position(minus)
                ) / (plus_step + minus_step)
            elif plus_step > 1.0e-15:
                plus = state.copy()
                plus[coordinate] += plus_step
                matrix[:, coordinate] = (
                    self.position(plus) - centre
                ) / plus_step
            elif minus_step > 1.0e-15:
                minus = state.copy()
                minus[coordinate] -= minus_step
                matrix[:, coordinate] = (
                    centre - self.position(minus)
                ) / minus_step
            else:
                matrix[:, coordinate] = 0.0
        if not np.all(np.isfinite(matrix)):
            raise FloatingPointError(
                "Finite-difference magnet-position Jacobian is non-finite."
            )
        return matrix

    def position_jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "magnet-Jacobian state")
        if not self._same(self._x_jacobian, state):
            if self._analytical_method is not None:
                matrix = self._coerce_analytical_jacobian(
                    self._analytical_method(state)
                )
            else:
                matrix = self._finite_difference_jacobian(state)
            self._x_jacobian = state.copy()
            self._position_jacobian = matrix.copy()
        return np.asarray(self._position_jacobian, dtype=float).copy()

    def geometry(
        self, state: Any
    ) -> tuple[float, Array, int, float, Array]:
        position = self.position(state)
        _, closest, distance, segment, fraction, displacement = (
            self.path.closest_geometry(position)
        )
        return distance, closest, segment, fraction, displacement

    def value(self, state: Any) -> float:
        distance, _, _, _, _ = self.geometry(state)
        return float((distance / self.radius_m) ** 2 - 1.0)

    def jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "exclusion-constraint state")
        _, _, _, _, displacement = self.geometry(state)
        J_magnet = self.position_jacobian(state)
        gradient = (2.0 / self.radius_m**2) * (displacement @ J_magnet)
        if gradient.shape != (7,) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "Source-magnet/lumen exclusion constraint Jacobian is invalid."
            )
        return gradient

    def physical_metrics(
        self, state: Any
    ) -> tuple[float, float, int, bool]:
        distance, _, segment, _, _ = self.geometry(state)
        margin = distance - self.radius_m
        satisfied = bool(
            margin
            >= -float(self.config.source_magnet_lumen_constraint_tolerance_m)
        )
        return float(distance), float(margin), int(segment), satisfied


class _NodeObjective:
    """Tolerance-normalized task residual and exact chained Jacobian."""

    def __init__(
        self,
        *,
        adapter: Any,
        desired_position: Array,
        desired_tangent: Array,
        previous_state: Array,
        state_min: Array,
        state_max: Array,
        config: InverseConfigurationPlannerConfig,
    ):
        self.adapter = adapter
        self.desired_position = _finite_vector(
            desired_position, 3, "desired position"
        )
        self.desired_tangent = _unit_vector(desired_tangent, "desired tangent")
        self.previous_state = _finite_vector(previous_state, 7, "previous state")
        self.state_min = _finite_vector(state_min, 7, "state_min")
        self.state_max = _finite_vector(state_max, 7, "state_max")
        self.config = config

        self.tangent_scale = 2.0 * math.sin(
            0.5 * float(config.tangent_tolerance_rad)
        )
        self.step_scale = np.concatenate(
            (
                np.asarray(config.maximum_joint_step_rad, dtype=float),
                np.array([float(config.maximum_insertion_step_m)]),
            )
        )
        self.continuity_sqrt = np.sqrt(
            np.asarray(config.continuity_weight, dtype=float)
        )
        self.centre_sqrt = np.sqrt(
            np.asarray(config.joint_centre_weight, dtype=float)
        )

        span = self.state_max - self.state_min
        self.state_centre = 0.5 * (self.state_max + self.state_min)
        self.state_half_range = np.where(span > 1.0e-12, 0.5 * span, 1.0)
        self._x_output: Array | None = None
        self._output: Array | None = None
        self._x_jacobian: Array | None = None
        self._jacobian: Array | None = None

    @staticmethod
    def _same(left: Array | None, right: Array) -> bool:
        return left is not None and np.array_equal(left, right)

    def output(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "objective state")
        if not self._same(self._x_output, state):
            output = np.asarray(
                self.adapter.forward_output(state, commit=False), dtype=float
            ).reshape(-1)
            if output.size < 6 or not np.all(np.isfinite(output[:6])):
                raise FloatingPointError(
                    "Full diagnostic adapter returned an invalid beam output."
                )
            self._x_output = state.copy()
            self._output = output[:6].copy()
        return np.asarray(self._output, dtype=float).copy()

    def refresh_output(self, state: Any, *, commit: bool = False) -> Array:
        """Force a new physical forward solve at ``state``.

        ``output()`` intentionally caches repeated residual evaluations at the
        same least-squares iterate.  That cache must not be used as the final
        physical feasibility check: the beam equilibrium solver is stateful
        and can otherwise leave a saved node paired with an earlier output.

        When ``commit=True`` the accepted equilibrium becomes the continuation
        baseline for the next inverse-path node.
        """
        state = _finite_vector(state, 7, "objective refresh state")
        output = np.asarray(
            self.adapter.forward_output(state, commit=bool(commit)),
            dtype=float,
        ).reshape(-1)
        if output.size < 6 or not np.all(np.isfinite(output[:6])):
            raise FloatingPointError(
                "Full diagnostic adapter returned an invalid refreshed "
                "beam output."
            )
        self._x_output = state.copy()
        self._output = output[:6].copy()
        return self._output.copy()

    def output_jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "Jacobian state")
        if not self._same(self._x_jacobian, state):
            matrix = np.asarray(
                self.adapter.continuous_output_jacobian(state), dtype=float
            )
            if matrix.shape != (6, 7) or not np.all(np.isfinite(matrix)):
                raise FloatingPointError(
                    "Full diagnostic adapter must return a finite 6x7 "
                    "continuous Jacobian."
                )
            self._x_jacobian = state.copy()
            self._jacobian = matrix.copy()
        return np.asarray(self._jacobian, dtype=float).copy()

    def residual(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "residual state")
        output = self.output(state)
        tangent = _unit_vector(output[3:6], "beam tangent")
        position_residual = (
            output[:3] - self.desired_position
        ) / float(self.config.position_tolerance_m)
        tangent_residual = (
            tangent - self.desired_tangent
        ) / self.tangent_scale
        continuity_residual = self.continuity_sqrt * (
            (state - self.previous_state) / self.step_scale
        )
        centre_residual = self.centre_sqrt * (
            (state - self.state_centre) / self.state_half_range
        )
        residual = np.concatenate(
            (
                position_residual,
                tangent_residual,
                continuity_residual,
                centre_residual,
            )
        )
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Inverse objective produced non-finite residuals.")
        return residual

    def jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "residual-Jacobian state")
        output = self.output(state)
        J_output = self.output_jacobian(state)
        J_position = J_output[:3, :] / float(self.config.position_tolerance_m)
        J_tangent = _normalized_tangent_jacobian(
            output[3:6], J_output[3:6, :]
        ) / self.tangent_scale
        J_continuity = np.diag(self.continuity_sqrt / self.step_scale)
        J_centre = np.diag(self.centre_sqrt / self.state_half_range)
        result = np.vstack(
            (J_position, J_tangent, J_continuity, J_centre)
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Inverse objective Jacobian is non-finite.")
        return result

    def task_jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "task-Jacobian state")
        output = self.output(state)
        J_output = self.output_jacobian(state)
        return np.vstack(
            (
                J_output[:3, :] / float(self.config.position_tolerance_m),
                _normalized_tangent_jacobian(
                    output[3:6], J_output[3:6, :]
                )
                / self.tangent_scale,
            )
        )

    def scalar_objective(self, state: Any) -> float:
        residual = self.residual(state)
        return float(residual @ residual)

    def scalar_objective_gradient(self, state: Any) -> Array:
        residual = self.residual(state)
        gradient = 2.0 * self.jacobian(state).T @ residual
        if gradient.shape != (7,) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Inverse scalar-objective gradient is invalid.")
        return gradient


def _strictly_feasible_initial_guess(
    guess: Array, lower: Array, upper: Array
) -> Array:
    """Move a bounded least-squares guess a tiny amount inside finite bounds."""
    guess = np.clip(np.asarray(guess, dtype=float), lower, upper)
    span = upper - lower
    epsilon = np.maximum(1.0e-12, 1.0e-10 * np.maximum(1.0, np.abs(span)))
    interior_lower = np.where(np.isfinite(lower), lower + epsilon, lower)
    interior_upper = np.where(np.isfinite(upper), upper - epsilon, upper)
    valid = interior_lower < interior_upper
    result = guess.copy()
    result[valid] = np.minimum(
        np.maximum(result[valid], interior_lower[valid]), interior_upper[valid]
    )
    return result


def _local_state_bounds(
    *,
    previous_state: Array,
    state_min: Array,
    state_max: Array,
    config: InverseConfigurationPlannerConfig,
) -> tuple[Array, Array]:
    maximum_change = np.concatenate(
        (
            np.asarray(config.maximum_joint_step_rad, dtype=float),
            np.array([float(config.maximum_insertion_step_m)]),
        )
    )
    lower = np.maximum(state_min, previous_state - maximum_change)
    upper = np.minimum(state_max, previous_state + maximum_change)
    if config.insertion_non_decreasing:
        lower[6] = max(lower[6], previous_state[6])

    # SciPy requires strict lower < upper.  At a hard insertion limit we retain
    # only a numerical-width interval inside the global hard bounds.
    for coordinate in range(7):
        if upper[coordinate] - lower[coordinate] <= 2.0e-12:
            if state_max[coordinate] - state_min[coordinate] <= 2.0e-12:
                raise ValueError(f"State coordinate {coordinate} has no valid range.")
            if np.isclose(upper[coordinate], state_max[coordinate]):
                lower[coordinate] = max(
                    state_min[coordinate], upper[coordinate] - 2.0e-12
                )
            else:
                upper[coordinate] = min(
                    state_max[coordinate], lower[coordinate] + 2.0e-12
                )
    if np.any(lower >= upper):
        raise ValueError("Local continuation bounds contain an empty interval.")
    return lower, upper


def _jacobian_predicted_state(
    *,
    adapter: Any,
    current_state: Array,
    current_position: Array,
    current_tangent: Array,
    desired_position: Array,
    desired_tangent: Array,
    lower: Array,
    upper: Array,
    config: InverseConfigurationPlannerConfig,
) -> Array | None:
    """First-order continuation predictor for the next node.

    Uses the analytical 6x7 chained output Jacobian at ``current_state`` to
    solve a damped least-norm joint step for the desired change in tip
    position (and, when ``jacobian_predictor_use_tangent``, tangent):

        dx = J_task^T (J_task J_task^T + lam^2 I)^-1 d_task

    then clips it to the per-node maximum joint/insertion change and the hard
    state bounds.  This replaces the secant ``2 x_k - x_{k-1}`` guess with one
    that accounts for the local sensitivity, the actual path step, and a
    rank-deficient Jacobian (the damped solve projects onto the controllable
    subspace instead of amplifying the near-null directions).

    Returns ``None`` when the Jacobian cannot be formed or the solve fails, so
    the caller falls back to the secant / previous-state guesses.
    """
    try:
        J = np.asarray(
            adapter.continuous_output_jacobian(
                _finite_vector(current_state, 7, "predictor state")
            ),
            dtype=float,
        ).reshape(6, 7)
    except Exception:
        return None
    if not np.all(np.isfinite(J)):
        return None

    lam = float(config.jacobian_predictor_damping)
    d_position = np.asarray(desired_position, dtype=float) - np.asarray(
        current_position, dtype=float
    )
    if config.jacobian_predictor_use_tangent:
        # Position is in metres, the tangent term is a dimensionless unit-vector
        # difference; scale each by its own tolerance so the least-norm solve
        # weights them the way the node objective does.  Near a sharp bend the
        # tangent term tends to demand a large, position-degrading joint swing,
        # which is why it is opt-in.
        position_tolerance = float(config.position_tolerance_m)
        tangent_scale = max(
            2.0 * math.sin(0.5 * float(config.tangent_tolerance_rad)), 1.0e-12
        )
        try:
            tangent_rows = _normalized_tangent_jacobian(
                _finite_vector(current_tangent, 3, "predictor tangent"),
                J[3:6, :],
            )
            d_tangent = (
                _unit_vector(desired_tangent, "desired tangent")
                - _unit_vector(current_tangent, "current tangent")
            )
        except (FloatingPointError, ValueError):
            return None
        task_rows = np.vstack(
            (J[:3, :] / position_tolerance, tangent_rows / tangent_scale)
        )
        d_task = np.concatenate(
            (d_position / position_tolerance, d_tangent / tangent_scale)
        )
    else:
        # Position-only: raw metres / radians, so ``lam`` is a plain Levenberg
        # parameter in m/rad and the solve is not silently rescaled.
        task_rows = J[:3, :]
        d_task = d_position

    gram = task_rows @ task_rows.T + (lam**2) * np.eye(task_rows.shape[0])
    try:
        dx = task_rows.T @ np.linalg.solve(gram, d_task)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(dx)):
        return None

    step_cap = np.concatenate(
        (
            np.asarray(config.maximum_joint_step_rad, dtype=float),
            np.array([float(config.maximum_insertion_step_m)]),
        )
    )
    dx = np.clip(dx, -step_cap, step_cap)
    predicted = np.clip(
        _finite_vector(current_state, 7, "predictor state") + dx, lower, upper
    )
    return predicted


def _candidate_guesses(
    *,
    previous_state: Array,
    extrapolated_state: Array | None,
    jacobian_predicted_state: Array | None,
    alternative_initial_states: Sequence[Array],
    lower: Array,
    upper: Array,
    config: InverseConfigurationPlannerConfig,
) -> list[Array]:
    guesses: list[Array] = []

    def append_unique(value: Any) -> None:
        candidate = _strictly_feasible_initial_guess(
            _finite_vector(value, 7, "initial guess"), lower, upper
        )
        if not any(np.allclose(candidate, old, atol=1.0e-12, rtol=0.0) for old in guesses):
            guesses.append(candidate)

    if config.use_jacobian_predictor_guess and jacobian_predicted_state is not None:
        append_unique(jacobian_predicted_state)
    if config.use_extrapolated_guess and extrapolated_state is not None:
        append_unique(extrapolated_state)
    append_unique(previous_state)
    for state in alternative_initial_states:
        append_unique(state)

    rng = np.random.default_rng(int(config.random_seed))
    joint_scale = float(config.multistart_joint_perturbation_rad)
    insertion_scale = float(config.multistart_insertion_perturbation_m)
    while len(guesses) < int(config.maximum_multistart_attempts):
        perturbation = np.concatenate(
            (
                rng.uniform(-joint_scale, joint_scale, size=6),
                np.array([rng.uniform(-insertion_scale, insertion_scale)]),
            )
        )
        append_unique(previous_state + perturbation)
        if joint_scale == 0.0 and insertion_scale == 0.0:
            break

    return guesses[: int(config.maximum_multistart_attempts)]


def _validate_adapter_chain_rule(
    adapter: Any,
    state: Array,
    config: InverseConfigurationPlannerConfig,
) -> dict[str, Any] | None:
    validator = getattr(adapter, "validate_chain_rule", None)
    if not callable(validator):
        return None
    result = dict(
        validator(
            state,
            joint_step_rad=float(config.finite_difference_joint_step_rad),
            insertion_step_m=float(config.finite_difference_insertion_step_m),
        )
    )
    relative = float(result.get("relative_frobenius_error", np.inf))
    if not np.isfinite(relative):
        raise FloatingPointError("Chain-rule validation returned a non-finite error.")
    if relative > float(config.maximum_chain_rule_relative_error):
        raise RuntimeError(
            "The existing chained Jacobian failed finite-difference validation: "
            f"relative error={relative:.6e}, configured maximum="
            f"{config.maximum_chain_rule_relative_error:.6e}."
        )
    return result


def _build_node_result(
    *,
    node_index: int,
    s_m: float,
    state: Array,
    previous_state: Array,
    desired_position: Array,
    desired_tangent: Array,
    adapter: Any,
    objective: _NodeObjective,
    exclusion_constraint: _MagnetLumenExclusionConstraint | None,
    config: InverseConfigurationPlannerConfig,
    state_min: Array,
    state_max: Array,
    solver_success: bool,
    termination_reason: str,
    function_evaluations: int,
    objective_value: float,
    attempt_count: int,
    chain_rule_relative_error: float = np.nan,
) -> InverseNodeResult:
    state = _finite_vector(state, 7, "node state")
    output = objective.output(state)
    position_error, tangent_error, tip_position, tip_tangent = (
        _physical_task_errors(output, desired_position, desired_tangent)
    )
    within_bounds = bool(
        np.all(state >= state_min - 1.0e-12)
        and np.all(state <= state_max + 1.0e-12)
    )
    if exclusion_constraint is None:
        exclusion_distance = np.nan
        exclusion_radius = np.nan
        exclusion_margin = np.nan
        exclusion_segment = -1
        exclusion_satisfied = True
    else:
        (
            exclusion_distance,
            exclusion_margin,
            exclusion_segment,
            exclusion_satisfied,
        ) = (
            exclusion_constraint.physical_metrics(state)
        )
        exclusion_radius = float(exclusion_constraint.radius_m)
    feasible = bool(
        within_bounds
        and position_error <= float(config.position_tolerance_m)
        and tangent_error <= float(config.tangent_tolerance_rad)
        and exclusion_satisfied
    )

    task_jacobian = objective.task_jacobian(state)
    condition, rank, _ = _effective_condition(task_jacobian)
    joint_margin = np.minimum(
        state[:6] - state_min[:6], state_max[:6] - state[:6]
    )
    state_margin = np.minimum(state - state_min, state_max - state)
    contact_active, minimum_clearance = _contact_diagnostics(adapter)

    return InverseNodeResult(
        node_index=int(node_index),
        s_m=float(s_m),
        q_rad=state[:6].copy(),
        insertion_m=float(state[6]),
        magnet_pose6=_magnet_pose6(adapter, state),
        tip_position_m=tip_position,
        tip_tangent=tip_tangent,
        desired_position_m=np.asarray(desired_position, dtype=float).copy(),
        desired_tangent=_unit_vector(desired_tangent, "desired tangent"),
        position_error_m=position_error,
        tangent_error_rad=tangent_error,
        feasible=feasible,
        solver_success=bool(solver_success),
        termination_reason=str(termination_reason),
        function_evaluations=int(function_evaluations),
        objective_value=float(objective_value),
        jacobian_condition=float(condition),
        jacobian_effective_rank=int(rank),
        minimum_joint_margin_rad=float(np.min(joint_margin)),
        minimum_state_margin=float(np.min(state_margin)),
        minimum_clearance_m=float(minimum_clearance),
        contact_active=bool(contact_active),
        source_magnet_lumen_distance_m=float(exclusion_distance),
        source_magnet_lumen_exclusion_radius_m=float(exclusion_radius),
        source_magnet_lumen_margin_m=float(exclusion_margin),
        source_magnet_lumen_segment_index=int(exclusion_segment),
        source_magnet_lumen_constraint_satisfied=bool(exclusion_satisfied),
        delta_q_norm_rad=float(np.linalg.norm(state[:6] - previous_state[:6])),
        delta_insertion_m=float(state[6] - previous_state[6]),
        attempt_count=int(attempt_count),
        chain_rule_relative_error=float(chain_rule_relative_error),
    )


def _solve_one_node(
    *,
    node_index: int,
    s_m: float,
    desired_position: Array,
    desired_tangent: Array,
    previous_state: Array,
    extrapolated_state: Array | None,
    jacobian_predicted_state: Array | None,
    alternative_initial_states: Sequence[Array],
    adapter: Any,
    state_min: Array,
    state_max: Array,
    config: InverseConfigurationPlannerConfig,
    exclusion_constraint: _MagnetLumenExclusionConstraint | None,
) -> InverseNodeResult:
    start_step = getattr(adapter, "start_step", None)
    if callable(start_step):
        start_step()

    local_lower, local_upper = _local_state_bounds(
        previous_state=previous_state,
        state_min=state_min,
        state_max=state_max,
        config=config,
    )
    objective = _NodeObjective(
        adapter=adapter,
        desired_position=desired_position,
        desired_tangent=desired_tangent,
        previous_state=previous_state,
        state_min=state_min,
        state_max=state_max,
        config=config,
    )
    guesses = _candidate_guesses(
        previous_state=previous_state,
        extrapolated_state=extrapolated_state,
        jacobian_predicted_state=jacobian_predicted_state,
        alternative_initial_states=alternative_initial_states,
        lower=local_lower,
        upper=local_upper,
        config=config,
    )

    candidates: list[InverseNodeResult] = []
    for attempt_index, guess in enumerate(guesses, start=1):
        try:
            if exclusion_constraint is None:
                solved = least_squares(
                    objective.residual,
                    guess,
                    jac=objective.jacobian,
                    bounds=(local_lower, local_upper),
                    method="trf",
                    x_scale="jac",
                    ftol=float(config.ftol),
                    xtol=float(config.xtol),
                    gtol=float(config.gtol),
                    max_nfev=int(config.maximum_function_evaluations),
                    verbose=0,
                )
                state = np.asarray(solved.x, dtype=float).reshape(7)
                reason = f"scipy_status_{solved.status}: {solved.message}"
                solver_objective = float(2.0 * solved.cost)
            else:
                # Tolerance-normalized task residuals can make the raw scalar
                # objective very large when a continuation guess is several
                # position tolerances from the next node.  Positive constant
                # scaling leaves the constrained minimizer unchanged and
                # avoids poor SLSQP line-search conditioning.
                objective_scale = max(
                    1.0, float(objective.scalar_objective(guess))
                )
                solved = minimize(
                    lambda state_value: (
                        objective.scalar_objective(state_value)
                        / objective_scale
                    ),
                    guess,
                    jac=lambda state_value: (
                        objective.scalar_objective_gradient(state_value)
                        / objective_scale
                    ),
                    bounds=Bounds(local_lower, local_upper),
                    constraints=(
                        {
                            "type": "ineq",
                            "fun": exclusion_constraint.value,
                            "jac": exclusion_constraint.jacobian,
                        },
                    ),
                    method="SLSQP",
                    options={
                        "maxiter": int(config.maximum_function_evaluations),
                        "ftol": float(config.ftol),
                        "disp": False,
                    },
                )
                state = np.asarray(solved.x, dtype=float).reshape(7)
                reason = (
                    f"scipy_slsqp_status_{solved.status}: {solved.message}; "
                    "exclusion_jacobian="
                    f"{exclusion_constraint.jacobian_source}"
                )
                solver_objective = float(objective.scalar_objective(state))

            # Do not certify the node from the residual callback's cached
            # output.  Re-run the physical forward model from this node's
            # frozen continuation baseline at SciPy's returned state.
            objective.refresh_output(state, commit=False)
            candidate = _build_node_result(
                node_index=node_index,
                s_m=s_m,
                state=state,
                previous_state=previous_state,
                desired_position=desired_position,
                desired_tangent=desired_tangent,
                adapter=adapter,
                objective=objective,
                exclusion_constraint=exclusion_constraint,
                config=config,
                state_min=state_min,
                state_max=state_max,
                solver_success=bool(solved.success),
                termination_reason=reason,
                function_evaluations=int(solved.nfev),
                objective_value=solver_objective,
                attempt_count=attempt_index,
            )
        except Exception as exc:
            # Preserve a diagnostic result for a failed numerical attempt when
            # the guess itself can still be evaluated.
            try:
                residual = objective.residual(guess)
                objective.refresh_output(guess, commit=False)
                candidate = _build_node_result(
                    node_index=node_index,
                    s_m=s_m,
                    state=guess,
                    previous_state=previous_state,
                    desired_position=desired_position,
                    desired_tangent=desired_tangent,
                    adapter=adapter,
                    objective=objective,
                    exclusion_constraint=exclusion_constraint,
                    config=config,
                    state_min=state_min,
                    state_max=state_max,
                    solver_success=False,
                    termination_reason=f"exception: {type(exc).__name__}: {exc}",
                    function_evaluations=0,
                    objective_value=float(residual @ residual),
                    attempt_count=attempt_index,
                )
            except Exception:
                continue

        candidates.append(candidate)
        if candidate.feasible:
            # Advance the beam continuation only after a candidate has passed
            # an independent physical post-validation.  Previously no
            # accepted inverse node was ever committed, so start_step() kept
            # freezing an obsolete beam equilibrium.
            precommit_position = candidate.tip_position_m.copy()
            accepted_state = np.concatenate(
                (candidate.q_rad, np.array([candidate.insertion_m]))
            )
            objective.refresh_output(accepted_state, commit=True)
            committed_candidate = _build_node_result(
                node_index=node_index,
                s_m=s_m,
                state=accepted_state,
                previous_state=previous_state,
                desired_position=desired_position,
                desired_tangent=desired_tangent,
                adapter=adapter,
                objective=objective,
                exclusion_constraint=exclusion_constraint,
                config=config,
                state_min=state_min,
                state_max=state_max,
                solver_success=candidate.solver_success,
                termination_reason=(
                    candidate.termination_reason
                    + "; fresh_postvalidated_and_committed"
                ),
                function_evaluations=candidate.function_evaluations,
                objective_value=candidate.objective_value,
                attempt_count=candidate.attempt_count,
            )
            commit_drift = float(
                np.linalg.norm(
                    committed_candidate.tip_position_m - precommit_position
                )
            )
            if not committed_candidate.feasible:
                raise RuntimeError(
                    "A candidate passed commit=False post-validation but "
                    "became infeasible when the same state was committed; "
                    f"node={node_index}, s={s_m:.6f} m, "
                    f"commit_tip_drift={1.0e3 * commit_drift:.6f} mm. "
                    "The forward equilibrium is not repeatable from the "
                    "frozen node baseline."
                )
            return committed_candidate

    if not candidates:
        raise RuntimeError(
            f"Every inverse-solve attempt failed numerically at s={s_m:.6f} m."
        )

    # Prefer physical task accuracy, then the complete regularized objective.
    return min(
        candidates,
        key=lambda item: (
            0.0
            if item.source_magnet_lumen_constraint_satisfied
            else max(0.0, -item.source_magnet_lumen_margin_m),
            item.position_error_m / float(config.position_tolerance_m)
            + item.tangent_error_rad / float(config.tangent_tolerance_rad),
            item.objective_value,
        ),
    )


def _fixed_initial_node(
    *,
    state: Array,
    s_m: float,
    desired_position: Array,
    desired_tangent: Array,
    adapter: Any,
    state_min: Array,
    state_max: Array,
    config: InverseConfigurationPlannerConfig,
    chain_rule_relative_error: float,
    exclusion_constraint: _MagnetLumenExclusionConstraint | None,
) -> InverseNodeResult:
    objective = _NodeObjective(
        adapter=adapter,
        desired_position=desired_position,
        desired_tangent=desired_tangent,
        previous_state=state,
        state_min=state_min,
        state_max=state_max,
        config=config,
    )
    residual = objective.residual(state)

    # Certify the fixed initial node with a fresh forward solve, then commit
    # that exact equilibrium so the first movable node has an explicit and
    # reproducible continuation baseline.
    objective.refresh_output(state, commit=False)
    result = _build_node_result(
        node_index=0,
        s_m=s_m,
        state=state,
        previous_state=state,
        desired_position=desired_position,
        desired_tangent=desired_tangent,
        adapter=adapter,
        objective=objective,
        exclusion_constraint=exclusion_constraint,
        config=config,
        state_min=state_min,
        state_max=state_max,
        solver_success=True,
        termination_reason="fixed_initial_state",
        function_evaluations=1,
        objective_value=float(residual @ residual),
        attempt_count=1,
        chain_rule_relative_error=chain_rule_relative_error,
    )
    if result.feasible:
        precommit_position = result.tip_position_m.copy()
        objective.refresh_output(state, commit=True)
        committed_result = _build_node_result(
            node_index=0,
            s_m=s_m,
            state=state,
            previous_state=state,
            desired_position=desired_position,
            desired_tangent=desired_tangent,
            adapter=adapter,
            objective=objective,
            exclusion_constraint=exclusion_constraint,
            config=config,
            state_min=state_min,
            state_max=state_max,
            solver_success=True,
            termination_reason="fixed_initial_state_feasible_and_committed",
            function_evaluations=1,
            objective_value=float(residual @ residual),
            attempt_count=1,
            chain_rule_relative_error=chain_rule_relative_error,
        )
        commit_drift = float(
            np.linalg.norm(
                committed_result.tip_position_m - precommit_position
            )
        )
        if not committed_result.feasible:
            raise RuntimeError(
                "The fixed initial state passed commit=False validation but "
                "became infeasible when committed; "
                f"commit_tip_drift={1.0e3 * commit_drift:.6f} mm."
            )
        result = committed_result
    else:
        result.termination_reason = "fixed_initial_state_outside_task_tolerance"
    return result


def _print_node(node: InverseNodeResult) -> None:
    condition_text = (
        f"{node.jacobian_condition:.3e}"
        if np.isfinite(node.jacobian_condition)
        else "inf"
    )
    exclusion_text = ""
    if np.isfinite(node.source_magnet_lumen_distance_m):
        exclusion_text = (
            "\n  EPM-lumen distance="
            f"{1.0e3 * node.source_magnet_lumen_distance_m:.3f} mm "
            "margin="
            f"{1.0e3 * node.source_magnet_lumen_margin_m:.3f} mm "
            f"segment={node.source_magnet_lumen_segment_index}"
        )
    print(
        f"[INVERSE PATH] node={node.node_index:04d} s={node.s_m:.6f} m\n"
        f"  position_error={1.0e3 * node.position_error_m:.4f} mm\n"
        f"  tangent_error={np.degrees(node.tangent_error_rad):.4f} deg\n"
        f"  evaluations={node.function_evaluations} attempts={node.attempt_count}\n"
        f"  ||delta_q||={node.delta_q_norm_rad:.6f} rad "
        f"delta_L={1.0e3 * node.delta_insertion_m:.4f} mm\n"
        f"  cond_eff(J)={condition_text} rank={node.jacobian_effective_rank}\n"
        f"  contact_active={node.contact_active} feasible={node.feasible}"
        f"{exclusion_text}"
    )


def solve_offline_inverse_configuration(
    *,
    adapter: Any,
    initial_state: Any,
    state_min: Any,
    state_max: Any,
    lumen_C: Any,
    config: InverseConfigurationPlannerConfig,
    output_dir: str | Path | None = None,
    alternative_initial_states: Iterable[Any] = (),
) -> InversePathResult:
    """Solve the contact-aware offline inverse configuration path.

    Parameters
    ----------
    adapter:
        Full six-output ``JointSpaceBeamMPCAdapter``.  The normal integration
        route passes ``controller_pack['plant_diagnostic_joint_adapter']``.

    initial_state:
        Fixed initial ``[q1..q6, insertion]`` state.

    state_min, state_max:
        Hard absolute joint and insertion bounds.  These are the same vectors
        returned by the current joint-space controller factory.

    lumen_C:
        Centreline array in the same robot-base/world frame as the beam output
        and source-magnet transform.  When
        ``source_magnet_lumen_exclusion_radius_m`` is set, the complete
        polyline is also the axis of the hard source-magnet exclusion region.

    output_dir:
        Optional diagnostics directory.  When supplied, CSV, JSON, NPZ and PNG
        files are written.
    """
    config.validate()
    state0 = _finite_vector(initial_state, 7, "initial_state")
    lower = _finite_vector(state_min, 7, "state_min")
    upper = _finite_vector(state_max, 7, "state_max")
    if np.any(lower >= upper):
        raise ValueError("Every hard state lower bound must be below its upper bound.")
    if np.any(state0 < lower) or np.any(state0 > upper):
        raise ValueError("initial_state lies outside the hard state bounds.")

    required_methods = (
        "forward_output",
        "continuous_output_jacobian",
        "magnet_transform",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise TypeError(
            "adapter is missing required JointSpaceBeamMPCAdapter methods: "
            + ", ".join(missing)
        )
    if int(getattr(adapter, "n_out", 6)) != 6:
        raise ValueError(
            "Offline position+tangent planning requires the full six-output "
            "diagnostic adapter, not the current n_out=3 MPC adapter."
        )

    contact_flags = _contact_model_flags(adapter)
    if config.require_contact_model:
        if contact_flags["enabled"] is not True:
            raise RuntimeError(
                "require_contact_model=True, but the supplied adapter's model "
                "does not report contact_cfg.enabled=True."
            )
        if contact_flags["use_in_jacobian"] is not True:
            raise RuntimeError(
                "require_contact_model=True, but contact is not enabled in the "
                "beam Jacobian."
            )

    path = CentrelinePath(lumen_C)
    exclusion_constraint = None
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        exclusion_constraint = _MagnetLumenExclusionConstraint(
            adapter=adapter,
            path=path,
            state_min=lower,
            state_max=upper,
            config=config,
        )
    alternative_states = [
        _finite_vector(value, 7, "alternative initial state")
        for value in alternative_initial_states
    ]

    start_step = getattr(adapter, "start_step", None)
    if callable(start_step):
        start_step()
    initial_output = np.asarray(
        adapter.forward_output(state0, commit=False), dtype=float
    ).reshape(-1)
    if initial_output.size < 6 or not np.all(np.isfinite(initial_output[:6])):
        raise FloatingPointError("Initial beam output is invalid.")

    if config.start_at_initial_tip_projection:
        s_start, _, projection_distance = path.project(initial_output[:3])
    else:
        s_start = 0.0
        projection_distance = float(
            np.linalg.norm(initial_output[:3] - path.position(0.0))
        )
    desired_position0, desired_tangent0 = path.reference(s_start)

    chain_rule_start = None
    chain_relative_start = np.nan
    if config.finite_difference_validation_at_start:
        chain_rule_start = _validate_adapter_chain_rule(adapter, state0, config)
        if chain_rule_start is not None:
            chain_relative_start = float(
                chain_rule_start.get("relative_frobenius_error", np.nan)
            )

    initial_node = _fixed_initial_node(
        state=state0,
        s_m=s_start,
        desired_position=desired_position0,
        desired_tangent=desired_tangent0,
        adapter=adapter,
        state_min=lower,
        state_max=upper,
        config=config,
        chain_rule_relative_error=chain_relative_start,
        exclusion_constraint=exclusion_constraint,
    )

    if not initial_node.feasible and config.solve_initial_node:
        initial_node = _solve_one_node(
            node_index=0,
            s_m=s_start,
            desired_position=desired_position0,
            desired_tangent=desired_tangent0,
            previous_state=state0,
            extrapolated_state=None,
            jacobian_predicted_state=None,
            alternative_initial_states=alternative_states,
            adapter=adapter,
            state_min=lower,
            state_max=upper,
            config=config,
            exclusion_constraint=exclusion_constraint,
        )
        initial_node.chain_rule_relative_error = chain_relative_start
        if initial_node.feasible:
            initial_node.termination_reason = (
                "initial_node_solved: " + initial_node.termination_reason
            )

    nodes = [initial_node]
    if config.debug:
        print(
            "[INVERSE PATH] initial projection "
            f"s={s_start:.6f} m distance={1.0e3 * projection_distance:.4f} mm"
        )
        _print_node(initial_node)

    if not initial_node.feasible:
        summary = {
            "all_nodes_feasible": False,
            "termination_reason": (
                "The fixed initial state does not satisfy the configured tip "
                "position/tangent tolerances and all enabled hard geometric "
                "constraints. Set solve_initial_node=True only if moving the "
                "initial state is scientifically intended."
            ),
            "path_total_length_m": path.total_length_m,
            "initial_progress_m": s_start,
            "initial_projection_distance_m": projection_distance,
            "contact_model": contact_flags,
            "source_magnet_lumen_exclusion_constraint": {
                "enabled": exclusion_constraint is not None,
                "radius_m": (
                    None
                    if exclusion_constraint is None
                    else exclusion_constraint.radius_m
                ),
                "jacobian_source": (
                    None
                    if exclusion_constraint is None
                    else exclusion_constraint.jacobian_source
                ),
            },
            "chain_rule_start": chain_rule_start,
        }
        result = InversePathResult(
            nodes=nodes,
            all_nodes_feasible=False,
            last_feasible_index=-1,
            last_feasible_progress_m=np.nan,
            configuration=_json_ready(config),
            summary=_json_ready(summary),
        )
        if output_dir is not None:
            save_inverse_path_result(result, output_dir)
        return result

    current_state = np.concatenate(
        (initial_node.q_rad, np.array([initial_node.insertion_m]))
    )
    current_s = float(s_start)
    step = min(float(config.initial_path_step_m), path.total_length_m - current_s)
    consecutive_successes = 0
    termination_reason = "path_end_reached"

    while current_s < path.total_length_m - 1.0e-12:
        step = max(float(config.minimum_path_step_m), step)
        target_s = min(path.total_length_m, current_s + step)
        desired_position, desired_tangent = path.reference(target_s)

        extrapolated = None
        if config.use_extrapolated_guess and len(nodes) >= 2:
            previous_previous_state = np.concatenate(
                (nodes[-2].q_rad, np.array([nodes[-2].insertion_m]))
            )
            extrapolated = current_state + (current_state - previous_previous_state)
            extrapolated = np.clip(extrapolated, lower, upper)

        jacobian_predicted = None
        if config.use_jacobian_predictor_guess:
            jacobian_predicted = _jacobian_predicted_state(
                adapter=adapter,
                current_state=current_state,
                current_position=nodes[-1].tip_position_m,
                current_tangent=nodes[-1].tip_tangent,
                desired_position=desired_position,
                desired_tangent=desired_tangent,
                lower=lower,
                upper=upper,
                config=config,
            )

        node = _solve_one_node(
            node_index=len(nodes),
            s_m=target_s,
            desired_position=desired_position,
            desired_tangent=desired_tangent,
            previous_state=current_state,
            extrapolated_state=extrapolated,
            jacobian_predicted_state=jacobian_predicted,
            alternative_initial_states=alternative_states,
            adapter=adapter,
            state_min=lower,
            state_max=upper,
            config=config,
            exclusion_constraint=exclusion_constraint,
        )

        stride = int(config.finite_difference_validation_stride)
        if node.feasible and stride > 0 and len(nodes) % stride == 0:
            state_node = np.concatenate((node.q_rad, np.array([node.insertion_m])))
            chain = _validate_adapter_chain_rule(adapter, state_node, config)
            if chain is not None:
                node.chain_rule_relative_error = float(
                    chain.get("relative_frobenius_error", np.nan)
                )

        if config.debug:
            _print_node(node)

        if node.feasible:
            nodes.append(node)
            current_state = np.concatenate(
                (node.q_rad, np.array([node.insertion_m]))
            )
            current_s = float(target_s)
            consecutive_successes += 1
            if consecutive_successes >= int(config.grow_step_after_successes):
                step = min(
                    float(config.maximum_path_step_m),
                    step * float(config.step_growth_factor),
                )
                consecutive_successes = 0
            continue

        consecutive_successes = 0
        if step > float(config.minimum_path_step_m) * (1.0 + 1.0e-12):
            new_step = max(float(config.minimum_path_step_m), 0.5 * step)
            if config.debug:
                print(
                    "[INVERSE PATH] refinement: failed at "
                    f"s={target_s:.6f} m; step {step:.6e} -> {new_step:.6e} m"
                )
            step = new_step
            continue

        nodes.append(node)
        termination_reason = (
            "No feasible configuration was found under the configured solver, "
            "tolerances, bounds, seeds and minimum path resolution."
        )
        break

    feasible_indices = [index for index, node in enumerate(nodes) if node.feasible]
    last_feasible_index = feasible_indices[-1] if feasible_indices else -1
    last_feasible_progress = (
        float(nodes[last_feasible_index].s_m)
        if last_feasible_index >= 0
        else np.nan
    )
    reached_end = bool(
        last_feasible_index >= 0
        and nodes[last_feasible_index].s_m >= path.total_length_m - 1.0e-12
    )
    all_feasible = bool(reached_end and all(node.feasible for node in nodes))

    position_errors = np.array([node.position_error_m for node in nodes], dtype=float)
    tangent_errors = np.array([node.tangent_error_rad for node in nodes], dtype=float)
    exclusion_distances = np.array(
        [node.source_magnet_lumen_distance_m for node in nodes], dtype=float
    )
    finite_exclusion_distances = exclusion_distances[
        np.isfinite(exclusion_distances)
    ]
    summary = {
        "all_nodes_feasible": all_feasible,
        "termination_reason": termination_reason,
        "num_nodes": len(nodes),
        "num_feasible_nodes": int(sum(node.feasible for node in nodes)),
        "last_feasible_index": last_feasible_index,
        "last_feasible_progress_m": last_feasible_progress,
        "path_total_length_m": path.total_length_m,
        "initial_progress_m": s_start,
        "initial_projection_distance_m": projection_distance,
        "maximum_position_error_m": float(np.max(position_errors)),
        "maximum_tangent_error_rad": float(np.max(tangent_errors)),
        "contact_model": contact_flags,
        "source_magnet_lumen_exclusion_constraint": {
            "enabled": exclusion_constraint is not None,
            "radius_m": (
                None
                if exclusion_constraint is None
                else exclusion_constraint.radius_m
            ),
            "jacobian_source": (
                None
                if exclusion_constraint is None
                else exclusion_constraint.jacobian_source
            ),
            "minimum_distance_m": (
                None
                if finite_exclusion_distances.size == 0
                else float(np.min(finite_exclusion_distances))
            ),
            "minimum_margin_m": (
                None
                if exclusion_constraint is None
                or finite_exclusion_distances.size == 0
                else float(
                    np.min(finite_exclusion_distances)
                    - exclusion_constraint.radius_m
                )
            ),
            "all_nodes_satisfied": bool(
                all(
                    node.source_magnet_lumen_constraint_satisfied
                    for node in nodes
                )
            ),
        },
        "chain_rule_start": chain_rule_start,
        "state_layout": "q1_q2_q3_q4_q5_q6_insertion",
        "jacobian_definition": "continuous d[tip_xyz,tangent_xyz]/d[q1..q6,L]",
    }
    result = InversePathResult(
        nodes=nodes,
        all_nodes_feasible=all_feasible,
        last_feasible_index=last_feasible_index,
        last_feasible_progress_m=last_feasible_progress,
        configuration=_json_ready(config),
        summary=_json_ready(summary),
    )
    if output_dir is not None:
        save_inverse_path_result(result, output_dir)
    return result


def solve_from_controller_pack(
    *,
    controller_pack: dict[str, Any],
    lumen_C: Any,
    config: InverseConfigurationPlannerConfig,
    output_dir: str | Path | None,
    alternative_initial_states: Iterable[Any] = (),
) -> InversePathResult:
    """Convenience integration with the existing joint-space factory output."""
    required = (
        "p0",
        "p_min",
        "p_max",
        "plant_diagnostic_joint_adapter",
    )
    missing = [key for key in required if key not in controller_pack]
    if missing:
        raise KeyError(
            "controller_pack is missing required entries: " + ", ".join(missing)
        )
    return solve_offline_inverse_configuration(
        adapter=controller_pack["plant_diagnostic_joint_adapter"],
        initial_state=controller_pack["p0"],
        state_min=controller_pack["p_min"],
        state_max=controller_pack["p_max"],
        lumen_C=lumen_C,
        config=config,
        output_dir=output_dir,
        alternative_initial_states=alternative_initial_states,
    )


def build_and_solve_from_experiment_config(
    *,
    exp_cfg: Any,
    design_cfg: Any,
    planner_config: InverseConfigurationPlannerConfig,
    robot_cfg: Any | None = None,
    output_dir: str | Path | None = None,
    alternative_initial_states: Iterable[Any] = (),
) -> InversePathResult:
    """Build the current simulation models and solve the offline inverse path.

    This is the executable repository-level entry point.  It mirrors the model
    construction at the start of ``run_experiment_grid.run_experiment`` but
    deliberately does not call ``run_simulation`` or the online MPC solver.

    ``exp_cfg`` is an existing ``ExperimentConfig`` and ``design_cfg`` is the
    existing ``ControllerDesignConfig`` used by the simulation.  Consequently,
    the lumen geometry, contact-aware plant, corrected UR model, calibrated
    magnet transform, initial magnet-pose IK, and absolute state bounds remain
    identical to the simulation configuration.
    """
    # Keep repository imports local so the module's deterministic self-test can
    # run without importing the complete research package.
    from proper_research.simulation.simulations.controller_factory_joint_space import (
        build_controller,
    )
    from proper_research.simulation.simulations.initial_conditions import (
        make_initial_poses,
    )
    from proper_research.simulation.simulations.model_factory import (
        build_model_bundle,
    )

    validator = getattr(exp_cfg, "validate", None)
    if callable(validator):
        validator()
    design_validator = getattr(design_cfg, "validate", None)
    if callable(design_validator):
        design_validator()

    pivot_point, start_point, L0, dt = make_initial_poses()
    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=exp_cfg.lumen,
        plant_contact=bool(exp_cfg.model.plant_contact),
    )
    plant_model = bundle.models["plant"]
    jacobian_model = bundle.models[exp_cfg.model.jacobian_variant]
    controller_pack = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant_model,
        jacobian_model=jacobian_model,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=design_cfg,
        robot_cfg=robot_cfg,
    )

    if output_dir is None:
        experiment_root = Path(getattr(exp_cfg, "out_root", Path.cwd()))
        output_dir = experiment_root / "offline_inverse_configuration"

    return solve_from_controller_pack(
        controller_pack=controller_pack,
        lumen_C=bundle.lumen_C,
        config=planner_config,
        output_dir=output_dir,
        alternative_initial_states=alternative_initial_states,
    )


def _node_csv_record(node: InverseNodeResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "node_index": node.node_index,
        "s_m": node.s_m,
        "insertion_m": node.insertion_m,
        "position_error_m": node.position_error_m,
        "position_error_mm": 1.0e3 * node.position_error_m,
        "tangent_error_rad": node.tangent_error_rad,
        "tangent_error_deg": np.degrees(node.tangent_error_rad),
        "feasible": int(node.feasible),
        "solver_success": int(node.solver_success),
        "termination_reason": node.termination_reason,
        "function_evaluations": node.function_evaluations,
        "objective_value": node.objective_value,
        "jacobian_condition": node.jacobian_condition,
        "jacobian_effective_rank": node.jacobian_effective_rank,
        "minimum_joint_margin_rad": node.minimum_joint_margin_rad,
        "minimum_state_margin": node.minimum_state_margin,
        "minimum_clearance_m": node.minimum_clearance_m,
        "contact_active": int(node.contact_active),
        "source_magnet_lumen_distance_m": (
            node.source_magnet_lumen_distance_m
        ),
        "source_magnet_lumen_exclusion_radius_m": (
            node.source_magnet_lumen_exclusion_radius_m
        ),
        "source_magnet_lumen_margin_m": node.source_magnet_lumen_margin_m,
        "source_magnet_lumen_segment_index": (
            node.source_magnet_lumen_segment_index
        ),
        "source_magnet_lumen_constraint_satisfied": int(
            node.source_magnet_lumen_constraint_satisfied
        ),
        "delta_q_norm_rad": node.delta_q_norm_rad,
        "delta_insertion_m": node.delta_insertion_m,
        "attempt_count": node.attempt_count,
        "chain_rule_relative_error": node.chain_rule_relative_error,
    }
    for index, value in enumerate(node.q_rad, start=1):
        record[f"q{index}_rad"] = float(value)
    for prefix, vector in (
        ("magnet_pose", node.magnet_pose6),
        ("tip", node.tip_position_m),
        ("tip_tangent", node.tip_tangent),
        ("desired", node.desired_position_m),
        ("desired_tangent", node.desired_tangent),
    ):
        labels = ("x", "y", "z", "rx", "ry", "rz") if vector.size == 6 else ("x", "y", "z")
        for label, value in zip(labels, vector):
            record[f"{prefix}_{label}"] = float(value)
    return record


def save_inverse_path_result(
    result: InversePathResult, output_dir: str | Path
) -> dict[str, str]:
    """Save CSV, JSON, NPZ and summary plot for an inverse path result."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "inverse_configuration_path.csv"
    json_path = output_root / "inverse_configuration_summary.json"
    npz_path = output_root / "inverse_configuration_diagnostics.npz"
    plot_path = output_root / "inverse_configuration_path.png"

    records = [_node_csv_record(node) for node in result.nodes]
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    payload = {
        "configuration": result.configuration,
        "summary": result.summary,
        "nodes": [_json_ready(node) for node in result.nodes],
    }
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, indent=2)

    s = np.array([node.s_m for node in result.nodes], dtype=float)
    states = np.array(
        [np.concatenate((node.q_rad, [node.insertion_m])) for node in result.nodes],
        dtype=float,
    )
    desired_position = np.array(
        [node.desired_position_m for node in result.nodes], dtype=float
    )
    achieved_position = np.array(
        [node.tip_position_m for node in result.nodes], dtype=float
    )
    desired_tangent = np.array(
        [node.desired_tangent for node in result.nodes], dtype=float
    )
    achieved_tangent = np.array(
        [node.tip_tangent for node in result.nodes], dtype=float
    )
    np.savez_compressed(
        npz_path,
        s_m=s,
        state_q_L=states,
        desired_position_m=desired_position,
        achieved_position_m=achieved_position,
        desired_tangent=desired_tangent,
        achieved_tangent=achieved_tangent,
        position_error_m=np.array(
            [node.position_error_m for node in result.nodes], dtype=float
        ),
        tangent_error_rad=np.array(
            [node.tangent_error_rad for node in result.nodes], dtype=float
        ),
        jacobian_condition=np.array(
            [node.jacobian_condition for node in result.nodes], dtype=float
        ),
        contact_active=np.array(
            [node.contact_active for node in result.nodes], dtype=bool
        ),
        source_magnet_lumen_distance_m=np.array(
            [node.source_magnet_lumen_distance_m for node in result.nodes],
            dtype=float,
        ),
        source_magnet_lumen_exclusion_radius_m=np.array(
            [
                node.source_magnet_lumen_exclusion_radius_m
                for node in result.nodes
            ],
            dtype=float,
        ),
        source_magnet_lumen_margin_m=np.array(
            [node.source_magnet_lumen_margin_m for node in result.nodes],
            dtype=float,
        ),
        source_magnet_lumen_segment_index=np.array(
            [node.source_magnet_lumen_segment_index for node in result.nodes],
            dtype=int,
        ),
        source_magnet_lumen_constraint_satisfied=np.array(
            [
                node.source_magnet_lumen_constraint_satisfied
                for node in result.nodes
            ],
            dtype=bool,
        ),
        feasible=np.array([node.feasible for node in result.nodes], dtype=bool),
    )

    try:
        _save_inverse_path_plot(result, plot_path)
    except Exception as exc:
        # Numerical results remain useful if a headless plotting dependency is
        # unavailable.  Record the plotting failure in the summary file.
        result.summary["plot_error"] = f"{type(exc).__name__}: {exc}"
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(_json_ready(payload), stream, indent=2)

    files = {
        "csv": str(csv_path),
        "json": str(json_path),
        "npz": str(npz_path),
        "plot": str(plot_path),
    }
    result.output_files.update(files)
    return files


def _save_inverse_path_plot(result: InversePathResult, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = result.nodes
    s = np.array([node.s_m for node in nodes], dtype=float)
    desired = np.array([node.desired_position_m for node in nodes], dtype=float)
    achieved = np.array([node.tip_position_m for node in nodes], dtype=float)
    magnet_position = np.array(
        [node.magnet_pose6[:3] for node in nodes], dtype=float
    )
    q = np.array([node.q_rad for node in nodes], dtype=float)
    insertion = np.array([node.insertion_m for node in nodes], dtype=float)
    position_error = 1.0e3 * np.array(
        [node.position_error_m for node in nodes], dtype=float
    )
    tangent_error = np.degrees(
        np.array([node.tangent_error_rad for node in nodes], dtype=float)
    )
    condition = np.array(
        [node.jacobian_condition for node in nodes], dtype=float
    )
    contact = np.array([node.contact_active for node in nodes], dtype=float)
    exclusion_margin = 1.0e3 * np.array(
        [node.source_magnet_lumen_margin_m for node in nodes], dtype=float
    )
    feasible = np.array([node.feasible for node in nodes], dtype=bool)

    figure = plt.figure(figsize=(15, 16), constrained_layout=True)
    grid = figure.add_gridspec(4, 2)
    axis_path = figure.add_subplot(grid[0, 0], projection="3d")
    axis_path.plot(*desired.T, "k--", linewidth=2.0, label="desired tip path")
    axis_path.plot(*achieved.T, color="tab:blue", linewidth=1.5, label="achieved")
    if np.all(np.isfinite(magnet_position)):
        axis_path.plot(
            *magnet_position.T,
            color="tab:green",
            linewidth=1.2,
            label="source-magnet centre",
        )
    if np.any(~feasible):
        axis_path.scatter(
            *achieved[~feasible].T,
            color="tab:red",
            marker="x",
            s=60,
            label="unresolved",
        )
    axis_path.set_xlabel("x [m]")
    axis_path.set_ylabel("y [m]")
    axis_path.set_zlabel("z [m]")
    axis_path.set_title("Desired and achieved beam-tip path")
    axis_path.legend()

    axis_error = figure.add_subplot(grid[0, 1])
    axis_error.plot(s, position_error, label="position [mm]")
    axis_error.axhline(
        1.0e3 * float(result.configuration["position_tolerance_m"]),
        color="tab:blue",
        linestyle="--",
        alpha=0.7,
    )
    axis_error.set_xlabel("path coordinate s [m]")
    axis_error.set_ylabel("position error [mm]")
    axis_error_t = axis_error.twinx()
    axis_error_t.plot(s, tangent_error, color="tab:orange", label="tangent [deg]")
    axis_error_t.axhline(
        np.degrees(float(result.configuration["tangent_tolerance_rad"])),
        color="tab:orange",
        linestyle="--",
        alpha=0.7,
    )
    axis_error_t.set_ylabel("tangent error [deg]")
    axis_error.set_title("Physical task errors")

    axis_joints = figure.add_subplot(grid[1, :])
    for joint in range(6):
        axis_joints.plot(s, q[:, joint], label=f"q{joint + 1}")
    axis_joints.set_xlabel("path coordinate s [m]")
    axis_joints.set_ylabel("joint angle [rad]")
    axis_joints.set_title("Solved robot joint configuration")
    axis_joints.legend(ncol=3)

    axis_insertion = figure.add_subplot(grid[2, 0])
    axis_insertion.plot(s, 1.0e3 * insertion, color="tab:green")
    axis_insertion.set_xlabel("path coordinate s [m]")
    axis_insertion.set_ylabel("insertion [mm]")
    axis_insertion.set_title("Solved insertion")

    axis_condition = figure.add_subplot(grid[2, 1])
    finite_condition = np.where(np.isfinite(condition), condition, np.nan)
    axis_condition.semilogy(s, finite_condition, color="tab:purple")
    axis_condition.set_xlabel("path coordinate s [m]")
    axis_condition.set_ylabel("effective condition number")
    axis_condition.set_title("Tolerance-scaled task Jacobian conditioning")

    axis_contact = figure.add_subplot(grid[3, 0])
    axis_contact.step(s, contact, where="post", color="tab:red")
    axis_contact.set_ylim(-0.1, 1.1)
    axis_contact.set_yticks([0, 1], labels=["inactive", "active"])
    axis_contact.set_xlabel("path coordinate s [m]")
    axis_contact.set_title("Contact and EPM/head-exclusion diagnostics")
    if np.any(np.isfinite(exclusion_margin)):
        axis_exclusion = axis_contact.twinx()
        axis_exclusion.plot(s, exclusion_margin, color="tab:green")
        axis_exclusion.axhline(
            0.0, color="black", linestyle="--", alpha=0.6
        )
        axis_exclusion.set_ylabel(
            "EPM exclusion margin [mm]", color="tab:green"
        )

    axis_feasible = figure.add_subplot(grid[3, 1])
    axis_feasible.step(s, feasible.astype(float), where="post", color="tab:blue")
    axis_feasible.set_ylim(-0.1, 1.1)
    axis_feasible.set_yticks([0, 1], labels=["unresolved", "feasible"])
    axis_feasible.set_xlabel("path coordinate s [m]")
    axis_feasible.set_title("Node feasibility")

    figure.suptitle("Offline inverse configuration path", fontsize=16)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


class _LinearMockAdapter:
    """Small deterministic model used by ``--self-test`` only."""

    n_out = 6
    model = None

    def start_step(self) -> None:
        return None

    def forward_output(self, state: Any, commit: bool = False) -> Array:
        del commit
        state = _finite_vector(state, 7, "mock state")
        raw_tangent = np.array([1.0, state[2], state[3]], dtype=float)
        tangent = raw_tangent / np.linalg.norm(raw_tangent)
        return np.concatenate(
            (
                np.array([state[0], state[1], state[6]], dtype=float),
                tangent,
            )
        )

    def continuous_output_jacobian(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "mock state")
        J = np.zeros((6, 7), dtype=float)
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[2, 6] = 1.0
        raw_tangent = np.array([1.0, state[2], state[3]], dtype=float)
        raw_J = np.zeros((3, 7), dtype=float)
        raw_J[1, 2] = 1.0
        raw_J[2, 3] = 1.0
        J[3:6, :] = _normalized_tangent_jacobian(raw_tangent, raw_J)
        return J

    def magnet_transform(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "mock state")
        T = np.eye(4)
        T[:3, 3] = [state[0], state[1], state[6]]
        return T

    def magnet_position_jacobian(self, state: Any) -> Array:
        _finite_vector(state, 7, "mock state")
        J = np.zeros((3, 7), dtype=float)
        J[0, 0] = 1.0
        J[1, 1] = 1.0
        J[2, 6] = 1.0
        return J

    def get_last_diag(self) -> dict[str, Any]:
        return {"contact_active": False, "minimum_gap_m": np.inf}

    def validate_chain_rule(
        self,
        state: Any,
        *,
        joint_step_rad: float,
        insertion_step_m: float,
    ) -> dict[str, Any]:
        state = _finite_vector(state, 7, "mock state")
        steps = np.full(7, float(joint_step_rad), dtype=float)
        steps[6] = float(insertion_step_m)
        J_fd = np.empty((6, 7), dtype=float)
        for index, step in enumerate(steps):
            delta = np.zeros(7, dtype=float)
            delta[index] = step
            J_fd[:, index] = (
                self.forward_output(state + delta)
                - self.forward_output(state - delta)
            ) / (2.0 * step)
        analytical = self.continuous_output_jacobian(state)
        difference = analytical - J_fd
        return {
            "relative_frobenius_error": float(
                np.linalg.norm(difference) / max(np.linalg.norm(J_fd), 1.0e-12)
            ),
            "maximum_absolute_element_error": float(np.max(np.abs(difference))),
        }


class _OffsetMagnetMockAdapter(_LinearMockAdapter):
    """Mock with a beam-nullspace joint controlling EPM radial distance."""

    def magnet_transform(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "offset-magnet mock state")
        T = np.eye(4)
        T[:3, 3] = [state[0], state[4], state[6]]
        return T

    def magnet_position_jacobian(self, state: Any) -> Array:
        _finite_vector(state, 7, "offset-magnet mock state")
        J = np.zeros((3, 7), dtype=float)
        J[0, 0] = 1.0
        J[1, 4] = 1.0
        J[2, 6] = 1.0
        return J


def run_self_test() -> None:
    """Run a known-solution inverse-path smoke test without repository models."""
    adapter = _LinearMockAdapter()
    centreline = np.column_stack(
        (
            np.linspace(0.0, 0.02, 21),
            np.zeros(21),
            np.zeros(21),
        )
    )
    config = InverseConfigurationPlannerConfig(
        position_tolerance_m=1.0e-6,
        tangent_tolerance_rad=1.0e-5,
        initial_path_step_m=5.0e-3,
        minimum_path_step_m=1.0e-3,
        maximum_path_step_m=5.0e-3,
        maximum_joint_step_rad=(0.01,) * 6,
        maximum_insertion_step_m=0.01,
        continuity_weight=(1.0e-8,) * 7,
        joint_centre_weight=(0.0,) * 7,
        maximum_multistart_attempts=2,
        require_contact_model=False,
        debug=False,
    )
    with tempfile.TemporaryDirectory(prefix="offline_inverse_self_test_") as directory:
        result = solve_offline_inverse_configuration(
            adapter=adapter,
            initial_state=np.zeros(7),
            state_min=np.array([-1.0] * 6 + [0.0]),
            state_max=np.array([1.0] * 6 + [0.1]),
            lumen_C=centreline,
            config=config,
            output_dir=directory,
        )
    if not result.all_nodes_feasible:
        raise AssertionError(f"Mock inverse path failed: {result.summary}")
    final_state = np.concatenate(
        (result.nodes[-1].q_rad, np.array([result.nodes[-1].insertion_m]))
    )
    if not np.isclose(final_state[0], 0.02, atol=2.0e-6):
        raise AssertionError(
            f"Known mock solution q1=0.02 was not recovered: {final_state[0]}"
        )

    # _jacobian_predicted_state: on this linear mock the damped predictor must
    # hit the requested tip position to O(damping) and stay inside the bounds.
    predictor_lower = np.array([-1.0] * 6 + [0.0])
    predictor_upper = np.array([1.0] * 6 + [0.1])
    current = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005])
    current_output = adapter.forward_output(current)
    target_position = np.array([0.013, 0.002, 0.006])
    predicted = _jacobian_predicted_state(
        adapter=adapter,
        current_state=current,
        current_position=current_output[:3],
        current_tangent=current_output[3:6],
        desired_position=target_position,
        desired_tangent=current_output[3:6],
        lower=predictor_lower,
        upper=predictor_upper,
        config=replace(
            config,
            maximum_joint_step_rad=(0.1,) * 6,
            maximum_insertion_step_m=0.1,
            jacobian_predictor_damping=1.0e-4,
        ),
    )
    if predicted is None:
        raise AssertionError("_jacobian_predicted_state returned None on the linear mock.")
    if np.any(predicted < predictor_lower - 1e-12) or np.any(predicted > predictor_upper + 1e-12):
        raise AssertionError(f"_jacobian_predicted_state left the bounds: {predicted}")
    achieved = adapter.forward_output(predicted)[:3]
    if np.linalg.norm(achieved - target_position) > 1.0e-3:
        raise AssertionError(
            "_jacobian_predicted_state did not move the tip toward the target: "
            f"achieved={achieved} target={target_position}"
        )
    # A rank-deficient position Jacobian must not blow the predictor up: force
    # a zero column and confirm the step stays bounded and finite.
    class _RankDeficientAdapter(_LinearMockAdapter):
        def continuous_output_jacobian(self, state: Any) -> Array:
            J = super().continuous_output_jacobian(state)
            J[:, 6] = 0.0  # insertion no longer reaches the tip
            return J

    degenerate = _jacobian_predicted_state(
        adapter=_RankDeficientAdapter(),
        current_state=current,
        current_position=current_output[:3],
        current_tangent=current_output[3:6],
        desired_position=target_position + np.array([0.0, 0.0, 0.05]),
        desired_tangent=current_output[3:6],
        lower=predictor_lower,
        upper=predictor_upper,
        config=config,
    )
    if degenerate is None or not np.all(np.isfinite(degenerate)):
        raise AssertionError("_jacobian_predicted_state failed on a rank-deficient Jacobian.")
    if abs(degenerate[6] - current[6]) > config.maximum_insertion_step_m + 1e-12:
        raise AssertionError(
            "_jacobian_predicted_state exceeded the insertion step cap in the "
            f"uncontrollable direction: {degenerate[6]} vs {current[6]}"
        )

    constrained_config = InverseConfigurationPlannerConfig(
        position_tolerance_m=1.0e-6,
        tangent_tolerance_rad=1.0e-5,
        initial_path_step_m=5.0e-3,
        minimum_path_step_m=1.0e-3,
        maximum_path_step_m=5.0e-3,
        maximum_joint_step_rad=(0.1,) * 6,
        maximum_insertion_step_m=0.01,
        continuity_weight=(1.0e-8,) * 7,
        joint_centre_weight=(0.0,) * 7,
        maximum_multistart_attempts=2,
        require_contact_model=False,
        solve_initial_node=True,
        source_magnet_lumen_exclusion_radius_m=5.0e-3,
        require_analytical_magnet_position_jacobian=True,
        debug=False,
    )
    constrained_initial = np.zeros(7, dtype=float)
    # Begin inside the 5 mm exclusion head.  The constrained solve must move
    # this beam-nullspace joint outward without disturbing beam-tip tracking.
    constrained_initial[4] = 1.0e-3
    constrained_result = solve_offline_inverse_configuration(
        adapter=_OffsetMagnetMockAdapter(),
        initial_state=constrained_initial,
        state_min=np.array([-1.0] * 6 + [0.0]),
        state_max=np.array([1.0] * 6 + [0.1]),
        lumen_C=centreline,
        config=constrained_config,
    )
    if not constrained_result.all_nodes_feasible:
        raise AssertionError(
            "Hard EPM/lumen exclusion self-test failed: "
            f"{constrained_result.summary}"
        )
    minimum_distance = min(
        node.source_magnet_lumen_distance_m
        for node in constrained_result.nodes
    )
    if minimum_distance < 5.0e-3 - 1.0e-6:
        raise AssertionError(
            "Hard EPM/lumen exclusion region was entered: "
            f"minimum distance={minimum_distance:.6e} m."
        )
    print(
        "[SELF TEST] PASS: known linear inverse path and hard EPM/lumen "
        f"exclusion recovered ({len(result.nodes)} unconstrained nodes, "
        f"{len(constrained_result.nodes)} constrained nodes)."
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline magnetic-beam inverse-configuration planner. The normal "
            "repository integration calls solve_from_controller_pack() after "
            "build_controller(); this CLI only supplies a dependency-free "
            "self-test."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the deterministic known-solution mock-model test",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test()
        return 0
    parser.print_help()
    print(
        "\nIntegrate this module after your existing build_controller() call; "
        "see solve_from_controller_pack() in the module docstring."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "IMPLEMENTATION_TAG",
    "build_and_solve_from_experiment_config",
    "CentrelinePath",
    "InverseConfigurationPlannerConfig",
    "InverseNodeResult",
    "InversePathResult",
    "save_inverse_path_result",
    "solve_from_controller_pack",
    "solve_offline_inverse_configuration",
]
