"""Global constrained configuration-path optimization for magnetic-beam control.

This module consumes the discrete result produced by
``offline_inverse_configuration.py`` and globally optimizes

    chi_j = [q1, q2, q3, q4, q5, q6, insertion]

over lumen arc length.  Beam-tip position and directed-tangent tolerances are
hard nonlinear constraints in the final solve.  Configuration first/second
differences, seed proximity, and joint centering are only secondary objectives.

Two modes are supported:

``refine_complete``f
    Smooth a complete feasible sequential inverse path.

``recover_partial``
    Use the feasible prefix of a sequential result as a seed, initialize the
    remaining centreline, run a temporary slack-based feasibility restoration,
    and then solve the original slack-free hard-constrained problem globally.

The module ends with a geometric configuration path parameterized by arc
length.  It intentionally does not assign time, run MPC, or command hardware.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
from scipy.sparse import bmat, block_diag, csc_matrix, diags, eye, lil_matrix
from scipy.spatial.transform import Rotation as Rot

try:
    from proper_research.planning.offline_inverse_configuration_head_exclusion import (
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )
except ModuleNotFoundError:  # Allows ``python this_file.py --self-test``.
    from offline_inverse_configuration_head_exclusion import (  # type: ignore
        CentrelinePath,
        InverseNodeResult,
        InversePathResult,
        save_inverse_path_result,
    )

try:
    from proper_research.planning.global_upgrades import (
        NodeExclusionConstraints,
        effective_task_tolerances,
        magnet_path_quadratic,
        tip_centring_quadratic,
        tip_curvature_quadratic,
    )
except ModuleNotFoundError:  # Allows ``python this_file.py --self-test``.
    from global_upgrades import (  # type: ignore
        NodeExclusionConstraints,
        effective_task_tolerances,
        magnet_path_quadratic,
        tip_centring_quadratic,
        tip_curvature_quadratic,
    )


Array = np.ndarray

# The beam forward model rejects insertion L <= 0 outright
# (forward_model._validate_p7).  trust-constr may evaluate the task constraint
# -- which calls that model -- at trial points a hair outside the box (finite
# differences, an early-stopped iterate), so the insertion component is floored
# just above zero everywhere a state reaches the beam model, and kept strictly
# feasible in the decision bounds.
_MIN_INSERTION_M = 1.0e-6


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


def _state_from_inverse_node(node: InverseNodeResult) -> Array:
    return np.concatenate(
        (_finite_vector(node.q_rad, 6, "inverse-node joints"), [float(node.insertion_m)])
    )


def _json_ready(value: Any) -> Any:
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


def _float_or_nan(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


@dataclass(frozen=True)
class GlobalConfigurationOptimizerConfig:
    """Configuration for global feasibility refinement and smoothing."""

    position_tolerance_m: float
    tangent_tolerance_rad: float
    mode: str = "refine_complete"

    path_step_m: float = 1e-3
    minimum_path_step_m: float = 2.5e-4
    dense_validation_samples_per_interval: int = 5
    maximum_refinement_rounds: int = 4
    maximum_nodes: int = 200

    # All quadratic terms operate on chi/configuration_scale so radians and
    # metres can be weighted coherently.
    configuration_scale: tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0e-2,
    )
    first_difference_weight: tuple[float, ...] = (1.0e-5,) * 7
    second_difference_weight: tuple[float, ...] = (1.0e-8,) * 7
    seed_deviation_weight: tuple[float, ...] = (1.0e-5,) * 7
    untrusted_seed_weight_scale: float = 0.0
    joint_centre_weight: tuple[float, ...] = (
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        1.0e-8,
        0.0,
    )
    quadratic_regularization: float = 1.0e-12

    maximum_adjacent_joint_change_rad: tuple[float, ...] = (0.5,) * 6
    maximum_adjacent_insertion_change_m: float = 3.0e-3
    insertion_non_decreasing: bool = False
    fix_initial_state: bool = True
    require_contact_model: bool = True
    allow_tolerance_override: bool = True

    maximum_iterations: int = 10
    constraint_tolerance: float = 1.0e-6
    optimality_tolerance: float = 1.0e-6
    step_tolerance: float = 1.0e-6
    maximum_multistart_attempts: int = 1
    multistart_joint_perturbation_rad: float = 0.03
    multistart_insertion_perturbation_m: float = 2.5e-4
    random_seed: int = 13

    # A full task-Jacobian evaluation requires one contact-aware beam
    # sensitivity solve per path node.  Bound the nonlinear solve by wall time
    # and by the number of new full-path evaluations that fail to improve the
    # best feasible objective.  ``None``/zero disables the corresponding stop.
    maximum_wall_time_s: float | None = None
    stagnation_function_evaluations: int = 0
    stagnation_relative_improvement: float = 1.0e-6

    feasibility_restoration_enabled: bool = True
    feasibility_restoration_maximum_slack: float = 1.0e3
    feasibility_restoration_slack_weight: float = 1.0e4
    feasibility_restoration_iterations: int = 200
    use_failed_nodes_as_warm_start: bool = True
    partial_extrapolation_gain: float = 0.5

    # Refinement/debugging controls.  The scientific/default path keeps dense
    # validation and Jacobian diagnostics enabled.  A diagnostic driver may
    # disable them temporarily to isolate the nonlinear optimizer itself.
    dense_validation_enabled: bool = True
    compute_node_jacobian_diagnostics: bool = True
    coarsen_complete_seed: bool = False
    keep_feasible_complete_seed: bool = True
    preserve_feasible_seed_on_failure: bool = True

    # When True the retained-seed fallback arms whenever the inverse seed
    # satisfies the *physical* task tolerance (position_tolerance_m /
    # tangent_tolerance_rad) and the other hard constraints -- even if the
    # tighter spent tolerance the optimiser is driving toward would reject it.
    # This is what makes "if the global solve does not beat the seed, return
    # the seed" actually hold: without it, any tolerance_spend_fraction < 1
    # (or a partial/degraded inverse prefix) disarms the safety net and a
    # failed 2-hour solve returns its infeasible last iterate instead of the
    # inverse path.  Set False to require the seed to meet the spent tolerance.
    preserve_feasible_seed_against_physical_tolerance: bool = True

    debug: bool = True
    trust_constr_verbose: int = 0

    # ---------------------------------------------------------------- safety
    # The source-magnet / lumen keep-out that layer 1 enforces.  Leaving this
    # at None reproduces the previous behaviour exactly: no exclusion
    # constraint.  Set it to the same radius the inverse planner used and the
    # smoother can no longer walk the magnet into the head.
    source_magnet_lumen_exclusion_radius_m: float | None = None
    source_magnet_lumen_constraint_tolerance_m: float = 1.0e-6
    require_analytical_magnet_position_jacobian: bool = False
    magnet_jacobian_joint_step_rad: float = 1.0e-6
    magnet_jacobian_insertion_step_m: float = 1.0e-6

    # Every hard constraint here (the keep-out, and the position/tangent task
    # tolerances) is enforced by the optimizer at the discrete decision NODES
    # only. Dense validation then re-checks all three at extra samples
    # strictly BETWEEN nodes; the smooth interpolation between two node states
    # that each individually clear a constraint can still cross back over it
    # by a hair -- a collocation gap, not a real path failure. A round that
    # fails dense validation for no reason other than one of these gaps being
    # within the matching acceptance slack below is still accepted
    # (round_feasible=True), instead of triggering the all-or-nothing
    # inverse-path fallback over a near-miss. 0.0 (the default for all three)
    # reproduces the previous strict behaviour exactly. None of these ever
    # loosen what the optimizer itself is asked to satisfy at the nodes, and
    # never change the reported per-sample margins/diagnostics -- only
    # whether a round this close counts as a usable result. See
    # _dense_sample_excusable and the "[GLOBAL DENSE GAP]" debug print, which
    # reports exactly which constraint(s) a rejected round missed by and how
    # much, every round -- read that before reaching for these.
    dense_magnet_exclusion_acceptance_slack_m: float = 0.0
    dense_position_acceptance_slack_m: float = 0.0
    dense_tangent_acceptance_slack_rad: float = 0.0

    # ------------------------------------------------------------ task space
    # Fraction of the physical task tolerance the smoother is allowed to
    # spend.  1.0 is the previous behaviour.  Below 1.0 the constraint the
    # optimiser sees is tighter than the tolerance it is reported against, so
    # the remainder survives as margin for the controller instead of being
    # consumed by joint smoothing.
    tolerance_spend_fraction: float = 1.0

    # Optional quadratic pull toward the centre of the tolerance ball,
    # linearised at the seed.  0.0 keeps the objective exactly as it was.
    tip_centring_weight: float = 0.0

    # Optional quadratic penalty on the curvature (second difference) of the
    # *achieved* tip position, linearised at the seed.  Unlike
    # tip_centring_weight (which pulls each node toward its own static
    # target) this couples adjacent nodes in tip space directly, which is
    # what first/second_difference_weight already do for chi but nothing did
    # for p(chi).  0.0 keeps the objective exactly as it was.
    tip_curvature_weight: float = 0.0

    # Optional quadratic penalty on the SOURCE-MAGNET path -- its arc length
    # (magnet_path_weight, a first-difference term) and/or its curvature
    # (magnet_curvature_weight, a second-difference term), linearised at the
    # seed via the analytical magnet-position Jacobian.  Nothing else in the
    # objective touches the magnet: it only appears as the hard keep-out
    # inequality.  So without these a solution that swings the external magnet
    # far out and sharply back is free as long as chi is smooth.  Set
    # magnet_path_weight > 0 to make the optimiser prefer a short, direct
    # magnet path.  0.0 keeps the objective exactly as it was.
    magnet_path_weight: float = 0.0
    magnet_curvature_weight: float = 0.0

    def validate(self) -> None:
        if self.mode not in {"refine_complete", "recover_partial"}:
            raise ValueError("mode must be 'refine_complete' or 'recover_partial'.")
        positive = {
            "position_tolerance_m": self.position_tolerance_m,
            "tangent_tolerance_rad": self.tangent_tolerance_rad,
            "path_step_m": self.path_step_m,
            "minimum_path_step_m": self.minimum_path_step_m,
            "maximum_adjacent_insertion_change_m": (
                self.maximum_adjacent_insertion_change_m
            ),
            "constraint_tolerance": self.constraint_tolerance,
            "optimality_tolerance": self.optimality_tolerance,
            "step_tolerance": self.step_tolerance,
            "feasibility_restoration_maximum_slack": (
                self.feasibility_restoration_maximum_slack
            ),
            "feasibility_restoration_slack_weight": (
                self.feasibility_restoration_slack_weight
            ),
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < self.tangent_tolerance_rad < math.pi:
            raise ValueError("tangent_tolerance_rad must lie in (0, pi).")
        if self.minimum_path_step_m > self.path_step_m:
            raise ValueError(f"minimum_path_step_m cannot exceed path_step_m. {self.path_step_m}")
        if self.dense_validation_samples_per_interval < 1:
            raise ValueError("dense_validation_samples_per_interval must be >= 1.")
        if self.maximum_refinement_rounds < 0:
            raise ValueError("maximum_refinement_rounds must be >= 0.")
        if self.maximum_nodes < 2:
            raise ValueError("maximum_nodes must be >= 2.")
        if self.maximum_iterations < 1 or self.feasibility_restoration_iterations < 1:
            raise ValueError("Optimizer iteration limits must be >= 1.")
        if self.maximum_multistart_attempts < 1:
            raise ValueError("maximum_multistart_attempts must be >= 1.")
        if self.stagnation_function_evaluations < 0:
            raise ValueError("stagnation_function_evaluations must be >= 0.")
        wall_time = self.maximum_wall_time_s
        if wall_time is not None and (
            not np.isfinite(wall_time) or float(wall_time) <= 0.0
        ):
            raise ValueError("maximum_wall_time_s must be positive when set.")
        improvement = float(self.stagnation_relative_improvement)
        if not np.isfinite(improvement) or improvement < 0.0:
            raise ValueError(
                "stagnation_relative_improvement must be finite and non-negative."
            )
        if self.trust_constr_verbose not in {0, 1, 2, 3}:
            raise ValueError("trust_constr_verbose must be 0, 1, 2 or 3.")

        for name in (
            "configuration_scale",
            "first_difference_weight",
            "second_difference_weight",
            "seed_deviation_weight",
            "joint_centre_weight",
        ):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size != 7 or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must contain seven finite values.")
            if name == "configuration_scale":
                if np.any(values <= 0.0):
                    raise ValueError("configuration_scale entries must be positive.")
            elif np.any(values < 0.0):
                raise ValueError(f"{name} entries must be non-negative.")

        joint_step = np.asarray(
            self.maximum_adjacent_joint_change_rad, dtype=float
        ).reshape(-1)
        if joint_step.size != 6 or np.any(~np.isfinite(joint_step)):
            raise ValueError(
                "maximum_adjacent_joint_change_rad must contain six finite values."
            )
        if np.any(joint_step <= 0.0):
            raise ValueError(
                "maximum_adjacent_joint_change_rad entries must be positive."
            )
        nonnegative = {
            "quadratic_regularization": self.quadratic_regularization,
            "multistart_joint_perturbation_rad": (
                self.multistart_joint_perturbation_rad
            ),
            "multistart_insertion_perturbation_m": (
                self.multistart_insertion_perturbation_m
            ),
            "partial_extrapolation_gain": self.partial_extrapolation_gain,
            "untrusted_seed_weight_scale": self.untrusted_seed_weight_scale,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        radius = self.source_magnet_lumen_exclusion_radius_m
        if radius is not None and (not np.isfinite(radius) or float(radius) <= 0.0):
            raise ValueError(
                "source_magnet_lumen_exclusion_radius_m must be positive when set."
            )
        for name in (
            "source_magnet_lumen_constraint_tolerance_m",
            "magnet_jacobian_joint_step_rad",
            "magnet_jacobian_insertion_step_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        slack = float(self.dense_magnet_exclusion_acceptance_slack_m)
        if not np.isfinite(slack) or slack < 0.0:
            raise ValueError(
                "dense_magnet_exclusion_acceptance_slack_m must be finite and non-negative."
            )
        if radius is not None and slack >= 0.1 * float(radius):
            raise ValueError(
                "dense_magnet_exclusion_acceptance_slack_m is >= 10% of the "
                "exclusion radius -- that is no longer a near-miss tolerance, "
                "it is disabling the constraint. Use a smaller slack."
            )
        position_slack = float(self.dense_position_acceptance_slack_m)
        if not np.isfinite(position_slack) or position_slack < 0.0:
            raise ValueError(
                "dense_position_acceptance_slack_m must be finite and non-negative."
            )
        if position_slack >= 0.1 * float(self.position_tolerance_m):
            raise ValueError(
                "dense_position_acceptance_slack_m is >= 10% of "
                "position_tolerance_m -- that is no longer a near-miss "
                "tolerance, it is loosening the tolerance itself. Use a "
                "smaller slack, or lower --tolerance-spend-fraction instead."
            )
        tangent_slack = float(self.dense_tangent_acceptance_slack_rad)
        if not np.isfinite(tangent_slack) or tangent_slack < 0.0:
            raise ValueError(
                "dense_tangent_acceptance_slack_rad must be finite and non-negative."
            )
        if tangent_slack >= 0.1 * float(self.tangent_tolerance_rad):
            raise ValueError(
                "dense_tangent_acceptance_slack_rad is >= 10% of "
                "tangent_tolerance_rad -- use a smaller slack, or lower "
                "--tolerance-spend-fraction instead."
            )
        fraction = float(self.tolerance_spend_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("tolerance_spend_fraction must lie in (0, 1].")
        centring = float(self.tip_centring_weight)
        if not np.isfinite(centring) or centring < 0.0:
            raise ValueError("tip_centring_weight must be finite and non-negative.")
        curvature = float(self.tip_curvature_weight)
        if not np.isfinite(curvature) or curvature < 0.0:
            raise ValueError("tip_curvature_weight must be finite and non-negative.")
        for name in ("magnet_path_weight", "magnet_curvature_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass
class DenseValidationSample:
    interval_index: int
    fraction: float
    s_m: float
    state: Array
    desired_position_m: Array
    achieved_position_m: Array
    desired_tangent: Array
    achieved_tangent: Array
    position_error_m: float
    tangent_error_rad: float
    position_margin_m: float
    tangent_margin_rad: float
    feasible: bool
    source_magnet_lumen_distance_m: float = np.nan
    source_magnet_lumen_margin_m: float = np.nan
    source_magnet_lumen_constraint_satisfied: bool = True


@dataclass
class GlobalConfigurationNodeResult:
    node_index: int
    s_m: float
    q_rad: Array
    insertion_m: float
    magnet_pose6: Array
    desired_position_m: Array
    achieved_position_m: Array
    desired_tangent: Array
    achieved_tangent: Array
    position_error_m: float
    tangent_error_rad: float
    feasible: bool
    position_constraint_margin_m: float
    tangent_constraint_margin_rad: float
    minimum_joint_margin_rad: float
    minimum_state_margin: float
    jacobian_condition: float
    jacobian_effective_rank: int
    contact_active: bool
    contact_diagnostic_available: bool
    minimum_gap_m: float
    contact_energy: float
    contact_force_norm: float
    seed_state: Array
    state_change_from_seed: Array


@dataclass
class GlobalConfigurationPathResult:
    nodes: list[GlobalConfigurationNodeResult]
    dense_samples: list[DenseValidationSample]
    globally_feasible: bool
    dense_validation_feasible: bool
    optimizer_success: bool
    termination_reason: str
    iterations: int
    objective_value: float
    maximum_position_error_m: float
    maximum_tangent_error_rad: float
    maximum_first_difference: Array
    maximum_second_difference: Array
    restoration_slack_maximum: float
    refinement_rounds: int
    configuration: dict[str, Any]
    summary: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)


@dataclass
class _SeedPath:
    s: Array
    states: Array
    desired_position: Array
    desired_tangent: Array
    trusted_seed_mask: Array
    initial_state: Array
    last_feasible_progress_m: float


@dataclass
class _SolveOutcome:
    x: Array
    success: bool
    status: int
    message: str
    iterations: int
    objective: float
    constraint_violation: float
    restoration_slack_maximum: float


def _contact_model_flags(adapter: Any) -> dict[str, bool | None]:
    model = getattr(adapter, "model", None)
    contact_cfg = getattr(model, "contact_cfg", None)
    if contact_cfg is None:
        return {"enabled": None, "use_in_jacobian": None}
    return {
        "enabled": bool(getattr(contact_cfg, "enabled", False)),
        "use_in_jacobian": bool(getattr(contact_cfg, "use_in_jacobian", False)),
    }


def _underlying_last_info(adapter: Any) -> dict[str, Any] | None:
    """Find contact/energy diagnostics without modifying the existing adapter."""
    beam_output_fn = getattr(adapter, "beam_output_fn", None)
    candidates = (
        adapter,
        beam_output_fn,
        getattr(beam_output_fn, "forward_adapter", None),
        getattr(adapter, "model", None),
    )
    for source in candidates:
        if source is None:
            continue
        value = getattr(source, "last_info", None)
        if isinstance(value, dict):
            return value
    return None


def _contact_diagnostics(adapter: Any) -> dict[str, Any]:
    info = _underlying_last_info(adapter)
    if not isinstance(info, dict):
        return {
            "available": False,
            "active": False,
            "minimum_gap_m": np.nan,
            "contact_energy": np.nan,
            "contact_force_norm": np.nan,
        }
    parts = info.get("parts", {}) or {}
    if not isinstance(parts, dict):
        parts = {}
    gap = _float_or_nan(parts.get("gap_min", np.nan))
    energy = _float_or_nan(parts.get("W_cf", np.nan))
    forces = parts.get("F_nodes", None)
    force_norm = (
        np.nan
        if forces is None
        else float(np.linalg.norm(np.asarray(forces, dtype=float)))
    )
    available = bool(np.isfinite(gap) or np.isfinite(energy) or np.isfinite(force_norm))
    active = bool(
        (np.isfinite(gap) and gap < 0.0)
        or (np.isfinite(energy) and energy > 1.0e-16)
        or (np.isfinite(force_norm) and force_norm > 1.0e-12)
    )
    return {
        "available": available,
        "active": active,
        "minimum_gap_m": gap,
        "contact_energy": energy,
        "contact_force_norm": force_norm,
    }


def _magnet_pose6(adapter: Any, state: Array) -> Array:
    transform_fn = getattr(adapter, "magnet_transform", None)
    if not callable(transform_fn):
        return np.full(6, np.nan, dtype=float)
    T = np.asarray(transform_fn(state), dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)):
        return np.full(6, np.nan, dtype=float)
    return np.concatenate((T[:3, 3], Rot.from_matrix(T[:3, :3]).as_rotvec()))


_MAGNET_JACOBIAN_METHOD_NAMES = (
    "magnet_position_jacobian",
    "continuous_magnet_position_jacobian",
    "source_magnet_position_jacobian",
)


def _magnet_position_jacobian(
    adapter: Any, state: Array, *, joint_step: float, insertion_step: float
) -> Array:
    """3x7 d(magnet position)/d(state), analytical if the adapter exposes it.

    Falls back to bound-safe central differences of ``magnet_transform`` (cheap
    forward kinematics -- no beam solve).  The insertion column is typically ~0:
    the external magnet rides the robot flange, insertion advances the beam.
    """
    state = _finite_vector(state, 7, "magnet-Jacobian state")
    for name in _MAGNET_JACOBIAN_METHOD_NAMES:
        method = getattr(adapter, name, None)
        if callable(method):
            matrix = np.asarray(method(state), dtype=float)
            if matrix.shape == (3, 6):
                matrix = np.column_stack((matrix, np.zeros(3, dtype=float)))
            if matrix.shape == (3, 7) and np.all(np.isfinite(matrix)):
                return np.ascontiguousarray(matrix, dtype=float)
    steps = np.full(7, float(joint_step), dtype=float)
    steps[6] = float(insertion_step)
    matrix = np.zeros((3, 7), dtype=float)
    centre = _magnet_pose6(adapter, state)[:3]
    for axis, step in enumerate(steps):
        plus = min(step, max(0.0, step))
        forward = state.copy()
        backward = state.copy()
        forward[axis] += step
        backward[axis] -= step
        matrix[:, axis] = (
            _magnet_pose6(adapter, forward)[:3]
            - _magnet_pose6(adapter, backward)[:3]
        ) / (2.0 * step)
        if not np.all(np.isfinite(matrix[:, axis])):
            matrix[:, axis] = (
                _magnet_pose6(adapter, forward)[:3] - centre
            ) / step
    return matrix


def _magnet_path_terms(
    *,
    adapter: Any,
    states: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> tuple[Array, Array]:
    """Per-node magnet position (N,3) and position Jacobian (N,3,7)."""
    states = np.asarray(states, dtype=float).reshape(-1, 7)
    positions = np.empty((states.shape[0], 3), dtype=float)
    jacobians = np.empty((states.shape[0], 3, 7), dtype=float)
    for index, state in enumerate(states):
        positions[index] = _magnet_pose6(adapter, state)[:3]
        jacobians[index] = _magnet_position_jacobian(
            adapter,
            state,
            joint_step=float(config.magnet_jacobian_joint_step_rad),
            insertion_step=float(config.magnet_jacobian_insertion_step_m),
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(jacobians)):
        raise FloatingPointError("Magnet forward kinematics returned non-finite values.")
    return positions, jacobians


def _normalized_tangent_jacobian(tangent_raw: Array, J_raw: Array) -> Array:
    tangent_raw = _finite_vector(tangent_raw, 3, "beam tangent")
    J_raw = np.asarray(J_raw, dtype=float).reshape(3, 7)
    norm = float(np.linalg.norm(tangent_raw))
    if norm <= 1.0e-12:
        raise FloatingPointError("Cannot normalize a near-zero beam tangent.")
    tangent = tangent_raw / norm
    return ((np.eye(3) - np.outer(tangent, tangent)) / norm) @ J_raw


def _effective_condition(matrix: Array) -> tuple[float, int]:
    matrix = np.asarray(matrix, dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return np.inf, 0
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    active = singular_values[singular_values > tolerance]
    if active.size == 0:
        return np.inf, 0
    return float(active[0] / active[-1]), int(active.size)


def _branch_forward_adapter(adapter: Any) -> Any | None:
    """Return the cache-owning pose adapter behind a joint-space adapter."""
    beam_output_fn = getattr(adapter, "beam_output_fn", None)
    forward_adapter = getattr(beam_output_fn, "forward_adapter", None)
    required = (
        "capture_cache_state",
        "restore_cache_state",
        "get_baseline_cache_copy",
        "set_baseline_cache",
        "reset_to_initial_baseline",
    )
    if forward_adapter is None or any(
        not callable(getattr(forward_adapter, name, None)) for name in required
    ):
        return None
    return forward_adapter


def _replay_node_baselines(
    adapter: Any,
    states: Array,
) -> tuple[list[Any] | None, Array | None]:
    """Replay a path from the construction equilibrium and save each baseline.

    Baseline ``i`` is the committed equilibrium immediately before state ``i``.
    Replaying is isolated: the caller's live model and frozen baseline are
    restored before this function returns.
    """
    states = np.asarray(states, dtype=float).reshape(-1, 7).copy()
    states[:, 6] = np.maximum(states[:, 6], _MIN_INSERTION_M)
    forward_adapter = _branch_forward_adapter(adapter)
    if forward_adapter is None:
        return None, None

    snapshot = forward_adapter.capture_cache_state()
    baselines: list[Any] = []
    outputs: list[Array] = []
    try:
        forward_adapter.reset_to_initial_baseline()
        for state in states:
            baselines.append(forward_adapter.get_baseline_cache_copy())
            output = np.asarray(
                adapter.forward_output(state, commit=True), dtype=float
            ).reshape(-1)[:6]
            if output.size != 6 or not np.all(np.isfinite(output)):
                raise FloatingPointError(
                    "Equilibrium replay produced an invalid six-output value."
                )
            outputs.append(output.copy())
    finally:
        forward_adapter.restore_cache_state(snapshot)
    return baselines, np.asarray(outputs, dtype=float)


class _BatchBeamEvaluator:
    """Serial, per-node cache for the stateful beam model and its Jacobian."""

    def __init__(
        self,
        adapter: Any,
        node_count: int,
        node_baselines: Sequence[Any] | None = None,
    ):
        self.adapter = adapter
        self.node_count = int(node_count)
        self._forward_adapter = _branch_forward_adapter(adapter)
        if node_baselines is None:
            self._node_baselines = None
        else:
            if len(node_baselines) != self.node_count:
                raise ValueError(
                    "node_baselines must contain one cache per evaluator node."
                )
            if self._forward_adapter is None:
                raise RuntimeError(
                    "Node baselines were supplied but the beam-output adapter "
                    "does not expose cache snapshot methods."
                )
            self._node_baselines = list(node_baselines)
        self._state_output: list[Array | None] = [None] * self.node_count
        self._output: list[Array | None] = [None] * self.node_count
        self._state_jacobian: list[Array | None] = [None] * self.node_count
        self._jacobian: list[Array | None] = [None] * self.node_count
        self.output_evaluations = 0
        self.jacobian_evaluations = 0

    def _at_node_baseline(self, index: int, callback):
        if self._node_baselines is None:
            return callback()
        assert self._forward_adapter is not None
        snapshot = self._forward_adapter.capture_cache_state()
        try:
            self._forward_adapter.set_baseline_cache(
                self._node_baselines[index],
                synchronize_model=True,
            )
            return callback()
        finally:
            self._forward_adapter.restore_cache_state(snapshot)

    @staticmethod
    def _same(left: Array | None, right: Array) -> bool:
        return left is not None and np.array_equal(left, right)

    def output(self, index: int, state: Array) -> Array:
        state = _finite_vector(state, 7, f"state at node {index}")
        if state[6] < _MIN_INSERTION_M:
            # The beam model rejects L <= 0.  keep_feasible on the bounds should
            # stop trust-constr trialing here, but a finite-difference probe or
            # a not-quite-converged iterate can still land a hair below.
            state = state.copy()
            state[6] = _MIN_INSERTION_M
        if not self._same(self._state_output[index], state):
            value = np.asarray(
                self._at_node_baseline(
                    index,
                    lambda: self.adapter.forward_output(state, commit=False),
                ),
                dtype=float,
            ).reshape(-1)
            if value.size < 6 or not np.all(np.isfinite(value[:6])):
                raise FloatingPointError(
                    f"Invalid six-output beam evaluation at node {index}."
                )
            self._state_output[index] = state.copy()
            self._output[index] = value[:6].copy()
            self.output_evaluations += 1
        return np.asarray(self._output[index], dtype=float).copy()

    def jacobian(self, index: int, state: Array) -> Array:
        state = _finite_vector(state, 7, f"Jacobian state at node {index}")
        if state[6] < _MIN_INSERTION_M:
            state = state.copy()
            state[6] = _MIN_INSERTION_M
        if not self._same(self._state_jacobian[index], state):
            value = np.asarray(
                self._at_node_baseline(
                    index,
                    lambda: self.adapter.continuous_output_jacobian(state),
                ),
                dtype=float,
            )
            if value.shape != (6, 7) or not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"Invalid 6x7 chained Jacobian at node {index}: {value.shape}."
                )
            self._state_jacobian[index] = state.copy()
            self._jacobian[index] = value.copy()
            self.jacobian_evaluations += 1
        return np.asarray(self._jacobian[index], dtype=float).copy()


class _TaskConstraints:
    """Normalized hard position/tangent constraints and sparse Jacobian."""

    def __init__(
        self,
        *,
        evaluator: _BatchBeamEvaluator,
        desired_position: Array,
        desired_tangent: Array,
        position_tolerance_m: float,
        tangent_tolerance_rad: float,
    ):
        self.evaluator = evaluator
        self.desired_position = np.asarray(desired_position, dtype=float).reshape(-1, 3)
        tangent = np.asarray(desired_tangent, dtype=float).reshape(-1, 3)
        self.desired_tangent = np.array(
            [_unit_vector(row, "desired tangent") for row in tangent], dtype=float
        )
        self.node_count = self.desired_position.shape[0]
        if self.desired_tangent.shape[0] != self.node_count:
            raise ValueError("Desired position and tangent node counts differ.")
        self.position_tolerance_m = float(position_tolerance_m)
        self.tangent_tolerance_rad = float(tangent_tolerance_rad)
        self.tangent_denominator = max(
            1.0 - math.cos(self.tangent_tolerance_rad), 1.0e-12
        )

    def values(self, flat_state: Array) -> Array:
        states = np.asarray(flat_state, dtype=float).reshape(self.node_count, 7)
        position_values = np.empty(self.node_count, dtype=float)
        tangent_values = np.empty(self.node_count, dtype=float)
        for index, state in enumerate(states):
            output = self.evaluator.output(index, state)
            error = output[:3] - self.desired_position[index]
            position_values[index] = 1.0 - (
                float(error @ error) / self.position_tolerance_m**2
            )
            tangent = _unit_vector(output[3:6], "beam tangent")
            tangent_values[index] = (
                float(tangent @ self.desired_tangent[index])
                - math.cos(self.tangent_tolerance_rad)
            ) / self.tangent_denominator
        return np.concatenate((position_values, tangent_values))

    def jacobian(self, flat_state: Array) -> csc_matrix:
        states = np.asarray(flat_state, dtype=float).reshape(self.node_count, 7)
        matrix = lil_matrix((2 * self.node_count, 7 * self.node_count), dtype=float)
        for index, state in enumerate(states):
            output = self.evaluator.output(index, state)
            J = self.evaluator.jacobian(index, state)
            error = output[:3] - self.desired_position[index]
            position_row = -2.0 * (error @ J[:3, :]) / self.position_tolerance_m**2
            tangent_row = (
                self.desired_tangent[index]
                @ _normalized_tangent_jacobian(output[3:6], J[3:6, :])
            ) / self.tangent_denominator
            columns = slice(7 * index, 7 * (index + 1))
            matrix[index, columns] = position_row
            matrix[self.node_count + index, columns] = tangent_row
        return matrix.tocsc()


def _first_difference_matrix(s: Array, scale: Array) -> csc_matrix:
    s = np.asarray(s, dtype=float).reshape(-1)
    node_count = s.size
    ds = np.diff(s)
    if node_count < 2 or np.any(ds <= 0.0):
        raise ValueError("Path coordinates must be strictly increasing.")
    matrix = lil_matrix((7 * (node_count - 1), 7 * node_count), dtype=float)
    for interval, delta_s in enumerate(ds):
        for coordinate in range(7):
            row = 7 * interval + coordinate
            coefficient = 1.0 / (delta_s * scale[coordinate])
            matrix[row, 7 * interval + coordinate] = -coefficient
            matrix[row, 7 * (interval + 1) + coordinate] = coefficient
    return matrix.tocsc()


def _second_difference_matrix(s: Array, scale: Array) -> csc_matrix:
    s = np.asarray(s, dtype=float).reshape(-1)
    node_count = s.size
    if node_count < 3:
        return csc_matrix((0, 7 * node_count), dtype=float)
    ds = np.diff(s)
    matrix = lil_matrix((7 * (node_count - 2), 7 * node_count), dtype=float)
    for interior in range(1, node_count - 1):
        left = ds[interior - 1]
        right = ds[interior]
        common = 2.0 / (left + right)
        for coordinate in range(7):
            row = 7 * (interior - 1) + coordinate
            inv_scale = 1.0 / scale[coordinate]
            matrix[row, 7 * (interior - 1) + coordinate] = (
                common * inv_scale / left
            )
            matrix[row, 7 * interior + coordinate] = (
                -common * inv_scale * (1.0 / left + 1.0 / right)
            )
            matrix[row, 7 * (interior + 1) + coordinate] = (
                common * inv_scale / right
            )
    return matrix.tocsc()


class _QuadraticPathObjective:
    """Sparse quadratic path objective with exact gradient and Hessian."""

    def __init__(
        self,
        *,
        s: Array,
        seed_states: Array,
        trusted_seed_mask: Array,
        state_min: Array,
        state_max: Array,
        config: GlobalConfigurationOptimizerConfig,
    ):
        self.s = np.asarray(s, dtype=float).reshape(-1)
        self.seed = np.asarray(seed_states, dtype=float).reshape(-1, 7)
        self.node_count = self.s.size
        if self.seed.shape[0] != self.node_count:
            raise ValueError("Seed state count does not match path node count.")
        trusted_seed_mask = np.asarray(trusted_seed_mask, dtype=bool).reshape(-1)
        if trusted_seed_mask.size != self.node_count:
            raise ValueError("trusted_seed_mask must contain one value per node.")
        scale = _finite_vector(config.configuration_scale, 7, "configuration_scale")
        D1 = _first_difference_matrix(self.s, scale)
        D2 = _second_difference_matrix(self.s, scale)
        w1 = np.tile(
            _finite_vector(config.first_difference_weight, 7, "first weights"),
            max(0, self.node_count - 1),
        )
        w2 = np.tile(
            _finite_vector(config.second_difference_weight, 7, "second weights"),
            max(0, self.node_count - 2),
        )
        W1 = diags(w1, format="csc") if w1.size else csc_matrix((0, 0))
        W2 = diags(w2, format="csc") if w2.size else csc_matrix((0, 0))
        H = csc_matrix((7 * self.node_count, 7 * self.node_count), dtype=float)
        if D1.shape[0]:
            H = H + D1.T @ W1 @ D1
        if D2.shape[0]:
            H = H + D2.T @ W2 @ D2

        seed_weight = _finite_vector(
            config.seed_deviation_weight, 7, "seed_deviation_weight"
        ) / scale**2
        node_seed_scale = np.where(
            trusted_seed_mask,
            1.0,
            float(config.untrusted_seed_weight_scale),
        )
        seed_diagonal = np.concatenate(
            [scale_value * seed_weight for scale_value in node_seed_scale]
        )
        H_seed = diags(seed_diagonal, format="csc")
        seed_flat = self.seed.reshape(-1)
        f = -(seed_diagonal * seed_flat)
        constant = 0.5 * float(seed_flat @ (seed_diagonal * seed_flat))
        H = H + H_seed

        centre_weight = _finite_vector(
            config.joint_centre_weight, 7, "joint_centre_weight"
        ) / scale**2
        centre = 0.5 * (state_min + state_max)
        centre_diagonal = np.tile(centre_weight, self.node_count)
        H_centre = diags(centre_diagonal, format="csc")
        centre_flat = np.tile(centre, self.node_count)
        H = H + H_centre
        f = f - centre_diagonal * centre_flat
        constant += 0.5 * float(centre_flat @ (centre_diagonal * centre_flat))

        if config.quadratic_regularization > 0.0:
            H = H + float(config.quadratic_regularization) * eye(
                H.shape[0], format="csc"
            )
        self.H = csc_matrix(0.5 * (H + H.T))
        self.f = np.asarray(f, dtype=float)
        self.constant = float(constant)
        self.D1 = D1
        self.D2 = D2

    def value(self, flat_state: Array) -> float:
        x = np.asarray(flat_state, dtype=float).reshape(-1)
        return float(0.5 * x @ (self.H @ x) + self.f @ x + self.constant)

    def gradient(self, flat_state: Array) -> Array:
        x = np.asarray(flat_state, dtype=float).reshape(-1)
        return np.asarray(self.H @ x + self.f, dtype=float)

    def hessian(self, _flat_state: Array) -> csc_matrix:
        return self.H

    def derivatives(self, states: Array) -> tuple[Array, Array]:
        flat = np.asarray(states, dtype=float).reshape(-1)
        first = np.asarray(self.D1 @ flat, dtype=float).reshape(-1, 7)
        second = (
            np.asarray(self.D2 @ flat, dtype=float).reshape(-1, 7)
            if self.D2.shape[0]
            else np.empty((0, 7), dtype=float)
        )
        return first, second


def _adjacent_linear_constraint(
    *,
    node_count: int,
    config: GlobalConfigurationOptimizerConfig,
) -> LinearConstraint | None:
    if node_count < 2:
        return None
    matrix = lil_matrix((7 * (node_count - 1), 7 * node_count), dtype=float)
    for interval in range(node_count - 1):
        rows = slice(7 * interval, 7 * (interval + 1))
        matrix[rows, 7 * interval : 7 * (interval + 1)] = -np.eye(7)
        matrix[rows, 7 * (interval + 1) : 7 * (interval + 2)] = np.eye(7)
    maximum = np.concatenate(
        (
            _finite_vector(
                config.maximum_adjacent_joint_change_rad,
                6,
                "maximum adjacent joint change",
            ),
            [float(config.maximum_adjacent_insertion_change_m)],
        )
    )
    upper = np.tile(maximum, node_count - 1)
    lower = -upper.copy()
    if config.insertion_non_decreasing:
        lower[np.arange(6, lower.size, 7)] = 0.0
    return LinearConstraint(matrix.tocsc(), lower, upper)


def _decision_bounds(
    *,
    node_count: int,
    state_min: Array,
    state_max: Array,
    initial_state: Array,
    fix_initial_state: bool,
) -> Bounds:
    lower = np.tile(np.asarray(state_min, dtype=float), node_count)
    upper = np.tile(np.asarray(state_max, dtype=float), node_count)
    insertion_indices = np.arange(6, lower.size, 7)
    lower[insertion_indices] = np.maximum(
        lower[insertion_indices], _MIN_INSERTION_M
    )
    keep_feasible = np.zeros(lower.size, dtype=bool)
    keep_feasible[insertion_indices] = True
    if fix_initial_state:
        lower[:7] = initial_state
        upper[:7] = initial_state
        # An equality-bounded component (lb == ub) has no interior; keeping it
        # "strictly feasible" is meaningless and trips scipy's x0 check.
        keep_feasible[:7] = False
    return Bounds(lower, upper, keep_feasible=keep_feasible)


def _trust_options(config: GlobalConfigurationOptimizerConfig, maxiter: int) -> dict:
    return {
        "maxiter": int(maxiter),
        "gtol": float(config.optimality_tolerance),
        "xtol": float(config.step_tolerance),
        "barrier_tol": float(config.constraint_tolerance),
        "verbose": int(config.trust_constr_verbose),
        "sparse_jacobian": True,
    }


def _exclusion_lower_bound(config: GlobalConfigurationOptimizerConfig) -> float:
    """Dimensionless keep-out lower bound including its physical tolerance.

    ``NodeExclusionConstraints.values`` returns ``distance**2 / radius**2 - 1``.
    Layer 1 certifies distance down to ``radius - tolerance``; using a zero
    lower bound here made that same saved seed microscopically infeasible and
    disabled ``keep_feasible`` at every active keep-out node.
    """
    radius = config.source_magnet_lumen_exclusion_radius_m
    if radius is None:
        return 0.0
    radius = float(radius)
    tolerance = min(
        float(config.source_magnet_lumen_constraint_tolerance_m),
        0.5 * radius,
    )
    return float(((radius - tolerance) / radius) ** 2 - 1.0)


def _exclusion_violation(
    exclusion: NodeExclusionConstraints | None,
    x: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> float:
    if exclusion is None:
        return 0.0
    lower = _exclusion_lower_bound(config)
    values = np.asarray(exclusion.values(x), dtype=float).reshape(-1)
    return float(max(0.0, lower - float(np.min(values))))


class _ProgressMonitor:
    """Retain the best feasible iterate and stop expensive stagnation."""

    def __init__(self, config: GlobalConfigurationOptimizerConfig):
        self.config = config
        self.started = time.perf_counter()
        self.best_feasible_x: Array | None = None
        self.best_feasible_objective = np.inf
        self.best_feasible_iteration = 0
        self.best_feasible_nfev = 0
        self.last_improvement_nfev = 0
        self.stop_reason: str | None = None

    def __call__(self, xk: Array, state: Any) -> bool:
        iteration = int(getattr(state, "nit", 0))
        nfev = int(getattr(state, "nfev", 0))
        objective = float(getattr(state, "fun", np.nan))
        violation = float(getattr(state, "constr_violation", np.inf))
        elapsed = time.perf_counter() - self.started
        feasible = bool(
            np.isfinite(violation)
            and violation <= float(self.config.constraint_tolerance)
        )
        threshold = float(self.config.stagnation_relative_improvement)
        required = threshold * max(1.0, abs(self.best_feasible_objective))
        if feasible and (
            self.best_feasible_x is None
            or objective < self.best_feasible_objective - required
        ):
            self.best_feasible_x = np.asarray(xk, dtype=float).copy()
            self.best_feasible_objective = objective
            self.best_feasible_iteration = iteration
            self.best_feasible_nfev = nfev
            self.last_improvement_nfev = nfev

        if self.config.debug:
            print(
                "[TRUST-CONSTR] "
                f"iteration={iteration} objective={objective:.6e} "
                f"optimality={float(getattr(state, 'optimality', np.nan)):.3e} "
                f"violation={violation:.3e} "
                f"radius={float(getattr(state, 'tr_radius', np.nan)):.3e} "
                f"nfev={nfev} elapsed_s={elapsed:.1f}",
                flush=True,
            )

        wall_time = self.config.maximum_wall_time_s
        if wall_time is not None and elapsed >= float(wall_time):
            self.stop_reason = (
                f"wall_time_limit_reached: {elapsed:.1f}s >= {float(wall_time):.1f}s"
            )
        stagnation = int(self.config.stagnation_function_evaluations)
        if (
            self.stop_reason is None
            and stagnation > 0
            and self.best_feasible_x is not None
            and nfev - self.last_improvement_nfev >= stagnation
        ):
            self.stop_reason = (
                "feasible_objective_stagnation: "
                f"no relative improvement >= {threshold:.3e} in "
                f"{stagnation} full-path evaluations"
            )
        if self.stop_reason is not None and self.config.debug:
            print(f"[TRUST-CONSTR STOP] {self.stop_reason}", flush=True)
        return self.stop_reason is not None


def _run_restoration(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
    exclusion: NodeExclusionConstraints | None = None,
) -> tuple[Array, float, Any]:
    node_count = tasks.node_count
    state_size = 7 * node_count
    slack_size = 2 * node_count
    x0 = np.clip(
        np.asarray(x0, dtype=float),
        np.asarray(bounds.lb, dtype=float),
        np.asarray(bounds.ub, dtype=float),
    )
    initial_task = tasks.values(x0)
    slack0 = np.clip(
        np.maximum(0.0, -initial_task) + 1.0e-8,
        0.0,
        float(config.feasibility_restoration_maximum_slack),
    )
    z0 = np.concatenate((x0, slack0))
    slack_weight = float(config.feasibility_restoration_slack_weight)
    H_slack = 2.0 * slack_weight * eye(slack_size, format="csc")
    H_aug = block_diag((objective.H, H_slack), format="csc")

    def value(z: Array) -> float:
        x = z[:state_size]
        sigma = z[state_size:]
        return objective.value(x) + slack_weight * float(sigma @ sigma)

    def gradient(z: Array) -> Array:
        return np.concatenate(
            (objective.gradient(z[:state_size]), 2.0 * slack_weight * z[state_size:])
        )

    def hessian(_z: Array) -> csc_matrix:
        return H_aug

    def constraint_values(z: Array) -> Array:
        return tasks.values(z[:state_size]) + z[state_size:]

    def constraint_jacobian(z: Array) -> csc_matrix:
        task_jacobian = tasks.jacobian(z[:state_size])
        return bmat([[task_jacobian, eye(slack_size, format="csc")]], format="csc")

    lower = np.concatenate((np.asarray(bounds.lb, float), np.zeros(slack_size)))
    upper = np.concatenate(
        (
            np.asarray(bounds.ub, float),
            np.full(slack_size, float(config.feasibility_restoration_maximum_slack)),
        )
    )
    state_keep_feasible = np.broadcast_to(
        np.asarray(bounds.keep_feasible, dtype=bool), (state_size,)
    )
    keep_feasible = np.concatenate(
        (state_keep_feasible, np.zeros(slack_size, dtype=bool))
    )
    constraints: list[Any] = [
        NonlinearConstraint(
            constraint_values,
            np.zeros(slack_size),
            np.full(slack_size, np.inf),
            jac=constraint_jacobian,
        )
    ]
    if adjacent is not None:
        A_aug = bmat(
            [[csc_matrix(adjacent.A), csc_matrix((adjacent.A.shape[0], slack_size))]],
            format="csc",
        )
        constraints.append(LinearConstraint(A_aug, adjacent.lb, adjacent.ub))
    if exclusion is not None:
        # Restoration is allowed to relax the *task* constraints through their
        # slacks; it is never allowed to relax the keep-out.  Recovering
        # tracking feasibility by moving the magnet into the head would be a
        # worse outcome than failing to recover it.
        def exclusion_values(z: Array) -> Array:
            return exclusion.values(z[:state_size])

        def exclusion_jacobian(z: Array) -> Any:
            return bmat(
                [[
                    csc_matrix(exclusion.jacobian(z[:state_size])),
                    csc_matrix((exclusion.node_count, slack_size)),
                ]],
                format="csc",
            )

        constraints.append(
            NonlinearConstraint(
                exclusion_values,
                np.full(
                    exclusion.node_count,
                    _exclusion_lower_bound(config),
                    dtype=float,
                ),
                np.full(exclusion.node_count, np.inf),
                jac=exclusion_jacobian,
            )
        )
    monitor = _ProgressMonitor(config)
    solved = minimize(
        value,
        z0,
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        bounds=Bounds(lower, upper, keep_feasible=keep_feasible),
        constraints=constraints,
        callback=monitor,
        options=_trust_options(config, config.feasibility_restoration_iterations),
    )
    if monitor.stop_reason is not None:
        solved.message = f"{solved.message}; {monitor.stop_reason}"
    sigma = np.asarray(solved.x[state_size:], dtype=float)
    return np.asarray(solved.x[:state_size], dtype=float), float(np.max(sigma)), solved

def _run_hard_solve(
    *,
    x0: Array,
    objective: _QuadraticPathObjective,
    tasks: _TaskConstraints,
    bounds: Bounds,
    adjacent: LinearConstraint | None,
    config: GlobalConfigurationOptimizerConfig,
    exclusion: NodeExclusionConstraints | None = None,
) -> tuple[Any, _ProgressMonitor]:
    # keep_feasible on the bounds requires a feasible start; a multistart
    # perturbation or an inserted refinement node can push a component (most
    # dangerously the insertion) a hair outside the box.
    x0 = np.clip(
        np.asarray(x0, dtype=float),
        np.asarray(bounds.lb, dtype=float),
        np.asarray(bounds.ub, dtype=float),
    )
    initial_task_values = tasks.values(x0)
    initial_task_feasible = bool(np.all(initial_task_values >= 0.0))
    keep_task_feasible = bool(
        config.keep_feasible_complete_seed
        and config.mode == "refine_complete"
        and initial_task_feasible
    )
    if config.debug:
        print(
            "[GLOBAL SEED] "
            f"minimum_task_constraint={np.min(initial_task_values):.6e} "
            f"node_constraints_feasible={initial_task_feasible} "
            f"keep_feasible={keep_task_feasible}",
            flush=True,
        )
    constraints: list[Any] = [
        NonlinearConstraint(
            tasks.values,
            np.zeros(2 * tasks.node_count),
            np.full(2 * tasks.node_count, np.inf),
            jac=tasks.jacobian,
            keep_feasible=keep_task_feasible,
        )
    ]
    if adjacent is not None:
        constraints.append(adjacent)
    if exclusion is not None:
        # The keep-out is a physical safety constraint, so it is never given
        # slack and never traded against smoothness.  keep_feasible is set
        # only when the seed already satisfies it, which is the normal case
        # for a layer-1 path that enforced the same inequality.
        initial_exclusion = exclusion.values(x0)
        exclusion_lower = _exclusion_lower_bound(config)
        constraints.append(
            NonlinearConstraint(
                exclusion.values,
                np.full(exclusion.node_count, exclusion_lower, dtype=float),
                np.full(exclusion.node_count, np.inf),
                jac=exclusion.jacobian,
                keep_feasible=bool(np.all(initial_exclusion >= exclusion_lower)),
            )
        )
        if config.debug:
            print(
                "[GLOBAL SEED] "
                f"minimum_exclusion_constraint={np.min(initial_exclusion):.6e} "
                f"jacobian_source={exclusion.jacobian_source}",
                flush=True,
            )
    monitor = _ProgressMonitor(config)
    solved = minimize(
        objective.value,
        x0,
        method="trust-constr",
        jac=objective.gradient,
        hess=objective.hessian,
        bounds=bounds,
        constraints=constraints,
        callback=monitor,
        options=_trust_options(config, config.maximum_iterations),
    )
    if monitor.stop_reason is not None:
        solved.message = f"{solved.message}; {monitor.stop_reason}"
    return solved, monitor


def _task_violation(tasks: _TaskConstraints, x: Array) -> float:
    values = tasks.values(x)
    return float(np.max(np.maximum(0.0, -values)))


def _linear_constraint_violation(
    constraint: LinearConstraint | None,
    x: Array,
) -> float:
    if constraint is None:
        return 0.0
    values = np.asarray(constraint.A @ x, dtype=float).reshape(-1)
    lower = np.asarray(constraint.lb, dtype=float).reshape(-1)
    upper = np.asarray(constraint.ub, dtype=float).reshape(-1)
    below = np.where(np.isfinite(lower), lower - values, 0.0)
    above = np.where(np.isfinite(upper), values - upper, 0.0)
    return float(max(0.0, np.max(below), np.max(above)))


def _bound_constraint_violation(bounds: Bounds, x: Array) -> float:
    """Return the maximum violation of the final decision-vector bounds."""
    values = np.asarray(x, dtype=float).reshape(-1)
    lower = np.asarray(bounds.lb, dtype=float).reshape(-1)
    upper = np.asarray(bounds.ub, dtype=float).reshape(-1)
    if values.size != lower.size or values.size != upper.size:
        raise ValueError("Decision vector and bound dimensions differ.")
    below = np.where(np.isfinite(lower), lower - values, 0.0)
    above = np.where(np.isfinite(upper), values - upper, 0.0)
    return float(max(0.0, np.max(below), np.max(above)))


def _recomputed_constraint_violation(
    *,
    tasks: _TaskConstraints,
    adjacent: LinearConstraint | None,
    bounds: Bounds,
    x: Array,
    config: GlobalConfigurationOptimizerConfig,
    exclusion: NodeExclusionConstraints | None = None,
) -> float:
    """Re-evaluate every hard constraint on the serialized candidate.

    ``trust-constr`` reports its aggregate violation at its own final iterate.
    The fixed initial state is restored exactly after that iterate is returned,
    so the solver-reported value can be stale.  Candidate acceptance must use
    the state that will actually be validated and saved.
    """
    return float(
        max(
            _task_violation(tasks, x),
            _linear_constraint_violation(adjacent, x),
            _bound_constraint_violation(bounds, x),
            _exclusion_violation(exclusion, x, config),
        )
    )


def _outcome_selection_key(
    outcome: _SolveOutcome,
    acceptance_tolerance: float,
) -> tuple[bool, float, float]:
    """Rank feasible outcomes by objective and infeasible ones by violation."""
    feasible = bool(outcome.constraint_violation <= acceptance_tolerance)
    if feasible:
        return (False, float(outcome.objective), float(outcome.constraint_violation))
    return (True, float(outcome.constraint_violation), float(outcome.objective))


def _select_best_outcome(
    outcomes: Sequence[_SolveOutcome],
    acceptance_tolerance: float,
) -> _SolveOutcome:
    if not outcomes:
        raise ValueError("At least one solve outcome is required.")
    return min(
        outcomes,
        key=lambda item: _outcome_selection_key(item, acceptance_tolerance),
    )


def _print_seed_physical_audit(
    *,
    tasks: _TaskConstraints,
    seed: _SeedPath,
    flat_state: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> None:
    states = np.asarray(flat_state, dtype=float).reshape(tasks.node_count, 7)
    position_errors = np.empty(tasks.node_count, dtype=float)
    tangent_errors = np.empty(tasks.node_count, dtype=float)
    for index, state in enumerate(states):
        output = tasks.evaluator.output(index, state)
        position_error, tangent_error, _, _ = _physical_errors(
            output,
            seed.desired_position[index],
            seed.desired_tangent[index],
        )
        position_errors[index] = position_error
        tangent_errors[index] = tangent_error
    worst_position = int(np.argmax(position_errors))
    worst_tangent = int(np.argmax(tangent_errors))
    feasible = bool(
        np.all(position_errors <= config.position_tolerance_m)
        and np.all(tangent_errors <= config.tangent_tolerance_rad)
    )
    print(
        "[GLOBAL SEED AUDIT] "
        f"nodes={tasks.node_count} feasible={feasible}\n"
        f"  worst_position_node={worst_position} "
        f"s={seed.s[worst_position]:.6f} m "
        f"error={1.0e3 * position_errors[worst_position]:.4f} mm\n"
        f"  worst_tangent_node={worst_tangent} "
        f"s={seed.s[worst_tangent]:.6f} m "
        f"error={np.degrees(tangent_errors[worst_tangent]):.4f} deg",
        flush=True,
    )


def _solve_global_problem(
    *,
    adapter: Any,
    seed: _SeedPath,
    state_min: Array,
    state_max: Array,
    config: GlobalConfigurationOptimizerConfig,
    path: CentrelinePath | None = None,
) -> tuple[_SolveOutcome, _QuadraticPathObjective, _TaskConstraints]:
    node_count = seed.s.size
    objective = _QuadraticPathObjective(
        s=seed.s,
        seed_states=seed.states,
        trusted_seed_mask=seed.trusted_seed_mask,
        state_min=state_min,
        state_max=state_max,
        config=config,
    )
    node_baselines, replay_outputs = _replay_node_baselines(
        adapter,
        seed.states,
    )
    if config.debug and replay_outputs is not None:
        print(
            "[GLOBAL BRANCH REPLAY] "
            f"nodes={node_count} node_specific_baselines=True",
            flush=True,
        )
    evaluator = _BatchBeamEvaluator(
        adapter,
        node_count,
        node_baselines=node_baselines,
    )
    # The optimiser is constrained against the tolerance it is *allowed to
    # spend*; reporting elsewhere still uses the physical tolerance, so the
    # difference becomes margin rather than budget.
    spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
        spend_fraction=config.tolerance_spend_fraction,
    )
    if config.debug and config.tolerance_spend_fraction < 1.0:
        print(
            "[GLOBAL TOLERANCE] "
            f"spend_fraction={config.tolerance_spend_fraction:.3f} "
            f"position={1.0e3 * spend_position_tolerance:.6f} mm of "
            f"{1.0e3 * config.position_tolerance_m:.6f} mm",
            flush=True,
        )
    tasks = _TaskConstraints(
        evaluator=evaluator,
        desired_position=seed.desired_position,
        desired_tangent=seed.desired_tangent,
        position_tolerance_m=spend_position_tolerance,
        tangent_tolerance_rad=spend_tangent_tolerance,
    )
    exclusion: NodeExclusionConstraints | None = None
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        if path is None:
            raise ValueError(
                "A source-magnet/lumen keep-out radius is configured but no "
                "CentrelinePath was supplied to the global solve. Pass "
                "path=CentrelinePath(lumen_C)."
            )
        exclusion = NodeExclusionConstraints(
            adapter=adapter,
            path=path,
            state_min=state_min,
            state_max=state_max,
            node_count=node_count,
            radius_m=float(config.source_magnet_lumen_exclusion_radius_m),
            constraint_tolerance_m=float(
                config.source_magnet_lumen_constraint_tolerance_m
            ),
            require_analytical_jacobian=bool(
                config.require_analytical_magnet_position_jacobian
            ),
            joint_step_rad=float(config.magnet_jacobian_joint_step_rad),
            insertion_step_m=float(config.magnet_jacobian_insertion_step_m),
        )
    if config.tip_centring_weight > 0.0:
        # One extra block-diagonal quadratic in the same QP.  Linearised at
        # the seed, which is also where this round's beam baselines are
        # frozen, so the cost and the constraints see the same model.
        H_tip, f_tip, constant_tip = tip_centring_quadratic(
            evaluator=evaluator,
            seed_states=seed.states,
            desired_position=seed.desired_position,
            position_tolerance_m=config.position_tolerance_m,
            weight=float(config.tip_centring_weight),
        )
        objective.H = csc_matrix(objective.H + H_tip)
        objective.f = np.asarray(objective.f, dtype=float) + f_tip
        objective.constant = float(objective.constant) + constant_tip
        if config.debug:
            print(
                "[GLOBAL TIP COST] "
                f"weight={config.tip_centring_weight:.6g} linearised at seed",
                flush=True,
            )
    if config.tip_curvature_weight > 0.0:
        # Couples adjacent nodes' *achieved* tip positions, linearised at the
        # same seed and beam baselines as everything else this round.  This is
        # the term that makes the tip path itself smooth; tip_centring_weight
        # above only pulls each node toward its own static target.
        H_curve, f_curve, constant_curve = tip_curvature_quadratic(
            evaluator=evaluator,
            seed_states=seed.states,
            s=seed.s,
            weight=float(config.tip_curvature_weight),
        )
        objective.H = csc_matrix(objective.H + H_curve)
        objective.f = np.asarray(objective.f, dtype=float) + f_curve
        objective.constant = float(objective.constant) + constant_curve
        if config.debug:
            print(
                "[GLOBAL TIP CURVATURE] "
                f"weight={config.tip_curvature_weight:.6g} linearised at seed",
                flush=True,
            )
    if config.magnet_path_weight > 0.0 or config.magnet_curvature_weight > 0.0:
        # Prefer a short, direct external-magnet path.  Linearised at the seed
        # via the analytical magnet-position Jacobian; the keep-out inequality
        # still holds the magnet outside the head, this only stops it wandering
        # inside the space the keep-out and the tip-tracking null space leave.
        magnet_positions, magnet_jacobians = _magnet_path_terms(
            adapter=adapter, states=seed.states, config=config
        )
        H_mag, f_mag, constant_mag = magnet_path_quadratic(
            magnet_positions=magnet_positions,
            magnet_jacobians=magnet_jacobians,
            seed_states=seed.states,
            s=seed.s,
            first_weight=float(config.magnet_path_weight),
            second_weight=float(config.magnet_curvature_weight),
        )
        objective.H = csc_matrix(objective.H + H_mag)
        objective.f = np.asarray(objective.f, dtype=float) + f_mag
        objective.constant = float(objective.constant) + constant_mag
        if config.debug:
            seed_length = float(
                np.sum(np.linalg.norm(np.diff(magnet_positions, axis=0), axis=1))
            )
            straight = float(
                np.linalg.norm(magnet_positions[-1] - magnet_positions[0])
            )
            print(
                "[GLOBAL MAGNET PATH] "
                f"path_weight={config.magnet_path_weight:.6g} "
                f"curvature_weight={config.magnet_curvature_weight:.6g} "
                f"seed magnet length={1.0e3 * seed_length:.3f} mm "
                f"straight-line={1.0e3 * straight:.3f} mm "
                f"(ratio {seed_length / max(straight, 1.0e-9):.2f})",
                flush=True,
            )
    bounds = _decision_bounds(
        node_count=node_count,
        state_min=state_min,
        state_max=state_max,
        initial_state=seed.initial_state,
        fix_initial_state=config.fix_initial_state,
    )
    adjacent = _adjacent_linear_constraint(node_count=node_count, config=config)
    base_x = np.clip(seed.states.reshape(-1), bounds.lb, bounds.ub)

    base_task_violation = _task_violation(tasks, base_x)
    base_adjacent_violation = _linear_constraint_violation(adjacent, base_x)
    base_bound_violation = _bound_constraint_violation(bounds, base_x)
    # The keep-out belongs in this total too.  Without it, a seed that is
    # tracking-feasible but sits inside the exclusion region reads as
    # violation-free, and preserve_feasible_seed_on_failure would then retain
    # an unsafe path in preference to the optimiser's safe one.
    base_exclusion_violation = _exclusion_violation(exclusion, base_x, config)
    base_violation = max(
        base_task_violation,
        base_adjacent_violation,
        base_bound_violation,
        base_exclusion_violation,
    )

    # Violation of the seed against the *physical* task tolerance (never the
    # spent one).  Used only to decide whether the retained-seed fallback is
    # armed; the spent tolerance still governs what the optimiser drives to.
    if (
        config.preserve_feasible_seed_against_physical_tolerance
        and config.tolerance_spend_fraction < 1.0
    ):
        physical_tasks = _TaskConstraints(
            evaluator=evaluator,
            desired_position=seed.desired_position,
            desired_tangent=seed.desired_tangent,
            position_tolerance_m=config.position_tolerance_m,
            tangent_tolerance_rad=config.tangent_tolerance_rad,
        )
        base_task_violation_physical = _task_violation(physical_tasks, base_x)
    else:
        base_task_violation_physical = base_task_violation
    base_violation_for_fallback = max(
        base_task_violation_physical,
        base_adjacent_violation,
        base_bound_violation,
        base_exclusion_violation,
    )
    if config.debug:
        _print_seed_physical_audit(
            tasks=tasks,
            seed=seed,
            flat_state=base_x,
            config=config,
        )
        print(
            "[GLOBAL SEED AUDIT] "
            f"task_violation={base_task_violation:.6e} "
            f"adjacent_violation={base_adjacent_violation:.6e} "
            f"bound_violation={base_bound_violation:.6e} "
            f"exclusion_violation={base_exclusion_violation:.6e}",
            flush=True,
        )

    rng = np.random.default_rng(int(config.random_seed))
    outcomes: list[_SolveOutcome] = []
    if (
        config.preserve_feasible_seed_on_failure
        and base_violation_for_fallback <= config.constraint_tolerance
    ):
        retained_note = (
            "feasible inverse seed retained as optimizer fallback"
            if base_violation <= config.constraint_tolerance
            else "inverse seed retained as optimizer fallback "
            "(meets physical tolerance; spent tolerance not required for the "
            "fallback)"
        )
        outcomes.append(
            _SolveOutcome(
                x=base_x.copy(),
                success=False,
                status=-1,
                message=retained_note,
                iterations=0,
                objective=float(objective.value(base_x)),
                constraint_violation=float(base_violation_for_fallback),
                restoration_slack_maximum=0.0,
            )
        )
    for attempt in range(int(config.maximum_multistart_attempts)):
        x0 = base_x.copy()
        if attempt > 0:
            states = x0.reshape(node_count, 7)
            perturbation = np.column_stack(
                (
                    rng.uniform(
                        -config.multistart_joint_perturbation_rad,
                        config.multistart_joint_perturbation_rad,
                        size=(node_count, 6),
                    ),
                    rng.uniform(
                        -config.multistart_insertion_perturbation_m,
                        config.multistart_insertion_perturbation_m,
                        size=node_count,
                    ),
                )
            )
            untrusted = ~seed.trusted_seed_mask
            states[untrusted] += perturbation[untrusted]
            states[:] = np.clip(states, state_min, state_max)
            states[:, 6] = np.maximum(states[:, 6], _MIN_INSERTION_M)
            if config.fix_initial_state:
                states[0] = seed.initial_state
            x0 = states.reshape(-1)

        restoration_slack = 0.0
        restoration_result = None
        if (
            config.feasibility_restoration_enabled
            and base_violation > config.constraint_tolerance
        ):
            if config.debug:
                print(
                    f"[GLOBAL PATH] phase=restoration attempt={attempt + 1} "
                    f"nodes={node_count}"
                )
            x0, restoration_slack, restoration_result = _run_restoration(
                x0=x0,
                objective=objective,
                tasks=tasks,
                bounds=bounds,
                adjacent=adjacent,
                config=config,
                exclusion=exclusion,
            )

        if config.debug:
            print(
                f"[GLOBAL PATH] phase=hard_constrained attempt={attempt + 1} "
                f"restoration_slack_max={restoration_slack:.6e}"
            )
        solved, monitor = _run_hard_solve(
            x0=x0,
            objective=objective,
            tasks=tasks,
            bounds=bounds,
            adjacent=adjacent,
            config=config,
            exclusion=exclusion,
        )
        x = np.asarray(solved.x, dtype=float)
        if config.fix_initial_state:
            # ``trust-constr`` may leave equality bounds at its numerical
            # feasibility tolerance.  Restore the explicitly fixed state
            # exactly before physical validation and serialization.
            x[:7] = seed.initial_state
        # Do not use ``solved.constr_violation`` here.  It describes SciPy's
        # pre-correction iterate and can remain nonzero after the fixed first
        # state has been restored exactly above.
        violation = _recomputed_constraint_violation(
            tasks=tasks,
            adjacent=adjacent,
            bounds=bounds,
            x=x,
            config=config,
            exclusion=exclusion,
        )
        message = str(solved.message)
        if restoration_result is not None and not bool(restoration_result.success):
            message = (
                f"restoration_status={restoration_result.status}: "
                f"{restoration_result.message}; hard_status={solved.status}: {message}"
            )
        outcomes.append(
            _SolveOutcome(
                x=x,
                success=bool(solved.success),
                status=int(solved.status),
                message=message,
                iterations=int(getattr(solved, "nit", 0)),
                objective=float(objective.value(x)),
                constraint_violation=float(violation),
                restoration_slack_maximum=float(restoration_slack),
            )
        )
        if monitor.best_feasible_x is not None:
            monitored_x = np.asarray(monitor.best_feasible_x, dtype=float).copy()
            if config.fix_initial_state:
                monitored_x[:7] = seed.initial_state
            if not np.allclose(monitored_x, x, atol=0.0, rtol=0.0):
                monitored_violation = _recomputed_constraint_violation(
                    tasks=tasks,
                    adjacent=adjacent,
                    bounds=bounds,
                    x=monitored_x,
                    config=config,
                    exclusion=exclusion,
                )
                outcomes.append(
                    _SolveOutcome(
                        x=monitored_x,
                        success=False,
                        status=3,
                        message=(
                            "best feasible iterate retained by progress monitor"
                            + (
                                f"; {monitor.stop_reason}"
                                if monitor.stop_reason is not None
                                else ""
                            )
                        ),
                        iterations=int(monitor.best_feasible_iteration),
                        objective=float(objective.value(monitored_x)),
                        constraint_violation=float(monitored_violation),
                        restoration_slack_maximum=float(restoration_slack),
                    )
                )
        if bool(solved.success) and violation <= config.constraint_tolerance:
            break

    acceptance_tolerance = float(config.constraint_tolerance)
    best = _select_best_outcome(
        outcomes,
        acceptance_tolerance,
    )
    if config.debug:
        for index, outcome in enumerate(outcomes):
            print(
                "[GLOBAL CANDIDATE] "
                f"index={index} "
                f"feasible={outcome.constraint_violation <= acceptance_tolerance} "
                f"objective={outcome.objective:.6e} "
                f"violation={outcome.constraint_violation:.6e} "
                f"solver_success={outcome.success}",
                flush=True,
            )
        print(
            "[GLOBAL SELECTION] "
            "rule=feasible_then_objective_else_violation "
            f"acceptance_tolerance={acceptance_tolerance:.6e} "
            f"selected_objective={best.objective:.6e} "
            f"selected_violation={best.constraint_violation:.6e}",
            flush=True,
        )
    return best, objective, tasks


def _merge_coordinates(values: Iterable[float], tolerance: float = 1.0e-10) -> Array:
    sorted_values = np.sort(np.asarray(list(values), dtype=float))
    if sorted_values.size == 0:
        return sorted_values
    merged = [float(sorted_values[0])]
    for value in sorted_values[1:]:
        if float(value) - merged[-1] > tolerance:
            merged.append(float(value))
    return np.asarray(merged, dtype=float)


def _uniform_path_coordinates(start: float, end: float, step: float) -> Array:
    if not 0.0 <= start <= end:
        raise ValueError("Invalid path start/end coordinates.")
    ratio = (end - start) / step
    nearest = round(ratio)
    # Avoid adding an entire extra interval when an exact decimal ratio is
    # represented as, for example, 54.00000000000001 in binary floating point.
    if math.isclose(ratio, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
        count = max(1, int(nearest))
    else:
        count = max(1, int(math.ceil(ratio)))
    return np.linspace(start, end, count + 1, dtype=float)


def _validate_inverse_result(
    *,
    inverse_result: InversePathResult,
    controller_pack: dict[str, Any],
    path: CentrelinePath,
    config: GlobalConfigurationOptimizerConfig,
) -> tuple[Array, Array, Array]:
    if not inverse_result.nodes:
        raise ValueError("inverse_result contains no nodes.")
    state_min = _finite_vector(controller_pack["p_min"], 7, "controller p_min")
    state_max = _finite_vector(controller_pack["p_max"], 7, "controller p_max")
    if np.any(state_min >= state_max):
        raise ValueError("Controller state bounds are invalid.")
    initial_state = _finite_vector(controller_pack["p0"], 7, "controller p0")

    s = np.asarray([node.s_m for node in inverse_result.nodes], dtype=float)
    if np.any(~np.isfinite(s)) or np.any(np.diff(s) <= 0.0):
        raise ValueError("Inverse path coordinates must be finite and increasing.")
    if s[0] < -1.0e-10 or s[-1] > path.total_length_m + 1.0e-8:
        raise ValueError("Inverse path coordinates lie outside the lumen path.")
    states = np.array(
        [_state_from_inverse_node(node) for node in inverse_result.nodes], dtype=float
    )
    if np.any(states < state_min - 1.0e-10) or np.any(states > state_max + 1.0e-10):
        raise ValueError("Inverse result contains a state outside absolute bounds.")
    if not np.allclose(states[0], initial_state, atol=1.0e-8, rtol=0.0):
        raise ValueError("Inverse result initial state differs from controller_pack['p0'].")
    for index, node in enumerate(inverse_result.nodes):
        _unit_vector(node.desired_tangent, f"desired tangent at inverse node {index}")
        _unit_vector(node.tip_tangent, f"tip tangent at inverse node {index}")

    inverse_position_tolerance = _float_or_nan(
        inverse_result.configuration.get("position_tolerance_m")
    )
    inverse_tangent_tolerance = _float_or_nan(
        inverse_result.configuration.get("tangent_tolerance_rad")
    )
    if not config.allow_tolerance_override:
        if np.isfinite(inverse_position_tolerance) and not np.isclose(
            inverse_position_tolerance,
            config.position_tolerance_m,
            atol=1.0e-12,
            rtol=1.0e-9,
        ):
            raise ValueError("Global position tolerance differs from inverse result.")
        if np.isfinite(inverse_tangent_tolerance) and not np.isclose(
            inverse_tangent_tolerance,
            config.tangent_tolerance_rad,
            atol=1.0e-12,
            rtol=1.0e-9,
        ):
            raise ValueError("Global tangent tolerance differs from inverse result.")

    feasible = np.asarray([bool(node.feasible) for node in inverse_result.nodes])
    complete = bool(
        inverse_result.all_nodes_feasible
        and np.all(feasible)
        and s[-1] >= path.total_length_m - max(config.minimum_path_step_m, 1.0e-8)
    )
    if config.mode == "refine_complete" and not complete:
        raise ValueError(
            "mode='refine_complete' requires an all-feasible inverse result that "
            "reaches the lumen-path end. Use mode='recover_partial'."
        )
    if not np.any(feasible):
        raise ValueError("Inverse result contains no feasible seed nodes.")
    return state_min, state_max, initial_state


def _coarsened_complete_coordinates(
    *,
    inverse_result: InversePathResult,
    config: GlobalConfigurationOptimizerConfig,
) -> Array:
    """Return a smaller exact subset of a complete inverse grid.

    The previous implementation retained all 113 inverse nodes even when
    ``path_step_m`` requested a 0.5 mm optimization grid.  Every nonlinear
    iteration therefore evaluated 113 contact-aware beam Jacobians.  This
    routine keeps exact saved states at a roughly uniform spacing, always keeps
    both endpoints, and also retains nodes on the source-magnet keep-out
    boundary.  Dense validation/refinement fills back any interval that needs
    more resolution.
    """
    inverse_s = np.asarray([node.s_m for node in inverse_result.nodes], dtype=float)
    if not config.coarsen_complete_seed or inverse_s.size <= 2:
        return inverse_s.copy()

    selected = {0, inverse_s.size - 1}
    last = 0
    target = float(config.path_step_m)
    for index in range(1, inverse_s.size - 1):
        if inverse_s[index] - inverse_s[last] >= target * (1.0 - 1.0e-9):
            selected.add(index)
            last = index

    if config.source_magnet_lumen_exclusion_radius_m is not None:
        active_threshold = max(
            2.0 * float(config.source_magnet_lumen_constraint_tolerance_m),
            1.0e-9,
        )
        for index, node in enumerate(inverse_result.nodes):
            margin = float(node.source_magnet_lumen_margin_m)
            if np.isfinite(margin) and margin <= active_threshold:
                selected.add(index)

    return inverse_s[np.asarray(sorted(selected), dtype=int)]


def _make_seed_path(
    *,
    inverse_result: InversePathResult,
    path: CentrelinePath,
    state_min: Array,
    state_max: Array,
    initial_state: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> _SeedPath:
    inverse_s = np.asarray([node.s_m for node in inverse_result.nodes], dtype=float)
    inverse_states = np.array(
        [_state_from_inverse_node(node) for node in inverse_result.nodes], dtype=float
    )
    feasible_mask = np.asarray([bool(node.feasible) for node in inverse_result.nodes])
    usable_mask = feasible_mask.copy()
    if config.use_failed_nodes_as_warm_start:
        usable_mask |= np.all(np.isfinite(inverse_states), axis=1)

    start_s = float(inverse_s[0])
    if config.mode == "refine_complete":
        complete_coordinates = _coarsened_complete_coordinates(
            inverse_result=inverse_result,
            config=config,
        )
        coordinates = complete_coordinates[
            np.isin(complete_coordinates, inverse_s[feasible_mask])
        ]
    else:
        uniform = _uniform_path_coordinates(
            start_s,
            path.total_length_m,
            config.path_step_m,
        )
        coordinates = _merge_coordinates(
            list(uniform) + list(inverse_s[feasible_mask]),
            tolerance=1.0e-9,
        )
    if coordinates.size > config.maximum_nodes:
        raise ValueError(
            f"Initial global grid has {coordinates.size} nodes, exceeding "
            f"maximum_nodes={config.maximum_nodes}. Increase path_step_m or the limit."
        )

    usable_s = inverse_s[usable_mask]
    usable_states = inverse_states[usable_mask]
    feasible_s = inverse_s[feasible_mask]
    feasible_states = inverse_states[feasible_mask]
    states = np.empty((coordinates.size, 7), dtype=float)
    last_usable_s = float(usable_s[-1])
    for coordinate in range(7):
        states[:, coordinate] = np.interp(
            coordinates,
            usable_s,
            usable_states[:, coordinate],
        )

    # Bounded extrapolation beyond the available inverse result.  This is only
    # a warm start; the hard nonlinear problem determines final feasibility.
    if coordinates[-1] > last_usable_s + 1.0e-12:
        if usable_s.size >= 2:
            ds_seed = usable_s[-1] - usable_s[-2]
            slope = (
                (usable_states[-1] - usable_states[-2]) / ds_seed
                if ds_seed > 1.0e-12
                else np.zeros(7)
            )
        else:
            slope = np.zeros(7)
        slope *= float(config.partial_extrapolation_gain)
        # Do not invent an insertion/path law.  When the successful inverse
        # prefix shows insertion growth, that measured slope is extrapolated;
        # otherwise insertion is initially held and left for restoration.
        last_index = int(np.searchsorted(coordinates, last_usable_s, side="right") - 1)
        for index in range(last_index + 1, coordinates.size):
            ds = coordinates[index] - coordinates[index - 1]
            proposal = states[index - 1] + ds * slope
            maximum = np.concatenate(
                (
                    np.asarray(config.maximum_adjacent_joint_change_rad, dtype=float),
                    [float(config.maximum_adjacent_insertion_change_m)],
                )
            )
            proposal = np.clip(proposal, states[index - 1] - maximum, states[index - 1] + maximum)
            if config.insertion_non_decreasing:
                proposal[6] = max(proposal[6], states[index - 1, 6])
            states[index] = np.clip(proposal, state_min, state_max)

    states[0] = initial_state
    states[:] = np.clip(states, state_min, state_max)
    # The beam model rejects insertion L <= 0; state_min[6] is exactly 0, so an
    # interpolated/extrapolated seed node can land on it.  Floor it.
    states[:, 6] = np.maximum(states[:, 6], _MIN_INSERTION_M)
    # Interpolated states within the all-feasible prefix are useful trusted
    # seeds, while extrapolated/recovered states beyond it should not be held
    # near an arbitrary initial guess by the seed-deviation objective.
    trusted = coordinates <= float(feasible_s[-1]) + 1.0e-10

    desired_position = np.array([path.position(s) for s in coordinates], dtype=float)
    desired_tangent = np.array([path.tangent(s) for s in coordinates], dtype=float)
    return _SeedPath(
        s=coordinates,
        states=states,
        desired_position=desired_position,
        desired_tangent=desired_tangent,
        trusted_seed_mask=trusted,
        initial_state=initial_state.copy(),
        last_feasible_progress_m=float(feasible_s[-1]),
    )


def _physical_errors(output: Array, desired_position: Array, desired_tangent: Array) -> tuple:
    output = _finite_vector(output, 6, "beam output")
    tangent = _unit_vector(output[3:6], "beam tangent")
    desired_tangent = _unit_vector(desired_tangent, "desired tangent")
    position_error = float(np.linalg.norm(output[:3] - desired_position))
    tangent_error = float(
        np.arccos(np.clip(tangent @ desired_tangent, -1.0, 1.0))
    )
    return position_error, tangent_error, output[:3].copy(), tangent


def audit_inverse_seed_round_trip(
    *,
    inverse_result: InversePathResult,
    controller_pack: dict[str, Any],
    lumen_C: Array,
    config: GlobalConfigurationOptimizerConfig,
    commit_forward: bool = False,
) -> dict[str, Any]:
    """Recompute a saved inverse seed without running global optimization.

    The audit isolates serialization, reference reconstruction, and beam
    equilibrium repeatability.  It intentionally evaluates forward outputs
    only; no chained beam Jacobians or dense between-node samples are computed.
    ``commit_forward=True`` is a diagnostic continuation experiment and must
    not be used inside the nonlinear optimizer because it makes evaluation
    history-dependent.
    """
    config.validate()
    required = ("p0", "p_min", "p_max", "plant_diagnostic_joint_adapter")
    missing = [key for key in required if key not in controller_pack]
    if missing:
        raise KeyError("controller_pack is missing: " + ", ".join(missing))
    adapter = controller_pack["plant_diagnostic_joint_adapter"]
    if int(getattr(adapter, "n_out", 6)) != 6:
        raise ValueError("Seed audit requires the full six-output adapter.")
    flags = _contact_model_flags(adapter)
    if config.require_contact_model:
        if flags["enabled"] is not True or flags["use_in_jacobian"] is not True:
            raise RuntimeError(
                "The supplied full-output adapter is not contact-aware in both "
                "the forward model and chained Jacobian."
            )

    path = CentrelinePath(lumen_C)
    state_min, state_max, initial_state = _validate_inverse_result(
        inverse_result=inverse_result,
        controller_pack=controller_pack,
        path=path,
        config=config,
    )
    seed = _make_seed_path(
        inverse_result=inverse_result,
        path=path,
        state_min=state_min,
        state_max=state_max,
        initial_state=initial_state,
        config=config,
    )
    node_baselines, replay_outputs = _replay_node_baselines(
        adapter,
        seed.states,
    )
    evaluator = _BatchBeamEvaluator(
        adapter,
        seed.s.size,
        node_baselines=node_baselines,
    )
    if commit_forward and replay_outputs is not None:
        outputs = replay_outputs.copy()
        forward_evaluations = int(seed.s.size)
    elif commit_forward:
        outputs = np.array(
            [
                np.asarray(
                    adapter.forward_output(state, commit=True), dtype=float
                ).reshape(-1)[:6]
                for state in seed.states
            ],
            dtype=float,
        )
        if outputs.shape != (seed.s.size, 6) or not np.all(np.isfinite(outputs)):
            raise FloatingPointError(
                "Committed seed audit produced an invalid beam output."
            )
        forward_evaluations = int(seed.s.size)
    else:
        outputs = np.array(
            [
                evaluator.output(index, state)
                for index, state in enumerate(seed.states)
            ],
            dtype=float,
        )
        forward_evaluations = int(evaluator.output_evaluations)

    position_errors = np.empty(seed.s.size, dtype=float)
    tangent_errors = np.empty(seed.s.size, dtype=float)
    for index, output in enumerate(outputs):
        position_errors[index], tangent_errors[index], _, _ = _physical_errors(
            output,
            seed.desired_position[index],
            seed.desired_tangent[index],
        )

    saved_tip_drift = []
    saved_tangent_drift = []
    desired_position_drift = []
    desired_tangent_drift = []
    for inverse_node in inverse_result.nodes:
        seed_index = int(np.argmin(np.abs(seed.s - inverse_node.s_m)))
        if abs(float(seed.s[seed_index]) - float(inverse_node.s_m)) > 1.0e-9:
            raise RuntimeError(
                "Saved inverse coordinate was not retained by seed construction."
            )
        output = outputs[seed_index]
        saved_tip_drift.append(
            float(
                np.linalg.norm(
                    output[:3]
                    - _finite_vector(
                        inverse_node.tip_position_m,
                        3,
                        "saved inverse tip position",
                    )
                )
            )
        )
        saved_tangent = _unit_vector(
            inverse_node.tip_tangent,
            "saved inverse tip tangent",
        )
        recomputed_tangent = _unit_vector(
            output[3:6],
            "recomputed seed tangent",
        )
        saved_tangent_drift.append(
            float(
                np.arccos(
                    np.clip(saved_tangent @ recomputed_tangent, -1.0, 1.0)
                )
            )
        )
        path_position, path_tangent = path.reference(float(inverse_node.s_m))
        desired_position_drift.append(
            float(
                np.linalg.norm(
                    _finite_vector(
                        inverse_node.desired_position_m,
                        3,
                        "saved desired position",
                    )
                    - path_position
                )
            )
        )
        saved_desired_tangent = _unit_vector(
            inverse_node.desired_tangent,
            "saved desired tangent",
        )
        desired_tangent_drift.append(
            float(
                np.arccos(
                    np.clip(saved_desired_tangent @ path_tangent, -1.0, 1.0)
                )
            )
        )

    task_values = np.concatenate(
        (
            1.0 - position_errors**2 / config.position_tolerance_m**2,
            (
                np.cos(tangent_errors)
                - math.cos(config.tangent_tolerance_rad)
            )
            / max(1.0 - math.cos(config.tangent_tolerance_rad), 1.0e-12),
        )
    )
    adjacent = _adjacent_linear_constraint(
        node_count=seed.s.size,
        config=config,
    )
    adjacent_violation = _linear_constraint_violation(
        adjacent,
        seed.states.reshape(-1),
    )
    worst_position = int(np.argmax(position_errors))
    worst_tangent = int(np.argmax(tangent_errors))
    node_feasible = bool(
        np.all(position_errors <= config.position_tolerance_m)
        and np.all(tangent_errors <= config.tangent_tolerance_rad)
        and adjacent_violation <= config.constraint_tolerance
    )
    report = {
        "node_feasible": node_feasible,
        "number_inverse_nodes": len(inverse_result.nodes),
        "number_seed_nodes": int(seed.s.size),
        "saved_maximum_position_error_m": float(
            max(node.position_error_m for node in inverse_result.nodes)
        ),
        "saved_maximum_tangent_error_rad": float(
            max(node.tangent_error_rad for node in inverse_result.nodes)
        ),
        "recomputed_maximum_position_error_m": float(
            position_errors[worst_position]
        ),
        "recomputed_maximum_tangent_error_rad": float(
            tangent_errors[worst_tangent]
        ),
        "worst_position_node": worst_position,
        "worst_position_s_m": float(seed.s[worst_position]),
        "worst_tangent_node": worst_tangent,
        "worst_tangent_s_m": float(seed.s[worst_tangent]),
        "maximum_saved_tip_position_drift_m": float(max(saved_tip_drift)),
        "maximum_saved_tip_tangent_drift_rad": float(max(saved_tangent_drift)),
        "maximum_desired_position_drift_m": float(max(desired_position_drift)),
        "maximum_desired_tangent_drift_rad": float(max(desired_tangent_drift)),
        "normalized_task_constraint_violation": float(
            np.max(np.maximum(0.0, -task_values))
        ),
        "adjacent_constraint_violation": float(adjacent_violation),
        "forward_evaluations": forward_evaluations,
        "commit_forward": bool(commit_forward),
        "node_specific_equilibrium_baselines": bool(
            node_baselines is not None
        ),
        "contact_model": flags,
    }
    if config.debug:
        print(
            "[ROUND-TRIP SEED AUDIT] "
            f"feasible={node_feasible} "
            f"commit_forward={bool(commit_forward)} "
            f"inverse_nodes={len(inverse_result.nodes)} "
            f"seed_nodes={seed.s.size}\n"
            f"  saved_max_position="
            f"{1.0e3 * report['saved_maximum_position_error_m']:.4f} mm "
            f"recomputed_max_position="
            f"{1.0e3 * report['recomputed_maximum_position_error_m']:.4f} mm\n"
            f"  saved_to_recomputed_tip_drift="
            f"{1.0e3 * report['maximum_saved_tip_position_drift_m']:.4f} mm "
            f"desired_reference_drift="
            f"{1.0e3 * report['maximum_desired_position_drift_m']:.6f} mm\n"
            f"  task_violation="
            f"{report['normalized_task_constraint_violation']:.6e} "
            f"adjacent_violation={adjacent_violation:.6e}",
            flush=True,
        )
    return _json_ready(report)


def _build_node_results(
    *,
    adapter: Any,
    seed: _SeedPath,
    states: Array,
    state_min: Array,
    state_max: Array,
    config: GlobalConfigurationOptimizerConfig,
    node_baselines: Sequence[Any] | None = None,
    compute_jacobian_diagnostics: bool | None = None,
) -> list[GlobalConfigurationNodeResult]:
    evaluator = _BatchBeamEvaluator(
        adapter,
        seed.s.size,
        node_baselines=node_baselines,
    )
    if compute_jacobian_diagnostics is None:
        compute_jacobian_diagnostics = config.compute_node_jacobian_diagnostics
    results: list[GlobalConfigurationNodeResult] = []
    for index, state in enumerate(states):
        output = evaluator.output(index, state)
        position_error, tangent_error, achieved_position, achieved_tangent = (
            _physical_errors(
                output, seed.desired_position[index], seed.desired_tangent[index]
            )
        )
        if compute_jacobian_diagnostics:
            J = evaluator.jacobian(index, state)
            normalized_tangent_J = _normalized_tangent_jacobian(
                output[3:6], J[3:6]
            )
            scaled_task_J = np.vstack(
                (
                    J[:3] / config.position_tolerance_m,
                    normalized_tangent_J
                    / max(
                        2.0 * math.sin(0.5 * config.tangent_tolerance_rad),
                        1.0e-12,
                    ),
                )
            )
            condition, rank = _effective_condition(scaled_task_J)
        else:
            condition, rank = np.nan, 0
        diagnostics = _contact_diagnostics(adapter)
        joint_margin = np.minimum(
            state[:6] - state_min[:6], state_max[:6] - state[:6]
        )
        state_margin = np.minimum(state - state_min, state_max - state)
        feasible = bool(
            position_error <= config.position_tolerance_m
            and tangent_error <= config.tangent_tolerance_rad
            and np.all(state >= state_min - 1.0e-10)
            and np.all(state <= state_max + 1.0e-10)
        )
        results.append(
            GlobalConfigurationNodeResult(
                node_index=index,
                s_m=float(seed.s[index]),
                q_rad=state[:6].copy(),
                insertion_m=float(state[6]),
                magnet_pose6=_magnet_pose6(adapter, state),
                desired_position_m=seed.desired_position[index].copy(),
                achieved_position_m=achieved_position,
                desired_tangent=seed.desired_tangent[index].copy(),
                achieved_tangent=achieved_tangent,
                position_error_m=position_error,
                tangent_error_rad=tangent_error,
                feasible=feasible,
                position_constraint_margin_m=(
                    config.position_tolerance_m - position_error
                ),
                tangent_constraint_margin_rad=(
                    config.tangent_tolerance_rad - tangent_error
                ),
                minimum_joint_margin_rad=float(np.min(joint_margin)),
                minimum_state_margin=float(np.min(state_margin)),
                jacobian_condition=condition,
                jacobian_effective_rank=rank,
                contact_active=bool(diagnostics["active"]),
                contact_diagnostic_available=bool(diagnostics["available"]),
                minimum_gap_m=float(diagnostics["minimum_gap_m"]),
                contact_energy=float(diagnostics["contact_energy"]),
                contact_force_norm=float(diagnostics["contact_force_norm"]),
                seed_state=seed.states[index].copy(),
                state_change_from_seed=(state - seed.states[index]),
            )
        )
    return results


def _dense_validate(
    *,
    adapter: Any,
    path: CentrelinePath,
    s_nodes: Array,
    states: Array,
    state_min: Array,
    state_max: Array,
    config: GlobalConfigurationOptimizerConfig,
    interval_baselines: Sequence[Any] | None = None,
) -> list[DenseValidationSample]:
    samples: list[DenseValidationSample] = []
    sample_count = int(config.dense_validation_samples_per_interval)
    spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
        spend_fraction=config.tolerance_spend_fraction,
    )
    interval_evaluator = _BatchBeamEvaluator(
        adapter,
        s_nodes.size - 1,
        node_baselines=interval_baselines,
    )
    dense_exclusion: NodeExclusionConstraints | None = None
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        dense_exclusion = NodeExclusionConstraints(
            adapter=adapter,
            path=path,
            state_min=state_min,
            state_max=state_max,
            node_count=1,
            radius_m=float(config.source_magnet_lumen_exclusion_radius_m),
            constraint_tolerance_m=float(
                config.source_magnet_lumen_constraint_tolerance_m
            ),
            require_analytical_jacobian=bool(
                config.require_analytical_magnet_position_jacobian
            ),
            joint_step_rad=float(config.magnet_jacobian_joint_step_rad),
            insertion_step_m=float(config.magnet_jacobian_insertion_step_m),
        )
    for interval in range(s_nodes.size - 1):
        for sample_index in range(1, sample_count + 1):
            fraction = sample_index / (sample_count + 1.0)
            s_value = float(
                (1.0 - fraction) * s_nodes[interval]
                + fraction * s_nodes[interval + 1]
            )
            state = (
                (1.0 - fraction) * states[interval]
                + fraction * states[interval + 1]
            )
            output = interval_evaluator.output(interval, state)
            desired_position, desired_tangent = path.reference(s_value)
            position_error, tangent_error, achieved_position, achieved_tangent = (
                _physical_errors(output, desired_position, desired_tangent)
            )
            exclusion_distance = np.nan
            exclusion_margin = np.nan
            exclusion_satisfied = True
            if dense_exclusion is not None:
                exclusion_metrics = dense_exclusion.metrics(state.reshape(-1))
                exclusion_distance = float(
                    np.asarray(exclusion_metrics["distance_m"], dtype=float).reshape(-1)[0]
                )
                exclusion_margin = exclusion_distance - float(
                    config.source_magnet_lumen_exclusion_radius_m
                )
                exclusion_satisfied = bool(
                    np.asarray(exclusion_metrics["satisfied"], dtype=bool).reshape(-1)[0]
                )
            feasible = bool(
                position_error <= spend_position_tolerance
                and tangent_error <= spend_tangent_tolerance
                and np.all(state >= state_min - 1.0e-10)
                and np.all(state <= state_max + 1.0e-10)
                and exclusion_satisfied
            )
            samples.append(
                DenseValidationSample(
                    interval_index=interval,
                    fraction=float(fraction),
                    s_m=s_value,
                    state=state.copy(),
                    desired_position_m=desired_position,
                    achieved_position_m=achieved_position,
                    desired_tangent=desired_tangent,
                    achieved_tangent=achieved_tangent,
                    position_error_m=position_error,
                    tangent_error_rad=tangent_error,
                    position_margin_m=spend_position_tolerance - position_error,
                    tangent_margin_rad=spend_tangent_tolerance - tangent_error,
                    feasible=feasible,
                    source_magnet_lumen_distance_m=exclusion_distance,
                    source_magnet_lumen_margin_m=exclusion_margin,
                    source_magnet_lumen_constraint_satisfied=exclusion_satisfied,
                )
            )
    return samples


def _dense_sample_violations(
    sample: DenseValidationSample,
    config: GlobalConfigurationOptimizerConfig,
) -> dict[str, float]:
    """Per-constraint excess over tolerance for one dense sample, >0 = violated.

    Box constraints are excluded on purpose: every dense sample is a convex
    combination of two box-feasible node states, and the box is itself
    convex, so a dense sample can never violate it -- position/tangent/
    exclusion are the only ways a dense sample fails.
    """
    spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
        spend_fraction=config.tolerance_spend_fraction,
    )
    margin = sample.source_magnet_lumen_margin_m
    return {
        "position_m": sample.position_error_m - spend_position_tolerance,
        "tangent_rad": sample.tangent_error_rad - spend_tangent_tolerance,
        "exclusion_m": (
            -margin if np.isfinite(margin) and not sample.source_magnet_lumen_constraint_satisfied
            else float("-inf")
        ),
    }


def _dense_sample_excusable(
    sample: DenseValidationSample,
    config: GlobalConfigurationOptimizerConfig,
) -> bool:
    """Is every constraint this sample violates within its acceptance slack?

    ``sample.feasible`` (and the reported per-sample margin diagnostics) are
    computed against the strict, zero-slack constraints and are never touched
    by this -- this only decides whether such a sample should still block
    round-acceptance. Each of the three slacks defaults to 0.0, reproducing
    the previous strict all-or-nothing behaviour exactly.
    """
    violations = _dense_sample_violations(sample, config)
    if violations["position_m"] > float(config.dense_position_acceptance_slack_m):
        return False
    if violations["tangent_rad"] > float(config.dense_tangent_acceptance_slack_rad):
        return False
    if violations["exclusion_m"] > float(config.dense_magnet_exclusion_acceptance_slack_m):
        return False
    return True


def _describe_dense_failure(
    sample: DenseValidationSample,
    config: GlobalConfigurationOptimizerConfig,
) -> str:
    """A one-line, human-readable breakdown of why the worst dense sample failed."""
    violations = _dense_sample_violations(sample, config)
    parts = []
    if violations["position_m"] > 0.0:
        parts.append(f"position over by {1.0e3 * violations['position_m']:.4f} mm")
    if violations["tangent_rad"] > 0.0:
        parts.append(f"tangent over by {math.degrees(violations['tangent_rad']):.4f} deg")
    if violations["exclusion_m"] > 0.0:
        parts.append(
            f"magnet-exclusion margin negative by {1.0e3 * violations['exclusion_m']:.4f} mm"
        )
    if not parts:
        parts.append("no violation (this sample should be feasible -- check sample.feasible directly)")
    return (
        f"interval={sample.interval_index} fraction={sample.fraction:.2f} "
        f"s={1.0e3 * sample.s_m:.2f}mm: " + "; ".join(parts)
    )


def _worst_dense_sample(
    samples: Sequence[DenseValidationSample],
    config: GlobalConfigurationOptimizerConfig,
) -> DenseValidationSample | None:
    failures = [
        sample for sample in samples
        if not sample.feasible and not _dense_sample_excusable(sample, config)
    ]
    if not failures:
        return None
    spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
        spend_fraction=config.tolerance_spend_fraction,
    )
    return max(
        failures,
        key=lambda sample: max(
            sample.position_error_m / spend_position_tolerance,
            sample.tangent_error_rad / spend_tangent_tolerance,
            (
                1.0
                + max(0.0, -sample.source_magnet_lumen_margin_m)
                / float(config.source_magnet_lumen_constraint_tolerance_m)
                if not sample.source_magnet_lumen_constraint_satisfied
                else 0.0
            ),
        ),
    )


def _insert_refinement_node(
    *,
    seed: _SeedPath,
    optimized_states: Array,
    sample: DenseValidationSample,
    path: CentrelinePath,
    config: GlobalConfigurationOptimizerConfig,
) -> _SeedPath:
    if seed.s.size + 1 > config.maximum_nodes:
        raise RuntimeError("Adaptive refinement would exceed maximum_nodes.")
    if np.min(np.abs(seed.s - sample.s_m)) < 0.5 * config.minimum_path_step_m:
        raise RuntimeError(
            "Dense violation cannot be refined without going below "
            "minimum_path_step_m."
        )
    insertion = int(np.searchsorted(seed.s, sample.s_m))
    s_new = np.insert(seed.s, insertion, sample.s_m)
    states_new = np.insert(optimized_states, insertion, sample.state, axis=0)
    position_new = np.insert(
        np.array([path.position(s) for s in seed.s]),
        insertion,
        path.position(sample.s_m),
        axis=0,
    )
    tangent_new = np.insert(
        np.array([path.tangent(s) for s in seed.s]),
        insertion,
        path.tangent(sample.s_m),
        axis=0,
    )
    trusted_new = np.insert(seed.trusted_seed_mask, insertion, False)
    return _SeedPath(
        s=s_new,
        states=states_new,
        desired_position=position_new,
        desired_tangent=tangent_new,
        trusted_seed_mask=trusted_new,
        initial_state=seed.initial_state.copy(),
        last_feasible_progress_m=seed.last_feasible_progress_m,
    )


# A round whose recomputed task-constraint violation exceeds this has driven
# the seed well outside the tolerance ball (|error| > ~10 tol).  More
# refinement rounds only compound it, and the inverse-path fallback below is
# strictly better, so the loop stops.
_DIVERGENCE_VIOLATION = 100.0


def _exact_inverse_seed_path(
    *,
    inverse_result: InversePathResult,
    path: CentrelinePath,
    state_min: Array,
    state_max: Array,
    initial_state: Array,
    config: GlobalConfigurationOptimizerConfig,
) -> _SeedPath:
    """The un-coarsened feasible inverse path, for use as the last-resort output.

    This is what "fall back to the inverse configuration path" means: every
    feasible saved node, at its own coordinate, with no coarsening and no
    extrapolated tail.  Built the same way regardless of the requested mode so
    a ``recover_partial`` run that fails still returns its feasible prefix.
    """
    return _make_seed_path(
        inverse_result=inverse_result,
        path=path,
        state_min=state_min,
        state_max=state_max,
        initial_state=initial_state,
        config=replace(
            config,
            mode="refine_complete",
            coarsen_complete_seed=False,
            use_failed_nodes_as_warm_start=False,
        ),
    )


def _copy_seed(seed: _SeedPath) -> _SeedPath:
    return _SeedPath(
        s=seed.s.copy(),
        states=seed.states.copy(),
        desired_position=seed.desired_position.copy(),
        desired_tangent=seed.desired_tangent.copy(),
        trusted_seed_mask=seed.trusted_seed_mask.copy(),
        initial_state=seed.initial_state.copy(),
        last_feasible_progress_m=seed.last_feasible_progress_m,
    )


def optimize_from_inverse_result(
    *,
    inverse_result: InversePathResult,
    controller_pack: dict[str, Any],
    lumen_C: Array,
    config: GlobalConfigurationOptimizerConfig,
    output_dir: str | Path | None,
) -> GlobalConfigurationPathResult:
    """Globally refine/recover and smooth an inverse configuration path.

    Guarantee: this never returns a path worse than the inverse feasible seed.
    The refinement loop keeps the best (feasible-then-least-violating) round it
    has seen; if no round produces a feasible path, the exact inverse feasible
    path is returned unchanged, flagged ``fallback_to_inverse_path``.
    """
    config.validate()
    required = ("p0", "p_min", "p_max", "plant_diagnostic_joint_adapter")
    missing = [key for key in required if key not in controller_pack]
    if missing:
        raise KeyError("controller_pack is missing: " + ", ".join(missing))
    adapter = controller_pack["plant_diagnostic_joint_adapter"]
    if int(getattr(adapter, "n_out", 6)) != 6:
        raise ValueError("Global position+tangent optimization requires n_out=6.")
    flags = _contact_model_flags(adapter)
    if config.require_contact_model:
        if flags["enabled"] is not True or flags["use_in_jacobian"] is not True:
            raise RuntimeError(
                "The supplied full-output adapter is not contact-aware in both "
                "the forward model and chained Jacobian."
            )

    path = CentrelinePath(lumen_C)
    state_min, state_max, initial_state = _validate_inverse_result(
        inverse_result=inverse_result,
        controller_pack=controller_pack,
        path=path,
        config=config,
    )
    seed = _make_seed_path(
        inverse_result=inverse_result,
        path=path,
        state_min=state_min,
        state_max=state_max,
        initial_state=initial_state,
        config=config,
    )
    if config.debug:
        saved_position_errors = np.asarray(
            [node.position_error_m for node in inverse_result.nodes], dtype=float
        )
        saved_tangent_errors = np.asarray(
            [node.tangent_error_rad for node in inverse_result.nodes], dtype=float
        )
        saved_worst_position = int(np.nanargmax(saved_position_errors))
        saved_worst_tangent = int(np.nanargmax(saved_tangent_errors))
        print(
            "[GLOBAL SEED GRID] "
            f"inverse_nodes={len(inverse_result.nodes)} "
            f"optimization_nodes={seed.s.size} "
            f"requested_path_step={config.path_step_m:.6e} m\n"
            + (
                "  Complete refinement uses the saved inverse coordinates "
                + (
                    "on an adaptive coarse grid; keep-out boundary nodes are retained."
                    if config.coarsen_complete_seed
                    else "exactly."
                )
                if config.mode == "refine_complete"
                else "  Recovery retains feasible inverse coordinates and "
                "adds missing uniform coordinates."
            ),
            flush=True,
        )
        print(
            "[SAVED INVERSE AUDIT] "
            f"worst_position_node={saved_worst_position} "
            f"error={1.0e3 * saved_position_errors[saved_worst_position]:.4f} mm "
            f"worst_tangent_node={saved_worst_tangent} "
            f"error={np.degrees(saved_tangent_errors[saved_worst_tangent]):.4f} deg",
            flush=True,
        )

    optimization_started = time.perf_counter()
    # Best round seen so far, ranked (feasible first, then least task violation,
    # then least tip error).  The optimiser output of a single round is never
    # returned directly -- only the best round is, and only if it is feasible;
    # otherwise the inverse-path fallback below wins.
    best_key: tuple[int, float, float] | None = None
    best_round: dict[str, Any] | None = None
    any_feasible_round = False
    refinement_round = 0
    termination_reason = "maximum_refinement_rounds_reached"
    for refinement_round in range(config.maximum_refinement_rounds + 1):
        round_config = config
        if config.maximum_wall_time_s is not None:
            remaining_wall_time = float(config.maximum_wall_time_s) - (
                time.perf_counter() - optimization_started
            )
            if remaining_wall_time <= 0.0 and best_round is not None:
                termination_reason = "maximum_wall_time_reached"
                break
            round_config = replace(
                config,
                maximum_wall_time_s=max(1.0, remaining_wall_time),
            )
        outcome, objective, _tasks = _solve_global_problem(
            adapter=adapter,
            seed=seed,
            state_min=state_min,
            state_max=state_max,
            config=round_config,
            path=path,
        )
        # A solve that stopped early (wall time / stagnation) can return an
        # iterate a hair outside the box; the beam model rejects L <= 0, so
        # clamp before it is replayed or validated.
        states = np.clip(
            outcome.x.reshape(seed.s.size, 7),
            np.asarray(state_min, dtype=float),
            np.asarray(state_max, dtype=float),
        )
        states[:, 6] = np.maximum(states[:, 6], _MIN_INSERTION_M)
        result_baselines, _result_replay_outputs = _replay_node_baselines(
            adapter,
            states,
        )
        interval_baselines = (
            None
            if result_baselines is None
            else result_baselines[1:]
        )
        if config.dense_validation_enabled:
            dense_samples = _dense_validate(
                adapter=adapter,
                path=path,
                s_nodes=seed.s,
                states=states,
                state_min=state_min,
                state_max=state_max,
                config=config,
                interval_baselines=interval_baselines,
            )
            worst = _worst_dense_sample(dense_samples, config)
            dense_excused_samples = [
                sample for sample in dense_samples
                if not sample.feasible and _dense_sample_excusable(sample, config)
            ]
        else:
            dense_samples = []
            worst = None
            dense_excused_samples = []
        node_results_for_check = _build_node_results(
            adapter=adapter,
            seed=seed,
            states=states,
            state_min=state_min,
            state_max=state_max,
            config=config,
            node_baselines=result_baselines,
        )
        nodes_feasible = all(node.feasible for node in node_results_for_check)
        dense_ok = (not config.dense_validation_enabled) or (worst is None)
        round_feasible = bool(
            float(outcome.constraint_violation) <= float(config.constraint_tolerance)
            and nodes_feasible
            and dense_ok
        )
        if config.debug and not dense_ok:
            # worst is the sample that will actually block this round (i.e. it
            # is NOT within any configured slack) -- say exactly why, in the
            # same units the CLI's slack/tolerance flags use, every round.
            # This is the answer to "why did dense validation fail", instead
            # of needing to reload the run and guess from the summary JSON.
            print(f"[GLOBAL DENSE GAP] {_describe_dense_failure(worst, config)}", flush=True)
        if round_feasible and dense_excused_samples:
            if config.debug:
                for label, key, slack, unit, scale in (
                    ("magnet-exclusion", "exclusion_m",
                     config.dense_magnet_exclusion_acceptance_slack_m, "mm", 1.0e3),
                    ("position", "position_m",
                     config.dense_position_acceptance_slack_m, "mm", 1.0e3),
                    ("tangent", "tangent_rad",
                     config.dense_tangent_acceptance_slack_rad, "deg", 180.0 / math.pi),
                ):
                    worst_violation = max(
                        (_dense_sample_violations(s, config)[key] for s in dense_excused_samples),
                        default=float("-inf"),
                    )
                    if worst_violation > 0.0:
                        print(
                            f"[GLOBAL DENSE SLACK] round accepted with a {label} "
                            f"near-miss excused (worst excess {scale * worst_violation:.4f} "
                            f"{unit}, slack {scale * slack:.4f} {unit})",
                            flush=True,
                        )
        any_feasible_round = any_feasible_round or round_feasible
        max_round_error = max(
            [node.position_error_m for node in node_results_for_check]
            + [sample.position_error_m for sample in dense_samples]
            + [0.0]
        )
        round_key = (
            0 if round_feasible else 1,
            float(outcome.constraint_violation),
            float(max_round_error),
        )
        if best_key is None or round_key < best_key:
            best_key = round_key
            best_round = {
                "outcome": outcome,
                "objective": objective,
                "states": states,
                "nodes": node_results_for_check,
                "dense_samples": dense_samples,
                "seed": _copy_seed(seed),
                "feasible": round_feasible,
                "dense_slack_excused_count": len(dense_excused_samples),
                "dense_slack_excused_worst_position_m": max(
                    (_dense_sample_violations(s, config)["position_m"]
                     for s in dense_excused_samples),
                    default=float("nan"),
                ),
                "dense_slack_excused_worst_tangent_rad": max(
                    (_dense_sample_violations(s, config)["tangent_rad"]
                     for s in dense_excused_samples),
                    default=float("nan"),
                ),
                "dense_slack_excused_worst_exclusion_m": max(
                    (_dense_sample_violations(s, config)["exclusion_m"]
                     for s in dense_excused_samples),
                    default=float("nan"),
                ),
            }
        if config.debug:
            max_position = max(node.position_error_m for node in node_results_for_check)
            max_tangent = max(node.tangent_error_rad for node in node_results_for_check)
            dense_status = (
                str(worst is None)
                if config.dense_validation_enabled
                else "skipped"
            )
            print(
                f"[GLOBAL PATH] refinement={refinement_round} "
                f"solver_success={outcome.success} nodes_feasible={nodes_feasible} "
                f"dense_feasible={dense_status} round_feasible={round_feasible}\n"
                f"  objective={outcome.objective:.6e} "
                f"constraint_violation={outcome.constraint_violation:.6e}\n"
                f"  max_position_error={1.0e3 * max_position:.4f} mm "
                f"max_tangent_error={np.degrees(max_tangent):.4f} deg"
            )
        if round_feasible:
            if dense_excused_samples:
                termination_reason = "hard_constraints_satisfied_dense_exclusion_excused"
            else:
                termination_reason = (
                    "node_constraints_satisfied_dense_validation_disabled"
                    if not config.dense_validation_enabled
                    else "hard_constraints_and_dense_validation_satisfied"
                )
            break
        if worst is None:
            termination_reason = "node_constraints_not_satisfied"
            break
        if float(outcome.constraint_violation) > _DIVERGENCE_VIOLATION:
            # The optimiser drove this seed far outside the tolerance ball.
            # Refining from that iterate only compounds it; stop and let the
            # inverse-path fallback take over.
            termination_reason = "optimizer_diverged_from_seed"
            break
        if (
            config.maximum_wall_time_s is not None
            and time.perf_counter() - optimization_started
            >= float(config.maximum_wall_time_s)
        ):
            termination_reason = "maximum_wall_time_reached_after_validation"
            break
        if refinement_round >= config.maximum_refinement_rounds:
            termination_reason = "maximum_refinement_rounds_reached"
            break
        seed = _insert_refinement_node(
            seed=seed,
            optimized_states=states,
            sample=worst,
            path=path,
            config=config,
        )

    assert best_round is not None
    fallback_to_inverse_path = not any_feasible_round
    if fallback_to_inverse_path:
        # No global round produced a feasible path.  Return the exact inverse
        # feasible path unchanged rather than a stalled/diverged iterate.
        fb_seed = _exact_inverse_seed_path(
            inverse_result=inverse_result,
            path=path,
            state_min=state_min,
            state_max=state_max,
            initial_state=initial_state,
            config=config,
        )
        fb_baselines, _fb_replay = _replay_node_baselines(adapter, fb_seed.states)
        fb_nodes = _build_node_results(
            adapter=adapter,
            seed=fb_seed,
            states=fb_seed.states,
            state_min=state_min,
            state_max=state_max,
            config=config,
            node_baselines=fb_baselines,
            compute_jacobian_diagnostics=False,
        )
        if config.dense_validation_enabled:
            fb_dense = _dense_validate(
                adapter=adapter,
                path=path,
                s_nodes=fb_seed.s,
                states=fb_seed.states,
                state_min=state_min,
                state_max=state_max,
                config=config,
                interval_baselines=(
                    None if fb_baselines is None else fb_baselines[1:]
                ),
            )
        else:
            fb_dense = []
        fb_objective = _QuadraticPathObjective(
            s=fb_seed.s,
            seed_states=fb_seed.states,
            trusted_seed_mask=fb_seed.trusted_seed_mask,
            state_min=state_min,
            state_max=state_max,
            config=config,
        )
        fb_flat = fb_seed.states.reshape(-1)
        fb_tasks = _TaskConstraints(
            evaluator=_BatchBeamEvaluator(
                adapter, fb_seed.s.size, node_baselines=fb_baselines
            ),
            desired_position=fb_seed.desired_position,
            desired_tangent=fb_seed.desired_tangent,
            position_tolerance_m=config.position_tolerance_m,
            tangent_tolerance_rad=config.tangent_tolerance_rad,
        )
        final_outcome = _SolveOutcome(
            x=fb_flat.copy(),
            success=False,
            status=-2,
            message=(
                "inverse configuration path returned unchanged: no global "
                "round produced a feasible path (inverse seed retained)"
            ),
            iterations=0,
            objective=float(fb_objective.value(fb_flat)),
            constraint_violation=float(_task_violation(fb_tasks, fb_flat)),
            restoration_slack_maximum=0.0,
        )
        final_objective = fb_objective
        final_states = fb_seed.states
        final_nodes = fb_nodes
        dense_samples = fb_dense
        seed = fb_seed
        if termination_reason not in (
            "optimizer_diverged_from_seed",
            "maximum_wall_time_reached",
            "maximum_wall_time_reached_after_validation",
        ):
            termination_reason = "inverse_path_fallback_no_feasible_global_round"
        if config.debug:
            print(
                "[GLOBAL FALLBACK] no feasible global round; returning the "
                f"exact inverse feasible path ({fb_seed.s.size} nodes, "
                f"worst tip error "
                f"{1.0e3 * max(n.position_error_m for n in fb_nodes):.4f} mm)",
                flush=True,
            )
    else:
        final_outcome = best_round["outcome"]
        final_objective = best_round["objective"]
        final_states = best_round["states"]
        final_nodes = best_round["nodes"]
        dense_samples = best_round["dense_samples"]
        seed = best_round["seed"]

    assert final_outcome is not None
    assert final_objective is not None
    assert final_states is not None
    assert final_nodes is not None
    nodes = final_nodes
    nodes_feasible = bool(all(node.feasible for node in nodes))
    dense_feasible = bool(
        config.dense_validation_enabled
        and all(sample.feasible for sample in dense_samples)
    )
    selection_acceptance_tolerance = float(config.constraint_tolerance)
    selected_candidate_feasible = bool(
        final_outcome.constraint_violation <= selection_acceptance_tolerance
    )
    globally_feasible = bool(
        selected_candidate_feasible and nodes_feasible and dense_feasible
    )
    first, second = final_objective.derivatives(final_states)
    maximum_first = (
        np.max(np.abs(first), axis=0) if first.size else np.zeros(7, dtype=float)
    )
    maximum_second = (
        np.max(np.abs(second), axis=0) if second.size else np.zeros(7, dtype=float)
    )
    maximum_position_error = float(
        max(
            [node.position_error_m for node in nodes]
            + [sample.position_error_m for sample in dense_samples]
        )
    )
    maximum_tangent_error = float(
        max(
            [node.tangent_error_rad for node in nodes]
            + [sample.tangent_error_rad for sample in dense_samples]
        )
    )
    exclusion_summary: dict[str, Any] = {
        "enabled": config.source_magnet_lumen_exclusion_radius_m is not None,
        "radius_m": config.source_magnet_lumen_exclusion_radius_m,
    }
    if config.source_magnet_lumen_exclusion_radius_m is not None:
        _final_exclusion = NodeExclusionConstraints(
            adapter=adapter,
            path=path,
            state_min=state_min,
            state_max=state_max,
            node_count=len(final_nodes),
            radius_m=float(config.source_magnet_lumen_exclusion_radius_m),
            constraint_tolerance_m=float(
                config.source_magnet_lumen_constraint_tolerance_m
            ),
            require_analytical_jacobian=bool(
                config.require_analytical_magnet_position_jacobian
            ),
            joint_step_rad=float(config.magnet_jacobian_joint_step_rad),
            insertion_step_m=float(config.magnet_jacobian_insertion_step_m),
        )
        _metrics = _final_exclusion.metrics(final_states.reshape(-1))
        exclusion_summary.update(
            {
                "minimum_distance_m": float(np.min(_metrics["distance_m"])),
                "minimum_margin_m": float(_metrics["minimum_margin_m"]),
                "all_nodes_satisfied": bool(_metrics["all_satisfied"]),
                "violating_node_count": int(
                    np.sum(~np.asarray(_metrics["satisfied"], dtype=bool))
                ),
                "jacobian_source": str(_metrics["jacobian_source"]),
                "dense_minimum_margin_m": (
                    float(
                        np.nanmin(
                            [
                                sample.source_magnet_lumen_margin_m
                                for sample in dense_samples
                            ]
                        )
                    )
                    if dense_samples
                    else np.nan
                ),
                "all_dense_samples_satisfied": bool(
                    all(
                        sample.source_magnet_lumen_constraint_satisfied
                        for sample in dense_samples
                    )
                ),
            }
        )
        if config.debug:
            print(
                "[GLOBAL EXCLUSION] "
                f"minimum_margin={1.0e3 * exclusion_summary['minimum_margin_m']:.6f} mm "
                f"satisfied={exclusion_summary['all_nodes_satisfied']}",
                flush=True,
            )

    normalized_state_change = (
        final_states - seed.states
    ) / np.asarray(config.configuration_scale, dtype=float)[None, :]
    maximum_normalized_state_change = float(
        np.max(np.linalg.norm(normalized_state_change, axis=1))
    )
    selected_seed_fallback = bool(
        "inverse seed retained" in final_outcome.message
    )
    summary = {
        "globally_feasible": globally_feasible,
        "source_magnet_lumen_exclusion": exclusion_summary,
        "dense_validation_feasible": dense_feasible,
        "optimizer_success": final_outcome.success,
        "termination_reason": termination_reason,
        "optimizer_message": final_outcome.message,
        "optimizer_status": final_outcome.status,
        "optimizer_iterations": final_outcome.iterations,
        "objective_value": final_outcome.objective,
        "selected_seed_fallback": selected_seed_fallback,
        "fallback_to_inverse_path": bool(fallback_to_inverse_path),
        "any_feasible_global_round": bool(any_feasible_round),
        "dense_magnet_exclusion_acceptance_slack_m": float(
            config.dense_magnet_exclusion_acceptance_slack_m
        ),
        "dense_position_acceptance_slack_m": float(
            config.dense_position_acceptance_slack_m
        ),
        "dense_tangent_acceptance_slack_rad": float(
            config.dense_tangent_acceptance_slack_rad
        ),
        "best_round_dense_slack_excused_count": int(
            best_round.get("dense_slack_excused_count", 0)
        ),
        "best_round_dense_slack_excused_worst_position_m": float(
            best_round.get("dense_slack_excused_worst_position_m", float("nan"))
        ),
        "best_round_dense_slack_excused_worst_tangent_rad": float(
            best_round.get("dense_slack_excused_worst_tangent_rad", float("nan"))
        ),
        "best_round_dense_slack_excused_worst_exclusion_m": float(
            best_round.get("dense_slack_excused_worst_exclusion_m", float("nan"))
        ),
        "maximum_normalized_state_change_from_seed": (
            maximum_normalized_state_change
        ),
        "normalized_constraint_violation": final_outcome.constraint_violation,
        "selected_candidate_feasible": selected_candidate_feasible,
        "candidate_selection_rule": "feasible_then_objective_else_violation",
        "candidate_acceptance_tolerance": selection_acceptance_tolerance,
        "restoration_slack_maximum": final_outcome.restoration_slack_maximum,
        "number_nodes": len(nodes),
        "number_dense_samples": len(dense_samples),
        "refinement_rounds": refinement_round,
        "maximum_position_error_m": maximum_position_error,
        "maximum_tangent_error_rad": maximum_tangent_error,
        "maximum_first_difference": maximum_first,
        "maximum_second_difference": maximum_second,
        "contact_model": flags,
        "state_layout": "q1_q2_q3_q4_q5_q6_insertion",
        "path_parameter": "lumen_arc_length_m",
        "time_parameterized": False,
    }
    result = GlobalConfigurationPathResult(
        nodes=nodes,
        dense_samples=dense_samples,
        globally_feasible=globally_feasible,
        dense_validation_feasible=dense_feasible,
        optimizer_success=final_outcome.success,
        termination_reason=termination_reason,
        iterations=final_outcome.iterations,
        objective_value=final_outcome.objective,
        maximum_position_error_m=maximum_position_error,
        maximum_tangent_error_rad=maximum_tangent_error,
        maximum_first_difference=maximum_first,
        maximum_second_difference=maximum_second,
        restoration_slack_maximum=final_outcome.restoration_slack_maximum,
        refinement_rounds=refinement_round,
        configuration=_json_ready(config),
        summary=_json_ready(summary),
    )
    if output_dir is not None:
        save_global_configuration_result(result, output_dir)
    return result


def _inverse_node_from_json(data: dict[str, Any]) -> InverseNodeResult:
    return InverseNodeResult(
        node_index=int(data["node_index"]),
        s_m=float(data["s_m"]),
        q_rad=np.asarray(data["q_rad"], dtype=float),
        insertion_m=float(data["insertion_m"]),
        magnet_pose6=np.asarray(data["magnet_pose6"], dtype=float),
        tip_position_m=np.asarray(data["tip_position_m"], dtype=float),
        tip_tangent=np.asarray(data["tip_tangent"], dtype=float),
        desired_position_m=np.asarray(data["desired_position_m"], dtype=float),
        desired_tangent=np.asarray(data["desired_tangent"], dtype=float),
        position_error_m=float(data["position_error_m"]),
        tangent_error_rad=float(data["tangent_error_rad"]),
        feasible=bool(data["feasible"]),
        solver_success=bool(data["solver_success"]),
        termination_reason=str(data["termination_reason"]),
        function_evaluations=int(data["function_evaluations"]),
        objective_value=float(data["objective_value"]),
        jacobian_condition=_float_or_nan(data.get("jacobian_condition")),
        jacobian_effective_rank=int(data.get("jacobian_effective_rank", 0)),
        minimum_joint_margin_rad=_float_or_nan(data.get("minimum_joint_margin_rad")),
        minimum_state_margin=_float_or_nan(data.get("minimum_state_margin")),
        minimum_clearance_m=_float_or_nan(data.get("minimum_clearance_m")),
        contact_active=bool(data.get("contact_active", False)),
        # Head-exclusion diagnostics were added after the original saved-path
        # schema.  ``get`` keeps older inverse results loadable; NaN/-1 means
        # the historical file contains no exclusion-distance diagnostic.
        source_magnet_lumen_distance_m=_float_or_nan(
            data.get("source_magnet_lumen_distance_m")
        ),
        source_magnet_lumen_exclusion_radius_m=_float_or_nan(
            data.get("source_magnet_lumen_exclusion_radius_m")
        ),
        source_magnet_lumen_margin_m=_float_or_nan(
            data.get("source_magnet_lumen_margin_m")
        ),
        source_magnet_lumen_segment_index=int(
            data.get("source_magnet_lumen_segment_index", -1)
        ),
        source_magnet_lumen_constraint_satisfied=bool(
            data.get("source_magnet_lumen_constraint_satisfied", True)
        ),
        delta_q_norm_rad=_float_or_nan(data.get("delta_q_norm_rad")),
        delta_insertion_m=_float_or_nan(data.get("delta_insertion_m")),
        attempt_count=int(data.get("attempt_count", 0)),
        chain_rule_relative_error=_float_or_nan(
            data.get("chain_rule_relative_error")
        ),
    )


def load_inverse_path_result(inverse_output_dir: str | Path) -> InversePathResult:
    """Load and cross-check the sequential planner's JSON/CSV/NPZ outputs."""
    root = Path(inverse_output_dir)
    json_path = root / "inverse_configuration_summary.json"
    csv_path = root / "inverse_configuration_path.csv"
    npz_path = root / "inverse_configuration_diagnostics.npz"
    for path in (json_path, csv_path, npz_path):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = [_inverse_node_from_json(item) for item in payload.get("nodes", [])]
    if not nodes:
        raise ValueError("Saved inverse JSON contains no nodes.")
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(csv_rows) != len(nodes):
        raise ValueError("Inverse CSV and JSON node counts differ.")
    with np.load(npz_path) as archive:
        s_npz = np.asarray(archive["s_m"], dtype=float)
        states_npz = np.asarray(archive["state_q_L"], dtype=float)
    states_json = np.array([_state_from_inverse_node(node) for node in nodes])
    s_json = np.array([node.s_m for node in nodes], dtype=float)
    if not np.allclose(s_npz, s_json, atol=1.0e-12, rtol=0.0):
        raise ValueError("Inverse NPZ and JSON path coordinates differ.")
    if not np.allclose(states_npz, states_json, atol=1.0e-10, rtol=0.0):
        raise ValueError("Inverse NPZ and JSON states differ.")
    summary = dict(payload.get("summary", {}))
    configuration = dict(payload.get("configuration", {}))
    feasible_indices = [index for index, node in enumerate(nodes) if node.feasible]
    last_feasible = feasible_indices[-1] if feasible_indices else -1
    return InversePathResult(
        nodes=nodes,
        all_nodes_feasible=bool(summary.get("all_nodes_feasible", False)),
        last_feasible_index=int(summary.get("last_feasible_index", last_feasible)),
        last_feasible_progress_m=_float_or_nan(
            summary.get(
                "last_feasible_progress_m",
                nodes[last_feasible].s_m if last_feasible >= 0 else np.nan,
            )
        ),
        configuration=configuration,
        summary=summary,
        output_files={
            "csv": str(csv_path),
            "json": str(json_path),
            "npz": str(npz_path),
        },
    )


def optimize_from_saved_inverse_result(
    *,
    inverse_output_dir: str | Path,
    controller_pack: dict[str, Any],
    lumen_C: Array,
    config: GlobalConfigurationOptimizerConfig,
    output_dir: str | Path | None,
) -> GlobalConfigurationPathResult:
    inverse_result = load_inverse_path_result(inverse_output_dir)
    return optimize_from_inverse_result(
        inverse_result=inverse_result,
        controller_pack=controller_pack,
        lumen_C=lumen_C,
        config=config,
        output_dir=output_dir,
    )


def _node_csv_record(node: GlobalConfigurationNodeResult) -> dict[str, Any]:
    record: dict[str, Any] = {
        "node_index": node.node_index,
        "s_m": node.s_m,
        "insertion_m": node.insertion_m,
        "position_error_m": node.position_error_m,
        "position_error_mm": 1.0e3 * node.position_error_m,
        "tangent_error_rad": node.tangent_error_rad,
        "tangent_error_deg": np.degrees(node.tangent_error_rad),
        "feasible": int(node.feasible),
        "position_constraint_margin_m": node.position_constraint_margin_m,
        "tangent_constraint_margin_rad": node.tangent_constraint_margin_rad,
        "minimum_joint_margin_rad": node.minimum_joint_margin_rad,
        "minimum_state_margin": node.minimum_state_margin,
        "jacobian_condition": node.jacobian_condition,
        "jacobian_effective_rank": node.jacobian_effective_rank,
        "contact_active": int(node.contact_active),
        "contact_diagnostic_available": int(node.contact_diagnostic_available),
        "minimum_gap_m": node.minimum_gap_m,
        "contact_energy": node.contact_energy,
        "contact_force_norm": node.contact_force_norm,
    }
    for index, value in enumerate(node.q_rad, start=1):
        record[f"q{index}_rad"] = float(value)
    for prefix, vector in (
        ("magnet_pose", node.magnet_pose6),
        ("desired", node.desired_position_m),
        ("achieved", node.achieved_position_m),
        ("desired_tangent", node.desired_tangent),
        ("achieved_tangent", node.achieved_tangent),
        ("seed", node.seed_state),
        ("change_from_seed", node.state_change_from_seed),
    ):
        if vector.size == 7:
            labels = ("q1", "q2", "q3", "q4", "q5", "q6", "L")
        elif vector.size == 6:
            labels = ("x", "y", "z", "rx", "ry", "rz")
        else:
            labels = ("x", "y", "z")
        for label, value in zip(labels, vector):
            record[f"{prefix}_{label}"] = float(value)
    return record


def _dense_csv_record(sample: DenseValidationSample) -> dict[str, Any]:
    record: dict[str, Any] = {
        "interval_index": sample.interval_index,
        "fraction": sample.fraction,
        "s_m": sample.s_m,
        "position_error_m": sample.position_error_m,
        "position_error_mm": 1.0e3 * sample.position_error_m,
        "tangent_error_rad": sample.tangent_error_rad,
        "tangent_error_deg": np.degrees(sample.tangent_error_rad),
        "position_margin_m": sample.position_margin_m,
        "tangent_margin_rad": sample.tangent_margin_rad,
        "source_magnet_lumen_distance_m": (
            sample.source_magnet_lumen_distance_m
        ),
        "source_magnet_lumen_margin_m": sample.source_magnet_lumen_margin_m,
        "source_magnet_lumen_constraint_satisfied": int(
            sample.source_magnet_lumen_constraint_satisfied
        ),
        "feasible": int(sample.feasible),
    }
    for index, value in enumerate(sample.state, start=1):
        label = f"q{index}_rad" if index <= 6 else "insertion_m"
        record[label] = float(value)
    return record


def save_global_configuration_result(
    result: GlobalConfigurationPathResult, output_dir: str | Path
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "global_configuration_path.csv"
    json_path = root / "global_configuration_summary.json"
    npz_path = root / "global_configuration_diagnostics.npz"
    plot_path = root / "global_configuration_path.png"
    dense_path = root / "global_configuration_dense_validation.csv"

    node_records = [_node_csv_record(node) for node in result.nodes]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(node_records[0].keys()))
        writer.writeheader()
        writer.writerows(node_records)
    dense_records = [_dense_csv_record(sample) for sample in result.dense_samples]
    with dense_path.open("w", newline="", encoding="utf-8") as stream:
        if dense_records:
            writer = csv.DictWriter(stream, fieldnames=list(dense_records[0].keys()))
            writer.writeheader()
            writer.writerows(dense_records)

    payload = {
        "configuration": result.configuration,
        "summary": result.summary,
        "nodes": [_json_ready(node) for node in result.nodes],
        "dense_samples": [_json_ready(sample) for sample in result.dense_samples],
    }
    json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    s = np.array([node.s_m for node in result.nodes], dtype=float)
    states = np.array(
        [np.concatenate((node.q_rad, [node.insertion_m])) for node in result.nodes]
    )
    np.savez_compressed(
        npz_path,
        s_m=s,
        state_q_L=states,
        seed_state_q_L=np.array([node.seed_state for node in result.nodes]),
        desired_position_m=np.array([node.desired_position_m for node in result.nodes]),
        achieved_position_m=np.array([node.achieved_position_m for node in result.nodes]),
        desired_tangent=np.array([node.desired_tangent for node in result.nodes]),
        achieved_tangent=np.array([node.achieved_tangent for node in result.nodes]),
        position_error_m=np.array([node.position_error_m for node in result.nodes]),
        tangent_error_rad=np.array([node.tangent_error_rad for node in result.nodes]),
        feasible=np.array([node.feasible for node in result.nodes], dtype=bool),
        maximum_first_difference=result.maximum_first_difference,
        maximum_second_difference=result.maximum_second_difference,
    )
    try:
        _save_plot(result, plot_path)
    except Exception as exc:
        result.summary["plot_error"] = f"{type(exc).__name__}: {exc}"
        json_path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    files = {
        "csv": str(csv_path),
        "json": str(json_path),
        "npz": str(npz_path),
        "plot": str(plot_path),
        "dense_csv": str(dense_path),
    }
    result.output_files.update(files)
    return files


def _save_plot(result: GlobalConfigurationPathResult, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes = result.nodes
    s = np.array([node.s_m for node in nodes])
    desired = np.array([node.desired_position_m for node in nodes])
    achieved = np.array([node.achieved_position_m for node in nodes])
    seed_states = np.array([node.seed_state for node in nodes])
    optimized_states = np.array(
        [np.concatenate((node.q_rad, [node.insertion_m])) for node in nodes]
    )
    position_error = 1.0e3 * np.array([node.position_error_m for node in nodes])
    tangent_error = np.degrees(np.array([node.tangent_error_rad for node in nodes]))
    condition = np.array([node.jacobian_condition for node in nodes])
    contact = np.array([node.contact_active for node in nodes], dtype=float)

    figure = plt.figure(figsize=(16, 20), constrained_layout=True)
    grid = figure.add_gridspec(5, 2)
    axis_path = figure.add_subplot(grid[0, 0], projection="3d")
    axis_path.plot(*desired.T, "k--", label="desired")
    axis_path.plot(*achieved.T, color="tab:blue", label="global optimized")
    axis_path.set_title("Beam-tip path")
    axis_path.set_xlabel("x [m]")
    axis_path.set_ylabel("y [m]")
    axis_path.set_zlabel("z [m]")
    axis_path.legend()

    axis_errors = figure.add_subplot(grid[0, 1])
    axis_errors.plot(s, position_error, label="position [mm]")
    axis_errors.axhline(
        1.0e3 * result.configuration["position_tolerance_m"], linestyle="--"
    )
    axis_errors_t = axis_errors.twinx()
    axis_errors_t.plot(s, tangent_error, color="tab:orange", label="tangent [deg]")
    axis_errors_t.axhline(
        np.degrees(result.configuration["tangent_tolerance_rad"]),
        color="tab:orange",
        linestyle="--",
    )
    axis_errors.set_xlabel("s [m]")
    axis_errors.set_ylabel("position error [mm]")
    axis_errors_t.set_ylabel("tangent error [deg]")
    axis_errors.set_title("Hard-task validation")

    axis_joints = figure.add_subplot(grid[1, :])
    for joint in range(6):
        axis_joints.plot(s, seed_states[:, joint], "--", alpha=0.35)
        axis_joints.plot(s, optimized_states[:, joint], label=f"q{joint + 1}")
    axis_joints.set_xlabel("s [m]")
    axis_joints.set_ylabel("joint angle [rad]")
    axis_joints.set_title("Seed (faint dashed) and globally optimized joints")
    axis_joints.legend(ncol=3)

    axis_insertion = figure.add_subplot(grid[2, 0])
    axis_insertion.plot(s, 1.0e3 * seed_states[:, 6], "--", label="seed")
    axis_insertion.plot(s, 1.0e3 * optimized_states[:, 6], label="optimized")
    axis_insertion.set_xlabel("s [m]")
    axis_insertion.set_ylabel("insertion [mm]")
    axis_insertion.legend()

    scale = np.asarray(result.configuration["configuration_scale"], dtype=float)
    ds = np.diff(s)
    first = np.diff(optimized_states, axis=0) / ds[:, None] / scale[None, :]
    axis_first = figure.add_subplot(grid[2, 1])
    axis_first.plot(s[:-1], np.linalg.norm(first, axis=1))
    axis_first.set_xlabel("s [m]")
    axis_first.set_ylabel("scaled first-difference norm")

    axis_second = figure.add_subplot(grid[3, 0])
    if first.shape[0] >= 2:
        second = 2.0 * np.diff(first, axis=0) / (ds[:-1] + ds[1:])[:, None]
        axis_second.plot(s[1:-1], np.linalg.norm(second, axis=1))
    axis_second.set_xlabel("s [m]")
    axis_second.set_ylabel("scaled second-difference norm")

    axis_condition = figure.add_subplot(grid[3, 1])
    axis_condition.semilogy(s, np.where(np.isfinite(condition), condition, np.nan))
    axis_condition.set_xlabel("s [m]")
    axis_condition.set_ylabel("effective condition number")

    axis_contact = figure.add_subplot(grid[4, 0])
    axis_contact.step(s, contact, where="post")
    axis_contact.set_yticks([0, 1], labels=["inactive", "active"])
    axis_contact.set_xlabel("s [m]")
    axis_contact.set_title("Contact diagnostic")

    axis_dense = figure.add_subplot(grid[4, 1])
    if result.dense_samples:
        spend_position_tolerance, spend_tangent_tolerance = effective_task_tolerances(
            position_tolerance_m=result.configuration["position_tolerance_m"],
            tangent_tolerance_rad=result.configuration["tangent_tolerance_rad"],
            spend_fraction=result.configuration["tolerance_spend_fraction"],
        )
        dense_s = np.array([sample.s_m for sample in result.dense_samples])
        dense_score = np.array(
            [
                max(
                    sample.position_error_m
                    / spend_position_tolerance,
                    sample.tangent_error_rad
                    / spend_tangent_tolerance,
                    (
                        1.0
                        if not sample.source_magnet_lumen_constraint_satisfied
                        else 0.0
                    ),
                )
                for sample in result.dense_samples
            ]
        )
        axis_dense.plot(dense_s, dense_score)
        axis_dense.axhline(1.0, color="tab:red", linestyle="--")
    axis_dense.set_xlabel("s [m]")
    axis_dense.set_ylabel("maximum normalized dense error")
    axis_dense.set_title("Between-node nonlinear validation")

    figure.suptitle("Global constrained configuration path", fontsize=16)
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)


class _LinearMockAdapter:
    n_out = 6
    model = None

    def forward_output(self, state: Any, commit: bool = False) -> Array:
        del commit
        state = _finite_vector(state, 7, "mock state")
        raw_tangent = np.array([1.0, state[2], state[3]], dtype=float)
        tangent = raw_tangent / np.linalg.norm(raw_tangent)
        return np.concatenate(([state[0], state[1], state[6]], tangent))

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
        J[3:6] = _normalized_tangent_jacobian(raw_tangent, raw_J)
        return J

    def magnet_transform(self, state: Any) -> Array:
        state = _finite_vector(state, 7, "mock state")
        T = np.eye(4)
        T[:3, 3] = [state[0], state[1], state[6]]
        return T


def _make_mock_inverse_result(partial: bool) -> tuple[InversePathResult, Array, dict]:
    centreline = np.column_stack(
        (np.linspace(0.0, 0.02, 21), np.zeros(21), np.zeros(21))
    )
    end = 0.01 if partial else 0.02
    s = np.linspace(0.0, end, int(round(end / 0.005)) + 1)
    nodes: list[InverseNodeResult] = []
    for index, s_value in enumerate(s):
        state = np.zeros(7)
        state[0] = s_value
        nodes.append(
            InverseNodeResult(
                node_index=index,
                s_m=float(s_value),
                q_rad=state[:6],
                insertion_m=0.0,
                magnet_pose6=np.zeros(6),
                tip_position_m=np.array([s_value, 0.0, 0.0]),
                tip_tangent=np.array([1.0, 0.0, 0.0]),
                desired_position_m=np.array([s_value, 0.0, 0.0]),
                desired_tangent=np.array([1.0, 0.0, 0.0]),
                position_error_m=0.0,
                tangent_error_rad=0.0,
                feasible=True,
                solver_success=True,
                termination_reason="mock",
                function_evaluations=1,
                objective_value=0.0,
                jacobian_condition=1.0,
                jacobian_effective_rank=5,
                minimum_joint_margin_rad=1.0,
                minimum_state_margin=0.1,
                minimum_clearance_m=np.nan,
                contact_active=False,
                source_magnet_lumen_distance_m=np.nan,
                source_magnet_lumen_exclusion_radius_m=np.nan,
                source_magnet_lumen_margin_m=np.nan,
                source_magnet_lumen_segment_index=-1,
                source_magnet_lumen_constraint_satisfied=True,
                delta_q_norm_rad=0.0 if index == 0 else 0.005,
                delta_insertion_m=0.0,
                attempt_count=1,
            )
        )
    result = InversePathResult(
        nodes=nodes,
        all_nodes_feasible=not partial,
        last_feasible_index=len(nodes) - 1,
        last_feasible_progress_m=float(s[-1]),
        configuration={
            "position_tolerance_m": 1.0e-5,
            "tangent_tolerance_rad": 1.0e-3,
        },
        summary={"all_nodes_feasible": not partial},
    )
    controller_pack = {
        "p0": np.zeros(7),
        "p_min": np.array([-1.0] * 6 + [0.0]),
        "p_max": np.array([1.0] * 6 + [0.1]),
        "plant_diagnostic_joint_adapter": _LinearMockAdapter(),
    }
    return result, centreline, controller_pack


def _finite_difference_gradient(function, x: Array, step: float = 1.0e-7) -> Array:
    gradient = np.empty_like(x)
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = step
        gradient[index] = (function(x + delta) - function(x - delta)) / (2.0 * step)
    return gradient


def _finite_difference_jacobian(function, x: Array, step: float = 1.0e-7) -> Array:
    base = np.asarray(function(x), dtype=float).reshape(-1)
    J = np.empty((base.size, x.size), dtype=float)
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = step
        J[:, index] = (
            np.asarray(function(x + delta)) - np.asarray(function(x - delta))
        ) / (2.0 * step)
    return J


def run_self_test() -> None:
    inverse, centreline, controller_pack = _make_mock_inverse_result(partial=True)
    config = GlobalConfigurationOptimizerConfig(
        mode="recover_partial",
        position_tolerance_m=1.0e-5,
        tangent_tolerance_rad=1.0e-3,
        path_step_m=5.0e-3,
        minimum_path_step_m=1.0e-3,
        dense_validation_samples_per_interval=2,
        maximum_refinement_rounds=1,
        first_difference_weight=(1.0e-8,) * 7,
        second_difference_weight=(1.0e-10,) * 7,
        seed_deviation_weight=(1.0e-10,) * 7,
        joint_centre_weight=(0.0,) * 7,
        maximum_adjacent_joint_change_rad=(0.01,) * 6,
        maximum_adjacent_insertion_change_m=0.01,
        require_contact_model=False,
        maximum_iterations=100,
        feasibility_restoration_iterations=80,
        maximum_multistart_attempts=1,
        partial_extrapolation_gain=1.0,
        optimality_tolerance=1.0e-9,
        constraint_tolerance=1.0e-8,
        debug=False,
    )

    # Feasibility is a category, not a continuously ranked objective.  Once
    # two candidates are within the acceptance tolerance, the lower path
    # objective must win even if the seed has an exactly zero residual.
    fallback = _SolveOutcome(
        x=np.zeros(1),
        success=False,
        status=-1,
        message="test fallback",
        iterations=0,
        objective=100.0,
        constraint_violation=0.0,
        restoration_slack_maximum=0.0,
    )
    feasible_improvement = _SolveOutcome(
        x=np.ones(1),
        success=False,
        status=0,
        message="test feasible improvement",
        iterations=1,
        objective=40.0,
        constraint_violation=0.5 * config.constraint_tolerance,
        restoration_slack_maximum=0.0,
    )
    selected = _select_best_outcome(
        [fallback, feasible_improvement],
        config.constraint_tolerance,
    )
    if selected is not feasible_improvement:
        raise AssertionError(
            "Feasible candidate selection did not prioritize the lower objective."
        )
    infeasible_low_objective = _SolveOutcome(
        x=np.full(1, 2.0),
        success=False,
        status=0,
        message="test infeasible low objective",
        iterations=1,
        objective=1.0,
        constraint_violation=10.0 * config.constraint_tolerance,
        restoration_slack_maximum=0.0,
    )
    selected = _select_best_outcome(
        [fallback, infeasible_low_objective],
        config.constraint_tolerance,
    )
    if selected is not fallback:
        raise AssertionError("An infeasible low-objective candidate was selected.")

    # Exact sparse objective derivative check.
    s = np.array([0.0, 0.005, 0.01])
    seed_states = np.zeros((3, 7))
    seed_states[:, 0] = s
    seed_states[:, 6] = 5.0e-3  # positive insertion: the beam model rejects L <= 0
    objective = _QuadraticPathObjective(
        s=s,
        seed_states=seed_states,
        trusted_seed_mask=np.ones(3, dtype=bool),
        state_min=controller_pack["p_min"],
        state_max=controller_pack["p_max"],
        config=config,
    )
    x = seed_states.reshape(-1) + 1.0e-4
    analytical_gradient = objective.gradient(x)
    numerical_gradient = _finite_difference_gradient(objective.value, x)
    gradient_error = np.linalg.norm(analytical_gradient - numerical_gradient) / max(
        np.linalg.norm(numerical_gradient), 1.0e-12
    )
    if gradient_error > 5.0e-5:
        raise AssertionError(f"Objective gradient error too large: {gradient_error}")

    # Position and normalized-tangent constraint Jacobian check.
    evaluator = _BatchBeamEvaluator(controller_pack["plant_diagnostic_joint_adapter"], 3)
    path = CentrelinePath(centreline)
    tasks = _TaskConstraints(
        evaluator=evaluator,
        desired_position=np.array([path.position(value) for value in s]),
        desired_tangent=np.array([path.tangent(value) for value in s]),
        position_tolerance_m=config.position_tolerance_m,
        tangent_tolerance_rad=config.tangent_tolerance_rad,
    )
    analytical_J = tasks.jacobian(x).toarray()
    numerical_J = _finite_difference_jacobian(tasks.values, x)
    jacobian_error = np.linalg.norm(analytical_J - numerical_J) / max(
        np.linalg.norm(numerical_J), 1.0e-12
    )
    if jacobian_error > 5.0e-5:
        raise AssertionError(f"Constraint Jacobian error too large: {jacobian_error}")

    # tip_curvature_quadratic: exact quadratic against the real evaluator, and
    # its gradient against a brute-force finite-difference of the same
    # second-difference-of-achieved-tip-position quantity it is meant to model.
    curvature_evaluator = _BatchBeamEvaluator(
        controller_pack["plant_diagnostic_joint_adapter"], 3
    )
    curvature_weight = 3.7
    H_curve, f_curve, constant_curve = tip_curvature_quadratic(
        evaluator=curvature_evaluator,
        seed_states=seed_states,
        s=s,
        weight=curvature_weight,
    )

    def _brute_force_tip_curvature(flat_state: Array) -> float:
        states = np.asarray(flat_state, dtype=float).reshape(3, 7)
        left, right = s[1] - s[0], s[2] - s[1]
        common = 2.0 / (left + right)
        p = np.array(
            [curvature_evaluator.output(i, states[i])[:3] for i in range(3)]
        )
        d2 = common * (p[0] / left - p[1] * (1.0 / left + 1.0 / right) + p[2] / right)
        return float(0.5 * curvature_weight * (d2 @ d2))

    seed_flat = seed_states.reshape(-1)
    quad_value = float(
        0.5 * seed_flat @ (H_curve @ seed_flat) + f_curve @ seed_flat + constant_curve
    )
    brute_value = _brute_force_tip_curvature(seed_flat)
    # Linearised at seed_states, so this must be EXACT at the seed itself,
    # regardless of how nonlinear the underlying adapter is away from it.
    if abs(quad_value - brute_value) > 1.0e-9 * max(1.0, abs(brute_value)):
        raise AssertionError(
            f"tip_curvature_quadratic value diverged from brute force at the "
            f"seed: quad={quad_value:.6e} brute={brute_value:.6e}"
        )
    # The gradient check needs the *true* (nonlinear-adapter) curvature at the
    # linearisation point to be non-degenerate, or a near-zero true gradient
    # makes the relative-error metric meaningless. seed_states is q1-linear-
    # in-s and everything else zero, which can be exactly straight in the
    # mock adapter; perturb off it non-uniformly per node/coordinate first.
    curvature_probe = seed_states + np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0e-3, -1.5e-3, 1.0e-3, -5.0e-4, 3.0e-4, -2.0e-4, 1.0e-4],
            [-1.0e-3, 2.5e-3, -5.0e-4, 1.0e-3, -3.0e-4, 4.0e-4, -2.0e-4],
        ]
    )
    H_probe, f_probe, _ = tip_curvature_quadratic(
        evaluator=curvature_evaluator,
        seed_states=curvature_probe,
        s=s,
        weight=curvature_weight,
    )
    probe_flat = curvature_probe.reshape(-1)
    numerical_curvature_gradient = _finite_difference_gradient(
        _brute_force_tip_curvature, probe_flat
    )
    # grad(0.5 x'Hx + f'x) = Hx + f, evaluated at the point it was linearised
    # around (that is where the offset terms inside H/f were taken).
    analytical_curvature_gradient = np.asarray(
        H_probe @ probe_flat + f_probe, dtype=float
    )
    curvature_gradient_error = np.linalg.norm(
        analytical_curvature_gradient - numerical_curvature_gradient
    ) / max(np.linalg.norm(numerical_curvature_gradient), 1.0e-12)
    if curvature_gradient_error > 5.0e-5:
        raise AssertionError(
            f"tip_curvature_quadratic gradient error too large: "
            f"{curvature_gradient_error}"
        )
    if not np.allclose(H_curve.toarray(), H_curve.toarray().T):
        raise AssertionError("tip_curvature_quadratic Hessian is not symmetric.")

    with tempfile.TemporaryDirectory(prefix="global_path_self_test_") as directory:
        inverse_directory = Path(directory) / "inverse"
        save_inverse_path_result(inverse, inverse_directory)
        loaded_inverse = load_inverse_path_result(inverse_directory)
        if len(loaded_inverse.nodes) != len(inverse.nodes):
            raise AssertionError("Saved inverse-result loader changed the node count.")

        result = optimize_from_inverse_result(
            inverse_result=loaded_inverse,
            controller_pack=controller_pack,
            lumen_C=centreline,
            config=config,
            output_dir=Path(directory) / "global",
        )
        expected_files = {
            "global_configuration_path.csv",
            "global_configuration_summary.json",
            "global_configuration_diagnostics.npz",
            "global_configuration_path.png",
            "global_configuration_dense_validation.csv",
        }
        missing = [
            name
            for name in expected_files
            if not (Path(directory) / "global" / name).exists()
        ]
        if missing:
            raise AssertionError(f"Missing self-test output files: {missing}")
    if not result.globally_feasible:
        raise AssertionError(f"Partial-path recovery failed: {result.summary}")
    final_state = np.concatenate((result.nodes[-1].q_rad, [result.nodes[-1].insertion_m]))
    if not np.isclose(final_state[0], 0.02, atol=2.0e-5):
        raise AssertionError(f"Expected terminal q1 near 0.02; got {final_state[0]}")
    if not np.allclose(result.nodes[0].q_rad, 0.0, atol=1.0e-12):
        raise AssertionError("The fixed initial state moved in the self-test.")

    # Complete-path mode must accept and retain an already feasible full seed.
    complete_inverse, _, _ = _make_mock_inverse_result(partial=False)
    complete_config = GlobalConfigurationOptimizerConfig(
        mode="refine_complete",
        position_tolerance_m=1.0e-5,
        tangent_tolerance_rad=1.0e-3,
        path_step_m=5.0e-3,
        minimum_path_step_m=1.0e-3,
        dense_validation_samples_per_interval=1,
        maximum_refinement_rounds=0,
        first_difference_weight=(1.0e-8,) * 7,
        second_difference_weight=(1.0e-10,) * 7,
        seed_deviation_weight=(1.0e-10,) * 7,
        joint_centre_weight=(0.0,) * 7,
        maximum_adjacent_joint_change_rad=(0.01,) * 6,
        maximum_adjacent_insertion_change_m=0.01,
        require_contact_model=False,
        maximum_iterations=80,
        debug=False,
    )
    complete_result = optimize_from_inverse_result(
        inverse_result=complete_inverse,
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=complete_config,
        output_dir=None,
    )
    if not complete_result.globally_feasible:
        raise AssertionError("Complete-path refinement mode failed its known solution.")

    # tip_curvature_weight end-to-end: the same known-feasible problem must
    # still solve through trust-constr with the new term's sparse Hessian and
    # gradient wired in, not just in isolation above.
    curvature_config = replace(complete_config, tip_curvature_weight=1.0e-3)
    curvature_result = optimize_from_inverse_result(
        inverse_result=complete_inverse,
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=curvature_config,
        output_dir=None,
    )
    if not curvature_result.globally_feasible:
        raise AssertionError(
            "Complete-path refinement with tip_curvature_weight > 0 failed."
        )

    # magnet_path_weight / magnet_curvature_weight: exact quadratic vs a
    # brute-force magnet path-length + curvature cost on the mock adapter, and
    # end-to-end through trust-constr.
    magnet_adapter = controller_pack["plant_diagnostic_joint_adapter"]
    magnet_seed = _make_seed_path(
        inverse_result=complete_inverse,
        path=CentrelinePath(centreline),
        state_min=controller_pack["p_min"],
        state_max=controller_pack["p_max"],
        initial_state=np.zeros(7),
        config=complete_config,
    )
    m_pos, m_jac = _magnet_path_terms(
        adapter=magnet_adapter, states=magnet_seed.states, config=complete_config
    )
    from global_upgrades import magnet_path_quadratic as _mpq
    H_m, f_m, c_m = _mpq(
        magnet_positions=m_pos,
        magnet_jacobians=m_jac,
        seed_states=magnet_seed.states,
        s=magnet_seed.s,
        first_weight=3.0,
        second_weight=2.0,
    )

    def _brute_magnet_cost(flat_state: Array) -> float:
        st = np.asarray(flat_state, dtype=float).reshape(magnet_seed.s.size, 7)
        p = m_pos + np.einsum("nij,nj->ni", m_jac, st - magnet_seed.states)
        total = 0.0
        s_nodes = magnet_seed.s
        for j in range(s_nodes.size - 1):
            d1 = (p[j + 1] - p[j]) / (s_nodes[j + 1] - s_nodes[j])
            total += 0.5 * 3.0 * float(d1 @ d1)
        for j in range(1, s_nodes.size - 1):
            left = s_nodes[j] - s_nodes[j - 1]
            right = s_nodes[j + 1] - s_nodes[j]
            common = 2.0 / (left + right)
            d2 = common * (
                p[j - 1] / left
                - p[j] * (1.0 / left + 1.0 / right)
                + p[j + 1] / right
            )
            total += 0.5 * 2.0 * float(d2 @ d2)
        return total

    seed_flat = magnet_seed.states.reshape(-1)
    quad_at_seed = float(
        0.5 * seed_flat @ (H_m @ seed_flat) + f_m @ seed_flat + c_m
    )
    if abs(quad_at_seed - _brute_magnet_cost(seed_flat)) > 1.0e-9 * max(
        1.0, abs(quad_at_seed)
    ):
        raise AssertionError(
            "magnet_path_quadratic value diverged from brute force at the seed."
        )
    probe = seed_flat + 1.0e-4
    grad_numeric = _finite_difference_gradient(_brute_magnet_cost, probe)
    grad_analytic = np.asarray(H_m @ probe + f_m, dtype=float)
    grad_error = np.linalg.norm(grad_analytic - grad_numeric) / max(
        np.linalg.norm(grad_numeric), 1.0e-12
    )
    if grad_error > 5.0e-5:
        raise AssertionError(
            f"magnet_path_quadratic gradient error too large: {grad_error}"
        )
    magnet_config = replace(
        complete_config, magnet_path_weight=1.0e-4, magnet_curvature_weight=1.0e-4
    )
    magnet_result = optimize_from_inverse_result(
        inverse_result=complete_inverse,
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=magnet_config,
        output_dir=None,
    )
    if not magnet_result.globally_feasible:
        raise AssertionError(
            "Complete-path refinement with magnet path/curvature weights failed."
        )

    # Retained-seed fallback under a spent tolerance: the known-feasible seed
    # must never be dropped in favour of an infeasible optimiser iterate just
    # because tolerance_spend_fraction < 1 tightened the constraint the solver
    # drives to.  With a very small iteration budget the solver cannot beat the
    # seed, so the result must either be globally feasible or an explicit seed
    # fallback -- never a silent infeasible failure.
    spent_config = replace(
        complete_config,
        tolerance_spend_fraction=0.5,
        maximum_iterations=1,
        preserve_feasible_seed_on_failure=True,
    )
    spent_result = optimize_from_inverse_result(
        inverse_result=complete_inverse,
        controller_pack=controller_pack,
        lumen_C=centreline,
        config=spent_config,
        output_dir=None,
    )
    if not (
        spent_result.globally_feasible
        or spent_result.summary.get("selected_seed_fallback")
        or spent_result.summary.get("fallback_to_inverse_path")
    ):
        raise AssertionError(
            "A spent-tolerance solve that could not beat the seed returned "
            "neither a feasible path nor a seed fallback: "
            f"{spent_result.summary}"
        )

    # Inverse-path fallback: an adapter whose tip y is stuck 2 mm off and
    # uncontrollable (zero Jacobian row) makes every node infeasible for ANY
    # tolerance the optimiser could reach.  optimize_from_inverse_result must
    # then return the EXACT inverse feasible path unchanged -- never a
    # stalled/diverged optimiser iterate.
    class _StuckTipAdapter(_LinearMockAdapter):
        def forward_output(self, state: Any, commit: bool = False) -> Array:
            out = np.array(super().forward_output(state, commit), dtype=float)
            out[1] = 2.0e-3
            return out

        def continuous_output_jacobian(self, state: Any) -> Array:
            J = np.array(super().continuous_output_jacobian(state), dtype=float)
            J[1, :] = 0.0
            return J

    stuck_pack = dict(controller_pack)
    stuck_pack["plant_diagnostic_joint_adapter"] = _StuckTipAdapter()
    impossible_config = replace(
        complete_config,
        position_tolerance_m=1.0e-4,
        tangent_tolerance_rad=1.0e-2,
        maximum_refinement_rounds=1,
        maximum_iterations=5,
    )
    impossible_result = optimize_from_inverse_result(
        inverse_result=complete_inverse,
        controller_pack=stuck_pack,
        lumen_C=centreline,
        config=impossible_config,
        output_dir=None,
    )
    if not impossible_result.summary.get("fallback_to_inverse_path"):
        raise AssertionError(
            "An unachievable tolerance did not trigger the inverse-path "
            f"fallback: {impossible_result.summary}"
        )
    if impossible_result.summary.get("any_feasible_global_round"):
        raise AssertionError("A feasible round was reported for an impossible tolerance.")
    inverse_states = np.array(
        [
            np.concatenate((node.q_rad, [node.insertion_m]))
            for node in complete_inverse.nodes
            if node.feasible
        ]
    )
    # The fallback floors insertion to _MIN_INSERTION_M (the beam model rejects
    # L <= 0); this mock's inverse insertion is exactly 0, so floor the target.
    inverse_states[:, 6] = np.maximum(inverse_states[:, 6], _MIN_INSERTION_M)
    returned_states = np.array(
        [np.concatenate((node.q_rad, [node.insertion_m])) for node in impossible_result.nodes]
    )
    if returned_states.shape != inverse_states.shape or not np.allclose(
        returned_states, inverse_states, atol=1.0e-9
    ):
        raise AssertionError(
            "The inverse-path fallback did not return the exact inverse feasible "
            f"states: shapes {returned_states.shape} vs {inverse_states.shape}."
        )
    if impossible_result.globally_feasible:
        raise AssertionError("The inverse-path fallback was reported globally feasible.")

    # Verify construction of insertion monotonicity rows.
    monotonic_config = GlobalConfigurationOptimizerConfig(
        position_tolerance_m=1.0e-5,
        tangent_tolerance_rad=1.0e-3,
        insertion_non_decreasing=True,
        require_contact_model=False,
    )
    adjacent = _adjacent_linear_constraint(node_count=3, config=monotonic_config)
    if adjacent is None or not np.all(
        np.asarray(adjacent.lb)[np.arange(6, 14, 7)] == 0.0
    ):
        raise AssertionError("Insertion monotonicity rows were not constructed.")
    print(
        "[SELF TEST] PASS: feasible-first candidate selection, sparse "
        "derivatives, saved-result loading, partial recovery, complete "
        "refinement, hard constraints, dense validation, monotonicity, and "
        "serialization."
    )


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Global constrained magnetic-beam configuration-path optimizer. "
            "Repository integrations call optimize_from_inverse_result() or "
            "optimize_from_saved_inverse_result()."
        )
    )
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            run_self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "DenseValidationSample",
    "GlobalConfigurationNodeResult",
    "GlobalConfigurationOptimizerConfig",
    "GlobalConfigurationPathResult",
    "audit_inverse_seed_round_trip",
    "load_inverse_path_result",
    "optimize_from_inverse_result",
    "optimize_from_saved_inverse_result",
    "save_global_configuration_result",
]
