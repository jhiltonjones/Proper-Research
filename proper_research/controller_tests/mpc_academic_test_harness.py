"""
Academic verification harness for MPCControllerTipXY.

The primary benchmark is a constrained 2-D double integrator:

    state   x = [px, py, vx, vy]
    input   u = [ax, ay]
    model   x[k+1] = A x[k] + B u[k]

Why this model?
    It is a standard linear MPC benchmark with an exact discrete-time model.
    It is simple enough to derive independently, yet it tests multi-step
    prediction, position/velocity coupling, reference preview, input bounds,
    input-rate penalties, and receding-horizon feedback.

The file has two layers:

1. Mathematical self-tests. These do not import the user's controller. They
   verify the plant, condensed prediction matrices, quadratic cost gradient,
   independent optimizer, and feasibility of the nominal reference.
2. Controller integration tests. These adapt MPCControllerTipXY's 8-parameter,
   7-input interface to the academic 4-state, 2-input plant and compare the
   controller against an independently assembled condensed MPC oracle.

Examples:
    # Run the mathematical tests only:
    python mpc_academic_test_harness.py --self-test-only

    # Run the mathematical and controller tests:
    python mpc_academic_test_harness.py

    # Use a different import path:
    python mpc_academic_test_harness.py \
        --controller my_package.my_controller:MPCControllerTipXY

    # Require the controller to be importable and create plots:
    python mpc_academic_test_harness.py --require-controller --plot

Dependencies:
    numpy, scipy
    matplotlib is required only with --plot
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessConfig:
    """Numerical choices shared by the oracle and the controller under test."""

    dt: float = 0.05
    horizon: int = 20

    # State order: px, py, vx, vy.
    q: tuple[float, float, float, float] = tuple(x*1 for x in (100.0, 100.0, 20.0, 20.0))


    # Active input order: ax, ay.
    r: tuple[float, float] = tuple(x*1 for x in (0.1, 0.1))
    rd: tuple[float, float] = (0.1, 0.1)

    u_max: float = 0.05
    solver_mode: str = "lti"

    # The project controller solves a 7-input problem whereas the oracle solves
    # the equivalent 2-input problem. Small solver/regularization differences
    # are allowed.
    first_move_tolerance: float = 2.0e-4
    sequence_tolerance: float = 5.0e-4
    inactive_input_tolerance: float = 1.0e-6
    bound_tolerance: float = 1.0e-8
    plant_step_tolerance: float = 1.0e-12


@dataclass(frozen=True)
class Scenario:
    """One closed-loop experiment."""

    name: str
    purpose: str
    steps: int
    x0: tuple[float, float, float, float]
    reference_kind: str
    target_xy: tuple[float, float] = (0.05, -0.03)
    start_time: float = 0.5
    duration: float = 6.0
    expect_saturation: bool = False


@dataclass
class Check:
    """One human-readable test result."""

    name: str
    status: str
    detail: str
    measured: Optional[float] = None
    limit: Optional[float] = None

    @classmethod
    def passed(
        cls,
        name: str,
        detail: str,
        measured: Optional[float] = None,
        limit: Optional[float] = None,
    ) -> "Check":
        return cls(name, "PASS", detail, measured, limit)

    @classmethod
    def failed(
        cls,
        name: str,
        detail: str,
        measured: Optional[float] = None,
        limit: Optional[float] = None,
    ) -> "Check":
        return cls(name, "FAIL", detail, measured, limit)

    @classmethod
    def skipped(cls, name: str, detail: str) -> "Check":
        return cls(name, "SKIP", detail)


@dataclass
class OracleSolution:
    """Solution of the independently assembled 2-input MPC problem."""

    U: Array
    X: Array
    objective: float
    success: bool
    message: str
    iterations: int


@dataclass
class ClosedLoopRun:
    """Numerical data recorded from one controller scenario."""

    scenario: Scenario
    time: Array
    reference: Array
    states: Array
    controls: Array
    oracle_controls: Array
    first_move_errors: Array
    sequence_errors: Array
    plant_step_errors: Array
    infeasible_flags: Array
    statuses: list[str]
    reference_index_mismatches: int
    solve_times_ms: Array
    input_limit: float

    def summary(self) -> dict[str, Any]:
        position_error = self.states[:, :2] - self.reference[:, :2]
        position_norm = np.linalg.norm(position_error, axis=1)
        active_controls = self.controls[:, :2]
        inactive_controls = self.controls[:, 2:]
        finite_sequence_errors = self.sequence_errors[
            np.isfinite(self.sequence_errors)
        ]

        return {
            "scenario": self.scenario.name,
            "rms_position_error": float(
                np.sqrt(np.mean(np.square(position_norm)))
            ),
            "max_position_error": float(np.max(position_norm)),
            "final_position_error": float(position_norm[-1]),
            "max_first_move_oracle_error": float(
                np.max(self.first_move_errors)
            ),
            "max_sequence_oracle_error": (
                float(np.max(finite_sequence_errors))
                if finite_sequence_errors.size
                else None
            ),
            "max_input_magnitude": float(np.max(np.abs(active_controls))),
            "max_inactive_input_magnitude": (
                float(np.max(np.abs(inactive_controls)))
                if inactive_controls.size
                else 0.0
            ),
            "saturation_fraction": float(
                np.mean(
                    np.abs(active_controls)
                    >= self.input_limit - 1.0e-8
                )
            ),
            "max_plant_step_error": float(np.max(self.plant_step_errors)),
            "infeasible_count": int(np.sum(self.infeasible_flags)),
            "reference_index_mismatches": self.reference_index_mismatches,
            "mean_solve_time_ms": float(np.mean(self.solve_times_ms)),
            "median_solve_time_ms": float(np.median(self.solve_times_ms)),
            "p95_solve_time_ms": float(
                np.percentile(self.solve_times_ms, 95)
            ),
        }


@dataclass
class HarnessReport:
    """Serializable report written after a harness run."""

    config: dict[str, Any]
    controller_spec: Optional[str]
    controller_loaded: bool
    controller_message: str
    checks: list[Check] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "controller_spec": self.controller_spec,
            "controller_loaded": self.controller_loaded,
            "controller_message": self.controller_message,
            "checks": [asdict(check) for check in self.checks],
            "scenarios": self.scenarios,
        }


# ---------------------------------------------------------------------------
# Exact academic plant and predictions
# ---------------------------------------------------------------------------


def double_integrator_matrices(dt: float) -> tuple[Array, Array]:
    """
    Return the exact zero-order-hold discrete model of a 2-D double integrator.

    Continuous equations:
        d(px)/dt = vx
        d(py)/dt = vy
        d(vx)/dt = ax
        d(vy)/dt = ay
    """

    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    return A, B


def double_integrator_step(x: Array, u: Array, dt: float) -> Array:
    """Advance the plant by one sample using the scalar kinematic equations."""

    px, py, vx, vy = np.asarray(x, dtype=float).reshape(4)
    ax, ay = np.asarray(u, dtype=float).reshape(2)
    return np.array(
        [
            px + dt * vx + 0.5 * dt**2 * ax,
            py + dt * vy + 0.5 * dt**2 * ay,
            vx + dt * ax,
            vy + dt * ay,
        ],
        dtype=float,
    )


def rollout_double_integrator(x0: Array, U: Array, dt: float) -> Array:
    """
    Apply every row of U and return [x1, ..., xN].

    The initial state x0 is deliberately not included, matching the usual MPC
    convention that stage 0 of the prediction is x[k+1].
    """

    x = np.asarray(x0, dtype=float).reshape(4).copy()
    U = np.asarray(U, dtype=float).reshape(-1, 2)
    X = np.empty((len(U), 4), dtype=float)
    for k, u in enumerate(U):
        x = double_integrator_step(x, u, dt)
        X[k] = x
    return X


def condensed_prediction_matrices(
    A: Array,
    B: Array,
    horizon: int,
) -> tuple[Array, Array]:
    """
    Build Sx and Su such that X = Sx*x0 + Su*U.

    X stacks x1 ... xN and U stacks u0 ... u(N-1). Su is block lower
    triangular because a future input cannot affect an earlier state.
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(f"B has incompatible shape {B.shape}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    n = A.shape[0]
    m = B.shape[1]
    Sx = np.zeros((horizon * n, n), dtype=float)
    Su = np.zeros((horizon * n, horizon * m), dtype=float)

    for prediction_step in range(horizon):
        rows = slice(prediction_step * n, (prediction_step + 1) * n)
        Sx[rows] = np.linalg.matrix_power(A, prediction_step + 1)

        for input_step in range(prediction_step + 1):
            columns = slice(input_step * m, (input_step + 1) * m)
            Su[rows, columns] = (
                np.linalg.matrix_power(A, prediction_step - input_step) @ B
            )

    return Sx, Su


def input_difference_matrices(
    horizon: int,
    input_dimension: int,
) -> tuple[Array, Array]:
    """
    Build D and E so DU - E*u_previous stacks input increments.

    Result:
        DU - E*u_previous =
            [u0-u_previous, u1-u0, ..., u(N-1)-u(N-2)]
    """

    if horizon < 1 or input_dimension < 1:
        raise ValueError("horizon and input_dimension must be positive")

    D = np.zeros(
        (horizon * input_dimension, horizon * input_dimension),
        dtype=float,
    )
    for k in range(horizon):
        rows = slice(k * input_dimension, (k + 1) * input_dimension)
        D[rows, rows] = np.eye(input_dimension)
        if k > 0:
            previous = slice(
                (k - 1) * input_dimension,
                k * input_dimension,
            )
            D[rows, previous] = -np.eye(input_dimension)

    E = np.zeros((horizon * input_dimension, input_dimension), dtype=float)
    E[:input_dimension] = np.eye(input_dimension)
    return D, E


def _weight_matrix(weights: Sequence[float], dimension: int, name: str) -> Array:
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if weights_array.size != dimension:
        raise ValueError(
            f"{name} must have {dimension} entries, got {weights_array.size}"
        )
    if np.any(weights_array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return np.diag(weights_array)


def condensed_tracking_cost(
    *,
    A: Array,
    B: Array,
    horizon: int,
    x0: Array,
    reference_window: Array,
    u_previous: Array,
    q: Sequence[float],
    r: Sequence[float],
    rd: Sequence[float],
) -> tuple[Array, Array, Callable[[Array], float], Array, Array]:
    """
    Condense the finite-horizon tracking cost into 0.5*U'HU + f'U.

    The uncondensed cost is

        sum_i (x_i-r_i)'Q(x_i-r_i)
              + u_i'R u_i
              + du_i'Rd du_i.

    Returning an evaluator for the uncondensed cost lets the gradient be
    checked independently with finite differences.
    """

    n = A.shape[0]
    m = B.shape[1]
    x0 = np.asarray(x0, dtype=float).reshape(n)
    reference_window = np.asarray(reference_window, dtype=float).reshape(
        horizon,
        n,
    )
    u_previous = np.asarray(u_previous, dtype=float).reshape(m)

    Q = _weight_matrix(q, n, "q")
    R = _weight_matrix(r, m, "r")
    Rd = _weight_matrix(rd, m, "rd")

    Sx, Su = condensed_prediction_matrices(A, B, horizon)
    D, E = input_difference_matrices(horizon, m)

    Qbar = np.kron(np.eye(horizon), Q)
    Rbar = np.kron(np.eye(horizon), R)
    Rdbar = np.kron(np.eye(horizon), Rd)

    reference_vector = reference_window.reshape(-1)
    free_tracking_error = Sx @ x0 - reference_vector

    H = 2.0 * (
        Su.T @ Qbar @ Su
        + Rbar
        + D.T @ Rdbar @ D
    )
    f = 2.0 * (
        Su.T @ Qbar @ free_tracking_error
        - D.T @ Rdbar @ E @ u_previous
    )

    # Numerical solvers benefit from exact symmetry after floating-point
    # matrix products.
    H = 0.5 * (H + H.T)

    def evaluate_uncondensed(U_vector: Array) -> float:
        U_vector = np.asarray(U_vector, dtype=float).reshape(horizon * m)
        X_vector = Sx @ x0 + Su @ U_vector
        tracking_error = X_vector - reference_vector
        increments = D @ U_vector - E @ u_previous
        return float(
            tracking_error @ Qbar @ tracking_error
            + U_vector @ Rbar @ U_vector
            + increments @ Rdbar @ increments
        )

    return H, f, evaluate_uncondensed, Sx, Su


def solve_condensed_oracle(
    *,
    config: HarnessConfig,
    x0: Array,
    reference_window: Array,
    u_previous: Array,
) -> OracleSolution:
    """
    Solve the independent, bound-constrained double-integrator MPC problem.

    SciPy's L-BFGS-B is intentionally separate from the project controller's
    QP construction and solver. Agreement therefore checks more than merely
    calling the same implementation twice.
    """

    A, B = double_integrator_matrices(config.dt)
    H, f, evaluate_cost, Sx, Su = condensed_tracking_cost(
        A=A,
        B=B,
        horizon=config.horizon,
        x0=x0,
        reference_window=reference_window,
        u_previous=u_previous,
        q=config.q,
        r=config.r,
        rd=config.rd,
    )

    number_of_variables = config.horizon * B.shape[1]
    initial_guess = np.tile(
        np.asarray(u_previous, dtype=float).reshape(2),
        config.horizon,
    )

    def condensed_objective(U_vector: Array) -> float:
        return float(0.5 * U_vector @ H @ U_vector + f @ U_vector)

    def condensed_gradient(U_vector: Array) -> Array:
        return H @ U_vector + f

    result = minimize(
        condensed_objective,
        initial_guess,
        jac=condensed_gradient,
        method="L-BFGS-B",
        bounds=[
            (-config.u_max, config.u_max)
            for _ in range(number_of_variables)
        ],
        options={
            "ftol": 1.0e-13,
            "gtol": 1.0e-10,
            "maxiter": 500,
            "maxls": 50,
        },
    )

    U = np.asarray(result.x, dtype=float).reshape(config.horizon, 2)
    X = (Sx @ np.asarray(x0) + Su @ U.reshape(-1)).reshape(
        config.horizon,
        4,
    )
    return OracleSolution(
        U=U,
        X=X,
        objective=evaluate_cost(U.reshape(-1)),
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
    )


# ---------------------------------------------------------------------------
# Reference trajectories and academic scenarios
# ---------------------------------------------------------------------------


def quintic_point_to_point_reference(
    *,
    number_of_samples: int,
    dt: float,
    target_xy: Sequence[float],
    start_time: float,
    duration: float,
) -> tuple[Array, Array, Array]:
    """
    Create a C2 point-to-point reference for [px, py, vx, vy].

    h(s) = 10s^3 - 15s^4 + 6s^5

    Position, velocity, and acceleration are all zero at the beginning and end
    of the transition. That makes feasibility easy to evaluate for a double
    integrator, because the required input is exactly the reference
    acceleration.
    """

    if number_of_samples < 1:
        raise ValueError("number_of_samples must be positive")
    if dt <= 0.0 or duration <= 0.0 or start_time < 0.0:
        raise ValueError("dt and duration must be positive; start_time >= 0")

    target = np.asarray(target_xy, dtype=float).reshape(2)
    time_axis = np.arange(number_of_samples, dtype=float) * dt
    s = np.clip((time_axis - start_time) / duration, 0.0, 1.0)

    h = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    h_dot = (
        30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    ) / duration
    h_ddot = (
        60.0 * s - 180.0 * s**2 + 120.0 * s**3
    ) / duration**2

    outside = (time_axis < start_time) | (
        time_axis > start_time + duration
    )
    h_dot[outside] = 0.0
    h_ddot[outside] = 0.0

    reference = np.zeros((number_of_samples, 4), dtype=float)
    reference[:, :2] = h[:, None] * target[None, :]
    reference[:, 2:] = h_dot[:, None] * target[None, :]
    reference_acceleration = h_ddot[:, None] * target[None, :]
    return time_axis, reference, reference_acceleration


def reference_window(reference: Array, k: int, horizon: int) -> Array:
    """Return r[k+1] ... r[k+horizon], padding with the final reference."""

    reference = np.asarray(reference, dtype=float)
    indices = np.arange(k + 1, k + 1 + horizon, dtype=int)
    indices = np.clip(indices, 0, len(reference) - 1)
    return reference[indices]


def default_scenarios() -> list[Scenario]:
    """
    Return three tests with deliberately different purposes.

    Regulation tests stabilization, feasible tracking tests preview without
    unavoidable saturation, and deliberate saturation tests constraint
    enforcement rather than tracking quality.
    """

    return [
        Scenario(
            name="regulation",
            purpose=(
                "Drive a nonzero position/velocity state to the origin. "
                "Tests signs, state ordering, feedback, and input-rate memory."
            ),
            steps=160,
            x0=(0.05, -0.03, 0.015, -0.01),
            reference_kind="zero",
        ),
        Scenario(
            name="feasible_quintic_tracking",
            purpose=(
                "Track a smooth, dynamically feasible point-to-point motion. "
                "Tests reference stacking, preview, and position/velocity "
                "coupling without confusing failure with actuator saturation."
            ),
            steps=180,
            x0=(0.0, 0.0, 0.0, 0.0),
            reference_kind="quintic",
            target_xy=(0.05, -0.03),
            start_time=0.5,
            duration=3.0,
        ),
        Scenario(
            name="deliberate_saturation",
            purpose=(
                "Command an intentionally infeasible fast movement. The "
                "correct behavior is to respect bounds and saturate; small "
                "tracking error is not a pass condition."
            ),
            steps=100,
            x0=(0.0, 0.0, 0.0, 0.0),
            reference_kind="quintic",
            target_xy=(0.05, -0.03),
            start_time=0.5,
            duration=0.25,
            expect_saturation=True,
        ),
    ]


def build_scenario_reference(
    scenario: Scenario,
    config: HarnessConfig,
) -> tuple[Array, Array, Array]:
    """Return time, state reference, and feedforward acceleration."""

    total_samples = scenario.steps + config.horizon + 1
    if scenario.reference_kind == "zero":
        return (
            np.arange(total_samples, dtype=float) * config.dt,
            np.zeros((total_samples, 4), dtype=float),
            np.zeros((total_samples, 2), dtype=float),
        )
    if scenario.reference_kind == "quintic":
        return quintic_point_to_point_reference(
            number_of_samples=total_samples,
            dt=config.dt,
            target_xy=scenario.target_xy,
            start_time=scenario.start_time,
            duration=scenario.duration,
        )
    raise ValueError(
        f"Unknown reference kind {scenario.reference_kind!r}"
    )


# ---------------------------------------------------------------------------
# Mathematical self-tests: independent of MPCControllerTipXY
# ---------------------------------------------------------------------------


def _finite_difference_gradient(
    objective: Callable[[Array], float],
    point: Array,
    epsilon: float = 1.0e-6,
) -> Array:
    point = np.asarray(point, dtype=float).reshape(-1)
    gradient = np.empty_like(point)
    for i in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[i] = epsilon
        gradient[i] = (
            objective(point + perturbation)
            - objective(point - perturbation)
        ) / (2.0 * epsilon)
    return gradient


def run_mathematical_self_tests(config: HarnessConfig) -> list[Check]:
    """Verify the benchmark and oracle before using either to judge the MPC."""

    checks: list[Check] = []
    rng = np.random.default_rng(20260729)
    A, B = double_integrator_matrices(config.dt)

    # Test 1: the matrix model and scalar kinematic equations must agree.
    x = rng.normal(size=4)
    u = rng.normal(size=2)
    matrix_next = A @ x + B @ u
    scalar_next = double_integrator_step(x, u, config.dt)
    error = float(np.max(np.abs(matrix_next - scalar_next)))
    checks.append(
        Check.passed(
            "plant equation",
            "A@x+B@u matches the independently written kinematic step.",
            error,
            1.0e-14,
        )
        if error <= 1.0e-14
        else Check.failed(
            "plant equation",
            "The matrix and scalar plant implementations disagree.",
            error,
            1.0e-14,
        )
    )

    # Test 2: condensed prediction must equal repeated simulation.
    worst_prediction_error = 0.0
    for horizon in (1, 2, 5, config.horizon):
        Sx, Su = condensed_prediction_matrices(A, B, horizon)
        x0 = rng.normal(size=4)
        U = rng.normal(size=(horizon, 2))
        condensed = (Sx @ x0 + Su @ U.reshape(-1)).reshape(horizon, 4)
        rolled_out = rollout_double_integrator(x0, U, config.dt)
        worst_prediction_error = max(
            worst_prediction_error,
            float(np.max(np.abs(condensed - rolled_out))),
        )
    checks.append(
        Check.passed(
            "condensed prediction",
            "Sx and Su reproduce repeated plant rollout.",
            worst_prediction_error,
            1.0e-12,
        )
        if worst_prediction_error <= 1.0e-12
        else Check.failed(
            "condensed prediction",
            "Sx/Su indexing or powers of A are incorrect.",
            worst_prediction_error,
            1.0e-12,
        )
    )

    # Test 3: analytic QP gradient must match finite differences of the
    # original, uncondensed cost.
    test_horizon = min(config.horizon, 6)
    _, reference, _ = quintic_point_to_point_reference(
        number_of_samples=test_horizon + 1,
        dt=config.dt,
        target_xy=(0.05, -0.03),
        start_time=0.0,
        duration=max(config.dt * test_horizon, config.dt),
    )
    x0 = rng.normal(scale=0.05, size=4)
    u_previous = rng.normal(scale=0.01, size=2)
    H, f, original_cost, _, _ = condensed_tracking_cost(
        A=A,
        B=B,
        horizon=test_horizon,
        x0=x0,
        reference_window=reference[1 : test_horizon + 1],
        u_previous=u_previous,
        q=config.q,
        r=config.r,
        rd=config.rd,
    )
    point = rng.normal(scale=0.01, size=test_horizon * 2)
    analytic_gradient = H @ point + f
    numerical_gradient = _finite_difference_gradient(original_cost, point)
    gradient_error = float(
        np.max(np.abs(analytic_gradient - numerical_gradient))
    )
    checks.append(
        Check.passed(
            "quadratic cost gradient",
            "H@U+f matches a finite-difference gradient of the original cost.",
            gradient_error,
            2.0e-7,
        )
        if gradient_error <= 2.0e-7
        else Check.failed(
            "quadratic cost gradient",
            "The condensed Hessian/linear term does not represent the stated cost.",
            gradient_error,
            2.0e-7,
        )
    )

    # Test 4: zero state + zero reference must have the zero sequence as its
    # optimum.
    zero_reference = np.zeros((config.horizon, 4), dtype=float)
    zero_solution = solve_condensed_oracle(
        config=config,
        x0=np.zeros(4),
        reference_window=zero_reference,
        u_previous=np.zeros(2),
    )
    zero_solution_error = float(np.max(np.abs(zero_solution.U)))
    zero_ok = zero_solution.success and zero_solution_error <= 1.0e-10
    checks.append(
        Check.passed(
            "oracle zero equilibrium",
            "The independent optimizer leaves the zero equilibrium unchanged.",
            zero_solution_error,
            1.0e-10,
        )
        if zero_ok
        else Check.failed(
            "oracle zero equilibrium",
            f"Oracle failure or nonzero optimum: {zero_solution.message}",
            zero_solution_error,
            1.0e-10,
        )
    )

    # Test 5: the nominal tracking reference must be dynamically feasible.
    feasible_scenario = next(
        scenario
        for scenario in default_scenarios()
        if scenario.name == "feasible_quintic_tracking"
    )
    _, _, required_acceleration = build_scenario_reference(
        feasible_scenario,
        config,
    )
    peak_required_acceleration = float(
        np.max(np.abs(required_acceleration))
    )
    checks.append(
        Check.passed(
            "nominal reference feasibility",
            (
                "Peak reference acceleration is within the input bound, so "
                "tracking error is not guaranteed by an impossible command."
            ),
            peak_required_acceleration,
            config.u_max,
        )
        if peak_required_acceleration <= config.u_max
        else Check.failed(
            "nominal reference feasibility",
            "The nominal tracking reference exceeds the acceleration bound.",
            peak_required_acceleration,
            config.u_max,
        )
    )

    return checks


# ---------------------------------------------------------------------------
# Adapter for MPCControllerTipXY
# ---------------------------------------------------------------------------


class DoubleIntegratorForwardModel:
    """Map the controller's temporary 8-D parameter vector into academic state."""

    def start_step(self) -> None:
        """Compatibility hook used by the project controller."""

    def __call__(self, p: Array, commit: bool = False) -> Array:
        del commit
        p = np.asarray(p, dtype=float).reshape(8)
        return np.array([p[0], p[1], p[2], p[7]], dtype=float)


class DoubleIntegratorInputJacobian:
    """
    Return the controller's 4-by-7 input matrix.

    Only controller inputs 0 and 1 are active. The five null columns are an
    intentional test: a correctly penalized optimizer should leave them zero.
    """

    def __init__(self, dt: float):
        _, active_B = double_integrator_matrices(dt)
        self.B = np.zeros((4, 7), dtype=float)
        self.B[:, :2] = active_B

    def __call__(self, p: Array) -> Array:
        np.asarray(p, dtype=float).reshape(8)
        return self.B.copy()

    def get_last_diag(self) -> dict[str, Any]:
        return {"benchmark": "double_integrator_full_state"}


def pose_from_academic_state(x: Array) -> Array:
    """
    Embed [px,py,vx,vy] into the project's temporary 8-D parameter layout.

    p[3]=1 preserves the identity orientation expected by the robot code.
    """

    px, py, vx, vy = np.asarray(x, dtype=float).reshape(4)
    p = np.zeros(8, dtype=float)
    p[0] = px
    p[1] = py
    p[2] = vx
    p[3] = 1.0
    p[7] = vy
    return p


def patch_double_integrator_plant(controller: Any, dt: float) -> None:
    """Replace only this controller instance's plant rollout."""

    def apply_control_sequence(
        self: Any,
        U_sequence: Array,
        number_to_apply: int,
    ) -> dict[str, Any]:
        U_sequence = np.asarray(U_sequence, dtype=float).reshape(
            self.Np,
            self.m,
        )
        number_to_apply = int(np.clip(number_to_apply, 1, self.Np))
        applied: list[Array] = []

        for j in range(number_to_apply):
            u = U_sequence[j].copy()
            applied.append(u)
            self.x = double_integrator_step(self.x, u[:2], dt)
            self.p = pose_from_academic_state(self.x)

        return {
            "benchmark": "double_integrator_full_state",
            "u_applied": np.asarray(applied),
        }

    controller._apply_control_sequence = types.MethodType(
        apply_control_sequence,
        controller,
    )


def attach_time_indexed_reference(controller: Any, reference: Array) -> None:
    """
    Supply r[k+1] ... r[k+Np] and bypass path-geometry reference selection.

    This isolates the optimizer. Reference-path search should be tested
    separately after the optimization core passes.
    """

    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != controller.n:
        raise ValueError(
            f"Reference shape {reference.shape} is incompatible with "
            f"controller state dimension {controller.n}"
        )

    # Known aliases used by the project's mixins.
    controller.ref = reference
    controller.ref_path = reference
    controller.reference = reference
    controller.ref_xy = reference
    controller.X_ref = reference
    controller.lumen_C = reference
    controller.lumen = reference
    controller.lumen_xy = reference
    controller._academic_reference_k = 0

    def select_reference_indices(self: Any, x0: Array) -> Array:
        k = int(self._academic_reference_k)
        indices = np.arange(k + 1, k + 1 + self.Np, dtype=int)
        indices = np.clip(indices, 0, len(self.lumen_C) - 1)

        self.idx_ref_last = indices.copy()
        self.i_ref_last = int(indices[0])
        self.i_closest_last = int(indices[0])
        self.dist_to_ref_last = float(
            np.linalg.norm(
                np.asarray(x0, dtype=float) - self.lumen_C[indices[0]]
            )
        )
        return indices

    controller._select_reference_indices = types.MethodType(
        select_reference_indices,
        controller,
    )


def load_controller_class(specification: str) -> type:
    """Load ``package.module:ClassName`` without hard-coding the project path."""

    if ":" not in specification:
        raise ValueError(
            "Controller specification must be 'package.module:ClassName'"
        )
    module_name, class_name = specification.rsplit(":", 1)
    module = importlib.import_module(module_name)
    controller_class = getattr(module, class_name)
    if not isinstance(controller_class, type):
        raise TypeError(f"{specification} does not resolve to a class")
    return controller_class


def build_adapted_controller(
    controller_class: type,
    config: HarnessConfig,
    reference: Array,
    x0: Array,
) -> Any:
    """Construct MPCControllerTipXY with robot-only constraints made inactive."""

    A, _ = double_integrator_matrices(config.dt)
    forward_model = DoubleIntegratorForwardModel()
    input_jacobian = DoubleIntegratorInputJacobian(config.dt)

    p_min = np.full(8, -1.0e3, dtype=float)
    p_max = np.full(8, 1.0e3, dtype=float)
    u_max = np.array(
        [config.u_max, config.u_max, 0.5, 0.5, 0.5, 0.5, 0.5],
        dtype=float,
    )

    # All seven controls receive positive weights. Inputs 2:7 have zero plant
    # effect and must therefore optimize to zero.
    w_u = np.array(
        [config.r[0], config.r[1], 0.1, 0.1, 0.1, 0.1, 0.1],
        dtype=float,
    )
    w_du = np.array(
        [config.rd[0], config.rd[1], 0.1, 0.1, 0.1, 0.1, 0.1],
        dtype=float,
    )

    controller = controller_class(
        Jxy_fn=input_jacobian,
        forward_tip_fn=forward_model,
        dt=config.dt,
        Np=config.horizon,
        n_out=4,
        n_p=8,
        n_u=7,
        model_mode="lti",
        solver_mode=config.solver_mode,
        w_xy=config.q,
        w_u=w_u,
        w_du=w_du,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        enable_hard_epm_tip_clearance=False,
        use_trust_region = False,
        epm_tip_hard_min_m=-1.0e3,
        dL_index=6,
        dL_back_max=1.0e3,
        dL_fwd_max=1.0e3,
        qp_reg=1.0e-9,
    )

    # These are set explicitly because they are project-specific switches.
    # If the controller caches prediction matrices during construction, expose
    # A as a constructor argument or rebuild that cache in the project code.
    controller.A = A.copy()
    controller.track_all_outputs = True
    controller.use_affine_matching = False
    controller.ref_stage_weights = np.ones(config.horizon, dtype=float)

    patch_double_integrator_plant(controller, config.dt)
    attach_time_indexed_reference(controller, reference)
    controller.set_initial_params(pose_from_academic_state(x0))

    expected_initial_state = np.asarray(x0, dtype=float).reshape(4)
    state_error = np.max(np.abs(controller.x - expected_initial_state))
    if state_error > 1.0e-14:
        raise RuntimeError(
            "set_initial_params/forward model did not preserve the academic "
            f"initial state; max error={state_error:.3e}"
        )
    return controller


# ---------------------------------------------------------------------------
# Closed-loop oracle and project-controller tests
# ---------------------------------------------------------------------------


def run_oracle_closed_loop(
    scenario: Scenario,
    config: HarnessConfig,
    reference: Array,
) -> tuple[Array, Array]:
    """Run the independent optimizer in receding-horizon feedback."""

    x = np.asarray(scenario.x0, dtype=float).reshape(4)
    u_previous = np.zeros(2, dtype=float)
    states = np.empty((scenario.steps, 4), dtype=float)
    controls = np.empty((scenario.steps, 2), dtype=float)

    for k in range(scenario.steps):
        solution = solve_condensed_oracle(
            config=config,
            x0=x,
            reference_window=reference_window(
                reference,
                k,
                config.horizon,
            ),
            u_previous=u_previous,
        )
        if not solution.success:
            raise RuntimeError(
                f"Oracle failed in {scenario.name} at step {k}: "
                f"{solution.message}"
            )
        u = solution.U[0]
        x = double_integrator_step(x, u, config.dt)
        controls[k] = u
        states[k] = x
        u_previous = u

    return states, controls


def run_controller_scenario(
    *,
    controller_class: type,
    scenario: Scenario,
    config: HarnessConfig,
    reference: Array,
) -> ClosedLoopRun:
    """
    Compare every controller solve against the independent oracle.

    The oracle is solved from the controller's measured state at each step.
    This local comparison prevents one small early discrepancy from making all
    later comparisons meaningless through accumulated state differences.
    """

    controller = build_adapted_controller(
        controller_class,
        config,
        reference,
        np.asarray(scenario.x0, dtype=float),
    )

    states = np.empty((scenario.steps, 4), dtype=float)
    controls = np.empty((scenario.steps, controller.m), dtype=float)
    oracle_controls = np.empty((scenario.steps, 2), dtype=float)
    first_move_errors = np.empty(scenario.steps, dtype=float)
    sequence_errors = np.full(scenario.steps, np.nan, dtype=float)
    plant_step_errors = np.empty(scenario.steps, dtype=float)
    infeasible_flags = np.empty(scenario.steps, dtype=int)
    solve_times_ms = np.empty(scenario.steps, dtype=float)
    statuses: list[str] = []
    reference_index_mismatches = 0
    u_previous = np.zeros(2, dtype=float)

    for k in range(scenario.steps):
        controller._academic_reference_k = k
        x_measured = np.asarray(controller.x, dtype=float).reshape(4).copy()
        expected_indices = np.arange(
            k + 1,
            k + 1 + config.horizon,
            dtype=int,
        )
        expected_indices = np.clip(
            expected_indices,
            0,
            len(reference) - 1,
        )

        oracle = solve_condensed_oracle(
            config=config,
            x0=x_measured,
            reference_window=reference_window(
                reference,
                k,
                config.horizon,
            ),
            u_previous=u_previous,
        )
        if not oracle.success:
            raise RuntimeError(
                f"Oracle failed at step {k}: {oracle.message}"
            )

        start = time.perf_counter()
        _, x_next, info = controller.step(
            x_meas=x_measured,
            rollout_steps=1,
            solver_mode=config.solver_mode,
        )
        solve_times_ms[k] = 1000.0 * (time.perf_counter() - start)

        x_next = np.asarray(x_next, dtype=float).reshape(4)
        u_controller = np.asarray(info["u0"], dtype=float).reshape(
            controller.m
        )
        expected_next = double_integrator_step(
            x_measured,
            u_controller[:2],
            config.dt,
        )

        states[k] = x_next
        controls[k] = u_controller
        oracle_controls[k] = oracle.U[0]
        first_move_errors[k] = np.max(
            np.abs(u_controller[:2] - oracle.U[0])
        )
        plant_step_errors[k] = np.max(np.abs(x_next - expected_next))
        infeasible_flags[k] = int(info.get("infeasible", 0))
        statuses.append(str(info.get("status", "unknown")))

        optimized_sequence = np.asarray(
            info.get("U_opt", []),
            dtype=float,
        )
        if optimized_sequence.size == config.horizon * controller.m:
            optimized_sequence = optimized_sequence.reshape(
                config.horizon,
                controller.m,
            )
            sequence_errors[k] = np.max(
                np.abs(optimized_sequence[:, :2] - oracle.U)
            )

        actual_indices = np.asarray(
            getattr(controller, "idx_ref_last", []),
            dtype=int,
        )
        if not np.array_equal(actual_indices, expected_indices):
            reference_index_mismatches += 1

        u_previous = u_controller[:2].copy()

    sample_times = np.arange(1, scenario.steps + 1, dtype=float) * config.dt
    used_reference = reference[1 : scenario.steps + 1]
    return ClosedLoopRun(
        scenario=scenario,
        time=sample_times,
        reference=used_reference,
        states=states,
        controls=controls,
        oracle_controls=oracle_controls,
        first_move_errors=first_move_errors,
        sequence_errors=sequence_errors,
        plant_step_errors=plant_step_errors,
        infeasible_flags=infeasible_flags,
        statuses=statuses,
        reference_index_mismatches=reference_index_mismatches,
        solve_times_ms=solve_times_ms,
        input_limit=config.u_max,
    )


def checks_for_controller_run(
    run: ClosedLoopRun,
    config: HarnessConfig,
) -> list[Check]:
    """Translate a numerical run into explicit pass/fail statements."""

    checks: list[Check] = []
    prefix = run.scenario.name
    summary = run.summary()

    first_move_error = summary["max_first_move_oracle_error"]
    checks.append(
        Check.passed(
            f"{prefix}: first move vs oracle",
            "Every applied active input agrees with the independent MPC.",
            first_move_error,
            config.first_move_tolerance,
        )
        if first_move_error <= config.first_move_tolerance
        else Check.failed(
            f"{prefix}: first move vs oracle",
            "Applied control differs from the independent MPC optimum.",
            first_move_error,
            config.first_move_tolerance,
        )
    )

    sequence_error = summary["max_sequence_oracle_error"]
    if sequence_error is None:
        checks.append(
            Check.skipped(
                f"{prefix}: full sequence vs oracle",
                "Controller info did not expose U_opt.",
            )
        )
    else:
        checks.append(
            Check.passed(
                f"{prefix}: full sequence vs oracle",
                "The active optimized sequence agrees with the oracle.",
                sequence_error,
                config.sequence_tolerance,
            )
            if sequence_error <= config.sequence_tolerance
            else Check.failed(
                f"{prefix}: full sequence vs oracle",
                "The controller's active sequence differs from the oracle.",
                sequence_error,
                config.sequence_tolerance,
            )
        )

    max_input = summary["max_input_magnitude"]
    bound_violation = max(0.0, max_input - config.u_max)
    checks.append(
        Check.passed(
            f"{prefix}: active input bounds",
            "Both acceleration commands remain inside their hard bounds.",
            bound_violation,
            config.bound_tolerance,
        )
        if bound_violation <= config.bound_tolerance
        else Check.failed(
            f"{prefix}: active input bounds",
            "At least one acceleration command violates its hard bound.",
            bound_violation,
            config.bound_tolerance,
        )
    )

    inactive = summary["max_inactive_input_magnitude"]
    checks.append(
        Check.passed(
            f"{prefix}: inactive inputs",
            "The five inputs with zero plant effect remain zero.",
            inactive,
            config.inactive_input_tolerance,
        )
        if inactive <= config.inactive_input_tolerance
        else Check.failed(
            f"{prefix}: inactive inputs",
            "A control with no plant effect is nonzero.",
            inactive,
            config.inactive_input_tolerance,
        )
    )

    plant_error = summary["max_plant_step_error"]
    checks.append(
        Check.passed(
            f"{prefix}: plant rollout",
            "Returned x_next matches the known plant equation.",
            plant_error,
            config.plant_step_tolerance,
        )
        if plant_error <= config.plant_step_tolerance
        else Check.failed(
            f"{prefix}: plant rollout",
            "The controller returned a state inconsistent with the plant.",
            plant_error,
            config.plant_step_tolerance,
        )
    )

    infeasible_count = summary["infeasible_count"]
    checks.append(
        Check.passed(
            f"{prefix}: solver feasibility",
            "Every nominal QP solve reported feasible.",
            float(infeasible_count),
            0.0,
        )
        if infeasible_count == 0
        else Check.failed(
            f"{prefix}: solver feasibility",
            "At least one solve reported infeasible.",
            float(infeasible_count),
            0.0,
        )
    )

    mismatches = summary["reference_index_mismatches"]
    checks.append(
        Check.passed(
            f"{prefix}: reference indexing",
            "Prediction stage i uses reference k+i.",
            float(mismatches),
            0.0,
        )
        if mismatches == 0
        else Check.failed(
            f"{prefix}: reference indexing",
            "The controller used an incorrectly shifted reference window.",
            float(mismatches),
            0.0,
        )
    )

    if run.scenario.expect_saturation:
        saturation_level = float(
            np.max(np.abs(run.oracle_controls))
        )
        saturation_gap = abs(saturation_level - config.u_max)
        checks.append(
            Check.passed(
                f"{prefix}: constraint activation",
                "The intentionally infeasible command activates the input bound.",
                saturation_gap,
                1.0e-6,
            )
            if saturation_gap <= 1.0e-6
            else Check.failed(
                f"{prefix}: constraint activation",
                "The stress test did not activate the intended input constraint.",
                saturation_gap,
                1.0e-6,
            )
        )

    return checks


# ---------------------------------------------------------------------------
# Reporting and plots
# ---------------------------------------------------------------------------


def print_checks(checks: Sequence[Check]) -> None:
    """Print a compact terminal report."""

    width = max(len(check.name) for check in checks)
    print("\nAcademic MPC verification")
    print("=" * (width + 35))
    for check in checks:
        measurement = ""
        if check.measured is not None:
            measurement = f" measured={check.measured:.3e}"
        if check.limit is not None:
            measurement += f" limit={check.limit:.3e}"
        print(
            f"{check.status:4s}  {check.name:<{width}s}"
            f"{measurement}"
        )
        print(f"      {check.detail}")

    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("PASS", "FAIL", "SKIP")
    }
    print(
        f"\nSummary: {counts['PASS']} passed, "
        f"{counts['FAIL']} failed, {counts['SKIP']} skipped"
    )


def save_plots(runs: Sequence[ClosedLoopRun], output_path: Path) -> None:
    """Save position, input, and error plots for each controller scenario."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(runs),
        3,
        figsize=(15, 4.2 * len(runs)),
        squeeze=False,
    )

    for row, run in enumerate(runs):
        position_axes, input_axes, error_axes = axes[row]
        t = run.time

        position_axes.plot(
            t,
            run.reference[:, 0],
            "k--",
            label="x reference",
        )
        position_axes.plot(
            t,
            run.reference[:, 1],
            color="0.45",
            linestyle="--",
            label="y reference",
        )
        position_axes.plot(t, run.states[:, 0], label="x")
        position_axes.plot(t, run.states[:, 1], label="y")
        position_axes.set_title(f"{run.scenario.name}: position")
        position_axes.set_xlabel("time [s]")
        position_axes.set_ylabel("position")
        position_axes.grid(True)
        position_axes.legend()

        input_axes.plot(t, run.controls[:, 0], label="controller ax")
        input_axes.plot(t, run.controls[:, 1], label="controller ay")
        input_axes.plot(
            t,
            run.oracle_controls[:, 0],
            "--",
            label="oracle ax",
        )
        input_axes.plot(
            t,
            run.oracle_controls[:, 1],
            "--",
            label="oracle ay",
        )
        input_axes.set_title("Active control")
        input_axes.set_xlabel("time [s]")
        input_axes.set_ylabel("acceleration")
        input_axes.grid(True)
        input_axes.legend()

        position_error = np.linalg.norm(
            run.states[:, :2] - run.reference[:, :2],
            axis=1,
        )
        error_axes.semilogy(
            t,
            np.maximum(position_error, 1.0e-12),
            label="position error",
        )
        error_axes.semilogy(
            t,
            np.maximum(run.first_move_errors, 1.0e-12),
            label="input vs oracle",
        )
        error_axes.set_title("Tracking and verification errors")
        error_axes.set_xlabel("time [s]")
        error_axes.set_ylabel("norm / max abs error")
        error_axes.grid(True)
        error_axes.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_harness(
    *,
    config: HarnessConfig,
    controller_spec: Optional[str],
    require_controller: bool,
    make_plots: bool,
    output_directory: Path,
) -> tuple[HarnessReport, int]:
    """Run all requested tests and return the report and process exit code."""

    output_directory.mkdir(parents=True, exist_ok=True)
    checks = run_mathematical_self_tests(config)
    controller_loaded = False
    controller_message = "Controller integration tests were not requested."
    controller_class: Optional[type] = None
    runs: list[ClosedLoopRun] = []
    scenario_summaries: list[dict[str, Any]] = []

    if controller_spec is not None:
        try:
            controller_class = load_controller_class(controller_spec)
            controller_loaded = True
            controller_message = f"Loaded {controller_spec}"
            checks.append(
                Check.passed(
                    "controller import",
                    controller_message,
                )
            )
        except Exception as error:  # Report import problems without hiding them.
            controller_message = (
                f"Could not load {controller_spec}: "
                f"{type(error).__name__}: {error}"
            )
            checks.append(
                Check.failed("controller import", controller_message)
                if require_controller
                else Check.skipped("controller import", controller_message)
            )

    if controller_class is not None:
        for scenario in default_scenarios():
            _, reference, required_acceleration = build_scenario_reference(
                scenario,
                config,
            )

            # The oracle-only run is useful performance context and also makes
            # oracle solver failures explicit before judging the controller.
            oracle_states, oracle_controls = run_oracle_closed_loop(
                scenario,
                config,
                reference,
            )
            run = run_controller_scenario(
                controller_class=controller_class,
                scenario=scenario,
                config=config,
                reference=reference,
            )
            runs.append(run)
            checks.extend(checks_for_controller_run(run, config))

            summary = run.summary()
            oracle_reference = reference[1 : scenario.steps + 1]
            oracle_position_error = np.linalg.norm(
                oracle_states[:, :2] - oracle_reference[:, :2],
                axis=1,
            )
            summary.update(
                {
                    "purpose": scenario.purpose,
                    "peak_required_reference_acceleration": float(
                        np.max(np.abs(required_acceleration))
                    ),
                    "oracle_rms_position_error": float(
                        np.sqrt(np.mean(np.square(oracle_position_error)))
                    ),
                    "oracle_final_position_error": float(
                        oracle_position_error[-1]
                    ),
                    "oracle_max_input_magnitude": float(
                        np.max(np.abs(oracle_controls))
                    ),
                }
            )
            scenario_summaries.append(summary)

    report = HarnessReport(
        config=asdict(config),
        controller_spec=controller_spec,
        controller_loaded=controller_loaded,
        controller_message=controller_message,
        checks=checks,
        scenarios=scenario_summaries,
    )

    report_path = output_directory / "mpc_harness_report.json"
    report_path.write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    if make_plots and runs:
        save_plots(runs, output_directory / "mpc_harness_plots.png")

    print_checks(checks)
    print(f"\nReport: {report_path.resolve()}")
    if make_plots and runs:
        print(
            "Plots: "
            f"{(output_directory / 'mpc_harness_plots.png').resolve()}"
        )
    if controller_spec is not None:
        print(f"Controller: {controller_message}")

    any_failure = any(check.status == "FAIL" for check in checks)
    exit_code = 1 if any_failure else 0
    return report, exit_code


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify MPCControllerTipXY against an independent constrained "
            "double-integrator MPC."
        )
    )
    parser.add_argument(
        "--controller",
        default=(
            "proper_research.simulation_controller.controller:"
            "MPCControllerTipXY"
        ),
        help="Controller import as package.module:ClassName.",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Run only the mathematical benchmark/oracle tests.",
    )
    parser.add_argument(
        "--require-controller",
        action="store_true",
        help="Treat failure to import the controller as a failed test.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save plots when controller tests run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mpc_harness_output"),
        help="Directory for JSON results and optional plots.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    controller_spec = (
        None if arguments.self_test_only else arguments.controller
    )
    _, exit_code = run_harness(
        config=HarnessConfig(),
        controller_spec=controller_spec,
        require_controller=arguments.require_controller,
        make_plots=arguments.plot,
        output_directory=arguments.output_dir,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
