"""Closed-loop simulation of a time-parameterized configuration path.

This module is the online stage that follows
``time_parameterized_configuration_path.py``.  It deliberately does *not*
re-solve the geometric beam path online and it does not use contouring or a
free progress variable.  The online MPC tracks the scheduled actuator state

    z = [q1, q2, q3, q4, q5, q6, insertion]

with the scheduled feedforward input

    v = [q1dot, q2dot, q3dot, q4dot, q5dot, q6dot, insertion_rate].

The prediction model is the exact actuator kinematics used by the existing
joint-space simulator,

    z[k + 1] = z[k] + dt * v[k].

At every sample the convex QP minimizes scheduled state error, scheduled input
error, and input increments.  It enforces joint/insertion position bounds,
velocity bounds, and acceleration bounds.  The existing nonlinear magnetic
beam model remains the simulated plant/output: it is advanced sequentially
with ``commit=True`` and its tip position/tangent are checked against the
saved reference.  Consequently, beam feasibility is monitored independently
of QP solver success.

The public ``ConfigurationTrackingMPC.solve`` method only needs the measured
seven-dimensional actuator state and a reference sample index, so the same
class can later be called from a hardware loop after replacing the simulated
state with measured robot/insertion feedback.

Typical use
-----------
Place this file at
``proper_research/simulation/simulations/simulate_time_parameterized_configuration_mpc.py``
and run

    python -u -m proper_research.simulation.simulations.\
        simulate_time_parameterized_configuration_mpc

For a direct CSV input:

    python -u -m proper_research.simulation.simulations.\
        simulate_time_parameterized_configuration_mpc \
        --reference /path/to/time_parameterized_configuration_path.csv

Run the controller-only deterministic test with

    python -m proper_research.simulation.simulations.\
        simulate_time_parameterized_configuration_mpc --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import LinearConstraint, minimize

try:  # The project MPC environment already includes OSQP.
    import osqp
except ImportError:  # Keep --self-test and offline inspection usable elsewhere.
    osqp = None


Array = np.ndarray
STATE_NAMES = ("q1", "q2", "q3", "q4", "q5", "q6", "insertion")
INPUT_NAMES = (
    "qd1",
    "qd2",
    "qd3",
    "qd4",
    "qd5",
    "qd6",
    "insertion_rate",
)


def _finite_vector(value: Any, size: int, name: str) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != int(size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values.")
    return result.copy()


def _finite_matrix(value: Any, columns: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != int(columns):
        raise ValueError(
            f"{name} must have shape (N, {columns}), got {result.shape}."
        )
    if result.shape[0] < 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least two finite rows.")
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
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_angle_rad(first: Any, second: Any) -> float:
    a = np.asarray(first, dtype=float).reshape(3)
    b = np.asarray(second, dtype=float).reshape(3)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1.0e-12 or b_norm <= 1.0e-12:
        return np.inf
    cosine = float(np.clip(np.dot(a, b) / (a_norm * b_norm), -1.0, 1.0))
    return float(np.arccos(cosine))


@dataclass(frozen=True)
class ConfigurationReference:
    """Uniformly sampled state/input reference used by the online MPC."""

    time_s: Array
    path_coordinate_m: Array
    state: Array
    input: Array
    acceleration: Array
    desired_position_m: Array
    desired_tangent: Array
    planned_beam_feasible: Array
    planned_position_error_m: Array
    planned_tangent_error_rad: Array
    sample_period_s: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_planned_beam_feasible: bool = True) -> None:
        time_values = np.asarray(self.time_s, dtype=float).reshape(-1)
        if time_values.size < 2 or not np.all(np.isfinite(time_values)):
            raise ValueError("Reference time must contain at least two samples.")
        if abs(float(time_values[0])) > 1.0e-10:
            raise ValueError("Reference time must start at zero.")
        differences = np.diff(time_values)
        if np.any(differences <= 0.0):
            raise ValueError("Reference time must be strictly increasing.")
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive.")
        if not np.allclose(
            differences,
            float(self.sample_period_s),
            atol=1.0e-11,
            rtol=1.0e-9,
        ):
            raise ValueError("Reference must have a constant sample period.")
        row_count = time_values.size
        for name, value, columns in (
            ("state", self.state, 7),
            ("input", self.input, 7),
            ("acceleration", self.acceleration, 7),
            ("desired_position_m", self.desired_position_m, 3),
            ("desired_tangent", self.desired_tangent, 3),
        ):
            matrix = _finite_matrix(value, columns, name)
            if matrix.shape[0] != row_count:
                raise ValueError(f"{name} row count differs from reference time.")
        _unit_rows(self.desired_tangent, "desired_tangent")
        for name, value in (
            ("path_coordinate_m", self.path_coordinate_m),
            ("planned_position_error_m", self.planned_position_error_m),
            ("planned_tangent_error_rad", self.planned_tangent_error_rad),
        ):
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size != row_count or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain one finite value per sample.")
        feasible = np.asarray(self.planned_beam_feasible, dtype=bool).reshape(-1)
        if feasible.size != row_count:
            raise ValueError("planned_beam_feasible has the wrong length.")
        if require_planned_beam_feasible and not bool(np.all(feasible)):
            failed = np.flatnonzero(~feasible)
            raise RuntimeError(
                "The time-parameterized input contains infeasible beam samples: "
                f"first failed index={int(failed[0])}."
            )

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.time_s).size)

    @property
    def duration_s(self) -> float:
        return float(np.asarray(self.time_s, dtype=float)[-1])

    def state_window(self, control_index: int, horizon: int) -> Array:
        indices = np.clip(
            int(control_index) + 1 + np.arange(int(horizon)),
            0,
            self.sample_count - 1,
        )
        return np.asarray(self.state, dtype=float)[indices].copy()

    def input_window(self, control_index: int, horizon: int) -> Array:
        indices = np.clip(
            int(control_index) + np.arange(int(horizon)),
            0,
            self.sample_count - 1,
        )
        return np.asarray(self.input, dtype=float)[indices].copy()


@dataclass(frozen=True)
class ConfigurationMPCConfig:
    """Convex configuration-tracking MPC settings.

    Cost terms are dimensionless:

    ``state_tracking_weight * ||(z-z_ref)/state_error_scale||^2``
    ``input_tracking_weight * ||(v-v_ref)/input_error_scale||^2``
    ``input_increment_weight * ||Delta v/input_increment_scale||^2``

    The scales make radians and metres comparable and give every number an
    interpretable physical meaning.
    """

    sample_period_s: float
    prediction_horizon: int
    state_min: tuple[float, ...]
    state_max: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    acceleration_limit: tuple[float, ...]

    state_error_scale: tuple[float, ...] = (
        math.radians(0.5),
        math.radians(0.5),
        math.radians(0.5),
        math.radians(0.5),
        math.radians(0.5),
        math.radians(0.5),
        0.25e-3,
    )
    input_error_scale: tuple[float, ...] = (
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        5.0e-3,
    )
    input_increment_scale: tuple[float, ...] | None = None
    state_tracking_weight: float = 1.0
    input_tracking_weight: float = 1.0e-2
    input_increment_weight: float = 1.0e-3
    terminal_weight_multiplier: float = 20.0
    hessian_regularization: float = 1.0e-9

    solver_backend: Literal["auto", "osqp", "scipy"] = "auto"
    solver_absolute_tolerance: float = 1.0e-7
    solver_relative_tolerance: float = 1.0e-7
    solver_maximum_iterations: int = 20_000
    solver_polish: bool = True
    solver_verbose: bool = False

    def validate(self) -> None:
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive.")
        if int(self.prediction_horizon) < 1:
            raise ValueError("prediction_horizon must be at least one.")
        lower = _finite_vector(self.state_min, 7, "state_min")
        upper = _finite_vector(self.state_max, 7, "state_max")
        if np.any(lower >= upper):
            raise ValueError("Every state_min entry must be below state_max.")
        velocity = _finite_vector(self.velocity_limit, 7, "velocity_limit")
        acceleration = _finite_vector(
            self.acceleration_limit, 7, "acceleration_limit"
        )
        if np.any(velocity <= 0.0) or np.any(acceleration <= 0.0):
            raise ValueError("Velocity and acceleration limits must be positive.")
        state_scale = _finite_vector(
            self.state_error_scale, 7, "state_error_scale"
        )
        input_scale = _finite_vector(
            self.input_error_scale, 7, "input_error_scale"
        )
        if np.any(state_scale <= 0.0) or np.any(input_scale <= 0.0):
            raise ValueError("Cost normalization scales must be positive.")
        if self.input_increment_scale is not None:
            increment_scale = _finite_vector(
                self.input_increment_scale, 7, "input_increment_scale"
            )
            if np.any(increment_scale <= 0.0):
                raise ValueError("input_increment_scale must be positive.")
        for name in (
            "state_tracking_weight",
            "input_tracking_weight",
            "input_increment_weight",
            "terminal_weight_multiplier",
            "hessian_regularization",
            "solver_absolute_tolerance",
            "solver_relative_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.terminal_weight_multiplier < 1.0:
            raise ValueError("terminal_weight_multiplier must be at least one.")
        if self.solver_backend not in {"auto", "osqp", "scipy"}:
            raise ValueError("solver_backend must be 'auto', 'osqp', or 'scipy'.")
        if int(self.solver_maximum_iterations) < 1:
            raise ValueError("solver_maximum_iterations must be positive.")

    @property
    def effective_input_increment_scale(self) -> Array:
        if self.input_increment_scale is not None:
            return _finite_vector(
                self.input_increment_scale, 7, "input_increment_scale"
            )
        return (
            float(self.sample_period_s)
            * _finite_vector(
                self.acceleration_limit, 7, "acceleration_limit"
            )
        )


@dataclass
class ConfigurationMPCStep:
    command: Array
    planned_input: Array
    predicted_states: Array
    predicted_inputs: Array
    objective: float
    status: str
    success: bool
    iterations: int
    solve_time_s: float
    primal_residual: float
    dual_residual: float


@dataclass(frozen=True)
class ConfigurationSimulationConfig:
    position_tolerance_m: float = 1.5e-3
    tangent_tolerance_rad: float = math.radians(90.0)
    terminal_hold_steps: int = 15
    stop_on_qp_failure: bool = True
    stop_on_beam_violation: bool = True
    require_planned_beam_feasible: bool = True
    evaluate_nonlinear_beam: bool = True
    maximum_control_steps: int | None = None
    progress_stride: int = 25
    state_constraint_tolerance: float = 1.0e-8
    input_constraint_tolerance: float = 1.0e-8

    def validate(self) -> None:
        if not np.isfinite(self.position_tolerance_m) or self.position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be finite and positive.")
        if not np.isfinite(self.tangent_tolerance_rad) or not (
            0.0 < self.tangent_tolerance_rad < math.pi
        ):
            raise ValueError("tangent_tolerance_rad must lie in (0, pi).")
        if int(self.terminal_hold_steps) < 0:
            raise ValueError("terminal_hold_steps cannot be negative.")
        if self.maximum_control_steps is not None and self.maximum_control_steps < 1:
            raise ValueError("maximum_control_steps must be positive when set.")
        if int(self.progress_stride) < 1:
            raise ValueError("progress_stride must be positive.")


@dataclass
class ConfigurationSimulationResult:
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    output_files: dict[str, str] = field(default_factory=dict)


def _column(rows: list[dict[str, str]], name: str, *, default: float = 0.0) -> Array:
    if name not in rows[0]:
        return np.full(len(rows), float(default), dtype=float)
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _read_summary_metadata(directory: Path) -> dict[str, Any]:
    path = directory / "time_parameterized_configuration_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid time-parameterization summary: {path}")
    return payload


def _load_reference_csv(
    csv_path: Path,
    *,
    require_planned_beam_feasible: bool,
) -> ConfigurationReference:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"Reference CSV has fewer than two rows: {csv_path}")
    required = {
        "time_s",
        "path_s_m",
        "insertion_m",
        "insertion_rate_m_s",
        "desired_x",
        "desired_y",
        "desired_z",
        "desired_tangent_x",
        "desired_tangent_y",
        "desired_tangent_z",
    }
    for coordinate in range(1, 7):
        required.update(
            {
                f"q{coordinate}_rad",
                f"qd{coordinate}_rad_s",
                f"qdd{coordinate}_rad_s2",
            }
        )
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise KeyError("Reference CSV is missing columns: " + ", ".join(missing))

    time_values = _column(rows, "time_s")
    sample_period = float(np.median(np.diff(time_values)))
    state = np.column_stack(
        [_column(rows, f"q{i}_rad") for i in range(1, 7)]
        + [_column(rows, "insertion_m")]
    )
    input_values = np.column_stack(
        [_column(rows, f"qd{i}_rad_s") for i in range(1, 7)]
        + [_column(rows, "insertion_rate_m_s")]
    )
    acceleration = np.column_stack(
        [_column(rows, f"qdd{i}_rad_s2") for i in range(1, 7)]
        + [_column(rows, "insertion_acceleration_m_s2")]
    )
    desired_position = np.column_stack(
        [_column(rows, f"desired_{axis}") for axis in ("x", "y", "z")]
    )
    desired_tangent = np.column_stack(
        [
            _column(rows, f"desired_tangent_{axis}")
            for axis in ("x", "y", "z")
        ]
    )
    feasible = np.asarray(
        [bool(int(float(row.get("beam_feasible", "1")))) for row in rows],
        dtype=bool,
    )
    metadata = _read_summary_metadata(csv_path.parent)
    reference = ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=_column(rows, "path_s_m"),
        state=state,
        input=input_values,
        acceleration=acceleration,
        desired_position_m=desired_position,
        desired_tangent=desired_tangent,
        planned_beam_feasible=feasible,
        planned_position_error_m=_column(rows, "position_error_m"),
        planned_tangent_error_rad=_column(rows, "tangent_error_rad"),
        sample_period_s=sample_period,
        source=str(csv_path),
        metadata=metadata,
    )
    reference.validate(
        require_planned_beam_feasible=require_planned_beam_feasible
    )
    summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
    if (
        require_planned_beam_feasible
        and summary
        and not bool(summary.get("globally_feasible", True))
    ):
        raise RuntimeError("Saved time-parameterized reference is not feasible.")
    return reference


def _load_reference_npz(
    directory: Path,
    *,
    require_planned_beam_feasible: bool,
) -> ConfigurationReference:
    path = directory / "time_parameterized_configuration_path.npz"
    metadata = _read_summary_metadata(directory)
    with np.load(path) as archive:
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
            raise KeyError("Reference NPZ is missing: " + ", ".join(missing))
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    time_values = np.asarray(arrays["time_s"], dtype=float).reshape(-1)
    sample_period = float(np.median(np.diff(time_values)))
    count = time_values.size
    planned_feasible = np.asarray(
        arrays.get("beam_sample_feasible", np.ones(count)), dtype=bool
    ).reshape(count)
    reference = ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=np.asarray(arrays["path_s_m"], dtype=float),
        state=np.asarray(arrays["state_reference"], dtype=float),
        input=np.asarray(arrays["input_reference"], dtype=float),
        acceleration=np.asarray(
            arrays["state_acceleration_reference"], dtype=float
        ),
        desired_position_m=np.asarray(arrays["desired_position_m"], dtype=float),
        desired_tangent=np.asarray(arrays["desired_tangent"], dtype=float),
        planned_beam_feasible=planned_feasible,
        planned_position_error_m=np.asarray(
            arrays.get("position_error_m", np.zeros(count)), dtype=float
        ),
        planned_tangent_error_rad=np.asarray(
            arrays.get("tangent_error_rad", np.zeros(count)), dtype=float
        ),
        sample_period_s=sample_period,
        source=str(path),
        metadata=metadata,
    )
    reference.validate(
        require_planned_beam_feasible=require_planned_beam_feasible
    )
    summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
    if (
        require_planned_beam_feasible
        and summary
        and not bool(summary.get("globally_feasible", True))
    ):
        raise RuntimeError("Saved time-parameterized reference is not feasible.")
    return reference


def load_configuration_reference(
    source: str | Path,
    *,
    require_planned_beam_feasible: bool = True,
) -> ConfigurationReference:
    """Load either a time-parameterization directory or its CSV/NPZ file."""
    source_path = Path(source)
    if source_path.is_dir():
        csv_path = source_path / "time_parameterized_configuration_path.csv"
        npz_path = source_path / "time_parameterized_configuration_path.npz"
        if csv_path.exists():
            return _load_reference_csv(
                csv_path,
                require_planned_beam_feasible=require_planned_beam_feasible,
            )
        if npz_path.exists():
            return _load_reference_npz(
                source_path,
                require_planned_beam_feasible=require_planned_beam_feasible,
            )
        raise FileNotFoundError(
            f"No time-parameterized CSV or NPZ exists in {source_path}."
        )
    if source_path.suffix.lower() == ".csv":
        return _load_reference_csv(
            source_path,
            require_planned_beam_feasible=require_planned_beam_feasible,
        )
    if source_path.suffix.lower() == ".npz":
        return _load_reference_npz(
            source_path.parent,
            require_planned_beam_feasible=require_planned_beam_feasible,
        )
    raise ValueError("Reference must be a directory, .csv file, or .npz file.")


class ConfigurationTrackingMPC:
    """Linear configuration MPC with fixed-schedule references.

    This controller is independent of the magnetic beam model.  The beam is a
    downstream nonlinear output and safety monitor in the simulation runner.
    """

    def __init__(
        self,
        *,
        reference: ConfigurationReference,
        config: ConfigurationMPCConfig,
    ) -> None:
        reference.validate(require_planned_beam_feasible=False)
        config.validate()
        if not np.isclose(
            reference.sample_period_s,
            config.sample_period_s,
            atol=1.0e-12,
            rtol=1.0e-10,
        ):
            raise ValueError(
                "MPC sample period differs from the time-parameterized reference."
            )
        self.reference = reference
        self.config = config
        self.dt = float(config.sample_period_s)
        self.N = int(config.prediction_horizon)
        self.n = 7
        self.m = 7
        self.nu = self.N * self.m
        self.state_min = _finite_vector(config.state_min, 7, "state_min")
        self.state_max = _finite_vector(config.state_max, 7, "state_max")
        self.velocity_limit = _finite_vector(
            config.velocity_limit, 7, "velocity_limit"
        )
        self.acceleration_limit = _finite_vector(
            config.acceleration_limit, 7, "acceleration_limit"
        )
        self._build_problem_matrices()
        self._warm_start: Array | None = None
        self._solver = None
        backend = config.solver_backend
        if backend == "auto":
            backend = "osqp" if osqp is not None else "scipy"
        if backend == "osqp" and osqp is None:
            raise ImportError(
                "solver_backend='osqp' requested, but osqp is not installed."
            )
        self.backend = str(backend)
        if self.backend == "osqp":
            self._setup_osqp()

    def _build_problem_matrices(self) -> None:
        identity_state = np.eye(self.n, dtype=float)
        self.E = np.kron(np.ones((self.N, 1)), identity_state)
        self.S = np.kron(
            np.tril(np.ones((self.N, self.N), dtype=float)),
            self.dt * identity_state,
        )
        D_scalar = np.eye(self.N, dtype=float)
        if self.N > 1:
            D_scalar[np.arange(1, self.N), np.arange(self.N - 1)] = -1.0
        self.D = np.kron(D_scalar, identity_state)

        state_scale = _finite_vector(
            self.config.state_error_scale, 7, "state_error_scale"
        )
        input_scale = _finite_vector(
            self.config.input_error_scale, 7, "input_error_scale"
        )
        increment_scale = self.config.effective_input_increment_scale
        Q = float(self.config.state_tracking_weight) * np.diag(
            1.0 / state_scale**2
        )
        R = float(self.config.input_tracking_weight) * np.diag(
            1.0 / input_scale**2
        )
        Rd = float(self.config.input_increment_weight) * np.diag(
            1.0 / increment_scale**2
        )
        Q_blocks = [Q.copy() for _ in range(self.N)]
        Q_blocks[-1] *= float(self.config.terminal_weight_multiplier)
        self.Qbar = sp.block_diag(Q_blocks, format="csc").toarray()
        self.Rbar = np.kron(np.eye(self.N), R)
        self.Rdbar = np.kron(np.eye(self.N), Rd)
        hessian = 2.0 * (
            self.S.T @ self.Qbar @ self.S
            + self.Rbar
            + self.D.T @ self.Rdbar @ self.D
        )
        hessian += float(self.config.hessian_regularization) * np.eye(self.nu)
        self.H = 0.5 * (hessian + hessian.T)
        self.A = np.vstack((np.eye(self.nu), self.S, self.D))
        self._input_rows = slice(0, self.nu)
        self._state_rows = slice(self.nu, 2 * self.nu)
        self._increment_rows = slice(2 * self.nu, 3 * self.nu)

    def _setup_osqp(self) -> None:
        assert osqp is not None
        self._solver = osqp.OSQP()
        zero = np.zeros(self.nu, dtype=float)
        lower, upper = self._constraint_bounds(
            state=np.zeros(7, dtype=float),
            previous_input=np.zeros(7, dtype=float),
            validate_state=False,
        )
        self._solver.setup(
            P=sp.triu(sp.csc_matrix(self.H), format="csc"),
            q=zero,
            A=sp.csc_matrix(self.A),
            l=lower,
            u=upper,
            eps_abs=float(self.config.solver_absolute_tolerance),
            eps_rel=float(self.config.solver_relative_tolerance),
            max_iter=int(self.config.solver_maximum_iterations),
            # ``polish`` and ``warm_start`` work with both the OSQP 0.6 API
            # used by many robotics environments and current OSQP releases.
            polish=bool(self.config.solver_polish),
            verbose=bool(self.config.solver_verbose),
            warm_start=True,
        )

    def _constraint_bounds(
        self,
        *,
        state: Array,
        previous_input: Array,
        validate_state: bool = True,
    ) -> tuple[Array, Array]:
        state = _finite_vector(state, 7, "measured state")
        previous_input = _finite_vector(previous_input, 7, "previous input")
        if validate_state and (
            np.any(state < self.state_min - 1.0e-8)
            or np.any(state > self.state_max + 1.0e-8)
        ):
            raise ValueError("Measured state lies outside the configured bounds.")
        input_lower = np.tile(-self.velocity_limit, self.N)
        input_upper = np.tile(self.velocity_limit, self.N)
        state_lower = np.tile(self.state_min, self.N) - self.E @ state
        state_upper = np.tile(self.state_max, self.N) - self.E @ state
        change = np.zeros(self.nu, dtype=float)
        change[: self.m] = previous_input
        delta = np.tile(self.dt * self.acceleration_limit, self.N)
        increment_lower = change - delta
        increment_upper = change + delta
        return (
            np.concatenate((input_lower, state_lower, increment_lower)),
            np.concatenate((input_upper, state_upper, increment_upper)),
        )

    def _linear_cost(
        self,
        *,
        state: Array,
        previous_input: Array,
        state_reference: Array,
        input_reference: Array,
    ) -> Array:
        free_state = self.E @ state
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input
        return 2.0 * (
            self.S.T
            @ self.Qbar
            @ (free_state - state_reference_vector)
            - self.Rbar @ input_reference_vector
            - self.D.T @ self.Rdbar @ previous_vector
        )

    def _feasible_warm_start(
        self,
        *,
        input_reference: Array,
        previous_input: Array,
    ) -> Array:
        if self._warm_start is not None and self._warm_start.size == self.nu:
            shifted = self._warm_start.reshape(self.N, self.m)
            guess = np.vstack((shifted[1:], shifted[-1:])).copy()
        else:
            guess = np.asarray(input_reference, dtype=float).reshape(self.N, self.m)
        previous = np.asarray(previous_input, dtype=float).reshape(7)
        maximum_change = self.dt * self.acceleration_limit
        for stage in range(self.N):
            guess[stage] = np.clip(
                guess[stage], previous - maximum_change, previous + maximum_change
            )
            guess[stage] = np.clip(
                guess[stage], -self.velocity_limit, self.velocity_limit
            )
            previous = guess[stage]
        return guess.reshape(self.nu)

    def _solve_scipy(
        self,
        *,
        linear_cost: Array,
        lower: Array,
        upper: Array,
        warm_start: Array,
    ) -> tuple[Array | None, dict[str, Any]]:
        constraint = LinearConstraint(self.A, lower, upper)

        def objective(value: Array) -> float:
            return float(0.5 * value @ self.H @ value + linear_cost @ value)

        def gradient(value: Array) -> Array:
            return self.H @ value + linear_cost

        started = time.perf_counter()
        result = minimize(
            objective,
            warm_start,
            jac=gradient,
            constraints=(constraint,),
            method="SLSQP",
            options={
                "ftol": float(self.config.solver_absolute_tolerance),
                "maxiter": int(self.config.solver_maximum_iterations),
                "disp": bool(self.config.solver_verbose),
            },
        )
        elapsed = time.perf_counter() - started
        value = None if result.x is None else np.asarray(result.x, dtype=float)
        return value, {
            "status": str(result.message).lower(),
            "success": bool(result.success),
            "iterations": int(getattr(result, "nit", 0)),
            "solve_time_s": float(elapsed),
            "primal_residual": np.nan,
            "dual_residual": np.nan,
        }

    def solve(
        self,
        *,
        measured_state: Any,
        control_index: int,
        previous_input: Any,
    ) -> ConfigurationMPCStep:
        """Solve one receding-horizon QP without advancing the plant."""
        state = _finite_vector(measured_state, 7, "measured_state")
        previous = _finite_vector(previous_input, 7, "previous_input")
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)
        linear_cost = self._linear_cost(
            state=state,
            previous_input=previous,
            state_reference=state_reference,
            input_reference=input_reference,
        )
        lower, upper = self._constraint_bounds(
            state=state, previous_input=previous
        )
        warm_start = self._feasible_warm_start(
            input_reference=input_reference, previous_input=previous
        )

        if self.backend == "osqp":
            assert self._solver is not None
            self._solver.update(q=linear_cost, l=lower, u=upper)
            self._solver.warm_start(x=warm_start)
            result = self._solver.solve()
            status = str(result.info.status).lower()
            success = status in {"solved", "solved inaccurate"}
            solution = (
                None
                if result.x is None
                else np.asarray(result.x, dtype=float).reshape(self.nu)
            )
            diagnostic = {
                "status": status,
                "success": success,
                "iterations": int(result.info.iter),
                "solve_time_s": float(result.info.run_time),
                "primal_residual": float(
                    getattr(
                        result.info,
                        "prim_res",
                        getattr(result.info, "pri_res", np.nan),
                    )
                ),
                "dual_residual": float(
                    getattr(
                        result.info,
                        "dual_res",
                        getattr(result.info, "dua_res", np.nan),
                    )
                ),
            }
        else:
            solution, diagnostic = self._solve_scipy(
                linear_cost=linear_cost,
                lower=lower,
                upper=upper,
                warm_start=warm_start,
            )

        if solution is None or not bool(diagnostic["success"]):
            command = np.zeros(7, dtype=float)
            predicted_inputs = np.zeros((self.N, self.m), dtype=float)
            predicted_states = np.tile(state, (self.N, 1))
            objective = np.inf
        else:
            predicted_inputs = solution.reshape(self.N, self.m)
            predicted_states = (
                self.E @ state + self.S @ solution
            ).reshape(self.N, self.n)
            command = predicted_inputs[0].copy()
            objective = float(
                0.5 * solution @ self.H @ solution + linear_cost @ solution
            )
            self._warm_start = solution.copy()

        return ConfigurationMPCStep(
            command=command,
            planned_input=np.asarray(input_reference[0], dtype=float).copy(),
            predicted_states=predicted_states,
            predicted_inputs=predicted_inputs,
            objective=objective,
            status=str(diagnostic["status"]),
            success=bool(diagnostic["success"]),
            iterations=int(diagnostic["iterations"]),
            solve_time_s=float(diagnostic["solve_time_s"]),
            primal_residual=float(diagnostic["primal_residual"]),
            dual_residual=float(diagnostic["dual_residual"]),
        )


def _beam_snapshot(
    *,
    forward6d: Any,
    state: Array,
    commit: bool,
) -> tuple[Array, Array]:
    start_step = getattr(forward6d, "start_step", None)
    if callable(start_step):
        start_step()
    output = np.asarray(forward6d(state, commit=bool(commit)), dtype=float).reshape(-1)
    if output.size < 6 or not np.all(np.isfinite(output[:6])):
        raise FloatingPointError(
            f"The nonlinear beam plant returned invalid output {output}."
        )
    tangent = output[3:6]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1.0e-12:
        raise FloatingPointError("The nonlinear beam plant returned zero tangent.")
    return output[:3].copy(), tangent / norm


def _magnet_pose(
    *,
    controller_pack: dict[str, Any],
    state: Array,
) -> tuple[Array, Array]:
    adapter = controller_pack.get("plant_diagnostic_joint_adapter")
    if adapter is None:
        adapter = controller_pack.get("plant_joint_adapter")
    if adapter is None or not callable(getattr(adapter, "magnet_transform", None)):
        return np.full(3, np.nan), np.full(4, np.nan)
    transform = np.asarray(adapter.magnet_transform(state), dtype=float).reshape(4, 4)
    try:
        from scipy.spatial.transform import Rotation

        quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    except Exception:
        quaternion = np.full(4, np.nan)
    return transform[:3, 3].copy(), quaternion


def _make_record(
    *,
    simulation_sample: int,
    reference_index: int,
    reference: ConfigurationReference,
    actual_state: Array,
    command: Array,
    previous_command: Array,
    mpc_step: ConfigurationMPCStep | None,
    achieved_position: Array,
    achieved_tangent: Array,
    magnet_position: Array,
    magnet_quaternion_xyzw: Array,
    position_tolerance_m: float,
    tangent_tolerance_rad: float,
) -> dict[str, Any]:
    index = int(np.clip(reference_index, 0, reference.sample_count - 1))
    state_reference = np.asarray(reference.state[index], dtype=float)
    input_reference = np.asarray(reference.input[index], dtype=float)
    desired_position = np.asarray(reference.desired_position_m[index], dtype=float)
    desired_tangent = np.asarray(reference.desired_tangent[index], dtype=float)
    position_error = float(np.linalg.norm(achieved_position - desired_position))
    tangent_error = _safe_angle_rad(achieved_tangent, desired_tangent)
    state_error = actual_state - state_reference
    input_error = command - input_reference
    acceleration = (
        (command - previous_command) / float(reference.sample_period_s)
        if simulation_sample > 0
        else np.zeros(7, dtype=float)
    )
    # print(f"Position Error: {position_error} Position Tolerance: {position_tolerance_m} Tangent error: {tangent_error}, Tangent tolerance: {tangent_tolerance_rad}")

    record: dict[str, Any] = {
        "simulation_sample": int(simulation_sample),
        "time_s": float(simulation_sample * reference.sample_period_s),
        "reference_index": index,
        "reference_time_s": float(reference.time_s[index]),
        "reference_path_s_m": float(reference.path_coordinate_m[index]),
        "reference_is_terminal_hold": int(simulation_sample >= reference.sample_count),
        "qp_status": "initial" if mpc_step is None else str(mpc_step.status),
        "qp_success": int(mpc_step is None or mpc_step.success),
        "qp_iterations": 0 if mpc_step is None else int(mpc_step.iterations),
        "qp_solve_time_s": 0.0 if mpc_step is None else float(mpc_step.solve_time_s),
        "qp_objective": np.nan if mpc_step is None else float(mpc_step.objective),
        "qp_primal_residual": (
            np.nan if mpc_step is None else float(mpc_step.primal_residual)
        ),
        "qp_dual_residual": (
            np.nan if mpc_step is None else float(mpc_step.dual_residual)
        ),
        "beam_position_error_m": position_error,
        "beam_position_error_mm": 1.0e3 * position_error,
        "beam_tangent_error_rad": tangent_error,
        "beam_tangent_error_deg": math.degrees(tangent_error),
        "beam_feasible": int(
            position_error <= position_tolerance_m
            and tangent_error <= tangent_tolerance_rad
        ),
        "desired_x": float(desired_position[0]),
        "desired_y": float(desired_position[1]),
        "desired_z": float(desired_position[2]),
        "achieved_x": float(achieved_position[0]),
        "achieved_y": float(achieved_position[1]),
        "achieved_z": float(achieved_position[2]),
        "desired_tangent_x": float(desired_tangent[0]),
        "desired_tangent_y": float(desired_tangent[1]),
        "desired_tangent_z": float(desired_tangent[2]),
        "achieved_tangent_x": float(achieved_tangent[0]),
        "achieved_tangent_y": float(achieved_tangent[1]),
        "achieved_tangent_z": float(achieved_tangent[2]),
        "magnet_x": float(magnet_position[0]),
        "magnet_y": float(magnet_position[1]),
        "magnet_z": float(magnet_position[2]),
        "magnet_qx": float(magnet_quaternion_xyzw[0]),
        "magnet_qy": float(magnet_quaternion_xyzw[1]),
        "magnet_qz": float(magnet_quaternion_xyzw[2]),
        "magnet_qw": float(magnet_quaternion_xyzw[3]),
    }
    for coordinate, name in enumerate(STATE_NAMES):
        record[f"{name}_actual"] = float(actual_state[coordinate])
        record[f"{name}_reference"] = float(state_reference[coordinate])
        record[f"{name}_error"] = float(state_error[coordinate])
    for coordinate, name in enumerate(INPUT_NAMES):
        record[f"{name}_command"] = float(command[coordinate])
        record[f"{name}_reference"] = float(input_reference[coordinate])
        record[f"{name}_error"] = float(input_error[coordinate])
        record[f"{name}_acceleration"] = float(acceleration[coordinate])
    return record


def _summarize_simulation(
    *,
    records: list[dict[str, Any]],
    reference: ConfigurationReference,
    mpc_config: ConfigurationMPCConfig,
    simulation_config: ConfigurationSimulationConfig,
    expected_control_steps: int,
    stopped_reason: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty simulation.")
    qp_records = records[1:]
    final = records[-1]
    joint_error = np.asarray(
        [[row[f"q{i}_error"] for i in range(1, 7)] for row in records],
        dtype=float,
    )
    insertion_error = np.asarray(
        [row["insertion_error"] for row in records], dtype=float
    )
    commands = np.asarray(
        [
            [row[f"{name}_command"] for name in INPUT_NAMES]
            for row in records
        ],
        dtype=float,
    )
    accelerations = np.asarray(
        [
            [row[f"{name}_acceleration"] for name in INPUT_NAMES]
            for row in records
        ],
        dtype=float,
    )
    velocity_limit = _finite_vector(
        mpc_config.velocity_limit, 7, "velocity_limit"
    )
    acceleration_limit = _finite_vector(
        mpc_config.acceleration_limit, 7, "acceleration_limit"
    )
    state_scale = _finite_vector(
        mpc_config.state_error_scale, 7, "state_error_scale"
    )
    final_state_error = np.asarray(
        [final[f"{name}_error"] for name in STATE_NAMES], dtype=float
    )
    completed = (len(records) - 1) >= int(expected_control_steps)
    all_qp_solved = bool(
        all(bool(row["qp_success"]) for row in qp_records)
        and not str(stopped_reason).startswith("qp_failure")
    )
    all_beam_feasible = all(bool(row["beam_feasible"]) for row in records)
    terminal_tracking_satisfied = bool(
        np.max(np.abs(final_state_error) / state_scale) <= 1.0 + 1.0e-8
    )
    final_rate_ratio = float(
        np.max(np.abs(commands[-1]) / np.maximum(velocity_limit, 1.0e-12))
    )
    success = bool(
        completed
        and all_qp_solved
        and all_beam_feasible
        and terminal_tracking_satisfied
        and final_rate_ratio <= 1.0e-3
    )
    solve_times = np.asarray(
        [row["qp_solve_time_s"] for row in qp_records], dtype=float
    )
    return {
        "success": success,
        "completed_requested_steps": completed,
        "all_qp_solved": all_qp_solved,
        "nonlinear_beam_evaluation_performed": bool(
            simulation_config.evaluate_nonlinear_beam
        ),
        "all_beam_samples_feasible": all_beam_feasible,
        "terminal_tracking_satisfied": terminal_tracking_satisfied,
        "terminal_rate_near_zero": bool(final_rate_ratio <= 1.0e-3),
        "stopped_reason": str(stopped_reason),
        "reference_source": reference.source,
        "reference_samples": reference.sample_count,
        "reference_duration_s": reference.duration_s,
        "simulation_samples": len(records),
        "simulation_duration_s": float(records[-1]["time_s"]),
        "maximum_absolute_joint_error_rad": float(np.max(np.abs(joint_error))),
        "maximum_absolute_joint_error_deg": float(
            np.degrees(np.max(np.abs(joint_error)))
        ),
        "maximum_absolute_insertion_error_m": float(
            np.max(np.abs(insertion_error))
        ),
        "maximum_absolute_insertion_error_mm": float(
            1.0e3 * np.max(np.abs(insertion_error))
        ),
        "final_scaled_state_error": float(
            np.max(np.abs(final_state_error) / state_scale)
        ),
        "maximum_beam_position_error_m": float(
            max(row["beam_position_error_m"] for row in records)
        ),
        "maximum_beam_position_error_mm": float(
            max(row["beam_position_error_mm"] for row in records)
        ),
        "maximum_beam_tangent_error_rad": float(
            max(row["beam_tangent_error_rad"] for row in records)
        ),
        "maximum_beam_tangent_error_deg": float(
            max(row["beam_tangent_error_deg"] for row in records)
        ),
        "maximum_velocity_limit_ratio": float(
            np.max(np.abs(commands) / velocity_limit[None, :])
        ),
        "maximum_acceleration_limit_ratio": float(
            np.max(np.abs(accelerations) / acceleration_limit[None, :])
        ),
        "final_velocity_limit_ratio": final_rate_ratio,
        "mean_qp_solve_time_ms": (
            float(1.0e3 * np.mean(solve_times)) if solve_times.size else 0.0
        ),
        "maximum_qp_solve_time_ms": (
            float(1.0e3 * np.max(solve_times)) if solve_times.size else 0.0
        ),
        "position_tolerance_m": simulation_config.position_tolerance_m,
        "tangent_tolerance_rad": simulation_config.tangent_tolerance_rad,
        "mpc_configuration": _json_ready(mpc_config),
        "simulation_configuration": _json_ready(simulation_config),
    }


def simulate_time_parameterized_configuration_mpc(
    *,
    reference: ConfigurationReference,
    controller_pack: dict[str, Any],
    mpc_config: ConfigurationMPCConfig,
    simulation_config: ConfigurationSimulationConfig,
    output_dir: str | Path | None,
    initial_state: Any | None = None,
) -> ConfigurationSimulationResult:
    """Run configuration MPC against the existing nonlinear beam plant."""
    simulation_config.validate()
    reference.validate(
        require_planned_beam_feasible=(
            simulation_config.require_planned_beam_feasible
        )
    )
    mpc_config.validate()
    controller_dt = getattr(controller_pack.get("controller"), "dt", None)
    if controller_dt is not None and not np.isclose(
        float(controller_dt),
        reference.sample_period_s,
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise ValueError(
            "The rebuilt controller dt does not match the time reference: "
            f"controller={float(controller_dt):.9f}, "
            f"reference={reference.sample_period_s:.9f}."
        )
    state = (
        np.asarray(reference.state[0], dtype=float).copy()
        if initial_state is None
        else _finite_vector(initial_state, 7, "initial_state")
    )
    state_min = _finite_vector(mpc_config.state_min, 7, "state_min")
    state_max = _finite_vector(mpc_config.state_max, 7, "state_max")
    if np.any(state < state_min) or np.any(state > state_max):
        raise ValueError("Initial state lies outside joint/insertion bounds.")
    forward6d = controller_pack.get("forward6d_plant")
    if simulation_config.evaluate_nonlinear_beam and forward6d is None:
        raise KeyError("controller_pack does not contain 'forward6d_plant'.")
    mpc = ConfigurationTrackingMPC(reference=reference, config=mpc_config)

    intended_steps = reference.sample_count - 1 + int(
        simulation_config.terminal_hold_steps
    )
    control_steps = intended_steps
    if simulation_config.maximum_control_steps is not None:
        control_steps = min(control_steps, int(simulation_config.maximum_control_steps))
    previous_input = np.zeros(7, dtype=float)
    if simulation_config.evaluate_nonlinear_beam:
        achieved_position, achieved_tangent = _beam_snapshot(
            forward6d=forward6d, state=state, commit=True
        )
    else:
        achieved_position = np.asarray(reference.desired_position_m[0], dtype=float)
        achieved_tangent = np.asarray(reference.desired_tangent[0], dtype=float)
    magnet_position, magnet_quaternion = _magnet_pose(
        controller_pack=controller_pack, state=state
    )
    records = [
        _make_record(
            simulation_sample=0,
            reference_index=0,
            reference=reference,
            actual_state=state,
            command=previous_input,
            previous_command=previous_input,
            mpc_step=None,
            achieved_position=achieved_position,
            achieved_tangent=achieved_tangent,
            magnet_position=magnet_position,
            magnet_quaternion_xyzw=magnet_quaternion,
            position_tolerance_m=simulation_config.position_tolerance_m,
            tangent_tolerance_rad=simulation_config.tangent_tolerance_rad,
        )
    ]
    stopped_reason = "completed"
    if not bool(records[0]["beam_feasible"]):
        stopped_reason = "initial_beam_output_infeasible"
        if simulation_config.stop_on_beam_violation:
            result = ConfigurationSimulationResult(records=records, summary={})
            result.summary = _summarize_simulation(
                records=records,
                reference=reference,
                mpc_config=mpc_config,
                simulation_config=simulation_config,
                expected_control_steps=control_steps,
                stopped_reason=stopped_reason,
            )
            if output_dir is not None:
                save_configuration_mpc_simulation(result, output_dir)
            return result

    for control_index in range(control_steps):
        step = mpc.solve(
            measured_state=state,
            control_index=control_index,
            previous_input=previous_input,
        )
        if not step.success:
            stopped_reason = f"qp_failure:{step.status}"
            if simulation_config.stop_on_qp_failure:
                break
        command = np.asarray(step.command, dtype=float).reshape(7)
        next_state = state + reference.sample_period_s * command
        state_violation = max(
            float(np.max(state_min - next_state)),
            float(np.max(next_state - state_max)),
            0.0,
        )
        input_violation = max(
            float(
                np.max(
                    np.abs(command)
                    - _finite_vector(mpc_config.velocity_limit, 7, "velocity_limit")
                )
            ),
            0.0,
        )
        acceleration = (command - previous_input) / reference.sample_period_s
        acceleration_violation = max(
            float(
                np.max(
                    np.abs(acceleration)
                    - _finite_vector(
                        mpc_config.acceleration_limit, 7, "acceleration_limit"
                    )
                )
            ),
            0.0,
        )
        if state_violation > simulation_config.state_constraint_tolerance:
            raise RuntimeError(
                f"Applied state violates a hard bound by {state_violation:.3e}."
            )
        if (
            max(input_violation, acceleration_violation)
            > simulation_config.input_constraint_tolerance
        ):
            raise RuntimeError(
                "Applied command violates a hard velocity/acceleration bound: "
                f"velocity={input_violation:.3e}, "
                f"acceleration={acceleration_violation:.3e}."
            )
        state = next_state
        old_input = previous_input.copy()
        previous_input = command.copy()
        reference_index = min(control_index + 1, reference.sample_count - 1)
        if simulation_config.evaluate_nonlinear_beam:
            achieved_position, achieved_tangent = _beam_snapshot(
                forward6d=forward6d, state=state, commit=True
            )
        else:
            achieved_position = np.asarray(
                reference.desired_position_m[reference_index], dtype=float
            )
            achieved_tangent = np.asarray(
                reference.desired_tangent[reference_index], dtype=float
            )
        magnet_position, magnet_quaternion = _magnet_pose(
            controller_pack=controller_pack, state=state
        )
        record = _make_record(
            simulation_sample=control_index + 1,
            reference_index=reference_index,
            reference=reference,
            actual_state=state,
            command=command,
            previous_command=old_input,
            mpc_step=step,
            achieved_position=achieved_position,
            achieved_tangent=achieved_tangent,
            magnet_position=magnet_position,
            magnet_quaternion_xyzw=magnet_quaternion,
            position_tolerance_m=simulation_config.position_tolerance_m,
            tangent_tolerance_rad=simulation_config.tangent_tolerance_rad,
        )
        records.append(record)
        if (
            control_index == 0
            or (control_index + 1) % simulation_config.progress_stride == 0
            or control_index + 1 == control_steps
        ):
            print(
                "[CONFIGURATION MPC] "
                f"sample={control_index + 1}/{control_steps} "
                f"reference={reference_index}/{reference.sample_count - 1} "
                f"status={step.status} "
                f"solve={1.0e3 * step.solve_time_s:.3f} ms "
                f"joint_error={np.degrees(np.max(np.abs(state[:6] - reference.state[reference_index, :6]))):.4f} deg "
                f"insertion_error={1.0e3 * abs(state[6] - reference.state[reference_index, 6]):.4f} mm "
                f"beam_position={record['beam_position_error_mm']:.4f} mm "
                f"beam_tangent={record['beam_tangent_error_deg']:.4f} deg",
                flush=True,
            )
        if not bool(record["beam_feasible"]):
            stopped_reason = "nonlinear_beam_validation_failed"
            if simulation_config.stop_on_beam_violation:
                break

    result = ConfigurationSimulationResult(records=records, summary={})
    result.summary = _summarize_simulation(
        records=records,
        reference=reference,
        mpc_config=mpc_config,
        simulation_config=simulation_config,
        expected_control_steps=control_steps,
        stopped_reason=stopped_reason,
    )
    if output_dir is not None:
        save_configuration_mpc_simulation(result, output_dir)
    return result


def _save_simulation_plot(
    result: ConfigurationSimulationResult,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result.records
    time_values = np.asarray([row["time_s"] for row in rows], dtype=float)
    figure, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    for coordinate in range(1, 7):
        error = np.asarray([row[f"q{coordinate}_error"] for row in rows])
        axes[0, 0].plot(time_values, np.degrees(error), label=f"q{coordinate}")
    axes[0, 0].set_title("Joint tracking error")
    axes[0, 0].set_ylabel("error [deg]")
    axes[0, 0].legend(ncol=3, fontsize=8)

    insertion_actual = np.asarray([row["insertion_actual"] for row in rows])
    insertion_reference = np.asarray(
        [row["insertion_reference"] for row in rows]
    )
    axes[0, 1].plot(time_values, 1.0e3 * insertion_reference, "--", label="reference")
    axes[0, 1].plot(time_values, 1.0e3 * insertion_actual, label="actual")
    axes[0, 1].set_title("Insertion tracking")
    axes[0, 1].set_ylabel("insertion [mm]")
    axes[0, 1].legend()

    for coordinate in range(1, 7):
        command = np.asarray([row[f"qd{coordinate}_command"] for row in rows])
        reference_value = np.asarray(
            [row[f"qd{coordinate}_reference"] for row in rows]
        )
        axes[1, 0].plot(time_values, command, label=f"qd{coordinate}")
        axes[1, 0].plot(time_values, reference_value, "--", alpha=0.35)
    axes[1, 0].set_title("Joint-rate command (solid) and feedforward (dashed)")
    axes[1, 0].set_ylabel("rate [rad/s]")
    axes[1, 0].legend(ncol=3, fontsize=8)

    insertion_command = np.asarray(
        [row["insertion_rate_command"] for row in rows]
    )
    insertion_input_reference = np.asarray(
        [row["insertion_rate_reference"] for row in rows]
    )
    axes[1, 1].plot(
        time_values, 1.0e3 * insertion_input_reference, "--", label="reference"
    )
    axes[1, 1].plot(
        time_values, 1.0e3 * insertion_command, label="command"
    )
    axes[1, 1].set_title("Insertion-rate tracking")
    axes[1, 1].set_ylabel("rate [mm/s]")
    axes[1, 1].legend()

    beam_position = np.asarray(
        [row["beam_position_error_mm"] for row in rows], dtype=float
    )
    beam_tangent = np.asarray(
        [row["beam_tangent_error_deg"] for row in rows], dtype=float
    )
    axes[2, 0].plot(time_values, beam_position, label="position")
    axes[2, 0].axhline(
        1.0e3 * result.summary["position_tolerance_m"],
        color="tab:red",
        linestyle="--",
        label="tolerance",
    )
    axes[2, 0].set_title("Nonlinear beam position validation")
    axes[2, 0].set_ylabel("error [mm]")
    axes[2, 0].legend()

    axes[2, 1].plot(time_values, beam_tangent, color="tab:orange", label="tangent")
    axes[2, 1].axhline(
        math.degrees(result.summary["tangent_tolerance_rad"]),
        color="tab:red",
        linestyle="--",
        label="tolerance",
    )
    axes[2, 1].set_title("Nonlinear beam tangent validation")
    axes[2, 1].set_ylabel("error [deg]")
    axes[2, 1].legend()

    solve_time = np.asarray([row["qp_solve_time_s"] for row in rows])
    axes[3, 0].plot(time_values, 1.0e3 * solve_time)
    axes[3, 0].axhline(
        1.0e3 * result.summary["mpc_configuration"]["sample_period_s"],
        color="tab:red",
        linestyle="--",
        label="sample period",
    )
    axes[3, 0].set_title("QP computation time")
    axes[3, 0].set_ylabel("time [ms]")
    axes[3, 0].legend()

    magnet = np.asarray(
        [[row["magnet_x"], row["magnet_y"], row["magnet_z"]] for row in rows],
        dtype=float,
    )
    if np.all(np.isfinite(magnet)):
        axes[3, 1].plot(time_values, 1.0e3 * (magnet - magnet[0]))
        axes[3, 1].legend(("x", "y", "z"), fontsize=8)
        axes[3, 1].set_title("Source-magnet translation from initial pose")
        axes[3, 1].set_ylabel("translation [mm]")
    else:
        axes[3, 1].text(0.5, 0.5, "Magnet FK unavailable", ha="center", va="center")
        axes[3, 1].set_title("Source-magnet translation")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("time [s]")
    figure.suptitle(
        "Time-scheduled joint/insertion MPC with nonlinear beam validation",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_configuration_mpc_simulation(
    result: ConfigurationSimulationResult,
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "configuration_mpc_simulation.csv"
    json_path = root / "configuration_mpc_simulation_summary.json"
    npz_path = root / "configuration_mpc_simulation.npz"
    plot_path = root / "configuration_mpc_simulation.png"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result.records[0].keys()))
        writer.writeheader()
        writer.writerows(result.records)
    json_path.write_text(
        json.dumps(_json_ready(result.summary), indent=2), encoding="utf-8"
    )
    actual_state = np.asarray(
        [
            [row[f"{name}_actual"] for name in STATE_NAMES]
            for row in result.records
        ],
        dtype=float,
    )
    reference_state = np.asarray(
        [
            [row[f"{name}_reference"] for name in STATE_NAMES]
            for row in result.records
        ],
        dtype=float,
    )
    command = np.asarray(
        [
            [row[f"{name}_command"] for name in INPUT_NAMES]
            for row in result.records
        ],
        dtype=float,
    )
    reference_input = np.asarray(
        [
            [row[f"{name}_reference"] for name in INPUT_NAMES]
            for row in result.records
        ],
        dtype=float,
    )
    achieved_position = np.asarray(
        [
            [row[f"achieved_{axis}"] for axis in ("x", "y", "z")]
            for row in result.records
        ],
        dtype=float,
    )
    achieved_tangent = np.asarray(
        [
            [row[f"achieved_tangent_{axis}"] for axis in ("x", "y", "z")]
            for row in result.records
        ],
        dtype=float,
    )
    np.savez_compressed(
        npz_path,
        time_s=np.asarray([row["time_s"] for row in result.records]),
        reference_index=np.asarray(
            [row["reference_index"] for row in result.records], dtype=int
        ),
        actual_state=actual_state,
        reference_state=reference_state,
        command=command,
        reference_input=reference_input,
        achieved_position_m=achieved_position,
        achieved_tangent=achieved_tangent,
        beam_position_error_m=np.asarray(
            [row["beam_position_error_m"] for row in result.records]
        ),
        beam_tangent_error_rad=np.asarray(
            [row["beam_tangent_error_rad"] for row in result.records]
        ),
    )
    try:
        _save_simulation_plot(result, plot_path)
    except Exception as exc:
        result.summary["plot_error"] = f"{type(exc).__name__}: {exc}"
        json_path.write_text(
            json.dumps(_json_ready(result.summary), indent=2), encoding="utf-8"
        )
    files = {
        "csv": str(csv_path),
        "json": str(json_path),
        "npz": str(npz_path),
        "plot": str(plot_path),
    }
    result.output_files.update(files)
    return files


def _controller_period(controller_pack: dict[str, Any]) -> float | None:
    controller = controller_pack.get("controller")
    value = getattr(controller, "dt", None)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None


def make_default_mpc_config(
    *,
    reference: ConfigurationReference,
    controller_pack: dict[str, Any],
    prediction_horizon: int = 15,
    joint_acceleration_limit_rad_s2: float = 0.5,
    insertion_acceleration_limit_m_s2: float = 0.02,
    velocity_safety_factor: float = 0.8,
    acceleration_safety_factor: float = 0.8,
    solver_backend: Literal["auto", "osqp", "scipy"] = "auto",
    solver_verbose: bool = False,
) -> ConfigurationMPCConfig:
    """Build defaults from the same limits used by the existing controller."""
    if not 0.0 < velocity_safety_factor <= 1.0:
        raise ValueError("velocity_safety_factor must lie in (0, 1].")
    if not 0.0 < acceleration_safety_factor <= 1.0:
        raise ValueError("acceleration_safety_factor must lie in (0, 1].")
    state_min = _finite_vector(controller_pack["p_min"], 7, "p_min")
    state_max = _finite_vector(controller_pack["p_max"], 7, "p_max")
    physical_velocity = _finite_vector(controller_pack["u_max"], 7, "u_max")
    velocity_limit = float(velocity_safety_factor) * physical_velocity
    physical_acceleration = np.asarray(
        [float(joint_acceleration_limit_rad_s2)] * 6
        + [float(insertion_acceleration_limit_m_s2)],
        dtype=float,
    )
    acceleration_limit = (
        float(acceleration_safety_factor) * physical_acceleration
    )
    return ConfigurationMPCConfig(
        sample_period_s=reference.sample_period_s,
        prediction_horizon=int(prediction_horizon),
        state_min=tuple(state_min),
        state_max=tuple(state_max),
        velocity_limit=tuple(velocity_limit),
        acceleration_limit=tuple(acceleration_limit),
        solver_backend=solver_backend,
        solver_verbose=bool(solver_verbose),
    )


def run_self_test() -> None:
    dt = 0.05
    time_values = np.arange(0.0, 1.0 + 0.5 * dt, dt)
    phase = 0.5 * (1.0 - np.cos(np.pi * time_values / time_values[-1]))
    phase_rate = (
        0.5
        * np.pi
        / time_values[-1]
        * np.sin(np.pi * time_values / time_values[-1])
    )
    phase_acceleration = (
        0.5
        * (np.pi / time_values[-1]) ** 2
        * np.cos(np.pi * time_values / time_values[-1])
    )
    displacement = np.asarray([0.05, -0.03, 0.02, 0.01, -0.02, 0.03, 0.002])
    state = phase[:, None] * displacement[None, :]
    input_values = phase_rate[:, None] * displacement[None, :]
    acceleration = phase_acceleration[:, None] * displacement[None, :]
    desired = np.column_stack((phase, np.zeros_like(phase), np.zeros_like(phase)))
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (time_values.size, 1))
    reference = ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=0.04 * phase,
        state=state,
        input=input_values,
        acceleration=acceleration,
        desired_position_m=desired,
        desired_tangent=tangent,
        planned_beam_feasible=np.ones(time_values.size, dtype=bool),
        planned_position_error_m=np.zeros(time_values.size),
        planned_tangent_error_rad=np.zeros(time_values.size),
        sample_period_s=dt,
        source="self_test",
    )
    config = ConfigurationMPCConfig(
        sample_period_s=dt,
        prediction_horizon=5,
        state_min=tuple([-1.0] * 6 + [-0.1]),
        state_max=tuple([1.0] * 6 + [0.1]),
        velocity_limit=tuple([0.5] * 6 + [0.05]),
        acceleration_limit=tuple([1.0] * 6 + [0.05]),
        solver_backend="scipy" if osqp is None else "osqp",
        solver_absolute_tolerance=1.0e-8,
        solver_relative_tolerance=1.0e-8,
    )
    controller = ConfigurationTrackingMPC(reference=reference, config=config)
    actual = state[0].copy()
    previous = np.zeros(7, dtype=float)
    maximum_error = 0.0
    for index in range(reference.sample_count - 1 + 5):
        result = controller.solve(
            measured_state=actual,
            control_index=index,
            previous_input=previous,
        )
        if not result.success:
            raise AssertionError(f"Self-test QP failed: {result.status}")
        actual = actual + dt * result.command
        previous = result.command.copy()
        reference_index = min(index + 1, reference.sample_count - 1)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(actual - state[reference_index]))),
        )
    if maximum_error > 5.0e-3:
        raise AssertionError(f"Self-test tracking error is too large: {maximum_error}")
    if np.max(np.abs(previous)) > 5.0e-4:
        raise AssertionError("Self-test did not settle the input near zero.")
    print(
        "[CONFIGURATION MPC SELF TEST] PASS "
        f"backend={controller.backend} maximum_error={maximum_error:.6e}"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate fixed-time joint/insertion MPC tracking with the existing "
            "nonlinear magnetic-beam plant."
        )
    )
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--terminal-hold-steps", type=int, default=15)
    parser.add_argument("--joint-acceleration-limit", type=float, default=0.5)
    parser.add_argument("--insertion-acceleration-limit", type=float, default=0.02)
    parser.add_argument("--velocity-safety-factor", type=float, default=0.8)
    parser.add_argument("--acceleration-safety-factor", type=float, default=0.8)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--tangent-tolerance-deg", type=float, default=90.0)
    parser.add_argument(
        "--initial-joint-offset-deg", type=float, nargs=6, default=(0.0,) * 6
    )
    parser.add_argument("--initial-insertion-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--progress-stride", type=int, default=25)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--skip-beam-evaluation", action="store_true")
    parser.add_argument("--continue-on-beam-violation", action="store_true")
    parser.add_argument("--continue-on-qp-failure", action="store_true")
    parser.add_argument("--allow-planned-beam-failure", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.horizon < 1:
        parser.error("--horizon must be at least one")
    if arguments.terminal_hold_steps < 0:
        parser.error("--terminal-hold-steps cannot be negative")
    if arguments.progress_stride < 1:
        parser.error("--progress-stride must be positive")
    if arguments.max_control_steps is not None and arguments.max_control_steps < 1:
        parser.error("--max-control-steps must be positive")
    return arguments


def main() -> None:
    arguments = _arguments()
    if arguments.self_test:
        run_self_test()
        return

    from proper_research.planning.planning_context import build_planning_context

    _, _, controller_pack, out_root = build_planning_context()
    reference_source = (
        out_root / "time_parameterized_configuration_path"
    )
    reference = load_configuration_reference(
        reference_source,
        require_planned_beam_feasible=not arguments.allow_planned_beam_failure,
    )
    controller_period = _controller_period(controller_pack)
    if controller_period is not None and not np.isclose(
        controller_period,
        reference.sample_period_s,
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise RuntimeError(
            "Time-reference dt does not match the simulation controller dt: "
            f"{reference.sample_period_s:.9f} versus {controller_period:.9f} s."
        )
    mpc_config = make_default_mpc_config(
        reference=reference,
        controller_pack=controller_pack,
        prediction_horizon=arguments.horizon,
        joint_acceleration_limit_rad_s2=arguments.joint_acceleration_limit,
        insertion_acceleration_limit_m_s2=arguments.insertion_acceleration_limit,
        velocity_safety_factor=arguments.velocity_safety_factor,
        acceleration_safety_factor=arguments.acceleration_safety_factor,
        solver_backend=arguments.solver,
        solver_verbose=arguments.solver_verbose,
    )
    simulation_config = ConfigurationSimulationConfig(
        position_tolerance_m=1.0e-3 * float(arguments.position_tolerance_mm),
        tangent_tolerance_rad=math.radians(float(arguments.tangent_tolerance_deg)),
        terminal_hold_steps=int(arguments.terminal_hold_steps),
        stop_on_qp_failure=not arguments.continue_on_qp_failure,
        stop_on_beam_violation=not arguments.continue_on_beam_violation,
        require_planned_beam_feasible=not arguments.allow_planned_beam_failure,
        evaluate_nonlinear_beam=not arguments.skip_beam_evaluation,
        maximum_control_steps=arguments.max_control_steps,
        progress_stride=int(arguments.progress_stride),
    )
    initial_offset = np.concatenate(
        (
            np.radians(np.asarray(arguments.initial_joint_offset_deg, dtype=float)),
            np.array([1.0e-3 * float(arguments.initial_insertion_offset_mm)]),
        )
    )
    initial_state = np.asarray(reference.state[0], dtype=float) + initial_offset
    output_directory = arguments.output_dir or (
        out_root / "time_parameterized_configuration_mpc_simulation"
    )
    print(
        "[CONFIGURATION MPC SETUP] "
        f"reference={reference.source} samples={reference.sample_count} "
        f"duration={reference.duration_s:.3f} s dt={reference.sample_period_s:.4f} s "
        f"horizon={mpc_config.prediction_horizon} "
        f"output={output_directory}",
        flush=True,
    )
    print(
        "[CONFIGURATION MPC LIMITS] "
        f"velocity={np.asarray(mpc_config.velocity_limit)} "
        f"acceleration={np.asarray(mpc_config.acceleration_limit)}",
        flush=True,
    )
    started = time.perf_counter()
    result = simulate_time_parameterized_configuration_mpc(
        reference=reference,
        controller_pack=controller_pack,
        mpc_config=mpc_config,
        simulation_config=simulation_config,
        output_dir=output_directory,
        initial_state=initial_state,
    )
    elapsed = time.perf_counter() - started
    summary = result.summary
    print(f"Elapsed wall time: {elapsed:.3f} s")
    print(f"Simulation success: {summary['success']}")
    print(f"Stopped reason: {summary['stopped_reason']}")
    print(f"All QPs solved: {summary['all_qp_solved']}")
    print(f"All beam samples feasible: {summary['all_beam_samples_feasible']}")
    print(
        "Maximum joint tracking error: "
        f"{summary['maximum_absolute_joint_error_deg']:.6f} deg"
    )
    print(
        "Maximum insertion tracking error: "
        f"{summary['maximum_absolute_insertion_error_mm']:.6f} mm"
    )
    print(
        "Maximum beam position error: "
        f"{summary['maximum_beam_position_error_mm']:.6f} mm"
    )
    print(
        "Maximum beam tangent error: "
        f"{summary['maximum_beam_tangent_error_deg']:.6f} deg"
    )
    print(
        "Mean / maximum QP solve time: "
        f"{summary['mean_qp_solve_time_ms']:.3f} / "
        f"{summary['maximum_qp_solve_time_ms']:.3f} ms"
    )
    print(f"Saved simulation: {output_directory}")
    if not bool(summary["success"]):
        raise RuntimeError(
            "Configuration MPC simulation did not meet its completion criteria; "
            f"reason={summary['stopped_reason']}."
        )


if __name__ == "__main__":
    main()


__all__ = [
    "ConfigurationMPCConfig",
    "ConfigurationMPCStep",
    "ConfigurationReference",
    "ConfigurationSimulationConfig",
    "ConfigurationSimulationResult",
    "ConfigurationTrackingMPC",
    "load_configuration_reference",
    "make_default_mpc_config",
    "save_configuration_mpc_simulation",
    "simulate_time_parameterized_configuration_mpc",
]
