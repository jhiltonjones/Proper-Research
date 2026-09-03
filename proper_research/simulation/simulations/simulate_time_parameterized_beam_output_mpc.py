"""Time-scheduled beam-output MPC with analytical beam Jacobians.

This module extends ``simulate_time_parameterized_configuration_mpc.py``.
The saved time parameterisation remains the nominal, actuator-feasible
feedforward trajectory, but beam-tip position is the primary feedback output.

Offline (before the control loop), an analytical beam-position Jacobian is
evaluated at every saved reference configuration::

    J_p[k] = d p_tip / d z | z_ref[k],   J_p[k] in R^(3 x 7)

The Jacobian is taken from the same exact interface used by
``offline_inverse_configuration.py``::

    controller_pack["plant_diagnostic_joint_adapter"]\
        .continuous_output_jacobian(z_ref[k])[:3, :]

The adapter returns the continuous 6-by-7 derivative of
``[tip_xyz, tangent_xyz]`` with respect to
``[q1, q2, q3, q4, q5, q6, insertion]``.  The MPC uses its first three rows,
not the tolerance-scaled inverse-objective Jacobian and not ``dt * J``.

Online, the controller measures the actuator configuration and beam-tip
position.  It estimates the local simulation-to-measurement residual and uses
the affine output model

    p_hat[j|k] = p_nom[k+j]
                 + J_p[k+j] (z[j|k] - z_ref[k+j])
                 + d_hat[k]

inside the condensed QP.  The actuator prediction remains

    z[k+1] = z[k] + dt v[k].

Only the first optimized velocity is applied, then the beam is measured and
the residual is estimated again.  No numerical/finite-difference Jacobian is
implemented in this file: failure to resolve the project's analytical
Jacobian is a hard error.

Typical project use
-------------------
Place this file beside ``simulate_time_parameterized_configuration_mpc.py``
and run it as a module.  The normal project route uses
``controller_pack['plant_diagnostic_joint_adapter']`` automatically.  An
explicit import path remains available only as an integration override::

    --beam-jacobian-provider package.module:function_name

The provider must return the beam-tip position Jacobian with state ordering

    [q1, q2, q3, q4, q5, q6, insertion]

and shape ``(3, 7)``.  A full ``(6, 7)`` position/tangent output Jacobian is
also accepted; its first three rows are used.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from scipy.optimize import LinearConstraint, minimize


Array = np.ndarray


def _load_configuration_mpc_module() -> ModuleType:
    """Import the existing controller, with a local-file fallback for review."""
    module_name = (
        "proper_research.simulation.simulations."
        "simulate_time_parameterized_configuration_mpc"
    )
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as project_error:
        candidates = (
            Path(__file__).with_name(
                "simulate_time_parameterized_configuration_mpc.py"
            ),
            Path(__file__).with_name(
                "simulate_time_parameterized_configuration_mpc(3).py"
            ),
            Path(__file__).with_name(
                "simulate_time_parameterized_configuration_mpc(4).py"
            ),
        )
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location(
                "_time_parameterized_configuration_mpc_base", path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
        raise ModuleNotFoundError(
            "Could not import simulate_time_parameterized_configuration_mpc. "
            "Place this script beside the existing configuration-MPC script."
        ) from project_error


_base = _load_configuration_mpc_module()

ConfigurationReference = _base.ConfigurationReference
ConfigurationMPCConfig = _base.ConfigurationMPCConfig
ConfigurationMPCStep = _base.ConfigurationMPCStep
ConfigurationSimulationConfig = _base.ConfigurationSimulationConfig
ConfigurationSimulationResult = _base.ConfigurationSimulationResult


def _finite_vector(value: Any, size: int, name: str) -> Array:
    return _base._finite_vector(value, size, name)


def _validate_reference_provenance(
    reference: ConfigurationReference,
    *,
    require_feasible: bool,
) -> None:
    """Reject an explicitly infeasible selected global-path candidate.

    Global optimizer convergence and candidate feasibility are deliberately
    separate.  A non-converged optimizer candidate may be used after exact
    node/dense validation, but an explicitly infeasible selected candidate may
    not enter beam-output MPC.  Older references without this additive
    metadata remain supported by the existing feasibility checks.
    """
    if not require_feasible:
        return
    metadata = getattr(reference, "metadata", {})
    summary = metadata.get("summary", {}) if isinstance(metadata, dict) else {}
    selected_feasible = (
        summary.get("source_selected_candidate_feasible")
        if isinstance(summary, dict)
        else None
    )
    if selected_feasible is not None and not bool(selected_feasible):
        raise RuntimeError(
            "The time reference was generated from an infeasible selected "
            "global-path candidate."
        )


@dataclass(frozen=True)
class BeamOutputMPCConfig:
    """Weights and estimation settings for beam-tip position feedback."""

    position_error_scale_m: tuple[float, float, float] = (
        0.5e-3,
        0.5e-3,
        0.5e-3,
    )
    position_tracking_weight: float = 1.0
    # Used only when ``use_dare_terminal_cost`` is false.
    terminal_weight_multiplier: float = 20.0
    use_dare_terminal_cost: bool = True
    # Infinity norm of final velocity divided componentwise by input scale.
    terminal_equilibrium_tolerance: float = 1.0e-6
    disturbance_filter_alpha: float = 0.0
    joint_trust_region_rad: float | None = None
    insertion_trust_region_m: float | None = None

    def validate(self) -> None:
        scale = _finite_vector(
            self.position_error_scale_m, 3, "position_error_scale_m"
        )
        if np.any(scale <= 0.0):
            raise ValueError("position_error_scale_m must be positive.")
        for name in ("position_tracking_weight", "terminal_weight_multiplier"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.terminal_weight_multiplier < 1.0:
            raise ValueError("terminal_weight_multiplier must be at least one.")
        equilibrium_tolerance = float(self.terminal_equilibrium_tolerance)
        if not np.isfinite(equilibrium_tolerance) or equilibrium_tolerance <= 0.0:
            raise ValueError(
                "terminal_equilibrium_tolerance must be finite and positive."
            )
        alpha = float(self.disturbance_filter_alpha)
        if not np.isfinite(alpha) or not 0.0 <= alpha < 1.0:
            raise ValueError("disturbance_filter_alpha must lie in [0, 1).")
        for name in ("joint_trust_region_rad", "insertion_trust_region_m"):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive when supplied.")

    @property
    def trust_region(self) -> Array | None:
        if (
            self.joint_trust_region_rad is None
            and self.insertion_trust_region_m is None
        ):
            return None
        if (
            self.joint_trust_region_rad is None
            or self.insertion_trust_region_m is None
        ):
            raise ValueError(
                "Both joint and insertion trust-region values are required."
            )
        return np.asarray(
            [float(self.joint_trust_region_rad)] * 6
            + [float(self.insertion_trust_region_m)],
            dtype=float,
        )


@dataclass
class BeamOutputMPCStep(ConfigurationMPCStep):
    measured_beam_position: Array
    instantaneous_output_residual: Array
    estimated_output_residual: Array
    predicted_beam_positions: Array
    predicted_beam_errors: Array
    first_predicted_beam_error_m: float


def _load_explicit_provider(import_path: str) -> Callable[..., Any]:
    if ":" not in import_path:
        raise ValueError(
            "--beam-jacobian-provider must be 'package.module:attribute'."
        )
    module_name, attribute_path = import_path.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for name in attribute_path.split("."):
        value = getattr(value, name)
    if not callable(value):
        raise TypeError(f"Analytical Jacobian provider is not callable: {import_path}")
    return value


def resolve_analytical_beam_jacobian(
    *,
    controller_pack: dict[str, Any],
    explicit_import_path: str | None = None,
) -> tuple[Callable[..., Any], str]:
    """Resolve the inverse planner's analytical Jacobian interface.

    The primary route is deliberately identical to
    ``offline_inverse_configuration._NodeObjective.output_jacobian``:
    ``plant_diagnostic_joint_adapter.continuous_output_jacobian(state)``.
    """
    if explicit_import_path:
        return _load_explicit_provider(explicit_import_path), explicit_import_path

    adapter = controller_pack.get("plant_diagnostic_joint_adapter")
    if adapter is not None:
        provider = getattr(adapter, "continuous_output_jacobian", None)
        if not callable(provider):
            raise TypeError(
                "controller_pack['plant_diagnostic_joint_adapter'] does not "
                "expose continuous_output_jacobian(state), which is the "
                "analytical Jacobian used by the inverse-path planner."
            )
        output_count = int(getattr(adapter, "n_out", 6))
        if output_count != 6:
            raise ValueError(
                "Beam-output MPC requires the full six-output diagnostic "
                "adapter used by the inverse-path planner; "
                f"received n_out={output_count}."
            )
        return (
            provider,
            "controller_pack['plant_diagnostic_joint_adapter']."
            "continuous_output_jacobian",
        )

    keys = (
        "analytical_beam_position_jacobian",
        "analytic_beam_position_jacobian",
        "beam_position_jacobian",
        "beam_tip_position_jacobian",
        "analytical_beam_jacobian",
        "analytic_beam_jacobian",
        "beam_jacobian",
    )
    for key in keys:
        provider = controller_pack.get(key)
        if callable(provider):
            return provider, f"controller_pack[{key!r}]"

    attribute_names = (
        "analytical_beam_position_jacobian",
        "analytic_beam_position_jacobian",
        "beam_position_jacobian",
        "beam_tip_position_jacobian",
        "analytical_jacobian",
    )
    for object_key in (
        "forward6d_plant",
        "beam_model",
        "beam_solver",
        "beam_theory",
    ):
        owner = controller_pack.get(object_key)
        if owner is None:
            continue
        for attribute in attribute_names:
            provider = getattr(owner, attribute, None)
            if callable(provider):
                return provider, f"controller_pack[{object_key!r}].{attribute}"

    jacobian_like = sorted(
        str(key)
        for key, value in controller_pack.items()
        if callable(value) and ("jac" in str(key).lower())
    )
    suffix = (
        " Jacobian-like callable keys found: " + ", ".join(jacobian_like)
        if jacobian_like
        else ""
    )
    raise KeyError(
        "controller_pack does not contain the inverse planner's "
        "'plant_diagnostic_joint_adapter'. Add that adapter to the planning "
        "context, expose an analytical beam-position Jacobian under a supported "
        "key, or pass --beam-jacobian-provider package.module:function_name. "
        "Numerical "
        f"differentiation is intentionally disabled.{suffix}"
    )


def _invoke_jacobian_provider(
    provider: Callable[..., Any],
    state: Array,
    controller_pack: dict[str, Any],
) -> Any:
    """Support common analytical-Jacobian call signatures."""
    state = _finite_vector(state, 7, "Jacobian state")
    attempts: list[tuple[str, Callable[[], Any]]] = [
        ("provider(state)", lambda: provider(state.copy())),
        (
            "provider(state, controller_pack)",
            lambda: provider(state.copy(), controller_pack),
        ),
        (
            "provider(q, insertion)",
            lambda: provider(state[:6].copy(), float(state[6])),
        ),
        (
            "provider(state=state)",
            lambda: provider(state=state.copy()),
        ),
        (
            "provider(configuration=state)",
            lambda: provider(configuration=state.copy()),
        ),
        (
            "provider(q=q, insertion=insertion)",
            lambda: provider(q=state[:6].copy(), insertion=float(state[6])),
        ),
    ]
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError):
        signature = None
    errors: list[str] = []
    for description, call in attempts:
        if signature is not None:
            try:
                if description == "provider(state)":
                    signature.bind(state.copy())
                elif description == "provider(state, controller_pack)":
                    signature.bind(state.copy(), controller_pack)
                elif description == "provider(q, insertion)":
                    signature.bind(state[:6].copy(), float(state[6]))
                elif description == "provider(state=state)":
                    signature.bind(state=state.copy())
                elif description == "provider(configuration=state)":
                    signature.bind(configuration=state.copy())
                else:
                    signature.bind(q=state[:6].copy(), insertion=float(state[6]))
            except TypeError:
                continue
        try:
            return call()
        except TypeError as exc:
            errors.append(f"{description}: {exc}")
            if signature is not None:
                raise
    raise TypeError(
        "Could not call the analytical beam Jacobian provider. Tried: "
        + "; ".join(errors)
    )


def _extract_position_jacobian(value: Any) -> Array:
    """Extract and validate a 3x7 position block from analytical output."""
    if isinstance(value, dict):
        for key in (
            "position_jacobian",
            "beam_position_jacobian",
            "tip_position_jacobian",
            "J_position",
            "Jp",
            "jacobian",
            "J",
        ):
            if key in value:
                try:
                    return _extract_position_jacobian(value[key])
                except (TypeError, ValueError):
                    pass
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _extract_position_jacobian(item)
            except (TypeError, ValueError):
                continue
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("Analytical Jacobian output is not numeric.") from exc
    if matrix.ndim != 2:
        raise ValueError(
            f"Analytical Jacobian must be a matrix, received shape {matrix.shape}."
        )
    if matrix.shape == (3, 7):
        result = matrix
    elif matrix.shape == (6, 7):
        result = matrix[:3]
    elif matrix.shape == (7, 3):
        result = matrix.T
    elif matrix.shape == (7, 6):
        result = matrix.T[:3]
    else:
        raise ValueError(
            "Analytical beam Jacobian must have shape (3, 7), (6, 7), "
            f"(7, 3), or (7, 6); received {matrix.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Analytical beam Jacobian contains non-finite values.")
    return np.asarray(result, dtype=float).copy()


def precompute_reference_beam_jacobians(
    *,
    reference: ConfigurationReference,
    provider: Callable[..., Any],
    controller_pack: dict[str, Any],
) -> Array:
    """Evaluate the project's analytical Jacobian along the full timed path."""
    jacobians = np.empty((reference.sample_count, 3, 7), dtype=float)
    for index, state in enumerate(np.asarray(reference.state, dtype=float)):
        raw = _invoke_jacobian_provider(provider, state, controller_pack)
        jacobians[index] = _extract_position_jacobian(raw)
    return jacobians


class BeamOutputTrackingMPC(_base.ConfigurationTrackingMPC):
    """Condensed QP with scheduled configuration and beam-position tracking."""

    def __init__(
        self,
        *,
        reference: ConfigurationReference,
        config: ConfigurationMPCConfig,
        beam_config: BeamOutputMPCConfig,
        reference_position_jacobians: Any,
        nominal_reference_positions_m: Any | None = None,
    ) -> None:
        beam_config.validate()
        super().__init__(reference=reference, config=config)
        jacobians = np.asarray(reference_position_jacobians, dtype=float)
        expected = (reference.sample_count, 3, 7)
        if jacobians.shape != expected or not np.all(np.isfinite(jacobians)):
            raise ValueError(
                "reference_position_jacobians must contain finite analytical "
                f"Jacobians with shape {expected}; received {jacobians.shape}."
            )
        if nominal_reference_positions_m is None:
            nominal_positions = np.asarray(
                reference.desired_position_m, dtype=float
            ).copy()
        else:
            nominal_positions = np.asarray(
                nominal_reference_positions_m, dtype=float
            )
        expected_position_shape = (reference.sample_count, 3)
        if (
            nominal_positions.shape != expected_position_shape
            or not np.all(np.isfinite(nominal_positions))
        ):
            raise ValueError(
                "nominal_reference_positions_m must have shape "
                f"{expected_position_shape}."
            )
        self.beam_config = beam_config
        self.reference_position_jacobians = jacobians.copy()
        self.nominal_reference_positions_m = nominal_positions.copy()
        self._filtered_output_residual: Array | None = None

        scale = _finite_vector(
            beam_config.position_error_scale_m, 3, "position_error_scale_m"
        )
        Qp = float(beam_config.position_tracking_weight) * np.diag(1.0 / scale**2)

        # ``ConfigurationTrackingMPC`` normally applies a heuristic multiplier
        # to its last state block.  In Riccati mode the last predicted state is
        # instead assigned the cost-to-go P[k + N], so retaining that block
        # would double-count the terminal state cost.  The input and increment
        # stage costs remain present for every optimized input.
        self.terminal_cost_schedule: Array | None = None
        self.terminal_feedback_schedule: Array | None = None
        self.terminal_dare_residual_norm = np.nan
        self.terminal_closed_loop_spectral_radius = np.nan
        self.terminal_equilibrium_normalized_input = np.nan
        self.last_terminal_cost_index: int | None = None
        if beam_config.use_dare_terminal_cost:
            Q = np.asarray(self.Qbar[: self.n, : self.n], dtype=float).copy()
            R = np.asarray(self.Rbar[: self.m, : self.m], dtype=float).copy()
            Rd = np.asarray(self.Rdbar[: self.m, : self.m], dtype=float).copy()
            terminal_slice = slice(
                (self.N - 1) * self.n,
                self.N * self.n,
            )
            self.Qbar[terminal_slice, terminal_slice] = 0.0
            base_hessian = 2.0 * (
                self.S.T @ self.Qbar @ self.S
                + self.Rbar
                + self.D.T @ self.Rdbar @ self.D
            )
            base_hessian += float(self.config.hessian_regularization) * np.eye(
                self.nu
            )
            self.H = 0.5 * (base_hessian + base_hessian.T)
        else:
            Q = np.asarray(self.Qbar[: self.n, : self.n], dtype=float).copy()
            R = np.asarray(self.Rbar[: self.m, : self.m], dtype=float).copy()
            Rd = np.asarray(self.Rdbar[: self.m, : self.m], dtype=float).copy()

        self._base_hessian = np.asarray(self.H, dtype=float).copy()
        blocks = [Qp.copy() for _ in range(self.N)]
        if beam_config.use_dare_terminal_cost:
            blocks[-1] = np.zeros_like(Qp)
        else:
            blocks[-1] *= float(beam_config.terminal_weight_multiplier)
        self.Qpbar = sp.block_diag(blocks, format="csc").toarray()
        if beam_config.use_dare_terminal_cost:
            (
                self.terminal_cost_schedule,
                self.terminal_feedback_schedule,
            ) = self._build_ltv_riccati_schedule(
                Q=Q,
                Qp=Qp,
                R=R,
                Rd=Rd,
            )
        if self.backend == "osqp":
            self._setup_variable_hessian_osqp()

    def _augmented_riccati_matrices(
        self,
        *,
        Q_effective: Array,
        R: Array,
        Rd: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Return the Markov model/cost for input-increment LQR.

        The augmented error state is ``[z - z_ref, u_previous - u_ref]`` and
        the tail input is ``w = u - u_previous``.  The cross term is required
        because the input-tracking cost is on ``u_previous + w``.
        """
        identity = np.eye(self.n, dtype=float)
        zero = np.zeros((self.n, self.n), dtype=float)
        A_aug = np.block(
            [
                [identity, self.dt * identity],
                [zero, identity],
            ]
        )
        B_aug = np.vstack((self.dt * identity, identity))
        Q_aug = np.block(
            [
                [Q_effective, zero],
                [zero, R],
            ]
        )
        N_aug = np.vstack((zero, R))
        R_aug = R + Rd
        minimum_r = float(np.min(np.linalg.eigvalsh(0.5 * (R_aug + R_aug.T))))
        if minimum_r <= 0.0:
            raise ValueError(
                "DARE terminal cost requires R + Rd to be positive definite; "
                f"minimum eigenvalue={minimum_r:.3e}."
            )
        return A_aug, B_aug, Q_aug, N_aug, R_aug

    @staticmethod
    def _symmetric(value: Array) -> Array:
        matrix = np.asarray(value, dtype=float)
        return 0.5 * (matrix + matrix.T)

    def _build_ltv_riccati_schedule(
        self,
        *,
        Q: Array,
        Qp: Array,
        R: Array,
        Rd: Array,
    ) -> tuple[Array, Array]:
        """Build P[i] backwards, seeded by a DARE at the final equilibrium.

        Only the quadratic cost-to-go is scheduled here.  Reference motion,
        nominal beam error and the estimated output residual create affine LQT
        terms; those remain handled explicitly over the online MPC horizon.
        """
        final_input = np.asarray(self.reference.input[-1], dtype=float)
        input_scale = _finite_vector(
            self.config.input_error_scale, self.m, "input_error_scale"
        )
        self.terminal_equilibrium_normalized_input = float(
            np.max(np.abs(final_input) / input_scale)
        )
        if (
            self.terminal_equilibrium_normalized_input
            > float(self.beam_config.terminal_equilibrium_tolerance)
        ):
            raise ValueError(
                "The final reference sample is not an equilibrium of "
                "z[k+1] = z[k] + dt*u[k]: its scaled final input infinity "
                "norm is "
                f"{self.terminal_equilibrium_normalized_input:.3e}, above "
                "terminal_equilibrium_tolerance="
                f"{self.beam_config.terminal_equilibrium_tolerance:.3e}. "
                "Append a zero-velocity terminal sample/hold before forming "
                "the DARE terminal cost."
            )

        sample_count = self.reference.sample_count
        augmented_size = 2 * self.n
        P_schedule = np.empty(
            (sample_count, augmented_size, augmented_size), dtype=float
        )
        K_schedule = np.empty(
            (sample_count, self.m, augmented_size), dtype=float
        )

        effective_weights = np.empty((sample_count, self.n, self.n), dtype=float)
        for index, jacobian in enumerate(self.reference_position_jacobians):
            effective_weights[index] = self._symmetric(
                Q + jacobian.T @ Qp @ jacobian
            )

        A_aug, B_aug, Q_final, N_aug, R_aug = (
            self._augmented_riccati_matrices(
                Q_effective=effective_weights[-1],
                R=R,
                Rd=Rd,
            )
        )
        try:
            P_final = solve_discrete_are(
                A_aug,
                B_aug,
                Q_final,
                R_aug,
                s=N_aug,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not solve the augmented terminal DARE at the final "
                "reference equilibrium."
            ) from exc
        P_schedule[-1] = self._symmetric(P_final)

        final_control_hessian = self._symmetric(
            R_aug + B_aug.T @ P_schedule[-1] @ B_aug
        )
        final_control_gradient = B_aug.T @ P_schedule[-1] @ A_aug + N_aug.T
        K_schedule[-1] = np.linalg.solve(
            final_control_hessian, final_control_gradient
        )
        closed_loop = A_aug - B_aug @ K_schedule[-1]
        self.terminal_closed_loop_spectral_radius = float(
            np.max(np.abs(np.linalg.eigvals(closed_loop)))
        )
        if self.terminal_closed_loop_spectral_radius >= 1.0:
            raise RuntimeError(
                "The final DARE solution is not stabilizing: closed-loop "
                "spectral radius="
                f"{self.terminal_closed_loop_spectral_radius:.6f}."
            )

        dare_residual = (
            A_aug.T @ P_schedule[-1] @ A_aug
            - P_schedule[-1]
            - (A_aug.T @ P_schedule[-1] @ B_aug + N_aug)
            @ K_schedule[-1]
            + Q_final
        )
        self.terminal_dare_residual_norm = float(
            np.linalg.norm(dare_residual, ord="fro")
        )

        for index in range(sample_count - 2, -1, -1):
            _, _, Q_aug, _, _ = self._augmented_riccati_matrices(
                Q_effective=effective_weights[index],
                R=R,
                Rd=Rd,
            )
            P_next = P_schedule[index + 1]
            control_hessian = self._symmetric(
                R_aug + B_aug.T @ P_next @ B_aug
            )
            control_gradient = B_aug.T @ P_next @ A_aug + N_aug.T
            gain = np.linalg.solve(control_hessian, control_gradient)
            P_current = (
                Q_aug
                + A_aug.T @ P_next @ A_aug
                - (A_aug.T @ P_next @ B_aug + N_aug) @ gain
            )
            P_schedule[index] = self._symmetric(P_current)
            K_schedule[index] = gain

        if not np.all(np.isfinite(P_schedule)):
            raise FloatingPointError(
                "The backward LTV Riccati schedule contains non-finite values."
            )
        minimum_p = min(
            float(np.min(np.linalg.eigvalsh(matrix))) for matrix in P_schedule
        )
        tolerance = 1.0e-8 * max(
            1.0,
            max(float(np.linalg.norm(matrix, ord=2)) for matrix in P_schedule),
        )
        if minimum_p < -tolerance:
            raise RuntimeError(
                "The backward LTV Riccati schedule is not positive "
                f"semidefinite; minimum eigenvalue={minimum_p:.3e}."
            )
        return P_schedule, K_schedule

    def _terminal_riccati_qp_terms(
        self,
        *,
        state: Array,
        state_reference: Array,
        input_reference: Array,
        control_index: int,
    ) -> tuple[Array, Array]:
        """Condense x_N' P[k + N] x_N into the current input sequence."""
        if self.terminal_cost_schedule is None:
            return np.zeros_like(self._base_hessian), np.zeros(self.nu)
        terminal_index = int(
            np.clip(
                int(control_index) + self.N,
                0,
                self.reference.sample_count - 1,
            )
        )
        self.last_terminal_cost_index = terminal_index
        P_terminal = self.terminal_cost_schedule[terminal_index]

        state_map = self.S[-self.n :, :]
        previous_input_map = np.zeros((self.m, self.nu), dtype=float)
        previous_input_map[:, -self.m :] = np.eye(self.m)
        terminal_map = np.vstack((state_map, previous_input_map))
        terminal_offset = np.concatenate(
            (
                state - np.asarray(state_reference[-1], dtype=float),
                -np.asarray(input_reference[-1], dtype=float),
            )
        )
        hessian = 2.0 * terminal_map.T @ P_terminal @ terminal_map
        linear = 2.0 * terminal_map.T @ P_terminal @ terminal_offset
        return self._symmetric(hessian), linear

    def _reference_indices(
        self, control_index: int, *, future: bool
    ) -> Array:
        offset = 1 if future else 0
        return np.clip(
            int(control_index) + offset + np.arange(self.N),
            0,
            self.reference.sample_count - 1,
        )

    def _estimate_output_residual(
        self,
        *,
        measured_state: Array,
        measured_beam_position: Array,
        control_index: int,
    ) -> tuple[Array, Array]:
        index = int(
            np.clip(control_index, 0, self.reference.sample_count - 1)
        )
        local_nominal = (
            self.nominal_reference_positions_m[index]
            + self.reference_position_jacobians[index]
            @ (measured_state - np.asarray(self.reference.state[index], dtype=float))
        )
        instantaneous = measured_beam_position - local_nominal
        alpha = float(self.beam_config.disturbance_filter_alpha)
        if self._filtered_output_residual is None:
            filtered = instantaneous.copy()
        else:
            filtered = (
                alpha * self._filtered_output_residual
                + (1.0 - alpha) * instantaneous
            )
        self._filtered_output_residual = filtered.copy()
        return instantaneous, filtered

    def _beam_prediction_terms(
        self,
        *,
        state: Array,
        control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        indices = self._reference_indices(control_index, future=True)
        state_reference = np.asarray(self.reference.state, dtype=float)[indices]
        desired_position = np.asarray(
            self.reference.desired_position_m, dtype=float
        )[indices]
        nominal_position = self.nominal_reference_positions_m[indices]
        jacobians = self.reference_position_jacobians[indices]
        Jbar = sp.block_diag(list(jacobians), format="csc").toarray()
        state_reference_vector = state_reference.reshape(self.nu)
        free_state = self.E @ state
        G = Jbar @ self.S
        constant_error = (
            nominal_position.reshape(3 * self.N)
            - desired_position.reshape(3 * self.N)
            + Jbar @ (free_state - state_reference_vector)
            + np.tile(estimated_residual, self.N)
        )
        return (
            G,
            constant_error,
            state_reference,
            desired_position,
            nominal_position,
            Jbar,
        )

    def _dynamic_qp_terms(
        self,
        *,
        state: Array,
        previous_input: Array,
        control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)
        base_linear = self._linear_cost(
            state=state,
            previous_input=previous_input,
            state_reference=state_reference,
            input_reference=input_reference,
        )
        (
            G,
            constant_error,
            _,
            desired_position,
            nominal_position,
            Jbar,
        ) = self._beam_prediction_terms(
            state=state,
            control_index=control_index,
            estimated_residual=estimated_residual,
        )
        hessian = self._base_hessian + 2.0 * (G.T @ self.Qpbar @ G)
        linear = base_linear + 2.0 * (G.T @ self.Qpbar @ constant_error)
        if self.beam_config.use_dare_terminal_cost:
            terminal_hessian, terminal_linear = (
                self._terminal_riccati_qp_terms(
                    state=state,
                    state_reference=state_reference,
                    input_reference=input_reference,
                    control_index=control_index,
                )
            )
            hessian = hessian + terminal_hessian
            linear = linear + terminal_linear
        hessian = 0.5 * (hessian + hessian.T)
        return hessian, linear, {
            "G": G,
            "constant_error": constant_error,
            "desired_position": desired_position,
            "nominal_position": nominal_position,
            "Jbar": Jbar,
            "input_reference": input_reference,
        }

    def _constraint_bounds_with_trust_region(
        self,
        *,
        state: Array,
        previous_input: Array,
        control_index: int,
    ) -> tuple[Array, Array]:
        lower, upper = self._constraint_bounds(
            state=state, previous_input=previous_input
        )
        trust = self.beam_config.trust_region
        if trust is None:
            return lower, upper
        reference_states = self.reference.state_window(control_index, self.N)
        free_state = (self.E @ state).reshape(self.N, self.n)
        trust_lower = (reference_states - trust[None, :] - free_state).reshape(
            self.nu
        )
        trust_upper = (reference_states + trust[None, :] - free_state).reshape(
            self.nu
        )
        lower[self._state_rows] = np.maximum(
            lower[self._state_rows], trust_lower
        )
        upper[self._state_rows] = np.minimum(
            upper[self._state_rows], trust_upper
        )
        return lower, upper

    def _upper_values(self, hessian: Array) -> Array:
        values = np.empty(self._p_pattern.nnz, dtype=float)
        for column in range(self.nu):
            start = self._p_pattern.indptr[column]
            stop = self._p_pattern.indptr[column + 1]
            rows = self._p_pattern.indices[start:stop]
            values[start:stop] = hessian[rows, column]
        return values

    def _setup_variable_hessian_osqp(self) -> None:
        if _base.osqp is None:
            raise ImportError("OSQP is unavailable.")
        self._p_pattern = sp.triu(
            sp.csc_matrix(np.ones((self.nu, self.nu), dtype=float)),
            format="csc",
        )
        values = self._upper_values(self._base_hessian)
        # Keep the complete upper-triangular sparsity pattern during setup.
        setup_values = values.copy()
        setup_values[np.abs(setup_values) < 1.0e-30] = 1.0e-30
        P = sp.csc_matrix(
            (
                setup_values,
                self._p_pattern.indices.copy(),
                self._p_pattern.indptr.copy(),
            ),
            shape=(self.nu, self.nu),
        )
        lower, upper = self._constraint_bounds(
            state=np.zeros(7), previous_input=np.zeros(7), validate_state=False
        )
        self._solver = _base.osqp.OSQP()
        self._solver.setup(
            P=P,
            q=np.zeros(self.nu),
            A=sp.csc_matrix(self.A),
            l=lower,
            u=upper,
            eps_abs=float(self.config.solver_absolute_tolerance),
            eps_rel=float(self.config.solver_relative_tolerance),
            max_iter=int(self.config.solver_maximum_iterations),
            polish=bool(self.config.solver_polish),
            verbose=bool(self.config.solver_verbose),
            warm_start=True,
        )

    def _solve_scipy_dynamic(
        self,
        *,
        hessian: Array,
        linear_cost: Array,
        lower: Array,
        upper: Array,
        warm_start: Array,
    ) -> tuple[Array | None, dict[str, Any]]:
        constraint = LinearConstraint(self.A, lower, upper)

        def objective(value: Array) -> float:
            return float(0.5 * value @ hessian @ value + linear_cost @ value)

        def gradient(value: Array) -> Array:
            return hessian @ value + linear_cost

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
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> BeamOutputMPCStep:
        """Solve one output-aware receding-horizon QP."""
        state = _finite_vector(measured_state, 7, "measured_state")
        beam_position = _finite_vector(
            measured_beam_position, 3, "measured_beam_position"
        )
        previous = _finite_vector(previous_input, 7, "previous_input")
        instantaneous, estimated = self._estimate_output_residual(
            measured_state=state,
            measured_beam_position=beam_position,
            control_index=control_index,
        )
        hessian, linear_cost, beam_terms = self._dynamic_qp_terms(
            state=state,
            previous_input=previous,
            control_index=control_index,
            estimated_residual=estimated,
        )
        input_reference = beam_terms["input_reference"]
        lower, upper = self._constraint_bounds_with_trust_region(
            state=state,
            previous_input=previous,
            control_index=control_index,
        )
        warm_start = self._feasible_warm_start(
            input_reference=input_reference, previous_input=previous
        )

        if self.backend == "osqp":
            assert self._solver is not None
            self._solver.update(
                Px=self._upper_values(hessian),
                q=linear_cost,
                l=lower,
                u=upper,
            )
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
            solution, diagnostic = self._solve_scipy_dynamic(
                hessian=hessian,
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
                0.5 * solution @ hessian @ solution + linear_cost @ solution
            )
            self._warm_start = solution.copy()

        predicted_beam_vector = (
            beam_terms["nominal_position"].reshape(3 * self.N)
            + beam_terms["Jbar"]
            @ (
                predicted_states.reshape(self.nu)
                - self.reference.state_window(control_index, self.N).reshape(
                    self.nu
                )
            )
            + np.tile(estimated, self.N)
        )
        predicted_beam = predicted_beam_vector.reshape(self.N, 3)
        predicted_error = predicted_beam - beam_terms["desired_position"]
        first_error = float(np.linalg.norm(predicted_error[0]))

        return BeamOutputMPCStep(
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
            measured_beam_position=beam_position.copy(),
            instantaneous_output_residual=instantaneous.copy(),
            estimated_output_residual=estimated.copy(),
            predicted_beam_positions=predicted_beam,
            predicted_beam_errors=predicted_error,
            first_predicted_beam_error_m=first_error,
        )


def _add_output_diagnostics(
    record: dict[str, Any], step: BeamOutputMPCStep | None
) -> None:
    axes = ("x", "y", "z")
    for coordinate, axis in enumerate(axes):
        record[f"beam_residual_{axis}_m"] = (
            np.nan
            if step is None
            else float(step.estimated_output_residual[coordinate])
        )
        record[f"beam_residual_instantaneous_{axis}_m"] = (
            np.nan
            if step is None
            else float(step.instantaneous_output_residual[coordinate])
        )
        record[f"predicted_next_beam_{axis}_m"] = (
            np.nan
            if step is None
            else float(step.predicted_beam_positions[0, coordinate])
        )
        record[f"predicted_next_beam_error_{axis}_m"] = (
            np.nan
            if step is None
            else float(step.predicted_beam_errors[0, coordinate])
        )
    record["beam_residual_norm_m"] = (
        np.nan
        if step is None
        else float(np.linalg.norm(step.estimated_output_residual))
    )
    record["predicted_next_beam_error_m"] = (
        np.nan if step is None else float(step.first_predicted_beam_error_m)
    )


def _output_summary(
    *,
    records: list[dict[str, Any]],
    reference: ConfigurationReference,
    mpc_config: ConfigurationMPCConfig,
    beam_config: BeamOutputMPCConfig,
    simulation_config: ConfigurationSimulationConfig,
    expected_control_steps: int,
    stopped_reason: str,
    jacobian_source: str,
) -> dict[str, Any]:
    summary = _base._summarize_simulation(
        records=records,
        reference=reference,
        mpc_config=mpc_config,
        simulation_config=simulation_config,
        expected_control_steps=expected_control_steps,
        stopped_reason=stopped_reason,
    )
    final = records[-1]
    final_beam_satisfied = bool(final["beam_feasible"])
    completed = bool(summary["completed_requested_steps"])
    all_qp_solved = bool(summary["all_qp_solved"])
    final_rate_near_zero = bool(summary["terminal_rate_near_zero"])
    summary.update(
        {
            "success": bool(
                completed
                and all_qp_solved
                and final_beam_satisfied
                and final_rate_near_zero
            ),
            "terminal_beam_tracking_satisfied": final_beam_satisfied,
            "nominal_terminal_configuration_satisfied": bool(
                summary["terminal_tracking_satisfied"]
            ),
            "beam_output_mpc_configuration": _base._json_ready(beam_config),
            "analytical_jacobian_source": str(jacobian_source),
            "maximum_estimated_beam_residual_m": float(
                np.nanmax(
                    np.asarray(
                        [row["beam_residual_norm_m"] for row in records],
                        dtype=float,
                    )
                )
            )
            if len(records) > 1
            else 0.0,
            "maximum_predicted_next_beam_error_m": float(
                np.nanmax(
                    np.asarray(
                        [
                            row["predicted_next_beam_error_m"]
                            for row in records
                        ],
                        dtype=float,
                    )
                )
            )
            if len(records) > 1
            else 0.0,
            "success_definition": (
                "completed, all QPs solved, terminal beam position/tangent "
                "within tolerance, and terminal command near zero"
            ),
        }
    )
    return summary


def simulate_time_parameterized_beam_output_mpc(
    *,
    reference: ConfigurationReference,
    controller_pack: dict[str, Any],
    mpc_config: ConfigurationMPCConfig,
    beam_config: BeamOutputMPCConfig,
    reference_position_jacobians: Any,
    simulation_config: ConfigurationSimulationConfig,
    output_dir: str | Path | None,
    initial_state: Any | None = None,
    jacobian_source: str = "analytical provider",
) -> ConfigurationSimulationResult:
    """Run output-aware MPC against the sequential nonlinear beam plant."""
    simulation_config.validate()
    reference.validate(
        require_planned_beam_feasible=(
            simulation_config.require_planned_beam_feasible
        )
    )
    _validate_reference_provenance(
        reference,
        require_feasible=simulation_config.require_planned_beam_feasible,
    )
    mpc_config.validate()
    beam_config.validate()
    if not simulation_config.evaluate_nonlinear_beam:
        raise ValueError(
            "Beam-output MPC requires beam measurements; do not use "
            "evaluate_nonlinear_beam=False."
        )
    controller_dt = getattr(controller_pack.get("controller"), "dt", None)
    if controller_dt is not None and not np.isclose(
        float(controller_dt),
        reference.sample_period_s,
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise ValueError(
            "Controller and time-reference sample periods differ: "
            f"{float(controller_dt):.9f} versus "
            f"{reference.sample_period_s:.9f}."
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
    if forward6d is None:
        raise KeyError("controller_pack does not contain 'forward6d_plant'.")

    mpc = BeamOutputTrackingMPC(
        reference=reference,
        config=mpc_config,
        beam_config=beam_config,
        reference_position_jacobians=reference_position_jacobians,
    )
    intended_steps = reference.sample_count - 1 + int(
        simulation_config.terminal_hold_steps
    )
    control_steps = intended_steps
    if simulation_config.maximum_control_steps is not None:
        control_steps = min(
            control_steps, int(simulation_config.maximum_control_steps)
        )

    previous_input = np.zeros(7, dtype=float)
    achieved_position, achieved_tangent = _base._beam_snapshot(
        forward6d=forward6d, state=state, commit=True
    )
    magnet_position, magnet_quaternion = _base._magnet_pose(
        controller_pack=controller_pack, state=state
    )
    initial_record = _base._make_record(
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
    _add_output_diagnostics(initial_record, None)
    records = [initial_record]
    stopped_reason = "completed"
    if (
        not bool(initial_record["beam_feasible"])
        and simulation_config.stop_on_beam_violation
    ):
        stopped_reason = "initial_beam_output_infeasible"
        result = ConfigurationSimulationResult(records=records, summary={})
        result.summary = _output_summary(
            records=records,
            reference=reference,
            mpc_config=mpc_config,
            beam_config=beam_config,
            simulation_config=simulation_config,
            expected_control_steps=control_steps,
            stopped_reason=stopped_reason,
            jacobian_source=jacobian_source,
        )
        if output_dir is not None:
            _base.save_configuration_mpc_simulation(result, output_dir)
        return result

    for control_index in range(control_steps):
        step = mpc.solve(
            measured_state=state,
            measured_beam_position=achieved_position,
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
        velocity_limit = _finite_vector(
            mpc_config.velocity_limit, 7, "velocity_limit"
        )
        acceleration_limit = _finite_vector(
            mpc_config.acceleration_limit, 7, "acceleration_limit"
        )
        input_violation = max(
            float(np.max(np.abs(command) - velocity_limit)), 0.0
        )
        acceleration = (
            command - previous_input
        ) / reference.sample_period_s
        acceleration_violation = max(
            float(np.max(np.abs(acceleration) - acceleration_limit)), 0.0
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
        reference_index = min(
            control_index + 1, reference.sample_count - 1
        )
        achieved_position, achieved_tangent = _base._beam_snapshot(
            forward6d=forward6d, state=state, commit=True
        )
        magnet_position, magnet_quaternion = _base._magnet_pose(
            controller_pack=controller_pack, state=state
        )
        record = _base._make_record(
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
        _add_output_diagnostics(record, step)
        records.append(record)

        if (
            control_index == 0
            or (control_index + 1) % simulation_config.progress_stride == 0
            or control_index + 1 == control_steps
        ):
            print(
                "[BEAM OUTPUT MPC] "
                f"sample={control_index + 1}/{control_steps} "
                f"reference={reference_index}/{reference.sample_count - 1} "
                f"status={step.status} "
                f"solve={1.0e3 * step.solve_time_s:.3f} ms "
                f"beam={record['beam_position_error_mm']:.4f} mm "
                f"residual={1.0e3 * np.linalg.norm(step.estimated_output_residual):.4f} mm "
                f"joint_error={np.degrees(np.max(np.abs(state[:6] - reference.state[reference_index, :6]))):.4f} deg",
                flush=True,
            )
        if (
            not bool(record["beam_feasible"])
            and simulation_config.stop_on_beam_violation
        ):
            stopped_reason = "nonlinear_beam_validation_failed"
            break

    result = ConfigurationSimulationResult(records=records, summary={})
    result.summary = _output_summary(
        records=records,
        reference=reference,
        mpc_config=mpc_config,
        beam_config=beam_config,
        simulation_config=simulation_config,
        expected_control_steps=control_steps,
        stopped_reason=stopped_reason,
        jacobian_source=jacobian_source,
    )
    if output_dir is not None:
        _base.save_configuration_mpc_simulation(result, output_dir)
    return result


def run_self_test() -> None:
    """Exercise analytical Jacobians, residual feedback, and variable Hessians."""
    dt = 0.1
    sample_count = 16
    time_values = np.arange(sample_count, dtype=float) * dt
    state_reference = np.zeros((sample_count, 7), dtype=float)
    input_reference = np.zeros_like(state_reference)
    desired_position = np.zeros((sample_count, 3), dtype=float)
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (sample_count, 1))
    reference = ConfigurationReference(
        time_s=time_values,
        path_coordinate_m=np.zeros(sample_count),
        state=state_reference,
        input=input_reference,
        acceleration=np.zeros_like(state_reference),
        desired_position_m=desired_position,
        desired_tangent=tangent,
        planned_beam_feasible=np.ones(sample_count, dtype=bool),
        planned_position_error_m=np.zeros(sample_count),
        planned_tangent_error_rad=np.zeros(sample_count),
        sample_period_s=dt,
        source="beam_output_self_test",
    )
    config = ConfigurationMPCConfig(
        sample_period_s=dt,
        prediction_horizon=5,
        state_min=tuple([-0.2] * 6 + [-0.02]),
        state_max=tuple([0.2] * 6 + [0.02]),
        velocity_limit=tuple([0.2] * 6 + [0.02]),
        acceleration_limit=tuple([1.0] * 6 + [0.1]),
        state_tracking_weight=1.0e-3,
        input_tracking_weight=1.0e-4,
        input_increment_weight=1.0e-4,
        solver_backend="osqp" if _base.osqp is not None else "scipy",
        solver_absolute_tolerance=1.0e-8,
        solver_relative_tolerance=1.0e-8,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(0.25e-3, 0.25e-3, 0.25e-3),
        position_tracking_weight=1.0,
        use_dare_terminal_cost=True,
    )
    jacobian_calls: list[Array] = []

    class AnalyticalDiagnosticAdapter:
        """Mirror the inverse planner's full six-output adapter API."""

        n_out = 6

        def continuous_output_jacobian(self, value: Array) -> Array:
            jacobian_calls.append(np.asarray(value, dtype=float).copy())
            matrix = np.zeros((6, 7), dtype=float)
            matrix[0, 0] = 1.0
            return matrix

    self_test_pack = {
        "plant_diagnostic_joint_adapter": AnalyticalDiagnosticAdapter()
    }
    provider, provider_source = resolve_analytical_beam_jacobian(
        controller_pack=self_test_pack
    )
    if "plant_diagnostic_joint_adapter" not in provider_source:
        raise AssertionError(
            "Self-test did not select the inverse planner's analytical adapter."
        )
    jacobians = precompute_reference_beam_jacobians(
        reference=reference,
        provider=provider,
        controller_pack=self_test_pack,
    )
    if len(jacobian_calls) != sample_count:
        raise AssertionError(
            "Analytical Jacobian was not evaluated once per reference sample."
        )
    controller = BeamOutputTrackingMPC(
        reference=reference,
        config=config,
        beam_config=beam_config,
        reference_position_jacobians=jacobians,
    )
    expected_schedule_shape = (sample_count, 14, 14)
    if (
        controller.terminal_cost_schedule is None
        or controller.terminal_cost_schedule.shape != expected_schedule_shape
    ):
        raise AssertionError(
            "Riccati schedule has the wrong shape; expected "
            f"{expected_schedule_shape}."
        )
    residual_scale = max(
        1.0,
        float(np.linalg.norm(controller.terminal_cost_schedule[-1], ord="fro")),
    )
    if controller.terminal_dare_residual_norm > 1.0e-7 * residual_scale:
        raise AssertionError(
            "The terminal DARE residual is too large: "
            f"{controller.terminal_dare_residual_norm:.3e}."
        )
    if controller.terminal_closed_loop_spectral_radius >= 1.0:
        raise AssertionError("The terminal DARE feedback is not stabilizing.")

    # The real beam has a 1 mm additive offset and an 0.8 slope, whereas the
    # analytical nominal Jacobian supplied to the MPC is 1.0 m/rad.
    state = np.zeros(7, dtype=float)
    previous = np.zeros(7, dtype=float)

    def measured_position(value: Array) -> Array:
        return np.array([0.8 * value[0] + 1.0e-3, 0.0, 0.0])

    initial_error = float(np.linalg.norm(measured_position(state)))
    maximum_velocity_ratio = 0.0
    for index in range(20):
        step = controller.solve(
            measured_state=state,
            measured_beam_position=measured_position(state),
            control_index=index,
            previous_input=previous,
        )
        if not step.success:
            raise AssertionError(f"Beam-output self-test QP failed: {step.status}")
        state = state + dt * step.command
        previous = step.command.copy()
        maximum_velocity_ratio = max(
            maximum_velocity_ratio,
            float(np.max(np.abs(previous) / np.asarray(config.velocity_limit))),
        )
    final_error = float(np.linalg.norm(measured_position(state)))
    if not final_error < 0.1 * initial_error:
        raise AssertionError(
            "Beam feedback did not reject the synthetic model mismatch: "
            f"initial={initial_error:.6e}, final={final_error:.6e}."
        )
    if maximum_velocity_ratio > 1.0 + 1.0e-6:
        raise AssertionError("Self-test violated a configured velocity limit.")
    expected_terminal_index = sample_count - 1
    if controller.last_terminal_cost_index != expected_terminal_index:
        raise AssertionError(
            "The controller did not select the clipped P[k + N] terminal "
            f"weight; expected index {expected_terminal_index}, received "
            f"{controller.last_terminal_cost_index}."
        )
    print(
        "[BEAM OUTPUT MPC SELF TEST] PASS "
        f"backend={controller.backend} "
        f"rho={controller.terminal_closed_loop_spectral_radius:.6f} "
        f"DARE_residual={controller.terminal_dare_residual_norm:.3e} "
        f"initial_error={1.0e3 * initial_error:.6f} mm "
        f"final_error={1.0e3 * final_error:.6f} mm"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track a fixed-time beam-tip trajectory using scheduled joint "
            "feedforward, measured beam residuals, and analytical beam "
            "position Jacobians."
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
    parser.add_argument("--beam-position-scale-mm", type=float, nargs=3, default=(0.5, 0.5, 0.5))
    parser.add_argument("--beam-position-weight", type=float, default=1.0)
    parser.add_argument(
        "--beam-terminal-multiplier",
        type=float,
        default=20.0,
        help="Legacy last-stage multiplier, used only with --disable-dare-terminal-cost.",
    )
    parser.add_argument(
        "--disable-dare-terminal-cost",
        action="store_true",
        help="Use the legacy last-stage multipliers instead of the scheduled Riccati tail.",
    )
    parser.add_argument("--state-tracking-weight", type=float, default=1.0e-2)
    parser.add_argument("--input-tracking-weight", type=float, default=1.0e-3)
    parser.add_argument("--input-increment-weight", type=float, default=1.0e-3)
    parser.add_argument("--disturbance-filter-alpha", type=float, default=0.0)
    parser.add_argument("--jacobian-trust-region-joint-deg", type=float, default=None)
    parser.add_argument("--jacobian-trust-region-insertion-mm", type=float, default=None)
    parser.add_argument("--beam-jacobian-provider", type=str, default=None)
    parser.add_argument(
        "--initial-joint-offset-deg", type=float, nargs=6, default=(0.0,) * 6
    )
    parser.add_argument("--initial-insertion-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--progress-stride", type=int, default=25)
    parser.add_argument("--solver", choices=("auto", "osqp", "scipy"), default="auto")
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--stop-on-beam-violation", action="store_true")
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
    trust_values = (
        arguments.jacobian_trust_region_joint_deg,
        arguments.jacobian_trust_region_insertion_mm,
    )
    if (trust_values[0] is None) != (trust_values[1] is None):
        parser.error(
            "Supply both Jacobian trust-region values or neither of them."
        )
    return arguments


def main() -> None:
    arguments = _arguments()
    if arguments.self_test:
        run_self_test()
        return

    from proper_research.planning.planning_context import build_planning_context

    _, _, controller_pack, out_root = build_planning_context()
    reference_source = arguments.reference or (
        out_root / "time_parameterized_configuration_path"
    )
    reference = _base.load_configuration_reference(
        reference_source,
        require_planned_beam_feasible=not arguments.allow_planned_beam_failure,
    )
    _validate_reference_provenance(
        reference,
        require_feasible=not arguments.allow_planned_beam_failure,
    )
    controller_period = _base._controller_period(controller_pack)
    if controller_period is not None and not np.isclose(
        controller_period,
        reference.sample_period_s,
        atol=1.0e-12,
        rtol=1.0e-10,
    ):
        raise RuntimeError(
            "Time-reference and controller sample periods differ: "
            f"{reference.sample_period_s:.9f} versus {controller_period:.9f}."
        )

    provider, provider_source = resolve_analytical_beam_jacobian(
        controller_pack=controller_pack,
        explicit_import_path=arguments.beam_jacobian_provider,
    )
    print(
        "[BEAM OUTPUT MPC OFFLINE] Precomputing analytical beam Jacobians "
        f"from {provider_source} for {reference.sample_count} samples...",
        flush=True,
    )
    reference_jacobians = precompute_reference_beam_jacobians(
        reference=reference,
        provider=provider,
        controller_pack=controller_pack,
    )

    mpc_config = _base.make_default_mpc_config(
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
    mpc_config = replace(
        mpc_config,
        state_tracking_weight=float(arguments.state_tracking_weight),
        input_tracking_weight=float(arguments.input_tracking_weight),
        input_increment_weight=float(arguments.input_increment_weight),
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=tuple(
            1.0e-3 * np.asarray(arguments.beam_position_scale_mm, dtype=float)
        ),
        position_tracking_weight=float(arguments.beam_position_weight),
        terminal_weight_multiplier=float(arguments.beam_terminal_multiplier),
        use_dare_terminal_cost=not bool(arguments.disable_dare_terminal_cost),
        disturbance_filter_alpha=float(arguments.disturbance_filter_alpha),
        joint_trust_region_rad=(
            None
            if arguments.jacobian_trust_region_joint_deg is None
            else math.radians(arguments.jacobian_trust_region_joint_deg)
        ),
        insertion_trust_region_m=(
            None
            if arguments.jacobian_trust_region_insertion_mm is None
            else 1.0e-3 * arguments.jacobian_trust_region_insertion_mm
        ),
    )
    simulation_config = ConfigurationSimulationConfig(
        position_tolerance_m=1.0e-3 * float(arguments.position_tolerance_mm),
        tangent_tolerance_rad=math.radians(float(arguments.tangent_tolerance_deg)),
        terminal_hold_steps=int(arguments.terminal_hold_steps),
        stop_on_qp_failure=not arguments.continue_on_qp_failure,
        stop_on_beam_violation=bool(arguments.stop_on_beam_violation),
        require_planned_beam_feasible=not arguments.allow_planned_beam_failure,
        evaluate_nonlinear_beam=True,
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
        out_root / "time_parameterized_beam_output_mpc_simulation"
    )
    print(
        "[BEAM OUTPUT MPC SETUP] "
        f"reference={reference.source} samples={reference.sample_count} "
        f"duration={reference.duration_s:.3f} s "
        f"dt={reference.sample_period_s:.4f} s "
        f"horizon={mpc_config.prediction_horizon} "
        f"output={output_directory}",
        flush=True,
    )
    started = time.perf_counter()
    result = simulate_time_parameterized_beam_output_mpc(
        reference=reference,
        controller_pack=controller_pack,
        mpc_config=mpc_config,
        beam_config=beam_config,
        reference_position_jacobians=reference_jacobians,
        simulation_config=simulation_config,
        output_dir=output_directory,
        initial_state=initial_state,
        jacobian_source=provider_source,
    )
    elapsed = time.perf_counter() - started
    summary = result.summary
    print(f"Elapsed wall time: {elapsed:.3f} s")
    print(f"Simulation success: {summary['success']}")
    print(f"Stopped reason: {summary['stopped_reason']}")
    print(f"All QPs solved: {summary['all_qp_solved']}")
    print(
        "Maximum / final beam position error: "
        f"{summary['maximum_beam_position_error_mm']:.6f} / "
        f"{1.0e3 * result.records[-1]['beam_position_error_m']:.6f} mm"
    )
    print(
        "Maximum estimated model residual: "
        f"{1.0e3 * summary['maximum_estimated_beam_residual_m']:.6f} mm"
    )
    print(
        "Mean / maximum QP solve time: "
        f"{summary['mean_qp_solve_time_ms']:.3f} / "
        f"{summary['maximum_qp_solve_time_ms']:.3f} ms"
    )
    print(f"Saved simulation: {output_directory}")
    if not bool(summary["success"]):
        raise RuntimeError(
            "Beam-output MPC simulation did not meet its completion criteria; "
            f"reason={summary['stopped_reason']}."
        )


if __name__ == "__main__":
    main()


__all__ = [
    "BeamOutputMPCConfig",
    "BeamOutputMPCStep",
    "BeamOutputTrackingMPC",
    "precompute_reference_beam_jacobians",
    "resolve_analytical_beam_jacobian",
    "run_self_test",
    "simulate_time_parameterized_beam_output_mpc",
]
