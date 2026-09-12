"""Resolved-rate inverse-Jacobian beam-tip controller.

The bottom rung of the ladder: one damped inverse of the beam Jacobian per
sample, no preview, no constraints beyond clipping.  It exists to price what
the MPCs above it are buying.

The control law
---------------
    e     = p*(k+1) - p_meas
    v_task = J^+ (kp * e / dt)                     damped least squares
    v_null = (I - J^+ J) kn (z_ref(k+1) - z) / dt
    v      = clip(v_task + v_null)

Two details are not decoration.

**The nullspace term is mandatory.**  The beam-position Jacobian is 3x7, so
four directions of joint motion leave the tip where it is.  Without a nullspace
term the configuration drifts along them until it hits a joint limit or a
singularity, and the comparison ends up measuring that drift rather than the
control law.  Pulling toward the reference configuration is the fairest choice:
it is the same trajectory the MPCs track in their state cost.

**Clipping is the same clipping the QPs get.**  Velocity limit, then
acceleration limit against the previous command, then the state box one step
ahead.  A controller allowed to move faster than its rivals is not a baseline,
it is a different experiment.  The order matters: the state box is applied last
and re-clipped to the velocity limit, because a bound that would require
exceeding the velocity limit to respect cannot be honoured in one step anyway.

Damping
-------
``damping`` is the Levenberg parameter in ``J^T (J J^T + lambda^2 I)^-1``.  It
bounds the command near a singularity at the cost of a steady-state error in the
directions J cannot reach.  The default is small; raise it if the run shows
large commands where the Jacobian's condition number spikes.

Selective damping (optional, 2026-09-12)
-----------------------------------------
The above is one ISOTROPIC lambda applied to all three output directions --
already fine for this system (weak-direction command fraction ~4%, see
beam-lateral-authority-limit memory) since DLS naturally damps small-sigma
directions more (the effect scales with sigma/(sigma^2+lambda^2)).
``selective_damping_gain`` (0 = disabled, the historical exact behaviour)
sharpens this further with an explicit per-singular-value extra penalty,
mirroring ``BeamOutputMPCConfig.directional_damping`` built for the MPC
variants: decompose J via SVD and add
``gain * (1/(sigma_i^2+floor^2) - min_i(...))`` on top of the isotropic
``damping**2`` for each singular direction i, so the near-null direction gets
extra suppression while the two well-conditioned directions are essentially
untouched (their extra weight is ~0 by construction). Plumbed in as an
option to compare against plain DLS, not because plain DLS was shown to need
it.

The controller returns a ``ConfigurationMPCStep``-shaped object, so it drops
into the same closed loop and the same record schema as the MPCs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:
    from proper_research.controllers.beam_jacobian_providers import declare_provider, provenance_line
except ImportError:  # standalone
    from proper_research.controllers.beam_jacobian_providers import declare_provider, provenance_line  # type: ignore

Array = np.ndarray


@dataclass
class InverseJacobianStep:
    """Mirrors the fields of ``ConfigurationMPCStep`` that the loop records."""

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
    # Beam-output diagnostics, so the record writer sees the same columns.
    measured_beam_position: Array
    instantaneous_output_residual: Array
    estimated_output_residual: Array
    predicted_beam_positions: Array
    predicted_beam_errors: Array
    first_predicted_beam_error_m: float


class InverseJacobianBeamController:
    """Damped resolved-rate control with nullspace configuration regulation."""

    name = "naive_inverse_jacobian"
    description = "Damped resolved-rate inverse Jacobian, nullspace-regulated"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        sample_period_s: float,
        velocity_limit: Any,
        acceleration_limit: Any,
        state_min: Any,
        state_max: Any,
        position_gain: float = 1.0,
        damping: float = 1.0e-3,
        nullspace_gain: float = 1.0,
        feedforward: bool = True,
        allow_undeclared_jacobian: bool = False,
        selective_damping_gain: float = 0.0,
        selective_damping_floor: float = 0.01,
    ) -> None:
        self.reference = reference
        # The controller records what it was handed. Ask any instance
        # `controller.jacobian_provenance` and it will tell you whether its
        # Jacobian is contact-free, rather than you having to remember.
        self.jacobian_provider, self.jacobian_provenance = declare_provider(
            jacobian_provider, allow_undeclared=allow_undeclared_jacobian
        )
        self.jacobian_is_contact_free = (
            self.jacobian_provenance.get("contact_used_in_jacobian") is False
        )
        self.dt = float(sample_period_s)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("sample_period_s must be finite and positive.")
        self.velocity_limit = np.asarray(velocity_limit, dtype=float).reshape(7)
        self.acceleration_limit = np.asarray(acceleration_limit, dtype=float).reshape(7)
        self.state_min = np.asarray(state_min, dtype=float).reshape(7)
        self.state_max = np.asarray(state_max, dtype=float).reshape(7)
        for name, value in (
            ("velocity_limit", self.velocity_limit),
            ("acceleration_limit", self.acceleration_limit),
        ):
            if np.any(value <= 0.0) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite and positive.")
        self.position_gain = float(position_gain)
        self.damping = float(damping)
        self.nullspace_gain = float(nullspace_gain)
        self.feedforward = bool(feedforward)
        if self.damping <= 0.0:
            raise ValueError("damping must be positive; it is a Levenberg parameter.")
        self.selective_damping_gain = float(selective_damping_gain)
        self.selective_damping_floor = float(selective_damping_floor)
        if self.selective_damping_gain < 0.0 or not np.isfinite(self.selective_damping_gain):
            raise ValueError("selective_damping_gain must be finite and >= 0.")
        if self.selective_damping_floor <= 0.0 or not np.isfinite(self.selective_damping_floor):
            raise ValueError("selective_damping_floor must be finite and > 0.")

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "description": self.description,
            "jacobian": provenance_line(self.jacobian_provenance),
            "jacobian_provenance": dict(self.jacobian_provenance),
            "position_gain": self.position_gain,
            "damping": self.damping,
            "nullspace_gain": self.nullspace_gain,
            "feedforward": self.feedforward,
            "selective_damping_gain": self.selective_damping_gain,
            "selective_damping_floor": self.selective_damping_floor,
        }

    def reset(self) -> None:
        return None

    def _reference_index(self, control_index: int) -> int:
        return int(
            np.clip(int(control_index) + 1, 0, self.reference.sample_count - 1)
        )

    def solve(
        self,
        *,
        measured_state: Any,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> InverseJacobianStep:
        started = time.perf_counter()
        state = np.asarray(measured_state, dtype=float).reshape(7)
        measured = np.asarray(measured_beam_position, dtype=float).reshape(3)
        previous = np.asarray(previous_input, dtype=float).reshape(7)
        index = self._reference_index(control_index)

        desired = np.asarray(self.reference.desired_position_m, dtype=float)[index]
        reference_state = np.asarray(self.reference.state, dtype=float)[index]
        reference_input = np.asarray(self.reference.input, dtype=float)[index]

        jacobian = np.asarray(
            self.jacobian_provider(state), dtype=float
        ).reshape(3, 7)
        if not np.all(np.isfinite(jacobian)):
            raise FloatingPointError("The beam Jacobian is not finite.")

        if self.selective_damping_gain > 0.0:
            pseudo = self._selective_damped_pseudo_inverse(jacobian)
        else:
            # Damped least squares: J^T (J J^T + lambda^2 I)^-1.
            gram = jacobian @ jacobian.T + (self.damping**2) * np.eye(3)
            pseudo = jacobian.T @ np.linalg.solve(gram, np.eye(3))

        task_velocity = pseudo @ (self.position_gain * (desired - measured) / self.dt)
        projector = np.eye(7) - pseudo @ jacobian
        nullspace_velocity = projector @ (
            self.nullspace_gain * (reference_state - state) / self.dt
        )
        command = task_velocity + nullspace_velocity
        if self.feedforward:
            # The feedforward is a nullspace-consistent addition: the task term
            # already contains the tip motion the reference asks for, so adding
            # the reference velocity outright would double-count it.
            command = command + projector @ reference_input

        command = self._clip(command, state, previous)
        elapsed = time.perf_counter() - started

        predicted_state = state + self.dt * command
        predicted_tip = measured + jacobian @ (self.dt * command)
        return InverseJacobianStep(
            command=command,
            planned_input=reference_input.copy(),
            predicted_states=predicted_state.reshape(1, 7),
            predicted_inputs=command.reshape(1, 7),
            objective=float(np.linalg.norm(desired - measured)),
            status="resolved_rate",
            success=True,
            iterations=1,
            solve_time_s=float(elapsed),
            primal_residual=float("nan"),
            dual_residual=float("nan"),
            measured_beam_position=measured.copy(),
            instantaneous_output_residual=np.full(3, np.nan),
            estimated_output_residual=np.full(3, np.nan),
            predicted_beam_positions=predicted_tip.reshape(1, 3),
            predicted_beam_errors=(predicted_tip - desired).reshape(1, 3),
            first_predicted_beam_error_m=float(
                np.linalg.norm(predicted_tip - desired)
            ),
        )

    def _selective_damped_pseudo_inverse(self, jacobian: Array) -> Array:
        """Per-singular-value damped pseudo-inverse (SDLS, 2026-09-12).

        ``J = U diag(sigma) V^T`` (economy SVD, 3x3/3/3x7).  Each direction i
        gets its own Levenberg term ``lambda_i^2 = damping^2 + gain *
        weight_i``, ``weight_i = 1/(sigma_i^2+floor^2) - min_j(...)`` -- the
        same bounded-weight construction validated for
        ``BeamOutputMPCConfig.directional_damping`` (floor caps the near-null
        direction's extra penalty instead of letting it blow up as
        sigma_i -> 0). ``weight_i`` is ~0 for the well-conditioned directions
        by construction (it is shifted by its own minimum), so they keep
        close to the plain isotropic-damping behaviour; only the near-null
        direction gets materially more suppression than ``damping`` alone
        would give it.

        Caveat (checked numerically against the real beam's singular values,
        ~0.086 / 0.037 / 5.9e-7): the continuous ``1/sigma^2`` weighting means
        the SECOND-best direction (0.037) also picks up a little extra
        damping (~5.5% of what the near-null direction gets) -- it is not a
        clean "leave the top-2 untouched" scheme. Small at this system's
        actual singular-value spread, but don't assume ``selective_damping_gain``
        is perfectly surgical; check the weak-direction fraction AND the
        overall command norm, not just one of them, when tuning it.
        """
        u, s, vt = np.linalg.svd(jacobian, full_matrices=False)
        floor2 = self.selective_damping_floor**2
        weight = 1.0 / (s**2 + floor2)
        weight = weight - weight.min()
        lam2 = self.damping**2 + self.selective_damping_gain * weight
        gains = s / (s**2 + lam2)
        return (vt.T * gains) @ u.T

    def _clip(self, command: Array, state: Array, previous: Array) -> Array:
        command = np.clip(command, -self.velocity_limit, self.velocity_limit)
        command = np.clip(
            command,
            previous - self.acceleration_limit * self.dt,
            previous + self.acceleration_limit * self.dt,
        )
        command = np.clip(
            command,
            (self.state_min - state) / self.dt,
            (self.state_max - state) / self.dt,
        )
        return np.clip(command, -self.velocity_limit, self.velocity_limit)


def build_inverse_jacobian_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    mpc_config: Any,
    position_gain: float = 1.0,
    damping: float = 1.0e-3,
    nullspace_gain: float = 1.0,
    feedforward: bool = True,
    allow_undeclared_jacobian: bool = True,
    selective_damping_gain: float = 0.0,
    selective_damping_floor: float = 0.01,
) -> InverseJacobianBeamController:
    """Build it from the same ``ConfigurationMPCConfig`` the MPCs use.

    Sharing the config is what guarantees the baseline gets the same limits,
    the same sample period and the same state box as the controllers it is
    being compared against.
    """
    return InverseJacobianBeamController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        sample_period_s=float(mpc_config.sample_period_s),
        velocity_limit=mpc_config.velocity_limit,
        acceleration_limit=mpc_config.acceleration_limit,
        state_min=mpc_config.state_min,
        state_max=mpc_config.state_max,
        position_gain=position_gain,
        damping=damping,
        nullspace_gain=nullspace_gain,
        feedforward=feedforward,
        allow_undeclared_jacobian=allow_undeclared_jacobian,
        selective_damping_gain=selective_damping_gain,
        selective_damping_floor=selective_damping_floor,
    )


__all__ = [
    "InverseJacobianBeamController",
    "InverseJacobianStep",
    "build_inverse_jacobian_controller",
]
