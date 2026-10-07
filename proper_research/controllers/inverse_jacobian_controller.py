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
    # 2026-10-07: hard-constraint-clip diagnostics (magnet exclusion /
    # z-workspace), all default-safe (None/False) so every existing caller's
    # InverseJacobianStep construction is unaffected. magnet_abort_reason
    # mirrors ProcessIsolatedDelayAwareAdapter's info["abort_reason"] hook
    # (controller_adapters.py's OfflineJointControllerAdapter.__call__
    # already reads it generically via getattr(step, "magnet_abort_reason",
    # None) -- no-op for any step object that doesn't set it).
    magnet_clip_active: bool = False
    magnet_exclusion_margin_m: float = float("nan")
    magnet_z_margin_m: float = float("nan")
    magnet_abort_reason: str | None = None


def project_joint_velocity_for_halfspace(
    command_joint6: Array, g_row: Array, rhs: float, velocity_limit_joint6: Array,
) -> tuple[Array, bool, float, float]:
    """Minimum-Euclidean-norm projection of a 6-vector joint-velocity command
    onto the half-space ``{x : g_row @ x >= rhs}``, then re-clipped to
    ``velocity_limit_joint6`` (a bound that would require exceeding the
    velocity limit to respect cannot be honoured in one step -- same
    principle as ``InverseJacobianBeamController._clip``).

    This is the resolved-rate-controller analogue of the linearized
    inequality `delay_aware_mpc.py`'s `_configure_magnet_exclusion`/
    `_configure_magnet_workspace` add to the MPC's QP: same first-order
    distance/height model (``g_row`` = constraint-gradient row, ``rhs`` =
    required one-tick rate), but enforced by a closed-form half-space
    projection instead of a QP inequality -- consistent with this
    controller's whole "no preview, no constraints beyond clipping" design.

    Returns ``(projected, was_clipped, slack_before, slack_after)``, where
    slack is ``g_row @ x - rhs`` (negative = constraint violated at that
    point; still negative in slack_after means the velocity limit alone
    was not enough to satisfy it in one tick).
    """
    g_norm2 = float(g_row @ g_row)
    val_before = float(g_row @ command_joint6)
    slack_before = val_before - rhs
    if slack_before >= 0.0 or g_norm2 < 1.0e-12:
        return command_joint6, False, slack_before, slack_before
    lam = (rhs - val_before) / g_norm2
    projected = command_joint6 + lam * g_row
    projected = np.clip(projected, -velocity_limit_joint6, velocity_limit_joint6)
    slack_after = float(g_row @ projected) - rhs
    return projected, True, slack_before, slack_after


def make_magnet_exclusion_hold_gate(
    *,
    magnet_position_fn: Callable[[Array], Array],
    magnet_position_jacobian_fn: Callable[[Array], Array],
    dt: float,
    magnet_exclusion_lumen_C_m: Any | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
) -> Callable[[Array, Any, dict], tuple[Array, dict]]:
    """Build a close_loop_path_follow._COMMAND_SAFETY_GATE callable: HOLDS
    position (returns an all-zero 7-vector command, robot stays exactly
    where it is) on any tick where the controller's RAW command --
    unmodified, computed with no awareness of these constraints -- would,
    over one tick, push the magnet closer than magnet_exclusion_radius_m
    to the nearest point in magnet_exclusion_lumen_C_m, or outside
    magnet_z_bounds_m.

    This is the deliberately punitive alternative to
    InverseJacobianBeamController's own anticipatory clip (which instead
    projects the command to the constraint boundary and lets the robot
    keep moving -- "the controller handles it"). Use THIS gate, not that
    clip, when the point of the experiment is to show the controller does
    NOT understand the constraint at all (unlike MPC, whose in-QP
    formulation structurally cannot produce an infeasible solution): the
    controller's own math is left completely untouched here (call it with
    no magnet_* kwargs at all, i.e. the plain naive baseline), and this
    gate sits entirely OUTSIDE it, between "controller decides" and
    "robot executes" -- the naive controller keeps trying to command
    motion into the excluded region every tick it wants to, and keeps
    getting held in place, which is the visible demonstration.

    Same first-order distance/height linearization as delay_aware_mpc.py's
    _configure_magnet_exclusion/_configure_magnet_workspace and this
    module's own project_joint_velocity_for_halfspace -- a pure go/no-go
    check here (no projection, no partial correction).
    """
    lumen = (
        None if magnet_exclusion_lumen_C_m is None
        else np.asarray(magnet_exclusion_lumen_C_m, dtype=float).reshape(-1, 3)
    )

    def gate(command: Array, measured_state: Any, info: dict) -> tuple[Array, dict]:
        if measured_state is None:
            return command, {"magnet_gate_held": False, "magnet_gate_reason": None}
        state = np.asarray(measured_state, dtype=float).reshape(7)
        command = np.asarray(command, dtype=float).reshape(7)
        joint_cmd = command[:6]

        p_mag = np.asarray(magnet_position_fn(state), dtype=float).reshape(3)
        J_mag = np.asarray(magnet_position_jacobian_fn(state), dtype=float).reshape(3, 7)[:, :6]

        reasons: list[str] = []
        if magnet_exclusion_radius_m is not None and lumen is not None:
            diffs = p_mag[None, :] - lumen
            dists = np.linalg.norm(diffs, axis=1)
            k = int(np.argmin(dists))
            d_nom = float(dists[k])
            normal = diffs[k] / max(d_nom, 1.0e-9)
            g = normal @ J_mag
            predicted = d_nom + float(g @ joint_cmd) * dt
            if predicted < magnet_exclusion_radius_m:
                reasons.append(
                    f"exclusion(predicted={predicted*1e3:.2f}mm < "
                    f"{magnet_exclusion_radius_m*1e3:.2f}mm)"
                )

        if magnet_z_bounds_m is not None:
            z_min, z_max = magnet_z_bounds_m
            g_z = J_mag[2, :]
            z_nom = float(p_mag[2])
            predicted_z = z_nom + float(g_z @ joint_cmd) * dt
            if predicted_z < z_min or predicted_z > z_max:
                reasons.append(
                    f"zworkspace(predicted_z={predicted_z*1e3:.1f}mm not in "
                    f"[{z_min*1e3:.1f},{z_max*1e3:.1f}]mm)"
                )

        if reasons:
            return np.zeros(7, dtype=float), {
                "magnet_gate_held": True,
                "magnet_gate_reason": "; ".join(reasons),
                "magnet_gate_raw_command": command.tolist(),
            }
        return command, {"magnet_gate_held": False, "magnet_gate_reason": None}

    return gate


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
        feedforward_full: bool = False,
        allow_undeclared_jacobian: bool = False,
        selective_damping_gain: float = 0.0,
        selective_damping_floor: float = 0.01,
        reference_position_jacobians: Any = None,
        magnet_position_fn: Callable[[Array], Array] | None = None,
        magnet_position_jacobian_fn: Callable[[Array], Array] | None = None,
        magnet_exclusion_lumen_C_m: Any | None = None,
        magnet_exclusion_radius_m: float | None = None,
        magnet_z_bounds_m: tuple[float, float] | None = None,
        magnet_constraint_violation_abort_m: float = 2.0e-3,
    ) -> None:
        self.reference = reference
        # The controller records what it was handed. Ask any instance
        # `controller.jacobian_provenance` and it will tell you whether its
        # Jacobian is contact-free, rather than you having to remember.
        self.jacobian_provider, self.jacobian_provenance = declare_provider(
            jacobian_provider, allow_undeclared=allow_undeclared_jacobian
        )
        # 2026-09-16: "inverse-LTV" -- a matched baseline for MPC-LTV. When
        # given, `solve()` looks up this precomputed per-sample Jacobian
        # instead of calling `jacobian_provider(state)`, exactly mirroring how
        # `mpc_ltv_offline` uses `reference_position_jacobians[index]`. This
        # exists because the fairest test of "does the MPC horizon help" is
        # holding the Jacobian *source* fixed and varying only the control
        # law (one-step resolved-rate vs finite-horizon QP) -- not comparing
        # a live/frozen provider against a scheduled one. See
        # `close_loop_logs/nullspace_motion_investigation_2026-09-16.md` §5-6.
        self._schedule: Array | None = None
        if reference_position_jacobians is not None:
            schedule = np.asarray(reference_position_jacobians, dtype=float)
            expected = (int(reference.sample_count), 3, 7)
            if schedule.shape != expected or not np.all(np.isfinite(schedule)):
                raise ValueError(
                    "reference_position_jacobians must have shape "
                    f"{expected}; received {schedule.shape}."
                )
            self._schedule = schedule
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
        self.feedforward_full = bool(feedforward_full)
        if self.damping <= 0.0:
            raise ValueError("damping must be positive; it is a Levenberg parameter.")
        self.selective_damping_gain = float(selective_damping_gain)
        self.selective_damping_floor = float(selective_damping_floor)
        if self.selective_damping_gain < 0.0 or not np.isfinite(self.selective_damping_gain):
            raise ValueError("selective_damping_gain must be finite and >= 0.")
        if self.selective_damping_floor <= 0.0 or not np.isfinite(self.selective_damping_floor):
            raise ValueError("selective_damping_floor must be finite and > 0.")

        # 2026-10-07: optional anticipatory hard-constraint clip for magnet
        # exclusion / z-workspace -- the two constraints MPC enforces inside
        # its own QP (delay_aware_mpc.py's _configure_magnet_exclusion/
        # _configure_magnet_workspace) that this controller previously had
        # NO protection against at all (only a toothless +-2pi joint box and
        # the generic velocity/accel clip -- see this controller's own
        # module docstring). All default None/disabled: zero behaviour
        # change for every existing caller that doesn't pass these.
        # magnet_position_fn/magnet_position_jacobian_fn: callables taking
        # the measured 7-vector state and returning the magnet's xyz (3,)
        # and its (3,7) position Jacobian w.r.t. [q1..q6,L] (insertion
        # column is always zero -- the magnet is rigidly on the end
        # effector, same convention as magnet_position_jacobian_m in
        # delay_aware_mpc.py). Same linearization MPC uses (closest point
        # in magnet_exclusion_lumen_C_m, z-row of the same Jacobian), but
        # applied as a closed-form half-space projection each tick instead
        # of a QP inequality -- see project_joint_velocity_for_halfspace.
        self.magnet_position_fn = magnet_position_fn
        self.magnet_position_jacobian_fn = magnet_position_jacobian_fn
        self.magnet_exclusion_lumen_C_m = (
            None if magnet_exclusion_lumen_C_m is None
            else np.asarray(magnet_exclusion_lumen_C_m, dtype=float).reshape(-1, 3)
        )
        self.magnet_exclusion_radius_m = (
            None if magnet_exclusion_radius_m is None else float(magnet_exclusion_radius_m)
        )
        self.magnet_z_bounds_m = (
            None if magnet_z_bounds_m is None
            else (float(magnet_z_bounds_m[0]), float(magnet_z_bounds_m[1]))
        )
        self.magnet_constraint_violation_abort_m = float(magnet_constraint_violation_abort_m)
        self._magnet_clip_enabled = (
            self.magnet_position_fn is not None
            and self.magnet_position_jacobian_fn is not None
            and (self.magnet_exclusion_radius_m is not None or self.magnet_z_bounds_m is not None)
        )
        if self.magnet_exclusion_radius_m is not None and self.magnet_exclusion_lumen_C_m is None:
            raise ValueError("magnet_exclusion_radius_m requires magnet_exclusion_lumen_C_m.")
        if self.magnet_z_bounds_m is not None and self.magnet_z_bounds_m[0] >= self.magnet_z_bounds_m[1]:
            raise ValueError("magnet_z_bounds_m must be (z_min, z_max) with z_min < z_max.")

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "description": self.description,
            "jacobian": (
                "scheduled (offline, one Jacobian per reference sample -- "
                "matches mpc_ltv_offline's schedule)"
                if self._schedule is not None
                else provenance_line(self.jacobian_provenance)
            ),
            "jacobian_provenance": dict(self.jacobian_provenance),
            "jacobian_scheduled": self._schedule is not None,
            "position_gain": self.position_gain,
            "damping": self.damping,
            "nullspace_gain": self.nullspace_gain,
            "feedforward": self.feedforward,
            "feedforward_full": self.feedforward_full,
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

        if self._schedule is not None:
            jacobian = self._schedule[index]
        else:
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
        if self.feedforward_full:
            # Full 2DOF feedforward: add the reference input UNPROJECTED.
            # task_velocity already corrects the tip error relative to the
            # planner's own desired position, so this does not double-count
            # the planned tip motion -- it is what lets the controller
            # actually track the reference velocity instead of only ever
            # producing the (small) correction term. Distinct from the
            # nullspace-only feedforward below, which was found to starve
            # the commanded velocity of most of the planned motion.
            command = command + reference_input
        elif self.feedforward:
            # The feedforward is a nullspace-consistent addition: the task term
            # already contains the tip motion the reference asks for, so adding
            # the reference velocity outright would double-count it.
            command = command + projector @ reference_input

        command = self._clip(command, state, previous)

        magnet_clip_active = False
        magnet_exclusion_margin_m = float("nan")
        magnet_z_margin_m = float("nan")
        magnet_abort_reason = None
        if self._magnet_clip_enabled:
            (
                command, magnet_clip_active, magnet_exclusion_margin_m,
                magnet_z_margin_m, magnet_abort_reason,
            ) = self._clip_magnet_constraints(command, state)

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
            magnet_clip_active=magnet_clip_active,
            magnet_exclusion_margin_m=magnet_exclusion_margin_m,
            magnet_z_margin_m=magnet_z_margin_m,
            magnet_abort_reason=magnet_abort_reason,
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

    def _clip_magnet_constraints(
        self, command: Array, state: Array,
    ) -> tuple[Array, bool, float, float, str | None]:
        """Anticipatory half-space projection for magnet-exclusion-radius
        and magnet-z-workspace, applied AFTER the standard velocity/accel/
        state-box clip (so this never asks for more than the velocity limit
        already allows -- project_joint_velocity_for_halfspace's own
        final re-clip enforces that per-constraint too).

        Sequential, not a joint QP: exclusion first, then z_min, then
        z_max, each only correcting if its own linearized margin would
        otherwise go negative one tick from now. Constraints that conflict
        within one tick's velocity budget are NOT guaranteed to be jointly
        satisfied -- an intentional limitation matching this controller's
        whole "no preview, no constraints beyond clipping" design, not an
        oversight. If the clip still leaves a real (not just linearization-
        noise-level) violation after projection, magnet_abort_reason is
        set so the caller's generic info["abort_reason"] hook (see
        controller_adapters.OfflineJointControllerAdapter.__call__) can
        stop the run -- the same anticipatory-clip-plus-reactive-monitor
        layering every other hard constraint in this project uses.
        """
        joint_cmd = command[:6].copy()
        vlim6 = self.velocity_limit[:6]
        clipped_any = False
        abort_reason = None

        p_mag = np.asarray(self.magnet_position_fn(state), dtype=float).reshape(3)
        J_mag = np.asarray(self.magnet_position_jacobian_fn(state), dtype=float).reshape(3, 7)[:, :6]

        excl_margin = float("nan")
        if self.magnet_exclusion_radius_m is not None:
            lumen = self.magnet_exclusion_lumen_C_m
            diffs = p_mag[None, :] - lumen
            dists = np.linalg.norm(diffs, axis=1)
            k = int(np.argmin(dists))
            d_nom = float(dists[k])
            normal = diffs[k] / max(d_nom, 1.0e-9)
            g = normal @ J_mag
            rhs = (self.magnet_exclusion_radius_m - d_nom) / self.dt
            joint_cmd, was_clipped, slack_before, slack_after = project_joint_velocity_for_halfspace(
                joint_cmd, g, rhs, vlim6,
            )
            clipped_any = clipped_any or was_clipped
            excl_margin = d_nom - self.magnet_exclusion_radius_m
            if slack_after < -self.magnet_constraint_violation_abort_m / max(self.dt, 1.0e-9):
                abort_reason = (
                    f"inverse_jacobian_magnet_exclusion_unclippable("
                    f"d_nom={d_nom*1e3:.2f}mm radius={self.magnet_exclusion_radius_m*1e3:.2f}mm)"
                )

        z_margin = float("nan")
        if self.magnet_z_bounds_m is not None:
            z_min, z_max = self.magnet_z_bounds_m
            z_nom = float(p_mag[2])
            g_z = J_mag[2, :]
            # z >= z_min
            joint_cmd, c1, _, slack_after_min = project_joint_velocity_for_halfspace(
                joint_cmd, g_z, (z_min - z_nom) / self.dt, vlim6,
            )
            # z <= z_max  <=>  (-g_z) . x >= (z_nom - z_max)/dt
            joint_cmd, c2, _, slack_after_max = project_joint_velocity_for_halfspace(
                joint_cmd, -g_z, (z_nom - z_max) / self.dt, vlim6,
            )
            clipped_any = clipped_any or c1 or c2
            z_margin = min(z_nom - z_min, z_max - z_nom)
            tol = self.magnet_constraint_violation_abort_m / max(self.dt, 1.0e-9)
            if slack_after_min < -tol or slack_after_max < -tol:
                reason = (
                    f"inverse_jacobian_magnet_zworkspace_unclippable("
                    f"z={z_nom*1e3:.1f}mm bounds=[{z_min*1e3:.1f},{z_max*1e3:.1f}]mm)"
                )
                abort_reason = abort_reason or reason

        command = command.copy()
        command[:6] = joint_cmd
        return command, clipped_any, excl_margin, z_margin, abort_reason

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
    feedforward_full: bool = False,
    allow_undeclared_jacobian: bool = True,
    selective_damping_gain: float = 0.0,
    selective_damping_floor: float = 0.01,
    reference_position_jacobians: Any = None,
    magnet_position_fn: Callable[[Array], Array] | None = None,
    magnet_position_jacobian_fn: Callable[[Array], Array] | None = None,
    magnet_exclusion_lumen_C_m: Any | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
    magnet_constraint_violation_abort_m: float = 2.0e-3,
) -> InverseJacobianBeamController:
    """Build it from the same ``ConfigurationMPCConfig`` the MPCs use.

    Sharing the config is what guarantees the baseline gets the same limits,
    the same sample period and the same state box as the controllers it is
    being compared against. Pass ``reference_position_jacobians`` (the same
    schedule ``mpc_ltv_offline`` uses) to additionally match the Jacobian
    *source* -- the "inverse-LTV" baseline.

    The ``magnet_*`` arguments (all optional, default None/disabled) add the
    anticipatory magnet-exclusion-radius / magnet-z-workspace clip -- see
    ``InverseJacobianBeamController``'s own docstring on those parameters.
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
        feedforward_full=feedforward_full,
        allow_undeclared_jacobian=allow_undeclared_jacobian,
        selective_damping_gain=selective_damping_gain,
        selective_damping_floor=selective_damping_floor,
        reference_position_jacobians=reference_position_jacobians,
        magnet_position_fn=magnet_position_fn,
        magnet_position_jacobian_fn=magnet_position_jacobian_fn,
        magnet_exclusion_lumen_C_m=magnet_exclusion_lumen_C_m,
        magnet_exclusion_radius_m=magnet_exclusion_radius_m,
        magnet_z_bounds_m=magnet_z_bounds_m,
        magnet_constraint_violation_abort_m=magnet_constraint_violation_abort_m,
    )


__all__ = [
    "InverseJacobianBeamController",
    "InverseJacobianStep",
    "build_inverse_jacobian_controller",
    "project_joint_velocity_for_halfspace",
    "make_magnet_exclusion_hold_gate",
]
