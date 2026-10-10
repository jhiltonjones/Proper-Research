"""Offline exploratory MPC secondary-objective variant (2026-09-21) --
NOT the validated live controller. See
`close_loop_logs/authority_experiment/SPEC.md` (mirrors the research spec
this was built from verbatim) for the full derivation and the gating
sequence (Phase 0 diagnostic -> this integration -> strict A/B/C replay ->
only then consider anything beyond offline).

Three `secondary_objective` modes:

    "reference_nullspace"  -- BIT-IDENTICAL to
                              `stagewise_task_nullspace.StagewiseTaskNullspaceDelayAwareMPC`
                              (literally reuses `build_stagewise_projectors`,
                              not a reimplementation). This is condition A.
    "none"                 -- no secondary state cost at all (H_Q=f_Q=0).
                              This is condition B.
    "manipulability"       -- REPLACES (does not add to) the reference
                              anchor with a first-order authority-ascent
                              linear term in f only, H unchanged from "none".
                              This is condition C.

Manipulability term (linear-in-decision-variable, added to f, NEVER to H):

    l_manip,j = -lambda_manip * g_N,j^T @ S_z^{-1} @ (z_j - z_lin,j)

    z_lin,j = z_ref,k+j  (REFERENCE-LINEARIZED, not adaptive -- g_N,j is
    read from a schedule precomputed ONCE offline against the reference
    trajectory, indexed the same way `reference_position_jacobians` is;
    the expensive state-dependent beam-model Jacobian is NEVER evaluated
    inside a replay solve)

Condensing (z_stack = Ep@x_exec + Sp@u, same convention as every other
piece of this codebase's condensed QP):

    l_manip = sum_j -lambda_manip * (g_N,j/S_z)^T @ z_j + const(u-independent)
            = -lambda_manip * c_stack^T @ (Sp@u + free_state) + const
    f_manip = -lambda_manip * Sp^T @ c_stack,   c_stack[j] = g_N,j / S_z

This is exactly analogous to how f_Q is built (same Sp, same condensing),
except linear rather than quadratic, so it contributes to f only.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from . import delay_aware_mpc as dam
from .stagewise_task_nullspace import build_stagewise_projectors
from .target_consistent import build_beam_plane_projection

Array = np.ndarray

SecondaryObjective = Literal["reference_nullspace", "none", "manipulability"]
_VALID_SECONDARY = ("reference_nullspace", "none", "manipulability")


class AuthorityShapedDelayAwareMPC(dam.DelayAwareBeamOutputTrackingMPC):
    """See module docstring. Everything else (Qp, R, Rd, delay-aware
    physical prediction, constraints, genuine LTV schedule) is completely
    untouched -- identical to `StagewiseTaskNullspaceDelayAwareMPC`."""

    def __init__(
        self,
        *,
        secondary_objective: SecondaryObjective,
        gamma: float = 0.0,
        manipulability_weight: float = 0.0,
        authority_schedule: dict[str, Array] | None = None,
        axial_axis_R=(-1.0, 0.0, 0.0),
        normal_axis_R=(0.0, 0.0, -1.0),
        **kwargs,
    ) -> None:
        if secondary_objective not in _VALID_SECONDARY:
            raise ValueError(f"secondary_objective must be one of {_VALID_SECONDARY}; got {secondary_objective!r}")
        super().__init__(**kwargs)
        self.secondary_objective = secondary_objective
        self.gamma = float(gamma)
        self.manipulability_weight = float(manipulability_weight)
        self._C = build_beam_plane_projection(axial_axis_R=axial_axis_R, normal_axis_R=normal_axis_R)
        self._state_error_scale = np.asarray(self.config.state_error_scale, dtype=float).reshape(7)

        if secondary_objective == "manipulability":
            if authority_schedule is None or "g_N" not in authority_schedule:
                raise ValueError(
                    "secondary_objective='manipulability' requires authority_schedule={'g_N': (N_ref,7) array, ...} "
                    "precomputed offline (see authority_phase0.py) -- this class never evaluates the expensive "
                    "state-dependent beam-model Jacobian itself (reference-linearized, reference-scheduled, per spec)."
                )
            g_N = np.asarray(authority_schedule["g_N"], dtype=float)
            if g_N.shape != (self.reference.sample_count, 7) or not np.all(np.isfinite(g_N)):
                raise ValueError(
                    f"authority_schedule['g_N'] must have shape ({self.reference.sample_count}, 7) and be finite; "
                    f"got shape {g_N.shape}."
                )
            self._g_N_schedule = g_N
        else:
            self._g_N_schedule = None

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)
        indices = self._reference_indices(control_index, future=True)
        n = self.n
        S_z = self._state_error_scale

        free_state = self.Ep @ x_exec
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input

        # ---- secondary state cost (Q_N / gamma*Q_R, or nothing) ----
        ranks = None
        if self.secondary_objective == "reference_nullspace":
            J_schedule = np.asarray(self.reference_position_jacobians, dtype=float)[indices]
            P_R, P_N, ranks = build_stagewise_projectors(
                J_schedule=J_schedule, C=self._C, state_error_scale=S_z,
            )
            M = np.diag(1.0 / S_z)
            Qbar_gamma = np.zeros((self.N * n, self.N * n), dtype=float)
            for j in range(self.N):
                combo = P_N[j] + self.gamma * P_R[j]
                Qbar_gamma[j * n:(j + 1) * n, j * n:(j + 1) * n] = M.T @ combo @ M
            Qbar_gamma[-n:, -n:] *= float(self.config.terminal_weight_multiplier)
            H_Q = 2.0 * self.Sp.T @ Qbar_gamma @ self.Sp
            f_Q = 2.0 * self.Sp.T @ Qbar_gamma @ (free_state - state_reference_vector)
        else:  # "none" and "manipulability" both start from zero secondary state cost
            H_Q = np.zeros((self.nu, self.nu), dtype=float)
            f_Q = np.zeros(self.nu, dtype=float)

        H_R = 2.0 * self.Rbar
        f_R = -2.0 * self.Rbar @ input_reference_vector
        H_Rd = 2.0 * self.D.T @ self.Rdbar @ self.D
        f_Rd = -2.0 * self.D.T @ self.Rdbar @ previous_vector
        base_linear = f_Q + f_R + f_Rd
        base_hessian = H_Q + H_R + H_Rd

        # ---- manipulability linear term: f ONLY, never H ----
        f_manip = np.zeros(self.nu, dtype=float)
        g_N_stack = None
        if self.secondary_objective == "manipulability" and self.manipulability_weight != 0.0:
            g_N_stack = self._g_N_schedule[indices]  # (N,7), z_lin,j = z_ref,k+j by construction (same indices)
            c_stack = (g_N_stack / S_z[None, :]).reshape(self.nu)
            f_manip = -self.manipulability_weight * (self.Sp.T @ c_stack)
            base_linear = base_linear + f_manip

        G, constant_error, _, desired_position, nominal_position, Jbar = (
            self._beam_prediction_terms_exec(
                x_exec=x_exec, control_index=control_index,
                estimated_residual=estimated_residual,
            )
        )
        hessian = base_hessian + 2.0 * (G.T @ self.Qpbar @ G)
        linear = base_linear + 2.0 * (G.T @ self.Qpbar @ constant_error)
        hessian = 0.5 * (hessian + hessian.T)
        hessian = hessian + float(self.config.hessian_regularization) * np.eye(self.nu)
        return hessian, linear, {
            "G": G, "constant_error": constant_error,
            "desired_position": desired_position, "nominal_position": nominal_position,
            "Jbar": Jbar, "input_reference": input_reference,
            "ranks": ranks, "f_manip": f_manip, "g_N_stack": g_N_stack,
            "reference_indices": indices,
        }
