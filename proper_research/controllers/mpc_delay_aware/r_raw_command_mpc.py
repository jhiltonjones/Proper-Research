"""Offline exploratory MPC variant (2026-09-22) -- NOT the validated live
controller. Compares two forms of the command-tracking term R:

    "feedforward" (validated, default): J_R = (u - u_ref)^T R (u - u_ref)
        -- penalizes DEVIATION from the feedforward reference; u=u_ref is
        free (zero cost).
    "raw":                              J_R = u^T R u
        -- penalizes the RAW command magnitude directly, regardless of
        u_ref; even u=u_ref costs ||u_ref||^2_R. Mechanically this is
        (u-u_ref)^T R (u-u_ref) with u_ref forced to 0 in the R term only
        -- H_R is unchanged (2R in both cases), only f_R changes
        (f_R=-2R@u_ref -> f_R=0).

Everything else (Q_N nullspace regularizer, Rd, beam prediction/disturbance
augmentation, constraints, d=2/beta_d=1/N=15/V_f=0) is byte-for-byte
identical to `StagewiseTaskNullspaceDelayAwareMPC` -- this file literally
reuses `build_stagewise_projectors`, not a reimplementation.

Per this project's established gating convention: this is an offline/replay
experimental mode only. Do not wire this into any live run script until an
offline counterfactual comparison against real recorded states shows it is
worth the live-testing risk -- "raw" removes the free-ride the feedforward
trajectory currently gets from R, so at any point along a fast-moving
reference (large ||u_ref||) it will actively fight the plan's own
feedforward motion, which the "feedforward" mode never does. This is a
structurally different, not obviously safe, change to try live untested.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from . import delay_aware_mpc as dam
from .stagewise_task_nullspace import build_stagewise_projectors
from .target_consistent import build_beam_plane_projection

Array = np.ndarray
RMode = Literal["feedforward", "raw"]


class StagewiseTaskNullspaceRRawMPC(dam.DelayAwareBeamOutputTrackingMPC):
    """Identical to StagewiseTaskNullspaceDelayAwareMPC except for the R
    term's reference point, controlled by `r_mode`. `r_mode="feedforward"`
    must reproduce the validated controller bit-for-bit (see regression
    test)."""

    def __init__(
        self, *, gamma: float, r_mode: RMode,
        axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0),
        **kwargs,
    ) -> None:
        if r_mode not in ("feedforward", "raw"):
            raise ValueError(f"r_mode must be 'feedforward' or 'raw'; got {r_mode!r}")
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self.r_mode = r_mode
        self._C = build_beam_plane_projection(axial_axis_R=axial_axis_R, normal_axis_R=normal_axis_R)
        self._state_error_scale = np.asarray(self.config.state_error_scale, dtype=float).reshape(7)

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)

        indices = self._reference_indices(control_index, future=True)
        J_schedule = np.asarray(self.reference_position_jacobians, dtype=float)[indices]
        P_R, P_N, ranks = build_stagewise_projectors(
            J_schedule=J_schedule, C=self._C, state_error_scale=self._state_error_scale,
        )
        S_z = self._state_error_scale
        M = np.diag(1.0 / S_z)
        n = self.n
        Qbar_gamma = np.zeros((self.N * n, self.N * n), dtype=float)
        for j in range(self.N):
            combo = P_N[j] + self.gamma * P_R[j]
            Q_j = M.T @ combo @ M
            Qbar_gamma[j * n:(j + 1) * n, j * n:(j + 1) * n] = Q_j
        Qbar_gamma[-n:, -n:] *= float(self.config.terminal_weight_multiplier)

        free_state = self.Ep @ x_exec
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input

        H_Q = 2.0 * self.Sp.T @ Qbar_gamma @ self.Sp
        f_Q = 2.0 * self.Sp.T @ Qbar_gamma @ (free_state - state_reference_vector)
        H_R = 2.0 * self.Rbar
        if self.r_mode == "feedforward":
            f_R = -2.0 * self.Rbar @ input_reference_vector
        else:  # "raw": J_R = u^T R u, i.e. reference point forced to 0
            f_R = np.zeros(self.nu, dtype=float)
        H_Rd = 2.0 * self.D.T @ self.Rdbar @ self.D
        f_Rd = -2.0 * self.D.T @ self.Rdbar @ previous_vector
        base_linear = f_Q + f_R + f_Rd
        base_hessian = H_Q + H_R + H_Rd

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
            "P_R": P_R, "P_N": P_N, "ranks": ranks, "r_mode": self.r_mode,
        }
