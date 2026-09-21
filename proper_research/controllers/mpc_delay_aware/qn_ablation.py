"""Q_N on/off ablation (2026-09-20).

Tests whether the nullspace state-tracking cost is doing anything beyond
what the full velocity reference (u_ref, including its redundant/nullspace
component) already provides, on top of the R700/gamma=0 formulation
(`stagewise_task_nullspace.py`, gamma=0 already means "no row-space weight
-- Q is exactly P_N", i.e. what this ablation calls "Q_N").

A: R700Gamma0MPC (StagewiseTaskNullspaceDelayAwareMPC, gamma=0.0) -- Q_N on,
   full u_ref, R700, Qp, Rd, constraints all unchanged.
B: QZeroMPC -- identical in every other respect, but the ENTIRE state cost
   (both row-space and null-space) is zeroed: Q=0 exactly. u_ref stays
   full (this ablation does NOT touch the input reference), so the
   hypothesis under test is specifically "does full u_ref already keep the
   controller near the planner's nullspace trajectory, making the Q_N
   anchor redundant."

Both classes still compute and return P_R/P_N/ranks in the terms dict
(via `build_stagewise_projectors`) so the same e_N(k) nullspace-deviation
diagnostic can be applied to both conditions' predicted trajectories,
even though B's optimizer never sees P_N in its cost.
"""
from __future__ import annotations

import numpy as np

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC, build_stagewise_projectors

Array = np.ndarray


class QZeroMPC(StagewiseTaskNullspaceDelayAwareMPC):
    """Same as StagewiseTaskNullspaceDelayAwareMPC but the state cost Q is
    identically zero at every stage (both row- and null-space components).
    Qp/R/Rd/constraints/u_ref are untouched."""

    def __init__(self, **kwargs) -> None:
        # gamma is irrelevant once Qbar_gamma is zeroed, but keep it at 0.0
        # so any accidental fallthrough matches condition A's row-space
        # weight rather than silently reintroducing gamma=1 behaviour.
        kwargs.setdefault("gamma", 0.0)
        super().__init__(**kwargs)

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

        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input

        # Q entirely zeroed: no H_Q, no f_Q contribution at all.
        H_R = 2.0 * self.Rbar
        f_R = -2.0 * self.Rbar @ input_reference_vector
        H_Rd = 2.0 * self.D.T @ self.Rdbar @ self.D
        f_Rd = -2.0 * self.D.T @ self.Rdbar @ previous_vector
        base_linear = f_R + f_Rd
        base_hessian = H_R + H_Rd

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
            "P_R": P_R, "P_N": P_N, "ranks": ranks,
        }
