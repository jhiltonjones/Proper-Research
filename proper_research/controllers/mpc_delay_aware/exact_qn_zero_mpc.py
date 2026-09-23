"""Exact Q_N=0 task-nullspace-free delay-aware MPC (2026-09-23).

Identical to `StagewiseTaskNullspaceDelayAwareMPC` except the stagewise
nullspace/range-space state-tracking term is forced to exactly zero at
every horizon stage (P_N=P_R=0), rather than approximated via gamma=0
(which leaves P_N fully active -- gamma only weights P_R, see that
class's own docstring) or a state_error_scale magnitude hack.

Built for the live vessel-navigation model-necessity study's frozen
Q_N=0 configuration (see `paper/model_necessity_study.tex`,
Sec. mn-exp4): the offline mechanism study found this secondary posture
term confounds the contact-vs-no-contact comparison -- it interacts with
the primary tracking term's effective gain (the more accurate contact
Jacobian produces a more aggressive correction that the fixed-weight
posture term fights harder), not with which Jacobian sources the
projector -- and resolved it by disabling Q_N exactly, not
approximately, before running the paired closed-loop comparison. This
class is that exact disable, made reusable for the live process-isolated
worker rather than only the offline `DecoupledProjectorMPC` used during
the study itself.

`gamma` is accepted (constructor-signature compatibility with the class
this subclasses) but ignored -- P_N and P_R are unconditionally zero
regardless of its value.
"""
from __future__ import annotations

import numpy as np

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC

Array = np.ndarray

__all__ = ["ExactQNZeroTaskNullspaceDelayAwareMPC"]


class ExactQNZeroTaskNullspaceDelayAwareMPC(StagewiseTaskNullspaceDelayAwareMPC):
    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        input_reference = self.reference.input_window(control_index, self.N)
        n = self.n

        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input
        input_reference_vector = input_reference.reshape(self.nu)

        # Q_N/Q_R term exactly zero (H_Q=f_Q=0), not merely gamma-weighted --
        # see module docstring.
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
        zeros_proj = np.zeros((self.N, n, n), dtype=float)
        return hessian, linear, {
            "G": G, "constant_error": constant_error,
            "desired_position": desired_position, "nominal_position": nominal_position,
            "Jbar": Jbar, "input_reference": input_reference,
            "P_R": zeros_proj, "P_N": zeros_proj, "ranks": np.zeros(self.N, dtype=int),
        }
