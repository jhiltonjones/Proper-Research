"""Task/nullspace-Q delay-aware MPC (Q_gamma), 2026-09-19.

Separates the state-tracking cost into a task-producing (row-space of the
scaled Jacobian) component and a task-null (everything else) component,
weighting them differently:

    z~ = S_z^{-1}(z - z_ref)          (normalized configuration error)
    J_s = J @ S_z                     (scaled Jacobian at a representative index)
    P_R = J_s^+ J_s                   (exact row-space projector, via SVD)
    P_N = I - P_R

    l_Q = ||P_N z~||^2 + gamma * ||P_R z~||^2

gamma=1 recovers the current (M0/MT-stagewise's own) isotropic Q exactly.
gamma=0 is pure task-nullspace regularization -- posture-only, no penalty
at all on task-producing configuration offsets. This project's EARLIER
Q=0-in-the-task-direction ablation (q_ablation.py's Q_no_null, done before
this delay-aware/target-consistent investigation existed) went rough/
saturated -- but that ablation paired null-projected Q with the ORIGINAL,
near-negligible R/Rd (the cost-gradient decomposition done since then
showed R/Rd sit ~3 orders of magnitude below Q/Qp's gradients). Once the
row-space Q stops regulating task-producing aggressiveness, R/Rd become
the ONLY thing left to do it, and at their original magnitude they can't.
This class deliberately does NOT touch R/Rd -- that ablation (R->alpha*R)
is a separate, later step once a gamma is chosen, not bundled in here.

Representative-Jacobian convention matches this project's established
precedent (q_ablation.py's NullAwareMPC): ONE Q_gamma per solve, built
from the Jacobian at the CURRENT tick's control_index (not stage-varying
across the horizon -- a fidelity simplification this project already made
once, not a new one).

H changes per solve here (Q_gamma varies with the schedule), unlike
MT-stagewise/MT-constant where only the linear term moved -- this
subclass inlines the state-cost linear/quadratic construction directly
rather than reusing `_linear_cost_exec`, since that method reads
`self.Qbar` (the base class's fixed, isotropic Q) directly.
"""
from __future__ import annotations

import numpy as np

from . import delay_aware_mpc as dam

Array = np.ndarray


def build_task_nullspace_Q(
    *, J_repr: Array, state_error_scale: Array, gamma: float,
) -> Array:
    """(7,7) Q_gamma = M (P_N + gamma*P_R) M, M = diag(1/state_error_scale)."""
    m_diag = 1.0 / np.asarray(state_error_scale, dtype=float).reshape(7)
    J_tilde = np.asarray(J_repr, dtype=float).reshape(3, 7) * (state_error_scale[None, :])
    # J_s = J @ S_z = J @ diag(state_error_scale) -- scaled Jacobian, matches
    # q_ablation.py's own m_inv=1/m_diag=state_error_scale convention exactly.
    _, S, Vt = np.linalg.svd(J_tilde, full_matrices=True)
    rank = int(np.sum(S > 1.0e-9 * max(S[0], 1.0)))
    V_row = Vt[:rank]
    P_R = V_row.T @ V_row
    P_N = np.eye(7) - P_R
    combo = P_N + float(gamma) * P_R
    return (m_diag[:, None] * combo) * m_diag[None, :]


class TaskNullspaceDelayAwareMPC(dam.DelayAwareBeamOutputTrackingMPC):
    def __init__(self, *, gamma: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gamma = float(gamma)
        self._state_error_scale = np.asarray(self.config.state_error_scale, dtype=float).reshape(7)

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)

        J_repr = np.asarray(self.reference_position_jacobians[control_index], dtype=float)
        Q_gamma = build_task_nullspace_Q(
            J_repr=J_repr, state_error_scale=self._state_error_scale, gamma=self.gamma,
        )
        Qbar_gamma = np.kron(np.eye(self.N), Q_gamma)
        # SAME terminal-block multiplier convention the base class's own
        # (isotropic) Qbar uses, applied here too for consistency (V_f=0,
        # so this is just a stage weight, not a Riccati terminal cost).
        n = self.n
        Qbar_gamma[-n:, -n:] *= float(self.config.terminal_weight_multiplier)

        free_state = self.Ep @ x_exec
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input

        H_Q = 2.0 * self.Sp.T @ Qbar_gamma @ self.Sp
        f_Q = 2.0 * self.Sp.T @ Qbar_gamma @ (free_state - state_reference_vector)
        H_R = 2.0 * self.Rbar
        f_R = -2.0 * self.Rbar @ input_reference_vector
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
            "Jbar": Jbar, "input_reference": input_reference, "Q_gamma": Q_gamma,
        }
