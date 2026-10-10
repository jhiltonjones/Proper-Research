"""Stagewise task/nullspace-Q delay-aware MPC (2026-09-19).

Generalizes `task_nullspace_mpc.py` in two ways the design discussion
identified as necessary:

1. The row/null split is built from the CONTROLLED task Jacobian
   `J^t_j = C @ J_j` (C = the same beam-plane projection
   `target_consistent.py` uses), not the full 3-row beam Jacobian -- the
   controlled task here is 2D (in-plane), so the nullspace is 5D, not 4D.
   Rank is taken numerically from each stage's own scaled Jacobian, never
   hard-coded.
2. The projector is built PER HORIZON STAGE (P_R,j / P_N,j at stage j's
   own reference index), not from one "representative" index shared
   across the whole horizon (task_nullspace_mpc.py's simplification).

State cost, generalized:

    l_Q,j = z~_j^T (P_N,j + gamma * P_R,j) z~_j,   z~_j = S_z^{-1}(z_j - z_ref,j)

gamma=1 must reproduce the base class's isotropic Q EXACTLY (P_N+P_R=I),
gamma=0 is pure task-nullspace regularization -- the row-space state
penalty vanishes entirely, so a task-producing configuration offset that
Qp has already justified is no longer fought by Q once established (see
this module's docstring discussion for why that differs structurally from
MT: with gamma=0, P_N(z~ - delta_z~^t) = P_N z~ for ANY task-producing
delta_z^t, i.e. shifting the Q target in a task-producing direction is
provably a no-op once Q is nullspace-only -- MT's whole mechanism
collapses to identity here, which is why this is a different, more
fundamental structural change, not a refinement of MT).

Everything else -- Qp, R, Rd, the physical prediction (Ep/Sp, d_k's role),
constraints, d=2/beta_d=1/N=15/V_f=0, execution C, process isolation --
is untouched.
"""
from __future__ import annotations

import numpy as np

from . import delay_aware_mpc as dam
from .target_consistent import build_beam_plane_projection

Array = np.ndarray


def build_stagewise_projectors(
    *, J_schedule: Array, C: Array, state_error_scale: Array, rank_tol: float = 1.0e-9,
) -> tuple[Array, Array, Array]:
    """J_schedule: (N,3,7) beam Jacobian at each stage's reference index.
    Returns (P_R (N,7,7), P_N (N,7,7), ranks (N,) int) -- exact orthogonal
    projectors in NORMALIZED configuration coordinates (z~ = S_z^{-1}(z-z_ref)),
    built from the numerical rank of each stage's own scaled, task-projected
    Jacobian J_s,j = (C @ J_j) @ S_z. Never assumes a fixed rank."""
    J_schedule = np.asarray(J_schedule, dtype=float).reshape(-1, 3, 7)
    N = J_schedule.shape[0]
    S_z = np.asarray(state_error_scale, dtype=float).reshape(7)
    P_R = np.empty((N, 7, 7), dtype=float)
    P_N = np.empty((N, 7, 7), dtype=float)
    ranks = np.empty(N, dtype=int)
    for j in range(N):
        Jt = C @ J_schedule[j]  # (2,7) -- controlled-task Jacobian
        J_s = Jt * S_z[None, :]  # (2,7) scaled: J_s = J^t @ S_z
        _, S, Vt = np.linalg.svd(J_s, full_matrices=True)  # Vt is (7,7)
        rank = int(np.sum(S > rank_tol * max(float(S[0]), 1.0))) if S.size else 0
        ranks[j] = rank
        V_r = Vt[:rank]  # (rank,7)
        Pr = V_r.T @ V_r
        P_R[j] = Pr
        P_N[j] = np.eye(7) - Pr
    return P_R, P_N, ranks


class StagewiseTaskNullspaceDelayAwareMPC(dam.DelayAwareBeamOutputTrackingMPC):
    def __init__(
        self, *, gamma: float, axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma = float(gamma)
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
        M = np.diag(1.0 / S_z)  # S_z^{-1}
        n = self.n
        Qbar_gamma = np.zeros((self.N * n, self.N * n), dtype=float)
        for j in range(self.N):
            combo = P_N[j] + self.gamma * P_R[j]
            Q_j = M.T @ combo @ M  # z~^T combo z~ = (z-zref)^T M^T combo M (z-zref)
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
            "Jbar": Jbar, "input_reference": input_reference,
            "P_R": P_R, "P_N": P_N, "ranks": ranks,
        }
