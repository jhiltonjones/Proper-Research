"""Insertion-position anchor on top of the exact task-nullspace state cost
(2026-09-21).

Live validation of the frozen gamma=0/R700 nullspace controller
(`run_mpc_delay_aware_nullq_r700.py`) exposed a repeatable, physically-
diagnosed failure: because insertion is strongly task-producing
(J_x,L~=+1 in this project's beam-plane convention), an insertion offset
L-L_ref lies mostly in the task row-space, so P_N barely penalizes it, and
R(u-u_ref) only penalizes velocity, not accumulated position offset -- once
u_L returns near u_ref,L its ongoing R-cost is small even with L-L_ref far
from zero. The corrected E_chosen/E_model decomposition (identity-verified,
see CONTROLLER_JUSTIFICATION.md / SUPPLEMENTARY_EVIDENCE.md addenda) showed
E_model,x stays small (~0.2-0.25mm) while E_chosen,x is large (~2.3mm valid
rep, ~10mm aborted rep) at every horizon stage -- the model correctly
predicts the bad outcome; the objective simply doesn't penalize choosing it.

This module adds EXACTLY ONE new scalar term to the existing exact
task-nullspace state cost, nothing else:

    l_Q,j = ||P_N,j @ e~_z,j||^2 + w_L * ((L_j|k - L_ref,k+j) / s_L)^2

using the PREDICTED PHYSICAL insertion state L_j|k (the L-component of the
physical-state stack z_phys_stack = Ep@x_exec + Sp@v), not u_L, not a
command increment, and not a shifted target. gamma stays fixed at 0 (the
frozen joint-side formulation is untouched); Qp, Rd, R=700*R0, u_ref, d=2,
beta_d=1, N=15, V_f=0 are all unchanged.

`insertion_state_anchor_weight=0.0` (the default) must reproduce the exact
gamma=0/R700 StagewiseTaskNullspaceDelayAwareMPC solution bit-for-bit -- see
`insertion_anchor_self_test.py`.
"""
from __future__ import annotations

import numpy as np

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC

Array = np.ndarray


class InsertionAnchoredTaskNullspaceMPC(StagewiseTaskNullspaceDelayAwareMPC):
    def __init__(
        self,
        *,
        insertion_state_anchor_weight: float = 0.0,
        insertion_state_anchor_scale: float = 0.25e-3,
        gamma: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(gamma=gamma, **kwargs)
        self.insertion_state_anchor_weight = float(insertion_state_anchor_weight)
        self.insertion_state_anchor_scale = float(insertion_state_anchor_scale)

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        hessian, linear, terms = super()._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=previous_input, control_index=control_index,
            estimated_residual=estimated_residual,
        )
        w_L = self.insertion_state_anchor_weight
        if w_L == 0.0:
            return hessian, linear, terms

        n = self.n  # 7 (6 joints + insertion) per stage
        N = self.N
        # Selector picking the insertion component (index 6) of every
        # stage's 7-dim physical-state block out of the (N*7,) stack
        # z_phys_stack = Ep@x_exec + Sp@v.
        Sel_L = np.zeros((N, N * n), dtype=float)
        for j in range(N):
            Sel_L[j, j * n + 6] = 1.0

        state_reference = self.reference.state_window(control_index, N)  # (N,7)
        L_ref_stack = state_reference[:, 6]  # (N,)

        free_state = self.Ep @ x_exec  # (N*7,)
        G_L = Sel_L @ self.Sp  # (N, nu) -- dL_j|k/dv
        c_L = Sel_L @ free_state - L_ref_stack  # (N,) -- L_j|k(v=0) - L_ref,k+j

        s_L2 = self.insertion_state_anchor_scale ** 2
        weight = w_L / s_L2

        # SAME terminal-block convention as the base class's own Qbar_gamma
        # (see stagewise_task_nullspace.py: Qbar_gamma[-n:,-n:] *=
        # terminal_weight_multiplier) -- the last stage's contribution to
        # this term is scaled the same way, for consistency with how Q_N's
        # own terminal block is already treated. Deliberate, documented
        # choice, not left implicit.
        tw = float(self.config.terminal_weight_multiplier)
        stage_scale = np.ones(N, dtype=float)
        stage_scale[-1] = tw

        H_L = 2.0 * weight * (G_L.T @ (stage_scale[:, None] * G_L))
        f_L = 2.0 * weight * (G_L.T @ (stage_scale * c_L))

        hessian = hessian + H_L
        linear = linear + f_L
        hessian = 0.5 * (hessian + hessian.T)
        terms = dict(terms)
        terms["G_L"] = G_L
        terms["c_L"] = c_L
        return hessian, linear, terms
