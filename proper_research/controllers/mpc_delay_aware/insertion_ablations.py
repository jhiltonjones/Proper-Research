"""Two diagnostic ablations isolating why MPC leans on insertion feedback
while the successful kp=1 inverse uses essentially none (2026-09-19).
Both subclass `StagewiseTaskNullspaceDelayAwareMPC` (gamma configurable,
same R/Rd/Qp/constraints/prediction/process-isolation machinery) and
change exactly one thing each.

JointOnlyFeedbackMPC: hard-constrains u_L,j = u_ref,L,j for every horizon
stage (insertion follows the planner exactly, zero feedback authority),
reproducing the kp=1 inverse's own actuator-allocation structure as
closely as possible within this formulation. NOT the same as setting the
insertion velocity to zero -- the planned insertion motion continues.

AuthorityAwareTaskCostMPC: zeroes Qp at the horizon stages where a new
joint decision has no physical authority yet (stages 1..delay_samples
under this delay model), keeping Qp at its normal weight from
stage delay_samples+1 onward. Diagnostic, not a proposed final
controller -- tests whether insertion's early-stage-only authority (not
its later-stage efficiency advantage) is what's driving its use.
"""
from __future__ import annotations

import numpy as np

from .stagewise_task_nullspace import StagewiseTaskNullspaceDelayAwareMPC

Array = np.ndarray


class JointOnlyFeedbackMPC(StagewiseTaskNullspaceDelayAwareMPC):
    def _constraint_bounds_exec(self, *, x_exec: Array, previous_input: Array) -> tuple[Array, Array]:
        lower, upper = super()._constraint_bounds_exec(x_exec=x_exec, previous_input=previous_input)
        lower = lower.copy()
        upper = upper.copy()
        # input_lower/input_upper occupy the first `nu` entries of the
        # concatenated bounds (see the base class's _constraint_bounds_exec:
        # concatenate((input_lower, state_lower, increment_lower))) -- pin
        # the insertion (index 6, 13, 20, ... every 7th) velocity bound to
        # EXACTLY u_ref,L,j at each stage, collapsing that decision variable
        # to the planner's own value without touching state/rate bounds.
        u_ref_window = self._last_input_reference  # set in _dynamic_qp_terms_exec below
        for j in range(self.N):
            idx = j * self.m + 6
            lower[idx] = u_ref_window[j, 6]
            upper[idx] = u_ref_window[j, 6]
        return lower, upper

    def _dynamic_qp_terms_exec(self, *, x_exec, previous_input, control_index, estimated_residual):
        self._last_input_reference = self.reference.input_window(control_index, self.N)
        return super()._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=previous_input, control_index=control_index,
            estimated_residual=estimated_residual,
        )


class AuthorityAwareTaskCostMPC(StagewiseTaskNullspaceDelayAwareMPC):
    def _dynamic_qp_terms_exec(self, *, x_exec, previous_input, control_index, estimated_residual):
        hessian, linear, terms = super()._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=previous_input, control_index=control_index,
            estimated_residual=estimated_residual,
        )
        # Zero Qp's contribution at stages 1..delay_samples (no joint
        # authority yet) by rebuilding H/linear with Qpbar's early blocks
        # zeroed -- recompute the Qp term directly (G/constant_error are
        # already returned in `terms`) rather than re-deriving hessian from
        # scratch, so only Qp's block structure changes.
        G, constant_error = terms["G"], terms["constant_error"]
        Qpbar_masked = self.Qpbar.copy()
        n_out = 3
        for j in range(self.delay_samples):
            Qpbar_masked[j * n_out:(j + 1) * n_out, j * n_out:(j + 1) * n_out] = 0.0
        # subtract the ORIGINAL Qp contribution, add the masked one
        H_Qp_orig = 2.0 * (G.T @ self.Qpbar @ G)
        f_Qp_orig = 2.0 * (G.T @ self.Qpbar @ constant_error)
        H_Qp_masked = 2.0 * (G.T @ Qpbar_masked @ G)
        f_Qp_masked = 2.0 * (G.T @ Qpbar_masked @ constant_error)
        hessian = hessian - H_Qp_orig + H_Qp_masked
        linear = linear - f_Qp_orig + f_Qp_masked
        hessian = 0.5 * (hessian + hessian.T)
        return hessian, linear, terms
