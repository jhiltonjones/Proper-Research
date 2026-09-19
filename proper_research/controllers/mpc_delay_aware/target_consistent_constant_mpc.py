"""MT-constant delay-aware MPC (2026-09-19).

Motivated by a diagnostic finding on `TargetConsistentDelayAwareMPC` (MT-
stagewise): the per-stage state-space gradient cosine cos(g^z_Q,j, g^z_Qp,j)
is STRONGLY POSITIVE at the early stages the target was locally computed
for (j=1..3: median +0.99, +0.98, +0.96 -- the static cancellation DOES
resolve the local conflict there), but flips increasingly NEGATIVE by the
back half of the horizon (j=9..15: median -0.99 to -1.00) -- because each
stage's delta_z^t_j was computed independently against that stage's own
(rotating) Jacobian, with nothing enforcing that delta_z^t_{j+1} is
reachable from delta_z^t_j in one dt. The condensed (post-S_p) u0 gradient
aggregates all 15 stages, so it still looks like near-total opposition
even though the construction was locally correct.

Fix tested here: compute ONE disturbance-correcting configuration offset,
at j=3 -- the first stage the delayed joint decision has physical
authority (d=2) -- and hold it CONSTANT across the whole horizon:

    delta_z*_k = -W^{-1} J_{k+3}^{tT} (J_{k+3}^t W^{-1} J_{k+3}^{tT})^{-1} d^t_k
    z^t_{k+j} = z_ref,k+j + delta_z*_k   for ALL j = 1..N

Since the offset no longer varies with j, z^t_{j+1} - z^t_j = z_ref,j+1 -
z_ref,j exactly, so the target-implied velocity u^t_j = u_ref,j exactly --
Q and R no longer ask the optimizer to chase incompatible state and
velocity targets. This won't cancel d_k exactly at every stage (J_j
varies), but Qp still sees the true predicted beam error and can make
the remaining correction -- this class doesn't touch Qp/R/Rd/constraints
either, same as MT-stagewise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import delay_aware_mpc as dam
from .target_consistent import build_beam_plane_projection, compute_target_shift

Array = np.ndarray


@dataclass
class TargetConsistentConstantBeamOutputMPCStep(dam.DelayAwareBeamOutputMPCStep):
    z_target: Array = None  # (N,7)
    delta_z_star: Array = None  # (7,) -- the single constant offset used this solve


class TargetConsistentConstantDelayAwareMPC(dam.DelayAwareBeamOutputTrackingMPC):
    def __init__(
        self, *, axial_axis_R: Any = (-1.0, 0.0, 0.0), normal_axis_R: Any = (0.0, 0.0, -1.0),
        authority_stage: int | None = None, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._C = build_beam_plane_projection(axial_axis_R=axial_axis_R, normal_axis_R=normal_axis_R)
        state_scale = np.asarray(self.config.state_error_scale, dtype=float).reshape(7)
        self._w_diag = 1.0 / state_scale ** 2
        # first stage the delayed joint decision has physical authority --
        # delay_samples=2 means u_0 first affects q_phys at stage j=3
        # (1-indexed); default derives this from delay_samples so it stays
        # correct if delay_samples is ever changed.
        self._authority_stage_1idx = (
            authority_stage if authority_stage is not None else self.delay_samples + 1
        )

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)  # z_ref, UNCHANGED
        input_reference = self.reference.input_window(control_index, self.N)

        indices = self._reference_indices(control_index, future=True)
        auth_idx = indices[self._authority_stage_1idx - 1]
        J_auth = np.asarray(self.reference_position_jacobians[auth_idx], dtype=float)
        d_k = self.beta_d * np.asarray(estimated_residual, dtype=float).reshape(3)
        delta_z_star, _ = compute_target_shift(J_stage=J_auth, d_k=d_k, C=self._C, w_diag=self._w_diag)
        z_target = state_reference + delta_z_star[None, :]  # broadcast: SAME offset at every stage

        base_linear = self._linear_cost_exec(
            x_exec=x_exec, previous_input=previous_input,
            state_reference=z_target, input_reference=input_reference,
        )
        G, constant_error, _, desired_position, nominal_position, Jbar = (
            self._beam_prediction_terms_exec(
                x_exec=x_exec, control_index=control_index,
                estimated_residual=estimated_residual,
            )
        )
        hessian = self._base_hessian + 2.0 * (G.T @ self.Qpbar @ G)
        linear = base_linear + 2.0 * (G.T @ self.Qpbar @ constant_error)
        hessian = 0.5 * (hessian + hessian.T)
        return hessian, linear, {
            "G": G, "constant_error": constant_error,
            "desired_position": desired_position, "nominal_position": nominal_position,
            "Jbar": Jbar, "input_reference": input_reference,
            "z_target": z_target, "delta_z_star": delta_z_star,
        }

    def solve_delay_aware(self, **kwargs) -> TargetConsistentConstantBeamOutputMPCStep:
        base_step = super().solve_delay_aware(**kwargs)
        z_meas = np.asarray(kwargs["z_meas"], dtype=float).reshape(7)
        beam_position = np.asarray(kwargs["measured_beam_position"], dtype=float).reshape(3)
        control_index = kwargs["control_index"]
        _, estimated = self._estimate_output_residual(
            measured_state=z_meas, measured_beam_position=beam_position, control_index=control_index,
        )
        state_reference = self.reference.state_window(control_index, self.N)
        indices = self._reference_indices(control_index, future=True)
        auth_idx = indices[self._authority_stage_1idx - 1]
        J_auth = np.asarray(self.reference_position_jacobians[auth_idx], dtype=float)
        d_k = self.beta_d * np.asarray(estimated, dtype=float).reshape(3)
        delta_z_star, _ = compute_target_shift(J_stage=J_auth, d_k=d_k, C=self._C, w_diag=self._w_diag)
        z_target = state_reference + delta_z_star[None, :]

        import dataclasses
        return TargetConsistentConstantBeamOutputMPCStep(
            **{f.name: getattr(base_step, f.name) for f in dataclasses.fields(base_step)
               if f.name not in ("z_target", "delta_z_star")},
            z_target=z_target, delta_z_star=delta_z_star,
        )
