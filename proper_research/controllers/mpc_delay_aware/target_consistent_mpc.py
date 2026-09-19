"""Target-consistent delay-aware MPC (MT), 2026-09-19.

Subclasses `DelayAwareBeamOutputTrackingMPC` (M0) and changes exactly one
thing: the Q-cost tracking TARGET, from the raw reference z_ref to a
disturbance-compensated z^t (see `target_consistent.py`'s module docstring
for the full derivation and the real-data finding that motivated it).
Everything else -- d=2, beta_d=1, N=15, V_f=0, Qp/R/Rd, the LTV schedule,
execution C, hard constraints, the physical prediction equation -- is
untouched. H is bit-identical to M0 (shifting a quadratic's center only
changes its linear term); only `_linear_cost_exec`'s `state_reference`
argument changes, and ONLY there -- `_beam_prediction_terms_exec` (the
physical prediction / task residual) still uses the raw z_ref internally,
exactly as M0's base class implementation already does, since this class
does not override it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import delay_aware_mpc as dam
from .target_consistent import build_beam_plane_projection, build_target_stack

Array = np.ndarray


@dataclass
class TargetConsistentBeamOutputMPCStep(dam.DelayAwareBeamOutputMPCStep):
    z_target: Array = None  # (N,7) -- the disturbance-compensated Q-cost target used this solve
    target_residual: Array = None  # (N,2) -- in-plane task residual of z_target's cancellation


class TargetConsistentDelayAwareMPC(dam.DelayAwareBeamOutputTrackingMPC):
    def __init__(
        self, *, axial_axis_R: Any = (-1.0, 0.0, 0.0), normal_axis_R: Any = (0.0, 0.0, -1.0),
        **kwargs,
    ) -> None:
        """`axial_axis_R`/`normal_axis_R`: SAME beam-plane convention
        `close_loop_path_follow.py`'s `project_error_to_beam_plane` uses
        (its own config defaults, reproduced here as this class's
        defaults) -- pass the live run's actual values if they ever
        differ, so the target calculator solves the identical task."""
        super().__init__(**kwargs)
        self._C = build_beam_plane_projection(axial_axis_R=axial_axis_R, normal_axis_R=normal_axis_R)
        state_scale = np.asarray(self.config.state_error_scale, dtype=float).reshape(7)
        self._w_diag = 1.0 / state_scale ** 2  # SAME scales as Q, no state_tracking_weight factor
        # (a uniform scalar on W doesn't change the minimum-norm DIRECTION,
        # only its overall magnitude would, and W's role here is purely a
        # normalization metric for choosing among many solutions of an
        # equality constraint -- see target_consistent.py)

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)  # z_ref, UNCHANGED
        input_reference = self.reference.input_window(control_index, self.N)

        indices = self._reference_indices(control_index, future=True)
        J_schedule = np.asarray(self.reference_position_jacobians, dtype=float)[indices]
        d_k = self.beta_d * np.asarray(estimated_residual, dtype=float).reshape(3)
        z_target, target_residual = build_target_stack(
            state_reference=state_reference, J_schedule=J_schedule,
            d_k=d_k, C=self._C, w_diag=self._w_diag,
        )

        # ONLY the Q-cost linear term uses z_target -- H is untouched by
        # this (shifting a quadratic's center changes only its linear
        # term), and _beam_prediction_terms_exec below still internally
        # re-derives state_reference = z_ref itself (unmodified base-class
        # code, not overridden here), so the physical prediction / task
        # residual / constant_error stay linearized about z_ref exactly as
        # in M0.
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
            "z_target": z_target, "target_residual": target_residual,
        }

    def solve_delay_aware(self, **kwargs) -> TargetConsistentBeamOutputMPCStep:
        base_step = super().solve_delay_aware(**kwargs)
        # re-derive z_target/target_residual for the ACTUAL tick solved (the
        # base class's solve_delay_aware already called
        # _dynamic_qp_terms_exec internally with these exact inputs; redoing
        # it here is a cheap, side-effect-free recomputation, not a second
        # solve, so this stays a thin wrapper rather than duplicating
        # solve_delay_aware's full body).
        z_meas = kwargs["z_meas"]
        beam_position = kwargs["measured_beam_position"]
        control_index = kwargs["control_index"]
        _, estimated = self._estimate_output_residual(
            measured_state=np.asarray(z_meas, dtype=float).reshape(7),
            measured_beam_position=np.asarray(beam_position, dtype=float).reshape(3),
            control_index=control_index,
        )
        state_reference = self.reference.state_window(control_index, self.N)
        indices = self._reference_indices(control_index, future=True)
        J_schedule = np.asarray(self.reference_position_jacobians, dtype=float)[indices]
        d_k = self.beta_d * np.asarray(estimated, dtype=float).reshape(3)
        z_target, target_residual = build_target_stack(
            state_reference=state_reference, J_schedule=J_schedule,
            d_k=d_k, C=self._C, w_diag=self._w_diag,
        )
        return TargetConsistentBeamOutputMPCStep(
            **{f.name: getattr(base_step, f.name) for f in dataclasses.fields(base_step)
               if f.name not in ("z_target", "target_residual")},
            z_target=z_target, target_residual=target_residual,
        )
