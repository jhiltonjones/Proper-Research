"""Delay-aware beam-output MPC (2026-09-18).

Subclasses `BeamOutputTrackingMPC` rather than mutating it -- the original
must stay behaviourally intact for the `current MPC` vs `delay-aware MPC`
ablation this was built for. Reuses everything that genuinely doesn't
change (weights Qbar/Rbar/Rdbar/Qpbar, the OSQP interface, velocity/rate
limits, the beam Jacobian schedule, `_estimate_output_residual`) and
overrides only the prediction-dependent pieces.

Three explicitly separate objects, kept separate all the way through (see
this package's earlier design discussion -- the whole point of NOT reusing
the old single `measured_state` argument for two different meanings):

    z_meas (7,)   = [q_meas(6), L_meas(1)]   -- feeds ONLY the disturbance
                                                 estimate d_k (unchanged
                                                 from the base class: "given
                                                 where the robot actually is
                                                 now, how wrong is the beam
                                                 model")
    x_exec (13,)  = [q_cmd(6), q_cmd_prev(6), L(1)]  -- feeds ONLY the
                                                 condensed prediction (what
                                                 the optimizer predicts will
                                                 happen), via the validated
                                                 `mpc_delay_aware.prediction`
                                                 module
    previous_input (7,) = u_applied,k-1        -- feeds ONLY the rate
                                                 (Delta-u) constraint and
                                                 cost, unchanged, exactly the
                                                 `u_prev` bookkeeping already
                                                 audited earlier in this
                                                 investigation

Two condensed representations of the SAME decision vector v are built from
x_exec, for two DIFFERENT purposes (this is the one part of the design that
is easy to get backwards, so it is spelled out everywhere it's used):

    (Ep, Sp)  physical-state stack  z_phys_stack = Ep@x_exec + Sp@v
              -- feeds the Q state-tracking cost AND the beam prediction
              (delay_samples=2: q_phys lags the command by 2 samples)

    (Ec, Sc)  command-state stack   q_cmd_stack  = Ec@x_exec + Sc@v
              -- feeds ONLY the joint-position box (safety) constraint,
              because q_cmd is the absolute target actually sent to
              servoJ (delay_samples=0: no shift, standard accumulator)

No DARE terminal cost (V_f=0) -- `beam_config.use_dare_terminal_cost` MUST
be False; this class raises otherwise rather than trying to reinterpret the
augmented-state terminal Riccati machinery for a 13D state it was never
derived for. `directional_damping` and `wall_avoidance_gain` must also be 0
(unused optional cost terms this development controller doesn't touch --
raising rather than silently ignoring them if set).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from proper_research.simulation.simulations import (
    simulate_time_parameterized_beam_output_mpc as beam_module,
)

from .prediction import build_delay_prediction_matrices

Array = np.ndarray
_finite_vector = beam_module._finite_vector  # noqa: SLF001 -- reuse, don't duplicate

__all__ = ["DelayAwareBeamOutputMPCStep", "DelayAwareBeamOutputTrackingMPC"]


@dataclass
class DelayAwareBeamOutputMPCStep(beam_module.BeamOutputMPCStep):
    predicted_commands: Array = None  # (N, 7) -- q_cmd_stack, NOT the physical prediction


class DelayAwareBeamOutputTrackingMPC(beam_module.BeamOutputTrackingMPC):
    """See module docstring. `delay_samples` in {0, 1, 2} (see
    `mpc_delay_aware.prediction`'s docstring for why 2 is the only
    validated, and 0-is-a-regression-test-only, value); `beta_d` scales the
    disturbance term uniformly across the horizon (beta_d=1 for the first
    ablation, per the frozen 2026-09-18 investigation sequence; beta_d=0.82
    is the earned-but-not-yet-adopted refinement -- see this package's
    README)."""

    def __init__(
        self,
        *,
        reference: Any,
        config: Any,
        beam_config: Any,
        reference_position_jacobians: Any,
        nominal_reference_positions_m: Any | None = None,
        delay_samples: int = 2,
        beta_d: float = 1.0,
    ) -> None:
        if beam_config.use_dare_terminal_cost:
            raise ValueError(
                "DelayAwareBeamOutputTrackingMPC requires "
                "beam_config.use_dare_terminal_cost=False (V_f=0) -- see module "
                "docstring for why this development controller does not adapt "
                "the terminal Riccati machinery to the augmented 13D state."
            )
        if float(beam_config.directional_damping) != 0.0:
            raise ValueError("directional_damping must be 0 for this development controller.")
        if float(beam_config.wall_avoidance_gain) != 0.0:
            raise ValueError("wall_avoidance_gain must be 0 for this development controller.")
        super().__init__(
            reference=reference, config=config, beam_config=beam_config,
            reference_position_jacobians=reference_position_jacobians,
            nominal_reference_positions_m=nominal_reference_positions_m,
        )
        self.delay_samples = int(delay_samples)
        self.beta_d = float(beta_d)
        n_joints = self.n - 1  # 6

        self.Ep, self.Sp = build_delay_prediction_matrices(
            N=self.N, dt=self.dt, delay_samples=self.delay_samples, n_joints=n_joints,
        )
        self.Ec, self.Sc = build_delay_prediction_matrices(
            N=self.N, dt=self.dt, delay_samples=0, n_joints=n_joints,
        )

        # Rebuild H using Sp (physical-state stack) in place of the base
        # class's S (which used the old undelayed n=7 model for EVERYTHING,
        # cost included) -- the R/Rd terms (Rbar, D.T@Rdbar@D) are pure
        # input-side costs, unaffected by the state augmentation, so they
        # carry over unchanged.
        base_hessian = 2.0 * (
            self.Sp.T @ self.Qbar @ self.Sp + self.Rbar + self.D.T @ self.Rdbar @ self.D
        )
        base_hessian += float(self.config.hessian_regularization) * np.eye(self.nu)
        self.H = 0.5 * (base_hessian + base_hessian.T)
        self._base_hessian = np.asarray(self.H, dtype=float).copy()
        if self.backend == "osqp":
            # Rebuild the OSQP sparsity pattern for the corrected H -- see
            # the Q-ablation NullAwareMPC precedent in
            # hardware/online/rectangle_stage_a/q_ablation.py for why
            # `_setup_osqp()` (sparsity from H alone) is wrong here and
            # `_setup_variable_hessian_osqp()` (includes the dense runtime
            # G.T@Qpbar@G term) is required.
            self._setup_variable_hessian_osqp()

    # ------------------------------------------------------------------
    # overridden prediction-dependent pieces
    # ------------------------------------------------------------------
    def _constraint_bounds_exec(
        self, *, x_exec: Array, previous_input: Array,
    ) -> tuple[Array, Array]:
        """Same as the base class's `_constraint_bounds`, EXCEPT the joint-
        position box uses (Ec, Sc) -- the COMMAND stack -- not the physical
        prediction. Velocity/rate terms are untouched (still act on the
        decision v and previous_input exactly as before)."""
        previous_input = _finite_vector(previous_input, self.m, "previous_input")
        input_lower = np.tile(-self.velocity_limit, self.N)
        input_upper = np.tile(self.velocity_limit, self.N)
        free_cmd = self.Ec @ x_exec
        state_lower = np.tile(self.state_min, self.N) - free_cmd
        state_upper = np.tile(self.state_max, self.N) - free_cmd
        change = np.zeros(self.nu, dtype=float)
        change[: self.m] = previous_input
        delta = np.tile(self.dt * self.acceleration_limit, self.N)
        increment_lower = change - delta
        increment_upper = change + delta
        return (
            np.concatenate((input_lower, state_lower, increment_lower)),
            np.concatenate((input_upper, state_upper, increment_upper)),
        )

    def _linear_cost_exec(
        self, *, x_exec: Array, previous_input: Array,
        state_reference: Array, input_reference: Array,
    ) -> Array:
        free_state = self.Ep @ x_exec  # PHYSICAL stack, not command stack
        state_reference_vector = state_reference.reshape(self.nu)
        input_reference_vector = input_reference.reshape(self.nu)
        previous_vector = np.zeros(self.nu, dtype=float)
        previous_vector[: self.m] = previous_input
        return 2.0 * (
            self.Sp.T @ self.Qbar @ (free_state - state_reference_vector)
            - self.Rbar @ input_reference_vector
            - self.D.T @ self.Rdbar @ previous_vector
        )

    def _beam_prediction_terms_exec(
        self, *, x_exec: Array, control_index: int, estimated_residual: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        indices = self._reference_indices(control_index, future=True)
        state_reference = np.asarray(self.reference.state, dtype=float)[indices]
        desired_position = np.asarray(self.reference.desired_position_m, dtype=float)[indices]
        nominal_position = self.nominal_reference_positions_m[indices]
        jacobians = self.reference_position_jacobians[indices]
        Jbar = sp.block_diag(list(jacobians), format="csc").toarray()
        state_reference_vector = state_reference.reshape(self.nu)
        free_state = self.Ep @ x_exec  # PHYSICAL stack
        G = Jbar @ self.Sp
        constant_error = (
            nominal_position.reshape(3 * self.N)
            - desired_position.reshape(3 * self.N)
            + Jbar @ (free_state - state_reference_vector)
            + self.beta_d * np.tile(estimated_residual, self.N)
        )
        return G, constant_error, state_reference, desired_position, nominal_position, Jbar

    def _dynamic_qp_terms_exec(
        self, *, x_exec: Array, previous_input: Array, control_index: int,
        estimated_residual: Array,
    ) -> tuple[Array, Array, dict[str, Array]]:
        state_reference = self.reference.state_window(control_index, self.N)
        input_reference = self.reference.input_window(control_index, self.N)
        base_linear = self._linear_cost_exec(
            x_exec=x_exec, previous_input=previous_input,
            state_reference=state_reference, input_reference=input_reference,
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
        }

    # ------------------------------------------------------------------
    # public entry point -- deliberately NOT `measured_state` alone
    # ------------------------------------------------------------------
    def solve_delay_aware(
        self,
        *,
        z_meas: Any,
        q_cmd: Any,
        q_cmd_prev: Any,
        insertion_m: float,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> DelayAwareBeamOutputMPCStep:
        """`z_meas`: [q_meas(6), L(1)] -- feeds ONLY the disturbance estimate.
        `q_cmd`/`q_cmd_prev`: the execution-C accumulator's OWN current and
        previous commanded joints -- feeds ONLY the prediction, via x_exec.
        `previous_input`: u_applied,k-1 -- feeds ONLY rate continuity."""
        z_meas = _finite_vector(z_meas, 7, "z_meas")
        q_cmd = _finite_vector(q_cmd, 6, "q_cmd")
        q_cmd_prev = _finite_vector(q_cmd_prev, 6, "q_cmd_prev")
        beam_position = _finite_vector(measured_beam_position, 3, "measured_beam_position")
        previous = _finite_vector(previous_input, 7, "previous_input")
        x_exec = np.concatenate([q_cmd, q_cmd_prev, [float(insertion_m)]])

        instantaneous, estimated = self._estimate_output_residual(
            measured_state=z_meas, measured_beam_position=beam_position,
            control_index=control_index,
        )
        hessian, linear_cost, beam_terms = self._dynamic_qp_terms_exec(
            x_exec=x_exec, previous_input=previous, control_index=control_index,
            estimated_residual=estimated,
        )
        input_reference = beam_terms["input_reference"]
        lower, upper = self._constraint_bounds_exec(x_exec=x_exec, previous_input=previous)
        warm_start = self._feasible_warm_start(
            input_reference=input_reference, previous_input=previous,
        )

        if self.backend == "osqp":
            assert self._solver is not None
            self._solver.update(Px=self._upper_values(hessian), q=linear_cost, l=lower, u=upper)
            self._solver.warm_start(x=warm_start)
            result = self._solver.solve()
            status = str(result.info.status).lower()
            success = status in {"solved", "solved inaccurate"}
            solution = None if result.x is None else np.asarray(result.x, dtype=float).reshape(self.nu)
            diagnostic = {
                "status": status, "success": success,
                "iterations": int(result.info.iter), "solve_time_s": float(result.info.run_time),
                "primal_residual": float(getattr(result.info, "prim_res", getattr(result.info, "pri_res", np.nan))),
                "dual_residual": float(getattr(result.info, "dual_res", getattr(result.info, "dua_res", np.nan))),
            }
        else:
            solution, diagnostic = self._solve_scipy_dynamic(
                hessian=hessian, linear_cost=linear_cost, lower=lower, upper=upper,
                warm_start=warm_start,
            )

        if solution is None or not bool(diagnostic["success"]):
            command = np.zeros(7, dtype=float)
            predicted_inputs = np.zeros((self.N, self.m), dtype=float)
            predicted_states = np.tile(np.concatenate([q_cmd, [insertion_m]]), (self.N, 1))
            predicted_commands = predicted_states.copy()
            objective = np.inf
        else:
            predicted_inputs = solution.reshape(self.N, self.m)
            predicted_states = (self.Ep @ x_exec + self.Sp @ solution).reshape(self.N, self.n)
            predicted_commands = (self.Ec @ x_exec + self.Sc @ solution).reshape(self.N, self.n)
            command = predicted_inputs[0].copy()
            objective = float(0.5 * solution @ hessian @ solution + linear_cost @ solution)
            self._warm_start = solution.copy()

        predicted_beam_vector = (
            beam_terms["nominal_position"].reshape(3 * self.N)
            + beam_terms["Jbar"] @ (
                predicted_states.reshape(self.nu)
                - self.reference.state_window(control_index, self.N).reshape(self.nu)
            )
            + self.beta_d * np.tile(estimated, self.N)
        )
        predicted_beam = predicted_beam_vector.reshape(self.N, 3)
        predicted_error = predicted_beam - beam_terms["desired_position"]
        first_error = float(np.linalg.norm(predicted_error[0]))

        return DelayAwareBeamOutputMPCStep(
            command=command,
            planned_input=np.asarray(input_reference[0], dtype=float).copy(),
            predicted_states=predicted_states,
            predicted_commands=predicted_commands,
            predicted_inputs=predicted_inputs,
            objective=objective,
            status=str(diagnostic["status"]), success=bool(diagnostic["success"]),
            iterations=int(diagnostic["iterations"]), solve_time_s=float(diagnostic["solve_time_s"]),
            primal_residual=float(diagnostic["primal_residual"]), dual_residual=float(diagnostic["dual_residual"]),
            measured_beam_position=beam_position.copy(),
            instantaneous_output_residual=instantaneous.copy(),
            estimated_output_residual=estimated.copy(),
            predicted_beam_positions=predicted_beam,
            predicted_beam_errors=predicted_error,
            first_predicted_beam_error_m=first_error,
        )
