#!/usr/bin/env python3
"""Regression guard for the resolved-rate controller's magnetic force/torque
actuation preference (2026-10-15) -- see `inverse_jacobian_controller.py`'s
`build_magnetic_force_preference_gradient_fn` docstring and the
nullspace_target comment in `InverseJacobianBeamController.solve`.

Checks:
  (a) force_preference_gain=0.0 (or gradient_fn=None) is an exact no-op.
  (b) a nonzero gain measurably changes the command.
  (c) the change is (almost) entirely a NULLSPACE motion: J @ command is
      unchanged up to the residual leak damped-least-squares pseudo-inverse
      already has (exact nullspace projection only holds at damping=0 --
      this is NOT claiming damping=0 is this controller's default).

Uses synthetic reference/Jacobian fixtures (no plan/schedule files needed),
mirroring `controller_adapters.run_self_test`'s own self-contained style.

Usage
-----
    python -m proper_research.controllers.inverse_jacobian_force_preference_self_test
"""
from __future__ import annotations

import numpy as np

from proper_research.simulation.simulations import (
    simulate_time_parameterized_configuration_mpc as base_module,
)
from .inverse_jacobian_controller import (
    build_inverse_jacobian_controller,
    build_magnetic_force_preference_gradient_fn,
)


def run_self_test() -> None:
    dt = 0.1
    sample_count = 20
    time_values = np.arange(sample_count, dtype=float) * dt
    state_reference = np.zeros((sample_count, 7), dtype=float)
    state_reference[:, 6] = 0.001 * np.arange(sample_count)
    input_reference = np.zeros_like(state_reference)
    desired_position = np.column_stack(
        [np.zeros(sample_count), np.zeros(sample_count), 5.0 * state_reference[:, 6]]
    )
    tangent = np.tile(np.array([1.0, 0.0, 0.0]), (sample_count, 1))

    reference = base_module.ConfigurationReference(
        time_s=time_values, path_coordinate_m=np.zeros(sample_count), state=state_reference,
        input=input_reference, acceleration=np.zeros_like(state_reference),
        desired_position_m=desired_position, desired_tangent=tangent,
        planned_beam_feasible=np.ones(sample_count, dtype=bool),
        planned_position_error_m=np.zeros(sample_count),
        planned_tangent_error_rad=np.zeros(sample_count), sample_period_s=dt,
        source="force_preference_self_test",
    )
    mpc_config = base_module.ConfigurationMPCConfig(
        sample_period_s=dt, prediction_horizon=5,
        state_min=tuple([-0.5] * 6 + [-0.02]), state_max=tuple([0.5] * 6 + [0.05]),
        velocity_limit=tuple([0.4] * 6 + [0.01]), acceleration_limit=tuple([2.0] * 6 + [0.1]),
        state_tracking_weight=1.0e-3, input_tracking_weight=1.0e-4, input_increment_weight=1.0e-4,
        solver_backend="scipy", solver_absolute_tolerance=1e-8, solver_relative_tolerance=1e-8,
    )

    def jacobian_provider(_state: np.ndarray) -> np.ndarray:
        J = np.zeros((3, 7))
        J[0, 0] = 0.3
        J[1, 1] = 0.2
        J[2, 6] = 5.0
        return J

    rng = np.random.default_rng(0)
    force_jac = rng.normal(size=(sample_count, 3, 7)) * 0.1
    torque_jac = rng.normal(size=(sample_count, 3, 7)) * 0.05
    nominal_force = np.zeros((sample_count, 3))
    nominal_torque = np.zeros((sample_count, 3))
    grad_fn = build_magnetic_force_preference_gradient_fn(
        reference_state=state_reference, force_jacobians=force_jac, torque_jacobians=torque_jac,
        nominal_force_N=nominal_force, nominal_torque_Nm=nominal_torque,
        force_desired_N=np.array([0.0, 0.0, 1.0]), torque_desired_Nm=np.zeros(3),
        lambda_force=1.0, lambda_torque=0.5,
    )

    state = np.array([0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.001])
    measured_beam_position = np.array([0.0, 0.0, 0.005])
    prev = np.zeros(7)

    def build(gain: float, use_grad_fn: bool):
        return build_inverse_jacobian_controller(
            reference=reference, jacobian_provider=jacobian_provider, mpc_config=mpc_config,
            allow_undeclared_jacobian=True,
            force_preference_gradient_fn=grad_fn if use_grad_fn else None,
            force_preference_gain=gain,
        )

    def solve(controller):
        return controller.solve(
            measured_state=state, measured_beam_position=measured_beam_position,
            control_index=3, previous_input=prev,
        )

    step_base = solve(build(0.0, use_grad_fn=False))
    step_disabled = solve(build(0.0, use_grad_fn=True))
    step_mod = solve(build(0.5, use_grad_fn=True))

    diff_disabled = float(np.max(np.abs(step_base.command - step_disabled.command)))
    assert diff_disabled < 1e-12, f"gain=0 should be an exact no-op, diff={diff_disabled:.3e}"
    print(f"[ij force_pref self-test] gain=0 EXACT no-op (diff={diff_disabled:.2e})")

    diff_mod = float(np.max(np.abs(step_base.command - step_mod.command)))
    assert diff_mod > 1e-6, "nonzero gain should measurably change the command"
    print(f"[ij force_pref self-test] gain=0.5 changes command (diff={diff_mod:.3e})")

    J = jacobian_provider(state)
    task_diff = float(np.max(np.abs(J @ step_base.command - J @ step_mod.command)))
    rel_leak = task_diff / diff_mod
    print(f"[ij force_pref self-test] task-space leak={task_diff:.3e} "
          f"({rel_leak:.3e} relative to command diff)")
    assert rel_leak < 1e-4, (
        "force-preference term leaked far more than DLS-damping noise into "
        "the task-space component -- nullspace projection is wrong"
    )

    print("[ij force_pref self-test] PASS")


if __name__ == "__main__":
    run_self_test()
