#!/usr/bin/env python3
"""Regression guard for the in-QP magnetic force/torque actuation preference
(2026-10-15) -- see `delay_aware_mpc.py`'s `_configure_magnetic_force_
preference` docstring. Mirrors `magnet_workspace_self_test.py`'s own
structure (same plan/schedule fixtures, same baseline-vs-modified diff
checks), adapted to a SOFT cost term rather than a hard constraint: there is
no "binding/not binding" notion here, so the checks are (a) disabled is an
exact no-op, (b) a strong enough preference measurably changes the command,
(c) the Hessian stays symmetric and the solve still succeeds with the term
active, (d) OSQP and scipy backends agree, (e) the duplicate wiring this
project's exact-Q_N=0 subclass requires (see `exact_qn_zero_mpc.py`) is
exercised too -- the live vessel controller uses THAT class, not the base
one, so a self-test that only covers the base class would miss a regression
there entirely (exactly the gap `magnet_clear_gain` had to be duplicated to
avoid).

The force/torque Jacobian schedule here is SYNTHETIC (built from the
magnet-position schedule the existing fixtures already provide, padded with
a zero insertion column) -- it is not claimed to be physically accurate;
this test is only checking the QP WIRING, not the physics module (that is
exercised directly by `simulation/magnetic_beam/force_preference.py`'s own
usage against a real beam model).

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.magnetic_force_preference_self_test
"""
from __future__ import annotations

import math
from dataclasses import replace as dc_replace

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)
from proper_research.planning.planning_context import make_robot_config

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC
from .exact_qn_zero_mpc import ExactQNZeroTaskNullspaceDelayAwareMPC

PLAN_DIR = "plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path"
SCHEDULE_CACHE = "/tmp/vessel_live_trimmed6mm_schedule_frozen.npy"


def _build_synthetic_force_schedule(reference, robot_kin):
    """(3,7)-per-sample force/torque Jacobian + nominal, built by padding
    the magnet POSITION Jacobian (already validated elsewhere) with a zero
    insertion column -- a convenient stand-in with the right shape/
    finiteness properties, not a physically accurate wrench model (see
    module docstring)."""
    q = np.asarray(reference.state[:, :6], dtype=float)
    S = q.shape[0]
    force_jac = np.zeros((S, 3, 7), dtype=float)
    nominal_force = np.zeros((S, 3), dtype=float)
    for i in range(S):
        J6 = urik.geometric_jacobian(q[i], robot_kin.dh, robot_kin.T_F_M)[:3, :]
        force_jac[i, :, :6] = 0.1 * J6  # arbitrary finite scale, not physical units
        nominal_force[i] = 0.0
    torque_jac = 0.5 * force_jac
    nominal_torque = np.zeros((S, 3), dtype=float)
    return force_jac, nominal_force, torque_jac, nominal_torque


def run_self_test(control_index: int = 60) -> None:
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    schedule = np.load(SCHEDULE_CACHE)
    robot_kin = _resolve_robot_kinematics(make_robot_config())
    force_jac, nominal_force, torque_jac, nominal_torque = _build_synthetic_force_schedule(
        reference, robot_kin,
    )

    N, DT = 15, 0.1
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [0.0]), state_max=tuple([2 * math.pi] * 6 + [0.065]),
        velocity_limit=tuple([0.10] * 6 + [2.0e-3]), acceleration_limit=tuple([0.40] * 6 + [0.02]),
        input_tracking_weight=1.0e-2 * 700.0, input_increment_weight=1.0e-3,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0, wall_avoidance_gain=0.0,
    )

    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float)
    z_meas = np.concatenate([q0, [l0]])
    prev = np.zeros(7)
    common = dict(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )

    def solve(mpc):
        return mpc.solve_delay_aware(
            z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
            measured_beam_position=p0, control_index=control_index, previous_input=prev,
        )

    # (a) disabled -> exact no-op, both on the base class and on the
    # exact-Q_N=0 subclass the live vessel controller actually runs.
    for cls in (DelayAwareBeamOutputTrackingMPC, ExactQNZeroTaskNullspaceDelayAwareMPC):
        extra = {} if cls is DelayAwareBeamOutputTrackingMPC else dict(gamma=0.0)
        baseline = cls(**extra, **common)
        disabled = cls(magnetic_force_preference=None, **extra, **common)
        diff = float(np.max(np.abs(solve(baseline).command - solve(disabled).command)))
        assert diff < 1e-9, f"{cls.__name__}: magnetic_force_preference=None should be a no-op, diff={diff:.3e}"
        print(f"[force_pref self-test] {cls.__name__}: disabled == baseline EXACT (diff={diff:.2e})")

    force_pref = dict(
        force_jacobians=force_jac, torque_jacobians=torque_jac,
        nominal_force_N=nominal_force, nominal_torque_Nm=nominal_torque,
        force_desired_N=np.array([0.0, 0.0, 5.0]), torque_desired_Nm=np.array([0.0, 0.0, 0.0]),
        lambda_force=2.0e2, lambda_torque=0.0,
    )

    # (b)/(c) a strong preference measurably changes the command and the
    # solve still succeeds, for BOTH classes.
    for cls in (DelayAwareBeamOutputTrackingMPC, ExactQNZeroTaskNullspaceDelayAwareMPC):
        extra = {} if cls is DelayAwareBeamOutputTrackingMPC else dict(gamma=0.0)
        baseline = cls(**extra, **common)
        modified = cls(magnetic_force_preference=force_pref, **extra, **common)
        step_base = solve(baseline)
        step_mod = solve(modified)
        assert step_mod.success, f"{cls.__name__}: force-preference solve failed to converge"
        diff = float(np.max(np.abs(step_base.command - step_mod.command)))
        assert diff > 1e-6, f"{cls.__name__}: strong force preference should change the command, diff={diff:.3e}"
        print(f"[force_pref self-test] {cls.__name__}: active preference changes command (diff={diff:.3e})")

        # Hessian the solver actually used must be exactly symmetric (both
        # the base and the OSQP-sparsity-pattern paths read off of it).
        hessian, _, _ = modified._dynamic_qp_terms_exec(
            x_exec=np.concatenate([q0, q0, [l0]]), previous_input=prev,
            control_index=control_index, estimated_residual=np.zeros(3),
        )
        asym = float(np.max(np.abs(hessian - hessian.T)))
        assert asym < 1e-9, f"{cls.__name__}: force-preference Hessian not symmetric, asym={asym:.3e}"

    # (d) OSQP vs scipy backend agreement, base class.
    scipy_config = dc_replace(mpc_config, solver_backend="scipy")
    modified_osqp = DelayAwareBeamOutputTrackingMPC(magnetic_force_preference=force_pref, **common)
    modified_scipy = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=scipy_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        magnetic_force_preference=force_pref,
    )
    step_osqp = solve(modified_osqp)
    step_scipy = solve(modified_scipy)
    assert step_scipy.success, "scipy backend force-preference solve failed to converge"
    diff_backend = float(np.max(np.abs(step_osqp.command - step_scipy.command)))
    assert diff_backend < 1e-4, f"OSQP vs scipy backend disagreement too large: {diff_backend:.3e}"
    print(f"[force_pref self-test] OSQP vs scipy agree (diff={diff_backend:.3e})")

    print("[force_pref self-test] PASS")


if __name__ == "__main__":
    run_self_test()
