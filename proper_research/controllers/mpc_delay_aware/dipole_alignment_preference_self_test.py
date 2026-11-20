#!/usr/bin/env python3
"""Regression guard for the in-QP dipole-to-point alignment preference
(2026-10-19) -- see `delay_aware_mpc.py`'s `_configure_dipole_alignment_
preference` docstring, and `controllers.dipole_alignment` for why this
replaces `magnetic_force_preference` for "point the magnet somewhere"
purposes (torque minimization can be satisfied by distance alone; this
term cannot, by construction). Mirrors `magnetic_force_preference_self_
test.py`'s structure exactly (same plan/schedule fixtures, same checks),
since this term is also live-relinearized (no schedule, no model needed
at all -- pure kinematics), its own setup is simpler.

Usage
-----
    python -m proper_research.controllers.mpc_delay_aware.dipole_alignment_preference_self_test
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
from proper_research.rig_calibration import beam_base_pose6

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC
from .exact_qn_zero_mpc import ExactQNZeroTaskNullspaceDelayAwareMPC

PLAN_DIR = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_2026-10-12/time_parameterized_configuration_path"
SCHEDULE_CACHE = "/tmp/vessel_c_schedule_floor240_BACKUP.npy"


def run_self_test(control_index: int = 60) -> None:
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    schedule = np.load(SCHEDULE_CACHE)
    robot_kin = _resolve_robot_kinematics(make_robot_config())
    beam_base_xyz = beam_base_pose6()[:3]
    m_body = np.array([0.0, 0.0, 1.0])  # synthetic -- this test checks QP WIRING, not real physics

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

    dipole_pref = dict(
        dh=robot_kin.dh, T_F_M=robot_kin.T_F_M, m_body=m_body,
        target_point_m=beam_base_xyz, weight=5.0e4, sign=1.0,
    )

    for cls in (DelayAwareBeamOutputTrackingMPC, ExactQNZeroTaskNullspaceDelayAwareMPC):
        extra = {} if cls is DelayAwareBeamOutputTrackingMPC else dict(gamma=0.0)
        baseline = cls(**extra, **common)
        disabled = cls(dipole_alignment_preference=None, **extra, **common)
        diff = float(np.max(np.abs(solve(baseline).command - solve(disabled).command)))
        assert diff < 1e-9, f"{cls.__name__}: dipole_alignment_preference=None should be a no-op, diff={diff:.3e}"
        print(f"[dipole_align self-test] {cls.__name__}: disabled == baseline EXACT (diff={diff:.2e})")

        modified = cls(dipole_alignment_preference=dipole_pref, **extra, **common)
        step_base = solve(baseline)
        step_mod = solve(modified)
        assert step_mod.success, f"{cls.__name__}: dipole-alignment solve failed to converge"
        diff_mod = float(np.max(np.abs(step_base.command - step_mod.command)))
        assert diff_mod > 1e-6, f"{cls.__name__}: active preference should change the command, diff={diff_mod:.3e}"
        print(f"[dipole_align self-test] {cls.__name__}: active preference changes command (diff={diff_mod:.3e})")

        hessian, _, _ = modified._dynamic_qp_terms_exec(
            x_exec=np.concatenate([q0, q0, [l0]]), previous_input=prev,
            control_index=control_index, estimated_residual=np.zeros(3),
        )
        asym = float(np.max(np.abs(hessian - hessian.T)))
        assert asym < 1e-9, f"{cls.__name__}: dipole-alignment Hessian not symmetric, asym={asym:.3e}"

    # OSQP vs scipy backend agreement, base class, moderate weight (avoid
    # the same ill-conditioning the force-preference self-test's own
    # comment warns about at extreme weights).
    moderate_pref = dict(dipole_pref)
    moderate_pref["weight"] = 2.0e2
    scipy_config = dc_replace(mpc_config, solver_backend="scipy")
    modified_osqp = DelayAwareBeamOutputTrackingMPC(dipole_alignment_preference=moderate_pref, **common)
    modified_scipy = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=scipy_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        dipole_alignment_preference=moderate_pref,
    )
    step_osqp = solve(modified_osqp)
    step_scipy = solve(modified_scipy)
    assert step_scipy.success, "scipy backend dipole-alignment solve failed to converge"
    diff_backend = float(np.max(np.abs(step_osqp.command - step_scipy.command)))
    assert diff_backend < 1e-4, f"OSQP vs scipy backend disagreement too large: {diff_backend:.3e}"
    print(f"[dipole_align self-test] OSQP vs scipy agree (diff={diff_backend:.3e})")

    print("[dipole_align self-test] PASS")


if __name__ == "__main__":
    run_self_test()
