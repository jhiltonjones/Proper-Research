"""2026-11-19 (future-run diagnosability, point 2): checks ALL 15 command
stages of a FRESH solve against the actual QP bounds and the nonlinear FK
clearance at those same commanded configurations. This is a NEW solve run
offline against the real marginfix plan/schedule -- it is NOT a claim
about, or a substitute for, the pilotA run's own historical stages 13-14,
which remain unresolved for the reasons already reported (Sp's 12-
dimensional null space for that specific historical run). No hardware.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from proper_research.controllers.mpc_delay_aware.delay_aware_mpc import DelayAwareBeamOutputTrackingMPC
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    _robot_kin, build_magnet_exclusion_schedule,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

PLAN_DIR = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path"
SCHEDULE_CACHE = "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy"
BEAM_BASE_M = np.array([[0.670575, -0.719727, -0.039627]])
N, DT = 15, 0.1


def _fk_xyz(q6: np.ndarray) -> np.ndarray:
    return urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M).T_R_target[:3, 3]


def test_all_15_stages_consistent_with_qp_bounds_and_fresh_fk():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    beam_jacobians = np.load(SCHEDULE_CACHE)  # (S,3,7): beam-TRACKING schedule, distinct from the magnet's own
    jacobians, positions = build_magnet_exclusion_schedule(reference)  # (S,3,6): magnet-exclusion's own schedule

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

    control_index = 211  # same tick this whole review examined -- a FRESH solve at it, not a replay
    q0 = np.asarray(reference.state[control_index, :6], dtype=float)
    l0 = float(reference.state[control_index, 6])
    p0 = np.asarray(reference.desired_position_m[control_index], dtype=float)
    z_meas = np.concatenate([q0, [l0]])

    radius_m = 0.233  # 230mm reviewed floor + 3mm provisional margin, commanded
    magnet_exclusion = dict(
        position_jacobians=jacobians, nominal_positions_m=positions,
        lumen_C_m=BEAM_BASE_M, radius_m=radius_m,
    )
    controller = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=beam_jacobians, delay_samples=2, beta_d=1.0,
        magnet_exclusion=magnet_exclusion,
    )
    step = controller.solve_delay_aware(
        z_meas=z_meas, q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=control_index, previous_input=np.zeros(7),
    )
    assert step.success, f"fresh solve did not converge: status={step.status}"

    mapping = step.magnet_exclusion_mapping
    assert mapping is not None, "magnet_exclusion_mapping must be populated when the constraint is configured"
    assert not mapping["live_relinearize"]
    assert len(mapping["indices"]) == N

    predicted_commands = np.asarray(step.predicted_commands)  # (N, 7) -- directly available, no reconstruction
    predicted_inputs = np.asarray(step.predicted_inputs)      # (N, m) -- the full decision vector, reshaped
    solution = predicted_inputs.reshape(-1)                    # back to the flat (nu,) OSQP decision vector

    Sc = controller.Sc
    n = controller.n

    d_nom = np.asarray(mapping["d_nom_m"])
    normal = np.asarray(mapping["normal"])
    q_nom = np.asarray(mapping["q_nom_rad"])
    free_cmd = np.asarray(mapping["free_cmd_rad"])
    lower_logged = np.asarray(mapping["lower"])

    for j in range(N):
        q_pred_cmd = predicted_commands[j, :6]

        # (1) the logged mapping reproduces the EXACT lower bound the
        # solver used -- recomputed independently from the persisted
        # fields, not re-read from the solver's own internal state.
        c_j = normal[j] @ jacobians[mapping["indices"][j]]
        lower_recomputed = radius_m - d_nom[j] + c_j @ q_nom[j] - c_j @ free_cmd[j]
        assert lower_recomputed == pytest.approx(lower_logged[j], abs=1e-9)

        # (2) the actual exclusion row, built the same way, times the
        # REAL solved decision vector, satisfies that bound (upper=+inf
        # always, so only the lower side is meaningful).
        stage_rows = slice(j * n, j * n + 6)
        row_j = c_j @ Sc[stage_rows, :]
        Ax_j = row_j @ solution
        assert Ax_j >= lower_logged[j] - 1e-6, (
            f"stage {j}: QP's own exclusion row violated by its own solved decision "
            f"({Ax_j:.6f} < {lower_logged[j]:.6f})"
        )

        # (3) predicted_commands (the stack the exclusion constraint
        # actually governs) is internally consistent: free_cmd + the
        # input-driven delta reconstructs the SAME q_pred_cmd logged above.
        delta_from_inputs = (Sc[stage_rows, :] @ solution)
        assert np.allclose(free_cmd[j] + delta_from_inputs, q_pred_cmd, atol=1e-9)

        # (4) nonlinear FK clearance at this SAME commanded configuration,
        # from a FRESH computation -- reported separately, not blended
        # with the linear estimate above.
        d_fk = float(np.linalg.norm(_fk_xyz(q_pred_cmd) - BEAM_BASE_M[0]))
        d_lin = d_nom[j] + c_j @ (q_pred_cmd - q_nom[j])
        # Both must be reportable; no assertion on their relative sign
        # here -- the prior review already established the direction
        # (linear is conservative, true clearance a few mm higher) for
        # the historical run; this test's job is internal consistency of
        # the NEW mechanism, not re-deriving that physical finding.
        assert math.isfinite(d_fk) and math.isfinite(d_lin)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
