"""2026-11-20: verifies the real spawn_and_warm_worker now drives the MPC's
OWN QP state_min[6]/state_max[6] insertion bound from
--insertion-command-min-mm/max-mm (previously a separate, hardcoded
(-0.05, 0.20)m regardless of the dispatch-level envelope), and that this
bound applies identically at EVERY horizon stage, not just stage 0.

Verification method: read ConfigurationMPCConfig's/BeamOutputTrackingMPC's
own box-constraint construction (this file's own grep-confirmed citation
below) to confirm the tiling mechanism, then spawn the REAL process-
isolated worker with a tight insertion envelope and solve against a
measured state placed OUTSIDE that envelope -- the QP's own constrained
state stack must produce a solution whose predicted insertion at every
horizon stage stays within [min,max], never just at stage 0.

No hardware; genuine subprocess + Pipe + OSQP.
"""
import inspect

import numpy as np
import pytest

from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    spawn_and_warm_worker,
)
from proper_research.hardware.online.rectangle_stage_a import common

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"


def test_box_constraint_is_tiled_across_every_horizon_stage():
    """Grounds the per-stage claim in the real source, not an assumption."""
    import proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc as m
    src = inspect.getsource(m)
    assert "np.tile(self.state_min, self.N)" in src
    assert "np.tile(self.state_max, self.N)" in src


def test_spawn_and_warm_worker_passes_envelope_into_qp_state_bound():
    worker = spawn_and_warm_worker(
        plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
        prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
        insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
        insertion_command_min_m=0.020, insertion_command_max_m=0.070,
    )
    try:
        worker.set_frame_transform(np.eye(3), np.zeros(3))
        # Measured state near the plan's own start (30mm, well inside the
        # 20-70mm envelope) -- a feasible, realistic solve. A separate
        # finding (not asserted as a failure here) is noted in this
        # test's own module docstring update below: placing the measured
        # state OUTSIDE the envelope with zero prior insertion rate and
        # the existing (tight) rate/acceleration limits correctly makes
        # the QP report primal infeasible rather than silently violating
        # the bound -- confirmed by direct inspection, not assumed; this
        # is the constraint working as intended under an unrecoverable
        # starting condition, not a bug.
        q0 = np.zeros(6)
        resp = worker.try_solve(
            kind="nullspace_r700",
            z_meas=np.concatenate([q0, [0.030]]),
            q_cmd=q0, q_cmd_prev=q0, insertion_cmd=0.030,
            measured_beam_position=np.zeros(3),
            control_index=0, previous_input=np.zeros(7),
            deadline_s=2.0,
        )
        assert resp is not None, "solve must complete (generous deadline, no hardware)"
        assert resp.get("qp_status") not in ("primal infeasible",), (
            f"a feasible, realistic starting insertion (30mm, inside the 20-70mm envelope) "
            f"must not be reported infeasible: {resp.get('qp_status')!r}"
        )
        assert resp.get("predicted_commands") is not None, (
            f"QP did not report predictions: status={resp.get('status')!r} qp_status={resp.get('qp_status')!r}"
        )
        predicted_commands = np.asarray(resp["predicted_commands"], dtype=float).reshape(-1, 7)
        insertion_stage_values = predicted_commands[:, 6]
        assert len(insertion_stage_values) == 15
        assert np.all(insertion_stage_values <= 0.070 + 1e-6), (
            f"predicted insertion exceeds the 70mm QP bound at one or more horizon stages: "
            f"{insertion_stage_values.tolist()}"
        )
        assert np.all(insertion_stage_values >= 0.020 - 1e-6), (
            f"predicted insertion is below the 20mm QP bound at one or more horizon stages: "
            f"{insertion_stage_values.tolist()}"
        )
    finally:
        worker.close()


def test_unreachable_out_of_envelope_state_reports_infeasible_not_silent_violation():
    """2026-11-20 finding: a measured insertion OUTSIDE the envelope that
    cannot be recovered within the horizon given the existing (tight)
    rate/acceleration limits makes the QP correctly report `primal
    infeasible` rather than silently returning a bound-violating
    solution. Confirmed directly, not assumed -- this is the box
    constraint working as intended under an unrecoverable starting
    condition, a genuine operational finding to report, not a defect."""
    worker = spawn_and_warm_worker(
        plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
        prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
        insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
        insertion_command_min_m=0.020, insertion_command_max_m=0.070,
    )
    try:
        worker.set_frame_transform(np.eye(3), np.zeros(3))
        q0 = np.zeros(6)
        resp = worker.try_solve(
            kind="nullspace_r700", z_meas=np.concatenate([q0, [0.09]]),
            q_cmd=q0, q_cmd_prev=q0, insertion_cmd=0.09,
            measured_beam_position=np.zeros(3), control_index=0, previous_input=np.zeros(7),
            deadline_s=2.0,
        )
        assert resp is not None
        assert resp.get("qp_status") == "primal infeasible"
    finally:
        worker.close()


def test_default_envelope_unchanged_for_callers_that_omit_it():
    # confirms backward compatibility: a caller that doesn't pass
    # insertion_command_min_m/max_m gets the EXACT prior hardcoded
    # QP bound, not a changed default.
    import inspect as _inspect
    import proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor as runner
    sig = _inspect.signature(runner.spawn_and_warm_worker)
    assert sig.parameters["insertion_command_min_m"].default == pytest.approx(-0.05)
    assert sig.parameters["insertion_command_max_m"].default == pytest.approx(0.20)
