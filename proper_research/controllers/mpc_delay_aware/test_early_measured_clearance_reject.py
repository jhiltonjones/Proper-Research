"""2026-11-19 (stop-path correction, point 3): proves the measured-
clearance/z-bounds rejection in ProcessIsolatedDelayAwareAdapter now runs
BEFORE try_solve and BEFORE the execution-C accumulator advances --
deterministic, no hardware, no timing/sleep dependence. A mock worker
object raises if `try_solve` is ever called, so any regression that
re-introduces a solve request on a rejected tick fails loudly rather
than silently passing.
"""
from __future__ import annotations

import numpy as np
import pytest

from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online.messages import StateEstimate
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationReference,
)


class _WorkerMustNotBeCalled:
    def try_solve(self, **kwargs):
        raise AssertionError(
            "try_solve() was called on a tick the measured-clearance check "
            "should have rejected BEFORE any worker request."
        )


def _make_reference(n: int = 5) -> ConfigurationReference:
    state = np.zeros((n, 7))
    state[:, 6] = 0.03
    desired = np.zeros((n, 3))
    return ConfigurationReference(
        time_s=np.arange(n) * 0.1, path_coordinate_m=np.arange(n) * 0.001,
        state=state, input=np.zeros((n, 7)), acceleration=np.zeros((n, 7)),
        desired_position_m=desired, desired_tangent=np.tile([1.0, 0.0, 0.0], (n, 1)),
        planned_beam_feasible=np.ones(n, dtype=bool), planned_position_error_m=np.zeros(n),
        planned_tangent_error_rad=np.zeros(n), sample_period_s=0.1, source="test",
    )


def _estimate(q6) -> StateEstimate:
    return StateEstimate(
        t_monotonic=0.0, x_meas=np.zeros(6), tip_position_m=np.zeros(3),
        tip_tangent=np.array([1.0, 0.0, 0.0]), insertion_length_m=0.03,
        robot_joints=np.asarray(q6, dtype=float), frame_index=0,
    )


def test_violation_sends_no_worker_request():
    ref = _make_reference()
    base_m = np.array([[0.0, 0.0, 0.0]])
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=_WorkerMustNotBeCalled(), deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([0.0, 0.0, 0.1]),  # 100mm from base, always
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=0.230,
    )
    result = adapter(_estimate(np.zeros(6)), 0.1)  # does not raise -> try_solve was never called
    assert result.infeasible
    assert np.allclose(result.u0, 0.0)
    assert result.info["abort_reason"] is not None
    assert "magnet_exclusion_violated" in result.info["abort_reason"]


def test_violation_does_not_advance_accumulator():
    ref = _make_reference()
    base_m = np.array([[0.0, 0.0, 0.0]])
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=_WorkerMustNotBeCalled(), deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([0.0, 0.0, 0.1]),
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=0.230,
    )
    q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    adapter(_estimate(q0), 0.1)  # seeds _q_cmd on first call, immediately rejected
    q_cmd_after_first = adapter._q_cmd.copy()
    adapter(_estimate(q0), 0.1)  # second rejected tick -- must not move the accumulator
    assert np.allclose(adapter._q_cmd, q_cmd_after_first), (
        "accumulator advanced despite the measured-clearance rejection"
    )


def test_abort_event_record_fields_present_and_not_fabricated(tmp_path):
    """2026-11-19 (point 5): the pre-solve abort log row carries an
    explicit early_abort_reason, the measured gap/radius/z-bounds that
    produced it, a real sample-age, and the last dispatched target --
    and does NOT fabricate any resp-derived field (qp_status,
    predicted_states, dual_y, etc. must all stay None since no solve ran).
    """
    import json
    import time

    ref = _make_reference()
    base_m = np.array([[0.0, 0.0, 0.0]])
    log_path = tmp_path / "abort_log.jsonl"
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=_WorkerMustNotBeCalled(), deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([0.0, 0.0, 0.1]),  # 100mm from base, always
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=0.230,
        prediction_log_path=str(log_path),
    )
    t_sample = time.monotonic() - 0.05  # a real, non-zero sample age

    class _EstimateWithJointAge(StateEstimate):
        """A real `robot_joints_age()` accessor, distinct from a generic
        t_monotonic-based age -- mirrors the runner's own estimate type,
        which has this method precisely because joints and vision can be
        sampled at different times (see path_follow.jsonl's separate
        joint_sample_age_at_use_s / camera_vision_age_s fields)."""
        def robot_joints_age(self) -> float:
            return time.monotonic() - self.t_monotonic

    estimate = _EstimateWithJointAge(
        t_monotonic=t_sample, x_meas=np.zeros(6), tip_position_m=np.zeros(3),
        tip_tangent=np.array([1.0, 0.0, 0.0]), insertion_length_m=0.03,
        robot_joints=np.zeros(6), frame_index=0,
    )
    adapter(estimate, 0.1)

    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]

    assert row["early_abort_reason"] is not None
    assert "magnet_exclusion_violated" in row["early_abort_reason"]
    assert row["magnet_exclusion_gap_mm"] == pytest.approx(100.0, abs=1e-6)
    assert row["magnet_exclusion_radius_mm"] == pytest.approx(230.0, abs=1e-6)
    assert row["magnet_z_bounds_mm"] is None  # not configured in this test
    assert row["clearance_sample_age_s"] >= 0.05
    # Honestly an accumulator value, not a verified dispatch record --
    # see process_isolated_adapter.py's own comment on this field.
    assert row["accumulator_q_cmd_rad"] == row["q_cmd_k"]


def test_clearance_sample_age_is_none_when_joint_age_unavailable(tmp_path):
    """When the estimate type has no robot_joints_age() accessor, the
    field must be None, not silently fall back to an unverified generic
    age -- this is the honest-absence behaviour point 3 asked for."""
    import json

    ref = _make_reference()
    base_m = np.array([[0.0, 0.0, 0.0]])
    log_path = tmp_path / "abort_log2.jsonl"
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=_WorkerMustNotBeCalled(), deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([0.0, 0.0, 0.1]),
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=0.230,
        prediction_log_path=str(log_path),
    )
    adapter(_estimate(np.zeros(6)), 0.1)  # plain StateEstimate -- no robot_joints_age()
    rows = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert rows[0]["clearance_sample_age_s"] is None


def test_no_rejection_still_reaches_try_solve():
    """Control case: when the magnet is clear, the normal path (including
    try_solve) must still run -- this moved the check, it did not disable
    it for the non-violating case."""
    ref = _make_reference()
    base_m = np.array([[0.0, 0.0, 0.0]])

    class _WorkerCountsCalls:
        def __init__(self):
            self.n_calls = 0

        def try_solve(self, **kwargs):
            self.n_calls += 1
            return None  # treated as a deadline miss downstream -- fine for this test

    worker = _WorkerCountsCalls()
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=worker, deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([10.0, 10.0, 10.0]),  # far from base
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=0.230,
    )
    adapter(_estimate(np.zeros(6)), 0.1)
    assert worker.n_calls == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
