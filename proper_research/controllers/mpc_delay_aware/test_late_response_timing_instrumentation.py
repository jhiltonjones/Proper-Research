"""2026-11-19 (minimal async request/response timing instrumentation,
points 2-3): proves that a response arriving AFTER its own deadline
already expired (a) is NEVER returned to the caller -- the caller's
hold/fallback behaviour is unchanged -- (b) its timing is persisted to a
REAL file (request id, reference index, parent send/timeout/drain times,
worker-reported timing, the gain that request's worker actually used),
with drain delay kept separate from worker response-arrival latency, and
(c) a request still outstanding when the worker is closed is recorded as
UNRESOLVED, not silently lost. Uses the REAL MPCWorkerHandle (a genuine
multiprocessing.Process + Pipe + OSQP backend), not a mock. No hardware.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCRATCH_DIR = os.environ.get("CLAUDE_JOB_TMP", "/tmp")
SCHEDULE_PATH = os.path.join(SCRATCH_DIR, "test_late_response_schedule.npy")
LOG_PATH = os.path.join(SCRATCH_DIR, "test_late_response_log.jsonl")


def _build_synthetic_schedule() -> str:
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    sched = np.tile(J0[None, :, :], (ref.sample_count, 1, 1))
    np.save(SCHEDULE_PATH, sched)
    return SCHEDULE_PATH


def _make_worker(log_path, via_setter: bool = False):
    schedule_path = _build_synthetic_schedule()
    worker = MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=schedule_path,
        mpc_config_kwargs=dict(
            sample_period_s=0.1, prediction_horizon=10,
            state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
            velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        ),
        beam_config_kwargs=dict(
            position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
            use_dare_terminal_cost=False, directional_damping=0.0,
        ),
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True,
        nullspace_r700_config_kwargs=dict(
            sample_period_s=0.1, prediction_horizon=10,
            state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
            velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        ),
        late_response_log_path=(None if via_setter else log_path),
    )
    if via_setter:
        # 2026-11-19 (point 2a): exercises the SAME mechanism
        # run_mpc_delay_aware_insertion_anchor.py now uses -- the worker
        # is spawned before the real runner's own output directory
        # exists, so the path is set afterward via this method, not at
        # construction time.
        worker.set_late_response_log_path(log_path)
    return worker


def _cleanup():
    for p in (SCHEDULE_PATH, LOG_PATH):
        if os.path.exists(p):
            os.remove(p)


def test_late_response_persisted_to_file_and_never_returned():
    _cleanup()
    worker = _make_worker(LOG_PATH)
    try:
        ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
        q0 = np.asarray(ref.state[0, :6], dtype=float)
        L0 = float(ref.state[0, 6])
        bp0 = np.asarray(ref.desired_position_m[0], dtype=float)
        prev_input = np.zeros(7)

        def _call(deadline_s, control_index):
            return worker.try_solve(
                kind="nullspace_r700", z_meas=np.concatenate([q0, [L0]]),
                q_cmd=q0, q_cmd_prev=q0, insertion_cmd=L0,
                measured_beam_position=bp0, control_index=control_index, previous_input=prev_input,
                deadline_s=deadline_s, measurement_frame_index=control_index, measurement_age_s=0.01,
            )

        seq_before = worker._seq
        resp1 = _call(1e-6, control_index=7)  # impossible deadline -- guaranteed miss
        assert resp1 is None, "an effectively-zero deadline must still be reported as a miss (hold behaviour unchanged)"
        assert worker._pending_seq == seq_before + 1
        assert len(worker.late_response_log) == 0
        assert not os.path.exists(LOG_PATH) or open(LOG_PATH).read() == "", "nothing should be written before anything is drained"

        time.sleep(1.0)  # let the real worker actually finish

        resp2 = _call(2.0, control_index=8)  # fresh request; drains the late one first
        assert resp2 is not None and resp2["status"] in ("ok", "infeasible")

        # --- READ THE FILE BACK (point 3) ---
        lines = open(LOG_PATH).read().strip().splitlines()
        assert len(lines) == 1, f"expected exactly one persisted record, got {len(lines)}"
        record = json.loads(lines[0])

        assert record["request_id"] == seq_before + 1, "must identify call 1's request, not call 2's"
        assert record["reference_index"] == 7
        assert record["resolved"] is True
        assert record["drain_delay_s"] > 0, "drained strictly after the deadline"
        assert record["parent_send_to_worker_timestamp_s"] > 0, (
            "parent request-send to the worker's own response_timestamp -- "
            "NOT a full round trip, distinct from drain_delay_s"
        )
        # drain_delay_s (deadline-expiry -> noticed) and
        # parent_send_to_worker_timestamp_s (send -> worker's own
        # response_timestamp, itself taken before the rest of response
        # assembly/pickling/pipe-write) measure DIFFERENT, non-overlapping
        # intervals and must not be asserted equal to each other.
        assert record["worker_reported_t_solve_ms"] is not None
        assert record["gain_actually_used"] is not None  # this worker has no jacobian-adaptation configured -> 0.0, but REPORTED, not defaulted

        # In-memory log matches what was persisted.
        assert len(worker.late_response_log) == 1
        assert worker.late_response_log[0]["request_id"] == record["request_id"]

    finally:
        worker.close()
        _cleanup()


def test_late_response_log_path_settable_after_construction():
    """2026-11-19 (point 2a): the real hardware runner spawns its worker
    BEFORE its own output directory exists, so it cannot pass
    late_response_log_path at construction time -- it calls
    set_late_response_log_path() once the directory is known instead.
    Confirms that path also persists a late response correctly."""
    _cleanup()
    worker = _make_worker(LOG_PATH, via_setter=True)
    try:
        ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
        q0 = np.asarray(ref.state[0, :6], dtype=float)
        L0 = float(ref.state[0, 6])
        bp0 = np.asarray(ref.desired_position_m[0], dtype=float)
        prev_input = np.zeros(7)

        def _call(deadline_s, control_index):
            return worker.try_solve(
                kind="nullspace_r700", z_meas=np.concatenate([q0, [L0]]),
                q_cmd=q0, q_cmd_prev=q0, insertion_cmd=L0,
                measured_beam_position=bp0, control_index=control_index, previous_input=prev_input,
                deadline_s=deadline_s, measurement_frame_index=control_index, measurement_age_s=0.01,
            )

        assert _call(1e-6, control_index=3) is None
        time.sleep(1.0)
        assert _call(2.0, control_index=4) is not None

        lines = open(LOG_PATH).read().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["reference_index"] == 3
        assert record["resolved"] is True
    finally:
        worker.close()
        _cleanup()


def test_queue_overflow_is_counted_not_blocking_and_not_crashing():
    """2026-11-19 (corrected timing-instrumentation fix): the control
    thread's enqueue is a bounded, non-blocking put. Force the queue full
    BEFORE enqueuing, so this is deterministic (no race against the
    writer thread actually draining it), and confirm overflow is counted
    rather than blocking or raising."""
    _cleanup()
    worker = _make_worker(LOG_PATH)
    try:
        assert worker.late_response_queue_overflow_count == 0
        # Swap in a maxsize=1 queue that's already full -- guarantees the
        # next enqueue overflows deterministically -- then swap the
        # REAL queue back before close(), so the writer thread (already
        # blocked on .get() against the real queue object) still
        # receives its shutdown sentinel on the object it's listening to.
        real_queue = worker._late_response_queue
        fake_queue = type(real_queue)(maxsize=1)
        fake_queue.put_nowait({"placeholder": True})
        worker._late_response_queue = fake_queue

        t0 = time.monotonic()
        worker._enqueue_late_response({"request_id": 999, "dummy": True})
        elapsed = time.monotonic() - t0

        worker._late_response_queue = real_queue  # restore before close()

        assert elapsed < 0.05, "an overflowing enqueue must not block"
        assert worker.late_response_queue_overflow_count == 1
    finally:
        worker.close()
        _cleanup()


def test_writer_errors_are_counted_and_never_crash_control_path():
    """A writer that can never open its target file (bad directory) must
    not crash the process or prevent try_solve from returning normally --
    only the error counter should reflect it."""
    _cleanup()
    bad_path = "/nonexistent_dir_for_test_xyz/late.jsonl"
    worker = _make_worker(bad_path)
    try:
        time.sleep(0.2)  # let the writer thread attempt (and fail) its open()
        assert worker.late_response_writer_error_count >= 1

        ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
        q0 = np.asarray(ref.state[0, :6], dtype=float)
        L0 = float(ref.state[0, 6])
        bp0 = np.asarray(ref.desired_position_m[0], dtype=float)
        prev_input = np.zeros(7)

        def _call(deadline_s, control_index):
            return worker.try_solve(
                kind="nullspace_r700", z_meas=np.concatenate([q0, [L0]]),
                q_cmd=q0, q_cmd_prev=q0, insertion_cmd=L0,
                measured_beam_position=bp0, control_index=control_index, previous_input=prev_input,
                deadline_s=deadline_s, measurement_frame_index=control_index, measurement_age_s=0.01,
            )

        assert _call(1e-6, control_index=10) is None  # hold behaviour unchanged
        time.sleep(1.0)
        resp = _call(2.0, control_index=11)  # drains the late one; writer still broken
        assert resp is not None and resp["status"] in ("ok", "infeasible")
        # Draining with a broken writer increments the error counter again
        # (one more attempted write against a still-unopened file).
        assert worker.late_response_writer_error_count >= 2
    finally:
        worker.close()


def test_pending_request_at_shutdown_recorded_as_unresolved():
    _cleanup()
    worker = _make_worker(LOG_PATH)
    try:
        ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
        q0 = np.asarray(ref.state[0, :6], dtype=float)
        L0 = float(ref.state[0, 6])
        bp0 = np.asarray(ref.desired_position_m[0], dtype=float)
        seq_before = worker._seq

        resp = worker.try_solve(
            kind="nullspace_r700", z_meas=np.concatenate([q0, [L0]]),
            q_cmd=q0, q_cmd_prev=q0, insertion_cmd=L0,
            measured_beam_position=bp0, control_index=42, previous_input=np.zeros(7),
            deadline_s=1e-6, measurement_frame_index=42, measurement_age_s=0.01,
        )
        assert resp is None  # guaranteed miss, request left outstanding
        assert worker._pending_seq == seq_before + 1

        # Close IMMEDIATELY -- do not give the worker time to finish, and
        # do not call try_solve again to drain it. The request's eventual
        # response (if the worker even gets to send one before being
        # terminated) is never read.
        worker.close()

        lines = open(LOG_PATH).read().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["request_id"] == seq_before + 1
        assert record["reference_index"] == 42
        assert record["resolved"] is False
        assert "unresolved_reason" in record and record["unresolved_reason"]
        assert record["worker_response_timestamp"] is None
        assert record["parent_send_to_worker_timestamp_s"] is None

        # Shutdown-drain completeness is observable, not assumed: the
        # writer thread had time to flush this one record before close()
        # returned (join(timeout=5.0) succeeded), so this must report True.
        assert worker.late_response_shutdown_drain_complete is True
        assert worker.late_response_pending_at_shutdown == 0

    finally:
        try:
            worker.close()  # idempotent if already closed
        except Exception:
            pass
        _cleanup()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
