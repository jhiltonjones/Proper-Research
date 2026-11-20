"""2026-11-19 (stop-path correction, point 4): deterministic shutdown-
ordering tests for BackgroundArmStreamer -- uses threading.Event gates
(not sleep-based timing) to force specific interleavings, and repeats
with the gate released at several different controlled points, rather
than relying on one favourable schedule. No hardware.

Structural property under test: `stop()` sets `self._running = False`
THEN blocks on `self._thread.join()`. Because the streaming loop checks
`while self._running:` only at the top of each cycle, and Python's bool
attribute read/write is atomic under the GIL, no new cycle (and hence no
new servo_j() dispatch) can begin once `_running` has been observed
False -- an ALREADY-STARTED call (one that passed its `while` check
before the flag flipped) is expected to run to completion; `stop()`'s
join() does not interrupt it.
"""
from __future__ import annotations

import sys
import threading
import time

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.close_loop_path_follow import BackgroundArmStreamer


class GatedMockRobot:
    """servo_j() blocks until `release_event` is set, recording exactly
    when it started and finished -- lets the test deterministically force
    a call to be "in flight" at the moment stop() is invoked."""

    def __init__(self):
        self.calls = []  # (start_t, end_t)
        self.started_event = threading.Event()
        self.release_event = threading.Event()

    def servo_j(self, target, time_s, lookahead_time, gain):
        t0 = time.monotonic()
        self.started_event.set()
        self.release_event.wait(timeout=5.0)
        t1 = time.monotonic()
        self.calls.append((t0, t1))
        return True


def _run_one_trial(hold_release_for_s: float) -> dict:
    """One controlled trial: publish a target, wait until servo_j has
    genuinely started (deterministic, via started_event -- not a sleep
    guess), THEN request stop() from a second thread, THEN attempt a
    late update_target() call, THEN release the in-flight call after
    `hold_release_for_s` (varying this across trials changes exactly
    where, relative to stop()'s join(), the in-flight call actually
    finishes -- a different controlled interleaving each time)."""
    robot = GatedMockRobot()
    streamer = BackgroundArmStreamer(robot, time_s=0.1, lookahead_time=0.1, gain=300, stream_interval_s=0.02)
    streamer.start()
    streamer.update_target([1.0] * 6)
    assert robot.started_event.wait(timeout=5.0), "servo_j never started -- test setup broken"

    stop_done = threading.Event()

    def _do_stop():
        streamer.stop(timeout=5.0)
        stop_done.set()

    stop_thread = threading.Thread(target=_do_stop)
    stop_thread.start()

    # Confirm the in-flight call has NOT finished yet (it's gated) --
    # this is checked, not assumed.
    time.sleep(0.01)
    in_flight_before_release = len(robot.calls) == 0

    # Attempt a late target publication WHILE stop()'s join is pending.
    late_target_published_at = time.monotonic()
    streamer.update_target([9.0] * 6)

    time.sleep(hold_release_for_s)
    robot.release_event.set()  # let the in-flight call finish
    stop_thread.join(timeout=5.0)

    return {
        "in_flight_before_release": in_flight_before_release,
        "n_servo_j_calls": len(robot.calls),
        "stop_actually_joined": not (streamer._thread and streamer._thread.is_alive()),
        "late_target_published_at": late_target_published_at,
        "calls": robot.calls,
    }


def test_in_flight_call_completes_and_exactly_one_call_total():
    for hold_s in (0.0, 0.01, 0.05):
        result = _run_one_trial(hold_release_for_s=hold_s)
        assert result["in_flight_before_release"], (
            f"(hold={hold_s}s) the in-flight call finished before being released -- "
            "test synchronization is not actually controlling the interleaving"
        )
        assert result["n_servo_j_calls"] == 1, (
            f"(hold={hold_s}s) expected exactly the one already-in-flight call to "
            f"complete, got {result['n_servo_j_calls']} total servo_j dispatches -- "
            "a late update_target() call during shutdown caused a NEW dispatch"
        )
        assert result["stop_actually_joined"], f"(hold={hold_s}s) stop() did not actually join the thread"


def test_many_repeated_trials_no_extra_dispatch():
    """Repeats the controlled interleaving many times with small variation
    in release timing -- not one lucky run."""
    total_calls = 0
    for i in range(20):
        result = _run_one_trial(hold_release_for_s=0.001 * i)
        total_calls += result["n_servo_j_calls"]
        assert result["n_servo_j_calls"] == 1, f"trial {i}: {result}"
    assert total_calls == 20


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
