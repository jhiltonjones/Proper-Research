"""Mocked (no hardware) validation that BackgroundArmStreamer's new
per-cycle event log correctly reports the REALIZED call rate/interval,
not the configured 20ms/50Hz target, when servo_j() itself is slow.

This is the core thing the old, unlogged design could never have caught:
before this patch, last_servo_ms/calls were computed and discarded, so a
systematically slow servo_j() would have been invisible -- the run would
just "look like" it was streaming at 50Hz because that's the configured
interval, with nothing to contradict it.

2026-11-15: added test_orderly_shutdown_drains_queue() -- confirms the
writer thread fully drains its queue before stop() returns, so no event
is silently dropped at the tail end of a real run.

Run directly (no pytest needed, no robot, no camera):
    python3 test_close_loop_path_follow_streamer.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/jack/Proper-Research")

from proper_research.hardware.online.close_loop_path_follow import (
    BackgroundArmStreamer, _STREAM_INTERVAL_S,
)


class SlowMockRobot:
    """servo_j() deliberately takes longer than the streamer's configured
    20ms interval, so the streamer's OWN scheduler (next_cycle += interval)
    cannot keep up -- each cycle effectively takes >= the call duration,
    not the configured interval."""

    def __init__(self, call_duration_s: float):
        self.call_duration_s = call_duration_s
        self.calls = []

    def servo_j(self, target, time_s=0.0, lookahead_time=0.0, gain=0):
        time.sleep(self.call_duration_s)
        self.calls.append(time.monotonic())


class InstantMockRobot:
    def servo_j(self, target, time_s=0.0, lookahead_time=0.0, gain=0):
        pass


def run_case(label: str, call_duration_s: float, run_seconds: float = 1.0):
    print(f"\n=== {label}: servo_j() takes {call_duration_s*1e3:.0f}ms "
          f"(configured interval = {_STREAM_INTERVAL_S*1e3:.0f}ms) ===")
    robot = SlowMockRobot(call_duration_s)
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "arm_streamer_events.jsonl"
        streamer = BackgroundArmStreamer(
            robot, time_s=0.04, lookahead_time=0.2, gain=200,
        )
        streamer.enable_event_log(log_path)
        streamer.start()
        t_end = time.monotonic() + run_seconds
        target = [0.0] * 6
        while time.monotonic() < t_end:
            streamer.update_target(target, state_age_s=0.01)
            time.sleep(0.1)  # mimic the 100ms MPC tick publishing new targets
        streamer.stop()

        events = [json.loads(l) for l in open(log_path)]
        servo_events = [e for e in events if e["event"] == "servo_j_call"]
        pickup_events = [e for e in events if e["event"] == "pickup"]

        n_calls = len(servo_events)
        realized_rate_hz = n_calls / run_seconds
        configured_rate_hz = 1.0 / _STREAM_INTERVAL_S

        servo_durations_ms = [e["servo_duration_ms"] for e in servo_events]
        intervals = [e["realized_interval_s"] for e in pickup_events if e["realized_interval_s"] is not None]

        print(f"  total events logged: {len(events)}")
        print(f"  servo_j() calls in {run_seconds}s: {n_calls}")
        print(f"  CONFIGURED rate: {configured_rate_hz:.1f} Hz (interval {_STREAM_INTERVAL_S*1e3:.0f}ms)")
        print(f"  REALIZED rate (from logged servo_j_call events): {realized_rate_hz:.1f} Hz")
        print(f"  mean servo_j() duration (logged): {sum(servo_durations_ms)/len(servo_durations_ms):.1f}ms")
        print(f"  mean realized cycle interval (logged): {1e3*sum(intervals)/len(intervals):.1f}ms "
              f"(vs configured {_STREAM_INTERVAL_S*1e3:.0f}ms)")
        return realized_rate_hz, configured_rate_hz


def test_orderly_shutdown_drains_queue():
    """Confirm the writer thread drains its queue fully before stop()
    returns -- no event silently dropped at the tail end of a run."""
    print("\n=== ORDERLY SHUTDOWN: confirm no events dropped at stop() ===")
    robot = InstantMockRobot()
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "events.jsonl"
        streamer = BackgroundArmStreamer(robot, time_s=0.04, lookahead_time=0.2, gain=200,
                                         stream_interval_s=0.002)  # fast cycle to pile up events quickly
        streamer.enable_event_log(log_path)
        streamer.start()
        target = [0.0] * 6
        n_publishes = 50
        for _ in range(n_publishes):
            streamer.update_target(target, state_age_s=0.0)
            time.sleep(0.001)
        # Stop immediately after the burst -- if the writer thread did NOT
        # drain its queue before exiting, some of the just-emitted events
        # would be missing from the file.
        streamer.stop()

        events = [json.loads(l) for l in open(log_path)]
        publish_events = [e for e in events if e["event"] == "publish"]
        print(f"  publishes issued: {n_publishes}")
        print(f"  publish events found in log after stop(): {len(publish_events)}")
        print(f"  total events logged: {len(events)}")
        assert len(publish_events) == n_publishes, (
            f"FAIL: expected {n_publishes} publish events, found {len(publish_events)} -- "
            f"events were dropped at shutdown"
        )
        print("  PASS: every publish event survived to disk -- writer thread drained fully before stop() returned.")


def main():
    # Case 1: fast servo_j (5ms, well under the 20ms interval) -- sanity
    # check that the instrumentation reports ~50Hz when nothing is wrong.
    fast_rate, configured = run_case("FAST (sanity check)", call_duration_s=0.005, run_seconds=1.0)
    assert fast_rate > 40, (
        f"FAIL: fast case should realize close to {configured:.0f}Hz, got {fast_rate:.1f}Hz"
    )
    print("  PASS: fast case realizes close to the configured rate, as expected.")

    # Case 2: slow servo_j (40ms, double the 20ms interval) -- the real test.
    slow_rate, configured = run_case("SLOW (the actual test)", call_duration_s=0.04, run_seconds=1.0)
    assert slow_rate < configured * 0.7, (
        f"FAIL: instrumentation did not catch the degraded rate -- "
        f"logged {slow_rate:.1f}Hz, expected well below configured {configured:.0f}Hz"
    )
    print(f"  PASS: instrumentation correctly shows REALIZED rate ({slow_rate:.1f}Hz) "
          f"well below the CONFIGURED rate ({configured:.0f}Hz) -- "
          f"the old design would have shown nothing, and this did.")

    test_orderly_shutdown_drains_queue()

    print("\n=== ALL ASSERTIONS PASSED ===")


if __name__ == "__main__":
    main()
