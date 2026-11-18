"""Tests for servo_diagnostic_logging_prep.py (v2) -- mocks only, no
hardware, no robot motion. Covers the 7 required failure modes plus the
cadence/control-state-unchanged check and integrated overhead
measurement.

Run with: python3 test_servo_diagnostic_logging_prep.py
"""
import tempfile
import time

from servo_diagnostic_logging_prep import (
    ConsistencyState,
    DiagnosticLogSink,
    StaleStreamDetector,
    correspond_to_command_seq,
    effective_servo_args,
    robot_side_snapshot,
    runtime_environment_snapshot,
)


# ---- basic sanity (carried over from v1) -----------------------------

def test_runtime_environment_snapshot_resolves_real_installed_rtde():
    out = runtime_environment_snapshot()
    assert out["python_executable"]
    assert out["rtde_control_file"] is not None
    assert out["ur_rtde_version"] is not None


def test_effective_servo_args_passthrough():
    out = effective_servo_args(time_s=0.04, lookahead_time=0.2, gain=200)
    assert out == {"servo_time_s": 0.04, "servo_lookahead_time_s": 0.2, "servo_gain": 200}


# ---- the 7 required failure-mode tests --------------------------------

class _FullMock:
    def __init__(self, ts=1.0, q=None):
        self.ts = ts
        self.q = q if q is not None else [0.1] * 6

    def getTimestamp(self):
        return self.ts

    def getActualQ(self):
        return self.q

    def getTargetQ(self):
        return self.q

    def getSpeedScaling(self):
        return 1.0

    def getSpeedScalingCombined(self):
        return 1.0


def test_1_missing_method_lookup_is_logged_not_silently_none():
    class _NoSpeedScaling:
        """Simulates an older library/firmware build that never defines
        getSpeedScaling at all -- the attribute lookup itself must fail
        gracefully, not just the call."""
        def getTimestamp(self):
            return 1.0
        def getActualQ(self):
            return [0.1] * 6
        def getTargetQ(self):
            return [0.1] * 6
        def getSpeedScalingCombined(self):
            return 1.0
        # getSpeedScaling intentionally absent
    m = _NoSpeedScaling()
    snap = robot_side_snapshot(m)
    assert snap.speed_scaling is None
    failures = [f for f in snap.read_failures if f["field"] == "getSpeedScaling"]
    assert len(failures) == 1
    assert failures[0]["reason"] == "method_missing"


def test_2_missing_timestamp_read_is_unknown_never_unchanged():
    class _NoTimestamp:
        def getTimestamp(self):
            return None  # read "succeeds" but returns nothing
        def getActualQ(self):
            return [0.1] * 6
        def getTargetQ(self):
            return [0.1] * 6
        def getSpeedScaling(self):
            return 1.0
        def getSpeedScalingCombined(self):
            return 1.0
    snap = robot_side_snapshot(_NoTimestamp())
    assert snap.within_snapshot_consistency == ConsistencyState.UNKNOWN
    assert snap.within_snapshot_consistency != ConsistencyState.UNCHANGED


def test_3_mid_read_update_is_detected_as_changed():
    class _ChangesMidRead:
        def __init__(self):
            self._n = 0
        def getTimestamp(self):
            self._n += 1
            return 100.0 if self._n == 1 else 100.02
        def getActualQ(self):
            return [0.2] * 6
        def getTargetQ(self):
            return [0.2] * 6
        def getSpeedScaling(self):
            return 1.0
        def getSpeedScalingCombined(self):
            return 1.0
    snap = robot_side_snapshot(_ChangesMidRead())
    assert snap.within_snapshot_consistency == ConsistencyState.CHANGED
    assert snap.robot_timestamp_before_raw != snap.robot_timestamp_after_raw


def test_4_frozen_timestamp_across_snapshots_flags_suspected_stale():
    detector = StaleStreamDetector()
    frozen = _FullMock(ts=5.0, q=[0.5] * 6)
    verdicts = []
    for _ in range(4):
        snap = robot_side_snapshot(frozen)
        verdicts.append(detector.update(snap)["verdict"])
    # first update has no history -> "unknown"; subsequent ones with
    # frozen ts + repeated joints -> "suspected_stale_stream"
    assert verdicts[0] == "unknown"
    assert all(v == "suspected_stale_stream" for v in verdicts[1:])


def test_5_stationary_but_fresh_is_distinguished_from_stale():
    """The legitimate case: joints genuinely unchanged (robot holding
    still) but the robot timestamp keeps advancing -- must NOT be
    flagged as a stale stream."""
    detector = StaleStreamDetector()
    verdicts = []
    for i in range(4):
        mock = _FullMock(ts=5.0 + i * 0.008, q=[0.5] * 6)  # ts advances, q fixed
        snap = robot_side_snapshot(mock)
        verdicts.append(detector.update(snap)["verdict"])
    assert verdicts[0] == "unknown"
    assert all(v == "stationary_fresh" for v in verdicts[1:])


def test_6_queue_overflow_is_counted_not_silently_lost():
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        sink = DiagnosticLogSink(tf.name, maxsize=2)
        # Don't start the writer thread -- force the queue to actually fill.
        ok1 = sink.emit({"i": 1})
        ok2 = sink.emit({"i": 2})
        ok3 = sink.emit({"i": 3})  # queue is full (maxsize=2) -> must be dropped, counted
        assert ok1 and ok2
        assert ok3 is False
        assert sink.dropped_count == 1


def test_7_incomplete_shutdown_is_detectable():
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        sink = DiagnosticLogSink(tf.name, maxsize=100)
        sink.start()
        for i in range(5):
            sink.emit({"i": i})
        # Normal case: give it enough time -> drains cleanly.
        drained = sink.stop(timeout=2.0)
        assert drained is True
        assert sink.drained_cleanly is True

    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf2:
        sink2 = DiagnosticLogSink(tf2.name, maxsize=100)
        sink2.start()
        for i in range(5):
            sink2.emit({"i": i})
        # Adversarial case: a near-zero timeout can legitimately still
        # catch the thread mid-drain -- the design must report this
        # honestly (True or False) rather than always claiming success.
        result = sink2.stop(timeout=0.0)
        assert result in (True, False)
        assert sink2.drained_cleanly == result
        # Clean up the real thread regardless of what the adversarial
        # near-zero timeout caught.
        sink2.stop(timeout=2.0)


# ---- correspondence (carried over) ------------------------------------

def test_correspond_to_command_seq_picks_the_most_recent_publish_at_or_before():
    publishes = [
        {"seq": 1, "publish_time_s": 10.0},
        {"seq": 2, "publish_time_s": 10.1},
        {"seq": 3, "publish_time_s": 10.2},
    ]
    assert correspond_to_command_seq(10.15, publishes) == 2
    assert correspond_to_command_seq(9.9, publishes) is None
    assert correspond_to_command_seq(10.2, publishes) == 3


# ---- cadence/control-state-unchanged + integrated overhead ------------

def test_cadence_and_control_state_values_unchanged_by_this_module():
    """2026-11-17 UPDATED: real, authorized production integration now
    exists (RobotJointStream diagnostic logging, wired into
    state_stream.py and close_loop_path_follow.py -- see
    ROBOT_JOINT_STREAM_DIAGNOSTIC_INTEGRATION_2026-11-17.md and its
    follow-up). This module is therefore now EXPECTED to be imported by
    exactly those two files. This test asserts the import set is exactly
    the expected one -- not zero, not anything unexpected (e.g. no
    vessel_stage_a script imports this, and no OTHER rectangle_stage_a
    file besides the one real runner does) -- rather than re-asserting
    the now-obsolete "zero imports" claim. Control-path-unchanged itself
    is separately, directly verified by
    test_robot_joint_stream_diagnostic_integration.py's
    test_control_state_and_cadence_identical_logging_on_vs_off (which
    exercises the real integrated code path), not by this import check."""
    import subprocess

    result = subprocess.run(
        [
            "grep", "-rl", "servo_diagnostic_logging_prep",
            "--include=*.py",
            "/home/jack/Proper-Research/proper_research/hardware/online/rectangle_stage_a",
            "/home/jack/Proper-Research/proper_research/hardware/online/vessel_stage_a",
            "/home/jack/Proper-Research/proper_research/hardware/online/close_loop_path_follow.py",
            "/home/jack/Proper-Research/proper_research/hardware/online/state_stream.py",
        ],
        capture_output=True, text=True,
    )
    importers = {
        line for line in result.stdout.splitlines()
        if "test_servo_diagnostic_logging_prep" not in line
    }
    expected = {
        "/home/jack/Proper-Research/proper_research/hardware/online/close_loop_path_follow.py",
        "/home/jack/Proper-Research/proper_research/hardware/online/state_stream.py",
    }
    assert importers == expected, (
        f"import set changed from the expected, reviewed integration points: "
        f"got {importers}, expected {expected}"
    )


def test_integrated_mock_overhead_end_to_end():
    """Overhead of the full proposed pipeline (snapshot + stale-stream
    update + sink.emit), as it would actually run per logged sample --
    not the isolated snapshot-only microbenchmark from the prior pass."""
    import tempfile

    mock = _FullMock(ts=1.0)
    detector = StaleStreamDetector()
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as tf:
        sink = DiagnosticLogSink(tf.name, maxsize=5000)
        sink.start()
        n = 5000
        t0 = time.perf_counter()
        for i in range(n):
            mock.ts = 1.0 + i * 0.001  # advance so it isn't flagged stale
            snap = robot_side_snapshot(mock)
            verdict = detector.update(snap)
            sink.emit({"seq": i, "verdict": verdict["verdict"],
                       "consistency": snap.within_snapshot_consistency.value})
        t1 = time.perf_counter()
        drained = sink.stop(timeout=3.0)
    per_call_us = (t1 - t0) / n * 1e6
    print(f"integrated end-to-end overhead: {per_call_us:.2f} us/call over {n} calls "
          f"(mock getters + stale-detector + queue emit; writer-thread drained_cleanly={drained})")
    assert drained is True
    # Sanity bound only -- not a hard performance claim against real RTDE I/O.
    assert per_call_us < 1000.0


if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
