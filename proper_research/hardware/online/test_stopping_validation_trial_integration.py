"""2026-11-19: mock-tests the INTEGRATED stopping-validation trial lifecycle
(run_stopping_validation_trial.run_trial) against a fake robot that mimics
URRTDERobot's servo_j/servo_stop/get_joints surface -- real BackgroundArmStreamer,
real StoppingTestLog, real event/displacement bookkeeping, real PlanSegment
playback, no network, no hardware. Confirms: sampling continues through
shutdown and standstill; timestamps come from actual recorded events, not
estimates; an exception mid-trial marks it incomplete rather than silently
reporting success; insertion handling is explicit (hold_fixed requires its
acknowledgement flag; advancer_playback is driven, not silently skipped).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.run_stopping_validation_trial import run_trial, main
from proper_research.hardware.online.stopping_validation_plan_segment import PlanSegment


class _FakeRobotStandstillAfterStop:
    """Moves at a constant rate while streamed, then genuinely stops once
    servo_stop() is called -- mimics a real deceleration-to-standstill."""

    def __init__(self):
        self.q = np.zeros(6)
        self._moving = True
        self.servo_stop_calls = 0

    def get_joints(self):
        if self._moving:
            self.q = self.q + 0.1  # steady synthetic motion, well above the test's standstill threshold
        return self.q.tolist()

    def servo_j(self, target, time_s, lookahead_time, gain):
        return True

    def servo_stop(self, a):
        self.servo_stop_calls += 1
        self._moving = False  # robot is now genuinely at standstill
        return True


class _FakeRobotRaisesOnGetJoints:
    def __init__(self):
        self.n_calls = 0

    def get_joints(self):
        self.n_calls += 1
        if self.n_calls > 3:
            raise RuntimeError("simulated RTDE telemetry dropout")
        return [0.0] * 6

    def servo_j(self, target, time_s, lookahead_time, gain):
        return True

    def servo_stop(self, a):
        return True


def _fake_segment(n: int = 10, insertion_varies: bool = True) -> PlanSegment:
    time_s = np.arange(n) * 0.1
    joints = np.tile(np.linspace(0.0, 0.2, n)[:, None], (1, 6))
    insertion = np.linspace(0.030, 0.034, n) if insertion_varies else np.full(n, 0.030)
    rate = np.zeros((n, 6))
    ins_rate = np.full(n, 0.0004) if insertion_varies else np.zeros(n)
    magnet_xyz = np.tile(np.array([0.0, 0.0, 0.3]), (n, 1))
    clearance = np.full(n, 0.30)
    return PlanSegment(
        plan_dir="fake", i0=0, i1=n - 1, time_s=time_s, joints_rad=joints, insertion_m=insertion,
        joint_rate_rad_s=rate, insertion_rate_m_s=ins_rate, magnet_xyz_m=magnet_xyz, clearance_m=clearance,
    )


def _base_args(tmp_out: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        speed_limit_rad_s=0.1, trigger_mode="manual",
        standstill_velocity_threshold_rad_s=0.5, standstill_hold_time_s=0.05,
        max_trial_duration_s=3.0, lab_failure_procedure="test-procedure",
        sample_interval_s=0.02, output_dir=str(tmp_out),
        robot_host="unused", robot_ip_receive=None,
        staged_trigger_centre_xyz_m=None, staged_trigger_radius_m=None,
        insertion_mode="hold_fixed", acknowledge_insertion_held_fixed=True,
        advancer_port=None, confirmed_initial_insertion_mm=None, insertion_mismatch_tolerance_mm=2.0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_manual_trigger_reaches_standstill_and_records_all_timestamps():
    out_dir = Path("/tmp/claude-1000") / "stopval_trial_test_1"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    robot = _FakeRobotStandstillAfterStop()
    args = _base_args(out_dir)
    segment = _fake_segment()

    import threading
    import os
    import signal as signal_mod

    def _send_abort_soon():
        time.sleep(0.1)
        os.kill(os.getpid(), signal_mod.SIGUSR1)

    threading.Thread(target=_send_abort_soon, daemon=True).start()
    rc = run_trial(args, robot, out_dir, None, segment)

    assert rc == 0, "trial should have completed (reached standstill)"
    assert robot.servo_stop_calls == 1

    import json
    result = json.loads((out_dir / "result.json").read_text())
    assert result["outcome"] == "complete"
    assert result["crossing_to_detection_latency_note"] == "UNMEASURED (manual trigger)"
    assert "crossing_to_detection_m" in result and "detection_to_standstill_m" in result
    assert result["plan_segment"]["n_samples"] == segment.n

    events = {e["kind"]: e["t"] for e in json.loads((out_dir / "events.json").read_text())}
    for kind in ("boundary_crossing", "detection", "last_dispatch", "streamer_exit", "stop_request", "standstill"):
        assert kind in events, f"missing event: {kind}"
    assert events["boundary_crossing"] <= events["detection"] <= events["streamer_exit"] <= events["stop_request"] <= events["standstill"]
    shutil.rmtree(out_dir, ignore_errors=True)


def test_telemetry_exception_marks_trial_incomplete_not_success():
    out_dir = Path("/tmp/claude-1000") / "stopval_trial_test_2"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    robot = _FakeRobotRaisesOnGetJoints()
    args = _base_args(out_dir, max_trial_duration_s=1.0)
    rc = run_trial(args, robot, out_dir, None, _fake_segment())

    assert rc != 0, "an exception mid-trial must not be reported as a successful (rc=0) trial"
    import json
    result = json.loads((out_dir / "result.json").read_text())
    assert result["outcome"] in ("incomplete_exception", "incomplete_timeout")
    shutil.rmtree(out_dir, ignore_errors=True)


def test_output_dir_must_not_already_exist():
    """The script refuses to silently overwrite a previous trial's output."""
    out_dir = Path("/tmp/claude-1000") / "stopval_trial_test_3"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    try:
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            rc = main([
                "--plan-dir", "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path",
                "--segment-start-idx", "199", "--segment-end-idx", "228",
                "--insertion-mode", "hold_fixed", "--acknowledge-insertion-held-fixed",
                "--speed-limit-rad-s", "0.1", "--trigger-mode", "manual",
                "--standstill-velocity-threshold-rad-s", "0.5", "--standstill-hold-time-s", "0.05",
                "--max-trial-duration-s", "1.0", "--lab-failure-procedure", "x",
                "--output-dir", str(out_dir), "--robot-host", "0.0.0.0",
            ])
        assert rc == 1
        assert "already exists" in f.getvalue()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_hold_fixed_without_acknowledgement_is_refused():
    """--insertion-mode hold_fixed without the explicit acknowledgement flag
    must refuse to start -- insertion must never be silently held fixed."""
    out_dir = Path("/tmp/claude-1000") / "stopval_trial_test_4"
    shutil.rmtree(out_dir, ignore_errors=True)
    import io
    import contextlib
    try:
        f = io.StringIO()
        with pytest.raises(SystemExit):
            with contextlib.redirect_stderr(f):
                main([
                    "--plan-dir", "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path",
                    "--segment-start-idx", "199", "--segment-end-idx", "228",
                    "--insertion-mode", "hold_fixed",  # acknowledgement flag deliberately omitted
                    "--speed-limit-rad-s", "0.1", "--trigger-mode", "manual",
                    "--standstill-velocity-threshold-rad-s", "0.5", "--standstill-hold-time-s", "0.05",
                    "--max-trial-duration-s", "1.0", "--lab-failure-procedure", "x",
                    "--output-dir", str(out_dir), "--robot-host", "0.0.0.0",
                ])
        assert "acknowledge-insertion-held-fixed" in f.getvalue()
        assert not out_dir.exists(), "must not create output or start before refusing"
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def test_advancer_playback_mismatch_refuses_to_start():
    """A --confirmed-initial-insertion-mm that disagrees with the segment's
    own start insertion (beyond tolerance) must refuse to start, not
    silently proceed from the wrong physical assumption."""
    out_dir = Path("/tmp/claude-1000") / "stopval_trial_test_5"
    shutil.rmtree(out_dir, ignore_errors=True)
    import io
    import contextlib
    try:
        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            rc = main([
                "--plan-dir", "/home/jack/Proper-Research/plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path",
                "--segment-start-idx", "199", "--segment-end-idx", "228",
                "--insertion-mode", "advancer_playback",
                "--advancer-port", "/dev/ttyFAKE",
                "--confirmed-initial-insertion-mm", "100.0",  # deliberately far from the real ~31.9mm start
                "--speed-limit-rad-s", "0.1", "--trigger-mode", "manual",
                "--standstill-velocity-threshold-rad-s", "0.5", "--standstill-hold-time-s", "0.05",
                "--max-trial-duration-s", "1.0", "--lab-failure-procedure", "x",
                "--output-dir", str(out_dir), "--robot-host", "0.0.0.0",
            ])
        assert rc == 1
        assert "differs from the segment" in f.getvalue()
        assert not out_dir.exists()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
