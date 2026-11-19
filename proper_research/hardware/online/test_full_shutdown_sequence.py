"""2026-11-19: deterministic test of the FULL shutdown sequence (streamer
shutdown -> robot stop command issuance), covering the window between
those two steps and after. No hardware. Confirms no servo_j dispatch can
occur once streamer.stop() has returned, through and past the point
robot.servo_stop() is issued -- including an explicit late update_target()
attempt in that post-shutdown window.
"""
import sys
import threading
import time

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.close_loop_path_follow import BackgroundArmStreamer


class RecordingMockRobot:
    def __init__(self):
        self.servo_j_calls = []
        self.servo_stop_calls = []

    def servo_j(self, target, time_s, lookahead_time, gain):
        self.servo_j_calls.append(time.monotonic())
        return True

    def servo_stop(self, a):
        self.servo_stop_calls.append(time.monotonic())
        return True


def test_no_dispatch_in_pre_dispatch_window_or_after_stop_command():
    robot = RecordingMockRobot()
    streamer = BackgroundArmStreamer(robot, time_s=0.1, lookahead_time=0.1, gain=300, stream_interval_s=0.02)
    streamer.start()
    streamer.update_target([1.0] * 6)
    time.sleep(0.05)  # let a few real dispatches happen under normal operation
    n_before_shutdown = len(robot.servo_j_calls)
    assert n_before_shutdown > 0, "setup problem: no dispatches happened before shutdown at all"

    # Step 1: streamer shutdown (mirrors close_loop_path_follow.py's finally block).
    streamer.stop(timeout=2.0)
    t_shutdown_complete = time.monotonic()

    # The "pre-dispatch window": between shutdown completing and the robot
    # stop command being issued. Attempt a late update_target() HERE.
    streamer.update_target([9.0] * 6)
    n_during_window = len(robot.servo_j_calls)

    # Step 2: robot stop command issuance.
    robot.servo_stop(2.0)
    t_stop_issued = time.monotonic()

    # Attempt another late update_target() AFTER the stop command too.
    streamer.update_target([9.0] * 6)
    time.sleep(0.05)  # give any (incorrect) residual activity a chance to show up
    n_after_stop_command = len(robot.servo_j_calls)

    assert t_stop_issued >= t_shutdown_complete
    assert n_during_window == n_before_shutdown, (
        f"a servo_j dispatch occurred in the pre-dispatch window (between shutdown "
        f"completing and the stop command being issued): before={n_before_shutdown} "
        f"during_window={n_during_window}"
    )
    assert n_after_stop_command == n_before_shutdown, (
        f"a servo_j dispatch occurred AFTER the stop command was issued: "
        f"before={n_before_shutdown} after={n_after_stop_command}"
    )
    assert len(robot.servo_stop_calls) == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
