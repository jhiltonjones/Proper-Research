"""End-to-end, hardware-blocked smoke test of the ACTUAL baseline runner
(run_mpc_delay_aware_insertion_anchor.py) and the REAL close_loop_path_
follow.main() control loop -- no copied row construction, no simplified
loop skeleton. Real CameraSource/StateEstimate construction, real
MPCWorkerHandle + ProcessIsolatedDelayAwareAdapter, real
BackgroundArmStreamer. Mocked: URRTDERobot, RobotJointStream (hardware
RTDE interfaces), and the camera's own device-opening half of CameraSource
(_grab_body's cv2.VideoCapture is never reached -- _reconstruct_body,
which builds the REAL StateEstimate, is driven directly from a background
thread using synthetic frames). AdvancerSink uses its own existing
dry_run=True (a real, intended no-hardware mode, not a test double).

One coherent monotonic clock (time.monotonic()) drives both the
synthetic camera frames and the mocked joint-timestamp reads -- no fake
epoch (the completion-check run two turns ago used one and produced
negative ages; not repeated here).
"""
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/jack/Proper-Research")
os.chdir("/home/jack/Proper-Research")

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.hardware.online.camera_source import CameraSource, CameraConfig
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
OUT_DIR = "/home/jack/.claude/jobs/043c0b8b/tmp/e2e_smoke_run"


class MockRobot:
    """Replaces URRTDERobot -- no RTDE connection ever opens."""

    def __init__(self, *_a, **_kw):
        self.servo_j_calls = []  # (t_monotonic, q_target_copy)
        self.closed = False
        self._lock = threading.Lock()

    def get_robot_mode(self):
        return 7

    def get_safety_mode(self):
        return 1

    def is_protective_stopped(self):
        return False

    def get_tcp_pose(self):
        return np.zeros(6)

    def servo_j(self, q, time_s=0.1, lookahead_time=0.1, gain=300):
        with self._lock:
            self.servo_j_calls.append((time.monotonic(), np.asarray(q, dtype=float).copy()))

    def servo_stop(self, _timeout):
        pass

    def close(self):
        self.closed = True


class MockReader:
    """Replaces RobotJointStream -- no RTDE connection ever opens. Reads
    the SAME single monotonic clock + shared joint-state holder the
    camera driver thread also writes, so joint samples and camera frames
    are coherent (same clock family, same underlying q6)."""

    def __init__(self, shared_state, *_a, **_kw):
        print(f"[MockReader] constructed with args={_a!r} kwargs={_kw!r}")
        self._shared = shared_state
        self.reads = 0
        self.read_failures = 0
        self.stale_repeats = 0
        self.last_error = ""

    def start(self):
        pass

    def stop(self):
        pass

    def latest_joints_with_timestamp(self, _max_age_s=None):
        self.reads += 1
        with self._shared["lock"]:
            return self._shared["q6"].copy(), self._shared["stamp"]

    def latest_joints(self, _max_age_s=None):
        with self._shared["lock"]:
            return self._shared["q6"].copy()

    def latest_pose(self, _max_age_s=None):
        return None

    def is_protective_stopped(self):
        return False

    def safety_mode(self):
        return 1

    def reconnect_receive(self):
        pass


class MockedCameraSource(CameraSource):
    """Real CameraSource, with ONLY start()/stop() overridden so no real
    camera device is ever opened. A background thread drives the REAL
    _reconstruct_body() with synthetic frames at a steady cadence,
    exactly mirroring the real grab+reconstruct architecture."""

    def __init__(self, *args, shared_state, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared = shared_state
        self._driver_thread = None
        self._driver_running = False

    def _driver_loop(self):
        tick = 0
        while self._driver_running:
            now = time.monotonic()
            with self._shared["lock"]:
                self._shared["q6"] = self._shared["q0"] + 0.0005 * tick
                self._shared["stamp"] = now - 0.003  # 3ms old, realistic
            self._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
            self._latest_frame_stamp = now
            self._reconstruct_body(now)
            tick += 1
            time.sleep(0.02)

    def start(self):
        self._driver_running = True
        self._driver_thread = threading.Thread(target=self._driver_loop, daemon=True)
        self._driver_thread.start()
        time.sleep(0.05)  # let the first synthetic frame land before healthy is checked

    @property
    def healthy(self):
        return self._driver_running and self._driver_thread is not None and self._driver_thread.is_alive()

    def stop(self):
        self._driver_running = False
        if self._driver_thread is not None:
            self._driver_thread.join(timeout=1.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    q0 = np.asarray(reference.state[0, :6], dtype=float)
    tip0 = np.asarray(reference.desired_position_m[0], dtype=float)

    shared_state = {"lock": threading.Lock(), "q0": q0, "q6": q0.copy(), "stamp": time.monotonic()}

    def synthetic_frame_processor(_frame):
        return np.concatenate([tip0, np.array([0.0, 0.0, 1.0])])

    pf.URRTDERobot = MockRobot
    pf.RobotJointStream = lambda *a, **kw: MockReader(shared_state, *a, **kw)
    def _camera_factory(*a, **kw):
        kw.pop("frame_processor", None)  # ignore the real mapper -- always use the synthetic one
        return MockedCameraSource(*a, shared_state=shared_state, frame_processor=synthetic_frame_processor, **kw)
    pf.CameraSource = _camera_factory

    # advancer: use its own real dry_run mode, not a mock -- pre-set on the
    # singleton CONFIG before the runner's main() reads/sets other fields.
    pf.CONFIG.advancer_dry_run = True

    scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
    out_subdir = os.path.join(OUT_DIR, scenario)
    os.makedirs(out_subdir, exist_ok=True)

    argv = [
        "run_mpc_delay_aware_insertion_anchor.py",
        "--plan-dir", PLAN_DIR, "--schedule-cache", SCHED,
        "--out-dir", out_subdir, "--run-name", f"e2e_smoke_{scenario}",
        "--skip-preflight", "--protective-config-reviewed",
        "--horizon", "15", "--jacobian-adaptation-mode", "off",
        "--input-penalty", "planned", "--dipole-alignment-weight", "0.0",
        "--measured-clearance-abort-mm", "230.0",
        "--insertion-command-min-mm", "20.0", "--insertion-command-max-mm", "70.0",
        "--max-control-steps", "8" if scenario != "abort" else "3",
    ]
    sys.argv = argv

    import proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor as runner
    print(f"[diag] pf.URRTDERobot is MockRobot: {pf.URRTDERobot is MockRobot}")
    print(f"[diag] pf.RobotJointStream: {pf.RobotJointStream!r}")
    print(f"[diag] runner.pf.RobotJointStream: {runner.pf.RobotJointStream!r}")
    print(f"[diag] pf is runner.pf: {pf is runner.pf}")
    runner.main()
    print(f"\n[e2e_smoke:{scenario}] completed, output in {out_subdir}")


if __name__ == "__main__":
    main()
