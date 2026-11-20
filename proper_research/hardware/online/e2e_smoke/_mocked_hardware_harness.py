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
import faulthandler
import sys as _sys_early

# argv: [script, scenario, out_dir, plan_dir, schedule_cache]. out_dir is
# required (no hardcoded job-tmp path) so the pytest wrapper can point
# this at a fresh tmp_path per test; plan_dir/schedule_cache default to
# the repo-relative plan used throughout this session's offline work.
_SCENARIO = _sys_early.argv[1] if len(_sys_early.argv) > 1 else "normal"
_OUT_DIR = _sys_early.argv[2] if len(_sys_early.argv) > 2 else "/tmp/e2e_mocked_hardware_smoke"
_PLAN_DIR = (
    _sys_early.argv[3] if len(_sys_early.argv) > 3 else
    "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
)
_SCHED = (
    _sys_early.argv[4] if len(_sys_early.argv) > 4 else
    "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
)

FAULT_LOG_PATH = f"{_OUT_DIR}/faulthandler_dump.txt"
import os as _os_early
_os_early.makedirs(_OUT_DIR, exist_ok=True)
_fault_fh = open(FAULT_LOG_PATH, "w")
faulthandler.enable(file=_fault_fh, all_threads=True)


# The real rtde_control/rtde_receive .so extensions are importable in this
# environment with no hardware I/O at import time (confirmed directly:
# "import rtde_control, rtde_receive" alone opens no socket) -- and a
# LEGACY import chain (AnalyticalBeamJacobianProvider ->
# beam_hardware_experiment_v2 -> beam_hardware_experiment -> lab_ready_mpc
# -> live_hardware_control -> robot_class.py) does "from rtde_control import
# RTDEControlInterface" at ITS OWN module level, unconditionally, purely to
# have the class name available -- NOT to connect. Poisoning sys.modules
# for the whole package (tried first; see preserved_e2e/ for that attempt)
# breaks this legitimate, hardware-unrelated import. So: let the real
# modules import normally, then replace ONLY the two classes that actually
# open a connection on construction, with classes whose __init__ raises
# immediately and explicitly -- construction fails hard, import does not.
import rtde_control as _real_rtde_control
import rtde_receive as _real_rtde_receive


class _BlockedRTDEControlInterface:
    def __init__(self, *a, **kw):
        raise RuntimeError(
            "BLOCKED: real rtde_control.RTDEControlInterface(...) was "
            f"about to be constructed (args={a!r} kwargs={kw!r}) inside a "
            "hardware-blocked test process."
        )


class _BlockedRTDEReceiveInterface:
    def __init__(self, *a, **kw):
        raise RuntimeError(
            "BLOCKED: real rtde_receive.RTDEReceiveInterface(...) was "
            f"about to be constructed (args={a!r} kwargs={kw!r}) inside a "
            "hardware-blocked test process."
        )


_real_rtde_control.RTDEControlInterface = _BlockedRTDEControlInterface
_real_rtde_receive.RTDEReceiveInterface = _BlockedRTDEReceiveInterface
print("[isolation] rtde_control/rtde_receive import normally; their interface classes are replaced with construction-blocking stand-ins")

import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/jack/Proper-Research")
os.chdir("/home/jack/Proper-Research")

import proper_research.hardware.ur_rtde_robot as _ur_rtde_robot_module
import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.hardware.online.camera_source import CameraSource, CameraConfig
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = _PLAN_DIR
SCHED = _SCHED
OUT_DIR = _OUT_DIR


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

    def get_forward_kinematics(self, joints=None):
        # AnalyticalBeamJacobianProvider finite-differences robot.get_
        # forward_kinematics() to build the joint Jacobian -- needs a real
        # FK, not a stub. Reuse the same calibrated DH model validated
        # elsewhere this session (urik.CONFIG, ~0.11mm match to the real
        # robot's own FK), not an arbitrary mock.
        import proper_research.hardware.ur_magnet_ik_jacobian_validation as _urik
        from scipy.spatial.transform import Rotation as _Rot
        q = np.asarray(joints if joints is not None else np.zeros(6), dtype=float)
        res = _urik.forward_kinematics(q, _urik.corrected_dh_from_config(_urik.CONFIG))
        xyz = res.T_R_F[:3, 3]
        rotvec = _Rot.from_matrix(res.T_R_F[:3, :3]).as_rotvec()
        return list(np.concatenate([xyz, rotvec]))

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
        # 2026-11-20: a prior version drifted q6 by a uniform
        # 0.0005*tick added to all 6 joints -- an arbitrary, physically
        # meaningless motion that does NOT correspond to the tip staying
        # at tip0 (measured_beam_position, held fixed by the frame
        # processor below). That self-inconsistency -- FK-implied tip
        # drifting away while the externally "measured" tip stays frozen
        # -- is what made every tick's QP solve genuinely infeasible
        # (confirmed by an isolated worker test: the EXACT real q0/tip0
        # pair with NO drift solves cleanly; see
        # isolate_worker_feasibility.py). Kept genuinely static here --
        # this harness exercises the control LOOP's machinery, not
        # tracking quality, so a frozen, self-consistent plant is the
        # correct choice, not an arbitrary moving one.
        while self._driver_running:
            now = time.monotonic()
            with self._shared["lock"]:
                self._shared["q6"] = self._shared["q0"].copy()
                self._shared["stamp"] = now - 0.003  # 3ms old, realistic
            self._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
            self._latest_frame_stamp = now
            self._reconstruct_body(now)
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

    # close_loop_path_follow.py's main() does a LOCAL
    # "from proper_research.hardware.ur_rtde_robot import URRTDERobot"
    # (re-executed every call), which bypasses a pf.URRTDERobot module-
    # attribute patch entirely -- confirmed by traceback evidence (the
    # real ur_rtde_robot.connect() -> rtde_control.RTDEControlInterface
    # ran despite pf.URRTDERobot being verified == MockRobot moments
    # before). The local import re-reads the SOURCE module's attribute
    # at call time, so patch that attribute instead.
    _ur_rtde_robot_module.URRTDERobot = MockRobot
    pf.URRTDERobot = MockRobot  # harmless belt-and-suspenders, not load-bearing
    pf.RobotJointStream = lambda *a, **kw: MockReader(shared_state, *a, **kw)
    def _camera_factory(*a, **kw):
        kw.pop("frame_processor", None)  # ignore the real mapper -- always use the synthetic one
        return MockedCameraSource(*a, shared_state=shared_state, frame_processor=synthetic_frame_processor, **kw)
    pf.CameraSource = _camera_factory

    # advancer: use its own real dry_run mode, not a mock -- pre-set on the
    # singleton CONFIG before the runner's main() reads/sets other fields.
    pf.CONFIG.advancer_dry_run = True

    scenario = _SCENARIO
    # measured clearance is pinned at ~280mm by the static plant (see
    # MockedCameraSource._driver_loop's comment) -- "abort" sets the
    # threshold ABOVE that so the real external-clearance check genuinely
    # fires; "timeout" gives the real worker a deadline it cannot meet.
    clearance_abort_mm = "300.0" if scenario == "abort" else "230.0"
    deadline_ms = "0.01" if scenario == "timeout" else "70.0"
    out_subdir = os.path.join(OUT_DIR, scenario)
    os.makedirs(out_subdir, exist_ok=True)

    # Condition E offline smoke check: --jacobian-adaptation-mode active
    # (not just "off", as every other scenario here uses) -- exercises
    # the online g_hat estimator across real consecutive ticks, which a
    # single isolated try_solve() call cannot (the estimator needs
    # consecutive measured-vs-predicted deltas). More ticks than the
    # other scenarios so the estimator has enough history to actually
    # update at least once.
    adaptation_mode = "active" if scenario == "adaptation_active" else "off"
    max_steps = "20" if scenario == "adaptation_active" else ("8" if scenario not in ("abort",) else "5")

    argv = [
        "run_mpc_delay_aware_insertion_anchor.py",
        "--plan-dir", PLAN_DIR, "--schedule-cache", SCHED,
        "--out-dir", out_subdir, "--run-name", f"e2e_smoke_{scenario}",
        "--skip-preflight", "--protective-config-reviewed",
        "--horizon", "15", "--jacobian-adaptation-mode", adaptation_mode,
        "--input-penalty", "planned", "--dipole-alignment-weight", "0.0",
        "--measured-clearance-abort-mm", clearance_abort_mm,
        "--deadline-ms", deadline_ms,
        "--insertion-command-min-mm", "20.0", "--insertion-command-max-mm", "70.0",
        "--max-control-steps", max_steps,
    ]
    if scenario == "frozen_jacobian":
        # Condition B offline smoke check: confirms --freeze-jacobian
        # (added 2026-11-20) builds a valid frozen schedule file and the
        # real worker accepts and solves against it, with no hardware.
        argv += ["--freeze-jacobian", "--freeze-jacobian-index", "0"]
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
