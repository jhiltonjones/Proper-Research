"""Observation-only diagnostic logging for the servoJ command path.
2026-11-17, v2 (9 corrections from user review of v1). NOT imported by
any production run script yet -- integration into
`RobotJointStream`/`close_loop_path_follow.py` remains a SEPARATE,
not-yet-authorized step requiring a real hardware connection. No
hardware motion is required or performed by anything here.

Corrections from v1, applied:
1. The real configured poll rate for the completed K2 run was **60Hz**
   (`reader_poll_hz=60.0`, confirmed in that run's own `summary.json`
   `.config.reader_poll_hz`, NOT the `RobotJointStream` class default
   of 125Hz cited in the prior proposal). Nothing in this module
   hardcodes a rate; `DEFAULT_OBSERVED_K2_POLL_HZ` below records the
   fact for documentation only -- any real integration must read
   `cfg.reader_poll_hz` at call time, never assume 125Hz.
2. Binary `atomic_suspect` replaced by `ConsistencyState` (CHANGED /
   UNCHANGED / UNKNOWN). A missing/failed timestamp read is UNKNOWN,
   never silently counted as UNCHANGED. UNCHANGED is documented
   everywhere as "consistent with atomicity," never "proof of
   atomicity" -- the stale-stream confound (item 4) means UNCHANGED can
   also mean frozen.
3. `_safe_optional_call()` guards both the `getattr` lookup (missing
   method on an older library/firmware) and the call itself, and routes
   every failure through an explicit, logged `read_failures` list
   instead of swallowing it.
4. `StaleStreamDetector` tracks robot timestamp + actual_q progression
   ACROSS consecutive snapshots, distinguishing "stationary_fresh"
   (joints repeat, timestamp progresses -- genuine stillness) from
   "suspected_stale_stream" (joints repeat, timestamp does NOT
   progress -- frozen packet).
5. The three real timing quantities (publish / pickup / servo-call) are
   named and documented explicitly as non-acknowledgements; see
   `THREE_TIMING_QUANTITIES_DOC` below. No new fields are added to
   `BackgroundArmStreamer` (it already keeps these three separate); this
   module only re-states the constraint so it cannot be lost on reuse.
6. `DiagnosticLogSink`: logging failures can never raise into the
   caller (`emit()` catches everything), queue overflow is counted
   (`dropped_count`), and `stop()` returns whether the writer thread
   actually drained within its timeout (`drained_cleanly`) instead of
   assuming it did.
7/8. See `test_servo_diagnostic_logging_prep.py` for the 7 required
   failure-mode tests, the cadence/control-state-unchanged check, and
   the integrated-mock overhead measurement.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

# Documentation only -- NOT used as a default anywhere below. The real
# value for any given run must come from that run's own cfg.reader_poll_hz.
DEFAULT_OBSERVED_K2_POLL_HZ = 60.0

THREE_TIMING_QUANTITIES_DOC = """
Three distinct host-side timing quantities already exist in
BackgroundArmStreamer's event log (close_loop_path_follow.py) and must
stay distinct -- NONE of them is a robot acknowledgement:
  - "publish"       : latest_publication_time_s -- when the host last
                       called update_target() (close_loop_path_follow.py:194-207)
  - "pickup"         : first_pickup_time_s -- when the 20ms streamer
                       thread first read that target (close_loop_path_follow.py:246-256)
  - "servo_j_call"   : host_servo_call_time_s -- when servo_j() was
                       actually invoked on the host side (close_loop_path_follow.py:257-281)
None of these three, nor anything in this module, means "the robot
confirmed receipt of this command."
"""


class ConsistencyState(str, Enum):
    CHANGED = "changed"       # robot timestamp moved during the snapshot window
    UNCHANGED = "unchanged"   # consistent WITH atomicity -- does NOT prove it
    UNKNOWN = "unknown"       # a timestamp read failed/was missing -- never
                               # silently treated as "unchanged"


def runtime_environment_snapshot() -> dict:
    """No hardware access. Pure interpreter/module introspection."""
    out: dict = {
        "python_executable": sys.executable,
        "sys_path_head": sys.path[:3],
    }
    for modname in ("rtde_control", "rtde_receive"):
        try:
            mod = __import__(modname)
            out[f"{modname}_file"] = getattr(mod, "__file__", None)
        except Exception as exc:  # noqa: BLE001
            out[f"{modname}_file"] = None
            out[f"{modname}_import_error"] = repr(exc)
    try:
        import importlib.metadata as _im

        out["ur_rtde_version"] = _im.version("ur_rtde")
    except Exception as exc:  # noqa: BLE001
        out["ur_rtde_version"] = None
        out["ur_rtde_version_error"] = repr(exc)
    return out


def effective_servo_args(time_s: float, lookahead_time: float, gain: int) -> dict:
    """Trivial: records exactly what a given servo_j() call site is about
    to pass, with no transformation."""
    return {
        "servo_time_s": float(time_s),
        "servo_lookahead_time_s": float(lookahead_time),
        "servo_gain": int(gain),
    }


def _safe_optional_call(
    obj: Any, method_name: str, read_failures: list
) -> Optional[Any]:
    """Guards BOTH the attribute lookup (method may not exist on a given
    library version/firmware) AND the call itself (method exists but
    raises). Every failure is appended to `read_failures` -- never
    silently swallowed."""
    fn = getattr(obj, method_name, None)
    if fn is None:
        read_failures.append({"field": method_name, "reason": "method_missing"})
        return None
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        read_failures.append(
            {"field": method_name, "reason": "call_raised", "error": repr(exc)}
        )
        return None


@dataclass
class RobotSideSnapshot:
    host_time_before_s: float
    host_time_after_s: float
    # Raw, unprocessed robot-side timestamp reads -- exactly as returned.
    robot_timestamp_before_raw: Optional[float]
    robot_timestamp_after_raw: Optional[float]
    actual_q: Optional[list]
    # Deliberately NOT named "target_ack"/"command_echo" -- semantics
    # (raw echo vs. internally-smoothed setpoint) are unverified.
    target_q_raw_semantics_unverified: Optional[list]
    speed_scaling: Optional[float]
    speed_scaling_combined: Optional[float]
    read_failures: list = field(default_factory=list)
    within_snapshot_consistency: ConsistencyState = field(init=False)

    def __post_init__(self) -> None:
        a, b = self.robot_timestamp_before_raw, self.robot_timestamp_after_raw
        if a is None or b is None:
            # A missing timestamp read must NEVER count as verified
            # consistency -- it is unknown, full stop.
            self.within_snapshot_consistency = ConsistencyState.UNKNOWN
        elif a != b:
            self.within_snapshot_consistency = ConsistencyState.CHANGED
        else:
            # Consistent WITH atomicity. Does NOT prove it -- a frozen/
            # stale RTDE stream would also report "unchanged" here; see
            # StaleStreamDetector for the cross-snapshot check that
            # distinguishes the two.
            self.within_snapshot_consistency = ConsistencyState.UNCHANGED


def robot_side_snapshot(rtde_r: Any) -> RobotSideSnapshot:
    """`rtde_r` is any object exposing getTimestamp/getActualQ/getTargetQ/
    getSpeedScaling/getSpeedScalingCombined -- a real rtde_receive.
    RTDEReceiveInterface, or (as used by this module's own self-test) a
    mock. No assumption that the calls below are atomic."""
    read_failures: list = []
    host_before = time.monotonic()
    ts_before = _safe_optional_call(rtde_r, "getTimestamp", read_failures)
    actual_q = _safe_optional_call(rtde_r, "getActualQ", read_failures)
    target_q = _safe_optional_call(rtde_r, "getTargetQ", read_failures)
    speed_scaling = _safe_optional_call(rtde_r, "getSpeedScaling", read_failures)
    speed_scaling_combined = _safe_optional_call(
        rtde_r, "getSpeedScalingCombined", read_failures
    )
    ts_after = _safe_optional_call(rtde_r, "getTimestamp", read_failures)
    host_after = time.monotonic()

    return RobotSideSnapshot(
        host_time_before_s=host_before,
        host_time_after_s=host_after,
        robot_timestamp_before_raw=ts_before,
        robot_timestamp_after_raw=ts_after,
        actual_q=actual_q,
        target_q_raw_semantics_unverified=target_q,
        speed_scaling=speed_scaling,
        speed_scaling_combined=speed_scaling_combined,
        read_failures=read_failures,
    )


class StaleStreamDetector:
    """Tracks robot-side timestamp + actual_q progression ACROSS
    consecutive snapshots (not just within one) to distinguish:
      - "stationary_fresh": joints repeat, but the robot TIMESTAMP keeps
        advancing -- the robot is genuinely holding still; the stream is
        alive.
      - "suspected_stale_stream": joints repeat AND the timestamp does
        NOT advance -- a frozen/cached packet (the exact failure mode
        already documented in state_stream.py's 2026-09-17 comment),
        not genuine stillness.
    These look identical in raw joint values alone; only the timestamp's
    progression over time tells them apart."""

    def __init__(self) -> None:
        self._last_ts: Optional[float] = None
        self._last_actual_q: Optional[list] = None
        self.flat_timestamp_streak = 0

    def update(self, snap: RobotSideSnapshot) -> dict:
        ts = snap.robot_timestamp_after_raw
        q = snap.actual_q
        timestamp_progressed: Optional[bool] = None
        joints_repeated: Optional[bool] = None
        if ts is not None and self._last_ts is not None:
            timestamp_progressed = ts != self._last_ts
        if q is not None and self._last_actual_q is not None:
            joints_repeated = q == self._last_actual_q

        if timestamp_progressed is False:
            self.flat_timestamp_streak += 1
        elif timestamp_progressed is True:
            self.flat_timestamp_streak = 0

        if ts is not None:
            self._last_ts = ts
        if q is not None:
            self._last_actual_q = q

        if joints_repeated is True and timestamp_progressed is True:
            verdict = "stationary_fresh"
        elif joints_repeated is True and timestamp_progressed is False:
            verdict = "suspected_stale_stream"
        elif joints_repeated is False:
            verdict = "moving"
        else:
            verdict = "unknown"  # not enough history yet, or missing reads
        return {
            "verdict": verdict,
            "timestamp_progressed": timestamp_progressed,
            "joints_repeated": joints_repeated,
            "flat_timestamp_streak": self.flat_timestamp_streak,
        }


def correspond_to_command_seq(
    snapshot_t_s: float, publish_events: list[dict]
) -> Optional[int]:
    """Host-side wall-clock correspondence ONLY -- never reads target_q.
    Establishes "this snapshot was taken after command N was published,
    before N+1" -- a temporal ordering, never a receipt acknowledgement."""
    candidates = [e for e in publish_events if e.get("publish_time_s", 1e18) <= snapshot_t_s]
    if not candidates:
        return None
    best = max(candidates, key=lambda e: e["publish_time_s"])
    return best.get("seq")


class DiagnosticLogSink:
    """Queue-based sink modeled on BackgroundArmStreamer's own existing
    writer-thread pattern, extended with explicit, observable failure
    accounting (item 6):
      (a) emit() never raises -- a logging failure of ANY kind (queue
          full, unexpected exception) is caught and counted, never
          propagated to the caller.
      (b) queue overflow is counted separately (`dropped_count`), not
          silently lost and not conflated with write failures.
      (c) a write-time failure inside the writer thread is counted
          SEPARATELY (`write_failure_count`) -- dropped-at-enqueue and
          failed-at-write are two different failure points and must be
          reportable as two different checks, not folded into one.
      (d) stop() reports whether the writer thread actually drained and
          joined within its timeout -- NOT assumed to have drained."""

    def __init__(self, path, maxsize: int = 2000) -> None:
        self._queue: "queue.Queue[dict]" = queue.Queue(maxsize=maxsize)
        self._path = path
        self.dropped_count = 0          # queue was full (or emit() itself failed)
        self.write_failure_count = 0    # writer thread's disk write raised
        self.enqueued_count = 0
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        self.drained_cleanly: Optional[bool] = None  # None until stop() is called

    def emit(self, event: dict) -> bool:
        try:
            self._queue.put_nowait(event)
            self.enqueued_count += 1
            return True
        except queue.Full:
            self.dropped_count += 1
            return False
        except Exception:  # noqa: BLE001 -- must never raise into the caller
            self.dropped_count += 1
            return False

    def start(self) -> None:
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="diag-log-writer", daemon=True
        )
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        import json

        with open(self._path, "a") as fh:
            while self._running or not self._queue.empty():
                try:
                    event = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    fh.write(json.dumps(event) + "\n")
                    fh.flush()
                except Exception:  # noqa: BLE001
                    self.write_failure_count += 1

    def stop(self, timeout: float = 2.0) -> bool:
        """Returns True iff the writer thread actually drained the queue
        and joined within `timeout` -- a real, checked result, not an
        assumption."""
        self._running = False
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout)
            self.drained_cleanly = (
                not self._writer_thread.is_alive() and self._queue.empty()
            )
            self._writer_thread = None
        else:
            self.drained_cleanly = True
        return self.drained_cleanly
