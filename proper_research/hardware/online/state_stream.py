"""Part 2: stream the full measured state (vision tip + live robot joints).

Builds on ``stream_vision.py``:

* **persistent robot connection.**  One ``robot_class.URRtde`` session is held
  open for the whole run; a background thread publishes the live joint angles
  and TCP pose to latest-value slots (``RobotJointStream``).  Nothing is
  commanded -- this is the sensing half of the real-time loop.
* **fast tip-only vision in the NEW beam frame.**  No skeletonize / centreline
  route (only the tip position is needed now), and the tip pixel is mapped
  through the ``robotics_frame_measurement_validation`` frame
  (``beam_frame_rotation_from_axes`` + the baked pivot), not the old
  ``pivot_local_point_to_robot`` path.  ~10x faster than
  ``reconstruct_beam_within_vessel``.
* **insertion length** via ``AdvancerSink`` (software integral; the advancer
  has no encoder).  ``connect_advancer=False`` keeps it a pure estimate.

The published state carries what the joint-space controllers need in part 3:
``z = [q1..q6, insertion]`` (from ``robot_joints`` + ``insertion_length_m``)
and ``tip_position_m``.

    python -m proper_research.hardware.online.state_stream
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.messages import (
    HeartbeatLoop,
    LatestSlot,
    now_monotonic,
)
from proper_research.hardware.online.servo_diagnostic_logging_prep import (
    DiagnosticLogSink,
    _safe_optional_call,
    runtime_environment_snapshot,
)
from proper_research.rig_calibration import BEAM_BASE_XYZ_M

# 2026-11-17: Z now shares the single canonical BEAM_BASE_XYZ_M source too
# (X/Y already did). This file's own Z used to be a separate hardcoded
# -0.016567 -- the pre-2026-10-02-remount value, already documented as
# stale and fixed in planning_context.py's _calibrated_default_initial_
# poses(), plan_shape_path.py, _bend_axes_R(), and
# robotics_frame_measurement_validation.py (see those files' own comments
# for the physical remount/live-FK-recalibration evidence). This was the
# one remaining unpatched call site of that same staleness: it fed
# start_tip_R in close_loop_path_follow.py's _fit_planner_to_robot(),
# producing a live/planner registration offset (controller_metadata.json's
# planner_to_live_t_fit_m) with a spurious ~23.06mm Z component on live
# rectangle runs taken after this file's own last Z edit -- confirmed by
# recomputing t_fit_z for those runs with this corrected value: landed at
# 0.096mm, matching the known-good pre-edit baseline run's own t_fit_z
# exactly. Checked for double-correction first: _fit_planner_to_robot()
# does an unconditional rigid-fit registration with no special-casing for
# this gap, so fixing the root default just makes that registration small
# again, not doubly-adjusted; the vessel rig is unaffected (it builds its
# own self-contained PivotConfig via raised_state_stream_config_factory /
# _raised_pivot_config(), never reading this constant at all).
_DEFAULT_T_ROBOT_BEAM_POSE6 = (
    float(BEAM_BASE_XYZ_M[0]), float(BEAM_BASE_XYZ_M[1]), float(BEAM_BASE_XYZ_M[2]), 0.0, -1.5707963, 0.0,
)


# =============================================================================
# USER CONFIGURATION -- edit here; there are no terminal arguments
# =============================================================================


@dataclass
class StateStreamConfig:
    # --- consumer loop --------------------------------------------------
    stream_hz: float = 30.0
    duration_s: Optional[float] = 30.0     # None -> run until Ctrl+C
    max_state_age_s: float = 0.30          # vision older than this -> stale
    warmup_timeout_s: float = 20.0
    print_every: int = 15

    # --- persistent robot (robot_class.URRtde) --------------------
    robot_ip: str = "192.168.56.101"
    robot_poll_hz: float = 125.0           # joint / pose poll rate
    robot_max_age_s: float = 0.10          # joints older than this -> not attached

    # --- advancer / insertion length ---------------------------------
    connect_advancer: bool = False         # False -> pure software insertion estimate
    advancer_port: str = "/dev/ttyACM0"
    advancer_dry_run: bool = True          # True even when connected -> no serial writes
    initial_insertion_m: float = 0.040

    # --- fast tip-only vision --------------------------------------
    cam_index: int = 0
    exposure: float = 28.0
    gain: float = 0.0
    warmup_frames: int = 15
    grab_period_s: float = 0.004
    reconstruct_period_s: float = 0.01     # tip detection is cheap; loop fast
    image_filename: str = "/dev/shm/proper_state_stream_frame.png"
    roi_polygon_path: str = "/home/jack/Proper-Research/custom_area.json"
    manual_boundary_path: str = (
        "/home/jack/Proper-Research/manual_vessel_boundaries.json"
    )
    calibration_points_path: str = (
        "/home/jack/Proper-Research/calibration_points.json"
    )
    pivot_hint_px: tuple[int, int] = (318, 292)
    marker_min_area_px2: float = 4.0
    marker_max_area_px2: float = 40000.0
    # 2026-09-14: how many red markers must be visible. Default (3-4) matches
    # every live-control caller's validated requirement -- do not lower this
    # for anything that needs a real tangent (mag_start/tangent_start
    # markers). Set marker_min_count=2 explicitly for a short-insertion
    # (<~25mm) calibration check, where only base+tip may be visible.
    marker_min_count: int = 3
    marker_max_count: int = 4
    marker_sat_min: int = 50
    marker_val_min: int = 40
    marker_red_hue1_high: int = 10
    marker_red_hue2_low: int = 170
    emit_chord_tangent: bool = True        # cheap tangent_start->tip chord (not the true tangent)

    # --- NEW beam frame B (matches robotics_frame_measurement_validation) --
    # 2026-09-14: re-calibrated against a 1cm-square checkerboard grid
    # (diagonal baseline) -- see calibration_points.json's "source" field.
    # Agrees with the old 38.0mm hand-clicked estimate to ~0.4% once measured
    # the same (diagonal, not naively-averaged) way -- a precision
    # refinement, not a correction of a real error.
    # SUPERSEDED 2026-11-17: the 80.62257748298549mm value above was found
    # stale (camera zoom/focus/position had changed since the 2026-09-14
    # capture) -- see calibration_points.json's "source" field for the
    # full recalibration writeup. This constant MUST be kept in sync with
    # calibration_points.json's own "known_distance_mm" field by hand --
    # NewFrameTipMapper only reads "points_px" from that JSON file; it
    # does NOT read "known_distance_mm" from it (confirmed by direct code
    # read of __init__: `known_calibration_distance_mm` always comes from
    # THIS Python default, never from the JSON). Updating only the JSON's
    # points without updating this constant combines NEW (closer-spaced)
    # points with the OLD (stale, larger) known-distance value, producing
    # an even MORE inflated scale than before the JSON update -- this is
    # exactly the bug caught live on 2026-11-17 (a post-recalibration
    # check read 38.51mm, worse than the original 34.0mm, until this
    # constant was corrected to match).
    #
    # SECOND SUPERSESSION, same day: the checkerboard-geometry-derived
    # value immediately above (50.99019513592785) was itself abandoned --
    # the board's residual top-bottom tilt (~3.7% CV, never fully
    # resolved despite two reposition attempts) made it unreliable at the
    # <5mm precision needed here. Replaced by a value back-solved
    # directly from a known physical ground truth (24.0mm measured
    # insertion length) instead of board geometry -- see
    # calibration_points.json's "source" field for the exact derivation
    # (correction_factor = 24.0 / 30.19 applied to the prior constant).
    # Keep this in sync with calibration_points.json's "known_distance_mm"
    # by hand, same caveat as above.
    # FOURTH SUPERSESSION, 2026-11-17: back-solved from a fresh 33.731mm
    # live reading against a new 41.0mm ground truth, after the advancer-
    # motion hypothesis was investigated and ruled out by code trace (see
    # calibration_points.json's "source" field for the full writeup,
    # including the flagged, unresolved discrepancy against an earlier
    # 26.3mm/~25mm reading from before the run's own move_j reset).
    # REVERTED 2026-11-17: back to the original 2026-09-14 value, per the
    # user's explicit decision to stop this session's recalibration
    # process. Not a finding that this value is more accurate -- live
    # measurements swung across many different values today regardless of
    # which calibration was active; see calibration_points.json's "source"
    # field. Kept in sync with that file's "known_distance_mm" by hand, as
    # always -- this is the exact field whose drift caused a real bug two
    # rounds earlier in this same session.
    known_calibration_distance_mm: float = 80.62257748298549
    saved_axis_signs_for_positive_beam_xy: tuple[float, float] = (-1.0, 1.0)
    saved_axis_convention: str = "image_cartesian"
    beam_axial_axis_R: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    beam_plane_normal_axis_R: tuple[float, float, float] = (0.0, 0.0, -1.0)
    T_robot_beam_pose6: tuple[float, float, float, float, float, float] = _DEFAULT_T_ROBOT_BEAM_POSE6
    anchor_beam_origin_to_detected_base: bool = True

    # --- output ------------------------------------------------------
    output_root: str = "state_stream_logs"
    run_name: str = "state_stream"


CONFIG = StateStreamConfig()


# =============================================================================
# persistent robot: one URRtde session, background joint/pose poll
# =============================================================================


class RobotJointStream:
    """Hold one robot session open; poll joints + pose in the background.

    ``receive_only=True`` opens **only** ``rtde_receive`` -- no control script,
    no RTDE watchdog, so a slow consumer can never fault the arm.  Use it for
    pure state streaming.  ``receive_only=False`` opens a full
    ``robot_class.URRtde`` (control + receive) so ``.robot`` can issue
    ``speed_j`` / ``getForwardKinematics``; then the caller is responsible for
    stopping the arm and not stalling a watchdog.
    """

    def __init__(
        self,
        ip: str,
        *,
        poll_hz: float = 125.0,
        receive_only: bool = True,
        enable_diagnostic_log: bool = False,
        diagnostic_log_path: Optional[str] = None,
        diagnostic_stall_threshold_s: Optional[float] = None,
        diagnostic_effective_servo_args: Optional[dict] = None,
        _test_rtde_r: Any = None,
    ) -> None:
        self._ip = str(ip)
        self._receive_only = bool(receive_only)
        self._robot = None
        self._rtde_r = None
        if _test_rtde_r is not None:
            # Test-only injection point. Used ONLY by
            # test_robot_joint_stream_diagnostic_integration.py to exercise
            # this REAL class/real _poll_body() against a mocked RTDE
            # receive object, with NO real `rtde_receive.
            # RTDEReceiveInterface` (and therefore no real hardware
            # connection) ever constructed. No production caller passes
            # this argument.
            if not receive_only:
                raise ValueError(
                    "_test_rtde_r injection only supported for receive_only=True"
                )
            self._rtde_r = _test_rtde_r
        elif self._receive_only:
            import rtde_receive

            self._rtde_r = rtde_receive.RTDEReceiveInterface(ip)
            if not self._rtde_r.isConnected():
                raise RuntimeError(f"rtde_receive could not connect to {ip}")
        else:
            from robot_class import URRtde

            self._robot = URRtde(ip)
        self._poll_hz = float(poll_hz)
        self.joints_slot: LatestSlot[np.ndarray] = LatestSlot()
        self.pose_slot: LatestSlot[np.ndarray] = LatestSlot()
        self._loop: Optional[HeartbeatLoop] = None
        self.reads = 0
        self.read_failures = 0
        self.last_error = ""
        # 2026-09-17: RTDE's getActualQ()/getActualTCPPose() can silently
        # keep returning the SAME cached packet (no exception, socket still
        # "connected") if the underlying data stream stalls -- confirmed
        # live twice (see servoj-realization-gain memory): joints_slot kept
        # refreshing its timestamp on a bit-identical reading for an entire
        # ~28s run, so cfg.robot_max_age_s's staleness bound never tripped
        # and a fully stale joint feed was consumed as "fresh" throughout.
        # Track the last raw reading per channel so an exact repeat can be
        # told apart from genuine (always-noisy) robot motion.
        self._last_raw_joints: Optional[np.ndarray] = None
        self._last_raw_pose: Optional[np.ndarray] = None
        self.stale_repeats = 0

        # --- 2026-11-17: additive, observation-only diagnostic logging ---
        # (SERVO_DIAGNOSTIC_LOGGING_V2_CORRECTIONS_2026-11-17.md). None of
        # this is read by, or can influence, _poll_body()'s existing
        # control-relevant logic above (joints/pose/reads/read_failures/
        # stale_repeats/last_error) -- it is appended strictly after that
        # logic, in its own attributes, wrapped in its own total
        # exception guard (_diagnostic_tick()).
        self._diagnostic_enabled = bool(enable_diagnostic_log) and self._receive_only
        self._diag_sink: Optional[DiagnosticLogSink] = None
        self._diag_log_path = diagnostic_log_path
        if self._diagnostic_enabled and diagnostic_log_path:
            self._diag_sink = DiagnosticLogSink(diagnostic_log_path)
            self._diag_sink.start()
            # Item C: library/version identity, servo args, and poll rate
            # must land in the REAL SAVED FILE, not just in-memory state --
            # written once, here, as the first row.
            startup_event = {"event": "startup", "poll_hz": self._poll_hz}
            startup_event.update(runtime_environment_snapshot())
            if diagnostic_effective_servo_args is not None:
                startup_event["effective_servo_args"] = diagnostic_effective_servo_args
            self._diag_sink.emit(startup_event)
        # elapsed-HOST-time basis for duplicate-packet vs. sustained-stall
        # (host monotonic time of the last OBSERVED robot-timestamp change)
        self._diag_last_ts_value: Optional[float] = None
        self._diag_last_ts_change_host_time: Optional[float] = None
        self._diag_stall_threshold_s = (
            float(diagnostic_stall_threshold_s)
            if diagnostic_stall_threshold_s is not None
            else max(0.5, 10.0 / self._poll_hz)
        )
        self.diag_ticks = 0
        self.diag_internal_errors = 0  # the diagnostic tick itself raised
        self.diag_duplicate_packets = 0
        self.diag_sustained_stalls = 0
        self.diag_shutdown_drained_cleanly: Optional[bool] = None

    def start(self) -> None:
        self._loop = HeartbeatLoop(
            "robot-poll", self._poll_body, period_s=1.0 / self._poll_hz
        )
        self._loop.start()

    def reconnect_receive(self) -> None:
        """Tear down and rebuild the receive-only RTDE connection.

        2026-09-17: ``getActualQ()``/``getActualTCPPose()`` have been
        observed to keep returning the SAME cached packet indefinitely, with
        no exception, after a stretch of heavy CPU-bound work in this same
        process (e.g. an LTV Jacobian schedule build -- confirmed live: 73%
        of polls were exact repeats right after one such build finished, see
        servoj-realization-gain memory). Waiting alone does not clear it --
        the stuck socket needs a fresh handshake. Call this after any such
        stretch and before trusting ``latest_joints``/``latest_pose`` again.
        Receive-only mode only.
        """
        if not self._receive_only:
            raise RuntimeError("reconnect_receive only applies to receive_only=True")
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        try:
            self._rtde_r.disconnect()
        except Exception:
            pass
        import rtde_receive

        self._rtde_r = rtde_receive.RTDEReceiveInterface(self._ip)
        if not self._rtde_r.isConnected():
            raise RuntimeError(f"rtde_receive could not reconnect to {self._ip}")
        self._last_raw_joints = None
        self._last_raw_pose = None
        self.start()

    def _poll_body(self, _t_start: float) -> None:
        try:
            if self._receive_only:
                joints = self._rtde_r.getActualQ()
                pose = self._rtde_r.getActualTCPPose()
            else:
                joints = self._robot.get_joints()
                pose = self._robot.get_pose()
        except Exception as exc:  # noqa: BLE001 - a dropped read must not kill the loop
            self.read_failures += 1
            self.last_error = repr(exc)
            if self._robot is not None:
                try:
                    self._robot.ensure_connected()
                except Exception:
                    pass
            return
        # Only refresh the slot (and its freshness timestamp) when the raw
        # reading actually changed. A real robot's joints/pose are never
        # bit-identical two polls in a row (sensor noise alone rules that
        # out) -- an exact repeat means the underlying RTDE stream is stuck,
        # not that the robot is stationary. Leaving the slot's old timestamp
        # in place lets max_age_s correctly age it out instead of treating
        # a frozen feed as live.
        if joints is not None:
            joints_arr = np.asarray(joints, dtype=float).reshape(6)
            if self._last_raw_joints is None or not np.array_equal(joints_arr, self._last_raw_joints):
                self._last_raw_joints = joints_arr
                self.joints_slot.set(joints_arr)
            else:
                self.stale_repeats += 1
                self.last_error = "stale_repeat: getActualQ() returned an identical reading"
        if pose is not None:
            pose_arr = np.asarray(pose, dtype=float).reshape(6)
            if self._last_raw_pose is None or not np.array_equal(pose_arr, self._last_raw_pose):
                self._last_raw_pose = pose_arr
                self.pose_slot.set(pose_arr)
        self.reads += 1

        # Strictly additive from here down: nothing above this line is
        # read, and nothing below this line writes to joints_slot/
        # pose_slot/reads/read_failures/stale_repeats/last_error. A
        # failure anywhere in this block can set diag_internal_errors but
        # can never raise out of _poll_body() or alter what was already
        # computed above.
        if self._diagnostic_enabled:
            try:
                self._diagnostic_tick()
            except Exception:  # noqa: BLE001
                self.diag_internal_errors += 1

    def _diagnostic_tick(self) -> None:
        """Observation-only. Reads getTimestamp/getTargetQ/getSpeedScaling/
        getSpeedScalingCombined from the SAME already-open `self._rtde_r`
        connection `_poll_body()` already uses for getActualQ() -- no
        additional RTDE connection. Guards both attribute lookup and
        invocation per-field (`_safe_optional_call`) so a missing method on
        an older library/firmware build degrades that one field to None
        with a logged reason, not a crash."""
        host_before = now_monotonic()
        failures: list = []
        ts_before = _safe_optional_call(self._rtde_r, "getTimestamp", failures)
        # Diagnostic-only SECOND read of actual_q, bracketed by the SAME
        # ts_before/ts_after pair as target_q below -- deliberately named
        # distinctly from _poll_body()'s own `joints` variable (the one
        # control code actually consumes via joints_slot/_last_raw_joints).
        # This read's result is NEVER assigned to joints_slot or
        # _last_raw_joints -- see
        # test_actual_q_diagnostic_read_never_reaches_control_slot for the
        # verification of that separation.
        actual_q_diagnostic_second_read = _safe_optional_call(
            self._rtde_r, "getActualQ", failures
        )
        target_q = _safe_optional_call(self._rtde_r, "getTargetQ", failures)
        speed_scaling = _safe_optional_call(self._rtde_r, "getSpeedScaling", failures)
        speed_scaling_combined = _safe_optional_call(
            self._rtde_r, "getSpeedScalingCombined", failures
        )
        ts_after = _safe_optional_call(self._rtde_r, "getTimestamp", failures)
        host_after = now_monotonic()

        if ts_before is None or ts_after is None:
            consistency = "unknown"
        elif ts_before != ts_after:
            consistency = "changed"
        else:
            consistency = "unchanged"  # consistent WITH atomicity, not proof of it

        # Duplicate-packet vs. sustained-timestamp-stall: based on ELAPSED
        # HOST WALL-CLOCK TIME since the robot timestamp last genuinely
        # changed, not a repeat-count heuristic. A handful of repeats in a
        # few milliseconds (packets arriving faster than they refresh) is
        # benign; the SAME value holding for longer than
        # `_diag_stall_threshold_s` of real elapsed host time is reported
        # as a sustained stall.
        duplicate_or_stall = None
        if ts_after is not None:
            if self._diag_last_ts_value is None or ts_after != self._diag_last_ts_value:
                self._diag_last_ts_value = ts_after
                self._diag_last_ts_change_host_time = host_after
            else:
                elapsed = host_after - (self._diag_last_ts_change_host_time or host_after)
                if elapsed >= self._diag_stall_threshold_s:
                    duplicate_or_stall = "sustained_timestamp_stall"
                    self.diag_sustained_stalls += 1
                else:
                    duplicate_or_stall = "duplicate_packet"
                    self.diag_duplicate_packets += 1

        self.diag_ticks += 1

        if self._diag_sink is not None:
            self._diag_sink.emit({
                "event": "tick",
                "host_time_before_s": host_before,
                "host_time_after_s": host_after,
                "robot_timestamp_before_raw": ts_before,
                "robot_timestamp_after_raw": ts_after,
                # Diagnostic-only second read -- NEVER the value joints_slot/
                # _last_raw_joints hold; same within_snapshot_consistency
                # bracket as target_q, since both are read between the
                # same ts_before/ts_after pair above.
                "actual_q_diagnostic_second_read": actual_q_diagnostic_second_read,
                "target_q_raw_semantics_unverified": target_q,
                "speed_scaling": speed_scaling,
                "speed_scaling_combined": speed_scaling_combined,
                "within_snapshot_consistency": consistency,
                "duplicate_or_stall": duplicate_or_stall,
                "read_failures": failures,
            })

    @property
    def robot(self) -> Any:
        """The live ``robot_class.URRtde`` session (for speed_j / speed_l)."""
        if self._robot is None:
            raise RuntimeError(
                "RobotJointStream is receive-only; construct with "
                "receive_only=False to get a control session."
            )
        return self._robot

    def latest_joints(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        value, _age = self.joints_slot.get(max_age_s)
        return value

    def latest_joints_with_age(
        self, max_age_s: Optional[float] = None
    ) -> "tuple[Optional[np.ndarray], float]":
        """RESTORED 2026-11-17: this method (and latest_joints_with_timestamp
        below) was accidentally deleted during this session's earlier
        diagnostic-logging edits to this class -- confirmed the real,
        reproducible root cause of two live hardware warmup failures
        ("no fresh tip+joints within warmup_timeout_s"): with this method
        missing, CameraSource's robot_joints_with_timestamp_getter lambda
        raised AttributeError on every call, silently caught by
        camera_source.py's own try/except, so estimate.robot_joints was
        always None regardless of how fresh the underlying RTDE stream
        actually was (confirmed healthy via robot_joint_stream_diagnostics.
        jsonl from the exact failed run: 1470/1470 ticks, 0 stalls, 0
        failures, continuously advancing timestamps).

        Same as latest_joints(), but also returns the age (seconds, host
        monotonic clock) of the joint reading -- previously computed inside
        joints_slot.get() and silently discarded by latest_joints() (the
        same class of gap already fixed for BackgroundArmStreamer's
        servo_j() duration).

        2026-11-16 superseded for the K2 instrumentation path: calling this
        SEPARATELY from whatever produced an existing StateEstimate's
        robot_joints (e.g. from CameraSource's own internal
        robot_joints_getter() call) is a SECOND, LATER read of this same
        slot -- it can return a newer sample than the one already baked
        into that estimate (confirmed as a real gap, not hypothetical, in
        this investigation). Left here unchanged as a standalone utility
        for callers that genuinely want "the freshest joint reading right
        now" with no existing estimate to pair with. For anything that
        must describe the SAME sample as an existing StateEstimate.
        robot_joints, use latest_joints_with_timestamp() instead (below) and
        thread the result through CameraSource's robot_joints_with_timestamp_
        getter, not this method.

        Note on provenance: this is the HOST's receipt timestamp (when this
        process's polling loop wrote the value into joints_slot), not a
        robot/RTDE-side acquisition timestamp. `rtde_receive.
        RTDEReceiveInterface` exposes its own `getTimestamp()` in the real
        RTDE API, but this codebase's `_poll_body()` never calls it --
        confirmed by reading `_poll_body()` directly (only `getActualQ()`/
        `getActualTCPPose()` are called). Exposing a true robot-side
        timestamp would need that added and verified against real
        hardware; not done here (no robot connection available), and not
        fabricated -- callers that need "robot acquisition age" specifically
        should treat it as genuinely unavailable until that is added.
        """
        return self.joints_slot.get(max_age_s)

    def latest_joints_with_timestamp(
        self, max_age_s: Optional[float] = None
    ) -> "tuple[Optional[np.ndarray], Optional[float]]":
        """Atomic (joints, raw_monotonic_timestamp) read -- see
        LatestSlot.get_with_timestamp() for the atomicity guarantee. This is
        the method CameraSource's frame-processing code uses (via the new
        robot_joints_with_timestamp_getter parameter) so the joint vector
        baked into a StateEstimate and its own timestamp come from ONE
        indivisible read, not two.
        """
        return self.joints_slot.get_with_timestamp(max_age_s)

    def latest_pose(self, max_age_s: Optional[float] = None) -> Optional[np.ndarray]:
        value, _age = self.pose_slot.get(max_age_s)
        return value

    def is_protective_stopped(self) -> bool:
        if self._rtde_r is not None:
            return bool(self._rtde_r.isProtectiveStopped())
        return bool(self._robot.protective_stopped())

    def safety_mode(self) -> int:
        if self._rtde_r is not None:
            return int(self._rtde_r.getSafetyMode())
        return int(self._robot.safety_mode())

    @property
    def healthy(self) -> bool:
        return (
            self._loop is not None
            and self._loop.alive
            and self._loop.error is None
        )

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        if self._robot is not None:
            try:
                self._robot.shutdown()
            except Exception:
                pass
        if self._rtde_r is not None:
            try:
                self._rtde_r.disconnect()
            except Exception:
                pass
        # Additive: shut down the diagnostic sink AFTER the control-
        # relevant teardown above. The real drained/joined result is
        # recorded, not assumed.
        if self._diag_sink is not None:
            dropped_before_stop = self._diag_sink.dropped_count
            write_failures_before_stop = self._diag_sink.write_failure_count
            self.diag_shutdown_drained_cleanly = self._diag_sink.stop(timeout=2.0)
            # Item C: the shutdown outcome and final counts must land in
            # the REAL SAVED FILE. The writer thread has already stopped
            # by this point (stop() joined it), so this is a direct,
            # separate append -- not routed through the (now-stopped)
            # queue/writer-thread pair.
            if self._diag_log_path:
                try:
                    with open(self._diag_log_path, "a") as fh:
                        fh.write(json.dumps({
                            "event": "shutdown",
                            "diag_ticks": self.diag_ticks,
                            "diag_internal_errors": self.diag_internal_errors,
                            "diag_duplicate_packets": self.diag_duplicate_packets,
                            "diag_sustained_stalls": self.diag_sustained_stalls,
                            "dropped_count": self._diag_sink.dropped_count,
                            "write_failure_count": self._diag_sink.write_failure_count,
                            "dropped_count_before_stop": dropped_before_stop,
                            "write_failure_count_before_stop": write_failures_before_stop,
                            "drained_cleanly": self.diag_shutdown_drained_cleanly,
                        }) + "\n")
                        fh.flush()
                except Exception:  # noqa: BLE001 -- shutdown logging must never raise
                    pass


# =============================================================================
# fast tip-only vision in the NEW beam frame
# =============================================================================


class NewFrameTipMapper:
    """``frame_bgr -> [tip_xyz(3), tangent_xyz(3)]`` in the robot base frame.

    Detects the red markers, takes the tip pixel, maps it through the same
    ``PlanarPixelCalibration`` + ``T_R_B`` (``beam_frame_rotation_from_axes`` +
    baked pivot) used by ``robotics_frame_measurement_validation``.  No
    skeletonize, no centreline route -- roughly the cost of one HSV threshold
    plus contour detection.
    """

    def __init__(self, cfg: StateStreamConfig) -> None:
        import json as _json

        from proper_research.hardware.robotics_frame_measurement_validation import (
            FrameTransform,
            PlanarPixelCalibration,
            beam_frame_rotation_from_axes,
            compute_metres_per_pixel,
        )
        from proper_research.vision import bounds_beam
        from proper_research.vision.detect_blue import (
            load_manual_vessel_boundaries_with_frame,
        )

        self._bounds_beam = bounds_beam
        self._cfg = cfg

        manual = load_manual_vessel_boundaries_with_frame(cfg.manual_boundary_path)
        self.manual_frame = manual
        with open(cfg.calibration_points_path, "r", encoding="utf-8") as handle:
            points = _json.load(handle)["points_px"]
        metres_per_pixel = compute_metres_per_pixel(
            points[0], points[1], cfg.known_calibration_distance_mm
        )
        self._calibration = PlanarPixelCalibration.from_basis_scale(
            origin_px=manual["base_px"],
            ex_saved=manual["ex_img"],
            ey_saved=manual["ey_img"],
            metres_per_pixel=metres_per_pixel,
            saved_axis_convention=cfg.saved_axis_convention,
            positive_axis_signs=cfg.saved_axis_signs_for_positive_beam_xy,
        )

        rotation = beam_frame_rotation_from_axes(
            cfg.beam_axial_axis_R, cfg.beam_plane_normal_axis_R
        )
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = np.asarray(cfg.T_robot_beam_pose6, dtype=float)[:3]
        self._T_R_B = FrameTransform("R", "B", matrix)

        self._roi_polygon = bounds_beam.load_polygon(cfg.roi_polygon_path)
        self.detections = 0
        self.detection_failures = 0
        self.last_error = ""

    @property
    def T_R_B(self) -> Any:
        return self._T_R_B

    @property
    def calibration(self) -> Any:
        return self._calibration

    def _tip_and_tangent_start_px(self, frame_bgr: np.ndarray):
        bb = self._bounds_beam
        cfg = self._cfg
        # marker_min_count defaults to 3 (the live-control-validated
        # requirement); a short beam (<~25mm insertion) may only expose 2
        # markers before the tangent_start marker clears the advancer/sheath
        # -- relax to 2 explicitly (e.g. a short-insertion calibration
        # script) rather than silently failing. Not changed by default, so
        # every existing caller keeps the original 3-4 requirement.
        candidates = bb.detect_4_red_markers_in_roi(
            frame_bgr,
            roi_box=None,
            roi_polygon=self._roi_polygon,
            min_area=cfg.marker_min_area_px2,
            max_area=cfg.marker_max_area_px2,
            sat_min=cfg.marker_sat_min,
            val_min=cfg.marker_val_min,
            hue1_high=cfg.marker_red_hue1_high,
            hue2_low=cfg.marker_red_hue2_low,
            show_debug=False,
            min_markers=cfg.marker_min_count,
            max_markers=cfg.marker_max_count,
        )
        valid_counts = tuple(range(cfg.marker_min_count, cfg.marker_max_count + 1))
        if candidates is None or len(candidates) not in valid_counts:
            raise RuntimeError(
                f"expected {cfg.marker_min_count}-{cfg.marker_max_count} red markers, "
                f"got {candidates}"
            )
        base_c, _mag_c, tangent_start_c, tip_c = bb.order_beam_marker_candidates(
            candidates, pivot_hint=tuple(cfg.pivot_hint_px)
        )
        if tangent_start_c is None:
            # 2-marker case (base, tip only) -- no tangent_start marker
            # visible. Callers that only need the tip position (e.g. a
            # length-calibration check) are unaffected; callers that need a
            # real tangent should not relax marker_min_count below 3.
            tangent_start_c = tip_c
        return (
            np.asarray(tip_c["point"], dtype=float),
            np.asarray(tangent_start_c["point"], dtype=float),
            np.asarray(base_c["point"], dtype=float),
        )

    def __call__(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        cfg = self._cfg
        try:
            tip_px, tan_px, base_px = self._tip_and_tangent_start_px(frame_bgr)
        except Exception as exc:  # noqa: BLE001
            self.detection_failures += 1
            self.last_error = repr(exc)
            return None

        offset_B = (
            self._calibration.pixels_to_beam(base_px)
            if cfg.anchor_beam_origin_to_detected_base
            else np.zeros(3)
        )
        tip_B = self._calibration.pixels_to_beam(tip_px) - offset_B
        tip_R = self._T_R_B.apply_points(tip_B)

        if cfg.emit_chord_tangent:
            tan_B = self._calibration.pixels_to_beam(tan_px) - offset_B
            chord = tip_B - tan_B
            norm = float(np.linalg.norm(chord))
            tangent_R = (
                self._T_R_B.apply_directions(chord / norm)
                if norm > 1e-9
                else np.zeros(3)
            )
        else:
            tangent_R = np.zeros(3)

        self.detections += 1
        return np.r_[np.asarray(tip_R, dtype=float).reshape(3), tangent_R]


# =============================================================================
# insertion-length estimate
# =============================================================================


class InsertionEstimate:
    """``initial + advancer.dispatched_mm`` (the advancer has no encoder)."""

    def __init__(self, initial_m: float, advancer: Optional[AdvancerSink]) -> None:
        self._initial_m = float(initial_m)
        self._advancer = advancer

    def get(self) -> float:
        if self._advancer is None:
            return self._initial_m
        try:
            return self._initial_m + 1e-3 * float(
                self._advancer.feedback().dispatched_distance_mm
            )
        except Exception:
            return self._initial_m


# =============================================================================
# helpers
# =============================================================================


class _DebugLineFilter:
    """stdout wrapper that drops the vision pipeline's chatty debug lines.

    Thread-safe by construction (line-buffered, single lock-free append) and
    far simpler than trying to give the reconstruct thread its own stdout.
    """

    _DROP = (
        "[DEBUG]", "[INFO]", "[CONTACT", "Calibration distance", "mm_per_pixel:",
        "Beam length", "C first", "C last", "x range", "y range", "R range",
        "t0 =", "z variation", "tip tangent:", "tip position:", "centreline shape:",
        "--- ", "num violating", "L_ins", "L_model", "wire_len", "tip_len",
        "wire segments", "tip segments", "n_centerline", "s range", "==========",
        "FORWARD SOLUTION", "max |", "mean|", "mean ", "min  ", "max  ", "tip  ",
        "base |B|", "F_tip", "Tau_tip", "M_force", "M_total", "|F_tip|",
        "|Tau_tip|", "|M_total|",
    )

    def __init__(self, real: Any) -> None:
        self._real = real
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not line.lstrip().startswith(self._DROP):
                self._real.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        self._real.flush()


def _make_output_dir(cfg: StateStreamConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _row(estimate: Any, tick: int, age_s: float) -> dict[str, Any]:
    joints = (
        None if estimate.robot_joints is None
        else [round(float(v), 6) for v in np.asarray(estimate.robot_joints).reshape(-1)]
    )
    insertion = _finite_or_none(estimate.insertion_length_m)
    z = None
    if joints is not None and insertion is not None:
        z = joints + [insertion]
    return {
        "tick": tick,
        "t_monotonic": now_monotonic(),
        "state_age_s": round(float(age_s), 4),
        "frame_index": int(estimate.frame_index),
        "tip_position_m": [round(float(v), 6) for v in estimate.tip_position_m],
        "tip_tangent": [round(float(v), 6) for v in estimate.tip_tangent],
        "robot_joints": joints,
        "robot_pose6": (
            None if estimate.robot_pose6 is None
            else [round(float(v), 6) for v in np.asarray(estimate.robot_pose6).reshape(-1)]
        ),
        "insertion_length_m": insertion,
        "z_measured_state": z,   # [q1..q6, insertion] for the joint-space controllers
    }


# =============================================================================
# main
# =============================================================================


def main() -> None:
    import sys

    cfg = CONFIG
    if cfg.stream_hz <= 0.0:
        raise ValueError("stream_hz must be positive")

    real_stdout = sys.stdout
    sys.stdout = _DebugLineFilter(real_stdout)

    output_dir = _make_output_dir(cfg)
    log_path = output_dir / "state_stream.jsonl"
    print(f"[state] target consumer rate = {cfg.stream_hz:.1f} Hz")
    print(f"[state] log -> {log_path}")

    mapper = NewFrameTipMapper(cfg)
    print(
        f"[state] T_R_B columns  B.x/B.y/B.z = "
        f"{np.round(mapper.T_R_B.rotation[:, 0], 3).tolist()} / "
        f"{np.round(mapper.T_R_B.rotation[:, 1], 3).tolist()} / "
        f"{np.round(mapper.T_R_B.rotation[:, 2], 3).tolist()}"
    )

    robot = RobotJointStream(cfg.robot_ip, poll_hz=cfg.robot_poll_hz)
    print(f"[state] persistent URRtde session open @ {cfg.robot_ip}")

    advancer: Optional[AdvancerSink] = None
    if cfg.connect_advancer:
        advancer = AdvancerSink(
            AdvancerSinkConfig(port=cfg.advancer_port, dry_run=cfg.advancer_dry_run)
        )
    insertion = InsertionEstimate(cfg.initial_insertion_m, advancer)

    camera_config = CameraConfig(
        cam_index=cfg.cam_index,
        exposure=cfg.exposure,
        gain=cfg.gain,
        warmup_frames=cfg.warmup_frames,
        grab_period_s=cfg.grab_period_s,
        reconstruct_period_s=cfg.reconstruct_period_s,
        image_filename=cfg.image_filename,
        roi_polygon_path=cfg.roi_polygon_path,
        manual_boundary_path=cfg.manual_boundary_path,
        pivot_hint=tuple(cfg.pivot_hint_px),
        show_vision=False,
    )
    camera = CameraSource(
        camera_config,
        pivot_point_pose6=np.asarray(cfg.T_robot_beam_pose6, dtype=float),
        robot_pose_getter=lambda: robot.latest_pose(cfg.robot_max_age_s),
        robot_joints_getter=lambda: robot.latest_joints(cfg.robot_max_age_s),
        insertion_length_getter=insertion.get,
        frame_processor=mapper,
    )

    stop_requested = {"flag": False}
    previous_handler = signal.signal(
        signal.SIGINT, lambda *_: stop_requested.__setitem__("flag", True)
    )

    period_s = 1.0 / cfg.stream_hz
    ticks = stale_ticks = new_frame_ticks = no_joint_ticks = 0
    last_frame_index = -1
    consumer_started = 0.0
    log_file = log_path.open("w", encoding="utf-8")

    try:
        robot.start()
        if advancer is not None:
            advancer.start()
        camera.start()
        print("[state] streams started; waiting for first tip detection...")

        deadline = now_monotonic() + cfg.warmup_timeout_s
        first = None
        while now_monotonic() < deadline and not stop_requested["flag"]:
            first, _age = camera.latest()
            if first is not None:
                break
            if not camera.healthy:
                raise RuntimeError(
                    f"camera unhealthy during warmup: {camera.last_error!r}"
                )
            time.sleep(0.05)
        if first is None:
            raise RuntimeError(
                "no tip estimate within warmup_timeout_s; "
                f"detection_failures={mapper.detection_failures}, "
                f"last_error={mapper.last_error!r}"
            )
        print("[state] streaming")

        consumer_started = now_monotonic()
        next_tick = consumer_started
        while not stop_requested["flag"]:
            now = now_monotonic()
            if cfg.duration_s is not None and (now - consumer_started) >= cfg.duration_s:
                break
            if next_tick > now:
                time.sleep(min(next_tick - now, period_s))
                continue
            next_tick = max(next_tick + period_s, now)

            if not camera.healthy:
                print(f"[state] ABORT camera unhealthy: {camera.last_error!r}")
                break
            if not robot.healthy:
                print(f"[state] ABORT robot poll unhealthy: {robot.last_error!r}")
                break

            estimate, age = camera.latest(cfg.max_state_age_s)
            ticks += 1
            if estimate is None:
                stale_ticks += 1
                if ticks % cfg.print_every == 0:
                    print(f"[{ticks:5d}] STALE vision age={age:.3f}s")
                continue
            if estimate.robot_joints is None:
                no_joint_ticks += 1

            if estimate.frame_index != last_frame_index:
                new_frame_ticks += 1
                last_frame_index = estimate.frame_index

            row = _row(estimate, ticks, age)
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

            if ticks % cfg.print_every == 0:
                tip_mm = 1e3 * np.asarray(estimate.tip_position_m, dtype=float)
                elapsed = now - consumer_started
                q = estimate.robot_joints
                q_txt = (
                    "joints=--"
                    if q is None
                    else "q_deg=[" + " ".join(f"{np.degrees(v):+6.1f}" for v in q) + "]"
                )
                print(
                    f"[{ticks:5d}] frame={estimate.frame_index:5d} age={age:.3f}s "
                    f"tip_mm=[{tip_mm[0]:+7.1f} {tip_mm[1]:+7.1f} {tip_mm[2]:+7.1f}] "
                    f"ins={1e3 * estimate.insertion_length_m:5.1f}mm  {q_txt} "
                    f"| vis {mapper.detections / max(elapsed, 1e-6):4.1f}Hz "
                    f"robot {robot.reads / max(elapsed, 1e-6):5.0f}Hz "
                    f"visfail={mapper.detection_failures}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        camera.stop()
        if advancer is not None:
            advancer.stop()
        robot.stop()
        log_file.close()
        sys.stdout = real_stdout

    elapsed = max(now_monotonic() - consumer_started, 1e-6)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "elapsed_s": round(elapsed, 3),
        "consumer_ticks": ticks,
        "consumer_rate_hz": round(ticks / elapsed, 2),
        "target_rate_hz": cfg.stream_hz,
        "stale_vision_ticks": stale_ticks,
        "ticks_without_joints": no_joint_ticks,
        "unique_frames_consumed": new_frame_ticks,
        "vision_detections": mapper.detections,
        "vision_detection_failures": mapper.detection_failures,
        "vision_fps": round(mapper.detections / elapsed, 2),
        "robot_reads": robot.reads,
        "robot_read_failures": robot.read_failures,
        "robot_poll_fps": round(robot.reads / elapsed, 1),
        "camera_frames_grabbed": camera.frames_grabbed,
        "camera_grab_fps": round(camera.frames_grabbed / elapsed, 1),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== STATE STREAM SUMMARY ===")
    for key, value in summary.items():
        if key == "config":
            continue
        print(f"  {key:26s}: {value}")
    print(f"  log                       : {log_path}")


if __name__ == "__main__":
    main()
