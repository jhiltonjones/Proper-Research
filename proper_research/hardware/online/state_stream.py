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
    exposure: float = 27.0
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
    marker_sat_min: int = 50
    marker_val_min: int = 40
    marker_red_hue1_high: int = 10
    marker_red_hue2_low: int = 170
    emit_chord_tangent: bool = True        # cheap tangent_start->tip chord (not the true tangent)

    # --- NEW beam frame B (matches robotics_frame_measurement_validation) --
    known_calibration_distance_mm: float = 38.0
    saved_axis_signs_for_positive_beam_xy: tuple[float, float] = (-1.0, 1.0)
    saved_axis_convention: str = "image_cartesian"
    beam_axial_axis_R: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    beam_plane_normal_axis_R: tuple[float, float, float] = (0.0, 0.0, -1.0)
    T_robot_beam_pose6: tuple[float, float, float, float, float, float] = (
        0.525575, -0.670028, -0.016567, 0.0, -1.5707963, 0.0,
    )
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
        self, ip: str, *, poll_hz: float = 125.0, receive_only: bool = True
    ) -> None:
        self._receive_only = bool(receive_only)
        self._robot = None
        self._rtde_r = None
        if self._receive_only:
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

    def start(self) -> None:
        self._loop = HeartbeatLoop(
            "robot-poll", self._poll_body, period_s=1.0 / self._poll_hz
        )
        self._loop.start()

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
        if joints is not None:
            self.joints_slot.set(np.asarray(joints, dtype=float).reshape(6))
        if pose is not None:
            self.pose_slot.set(np.asarray(pose, dtype=float).reshape(6))
        self.reads += 1

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
            min_markers=3,
            max_markers=4,
        )
        if candidates is None or len(candidates) not in (3, 4):
            raise RuntimeError(f"expected 3-4 red markers, got {candidates}")
        base_c, _mag_c, tangent_start_c, tip_c = bb.order_beam_marker_candidates(
            candidates, pivot_hint=tuple(cfg.pivot_hint_px)
        )
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
