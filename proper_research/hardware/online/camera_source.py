"""CameraSource: hold the camera open, reconstruct the beam in a thread.

The old loop did ``new_capture()`` (open device -> warm up -> grab -> write JPG
-> release) every MPC step, then read the JPG back off disk for the vision
pipeline.  That is ~0.5-1 s of dead time per step and forces the controller to
stop and wait.

``CameraSource`` instead:

* opens ``cv2.VideoCapture`` once and keeps it open for the whole run;
* runs a fast **grab thread** that keeps the driver buffer drained so a frame is
  always current;
* runs a **reconstruct thread** that takes the freshest frame, runs the existing
  ``reconstruct_beam_within_vessel`` + ``vision_result_to_x_meas_robot``
  pipeline, and publishes a :class:`StateEstimate` to a :class:`LatestSlot`;
* keeps the JPG round-trip only as a RAM-backed handoff: the vision pipeline is
  file-path based, so the frame is written to ``/dev/shm`` (tmpfs) and read
  straight back — no spinning disk, no per-step device open/close.

The controller never blocks on the camera; it reads ``source.state_slot`` and
lets the safety supervisor act on staleness.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .messages import HeartbeatLoop, LatestSlot, StateEstimate, now_monotonic

__all__ = ["CameraConfig", "CameraSource"]


@dataclass
class CameraConfig:
    """Everything CameraSource needs; mirrors the old ``new_capture`` defaults."""

    # --- capture device ------------------------------------------------
    cam_index: int = 0
    backend: Optional[int] = None  # default cv2.CAP_V4L2, resolved in CameraSource
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    warmup_frames: int = 15
    auto_exposure_manual: float = 1.0
    exposure: float = 18.0
    gain: float = 0.0
    brightness: Optional[float] = None
    gamma: Optional[float] = None
    grab_period_s: float = 0.004  # ~250 Hz drain; real rate is capped by the driver

    # --- vision pipeline (same knobs as HardwareRunConfig) ------------
    image_filename: str = "/dev/shm/proper_online_frame.png"
    roi_polygon_path: str = "/home/jack/Proper-Research/custom_area.json"
    blue_roi_path: str = "blue_roi_box.json"
    green_roi_path: str = "green_roi_box.json"
    manual_boundary_path: str = "/home/jack/Proper-Research/manual_vessel_boundaries.json"
    pivot_hint: Tuple[int, int] = (309, 330)
    show_vision: bool = False
    overlay_dir: Optional[str] = None

    # --- reconstruction cadence -------------------------------------
    reconstruct_period_s: float = 0.02  # loop as fast as vision allows
    align_tangent_with_lumen: bool = True


class CameraSource:
    """Held-open camera + background beam reconstruction.

    Parameters
    ----------
    config:
        Capture and vision configuration.
    pivot_point_pose6:
        The pivot pose ``[x, y, z, rx, ry, rz]`` passed to
        ``vision_result_to_x_meas_robot``.
    lumen_C_robot_m, lumen_R_robot_m:
        Optional lumen centreline / radius, attached to the vision result so the
        tangent can be aligned with the lumen (as the old runner did).
    robot_pose_getter:
        Optional ``() -> pose6`` callable (usually ``RobotSink.latest_tcp_pose``)
        so each estimate carries the concurrent robot pose.
    insertion_length_getter:
        Optional ``() -> metres`` callable (usually from the insertion estimator)
        so each estimate carries the current beam length.
    robot_joints_getter:
        Optional ``() -> [q1..q6]`` callable (usually ``URRTDERobot.get_joints``)
        so each estimate carries the concurrent UR joint angles.  The joint-space
        offline controllers need these to build ``z = [q1..q6, insertion]``.
    """

    def __init__(
        self,
        config: CameraConfig,
        *,
        pivot_point_pose6,
        lumen_C_robot_m=None,
        lumen_R_robot_m=None,
        robot_pose_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        insertion_length_getter: Optional[Callable[[], float]] = None,
        robot_joints_getter: Optional[Callable[[], Optional[np.ndarray]]] = None,
        frame_processor: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self.config = config
        self.pivot_point_pose6 = np.asarray(pivot_point_pose6, dtype=float).reshape(6)
        # Optional full override of the vision middle: given the raw BGR frame,
        # return the x_meas vector ([tip_xyz(3), tangent_xyz(3), ...]) or None to
        # skip the frame.  When set, the default reconstruct_beam_within_vessel +
        # vision_result_to_x_meas_robot path (and the /dev/shm JPG round-trip) is
        # bypassed -- used for the fast tip-only new-frame pipeline.
        self._frame_processor = frame_processor
        self.lumen_C_robot_m = (
            None if lumen_C_robot_m is None else np.asarray(lumen_C_robot_m, dtype=float)
        )
        self.lumen_R_robot_m = (
            None if lumen_R_robot_m is None else np.asarray(lumen_R_robot_m, dtype=float)
        )
        self._robot_pose_getter = robot_pose_getter
        self._insertion_length_getter = insertion_length_getter
        self._robot_joints_getter = robot_joints_getter
        self._on_error = on_error

        self.state_slot: LatestSlot[StateEstimate] = LatestSlot()

        self._cap = None
        self._grab_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_stamp: float = 0.0
        self._processed_frame_stamp: float = 0.0

        self._grab_loop: Optional[HeartbeatLoop] = None
        self._recon_loop: Optional[HeartbeatLoop] = None

        # loaded lazily in start()
        self._roi_polygon = None
        self._manual = None
        self._reconstruct_fn = None
        self._to_x_meas_fn = None

        self.frames_grabbed = 0
        self.frames_reconstructed = 0
        self.reconstruct_failures = 0
        self.last_error: str = ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        import cv2

        cfg = self.config
        backend = cfg.backend if cfg.backend is not None else cv2.CAP_V4L2

        # Resolve the vision pipeline lazily so importing this module stays cheap
        # and machines without the calibration files can still import it.
        from proper_research.vision.bounds_beam import (
            load_polygon,
            reconstruct_beam_within_vessel,
        )
        from proper_research.vision.detect_blue import (
            load_manual_vessel_boundaries_with_frame,
        )
        from proper_research.control.lab_ready_mpc import vision_result_to_x_meas_robot

        self._reconstruct_fn = reconstruct_beam_within_vessel
        self._to_x_meas_fn = vision_result_to_x_meas_robot
        self._roi_polygon = load_polygon(cfg.roi_polygon_path)
        self._manual = load_manual_vessel_boundaries_with_frame(cfg.manual_boundary_path)

        if cfg.overlay_dir:
            os.makedirs(cfg.overlay_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.image_filename) or ".", exist_ok=True)

        cap = cv2.VideoCapture(cfg.cam_index, backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {cfg.cam_index} with backend {backend}"
            )
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(cfg.auto_exposure_manual))
        cap.set(cv2.CAP_PROP_GAIN, float(cfg.gain))
        cap.set(cv2.CAP_PROP_EXPOSURE, float(cfg.exposure))
        if cfg.frame_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.frame_width))
        if cfg.frame_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.frame_height))
        if cfg.brightness is not None:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, float(cfg.brightness))
        if cfg.gamma is not None:
            cap.set(cv2.CAP_PROP_GAMMA, float(cfg.gamma))
        # Keep the driver buffer shallow so grab() always yields a fresh frame.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        for _ in range(int(cfg.warmup_frames)):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError("Camera opened but the first frame read failed")
        with self._grab_lock:
            self._latest_frame = frame
            self._latest_frame_stamp = now_monotonic()
        self._cap = cap

        self._grab_loop = HeartbeatLoop(
            "camera-grab",
            self._grab_body,
            period_s=cfg.grab_period_s,
            on_error=self._on_error,
        )
        self._recon_loop = HeartbeatLoop(
            "camera-reconstruct",
            self._reconstruct_body,
            period_s=cfg.reconstruct_period_s,
            on_error=self._on_error,
        )
        self._grab_loop.start()
        self._recon_loop.start()

    def stop(self) -> None:
        for loop in (self._recon_loop, self._grab_loop):
            if loop is not None:
                loop.stop()
        self._recon_loop = None
        self._grab_loop = None
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def __enter__(self) -> "CameraSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def latest(self, max_age_s: Optional[float] = None) -> Tuple[Optional[StateEstimate], float]:
        return self.state_slot.get(max_age_s)

    @property
    def healthy(self) -> bool:
        loops = (self._grab_loop, self._recon_loop)
        return all(loop is not None and loop.alive and loop.error is None for loop in loops)

    # ------------------------------------------------------------------
    # threads
    # ------------------------------------------------------------------
    def _grab_body(self, _t_start: float) -> None:
        cap = self._cap
        if cap is None:
            return
        if not cap.grab():
            return
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            return
        with self._grab_lock:
            self._latest_frame = frame
            self._latest_frame_stamp = now_monotonic()
        self.frames_grabbed += 1

    def _reconstruct_body(self, t_start: float) -> None:
        import cv2

        with self._grab_lock:
            frame = self._latest_frame
            stamp = self._latest_frame_stamp
        if frame is None or stamp <= self._processed_frame_stamp:
            return  # nothing new since the last reconstruction

        cfg = self.config
        vision_result: Optional[dict] = None

        try:
            if self._frame_processor is not None:
                # Fast path: caller maps the raw frame straight to x_meas.
                raw = self._frame_processor(frame)
                if raw is None:
                    self.reconstruct_failures += 1
                    self.last_error = "frame_processor returned None"
                    self._processed_frame_stamp = stamp
                    return
                x_meas = np.asarray(raw, dtype=float).reshape(-1)
            else:
                # Default path: RAM-backed JPG handoff -> file-based pipeline.
                # Keep the image extension on the temp file -- cv2.imwrite picks
                # the encoder from the extension and has no writer for ".tmp".
                root, ext = os.path.splitext(cfg.image_filename)
                tmp_path = f"{root}.tmp{ext or '.png'}"
                if not cv2.imwrite(tmp_path, frame):
                    raise RuntimeError(f"cv2.imwrite failed for {tmp_path}")
                os.replace(tmp_path, cfg.image_filename)

                overlay_path = None
                if cfg.overlay_dir:
                    overlay_path = os.path.join(
                        cfg.overlay_dir, f"frame_{self.frames_reconstructed:06d}.png"
                    )

                vision_result = self._reconstruct_fn(
                    image_filename=cfg.image_filename,
                    red_roi_polygon=self._roi_polygon,
                    blue_roi_path=cfg.blue_roi_path,
                    green_roi_path=cfg.green_roi_path,
                    pivot_hint=tuple(cfg.pivot_hint),
                    show=bool(cfg.show_vision),
                    save_overlay_path=overlay_path,
                    base_px_ref=self._manual["base_px"],
                    ex_ref=self._manual["ex_img"],
                    ey_ref=self._manual["ey_img"],
                )
                vision_result["base_px_ref"] = self._manual["base_px"]
                vision_result["ex_ref"] = self._manual["ex_img"]
                vision_result["ey_ref"] = self._manual["ey_img"]
                if self.lumen_C_robot_m is not None:
                    vision_result["lumen_C_robot_m"] = self.lumen_C_robot_m
                if self.lumen_R_robot_m is not None:
                    vision_result["lumen_R_robot_m"] = self.lumen_R_robot_m

                x_meas = np.asarray(
                    self._to_x_meas_fn(
                        vision_result,
                        pivot_point_pose6=self.pivot_point_pose6,
                        align_tangent_with_lumen=bool(cfg.align_tangent_with_lumen),
                    ),
                    dtype=float,
                ).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the loop
            self.reconstruct_failures += 1
            self.last_error = repr(exc)
            self._processed_frame_stamp = stamp
            return

        if x_meas.size < 6 or not np.all(np.isfinite(x_meas[:6])):
            self.reconstruct_failures += 1
            self.last_error = f"vision returned non-finite / short x_meas: {x_meas!r}"
            self._processed_frame_stamp = stamp
            return

        robot_pose6 = None
        if self._robot_pose_getter is not None:
            try:
                pose = self._robot_pose_getter()
                if pose is not None:
                    robot_pose6 = np.asarray(pose, dtype=float).reshape(6)
            except Exception:
                robot_pose6 = None

        insertion_m = float("nan")
        if self._insertion_length_getter is not None:
            try:
                insertion_m = float(self._insertion_length_getter())
            except Exception:
                insertion_m = float("nan")

        robot_joints = None
        if self._robot_joints_getter is not None:
            try:
                joints = self._robot_joints_getter()
                if joints is not None:
                    robot_joints = np.asarray(joints, dtype=float).reshape(6)
            except Exception:
                robot_joints = None

        beam_len_mm = float("nan")
        if vision_result is not None:
            try:
                raw = vision_result.get("beam_length_mm")
                if raw is not None:
                    beam_len_mm = float(raw)
            except Exception:
                pass
        elif x_meas.size > 6 and np.isfinite(x_meas[6]):
            beam_len_mm = 1.0e3 * float(x_meas[6])  # fast path may append length (m)

        estimate = StateEstimate(
            t_monotonic=stamp,
            x_meas=x_meas,
            tip_position_m=x_meas[:3].copy(),
            tip_tangent=x_meas[3:6].copy(),
            insertion_length_m=insertion_m,
            robot_pose6=robot_pose6,
            robot_joints=robot_joints,
            source="vision",
            frame_index=self.frames_reconstructed,
            valid=True,
            vision_beam_length_mm=beam_len_mm,
        )
        self.state_slot.set(estimate)
        self._processed_frame_stamp = stamp
        self.frames_reconstructed += 1
