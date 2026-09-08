"""Part 1 of the real-time controller: stream the measured beam state.

This runs the vision half of the online stack on its own -- no robot motion,
no advancer -- so the streaming measured state can be checked before any
controller is wired in.

    CameraSource ── StateEstimate ──▶ (this script's consumer loop @ stream_hz)
       held-open camera,                  prints + JSONL logs the state,
       grab thread (~250 Hz),             tracks the true reconstruction rate
       reconstruct thread (vision
       pipeline -> x_meas)

``CameraSource`` already owns the held-open ``cv2.VideoCapture``, the fast grab
thread and the background reconstruction thread that calls the existing
``reconstruct_beam_within_vessel`` + ``vision_result_to_x_meas_robot`` pipeline
and publishes a :class:`StateEstimate`.  All this file adds is:

* a fixed-rate **consumer loop** -- what ``OnlineMPCRunner._control_body`` will
  be, minus the controller: read ``camera.latest(max_age)`` every tick;
* rate bookkeeping -- grab FPS, reconstruction FPS, how often the consumer
  actually sees a *new* frame, and how many ticks got a stale / missing state;
* a JSONL log of every consumed estimate (``tail -f`` friendly).

No robot is required.  With ``connect_robot_receive_only = True`` a passive
``rtde_receive`` interface is opened (no control script upload, no motion) so
each estimate also carries the concurrent TCP pose and joint angles -- which
the joint-space Jacobian / MPC controllers need in part 2.

There are no command-line arguments; edit the configuration block below.

    python -m proper_research.hardware.online.stream_vision
"""

from __future__ import annotations

import contextlib
import io
import json
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.messages import now_monotonic


# =============================================================================
# USER CONFIGURATION -- edit here; there are no terminal arguments
# =============================================================================


@dataclass
class StreamVisionConfig:
    # --- consumer loop --------------------------------------------------
    stream_hz: float = 20.0            # rate the measured state is consumed at
    duration_s: Optional[float] = 30.0  # None -> run until Ctrl+C
    max_state_age_s: float = 0.40     # older than this -> counted as stale
    warmup_timeout_s: float = 20.0    # wait this long for the first reconstruction
    print_every: int = 10             # print one status line every N ticks

    # --- camera + vision pipeline ------------------------------------
    cam_index: int = 0
    exposure: float = 18.0
    gain: float = 0.0
    warmup_frames: int = 15
    grab_period_s: float = 0.004       # ~250 Hz driver drain
    reconstruct_period_s: float = 0.03  # lower bound; real rate is vision-limited

    image_filename: str = "/dev/shm/proper_online_frame.png"
    roi_polygon_path: str = "/home/jack/Proper-Research/custom_area.json"
    manual_boundary_path: str = (
        "/home/jack/Proper-Research/manual_vessel_boundaries.json"
    )
    blue_roi_path: str = "/home/jack/Proper-Research/blue_roi_box.json"
    green_roi_path: str = "/home/jack/Proper-Research/green_roi_box.json"
    pivot_hint_px: tuple[int, int] = (318, 292)
    align_tangent_with_lumen: bool = True

    # Pivot pose6 [x, y, z, rx, ry, rz] handed to vision_result_to_x_meas_robot
    # to map the measured tip from the beam-local frame into the robot base
    # frame.  This is the OLD (pivot_local_point_to_robot) convention that the
    # existing online controllers expect -- keep it consistent with whatever
    # the Jacobian provider is built against in part 2.  Defaults to the v2
    # experiment pivot.
    pivot_pose6: Optional[tuple[float, float, float, float, float, float]] = None

    # Save one vision overlay PNG per reconstructed frame here (slower).  None
    # disables overlay writing.
    overlay_dir: Optional[str] = None

    # --- optional passive robot read -------------------------------
    connect_robot_receive_only: bool = True
    robot_ip: str = "192.168.56.101"

    # --- output ------------------------------------------------------
    output_root: str = "vision_stream_logs"
    run_name: str = "stream_vision"

    # Silence the per-frame matplotlib debug plot inside
    # reconstruct_beam_within_vessel (it writes a PNG every call and dominates
    # the reconstruction time).  Leave True for real streaming.
    suppress_vision_debug_plot: bool = True


CONFIG = StreamVisionConfig()


# =============================================================================
# passive robot read (no control script, no motion)
# =============================================================================


class _ReceiveOnlyRobot:
    """Thin wrapper over ``rtde_receive`` for pose/joint getters only."""

    def __init__(self, ip: str, frequency: float = 125.0) -> None:
        import rtde_receive

        self._r = rtde_receive.RTDEReceiveInterface(ip, frequency)
        if not self._r.isConnected():
            raise RuntimeError(f"rtde_receive could not connect to {ip}")

    def get_tcp_pose(self) -> Optional[list[float]]:
        try:
            return list(self._r.getActualTCPPose())
        except Exception:
            return None

    def get_joints(self) -> Optional[list[float]]:
        try:
            return list(self._r.getActualQ())
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._r.disconnect()
        except Exception:
            pass


# =============================================================================
# helpers
# =============================================================================


def _resolve_pivot_pose6(cfg: StreamVisionConfig) -> np.ndarray:
    if cfg.pivot_pose6 is not None:
        return np.asarray(cfg.pivot_pose6, dtype=float).reshape(6)
    import proper_research.hardware.beam_hardware_experiment_v2 as experiment_v2

    return np.asarray(experiment_v2.CONFIG.pivot_pose6, dtype=float).reshape(6)


def _quiet_vision_pipeline(cfg: StreamVisionConfig) -> None:
    """Silence the per-frame matplotlib debug plot and the pipeline's prints.

    ``reconstruct_beam_within_vessel`` writes a matplotlib PNG and prints dozens
    of ``[DEBUG]`` / ``[INFO]`` lines on every call.  Both dominate the loop and
    drown the stream output, so patch them out before ``CameraSource.start``
    binds ``reconstruct_beam_within_vessel``.
    """

    if not cfg.suppress_vision_debug_plot:
        return
    from proper_research.vision import bounds_beam

    bounds_beam.plot_contact_lumen_debug = lambda *a, **k: None  # type: ignore[assignment]

    _original = bounds_beam.reconstruct_beam_within_vessel

    def _quiet(*args: Any, **kwargs: Any) -> Any:
        with contextlib.redirect_stdout(io.StringIO()):
            return _original(*args, **kwargs)

    bounds_beam.reconstruct_beam_within_vessel = _quiet  # type: ignore[assignment]


def _build_camera(cfg: StreamVisionConfig, pivot_pose6: np.ndarray) -> CameraSource:
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
        blue_roi_path=cfg.blue_roi_path,
        green_roi_path=cfg.green_roi_path,
        pivot_hint=tuple(cfg.pivot_hint_px),
        show_vision=False,
        overlay_dir=cfg.overlay_dir,
        align_tangent_with_lumen=cfg.align_tangent_with_lumen,
    )
    robot = None
    pose_getter = None
    joints_getter = None
    if cfg.connect_robot_receive_only:
        robot = _ReceiveOnlyRobot(cfg.robot_ip)
        pose_getter = robot.get_tcp_pose
        joints_getter = robot.get_joints
        print(f"[stream] passive rtde_receive connected to {cfg.robot_ip}")

    source = CameraSource(
        camera_config,
        pivot_point_pose6=pivot_pose6,
        robot_pose_getter=pose_getter,
        robot_joints_getter=joints_getter,
    )
    source._receive_only_robot = robot  # keep a handle for cleanup
    return source


def _make_output_dir(cfg: StreamVisionConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _estimate_row(estimate: Any, tick: int, age_s: float) -> dict[str, Any]:
    return {
        "tick": tick,
        "t_monotonic": now_monotonic(),
        "state_age_s": round(float(age_s), 4),
        "frame_index": int(estimate.frame_index),
        "tip_position_m": [round(float(v), 6) for v in estimate.tip_position_m],
        "tip_tangent": [round(float(v), 6) for v in estimate.tip_tangent],
        "x_meas": [round(float(v), 6) for v in np.asarray(estimate.x_meas).reshape(-1)],
        "insertion_length_m": _finite_or_none(estimate.insertion_length_m),
        "vision_beam_length_mm": _finite_or_none(estimate.vision_beam_length_mm),
        "robot_pose6": (
            None if estimate.robot_pose6 is None
            else [round(float(v), 6) for v in np.asarray(estimate.robot_pose6).reshape(-1)]
        ),
        "robot_joints": (
            None if estimate.robot_joints is None
            else [round(float(v), 6) for v in np.asarray(estimate.robot_joints).reshape(-1)]
        ),
    }


# =============================================================================
# main streaming loop
# =============================================================================


def main() -> None:
    cfg = CONFIG
    if cfg.stream_hz <= 0.0:
        raise ValueError("stream_hz must be positive")

    _quiet_vision_pipeline(cfg)
    pivot_pose6 = _resolve_pivot_pose6(cfg)
    output_dir = _make_output_dir(cfg)
    log_path = output_dir / "stream.jsonl"

    print(f"[stream] pivot_pose6 = {np.round(pivot_pose6, 5).tolist()}")
    print(f"[stream] target consumer rate = {cfg.stream_hz:.1f} Hz")
    print(f"[stream] log -> {log_path}")

    source = _build_camera(cfg, pivot_pose6)

    stop_requested = {"flag": False}

    def _handle_sigint(_signum: int, _frame: Any) -> None:
        stop_requested["flag"] = True

    previous_handler = signal.signal(signal.SIGINT, _handle_sigint)

    period_s = 1.0 / cfg.stream_hz
    ticks = 0
    stale_ticks = 0
    new_frame_ticks = 0
    last_frame_index = -1
    consumer_started = 0.0
    tip_history: list[np.ndarray] = []

    log_file = log_path.open("w", encoding="utf-8")
    try:
        source.start()
        print("[stream] camera started; waiting for first reconstruction...")

        warmup_deadline = now_monotonic() + cfg.warmup_timeout_s
        first_estimate = None
        while now_monotonic() < warmup_deadline and not stop_requested["flag"]:
            first_estimate, _age = source.latest()
            if first_estimate is not None:
                break
            if not source.healthy:
                raise RuntimeError(
                    f"camera unhealthy during warmup: {source.last_error or 'thread down'}"
                )
            time.sleep(0.05)
        if first_estimate is None:
            raise RuntimeError(
                "no StateEstimate produced within warmup_timeout_s "
                f"({cfg.warmup_timeout_s}s); reconstruct_failures="
                f"{source.reconstruct_failures}, last_error={source.last_error!r}"
            )
        print(
            f"[stream] first frame after "
            f"{cfg.warmup_timeout_s - (warmup_deadline - now_monotonic()):.2f}s; streaming"
        )

        consumer_started = now_monotonic()
        next_tick = consumer_started
        while not stop_requested["flag"]:
            now = now_monotonic()
            if cfg.duration_s is not None and (now - consumer_started) >= cfg.duration_s:
                break
            if next_tick > now:
                time.sleep(min(next_tick - now, period_s))
                continue
            next_tick += period_s
            if next_tick < now:  # fell behind; resync
                next_tick = now + period_s

            if not source.healthy:
                print(
                    f"[stream] ABORT camera unhealthy: "
                    f"{source.last_error or 'thread down'}"
                )
                break

            estimate, age = source.latest(cfg.max_state_age_s)
            ticks += 1
            if estimate is None:
                stale_ticks += 1
                if ticks % cfg.print_every == 0:
                    print(
                        f"[{ticks:5d}] STALE  age={age:.3f}s  "
                        f"recon={source.frames_reconstructed} "
                        f"fail={source.reconstruct_failures}"
                    )
                continue

            is_new = estimate.frame_index != last_frame_index
            if is_new:
                new_frame_ticks += 1
                last_frame_index = estimate.frame_index
                tip_history.append(np.asarray(estimate.tip_position_m, dtype=float))

            row = _estimate_row(estimate, ticks, age)
            row["new_frame"] = bool(is_new)
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

            if ticks % cfg.print_every == 0:
                tip_mm = 1.0e3 * np.asarray(estimate.tip_position_m, dtype=float)
                elapsed = now - consumer_started
                recon_fps = source.frames_reconstructed / max(elapsed, 1e-6)
                grab_fps = source.frames_grabbed / max(elapsed, 1e-6)
                print(
                    f"[{ticks:5d}] frame={estimate.frame_index:5d} age={age:.3f}s "
                    f"tip_mm=[{tip_mm[0]:+7.1f} {tip_mm[1]:+7.1f} {tip_mm[2]:+7.1f}] "
                    f"beam_len={estimate.vision_beam_length_mm:6.1f}mm "
                    f"| recon {recon_fps:4.1f}Hz grab {grab_fps:5.0f}Hz "
                    f"fail={source.reconstruct_failures}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        source.stop()
        robot = getattr(source, "_receive_only_robot", None)
        if robot is not None:
            robot.close()
        log_file.close()

    elapsed = max(now_monotonic() - consumer_started, 1e-6)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "pivot_pose6": pivot_pose6.tolist(),
        "elapsed_s": round(elapsed, 3),
        "consumer_ticks": ticks,
        "consumer_rate_hz": round(ticks / elapsed, 2),
        "target_rate_hz": cfg.stream_hz,
        "stale_ticks": stale_ticks,
        "stale_fraction": round(stale_ticks / max(ticks, 1), 4),
        "unique_frames_consumed": new_frame_ticks,
        "frames_grabbed": source.frames_grabbed,
        "frames_reconstructed": source.frames_reconstructed,
        "reconstruct_failures": source.reconstruct_failures,
        "grab_fps": round(source.frames_grabbed / elapsed, 1),
        "reconstruct_fps": round(source.frames_reconstructed / elapsed, 2),
        "last_error": source.last_error,
    }
    if len(tip_history) >= 2:
        tips = np.vstack(tip_history)
        summary["tip_position_mean_mm"] = (1.0e3 * tips.mean(axis=0)).round(3).tolist()
        summary["tip_position_std_mm"] = (1.0e3 * tips.std(axis=0)).round(3).tolist()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== STREAM SUMMARY ===")
    for key in (
        "elapsed_s", "consumer_ticks", "consumer_rate_hz", "target_rate_hz",
        "stale_ticks", "stale_fraction", "unique_frames_consumed",
        "frames_grabbed", "grab_fps", "frames_reconstructed", "reconstruct_fps",
        "reconstruct_failures",
    ):
        print(f"  {key:24s}: {summary[key]}")
    if "tip_position_std_mm" in summary:
        print(f"  tip_position_mean_mm    : {summary['tip_position_mean_mm']}")
        print(f"  tip_position_std_mm     : {summary['tip_position_std_mm']}")
    print(f"  log                     : {log_path}")
    print(f"  summary                 : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
