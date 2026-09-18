#!/usr/bin/env python3
"""One-shot camera measurement of the beam's CURRENT insertion length, for
seeding --l0-mm before a live acquire.py run (the advancer has no encoder,
so this is the only ground truth available without a physical ruler).

Uses the exact same production pipeline (NewFrameTipMapper + the new-frame
calibration in StateStreamConfig) that acquire.py/close_loop_path_follow.py
use for live control -- not the older `reconstruct_beam_within_vessel`
default path, which depends on files (blue_roi_box.json/green_roi_box.json)
that don't exist in this checkout and a superseded 38mm calibration
constant (see the "Final frame calibration" memory -- the correct constant
is state_stream.StateStreamConfig.known_calibration_distance_mm=80.6mm).

With `anchor_beam_origin_to_detected_base=True` (the default), the reported
tip position is already the detected-base-to-detected-tip vector, rotated
into the robot frame and offset by the nominal pivot translation. Its norm
is the straight-line (chord) distance from the base marker to the tip
marker -- a good approximation of insertion length for a lightly-bent beam,
not exact arc length for a strongly-curved one.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.measure_l0 --n 20
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.messages import now_monotonic
from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig

PIVOT_XYZ = np.array([0.525575, -0.670028, -0.016567])  # StateStreamConfig.T_robot_beam_pose6[:3]


def build_measurement_camera() -> tuple[CameraSource, NewFrameTipMapper]:
    """The same production tip-only pipeline acquire.py uses, standalone."""
    scfg = StateStreamConfig(exposure=29.0, marker_min_count=2, reconstruct_period_s=0.02)
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=scfg.exposure, warmup_frames=scfg.warmup_frames,
            grab_period_s=scfg.grab_period_s, reconstruct_period_s=scfg.reconstruct_period_s,
            image_filename="/dev/shm/measure_l0_frame.png",
            roi_polygon_path=scfg.roi_polygon_path,
            manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=None, robot_pose_getter=None,
        insertion_length_getter=lambda: float("nan"),
        frame_processor=mapper,
    )
    return camera, mapper


def measure_l_mm(camera: CameraSource, mapper: NewFrameTipMapper, *, n: int = 20, timeout_s: float = 20.0) -> np.ndarray:
    """n fresh vision-measured insertion lengths (mm) from an already-started camera."""
    L_mm = []
    deadline = now_monotonic() + timeout_s
    last_seq = -1
    while len(L_mm) < n and now_monotonic() < deadline:
        est, seq = camera.state_slot.peek()
        if est is not None and seq != last_seq:
            last_seq = seq
            tip = np.asarray(est.tip_position_m, dtype=float)
            L_mm.append(float(np.linalg.norm(tip - PIVOT_XYZ)) * 1e3)
        time.sleep(0.01)
    if not L_mm:
        raise RuntimeError(f"no tip detections within {timeout_s}s "
                            f"(detection_failures={mapper.detection_failures}, "
                            f"last_error={mapper.last_error!r})")
    return np.array(L_mm)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20, help="number of valid estimates to average")
    p.add_argument("--timeout-s", type=float, default=20.0)
    args = p.parse_args()

    camera, mapper = build_measurement_camera()
    camera.start()
    try:
        print("[measure_l0] camera started, collecting estimates...")
        L_mm = measure_l_mm(camera, mapper, n=args.n, timeout_s=args.timeout_s)
        print(f"\n[measure_l0] n={L_mm.size} samples: "
              f"mean={L_mm.mean():.3f}mm std={L_mm.std():.3f}mm "
              f"min={L_mm.min():.3f}mm max={L_mm.max():.3f}mm")
        print(f"[measure_l0] --l0-mm {L_mm.mean():.2f}")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
