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
from proper_research.rig_calibration import BEAM_BASE_XYZ_M

# X/Y from the single canonical rig_calibration.py source; this file's own
# Z (-0.016567) predates and differs from the unraised-rig default there.
PIVOT_XYZ = np.array([BEAM_BASE_XYZ_M[0], BEAM_BASE_XYZ_M[1], -0.016567])  # StateStreamConfig.T_robot_beam_pose6[:3]

# 2026-11-17 empirical affine correction: true_mm = a*raw_mm + b.
#
# This is a STAND-IN for a proper 3D re-derivation of PIVOT_XYZ/BEAM_BASE_XYZ_M
# (which would need robot-arm motion to re-triangulate the pivot and is out of
# scope right now -- robot-arm motion was not authorized for this task). Three
# earlier single-point rescales of `known_distance_mm` (the 2D pixel scale)
# were each tried and failed to hold up across different physical lengths
# (see calibration_points.json's "source" field for that history), which is
# consistent with a genuine 3D pivot-location error rather than a 2D scale
# error -- a 2D rescale can't fix that, but an affine correction on the
# downstream *length* can, as long as it's along roughly the same axis as the
# physical motion that was used to calibrate it (which is the case here: the
# beam only moves by advancing/retracting along its own insertion axis).
#
# SUPERSEDED FIT (kept here for history, do not use): an earlier same-day
# attempt fit a=1.110557, b=-17.734810 from 3 points, two of which (true=20mm,
# true=21mm) turned out to not be a genuine function of raw_mm -- a repeat
# camera reading at an UNCHANGED physical position gave raw=47.394mm then,
# minutes later with nothing touched, raw=41.351mm. Investigation first
# suspected mechanical ring-down (wrong -- this advancer/beam system doesn't
# oscillate, confirmed by the person who built it) and then camera buffering
# or the advancer's move-complete signal (also ruled out directly: a
# continuous camera session sampled every ~100ms from the instant the
# advancer's firmware confirmed completion showed the reading already at its
# final value on the very first sample, flat for 8+ seconds after -- no lag,
# no buffering, no gradual convergence). The real cause: those earlier points
# were produced by CHAINING several open-loop advancer deltas off one
# physically-confirmed anchor. This advancer has no encoder (see
# `reset_insertion.py`'s own module docstring, which documents a prior
# +18.85mm of uncorrected drift over just 2 open-loop repeats) -- chaining
# deltas without re-measuring the physical length each time accumulates real
# positioning error that has nothing to do with vision. Confirmed directly: a
# nominal "same" length reached via different open-loop move chains read
# consistently (std < 0.06mm) but at DIFFERENT raw values (~44.3mm vs the
# fresh single-move value), and direction of approach made no difference
# (ruling out backlash too) -- it was purely chain-accumulated drift.
#
# CURRENT FIT, from four points each independently physically measured by the
# user with a ruler/calipers at the time of the camera reading (not chained
# from an earlier point), each camera reading itself confirmed stable across
# 3 independent camera sessions a few seconds apart before being accepted
# (session-to-session std listed below), under the CURRENT
# (reverted-to-2026-09-14) calibration_points.json/
# known_calibration_distance_mm=80.62257748298549:
#   (raw=41.366, true=35.0)  -- std=0.018mm across 3 sessions -- FIT
#   (raw=39.617, true=31.0)  -- std=0.015mm across 3 sessions -- FIT
#   (raw=50.006, true=40.0)  -- std=0.178mm across 3 sessions -- FIT
#   (raw=31.964, true=25.0)  -- std=0.003mm across 3 sessions -- HELD OUT
#                                (not used in the fit below)
# OLS fit (numpy.linalg.lstsq) on the 3 FIT points:
#   a = 0.774063, b = 1.535419
# In-sample residuals on the 3 fit points: -1.45mm, +1.20mm, +0.24mm.
# Held-out check (true=25.0mm, not used in the fit): predicted 26.28mm,
# residual +1.28mm -- within the requested 1-2mm tolerance on a genuinely
# independent point.
_AFFINE_CORRECTION_A = 0.774063
_AFFINE_CORRECTION_B = 1.535419


def apply_affine_insertion_correction(raw_mm: float) -> float:
    """Map a raw ||tip-PIVOT_XYZ|| chord-length reading (mm) to the empirical
    true-length estimate. See the fit derivation in the comment above."""
    return _AFFINE_CORRECTION_A * raw_mm + _AFFINE_CORRECTION_B


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
            raw_mm = float(np.linalg.norm(tip - PIVOT_XYZ)) * 1e3
            L_mm.append(apply_affine_insertion_correction(raw_mm))
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
