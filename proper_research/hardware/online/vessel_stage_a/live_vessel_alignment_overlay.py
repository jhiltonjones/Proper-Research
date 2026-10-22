#!/usr/bin/env python3
"""Live camera overlay of the vessel lumen boundaries, for re-aligning a
physically-moved vessel phantom back to its calibrated position.

Projects the SAME lumen geometry the planner/controller use (robot-frame
centreline + radius from a `vessel_lumen_robot_frame_*.json` file) through
the SAME camera calibration the live vision pipeline uses
(`NewFrameTipMapper`'s `T_R_B` + `PlanarPixelCalibration`, i.e. the
project's real, validated pixel<->robot-frame mapping -- not a separate/
approximate one), and draws it live over the camera feed. Move the
physical vessel until its walls line up with the drawn boundary lines,
then re-run whatever camera health check you normally use before trusting
the alignment again.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.live_vessel_alignment_overlay \\
        --lumen-file vessel_lumen_robot_frame_zraise42_2026-09-29.json \\
        --z-raise-mm 42.044008750641574

--z-raise-mm must match whatever value the live closed-loop runs for this
setup use (the same number you pass to run_mpc_delay_aware_vessel.py's own
--z-raise-mm) -- it feeds the same T_robot_beam_pose6 pivot the live vision
pipeline uses, so a mismatch here would silently misplace the overlay even
if everything else lines up.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def build_calibration(z_raise_mm: float):
    from proper_research.hardware.online.vessel_stage_a import common
    from proper_research.hardware.online.state_stream import NewFrameTipMapper

    common.Z_RAISE_M = z_raise_mm / 1000.0
    scfg = common._raised_stream_stream_config()
    mapper = NewFrameTipMapper(scfg)
    return mapper.T_R_B, mapper.calibration, scfg


def project_lumen_to_pixels(lumen_file: str, T_R_B, calibration):
    from proper_research.vision.detect_blue import load_vessel_lumen_robot_frame

    C_R, R_m, provenance = load_vessel_lumen_robot_frame(lumen_file)
    C_R = np.asarray(C_R, dtype=float)
    R_m = np.asarray(R_m, dtype=float)

    T_B_R = T_R_B.inverse()
    C_B = T_B_R.apply_points(C_R)  # (N,3); B.z should be ~0 (planar calibration)

    # Local in-plane tangent (finite difference along the centreline) and
    # its +90deg in-plane normal -- the calibration is defined on the
    # B.z=0 plane, so a simple 2D rotation is the correct perpendicular,
    # no separate out-of-plane axis needed.
    tangent = np.gradient(C_B[:, :2], axis=0)
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.clip(tnorm, 1e-9, None)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    left_B = C_B.copy()
    right_B = C_B.copy()
    left_B[:, :2] = C_B[:, :2] + R_m[:, None] * normal
    right_B[:, :2] = C_B[:, :2] - R_m[:, None] * normal

    center_px = calibration.beam_to_pixels(C_B)
    left_px = calibration.beam_to_pixels(left_B)
    right_px = calibration.beam_to_pixels(right_B)
    return center_px, left_px, right_px, provenance


def open_camera(cam_index: int, exposure: float):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    # Same manual-exposure recipe as CameraSource (camera_source.py) --
    # matches what the live vision pipeline actually sees.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
    cap.set(cv2.CAP_PROP_GAIN, 0.0)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    return cap


def draw_overlay(frame_bgr, center_px, left_px, right_px):
    from proper_research.vision.detect_blue import draw_centerline_radius_overlay

    return draw_centerline_radius_overlay(
        image_bgr=frame_bgr,
        centerline_px=center_px,
        left_boundary_px=left_px,
        right_boundary_px=right_px,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True,
                    help="the vessel_lumen_robot_frame_*.json this plan/controller was built against")
    p.add_argument("--z-raise-mm", type=float, required=True,
                    help="must match the live controller's --z-raise-mm for this setup")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--exposure", type=float, default=29.0)
    args = p.parse_args()

    print(f"[align] building calibration (z_raise={args.z_raise_mm}mm)...")
    T_R_B, calibration, _scfg = build_calibration(args.z_raise_mm)

    print(f"[align] projecting lumen '{args.lumen_file}' into pixel space...")
    center_px, left_px, right_px, provenance = project_lumen_to_pixels(
        args.lumen_file, T_R_B, calibration,
    )
    print(f"[align] lumen provenance: {provenance}")
    print(f"[align] {len(center_px)} centreline samples projected")

    print(f"[align] opening camera index {args.camera_index}...")
    cap = open_camera(args.camera_index, args.exposure)

    window = "Vessel alignment overlay -- move vessel to match the lines (q/ESC to quit)"
    print("[align] cyan = centreline, green = left wall, red = right wall, magenta = radius ticks")
    print("[align] move the physical vessel until its walls line up with the drawn lines.")
    print("[align] press q or ESC when aligned.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[align] WARNING: could not read a frame, retrying...")
                continue
            vis = draw_overlay(frame, center_px, left_px, right_px)
            cv2.imshow(window, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("[align] done. Re-run your usual camera health check before trusting the alignment.")


if __name__ == "__main__":
    main()
