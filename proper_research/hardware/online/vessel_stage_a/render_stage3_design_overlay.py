"""Render a SINGLE camera frame showing a Stage-3 design
(design_stage3_tracked_path.py's output) overlaid on the real vessel,
for review before committing to the offline-configuration run.

Reuses live_vessel_alignment_overlay.py's calibration/projection/drawing
helpers (same T_R_B + PlanarPixelCalibration chain the live vision
pipeline actually uses), but captures ONE frame and saves a PNG instead
of opening a continuous live window -- this is a background job with no
display, and we only need a single reviewable snapshot.

Draws:
  - cyan = full real vessel centreline, green = left wall, red = right
    wall, magenta = radius ticks (live_vessel_alignment_overlay.py's own
    convention)
  - orange = the designed tracked path (tip0 -> N mm short of the far
    end, shifted by the design's own left_shift_mm toward the left/green
    wall)
  - a filled yellow circle at the model-predicted initial tip position

Usage: python render_stage3_design_overlay.py [design_json_path]
  (defaults to the most recently generated phi40_L30 design)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from proper_research.hardware.online.vessel_stage_a.live_vessel_alignment_overlay import (
    build_calibration, draw_overlay, open_camera, project_lumen_to_pixels,
)
from proper_research.rig_calibration import CURRENT_LUMEN_FILE

DESIGN_DIR = Path(__file__).resolve().parents[4] / "plans" / "stage3_design"
if len(sys.argv) > 1:
    DESIGN_PATH = Path(sys.argv[1])
else:
    candidates = sorted(DESIGN_DIR.glob("*_design.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"no *_design.json found in {DESIGN_DIR}; run design_stage3_tracked_path.py first")
    DESIGN_PATH = candidates[-1]
OUT_PNG = DESIGN_PATH.with_name(DESIGN_PATH.stem.replace("_design", "_overlay") + ".png")


def points_to_pixels(points_R, T_R_B, calibration):
    T_B_R = T_R_B.inverse()
    points_B = T_B_R.apply_points(np.asarray(points_R, dtype=float))
    return calibration.beam_to_pixels(points_B)


def main():
    with open(DESIGN_PATH) as f:
        design = json.load(f)

    print(f"[render] building calibration...")
    T_R_B, calibration, _scfg = build_calibration(0.0)  # z_raise_mm is a no-op here, see module note

    print(f"[render] projecting real vessel lumen '{CURRENT_LUMEN_FILE}'...")
    center_px, left_px, right_px, provenance = project_lumen_to_pixels(CURRENT_LUMEN_FILE, T_R_B, calibration)

    tracked_path_px = points_to_pixels(design["tracked_path_R"], T_R_B, calibration)
    tip0_px = points_to_pixels([design["tip0"]], T_R_B, calibration)[0]

    print(f"[render] opening camera for a single frame...")
    cap = open_camera(0, 29.0)
    try:
        ok, frame = None, None
        for _ in range(10):
            ok, frame = cap.read()
            if ok:
                break
        if not ok:
            raise RuntimeError("could not read a camera frame")
    finally:
        cap.release()

    vis = draw_overlay(frame, center_px, left_px, right_px)

    pts = np.round(tracked_path_px).astype(np.int32)
    for i in range(len(pts) - 1):
        cv2.line(vis, tuple(pts[i]), tuple(pts[i + 1]), (0, 140, 255), 3)  # orange, BGR

    tip_xy = tuple(np.round(tip0_px).astype(int))
    cv2.circle(vis, tip_xy, 7, (0, 255, 255), -1)  # filled yellow
    cv2.putText(vis, "tip0", (tip_xy[0] + 10, tip_xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"orange=tracked path (phi={design['phi_deg']}deg, L0={design['L0_mm']}mm, "
                      f"{design['left_shift_mm']}mm left of centre, {design['end_margin_mm']}mm end margin)",
                (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_PNG), vis)
    print(f"[render] saved overlay to {OUT_PNG}")

    print(f"\nmagnet xyz0 (R)    = {design['magnet_xyz0']}")
    print(f"magnet rotvec0 (R) = {design['magnet_rotvec0']}")
    print(f"model-predicted tip0 (R) = {design['tip0']}")


if __name__ == "__main__":
    main()
