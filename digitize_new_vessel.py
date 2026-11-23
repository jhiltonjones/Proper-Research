"""One-off wrapper: digitize a NEW vessel using the existing beam-frame
calibration, saving to a clearly-named, non-colliding output file (the
default VESSEL_LUMEN_FILE would silently overwrite vessel_lumen_robot_frame.
json, which already has prior-session history -- avoid that).

Usage: edit IMAGE_FILENAME and SAVE_PATH below, then run from
/home/jack/Proper-Research:
    python3 /path/to/this/digitize_new_vessel.py
"""
import sys
sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.vision.detect_blue import draw_vessel_lumen_for_planner

IMAGE_FILENAME = "focused_image.jpg"  # <-- change to your photo's filename
SAVE_PATH = "/home/jack/Proper-Research/vessel_lumen_robot_frame_NEW_2026-11-24.json"  # <-- rename as you like

result = draw_vessel_lumen_for_planner(
    image_filename=IMAGE_FILENAME,
    save_path=SAVE_PATH,
    blue_roi_path="blue_roi_box.json",  # optional; fine if this file doesn't exist
)
print(f"[done] saved to {SAVE_PATH}")
