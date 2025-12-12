import csv
import time
import numpy as np

from proper_research.parameters import ROBOT_IP
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.vision.camera import detect_red_points_and_angle, new_capture
from proper_research.vision.measure_length import measure_beam_length_mm_with_checkerboard
from proper_research.advancer_unit.advancer_control import advancer_go


def safe_xy(pt):
    """Convert a point to (x, y) if available; otherwise (None, None)."""
    if pt is None:
        return None, None
    try:
        return float(pt[0]), float(pt[1])
    except Exception:
        return None, None


def safe_roi(roi):
    """Convert ROI box to 4-tuple if available; otherwise (None, None, None, None)."""
    if roi is None:
        return None, None, None, None
    try:
        # expected (x, y, w, h) or similar
        return tuple(roi[:4])
    except Exception:
        return None, None, None, None


OUTPUT_CSV = "beam_calibration_results.csv"

robo = URRtde(ROBOT_IP)

try:
    base_pose = get_point(0, 0)
    robo.moveL(base_pose)

    values = np.arange(-90, 100, 10)
    lengths = np.arange(25, 39, 2)

    # Open CSV once, write header
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "commanded_length_mm",
                "measured_length_mm",
                "commanded_angle_value",
                "detected_angle_deg",
                "pt1_x",
                "pt1_y",
                "pt2_x",
                "pt2_y",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h",
                "image_file",
            ],
        )
        writer.writeheader()

        for l in lengths:
            robo.moveL(base_pose)
            print("LENGTH:", l)
            advancer_go(l)

            # Measure length after advancing
            result = measure_beam_length_mm_with_checkerboard(
                image_filename="focused_image.jpg",
                use_roi=True,
                show=False,
            )
            length_curr = result.get("length_mm", None)
            print(f"Length after ADY is: {length_curr}")

            for i in values:
                print("ANGLE", i)
                pose = get_point(0, i)
                robo.moveL(pose)

                img_file = new_capture(filename="focused_image.jpg")

                pt1, pt2, angle, roi_box = detect_red_points_and_angle(
                    img_file,
                    show=False,
                    use_roi=True,
                )

                pt1_x, pt1_y = safe_xy(pt1)
                pt2_x, pt2_y = safe_xy(pt2)
                roi_x, roi_y, roi_w, roi_h = safe_roi(roi_box)

                print(angle)

                writer.writerow(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "commanded_length_mm": float(l),
                        "measured_length_mm": float(length_curr) if length_curr is not None else None,
                        "commanded_angle_value": float(i),
                        "detected_angle_deg": float(angle) if angle is not None else None,
                        "pt1_x": pt1_x,
                        "pt1_y": pt1_y,
                        "pt2_x": pt2_x,
                        "pt2_y": pt2_y,
                        "roi_x": roi_x,
                        "roi_y": roi_y,
                        "roi_w": roi_w,
                        "roi_h": roi_h,
                        "image_file": img_file,
                    }
                )
                f.flush()

finally:
    robo.shutdown()

print(f"Saved results to: {OUTPUT_CSV}")
