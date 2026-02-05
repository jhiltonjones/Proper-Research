import csv
import time
import numpy as np
import os

from proper_research.parameters import ROBOT_IP
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.vision.vision_w_arco import measure_lengths_mm

OUTPUT_CSV = "bending_results_55mm.csv"

IMG_DIR = "bending_images_55"
os.makedirs(IMG_DIR, exist_ok=True)

CSV_FIELDS = [
    "timestamp",
    "commanded_angle_value",
    "wire_length_mm",
    "total_length_mm",
    "angle_deg",
    "tip_x_m",
    "tip_y_m",
    "image_file",
]

def make_image_filename(angle_deg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    return os.path.join(IMG_DIR, f"img_a{angle_deg:+04d}_{ts}_{ms:03d}.jpg")

def result_to_row(result, commanded_angle_value):
    tip_x, tip_y = result["tip_world_m"]
    return {
        "timestamp": time.time(),
        "commanded_angle_value": commanded_angle_value,
        "wire_length_mm": result["wire_length_mm"],
        "total_length_mm": result["total_length_mm"],
        "angle_deg": result["angle_centerline_to_tip_deg"],
        "tip_x_m": tip_x,
        "tip_y_m": tip_y,
        "image_file": result["image_file"],
    }

def append_result_row(csv_path, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()

robo = URRtde(ROBOT_IP)

try:
    base_pose = get_point(0, 0)
    robo.moveL(base_pose)

    angles = np.arange(-70, 75, 5)

    for a in angles:
        print("ANGLE", a)
        pose = get_point(0, a)
        robo.moveL(pose)

        img_name = make_image_filename(a)

        try:
            result = measure_lengths_mm(
                image_filename=img_name,
                use_roi=True,
                show=False,
                cam_index=0
            )
        except Exception as e:
            print("[ERROR] measure failed:", repr(e))
            continue

        row = result_to_row(result, commanded_angle_value=a)
        append_result_row(OUTPUT_CSV, row)

finally:
    robo.shutdown()

print(f"Saved results to: {OUTPUT_CSV}")
print(f"Saved images to: {IMG_DIR}/")
