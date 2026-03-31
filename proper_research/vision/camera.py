import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from proper_research.vision.measure_length import detect_red_markers_in_roi, new_capture
ROI_CONFIG_FILE = "blue_roi_box.json"  

def save_roi_box(box, path=ROI_CONFIG_FILE):
    """
    box: (x, y, w, h) in image pixels.
    """
    data = {
        "x": int(box[0]),
        "y": int(box[1]),
        "w": int(box[2]),
        "h": int(box[3]),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[INFO] ROI saved to {path}: {data}")


def load_roi_box(path=ROI_CONFIG_FILE):
    """
    Returns box (x, y, w, h) if file exists, else None.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    print(f"[INFO] Loaded ROI from {path}: {data}")
    return box


def select_roi_interactive(image, window_name="Select red search area"):
    """
    Lets you draw a box with the mouse. Returns (x, y, w, h).
    Uses OpenCV's built-in ROI selector.
    """
    img_copy = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)
import cv2
import numpy as np
import json
import os

POLYGON_CONFIG_FILE = "custom_area.json"

def save_polygon(points, path=POLYGON_CONFIG_FILE):
    data = {"points": [[int(x), int(y)] for x, y in points]}
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[INFO] Polygon saved to {path}: {data}")

def load_polygon(path=POLYGON_CONFIG_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    points = [(int(x), int(y)) for x, y in data["points"]]
    print(f"[INFO] Loaded polygon from {path}: {data}")
    return points

def polygon_to_mask(image_shape, polygon):
    """
    polygon: list of (x, y)
    returns uint8 mask with 255 inside polygon, 0 outside
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def select_polygon_interactive(image, window_name="Select custom area"):
    """
    Left click to add polygon points.
    Press ENTER or SPACE to finish.
    Press 'u' to undo last point.
    Press ESC or 'q' to cancel.
    """
    display = image.copy()
    points = []

    def redraw():
        nonlocal display
        display = image.copy()

        # draw polygon lines
        if len(points) > 0:
            for i, pt in enumerate(points):
                cv2.circle(display, pt, 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, points[i - 1], pt, (0, 255, 0), 2)

        # preview closed polygon if >= 3 points
        if len(points) >= 3:
            cv2.line(display, points[-1], points[0], (255, 0, 0), 1)

        cv2.imshow(window_name, display)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if len(points) >= 3:
                break
        elif key == ord('u'):
            if points:
                points.pop()
                redraw()
        elif key in (27, ord('q')):  # Esc or q
            points = None
            break

    cv2.destroyWindow(window_name)
    return points
class AngleUnwrapper:
    def __init__(self):
        self.prev = None
    def __call__(self, a):
        if self.prev is None:
            self.prev = a
            return a
        delta = a - self.prev
        if   delta > 180.0: a -= 360.0
        elif delta < -180.0: a += 360.0
        self.prev = a
        return a

unwrap_angle = AngleUnwrapper()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- keep your compute_signed_angle somewhere above ----
def compute_signed_angle(v1, v2):
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_deg = np.degrees(angle2 - angle1)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return float(angle_deg)



def measure_beam_angle_deg(
    image_filename="focused_image.jpg",
    use_roi=True,
    show=False,
    show_debug_markers=False,
    use_segment="base_to_tip",   # "base_to_tip" or "base_to_mag"
):
    """
    Captures an image, detects 3 red markers (base, mag_start, tip),
    returns beam angle in degrees.
    """
    img_file = new_capture(filename=image_filename)
    image_bgr = cv2.imread(img_file)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {img_file}")

    base_px, mag_start_px, tip_px, search_region = detect_red_markers_in_roi(
        image_bgr,
        use_roi=True,
        use_polygon=True,
        load_polygon_fn=load_polygon,
        save_polygon_fn=save_polygon,
        select_polygon_fn=select_polygon_interactive,
        expected_markers=3,
        show_debug=True,
    )

    base = np.array(base_px, dtype=np.float32)
    mag  = np.array(mag_start_px, dtype=np.float32)
    tip  = np.array(tip_px, dtype=np.float32)

    v = (mag - base) if (use_segment == "base_to_mag") else (tip - base)

    reference = np.array([0.0, 1.0], dtype=np.float32)  # down in image coords
    angle_deg = compute_signed_angle(reference, v)

    if show:
        vis = image_bgr.copy()

    if search_region is not None:
        if isinstance(search_region, tuple) and len(search_region) == 4:
            x, y, w, h = search_region
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        else:
            pts = np.array(search_region, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.circle(vis, (int(base[0]), int(base[1])), 7, (255,   0,   0), -1)
        cv2.circle(vis, (int(mag[0]),  int(mag[1])),  7, (0, 255, 255), -1)
        cv2.circle(vis, (int(tip[0]),  int(tip[1])),  7, (0,   0, 255), -1)

        p2 = mag if use_segment == "base_to_mag" else tip
        cv2.line(vis, (int(base[0]), int(base[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

        plt.figure(figsize=(7, 7))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam angle: {angle_deg:.2f}° ({use_segment})")
        plt.axis("off")
        plt.show()

    return angle_deg, (tuple(base_px), tuple(mag_start_px), tuple(tip_px))


if __name__ == "__main__":
    angle_deg, (base_px, mag_start_px, tip_px) = measure_beam_angle_deg(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        show_debug_markers=True,
        use_segment="base_to_tip",
    )

    print("base_px:", base_px)
    print("mag_start_px:", mag_start_px)
    print("tip_px:", tip_px)
    print("Angle (deg):", angle_deg)

