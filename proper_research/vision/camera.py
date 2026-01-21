import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from proper_research.vision.measure_length import detect_red_markers_in_roi, new_capture
ROI_CONFIG_FILE = "red_roi_box.json"  

# def measure_theta_from_camera():
#     img_file = new_capture(filename="focused_image.jpg")
#     pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
#         img_file,
#         show=False,
#         use_roi=True
#     )
#     return np.deg2rad(angle_deg) 
# def new_capture(filename='focused_image.jpg', focus=255):
#     cap = cv2.VideoCapture(2)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")

#     for _ in range(5):
#         cap.read()

#     ret, frame = cap.read()
#     cap.release()

#     if ret and frame is not None:
#         cv2.imwrite(filename, frame)
#         return filename
#     else:
#         raise RuntimeError("Failed to capture image")
import cv2
import numpy as np

import cv2

# def new_capture(filename="focused_image.jpg",
#                 cam_index=2,
#                 backend=cv2.CAP_V4L2,
#                 warmup_frames=15,
#                 exposure=50.0,     # try 200..5000 initially
#                 gain=0.0,
#                 auto_exposure_manual=1.0,  # working for you
#                 brightness=None,     # e.g. 0.0
#                 gamma=None):         # e.g. 0.7
#     cap = cv2.VideoCapture(cam_index, backend)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open camera index {cam_index} with backend {backend}")

#     def try_set(prop, val, name):
#         ok = cap.set(prop, val)
#         got = cap.get(prop)
#         print(f"{name}: set({val}) -> {ok}, get() -> {got}")
#         return ok, got

#     # Put camera in manual exposure mode (as supported by your driver mapping)
#     try_set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure_manual), "AUTO_EXPOSURE(manual)")

#     # Reduce gain first
#     try_set(cv2.CAP_PROP_GAIN, float(gain), "GAIN")

#     # Set exposure (absolute value for your camera/driver)
#     try_set(cv2.CAP_PROP_EXPOSURE, float(exposure), "EXPOSURE(abs)")

#     # Optional tweaks if supported
#     if brightness is not None:
#         try_set(cv2.CAP_PROP_BRIGHTNESS, float(brightness), "BRIGHTNESS")
#     if gamma is not None:
#         try_set(cv2.CAP_PROP_GAMMA, float(gamma), "GAMMA")

#     for _ in range(warmup_frames):
#         cap.read()

#     ret, frame = cap.read()
#     cap.release()

#     if not ret or frame is None:
#         raise RuntimeError("Failed to capture image")

#     cv2.imwrite(filename, frame)
#     return filename


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

# def compute_signed_angle(v1, v2):
#     """Returns the signed angle in degrees from v1 to v2 (positive = CCW, negative = CW)."""
#     angle1 = np.arctan2(v1[1], v1[0])
#     angle2 = np.arctan2(v2[1], v2[0])
#     angle_rad = angle2 - angle1
#     angle_deg = np.degrees(angle_rad)

#     if angle_deg > 90:
#         angle_deg -= 180
#     elif angle_deg < -90:
#         angle_deg += 180

#     return angle_deg

# def detect_red_points_and_angle(
#     image_path,
#     show=False,
#     use_roi=True,
#     extra_points=None,          # NEW: dict or list of (label, (x,y))
#     show_mask=False,            # NEW: visualize red mask (debug)
#     extra_point_radius=6,       # NEW
# ):
#     """
#     Detects 2 biggest red blobs in the image (optionally inside a stored / selected ROI),
#     returns their pixel coordinates and the angle of the vector from pt1 -> pt2.

#     NEW:
#       extra_points: overlay additional points when show=True.
#         - dict: {"base": (x,y), "tip": (x,y)}
#         - or list: [("base",(x,y)), ("target0",(x,y)), ...]
#       show_mask: if True, show the red mask used for contour detection.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not read image at {image_path}")

#     h_full, w_full = image.shape[:2]

#     roi_box = None
#     if use_roi:
#         roi_box = load_roi_box()
#         if roi_box is None:
#             print("[INFO] No ROI stored yet. Draw a box around the beam region.")
#             roi_box = select_roi_interactive(image)
#             if roi_box is None:
#                 raise RuntimeError("No ROI selected.")
#             save_roi_box(roi_box)

#     if roi_box is not None:
#         x, y, w, h = roi_box
#         x = max(0, min(x, w_full - 1))
#         y = max(0, min(y, h_full - 1))
#         w = max(1, min(w, w_full - x))
#         h = max(1, min(h, h_full - y))
#         roi_box = (x, y, w, h)
#         roi_img = image[y:y+h, x:x+w]
#     else:
#         x, y, w, h = 0, 0, w_full, h_full
#         roi_img = image

#     image_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

#     red_ranges = [
#         (np.array([0, 50, 50]),   np.array([10, 255, 255])),
#         (np.array([160, 50, 50]), np.array([180, 255, 255]))
#     ]

#     red_mask = None
#     for lower_red, upper_red in red_ranges:
#         temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
#         red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

#     contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) < 2:
#         if show and show_mask:
#             plt.figure(figsize=(6, 4))
#             plt.imshow(red_mask, cmap="gray")
#             plt.title("Red mask (ROI)")
#             plt.axis("off")
#             plt.show()
#         raise ValueError("Less than two red points detected inside ROI!")

#     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
#     red_centers = []
#     for cnt in sorted_contours:
#         M = cv2.moments(cnt)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             full_cx = cx + x
#             full_cy = cy + y
#             red_centers.append((full_cx, full_cy))

#     if len(red_centers) < 2:
#         raise ValueError("Could not compute both marker centroids.")

#     red_centers.sort(key=lambda p: (p[1], p[0]))
#     pt1, pt2 = red_centers

#     vector = np.array(pt2, dtype=np.float32) - np.array(pt1, dtype=np.float32)
#     reference = np.array([0.0, 1.0])  # "down"
#     raw_angle = compute_signed_angle(reference, vector)
#     angle = unwrap_angle(raw_angle)

#     if show:
#         vis = image.copy()

#         # ROI box
#         if roi_box is not None:
#             rx, ry, rw, rh = roi_box
#             cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

#         # Detected red points
#         cv2.circle(vis, pt1, 6, (255, 0, 0), -1)  # blue
#         cv2.circle(vis, pt2, 6, (0, 0, 255), -1)  # red
#         cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

#         cv2.putText(vis, "pt1", (pt1[0] + 5, pt1[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(vis, "pt2", (pt2[0] + 5, pt2[1] - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

#         # Extra overlay points
#         if extra_points is not None:
#             if isinstance(extra_points, dict):
#                 items = list(extra_points.items())
#             else:
#                 items = list(extra_points)

#             for label, pt in items:
#                 if pt is None:
#                     continue
#                 px, py = int(pt[0]), int(pt[1])
#                 cv2.circle(vis, (px, py), int(extra_point_radius), (0, 255, 255), -1)  # yellow
#                 cv2.putText(vis, str(label), (px + 6, py + 6),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

#         plt.figure(figsize=(7, 7))
#         plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
#         plt.title(f"Beam Angle: {angle:.2f}°")
#         plt.axis("off")
#         plt.show()

#         if show_mask:
#             plt.figure(figsize=(6, 4))
#             plt.imshow(red_mask, cmap="gray")
#             plt.title("Red mask (ROI)")
#             plt.axis("off")
#             plt.show()

#     return pt1, pt2, angle, roi_box

# def measure_beam_angle_deg(image_filename="focused_image.jpg", use_roi=True, show=False):
#     """
#     Captures an image (or uses an existing file), detects the two red markers,
#     and returns the measured beam angle in degrees.
#     """
#     img_file = new_capture(filename=image_filename)
    
#     pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
#         img_file,
#         show=show,
#         use_roi=use_roi
#     )
#     return angle_deg, (pt1, pt2), roi_box


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

    base_px, mag_start_px, tip_px, roi_box = detect_red_markers_in_roi(
        image_bgr,
        use_roi=use_roi,
        expected_markers=3,
        show_debug=show_debug_markers
    )

    base = np.array(base_px, dtype=np.float32)
    mag  = np.array(mag_start_px, dtype=np.float32)
    tip  = np.array(tip_px, dtype=np.float32)

    v = (mag - base) if (use_segment == "base_to_mag") else (tip - base)

    reference = np.array([0.0, 1.0], dtype=np.float32)  # down in image coords
    angle_deg = compute_signed_angle(reference, v)

    if show:
        vis = image_bgr.copy()

        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

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

    return angle_deg, (tuple(base_px), tuple(mag_start_px), tuple(tip_px)), roi_box


if __name__ == "__main__":
    angle_deg, (base_px, mag_start_px, tip_px), roi_box = measure_beam_angle_deg(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        show_debug_markers=False,
        use_segment="base_to_tip",
    )

    print("base_px:", base_px)
    print("mag_start_px:", mag_start_px)
    print("tip_px:", tip_px)
    print("Angle (deg):", angle_deg)
    print("ROI:", roi_box)

