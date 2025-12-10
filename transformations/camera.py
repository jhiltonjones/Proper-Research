import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os

ROI_CONFIG_FILE = "red_roi_box.json"   # where we store the box between runs

# ─── CAMERA CAPTURE ─────────────────────────────────────────────────────
def new_capture(filename='focused_image.jpg', focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # Warm-up
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(filename, frame)
        return filename
    else:
        raise RuntimeError("Failed to capture image")

# ─── ROI STORAGE / SELECTION ────────────────────────────────────────────
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
    # clone to avoid modifying original
    img_copy = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # fromCenter=False = drag from corner to corner
    roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)

# ─── ANGLE UTILS ────────────────────────────────────────────────────────
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

def compute_signed_angle(v1, v2):
    """Returns the signed angle in degrees from v1 to v2 (positive = CCW, negative = CW)."""
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_rad = angle2 - angle1
    angle_deg = np.degrees(angle_rad)

    # Normalize to [-180, 180] but restricted to [-90, 90] in your design
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg

# ─── RED DETECTION WITH OPTIONAL ROI ────────────────────────────────────
def detect_red_points_and_angle(image_path, show=False, use_roi=True):
    """
    Detects 2 biggest red blobs in the image (optionally inside a stored / selected ROI),
    returns their pixel coordinates and the angle of the vector from pt1 -> pt2.

    If use_roi=True:
      - On first run (no ROI file), lets you draw a box with the mouse, saves it.
      - On future runs, loads that box and only looks for red inside it.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    h_full, w_full = image.shape[:2]

    # Decide ROI
    roi_box = None
    if use_roi:
        roi_box = load_roi_box()
        if roi_box is None:
            print("[INFO] No ROI stored yet. Draw a box around the beam region.")
            roi_box = select_roi_interactive(image)
            if roi_box is None:
                raise RuntimeError("No ROI selected.")
            save_roi_box(roi_box)

    if roi_box is not None:
        x, y, w, h = roi_box
        # clamp to image bounds just in case
        x = max(0, min(x, w_full - 1))
        y = max(0, min(y, h_full - 1))
        w = max(1, min(w, w_full - x))
        h = max(1, min(h, h_full - y))
        roi_box = (x, y, w, h)
        roi_img = image[y:y+h, x:x+w]
    else:
        # full image
        x, y, w, h = 0, 0, w_full, h_full
        roi_img = image

    # Convert ROI region to HSV
    image_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

    # Red thresholds
    # red_ranges = [
    #     (np.array([0,   140, 80]), np.array([10, 255, 255])),
    #     (np.array([170, 140, 80]), np.array([180, 255, 255]))
    # ]
    red_ranges = [
        (np.array([0, 50, 50]), np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255]))
    ]
    red_mask = None
    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        raise ValueError("Less than two red points detected inside ROI!")

    # Take the largest two contours
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    red_centers = []
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # convert ROI coords -> full image coords
            full_cx = cx + x
            full_cy = cy + y
            red_centers.append((full_cx, full_cy))

    if len(red_centers) < 2:
        raise ValueError("Could not compute both marker centroids.")

    # sort by y then x (as you had)
    red_centers.sort(key=lambda p: (p[1], p[0]))
    pt1, pt2 = red_centers

    vector = np.array(pt2, dtype=np.float32) - np.array(pt1, dtype=np.float32)
    reference = np.array([0.0, 1.0])  # "down"
    raw_angle = compute_signed_angle(reference, vector)
    angle = unwrap_angle(raw_angle)

    if show:
        vis = image.copy()

        # draw ROI box if used
        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # draw points and line
        cv2.circle(vis, pt1, 5, (255, 0, 0), -1)
        cv2.circle(vis, pt2, 5, (0, 0, 255), -1)
        cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Beam Angle: {angle:.2f}°")
        plt.axis("off")
        plt.show()

    return pt1, pt2, angle, roi_box

# ─── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Capture or use existing
    img_file = new_capture(filename="focused_image.jpg")
    # img_file = "focused_image.jpg"  # if you want to reuse existing

    pt1, pt2, angle, roi_box = detect_red_points_and_angle(
        img_file,
        show=True,
        use_roi=True
    )

    print("Red points (image coordinates):")
    print("  pt1:", pt1)
    print("  pt2:", pt2)
    print("Angle (deg):", angle)
    if roi_box is not None:
        print("ROI used (x, y, w, h):", roi_box)
