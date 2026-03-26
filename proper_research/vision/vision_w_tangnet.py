import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from proper_research.vision.measure_length import new_capture

ROI_CONFIG_FILE = "red_roi_box.json"


# ----------------------------
# ROI utilities
# ----------------------------
def save_roi_box(box, path=ROI_CONFIG_FILE):
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
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    print(f"[INFO] Loaded ROI from {path}: {data}")
    return box


def select_roi_interactive(image, window_name="Select red search area"):
    img_copy = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)


# ----------------------------
# Angle helper
# ----------------------------
class AngleUnwrapper:
    def __init__(self):
        self.prev = None

    def __call__(self, a_deg):
        if self.prev is None:
            self.prev = a_deg
            return a_deg

        delta = a_deg - self.prev
        if delta > 180.0:
            a_deg -= 360.0
        elif delta < -180.0:
            a_deg += 360.0

        self.prev = a_deg
        return a_deg


unwrap_tip_angle = AngleUnwrapper()
def signed_angle_between_vectors(v1, v2):
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    ang = np.degrees(a2 - a1)

    while ang > 180:
        ang -= 360
    while ang < -180:
        ang += 360

    return float(ang)

# ----------------------------
# Geometry helpers
# ----------------------------
def angle_deg_from_points(p1, p2):
    """
    Returns angle of vector p1 -> p2 in standard Cartesian coordinates:
      +x to the right
      +y upward
    """
    dx = float(p2[0] - p1[0])
    dy = float(-(p2[1] - p1[1]))   # flip image y to Cartesian y
    return float(np.degrees(np.arctan2(dy, dx)))


def image_to_base_frame(point_px, base_px):
    """
    Converts image pixel coordinates into a base-centered 2D Cartesian frame:
      x positive right
      y positive up
    """
    x = float(point_px[0] - base_px[0])
    y = float(-(point_px[1] - base_px[1]))
    return np.array([x, y], dtype=np.float32)


def order_four_markers(points, pivot_hint=None):
    """
    Orders 4 detected marker centroids as:
      [base, mag_start, tangent_start, tip]

    Strategy:
    - If pivot_hint is provided, choose detected point nearest to pivot_hint as base.
    - Then greedily walk to the nearest remaining point.
    - This usually works well when markers lie along the wire/tip path.

    points: list of (x, y)
    pivot_hint: optional (x, y)
    """
    pts = [np.array(p, dtype=np.float32) for p in points]
    if len(pts) != 4:
        raise ValueError(f"Expected 4 points, got {len(pts)}")

    if pivot_hint is not None:
        pivot_hint = np.array(pivot_hint, dtype=np.float32)
        start_idx = int(np.argmin([np.linalg.norm(p - pivot_hint) for p in pts]))
    else:
        # fallback: leftmost point as base
        start_idx = int(np.argmin([p[0] for p in pts]))

    ordered = [pts.pop(start_idx)]

    while pts:
        last = ordered[-1]
        next_idx = int(np.argmin([np.linalg.norm(p - last) for p in pts]))
        ordered.append(pts.pop(next_idx))

    return [tuple(map(float, p)) for p in ordered]


# ----------------------------
# Main measurement
# ----------------------------
def measure_tip_state_4markers(
    image_filename="focused_image2.jpg",
    roi_box=None,
    show=True,
    show_debug_markers=False,
    unwrap_angle=True,
    pivot_hint=None,
):
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    detected_points = detect_4_red_markers_in_roi(
        image_bgr,
        roi_box=roi_box,
        min_area=2,
        max_area=40000,
        sat_min=10,
        val_min=10,
        hue1_high=10,
        hue2_low=160,
        show_debug=show_debug_markers,
    )

def detect_4_red_markers_in_roi(
    image_bgr,
    roi_box=None,
    min_area=2,
    max_area=40000,
    show_debug=False,
    sat_min=20,
    val_min=20,
    hue1_low=0,
    hue1_high=20,
    hue2_low=160,
    hue2_high=180,
    debug_allow_less_than_4=False,
):
    if roi_box is not None:
        x, y, w, h = roi_box
        roi = image_bgr[y:y+h, x:x+w].copy()
        x0, y0 = x, y
    else:
        roi = image_bgr.copy()
        x0, y0 = 0, 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower1 = np.array([hue1_low, sat_min, val_min], dtype=np.uint8)
    upper1 = np.array([hue1_high, 255, 255], dtype=np.uint8)

    lower2 = np.array([hue2_low, sat_min, val_min], dtype=np.uint8)
    upper2 = np.array([hue2_high, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Try opening only first; avoid closing because it can merge blobs
    k = np.ones((3, 3), np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"[DEBUG] roi_box = {roi_box}")
    print(f"[DEBUG] mask nonzero = {int(np.count_nonzero(mask_open))}")
    print(f"[DEBUG] found {len(contours)} raw contours")

    candidates = []
    raw_debug = roi.copy()
    kept_debug = roi.copy()

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        if abs(M["m00"]) > 1e-8:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = -1, -1

        print(
            f"[DEBUG] contour {i}: area={area:.2f}, "
            f"bbox=({bx},{by},{bw},{bh}), center=({cx:.1f},{cy:.1f})"
        )

        # draw every raw contour
        cv2.drawContours(raw_debug, [cnt], -1, (0, 255, 0), 1)
        cv2.rectangle(raw_debug, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1)
        cv2.putText(raw_debug, f"{i}", (bx, max(0, by - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if area < min_area or area > max_area:
            continue
        if abs(M["m00"]) <= 1e-8:
            continue

        candidates.append({
            "point": (float(cx + x0), float(cy + y0)),
            "area": float(area),
            "contour": cnt,
            "local_center": (float(cx), float(cy)),
            "bbox": (bx, by, bw, bh),
        })

    candidates = sorted(candidates, key=lambda d: d["area"], reverse=True)

    print(f"[DEBUG] kept {len(candidates)} candidates after filtering")

    # draw kept candidates
    for j, c in enumerate(candidates):
        cnt = c["contour"]
        cx, cy = c["local_center"]
        bx, by, bw, bh = c["bbox"]
        cv2.drawContours(kept_debug, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(kept_debug, (bx, by), (bx + bw, by + bh), (255, 255, 0), 1)
        cv2.circle(kept_debug, (int(round(cx)), int(round(cy))), 4, (255, 0, 0), -1)
        cv2.putText(kept_debug, f"{j}", (bx, max(0, by - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if show_debug:
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Raw red mask")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(mask_open, cmap="gray")
        plt.title("Opened mask")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(raw_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"All raw contours: {len(contours)}")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(kept_debug, cv2.COLOR_BGR2RGB))
        plt.title(f"Kept candidates: {len(candidates)}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    if len(candidates) < 4 and not debug_allow_less_than_4:
        raise RuntimeError(f"Expected at least 4 red markers, found {len(candidates)}")

    points = [c["point"] for c in candidates[:4]]
    return points
def show_red_mask(image_bgr, roi_box=None, sat_min=40, val_min=30, hue1_high=15, hue2_low=165):
    if roi_box is not None:
        x, y, w, h = roi_box
        roi = image_bgr[y:y+h, x:x+w].copy()
    else:
        roi = image_bgr.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, sat_min, val_min], dtype=np.uint8)
    upper1 = np.array([hue1_high, 255, 255], dtype=np.uint8)

    lower2 = np.array([hue2_low, sat_min, val_min], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("ROI")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Red mask")
    plt.axis("off")
    plt.show()
    
if __name__ == "__main__":
    img_file = new_capture(filename="focused_image2.jpg")
    image_bgr = cv2.imread("focused_image2.jpg")
    if image_bgr is None:
        raise FileNotFoundError("Could not read focused_image.jpg")
    red_roi_box = load_roi_box("red_roi_box.json")

    result = measure_tip_state_4markers(
        image_filename=img_file,
        roi_box=red_roi_box,
        show=True,
        show_debug_markers=True,
        unwrap_angle=True,
        pivot_hint=None,
    )

    # print("\n--- OUTPUT ---")
    # print("Tip image px:", result["tip_image_px"])
    # print("Tip x,y from base:", result["tip_xy_from_base"])
    # print("Tip tangent angle (deg):", result["tip_tangent_angle_deg"])
    # print("Markers:", result["markers"])
    # print("ROI:", result["roi_box"])
    roi_box = load_roi_box()

    image_bgr = cv2.imread("focused_image2.jpg")
    if image_bgr is None:
        raise FileNotFoundError("Could not read focused_image.jpg")

    show_red_mask(
        image_bgr=image_bgr,
        roi_box=roi_box,
        sat_min=10,
        val_min=15,
        hue1_high=20,
        hue2_low=160
    )