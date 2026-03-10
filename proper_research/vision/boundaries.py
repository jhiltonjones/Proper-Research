import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from proper_research.vision.measure_length import new_capture

ROI_CONFIG_FILE = "red_roi_box.json"
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
def detect_blue_vessel_boundaries(
    image_bgr,
    roi_box=None,
    blue_h_low=90,
    blue_h_high=140,
    sat_min=40,
    val_min=40,
    min_pixels_per_row=2,
    show_debug=False,
):
    """
    Detect blue vessel boundary lines and reconstruct left/right boundaries.

    Returns:
        result = {
            "left_boundary_px": [(x1, y1), ...],
            "right_boundary_px": [(x1, y1), ...],
            "centerline_px": [(xc1, y1), ...],   # optional helper
            "roi_box": roi_box,
        }

    Notes:
    - image coords: x right, y down
    - boundaries are returned in full-image pixel coordinates
    """

    # ----------------------------
    # Crop ROI if provided
    # ----------------------------
    if roi_box is not None:
        x0, y0, w, h = roi_box
        roi = image_bgr[y0:y0+h, x0:x0+w].copy()
    else:
        roi = image_bgr.copy()
        x0, y0 = 0, 0
        h, w = roi.shape[:2]

    # ----------------------------
    # Threshold blue in HSV
    # ----------------------------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([blue_h_low, sat_min, val_min], dtype=np.uint8)
    upper_blue = np.array([blue_h_high, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # light cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ----------------------------
    # Row-wise reconstruction
    # ----------------------------
    left_boundary = []
    right_boundary = []
    centerline = []

    H, W = mask.shape

    for y in range(H):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) < min_pixels_per_row:
            continue

        # leftmost and rightmost blue pixel in this row
        x_left = int(xs.min())
        x_right = int(xs.max())

        # reject degenerate rows where only one side appears
        if x_right <= x_left:
            continue

        x_center = 0.5 * (x_left + x_right)

        left_boundary.append((x_left + x0, y + y0))
        right_boundary.append((x_right + x0, y + y0))
        centerline.append((x_center + x0, y + y0))

    if len(left_boundary) < 10 or len(right_boundary) < 10:
        raise RuntimeError(
            f"Too few boundary points detected. "
            f"left={len(left_boundary)}, right={len(right_boundary)}"
        )

    result = {
        "left_boundary_px": left_boundary,
        "right_boundary_px": right_boundary,
        "centerline_px": centerline,
        "roi_box": roi_box,
        "mask": mask,
    }

    if show_debug:
        vis = roi.copy()

        for (x, y) in left_boundary:
            cv2.circle(vis, (int(x - x0), int(y - y0)), 1, (0, 255, 0), -1)

        for (x, y) in right_boundary:
            cv2.circle(vis, (int(x - x0), int(y - y0)), 1, (0, 0, 255), -1)

        for (x, y) in centerline:
            cv2.circle(vis, (int(x - x0), int(y - y0)), 1, (255, 255, 0), -1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.title("ROI")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Blue mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed boundaries")
        plt.axis("off")

        plt.show()

    return result
def smooth_boundary(boundary_points, window=9):
    """
    Smooth x-values as a function of y using a moving average.
    boundary_points: list of (x, y), assumed ordered by y
    """
    pts = np.array(boundary_points, dtype=np.float32)

    xs = pts[:, 0]
    ys = pts[:, 1]

    if len(xs) < window:
        return boundary_points

    kernel = np.ones(window, dtype=np.float32) / window
    xs_smooth = np.convolve(xs, kernel, mode="same")

    smoothed = [(float(x), float(y)) for x, y in zip(xs_smooth, ys)]
    return smoothed
def plot_vessel_boundaries(left_boundary, right_boundary):
    left = np.array(left_boundary, dtype=np.float32)
    right = np.array(right_boundary, dtype=np.float32)

    # flip y so plot is Cartesian-like
    plt.figure(figsize=(6, 8))
    plt.plot(left[:, 0], -left[:, 1], label="left boundary")
    plt.plot(right[:, 0], -right[:, 1], label="right boundary")
    plt.axis("equal")
    plt.xlabel("x (px)")
    plt.ylabel("y (px, upward)")
    plt.title("Reconstructed vessel boundaries")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    img_file = new_capture(filename="focused_image.jpg")
    image_bgr = cv2.imread("focused_image.jpg")
    if image_bgr is None:
        raise FileNotFoundError("Could not read focused_image.jpg")

    roi_box = load_roi_box()   # or use a new ROI for the vessel region

    vessel = detect_blue_vessel_boundaries(
        image_bgr=image_bgr,
        roi_box=roi_box,
        blue_h_low=90,
        blue_h_high=140,
        sat_min=40,
        val_min=40,
        show_debug=True,
    )

    left_smooth = smooth_boundary(vessel["left_boundary_px"], window=11)
    right_smooth = smooth_boundary(vessel["right_boundary_px"], window=11)

    print("Left boundary points:", len(left_smooth))
    print("Right boundary points:", len(right_smooth))
    plot_vessel_boundaries(left_smooth, right_smooth)