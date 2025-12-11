import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math

# =====================================================================
# CONFIG
# =====================================================================
ROI_CONFIG_FILE = "red_roi_box.json"

CAMERA_TO_CHECKERBOARD_MM = 80.0
BEAM_RELATIVE_Z_OFFSET_MM = 40.0   # beam distance relative to checkerboard along Z

CHECKERBOARD_SQUARE_SIZE_MM = 5.0
CHECKERBOARD_SQUARES_X = 9
CHECKERBOARD_SQUARES_Y = 6

CHECKERBOARD_PATTERN_SIZE = (
    CHECKERBOARD_SQUARES_X - 1,
    CHECKERBOARD_SQUARES_Y - 1
)

# =====================================================================
# BASIC CAPTURE & ROI UTILS
# =====================================================================

def new_capture(filename="focused_image.jpg", focus=255):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # warm-up
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture image")

    cv2.imwrite(filename, frame)
    return filename


def load_roi_box(path=ROI_CONFIG_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    print(f"[INFO] Loaded ROI from {path}: {data}")
    return box


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


def select_roi_interactive(image, window_name="Select beam ROI"):
    img_copy = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)

# =====================================================================
# CHECKERBOARD → HOMOGRAPHY (PIXELS → mm ON CHECKERBOARD PLANE)
# =====================================================================

def find_checkerboard_debug(image_bgr, pattern_size, show_debug=False):
    gray_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    gray1 = gray_orig.copy()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray2 = clahe.apply(gray_orig)

    variants = [
        ("plain", gray1),
        ("clahe", gray2),
    ]

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    # Try SB first if available
    for name, g in variants:
        try:
            if hasattr(cv2, "findChessboardCornersSB"):
                found, corners = cv2.findChessboardCornersSB(
                    g, pattern_size,
                    flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
                )
                if found:
                    print(f"[DEBUG] chessboard found with SB on {name}")
                    return g, corners
        except cv2.error:
            pass

    # Fallback: classic with flags
    for name, g in variants:
        found, corners = cv2.findChessboardCorners(g, pattern_size, flags)
        if found:
            print(f"[DEBUG] chessboard found with classic+flags on {name}")
            return g, corners

    # Last resort: classic with no flags
    for name, g in variants:
        found, corners = cv2.findChessboardCorners(g, pattern_size)
        if found:
            print(f"[DEBUG] chessboard found with classic (no flags) on {name}")
            return g, corners

    if show_debug:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("gray plain")
        plt.imshow(gray1, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("gray CLAHE")
        plt.imshow(gray2, cmap="gray")
        plt.axis("off")
        plt.show()

    raise RuntimeError("Checkerboard not found in any variant.")


def compute_checkerboard_homography(image_bgr):
    gray, corners = find_checkerboard_debug(image_bgr, CHECKERBOARD_PATTERN_SIZE, show_debug=False)

    # Subpixel refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    num_inner_x, num_inner_y = CHECKERBOARD_PATTERN_SIZE
    objp = np.zeros((num_inner_x * num_inner_y, 2), np.float32)
    objp[:, :2] = np.mgrid[0:num_inner_x, 0:num_inner_y].T.reshape(-1, 2)
    objp *= CHECKERBOARD_SQUARE_SIZE_MM

    src_pts = corners_refined.reshape(-1, 2)
    dst_pts = objp

    H_img_to_mm, _ = cv2.findHomography(src_pts, dst_pts)
    if H_img_to_mm is None:
        raise RuntimeError("Homography computation failed.")

    return H_img_to_mm


def image_points_to_mm(points, H_img_to_mm):
    pts = np.array(points, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    pts_mm_h = (H_img_to_mm @ pts_h.T).T
    pts_mm = pts_mm_h[:, :2] / pts_mm_h[:, 2:3]
    return [tuple(p) for p in pts_mm]

# =====================================================================
# DETECT RED MARKERS INSIDE ROI (2 markers → beam ends)
# =====================================================================

def detect_red_markers_in_roi(image_bgr, use_roi=True, expected_markers=2):
    h_full, w_full = image_bgr.shape[:2]

    roi_box = None
    if use_roi:
        roi_box = load_roi_box()
        if roi_box is None:
            print("[INFO] No ROI stored yet. Draw a box around the beam region.")
            roi_box = select_roi_interactive(image_bgr)
            if roi_box is None:
                raise RuntimeError("No ROI selected.")
            save_roi_box(roi_box)

    if roi_box is not None:
        x, y, w, h = roi_box
        x = max(0, min(x, w_full - 1))
        y = max(0, min(y, h_full - 1))
        w = max(1, min(w, w_full - x))
        h = max(1, min(h, h_full - y))
        roi_box = (x, y, w, h)
        roi_img = image_bgr[y:y+h, x:x+w]
    else:
        x, y, w, h = 0, 0, w_full, h_full
        roi_img = image_bgr

    image_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    red_ranges = [
        (np.array([0, 50, 50]),   np.array([10, 255, 255])),
        (np.array([160, 50, 50]), np.array([180, 255, 255])),
    ]
    red_mask = None
    for lower_red, upper_red in red_ranges:
        temp_mask = cv2.inRange(image_hsv, lower_red, upper_red)
        red_mask = temp_mask if red_mask is None else cv2.bitwise_or(red_mask, temp_mask)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < expected_markers:
        raise ValueError(f"Expected at least {expected_markers} red markers, found {len(contours)}.")

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:expected_markers]

    centers = []
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            full_cx = cx + x
            full_cy = cy + y
            centers.append((full_cx, full_cy))

    if len(centers) < expected_markers:
        raise ValueError("Could not compute all marker centroids.")

    centers.sort(key=lambda p: (p[1], p[0]))
    pt1, pt2 = centers
    return pt1, pt2, roi_box

# =====================================================================
# TARGET PICKING VIA MATPLOTLIB
# =====================================================================

def pick_target_point(image_bgr):
    """
    Let the user click a single target point using Matplotlib.
    Returns (x, y) in pixel coordinates.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_rgb)
    plt.title("Click target point, then close window")
    pts = plt.ginput(1)
    plt.close()

    if not pts:
        return None

    x, y = pts[0]
    return (float(x), float(y))

# =====================================================================
# MAIN: USER CLICKS TARGET → LENGTH + ANGLE (BASE → TARGET)
# =====================================================================

def compute_beam_line_to_target(image_filename="focused_image.jpg", use_roi=True, show=True):
    """
    - Captures an image.
    - Finds checkerboard homography.
    - Detects red markers (base, tip) in ROI.
    - Lets the user click a target point.
    - Computes:
        * base→target straight-line length in beam plane (mm)
        * base→target angle in image coordinates (same convention as your beam angle).
    """
    img_file = new_capture(filename=image_filename)
    image = cv2.imread(img_file)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {img_file}")

    H_img_to_mm = compute_checkerboard_homography(image)

    # base_px, tip_px (full image coordinates)
    base_px, tip_px, roi_box = detect_red_markers_in_roi(image, use_roi=use_roi, expected_markers=2)

    # Note: order here follows your choice; keep as-is if it matches your system.
    (tip_mm_board, base_mm_board) = image_points_to_mm([base_px, tip_px], H_img_to_mm)
    base_mm_board = np.array(base_mm_board, dtype=np.float32)
    tip_mm_board  = np.array(tip_mm_board,  dtype=np.float32)

    # Depth correction
    z_board = float(CAMERA_TO_CHECKERBOARD_MM)
    z_beam  = z_board + float(BEAM_RELATIVE_Z_OFFSET_MM)
    if z_beam <= 0:
        raise ValueError("Invalid geometry: z_beam must be > 0.")

    depth_scale = z_beam / z_board
    base_mm = base_mm_board * depth_scale
    tip_mm  = tip_mm_board  * depth_scale

    # User selects target
    target_px = pick_target_point(image.copy())
    if target_px is None:
        raise RuntimeError("No target point selected.")

    (target_mm_board,) = image_points_to_mm([target_px], H_img_to_mm)
    target_mm_board = np.array(target_mm_board, dtype=np.float32)
    target_mm = target_mm_board * depth_scale

    # 1) desired length = straight-line base→target in beam plane
    vec_mm = target_mm - base_mm
    length_mm = float(np.linalg.norm(vec_mm))

    # 2) angle in image space, from base to target,
    #    using same convention as your beam angle (angle from vertical "down")
    base_px_arr   = np.array(base_px,   dtype=np.float32)
    target_px_arr = np.array(target_px, dtype=np.float32)
    target_vec_px = target_px_arr - base_px_arr

    ref = np.array([0.0, 1.0], dtype=np.float32)  

    def angle_from_ref(v):
        """Angle (deg) from 'down' to v, wrapped to [-90, 90] like your beam angle."""
        a_v   = math.atan2(v[1], v[0])
        a_ref = math.atan2(ref[1], ref[0])
        a_deg = math.degrees(a_ref - a_v) +90
        if a_deg > 90:
            a_deg -= 180
        elif a_deg < -90:
            a_deg += 180
        return a_deg 

    theta_target_deg = angle_from_ref(target_vec_px)
    theta_target_rad = math.radians(theta_target_deg)

    print("\n=== Base → target (line model) ===")
    print(f"Base (mm):          ({base_mm[0]:.2f}, {base_mm[1]:.2f})")
    print(f"Target (mm):        ({target_mm[0]:.2f}, {target_mm[1]:.2f})")
    print(f"L_des (base→target): {length_mm:.2f} mm")
    print(f"theta_target:       {theta_target_deg:.2f} deg")

    # Visualisation
    if show:
        vis = image.copy()

        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        base_px_int   = (int(base_px[0]),   int(base_px[1]))
        tip_px_int    = (int(tip_px[0]),    int(tip_px[1]))
        target_px_int = (int(target_px[0]), int(target_px[1]))

        cv2.circle(vis, base_px_int,   7, (0, 255, 0), -1)   # base
        cv2.circle(vis, tip_px_int,    7, (0, 0, 255), -1)   # current tip
        cv2.circle(vis, target_px_int, 7, (255, 255, 0), -1) # target

        cv2.line(vis, base_px_int, tip_px_int,    (255, 255, 255), 2)
        cv2.line(vis, base_px_int, target_px_int, (0, 255, 255), 1)

        text1 = f"L_des = {length_mm:.1f} mm"
        text2 = f"theta_target = {theta_target_deg:.1f}°"

        cv2.putText(vis, text1, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, text2, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Base, tip, and clicked target (line model)")
        plt.axis("off")
        plt.show()

    return {
        "base_mm": tuple(base_mm),
        "tip_mm": tuple(tip_mm),
        "target_mm": tuple(target_mm),

        "length_mm": length_mm,           # straight distance base→target
        "theta_target_rad": theta_target_rad,
        "theta_target_deg": theta_target_deg,

        "base_px": tuple(base_px),
        "tip_px": tuple(tip_px),
        "target_px": tuple(target_px),
    }


if __name__ == "__main__":
    result = compute_beam_line_to_target(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
    )

    print("\nResult dictionary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
