import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# =====================================================================
# CONFIG
# =====================================================================
ROI_CONFIG_FILE = "red_roi_box.json"

CAMERA_TO_CHECKERBOARD_MM = 30.0
BEAM_RELATIVE_Z_OFFSET_MM = 40.0   # beam distance relative to checkerboard along Z

CHECKERBOARD_SQUARE_SIZE_MM = 6
CHECKERBOARD_SQUARES_X = 9
CHECKERBOARD_SQUARES_Y = 6

CHECKERBOARD_PATTERN_SIZE = (
    CHECKERBOARD_SQUARES_X - 1,
    CHECKERBOARD_SQUARES_Y - 1
)

# def new_capture(filename='focused_image.jpg', focus=255):
#     cap = cv2.VideoCapture(2)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")

#     # warm-up
#     for _ in range(5):
#         cap.read()

#     ret, frame = cap.read()
#     cap.release()

#     if not ret or frame is None:
#         raise RuntimeError("Failed to capture image")

#     cv2.imwrite(filename, frame)
#     return filename
def new_capture(filename="focused_image.jpg",
                cam_index=0,
                backend=cv2.CAP_V4L2,
                warmup_frames=18,
                exposure=100.0,     # try 200..5000 initially
                gain=0.0,
                auto_exposure_manual=1.0,  # working for you
                brightness=None,     # e.g. 0.0
                gamma=None):         # e.g. 0.7
    cap = cv2.VideoCapture(cam_index, backend) 
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index} with backend {backend}")

    def try_set(prop, val, name):
        ok = cap.set(prop, val)
        got = cap.get(prop)
        print(f"{name}: set({val}) -> {ok}, get() -> {got}")
        return ok, got

    # Put camera in manual exposure mode (as supported by your driver mapping)
    try_set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exposure_manual), "AUTO_EXPOSURE(manual)")

    # Reduce gain first
    try_set(cv2.CAP_PROP_GAIN, float(gain), "GAIN")

    # Set exposure (absolute value for your camera/driver)
    try_set(cv2.CAP_PROP_EXPOSURE, float(exposure), "EXPOSURE(abs)")

    # Optional tweaks if supported
    if brightness is not None:
        try_set(cv2.CAP_PROP_BRIGHTNESS, float(brightness), "BRIGHTNESS")
    if gamma is not None:
        try_set(cv2.CAP_PROP_GAMMA, float(gamma), "GAMMA")

    for _ in range(warmup_frames):
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
# (FROM YOUR EXISTING CHECKERBOARD CODE, SLIGHTLY SHORTENED)
# =====================================================================

def find_checkerboard_debug(image_bgr, pattern_size, show_debug=True):
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

    # Fallback
    for name, g in variants:
        found, corners = cv2.findChessboardCorners(g, pattern_size, flags)
        if found:
            print(f"[DEBUG] chessboard found with classic+flags on {name}")
            return g, corners

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
        # plt.show()

    raise RuntimeError("Checkerboard not found in any variant.")

def compute_checkerboard_homography(image_bgr):
    gray, corners = find_checkerboard_debug(image_bgr, CHECKERBOARD_PATTERN_SIZE)

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

    H_img_to_mm, mask = cv2.findHomography(src_pts, dst_pts)
    if H_img_to_mm is None:
        raise RuntimeError("Homography computation failed.")

    return H_img_to_mm


# def detect_red_markers_in_roi(
#     image_bgr,
#     use_roi=True,
#     expected_markers=3,

#     # --- detection sensitivity ---
#     min_area=1.5,
#     merge_dist=1.5,

#     # --- morphology (less aggressive; avoid merging markers) ---
#     morph_kernel=(3, 3),
#     close_iters=1,
#     open_iters=1,

#     # --- HSV thresholds (tighter hue, lower S/V mins) ---
#     s_min=50,
#     v_min=50,
#     h_low1=0,  h_high1=15,
#     h_low2=165, h_high2=180,

#     # --- behavior when middle marker is missing ---
#     allow_two_markers_when_expected_three=True,  # NEW

#     # --- debug ---
#     show_debug=True,

#     # --- internal guard to prevent infinite fallback recursion ---
#     _did_fallback=False,  # NEW (do not set from outside)
# ):
#     """
#     Detect red blobs inside ROI. Returns marker centers in FULL-IMAGE coordinates.

#     For expected_markers=3:
#       base -> mag_start -> tip

#     Output ordered by increasing y (then x) internally, but returned as:
#       base (largest y), mag_start (middle y or None), tip (smallest y)
#     """
#     _show_debug_here = bool(show_debug) and bool(_did_fallback)
#     def merge_close_points(points, dist):
#         merged = []
#         for p in points:
#             p = np.asarray(p, dtype=float)
#             placed = False
#             for i, q in enumerate(merged):
#                 q = np.asarray(q, dtype=float)
#                 if np.linalg.norm(p - q) < dist:
#                     merged[i] = tuple(((p + q) / 2.0).tolist())
#                     placed = True
#                     break
#             if not placed:
#                 merged.append(tuple(p.tolist()))
#         return merged

#     h_full, w_full = image_bgr.shape[:2]

#     roi_box = None
#     if use_roi:
#         roi_box = load_roi_box()
#         if roi_box is None:
#             print("[INFO] No ROI stored yet. Draw a box around the beam region.")
#             roi_box = select_roi_interactive(image_bgr)
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
#         roi_img = image_bgr[y:y+h, x:x+w]
#     else:
#         x, y, w, h = 0, 0, w_full, h_full
#         roi_img = image_bgr

#     hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

#     red_ranges = [
#         (np.array([h_low1, s_min, v_min]), np.array([h_high1, 255, 255])),
#         (np.array([h_low2, s_min, v_min]), np.array([h_high2, 255, 255])),
#     ]

#     red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
#     for lo, hi in red_ranges:
#         red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))

#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
#     if close_iters > 0:
#         red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
#     if open_iters > 0:
#         red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  k, iterations=open_iters)

#     contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     centers = []
#     areas = []
#     for cnt in contours:
#         area = float(cv2.contourArea(cnt))
#         if area < float(min_area):
#             continue
#         M = cv2.moments(cnt)
#         if M["m00"] == 0:
#             continue
#         cx = float(M["m10"] / M["m00"]) + x
#         cy = float(M["m01"] / M["m00"]) + y
#         centers.append((cx, cy))
#         areas.append(area)

#     if _show_debug_here:
#         dbg = image_bgr.copy()
#         if roi_box is not None:
#             rx, ry, rw, rh = roi_box
#             cv2.rectangle(dbg, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
#         for (cx, cy) in centers:
#             cv2.circle(dbg, (int(cx), int(cy)), 6, (0, 255, 255), -1)
#         print(f"[DEBUG] raw contours={len(contours)}, kept after min_area={len(centers)}")
#         if len(areas) > 0:
#             print("[DEBUG] areas (kept):", sorted([round(a, 1) for a in areas], reverse=True)[:10])

#         plt.figure(figsize=(10, 4))
#         plt.subplot(1, 2, 1)
#         plt.title("Red mask (ROI)")
#         plt.imshow(red_mask, cmap="gray")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.title("Detections overlay")
#         plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
#         plt.axis("off")
#         # plt.show()

#     if len(centers) == 0:
#         raise ValueError("No red markers detected inside ROI (after filtering).")

#     centers = merge_close_points(centers, merge_dist)

#     # --- Fallback pass: do it at most once ---
#     if (len(centers) < expected_markers) and (not _did_fallback):
#         return detect_red_markers_in_roi(
#             image_bgr,
#             use_roi=use_roi,
#             expected_markers=expected_markers,
#             min_area=max(5, int(min_area * 0.5)),
#             merge_dist=merge_dist,
#             morph_kernel=(3, 3),
#             close_iters=0,  # key change
#             open_iters=1,
#             s_min=max(15, int(s_min * 0.5)),
#             v_min=max(15, int(v_min * 0.5)),
#             h_low1=h_low1, h_high1=h_high1,
#             h_low2=h_low2, h_high2=h_high2,
#             allow_two_markers_when_expected_three=allow_two_markers_when_expected_three,
#             show_debug=show_debug,
#             _did_fallback=True,
#         )

#     # Sort along beam (y then x)
#     centers.sort(key=lambda p: (p[1], p[0]))

#     # If more than expected (3-case), pick base, middle-near-midpoint, tip
#     if expected_markers == 3 and len(centers) > 3:
#         base = centers[-1]
#         tip = centers[0]
#         y_mid = 0.5 * (base[1] + tip[1])
#         middle = min(centers[1:-1], key=lambda p: abs(p[1] - y_mid))
#         centers = [tip, middle, base]
#         centers.sort(key=lambda p: (p[1], p[0]))
#     else:
#         centers = centers[:expected_markers]

#     # --- Handle 2-marker mode when expected_markers == 3 ---
#     if expected_markers == 3:
#         if len(centers) == 3:
#             # centers sorted: [tip, mid, base] by y
#             tip_px, mag_start_px, base_px = centers[0], centers[1], centers[2]
#             return base_px, mag_start_px, tip_px, roi_box

#         if len(centers) == 2 and allow_two_markers_when_expected_three:
#             # centers sorted: [tip, base] by y
#             tip_px, base_px = centers[0], centers[1]
#             mag_start_px = None
#             return base_px, mag_start_px, tip_px, roi_box

#         raise ValueError(
#             f"Expected 3 markers but detected {len(centers)} after fallback; "
#             "set allow_two_markers_when_expected_three=True to accept base+tip only."
#         )

#     # Non-3-marker case: keep your original behavior
#     return centers, roi_box
def detect_red_markers_in_roi(
    image_bgr,
    use_roi=True,
    expected_markers=3,

    # NEW: custom area support
    roi_polygon=None,                 # list of (x, y) points
    use_polygon=False,                # if True, prefer polygon over box
    select_polygon_fn=None,           # function(image) -> list[(x,y)] or None
    load_polygon_fn=None,             # function() -> polygon or None
    save_polygon_fn=None,             # function(polygon) -> None

    # --- detection sensitivity ---
    min_area=1.5,
    merge_dist=1.5,

    # --- morphology (less aggressive; avoid merging markers) ---
    morph_kernel=(3, 3),
    close_iters=1,
    open_iters=1,

    # --- HSV thresholds (tighter hue, lower S/V mins) ---
    s_min=50,
    v_min=50,
    h_low1=0,  h_high1=15,
    h_low2=165, h_high2=180,

    # --- behavior when middle marker is missing ---
    allow_two_markers_when_expected_three=True,

    # --- debug ---
    show_debug=True,

    # --- internal guard to prevent infinite fallback recursion ---
    _did_fallback=False,
):
    """
    Detect red blobs inside a search area. Returns marker centers in FULL-IMAGE coordinates.

    Search area options:
      - rectangular ROI via load_roi_box/select_roi_interactive
      - custom polygon via roi_polygon or load/select polygon callbacks

    For expected_markers=3:
      returns: base_px, mag_start_px, tip_px, search_region

    search_region is:
      - roi_box tuple (x, y, w, h) when using rectangle
      - polygon list[(x, y), ...] when using polygon
      - None when using full image
    """
    _show_debug_here = bool(show_debug) and bool(_did_fallback)

    def merge_close_points(points, dist):
        merged = []
        for p in points:
            p = np.asarray(p, dtype=float)
            placed = False
            for i, q in enumerate(merged):
                q = np.asarray(q, dtype=float)
                if np.linalg.norm(p - q) < dist:
                    merged[i] = tuple(((p + q) / 2.0).tolist())
                    placed = True
                    break
            if not placed:
                merged.append(tuple(p.tolist()))
        return merged

    def polygon_to_mask(image_shape, polygon):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    h_full, w_full = image_bgr.shape[:2]

    roi_box = None
    polygon = None
    search_region = None

    # ------------------------------------------------------------
    # Select/load search region
    # ------------------------------------------------------------
    if use_polygon:
        if roi_polygon is not None:
            polygon = roi_polygon
        elif load_polygon_fn is not None:
            polygon = load_polygon_fn()

        if polygon is None and use_roi:
            if select_polygon_fn is None:
                raise RuntimeError("use_polygon=True but no polygon selector was provided.")
            print("[INFO] No custom area stored yet. Draw a polygon around the beam region.")
            polygon = select_polygon_fn(image_bgr)
            if polygon is None:
                raise RuntimeError("No custom area selected.")
            if save_polygon_fn is not None:
                save_polygon_fn(polygon)

        if polygon is not None:
            polygon = [(int(px), int(py)) for px, py in polygon]
            search_region = polygon

    else:
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
            search_region = roi_box

    # ------------------------------------------------------------
    # Build masked image for detection
    # ------------------------------------------------------------
    if polygon is not None:
        area_mask = polygon_to_mask(image_bgr.shape, polygon)
        work_img = cv2.bitwise_and(image_bgr, image_bgr, mask=area_mask)
        hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)

    elif roi_box is not None:
        x, y, w, h = roi_box
        roi_img = image_bgr[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        area_mask = None

    else:
        x, y, w, h = 0, 0, w_full, h_full
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        area_mask = None

    # ------------------------------------------------------------
    # Threshold red
    # ------------------------------------------------------------
    red_ranges = [
        (np.array([h_low1, s_min, v_min]), np.array([h_high1, 255, 255])),
        (np.array([h_low2, s_min, v_min]), np.array([h_high2, 255, 255])),
    ]

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in red_ranges:
        red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lo, hi))

    # If using polygon, enforce polygon mask explicitly
    if polygon is not None:
        red_mask = cv2.bitwise_and(red_mask, area_mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    if close_iters > 0:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    if open_iters > 0:
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k, iterations=open_iters)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ------------------------------------------------------------
    # Compute centers in FULL-IMAGE coordinates
    # ------------------------------------------------------------
    centers = []
    areas = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # ROI contour coordinates are local to cropped ROI
        if polygon is None and roi_box is not None:
            cx += x
            cy += y

        centers.append((cx, cy))
        areas.append(area)

    # ------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------
    if _show_debug_here:
        dbg = image_bgr.copy()

        if roi_box is not None:
            rx, ry, rw, rh = roi_box
            cv2.rectangle(dbg, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

        if polygon is not None:
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(dbg, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        for (cx, cy) in centers:
            cv2.circle(dbg, (int(cx), int(cy)), 6, (0, 255, 255), -1)

        print(f"[DEBUG] raw contours={len(contours)}, kept after min_area={len(centers)}")
        if len(areas) > 0:
            print("[DEBUG] areas (kept):", sorted([round(a, 1) for a in areas], reverse=True)[:10])

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Red mask")
        plt.imshow(red_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Detections overlay")
        plt.imshow(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    if len(centers) == 0:
        raise ValueError("No red markers detected inside selected search area (after filtering).")

    centers = merge_close_points(centers, merge_dist)

    # ------------------------------------------------------------
    # Fallback pass: do it at most once
    # ------------------------------------------------------------
    if (len(centers) < expected_markers) and (not _did_fallback):
        return detect_red_markers_in_roi(
            image_bgr,
            use_roi=use_roi,
            expected_markers=expected_markers,

            roi_polygon=polygon,
            use_polygon=use_polygon,
            select_polygon_fn=select_polygon_fn,
            load_polygon_fn=load_polygon_fn,
            save_polygon_fn=save_polygon_fn,

            min_area=max(1.0, float(min_area) * 0.5),
            merge_dist=merge_dist,
            morph_kernel=(3, 3),
            close_iters=0,
            open_iters=1,
            s_min=max(15, int(s_min * 0.5)),
            v_min=max(15, int(v_min * 0.5)),
            h_low1=h_low1, h_high1=h_high1,
            h_low2=h_low2, h_high2=h_high2,
            allow_two_markers_when_expected_three=allow_two_markers_when_expected_three,
            show_debug=show_debug,
            _did_fallback=True,
        )

    # Sort along beam (y then x)
    centers.sort(key=lambda p: (p[1], p[0]))

    # If more than expected (3-case), pick base, middle-near-midpoint, tip
    if expected_markers == 3 and len(centers) > 3:
        base = centers[-1]
        tip = centers[0]
        y_mid = 0.5 * (base[1] + tip[1])
        middle = min(centers[1:-1], key=lambda p: abs(p[1] - y_mid))
        centers = [tip, middle, base]
        centers.sort(key=lambda p: (p[1], p[0]))
    else:
        centers = centers[:expected_markers]

    # ------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------
    if expected_markers == 3:
        if len(centers) == 3:
            tip_px, mag_start_px, base_px = centers[0], centers[1], centers[2]
            return base_px, mag_start_px, tip_px, search_region

        if len(centers) == 2 and allow_two_markers_when_expected_three:
            tip_px, base_px = centers[0], centers[1]
            mag_start_px = None
            return base_px, mag_start_px, tip_px, search_region

        raise ValueError(
            f"Expected 3 markers but detected {len(centers)} after fallback; "
            "set allow_two_markers_when_expected_three=True to accept base+tip only."
        )

    return centers, search_region
def image_points_to_mm(points, H_img_to_mm):
    pts = np.array(points, dtype=np.float32)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    pts_mm_h = (H_img_to_mm @ pts_h.T).T
    pts_mm = pts_mm_h[:, :2] / pts_mm_h[:, 2:3]
    return [tuple(p) for p in pts_mm]


import json
import os

POLYGON_CONFIG_FILE = "custom_area.json"

def save_polygon(points, path=POLYGON_CONFIG_FILE):
    data = {"points": [[int(x), int(y)] for x, y in points]}
    with open(path, "w") as f:
        json.dump(data, f)

def load_polygon(path=POLYGON_CONFIG_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return [(int(x), int(y)) for x, y in data["points"]]
def select_polygon_interactive(image, window_name="Select custom area"):
    display = image.copy()
    points = []

    def redraw():
        temp = image.copy()
        for i, pt in enumerate(points):
            cv2.circle(temp, pt, 4, (0, 255, 0), -1)
            if i > 0:
                cv2.line(temp, points[i - 1], pt, (0, 255, 0), 2)
        if len(points) >= 3:
            cv2.line(temp, points[-1], points[0], (255, 0, 0), 1)
        cv2.imshow(window_name, temp)

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
        elif key == ord("u"):
            if points:
                points.pop()
                redraw()
        elif key in (27, ord("q")):
            points = None
            break

    cv2.destroyWindow(window_name)
    return points
def measure_beam_and_tip_lengths_mm_with_checkerboard(
    image_filename="focused_image.jpg",
    use_roi=True,
    show=False,
    show_debug_markers=False,   # pass-through to detector
):
    # Step 1: capture and load
    img_file = new_capture(filename=image_filename)
    image = cv2.imread(img_file)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {img_file}")

    # Step 2: homography from checkerboard
    H_img_to_mm = compute_checkerboard_homography(image)

    # Step 3: depth scale (checkerboard plane -> beam plane)
    z_board = float(CAMERA_TO_CHECKERBOARD_MM)
    z_beam  = z_board + float(BEAM_RELATIVE_Z_OFFSET_MM)
    if z_beam <= 0:
        raise ValueError("Invalid geometry: z_beam must be > 0.")
    depth_scale = z_beam / z_board

    base_px, mag_start_px, tip_px, search_region = detect_red_markers_in_roi(
        image,
        use_roi=True,
        use_polygon=True,
        load_polygon_fn=load_polygon,
        save_polygon_fn=save_polygon,
        select_polygon_fn=select_polygon_interactive,
        expected_markers=3,
        show_debug=True,
    )

    # Step 5: pixel -> mm on checkerboard plane -> scale to beam plane
    # Only convert points that exist (mag_start may be None)
    px_points = [base_px, tip_px] if mag_start_px is None else [base_px, mag_start_px, tip_px]
    mm_points_board = image_points_to_mm(px_points, H_img_to_mm)

    if mag_start_px is None:
        base_mm_board, tip_mm_board = mm_points_board
        base_mm = np.array(base_mm_board, dtype=np.float32) * depth_scale
        tip_mm  = np.array(tip_mm_board,  dtype=np.float32) * depth_scale
        mag_mm  = None
    else:
        base_mm_board, mag_mm_board, tip_mm_board = mm_points_board
        base_mm = np.array(base_mm_board, dtype=np.float32) * depth_scale
        mag_mm  = np.array(mag_mm_board,  dtype=np.float32) * depth_scale
        tip_mm  = np.array(tip_mm_board,  dtype=np.float32) * depth_scale

    # Step 6: lengths (mm) + (px)
    base_px_arr = np.array(base_px, dtype=np.float32)
    tip_px_arr  = np.array(tip_px,  dtype=np.float32)

    total_len_mm = float(np.linalg.norm(tip_mm - base_mm))
    total_len_px = float(np.linalg.norm(tip_px_arr - base_px_arr))

    if mag_start_px is None:
        wire_len_mm = None
        mag_len_mm  = None
        wire_len_px = None
        mag_len_px  = None

        print(f"[Beam] wire length:     N/A (middle marker not detected)")
        print(f"[Beam] magnetic length: N/A (middle marker not detected)")
        print(f"[Beam] total length:    {total_len_mm:.2f} mm ({total_len_px:.1f} px)")
    else:
        mag_px_arr = np.array(mag_start_px, dtype=np.float32)

        wire_len_mm = float(np.linalg.norm(mag_mm - base_mm))
        mag_len_mm  = float(np.linalg.norm(tip_mm - mag_mm))

        wire_len_px = float(np.linalg.norm(mag_px_arr - base_px_arr))
        mag_len_px  = float(np.linalg.norm(tip_px_arr - mag_px_arr))

        print(f"[Beam] wire length:     {wire_len_mm:.2f} mm ({wire_len_px:.1f} px)")
        print(f"[Beam] magnetic length: {mag_len_mm:.2f} mm ({mag_len_px:.1f} px)")
        print(f"[Beam] total length:    {total_len_mm:.2f} mm ({total_len_px:.1f} px)")

    # Step 7: visualisation (optional)
    if show:
        vis = image.copy()

    if search_region is not None:
        if isinstance(search_region, tuple) and len(search_region) == 4:
            x, y, w, h = search_region
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        else:
            pts = np.array(search_region, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        # draw base & tip always
        cv2.circle(vis, (int(base_px[0]), int(base_px[1])), 7, (255, 0, 0), -1)   # base = blue
        cv2.circle(vis, (int(tip_px[0]),  int(tip_px[1])),  7, (0, 0, 255), -1)   # tip = red

        # draw middle + segments only if present
        if mag_start_px is not None:
            cv2.circle(vis, (int(mag_start_px[0]), int(mag_start_px[1])), 7, (0, 255, 255), -1)  # mag start = yellow

            cv2.line(vis,
                     (int(base_px[0]), int(base_px[1])),
                     (int(mag_start_px[0]), int(mag_start_px[1])),
                     (255, 255, 255), 2)
            cv2.line(vis,
                     (int(mag_start_px[0]), int(mag_start_px[1])),
                     (int(tip_px[0]), int(tip_px[1])),
                     (0, 255, 0), 2)

            cv2.putText(vis, f"wire: {wire_len_mm:.1f} mm", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"mag:  {mag_len_mm:.1f} mm", (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"tot:  {total_len_mm:.1f} mm", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Only total available
            cv2.line(vis,
                     (int(base_px[0]), int(base_px[1])),
                     (int(tip_px[0]), int(tip_px[1])),
                     (255, 255, 255), 2)
            cv2.putText(vis, f"tot: {total_len_mm:.1f} mm", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        plt.figure(figsize=(7, 7))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Wire + magnetic tip length measurement")
        plt.axis("off")
        plt.show()

    # Return dict: keep keys stable; use None where mag is missing
    out = {
        "image_file": img_file,
        "H_img_to_mm": H_img_to_mm,
        "depth_scale": depth_scale,
        # "roi_box": roi_box,

        "base_px": tuple(base_px),
        "mag_start_px": None if mag_start_px is None else tuple(mag_start_px),
        "tip_px": tuple(tip_px),

        "base_mm": (float(base_mm[0]), float(base_mm[1])),
        "mag_start_mm": None if mag_mm is None else (float(mag_mm[0]), float(mag_mm[1])),
        "tip_mm": (float(tip_mm[0]), float(tip_mm[1])),

        "wire_length_mm": wire_len_mm,
        "mag_length_mm": mag_len_mm,
        "total_length_mm": total_len_mm,

        "wire_length_px": wire_len_px,
        "mag_length_px": mag_len_px,
        "total_length_px": total_len_px,
    }
    return out





if __name__ == "__main__":
    result = measure_beam_and_tip_lengths_mm_with_checkerboard(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        show_debug_markers=True,   # shows mask/overlay from detector
    )

    print("\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(result[""])


    # print("\nResult:")
    # for k, v in result.items():
    #     print(f"  {k}: {v}")
    # print("Length is:  ",result["length_mm"])
