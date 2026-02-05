import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER CONFIG
# ============================================================

# IMPORTANT: INNER corners, not squares.
# If your board is "8x6 squares", try INNER_CORNERS = (7, 5).
INNER_CORNERS = (6, 8)         # (cols, rows) inner corners
SQUARE_SIZE_MM = 5.0           # your checker square size in mm

BEAM_Z_OFFSET_MM = 40.0        # beam plane is +40 mm above checkerboard plane

CALIB_FILE = "camera_calib.json"
ROI_CONFIG_FILE = "red_roi_box.json"


# ============================================================
# CAPTURE (reuse your capture if you want)
# ============================================================

# def new_capture(filename="focused_image.jpg", cam_index=0, backend=cv2.CAP_V4L2, warmup_frames=10):
#     cap = cv2.VideoCapture(cam_index, backend)
#     if not cap.isOpened():
#         raise RuntimeError("Cannot open camera")
#     for _ in range(warmup_frames):
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
                warmup_frames=15,
                exposure=50.0,     # try 200..5000 initially
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
# ============================================================
# ROI helpers (from your code, shortened)
# ============================================================

def load_roi_box(path=ROI_CONFIG_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))

def save_roi_box(box, path=ROI_CONFIG_FILE):
    data = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2]), "h": int(box[3])}
    with open(path, "w") as f:
        json.dump(data, f)

def select_roi_interactive(image, window_name="Select beam ROI"):
    img_copy = image.copy()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    roi = cv2.selectROI(window_name, img_copy, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (x, y, w, h)


# ============================================================
# Red marker detection (reuse yours if you prefer)
# This is a simpler version; you can plug yours in directly.
# Expected 3 markers: base, mag_start, tip
# ============================================================

def detect_red_markers_in_roi_simple(image_bgr, use_roi=True, expected_markers=3, show_debug=False):
    h_full, w_full = image_bgr.shape[:2]
    roi_box = None

    if use_roi:
        roi_box = load_roi_box()
        if roi_box is None:
            roi_box = select_roi_interactive(image_bgr)
            if roi_box is None:
                raise RuntimeError("No ROI selected.")
            save_roi_box(roi_box)

    if roi_box is not None:
        x, y, w, h = roi_box
        roi = image_bgr[y:y+h, x:x+w]
    else:
        x, y, w, h = 0, 0, w_full, h_full
        roi = image_bgr

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # red wraps hue
    mask1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2.0:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = (M["m10"] / M["m00"]) + x
        cy = (M["m01"] / M["m00"]) + y
        centers.append((float(cx), float(cy)))

    if len(centers) < 2:
        raise RuntimeError("Not enough red markers detected.")

    # Sort along beam direction in image (y then x). Adjust if your beam is horizontal.
    centers.sort(key=lambda p: (p[1], p[0]))

    # For 3 markers, assume [tip, mag_start, base] after sorting by y.
    if expected_markers == 3:
        if len(centers) < 3:
            # allow base+tip only
            tip_px = centers[0]
            base_px = centers[-1]
            mag_px = None
        else:
            tip_px, mag_px, base_px = centers[0], centers[1], centers[2]
        return base_px, mag_px, tip_px, roi_box

    return centers, roi_box


# ============================================================
# Checkerboard detection & camera calibration
# ============================================================

def find_checkerboard_corners(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, INNER_CORNERS, flags)

    if not found:
        # Try SB if available
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray, INNER_CORNERS,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
    if not found:
        raise RuntimeError("Checkerboard not found.")

    # Subpixel refine (works for both classic and SB output)
    if corners is not None and corners.dtype != np.float32:
        corners = corners.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)

def make_checkerboard_object_points():
    cols, rows = INNER_CORNERS
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] *= float(SQUARE_SIZE_MM)
    # Z=0 on checkerboard plane
    return obj

def save_calibration(K, dist, image_size, path=CALIB_FILE):
    data = {
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "image_size": list(image_size),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved calibration to {path}")

def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    image_size = tuple(data["image_size"])
    return K, dist, image_size

def calibrate_camera_from_images(image_files, save_path=CALIB_FILE, show=False):
    objp = make_checkerboard_object_points()

    objpoints = []
    imgpoints = []

    image_size = None

    for fp in image_files:
        img = cv2.imread(fp)
        if img is None:
            continue
        if image_size is None:
            h, w = img.shape[:2]
            image_size = (w, h)

        corners = find_checkerboard_corners(img)
        objpoints.append(objp.copy())
        imgpoints.append(corners.astype(np.float32))

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, INNER_CORNERS, corners.reshape(-1, 1, 2), True)
            plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(os.path.basename(fp))
            plt.axis("off")
            plt.show()

    if len(objpoints) < 8:
        raise RuntimeError("Need more images for calibration (aim for 10–20 good views).")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    print(f"[OK] calib reproj RMS: {ret:.4f}")
    save_calibration(K, dist, image_size, save_path)
    return K, dist, image_size


# ============================================================
# Core geometry: pixel -> (X,Y) on a plane at Z = plane_z_mm in board coords
# ============================================================

def pixel_to_plane_xy_mm(u, v, K, dist, rvec, tvec, plane_z_mm):
    """
    Board coordinate frame:
      - Checkerboard lies on Z=0 plane
      - X,Y in mm along the board
      - plane_z_mm is height above board plane in the same direction as +Z

    Steps:
      1) undistort pixel to normalized ray in camera coords
      2) transform ray into board coords
      3) intersect with plane Z = plane_z_mm
    """
    # undistort pixel -> normalized camera coordinates
    pts = np.array([[[u, v]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist)  # returns normalized coords (x,y)
    x, y = und[0, 0, 0], und[0, 0, 1]
    ray_c = np.array([x, y, 1.0], dtype=np.float64)  # camera ray direction

    # board->camera transform
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    # camera center in board coords: Cb = -R^T t
    Rt = R.T
    Cb = -Rt @ t

    # ray direction in board coords: db = R^T * ray_c
    db = Rt @ ray_c

    # intersect Cb + s*db with Z = plane_z_mm
    if abs(db[2]) < 1e-9:
        raise RuntimeError("Ray parallel to plane (db_z ~ 0).")

    s = (plane_z_mm - Cb[2]) / db[2]
    P = Cb + s * db  # in board coords (mm)

    return float(P[0]), float(P[1])


# ============================================================
# Measurement: wire length and magnet length
# ============================================================

def measure_lengths_mm(
    image_filename="focused_image.jpg",
    use_roi=True,
    show=False,
    cam_index=0
):
    # capture
    img_path = new_capture(image_filename, cam_index=cam_index)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # load calibration
    calib = load_calibration(CALIB_FILE)
    if calib is None:
        raise RuntimeError(
            f"No camera calibration found at {CALIB_FILE}.\n"
            "Calibrate first using calibrate_camera_from_images([...])."
        )
    K, dist, _ = calib

    # detect checkerboard corners
    corners_img = find_checkerboard_corners(img)  # (N,2)
    objp = make_checkerboard_object_points()      # (N,3)

    # solve board pose
    ok, rvec, tvec = cv2.solvePnP(
        objp.astype(np.float64),
        corners_img.astype(np.float64),
        K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed.")

    # detect red markers
    base_px, mag_px, tip_px, roi_box = detect_red_markers_in_roi_simple(
        img, use_roi=use_roi, expected_markers=3, show_debug=False
    )

    # convert marker pixels -> mm on beam plane (Z = +BEAM_Z_OFFSET_MM)
    base_xy = pixel_to_plane_xy_mm(base_px[0], base_px[1], K, dist, rvec, tvec, BEAM_Z_OFFSET_MM)
    tip_xy  = pixel_to_plane_xy_mm(tip_px[0],  tip_px[1],  K, dist, rvec, tvec, BEAM_Z_OFFSET_MM)

    base_mm = np.array(base_xy, dtype=np.float64)
    tip_mm  = np.array(tip_xy,  dtype=np.float64)

    # magnet point (optional)
    mag_mm = None
    if mag_px is not None:
        mag_xy = pixel_to_plane_xy_mm(mag_px[0], mag_px[1], K, dist, rvec, tvec, BEAM_Z_OFFSET_MM)
        mag_mm = np.array(mag_xy, dtype=np.float64)

    # lengths (mm)
    total_len_mm = float(np.linalg.norm(tip_mm - base_mm))
    wire_len_mm = None
    mag_len_mm = None
    if mag_mm is not None:
        wire_len_mm = float(np.linalg.norm(mag_mm - base_mm))
        mag_len_mm  = float(np.linalg.norm(tip_mm - mag_mm))

    # ============================================================
    # WORLD COORDS (meters) + CENTERLINE ANGLE
    # ============================================================

    BASE_WORLD_M = np.array([0.8581328220229531, -0.7055298925316631], dtype=np.float64)

    # Flip board X so that "further along beam" decreases world X
    A = np.array([[-1.0, 0.0],
                [ 0.0, 1.0]], dtype=np.float64)

    base_world_m = BASE_WORLD_M
    tip_world_m = mm_to_world_m(base_mm, tip_mm, BASE_WORLD_M, A_2x2=A)
    mag_world_m = None if mag_mm is None else mm_to_world_m(base_mm, mag_mm, BASE_WORLD_M, A_2x2=A)

    # Angle relative to centerline pointing toward -X in world coords
    v_ref = np.array([-1.0, 0.0], dtype=np.float64)
    v_tip = tip_world_m - base_world_m
    angle_deg = signed_angle_deg(v_ref, v_tip)

    print(f"[WORLD] base (m): {base_world_m}")
    print(f"[WORLD] tip  (m): {tip_world_m}")
    if mag_world_m is not None:
        print(f"[WORLD] mag  (m): {mag_world_m}")
    print(f"[ANGLE] centerline→tip: {angle_deg:.3f} deg")

    if mag_mm is not None:
        print(f"[OK] mag(mm):  {mag_mm}")

    print(f"[LEN] total: {total_len_mm:.3f} mm")
    if mag_mm is None:
        print("[LEN] wire: N/A (middle marker missing)")
        print("[LEN] mag:  N/A (middle marker missing)")
    else:
        print(f"[LEN] wire: {wire_len_mm:.3f} mm")
        print(f"[LEN] mag:  {mag_len_mm:.3f} mm")

    print(f"[WORLD] base (m): {base_world_m}")
    print(f"[WORLD] tip  (m): {tip_world_m}")
    if mag_world_m is not None:
        print(f"[WORLD] mag  (m): {mag_world_m}")
    print(f"[ANGLE] centerline→tip: {angle_deg:.3f} deg")

    # visualization (unchanged from yours, but must use wire_len_mm/mag_len_mm names)
    if show:
        vis = img.copy()
        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)

        def draw_pt(p, color, name):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, color, -1)
            cv2.putText(vis, name, (int(p[0])+8, int(p[1])-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        draw_pt(base_px, (255, 0, 0), "base")
        draw_pt(tip_px,  (0, 0, 255), "tip")
        if mag_px is not None:
            draw_pt(mag_px, (0, 255, 255), "mag_start")
            cv2.line(vis, (int(base_px[0]), int(base_px[1])), (int(mag_px[0]), int(mag_px[1])), (255, 255, 255), 2)
            cv2.line(vis, (int(mag_px[0]), int(mag_px[1])), (int(tip_px[0]), int(tip_px[1])), (0, 255, 0), 2)
            cv2.putText(vis, f"wire {wire_len_mm:.2f}mm", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"mag  {mag_len_mm:.2f}mm", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(vis, f"total {total_len_mm:.2f}mm", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"ang {angle_deg:.2f}deg", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        plt.figure(figsize=(7, 7))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Lengths + tip in world frame")
        plt.axis("off")
        plt.show()

    return {
        "base_px": base_px,
        "mag_px": mag_px,
        "tip_px": tip_px,

        "base_mm": tuple(base_mm.tolist()),
        "mag_mm": None if mag_mm is None else tuple(mag_mm.tolist()),
        "tip_mm": tuple(tip_mm.tolist()),

        "wire_length_mm": wire_len_mm,
        "mag_length_mm": mag_len_mm,
        "total_length_mm": total_len_mm,

        "base_world_m": tuple(base_world_m.tolist()),
        "mag_world_m": None if mag_world_m is None else tuple(mag_world_m.tolist()),
        "tip_world_m": tuple(tip_world_m.tolist()),
        "angle_centerline_to_tip_deg": angle_deg,

        "image_file": img_path,
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "plane_z_mm": float(BEAM_Z_OFFSET_MM),
        "A_2x2": A.tolist(),
        "BASE_WORLD_M": BASE_WORLD_M.tolist(),

    }

def signed_angle_deg(v_ref, v):
    v_ref = np.asarray(v_ref, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a1 = np.arctan2(v_ref[1], v_ref[0])
    a2 = np.arctan2(v[1], v[0])
    ang = np.degrees(a2 - a1)
    ang = (ang + 180.0) % 360.0 - 180.0
    return float(ang)
def mm_to_world_m(base_mm, pt_mm, base_world_m, A_2x2=None):
    """
    Converts board mm coordinates to world m coordinates with an optional 2x2 axis transform.

    pt_world = base_world + A * (pt_mm - base_mm)/1000
    """
    base_mm = np.asarray(base_mm, dtype=np.float64)
    pt_mm = np.asarray(pt_mm, dtype=np.float64)
    base_world_m = np.asarray(base_world_m, dtype=np.float64)

    d_board_m = (pt_mm - base_mm) / 1000.0  # (dx, dy) in meters

    if A_2x2 is None:
        A_2x2 = np.eye(2, dtype=np.float64)
    else:
        A_2x2 = np.asarray(A_2x2, dtype=np.float64).reshape(2, 2)

    d_world_m = A_2x2 @ d_board_m
    return base_world_m + d_world_m


def pick_multiple_target_points_world(
    *,
    n_points=None,
    image_filename="focused_image.jpg",
    cam_index=0,
    use_roi=True,
    show_debug=False,
    # world mapping params (same as your measure_lengths_mm)
    BASE_WORLD_M=np.array([0.8581328220229531, -0.7055298925316631], dtype=np.float64),
    A_2x2=np.array([[-1.0, 0.0],
                    [ 0.0, 1.0]], dtype=np.float64),
    plane_z_mm=BEAM_Z_OFFSET_MM,
):
    """
    Captures an image, lets user click multiple points, and converts each click
    to world/global coordinates (meters) using:
      pixel -> board plane (mm) via solvePnP + ray-plane intersection
      board plane (mm) -> world (m) via mm_to_world_m anchored at detected base

    Returns:
      targets_world_m: list of (x_world, y_world) in meters
      debug: dict with image, clicked px, base_px, base_mm, etc.
    """

    # --- 1) capture image ---
    img_path = new_capture(image_filename, cam_index=cam_index)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    # --- 2) load calibration ---
    calib = load_calibration(CALIB_FILE)
    if calib is None:
        raise RuntimeError(
            f"No camera calibration found at {CALIB_FILE}. "
            "Run calibrate_camera_from_images(...) first."
        )
    K, dist, _ = calib

    # --- 3) solve board pose from checkerboard ---
    corners_img = find_checkerboard_corners(img)          # (N,2)
    objp = make_checkerboard_object_points()              # (N,3)

    ok, rvec, tvec = cv2.solvePnP(
        objp.astype(np.float64),
        corners_img.astype(np.float64),
        K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed.")

    # --- 4) detect base marker (needed to anchor world transform) ---
    base_px, mag_px, tip_px, roi_box = detect_red_markers_in_roi_simple(
        img, use_roi=use_roi, expected_markers=3, show_debug=False
    )

    # base on beam plane in board-mm
    base_xy_mm = pixel_to_plane_xy_mm(base_px[0], base_px[1], K, dist, rvec, tvec, plane_z_mm)
    base_mm = np.array(base_xy_mm, dtype=np.float64)

    # --- 5) click points (pixel coords) ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img_rgb)
    if n_points is None:
        plt.title("Click targets (any number), then close window")
        pts = plt.ginput(n=-1, timeout=0)
    else:
        plt.title(f"Click {n_points} targets, then close window")
        pts = plt.ginput(n=n_points, timeout=0)
    plt.close()

    if not pts:
        return [], {
            "image_file": img_path,
            "base_px": base_px,
            "base_mm": base_mm,
            "clicked_px": [],
            "clicked_mm": [],
            "clicked_world_m": [],
        }

    clicked_px = [(float(x), float(y)) for (x, y) in pts]

    # --- 6) convert each click pixel -> board plane mm -> world m ---
    clicked_mm = []
    clicked_world_m = []

    for (u, v) in clicked_px:
        xy_mm = pixel_to_plane_xy_mm(u, v, K, dist, rvec, tvec, plane_z_mm)
        pt_mm = np.array(xy_mm, dtype=np.float64)
        clicked_mm.append(pt_mm)

        pt_world_m = mm_to_world_m(base_mm, pt_mm, BASE_WORLD_M, A_2x2=A_2x2)
        clicked_world_m.append(pt_world_m)

    targets_world_m = [(float(p[0]), float(p[1])) for p in clicked_world_m]

    if show_debug:
        print("[CLICK] base_px:", base_px)
        print("[CLICK] base_mm:", base_mm)
        for i, (px, mm, wm) in enumerate(zip(clicked_px, clicked_mm, clicked_world_m)):
            print(f"  {i:02d} px={px}  mm={mm}  world_m={wm}")

    debug = {
        "image_file": img_path,
        "roi_box": roi_box,
        "base_px": base_px,
        "tip_px": tip_px,
        "base_mm": base_mm,
        "clicked_px": clicked_px,
        "clicked_mm": [tuple(p.tolist()) for p in clicked_mm],
        "clicked_world_m": [tuple(p.tolist()) for p in clicked_world_m],
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
    }

    return targets_world_m, debug

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # 1) Calibration (run once):
    # Put 10–20 checkerboard images in a folder and run:
    #
    # images = [f"calib_imgs/{fn}" for fn in os.listdir("calib_imgs") if fn.lower().endswith((".png",".jpg",".jpeg"))]
    # calibrate_camera_from_images(images, save_path=CALIB_FILE, show=False)
    # #
    # 2) Measurement (daily):
    result = measure_lengths_mm(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        cam_index=0
    )
    print("\nResult dict:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    targets_world_m, dbg = pick_multiple_target_points_world(
        n_points=None,          # or an int
        cam_index=0,
        use_roi=True,
        show_debug=True
    )

    print("World targets (m):", targets_world_m)
    # Example: pick first target
    target_xy_world = np.array(targets_world_m[0])