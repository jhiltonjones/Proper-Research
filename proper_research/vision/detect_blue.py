import numpy as np
import cv2
import os
import json

MANUAL_VESSEL_BOUNDARY_FILE = "manual_vessel_boundaries.json"

# manual_vessel_boundaries.json is NOT vessel-specific data -- it is the BEAM's
# own frame-calibration artifact (base_px/ex_img/ey_img), loaded by
# state_stream.py::NewFrameTipMapper to build the pixel->beam-plane (B) and
# beam->robot (R) transforms the whole online stack uses. Never overwrite it
# from a vessel-boundary click session. Vessel lumen geometry goes to its own
# file instead -- see create_vessel_lumen_in_robot_frame / VESSEL_LUMEN_FILE.
BEAM_FRAME_CALIBRATION_FILE = "/home/jack/Proper-Research/manual_vessel_boundaries.json"
CALIBRATION_POINTS_FILE = "/home/jack/Proper-Research/calibration_points.json"
VESSEL_LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame.json"

# Must match StateStreamConfig's defaults (state_stream.py) exactly, or the
# vessel geometry and the live-tracked beam tip land in different frames.
# 2026-09-14: re-calibrated against a 1cm-square checkerboard grid (diagonal
# baseline, 67 corner-to-corner spacings cross-checked) -- see
# calibration_points.json's "source" field. Was 38.0 (hand-clicked
# two-point estimate); the new value agrees with it to ~0.4% once measured
# via the same (diagonal, not naively-averaged) method, so this is a
# precision refinement, not a correction of a real error.
_DEFAULT_KNOWN_CALIBRATION_DISTANCE_MM = 80.62257748298549
_DEFAULT_SAVED_AXIS_CONVENTION = "image_cartesian"
_DEFAULT_POSITIVE_AXIS_SIGNS = (-1.0, 1.0)
_DEFAULT_BEAM_AXIAL_AXIS_R = (-1.0, 0.0, 0.0)
_DEFAULT_BEAM_PLANE_NORMAL_AXIS_R = (0.0, 0.0, -1.0)
_DEFAULT_T_ROBOT_BEAM_POSE6 = (0.525575, -0.670028, -0.016567, 0.0, -1.5707963, 0.0)
def resample_polyline_by_arclength(points, n_samples=200):
    pts = np.asarray(points, dtype=float)

    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s[-1]

    if total < 1e-12:
        raise ValueError("Polyline length is too small.")

    s_new = np.linspace(0.0, total, n_samples)

    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])

    return [tuple(map(float, p)) for p in np.column_stack([x_new, y_new])]

def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)

def save_manual_vessel_boundaries_with_frame(
    left_boundary_px,
    right_boundary_px,
    centerline_px,
    base_px,
    ref_px,
    path,
):
    base_px = np.asarray(base_px, float)
    ref_px = np.asarray(ref_px, float)

    ex_img = unit(ref_px - base_px)
    ey_img = np.array([-ex_img[1], ex_img[0]])

    data = {
        "base_px": base_px.tolist(),
        "ref_px": ref_px.tolist(),
        "ex_img": ex_img.tolist(),
        "ey_img": ey_img.tolist(),
        "left_boundary_px": [[float(x), float(y)] for x, y in left_boundary_px],
        "right_boundary_px": [[float(x), float(y)] for x, y in right_boundary_px],
        "centerline_px": [[float(x), float(y)] for x, y in centerline_px],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_manual_vessel_boundaries_with_frame(path):
    with open(path, "r") as f:
        data = json.load(f)

    return {
        "base_px": np.asarray(data["base_px"], float),
        "ref_px": np.asarray(data["ref_px"], float),
        "ex_img": np.asarray(data["ex_img"], float),
        "ey_img": np.asarray(data["ey_img"], float),
        "left_boundary_px": [tuple(map(float, p)) for p in data["left_boundary_px"]],
        "right_boundary_px": [tuple(map(float, p)) for p in data["right_boundary_px"]],
        "centerline_px": [tuple(map(float, p)) for p in data.get("centerline_px", [])],
    }

def image_points_to_base_local(points_px, base_px, ex_img, ey_img, mm_per_pixel, z_mm=0.0):
    """
    Converts image pixel points into local metric coordinates.

    local x = projection along clicked reference direction
    local y = projection perpendicular to clicked reference direction
    base_px becomes exactly (0, 0)
    """
    pts = np.asarray(points_px, float)
    base_px = np.asarray(base_px, float)
    ex_img = unit(ex_img)
    ey_img = unit(ey_img)

    dpx = pts - base_px[None, :]

    x_mm = dpx @ ex_img * mm_per_pixel
    y_mm = dpx @ ey_img * mm_per_pixel

    z_mm_arr = np.full_like(x_mm, float(z_mm))
    return np.column_stack([x_mm, y_mm, z_mm_arr])


def build_lumen_from_manual_boundaries_with_frame(
    left_boundary_px,
    right_boundary_px,
    base_px,
    ex_img,
    ey_img,
    mm_per_pixel,
    z_mm=0.0,
):
    left = np.asarray(left_boundary_px, float)
    right = np.asarray(right_boundary_px, float)

    if len(left) != len(right):
        raise ValueError("Left and right boundaries must have same number of samples.")

    center_px = 0.5 * (left + right)
    radius_px = 0.5 * np.linalg.norm(right - left, axis=1)

    lumen_C_mm = image_points_to_base_local(
        center_px,
        base_px=base_px,
        ex_img=ex_img,
        ey_img=ey_img,
        mm_per_pixel=mm_per_pixel,
        z_mm=z_mm,
    )

    lumen_R_mm = radius_px * mm_per_pixel

    lumen_C_m = lumen_C_mm / 1000.0
    lumen_R_m = lumen_R_mm / 1000.0

    return lumen_C_m, lumen_R_m, lumen_C_mm, lumen_R_mm


def create_vessel_lumen_in_robot_frame(
    left_boundary_px,
    right_boundary_px,
    *,
    manual_boundary_path=BEAM_FRAME_CALIBRATION_FILE,
    calibration_points_path=CALIBRATION_POINTS_FILE,
    known_calibration_distance_mm=_DEFAULT_KNOWN_CALIBRATION_DISTANCE_MM,
    saved_axis_convention=_DEFAULT_SAVED_AXIS_CONVENTION,
    positive_axis_signs=_DEFAULT_POSITIVE_AXIS_SIGNS,
    beam_axial_axis_R=_DEFAULT_BEAM_AXIAL_AXIS_R,
    beam_plane_normal_axis_R=_DEFAULT_BEAM_PLANE_NORMAL_AXIS_R,
    T_robot_beam_pose6=_DEFAULT_T_ROBOT_BEAM_POSE6,
):
    """Convert clicked vessel-wall pixels into a lumen (centreline + radius)
    in the ROBOT frame R -- the same frame the offline planner's ``pivot_point``
    / forward kinematics use (see planner-legacy-frame-lock).

    This reuses the EXACT SAME calibration chain ``state_stream.py``'s
    ``NewFrameTipMapper`` uses for the live tip: pixels -> beam plane B (via
    ``PlanarPixelCalibration``, built from the EXISTING beam-frame calibration
    in ``manual_boundary_path`` + the pixel scale in ``calibration_points_path``)
    -> robot frame R (via the fixed ``T_R_B`` rigid transform). Using anything
    else here (a fresh local frame from new clicks, a hand-rolled px->mm
    scale, ...) would put the vessel geometry in a DIFFERENT frame than the
    live-tracked beam tip and every constraint the planner already enforces
    (magnet-to-base exclusion, Z floor, ...) -- silently wrong, not just
    imprecise.

    ``left_boundary_px``/``right_boundary_px`` must be paired, equal-length,
    full-frame pixel coordinates (e.g. the output of
    ``create_vessel_geometry_from_clicks`` / ``centerline_to_offset_boundaries``).
    Returns ``(lumen_C_m, lumen_R_m, provenance)``: ``lumen_C_m`` is (N, 3),
    ``lumen_R_m`` is (N,), both in metres in frame R.
    """
    from proper_research.hardware.robotics_frame_measurement_validation import (
        FrameTransform,
        PlanarPixelCalibration,
        beam_frame_rotation_from_axes,
        compute_metres_per_pixel,
    )

    left_px = np.asarray(left_boundary_px, dtype=float)
    right_px = np.asarray(right_boundary_px, dtype=float)
    if left_px.shape != right_px.shape:
        raise ValueError(
            f"left/right boundary must be paired and equal length; got "
            f"{left_px.shape} vs {right_px.shape}."
        )

    manual = load_manual_vessel_boundaries_with_frame(manual_boundary_path)
    with open(calibration_points_path, "r", encoding="utf-8") as handle:
        cal_points = json.load(handle)["points_px"]
    metres_per_pixel = compute_metres_per_pixel(
        cal_points[0], cal_points[1], known_calibration_distance_mm
    )
    calibration = PlanarPixelCalibration.from_basis_scale(
        origin_px=manual["base_px"],
        ex_saved=manual["ex_img"],
        ey_saved=manual["ey_img"],
        metres_per_pixel=metres_per_pixel,
        saved_axis_convention=saved_axis_convention,
        positive_axis_signs=positive_axis_signs,
    )

    rotation = beam_frame_rotation_from_axes(beam_axial_axis_R, beam_plane_normal_axis_R)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(T_robot_beam_pose6, dtype=float)[:3]
    T_R_B = FrameTransform("R", "B", matrix)

    left_B = calibration.pixels_to_beam(left_px)   # (N, 3), z=0 by construction
    right_B = calibration.pixels_to_beam(right_px)
    left_R = T_R_B.apply_points(left_B)
    right_R = T_R_B.apply_points(right_B)

    lumen_C_m = 0.5 * (left_R + right_R)
    lumen_R_m = 0.5 * np.linalg.norm(right_R - left_R, axis=1)

    provenance = {
        "manual_boundary_path": str(manual_boundary_path),
        "calibration_points_path": str(calibration_points_path),
        "metres_per_pixel": float(metres_per_pixel),
        "known_calibration_distance_mm": float(known_calibration_distance_mm),
        "saved_axis_convention": saved_axis_convention,
        "positive_axis_signs": list(positive_axis_signs),
        "beam_axial_axis_R": list(beam_axial_axis_R),
        "beam_plane_normal_axis_R": list(beam_plane_normal_axis_R),
        "T_robot_beam_pose6": list(T_robot_beam_pose6),
    }
    return lumen_C_m, lumen_R_m, provenance


def save_vessel_lumen_robot_frame(lumen_C_m, lumen_R_m, provenance, path=VESSEL_LUMEN_FILE):
    """Save a lumen (centreline + radius, robot frame R) for the offline
    planner. Distinct from ``save_manual_vessel_boundaries_with_frame`` --
    that one is pixel-space + the beam's OWN frame calibration and must not
    be conflated with this, the actual planner-ready vessel geometry."""
    data = {
        "frame": "R",
        "lumen_C_m": np.asarray(lumen_C_m, dtype=float).tolist(),
        "lumen_R_m": np.asarray(lumen_R_m, dtype=float).tolist(),
        "provenance": provenance,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_vessel_lumen_robot_frame(path=VESSEL_LUMEN_FILE):
    with open(path, "r") as f:
        data = json.load(f)
    if data.get("frame") != "R":
        raise ValueError(
            f"{path} is not a robot-frame vessel lumen file (frame={data.get('frame')!r})."
        )
    lumen_C_m = np.asarray(data["lumen_C_m"], dtype=float)
    lumen_R_m = np.asarray(data["lumen_R_m"], dtype=float)
    return lumen_C_m, lumen_R_m, data.get("provenance", {})


def resample_polyline_by_arclength_preserve_vertices(points, samples_per_segment=20):
    pts = np.asarray(points, dtype=float)

    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")

    out = [pts[0]]

    for i in range(len(pts) - 1):
        p = pts[i]
        q = pts[i + 1]

        for j in range(1, samples_per_segment + 1):
            a = j / samples_per_segment
            out.append((1.0 - a) * p + a * q)

    return [tuple(map(float, p)) for p in out]
def load_roi_box(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])

def resample_boundaries_by_centerline_normals(
    left_points_px,
    right_points_px,
    n_samples=200,
    dense_samples=1000,
):
    """
    Build continuous non-overlapping vessel walls using local normals of the
    vessel centerline.

    This is better than using one global 90-degree direction, because the vessel
    is curved. Each wall point is paired using the local centerline normal.
    """

    left_dense = np.asarray(
        resample_polyline_by_arclength(left_points_px, n_samples=dense_samples),
        float,
    )
    right_dense = np.asarray(
        resample_polyline_by_arclength(right_points_px, n_samples=dense_samples),
        float,
    )

    # Provisional centerline from equal fractional progress along each wall
    center_dense = 0.5 * (left_dense + right_dense)

    center = np.asarray(
        resample_polyline_by_arclength(center_dense, n_samples=n_samples),
        float,
    )

    # Centerline tangents
    tangent = np.zeros_like(center)
    tangent[1:-1] = center[2:] - center[:-2]
    tangent[0] = center[1] - center[0]
    tangent[-1] = center[-1] - center[-2]

    tangent = np.array([unit(t) for t in tangent])

    # Local normal at each centerline point
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    left_out = []
    right_out = []

    for c, t, n in zip(center, tangent, normal):
        # For every dense wall point, compute local coordinates around center point
        dL = left_dense - c[None, :]
        dR = right_dense - c[None, :]

        axial_L = dL @ t
        axial_R = dR @ t

        normal_L = dL @ n
        normal_R = dR @ n

        # Prefer points close to the local normal cross-section
        score_L = np.abs(axial_L)
        score_R = np.abs(axial_R)

        iL = int(np.argmin(score_L))
        iR = int(np.argmin(score_R))

        qL = left_dense[iL]
        qR = right_dense[iR]

        # Preserve the user's actual left/right labels.
        # Do NOT force swap based on normal sign.
        left_out.append(tuple(map(float, qL)))
        right_out.append(tuple(map(float, qR)))

    return left_out, right_out

def line_intersection_2d(p1, d1, p2, d2, eps=1e-12):
    A = np.column_stack([d1, -d2])
    b = p2 - p1

    det = np.linalg.det(A)
    if abs(det) < eps:
        return None

    a, _ = np.linalg.solve(A, b)
    return p1 + a * d1


def centerline_to_offset_boundaries(
    centerline_px,
    radius_px=18.0,
    samples_per_segment=20,
    miter_limit=2.0,
):
    raw = np.asarray(centerline_px, dtype=float)

    if len(raw) < 2:
        raise ValueError("Need at least 2 centerline points.")

    # Dense centerline, but preserve sharp clicked vertices.
    center = np.asarray(
        resample_polyline_by_arclength_preserve_vertices(
            raw,
            samples_per_segment=samples_per_segment,
        ),
        dtype=float,
    )

    n = len(center)
    left = np.zeros_like(center)
    right = np.zeros_like(center)

    seg_t = np.array([unit(center[i + 1] - center[i]) for i in range(n - 1)])
    seg_n = np.column_stack([-seg_t[:, 1], seg_t[:, 0]])

    for i in range(n):
        if i == 0:
            normal = seg_n[0]
            left[i] = center[i] - radius_px * normal
            right[i] = center[i] + radius_px * normal

        elif i == n - 1:
            normal = seg_n[-1]
            left[i] = center[i] - radius_px * normal
            right[i] = center[i] + radius_px * normal

        else:
            t0, t1 = seg_t[i - 1], seg_t[i]
            n0, n1 = seg_n[i - 1], seg_n[i]

            for sign, out in [(-1.0, left), (+1.0, right)]:
                p0 = center[i] + sign * radius_px * n0
                p1 = center[i] + sign * radius_px * n1

                q = line_intersection_2d(p0, t0, p1, t1)

                if q is None or np.linalg.norm(q - center[i]) > miter_limit * radius_px:
                    # Safe bevel fallback
                    q = 0.5 * (p0 + p1)

                out[i] = q

    return (
        [tuple(map(float, p)) for p in left],
        [tuple(map(float, p)) for p in right],
        [tuple(map(float, p)) for p in center],
    )
def create_vessel_geometry_from_clicks(
    center_points_roi,
    left_points_roi,
    right_points_roi,
    to_full_px,
    radius_px=16.0,
    samples_per_segment=30,
    boundary_samples=60,   # changed from 200 to 60
    smooth_iter=3,
    mode="auto",
):
    """
    Creates left boundary, right boundary, and centreline from either:

    1. Clicked centreline:
       - center_points_roi must contain at least 2 points.
       - left/right boundaries are generated as offsets.

    2. Clicked left and right boundaries:
       - left_points_roi and right_points_roi must each contain at least 2 points.
       - centreline is computed from paired wall samples.

    mode:
        "auto"       : prefer clicked centreline if available, otherwise use boundaries.
        "centerline" : force centreline-first.
        "boundaries" : force boundary-first.
    """

    if mode not in {"auto", "centerline", "boundaries"}:
        raise ValueError(f"Unknown vessel creation mode: {mode}")

    have_centerline = len(center_points_roi) >= 2
    have_boundaries = len(left_points_roi) >= 2 and len(right_points_roi) >= 2

    if mode == "centerline":
        if not have_centerline:
            raise ValueError("Need at least 2 centreline points for centreline mode.")

    elif mode == "boundaries":
        if not have_boundaries:
            raise ValueError("Need at least 2 left and 2 right boundary points for boundary mode.")

    elif mode == "auto":
        if not have_centerline and not have_boundaries:
            raise ValueError(
                "Need either at least 2 centreline points, or at least 2 left and 2 right boundary points."
            )

    # ------------------------------------------------------------
    # Mode 1: derive boundaries from clicked centreline
    # ------------------------------------------------------------
    if mode == "centerline" or (mode == "auto" and have_centerline):
        centerline_clicked_px = [to_full_px(p) for p in center_points_roi]

        left_boundary_px, right_boundary_px, centerline_px = centerline_to_offset_boundaries(
            centerline_clicked_px,
            radius_px=radius_px,
            samples_per_segment=samples_per_segment,
        )

        creation_mode = "centerline"

    # ------------------------------------------------------------
    # Mode 2: derive centreline from clicked left/right boundaries
    # ------------------------------------------------------------
    else:
        left_clicked_px = [to_full_px(p) for p in left_points_roi]
        right_clicked_px = [to_full_px(p) for p in right_points_roi]

        left_boundary_px, right_boundary_px, centerline_px, width_px = (
            resample_smooth_boundaries_monotonic(
                left_clicked_px,
                right_clicked_px,
                n_samples=boundary_samples,
                smooth_iter=smooth_iter,
            )
        )

        creation_mode = "boundaries"

    # Force final geometry to exactly 60 equally spaced reference points.
    left_boundary_px = resample_polyline_by_arclength(
        left_boundary_px,
        n_samples=boundary_samples,
    )

    right_boundary_px = resample_polyline_by_arclength(
        right_boundary_px,
        n_samples=boundary_samples,
    )

    centerline_px = resample_polyline_by_arclength(
        centerline_px,
        n_samples=boundary_samples,
    )

    return left_boundary_px, right_boundary_px, centerline_px, creation_mode
def draw_centerline_radius_overlay(
    image_bgr,
    centerline_px,
    left_boundary_px,
    right_boundary_px,
    x0=0,
    y0=0,
    radius_px=18.0,
):
    vis = image_bgr.copy()

    offset = np.array([x0, y0], dtype=float)

    center = np.asarray(centerline_px, float) - offset
    left = np.asarray(left_boundary_px, float) - offset
    right = np.asarray(right_boundary_px, float) - offset

    def draw_polyline(pts, color, thickness):
        pts_i = np.round(pts).astype(np.int32)
        for i in range(len(pts_i) - 1):
            cv2.line(vis, tuple(pts_i[i]), tuple(pts_i[i + 1]), color, thickness)

    draw_polyline(center, (255, 255, 0), 2)  # cyan/yellow center
    draw_polyline(left, (0, 255, 0), 2)      # green wall
    draw_polyline(right, (0, 0, 255), 2)     # red wall

    for i in range(0, len(center), 10):
        cv2.line(
            vis,
            tuple(np.round(left[i]).astype(int)),
            tuple(np.round(right[i]).astype(int)),
            (255, 0, 255),
            1,
        )

    cv2.putText(
        vis,
        f"Live overlay: centerline +/- {radius_px:.1f}px | q/ESC=quit",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return vis


def live_camera_centerline_overlay(
    centerline_px,
    left_boundary_px,
    right_boundary_px,
    camera_index=0,
    radius_px=18.0,
    window_name="Live vessel alignment overlay",
):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    print("[LIVE] Move the vessel until it matches the overlay.")
    print("[LIVE] Press q or ESC to close live overlay.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera.")
            break

        vis = draw_centerline_radius_overlay(
            image_bgr=frame,
            centerline_px=centerline_px,
            left_boundary_px=left_boundary_px,
            right_boundary_px=right_boundary_px,
            x0=0,
            y0=0,
            radius_px=radius_px,
        )

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyWindow(window_name)
def draw_manual_vessel_boundaries_with_origin_and_axis(
    image_filename="focused_image.jpg",
    save_path=MANUAL_VESSEL_BOUNDARY_FILE,
    blue_roi_path="blue_roi_box.json",
):
    center_points = []
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    roi_box = load_roi_box(blue_roi_path)

    if roi_box is not None:
        x0, y0, w, h = roi_box
        display = image_bgr[y0:y0 + h, x0:x0 + w].copy()
    else:
        x0, y0 = 0, 0
        display = image_bgr.copy()

    left_points = []
    right_points = []
    base_point = {"pt": None}
    ref_point = {"pt": None}

    current_mode = {"name": "base"}  # base, ref, left, right, center
    vessel_creation_mode = {"name": "auto"}  # auto, centerline, boundaries
    window_name = "Draw vessel boundaries + local frame"

    def to_full_px(p_roi):
        return float(p_roi[0] + x0), float(p_roi[1] + y0)

    def redraw():
        vis = display.copy()

        cv2.putText(
            vis,
            f"Mode: {current_mode['name'].upper()}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            "b=base | a=axis/ref | l=left | r=right | m=center | 1=centerline mode | 2=boundary mode | 0=auto | z=undo | c=clear | s=save | q=quit""b=base | a=axis/ref | l=left | r=right | m=center | 1=centerline mode | 2=boundary mode | 0=auto | z=undo | c=clear | s=save | q=quit",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis,
            f"Vessel creation: {vessel_creation_mode['name'].upper()}",
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )
        if base_point["pt"] is not None:
            bx, by = base_point["pt"]
            cv2.circle(vis, (int(bx), int(by)), 6, (255, 0, 255), -1)
            cv2.putText(vis, "BASE (0,0)", (int(bx) + 8, int(by) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        if ref_point["pt"] is not None:
            rx, ry = ref_point["pt"]
            cv2.circle(vis, (int(rx), int(ry)), 6, (0, 255, 255), -1)
            cv2.putText(vis, "+X REF", (int(rx) + 8, int(ry) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if base_point["pt"] is not None and ref_point["pt"] is not None:
            b = np.asarray(base_point["pt"], float)
            r = np.asarray(ref_point["pt"], float)
            ex = unit(r - b)
            ey = np.array([-ex[1], ex[0]])

            axis_len = 80.0
            x_end = b + axis_len * ex
            y_end = b + axis_len * ey

            cv2.arrowedLine(vis, tuple(np.round(b).astype(int)), tuple(np.round(x_end).astype(int)),
                            (0, 255, 255), 2, tipLength=0.15)
            cv2.arrowedLine(vis, tuple(np.round(b).astype(int)), tuple(np.round(y_end).astype(int)),
                            (255, 0, 255), 2, tipLength=0.15)

            cv2.putText(vis, "+x", tuple(np.round(x_end).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(vis, "+y", tuple(np.round(y_end).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if base_point["pt"] is not None and len(left_points) > 0 and len(right_points) > 0:
            b = np.asarray(base_point["pt"], float).copy()
            start_y = 0.5 * (left_points[0][1] + right_points[0][1])
            b_snap = np.array([b[0], start_y])

            cv2.circle(vis, tuple(np.round(b_snap).astype(int)), 6, (0, 165, 255), 2)
            cv2.putText(
                vis,
                "SNAPPED BASE",
                (int(b_snap[0]) + 8, int(b_snap[1]) + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 165, 255),
                1,
            )
        for pts, color, line_color, label in [
            (left_points, (0, 255, 0), (0, 150, 0), "L"),
            (right_points, (0, 0, 255), (0, 0, 180), "R"),
            (center_points, (255, 255, 0), (200, 200, 0), "C"),
        ]:
            for i, (x, y) in enumerate(pts):
                cv2.circle(vis, (int(x), int(y)), 3, color, -1)
                cv2.putText(vis, label, (int(x) + 4, int(y) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                if i > 0:
                    cv2.line(vis,
                            (int(pts[i - 1][0]), int(pts[i - 1][1])),
                            (int(x), int(y)),
                            line_color,
                            1)

            if len(pts) >= 2:
                try:
                    preview = resample_polyline_by_arclength_preserve_vertices(
                                pts,
                                samples_per_segment=30,
                            )
                    for i in range(len(preview) - 1):
                        p1 = preview[i]
                        p2 = preview[i + 1]
                        cv2.line(
                            vis,
                            (int(round(p1[0])), int(round(p1[1]))),
                            (int(round(p2[0])), int(round(p2[1]))),
                            color,
                            2,
                        )
                except Exception:
                    pass

        cv2.imshow(window_name, vis)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        p = (float(x), float(y))
        mode = current_mode["name"]

        if mode == "base":
            base_point["pt"] = p
        elif mode == "ref":
            ref_point["pt"] = p
        elif mode == "left":
            left_points.append(p)
        elif mode == "right":
            right_points.append(p)
        elif mode == "center":
            center_points.append(p)
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("b"):
            current_mode["name"] = "base"
            redraw()

        elif key == ord("a"):
            current_mode["name"] = "ref"
            redraw()

        elif key == ord("l"):
            current_mode["name"] = "left"
            redraw()

        elif key == ord("r"):
            current_mode["name"] = "right"
            redraw()

        elif key == ord("m"):
            current_mode["name"] = "center"
            redraw()
        elif key == ord("v"):
            try:
                left_boundary_px, right_boundary_px, centerline_px, used_creation_mode = (
                    create_vessel_geometry_from_clicks(
                        center_points_roi=center_points,
                        left_points_roi=left_points,
                        right_points_roi=right_points,
                        to_full_px=to_full_px,
                        radius_px=14.0,
                        samples_per_segment=30,
                        boundary_samples=60,
                        smooth_iter=3,
                        mode=vessel_creation_mode["name"],
                    )
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                print("[INFO] Need either centreline clicks or left/right boundary clicks before live overlay.")
                continue

            print(f"[LIVE] Overlay generated using: {used_creation_mode}")

            live_camera_centerline_overlay(
                centerline_px=centerline_px,
                left_boundary_px=left_boundary_px,
                right_boundary_px=right_boundary_px,
                camera_index=0,
                radius_px=14.0,
            )

            redraw()
        elif key == ord("z"):
            mode = current_mode["name"]
            if mode == "left" and left_points:
                left_points.pop()
            elif mode == "right" and right_points:
                right_points.pop()
            elif mode == "center" and center_points:
                center_points.pop()
            elif mode == "base":
                base_point["pt"] = None
            elif mode == "ref":
                ref_point["pt"] = None
            redraw()

        elif key == ord("c"):
            mode = current_mode["name"]
            if mode == "left":
                left_points.clear()
            elif mode == "right":
                right_points.clear()
            elif mode == "center":
                center_points.clear()
            elif mode == "base":
                base_point["pt"] = None
            elif mode == "ref":
                ref_point["pt"] = None
            redraw()
        elif key == ord("1"):
            vessel_creation_mode["name"] = "centerline"
            print("[INFO] Vessel creation mode: centreline -> offset boundaries")
            redraw()

        elif key == ord("2"):
            vessel_creation_mode["name"] = "boundaries"
            print("[INFO] Vessel creation mode: left/right boundaries -> centreline")
            redraw()

        elif key == ord("0"):
            vessel_creation_mode["name"] = "auto"
            print("[INFO] Vessel creation mode: auto")
            redraw()
        elif key == ord("s"):
            if base_point["pt"] is None:
                print("[WARN] Set base/origin first: press b, then click base.")
                continue

            if ref_point["pt"] is None:
                print("[WARN] Set reference axis first: press a, then click a point along desired +x.")
                continue

            base_px_clicked = to_full_px(base_point["pt"])
            ref_px_clicked = to_full_px(ref_point["pt"])

            try:
                left_boundary_px, right_boundary_px, centerline_px, used_creation_mode = (
                    create_vessel_geometry_from_clicks(
                        center_points_roi=center_points,
                        left_points_roi=left_points,
                        right_points_roi=right_points,
                        to_full_px=to_full_px,
                        radius_px=16.0,
                        samples_per_segment=30,
                        boundary_samples=60,
                        smooth_iter=3,
                        mode=vessel_creation_mode["name"],
                    )
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                print("[INFO] Either press m and click centreline, or press l/r and click both walls.")
                continue

            print(f"[INFO] Created vessel geometry using: {used_creation_mode}")

            show_boundary_preview(
                display,
                left_boundary_px,
                right_boundary_px,
                centerline_px,
                x0=x0,
                y0=y0,
                window_name="Boundary preview before save",
            )

            print("[PREVIEW] qPress y to accept/save, n to reject and continue editing.")

            accept_preview = False
            while True:
                k = cv2.waitKey(20) & 0xFF
                if k == ord("y"):
                    accept_preview = True
                    cv2.destroyWindow("Boundary preview before save")
                    break
                elif k == ord("n") or k == 27:
                    accept_preview = False
                    cv2.destroyWindow("Boundary preview before save")
                    break

            if not accept_preview:
                print("[INFO] Preview rejected. Continue editing.")
                redraw()
                continue
            # snap base onto same image-y as the start of the two vessel walls
            # base_px = snap_base_to_boundary_start_y(
            #     base_px_clicked,
            #     left_boundary_px,
            #     right_boundary_px,
            # )
            base_px = base_px_clicked
            ref_px = ref_px_clicked
            # keep reference point on same vertical offset relative to the snapped base


            save_manual_vessel_boundaries_with_frame(
                left_boundary_px=left_boundary_px,
                right_boundary_px=right_boundary_px,
                centerline_px=centerline_px,
                base_px=base_px,
                ref_px=ref_px,
                path=save_path,
            )

            cv2.destroyWindow(window_name)

            return {
                "left_boundary_px": left_boundary_px,
                "right_boundary_px": right_boundary_px,
                "centerline_px": centerline_px,
                "base_px": base_px,
                "ref_px": ref_px,
            }

        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            print("[INFO] Boundary drawing cancelled.")
            return None
def debug_plot_parametric_pairs_px(
    image_bgr,
    left_boundary_px,
    right_boundary_px,
    out_path="debug_outputs/paired_boundary_debug.png",
):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    left = np.asarray(left_boundary_px, float)
    right = np.asarray(right_boundary_px, float)
    center = 0.5 * (left + right)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    plt.plot(left[:, 0], left[:, 1], "g-o", label="left clicked")
    plt.plot(right[:, 0], right[:, 1], "r-o", label="right clicked")
    plt.plot(center[:, 0], center[:, 1], "c-o", label="computed center")

    for i in range(len(left)):
        plt.plot(
            [left[i, 0], right[i, 0]],
            [left[i, 1], right[i, 1]],
            "m-",
            alpha=0.5,
        )
        plt.text(center[i, 0], center[i, 1], str(i), color="yellow")

    plt.axis("equal")
    plt.legend()
    plt.title("Boundary point pairing used by contact model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[DBG] Saved paired-boundary debug plot: {out_path}")
def snap_base_to_boundary_start_y(base_px, left_boundary_px, right_boundary_px):
    base_px = np.asarray(base_px, float).copy()
    left0 = np.asarray(left_boundary_px[0], float)
    right0 = np.asarray(right_boundary_px[0], float)

    start_y = 0.5 * (left0[1] + right0[1])
    base_px[1] = start_y

    return tuple(base_px)
def chaikin_smooth(points, n_iter=3):
    pts = np.asarray(points, float)
    if len(pts) < 3:
        return pts

    for _ in range(n_iter):
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        new_pts.append(pts[-1])
        pts = np.asarray(new_pts, float)

    return pts


def resample_smooth_boundaries_monotonic(
    left_points_px,
    right_points_px,
    n_samples=200,
    smooth_iter=3,
):
    left_raw = np.asarray(left_points_px, float)
    right_raw = np.asarray(right_points_px, float)

    if len(left_raw) < 2 or len(right_raw) < 2:
        raise ValueError("Need at least 2 points on each boundary.")

    # Make sure both walls run in the same physical direction.
    same_dir_cost = (
        np.linalg.norm(left_raw[0] - right_raw[0])
        + np.linalg.norm(left_raw[-1] - right_raw[-1])
    )

    opposite_dir_cost = (
        np.linalg.norm(left_raw[0] - right_raw[-1])
        + np.linalg.norm(left_raw[-1] - right_raw[0])
    )

    if opposite_dir_cost < same_dir_cost:
        print("[INFO] Reversing right boundary so both walls have same direction.")
        right_raw = right_raw[::-1]

    left_smooth = chaikin_smooth(left_raw, n_iter=smooth_iter)
    right_smooth = chaikin_smooth(right_raw, n_iter=smooth_iter)

    # Fixed-count resampling: both walls now have exactly n_samples points.
    left_resampled = np.asarray(
        resample_polyline_by_arclength(left_smooth, n_samples=n_samples),
        float,
    )

    right_resampled = np.asarray(
        resample_polyline_by_arclength(right_smooth, n_samples=n_samples),
        float,
    )

    if left_resampled.shape != right_resampled.shape:
        raise RuntimeError(
            f"Boundary resampling failed: left={left_resampled.shape}, "
            f"right={right_resampled.shape}"
        )

    center = 0.5 * (left_resampled + right_resampled)
    width = np.linalg.norm(right_resampled - left_resampled, axis=1)

    if np.any(width < 1.0):
        print("[WARN] Some vessel widths are below 1 px. Check clicked boundaries.")

    return (
        [tuple(map(float, p)) for p in left_resampled],
        [tuple(map(float, p)) for p in right_resampled],
        [tuple(map(float, p)) for p in center],
        width,
    )
def resample_polyline_fixed_n(points, n_samples):
    """
    Resample a 2D or 3D polyline to exactly n_samples points,
    equally spaced by cumulative arclength.
    """
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2:
        raise ValueError(f"Expected points with shape (N,D), got {pts.shape}")

    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 points.")

    if int(n_samples) < 2:
        raise ValueError("n_samples must be at least 2.")

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s[-1]

    if total < 1e-12:
        raise ValueError("Polyline length is too small.")

    s_new = np.linspace(0.0, total, int(n_samples))

    out = np.column_stack([
        np.interp(s_new, s, pts[:, d])
        for d in range(pts.shape[1])
    ])

    return out, s_new
def show_boundary_preview(
    image_bgr,
    left_boundary_px,
    right_boundary_px,
    center_px,
    x0=0,
    y0=0,
    window_name="Boundary preview",
):
    vis = image_bgr.copy()

    left = np.asarray(left_boundary_px, float) - np.array([x0, y0], float)
    right = np.asarray(right_boundary_px, float) - np.array([x0, y0], float)
    center = np.asarray(center_px, float) - np.array([x0, y0], float)

    def draw_polyline(pts, color, thickness):
        pts_i = np.round(pts).astype(np.int32)
        for i in range(len(pts_i) - 1):
            cv2.line(vis, tuple(pts_i[i]), tuple(pts_i[i + 1]), color, thickness)

    draw_polyline(left, (0, 255, 0), 2)
    draw_polyline(right, (0, 0, 255), 2)
    draw_polyline(center, (255, 255, 0), 2)

    for i in range(0, len(left), 10):
        if i > 0:
            dprev = center[i] - center[i - 1]
            dnext = center[min(i + 1, len(center) - 1)] - center[i]
            if np.linalg.norm(dprev) > 1e-9 and np.linalg.norm(dnext) > 1e-9:
                cosang = np.dot(unit(dprev), unit(dnext))
                if cosang < 0.5:   # near sharp corner
                    continue

        cv2.line(
            vis,
            tuple(np.round(left[i]).astype(int)),
            tuple(np.round(right[i]).astype(int)),
            (255, 0, 255),
            1,
        )
    cv2.putText(
        vis,
        "Preview: green=left, red=right, cyan=center, magenta=cross-sections",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow(window_name, vis)
    cv2.waitKey(1)
    return vis


def draw_vessel_lumen_for_planner(
    image_filename="focused_image.jpg",
    save_path=VESSEL_LUMEN_FILE,
    blue_roi_path="blue_roi_box.json",
    manual_boundary_path=BEAM_FRAME_CALIBRATION_FILE,
    calibration_points_path=CALIBRATION_POINTS_FILE,
):
    """Click a new vessel's walls (or centreline) on a photo and save a
    planner-ready lumen (centreline + radius, ROBOT frame R) to ``save_path``.

    Unlike ``draw_manual_vessel_boundaries_with_origin_and_axis``, this does
    NOT ask for base/reference-axis clicks -- the frame comes from the
    EXISTING beam-frame calibration (``manual_boundary_path`` +
    ``calibration_points_path``, the same files ``state_stream.py`` uses for
    the live tip), so the new vessel's geometry lands in exactly the frame
    the live tip and the offline planner's forward kinematics already share.
    That file (default ``manual_vessel_boundaries.json``) is READ here, never
    written -- this tool cannot corrupt the beam's own frame calibration.
    """
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    roi_box = load_roi_box(blue_roi_path)
    if roi_box is not None:
        x0, y0, w, h = roi_box
        display = image_bgr[y0:y0 + h, x0:x0 + w].copy()
    else:
        x0, y0 = 0, 0
        display = image_bgr.copy()

    left_points: list = []
    right_points: list = []
    center_points: list = []
    current_mode = {"name": "left"}       # left, right, center
    vessel_creation_mode = {"name": "auto"}
    window_name = "Click vessel walls (or centreline) for the planner"

    def to_full_px(p_roi):
        return float(p_roi[0] + x0), float(p_roi[1] + y0)

    def redraw():
        vis = display.copy()
        cv2.putText(vis, f"Mode: {current_mode['name'].upper()}  (frame: EXISTING beam calibration, not re-clicked)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis,
                    "l=left wall | r=right wall | m=centreline | 1/2/0=creation mode | "
                    "v=live preview | z=undo | c=clear | s=save lumen | q=quit",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(vis, f"Vessel creation: {vessel_creation_mode['name'].upper()}",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        for pts, color, line_color, label in [
            (left_points, (0, 255, 0), (0, 150, 0), "L"),
            (right_points, (0, 0, 255), (0, 0, 180), "R"),
            (center_points, (255, 255, 0), (200, 200, 0), "C"),
        ]:
            for i, (x, y) in enumerate(pts):
                cv2.circle(vis, (int(x), int(y)), 3, color, -1)
                cv2.putText(vis, label, (int(x) + 4, int(y) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                if i > 0:
                    cv2.line(vis, (int(pts[i - 1][0]), int(pts[i - 1][1])), (int(x), int(y)), line_color, 1)
        cv2.imshow(window_name, vis)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        p = (float(x), float(y))
        mode = current_mode["name"]
        if mode == "left":
            left_points.append(p)
        elif mode == "right":
            right_points.append(p)
        elif mode == "center":
            center_points.append(p)
        redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("l"):
            current_mode["name"] = "left"; redraw()
        elif key == ord("r"):
            current_mode["name"] = "right"; redraw()
        elif key == ord("m"):
            current_mode["name"] = "center"; redraw()
        elif key == ord("1"):
            vessel_creation_mode["name"] = "centerline"; redraw()
        elif key == ord("2"):
            vessel_creation_mode["name"] = "boundaries"; redraw()
        elif key == ord("0"):
            vessel_creation_mode["name"] = "auto"; redraw()
        elif key == ord("z"):
            mode = current_mode["name"]
            if mode == "left" and left_points: left_points.pop()
            elif mode == "right" and right_points: right_points.pop()
            elif mode == "center" and center_points: center_points.pop()
            redraw()
        elif key == ord("c"):
            mode = current_mode["name"]
            if mode == "left": left_points.clear()
            elif mode == "right": right_points.clear()
            elif mode == "center": center_points.clear()
            redraw()
        elif key == ord("v"):
            try:
                left_boundary_px, right_boundary_px, centerline_px, used_creation_mode = (
                    create_vessel_geometry_from_clicks(
                        center_points_roi=center_points,
                        left_points_roi=left_points,
                        right_points_roi=right_points,
                        to_full_px=to_full_px,
                        radius_px=14.0,
                        mode=vessel_creation_mode["name"],
                    )
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                continue
            live_camera_centerline_overlay(
                centerline_px=centerline_px, left_boundary_px=left_boundary_px,
                right_boundary_px=right_boundary_px, radius_px=14.0,
            )
            redraw()
        elif key == ord("s"):
            try:
                left_boundary_px, right_boundary_px, centerline_px, used_creation_mode = (
                    create_vessel_geometry_from_clicks(
                        center_points_roi=center_points,
                        left_points_roi=left_points,
                        right_points_roi=right_points,
                        to_full_px=to_full_px,
                        radius_px=16.0,
                        samples_per_segment=30,
                        boundary_samples=60,
                        smooth_iter=3,
                        mode=vessel_creation_mode["name"],
                    )
                )
            except ValueError as e:
                print(f"[WARN] {e}")
                print("[INFO] Either press m and click a centreline, or press l/r and click both walls.")
                continue

            show_boundary_preview(display, left_boundary_px, right_boundary_px, centerline_px,
                                   x0=x0, y0=y0, window_name="Boundary preview before save")
            print("[PREVIEW] Press y to accept/save, n to reject and continue editing.")
            accept_preview = False
            while True:
                k = cv2.waitKey(20) & 0xFF
                if k == ord("y"):
                    accept_preview = True
                    cv2.destroyWindow("Boundary preview before save")
                    break
                elif k == ord("n") or k == 27:
                    cv2.destroyWindow("Boundary preview before save")
                    break
            if not accept_preview:
                print("[INFO] Preview rejected. Continue editing.")
                redraw()
                continue

            lumen_C_m, lumen_R_m, provenance = create_vessel_lumen_in_robot_frame(
                left_boundary_px, right_boundary_px,
                manual_boundary_path=manual_boundary_path,
                calibration_points_path=calibration_points_path,
            )
            written = save_vessel_lumen_robot_frame(lumen_C_m, lumen_R_m, provenance, path=save_path)

            arclength_m = float(np.sum(np.linalg.norm(np.diff(lumen_C_m, axis=0), axis=1)))
            print(f"[SAVED] {written}")
            print(f"[SANITY] {len(lumen_C_m)} samples, frame=R, "
                  f"radius {lumen_R_m.min()*1000:.2f}-{lumen_R_m.max()*1000:.2f} mm, "
                  f"centreline length {arclength_m*1000:.1f} mm, "
                  f"C[0]={lumen_C_m[0]}, C[-1]={lumen_C_m[-1]} (m, robot frame)")
            print("[SANITY] Check these against the vessel's known dimensions before "
                  "handing this to the offline planner.")

            cv2.destroyWindow(window_name)
            return {"lumen_C_m": lumen_C_m, "lumen_R_m": lumen_R_m, "provenance": provenance, "path": written}
        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            print("[INFO] Vessel lumen digitisation cancelled.")
            return None


if __name__ == "__main__":
    import sys

    if "--beam-frame" in sys.argv:
        # Legacy workflow: (re)calibrate the BEAM's own frame. Only run this
        # if the beam-base markers/camera have moved and manual_vessel_boundaries.json
        # itself needs rebuilding -- NOT for digitising a new vessel.
        manual = draw_manual_vessel_boundaries_with_origin_and_axis(
            image_filename="focused_image.jpg",
            save_path=MANUAL_VESSEL_BOUNDARY_FILE,
            blue_roi_path="blue_roi_box.json",
        )
    else:
        # Default: digitise a vessel's lumen for the offline planner, in the
        # robot frame, using the EXISTING beam-frame calibration.
        result = draw_vessel_lumen_for_planner(
            image_filename="focused_image.jpg",
            save_path=VESSEL_LUMEN_FILE,
            blue_roi_path="blue_roi_box.json",
        )