import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from proper_research.vision.vision_w_tangnet import detect_4_red_markers_in_roi
from proper_research.vision.measure_length import new_capture
from proper_research.vision.boundaries import detect_blue_vessel_boundaries, smooth_boundary
RED_ROI_CONFIG_FILE = "red_roi_box.json"
BLUE_ROI_CONFIG_FILE = "blue_roi_box.json"
GREEN_ROI_CONFIG_FILE = "green_roi_box.json"

def save_roi_box(box, path):
    data = {
        "x": int(box[0]),
        "y": int(box[1]),
        "w": int(box[2]),
        "h": int(box[3]),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[INFO] ROI saved to {path}: {data}")


def load_roi_box(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    print(f"[INFO] Loaded ROI from {path}: {data}")
    return box

def compute_tip_wall_distances_with_radius(
    tip_px,
    left_boundary_px,
    right_boundary_px,
    tip_radius_px=0.0,
):
    """
    Compute horizontal distances from the tip to the vessel walls
    at the same y-level as the tip, accounting for tip radius.

    tip_px is the tip center.
    tip_radius_px is the horizontal radius of the tip blob in pixels.

    Returns both center-based and edge-based distances.
    """
    tip_x = float(tip_px[0])
    tip_y = float(tip_px[1])

    x_left_at_tip = float(interpolate_boundary_x_at_y(left_boundary_px, np.array([tip_y]))[0])
    x_right_at_tip = float(interpolate_boundary_x_at_y(right_boundary_px, np.array([tip_y]))[0])

    # center-based distances
    dist_left_center = tip_x - x_left_at_tip
    dist_right_center = x_right_at_tip - tip_x

    # edge-based distances
    dist_left_edge = (tip_x - tip_radius_px) - x_left_at_tip
    dist_right_edge = x_right_at_tip - (tip_x + tip_radius_px)

    if dist_left_edge <= dist_right_edge:
        closest_wall = "left"
        closest_distance = dist_left_edge
    else:
        closest_wall = "right"
        closest_distance = dist_right_edge

    return {
        "tip_x": tip_x,
        "tip_y": tip_y,
        "x_left_at_tip": x_left_at_tip,
        "x_right_at_tip": x_right_at_tip,
        "tip_radius_px": float(tip_radius_px),

        "dist_left_center": float(dist_left_center),
        "dist_right_center": float(dist_right_center),

        "dist_left_edge": float(dist_left_edge),
        "dist_right_edge": float(dist_right_edge),

        "closest_wall": closest_wall,
        "closest_distance": float(closest_distance),
    }
def fit_beam_centerline_from_markers(markers, y_samples=None):
    """
    Fit beam centerline as x(y) using a cubic through the 4 ordered markers.

    markers dict must contain:
      base_px, mag_start_px, tangent_start_px, tip_px

    Returns:
        beam_points_px: list of (x, y)
        poly_coeffs: cubic coefficients for x(y)
    """
    pts = np.array([
        markers["base_px"],
        markers["mag_start_px"],
        markers["tangent_start_px"],
        markers["tip_px"],
    ], dtype=np.float32)

    xs = pts[:, 0]
    ys = pts[:, 1]

    # sort by y so the fit is stable in the vessel direction
    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]

    # cubic fit: x = f(y)
    coeffs = np.polyfit(ys, xs, deg=3)
    poly = np.poly1d(coeffs)

    if y_samples is None:
        y_samples = np.linspace(float(ys.min()), float(ys.max()), 200)

    x_samples = poly(y_samples)

    beam_points_px = [(float(x), float(y)) for x, y in zip(x_samples, y_samples)]
    return beam_points_px, coeffs

def boundary_to_xy_arrays(boundary_points):
    pts = np.array(boundary_points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    order = np.argsort(ys)
    return xs[order], ys[order]
def draw_beam_and_vessel_overlay(
    image_bgr,
    beam_points_px,
    left_boundary_px,
    right_boundary_px,
    markers=None,
    red_roi_box=None,
    blue_roi_box=None,
    tip_distance_info=None,
    tip_wall_angle_info=None,
):
    vis = image_bgr.copy()

    # if red_roi_box is not None:
    #     x, y, w, h = red_roi_box
    #     cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # if blue_roi_box is not None:
    #     x, y, w, h = blue_roi_box
    #     cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # draw vessel boundaries
    for x, y in left_boundary_px:
        cv2.circle(vis, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

    for x, y in right_boundary_px:
        cv2.circle(vis, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)

    # draw beam centerline
    beam_int = [(int(round(x)), int(round(y))) for x, y in beam_points_px]
    for i in range(len(beam_int) - 1):
        cv2.line(vis, beam_int[i], beam_int[i + 1], (255, 255, 255), 2)

    # draw markers
    if markers is not None:
        color_map = {
            "base_px": (255, 0, 0),
            "mag_start_px": (0, 255, 255),
            "tangent_start_px": (255, 0, 255),
            "tip_px": (0, 0, 255),
        }
        for key, p in markers.items():
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, color_map[key], -1)
            cv2.putText(
                vis, key.replace("_px", ""),
                (int(p[0]) + 5, int(p[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 5)
            cv2.putText(
                vis, key.replace("_px", ""),
                (int(p[0]) + 5, int(p[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # draw closest wall tangent near the tip
    if tip_wall_angle_info is not None:
        p_minus = tip_wall_angle_info["wall_tangent_points"]["p_minus"]
        p_plus = tip_wall_angle_info["wall_tangent_points"]["p_plus"]

        p1 = (int(round(p_minus[0])), int(round(p_minus[1])))
        p2 = (int(round(p_plus[0])), int(round(p_plus[1])))

        cv2.line(vis, p1, p2, (180, 105, 255), 3)  # yellow wall tangent segment

        txt_angle = f"Beam-wall tangent angle = {tip_wall_angle_info['beam_wall_tangent_angle_deg']:.2f} deg"

        cv2.putText(vis, txt_angle, (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 5)
        cv2.putText(vis, txt_angle, (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Beam reconstruction inside vessel bounds")
    plt.axis("off")
    plt.show()

def interpolate_boundary_x_at_y(boundary_points, y_query):
    """
    Linear interpolation of x(y) for a boundary represented by (x, y) points.
    """
    xs, ys = boundary_to_xy_arrays(boundary_points)

    y_min, y_max = ys.min(), ys.max()
    y_query = np.asarray(y_query)

    x_query = np.interp(
        np.clip(y_query, y_min, y_max),
        ys,
        xs
    )
    return x_query
def compare_beam_to_vessel(beam_points_px, left_boundary_px, right_boundary_px):
    """
    Compare beam centerline to vessel walls row-by-row.

    Returns:
        result dict with inside mask and clearances
    """
    beam = np.array(beam_points_px, dtype=np.float32)
    x_beam = beam[:, 0]
    y_beam = beam[:, 1]

    x_left = interpolate_boundary_x_at_y(left_boundary_px, y_beam)
    x_right = interpolate_boundary_x_at_y(right_boundary_px, y_beam)

    inside = (x_beam >= x_left) & (x_beam <= x_right)

    clearance_left = x_beam - x_left
    clearance_right = x_right - x_beam

    return {
        "y": y_beam,
        "x_beam": x_beam,
        "x_left": x_left,
        "x_right": x_right,
        "inside": inside,
        "clearance_left": clearance_left,
        "clearance_right": clearance_right,
        "min_clearance_left": float(np.min(clearance_left)),
        "min_clearance_right": float(np.min(clearance_right)),
        "all_inside": bool(np.all(inside)),
    }
def mm_to_px_distance(dist_mm, mm_per_pixel):
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be positive.")
    return float(dist_mm / mm_per_pixel)
def compute_beam_length_px(beam_points_px):
    pts = np.array(beam_points_px, dtype=np.float32)
    if len(pts) < 2:
        return 0.0

    diffs = np.diff(pts, axis=0)              # consecutive segment vectors
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return float(np.sum(seg_lengths))
def reconstruct_beam_within_vessel(
    image_filename="focused_image.jpg",
    red_roi_path="red_roi_box.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=True,
):
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    red_roi_box = load_roi_box(red_roi_path)
    blue_roi_box = load_roi_box(blue_roi_path)
    green_roi_box = load_roi_box(green_roi_path)

    # --- green calibration points ---
    green_result = detect_2_green_calibration_points(
        image_bgr=image_bgr,
        roi_box=green_roi_box,
        green_h_low=35,
        green_h_high=85,
        sat_min=60,
        val_min=60,
        min_area=20,
        max_area=50000,
        show_debug=False,
    )

    green_pt1, green_pt2 = green_result["points_px"]
    mm_per_pixel = compute_mm_per_pixel(green_pt1, green_pt2, known_distance_mm=40.0)

    # --- red markers / beam tip state ---
    tip_result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=red_roi_box,
        show=show,
        show_debug_markers=show,
        unwrap_angle=True,
        pivot_hint=pivot_hint,
    )

    markers = tip_result["markers"]

    # --- vessel boundaries ---
    vessel = detect_blue_vessel_boundaries(
        image_bgr=image_bgr,
        roi_box=blue_roi_box,
        blue_h_low=90,
        blue_h_high=140,
        sat_min=40,
        val_min=40,
        show_debug=show,
    )

    left_smooth = smooth_boundary(vessel["left_boundary_px"], window=11)
    right_smooth = smooth_boundary(vessel["right_boundary_px"], window=11)

    # --- beam reconstruction ---
    beam_points_px, beam_coeffs = fit_beam_centerline_from_markers(markers)
    beam_length_px = compute_beam_length_px(beam_points_px)
    beam_length_mm = beam_length_px * mm_per_pixel

    comparison = compare_beam_to_vessel(
        beam_points_px=beam_points_px,
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
    )

    tip_radius_mm = 1.8
    tip_radius_px = mm_to_px_distance(tip_radius_mm, mm_per_pixel)
    tip_distance_info = compute_tip_wall_distances_with_radius(
        tip_px=markers["tip_px"],
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
        tip_radius_px=tip_radius_px,
    )

    tip_wall_angle_info = compute_tip_to_wall_tangent_angle(
        markers=markers,
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
        tip_distance_info=tip_distance_info,
        dy=5.0,
    )

    # convert selected distances to mm
    comparison_mm = {
        "min_clearance_left_mm": px_to_mm_distance(comparison["min_clearance_left"], mm_per_pixel),
        "min_clearance_right_mm": px_to_mm_distance(comparison["min_clearance_right"], mm_per_pixel),
    }

    tip_distance_info_mm = {
        "dist_left_center_mm": px_to_mm_distance(tip_distance_info["dist_left_center"], mm_per_pixel),
        "dist_right_center_mm": px_to_mm_distance(tip_distance_info["dist_right_center"], mm_per_pixel),
        "dist_left_edge_mm": px_to_mm_distance(tip_distance_info["dist_left_edge"], mm_per_pixel),
        "dist_right_edge_mm": px_to_mm_distance(tip_distance_info["dist_right_edge"], mm_per_pixel),
        "closest_distance_mm": px_to_mm_distance(tip_distance_info["closest_distance"], mm_per_pixel),
    }

    result = {
        "green_result": green_result,
        "tip_result": tip_result,
        "vessel_result": vessel,
        "left_boundary_px": left_smooth,
        "right_boundary_px": right_smooth,
        "beam_centerline_px": beam_points_px,
        "beam_poly_coeffs": beam_coeffs,
        "comparison": comparison,
        "comparison_mm": comparison_mm,
        "tip_distance_info": tip_distance_info,
        "tip_distance_info_mm": tip_distance_info_mm,
        "tip_wall_angle_info": tip_wall_angle_info,
        "beam_length_px": beam_length_px,
        "mm_per_pixel": mm_per_pixel,
        "beam_length_mm": beam_length_mm,
    }

    print("Beam-to-wall tangent angle (deg):", tip_wall_angle_info["beam_wall_tangent_angle_deg"])
    print("Calibration distance (px):", green_result["distance_px"])
    print("mm_per_pixel:", mm_per_pixel)
    print("Beam length (px):", beam_length_px)
    print("Beam length (mm):", beam_length_mm)

    if show:
        draw_beam_and_vessel_overlay(
            image_bgr=image_bgr,
            beam_points_px=beam_points_px,
            left_boundary_px=left_smooth,
            right_boundary_px=right_smooth,
            markers=markers,
            red_roi_box=red_roi_box,
            blue_roi_box=blue_roi_box,
            tip_distance_info=tip_distance_info,
            tip_wall_angle_info=tip_wall_angle_info,
        )

        print("\n--- BEAM WITHIN VESSEL ---")
        print("Tip image px:", tip_result["tip_image_px"])
        print("Tip x,y from base:", tip_result["tip_xy_from_base"])
        print("Tip tangent angle (deg):", tip_result["tip_tangent_angle_deg"])
        print("All beam points inside vessel:", comparison["all_inside"])
        print("Min clearance to left wall (px):", comparison["min_clearance_left"])
        print("Min clearance to right wall (px):", comparison["min_clearance_right"])
        print("Min clearance to left wall (mm):", comparison_mm["min_clearance_left_mm"])
        print("Min clearance to right wall (mm):", comparison_mm["min_clearance_right_mm"])

        print("Tip edge distance to left wall (mm):", tip_distance_info_mm["dist_left_edge_mm"])
        print("Tip edge distance to right wall (mm):", tip_distance_info_mm["dist_right_edge_mm"])
        print("Tip center distance to left wall (mm):", tip_distance_info_mm["dist_left_center_mm"])
        print("Tip center distance to right wall (mm):", tip_distance_info_mm["dist_right_center_mm"])
        print("Tip edge distance to left wall (mm):", tip_distance_info_mm["dist_left_edge_mm"])
        print("Tip edge distance to right wall (mm):", tip_distance_info_mm["dist_right_edge_mm"])
        print("Closest wall:", tip_distance_info["closest_wall"])
        print("Closest tip-wall distance (px):", tip_distance_info["closest_distance"])
        print("Closest tip-wall distance (mm):", tip_distance_info_mm["closest_distance_mm"])

    return result
def detect_2_green_calibration_points(
    image_bgr,
    roi_box=None,
    green_h_low=35,
    green_h_high=95,
    sat_min=40,
    val_min=40,
    min_area=20,
    max_area=50000,
    show_debug=False,
):
    """
    Detect exactly 2 green calibration markers and return their centroids.

    Returns:
        {
            "points_px": [(x1, y1), (x2, y2)],
            "distance_px": ...,
            "mask": ...,
        }
    """
    if roi_box is not None:
        x0, y0, w, h = roi_box
        roi = image_bgr[y0:y0 + h, x0:x0 + w].copy()
    else:
        x0, y0 = 0, 0
        roi = image_bgr.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([green_h_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([green_h_high, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # clean mask
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-8:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        candidates.append({
            "area": float(area),
            "centroid_roi": (float(cx), float(cy)),
            "contour": cnt,
        })

    if len(candidates) < 2:
        raise RuntimeError(f"Expected at least 2 green blobs, found {len(candidates)}")

    # Keep the 2 largest blobs
    candidates = sorted(candidates, key=lambda d: d["area"], reverse=True)[:2]

    pts = []
    for c in candidates:
        cx, cy = c["centroid_roi"]
        pts.append((float(cx + x0), float(cy + y0)))

    # Sort left-to-right for consistency
    pts = sorted(pts, key=lambda p: p[0])

    p1 = np.array(pts[0], dtype=np.float32)
    p2 = np.array(pts[1], dtype=np.float32)
    distance_px = float(np.linalg.norm(p2 - p1))

    if show_debug:
        vis = roi.copy()

        for c in candidates:
            cv2.drawContours(vis, [c["contour"]], -1, (255, 255, 255), 2)

        for i, (px, py) in enumerate(pts):
            px_roi = int(round(px - x0))
            py_roi = int(round(py - y0))
            cv2.circle(vis, (px_roi, py_roi), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"G{i+1}",
                (px_roi + 6, py_roi - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        cv2.line(
            vis,
            (int(round(pts[0][0] - x0)), int(round(pts[0][1] - y0))),
            (int(round(pts[1][0] - x0)), int(round(pts[1][1] - y0))),
            (0, 255, 255),
            2,
        )

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Green mask")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"Green calibration points, dist = {distance_px:.2f} px")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "points_px": pts,
        "distance_px": distance_px,
        "mask": mask,
    }
def compute_mm_per_pixel(p1_px, p2_px, known_distance_mm=40.0):
    p1 = np.array(p1_px, dtype=np.float32)
    p2 = np.array(p2_px, dtype=np.float32)

    dist_px = np.linalg.norm(p2 - p1)

    if dist_px < 1e-6:
        raise ValueError("Reference points are too close or identical.")

    mm_per_pixel = known_distance_mm / dist_px
    return float(mm_per_pixel)
def pixel_to_mm(point_px, origin_px, mm_per_pixel):
    point = np.array(point_px, dtype=np.float32)
    origin = np.array(origin_px, dtype=np.float32)

    delta_px = point - origin
    delta_mm = delta_px * mm_per_pixel

    return tuple(delta_mm.tolist())
def px_to_mm_distance(dist_px, mm_per_pixel):
    return float(dist_px * mm_per_pixel)
def order_four_markers(points, pivot_hint=None):
    pts = [np.array(p, dtype=np.float32) for p in points]
    if len(pts) != 4:
        raise ValueError(f"Expected 4 points, got {len(pts)}")

    if pivot_hint is not None:
        pivot_hint = np.array(pivot_hint, dtype=np.float32)
        start_idx = int(np.argmin([np.linalg.norm(p - pivot_hint) for p in pts]))
    else:
        start_idx = int(np.argmin([p[0] for p in pts]))

    ordered = [pts.pop(start_idx)]

    while pts:
        last = ordered[-1]
        next_idx = int(np.argmin([np.linalg.norm(p - last) for p in pts]))
        ordered.append(pts.pop(next_idx))

    return [tuple(map(float, p)) for p in ordered]


def image_to_base_frame(point_px, base_px):
    x = float(point_px[0] - base_px[0])
    y = float(-(point_px[1] - base_px[1]))
    return np.array([x, y], dtype=np.float32)


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


def measure_tip_state_4markers(
    image_filename="focused_image.jpg",
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
        min_area=10,
        max_area=40000,
        sat_min=10,
        val_min=10,
        hue1_high=25,
        hue2_low=155,
        show_debug=show_debug_markers,
    )

    if detected_points is None or len(detected_points) != 4:
        raise RuntimeError(f"Expected 4 detected red markers, got: {detected_points}")

    ordered = order_four_markers(detected_points, pivot_hint=pivot_hint)
    tip_px, tangent_start_px, mag_start_px, base_px = ordered
    base = np.array(base_px, dtype=np.float32)
    mag_start = np.array(mag_start_px, dtype=np.float32)
    tangent_start = np.array(tangent_start_px, dtype=np.float32)
    tip = np.array(tip_px, dtype=np.float32)

    tip_xy = image_to_base_frame(tip, base)

    ref_vec = np.array([
        mag_start[0] - base[0],
        -(mag_start[1] - base[1])
    ], dtype=np.float32)

    tan_vec = np.array([
        tip[0] - tangent_start[0],
        -(tip[1] - tangent_start[1])
    ], dtype=np.float32)

    tangent_angle_deg = signed_angle_between_vectors(ref_vec, tan_vec)

    if unwrap_angle:
        tangent_angle_deg = unwrap_tip_angle(tangent_angle_deg)

    result = {
        "tip_image_px": (float(tip[0]), float(tip[1])),
        "tip_xy_from_base": (float(tip_xy[0]), float(tip_xy[1])),
        "tip_tangent_angle_deg": float(tangent_angle_deg),
        "markers": {
            "base_px": tuple(map(float, base_px)),
            "mag_start_px": tuple(map(float, mag_start_px)),
            "tangent_start_px": tuple(map(float, tangent_start_px)),
            "tip_px": tuple(map(float, tip_px)),
        },
        "roi_box": roi_box,
    }

    if show:
        vis = image_bgr.copy()

        if roi_box is not None:
            x, y, w, h = roi_box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        color_map = {
            "base_px": (255, 0, 0),
            "mag_start_px": (0, 255, 255),
            "tangent_start_px": (255, 0, 255),
            "tip_px": (0, 0, 255),
        }

        for key, p in result["markers"].items():
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, color_map[key], -1)

        cv2.line(vis, (int(base[0]), int(base[1])), (int(mag_start[0]), int(mag_start[1])), (0, 255, 0), 2)
        cv2.line(vis, (int(mag_start[0]), int(mag_start[1])), (int(tangent_start[0]), int(tangent_start[1])), (0, 255, 0), 2)
        cv2.line(vis, (int(tangent_start[0]), int(tangent_start[1])), (int(tip[0]), int(tip[1])), (0, 255, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("4-marker tip state measurement")
        plt.axis("off")
        plt.show()

    return result

def unit_vector(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def unsigned_angle_between_vectors(v1, v2):
    """
    Smallest angle between two vectors, in degrees, in [0, 180].
    """
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def compute_boundary_tangent_vector_at_y(boundary_points, y0, dy=5.0):
    """
    Estimate wall tangent vector at y = y0 using two nearby interpolated points.

    boundary_points: list of (x, y) in image coordinates
    y0: y-location of interest (image coordinates)
    dy: vertical offset used for local finite difference

    Returns:
        {
            "p_minus": (x1, y1),
            "p_plus": (x2, y2),
            "tangent_vec_image": np.array([dx, dy_img]),
            "tangent_vec_cartesian": np.array([dx, dy_cart])
        }
    """
    xs, ys = boundary_to_xy_arrays(boundary_points)

    y_min = float(ys.min())
    y_max = float(ys.max())

    y1 = max(y_min, y0 - dy)
    y2 = min(y_max, y0 + dy)

    # if tip is very close to one end, force a small separation
    if abs(y2 - y1) < 1e-6:
        raise ValueError("Not enough boundary extent near y0 to estimate tangent.")

    x1 = float(np.interp(y1, ys, xs))
    x2 = float(np.interp(y2, ys, xs))

    # image-coordinate tangent
    tangent_img = np.array([x2 - x1, y2 - y1], dtype=np.float32)

    # Cartesian tangent: x right, y up
    tangent_cart = np.array([x2 - x1, -(y2 - y1)], dtype=np.float32)

    return {
        "p_minus": (x1, y1),
        "p_plus": (x2, y2),
        "tangent_vec_image": tangent_img,
        "tangent_vec_cartesian": tangent_cart,
    }


def compute_tip_to_wall_tangent_angle(markers, left_boundary_px, right_boundary_px, tip_distance_info, dy=5.0):
    """
    Compute the angle between the beam tip tangent and the tangent of the closest vessel wall.

    Beam tangent uses tangent_start -> tip.
    Wall tangent uses local finite difference on the closest wall at tip_y.

    Returns:
        {
            "closest_wall": ...,
            "beam_tangent_vec_cartesian": ...,
            "wall_tangent_vec_cartesian": ...,
            "wall_tangent_points": {"p_minus": ..., "p_plus": ...},
            "beam_wall_tangent_angle_deg": ...
        }
    """
    tip = np.array(markers["tip_px"], dtype=np.float32)
    tangent_start = np.array(markers["tangent_start_px"], dtype=np.float32)

    # beam tangent in Cartesian coordinates
    beam_tangent_vec = np.array([
        tip[0] - tangent_start[0],
        -(tip[1] - tangent_start[1])
    ], dtype=np.float32)

    tip_y = float(tip[1])
    closest_wall = tip_distance_info["closest_wall"]

    if closest_wall == "left":
        wall_info = compute_boundary_tangent_vector_at_y(left_boundary_px, tip_y, dy=dy)
    else:
        wall_info = compute_boundary_tangent_vector_at_y(right_boundary_px, tip_y, dy=dy)

    wall_tangent_vec = wall_info["tangent_vec_cartesian"]

    angle_deg = unsigned_angle_between_vectors(beam_tangent_vec, wall_tangent_vec)

    # because tangents are directionless lines, angle > 90 should be folded back
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg

    return {
        "closest_wall": closest_wall,
        "beam_tangent_vec_cartesian": beam_tangent_vec,
        "wall_tangent_vec_cartesian": wall_tangent_vec,
        "wall_tangent_points": {
            "p_minus": wall_info["p_minus"],
            "p_plus": wall_info["p_plus"],
        },
        "beam_wall_tangent_angle_deg": float(angle_deg),
    }
if __name__ == "__main__":
    new_capture()
    
    result = reconstruct_beam_within_vessel(
        image_filename="focused_image.jpg",
        red_roi_path="red_roi_box.json",
        blue_roi_path="blue_roi_box.json",
        pivot_hint=None,
        show=True,
    )