import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from proper_research.vision.vision_w_tangnet import detect_4_red_markers_in_roi
from proper_research.vision.measure_length import new_capture
from proper_research.vision.line_fit_through_points import measure_tip_base_angles,make_beam_frame_from_proximal_markers, project_point_to_beam_frame
from proper_research.vision.detect_blue import load_manual_vessel_boundaries_with_frame
# from proper_research.vision.boundaries import detect_blue_vessel_boundaries, smooth_boundary
RED_ROI_CONFIG_FILE = "red_roi_box.json"
BLUE_ROI_CONFIG_FILE = "blue_roi_box.json"
GREEN_ROI_CONFIG_FILE = "green_roi_box.json"
MANUAL_VESSEL_BOUNDARY_FILE = "manual_vessel_boundaries.json"
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
def load_search_area(path):
    """
    Supports either:
      rectangle JSON: {"x":..., "y":..., "w":..., "h":...}
      polygon JSON:   {"points": [[x1,y1], [x2,y2], ...]}
    """
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    if "points" in data:
        polygon = [tuple(map(float, p)) for p in data["points"]]
        print(f"[INFO] Loaded polygon area from {path}: {len(polygon)} points")
        return {
            "type": "polygon",
            "polygon": polygon,
        }

    if all(k in data for k in ("x", "y", "w", "h")):
        box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
        print(f"[INFO] Loaded box area from {path}: {data}")
        return {
            "type": "box",
            "box": box,
        }

    raise ValueError(f"Unrecognized ROI/custom area format in {path}")


def area_to_bbox(area, image_shape=None):
    """
    Convert either box or polygon area to a bounding box (x, y, w, h).
    """
    if area is None:
        if image_shape is None:
            return None
        h, w = image_shape[:2]
        return (0, 0, w, h)

    if area["type"] == "box":
        return area["box"]

    if area["type"] == "polygon":
        pts = np.array(area["polygon"], dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        x0 = int(np.floor(xs.min()))
        y0 = int(np.floor(ys.min()))
        x1 = int(np.ceil(xs.max()))
        y1 = int(np.ceil(ys.max()))

        if image_shape is not None:
            h_img, w_img = image_shape[:2]
            x0 = max(0, min(x0, w_img - 1))
            y0 = max(0, min(y0, h_img - 1))
            x1 = max(x0 + 1, min(x1, w_img))
            y1 = max(y0 + 1, min(y1, h_img))

        return (x0, y0, x1 - x0, y1 - y0)

    raise ValueError(f"Unknown area type: {area['type']}")


def draw_area_overlay(image_bgr, area, color=(0, 255, 255), thickness=2):
    """
    Draw either a rectangle or polygon on an image.
    """
    if area is None:
        return image_bgr

    vis = image_bgr

    if area["type"] == "box":
        x, y, w, h = area["box"]
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)

    elif area["type"] == "polygon":
        pts = np.array(area["polygon"], dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

    else:
        raise ValueError(f"Unknown area type: {area['type']}")

    return vis
def plot_contact_lumen_debug(
    lumen_C_m,
    lumen_R_m,
    out_path="debug_outputs/contact_lumen_debug.png",
    beam_points_m=None,
    beam_radius_m=0.0,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    C = np.asarray(lumen_C_m, float)
    R = np.asarray(lumen_R_m, float).reshape(-1)

    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError(f"Expected lumen_C_m shape (N,3), got {C.shape}")

    if len(C) != len(R):
        raise ValueError("lumen_C_m and lumen_R_m must have same length")

    xy = C[:, :2]

    tangents = np.zeros_like(xy)
    tangents[1:-1] = xy[2:] - xy[:-2]
    tangents[0] = xy[1] - xy[0]
    tangents[-1] = xy[-1] - xy[-2]

    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    upper = xy + R[:, None] * normals
    lower = xy - R[:, None] * normals

    C_mm = 1e3 * xy
    upper_mm = 1e3 * upper
    lower_mm = 1e3 * lower

    plt.figure(figsize=(7, 7))

    plt.plot(C_mm[:, 0], C_mm[:, 1], "k-o", markersize=3, label="Contact centerline")
    plt.plot(upper_mm[:, 0], upper_mm[:, 1], "r--", label="Contact wall +R")
    plt.plot(lower_mm[:, 0], lower_mm[:, 1], "g--", label="Contact wall -R")

    for i in range(len(C)):
        plt.plot(
            [upper_mm[i, 0], lower_mm[i, 0]],
            [upper_mm[i, 1], lower_mm[i, 1]],
            "m-",
            alpha=0.25,
        )

    if beam_points_m is not None:
        B = np.asarray(beam_points_m, float)
        B_mm = 1e3 * B[:, :2]
        plt.plot(B_mm[:, 0], B_mm[:, 1], "b-o", markersize=3, label="Beam nodes")

        if beam_radius_m > 0:
            plt.scatter(
                B_mm[:, 0],
                B_mm[:, 1],
                s=(1e3 * beam_radius_m * 12) ** 2,
                alpha=0.15,
                label="Beam radius visual",
            )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("contact local x [mm]")
    plt.ylabel("contact local y [mm]")
    plt.title("What the contact model sees: centerline + radius")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[DBG] Saved contact lumen debug plot: {out_path}")
def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)
def build_lumen_from_parametric_boundaries(
    left_boundary_px,
    right_boundary_px,
    base_px_ref,
    ex_ref,
    ey_ref,
    mm_per_pixel,
    z_mm=0.0,
):
    left = np.asarray(left_boundary_px, dtype=float)
    right = np.asarray(right_boundary_px, dtype=float)

    if len(left) != len(right):
        raise ValueError("Left and right boundaries must have same number of samples.")

    center = 0.5 * (left + right)
    radius_px = 0.5 * np.linalg.norm(right - left, axis=1)

    base = np.asarray(base_px_ref, dtype=float).reshape(2,)
    ex = unit(np.asarray(ex_ref, dtype=float).reshape(2,))
    ey = unit(np.asarray(ey_ref, dtype=float).reshape(2,))

    # IMPORTANT:
    # Do NOT flip image y here.
    # ex and ey are already defined in image pixel coordinates.
    dpx = center - base[None, :]

    x_mm = dpx @ ex * mm_per_pixel
    y_mm = dpx @ ey * mm_per_pixel

    # If your downstream beam model expects forward to be negative x,
    # apply that convention here only once.
    x_mm = -x_mm
    y_mm = -y_mm
    z_mm_arr = np.full_like(x_mm, float(z_mm))

    lumen_C_mm = np.column_stack([x_mm, y_mm, z_mm_arr])
    lumen_R_mm = radius_px * mm_per_pixel

    lumen_C_m = lumen_C_mm / 1000.0
    lumen_R_m = lumen_R_mm / 1000.0

    return lumen_C_m, lumen_R_m, lumen_C_mm, lumen_R_mm
def make_fixed_local_frame(base_px_ref, mag_start_px_ref=None, tangent_start_px_ref=None):
    if base_px_ref is None:
        raise ValueError("base_px_ref is None")

    base = np.asarray(base_px_ref, dtype=np.float32).reshape(2,)

    if mag_start_px_ref is not None:
        ref_pt = np.asarray(mag_start_px_ref, dtype=np.float32).reshape(2,)
    elif tangent_start_px_ref is not None:
        ref_pt = np.asarray(tangent_start_px_ref, dtype=np.float32).reshape(2,)
    else:
        raise ValueError(
            "Need either mag_start_px_ref or tangent_start_px_ref to build fixed local frame."
        )

    ref = np.array([
        ref_pt[0] - base[0],
        -(ref_pt[1] - base[1]),
    ], dtype=np.float32)

    nr = np.linalg.norm(ref)
    if nr < 1e-12:
        raise ValueError("Reference base->reference_point vector is zero length.")

    ex_ref = ref / nr
    ey_ref = np.array([-ex_ref[1], ex_ref[0]], dtype=np.float32)

    return base, ex_ref, ey_ref
def make_fixed_local_frame(base_px_ref, mag_start_px_ref=None, tangent_start_px_ref=None):
    if base_px_ref is None:
        raise ValueError("base_px_ref is None")

    base = np.asarray(base_px_ref, dtype=np.float32).reshape(2,)

    if mag_start_px_ref is not None:
        ref_pt = np.asarray(mag_start_px_ref, dtype=np.float32).reshape(2,)
    elif tangent_start_px_ref is not None:
        ref_pt = np.asarray(tangent_start_px_ref, dtype=np.float32).reshape(2,)
    else:
        raise ValueError(
            "Need either mag_start_px_ref or tangent_start_px_ref to build fixed local frame."
        )

    ref = np.array([
        ref_pt[0] - base[0],
        -(ref_pt[1] - base[1]),
    ], dtype=np.float32)

    nr = np.linalg.norm(ref)
    if nr < 1e-12:
        raise ValueError("Reference base->reference_point vector is zero length.")

    ex_ref = ref / nr
    ey_ref = np.array([-ex_ref[1], ex_ref[0]], dtype=np.float32)

    return base, ex_ref, ey_ref
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
def beam_polyline_from_markers(markers, n_samples_per_segment=40):
    """
    Build a piecewise-linear beam centerline through the ordered markers.

    Order:
      base_px -> mag_start_px -> tangent_start_px -> tip_px

    If mag_start_px is missing, it is skipped.

    Returns
    -------
    beam_points_px : list of (x, y)
        Densely sampled polyline for plotting.
    ordered_pts : list of np.ndarray shape (2,)
        The actual marker points used.
    """
    ordered_keys = ["base_px", "mag_start_px", "tangent_start_px", "tip_px"]

    ordered_pts = []
    for key in ordered_keys:
        p = markers.get(key, None)
        if p is None:
            continue
        ordered_pts.append(np.asarray(p, dtype=float).reshape(2,))

    if len(ordered_pts) < 2:
        raise ValueError(f"Need at least 2 valid markers, got {len(ordered_pts)}")

    beam_points_px = []

    for i in range(len(ordered_pts) - 1):
        p0 = ordered_pts[i]
        p1 = ordered_pts[i + 1]

        ts = np.linspace(0.0, 1.0, n_samples_per_segment, endpoint=False)
        for t in ts:
            p = (1.0 - t) * p0 + t * p1
            beam_points_px.append((float(p[0]), float(p[1])))

    # append final endpoint once
    beam_points_px.append((float(ordered_pts[-1][0]), float(ordered_pts[-1][1])))

    return beam_points_px, ordered_pts
def compute_beam_length_from_ordered_pts(ordered_pts):
    ordered_pts = [np.asarray(p, dtype=float).reshape(2,) for p in ordered_pts]

    if len(ordered_pts) < 2:
        raise ValueError(f"Need at least 2 points, got {len(ordered_pts)}")

    length_px = 0.0
    for i in range(len(ordered_pts) - 1):
        length_px += np.linalg.norm(ordered_pts[i + 1] - ordered_pts[i])

    return float(length_px)
def fit_beam_centerline_from_markers(markers, y_samples=None):
    """
    Fit beam centerline as x(y) from available ordered markers.

    Expected marker keys:
      base_px, mag_start_px, tangent_start_px, tip_px

    Uses:
      - cubic fit for 4 valid markers
      - quadratic fit for 3 valid markers

    Returns:
        beam_points_px: list of (x, y)
        poly_coeffs: polynomial coefficients for x(y)
        poly_degree: polynomial degree used
    """
    ordered_keys = ["base_px", "mag_start_px", "tangent_start_px", "tip_px"]

    valid_pts = []
    for key in ordered_keys:
        p = markers.get(key, None)
        if p is None:
            continue
        valid_pts.append(np.asarray(p, dtype=np.float32).reshape(2,))

    if len(valid_pts) < 3:
        raise ValueError(
            f"Need at least 3 valid markers to fit beam centerline, got {len(valid_pts)}"
        )

    pts = np.vstack(valid_pts)

    xs = pts[:, 0]
    ys = pts[:, 1]

    order = np.argsort(ys)
    ys = ys[order]
    xs = xs[order]

    if np.ptp(ys) < 1e-6:
        raise ValueError("Cannot fit beam centerline: y values have near-zero spread.")

    poly_degree = min(len(valid_pts) - 1, 3)

    coeffs = np.polyfit(ys, xs, deg=poly_degree)
    poly = np.poly1d(coeffs)

    if y_samples is None:
        y_samples = np.linspace(float(ys.min()), float(ys.max()), 200)

    x_samples = poly(y_samples)

    beam_points_px = [(float(x), float(y)) for x, y in zip(x_samples, y_samples)]
    return beam_points_px, coeffs, poly_degree
def boundary_to_xy_arrays(boundary_points):
    pts = np.array(boundary_points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    order = np.argsort(ys)
    return xs[order], ys[order]
import cv2
import numpy as np
# from skimage.morphology import skeletonize
# from skimage.graph import route_through_array

def compute_marker_anchored_black_beam_length_px(
    image_bgr,
    ordered_pts,
    threshold=100,
    tube_radius_px=25,
    bridge_radius_px=8,
):
    """
    Estimate beam length using red markers as anchors and black beam pixels
    between markers.

    ordered_pts:
        array/list of marker centres in order, shape (M, 2), as (x, y).

    Strategy:
        - threshold black beam
        - force small disks around marker centres to be traversable
        - restrict path search to a tube around each marker-marker chord
        - route through black skeleton/cost between consecutive markers
        - sum segment lengths
    """
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array

    ordered_pts = np.asarray(ordered_pts, float).reshape(-1, 2)

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Black beam mask
    beam_mask = image_gray < threshold
    beam_mask = beam_mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    beam_mask = cv2.morphologyEx(beam_mask, cv2.MORPH_CLOSE, kernel)

    # Important: bridge red marker holes.
    # Add small disks around marker centres into the traversable mask.
    H, W = beam_mask.shape
    for x, y in ordered_pts:
        cv2.circle(
            beam_mask,
            (int(round(x)), int(round(y))),
            int(bridge_radius_px),
            1,
            thickness=-1,
        )

    beam_mask_bool = beam_mask.astype(bool)
    skeleton = skeletonize(beam_mask_bool)

    total_length_px = 0.0
    all_path_xy = []

    yy, xx = np.mgrid[0:H, 0:W]

    for a, b in zip(ordered_pts[:-1], ordered_pts[1:]):
        ax, ay = a
        bx, by = b

        # Build a local tube around the straight segment a-b.
        ab = np.array([bx - ax, by - ay], float)
        ab_len = np.linalg.norm(ab)

        if ab_len < 1e-9:
            continue

        ab_unit = ab / ab_len

        apx = xx - ax
        apy = yy - ay

        proj = apx * ab_unit[0] + apy * ab_unit[1]
        closest_x = ax + np.clip(proj, 0.0, ab_len) * ab_unit[0]
        closest_y = ay + np.clip(proj, 0.0, ab_len) * ab_unit[1]

        dist_to_segment = np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)
        tube = dist_to_segment <= tube_radius_px

        # Cost: prefer skeleton, allow black mask, strongly discourage outside tube.
        cost = np.full((H, W), 1e6, dtype=float)
        cost[tube & beam_mask_bool] = 10.0
        cost[tube & skeleton] = 1.0

        # Ensure marker centre pixels are reachable.
        start_rc = (int(round(ay)), int(round(ax)))
        end_rc = (int(round(by)), int(round(bx)))

        cv2.circle(cost, (int(round(ax)), int(round(ay))), int(bridge_radius_px), 1.0, thickness=-1)
        cv2.circle(cost, (int(round(bx)), int(round(by))), int(bridge_radius_px), 1.0, thickness=-1)

        path_rc, _ = route_through_array(
            cost,
            start_rc,
            end_rc,
            fully_connected=True,
        )

        path_rc = np.asarray(path_rc, dtype=float)
        path_xy = np.column_stack([path_rc[:, 1], path_rc[:, 0]])

        diffs = np.diff(path_xy, axis=0)
        seg_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

        total_length_px += seg_len

        if len(all_path_xy) == 0:
            all_path_xy.append(path_xy)
        else:
            all_path_xy.append(path_xy[1:])

    if len(all_path_xy) > 0:
        full_path_xy = np.vstack(all_path_xy)
    else:
        full_path_xy = ordered_pts.copy()

    return total_length_px, skeleton, full_path_xy
def draw_beam_and_vessel_overlay(
    image_bgr,
    beam_points_px,
    left_boundary_px,
    right_boundary_px,
    markers=None,
    red_area=None,
    blue_area=None,
    tip_distance_info=None,
    tip_wall_angle_info=None,
    save_path=None,
    show_plot=True,
):
    vis = image_bgr.copy()

    # draw vessel boundaries
    for x, y in left_boundary_px:
        cv2.circle(vis, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)

    for x, y in right_boundary_px:
        cv2.circle(vis, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1)
    # if red_area is not None:
    #     draw_area_overlay(vis, red_area, color=(0, 255, 255), thickness=2)

    if blue_area is not None:
        draw_area_overlay(vis, blue_area, color=(255, 255, 0), thickness=2)
    # draw beam centerline
    # beam_int = [(int(round(x)), int(round(y))) for x, y in beam_points_px]
    # for i in range(len(beam_int) - 1):
    #     cv2.line(vis, beam_int[i], beam_int[i + 1], (255, 255, 255), 2)

    # draw markers
    if markers is not None:
        color_map = {
            "base_px": (255, 0, 0),
            "mag_start_px": (0, 255, 255),
            "tangent_start_px": (255, 0, 255),
            "tip_px": (0, 0, 255),
        }
        for key, p in markers.items():
            if p is None:
                continue
            if key not in color_map:
                continue

            x = int(round(p[0]))
            y = int(round(p[1]))

            cv2.circle(vis, (x, y), 6, color_map[key], -1)
            cv2.putText(
                vis, key.replace("_px", ""),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 5
            )
            cv2.putText(
                vis, key.replace("_px", ""),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

    # # draw closest wall tangent near the tip
    # if tip_wall_angle_info is not None:
    #     p_minus = tip_wall_angle_info["right_wall_tangent_points"]["p_minus"]
    #     p_plus = tip_wall_angle_info["right_wall_tangent_points"]["p_plus"]

    #     p1 = (int(round(p_minus[0])), int(round(p_minus[1])))
    #     p2 = (int(round(p_plus[0])), int(round(p_plus[1])))

    #     cv2.line(vis, p1, p2, (180, 105, 255), 3)

    #     txt_angle = (
    #         f"Beam-wall tangent angle = "
    #         f"{tip_wall_angle_info['beam_right_wall_tangent_angle_deg']:.2f} deg"
    #     )

    #     cv2.putText(
    #         vis, txt_angle, (20, 125),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 5
    #     )
    #     cv2.putText(
    #         vis, txt_angle, (20, 125),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
    #     )

    # save raw overlay image if requested
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_path, vis)
        print(f"[SAVE] reconstruction overlay saved to {save_path}")

    if show_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Beam reconstruction inside vessel bounds")
        plt.axis("off")
        plt.show()

    return vis

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
def load_manual_vessel_boundaries(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    left_boundary_px = [tuple(map(float, p)) for p in data["left_boundary_px"]]
    right_boundary_px = [tuple(map(float, p)) for p in data["right_boundary_px"]]

    print(f"[INFO] Loaded manual vessel boundaries from {path}")
    return {
        "left_boundary_px": left_boundary_px,
        "right_boundary_px": right_boundary_px,
    }


def resample_polyline_by_arclength(points, n_samples=200):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points.")

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = s[-1]

    if total_len < 1e-6:
        raise ValueError("Polyline length is too small.")

    s_samples = np.linspace(0.0, total_len, n_samples)
    x_samples = np.interp(s_samples, s, pts[:, 0])
    y_samples = np.interp(s_samples, s, pts[:, 1])

    return [(float(x), float(y)) for x, y in zip(x_samples, y_samples)]
def save_manual_vessel_boundaries(left_boundary_px, right_boundary_px, path):
    data = {
        "left_boundary_px": [[float(x), float(y)] for x, y in left_boundary_px],
        "right_boundary_px": [[float(x), float(y)] for x, y in right_boundary_px],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Manual vessel boundaries saved to {path}")
def draw_manual_vessel_boundaries(
    image_filename="focused_image.jpg",
    save_path=MANUAL_VESSEL_BOUNDARY_FILE,
    blue_roi_path="blue_roi_box.json",
):
    """
    Interactive boundary drawing.

    Controls:
      - Left mouse click: add point to current wall
      - z: undo last point on current wall
      - l: switch to LEFT wall
      - r: switch to RIGHT wall
      - c: clear current wall
      - s: save
      - q or Esc: quit without saving
    """
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    area = load_search_area(blue_roi_path)

    if area is not None:
        x0, y0, w, h = area_to_bbox(area, image_bgr.shape)
        display = image_bgr[y0:y0+h, x0:x0+w].copy()
    else:
        x0, y0 = 0, 0
        display = image_bgr.copy()

    left_points = []
    right_points = []
    current_side = {"name": "left"}

    window_name = "Draw vessel boundaries"

    def redraw():
        vis = display.copy()

        # draw current instructions
        text1 = f"Current wall: {current_side['name'].upper()}"
        text2 = "Left click=add | z=undo | l/r=switch | c=clear current | s=save | q=quit"
        cv2.putText(vis, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # draw left points/lines in green
        for i, (x, y) in enumerate(left_points):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(
                    vis,
                    (int(left_points[i-1][0]), int(left_points[i-1][1])),
                    (int(x), int(y)),
                    (0, 255, 0),
                    1,
                )

        # draw right points/lines in red
        for i, (x, y) in enumerate(right_points):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(
                    vis,
                    (int(right_points[i-1][0]), int(right_points[i-1][1])),
                    (int(x), int(y)),
                    (0, 0, 255),
                    1,
                )

        cv2.imshow(window_name, vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_side["name"] == "left":
                left_points.append((float(x), float(y)))
            else:
                right_points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("z"):
            if current_side["name"] == "left" and left_points:
                left_points.pop()
            elif current_side["name"] == "right" and right_points:
                right_points.pop()
            redraw()

        elif key == ord("l"):
            current_side["name"] = "left"
            redraw()

        elif key == ord("r"):
            current_side["name"] = "right"
            redraw()

        elif key == ord("c"):
            if current_side["name"] == "left":
                left_points.clear()
            else:
                right_points.clear()
            redraw()

        elif key == ord("s"):
            if len(left_points) < 2 or len(right_points) < 2:
                print("[WARN] Need at least 2 points on each wall before saving.")
                continue

            left_boundary_roi = resample_polyline_by_arclength(left_points, n_samples=200)
            right_boundary_roi = resample_polyline_by_arclength(right_points, n_samples=200)

            left_boundary_px = [(x + x0, y + y0) for x, y in left_boundary_roi]
            right_boundary_px = [(x + x0, y + y0) for x, y in right_boundary_roi]

            save_manual_vessel_boundaries(left_boundary_px, right_boundary_px, save_path)
            cv2.destroyWindow(window_name)
            return {
                "left_boundary_px": left_boundary_px,
                "right_boundary_px": right_boundary_px,
            }

        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            print("[INFO] Boundary drawing cancelled.")
            return None
def closest_point_on_polyline(query_pt, polyline_pts):
    q = np.array(query_pt, dtype=np.float32)
    pts = np.array(polyline_pts, dtype=np.float32)

    best_pt = None
    best_dist = np.inf

    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        v = p1 - p0
        vv = float(np.dot(v, v))

        if vv < 1e-12:
            cand = p0
        else:
            t = float(np.dot(q - p0, v) / vv)
            t = max(0.0, min(1.0, t))
            cand = p0 + t * v

        d = float(np.linalg.norm(q - cand))
        if d < best_dist:
            best_dist = d
            best_pt = cand

    return tuple(map(float, best_pt)), float(best_dist)
def plot_tip_measurement_vs_lumen_local(result, tangent_scale_mm=8.0, n_tangent_pts=15):
    """
    Plot lumen centerline, lumen boundaries, measured tip position,
    measured tip tangent, and local centerline tangent in the same LOCAL frame.

    result = output of reconstruct_beam_within_vessel(...)
    """

    C_mm = np.asarray(result["lumen_C_mm"], dtype=float)   # (N,3)
    R_mm = np.asarray(result["lumen_R_mm"], dtype=float)   # (N,)
    tip_result = result["tip_result"]
    markers = tip_result["markers"]

    # measured tip position in same local frame as lumen_C_mm
    tip_xy = np.asarray(tip_result["tip_xy_from_base"], dtype=float).reshape(2,)
    tip_xy_mm = tip_xy * float(result["mm_per_pixel"])


    tip_px = np.asarray(markers["tip_px"], dtype=float).reshape(2,)
    tan_start_px = np.asarray(markers["tangent_start_px"], dtype=float).reshape(2,)
    base_px = np.asarray(markers["base_px"], dtype=float).reshape(2,)

    mag_start_raw = markers.get("mag_start_px", None)
    if mag_start_raw is not None:
        ref_pt_px = np.asarray(mag_start_raw, dtype=float).reshape(2,)
    else:
        ref_pt_px = tan_start_px

    v_img = np.array([
        tip_px[0] - tan_start_px[0],
        -(tip_px[1] - tan_start_px[1]),
    ], dtype=float)
    nv = np.linalg.norm(v_img)
    if nv < 1e-12:
        raise ValueError("Measured tangent vector is zero-length.")
    v_img /= nv

    # same beam-local basis used by image_to_base_frame() / build_lumen_from_parametric_boundaries()
    ref = np.array([
        ref_pt_px[0] - base_px[0],
        -(ref_pt_px[1] - base_px[1]),
    ], dtype=float)
    nr = np.linalg.norm(ref)
    if nr < 1e-12:
        raise ValueError("Base-to-reference vector is zero-length.")
    ex = ref / nr
    ey = np.array([-ex[1], ex[0]], dtype=float)

    # project measured tangent into beam-local frame
    tx_local = -float(np.dot(v_img, ex))
    ty_local =  float(np.dot(v_img, ey))
    t_meas_local = np.array([tx_local, ty_local], dtype=float)
    t_meas_local /= (np.linalg.norm(t_meas_local) + 1e-12)

    # nearest centerline point to measured tip
    d2 = np.sum((C_mm[:, :2] - tip_xy_mm.reshape(1, 2)) ** 2, axis=1)
    i_near = int(np.argmin(d2))

    # local centerline tangent
    i0 = max(0, i_near - n_tangent_pts)
    i1 = min(len(C_mm) - 1, i_near + n_tangent_pts)
    if i1 <= i0:
        if i_near == 0:
            c_tan = C_mm[1, :2] - C_mm[0, :2]
        else:
            c_tan = C_mm[i_near, :2] - C_mm[i_near - 1, :2]
    else:
        c_tan = C_mm[i1, :2] - C_mm[i0, :2]

    c_tan /= (np.linalg.norm(c_tan) + 1e-12)

    # build lumen boundaries in the same local XY plane
    left_bd = np.zeros((len(C_mm), 2), dtype=float)
    right_bd = np.zeros((len(C_mm), 2), dtype=float)

    for i in range(len(C_mm)):
        if i == 0:
            t = C_mm[1, :2] - C_mm[0, :2]
        elif i == len(C_mm) - 1:
            t = C_mm[-1, :2] - C_mm[-2, :2]
        else:
            t = C_mm[i + 1, :2] - C_mm[i - 1, :2]

        nt = np.linalg.norm(t)
        if nt < 1e-12:
            n = np.array([0.0, 0.0])
        else:
            t = t / nt
            n = np.array([-t[1], t[0]])

        left_bd[i] = C_mm[i, :2] + R_mm[i] * n
        right_bd[i] = C_mm[i, :2] - R_mm[i] * n

    # angle between measured tangent and local centerline tangent
    dot_tc = np.clip(float(np.dot(t_meas_local, c_tan)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot_tc))
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg

    # plot
    plt.figure(figsize=(8, 8))
    plt.plot(C_mm[:, 0], C_mm[:, 1], "-", label="lumen centerline")
    plt.plot(left_bd[:, 0], left_bd[:, 1], "--", label="lumen boundary +R")
    plt.plot(right_bd[:, 0], right_bd[:, 1], "--", label="lumen boundary -R")

    plt.plot(tip_xy_mm[0], tip_xy_mm[1], "ro", label="measured tip")
    plt.plot(C_mm[i_near, 0], C_mm[i_near, 1], "ks", label="nearest centerline point")

    # measured tip tangent arrow
    plt.arrow(
        tip_xy_mm[0], tip_xy_mm[1],
        tangent_scale_mm * t_meas_local[0],
        tangent_scale_mm * t_meas_local[1],
        head_width=0.8, length_includes_head=True
    )

    # centerline tangent arrow
    plt.arrow(
        C_mm[i_near, 0], C_mm[i_near, 1],
        tangent_scale_mm * c_tan[0],
        tangent_scale_mm * c_tan[1],
        head_width=0.8, length_includes_head=True
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("local x [mm]")
    plt.ylabel("local y [mm]")
    plt.title(
        f"Tip measurement vs lumen local frame\n"
        f"nearest idx={i_near}, tip-centerline dist={np.sqrt(d2[i_near]):.2f} mm, "
        f"angle={angle_deg:.2f} deg"
    )
    plt.legend()
    plt.show()

    # print("[DBG LOCAL PLOT]")
    # print("  tip_xy_mm =", tip_xy_mm)
    # print("  nearest centerline idx =", i_near)
    # print("  nearest centerline point [mm] =", C_mm[i_near, :2])
    # print("  tip-centerline distance [mm] =", np.sqrt(d2[i_near]))
    # print("  measured tangent local =", t_meas_local)
    # print("  centerline tangent local =", c_tan)
    # print("  angle between tangents [deg] =", angle_deg)
def compute_tip_wall_distances_general(tip_px, left_boundary_px, right_boundary_px, tip_radius_px=0.0):
    left_pt, dist_left_center = closest_point_on_polyline(tip_px, left_boundary_px)
    right_pt, dist_right_center = closest_point_on_polyline(tip_px, right_boundary_px)

    dist_left_edge = dist_left_center - tip_radius_px
    dist_right_edge = dist_right_center - tip_radius_px

    if dist_left_edge <= dist_right_edge:
        closest_wall = "left"
        closest_distance = dist_left_edge
    else:
        closest_wall = "right"
        closest_distance = dist_right_edge

    return {
        "tip_x": float(tip_px[0]),
        "tip_y": float(tip_px[1]),
        "closest_left_point": left_pt,
        "closest_right_point": right_pt,
        "dist_left_center": float(dist_left_center),
        "dist_right_center": float(dist_right_center),
        "dist_left_edge": float(dist_left_edge),
        "dist_right_edge": float(dist_right_edge),
        "closest_wall": closest_wall,
        "closest_distance": float(closest_distance),
    }

def load_calibration_points(path="/home/jack/Proper-Research/calibration_points.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data
def get_saved_2_point_calibration(path="calibration_points.json"):
    data = load_calibration_points(path)
    return {
        "points_px": [tuple(map(float, p)) for p in data["points_px"]],
        "distance_px": float(data["distance_px"]),
        "mask": None,
        "mode": "manual_click",
    }
def reconstruct_beam_within_vessel(
    image_filename="focused_image.jpg",
    red_roi_path="/home/jack/Proper-Research/custom_area.json",
    red_roi_polygon=None,
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=False,
    save_overlay_path=None,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
):
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    red_area = load_search_area(red_roi_path)
    blue_area = load_search_area(blue_roi_path)
    green_area = load_search_area(green_roi_path)

    # --- green calibration points ---
    # green_result = detect_2_green_calibration_points(
    #     image_bgr=image_bgr,
    #     roi_box=green_roi_box,
    #     green_h_low=35,
    #     green_h_high=95,
    #     sat_min=40,
    #     val_min=40,
    #     min_area=3,
    #     max_area=50000,
    #     show_debug=True,
    # )
    green_result = get_saved_2_point_calibration("/home/jack/Proper-Research/calibration_points.json")
    green_pt1, green_pt2 = green_result["points_px"]
    
    mm_per_pixel = compute_mm_per_pixel(green_pt1, green_pt2, known_distance_mm=23)
    red_area = load_search_area(red_roi_path)
    # --- red markers / beam tip state ---
    red_box = red_area["box"] if (red_area is not None and red_area["type"] == "box") else None
    red_polygon = red_area["polygon"] if (red_area is not None and red_area["type"] == "polygon") else None

    # First detect markers once, only to get fallback frame if needed
    tip_result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=None,
        roi_polygon=red_roi_polygon,
        show=show,
        show_debug_markers=show,
        unwrap_angle=True,
        pivot_hint=pivot_hint,
    )

    markers = tip_result["markers"]

    manual_vessel = load_manual_vessel_boundaries_with_frame(MANUAL_VESSEL_BOUNDARY_FILE)

    if base_px_ref is None or ex_ref is None or ey_ref is None:
        base_px_ref = manual_vessel["base_px"]
        ex_ref = manual_vessel["ex_img"]
        ey_ref = manual_vessel["ey_img"]
    else:
        base_px_ref = np.asarray(base_px_ref, dtype=np.float32).reshape(2,)
        ex_ref = np.asarray(ex_ref, dtype=np.float32).reshape(2,)
        ey_ref = np.asarray(ey_ref, dtype=np.float32).reshape(2,)

    ex_ref = ex_ref / (np.linalg.norm(ex_ref) + 1e-12)
    ey_ref = ey_ref / (np.linalg.norm(ey_ref) + 1e-12)

    tip_result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=None,
        roi_polygon=red_roi_polygon,
        show=show,
        show_debug_markers=show,
        unwrap_angle=True,
        pivot_hint=pivot_hint,
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
    )

    markers = tip_result["markers"]
    left_smooth = manual_vessel["left_boundary_px"]
    right_smooth = manual_vessel["right_boundary_px"]
    vessel = {"mode": "manual"}
    print("[INFO] Using manually drawn vessel boundaries.")

    tip_xy_fixed = image_to_fixed_local_frame(
        markers["tip_px"],
        base_px_ref,
        ex_ref,
        ey_ref,
    )
    tip_result["tip_xy_from_base"] = (float(tip_xy_fixed[0]), float(tip_xy_fixed[1]))
    lumen_C_m, lumen_R_m, lumen_C_mm, lumen_R_mm = build_lumen_from_parametric_boundaries(
        left_smooth,
        right_smooth,
        base_px_ref=base_px_ref,
        ex_ref=ex_ref,
        ey_ref=ey_ref,
        mm_per_pixel=mm_per_pixel,
        z_mm=0.0,
    )
    print("[CONTACT LUMEN LOCAL CHECK]")
    print("C first mm:", lumen_C_m[0])
    print("C last  mm:", lumen_C_m[-1])
    print("x range mm:", lumen_C_m[:, 0].min(), lumen_C_m[:, 0].max())
    print("y range mm:", lumen_C_m[:, 1].min(), lumen_C_m[:, 1].max())
    print("R range mm:", lumen_R_m.min(), lumen_R_m.max())
    plot_contact_lumen_debug(
        lumen_C_m=lumen_C_m,
        lumen_R_m=lumen_R_m,
        out_path="debug_outputs/contact_lumen_debug.png",
    )
    # print("lumen_C_m first local =", lumen_C_m[0])
    # print("lumen_C_m last local  =", lumen_C_m[-1])
    # print("[DBG lumen frame actually used]")
    # print("  base_px_ref =", base_px_ref)
    # print("  ex_ref =", ex_ref)
    # print("  ey_ref =", ey_ref)
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # --- beam reconstruction ---
    # Original marker-ordered polyline, still useful as a fallback/debug path.
    beam_points_marker_px, ordered_pts = beam_polyline_from_markers(
        markers,
        n_samples_per_segment=40,
    )

    ordered_pts = np.asarray(ordered_pts, float)

    beam_length_px, skeleton, beam_path_px = compute_marker_anchored_black_beam_length_px(
        image_bgr=image_bgr,
        ordered_pts=ordered_pts,
        threshold=100,
        tube_radius_px=25,
        bridge_radius_px=8,
    )

    # Use the routed black-beam path as the beam centreline downstream.
    beam_points_px = np.asarray(beam_path_px, float)

    # Fallback if route failed or returned too few points.
    if beam_points_px.ndim != 2 or beam_points_px.shape[0] < 2 or beam_points_px.shape[1] != 2:
        print("[WARN] black-beam routed path failed; falling back to marker polyline")
        beam_points_px = np.asarray(beam_points_marker_px, float)
        beam_length_px = compute_beam_length_from_ordered_pts(ordered_pts)

    beam_length_mm = beam_length_px * mm_per_pixel
    beam_coeffs = None
    beam_degree = None
    # beam_length_mm = beam_length_px * mm_per_pixel
    # beam_length_px, skeleton, beam_path_px = compute_black_beam_length_px(
    #     image_gray,
    #     markers["base_px"],
    #     markers["tip_px"],
    #     threshold=100
    # )

    beam_length_mm = beam_length_px * mm_per_pixel
    beam_coeffs = None
    beam_degree = None

    comparison = compare_beam_to_vessel(
        beam_points_px=beam_points_px,
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
    )

    tip_radius_mm = 0.0
    tip_radius_px = mm_to_px_distance(tip_radius_mm, mm_per_pixel)
    tip_distance_info = compute_tip_wall_distances_general(
        tip_px=markers["tip_px"],
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
        tip_radius_px=tip_radius_px,
    )

    tip_wall_angle_info = compute_tip_to_both_wall_tangents(
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

        "lumen_C_m": lumen_C_m,
        "lumen_R_m": lumen_R_m,
        "lumen_C_mm": lumen_C_mm,
        "lumen_R_mm": lumen_R_mm,
    }

    # print("Beam-to-wall tangent angle (deg):", tip_wall_angle_info["beam_wall_tangent_angle_deg"])
    print("Calibration distance (px):", green_result["distance_px"])
    print("mm_per_pixel:", mm_per_pixel)
    print("Beam length (px):", beam_length_px)
    print("Beam length (mm):", beam_length_mm)
    draw_beam_and_vessel_overlay(
        image_bgr=image_bgr,
        beam_points_px=beam_points_px,
        left_boundary_px=left_smooth,
        right_boundary_px=right_smooth,
        markers=markers,
        red_area=red_area,
        blue_area=blue_area,
        tip_distance_info=tip_distance_info,
        tip_wall_angle_info=tip_wall_angle_info,
        save_path=save_overlay_path,
        show_plot=show,
    )
    if show:
        draw_lumen_centerline_overlay(
            image_bgr,
            lumen_C_mm,
            base_px_ref,
            ex_ref,
            ey_ref,
            mm_per_pixel,
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
def draw_lumen_centerline_overlay(image_bgr, lumen_C_mm, base_px_ref, ex_ref, ey_ref, mm_per_pixel):
    vis = image_bgr.copy()

    base = np.asarray(base_px_ref, dtype=np.float32)
    ex = np.asarray(ex_ref, dtype=np.float32)
    ey = np.asarray(ey_ref, dtype=np.float32)

    # optional safety
    ex = ex / (np.linalg.norm(ex) + 1e-12)
    ey = ey / (np.linalg.norm(ey) + 1e-12)

    pts = []
    for x_mm, y_mm, z_mm in lumen_C_mm:
        # local -> image-cartesian
        # because local x was defined as negative along ex_ref
        v_mm = (-x_mm) * ex + (y_mm) * ey

        x_px = base[0] + (v_mm[0] / mm_per_pixel)
        y_px = base[1] - (v_mm[1] / mm_per_pixel)   # cartesian y-up -> image y-down
        pts.append((int(round(x_px)), int(round(y_px))))

    for i in range(len(pts) - 1):
        cv2.line(vis, pts[i], pts[i + 1], (255, 255, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Camera-derived lumen centerline (fixed frame)")
    plt.axis("off")
    plt.show()
def detect_2_green_calibration_points(
    image_bgr,
    roi_box=None,
    roi_polygon=None,
    green_h_low=15,
    green_h_high=110,
    sat_min=20,
    val_min=20,
    min_area=5,
    max_area=50000,
    show_debug=False,
):
    if roi_polygon is not None:
        pts = np.array(roi_polygon, dtype=np.float32)
        xs = pts[:, 0]
        ys = pts[:, 1]

        x0 = int(np.floor(xs.min()))
        y0 = int(np.floor(ys.min()))
        x1 = int(np.ceil(xs.max()))
        y1 = int(np.ceil(ys.max()))

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image_bgr.shape[1], x1)
        y1 = min(image_bgr.shape[0], y1)

        roi = image_bgr[y0:y1, x0:x1].copy()

        mask_spatial = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        pts_local = np.array(
            [[int(px - x0), int(py - y0)] for px, py in roi_polygon],
            dtype=np.int32
        )
        cv2.fillPoly(mask_spatial, [pts_local], 255)

    elif roi_box is not None:
        x0, y0, w, h = roi_box
        roi = image_bgr[y0:y0 + h, x0:x0 + w].copy()
        mask_spatial = np.full((roi.shape[0], roi.shape[1]), 255, dtype=np.uint8)

    else:
        x0, y0 = 0, 0
        roi = image_bgr.copy()
        mask_spatial = np.full((roi.shape[0], roi.shape[1]), 255, dtype=np.uint8)


    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([green_h_low, sat_min, val_min], dtype=np.uint8)
    upper = np.array([green_h_high, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, mask_spatial)
    # only opening; avoid closing because it merges blobs
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.bitwise_and(mask, mask_spatial)
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

    vis = roi.copy()

    if len(candidates) >= 2:
        candidates = sorted(candidates, key=lambda d: d["area"], reverse=True)[:2]
        pts = []
        for c in candidates:
            cx, cy = c["centroid_roi"]
            pts.append((float(cx + x0), float(cy + y0)))
        pts = sorted(pts, key=lambda p: p[0])
        mode = "two_blobs"

    elif len(candidates) == 1:
        cnt = candidates[0]["contour"]
        pts_cnt = cnt.reshape(-1, 2).astype(np.float32)

        left_idx = int(np.argmin(pts_cnt[:, 0]))
        right_idx = int(np.argmax(pts_cnt[:, 0]))

        left_pt_roi = pts_cnt[left_idx]
        right_pt_roi = pts_cnt[right_idx]

        pts = [
            (float(left_pt_roi[0] + x0), float(left_pt_roi[1] + y0)),
            (float(right_pt_roi[0] + x0), float(right_pt_roi[1] + y0)),
        ]
        pts = sorted(pts, key=lambda p: p[0])
        mode = "split_one_blob"

    else:
        raise RuntimeError("Expected at least 1 green blob, found 0")

    p1 = np.array(pts[0], dtype=np.float32)
    p2 = np.array(pts[1], dtype=np.float32)
    distance_px = float(np.linalg.norm(p2 - p1))

    if show_debug:
        if len(candidates) > 0:
            for c in candidates:
                cv2.drawContours(vis, [c["contour"]], -1, (255, 255, 255), 2)

        for i, (px, py) in enumerate(pts):
            px_roi = int(round(px - x0))
            py_roi = int(round(py - y0))
            cv2.circle(vis, (px_roi, py_roi), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"G{i+1}", (px_roi + 6, py_roi - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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
        plt.title(f"{mode}, dist = {distance_px:.2f} px")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "points_px": pts,
        "distance_px": distance_px,
        "mask": mask,
        "mode": mode,
    }
def compute_mm_per_pixel(p1_px, p2_px, known_distance_mm=23):
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
def order_beam_markers(points, pivot_hint=None):
    pts = [np.array(p, dtype=np.float32) for p in points]

    if len(pts) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 points, got {len(pts)}")

    # choose base
    if pivot_hint is not None:
        pivot_hint = np.array(pivot_hint, dtype=np.float32)
        start_idx = int(np.argmin([np.linalg.norm(p - pivot_hint) for p in pts]))
    else:
        start_idx = int(np.argmin([p[0] for p in pts]))

    base = pts.pop(start_idx)

    # chain remaining points by nearest-neighbour from base
    ordered = [base]
    while pts:
        last = ordered[-1]
        next_idx = int(np.argmin([np.linalg.norm(p - last) for p in pts]))
        ordered.append(pts.pop(next_idx))

    # If 4 markers exist: [base, mag_start, tangent_start, tip]
    if len(ordered) == 4:
        base, mag_start, tangent_start, tip = ordered

    # If 3 markers exist, mag_start is assumed missing:
    # [base, tangent_start, tip]
    else:
        base, tangent_start, tip = ordered
        mag_start = None

    return (
        tuple(map(float, base)),
        None if mag_start is None else tuple(map(float, mag_start)),
        tuple(map(float, tangent_start)),
        tuple(map(float, tip)),
    )

def image_to_base_frame(point_px, base_px, mag_start_px):
    # image -> Cartesian
    v = np.array([
        float(point_px[0] - base_px[0]),
        float(-(point_px[1] - base_px[1])),
    ], dtype=np.float32)

    ref = np.array([
        float(mag_start_px[0] - base_px[0]),
        float(-(mag_start_px[1] - base_px[1])),
    ], dtype=np.float32)

    nr = np.linalg.norm(ref)
    if nr < 1e-12:
        raise ValueError("base-to-magnet reference vector is zero length")

    ex = ref / nr                    # beam reference direction in image-cartesian frame
    ey = np.array([-ex[1], ex[0]], dtype=np.float32)   # +90 deg perpendicular

    # Beam-local convention:
    # straight beam should be along negative local x
    x_local = -float(np.dot(v, ex))
    y_local =  float(np.dot(v, ey))

    return np.array([x_local, y_local], dtype=np.float32)


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

def order_beam_marker_candidates(candidates, pivot_hint=None):
    items = list(candidates)

    if len(items) not in (3, 4):
        raise ValueError(f"Expected 3 or 4 candidates, got {len(items)}")

    pts = [np.array(c["point"], dtype=np.float32) for c in items]

    if pivot_hint is not None:
        pivot_hint = np.array(pivot_hint, dtype=np.float32)
        start_idx = int(np.argmin([np.linalg.norm(p - pivot_hint) for p in pts]))
    else:
        start_idx = int(np.argmin([p[0] for p in pts]))

    ordered = [items.pop(start_idx)]

    while items:
        last_pt = np.array(ordered[-1]["point"], dtype=np.float32)
        next_idx = int(np.argmin([
            np.linalg.norm(np.array(c["point"], dtype=np.float32) - last_pt)
            for c in items
        ]))
        ordered.append(items.pop(next_idx))

    if len(ordered) == 4:
        base_c, mag_start_c, tangent_start_c, tip_c = ordered
    else:
        base_c, tangent_start_c, tip_c = ordered
        mag_start_c = None

    return base_c, mag_start_c, tangent_start_c, tip_c
def signed_angle_between_vectors(v1, v2):
    a1 = np.arctan2(v1[1], v1[0])
    a2 = np.arctan2(v2[1], v2[0])
    ang = np.degrees(a2 - a1)

    while ang > 180:
        ang -= 360
    while ang < -180:
        ang += 360

    return float(ang)
def normalize_2d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(2,)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n
def measure_tip_state_4markers(
    image_filename="focused_image.jpg",
    roi_box=None,
    roi_polygon=None,
    show=True,
    show_debug_markers=False,
    unwrap_angle=True,
    pivot_hint=None,
    base_px_ref=None,
    ex_ref=None,
    ey_ref=None,
):
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_filename}")

    detected_candidates = detect_4_red_markers_in_roi(
        image_bgr,
        roi_box=roi_box,
        roi_polygon=roi_polygon,
        min_area=4,
        max_area=40000,
        sat_min=50,
        val_min=40,
        hue1_high=10,
        hue2_low=170,
        show_debug=show_debug_markers,
        min_markers=3,
        max_markers=4,
    )

    if detected_candidates is None or len(detected_candidates) < 3:
        raise RuntimeError(
            f"Expected at least 3 detected red markers, got: {detected_candidates}"
        )

    base_c, mag_start_c, tangent_start_c, tip_c = order_beam_marker_candidates(
        detected_candidates,
        pivot_hint=pivot_hint
    )

    base_px = base_c["point"]
    mag_start_px = None if mag_start_c is None else mag_start_c["point"]
    tangent_start_px = tangent_start_c["point"]
    tip_px = tip_c["point"]

    base = np.array(base_px, dtype=np.float32)
    tangent_start = np.array(tangent_start_px, dtype=np.float32)
    tip = np.array(tip_px, dtype=np.float32)

    mag_start = None if mag_start_px is None else np.array(mag_start_px, dtype=np.float32)

    # -------------------------------------------------
    # USE STRAIGHT-IMAGE FRAME IF PROVIDED
    # -------------------------------------------------
    if base_px_ref is not None and ex_ref is not None and ey_ref is not None:
        base_ref = np.asarray(base_px_ref, dtype=float).reshape(2,)
        base_cart = np.array([base_ref[0], -base_ref[1]], dtype=float)

        ex_fit = np.asarray(ex_ref, dtype=float).reshape(2,)
        ey_fit = np.asarray(ey_ref, dtype=float).reshape(2,)

        ex_fit = ex_fit / (np.linalg.norm(ex_fit) + 1e-12)
        ey_fit = ey_fit / (np.linalg.norm(ey_fit) + 1e-12)
    else:
        if mag_start is not None:
            base_cart, ex_fit, ey_fit = make_beam_frame_from_proximal_markers(
                base_px=base,
                mag_start_px=mag_start,
                tangent_start_px=tangent_start,
            )
        else:
            # fallback: use base -> tangent_start as beam axis
            base_cart = np.array([base[0], -base[1]], dtype=float)

            ex_fit = np.array([
                tangent_start[0] - base[0],
                -(tangent_start[1] - base[1]),
            ], dtype=float)
            ex_fit = ex_fit / (np.linalg.norm(ex_fit) + 1e-12)

            ey_fit = np.array([-ex_fit[1], ex_fit[0]], dtype=float)

    tip_xy = project_point_to_beam_frame(
        point_px=tip,
        origin_cart=base_cart,
        ex=ex_fit,
        ey=ey_fit,
    )

    # tangent in Cartesian convention
    tan_vec = np.array([
        tip[0] - tangent_start[0],
        -(tip[1] - tangent_start[1])
    ], dtype=np.float32)

    ref_vec = ex_fit.astype(np.float32)

    tangent_angle_deg = signed_angle_between_vectors(ref_vec, tan_vec)

    if unwrap_angle:
        tangent_angle_deg = unwrap_tip_angle(tangent_angle_deg)

    tip_base_angle_info = measure_tip_base_angles(
        base_px=base if base_px_ref is None else np.asarray(base_px_ref, dtype=np.float32),
        tip_px=tip,
        ex_ref=ex_fit,
    )

    result = {
        "tip_image_px": (float(tip[0]), float(tip[1])),
        "tip_xy_from_base": (float(tip_xy[0]), float(tip_xy[1])),
        "tip_tangent_angle_deg": float(tangent_angle_deg),
        "tip_base_abs_angle_deg": float(tip_base_angle_info["tip_base_abs_angle_deg"]),
        "tip_base_angle_from_vertical_deg": float(tip_base_angle_info["tip_base_angle_from_vertical_deg"]),
        "tip_base_angle_from_ref_deg": float(tip_base_angle_info["tip_base_angle_from_ref_deg"]),
        "markers": {
            "base_px": tuple(map(float, base_px)),
            "mag_start_px": None if mag_start_px is None else tuple(map(float, mag_start_px)),
            "tangent_start_px": tuple(map(float, tangent_start_px)),
            "tip_px": tuple(map(float, tip_px)),
        },
        "marker_diameters_px": {
            "base_px": float(base_c["diameter_px"]),
            "mag_start_px": None if mag_start_c is None else float(mag_start_c["diameter_px"]),
            "tangent_start_px": float(tangent_start_c["diameter_px"]),
            "tip_px": float(tip_c["diameter_px"]),
        },
        "marker_areas_px": {
            "base_px": float(base_c["area"]),
            "mag_start_px": None if mag_start_c is None else float(mag_start_c["area"]),
            "tangent_start_px": float(tangent_start_c["area"]),
            "tip_px": float(tip_c["area"]),
        },
        "beam_frame_fit": {
            "origin_cart": tuple(map(float, base_cart)),
            "ex": tuple(map(float, ex_fit)),
            "ey": tuple(map(float, ey_fit)),
        },
        "roi_box": roi_box,
        "roi_polygon": roi_polygon,
    }

    # print("[DBG markers]")
    # print("  base_px         =", base_px)
    # print("  mag_start_px    =", mag_start_px)
    # print("  tangent_start_px=", tangent_start_px)
    # print("  tip_px          =", tip_px)

    # print("[DBG fitted frame]")
    # print("  ex_fit =", ex_fit)
    # print("  ey_fit =", ey_fit)

    # print("[DBG converted]")
    # print("  tip_xy_from_base =", tip_xy)

    if show:
        vis = image_bgr.copy()

        if roi_polygon is not None:
            pts = np.array(roi_polygon, dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    elif roi_box is not None:
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

        # draw the REFERENCE frame through reference base
        if base_px_ref is not None:
            p0 = np.array(base_px_ref, dtype=float)
        else:
            p0 = np.array([base[0], base[1]], dtype=float)

        ex_img = np.array([ex_fit[0], -ex_fit[1]], dtype=float)
        p1 = p0 - 300 * ex_img
        p2 = p0 + 300 * ex_img

        cv2.line(
            vis,
            (int(round(p1[0])), int(round(p1[1]))),
            (int(round(p2[0])), int(round(p2[1]))),
            (255, 255, 0),
            2,
        )

        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Tip measurement in fixed straight-reference frame")
        plt.axis("off")
        plt.show()

    return result
def draw_arrow_from_point(vis, p0, vec, color, length_px=120, thickness=2):
    p0 = np.asarray(p0, dtype=float).reshape(2,)
    vec = np.asarray(vec, dtype=float).reshape(2,)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return

    u = vec / n
    p1 = p0 + length_px * u

    cv2.arrowedLine(
        vis,
        (int(round(p0[0])), int(round(p0[1]))),
        (int(round(p1[0])), int(round(p1[1]))),
        color,
        thickness,
        tipLength=0.15,
    )
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
def compute_tip_to_both_wall_tangents(markers, left_boundary_px, right_boundary_px, tip_distance_info, dy=5.0):
    tip = np.array(markers["tip_px"], dtype=np.float32)
    tangent_start = np.array(markers["tangent_start_px"], dtype=np.float32)

    beam_tangent_vec = np.array([
        tip[0] - tangent_start[0],
        -(tip[1] - tangent_start[1])
    ], dtype=np.float32)

    tip_y = float(tip[1])

    left_info = compute_boundary_tangent_vector_at_y(left_boundary_px, tip_y, dy=dy)
    right_info = compute_boundary_tangent_vector_at_y(right_boundary_px, tip_y, dy=dy)

    left_tangent = left_info["tangent_vec_cartesian"]
    right_tangent = right_info["tangent_vec_cartesian"]

    left_angle = unsigned_angle_between_vectors(beam_tangent_vec, left_tangent)
    right_angle = unsigned_angle_between_vectors(beam_tangent_vec, right_tangent)

    if left_angle > 90.0:
        left_angle = 180.0 - left_angle
    if right_angle > 90.0:
        right_angle = 180.0 - right_angle

    return {
        "closest_wall": tip_distance_info["closest_wall"],
        "beam_tangent_vec_cartesian": beam_tangent_vec,

        "left_wall_tangent_vec_cartesian": left_tangent,
        "left_wall_tangent_points": {
            "p_minus": left_info["p_minus"],
            "p_plus": left_info["p_plus"],
        },
        "beam_left_wall_tangent_angle_deg": float(left_angle),

        "right_wall_tangent_vec_cartesian": right_tangent,
        "right_wall_tangent_points": {
            "p_minus": right_info["p_minus"],
            "p_plus": right_info["p_plus"],
        },
        "beam_right_wall_tangent_angle_deg": float(right_angle),
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
def image_to_fixed_local_frame(point_px, base_px_ref, ex_ref, ey_ref):
    v = np.array([
        float(point_px[0] - base_px_ref[0]),
        float(-(point_px[1] - base_px_ref[1])),
    ], dtype=np.float32)

    x_local = -float(np.dot(v, ex_ref))
    y_local =  float(np.dot(v, ey_ref))

    return np.array([x_local, y_local], dtype=np.float32)
def load_polygon(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    if "points" not in data:
        raise ValueError(f"No 'points' key found in polygon file: {path}")

    polygon = [tuple(map(float, p)) for p in data["points"]]
    return polygon
if __name__ == "__main__":
    new_capture()

    pivot_hint = (300, 391)
    roi_polygon = load_polygon("/home/jack/Proper-Research/custom_area.json")

    result = reconstruct_beam_within_vessel(
        image_filename="/home/jack/Proper-Research/focused_image.jpg",
        red_roi_polygon=roi_polygon,
        blue_roi_path="blue_roi_box.json",
        pivot_hint=pivot_hint,
        show=True,
        save_overlay_path="debug_outputs/reconstruction_overlay.png",
    )

    plot_tip_measurement_vs_lumen_local(result)
