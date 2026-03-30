
import numpy as np
import cv2
import os
import json

MANUAL_VESSEL_BOUNDARY_FILE = "manual_vessel_boundaries.json"

def build_lumen_from_parametric_boundaries(
    left_boundary_px,
    right_boundary_px,
    base_px,
    mm_per_pixel,
    z_mm=0.0,
):
    left = np.array(left_boundary_px, dtype=np.float32)
    right = np.array(right_boundary_px, dtype=np.float32)

    if len(left) != len(right):
        raise ValueError("Left and right boundaries must have same number of samples.")

    center = 0.5 * (left + right)
    radius_px = 0.5 * np.linalg.norm(right - left, axis=1)

    base_x = float(base_px[0])
    base_y = float(base_px[1])

    x_mm = (center[:, 0] - base_x) * mm_per_pixel
    y_mm = -(center[:, 1] - base_y) * mm_per_pixel
    z_mm_arr = np.full_like(x_mm, float(z_mm))

    lumen_C_mm = np.column_stack([x_mm, y_mm, z_mm_arr]).astype(np.float64)
    lumen_R_mm = (radius_px * mm_per_pixel).astype(np.float64)

    lumen_C_m = lumen_C_mm / 1000.0
    lumen_R_m = lumen_R_mm / 1000.0

    return lumen_C_m, lumen_R_m, lumen_C_mm, lumen_R_mm
def load_roi_box(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    box = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    print(f"[INFO] Loaded ROI from {path}: {data}")
    return box
def save_manual_vessel_boundaries(left_boundary_px, right_boundary_px, path):
    data = {
        "left_boundary_px": [[float(x), float(y)] for x, y in left_boundary_px],
        "right_boundary_px": [[float(x), float(y)] for x, y in right_boundary_px],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Manual vessel boundaries saved to {path}")


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
def resample_drawn_boundary(points, n_samples=200):
    """
    Resample manually clicked boundary points into a smooth x(y) boundary.
    Assumes boundary is single-valued in y.
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to resample a boundary.")

    xs = pts[:, 0]
    ys = pts[:, 1]

    order = np.argsort(ys)
    xs = xs[order]
    ys = ys[order]

    y_unique, unique_idx = np.unique(ys, return_index=True)
    x_unique = xs[unique_idx]

    if len(y_unique) < 2:
        raise ValueError("Boundary points must span at least 2 distinct y-values.")

    y_samples = np.linspace(float(y_unique.min()), float(y_unique.max()), n_samples)
    x_samples = np.interp(y_samples, y_unique, x_unique)

    return [(float(x), float(y)) for x, y in zip(x_samples, y_samples)]


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

    roi_box = load_roi_box(blue_roi_path)
    if roi_box is not None:
        x0, y0, w, h = roi_box
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

        text1 = f"Current wall: {current_side['name'].upper()}"
        text2 = "Left click=add | z=undo | l/r=switch | c=clear current | s=save | q=quit"
        cv2.putText(vis, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # raw clicked points
        for i, (x, y) in enumerate(left_points):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(
                    vis,
                    (int(left_points[i-1][0]), int(left_points[i-1][1])),
                    (int(x), int(y)),
                    (0, 120, 0),
                    1,
                )

        for i, (x, y) in enumerate(right_points):
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            if i > 0:
                cv2.line(
                    vis,
                    (int(right_points[i-1][0]), int(right_points[i-1][1])),
                    (int(x), int(y)),
                    (0, 0, 120),
                    1,
                )

        # preview resampled boundaries if enough points exist
        if len(left_points) >= 2:
            try:
                left_preview = resample_drawn_boundary(left_points, n_samples=200)
                for i in range(len(left_preview) - 1):
                    p1 = left_preview[i]
                    p2 = left_preview[i + 1]
                    cv2.line(
                        vis,
                        (int(round(p1[0])), int(round(p1[1]))),
                        (int(round(p2[0])), int(round(p2[1]))),
                        (0, 255, 255),
                        2,
                    )
            except Exception:
                pass

        if len(right_points) >= 2:
            try:
                right_preview = resample_drawn_boundary(right_points, n_samples=200)
                for i in range(len(right_preview) - 1):
                    p1 = right_preview[i]
                    p2 = right_preview[i + 1]
                    cv2.line(
                        vis,
                        (int(round(p1[0])), int(round(p1[1]))),
                        (int(round(p2[0])), int(round(p2[1]))),
                        (255, 255, 0),
                        2,
                    )
            except Exception:
                pass

        cv2.imshow(window_name, vis)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_side["name"] == "left":
                left_points.append((float(x), float(y)))
            else:
                right_points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

            # build common y support for both walls
            left_pts = np.array(left_points, dtype=np.float32)
            right_pts = np.array(right_points, dtype=np.float32)

            left_order = np.argsort(left_pts[:, 1])
            right_order = np.argsort(right_pts[:, 1])

            left_pts = left_pts[left_order]
            right_pts = right_pts[right_order]

            left_xs, left_ys = left_pts[:, 0], left_pts[:, 1]
            right_xs, right_ys = right_pts[:, 0], right_pts[:, 1]

            y_min = min(float(np.min(left_ys)), float(np.min(right_ys)))
            y_max = max(float(np.max(left_ys)), float(np.max(right_ys)))

            y_samples = np.linspace(y_min, y_max, 200)

            left_x_samples = np.interp(
                np.clip(y_samples, float(np.min(left_ys)), float(np.max(left_ys))),
                left_ys, left_xs
            )
            right_x_samples = np.interp(
                np.clip(y_samples, float(np.min(right_ys)), float(np.max(right_ys))),
                right_ys, right_xs
            )

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
manual = draw_manual_vessel_boundaries(
    image_filename="focused_image.jpg",
    save_path=MANUAL_VESSEL_BOUNDARY_FILE,
    blue_roi_path="blue_roi_box.json",
)