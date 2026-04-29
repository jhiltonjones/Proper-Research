import numpy as np
import cv2
import os
import json

MANUAL_VESSEL_BOUNDARY_FILE = "manual_vessel_boundaries.json"


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def save_manual_vessel_boundaries_with_frame(
    left_boundary_px,
    right_boundary_px,
    base_px,
    ref_px,
    path,
):
    base_px = np.asarray(base_px, float)
    ref_px = np.asarray(ref_px, float)

    ex_img = unit(base_px - ref_px)          # local -x from clicked ref direction         # local +x direction in image pixels
    ey_img = np.array([-ex_img[1], ex_img[0]])  # local +y, right-handed in image plane

    data = {
        "base_px": base_px.tolist(),
        "ref_px": ref_px.tolist(),
        "ex_img": ex_img.tolist(),
        "ey_img": ey_img.tolist(),
        "left_boundary_px": [[float(x), float(y)] for x, y in left_boundary_px],
        "right_boundary_px": [[float(x), float(y)] for x, y in right_boundary_px],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Saved manual vessel boundaries with frame to {path}")
    print("[FRAME] base_px =", base_px)
    print("[FRAME] ref_px  =", ref_px)
    print("[FRAME] ex_img  =", ex_img)
    print("[FRAME] ey_img  =", ey_img)


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


def resample_polyline_by_arclength(points, n_samples=200):
    pts = np.asarray(points, dtype=np.float32)
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


def load_roi_box(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])


def draw_manual_vessel_boundaries_with_origin_and_axis(
    image_filename="focused_image.jpg",
    save_path=MANUAL_VESSEL_BOUNDARY_FILE,
    blue_roi_path="blue_roi_box.json",
):
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

    current_mode = {"name": "base"}  # base, ref, left, right

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
            "b=base/origin | a=axis/ref | l=left wall | r=right wall | z=undo | c=clear mode | s=save | q=quit",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
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
                    preview = resample_polyline_by_arclength(pts, n_samples=200)
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

        elif key == ord("z"):
            mode = current_mode["name"]
            if mode == "left" and left_points:
                left_points.pop()
            elif mode == "right" and right_points:
                right_points.pop()
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
            elif mode == "base":
                base_point["pt"] = None
            elif mode == "ref":
                ref_point["pt"] = None
            redraw()

        elif key == ord("s"):
            if base_point["pt"] is None:
                print("[WARN] Set base/origin first: press b, then click base.")
                continue

            if ref_point["pt"] is None:
                print("[WARN] Set reference axis first: press a, then click a point along desired +x.")
                continue

            if len(left_points) < 2 or len(right_points) < 2:
                print("[WARN] Need at least 2 points on each wall.")
                continue    

            left_boundary_roi = resample_polyline_by_arclength(left_points, n_samples=200)                  
            right_boundary_roi = resample_polyline_by_arclength(right_points, n_samples=200)

            left_boundary_px = [to_full_px(p) for p in left_boundary_roi]
            right_boundary_px = [to_full_px(p) for p in right_boundary_roi]
            base_px_clicked = to_full_px(base_point["pt"])
            ref_px_clicked = to_full_px(ref_point["pt"])

            # snap base onto same image-y as the start of the two vessel walls
            base_px = snap_base_to_boundary_start_y(
                base_px_clicked,
                left_boundary_px,
                right_boundary_px,
            )

            # keep reference point on same vertical offset relative to the snapped base
            dy_snap = base_px[1] - base_px_clicked[1]
            ref_px = (ref_px_clicked[0], ref_px_clicked[1] + dy_snap)

            save_manual_vessel_boundaries_with_frame(
                left_boundary_px=left_boundary_px,              
                right_boundary_px=right_boundary_px,
                base_px=base_px,
                ref_px=ref_px,
                path=save_path,
            )

            cv2.destroyWindow(window_name)

            return {
                "left_boundary_px": left_boundary_px,
                "right_boundary_px": right_boundary_px,
                "base_px": base_px,
                "ref_px": ref_px,
            }

        elif key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            print("[INFO] Boundary drawing cancelled.")
            return None
def snap_base_to_boundary_start_y(base_px, left_boundary_px, right_boundary_px):
    base_px = np.asarray(base_px, float).copy()
    left0 = np.asarray(left_boundary_px[0], float)
    right0 = np.asarray(right_boundary_px[0], float)

    start_y = 0.5 * (left0[1] + right0[1])
    base_px[1] = start_y

    return tuple(base_px)
if __name__ == "__main__":
    manual = draw_manual_vessel_boundaries_with_origin_and_axis(
        image_filename="focused_image.jpg",
        save_path=MANUAL_VESSEL_BOUNDARY_FILE,
        blue_roi_path="blue_roi_box.json",
    )