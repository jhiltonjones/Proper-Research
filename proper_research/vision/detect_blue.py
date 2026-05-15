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
            "b=base | a=axis/ref | l=left wall | r=right wall | m=centerline | z=undo | c=clear | s=save | q=quit",
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
            if len(center_points) < 2:
                print("[WARN] Need at least 2 centerline points before live overlay. Press m and click centerline.")
                continue

            centerline_clicked_px = [to_full_px(p) for p in center_points]

            left_boundary_px, right_boundary_px, centerline_px = centerline_to_offset_boundaries(
                centerline_clicked_px,
                radius_px=14.0,
                samples_per_segment=30,
            )

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

        elif key == ord("s"):
            if base_point["pt"] is None:
                print("[WARN] Set base/origin first: press b, then click base.")
                continue

            if ref_point["pt"] is None:
                print("[WARN] Set reference axis first: press a, then click a point along desired +x.")
                continue

            if len(center_points) < 2:
                print("[WARN] Need at least 2 centerline points. Press m and click the centerline.")
                continue

            base_px_clicked = to_full_px(base_point["pt"])
            ref_px_clicked = to_full_px(ref_point["pt"])

            centerline_clicked_px = [to_full_px(p) for p in center_points]

            left_boundary_px, right_boundary_px, centerline_px = centerline_to_offset_boundaries(
                centerline_clicked_px,
                radius_px=16.0,
                samples_per_segment=30,
            )

            show_boundary_preview(
                display,
                left_boundary_px,
                right_boundary_px,
                centerline_px,
                x0=x0,
                y0=y0,
                window_name="Boundary preview before save",
            )

            print("[PREVIEW] Press y to accept/save, n to reject and continue editing.")

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

    left_resampled = np.asarray(
        resample_polyline_by_arclength_preserve_vertices(
            left_smooth,
            samples_per_segment=30,
        ),
        float,
    )

    right_resampled = np.asarray(
        resample_polyline_by_arclength_preserve_vertices(
            right_smooth,
            samples_per_segment=30,
        ),
        float,
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
if __name__ == "__main__":
    manual = draw_manual_vessel_boundaries_with_origin_and_axis(
        image_filename="focused_image.jpg",
        save_path=MANUAL_VESSEL_BOUNDARY_FILE,
        blue_roi_path="blue_roi_box.json",
    )