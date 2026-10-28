import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from proper_research.vision.measure_length import new_capture
from proper_research.vision.bounds_beam import measure_tip_state_4markers


# ============================================================
# Config
# ============================================================

REFERENCE_IMAGE = "focused_image_straight.jpg"
LIVE_IMAGE = "focused_image.jpg"
RED_ROI_PATH = "red_roi_box.json"

PIVOT_HINT = (318.200927734375, 369.6798095703125)

CAPTURE_DELAY_S = 0.2
ANGLE_TOL_DEG = 0.5

SHOW_DEBUG_MARKERS = False


# ============================================================
# Helpers
# ============================================================

def load_roi_box(path: str):
    if not os.path.exists(path):
        return None
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))


def unit(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"Zero-length vector: {v}")
    return v / n


def signed_angle_deg_2d(v_ref, v_cur):
    """
    Signed angle from v_ref to v_cur in degrees.
    Positive = CCW in Cartesian coordinates.
    """
    a = unit(v_ref[:2])
    b = unit(v_cur[:2])

    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    det = a[0] * b[1] - a[1] * b[0]

    return float(np.degrees(np.arctan2(det, dot)))


def get_beam_frame_from_image(image_filename, roi_box, pivot_hint, show=False):
    result = measure_tip_state_4markers(
        image_filename=image_filename,
        roi_box=roi_box,
        show=show,
        show_debug_markers=SHOW_DEBUG_MARKERS,
        unwrap_angle=False,
        pivot_hint=pivot_hint,
        base_px_ref=None,
        ex_ref=None,
        ey_ref=None,
    )

    base_px = np.asarray(result["markers"]["base_px"], dtype=float)
    ex = np.asarray(result["beam_frame_fit"]["ex"], dtype=float)
    ey = np.asarray(result["beam_frame_fit"]["ey"], dtype=float)

    ex = unit(ex)
    ey = unit(ey)

    return {
        "base_px": base_px,
        "ex": ex,
        "ey": ey,
        "raw": result,
    }


def draw_alignment_overlay(
    image_filename,
    ref_frame,
    cur_frame,
    angle_err_deg,
    out_path=None,
):
    image_bgr = cv2.imread(image_filename)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_filename}")

    vis = image_bgr.copy()

    base_ref = ref_frame["base_px"]
    ex_ref = ref_frame["ex"]

    base_cur = cur_frame["base_px"]
    ex_cur = cur_frame["ex"]

    # beam_frame_fit is Cartesian-like:
    # ex = [x_right, y_up]
    # image y is downward, so convert for drawing
    def to_img_dir(v):
        return np.array([v[0], -v[1]], dtype=float)

    ex_ref_img = to_img_dir(ex_ref)
    ex_cur_img = to_img_dir(ex_cur)

    L = 120.0

    p0_ref = base_ref
    p1_ref = p0_ref + L * ex_ref_img

    p0_cur = base_cur
    p1_cur = p0_cur + L * ex_cur_img

    # reference axis = cyan
    cv2.arrowedLine(
        vis,
        (int(round(p0_ref[0])), int(round(p0_ref[1]))),
        (int(round(p1_ref[0])), int(round(p1_ref[1]))),
        (255, 255, 0),
        2,
        tipLength=0.12,
    )

    # current axis = magenta
    cv2.arrowedLine(
        vis,
        (int(round(p0_cur[0])), int(round(p0_cur[1]))),
        (int(round(p1_cur[0])), int(round(p1_cur[1]))),
        (255, 0, 255),
        2,
        tipLength=0.12,
    )

    # base markers
    cv2.circle(vis, (int(round(base_ref[0])), int(round(base_ref[1]))), 5, (255, 255, 0), -1)
    cv2.circle(vis, (int(round(base_cur[0])), int(round(base_cur[1]))), 5, (255, 0, 255), -1)

    txt1 = f"Angle error: {angle_err_deg:+.3f} deg"
    txt2 = f"Target tol: +/- {ANGLE_TOL_DEG:.2f} deg"
    txt3 = "cyan = reference, magenta = current"

    ok = abs(angle_err_deg) <= ANGLE_TOL_DEG
    txt4 = "ALIGNED" if ok else "NOT ALIGNED"

    cv2.putText(vis, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(vis, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    cv2.putText(vis, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(vis, txt3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(vis, txt3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    color = (0, 255, 0) if ok else (0, 0, 255)
    cv2.putText(vis, txt4, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(vis, txt4, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if out_path is not None:
        cv2.imwrite(out_path, vis)

    return vis


# ============================================================
# Main loop
# ============================================================

def main():
    roi_box = load_roi_box(RED_ROI_PATH)

    if not os.path.exists(REFERENCE_IMAGE):
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_IMAGE}\n"
            "First capture a straight reference image in the desired aligned pose."
        )

    print("\nLoading reference beam frame...")
    ref_frame = get_beam_frame_from_image(
        image_filename=REFERENCE_IMAGE,
        roi_box=roi_box,
        pivot_hint=PIVOT_HINT,
        show=False,
    )

    print("Reference base_px =", ref_frame["base_px"])
    print("Reference ex      =", ref_frame["ex"])
    print("Reference ey      =", ref_frame["ey"])

    print("\nControls:")
    print("  q = quit")
    print("  space = capture/update once")
    print("  continuous loop is running already\n")

    while True:
        try:
            new_capture()  # saves current image to LIVE_IMAGE
            time.sleep(CAPTURE_DELAY_S)

            cur_frame = get_beam_frame_from_image(
                image_filename=LIVE_IMAGE,
                roi_box=roi_box,
                pivot_hint=PIVOT_HINT,
                show=False,
            )

            angle_err_deg = signed_angle_deg_2d(ref_frame["ex"], cur_frame["ex"])

            print(
                f"Angle error = {angle_err_deg:+.4f} deg | "
                f"{'ALIGNED' if abs(angle_err_deg) <= ANGLE_TOL_DEG else 'adjust...'}"
            )

            vis = draw_alignment_overlay(
                image_filename=LIVE_IMAGE,
                ref_frame=ref_frame,
                cur_frame=cur_frame,
                angle_err_deg=angle_err_deg,
                out_path="alignment_overlay.png",
            )

            cv2.imshow("Pivot/tool alignment check", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                pass

        except Exception as e:
            print(f"[WARN] {e}")
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()