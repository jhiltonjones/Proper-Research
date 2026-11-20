import cv2
import json
import numpy as np


def collect_two_points_from_camera(
    camera_index=0,
    save_path="calibration_points.json",
    window_name="Click 2 calibration points",
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    clicked_points = []
    frozen_frame = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal clicked_points, frozen_frame

        if event == cv2.EVENT_LBUTTONDOWN and frozen_frame is not None:
            if len(clicked_points) < 2:
                clicked_points.append((int(x), int(y)))
                print(f"Point {len(clicked_points)}: {(x, y)}")

    cv2.namedWindow(window_name,  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Press SPACE to freeze frame")
    print("Click 2 calibration points")
    print("Press r to reset clicked points")
    print("Press s to save once 2 points are selected")
    print("Press q or ESC to quit")

    while True:
        if frozen_frame is None:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Failed to read frame from camera")
            display = frame.copy()
            cv2.putText(
                display,
                "Live view - press SPACE to freeze",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            display = frozen_frame.copy()
            cv2.putText(
                display,
                "Frozen - click 2 points, s=save, r=reset",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        for i, (px, py) in enumerate(clicked_points):
            cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"P{i+1}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if len(clicked_points) == 2:
            cv2.line(display, clicked_points[0], clicked_points[1], (0, 255, 255), 2)
            dist = float(np.linalg.norm(np.array(clicked_points[1]) - np.array(clicked_points[0])))
            cv2.putText(
                display,
                f"Distance: {dist:.2f} px",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 32:  # SPACE
            if frozen_frame is None:
                frozen_frame = frame.copy()
                clicked_points = []
                print("Frame frozen. Click 2 points.")
            else:
                frozen_frame = None
                clicked_points = []
                print("Back to live view.")

        elif key == ord("r"):
            clicked_points = []
            print("Points reset.")

        elif key == ord("s"):
            if len(clicked_points) != 2:
                print("Select exactly 2 points before saving.")
                continue

            pts = sorted(clicked_points, key=lambda p: p[0])  # left-to-right
            p1 = np.array(pts[0], dtype=np.float32)
            p2 = np.array(pts[1], dtype=np.float32)
            distance_px = float(np.linalg.norm(p2 - p1))

            data = {
                "points_px": pts,
                "distance_px": distance_px,
            }

            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Saved calibration to {save_path}")
            print(data)

        elif key == ord("q") or key == 27:  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def load_calibration_points(path="calibration_points.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    collect_two_points_from_camera()