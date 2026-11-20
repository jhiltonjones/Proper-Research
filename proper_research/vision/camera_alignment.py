import cv2

def run_camera_alignment(camera_index=0, draw_crosshair=True, draw_vertical_only=False):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        h, w = frame.shape[:2]

        # Center of image
        cx = w // 2
        cy = h // 2

        # Draw lines
        if draw_vertical_only:
            cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
        elif draw_crosshair:
            # Vertical line
            cv2.line(frame, (cx, 0), (cx, h), (0, 255, 0), 2)
            # Horizontal line
            cv2.line(frame, (0, cy), (w, cy), (0, 255, 0), 2)

        # Optional: center dot
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.imshow("Camera Alignment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_alignment(
        camera_index=0,
        draw_crosshair=True,     # set False if you only want vertical
        draw_vertical_only=False # set True for single axis
    )