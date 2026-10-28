import os
import time
import cv2
import numpy as np

# =========================
# CONFIG (EDIT THESE)
# =========================

# IMPORTANT: use INNER corners, not squares.
# If your board is "8x6 squares", inner corners are typically (7, 5).
# If your board is "8x6 inner corners", use (8, 6).
INNER_CORNERS = (6, 8)      # (cols, rows) inner corners
CAM_INDEX = 0
BACKEND = cv2.CAP_V4L2      # on Linux; on Windows you can remove or use cv2.CAP_DSHOW
OUT_DIR = "calib_imgs"
TARGET_COUNT = 20           # capture 10–20
MIN_SECONDS_BETWEEN_SAVES = 0.6

# Optional camera controls (may or may not work depending on your camera/driver)
WARMUP_FRAMES = 15
AUTO_EXPOSURE_MANUAL = 1.0  # for some V4L2 cameras: 1.0=manual, 3.0=auto
EXPOSURE = 50.0
GAIN = 0.0

# =========================
# HELPERS
# =========================

def try_set(cap, prop, val, name):
    ok = cap.set(prop, val)
    got = cap.get(prop)
    print(f"{name}: set({val}) -> {ok}, get() -> {got}")
    return ok, got

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def next_filename(out_dir, idx):
    return os.path.join(out_dir, f"calib_{idx:03d}.jpg")

# =========================
# MAIN
# =========================

def main():
    ensure_dir(OUT_DIR)

    cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAM_INDEX}")

    # Optional camera tuning (safe to comment out if unsupported)
    try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, float(AUTO_EXPOSURE_MANUAL), "AUTO_EXPOSURE")
    try_set(cap, cv2.CAP_PROP_EXPOSURE, float(EXPOSURE), "EXPOSURE")
    try_set(cap, cv2.CAP_PROP_GAIN, float(GAIN), "GAIN")

    # warmup
    for _ in range(WARMUP_FRAMES):
        cap.read()

    print("\nControls:")
    print("  SPACE  -> save frame (only if checkerboard detected)")
    print("  r      -> reset counter (does not delete files)")
    print("  q/ESC  -> quit\n")
    print("Tips: move/tilt board; fill different parts of frame; vary angles.\n")

    saved = 0
    last_save_t = 0.0

    # Use classic chessboard detection flags (good default)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame grab failed.")
            break

        view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, INNER_CORNERS, flags)

        # If classic fails and SB exists, try SB for robustness
        if (not found) and hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray, INNER_CORNERS,
                flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )

        # Draw status
        status = f"Detected: {found} | Saved: {saved}/{TARGET_COUNT}"
        cv2.putText(view, status, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)

        # Draw corners if found
        if found and corners is not None:
            cv2.drawChessboardCorners(view, INNER_CORNERS, corners, found)

        cv2.imshow("Calibration capture", view)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break

        if key == ord('r'):
            saved = 0
            print("[INFO] Counter reset to 0 (files not deleted).")

        if key == 32:  # SPACE
            now = time.time()
            if (now - last_save_t) < MIN_SECONDS_BETWEEN_SAVES:
                continue

            if not found:
                print("[WARN] Checkerboard NOT detected. Move/tilt board and try again.")
                continue

            fn = next_filename(OUT_DIR, saved)
            cv2.imwrite(fn, frame)
            saved += 1
            last_save_t = now
            print(f"[OK] Saved {fn}")

            if saved >= TARGET_COUNT:
                print("[DONE] Captured enough images.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
