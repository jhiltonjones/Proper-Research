from pathlib import Path
from PIL import Image
import imageio.v2 as imageio


INPUT_DIR = Path("/Users/jackhilton-jones/Proper-Research/mpc_run_centreline_track_day21stepstepwobc10nodes")
OUTPUT_GIF = INPUT_DIR / "reference_debug.gif"

FRAME_DURATION = 0.15
LOOP = 0

START_FRAME = 0
END_FRAME = None


def main():
    frames = []
    frame_idx = START_FRAME
    target_size = None

    while True:
        if END_FRAME is not None and frame_idx > END_FRAME:
            break

        png = INPUT_DIR / f"step_{frame_idx:04d}_reference_debug.png"

        if not png.exists():
            if END_FRAME is None:
                break
            print(f"Skipping missing file: {png}")
            frame_idx += 1
            continue

        img = Image.open(png).convert("RGB")

        if target_size is None:
            target_size = img.size
            print(f"Using target frame size: {target_size}")

        if img.size != target_size:
            print(f"Resizing {png.name} from {img.size} to {target_size}")
            img = img.resize(target_size)

        frames.append(img)
        frame_idx += 1

    if not frames:
        raise FileNotFoundError(f"No matching PNG files found in {INPUT_DIR}")

    imageio.mimsave(
        OUTPUT_GIF,
        frames,
        duration=FRAME_DURATION,
        loop=LOOP,
    )

    print(f"Saved GIF to: {OUTPUT_GIF}")
    print(f"Number of frames: {len(frames)}")


if __name__ == "__main__":
    main()