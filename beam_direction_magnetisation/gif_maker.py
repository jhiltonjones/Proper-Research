from pathlib import Path
import re
import imageio.v2 as imageio


INPUT_DIR = Path("/home/jack/Proper-Research/debug_outputs_opti_mid_3step")
OUTPUT_GIF = INPUT_DIR / "reconstruction_overlay.gif"

# seconds per frame
FRAME_DURATION = 0.15

# 0 = infinite loop
LOOP = 0


def numeric_key(path: Path):
    m = re.search(r"(\d+)(?=\.png$)", path.name)
    return int(m.group(1)) if m else -1


def main():
    png_files = sorted(
        INPUT_DIR.glob("reconstruction_overlay_step_*.png"),
        key=numeric_key
    )

    if not png_files:
        raise FileNotFoundError(f"No matching PNG files found in {INPUT_DIR}")

    frames = [imageio.imread(png) for png in png_files]

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