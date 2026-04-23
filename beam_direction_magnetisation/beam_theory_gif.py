from pathlib import Path
import re
import imageio.v2 as imageio

INPUT_DIR = Path("/home/jack/Proper-Research/results_sweep_beam_theory")
OUTPUT_GIF = INPUT_DIR / "overlay_sweep.gif"

FRAME_DURATION = 0.20   # seconds per frame
LOOP = 0                # 0 = infinite loop


def extract_j_value(path: Path) -> int:
    m = re.search(r"overlay_j_(-?\d+)\.png$", path.name)
    if not m:
        return 10**9
    return int(m.group(1))


def main():
    png_files = sorted(
        INPUT_DIR.glob("overlay_j_*.png"),
        key=extract_j_value
    )

    if not png_files:
        raise FileNotFoundError(f"No overlay_j_*.png files found in {INPUT_DIR}")

    frames = [imageio.imread(p) for p in png_files]

    imageio.mimsave(
        OUTPUT_GIF,
        frames,
        duration=FRAME_DURATION,
        loop=LOOP,
    )

    print(f"Saved GIF to: {OUTPUT_GIF}")
    print(f"Frames: {len(frames)}")


if __name__ == "__main__":
    main()