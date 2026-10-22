import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_points_on_image(
    image_filename,
    tip_px,
    target_px,
    distance_mm=None,
    title="Tip vs Target (final image)",
):
    """
    Overlays the tip and target points on the given image.

    Parameters
    ----------
    image_filename : str
        Path to the image file (e.g. the last focused_image.jpg).
    tip_px : tuple (x, y)
        Final tip location in pixel coordinates.
    target_px : tuple (x, y)
        Target location in pixel coordinates (in the same image frame).
    distance_mm : float, optional
        Physical distance between target and tip in mm, for annotation.
    title : str
        Plot title.
    """
    img_bgr = cv2.imread(image_filename)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_filename}'")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tip_x, tip_y = tip_px
    tgt_x, tgt_y = target_px

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)

    # Plot target + tip
    plt.scatter([tgt_x], [tgt_y], s=80, marker='x', color='yellow', label='Target', zorder=3)
    plt.scatter([tip_x], [tip_y], s=80, marker='o', color='red',    label='Final tip', zorder=3)

    # Line between them
    plt.plot([tgt_x, tip_x], [tgt_y, tip_y], linestyle='--', linewidth=1, color='white', zorder=2)

    # Optional distance annotation (in mm)
    if distance_mm is not None:
        mid_x = 0.5 * (tgt_x + tip_x)
        mid_y = 0.5 * (tgt_y + tip_y)
        plt.text(
            mid_x, mid_y,
            f"{distance_mm:.2f} mm",
            color="cyan",
            fontsize=10,
            ha="center",
            va="bottom",
        )

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.show()
