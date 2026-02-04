import os
import cv2
import numpy as np

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

MM_TO_PT = 72.0 / 25.4  # 1 mm in PDF points

def mm(mm_val: float) -> float:
    return mm_val * MM_TO_PT

def make_aruco_marker_png(marker_id: int, dictionary, px: int = 800) -> str:
    """
    Generates a single ArUco marker image (PNG) and returns its filepath.
    """
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, px)  # grayscale uint8
    # Ensure it's a 3-channel image for compatibility with some PDF pipelines
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    out_dir = "aruco_print"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"aruco_{marker_id}.png")
    cv2.imwrite(path, img_bgr)
    return path

def create_printable_pdf(
    marker_ids=(0, 1, 2, 3),
    marker_size_mm=20.0,     # <-- set your exact physical marker width here
    padding_mm=10.0,         # white space around each marker on paper
    marker_px=800,
    pdf_path="aruco_markers_to_scale.pdf"
):
    # ArUco dictionary selection
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Create marker PNGs
    marker_paths = [make_aruco_marker_png(mid, dictionary, px=marker_px) for mid in marker_ids]

    # PDF setup (A4)
    page_w, page_h = A4
    c = canvas.Canvas(pdf_path, pagesize=A4)

    # Layout: 2x2 grid
    # Compute positions in mm then convert to points
    msize_pt = mm(marker_size_mm)
    pad_pt = mm(padding_mm)

    # Place markers with generous spacing
    # Top-left origin in reportlab is bottom-left, so we compute accordingly.
    x0 = mm(20)  # left margin
    y0 = page_h - mm(30) - msize_pt  # top margin

    dx = msize_pt + pad_pt
    dy = msize_pt + pad_pt

    positions = [
        (x0,         y0),          # TL
        (x0 + dx,    y0),          # TR
        (x0,         y0 - dy),     # BL
        (x0 + dx,    y0 - dy),     # BR
    ]

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(mm(20), page_h - mm(15), f"ArUco markers at {marker_size_mm:.1f} mm width (print at 100%)")

    # Draw markers + labels
    c.setFont("Helvetica", 12)
    for (mid, path), (x, y) in zip(zip(marker_ids, marker_paths), positions):
        c.drawImage(ImageReader(path), x, y, width=msize_pt, height=msize_pt, mask='auto')
        c.drawString(x, y - mm(6), f"ID {mid}")

    # Add a 50 mm scale bar for verification
    bar_mm = 50.0
    bar_x = mm(20)
    bar_y = mm(20)
    c.setLineWidth(2)
    c.line(bar_x, bar_y, bar_x + mm(bar_mm), bar_y)
    c.setFont("Helvetica", 10)
    c.drawString(bar_x, bar_y + mm(3), f"{bar_mm:.0f} mm scale bar (measure with a ruler)")

    # Add a note about print settings
    c.setFont("Helvetica", 9)
    c.drawString(mm(20), mm(12), "Print settings: 'Actual size' / 100% scale. Disable 'Fit to page'.")

    c.showPage()
    c.save()
    print(f"[OK] Wrote {pdf_path}")

if __name__ == "__main__":
    create_printable_pdf(
        marker_ids=(0, 1, 2, 3),
        marker_size_mm=20.0,      # <-- change this to 15.0 / 25.0 etc
        padding_mm=20.0,
        marker_px=1000,
        pdf_path="aruco_4markers_20mm.pdf"
    )