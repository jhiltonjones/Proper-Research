"""Top-down (world X-Y) schematic of the Stage-3 initial source-magnet
position: the beam base, the phi/radius arc the magnet sits on, the
dipole direction, and the hard magnet-to-beam-base exclusion-radius
constraint -- none of which the live camera overlay shows, since the
magnet itself sits well outside the camera's field of view.

Usage: read straight from a design_stage3_tracked_path.py design file
(default: the newest one), or override phi/radius/exclusion directly.

    python plot_stage3_arc_geometry.py
    python plot_stage3_arc_geometry.py --design-json plans/stage3_design/phi30_L30_left3.5mm_design.json
    python plot_stage3_arc_geometry.py --phi-deg 30 --radius-mm 225 --exclusion-floor-mm 210
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proper_research.rig_calibration import BEAM_BASE_XYZ_M

DESIGN_DIR = Path(__file__).resolve().parents[4] / "plans" / "stage3_design"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--design-json", type=Path, default=None,
                    help="design_stage3_tracked_path.py output to read phi/radius/exclusion/tip0 "
                         "from (default: newest *_design.json in plans/stage3_design)")
    p.add_argument("--phi-deg", type=float, default=None, help="override the design's phi")
    p.add_argument("--radius-mm", type=float, default=None, help="override the design's radius")
    p.add_argument("--exclusion-floor-mm", type=float, default=None,
                    help="override the design's hard exclusion-radius constraint")
    p.add_argument("--out-png", type=Path, default=None,
                    help="default: <design-json stem>_arc_geometry.png next to the design file")
    return p.parse_args()


def main():
    args = parse_args()

    if args.design_json is not None:
        design_path = args.design_json
    else:
        candidates = sorted(DESIGN_DIR.glob("*_design.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit(f"no *_design.json found in {DESIGN_DIR}")
        design_path = candidates[-1]
    with open(design_path) as f:
        design = json.load(f)

    phi_deg = args.phi_deg if args.phi_deg is not None else design["phi_deg"]
    radius_mm = args.radius_mm if args.radius_mm is not None else design["radius_mm"]
    exclusion_floor_mm = (
        args.exclusion_floor_mm if args.exclusion_floor_mm is not None else design["exclusion_floor_mm"]
    )
    magnet_xyz0 = np.asarray(design["magnet_xyz0"])
    tip0 = np.asarray(design["tip0"])
    beam_base_xyz = BEAM_BASE_XYZ_M.copy()

    print(f"design: {design_path}")
    print(f"phi_deg={phi_deg} radius_mm={radius_mm} exclusion_floor_mm={exclusion_floor_mm}")
    print(f"beam_base xy (R) = {beam_base_xyz[:2]}")
    print(f"magnet xy (R)    = {magnet_xyz0[:2]}")
    actual_dist_mm = float(np.linalg.norm(magnet_xyz0[:2] - beam_base_xyz[:2])) * 1000.0
    print(f"actual magnet-to-beam-base distance = {actual_dist_mm:.2f}mm "
          f"(exclusion floor {exclusion_floor_mm:.1f}mm -> "
          f"{actual_dist_mm - exclusion_floor_mm:.2f}mm of slack)")

    fig, ax = plt.subplots(figsize=(8, 7.5))
    base_xy = beam_base_xyz[:2]

    theta = np.linspace(0, 2 * np.pi, 200)
    # radius circle (the full set of reachable arc positions at this radius)
    ax.plot(base_xy[0] + radius_mm / 1000.0 * np.cos(theta),
            base_xy[1] + radius_mm / 1000.0 * np.sin(theta),
            "--", color="tab:blue", lw=1.2, label=f"r={radius_mm:.0f}mm arc")

    # hard exclusion-radius constraint (magnet must stay OUTSIDE this)
    ax.fill(base_xy[0] + exclusion_floor_mm / 1000.0 * np.cos(theta),
            base_xy[1] + exclusion_floor_mm / 1000.0 * np.sin(theta),
            color="tab:red", alpha=0.15)
    ax.plot(base_xy[0] + exclusion_floor_mm / 1000.0 * np.cos(theta),
            base_xy[1] + exclusion_floor_mm / 1000.0 * np.sin(theta),
            "-", color="tab:red", lw=1.5, label=f"hard exclusion floor ({exclusion_floor_mm:.0f}mm)")

    ax.plot(*base_xy, "ks", markersize=10, label="beam base")
    ax.plot(*magnet_xyz0[:2], "o", color="tab:purple", markersize=12,
            label=f"magnet xyz0 (phi={phi_deg:.0f}deg)")
    ax.plot(*tip0[:2], "*", color="gold", markeredgecolor="k", markersize=16, label="tip0 (model-predicted)")

    # dipole direction: aligned means pointing from the magnet back toward
    # the beam base (reference_orientation_matrix's convention) -- draw it
    # as an arrow from the magnet position.
    dipole_dir = (base_xy - magnet_xyz0[:2])
    dipole_dir = dipole_dir / np.linalg.norm(dipole_dir)
    arrow_len = 0.03
    ax.annotate("", xy=tuple(magnet_xyz0[:2] + arrow_len * dipole_dir), xytext=tuple(magnet_xyz0[:2]),
                arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2))
    ax.text(*(magnet_xyz0[:2] + 1.15 * arrow_len * dipole_dir), "dipole\n(aligned)",
            color="tab:purple", fontsize=9, ha="center")

    ax.annotate(f"phi={phi_deg:.0f}deg", xy=tuple(magnet_xyz0[:2]), xytext=(10, 10),
                textcoords="offset points", fontsize=10)
    ax.annotate(f"dist={actual_dist_mm:.1f}mm", xy=tuple(0.5 * (base_xy + magnet_xyz0[:2])),
                fontsize=9, color="tab:blue")

    ax.set_aspect("equal")
    ax.set_xlabel("world X (R), m")
    ax.set_ylabel("world Y (R), m")
    ax.set_title(f"Stage-3 initial position: phi={phi_deg:.0f}deg, r={radius_mm:.0f}mm\n"
                 f"exclusion floor={exclusion_floor_mm:.0f}mm (slack={actual_dist_mm - exclusion_floor_mm:.1f}mm)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_png = args.out_png if args.out_png is not None else design_path.with_name(design_path.stem.replace("_design", "_arc_geometry") + ".png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
