"""Stage-3 design step: compute the tracked path and initial source-magnet
pose for the offline inverse-configuration planner, BEFORE committing to
running that optimizer.

Every design choice is a CLI parameter -- change one flag, re-run, re-run
render_stage3_design_overlay.py (it auto-picks the newest design file),
no code edits needed. Defaults match the current Stage-3 design (phi=40deg
arc, r=225mm, psi=0, L0=30mm, 3.5mm toward the left/green wall, ending
20mm short of the vessel's far end).

Usage
-----
    python design_stage3_tracked_path.py \\
        --phi-deg 40 --radius-mm 225 --l0-mm 30 \\
        --left-shift-mm 3.5 --end-margin-mm 20

    # change ONE thing, e.g. just the wall offset:
    python design_stage3_tracked_path.py --left-shift-mm 5.0

Conventions reused, not reimplemented:
  - starting magnet pose: same phi/radius arc convention as Stage 1
    (sweep_free_space_arc_dipole.reference_orientation_matrix).
  - left/right wall shift: shift_lumen_centerline.shift_lumen's own
    left=green/right=red, +90deg-tangent-normal convention.
  - plant model: bundle.models["contact"] (this project's default
    plant_contact=True), evaluated against rig_calibration.CURRENT_LUMEN_FILE
    (override with --lumen-file if you need a different digitized vessel).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import (
    BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose,
)
from proper_research.hardware.online.vessel_stage_a.shift_lumen_centerline import shift_lumen
from proper_research.vision.detect_blue import load_vessel_lumen_robot_frame

OUT_DIR = Path(__file__).resolve().parents[4] / "plans" / "stage3_design"


def arclength(points: np.ndarray) -> np.ndarray:
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phi-deg", type=float, default=40.0,
                    help="source-magnet arc angle, degrees (default: 40.0)")
    p.add_argument("--radius-mm", type=float, default=225.0,
                    help="source-magnet arc radius from the beam base, mm (default: 225.0)")
    p.add_argument("--psi-deg", type=float, default=0.0,
                    help="dipole in-plane rotation on top of the aligned orientation, degrees (default: 0.0)")
    p.add_argument("--l0-mm", type=float, default=30.0,
                    help="initial insertion length, mm (default: 30.0)")
    p.add_argument("--left-shift-mm", type=float, default=3.5,
                    help="how far toward the left/green wall the tracked path sits, mm "
                         "(negative = toward the right/red wall instead; default: 3.5)")
    p.add_argument("--end-margin-mm", type=float, default=20.0,
                    help="how far short of the vessel centreline's far/distal end the "
                         "tracked path stops, mm (default: 20.0)")
    p.add_argument("--lumen-file", default=CURRENT_LUMEN_FILE,
                    help=f"digitized vessel lumen file (default: rig_calibration.CURRENT_LUMEN_FILE "
                         f"= {CURRENT_LUMEN_FILE})")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help=f"where to save the design JSON (default: {OUT_DIR})")
    p.add_argument("--tag", default=None,
                    help="override the auto-generated output filename tag "
                         "(default: phi{phi}_L{l0}_left{shift}mm, right{+}/left{-} "
                         "automatically for a negative shift)")
    return p.parse_args()


def design_tracked_path(
    *, phi_deg, radius_mm, psi_deg, l0_mm, left_shift_mm, end_margin_mm, lumen_file,
):
    """Pure computation, no file I/O -- returns the design dict. Split out
    from main() so another script can call this directly (e.g. a sweep over
    several candidate shifts) without going through argparse/stdout."""
    beam_base_xyz = BEAM_BASE_XYZ_M.copy()
    phi = np.radians(phi_deg)
    xyz0 = beam_base_xyz + (radius_mm / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
    R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(psi_deg)]).as_matrix()
    R0 = R_aligned @ R_psi
    rotvec0 = Rot.from_matrix(R0).as_rotvec()
    L0_m = l0_mm / 1000.0

    print(f"phi_deg={phi_deg} radius_mm={radius_mm} psi_deg={psi_deg} l0_mm={l0_mm}")
    print(f"magnet xyz0 (R)    = {xyz0}")
    print(f"magnet rotvec0 (R) = {rotvec0}")

    bundle = build_model_bundle(lumen_file, beam_base_pose6(), REFERENCE_MAGNET_POSE6, L0_m)
    plant_model = bundle.models["contact"]  # plant_contact=True is this project's default
    out = solve_pose(plant_model, xyz0, rotvec0, L0_m)
    tip0 = np.asarray(out["tip"])
    print(f"model-predicted initial tip (contact-aware plant) = {tip0}")

    lumen_C, lumen_R, provenance = load_vessel_lumen_robot_frame(lumen_file)
    print(f"lumen: {lumen_C.shape[0]} samples, radius {1e3*lumen_R.min():.2f}-{1e3*lumen_R.max():.2f}mm")

    direction = "left" if left_shift_mm >= 0 else "right"
    shifted_C = shift_lumen(lumen_C, abs(left_shift_mm) / 1000.0, direction, 0.0)
    arc = arclength(shifted_C)
    total_len = arc[-1]
    target_cut_len = total_len - end_margin_mm / 1000.0
    if not (0.0 < target_cut_len < total_len):
        raise ValueError(
            f"end_margin_mm={end_margin_mm} leaves no valid path on a "
            f"{total_len*1e3:.1f}mm-long centreline"
        )
    end_cut_idx = int(np.searchsorted(arc, target_cut_len))
    # Interpolate the exact cut point between samples [end_cut_idx-1, end_cut_idx]
    # rather than snapping to the nearest existing sample (coarse ~2mm sample
    # spacing would otherwise leave the margin off by up to one sample).
    i0, i1 = end_cut_idx - 1, end_cut_idx
    frac = (target_cut_len - arc[i0]) / (arc[i1] - arc[i0])
    end_cut_point = shifted_C[i0] + frac * (shifted_C[i1] - shifted_C[i0])
    shifted_C_trimmed_end = np.vstack([shifted_C[:end_cut_idx], end_cut_point[None, :]])
    print(f"shifted centreline total length = {total_len*1e3:.1f}mm; "
          f"end cut interpolated at arc-length {target_cut_len*1e3:.2f}mm "
          f"(exactly {end_margin_mm}mm from the far end, between samples {i0}/{i1})")

    dists_to_tip = np.linalg.norm(shifted_C - tip0[None, :], axis=1)
    entry_idx = int(np.argmin(dists_to_tip))
    print(f"shifted centreline point nearest the initial tip: index {entry_idx}, "
          f"distance {dists_to_tip[entry_idx]*1e3:.2f}mm, at arc-length {arc[entry_idx]*1e3:.1f}mm")

    if entry_idx >= end_cut_idx:
        raise RuntimeError(
            f"entry point (idx {entry_idx}) is at or past the end-margin cut (idx {end_cut_idx}) -- "
            f"end_margin_mm={end_margin_mm} leaves no path to track from this starting tip."
        )

    tracked_path = np.vstack([tip0[None, :], shifted_C_trimmed_end[entry_idx:]])
    print(f"tracked_path: {tracked_path.shape[0]} points, "
          f"from tip0 to {target_cut_len*1e3:.2f}mm along the shifted centreline "
          f"(exactly {end_margin_mm}mm short of the {total_len*1e3:.1f}mm far end)")

    return {
        "phi_deg": phi_deg, "radius_mm": radius_mm, "psi_deg": psi_deg, "L0_mm": l0_mm,
        "left_shift_mm": left_shift_mm, "end_margin_mm": end_margin_mm,
        "magnet_xyz0": xyz0.tolist(), "magnet_rotvec0": rotvec0.tolist(),
        "tip0": tip0.tolist(),
        "lumen_file": lumen_file,
        "shifted_centreline_entry_idx": entry_idx, "shifted_centreline_end_idx": end_cut_idx,
        "tracked_path_R": tracked_path.tolist(),
        "full_shifted_centreline_R": shifted_C.tolist(),
        "lumen_R_m": lumen_R.tolist(),
    }


def main():
    args = parse_args()
    design = design_tracked_path(
        phi_deg=args.phi_deg, radius_mm=args.radius_mm, psi_deg=args.psi_deg, l0_mm=args.l0_mm,
        left_shift_mm=args.left_shift_mm, end_margin_mm=args.end_margin_mm,
        lumen_file=args.lumen_file,
    )

    if args.tag:
        tag = args.tag
    else:
        side = "left" if args.left_shift_mm >= 0 else "right"
        tag = f"phi{args.phi_deg:.0f}_L{args.l0_mm:.0f}_{side}{abs(args.left_shift_mm):g}mm"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{tag}_design.json"
    with open(out_path, "w") as f:
        json.dump(design, f, indent=2, default=float)
    print(f"\nsaved design to {out_path}")
    return design


if __name__ == "__main__":
    main()
