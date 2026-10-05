"""Stage-3 design step: compute the tracked path and initial source-magnet
pose for the offline inverse-configuration planner, BEFORE committing to
running that optimizer.

Per the user's explicit spec (2026-10-05):
  - starting position: the established phi=40deg arc (r=225mm, psi=0,
    matching the Stage-1 convention), initial insertion L0=30mm.
  - tracked path: starts at the model-predicted initial tip position for
    that starting pose, ends 20mm short of the far/distal end of the real
    digitized vessel centreline.
  - the tracked path should sit 2mm toward the green/left wall (the same
    left=green/right=red, +90deg-tangent-normal convention
    shift_lumen_centerline.py already established).

This script only COMPUTES and SAVES the design (JSON) and renders a
single-frame camera overlay (reusing live_vessel_alignment_overlay.py's
calibration/projection/drawing helpers) so it can be reviewed before any
offline_inverse_configuration / build_vessel_plan run.
"""
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

PHI_DEG = 40.0
RADIUS_MM = 225.0
L0_MM = 30.0
LEFT_SHIFT_MM = 1.0
END_MARGIN_MM = 20.0

OUT_DIR = Path(__file__).resolve().parents[4] / "plans" / "stage3_design"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def arclength(points: np.ndarray) -> np.ndarray:
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def main():
    beam_base_xyz = BEAM_BASE_XYZ_M.copy()
    phi = np.radians(PHI_DEG)
    xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    R0 = reference_orientation_matrix(xyz0, beam_base_xyz)
    rotvec0 = Rot.from_matrix(R0).as_rotvec()
    L0_m = L0_MM / 1000.0

    print(f"PHI_DEG={PHI_DEG} RADIUS_MM={RADIUS_MM} L0_MM={L0_MM}")
    print(f"magnet xyz0 (R)    = {xyz0}")
    print(f"magnet rotvec0 (R) = {rotvec0}")

    bundle = build_model_bundle(CURRENT_LUMEN_FILE, beam_base_pose6(), REFERENCE_MAGNET_POSE6, L0_m)
    plant_model = bundle.models["contact"]  # plant_contact=True is this project's default
    out = solve_pose(plant_model, xyz0, rotvec0, L0_m)
    tip0 = np.asarray(out["tip"])
    print(f"model-predicted initial tip (contact-aware plant) = {tip0}")

    lumen_C, lumen_R, provenance = load_vessel_lumen_robot_frame(CURRENT_LUMEN_FILE)
    print(f"lumen: {lumen_C.shape[0]} samples, radius {1e3*lumen_R.min():.2f}-{1e3*lumen_R.max():.2f}mm")

    shifted_C = shift_lumen(lumen_C, LEFT_SHIFT_MM / 1000.0, "left", 0.0)
    arc = arclength(shifted_C)
    total_len = arc[-1]
    target_cut_len = total_len - END_MARGIN_MM / 1000.0
    end_cut_idx = int(np.searchsorted(arc, target_cut_len))
    # Interpolate the exact cut point between samples [end_cut_idx-1, end_cut_idx]
    # rather than snapping to the nearest existing sample (the ~1.9mm sample
    # spacing here would otherwise leave an 18.3mm margin instead of 20.0mm).
    i0, i1 = end_cut_idx - 1, end_cut_idx
    frac = (target_cut_len - arc[i0]) / (arc[i1] - arc[i0])
    end_cut_point = shifted_C[i0] + frac * (shifted_C[i1] - shifted_C[i0])
    shifted_C_trimmed_end = np.vstack([shifted_C[:end_cut_idx], end_cut_point[None, :]])
    print(f"shifted centreline total length = {total_len*1e3:.1f}mm; "
          f"end cut interpolated at arc-length {target_cut_len*1e3:.2f}mm "
          f"(exactly {END_MARGIN_MM}mm from the far end, between samples {i0}/{i1})")

    dists_to_tip = np.linalg.norm(shifted_C - tip0[None, :], axis=1)
    entry_idx = int(np.argmin(dists_to_tip))
    print(f"shifted centreline point nearest the initial tip: index {entry_idx}, "
          f"distance {dists_to_tip[entry_idx]*1e3:.2f}mm, at arc-length {arc[entry_idx]*1e3:.1f}mm")

    if entry_idx >= end_cut_idx:
        raise RuntimeError(
            f"entry point (idx {entry_idx}) is at or past the end-margin cut (idx {end_cut_idx}) -- "
            f"the requested 20mm end margin leaves no path to track from this starting tip."
        )

    tracked_path = np.vstack([tip0[None, :], shifted_C_trimmed_end[entry_idx:]])
    print(f"tracked_path: {tracked_path.shape[0]} points, "
          f"from tip0 to {target_cut_len*1e3:.2f}mm along the shifted centreline "
          f"(exactly {END_MARGIN_MM}mm short of the {total_len*1e3:.1f}mm far end)")

    design = {
        "phi_deg": PHI_DEG, "radius_mm": RADIUS_MM, "L0_mm": L0_MM,
        "left_shift_mm": LEFT_SHIFT_MM, "end_margin_mm": END_MARGIN_MM,
        "magnet_xyz0": xyz0.tolist(), "magnet_rotvec0": rotvec0.tolist(),
        "tip0": tip0.tolist(),
        "lumen_file": CURRENT_LUMEN_FILE,
        "shifted_centreline_entry_idx": entry_idx, "shifted_centreline_end_idx": end_cut_idx,
        "tracked_path_R": tracked_path.tolist(),
        "full_shifted_centreline_R": shifted_C.tolist(),
        "lumen_R_m": lumen_R.tolist(),
    }
    out_path = OUT_DIR / f"phi40_L30_left{LEFT_SHIFT_MM:.0f}mm_design.json"
    with open(out_path, "w") as f:
        json.dump(design, f, indent=2, default=float)
    print(f"\nsaved design to {out_path}")
    return design


if __name__ == "__main__":
    main()
