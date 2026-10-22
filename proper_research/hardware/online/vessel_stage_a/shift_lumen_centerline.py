#!/usr/bin/env python3
"""Rigidly shift a detect_blue.py-format vessel lumen file's TRACKING
CENTRELINE sideways, toward one wall, by a fixed distance -- radii
(vessel diameter) are left unchanged, so this moves where the plan aims
within the same tube, not the tube itself.

Direction convention matches this project's own established left=green/
right=red overlay (`draw_centerline_radius_overlay`, reused live by
`live_vessel_alignment_overlay.py`): at each centreline point, take the
local in-plane tangent (finite difference along the centreline, in the
beam's own planar frame B), rotate it +90 degrees to get the outward
normal, then:

    right = centreline - shift * normal
    left  = centreline + shift * normal

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.shift_lumen_centerline \\
        --lumen-file vessel_lumen_robot_frame.json \\
        --shift-mm 1.0 --direction right \\
        --z-raise-mm 0.0 \\
        --out vessel_lumen_robot_frame_right1mm.json

This changes the OFFLINE plan's own target path (and, since the same
lumen file is also the contact model's wall geometry, how close the
plan's own Layer 1 solve is allowed to get to the shifted centreline) --
you still need to rebuild the plan (`build_vessel_plan.py`) and the
Jacobian schedules afterward; see HOWTO_VESSEL_PLANNING.md section 1c and
HOWTO_CLOSED_LOOP_MPC.md section 2b. If you only want the LIVE controller's
tracking target shifted, for a single run, without touching the offline
plan at all, use `run_mpc_delay_aware_vessel.py --right-shift-mm` instead
-- see HOWTO_CLOSED_LOOP_MPC.md section 2b for that much cheaper option and
when each is the right tool.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def shift_lumen(C_R: np.ndarray, shift_m: float, direction: str, z_raise_mm: float):
    from proper_research.hardware.online.vessel_stage_a import common
    from proper_research.hardware.online.state_stream import NewFrameTipMapper

    common.Z_RAISE_M = z_raise_mm / 1000.0
    scfg = common._raised_stream_stream_config()
    mapper = NewFrameTipMapper(scfg)
    T_R_B = mapper.T_R_B
    T_B_R = T_R_B.inverse()

    C_B = T_B_R.apply_points(C_R)
    tangent = np.gradient(C_B[:, :2], axis=0)
    tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.clip(tnorm, 1e-9, None)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

    sign = -1.0 if direction == "right" else 1.0
    shifted_B = C_B.copy()
    shifted_B[:, :2] = C_B[:, :2] + sign * shift_m * normal
    return T_R_B.apply_points(shifted_B)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--shift-mm", type=float, required=True)
    p.add_argument("--direction", choices=["left", "right"], default="right")
    p.add_argument("--z-raise-mm", type=float, default=0.0,
                    help="must match the z-raise this lumen file's own pivot/frame was built "
                         "against -- see HOWTO_VESSEL_PLANNING.md's note on --z-raise-mm")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.lumen_file) as f:
        d = json.load(f)
    if d.get("frame") != "R":
        raise ValueError(f"expected a robot-frame ('R') lumen file, got frame={d.get('frame')!r}")

    C = np.asarray(d["lumen_C_m"], dtype=float)
    R = np.asarray(d["lumen_R_m"], dtype=float)
    C_shifted = shift_lumen(C, args.shift_mm / 1000.0, args.direction, args.z_raise_mm)
    max_disp_mm = float(np.linalg.norm(C_shifted - C, axis=1).max()) * 1000.0

    d["lumen_C_m"] = C_shifted.tolist()
    d.setdefault("provenance", {})["centreline_shift_note"] = (
        f"{args.shift_mm}mm toward the {args.direction} wall, derived from "
        f"{args.lumen_file} (radii unchanged, z_raise_mm={args.z_raise_mm})"
    )

    with open(args.out, "w") as f:
        json.dump(d, f, indent=2)

    print(f"[shift] {args.shift_mm}mm nominal toward {args.direction}, "
          f"max actual displacement {max_disp_mm:.3f}mm ({len(C)} points)")
    print(f"[shift] saved -> {args.out}")


if __name__ == "__main__":
    main()
