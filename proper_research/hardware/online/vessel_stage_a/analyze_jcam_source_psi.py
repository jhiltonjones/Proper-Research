"""Stage-1 final symmetry-breaking test: compare model vs camera in-plane
source-pose Jacobians (x, y, rz columns only) at phi=35deg, L=30mm, for
psi=+30deg and psi=-30deg, and -- the key claim -- check that both model
and camera show the SAME qualitative rotation of the dominant mobility
direction when psi flips sign.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[4]
RESULTS_DIR = REPO_ROOT / "calibration_2026-10-05"
LABELS = ["x", "y", "rz"]


def model_J_source(psi_deg):
    script = HERE.parent / "jnc_source_jacobian_psi.py"
    out = subprocess.run(
        [sys.executable, str(script), str(psi_deg)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    lines = [l for l in out.splitlines() if l.startswith("axis ")]
    cols = {}
    for l in lines:
        axis = l.split()[1].rstrip(":")
        col_str = l.split("col=[")[1].split("]")[0]
        cols[axis] = np.array([float(x) for x in col_str.split()])
    return np.array([cols[a][:2] for a in LABELS]).T  # 2x3, XY rows only


def camera_J_source(psi_deg):
    path = RESULTS_DIR / f"jcam_source_phi35_L30_psi{psi_deg:+.0f}.json"
    with open(path) as f:
        r = json.load(f)
    cols = {c["axis"]: np.array(c["J_col"])[:2] for c in r["columns"]}
    return np.array([cols[a] for a in LABELS]).T, r  # 2x3


def per_axis_compare(J_model, J_cam):
    print(f"{'axis':>4s} {'theta_deg':>10s} {'gain(model/cam)':>16s} {'|model|':>10s} {'|cam|':>10s}")
    for i, label in enumerate(LABELS):
        a, b = J_model[:, i], J_cam[:, i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos_t = np.clip(np.dot(a, b) / (na * nb), -1, 1) if na > 1e-9 and nb > 1e-9 else np.nan
        theta = np.degrees(np.arccos(cos_t)) if not np.isnan(cos_t) else float("nan")
        gain = na / nb if nb > 1e-9 else float("nan")
        print(f"{label:>4s} {theta:10.2f} {gain:16.3f} {na:10.6f} {nb:10.6f}")


def dominant_dir(J):
    U, S, _ = np.linalg.svd(J)
    return U[:, 0], S


def angle_of(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def main():
    results = {}
    for psi in (30.0, -30.0):
        print(f"\n{'='*60}\npsi = {psi:+.0f} deg\n{'='*60}")
        J_model = model_J_source(psi)
        J_cam, raw = camera_J_source(psi)
        print("\nJ_NC_source (model, XY rows, x/y/rz cols):")
        print(np.round(J_model, 6))
        print("J_cam_source (camera, XY rows, x/y/rz cols):")
        print(np.round(J_cam, 6))
        print()
        per_axis_compare(J_model, J_cam)

        print("\n--- repeatability ---")
        for ret in raw["returns"]:
            print(f"{ret['axis']}: drift1={ret['drift1_mm']:.3f}mm  drift2={ret['drift2_mm']:.3f}mm")

        u_m, s_m = dominant_dir(J_model)
        u_c, s_c = dominant_dir(J_cam)
        cos_u = np.clip(abs(np.dot(u_m, u_c)), -1, 1)
        print(f"\nmodel dominant dir: {np.round(u_m, 4)}  S={np.round(s_m, 5)}")
        print(f"cam   dominant dir: {np.round(u_c, 4)}  S={np.round(s_c, 5)}")
        print(f"angle between dominant directions: {np.degrees(np.arccos(cos_u)):.2f}deg")

        results[psi] = {"u_model": u_m, "u_cam": u_c}

    print(f"\n{'='*60}\nkey claim: dominant-direction rotation when psi flips sign\n{'='*60}")
    rot_model = angle_of(results[-30.0]["u_model"]) - angle_of(results[30.0]["u_model"])
    rot_cam = angle_of(results[-30.0]["u_cam"]) - angle_of(results[30.0]["u_cam"])
    # normalize to [-180, 180)
    rot_model = ((rot_model + 180) % 360) - 180
    rot_cam = ((rot_cam + 180) % 360) - 180
    print(f"model rotation (psi=+30 -> psi=-30): {rot_model:.2f} deg")
    print(f"camera rotation (psi=+30 -> psi=-30): {rot_cam:.2f} deg")
    same_sign = (rot_model > 0) == (rot_cam > 0)
    print(f"same sign: {same_sign}  |difference|={abs(rot_model - rot_cam):.2f}deg")


if __name__ == "__main__":
    main()
