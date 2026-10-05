"""Stage-2 final analysis: J_cam^contact vs J_C^fast vs J_C^FD vs J_NC,
plus J_cam^free vs J_cam^contact, at phi=35deg, r=225mm, L=40mm,
psi=+-30deg -- per the user's explicit four-way test design.

Also computes wall-normal/tangential mobility s_n(u)=|n_c^T J u|,
s_t(u)=|t_c^T J u| for free vs contact, using the local wall
normal/tangent at each psi's nominal contact point (already recorded by
verify_contact_jacobian_implementation.py's Check 2).
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
L_MM = 40.0

# recorded by verify_contact_jacobian_implementation.py (Check 2) at this L/phi:
WALL_NORMAL = {30.0: np.array([0.2138, 0.9769]), -30.0: np.array([0.5900, 0.8074])}
WALL_TANGENT = {30.0: np.array([-0.9716, 0.2366]), -30.0: np.array([-0.9143, 0.4051])}


def model_J_sources(psi_deg):
    script = HERE.parent / "jnc_contact_source_jacobian.py"
    out = subprocess.run(
        [sys.executable, str(script), str(psi_deg), str(L_MM)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    blocks = {}
    for tag in ("J_C^FD_source", "J_C^fast_source", "J_C^accurate_source"):
        idx = out.index(f"{tag} (3x6):")
        chunk = out[idx:].splitlines()[1:4]
        rows = [np.fromstring(l.strip(" []"), sep=" ") for l in chunk]
        blocks[tag] = np.vstack(rows)

    script_nc = HERE.parent / "jnc_source_jacobian_psi.py"
    out_nc = subprocess.run(
        [sys.executable, str(script_nc), str(psi_deg), str(L_MM)],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    idx = out_nc.index("J_NC_source (3x6):")
    chunk = out_nc[idx:].splitlines()[1:4]
    rows = [np.fromstring(l.strip(" []"), sep=" ") for l in chunk]
    blocks["J_NC_source"] = np.vstack(rows)
    return blocks


def camera_J(psi_deg, campaign):
    path = RESULTS_DIR / f"jcam_{campaign}_phi35_L{L_MM:.0f}_psi{psi_deg:+.0f}.json"
    with open(path) as f:
        r = json.load(f)
    cols = {c["axis"]: np.array(c["J_col"])[:2] for c in r["columns"]}
    return np.array([cols[a] for a in LABELS]).T, r


def per_axis_compare(label_a, J_a, label_b, J_b):
    print(f"{'axis':>4s} {'theta_deg':>10s} {f'gain({label_a}/{label_b})':>20s} {'|'+label_a+'|':>10s} {'|'+label_b+'|':>10s}")
    for i, lab in enumerate(LABELS):
        a, b = J_a[:, i], J_b[:, i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos_t = np.clip(np.dot(a, b) / (na * nb), -1, 1) if na > 1e-9 and nb > 1e-9 else np.nan
        theta = np.degrees(np.arccos(cos_t)) if not np.isnan(cos_t) else float("nan")
        gain = na / nb if nb > 1e-9 else float("nan")
        print(f"{lab:>4s} {theta:10.2f} {gain:20.3f} {na:10.6f} {nb:10.6f}")


def dominant_dir(J):
    U, S, _ = np.linalg.svd(J)
    return U[:, 0], S


def wall_mobility(J, n, t):
    """s_n/s_t per unit-input column (x,y,rz each treated as a unit input)."""
    s_n = np.array([abs(np.dot(n, J[:, i])) for i in range(3)])
    s_t = np.array([abs(np.dot(t, J[:, i])) for i in range(3)])
    return s_n, s_t


def main():
    for psi in (30.0, -30.0):
        print(f"\n{'='*72}\npsi = {psi:+.0f} deg  (phi=35deg, r=225mm, L={L_MM:.0f}mm)\n{'='*72}")
        models = model_J_sources(psi)
        J_C_fast = models["J_C^fast_source"][:2][:, [0, 1, 5]]
        J_C_FD = models["J_C^FD_source"][:2][:, [0, 1, 5]]
        J_NC = models["J_NC_source"][:2][:, [0, 1, 5]]
        J_cam_contact, raw_c = camera_J(psi, "contact")
        J_cam_free, raw_f = camera_J(psi, "source")

        print("\n--- step 1: does contact change measured camera mobility? J_cam^free vs J_cam^contact ---")
        per_axis_compare("free", J_cam_free, "contact", J_cam_contact)
        u_f, s_f = dominant_dir(J_cam_free)
        u_c, s_c = dominant_dir(J_cam_contact)
        cos_u = np.clip(abs(np.dot(u_f, u_c)), -1, 1)
        print(f"J_cam^free     S={np.round(s_f,5)}  dominant={np.round(u_f,4)}")
        print(f"J_cam^contact  S={np.round(s_c,5)}  dominant={np.round(u_c,4)}")
        print(f"angle between free/contact dominant directions: {np.degrees(np.arccos(cos_u)):.2f}deg")

        print("\n--- step 2a: J_cam^contact vs J_C^fast ---")
        per_axis_compare("cam", J_cam_contact, "C_fast", J_C_fast)
        print("\n--- step 2b: J_cam^contact vs J_C^FD ---")
        per_axis_compare("cam", J_cam_contact, "C_FD", J_C_FD)
        print("\n--- step 2c: J_cam^contact vs J_NC (is it closer to no-contact than to either contact Jacobian?) ---")
        per_axis_compare("cam", J_cam_contact, "NC", J_NC)

        for name, J in (("J_cam^contact", J_cam_contact), ("J_C^fast", J_C_fast), ("J_C^FD", J_C_FD), ("J_NC", J_NC)):
            u, s = dominant_dir(J)
            print(f"{name:14s} S={np.round(s,5)}  dominant={np.round(u,4)}")

        print("\n--- step 3: wall-normal/tangential mobility, free vs contact ---")
        n = WALL_NORMAL[psi] / np.linalg.norm(WALL_NORMAL[psi])
        t = WALL_TANGENT[psi] / np.linalg.norm(WALL_TANGENT[psi])
        s_n_free, s_t_free = wall_mobility(J_cam_free, n, t)
        s_n_contact, s_t_contact = wall_mobility(J_cam_contact, n, t)
        print(f"wall normal n={np.round(n,4)}  tangent t={np.round(t,4)}")
        print(f"{'axis':>4s} {'s_n_free':>10s} {'s_n_contact':>12s} {'ratio':>8s}   {'s_t_free':>10s} {'s_t_contact':>12s} {'ratio':>8s}")
        for i, lab in enumerate(LABELS):
            rn = s_n_contact[i] / s_n_free[i] if s_n_free[i] > 1e-9 else float("nan")
            rt = s_t_contact[i] / s_t_free[i] if s_t_free[i] > 1e-9 else float("nan")
            print(f"{lab:>4s} {s_n_free[i]:10.6f} {s_n_contact[i]:12.6f} {rn:8.3f}   {s_t_free[i]:10.6f} {s_t_contact[i]:12.6f} {rt:8.3f}")
        print("(hypothesis: s_n ratio << s_t ratio -- contact suppresses wall-normal mobility specifically, not a uniform scaling)")

        print("\n--- repeatability (contact campaign) ---")
        for ret in raw_c["returns"]:
            print(f"{ret['axis']}: drift1={ret['drift1_mm']:.3f}mm  drift2={ret['drift2_mm']:.3f}mm")


if __name__ == "__main__":
    main()
