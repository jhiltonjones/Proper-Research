import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose,
)

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = np.concatenate([beam_base_xyz, [3.14159265, 0.0, 0.0]])
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_left1p5mm_zcorrected.json"
placeholder_magnet_pose6 = np.array(
    [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
     -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
)

PHI_DEG = 35.0
RADIUS_MM = 225.0
L_M = 0.040
PSI_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
R0 = R_aligned @ R_psi  # right-multiply: pure rotation about the magnet's own local Z (established convention)
rotvec0 = Rot.from_matrix(R0).as_rotvec()

bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_M)
model = bundle.models["no_contact"]


def model_tip(xyz, rotvec):
    out = solve_pose(model, xyz, rotvec, L_M)
    return np.asarray(out["tip"])


def perturbed_pose(axis_idx, eps, sign):
    if axis_idx < 3:
        d = np.zeros(3)
        d[axis_idx] = sign * eps
        return xyz0 + d, rotvec0
    else:
        ax = axis_idx - 3
        d = np.zeros(3)
        d[ax] = sign * eps
        R_pert = Rot.from_rotvec(d).as_matrix() @ R0
        return xyz0, Rot.from_matrix(R_pert).as_rotvec()


eps_trans = 0.002
eps_rot = np.radians(1.0)

J_NC_source = np.zeros((3, 6))
print(f"PHI_DEG={PHI_DEG} L_M={L_M} PSI_DEG={PSI_DEG}")
print(f"nominal xyz0: {xyz0}")
print(f"nominal rotvec0: {rotvec0}\n")

for i in range(6):
    eps = eps_trans if i < 3 else eps_rot
    xyz_p, rv_p = perturbed_pose(i, eps, +1)
    xyz_m, rv_m = perturbed_pose(i, eps, -1)
    tip_p = model_tip(xyz_p, rv_p)
    tip_m = model_tip(xyz_m, rv_m)
    col = (tip_p - tip_m) / (2 * eps)
    J_NC_source[:, i] = col
    label = ["x", "y", "z", "rx", "ry", "rz"][i]
    print(f"axis {label}: col={np.round(col,5)}  norm={np.linalg.norm(col):.5f}")

print("\nJ_NC_source (3x6):")
print(np.round(J_NC_source, 5))
U, S, Vt = np.linalg.svd(J_NC_source)
print(f"\nsingular values: {np.round(S,5)}")
print(f"dominant output direction: {np.round(U[:,0],4)}")
