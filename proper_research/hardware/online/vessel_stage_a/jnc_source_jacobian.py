import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import (
    BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6,
)
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose,
)

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = beam_base_pose6()
LUMEN_FILE = CURRENT_LUMEN_FILE
placeholder_magnet_pose6 = REFERENCE_MAGNET_POSE6

L_M = 0.025
PHI_DEG = 40.0
RADIUS_MM = 225.0

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R0 = reference_orientation_matrix(xyz0, beam_base_xyz)
rotvec0 = Rot.from_matrix(R0).as_rotvec()

bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_M)
model = bundle.models["no_contact"]


def model_tip(xyz, rotvec):
    out = solve_pose(model, xyz, rotvec, L_M)
    return np.asarray(out["tip"])


def perturbed_pose(axis_idx, eps, sign):
    """axis_idx 0,1,2 = translation x,y,z (world frame, beam-base relative
    is the same since beam base is fixed); 3,4,5 = rotation about world
    x,y,z (left-multiplied spatial perturbation: R_pert = Rot(axis*eps) @ R0)."""
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


eps_trans = 0.002  # 2mm
eps_rot = np.radians(1.0)  # 1 deg

J_NC_source = np.zeros((3, 6))
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
    pred_disp_mm = np.linalg.norm(col) * eps * 1000
    print(f"axis {label}: eps={eps:.5f}  col={np.round(col,5)}  predicted_disp_at_this_eps_mm={pred_disp_mm:.3f}")

print("\nJ_NC_source (3x6):")
print(np.round(J_NC_source, 5))

U, S, Vt = np.linalg.svd(J_NC_source)
print(f"\nsingular values: {np.round(S,5)}")
print(f"dominant output direction: {np.round(U[:,0],4)}")
