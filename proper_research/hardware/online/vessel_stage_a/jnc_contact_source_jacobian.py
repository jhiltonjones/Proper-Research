"""Source-space (magnet-pose) contact-model Jacobian, both analytic
(J_C^fast / J_C^accurate, via model.jacobian_output_actuation_tangent)
and finite-difference (J_C^FD, central difference directly on
solve_pose) -- the contact-model analogue of jnc_source_jacobian_psi.py
(no-contact), at phi=35deg, r=225mm, L, psi, against the real digitized
vessel wall.

Computing both in SOURCE space (not joint space, unlike
verify_contact_jacobian_implementation.py's Check 1) so they compare
directly against J_cam^contact, which is also a source-space
measurement -- matching Stage 1's established convention.

Usage: python jnc_contact_source_jacobian.py <psi_deg> <L_mm>
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose,
)

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = np.concatenate([beam_base_xyz, [3.14159265, 0.0, 0.0]])
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_zcorrected.json"
placeholder_magnet_pose6 = np.array(
    [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
     -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
)

PHI_DEG = 35.0
RADIUS_MM = 225.0
PSI_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
L_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
L_M = L_MM / 1000.0

phi = np.radians(PHI_DEG)
xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(PSI_DEG)]).as_matrix()
R0 = R_aligned @ R_psi
rotvec0 = Rot.from_matrix(R0).as_rotvec()

bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_M)
model = bundle.models["contact"]


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


def analytic_source_jacobian(mode):
    """d[tip,tangent]/d[world_xyz(3), world_rotvec(3), L] -- the SAME
    quantity _make_beam_jacobian_callback differentiates, called directly
    on the contact model in source space (no robot-kinematics composition)."""
    p7 = np.concatenate([xyz0, rotvec0, [L_M]])
    model.solve(p7, commit=True, reuse_cache=True)
    if hasattr(model, "jacobian_output_actuation_tangent"):
        J = np.asarray(model.jacobian_output_actuation_tangent(p7, solve_if_needed=False, mode=mode), dtype=float).reshape(6, 7)
    else:
        J = np.asarray(model.jacobian_tip_actuation_tangent(p7, solve_if_needed=False, mode=mode), dtype=float).reshape(3, 7)
    return J[:3]  # tip rows only, 3x7 (world_xyz, world_rotvec, L)


eps_trans = 0.002
eps_rot = np.radians(1.0)

J_FD = np.zeros((3, 6))
print(f"PHI_DEG={PHI_DEG} L_MM={L_MM} PSI_DEG={PSI_DEG}  (CONTACT model, source space)")
print(f"nominal xyz0: {xyz0}")
print(f"nominal rotvec0: {rotvec0}\n")

for i in range(6):
    eps = eps_trans if i < 3 else eps_rot
    xyz_p, rv_p = perturbed_pose(i, eps, +1)
    xyz_m, rv_m = perturbed_pose(i, eps, -1)
    tip_p = model_tip(xyz_p, rv_p)
    tip_m = model_tip(xyz_m, rv_m)
    col = (tip_p - tip_m) / (2 * eps)
    J_FD[:, i] = col
    label = ["x", "y", "z", "rx", "ry", "rz"][i]
    print(f"axis {label}: col_FD={np.round(col,5)}  norm={np.linalg.norm(col):.5f}")

print("\nJ_C^FD_source (3x6):")
print(np.round(J_FD, 5))

for mode in ("fast", "accurate"):
    J_an = analytic_source_jacobian(mode)[:, :6]  # drop the L column -> 3x6 (x,y,z,rx,ry,rz)
    print(f"\nJ_C^{mode}_source (3x6):")
    print(np.round(J_an, 5))
