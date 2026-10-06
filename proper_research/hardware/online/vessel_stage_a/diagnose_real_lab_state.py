"""Diagnose the contact model at a REAL, physically-achieved joint
configuration + insertion observed in the lab (the user reports the real
beam can follow a path closer to the left wall than the centreline under
contact -- this checks what the offline-configuration optimizer's own
model predicts there, and whether THAT state is well- or ill-conditioned,
to help explain why the optimizer struggles to find it).

Usage: python diagnose_real_lab_state.py <q1> <q2> <q3> <q4> <q5> <q6> <insertion_mm>
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import build_model_bundle, solve_pose
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter

CONTACT_BAND_M = 0.5e-3


def contact_profile(model, xyz, rotvec, insertion_m):
    out = solve_pose(model, xyz, rotvec, insertion_m)
    centerline = np.asarray(out["centerline"])
    r_beam = float(model.contact_cfg.params.r_beam)
    delta, Rloc, _, grad_delta, _ = model.lumen_query.closest_many_with_gradients(centerline, window=None)
    c = Rloc - delta - r_beam
    M = centerline.shape[0]
    i_min = int(np.argmin(c[1:])) + 1 if M > 1 else 0
    c_min = float(c[i_min])
    seg = np.diff(centerline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1] if arc[-1] > 1e-9 else 1.0
    s_frac = float(arc[i_min] / total_len)
    normal_c = grad_delta[i_min]
    if i_min == 0:
        tangent_c = centerline[1] - centerline[0]
    elif i_min == M - 1:
        tangent_c = centerline[-1] - centerline[-2]
    else:
        tangent_c = centerline[i_min + 1] - centerline[i_min - 1]
    tnorm = np.linalg.norm(tangent_c)
    tangent_c = tangent_c / tnorm if tnorm > 1e-12 else np.array([1.0, 0.0, 0.0])
    return c_min, s_frac, normal_c, tangent_c, centerline, out["tip"]


def main():
    q = np.array([float(x) for x in sys.argv[1:7]])
    insertion_mm = float(sys.argv[7])
    L_m = insertion_mm / 1000.0

    print(f"q (rad) = {q.tolist()}")
    print(f"q (deg) = {np.degrees(q).tolist()}")
    print(f"insertion = {insertion_mm}mm")

    fk = urik.forward_kinematics(q, rk.dh, rk.T_F_M)
    xyz = np.asarray(fk.T_R_target)[:3, 3]
    rotvec = Rot.from_matrix(np.asarray(fk.T_R_target)[:3, :3]).as_rotvec()
    print(f"\nmagnet xyz (R)    = {xyz}")
    print(f"magnet rotvec (R) = {rotvec}")

    beam_base_xyz = BEAM_BASE_XYZ_M.copy()
    dist_to_base_mm = float(np.linalg.norm(xyz - beam_base_xyz)) * 1000.0
    dist_to_base_xy_mm = float(np.linalg.norm((xyz - beam_base_xyz)[:2])) * 1000.0
    phi_deg = float(np.degrees(np.arctan2((beam_base_xyz[1] - xyz[1]), (beam_base_xyz[0] - xyz[0]))))
    print(f"distance to beam base (3D) = {dist_to_base_mm:.2f}mm, (XY only) = {dist_to_base_xy_mm:.2f}mm")
    print(f"implied phi (XY arc angle from beam base, world convention) = {phi_deg:.2f}deg")

    # Build the bundle with a SAFE placeholder insertion (build_controller's
    # default insertion_max_m is only 50mm, a separate limit from model.solve()
    # itself -- solve_pose below passes the REAL target insertion independently,
    # the bundle-construction placeholder here is irrelevant to that solve).
    bundle = build_model_bundle(CURRENT_LUMEN_FILE, beam_base_pose6(), REFERENCE_MAGNET_POSE6, 0.03)
    model_c = bundle.models["contact"]

    c_min, s_frac, normal_c, tangent_c, centerline, tip = contact_profile(model_c, xyz, rotvec, L_m)
    print(f"\n--- contact profile at this real lab state ---")
    print(f"tip = {tip}")
    print(f"c_min = {c_min*1000:.3f}mm  (in_contact={c_min <= CONTACT_BAND_M})")
    print(f"s_c (arc-length fraction along beam) = {s_frac:.3f}")
    print(f"normal_c = {normal_c}")
    print(f"tangent_c = {tangent_c}")
    print(f"centerline shape = {centerline.shape}, first point = {centerline[0]}, last point = {centerline[-1]}")

    # Jacobian conditioning check at this exact state (same methodology as
    # verify_contact_jacobian_implementation.py's Check 1).
    controller_pack = {"robot_dh": rk.dh, "T_F_M": rk.T_F_M}
    adapter_c = build_diagnostic_adapter(beam_model=model_c, controller_pack=controller_pack, jacobian_mode="accurate")
    state0 = np.concatenate([q, [L_m]])
    chain = adapter_c.validate_chain_rule(state0, joint_step_rad=1.5e-2, insertion_step_m=5.0e-3)
    print(f"\n--- Jacobian conditioning at this real lab state ---")
    print(f"relative_frobenius_error (analytic vs FD) = {chain['relative_frobenius_error']:.4f}")
    print(f"max_abs_element_error = {chain['maximum_absolute_element_error']:.4e}")
    J = chain["analytical"][:3]
    U, S, _ = np.linalg.svd(J)
    print(f"full 3x7 Jacobian singular values: {np.round(S, 5)}")
    print(f"condition number (sigma1/sigma_min_nonzero) = {S[0]/max(S[-1], 1e-12):.3e}")


if __name__ == "__main__":
    main()
