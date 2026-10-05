"""Stage-2 pre-flight: three software-only checks on the contact-aware
model/Jacobian, before spending any robot time, per the user's explicit
"verify the contact Jacobian implementation first, but validate the
contact model physically with the experiment" plan (2026-10-05).

Representative states: phi=35deg, r=225mm, L=40mm, psi=+30deg and
psi=-30deg -- the exact matched geometry Stage-1 already validated in
free space, now evaluated against the real digitized vessel wall
(vessel_lumen_robot_frame_zcorrected.json; the as-mapped file needed a
-23.060mm Z-correction, the same stale-reference-height issue already
documented in vessel_lumen_robot_frame_left1p5mm_zcorrected.json).

Check 1 -- implementation: does J_C^fast (the adapter's analytic
"continuous_output_jacobian", built with jacobian_mode="accurate", i.e. the
same object beam_jacobian_providers.from_model_bundle(contact=True,
jacobian_mode="accurate") would hand the controller) match a hand-rolled
finite-difference Jacobian of the SAME contact-model adapter
(JointSpaceBeamMPCAdapter.finite_difference_output_jacobian, which always
perturbs the true nonlinear forward solve regardless of jacobian_mode)?
Uses the adapter's own validate_chain_rule -- this is the contact-model
analogue of the free-space J_fast-vs-J_FD check already done for J_NC.

Check 2 -- contact sanity: at the nominal state, record c_min (wall
clearance, LumenQuery convention: positive = free lumen, 0 = beam
surface at the wall, negative = penetration), s_c (arc-length fraction
of the closest-approach node), and the local wall normal/tangent. Then
perturb x, y, rz by the SAME eps used in Stage 1's live campaigns and
confirm +eps and -eps stay on the SAME side of the contact/free
boundary -- a column whose two probe points straddle that boundary is
measuring a mode switch, not a local contacted-regime Jacobian.

Check 3 -- qualitative mechanical effect: compare J_C and J_NC (both via
from_model_bundle) at the same contacted state. Not required to match
any hoped-for result -- just confirm contact is doing something
mechanically meaningful (a real singular-value/direction change) rather
than nothing or something pathological (NaN, zero, identical to J_NC).

If all three pass, the next step is collecting J_cam^contact on
hardware -- NOT further model tuning (tuning before seeing the hardware
result risks building the conclusion into the model, per the user's
explicit instruction).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import (
    build_model_bundle, reference_orientation_matrix, solve_pose, build_seed_pool, robust_ik_for_pose,
)
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter, from_model_bundle

PHI_DEG = 35.0
RADIUS_MM = 225.0
L_MM = 40.0
L_M = L_MM / 1000.0
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_zcorrected.json"
CONTACT_BAND_M = 0.5e-3  # matches this project's own --mpc-wall-avoidance-margin-mm default

beam_base_xyz = BEAM_BASE_XYZ_M.copy()
beam_base_xyz6 = np.concatenate([beam_base_xyz, [3.14159265, 0.0, 0.0]])
placeholder_magnet_pose6 = np.array(
    [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
     -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
)
controller_pack = {"robot_dh": rk.dh, "T_F_M": rk.T_F_M}

EPS = {"x": 0.015, "y": {30.0: 0.0169, -30.0: 0.020}.get, "rz": {30.0: np.radians(8.54), -30.0: np.radians(10.53)}.get}
AXIS_IDX = {"x": 0, "y": 1, "rz": 5}


def nominal_pose(psi_deg):
    phi = np.radians(PHI_DEG)
    xyz0 = beam_base_xyz + (RADIUS_MM / 1000.0) * np.array([-np.cos(phi), np.sin(phi), 0.0])
    R_aligned = reference_orientation_matrix(xyz0, beam_base_xyz)
    R_psi = Rot.from_rotvec([0.0, 0.0, np.radians(psi_deg)]).as_matrix()
    R0 = R_aligned @ R_psi
    rotvec0 = Rot.from_matrix(R0).as_rotvec()
    return xyz0, rotvec0, R0


def perturbed_pose(xyz0, rotvec0, R0, axis, sign, psi_deg):
    eps = EPS["y"](psi_deg, 0.017) if axis == "y" else (EPS["rz"](psi_deg, np.radians(9.5)) if axis == "rz" else EPS["x"])
    idx = AXIS_IDX[axis]
    if idx < 3:
        d = np.zeros(3)
        d[idx] = sign * eps
        return xyz0 + d, rotvec0
    ax = idx - 3
    d = np.zeros(3)
    d[ax] = sign * eps
    R_pert = Rot.from_rotvec(d).as_matrix() @ R0
    return xyz0, Rot.from_matrix(R_pert).as_rotvec()


def ik_q0(xyz, rotvec, q_seed):
    pool = build_seed_pool(q_seed)
    sols = robust_ik_for_pose(xyz, rotvec, pool, rk.dh, rk.ik_cfg)
    if not sols:
        raise RuntimeError(f"no IK solution for xyz={xyz} rotvec={rotvec}")
    return min(sols, key=lambda qq: float(np.max(np.abs(urik.wrapped_joint_difference(qq, q_seed)))))


def contact_profile(model, xyz, rotvec, insertion_m):
    """c_min / s_frac / normal / tangent at the closest wall approach along
    the beam's solved centreline, excluding the fixed base anchor node
    (same exclusion as analyze_contact_mechanism_v2.beam_contact_profile,
    for the same reason: node 0 doesn't move and trivially wins every
    search in a tight tube)."""
    out = solve_pose(model, xyz, rotvec, insertion_m)
    centerline = np.asarray(out["centerline"])
    seg = np.diff(centerline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1] if arc[-1] > 1e-9 else 1.0

    r_beam = float(model.contact_cfg.params.r_beam)
    delta, Rloc, _, grad_delta, _ = model.lumen_query.closest_many_with_gradients(centerline, window=None)
    c = Rloc - delta - r_beam
    M = centerline.shape[0]
    i_min = int(np.argmin(c[1:])) + 1 if M > 1 else 0
    c_min = float(c[i_min])
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
    print(f"Building model bundle (contact model wired to real vessel lumen)...")
    bundle = build_model_bundle(LUMEN_FILE, beam_base_xyz6, placeholder_magnet_pose6, L_M)
    model_c = bundle.models["contact"]
    model_nc = bundle.models["no_contact"]
    r_beam_mm = float(model_c.contact_cfg.params.r_beam) * 1000.0
    print(f"contact enabled={model_c.contact_cfg.enabled} use_in_jacobian={model_c.contact_cfg.use_in_jacobian} r_beam={r_beam_mm:.2f}mm")

    adapter_c = build_diagnostic_adapter(beam_model=model_c, controller_pack=controller_pack, jacobian_mode="accurate")

    q_seed = np.array([-0.655, -1.767, -1.881, -1.064, 1.571, -0.662])  # arbitrary reasonable UR seed

    for psi_deg in (30.0, -30.0):
        print(f"\n{'='*70}\npsi = {psi_deg:+.0f} deg  (phi={PHI_DEG}deg, r={RADIUS_MM}mm, L={L_MM}mm)\n{'='*70}")
        xyz0, rotvec0, R0 = nominal_pose(psi_deg)
        q0 = ik_q0(xyz0, rotvec0, q_seed)
        state0 = np.concatenate([q0, [L_M]])

        print("\n--- Check 1: J_C^accurate vs J_C^FD (implementation) ---")
        # Library defaults (1e-6/1e-6) are far smaller than the contact
        # solver's own numerical precision and produce a spurious ~100%
        # validation failure -- documented 2026-10-01 in build_vessel_plan.py
        # (--finite-difference-joint-step-rad/--finite-difference-insertion-step-m).
        # Use the same battle-tested step sizes here.
        chain = adapter_c.validate_chain_rule(state0, joint_step_rad=1.5e-2, insertion_step_m=5.0e-3)
        print(f"relative_frobenius_error = {chain['relative_frobenius_error']:.6f}")
        print(f"max_abs_element_error    = {chain['maximum_absolute_element_error']:.6e}")
        print("J_C^accurate (3 rows x,y,z x 7 cols q1..q6,L):")
        print(np.round(chain["analytical"][:3], 6))
        print("J_C^FD:")
        print(np.round(chain["finite_difference"][:3], 6))

        print("\n--- Check 1b: directional comparison (same metrics as Stage 1) ---")
        # The Frobenius relative error above conflates gain and direction.
        # Per the user's explicit follow-up: if the dominant/weak directions
        # agree well even though gains differ by 20-30%, that is a tolerable
        # numerical-sensitivity issue, not a reason to distrust J_C's
        # qualitative structure. A >30deg dominant-direction error, or a
        # different weak direction/rank structure, would be a stop sign.
        J_fast_xy = chain["analytical"][:2]   # XY rows x 7 cols (q1..q6, L)
        J_fd_xy = chain["finite_difference"][:2]
        labels7 = ["q1", "q2", "q3", "q4", "q5", "q6", "L"]
        print(f"{'col':>4s} {'theta_deg':>10s} {'gain(fast/FD)':>14s} {'|fast|':>10s} {'|FD|':>10s}")
        for i, lab in enumerate(labels7):
            a, b = J_fast_xy[:, i], J_fd_xy[:, i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos_t = np.clip(np.dot(a, b) / (na * nb), -1, 1) if na > 1e-9 and nb > 1e-9 else np.nan
            theta = np.degrees(np.arccos(cos_t)) if not np.isnan(cos_t) else float("nan")
            gain = na / nb if nb > 1e-9 else float("nan")
            print(f"{lab:>4s} {theta:10.2f} {gain:14.3f} {na:10.6f} {nb:10.6f}")

        U_f, S_f, _ = np.linalg.svd(J_fast_xy)
        U_fd, S_fd, _ = np.linalg.svd(J_fd_xy)
        cos_dom = np.clip(abs(np.dot(U_f[:, 0], U_fd[:, 0])), -1, 1)
        theta_dom = np.degrees(np.arccos(cos_dom))
        cos_weak = np.clip(abs(np.dot(U_f[:, 1], U_fd[:, 1])), -1, 1)
        theta_weak = np.degrees(np.arccos(cos_weak))
        print(f"fast singular values: {np.round(S_f, 5)}  dominant dir: {np.round(U_f[:,0],4)}")
        print(f"FD   singular values: {np.round(S_fd, 5)}  dominant dir: {np.round(U_fd[:,0],4)}")
        print(f"theta_dom (fast vs FD) = {theta_dom:.2f}deg   theta_weak = {theta_weak:.2f}deg")
        verdict = "PASS" if theta_dom < 10.0 else ("MARGINAL" if theta_dom < 30.0 else "FAIL")
        print(f"directional verdict: {verdict}  (PASS<10deg, MARGINAL<30deg, FAIL>=30deg)")

        print("\n--- Check 2: contact state + mode-switch sanity ---")
        c_min, s_frac, normal_c, tangent_c, centerline, tip = contact_profile(model_c, xyz0, rotvec0, L_M)
        in_contact_nominal = c_min <= CONTACT_BAND_M
        print(f"c_min={c_min*1000:.3f}mm  s_c={s_frac:.3f}  normal_c={np.round(normal_c,4)}  tangent_c={np.round(tangent_c,4)}")
        print(f"nominal in_contact (<= {CONTACT_BAND_M*1000:.1f}mm band) = {in_contact_nominal}")
        print(f"tip = {np.round(tip, 5)}")

        mode_switch_found = False
        for axis in ("x", "y", "rz"):
            xyz_p, rv_p = perturbed_pose(xyz0, rotvec0, R0, axis, +1, psi_deg)
            xyz_m, rv_m = perturbed_pose(xyz0, rotvec0, R0, axis, -1, psi_deg)
            c_p, s_p, _, _, _, _ = contact_profile(model_c, xyz_p, rv_p, L_M)
            c_m, s_m, _, _, _, _ = contact_profile(model_c, xyz_m, rv_m, L_M)
            side_p = c_p <= CONTACT_BAND_M
            side_m = c_m <= CONTACT_BAND_M
            same_side = side_p == side_m
            if not same_side:
                mode_switch_found = True
            print(f"  axis {axis}: c_min(+eps)={c_p*1000:+7.3f}mm (in_contact={side_p})  "
                  f"c_min(-eps)={c_m*1000:+7.3f}mm (in_contact={side_m})  same_side={same_side}")
        print(f"ANY mode switch across the 3 perturbed axes: {mode_switch_found}")

        print("\n--- Check 3: J_C vs J_NC qualitative mechanical effect ---")
        provider_c = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=True, jacobian_mode="accurate")
        provider_nc = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate")
        J_C = provider_c(state0)[:2, :]   # XY rows only, matching Stage-1's camera-blind-Z convention
        J_NC = provider_nc(state0)[:2, :]
        J_C_xyrz = J_C[:, [0, 1, 5]]
        J_NC_xyrz = J_NC[:, [0, 1, 5]]
        U_c, S_c, _ = np.linalg.svd(J_C_xyrz)
        U_nc, S_nc, _ = np.linalg.svd(J_NC_xyrz)
        print(f"J_C   (x,y,rz cols, XY rows):\n{np.round(J_C_xyrz, 6)}")
        print(f"J_NC  (x,y,rz cols, XY rows):\n{np.round(J_NC_xyrz, 6)}")
        print(f"J_C  singular values: {np.round(S_c, 5)}  dominant dir: {np.round(U_c[:,0],4)}")
        print(f"J_NC singular values: {np.round(S_nc, 5)}  dominant dir: {np.round(U_nc[:,0],4)}")
        ratio = S_c[0] / S_nc[0] if S_nc[0] > 1e-12 else float("nan")
        print(f"sigma1 ratio (contact/no-contact) = {ratio:.4f}  (hypothesis: <1 if contact reduces mobility)")
        cos_u = np.clip(abs(np.dot(U_c[:, 0], U_nc[:, 0])), -1, 1)
        print(f"angle between dominant directions (C vs NC): {np.degrees(np.arccos(cos_u)):.2f}deg")

        identical = np.allclose(J_C_xyrz, J_NC_xyrz, atol=1e-9)
        pathological = (not np.all(np.isfinite(J_C_xyrz))) or np.allclose(J_C_xyrz, 0.0, atol=1e-12)
        print(f"J_C identical to J_NC (pathology: contact doing nothing): {identical}")
        print(f"J_C non-finite or all-zero (pathology): {pathological}")


if __name__ == "__main__":
    main()
