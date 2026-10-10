"""Diagnose the contact model at a REAL, physically-achieved joint
configuration + insertion observed in the lab (the user reports the real
beam can follow a path closer to the left wall than the centreline under
contact -- this checks what the offline-configuration optimizer's own
model predicts there, and whether THAT state is well- or ill-conditioned,
to help explain why the optimizer struggles to find it).

Also saves a top-down (world X-Y) + side (world X-Z) plot of the real
beam's full predicted centreline against the real vessel, since the
side view is the only one that shows the Z-offset finding at all (a
top-down-only view would hide it completely).

Usage: python diagnose_real_lab_state.py <q1> <q2> <q3> <q4> <q5> <q6> <insertion_mm> [--out-png PATH]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin as rk
from proper_research.rig_calibration import BEAM_BASE_XYZ_M, CURRENT_LUMEN_FILE, REFERENCE_MAGNET_POSE6, beam_base_pose6
from proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole import build_model_bundle, solve_pose
from proper_research.controllers.beam_jacobian_providers import build_diagnostic_adapter
from proper_research.vision.detect_blue import load_vessel_lumen_robot_frame

CONTACT_BAND_M = 0.5e-3
OUT_DIR = Path(__file__).resolve().parents[5] / "plans" / "stage3_design"  # archive/ adds one level


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
    out_png = None
    if "--out-png" in sys.argv:
        out_png = Path(sys.argv[sys.argv.index("--out-png") + 1])

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

    # ---------------------------------------------------------------
    # Plot: top-down (X-Y) + side (X-Z) views of the real beam shape
    # against the real vessel. The side view is the only one that shows
    # the magnet-Z-vs-beam-base-Z offset at all.
    # ---------------------------------------------------------------
    lumen_C, lumen_R, _ = load_vessel_lumen_robot_frame(CURRENT_LUMEN_FILE)
    contact_pt = centerline[int(round(s_frac * (centerline.shape[0] - 1)))]

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(13, 6.5))

    for ax, cols, labels in ((ax_xy, (0, 1), ("world X (R), m", "world Y (R), m")),
                              (ax_xz, (0, 2), ("world X (R), m", "world Z (R), m"))):
        i, j = cols
        ax.plot(lumen_C[:, i], lumen_C[:, j], "-", color="tab:cyan", lw=2, label="real vessel centreline")
        ax.fill_between(lumen_C[:, i], lumen_C[:, j] - lumen_R, lumen_C[:, j] + lumen_R,
                         color="tab:cyan", alpha=0.15, label="vessel wall (+-radius)")
        ax.plot(beam_base_xyz[i], beam_base_xyz[j], "ks", markersize=10, label="beam base")
        ax.plot(centerline[:, i], centerline[:, j], "o-", color="tab:orange", lw=2, markersize=4,
                label="real-state beam centreline (model)")
        ax.plot(xyz[i], xyz[j], "D", color="tab:purple", markersize=10, label="magnet position")
        ax.plot(tip[i], tip[j], "*", color="gold", markeredgecolor="k", markersize=16, label="tip")
        ax.plot(contact_pt[i], contact_pt[j], "x", color="red", markersize=14, markeredgewidth=3,
                label=f"contact point (c_min={c_min*1e3:.2f}mm)")
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    ax_xz.axhline(beam_base_xyz[2], color="gray", ls="--", lw=1, alpha=0.7)
    ax_xz.annotate(f"beam-base Z = {beam_base_xyz[2]*1e3:.1f}mm", xy=(beam_base_xyz[0], beam_base_xyz[2]),
                    xytext=(10, 8), textcoords="offset points", fontsize=8, color="gray")
    ax_xz.annotate(f"magnet Z = {xyz[2]*1e3:.1f}mm\n(offset {abs(xyz[2]-beam_base_xyz[2])*1e3:.1f}mm)",
                    xy=(xyz[0], xyz[2]), xytext=(10, -18), textcoords="offset points",
                    fontsize=8, color="tab:purple")

    ax_xy.set_title("Top-down (X-Y)")
    ax_xz.set_title("Side (X-Z) -- shows the Z offset")
    ax_xy.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Real lab state: insertion={insertion_mm:.0f}mm, c_min={c_min*1e3:.2f}mm, "
                 f"cond(J)={S[0]/max(S[-1],1e-12):.1f}, rank={int(np.sum(S > 1e-6*S[0]))}", fontsize=11)
    fig.tight_layout()

    if out_png is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_png = OUT_DIR / f"real_lab_state_ins{insertion_mm:.0f}mm.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nsaved plot to {out_png}")


if __name__ == "__main__":
    main()
