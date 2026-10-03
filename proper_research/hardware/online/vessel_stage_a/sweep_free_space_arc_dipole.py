#!/usr/bin/env python3
"""Free-space fixed-radius arc + dipole-rotation map.

Decisive first experiment for the "is the no-contact beam model
overbending?" question: hold insertion fixed, remove vessel contact, keep
the source magnet at a FIXED radius r=101.5mm from the beam base (the same
range the recalibrated vessel-insertion start config actually operates at
-- 101.80mm, see vessel_magnet_initial_position_2026-10-02_recalibrated.json
-- NOT the much larger 28-38cm range used by the 2026-09-10 calibration
sweep), and sweep its position around the beam base through
phi_arc in {-80,-40,0,40,80} deg (world x-y plane, phi=0 = magnet directly
along world -X from the base, i.e. coaxial with the beam's own growth
direction -- same convention as calibration_2026-09-10/sweep_dipole_arc.py).
At each arc position, rotate the source dipole about world Z through
psi in {0,45,...,315} deg relative to a single fixed reference orientation
(dipole along world -X) -- i.e. position and dipole-rotation are swept
independently, exactly as in that script's Part A/Part B split.

For every one of the 5x8=40 static poses, this records:
  - the exact magnet pose6 (position + rotvec, robot frame) and the
    implied TCP pose6 and joint solution,
  - the full model-predicted centerline + tip (no-contact model),
  - the model's internal B (field at the beam magnet), torque, and force,
    computed from the dipole-dipole formulas using the model's own real
    parameters (source moment 900 A*m^2, beam moment-per-length
    0.6883145406820589 A*m, EI_tip=1.2566370614359175e-6 N*m^2 -- all
    re-derived live in the 2026-10-03 dipole-rotation investigation),
  - (hardware mode only) the camera-observed tip position + chord tangent,
    and e_tip = ||tip_model - tip_cam||.

Modes
-----
--mode model-only (default): pure physics, no robot/camera. Safe to run
    anywhere, answers "what does the model predict across this map".
--mode dry-run-ik: additionally solves IK for every pose and reports
    convergence / joint-step-from-seed, WITHOUT touching the robot. Use
    this to check reachability/safety margins before ever moving hardware.
--mode hardware: actually connects to the robot + camera and executes the
    sweep. Requires --i-confirm-hardware-motion as an extra explicit guard.
    This moves the magnet on a tight ~101.5mm-radius arc -- much closer to
    the beam base/insertion assembly than any prior sweep (the 2026-09-10
    calibration used >=27.8cm) -- do not run this mode unsupervised.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.sweep_free_space_arc_dipole \\
        --lumen-file vessel_lumen_robot_frame_left1p5mm_zcorrected.json \\
        --mode model-only --out debug_outputs/sweep_free_space_arc_dipole.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

MU0 = 4.0 * np.pi * 1e-7

# Source-magnet dipole direction in the magnet body frame (unit vector).
SOURCE_DIPOLE_BODY_AXIS = np.array([-0.932073, 0.361306, 0.026427])
# Source magnet moment magnitude -- confirmed 2026-10-03 from the model's
# own m_body (NOT the beam's own ~0.017 A*m^2 moment -- that earlier mixup
# was a 52,325x error, see the dipole-rotation investigation writeup).
SOURCE_MOMENT_A_M2 = 900.0
# Beam moment-per-length (A*m), constant material property -- total beam
# moment magnitude = MOMENT_PER_LENGTH_A_M * L_insertion_m. From
# magnetic_beam_command_input.py's "[COMPOSITE BEAM MAGNETISATION]" log.
MOMENT_PER_LENGTH_A_M = 0.6883145406820589
# Beam tip bending stiffness (N*m^2) -- the whole beam is in this "tip"
# material zone for any insertion <= overlap_length+wire transition, which
# covers every insertion used in this sweep. From model.Kinv_fun's closure.
EI_TIP = 1.2566370614359175e-06

R_ARC_M = 0.1015
PHI_ARC_DEG = (-80.0, -40.0, 0.0, 40.0, 80.0)
PSI_DIPOLE_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
INSERTION_MM_DEFAULT = 25.0


def _minimal_rotation_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix R such that R @ a == b, for unit a, b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # a and b anti-parallel: any perpendicular axis works.
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis = axis / np.linalg.norm(axis)
        return Rot.from_rotvec(axis * np.pi).as_matrix()
    axis = axis / s
    angle = np.arctan2(s, c)
    return Rot.from_rotvec(axis * angle).as_matrix()


def reference_orientation_matrix() -> np.ndarray:
    """R_REF: world orientation with the source dipole along world -X,
    the minimal (no extra roll) rotation from SOURCE_DIPOLE_BODY_AXIS."""
    return _minimal_rotation_matrix(SOURCE_DIPOLE_BODY_AXIS, np.array([-1.0, 0.0, 0.0]))


def build_pose_list(beam_base_xyz: np.ndarray) -> list[dict]:
    """40 (phi, psi) poses: magnet position on the arc, orientation = psi
    z-rotation of the fixed reference orientation (dipole along world -X)."""
    R_REF = reference_orientation_matrix()
    poses = []
    for phi_deg in PHI_ARC_DEG:
        phi = np.deg2rad(phi_deg)
        direction = np.array([-np.cos(phi), np.sin(phi), 0.0])
        magnet_xyz = beam_base_xyz + R_ARC_M * direction
        for psi_deg in PSI_DIPOLE_DEG:
            psi = np.deg2rad(psi_deg)
            R_psi = Rot.from_rotvec([0.0, 0.0, psi]).as_matrix()
            R_pose = R_psi @ R_REF
            rotvec = Rot.from_matrix(R_pose).as_rotvec()
            poses.append({
                "label": f"phi{phi_deg:+.0f}_psi{psi_deg:03.0f}",
                "phi_arc_deg": phi_deg,
                "psi_dipole_deg": psi_deg,
                "magnet_xyz": magnet_xyz.tolist(),
                "magnet_rotvec": rotvec.tolist(),
            })
    return poses


def dipole_world_direction(magnet_rotvec: np.ndarray) -> np.ndarray:
    R = Rot.from_rotvec(magnet_rotvec).as_matrix()
    d = R @ SOURCE_DIPOLE_BODY_AXIS
    return d / np.linalg.norm(d)


def compute_B_tau_F(
    magnet_xyz: np.ndarray, magnet_rotvec: np.ndarray,
    beam_point: np.ndarray, beam_tangent_unit: np.ndarray,
) -> dict:
    """Dipole-dipole B/torque/force at `beam_point`, using the beam's own
    local -X magnetisation axis convention: m_beam = moment_mag * (-tangent)
    is wrong in general -- the beam's magnetisation is fixed to its own
    undeformed local -X axis, which after bending points along the local
    tangent at that material point (to leading order for a thin rod).
    """
    m_src_dir = dipole_world_direction(magnet_rotvec)
    m_src = SOURCE_MOMENT_A_M2 * m_src_dir

    r_vec = beam_point - magnet_xyz
    r = float(np.linalg.norm(r_vec))
    r_hat = r_vec / r

    B = (MU0 / (4.0 * np.pi * r**3)) * (3.0 * np.dot(m_src, r_hat) * r_hat - m_src)

    m_beam_mag = MOMENT_PER_LENGTH_A_M * INSERTION_MM_DEFAULT / 1000.0
    m_beam = m_beam_mag * beam_tangent_unit

    tau = np.cross(m_beam, B)

    F = (3.0 * MU0 / (4.0 * np.pi * r**4)) * (
        np.dot(m_src, r_hat) * m_beam
        + np.dot(m_beam, r_hat) * m_src
        + np.dot(m_src, m_beam) * r_hat
        - 5.0 * np.dot(m_src, r_hat) * np.dot(m_beam, r_hat) * r_hat
    )

    alpha1 = float(np.degrees(np.arccos(np.clip(np.dot(m_src_dir, beam_tangent_unit), -1, 1))))
    B_norm = np.linalg.norm(B)
    alpha2 = (
        float(np.degrees(np.arccos(np.clip(np.dot(B / B_norm, beam_tangent_unit), -1, 1))))
        if B_norm > 0 else float("nan")
    )
    alpha3 = (
        float(np.degrees(np.arccos(np.clip(np.dot(B / B_norm, m_beam / m_beam_mag), -1, 1))))
        if B_norm > 0 else float("nan")
    )

    return {
        "r_mm": r * 1000.0, "B": B.tolist(), "B_norm_T": float(B_norm),
        "tau": tau.tolist(), "tau_norm": float(np.linalg.norm(tau)),
        "F": F.tolist(), "F_norm": float(np.linalg.norm(F)),
        "alpha1_src_vs_tangent_deg": alpha1,
        "alpha2_field_vs_tangent_deg": alpha2,
        "alpha3_field_vs_beammoment_deg": alpha3,
        "elastic_phi_pred_deg": float(np.degrees(np.linalg.norm(tau) * (INSERTION_MM_DEFAULT / 1000.0) / EI_TIP)),
    }


def _centerline_of(result) -> np.ndarray:
    c = np.asarray(result.p, dtype=float)
    if c.shape[0] == 3 and c.shape[1] != 3:
        c = c.T
    return c


def build_model_bundle(lumen_file: str, beam_base_xyz6: np.ndarray, placeholder_magnet_pose6: np.ndarray, insertion_m: float):
    import proper_research.simulation.simulations.initial_conditions as initial_conditions_mod
    import proper_research.planning.planning_context as planning_context_mod

    def _make_initial_poses():
        return beam_base_xyz6.copy(), placeholder_magnet_pose6.copy(), insertion_m, 0.01

    initial_conditions_mod.make_initial_poses = _make_initial_poses
    planning_context_mod.make_initial_poses = _make_initial_poses

    from proper_research.planning.vessel_context import build_vessel_planning_context
    exp_cfg, bundle, controller_pack, out_root, lumen_C, lumen_R, provenance = (
        build_vessel_planning_context(lumen_file=lumen_file, insertion_max_m=0.12)
    )
    return bundle


def solve_pose(model, magnet_xyz: np.ndarray, magnet_rotvec: np.ndarray, insertion_m: float) -> dict:
    p7 = np.concatenate([magnet_xyz, magnet_rotvec, [insertion_m]])
    result = model.solve(p7, commit=False, reuse_cache=False)
    centerline = _centerline_of(result)
    tip = np.asarray(result.tip, dtype=float)
    beam_tangent = centerline[-1] - centerline[-2]
    beam_tangent = beam_tangent / np.linalg.norm(beam_tangent)
    bf = compute_B_tau_F(magnet_xyz, magnet_rotvec, tip, beam_tangent)
    info = result.info if hasattr(result, "info") else {}
    parts = info.get("parts", {}) if info else {}
    return {
        "success": bool(info.get("success", True)),
        "max_bend_deg": float(info.get("max_bend", float("nan"))),
        "mean_bend_deg": float(info.get("mean_bend", float("nan"))),
        "W_el": float(parts.get("W_el", float("nan"))),
        "W_m": float(parts.get("W_m", float("nan"))),
        "W_total": float(info.get("W", float("nan"))),
        "solve_path": info.get("solve_path"),
        "tip": tip.tolist(),
        "centerline": centerline.tolist(),
        "beam_tangent_at_tip": beam_tangent.tolist(),
        **bf,
    }


def ik_dry_run(magnet_xyz: np.ndarray, magnet_rotvec: np.ndarray, seed_q: np.ndarray):
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6

    tcp_off = np.array(TCP_TO_MAGNET_POSE6[:3])
    Rm = Rot.from_rotvec(magnet_rotvec).as_matrix()
    tcp_xyz = magnet_xyz - Rm @ tcp_off
    tcp6 = np.concatenate([tcp_xyz, magnet_rotvec])

    DH = urik.corrected_dh_from_config(urik.CONFIG)
    r = urik.inverse_kinematics_dls(
        T_R_target=urik.pose6_to_T(tcp6), q_seed_rad=tuple(seed_q), dh=DH,
        T_F_target=None, cfg=urik.CONFIG,
    )
    q = np.asarray(r.q_rad)
    return {
        "tcp6": tcp6.tolist(), "q_rad": q.tolist(), "converged": bool(r.converged),
        "final_position_error_mm": float(r.final_position_error_m * 1000.0),
        "max_joint_step_from_seed_deg": float(np.degrees(np.max(np.abs(q - seed_q)))),
    }, q


def main() -> None:
    global INSERTION_MM_DEFAULT

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--mode", choices=["model-only", "dry-run-ik", "hardware"], default="model-only")
    p.add_argument("--insertion-mm", type=float, default=INSERTION_MM_DEFAULT)
    p.add_argument("--seed-joints", type=float, nargs=6, default=None,
                    help="seed joint config for IK (dry-run-ik/hardware modes). "
                         "Default: the recalibrated start-config joints.")
    p.add_argument("--i-confirm-hardware-motion", action="store_true",
                    help="required in addition to --mode hardware -- extra explicit guard.")
    p.add_argument("--out", type=Path, default=Path("debug_outputs/sweep_free_space_arc_dipole.json"))
    args = p.parse_args()

    INSERTION_MM_DEFAULT = args.insertion_mm
    insertion_m = args.insertion_mm / 1000.0

    if args.mode == "hardware" and not args.i_confirm_hardware_motion:
        raise SystemExit(
            "--mode hardware requires --i-confirm-hardware-motion as an explicit extra "
            "guard -- this sweep moves the magnet on a tight ~101.5mm-radius arc, much "
            "closer to the beam base/insertion assembly than any prior sweep."
        )

    from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import (
        BEAM_BASE_PIVOT_Z, BEAM_BASE_PIVOT_XY_ROT,
    )
    beam_base_xyz6 = np.array([
        BEAM_BASE_PIVOT_XY_ROT[0], BEAM_BASE_PIVOT_XY_ROT[1], BEAM_BASE_PIVOT_Z,
        3.14159265, 0.0, 0.0,
    ])
    beam_base_xyz = beam_base_xyz6[:3]

    seed_q = np.asarray(args.seed_joints, dtype=float) if args.seed_joints is not None else np.array(
        [-0.6261470953570765, -1.7615391216673792, -1.8847217559814453,
         -1.0656255048564454, 1.5728814601898193, -1.091161076222555]
    )

    poses = build_pose_list(beam_base_xyz)
    print(f"[sweep] {len(poses)} poses (phi x psi = {len(PHI_ARC_DEG)} x {len(PSI_DIPOLE_DEG)}), "
          f"r={R_ARC_M*1000:.1f}mm, insertion={args.insertion_mm:.1f}mm, mode={args.mode}")

    # build_vessel_planning_context re-solves its own IK internally against a
    # fixed historical seed (initial_conditions.LIVE_JOINTS_RAD) regardless of
    # what start_point make_initial_poses() returns -- it is NOT reachable
    # from an arbitrary sweep pose (confirmed: poses[0]=phi-80/psi0 fails IK
    # here with reason=line_search_stalled). Use the known-good recalibrated
    # reference pose as the placeholder instead (same pattern as
    # probe_beam_configuration.py); every actual sweep pose below still goes
    # through model.solve(p7) directly, bypassing this IK path entirely.
    placeholder_magnet_pose6 = np.array(
        [0.49596750885047175, -0.5726262253988297, -0.0396560038331531,
         -2.6740649395181184, 1.6483563099965164, -0.0008255535458713929]
    )
    print("[sweep] building model bundle (no-contact)...")
    bundle = build_model_bundle(args.lumen_file, beam_base_xyz6, placeholder_magnet_pose6, insertion_m)
    model = bundle.models["no_contact"]

    results = []
    seed = seed_q.copy()
    for pose in poses:
        magnet_xyz = np.asarray(pose["magnet_xyz"])
        magnet_rotvec = np.asarray(pose["magnet_rotvec"])
        rec = dict(pose)
        rec["insertion_mm"] = args.insertion_mm
        rec["distance_to_base_mm"] = float(np.linalg.norm(magnet_xyz - beam_base_xyz) * 1000.0)

        model_out = solve_pose(model, magnet_xyz, magnet_rotvec, insertion_m)
        rec["model"] = model_out

        if args.mode in ("dry-run-ik", "hardware"):
            ik_out, q_sol = ik_dry_run(magnet_xyz, magnet_rotvec, seed)
            rec["ik"] = ik_out
            if ik_out["converged"] and ik_out["final_position_error_mm"] < 2.0:
                seed = q_sol

        print(f"  {pose['label']:16s} tip={np.round(model_out['tip'],4).tolist()} "
              f"max_bend={model_out['max_bend_deg']:.1f}deg "
              f"alpha3={model_out['alpha3_field_vs_beammoment_deg']:.1f}deg "
              f"|B|={model_out['B_norm_T']*1000:.2f}mT |tau|={model_out['tau_norm']*1e6:.2f}uN*m "
              + (f"IK conv={rec['ik']['converged']} dq={rec['ik']['max_joint_step_from_seed_deg']:.1f}deg" if "ik" in rec else ""))

        results.append(rec)

    if args.mode == "hardware":
        print("[sweep] hardware mode: execution not yet run from this entry point in this "
              "session -- model-only/dry-run-ik results above are what's being reported back "
              "to the user before any real robot motion is authorized per-pose.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "r_arc_m": R_ARC_M, "phi_arc_deg": list(PHI_ARC_DEG), "psi_dipole_deg": list(PSI_DIPOLE_DEG),
            "insertion_mm": args.insertion_mm, "beam_base_xyz": beam_base_xyz.tolist(),
            "mode": args.mode, "results": results,
        }, f, indent=1)
    print(f"[sweep] saved {len(results)} poses -> {args.out}")


if __name__ == "__main__":
    main()
