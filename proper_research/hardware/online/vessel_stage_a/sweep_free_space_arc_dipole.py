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
At each arc position, rotate the source dipole about THE MAGNET'S OWN
LOCAL Z AXIS (equivalently, the robot's own last joint / wrist 3 -- see
build_pose_list()'s docstring) through psi in {0,45,...,315} deg relative
to a single fixed reference orientation (dipole along world -X) -- i.e.
position and dipole-rotation are swept independently, exactly as in that
script's Part A/Part B split. This decomposition is not just a modeling
convenience: TCP_TO_MAGNET_POSE6 is a pure flange-local +Z offset, so a
psi sweep at fixed phi is EXACTLY realized by moving joint 6 alone (zero
IK, zero magnet-position change, zero path-safety risk -- confirmed
empirically to <1um). Position (phi) changes need a real IK/path-safety
solve; orientation (psi) changes at fixed phi never do.

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
# 2026-10-03 recalibration: re-derived from a fresh ground-truth pose (magnet
# manually jogged in line with the beam base along world X, dipole visibly
# pointing at the base) -- see initial_conditions.SOURCE_DIPOLE_BODY_AXIS's
# own docstring for the full derivation. Old value (-0.932073, 0.361306,
# 0.026427) predated the 2026-10-02 TCP_TO_MAGNET_POSE6 remount and was
# ~176deg off (effectively negated polarity).
SOURCE_DIPOLE_BODY_AXIS = np.array([0.910293410, -0.413963475, 0.000386004])
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
    """40 (phi, psi) poses: magnet position on the arc (world x-y plane,
    constant radius/z around the beam base), orientation = psi rotation of
    the fixed reference orientation ABOUT THE MAGNET'S OWN LOCAL Z AXIS
    (right-multiply: R_pose = R_REF @ Rz(psi)), not a world-frame rotation.

    This matters mechanically, not just mathematically: TCP_TO_MAGNET_POSE6
    is a pure flange-local +Z offset (zero relative rotation), so the
    magnet's local Z axis IS the flange's local Z axis IS the robot's own
    last joint's (wrist 3) rotation axis. A right-multiplied local-Z psi
    rotation is therefore EXACTLY realized by incrementing joint 6 alone --
    confirmed empirically (forward-kinematics magnet position shifts by
    <1um for a 180deg q6 sweep) -- with zero IK re-solve and zero path-
    safety risk, since the magnet doesn't move at all. A world-frame psi
    rotation (left-multiply, an earlier version of this function) instead
    requires a generic coupled position+orientation IK re-solve for every
    psi value, which is what caused the branch-jumping/path-safety
    failures this session fought before the mechanism was pointed out."""
    R_REF = reference_orientation_matrix()
    poses = []
    for phi_deg in PHI_ARC_DEG:
        phi = np.deg2rad(phi_deg)
        direction = np.array([-np.cos(phi), np.sin(phi), 0.0])
        magnet_xyz = beam_base_xyz + R_ARC_M * direction
        for psi_deg in PSI_DIPOLE_DEG:
            psi = np.deg2rad(psi_deg)
            R_psi = Rot.from_rotvec([0.0, 0.0, psi]).as_matrix()
            R_pose = R_REF @ R_psi
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


def _magnet_tcp6(magnet_xyz: np.ndarray, magnet_rotvec: np.ndarray) -> np.ndarray:
    from proper_research.simulation.simulations.initial_conditions import TCP_TO_MAGNET_POSE6
    tcp_off = np.array(TCP_TO_MAGNET_POSE6[:3])
    Rm = Rot.from_rotvec(magnet_rotvec).as_matrix()
    tcp_xyz = magnet_xyz - Rm @ tcp_off
    return np.concatenate([tcp_xyz, magnet_rotvec])


def structured_seed_deltas() -> list[np.ndarray]:
    """Branch-like perturbations (no closed-form multi-branch UR IK exists in
    this codebase -- inverse_kinematics_dls is local/DLS-based and only ever
    explores the basin around its seed). +-pi flips on single joints are a
    cheap way to probe qualitatively different wrist/elbow/shoulder
    configurations; most will fail to converge and are simply discarded."""
    out = []
    for j in range(6):
        for sign in (1.0, -1.0):
            d = np.zeros(6)
            d[j] = sign * np.pi
            out.append(d)
    d = np.zeros(6); d[1] = np.pi / 2
    out.append(d.copy()); out.append(-d)
    return out


def build_seed_pool(reference_seed: np.ndarray, n_random: int = 16, seed_rng: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed_rng)
    pool = [reference_seed.copy()]
    for d in structured_seed_deltas():
        pool.append(reference_seed + d)
    for _ in range(n_random):
        spread = rng.uniform(0.3, 2.2)
        pool.append(reference_seed + rng.uniform(-spread, spread, size=6))
    return pool


def robust_ik_for_pose(
    magnet_xyz: np.ndarray, magnet_rotvec: np.ndarray, seed_pool: list[np.ndarray], DH, ik_cfg,
    *, pos_tol_m: float = 0.002,
) -> list[np.ndarray]:
    """Try IK from every seed in the pool independently (no chaining -- a
    failure at one seed never contaminates another). Returns every distinct
    converged solution found (deduplicated to ~1deg/joint), not just the
    first."""
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

    tcp6 = _magnet_tcp6(magnet_xyz, magnet_rotvec)
    target_T = urik.pose6_to_T(tcp6)
    solutions: list[np.ndarray] = []
    for seed in seed_pool:
        r = urik.inverse_kinematics_dls(
            T_R_target=target_T, q_seed_rad=tuple(seed), dh=DH, T_F_target=None, cfg=ik_cfg,
        )
        if not (r.converged and r.final_position_error_m < pos_tol_m):
            continue
        q = np.asarray(r.q_rad)
        dup = False
        for s in solutions:
            diff = urik.wrapped_joint_difference(q, s)
            if np.max(np.abs(diff)) < np.deg2rad(1.0):
                dup = True
                break
        if not dup:
            solutions.append(q)
    return solutions


def find_safe_path(
    from_q: np.ndarray, from_xyz: np.ndarray, from_rotvec: np.ndarray,
    to_q: np.ndarray, to_xyz: np.ndarray, to_rotvec: np.ndarray,
    *, DH, ik_cfg, T_F_M, lumen_C_m: np.ndarray, exclusion_radius_m: float,
    z_bounds_m: tuple[float, float], beam_base_xyz: np.ndarray, n_substeps: int = 20,
) -> tuple[list[np.ndarray], bool, str]:
    """Direct joint-space path first; if unsafe, fall back to a multi-
    waypoint path interpolated ALONG THE ARC (constant radius r=R_ARC_M,
    constant z=beam_base z, phi interpolated linearly) rather than a
    straight Cartesian line. A straight xyz lerp between two points on a
    circle cuts INSIDE the circle -- confirmed live: at t=1/12 toward a
    153deg-separated target, the straight-line waypoint was already only
    85.9mm from the base, inside the 97mm exclusion floor, independent of
    IK/joint-space effects entirely. Since every pose in this sweep (and
    the reference pose, to within 0.3mm) lies on the same r=101.5mm circle
    by construction, interpolating phi at fixed r/z keeps every waypoint
    on that circle, clearing the exclusion radius by the same ~4.5mm
    margin the experiment itself operates at. Orientation is interpolated
    with slerp as before (independent of phi in this script's pose-list
    convention -- psi is a z-rotation applied on top of a phi-independent
    reference orientation)."""
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from scipy.spatial.transform import Slerp

    safe, msg = path_safety_check(
        from_q, to_q, DH=DH, T_F_M=T_F_M, lumen_C_m=lumen_C_m,
        exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
    )
    if safe:
        return [to_q], True, "direct"

    def _arc_phi_of(xyz: np.ndarray) -> float:
        d = xyz - beam_base_xyz
        return float(np.arctan2(d[1], -d[0]))

    phi_from = _arc_phi_of(from_xyz)
    phi_to = _arc_phi_of(to_xyz)
    dphi = float(np.angle(np.exp(1j * (phi_to - phi_from))))  # shortest angular path, wrapped to [-pi,pi]
    r_base = 0.5 * (float(np.linalg.norm(from_xyz - beam_base_xyz)) + float(np.linalg.norm(to_xyz - beam_base_xyz)))

    key_rots = Rot.from_rotvec(np.stack([from_rotvec, to_rotvec]))
    slerp = Slerp([0.0, 1.0], key_rots)

    # Try the plain constant-radius arc first; if some leg still fails (a
    # large reconfiguration can still force an unsafe joint-space detour
    # even along the arc -- confirmed live for one psi=180 phi=0->-80
    # transition), retry with the radius bulged outward mid-transition
    # (more clearance during the big reconfiguration), increasing bulge
    # until one works or we give up.
    for bulge_m in (0.0, 0.03, 0.06, 0.10, 0.15):
        q_prev = from_q.copy()
        waypoints: list[np.ndarray] = []
        fail_reason = None
        for i in range(1, n_substeps + 1):
            t = i / n_substeps
            phi_t = phi_from + t * dphi
            r_t = r_base + bulge_m * np.sin(np.pi * t)
            xyz_t = beam_base_xyz + r_t * np.array([-np.cos(phi_t), np.sin(phi_t), 0.0])
            rotvec_t = slerp([t])[0].as_rotvec()
            tcp6_t = _magnet_tcp6(xyz_t, rotvec_t)
            target_T = urik.pose6_to_T(tcp6_t)
            candidate_seeds = [q_prev] + [q_prev + d for d in structured_seed_deltas()]
            # Collect every converging candidate and keep the one CLOSEST to
            # q_prev in joint space -- taking the first match (as an earlier
            # version did) could land on a far-away branch via the
            # structured-perturbation fallback, confirmed live: a 455mm
            # magnet-z excursion between consecutive waypoints from exactly
            # this failure mode (joint-space branch jump, not a real
            # physical constraint).
            q_t = None
            best_step = None
            for cand_seed in candidate_seeds:
                r = urik.inverse_kinematics_dls(
                    T_R_target=target_T, q_seed_rad=tuple(cand_seed), dh=DH,
                    T_F_target=None, cfg=ik_cfg,
                )
                if r.converged and r.final_position_error_m < 0.002:
                    cand_q = np.asarray(r.q_rad)
                    step = float(np.max(np.abs(urik.wrapped_joint_difference(cand_q, q_prev))))
                    if best_step is None or step < best_step:
                        best_step = step
                        q_t = cand_q
            if q_t is None:
                fail_reason = f"sub-waypoint {i}/{n_substeps} IK failed to converge (tried {len(candidate_seeds)} seeds)"
                break
            safe_leg, msg_leg = path_safety_check(
                q_prev, q_t, DH=DH, T_F_M=T_F_M, lumen_C_m=lumen_C_m,
                exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m, n_samples=30,
            )
            if not safe_leg:
                fail_reason = f"sub-waypoint {i}/{n_substeps} leg unsafe: {msg_leg[:200]}"
                break
            waypoints.append(q_t)
            q_prev = q_t
        if fail_reason is None:
            return waypoints, True, f"task-space {n_substeps}-waypoint path (bulge={bulge_m*1e3:.0f}mm)"
    return [to_q], False, f"no safe path at any bulge radius (last failure: {fail_reason})"


def path_safety_check(
    from_q: np.ndarray, to_q: np.ndarray, *, DH, T_F_M, lumen_C_m: np.ndarray,
    exclusion_radius_m: float, z_bounds_m: tuple[float, float], n_samples: int = 150,
) -> tuple[bool, str]:
    """Straight-line joint-space interpolation safety check, reusing
    vessel_stage_a/common.py's own `_validate_reset_path_safe` logic (same
    magnet-exclusion-radius + magnet-z-bounds checks used before any real
    move_j in this codebase) rather than reimplementing it."""
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.hardware.online.vessel_stage_a import common as vsa_common

    def magnet_transform_fn(q: np.ndarray) -> np.ndarray:
        fk = urik.forward_kinematics(np.asarray(q, dtype=float), DH, T_F_M)
        return np.asarray(fk.T_R_target)[:3, 3]

    try:
        vsa_common._validate_reset_path_safe(
            np.asarray(from_q), np.asarray(to_q),
            magnet_transform_fn=magnet_transform_fn,
            magnet_exclusion_lumen_C_m=lumen_C_m,
            magnet_exclusion_radius_m=exclusion_radius_m,
            magnet_z_bounds_m=z_bounds_m,
            n_samples=n_samples,
        )
        return True, "ok"
    except RuntimeError as e:
        return False, str(e)


def solve_arc_sequence(
    phi_order_deg: list[float], seed_q: np.ndarray, seed_xyz: np.ndarray, seed_rotvec: np.ndarray,
    beam_base_xyz: np.ndarray, *, DH, ik_cfg, T_F_M, lumen_C: np.ndarray,
    exclusion_radius_m: float, z_bounds_m: tuple[float, float],
) -> dict[float, np.ndarray]:
    """Find a path-safe joint solution at psi=0 for every phi in
    phi_order_deg, visited in that order starting from (seed_q, seed_xyz,
    seed_rotvec). Each phi position typically has 2-4 independent IK
    solutions (different arm branches); greedily picking whichever is
    closest to the PREVIOUS pose's solution can strand you on a branch
    that has no safe path to the NEXT pose -- confirmed live: phi=0's
    closest-to-previous solution had no safe path to phi=-40 (every
    candidate >=150deg away), while a different phi=0 branch (not the
    closest one) was only 9.3deg from a phi=-40 solution and immediately
    path-safe. This does a small DFS with backtracking over the branch
    choices instead of committing greedily, which is tractable here since
    there are only len(phi_order_deg) positions with a handful of branches
    each."""
    pool = build_seed_pool(seed_q)
    sols_by_phi: dict[float, list[np.ndarray]] = {}
    xyz_by_phi: dict[float, np.ndarray] = {}
    for phi_deg in phi_order_deg:
        phi = np.deg2rad(phi_deg)
        direction = np.array([-np.cos(phi), np.sin(phi), 0.0])
        magnet_xyz = beam_base_xyz + R_ARC_M * direction
        R_REF = reference_orientation_matrix()
        magnet_rotvec = Rot.from_matrix(R_REF).as_rotvec()
        sols = robust_ik_for_pose(magnet_xyz, magnet_rotvec, pool, DH, ik_cfg)
        if not sols:
            raise RuntimeError(f"phi={phi_deg}deg: no IK solution found at psi=0.")
        pool.extend(sols)
        sols_by_phi[phi_deg] = sols
        xyz_by_phi[phi_deg] = magnet_xyz

    chosen: dict[float, np.ndarray] = {}

    def _dfs(i: int, prev_q: np.ndarray, prev_xyz: np.ndarray, prev_rotvec: np.ndarray) -> bool:
        if i == len(phi_order_deg):
            return True
        phi_deg = phi_order_deg[i]
        magnet_xyz = xyz_by_phi[phi_deg]
        R_REF = reference_orientation_matrix()
        magnet_rotvec = Rot.from_matrix(R_REF).as_rotvec()
        candidates = sorted(
            sols_by_phi[phi_deg],
            key=lambda q: float(np.max(np.abs(wrapped_joint_difference_fallback(q, prev_q)))),
        )
        for q in candidates:
            wps, safe, msg = find_safe_path(
                prev_q, prev_xyz, prev_rotvec, q, magnet_xyz, magnet_rotvec,
                DH=DH, ik_cfg=ik_cfg, T_F_M=T_F_M, lumen_C_m=lumen_C,
                exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
                beam_base_xyz=beam_base_xyz,
            )
            if not safe:
                continue
            chosen[phi_deg] = q
            if _dfs(i + 1, q, magnet_xyz, magnet_rotvec):
                return True
            del chosen[phi_deg]
        return False

    ok = _dfs(0, seed_q, seed_xyz, seed_rotvec)
    if not ok:
        raise RuntimeError(
            f"No path-safe branch combination found across phi sequence {phi_order_deg} "
            f"-- tried all {[len(sols_by_phi[p]) for p in phi_order_deg]} branch combinations."
        )
    return chosen


def wrapped_joint_difference_fallback(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    return urik.wrapped_joint_difference(q_a, q_b)


def run_hardware(
    phi_order_deg: list[float], psi_list_deg: list[float], recs_by_label: dict,
    beam_base_xyz: np.ndarray, placeholder_magnet_pose6: np.ndarray, seed_q: np.ndarray,
    *, DH, ik_cfg, T_F_M, lumen_C: np.ndarray, exclusion_radius_m: float,
    z_bounds_m: tuple[float, float], insertion_m: float, out_path: Path,
) -> None:
    """Execute a phi-position arc sequence on the real robot (each hop a
    real IK/path-safety-verified move); at each phi, sweep psi via PURE
    joint-6 motion -- no IK, no path check, zero magnet-position change
    (confirmed empirically to <1um -- see build_pose_list()'s docstring).
    Records camera-measured tip position/tangent at every (phi, psi) pose
    and compares against the no-contact model's prediction already
    computed in recs_by_label[label]['model']."""
    import time

    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.hardware.online.vessel_stage_a import common as vsa_common
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    robot = URRTDERobot(vsa_common.ROBOT_IP, frequency=125.0)
    robot.connect()
    camera = None
    try:
        q_now = np.array(robot.get_joints())
        dq0_deg = float(np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_now, seed_q)))))
        print(f"[hardware] current joints: {np.round(q_now, 4).tolist()} (max offset from "
              f"reference: {dq0_deg:.2f}deg)")
        if dq0_deg > 2.0:
            raise RuntimeError(
                f"robot is not at the reference position (max joint offset {dq0_deg:.2f}deg "
                f"> 2.0deg tolerance) -- refusing to start. Move to the reference joints "
                f"{seed_q.tolist()} first (e.g. a plain robot.move_j under direct supervision)."
            )
        safety_mode = robot.get_safety_mode()
        prot_stopped = robot.is_protective_stopped()
        if prot_stopped or safety_mode not in (1, 2):
            raise RuntimeError(
                f"robot not in a safe/normal state (safety_mode={safety_mode}, "
                f"protective_stopped={prot_stopped}) -- refusing to start."
            )
        print(f"[hardware] robot verified at reference position, safety_mode={safety_mode}, "
              f"not protective-stopped -- proceeding")

        print(f"[hardware] solving phi-only arc sequence {phi_order_deg} (psi=0 baseline)...")
        seed_xyz = placeholder_magnet_pose6[:3]
        seed_rotvec = placeholder_magnet_pose6[3:6]
        phi_solutions = solve_arc_sequence(
            phi_order_deg, q_now, seed_xyz, seed_rotvec, beam_base_xyz,
            DH=DH, ik_cfg=ik_cfg, T_F_M=T_F_M, lumen_C=lumen_C,
            exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
        )
        print(f"[hardware] arc sequence solved for all {len(phi_solutions)} phi positions "
              f"-- no real motion yet, verifying camera before moving")

        q_lower = np.asarray(ik_cfg.joint_lower_bounds_rad, dtype=float)
        q_upper = np.asarray(ik_cfg.joint_upper_bounds_rad, dtype=float)

        scfg = vsa_common._raised_stream_stream_config()
        mapper = NewFrameTipMapper(scfg)
        camera = CameraSource(
            CameraConfig(
                cam_index=0, exposure=29.0,
                image_filename="/dev/shm/sweep_free_space_arc_dipole.png",
                roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
                pivot_hint=tuple(scfg.pivot_hint_px),
            ),
            pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
            robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
            insertion_length_getter=lambda: insertion_m, frame_processor=mapper,
        )
        camera.start()
        time.sleep(1.5)
        found = 0
        for _ in range(10):
            est, _age = camera.latest(0.5)
            if est is not None:
                found += 1
            time.sleep(0.1)
        print(f"[hardware] camera health check: {found}/10 frames had a valid tip estimate")
        if found < 7:
            raise RuntimeError(f"camera unhealthy ({found}/10 valid frames) -- refusing to start.")

        hw_results = []
        R_REF = reference_orientation_matrix()
        ref_rotvec_psi0 = Rot.from_matrix(R_REF).as_rotvec()
        prev_q, prev_xyz, prev_rotvec = q_now.copy(), seed_xyz.copy(), seed_rotvec.copy()
        for phi_deg in phi_order_deg:
            q_psi0 = phi_solutions[phi_deg]
            phi = np.deg2rad(phi_deg)
            direction = np.array([-np.cos(phi), np.sin(phi), 0.0])
            magnet_xyz = beam_base_xyz + R_ARC_M * direction

            wps, safe, msg = find_safe_path(
                prev_q, prev_xyz, prev_rotvec, q_psi0, magnet_xyz, ref_rotvec_psi0,
                DH=DH, ik_cfg=ik_cfg, T_F_M=T_F_M, lumen_C_m=lumen_C,
                exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m, beam_base_xyz=beam_base_xyz,
            )
            if not safe:
                raise RuntimeError(
                    f"phi={phi_deg}deg: no safe path from the previous arc position at "
                    f"execution time (last attempt: {msg}) -- aborting."
                )
            print(f"[hardware] -> phi={phi_deg:+.0f}deg: executing {len(wps)}-waypoint arc move")
            for q_wp in wps:
                if robot.is_protective_stopped():
                    raise RuntimeError(f"phi={phi_deg}deg: protective stop detected mid-sequence -- aborting.")
                robot.move_j(list(q_wp), speed=0.25, acceleration=0.25)
                time.sleep(0.05)
            time.sleep(0.6)
            q_arc_actual = np.array(robot.get_joints())
            arc_reach_err_deg = float(np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_arc_actual, q_psi0)))))
            if arc_reach_err_deg > 1.5:
                print(f"[hardware] WARNING: phi={phi_deg}deg arc move did not reach commanded "
                      f"joints (offset {arc_reach_err_deg:.2f}deg)")

            for psi_deg in psi_list_deg:
                label = f"phi{phi_deg:+.0f}_psi{psi_deg:03.0f}"
                rec = recs_by_label.get(label)
                dpsi = np.deg2rad(psi_deg)
                q_target = q_arc_actual.copy()
                j6 = q_arc_actual[5] + dpsi
                if not (q_lower[5] <= j6 <= q_upper[5]):
                    j6_wrapped = j6 - 2.0 * np.pi * np.round(j6 / (2.0 * np.pi))
                    if q_lower[5] <= j6_wrapped <= q_upper[5]:
                        j6 = j6_wrapped
                    else:
                        print(f"[hardware] SKIP {label}: J6 target {np.degrees(j6):.1f}deg "
                              f"outside joint limits [{np.degrees(q_lower[5]):.1f},"
                              f"{np.degrees(q_upper[5]):.1f}]deg even after 2pi-wrap")
                        continue
                q_target[5] = j6

                if robot.is_protective_stopped():
                    raise RuntimeError(f"{label}: protective stop detected mid-sequence -- aborting.")
                robot.move_j(list(q_target), speed=0.3, acceleration=0.3)
                time.sleep(0.4)
                q_actual = np.array(robot.get_joints())
                reach_err_deg = float(np.degrees(np.max(np.abs(urik.wrapped_joint_difference(q_actual, q_target)))))
                if reach_err_deg > 1.5:
                    print(f"[hardware] WARNING: {label} did not reach commanded J6 "
                          f"(offset {reach_err_deg:.2f}deg)")

                tips, tangents = [], []
                t0 = time.monotonic()
                while time.monotonic() - t0 < 6.0 and len(tips) < 25:
                    est, _age = camera.latest(0.5)
                    if est is not None:
                        tip = np.asarray(est.tip_position_m, dtype=float)
                        tan = np.asarray(est.tip_tangent, dtype=float)
                        if np.all(np.isfinite(tip)) and np.all(np.isfinite(tan)):
                            tips.append(tip)
                            tangents.append(tan)
                    time.sleep(0.1)

                model_tip = None if rec is None else np.array(rec["model"]["tip"])
                model_tangent = None if rec is None else np.array(rec["model"]["beam_tangent_at_tip"])
                if tips:
                    cam_tip = np.mean(np.vstack(tips), axis=0)
                    cam_tangent = np.mean(np.vstack(tangents), axis=0)
                    cam_tangent = cam_tangent / np.linalg.norm(cam_tangent)
                    e_tip_mm = None if model_tip is None else float(np.linalg.norm(model_tip - cam_tip) * 1000.0)
                    tangent_err_deg = None if model_tangent is None else float(
                        np.degrees(np.arccos(np.clip(np.dot(model_tangent, cam_tangent), -1, 1)))
                    )
                else:
                    cam_tip, cam_tangent, e_tip_mm, tangent_err_deg = None, None, None, None

                print(f"[hardware] {label}: model_tip={None if model_tip is None else np.round(model_tip, 4).tolist()} "
                      f"cam_tip={None if cam_tip is None else np.round(cam_tip, 4).tolist()} "
                      f"e_tip={('n/a' if e_tip_mm is None else f'{e_tip_mm:.2f}mm')} "
                      f"tangent_err={('n/a' if tangent_err_deg is None else f'{tangent_err_deg:.1f}deg')} "
                      f"({len(tips)} valid frames)")

                hw_results.append({
                    "label": label, "phi_deg": phi_deg, "psi_deg": psi_deg,
                    "q_commanded": q_target.tolist(), "q_actual": q_actual.tolist(),
                    "reach_err_deg": reach_err_deg, "n_camera_frames": len(tips),
                    "model_tip": None if model_tip is None else model_tip.tolist(),
                    "model_tangent": None if model_tangent is None else model_tangent.tolist(),
                    "cam_tip": None if cam_tip is None else cam_tip.tolist(),
                    "cam_tangent": None if cam_tangent is None else cam_tangent.tolist(),
                    "e_tip_mm": e_tip_mm, "tangent_err_deg": tangent_err_deg,
                })
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"phi_order_deg": phi_order_deg, "psi_list_deg": psi_list_deg,
                                "hardware_results": hw_results}, f, indent=1)

            prev_q, prev_xyz, prev_rotvec = q_arc_actual, magnet_xyz, ref_rotvec_psi0

        print("[hardware] sequence complete -- returning to reference position")
        ref_safe, ref_msg = path_safety_check(
            prev_q, seed_q, DH=DH, T_F_M=T_F_M, lumen_C_m=lumen_C,
            exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
        )
        if ref_safe:
            robot.move_j(list(seed_q), speed=0.25, acceleration=0.25)
        else:
            wps, safe, msg = find_safe_path(
                prev_q, prev_xyz, prev_rotvec, seed_q, seed_xyz, seed_rotvec,
                DH=DH, ik_cfg=ik_cfg, T_F_M=T_F_M, lumen_C_m=lumen_C,
                exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m, beam_base_xyz=beam_base_xyz,
            )
            if safe:
                for q_wp in wps:
                    robot.move_j(list(q_wp), speed=0.25, acceleration=0.25)
            else:
                print(f"[hardware] WARNING: could not verify a safe return path ({msg}) -- "
                      f"leaving the robot at its last pose for manual recovery.")
        print(f"[hardware] done -- results saved to {out_path}")
    finally:
        if camera is not None:
            camera.stop()
        robot.close()


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
    p.add_argument("--phi-order-deg", type=str, default=None,
                    help="comma-separated phi arc positions (deg, e.g. 80,40,0,-40,-80) giving "
                         "the execution order for --mode hardware's arc sequence. Required for "
                         "that mode. At each phi, --psi-list-deg is swept via pure joint-6 "
                         "motion (no IK/path check needed -- see build_pose_list()'s docstring).")
    p.add_argument("--psi-list-deg", type=str, default=",".join(f"{p:.0f}" for p in PSI_DIPOLE_DEG),
                    help="comma-separated psi dipole-rotation values (deg) to sweep at each phi "
                         "via pure joint-6 motion. Default: all 8 values in PSI_DIPOLE_DEG.")
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
    recs_by_label = {}
    for pose in poses:
        magnet_xyz = np.asarray(pose["magnet_xyz"])
        magnet_rotvec = np.asarray(pose["magnet_rotvec"])
        rec = dict(pose)
        rec["insertion_mm"] = args.insertion_mm
        rec["distance_to_base_mm"] = float(np.linalg.norm(magnet_xyz - beam_base_xyz) * 1000.0)

        model_out = solve_pose(model, magnet_xyz, magnet_rotvec, insertion_m)
        rec["model"] = model_out
        results.append(rec)
        recs_by_label[pose["label"]] = rec

        print(f"  {pose['label']:16s} tip={np.round(model_out['tip'],4).tolist()} "
              f"max_bend={model_out['max_bend_deg']:.1f}deg "
              f"alpha3={model_out['alpha3_field_vs_beammoment_deg']:.1f}deg "
              f"|B|={model_out['B_norm_T']*1000:.2f}mT |tau|={model_out['tau_norm']*1e6:.2f}uN*m")

    if args.mode in ("dry-run-ik", "hardware"):
        from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
        from proper_research.hardware.online.vessel_stage_a.run_open_loop_vessel import _robot_kin

        DH = _robot_kin.dh
        ik_cfg = _robot_kin.ik_cfg
        # Exclusion target is the beam-base point itself (single point), NOT
        # the full digitized vessel-lumen centerline -- this is a free-space
        # (no-contact) experiment, so the physical hazard the exclusion
        # radius protects against is the magnet swinging into the beam-base/
        # insertion-assembly mount, not the vessel tube (which isn't even
        # physically relevant here). Passing the full lumen array here was a
        # bug: it made some OTHER point along the digitized tube the nearest
        # "obstacle" instead of the base, reporting a 66.5mm gap at the
        # reference pose that is actually 101.8mm from the base itself.
        lumen_C = beam_base_xyz.reshape(1, 3)
        exclusion_radius_m = 0.097
        z_bounds_m = (beam_base_xyz[2] - 0.010, beam_base_xyz[2] + 0.010)

    if args.mode == "dry-run-ik":
        print(f"\n[ik] robust multi-seed IK search (pos_tol=2mm, exclusion={exclusion_radius_m*1e3:.0f}mm, "
              f"z_bounds=[{z_bounds_m[0]*1e3:.1f},{z_bounds_m[1]*1e3:.1f}]mm)")
        pool = build_seed_pool(seed_q)
        pending = list(poses)
        for pass_i in range(3):
            still_pending = []
            n_found_this_pass = 0
            for pose in pending:
                lab = pose["label"]
                magnet_xyz = np.asarray(pose["magnet_xyz"])
                magnet_rotvec = np.asarray(pose["magnet_rotvec"])
                sols = robust_ik_for_pose(magnet_xyz, magnet_rotvec, pool, DH, ik_cfg)
                if not sols:
                    still_pending.append(pose)
                    continue
                n_found_this_pass += 1
                # Prefer the path-safe solution closest (smallest joint step) to
                # the reference seed; fall back to the closest unsafe one (still
                # reported, but flagged) if none are path-safe.
                scored = sorted(sols, key=lambda q: float(np.max(np.abs(urik.wrapped_joint_difference(q, seed_q)))))
                best = None
                best_path = None
                for q in scored:
                    waypoints, safe, msg = find_safe_path(
                        seed_q, placeholder_magnet_pose6[:3], placeholder_magnet_pose6[3:6],
                        q, magnet_xyz, magnet_rotvec,
                        DH=DH, ik_cfg=ik_cfg, T_F_M=_robot_kin.T_F_M, lumen_C_m=lumen_C,
                        exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
                        beam_base_xyz=beam_base_xyz,
                    )
                    if safe:
                        best, best_path = q, (waypoints, True, msg)
                        break
                if best is None:
                    q = scored[0]
                    waypoints, safe, msg = find_safe_path(
                        seed_q, placeholder_magnet_pose6[:3], placeholder_magnet_pose6[3:6],
                        q, magnet_xyz, magnet_rotvec,
                        DH=DH, ik_cfg=ik_cfg, T_F_M=_robot_kin.T_F_M, lumen_C_m=lumen_C,
                        exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
                        beam_base_xyz=beam_base_xyz,
                    )
                    best, best_path = q, (waypoints, safe, msg)
                recs_by_label[lab]["ik"] = {
                    "n_solutions_found": len(sols),
                    "best_q_rad": best.tolist(),
                    "max_joint_step_from_ref_deg": float(np.degrees(np.max(np.abs(urik.wrapped_joint_difference(best, seed_q))))),
                    "path_safe_from_ref": bool(best_path[1]),
                    "path_n_waypoints": len(best_path[0]),
                    "path_safety_detail": best_path[2],
                }
                # Grow the pool with every newly found solution so later poses
                # (and the next pass, for poses still pending) benefit.
                for q in sols:
                    pool.append(q)
            print(f"  pass {pass_i+1}: {n_found_this_pass} newly solved, {len(still_pending)} still pending "
                  f"(pool size now {len(pool)})")
            pending = still_pending
            if not pending:
                break
        for pose in pending:
            recs_by_label[pose["label"]]["ik"] = {"n_solutions_found": 0}

        n_ok = sum(1 for r in results if r.get("ik", {}).get("n_solutions_found", 0) > 0)
        n_path_safe = sum(1 for r in results if r.get("ik", {}).get("path_safe_from_ref"))
        print(f"[ik] {n_ok}/{len(results)} poses reachable (>=1 IK solution); "
              f"{n_path_safe}/{len(results)} have a path-safe solution from the reference seed")

    if args.mode == "hardware":
        if args.phi_order_deg is None:
            raise SystemExit(
                "--mode hardware requires --phi-order-deg <80,40,0,-40,-80> -- an explicit "
                "arc execution sequence. Refusing to invent an order at motion time."
            )
        phi_order_deg = [float(x) for x in args.phi_order_deg.split(",")]
        psi_list_deg = [float(x) for x in args.psi_list_deg.split(",")]
        valid_phi = set(PHI_ARC_DEG)
        bad_phi = [p for p in phi_order_deg if p not in valid_phi]
        if bad_phi:
            raise SystemExit(f"--phi-order-deg has values not in PHI_ARC_DEG {PHI_ARC_DEG}: {bad_phi}")
        run_hardware(
            phi_order_deg, psi_list_deg, recs_by_label, beam_base_xyz, placeholder_magnet_pose6,
            seed_q, DH=_robot_kin.dh, ik_cfg=_robot_kin.ik_cfg, T_F_M=_robot_kin.T_F_M,
            lumen_C=lumen_C, exclusion_radius_m=exclusion_radius_m, z_bounds_m=z_bounds_m,
            insertion_m=insertion_m, out_path=args.out,
        )

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
