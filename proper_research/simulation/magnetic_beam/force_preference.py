"""Resultant magnetic force/torque on the beam, and its sensitivity to the
source magnet's actuation -- built to support a SOFT "actuation preference"
tie-breaking term (not a hard constraint) in the offline configuration solve
and in the online controllers.

Physics note (see this session's design discussion): for a rigid point
dipole, F = grad(m.B) and tau = m x B, and aligning the dipole with the
local field drives torque toward zero while force depends on the field
GRADIENT as well as orientation -- force and torque are not generally
maximized by the same configuration, so this module reports both
separately rather than conflating them into one "alignment" score.

This beam has embedded, DISTRIBUTED magnetization (not a single lumped tip
dipole) -- ``magnetic_wrench_density_cosserat_profile_segments`` already
accounts for that by computing a force/torque DENSITY at every segment
midpoint along the solved centerline. "The magnetic force/torque on the
beam" here therefore means the net RESULTANT wrench obtained by integrating
that density over arclength -- the same density the beam solver itself
uses internally (via ``energy.py``/``energy_optimized.py``), not a separate
lumped-dipole approximation layered on top.

Everything here is a READ of an already-solved/solvable beam state (via
``MagneticBeamForwardModel.solve``/``build_problem``) plus one new
finite-difference Jacobian of the resultant wrench w.r.t. the magnet's own
pose+insertion (``p7``). The pose7->joint-space chain rule reuses
``controller_adapters.assemble_tip_robot_jacobian`` UNCHANGED (it is generic
over any (3,7) linear map, not specific to tip position -- see that
function's own docstring): callers pass a (3,7) force or torque sensitivity
in place of the tip-Jacobian it was written for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller_adapters import assemble_tip_robot_jacobian
from .kinematics import quat_from_rotvec_ur
from .magnetism import (
    dipole_field_from_source,
    dipole_from_pose,
    magnetic_force_analytical,
    magnetic_wrench_density_cosserat_profile_segments,
)

Array = np.ndarray

__all__ = [
    "MagneticWrench",
    "total_magnetic_wrench",
    "magnetic_wrench_pose7_jacobian",
    "magnetic_wrench_joint_jacobian",
    "BeamSegmentSnapshot",
    "solved_beam_segments",
    "frozen_shape_wrench",
    "frozen_shape_wrench_pose7_jacobian",
    "frozen_shape_wrench_joint_jacobian",
]


@dataclass(frozen=True)
class MagneticWrench:
    force_N: Array        # (3,) net force on the beam, world/robot-base frame
    torque_Nm: Array       # (3,) net moment about `about_point_m`, same frame
    about_point_m: Array   # (3,) the point the moment was taken about


def total_magnetic_wrench(
    model, p7: Array, *, about_point_m: Array | None = None, reuse_cache: bool = True,
) -> MagneticWrench:
    """Net force + moment the source magnet's field exerts on the WHOLE beam
    at actuation ``p7`` (``[magnet_x,y,z,rx,ry,rz,L_inserted]``), by
    integrating the model's own per-segment wrench density over arclength.

    ``about_point_m``: reference point for the reported moment (default: the
    beam's own base node, i.e. the first solved centerline point -- a
    natural anchor since that is where this project's other beam-base
    geometry, ``rig_calibration.BEAM_BASE_XYZ_M``, already pivots). Changing
    it only redistributes how much of ``f_mid``'s own moment-arm contributes
    to the reported torque; the per-segment ``tau_mid`` (the local m x B
    term) is independent of this choice.
    """
    p7 = np.asarray(p7, dtype=float).reshape(7)
    result = model.solve(p7, commit=True, reuse_cache=reuse_cache)
    problem = model.build_problem(p7)
    s = np.linspace(0.0, float(problem.L_model), int(problem.N_nodes))
    f_mid, tau_mid, _B_mid, _m_mid, _s_mid = magnetic_wrench_density_cosserat_profile_segments(
        result.p, result.q, s, problem.m_src, problem.r_src,
        problem.m_local_fun, problem.m_moment,
    )
    ds = np.diff(s)
    p_mid = 0.5 * (result.p[:, :-1] + result.p[:, 1:])
    about = (
        p_mid[:, 0].copy() if about_point_m is None
        else np.asarray(about_point_m, dtype=float).reshape(3)
    )
    arm = p_mid - about[:, None]
    tau_from_force = np.cross(arm.T, f_mid.T).T
    force_total = np.sum(f_mid * ds[None, :], axis=1)
    torque_total = np.sum((tau_mid + tau_from_force) * ds[None, :], axis=1)
    return MagneticWrench(force_N=force_total, torque_Nm=torque_total, about_point_m=about)


def magnetic_wrench_pose7_jacobian(
    model, p7: Array, *,
    eps_pos_m: float = 2.0e-4, eps_rot_rad: float = 2.0e-3, eps_L_m: float = 5.0e-4,
    about_point_m: Array | None = None,
) -> tuple[Array, Array]:
    """Central-difference d(F_total)/d(p7), d(tau_total)/d(p7), each (3,7).

    Step sizes default to the same order of magnitude this project already
    uses for its other finite-difference beam Jacobians (see
    ``build_vessel_plan.py``'s ``--finite-difference-joint-step-rad``/
    ``--finite-difference-insertion-step-m``): well above the beam solver's
    own convergence tolerance (so this measures the real local slope, not
    solver noise) and small enough to stay in the roughly-linear regime.

    This is a NEW finite-difference Jacobian -- of the RESULTANT WRENCH, not
    of the tip position/pose this project's existing, chain-rule-validated
    FD Jacobians target -- so it has not itself been validated against an
    analytic derivative. Treat it as a reasonable linearization for a SOFT
    tie-breaking cost term, not as a safety-critical constraint gradient.
    """
    p7 = np.asarray(p7, dtype=float).reshape(7)
    about = (
        total_magnetic_wrench(model, p7, reuse_cache=False).about_point_m
        if about_point_m is None else np.asarray(about_point_m, dtype=float).reshape(3)
    )
    steps = np.array([eps_pos_m] * 3 + [eps_rot_rad] * 3 + [eps_L_m], dtype=float)
    dF_dp7 = np.zeros((3, 7), dtype=float)
    dtau_dp7 = np.zeros((3, 7), dtype=float)
    for i in range(7):
        step = np.zeros(7, dtype=float)
        step[i] = steps[i]
        plus = total_magnetic_wrench(model, p7 + step, about_point_m=about, reuse_cache=False)
        minus = total_magnetic_wrench(model, p7 - step, about_point_m=about, reuse_cache=False)
        dF_dp7[:, i] = (plus.force_N - minus.force_N) / (2.0 * steps[i])
        dtau_dp7[:, i] = (plus.torque_Nm - minus.torque_Nm) / (2.0 * steps[i])
    return dF_dp7, dtau_dp7


def magnetic_wrench_joint_jacobian(
    model, p7: Array, *, J_arm_spatial: Array, insertion_jacobian: float | Array = 1.0,
    source_offset_world: Array | None = None, about_point_m: Array | None = None,
    **fd_kwargs,
) -> tuple[Array, Array]:
    """d(F_total)/d[q_arm, L] and d(tau_total)/d[q_arm, L], each (3, n_arm+1).

    Chains ``magnetic_wrench_pose7_jacobian``'s (3,7) pose-tangent
    sensitivity through the ROBOT's own spatial Jacobian via
    ``assemble_tip_robot_jacobian`` -- the exact same chain-rule helper this
    project already uses to build the tip-position/robot Jacobian
    (``J_arm_spatial`` = e.g. ``ur_magnet_ik_jacobian_validation.
    geometric_jacobian(q, dh, T_F_M)``, the full (6, n_arm) [v;omega]
    spatial Jacobian at the magnet/TCP point -- NOT the position-only (3,6)
    slice some other callers in this project take). Reused unchanged because
    it is generic over any (3,7) linear map; it has no idea whether its
    input is a tip-position sensitivity or a force/torque sensitivity.
    """
    dF_dp7, dtau_dp7 = magnetic_wrench_pose7_jacobian(
        model, p7, about_point_m=about_point_m, **fd_kwargs,
    )
    dF_dq = assemble_tip_robot_jacobian(
        dF_dp7, J_arm_spatial, insertion_jacobian=insertion_jacobian,
        source_offset_world=source_offset_world,
    )
    dtau_dq = assemble_tip_robot_jacobian(
        dtau_dp7, J_arm_spatial, insertion_jacobian=insertion_jacobian,
        source_offset_world=source_offset_world,
    )
    return dF_dq, dtau_dq


# =============================================================================
# Frozen-shape (closed-form, no-resolve) variant -- 2026-10-16.
#
# ``magnetic_wrench_pose7_jacobian`` above is exact in the sense that it
# accounts for the beam's own equilibrium shape RESPONDING to the magnet's
# motion (it re-solves the full nonlinear beam model at each of 14
# perturbed poses) -- correct, but each of those solves costs real time, so
# building the (S,3,7) schedule the online controllers read costs ~14x a
# single beam solve PER REFERENCE SAMPLE (hours for a few hundred samples).
#
# The two dipole formulas this all reduces to (``dipole_field_from_source``,
# ``magnetic_force_analytical``) are themselves CLOSED FORM in the source
# magnet's pose (r_src, m_src) -- no solve at all. If the beam's own SHAPE
# (segment midpoint positions + local magnetization, both already in world
# frame) is held FIXED at whatever the beam model's last solve produced,
# differentiating the resultant wrench w.r.t. the magnet's pose costs
# nothing but a few evaluations of those closed-form formulas -- exactly
# the same kind of approximation this project's ``live_relinearize`` option
# already makes for the magnet-exclusion/workspace terms (treat the
# magnet's OWN pose-to-Cartesian map as cheap kinematics), just carried one
# step further: here the thing held fixed is the beam's shape, not just the
# linearization point.
#
# What this misses: the beam's own equilibrium RESPONSE to the magnet
# moving -- i.e. the indirect term ``magnetic_wrench_pose7_jacobian``
# captures and this does not. For the SMALL, intra-tick/intra-horizon
# deviations this soft tie-breaking term is meant to shape (it never
# overrides the primary task-tracking cost), that is the appropriate
# trade: one solve to get a shape snapshot, then free differentiation
# around it, rather than 14 solves to also capture a second-order effect
# on a term that is itself only a tie-breaker.
#
# p7[6] (insertion L) has IDENTICALLY ZERO effect under this approximation
# -- the beam's shape, including whatever length is currently inserted, is
# already baked into the frozen snapshot, and nothing here re-derives it
# from L. If a use case specifically needs the preference to respond to
# insertion changes, it needs the exact (resolving) variant above instead.
# =============================================================================
@dataclass(frozen=True)
class BeamSegmentSnapshot:
    r_mid_world: Array   # (3,N) segment midpoint positions, world frame
    m_mid_world: Array   # (3,N) segment local magnetization, world frame
    ds: Array            # (N,) segment arclength widths
    m_body: Array        # (3,) source magnet's OWN dipole moment, body frame


def solved_beam_segments(model, p7: Array) -> BeamSegmentSnapshot:
    """One beam solve -> the per-segment shape/magnetization snapshot the
    closed-form functions below hold fixed. Exactly the same solve
    ``total_magnetic_wrench`` already pays for the nominal wrench -- this
    does not add a second one; call it once per reference sample (or once
    per tick, if ever wired live) and reuse the snapshot for both the
    nominal wrench and the cheap Jacobian below.
    """
    p7 = np.asarray(p7, dtype=float).reshape(7)
    result = model.solve(p7, commit=True, reuse_cache=True)
    problem = model.build_problem(p7)
    s = np.linspace(0.0, float(problem.L_model), int(problem.N_nodes))
    _f_mid, _tau_mid, _B_mid, m_mid_world, _s_mid = magnetic_wrench_density_cosserat_profile_segments(
        result.p, result.q, s, problem.m_src, problem.r_src,
        problem.m_local_fun, problem.m_moment,
    )
    ds = np.diff(s)
    r_mid_world = 0.5 * (result.p[:, :-1] + result.p[:, 1:])
    return BeamSegmentSnapshot(
        r_mid_world=r_mid_world, m_mid_world=m_mid_world, ds=ds,
        m_body=np.asarray(model.m_body, dtype=float).reshape(3),
    )


def frozen_shape_wrench(
    snapshot: BeamSegmentSnapshot, r_src: Array, m_src: Array, *, about_point_m: Array,
) -> MagneticWrench:
    """Resultant force/torque at source pose (r_src, m_src), with the
    beam's shape held at ``snapshot`` -- closed form, no solve. Identical
    arithmetic to ``total_magnetic_wrench``'s own integration, just reading
    (r_mid_world, m_mid_world, ds) from the snapshot instead of from a
    fresh solve.
    """
    r_src = np.asarray(r_src, dtype=float).reshape(3)
    m_src = np.asarray(m_src, dtype=float).reshape(3)
    about = np.asarray(about_point_m, dtype=float).reshape(3)
    r_mid_pts = snapshot.r_mid_world.T  # (N,3)
    B_mid = dipole_field_from_source(r_mid_pts, r_src, m_src).T          # (3,N)
    f_mid = magnetic_force_analytical(r_mid_pts, snapshot.m_mid_world.T, r_src, m_src).T  # (3,N)
    tau_mid = np.cross(snapshot.m_mid_world.T, B_mid.T).T                # (3,N)
    arm = snapshot.r_mid_world - about[:, None]
    tau_from_force = np.cross(arm.T, f_mid.T).T
    force_total = np.sum(f_mid * snapshot.ds[None, :], axis=1)
    torque_total = np.sum((tau_mid + tau_from_force) * snapshot.ds[None, :], axis=1)
    return MagneticWrench(force_N=force_total, torque_Nm=torque_total, about_point_m=about)


def frozen_shape_wrench_pose7_jacobian(
    snapshot: BeamSegmentSnapshot, p7: Array, *, about_point_m: Array,
    eps_pos_m: float = 2.0e-4, eps_rot_rad: float = 2.0e-3,
) -> tuple[Array, Array]:
    """Central-difference d(F_total)/d(p7), d(tau_total)/d(p7), each (3,7)
    -- of the CLOSED-FORM ``frozen_shape_wrench`` above, so this costs a
    handful of microsecond-scale evaluations, not beam solves. Column 6
    (insertion L) is exactly zero -- see module-section docstring.
    """
    p7 = np.asarray(p7, dtype=float).reshape(7)

    def _wrench_at(p7_: Array) -> MagneticWrench:
        r_src = p7_[0:3]
        m_src = dipole_from_pose(quat_from_rotvec_ur(p7_[3:6]), snapshot.m_body)
        return frozen_shape_wrench(snapshot, r_src, m_src, about_point_m=about_point_m)

    steps = np.array([eps_pos_m] * 3 + [eps_rot_rad] * 3, dtype=float)
    dF_dp6 = np.zeros((3, 6), dtype=float)
    dtau_dp6 = np.zeros((3, 6), dtype=float)
    for i in range(6):
        step = np.zeros(7, dtype=float)
        step[i] = steps[i]
        plus = _wrench_at(p7 + step)
        minus = _wrench_at(p7 - step)
        dF_dp6[:, i] = (plus.force_N - minus.force_N) / (2.0 * steps[i])
        dtau_dp6[:, i] = (plus.torque_Nm - minus.torque_Nm) / (2.0 * steps[i])
    dF_dp7 = np.concatenate([dF_dp6, np.zeros((3, 1))], axis=1)
    dtau_dp7 = np.concatenate([dtau_dp6, np.zeros((3, 1))], axis=1)
    return dF_dp7, dtau_dp7


def frozen_shape_wrench_joint_jacobian(
    snapshot: BeamSegmentSnapshot, p7: Array, *, J_arm_spatial: Array,
    about_point_m: Array, insertion_jacobian: float | Array = 1.0,
    source_offset_world: Array | None = None, **fd_kwargs,
) -> tuple[Array, Array]:
    """``frozen_shape_wrench_pose7_jacobian`` chained into joint space --
    same ``assemble_tip_robot_jacobian`` reuse as ``magnetic_wrench_joint_
    jacobian``. The returned (3,n_arm+1) Jacobians' last column (insertion)
    is exactly zero, inherited from the pose7 Jacobian above.
    """
    dF_dp7, dtau_dp7 = frozen_shape_wrench_pose7_jacobian(
        snapshot, p7, about_point_m=about_point_m, **fd_kwargs,
    )
    dF_dq = assemble_tip_robot_jacobian(
        dF_dp7, J_arm_spatial, insertion_jacobian=insertion_jacobian,
        source_offset_world=source_offset_world,
    )
    dtau_dq = assemble_tip_robot_jacobian(
        dtau_dp7, J_arm_spatial, insertion_jacobian=insertion_jacobian,
        source_offset_world=source_offset_world,
    )
    return dF_dq, dtau_dq
