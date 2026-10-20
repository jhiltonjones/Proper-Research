"""Dipole-to-point alignment: a cheap, purely KINEMATIC actuation
preference -- encourage the source magnet's own dipole moment to point
toward a fixed target point (e.g. the beam's own base/anchor), rather
than minimizing torque on the beam.

Why this replaces the torque-based preference (``simulation.magnetic_
beam.force_preference``) for day-to-day use (2026-10-19 session
decision, not a deletion of that module -- it stays available): two
problems showed up together.

1. The torque-based preference needs the beam's own magnetization/shape,
   which only the full nonlinear beam model can produce -- expensive
   exactly (14 solves/sample) or materially wrong cheaply (the frozen-
   shape approximation's torque Jacobian was off by ~150% at a test
   pose, see that module's docstring).
2. Wired as an OFFLINE TIE-BREAKER (choose among a couple of already-
   converged multistart candidates, never inside the solver's own
   objective), it was too weak to visibly change anything: checked
   directly, the floor240-forcepref plan's dipole-to-beam-base angle
   (mean ~39 deg misaligned) was within ~1 deg of the plan built with
   no preference at all, over the whole path -- the tie-breaker had
   only 1-2 near-identical candidates to choose between per node, so it
   could not meaningfully reshape the solution.

This module fixes both: the quantity is PURE RIGID-BODY KINEMATICS (the
magnet's own position/orientation and a fixed body-frame moment -- no
beam solve, no approximation caveat) and it is meant to be wired as a
REAL residual/cost term inside the solver's own objective (offline
least-squares/SLSQP, online QP, or online nullspace descent), not a
post-hoc selector -- see each caller for how.

Sign convention: ``sign=+1.0`` (default) asks the dipole to point FROM
the magnet TOWARD ``target_point_m``; ``sign=-1.0`` asks it to point
AWAY. Residual is ``m_hat - u_hat`` (zero at perfect alignment, in the
chosen direction); callers that want a scalar score/angle for
diagnostics should use ``dipole_alignment_angle_deg`` instead.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

Array = np.ndarray

__all__ = [
    "dipole_pose_from_state",
    "dipole_alignment_residual",
    "dipole_alignment_residual_and_jacobian",
    "dipole_alignment_angle_deg",
    "dipole_pose_from_q",
    "dipole_alignment_residual_from_q",
    "dipole_alignment_residual_and_jacobian_from_q",
    "dipole_alignment_angle_deg_from_q",
    "build_gradient_fn_live",
]


def _unit(v: Array, *, name: str = "vector") -> Array:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        raise ValueError(f"cannot normalize a near-zero {name}.")
    return v / n


def _residual_from_pose(r_src: Array, R_src: Array, m_body: Array, target_point_m: Array, sign: float) -> Array:
    """Core math, shared by both the adapter-based (offline) and raw-
    kinematics (online) entry points below: ``m_hat - u_hat`` (3,), zero
    when the magnet's dipole moment points exactly toward (``sign=+1``)
    or away from (``sign=-1``) ``target_point_m``. Depends only on
    DIRECTION (both terms are unit vectors) -- moving the magnet farther
    from or closer to the target changes neither term, so this residual
    cannot be reduced by distance alone, only by genuine alignment. That
    is the whole point: ``simulation.magnetic_beam.force_preference``'s
    torque (m x B) CAN be driven toward zero either by aligning m with B
    OR by increasing distance (B -> 0 as 1/r^3) -- this residual cannot.
    """
    m_hat = _unit(R_src @ np.asarray(m_body, dtype=float).reshape(3), name="dipole moment")
    u_hat = _unit(float(sign) * (np.asarray(target_point_m, dtype=float).reshape(3) - r_src),
                  name="target direction")
    return m_hat - u_hat


# ---------------------------------------------------------------------------
# Offline planner entry points (adapter + 7-vector state convention).
# ---------------------------------------------------------------------------
def dipole_pose_from_state(adapter: Any, state: Array) -> tuple[Array, Array]:
    """``(r_src, R_src)`` -- the magnet's world position and rotation
    matrix at the 7-vector ``[q1..q6, L]`` -- via ``adapter.magnet_
    transform``, the SAME rigid kinematics every other magnet-pose
    consumer in this project already uses (no new FK path)."""
    state = np.asarray(state, dtype=float).reshape(7)
    T = adapter.magnet_transform(state)
    return T[:3, 3].copy(), T[:3, :3].copy()


def dipole_alignment_residual(
    adapter: Any, state: Array, m_body: Array, target_point_m: Array, *, sign: float = 1.0,
) -> Array:
    r_src, R_src = dipole_pose_from_state(adapter, state)
    return _residual_from_pose(r_src, R_src, m_body, target_point_m, float(sign))


def dipole_alignment_residual_and_jacobian(
    adapter: Any, state: Array, m_body: Array, target_point_m: Array, *,
    sign: float = 1.0, eps_rad: float = 1.0e-6,
) -> tuple[Array, Array]:
    """Central-difference ``(residual (3,), d(residual)/d(state) (3,7))``.

    Cheap: pure rigid-body kinematics (``adapter.magnet_transform``), no
    beam solve -- 12 FK evaluations, not 14 beam equilibria. Column 6
    (insertion L) is exactly zero: the magnet's pose does not depend on
    insertion length (rigidly on the end effector).
    """
    state = np.asarray(state, dtype=float).reshape(7)
    r0 = dipole_alignment_residual(adapter, state, m_body, target_point_m, sign=sign)
    J = np.zeros((3, 7), dtype=float)
    for i in range(6):
        step = np.zeros(7, dtype=float)
        step[i] = eps_rad
        plus = dipole_alignment_residual(adapter, state + step, m_body, target_point_m, sign=sign)
        minus = dipole_alignment_residual(adapter, state - step, m_body, target_point_m, sign=sign)
        J[:, i] = (plus - minus) / (2.0 * eps_rad)
    return r0, J


def dipole_alignment_angle_deg(
    adapter: Any, state: Array, m_body: Array, target_point_m: Array, *, sign: float = 1.0,
) -> float:
    """Diagnostic angle (degrees) between the dipole moment and the
    requested direction -- 0 deg is perfect alignment, 180 deg is
    perfectly anti-aligned. Not used by the residual/Jacobian above
    (which carry the full vector, not just this scalar), but convenient
    for before/after reporting."""
    r_src, R_src = dipole_pose_from_state(adapter, state)
    cos_angle = float(np.clip(
        np.dot(
            _unit(R_src @ np.asarray(m_body, dtype=float).reshape(3)),
            _unit(float(sign) * (np.asarray(target_point_m, dtype=float).reshape(3) - r_src)),
        ), -1.0, 1.0,
    ))
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Online controller entry points (raw robot kinematics, no adapter object --
# MPC/inverse-Jacobian callers already have `dh`/`T_F_M` on hand, e.g. from
# `ur_magnet_ik_jacobian_validation.corrected_dh_from_config`, the same
# convention `_configure_magnet_exclusion`'s `live_relinearize` already
# uses). All of these are cheap enough to call EVERY tick/solve -- no
# schedule file, unlike the torque-based preference.
# ---------------------------------------------------------------------------
def dipole_pose_from_q(q6: Array, dh: Any, T_F_M: Any) -> tuple[Array, Array]:
    """``(r_src, R_src)`` via ``ur_magnet_ik_jacobian_validation.
    forward_kinematics`` -- the magnet's pose directly from the 6 joint
    angles, no insertion dependence (rigidly on the end effector)."""
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

    fk = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), dh, T_F_M)
    return fk.T_R_target[:3, 3].copy(), fk.T_R_target[:3, :3].copy()


def dipole_alignment_residual_from_q(
    q6: Array, dh: Any, T_F_M: Any, m_body: Array, target_point_m: Array, *, sign: float = 1.0,
) -> Array:
    r_src, R_src = dipole_pose_from_q(q6, dh, T_F_M)
    return _residual_from_pose(r_src, R_src, m_body, target_point_m, float(sign))


def dipole_alignment_residual_and_jacobian_from_q(
    q6: Array, dh: Any, T_F_M: Any, m_body: Array, target_point_m: Array, *,
    sign: float = 1.0, eps_rad: float = 1.0e-6,
) -> tuple[Array, Array]:
    """Central-difference ``(residual (3,), d(residual)/dq (3,6))`` -- pure
    FK, no beam solve, no insertion column (the caller's own state/command
    stack has 7 slots; this Jacobian is only ever assembled into columns
    0:6 by callers, exactly like the adapter-based variant's zero column 6).
    """
    q6 = np.asarray(q6, dtype=float).reshape(6)
    r0 = dipole_alignment_residual_from_q(q6, dh, T_F_M, m_body, target_point_m, sign=sign)
    J = np.zeros((3, 6), dtype=float)
    for i in range(6):
        step = np.zeros(6, dtype=float)
        step[i] = eps_rad
        plus = dipole_alignment_residual_from_q(q6 + step, dh, T_F_M, m_body, target_point_m, sign=sign)
        minus = dipole_alignment_residual_from_q(q6 - step, dh, T_F_M, m_body, target_point_m, sign=sign)
        J[:, i] = (plus - minus) / (2.0 * eps_rad)
    return r0, J


def dipole_alignment_angle_deg_from_q(
    q6: Array, dh: Any, T_F_M: Any, m_body: Array, target_point_m: Array, *, sign: float = 1.0,
) -> float:
    r_src, R_src = dipole_pose_from_q(q6, dh, T_F_M)
    cos_angle = float(np.clip(
        np.dot(
            _unit(R_src @ np.asarray(m_body, dtype=float).reshape(3)),
            _unit(float(sign) * (np.asarray(target_point_m, dtype=float).reshape(3) - r_src)),
        ), -1.0, 1.0,
    ))
    return float(np.degrees(np.arccos(cos_angle)))


def build_gradient_fn_live(
    *, dh: Any, T_F_M: Any, m_body: Array, target_point_m: Array, sign: float = 1.0,
    eps_rad: float = 1.0e-6,
) -> Callable[[Array, int], Array]:
    """Build a per-tick ``(state, index) -> d(cost)/d[q1..q6,L] (7,)``
    callable -- matches ``inverse_jacobian_controller.build_magnetic_
    force_preference_gradient_fn``'s signature EXACTLY (the ``index``
    argument is accepted and ignored: this term is purely kinematic, not
    keyed to a reference sample), so it drops straight into
    ``InverseJacobianBeamController``'s existing ``force_preference_
    gradient_fn``/``force_preference_gain`` nullspace-descent mechanism
    with no changes to that controller at all. ``cost = ||residual||^2``,
    ``d(cost)/dq = 2 J^T residual``; column 6 (insertion) is exactly zero.
    """
    def gradient_fn(state: Array, _index: int) -> Array:
        state = np.asarray(state, dtype=float).reshape(7)
        residual, J6 = dipole_alignment_residual_and_jacobian_from_q(
            state[:6], dh, T_F_M, m_body, target_point_m, sign=sign, eps_rad=eps_rad,
        )
        grad = np.zeros(7, dtype=float)
        grad[:6] = 2.0 * J6.T @ residual
        return grad

    return gradient_fn
