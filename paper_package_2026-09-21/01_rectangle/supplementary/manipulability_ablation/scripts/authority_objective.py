"""Shared math for the offline manipulability/authority-shaping experiment
(NOT part of the validated live controller -- see the spec this was built
from). Every function here is pure/stateless so Phase-0 diagnostics, the
conflict diagnostic, and (conditionally) the QP integration can all reuse
the identical implementation.

phi(z) = logdet(I + Jbar(z) Jbar(z)^T) = sum_i log(1 + sigma_i(Jbar)^2)

Jbar = S_p^{-1} @ J_task(z) @ D_u * dt   (W_task = I for this first experiment)

    S_p  = beam_config.position_error_scale_m (3,), the SAME task-position
           normalization Qp already uses -- NOT a new scale.
    D_u  = config.velocity_limit (7,), the HARD physical actuator velocity
           limits -- deliberately NOT the MPC input-cost scale s_u (that
           answers "how expensive", this answers "how much authority").
    dt   = config.sample_period_s

Nullspace projector P_N is built with EXACTLY the same numerical-rank SVD
logic as `stagewise_task_nullspace.build_stagewise_projectors` (copied, not
reimplemented independently, so there is no risk of the two projectors
silently disagreeing).
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def build_Jbar(J_task: Array, S_p: Array, D_u: Array, dt: float, W_task_half: Array | None = None) -> Array:
    """J_task: (2,7) or (3,7) controlled-task Jacobian (already C @ J).
    S_p: (k,) task-position normalization (k = J_task.shape[0]).
    D_u: (7,) hard physical actuator velocity limits.
    Returns Jbar = W_task^(1/2) @ S_p^{-1} @ J_task @ D_u * dt."""
    J_task = np.asarray(J_task, dtype=float)
    S_p = np.asarray(S_p, dtype=float).reshape(-1)
    D_u = np.asarray(D_u, dtype=float).reshape(-1)
    Jbar = (J_task / S_p[:, None]) * D_u[None, :] * float(dt)
    if W_task_half is not None:
        Jbar = W_task_half @ Jbar
    return Jbar


def phi_authority(Jbar: Array) -> tuple[float, Array]:
    """Returns (phi, singular_values). phi = sum_i log(1+sigma_i^2), computed
    via SVD (not via logdet of the full I+JJ^T matrix, though the two are
    verified equal in the regression tests -- SVD is numerically better
    conditioned when a singular value is near zero)."""
    sv = np.linalg.svd(Jbar, compute_uv=False)
    phi = float(np.sum(np.log1p(sv ** 2)))
    return phi, sv


def logdet_authority(Jbar: Array) -> float:
    """Direct logdet(I + Jbar Jbar^T), used ONLY to cross-check phi_authority
    (regression test D) -- not used in the hot path."""
    k = Jbar.shape[0]
    M = np.eye(k) + Jbar @ Jbar.T
    sign, logdet = np.linalg.slogdet(M)
    return float(logdet) if sign > 0 else float("nan")


def nullspace_projector(J_task: Array, S_z: Array, rank_tol: float = 1.0e-9) -> tuple[Array, Array, int]:
    """EXACT copy of stagewise_task_nullspace.build_stagewise_projectors's
    per-stage logic, for a single state. Returns (P_R, P_N, rank)."""
    S_z = np.asarray(S_z, dtype=float).reshape(7)
    J_s = J_task * S_z[None, :]
    _, S, Vt = np.linalg.svd(J_s, full_matrices=True)
    rank = int(np.sum(S > rank_tol * max(float(S[0]), 1.0))) if S.size else 0
    V_r = Vt[:rank]
    Pr = V_r.T @ V_r
    P_N = np.eye(7) - Pr
    return Pr, P_N, rank


def fd_gradient_normalized(
    phi_fn, z_lin: Array, S_z: Array, eps: float,
) -> Array:
    """Centered finite-difference gradient of phi w.r.t. NORMALIZED
    configuration coordinates z_tilde = S_z^{-1} z, evaluated at z_lin.

    g_i = [phi(z_lin + eps*S_z*e_i) - phi(z_lin - eps*S_z*e_i)] / (2*eps)

    phi_fn: callable z (7,) -> float (already wraps J(z) -> C@J -> Jbar -> phi).
    """
    z_lin = np.asarray(z_lin, dtype=float).reshape(7)
    S_z = np.asarray(S_z, dtype=float).reshape(7)
    g = np.zeros(7, dtype=float)
    for i in range(7):
        step = np.zeros(7)
        step[i] = eps * S_z[i]
        g[i] = (phi_fn(z_lin + step) - phi_fn(z_lin - step)) / (2.0 * eps)
    return g
