from __future__ import annotations

import inspect

import numpy as np

from .kinematics import quat_normalize, quat_to_R


MU0_OVER_4PI = 1e-7


def _m_local_fun_is_field_dependent(m_local_fun) -> bool:
    """True if ``m_local_fun`` wants the local field (an INDUCED/field-following
    magnetisation model), false for the legacy fixed-moment signature
    ``m_local_fun(s_mid, m_moment)``.

    Detected by signature inspection so existing ``m_local_fun`` callables
    (2 positional args, no kwargs) are called exactly as before -- this is a
    purely additive capability.
    """
    try:
        sig = inspect.signature(m_local_fun)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return "B_local_mid" in params or any(
        p.kind == p.VAR_KEYWORD for p in params.values()
    )


def _call_m_local_fun(m_local_fun, s_mid, m_front_or_overhead, *, B_local_mid=None, R_mid=None):
    """Call ``m_local_fun`` with the local field if it wants one, else legacy 2-arg."""
    if _m_local_fun_is_field_dependent(m_local_fun):
        return m_local_fun(s_mid, m_front_or_overhead, B_local_mid=B_local_mid, R_mid=R_mid)
    return m_local_fun(s_mid, m_front_or_overhead)


def quat_stack_to_R(q: np.ndarray) -> np.ndarray:
    """
    Convert a stack of quaternions [w, x, y, z] to rotation matrices.

    Accepts:
        q shape (4, N)

    Returns:
        R shape (N, 3, 3)
    """
    q = np.asarray(q, float)

    if q.ndim != 2 or q.shape[0] != 4:
        raise ValueError(f"Expected q with shape (4, N), got {q.shape}.")

    norms = np.linalg.norm(q, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Cannot normalise a near-zero quaternion.")
    qn = q / norms
    w, x, y, z = qn

    N = q.shape[1]
    R = np.empty((N, 3, 3), dtype=float)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return R


def dipole_field_from_source(
    r_pts: np.ndarray,
    r_src: np.ndarray,
    m_src: np.ndarray,
    *,
    r_min: float = 1e-6,
) -> np.ndarray:
    """
    Magnetic field of a point dipole.

    Parameters
    ----------
    r_pts:
        Evaluation points, shape (M, 3).

    r_src:
        Source dipole position, shape (3,).

    m_src:
        Source dipole moment, shape (3,).

    Returns
    -------
    B:
        Magnetic field at r_pts, shape (M, 3).
    """
    r_pts = np.asarray(r_pts, float)
    r_src = np.asarray(r_src, float).reshape(3)
    m_src = np.asarray(m_src, float).reshape(3)

    if r_pts.ndim != 2 or r_pts.shape[1] != 3:
        raise ValueError(f"r_pts must have shape (M, 3), got {r_pts.shape}.")

    R = r_pts - r_src[None, :]
    R2 = np.sum(R * R, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, r_min**2))

    Rhat = R / Rnorm[:, None]
    mdot = Rhat @ m_src

    invR3 = 1.0 / (Rnorm**3)

    B = MU0_OVER_4PI * invR3[:, None] * (
        3.0 * mdot[:, None] * Rhat - m_src[None, :]
    )

    return B


def magnetic_force_analytical(
    r_pts: np.ndarray,
    m_pts: np.ndarray,
    r_src: np.ndarray,
    m_src: np.ndarray,
    *,
    r_min: float = 1e-6,
) -> np.ndarray:
    """
    Analytical force on dipoles m_pts due to source dipole m_src.

    Parameters
    ----------
    r_pts:
        Evaluation points, shape (M, 3).

    m_pts:
        Dipoles at evaluation points, shape (M, 3) or shape (3,).

    r_src:
        Source dipole position, shape (3,).

    m_src:
        Source dipole moment, shape (3,).

    Returns
    -------
    F:
        Force density / force-like quantity, shape (M, 3).
    """
    r_pts = np.asarray(r_pts, float)
    m_pts = np.asarray(m_pts, float)
    r_src = np.asarray(r_src, float).reshape(3)
    m_src = np.asarray(m_src, float).reshape(3)

    if r_pts.ndim != 2 or r_pts.shape[1] != 3:
        raise ValueError(f"r_pts must have shape (M, 3), got {r_pts.shape}.")

    if m_pts.ndim == 1:
        m = np.repeat(m_pts.reshape(1, 3), r_pts.shape[0], axis=0)
    elif m_pts.ndim == 2 and m_pts.shape == r_pts.shape:
        m = m_pts
    else:
        raise ValueError(
            f"m_pts must have shape (3,) or {r_pts.shape}, got {m_pts.shape}."
        )

    R = r_pts - r_src[None, :]
    R2 = np.sum(R * R, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, r_min**2))

    m_dot_R = np.sum(m * R, axis=1)
    m0_dot_R = R @ m_src
    m_dot_m0 = m @ m_src

    invR5 = 1.0 / (Rnorm**5)
    invR2 = 1.0 / (Rnorm**2)

    term = (
        m_dot_R[:, None] * m_src[None, :]
        + m0_dot_R[:, None] * m
        + m_dot_m0[:, None] * R
        - 5.0 * (m_dot_R * m0_dot_R * invR2)[:, None] * R
    )

    F = 3.0 * MU0_OVER_4PI * invR5[:, None] * term

    return F


def midpoint_quaternions(q: np.ndarray) -> np.ndarray:
    """
    Compute normalised midpoint quaternions between adjacent node quaternions.

    Input:
        q shape (4, N)

    Output:
        q_mid shape (4, N - 1)
    """
    q = np.asarray(q, float)

    if q.ndim != 2 or q.shape[0] != 4:
        raise ValueError(f"Expected q with shape (4, N), got {q.shape}.")

    q0 = q[:, :-1].copy()
    q1 = q[:, 1:].copy()

    # Enforce same hemisphere per segment.
    flip = np.sum(q0 * q1, axis=0) < 0.0
    q1[:, flip] *= -1.0

    q_mid = q0 + q1
    q_mid /= np.linalg.norm(q_mid, axis=0, keepdims=True) + 1e-12

    return q_mid


def magnetic_wrench_density_cosserat_profile_segments(
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    m_ext: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_front_or_overhead,
    *,
    r_min: float = 1e-6,
):
    """
    Segment-based magnetic quantities evaluated at segment midpoints.

    Parameters
    ----------
    p:
        Node positions, shape (3, N).

    q:
        Node quaternions, shape (4, N), convention [w, x, y, z].

    s:
        Node arclength coordinates, shape (N,).

    m_ext:
        External/source dipole in world frame, shape (3,).

    r_src:
        External/source position in world frame, shape (3,).

    m_local_fun:
        Callable returning local magnetisation at segment midpoints.
        Expected output shape: (3, N - 1).

    m_front_or_overhead:
        Passed through to m_local_fun.

    Returns
    -------
    f_mid:
        Magnetic force density at segment midpoints, shape (3, N - 1).

    tau_mid:
        Magnetic torque density at segment midpoints, shape (3, N - 1).

    B_mid:
        Magnetic field at segment midpoints, shape (3, N - 1).

    m_mid_world:
        Magnetisation at segment midpoints in world frame, shape (3, N - 1).

    s_mid:
        Segment midpoint arclengths, shape (N - 1,).
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    s = np.asarray(s, float).ravel()
    m_ext = np.asarray(m_ext, float).reshape(3)
    r_src = np.asarray(r_src, float).reshape(3)

    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"p must have shape (3, N), got {p.shape}.")
    if q.ndim != 2 or q.shape[0] != 4:
        raise ValueError(f"q must have shape (4, N), got {q.shape}.")
    if p.shape[1] != q.shape[1]:
        raise ValueError(f"p and q node counts differ: {p.shape[1]} vs {q.shape[1]}.")
    if s.size != p.shape[1]:
        raise ValueError(f"s length {s.size} does not match p nodes {p.shape[1]}.")
    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    s_mid = 0.5 * (s[:-1] + s[1:])
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])

    q_mid = midpoint_quaternions(q)
    R_mid = quat_stack_to_R(q_mid)

    r_mid_pts = p_mid.T

    # B is computed BEFORE m_local_fun: an INDUCED/field-following
    # m_local_fun (_m_local_fun_is_field_dependent) needs the local field to
    # compute the beam's own magnetisation; a legacy fixed-moment m_local_fun
    # ignores the field argument and this reordering changes nothing for it.
    B_mid = dipole_field_from_source(
        r_mid_pts,
        r_src,
        m_ext,
        r_min=r_min,
    ).T

    # World field rotated into each segment's LOCAL frame -- (R_mid[n])^T @ B_mid[:, n].
    B_local_mid = np.einsum("nji,jn->in", R_mid, B_mid)

    m_loc_mid = np.asarray(
        _call_m_local_fun(
            m_local_fun, s_mid, m_front_or_overhead,
            B_local_mid=B_local_mid, R_mid=R_mid,
        ),
        float,
    )

    expected_m_shape = (3, s.size - 1)
    if m_loc_mid.shape != expected_m_shape:
        raise ValueError(
            f"m_local_fun returned shape {m_loc_mid.shape}, "
            f"expected {expected_m_shape}."
        )

    # R_mid shape: (N - 1, 3, 3)
    # m_loc_mid.T shape: (N - 1, 3)
    m_mid_world = np.einsum("nij,nj->in", R_mid, m_loc_mid.T)

    f_mid = magnetic_force_analytical(
        r_mid_pts,
        m_mid_world.T,
        r_src,
        m_ext,
        r_min=r_min,
    ).T

    tau_mid = np.cross(m_mid_world.T, B_mid.T).T

    return f_mid, tau_mid, B_mid, m_mid_world, s_mid


def magnetic_energy_quantities_cosserat_profile_segments(
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    m_ext: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_front_or_overhead,
    *,
    r_min: float = 1e-6,
):
    """
    Return only the quantities required by ``-integral(m.B) ds``.

    The optimizer used to call the full wrench routine for every scalar-energy
    evaluation, even though force and torque are only needed by the gradient
    and final diagnostics.  Avoiding those calculations reduces line-search
    cost without changing the energy.
    """
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    s = np.asarray(s, float).ravel()
    m_ext = np.asarray(m_ext, float).reshape(3)
    r_src = np.asarray(r_src, float).reshape(3)

    if p.shape != (3, s.size):
        raise ValueError(f"p must have shape (3, {s.size}), got {p.shape}.")
    if q.shape != (4, s.size):
        raise ValueError(f"q must have shape (4, {s.size}), got {q.shape}.")
    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    s_mid = 0.5 * (s[:-1] + s[1:])
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    q_mid = midpoint_quaternions(q)
    R_mid = quat_stack_to_R(q_mid)

    # Same B-before-m_local_fun ordering as the full wrench routine (see there
    # for why): an induced m_local_fun needs the local field.
    B_mid = dipole_field_from_source(
        p_mid.T,
        r_src,
        m_ext,
        r_min=r_min,
    ).T
    B_local_mid = np.einsum("nji,jn->in", R_mid, B_mid)

    m_loc_mid = np.asarray(
        _call_m_local_fun(
            m_local_fun, s_mid, m_front_or_overhead,
            B_local_mid=B_local_mid, R_mid=R_mid,
        ),
        float,
    )
    expected = (3, s.size - 1)
    if m_loc_mid.shape != expected:
        raise ValueError(
            f"m_local_fun returned shape {m_loc_mid.shape}, expected {expected}."
        )

    m_mid_world = np.einsum("nij,nj->in", R_mid, m_loc_mid.T)
    return B_mid, m_mid_world, s_mid


def dipole_from_pose(q_src: np.ndarray, m_body: np.ndarray) -> np.ndarray:
    """
    Rotate a body-frame dipole into world frame.

    q_src:
        Source pose quaternion [w, x, y, z].

    m_body:
        Body-frame dipole moment, shape (3,).
    """
    q_src = quat_normalize(q_src)
    m_body = np.asarray(m_body, float).reshape(3)

    R = quat_to_R(q_src)

    return R @ m_body
