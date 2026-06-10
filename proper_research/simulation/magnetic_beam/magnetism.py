from __future__ import annotations

import numpy as np

from .kinematics import quat_normalize, quat_to_R


MU0_OVER_4PI = 1e-7


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

    N = q.shape[1]
    R = np.empty((N, 3, 3), dtype=float)

    for i in range(N):
        R[i] = quat_to_R(q[:, i])

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

    m_loc_mid = np.asarray(m_local_fun(s_mid, m_front_or_overhead), float)

    expected_m_shape = (3, s.size - 1)
    if m_loc_mid.shape != expected_m_shape:
        raise ValueError(
            f"m_local_fun returned shape {m_loc_mid.shape}, "
            f"expected {expected_m_shape}."
        )

    # R_mid shape: (N - 1, 3, 3)
    # m_loc_mid.T shape: (N - 1, 3)
    m_mid_world = np.einsum("nij,nj->in", R_mid, m_loc_mid.T)

    r_mid_pts = p_mid.T

    B_mid = dipole_field_from_source(
        r_mid_pts,
        r_src,
        m_ext,
        r_min=r_min,
    ).T

    f_mid = magnetic_force_analytical(
        r_mid_pts,
        m_mid_world.T,
        r_src,
        m_ext,
        r_min=r_min,
    ).T

    tau_mid = np.cross(m_mid_world.T, B_mid.T).T

    return f_mid, tau_mid, B_mid, m_mid_world, s_mid


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