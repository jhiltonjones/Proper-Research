"""Quaternion utilities using the source project's wxyz convention."""

from __future__ import annotations

import numpy as np

from .external import load_external_dependencies


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions stored as [w, x, y, z]."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_small_rotation(delta_phi: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a wxyz quaternion."""
    delta_phi = np.asarray(delta_phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(delta_phi))
    if angle < 1e-14:
        return np.array([1.0, *(0.5 * delta_phi)], dtype=float)
    axis = delta_phi / angle
    half = 0.5 * angle
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float)


def perturb_quaternion(
    q_src: np.ndarray,
    delta_phi: np.ndarray,
    *,
    convention: str = "world",
) -> np.ndarray:
    """Apply a small world- or body-frame rotation to a source quaternion."""
    ext = load_external_dependencies()
    q_src = ext.quat_normalize(np.asarray(q_src, dtype=float).reshape(4))
    dq = quat_from_small_rotation(delta_phi)
    if convention == "world":
        return ext.quat_normalize(quat_mul(dq, q_src))
    if convention == "body":
        return ext.quat_normalize(quat_mul(q_src, dq))
    raise ValueError("convention must be 'world' or 'body'")


def dipole_from_pose(q_src: np.ndarray, m_body: np.ndarray) -> np.ndarray:
    """Rotate a body-frame dipole vector into the world frame."""
    ext = load_external_dependencies()
    rotation = ext.quat_to_R(ext.quat_normalize(np.asarray(q_src, dtype=float).reshape(4)))
    return rotation @ np.asarray(m_body, dtype=float).reshape(3)


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 cross-product matrix [v]_x."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def source_dipole_rotation_jacobian(
    q_src: np.ndarray,
    m_body: np.ndarray,
    *,
    convention: str = "world",
) -> np.ndarray:
    """Return d(m_src)/d(delta_phi) for a rotation-vector perturbation."""
    ext = load_external_dependencies()
    q_src = ext.quat_normalize(np.asarray(q_src, dtype=float).reshape(4))
    rotation = ext.quat_to_R(q_src)
    m_body = np.asarray(m_body, dtype=float).reshape(3)
    m_src = rotation @ m_body
    if convention == "world":
        return -skew(m_src)
    if convention == "body":
        return -rotation @ skew(m_body)
    raise ValueError("convention must be 'world' or 'body'")


def dnormalize_dq(q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Jacobian of q / ||q|| with respect to the raw quaternion q."""
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < eps:
        raise ValueError("quaternion norm is too small")
    qhat = q / norm
    return (np.eye(4) - np.outer(qhat, qhat)) / norm
