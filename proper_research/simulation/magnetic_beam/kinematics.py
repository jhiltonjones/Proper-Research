from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    n = float(np.linalg.norm(q))

    if n < 1e-12:
        raise ValueError("Cannot normalise near-zero quaternion.")

    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product for quaternions in [w, x, y, z] convention.
    """
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)

    return quat_normalize(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    """
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_exp_body(u: np.ndarray, ds: float) -> np.ndarray:
    """
    Quaternion exponential for a body-frame rotation vector u * ds.

    u:
        Curvature/twist vector, shape (3,).

    ds:
        Segment length.
    """
    rotvec = np.asarray(u, float).reshape(3) * float(ds)
    angle = float(np.linalg.norm(rotvec))

    if angle < 1e-12:
        half = 0.5 * rotvec
        return quat_normalize(
            np.array([1.0, half[0], half[1], half[2]], dtype=float)
        )

    axis = rotvec / angle
    half_angle = 0.5 * angle

    return quat_normalize(
        np.array(
            [
                np.cos(half_angle),
                *(np.sin(half_angle) * axis),
            ],
            dtype=float,
        )
    )

def integrate_pq_and_sens_from_u(
    u_flat,
    *,
    p0,
    q0,
    s,
    e1=np.array([-1.0, 0.0, 0.0]),
    eps_q=1e-7,
    eps_u=1e-7,
):
    """
    Discrete sensitivity version of integrate_pq_from_u().

    This intentionally matches the exact same forward update:

        p_{i+1} = p_i + ds_i * R(q_i) e1
        q_{i+1} = normalize(q_i * exp_body(u_i, ds_i))

    Therefore S_p and S_q are sensitivities of the same discrete map used
    by energy_from_u().
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    p0 = np.asarray(p0, float).reshape(3,)
    q0 = quat_normalize(np.asarray(q0, float).reshape(4,))
    e1 = np.asarray(e1, float).reshape(3,)

    N = s.size
    n_seg = N - 1
    n_u = u_flat.size

    if n_u != 3 * n_seg:
        raise ValueError(
            f"Expected u_flat.size = {3*n_seg}, got {n_u}"
        )

    ds = np.diff(s)
    u_seg = u_flat.reshape(n_seg, 3)

    p = np.zeros((3, N), dtype=float)
    q = np.zeros((4, N), dtype=float)

    S_p = np.zeros((3, N, n_u), dtype=float)
    S_q = np.zeros((4, N, n_u), dtype=float)

    p[:, 0] = p0
    q[:, 0] = q0

    for i in range(n_seg):
        h = float(ds[i])
        ui = u_seg[i].copy()

        col0 = 3 * i
        col1 = col0 + 3

        q_i = q[:, i].copy()
        p_i = p[:, i].copy()

        # ------------------------------------------------------------
        # Forward update: exactly match integrate_pq_from_u()
        # ------------------------------------------------------------
        R_i = quat_to_R(q_i)
        tangent_i = R_i @ e1

        p_next = p_i + h * tangent_i

        dq = quat_exp_body(ui, h)
        q_next = quat_normalize(quat_mul(q_i, dq))

        p[:, i + 1] = p_next
        q[:, i + 1] = q_next

        # ------------------------------------------------------------
        # Sensitivity of p_{i+1}
        #
        # p_{i+1} = p_i + h * tangent(q_i)
        #
        # No direct dependence on current ui, because your plain
        # integrator uses q_i, not q_{i+1} or q_mid, for position.
        # ------------------------------------------------------------
        def tangent_from_q(q_raw):
            qn = quat_normalize(q_raw)
            return quat_to_R(qn) @ e1

        d_tangent_dq = numerical_jacobian(
            tangent_from_q,
            q_i,
            eps=eps_q,
        )  # shape (3, 4)

        S_p[:, i + 1, :] = (
            S_p[:, i, :]
            + h * d_tangent_dq @ S_q[:, i, :]
        )

        # ------------------------------------------------------------
        # Sensitivity of q_{i+1}
        #
        # q_{i+1} = normalize(q_i * exp_body(ui, h))
        # ------------------------------------------------------------
        def qnext_from_q(q_raw):
            qn = quat_normalize(q_raw)
            dq_local = quat_exp_body(ui, h)
            return quat_normalize(quat_mul(qn, dq_local))

        def qnext_from_u(u_local):
            dq_local = quat_exp_body(u_local, h)
            return quat_normalize(quat_mul(q_i, dq_local))

        dqnext_dq = numerical_jacobian(
            qnext_from_q,
            q_i,
            eps=eps_q,
        )  # shape (4, 4)

        dqnext_du = numerical_jacobian(
            qnext_from_u,
            ui,
            eps=eps_u,
        )  # shape (4, 3)

        S_q[:, i + 1, :] = dqnext_dq @ S_q[:, i, :]
        S_q[:, i + 1, col0:col1] += dqnext_du

    return p, q, S_p, S_q
def numerical_jacobian(f, x, eps=1e-7):
    x = np.asarray(x, float).reshape(-1)
    y0 = np.asarray(f(x), float).reshape(-1)

    J = np.zeros((y0.size, x.size), dtype=float)

    for k in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps

        yp = np.asarray(f(xp), float).reshape(-1)
        ym = np.asarray(f(xm), float).reshape(-1)

        J[:, k] = (yp - ym) / (2.0 * eps)

    return J
def integrate_pq_from_u(
    u_flat: np.ndarray,
    *,
    p0: np.ndarray,
    q0: np.ndarray,
    s: np.ndarray,
    e1: np.ndarray = np.array([-1.0, 0.0, 0.0]),
):
    """
    Integrate Cosserat centerline and orientation from segment strains.

    u_flat:
        Flattened strain vector, shape (3 * (N - 1),).

    Returns:
        p: shape (3, N)
        q: shape (4, N), quaternion [w, x, y, z]
        u_seg: shape (N - 1, 3)
    """
    s = np.asarray(s, float).ravel()
    p0 = np.asarray(p0, float).reshape(3)
    q0 = quat_normalize(q0)
    e1 = np.asarray(e1, float).reshape(3)

    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    if not np.all(np.diff(s) > 0):
        raise ValueError("s must be strictly increasing.")

    N = s.size
    ds = np.diff(s)

    u_seg = np.asarray(u_flat, float).reshape(N - 1, 3)

    p = np.zeros((3, N), dtype=float)
    q = np.zeros((4, N), dtype=float)

    p[:, 0] = p0
    q[:, 0] = q0

    for i in range(N - 1):
        R = quat_to_R(q[:, i])
        p[:, i + 1] = p[:, i] + ds[i] * (R @ e1)

        dq = quat_exp_body(u_seg[i], ds[i])
        q[:, i + 1] = quat_mul(q[:, i], dq)

    return p, q, u_seg


def quat_from_rotvec_ur(rvec: np.ndarray) -> np.ndarray:
    """
    Convert UR/scipy rotation vector to quaternion [w, x, y, z].
    """
    rvec = np.asarray(rvec, float).reshape(3)

    q_xyzw = Rot.from_rotvec(rvec).as_quat()
    q_wxyz = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=float,
    )

    return quat_normalize(q_wxyz)


def quat_wxyz_to_scipy(q_wxyz: np.ndarray) -> Rot:
    """
    Convert quaternion [w, x, y, z] to scipy Rotation.
    """
    q_wxyz = quat_normalize(q_wxyz)

    q_xyzw = np.array(
        [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
        dtype=float,
    )

    return Rot.from_quat(q_xyzw)


def scipy_to_quat_wxyz(R_obj: Rot) -> np.ndarray:
    """
    Convert scipy Rotation to quaternion [w, x, y, z].
    """
    q_xyzw = R_obj.as_quat()

    q_wxyz = np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=float,
    )

    return quat_normalize(q_wxyz)


def perturb_quat_world(q_wxyz: np.ndarray, drot_world: np.ndarray) -> np.ndarray:
    """
    Apply a small world-frame rotation perturbation.

    Convention:
        R_new = dR_world @ R0
    """
    R0 = quat_wxyz_to_scipy(q_wxyz)
    dR = Rot.from_rotvec(np.asarray(drot_world, float).reshape(3))

    return scipy_to_quat_wxyz(dR * R0)


def effective_lengths(
    L_ins: float,
    *,
    L_tip_full: float = 0.04,
    L_tip_min: float = 0.01,
) -> tuple[float, float, float]:
    """
    Convert commanded insertion length into model, wire, and magnetic-tip lengths.
    """
    L_ins = float(L_ins)
    L_tip_full = float(L_tip_full)
    L_tip_min = float(L_tip_min)

    if not np.isfinite(L_ins):
        raise ValueError(f"L_ins must be finite, got {L_ins}.")
    if L_ins < 0:
        raise ValueError(f"L_ins must be non-negative, got {L_ins}.")
    if L_tip_full <= 0:
        raise ValueError(f"L_tip_full must be positive, got {L_tip_full}.")
    if L_tip_min <= 0:
        raise ValueError(f"L_tip_min must be positive, got {L_tip_min}.")

    tip_len = min(L_ins, L_tip_full)
    wire_len = max(L_ins - L_tip_full, 0.0)
    L_model = max(L_ins, L_tip_min)

    tip_len = min(tip_len, L_model)

    return float(L_model), float(wire_len), float(tip_len)


def pose8_quat_to_pose7_rotvec(p8: np.ndarray) -> np.ndarray:
    """
    Convert p8 = [x, y, z, qw, qx, qy, qz, L]
    to p7 = [x, y, z, rx, ry, rz, L].
    """
    p8 = np.asarray(p8, float).reshape(-1)

    if p8.size != 8:
        raise ValueError(
            "Expected p8 = [x, y, z, qw, qx, qy, qz, L], "
            f"got shape {p8.shape}."
        )

    t = p8[0:3]
    q_wxyz = p8[3:7]
    L = float(p8[7])

    R_obj = quat_wxyz_to_scipy(q_wxyz)
    rvec = R_obj.as_rotvec()

    return np.array(
        [t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L],
        dtype=float,
    )


def unit(v: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    """
    Normalise a vector safely.
    """
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))

    if n > 1e-12:
        return v / n

    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)

    fallback = np.asarray(fallback, float).reshape(3)
    return fallback / (np.linalg.norm(fallback) + 1e-12)


def tip_tangent_from_centerline(centerline: np.ndarray) -> np.ndarray:
    """
    Estimate tip tangent from the final two centerline points.

    Accepts:
        (3, N) or (N, 3)
    """
    C = np.asarray(centerline, float)

    if C.ndim != 2:
        raise ValueError(f"centerline must be 2D, got shape {C.shape}.")

    if C.shape[0] == 3 and C.shape[1] >= 2:
        p_prev = C[:, -2]
        p_tip = C[:, -1]
    elif C.shape[1] == 3 and C.shape[0] >= 2:
        p_prev = C[-2, :]
        p_tip = C[-1, :]
    else:
        raise ValueError(f"Unexpected centerline shape: {C.shape}.")

    return unit(p_tip - p_prev)