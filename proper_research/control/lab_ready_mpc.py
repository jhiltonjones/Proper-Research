import numpy as np
import osqp
import scipy.sparse as sp
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
from beam_direction_magnetisation.post_processing.post_processing import quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, WarmForwardP8TipTangent, DeterministicForward6D
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
from beam_direction_magnetisation.post_processing.debug import closest_point_polyline
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel, load_polygon
from proper_research.vision.measure_length import new_capture
import matplotlib.pyplot as plt
import time
from proper_research.robot.transformations import get_point
import csv
from proper_research.robot.live_hardware_control import LiveHardwareController
mag_params = default_magnet_params()
beam_params = default_beam_params()
L_MAG = 0.04

from dataclasses import dataclass

def build_Pomega_world(p_seq, dt, Np, m=7):
    """
    Map stacked U -> stacked delta-theta in WORLD, using p_seq[j] orientation.
    Returns: (3Np, mNp)
    """
    Nu = m*Np
    P = np.zeros((3*Np, Nu), float)
    for k in range(Np):
        for j in range(k+1):
            qj = np.asarray(p_seq[j][3:7], float)
            Rj = quat_wxyz_to_R(qj)  # (3,3)
            # world increment: dt * Rj * ω_body
            P[3*k:3*k+3, m*j+3:m*j+6] += dt * Rj
    return P
@dataclass
class DebugCfg:
    level: int = 1
    every: int = 1          # print every N steps
    every_heavy: int = 25   # heavy prints every N steps
    tol_pred1: float = 5e-3 # m: acceptable one-step prediction mismatch
    tol_q_scale: float = 1e-12
    max_horizon_print: int = 4

DBG = DebugCfg(level=2, every=1, every_heavy=25)

def dbg_print(level, *args, **kwargs): 
    if DBG.level >= level:
        print(*args, **kwargs)



def sigmoid(z):
    z = float(z)
    if z >= 50:
        return 1.0
    if z <= -50:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))

def centerline_tangent(C, i):
    C = np.asarray(C, float)
    M = C.shape[0]
    i0 = int(np.clip(i, 0, M-2))
    return unit(C[i0+1] - C[i0])

def forward_tangent_indexed(C, i, look=3):
    """
    Forward tangent = direction of increasing centerline index.
    Uses lookahead for stability but NEVER flips sign based on t_prev.
    """
    C = np.asarray(C, float)
    M = C.shape[0]
    i = int(np.clip(i, 0, M-2))

    j = min(i + max(1, int(look)), M-1)
    t = C[j] - C[i]
    n = np.linalg.norm(t)
    if n < 1e-12:
        # fallback to immediate segment
        t = C[min(i+1, M-1)] - C[i]
        n = np.linalg.norm(t)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0], float)
    return t / n
def forward_tangent_smooth(C, i, t_prev=None, look=3):
    t_raw = forward_tangent_indexed(C, i, look=1)   # true forward
    t = forward_tangent_indexed(C, i, look=look)    # smoothed forward

    # Anchor sign to the true forward direction
    if float(np.dot(t, t_raw)) < 0.0:
        t = -t

    # Optional: keep close to previous without sign flips
    if t_prev is not None and float(np.dot(t, t_prev)) < -0.95:
        # near-opposite due to noise; fall back to raw forward
        t = t_raw

    return t



def unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def closest_index_in_window_monotone(C, x, i_last, window=120):
    """
    Find the closest centerline index to x, but only search forward
    from i_last to preserve monotonic progress.
    """
    C = np.asarray(C, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(3,)

    i0 = max(0, int(i_last))
    i1 = min(len(C), i0 + int(window))

    if i0 >= i1:
        return min(max(i0, 0), len(C) - 1)

    seg = C[i0:i1]
    d2 = np.sum((seg - x[None, :]) ** 2, axis=1)
    j = int(np.argmin(d2))
    return i0 + j


def centerline_tangent_at_index(C, i):
    """
    Local centerline tangent using finite differences.
    """
    C = np.asarray(C, dtype=np.float64)
    M = len(C)

    i = int(np.clip(i, 0, M - 1))

    if M < 2:
        raise ValueError("Centerline must contain at least 2 points.")

    if i == 0:
        t = C[1] - C[0]
    elif i == M - 1:
        t = C[M - 1] - C[M - 2]
    else:
        t = C[i + 1] - C[i - 1]

    return unit(t)


def wall_margin_and_gate_from_lumen(C, R, x, i, delta=5e-4, sigma_m=5e-4):
    """
    Compute margin and soft wall-risk gate from lumen centerline/radius.

    margin = R(i) - ||x - C(i)|| - delta

    Positive margin  -> inside with safety buffer
    Zero margin      -> on safety boundary
    Negative margin  -> risky / outside buffer

    gate g in [0,1]:
      ~0 when comfortably safe
      ~1 near / beyond wall
    """
    C = np.asarray(C, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(3,)

    i = int(np.clip(i, 0, len(C) - 1))

    ci = C[i]
    ri = float(R[i])

    rho = float(np.linalg.norm(x - ci))
    margin = ri - rho - float(delta)

    # logistic gate: higher when margin is small/negative
    g = 1.0 / (1.0 + np.exp(margin / max(float(sigma_m), 1e-9)))

    tv = centerline_tangent_at_index(C, i)
    return float(margin), float(g), tv


def predictive_risk_along_horizon_from_vision(
    Y_seq,            # (Np,n), expects n>=6: [x(3), t(3), ...]
    lumen_C,          # (M,3) centerline from vision
    lumen_R,          # (M,)  radius from vision
    i_ref,            # current cursor index
    window=120,
    delta=5e-4,
    sigma_m=5e-4,
    theta_crit_deg=40.0,
):
    """
    Predictive risk evaluation using lumen geometry built from vision.

    For each predicted step k:
      - map predicted tip position x_k to nearest forward centerline index i_k
      - compute local wall margin from lumen radius
      - compute soft wall gate g_k
      - compute angle theta_k between predicted tip tangent and centerline tangent

    Returns:
        {
            "idx_k": ...,
            "margin_k": ...,
            "g_k": ...,
            "theta_deg_k": ...,
            "t_vessel_k": ...,
            "theta_crit_deg": ...
        }
    """
    C = np.asarray(lumen_C, dtype=np.float64)
    R = np.asarray(lumen_R, dtype=np.float64).reshape(-1)
    Y_seq = np.asarray(Y_seq, dtype=np.float64)

    if Y_seq.ndim != 2:
        raise ValueError("Y_seq must be 2D, shape (Np, n).")
    if Y_seq.shape[1] < 6:
        raise ValueError("Y_seq must have at least 6 columns: [x(3), t(3)].")
    if len(C) != len(R):
        raise ValueError("lumen_C and lumen_R must have matching lengths.")
    if len(C) < 2:
        raise ValueError("lumen_C must contain at least 2 points.")

    Np = Y_seq.shape[0]

    idx_k = np.zeros(Np, dtype=int)
    margin_k = np.zeros(Np, dtype=np.float64)
    g_k = np.zeros(Np, dtype=np.float64)
    theta_deg_k = np.zeros(Np, dtype=np.float64)
    t_vessel_k = np.zeros((Np, 3), dtype=np.float64)

    i_last = int(np.clip(i_ref, 0, len(C) - 1))

    for k in range(Np):
        xk = np.asarray(Y_seq[k, 0:3], dtype=np.float64)
        tk = unit(Y_seq[k, 3:6])

        ik = closest_index_in_window_monotone(C, xk, i_last, window=window)
        i_last = ik

        margin, g, tv = wall_margin_and_gate_from_lumen(
            C=C,
            R=R,
            x=xk,
            i=ik,
            delta=delta,
            sigma_m=sigma_m,
        )

        cang = float(np.clip(np.dot(unit(tk), unit(tv)), -1.0, 1.0))
        theta_deg = float(np.degrees(np.arccos(cang)))

        idx_k[k] = ik
        margin_k[k] = margin
        g_k[k] = g
        theta_deg_k[k] = theta_deg
        t_vessel_k[k, :] = tv

    return {
        "idx_k": idx_k,
        "margin_k": margin_k,
        "g_k": g_k,
        "theta_deg_k": theta_deg_k,
        "t_vessel_k": t_vessel_k,
        "theta_crit_deg": float(theta_crit_deg),
    }
def forward_y_live(p8):
    p7 = pose8_quat_to_pose7_rotvec(p8)
    _ = forward_model(p7)

    x_tip = np.asarray(forward_model.last_tip, float).reshape(3,) \
        if getattr(forward_model, "last_tip", None) is not None else np.asarray(forward_model(p7), float).reshape(3,)

    C = forward_model.last_p_centerline  # (3,N) or (N,3)

    # expose centerline on the callable for MPC
    forward_y_live.last_p_centerline = None if C is None else np.asarray(C, float).copy()

    if C is None:
        t_tip = np.array([1.0, 0.0, 0.0], float)
        print("Centerline is not found")
    else:
        C = np.asarray(C, float)
        if C.shape[0] == 3:
            p_end  = C[:, -1]
            p_prev = C[:, -2] if C.shape[1] >= 2 else C[:, -1]
        else:
            p_end  = C[-1, :]
            p_prev = C[-2, :] if C.shape[0] >= 2 else C[-1, :]
        t_tip = unit(p_end - p_prev)
        if np.linalg.norm(t_tip) < 1e-9:
            t_tip = np.array([1.0, 0.0, 0.0], float)

    return np.hstack([x_tip, t_tip])
forward_y_live.last_p_centerline = None

def pose7_rotvec_to_pose8_quat(p7):
    p7 = np.asarray(p7, float).ravel()
    t = p7[0:3]
    rvec = p7[3:6]
    L = p7[6]
    q = rotvec_to_quat_wxyz(rvec)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3], L], float)

def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)

def integrate_pose8_body(p8, u7, dt):
    """
    p8 = [x,y,z, qw,qx,qy,qz, L]
    u7 = [vx,vy,vz, wx,wy,wz, dL]  (wx..wz in BODY frame)
    """
    p8 = np.asarray(p8, float).copy()
    u7 = np.asarray(u7, float).ravel()

    # translation + length
    p8[0:3] += dt * u7[0:3]
    p8[7]   += dt * u7[6]

    # quaternion update: q_new = q ⊗ δq(dt*ω_body)
    q = quat_wxyz_normalize(p8[3:7])
    dphi = dt * u7[3:6]
    dq = small_rot_quat_wxyz(dphi)
    q_new = quat_wxyz_mul(q, dq)         # body-frame increment
    p8[3:7] = quat_wxyz_normalize(q_new)

    return p8



def seq_mat_ltv(A, B_list):
    """
    Build stacked prediction matrices for time-varying B_k (LTV system)
      x_{k+1} = A x_k + B_k u_k

    Returns:
      Mx: (Np*n, n)
      Mc: (Np*n, Np*m)

    B_list: list length Np with each B_k shape (n,m)
            where B_0 corresponds to step from x0 -> x1
    """
    B_list = [np.asarray(B) for B in B_list]
    Np = len(B_list)
    n, m = B_list[0].shape

    Mx = np.zeros((Np*n, n))
    Mc = np.zeros((Np*n, Np*m))

    A_pow = np.eye(n)

    for i in range(Np):
        # x_{i+1} = A^{i+1} x0 + sum_{j=0..i} A^{i-j} B_j u_j
        A_pow = A @ A_pow
        Mx[i*n:(i+1)*n, :] = A_pow

        for j in range(i + 1):
            A_ij = np.linalg.matrix_power(A, i - j)
            Mc[i*n:(i+1)*n, j*m:(j+1)*m] = A_ij @ B_list[j]

    return Mx, Mc

def seq_mat_lti(A, B, N):
    n, m = B.shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    A_pow = np.eye(n)

    A_pow = A @ A_pow
    Mx[0:n, :] = A_pow
    Mc[0:n, 0:m] = B

    for i in range(1, N):
        A_pow = A @ A_pow
        Mx[i*n:(i+1)*n, :] = A_pow
        Mc[i*n:(i+1)*n, 0:i*m] = A @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B

    return Mx, Mc


def solve_qp_osqp(H, f, A, l, u, U_warm=None):
    P = sp.csc_matrix(0.5 * (H + H.T))
    q = f.astype(float)
    A = sp.csc_matrix(A)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    if U_warm is not None:
        prob.warm_start(x=U_warm)

    res = prob.solve()
    status = res.info.status
    if status not in ("solved", "solved inaccurate"):
        return None, None, status

    return res.x, res.y, status
def build_Pm_world(dt, Np, m=7):
    # r_stack = r0_stack + Pm U
    S = np.tril(np.ones((Np, Np))) * dt               # (Np,Np)
    Pm = np.kron(S, np.hstack([np.eye(3), np.zeros((3, m-3))]))  # (3Np, mNp)
    return Pm

def project_to_polyline_s_monotone(C, s_path, x, i_start, window=120):
    """
    Project point x onto polyline segments [i_start .. i_start+window) and return arc-length s_hat.
    C: (M,3)
    s_path: (M,) cumulative arc length at vertices (same length as C)
    x: (3,)
    i_start: monotone search start index (vertex index)
    window: number of vertices ahead to search (segments are within that)
    Returns: (s_hat, i_seg, lam, d2_best)
    """
    C = np.asarray(C, float)
    s_path = np.asarray(s_path, float).ravel()
    x = np.asarray(x, float).reshape(3,)

    M = C.shape[0]
    i0 = int(np.clip(i_start, 0, M-2))          # segment start index
    i1 = int(min(M-1, i0 + int(window)))        # vertex end bound
    if i1 <= i0:
        return float(s_path[i0]), i0, 0.0, float("inf")

    best_d2 = np.inf
    best_s  = float(s_path[i0])
    best_i  = i0
    best_l  = 0.0

    # iterate segments i..i+1
    for i in range(i0, i1-1):
        a = C[i]
        b = C[i+1]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-16:
            continue

        lam = float(((x - a) @ ab) / L2)
        lam = 0.0 if lam < 0.0 else (1.0 if lam > 1.0 else lam)
        p = a + lam * ab
        d = x - p
        d2 = float(d @ d)

        if d2 < best_d2:
            best_d2 = d2
            seg_len = float(np.sqrt(L2))
            best_s  = float(s_path[i] + lam * seg_len)
            best_i  = int(i)
            best_l  = float(lam)

    return best_s, best_i, best_l, best_d2
def pick_dipole_axis_forward(d_body_nominal, q_wxyz, t_forward):
    d_body_nominal = np.asarray(d_body_nominal, float).reshape(3,)
    d_body_nominal /= (np.linalg.norm(d_body_nominal) + 1e-12)

    Rk = quat_wxyz_to_R(np.asarray(q_wxyz, float).reshape(4,))
    dk = Rk @ d_body_nominal
    dk /= (np.linalg.norm(dk) + 1e-12)

    t_forward = np.asarray(t_forward, float).reshape(3,)
    t_forward /= (np.linalg.norm(t_forward) + 1e-12)

    # If the dipole points backwards relative to "forward tangent", flip the BODY axis
    if float(np.dot(dk, t_forward)) < 0.0:
        return -d_body_nominal
    return d_body_nominal
def forward_tangent(Cc, i, t_prev=None):
    t = centerline_tangent(Cc, int(i))
    t = t / (np.linalg.norm(t) + 1e-12)
    if t_prev is not None and np.dot(t, t_prev) < 0.0:
        t = -t
    return t
def plot_reference_debug_simple(mpc, x_meas, n_ref=10):
    C = np.asarray(mpc.lumen_C, float)
    i_ref = int(getattr(mpc, "i_ref_last", 0))
    idx_ref = np.clip(i_ref + np.arange(n_ref), 0, len(C) - 1)

    plt.figure(figsize=(7, 7))
    plt.plot(C[:, 0], C[:, 1], "-", label="lumen")
    plt.plot(x_meas[0], x_meas[1], "o", label="tip")
    plt.plot(C[i_ref, 0], C[i_ref, 1], "s", label="i_ref")
    plt.plot(C[idx_ref, 0], C[idx_ref, 1], "x--", label="future ref")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Reference points on lumen")
    # plt.show()

import os
import matplotlib.pyplot as plt
import numpy as np

def plot_reference_debug(
    mpc,
    x_meas,
    info=None,
    n_ref=10,
    show_boundaries=True,
    save_path=None,
    mag_pos=None,
    mag_dir=None,
    mag_pos_next=None,
    mag_dir_next=None,
):
    C = np.asarray(mpc.lumen_C, float)
    R = np.asarray(mpc.lumen_R, float).reshape(-1) if getattr(mpc, "lumen_R", None) is not None else None

    i_ref = int(getattr(mpc, "i_ref_last", 0))
    idx_ref = np.clip(i_ref + np.arange(n_ref), 0, len(C) - 1)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(C[:, 0], C[:, 1], "-", label="lumen centerline")

    if show_boundaries and (R is not None) and (len(C) >= 2):
        left_bd = np.zeros_like(C[:, :2])
        right_bd = np.zeros_like(C[:, :2])

        for i in range(len(C)):
            if i == 0:
                t = C[1, :2] - C[0, :2]
            elif i == len(C) - 1:
                t = C[-1, :2] - C[-2, :2]
            else:
                t = C[i + 1, :2] - C[i - 1, :2]

            nt = np.linalg.norm(t)
            if nt < 1e-12:
                n = np.array([0.0, 0.0])
            else:
                t = t / nt
                n = np.array([-t[1], t[0]])

            left_bd[i] = C[i, :2] + R[i] * n
            right_bd[i] = C[i, :2] - R[i] * n

        ax.plot(left_bd[:, 0], left_bd[:, 1], "--", label="lumen boundary +R")
        ax.plot(right_bd[:, 0], right_bd[:, 1], "--", label="lumen boundary -R")

    ax.plot(x_meas[0], x_meas[1], "o", markersize=8, label="measured tip")
    ax.plot(C[i_ref, 0], C[i_ref, 1], "s", markersize=8, label="i_ref")
    ax.plot(C[idx_ref, 0], C[idx_ref, 1], "x--", label="future ref")

    if len(x_meas) >= 6:
        ax.arrow(
            x_meas[0], x_meas[1],
            0.005 * x_meas[3], 0.005 * x_meas[4],
            head_width=0.0001, length_includes_head=True
        )

    if info is not None:
        X_pred = info.get("X_pred", None)
        if X_pred is not None:
            X_pred = np.asarray(X_pred, float)
            if X_pred.ndim == 2 and X_pred.shape[1] >= 2 and np.all(np.isfinite(X_pred[:, :2])):
                ax.plot(X_pred[:, 0], X_pred[:, 1], "o-.", label="predicted tip")
                ax.plot(X_pred[0, 0], X_pred[0, 1], "D", markersize=8, label="predicted tip[0]")
                if X_pred.shape[1] >= 6:
                    ax.arrow(
                        X_pred[0, 0], X_pred[0, 1],
                        0.005 * X_pred[0, 3], 0.005 * X_pred[0, 4],
                        head_width=0.0001, length_includes_head=True
                    )

        X_nom_last = info.get("X_nom_last", None)
        if X_nom_last is not None:
            X_nom_last = np.asarray(X_nom_last, float)
            if X_nom_last.ndim == 2 and X_nom_last.shape[1] >= 2 and np.all(np.isfinite(X_nom_last[:, :2])):
                ax.plot(X_nom_last[:, 0], X_nom_last[:, 1], ":+", label="nominal tip")
    # current external magnet
    if mag_pos is not None:
        mag_pos = np.asarray(mag_pos, float).reshape(3,)
        ax.plot(mag_pos[0], mag_pos[1], "md", markersize=9, label="magnet current")

        if mag_dir is not None:
            mag_dir = np.asarray(mag_dir, float).reshape(3,)
            nrm = np.linalg.norm(mag_dir[:2])
            if nrm > 1e-12:
                dxy = mag_dir[:2] / nrm
                arrow_len = 0.01  # 10 mm in plot units [m]
                ax.arrow(
                    mag_pos[0], mag_pos[1],
                    arrow_len * dxy[0], arrow_len * dxy[1],
                    head_width=0.0015,
                    head_length=0.0025,
                    length_includes_head=True,
                    color="m",
                )

    # next commanded external magnet
    if mag_pos_next is not None:
        mag_pos_next = np.asarray(mag_pos_next, float).reshape(3,)
        ax.plot(mag_pos_next[0], mag_pos_next[1], "cs", markersize=8, label="magnet next")

        if mag_dir_next is not None:
            mag_dir_next = np.asarray(mag_dir_next, float).reshape(3,)
            nrm = np.linalg.norm(mag_dir_next[:2])
            if nrm > 1e-12:
                dxy = mag_dir_next[:2] / nrm
                arrow_len = 0.01
                ax.arrow(
                    mag_pos_next[0], mag_pos_next[1],
                    arrow_len * dxy[0], arrow_len * dxy[1],
                    head_width=0.0015,
                    head_length=0.0025,
                    length_includes_head=True,
                    color="c",
                )
    print("tip norm:", np.linalg.norm(x_meas[:3]))
    print("lumen norm:", np.linalg.norm(mpc.lumen_C[0]))
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Reference, lumen, and MPC prediction")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[SAVE] plot saved to {save_path}")

    # plt.show()
    # plt.close(fig)


from scipy.spatial.transform import Rotation as Rot
import numpy as np

def dipole_dir_from_p8(p8):
    p8 = np.asarray(p8, float).ravel()
    q_wxyz = p8[3:7]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], float)
    R = Rot.from_quat(q_xyzw).as_matrix()

    # body dipole axis of the external magnet
    m_body = np.array([1.0, 0.0, 0.0], float)
    m_world = R @ m_body
    return m_world / (np.linalg.norm(m_world) + 1e-12)
def dipole_dir_from_pose6(pose6):
    pose6 = np.asarray(pose6, float).ravel()
    R = Rot.from_rotvec(pose6[3:6]).as_matrix()
    m_body = np.array([1.0, 0.0, 0.0], float)
    m_world = R @ m_body
    return m_world / (np.linalg.norm(m_world) + 1e-12)
class mpc_controller_tipxy_LTI:
    def __init__(self, *, Jxy_fn, forward_tip_fn,
                 dt=0.05, Np=10,
                 w_xy=(5.0, 5.0, 0.0),
                 w_u=None,
                 w_du=None,
                 u_max=None,
                 p_min=None,
                 p_max=None,
                 N_sqp=3,
                 n_out = 3,
                 n_p=8,
                 n_u=7,
                 model_mode = "lti",
                 d_min_tip_mag=0.10,  
                 d_alpha=0.15,
                 overhead_magnet = False,
                 w_mag = 1,
                 ):
        self.mode = "centreline_only"  # "full" (your current) or "centerline_only"
        self.np = int(n_p)     # pose dimension
        self.m  = int(n_u)     # control dimension (still 7)
        self.n = n_out
        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn
        self.x_pred_prev_1 = None
        self.dt = float(dt)
        self.Np = int(Np)

        self.A = np.eye(self.n)



        self.U_warm = None
        self.model_mode = str(model_mode).lower()
        if self.model_mode not in ("lti", "ltv"):
            raise ValueError("model_mode must be 'lti' or 'ltv'")
        self.model_mode = str(model_mode).lower()
        if self.model_mode not in ("lti", "ltv"):
            raise ValueError("model_mode must be 'lti' or 'ltv'")
        # weights defaults
        if w_u is None:
            w_u = (1e-3,) * self.m
        if w_du is None:
            w_du = (0.0,) * self.m

        w_xy = np.asarray(w_xy, float).ravel()
        w_u  = np.asarray(w_u,  float).ravel()
        w_du = np.asarray(w_du, float).ravel()

        if w_xy.size != self.n:
            raise ValueError(f"w_xy must have length {self.n}, got {w_xy.size}")
        if w_u.size != self.m:
            raise ValueError(f"w_u must have length {self.m}, got {w_u.size}")
        if w_du.size != self.m:
            raise ValueError(f"w_du must have length {self.m}, got {w_du.size}")

        self.Q  = np.diag(w_xy)
        self.R  = np.diag(w_u)
        self.Rd = np.diag(w_du)

        if u_max is None:
            u_max = np.full(self.m, np.inf)
        if p_min is None:
            p_min = -np.full(self.np, np.inf)
        if p_max is None:
            p_max = +np.full(self.np, np.inf)
        self.u_max = np.asarray(u_max, float).ravel()
        if self.u_max.size != self.m:
            raise ValueError("u_max wrong length")
        self.p_min = np.asarray(p_min, float).ravel()
        self.p_max = np.asarray(p_max, float).ravel()

        if self.p_min.size != self.np: raise ValueError("p_min wrong length")
        if self.p_max.size != self.np: raise ValueError("p_max wrong length")

        self.d  = np.zeros(self.n, float)
        self.p  = None
        self.x  = None
        self.Qf = None

        self.N_sqp = int(N_sqp)
        self.d_alpha = float(d_alpha)

        # matrices
        self._rebuild_S()
        self.Du = self._build_Du_matrix()
        self.p_last_meas = None
        self.x_last_meas = None

        self.sigma_m = 5e-4           # margin softness (m)
        self.delta_wall = 1e-3        # safety margin (m)
        # --- predictive wall/tangent shaping ---
        self.risk_window = 120          # how far ahead to search on centerline for predicted mapping
        self.theta_crit_deg = 40.0
        # --- NEW: contact-based reweighting (only active near wall) ---
        self.w_adv = 3    # start tiny (1e-5 .. 1e-3)
        # self.w_adv = 0
        # baseline multipliers (keep 1.0 unless you want global scaling)
        self.w_adv_base = float(self.w_adv)
        self.w_adv_eff = float(self.w_adv)   # can be overridden per-step
        self.debug = True
        self._dbg_last = {}
        self.overhead_magnet = overhead_magnet
        # __init__

        self.s_des = 5e-4  

        self.enable_dipole_align = True
        self.w_dipole_align = 0.1        # start small: 0.1..10
        self.dipole_body_axis = np.array([1.0, 0.0, 0.0])  # or [0,0,1]

        self.enable_mag_center_standoff = True
        self.w_mag_center_standoff = 12
        self.mag_center_standoff_m = 0.25
        self.dL_back_max = 0.002      # or 0.001 if small pullback allowed
        self.dL_fwd_max  = np.inf   # or some finite cap (per-step dL rate)
        from collections import deque

    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

    def   _build_Du_matrix(self):
        Np = self.Np
        m = self.m
        if Np <= 1:
            return np.zeros((0, Np*m))

        D1 = np.zeros((Np-1, Np))
        for i in range(Np-1):
            D1[i, i]   = -1.0
            D1[i, i+1] = +1.0

        return np.kron(D1, np.eye(m))
    def set_measured_params(self, p_meas):
        p_meas = np.asarray(p_meas, float).reshape(self.np,)
        self.p = self._clamp_p(p_meas)
        self.p_last_meas = self.p.copy()

    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.np,)
        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()
            self.x = np.asarray(self.forward_tip_fn(self.p, commit=False), float).reshape(self.n,)
        else:
            self.x = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
        self.d = np.zeros(self.n, float)
        self.U_warm = None

    def _clamp_p(self, p):
        p = np.asarray(p, float).copy()

        # clamp x,y,z
        p[0:3] = np.minimum(np.maximum(p[0:3], self.p_min[0:3]), self.p_max[0:3])

        # clamp L
        p[7] = float(np.minimum(np.maximum(p[7], self.p_min[7]), self.p_max[7]))

        # normalize quaternion
        p[3:7] = quat_wxyz_normalize(p[3:7])
        return p

    def _p_seq_from_U(self, p0, U_seq):
        p_running = p0.copy()
        p_list = []
        for i in range(self.Np):
            p_running = integrate_pose8_body(p_running, U_seq[i], self.dt)
            p_running = self._clamp_p(p_running)
            p_list.append(p_running.copy())
        return np.array(p_list)



    def _build_prediction_mats(self, p0, U_guess):
        t0 = time.perf_counter()

        n, m, Np = self.n, self.m, self.Np

        if self.model_mode == "lti":
            tJ0 = time.perf_counter()
            J0 = np.asarray(self.Jxy_fn(p0), float)
            tJ1 = time.perf_counter()

            B0 = J0

            tM0 = time.perf_counter()
            Mx, Mc = seq_mat_lti(self.A, B0, Np)
            tM1 = time.perf_counter()

            if U_guess is None:
                U_guess = np.zeros((Np, m))

            tp0 = time.perf_counter()
            p_seq = self._p_seq_from_U(p0, U_guess)
            tp1 = time.perf_counter()

            t1 = time.perf_counter()
            print(f"[TIME] Jxy_fn: {(tJ1-tJ0)*1e3:.2f} ms")
            print(f"[TIME] seq_mat_lti: {(tM1-tM0)*1e3:.2f} ms")
            print(f"[TIME] _p_seq_from_U: {(tp1-tp0)*1e3:.2f} ms")
            print(f"[TIME] total build_prediction_mats: {(t1-t0)*1e3:.2f} ms")

            return p_seq, Mx, Mc, B0

        if U_guess is None:
            U_guess = np.zeros((Np, m))

        p_seq = self._p_seq_from_U(p0, U_guess)  
        B_list = []
        for i in range(Np):
            Ji = np.asarray(self.Jxy_fn(p_seq[i]), float)  # Ji is B_i directly
            B_list.append(Ji)

        Mx, Mc = seq_mat_ltv(self.A, B_list)
        B0 = B_list[0]
        return p_seq, Mx, Mc, B0

    def step(self, x_meas=None):
        """
        MPC step for real-camera-in-the-loop operation.

        What stays model-based:
        - Jacobian / prediction matrices
        - nonlinear rollout for affine matching
        - predictive risk / penalties / constraints
        - QP solve

        What is no longer simulation-based:
        - the post-solve state update does NOT use forward_tip_fn as plant truth
        - self.x is corrected from the measured camera state x_meas
        - prediction error is computed against the measured next state

        Assumptions:
        - x_meas is the measured output at the current control instant, shape (n,)
        - outputs y are typically [x,y,z,tx,ty,tz]
        - self.p is still an internal actuator/configuration estimate
        """
        print("ENTER step")
        print("model_mode:", self.model_mode)
        print("x_meas is None:", x_meas is None)
        print("self.p is None:", self.p is None)
        def _pos_row_idx(n, Np):
            return np.array([k * n + i for k in range(Np) for i in (0, 1, 2)], dtype=int)

        def _pad_A(Au, nz, Nu):
            Au = np.asarray(Au, float)
            if nz == Nu:
                return Au
            nr = Au.shape[0]
            Az = np.zeros((nr, nz), float)
            Az[:, :Nu] = Au
            return Az

        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        if x_meas is None:
            raise ValueError("step(x_meas=...) now requires a real measured state.")

        # freeze forward baseline once per MPC step for model prediction consistency
        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()

        p_prev = self.p.copy()
        x_prev = self.x.copy() if self.x is not None else np.asarray(x_meas, dtype=float).reshape(self.n,)

        # measured state is truth
        self.x = np.asarray(x_meas, dtype=float).reshape(self.n,)

        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        xk = self.x.reshape(n, 1)

        centerline_only = (getattr(self, "mode", "full") == "centerline_only")

        enable_adv_eff = ((self.w_adv != 0.0) and (not centerline_only))
        enable_standoff_eff = self.enable_mag_center_standoff
        enable_inline_eff = bool(getattr(self, "enable_mag_tangent_inline", True))
        enable_dipole_eff = self.enable_dipole_align

        # warm start
        if self.U_warm is not None and self.U_warm.size == Np * m:
            U_opt_vec = self.U_warm.copy()
            U_guess = U_opt_vec.reshape(Np, m)
        else:
            U_guess = np.zeros((Np, m))
            U_opt_vec = None

        status_last = "init"

        Mc_last = None
        X_aff_last = None
        X_nom_last = None
        p_seq_last = None
        B_first = None
        p_lin = None
        p_first = None
        dbg_terms = {} if self.debug else None

        # defaults for outputs/debug
        H_effort = None
        H_smooth = None
        H_track = None
        f_track = None
        H_z = None
        f_z = None
        Z_opt = None
        tan1_err = np.nan

        for it in range(int(self.N_sqp)):
            p0 = self.p.copy()
            print(f"P0 is {p0}")
            print("[DBG] Jxy_fn(p0) shape =", np.asarray(self.Jxy_fn(p0)).shape)
            print("[DBG] expected shape =", (self.n, self.m))

            # Evaluate wrapped forward model once at p0
            y0 = np.asarray(self.forward_tip_fn(p0, commit=False), float).reshape(-1)
            tip = y0[:3]

            # Get raw centerline from underlying forward model
            pcl = getattr(self.forward_tip_fn.fwd, "last_p_centerline", None)
            if pcl is None:
                print("[DBG] last_p_centerline is None")
            else:
                pcl = np.asarray(pcl, float)

                print("y0 =", y0)
                print("tip =", tip)
                print("pcl shape =", pcl.shape)

                if pcl.ndim != 2:
                    print("[DBG] unexpected pcl ndim:", pcl.ndim)
                else:
                    # Handle either (3,N) or (N,3)
                    if pcl.shape[0] == 3:
                        p_first = pcl[:, 0]
                        p_last  = pcl[:, -1]
                        dist_first = np.linalg.norm(tip - p_first)
                        dist_last  = np.linalg.norm(tip - p_last)

                        print("pcl first =", p_first)
                        print("pcl last  =", p_last)
                        print("dist tip to first =", dist_first)
                        print("dist tip to last  =", dist_last)

                        if dist_first < dist_last:
                            print("[DBG] centerline appears ordered TIP->BASE")
                        else:
                            print("[DBG] centerline appears ordered BASE->TIP")

                    elif pcl.shape[1] == 3:
                        p_first = pcl[0, :]
                        p_last  = pcl[-1, :]
                        dist_first = np.linalg.norm(tip - p_first)
                        dist_last  = np.linalg.norm(tip - p_last)

                        print("pcl first =", p_first)
                        print("pcl last  =", p_last)
                        print("dist tip to first =", dist_first)
                        print("dist tip to last  =", dist_last)

                        if dist_first < dist_last:
                            print("[DBG] centerline appears ordered TIP->BASE")
                        else:
                            print("[DBG] centerline appears ordered BASE->TIP")

                    else:
                        print("[DBG] unexpected pcl shape:", pcl.shape)
            
            p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)
            print("built prediction mats")
            print("p_seq shape:", np.shape(p_seq))
            print("Mx shape:", np.shape(Mx))
            print("Mc shape:", np.shape(Mc))
            print("B0 shape:", np.shape(B0))
            B_first = np.asarray(B0, float).copy()
            x_pred_prev_1 = getattr(self, "x_pred_prev_1", None)

            pred1_err_xy = np.nan
            pred1_err_xyz = np.nan

            if x_pred_prev_1 is not None:
                e = x_meas[:3] - x_pred_prev_1[:3]
                pred1_err_xy = np.linalg.norm(e[:2])
                pred1_err_xyz = np.linalg.norm(e[:3])
            # debug Jacobian info
            B = np.asarray(B0, float)
            _, mB = B.shape
            umax = np.asarray(self.u_max, float)
            delta_max = np.sum(np.abs(B) * umax.reshape(1, -1), axis=1)

            print("max |Δx| per step (mm):", 1e3 * delta_max[0])
            print("max |Δy| per step (mm):", 1e3 * delta_max[1])
            print("max |Δz| per step (mm):", 1e3 * delta_max[2])

            col_names = ["vx", "vy", "vz", "wx", "wy", "wz", "dL"]
            row_names = [f"y{i}" for i in range(n)]
            if n >= 6:
                row_names = ["x", "y", "z", "tx", "ty", "tz"] + [f"y{i}" for i in range(6, n)]

            print(f"\n[DBG] SQP it={it}  B0 shape = {B.shape}")
            print("[DBG] B0 column norms:")
            for j in range(mB):
                print(f"  {col_names[j]:>2s}: {np.linalg.norm(B[:, j]):.3e}")

            print("[DBG] B0 rows (scientific):")
            for i in range(n):
                vals = " ".join([f"{B[i, j]:+10.3e}" for j in range(mB)])
                print(f"  {row_names[i]:>3s}: {vals}")
            print("       " + " ".join([f"{c:>10s}" for c in col_names]))

            Bt = B[:, 0:3]
            Bw = B[:, 3:6]
            BdL = B[:, 6:7]

            print("[DBG] authority norms:")
            print(f"  ||B_trans||_F = {np.linalg.norm(Bt, 'fro'):.3e}")
            print(f"  ||B_omega||_F = {np.linalg.norm(Bw, 'fro'):.3e}")
            print(f"  ||B_dL||_F    = {np.linalg.norm(BdL, 'fro'):.3e}")

            eps = 1e-12
            print("[DBG] ratios:")
            print(f"  trans/omega = {np.linalg.norm(Bt, 'fro') / (np.linalg.norm(Bw, 'fro') + eps):.3f}")
            print(f"  dL/omega    = {np.linalg.norm(BdL, 'fro') / (np.linalg.norm(Bw, 'fro') + eps):.3f}")

            U_svd, S, Vt = np.linalg.svd(B, full_matrices=False)
            cond = (S[0] / max(S[-1], 1e-16)) if S.size else np.inf
            rank = int(np.sum(S > 1e-10))

            self._jac_svd_last = dict(
                S=S.copy(),
                cond=float(cond),
                rank=int(rank),
                trans_norm=float(np.linalg.norm(B[:, 0:3], 'fro')),
                omega_norm=float(np.linalg.norm(B[:, 3:6], 'fro')),
                dL_norm=float(np.linalg.norm(B[:, 6:7], 'fro')),
            )

            print("[DBG] svd singular values:", np.array2string(S, precision=3, suppress_small=False))
            print(f"[DBG] rank≈{rank}/{min(B.shape)}   cond≈{cond:.3e}")

            print("[DBG] top input directions (Vt rows, mapped to [vx vy vz wx wy wz dL]):")
            for k_show in range(min(3, Vt.shape[0])):
                v = Vt[k_show, :]
                print(f"  mode{k_show}: " + " ".join([f"{col_names[j]}:{v[j]:+0.3f}" for j in range(mB)]))

            if n >= 6:
                print("[DBG] top output directions (U cols -> [x y z tx ty tz]):")
                names = ["x", "y", "z", "tx", "ty", "tz"]
                for k_show in range(min(3, U_svd.shape[1])):
                    uk = U_svd[:, k_show]
                    print("  out_mode%d:" % k_show, " ".join([f"{names[i]}:{uk[i]:+0.3f}" for i in range(6)]))

            # model rollout for prediction only
            p_lin = p_prev.copy()
            p_first = p_seq[0].copy()

            Y_nom = np.vstack([
                self.forward_tip_fn(p_seq[i], commit=False) for i in range(Np)
            ]).reshape(Np, n)
            X_nom = Y_nom.reshape(Np * n, 1)

            risk = predictive_risk_along_horizon_from_vision(
                Y_seq=Y_nom,
                lumen_C=self.lumen_C,
                lumen_R=self.lumen_R,
                i_ref=int(getattr(self, "i_ref_last", 0)),
                window=int(self.risk_window),
                delta=float(self.delta_wall),
                sigma_m=float(self.sigma_m),
                theta_crit_deg=float(self.theta_crit_deg),
            )
            self._risk_last = risk

            Cc = np.asarray(self.lumen_C, float)
            M = Cc.shape[0]

            idx_k_risk = np.asarray(risk.get("idx_k", np.zeros(Np, dtype=int)), dtype=int).reshape(-1)
            if idx_k_risk.size != Np:
                i_ref0 = int(getattr(self, "i_ref_last", 0))
                idx_k_risk = i_ref0 + np.arange(Np)
            idx_k_risk = np.clip(idx_k_risk, 0, M - 1)

            idx_shift = int(getattr(self, "idx_ahead", 3))
            idx_k_epm = np.clip(idx_k_risk + idx_shift, 0, M - 1)

            theta_seq = np.asarray(risk["theta_deg_k"], float)
            g_seq = np.asarray(risk["g_k"], float)
            t_vessel = np.asarray(risk["t_vessel_k"], float)

            g_wall = np.asarray(g_seq, float).reshape(Np,)
            theta = np.asarray(theta_seq, float).reshape(Np,)

            theta_crit = float(self.theta_crit_deg)
            theta_band = float(getattr(self, "theta_gate_band_deg", 5.0))
            g_theta = 1.0 / (1.0 + np.exp(-(theta - theta_crit) / max(theta_band, 1e-6)))
            g_unsafe0 = float(g_wall[0] * g_theta[0])

            Nu = Np * m
            U_guess_vec = U_guess.reshape(-1, 1)
            Rtil = np.kron(np.eye(Np), self.R)

            if Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np - 1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            H_effort = 2.0 * Rtil
            H_smooth = 2.0 * H_du if not np.isscalar(H_du) else None
            # H = H_effort.copy()
            f = np.zeros((Np * m, 1), float)

            if dbg_terms is not None:
                zero_f = np.zeros((Np * m, 1), float)
                dbg_terms["effort"] = (H_effort.copy(), zero_f.copy())
                if H_smooth is not None:
                    dbg_terms["smooth"] = (H_smooth.copy(), zero_f.copy())

            use_affine_matching = bool(getattr(self, "use_affine_matching", True))
            if use_affine_matching:
                X_aff = X_nom - Mc @ U_guess_vec
            else:
                X_aff = (Mx @ xk).reshape(Np * n, 1)

            i0 = int(getattr(self, "i_ref_last", 0))
            look = int(getattr(self, "ref_lookahead_pts", 8))
            idx_ref = np.clip(i0 + look + np.arange(Np), 0, M - 1)
            print(f"REFERENCE {idx_ref}")

            X_ref = np.zeros((Np * n, 1), float)
            for k in range(Np):
                X_ref[k * n + 0, 0] = Cc[idx_ref[k], 0]
                X_ref[k * n + 1, 0] = Cc[idx_ref[k], 1]
                X_ref[k * n + 2, 0] = Cc[idx_ref[k], 2]
            X_aff_fixed = X_ref.copy()

            for k in range(Np):
                sl = slice(k * n, (k + 1) * n)
                X_aff_fixed[sl.start + 0, 0] = X_aff[sl.start + 0, 0]
                X_aff_fixed[sl.start + 1, 0] = X_aff[sl.start + 1, 0]
            Qtil = np.kron(np.eye(Np), self.Q)

            Mc_last = Mc
            X_aff_last = X_aff
            X_nom_last = X_nom
            p_seq_last = p_seq

            H_track = 2.0 * (Mc.T @ Qtil @ Mc)
            f_track = 2.0 * (Mc.T @ Qtil @ (X_aff_fixed - X_ref))
            H = H_track.copy()
            f += f_track
            print(f"x_aff is: {X_aff_fixed}")
            print(f"x_ref is: {X_ref}")
            if dbg_terms is not None:
                dbg_terms["track"] = (H_track.copy(), f_track.copy())
                dbg_terms["track_model"] = dict(
                    Mc=Mc.copy(),
                    X_aff=X_aff.copy(),
                    X_ref=X_ref.copy(),
                    Qtil=Qtil.copy(),
                    Q_diag=np.diag(self.Q).copy(),
                    idx_ref=np.asarray(idx_ref, int).copy(),
                )

            if t_vessel is not None and enable_adv_eff:
                t_v = np.asarray(t_vessel, float).copy()
                for k in range(1, Np):
                    if np.dot(t_v[k], t_v[k - 1]) < 0.0:
                        t_v[k] *= -1.0

                t0 = forward_tangent(Cc, int(idx_k_epm[0]), None)
                if np.dot(t_v[0], t0) < 0.0:
                    t_v[0] *= -1.0

                t_delta = np.zeros((3 * Np, 1))
                for k in range(Np):
                    t_delta[3 * k:3 * k + 3, 0] = t_v[k]

                D1_inc = np.zeros((Np, Np))
                D1_inc[0, 0] = +1.0
                for k in range(1, Np):
                    D1_inc[k, k - 1] = -1.0
                    D1_inc[k, k] = +1.0
                Dpos = np.kron(D1_inc, np.eye(3))

                idx_pos = _pos_row_idx(n, Np)
                Mc_pos = Mc[idx_pos, :]
                g_adv = Mc_pos.T @ (Dpos.T @ t_delta)

                t0 = unit(t_v[0])
                tt = unit(xk[3:6, 0])
                theta0 = np.degrees(np.arccos(np.clip(float(np.dot(t0, tt)), -1.0, 1.0)))

                beta = 0.9
                w_adv_eff = float(getattr(self, "w_adv_eff", self.w_adv)) * (1.0 - beta * g_unsafe0)
                s_des_eff = max(self.s_des * (1.0 - beta * g_unsafe0), 0.1 * self.s_des)
                s_des_eff_local = s_des_eff

                f_adv = -w_adv_eff * g_adv
                f = f + f_adv

                if dbg_terms is not None:
                    dbg_terms["advance"] = (np.zeros((Np * m, Np * m)), f_adv.copy())
                    dbg_terms["advance_model"] = dict(
                        g_adv=g_adv.copy(),
                        t_v=t_v.copy(),
                        w_adv_eff=float(w_adv_eff),
                        s_des_eff=float(s_des_eff),
                    )
            else:
                s_des_eff_local = self.s_des
                theta0 = np.nan
                w_adv_eff = 0.0

            enable_tangent_pen = bool(getattr(self, "enable_tangent_penalty", False)) and (t_vessel is not None)

            if enable_tangent_pen and (self.lumen_R is not None):
                print("Here")
                clearance_m_obj = np.full(Np, np.inf, float)
                Cc_tmp = np.asarray(self.lumen_C, float)
                Rr_tmp = np.asarray(self.lumen_R, float).reshape(-1,)

                for k in range(Np):
                    x_tip_k = X_nom[k * n:k * n + 3, 0].reshape(3,)
                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc_tmp[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc_tmp.shape[0] - 2))

                    rho_k = float(np.linalg.norm(x_tip_k - c_closest.reshape(3,)))
                    idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr_tmp.shape[0] - 1))
                    R_k = float(Rr_tmp[idx_v])
                    clearance_m_obj[k] = R_k - rho_k

                theta_ref_deg = float(getattr(self, "theta_ref_deg", 0.0))
                cos_ref = float(np.cos(np.deg2rad(theta_ref_deg)))

                d_gate = float(getattr(self, "tangent_gate_clearance_m", 1.5e-3))
                d_band = float(getattr(self, "tangent_gate_band_m", 0.25e-3))
                w_tan_min = float(getattr(self, "w_tangent_min", 0.1))
                w_tan_max = float(getattr(self, "w_tangent_max", 1.0))

                A_tan = np.zeros((Np, Nu), float)
                b_tan = np.zeros((Np, 1), float)
                w_tan_k = np.zeros(Np, float)

                for k in range(Np):
                    tv = np.asarray(t_vessel[k], float).reshape(3,)
                    tv /= (np.linalg.norm(tv) + 1e-12)

                    rows_t = np.array([k * n + 3, k * n + 4, k * n + 5], dtype=int)
                    Mc_tk = Mc[rows_t, :]
                    t_aff = X_aff[rows_t, 0].reshape(3,)

                    A_tan[k, :] = -(tv.reshape(1, 3) @ Mc_tk).reshape(-1)
                    b_tan[k, 0] = cos_ref - float(tv @ t_aff)

                    g_clr = 1.0 / (1.0 + np.exp((clearance_m_obj[k] - d_gate) / max(d_band, 1e-9)))
                    w_tan_k[k] = w_tan_min + (w_tan_max - w_tan_min) * g_clr

                W_tan = np.diag(w_tan_k)
                H_tan = 2.0 * (A_tan.T @ W_tan @ A_tan)
                f_tan = 2.0 * (A_tan.T @ W_tan @ b_tan)

                # H += H_tan
                # f += f_tan

                if dbg_terms is not None:
                    dbg_terms["tangent"] = (H_tan.copy(), f_tan.copy())
                    dbg_terms["tangent_clearance_m"] = clearance_m_obj.copy()
                    dbg_terms["tangent_weights"] = w_tan_k.copy()
                    dbg_terms["tangent_model"] = dict(
                        A=A_tan.copy(),
                        b=b_tan.copy(),
                        cos_ref=float(cos_ref),
                        theta_ref_deg=float(theta_ref_deg),
                    )
            else:
                theta_ref_deg = np.nan
                w_tan_min = 0.0
                w_tan_max = 0.0
                clearance_m_obj = np.full(Np, np.nan)

            enable_standoff_soft = bool(getattr(self, "enable_standoff_soft", enable_standoff_eff))
            enable_inline_soft = bool(getattr(self, "enable_inline_soft", enable_inline_eff))
            enable_dipole_soft = bool(getattr(self, "enable_dipole_soft", enable_dipole_eff)) and (t_vessel is not None)

            need_epm_pos_model = enable_standoff_soft or enable_inline_soft
            U_guess_vec = U_guess.reshape(-1, 1)

            if need_epm_pos_model:
                Pm = build_Pm_world(self.dt, Np, m=m)
                r0 = p0[:3].copy().reshape(3, 1)
                r0_stack = np.tile(r0, (Np, 1))
                r_nom = r0_stack + Pm @ U_guess_vec

            if enable_standoff_soft:
                d0 = float(self.mag_center_standoff_m)
                A_s = np.zeros((Np, Nu), float)
                b_s = np.zeros((Np, 1), float)

                eps_st = 1e-9
                for k in range(Np):
                    ck = Cc[idx_k_epm[k], :3].reshape(3,)
                    rk = r_nom[3 * k:3 * k + 3, 0]
                    vk = rk - ck
                    dk = float(np.linalg.norm(vk))

                    uk = np.array([1.0, 0.0, 0.0], float) if dk < eps_st else (vk / dk)
                    dk = max(dk, eps_st)

                    Pm_k = Pm[3 * k:3 * k + 3, :]
                    a_k = (uk.reshape(1, 3) @ Pm_k).reshape(Nu,)
                    b_k = (dk - d0) - float(a_k @ U_guess_vec[:, 0])

                    A_s[k, :] = a_k
                    b_s[k, 0] = b_k

            if enable_inline_soft:
                idx_ahead_inline = int(getattr(self, "idx_ahead_inline", 5))
                A_lat = np.zeros((2 * Np, Nu), float)
                b_lat = np.zeros((2 * Np, 1), float)

                t_prev = None
                for k in range(Np):
                    r_nom_k = r_nom[3 * k:3 * k + 3, :]
                    Pm_k = Pm[3 * k:3 * k + 3, :]

                    x_tip_k = X_nom[k * n:k * n + 3, 0].reshape(3,)

                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc.shape[0] - 2))
                    i_anchor = int(np.clip(i_seg + idx_ahead_inline, 0, Cc.shape[0] - 1))
                    c = Cc[i_anchor, :3].reshape(3, 1)

                    t_k = Cc[i_seg + 1, :3] - Cc[i_seg, :3]
                    t_k = t_k / (np.linalg.norm(t_k) + 1e-12)
                    if t_prev is not None and np.dot(t_k, t_prev) < 0.0:
                        t_k = -t_k

                    ref = np.array([1.0, 0.0, 0.0])
                    if abs(np.dot(ref, t_k)) > 0.9:
                        ref = np.array([0.0, 1.0, 0.0])

                    n_k = ref - np.dot(ref, t_k) * t_k
                    n_k /= (np.linalg.norm(n_k) + 1e-12)
                    b_kv = np.cross(t_k, n_k)
                    b_kv /= (np.linalg.norm(b_kv) + 1e-12)

                    t_prev = t_k.copy()

                    a_n = (n_k.reshape(1, 3) @ Pm_k).reshape(-1)
                    a_b = (b_kv.reshape(1, 3) @ Pm_k).reshape(-1)

                    b_n = float(n_k @ (r_nom_k[:, 0] - c[:, 0])) - float(a_n @ U_guess_vec[:, 0])
                    b_b = float(b_kv @ (r_nom_k[:, 0] - c[:, 0])) - float(a_b @ U_guess_vec[:, 0])

                    A_lat[2 * k + 0, :] = a_n
                    b_lat[2 * k + 0, 0] = b_n
                    A_lat[2 * k + 1, :] = a_b
                    b_lat[2 * k + 1, 0] = b_b

            def skew3(v):
                v = np.asarray(v, float).reshape(3,)
                x_, y_, z_ = v
                return np.array([
                    [0.0, -z_,  y_],
                    [z_,  0.0, -x_],
                    [-y_, x_,  0.0]
                ], float)

            if enable_dipole_soft:
                if not hasattr(self, "_d_body_locked"):
                    d_body0 = np.asarray(self.dipole_body_axis, float).reshape(3,)
                    t0 = forward_tangent_smooth(Cc, int(idx_k_epm[0]), None, look=3)
                    q0 = np.asarray(p_seq[0][3:7], float).reshape(4,)
                    self._d_body_locked = pick_dipole_axis_forward(d_body0, q0, t0)

                d_body = self._d_body_locked

                d_nom = np.zeros((3 * Np, 1), float)
                t_tar = np.zeros((3 * Np, 1), float)

                t_prev = None
                for k in range(Np):
                    qk = np.asarray(p_seq[k][3:7], float).reshape(4,)
                    Rk = quat_wxyz_to_R(qk)

                    dk = Rk @ d_body
                    dk /= (np.linalg.norm(dk) + 1e-12)

                    tk = forward_tangent_smooth(Cc, int(idx_k_epm[k]), t_prev, look=3)
                    t_prev = tk.copy()
                    tk /= (np.linalg.norm(tk) + 1e-12)

                    d_nom[3 * k:3 * k + 3, 0] = dk
                    t_tar[3 * k:3 * k + 3, 0] = tk

                Pomega_w = build_Pomega_world(p_seq, self.dt, Np, m=m)
                Pomega_w = np.asarray(Pomega_w, float)

                assert Pomega_w.shape == (3 * Np, Nu), f"Pomega_w shape {Pomega_w.shape} expected {(3 * Np, Nu)}"

                A_align = np.zeros((3 * Np, Nu), float)
                for k in range(Np):
                    dk = d_nom[3 * k:3 * k + 3, 0].reshape(3,)
                    Sk = skew3(dk)
                    Pk = Pomega_w[3 * k:3 * k + 3, :]
                    A_align[3 * k:3 * k + 3, :] = -Sk @ Pk

                b_align = (d_nom - t_tar)

            A_list, l_list, u_list = [], [], []

            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np * m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            dL_back_max = float(getattr(self, "dL_back_max", 0.03))
            A_dL = np.zeros((Np, Np * m), float)
            for k in range(Np):
                A_dL[k, k * m + 6] = 1.0
            A_list.append(A_dL)
            l_list.append(-dL_back_max * np.ones(Np))
            u_list.append(np.full(Np, np.inf))

            enable_theta_eff = bool(getattr(self, "enable_hard_theta", False)) and (t_vessel is not None)
            theta_clearance_thresh_m = float(getattr(self, "theta_clearance_thresh_m", 0.4e-3))

            clearance_m = np.full(Np, np.inf, float)

            if enable_theta_eff and (self.lumen_R is not None):
                Rr = np.asarray(self.lumen_R, float).reshape(-1,)

                for k in range(Np):
                    x_tip_k = X_nom[k * n:k * n + 3, 0].reshape(3,)
                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc.shape[0] - 2))

                    rho_k = float(np.linalg.norm(x_tip_k - c_closest.reshape(3,)))
                    idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr.shape[0] - 1))
                    R_k = float(Rr[idx_v])

                    clearance_m[k] = R_k - rho_k

                hard_theta_mask = (clearance_m < theta_clearance_thresh_m)
            else:
                hard_theta_mask = np.zeros(Np, dtype=bool)

            enable_hard_epm_tip_clearance = bool(getattr(self, "enable_hard_epm_tip_clearance", True))
            epm_tip_clearance_min_m = float(getattr(self, "epm_tip_clearance_min_m", 0.025))

            if enable_hard_epm_tip_clearance:
                if "Pm" not in locals() or "r_nom" not in locals():
                    Pm = build_Pm_world(self.dt, Np, m=m)
                    U_guess_vec = U_guess.reshape(-1, 1)
                    r0 = p0[:3].copy().reshape(3, 1)
                    r0_stack = np.tile(r0, (Np, 1))
                    r_nom = r0_stack + Pm @ U_guess_vec

                A_epm_tip = np.zeros((Np, Nu), float)
                l_epm_tip = np.full(Np, -np.inf, float)
                u_epm_tip = np.full(Np, np.inf, float)

                eps_dist = 1e-9
                for k in range(Np):
                    r_nom_k = r_nom[3 * k:3 * k + 3, 0].reshape(3,)
                    Pm_k = Pm[3 * k:3 * k + 3, :]

                    rows_x = np.array([k * n + 0, k * n + 1, k * n + 2], dtype=int)
                    Mc_xk = Mc[rows_x, :]
                    x_nom_k = X_aff[rows_x, 0].reshape(3,) + (Mc_xk @ U_guess_vec).reshape(3,)

                    v_nom = r_nom_k - x_nom_k
                    d_nom = float(np.linalg.norm(v_nom))

                    if d_nom < eps_dist:
                        uhat = np.array([1.0, 0.0, 0.0], float)
                        d_nom = eps_dist
                    else:
                        uhat = v_nom / d_nom

                    a_k = (uhat.reshape(1, 3) @ (Pm_k - Mc_xk)).reshape(Nu,)
                    b_k = d_nom - float(a_k @ U_guess_vec[:, 0])

                    A_epm_tip[k, :] = a_k
                    l_epm_tip[k] = epm_tip_clearance_min_m - b_k
                    u_epm_tip[k] = np.inf

                A_list.append(A_epm_tip)
                l_list.append(l_epm_tip)
                u_list.append(u_epm_tip)

            ns_standoff = Np if enable_standoff_soft else 0
            ns_inline = Np if enable_inline_soft else 0
            ns_dipole = Np if enable_dipole_soft else 0

            off_standoff = Nu
            off_inline = off_standoff + ns_standoff
            off_dipole = off_inline + ns_inline

            ns = ns_standoff + ns_inline + ns_dipole
            nz = Nu + ns

            H_z = np.zeros((nz, nz), float)
            H_z[:Nu, :Nu] = H

            f_z = np.zeros((nz, 1), float)
            f_z[:Nu, :] = f

            eps_standoff = float(getattr(self, "eps_standoff_m", 0.03))
            eps_inline_lat = float(getattr(self, "eps_inline_lat_m", 0.01))
            eps_dipole = float(getattr(self, "eps_dipole", 1e-2))

            if ns_standoff > 0:
                w_slack_standoff = float(getattr(self, "w_slack_standoff", self.w_mag_center_standoff))
                i0s = off_standoff
                H_z[i0s:i0s + ns_standoff, i0s:i0s + ns_standoff] = 2.0 * w_slack_standoff * np.eye(ns_standoff)
            else:
                w_slack_standoff = 0.0

            if ns_inline > 0:
                w_slack_inline = float(getattr(self, "w_slack_inline", getattr(self, "w_mag_lat_inline", 0.1)))
                i0i = off_inline
                H_z[i0i:i0i + ns_inline, i0i:i0i + ns_inline] = 2.0 * w_slack_inline * np.eye(ns_inline)
            else:
                w_slack_inline = 0.0

            if ns_dipole > 0:
                w_slack_dipole = float(getattr(self, "w_slack_dipole", self.w_dipole_align))
                i0d = off_dipole
                H_z[i0d:i0d + ns_dipole, i0d:i0d + ns_dipole] = 2.0 * w_slack_dipole * np.eye(ns_dipole)
            else:
                w_slack_dipole = 0.0

            if A_list:
                A_osqp = np.vstack([_pad_A(Ai, nz, Nu) for Ai in A_list])
                l_osqp = np.concatenate(l_list).astype(float)
                u_osqp = np.concatenate(u_list).astype(float)
            else:
                A_osqp = np.zeros((0, nz), float)
                l_osqp = np.zeros(0, float)
                u_osqp = np.zeros(0, float)

            if ns > 0:
                A_s_nonneg = np.zeros((ns, nz), float)
                A_s_nonneg[:, Nu:] = np.eye(ns)
                A_osqp = np.vstack([A_osqp, A_s_nonneg])
                l_osqp = np.concatenate([l_osqp, np.zeros(ns)])
                u_osqp = np.concatenate([u_osqp, np.full(ns, np.inf)])

            if ns_standoff > 0:
                A_st = np.zeros((2 * Np, nz), float)
                l_st = np.full(2 * Np, -np.inf, float)
                u_st = np.full(2 * Np, np.inf, float)

                for k in range(Np):
                    col_s = off_standoff + k

                    A_st[2 * k, :Nu] = A_s[k, :]
                    A_st[2 * k, col_s] = -1.0
                    u_st[2 * k] = eps_standoff - b_s[k, 0]

                    A_st[2 * k + 1, :Nu] = -A_s[k, :]
                    A_st[2 * k + 1, col_s] = -1.0
                    u_st[2 * k + 1] = eps_standoff + b_s[k, 0]

                A_osqp = np.vstack([A_osqp, A_st])
                l_osqp = np.concatenate([l_osqp, l_st])
                u_osqp = np.concatenate([u_osqp, u_st])

            if ns_inline > 0:
                A_inl = np.zeros((4 * Np, nz), float)
                l_inl = np.full(4 * Np, -np.inf, float)
                u_inl = np.full(4 * Np, np.inf, float)

                for k in range(Np):
                    col_s = off_inline + k

                    a_n = A_lat[2 * k + 0, :]
                    b_n = b_lat[2 * k + 0, 0]
                    a_b = A_lat[2 * k + 1, :]
                    b_bv = b_lat[2 * k + 1, 0]

                    A_inl[4 * k + 0, :Nu] = a_n
                    A_inl[4 * k + 0, col_s] = -1.0
                    u_inl[4 * k + 0] = eps_inline_lat - b_n

                    A_inl[4 * k + 1, :Nu] = -a_n
                    A_inl[4 * k + 1, col_s] = -1.0
                    u_inl[4 * k + 1] = eps_inline_lat + b_n

                    A_inl[4 * k + 2, :Nu] = a_b
                    A_inl[4 * k + 2, col_s] = -1.0
                    u_inl[4 * k + 2] = eps_inline_lat - b_bv

                    A_inl[4 * k + 3, :Nu] = -a_b
                    A_inl[4 * k + 3, col_s] = -1.0
                    u_inl[4 * k + 3] = eps_inline_lat + b_bv

                A_osqp = np.vstack([A_osqp, A_inl])
                l_osqp = np.concatenate([l_osqp, l_inl])
                u_osqp = np.concatenate([u_osqp, u_inl])

            if ns_dipole > 0:
                A_dip_soft = np.zeros((6 * Np, nz), float)
                l_dip_soft = np.full(6 * Np, -np.inf, float)
                u_dip_soft = np.full(6 * Np, np.inf, float)

                for k in range(Np):
                    col_s = off_dipole + k
                    for j in range(3):
                        a = A_align[3 * k + j, :]
                        b = b_align[3 * k + j, 0]

                        A_dip_soft[6 * k + 2 * j, :Nu] = a
                        A_dip_soft[6 * k + 2 * j, col_s] = -1.0
                        u_dip_soft[6 * k + 2 * j] = eps_dipole - b

                        A_dip_soft[6 * k + 2 * j + 1, :Nu] = -a
                        A_dip_soft[6 * k + 2 * j + 1, col_s] = -1.0
                        u_dip_soft[6 * k + 2 * j + 1] = eps_dipole + b

                A_osqp = np.vstack([A_osqp, A_dip_soft])
                l_osqp = np.concatenate([l_osqp, l_dip_soft])
                u_osqp = np.concatenate([u_osqp, u_dip_soft])

            if np.any(hard_theta_mask):
                theta_max_deg = float(getattr(self, "theta_max_deg", 40.0))
                cos_max = float(np.cos(np.deg2rad(theta_max_deg)))

                active_k = np.flatnonzero(hard_theta_mask)
                A_theta = np.zeros((active_k.size, nz), float)
                l_theta = np.full(active_k.size, -np.inf, float)
                u_theta = np.full(active_k.size, np.inf, float)

                for row, k in enumerate(active_k):
                    tv = np.asarray(t_vessel[k], float).reshape(3,)
                    tv /= (np.linalg.norm(tv) + 1e-12)

                    rows_t = np.array([k * n + 3, k * n + 4, k * n + 5], dtype=int)
                    Mc_tk = Mc[rows_t, :]
                    t_aff = X_aff[rows_t, 0].reshape(3,)

                    aU = (tv.reshape(1, 3) @ Mc_tk).reshape(-1)
                    b = float(cos_max - (tv @ t_aff))

                    A_theta[row, :Nu] = aU
                    l_theta[row] = b
                    u_theta[row] = np.inf
            else:
                theta_max_deg = float(getattr(self, "theta_max_deg", 40.0))
                A_theta = None

            if np.any(hard_theta_mask):
                A_osqp = np.vstack([A_osqp, A_theta])
                l_osqp = np.concatenate([l_osqp, l_theta])
                u_osqp = np.concatenate([u_osqp, u_theta])

            if dbg_terms is not None and enable_standoff_soft:
                dbg_terms["standoff_model"] = dict(A=A_s.copy(), b=b_s.copy(), eps=float(eps_standoff), d0=float(d0))

            if dbg_terms is not None and enable_inline_soft:
                dbg_terms["inline_model"] = dict(A=A_lat.copy(), b=b_lat.copy(), eps=float(eps_inline_lat))

            if dbg_terms is not None and enable_dipole_soft:
                dbg_terms["dipole_model"] = dict(A=A_align.copy(), b=b_align.copy(), eps=float(eps_dipole))

            if dbg_terms is not None:
                dbg_terms["slack_weights"] = dict(
                    standoff=float(w_slack_standoff) if ns_standoff > 0 else 0.0,
                    inline=float(w_slack_inline) if ns_inline > 0 else 0.0,
                    dipole=float(w_slack_dipole) if ns_dipole > 0 else 0.0,
                )

            Z_warm = None
            if (U_opt_vec is not None) and (ns > 0):
                Z_warm = np.zeros(nz, float)
                Z_warm[:Nu] = U_opt_vec.copy()

            Z_opt, _, status = solve_qp_osqp(H_z, f_z.ravel(), A_osqp, l_osqp, u_osqp, U_warm=Z_warm)
            status_last = status
            U_opt_vec = None
            print("QP dims:")
            print("H_z:", H_z.shape)
            print("f_z:", f_z.shape)
            print("A_osqp:", A_osqp.shape)
            print("l_osqp:", l_osqp.shape)
            print("u_osqp:", u_osqp.shape)
            print("all finite H:", np.all(np.isfinite(H_z)))
            print("all finite f:", np.all(np.isfinite(f_z)))
            print("all finite A:", np.all(np.isfinite(A_osqp)))
            print("all finite l:", np.all(np.isfinite(l_osqp)))
            print("all finite u:", np.all(np.isfinite(u_osqp)))
            print("any l > u:", np.any(l_osqp > u_osqp))
            if status in ("solved", "solved inaccurate") and (Z_opt is not None):
                if ns > 0:
                    U_opt_vec = Z_opt[:Nu].copy()
                    self._slack_last = Z_opt[Nu:].copy()
                else:
                    U_opt_vec = np.asarray(Z_opt, float).copy()

            if U_opt_vec is None:
                infeas = True
                break
            else:
                infeas = False
                U_guess = U_opt_vec.reshape(Np, m)

        infeas_final = (U_opt_vec is None)

        if infeas_final:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
        else:
            U_seq = U_opt_vec.reshape(Np, m)
            u0 = U_seq[0, :].copy()

        X_pred = np.full((Np, n), np.nan)
        if (not infeas_final) and (Mc_last is not None) and (X_aff_last is not None):
            U_vec = U_opt_vec.reshape(-1, 1)
            X_pred_stack = X_aff_last + Mc_last @ U_vec
            X_pred = X_pred_stack.reshape(Np, n)

        # real-control update:
        # - propagate p as internal actuator/configuration estimate
        # - use camera measurement as true state x
        p_next_est = self._clamp_p(integrate_pose8_body(p_prev, u0, self.dt))
        x_next_used = np.asarray(x_meas, dtype=float).reshape(n,)

        self.p = p_next_est.copy()
        self.x = x_next_used.copy()
        self.p_last_meas = self.p.copy()
        self.x_last_meas = self.x.copy()

        if not infeas_final:
            U_shift = np.vstack([U_seq[1:], np.zeros((1, m))])
            self.U_warm = U_shift.reshape(-1)
        else:
            self.U_warm = None



        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            e_now_pred = x_next_used[:3] - X_pred[0][:3]
            pred1_err_xy_nvslat = float(np.linalg.norm(e_now_pred[:2]))
            pred1_err_xyz_nvslat = float(np.linalg.norm(e_now_pred))
            tan1_err_nvslat = float(np.linalg.norm(x_next_used[3:6] - X_pred[0][3:6])) if (n >= 6 and X_pred.shape[1] >= 6) else np.nan

        if self.debug and (U_opt_vec is not None):
            U_dbg = U_opt_vec.reshape(Nu, 1)

            if ns > 0 and hasattr(self, "_slack_last") and (self._slack_last is not None):
                s_dbg = np.asarray(self._slack_last, float).reshape(-1, 1)
            else:
                s_dbg = np.zeros((ns, 1), float)

            def _term_cost(Ht, ft, U):
                if Ht is None:
                    Ht = np.zeros((U.shape[0], U.shape[0]), float)
                if ft is None:
                    ft = np.zeros((U.shape[0], 1), float)
                return float(0.5 * (U.T @ Ht @ U)[0, 0] + (ft.T @ U)[0, 0])

            dbg_costs = {}
            dbg_weights = {}

            dbg_costs["effort"] = _term_cost(H_effort, np.zeros((Nu, 1)), U_dbg)
            dbg_weights["R_diag"] = np.diag(self.R).copy().tolist()

            if H_smooth is not None:
                dbg_costs["smooth"] = _term_cost(H_smooth, np.zeros((Nu, 1)), U_dbg)
                dbg_weights["Rd_diag"] = np.diag(self.Rd).copy().tolist()
            else:
                dbg_costs["smooth"] = 0.0
                dbg_weights["Rd_diag"] = []

            dbg_costs["track"] = _term_cost(H_track, f_track, U_dbg)
            dbg_weights["track_Q_diag"] = np.diag(self.Q).copy().tolist()

            if "advance" in dbg_terms:
                H_adv_dbg, f_adv_dbg = dbg_terms["advance"]
                dbg_costs["advance"] = _term_cost(H_adv_dbg, f_adv_dbg, U_dbg)
            else:
                dbg_costs["advance"] = 0.0

            if "tangent" in dbg_terms:
                H_tan_dbg, f_tan_dbg = dbg_terms["tangent"]
                dbg_costs["tangent"] = _term_cost(H_tan_dbg, f_tan_dbg, U_dbg)
            else:
                dbg_costs["tangent"] = 0.0

            s_off = 0
            if ns_standoff > 0:
                s_st = s_dbg[s_off:s_off + ns_standoff]
                dbg_costs["slack_standoff"] = float(w_slack_standoff * np.sum(s_st ** 2))
                s_off += ns_standoff
            else:
                dbg_costs["slack_standoff"] = 0.0

            if ns_inline > 0:
                s_in = s_dbg[s_off:s_off + ns_inline]
                dbg_costs["slack_inline"] = float(w_slack_inline * np.sum(s_in ** 2))
                s_off += ns_inline
            else:
                dbg_costs["slack_inline"] = 0.0

            if ns_dipole > 0:
                s_dp = s_dbg[s_off:s_off + ns_dipole]
                dbg_costs["slack_dipole"] = float(w_slack_dipole * np.sum(s_dp ** 2))
                s_off += ns_dipole
            else:
                dbg_costs["slack_dipole"] = 0.0

            dbg_costs["total_objective"] = float(
                0.5 * (np.asarray(Z_opt, float).reshape(-1, 1).T @ H_z @ np.asarray(Z_opt, float).reshape(-1, 1))[0, 0]
                + (f_z.T @ np.asarray(Z_opt, float).reshape(-1, 1))[0, 0]
            )

            dbg_pen = {}

            if "advance_model" in dbg_terms:
                advm = dbg_terms["advance_model"]
                g_adv_val = (advm["g_adv"].T @ U_dbg).reshape(-1)
                dbg_pen["advance"] = dict(
                    weight=float(advm["w_adv_eff"]),
                    directional_progress=g_adv_val.copy(),
                    mean_directional_progress=float(np.mean(g_adv_val)),
                    min_directional_progress=float(np.min(g_adv_val)),
                    max_directional_progress=float(np.max(g_adv_val)),
                    s_des_eff=float(advm["s_des_eff"]),
                )

            if "tangent_model" in dbg_terms:
                tanm = dbg_terms["tangent_model"]
                e_tan = (tanm["A"] @ U_dbg + tanm["b"]).reshape(-1)
                cos_pred = tanm["cos_ref"] - e_tan
                cos_pred = np.clip(cos_pred, -1.0, 1.0)
                theta_pred = np.degrees(np.arccos(cos_pred))

                dbg_pen["tangent"] = dict(
                    theta_ref_deg=float(tanm["theta_ref_deg"]),
                    weights=np.asarray(dbg_terms["tangent_weights"], float).copy(),
                    clearance_m=np.asarray(dbg_terms["tangent_clearance_m"], float).copy(),
                    residual=e_tan.copy(),
                    theta_pred_deg=theta_pred.copy(),
                    theta_pred_deg_min=float(np.min(theta_pred)),
                    theta_pred_deg_max=float(np.max(theta_pred)),
                    theta_pred_deg_mean=float(np.mean(theta_pred)),
                )

            dbg_con = {}

            if ns_standoff > 0 and "standoff_model" in dbg_terms:
                mdl = dbg_terms["standoff_model"]
                e = (mdl["A"] @ U_dbg + mdl["b"]).reshape(-1)
                if ns > 0:
                    s_st = np.asarray(self._slack_last[:ns_standoff], float)
                else:
                    s_st = np.zeros(ns_standoff, float)
                viol = np.maximum(np.abs(e) - mdl["eps"] - s_st, 0.0)
                dbg_con["standoff_soft"] = dict(
                    weight=float(w_slack_standoff),
                    eps=float(mdl["eps"]),
                    residual=e.copy(),
                    slack=s_st.copy(),
                    violation=viol.copy(),
                    residual_abs_max=float(np.max(np.abs(e))),
                    slack_max=float(np.max(s_st)) if s_st.size else 0.0,
                    violation_max=float(np.max(viol)) if viol.size else 0.0,
                )

            if ns_inline > 0 and "inline_model" in dbg_terms:
                mdl = dbg_terms["inline_model"]
                e = (mdl["A"] @ U_dbg + mdl["b"]).reshape(-1)
                if ns > 0:
                    s0 = ns_standoff
                    s_in = np.asarray(self._slack_last[s0:s0 + ns_inline], float)
                else:
                    s_in = np.zeros(ns_inline, float)

                e_n = e[0::2]
                e_b = e[1::2]
                rho_lat = np.sqrt(e_n ** 2 + e_b ** 2)

                viol_n = np.maximum(np.abs(e_n) - mdl["eps"] - s_in, 0.0)
                viol_b = np.maximum(np.abs(e_b) - mdl["eps"] - s_in, 0.0)

                dbg_con["inline_soft"] = dict(
                    weight=float(w_slack_inline),
                    eps=float(mdl["eps"]),
                    e_n=e_n.copy(),
                    e_b=e_b.copy(),
                    rho_lat=rho_lat.copy(),
                    slack=s_in.copy(),
                    violation_n=viol_n.copy(),
                    violation_b=viol_b.copy(),
                    rho_lat_max=float(np.max(rho_lat)) if rho_lat.size else 0.0,
                    slack_max=float(np.max(s_in)) if s_in.size else 0.0,
                    violation_max=float(max(np.max(viol_n), np.max(viol_b))) if viol_n.size else 0.0,
                )

            if ns_dipole > 0 and "dipole_model" in dbg_terms:
                mdl = dbg_terms["dipole_model"]
                e = (mdl["A"] @ U_dbg + mdl["b"]).reshape(-1)
                if ns > 0:
                    s0 = ns_standoff + ns_inline
                    s_dp = np.asarray(self._slack_last[s0:s0 + ns_dipole], float)
                else:
                    s_dp = np.zeros(ns_dipole, float)

                e_stage = e.reshape(Np, 3)
                e_norm = np.linalg.norm(e_stage, axis=1)
                viol_stage = np.maximum(np.max(np.abs(e_stage), axis=1) - mdl["eps"] - s_dp, 0.0)

                dbg_con["dipole_soft"] = dict(
                    weight=float(w_slack_dipole),
                    eps=float(mdl["eps"]),
                    residual=e_stage.copy(),
                    residual_norm=e_norm.copy(),
                    slack=s_dp.copy(),
                    violation=viol_stage.copy(),
                    residual_norm_max=float(np.max(e_norm)) if e_norm.size else 0.0,
                    slack_max=float(np.max(s_dp)) if s_dp.size else 0.0,
                    violation_max=float(np.max(viol_stage)) if viol_stage.size else 0.0,
                )

            if "track_model" in dbg_terms:
                trk = dbg_terms["track_model"]
                X_pred_dbg = trk["X_aff"] + trk["Mc"] @ U_dbg
                e_track = (X_pred_dbg - trk["X_ref"]).reshape(Np, n)

                e_pos = e_track[:, 0:3]
                pos_err_norm = np.linalg.norm(e_pos, axis=1)

                if n >= 6:
                    e_tan = e_track[:, 3:6]
                    tan_err_norm = np.linalg.norm(e_tan, axis=1)
                else:
                    e_tan = np.zeros((Np, 0), float)
                    tan_err_norm = np.full(Np, np.nan)

                dbg_pen["track"] = dict(
                    Q_diag=np.asarray(trk["Q_diag"], float).copy(),
                    idx_ref=np.asarray(trk["idx_ref"], int).copy(),
                    X_ref=trk["X_ref"].reshape(Np, n).copy(),
                    X_pred=X_pred_dbg.reshape(Np, n).copy(),
                    residual=e_track.copy(),
                    pos_err=e_pos.copy(),
                    pos_err_norm=pos_err_norm.copy(),
                    tan_err=e_tan.copy(),
                    tan_err_norm=tan_err_norm.copy(),
                    cost=float(0.5 * (U_dbg.T @ H_track @ U_dbg)[0, 0] + (f_track.T @ U_dbg)[0, 0]),
                    pos_err_norm_min=float(np.min(pos_err_norm)) if pos_err_norm.size else np.nan,
                    pos_err_norm_mean=float(np.mean(pos_err_norm)) if pos_err_norm.size else np.nan,
                    pos_err_norm_max=float(np.max(pos_err_norm)) if pos_err_norm.size else np.nan,
                )

            if np.any(hard_theta_mask):
                cos_vals = []
                theta_vals = []
                margins = []

                for k in np.flatnonzero(hard_theta_mask):
                    tv = np.asarray(t_vessel[k], float).reshape(3,)
                    tv /= (np.linalg.norm(tv) + 1e-12)

                    rows_t = np.array([k * n + 3, k * n + 4, k * n + 5], dtype=int)
                    t_pred = (X_aff[rows_t, :].reshape(3, 1) + Mc[rows_t, :] @ U_dbg).reshape(3,)
                    t_pred /= (np.linalg.norm(t_pred) + 1e-12)

                    cosk = float(np.clip(tv @ t_pred, -1.0, 1.0))
                    thetak = float(np.degrees(np.arccos(cosk)))
                    margin = cosk - float(np.cos(np.deg2rad(theta_max_deg)))

                    cos_vals.append(cosk)
                    theta_vals.append(thetak)
                    margins.append(margin)

                dbg_con["theta_hard"] = dict(
                    theta_max_deg=float(theta_max_deg),
                    active_k=np.flatnonzero(hard_theta_mask).copy(),
                    clearance_m=clearance_m[hard_theta_mask].copy(),
                    cos_vals=np.asarray(cos_vals, float),
                    theta_deg=np.asarray(theta_vals, float),
                    margin=np.asarray(margins, float),
                    min_margin=float(np.min(margins)) if len(margins) else np.nan,
                )

            self._mpc_dbg_last = dict(
                weights=dict(
                    w_adv_eff=float(w_adv_eff) if t_vessel is not None and enable_adv_eff else 0.0,
                    w_tan_min=float(w_tan_min) if enable_tangent_pen and (self.lumen_R is not None) else 0.0,
                    w_tan_max=float(w_tan_max) if enable_tangent_pen and (self.lumen_R is not None) else 0.0,
                    w_slack_standoff=float(w_slack_standoff) if ns_standoff > 0 else 0.0,
                    w_slack_inline=float(w_slack_inline) if ns_inline > 0 else 0.0,
                    w_slack_dipole=float(w_slack_dipole) if ns_dipole > 0 else 0.0,
                    theta_ref_deg=float(theta_ref_deg) if enable_tangent_pen and (self.lumen_R is not None) else np.nan,
                    theta_max_deg=float(theta_max_deg),
                ),
                costs=dbg_costs,
                penalties=dbg_pen,
                constraints=dbg_con,
            )
            self.x_pred_prev_1 = X_pred[0].copy()
            print("\n[DBG MPC] objective term costs")
            for k_, v_ in dbg_costs.items():
                print(f"  {k_:>18s}: {v_: .6e}")

            print("[DBG MPC] weights")
            for k_, v_ in self._mpc_dbg_last["weights"].items():
                if np.isscalar(v_):
                    print(f"  {k_:>18s}: {v_}")

            if "advance" in dbg_pen:
                ap = dbg_pen["advance"]
                print("[DBG MPC] advance")
                print(
                    f"  weight={ap['weight']:.6e}  mean={ap['mean_directional_progress']:.6e}  "
                    f"min={ap['min_directional_progress']:.6e}  max={ap['max_directional_progress']:.6e}"
                )

            if "tangent" in dbg_pen:
                tp = dbg_pen["tangent"]
                print("[DBG MPC] tangent")
                print(f"  theta_ref_deg={tp['theta_ref_deg']:.2f}")
                print(
                    f"  theta_pred_deg min/mean/max = "
                    f"{tp['theta_pred_deg_min']:.3f} / {tp['theta_pred_deg_mean']:.3f} / {tp['theta_pred_deg_max']:.3f}"
                )
                print(f"  weight min/max = {np.min(tp['weights']):.6e} / {np.max(tp['weights']):.6e}")
                print(f"  clearance min/max [mm] = {1e3 * np.min(tp['clearance_m']):.3f} / {1e3 * np.max(tp['clearance_m']):.3f}")

            for name in ("standoff_soft", "inline_soft", "dipole_soft", "theta_hard"):
                if name in dbg_con:
                    print(f"[DBG MPC] {name}")
                    for kk, vv in dbg_con[name].items():
                        if np.isscalar(vv):
                            print(f"  {kk}: {vv}")

        info = dict(
            status=status_last,
            infeasible=int(infeas_final),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            d=self.d.copy(),
            X_pred=X_pred.copy(),
            X_nom = X_nom.copy(),
            U_seq=U_seq.copy(),
            N_sqp=int(self.N_sqp),
            p_prev=p_prev.copy(),
            x_prev=x_prev.copy(),
            B_first=B_first.copy() if B_first is not None else None,
            p_lin=p_lin.copy() if isinstance(p_lin, np.ndarray) else p_lin,
            p_first=p_first.copy() if isinstance(p_first, np.ndarray) else p_first,
            X_aff_last=X_aff_last.copy() if X_aff_last is not None else None,
            Mc_last=Mc_last.copy() if Mc_last is not None else None,
            X_nom_last=X_nom_last.reshape(Np, n).copy() if X_nom_last is not None else None,
            p_seq_last=p_seq_last.copy() if p_seq_last is not None else None,
            pred1_err_xy=float(pred1_err_xy) if x_pred_prev_1 is not None else None,
            pred1_err_xyz=float(pred1_err_xyz) if x_pred_prev_1 is not None else None,
            pred1_err_xy_nvslat=float(pred1_err_xy_nvslat)if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])) else None,
            pred1_err_xyz_nvslat=float(pred1_err_xyz_nvslat) if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])) else None,
            tan1_err=float(tan1_err) if np.isfinite(tan1_err) else np.nan,
            theta0_deg=float(theta0) if np.isfinite(theta0) else np.nan,
            theta_max_deg=float(theta_max_deg),
            jac_svd_S=self._jac_svd_last["S"].copy() if hasattr(self, "_jac_svd_last") else None,
            jac_svd_cond=float(self._jac_svd_last["cond"]) if hasattr(self, "_jac_svd_last") else np.nan,
            jac_svd_rank=int(self._jac_svd_last["rank"]) if hasattr(self, "_jac_svd_last") else -1,
            jac_trans_norm=float(self._jac_svd_last["trans_norm"]) if hasattr(self, "_jac_svd_last") else np.nan,
            jac_omega_norm=float(self._jac_svd_last["omega_norm"]) if hasattr(self, "_jac_svd_last") else np.nan,
            jac_dL_norm=float(self._jac_svd_last["dL_norm"]) if hasattr(self, "_jac_svd_last") else np.nan,
            mpc_debug=self._mpc_dbg_last.copy() if hasattr(self, "_mpc_dbg_last") else None,
        )

        return self.p.copy(), self.x.copy(), info
def vision_result_to_x_meas_robot(
    vision_result,
    pivot_point_pose6,
    align_tangent_with_lumen=True,
):
    tip_result = vision_result["tip_result"]
    markers = tip_result["markers"]
    mm_per_pixel = float(vision_result["mm_per_pixel"])

    tip_xy_px = np.asarray(tip_result["tip_xy_from_base"], dtype=float).reshape(2,)
    tip_xy_m = (tip_xy_px * mm_per_pixel) / 1000.0

    p_local = np.array([
        tip_xy_m[0],
        -tip_xy_m[1],
        0.0,
    ], dtype=float)

    p_local_wrong = np.array([tip_xy_m[0], tip_xy_m[1], 0.0], dtype=float)
    p_local_right = np.array([-tip_xy_m[0], tip_xy_m[1], 0.0], dtype=float)

    p_pivot = np.asarray(pivot_point_pose6[:3], float)

    print("wrong p_robot =", p_pivot + p_local_wrong)
    print("right p_robot =", p_pivot + p_local_right)

    tip_px = np.asarray(markers["tip_px"], dtype=float)
    tan_start_px = np.asarray(markers["tangent_start_px"], dtype=float)
    base_px = np.asarray(markers["base_px"], dtype=float)

    mag_start_px_raw = markers.get("mag_start_px", None)

    if mag_start_px_raw is not None:
        mag_start_px = np.asarray(mag_start_px_raw, dtype=float)
        ref = np.array([
            mag_start_px[0] - base_px[0],
            -(mag_start_px[1] - base_px[1]),
        ], dtype=float)
        ref_source = "base->mag_start"
    else:
        # fallback convention
        ref = np.array([
            tip_px[0] - base_px[0],
            -(tip_px[1] - base_px[1]),
        ], dtype=float)
        ref_source = "base->tip (fallback)"

    nref = np.linalg.norm(ref)
    if nref < 1e-12:
        raise ValueError(f"Reference vector is zero length. source={ref_source}")
    ex = ref / nref
    ey = np.array([-ex[1], ex[0]], dtype=float)

    v_img = np.array([
        tip_px[0] - tan_start_px[0],
        -(tip_px[1] - tan_start_px[1]),
    ], dtype=float)

    nv = np.linalg.norm(v_img)
    if nv < 1e-12:
        raise ValueError("Measured tangent from vision is zero-length.")
    v_img /= nv

    tx_local = -float(np.dot(v_img, ex))
    ty_local =  float(np.dot(v_img, ey))

    t_local = np.array([tx_local, ty_local, 0.0], dtype=float)

    nrm = np.linalg.norm(t_local)
    if nrm < 1e-12:
        raise ValueError("Measured tangent in local frame is zero-length.")
    t_local /= nrm

    p_pivot_robot = np.asarray(pivot_point_pose6[:3], dtype=float).reshape(3,)
    p_robot = p_pivot_robot + p_local
    t_robot = t_local.copy()

    nrm_t = np.linalg.norm(t_robot)
    if nrm_t < 1e-12:
        raise ValueError("Robot-frame tangent became zero-length.")
    t_robot /= nrm_t

    print("[DBG TAN]")
    print("  tip_xy_m =", tip_xy_m)
    print("  p_local =", p_local)
    print("  ref_source =", ref_source)
    print("  ref =", ref)
    print("  v_img =", v_img)
    print("  ex =", ex)
    print("  ey =", ey)
    print("  tx_local, ty_local =", tx_local, ty_local)
    print("  t_local =", t_local)
    print("  p_robot =", p_robot)
    print("  t_robot =", t_robot)

    lumen_C_robot_m = vision_result.get("lumen_C_robot_m", None)
    if align_tangent_with_lumen and (lumen_C_robot_m is not None):
        C = np.asarray(lumen_C_robot_m, dtype=float)
        if C.ndim == 2 and C.shape[0] >= 2 and C.shape[1] >= 3:
            d2 = np.sum((C - p_robot.reshape(1, 3)) ** 2, axis=1)
            i = int(np.argmin(d2))

            if i <= 0:
                t_lumen = C[1] - C[0]
            elif i >= len(C) - 1:
                t_lumen = C[-1] - C[-2]
            else:
                t_lumen = C[i + 1] - C[i - 1]

            nrm_l = np.linalg.norm(t_lumen)
            if nrm_l > 1e-12:
                t_lumen /= nrm_l
                if float(np.dot(t_robot, t_lumen)) < 0.0:
                    t_robot = -t_robot

    return np.array([
        p_robot[0],
        p_robot[1],
        p_robot[2],
        t_robot[0],
        t_robot[1],
        t_robot[2],
    ], dtype=float)
import numpy as np
import time

def quat_from_yaw_wxyz(dpsi):
    c = np.cos(0.5 * dpsi)
    s = np.sin(0.5 * dpsi)
    return np.array([c, 0.0, 0.0, s], dtype=float)   # rotation about z

def numerical_J_robot_xy_yaw_dL(
    p8,
    forward_y_fn,
    dx=5e-3,
    dy=5e-3,
    dyaw=np.deg2rad(5.0),
    dL=1e-3,
    n_out=3,
):
    import time
    import numpy as np

    t0 = time.perf_counter()

    p8 = np.asarray(p8, float).ravel().copy()
    J = np.zeros((n_out, 4), float)

    def eval_y(p):
        y_full = np.asarray(forward_y_fn(p), float).reshape(-1)
        return y_full[:n_out]

    print("[DBG] p8 =", p8)

    # nominal
    y0 = eval_y(p8)
    print("[DBG] y0 =", y0)

    # 1) x central difference
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[0] += dx
    p_minus[0] -= dx

    y_plus = eval_y(p_plus)
    y_minus = eval_y(p_minus)

    print("[DBG] x+ =", y_plus, "x- =", y_minus, "central dy =", y_plus - y_minus)
    J[:, 0] = (y_plus - y_minus) / (2.0 * dx)

    # 2) y central difference
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[1] += dy
    p_minus[1] -= dy

    y_plus = eval_y(p_plus)
    y_minus = eval_y(p_minus)

    print("[DBG] y+ =", y_plus, "y- =", y_minus, "central dy =", y_plus - y_minus)
    J[:, 1] = (y_plus - y_minus) / (2.0 * dy)

    # 3) yaw-z central difference
    p_plus = p8.copy()
    p_minus = p8.copy()

    q = quat_wxyz_normalize(p8[3:7])
    dqz_plus = quat_from_yaw_wxyz(+dyaw)
    dqz_minus = quat_from_yaw_wxyz(-dyaw)

    # world-z perturbation
    p_plus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_plus, q))
    p_minus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_minus, q))

    y_plus = eval_y(p_plus)
    y_minus = eval_y(p_minus)

    print("[DBG] yaw+ =", y_plus, "yaw- =", y_minus, "central dy =", y_plus - y_minus)
    J[:, 2] = (y_plus - y_minus) / (2.0 * dyaw)

    # 4) L central difference
    p_plus = p8.copy()
    p_minus = p8.copy()
    p_plus[7] += dL
    p_minus[7] -= dL

    y_plus = eval_y(p_plus)
    y_minus = eval_y(p_minus)

    print("[DBG] L+ =", y_plus, "L- =", y_minus, "central dy =", y_plus - y_minus)
    J[:, 3] = (y_plus - y_minus) / (2.0 * dL)

    t1 = time.perf_counter()
    print("[DBG] J =\n", J)
    print(f"[TIME] numerical_J_robot total: {(t1-t0)*1e3:.2f} ms")

    return J
def J_full_from_robot_reduced(J_red, n_out_full=6):
    J_full = np.zeros((n_out_full, 7), float)
    J_full[0:3, 0] = J_red[:, 0]   # x translation
    J_full[0:3, 1] = J_red[:, 1]   # y translation
    J_full[0:3, 5] = J_red[:, 2]   # z rotation
    J_full[0:3, 6] = J_red[:, 3]   # dL
    return J_full
def numerical_B_y_wrt_u(p8, forward_y_fn, dt, eps_u, n_out):
    import time
    t0 = time.perf_counter()

    p8 = np.asarray(p8, float).ravel()
    B = np.zeros((n_out, 7), float)

    tint = 0.0
    tfwd = 0.0

    for i in range(6):
        du = np.zeros(7); du[i] = eps_u[i]

        a = time.perf_counter()
        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)
        b = time.perf_counter()
        tint += (b - a)

        a = time.perf_counter()
        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
        b = time.perf_counter()
        tfwd += (b - a)

        B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

    delta_L = 1e-3
    p_plus = p8.copy();  p_plus[7]  += delta_L
    p_minus = p8.copy(); p_minus[7] -= delta_L

    a = time.perf_counter()
    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
    b = time.perf_counter()
    tfwd += (b - a)

    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
    B[:, 6] = dt * dy_dL

    t1 = time.perf_counter()
    print(f"[TIME] numerical_B total: {(t1-t0)*1e3:.2f} ms")
    print(f"[TIME] integrate total: {(tint)*1e3:.2f} ms")
    print(f"[TIME] forward total: {(tfwd)*1e3:.2f} ms")

    return B
def numerical_B_y_wrt_u_forward(p8, forward_y_fn, dt, eps_u, n_out):
    p8 = np.asarray(p8, float).ravel()
    B = np.zeros((n_out, 7), float)

    y0 = np.asarray(forward_y_fn(p8), float).reshape(n_out,)

    for i in range(6):
        du = np.zeros(7)
        du[i] = eps_u[i]
        p_plus = integrate_pose8_body(p8, du, dt)
        y_plus = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        B[:, i] = (y_plus - y0) / eps_u[i]

    delta_L = 1e-3
    p_plus = p8.copy()
    p_plus[7] += delta_L
    y_plus = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    dy_dL = (y_plus - y0) / delta_L
    B[:, 6] = dt * dy_dL

    return B
def build_measured_p8_from_pose6_and_length(pose6, L_meas, z_offset=0.25):
    pose6 = np.asarray(pose6, float).reshape(6,)
    p7 = np.array([
        pose6[0],
        pose6[1],
        pose6[2] - z_offset,
        pose6[3],
        pose6[4],
        pose6[5],
        float(L_meas),
    ], dtype=float)
    return pose7_rotvec_to_pose8_quat(p7)
def predicted_targets_from_info(info, Np=None):
    X_pred = info.get("X_pred", None)
    if X_pred is None:
        return None
    X_pred = np.asarray(X_pred, float)
    if X_pred.ndim == 1:
        if Np is None:
            return None
        X_pred = X_pred.reshape(Np, -1)
    if X_pred.shape[1] < 3:
        return None
    pred = X_pred[:, :3]
    if not np.all(np.isfinite(pred)):
        return None
    return pred


def arc_length_param(C):
    """C: (M,3) -> s: (M,) cumulative arc-length."""
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s


def resample_polyline(C, ds_target=1e-3):
    """
    Resample polyline C to approximately uniform spacing ds_target.
    Returns C_rs: (Mr,3), s_rs: (Mr,)
    """
    C = np.asarray(C, float)
    s = arc_length_param(C)
    L = s[-1]
    if L < 1e-12:
        return C.copy(), s

    s_rs = np.arange(0.0, L + 0.5*ds_target, ds_target)
    C_rs = np.zeros((len(s_rs), 3), float)

    # piecewise-linear interpolation in arc-length
    for k in range(3):
        C_rs[:, k] = np.interp(s_rs, s, C[:, k])

    return C_rs, s_rs
def advance_cursor_monotone(path, x, i_ref, window=30):
    """
    Find closest index to x, but only search forward from i_ref.
    window: how far ahead you allow matching (in points).
    """
    M = path.shape[0]
    i_lo = int(i_ref)
    i_hi = int(min(M, i_ref + window))
    seg = path[i_lo:i_hi]
    if seg.shape[0] == 0:
        return M - 1
    d2 = np.sum((seg - x.reshape(1, 3))**2, axis=1)
    return i_lo + int(np.argmin(d2))

def snapshot_forward(forward6d, p8, *, commit=False):
    """
    Returns a self-consistent snapshot from ONE forward call.
    """
    y = np.asarray(forward6d(p8, commit=commit), float).reshape(6,)
    C = getattr(forward6d, "last_p_centerline", None)
    C = None if C is None else np.asarray(C, float).copy()

    tip = y[:3].copy()
    tan = y[3:6].copy()
    return tip, tan, C, y
import numpy as np
import time
import copy

def snapshot_forward_cache(fwd_model):
    snap = {}
    if hasattr(fwd_model, "_last"):
        snap["_last"] = copy.deepcopy(fwd_model._last)
    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if hasattr(fwd_model, name):
            snap[name] = copy.deepcopy(getattr(fwd_model, name))
    return snap

def restore_forward_cache(fwd_model, snap):
    if "_last" in snap and hasattr(fwd_model, "_last"):
        fwd_model._last = copy.deepcopy(snap["_last"])
    for name in ["last_tip", "last_p_centerline", "last_theta", "last_info"]:
        if name in snap and hasattr(fwd_model, name):
            setattr(fwd_model, name, copy.deepcopy(snap[name]))

def quat_from_yaw_wxyz(dpsi):
    c = np.cos(0.5 * dpsi)
    s = np.sin(0.5 * dpsi)
    return np.array([c, 0.0, 0.0, s], dtype=float)
def J_full_from_robot_reduced_tip_tangent(J_red, n_out_full=5):
    """
    J_red shape: (5,4)
        columns = [x, y, yaw_z, L]

    Returns full robot Jacobian in 7 controls:
        [vx, vy, vz, wx, wy, wz, dL]
    """
    J_full = np.zeros((n_out_full, 7), float)
    J_full[:, 0] = J_red[:, 0]   # x translation
    J_full[:, 1] = J_red[:, 1]   # y translation
    J_full[:, 5] = J_red[:, 2]   # yaw about z
    J_full[:, 6] = J_red[:, 3]   # insertion
    return J_full
def numerical_J_robot_xy_yaw_dL_warm_branch(
    p8,
    forward_model,   # the actual warm wrapper object, not just a plain fn
    dx=5e-3,
    dy=5e-3,
    dyaw=np.deg2rad(5.0),
    dL=1e-3,
    n_out=5,         # e.g. [tip_x, tip_y, tip_z, tx, ty]
):
    t0 = time.perf_counter()

    p8 = np.asarray(p8, float).ravel().copy()
    J = np.zeros((n_out, 4), float)

    def eval_y_from_p8(p):
        y = np.asarray(forward_model(p), float).reshape(-1)
        return y[:n_out]

    # First solve nominal point ONCE to establish current branch
    y0 = eval_y_from_p8(p8)

    # Snapshot warm branch state at nominal point
    snap0 = snapshot_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model)

    # 1) x
    p_plus = p8.copy();  p_plus[0] += dx
    p_minus = p8.copy(); p_minus[0] -= dx

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 0] = (y_plus - y_minus) / (2.0 * dx)

    # 2) y
    p_plus = p8.copy();  p_plus[1] += dy
    p_minus = p8.copy(); p_minus[1] -= dy

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 1] = (y_plus - y_minus) / (2.0 * dy)

    # 3) yaw
    p_plus = p8.copy()
    p_minus = p8.copy()

    q = quat_wxyz_normalize(p8[3:7])
    dqz_plus = quat_from_yaw_wxyz(+dyaw)
    dqz_minus = quat_from_yaw_wxyz(-dyaw)

    p_plus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_plus, q))
    p_minus[3:7] = quat_wxyz_normalize(quat_wxyz_mul(dqz_minus, q))

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 2] = (y_plus - y_minus) / (2.0 * dyaw)

    # 4) L
    p_plus = p8.copy();  p_plus[7] += dL
    p_minus = p8.copy(); p_minus[7] -= dL

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_plus = eval_y_from_p8(p_plus)

    restore_forward_cache(forward_model.fwd if hasattr(forward_model, "fwd") else forward_model, snap0)
    y_minus = eval_y_from_p8(p_minus)

    J[:, 3] = (y_plus - y_minus) / (2.0 * dL)

    t1 = time.perf_counter()
    print(f"[TIME] numerical_J_robot_warm_branch total: {(t1-t0)*1e3:.2f} ms")

    return J

def make_initial_poses_single_use(hw) -> tuple[np.ndarray, np.ndarray, float, float]:


    
    L0 = 0.0165
    pivot_point = np.array([
    0.8281328220229531, -0.6812731669220016, -0.1,  np.pi, 0.001,0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L0 + 0.25),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    start_point[2] = -0.1
    pose6 = start_point
    # pose6 = np.asarray(get_point(0, 0), dtype=float)
    # pose6 = hw.get_robot_pose_once()
    # pose6 = np.array([0.6464871989117105, -0.5659848620070453, 0.18000000000000002, 2.822035029384029, -1.334168531318762, -0.06004322067724725], float)
    # pose6[2] = -0.1
    # pose6[0] = 0.3
    # print(f"POSE6 is {pose6}")
    T = ur_pose6_to_T(pose6)
    p, q_wxyz = T_to_p_quat_wxyz(T)

    p_now = np.concatenate([p, q_wxyz, [L0]])
    u0 = np.zeros(7, dtype=float)
    hw.send_step(p_now=p_now, u0=u0, dt=0.1)
    # start_point = np.array([
    # 0.665894307606053, -0.7112810117612073, -0.1, np.pi, 0,0
    # ], float)
    # robot_pose6 = hw.get_robot_pose_once()
    # robot_pose6[2] = -0.1

    # start_point = pose6
    print(f"START POINT: {start_point}")
    # L0 = 0.065
    dt = 0.01
    return pivot_point, start_point, L0, dt



def build_controller(
    start_point,
    L0,
    dt,
    forward_model,
    lumen_C,
    lumen_R,
):
    start_point_pose6 = start_point
    p0_pose7 = np.array([
        start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
        start_point_pose6[3], start_point_pose6[4], start_point_pose6[5], L0
    ], dtype=float)

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)

    p_min = np.array([0.2, -1, start_point[2], -np.inf, -np.inf, -np.inf, -np.inf, 0.01])
    p_max = np.array([start_point[0] + 0.5, 1.5, start_point[2], +np.inf, +np.inf, +np.inf, +np.inf, 0.12])

    w_u = np.array([1e-5, 1e-5, 1e-4, 5e-1, 5e-1, 1e-6, 1e-4], dtype=float)
    w_du = np.array([1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8], dtype=float)

    u_max = np.array([1, 1, 1, np.deg2rad(60), np.deg2rad(60), np.deg2rad(360), 0.1], dtype=float)

    dr = 5e-3
    dtheta = np.deg2rad(50.0)
    dL = 1e-3

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        dtheta / dt, dtheta / dt, dtheta / dt,
        dL / dt
    ], dtype=float)


    forward6d_pred = WarmForwardP8TipTangent(copy.deepcopy(forward_model))

    def J_fn(p8):
        forward6d_jac = WarmForwardP8TipTangent(copy.deepcopy(forward_model))
        Jred_state = numerical_J_robot_xy_yaw_dL_warm_branch(
            p8,
            forward6d_jac,
            dx=1e-2,
            dy=1e-2,
            dyaw=3e-1,
            dL=1e-3,
            n_out=6,
        )
        Jred_control = Jred_state.copy()
        Jred_control[:, 0] *= dt
        Jred_control[:, 1] *= dt
        Jred_control[:, 2] *= dt
        Jred_control[:, 3] *= dt
        return J_full_from_robot_reduced_tip_tangent(Jred_control, n_out_full=6)
    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d_pred,
        dt=dt,
        Np=1,
        n_out=6,
        n_u=7,
        n_p=8,
        w_xy=(5.0, 5.0, 0.0, 0.0, 0.0, 0.0),
        w_u=w_u,
        w_du=w_du,
        model_mode="lti",
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        N_sqp=1,
    )

    mpc.R = np.diag([1e-3] * 7)
    mpc.Rd = np.diag([1e-4] * 7)

    mpc.enable_mag_center_standoff = True
    mpc.enable_mag_tangent_inline = False
    mpc.enable_dipole_align = True
    mpc.enable_hard_epm_tip_clearance = True
    mpc.enable_standoff_soft = True
    mpc.enable_inline_soft = True
    mpc.enable_dipole_soft = True
    mpc.enable_hard_theta = False
    mpc.enable_tangent_penalty = False

    mpc.lumen_C = np.asarray(lumen_C, float)
    mpc.lumen_R = np.asarray(lumen_R, float)
    mpc.set_initial_params(p0)
    print(f"Finished building controller")
    return mpc, p0, p_min, p_max, u_max, forward6d_pred

def transform_local_points_to_robot(
    points_local_m,
    pivot_pose6,
    flip_y: bool = False,
):
    """
    Convert lumen points from the vision/local frame into robot/world frame.

    Assumes points_local_m are expressed in a pivot-attached local frame,
    except for the historical image convention where +y must be flipped.
    """
    P_local = np.asarray(points_local_m, dtype=float).copy()
    if P_local.ndim == 1:
        P_local = P_local.reshape(1, 3)

    # Preserve your existing vision sign convention.
    if flip_y:
        P_local[:, 1] *= -1.0

    R_pivot = pivot_rotation_matrix(pivot_pose6)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(1, 3)

    P_robot = (R_pivot @ P_local.T).T + p_pivot
    return P_robot

def pivot_rotation_matrix(pivot_pose6: np.ndarray) -> np.ndarray:
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    return T[:3, :3]

# def transform_local_points_to_robot(points_local_m, pivot_point_pose6):
#     p_pivot_robot = np.asarray(pivot_point_pose6[:3], dtype=float).reshape(3,)
#     P_local = np.asarray(points_local_m, dtype=float)
#     return P_local + p_pivot_robot.reshape(1, 3)

def run_control(
    mpc,
    pivot_point,
    image_filename="focused_image.jpg",
    red_roi_path="/home/jack/Proper-Research/custom_area_w_wall.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    max_steps=100,
    show=False,
    send_commands=False,
    hw=None,
    save_plots=True,
    plot_dir="mpc_debug_plots",
    csv_log_path="control_run_log_test_opti_mid_low_25_mag_opti.csv",
):
    history = []
    csv_rows = []

    for k in range(max_steps):
        print(f"\n================ CONTROL STEP {k} ================")

        new_capture()
        overlay_path = f"debug_outputs_easter/reconstruction_step_{k:04d}.png"
        roi_polygon = load_polygon(red_roi_path)

        vision_result = reconstruct_beam_within_vessel(
            image_filename="focused_image.jpg",
            red_roi_polygon=roi_polygon,
            blue_roi_path="blue_roi_box.json",
            pivot_hint=pivot_hint,
            show=show,
            save_overlay_path=f"debug_outputs_opti_mid_low_25_mag_opti/reconstruction_overlay_step_{k:04d}.png",
        )

        vision_result["lumen_C_robot_m"] = transform_local_points_to_robot(
            vision_result["lumen_C_m"],
            pivot_point
        )

        mpc.lumen_C = np.asarray(vision_result["lumen_C_robot_m"], dtype=float)
        mpc.lumen_R = np.asarray(vision_result["lumen_R_m"], dtype=float)
        forward_pred = mpc.forward_tip_fn   # or whatever object holds the wrapper

        forward_pred.fwd.lumen_C = np.asarray(vision_result["lumen_C_robot_m"], dtype=float)
        forward_pred.fwd.lumen_R = np.asarray(vision_result["lumen_R_m"], dtype=float)
        forward_pred.reset_cache()
        print("[VISION] lumen first point robot =", mpc.lumen_C[0])
        print("[VISION] lumen last point robot  =", mpc.lumen_C[-1])
        print("[VISION] lumen radius min/max [mm] =",
            1000.0 * np.min(mpc.lumen_R),
            1000.0 * np.max(mpc.lumen_R))

        x_meas = vision_result_to_x_meas_robot(
            vision_result,
            pivot_point_pose6=pivot_point,
        )
        x_meas = np.asarray(x_meas, dtype=float).reshape(-1)
        print("same tip x_meas:", x_meas[:3])

        C_live = vision_result.get("lumen_C_robot_m", None)
        if C_live is not None:
            C_live = np.asarray(C_live, float)
            d2_live = np.sum((C_live - x_meas[:3][None, :])**2, axis=1)
            i_live = int(np.argmin(d2_live))
            print("LIVE lumen nearest dist [mm] =", 1000*np.sqrt(d2_live[i_live]))

        C_mpc = np.asarray(mpc.lumen_C, float)
        d2_mpc = np.sum((C_mpc - x_meas[:3][None, :])**2, axis=1)
        i_mpc = int(np.argmin(d2_mpc))
        print("MPC lumen nearest dist [mm] =", 1000*np.sqrt(d2_mpc[i_mpc]))
        print("pivot robot xyz:", pivot_point[:3])
        print("x_meas robot xyz:", x_meas[:3])
        print("distance from pivot [m]:", np.linalg.norm(x_meas[:3] - pivot_point[:3]))
        print("measured tangent robot:", x_meas[3:6])

        print("[VISION] x_meas =", x_meas)
        print("[VISION] tip position norm [m] =", np.linalg.norm(x_meas[:3]))
        print("[VISION] tangent norm =", np.linalg.norm(x_meas[3:6]))
        print("[VISION] lumen_C shape =", mpc.lumen_C.shape)
        print("[VISION] lumen_R shape =", mpc.lumen_R.shape)
        print("[VISION] closest wall distance [mm] =", vision_result["tip_distance_info_mm"]["closest_distance_mm"])
        print("[VISION] left beam-wall angle [deg] =", vision_result["tip_wall_angle_info"]["beam_left_wall_tangent_angle_deg"])
        print("[VISION] right beam-wall angle [deg] =", vision_result["tip_wall_angle_info"]["beam_right_wall_tangent_angle_deg"])
        print("[DBG] i_ref_last before search =", getattr(mpc, "i_ref_last", None))
        print("[DBG] risk_window =", getattr(mpc, "risk_window", None))
        print("[DBG] len(lumen_C) =", len(mpc.lumen_C))

        tip_xyz = x_meas[:3]
        i_ref = closest_index_in_window_monotone(
            mpc.lumen_C,
            tip_xyz,
            int(getattr(mpc, "i_ref_last", 0)),
            window=int(getattr(mpc, "risk_window", 120)),
        )
        C = np.asarray(mpc.lumen_C, float)
        d2_all = np.sum((C - tip_xyz[None, :])**2, axis=1)
        i_ref_global = int(np.argmin(d2_all))
        print("[DBG] global closest index =", i_ref_global)
        print("[DBG] global closest dist [mm] =", 1000*np.sqrt(d2_all[i_ref_global]))

        mpc.i_ref_last = int(i_ref)
        print("[VISION] i_ref =", i_ref)
        print("[VISION] lumen point at i_ref =", mpc.lumen_C[i_ref])
        print("[VISION] tip-to-reference distance [mm] =",
            1000.0 * np.linalg.norm(x_meas[:3] - mpc.lumen_C[i_ref]))

        mag_pos_current = np.array([np.nan, np.nan, np.nan], dtype=float)
        mag_dir_current = np.array([np.nan, np.nan, np.nan], dtype=float)


        if hw is not None:
            L_meas = float(vision_result["beam_length_mm"]) / 1000.0
            robot_pose6 = hw.get_robot_pose_once()
            p_meas8 = build_measured_p8_from_pose6_and_length(
                robot_pose6,
                L_meas,
                z_offset=hw.z_offset,
            )
            mpc.set_measured_params(p_meas8)
            mag_pos_current = np.asarray(p_meas8[:3], dtype=float)
            mag_dir_current = np.asarray(dipole_dir_from_p8(p_meas8), dtype=float)

            print("[MEAS P] robot pose6 =", robot_pose6)
            print("[MEAS P] L_meas =", L_meas)
            print("[MEAS P] p_meas8 =", p_meas8)

        plot_reference_debug_simple(mpc, x_meas, n_ref=10)

        p_now, x_now, info = mpc.step(x_meas=x_meas)

        mag_pos_next = np.asarray(p_now[:3], dtype=float)
        mag_dir_next = np.asarray(dipole_dir_from_p8(p_now), dtype=float)

        plot_path = None
        if save_plots:
            plot_path = os.path.join(plot_dir, f"step_{k:04d}_reference_debug.png")

        plot_reference_debug(
            mpc,
            x_meas,
            info=info,
            n_ref=3,
            show_boundaries=True,
            save_path=plot_path,
            mag_pos=mag_pos_current,
            mag_dir=mag_dir_current,
            mag_pos_next=mag_pos_next,
            mag_dir_next=mag_dir_next,
        )

        p7_now = pose8_quat_to_pose7_rotvec(p_now)
        ur_pose6_next = p7_now[:6]
        L_next = p7_now[6]

        print("UR next pose:", ur_pose6_next)
        print("Insertion next:", L_next)

        u0 = np.asarray(info["u0"], dtype=float).copy()

        print("[MPC] jac cond =", info.get("jac_svd_cond", np.nan))
        print("[MPC] jac rank =", info.get("jac_svd_rank", np.nan))
        print(f"[CONTROL] step={k}")
        print("x_meas =", x_meas)
        print("p_now =", p_now)
        print("x_now =", x_now)
        print("Error on 1 step ", info["pred1_err_xy"])
        print("u0_proposed =", info["u0"])
        print("i_ref =", mpc.i_ref_last)
        print("closest wall distance [mm] =", vision_result["tip_distance_info_mm"]["closest_distance_mm"])
        print("status =", info.get("status"))
        print("infeasible =", info.get("infeasible"))

        left_tan = np.asarray(
            vision_result["tip_wall_angle_info"]["left_wall_tangent_vec_cartesian"],
            dtype=float
        )
        right_tan = np.asarray(
            vision_result["tip_wall_angle_info"]["right_wall_tangent_vec_cartesian"],
            dtype=float
        )

        mag_pos_current = np.array([np.nan, np.nan, np.nan], dtype=float)
        if hw is not None:
            mag_pos_current = np.asarray(p_meas8[:3], dtype=float)

        # use actual predicted tip, not x_now
        X_pred = np.asarray(info["X_pred"], dtype=float)
        X_nom = np.asarray(info["X_nom"], dtype=float).reshape(-1)

        pred_tip = np.asarray(X_pred[0, :3], dtype=float)
        nom_tip = np.asarray(X_nom[:3], dtype=float)

        meas_tip = np.asarray(x_meas[:3], dtype=float)

        err_xyz = meas_tip - pred_tip
        err_xy = np.linalg.norm(err_xyz[:2])

        err_nom_xyz = meas_tip - nom_tip
        err_nom_xy = np.linalg.norm(err_nom_xyz[:2])

        row = {
            "step": int(k),

            "mag_x": float(mag_pos_current[0]),
            "mag_y": float(mag_pos_current[1]),
            "mag_z": float(mag_pos_current[2]),

            "pred_x": float(pred_tip[0]),
            "pred_y": float(pred_tip[1]),
            "pred_z": float(pred_tip[2]),

            "meas_x": float(meas_tip[0]),
            "meas_y": float(meas_tip[1]),
            "meas_z": float(meas_tip[2]),

            "err_xy": float(err_xy),
            "nom_x": float(nom_tip[0]),
            "nom_y": float(nom_tip[1]),
            "nom_z": float(nom_tip[2]),
            "err_nom_xy": float(err_nom_xy),

            "i_ref": int(mpc.i_ref_last),

            "left_wall_tan_x": float(left_tan[0]),
            "left_wall_tan_y": float(left_tan[1]),
            "right_wall_tan_x": float(right_tan[0]),
            "right_wall_tan_y": float(right_tan[1]),

            "dist_left_center_mm": float(vision_result["tip_distance_info_mm"]["dist_left_center_mm"]),
            "dist_right_center_mm": float(vision_result["tip_distance_info_mm"]["dist_right_center_mm"]),
            "dist_left_edge_mm": float(vision_result["tip_distance_info_mm"]["dist_left_edge_mm"]),
            "dist_right_edge_mm": float(vision_result["tip_distance_info_mm"]["dist_right_edge_mm"]),
            "closest_distance_mm": float(vision_result["tip_distance_info_mm"]["closest_distance_mm"]),

            "beam_left_wall_angle_deg": float(vision_result["tip_wall_angle_info"]["beam_left_wall_tangent_angle_deg"]),
            "beam_right_wall_angle_deg": float(vision_result["tip_wall_angle_info"]["beam_right_wall_tangent_angle_deg"]),
        }
        csv_rows.append(row)

        # write CSV each step so you do not lose data if run stops
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

        history.append({
            "k": int(k),
            "x_meas": x_meas.copy(),
            "x_now": np.asarray(x_now, dtype=float).copy(),
            "p_now": np.asarray(p_now, dtype=float).copy(),
            "u0_proposed": u0.copy(),
            "i_ref": int(mpc.i_ref_last),
            "vision_result": vision_result,
            "info": info,
        })

        if send_commands:
            print("[DBG] before hw.send_step")
            hw.send_step(p_now=p_now, u0=u0, dt=mpc.dt)
            print("[DBG] after hw.send_step")

    print(f"[SAVE] CSV log saved to {csv_log_path}")
    return history
def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
    """
    L_ins      : commanded insertion (what MPC tracks)
    L_tip_full : physical magnetic tip length (4 cm)
    L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

    Returns (L_model, wire_len, tip_len)
    """
    L_ins = float(L_ins)

    # Magnetised tip inside grows with insertion until full tip is inside
    tip_len = min(L_ins, L_tip_full)

    # Wire is everything beyond the physical tip length
    wire_len = max(L_ins - L_tip_full, 0.0)

    # Total model length is the inserted length, but don't go below minimum model length
    L_model = max(L_ins, L_tip_min)

    # If we are below L_tip_min, we still model a minimum rod,
    # but magnetisation should NOT exceed what's actually inserted:
    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len
def build_initial_lumen_from_vision(
    pivot_point,
    image_filename="focused_image.jpg",
    red_roi_path="red_roi_box.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    show=True,
):
    new_capture()
    roi_polygon_path="/home/jack/Proper-Research/custom_area.json"
    roi_polygon = load_polygon(roi_polygon_path)
    vision_result = reconstruct_beam_within_vessel(
        image_filename="focused_image.jpg",
        red_roi_polygon=roi_polygon,
        blue_roi_path="blue_roi_box.json",
        pivot_hint=pivot_hint,
        show=show,
        save_overlay_path="debug_outputs_new/reconstruction_overlay2.png",
    )

    lumen_C_robot_m = transform_local_points_to_robot(
        vision_result["lumen_C_m"],
        pivot_point,
    )
    lumen_R_robot_m = np.asarray(vision_result["lumen_R_m"], float)

    vision_result["lumen_C_robot_m"] = lumen_C_robot_m
    vision_result["lumen_R_robot_m"] = lumen_R_robot_m

    return vision_result, lumen_C_robot_m, lumen_R_robot_m

def build_forward_models_from_lumen(pivot_point, L0, lumen_C, lumen_R):
    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    L_tip_full_physical = 0.04
    L_tip_min_physical = 0.01

    L_model, wire_len_model, tip_len_model = effective_lengths(
        L0,
        L_tip_full=L_tip_full_physical,
        L_tip_min=L_tip_min_physical,
    )
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)

    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )

    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )

    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]
    EI_tip = tip["EI"]
    GJ_tip = tip["GJ"]

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, float),
        lumen_R=np.asarray(lumen_R, float),
        N_nodes=10,
        maxiter=1e7,
        L0_init=0.01,
        dL_internal=0.002,
        use_lumen_jac=True,
        L_tip_full=L_tip_full_physical,
        L_tip_min=L_tip_min_physical,
    )

    print("Finished_building_model")
    return p0_ur, q0_ur, forward_model
def make_Kbt_inv_profile(EI_wire, EI_tip, GJ_wire, GJ_tip, bend_soft=1.0, tors_soft=1.0):
    def Kbt_inv_profile(s, len_wire):
        s = np.asarray(s, float)
        mask_tip = (s >= len_wire)

        EI_s = np.where(mask_tip, EI_tip, EI_wire)
        GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)

        Kinv = np.zeros((3, 3, s.size), float)
        Kinv[0, 0, :] = tors_soft / GJ_s
        Kinv[1, 1, :] = bend_soft / EI_s
        Kinv[2, 2, :] = bend_soft / EI_s
        return Kinv

    return Kbt_inv_profile
def rod_section_stiffness(r, E, nu):
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    EA = E * A
    EI = E * I
    GJ = G * J

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": EA,
        "EI": EI,
        "GJ": GJ,
    }
if __name__ == "__main__":
    pivot_hint = (325, 371)
        # 4. dry-run control loop
    hw = LiveHardwareController(
        robot_ip="192.168.56.101",
        dry_run=False,                 # True first
        use_advancer=True,
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=0.28,
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )
    pivot_point, start_point, L0, dt = make_initial_poses_single_use(hw)

    # 1. build lumen once from vision
    vision_init, lumen_C_robot_m, lumen_R_robot_m = build_initial_lumen_from_vision(
        pivot_point=pivot_point,
        image_filename="focused_image.jpg",
        red_roi_path="/home/jack/Proper-Research/custom_area.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",
        pivot_hint=pivot_hint,
        show=False,
    )

    # 2. build forward model from that same lumen
    p0_ur, q0_ur, forward_model = build_forward_models_from_lumen(
        pivot_point=pivot_point,
        L0=L0,
        lumen_C=lumen_C_robot_m,
        lumen_R=lumen_R_robot_m,
    )

    # 3. build controller from same lumen + same forward model
    mpc, p0, p_min, p_max, u_max, forward6d = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        forward_model=forward_model,
        lumen_C=lumen_C_robot_m,
        lumen_R=lumen_R_robot_m,
    )



    try:
        history = run_control(
            mpc=mpc,
            pivot_point=pivot_point,
            image_filename="focused_image.jpg",
            red_roi_path="/home/jack/Proper-Research/custom_area.json",
            blue_roi_path="blue_roi_box.json",
            green_roi_path="green_roi_box.json",
            pivot_hint=pivot_hint,
            max_steps=100,
            show=False,
            send_commands=True,
            hw=hw,
            save_plots=True,
            plot_dir="mpc_debug_plots_opti_mid_low_25_mag_opti",
        )
    finally:
        hw.shutdown()