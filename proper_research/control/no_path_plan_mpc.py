import numpy as np
import osqp
import copy
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import unpack_pose_ur_rotvec_L
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d, quat_wxyz_to_R, make_lumen_centerline_double_turn
from beam_direction_magnetisation.post_processing.results_sim_paper import analyze_run 
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, effective_lengths
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
from scipy.stats import skew
mag_params = default_magnet_params()
beam_params = default_beam_params()
L_MAG = 0.04

from dataclasses import dataclass

import numpy as np

def arc_length_param(C):
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s

def unit(v, eps=1e-12):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)

def closest_point_polyline(C, x):
    """
    Return closest point on polyline C to point x.
    Outputs:
      i_seg: segment index (0..M-2)
      u:     segment parameter in [0,1]
      c:     closest point (3,)
      d2:    squared distance
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3,)
    V = C[1:] - C[:-1]                 # (M-1,3)
    W = x.reshape(1,3) - C[:-1]        # (M-1,3)
    VV = np.sum(V*V, axis=1) + 1e-15
    u = np.sum(W*V, axis=1) / VV
    u = np.clip(u, 0.0, 1.0)
    P = C[:-1] + u.reshape(-1,1)*V
    d2 = np.sum((P - x.reshape(1,3))**2, axis=1)
    i_seg = int(np.argmin(d2))
    return i_seg, float(u[i_seg]), P[i_seg], float(d2[i_seg])

def tnb_from_centerline(C, i_seg, t_prev=None):
    """
    Build a consistent-ish TNB frame at segment i_seg using local tangent and a propagated normal.
    - t = unit(C[i+1]-C[i])
    - n chosen to be perpendicular to t, consistent with previous n if provided
    - b = t x n
    """
    C = np.asarray(C, float)
    t = unit(C[i_seg+1] - C[i_seg])

    if t_prev is not None and np.dot(t, t_prev) < 0.0:
        t = -t  # keep tangent direction consistent

    # pick a normal: use previous if available; otherwise choose any vector not parallel to t
    if t_prev is None or t_prev is None:
        pass

    if t_prev is None:
        # choose a reference axis not parallel to t
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n = unit(ref - np.dot(ref, t)*t)
    else:
        # propagate normal: remove component along t
        n = t_prev  # (not correct: we need previous n; so pass prev_n in practice)
        # If you have prev_n, use it instead.

    # If you want true propagation: pass prev_n. Here's a safer approach:
    # We'll just recompute from a fixed ref each time if you don't pass prev_n.
    b = unit(np.cross(t, n))
    n = unit(np.cross(b, t))
    return t, n, b

def tip_in_vessel_frame(x_tip, C, R, s_path, t_prev=None, n_prev=None):
    """
    Returns dict with:
      idx_seg, u, s, c (closest point), t,n,b, x_perp,y_perp,rho, clearance, R_here
    """
    i_seg, u, c, d2 = closest_point_polyline(C, x_tip)

    # arc-length at closest point
    s = float(s_path[i_seg] + u * (s_path[i_seg+1] - s_path[i_seg]))

    # tangent
    t = unit(C[i_seg+1] - C[i_seg])
    if t_prev is not None and np.dot(t, t_prev) < 0.0:
        t = -t

    # normal/binormal: propagate prev normal if given; otherwise build from fixed reference
    if n_prev is None:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        n = unit(ref - np.dot(ref, t)*t)
    else:
        n = unit(n_prev - np.dot(n_prev, t)*t)
        if np.linalg.norm(n) < 1e-9:
            # fallback
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ref, t)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            n = unit(ref - np.dot(ref, t)*t)

    b = unit(np.cross(t, n))
    n = unit(np.cross(b, t))  # re-orthonormalize

    r = np.asarray(x_tip, float).reshape(3,) - np.asarray(c, float).reshape(3,)
    x_perp = float(np.dot(n, r))
    y_perp = float(np.dot(b, r))
    rho = float(np.hypot(x_perp, y_perp))

    # radius at this segment (linear interp if you want; simplest: take R[i_seg])
    R_here = float(R[i_seg]) if np.ndim(R) > 0 else float(R)
    clearance = float(R_here - rho)

    return dict(
        idx_seg=i_seg, u=u, s=s, c=c, t=t, n=n, b=b,
        x_perp=x_perp, y_perp=y_perp, rho=rho, clearance=clearance, R=R_here
    )
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

def fmt_vec(v, prec=4):
    v = np.asarray(v).ravel()
    return "[" + ",".join([f"{x:+.{prec}f}" for x in v]) + "]"

def assert_finite(name, arr):
    arr = np.asarray(arr)
    if not np.all(np.isfinite(arr)):
        dbg_print(0, f"[FATAL] {name} has non-finite values.")
        return False
    return True
def quat_wxyz_normalize(qwxyz):
    q = np.asarray(qwxyz, float).copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n
def nearest_index_in_window(path, x, i_ref, window=80):
    """Closest index to x within [i_ref, i_ref+window)."""
    M = path.shape[0]
    i_lo = int(np.clip(i_ref, 0, M-1))
    i_hi = int(min(M, i_ref + window))
    seg = path[i_lo:i_hi]
    if seg.shape[0] == 0:
        return M - 1
    d2 = np.sum((seg - x.reshape(1, 3))**2, axis=1)
    return i_lo + int(np.argmin(d2))

def _angle_deg(u, v):
    u = np.asarray(u, float).ravel()
    v = np.asarray(v, float).ravel()
    if u.size != 3 or v.size != 3:
        raise ValueError(f"_angle_deg expects 3-vectors; got u.shape={u.shape}, v.shape={v.shape}")
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    if un < 1e-12 or vn < 1e-12:
        return np.nan
    c = float(np.clip(np.dot(u, v) / (un * vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))
def horizon_pred_errors(X_nl, X_pred, ph=None):
    """
    X_nl, X_pred: (Np,6) arrays (pos xyz + tangent tx ty tz)
    Returns:
      e_pos_m: (ph,) position L2 error in meters
      e_tan_deg: (ph,) tangent angle error in degrees
    """
    X_nl = np.asarray(X_nl, float)
    X_pred = np.asarray(X_pred, float)
    Np = min(X_nl.shape[0], X_pred.shape[0])
    ph = Np if ph is None else min(int(ph), Np)

    e_pos = np.zeros(ph, float)
    e_tan = np.zeros(ph, float)
    for i in range(ph):
        e_pos[i] = np.linalg.norm(X_nl[i, 0:3] - X_pred[i, 0:3])
        e_tan[i] = _angle_deg(X_nl[i, 3:6], X_pred[i, 3:6])
    return e_pos, e_tan
def update_progress_cursor_s(
    C, s_path, x_prev, x_now, i_ref,
    window=120,
    s_advance=0.5e-3,        # meters: how much Δs counts as progress
    stall_steps=8,
    force_advance_pts=1,
    dist_ok_max=0.010,       # meters: don't force-advance if far from centerline
    state=None,
):
    """
    Option B: progress is Δs along centerline (arc-length), using monotone projection.

    Returns: (i_ref_new, state, dbg)
      dbg has: i_seg_now, s_prev, s_now, prog, d_now
    """
    if state is None:
        state = {"stall": 0, "s_last": None}

    M = C.shape[0]
    i_ref = int(np.clip(i_ref, 0, M-2))

    # Project previous and current to arc-length
    s_prev, i_seg_prev, _, d2_prev = project_to_polyline_s_monotone(
        C, s_path, x_prev, i_ref, window=window
    )
    # For monotonicity, start the "now" search from max(i_ref, i_seg_prev)
    i_start_now = max(i_ref, i_seg_prev)
    s_now, i_seg_now, lam_now, d2_now = project_to_polyline_s_monotone(
        C, s_path, x_now, i_start_now, window=window
    )

    prog = float(s_now - s_prev)   # meters of forward progress along centerline

    # Cursor update: follow the segment we are currently closest to, monotone
    i_ref_new = max(i_ref, i_seg_now)

    progressed = (prog > float(s_advance))
    if progressed:
        state["stall"] = 0
    else:
        state["stall"] += 1

    # distance sanity check (use sqrt(d2_now))
    d_now = float(np.sqrt(d2_now))
    dist_ok = (d_now < float(dist_ok_max))

    # force advance when stalled, but only if localized (dist_ok)
    forced = False
    if state["stall"] >= int(stall_steps):
        if dist_ok:
            i_ref_new = int(min(M-2, i_ref_new + int(force_advance_pts)))
            forced = True
        state["stall"] = 0

    dbg = dict(
        i_seg_now=int(i_seg_now),
        s_prev=float(s_prev),
        s_now=float(s_now),
        prog=float(prog),
        d_now=float(d_now),
        forced=bool(forced),
        stall=int(state["stall"]),
    )
    return int(i_ref_new), state, dbg
def unit(v, eps=1e-12):
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

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
def closest_index_in_window_monotone(C, x, i_start, window=120):
    """
    Like nearest_index_in_window but monotone forward w.r.t. i_start.
    C: (M,3)
    x: (3,)
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3,)
    M = C.shape[0]
    i_lo = int(np.clip(i_start, 0, M-1))
    i_hi = int(min(M, i_lo + int(window)))
    seg = C[i_lo:i_hi]
    if seg.shape[0] == 0:
        return M - 1
    d2 = np.sum((seg - x.reshape(1,3))**2, axis=1)
    return i_lo + int(np.argmin(d2))


def wall_margin_and_gate(C, R, x, i_idx, delta=5e-4, sigma_m=5e-4):
    """
    Computes cross-sectional wall margin m and gate g at centerline index i_idx
    for a point x (3,).

    m = (R - delta) - d_perp, with d_perp computed w.r.t. vessel tangent at i_idx.
    g = sigmoid((-m)/sigma_m)  ~ 1 when near/over wall, ~0 when deep inside.
    """
    C = np.asarray(C, float)
    R = np.asarray(R, float).ravel()
    x = np.asarray(x, float).reshape(3,)

    t = centerline_tangent(C, i_idx)  # vessel tangent
    c = C[i_idx]
    r = x - c
    r_perp = r - (r @ t) * t
    d_perp = float(np.linalg.norm(r_perp))
    m = float((R[i_idx] - delta) - d_perp)
    g = sigmoid((-m) / float(max(sigma_m, 1e-12)))
    return m, g, t


def predictive_risk_along_horizon(
    Y_seq,            # (Np,n) predicted outputs, expects n>=6: [x(3), t(3), ...]
    lumen_C, lumen_R, # (M,3), (M,)
    i_ref,            # current cursor index
    window=120,
    delta=5e-4,
    sigma_m=5e-4,
    theta_crit_deg=40.0,
):
    """
    For each predicted step k:
      - map x_k to centerline index i_k (monotone forward from i_ref)
      - compute wall gate g_k from margin
      - compute misalignment theta_k between tip tangent t_k and vessel tangent at i_k

    Returns dict with arrays length Np:
      idx_k, margin_k, g_k, theta_deg_k, t_vessel_k
    """
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()

    Y_seq = np.asarray(Y_seq, float)
    Np = Y_seq.shape[0]
    if Y_seq.shape[1] < 6:
        raise ValueError("predictive_risk_along_horizon expects Y_seq with at least 6 dims: [x(3), t(3)].")

    idx_k = np.zeros(Np, dtype=int)
    margin_k = np.zeros(Np, dtype=float)
    g_k = np.zeros(Np, dtype=float)
    theta_deg_k = np.zeros(Np, dtype=float)
    t_vessel_k = np.zeros((Np, 3), dtype=float)

    i_last = int(i_ref)

    for k in range(Np):
        xk = Y_seq[k, 0:3]
        tk = unit(Y_seq[k, 3:6])

        # monotone forward mapping
        ik = closest_index_in_window_monotone(C, xk, i_last, window=window)
        i_last = ik

        m, g, tv = wall_margin_and_gate(C, R, xk, ik, delta=delta, sigma_m=sigma_m)

        # angle between predicted tip tangent and local vessel tangent
        cang = float(np.clip(np.dot(unit(tk), unit(tv)), -1.0, 1.0))
        th = float(np.arccos(cang))
        th_deg = float(np.rad2deg(th))

        idx_k[k] = ik
        margin_k[k] = m
        g_k[k] = g
        theta_deg_k[k] = th_deg
        t_vessel_k[k, :] = tv

    return dict(
        idx_k=idx_k,
        margin_k=margin_k,
        g_k=g_k,
        theta_deg_k=theta_deg_k,
        t_vessel_k=t_vessel_k,
        theta_crit_deg=float(theta_crit_deg),
    )

def _angle_deg(u, v, eps=1e-12):
    u = np.asarray(u, float).reshape(3,)
    v = np.asarray(v, float).reshape(3,)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un < eps or vn < eps:
        return np.nan
    c = float(np.clip(np.dot(u/un, v/vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _closest_centerline_tangent(Cc, x):
    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x)
    i_seg = int(np.clip(i_seg, 0, Cc.shape[0]-2))
    t = Cc[i_seg+1, :3] - Cc[i_seg, :3]     # segment tangent at closest point
    t /= (np.linalg.norm(t) + 1e-12)
    return i_seg, float(u_seg), c_closest.reshape(3,), t
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
def forward_y_det(p8):
    import copy
    last = copy.deepcopy(forward_model._last)
    last_centerline = None if forward_model.last_p_centerline is None else forward_model.last_p_centerline.copy()
    last_tip = None if forward_model.last_tip is None else forward_model.last_tip.copy()
    last_info = copy.deepcopy(getattr(forward_model, "last_info", None))
    last_hist = getattr(forward_model, "last_hist", None)

    y = forward_y_live(p8)

    forward_model._last = last
    forward_model.last_p_centerline = last_centerline
    forward_model.last_tip = last_tip
    forward_model.last_info = last_info
    forward_model.last_hist = last_hist
    return y
def quat_wxyz_mul(q1, q2):
    # (w,x,y,z) ⊗ (w,x,y,z)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], float)

def rotvec_to_quat_wxyz(rvec):
    r = Rot.from_rotvec(np.asarray(rvec, float))
    q_xyzw = r.as_quat()  # [x,y,z,w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)

def quat_wxyz_to_rotvec(qwxyz):
    qw, qx, qy, qz = quat_wxyz_normalize(qwxyz)
    r = Rot.from_quat([qx, qy, qz, qw])  # xyzw
    return r.as_rotvec()

def small_rot_quat_wxyz(dphi):
    # dphi is a small rotation vector in radians (axis*angle)
    dphi = np.asarray(dphi, float).ravel()
    a = np.linalg.norm(dphi)
    if a < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    axis = dphi / a
    half = 0.5 * a
    return np.array([np.cos(half), *(np.sin(half) * axis)], float)

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

def wire_len_from_L(L, L_mag=L_MAG):
    L = float(L)
    wire_len = L - float(L_mag)
    # if wire_len <= 0:
    #     # if beam shorter than magnet segment, clamp or error
    #     raise ValueError(f"L={L:.4f} must be > L_mag={L_mag:.4f}")
    return wire_len



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
def build_Pm_body(p_seq, dt, Np, m=7):
    Pm = np.zeros((3*Np, m*Np), float)
    for k in range(Np):
        for j in range(k+1):
            q = np.asarray(p_seq[j][3:7], float)  # qw,qx,qy,qz
            Rj = quat_wxyz_to_R(q)                # your helper, (3,3)
            Pm[3*k:3*k+3, m*j:m*j+3] += dt * Rj   # only vx,vy,vz columns
    return Pm
def pos_row_idx(n, Np):
    idx = []
    for k in range(Np):
        idx += [k*n + 0, k*n + 1, k*n + 2]
    return np.array(idx, dtype=int)
def forward_tnb(Cc, idx, t_prev=None, n_prev=None):
    """
    Returns a robust (t,n,b) frame at centerline index idx.
    - t uses your forward_tangent convention (no sign flips backward)
    - n from dt, with continuity fallback
    """
    t = forward_tangent(Cc, int(idx), t_prev)
    t = t / (np.linalg.norm(t) + 1e-12)

    # tangent ahead for curvature
    i = int(np.clip(idx, 0, Cc.shape[0]-2))
    t1 = forward_tangent(Cc, min(i+1, Cc.shape[0]-2), t)
    t1 = t1 / (np.linalg.norm(t1) + 1e-12)

    dn = t1 - t
    if np.linalg.norm(dn) < 1e-6:
        # low curvature: keep previous normal if available
        if n_prev is not None and np.linalg.norm(n_prev) > 1e-9:
            n = n_prev.copy()
        else:
            # pick any vector not parallel to t
            a = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, t)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            n = a - np.dot(a, t) * t
            n /= (np.linalg.norm(n) + 1e-12)
    else:
        n = dn / (np.linalg.norm(dn) + 1e-12)

    # enforce continuity (avoid sign flips)
    if n_prev is not None and float(np.dot(n, n_prev)) < 0.0:
        n = -n

    b = np.cross(t, n)
    nb = np.linalg.norm(b)
    if nb < 1e-9:
        # degenerate; fallback binormal orthogonal to t
        if n_prev is not None:
            b = np.cross(t, n_prev)
        if np.linalg.norm(b) < 1e-9:
            # final fallback
            a = np.array([0.0, 0.0, 1.0])
            b = np.cross(t, a)
        b /= (np.linalg.norm(b) + 1e-12)
        n = np.cross(b, t)
        n /= (np.linalg.norm(n) + 1e-12)
    else:
        b = b / nb

    return t, n, b
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
def Mc_pos_stage(Mc, n, m, Np, k):
    # rows for x,y,z at stage k in stacked output [y1..yNp]
    rows = np.array([k*n + 0, k*n + 1, k*n + 2], dtype=int)
    return Mc[rows, :]  # (3, m*Np)
class mpc_controller_tipxy_LTI:
    def __init__(self, *, Jxy_fn, forward_tip_fn,
                 dt=0.05, Np=10,
                 w_xy=(50.0, 50.0, 50.0),
                 w_u=None,
                 w_du=None,
                 u_max=None,
                 p_min=None,
                 p_max=None,
                 N_sqp=3,
                 use_offset_free=True,
                 n_out = 3,
                 n_p=8,
                 n_u=7,
                 model_mode = "ltv",
                 d_min_tip_mag=0.10,  
                 enable_tip_keepout=False,
                 d_alpha=0.15,
                 overhead_magnet = False,
                 w_mag = 1,
                 ):
        self.mode = "full"  # "full" (your current) or "centerline_only"
        self.np = int(n_p)     # pose dimension
        self.m  = int(n_u)     # control dimension (still 7)
        self.n = n_out
        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

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
        self.use_offset_free = bool(use_offset_free)
        self.d_alpha = float(d_alpha)

        # matrices
        self._rebuild_S()
        self.Du = self._build_Du_matrix()
        self.d_min_tip_mag = float(d_min_tip_mag)
        self.enable_tip_keepout = bool(enable_tip_keepout)
        self.p_last_meas = None
        self.x_last_meas = None

        # --- adaptive Q gating params ---
        self.enable_adaptive_Q = True
        self.sigma_m = 5e-4           # margin softness (m)
        self.delta_wall = 1e-3        # safety margin (m)
        # --- predictive wall/tangent shaping ---
        self.risk_window = 120          # how far ahead to search on centerline for predicted mapping
        self.theta_crit_deg = 40.0

        # # per-step tangent inflation with g_k
        # self.q_tan_scale = 0        # you already have this
        # self.q_pos_scale = 0          # optional
        # --- NEW: contact-based reweighting (only active near wall) ---
        self.q_tan_gain = 100        # tangent inflation at wall (try 10–100)
        self.q_pos_drop = 0         # fraction to drop position weight at wall (0..0.95)
        self.q_gate_pow = 2.0         # make it kick in mostly near contact (1..4)
        self.w_adv = 5e-5     # start tiny (1e-5 .. 1e-3)
        self.w_adv_gate_pow = 0  # optional: reduce reward near wall
        # baseline multipliers (keep 1.0 unless you want global scaling)
        self.q_pos_base = 1.0
        self.q_tan_base = 0.001
        self.debug = True
        self._dbg_last = {}
        self.overhead_magnet = overhead_magnet
        # __init__
        self.w_mag_xy = float(w_mag)
        self.w_adv_base = float(self.w_adv)
        self.w_adv_eff = float(self.w_adv)   # can be overridden per-step
        self.s_min_progress = 0.0            # meters, 0 disables hard constraint
        self.s_des = 5e-4  
        self.enable_soft_wall = True
        self.enable_soft_progress = True
        self.w_slack_wall = 1e5      # tune
        self.w_slack_prog = 1e3      # tune


        self.mag_center_use_pred_idx = True     # use risk idx_k (monotone) if available
        self.enable_dipole_align = True
        self.w_dipole_align = 1        # start small: 0.1..10
        self.dipole_body_axis = np.array([1.0, 0.0, 0.0])  # or [0,0,1]

        self.enable_mag_center_standoff = True
        self.w_mag_center_standoff = 20
        self.mag_center_standoff_m = 0.12
        self.dL_back_max = 0.002      # or 0.001 if small pullback allowed
        self.dL_fwd_max  = np.inf   # or some finite cap (per-step dL rate)
        self.enable_mag_inline_centerline = True
        self.w_mag_inline_centerline = 1e2  # start here; tune 1e-4..1e-2
        self.w_slack_wall_min = 1e2       # keep small baseline for conditioning
        self.w_slack_wall_max = 1e5       # your current strong penalty
        self.theta_gate_band_deg = 5.0
        self.wall_gate_pow = 2.0
        self.theta_gate_pow = 2.0
        from collections import deque
        self._epm_hist_W = 15
        self._epm_aligned_hist = deque(maxlen=self._epm_hist_W)
        self._epm_bad_hist     = deque(maxlen=self._epm_hist_W)
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
    def _debug_objective_breakdown(
        self, *, Z_opt, H_z, f_z, Nu, ns_wall, ns_prog,
        Rtil=None, H_du=None,
        w_slack_wall=None, w_slack_prog=None,
        w_adv_eff=None, g_adv=None,
        overhead_terms=None,   # dict with keys {"H_add","f_add"} if you want
    ):
        """
        Prints per-component objective values at the QP solution.
        Assumes objective used in OSQP is: 0.5 z^T H_z z + f_z^T z
        """
        z = np.asarray(Z_opt, float).reshape(-1, 1)
        H = np.asarray(H_z, float)
        f = np.asarray(f_z, float).reshape(-1, 1)

        # total as OSQP sees it
        J_total = float((0.5 * (z.T @ H @ z) + (f.T @ z)).item())

        U = z[:Nu, :]
        s_wall = z[Nu:Nu+ns_wall, :] if ns_wall > 0 else None
        s_prog = z[Nu+ns_wall:Nu+ns_wall+ns_prog, :] if ns_prog > 0 else None

        out = {}
        out["J_total"] = J_total

        # Input cost pieces (if you pass Rtil and H_du)
        if Rtil is not None:
            Rt = np.asarray(Rtil, float)
            out["J_u"] = float(U.T @ Rt @ U)
        if H_du is not None and not np.isscalar(H_du):
            Hd = np.asarray(H_du, float)
            out["J_du"] = float(U.T @ Hd @ U)

        # Slack costs (these are clean and very interpretable)
        if ns_wall > 0 and w_slack_wall is not None:
            out["J_wall_slack"] = float(w_slack_wall * (s_wall.T @ s_wall))
            out["wall_slack_max_mm"] = 1e3 * float(np.max(s_wall))
            out["wall_slack_rms_mm"] = 1e3 * float(np.sqrt(np.mean(s_wall**2)))
        if ns_prog > 0 and w_slack_prog is not None:
            out["J_prog_slack"] = float(w_slack_prog * (s_prog.T @ s_prog))
            out["prog_slack_max_mm"] = 1e3 * float(np.max(s_prog))
            out["prog_slack_rms_mm"] = 1e3 * float(np.sqrt(np.mean(s_prog**2)))

        # Progress reward (linear term)
        if (w_adv_eff is not None) and (g_adv is not None):
            g = np.asarray(g_adv, float).reshape(Nu, 1)
            out["J_adv_reward"] = float(-(w_adv_eff * (g.T @ U)))  # negative = "reward"

        # Optional overhead magnet contribution (if you want)
        if overhead_terms is not None:
            H_add = overhead_terms.get("H_add", None)
            f_add = overhead_terms.get("f_add", None)
            if H_add is not None:
                out["J_overhead_quad"] = float(U.T @ np.asarray(H_add, float) @ U)
            if f_add is not None:
                out["J_overhead_lin"] = float(np.asarray(f_add, float).reshape(1, -1) @ U)

        # Print nicely
        print("   [OBJ] total(OSQP) =", out["J_total"])
        for k in ["J_u","J_du","J_wall_slack","J_prog_slack","J_adv_reward","J_overhead_quad","J_overhead_lin"]:
            if k in out:
                print(f"   [OBJ] {k:>14s} = {out[k]: .6e}")
        for k in ["wall_slack_max_mm","wall_slack_rms_mm","prog_slack_max_mm","prog_slack_rms_mm"]:
            if k in out:
                print(f"   [OBJ] {k:>14s} = {out[k]: .4f}")

    def set_dt(self, dt):
        self.dt = float(dt)
        self._rebuild_S()
        self.Du = self._build_Du_matrix()

    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.np,)
        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()
            self.x = np.asarray(self.forward_tip_fn(self.p, commit=False), float).reshape(self.n,)
        else:
            self.x = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
        self.d = np.zeros(self.n, float)
        self.U_warm = None

    def _build_Qtil_from_Qseq(self, Q_seq, Qf=None):
        """
        Q_seq: list length Np-1 of (n x n) stage costs
        Qf: (n x n) terminal cost
        """
        n = self.n
        Np = self.Np
        Qtil = np.zeros((Np*n, Np*n))
        for k in range(Np-1):
            Qtil[k*n:(k+1)*n, k*n:(k+1)*n] = Q_seq[k]
        Qtil[(Np-1)*n:, (Np-1)*n:] = Qf if Qf is not None else self.Q
        return Qtil
    def _disturbance_stack(self, d):
        n, Np = self.n, self.Np
        d = np.asarray(d, float).reshape(n, 1)         # (n,1)
        return np.tile(d, (Np, 1)) 
 



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
        """
        Returns: p_seq, Mx, Mc, B0
        - p_seq: (Np,m) predicted p (only meaningful in LTV; in LTI it's still returned for convenience)
        - Mx, Mc: stacked prediction matrices
        - B0: first-step input matrix (for terminal cost DARE)
        """
        n, m, Np = self.n, self.m, self.Np

        if self.model_mode == "lti":
            # J0 = np.asarray(self.Jxy_fn(p0), dtype=float)
            # if J0.shape != (n, m):
            #     raise ValueError(f"Jacobian must be {(n,m)} but got {J0.shape}")
            # B0 = self.dt * J0
            J0 = np.asarray(self.Jxy_fn(p0), float)   # now returns B directly (n x 7)
            B0 = J0
            Mx, Mc = seq_mat_lti(self.A, B0, Np)
            # col_names = ["vx","vy","vz","wx","wy","wz","dL"]
            # row_names = ["x","y","z"]  # only if n_out==3 and it's xyz

            # for r in range(B0.shape[0]):
            #     row_label = row_names[r] if r < len(row_names) else f"y{r}"
            #     vals = " ".join([f"{B0[r,c]:+10.3e}" for c in range(B0.shape[1])])
            #     print(f"  {row_label}: {vals}")
            # print("      cols:", " ".join([f"{n:>10s}" for n in col_names]))
            if U_guess is None:
                U_guess = np.zeros((Np, m))
            p_seq = self._p_seq_from_U(p0, U_guess)

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

    def _compute_tracking_and_centerline_debug(self, *, X_pred, X_ref, X_nom, n, Np, Cc, idx_ref, x_now=None):
        """
        Returns per-stage and summary tracking/alignment metrics for the predicted horizon.
        All distances in meters.
        """
        out = {}

        if X_pred is None or not np.all(np.isfinite(X_pred)):
            return out

        X_pred = np.asarray(X_pred, float).reshape(Np, n)
        X_ref  = np.asarray(X_ref,  float).reshape(Np*n, 1)

        # stage-wise reference positions (from stacked X_ref)
        ref_pos = np.zeros((Np, 3), float)
        for k in range(Np):
            ref_pos[k, :] = X_ref[k*n:k*n+3, 0]

        pred_pos = X_pred[:, :3]

        # predicted tracking error to reference (what optimizer is targeting)
        e_pred = pred_pos - ref_pos
        e_pred_norm = np.linalg.norm(e_pred, axis=1)

        out["pred_ref_err_xyz_m"] = e_pred_norm.copy()
        out["pred_ref_err_rms_m"] = float(np.sqrt(np.mean(e_pred_norm**2)))
        out["pred_ref_err_max_m"] = float(np.max(e_pred_norm))

        # Optional: nominal nonlinear (linearization point) vs ref
        if X_nom is not None:
            X_nom = np.asarray(X_nom, float).reshape(Np, n)
            e_nom = X_nom[:, :3] - ref_pos
            e_nom_norm = np.linalg.norm(e_nom, axis=1)
            out["nom_ref_err_xyz_m"] = e_nom_norm.copy()
            out["nom_ref_err_rms_m"] = float(np.sqrt(np.mean(e_nom_norm**2)))

        # Centreline alignment metrics wrt idx_ref[k]
        # (radial offset from centreline point, plus along-track error relative to local tangent)
        Cc = np.asarray(Cc, float)
        idx_ref = np.asarray(idx_ref, int).reshape(Np,)

        rho = np.zeros(Np, float)        # radial distance to local centreline frame
        e_tan = np.zeros(Np, float)      # along-track signed error wrt centreline point
        clearance = np.full(Np, np.nan)  # if lumen_R exists
        t_prev = None

        Rr = getattr(self, "lumen_R", None)
        if Rr is not None:
            Rr = np.asarray(Rr, float)

        for k in range(Np):
            i = int(np.clip(idx_ref[k], 0, Cc.shape[0]-1))
            c = Cc[i, :3]
            x = pred_pos[k, :]

            # tangent
            if i < Cc.shape[0]-1:
                t = Cc[i+1] - Cc[i]
            else:
                t = Cc[i] - Cc[i-1]
            t = t / (np.linalg.norm(t) + 1e-12)
            if t_prev is not None and np.dot(t, t_prev) < 0:
                t = -t
            t_prev = t.copy()

            r = x - c
            e_tan[k] = float(np.dot(r, t))
            r_perp = r - e_tan[k] * t
            rho[k] = float(np.linalg.norm(r_perp))

            if Rr is not None and Rr.size > 0:
                Ri = float(Rr[i] if Rr.ndim > 0 else Rr)
                clearance[k] = Ri - rho[k]

        out["pred_centerline_rho_m"] = rho.copy()
        out["pred_centerline_rho_rms_m"] = float(np.sqrt(np.mean(rho**2)))
        out["pred_centerline_rho_max_m"] = float(np.max(rho))
        out["pred_centerline_etan_m"] = e_tan.copy()
        out["pred_centerline_etan_rms_m"] = float(np.sqrt(np.mean(e_tan**2)))

        if np.any(np.isfinite(clearance)):
            out["pred_clearance_m"] = clearance.copy()
            out["pred_clearance_min_m"] = float(np.nanmin(clearance))

        # current measured/estimated tip alignment to first target (useful at step level)
        if x_now is not None:
            x_now = np.asarray(x_now, float).reshape(-1)
            if x_now.size >= 3:
                out["xnow_ref0_err_xyz_m"] = float(np.linalg.norm(x_now[:3] - ref_pos[0]))

        return out
    def _eval_qp_terms(self, z, Nu, dbg_terms, ns_wall=0, ns_prog=0):
        """
        Evaluate individual U-space objective terms at solution z.
        Returns dict of scalar contributions for each named term.
        Assumes terms are stored as (H_term, f_term) acting on U only.
        """
        out = {}
        if dbg_terms is None:
            return out

        z = np.asarray(z, float).reshape(-1, 1)
        U = z[:Nu, :]

        for name, val in dbg_terms.items():
            if not isinstance(val, tuple):
                continue
            Ht, ft = val
            Ht = np.asarray(Ht, float)
            ft = np.asarray(ft, float).reshape(-1, 1)
            # objective contribution in OSQP form: 0.5 U^T H U + f^T U
            J = float(0.5 * (U.T @ Ht @ U) + (ft.T @ U))
            out[f"J_{name}"] = J

        # slack contributions (z-space)
        if ns_wall > 0:
            s_wall = z[Nu:Nu+ns_wall, :]
            out["wall_slack_max_mm"] = 1e3 * float(np.max(s_wall))
            out["wall_slack_rms_mm"] = 1e3 * float(np.sqrt(np.mean(s_wall**2)))
        if ns_prog > 0:
            s_prog = z[Nu+ns_wall:Nu+ns_wall+ns_prog, :]
            out["prog_slack_max_mm"] = 1e3 * float(np.max(s_prog))
            out["prog_slack_rms_mm"] = 1e3 * float(np.sqrt(np.mean(s_prog**2)))

        return out
    def step(self, x_meas=None):
        """
        MPC step with Option A (soft lumen constraints via slacks).

        Assumptions (consistent with your codebase):
        - outputs y are n-dim, typically n=6: [x,y,z, tx,ty,tz]
        - linear prediction uses X = X_aff + Mc U (stacked over horizon)
        - U decision is (Np*m,)
        - Optional soft lumen constraints introduce slacks s_k >= 0, one per stage k
            and enforce: m_k(U) + s_k >= 0 (approx via linearized margin)
        """

        # -----------------------
        # Helpers (local)
        # -----------------------
        def _pos_row_idx(n, Np):
            # stacked outputs are [y1,y2,...,yNp], each yk length n
            return np.array([k*n + i for k in range(Np) for i in (0, 1, 2)], dtype=int)

        def _Mc_pos_stage(Mc, n, m, Np, k):
            # return 3 x (Np*m) block mapping U-> position at stage k
            rows = np.array([k*n + 0, k*n + 1, k*n + 2], dtype=int)
            return Mc[rows, :]

        def _pad_A(Au, nz, Nu):
            """Pad a U-only constraint matrix Au (nr x Nu) to (nr x nz)."""
            Au = np.asarray(Au, float)
            if nz == Nu:
                return Au
            nr = Au.shape[0]
            Az = np.zeros((nr, nz), float)
            Az[:, :Nu] = Au
            return Az

        # -----------------------
        # Step start / measurement
        # -----------------------
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        # freeze forward baseline once per MPC step (important with your deterministic wrapper)
        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()

        p_prev = self.p.copy()
        x_prev = self.x.copy()

        # measurement update
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(self.n,)

        # offset-free disturbance update
        if self.use_offset_free and x_meas is not None:
            try:
                x_model = np.asarray(self.forward_tip_fn(self.p, commit=False), float).reshape(self.n,)
            except TypeError:
                x_model = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
            r = self.x - x_model
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        xk = self.x.reshape(n, 1)
        centerline_only = (getattr(self, "mode", "full") == "centerline_only")

        # Effective toggles (baseline disables everything except tracking+effort)
        enable_soft_wall_eff     = (self.enable_soft_wall     and (not centerline_only))
        enable_soft_progress_eff = (self.enable_soft_progress and (not centerline_only))

        enable_adv_eff      = ((self.w_adv != 0.0) and (not centerline_only))
        enable_standoff_eff = (self.enable_mag_center_standoff)
        enable_inline_eff   = (getattr(self, "enable_mag_tangent_inline", True))
        enable_dipole_eff   = (self.enable_dipole_align )

        # Also (optional) disable your dL directional constraint for baseline if you truly want “no constraints”
        # baseline_no_dL_bound = True
        baseline_no_dL_bound = False
        # Warm start / SQP init
        if self.U_warm is not None and self.U_warm.size == Np*m:
            U_opt_vec = self.U_warm.copy()
            U_guess = U_opt_vec.reshape(Np, m)
        else:
            U_guess = np.zeros((Np, m))
            U_opt_vec = None

        status_last = "init"

        # debug caches
        Mc_last = None
        X_aff_last = None
        X_nom_last = None
        p_seq_last = None
        B_first = None
        p_lin = None
        p_first = None
        dbg_terms = {} if self.debug else None
        # -----------------------
        # SQP loop
        # -----------------------
        for it in range(int(self.N_sqp)):

            # ---- linearization / prediction matrices ----
            p0 = self.p.copy()
            p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)
            
            if it % 1 == 0:
                # ---- DEBUG: print Jacobian / B0 ----
                B = np.asarray(B0, float)
                n, m = B.shape
                umax = np.asarray(self.u_max, float)

                # bound on |Δy_i| ≈ sum_j |B_ij| * umax_j
                delta_max = np.sum(np.abs(B) * umax.reshape(1,-1), axis=1)

                print("max |Δx| per step (mm):", 1e3*delta_max[0])
                print("max |Δy| per step (mm):", 1e3*delta_max[1])
                print("max |Δz| per step (mm):", 1e3*delta_max[2])
                col_names = ["vx","vy","vz","wx","wy","wz","dL"]
                row_names = [f"y{i}" for i in range(n)]
                if n >= 6:
                    row_names = ["x","y","z","tx","ty","tz"] + [f"y{i}" for i in range(6, n)]

                print(f"\n[DBG] SQP it={it}  B0 shape = {B.shape}")
                print("[DBG] B0 column norms:")
                for j in range(m):
                    print(f"  {col_names[j]:>2s}: {np.linalg.norm(B[:, j]):.3e}")

                print("[DBG] B0 rows (scientific):")
                for i in range(n):
                    vals = " ".join([f"{B[i,j]:+10.3e}" for j in range(m)])
                    print(f"  {row_names[i]:>3s}: {vals}")
                print("       " + " ".join([f"{c:>10s}" for c in col_names]))

                Bt = B[:, 0:3]      # vx vy vz
                Bw = B[:, 3:6]      # wx wy wz
                BdL = B[:, 6:7]     # dL

                print("[DBG] authority norms:")
                print(f"  ||B_trans||_F = {np.linalg.norm(Bt, 'fro'):.3e}")
                print(f"  ||B_omega||_F = {np.linalg.norm(Bw, 'fro'):.3e}")
                print(f"  ||B_dL||_F    = {np.linalg.norm(BdL, 'fro'):.3e}")

                # optional: ratios
                eps = 1e-12
                print("[DBG] ratios:")
                print(f"  trans/omega = {np.linalg.norm(Bt,'fro')/(np.linalg.norm(Bw,'fro')+eps):.3f}")
                print(f"  dL/omega    = {np.linalg.norm(BdL,'fro')/(np.linalg.norm(Bw,'fro')+eps):.3f}")
                # ---- DEBUG: SVD / conditioning ----
                U, S, Vt = np.linalg.svd(B, full_matrices=False)
                cond = (S[0] / max(S[-1], 1e-16)) if S.size else np.inf
                rank = int(np.sum(S > 1e-10))
                # ---- NEW: cache Jacobian SVD stats for this MPC step ----
                self._jac_svd_last = dict(
                    S=S.copy(),
                    cond=float(cond),
                    rank=int(rank),
                    # optional: store norms you already compute
                    trans_norm=float(np.linalg.norm(B[:, 0:3], 'fro')),
                    omega_norm=float(np.linalg.norm(B[:, 3:6], 'fro')),
                    dL_norm=float(np.linalg.norm(B[:, 6:7], 'fro')),
                )
                print("[DBG] svd singular values:", np.array2string(S, precision=3, suppress_small=False))
                print(f"[DBG] rank≈{rank}/{min(B.shape)}   cond≈{cond:.3e}")

                # Right-singular vectors show combinations of inputs that matter
                # (rows of Vt correspond to singular directions in input space)
                print("[DBG] top input directions (Vt rows, mapped to [vx vy vz wx wy wz dL]):")
                for k_show in range(min(3, Vt.shape[0])):
                    v = Vt[k_show, :]
                    print(f"  mode{k_show}: " + " ".join([f"{col_names[j]}:{v[j]:+0.3f}" for j in range(m)]))
                print("[DBG] top output directions (U cols -> [x y z tx ty tz]):")
                names = ["x","y","z","tx","ty","tz"]
                for k_show in range(3):
                    uk = U[:, k_show]
                    print("  out_mode%d:"%k_show, " ".join([f"{names[i]}:{uk[i]:+0.3f}" for i in range(6)]))

            # ---- nominal nonlinear rollout at this iterate (for affine matching + risk) ----
            p_lin = p_prev.copy()
            p_first = p_seq[0].copy()

            # IMPORTANT: for deterministic wrapper, these are "pure" evals (commit=False)
            Y_nom = np.vstack([self.forward_tip_fn(p_seq[i], commit=False) for i in range(Np)]).reshape(Np, n)
            X_nom = Y_nom.reshape(Np*n, 1)
            
            risk = predictive_risk_along_horizon(
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
            M  = Cc.shape[0]

            idx_k_risk = np.asarray(risk.get("idx_k", np.zeros(Np, dtype=int)), dtype=int).reshape(-1)
            if idx_k_risk.size != Np:
                i_ref0 = int(getattr(self, "i_ref_last", 0))
                idx_k_risk = i_ref0 + np.arange(Np)
            idx_k_risk = np.clip(idx_k_risk, 0, M-1)

            idx_shift = int(getattr(self, "idx_ahead", 3))
            idx_k_epm = np.clip(idx_k_risk + idx_shift, 0, M-1)

            theta_seq = np.asarray(risk["theta_deg_k"], float)     # (Np,)
            info_theta0 = float(theta_seq[0])
            info_thetamax = float(np.max(theta_seq))
            g_seq = np.asarray(risk["g_k"], float)
            t_vessel = np.asarray(risk["t_vessel_k"], float)  # (Np,3)

            g_wall = np.asarray(g_seq, float).reshape(Np,)
            theta  = np.asarray(theta_seq, float).reshape(Np,)

            theta_crit = float(self.theta_crit_deg)
            theta_band = float(getattr(self, "theta_gate_band_deg", 5.0))
            g_theta = 1.0 / (1.0 + np.exp(-(theta - theta_crit)/max(theta_band, 1e-6)))
            g_unsafe0 = float(g_wall[0] * g_theta[0])

            recent_aligned = bool(self._epm_aligned_hist) and any(self._epm_aligned_hist)
            recent_bad     = bool(self._epm_bad_hist) and any(self._epm_bad_hist)
            allow_relax    = recent_aligned and recent_bad

            if allow_relax:
                alpha_min = float(getattr(self, "epm_alpha_min", 0.05))
                alpha = alpha_min + (1.0 - alpha_min) * (1.0 - g_unsafe0)
            else:
                alpha = 1.0

            w_standoff_eff = float(self.w_mag_center_standoff) * alpha
            w_inline_eff   = float(self.w_mag_inline_centerline) * alpha
            w_dipole_eff   = float(self.w_dipole_align) * alpha


            Rtil = np.kron(np.eye(Np), self.R)

            if Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np - 1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            # ---- base objective: effort only ----
            H_effort = 2.0 * Rtil
            H_smooth = 2.0 * H_du if not np.isscalar(H_du) else None
            H = H_effort.copy()
            if H_smooth is not None:
                H = H + H_smooth
            f = np.zeros((Np*m, 1), float)
            if dbg_terms is not None:
                zero_f = np.zeros((Np*m, 1), float)
                dbg_terms["effort"] = (H_effort.copy(), zero_f.copy())
                if H_smooth is not None:
                    dbg_terms["smooth"] = (H_smooth.copy(), zero_f.copy())

            # disturbance consistent injection
            if self.use_offset_free:
                X_nom = X_nom + self._disturbance_stack(self.d)
            # affine so linear model matches nominal at U_guess
            U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)
            X_aff = X_nom - Mc @ U_guess_vec
            # ---- Centerline tracking reference (stacked) ----
            # Choose which centerline index sequence you want to track:
            # - idx_k_risk: “closest / risk-mapped” centerline station per stage
            # - or monotone: i_ref_last + k
            Cc = np.asarray(self.lumen_C, float)  # (M,3) assumed
            M  = Cc.shape[0]
            if centerline_only:
                i0 = int(getattr(self, "i_ref_last", 0))
                look = int(getattr(self, "ref_lookahead_pts", 3))
                idx_ref = np.clip(i0 + look + np.arange(Np), 0, M-1)
            else:
                idx_ref = idx_k_risk
            # # Use risk mapping if available, else monotone
            # if "idx_k_risk" in locals():
            #     idx_ref = np.asarray(idx_k_risk, dtype=int)
            # else:
            #     i0 = int(getattr(self, "i_ref_last", 0))
            #     idx_ref = i0 + np.arange(Np)

            # idx_ref = np.clip(idx_ref, 0, M-1)

            # Build stacked X_ref (Np*n,1). Track only position; leave other outputs (e.g. tangent) unchanged.
            X_ref = np.zeros((Np*n, 1), float)
            for k in range(Np):
                X_ref[k*n + 0, 0] = Cc[idx_ref[k], 0]
                X_ref[k*n + 1, 0] = Cc[idx_ref[k], 1]
                X_ref[k*n + 2, 0] = Cc[idx_ref[k], 2]

            # ---- Tracking cost: sum_k ||x_k - xref_k||_Q ----
            # Use your existing Q (n x n). If you only want position tracking, ensure Q only weights xyz.
            Qtil = np.kron(np.eye(Np), self.Q)  # (Np*n, Np*n)

            # keep for outputs/debug
            Mc_last = Mc
            X_aff_last = X_aff
            X_nom_last = X_nom
            p_seq_last = p_seq

            H_track = 2.0 * (Mc.T @ Qtil @ Mc)
            f_track = 2.0 * (Mc.T @ Qtil @ (X_aff - X_ref))
            # H += H_track
            # f += f_track

            if dbg_terms is not None:
                dbg_terms["track"] = (H_track.copy(), f_track.copy())
                dbg_terms["X_ref"] = X_ref.copy()

            # ---- base stacked open-loop (no control) for band constraints ----
            X0_stack = (Mx @ xk).reshape(Np*n, 1)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)

            # ---- advancement reward (linear term) ----
            if t_vessel is not None and enable_adv_eff:
                t_v = np.asarray(t_vessel, float).copy()
                for k in range(1, Np):
                    if np.dot(t_v[k], t_v[k-1]) < 0.0:
                        t_v[k] *= -1.0
                # optionally also anchor t_v[0] to centerline forward direction:
                t0 = forward_tangent(Cc, int(idx_k_epm[0]), None)
                if np.dot(t_v[0], t0) < 0.0:
                    t_v[0] *= -1.0

                # build t_delta once
                t_delta = np.zeros((3*Np, 1))
                for k in range(Np):
                    t_delta[3*k:3*k+3, 0] = t_v[k]

                # build Dpos once
                D1_inc = np.zeros((Np, Np))
                D1_inc[0, 0] = +1.0
                for k in range(1, Np):
                    D1_inc[k, k-1] = -1.0
                    D1_inc[k, k]   = +1.0
                Dpos = np.kron(D1_inc, np.eye(3))

                idx_pos = _pos_row_idx(n, Np)
                Mc_pos = Mc[idx_pos, :]
                g_adv = Mc_pos.T @ (Dpos.T @ t_delta)

                # unsafe gate
                t0 = unit(t_v[0])
                tt = unit(xk[3:6,0])
                theta0 = np.degrees(np.arccos(np.clip(float(np.dot(t0,tt)), -1.0, 1.0)))
                g_theta0 = 1.0/(1.0 + np.exp(-(theta0-40.0)/5.0))
                g_wall0 = float(g_seq[0])

                beta = 0.9
                w_adv_eff = float(getattr(self, "w_adv_eff", self.w_adv)) * (1.0 - beta*g_unsafe0)
                s_des_eff = max(self.s_des * (1.0 - beta*g_unsafe0), 0.1*self.s_des)

                # store for later use in constraints
                s_des_eff_local = s_des_eff

                # apply progress reward
                if t_vessel is not None and enable_adv_eff:
                    f_adv = -w_adv_eff * g_adv
                    f = f + f_adv
                    if dbg_terms is not None:
                        dbg_terms["advance"] = (np.zeros((Np*m, Np*m)), f_adv.copy())
            else:
                s_des_eff_local = self.s_des

            # ---- penalty: magnet standoff to centerline ----
            if enable_standoff_eff and (self.w_mag_center_standoff > 0.0):
                w = w_standoff_eff
                d0 = float(self.mag_center_standoff_m)

                Nu = Np*m
                Pm = build_Pm_world(self.dt, Np, m=m)          # (3Np, Nu)
                U_guess_vec = U_guess.reshape(-1, 1)           # (Nu,1)

                r0 = p0[:3].copy().reshape(3,1)
                r0_stack = np.tile(r0, (Np, 1))                # (3Np,1)
                r_nom = r0_stack + Pm @ U_guess_vec            # (3Np,1)

                A_s = np.zeros((Np, Nu), float)
                b_s = np.zeros((Np, 1), float)

                eps = 1e-9
                for k in range(Np):
                    ck = Cc[idx_k_epm[k], :3].reshape(3,)
                    rk = r_nom[3*k:3*k+3, 0]
                    vk = rk - ck
                    dk = float(np.linalg.norm(vk))

                    uk = np.array([1.0, 0.0, 0.0], float) if dk < eps else (vk / dk)
                    dk = max(dk, eps)

                    Pm_k = Pm[3*k:3*k+3, :]                    # (3,Nu)
                    a_k = (uk.reshape(1,3) @ Pm_k).reshape(Nu,)
                    b_k = (dk - d0) - float(a_k @ U_guess_vec[:,0])

                    A_s[k, :] = a_k
                    b_s[k, 0] = b_k
                if self.debug:
                    Ck = np.zeros((Np, 3), float)
                    d_nom = np.zeros(Np, float)
                    for kk in range(Np):
                        Ck[kk] = Cc[idx_k_epm[kk], :3]
                        d_nom[kk] = np.linalg.norm(r_nom[3*kk:3*kk+3, 0] - Ck[kk])

                    self._standoff_dbg = dict(
                        d0=float(d0),
                        idx_k=np.asarray(idx_k_epm, int).copy(),
                        Ck=Ck.copy(),
                        d_nom=d_nom.copy(),
                        Pm=Pm.copy(),
                        r0_stack=r0_stack.copy(),
                        A=A_s.copy(),
                        b=b_s.copy(),
                    )
                if enable_standoff_eff and (self.w_mag_center_standoff > 0.0):
                    H_standoff = 2.0*w*(A_s.T @ A_s)
                    f_standoff = 2.0*w*(A_s.T @ b_s)
                    H += H_standoff
                    f += f_standoff
                    if dbg_terms is not None:
                        dbg_terms["standoff"] = (H_standoff.copy(), f_standoff.copy())
            # ---- penalty: position inline with forward tangent + forward offset ----
            if enable_inline_eff:
                w_lat   = float(getattr(self, "w_mag_lat_inline", 10.0))  # lateral (n,b) penalty
                w_fwd   = float(getattr(self, "w_mag_fwd", 1.0))          # forward penalty
                s_ahead = float(getattr(self, "s_inline_ahead_m", .12))    # meters

                if (w_lat > 0.0) or (w_fwd > 0.0):
                    # A_lat U + b_lat stacks [e_n0,e_b0,e_n1,e_b1,...]
                    A_lat = np.zeros((2*Np, Nu), float)
                    b_lat = np.zeros((2*Np, 1), float)

                    # A_fwd U + b_fwd stacks [e_t0,e_t1,...]
                    A_fwd = np.zeros((Np, Nu), float)
                    b_fwd = np.zeros((Np, 1), float)

                    t_prev = None
                    n_prev = None
                    for k in range(Np):


                        # stage nominal position at linearization point
                        r_nom_k = r_nom[3*k:3*k+3, :]     # (3,1)
                        Pm_k    = Pm[3*k:3*k+3, :]        # (3,Nu)

                        # nominal TIP position at stage k (used ONLY to anchor vessel frame)
                        x_tip_k = X_nom[k*n:k*n+3, 0].reshape(3,)

                        # closest point on centerline polyline to tip
                        i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x_tip_k)
                        i_seg = int(np.clip(i_seg, 0, Cc.shape[0]-2))
                        c = c_closest.reshape(3, 1)

                        # tangent from closest segment
                        t_k = Cc[i_seg+1, :3] - Cc[i_seg, :3]
                        t_k = t_k / (np.linalg.norm(t_k) + 1e-12)
                        if t_prev is not None and np.dot(t_k, t_prev) < 0.0:
                            t_k = -t_k

                        # nearest vertex only for frame helper
                        idx = int(np.clip(i_seg + (u_seg >= 0.5), 0, Cc.shape[0]-1))

                        # IMPORTANT: use a basis consistent with t_k (preferred: build from t_k)
                        # If using forward_tnb, at least take its tangent consistently:
                        # keep t_k from segment, then:
                        ref = np.array([1.0,0.0,0.0])
                        if abs(np.dot(ref, t_k)) > 0.9:
                            ref = np.array([0.0,1.0,0.0])
                        n_k = ref - np.dot(ref, t_k)*t_k
                        n_k /= (np.linalg.norm(n_k) + 1e-12)
                        b_k = np.cross(t_k, n_k)
                        b_k /= (np.linalg.norm(b_k) + 1e-12)
                        t_prev = t_k.copy()
                        n_prev = n_k.copy()

                        # --- lateral residuals: nᵀ(r-c)=0, bᵀ(r-c)=0
                        a_n = (n_k.reshape(1,3) @ Pm_k).reshape(-1)
                        a_b = (b_k.reshape(1,3) @ Pm_k).reshape(-1)

                        b_n = float(n_k @ (r_nom_k[:,0] - c[:,0])) - float(a_n @ U_guess_vec[:,0])
                        b_b = float(b_k @ (r_nom_k[:,0] - c[:,0])) - float(a_b @ U_guess_vec[:,0])

                        A_lat[2*k+0, :] = a_n
                        b_lat[2*k+0, 0] = b_n
                        A_lat[2*k+1, :] = a_b
                        b_lat[2*k+1, 0] = b_b

                        # --- forward residual: tᵀ(r-c) - s_ahead = 0
                        a_t = (t_k.reshape(1,3) @ Pm_k).reshape(-1)
                        b_t = float(t_k @ (r_nom_k[:,0] - c[:,0])) - s_ahead - float(a_t @ U_guess_vec[:,0])

                        A_fwd[k, :] = a_t
                        b_fwd[k, 0] = b_t

                    if enable_inline_eff:
                        if w_lat > 0.0:
                            H_inline_lat = 2.0*w_lat*(A_lat.T @ A_lat)
                            f_inline_lat = 2.0*w_lat*(A_lat.T @ b_lat)
                            H += H_inline_lat
                            f += f_inline_lat
                            if dbg_terms is not None:
                                dbg_terms["inline_lat"] = (H_inline_lat.copy(), f_inline_lat.copy())

                        if w_fwd > 0.0:
                            H_inline_fwd = 2.0*w_fwd*(A_fwd.T @ A_fwd)
                            f_inline_fwd = 2.0*w_fwd*(A_fwd.T @ b_fwd)
                            H += H_inline_fwd
                            f += f_inline_fwd
                            if dbg_terms is not None:
                                dbg_terms["inline_fwd"] = (H_inline_fwd.copy(), f_inline_fwd.copy())
                    if self.debug:
                        # store raw alignment residual models for post-solve evaluation
                        self._inline_dbg = dict(
                            idx_k=np.asarray(idx_k_epm, int).copy(),
                            A_lat=A_lat.copy(),
                            b_lat=b_lat.copy(),
                            A_fwd=A_fwd.copy(),
                            b_fwd=b_fwd.copy(),
                            w_lat=float(w_lat),
                            w_fwd=float(w_fwd),
                            s_ahead=float(s_ahead),
                        )

                        # nominal residuals at current SQP linearization point (U = U_guess)
                        e_lat_nom = (A_lat @ U_guess_vec + b_lat).reshape(-1)   # length 2*Np
                        e_fwd_nom = (A_fwd @ U_guess_vec + b_fwd).reshape(-1)   # length Np

                        e_n_nom = e_lat_nom[0::2]
                        e_b_nom = e_lat_nom[1::2]
                        rho_lat_nom = np.sqrt(e_n_nom**2 + e_b_nom**2)

                        self._inline_dbg["e_n_nom"] = e_n_nom.copy()
                        self._inline_dbg["e_b_nom"] = e_b_nom.copy()
                        self._inline_dbg["rho_lat_nom"] = rho_lat_nom.copy()
                        self._inline_dbg["e_fwd_nom"] = e_fwd_nom.copy()
            def skew3(v):
                v = np.asarray(v, float).reshape(3,)
                x, y, z = v
                return np.array([[0.0, -z,  y],
                                [z,  0.0, -x],
                                [-y, x,  0.0]], float)

            # ---- penalty: align dipole with vessel centerline tangent ----
            # ---- penalty: align dipole with vessel centerline tangent ----
            if enable_dipole_eff and (self.w_dipole_align > 0.0) and (t_vessel is not None):

                w = w_dipole_eff

                # lock dipole body axis sign ONCE (per controller instance)
                if not hasattr(self, "_d_body_locked"):
                    d_body0 = np.asarray(self.dipole_body_axis, float).reshape(3,)
                    t0 = forward_tangent_smooth(Cc, int(idx_k_epm[0]), None, look=3)
                    q0 = np.asarray(p_seq[0][3:7], float).reshape(4,)
                    self._d_body_locked = pick_dipole_axis_forward(d_body0, q0, t0)

                d_body = self._d_body_locked  # fixed sign from here on

                d_nom = np.zeros((3*Np, 1), float)
                t_tar = np.zeros((3*Np, 1), float)

                t_prev = None
                for k in range(Np):
                    qk = np.asarray(p_seq[k][3:7], float).reshape(4,)
                    Rk = quat_wxyz_to_R(qk)

                    dk = Rk @ d_body
                    dk /= (np.linalg.norm(dk) + 1e-12)

                    tk = forward_tangent_smooth(Cc, int(idx_k_epm[k]), t_prev, look=3)
                    t_prev = tk.copy()
                    tk /= (np.linalg.norm(tk) + 1e-12)

                    d_nom[3*k:3*k+3, 0] = dk
                    t_tar[3*k:3*k+3, 0] = tk
                if self.debug:
                    # angle(dipole, tangent) in degrees at the NOMINAL trajectory (pre-QP)
                    ang_nom_deg = np.zeros(Np, float)
                    for k in range(Np):
                        dk = d_nom[3*k:3*k+3, 0]
                        tk = t_tar[3*k:3*k+3, 0]
                        c = float(np.clip(np.dot(dk, tk), -1.0, 1.0))
                        ang_nom_deg[k] = float(np.degrees(np.arccos(c)))
                    # store for printing after solve
                    self._dipole_dbg = dict(
                        w_dipole=float(w),
                        ang_nom_deg=ang_nom_deg,
                        # store these so we can compute an "opt" estimate if desired
                        d_body=d_body.copy(),
                        idx_k=idx_k_epm.copy(),
                        t_tar=t_tar.copy(),         # (3Np,1)
                        p_seq=p_seq.copy(),         # pose seq (nominal)
                    )
                Pomega_w = build_Pomega_world(p_seq, self.dt, Np, m=m)  # (3Np, Nu)
                Pomega_w = np.asarray(Pomega_w, float)
                Nu = Np*m

                assert Pomega_w.shape == (3*Np, Nu), f"Pomega_w shape {Pomega_w.shape} expected {(3*Np, Nu)}"

                A_align = np.zeros((3*Np, Nu), float)

                for k in range(Np):
                    dk = d_nom[3*k:3*k+3, 0].reshape(3,)
                    Sk = skew3(dk)
                    Pk = Pomega_w[3*k:3*k+3, :]   # (3,Nu)

                    # sanity checks
                    assert Sk.shape == (3,3), f"Sk shape {Sk.shape}, dk={dk}"
                    assert Pk.shape == (3,Nu), f"Pk shape {Pk.shape}"

                    A_align[3*k:3*k+3, :] = -Sk @ Pk

                b_align = (d_nom - t_tar)  # (3Np,1)
                if enable_dipole_eff and (self.w_dipole_align > 0.0) and (t_vessel is not None):
                    H_dip = 2.0*w*(A_align.T @ A_align)
                    f_dip = 2.0*w*(A_align.T @ b_align)
                    H += H_dip
                    f += f_dip
                    if dbg_terms is not None:
                        dbg_terms["dipole"] = (H_dip.copy(), f_dip.copy())

            # -----------------------
            # Build all U-only constraints FIRST
            # -----------------------
            A_list, l_list, u_list = [], [], []

            # (1) symmetric input bounds (only if finite)
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np*m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            # (2) dL lower bound: dL_k >= -dL_back_max
            dL_back_max = float(getattr(self, "dL_back_max", 0.0))
            A_dL = np.zeros((Np, Np*m), float)
            for k in range(Np):
                A_dL[k, k*m + 6] = 1.0
            A_list.append(A_dL)
            l_list.append(-dL_back_max * np.ones(Np))
            u_list.append(np.full(Np, np.inf))
            # (3) optional minimum progress constraint: t0^T(x1-x0) >= s_min
            s_min = float(getattr(self, "s_min_progress", 0.0))
            if s_min > 0.0 and (t_vessel is not None):
                x0 = xk[0:3, :]  # (3,1)
                idx_x1 = np.array([0*n + 0, 0*n + 1, 0*n + 2], dtype=int)
                Mc_x1 = Mc[idx_x1, :]                 # (3, Np*m)
                x1_aff = X_aff[idx_x1, :].reshape(3, 1)

                t0 = np.asarray(t_vessel[0], float).reshape(1, 3)
                a = (t0 @ Mc_x1)                      # (1, Np*m)
                b = float(s_min - (t0 @ (x1_aff - x0))[0, 0])

                A_list.append(a)
                l_list.append(np.array([b], float))
                u_list.append(np.array([+np.inf], float))

            # -----------------------
            # OPTION A: Lift to z=[U;s] and add soft lumen constraints
            # -----------------------
            ns_wall = Np if enable_soft_wall_eff else 0
            ns_prog = Np if enable_soft_progress_eff else 0
            ns = ns_wall + ns_prog
            Nu = Np*m
            nz = Nu + ns

            H_z = np.zeros((nz, nz), float)
            H_z[:Nu,:Nu] = H

            f_z = np.zeros((nz, 1), float)
            f_z[:Nu,:] = f

            # ---- wall slack penalty (adaptive per stage) ----
            if ns_wall > 0:
                g_wall = np.asarray(g_seq, float).reshape(Np,)
                theta  = np.asarray(theta_seq, float).reshape(Np,)

                theta_crit = float(self.theta_crit_deg)
                theta_band = float(getattr(self, "theta_gate_band_deg", 5.0))
                g_theta = 1.0 / (1.0 + np.exp(-(theta - theta_crit) / max(theta_band, 1e-6)))

                p = float(getattr(self, "wall_gate_pow", 2.0))
                q = float(getattr(self, "theta_gate_pow", 2.0))
                gate = (g_wall**p) * (g_theta**q)

                w_min = float(getattr(self, "w_slack_wall_min", 1e2))
                w_max = float(getattr(self, "w_slack_wall_max", self.w_slack_wall))
                w_stage = w_min + (w_max - w_min) * gate

                H_z[Nu:Nu+ns_wall, Nu:Nu+ns_wall] = 2.0 * np.diag(w_stage)

            # ---- progress slack penalty (unchanged, independent) ----
            if ns_prog > 0:
                i0 = Nu + ns_wall
                H_z[i0:i0+ns_prog, i0:i0+ns_prog] = 2.0 * self.w_slack_prog * np.eye(ns_prog)

            # nonnegativity of slacks
            # s_wall >= 0
            # s_prog >= 0
            A_s = np.zeros((ns, nz), float)
            A_s[:, Nu:] = np.eye(ns)
            l_s = np.zeros(ns)
            u_s = np.full(ns, np.inf)
            # stack existing constraints (pad to nz if needed)
            if A_list:
                A_osqp = np.vstack([_pad_A(Ai, nz, Nu) for Ai in A_list])
                l_osqp = np.concatenate(l_list).astype(float)
                u_osqp = np.concatenate(u_list).astype(float)
            else:
                A_osqp = np.zeros((0, nz), float)
                l_osqp = np.zeros(0, float)
                u_osqp = np.zeros(0, float)

            # slack nonnegativity: s >= 0
            if ns > 0:
                A_s = np.zeros((ns, nz), float)
                A_s[:, Nu:] = np.eye(ns)
                A_osqp = np.vstack([A_osqp, A_s])
                l_osqp = np.concatenate([l_osqp, np.zeros(ns)])
                u_osqp = np.concatenate([u_osqp, np.full(ns, np.inf)])

            # lumen soft margin constraints: aU*U + s_k >= -mbar_k
            if ns_wall > 0:
                risk = getattr(self, "_risk_last", None)
                if risk is None:
                    risk = predictive_risk_along_horizon(
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

                idx_k = np.asarray(risk["idx_k"], int)
                Cc = np.asarray(self.lumen_C, float)
                Rr = np.asarray(self.lumen_R, float)

                soft_wall_delta = float(getattr(self, "soft_wall_delta", getattr(self, "delta_wall", 5e-4)))
                soft_eps = float(getattr(self, "soft_margin_eps", 1e-9))

                A_marg = np.zeros((Np, nz), float)
                l_marg = np.full(Np, -np.inf, float)
                u_marg = np.full(Np, +np.inf, float)

                for kcon in range(Np):
                    # affine predicted position at stage k
                    xbar = np.asarray(X_nom[kcon*n:kcon*n+3, 0], float)

                    ik = int(idx_k[kcon])
                    tv = centerline_tangent(Cc, ik)
                    c = Cc[ik]
                    r = xbar - c
                    r_perp = r - (r @ tv) * tv
                    d = float(np.linalg.norm(r_perp))

                    # margin at linearization point
                    mbar = float((Rr[ik] - soft_wall_delta) - d)

                    # dm/dx = -u_perp where u_perp is outward unit direction in cross-section
                    if d < soft_eps:
                        u_perp = np.zeros(3)
                    else:
                        u_perp = r_perp / d
                    dm_dx = (-u_perp).reshape(1, 3)  # (1,3)

                    Mc_k = _Mc_pos_stage(Mc, n, m, Np, kcon)  # (3, Nu)
                    aU = (dm_dx @ Mc_k).reshape(-1)          # (Nu,)

                    A_marg[kcon, :Nu] = aU
                    A_marg[kcon, Nu + kcon] = 1.0    # wall slack
                    l_marg[kcon] = -mbar

                A_osqp = np.vstack([A_osqp, A_marg])
                l_osqp = np.concatenate([l_osqp, l_marg])
                u_osqp = np.concatenate([u_osqp, u_marg])

            # final dimension sanity
            assert A_osqp.shape[1] == nz
            assert H_z.shape[0] == H_z.shape[1] == nz
            if ns_prog > 0 and (t_vessel is not None):

                if s_des_eff_local > 0.0:
                    # stacked affine positions
                    idx_pos = _pos_row_idx(n, Np)
                    Mc_pos = Mc[idx_pos, :]                 # (3Np, Nu)
                    Xaff_pos = X_aff[idx_pos, :].reshape(3*Np, 1)

                    # increments operator (same as you use for reward)
                    D1_inc = np.zeros((Np, Np))
                    D1_inc[0, 0] = +1.0
                    for k in range(1, Np):
                        D1_inc[k, k-1] = -1.0
                        D1_inc[k, k]   = +1.0
                    Dpos = np.kron(D1_inc, np.eye(3))      # (3Np,3Np)

                    # BUT stage 0 should be x1 - x_now, not x1 - 0
                    x_now = xk[0:3, :].reshape(3,1)
                    b0 = np.zeros((3*Np,1))
                    b0[0:3,:] = -x_now

                    # Δx(U) = Dpos*(Xaff_pos + Mc_pos U) + b0
                    A_dX = Dpos @ Mc_pos                   # (3Np, Nu)
                    dX_aff = Dpos @ Xaff_pos + b0          # (3Np,1)

                    # build constraints: t_k^T Δx_k + p_k >= s_des
                    A_prog = np.zeros((Np, nz), float)
                    l_prog = np.full(Np, -np.inf, float)
                    u_prog = np.full(Np, +np.inf, float)

                    for k in range(Np):
                        tk = np.asarray(t_v[k], float).reshape(1,3)

                        rows = slice(3*k, 3*k+3)
                        aU = (tk @ A_dX[rows, :]).reshape(-1)         # (Nu,)
                        b  = float(s_des_eff_local - (tk @ dX_aff[rows,:])[0,0])

                        A_prog[k, :Nu] = aU
                        # progress slack index: Nu + ns_wall + k
                        A_prog[k, Nu + ns_wall + k] = 1.0
                        l_prog[k] = b

                    A_osqp = np.vstack([A_osqp, A_prog])
                    l_osqp = np.concatenate([l_osqp, l_prog])
                    u_osqp = np.concatenate([u_osqp, u_prog])

            # -----------------------
            # Solve QP in z-space (or U-space if ns=0)
            # -----------------------
            Z_warm = None
            if (U_opt_vec is not None) and (ns > 0):
                Z_warm = np.zeros(nz, float)
                Z_warm[:Nu] = U_opt_vec.copy()

            Z_opt, _, status = solve_qp_osqp(H_z, f_z.ravel(), A_osqp, l_osqp, u_osqp, U_warm=Z_warm)
            status_last = status
            # ---- DEBUG print standoff distances at the QP solution ----
            if getattr(self, "_standoff_dbg", None) is not None and self.debug:
                dbg = self._standoff_dbg
                d0_dbg = dbg["d0"]
                Ck = dbg["Ck"]              # (Np,3)
                idx_k_dbg = dbg["idx_k"]
                Pm_dbg = dbg["Pm"]
                r0_stack_dbg = dbg["r0_stack"]
                d_nom = dbg["d_nom"]

            # Only update history/debug if we actually got a solution vector
            if status in ("solved", "solved inaccurate") and (Z_opt is not None):

                # extract U (needed for any magnet trajectory debug)
                if ns > 0:
                    U_vec = np.asarray(Z_opt[:Nu], float).reshape(Nu, 1)
                else:
                    U_vec = np.asarray(Z_opt, float).reshape(Nu, 1)

                dbg = getattr(self, "_standoff_dbg", None)
                if dbg is not None:
                    idx0 = int(dbg["idx_k"][0])  # shifted index used by EPM penalties
                    Cc_local = np.asarray(self.lumen_C, float)

                    # magnet trajectory at the QP solution
                    r_opt = dbg["r0_stack"] + dbg["Pm"] @ U_vec   # (3Np,1)

                    # stage-0 alignment checks (relative to shifted station)
                    rk0 = r_opt[0:3, 0]
                    ck0 = Cc_local[idx0, :3].reshape(3,)

                    tk0 = centerline_tangent(Cc_local, idx0).reshape(3,)
                    tk0 /= (np.linalg.norm(tk0) + 1e-12)

                    d0 = float(self.mag_center_standoff_m)
                    d  = float(np.linalg.norm(rk0 - ck0))
                    e_inline = float(np.dot(tk0, (rk0 - ck0)))

                    # dipole direction (nominal from p_seq[0])
                    d_body = np.asarray(self.dipole_body_axis, float).reshape(3,)
                    d_body /= (np.linalg.norm(d_body) + 1e-12)

                    q0 = np.asarray(p_seq[0][3:7], float).reshape(4,)
                    R0 = quat_wxyz_to_R(q0)
                    d_world = R0 @ d_body
                    d_world /= (np.linalg.norm(d_world) + 1e-12)

                    cang = float(np.clip(np.dot(d_world, tk0), -1.0, 1.0))
                    ang  = float(np.degrees(np.arccos(cang)))
                    ang  = min(ang, 180.0 - ang)

                    dipole_tol   = float(getattr(self, "dipole_tol_deg", 10.0))
                    standoff_tol = float(getattr(self, "standoff_tol_m", 0.02))
                    inline_tol   = float(getattr(self, "inline_tol_m", 0.02))

                    aligned_ok = (ang <= dipole_tol) and (abs(d - d0) <= standoff_tol) and (abs(e_inline) <= inline_tol)

                    bad_wall = (float(g_seq[0]) > float(getattr(self, "g_wall_bad", 0.5))) or (float(theta_seq[0]) > float(self.theta_crit_deg))
                    still_bad = bool(bad_wall)

                    self._epm_aligned_hist.append(aligned_ok)
                    self._epm_bad_hist.append(still_bad)

                    # standoff debug print (consistent with the penalty)
                    Ck = dbg["Ck"]
                    d_nom = dbg["d_nom"]
                    idx_k_dbg = dbg["idx_k"]
                    d0_dbg = float(dbg["d0"])

                    d_opt = np.zeros(Np, float)
                    for k in range(Np):
                        rk = r_opt[3*k:3*k+3, 0]
                        d_opt[k] = float(np.linalg.norm(rk - Ck[k]))

                    e_lin = (dbg["A"] @ U_vec + dbg["b"]).reshape(Np,)

                    print(f"[STANDOFF] target d0 = {d0_dbg:.4f} m")
                    for k in range(Np):
                        print(
                            f"  k={k:02d} idx={int(idx_k_dbg[k]):4d} "
                            f"d_nom={d_nom[k]:.4f} d_opt={d_opt[k]:.4f} "
                            f"(d_opt-d0)={(d_opt[k]-d0_dbg):+.4f} e_lin={e_lin[k]:+.4f}"
                        )
                    print("[STANDOFF] idx_k:", idx_k_dbg.tolist())
                    print("[STANDOFF] Ck[0]:", Ck[0], "Ck[-1]:", Ck[-1])
            if status in ("solved","solved inaccurate") and Z_opt is not None and self.debug:
                self._debug_objective_breakdown(
                    Z_opt=Z_opt,
                    H_z=H_z,
                    f_z=f_z,
                    Nu=Nu,
                    ns_wall=ns_wall,
                    ns_prog=ns_prog,
                    Rtil=Rtil,
                    H_du=H_du,
                    w_slack_wall=self.w_slack_wall if ns_wall > 0 else None,
                    w_slack_prog=self.w_slack_prog if ns_prog > 0 else None,
                    w_adv_eff=w_adv_eff if "w_adv_eff" in locals() else None,
                    g_adv=g_adv if "g_adv" in locals() else None,
                    overhead_terms=None,  # optional
                )
            if status in ("solved", "solved inaccurate") and (Z_opt is not None):
                if ns > 0:
                    U_opt_vec = Z_opt[:Nu].copy()
                    self._slack_last = Z_opt[Nu:].copy()
                else:
                    U_opt_vec = np.asarray(Z_opt, float).copy()
            else:
                U_opt_vec = None

            infeas = (U_opt_vec is None)
            if infeas:
                U_guess = np.zeros((Np, m))
                break

            # update SQP iterate
            U_guess = U_opt_vec.reshape(Np, m)
        # ---- DEBUG dipole alignment + weights summary ----
        if self.debug:
            # weights summary (what is active right now)
            w_sum = dict(
                w_adv=float(getattr(self, "w_adv", 0.0)),
                w_adv_eff=float(w_adv_eff) if "w_adv_eff" in locals() else float(getattr(self, "w_adv", 0.0)),
                w_standoff=w_standoff_eff,
                d0=float(getattr(self, "mag_center_standoff_m", np.nan)),
                w_inline=w_inline_eff,
                w_dipole=w_dipole_eff,
                w_slack_wall=float(getattr(self, "w_slack_wall", 0.0)),
                w_slack_prog=float(getattr(self, "w_slack_prog", 0.0)),
            )
            print("[WEIGHTS]", w_sum)

        if getattr(self, "_dipole_dbg", None) is not None and self.debug:
            ddbg = self._dipole_dbg
            ang = np.asarray(ddbg["ang_nom_deg"], float)
            print("[DIPOLE] alignment to vessel tangent (nominal, pre-QP):")
            for k in range(min(Np, 5)):  # print first few
                print(f"  k={k:02d}  angle={ang[k]:6.2f} deg")
            print(f"  max={np.max(ang):.2f} deg  mean={np.mean(ang):.2f} deg  w_dipole={ddbg['w_dipole']:.3g}")
        # -----------------------
        # Apply first control
        # -----------------------
        infeas_final = (U_opt_vec is None)

        if infeas_final:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
        else:
            U_seq = U_opt_vec.reshape(Np, m)
            u0 = U_seq[0, :].copy()

        # predicted horizon (linear) for debug
        X_pred = np.full((Np, n), np.nan)
        if (not infeas_final) and (Mc_last is not None) and (X_aff_last is not None):
            U_vec = U_opt_vec.reshape(-1, 1)
            X_pred_stack = X_aff_last + Mc_last @ U_vec
            X_pred = X_pred_stack.reshape(Np, n)

        # plant update (one step)
        p_next_true = self._clamp_p(integrate_pose8_body(p_prev, u0, self.dt))
        try:
            x_next_true = np.asarray(self.forward_tip_fn(p_next_true, commit=False), float).reshape(n,)
        except TypeError:
            x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(n,)

        if self.use_offset_free:
            x_next_true = x_next_true + self.d

        # commit internal state
        self.p = p_next_true.copy()
        self.x = x_next_true.copy()

        # warm start
        self.U_warm = None if infeas_final else U_opt_vec.copy()

        # one-step prediction error (pos only)
        # one-step prediction error
        pred1_err_xy = np.nan
        pred1_err_xyz = np.nan

        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            e = x_next_true[:3] - X_pred[0][:3]
            pred1_err_xy  = float(np.linalg.norm(e[:2]))   # XY only
            pred1_err_xyz = float(np.linalg.norm(e))       # full XYZ (optional)
            tan1_err = float(np.linalg.norm(x_next_true[3:6] - X_pred[0][3:6])) if (n >= 6 and X_pred.shape[1] >= 6) else np.nan
        if self.debug and (Mc_last is not None):
            xbase0 = X0_stack[0:n].ravel()
            xaff0  = X_aff_last[0:n].ravel()
            print("xbase0", xbase0[:3], "xaff0", xaff0[:3])
        # ---- tracking + centreline alignment debug ----
        track_dbg = {}
        try:
            if (X_pred is not None) and (Mc_last is not None) and (X_aff_last is not None):
                track_dbg = self._compute_tracking_and_centerline_debug(
                    X_pred=X_pred,
                    X_ref=dbg_terms.get("X_ref", None) if (self.debug and dbg_terms is not None) else None,
                    X_nom=(X_nom_last.reshape(Np, n) if X_nom_last is not None else None),
                    n=n, Np=Np,
                    Cc=np.asarray(self.lumen_C, float),
                    idx_ref=np.asarray(idx_ref, int),
                    x_now=self.x
                )
        except Exception as e:
            if self.debug:
                print("[DBG] tracking/centerline debug failed:", repr(e))
        if self.debug and track_dbg:
            print(
                "[TRACK] "
                f"pred_ref_rms={1e3*track_dbg.get('pred_ref_err_rms_m', np.nan):.3f}mm "
                f"pred_ref_max={1e3*track_dbg.get('pred_ref_err_max_m', np.nan):.3f}mm "
                f"rho_rms={1e3*track_dbg.get('pred_centerline_rho_rms_m', np.nan):.3f}mm "
                f"rho_max={1e3*track_dbg.get('pred_centerline_rho_max_m', np.nan):.3f}mm "
                f"clear_min={1e3*track_dbg.get('pred_clearance_min_m', np.nan):.3f}mm"
            )
       
        info = dict(
            status=status_last,
            infeasible=int(infeas_final),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            d=self.d.copy(),
            X_pred=X_pred.copy(),
            U_seq=U_seq.copy(),
            N_sqp=int(self.N_sqp),
            pred1_err_xy=float(pred1_err_xy) if np.isfinite(pred1_err_xy) else np.nan,
            p_prev=p_prev.copy(),
            x_prev=x_prev.copy(),
            B_first=B_first.copy() if B_first is not None else None,
            p_lin=p_lin.copy() if isinstance(p_lin, np.ndarray) else p_lin,
            p_first=p_first.copy() if isinstance(p_first, np.ndarray) else p_first,
            X_aff_last=X_aff_last.copy() if X_aff_last is not None else None,
            Mc_last=Mc_last.copy() if Mc_last is not None else None,
            X_nom_last=X_nom_last.reshape(Np, n).copy() if X_nom_last is not None else None,
            p_seq_last=p_seq_last.copy() if p_seq_last is not None else None,
        )
        # --- inside step(), after you have x_now (measured) ---
        if self.debug and (self.x is not None) and (self.x.size >= 6):
            tip_pos = self.x[:3]
            tip_tan = self.x[3:6]

            i_seg, u_seg, c_cl, t_v = _closest_centerline_tangent(np.asarray(self.lumen_C, float), tip_pos)
            ang = _angle_deg(tip_tan, t_v)
            print("[DBG] tip_tan type/shape:", type(tip_tan), np.asarray(tip_tan).shape, "val:", tip_tan)
            # optional: contact-ish scalar using lumen_R if present
            rho = np.linalg.norm((tip_pos - c_cl) - np.dot(tip_pos - c_cl, t_v)*t_v)
            clearance = np.nan
            if hasattr(self, "lumen_R") and self.lumen_R is not None:
                Rr = np.asarray(self.lumen_R, float)
                # radius at nearest vertex (ok for debug)
                idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr.shape[0]-1))
                clearance = float(Rr[idx_v] - rho)

            print(
                f"[TIP×VESSEL] seg={i_seg} u={u_seg:.2f} "
                f"angle={ang:.2f}deg rho={1e3*rho:.2f}mm "
                f"clear={1e3*clearance:.2f}mm" if np.isfinite(clearance) else
                f"[TIP×VESSEL] seg={i_seg} u={u_seg:.2f} angle={ang:.2f}deg rho={1e3*rho:.2f}mm"
            )
        if status in ("solved", "solved inaccurate") and Z_opt is not None and self.debug:
            term_vals = self._eval_qp_terms(
                z=Z_opt, Nu=Nu, dbg_terms=dbg_terms,
                ns_wall=ns_wall, ns_prog=ns_prog
            )
            self._cost_terms_last = term_vals.copy()
            print("[COST_TERMS]", {k: (float(v) if np.isfinite(v) else v) for k, v in term_vals.items() if k.startswith("J_")})
        info["X_base"] = (Mx @ xk).reshape(Np, n).copy()
        if self.debug and hasattr(self, "_inline_dbg"):
            idbg = self._inline_dbg

            e_lat_opt = (idbg["A_lat"] @ U_vec + idbg["b_lat"]).reshape(-1)   # 2*Np
            e_fwd_opt = (idbg["A_fwd"] @ U_vec + idbg["b_fwd"]).reshape(-1)   # Np

            e_n_opt = e_lat_opt[0::2]
            e_b_opt = e_lat_opt[1::2]
            rho_lat_opt = np.sqrt(e_n_opt**2 + e_b_opt**2)

            print(f"[INLINE] s_ahead={1e3*idbg['s_ahead']:.2f} mm  "
                f"w_lat={idbg['w_lat']:.3g}  w_fwd={idbg['w_fwd']:.3g}")

            for k in range(Np):
                print(
                    f"  k={k:02d} idx={int(idbg['idx_k'][k]):4d} "
                    f"lat_nom=({1e3*idbg['e_n_nom'][k]:+.2f},{1e3*idbg['e_b_nom'][k]:+.2f})mm "
                    f"rho_nom={1e3*idbg['rho_lat_nom'][k]:.2f}mm "
                    f"fwd_nom={1e3*idbg['e_fwd_nom'][k]:+.2f}mm | "
                    f"lat_opt=({1e3*e_n_opt[k]:+.2f},{1e3*e_b_opt[k]:+.2f})mm "
                    f"rho_opt={1e3*rho_lat_opt[k]:.2f}mm "
                    f"fwd_opt={1e3*e_fwd_opt[k]:+.2f}mm"
                )

            # compact summary metrics (great for papers/logging)
            self._inline_metrics_last = dict(
                rho_lat_nom_rms_m=float(np.sqrt(np.mean(idbg["rho_lat_nom"]**2))),
                rho_lat_opt_rms_m=float(np.sqrt(np.mean(rho_lat_opt**2))),
                rho_lat_opt_max_m=float(np.max(rho_lat_opt)),
                e_fwd_nom_rms_m=float(np.sqrt(np.mean(idbg["e_fwd_nom"]**2))),
                e_fwd_opt_rms_m=float(np.sqrt(np.mean(e_fwd_opt**2))),
                e_fwd_opt_max_abs_m=float(np.max(np.abs(e_fwd_opt))),
            )
                
        if hasattr(self, "_cost_terms_last"):
            info["cost_terms"] = dict(self._cost_terms_last)
        if track_dbg:
            info["track_dbg"] = track_dbg
            info["pred_ref_err_rms_m"] = float(track_dbg.get("pred_ref_err_rms_m", np.nan))
            info["pred_ref_err_max_m"] = float(track_dbg.get("pred_ref_err_max_m", np.nan))
            info["pred_centerline_rho_rms_m"] = float(track_dbg.get("pred_centerline_rho_rms_m", np.nan))
            info["pred_centerline_rho_max_m"] = float(track_dbg.get("pred_centerline_rho_max_m", np.nan))
            info["pred_clearance_min_m"] = float(track_dbg.get("pred_clearance_min_m", np.nan))
        if self.debug and hasattr(self, "_magpos_dbg"):
            dbg = self._magpos_dbg
            s_nom = dbg["s_nom"]
            e_nom = dbg["e_nom"]
            print("[MAGPOS] tangential projection (nominal):")
            for k in range(min(Np, 5)):
                print(f"  k={k:02d} idx={dbg['idx_k'][k]:4d}  s_nom={1e3*s_nom[k]:+7.2f} mm  "
                    f"e_nom={1e3*e_nom[k]:+7.2f} mm  angΔt={dbg['ang_t_deg'][k]:5.1f} deg")
            print(f"  target s_ahead = {1e3*dbg['s_ahead']:.2f} mm  w_inline={dbg['w_inline']}")
        if self.debug:
            pred1 = float(pred1_err_xy) if np.isfinite(pred1_err_xy) else np.nan
            dbg = self._dbg_last if isinstance(self._dbg_last, dict) else {}
            mode = dbg.get("mpc_mode", "unknown")

            msg = (
            f"[MPC] mode={mode} status={status_last} infeas={int(infeas_final)} "
            )
            if mode == "predictive_Q":
                g0 = float(dbg["g_seq"][0])
                th0 = float(dbg["theta_seq"][0])
                m0 = float(dbg["margin_seq"][0])
                st0 = float(dbg["s_tan_seq"][0])
                msg += f" | g0={g0:.3f} m0={m0*1e3:+.2f}mm th0={th0:.1f}deg s_tan0={st0:.2f}"
            elif mode == "scalar_g0":
                msg += f" | g0={dbg.get('g0',0.0):.3f}"
            dbg_print(1, msg)
            # after you have X_pred
            dx_world = X_pred[0,0] - xk[0,0]
            print("predicted world Δx =", dx_world, "from u0 dx =", u0[0])
            if np.isfinite(pred1_err_xy) and (pred1_err_xy > DBG.tol_pred1):
                dbg_print(1, f"[WARN] pred1_err is large ({pred1_err_xy:.4e} m). Model/J or dt scaling may be off.")
            jac = getattr(self, "_jac_svd_last", None)
            if jac is not None:
                info["jac_svd_S"] = np.asarray(jac["S"], float).copy()
                info["jac_svd_cond"] = float(jac["cond"])
                info["jac_svd_rank"] = int(jac["rank"])
                info["jac_trans_norm"] = float(jac["trans_norm"])
                info["jac_omega_norm"] = float(jac["omega_norm"])
                info["jac_dL_norm"] = float(jac["dL_norm"])
            else:
                info["jac_svd_S"] = None
                info["jac_svd_cond"] = np.nan
                info["jac_svd_rank"] = -1
            info["theta_seq_deg"] = theta_seq.copy()
            info["theta0_deg"] = info_theta0
            info["theta_max_deg"] = info_thetamax
        return self.p.copy(), self.x.copy(), info


def numerical_B_y_wrt_u(p8, forward_y_fn, dt, eps_u, n_out):
    p8 = np.asarray(p8, float).ravel()
    y0 = np.asarray(forward_y_fn(p8), float).reshape(n_out,)
    B = np.zeros((n_out, 7), float)
    # regular columns 0..5
    for i in range(6):
        du = np.zeros(7); du[i] = eps_u[i]
        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)
        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
        B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

    # dL column: differentiate w.r.t L directly (small delta_L), then map to u via dt
    delta_L = 1e-3  # 0.1 mm
    p_plus = p8.copy();  p_plus[7]  += delta_L
    p_minus = p8.copy(); p_minus[7] -= delta_L
    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)
    # print("L0", p8[7], "Lplus", p_plus[7], "Lminus", p_minus[7])

    B[:, 6] = dt * dy_dL   # because u[6] is dL/dt

    return B
# def numerical_B_y_wrt_u(p8, forward_y_fn, dt, eps_u, n_out):
#     """
#     Returns B (n_out x 7) s.t. y_next ≈ y_now + B u
#     """
#     p8 = np.asarray(p8, float).ravel()
#     eps_u = np.asarray(eps_u, float).ravel()
#     assert eps_u.size == 7

#     y0 = np.asarray(forward_y_fn(p8), float).reshape(n_out,)
#     B = np.zeros((n_out, 7), float)

#     for i in range(7):
#         du = np.zeros(7)
#         du[i] = eps_u[i]

#         p_plus  = integrate_pose8_body(p8, +du, dt)
#         p_minus = integrate_pose8_body(p8, -du, dt)

#         y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
#         y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

#         B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])
    
#     return B
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
def rollout_open_loop_from_plan(mpc, p_start, U_seq):
    """
    Roll forward the nonlinear plant using the planned control sequence U_seq (Np,m).
    Returns:
      P_nl: (Np, m) parameter trajectory (p1..pNp)
      X_nl: (Np, n) tip trajectory
    """
    p = p_start.copy()
    P_nl = []
    X_nl = []
    for i in range(U_seq.shape[0]):
        p = mpc._clamp_p(integrate_pose8_body(p, U_seq[i], mpc.dt))
        x = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)
        P_nl.append(p.copy())
        X_nl.append(x.copy())
    return np.vstack(P_nl), np.vstack(X_nl)
def debug_step_pose7_no_targets(
    k,
    p_now,
    x_now,
    info,
    mpc,
    i_ref=None,
    print_horizon=4,
    do_nl_rollout=True,
    ):
    """
    Reference-free debug print for a 6D output MPC:
      y = [x,y,z, tx,ty,tz]

    Prints:
      - current measured/estimated tip position + tangent (x_now / info["x_now"])
      - current pose p_now (pose8 -> pose7 rotvec + L)
      - applied first control u0
      - one-step prediction mismatch pred1_err (if provided)
      - controller's predicted horizon X_pred (pos + tan)
      - nominal nonlinear at SQP lin point X_nom_last (if provided)
      - internal consistency check: (X_aff0 + Mc0U) vs X_pred[0]
      - optional nonlinear open-loop rollout using U_seq (commit rollout)
    """

    # -----------------------
    # Horizon sizes
    # -----------------------
    Np = int(getattr(mpc, "Np", 0))
    ph = min(int(print_horizon), Np) if Np > 0 else int(print_horizon)

    # -----------------------
    # Current output (measured/estimated)
    # -----------------------
    x_now6 = info.get("x_now", None)
    if x_now6 is None:
        x_now6 = np.asarray(x_now, float).ravel()
    else:
        x_now6 = np.asarray(x_now6, float).ravel()

    if x_now6.size >= 6:
        pos_now = x_now6[:3]
        tan_now = x_now6[3:6]
    else:
        pos_now = x_now6[:3]
        tan_now = np.full(3, np.nan)

    # -----------------------
    # Header
    # -----------------------
    status = info.get("status", "?")
    infeas = info.get("infeasible", -1)
    pred_err = float(info.get("pred1_err_xy", np.nan))

    print(
        f"k={k:04d} "
        f"tip=[{pos_now[0]:+.4f},{pos_now[1]:+.4f},{pos_now[2]:+.4f}] "
        f"tan=[{tan_now[0]:+.3f},{tan_now[1]:+.3f},{tan_now[2]:+.3f}] "
        f"status={status} infeas={infeas} pred1_err={pred_err:.4e}"
    )
    if i_ref is not None:
        print(f"   path index i_ref={int(i_ref)}")

    # -----------------------
    # Pose (magnet pose + insertion)
    # -----------------------
    p7 = pose8_quat_to_pose7_rotvec(p_now)
    rvec = p7[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))
    print(
        f"   p_now: x={p7[0]:+.3f} y={p7[1]:+.3f} z={p7[2]:+.3f}  "
        f"rotvec=[{rvec[0]:+.3f},{rvec[1]:+.3f},{rvec[2]:+.3f}] |theta|={theta_deg:.1f}deg  "
        f"L={p7[6]:.3f}"
    )

    # -----------------------
    # Control applied
    # -----------------------
    u0 = np.asarray(info.get("u0", np.zeros(getattr(mpc, "m", 7))), float).ravel()
    if u0.size >= 7:
        print(
            "   u0:",
            f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
            f"omega_body=[{u0[3]:+.4f},{u0[4]:+.4f},{u0[5]:+.4f}]  dL={u0[6]:+.5f}",
        )
    else:
        print("   u0:", u0)

    # -----------------------
    # Predicted horizon (linear MPC prediction)
    # -----------------------
    X_pred = info.get("X_pred", None)      # expected (Np,n)
    U_seq  = info.get("U_seq", None)       # expected (Np,m)

    if X_pred is None:
        print("   X_pred missing -> infeasible/failed QP or not stored.")
        print("--------------------------------------------------------------------")
        return

    X_pred = np.asarray(X_pred, float)
    if X_pred.ndim == 1:
        # try reshape if flat
        if Np > 0:
            X_pred = X_pred.reshape(Np, -1)
        else:
            X_pred = X_pred.reshape(1, -1)

    if not np.all(np.isfinite(X_pred)):
        print("   X_pred contains NaNs -> infeasible/failed QP.")
        print("--------------------------------------------------------------------")
        return

    # Ensure ph is valid
    if Np <= 0:
        Np = X_pred.shape[0]
        ph = min(ph, Np)

    print("   Horizon: predicted (linear MPC)")
    for i in range(min(ph, X_pred.shape[0])):
        p = X_pred[i, :3]
        if X_pred.shape[1] >= 6:
            t = X_pred[i, 3:6]
            print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]  "
                  f"tan=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]")
        else:
            print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]")

    # -----------------------
    # Internal consistency check: (X_aff0 + Mc0U) vs X_pred[0]
    # -----------------------
    Mc_last   = info.get("Mc_last", None)
    X_aff_last = info.get("X_aff_last", None)

    if (Mc_last is not None) and (X_aff_last is not None) and (U_seq is not None):
        Mc_last = np.asarray(Mc_last, float)
        X_aff_last = np.asarray(X_aff_last, float)
        U_seq = np.asarray(U_seq, float)
        n_out = int(getattr(mpc, "n", X_pred.shape[1]))
        m_in  = int(getattr(mpc, "m", U_seq.shape[1] if U_seq.ndim == 2 else 7))

        U_vec = U_seq.reshape(-1, 1)                  # (Np*m,1)
        Mc0 = Mc_last[0:n_out, :]                     # (n, Np*m)
        x1_from_blocks = (X_aff_last[0:n_out, :] + Mc0 @ U_vec).reshape(n_out,)

        xpred0 = X_pred[0, :n_out]
        print("   |(X_aff0 + Mc0U) - X_pred[0]| =", float(np.linalg.norm(x1_from_blocks - xpred0)))

    # -----------------------
    # Nominal nonlinear at SQP linearization point (optional)
    # -----------------------
    X_nom_last = info.get("X_nom_last", None)  # typically (Np,n)
    if X_nom_last is not None:
        X_nom_last = np.asarray(X_nom_last, float)
        if X_nom_last.ndim == 2 and X_nom_last.shape[0] >= 1:
            print("   Horizon: nominal nonlinear (SQP lin point eval)")
            for i in range(min(ph, X_nom_last.shape[0])):
                p = X_nom_last[i, :3]
                if X_nom_last.shape[1] >= 6:
                    t = X_nom_last[i, 3:6]
                    print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]  "
                          f"tan=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]")
                else:
                    print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]")

    # -----------------------
    # Nonlinear open-loop rollout following planned U_seq (optional)
    # -----------------------
    if do_nl_rollout and (U_seq is not None):
        U_seq = np.asarray(U_seq, float)
        if U_seq.ndim == 2 and np.all(np.isfinite(U_seq)):
            p_start = info.get("p_prev", None)
            if p_start is not None:
                P_nl, X_nl = rollout_open_loop_from_plan_commit(mpc, p_start, U_seq)
                print("   Horizon: NL rollout COMMIT (plant-like along planned U)")
                ph2 = min(ph, X_nl.shape[0])
                for i in range(ph2):
                    p = X_nl[i, :3]
                    if X_nl.shape[1] >= 6:
                        t = X_nl[i, 3:6]
                        print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]  "
                              f"tan=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]")
                    else:
                        print(f"   {i:02d}: pos=[{p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}]")

                # linear vs nonlinear horizon mismatch
                if X_pred is not None and X_pred.shape[0] >= 1:
                    ph3 = min(ph, X_nl.shape[0], X_pred.shape[0])
                    e_pos_m, e_tan_deg = horizon_pred_errors(X_nl, X_pred, ph=ph3)
                    print("   Horizon errors (NL commit rollout vs linear pred):")
                    for i in range(ph3):
                        print(f"   {i:02d}: pos={1e3*e_pos_m[i]:.3f}mm  tan={e_tan_deg[i]:.2f}deg")

    print("--------------------------------------------------------------------")
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
class DeterministicForward6D:
    """
    Wraps EnergyMinForwardWithLumen to guarantee:
      - within a step: all evals start from identical internal state
      - optional commit at end of step updates the baseline warm-start
    Returns y = [tip_xyz(3), tip_tangent(3)].
    """

    def __init__(self, forward_model):
        self.fm = forward_model
        self._base = None  # snapshot used within current step
        self.last_p_centerline = None
        self.last_tip = None

    def _snapshot(self):
        fm = self.fm
        return dict(
            _last=copy.deepcopy(getattr(fm, "_last", None)),
            last_p_centerline=None if fm.last_p_centerline is None else fm.last_p_centerline.copy(),
            last_tip=None if fm.last_tip is None else fm.last_tip.copy(),
            last_info=copy.deepcopy(getattr(fm, "last_info", None)),
            last_hist=getattr(fm, "last_hist", None),  # might be big; shallow is fine unless you mutate it
        )

    def _restore(self, snap):
        fm = self.fm
        fm._last = copy.deepcopy(snap["_last"])
        fm.last_p_centerline = None if snap["last_p_centerline"] is None else snap["last_p_centerline"].copy()
        fm.last_tip = None if snap["last_tip"] is None else snap["last_tip"].copy()
        fm.last_info = copy.deepcopy(snap["last_info"])
        fm.last_hist = snap["last_hist"]

    def start_step(self):
        """Freeze the forward model warm-start state for this MPC step."""
        self._base = self._snapshot()

    def _eval_pose8_once(self, p8):
        """Evaluate forward model ONCE and build 6D y."""
        p7 = pose8_quat_to_pose7_rotvec(p8)
        # print("p8.L", p8[7], "p7.L", p7[6])
        # print("[DBG] p7.L =", float(np.asarray(p7).ravel()[6]))
        tip = self.fm(p7)  # should set fm.last_tip + fm.last_p_centerline
        x_tip = np.asarray(tip if tip is not None else self.fm.last_tip, float).reshape(3,)

        C = self.fm.last_p_centerline
        self.last_p_centerline = None if C is None else np.asarray(C, float).copy()
        self.last_tip = x_tip.copy()

        # tangent from end of centerline
        if self.last_p_centerline is None:
            t_tip = np.array([1.0, 0.0, 0.0], float)
        else:
            Cc = self.last_p_centerline
            if Cc.shape[0] == 3:
                p_end, p_prev = Cc[:, -1], Cc[:, -2]
            else:
                p_end, p_prev = Cc[-1, :], Cc[-2, :]
            t_tip = unit(p_end - p_prev)
            if np.linalg.norm(t_tip) < 1e-12:
                t_tip = np.array([1.0, 0.0, 0.0], float)

        return np.hstack([x_tip, t_tip])

    def __call__(self, p8, *, commit=False):
        """
        If commit=False: PURE evaluation (restores base before+after).
        If commit=True : updates base to post-eval (use for plant update once per step).
        """
        if self._base is None:
            # if user forgot, define a baseline anyway
            self._base = self._snapshot()

        # Always start from the step baseline
        self._restore(self._base)
        y = self._eval_pose8_once(p8)

        if commit:
            # adopt the new solver state as baseline for next calls/next step
            self._base = self._snapshot()
        else:
            # restore baseline so subsequent calls are identical
            self._restore(self._base)

        # also mirror attributes for MPC consumption
        self.last_p_centerline = None if self.last_p_centerline is None else self.last_p_centerline.copy()
        return y

def save_step_artifacts(
    *,
    k: int,
    frames_dir,
    log_csv_path,
    u0, p_now, y_now,
    i_ref: int,
    info: dict,
    centerline_tip,
    lumen_C, lumen_R, p0_ur,
    tip_pos, tip_tan, fixed_limits,
    tip_from_centerline=None,
):
    u0 = np.asarray(u0, float).ravel()

    title = (f"k={k:04d} i_ref={i_ref} "
             f"u0=[{u0[0]:+.3f},{u0[1]:+.3f},{u0[2]:+.3f},"
             f"{u0[3]:+.2f},{u0[4]:+.2f},{u0[5]:+.2f},{u0[6]:+.3f}] "
             f"status={info.get('status','?')} infeas={info.get('infeasible',-1)}")

    # ---------- 1) save plot frame ----------
    plot_energy_only_3d(
        centerline_tip,
        lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
        tip=tip_pos,
        tip_from_centerline=tip_from_centerline,
        p_mag=p_now,              # <-- FIXED
        mag_axis="x",
        mag_arrow_len=0.02,
        title=title,              # <-- USE IT
        show=False,
        fixed_limits=fixed_limits,
        zoom_out=1.5,
    )

    # ---------- compute contact-angle metrics ----------
    Cc = np.asarray(lumen_C, float)
    tip_pos = np.asarray(tip_pos, float).reshape(3,)
    tip_tan = np.asarray(tip_tan, float).reshape(3,)

    i_seg, u_seg, c_cl, t_v = _closest_centerline_tangent(Cc, tip_pos)
   
    tip_vessel_angle_deg = _angle_deg(tip_tan, t_v)

    r = tip_pos - c_cl
    etan = float(np.dot(r, t_v))
    r_perp = r - etan * t_v
    rho = float(np.linalg.norm(r_perp))

    clearance = np.nan
    if lumen_R is not None:
        Rr = np.asarray(lumen_R, float).ravel()
        idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr.size - 1))
        clearance = float(Rr[idx_v] - rho)

    # ---------- save frame ----------
    fig = plt.gcf()
    fig_path = frames_dir / f"frame_{k:06d}.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2) append one row to CSV ----------
    p_now = np.asarray(p_now, float).ravel()
    y_now = np.asarray(y_now, float).ravel()

    row = [
        int(k), int(i_ref),
        *u0.tolist(),
        *p_now.tolist(),
        *y_now.tolist(),
        str(info.get("status","")),
        int(info.get("infeasible", -1)),
        float(info.get("pred1_err_xy", np.nan)),
        float(tip_vessel_angle_deg),
        int(i_seg),
        float(u_seg),
        float(rho),
        float(clearance),
    ]
    with open(log_csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)
def rollout_open_loop_from_plan_commit(mpc, p_start, U_seq):
    """
    Roll forward the nonlinear 'plant' using U_seq, committing solver state each step.
    Returns:
        P_nl: (Np, n_p)  parameter trajectory (p1..pNp)
        X_nl: (Np, n)    output trajectory
    """
    p = np.asarray(p_start, float).copy()
    U_seq = np.asarray(U_seq, float)

    # Freeze baseline ONCE for this open-loop simulation
    if hasattr(mpc.forward_tip_fn, "start_step"):
        mpc.forward_tip_fn.start_step()

    P_nl = []
    X_nl = []

    for i in range(U_seq.shape[0]):
        p = mpc._clamp_p(integrate_pose8_body(p, U_seq[i], mpc.dt))
        # IMPORTANT: commit=True so warm-start continues along the trajectory
        try:
            x = np.asarray(mpc.forward_tip_fn(p, commit=True), float).reshape(mpc.n,)
        except TypeError:
            # if forward doesn't accept commit kwarg
            x = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)

        P_nl.append(p.copy())
        X_nl.append(x.copy())

    return np.vstack(P_nl), np.vstack(X_nl)

def quat_flip_180_about_body_axis(q_wxyz, axis_body):
    axis_body = np.asarray(axis_body, float).ravel()
    axis_body = axis_body / (np.linalg.norm(axis_body) + 1e-12)
    # rotvec = pi * axis
    dq = small_rot_quat_wxyz(np.pi * axis_body)
    # body-frame increment: q_new = q ⊗ dq  (same convention as integrate_pose8_body)
    return quat_wxyz_normalize(quat_wxyz_mul(q_wxyz, dq))
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

if __name__ == "__main__":
    pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)

    L0 = 0.04
    dt=0.01
    start_point = np.array([
    0.7181328220229531, -0.7055298925316631, -0.09 ,-3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)
    # start_point = np.array([
    #     .699, -0.836, -0.09,
    #     -2.683,-1.529,+0.024
    # ], float)
    dt=0.01
    # t = start_point[:3]
    # rvec = start_point[3:]

    # R0 = Rot.from_rotvec(rvec)

    # R_spin_localZ = Rot.from_rotvec([0, 0, np.pi])   # 180° about local Z
    # R_new = R0 * R_spin_localZ                     # right-multiply = local rotation

    # start_point = np.hstack([t, R_new.as_rotvec()])
    # start_point[2] +=0.17
    T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    L_tip_full  = 0.04
    L_tip_model = 0.01

    L_model, wire_len_model, tip_len_model = effective_lengths(
        L0
    )

    print(f"[INIT] L_ins={L0:.3f} -> L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}")



    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])  
    s_straight = 0.04

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.05 + s_straight,     
        n_pts=130,                      
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(90.0),
        bend_start=0.0 + s_straight,    
        bend_end=0.03 + s_straight       
    )
    #     lumen_C = make_lumen_centerline_double_turn(
    #     p0_ur, t0,
    #     length=0.05, n_pts=60,
    #     bend_axis=np.array([0., 0., 1.]),
    #     bend1_angle=np.deg2rad(-90.0),
    #     bend1_start=0.015, bend1_end=0.025,
    #     bend2_angle=np.deg2rad(110.0),
    #     bend2_start=0.03, bend2_end=0.04,  # <= length (0.08)
    # )
    # lumen_C = make_lumen_centerline_double_turn(
    #     p0_ur, t0,
    #     length=0.09, n_pts=60,
    #     bend_axis=np.array([0., 0., 1.]),
    #     bend1_angle=np.deg2rad(90.0),
    #     bend1_start=0.02, bend1_end=0.04,
    #     bend2_angle=np.deg2rad(-110.0),
    #     bend2_start=0.05, bend2_end=0.07,  # <= length (0.08)
    # )

    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.0027)
    lumen_path = lumen_C

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur, q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N_nodes=35, maxiter=70,
        L0_init=0.01, dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01
    )
    forward_model_wrong = EnergyMinForwardWithLumen(
        p0_ur=p0_ur, q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N_nodes=35, maxiter=70,
        L0_init=0.01, dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01, use_lumen_jac=False,
    )


    start_point_pose6 = start_point
    p0_pose7 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
                        start_point_pose6[3], start_point_pose6[4], start_point_pose6[5], L0], float)

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)   # now 8D
    p_min = np.array([0.2, -1, start_point[2],  -np.inf, -np.inf, -np.inf, -np.inf, 0.01])
    p_max = np.array([start_point[0]+0.5,  1.5,  start_point[2],  +np.inf, +np.inf, +np.inf, +np.inf, 0.12])
    w_u = np.array([
        1e-5, 1e-5, 1e-4,     # vx,vy,vz
        5e-1, 5e-1, 1e-6,     # wx,wy,wz  (encourage wz)
        1e-4                # dL
    ])    
    w_du = np.array([
    1e-8, 1e-8, 1e-8,     # vx,vy,vz
    1e-8, 1e-8, 1e-8,     # wx,wy,wz  (encourage wz)
    1e-8                 # dL
    ])

    u_max = np.array([ 1, 1, 1, np.deg2rad(60), np.deg2rad(60), np.deg2rad(90),  0.1])
    dr = 5e-3          # 1 mm
    dtheta = np.deg2rad(50.0)
    dL = 1e-3          # 0.5 mm

    eps_u = np.array([
        dr/dt, dr/dt, dr/dt,
        dtheta/dt, dtheta/dt, dtheta/dt,
        dL/dt
    ])
    forward6d = DeterministicForward6D(forward_model)
    forward_6d_wrong = DeterministicForward6D(forward_model_wrong)
    # Use forward6d everywhere: MPC + Jacobian + debug rollouts
    J_fn = lambda p8: numerical_B_y_wrt_u(
        p8, forward6d, dt=dt, eps_u=eps_u, n_out=6
    )

    w_pos_x = 10.0
    w_pos_y = 10.0  
    w_pos_z = 0.0
    w_tan   = 0.001

    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d,
        dt=dt,
        Np=1,
        n_out=6,
        n_u=7,
        n_p=8,
        w_xy=(w_pos_x, w_pos_y, w_pos_z,  w_tan, w_tan, w_tan),
        w_u=w_u,
        w_du=w_du,
        model_mode="lti",
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        N_sqp=1,
        use_offset_free=False,
        overhead_magnet=False,
    )
    mpc.lumen_C = lumen_C
    mpc.lumen_R = lumen_R
    mpc.set_initial_params(p0)
    Np = mpc.Np
    p_test = p0.copy()
    # J_test = J_fn(p_test)
    # print("J shape:", J_test.shape)  

    x_start = mpc.x.copy()
    n=15
    # path = np.linspace(x_start, x_target, n)
    M = lumen_path.shape[0]
    # print(f"print path {M}")
    # ds = np.linalg.norm(np.diff(lumen_C, axis=0), axis=1)
    # print("M =", lumen_C.shape[0])
    # print("Arc length (m) =", ds.sum())
    # print("Expected (m)   =", 0.08 + s_straight)
    # print("Mean ds (m)    =", ds.mean(), "min", ds.min(), "max", ds.max())
    max_steps = 500       # safety limit so it doesn't run forever
    window = 20            # search ahead only (50mm if ds=1mm)

    cursor_state = {"stall": 0}
    x_prev = mpc.x[:3].copy()
    out_root = Path("mpc_run_test_90_contact")
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_root / "log.csv"
    log_meta_path = out_root / "meta.json"
    np.save(out_root / "lumen_C.npy", lumen_C)
    np.save(out_root / "lumen_R.npy", lumen_R)
    # # Save run metadata once (optional)
    meta = dict(
        dt=float(mpc.dt),
        Np=int(mpc.Np),
        u_max=u_max.tolist(),
        p_min=p_min.tolist(),
        p_max=p_max.tolist(),
    )
    with open(log_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # CSV header (write once)
    with open(log_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "k", "i_ref",
            "u0_vx","u0_vy","u0_vz","u0_wx","u0_wy","u0_wz","u0_dL",
            "p_x","p_y","p_z","p_qw","p_qx","p_qy","p_qz","p_L",
            "y_x","y_y","y_z","y_tx","y_ty","y_tz",
            "status","infeasible","pred1_err"
        ])
    svd_S_hist = []
    svd_cond_hist = []
    svd_rank_hist = []
    pred1_hist = []
    k_hist = []
    theta0_hist = []
    thetaMax_hist = []
    trans_hist, omega_hist, dL_hist = [], [], []
    cursor_state = {"stall": 0}
    cur_dbg = {"prog": 0.0, "d_now": 0.0, "i_seg_now": 0, "forced": False}  # safe defaults
    i_ref = 0
    fixed_limits = None
        # keep effort / smoothness
    mpc.R  = np.diag([1e-3]*7)
    mpc.Rd = np.diag([1e-4]*7)  # optional
    theta0_last = np.inf
    # ensure Q tracks only xyz if n_out>3
    # e.g., if n_out=6:
    # mpc.Q  = np.diag([100, 100, 100, 0, 0, 0])
    for k in range(max_steps):
        mpc.mode = "lti"

        # mpc.enable_soft_wall = False
        # mpc.enable_soft_progress = False

        # mpc.w_adv = 0.0
        mpc.enable_mag_center_standoff = True
        mpc.enable_mag_tangent_inline = True
        mpc.enable_dipole_align = True
        # mpc.s_min_progress = 0.0
        # mpc.w_adv = 0.0

        # A) Measure current tip/tangent from current p (plant state)
        p_pre = mpc.p.copy()
        tip_pre, tan_pre, C_pre, y_pre = snapshot_forward(forward6d, p_pre, commit=False)

        # B) Initialize cursor once (optional)
        if k == 0:
            i_ref = advance_cursor_monotone(lumen_path, tip_pre, i_ref, window=window)
        _, _, _, B0 = mpc._build_prediction_mats(p_pre, U_guess=None)



        # D) Set forcing parameters for THIS solve (use last cursor info)
        near_centerline = (cur_dbg["d_now"] <= 0.010)  # meters
        stalling = (cur_dbg["prog"] <= 1e-4)
        stall_cnt = int(cursor_state.get("stall", 0))
        force_ok = stalling and (theta0_last < 40.0)

        w_adv_base = mpc.w_adv_base  # set once when you create mpc
        scale = (1.0 + 2.0 * max(0, stall_cnt - 3)) if force_ok else 1
        mpc.w_adv_eff = w_adv_base * scale   # store separately if you want
        mpc.w_adv = w_adv_base               # keep constant
        mpc.i_ref_last = int(i_ref)
        # E) Step MPC
        p_post, y_post, info = mpc.step(x_meas=None)
        theta0_last = float(info.get("theta0_deg", np.inf))
        theta0_hist.append(float(info.get("theta0_deg", np.nan)))
        thetaMax_hist.append(float(info.get("theta_max_deg", np.nan)))
        # F) Read new tip after applying u0 (mpc.x is updated inside step)
        tip_post = mpc.x[:3].copy()

        # G) Update cursor using ACTUAL motion made this step
        i_ref, cursor_state, cur_dbg = update_progress_cursor_s(
            lumen_C, s_path,
            x_prev=tip_pre,
            x_now=tip_post,
            i_ref=i_ref,
            window=120,
            s_advance=1e-4,
            stall_steps=15,
            force_advance_pts=1,
            dist_ok_max=0.010,
            state=cursor_state,
        )

        # H) Debug/logging now safe: cur_dbg/ref_info/info all defined
        if (k % DBG.every) == 0:
            dbg_print(1,
                f"[CURSOR] k={k:04d} i_ref={i_ref:4d} i_seg={cur_dbg['i_seg_now']:4d} "
                f"Δs={cur_dbg['prog']*1e3:+7.3f}mm d={cur_dbg['d_now']*1e3:6.2f}mm "
                f"stall={cursor_state['stall']:2d} forced={int(cur_dbg['forced'])}"
                f"Advancement weight = {mpc.w_adv_eff}"
            )


        # Optional: snapshot forward at the *post* parameters for extra diagnostics
        tip_post2, tan_post, C_post, y_post2 = snapshot_forward(forward6d, p_post, commit=False)

        # --- histories ---
        k_hist.append(k)
        pred1_hist.append(float(info.get("pred1_err_xy", np.nan)))

        S = info.get("jac_svd_S", None)
        if S is None:
            svd_S_hist.append(np.full((6,), np.nan))
        else:
            S = np.asarray(S, float).ravel()
            if S.size < 6:
                S = np.hstack([S, np.full((6 - S.size,), np.nan)])
            elif S.size > 6:
                S = S[:6]
            svd_S_hist.append(S)

        svd_cond_hist.append(float(info.get("jac_svd_cond", np.nan)))
        svd_rank_hist.append(int(info.get("jac_svd_rank", -1)))

        trans_hist.append(float(info.get("jac_trans_norm", np.nan)))
        omega_hist.append(float(info.get("jac_omega_norm", np.nan)))
        dL_hist.append(float(info.get("jac_dL_norm", np.nan)))

        # post-step tip (you already have tip_post = mpc.x[:3])
        x_post = y_post[:3].copy()

        # predictive-Q sanity warning (unchanged)
        dbg = getattr(mpc, "_dbg_last", {})
        if isinstance(dbg, dict) and dbg.get("mpc_mode") == "predictive_Q":
            if dbg["g_seq"][0] > 0.5 and dbg["s_tan_seq"][0] <= 1.0 + DBG.tol_q_scale:
                dbg_print(1, "[WARN] g high but s_tan not inflated -> check q_tan_scale and Q dims (n>=6).")

        # --- progress diagnostics (use actual pre→post motion) ---
        print("Δtip (mm) =", 1e3 * np.linalg.norm(tip_post - tip_pre))
        print(
            "Δs_proj (mm) =",
            1e3 * float(unit(lumen_C[i_ref+1] - lumen_C[i_ref]) @ (tip_post - tip_pre))
        )

        u0 = info["u0"]

        # centerline tip from info (post-step); keep robust extraction
        centerline_tip = info.get("C", None)
        tip_from_centerline = None
        if centerline_tip is not None:
            ct = np.asarray(centerline_tip)
            tip_from_centerline = ct[:, -1] if (ct.ndim == 2 and ct.shape[0] == 3) else ct[-1, :]
        fixed_limits = plot_energy_only_3d(
            C_pre,
            lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
            tip=tip_pre,
            p_mag=p_post,
            show=False,
            fixed_limits=fixed_limits,
            zoom_out=1.5,   # try 1.5 or 2.0 if you want lots of slack
        )
        tip_tan_pre = np.asarray(y_pre, float).ravel()[3:6]
        # Save artifacts: use C_pre (current-step pre snapshot), and tip_pre/targets_pre
        save_step_artifacts(
            k=k,
            frames_dir=frames_dir,
            log_csv_path=log_csv_path,
            u0=u0,
            p_now=p_post,
            y_now=y_post,
            i_ref=i_ref,
            info=info,
            centerline_tip=C_pre,  # <-- was C_prev
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            p0_ur=p0_ur,
            tip_pos=tip_pre,
            tip_tan=tan_pre,
            fixed_limits=fixed_limits,
            tip_from_centerline=(
                C_pre[:, -1] if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                else (C_pre[-1] if C_pre is not None else None)
            ),
        )

        debug_step_pose7_no_targets(
            k=k,
            p_now=p_post,
            x_now=y_post,      # or mpc.x
            info=info,
            mpc=mpc,
            i_ref=i_ref,
            print_horizon=mpc.Np,
            do_nl_rollout=True,
        )
        pred_targets = predicted_targets_from_info(info, Np=mpc.Np)  # (Np,3) = x1..xNp
        if pred_targets is None:
            pred_targets = np.empty((0,3))

        # prepend the actual next tip so the first “target” is the achieved x1
        pred_plus_actual = np.vstack([tip_post.reshape(1,3), pred_targets]) if pred_targets.size else tip_post.reshape(1,3)
        if (i_ref >= M - 6):
            dbg_print(1, "[DONE] reached final point")
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=pred_plus_actual,
                tip=tip_pre,
                tip_from_centerline=(C_pre[:, -1] if C_pre is not None and C_pre.shape[0]==3 else (C_pre[-1] if C_pre is not None else None)),
                p_mag=p_pre,
                title=f"PRE step k={k} i_ref={i_ref} (tip_pre + actual x1 + predicted horizon)",
                show=True,
            )

                        # --- histories (safe if lists exist, even if short) ---
            K = np.asarray(k_hist)
            pred1 = np.asarray(pred1_hist)        # meters
            Smat = np.asarray(svd_S_hist)         # (T,6)
            cond = np.asarray(svd_cond_hist)

            # 1) pred1_err vs k
            plt.figure()
            plt.plot(K, 1e3 * pred1)  # mm
            plt.xlabel("k")
            plt.ylabel("pred1_err (mm)")
            plt.title("One-step prediction error vs step")
            plt.grid(True)
            plt.show()

            # 2) singular values vs k (log scale)
            plt.figure()
            if Smat.ndim == 2 and Smat.shape[0] == K.size:
                for i in range(Smat.shape[1]):
                    plt.plot(K, Smat[:, i], label=f"σ{i+1}")
            plt.yscale("log")
            plt.xlabel("k")
            plt.ylabel("singular values of B (log)")
            plt.title("Jacobian singular values vs step")
            plt.grid(True)
            plt.legend()
            plt.show()

            # 3) condition number vs k (log)
            plt.figure()
            plt.plot(K, cond)
            plt.yscale("log")
            plt.xlabel("k")
            plt.ylabel("cond(B) (log)")
            plt.title("Jacobian conditioning vs step")
            plt.grid(True)
            plt.show()

            # 4) combined plot: pred1 vs cond(B)
            fig, ax1 = plt.subplots()
            l1, = ax1.plot(K, 1e3 * pred1, label="pred1_err (mm)")
            ax1.set_xlabel("k")
            ax1.set_ylabel("pred1_err (mm)")
            ax1.grid(True)

            ax2 = ax1.twinx()
            l2, = ax2.plot(K, cond, label="cond(B)")
            ax2.set_yscale("log")
            ax2.set_ylabel("cond(B) (log)")

            ax1.legend(handles=[l1, l2], loc="best")
            plt.title("pred1_err vs Jacobian conditioning")
            plt.show()
            lumen_C = np.load("mpc_run_bc_centreline_80_debug/lumen_C.npy")
            lumen_R = np.load("mpc_run_bc_centreline_80_debug/lumen_R.npy")
            summary, series = analyze_run("mpc_run_bc_centreline_80_debug/log.csv", lumen_C, lumen_R, dt=0.01)
            print(summary)
            break
        # if (k % 2) == 0:
        #     theta0 = float(info.get("theta0_deg", np.nan))
        #     g0 = float(np.asarray(info.get("g_seq", [np.nan]))[0]) if "g_seq" in info else np.nan
        #     title = f"PRE k={k} i_ref={i_ref}  theta0={theta0:.1f}deg"
        #     # --- PRE-step visualization: use the "pre" snapshot variables ---
        #     plot_energy_only_3d(
        #         C_pre,
        #         lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
        #         targets=pred_plus_actual,
        #         tip=tip_pre,
        #         tip_from_centerline=(C_pre[:, -1] if C_pre is not None and C_pre.shape[0]==3 else (C_pre[-1] if C_pre is not None else None)),
        #         p_mag=p_pre,
        #         title=title,
        #         show=True,
        #     )
        #     pred_targets = predicted_targets_from_info(info, Np=mpc.Np)
        #     print("tip_pre", tip_pre)
        #     print("pred[0]", pred_targets[0])
        #     print("tip_post", tip_post)
        #     print("||pred[0]-tip_post|| (mm)", 1e3*np.linalg.norm(pred_targets[0]-tip_post))
        #     plt.figure()
        #     plt.plot(k_hist, theta0_hist, label="theta0 (deg)")
        #     plt.plot(k_hist, thetaMax_hist, label="theta_max over horizon (deg)")
        #     plt.axhline(40.0, linestyle="--", label="theta_crit=40deg")
        #     plt.xlabel("k")
        #     plt.ylabel("theta (deg)")
        #     plt.title("Tip–vessel tangent misalignment")
        #     plt.grid(True)
        #     plt.legend()
        #     plt.show()
        #     # --- histories (safe if lists exist, even if short) ---
        #     K = np.asarray(k_hist)
        #     pred1 = np.asarray(pred1_hist)        # meters
        #     Smat = np.asarray(svd_S_hist)         # (T,6)
        #     cond = np.asarray(svd_cond_hist)

        #     # 1) pred1_err vs k
        #     plt.figure()
        #     plt.plot(K, 1e3 * pred1)  # mm
        #     plt.xlabel("k")
        #     plt.ylabel("pred1_err (mm)")
        #     plt.title("One-step prediction error vs step")
        #     plt.grid(True)
        #     plt.show()

            # # 2) singular values vs k (log scale)
            # plt.figure()
            # if Smat.ndim == 2 and Smat.shape[0] == K.size:
            #     for i in range(Smat.shape[1]):
            #         plt.plot(K, Smat[:, i], label=f"σ{i+1}")
            # plt.yscale("log")
            # plt.xlabel("k")
            # plt.ylabel("singular values of B (log)")
            # plt.title("Jacobian singular values vs step")
            # plt.grid(True)
            # plt.legend()
            # plt.show()

            # # 3) condition number vs k (log)
            # plt.figure()
            # plt.plot(K, cond)
            # plt.yscale("log")
            # plt.xlabel("k")
            # plt.ylabel("cond(B) (log)")
            # plt.title("Jacobian conditioning vs step")
            # plt.grid(True)
            # plt.show()

            # # 4) combined plot: pred1 vs cond(B)
            # fig, ax1 = plt.subplots()
            # l1, = ax1.plot(K, 1e3 * pred1, label="pred1_err (mm)")
            # ax1.set_xlabel("k")
            # ax1.set_ylabel("pred1_err (mm)")
            # ax1.grid(True)

            # ax2 = ax1.twinx()
            # l2, = ax2.plot(K, cond, label="cond(B)")
            # ax2.set_yscale("log")
            # ax2.set_ylabel("cond(B) (log)")

            # ax1.legend(handles=[l1, l2], loc="best")
            # plt.title("pred1_err vs Jacobian conditioning")
            # plt.show()
        print("END OF STEP ----------------------------------------------------------------")

    print("UR pose6:", repr(p_post))
    y_dbg = forward6d(p_post, commit=False)
    C_dbg = forward6d.last_p_centerline
    tip_centerline_end = None if C_dbg is None else (C_dbg[:, -1] if C_dbg.shape[0] == 3 else C_dbg[-1, :])
