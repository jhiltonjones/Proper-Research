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
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
mag_params = default_magnet_params()
beam_params = default_beam_params()
L_MAG = 0.04

from dataclasses import dataclass

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
def _split_pred_errors(x_true, x_lin):
    x_true = np.asarray(x_true, float).ravel()
    x_lin  = np.asarray(x_lin,  float).ravel()
    # position
    pos_err_m  = float(np.linalg.norm(x_true[:3] - x_lin[:3]))
    pos_err_mm = 1e3 * pos_err_m

    # tangent (unitless)
    tan_true = x_true[3:6]
    tan_lin  = x_lin[3:6]
    tan_err  = float(np.linalg.norm(tan_true - tan_lin))

    # angle (deg), safe even if not perfectly unit length
    nt = float(np.linalg.norm(tan_true))
    np_ = float(np.linalg.norm(tan_lin))
    if nt < 1e-12 or np_ < 1e-12:
        tan_ang_deg = float("nan")
    else:
        c = float(np.clip(np.dot(tan_true/nt, tan_lin/np_), -1.0, 1.0))
        tan_ang_deg = float(np.degrees(np.arccos(c)))
    return pos_err_mm, tan_err, tan_ang_deg

def forward_repeatability_test(forward6d, p8, n_rep=3, label=""):
    """
    Evaluates forward6d(p8, commit=False) multiple times inside the same frozen step baseline.
    """
    print(f"\n[CONSISTENCY] repeatability test {label} reps={n_rep}")
    forward6d.start_step()  # freeze baseline
    ys = []
    for k in range(n_rep):
        y = np.asarray(forward6d(p8, commit=False), float).reshape(6,)
        ys.append(y)
        print(f"  rep{k}: pos={y[:3]} tan={y[3:6]}")
    y0 = ys[0]
    for k in range(1, n_rep):
        dp = np.linalg.norm(ys[k][:3] - y0[:3])
        da = _angle_deg(ys[k][3:6], y0[3:6])
        print(f"  Δ(rep{k}-rep0): pos={dp*1e3:.6f} mm, tan={da:.6f} deg")
def _angle_deg(u, v, eps=1e-12):
    u = unit(u, eps)
    v = unit(v, eps)
    # clamp for numerical safety
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
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
def build_yref_from_centerline_tipanchored(
    lumen_C, lumen_R, x_now, t_tip_now, Np, i_ref, step_idx=1,
    # vessel safety / gating
    delta=5e-4,
    sigma_m=5e-4,
    # pull-in blending (distance-to-centerline)
    d_pull=2e-3,
    p_pull=1.5,
    # horizon blending to centerline
    lam_max=0.9,
    lam_tau=2.0,
    # reachability / step sizing
    dt=0.05,
    u_max=None,
    v_assumed=None,
    step_min=5e-4,
    step_max=3e-3,
    # --- NEW: wall-unsafe soft mode ---
    theta_crit_deg=40.0,      # unsafe if theta > this AND in contact
    unsafe_ds_scale=0.15,     # shrink ds when unsafe (0.1–0.3 typical)
    unsafe_lam_max=0.2,       # reduce centerline pull when unsafe (so you move inward from tip)
    freeze_lookahead=True,    # keep idx constant when unsafe
):
    """
    Tip-anchored reference with wall-contact angle gating (soft).
    - Uses cross-sectional distance to centerline (perp to vessel tangent).
    - If in contact and misaligned (>theta_crit), references push inward and reduce forward pull.
    """
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()
    M = C.shape[0]
    x_tip = np.asarray(x_now, float).ravel()
    t_tip = unit(t_tip_now)
    i0 = closest_index_in_window_monotone(C, x_tip, i_ref, window=120)  # or 80
    idx = i0 + step_idx*np.arange(Np)
    idx = np.clip(idx, 0, M-1).astype(int)
    # ---- horizon indices on centerline ----
    # idx = i_ref + step_idx * np.arange(Np)
    # idx = np.clip(idx, 0, M-1).astype(int)

    c0 = C[idx[0]]
    R0 = float(R[idx[0]])
    t0 = centerline_tangent(C, idx[0])     # vessel tangent

    # ---- cross-sectional geometry (correct wall distance) ----
    r = x_tip - c0
    r_perp = r - (r @ t0) * t0
    d_perp = float(np.linalg.norm(r_perp))

    # inward direction in cross-section plane
    n_in = unit(-r_perp)  # points toward centerline from tip (in cross-section)
    if np.linalg.norm(n_in) < 1e-9:
        # fallback if perfectly on centerline
        n_in = unit(c0 - x_tip) if np.linalg.norm(c0 - x_tip) > 1e-9 else t0.copy()

    # margin to wall (positive = safe)
    m0 = (R0 - delta) - d_perp
    g_wall = sigmoid((-m0) / sigma_m)

    # ---- contact angle between tip tangent and vessel tangent ----
    cang = float(np.clip(np.dot(unit(t_tip), unit(t0)), -1.0, 1.0))
    theta = float(np.arccos(cang))
    theta_deg = float(np.rad2deg(theta))

    in_contact = (m0 <= 0.0)
    unsafe = bool(in_contact and (theta_deg > theta_crit_deg))

    # ---- base pull-in blending (distance-based) ----
    if d_pull <= 1e-12:
        a_dist = 1.0
    else:
        a_dist = np.clip((d_perp / d_pull), 0.0, 1.0) ** float(p_pull)

    # normal behavior: blend toward inward when near wall/off-center
    a = np.clip(0.5 * a_dist + 0.5 * g_wall, 0.0, 1.0)

    # ---- NEW: wall-unsafe soft mode modifications ----
    # If unsafe: heavily bias inward; reduce ds; optionally freeze lookahead; reduce centerline blending.
    if unsafe:
        a = 1.0                         # full inward direction
        ds_scale = float(unsafe_ds_scale)
        lam_max_eff = float(min(lam_max, unsafe_lam_max))
        if freeze_lookahead:
            idx = np.full(Np, idx[0], dtype=int)
    else:
        ds_scale = 1.0
        lam_max_eff = float(lam_max)

    # ---- choose reference direction ----
    # when a=0: follow vessel tangent; when a=1: go inward (cross-section plane)
    d_ref = unit((1.0 - a) * t0 + a * n_in)
    if np.linalg.norm(d_ref) < 1e-9:
        d_ref = t0.copy()

    # ---- reachability heuristic for ds ----
    if u_max is not None:
        u_max = np.asarray(u_max, float).ravel()
        v_max = float(np.linalg.norm(u_max[0:3]))
    else:
        v_max = float(v_assumed) if (v_assumed is not None) else 0.02

    ds_reach = max(step_min, min(step_max, dt * v_max))
    ds = ds_reach * (0.5 + 0.5 * (1.0 - a_dist))
    ds = float(np.clip(ds, step_min, step_max))
    ds *= ds_scale
    ds = float(max(0.2 * step_min, ds))  # don’t go to 0; keep numerically alive

    # ---- build reference ----
    yref = np.zeros((Np, 6), float)
    for k in range(Np):
        ck = C[idx[k]]
        tk = centerline_tangent(C, idx[k])

        s_k = (k + 1) * ds
        x_tip_k = x_tip + s_k * d_ref

        lam_k = lam_max_eff * (1.0 - np.exp(-float(k) / float(max(lam_tau, 1e-6))))
        x_ref_k = (1.0 - lam_k) * x_tip_k + lam_k * ck

        yref[k, 0:3] = x_ref_k
        yref[k, 3:6] = tk
    mode = "UNSAFE" if unsafe else "NORMAL"
    dbg = dict(
        mode=mode,
        idx0=int(idx[0]),
        d_perp=float(d_perp),
        margin0=float(m0),
        g_wall=float(g_wall),
        theta_deg=float(theta_deg),
        a=float(a),
        a_dist=float(a_dist),
        ds=float(ds),
        lam_max_eff=float(lam_max_eff),
        freeze_lookahead=bool(freeze_lookahead and unsafe),
    )

    info = dict(
        idx=idx,
        c0=c0,
        t0=t0,
        d_perp=d_perp,
        margin0=m0,
        g_wall=g_wall,
        a_dist=a_dist,
        a=a,
        n_in=n_in,
        d_ref=d_ref,
        ds=ds,
        lam_max_eff=lam_max_eff,
        unsafe=unsafe,
        theta_deg=theta_deg,
        in_contact=in_contact,
        dbg=dbg,   # <-- only place 'mode' lives
    )
    return yref, info
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
class mpc_controller_tipxy_LTI:
    def __init__(self, *, Jxy_fn, forward_tip_fn,
                 dt=0.05, Np=10,
                 w_xy=(50.0, 50.0, 50.0),
                 w_u=None,
                 w_du=None,
                 band_xy=0.0,
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

        self.np = int(n_p)     # pose dimension
        self.m  = int(n_u)     # control dimension (still 7)
        self.n = n_out
        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

        self.dt = float(dt)
        self.Np = int(Np)

        self.A = np.eye(self.n)


        self.band_xy = float(band_xy)

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
        self.delta_wall = 5e-4        # safety margin (m)
        # --- predictive wall/tangent shaping ---
        self.enable_predictive_Q = True
        self.risk_window = 120          # how far ahead to search on centerline for predicted mapping
        self.theta_crit_deg = 40.0

        # # per-step tangent inflation with g_k
        # self.q_tan_scale = 0        # you already have this
        # self.q_pos_scale = 0          # optional
        # --- NEW: contact-based reweighting (only active near wall) ---
        self.q_tan_gain = 2        # tangent inflation at wall (try 10–100)
        self.q_pos_drop = 0         # fraction to drop position weight at wall (0..0.95)
        self.q_gate_pow = 2.0         # make it kick in mostly near contact (1..4)
        self.w_adv = 1e-5      # start tiny (1e-5 .. 1e-3)
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


    def step(self, xref_seq, x_meas=None):
        """
        xref_seq: (Np,2)
        x_meas: measured tip (2,) for offset-free update. If None, uses internal x.
        """
        # step()
        w_mag = float(self.w_mag_xy)
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")
        # Freeze forward baseline ONCE for this MPC step
        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()
        p_prev = self.p.copy()
        x_prev = self.x.copy()
        # measurement update
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(self.n,)

        # --- offset-free disturbance update ---
        # Use model residual to update d (low-pass filtered)
        if self.use_offset_free and x_meas is not None:
            try:
                x_model = np.asarray(self.forward_tip_fn(self.p, commit=False), float).reshape(self.n,)
            except TypeError:
                x_model = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
            r = self.x - x_model
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        n = self.n
        m = self.m
        Np = self.Np
        Mc_last = None
        X_aff_last = None
        p_seq_last = None
        X_nom_last = None
        B_first = None
        p_lin = None
        p_first = None
        xref_seq = np.asarray(xref_seq, float).reshape(Np, self.n)
        xref_stack = xref_seq.reshape(Np*self.n, 1)
        xk = self.x.reshape(self.n, 1)

        # --- SQP / successive linearization loop ---
        # initial guess U: warm-start if available else zeros
        if self.U_warm is not None and self.U_warm.size == Np*m:
            U_opt_vec = self.U_warm.copy()
            U_guess = U_opt_vec.reshape(Np, m)
        else:
            U_guess = np.zeros((Np, m))
            U_opt_vec = None

        status_last = "init"

        for it in range(self.N_sqp):
            p0 = self.p.copy()
            p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)

            B_first = B0         
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
            p_lin   = p_prev.copy()           

            p_first = p_seq[0].copy()
            Y_nom = np.vstack([self.forward_tip_fn(p_seq[i]) for i in range(Np)]).reshape(Np, n)
            X_nom = Y_nom.reshape(Np*n, 1)
            # --- build Qtil (predictive, time-varying) ---
            if (self.enable_adaptive_Q and self.n >= 6) and self.enable_predictive_Q:
                Q_base = self.Q.copy()

                # Build nominal trajectory at this SQP iterate (already computed below as X_nom)
                # BUT at this point in the loop we haven't computed X_nom yet.
                # So we do a quick nominal rollout *here* (cost: Np forward calls, same as later).
                # If you want to avoid double calls, you can move this block to AFTER X_nom is computed.
                # --- compute nominal predicted outputs ONCE (used for risk + affine) ---


                # Predictive risk along horizon
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

                g_seq = risk["g_k"]
                theta_seq = risk["theta_deg_k"]
                # --- build advancement reward vector in stacked output space ---

                t_vessel = np.asarray(risk["t_vessel_k"], float)   # (Np,3)
                g_seq    = np.asarray(risk["g_k"], float)          # (Np,)

                theta_seq = np.asarray(risk["theta_deg_k"], float)  # (Np,)


                # Build time-varying Q_seq
                Q_seq = []
                for kq in range(self.Np - 1):
                    gk = float(g_seq[kq])                 # 0..1
                    g_eff = gk ** float(self.q_gate_pow)  # sharper onset near contact

                    # --- position drops near wall ---
                    # factor in [1-q_pos_drop, 1]
                    s_pos = self.q_pos_base * (1.0 - float(self.q_pos_drop) * g_eff)

                    # --- tangent increases near wall ---
                    # factor in [1, 1+q_tan_gain]
                    s_tan = self.q_tan_base * (1.0 + float(self.q_tan_gain) * g_eff)

                    Qk = Q_base.copy()
                    Qk[0:3, 0:3] *= s_pos
                    Qk[3:6, 3:6] *= s_tan
                    Q_seq.append(Qk)

                # Debug summary ONCE
                s_tan_list = [float(0.1 + self.q_tan_base * float(g_seq[kq])) for kq in range(self.Np - 1)]
                s_pos_list = [float(1.0 + self.q_pos_base * float(g_seq[kq])) for kq in range(self.Np - 1)]
                self._dbg_last = dict(
                    mpc_mode="predictive_Q",
                    i_ref=int(getattr(self, "i_ref_last", 0)),
                    g_seq=np.array(g_seq, float),
                    theta_seq=np.array(theta_seq, float),
                    margin_seq=np.array(risk["margin_k"], float),
                    s_tan_seq=np.array(s_tan_list, float),
                    s_pos_seq=np.array(s_pos_list, float),
                )

                Qf = Q_base.copy()
                Qtil = self._build_Qtil_from_Qseq(Q_seq, Qf=Qf)

                # Store for debugging / logging
                self._risk_last = risk

            elif self.enable_adaptive_Q and self.n >= 6:
                # fallback: your old scalar gating (still works)
                g0 = float(getattr(self, "g0_last", 0.0))
                Q_base = self.Q.copy()
                s_tan = 0.1 + self.q_tan_scale * g0
                s_pos = 1.0 + self.q_pos_scale * g0
                Q_seq = []
                for kq in range(self.Np - 1):
                    Qk = Q_base.copy()
                    Qk[0:3, 0:3] *= s_pos
                    Qk[3:6, 3:6] *= s_tan
                    Q_seq.append(Qk)
                Qf = Q_base.copy()
                Qtil = self._build_Qtil_from_Qseq(Q_seq, Qf=Qf)
                self._dbg_last = dict(
                    mpc_mode="scalar_g0",
                    g0=float(getattr(self, "g0_last", 0.0)),
                )
            else:
                Qtil = self._compute_Qtil(B0)
                self._dbg_last = dict(mpc_mode="fixed_Q")
        
            
            Rtil = np.kron(np.eye(Np), self.R)

            # delta-u penalty
            if self.Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np-1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            xk = self.x.reshape(self.n, 1)

            # baseline predicted state stack (no control)
            # X0_stack = (Mx @ xk).reshape(Np*n, 1)

            X0_stack = (Mx @ xk).reshape(Np*n, 1)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)


            # objective
            U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)


            # Optional offset-free disturbance: add it consistently to nominal and prediction
            if self.use_offset_free:
                X_nom = X_nom + self._disturbance_stack(self.d)
            X_base = (Mx @ xk)
            # Affine offset so linear model matches nominal at U_guess
            X_aff = X_nom - Mc @ U_guess_vec
            Mc_last = Mc
            X_aff_last = X_aff
            p_seq_last = p_seq
            X_nom_last = X_nom
            # Now prediction is X_aff + Mc U
            H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
            f = 2.0 * (Mc.T @ Qtil @ (X_aff - xref_stack))
            # indices of position rows in stacked output
            idx_pos = pos_row_idx(self.n, self.Np)      # picks x,y,z for each stage
            Mc_pos  = Mc[idx_pos, :]                   # (3Np) x (mNp)
            Xaff_pos = X_aff[idx_pos, :].reshape(-1,1) # (3Np) x 1
            # --- incremental advance reward:  sum_{k=1..Np-1} t_k^T (x_k - x_{k-1}) ---
            # --- incremental advance reward:  sum_{k=0..Np-1} t_k^T (x_{k+1} - x_k)
            # where Xpos stack is [x1..xNp] (so first "increment" is x1 - x_now, constant part ignored)

            t_v = np.asarray(t_vessel, float).copy()   # (Np,3)
            s_min = float(getattr(self, "s_min_progress", 0.0))
            A_list, l_list, u_list = [], [], []
            if s_min > 0.0:
                # Enforce: t0^T (x1 - x0) >= s_min

                # x0 (current) is xk[:3]
                x0 = xk[0:3, :]  # (3,1)

                # x1 predicted pos block corresponds to stage 0 in your stacked outputs
                idx_x1 = np.array([0*self.n + 0, 0*self.n + 1, 0*self.n + 2], dtype=int)
                Mc_x1  = Mc[idx_x1, :]                  # (3, mNp)
                x1_aff = X_aff[idx_x1, :].reshape(3,1)  # (3,1)

                t0 = np.asarray(t_v[0], float).reshape(1,3)   # (1,3)
                a = (t0 @ Mc_x1)                               # (1, mNp)

                # constraint: a U >= b
                b = float(s_min - (t0 @ (x1_aff - x0))[0,0])

                A_list.append(a)
                l_list.append(np.array([b], float))
                u_list.append(np.array([+np.inf], float))
            # Align tangents with forward reference direction (increasing ref index)
            xref_pos = xref_seq[:, :3]  # make sure xref_seq is (Np,6) or at least (Np,3)

            for k in range(Np):
                k2 = min(k+1, Np-1)
                dref = xref_pos[k2] - xref_pos[k]
                nrm = np.linalg.norm(dref)
                if nrm > 1e-12:
                    dref /= nrm
                    if float(np.dot(t_v[k], dref)) < 0.0:
                        t_v[k] *= -1.0

            # Enforce consistency stage-to-stage
            for k in range(1, Np):
                if float(np.dot(t_v[k], t_v[k-1])) < 0.0:
                    t_v[k] *= -1.0

            # Tangent stack for each increment (Np increments)
            t_delta = np.zeros((3*Np, 1))
            for k in range(Np):
                t_delta[3*k:3*k+3, 0] = t_v[k]

            # Difference operator for Xpos = [x1..xNp]
            D1_inc = np.zeros((Np, Np))
            D1_inc[0, 0] = +1.0
            for k in range(1, Np):
                D1_inc[k, k-1] = -1.0
                D1_inc[k, k]   = +1.0
            Dpos = np.kron(D1_inc, np.eye(3))   # (3Np x 3Np)

            # Pull position rows from stacked outputs
            idx_pos = np.array([k*self.n + i for k in range(Np) for i in (0,1,2)], dtype=int)
            Mc_pos  = Mc[idx_pos, :]            # (3Np x mNp)

            # gradient wrt U
            g_adv = Mc_pos.T @ (Dpos.T @ t_delta)  # (mNp x 1)

            w_adv_eff = float(getattr(self, "w_adv_eff", self.w_adv))
            f = f - w_adv_eff * g_adv

            # debug scalar safely (no numpy deprecation warning)
            idx_dL0 = 6
            print("g_adv dL0 =", float(g_adv[idx_dL0, 0]))

            # constraints
            

            # (1) input bounds
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np * self.m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            if self.band_xy > 0.0:
                band = float(self.band_xy)
                band_stack = band * np.ones((Np*n,))
                X_base_stack = (Mx @ xk).reshape(Np*n, 1)
                if self.use_offset_free:
                    X_base_stack += self._disturbance_stack(self.d)

                rhs_p = band_stack + (xref_stack - X_base_stack).reshape(-1)
                rhs_n = band_stack + (X_base_stack - xref_stack).reshape(-1)

                A_list.append(Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_p)

                A_list.append(-Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_n)

            # --- SOFT overhead magnet "stay near tip" reward (quadratic) ---
            if self.overhead_magnet:
                w_mag = float(self.w_mag_xy)   # you already set this at top
                if w_mag > 0.0:
                    r0 = p0[:3].copy()
                    r0_stack = np.tile(r0, self.Np)

                    Pm = build_Pm_world(self.dt, self.Np, m=self.m)

                    idx_pos  = pos_row_idx(self.n, self.Np)
                    Mc_pos   = Mc[idx_pos, :]
                    Xaff_pos = X_aff[idx_pos, :].ravel()

                    keep_xy = np.array([3*k + i for k in range(self.Np) for i in (0,1)], dtype=int)

                    Pm_xy   = Pm[keep_xy, :]
                    Mc_xy   = Mc_pos[keep_xy, :]
                    r0_xy   = r0_stack[keep_xy]
                    xaff_xy = Xaff_pos[keep_xy]

                    A_xy = (Pm_xy - Mc_xy)
                    b_xy = (r0_xy - xaff_xy).reshape(-1, 1)

                    H = H + 2.0 * w_mag * (A_xy.T @ A_xy)
                    f = f + 2.0 * w_mag * (A_xy.T @ b_xy)

                    if self.debug:
                        sep0 = np.linalg.norm(b_xy.reshape(-1,2), axis=1).mean()
                        print("[DBG] mag-tip mean XY sep at U=0 (m):", float(sep0))

            # --- stack constraints for OSQP (ALWAYS) ---
            if A_list:
                A_osqp = np.vstack(A_list)
                l_osqp = np.concatenate(l_list)
                u_osqp = np.concatenate(u_list)
            else:
                A_osqp = np.zeros((0, Np*m))
                l_osqp = np.zeros(0)
                u_osqp = np.zeros(0)

            # --- solve QP (ALWAYS) ---
            U_warm_vec = U_opt_vec if U_opt_vec is not None else self.U_warm
            U_opt_vec, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=U_warm_vec)
            status_last = status

            infeas = (status not in ("solved", "solved inaccurate")) or (U_opt_vec is None)
            if infeas:
                U_guess = np.zeros((Np, m))
                U_opt_vec = None
                break

            U_guess = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)

            # --- debug "after solve" ONLY after a successful solve ---
            if self.overhead_magnet and self.debug and (w_mag > 0.0):
                Uv = U_opt_vec.reshape(-1, 1)
                sep_pred = A_xy @ Uv + b_xy
                sep_pred_mean = np.linalg.norm(sep_pred.reshape(-1,2), axis=1).mean()
                print("[DBG] mag-tip mean XY sep after solve (m):", float(sep_pred_mean))

        # --- choose control to apply ---
        infeas_final = (U_opt_vec is None)

        if infeas_final:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
        else:
            U_seq = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]
            

        # --- build predicted horizon (linear) if we have final linearization ---
        X_pred = np.full((Np, n), np.nan)
        if (not infeas_final) and (Mc_last is not None) and (X_aff_last is not None):
            U_vec = U_opt_vec.reshape(-1, 1)                 # (Np*m,1)
            X_pred_stack = X_aff_last + Mc_last @ U_vec      # (Np*n,1)
            X_pred = X_pred_stack.reshape(Np, n)

        # --- apply one step to parameters ---
        p_next_true = self._clamp_p(integrate_pose8_body(p_prev, u0, self.dt))

        # --- evaluate forward model once ---
        if hasattr(self.forward_tip_fn, "__call__"):
            try:
                x_next_true = np.asarray(self.forward_tip_fn(p_next_true, commit=False), float).reshape(self.n,)
            except TypeError:
                x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
        else:
            x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
        C_attr = getattr(self.forward_tip_fn, "last_p_centerline", None)
        C = None if C_attr is None else np.asarray(C_attr).copy()
        if self.use_offset_free:
            x_next_true = x_next_true + self.d

        # --- commit internal state ONCE ---
        self.p = p_next_true.copy()
        self.x = x_next_true.copy()

        # warm-start
        self.U_warm = np.asarray(U_opt_vec, dtype=float).copy() if not infeas_final else None
        # safe pred1_err
        pred1_err = np.nan
        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            pred1_err = np.linalg.norm(x_next_true[:3] - X_pred[0][:3])
            tan1_err   = np.linalg.norm(x_next_true[3:6] - X_pred[0][3:6])
        pos_err_mm = tan_err = tan_ang_deg = np.nan
        if self.debug and (Mc_last is not None):
            xref0 = xref_stack[0:n].ravel()
            xbase0 = X0_stack[0:n].ravel()
            xaff0  = X_aff_last[0:n].ravel()
            print("xref0", xref0[:3], "xbase0", xbase0[:3], "xaff0", xaff0[:3])

        info = dict(
            status=status_last,
            infeasible=int(infeas_final),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            d=self.d.copy(),
            X_pred=X_pred.copy(),
            U_seq=U_seq.copy(),
            N_sqp=self.N_sqp,
            pred1_err = pred1_err,
            p_prev=p_prev.copy(),
            x_prev = x_prev.copy(),
            C = None if C is None else C.copy(),
            x_next_true = x_next_true.copy(),
            X_pred0 = X_pred[0].copy(),
            pred0_vec_err = (x_next_true - X_pred[0]).copy(),
            B_first=B_first.copy() if B_first is not None else None,
            p_lin=p_lin.copy() if isinstance(p_lin, np.ndarray) else p_lin,
            p_first=p_first.copy() if isinstance(p_first, np.ndarray) else p_first,
            X_aff_last = X_aff_last.copy() if X_aff_last is not None else None,
            Mc_last    = Mc_last.copy()    if Mc_last is not None else None,
            X_nom_last = X_nom_last.reshape(Np, n).copy() if X_nom_last is not None else None,
            p_seq_last = p_seq_last.copy() if p_seq_last is not None else None,
        )
        info["xref_seq"] = xref_seq.copy()
        info["X_base"]   = X_base.reshape(Np, n).copy()
        if self.debug:
            # quick health metrics
            pred1 = float(pred1_err) if np.isfinite(pred1_err) else np.nan
            dbg = self._dbg_last if isinstance(self._dbg_last, dict) else {}
            mode = dbg.get("mpc_mode", "unknown")

            msg = (
            f"[MPC] mode={mode} status={status_last} infeas={int(infeas_final)} "
            f"pos1_err={pos_err_mm:.2f}mm tan1_err={tan_err:.4f} tan1_ang={tan_ang_deg:.2f}deg"
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
            if np.isfinite(pred1_err) and (pred1_err > DBG.tol_pred1):
                dbg_print(1, f"[WARN] pred1_err is large ({pred1_err:.4e} m). Model/J or dt scaling may be off.")
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
        return self.p.copy(), self.x.copy(), info




def forward_cosserat_from_pose_ur_rotvec_L(p, model, *, m_body, L_mag=L_MAG):
    r_src, q_src, L = unpack_pose_ur_rotvec_L(p)

    wire_len = wire_len_from_L(L, L_mag)

    # rebuild magnetisation profile for this L
    m_local_fun = make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0)

    # run forward with per-call wire_len + per-call m_local_fun
    # easiest: temporarily override model.m_local_fun
    old_fun = model.m_local_fun
    try:
        model.m_local_fun = m_local_fun
        out = model.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body, wire_len=wire_len)
    finally:
        model.m_local_fun = old_fun

    if not out["solved"]:
        return np.array([1e3, 1e3, 1e3], float)
    return np.asarray(out["p_tip"], float).reshape(3,)

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
    delta_L = 1e-4  # 0.1 mm
    p_plus = p8.copy();  p_plus[7]  += delta_L
    p_minus = p8.copy(); p_minus[7] -= delta_L
    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)

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
def debug_step_pose7(
    k,
    x_target,
    xref_seq,
    p_now,
    x_now,
    info,
    mpc,
    i_ref=None,
    print_horizon=4,
    do_nl_rollout=True,
):
    """
    Robust debug print for a 6D output MPC:
      y = [x,y,z, tx,ty,tz]
    Inputs:
      xref_seq: typically (Np,3) positional refs (kept for compatibility)
      info["xref_seq"]: (Np,6) full refs (pos+tangent)  <-- preferred
      x_now: can be (3,) or (6,) (kept for compatibility)
      info["x_now"]: should be (6,) if MPC tracks 6D
    """

    # ---- pull reference ----
    xref_full = info.get("xref_seq", None)
    if xref_full is not None:
        xref_full = np.asarray(xref_full, float)
        if xref_full.ndim != 2 or xref_full.shape[1] < 3:
            xref_full = None

    # fallback: user-provided xref_seq (pos only)
    xref_pos_only = np.asarray(xref_seq, float)
    if xref_pos_only.ndim == 1:
        xref_pos_only = xref_pos_only.reshape(1, -1)
    if xref_pos_only.shape[1] != 3:
        # last-ditch: try reshape to (-1,3)
        xref_pos_only = xref_pos_only.reshape(-1, 3)

    # Determine horizon length
    Np = int(getattr(mpc, "Np", xref_pos_only.shape[0]))
    ph = min(int(print_horizon), Np)

    if xref_full is not None and xref_full.shape[0] != Np:
        # keep it usable if caller stored a different length
        Np = xref_full.shape[0]
        ph = min(ph, Np)

    # pos/tan references
    if xref_full is not None and xref_full.shape[1] >= 6:
        xref_pos = xref_full[:, :3]
        xref_tan = xref_full[:, 3:6]
    else:
        xref_pos = xref_pos_only[:Np, :3]
        xref_tan = None

    # ---- pull current state/output ----
    x_now6 = info.get("x_now", None)
    if x_now6 is None:
        x_now6 = np.asarray(x_now, float).ravel()
    else:
        x_now6 = np.asarray(x_now6, float).ravel()

    # If only 3D provided, pad tangent with NaNs so printing still works
    if x_now6.size >= 6:
        pos_now = x_now6[:3]
        tan_now = x_now6[3:6]
    else:
        pos_now = x_now6[:3]
        tan_now = np.full(3, np.nan)

    # ---- errors at step 0 and terminal ----
    e_pos0 = xref_pos[0] - pos_now
    err_pos0_mm = 1e3 * np.linalg.norm(e_pos0)

    if xref_tan is not None and np.all(np.isfinite(tan_now)):
        e_tan0 = xref_tan[0] - tan_now
        err_tan0 = np.linalg.norm(e_tan0)
    else:
        e_tan0 = None
        err_tan0 = np.nan

    e_posN = xref_pos[min(Np - 1, ph - 1)] - pos_now
    err_posN_mm = 1e3 * np.linalg.norm(e_posN)

    # ---- header ----
    status = info.get("status", "?")
    infeas = info.get("infeasible", -1)

    print(
        f"k={k:02d} tip=[{pos_now[0]:+.4f},{pos_now[1]:+.4f},{pos_now[2]:+.4f}] "
        f"e_pos0={err_pos0_mm:.2f}mm e_posN={err_posN_mm:.2f}mm "
        f"e_tan0={err_tan0:.4f} "
        f"status={status} infeas={infeas}"
    )

    # ---- pose ----
    p7 = pose8_quat_to_pose7_rotvec(p_now)
    rvec = p7[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))
    print(
        f"   p_now: x={p7[0]:+.3f} y={p7[1]:+.3f} z={p7[2]:+.3f}  "
        f"rotvec=[{rvec[0]:+.3f},{rvec[1]:+.3f},{rvec[2]:+.3f}] |theta|={theta_deg:.1f}deg  "
        f"L={p7[6]:.3f}"
    )

    # ---- control ----
    u0 = np.asarray(info.get("u0", np.zeros(getattr(mpc, "m", 7))), float).ravel()
    if u0.size >= 7:
        print(
            "   u0:",
            f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
            f"omega_body=[{u0[3]:+.4f},{u0[4]:+.4f},{u0[5]:+.4f}]  dL={u0[6]:+.5f}",
        )
    else:
        print("   u0:", u0)

    pred_err = float(info.get("pred1_err", np.nan))
    print(f"   pred1_err (one-step tip mismatch): {pred_err}")
    if i_ref is not None:
        print(f"   path index i_ref={i_ref}")

    # ---- internal consistency check: (X_aff0 + Mc0U) vs X_pred[0] ----
    X_pred = info.get("X_pred", None)      # expected (Np,6)
    U_seq = info.get("U_seq", None)        # expected (Np,7)
    Mc_last = info.get("Mc_last", None)
    X_aff_last = info.get("X_aff_last", None)

    if X_pred is not None:
        X_pred = np.asarray(X_pred, float)
    if U_seq is not None:
        U_seq = np.asarray(U_seq, float)

    if (Mc_last is not None) and (X_aff_last is not None) and (U_seq is not None):
        Mc_last = np.asarray(Mc_last, float)
        X_aff_last = np.asarray(X_aff_last, float)

        n = int(getattr(mpc, "n", 6))
        U_vec = U_seq.reshape(-1, 1)             # (Np*m,1)
        Mc0 = Mc_last[0:n, :]                    # (n, Np*m)
        x1_from_blocks = (X_aff_last[0:n, :] + Mc0 @ U_vec).reshape(n,)
        if X_pred is not None and X_pred.size >= n:
            xpred0 = X_pred.reshape(Np, -1)[0, :n]
            print("   |(X_aff0 + Mc0U) - X_pred[0]| =", np.linalg.norm(x1_from_blocks - xpred0))

    # ---- horizon print: pred vs ref ----
    if X_pred is None or not np.all(np.isfinite(X_pred)):
        print("   X_pred NaNs or missing -> infeasible/failed QP")
    else:
        X_pred = X_pred.reshape(Np, -1)
        print("   Horizon:  pred(pos)                    ref(pos)")
        for i in range(ph):
            print(f"   {i:02d}: {X_pred[i, :3]}   {xref_pos[i]}")

        # optional tangent display
        if (xref_tan is not None) and (X_pred.shape[1] >= 6):
            print("   Horizon:  pred(tan)                    ref(tan)")
            for i in range(ph):
                print(f"   {i:02d}: {X_pred[i, 3:6]}   {xref_tan[i]}")

        X_nom_last = info.get("X_nom_last", None)
        if X_nom_last is not None:
            X_nom_last = np.asarray(X_nom_last, float)
            if X_nom_last.ndim == 2 and X_nom_last.shape[0] >= ph:
                print("   Horizon:  nominal (nonlinear @ SQP lin point) pos")
                for i in range(ph):
                    print(f"   {i:02d}: {X_nom_last[i, :3]}")

    # ---- nonlinear open-loop rollout ----
    if do_nl_rollout and (U_seq is not None) and np.all(np.isfinite(U_seq)):
        p_start = info.get("p_prev", None)
        if p_start is not None:
            P_nl, X_nl = rollout_open_loop_from_plan_commit(mpc, p_start, U_seq)
            print("   Horizon:  NL rollout COMMIT (plant-like along U_seq)")
            ph2 = min(ph, X_nl.shape[0])
            for i in range(ph2):
                print(f"   {i:02d}: {X_nl[i, :3]}")

            if X_pred is not None and np.all(np.isfinite(X_pred)):
                ph2 = min(ph, X_nl.shape[0], X_pred.shape[0])
                e_pos_m, e_tan_deg = horizon_pred_errors(X_nl, X_pred, ph=ph2)
                print("   Horizon errors vs linear (commit rollout):")
                for i in range(ph2):
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
    ref_info: dict,
    centerline_tip,
    lumen_C, lumen_R, p0_ur,
    tip_pos,
    yref_seq_pos,
    tip_from_centerline=None,
):
    # ---------- 1) save plot frame ----------
    # Put u0 in title (compact)
    u0 = np.asarray(u0, float).ravel()
    title = (f"k={k:04d} i_ref={i_ref} "
             f"u0=[{u0[0]:+.3f},{u0[1]:+.3f},{u0[2]:+.3f},"
             f"{u0[3]:+.2f},{u0[4]:+.2f},{u0[5]:+.2f},{u0[6]:+.3f}] "
             f"status={info.get('status','?')} infeas={info.get('infeasible',-1)}")

    # Your plotting function (assumed to create a matplotlib figure)
    plot_energy_only_3d(
        centerline_tip,
        lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
        targets=yref_seq_pos,
        tip=tip_pos,
        tip_from_centerline=tip_from_centerline,
        p_mag=p_post,                 # NEW
        mag_axis="x",                 # or "z" depending on how your magnet is defined
        mag_arrow_len=0.02,           # 2 cm arrow; tune for visibility
        title=f"Energy-min centreline (k={k}, i_ref={i_ref})",
        show=False,
    )

    # Grab current figure and save
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
        float(info.get("pred1_err", np.nan)),
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

    L0 = 0.041 
    dt=0.01
    start_point = np.array([
    0.7081328220229531, -0.7055298925316631, -0.09 ,-3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)
    # start_point = np.array([
    #     0.8581328220229531-L0, -0.7055298925316631, -0.1,
    #     -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    # ], float)
    # dt=0.01
    # t = start_point[:3]
    # rvec = start_point[3:]

    # R0 = Rot.from_rotvec(rvec)

    # R_spin_localZ = Rot.from_rotvec([0, 0, np.pi])   # 180° about local Z
    # R_new = R0 * R_spin_localZ                     # right-multiply = local rotation

    # start_point = np.hstack([t, R_new.as_rotvec()])
    # start_point[2] +=0.17
    T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    wire_len0 = wire_len_from_L(L0, L_MAG)
    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(wire_len0, mode="axial", alpha_end=0.0),
        m_moment=0.0,
        wire_len = wire_len0,
    )

    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])  
    s_straight = 0.01

    # lumen_C = make_lumen_centerline_turning(
    #     p_start=p0_ur,
    #     t0=t0,
    #     length=0.06 + s_straight,     
    #     n_pts=130,                      
    #     bend_axis=np.array([0.0, 0.0, 1.0]),
    #     bend_angle=np.deg2rad(-40.0),
    #     bend_start=0.01 + s_straight,    
    #     bend_end=0.06 + s_straight       
    # )
    lumen_C = make_lumen_centerline_double_turn(
        p0_ur, t0,
        length=0.08, n_pts=60,
        bend_axis=np.array([0., 0., 1.]),
        bend1_angle=np.deg2rad(90.0),
        bend1_start=0.03, bend1_end=0.05,
        bend2_angle=np.deg2rad(-90.0),
        bend2_start=0.05, bend2_end=0.06, 
    )
    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.004)
    lumen_path = lumen_C

    forward_model = EnergyMinForwardWithLumen(
            p0_ur=p0_ur, q0_ur=q0_ur,
            Kinv_fun=Kbt_inv_profile,
            u_star=np.zeros(3),
            mag_len=beam_params.length_of_mag,
            m_body=m_body,
            lumen_C=lumen_C, lumen_R=lumen_R,
            N_nodes=35, maxiter=70,
            L0_init=0.01, dL_internal=0.002
        )

    forward_model_wrong = EnergyMinForwardWithLumen(
            p0_ur=p0_ur, q0_ur=q0_ur,
            Kinv_fun=Kbt_inv_profile,
            u_star=np.zeros(3),
            mag_len=beam_params.length_of_mag,
            m_body=m_body,
            lumen_C=lumen_C, lumen_R=lumen_R,
            N_nodes=35, maxiter=70,
            L0_init=0.01, dL_internal=0.002, use_lumen_jac=False
        )


    start_point_pose6 = start_point
    p0_pose7 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
                        start_point_pose6[3], start_point_pose6[4], start_point_pose6[5], L0], float)

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)   # now 8D
    p_min = np.array([0.2, -1, start_point[2],  -np.inf, -np.inf, -np.inf, -np.inf, 0.03])
    p_max = np.array([start_point[0],  1.5,  start_point[2],  +np.inf, +np.inf, +np.inf, +np.inf, 0.12])
    w_u = np.array([
        1e-4, 1e-4, 1e-4,     # vx,vy,vz
        5e-1, 5e-1, 1e-6,     # wx,wy,wz  (encourage wz)
        1e-4                # dL
    ])    
    w_du = np.array([
    1e-8, 1e-8, 1e-8,     # vx,vy,vz
    1e-8, 1e-8, 1e-8,     # wx,wy,wz  (encourage wz)
    1e-8                 # dL
    ])

    u_max = np.array([ 1, 1, 1, np.deg2rad(60), np.deg2rad(60), np.deg2rad(90),  1.5])
    dr = 2e-3          # 1 mm
    dtheta = np.deg2rad(50.0)
    dL = 1e-4          # 0.5 mm

    eps_u = np.array([
        dr/dt, dr/dt, dr/dt,
        dtheta/dt, dtheta/dt, dtheta/dt,
        dL/dt
    ])
    forward6d = DeterministicForward6D(forward_model)
    forward_wrong = DeterministicForward6D(forward_model_wrong)
    # Use forward6d everywhere: MPC + Jacobian + debug rollouts
    J_fn = lambda p8: numerical_B_y_wrt_u(
        p8, forward6d, dt=dt, eps_u=eps_u, n_out=6
    )

    w_pos_x = 5.0
    w_pos_y = 5.0  
    w_pos_z = 0.0
    w_tan   = 0.001

    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d,
        dt=dt,
        Np=3,
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
    x_target = np.array([0.80020363, -0.71817534, -0.09914591])
    Np = mpc.Np
    xref_seq = np.tile(x_target, (Np, 1))   
    p_test = p0.copy()
    # J_test = J_fn(p_test)
    # print("J shape:", J_test.shape)  

    x_start = mpc.x.copy()
    n=15
    # path = np.linspace(x_start, x_target, n)
    M = lumen_path.shape[0]
    print(f"print path {M}")
    ds = np.linalg.norm(np.diff(lumen_C, axis=0), axis=1)
    print("M =", lumen_C.shape[0])
    print("Arc length (m) =", ds.sum())
    print("Expected (m)   =", 0.08 + s_straight)
    print("Mean ds (m)    =", ds.mean(), "min", ds.min(), "max", ds.max())
    max_steps = 5000       # safety limit so it doesn't run forever
    window = 20            # search ahead only (50mm if ds=1mm)

    cursor_state = {"stall": 0}
    x_prev = mpc.x[:3].copy()
    out_root = Path("mpc_run_022")
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_root / "log.csv"
    log_meta_path = out_root / "meta.json"

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
    trans_hist, omega_hist, dL_hist = [], [], []
    cursor_state = {"stall": 0}
    cur_dbg = {"prog": 0.0, "d_now": 0.0, "i_seg_now": 0, "forced": False}  # safe defaults
    i_ref = 0

    for k in range(max_steps):

        # A) Measure current tip/tangent from current p (plant state)
        p_pre = mpc.p.copy()
        tip_pre, tan_pre, C_pre, y_pre = snapshot_forward(forward6d, p_pre, commit=False)

        # B) Initialize cursor once (optional)
        if k == 0:
            i_ref = advance_cursor_monotone(lumen_path, tip_pre, i_ref, window=window)

        yref_seq, ref_info = build_yref_from_centerline_tipanchored(
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            x_now=tip_pre,
            t_tip_now=tan_pre,
            Np=mpc.Np,
            i_ref=i_ref,
            step_idx=1,
            delta=1e-4,
            sigma_m=5e-4,
            p_pull=3,
            lam_max=0.9,
            lam_tau=2.0,
            dt=mpc.dt,
            u_max=u_max,
            step_min=1e-4,
            step_max=1e-3,
            theta_crit_deg=40.0,  
            unsafe_ds_scale=0.15,   
            unsafe_lam_max=0.2,      
            freeze_lookahead=True,   
        )
        targets_pre = yref_seq[:, :3].copy()

        # D) Set forcing parameters for THIS solve (use last cursor info)
        stalling = (cur_dbg["prog"] <= 1e-4)           # meters
        near_centerline = (cur_dbg["d_now"] <= 0.010)  # meters
        unsafe = bool(ref_info.get("unsafe", False))

        force_ok = stalling and near_centerline and (not unsafe)
        stall_cnt = int(cursor_state.get("stall", 0))

        mpc.s_min_progress = (0.2e-3 if (force_ok and stall_cnt >= 10) else 0.0)
        mpc.w_adv_eff = (mpc.w_adv * (1.0 + 0.5 * max(0, stall_cnt - 3))) if force_ok else mpc.w_adv

        # E) Step MPC
        p_post, y_post, info = mpc.step(yref_seq, x_meas=None)

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

        # then:
        # p_post, y_post, info = mpc.step(yref_seq, x_meas=None)
        # targets_pre = yref_seq[:, :3].copy()
        if (k % DBG.every) == 0:
            dbg_ref = ref_info["dbg"]
            dbg_print(1,
                f"[REF] k={k:04d} mode={dbg_ref['mode']:<6s} "
                f"i_ref={i_ref:4d} idx0={dbg_ref['idx0']:4d} "
                f"m0={dbg_ref['margin0']*1e3:+7.3f}mm g={dbg_ref['g_wall']:.3f} "
                f"theta={dbg_ref['theta_deg']:.1f}deg "
                f"a={dbg_ref['a']:.2f} ds={dbg_ref['ds']*1e3:.2f}mm "
                f"lam_max={dbg_ref['lam_max_eff']:.2f}"
            )
            if ref_info["unsafe"]:
                if dbg_ref["a"] <= 0.99:
                    dbg_print(1, "[WARN] unsafe but a is not ~1.0 (expected full inward)")
                if dbg_ref["lam_max_eff"] > 0.21:
                    dbg_print(1, "[WARN] unsafe but lam_max_eff not reduced as expected")
        # --- heavy debug (use tip_pre, not x_now) ---
        if (k % DBG.every_heavy) == 0:
            dbg_print(
                2,
                f"      tip_pre={fmt_vec(tip_pre,4)} c0={fmt_vec(ref_info['c0'],4)} "
                f"d_ref={fmt_vec(ref_info['d_ref'],4)}"
            )

        # feed g_wall back to MPC (for scalar gating fallback, if used)
        mpc.g0_last = ref_info["g_wall"]

        # Optional: snapshot forward at the *post* parameters for extra diagnostics
        tip_post2, tan_post, C_post, y_post2 = snapshot_forward(forward6d, p_post, commit=False)

        # --- histories ---
        k_hist.append(k)
        pred1_hist.append(float(info.get("pred1_err", np.nan)))

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
            ref_info=ref_info,
            centerline_tip=C_pre,  # <-- was C_prev
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            p0_ur=p0_ur,
            tip_pos=tip_pre,       # <-- was tip_prev
            yref_seq_pos=targets_pre,
            tip_from_centerline=(
                C_pre[:, -1] if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                else (C_pre[-1] if C_pre is not None else None)
            ),
        )

        # Debug step call: uses the ref that was actually solved
        debug_step_pose7(
            k, lumen_path[-1], yref_seq[:, :3],
            p_post, x_post, info, mpc,
            i_ref=i_ref, print_horizon=mpc.Np, do_nl_rollout=True
        )

        # Replace x_now -> tip_pre everywhere
        print("||xref0 - x_tip|| =", np.linalg.norm(yref_seq[0, :3] - tip_pre))
        print("a_dist, a, g_wall =", ref_info["a_dist"], ref_info["a"], ref_info["g_wall"])
        print("d_ref =", ref_info["d_ref"], "ds =", ref_info["ds"])
        print("c0 =", ref_info["c0"], "xref0 =", yref_seq[0, :3])

        # extra delta check (unchanged but uses correct c0 index)
        c0 = lumen_C[ref_info["idx"][0]]
        print("c0", c0, "xref0", yref_seq[0, :3], "delta", yref_seq[0, :3] - c0)

        if (i_ref >= M - 2):
            dbg_print(1, "[DONE] reached final point")
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=targets_pre,
                tip=tip_pre,
                tip_from_centerline=(
                    C_pre[:, -1] if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                    else (C_pre[-1] if C_pre is not None else None)
                ),
                p_mag=p_pre,   # plant pose BEFORE applying u0
                title=f"PRE step k={k} i_ref={i_ref}",
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
            break
        if (k % 5) == 0:

            # --- PRE-step visualization: use the "pre" snapshot variables ---
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=targets_pre,
                tip=tip_pre,
                tip_from_centerline=(
                    C_pre[:, -1] if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                    else (C_pre[-1] if C_pre is not None else None)
                ),
                p_mag=p_pre,   # plant pose BEFORE applying u0
                title=f"PRE step k={k} i_ref={i_ref}",
                show=False,
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
        print("END OF STEP ----------------------------------------------------------------")

    print("UR pose6:", repr(p_post))
    y_dbg = forward6d(p_post, commit=False)
    C_dbg = forward6d.last_p_centerline
    tip_centerline_end = None if C_dbg is None else (C_dbg[:, -1] if C_dbg.shape[0] == 3 else C_dbg[-1, :])

