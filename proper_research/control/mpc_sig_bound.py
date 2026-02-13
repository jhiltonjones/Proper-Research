import numpy as np
import osqp
import copy
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import unpack_pose_ur_rotvec_L
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d
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
def update_progress_cursor(
    path, x_prev, x_now, i_ref,
    window=80,
    ds_path=1e-3,          # your resampled spacing
    s_advance=0.5e-3,      # how much along-path progress to count as "progress"
    stall_steps=8,         # after this many no-progress steps, force advance
    force_advance_pts=1,   # forced cursor increment in points
    state=None
):
    """
    Progress-based cursor update.
    Returns (i_ref_new, state).
    state holds a stall counter.
    """
    if state is None:
        state = {"stall": 0}

    M = path.shape[0]

    # 1) find nearest index in forward window to current tip
    i_near = nearest_index_in_window(path, x_now, i_ref, window=window)

    # 2) compute local tangent at current cursor (or i_near)
    i_tan = int(np.clip(i_near, 0, M-2))
    t = path[i_tan+1] - path[i_tan]
    nt = np.linalg.norm(t)
    if nt < 1e-12:
        t_hat = np.array([1.0, 0.0, 0.0])
    else:
        t_hat = t / nt

    # 3) along-tangent progress (signed)
    dx = (x_now - x_prev).reshape(3,)
    prog = float(dx @ t_hat)          # meters forward (+) / backward (-)

    # 4) if we moved forward enough, allow i_ref to move toward i_near / forward
    progressed = prog > s_advance

    if progressed:
        state["stall"] = 0
        i_ref_new = max(i_ref, i_near)  # monotone forward
    else:
        state["stall"] += 1
        i_ref_new = i_ref

    # 5) stall escape: force cursor forward occasionally
    if state["stall"] >= stall_steps:
        i_ref_new = min(M-1, i_ref_new + int(force_advance_pts))
        state["stall"] = 0

    return int(i_ref_new), state, {"i_near": i_near, "prog": prog, "t_hat": t_hat}
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

    # ---- horizon indices on centerline ----
    idx = i_ref + step_idx * np.arange(Np)
    idx = np.clip(idx, 0, M-1).astype(int)

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

u_max = np.array([ .2, .2, .2, np.deg2rad(60), np.deg2rad(60), np.deg2rad(60),  0.02])
eps_u = np.array([
    1e-2, 1e-2, 1e-2,          # m/s perturbations
    np.deg2rad(1), np.deg2rad(1), np.deg2rad(1),  # rad/s perturbations
    1e-3                        # L rate (m/s)
], float)


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

        # per-step tangent inflation with g_k
        self.q_tan_scale = 50.0         # you already have this
        self.q_pos_scale = 0.0          # optional
        self.debug = True
        self._dbg_last = {}
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
            self.x = np.asarray(self.forward_tip_fn(self.p, commit=True), float).reshape(self.n,)
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
            p_lin   = p_prev.copy()
            p_first = p_seq[0].copy()
            Y_nom = np.vstack([self.forward_tip_fn(p_seq[i]) for i in range(Np)]).reshape(Np, n)
            X_nom = Y_nom.reshape(Np*n, 1)
            # --- build Qtil (adaptive) after B0 exists ---
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

                # Build time-varying Q_seq
                Q_seq = []
                for kq in range(self.Np - 1):
                    gk = float(g_seq[kq])
                    s_tan = 1.0 + self.q_tan_scale * gk
                    s_pos = 1.0 + self.q_pos_scale * gk

                    Qk = Q_base.copy()
                    Qk[0:3, 0:3] *= s_pos
                    Qk[3:6, 3:6] *= s_tan
                    Q_seq.append(Qk)

                # Debug summary ONCE
                s_tan_list = [float(1.0 + self.q_tan_scale * float(g_seq[kq])) for kq in range(self.Np - 1)]
                s_pos_list = [float(1.0 + self.q_pos_scale * float(g_seq[kq])) for kq in range(self.Np - 1)]
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
                s_tan = 1.0 + self.q_tan_scale * g0
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

            # Affine offset so linear model matches nominal at U_guess
            X_aff = X_nom - Mc @ U_guess_vec
            Mc_last = Mc
            X_aff_last = X_aff
            p_seq_last = p_seq
            X_nom_last = X_nom
            # Now prediction is X_aff + Mc U
            H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
            f = 2.0 * (Mc.T @ Qtil @ (X_aff - xref_stack))

            # --- tip keep-out constraint (magnet must stay >= dmin from tip) ---
            A_list, l_list, u_list = [], [], []
            if self.enable_tip_keepout:
                # Current iterate as vector
                U_guess_vec = U_guess.reshape(-1)

                A_ko, l_ko, u_ko = self._build_tip_keepout_constraints(
                    Mc=Mc,
                    X0_stack=X_aff,
                    p0=self.p.copy(),
                    U_guess_vec=U_guess_vec
                )
                if A_ko is not None:
                    A_list.append(A_ko)
                    l_list.append(l_ko)
                    u_list.append(u_ko)
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
                rhs_p = band_stack + (xref_stack - X_aff).reshape(-1)
                rhs_n = band_stack + (X_aff - xref_stack).reshape(-1)


                A_list.append(Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_p)

                A_list.append(-Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_n)

            # stack constraints for OSQP
            if A_list:
                A_osqp = np.vstack(A_list)
                l_osqp = np.concatenate(l_list)
                u_osqp = np.concatenate(u_list)
            else:
                A_osqp = np.zeros((0, Np*m))
                l_osqp = np.zeros(0)
                u_osqp = np.zeros(0)

            # solve QP with warm-start (inside SQP loop we warm-start from last iterate)
            U_warm_vec = U_opt_vec if U_opt_vec is not None else self.U_warm
            U_opt_vec, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=U_warm_vec)
            status_last = status

            infeas = (status not in ("solved", "solved inaccurate")) or (U_opt_vec is None)
            if infeas:
                # if infeasible, break SQP loop and apply zero control
                U_guess = np.zeros((Np, m))
                U_opt_vec = None
                break

            # update guess for next SQP iteration
            U_guess = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)

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
                x_next_true = np.asarray(self.forward_tip_fn(p_next_true, commit=True), float).reshape(self.n,)
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
            pred1_err = float(np.linalg.norm(x_next_true - X_pred[0]))
        pos_err_mm = tan_err = tan_ang_deg = np.nan
        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            pos_err_mm, tan_err, tan_ang_deg = _split_pred_errors(x_next_true, X_pred[0])

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

            if np.isfinite(pred1_err) and (pred1_err > DBG.tol_pred1):
                dbg_print(1, f"[WARN] pred1_err is large ({pred1_err:.4e} m). Model/J or dt scaling may be off.")
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
    """
    Returns B (n_out x 7) s.t. y_next ≈ y_now + B u
    """
    p8 = np.asarray(p8, float).ravel()
    eps_u = np.asarray(eps_u, float).ravel()
    assert eps_u.size == 7

    y0 = np.asarray(forward_y_fn(p8), float).reshape(n_out,)
    B = np.zeros((n_out, 7), float)

    for i in range(7):
        du = np.zeros(7)
        du[i] = eps_u[i]

        p_plus  = integrate_pose8_body(p8, +du, dt)
        p_minus = integrate_pose8_body(p8, -du, dt)

        y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
        y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)

        B[:, i] = (y_plus - y_minus) / (2.0 * eps_u[i])

    return B

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
def debug_step_pose7(k, x_target, xref_seq, p_now, x_now, info, mpc,i_ref=None,
                     print_horizon=4, do_nl_rollout=True):
    xref_seq = np.asarray(xref_seq, float).reshape(-1, 3)
    e_ref0 = xref_seq[0] - x_now
    err_ref0_mm = 1e3 * np.linalg.norm(e_ref0)

    # optional: also show along-horizon terminal ref error
    e_ref_end = xref_seq[-1] - x_now
    err_ref_end_mm = 1e3 * np.linalg.norm(e_ref_end)

    print(
        f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f},{x_now[2]:+.4f}] "
        f"||e_ref0||={err_ref0_mm:.2f}mm ||e_refN||={err_ref_end_mm:.2f}mm "
        f"status={info['status']} infeas={info['infeasible']}"
    )
    u0 = info["u0"]
    X_pred = info.get("X_pred", None)   # (Np,3)
    U_seq  = info.get("U_seq", None)    # (Np,7)
    pred_err = info.get("pred1_err", np.nan)

    p7 = pose8_quat_to_pose7_rotvec(p_now)
    r_src, q_src, L = unpack_pose_ur_rotvec_L(p7)
    rvec = p7[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))

    print(f"   p_now: x={p7[0]:+.3f} y={p7[1]:+.3f} z={p7[2]:+.3f}  "
        f"rotvec=[{rvec[0]:+.3f},{rvec[1]:+.3f},{rvec[2]:+.3f}] |theta|={theta_deg:.1f}deg  "
        f"L={p7[6]:.3f}")

    print("   u0:",
          f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
          f"omega_body=[{u0[3]:+.4f},{u0[4]:+.4f},{u0[5]:+.4f}]  dL={u0[6]:+.5f}")

    print(f"   pred1_err (one-step tip mismatch): {pred_err}")
    if i_ref is not None:
        print(f"   path index i_ref={i_ref}")
    # ---- internal consistency check: should be ~0 ----
    Mc_last = info.get("Mc_last", None)
    X_aff_last = info.get("X_aff_last", None)
    if (Mc_last is not None) and (X_aff_last is not None) and (U_seq is not None):
        n = mpc.n
        U_vec = U_seq.reshape(-1, 1)          # (Np*m,1)
        Mc0 = Mc_last[0:n, :]                 # (n, Np*m)
        x1_from_blocks = (X_aff_last[0:n, :] + Mc0 @ U_vec).reshape(n,)
        if X_pred is not None and np.all(np.isfinite(X_pred)):
            print("   |(X_aff0 + Mc0U) - X_pred[0]| =",
                  np.linalg.norm(x1_from_blocks - X_pred[0]))

    # ---- horizon print: pred vs ref (and optional nominal nonlinear) ----
    xref_seq = np.asarray(xref_seq, float)   # (Np,3)
    Np = xref_seq.shape[0]
    ph = min(print_horizon, Np)

    X_nom_last = info.get("X_nom_last", None)   # optional (Np,3)

    if X_pred is None or not np.all(np.isfinite(X_pred)):
        print("   X_pred NaNs or missing -> infeasible/failed QP")
    else:
        print("   Horizon:  pred (linear)              ref")
        for i in range(ph):
            print(f"   {i:02d}: {X_pred[i]}   {xref_seq[i]}")

        if X_nom_last is not None and np.all(np.isfinite(X_nom_last)):
            print("   Horizon:  nominal (nonlinear @ SQP lin point)")
            for i in range(ph):
                print(f"   {i:02d}: {X_nom_last[i]}")

    # ---- nonlinear open-loop rollout under U_seq ----
    if do_nl_rollout and (U_seq is not None) and np.all(np.isfinite(U_seq)):
        p_start = info["p_prev"]
        P_nl, X_nl = rollout_open_loop_from_plan(mpc, p_start, U_seq)
        print("   Horizon:  NL rollout (true f(p) along applied U_seq)")
        for i in range(min(ph, X_nl.shape[0])):
            if X_pred is not None and np.all(np.isfinite(X_pred)):
                print(f"   {i:02d}: {X_nl[i]}   (lin {X_pred[i]})")
            else:
                print(f"   {i:02d}: {X_nl[i]}")

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

if __name__ == "__main__":
    pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)


    start_point = np.array([
    0.6681328220229531, -0.7055298925316631, 0.1517853768068757, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)
    start_point[2] -=0.25
    T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    L0 = 0.049 
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
    s_straight = 0.02

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.08 + s_straight,     
        n_pts=130,                      
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(120.0),
        bend_start=0.01 + s_straight,    
        bend_end=0.04 + s_straight       
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
            N_nodes=15, maxiter=15,
            L0_init=0.01, dL_internal=0.002
        )




    start_point_pose6 = start_point
    p0_pose7 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
                        start_point_pose6[3], start_point_pose6[4], start_point_pose6[5], L0], float)

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)   # now 8D
    p_min = np.array([0.2, -1, -0.2,  -np.inf, -np.inf, -np.inf, -np.inf, 0.03])
    p_max = np.array([0.85,  1,  1.0,  +np.inf, +np.inf, +np.inf, +np.inf, 0.12])
    w_u = np.array([
        1e-3, 1e-3, 1e-2,     # vx,vy,vz
        1e-2, 1e-2, 1e-6,     # wx,wy,wz  (encourage wz)
        1e-3                 # dL
    ])    
    w_du = np.array([
    1e-3, 1e-3, 1e-3,     # vx,vy,vz
    1e-2, 1e-2, 1e-4,     # wx,wy,wz  (encourage wz)
    1e-3                 # dL
    ])

    w_pos = 1.0
    w_tan = 0.1   # base tangent weight (will be gated up near wall)
    forward6d = DeterministicForward6D(forward_model)

    # Use forward6d everywhere: MPC + Jacobian + debug rollouts
    J_fn = lambda p8: numerical_B_y_wrt_u(
        p8, forward6d, dt=0.05, eps_u=eps_u, n_out=6
    )

    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d, 
        dt=0.05,
        Np=4,
        n_out=6,
        n_u=7,
        n_p=8,
        w_xy=(w_pos, w_pos, w_pos,  w_tan, w_tan, w_tan),
        w_u=w_u,
        w_du=w_du,
        model_mode="lti",
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        N_sqp=4,
        use_offset_free=False
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
    B = J_fn(p0)
    print("B transl norms:", np.linalg.norm(B[:,0]), np.linalg.norm(B[:,1]), np.linalg.norm(B[:,2]))
    print("B omega norms: ", np.linalg.norm(B[:,3]), np.linalg.norm(B[:,4]), np.linalg.norm(B[:,5]))
    print("B dL norm:    ", np.linalg.norm(B[:,6]))
    x_start = mpc.x.copy()
    n=15
    # path = np.linspace(x_start, x_target, n)
    # path is lumen_path (already resampled ~1mm)
    M = lumen_path.shape[0]

    i_ref = 0              # start at beginning (or compute once from initial x)
    max_steps = 5000       # safety limit so it doesn't run forever
    window = 80            # search ahead only (50mm if ds=1mm)

    cursor_state = {"stall": 0}
    x_prev = mpc.x[:3].copy()

    for k in range(max_steps):
        y_now = mpc.x.copy()
        x_now = y_now[:3].copy()
        if k == 0:
            i_ref_start = advance_cursor_monotone(lumen_path, x_now, i_ref, window=window)
            # progress-based update (instead of only "close to ref" rule)
            i_ref, cursor_state, cur_dbg = update_progress_cursor(
                lumen_path, x_prev, x_now, i_ref_start,
                window=window,
                ds_path=1e-3,
                s_advance=0.2e-3,       # 0.2 mm forward motion counts
                stall_steps=10,
                force_advance_pts=1
            )
        else:
            i_ref, cursor_state, cur_dbg = update_progress_cursor(
                lumen_path, x_prev, x_now, i_ref,
                window=window,
                ds_path=1e-3,
                s_advance=0.2e-3,       # 0.2 mm forward motion counts
                stall_steps=10,
                force_advance_pts=1
            )
        mpc.i_ref_last = i_ref
        if (k % DBG.every) == 0:
            dbg_print(1,
                f"[CURSOR] k={k:04d} i_ref={i_ref:4d} i_near={cur_dbg['i_near']:4d} "
                f"prog={cur_dbg['prog']*1e3:+6.3f}mm stall={cursor_state['stall']:2d}"
            )

        t_tip_now = y_now[3:6].copy()

        yref_seq, ref_info = build_yref_from_centerline_tipanchored(
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            x_now=x_now,
            t_tip_now=t_tip_now,     # NEW
            Np=mpc.Np,
            i_ref=i_ref,
            step_idx=1,
            delta=5e-4,
            sigma_m=5e-4,
            d_pull=2e-3,
            p_pull=1.5,
            lam_max=0.9,
            lam_tau=2.0,
            dt=mpc.dt,
            u_max=u_max,
            step_min=5e-4,
            step_max=3e-3,
            theta_crit_deg=40.0,  
            unsafe_ds_scale=0.15,   
            unsafe_lam_max=0.2,      
            freeze_lookahead=True,   
        )
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
        if (k % DBG.every_heavy) == 0:
            dbg_print(2, f"      x_now={fmt_vec(x_now,4)} c0={fmt_vec(ref_info['c0'],4)} d_ref={fmt_vec(ref_info['d_ref'],4)}")
        mpc.g0_last = ref_info["g_wall"]
        p_post, y_post, info = mpc.step(yref_seq, x_meas=None)
        x_post = y_post[:3].copy()
        dbg = getattr(mpc, "_dbg_last", {})
        if isinstance(dbg, dict) and dbg.get("mpc_mode") == "predictive_Q":
            if dbg["g_seq"][0] > 0.5 and dbg["s_tan_seq"][0] <= 1.0 + DBG.tol_q_scale:
                dbg_print(1, "[WARN] g high but s_tan not inflated -> check q_tan_scale and Q dims (n>=6).")
        # update x_prev for next iteration
        x_prev = x_now.copy()

        centerline_tip = info["C"]
        centerline_tip_end_test = None if centerline_tip is None else centerline_tip[:, -1]
        # debug expects xref_seq as (Np,3)
        debug_step_pose7(
            k, lumen_path[-1], yref_seq[:, :3],
            p_post, x_post, info, mpc,
            i_ref=i_ref, print_horizon=mpc.Np, do_nl_rollout=True
        )
        print("||xref0 - x_tip|| =", np.linalg.norm(yref_seq[0,:3] - x_now))
        print("a_dist, a, g_wall =", ref_info["a_dist"], ref_info["a"], ref_info["g_wall"])
        print("d_ref =", ref_info["d_ref"], "ds =", ref_info["ds"])
        print("c0 =", ref_info["c0"], "xref0 =", yref_seq[0,:3])
        # progress check should use position reference
        if np.linalg.norm(yref_seq[0, :3] - x_post) < 3e-3:
            i_ref = min(i_ref + 1, M-1)

        # and also:
        c0 = lumen_C[ref_info["idx"][0]]
        print("c0", c0, "xref0", yref_seq[0, :3], "delta", yref_seq[0,:3]-c0)
        if k % 5==0: 
            y_dbg = forward6d(p_post, commit=False)
            C_dbg = forward6d.last_p_centerline
            tip_centerline_end = None if C_dbg is None else (C_dbg[:, -1] if C_dbg.shape[0] == 3 else C_dbg[-1, :])

            plot_energy_only_3d(
                centerline_tip,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=yref_seq[:, :3],
                tip=x_post, tip_from_centerline=centerline_tip_end_test,
                title=f"Energy-min centreline (k={k}, i_ref={i_ref})"
            )
    print("UR pose6:", repr(p_post))
    y_dbg = forward6d(p_post, commit=False)
    C_dbg = forward6d.last_p_centerline
    tip_centerline_end = None if C_dbg is None else (C_dbg[:, -1] if C_dbg.shape[0] == 3 else C_dbg[-1, :])


    plot_energy_only_3d(
        C_dbg,
        lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
        targets=yref_seq[:, :3],
        tip=x_post, tip_from_centerline=C_dbg[:, -1].copy(),
        title="Energy-min centreline at final MPC pose"
    )
        
        

    
    
