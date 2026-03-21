import numpy as np
import osqp
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d, quat_wxyz_to_R, make_lumen_centerline_double_turn, maybe_plot_summary
from scipy.spatial.transform import Rotation as Rot
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen, effective_lengths, DeterministicForward6D
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
from beam_direction_magnetisation.post_processing.debug import debug_step_pose7_no_targets, closest_point_polyline, save_step_artifacts, setup_output_dirs

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
                 model_mode = "ltv",
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
        self.w_dipole_align = 3        # start small: 0.1..10
        self.dipole_body_axis = np.array([1.0, 0.0, 0.0])  # or [0,0,1]

        self.enable_mag_center_standoff = True
        self.w_mag_center_standoff = 20
        self.mag_center_standoff_m = 0.12
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

        # if U_guess is None:
        #     U_guess = np.zeros((Np, m))

        # p_seq = self._p_seq_from_U(p0, U_guess)  
        # B_list = []
        # for i in range(Np):
        #     Ji = np.asarray(self.Jxy_fn(p_seq[i]), float)  # Ji is B_i directly
        #     B_list.append(Ji)

        # Mx, Mc = seq_mat_ltv(self.A, B_list)
        # B0 = B_list[0]
        # return p_seq, Mx, Mc, B0

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

        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        xk = self.x.reshape(n, 1)
        centerline_only = (getattr(self, "mode", "full") == "centerline_only")
        # centerline_only=True

        enable_adv_eff      = ((self.w_adv != 0.0) and (not centerline_only))
        enable_standoff_eff = (self.enable_mag_center_standoff)
        enable_inline_eff   = (getattr(self, "enable_mag_tangent_inline", True))
        enable_dipole_eff   = (self.enable_dipole_align )


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

            g_seq = np.asarray(risk["g_k"], float)
            t_vessel = np.asarray(risk["t_vessel_k"], float)  # (Np,3)

            g_wall = np.asarray(g_seq, float).reshape(Np,)
            theta  = np.asarray(theta_seq, float).reshape(Np,)

            theta_crit = float(self.theta_crit_deg)
            theta_band = float(getattr(self, "theta_gate_band_deg", 5.0))
            g_theta = 1.0 / (1.0 + np.exp(-(theta - theta_crit)/max(theta_band, 1e-6)))
            g_unsafe0 = float(g_wall[0] * g_theta[0])
            Nu = Np * m
            U_guess_vec = U_guess.reshape(-1, 1)
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
            # if H_smooth is not None:
            #     H = H + H_smooth
            f = np.zeros((Np*m, 1), float)
            if dbg_terms is not None:
                zero_f = np.zeros((Np*m, 1), float)
                dbg_terms["effort"] = (H_effort.copy(), zero_f.copy())
                if H_smooth is not None:
                    dbg_terms["smooth"] = (H_smooth.copy(), zero_f.copy())

            U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)

            use_affine_matching = bool(getattr(self, "use_affine_matching", True))

            if use_affine_matching:
                # local first-order model matched to nonlinear rollout at U_guess
                X_aff = X_nom - Mc @ U_guess_vec
            else:
                # pure Jacobian-based prediction from current state, no nonlinear offset matching
                X_aff = (Mx @ xk).reshape(Np*n, 1)

            # ---- Centerline tracking reference (stacked) ----
            # Choose which centerline index sequence you want to track:
            # - idx_k_risk: “closest / risk-mapped” centerline station per stage
            # - or monotone: i_ref_last + k
            Cc = np.asarray(self.lumen_C, float)  # (M,3) assumed
            M  = Cc.shape[0]
            # if centerline_only:
            i0 = int(getattr(self, "i_ref_last", 0))
            look = int(getattr(self, "ref_lookahead_pts", 1))
            idx_ref = np.clip(i0 + look + np.arange(Np), 0, M-1)
            # else:
            #     idx_ref = idx_k_risk
            print(f"REFERENCE {idx_ref}")

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
            H += H_track
            f += f_track

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
                        dbg_terms["advance_model"] = dict(
                            g_adv=g_adv.copy(),
                            t_v=t_v.copy(),
                            w_adv_eff=float(w_adv_eff),
                            s_des_eff=float(s_des_eff),
                        )
            else:
                s_des_eff_local = self.s_des

            # ---- tangent alignment penalty with clearance-dependent weight ----
            enable_tangent_pen = bool(getattr(self, "enable_tangent_penalty", True)) and (t_vessel is not None)

            if enable_tangent_pen and (self.lumen_R is not None):
                # clearance at each stage for weighting
                clearance_m_obj = np.full(Np, np.inf, float)
                Cc_tmp = np.asarray(self.lumen_C, float)
                Rr_tmp = np.asarray(self.lumen_R, float).reshape(-1,)

                for k in range(Np):
                    x_tip_k = X_nom[k*n:k*n+3, 0].reshape(3,)
                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc_tmp[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc_tmp.shape[0] - 2))

                    rho_k = float(np.linalg.norm(x_tip_k - c_closest.reshape(3,)))
                    idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr_tmp.shape[0] - 1))
                    R_k = float(Rr_tmp[idx_v])
                    clearance_m_obj[k] = R_k - rho_k

                theta_ref_deg = float(getattr(self, "theta_ref_deg", 30.0))
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

                    rows_t = np.array([k*n + 3, k*n + 4, k*n + 5], dtype=int)
                    Mc_tk = Mc[rows_t, :]
                    t_aff = X_aff[rows_t, 0].reshape(3,)

                    A_tan[k, :] = -(tv.reshape(1, 3) @ Mc_tk).reshape(-1)
                    b_tan[k, 0] = cos_ref - float(tv @ t_aff)

                    g_clr = 1.0 / (1.0 + np.exp((clearance_m_obj[k] - d_gate) / max(d_band, 1e-9)))
                    w_tan_k[k] = w_tan_min + (w_tan_max - w_tan_min) * g_clr

                W_tan = np.diag(w_tan_k)
                H_tan = 2.0 * (A_tan.T @ W_tan @ A_tan)
                f_tan = 2.0 * (A_tan.T @ W_tan @ b_tan)

                H += H_tan
                f += f_tan

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
            enable_standoff_soft = bool(getattr(self, "enable_standoff_soft", enable_standoff_eff))
            enable_inline_soft   = bool(getattr(self, "enable_inline_soft", enable_inline_eff))
            enable_dipole_soft   = bool(getattr(self, "enable_dipole_soft", enable_dipole_eff)) and (t_vessel is not None)
            

            need_epm_pos_model = enable_standoff_soft or enable_inline_soft

            Nu = Np * m
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

                eps = 1e-9
                for k in range(Np):
                    ck = Cc[idx_k_epm[k], :3].reshape(3,)
                    rk = r_nom[3*k:3*k+3, 0]
                    vk = rk - ck
                    dk = float(np.linalg.norm(vk))

                    uk = np.array([1.0, 0.0, 0.0], float) if dk < eps else (vk / dk)
                    dk = max(dk, eps)

                    Pm_k = Pm[3*k:3*k+3, :]
                    a_k = (uk.reshape(1, 3) @ Pm_k).reshape(Nu,)
                    b_k = (dk - d0) - float(a_k @ U_guess_vec[:, 0])

                    A_s[k, :] = a_k
                    b_s[k, 0] = b_k
            if enable_inline_soft:
                idx_ahead_inline = int(getattr(self, "idx_ahead_inline", 5))
                A_lat = np.zeros((2*Np, Nu), float)
                b_lat = np.zeros((2*Np, 1), float)

                t_prev = None
                for k in range(Np):
                    # stage nominal position at linearization point
                    r_nom_k = r_nom[3*k:3*k+3, :]     # (3,1)
                    Pm_k    = Pm[3*k:3*k+3, :]        # (3,Nu)

                    # nominal TIP position at stage k (used ONLY to anchor vessel frame)
                    x_tip_k = X_nom[k*n:k*n+3, 0].reshape(3,)

                    # closest point on centerline polyline to tip
                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc.shape[0]-2))
                    i_anchor = int(np.clip(i_seg + idx_ahead_inline, 0, Cc.shape[0] - 1))
                    c = Cc[i_anchor, :3].reshape(3, 1)

                    # tangent from closest segment
                    t_k = Cc[i_seg+1, :3] - Cc[i_seg, :3]
                    t_k = t_k / (np.linalg.norm(t_k) + 1e-12)
                    if t_prev is not None and np.dot(t_k, t_prev) < 0.0:
                        t_k = -t_k

                    # nearest vertex only for frame helper
                    idx = int(np.clip(i_seg + (u_seg >= 0.5), 0, Cc.shape[0]-1))


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

            def skew3(v):
                v = np.asarray(v, float).reshape(3,)
                x, y, z = v
                return np.array([[0.0, -z,  y],
                                [z,  0.0, -x],
                                [-y, x,  0.0]], float)

            if enable_dipole_soft:

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

                Pomega_w = build_Pomega_world(p_seq, self.dt, Np, m=m)  # (3Np, Nu)
                Pomega_w = np.asarray(Pomega_w, float)
                Nu = Np*m

                assert Pomega_w.shape == (3*Np, Nu), f"Pomega_w shape {Pomega_w.shape} expected {(3*Np, Nu)}"

                A_align = np.zeros((3*Np, Nu), float)

                for k in range(Np):
                    dk = d_nom[3*k:3*k+3, 0].reshape(3,)
                    Sk = skew3(dk)
                    Pk = Pomega_w[3*k:3*k+3, :]   # (3,Nu)

                    assert Sk.shape == (3,3), f"Sk shape {Sk.shape}, dk={dk}"
                    assert Pk.shape == (3,Nu), f"Pk shape {Pk.shape}"

                    A_align[3*k:3*k+3, :] = -Sk @ Pk

                b_align = (d_nom - t_tar)  # (3Np,1)


            A_list, l_list, u_list = [], [], []

            # (1) symmetric input bounds
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np*m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            # (2) dL lower bound
            dL_back_max = float(getattr(self, "dL_back_max", 0.03))
            A_dL = np.zeros((Np, Np*m), float)
            for k in range(Np):
                A_dL[k, k*m + 6] = 1.0
            A_list.append(A_dL)
            l_list.append(-dL_back_max * np.ones(Np))
            u_list.append(np.full(Np, np.inf))

            # -----------------------
            # Lift to z = [U; s_prog]
            # -----------------------
            enable_theta_eff = bool(getattr(self, "enable_hard_theta", True)) and (t_vessel is not None)

            # threshold: 0.8 mm = 0.0008 m
            theta_clearance_thresh_m = float(getattr(self, "theta_clearance_thresh_m", 0.4e-3))

            # nominal tip-to-wall clearance at each stage
            clearance_m = np.full(Np, np.inf, float)

            if enable_theta_eff and (self.lumen_R is not None):
                Cc = np.asarray(self.lumen_C, float)
                Rr = np.asarray(self.lumen_R, float).reshape(-1,)

                for k in range(Np):
                    x_tip_k = X_nom[k*n:k*n+3, 0].reshape(3,)

                    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x_tip_k)
                    i_seg = int(np.clip(i_seg, 0, Cc.shape[0] - 2))

                    rho_k = float(np.linalg.norm(x_tip_k - c_closest.reshape(3,)))

                    idx_v = int(np.clip(i_seg + (u_seg >= 0.5), 0, Rr.shape[0] - 1))
                    R_k = float(Rr[idx_v])

                    clearance_m[k] = R_k - rho_k

                hard_theta_mask = (clearance_m < theta_clearance_thresh_m)
            else:
                hard_theta_mask = np.zeros(Np, dtype=bool)
            # (3) hard minimum distance between external magnet and tip
            enable_hard_epm_tip_clearance = bool(
                getattr(self, "enable_hard_epm_tip_clearance", True)
            )
            epm_tip_clearance_min_m = float(
                getattr(self, "epm_tip_clearance_min_m", 0.1)
            )

            if enable_hard_epm_tip_clearance:
                # Need magnet position model. Build it if not already available.
                if "Pm" not in locals() or "r_nom" not in locals():
                    Pm = build_Pm_world(self.dt, Np, m=m)          # (3Np, Nu)
                    U_guess_vec = U_guess.reshape(-1, 1)           # (Nu,1)
                    r0 = p0[:3].copy().reshape(3, 1)
                    r0_stack = np.tile(r0, (Np, 1))               # (3Np,1)
                    r_nom = r0_stack + Pm @ U_guess_vec           # (3Np,1)

                A_epm_tip = np.zeros((Np, Nu), float)
                l_epm_tip = np.full(Np, -np.inf, float)
                u_epm_tip = np.full(Np,  np.inf, float)

                eps_dist = 1e-9

                for k in range(Np):
                    # nominal magnet position at stage k
                    r_nom_k = r_nom[3*k:3*k+3, 0].reshape(3,)
                    Pm_k = Pm[3*k:3*k+3, :]                      # (3,Nu)

                    # nominal tip position affine model at stage k
                    rows_x = np.array([k*n + 0, k*n + 1, k*n + 2], dtype=int)
                    Mc_xk = Mc[rows_x, :]                        # (3,Nu)
                    x_nom_k = X_aff[rows_x, 0].reshape(3,) + (Mc_xk @ U_guess_vec).reshape(3,)

                    # nominal separation
                    v_nom = r_nom_k - x_nom_k
                    d_nom = float(np.linalg.norm(v_nom))

                    if d_nom < eps_dist:
                        # fallback direction if nominal positions coincide
                        uhat = np.array([1.0, 0.0, 0.0], float)
                        d_nom = eps_dist
                    else:
                        uhat = v_nom / d_nom

                    # linearized distance:
                    # d(U) ≈ d_nom + uhat^T[(Pm_k - Mc_xk)(U - U_guess)]
                    #      = [uhat^T (Pm_k - Mc_xk)] U + const
                    a_k = (uhat.reshape(1, 3) @ (Pm_k - Mc_xk)).reshape(Nu,)
                    b_k = d_nom - float(a_k @ U_guess_vec[:, 0])

                    # impose: a_k U + b_k >= d_min
                    A_epm_tip[k, :] = a_k
                    l_epm_tip[k] = epm_tip_clearance_min_m - b_k
                    u_epm_tip[k] = np.inf

                A_list.append(A_epm_tip)
                l_list.append(l_epm_tip)
                u_list.append(u_epm_tip)
            Nu = Np * m

            ns_standoff = Np if enable_standoff_soft else 0
            ns_inline   = Np if enable_inline_soft else 0
            ns_dipole   = Np if enable_dipole_soft else 0

            off_standoff = Nu
            off_inline   = off_standoff + ns_standoff
            off_dipole   = off_inline + ns_inline

            ns = ns_standoff + ns_inline + ns_dipole
            nz = Nu + ns

            # objective
            H_z = np.zeros((nz, nz), float)
            H_z[:Nu, :Nu] = H

            f_z = np.zeros((nz, 1), float)
            f_z[:Nu, :] = f

            eps_standoff   = float(getattr(self, "eps_standoff_m", 0.12))
            eps_inline_lat = float(getattr(self, "eps_inline_lat_m", 0.02))
            eps_dipole     = float(getattr(self, "eps_dipole", 1e-2))


            if ns_standoff > 0:
                w_slack_standoff = float(getattr(self, "w_slack_standoff", self.w_mag_center_standoff))
                i0 = off_standoff
                H_z[i0:i0+ns_standoff, i0:i0+ns_standoff] = 2.0 * w_slack_standoff * np.eye(ns_standoff)

            if ns_inline > 0:
                w_slack_inline = float(
                    getattr(self, "w_slack_inline", getattr(self, "w_mag_lat_inline", 4.0))
                )
                i0 = off_inline
                H_z[i0:i0+ns_inline, i0:i0+ns_inline] = 2.0 * w_slack_inline * np.eye(ns_inline)

            if ns_dipole > 0:
                w_slack_dipole = float(getattr(self, "w_slack_dipole", self.w_dipole_align))
                i0 = off_dipole
                H_z[i0:i0+ns_dipole, i0:i0+ns_dipole] = 2.0 * w_slack_dipole * np.eye(ns_dipole)
            
            # existing hard constraints
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
                A_st = np.zeros((2*Np, nz), float)
                l_st = np.full(2*Np, -np.inf, float)
                u_st = np.full(2*Np,  np.inf, float)

                for k in range(Np):
                    col_s = off_standoff + k

                    #  A_s[k] U + b_s[k] <= eps + s_k
                    A_st[2*k, :Nu] = A_s[k, :]
                    A_st[2*k, col_s] = -1.0
                    u_st[2*k] = eps_standoff - b_s[k, 0]

                    # -A_s[k] U - b_s[k] <= eps + s_k
                    A_st[2*k+1, :Nu] = -A_s[k, :]
                    A_st[2*k+1, col_s] = -1.0
                    u_st[2*k+1] = eps_standoff + b_s[k, 0]

                A_osqp = np.vstack([A_osqp, A_st])
                l_osqp = np.concatenate([l_osqp, l_st])
                u_osqp = np.concatenate([u_osqp, u_st])

            if ns_inline > 0:
                A_inl = np.zeros((4*Np, nz), float)
                l_inl = np.full(4*Np, -np.inf, float)
                u_inl = np.full(4*Np,  np.inf, float)

                for k in range(Np):
                    col_s = off_inline + k

                    a_n = A_lat[2*k + 0, :]
                    b_n = b_lat[2*k + 0, 0]

                    a_b = A_lat[2*k + 1, :]
                    b_bv = b_lat[2*k + 1, 0]

                    #  e_n <= eps + s_k
                    A_inl[4*k + 0, :Nu] = a_n
                    A_inl[4*k + 0, col_s] = -1.0
                    u_inl[4*k + 0] = eps_inline_lat - b_n

                    # -e_n <= eps + s_k
                    A_inl[4*k + 1, :Nu] = -a_n
                    A_inl[4*k + 1, col_s] = -1.0
                    u_inl[4*k + 1] = eps_inline_lat + b_n

                    #  e_b <= eps + s_k
                    A_inl[4*k + 2, :Nu] = a_b
                    A_inl[4*k + 2, col_s] = -1.0
                    u_inl[4*k + 2] = eps_inline_lat - b_bv

                    # -e_b <= eps + s_k
                    A_inl[4*k + 3, :Nu] = -a_b
                    A_inl[4*k + 3, col_s] = -1.0
                    u_inl[4*k + 3] = eps_inline_lat + b_bv

                A_osqp = np.vstack([A_osqp, A_inl])
                l_osqp = np.concatenate([l_osqp, l_inl])
                u_osqp = np.concatenate([u_osqp, u_inl])


            if ns_dipole > 0:
                A_dip_soft = np.zeros((6*Np, nz), float)
                l_dip_soft = np.full(6*Np, -np.inf, float)
                u_dip_soft = np.full(6*Np,  np.inf, float)

                for k in range(Np):
                    col_s = off_dipole + k
                    for j in range(3):
                        a = A_align[3*k + j, :]
                        b = b_align[3*k + j, 0]

                        A_dip_soft[6*k + 2*j, :Nu] = a
                        A_dip_soft[6*k + 2*j, col_s] = -1.0
                        u_dip_soft[6*k + 2*j] = eps_dipole - b

                        A_dip_soft[6*k + 2*j + 1, :Nu] = -a
                        A_dip_soft[6*k + 2*j + 1, col_s] = -1.0
                        u_dip_soft[6*k + 2*j + 1] = eps_dipole + b

                A_osqp = np.vstack([A_osqp, A_dip_soft])
                l_osqp = np.concatenate([l_osqp, l_dip_soft])
                u_osqp = np.concatenate([u_osqp, u_dip_soft])
            # -----------------------
            # HARD angle constraint only when clearance < 0.8 mm
            # -----------------------
            if np.any(hard_theta_mask):
                theta_max_deg = float(getattr(self, "theta_max_deg", 40.0))
                cos_max = float(np.cos(np.deg2rad(theta_max_deg)))

                active_k = np.flatnonzero(hard_theta_mask)
                A_theta = np.zeros((active_k.size, nz), float)
                l_theta = np.full(active_k.size, -np.inf, float)
                u_theta = np.full(active_k.size, +np.inf, float)

                for row, k in enumerate(active_k):
                    tv = np.asarray(t_vessel[k], float).reshape(3,)
                    tv /= (np.linalg.norm(tv) + 1e-12)

                    rows_t = np.array([k*n + 3, k*n + 4, k*n + 5], dtype=int)
                    Mc_tk = Mc[rows_t, :]
                    t_aff = X_aff[rows_t, 0].reshape(3,)

                    aU = (tv.reshape(1, 3) @ Mc_tk).reshape(-1)
                    b = float(cos_max - (tv @ t_aff))

                    A_theta[row, :Nu] = aU
                    l_theta[row] = b
                    u_theta[row] = np.inf

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
            # -----------------------
            # Solve QP in z-space (or U-space if ns=0)
            # -----------------------
            Z_warm = None
            if (U_opt_vec is not None) and (ns > 0):
                Z_warm = np.zeros(nz, float)
                Z_warm[:Nu] = U_opt_vec.copy()

            Z_opt, _, status = solve_qp_osqp(H_z, f_z.ravel(), A_osqp, l_osqp, u_osqp, U_warm=Z_warm)
            status_last = status

            U_opt_vec = None

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



        # commit internal state
        self.p = p_next_true.copy()
        self.x = x_next_true.copy()

        # warm start
        if not infeas_final:
            U_seq = U_opt_vec.reshape(Np, m)
            U_shift = np.vstack([U_seq[1:], np.zeros((1, m))])
            self.U_warm = U_shift.reshape(-1)
        else:
            self.U_warm = None

        # one-step prediction error
        pred1_err_xy = np.nan
        pred1_err_xyz = np.nan

        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            e = x_next_true[:3] - X_pred[0][:3]
            pred1_err_xy  = float(np.linalg.norm(e[:2]))   # XY only
            pred1_err_xyz = float(np.linalg.norm(e))       # full XYZ (optional)
            tan1_err = float(np.linalg.norm(x_next_true[3:6] - X_pred[0][3:6])) if (n >= 6 and X_pred.shape[1] >= 6) else np.nan
         # -----------------------
        # Detailed debug breakdown
        # -----------------------

        
        
        
        
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

            # objective terms
            dbg_costs["effort"] = _term_cost(H_effort, np.zeros((Nu, 1)), U_dbg)
            dbg_weights["R_diag"] = np.diag(self.R).copy().tolist()

            if H_smooth is not None:
                dbg_costs["smooth"] = _term_cost(H_smooth, np.zeros((Nu, 1)), U_dbg)
                dbg_weights["Rd_diag"] = np.diag(self.Rd).copy().tolist()
            else:
                dbg_costs["smooth"] = 0.0
                dbg_weights["Rd_diag"] = []

            dbg_costs["track"] = _term_cost(H_track, f_track, U_dbg)  # useful even if commented out
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

            # slack costs
            s_off = 0
            if ns_standoff > 0:
                s_st = s_dbg[s_off:s_off+ns_standoff]
                dbg_costs["slack_standoff"] = float(w_slack_standoff * np.sum(s_st**2))
                s_off += ns_standoff
            else:
                dbg_costs["slack_standoff"] = 0.0

            if ns_inline > 0:
                s_in = s_dbg[s_off:s_off+ns_inline]
                dbg_costs["slack_inline"] = float(w_slack_inline * np.sum(s_in**2))
                s_off += ns_inline
            else:
                dbg_costs["slack_inline"] = 0.0

            if ns_dipole > 0:
                s_dp = s_dbg[s_off:s_off+ns_dipole]
                dbg_costs["slack_dipole"] = float(w_slack_dipole * np.sum(s_dp**2))
                s_off += ns_dipole
            else:
                dbg_costs["slack_dipole"] = 0.0

            dbg_costs["total_objective"] = float(
                0.5 * (np.asarray(Z_opt, float).reshape(-1,1).T @ H_z @ np.asarray(Z_opt, float).reshape(-1,1))[0,0]
                + (f_z.T @ np.asarray(Z_opt, float).reshape(-1,1))[0,0]
            )

            # advancement diagnostics
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

            # tangent diagnostics
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

            # soft constraint residual/violation diagnostics
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
                    s_in = np.asarray(self._slack_last[s0:s0+ns_inline], float)
                else:
                    s_in = np.zeros(ns_inline, float)

                e_n = e[0::2]
                e_b = e[1::2]
                rho_lat = np.sqrt(e_n**2 + e_b**2)

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
                    s_dp = np.asarray(self._slack_last[s0:s0+ns_dipole], float)
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
                U_dbg = np.asarray(U_opt_vec, float).reshape(Nu, 1)
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
            # hard theta diagnostics
            if np.any(hard_theta_mask):
                cos_vals = []
                theta_vals = []
                margins = []

                for k in np.flatnonzero(hard_theta_mask):
                    tv = np.asarray(t_vessel[k], float).reshape(3,)
                    tv /= (np.linalg.norm(tv) + 1e-12)

                    rows_t = np.array([k*n + 3, k*n + 4, k*n + 5], dtype=int)
                    t_pred = (X_aff[rows_t, :].reshape(3,1) + Mc[rows_t, :] @ U_dbg).reshape(3,)
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
                    theta_max_deg=float(theta_max_deg) if np.any(hard_theta_mask) else float(getattr(self, "theta_max_deg", 40.0)),
                ),
                costs=dbg_costs,
                penalties=dbg_pen,
                constraints=dbg_con,
            )

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
                print(f"  weight={ap['weight']:.6e}  mean={ap['mean_directional_progress']:.6e}  "
                      f"min={ap['min_directional_progress']:.6e}  max={ap['max_directional_progress']:.6e}")

            if "tangent" in dbg_pen:
                tp = dbg_pen["tangent"]
                print("[DBG MPC] tangent")
                print(f"  theta_ref_deg={tp['theta_ref_deg']:.2f}")
                print(f"  theta_pred_deg min/mean/max = "
                      f"{tp['theta_pred_deg_min']:.3f} / {tp['theta_pred_deg_mean']:.3f} / {tp['theta_pred_deg_max']:.3f}")
                print(f"  weight min/max = {np.min(tp['weights']):.6e} / {np.max(tp['weights']):.6e}")
                print(f"  clearance min/max [mm] = {1e3*np.min(tp['clearance_m']):.3f} / {1e3*np.max(tp['clearance_m']):.3f}")

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
            pred1_err_xy=float(pred1_err_xy),
            pred1_err_xyz=float(pred1_err_xyz),
            tan1_err=float(tan1_err) if np.isfinite(tan1_err) else np.nan,
            mpc_debug=self._mpc_dbg_last.copy() if hasattr(self, "_mpc_dbg_last") else None,
        )

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


def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    pivot_point = np.array([
        0.8581328220229531, -0.7055298925316631, -0.1,
        -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], dtype=float)

    start_point = np.array([
        0.7181328220229531, -0.7055298925316631, -0.09,
        -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], dtype=float)

    L0 = 0.032
    dt = 0.01
    return pivot_point, start_point, L0, dt


def build_lumen_and_forward_models(pivot_point: np.ndarray, L0: float):
    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )

    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])

    s_straight = 0.03
    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.06 + s_straight,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(90.0),
        bend_start=0.0 + s_straight,
        bend_end=0.03 + s_straight,
    )

    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.005)
    lumen_path = lumen_C

    forward_model = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        N_nodes=35,
        maxiter=70,
        L0_init=0.01,
        dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01,
        use_lumen_jac=True
    )

    forward_model_wrong = EnergyMinForwardWithLumen(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        N_nodes=35,
        maxiter=70,
        L0_init=0.01,
        dL_internal=0.002,
        L_tip_full=0.04,
        L_tip_min=0.01,
        use_lumen_jac=False,
    )

    return p0_ur, q0_ur, lumen_C, lumen_R, lumen_path, s_path, forward_model, forward_model_wrong


def build_controller(start_point: np.ndarray, L0: float, dt: float, forward_model, forward_model_wrong,
                     lumen_C: np.ndarray, lumen_R: np.ndarray):
    start_point_pose6 = start_point
    p0_pose7 = np.array([
        start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
        start_point_pose6[3], start_point_pose6[4], start_point_pose6[5], L0
    ], dtype=float)

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)

    p_min = np.array([0.2, -1, start_point[2], -np.inf, -np.inf, -np.inf, -np.inf, 0.01])
    p_max = np.array([start_point[0] + 0.5, 1.5, start_point[2], +np.inf, +np.inf, +np.inf, +np.inf, 0.12])

    w_u = np.array([
        1e-5, 1e-5, 1e-4,
        5e-1, 5e-1, 1e-6,
        1e-4
    ], dtype=float)

    w_du = np.array([
        1e-8, 1e-8, 1e-8,
        1e-8, 1e-8, 1e-8,
        1e-8
    ], dtype=float)

    u_max = np.array([1, 1, 1, np.deg2rad(60), np.deg2rad(60), np.deg2rad(360), 0.1], dtype=float)

    dr = 5e-3
    dtheta = np.deg2rad(50.0)
    dL = 1e-3

    eps_u = np.array([
        dr / dt, dr / dt, dr / dt,
        dtheta / dt, dtheta / dt, dtheta / dt,
        dL / dt
    ], dtype=float)

    forward6d = DeterministicForward6D(forward_model)
    forward6d_wrong = DeterministicForward6D(forward_model_wrong)

    J_fn = lambda p8: numerical_B_y_wrt_u(
        p8, forward6d, dt=dt, eps_u=eps_u, n_out=6
    )

    w_pos_x = 1.0
    w_pos_y = 1.0
    w_pos_z = 0.0
    w_tan = 0

    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d,
        dt=dt,
        Np=6,
        n_out=6,
        n_u=7,
        n_p=8,
        w_xy=(w_pos_x, w_pos_y, w_pos_z, w_tan, w_tan, w_tan),
        w_u=w_u,
        w_du=w_du,
        model_mode="lti",
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        N_sqp=1,
    )

    mpc.lumen_C = lumen_C
    mpc.lumen_R = lumen_R
    mpc.set_initial_params(p0)

    # keep effort / smoothness
    mpc.R = np.diag([1e-3] * 7)
    mpc.Rd = np.diag([1e-4] * 7)

    return mpc, p0_ur, p0, p_min, p_max, u_max, forward6d, forward6d_wrong




def run_simulation(mpc, forward6d, p0_ur, p0, lumen_C, lumen_R, lumen_path, s_path,
                   frames_dir: Path, log_csv_path: Path):
    Np = mpc.Np
    M = lumen_path.shape[0]

    max_steps = 500
    window = 20

    cursor_state = {"stall": 0}
    cur_dbg = {"prog": 0.0, "d_now": 0.0, "i_seg_now": 0, "forced": False}
    i_ref = 0
    fixed_limits = None
    theta0_last = np.inf

    k_hist = []
    pred1_hist = []
    svd_S_hist = []
    svd_cond_hist = []
    svd_rank_hist = []
    theta0_hist = []
    thetaMax_hist = []
    trans_hist, omega_hist, dL_hist = [], [], []

    for k in range(max_steps):
        mpc.mode = "ltv"
        mpc.enable_soft_progress = True
        mpc.enable_mag_center_standoff = True
        mpc.enable_mag_tangent_inline = True
        mpc.enable_dipole_align = True

        p_pre = mpc.p.copy()
        tip_pre, tan_pre, C_pre, y_pre = snapshot_forward(forward6d, p_pre, commit=False)

        if k == 0:
            i_ref = advance_cursor_monotone(lumen_path, tip_pre, i_ref, window=window)

        _, _, _, B0 = mpc._build_prediction_mats(p_pre, U_guess=None)

        stalling = (cur_dbg["prog"] <= 1e-4)
        stall_cnt = int(cursor_state.get("stall", 0))
        force_ok = stalling and (theta0_last < 40.0)

        w_adv_base = mpc.w_adv_base
        scale = (1.0 + 2.0 * max(0, stall_cnt - 3)) if force_ok else 1.0
        mpc.w_adv_eff = w_adv_base * scale
        mpc.w_adv = w_adv_base
        mpc.i_ref_last = int(i_ref)

        p_post, y_post, info = mpc.step(x_meas=None)

        theta0_last = float(info.get("theta0_deg", np.inf))
        theta0_hist.append(float(info.get("theta0_deg", np.nan)))
        thetaMax_hist.append(float(info.get("theta_max_deg", np.nan)))

        tip_post = mpc.x[:3].copy()

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

        if (k % DBG.every) == 0:
            dbg_print(
                1,
                f"[CURSOR] k={k:04d} i_ref={i_ref:4d} i_seg={cur_dbg['i_seg_now']:4d} "
                f"Δs={cur_dbg['prog']*1e3:+7.3f}mm d={cur_dbg['d_now']*1e3:6.2f}mm "
                f"stall={cursor_state['stall']:2d} forced={int(cur_dbg['forced'])} "
                f"Advancement weight={mpc.w_adv_eff:.6g}"
            )

        tip_post2, tan_post, C_post, y_post2 = snapshot_forward(forward6d_wrong, p_post, commit=False)

        # histories
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

        dbg = getattr(mpc, "_dbg_last", {})
        if isinstance(dbg, dict) and dbg.get("mpc_mode") == "predictive_Q":
            if dbg["g_seq"][0] > 0.5 and dbg["s_tan_seq"][0] <= 1.0 + DBG.tol_q_scale:
                dbg_print(1, "[WARN] g high but s_tan not inflated -> check q_tan_scale and Q dims (n>=6).")

        print("Δtip (mm) =", 1e3 * np.linalg.norm(tip_post - tip_pre))
        print(
            "Δs_proj (mm) =",
            1e3 * float(unit(lumen_C[i_ref + 1] - lumen_C[i_ref]) @ (tip_post - tip_pre))
        )

        u0 = info["u0"]

        fixed_limits = plot_energy_only_3d(
            C_pre,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            p0=p0_ur,
            tip=tip_pre,
            p_mag=p_post,
            show=False,
            fixed_limits=fixed_limits,
            zoom_out=1.5,
        )

        save_step_artifacts(
            k=k,
            frames_dir=frames_dir,
            log_csv_path=log_csv_path,
            u0=u0,
            p_now=p_post,
            y_now=y_post,
            i_ref=i_ref,
            info=info,
            centerline_tip=C_pre,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            p0_ur=p0_ur,
            tip_pos=tip_pre,
            tip_tan=tan_pre,
            fixed_limits=fixed_limits,
            tip_from_centerline=(
                C_pre[:, -1]
                if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                else (C_pre[-1] if C_pre is not None else None)
            ),
        )

        debug_step_pose7_no_targets(
            k=k,
            p_now=p_post,
            x_now=y_post,
            info=info,
            mpc=mpc,
            i_ref=i_ref,
            print_horizon=mpc.Np,
            do_nl_rollout=True,
        )

        pred_targets = predicted_targets_from_info(info, Np=mpc.Np)
        if pred_targets is None:
            pred_targets = np.empty((0, 3))

        pred_plus_actual = (
            np.vstack([tip_post.reshape(1, 3), pred_targets])
            if pred_targets.size
            else tip_post.reshape(1, 3)
        )

        if i_ref >= M - 6:
            dbg_print(1, "[DONE] reached final point")
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                p0=p0_ur,
                targets=pred_plus_actual,
                tip=tip_pre,
                tip_from_centerline=(
                    C_pre[:, -1] if C_pre is not None and C_pre.shape[0] == 3
                    else (C_pre[-1] if C_pre is not None else None)
                ),
                p_mag=p_pre,
                title=f"PRE step k={k} i_ref={i_ref} (tip_pre + actual x1 + predicted horizon)",
                show=False,
            )
            maybe_plot_summary(k_hist, pred1_hist, svd_S_hist, svd_cond_hist)
            break

    return dict(
        k_hist=k_hist,
        pred1_hist=pred1_hist,
        svd_S_hist=svd_S_hist,
        svd_cond_hist=svd_cond_hist,
        svd_rank_hist=svd_rank_hist,
        theta0_hist=theta0_hist,
        thetaMax_hist=thetaMax_hist,
        trans_hist=trans_hist,
        omega_hist=omega_hist,
        dL_hist=dL_hist,
    )


if __name__ == "__main__":
    out_root = Path("/Users/jackhilton-jones/Proper-Research/mpc_run_90_beam_tan")

    pivot_point, start_point, L0, dt = make_initial_poses()

    (
        p0_ur,
        q0_ur,
        lumen_C,
        lumen_R,
        lumen_path,
        s_path,
        forward_model,
        forward_model_wrong,
    ) = build_lumen_and_forward_models(pivot_point, L0)

    (
        mpc,
        p0_ur,
        p0,
        p_min,
        p_max,
        u_max,
        forward6d,
        forward6d_wrong,
    ) = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        forward_model=forward_model,
        forward_model_wrong=forward_model_wrong,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
    )

    frames_dir, log_csv_path = setup_output_dirs(
        out_root=out_root,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        mpc=mpc,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
    )

    run_stats = run_simulation(
        mpc=mpc,
        forward6d=forward6d,
        p0_ur=p0_ur,
        p0=p0,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_path=lumen_path,
        s_path=s_path,
        frames_dir=frames_dir,
        log_csv_path=log_csv_path,
    )
 

