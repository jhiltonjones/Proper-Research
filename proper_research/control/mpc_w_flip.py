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

def stage_cost_wall_progress(
    x_prev, x_now, lumen_C, lumen_R, s_path, i_ref,
    delta=1e-3, w_wall=1e6, w_prog=1.0
):
    """
    Lower is better.
    - wall penalty: hinge on negative margin (quadratic)
    - progress reward: forward progress along centerline (subtract)
    """
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()
    s_path = np.asarray(s_path, float).ravel()

    x_prev = np.asarray(x_prev, float).reshape(3,)
    x_now  = np.asarray(x_now,  float).reshape(3,)

    # progress along centerline (monotone projection)
    s0, i0, _, _ = project_to_polyline_s_monotone(C, s_path, x_prev, i_ref, window=120)
    s1, i1, _, _ = project_to_polyline_s_monotone(C, s_path, x_now,  i0,   window=120)
    prog = max(0.0, float(s1 - s0))

    # wall margin at x_now (use local tangent at closest index)
    ik = closest_index_in_window_monotone(C, x_now, i_ref, window=120)
    m, g, tv = wall_margin_and_gate(C, R, x_now, ik, delta=delta, sigma_m=5e-4)
    viol = max(0.0, -m)  # meters

    J = (w_wall * (viol**2)) - (w_prog * prog)
    return float(J), int(i1)
def quat_rotate_about_body_axis(q_wxyz, axis_body, angle_rad):
    """
    Right-multiply by body-frame rotation: q_new = q ⊗ dq(axis_body, angle).
    Matches your integrate_pose8_body convention (body-frame increment).
    """
    axis_body = np.asarray(axis_body, float).ravel()
    axis_body = axis_body / (np.linalg.norm(axis_body) + 1e-12)
    dq = small_rot_quat_wxyz(axis_body * float(angle_rad))
    return quat_wxyz_normalize(quat_wxyz_mul(q_wxyz, dq))

def quat_rotate_about_body_z(q_wxyz, angle_rad):
    return quat_rotate_about_body_axis(q_wxyz, axis_body=[0.0, 0.0, 1.0], angle_rad=angle_rad)

def quat_flip_about_body_z(q_wxyz):
    # 180 deg around body z
    return quat_rotate_about_body_z(q_wxyz, np.pi)

def rollout_open_loop_fixed_mode(
    *,
    mpc,
    p_start,
    U_seq,
    lumen_C, lumen_R, s_path,
    i_ref_start,
    delta_wall=1e-3,
    w_wall=1e6,
    w_prog=1.0,
):
    """
    Rollout with a FIXED mode (p_start already flipped or not).
    Uses stage_cost_wall_progress(...) to score the horizon.
    Returns: P_nl (Np,8), X_nl (Np,n), J_total (float)
    """
    p = np.asarray(p_start, float).copy()
    U_seq = np.asarray(U_seq, float)
    Np = U_seq.shape[0]

    # Freeze baseline once for deterministic forward
    if hasattr(mpc.forward_tip_fn, "start_step"):
        mpc.forward_tip_fn.start_step()

    P_nl = []
    X_nl = []
    J = 0.0

    i_ref = int(i_ref_start)

    # --- initial tip position for progress computation ---
    try:
        y0 = np.asarray(mpc.forward_tip_fn(p, commit=False), float).reshape(mpc.n,)
    except TypeError:
        y0 = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)
    x_prev = y0[:3].copy()

    for k in range(Np):
        # integrate 1 step
        p = mpc._clamp_p(integrate_pose8_body(p, U_seq[k], mpc.dt))

        # evaluate (pure)
        try:
            y = np.asarray(mpc.forward_tip_fn(p, commit=False), float).reshape(mpc.n,)
        except TypeError:
            y = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)

        x_now = y[:3].copy()

        # score stage
        Jk, i_ref = stage_cost_wall_progress(
            x_prev=x_prev,
            x_now=x_now,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            s_path=s_path,
            i_ref=i_ref,
            delta=delta_wall,   # <-- map name
            w_wall=w_wall,
            w_prog=w_prog,
        )

        J += float(Jk)
        P_nl.append(p.copy())
        X_nl.append(y.copy())

        x_prev = x_now  # advance for next stage

    return np.vstack(P_nl), np.vstack(X_nl), float(J)
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

def reachability_gate_one_step(
    B0, x_tip, i_ref, lumen_C, lumen_R,
    u_max,
    delta=5e-4,
    use_slack=True,
    w_slack=1e6,
    rho_u=1e-6,
    soft_eps=1e-9,
):
    """
    One-step reachability gate.
    Returns:
      u0_star (m,),
      x1_star (3,),
      gate_info dict
    Requires a small QP/LP solver. If you already use OSQP, reuse your solve_qp_osqp.
    """

    B0 = np.asarray(B0, float)
    Bpos = B0[0:3, :]              # (3,m)
    m = Bpos.shape[1]

    x0 = np.asarray(x_tip, float).reshape(3,)
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).reshape(-1)

    ik = int(i_ref)
    c0 = C[ik]
    t0 = centerline_tangent(C, ik)         # (3,) unit
    t0 = np.asarray(t0, float).reshape(3,)

    # Geometry at linearization point (use current tip position)
    r = x0 - c0
    r_perp = r - (r @ t0) * t0
    d = float(np.linalg.norm(r_perp))
    R0 = float(R[ik])

    mbar = (R0 - float(delta)) - d   # margin at x0 (positive safe)
    if d < soft_eps:
        u_perp = np.zeros(3)
    else:
        u_perp = r_perp / d

    # dm/dx = -u_perp
    dm_dx = (-u_perp).reshape(1, 3)   # (1,3)

    # Constraint: m(x1) >= 0
    # x1 = x0 + Bpos u0  => m ≈ mbar + dm_dx (Bpos u0) >= 0
    a = (dm_dx @ Bpos).reshape(1, m)  # (1,m)
    b = -mbar                          # want: a u0 >= b  (since mbar + a u0 >= 0)

    # Progress objective: maximize t0^T Bpos u0  => minimize -(t0^T Bpos) u0
    c_prog = (Bpos.T @ t0.reshape(3,1)).reshape(m,)  # (m,)
    f_u = -c_prog

    # Build QP in z=[u0;s] if slack enabled
    if use_slack:
        nz = m + 1
        H = np.zeros((nz, nz), float)
        H[:m, :m] = rho_u * np.eye(m)        # small regularization
        H[m, m] = 2.0 * w_slack              # strong slack penalty (quadratic)

        f = np.zeros((nz,), float)
        f[:m] = f_u
        # no linear term on slack

        # Constraints in OSQP form: l <= A z <= u
        A_list, l_list, u_list = [], [], []

        # u bounds: -u_max <= u0 <= u_max  -> A=I on u-part
        A_u = np.zeros((m, nz), float)
        A_u[:, :m] = np.eye(m)
        A_list.append(A_u)
        l_list.append(-np.asarray(u_max, float).reshape(m,))
        u_list.append(+np.asarray(u_max, float).reshape(m,))

        # slack >= 0
        A_s = np.zeros((1, nz), float)
        A_s[0, m] = 1.0
        A_list.append(A_s)
        l_list.append(np.array([0.0]))
        u_list.append(np.array([np.inf]))

        # margin: a u0 + s >= b
        A_m = np.zeros((1, nz), float)
        A_m[0, :m] = a
        A_m[0, m] = 1.0
        A_list.append(A_m)
        l_list.append(np.array([b]))
        u_list.append(np.array([np.inf]))

        A = np.vstack(A_list)
        l = np.concatenate(l_list)
        u = np.concatenate(u_list)

        # Solve with your OSQP wrapper
        z_opt, _, status = solve_qp_osqp(H, f, A, l, u, U_warm=None)
        if status not in ("solved", "solved inaccurate") or z_opt is None:
            # fallback: zero
            u0 = np.zeros(m)
            s  = np.inf
            status = "infeasible_gate"
        else:
            u0 = np.asarray(z_opt[:m], float)
            s  = float(z_opt[m])

    else:
        # Hard constraint version (LP/QP); easiest is still OSQP with small rho_u
        nz = m
        H = rho_u * np.eye(m)
        f = f_u.copy()

        A_list, l_list, u_list = [], [], []

        # bounds
        A_list.append(np.eye(m))
        l_list.append(-np.asarray(u_max, float).reshape(m,))
        u_list.append(+np.asarray(u_max, float).reshape(m,))

        # margin: a u0 >= b
        A_list.append(a.reshape(1, m))
        l_list.append(np.array([b]))
        u_list.append(np.array([np.inf]))

        A = np.vstack(A_list)
        l = np.concatenate(l_list)
        u = np.concatenate(u_list)

        u_opt, _, status = solve_qp_osqp(H, f, A, l, u, U_warm=None)
        if status not in ("solved", "solved inaccurate") or u_opt is None:
            u0 = np.zeros(m)
            s  = 0.0
            status = "infeasible_gate"
        else:
            u0 = np.asarray(u_opt, float)
            s  = 0.0

    x1 = x0 + (Bpos @ u0.reshape(m,1)).reshape(3,)

    gate_info = dict(
        status=status,
        i_ref=ik,
        margin0=mbar,
        b=float(b),
        a=a.reshape(-1),
        slack=float(s),
        prog_lin=float(c_prog @ u0),
        d_perp=d,
    )
    return u0, x1, gate_info
import numpy as np

def build_yref_from_centerline_tipanchored(
    lumen_C, lumen_R, x_now, t_tip_now, Np, i_ref, step_idx=1,
    delta=5e-4,
    sigma_m=5e-4,
    d_pull=2e-3,
    p_pull=1.5,
    lam_max=0.9,
    lam_tau=2.0,
    B0=None, use_reach_gate=False, u_max=None,
    dt=0.05,
    v_assumed=None,
    step_min=5e-4,
    step_max=3e-3,
    theta_crit_deg=40.0,
    unsafe_ds_scale=0.15,
    unsafe_lam_max=0.2,
    freeze_lookahead=True,
):
    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()
    M = C.shape[0]

    x_tip = np.asarray(x_now, float).ravel()
    t_tip = unit(t_tip_now)

    # --- find an initial centerline index near the current tip (monotone window) ---
    i0 = closest_index_in_window_monotone(C, x_tip, int(i_ref), window=120)
    i0 = int(np.clip(i0, 0, M - 1))

    # ---- optional: one-step reachability gate (compute x1_gate FIRST) ----
    x1_gate = None
    gate_info = None
    if use_reach_gate and (B0 is not None) and (u_max is not None):
        u0_star, x1_star, gate_info = reachability_gate_one_step(
            B0=B0,
            x_tip=x_tip,
            i_ref=i0,
            lumen_C=C,
            lumen_R=R,
            u_max=u_max,
            delta=delta,
            use_slack=True,
            w_slack=1e6,
            rho_u=1e-6,
        )
        x1_gate = np.asarray(x1_star, float).copy()

    # ---- choose anchor for the rest of the horizon ----
    x_anchor = x_tip if (x1_gate is None) else x1_gate

    # recompute anchor index near x_anchor (keeps monotonicity around i0)
    i_anchor = closest_index_in_window_monotone(C, x_anchor, i0, window=120)
    i_anchor = int(np.clip(i_anchor, 0, M - 1))

    # build horizon indices from i_anchor
    idx = i_anchor + step_idx * np.arange(Np)
    idx = np.clip(idx, 0, M - 1).astype(int)

    # ---- geometry at the anchor (IMPORTANT: use x_anchor, not x_tip) ----
    c0 = C[idx[0]]
    R0 = float(R[idx[0]])
    t0 = centerline_tangent(C, idx[0])  # vessel tangent (unit)

    r = x_anchor - c0
    r_perp = r - (r @ t0) * t0
    d_perp = float(np.linalg.norm(r_perp))

    n_in = unit(-r_perp)
    if np.linalg.norm(n_in) < 1e-9:
        n_in = unit(c0 - x_anchor) if np.linalg.norm(c0 - x_anchor) > 1e-9 else t0.copy()

    m0 = (R0 - float(delta)) - d_perp
    g_wall = sigmoid((-m0) / float(sigma_m))

    # contact angle uses t_tip vs vessel tangent
    cang = float(np.clip(np.dot(unit(t_tip), unit(t0)), -1.0, 1.0))
    theta_deg = float(np.rad2deg(np.arccos(cang)))

    in_contact = (m0 <= 0.0)
    unsafe = bool(in_contact and (theta_deg > float(theta_crit_deg)))

    # ---- distance-based pull-in blending ----
    if d_pull <= 1e-12:
        a_dist = 1.0
    else:
        a_dist = np.clip((d_perp / float(d_pull)), 0.0, 1.0) ** float(p_pull)

    a = np.clip(0.5 * a_dist + 0.5 * g_wall, 0.0, 1.0)

    # unsafe modifications
    if unsafe:
        a = 1.0
        ds_scale = float(unsafe_ds_scale)
        lam_max_eff = float(min(lam_max, unsafe_lam_max))
        if freeze_lookahead:
            idx = np.full(Np, idx[0], dtype=int)
    else:
        ds_scale = 1.0
        lam_max_eff = float(lam_max)

    # direction reference
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
    ds = float(max(0.2 * step_min, ds))

    # ---- build reference ----
    yref = np.zeros((Np, 6), float)
    for k in range(Np):
        ck = C[idx[k]]
        tk = centerline_tangent(C, idx[k])

        if (k == 0) and (x1_gate is not None):
            x_ref_k = x1_gate
        else:
            # if x1_gate exists, stage-1 corresponds to "1 step from anchor" => use k*ds
            s_k = (k if x1_gate is not None else (k + 1)) * ds
            x_tip_k = x_anchor + s_k * d_ref

            kk = (k - 1) if (x1_gate is not None) else k
            kk = max(0, kk)
            lam_k = lam_max_eff * (1.0 - np.exp(-float(kk) / float(max(lam_tau, 1e-6))))

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
        idx0=i0,
        i_anchor=i_anchor,
        x_anchor=x_anchor.copy(),
        x1_gate=None if x1_gate is None else x1_gate.copy(),
        gate_info=gate_info,
        mode=mode,
        c0=c0,
        t0=t0,
        d_perp=float(d_perp),
        margin0=float(m0),
        g_wall=float(g_wall),
        a_dist=float(a_dist),
        a=float(a),
        n_in=n_in,
        d_ref=d_ref,
        ds=float(ds),
        lam_max_eff=float(lam_max_eff),
        unsafe=bool(unsafe),
        theta_deg=float(theta_deg),
        in_contact=bool(in_contact),
        dbg=dbg,
    )


    return yref, info
import numpy as np

def build_yref_from_centerline_tipanchored_progress(
    lumen_C, lumen_R, x_now, t_tip_now, Np, i_ref,
    s_path=None,                 # NEW: arc-length array (len M). If None, computed.
    # vessel safety / gating
    delta=5e-4,
    sigma_m=5e-4,
    # pull-in blending (distance-to-centerline)
    d_pull=2e-3,
    p_pull=1.5,
    # horizon blending to centerline
    lam_max=0.9,
    lam_tau=2.0,
    # reachability gate
    B0=None, use_reach_gate=False, u_max=None,
    # progress step sizing
    dt=0.05,
    v_assumed=None,
    step_min=5e-4,
    step_max=3e-3,
    # wall-unsafe soft mode
    theta_crit_deg=40.0,
    unsafe_ds_scale=0.15,
    unsafe_lam_max=0.2,
    freeze_lookahead=True,
    # mapping params
    window=120,                  # monotone search window in points
):
    """
    Progress-prioritised tip-anchored reference:
    - Picks an anchor index (monotone) and constructs horizon points by advancing in arc-length.
    - Optionally uses a one-step reachability gate to choose x_ref[0] (max progress + (soft) safety).
    - Pulls inward near wall/off-center; in unsafe contact+misalignment it freezes lookahead and shrinks ds.
    Returns:
      yref: (Np,6)  [pos(3), vessel_tangent(3)]
      info: dict
    """

    C = np.asarray(lumen_C, float)
    R = np.asarray(lumen_R, float).ravel()
    M = C.shape[0]
    if M < 2:
        raise ValueError("lumen_C must have at least 2 points")

    if s_path is None:
        s_path = arc_length_param(C)
    else:
        s_path = np.asarray(s_path, float).ravel()
        if s_path.shape[0] != M:
            raise ValueError("s_path must have same length as lumen_C")

    x_tip = np.asarray(x_now, float).ravel()
    t_tip = unit(t_tip_now)

    # --- initial index near current tip (monotone forward) ---
    i0 = closest_index_in_window_monotone(C, x_tip, int(i_ref), window=window)
    i0 = int(np.clip(i0, 0, M-1))

    # --- optional reachability first-step gate (maximises progress along tangent at i0) ---
    x1_gate = None
    gate_info = None
    if use_reach_gate and (B0 is not None) and (u_max is not None):
        _, x1_star, gate_info = reachability_gate_one_step(
            B0=B0,
            x_tip=x_tip,
            i_ref=i0,
            lumen_C=C,
            lumen_R=R,
            u_max=u_max,
            delta=delta,
            use_slack=True,
            w_slack=1e6,
            rho_u=1e-6,
        )
        x1_gate = np.asarray(x1_star, float).copy()

    # --- anchor for the remainder of horizon ---
    x_anchor = x_tip if (x1_gate is None) else x1_gate

    # recompute anchor index near anchor (still monotone around i0)
    i_anchor = closest_index_in_window_monotone(C, x_anchor, i0, window=window)
    i_anchor = int(np.clip(i_anchor, 0, M-1))

    # geometry at anchor
    c0 = C[i_anchor]
    R0 = float(R[i_anchor])
    t0 = centerline_tangent(C, i_anchor)  # unit

    r = x_anchor - c0
    r_perp = r - (r @ t0) * t0
    d_perp = float(np.linalg.norm(r_perp))

    n_in = unit(-r_perp)
    if np.linalg.norm(n_in) < 1e-9:
        n_in = unit(c0 - x_anchor) if np.linalg.norm(c0 - x_anchor) > 1e-9 else t0.copy()

    m0 = (R0 - float(delta)) - d_perp
    g_wall = sigmoid((-m0) / float(sigma_m))

    # contact misalignment gating
    cang = float(np.clip(np.dot(unit(t_tip), unit(t0)), -1.0, 1.0))
    theta_deg = float(np.rad2deg(np.arccos(cang)))
    in_contact = (m0 <= 0.0)
    unsafe = bool(in_contact and (theta_deg > float(theta_crit_deg)))

    # blending parameter toward inward direction
    if d_pull <= 1e-12:
        a_dist = 1.0
    else:
        a_dist = np.clip((d_perp / float(d_pull)), 0.0, 1.0) ** float(p_pull)
    a = np.clip(0.5 * a_dist + 0.5 * g_wall, 0.0, 1.0)

    # unsafe modifications
    if unsafe:
        a = 1.0
        ds_scale = float(unsafe_ds_scale)
        lam_max_eff = float(min(lam_max, unsafe_lam_max))
    else:
        ds_scale = 1.0
        lam_max_eff = float(lam_max)

    # direction reference (used for tip-anchored "push forward/inward" component)
    d_ref = unit((1.0 - a) * t0 + a * n_in)
    if np.linalg.norm(d_ref) < 1e-9:
        d_ref = t0.copy()

    # --- choose ds from physical bounds (progress priority) ---
    if u_max is not None:
        u_max = np.asarray(u_max, float).ravel()
        v_max = float(np.linalg.norm(u_max[0:3]))
    else:
        v_max = float(v_assumed) if (v_assumed is not None) else 0.02

    ds_reach = max(step_min, min(step_max, dt * v_max))
    # shrink ds when off-center; this keeps progress but allows inward correction
    ds = ds_reach * (0.5 + 0.5 * (1.0 - a_dist))
    ds = float(np.clip(ds, step_min, step_max)) * ds_scale
    ds = float(max(0.2 * step_min, ds))

    # --- compute horizon indices by arc-length advance (progress along lumen) ---
    s0 = float(s_path[i_anchor])
    idx = np.zeros(Np, dtype=int)

    # if unsafe and freeze_lookahead: keep all indices the same (progress pauses)
    if unsafe and freeze_lookahead:
        idx[:] = i_anchor
    else:
        for k in range(Np):
            # stage-0 corresponds to next step if no gate; if gate exists, stage-0 is already x1_gate.
            # We still use lumen tangent at increasing s for stages.
            kk = k if (x1_gate is None) else k  # keep tangent progression
            s_k = s0 + (kk + 1) * ds
            # search forward to keep monotone
            i_hi = min(M-1, i_anchor + window)
            j = int(np.searchsorted(s_path[i_anchor:i_hi+1], s_k, side="left")) + i_anchor
            idx[k] = int(np.clip(j, i_anchor, M-1))

    # --- build reference ---
    yref = np.zeros((Np, 6), float)

    for k in range(Np):
        ck = C[idx[k]]
        tk = centerline_tangent(C, idx[k])

        if (k == 0) and (x1_gate is not None):
            x_ref_k = x1_gate
        else:
            # Tip-anchored forward/inward propagation from x_anchor:
            # if x1_gate exists, stage-1 should start at 1*ds from anchor -> use k*ds
            step_count = (k + 1) if (x1_gate is None) else k
            s_k_tip = float(step_count) * ds
            x_tip_k = x_anchor + s_k_tip * d_ref

            # blend to centerline point ck with lam schedule
            kk = (k if x1_gate is None else max(0, k-1))
            lam_k = lam_max_eff * (1.0 - np.exp(-float(kk) / float(max(lam_tau, 1e-6))))
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
        idx0=i0,
        i_anchor=i_anchor,
        x_anchor=x_anchor.copy(),
        x1_gate=None if x1_gate is None else x1_gate.copy(),
        gate_info=gate_info,
        c0=c0,
        t0=t0,
        d_perp=float(d_perp),
        margin0=float(m0),
        g_wall=float(g_wall),
        a_dist=float(a_dist),
        a=float(a),
        n_in=n_in,
        d_ref=d_ref,
        ds=float(ds),
        lam_max_eff=float(lam_max_eff),
        unsafe=bool(unsafe),
        theta_deg=float(theta_deg),
        in_contact=bool(in_contact),
        freeze_lookahead=bool(freeze_lookahead and unsafe),
        s0=s0,
        dbg =dbg,
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
        self.w_adv = 1e-3     # start tiny (1e-5 .. 1e-3)
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
        # --- flip rollout mode selection ---
        self.enable_flip_rollout = True
        self.flip_spin_per_step_rad = 0.0   # optional: spin around body-z during rollout (rad/step)
        self.flip_w_wall = 1e6              # wall violation weight used in stage_cost_wall_progress
        self.flip_w_prog = 1.0              # progress weight used in stage_cost_wall_progress
        self.s_path = None                  # must be set externally: mpc.s_path = s_path
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
        J_total = float(0.5 * (z.T @ H @ z) + (f.T @ z))

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
            Y_nom = np.vstack([self.forward_tip_fn(p_seq[i]) for i in range(Np)]).reshape(Np, n)
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
            theta_seq = np.asarray(risk["theta_deg_k"], float)     # (Np,)
            info_theta0 = float(theta_seq[0])
            info_thetamax = float(np.max(theta_seq))
            g_seq = np.asarray(risk["g_k"], float)
            t_vessel = np.asarray(risk["t_vessel_k"], float)  # (Np,3)
            # disturbance consistent injection
            if self.use_offset_free:
                X_nom = X_nom + self._disturbance_stack(self.d)
            # affine so linear model matches nominal at U_guess
            U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)
            X_aff = X_nom - Mc @ U_guess_vec

            # keep for outputs/debug
            Mc_last = Mc
            X_aff_last = X_aff
            X_nom_last = X_nom
            p_seq_last = p_seq

            # ---- build Rtil and delta-u penalty ----
            Rtil = np.kron(np.eye(Np), self.R)

            if Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np - 1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            # ---- base stacked open-loop (no control) for band constraints ----
            X0_stack = (Mx @ xk).reshape(Np*n, 1)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)

            H = 2.0 * (Rtil + H_du)
            f = np.zeros((Np*m, 1))
            # ---- advancement reward (linear term) ----
            if t_vessel is not None and self.w_adv != 0.0:
                t_v = np.asarray(t_vessel, float).copy()

                # sign alignment (no xref)
                if np.dot(t_v[0], xk[3:6,0]) < 0.0:
                    t_v[0] *= -1.0
                for k in range(1, Np):
                    if np.dot(t_v[k], t_v[k-1]) < 0.0:
                        t_v[k] *= -1.0

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
                g_unsafe0 = g_wall0 * g_theta0

                beta = 0.9
                w_adv_eff = self.w_adv * (1.0 - beta*g_unsafe0)
                s_des_eff = max(self.s_des * (1.0 - beta*g_unsafe0), 0.1*self.s_des)

                # store for later use in constraints
                s_des_eff_local = s_des_eff

                # apply progress reward
                f = f - w_adv_eff * g_adv
            else:
                s_des_eff_local = self.s_des

            # ---- overhead magnet quadratic term (must happen BEFORE lifting to z) ----
            if bool(getattr(self, "overhead_magnet", False)) and float(getattr(self, "w_mag_xy", 0.0)) > 0.0:
                w_mag = float(self.w_mag_xy)
                r0 = p0[:3].copy()
                r0_stack = np.tile(r0, self.Np)

                Pm = build_Pm_world(self.dt, self.Np, m=self.m)

                idx_pos = _pos_row_idx(n, Np)
                Mc_pos = Mc[idx_pos, :]
                Xaff_pos = X_aff[idx_pos, :].ravel()

                keep_xy = np.array([3*k + i for k in range(self.Np) for i in (0, 1)], dtype=int)

                Pm_xy = Pm[keep_xy, :]
                Mc_xy = Mc_pos[keep_xy, :]
                r0_xy = r0_stack[keep_xy]
                xaff_xy = Xaff_pos[keep_xy]

                A_xy = (Pm_xy - Mc_xy)
                b_xy = (r0_xy - xaff_xy).reshape(-1, 1)

                H = H + 2.0 * w_mag * (A_xy.T @ A_xy)
                f = f + 2.0 * w_mag * (A_xy.T @ b_xy)

            # -----------------------
            # Build all U-only constraints FIRST
            # -----------------------
            A_list, l_list, u_list = [], [], []

            # (1) input bounds
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np * m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

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
            ns_wall = Np if self.enable_soft_wall else 0
            ns_prog = Np if self.enable_soft_progress else 0
            ns = ns_wall + ns_prog
            Nu = Np*m
            nz = Nu + ns

            H_z = np.zeros((nz, nz), float)
            H_z[:Nu,:Nu] = H

            f_z = np.zeros((nz, 1), float)
            f_z[:Nu,:] = f

            # quadratic slack penalties
            if ns_wall > 0:
                H_z[Nu:Nu+ns_wall, Nu:Nu+ns_wall] = 2.0*self.w_slack_wall*np.eye(ns_wall)
            if ns_prog > 0:
                i0 = Nu+ns_wall
                H_z[i0:i0+ns_prog, i0:i0+ns_prog] = 2.0*self.w_slack_prog*np.eye(ns_prog)

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

        # -----------------------
        # Plant update (one step) WITH 2-mode horizon evaluation
        # -----------------------
        flip_stage0 = False
        flip_status = "skipped"
        J_no = np.inf
        J_fl = np.inf
        P_no = X_no = P_fl = X_fl = None

        enable_flip = bool(getattr(self, "enable_flip_rollout", False))
        s_path_local = getattr(self, "s_path", None)
        C = getattr(self, "lumen_C", None)
        R = getattr(self, "lumen_R", None)
        have_lumen = (C is not None) and (R is not None)

        if enable_flip and (not infeas_final) and have_lumen and (s_path_local is not None):
            try:
                i0 = int(getattr(self, "i_ref_last", 0))

                # Mode 0: no flip
                P_no, X_no, J_no = rollout_open_loop_fixed_mode(
                    mpc=self,
                    p_start=p_prev,
                    U_seq=U_seq,
                    lumen_C=C, lumen_R=R, s_path=s_path_local,
                    i_ref_start=i0,
                    delta_wall=float(getattr(self, "delta_wall", 1e-3)),
                    w_wall=float(getattr(self, "flip_w_wall", 1e6)),
                    w_prog=float(getattr(self, "flip_w_prog", 1.0)),
                )

                # Mode 1: flip at START (this is the important convention)
                p_prev_flip = p_prev.copy()
                p_prev_flip[3:7] = quat_flip_about_body_z(p_prev_flip[3:7])
                p_prev_flip = self._clamp_p(p_prev_flip)

                P_fl, X_fl, J_fl = rollout_open_loop_fixed_mode(
                    mpc=self,
                    p_start=p_prev_flip,
                    U_seq=U_seq,
                    lumen_C=C, lumen_R=R, s_path=s_path_local,
                    i_ref_start=i0,
                    delta_wall=float(getattr(self, "delta_wall", 1e-3)),
                    w_wall=float(getattr(self, "flip_w_wall", 1e6)),
                    w_prog=float(getattr(self, "flip_w_prog", 1.0)),
                )

                flip_stage0 = bool(J_fl < J_no)
                flip_status = "ok"
            except Exception as e:
                flip_status = f"error:{type(e).__name__}"

        # EXECUTION MUST MATCH ROLLOUT CONVENTION:
        # flip BEFORE integrate
        p_exec0 = p_prev.copy()
        if flip_stage0:
            p_exec0[3:7] = quat_flip_about_body_z(p_exec0[3:7])
        p_exec0 = self._clamp_p(p_exec0)
        # -----------------------
        # Mode-consistent prediction recompute (DEBUG SANITY)
        # -----------------------
        X_pred_mode = None          # (Np,n) predicted horizon consistent with chosen flip mode
        pred1_err_mode = np.nan     # error vs X_pred_mode[0]
        pred1_err_raw  = np.nan     # keep old one too (computed later)

        try:
            if (not infeas_final) and (U_opt_vec is not None):
                # Ensure deterministic forward is frozen for this recompute
                if hasattr(self.forward_tip_fn, "start_step"):
                    self.forward_tip_fn.start_step()

                # 1) Mode-consistent "current output" is y(p_exec0), not y(p_prev)
                try:
                    xk_mode = np.asarray(self.forward_tip_fn(p_exec0, commit=False), float).reshape(n, 1)
                except TypeError:
                    xk_mode = np.asarray(self.forward_tip_fn(p_exec0), float).reshape(n, 1)

                if self.use_offset_free:
                    xk_mode = xk_mode + self.d.reshape(n, 1)

                # 2) Build prediction matrices at the executed mode start pose
                #    (Use U_seq as the "guess" so affine matches this plan)
                p_seq_mode, Mx_mode, Mc_mode, _B0_mode = self._build_prediction_mats(p_exec0, U_seq)

                # 3) Nominal nonlinear outputs along that parameter rollout (commit=False)
                Y_nom_mode = np.vstack([self.forward_tip_fn(p_seq_mode[i]) for i in range(Np)]).reshape(Np, n)
                X_nom_mode = Y_nom_mode.reshape(Np * n, 1)
                if self.use_offset_free:
                    X_nom_mode = X_nom_mode + self._disturbance_stack(self.d)

                # 4) Affine term consistent with THIS planned control
                U_vec = U_opt_vec.reshape(-1, 1)  # same as U_seq.reshape(-1,1)
                X_aff_mode = X_nom_mode - Mc_mode @ U_vec

                # 5) Final mode-consistent predicted horizon
                X_pred_mode_stack = X_aff_mode + Mc_mode @ U_vec
                X_pred_mode = X_pred_mode_stack.reshape(Np, n)

                # Store for debugging
                # (Optional) also store the mode-consistent baseline open-loop stack:
                X_base_mode = (Mx_mode @ xk_mode).reshape(Np, n)
                if self.use_offset_free:
                    X_base_mode = X_base_mode + self._disturbance_stack(self.d).reshape(Np, n)

        except Exception as e:
            # Don't break the step if debug recompute fails
            X_pred_mode = None
            pred1_err_mode = np.nan
        p_next_true = self._clamp_p(integrate_pose8_body(p_exec0, u0, self.dt))

        # evaluate plant
        try:
            x_next_true = np.asarray(self.forward_tip_fn(p_next_true, commit=False), float).reshape(n,)
        except TypeError:
            x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(n,)

        if self.use_offset_free:
            x_next_true = x_next_true + self.d

        self.p = p_next_true.copy()
        self.x = x_next_true.copy()

        # warm start
        self.U_warm = None if infeas_final else U_opt_vec.copy()

        # one-step prediction error (pos only)
        # -----------------------
        # Prediction errors: raw vs mode-consistent
        # -----------------------
        pred1_err_raw = np.nan
        pred1_err_mode = np.nan

        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            pred1_err_raw = np.linalg.norm(x_next_true[:3] - X_pred[0][:3])

        if (X_pred_mode is not None) and (X_pred_mode.shape[0] > 0) and np.all(np.isfinite(X_pred_mode[0])):
            pred1_err_mode = np.linalg.norm(x_next_true[:3] - X_pred_mode[0][:3])

        # Choose which one you want to report as pred1_err (I recommend mode-consistent)
        pred1_err = pred1_err_mode if np.isfinite(pred1_err_mode) else pred1_err_raw
        if self.debug and (Mc_last is not None):
            xbase0 = X0_stack[0:n].ravel()
            xaff0  = X_aff_last[0:n].ravel()
            print("xbase0", xbase0[:3], "xaff0", xaff0[:3])

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
            pred1_err=float(pred1_err) if np.isfinite(pred1_err) else np.nan,
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
        info["X_base"] = (Mx @ xk).reshape(Np, n).copy()


        if self.debug:
            pred1 = float(pred1_err) if np.isfinite(pred1_err) else np.nan
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
            info["theta_seq_deg"] = theta_seq.copy()
            info["theta0_deg"] = info_theta0
            info["theta_max_deg"] = info_thetamax
            info["flip_status"] = flip_status
            info["flip_stage0"] = int(flip_stage0)
            info["flip_J_no"] = float(J_no) if np.isfinite(J_no) else np.nan
            info["flip_J_fl"] = float(J_fl) if np.isfinite(J_fl) else np.nan
            info["flip_rollout_no"] = None if X_no is None else X_no.copy()
            info["flip_rollout_fl"] = None if X_fl is None else X_fl.copy()
            pred1_err_raw=float(pred1_err_raw) if np.isfinite(pred1_err_raw) else np.nan,
            pred1_err_mode=float(pred1_err_mode) if np.isfinite(pred1_err_mode) else np.nan,
            X_pred_mode=None if X_pred_mode is None else X_pred_mode.copy(),
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
    delta_L = 1e-4  # 0.1 mm
    p_plus = p8.copy();  p_plus[7]  += delta_L
    p_minus = p8.copy(); p_minus[7] -= delta_L
    y_plus  = np.asarray(forward_y_fn(p_plus), float).reshape(n_out,)
    y_minus = np.asarray(forward_y_fn(p_minus), float).reshape(n_out,)
    dy_dL = (y_plus - y_minus) / (2.0 * delta_L)

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
def debug_step_pose7_flip_compare(
    k,
    p_now,
    x_now,
    info,
    mpc,
    i_ref=None,
    print_horizon=4,
    do_mode_rollouts=True,
):
    """
    Debug print that compares the TWO fixed modes for THIS MPC step:
      - Mode NO-FLIP  : start pose = p_prev
      - Mode FLIP     : start pose = flip_about_body_z(p_prev)
    and shows:
      - the two NL rollouts (positions + tangents)
      - the total rollout cost for each
      - which mode was chosen for stage-0
      - whether the executed next pose matches the chosen convention (flip-before-integrate)

    Requirements / assumptions:
      - mpc has: lumen_C, lumen_R, s_path, delta_wall
      - info contains at least: p_prev, U_seq, flip_stage0 (optional), flip_J_no/flip_J_fl (optional)
      - You have:
          quat_flip_about_body_z(qwxyz) OR quat_flip_180_about_body_axis(q, axis=[0,0,1])
          rollout_open_loop_fixed_mode(...)  -> (P_nl, X_nl, J_total)
        If you don't, see notes at bottom for drop-in fallbacks.
    """

    def _fmt3(v, prec=4):
        v = np.asarray(v, float).ravel()
        return f"[{v[0]:+.{prec}f},{v[1]:+.{prec}f},{v[2]:+.{prec}f}]"

    def _fmt_tan(t, prec=3):
        t = np.asarray(t, float).ravel()
        return f"[{t[0]:+.{prec}f},{t[1]:+.{prec}f},{t[2]:+.{prec}f}]"

    # -----------------------
    # Basics / header
    # -----------------------
    Np = int(getattr(mpc, "Np", 0))
    ph = min(int(print_horizon), Np) if Np > 0 else int(print_horizon)

    x_now6 = info.get("x_now", None)
    x_now6 = np.asarray(x_now6 if x_now6 is not None else x_now, float).ravel()

    pos_now = x_now6[:3]
    tan_now = x_now6[3:6] if x_now6.size >= 6 else np.full(3, np.nan)

    status = info.get("status", "?")
    infeas = int(info.get("infeasible", -1))
    pred_err = float(info.get("pred1_err", np.nan))

    print(
        f"\n[DBG-FLIP] k={k:04d} status={status} infeas={infeas} pred1_err={pred_err:.3e} "
        f"tip={_fmt3(pos_now)} tan={_fmt_tan(tan_now)}"
    )
    if i_ref is not None:
        print(f"   i_ref={int(i_ref)}")

    # Pose info (current)
    p7 = pose8_quat_to_pose7_rotvec(np.asarray(p_now, float))
    rvec = p7[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))
    print(
        f"   p_now: xyz={_fmt3(p7[0:3],3)} rotvec={_fmt3(rvec,3)} |theta|={theta_deg:.1f}deg L={p7[6]:.4f}"
    )

    # Control / planned horizon
    U_seq = info.get("U_seq", None)
    if U_seq is None:
        print("   [DBG-FLIP] No U_seq in info -> cannot do rollouts.")
        print("--------------------------------------------------------------------")
        return
    U_seq = np.asarray(U_seq, float)
    if U_seq.ndim != 2 or U_seq.shape[0] != Np:
        print(f"   [DBG-FLIP] U_seq shape unexpected: {U_seq.shape}, expected (Np,m)=({Np},{getattr(mpc,'m',7)})")
        print("--------------------------------------------------------------------")
        return

    u0 = U_seq[0].copy()
    print(
        "   u0:",
        f"v={_fmt3(u0[0:3],4)}  "
        f"omega_body={_fmt3(u0[3:6],4)}  dL={u0[6]:+.5f}"
    )

    p_prev = info.get("p_prev", None)
    if p_prev is None:
        print("   [DBG-FLIP] No p_prev in info -> cannot do mode comparison.")
        print("--------------------------------------------------------------------")
        return
    p_prev = np.asarray(p_prev, float).copy()

    # -----------------------
    # Rollouts: NO-FLIP vs FLIP (fixed mode over horizon)
    # -----------------------
    have_geom = (getattr(mpc, "lumen_C", None) is not None) and (getattr(mpc, "lumen_R", None) is not None)
    have_s = getattr(mpc, "s_path", None) is not None
    enable = bool(getattr(mpc, "enable_flip_rollout", True))

    if not (do_mode_rollouts and enable and have_geom and have_s):
        print("   [DBG-FLIP] Mode rollouts skipped (missing geometry/s_path or disabled).")
        print("--------------------------------------------------------------------")
        return

    C = np.asarray(mpc.lumen_C, float)
    R = np.asarray(mpc.lumen_R, float).ravel()
    s_path = np.asarray(mpc.s_path, float).ravel()

    i0 = int(getattr(mpc, "i_ref_last", i_ref if i_ref is not None else 0))

    # Flip helper: prefer quat_flip_about_body_z if present, else fall back to generic
    def _flip_q(q):
        if "quat_flip_about_body_z" in globals():
            return quat_flip_about_body_z(q)
        return quat_flip_180_about_body_axis(q, np.array([0.0, 0.0, 1.0]))

    # Build start poses (important: flip BEFORE any integration)
    p0_no = p_prev.copy()
    p0_fl = p_prev.copy()
    p0_fl[3:7] = _flip_q(p0_fl[3:7])
    p0_no = mpc._clamp_p(p0_no)
    p0_fl = mpc._clamp_p(p0_fl)

    # Run rollouts
    try:
        P_no, X_no, J_no = rollout_open_loop_fixed_mode(
            mpc=mpc,
            p_start=p0_no,
            U_seq=U_seq,
            lumen_C=C, lumen_R=R, s_path=s_path,
            i_ref_start=i0,
            delta_wall=float(getattr(mpc, "delta_wall", 1e-3)),
            w_wall=float(getattr(mpc, "flip_w_wall", 1e6)),
            w_prog=float(getattr(mpc, "flip_w_prog", 1.0)),
        )
        P_fl, X_fl, J_fl = rollout_open_loop_fixed_mode(
            mpc=mpc,
            p_start=p0_fl,
            U_seq=U_seq,
            lumen_C=C, lumen_R=R, s_path=s_path,
            i_ref_start=i0,
            delta_wall=float(getattr(mpc, "delta_wall", 1e-3)),
            w_wall=float(getattr(mpc, "flip_w_wall", 1e6)),
            w_prog=float(getattr(mpc, "flip_w_prog", 1.0)),
        )
    except Exception as e:
        print(f"   [DBG-FLIP] rollout_open_loop_fixed_mode failed: {type(e).__name__}: {e}")
        print("--------------------------------------------------------------------")
        return

    X_no = np.asarray(X_no, float)
    X_fl = np.asarray(X_fl, float)

    # decision: prefer info’s flip_stage0 if present, else infer from costs
    flip_stage0_info = info.get("flip_stage0", None)
    flip_stage0 = bool(flip_stage0_info) if flip_stage0_info is not None else bool(J_fl < J_no)

    # costs summary
    print(f"   Mode costs:  J_no_flip={J_no:.6e}   J_flip={J_fl:.6e}   -> chosen={'FLIP' if flip_stage0 else 'NO-FLIP'}")
    if "flip_status" in info:
        print(f"   flip_status(info)={info['flip_status']}  flip_stage0(info)={info.get('flip_stage0', None)}")

    # -----------------------
    # Print the two rollouts side-by-side (first ph stages)
    # -----------------------
    ph2 = min(ph, X_no.shape[0], X_fl.shape[0])

    print("   Horizon NL rollouts (fixed-mode)   [k:  NO-FLIP  |  FLIP ]")
    for j in range(ph2):
        p_no = X_no[j, 0:3]
        t_no = X_no[j, 3:6] if X_no.shape[1] >= 6 else np.full(3, np.nan)
        p_fl = X_fl[j, 0:3]
        t_fl = X_fl[j, 3:6] if X_fl.shape[1] >= 6 else np.full(3, np.nan)

        tag = "<-- chosen" if ((j == 0) and flip_stage0) else ("<-- chosen" if ((j == 0) and (not flip_stage0)) else "")
        # show tag only on stage-0 line:
        if j == 0:
            tag = "<-- chosen FLIP" if flip_stage0 else "<-- chosen NO-FLIP"
        else:
            tag = ""

        print(
            f"   {j:02d}: "
            f"pos={_fmt3(p_no)} tan={_fmt_tan(t_no)}  |  "
            f"pos={_fmt3(p_fl)} tan={_fmt_tan(t_fl)}  {tag}"
        )

    # -----------------------
    # Verify executed pose is consistent with “flip-before-integrate”
    # -----------------------
    # Expected executed next pose depending on chosen mode:
    p_exec0 = p_prev.copy()
    if flip_stage0:
        p_exec0[3:7] = _flip_q(p_exec0[3:7])
    p_exec0 = mpc._clamp_p(p_exec0)
    p_next_expected = mpc._clamp_p(integrate_pose8_body(p_exec0, U_seq[0], mpc.dt))

    p_now_internal = np.asarray(info.get("p_now", p_now), float).copy()
    # Note: p_now in info is after the step; compare to p_next_expected
    dx = np.linalg.norm(p_now_internal[0:3] - p_next_expected[0:3])
    dq = np.linalg.norm(quat_wxyz_normalize(p_now_internal[3:7]) - quat_wxyz_normalize(p_next_expected[3:7]))

    print(f"   Exec check vs expected (flip-before-integrate): |Δpos|={1e3*dx:.3f}mm  |Δquat|={dq:.3e}")

    # Also compare executed x_now vs the chosen rollout stage-0 output (if dimensions match)
    x_post = np.asarray(info.get("x_now", x_now), float).ravel()
    if x_post.size >= 3 and ph2 > 0:
        x0_chosen = (X_fl[0] if flip_stage0 else X_no[0])
        dx_y = np.linalg.norm(x_post[:3] - x0_chosen[:3])
        print(f"   Output check vs chosen rollout stage0: |Δtip|={1e3*dx_y:.3f}mm")

    print("--------------------------------------------------------------------")
    """C: (M,3) -> s: (M,) cumulative arc-length."""
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s
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
    centerline_tip,
    lumen_C, lumen_R, p0_ur,
    tip_pos,
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
        tip=tip_pos,
        tip_from_centerline=tip_from_centerline,
        p_mag=p_now,                 # NEW
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

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.06 + s_straight,     
        n_pts=130,                      
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(-30.0),
        bend_start=0.01 + s_straight,    
        bend_end=0.06 + s_straight       
    )
    # lumen_C = make_lumen_centerline_double_turn(
    #     p0_ur, t0,
    #     length=0.08, n_pts=60,
    #     bend_axis=np.array([0., 0., 1.]),
    #     bend1_angle=np.deg2rad(90.0),
    #     bend1_start=0.03, bend1_end=0.05,
    #     bend2_angle=np.deg2rad(-90.0),
    #     bend2_start=0.05, bend2_end=0.06, 
    # )
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

    u_max = np.array([ 1, 1, 1, np.deg2rad(60), np.deg2rad(60), np.deg2rad(90),  0.5])
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

    w_pos_x = 10.0
    w_pos_y = 10.0  
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

    mpc.s_path  = s_path   # <-- REQUIRED for flip rollout
    Np = mpc.Np
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
    out_root = Path("mpc_run_025")
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
    theta0_hist = []
    thetaMax_hist = []
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
        _, _, _, B0 = mpc._build_prediction_mats(p_pre, U_guess=None)


        # D) forcing params
        stalling = (cur_dbg["prog"] <= 1e-4)
        near_centerline = (cur_dbg["d_now"] <= 0.010)
        force_ok = stalling and near_centerline
        stall_cnt = int(cursor_state.get("stall", 0))

        mpc.s_min_progress = (0.2e-3 if (force_ok and stall_cnt >= 10) else 0.0)

        # IMPORTANT: actually affects step()
        mpc.w_adv = (mpc.w_adv_base * (1.0 + 2.0 * max(0, stall_cnt - 3))) if force_ok else mpc.w_adv_base

        mpc.i_ref_last = int(i_ref)

        # E) step once
        p_post, y_post, info = mpc.step(x_meas=None)
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
            centerline_tip=C_pre,  # <-- was C_prev
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            p0_ur=p0_ur,
            tip_pos=tip_pre,       # <-- was tip_prev
            tip_from_centerline=(
                C_pre[:, -1] if (C_pre is not None and getattr(C_pre, "shape", None) is not None and C_pre.shape[0] == 3)
                else (C_pre[-1] if C_pre is not None else None)
            ),
        )

        debug_step_pose7_flip_compare(
            k=k,
            p_now=p_post,
            x_now=y_post,
            info=info,
            mpc=mpc,
            i_ref=i_ref,
            print_horizon=mpc.Np,
            do_mode_rollouts=True,
        )
        pred_targets = predicted_targets_from_info(info, Np=mpc.Np)  # (Np,3) = x1..xNp
        if pred_targets is None:
            pred_targets = np.empty((0,3))

        # prepend the actual next tip so the first “target” is the achieved x1
        pred_plus_actual = np.vstack([tip_post.reshape(1,3), pred_targets]) if pred_targets.size else tip_post.reshape(1,3)
        if (i_ref >= M - 2):
            dbg_print(1, "[DONE] reached final point")
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=pred_plus_actual,
                tip=tip_pre,
                tip_from_centerline=C_post,
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
            break
        if (k % 2) == 0:
            theta0 = float(info.get("theta0_deg", np.nan))
            g0 = float(np.asarray(info.get("g_seq", [np.nan]))[0]) if "g_seq" in info else np.nan
            title = f"PRE k={k} i_ref={i_ref}  theta0={theta0:.1f}deg"
            # --- PRE-step visualization: use the "pre" snapshot variables ---
            plot_energy_only_3d(
                C_pre,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=pred_plus_actual,
                tip=tip_pre,
                tip_from_centerline=(C_pre[:, -1] if C_pre is not None and C_pre.shape[0]==3 else (C_pre[-1] if C_pre is not None else None)),
                p_mag=p_post,
                title=title,
                show=True,
            )
            pred_targets = predicted_targets_from_info(info, Np=mpc.Np)
            print("tip_pre", tip_pre)
            print("pred[0]", pred_targets[0])
            print("tip_post", tip_post)
            print("||pred[0]-tip_post|| (mm)", 1e3*np.linalg.norm(pred_targets[0]-tip_post))
            plt.figure()
            plt.plot(k_hist, theta0_hist, label="theta0 (deg)")
            plt.plot(k_hist, thetaMax_hist, label="theta_max over horizon (deg)")
            plt.axhline(40.0, linestyle="--", label="theta_crit=40deg")
            plt.xlabel("k")
            plt.ylabel("theta (deg)")
            plt.title("Tip–vessel tangent misalignment")
            plt.grid(True)
            plt.legend()
            plt.show()
            # # --- histories (safe if lists exist, even if short) ---
            # K = np.asarray(k_hist)
            # pred1 = np.asarray(pred1_hist)        # meters
            # Smat = np.asarray(svd_S_hist)         # (T,6)
            # cond = np.asarray(svd_cond_hist)

            # # 1) pred1_err vs k
            # plt.figure()
            # plt.plot(K, 1e3 * pred1)  # mm
            # plt.xlabel("k")
            # plt.ylabel("pred1_err (mm)")
            # plt.title("One-step prediction error vs step")
            # plt.grid(True)
            # plt.show()

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
