import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from beam_direction_magnetisation.quarternions.quarternions_functions import (quat_normalize, 
                                                                            quat_to_R,
                                                                              T_to_p_quat_wxyz)
from beam_direction_magnetisation.cosserat_w_minimal_energy import contact_barrier_energy_and_force_fast, LumenQuery
from proper_research.robot.transformations import get_point
from scipy.optimize import root

from scipy.optimize import root
def rod_kinematics_from_state(p, theta_twist, eps=1e-12):
    e, t, ell = edge_tangents_and_lengths(p, eps=eps)

    d1_ref, d2_ref, d3_ref, _ = parallel_transport_reference_frames(p)
    m1, m2, m3 = material_frames_from_twist(d1_ref, d2_ref, d3_ref, theta_twist)

    kb = curvature_binormals_from_tangents(t, ell, eps=eps)
    dl_voronoi = voronoi_lengths_from_edge_lengths(ell)

    m1_v = m1[:, :-1] + m1[:, 1:]
    m2_v = m2[:, :-1] + m2[:, 1:]

    m1_v /= (np.linalg.norm(m1_v, axis=0, keepdims=True) + eps)
    m2_v /= (np.linalg.norm(m2_v, axis=0, keepdims=True) + eps)

    kappa1 = np.sum(kb * m2_v, axis=0) / (dl_voronoi + eps)
    kappa2 = -np.sum(kb * m1_v, axis=0) / (dl_voronoi + eps)

    mid = segment_midpoints(p)

    return {
        "e": e,
        "t": t,
        "ell": ell,
        "mid": mid,
        "d1_ref": d1_ref,
        "d2_ref": d2_ref,
        "d3_ref": d3_ref,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "kb": kb,
        "dl_voronoi": dl_voronoi,
        "kappa1": kappa1,
        "kappa2": kappa2,
    }
def residual_static_force_balance(
    q_flat, *,
    p0, p1, N,
    EI1_v, EI2_v, GJ_v,
    ell_ref, EA_seg,
    wire_len, r_src, m_src,
    lumen_query=None, use_lumen=False,
    M_ref_local=None,
    ref_twist=None,
):
    f_min, _ = total_force_from_state_fd(
        info_min["q_opt"],
        p0=p0_ur, p1=p_straight[:, 1].copy(), N=N,
        EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
        ell_ref=ell_ref, EA_seg=EA_seg,
        wire_len=wire_len,
        r_src=r_src_ur, m_src=m_src,
        lumen_query=lumen_query, use_lumen=USE_LUMEN,
        M_ref_local=M_ref_local,
        ref_twist=None,
        eps=1e-6,
    )
    print("Residual norm at minimizer:", np.linalg.norm(f_min))
    print("Max residual at minimizer:", np.max(np.abs(f_min)))

    # static equilibrium: total generalized force = 0
    return f_min

from scipy.optimize import root

def solve_nodes_twist_residual(
    *, p0, q0, L, N,
    wire_len, Kinv_fun,
    r_src, m_src, mu_tip,
    EA_wire, EA_tip,
    lumen_C=None, lumen_R=None,
    use_lumen=False,
    maxiter=100,
    M_ref_local=None,
    ref_twist=None,
    q_init=None,
):
    # fixed second node from base tangent / straight reference
    p_straight = make_initial_nodes_straight(p0, q0, L, N)
    p1 = p_straight[:, 1].copy()

    # initial guess
    if q_init is None:
        p_init = make_initial_nodes_perturbed_feasible(p0, q0, L, N, amp=2e-4)
        theta0 = np.zeros(N - 1, float)
        q0_flat = pack_q_from_nodes_twist(p_init, theta0)
    else:
        q0_flat = np.asarray(q_init, float).copy()

    s_nodes = np.linspace(0.0, L, N)
    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(s_nodes, Kinv_fun, wire_len)

    ell_ref = np.full(N - 1, L / (N - 1), float)
    EA_seg = build_EA_edge_profile_from_params(
        s_nodes,
        wire_len=wire_len,
        EA_wire=EA_wire,
        EA_tip=EA_tip,
    )

    lumen_query = None
    if (lumen_C is not None) and (lumen_R is not None):
        lumen_query = LumenQuery(lumen_C, lumen_R)

    def Rfun(q_flat):
        return residual_static_force_balance(
            q_flat,
            p0=p0, p1=p1, N=N,
            EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
            ell_ref=ell_ref, EA_seg=EA_seg,
            wire_len=wire_len,
            r_src=r_src, m_src=m_src,
            lumen_query=lumen_query,
            use_lumen=use_lumen,
            M_ref_local=M_ref_local,
            ref_twist=ref_twist,
        )

    sol = root(
        Rfun,
        q0_flat,
        method="hybr",
        options={"maxfev": maxiter * q0_flat.size},
    )

    p_opt, theta_opt = unpack_q_to_nodes_twist(sol.x, p0, p1, N)

    W, dbg = total_energy_from_state(
        sol.x,
        p0=p0, p1=p1, N=N,
        EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
        ell_ref=ell_ref, EA_seg=EA_seg,
        wire_len=wire_len, r_src=r_src, m_src=m_src,
        lumen_query=lumen_query, use_lumen=use_lumen,
        M_ref_local=M_ref_local,
        ref_twist=ref_twist,
    )

    info = dict(
        success=bool(sol.success),
        message=str(sol.message),
        nit=int(getattr(sol, "nfev", -1)),
        W=float(W),
        dbg=dbg,
        q_opt=np.asarray(sol.x, float).copy(),
        theta_twist_opt=theta_opt.copy(),
        EI1_v=EI1_v.copy(),
        EI2_v=EI2_v.copy(),
        GJ_v=GJ_v.copy(),
        EA_seg=EA_seg.copy(),
        residual_norm=float(np.linalg.norm(Rfun(sol.x))),
        sol=sol,
    )
    return p_opt, theta_opt, info
def stretch_strain_from_lengths(ell, ell_ref, eps=1e-12):
    ell = np.asarray(ell, float)
    ell_ref = np.asarray(ell_ref, float)
    return ell / (ell_ref + eps) - 1.0
def voronoi_lengths_from_edge_lengths(ell):
    """
    ell: (N-1,) edge lengths
    Returns interior Voronoi lengths Delta_l_i, shape (N-2,)
    """
    ell = np.asarray(ell, float)
    return 0.5 * (ell[:-1] + ell[1:])

def build_EA_edge_profile_from_params(s_nodes, wire_len, EA_wire, EA_tip):
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    EA_seg = np.where(s_mid < wire_len, EA_wire, EA_tip).astype(float)
    return EA_seg

def build_bending_twist_profiles(s_nodes, Kinv_fun, wire_len):
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    Kinv_mid = Kinv_fun(s_mid, wire_len)

    EI1_seg = np.zeros(s_mid.size)
    EI2_seg = np.zeros(s_mid.size)
    GJ_seg  = np.zeros(s_mid.size)

    for i in range(s_mid.size):
        K = np.linalg.inv(Kinv_mid[:, :, i])
        GJ_seg[i]  = K[0,0]
        EI1_seg[i] = K[1,1]
        EI2_seg[i] = K[2,2]

    EI1_v = 0.5 * (EI1_seg[:-1] + EI1_seg[1:])
    EI2_v = 0.5 * (EI2_seg[:-1] + EI2_seg[1:])
    GJ_v  = 0.5 * (GJ_seg[:-1] + GJ_seg[1:])
    return EI1_v, EI2_v, GJ_v


def pack_q_from_nodes_twist(p, theta_twist):
    """
    p: (3,N) full node array with p[:,0], p[:,1] already fixed externally
    theta_twist: (N-1,) edge twist DOFs
    Returns q_flat = [x(flat), theta]
    where x(flat) contains nodes 2..N-1 only.
    """
    x_flat = p[:, 2:].T.reshape(-1)
    theta_twist = np.asarray(theta_twist, float).reshape(-1)
    return np.concatenate([x_flat, theta_twist])

def unpack_q_to_nodes_twist(q_flat, p0, p1, N):
    """
    q_flat = [x(flat), theta_twist]
    x(flat) corresponds to nodes 2..N-1
    theta_twist has size N-1
    """
    q_flat = np.asarray(q_flat, float).reshape(-1)

    n_x = 3 * (N - 2)
    n_th = N - 1

    x_flat = q_flat[:n_x]
    theta_twist = q_flat[n_x:n_x + n_th]

    p = unpack_x_to_nodes(x_flat, p0, p1, N)
    return p, theta_twist

def twist_strain_from_theta(theta_twist, dl_voronoi, ref_twist=None, eps=1e-12):
    """
    theta_twist: (N-1,) edge twist angles
    dl_voronoi: (N-2,) interior Voronoi lengths
    ref_twist: (N-2,) optional reference twist term, default zero

    Returns kappa3: (N-2,)
    """
    theta_twist = np.asarray(theta_twist, float)
    dl_voronoi = np.asarray(dl_voronoi, float)

    dtheta = theta_twist[1:] - theta_twist[:-1]   # (N-2,)

    if ref_twist is None:
        ref_twist = np.zeros_like(dtheta)
    else:
        ref_twist = np.asarray(ref_twist, float).reshape(dtheta.shape)

    kappa3 = (dtheta + ref_twist) / (dl_voronoi + eps)
    return kappa3
def numerical_force_from_energy(energy_fun, q, eps=1e-6):
    q = np.asarray(q, float).copy()
    f = np.zeros_like(q)

    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        fp = energy_fun(q + dq)
        fm = energy_fun(q - dq)
        f[i] = -(fp - fm) / (2.0 * eps)

    return f

def elastic_force_fd(
    q_flat, *,
    p0, p1, N,
    EI1_v, EI2_v, GJ_v,
    ell_ref, EA_seg,
    ref_twist=None,
    eps=1e-6,
):
    def elastic_energy_only(q):
        p, theta = unpack_q_to_nodes_twist(q, p0, p1, N)
        kin = rod_kinematics_from_state(p, theta)

        Ws, _ = stretch_energy_from_kinematics(kin, ell_ref, EA_seg)
        Wb, _ = bending_energy_from_kinematics(kin, EI1_v, EI2_v)
        Wt, _ = twist_energy_from_kinematics(theta, kin, GJ_v, ref_twist=ref_twist)
        return Ws + Wb + Wt

    return numerical_force_from_energy(elastic_energy_only, q_flat, eps=eps)
def magnetic_force_fd(
    q_flat, *,
    p0, p1, N,
    wire_len, r_src, m_src, M_ref_local,
    eps=1e-6,
):
    def magnetic_energy_only(q):
        p, theta = unpack_q_to_nodes_twist(q, p0, p1, N)
        kin = rod_kinematics_from_state(p, theta)
        Wm, _ = magnetic_energy_from_kinematics(
            kin, wire_len, r_src, m_src, M_ref_local
        )
        return Wm

    return numerical_force_from_energy(magnetic_energy_only, q_flat, eps=eps)
def contact_force_fd(
    q_flat, *,
    p0, p1, N,
    lumen_query,
    eps=1e-6,
):
    def contact_energy_only(q):
        p, _ = unpack_q_to_nodes_twist(q, p0, p1, N)
        Wc, _ = contact_energy_from_nodes(p, lumen_query)
        return Wc

    return numerical_force_from_energy(contact_energy_only, q_flat, eps=eps)
def total_force_from_state_fd(
    q_flat, *,
    p0, p1, N,
    EI1_v, EI2_v, GJ_v,
    ell_ref, EA_seg,
    wire_len, r_src, m_src,
    lumen_query=None, use_lumen=False,
    M_ref_local=None,
    ref_twist=None,
    eps=1e-6,
):
    if M_ref_local is None:
        raise ValueError("M_ref_local must be provided explicitly")
    f_el = elastic_force_fd(
        q_flat,
        p0=p0, p1=p1, N=N,
        EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
        ell_ref=ell_ref, EA_seg=EA_seg,
        ref_twist=ref_twist,
        eps=eps,
    )

    f_mag = magnetic_force_fd(
        q_flat,
        p0=p0, p1=p1, N=N,
        wire_len=wire_len,
        r_src=r_src,
        m_src=m_src,
        M_ref_local=M_ref_local,
        eps=eps,
    )

    f_con = np.zeros_like(q_flat)
    if use_lumen and (lumen_query is not None):
        f_con = contact_force_fd(
            q_flat,
            p0=p0, p1=p1, N=N,
            lumen_query=lumen_query,
            eps=eps,
        )

    f_total = f_el + f_mag + f_con

    return f_total, {
        "f_elastic": f_el,
        "f_magnetic": f_mag,
        "f_contact": f_con,
    }

def total_energy_from_state(
    q_flat, *,
    p0, p1, N,
    EI1_v, EI2_v, GJ_v,
    ell_ref, EA_seg,
    wire_len, r_src, m_src,
    lumen_query=None, use_lumen=False,
    M_ref_local=None,
    ref_twist=None,
):
    p, theta_twist = unpack_q_to_nodes_twist(q_flat, p0, p1, N)
    kin = rod_kinematics_from_state(p, theta_twist)

    Ws, stretch_dbg = stretch_energy_from_kinematics(kin, ell_ref, EA_seg)
    Wb, bend_dbg = bending_energy_from_kinematics(kin, EI1_v, EI2_v)
    Wt, twist_dbg = twist_energy_from_kinematics(theta_twist, kin, GJ_v, ref_twist=ref_twist)

    if M_ref_local is None:
        raise ValueError("M_ref_local must be provided explicitly")

    Wm, mag_dbg = magnetic_energy_from_kinematics(
        kin, wire_len, r_src, m_src, M_ref_local
    )

    Wc = 0.0
    con_dbg = {}
    if use_lumen and (lumen_query is not None):
        Wc, con_dbg = contact_energy_from_nodes(p, lumen_query)

    W = Ws + Wb + Wt + Wm + Wc

    dbg = {
        "Ws": Ws,
        "Wb": Wb,
        "Wt": Wt,
        "Wm": Wm,
        "Wc": Wc,
        "kin": kin,
        "stretch": stretch_dbg,
        "bend": bend_dbg,
        "twist": twist_dbg,
        "mag": mag_dbg,
        "contact": con_dbg,
    }
    return float(W), dbg
def segment_length_residuals_q(q_flat, *, p0, p1, N, ell):
    p, theta_twist = unpack_q_to_nodes_twist(q_flat, p0, p1, N)
    Ls = edge_lengths(p)
    return Ls[1:]**2 - ell**2
def solve_nodes_twist_min(
    *, 
    p0, q0, L, N,
    wire_len, Kinv_fun,
    r_src, m_src, mu_tip,
    EA_wire, EA_tip,
    lumen_C=None, lumen_R=None,
    use_lumen=False,
    maxiter=300,
    M_ref_local=None,
    ref_twist=None,
    enforce_inextensibility=False,
    amp_init=2e-4,
):
    """
    Solve DER-style node + twist energy minimization.

    Parameters
    ----------
    p0, q0 : base position and base orientation
    L, N : total modeled length and number of nodes
    wire_len : length of passive wire region
    Kinv_fun : constitutive inverse stiffness profile callback
    r_src, m_src : permanent magnet position and dipole in world frame
    mu_tip : magnetic magnitude scale for default axial/transverse M_ref_local
    EA_wire, EA_tip : axial stiffness (E*A) for wire and tip regions
    enforce_inextensibility : if True, keep hard edge-length constraints for now.
                              if False, rely on stretch energy only.
    """

    ell0 = L / (N - 1)

    # -------------------------
    # Initial geometry / twist
    # -------------------------
    p_init = make_initial_nodes_perturbed_feasible(p0, q0, L, N, amp=amp_init)
    p1 = p_init[:, 1].copy()

    theta0 = np.zeros(N - 1, float)
    q0_flat = pack_q_from_nodes_twist(p_init, theta0)

    # -------------------------
    # Discrete constitutive profiles
    # -------------------------
    s_nodes = np.linspace(0.0, L, N)

    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(
        s_nodes, Kinv_fun, wire_len
    )

    ell_ref = np.full(N - 1, ell0, float)

    EA_seg = build_EA_edge_profile_from_params(
        s_nodes,
        wire_len=wire_len,
        EA_wire=EA_wire,
        EA_tip=EA_tip,
    )

    # -------------------------
    # Magnetic reference direction
    # -------------------------
    if M_ref_local is None:
        M_ref_local = np.array([0.0, 0.0, mu_tip], float)

    # -------------------------
    # Lumen query
    # -------------------------
    lumen_query = None
    if (lumen_C is not None) and (lumen_R is not None):
        lumen_query = LumenQuery(lumen_C, lumen_R)

    # -------------------------
    # Objective
    # -------------------------
    def obj(q_flat):
        W, _ = total_energy_from_state(
            q_flat,
            p0=p0, p1=p1, N=N,
            EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
            ell_ref=ell_ref, EA_seg=EA_seg,
            wire_len=wire_len,
            r_src=r_src, m_src=m_src,
            lumen_query=lumen_query,
            use_lumen=use_lumen,
            M_ref_local=M_ref_local,
            ref_twist=ref_twist,
        )
        return W

    # -------------------------
    # Optional hard inextensibility constraints
    # -------------------------
    constraints = []
    if enforce_inextensibility:
        cons = NonlinearConstraint(
            lambda q_flat: segment_length_residuals_q(
                q_flat, p0=p0, p1=p1, N=N, ell=ell0
            ),
            lb=np.zeros(N - 2),
            ub=np.zeros(N - 2),
        )
        constraints = [cons]

    # -------------------------
    # Solve
    # -------------------------
    res = minimize(
        obj,
        q0_flat,
        method="SLSQP",
        constraints=constraints,
        options=dict(
            maxiter=maxiter,
            ftol=1e-6,
            eps=1e-7,
            disp=True,
        ),
    )

    # -------------------------
    # Final state / diagnostics
    # -------------------------
    p_opt, theta_opt = unpack_q_to_nodes_twist(res.x, p0, p1, N)

    W, dbg = total_energy_from_state(
        res.x,
        p0=p0, p1=p1, N=N,
        EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
        ell_ref=ell_ref, EA_seg=EA_seg,
        wire_len=wire_len,
        r_src=r_src, m_src=m_src,
        lumen_query=lumen_query,
        use_lumen=use_lumen,
        M_ref_local=M_ref_local,
        ref_twist=ref_twist,
    )

    info = dict(
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit),
        W=float(W),
        dbg=dbg,
        q_opt=np.asarray(res.x, float).copy(),
        theta_twist_opt=theta_opt.copy(),
        EI1_v=EI1_v.copy(),
        EI2_v=EI2_v.copy(),
        GJ_v=GJ_v.copy(),
        EA_seg=EA_seg.copy(),
        ell_ref=ell_ref.copy(),
        enforce_inextensibility=bool(enforce_inextensibility),
        M_ref_local=np.asarray(M_ref_local, float).copy(),
        res=res,
    )
    return p_opt, theta_opt, info

def curvature_binormals_from_tangents(t, ell, eps=1e-12):
    """
    t: (3, N-1) unit edge tangents
    ell: (N-1,) edge lengths

    Returns kb: (3, N-2), one curvature-binormal vector per interior node.
    Uses the standard DER discrete curvature binormal:
        kb_i = 2 * (t_{i-1} x t_i) / (1 + t_{i-1} . t_i)
    This is the common unitless/binormal form; dividing by Voronoi length gives curvature scale.
    """
    t_im1 = t[:, :-1]    # (3, N-2)
    t_i   = t[:, 1:]     # (3, N-2)

    cross = np.cross(t_im1.T, t_i.T).T
    denom = 1.0 + np.sum(t_im1 * t_i, axis=0)
    denom = np.maximum(denom, eps)

    kb = 2.0 * cross / denom[None, :]
    return kb
def bending_energy_from_kinematics(kin, EI1_v, EI2_v):
    kappa1 = kin["kappa1"]
    kappa2 = kin["kappa2"]
    dl = kin["dl_voronoi"]

    Wb = 0.5 * np.sum((EI1_v * kappa1**2 + EI2_v * kappa2**2) * dl)
    return float(Wb), {
        "kappa1": kappa1,
        "kappa2": kappa2,
        "kb": kin["kb"],
        "dl_voronoi": dl,
    }
def twist_energy_from_kinematics(theta_twist, kin, GJ_v, ref_twist=None, eps=1e-12):
    dl_voronoi = kin["dl_voronoi"]
    kappa3 = twist_strain_from_theta(theta_twist, dl_voronoi, ref_twist=ref_twist, eps=eps)
    Wt = 0.5 * np.sum(GJ_v * (kappa3**2) * dl_voronoi)
    return float(Wt), {"kappa3": kappa3}
def stretch_energy_from_kinematics(kin, ell_ref, EA_seg):
    ell = kin["ell"]
    eps_s = stretch_strain_from_lengths(ell, ell_ref)
    Ws = 0.5 * np.sum(EA_seg * (eps_s**2) * ell_ref)
    return float(Ws), {"eps_s": eps_s}
def magnetic_energy_from_kinematics(
    kin, wire_len, r_src, m_src, M_ref_local
):
    m1 = kin["m1"]
    m2 = kin["m2"]
    m3 = kin["m3"]
    ell_i = kin["ell"]
    mid = kin["mid"]

    s_nodes = np.zeros(ell_i.size + 1)
    s_nodes[1:] = np.cumsum(ell_i)
    s_seg = 0.5 * (s_nodes[:-1] + s_nodes[1:])

    B = dipole_field_points(mid, r_src, m_src)

    M_ref_local = np.asarray(M_ref_local, float).reshape(3,)
    mag_mask = (s_seg >= wire_len).astype(float)

    M_world = (
        m1 * M_ref_local[0]
        + m2 * M_ref_local[1]
        + m3 * M_ref_local[2]
    )
    M_world *= mag_mask[None, :]

    wm = -np.sum(M_world * B, axis=0)
    Wm = np.sum(wm * ell_i)

    return float(Wm), {
        "B": B,
        "M_world": M_world,
        "mag_mask": mag_mask,
        "s_seg": s_seg,
    }
def contact_energy_from_nodes(p, lumen_query):
    Wc, dbg = lumen_contact_energy_nodes(p, lumen_query)
    return float(Wc), dbg


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def edge_tangents_and_lengths(p, eps=1e-12):
    e = p[:, 1:] - p[:, :-1]                 # (3, N-1)
    ell = np.linalg.norm(e, axis=0)          # (N-1,)
    t = e / (ell[None, :] + eps)             # (3, N-1)
    return e, t, ell

def choose_initial_reference_frame(t0):
    """
    Build an orthonormal pair (d1,d2) perpendicular to t0.
    """
    t0 = normalize(t0)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    d1 = ref - np.dot(ref, t0) * t0
    d1 = normalize(d1)
    d2 = normalize(np.cross(t0, d1))
    return d1, d2

def rot_about_axis(axis, angle):
    axis = normalize(axis)
    ax = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    I = np.eye(3)
    return I + np.sin(angle)*ax + (1.0 - np.cos(angle))*(ax @ ax)
def minimal_rotation(a, b, eps=1e-12):
    """
    Rotation matrix sending unit vector a to unit vector b by minimal rotation.
    """
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < eps:
        if c > 0:
            return np.eye(3)
        # 180 deg case
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, ref))
        return rot_about_axis(axis, np.pi)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s**2))

def parallel_transport_reference_frames(p):
    """
    Returns reference adapted frames on edges:
      d1_ref, d2_ref, d3_ref each shape (3, N-1)
    """
    e, t, ell = edge_tangents_and_lengths(p)
    n_edge = t.shape[1]

    d1 = np.zeros((3, n_edge))
    d2 = np.zeros((3, n_edge))
    d3 = t.copy()

    d1[:, 0], d2[:, 0] = choose_initial_reference_frame(t[:, 0])

    for i in range(1, n_edge):
        R = minimal_rotation(t[:, i-1], t[:, i])
        d1[:, i] = normalize(R @ d1[:, i-1])
        d2[:, i] = normalize(R @ d2[:, i-1])

        # re-orthogonalize against tangent for robustness
        d1[:, i] = normalize(d1[:, i] - np.dot(d1[:, i], d3[:, i]) * d3[:, i])
        d2[:, i] = normalize(np.cross(d3[:, i], d1[:, i]))

    return d1, d2, d3, ell

def material_frames_from_twist(d1_ref, d2_ref, d3, theta):
    """
    theta: (N-1,) twist angle per edge
    Returns material frames m1,m2,m3 on edges.
    """
    n_edge = d3.shape[1]
    m1 = np.zeros_like(d1_ref)
    m2 = np.zeros_like(d2_ref)
    m3 = d3.copy()

    for i in range(n_edge):
        c = np.cos(theta[i])
        s = np.sin(theta[i])
        m1[:, i] = c * d1_ref[:, i] + s * d2_ref[:, i]
        m2[:, i] = -s * d1_ref[:, i] + c * d2_ref[:, i]

    return m1, m2, m3
def unpack_x_to_nodes(x, p0, p1, N):
    """
    x contains flattened coordinates of nodes 2..N-1
    Returns p shape (3,N)
    """
    p = np.zeros((3, N), float)
    p[:, 0] = p0
    p[:, 1] = p1
    p[:, 2:] = x.reshape(N-2, 3).T
    return p

def edge_vectors(p):
    return p[:, 1:] - p[:, :-1]   # (3, N-1)

def edge_lengths(p):
    e = edge_vectors(p)
    return np.linalg.norm(e, axis=0)

def tangents_from_nodes(p, eps=1e-12):
    e = edge_vectors(p)
    L = np.linalg.norm(e, axis=0) + eps
    return e / L[None, :]   # (3, N-1)

def segment_midpoints(p):
    return 0.5 * (p[:, :-1] + p[:, 1:])   # (3, N-1)

def base_tangent_from_q0(q0, e1=np.array([-1.0, 0.0, 0.0])):
    R0 = quat_to_R(quat_normalize(q0))
    return R0 @ e1

def make_initial_nodes_straight(p0, q0, L, N):
    ell = L / (N - 1)
    t0 = base_tangent_from_q0(q0)
    p = np.zeros((3, N), float)
    for i in range(N):
        p[:, i] = p0 + i * ell * t0
    return p


def dipole_field_points(x, r_src, m_src, mu0_over_4pi=1e-7, r_min=1e-6):
    """
    x: (3,M) field points
    r_src: (3,)
    m_src: (3,)
    Returns B: (3,M)
    """
    x = np.asarray(x, float)
    r_src = np.asarray(r_src, float).reshape(3,1)
    m_src = np.asarray(m_src, float).reshape(3,1)

    r = x - r_src
    rn = np.linalg.norm(r, axis=0)
    rn = np.maximum(rn, r_min)

    rhat = r / rn[None, :]
    mdotr = np.sum(m_src * rhat, axis=0)

    B = mu0_over_4pi * (
        (3.0 * rhat * mdotr[None, :] - m_src) / (rn[None, :]**3)
    )
    return B
def magnetic_density_profile_segments(s_seg, wire_len, mu_tip):
    """
    scalar dipole density magnitude on each segment
    """
    mu = np.zeros_like(s_seg)
    mu[s_seg >= wire_len] = mu_tip
    return mu

def lumen_contact_energy_nodes(p, lumen_query):
    C_nodes, _, d_nodes = contact_barrier_energy_and_force_fast(
        p, lumen_query,
        Kc=5,
        d_tilde=5e-4,
        eps=1e-9,
        penalize_outside=True,
        k_out=1e5,
        pen_switch=5e-4,
        k_hard=3e5,
        window=3,
    )
    Wc = float(np.sum(C_nodes))
    return Wc, dict(C_nodes=C_nodes, d_nodes=d_nodes)

def make_initial_nodes_perturbed_feasible(p0, q0, L, N, amp=2e-4):
    ell = L / (N - 1)
    p = make_initial_nodes_straight(p0, q0, L, N)

    t0 = base_tangent_from_q0(q0)

    ref = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(t0, ref)
    if np.linalg.norm(n1) < 1e-10:
        ref = np.array([0.0, 1.0, 0.0])
        n1 = np.cross(t0, ref)
    n1 = n1 / (np.linalg.norm(n1) + 1e-12)

    xi = np.linspace(0.0, 1.0, N)
    shape = np.sin(np.pi * xi)

    # perturb interior nodes
    for i in range(2, N):
        p[:, i] += amp * shape[i] * n1

    # re-project sequentially to exact segment length
    p[:, 1] = p0 + ell * t0
    for i in range(1, N - 1):
        d = p[:, i+1] - p[:, i]
        d = d / (np.linalg.norm(d) + 1e-12)
        p[:, i+1] = p[:, i] + ell * d

    return p
def segment_length_residuals(x, *, p0, p1, N, ell):
    p = unpack_x_to_nodes(x, p0, p1, N)
    Ls = edge_lengths(p)
    # skip first segment because p0 and p1 are fixed
    return Ls[1:]**2 - ell**2

def set_axes_equal_3d(ax):
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])

    rx = 0.5 * (xlim[1] - xlim[0])
    ry = 0.5 * (ylim[1] - ylim[0])
    rz = 0.5 * (zlim[1] - zlim[0])
    r = max(rx, ry, rz)

    ax.set_xlim3d([xmid - r, xmid + r])
    ax.set_ylim3d([ymid - r, ymid + r])
    ax.set_zlim3d([zmid - r, zmid + r])

def plot_nodes_and_magnet_3d(
    p_nodes,
    *,
    p0=None,
    r_src=None,
    m_src=None,
    q_src=None,
    m_body=None,
    p_straight=None,
    lumen_C=None,
    title="Node optimizer + permanent magnet"
):
    """
    p_nodes : (3,N)
    p0      : (3,)
    r_src   : (3,) permanent magnet position
    m_src   : (3,) permanent magnet dipole in world frame
    q_src   : (4,) quaternion [w,x,y,z] of magnet pose
    m_body  : (3,) dipole in magnet body frame
    p_straight : (3,N) optional straight baseline
    lumen_C : (M,3) optional lumen centerline
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # rod / nodes
    ax.plot(
        p_nodes[0, :], p_nodes[1, :], p_nodes[2, :],
        "-o", lw=2, ms=4, label="optimized rod"
    )

    # optional straight baseline
    if p_straight is not None:
        ax.plot(
            p_straight[0, :], p_straight[1, :], p_straight[2, :],
            "--", lw=1.5, label="straight baseline"
        )

    # optional lumen centerline
    if lumen_C is not None:
        lumen_C = np.asarray(lumen_C, float)
        ax.plot(
            lumen_C[:, 0], lumen_C[:, 1], lumen_C[:, 2],
            "k:", lw=1.5, label="lumen centerline"
        )

    # base point
    if p0 is not None:
        p0 = np.asarray(p0, float).reshape(3,)
        ax.scatter(
            [p0[0]], [p0[1]], [p0[2]],
            s=80, marker="o", label="base"
        )

    # magnet position
    if r_src is not None:
        r_src = np.asarray(r_src, float).reshape(3,)
        ax.scatter(
            [r_src[0]], [r_src[1]], [r_src[2]],
            s=100, marker="s", label="perm. magnet position"
        )

    # choose dipole direction
    m_plot = None
    if m_src is not None:
        m_plot = np.asarray(m_src, float).reshape(3,)
    elif (q_src is not None) and (m_body is not None):
        q = np.asarray(q_src, float).reshape(4,)
        Rm = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        m_plot = Rm @ np.asarray(m_body, float).reshape(3,)

    # magnet dipole arrow
    if (r_src is not None) and (m_plot is not None):
        mnorm = np.linalg.norm(m_plot)
        if mnorm > 1e-12:
            mhat = m_plot / mnorm

            # choose a visible arrow length based on scene size
            pts = [p_nodes]
            if p_straight is not None:
                pts.append(p_straight)
            if lumen_C is not None:
                pts.append(lumen_C.T if lumen_C.shape[0] == 3 else lumen_C.T)

            all_pts = np.hstack([arr if arr.shape[0] == 3 else arr.T for arr in pts])
            scene_min = np.min(all_pts, axis=1)
            scene_max = np.max(all_pts, axis=1)
            scene_size = np.linalg.norm(scene_max - scene_min)

            arrow_len = max(0.01, 0.15 * scene_size)

            ax.quiver(
                r_src[0], r_src[1], r_src[2],
                mhat[0], mhat[1], mhat[2],
                length=arrow_len,
                normalize=False,
                linewidth=2.5,
                label="perm. magnet dipole"
            )

            # optional reverse half-arrow to make dipole line easier to see
            ax.plot(
                [r_src[0] - 0.5 * arrow_len * mhat[0], r_src[0] + arrow_len * mhat[0]],
                [r_src[1] - 0.5 * arrow_len * mhat[1], r_src[1] + arrow_len * mhat[1]],
                [r_src[2] - 0.5 * arrow_len * mhat[2], r_src[2] + arrow_len * mhat[2]],
                lw=1.5
            )

    # tip point
    ax.scatter(
        [p_nodes[0, -1]], [p_nodes[1, -1]], [p_nodes[2, -1]],
        s=70, marker="^", label="tip"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()
def plot_reference_and_material_frames_3d(
    p, d1_ref, d2_ref, d3_ref, m1, m2, m3,
    scale=0.002, stride=2,
    title="Reference and material frames"
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p[0, :], p[1, :], p[2, :], "-o", lw=2, ms=4, label="rod")

    mid = 0.5 * (p[:, :-1] + p[:, 1:])   # edge midpoints

    for i in range(0, mid.shape[1], stride):
        x = mid[:, i]

        # reference frame
        ax.quiver(x[0], x[1], x[2], d1_ref[0, i], d1_ref[1, i], d1_ref[2, i],
                  length=scale, normalize=True)
        ax.quiver(x[0], x[1], x[2], d2_ref[0, i], d2_ref[1, i], d2_ref[2, i],
                  length=scale, normalize=True)
        ax.quiver(x[0], x[1], x[2], d3_ref[0, i], d3_ref[1, i], d3_ref[2, i],
                  length=scale, normalize=True)

        # material frame as dashed line segments for distinction
        ax.plot(
            [x[0], x[0] + scale * m1[0, i]],
            [x[1], x[1] + scale * m1[1, i]],
            [x[2], x[2] + scale * m1[2, i]],
            "--", lw=1.5
        )
        ax.plot(
            [x[0], x[0] + scale * m2[0, i]],
            [x[1], x[1] + scale * m2[1, i]],
            [x[2], x[2] + scale * m2[2, i]],
            "--", lw=1.5
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    set_axes_equal_3d(ax)
    plt.tight_layout()
    plt.show()
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
def summarize_solution(
    name,
    p,
    theta,
    info,
    *,
    p0,
    q0,
    L,
    N,
    wire_len,
    Kinv_fun,
    r_src,
    m_src,
    EA_wire,
    EA_tip,
    M_ref_local,
):
    p_straight = make_initial_nodes_straight(p0, q0, L, N)
    tip = p[:, -1]
    tip_straight = p_straight[:, -1]
    tip_deflection = tip - tip_straight

    kin = rod_kinematics_from_state(p, theta)
    ell = kin["ell"]
    eps_s = stretch_strain_from_lengths(ell, np.full(N - 1, L / (N - 1)))

    s_nodes = np.linspace(0.0, L, N)
    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(s_nodes, Kinv_fun, wire_len)
    s_vertex = s_nodes[1:-1]

    kappa1 = kin["kappa1"]
    kappa2 = kin["kappa2"]
    kappa3 = twist_strain_from_theta(theta, kin["dl_voronoi"])

    mag_dbg = info["dbg"]["mag"]
    M_world = mag_dbg["M_world"]
    B = mag_dbg["B"]

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"success            : {info['success']}")
    print(f"message            : {info['message']}")
    print(f"iterations / evals : {info['nit']}")
    print(f"W total            : {info['W']:.6e}")
    print(f"Ws                 : {info['dbg']['Ws']:.6e}")
    print(f"Wb                 : {info['dbg']['Wb']:.6e}")
    print(f"Wt                 : {info['dbg']['Wt']:.6e}")
    print(f"Wm                 : {info['dbg']['Wm']:.6e}")
    print(f"Wc                 : {info['dbg']['Wc']:.6e}")
    if "residual_norm" in info:
        print(f"residual norm      : {info['residual_norm']:.6e}")

    print("\n--- tip / geometry ---")
    print(f"tip position       : {tip}")
    print(f"straight tip       : {tip_straight}")
    print(f"tip deflection vec : {tip_deflection}")
    print(f"tip deflection mag : {np.linalg.norm(tip_deflection):.6e}")

    print("\n--- stretch ---")
    print(f"max |eps_s|        : {np.max(np.abs(eps_s)):.6e}")
    print(f"mean|eps_s|        : {np.mean(np.abs(eps_s)):.6e}")
    print(f"min edge length    : {np.min(ell):.6e}")
    print(f"max edge length    : {np.max(ell):.6e}")

    print("\n--- curvature / twist ---")
    print(f"max |kappa1|       : {np.max(np.abs(kappa1)):.6e}")
    print(f"max |kappa2|       : {np.max(np.abs(kappa2)):.6e}")
    print(f"max |kappa3|       : {np.max(np.abs(kappa3)):.6e}")
    print(f"mean dl_voronoi    : {np.mean(kin['dl_voronoi']):.6e}")

    print("\n--- magnetic ---")
    print(f"max |M_world|      : {np.max(np.linalg.norm(M_world, axis=0)):.6e}")
    print(f"max |B|            : {np.max(np.linalg.norm(B, axis=0)):.6e}")
    print(f"active mag segs    : {int(np.sum(mag_dbg['mag_mask'] > 0.5))}")

    wire_mask = s_vertex < wire_len
    tip_mask = s_vertex >= wire_len
    print("\n--- curvature localization ---")
    if np.any(wire_mask):
        print(f"wire mean |kappa|  : {np.mean(np.sqrt(kappa1[wire_mask]**2 + kappa2[wire_mask]**2)):.6e}")
    if np.any(tip_mask):
        print(f"tip mean |kappa|   : {np.mean(np.sqrt(kappa1[tip_mask]**2 + kappa2[tip_mask]**2)):.6e}")
def compare_solutions(name_a, p_a, theta_a, info_a, name_b, p_b, theta_b, info_b):
    tip_a = p_a[:, -1]
    tip_b = p_b[:, -1]

    print(f"\n{'-'*80}")
    print(f"COMPARISON: {name_a} vs {name_b}")
    print(f"{'-'*80}")
    print(f"tip diff vec        : {tip_a - tip_b}")
    print(f"tip diff norm       : {np.linalg.norm(tip_a - tip_b):.6e}")
    print(f"theta diff norm     : {np.linalg.norm(theta_a - theta_b):.6e}")
    print(f"energy diff         : {abs(info_a['W'] - info_b['W']):.6e}")
    if ("residual_norm" in info_a) and ("residual_norm" in info_b):
        print(f"residual norms      : {info_a['residual_norm']:.6e} vs {info_b['residual_norm']:.6e}")
def plot_solution_diagnostics(p, theta, info, *, L, N, wire_len, title_prefix="solution"):
    kin = rod_kinematics_from_state(p, theta)
    s_nodes = np.linspace(0.0, L, N)
    s_vertex = s_nodes[1:-1]
    s_edge = 0.5 * (s_nodes[:-1] + s_nodes[1:])

    kappa1 = kin["kappa1"]
    kappa2 = kin["kappa2"]
    kappa3 = twist_strain_from_theta(theta, kin["dl_voronoi"])
    eps_s = stretch_strain_from_lengths(kin["ell"], np.full(N - 1, L / (N - 1)))
    Bmag = np.linalg.norm(info["dbg"]["mag"]["B"], axis=0)
    Mmag = np.linalg.norm(info["dbg"]["mag"]["M_world"], axis=0)

    plt.figure()
    plt.plot(s_vertex, kappa1, "-o", label="kappa1")
    plt.plot(s_vertex, kappa2, "-o", label="kappa2")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("bending strain [1/m]")
    plt.title(f"{title_prefix}: bending strains")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(s_vertex, kappa3, "-o", label="kappa3")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("twist strain [1/m]")
    plt.title(f"{title_prefix}: twist strain")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(s_edge, eps_s, "-o", label="eps_s")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("stretch strain [-]")
    plt.title(f"{title_prefix}: stretch strain")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(s_edge, Bmag, "-o", label="|B|")
    plt.plot(s_edge, Mmag, "-o", label="|M_world|")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("magnitude")
    plt.title(f"{title_prefix}: magnetic quantities")
    plt.legend()
    plt.grid(True)

    plt.show()
def fd_force_sensitivity_check(
    q_flat,
    *,
    p0, p1, N,
    EI1_v, EI2_v, GJ_v,
    ell_ref, EA_seg,
    wire_len, r_src, m_src,
    M_ref_local,
    ref_twist=None,
    eps_list=(1e-5, 3e-6, 1e-6, 3e-7),
):
    print(f"\n{'-'*80}")
    print("FINITE-DIFFERENCE FORCE SENSITIVITY")
    print(f"{'-'*80}")

    prev = None
    for eps in eps_list:
        f, dbg = total_force_from_state_fd(
            q_flat,
            p0=p0, p1=p1, N=N,
            EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
            ell_ref=ell_ref, EA_seg=EA_seg,
            wire_len=wire_len, r_src=r_src, m_src=m_src,
            M_ref_local=M_ref_local,
            ref_twist=ref_twist,
            eps=eps,
        )
        fnorm = np.linalg.norm(f)
        print(f"eps={eps:.1e}  ||f||={fnorm:.6e}")
        if prev is not None:
            rel = np.linalg.norm(f - prev) / (np.linalg.norm(prev) + 1e-12)
            print(f"             rel change from previous = {rel:.6e}")
        prev = f.copy()
def wrap_angle_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot

    from proper_research.parameters import default_magnet_params, default_beam_params
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz

    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    # User toggles
    # ------------------------------------------------------------------
    RUN_MIN_SOLVER = True
    RUN_RESIDUAL_SOLVER = False
    MAKE_PLOTS = True
    SHOW_FRAMES = False
    RUN_FD_SENSITIVITY = False
    USE_LUMEN = True

    # ------------------------------------------------------------------
    # Problem setup
    # ------------------------------------------------------------------
    beam_params = default_beam_params()
    mag_params = default_magnet_params()

    L_cmd = 0.055
    mag_len = beam_params.length_of_mag
    wire_len = L_cmd - mag_len
    N = 9

    mu_tip = mag_params.mag_epm
    # M_ref_local = np.array([0.0, 0.0, mu_tip], float)
    mag_total = mag_params.mag_epm          # total dipole moment of the whole tip magnet
    beam_params =default_beam_params()
    mu_line = beam_params.mag * beam_params.A_cs     # dipole moment per unit length
    M_ref_local = np.array([0, 0.0, mu_line], float)
    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], float)

    pivot_point = np.array([
        0.7681328220229531, -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
    ], float)

    start_point = get_point(0, 30)
    start_point[2] = -0.1

    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    Rm = Rot.from_quat([q_src_ur[1], q_src_ur[2], q_src_ur[3], q_src_ur[0]]).as_matrix()
    m_src = Rm @ m_body

    # ------------------------------------------------------------------
    # Material / section properties
    # ------------------------------------------------------------------
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    # tip = rod_section_stiffness(
    #     r=2e-3,
    #     E=3e6,
    #     nu=0.49,
    # )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )
    EA_wire = wire["EA"]
    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]

    EA_tip = tip["EA"]
    EI_tip = tip["EI"]
    GJ_tip = tip["GJ"]

    print("\n" + "=" * 80)
    print("CASE SETUP")
    print("=" * 80)
    print(f"L_cmd            : {L_cmd:.6e}")
    print(f"mag_len          : {mag_len:.6e}")
    print(f"wire_len         : {wire_len:.6e}")
    print(f"N                : {N}")
    print(f"base position    : {p0_ur}")
    print(f"magnet position  : {r_src_ur}")
    print(f"magnet dipole    : {m_src}")
    print(f"M_ref_local      : {M_ref_local}")
    print(f"EA_wire          : {EA_wire:.6e}")
    print(f"EI_wire          : {EI_wire:.6e}")
    print(f"GJ_wire          : {GJ_wire:.6e}")
    print(f"EA_tip           : {EA_tip:.6e}")
    print(f"EI_tip           : {EI_tip:.6e}")
    print(f"GJ_tip           : {GJ_tip:.6e}")

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )

    p_straight = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N)

    # ------------------------------------------------------------------
    # Shared constitutive data for debug checks
    # ------------------------------------------------------------------
    s_nodes = np.linspace(0.0, L_cmd, N)
    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(s_nodes, Kinv_fun, wire_len)
    ell_ref = np.full(N - 1, L_cmd / (N - 1), float)
    EA_seg = build_EA_edge_profile_from_params(
        s_nodes,
        wire_len=wire_len,
        EA_wire=EA_wire,
        EA_tip=EA_tip,
    )

    from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
    from proper_research.control.mpc_boundary import resample_polyline

    lumen_C = None
    lumen_R = None
    lumen_query = None

    if USE_LUMEN:
        # base tangent from your pivot pose
        Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
        t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

        s_straight = 0.02
        lumen_C = make_lumen_centerline_turning(
            p_start=p0_ur,
            t0=t0,
            length=0.06 + s_straight,
            n_pts=130,
            bend_axis=np.array([0.0, 0.0, 1.0]),
            bend_angle=np.deg2rad(40.0),
            bend_start=0.0 + s_straight,
            bend_end=0.03 + s_straight,
        )

        lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
        lumen_R = np.full(len(lumen_C), 0.004)

        lumen_query = LumenQuery(lumen_C, lumen_R)

        print("\n--- LUMEN SETUP ---")
        print("t0 =", t0)
        print("z variation in lumen =", lumen_C[:, 2].min(), lumen_C[:, 2].max())
        print("lumen pts =", len(lumen_C))
        print("lumen radius =", lumen_R[0])
    # ------------------------------------------------------------------
    # Run solvers
    # ------------------------------------------------------------------
    p_res = theta_res = info_res = None
    p_min = theta_min = info_min = None
    if RUN_MIN_SOLVER:
        p_min, theta_min, info_min = solve_nodes_twist_min(
            p0=p0_ur,
            q0=q0_ur,
            L=L_cmd,
            N=N,
            wire_len=wire_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src_ur,
            m_src=m_src,
            mu_tip=mu_tip,
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=USE_LUMEN,
            maxiter=200,
            M_ref_local=M_ref_local,
            ref_twist=None,
            enforce_inextensibility=False,
        )
    if RUN_RESIDUAL_SOLVER:
        p_res, theta_res, info_res = solve_nodes_twist_residual(
            p0=p0_ur,
            q0=q0_ur,
            L=L_cmd,
            N=N,
            wire_len=wire_len,
            Kinv_fun=Kinv_fun,
            r_src=r_src_ur,
            m_src=m_src,
            mu_tip=mu_line,
            EA_wire=EA_wire,
            EA_tip=EA_tip,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=USE_LUMEN,
            maxiter=200,
            M_ref_local=M_ref_local,
            ref_twist=None,
            q_init=info_min["q_opt"],
        )



    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    if info_res is not None:
        summarize_solution(
            "RESIDUAL SOLVER",
            p_res, theta_res, info_res,
            p0=p0_ur, q0=q0_ur, L=L_cmd, N=N,
            wire_len=wire_len, Kinv_fun=Kinv_fun,
            r_src=r_src_ur, m_src=m_src,
            EA_wire=EA_wire, EA_tip=EA_tip,
            M_ref_local=M_ref_local,
        )

    if info_min is not None:
        summarize_solution(
            "ENERGY MINIMIZATION SOLVER",
            p_min, theta_min, info_min,
            p0=p0_ur, q0=q0_ur, L=L_cmd, N=N,
            wire_len=wire_len, Kinv_fun=Kinv_fun,
            r_src=r_src_ur, m_src=m_src,
            EA_wire=EA_wire, EA_tip=EA_tip,
            M_ref_local=M_ref_local,
        )

    if (info_res is not None) and (info_min is not None):
        compare_solutions(
            "residual", p_res, theta_res, info_res,
            "minimization", p_min, theta_min, info_min
        )

    # ------------------------------------------------------------------
    # Extra debug: force sensitivity around converged residual solution
    # ------------------------------------------------------------------
    if RUN_FD_SENSITIVITY and (info_res is not None):
        q_res = info_res["q_opt"]
        p1_res = p_straight[:, 1].copy()

        fd_force_sensitivity_check(
            q_res,
            p0=p0_ur, p1=p1_res, N=N,
            EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
            ell_ref=ell_ref, EA_seg=EA_seg,
            wire_len=wire_len,
            r_src=r_src_ur, m_src=m_src,
            M_ref_local=M_ref_local,
            ref_twist=None,
            eps_list=(1e-5, 3e-6, 1e-6, 3e-7),
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if MAKE_PLOTS and (info_res is not None):
        plot_nodes_and_magnet_3d(
            p_res,
            p0=p0_ur,
            r_src=r_src_ur,
            m_src=m_src,
            p_straight=p_straight,
            lumen_C=lumen_C,
            title="Residual solver result"
        )
        plot_solution_diagnostics(
            p_res, theta_res, info_res,
            L=L_cmd, N=N, wire_len=wire_len,
            title_prefix="residual solver"
        )

    if MAKE_PLOTS and (info_min is not None):
        plot_nodes_and_magnet_3d(
            p_min,
            p0=p0_ur,
            r_src=r_src_ur,
            m_src=m_src,
            p_straight=p_straight,
            lumen_C=lumen_C,
            title="Minimization solver result"
        )
        plot_solution_diagnostics(
            p_min, theta_min, info_min,
            L=L_cmd, N=N, wire_len=wire_len,
            title_prefix="minimization solver"
        )

    if SHOW_FRAMES and (info_res is not None):
        kin = info_res["dbg"]["kin"]
        plot_reference_and_material_frames_3d(
            p_res,
            kin["d1_ref"], kin["d2_ref"], kin["d3_ref"],
            kin["m1"], kin["m2"], kin["m3"],
            scale=0.0015,
            stride=3,
            title="Reference/material frames along residual solution"
        )