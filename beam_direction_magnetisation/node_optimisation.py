import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from beam_direction_magnetisation.quarternions.quarternions_functions import (quat_normalize, 
                                                                            quat_to_R,
                                                                              T_to_p_quat_wxyz)
from beam_direction_magnetisation.cosserat_w_minimal_energy import contact_barrier_energy_and_force_fast, LumenQuery
from proper_research.robot.transformations import get_point
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
def build_EA_edge_profile(s_nodes, Kinv_fun, wire_len):
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    Kinv_mid = Kinv_fun(s_mid, wire_len)

    EA_seg = np.zeros(s_mid.size)
    for i in range(s_mid.size):
        K = np.linalg.inv(Kinv_mid[:, :, i])
        # placeholder: use your best available axial stiffness entry here
        # adjust depending on your constitutive convention
        EA_seg[i] = 1.0 / max(Kinv_mid[0,0,i], 1e-12)  # likely needs refinement
    return EA_seg
def stretch_energy_der(ell, ell_ref, EA_seg):
    eps_s = stretch_strain_from_lengths(ell, ell_ref)
    Ws = 0.5 * np.sum(EA_seg * (eps_s**2) * ell_ref)
    return float(Ws), dict(eps_s=eps_s)
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
def bending_energy_der_anisotropic(p, EI1_v, EI2_v, theta_twist=None, eps=1e-12):
    der = der_bending_strains_from_frames(p, theta_twist=theta_twist, eps=eps)

    kappa1 = der["kappa1"]
    kappa2 = der["kappa2"]
    dl = der["dl_voronoi"]

    Wb = 0.5 * np.sum((EI1_v * kappa1**2 + EI2_v * kappa2**2) * dl)

    dbg = dict(
        kappa1=kappa1,
        kappa2=kappa2,
        kb=der["kb"],
        dl_voronoi=dl,
        t_bend=der["t"],
        d1_ref=der["d1_ref"],
        d2_ref=der["d2_ref"],
        d3_ref=der["d3_ref"],
        m1=der["m1"],
        m2=der["m2"],
        m3=der["m3"],
    )
    return float(Wb), dbg
def magnetic_energy_segments_material_frame(
    p, theta_twist, wire_len, r_src, m_src, M_ref_local
):
    """
    DER-style edge magnetic energy using the material frame on each edge.

    p            : (3,N) node positions
    theta_twist  : (N-1,) twist angle per edge
    wire_len     : scalar, segments with s_seg >= wire_len are magnetic
    r_src        : (3,) permanent magnet position
    m_src        : (3,) permanent magnet dipole in world frame
    M_ref_local  : (3,) reference magnetization vector in edge material frame
                   e.g. axial magnetization -> [0, 0, mu_tip]

    Returns
    -------
    Wm, dbg
    """
    der = der_bending_strains_from_frames(p, theta_twist=theta_twist)

    m1 = der["m1"]   # (3, N-1)
    m2 = der["m2"]
    m3 = der["m3"]
    ell_i = der["ell"]
    t = der["t"]

    mid = segment_midpoints(p)   # (3, N-1)

    s_nodes = np.zeros(p.shape[1])
    s_nodes[1:] = np.cumsum(ell_i)
    s_seg = 0.5 * (s_nodes[:-1] + s_nodes[1:])

    B = dipole_field_points(mid, r_src, m_src)   # (3, N-1)

    M_ref_local = np.asarray(M_ref_local, float).reshape(3,)
    mag_mask = (s_seg >= wire_len).astype(float)

    # world magnetization on each edge from material frame
    M_world = (
        m1 * M_ref_local[0]
        + m2 * M_ref_local[1]
        + m3 * M_ref_local[2]
    )  # (3, N-1)

    # zero magnetization in wire region
    M_world = M_world * mag_mask[None, :]

    wm = -np.sum(M_world * B, axis=0)
    Wm = np.sum(wm * ell_i)

    dbg = dict(
        B=B,
        s_seg=s_seg,
        M_world=M_world,
        mag_mask=mag_mask,
        t_seg=t,
        m1=m1,
        m2=m2,
        m3=m3,
    )
    return float(Wm), dbg
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

def build_GJ_vertex_profile(s_nodes, Kinv_fun, wire_len):
    """
    Returns GJ on interior vertices, shape (N-2,)
    """
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    Kinv_mid = Kinv_fun(s_mid, wire_len)

    GJ_seg = np.zeros(s_mid.size)
    for i in range(s_mid.size):
        K = np.linalg.inv(Kinv_mid[:, :, i])
        GJ_seg[i] = K[0, 0]

    GJ_v = 0.5 * (GJ_seg[:-1] + GJ_seg[1:])
    return GJ_v

def twist_energy_der(theta_twist, dl_voronoi, GJ_v, ref_twist=None, eps=1e-12):
    """
    DER-style twist energy:
      Wt = 0.5 * sum GJ * kappa3^2 * dl
    """
    kappa3 = twist_strain_from_theta(theta_twist, dl_voronoi, ref_twist=ref_twist, eps=eps)
    Wt = 0.5 * np.sum(GJ_v * (kappa3**2) * dl_voronoi)
    dbg = dict(kappa3=kappa3)
    return float(Wt), dbg
def total_energy_der_q(
    q_flat, *, p0, p1, N, EI1_v, EI2_v, GJ_v,
    wire_len, r_src, m_src, mu_tip,
    lumen_query=None, use_lumen=False
):
    p, theta_twist = unpack_q_to_nodes_twist(q_flat, p0, p1, N)

    Wb, bend_dbg = bending_energy_der_anisotropic(
        p, EI1_v, EI2_v, theta_twist=theta_twist
    )

    dl_voronoi = bend_dbg["dl_voronoi"]
    Wt, twist_dbg = twist_energy_der(theta_twist, dl_voronoi, GJ_v, ref_twist=None)

    M_ref_local = np.array([0.0, 0.0, mu_tip])
    # M_ref_local = np.array([mu_tip, 0.0, 0.0])
    Wm, mag_dbg = magnetic_energy_segments_material_frame(
        p, theta_twist, wire_len, r_src, m_src, M_ref_local
    )

    Wc = 0.0
    con_dbg = {}
    if use_lumen and (lumen_query is not None):
        Wc, con_dbg = lumen_contact_energy_nodes(p, lumen_query)

    W = Wb + Wt + Wm + Wc
    dbg = {
        "Wb": Wb,
        "Wt": Wt,
        "Wm": Wm,
        "Wc": Wc,
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
    *, p0, q0, L, N,
    wire_len, Kinv_fun,
    r_src, m_src, mu_tip,
    lumen_C=None, lumen_R=None,
    use_lumen=False,
    maxiter=300
):
    ell = L / (N - 1)

    p_init = make_initial_nodes_perturbed_feasible(p0, q0, L, N, amp=2e-4)
    p1 = p_init[:, 1].copy()

    s_nodes = np.linspace(0.0, L, N)
    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(s_nodes, Kinv_fun, wire_len)

    theta0 = np.zeros(N-1, float)
    q0_flat = pack_q_from_nodes_twist(p_init, theta0)

    lumen_query = None
    if (lumen_C is not None) and (lumen_R is not None):
        lumen_query = LumenQuery(lumen_C, lumen_R)

    def obj(q_flat):
        W, _ = total_energy_der_q(
            q_flat,
            p0=p0, p1=p1, N=N,
            EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
            wire_len=wire_len, r_src=r_src, m_src=m_src, mu_tip=mu_tip,
            lumen_query=lumen_query, use_lumen=use_lumen
        )
        return W

    cons = NonlinearConstraint(
        lambda q_flat: segment_length_residuals_q(q_flat, p0=p0, p1=p1, N=N, ell=ell),
        lb=np.zeros(N-2),
        ub=np.zeros(N-2),
    )

    res = minimize(
        obj,
        q0_flat,
        method="SLSQP",
        constraints=[cons],
        options=dict(
            maxiter=maxiter,
            ftol=1e-6,
            eps=1e-7,
            disp=True,
        ),
    )

    p_opt, theta_opt = unpack_q_to_nodes_twist(res.x, p0, p1, N)
    W, dbg = total_energy_der_q(
        res.x,
        p0=p0, p1=p1, N=N,
        EI1_v=EI1_v, EI2_v=EI2_v, GJ_v=GJ_v,
        wire_len=wire_len, r_src=r_src, m_src=m_src, mu_tip=mu_tip,
        lumen_query=lumen_query, use_lumen=use_lumen
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


def der_bending_strains_from_frames(p, theta_twist=None, eps=1e-12):
    """
    Compute DER-style bending strains at interior vertices.

    Inputs
    ------
    p : (3, N) node positions
    theta_twist : (N-1,) per-edge twist angles.
                  If None, uses zero twist.

    Returns
    -------
    out : dict with
      ell        : (N-1,) edge lengths
      t          : (3, N-1) edge tangents
      d1_ref     : (3, N-1)
      d2_ref     : (3, N-1)
      d3_ref     : (3, N-1)
      m1         : (3, N-1)
      m2         : (3, N-1)
      m3         : (3, N-1)
      kb         : (3, N-2) curvature binormals at interior vertices
      dl_voronoi : (N-2,) interior Voronoi lengths
      kappa1     : (N-2,)
      kappa2     : (N-2,)
    """
    e, t, ell = edge_tangents_and_lengths(p, eps=eps)
    n_edge = t.shape[1]

    if theta_twist is None:
        theta_twist = np.zeros(n_edge, float)
    else:
        theta_twist = np.asarray(theta_twist, float).reshape(n_edge,)

    d1_ref, d2_ref, d3_ref, ell_chk = parallel_transport_reference_frames(p)
    m1, m2, m3 = material_frames_from_twist(d1_ref, d2_ref, d3_ref, theta_twist)

    kb = curvature_binormals_from_tangents(t, ell, eps=eps)   # (3, N-2)
    dl_voronoi = voronoi_lengths_from_edge_lengths(ell)       # (N-2,)

    # vertex material directions: average adjacent edge material frames
    m1_v = m1[:, :-1] + m1[:, 1:]   # (3, N-2)
    m2_v = m2[:, :-1] + m2[:, 1:]   # (3, N-2)

    # normalize vertex frame directions
    m1_v /= (np.linalg.norm(m1_v, axis=0, keepdims=True) + eps)
    m2_v /= (np.linalg.norm(m2_v, axis=0, keepdims=True) + eps)

    # DER-style projected bending strains
    # sign conventions vary by reference; consistency matters more than sign at this stage
    kappa1 = np.sum(kb * m2_v, axis=0) / (dl_voronoi + eps)
    kappa2 = -np.sum(kb * m1_v, axis=0) / (dl_voronoi + eps)

    return dict(
        ell=ell,
        t=t,
        d1_ref=d1_ref, d2_ref=d2_ref, d3_ref=d3_ref,
        m1=m1, m2=m2, m3=m3,
        kb=kb,
        dl_voronoi=dl_voronoi,
        kappa1=kappa1,
        kappa2=kappa2,
    )
# def bending_energy_der(p, EI_v, theta_twist=None, eps=1e-12):
#     """
#     DER-style bending energy using curvature binormal projected into material directions.

#     EI_v: (N-2,) effective bending stiffness at interior vertices
#     theta_twist: (N-1,) edge twist angles, or None -> zeros
#     """
#     der = der_bending_strains_from_frames(p, theta_twist=theta_twist, eps=eps)

#     kappa1 = der["kappa1"]
#     kappa2 = der["kappa2"]
#     dl = der["dl_voronoi"]

#     Wb = 0.5 * np.sum(EI_v * (kappa1**2 + kappa2**2) * dl)

#     dbg = dict(
#         kappa1=kappa1,
#         kappa2=kappa2,
#         kb=der["kb"],
#         dl_voronoi=dl,
#         t_bend=der["t"],
#         d1_ref=der["d1_ref"],
#         d2_ref=der["d2_ref"],
#         d3_ref=der["d3_ref"],
#         m1=der["m1"],
#         m2=der["m2"],
#         m3=der["m3"],
#     )
#     return float(Wb), dbg
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

def vertex_arclengths(seg_len):
    """
    Voronoi-style lengths for interior vertices
    """
    return 0.5 * (seg_len[:-1] + seg_len[1:])
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
def build_EI_vertex_profile(s_nodes, Kinv_fun, wire_len):
    """
    Returns EI on interior vertices, shape (N-2,)
    """
    s_mid = 0.5 * (s_nodes[:-1] + s_nodes[1:])      # segment locations
    Kinv_mid = Kinv_fun(s_mid, wire_len)            # (3,3,N-1)

    EI_seg = np.zeros(s_mid.size)
    for i in range(s_mid.size):
        K = np.linalg.inv(Kinv_mid[:, :, i])
        EI_seg[i] = 0.5 * (K[1,1] + K[2,2])

    # map segment stiffness to interior vertices
    EI_v = 0.5 * (EI_seg[:-1] + EI_seg[1:])
    return EI_v
def bending_energy_turning_angle(p, EI_v, ell, eps=1e-12):
    t = tangents_from_nodes(p, eps=eps)  # (3,N-1)

    dots = np.sum(t[:, :-1] * t[:, 1:], axis=0)
    dots = np.clip(dots, -1.0, 1.0)
    theta = np.arccos(dots)   # (N-2,)

    Wb = 0.5 * np.sum(EI_v * (theta**2) / ell)
    return float(Wb), dict(theta=theta, t_bend=t)
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
def magnetic_energy_segments(p, wire_len, r_src, m_src, mu_tip):
    mid = segment_midpoints(p)          # (3,N-1)
    seg = edge_vectors(p)
    ell_i = np.linalg.norm(seg, axis=0)
    t = seg / (ell_i[None, :] + 1e-12)

    s_nodes = np.zeros(p.shape[1])
    s_nodes[1:] = np.cumsum(ell_i)
    s_seg = 0.5 * (s_nodes[:-1] + s_nodes[1:])

    mu = magnetic_density_profile_segments(s_seg, wire_len, mu_tip)
    B = dipole_field_points(mid, r_src, m_src)

    m_world = t * mu[None, :]
    wm = -np.sum(m_world * B, axis=0)

    Wm = np.sum(wm * ell_i)
    return float(Wm), dict(B=B, s_seg=s_seg, mu=mu, t_seg=t)
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
def total_energy_nodes(
    x, *, p0, p1, N, EI_v, ell,
    wire_len, r_src, m_src, mu_tip,
    lumen_query=None, use_lumen=False
):
    p = unpack_x_to_nodes(x, p0, p1, N)

    theta_twist = np.zeros(N-1, float)
    Wb, bend_dbg = bending_energy_der_anisotropic(
    p, EI1_v, EI2_v, theta_twist=theta_twist
    )
    Wm, mag_dbg  = magnetic_energy_segments(p, wire_len, r_src, m_src, mu_tip)

    Wc = 0.0
    con_dbg = {}
    if use_lumen and (lumen_query is not None):
        Wc, con_dbg = lumen_contact_energy_nodes(p, lumen_query)

    W = Wb + Wm + Wc
    dbg = dict(Wb=Wb, Wm=Wm, Wc=Wc, **bend_dbg, **mag_dbg, **con_dbg)
    return float(W), dbg
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
def solve_nodes_min(
    *, p0, q0, L, N,
    wire_len, Kinv_fun,
    r_src, m_src, mu_tip,
    lumen_C=None, lumen_R=None,
    use_lumen=False,
    maxiter=300
):
    ell = L / (N - 1)

    p_init = make_initial_nodes_perturbed_feasible(p0, q0, L, N, amp=2e-4)
    p1 = p_init[:, 1].copy()

    s_nodes = np.linspace(0.0, L, N)
    EI_v = build_EI_vertex_profile(s_nodes, Kinv_fun, wire_len)

    x0 = p_init[:, 2:].T.reshape(-1)

    lumen_query = None
    if (lumen_C is not None) and (lumen_R is not None):
        lumen_query = LumenQuery(lumen_C, lumen_R)

    def obj(x):
        W, _ = total_energy_nodes(
            x,
            p0=p0, p1=p1, N=N, EI_v=EI_v, ell=ell,
            wire_len=wire_len, r_src=r_src, m_src=m_src, mu_tip=mu_tip,
            lumen_query=lumen_query, use_lumen=use_lumen
        )
        return W

    cons = NonlinearConstraint(
        lambda x: segment_length_residuals(x, p0=p0, p1=p1, N=N, ell=ell),
        lb=np.zeros(N-2),
        ub=np.zeros(N-2),
    )

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        constraints=[cons],
        options=dict(
            maxiter=maxiter,
            ftol=1e-6,
            eps=1e-7,
            disp=True,
        ),
    )

    p_opt = unpack_x_to_nodes(res.x, p0, p1, N)
    W, dbg = total_energy_nodes(
        res.x,
        p0=p0, p1=p1, N=N, EI_v=EI_v, ell=ell,
        wire_len=wire_len, r_src=r_src, m_src=m_src, mu_tip=mu_tip,
        lumen_query=lumen_query, use_lumen=use_lumen
    )

    info = dict(success=bool(res.success), message=res.message, nit=res.nit, W=W, dbg=dbg, res=res)
    return p_opt, info
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

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
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as Rot

    from proper_research.parameters import default_magnet_params, default_beam_params
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
    from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile

    beam_params = default_beam_params()
    mag_params = default_magnet_params()

    # -----------------------------
    # Problem setup
    # -----------------------------
    L_cmd = 0.045
    mag_len = beam_params.length_of_mag
    wire_len = L_cmd - mag_len

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0])

    pivot_point = np.array([
    0.7681328220229531, -0.7112731669220016,-0.1,  np.pi, 0.001,0.001
    ], float)
    start_point = get_point(0,30)
    start_point[2] = -0.1
    # start_point = np.array([
    #     0.601, -0.649, 0.092,
    #     -3.054, -0.476, 0.05
    # ], float)

    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    # magnet dipole in world frame
    q = q_src_ur
    Rm = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    m_src = Rm @ m_body

    # -----------------------------
    # Choose a scalar mu_tip
    # -----------------------------
    mu_tip = mag_params.mag_epm

    # -----------------------------
    # Solve node-based model
    # -----------------------------
    N = 25
    # p_nodes, info = solve_nodes_min(
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     L=L_cmd,
    #     N=N,
    #     wire_len=wire_len,
    #     Kinv_fun=Kbt_inv_profile,
    #     r_src=r_src_ur,
    #     m_src=m_src,
    #     mu_tip=mu_tip,
    #     lumen_C=None,
    #     lumen_R=None,
    #     use_lumen=False,
    #     maxiter=1000,
    # )
    p_nodes, theta_twist_opt, info = solve_nodes_twist_min(
        p0=p0_ur,
        q0=q0_ur,
        L=L_cmd,
        N=N,
        wire_len=wire_len,
        Kinv_fun=Kbt_inv_profile,
        r_src=r_src_ur,
        m_src=m_src,
        mu_tip=mu_tip,
        lumen_C=None,
        lumen_R=None,
        use_lumen=False,
        maxiter=200,
    )
    print("\n--- NODE SOLVER INFO ---")
    print("success:", info["success"])
    print("message:", info["message"])
    print("iterations:", info["nit"])
    print("W total:", info["W"])
    print("Wb:", info["dbg"]["Wb"])
    print("Wm:", info["dbg"]["Wm"])
    print("tip position:", p_nodes[:, -1])
    p_straight = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N)
    der_dbg = der_bending_strains_from_frames(p_nodes, theta_twist=theta_twist_opt)
    print("Wt:", info["dbg"]["Wt"])
    print("max |kappa3|:", np.max(np.abs(info["dbg"]["twist"]["kappa3"])))
    M_world = info["dbg"]["mag"]["M_world"]
    print("max |M_world| on edges:", np.max(np.linalg.norm(M_world, axis=0)))
    print("max |kappa1|:", np.max(np.abs(der_dbg["kappa1"])))
    print("max |kappa2|:", np.max(np.abs(der_dbg["kappa2"])))
    print("mean Voronoi length:", np.mean(der_dbg["dl_voronoi"]))

    plot_reference_and_material_frames_3d(
        p_nodes,
        der_dbg["d1_ref"], der_dbg["d2_ref"], der_dbg["d3_ref"],
        der_dbg["m1"], der_dbg["m2"], der_dbg["m3"],
        scale=0.0015,
        stride=3,
        title="Reference/material frames along optimized rod"
    )
    plot_nodes_and_magnet_3d(
        p_nodes,
        p0=p0_ur,
        r_src=r_src_ur,
        m_src=m_src,          # world dipole vector
        p_straight=p_straight,
        title="Node-optimised rod with permanent magnet"
    )   
    print("straight tip:", p_straight[:, -1])
    print("solved tip:  ", p_nodes[:, -1])
    print("tip diff:    ", p_nodes[:, -1] - p_straight[:, -1])
    print("tip diff norm:", np.linalg.norm(p_nodes[:, -1] - p_straight[:, -1]))
    # -----------------------------
    # Constraint check
    # -----------------------------
    ell = L_cmd / (N - 1)
    p_init = make_initial_nodes_straight(p0_ur, q0_ur, L_cmd, N)
    p1 = p_init[:, 1]
    x_opt = p_nodes[:, 2:].T.reshape(-1)

    cres = segment_length_residuals(x_opt, p0=p0_ur, p1=p1, N=N, ell=ell)
    print("max |segment length residual|:", np.max(np.abs(cres)))

    # -----------------------------
    # Curvature localization check
    # -----------------------------
    kappa1 = info["dbg"]["bend"]["kappa1"]
    kappa2 = info["dbg"]["bend"]["kappa2"]
    s_nodes = np.linspace(0.0, L_cmd, N)
    EI1_v, EI2_v, GJ_v = build_bending_twist_profiles(s_nodes, Kbt_inv_profile, wire_len)

    s_vertex = s_nodes[1:-1]
    wire_mask = s_vertex < wire_len
    tip_mask  = s_vertex >= wire_len
    print("wire_len:", wire_len)
    print("s_vertex[0]:", s_vertex[0], "s_vertex[-1]:", s_vertex[-1])
    print("num wire vertices:", np.sum(wire_mask))
    print("num tip vertices:", np.sum(tip_mask))
    print("EI1 wire mean:", EI1_v[wire_mask].mean() if np.any(wire_mask) else None)
    print("EI2 wire mean:", EI2_v[wire_mask].mean() if np.any(wire_mask) else None)
    print("EI1 tip mean:", EI1_v[tip_mask].mean() if np.any(tip_mask) else None)
    print("EI2 tip mean:", EI2_v[tip_mask].mean() if np.any(tip_mask) else None)
    plt.figure()
    plt.plot(s_vertex, kappa1, "-o", label="kappa1")
    plt.plot(s_vertex, kappa2, "-o", label="kappa2")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("bending strain [1/m]")
    plt.title("DER bending strains along rod")
    plt.legend()
    plt.show()

