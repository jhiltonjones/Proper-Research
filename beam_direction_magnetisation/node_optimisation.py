import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from beam_direction_magnetisation.quarternions.quarternions_functions import (quat_derivative_body, quat_normalize, 
                                                                              quat_to_rot, quat_to_R,
                                                                              T_to_p_quat_wxyz)
from beam_direction_magnetisation.cosserat_w_minimal_energy import contact_barrier_energy_and_force_fast, LumenQuery
from proper_research.robot.transformations import get_point
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

    Wb, bend_dbg = bending_energy_turning_angle(p, EI_v, ell)
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
    L_cmd = 0.012
    mag_len = beam_params.length_of_mag
    wire_len = L_cmd - mag_len

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0])

    pivot_point = np.array([
    0.7681328220229531, -0.7112731669220016,-0.1,  np.pi, 0.001,0.001
    ], float)
    start_point = get_point(0,-30)
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
    p_nodes, info = solve_nodes_min(
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
        maxiter=1000,
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
    theta = info["dbg"]["theta"]
    s_nodes = np.linspace(0.0, L_cmd, N)
    s_vertex = s_nodes[1:-1]

    plt.figure()
    plt.plot(p_nodes[0, :], p_nodes[1, :], "-o")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Node-based centerline projection (x-y)")
    plt.show()

    plt.figure()
    plt.plot(s_vertex, theta, "-o")
    plt.axvline(wire_len, linestyle="--")
    plt.xlabel("s [m]")
    plt.ylabel("turning angle [rad]")
    plt.title("Curvature proxy along rod")
    plt.show()