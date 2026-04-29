import numpy as np
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile, magnetic_wrench_density_cosserat_profile_segments
from beam_direction_magnetisation.quarternions.quarternions_functions import (quat_derivative_body, quat_normalize, 
                                                                              quat_to_rot, quat_to_R,
                                                                              T_to_p_quat_wxyz)
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
from beam_direction_magnetisation.quarternions.shared_rotations import Ry, Rz
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
from proper_research.parameters import default_magnet_params, default_beam_params
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.post_processing.post_processing import (plot_centerlines_with_lumen_3d, make_lumen_centerline_turning, 
                                                                          plot_error_vs_s, closest_point_on_segment, point_to_polyline_distance)
from proper_research.robot.transformations import get_point
from scipy.optimize import minimize
from beam_direction_magnetisation.cosserat_w_minimal_energy import (integrate_pq_from_u, precompute_K_segments, LumenQuery, contact_barrier_energy_and_force_fast,resample_polyline, rod_section_stiffness, 
                                                                    effective_lengths, dipole_from_pose, interpolate_radius, make_Kbt_inv_profile, lumen_violation_profile)
from beam_direction_magnetisation.ana_jaccobian import perturb_q_src, benchmark_function, print_benchmark_result
beam_params = default_beam_params()
mag_params = default_magnet_params()
L_tip_full = 0.04
def energy_min_tip_jacobian_implicit(
    *,
    u_opt,
    p0, q0,
    L,
    wire_len,
    Kinv_fun,
    u_star,
    r_src,
    m_src,
    m_local_fun,
    m_moment,
    N,
    theta_builder,
    energy_grad_fun,
    eps_theta=1e-6,
    eps_hess=1e-6,
):
    """
    Implicit Jacobian for energy minimisation.

    theta is whatever controls you want:
      [r_src(3), source rotation(3), L(1)] -> 7 columns.

    Requires:
      energy_grad_fun(u_flat, theta) -> dE/du_flat
      theta_builder(theta) -> dict containing updated L, s, K_seg, r_src, m_src, etc.
    """

    u_opt = np.asarray(u_opt, float).reshape(-1)
    n_u = u_opt.size

    theta0 = np.asarray(theta_builder(None)["theta0"], float).reshape(-1)
    n_theta = theta0.size

    # -------------------------------------------------
    # Hessian H = d/du grad_u E
    # -------------------------------------------------
    g0 = energy_grad_fun(u_opt, theta0)
    H = np.zeros((n_u, n_u), dtype=float)

    for k in range(n_u):
        up = u_opt.copy()
        um = u_opt.copy()
        up[k] += eps_hess
        um[k] -= eps_hess

        gp = energy_grad_fun(up, theta0)
        gm = energy_grad_fun(um, theta0)

        H[:, k] = (gp - gm) / (2.0 * eps_hess)

    # regularise very lightly for numerical safety
    H_reg = H + 1e-10 * np.eye(n_u)

    # -------------------------------------------------
    # Gtheta = d/dtheta grad_u E
    # -------------------------------------------------
    Gtheta = np.zeros((n_u, n_theta), dtype=float)

    for j in range(n_theta):
        thp = theta0.copy()
        thm = theta0.copy()
        thp[j] += eps_theta
        thm[j] -= eps_theta

        gp = energy_grad_fun(u_opt, thp)
        gm = energy_grad_fun(u_opt, thm)

        Gtheta[:, j] = (gp - gm) / (2.0 * eps_theta)

    # -------------------------------------------------
    # du*/dtheta = -H^{-1} Gtheta
    # -------------------------------------------------
    du_dtheta = -np.linalg.solve(H_reg, Gtheta)

    # -------------------------------------------------
    # dp_tip/du
    # -------------------------------------------------
    p_base, q_base, _ = integrate_pq_from_u(
        u_opt,
        p0=p0,
        q0=q0,
        s=np.linspace(0.0, float(L), int(N)),
    )
    p_tip_base = p_base[:, -1]

    P_u = np.zeros((3, n_u), dtype=float)

    for k in range(n_u):
        up = u_opt.copy()
        um = u_opt.copy()
        up[k] += eps_hess
        um[k] -= eps_hess

        pp, _, _ = integrate_pq_from_u(
            up, p0=p0, q0=q0, s=np.linspace(0.0, float(L), int(N))
        )
        pm, _, _ = integrate_pq_from_u(
            um, p0=p0, q0=q0, s=np.linspace(0.0, float(L), int(N))
        )

        P_u[:, k] = (pp[:, -1] - pm[:, -1]) / (2.0 * eps_hess)

    # -------------------------------------------------
    # direct dp_tip/dtheta
    # Usually zero for magnet pose.
    # For L, direct term is approximately tip tangent.
    # -------------------------------------------------
    P_theta_direct = np.zeros((3, n_theta), dtype=float)

    # if final theta column is L:
    q_tip = q_base[:, -1]
    R_tip = quat_to_R(q_tip)
    t_tip = R_tip @ np.array([-1.0, 0.0, 0.0])
    P_theta_direct[:, -1] = t_tip

    J_tip_theta = P_u @ du_dtheta + P_theta_direct

    return J_tip_theta, dict(
        H=H,
        Gtheta=Gtheta,
        du_dtheta=du_dtheta,
        P_u=P_u,
        p_tip_base=p_tip_base,
    )
def elastic_energy_gradient_u(u_flat, *, s, K_seg, u_star):
    u_seg = np.asarray(u_flat, float).reshape(len(s) - 1, 3)
    ds = np.diff(s)
    u_star = np.asarray(u_star, float).reshape(3,)

    grad = np.zeros_like(u_seg)

    for i in range(len(ds)):
        grad[i] = ds[i] * (K_seg[i] @ (u_seg[i] - u_star))

    return grad.reshape(-1)
def quat_wxyz_to_scipy(q):
    q = np.asarray(q, float).reshape(4,)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def q_mid_average(q):
    """
    q: (4,N)
    returns midpoint quaternions, shape (4,N-1)
    """
    q = quat_normalize(q)
    qmid = q[:, :-1] + q[:, 1:]

    # avoid sign cancellation
    dots = np.sum(q[:, :-1] * q[:, 1:], axis=0)
    flip = dots < 0.0
    qmid[:, flip] = q[:, :-1][:, flip] - q[:, 1:][:, flip]

    return quat_normalize(qmid)


def rotation_vector_between(q_base, q_pert):
    """
    Small world-frame rotation vector taking q_base -> q_pert.
    Both quaternions are wxyz.
    """
    R0 = Rot.from_quat(quat_wxyz_to_scipy(q_base))
    Rp = Rot.from_quat(quat_wxyz_to_scipy(q_pert))
    return (Rp * R0.inv()).as_rotvec()
def magnetic_energy_gradient_u_virtual_work(
    u_flat,
    *,
    p0,
    q0,
    s,
    m_src,
    r_src,
    m_local_fun,
    m_moment,
    eps_kin=1e-7,
):
    """
    Gradient dW_m / du_flat using magnetic virtual work.

    W_m = - integral m_world · B ds

    Uses:
        dW_m = - integral (f · dp + tau · dtheta) ds

    Returns
    -------
    grad : shape same as u_flat
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)

    p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    f_mid, tau_mid, B_mid, m_mid_world, s_mid = magnetic_wrench_density_cosserat_profile_segments(
        p,
        q,
        s,
        m_src,
        r_src,
        m_local_fun,
        m_moment,
        r_min=1e-6,
    )

    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    q_mid = q_mid_average(q)

    grad = np.zeros_like(u_flat)

    for k in range(u_flat.size):
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps_kin
        um[k] -= eps_kin

        pp, qp, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, qm, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        pp_mid = 0.5 * (pp[:, :-1] + pp[:, 1:])
        pm_mid = 0.5 * (pm[:, :-1] + pm[:, 1:])

        qp_mid = q_mid_average(qp)
        qm_mid = q_mid_average(qm)

        dp_du = (pp_mid - pm_mid) / (2.0 * eps_kin)

        dtheta_du = np.zeros((3, s.size - 1), dtype=float)
        for i in range(s.size - 1):
            theta_p = rotation_vector_between(q_mid[:, i], qp_mid[:, i])
            theta_m = rotation_vector_between(q_mid[:, i], qm_mid[:, i])
            dtheta_du[:, i] = (theta_p - theta_m) / (2.0 * eps_kin)

        dW = 0.0
        for i in range(s.size - 1):
            dW += -(
                np.dot(f_mid[:, i], dp_du[:, i])
                + np.dot(tau_mid[:, i], dtheta_du[:, i])
            ) * ds[i]

        grad[k] = dW

    return grad
def contact_energy_gradient_u(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    eps_kin=1e-7,
):
    """
    Analytic contact gradient via chain rule:

        dW/du = - sum_j F_j · (dp_j/du)

    where F_j comes from contact_barrier_energy_and_force_fast.
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)

    # base configuration
    p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    # contact forces at nodes
    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=0.001,      # or whatever your beam radius variable is
        k_contact=1e3,
        pen_switch=5e-5,
        k_hard=1e8,
        eps=1e-12,
        window=3,
        smooth=True,              # start with nonsmooth pure contact
        smooth_eps=1e-5,
    )

    grad = np.zeros_like(u_flat)

    # finite-difference kinematics only (NOT energy)
    for k in range(u_flat.size):
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps_kin
        um[k] -= eps_kin

        pp, _, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, _, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        dp_du = (pp - pm) / (2.0 * eps_kin)   # (3,N)

        # chain rule
        dW = 0.0
        for j in range(p.shape[1]):
            dW += -np.dot(F_nodes[:, j], dp_du[:, j])

        # consistent scaling with your energy:
        dW *= (s[1] - s[0])

        grad[k] = dW

    return grad
def energy_gradient_u(
    u_flat,
    *,
    p0,
    q0,
    s,
    K_seg,
    u_star,
    m_src,
    r_src,
    m_local_fun,
    m_moment,
    lumen_query=None,
    use_magnetic=True,
    use_contact=True,
):
    grad = elastic_energy_gradient_u(
        u_flat,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
    )

    if use_magnetic:
        grad += magnetic_energy_gradient_u_virtual_work(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
        )

    if use_contact and (lumen_query is not None):
        grad += contact_energy_gradient_u(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            lumen_query=lumen_query,
        )

    return grad
def contact_energy_gradient_u_faster_fdkin(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    eps_kin=1e-7,
):
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    N = s.size
    n_seg = N - 1
    h = float(s[1] - s[0])

    p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=beam_params.r,
        k_contact=1e5,
        pen_switch=5e-5,
        k_hard=1e10,
        eps=1e-12,
        window=3,
        smooth=True,
        smooth_eps=1e-7,
    )

    W_p = -h * F_nodes
    active_nodes = np.where(np.linalg.norm(W_p, axis=0) > 0.0)[0]

    if active_nodes.size == 0:
        return np.zeros_like(u_flat)

    grad = np.zeros_like(u_flat)

    for k in range(u_flat.size):
        seg_i = k // 3

        # p_j is affected only if j > seg_i
        active = active_nodes[active_nodes > seg_i]
        if active.size == 0:
            continue

        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps_kin
        um[k] -= eps_kin

        pp, _, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, _, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        dp_du_k = (pp[:, active] - pm[:, active]) / (2.0 * eps_kin)
        grad[k] = np.sum(W_p[:, active] * dp_du_k)

    return grad
def solve_energy_min_3d(
    *, p0, q0, L, wire_len, Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    N=60, u0_flat=None, maxiter=300,
    lumen_C=None, lumen_R=None, use_lumen=True,
    contact_k=1e3, contact_beta=50.0, contact_delta=5e-4,
    contact_mode="tip", contact_s_on=0.0, contact_s_off=0.0,
    energy_scale=1e-8,
):
    s = np.linspace(0.0, float(L), int(N))
    n_seg = N - 1
    K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

    if u0_flat is None:
        u0_flat = 1e-3 * np.random.randn(3 * n_seg)
    else:
        u0_flat = np.asarray(u0_flat, float).reshape(-1)
        if u0_flat.size != 3 * n_seg:
            raise ValueError(
                f"u0_flat has size {u0_flat.size}, expected {3*n_seg} for N={N}"
            )

    u_scale = 30.0
    z0 = u0_flat / u_scale

    lumen_query = None
    if lumen_C is not None and lumen_R is not None:
        lumen_query = LumenQuery(lumen_C, lumen_R)

    def obj(z):
        u_flat = u_scale * np.asarray(z, float)
        W, _ = energy_from_u(
            u_flat,
            p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
            m_src=m_src, r_src=r_src,
            m_local_fun=m_local_fun, m_moment=m_moment,
            wire_len=wire_len,
            include_gravity=False,
            lumen_C=lumen_C, lumen_R=lumen_R,
            lumen_query=lumen_query,
            use_lumen=use_lumen,
            contact_k=contact_k,
            contact_beta=contact_beta,
            contact_delta=contact_delta,
            contact_mode=contact_mode,
            contact_s_on=contact_s_on,
            contact_s_off=contact_s_off,
        )
        return W / energy_scale

    def grad_obj(z):
        u_flat = u_scale * np.asarray(z, float)

        grad_u = energy_gradient_u(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            K_seg=K_seg,
            u_star=u_star,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            lumen_query=lumen_query,
            use_magnetic=True,
            use_contact=use_lumen,
        )

        return (u_scale / energy_scale) * grad_u

    W0_scaled = obj(z0)

    res = minimize(
        obj,
        z0,
        jac=grad_obj,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            ftol=1e-12,
            gtol=1e-10,
            maxls=100,
        ),
    )

    u_flat_opt = u_scale * np.asarray(res.x, float)
    p, q, u_seg = integrate_pq_from_u(u_flat_opt, p0=p0, q0=q0, s=s)

    W, parts = energy_from_u(
        u_flat_opt,
        p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
        m_src=m_src, r_src=r_src,
        m_local_fun=m_local_fun, m_moment=m_moment,
        wire_len=wire_len,
        include_gravity=False,
        lumen_C=lumen_C, lumen_R=lumen_R,
        lumen_query=lumen_query,
        use_lumen=use_lumen,
        contact_k=contact_k,
        contact_beta=contact_beta,
        contact_delta=contact_delta,
        contact_mode=contact_mode,
        contact_s_on=contact_s_on,
        contact_s_off=contact_s_off,
    )

    W0 = W0_scaled * energy_scale

    info = dict(
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit),
        W=float(W),
        parts=parts,
        s=s,
        W0=float(W0),
        dW=float(W - W0),
        u_flat_opt=u_flat_opt.copy(),
        z_opt=np.asarray(res.x, float).copy(),
        grad_norm_scaled=float(np.linalg.norm(grad_obj(res.x))),
        max_bend=float(np.max(np.linalg.norm(u_seg[:, 1:3], axis=1))),
        mean_bend=float(np.mean(np.linalg.norm(u_seg[:, 1:3], axis=1))),
    )

    return p, q, u_seg, info
def energy_from_u(
    u_flat, *, p0, q0, s, K_seg, u_star,
    m_src, r_src, m_local_fun, m_moment, wire_len,
    lumen_C=None, lumen_R=None,
    contact_k=1e3, contact_beta=50.0, contact_delta=5e-4,
    contact_mode="tip", include_gravity=False,
    contact_s_on=0.0, contact_s_off=0.0,
    lumen_query=None, use_lumen=True,
    debug_mag=False, debug_every=1, debug_head=8
):
    """
    Computes total potential energy Π(u) in 3D using:
      - elastic energy
      - distributed magnetic energy only
      - optional gravity
      - optional lumen/contact penalty

    u_flat is full segment strain: shape (3*(N-1),)
    """
    p, q, u_seg = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)
    u_star = np.asarray(u_star, float).reshape(3,)

    # -------------------------
    # Elastic energy
    # -------------------------
    W_el = 0.0
    W_t = 0.0
    W_b = 0.0

    W_t_seg = []
    W_b_seg = []
    W_el_seg = []

    for i in range(len(ds)):
        du = u_seg[i] - u_star
        K = K_seg[i]

        # full elastic contribution
        dW_el = 0.5 * du @ K @ du * ds[i]

        # split twist vs bending
        du_t = du[0:1]          # twist
        du_b = du[1:3]          # bending
        K_tt = K[0:1, 0:1]
        K_bb = K[1:3, 1:3]

        dW_t = 0.5 * du_t @ K_tt @ du_t * ds[i]
        dW_b = 0.5 * du_b @ K_bb @ du_b * ds[i]

        W_el += dW_el
        W_t += float(dW_t)
        W_b += float(dW_b)

        W_el_seg.append(float(dW_el))
        W_t_seg.append(float(dW_t))
        W_b_seg.append(float(dW_b))

    # -------------------------
    # Distributed magnetic energy
    # -------------------------
    # _, _, B = magnetic_wrench_density_cosserat_profile(
    #     p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    # )  # B: (3,N), world frame

    # qn = quat_normalize(q)
    # R_all = quat_to_rot(qn)  # (N,3,3)

    # # body magnetization profile -> world frame
    # m_body = np.asarray(m_local_fun(s, m_moment), float)   # (3,N)
    # m_world = np.einsum('nij,jn->in', R_all, m_body)       # (3,N)
    # dp = np.diff(p, axis=1)                        # (3, N-1)
    # t_mid = dp / (np.linalg.norm(dp, axis=0, keepdims=True) + 1e-12)
    # t_nodes = np.zeros_like(p)
    # t_nodes[:, 1:-1] = 0.5 * (t_mid[:, :-1] + t_mid[:, 1:])
    # t_nodes[:, 0] = t_mid[:, 0]
    # t_nodes[:, -1] = t_mid[:, -1]
    # t_nodes /= (np.linalg.norm(t_nodes, axis=0, keepdims=True) + 1e-12)
    # mnorm = np.linalg.norm(m_world, axis=0) + 1e-12
    # mhat = m_world / mnorm[None, :]
    # cos_axial = np.sum(mhat * t_nodes, axis=0)

    # # print("\n[COSSERAT AXIS CHECK]")
    # # print("mean cos(m,t) =", np.mean(cos_axial))
    # # print("min  cos(m,t) =", np.min(cos_axial))
    # # print("max  cos(m,t) =", np.max(cos_axial))
    # m_dot_B = np.sum(m_world * B, axis=0)
    # w_m = -m_dot_B
    # W_m = np.trapezoid(w_m, s)
    # Bnorm = np.linalg.norm(B, axis=0)
    # mnorm = np.linalg.norm(m_world, axis=0)

    # cos_th = np.sum(m_world * B, axis=0) / (Bnorm * mnorm + 1e-16)
    # cos_th = np.clip(cos_th, -1.0, 1.0)
    # th_deg = np.degrees(np.arccos(cos_th))

    # tau_proxy = np.cross(m_world.T, B.T).T
    # tau_norm = np.linalg.norm(tau_proxy, axis=0)
    # -------------------------
    # Distributed magnetic energy (segment-based)
    # -------------------------
    f_mid, tau_mid, B_mid, m_world_mid, s_mid = magnetic_wrench_density_cosserat_profile_segments(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    # segment lengths
    ds = np.diff(s)   # already computed earlier; reuse if available

    # magnetic energy density at segment midpoints
    m_dot_B_mid = np.sum(m_world_mid * B_mid, axis=0)     # (N-1,)
    w_m_mid = -m_dot_B_mid                                 # (N-1,)

    # midpoint quadrature
    W_m = np.sum(w_m_mid * ds)

    Bnorm = np.linalg.norm(B_mid, axis=0)
    mnorm = np.linalg.norm(m_world_mid, axis=0)

    cos_th = np.sum(m_world_mid * B_mid, axis=0) / (Bnorm * mnorm + 1e-16)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th_deg = np.degrees(np.arccos(cos_th))

    tau_norm = np.linalg.norm(tau_mid, axis=0)
    # -------------------------
    # Gravity
    # -------------------------
    W_g = 0.0
    if include_gravity:
        fg = np.asarray(beam_params.f_g, float).reshape(3,)
        W_g = -np.trapezoid(np.sum(fg[:, None] * p, axis=0), s)

    # -------------------------
    # Contact / lumen
    # -------------------------
    W_cf = 0.0
    min_gap = np.nan

    if use_lumen and (lumen_query is not None):
        C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
            p,
            lumen_query,
            r_beam=beam_params.r,
            k_contact=1e3,
            pen_switch=2e-4,
            k_hard=1e10,
            eps=1e-12,
            window=3,
            smooth=True,
            smooth_eps=1e-5,
        )

        h = float(s[1] - s[0])
        W_cf = h * float(np.sum(C_nodes))
        min_gap = float(np.min(gap_nodes))

    W_total = W_el + W_m + W_g + W_cf
    # print(f"Energy on the beam: Magnetic: {W_m}, Elastic: {W_el}")
    if debug_mag:
        Bnorm = np.linalg.norm(B_mid, axis=0) + 1e-16
        mnorm = np.linalg.norm(m_world_mid, axis=0) + 1e-16
        cos_th = np.sum(m_world_mid * B_mid, axis=0) / (Bnorm * mnorm)
        cos_th = np.clip(cos_th, -1.0, 1.0)
        th_deg = np.degrees(np.arccos(cos_th))
        tau_proxy = np.cross(m_world_mid.T, B_mid.T).T
        tau_norm = np.linalg.norm(tau_proxy, axis=0)

        print("\n[DBG-MAG]")
        print(f"W_s={W_el:.6e}  W_m={W_m:.6e}  W_g={W_g:.6e}  W_cf={W_cf:.6e}")
        print(f"max |B|={np.max(Bnorm):.6e}")
        print(f"max |m|={np.max(mnorm):.6e}")
        print(f"max |m x B|={np.max(tau_norm):.6e}")
        print(f"max angle(m,B) [deg]={np.max(th_deg):.3f}")
        # if use_lumen and (lumen_query is not None):
        #     print(f"min clearance d={min_d:.6e}")

    parts = dict(
        W_el=float(W_el),
        W_b=float(W_b),
        W_t=float(W_t),
        W_m=float(W_m),
        W_g=float(W_g),
        W_cf=float(W_cf),
        gap_min=float(min_gap),
        gap_nodes=np.asarray(gap_nodes, float).copy() if use_lumen and lumen_query is not None else None,
        s=np.asarray(s, float).copy(),
        B=np.asarray(B_mid, float).copy(),
        Bnorm=np.asarray(Bnorm, float).copy(),
        m_world=np.asarray(m_world_mid, float).copy(),
        mnorm=np.asarray(mnorm, float).copy(),
        m_dot_B=np.asarray(m_dot_B_mid, float).copy(),
        w_m=np.asarray(w_m_mid, float).copy(),
        angle_deg=np.asarray(th_deg, float).copy(),
        tau_norm=np.asarray(tau_norm, float).copy(),
        s_mid=np.asarray(s_mid, float).copy(),

        W_el_seg=np.asarray(W_el_seg, float),
        W_b_seg=np.asarray(W_b_seg, float),
        W_t_seg=np.asarray(W_t_seg, float),
    )
    return float(W_total), parts

def tip_len_fun(LL):
    # magnet grows with LL until full physical tip inside
    return min(float(LL), L_tip_full)

def wire_len_fun(LL):
    # wire exists only after full tip is inside
    return max(float(LL) - L_tip_full, 0.0)

def solve_quasistatic_insertion(*,
    p0, q0,
    L0, Lf, dL,
    wire_len_fun,          # function wire_len(L_model)
    Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    lumen_C, lumen_R,
    N=30, maxiter=200,
    u_init=None,
    use_lumen=True,
    tip_len_fun=None,      # OPTIONAL: function tip_len(L_model)
    debug=False
):
    """
    Continuation in model length L (NOT insertion length).
    """
    hist = []
    u0 = u_init

    L = float(L0)
    Lf = float(Lf)
    dL = float(dL)

    while L <= Lf + 1e-12:
        L_model = float(L)  # <-- the loop variable is the model length
        len_wire = float(wire_len_fun(L_model))

        if tip_len_fun is None:
            # If you don't provide one, infer tip length as remainder
            len_tip = max(L_model - len_wire, 0.0)
        else:
            len_tip = float(tip_len_fun(L_model))

        # if debug:
        #     print(f"[CONT] L_model={L_model:.3f} len_wire={len_wire:.3f} len_tip={len_tip:.3f}")

        # IMPORTANT: regenerate m_local_fun per step so tip magnetisation window moves correctly
        m_local_fun_k = make_m_local_fun_wire_tip(len_wire, len_tip=len_tip, mode="axial", eps=1e-3)
        # print("\n[SOLVE DEBUG]")
        # print("r_src =", r_src)
        # print("m_src =", m_src)
        # print("m_local_fun =", m_local_fun)
        # print("m_moment =", m_moment)
        # print("Kinv_fun =", Kinv_fun)
        # s = np.linspace(0.0, L, N)
        # m = m_local_fun(s, None)      # (3,N)
        # mx_int = np.trapezoid(m[0, :], s)
        # my_int = np.trapezoid(m[1, :], s)
        # mz_int = np.trapezoid(m[2, :], s)
        # print(f"DEBUG for magnetisation: {N, mx_int, my_int, mz_int}")
        mesh_schedule = [N]

        for N in mesh_schedule:
            # if u0 is not None and N_prev is not None and N_prev != N:
            #     print(f"\n[INTERP DEBUG] {N_prev} -> {N}")
            #     u_prev_seg = np.asarray(u0, float).reshape(N_prev - 1, 3)
            #     print(
            #         "before interp component rms:",
            #         np.sqrt(np.mean(u_prev_seg[:, 0] ** 2)),
            #         np.sqrt(np.mean(u_prev_seg[:, 1] ** 2)),
            #         np.sqrt(np.mean(u_prev_seg[:, 2] ** 2)),
            #     )

            #     u0 = interp_u_flat_between_meshes(u0, L_model, N_from=N_prev, N_to=N)

            #     u_new_seg = np.asarray(u0, float).reshape(N - 1, 3)
            #     print(
            #         "after interp component rms:",
            #         np.sqrt(np.mean(u_new_seg[:, 0] ** 2)),
            #         np.sqrt(np.mean(u_new_seg[:, 1] ** 2)),
            #         np.sqrt(np.mean(u_new_seg[:, 2] ** 2)),
            #     )

            u0_before = None if u0 is None else u0.copy()
            if N >=200:
                u0_pert = u0.copy()
                u0_pert += 1e-4 * np.random.randn(*u0_pert.shape)
                p, q, u_seg, info = solve_energy_min_3d(
                    p0=p0, q0=q0, L=L_model,
                    wire_len=len_wire, Kinv_fun=Kinv_fun, u_star=u_star,
                    r_src=r_src, m_src=m_src,
                    m_local_fun=m_local_fun_k, m_moment=m_moment,
                    N=N, u0_flat=u0_pert, maxiter=maxiter,
                    lumen_C=lumen_C, lumen_R=lumen_R, use_lumen=use_lumen
                )
            else:
                p, q, u_seg, info = solve_energy_min_3d(
                p0=p0, q0=q0, L=L_model,
                wire_len=len_wire, Kinv_fun=Kinv_fun, u_star=u_star,
                r_src=r_src, m_src=m_src,
                m_local_fun=m_local_fun_k, m_moment=m_moment,
                N=N, u0_flat=u0, maxiter=maxiter,
                lumen_C=lumen_C, lumen_R=lumen_R, use_lumen=use_lumen
                )
            u_opt = info["u_flat_opt"].copy()
            parts = info["parts"]
            # print(f"message={info['message']}")
            du_init = np.nan
            if u0_before is not None and u0_before.size == u_opt.size:
                du_init = np.linalg.norm(u_opt - u0_before)

            # print(
            #     f"N={N:3d}, success={info['success']}, nit={info['nit']:4d}, "
            #     f"W0={info['W0']:.6e}, W={info['W']:.6e}, dW={info['dW']:.6e}, "
            #     f"W_el={parts['W_el']:.6e}, W_m={parts['W_m']:.6e}, W_cf={parts['W_cf']:.6e}, "
            #     f"||u||={np.linalg.norm(u_opt):.6e}, ||u-u_init||={du_init:.6e}"
            # )

            # CRITICAL: carry optimized solution forward
            u0 = u_opt
            N_prev = N

        hist.append(dict(L=L_model, p=p, q=q, info=info, len_wire=len_wire, len_tip=len_tip))
        L += dL

    return hist
def magnetic_energy_only(
    u_flat,
    *,
    p0,
    q0,
    s,
    m_src,
    r_src,
    m_local_fun,
    m_moment,
):
    p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    f_mid, tau_mid, B_mid, m_mid_world, s_mid = magnetic_wrench_density_cosserat_profile_segments(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    ds = np.diff(s)
    return float(np.sum(-np.sum(m_mid_world * B_mid, axis=0) * ds))


def check_magnetic_gradient(
    u_flat,
    *,
    p0,
    q0,
    s,
    m_src,
    r_src,
    m_local_fun,
    m_moment,
    eps=1e-6,
    n_checks=10,
):
    g = magnetic_energy_gradient_u_virtual_work(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
    )

    idxs = np.linspace(0, len(u_flat) - 1, min(n_checks, len(u_flat))).astype(int)

    print("\nMAGNETIC GRADIENT CHECK")
    for k in idxs:
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps
        um[k] -= eps

        Wp = magnetic_energy_only(
            up,
            p0=p0,
            q0=q0,
            s=s,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
        )
        Wm = magnetic_energy_only(
            um,
            p0=p0,
            q0=q0,
            s=s,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
        )

        g_fd = (Wp - Wm) / (2.0 * eps)

        print(
            f"k={k:4d}  analytic={g[k]: .6e}  fd={g_fd: .6e}  "
            f"err={g[k] - g_fd: .3e}"
        )
def check_contact_gradient(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    eps=1e-6,
    n_checks=10,
):
    g = contact_energy_gradient_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
    )

    idxs = np.linspace(0, len(u_flat)-1, min(n_checks, len(u_flat))).astype(int)

    print("\nCONTACT GRADIENT CHECK")
    for k in idxs:
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps
        um[k] -= eps
        K_seg = precompute_K_segments(s, Kinv_fun, wire_len)
        r_src_ur, _ = T_to_p_quat_wxyz(T_ur_mag)
        Wp, _ = energy_from_u(
            up,
            p0=p0, q0=q0, s=s,
            K_seg=K_seg, u_star=np.zeros(3),
            wire_len=wire_len,
            m_src=m_src, r_src=r_src_ur,
            m_local_fun=m_local_fun, m_moment=0.0,
            lumen_query=lumen_query,
            use_lumen=True,
        )

        Wm, _ = energy_from_u(
            um,
            p0=p0, q0=q0, s=s,
            K_seg=K_seg, u_star=np.zeros(3),
            wire_len=wire_len,
            m_src=m_src, r_src=r_src_ur,
            m_local_fun=m_local_fun, m_moment=0.0,
            lumen_query=lumen_query,
            use_lumen=True,
        )

        g_fd = (Wp - Wm) / (2*eps)

        print(
            f"k={k:4d} analytic={g[k]: .6e} fd={g_fd: .6e} "
            f"err={g[k]-g_fd: .3e}"
        )
def energy_min_tip_jacobian_fd_pose_length(
    *,
    u_opt,
    p0,
    q0,
    L,
    r_src,
    q_src,
    m_body,
    Kinv_fun,
    u_star,
    m_moment,
    N,
    eps_pos=1e-5,
    eps_rot=1e-5,
    eps_L=1e-5,
    rotation_convention="world",
    use_magnetic=True,
    use_lumen=False,
    lumen_C=None,
    lumen_R=None,
    maxiter=200,
):
    q_src = quat_normalize(q_src)
    r_src = np.asarray(r_src, float).reshape(3,)

    def solve_at(rp, qp, Lp, u_init):
        wire_len = wire_len_fun(Lp)
        tip_len = tip_len_fun(Lp)

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

        m_src = dipole_from_pose(qp, m_body)

        p, q, u_seg, info = solve_energy_min_3d(
            p0=p0,
            q0=q0,
            L=Lp,
            wire_len=wire_len,
            Kinv_fun=Kinv_fun,
            u_star=u_star,
            r_src=rp,
            m_src=m_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            N=N,
            u0_flat=u_init,
            maxiter=maxiter,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=use_lumen,
        )
        print(
            "FD solve:",
            "L=", Lp,
            "success=", info["success"],
            "nit=", info["nit"],
            "grad_norm_scaled=", info.get("grad_norm_scaled", None),
            "tip=", p[:, -1],
        )
        return p[:, -1], info["u_flat_opt"]

    p0_tip, _ = solve_at(r_src, q_src, L, u_opt)

    J = np.zeros((3, 7), dtype=float)

    for j in range(3):
        rp = r_src.copy()
        rm = r_src.copy()
        rp[j] += eps_pos
        rm[j] -= eps_pos

        pp, _ = solve_at(rp, q_src, L, u_opt)
        pm, _ = solve_at(rm, q_src, L, u_opt)

        J[:, j] = (pp - pm) / (2.0 * eps_pos)

    for j in range(3):
        dphi = np.zeros(3)
        dphi[j] = eps_rot

        qp = perturb_q_src(q_src, dphi, convention=rotation_convention)
        qm = perturb_q_src(q_src, -dphi, convention=rotation_convention)

        pp, _ = solve_at(r_src, qp, L, u_opt)
        pm, _ = solve_at(r_src, qm, L, u_opt)

        J[:, 3 + j] = (pp - pm) / (2.0 * eps_rot)

    pp, _ = solve_at(r_src, q_src, L + eps_L, u_opt)
    pm, _ = solve_at(r_src, q_src, L - eps_L, u_opt)

    J[:, 6] = (pp - pm) / (2.0 * eps_L)

    return J
def make_energy_grad_fun_for_pose(
    *,
    p0,
    q0,
    q_src0,
    m_body,
    Kinv_fun,
    u_star,
    m_moment,
    N,
    rotation_convention="world",
    use_magnetic=True,
    use_contact=False,
    lumen_query=None,
):
    q_src0 = quat_normalize(q_src0)
    r_src0 = None

    def energy_grad_fun(u_flat, theta):
        theta = np.asarray(theta, float).reshape(7,)

        r_src = theta[0:3]
        dphi = theta[3:6]
        L = float(theta[6])

        q_src = perturb_q_src(
            q_src0,
            dphi,
            convention=rotation_convention,
        )
        m_src = dipole_from_pose(q_src, m_body)

        s = np.linspace(0.0, L, int(N))
        wire_len = wire_len_fun(L)
        tip_len = tip_len_fun(L)

        K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

        return energy_gradient_u(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            K_seg=K_seg,
            u_star=u_star,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            use_magnetic=use_magnetic,
            use_contact=use_contact,
            lumen_query=lumen_query,
        )

    return energy_grad_fun
def check_Gtheta_columns(
    *,
    u_opt,
    theta0,
    energy_grad_fun,
    eps_theta=1e-6,
    n_checks=10,
):
    g0 = energy_grad_fun(u_opt, theta0)
    n_u = u_opt.size
    n_th = theta0.size

    print("\nGTHETA CHECK: d/du,dtheta E")

    for j in range(n_th):
        thp = theta0.copy()
        thm = theta0.copy()
        thp[j] += eps_theta
        thm[j] -= eps_theta

        gp = energy_grad_fun(u_opt, thp)
        gm = energy_grad_fun(u_opt, thm)

        Gj = (gp - gm) / (2.0 * eps_theta)

        print(
            f"theta col {j}: "
            f"||G_j||={np.linalg.norm(Gj):.6e}, "
            f"max|G_j|={np.max(np.abs(Gj)):.6e}"
        )
if __name__ == "__main__":
    DEBUG = True
    L_cmd = 0.01
    nodes = 10
    dL = 0.005
    beam_params = default_beam_params()
    mag_params = default_magnet_params()
    mag_len = beam_params.length_of_mag
    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0])
    pivot_point = np.array([
        0.7981328220229531, -0.7112731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.1),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    start_point[2] = -0.1
    # start_point[2] -=0.25
    L_model, wire_len, tip_len = effective_lengths(
        L_cmd,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    m_local_fun = make_m_local_fun_wire_tip(
        wire_len,
        len_tip=tip_len,
        mode="axial",
        alpha_end=0.0,
        eps=1e-3,
    )
    T_ur_pivot = ur_pose6_to_T(pivot_point)   
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)      
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=3e6,
        nu=0.49,
    )
    EA_wire = wire["EA"]
    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]

    EA_tip = tip["EA"]
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


    # base tangent direction
    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
    global INSERTION_DIR_WORLD
    INSERTION_DIR_WORLD = t0 / (np.linalg.norm(t0) + 1e-12)
    # lumen centerline starts at pivot base and bends
    # s_straight = 0.01
    Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
    t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.03,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(90.0),
        bend_start=0.005,
        bend_end=0.03,
    )
    lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.002)
    L_model, wire_len, tip_len = effective_lengths(
    L_cmd,
    L_tip_full=0.04,
    L_tip_min=0.01,
    )

    u_init = None

    m_src = dipole_from_pose(q_src_ur, m_body)
    hist = solve_quasistatic_insertion(
        p0=p0_ur, q0=q0_ur,
        L0=0.010, Lf=L_cmd, dL=dL,
        wire_len_fun=wire_len_fun,
        tip_len_fun=tip_len_fun,
        Kinv_fun=Kinv_fun, u_star=np.zeros(3),
        r_src=r_src_ur, m_src=m_src,
        m_local_fun=m_local_fun, m_moment=0.0,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N=nodes, maxiter=30,
        use_lumen=True,
        u_init=u_init,
        debug=True
    )
    # take final
    pE = hist[-1]["p"]
    qE = hist[-1]["q"]
    info = hist[-1]["info"]
    L_ref = hist[-1]["L"]
    wire_len_ref = hist[-1]["len_wire"]
    tip_len_ref = hist[-1]["len_tip"]
    info = hist[-1]["info"]
    u_ref = info["u_flat_opt"]

    theta0 = np.hstack([r_src_ur, np.zeros(3), L_ref])
    viol = lumen_violation_profile(pE, lumen_C, lumen_R)
    print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
    print("mean positive violation [m] =", np.maximum(viol,0).mean())
    print(info["parts"])
    print("energy-min tip:", pE[:, -1])
    # choose a common comparison grid
    p_energy = pE

    check_magnetic_gradient(
    info["u_flat_opt"],
    p0=p0_ur,
    q0=q0_ur,
    s=info["s"],
    m_src=m_src,
    r_src=r_src_ur,
    m_local_fun=m_local_fun,
    m_moment=0.0,
    )
    lumen_query = LumenQuery(lumen_C, lumen_R)

    check_contact_gradient(
        info["u_flat_opt"],
        p0=p0_ur,
        q0=q0_ur,
        s=info["s"],
        lumen_query=lumen_query,
        eps=1e-6,
        n_checks=10,
    )

    energy_grad_fun = make_energy_grad_fun_for_pose(
        p0=p0_ur,
        q0=q0_ur,
        q_src0=q_src_ur,
        m_body=m_body,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_moment=0.0,
        N=nodes,
        rotation_convention="world",
        use_magnetic=True,
        use_contact=False,
        lumen_query=None,
    )

    def theta_builder(theta):
        if theta is None:
            return {"theta0": theta0}
        theta = np.asarray(theta, float).reshape(7,)
        return {"theta0": theta0, "theta": theta}
    g_ref = energy_grad_fun(u_ref, theta0)
    print("stationarity ||grad E||:", np.linalg.norm(g_ref))
    print("L_ref:", L_ref, "L_cmd:", L_cmd)
    check_Gtheta_columns(
        u_opt=u_ref,
        theta0=theta0,
        energy_grad_fun=energy_grad_fun,
        eps_theta=1e-6,
    )

    J_tip_theta, sens_info = energy_min_tip_jacobian_implicit(
        u_opt=u_ref,
        p0=p0_ur,
        q0=q0_ur,
        L=L_ref,
        wire_len=wire_len_ref,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        r_src=r_src_ur,
        m_src=m_src,
        m_local_fun=make_m_local_fun_wire_tip(
            wire_len_ref,
            len_tip=tip_len_ref,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        ),
        m_moment=0.0,
        N=nodes,
        theta_builder=theta_builder,
        energy_grad_fun=energy_grad_fun,
    )
    print("Energy-min implicit J_tip wrt [dr_src, dphi, L]:")
    print(J_tip_theta)
    print("translation block:")
    print(J_tip_theta[:, 0:3])
    print("rotation block:")
    print(J_tip_theta[:, 3:6])
    print("length column:")
    print(J_tip_theta[:, 6])

    J_fd_energy = energy_min_tip_jacobian_fd_pose_length(
        u_opt=info["u_flat_opt"],
        p0=p0_ur,
        q0=q0_ur,
        L=L_ref,
        r_src=r_src_ur,
        q_src=q_src_ur,
        m_body=m_body,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_moment=0.0,
        N=nodes,
        eps_pos=1e-4,
        eps_rot=1e-4,
        eps_L=1e-5,
        rotation_convention="world",
        use_magnetic=True,
        use_lumen=False,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        maxiter=200,
    )

    print("\nFD energy-min J_tip wrt [dr_src, dphi, L]:")
    print(J_fd_energy)

    diff = J_tip_theta - J_fd_energy

    print("\nAnalytic - FD:")
    print(diff)

    print("\nRelative full error:")
    print(np.linalg.norm(diff) / max(np.linalg.norm(J_fd_energy), 1e-12))

    print("\nRelative translation error:")
    print(
        np.linalg.norm(diff[:, 0:3])
        / max(np.linalg.norm(J_fd_energy[:, 0:3]), 1e-12)
    )

    print("\nRelative rotation error:")
    print(
        np.linalg.norm(diff[:, 3:6])
        / max(np.linalg.norm(J_fd_energy[:, 3:6]), 1e-12)
    )

    print("\nRelative length error:")
    print(
        np.linalg.norm(diff[:, 6])
        / max(np.linalg.norm(J_fd_energy[:, 6]), 1e-12)
    )
    def run_analytic_energy_jacobian():
        J, info_sens = energy_min_tip_jacobian_implicit(
            u_opt=u_ref,
            p0=p0_ur,
            q0=q0_ur,
            L=L_ref,
            wire_len=wire_len_ref,
            Kinv_fun=Kinv_fun,
            u_star=np.zeros(3),
            r_src=r_src_ur,
            m_src=m_src,
            m_local_fun=m_local_fun,
            m_moment=0.0,
            N=nodes,
            theta_builder=theta_builder,
            energy_grad_fun=energy_grad_fun,
            eps_theta=1e-6,
            eps_hess=1e-6,
        )
        return J


    def run_fd_energy_jacobian():
        return energy_min_tip_jacobian_fd_pose_length(
            u_opt=u_ref,
            p0=p0_ur,
            q0=q0_ur,
            L=L_ref,
            r_src=r_src_ur,
            q_src=q_src_ur,
            m_body=m_body,
            Kinv_fun=Kinv_fun,
            u_star=np.zeros(3),
            m_moment=0.0,
            N=nodes,
            eps_pos=1e-5,
            eps_rot=1e-5,
            eps_L=1e-5,
            rotation_convention="world",
            use_magnetic=True,
            use_lumen=False,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            maxiter=200,
        )


    bench_analytic = benchmark_function(
        run_analytic_energy_jacobian,
        repeats=3,
        warmup=1,
        label="Energy-min implicit analytic Jacobian",
    )
    print_benchmark_result(bench_analytic)

    bench_fd = benchmark_function(
        run_fd_energy_jacobian,
        repeats=3,
        warmup=1,
        label="Energy-min full finite-difference Jacobian",
    )
    print_benchmark_result(bench_fd)

    J_an = bench_analytic["last_result"]
    J_fd = bench_fd["last_result"]

    print("\nSpeedup FD / analytic:")
    print(bench_fd["mean_s"] / max(bench_analytic["mean_s"], 1e-12))

    print("\nRelative Jacobian error:")
    print(np.linalg.norm(J_an - J_fd) / max(np.linalg.norm(J_fd), 1e-12))