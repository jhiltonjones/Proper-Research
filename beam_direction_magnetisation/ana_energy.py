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
from beam_direction_magnetisation.cosserat_w_minimal_energy import (quat_exp_body,integrate_pq_from_u, precompute_K_segments, LumenQuery, contact_barrier_energy_and_force_fast,resample_polyline, rod_section_stiffness, 
                                                                    effective_lengths, dipole_from_pose, interpolate_radius, make_Kbt_inv_profile, lumen_violation_profile)
from beam_direction_magnetisation.ana_jaccobian import dq_s_du_analytic,dq_s_dqhat_direct_analytic ,dnormalize_dq,quat_mul,dp_s_dq_analytic,perturb_q_src, benchmark_function, print_benchmark_result
from scipy.integrate import solve_ivp
from pathlib import Path
import json
import csv
import time
import numpy as np
from dataclasses import dataclass
# ============================================================
# Plot style
# ============================================================
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,          # Default text size
    "axes.labelsize": 12,     # X/Y axis label size
    "axes.titlesize": 24,     # Title size
    "xtick.labelsize": 12,    # X tick number size
    "ytick.labelsize": 12,    # Y tick number size
    "legend.fontsize": 16,    # Legend size
})
@dataclass(frozen=True)
class ContactParams:
    r_beam: float = 0.001
    k: float = 1e8
    pen_switch: float = 5e-5
    k_hard: float = 1e10
    smooth: bool = True
    smooth_eps: float = 1e-5
    window: int = 3

beam_params = default_beam_params()
mag_params = default_magnet_params()
L_tip_full = 0.04
def node_quadrature_weights(s):
    """
    Node weights compatible with nodal contact energy.

    For uniform s, this gives approximately h at interior nodes
    and half-weights at endpoints.

    If you want exact consistency with your current W_cf = h * sum(C_nodes),
    use uniform_node_weights_like_current_energy() below instead.
    """
    s = np.asarray(s, float).ravel()
    N = s.size

    if N < 2:
        raise ValueError("s must contain at least two nodes")

    ds = np.diff(s)

    w = np.zeros(N, dtype=float)
    w[0] = 0.5 * ds[0]
    w[-1] = 0.5 * ds[-1]

    if N > 2:
        w[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    return w
def contact_energy_gradient_u_analytic(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    return_debug=False,
    debug_print=False,
):
    contact = contact or ContactParams()

    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()

    p, q, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    W_cf, C_nodes, F_nodes, gap_nodes, w = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )

    if F_nodes.shape != p.shape:
        raise ValueError(f"F_nodes shape {F_nodes.shape} does not match p shape {p.shape}")

    if S_p.shape != (3, p.shape[1], u_flat.size):
        raise ValueError(
            f"Unexpected S_p shape {S_p.shape}; expected {(3, p.shape[1], u_flat.size)}"
        )

    grad = -np.einsum("ij,ijk,j->k", F_nodes, S_p, w)

    if debug_print:
        print("\nCONTACT ANALYTIC GRAD DEBUG")
        print("sum(C_nodes)    =", float(np.sum(C_nodes)))
        print("dot(w,C_nodes)  =", float(np.dot(w, C_nodes)))
        print("W_cf            =", float(W_cf))
        print("min gap         =", float(np.min(gap_nodes)))
        print("active nodes    =", np.flatnonzero(np.linalg.norm(F_nodes, axis=0) > 0.0))
        print("weights         =", w)
        print("C_nodes         =", C_nodes)
        print("gap_nodes       =", gap_nodes)
        print("||F_nodes||     =", np.linalg.norm(F_nodes, axis=0))
        print("||grad||        =", np.linalg.norm(grad))
        print("max|grad|       =", np.max(np.abs(grad)))

    if not return_debug:
        return grad

    debug = dict(
        p=p,
        q=q,
        S_p=S_p,
        S_q=S_q,
        C_nodes=C_nodes,
        F_nodes=F_nodes,
        gap_nodes=gap_nodes,
        weights=w,
        W_cf=float(W_cf),
        W_contact_reconstructed=float(np.dot(w, C_nodes)),
        min_gap=float(np.min(gap_nodes)),
        active_nodes=np.flatnonzero(np.linalg.norm(F_nodes, axis=0) > 0.0),
    )

    return grad, debug
def check_forward_integrator_consistency(u_flat, *, p0, q0, s):
    p_a, q_a, u_seg = integrate_pq_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    p_b, q_b, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    print("\nFORWARD INTEGRATOR CONSISTENCY")
    print("||p_sens - p_plain|| =", np.linalg.norm(p_b - p_a))
    print("max|p_sens - p_plain| =", np.max(np.abs(p_b - p_a)))
    print("plain tip =", p_a[:, -1])
    print("sens  tip =", p_b[:, -1])

    return p_a, p_b
def uniform_node_weights_like_current_energy(s):
    """
    Match current energy_from_u convention:

        W_cf = h * sum(C_nodes)

    where h is the uniform segment spacing.
    """
    s = np.asarray(s, float).ravel()
    if s.size < 2:
        raise ValueError("s must contain at least two nodes")

    ds = np.diff(s)
    if not np.allclose(ds, ds[0], rtol=1e-8, atol=1e-12):
        raise ValueError(
            "uniform_node_weights_like_current_energy assumes uniform s. "
            "Use node_quadrature_weights(s) for nonuniform grids."
        )

    h = float(ds[0])
    return np.full(s.size, h, dtype=float)
def quat_conj(q):
    q = np.asarray(q, float).reshape(4,)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def dtheta_world_from_dq(q, dq):
    """
    Convert quaternion perturbation dq into a small world-frame rotation vector.

    For a small left/world perturbation:
        q_pert ≈ delta_q_world ⊗ q

    therefore:
        delta_q_world ≈ dq ⊗ q^{-1}

    and:
        delta_theta ≈ 2 * vec(delta_q_world)
    """
    q = quat_normalize(np.asarray(q, float).reshape(4,))
    dq = np.asarray(dq, float).reshape(4,)

    dq_world = quat_mul(dq, quat_conj(q))

    return 2.0 * dq_world[1:4]

def numerical_jacobian(f, x, eps=1e-7):
    x = np.asarray(x, float).reshape(-1)
    y0 = np.asarray(f(x), float).reshape(-1)

    J = np.zeros((y0.size, x.size), dtype=float)

    for k in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps

        yp = np.asarray(f(xp), float).reshape(-1)
        ym = np.asarray(f(xm), float).reshape(-1)

        J[:, k] = (yp - ym) / (2.0 * eps)

    return J


def integrate_pq_and_sens_from_u(
    u_flat,
    *,
    p0,
    q0,
    s,
    e1=np.array([-1.0, 0.0, 0.0]),
    eps_q=1e-7,
    eps_u=1e-7,
):
    """
    Discrete sensitivity version of integrate_pq_from_u().

    This intentionally matches the exact same forward update:

        p_{i+1} = p_i + ds_i * R(q_i) e1
        q_{i+1} = normalize(q_i * exp_body(u_i, ds_i))

    Therefore S_p and S_q are sensitivities of the same discrete map used
    by energy_from_u().
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    p0 = np.asarray(p0, float).reshape(3,)
    q0 = quat_normalize(np.asarray(q0, float).reshape(4,))
    e1 = np.asarray(e1, float).reshape(3,)

    N = s.size
    n_seg = N - 1
    n_u = u_flat.size

    if n_u != 3 * n_seg:
        raise ValueError(
            f"Expected u_flat.size = {3*n_seg}, got {n_u}"
        )

    ds = np.diff(s)
    u_seg = u_flat.reshape(n_seg, 3)

    p = np.zeros((3, N), dtype=float)
    q = np.zeros((4, N), dtype=float)

    S_p = np.zeros((3, N, n_u), dtype=float)
    S_q = np.zeros((4, N, n_u), dtype=float)

    p[:, 0] = p0
    q[:, 0] = q0

    for i in range(n_seg):
        h = float(ds[i])
        ui = u_seg[i].copy()

        col0 = 3 * i
        col1 = col0 + 3

        q_i = q[:, i].copy()
        p_i = p[:, i].copy()

        # ------------------------------------------------------------
        # Forward update: exactly match integrate_pq_from_u()
        # ------------------------------------------------------------
        R_i = quat_to_R(q_i)
        tangent_i = R_i @ e1

        p_next = p_i + h * tangent_i

        dq = quat_exp_body(ui, h)
        q_next = quat_normalize(quat_mul(q_i, dq))

        p[:, i + 1] = p_next
        q[:, i + 1] = q_next

        # ------------------------------------------------------------
        # Sensitivity of p_{i+1}
        #
        # p_{i+1} = p_i + h * tangent(q_i)
        #
        # No direct dependence on current ui, because your plain
        # integrator uses q_i, not q_{i+1} or q_mid, for position.
        # ------------------------------------------------------------
        def tangent_from_q(q_raw):
            qn = quat_normalize(q_raw)
            return quat_to_R(qn) @ e1

        d_tangent_dq = numerical_jacobian(
            tangent_from_q,
            q_i,
            eps=eps_q,
        )  # shape (3, 4)

        S_p[:, i + 1, :] = (
            S_p[:, i, :]
            + h * d_tangent_dq @ S_q[:, i, :]
        )

        # ------------------------------------------------------------
        # Sensitivity of q_{i+1}
        #
        # q_{i+1} = normalize(q_i * exp_body(ui, h))
        # ------------------------------------------------------------
        def qnext_from_q(q_raw):
            qn = quat_normalize(q_raw)
            dq_local = quat_exp_body(ui, h)
            return quat_normalize(quat_mul(qn, dq_local))

        def qnext_from_u(u_local):
            dq_local = quat_exp_body(u_local, h)
            return quat_normalize(quat_mul(q_i, dq_local))

        dqnext_dq = numerical_jacobian(
            qnext_from_q,
            q_i,
            eps=eps_q,
        )  # shape (4, 4)

        dqnext_du = numerical_jacobian(
            qnext_from_u,
            ui,
            eps=eps_u,
        )  # shape (4, 3)

        S_q[:, i + 1, :] = dqnext_dq @ S_q[:, i, :]
        S_q[:, i + 1, col0:col1] += dqnext_du

    return p, q, S_p, S_q
def pq_midpoint_sensitivities(p, q, S_p, S_q):
    """
    Build midpoint p, q and analytic sensitivities.

    Parameters
    ----------
    p : (3, N)
    q : (4, N)
    S_p : (3, N, n_u)
    S_q : (4, N, n_u)

    Returns
    -------
    p_mid : (3, N-1)
    q_mid : (4, N-1)
    S_p_mid : (3, N-1, n_u)
    S_theta_mid : (3, N-1, n_u)
        World-frame small-rotation sensitivity of midpoint orientation.
    """
    p = np.asarray(p, float)
    q = quat_normalize(np.asarray(q, float))

    S_p = np.asarray(S_p, float)
    S_q = np.asarray(S_q, float)

    N = p.shape[1]
    n_u = S_p.shape[2]

    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    S_p_mid = 0.5 * (S_p[:, :-1, :] + S_p[:, 1:, :])

    q_mid = np.zeros((4, N - 1), dtype=float)
    S_q_mid = np.zeros((4, N - 1, n_u), dtype=float)

    for i in range(N - 1):
        qa = q[:, i]
        qb = q[:, i + 1]

        Sa = S_q[:, i, :]
        Sb = S_q[:, i + 1, :]

        # Same sign-handling idea as q_mid_average().
        if np.dot(qa, qb) >= 0.0:
            q_sum = qa + qb
            S_sum = Sa + Sb
        else:
            q_sum = qa - qb
            S_sum = Sa - Sb

        q_mid_i = quat_normalize(q_sum)
        J_norm = dnormalize_dq(q_sum)

        q_mid[:, i] = q_mid_i
        S_q_mid[:, i, :] = J_norm @ S_sum

    S_theta_mid = np.zeros((3, N - 1, n_u), dtype=float)

    for i in range(N - 1):
        qmi = q_mid[:, i]

        for k in range(n_u):
            dqmi_du = S_q_mid[:, i, k]
            S_theta_mid[:, i, k] = dtheta_world_from_dq(qmi, dqmi_du)

    return p_mid, q_mid, S_p_mid, S_theta_mid
def magnetic_energy_gradient_u_virtual_work_analytic(
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
    """
    Analytic gradient dW_m / du_flat using magnetic virtual work.

    W_m = - integral m_world · B ds

    Uses:
        dW_m = - integral (f · dp + tau · dtheta) ds

    but computes dp/du and dtheta/du analytically by propagating
    kinematic sensitivities, not by finite differences.

    Returns
    -------
    grad : shape same as u_flat
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)

    # Analytic kinematics and sensitivities wrt all u variables
    p, q, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    # Magnetic wrench density evaluated along the nominal rod
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

    # Midpoint sensitivities
    p_mid, q_mid, S_p_mid, S_theta_mid = pq_midpoint_sensitivities(
        p,
        q,
        S_p,
        S_q,
    )

    # Virtual-work gradient:
    #
    # grad[k] = - sum_i ( f_i · dp_i/du_k + tau_i · dtheta_i/du_k ) ds_i
    grad_force = -np.einsum(
        "ij,ijk,j->k",
        f_mid,
        S_p_mid,
        ds,
    )

    grad_torque = -np.einsum(
        "ij,ijk,j->k",
        tau_mid,
        S_theta_mid,
        ds,
    )

    grad = grad_force + grad_torque

    return grad
def hessian_from_scalar_energy(E_fun, u_ref, eps=1e-5):
    """
    Symmetric central-difference Hessian of scalar energy E(u).

    Cost: O(n_u^2) energy evaluations. Fine for n_u=27 debugging.
    """
    u_ref = np.asarray(u_ref, float).reshape(-1)
    n = u_ref.size

    H = np.zeros((n, n), dtype=float)
    E0 = float(E_fun(u_ref))

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0

        E_p = float(E_fun(u_ref + eps * ei))
        E_m = float(E_fun(u_ref - eps * ei))

        H[i, i] = (E_p - 2.0 * E0 + E_m) / (eps ** 2)

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = 1.0

            E_pp = float(E_fun(u_ref + eps * ei + eps * ej))
            E_pm = float(E_fun(u_ref + eps * ei - eps * ej))
            E_mp = float(E_fun(u_ref - eps * ei + eps * ej))
            E_mm = float(E_fun(u_ref - eps * ei - eps * ej))

            Hij = (E_pp - E_pm - E_mp + E_mm) / (4.0 * eps ** 2)

            H[i, j] = Hij
            H[j, i] = Hij

    return H
def make_energy_fun_for_pose(
    *,
    p0,
    q0,
    q_src0,
    m_body,
    Kinv_fun,
    u_star,
    m_moment,
    N,
    L_tip_full,
    L_tip_min,
    rotation_convention="world",
    use_magnetic=True,
    use_contact=True,
    lumen_query=None,
    contact=None,
):
    contact = contact or ContactParams()
    q_src0 = quat_normalize(q_src0)

    def energy_fun(u_flat, theta):
        theta = np.asarray(theta, float).reshape(7)

        r_src = theta[0:3]
        dphi = theta[3:6]
        L_ins_or_model = float(theta[6])

        L_model, wire_len, tip_len = effective_lengths(
            L_ins_or_model,
            L_tip_full=L_tip_full,
            L_tip_min=L_tip_min,
        )

        q_src = perturb_q_src(
            q_src0,
            dphi,
            convention=rotation_convention,
        )

        m_src = dipole_from_pose(q_src, m_body)

        s = np.linspace(0.0, L_model, int(N))
        K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

        W, parts = energy_from_u(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            K_seg=K_seg,
            u_star=u_star,
            m_src=m_src if use_magnetic else np.zeros(3),
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            lumen_query=lumen_query if use_contact else None,
            use_lumen=use_contact,
            contact=contact,
        )

        if not use_magnetic:
            W -= parts["W_m"]

        return float(W)

    return energy_fun
def sweep_eps_hess_implicit_jacobian(
    *,
    eps_hess_values,
    u_opt,
    p0,
    q0,
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
    energy_fun,
    lumen_query=None,
    contact=None,
    J_fd_ref=None,
    eps_theta=1e-6,
    run_dir=None,
):
    """
    Sweep scalar-Hessian finite-difference step size and record
    Hessian conditioning, definiteness, and Jacobian agreement.

    Parameters
    ----------
    eps_hess_values : iterable of float
        Hessian finite-difference step sizes to test.
    J_fd_ref : optional, shape (3,7)
        Full finite-difference Jacobian reference.
    run_dir : optional pathlib.Path
        If provided, saves CSV/NPZ/JSON output.

    Returns
    -------
    rows : list[dict]
        Scalar diagnostics for each eps_hess.
    sweep_data : dict
        Full matrices for post-processing.
    """
    rows = []
    sweep_data = {
        "eps_hess": [],
        "J": [],
        "H": [],
        "Gtheta": [],
        "du_dtheta": [],
        "P_u": [],
    }

    for eps_hess in eps_hess_values:
        print("\n" + "=" * 70)
        print(f"EPS_HESS SWEEP: eps_hess = {eps_hess:.3e}")
        print("=" * 70)

        J, info_sens = energy_min_tip_jacobian_implicit(
            u_opt=u_opt,
            p0=p0,
            q0=q0,
            L=L,
            wire_len=wire_len,
            Kinv_fun=Kinv_fun,
            u_star=u_star,
            r_src=r_src,
            m_src=m_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            N=N,
            theta_builder=theta_builder,
            energy_grad_fun=energy_grad_fun,
            energy_fun=energy_fun,
            use_scalar_hessian=True,
            lumen_query=lumen_query,
            contact=contact,
            debug_jac=False,
            debug_hessian_terms=False,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
        )

        H = info_sens["H"]
        H_sym = 0.5 * (H + H.T)

        eig = np.linalg.eigvalsh(H_sym)
        H_norm = np.linalg.norm(H)
        H_asym = np.linalg.norm(H - H.T) / max(H_norm, 1e-30)
        H_cond = np.linalg.cond(H_sym + 1e-10 * np.eye(H.shape[0]))

        row = {
            "eps_hess": float(eps_hess),
            "H_norm": float(H_norm),
            "H_asymmetry": float(H_asym),
            "H_cond": float(H_cond),
            "H_eig_min": float(np.min(eig)),
            "H_eig_max": float(np.max(eig)),
            "H_num_negative_eigs": int(np.sum(eig < -1e-12)),
            "J_norm": float(np.linalg.norm(J)),
            "J_length_norm": float(np.linalg.norm(J[:, -1])),
            "J_L_x": float(J[0, -1]),
            "J_L_y": float(J[1, -1]),
            "J_L_z": float(J[2, -1]),
        }

        if J_fd_ref is not None:
            diff = J - J_fd_ref
            row.update({
                "rel_full_error": float(
                    np.linalg.norm(diff)
                    / max(np.linalg.norm(J_fd_ref), 1e-12)
                ),
                "rel_translation_error": float(
                    np.linalg.norm(diff[:, 0:3])
                    / max(np.linalg.norm(J_fd_ref[:, 0:3]), 1e-12)
                ),
                "rel_rotation_error": float(
                    np.linalg.norm(diff[:, 3:6])
                    / max(np.linalg.norm(J_fd_ref[:, 3:6]), 1e-12)
                ),
                "rel_length_error": float(
                    np.linalg.norm(diff[:, 6])
                    / max(np.linalg.norm(J_fd_ref[:, 6]), 1e-12)
                ),
                "diff_norm": float(np.linalg.norm(diff)),
            })

        rows.append(row)

        sweep_data["eps_hess"].append(float(eps_hess))
        sweep_data["J"].append(J.copy())
        sweep_data["H"].append(H.copy())
        sweep_data["Gtheta"].append(info_sens["Gtheta"].copy())
        sweep_data["du_dtheta"].append(info_sens["du_dtheta"].copy())
        sweep_data["P_u"].append(info_sens["P_u"].copy())

        print("H eig min/max =", row["H_eig_min"], row["H_eig_max"])
        print("H cond        =", row["H_cond"])
        print("J length col  =", J[:, -1])

        if J_fd_ref is not None:
            print("rel full err  =", row["rel_full_error"])
            print("rel length err=", row["rel_length_error"])

    # Convert full matrix lists to arrays where possible
    sweep_data["eps_hess"] = np.asarray(sweep_data["eps_hess"], float)
    sweep_data["J"] = np.asarray(sweep_data["J"], float)
    sweep_data["H"] = np.asarray(sweep_data["H"], float)
    sweep_data["Gtheta"] = np.asarray(sweep_data["Gtheta"], float)
    sweep_data["du_dtheta"] = np.asarray(sweep_data["du_dtheta"], float)
    sweep_data["P_u"] = np.asarray(sweep_data["P_u"], float)

    if run_dir is not None:
        import csv

        csv_path = run_dir / "eps_hess_sweep.csv"
        fieldnames = list(rows[0].keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        np.savez(
            run_dir / "eps_hess_sweep_matrices.npz",
            eps_hess=sweep_data["eps_hess"],
            J=sweep_data["J"],
            H=sweep_data["H"],
            Gtheta=sweep_data["Gtheta"],
            du_dtheta=sweep_data["du_dtheta"],
            P_u=sweep_data["P_u"],
            J_fd_ref=J_fd_ref if J_fd_ref is not None else np.array([]),
        )

        save_json(run_dir / "eps_hess_sweep.json", {"rows": rows})

        print("\nSaved eps_hess sweep to:")
        print(" ", csv_path)
        print(" ", run_dir / "eps_hess_sweep_matrices.npz")

    return rows, sweep_data
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

    energy_fun=None,

    use_scalar_hessian=False,

    lumen_query=None,

    contact=None,

    debug_jac=False,

    debug_hessian_terms=False,

    eps_theta=1e-6,

    eps_hess=1e-4,

):

    contact = contact or ContactParams()

    u_opt = np.asarray(u_opt, float).reshape(-1)

    n_u = u_opt.size

    theta0 = np.asarray(theta_builder(None)["theta0"], float).reshape(-1)

    n_theta = theta0.size

    # Rebuild the same nominal discretisation used by the Jacobian.

    s = np.linspace(0.0, float(L), int(N))

    K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

    if debug_hessian_terms:

        hessian_diagnostic_for_grad(
            "elastic only",
            lambda u: elastic_energy_gradient_u(
                u,
                s=s,
                K_seg=K_seg,
                u_star=u_star,
            ),
            u_opt,
            eps=eps_hess,
        )

        hessian_diagnostic_for_grad(
            "magnetic only",
            lambda u: magnetic_energy_gradient_u_virtual_work_analytic(
                u,
                p0=p0,
                q0=q0,
                s=s,
                m_src=m_src,
                r_src=r_src,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
            ),
            u_opt,
            eps=eps_hess,
        )

        check_magnetic_grad_against_energy_fd(
            u_opt,
            p0=p0,
            q0=q0,
            s=s,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            eps_list=(1e-3, 3e-4, 1e-4, 3e-5),
            max_cols=12,
        )

        if lumen_query is not None:
            hessian_diagnostic_for_grad(
                "contact only",
                lambda u: contact_energy_gradient_u_analytic(
                    u,
                    p0=p0,
                    q0=q0,
                    s=s,
                    lumen_query=lumen_query,
                    contact=contact,
                ),
                u_opt,
                eps=eps_hess,
            )

        hessian_diagnostic_for_grad(
            "full energy_grad_fun",
            lambda u: energy_grad_fun(u, theta0),
            u_opt,
            eps=eps_hess,
        )

    # existing Hessian code continues here

    g0 = energy_grad_fun(u_opt, theta0)

    H = np.zeros((n_u, n_u), dtype=float)
    if debug_jac:
        print("\n--- IMPLICIT JAC STATIONARITY ---")
        print("||g0|| =", np.linalg.norm(g0))
        print("max |g0| =", np.max(np.abs(g0)))
        print("mean |g0| =", np.mean(np.abs(g0)))
        print("||u_opt|| =", np.linalg.norm(u_opt))

    if use_scalar_hessian:
        if energy_fun is None:
            raise ValueError("use_scalar_hessian=True requires energy_fun")

        H = hessian_from_scalar_energy(
            lambda uu: energy_fun(uu, theta0),
            u_opt,
            eps=eps_hess,
        )
    else:
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

    if debug_jac:
        H_sym_dbg = 0.5 * (H + H.T)
        print("\n--- HESSIAN DIAGNOSTIC ---")
        print("||H|| =", np.linalg.norm(H))
        print("cond(H_reg) =", np.linalg.cond(H_reg))
        print(
            "min/max eig sym(H) =",
            np.min(np.linalg.eigvalsh(H_sym_dbg)),
            np.max(np.linalg.eigvalsh(H_sym_dbg)),
        )
        print("H asymmetry =", np.linalg.norm(H - H.T) / (np.linalg.norm(H) + 1e-12))
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
    # du_dtheta = -np.linalg.solve(H_reg, Gtheta)
    # H_sym = 0.5 * (H + H.T)
    # H_reg = H_sym + 1e-10 * np.eye(n_u)
    # du_dtheta = -np.linalg.solve(H_reg, Gtheta)
    # -------------------------------------------------
    # du*/dtheta = -H^{-1} Gtheta
    # -------------------------------------------------

    H_raw_reg = H + 1e-10 * np.eye(n_u)

    H_sym = 0.5 * (H + H.T)
    H_sym_reg = H_sym + 1e-10 * np.eye(n_u)

    du_dtheta_raw = -np.linalg.solve(H_raw_reg, Gtheta)
    du_dtheta_sym = -np.linalg.solve(H_sym_reg, Gtheta)

    if debug_jac:
        print("\n--- DU_DTHETA RAW VS SYM ---")
        print("||du_dtheta_raw|| =", np.linalg.norm(du_dtheta_raw))
        print("||du_dtheta_sym|| =", np.linalg.norm(du_dtheta_sym))
        print("||raw - sym|| =", np.linalg.norm(du_dtheta_raw - du_dtheta_sym))

        print("||du_dtheta_raw columns|| =", np.linalg.norm(du_dtheta_raw, axis=0))
        print("||du_dtheta_sym columns|| =", np.linalg.norm(du_dtheta_sym, axis=0))

        print("du_dtheta_raw L norm =", np.linalg.norm(du_dtheta_raw[:, -1]))
        print("du_dtheta_sym L norm =", np.linalg.norm(du_dtheta_sym[:, -1]))

    # Choose which one to use downstream.
    # For now, keep the old behavior:
    du_dtheta = du_dtheta_sym
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

    p_base, q_base, S_p_base, S_q_base = integrate_pq_and_sens_from_u(
        u_opt,
        p0=p0,
        q0=q0,
        s=np.linspace(0.0, float(L), int(N)),
    )

    p_tip_base = p_base[:, -1]
    P_u = S_p_base[:, -1, :]

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
    # P_theta_direct[:, -1] = t_tip
    def tip_from_fixed_u_at_L(L_eval):
        p_eval, q_eval, _ = integrate_pq_from_u(
            u_opt,
            p0=p0,
            q0=q0,
            s=np.linspace(0.0, float(L_eval), int(N)),
        )
        return p_eval[:, -1]

    eps_L_direct = 1e-5
    p_plus = tip_from_fixed_u_at_L(L + eps_L_direct)
    p_minus = tip_from_fixed_u_at_L(L - eps_L_direct)

    P_theta_direct[:, -1] = (p_plus - p_minus) / (2.0 * eps_L_direct)

    if debug_jac:
        print("tangent direct L approx =", t_tip)
        print("FD fixed-u direct L     =", P_theta_direct[:, -1])
    J_implicit_raw = P_u @ du_dtheta_raw
    J_implicit_sym = P_u @ du_dtheta_sym

    J_raw = J_implicit_raw + P_theta_direct
    J_sym = J_implicit_sym + P_theta_direct

    if debug_jac:
        print("\n--- RAW VS SYM IMPLICIT J ---")
        print("J_raw:")
        print(J_raw)
        print("J_sym:")
        print(J_sym)
        print("J_raw - J_sym:")
        print(J_raw - J_sym)

        print("length raw:", J_raw[:, -1])
        print("length sym:", J_sym[:, -1])

    # Choose final Jacobian returned by function.
    J_tip_theta = J_sym
    J_implicit = J_implicit_sym
    J_direct = P_theta_direct

    J_tip_theta = J_implicit + J_direct

    if debug_jac:
        print("\n--- ANALYTIC J L BREAKDOWN ---")
        print("direct L term:")
        print(J_direct[:, -1])
        print("implicit L term:")
        print(J_implicit[:, -1])
        print("total L term:")
        print(J_tip_theta[:, -1])
        print("||Gtheta_L|| =", np.linalg.norm(Gtheta[:, -1]))
        print("||du_dtheta_L|| =", np.linalg.norm(du_dtheta[:, -1]))
        print("||P_u|| =", np.linalg.norm(P_u))

    return J_tip_theta, dict(
        H=H,
        H_sym=H_sym,
        H_reg=H_sym_reg,
        Gtheta=Gtheta,
        du_dtheta=du_dtheta,
        du_dtheta_raw=du_dtheta_raw,
        du_dtheta_sym=du_dtheta_sym,
        P_u=P_u,
        J_direct=J_direct,
        J_implicit=J_implicit,
        J_tip=J_tip_theta,
        J_raw=J_raw,
        J_sym=J_sym,
        J_implicit_raw=J_implicit_raw,
        J_implicit_sym=J_implicit_sym,
        p_tip_base=p_tip_base,
        stationarity_norm=float(np.linalg.norm(g0)),
        H_condition=float(np.linalg.cond(H_sym_reg)),
        Gtheta_norm=float(np.linalg.norm(Gtheta)),
        du_dtheta_norm=float(np.linalg.norm(du_dtheta)),
        P_u_norm=float(np.linalg.norm(P_u)),
        J_direct_norm=float(np.linalg.norm(J_direct)),
        J_implicit_norm=float(np.linalg.norm(J_implicit)),
        J_tip_norm=float(np.linalg.norm(J_tip_theta)),
    )
def hessian_diagnostic_for_grad(name, grad_fun, u_ref, eps=1e-5):
    u_ref = np.asarray(u_ref, float).reshape(-1)
    n = u_ref.size

    g0 = grad_fun(u_ref)
    H = np.zeros((n, n), float)

    for k in range(n):
        up = u_ref.copy()
        um = u_ref.copy()
        up[k] += eps
        um[k] -= eps

        gp = grad_fun(up)
        gm = grad_fun(um)

        H[:, k] = (gp - gm) / (2.0 * eps)

    Hs = 0.5 * (H + H.T)

    print(f"\n--- H DIAG: {name} ---")
    print("||g0|| =", np.linalg.norm(g0))
    print("||H|| =", np.linalg.norm(H))
    print("H asymmetry =", np.linalg.norm(H - H.T) / (np.linalg.norm(H) + 1e-30))
    print("eig sym min/max =", np.min(np.linalg.eigvalsh(Hs)), np.max(np.linalg.eigvalsh(Hs)))
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
def magnetic_energy_gradient_u_virtual_work_fd(
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
def compare_contact_geometry_paths(
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
    wire_len,
    lumen_C,
    lumen_R,
    lumen_query,
    contact,
):
    # Path A: energy_from_u path
    W, parts = energy_from_u(
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
        wire_len=wire_len,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_query=lumen_query,
        use_lumen=True,
        contact=contact,
    )

    # You need energy_from_u to return p in debug.
    # If it does not, temporarily add p to parts/debug.
    p_energy = parts.get("p", None)

    # Path B: analytic sensitivity path
    p_sens, q_sens, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    C_sens, F_sens, gap_sens = contact_barrier_energy_and_force_fast(
        p_sens,
        lumen_query,
        r_beam=contact.r_beam,
        k_contact=contact.k,
        pen_switch=contact.pen_switch,
        k_hard=contact.k_hard,
        eps=1e-12,
        window=contact.window,
        smooth=contact.smooth,
        smooth_eps=contact.smooth_eps,
    )

    w = uniform_node_weights_like_current_energy(s)

    print("\nCOMPARE CONTACT GEOMETRY PATHS")
    print("W_cf energy_from_u:", parts.get("W_cf", None))
    print("W_cf reconstructed from sens p:", float(np.dot(w, C_sens)))
    print("gap_sens min:", float(np.min(gap_sens)))
    print("gap_sens:", gap_sens)
    print("C_sens:", C_sens)

    if p_energy is not None:
        print("||p_energy - p_sens||:", np.linalg.norm(p_energy - p_sens))
        print("max|p_energy - p_sens|:", np.max(np.abs(p_energy - p_sens)))
        print("p_energy tip:", p_energy[:, -1])
        print("p_sens tip:", p_sens[:, -1])
    else:
        print("energy_from_u did not expose p; temporarily add it to parts/debug.")
def contact_energy_gradient_u_fdkin(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    eps_kin=1e-7,
):
    contact = contact or ContactParams()

    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()

    p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=contact.r_beam,
        k_contact=contact.k,
        pen_switch=contact.pen_switch,
        k_hard=contact.k_hard,
        eps=1e-12,
        window=contact.window,
        smooth=contact.smooth,
        smooth_eps=contact.smooth_eps,
    )

    h = float(s[1] - s[0])
    grad = np.zeros_like(u_flat)

    for k in range(u_flat.size):
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps_kin
        um[k] -= eps_kin

        pp, _, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, _, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        dp_du = (pp - pm) / (2.0 * eps_kin)

        dW = 0.0
        for j in range(p.shape[1]):
            dW += -np.dot(F_nodes[:, j], dp_du[:, j])

        grad[k] = h * dW

    return grad
# def contact_energy_gradient_u(
#     u_flat,
#     *,
#     p0,
#     q0,
#     s,
#     lumen_query,
#     contact=None,
#     eps_kin=1e-7,
# ):
#     contact = contact or ContactParams()
#     """
#     Analytic contact gradient via chain rule:

#         dW/du = - sum_j F_j · (dp_j/du)

#     where F_j comes from contact_barrier_energy_and_force_fast.
#     """
#     u_flat = np.asarray(u_flat, float).reshape(-1)
#     s = np.asarray(s, float).ravel()
#     ds = np.diff(s)

#     # base configuration
#     p, q, _ = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

#     # contact forces at nodes
#     C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
#         p,
#         lumen_query,
#         r_beam=contact.r_beam,
#         k_contact=contact.k,
#         pen_switch=contact.pen_switch,
#         k_hard=contact.k_hard,
#         eps=1e-12,
#         window=contact.window,
#         smooth=contact.smooth,
#         smooth_eps=contact.smooth_eps,
#     )

#     min_gap = float(np.min(gap_nodes))
#     j_min = int(np.argmin(gap_nodes))
#     active = np.flatnonzero(gap_nodes < 0.0)

#     # print("min surface gap [mm] =", 1e3 * min_gap)
#     # print("active penetration nodes =", active.size)
#     # print("most penetrating node =", j_min)
#     # print("force at most penetrating node |F| =", np.linalg.norm(F_nodes[:, j_min]))
#     # print("force at most penetrating node =", F_nodes[:, j_min])

#     # if active.size > 0:
#     #     print("[CONTACT DBG] penetrating nodes:")
#     #     for j in active:
#     #         print(
#     #             "j=", int(j),
#     #             "gap[mm]=", 1e3 * gap_nodes[j],
#     #             "|F|=", np.linalg.norm(F_nodes[:, j]),
#     #             "F=", F_nodes[:, j],
#     #             "C=", C_nodes[j],
#     #         )
#     grad = np.zeros_like(u_flat)

#     # finite-difference kinematics only (NOT energy)
#     for k in range(u_flat.size):
#         up = u_flat.copy()
#         um = u_flat.copy()
#         up[k] += eps_kin
#         um[k] -= eps_kin

#         pp, _, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
#         pm, _, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

#         dp_du = (pp - pm) / (2.0 * eps_kin)   # (3,N)

#         # chain rule
#         dW = 0.0
#         for j in range(p.shape[1]):
#             dW += -np.dot(F_nodes[:, j], dp_du[:, j])

#         # consistent scaling with your energy:
#         dW *= (s[1] - s[0])

#         grad[k] = dW

#     return grad
def energy_gradient_u(
    u_flat,
    *,
    p0, q0, s, K_seg, u_star,
    m_src, r_src, m_local_fun, m_moment,
    lumen_query=None,
    use_magnetic=True,
    use_contact=True,
    contact=None,
):
    contact = contact or ContactParams()
    grad = elastic_energy_gradient_u(
        u_flat,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
    )

    if use_magnetic:
        grad += magnetic_energy_gradient_u_virtual_work_analytic(
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
        grad += contact_energy_gradient_u_sens(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
        )

    return grad
def compare_contact_gradient_analytic_vs_fdkin(
    u_ref,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    eps_kin=1e-7,
):
    contact = contact or ContactParams()

    g_fd = contact_energy_gradient_u_fdkin(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        eps_kin=eps_kin,
    )

    g_an, dbg = contact_energy_gradient_u_analytic(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_debug=True,
    )

    rel_err = np.linalg.norm(g_an - g_fd) / max(np.linalg.norm(g_fd), 1e-12)
    cos_sim = np.dot(g_an, g_fd) / (
        np.linalg.norm(g_an) * np.linalg.norm(g_fd) + 1e-30
    )

    print("\n==============================")
    print(" CONTACT GRADIENT ANALYTIC VS FDKIN ")
    print("==============================")
    print("||g_fd|| =", np.linalg.norm(g_fd))
    print("||g_an|| =", np.linalg.norm(g_an))
    print("rel err  =", rel_err)
    print("cos sim  =", cos_sim)
    print("max abs diff =", np.max(np.abs(g_an - g_fd)))
    print("min gap =", dbg["min_gap"])
    print("active nodes =", dbg["active_nodes"])

    idx = np.argsort(np.abs(g_fd))[-10:]

    print("\nLargest FD contact gradient components:")
    for k in idx:
        print(
            f"k={k:3d} "
            f"g_fd={g_fd[k]: .6e} "
            f"g_an={g_an[k]: .6e} "
            f"diff={g_an[k] - g_fd[k]: .6e}"
        )

    return g_an, g_fd, dbg
def compare_contact_gradient_fdkin_vs_sens(
    u_ref,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    eps_kin=1e-7,
):
    contact = contact or ContactParams()

    g_fd = contact_energy_gradient_u_fdkin(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        eps_kin=eps_kin,
    )

    g_sens = contact_energy_gradient_u_sens(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
    )

    abs_err = np.linalg.norm(g_sens - g_fd)
    rel_err = abs_err / max(np.linalg.norm(g_fd), 1e-30)

    print("\nCONTACT GRADIENT COMPARISON")
    print("||g_fd||              =", np.linalg.norm(g_fd))
    print("||g_sens||            =", np.linalg.norm(g_sens))
    print("||g_sens - g_fd||     =", abs_err)
    print("relative error        =", rel_err)
    print("max abs error         =", np.max(np.abs(g_sens - g_fd)))

    return g_fd, g_sens
def contact_energy_only_from_u(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
):
    contact = contact or ContactParams()

    K_dummy = np.zeros((len(s) - 1, 3, 3), dtype=float)

    W, parts = energy_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
        K_seg=K_dummy,
        u_star=np.zeros(3),
        m_src=np.zeros(3),
        r_src=np.zeros(3),
        m_local_fun=lambda ss, mm: np.zeros((3, len(np.asarray(ss).ravel()))),
        m_moment=0.0,
        wire_len=0.0,
        lumen_query=lumen_query,
        use_lumen=True,
        contact=contact,
    )

    return parts["W_cf"]
def check_contact_gradient_against_energy_fd(
    u_ref,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact,
    eps=1e-6,
    max_cols=10,
):
    u_ref = np.asarray(u_ref, float).reshape(-1)

    g_contact = contact_energy_gradient_u_analytic(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
    )

    def W_contact(u):
        p, q, _ = integrate_pq_from_u(u, p0=p0, q0=q0, s=s)

        C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
            p,
            lumen_query,
            r_beam=contact.r_beam,
            k_contact=contact.k,
            pen_switch=contact.pen_switch,
            k_hard=contact.k_hard,
            eps=1e-12,
            window=contact.window,
            smooth=contact.smooth,
            smooth_eps=contact.smooth_eps,
        )

        h = float(s[1] - s[0])
        return h * float(np.sum(C_nodes))

    print("\n==============================")
    print(" CONTACT GRADIENT VS ENERGY FD ")
    print("==============================")
    print("||g_contact|| =", np.linalg.norm(g_contact))

    for k in range(min(max_cols, u_ref.size)):
        up = u_ref.copy()
        um = u_ref.copy()
        up[k] += eps
        um[k] -= eps

        g_fd_k = (W_contact(up) - W_contact(um)) / (2.0 * eps)

        print(
            f"k={k:3d}: "
            f"analytic={g_contact[k]: .6e}, "
            f"energy_fd={g_fd_k: .6e}, "
            f"diff={g_contact[k] - g_fd_k: .6e}"
        )
def solve_energy_min_3d(

    *,

    p0, q0, L, wire_len, Kinv_fun, u_star,

    r_src, m_src, m_local_fun, m_moment,

    N=60,

    u0_flat=None,

    maxiter=300,

    lumen_C=None,

    lumen_R=None,

    use_lumen=True,

    contact=None,
    lumen_query=None,
    energy_scale=1e-8,
    use_fast_contact_grad=False,
):

    contact = contact or ContactParams()

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

    u_scale = 30
    z0 = u0_flat / u_scale

    if lumen_query is None and lumen_C is not None and lumen_R is not None:
        lumen_query = LumenQuery(lumen_C, lumen_R)

    def obj(z):
        u_flat = u_scale * np.asarray(z, float)
        W, _ = energy_from_u(
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
            wire_len=wire_len,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            lumen_query=lumen_query,
            use_lumen=use_lumen,
            contact=contact,
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
            contact=contact,
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
            ftol=1e-22,
            gtol=1e-14,
            maxls=200,
            maxfun=100000,
            maxcor=50,
        ),
    )

    u_flat_opt = u_scale * np.asarray(res.x, float)
    p, q, u_seg = integrate_pq_from_u(u_flat_opt, p0=p0, q0=q0, s=s)

    W, parts = energy_from_u(
        u_flat_opt,
        p0=p0,
        q0=q0,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_query=lumen_query,
        use_lumen=use_lumen,
        contact=contact,
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
def contact_energy_gradient_u_semianalytic_fd_kinematics(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    eps=1e-7,
    debug_print=False,
):
    contact = contact or ContactParams()
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()

    p, q, _ = integrate_pq_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    W_cf, C_nodes, F_nodes, gap_nodes, w = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )

    n_u = u_flat.size
    grad = np.zeros(n_u)

    for k in range(n_u):
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps
        um[k] -= eps

        pp, _, _ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, _, _ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        dp_du_k = (pp - pm) / (2.0 * eps)

        # dW/du_k = - sum_j w_j F_j dot dp_j/du_k
        grad[k] = -np.einsum("ij,ij,j->", F_nodes, dp_du_k, w)

    if debug_print:
        print("\nCONTACT SEMIANALYTIC FD-KIN DEBUG")
        print("W_cf =", W_cf)
        print("min gap =", np.min(gap_nodes))
        print("C_nodes =", C_nodes)
        print("gap_nodes =", gap_nodes)
        print("||grad|| =", np.linalg.norm(grad))
        print("max|grad| =", np.max(np.abs(grad)))

    return grad
def compare_integrator_nodes(u_flat, *, p0, q0, s):
    p_plain, q_plain, _ = integrate_pq_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    p_sens, q_sens, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    dp = p_sens - p_plain

    print("\nNODEWISE FORWARD INTEGRATOR MISMATCH")
    print("j | s[j] | ||dp_j|| | dp_j")
    for j in range(len(s)):
        print(
            f"{j:2d} | {s[j]: .8f} | "
            f"{np.linalg.norm(dp[:, j]): .6e} | "
            f"{dp[:, j]}"
        )

    # Quaternion sign can flip, so compare rotations if possible.
    dq_raw = q_sens - q_plain
    print("\nQuaternion raw mismatch:")
    print("||q_sens - q_plain|| =", np.linalg.norm(dq_raw))
    print("max|q_sens - q_plain| =", np.max(np.abs(dq_raw)))

    return p_plain, p_sens, q_plain, q_sens
def energy_from_u(
    u_flat,

    *,

    p0, q0, s, K_seg, u_star,

    m_src, r_src, m_local_fun, m_moment, wire_len,

    lumen_C=None,

    lumen_R=None,

    lumen_query=None,

    use_lumen=True,

    contact=None,
    debug_mag=False,
    debug_every=1,
    debug_head=8,
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
    # if include_gravity:
    #     fg = np.asarray(beam_params.f_g, float).reshape(3,)
    #     W_g = -np.trapezoid(np.sum(fg[:, None] * p, axis=0), s)

    # -------------------------
    # Contact / lumen
    # -------------------------
# -------------------------
# Contact / lumen
# -------------------------
    # -------------------------
    # Contact / lumen
    # -------------------------
    W_cf = 0.0
    min_gap = np.nan
    gap_nodes = None
    C_nodes = None
    F_nodes = None
    w_contact = None

    if use_lumen:
        if lumen_query is None:
            if lumen_C is None or lumen_R is None:
                raise ValueError(
                    "use_lumen=True requires either lumen_query or both lumen_C and lumen_R."
                )
            lumen_query = LumenQuery(lumen_C, lumen_R)

        W_cf, C_nodes, F_nodes, gap_nodes, w_contact = contact_energy_from_p(
            p,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
            return_force=True,
        )

        min_gap = float(np.min(gap_nodes))
    # if use_lumen:
    #     print("\n[energy_from_u CONTACT DEBUG]")
    #     print("W_cf =", W_cf)
    #     print("sum(C_nodes) =", float(np.sum(C_nodes)))
    #     print("dot(w_contact, C_nodes) =", float(np.dot(w_contact, C_nodes)))
    #     print("min gap =", float(np.min(gap_nodes)))
    #     print("gap_nodes =", gap_nodes)
    #     print("weights =", w_contact)
    #     print("C_nodes =", C_nodes)

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
        gap_nodes=np.asarray(gap_nodes, float).copy() if gap_nodes is not None else None,
        contact_weights=np.asarray(w_contact, float).copy() if w_contact is not None else None,
        C_nodes=np.asarray(C_nodes, float).copy() if C_nodes is not None else None,
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
    parts["p"] = p
    parts["q"] = q
    return float(W_total), parts
def contact_energy_gradient_u_sens(
    u_flat,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    return_debug=False,
):
    contact = contact or ContactParams()

    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()

    # One forward pass that also propagates sensitivities.
    # S_p has shape (3, N, n_u)
    p, q, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    # Contact force at the nominal configuration.
    # This will use your fast contact kernel if you have installed the patch.
    W_cf, C_nodes, F_nodes, gap_nodes, w = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )

    if F_nodes.shape != p.shape:
        raise ValueError(
            f"F_nodes shape {F_nodes.shape} does not match p shape {p.shape}"
        )

    if S_p.shape != (3, p.shape[1], u_flat.size):
        raise ValueError(
            f"S_p shape {S_p.shape} does not match expected "
            f"{(3, p.shape[1], u_flat.size)}"
        )

    # grad[k] = - sum_j w[j] * F[:,j] dot S_p[:,j,k]
    grad = -np.einsum("ij,ijk,j->k", F_nodes, S_p, w)

    if not return_debug:
        return grad

    debug = dict(
        p=p,
        q=q,
        S_p=S_p,
        S_q=S_q,
        W_cf=float(W_cf),
        C_nodes=C_nodes,
        F_nodes=F_nodes,
        gap_nodes=gap_nodes,
        weights=w,
        grad_norm=float(np.linalg.norm(grad)),
        min_gap=float(np.min(gap_nodes)),
        active_nodes=np.flatnonzero(np.linalg.norm(F_nodes, axis=0) > 0.0),
    )

    return grad, debug
def contact_energy_from_p(
    p,
    *,
    s,
    lumen_query,
    contact=None,
    return_force=False,
):
    contact = contact or ContactParams()

    p = np.asarray(p, float)
    s = np.asarray(s, float).ravel()

    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=contact.r_beam,
        k_contact=contact.k,
        pen_switch=contact.pen_switch,
        k_hard=contact.k_hard,
        eps=1e-12,
        window=contact.window,
        smooth=contact.smooth,
        smooth_eps=contact.smooth_eps,
    )

    # This must match energy_from_u exactly.
    w = uniform_node_weights_like_current_energy(s)

    W_cf = float(np.dot(w, C_nodes))

    if not return_force:
        return W_cf

    return W_cf, C_nodes, F_nodes, gap_nodes, w
def check_magnetic_grad_against_energy_fd(
    u_ref,
    *,
    p0,
    q0,
    s,
    m_src,
    r_src,
    m_local_fun,
    m_moment,
    wire_len,
    eps_list=(1e-3, 3e-4, 1e-4, 3e-5),
    max_cols=10,
):
    u_ref = np.asarray(u_ref, float).reshape(-1)

    g_vw = magnetic_energy_gradient_u_virtual_work_analytic(
        u_ref,
        p0=p0,
        q0=q0,
        s=s,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
    )

    def Wmag(u):
        W, parts = energy_from_u(
            u,
            p0=p0,
            q0=q0,
            s=s,
            K_seg=np.zeros((len(s) - 1, 3, 3)),  # ignored if elastic manually zeroed? see note below
            u_star=np.zeros(3),
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            lumen_query=lumen_query,
            use_lumen=False,
        )
        return parts["W_m"]

    print("\n==============================")
    print(" MAGNETIC GRADIENT VS ENERGY FD ")
    print("==============================")
    print("||g_virtual_work|| =", np.linalg.norm(g_vw))

    cols = list(range(min(max_cols, u_ref.size)))

    for eps in eps_list:
        g_fd = np.zeros_like(g_vw)

        for k in cols:
            up = u_ref.copy()
            um = u_ref.copy()
            up[k] += eps
            um[k] -= eps

            g_fd[k] = (Wmag(up) - Wmag(um)) / (2.0 * eps)

        err = g_vw[cols] - g_fd[cols]

        print(f"\neps = {eps}")
        print("||g_fd subset|| =", np.linalg.norm(g_fd[cols]))
        print("||err subset||  =", np.linalg.norm(err))
        print("rel err subset  =", np.linalg.norm(err) / (np.linalg.norm(g_fd[cols]) + 1e-12))

        for k in cols:
            print(
                f"k={k:3d}: "
                f"vw={g_vw[k]: .6e}, "
                f"fd={g_fd[k]: .6e}, "
                f"err={g_vw[k] - g_fd[k]: .6e}"
            )
def tip_len_fun(LL):
    # magnet grows with LL until full physical tip inside
    return min(float(LL), L_tip_full)
def energy_min_tip_jacobian_fd_pose_length_robust(
    *,
    theta0,
    q_src0,
    m_body,
    u_opt,
    p0,
    q0,
    Kinv_fun,
    u_star,
    m_moment,
    N,
    L_tip_full,
    L_tip_min,
    rotation_convention="world",
    use_magnetic=True,
    use_lumen=True,
    lumen_C=None,
    lumen_R=None,
    contact=None,
    maxiter=500,
    eps_pos=1e-4,
    eps_rot=1e-4,
    eps_L=1e-5,
    verbose=True,
):
    """
    Robust finite-difference check for:
        tip(theta) where theta = [r_src, dphi, L]

    Columns:
        0,1,2 = source translation x,y,z [m/m]
        3,4,5 = source rotation x,y,z [m/rad]
        6     = insertion/model length [m/m]

    This intentionally mirrors make_energy_grad_fun_for_pose().
    """

    contact = contact or ContactParams()
    theta0 = np.asarray(theta0, float).reshape(7,)
    q_src0 = quat_normalize(q_src0)

    step = np.array(
        [eps_pos, eps_pos, eps_pos, eps_rot, eps_rot, eps_rot, eps_L],
        dtype=float,
    )

    def unpack_theta(theta):
        theta = np.asarray(theta, float).reshape(7,)
        r_src = theta[0:3].copy()
        dphi = theta[3:6].copy()
        L_ins_or_model = float(theta[6])

        L_model, wire_len, tip_len = effective_lengths(
            L_ins_or_model,
            L_tip_full=L_tip_full,
            L_tip_min=L_tip_min,
        )

        q_src = perturb_q_src(
            q_src0,
            dphi,
            convention=rotation_convention,
        )

        m_src = dipole_from_pose(q_src, m_body)

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

        return r_src, q_src, m_src, L_model, wire_len, tip_len, m_local_fun

    def unscaled_grad_norm(u_flat, theta):
        r_src, q_src, m_src, L_model, wire_len, tip_len, m_local_fun = unpack_theta(theta)

        s = np.linspace(0.0, float(L_model), int(N))
        K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

        lumen_query = None
        if use_lumen and lumen_C is not None and lumen_R is not None:
            lumen_query = LumenQuery(lumen_C, lumen_R)

        g = energy_gradient_u(
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
            use_magnetic=use_magnetic,
            use_contact=use_lumen,
            contact=contact,
        )

        return np.linalg.norm(g), np.max(np.abs(g))

    def solve_theta(theta, u_init, label):
        r_src, q_src, m_src, L_model, wire_len, tip_len, m_local_fun = unpack_theta(theta)

        p, q, u_seg, info = solve_energy_min_3d(
            p0=p0,
            q0=q0,
            L=L_model,
            wire_len=wire_len,
            Kinv_fun=Kinv_fun,
            u_star=u_star,
            r_src=r_src,
            m_src=m_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            N=N,
            u0_flat=u_init,
            maxiter=maxiter,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            use_lumen=use_lumen,
            contact=contact,
        )

        u_sol = info["u_flat_opt"]
        tip = p[:, -1].copy()

        g_norm, g_inf = unscaled_grad_norm(u_sol, theta)
        print("||grad_u E|| =", g_norm)

        if verbose:
            print(
                f"[FD {label}] "
                f"success={info.get('success')} "
                f"nit={info.get('nit')} "
                f"scaled_grad={info.get('grad_norm_scaled', None)} "
                f"||g||={g_norm:.3e} "
                f"||g||_inf={g_inf:.3e} "
                f"tip={tip}"
            )

        return tip, u_sol, info, g_norm, g_inf

    # Re-solve the nominal problem first. Do not blindly trust u_opt.
    p_nom, u_nom, info_nom, g_nom, ginf_nom = solve_theta(
        theta0,
        u_opt,
        "nominal",
    )

    J = np.zeros((3, 7), dtype=float)
    diagnostics = {
        "theta0": theta0.copy(),
        "p_nom": p_nom.copy(),
        "u_nom": u_nom.copy(),
        "nominal_info": info_nom,
        "columns": [],
    }

    for j in range(7):
        h = step[j]

        th_p = theta0.copy()
        th_m = theta0.copy()
        th_p[j] += h
        th_m[j] -= h

        # Important: both sides start from the same nominal equilibrium.
        # This avoids FD asymmetry from continuation history.
        p_p, u_p, info_p, g_p, ginf_p = solve_theta(th_p, u_nom, f"col {j} +")
        p_m, u_m, info_m, g_m, ginf_m = solve_theta(th_m, u_nom, f"col {j} -")

        J[:, j] = (p_p - p_m) / (2.0 * h)

        col_diag = {
            "j": j,
            "h": h,
            "theta_plus": th_p,
            "theta_minus": th_m,
            "p_plus": p_p,
            "p_minus": p_m,
            "dp": p_p - p_m,
            "J_col": J[:, j].copy(),
            "info_plus": info_p,
            "info_minus": info_m,
            "g_norm_plus": g_p,
            "g_norm_minus": g_m,
            "g_inf_plus": ginf_p,
            "g_inf_minus": ginf_m,
            "du_plus_from_nom": np.linalg.norm(u_p - u_nom),
            "du_minus_from_nom": np.linalg.norm(u_m - u_nom),
        }
        diagnostics["columns"].append(col_diag)

        if verbose:
            name = ["x_src", "y_src", "z_src", "rx", "ry", "rz", "L"][j]
            print(f"\nFD column {j} ({name})")
            print("h =", h)
            print("p_plus  =", p_p)
            print("p_minus =", p_m)
            print("dp      =", p_p - p_m)
            print("J_col   =", J[:, j])
            print("||du_plus - u_nom||  =", col_diag["du_plus_from_nom"])
            print("||du_minus - u_nom|| =", col_diag["du_minus_from_nom"])

    return J, diagnostics
def wire_len_fun(LL):
    # wire exists only after full tip is inside
    return max(float(LL) - L_tip_full, 0.0)
def solve_quasistatic_insertion(

    *,

    p0, q0,

    L0, Lf, dL,


    Kinv_fun, u_star,

    r_src, m_src, m_local_fun, m_moment,

    lumen_C, lumen_R,

    N=30,

    maxiter=200,

    u_init=None,

    use_lumen=True,
    lumen_query = None,
    tip_len_fun=None,

    contact=None,

    debug=False,
    use_fast_contact_grad=False,
):

    contact = contact or ContactParams()
    """
    Continuation in model length L (NOT insertion length).
    """
    hist = []
    u0 = u_init

    L0 = float(L0)
    Lf = float(Lf)
    dL = float(dL)

    if dL <= 0:
        raise ValueError("dL must be positive")

    L_values = list(np.arange(L0, Lf, dL))

    # Always include final target length exactly
    if len(L_values) == 0 or abs(L_values[-1] - Lf) > 1e-12:
        L_values.append(Lf)

    for L_model in L_values:
        L_model = float(L_model)

        len_wire = float(wire_len_fun(L_model))

        if tip_len_fun is None:
            len_tip = max(L_model - len_wire, 0.0)
        else:
            len_tip = float(tip_len_fun(L_model))

        m_local_fun_k = make_m_local_fun_wire_tip(
            len_wire,
            len_tip=len_tip,
            mode="axial",
            eps=1e-3,
        )

        mesh_schedule = [N]

        for N_use in mesh_schedule:
            p, q, u_seg, info = solve_energy_min_3d(
                p0=p0,
                q0=q0,
                L=L_model,
                wire_len=len_wire,
                Kinv_fun=Kinv_fun,
                u_star=u_star,
                r_src=r_src,
                m_src=m_src,
                m_local_fun=m_local_fun_k,
                m_moment=m_moment,
                N=N_use,
                u0_flat=u0,
                maxiter=maxiter,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                lumen_query=lumen_query ,
                use_lumen=use_lumen,
                contact=contact,
            
                use_fast_contact_grad=use_fast_contact_grad,
            )

            u0 = info["u_flat_opt"].copy()

        hist.append(dict(
            L=L_model,
            p=p,
            q=q,
            info=info,
            len_wire=len_wire,
            len_tip=len_tip,
        ))

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
    g = magnetic_energy_gradient_u_virtual_work_analytic(
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
    contact=None,
    eps=1e-6,
    n_checks=10,
):
    contact = contact or ContactParams()
    g = contact_energy_gradient_u_analytic(
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
            use_lumen=True, contact=contact
        )

        Wm, _ = energy_from_u(
            um,
            p0=p0, q0=q0, s=s,
            K_seg=K_seg, u_star=np.zeros(3),
            wire_len=wire_len,
            m_src=m_src, r_src=r_src_ur,
            m_local_fun=m_local_fun, m_moment=0.0,
            lumen_query=lumen_query,
            use_lumen=True,contact=contact
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
    contact=None,
    lumen_C=None,
    lumen_R=None,
    maxiter=1000,
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
            contact=contact,
        )

        du_norm = np.linalg.norm(info["u_flat_opt"] - u_init)

        print(
            "FD solve:",
            "L=", Lp,
            "success=", info["success"],
            "nit=", info["nit"],
            "grad_norm_scaled=", info.get("grad_norm_scaled", None),
            "||du||=", du_norm,
            "tip=", p[:, -1],
        )

        return p[:, -1].copy(), info["u_flat_opt"].copy(), info

    # Re-solve nominal once. Use this as the common initial guess.
    p0_tip, u0_fd, info0 = solve_at(r_src, q_src, L, u_opt)

    if info0.get("grad_norm_scaled", np.inf) > 1e-2:
        print("[WARNING] FD base solve not tightly stationary")
        print("grad_norm_scaled =", info0.get("grad_norm_scaled", None))
        print("message =", info0.get("message", None))

    J = np.zeros((3, 7), dtype=float)

    # -------------------------------------------------
    # Translation columns: perturb r_src only
    # -------------------------------------------------
    for j in range(3):
        rp = r_src.copy()
        rm = r_src.copy()

        rp[j] += eps_pos
        rm[j] -= eps_pos

        pp, _, infop = solve_at(rp, q_src, L, u0_fd)
        pm, _, infom = solve_at(rm, q_src, L, u0_fd)

        J[:, j] = (pp - pm) / (2.0 * eps_pos)

        print(f"\nTRANS FD column {j}")
        print("eps_pos =", eps_pos)
        print("p_plus =", pp)
        print("p_minus =", pm)
        print("p_plus - p_minus =", pp - pm)
        print("J_col =", J[:, j])

    # -------------------------------------------------
    # Rotation columns: perturb q_src only
    # -------------------------------------------------
    for j in range(3):
        dphi = np.zeros(3)
        dphi[j] = eps_rot

        qp = perturb_q_src(q_src, dphi, convention=rotation_convention)
        qm = perturb_q_src(q_src, -dphi, convention=rotation_convention)

        mp = dipole_from_pose(qp, m_body)
        mm = dipole_from_pose(qm, m_body)

        print("\nROT FD DEBUG col", j)
        print("eps_rot =", eps_rot)
        print("q_src =", q_src)
        print("qp =", qp)
        print("qm =", qm)
        print("m_nom =", dipole_from_pose(q_src, m_body))
        print("m_plus =", mp)
        print("m_minus =", mm)
        print("||m_plus - m_minus|| =", np.linalg.norm(mp - mm))

        pp, _, infop = solve_at(r_src, qp, L, u0_fd)
        pm, _, infom = solve_at(r_src, qm, L, u0_fd)

        J[:, 3 + j] = (pp - pm) / (2.0 * eps_rot)

        print("p_plus =", pp)
        print("p_minus =", pm)
        print("p_plus - p_minus =", pp - pm)
        print("J_col =", J[:, 3 + j])

    # -------------------------------------------------
    # Length column: perturb L only
    # -------------------------------------------------
    pp, _, infop = solve_at(r_src, q_src, L + eps_L, u0_fd)
    pm, _, infom = solve_at(r_src, q_src, L - eps_L, u0_fd)

    J[:, 6] = (pp - pm) / (2.0 * eps_L)

    print("\nLENGTH FD column")
    print("eps_L =", eps_L)
    print("p_plus =", pp)
    print("p_minus =", pm)
    print("p_plus - p_minus =", pp - pm)
    print("J_col =", J[:, 6])

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
    L_tip_full,
    L_tip_min,
    rotation_convention="world",
    use_magnetic=True,
    use_contact=True,
    lumen_query=None,
    contact=None,
):
    contact = contact or ContactParams()
    q_src0 = quat_normalize(q_src0)

    def energy_grad_fun(u_flat, theta):
        theta = np.asarray(theta, float).reshape(7)

        r_src = theta[0:3]
        dphi = theta[3:6]
        L_ins_or_model = float(theta[6])

        L_model, wire_len, tip_len = effective_lengths(
            L_ins_or_model,
            L_tip_full=L_tip_full,
            L_tip_min=L_tip_min,
        )

        q_src = perturb_q_src(
            q_src0,
            dphi,
            convention=rotation_convention,
        )

        m_src = dipole_from_pose(q_src, m_body)

        s = np.linspace(0.0, L_model, int(N))
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
            lumen_query=lumen_query,
            use_magnetic=use_magnetic,
            use_contact=use_contact,
            contact=contact,
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
def check_contact_force_position_gradient(
    p,
    *,
    lumen_query,
    contact=None,
    node_ids=None,
    eps=1e-7,
):
    contact = contact or ContactParams()
    p = np.asarray(p, float).copy()

    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=contact.r_beam,
        k_contact=contact.k,
        pen_switch=contact.pen_switch,
        k_hard=contact.k_hard,
        eps=1e-12,
        window=contact.window,
        smooth=contact.smooth,
        smooth_eps=contact.smooth_eps,
    )

    if node_ids is None:
        # Check active or near-active nodes first
        node_ids = np.argsort(gap_nodes)[:min(5, p.shape[1])]

    print("\n==============================")
    print(" CONTACT FORCE VS POSITION FD ")
    print("==============================")

    for j in node_ids:
        j = int(j)

        grad_fd = np.zeros(3)

        for a in range(3):
            pp = p.copy()
            pm = p.copy()

            pp[a, j] += eps
            pm[a, j] -= eps

            Cp, _, _ = contact_barrier_energy_and_force_fast(
                pp,
                lumen_query,
                r_beam=contact.r_beam,
                k_contact=contact.k,
                pen_switch=contact.pen_switch,
                k_hard=contact.k_hard,
                eps=1e-12,
                window=contact.window,
                smooth=contact.smooth,
                smooth_eps=contact.smooth_eps,
            )

            Cm, _, _ = contact_barrier_energy_and_force_fast(
                pm,
                lumen_query,
                r_beam=contact.r_beam,
                k_contact=contact.k,
                pen_switch=contact.pen_switch,
                k_hard=contact.k_hard,
                eps=1e-12,
                window=contact.window,
                smooth=contact.smooth,
                smooth_eps=contact.smooth_eps,
            )

            # derivative of nodal contact energy C_j wrt p_j[a]
            grad_fd[a] = (Cp[j] - Cm[j]) / (2.0 * eps)

        print(f"\nnode {j}")
        print("gap =", gap_nodes[j])
        print("C_j =", C_nodes[j])
        print("F_node =", F_nodes[:, j])
        print("grad_fd dC/dp =", grad_fd)
        print("-F_node =", -F_nodes[:, j])
        print("||grad_fd + F|| =", np.linalg.norm(grad_fd + F_nodes[:, j]))
        print("||grad_fd - F|| =", np.linalg.norm(grad_fd - F_nodes[:, j]))
def check_S_p_columns(
    u_flat,
    *,
    p0,
    q0,
    s,
    cols=(0, 1, 2, 5, 8),
    eps=1e-7,
):
    p, q, S_p, S_q = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    for k in cols:
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps
        um[k] -= eps

        pp, _,_ = integrate_pq_from_u(up, p0=p0, q0=q0, s=s)
        pm, _,_ = integrate_pq_from_u(um, p0=p0, q0=q0, s=s)

        S_fd = (pp - pm) / (2.0 * eps)
        S_an = S_p[:, :, k]

        rel = np.linalg.norm(S_an - S_fd) / max(np.linalg.norm(S_fd), 1e-12)

        print(f"S_p col {k}: rel err={rel:.3e}")
        print("  ||S_fd|| =", np.linalg.norm(S_fd))
        print("  ||S_an|| =", np.linalg.norm(S_an))
        print("  max abs diff =", np.max(np.abs(S_an - S_fd)))
def make_run_dir(base="results", name=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = name or f"run_{timestamp}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def sweep_fd_steps_energy_jacobian(
    *,
    step_rows,
    theta0,
    q_src0,
    m_body,
    u_opt,
    p0,
    q0,
    Kinv_fun,
    u_star,
    m_moment,
    N,
    L_tip_full,
    L_tip_min,
    rotation_convention,
    use_magnetic,
    use_lumen,
    lumen_C,
    lumen_R,
    contact,
    maxiter,
    J_an_ref=None,
    run_dir=None,
):
    """
    Sweep finite-difference steps used by the full re-solve FD Jacobian.

    step_rows should be a list of dictionaries, e.g.
        [
            {"eps_pos": 3e-4, "eps_rot": 3e-4, "eps_L": 3e-6},
            {"eps_pos": 1e-3, "eps_rot": 1e-3, "eps_L": 1e-5},
        ]

    Returns
    -------
    rows : list[dict]
        Scalar diagnostics for each FD step set.
    data : dict
        Full matrices for each run.
    """
    rows = []
    data = {}

    for row in step_rows:
        eps_pos = float(row["eps_pos"])
        eps_rot = float(row["eps_rot"])
        eps_L = float(row["eps_L"])

        print("\n" + "=" * 70)
        print(
            "FD STEP SWEEP: "
            f"eps_pos={eps_pos:.3e}, "
            f"eps_rot={eps_rot:.3e}, "
            f"eps_L={eps_L:.3e}"
        )
        print("=" * 70)

        J_fd, fd_info = energy_min_tip_jacobian_fd_pose_length_robust(
            theta0=theta0,
            q_src0=q_src0,
            m_body=m_body,
            u_opt=u_opt,
            p0=p0,
            q0=q0,
            Kinv_fun=Kinv_fun,
            u_star=u_star,
            m_moment=m_moment,
            N=N,
            L_tip_full=L_tip_full,
            L_tip_min=L_tip_min,
            rotation_convention=rotation_convention,
            use_magnetic=use_magnetic,
            use_lumen=use_lumen,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            contact=contact,
            maxiter=maxiter,
            eps_pos=eps_pos,
            eps_rot=eps_rot,
            eps_L=eps_L,
            verbose=False,
        )

        out = {
            "eps_pos": eps_pos,
            "eps_rot": eps_rot,
            "eps_L": eps_L,
            "J_fd_norm": float(np.linalg.norm(J_fd)),
            "J_translation_norm": float(np.linalg.norm(J_fd[:, 0:3])),
            "J_rotation_norm": float(np.linalg.norm(J_fd[:, 3:6])),
            "J_length_norm": float(np.linalg.norm(J_fd[:, 6])),
            "J_L_x": float(J_fd[0, 6]),
            "J_L_y": float(J_fd[1, 6]),
            "J_L_z": float(J_fd[2, 6]),
        }

        if J_an_ref is not None:
            J_diff = J_an_ref - J_fd
            out.update({
                "rel_full_error_vs_analytic": relative_error(J_an_ref, J_fd),
                "rel_translation_error_vs_analytic": relative_error(J_an_ref[:, 0:3], J_fd[:, 0:3]),
                "rel_rotation_error_vs_analytic": relative_error(J_an_ref[:, 3:6], J_fd[:, 3:6]),
                "rel_length_error_vs_analytic": relative_error(J_an_ref[:, 6], J_fd[:, 6]),
                "diff_norm_vs_analytic": float(np.linalg.norm(J_diff)),
            })

        # Compare this FD result to the previous FD result in the sweep.
        if rows:
            prev_key = rows[-1]["key"]
            J_prev = data[prev_key]["J_fd"]
            out.update({
                "rel_change_from_previous_fd": relative_error(J_fd, J_prev),
                "rel_length_change_from_previous_fd": relative_error(J_fd[:, 6], J_prev[:, 6]),
            })
        else:
            out.update({
                "rel_change_from_previous_fd": np.nan,
                "rel_length_change_from_previous_fd": np.nan,
            })

        key = f"pos{eps_pos:.1e}_rot{eps_rot:.1e}_L{eps_L:.1e}"
        out["key"] = key

        rows.append(out)
        data[key] = {
            "J_fd": J_fd,
            "fd_info": fd_info,
        }

        print("J length col =", J_fd[:, 6])
        print("||J_fd||     =", np.linalg.norm(J_fd))
        if J_an_ref is not None:
            print("rel full err =", out["rel_full_error_vs_analytic"])
            print("rel L err    =", out["rel_length_error_vs_analytic"])
        if len(rows) > 1:
            print("rel FD change from previous =", out["rel_change_from_previous_fd"])

    if run_dir is not None:
        # Save scalar sweep table.
        csv_rows = []
        for r in rows:
            csv_rows.append([
                r["eps_pos"],
                r["eps_rot"],
                r["eps_L"],
                r["J_fd_norm"],
                r["J_translation_norm"],
                r["J_rotation_norm"],
                r["J_length_norm"],
                r["J_L_x"],
                r["J_L_y"],
                r["J_L_z"],
                r.get("rel_full_error_vs_analytic", np.nan),
                r.get("rel_translation_error_vs_analytic", np.nan),
                r.get("rel_rotation_error_vs_analytic", np.nan),
                r.get("rel_length_error_vs_analytic", np.nan),
                r.get("rel_change_from_previous_fd", np.nan),
                r.get("rel_length_change_from_previous_fd", np.nan),
            ])

        np.savetxt(
            run_dir / "fd_step_sweep.csv",
            np.asarray(csv_rows, float),
            delimiter=",",
            header=(
                "eps_pos,eps_rot,eps_L,"
                "J_fd_norm,J_translation_norm,J_rotation_norm,J_length_norm,"
                "J_L_x,J_L_y,J_L_z,"
                "rel_full_error_vs_analytic,"
                "rel_translation_error_vs_analytic,"
                "rel_rotation_error_vs_analytic,"
                "rel_length_error_vs_analytic,"
                "rel_change_from_previous_fd,"
                "rel_length_change_from_previous_fd"
            ),
            comments="",
        )

        # Save all full matrices.
        np.savez(
            run_dir / "fd_step_sweep_matrices.npz",
            **{f"{key}_J_fd": val["J_fd"] for key, val in data.items()},
        )

        save_json(run_dir / "fd_step_sweep.json", rows)

    return rows, data
def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float64, np.float32)):
        return float(x)
    if isinstance(x, (np.int64, np.int32)):
        return int(x)
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    return x


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(to_serializable(data), f, indent=2)


def relative_error(A, B, eps=1e-12):
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), eps))
import matplotlib.pyplot as plt


def singular_value_diagnostics(J, name):
    """
    J is expected to be shape (3, 7), i.e. tip position wrt:
        [x_src, y_src, z_src, rx, ry, rz, L]

    Returns useful singular-value diagnostics.
    """
    J = np.asarray(J, float)

    U, svals, Vt = np.linalg.svd(J, full_matrices=False)

    cond = svals[0] / max(svals[-1], 1e-30)

    print(f"\n--- SINGULAR VALUES: {name} ---")
    print("J shape =", J.shape)
    print("singular values =", svals)
    print("condition number =", cond)
    print("rank tol 1e-10 =", np.sum(svals > 1e-10))

    return {
        "name": name,
        "singular_values": svals,
        "condition_number": float(cond),
        "U": U,
        "Vt": Vt,
    }
def scale_jacobian_columns(J, theta_scale):
    J = np.asarray(J, float)
    theta_scale = np.asarray(theta_scale, float).reshape(-1)

    if J.shape[1] != theta_scale.size:
        raise ValueError(
            f"J has {J.shape[1]} columns, but theta_scale has {theta_scale.size} entries"
        )

    return J @ np.diag(theta_scale)


def singular_value_diagnostics_scaled(J, name, theta_scale):
    J_scaled = scale_jacobian_columns(J, theta_scale)

    svals_raw = np.linalg.svd(J, compute_uv=False)
    svals_scaled = np.linalg.svd(J_scaled, compute_uv=False)

    print(f"\n--- SINGULAR VALUES: {name} ---")
    print("Raw singular values:")
    print(svals_raw)
    print("Scaled singular values:")
    print(svals_scaled)
    print("Scaled condition number:")
    print(svals_scaled[0] / max(svals_scaled[-1], 1e-30))

    return {
        "J_raw": J,
        "J_scaled": J_scaled,
        "singular_values_raw": svals_raw,
        "singular_values_scaled": svals_scaled,
        "condition_scaled": float(svals_scaled[0] / max(svals_scaled[-1], 1e-30)),
    }
def plot_jacobian_column_norms(
    J_no_bc,
    J_with_bc,
    *,
    run_dir=None,
    title="Tip Jacobian column norms",
    filename="jacobian_column_norms_with_without_bc.png",
    print_table=True,
    use_log_scale=False,
):
    """
    Compare column norms of tip Jacobians.

    Columns:
        x_src, y_src, z_src, rx, ry, rz, L

    J_no_bc:
        Tip Jacobian without contact-aware modelling, shape (3, 7).

    J_with_bc:
        Tip Jacobian with contact-aware modelling, shape (3, 7).
    """
    labels = ["x", "y", "z", "rx", "ry", "rz", "L"]

    J_no_bc = np.asarray(J_no_bc, float)
    J_with_bc = np.asarray(J_with_bc, float)

    if J_no_bc.shape != (3, 7):
        raise ValueError(f"J_no_bc must have shape (3, 7), got {J_no_bc.shape}.")

    if J_with_bc.shape != (3, 7):
        raise ValueError(f"J_with_bc must have shape (3, 7), got {J_with_bc.shape}.")

    col_no = np.linalg.norm(J_no_bc, axis=0)
    col_bc = np.linalg.norm(J_with_bc, axis=0)

    ratio = col_bc / (col_no + 1e-12)

    if print_table:
        print("\n--- JACOBIAN COLUMN NORMS ---")
        print(f"{'col':>4s} {'no_contact':>14s} {'contact':>14s} {'contact/no':>14s}")
        for lab, a, b, r in zip(labels, col_no, col_bc, ratio):
            print(f"{lab:>4s} {a:14.6e} {b:14.6e} {r:14.6e}")

        print("\nL columns:")
        print("  no_contact J[:, L] =", J_no_bc[:, 6])
        print("  contact    J[:, L] =", J_with_bc[:, 6])
        print("  ||no_contact L||   =", col_no[6])
        print("  ||contact L||      =", col_bc[6])

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7.0, 4.0))
    plt.bar(x - width / 2, col_no, width, label="Without Contact-aware Modelling")
    plt.bar(x + width / 2, col_bc, width, label="With Contact-aware Modelling")
    plt.xticks(x, labels)
    plt.ylabel("Column norm")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    if use_log_scale:
        plt.yscale("log")
        plt.ylabel("Column norm, log scale")

    plt.legend()
    plt.tight_layout()

    if run_dir is not None:
        out_path = run_dir / filename
        plt.savefig(out_path, dpi=300)
        print("Saved column-norm plot to:", out_path)

    plt.show()

    return col_no, col_bc
def print_L_column_decomposition(
    *,
    J_direct,
    J_implicit,
    label="contact",
):
    J_direct = np.asarray(J_direct, float)
    J_implicit = np.asarray(J_implicit, float)

    if J_direct.shape != (3, 7):
        raise ValueError(f"J_direct must have shape (3, 7), got {J_direct.shape}.")

    if J_implicit.shape != (3, 7):
        raise ValueError(
            f"J_implicit must have shape (3, 7), got {J_implicit.shape}."
        )

    d = J_direct[:, 6]
    i = J_implicit[:, 6]
    t = d + i

    cos_di = float(np.dot(d, i) / ((np.linalg.norm(d) * np.linalg.norm(i)) + 1e-12))

    print(f"\n--- L COLUMN DECOMPOSITION: {label} ---")
    print("direct L term:  ", d)
    print("implicit L term:", i)
    print("total L term:   ", t)
    print("||direct||      =", np.linalg.norm(d))
    print("||implicit||    =", np.linalg.norm(i))
    print("||total||       =", np.linalg.norm(t))
    print("cos(direct, implicit) =", cos_di)
def make_benchmark_case():
    DEBUG = False
    lumen_used = True   # IMPORTANT: must be True to test contact

    L_cmd = 0.023
    nodes = 20
    dL = 0.04

    beam_params = default_beam_params()
    mag_params = default_magnet_params()

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0])

    pivot_point_lumen = np.array([
        0.7981328220229531, -0.7212731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    pivot_point = np.array([
        0.7981328220229531, -0.70992731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.12),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)

    start_point = np.asarray(get_point(0, -60, base_point, pivot_point), dtype=float)

    L_model, wire_len, tip_len = effective_lengths(
        L_cmd,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    contact = ContactParams(
        r_beam=0.001,
        k=1e5,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=None,    # IMPORTANT: needed for the fast contact path
    )

    T_ur_pivot_lumen = ur_pose6_to_T(pivot_point_lumen)
    T_ur_pivot = ur_pose6_to_T(pivot_point_lumen)

    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    p0_ur_lumen, q0_ur_lumen = T_to_p_quat_wxyz(T_ur_pivot_lumen)

    T_ur_mag = ur_pose6_to_T(start_point)
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )

    tip = rod_section_stiffness(
        r=beam_params.r,
        E=1e6,
        nu=0.49,
    )

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=wire["EI"],
        EI_tip=tip["EI"],
        GJ_wire=wire["GJ"],
        GJ_tip=tip["GJ"],
        bend_soft=1.0,
        tors_soft=1.0,
    )

    Rbase = Rot.from_quat([
        q0_ur_lumen[1],
        q0_ur_lumen[2],
        q0_ur_lumen[3],
        q0_ur_lumen[0],
    ]).as_matrix()

    t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur_lumen,
        t0=t0,
        length=0.041,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(-60.0),
        bend_start=0.015,
        bend_end=0.021,
    )

    lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.004)
    lumen_query = LumenQuery(lumen_C, lumen_R)

    m_src = dipole_from_pose(q_src_ur, m_body)

    # Test points for contact-only benchmark.
    # This creates a quick cheap geometry benchmark before running the full solve.
    p_test = lumen_C[::max(1, len(lumen_C) // nodes)].T[:, :nodes]
    s_test = np.linspace(0.0, L_model, p_test.shape[1])

    return {
        "lumen_C": lumen_C,
        "lumen_R": lumen_R,
        "lumen_query": lumen_query,
        "contact": contact,
        "p": p_test,
        "s": s_test,

        # Full solve benchmark.
        "solve_kwargs": dict(
            p0=p0_ur,
            q0=q0_ur,
            L0=0.010,
            Lf=L_cmd,
            dL=dL,
            tip_len_fun=tip_len_fun,
            Kinv_fun=Kinv_fun,
            u_star=np.zeros(3),
            r_src=r_src_ur,
            m_src=m_src,
            m_local_fun=None,
            m_moment=0.0,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            N=nodes,
            maxiter=1000,
            use_lumen=True,
            u_init=None,
            contact=contact,
            lumen_query=lumen_query,
            debug=False,
        ),
    }

def plot_scaled_jacobian_singular_values(
    J_no_bc,
    J_with_bc,
    *,
    theta_scale,
    run_dir=None,
    title="Scaled analytical tip Jacobian singular values",
    filename="scaled_jacobian_singular_values_with_without_bc.png",
):
    J_no_bc_scaled = scale_jacobian_columns(J_no_bc, theta_scale)
    J_with_bc_scaled = scale_jacobian_columns(J_with_bc, theta_scale)

    sv_no = np.linalg.svd(J_no_bc_scaled, compute_uv=False)
    sv_bc = np.linalg.svd(J_with_bc_scaled, compute_uv=False)

    k_no = np.arange(1, len(sv_no) + 1)
    k_bc = np.arange(1, len(sv_bc) + 1)

    plt.figure(figsize=(6.0, 4.0))
    plt.semilogy(k_no, sv_no, "o-", label="Without Contact-aware Modelling")
    plt.semilogy(k_bc, sv_bc, "s-", label="With Contact-aware Modelling")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value [m per scaled parameter step]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if run_dir is not None:
        out_path = run_dir / filename
        plt.savefig(out_path, dpi=300)
        print("Saved scaled singular-value plot to:", out_path)

    plt.show()

    return sv_no, sv_bc
import time

def benchmark_contact_gradient_methods(
    u_ref,
    *,
    p0,
    q0,
    s,
    lumen_query,
    contact=None,
    repeats=5,
):
    contact = contact or ContactParams()

    def time_best(fn):
        best = float("inf")
        out = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            best = min(best, dt)
        return best, out

    t_fd, g_fd = time_best(
        lambda: contact_energy_gradient_u_fdkin(
            u_ref,
            p0=p0,
            q0=q0,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
        )
    )

    t_sens, g_sens = time_best(
        lambda: contact_energy_gradient_u_sens(
            u_ref,
            p0=p0,
            q0=q0,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
        )
    )

    abs_err = np.linalg.norm(g_sens - g_fd)
    rel_err = abs_err / max(np.linalg.norm(g_fd), 1e-30)

    print("\nCONTACT GRADIENT BENCHMARK")
    print(f"fdkin best      : {t_fd:.6e} s")
    print(f"sens best       : {t_sens:.6e} s")
    print(f"speedup         : {t_fd / t_sens:.2f}x")
    print(f"relative error  : {rel_err:.6e}")
    print(f"max abs error   : {np.max(np.abs(g_sens - g_fd)):.6e}")

    return g_fd, g_sens
def theta_builder(theta):
    if theta is None:
        return {"theta0": theta0}
    theta = np.asarray(theta, float).reshape(7,)
    return {"theta0": theta0, "theta": theta}
if __name__ == "__main__":
    DEBUG = True
    lumen_used = True

    L_cmd = 0.026
    nodes = 20
    dL = 0.04
    beam_params = default_beam_params()
    mag_params = default_magnet_params()
    mag_len = beam_params.length_of_mag
    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0])
    pivot_point_lumen = np.array([
        0.7981328220229531, -0.7112731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)
    pivot_point = np.array([
        0.7981328220229531, -0.70992731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)
    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.12),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)
    # start_point = np.array([0.753, -0.61, 0.1800201230581867, 3.14, 0.003, 0.001], float)
    # start_point = np.array([
    #     0.7981328220229531, -0.70992731669220016, -0.1+0.15,
    #     np.pi, 0.001, 0.001
    # ], float)
    start_point = np.asarray(get_point(0, -60, base_point, pivot_point), dtype=float)
    # start_point[2] = -0.1
    # start_point[2] -=0.25
    L_model, wire_len, tip_len = effective_lengths(
        L_cmd,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )
    contact = ContactParams(
        r_beam=0.001,
        k=1e5,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=None,
    )
    m_local_fun = make_m_local_fun_wire_tip(
        wire_len,
        len_tip=tip_len,
        mode="axial",
        alpha_end=0.0,
        eps=1e-3,
    )
    T_ur_pivot_lumen = ur_pose6_to_T(pivot_point_lumen)   
    T_ur_pivot = ur_pose6_to_T(pivot_point)  
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    p0_ur_lumen, q0_ur_lumen = T_to_p_quat_wxyz(T_ur_pivot_lumen)
    T_ur_mag = ur_pose6_to_T(start_point)      
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=1e6,
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
    run_dir = make_run_dir(
        base="results",
        name=f"N{nodes}_L{L_cmd:.3f}_lumen{int(lumen_used)}_60_con"
    )
    print("Saving results to:", run_dir)
    config = {
        "L_cmd": L_cmd,
        "nodes": nodes,
        "dL": dL,
        "lumen_used": lumen_used,
        "L_tip_full": 0.04,
        "L_tip_min": 0.01,
        "contact": {
            "r_beam": contact.r_beam,
            "k": contact.k,
            "pen_switch": contact.pen_switch,
            "k_hard": contact.k_hard,
            "smooth": contact.smooth,
            "smooth_eps": contact.smooth_eps,
            "window": contact.window,
        },
        "wire": {
            "r": 200e-6,
            "E": 50e6,
            "nu": 0.4,
            "EA": EA_wire,
            "EI": EI_wire,
            "GJ": GJ_wire,
        },
        "tip": {
            "r": beam_params.r,
            "E": 1e6,
            "nu": 0.49,
            "EA": EA_tip,
            "EI": EI_tip,
            "GJ": GJ_tip,
        },
        "pivot_point": pivot_point,
        "start_point": start_point,
        "r_src_ur": r_src_ur,
        "q_src_ur": q_src_ur,
        "p0_ur": p0_ur,
        "q0_ur": q0_ur,
    }

    save_json(run_dir / "config.json", config)
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
        p_start=p0_ur_lumen,
        t0=t0,
        length=0.041,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(-90.0),
        bend_start=0.015,
        bend_end=0.021,
    )
    lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.004)
    L_model, wire_len, tip_len = effective_lengths(
    L_cmd,
    L_tip_full=0.04,
    L_tip_min=0.01,
    )
    lumen_query = LumenQuery(lumen_C, lumen_R)
    # lumen_query = None
    u_init = None

    m_src = dipole_from_pose(q_src_ur, m_body)
    hist = solve_quasistatic_insertion(
        p0=p0_ur, q0=q0_ur,
        L0=0.010, Lf=L_cmd, dL=dL,
        tip_len_fun=tip_len_fun,
        Kinv_fun=Kinv_fun, u_star=np.zeros(3),
        r_src=r_src_ur, m_src=m_src,
        m_local_fun=m_local_fun, m_moment=0.0,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N=nodes, maxiter=1000,
        use_lumen=lumen_used,
        u_init=u_init,
        contact=contact,
        lumen_query = lumen_query,
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
    s_nodes = info["s"]

    centerline_data = np.column_stack([
        s_nodes,
        pE.T,
        qE.T,
    ])

    np.savetxt(
        run_dir / "centerline.csv",
        centerline_data,
        delimiter=",",
        header="s,px,py,pz,qw,qx,qy,qz",
        comments="",
    )
    np.savetxt(
    run_dir / "lumen_centerline.csv",
    lumen_C,
    delimiter=",",
    header="x,y,z",
    comments="",
    )

    np.savetxt(
        run_dir / "lumen_radius.csv",
        lumen_R.reshape(-1, 1),
        delimiter=",",
        header="radius",
        comments="",
    )
    theta0 = np.hstack([r_src_ur, np.zeros(3), L_ref])
    viol = lumen_violation_profile(pE, lumen_C, lumen_R)
    print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
    print("mean positive violation [m] =", np.maximum(viol,0).mean())
    print(info["parts"])
    print("energy-min tip:", pE[:, -1])
    parts = info["parts"]

    # forward_metrics = {
    #     "tip_position": pE[:, -1],
    #     "L_ref": L_ref,
    #     "wire_len_ref": wire_len_ref,
    #     "tip_len_ref": tip_len_ref,
    #     "success": info.get("success", None),
    #     "message": info.get("message", None),
    #     "nit": info.get("nit", None),
    #     "W_total": info.get("W", None),
    #     "W0": info.get("W0", None),
    #     "dW": info.get("dW", None),
    #     "W_el": parts.get("W_el", None),
    #     "W_b": parts.get("W_b", None),
    #     "W_t": parts.get("W_t", None),
    #     "W_m": parts.get("W_m", None),
    #     "W_cf": parts.get("W_cf", None),
    #     "W_g": parts.get("W_g", None),
    #     "gap_min": parts.get("gap_min", None),
    #     "max_lumen_violation": float(np.max(viol)),
    #     "mean_positive_violation": float(np.maximum(viol, 0).mean()),
    #     "max_bend": info.get("max_bend", None),
    #     "mean_bend": info.get("mean_bend", None),
    #     "grad_norm_scaled": info.get("grad_norm_scaled", None),
    # }

    # save_json(run_dir / "forward_metrics.json", forward_metrics)

    # energy_terms = np.array([[
    #     parts.get("W_el", np.nan),
    #     parts.get("W_b", np.nan),
    #     parts.get("W_t", np.nan),
    #     parts.get("W_m", np.nan),
    #     parts.get("W_cf", np.nan),
    #     parts.get("W_g", np.nan),
    #     info.get("W", np.nan),
    # ]])

    # np.savetxt(
    #     run_dir / "energy_terms.csv",
    #     energy_terms,
    #     delimiter=",",
    #     header="W_el,W_b,W_t,W_m,W_cf,W_g,W_total",
    #     comments="",
    # )

    # # choose a common comparison grid
    p_energy = pE
    s_ref = np.linspace(0.0, float(L_ref), int(nodes))

    # g_fd = magnetic_energy_gradient_u_virtual_work_fd(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=s_ref,
    #     m_src=m_src,
    #     r_src=r_src_ur,
    #     m_local_fun=m_local_fun,
    #     m_moment=0.0,
    #     eps_kin=1e-7,
    # )

    # g_an = magnetic_energy_gradient_u_virtual_work_analytic(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=s_ref,
    #     m_src=m_src,
    #     r_src=r_src_ur,
    #     m_local_fun=m_local_fun,
    #     m_moment=0.0,
    # )
    s_cmp = info["s"]
    # rel_err = np.linalg.norm(g_an - g_fd) / max(np.linalg.norm(g_fd), 1e-12)
    # mag_grad_metrics = {
    #     "relative_error": rel_err,
    #     "norm_fd": float(np.linalg.norm(g_fd)),
    #     "norm_analytic": float(np.linalg.norm(g_an)),
    #     "max_abs_diff": float(np.max(np.abs(g_an - g_fd))),
    # }

    # save_json(run_dir / "magnetic_gradient_metrics.json", mag_grad_metrics)

    # np.savez(
    #     run_dir / "magnetic_gradient_validation.npz",
    #     g_fd=g_fd,
    #     g_an=g_an,
    #     diff=g_an - g_fd,
    # )
    # print("magnetic gradient analytic-vs-FD relative error:", rel_err)
    # print("||g_fd|| =", np.linalg.norm(g_fd))
    # print("||g_an|| =", np.linalg.norm(g_an))
    # print("max abs diff =", np.max(np.abs(g_an - g_fd)))
    p_straight = p0_ur.reshape(3,1) + t0.reshape(3,1) * s_cmp.reshape(1,-1)
    # print("external magnet position r_src_ur:", r_src_ur)
    # print("external magnet pose start_point:", start_point)
    hist_2 = solve_quasistatic_insertion(
        p0=p0_ur, q0=q0_ur,
        L0=0.010, Lf=L_cmd, dL=dL,
        tip_len_fun=tip_len_fun,
        Kinv_fun=Kinv_fun, u_star=np.zeros(3),
        r_src=r_src_ur, m_src=m_src,
        m_local_fun=m_local_fun, m_moment=0.0,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N=nodes, maxiter=1000,
        use_lumen=False,
        u_init=u_init,
        contact=contact,
        lumen_query = lumen_query,
        debug=True
    )
    # take final
    pE_2 = hist_2[-1]["p"]
    qE_2 = hist_2[-1]["q"]
    # info_2 = hist_2[-1]["info"]
    # L_ref_2 = hist_2[-1]["L"]
    # wire_len_ref_2 = hist_2[-1]["len_wire"]
    # tip_len_ref_2 = hist_2[-1]["len_tip"]
    # info_2 = info_2[-1]["info"]
    # u_ref_2 = info_2["u_flat_opt"]
    # s_nodes_2 = info_2["s"]
    p_bvp = pE_2

    plot_centerlines_with_lumen_3d(
        p_bvp=p_energy,
        p_energy=p_energy,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        p0=p0_ur,
        p_straight=p_straight,
        r_src=r_src_ur,
        q_src=q_src_ur,
        contact=contact,
        show_beam_tube=True,
        title="Energy-min beam with physical diameter",
    )
    # check_forward_integrator_consistency(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=info["s"],
    # )
    # compare_integrator_nodes(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=info["s"],
    # )
    # check_magnetic_gradient(
    # info["u_flat_opt"],
    # p0=p0_ur,
    # q0=q0_ur,
    # s=info["s"],
    # m_src=m_src,
    # r_src=r_src_ur,
    # m_local_fun=m_local_fun,
    # m_moment=0.0,
    # )


    # check_contact_gradient(
    #     info["u_flat_opt"],
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=info["s"],
    #     lumen_query=lumen_query,
    #     contact=contact,
    #     eps=1e-6,
    #     n_checks=10,
    # )

    # # ------------------------------------------------------------
    # # Analytical Jacobian WITHOUT lumen/contact boundary condition
    # # ------------------------------------------------------------
    energy_grad_fun_no_bc = make_energy_grad_fun_for_pose(
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
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    energy_fun_no_bc = make_energy_fun_for_pose(
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
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    J_no_bc, sens_no_bc = energy_min_tip_jacobian_implicit(
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
        lumen_query=None,
        contact=contact,
        theta_builder=theta_builder,
        energy_grad_fun=energy_grad_fun_no_bc,
        energy_fun=energy_fun_no_bc,
        use_scalar_hessian=True,
        debug_jac=True,
        debug_hessian_terms=False,
        eps_theta=1e-6,
        eps_hess=1e-4,
    )


    # # ------------------------------------------------------------
    # # Analytical Jacobian WITH lumen/contact boundary condition
    # # ------------------------------------------------------------
    energy_grad_fun_with_bc = make_energy_grad_fun_for_pose(
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
        use_contact=True,
        lumen_query=lumen_query,
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    energy_fun_with_bc = make_energy_fun_for_pose(
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
        use_contact=True,
        lumen_query=lumen_query,
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    J_with_bc, sens_with_bc = energy_min_tip_jacobian_implicit(
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
        lumen_query=lumen_query,
        contact=contact,
        theta_builder=theta_builder,
        energy_grad_fun=energy_grad_fun_with_bc,
        energy_fun=energy_fun_with_bc,
        use_scalar_hessian=True,
        debug_jac=True,
        debug_hessian_terms=False,
        eps_theta=1e-6,
        eps_hess=1e-4,
    )
    sv_no_bc_info = singular_value_diagnostics(J_no_bc, "analytic no boundary/contact")
    sv_with_bc_info = singular_value_diagnostics(J_with_bc, "analytic with boundary/contact")
    theta_scale = np.array([
        1e-3, 1e-3, 1e-3,      # x_src, y_src, z_src: 1 mm
        1e-2, 1e-2, 1e-2,      # rx, ry, rz: 0.01 rad
        1e-3,                  # L: 1 mm
    ])
    diag_no_bc = singular_value_diagnostics_scaled(
        J_no_bc,
        "Analytic No Contact-aware Modelling",
        theta_scale,
    )

    diag_with_bc = singular_value_diagnostics_scaled(
        J_with_bc,
        "Analytic with Contact-aware Modelling",
        theta_scale,
    )

    sv_no_bc = diag_no_bc["singular_values_scaled"]
    sv_with_bc = diag_with_bc["singular_values_scaled"]

    col_no_bc, col_with_bc = plot_jacobian_column_norms(
        J_no_bc,
        J_with_bc,
        run_dir=run_dir,
        title="Analytical tip Jacobian column norms"
    )
    def sens_get(sens, key):
        if isinstance(sens, dict):
            return sens[key]
        return getattr(sens, key)
    print_L_column_decomposition(
        J_direct=sens_get(sens_with_bc, "J_direct"),
        J_implicit=sens_get(sens_with_bc, "J_implicit"),
        label="contact",
    )

    print_L_column_decomposition(
        J_direct=sens_get(sens_no_bc, "J_direct"),
        J_implicit=sens_get(sens_no_bc, "J_implicit"),
        label="no contact",
    )

    sv_no_bc_scaled, sv_with_bc_scaled = plot_scaled_jacobian_singular_values(
        J_no_bc,
        J_with_bc,
        theta_scale=theta_scale,
        run_dir=run_dir,
    )
    np.savez(
        run_dir / "jacobian_singular_values_with_without_bc.npz",
        J_no_bc=J_no_bc,
        J_with_bc=J_with_bc,
        singular_values_no_bc=sv_no_bc,
        singular_values_with_bc=sv_with_bc,
        column_norms_no_bc=col_no_bc,
        column_norms_with_bc=col_with_bc,
        H_no_bc=sens_no_bc["H"],
        H_with_bc=sens_with_bc["H"],
        Gtheta_no_bc=sens_no_bc["Gtheta"],
        Gtheta_with_bc=sens_with_bc["Gtheta"],
        du_dtheta_no_bc=sens_no_bc["du_dtheta"],
        du_dtheta_with_bc=sens_with_bc["du_dtheta"],
        P_u_no_bc=sens_no_bc["P_u"],
        P_u_with_bc=sens_with_bc["P_u"],
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
        use_contact=lumen_used,
        lumen_query=lumen_query if lumen_used else None,
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )
    energy_fun = make_energy_fun_for_pose(
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
        use_contact=lumen_used,
        lumen_query=lumen_query if lumen_used else None,
        contact=contact,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )
    def theta_builder(theta):
        if theta is None:
            return {"theta0": theta0}
        theta = np.asarray(theta, float).reshape(7,)
        return {"theta0": theta0, "theta": theta}
    g_ref = energy_grad_fun(u_ref, theta0)
    print("stationarity ||grad E||:", np.linalg.norm(g_ref))
    print("L_ref:", L_ref, "L_cmd:", L_cmd)
    stationarity_metrics = {
        "grad_norm": float(np.linalg.norm(g_ref)),
        "grad_inf": float(np.max(np.abs(g_ref))),
        "grad_mean_abs": float(np.mean(np.abs(g_ref))),
    }

    save_json(run_dir / "stationarity_metrics.json", stationarity_metrics)
    np.save(run_dir / "g_ref.npy", g_ref)
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
        lumen_query=lumen_query if lumen_used else None,
        contact=contact,
        theta_builder=theta_builder,
        energy_grad_fun=energy_grad_fun,
        energy_fun=energy_fun,
        use_scalar_hessian=True,
        debug_jac=True,
        debug_hessian_terms=False,
        eps_theta=1e-6,
        eps_hess=1e-4,
    )
    print("Energy-min implicit J_tip wrt [dr_src, dphi, L]:")
    print(J_tip_theta)
    print("translation block:")
    print(J_tip_theta[:, 0:3])
    print("rotation block:")
    print(J_tip_theta[:, 3:6])
    print("length column:")
    print(J_tip_theta[:, 6])

    # J_fd_energy, fd_info = energy_min_tip_jacobian_fd_pose_length_robust(
    #     theta0=theta0,
    #     q_src0=q_src_ur,
    #     m_body=m_body,
    #     u_opt=u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     Kinv_fun=Kinv_fun,
    #     u_star=np.zeros(3),
    #     m_moment=0.0,
    #     N=nodes,
    #     L_tip_full=0.04,
    #     L_tip_min=0.01,
    #     rotation_convention="world",
    #     use_magnetic=True,
    #     use_lumen=lumen_used,
    #     lumen_C=lumen_C,
    #     lumen_R=lumen_R,
    #     contact=contact,
    #     maxiter=800,

    #     # For contact, start smaller than 1e-3.
    #     eps_pos=1e-3,
    #     eps_rot=1e-3,
    #     eps_L=1e-5,

    #     verbose=True,
    # )

    # print("\nFD energy-min J_tip wrt [dr_src, dphi, L]:")
    # print(J_fd_energy)

    # diff = J_tip_theta - J_fd_energy

    # print("\nAnalytic - FD:")
    # print(diff)

    # print("\nRelative full error:")
    # print(np.linalg.norm(diff) / max(np.linalg.norm(J_fd_energy), 1e-12))

    # print("\nRelative translation error:")
    # print(
    #     np.linalg.norm(diff[:, 0:3])
    #     / max(np.linalg.norm(J_fd_energy[:, 0:3]), 1e-12)
    # )

    # print("\nRelative rotation error:")
    # print(
    #     np.linalg.norm(diff[:, 3:6])
    #     / max(np.linalg.norm(J_fd_energy[:, 3:6]), 1e-12)
    # )

    # print("\nRelative length error:")
    # print(
    #     np.linalg.norm(diff[:, 6])
    #     / max(np.linalg.norm(J_fd_energy[:, 6]), 1e-12)
    # )
    # J_an = J_tip_theta
    # J_fd = J_fd_energy
    # J_diff = J_an - J_fd
    # fd_step_rows = [
    #     {"eps_pos": 1e-4, "eps_rot": 1e-4, "eps_L": 1e-6},
    #     {"eps_pos": 3e-4, "eps_rot": 3e-4, "eps_L": 3e-6},
    #     {"eps_pos": 1e-3, "eps_rot": 1e-3, "eps_L": 1e-5},
    #     {"eps_pos": 3e-3, "eps_rot": 3e-3, "eps_L": 3e-5},
    #     {"eps_pos": 1e-2, "eps_rot": 1e-2, "eps_L": 1e-4},
    # ]

    # # fd_step_sweep_rows, fd_step_sweep_data = sweep_fd_steps_energy_jacobian(
    # #     step_rows=fd_step_rows,
    # #     theta0=theta0,
    # #     q_src0=q_src_ur,
    # #     m_body=m_body,
    # #     u_opt=u_ref,
    # #     p0=p0_ur,
    # #     q0=q0_ur,
    # #     Kinv_fun=Kinv_fun,
    # #     u_star=np.zeros(3),
    # #     m_moment=0.0,
    # #     N=nodes,
    # #     L_tip_full=0.04,
    # #     L_tip_min=0.01,
    # #     rotation_convention="world",
    # #     use_magnetic=True,
    # #     use_lumen=lumen_used,
    # #     lumen_C=lumen_C,
    # #     lumen_R=lumen_R,
    # #     contact=contact,
    # #     maxiter=800,
    # #     J_an_ref=J_an,
    # #     run_dir=run_dir,
    # # )
    # jacobian_metrics = {
    #     "relative_full_error": relative_error(J_an, J_fd),
    #     "relative_translation_error": relative_error(J_an[:, 0:3], J_fd[:, 0:3]),
    #     "relative_rotation_error": relative_error(J_an[:, 3:6], J_fd[:, 3:6]),
    #     "relative_length_error": relative_error(J_an[:, 6], J_fd[:, 6]),
    #     "norm_J_an": float(np.linalg.norm(J_an)),
    #     "norm_J_fd": float(np.linalg.norm(J_fd)),
    #     "norm_diff": float(np.linalg.norm(J_diff)),
    #     "norm_translation_fd": float(np.linalg.norm(J_fd[:, 0:3])),
    #     "norm_rotation_fd": float(np.linalg.norm(J_fd[:, 3:6])),
    #     "norm_length_fd": float(np.linalg.norm(J_fd[:, 6])),
    # }

    # save_json(run_dir / "jacobian_metrics.json", jacobian_metrics)
    # eps_hess_values = [
    #     3e-5,
    #     1e-4,
    #     2e-4,
    #     3e-4,
    #     1e-3,
    # ]

    # # eps_hess_rows, eps_hess_data = sweep_eps_hess_implicit_jacobian(
    # #     eps_hess_values=eps_hess_values,
    # #     u_opt=u_ref,
    # #     p0=p0_ur,
    # #     q0=q0_ur,
    # #     L=L_ref,
    # #     wire_len=wire_len_ref,
    # #     Kinv_fun=Kinv_fun,
    # #     u_star=np.zeros(3),
    # #     r_src=r_src_ur,
    # #     m_src=m_src,
    # #     m_local_fun=make_m_local_fun_wire_tip(
    # #         wire_len_ref,
    # #         len_tip=tip_len_ref,
    # #         mode="axial",
    # #         alpha_end=0.0,
    # #         eps=1e-3,
    # #     ),
    # #     m_moment=0.0,
    # #     N=nodes,
    # #     theta_builder=theta_builder,
    # #     energy_grad_fun=energy_grad_fun,
    # #     energy_fun=energy_fun,
    # #     lumen_query=lumen_query if lumen_used else None,
    # #     contact=contact,
    # #     J_fd_ref=J_fd_energy,
    # #     eps_theta=1e-6,
    # #     run_dir=run_dir,
    # # )
    # H = sens_info["H"]
    # H_sym = 0.5 * (H + H.T)

    # hessian_metrics = {
    #     "H_norm": float(np.linalg.norm(H)),
    #     "H_asymmetry": float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-30)),
    #     "H_condition": float(np.linalg.cond(H + 1e-10 * np.eye(H.shape[0]))),
    #     "H_eig_min": float(np.min(np.linalg.eigvalsh(H_sym))),
    #     "H_eig_max": float(np.max(np.linalg.eigvalsh(H_sym))),
    #     "use_scalar_hessian": True,
    #     "eps_hess": 1e-5,
    #     "eps_theta": 1e-6,
    # }

    # save_json(run_dir / "hessian_metrics.json", hessian_metrics)

    # np.savez(
    #     run_dir / "jacobians.npz",
    #     J_an=J_an,
    #     J_fd=J_fd,
    #     J_diff=J_diff,
    #     H=H,
    #     Gtheta=sens_info.get("Gtheta", None),
    #     du_dtheta=sens_info.get("du_dtheta", None),
    #     P_u=sens_info.get("P_u", None),
    #     p_tip_base=sens_info.get("p_tip_base", None),
    # )

    # np.savetxt(
    #     run_dir / "J_analytic.csv",
    #     J_an,
    #     delimiter=",",
    #     header="x_src,y_src,z_src,rx,ry,rz,L",
    #     comments="",
    # )

    # np.savetxt(
    #     run_dir / "J_fd.csv",
    #     J_fd,
    #     delimiter=",",
    #     header="x_src,y_src,z_src,rx,ry,rz,L",
    #     comments="",
    # )

    # np.savetxt(
    #     run_dir / "J_diff.csv",
    #     J_diff,
    #     delimiter=",",
    #     header="x_src,y_src,z_src,rx,ry,rz,L",
    #     comments="",
    # )

    # comparison_rows = np.array([[
    #     jacobian_metrics["relative_full_error"],
    #     jacobian_metrics["relative_translation_error"],
    #     jacobian_metrics["relative_rotation_error"],
    #     jacobian_metrics["relative_length_error"],
    #     jacobian_metrics["norm_J_fd"],
    #     jacobian_metrics["norm_diff"],
    # ]])

    # np.savetxt(
    #     run_dir / "jacobian_comparison_summary.csv",
    #     comparison_rows,
    #     delimiter=",",
    #     header="rel_full,rel_translation,rel_rotation,rel_length,norm_J_fd,norm_diff",
    #     comments="",
    # )
    # def run_analytic_energy_jacobian():
    #     J, _ = energy_min_tip_jacobian_implicit(
    #         u_opt=u_ref,
    #         p0=p0_ur,
    #         q0=q0_ur,
    #         L=L_ref,
    #         wire_len=wire_len_ref,
    #         Kinv_fun=Kinv_fun,
    #         u_star=np.zeros(3),
    #         r_src=r_src_ur,
    #         m_src=m_src,
    #         m_local_fun=make_m_local_fun_wire_tip(
    #             wire_len_ref,
    #             len_tip=tip_len_ref,
    #             mode="axial",
    #             alpha_end=0.0,
    #             eps=1e-3,
    #         ),
    #         m_moment=0.0,
    #         N=nodes,
    #         lumen_query=lumen_query if lumen_used else None,
    #         contact=contact,
    #         theta_builder=theta_builder,
    #         energy_grad_fun=energy_grad_fun,
    #         energy_fun=energy_fun,
    #         use_scalar_hessian=True,
    #         debug_jac=False,
    #         debug_hessian_terms=False,
    #         eps_theta=1e-6,
    #         eps_hess=1e-4,
    #     )
    #     return J


    # def run_fd_energy_jacobian():
    #     J_fd, _ = energy_min_tip_jacobian_fd_pose_length_robust(
    #         theta0=theta0,
    #         q_src0=q_src_ur,
    #         m_body=m_body,
    #         u_opt=u_ref,
    #         p0=p0_ur,
    #         q0=q0_ur,
    #         Kinv_fun=Kinv_fun,
    #         u_star=np.zeros(3),
    #         m_moment=0.0,
    #         N=nodes,
    #         L_tip_full=0.04,
    #         L_tip_min=0.01,
    #         rotation_convention="world",
    #         use_magnetic=True,
    #         use_lumen=lumen_used,
    #         lumen_C=lumen_C,
    #         lumen_R=lumen_R,
    #         contact=contact,
    #         maxiter=800,
    #         eps_pos=3e-4,
    #         eps_rot=3e-4,
    #         eps_L=1e-6,
    #         verbose=False,
    #     )
    #     return J_fd


    # bench_analytic = benchmark_function(
    #     run_analytic_energy_jacobian,
    #     repeats=3,
    #     warmup=1,
    #     label="Energy-min implicit analytic Jacobian",
    # )
    # print_benchmark_result(bench_analytic)

    # bench_fd = benchmark_function(
    #     run_fd_energy_jacobian,
    #     repeats=3,
    #     warmup=1,
    #     label="Energy-min full finite-difference Jacobian",
    # )
    # print_benchmark_result(bench_fd)

    # J_an_bench = bench_analytic["last_result"]
    # J_fd_bench = bench_fd["last_result"]

    # print("\nSpeedup FD / analytic:")
    # print(bench_fd["mean_s"] / max(bench_analytic["mean_s"], 1e-12))

    # print("\nBenchmark-run relative Jacobian error:")
    # print(np.linalg.norm(J_an_bench - J_fd_bench) / max(np.linalg.norm(J_fd_bench), 1e-12))

    # benchmark_metrics = {
    #     "analytic_mean_s": bench_analytic["mean_s"],
    #     "analytic_std_s": bench_analytic["std_s"],
    #     "analytic_min_s": bench_analytic["min_s"],
    #     "analytic_max_s": bench_analytic["max_s"],
    #     "fd_mean_s": bench_fd["mean_s"],
    #     "fd_std_s": bench_fd["std_s"],
    #     "fd_min_s": bench_fd["min_s"],
    #     "fd_max_s": bench_fd["max_s"],
    #     "speedup_fd_over_analytic": bench_fd["mean_s"] / max(bench_analytic["mean_s"], 1e-12),
    #     "benchmark_relative_error": float(
    #         np.linalg.norm(J_an_bench - J_fd_bench)
    #         / max(np.linalg.norm(J_fd_bench), 1e-12)
    #     ),
    # }

    # save_json(run_dir / "benchmark_metrics.json", benchmark_metrics)

    # check_contact_gradient_against_energy_fd(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=info["s"],
    #     lumen_query=lumen_query,
    #     contact=contact,
    #     eps=1e-6,
    #     max_cols=10,
    # )
    # check_contact_force_position_gradient(
    #     pE,
    #     lumen_query=lumen_query,
    #     contact=contact,
    #     eps=1e-7,
    # )
    # g_contact_an, contact_dbg = contact_energy_gradient_u_analytic(
    #     u_ref,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     s=info["s"],
    #     lumen_query=lumen_query,
    #     contact=contact,
    #     return_debug=True,
    #     debug_print=True,
    # )

    # contact_metrics = {
    #     "W_contact_reconstructed": float(contact_dbg["W_contact_reconstructed"]),
    #     "W_cf_from_energy_from_u": float(info["parts"]["W_cf"]),
    #     "W_contact_ratio": float(
    #         contact_dbg["W_contact_reconstructed"]
    #         / max(abs(info["parts"]["W_cf"]), 1e-30)
    #     ),
    #     "contact_grad_norm": float(np.linalg.norm(g_contact_an)),
    #     "contact_grad_max_abs": float(np.max(np.abs(g_contact_an))),
    #     "contact_min_gap": float(contact_dbg["min_gap"]),
    #     "contact_active_nodes": np.asarray(contact_dbg["active_nodes"], dtype=int).tolist(),
    # }

    # save_json(run_dir / "contact_metrics.json", contact_metrics)

    # np.savez(
    #     run_dir / "contact_validation.npz",
    #     g_contact_an=g_contact_an,
    #     C_nodes=contact_dbg["C_nodes"],
    #     F_nodes=contact_dbg["F_nodes"],
    #     gap_nodes=contact_dbg["gap_nodes"],
    #     weights=contact_dbg["weights"],
    #     p=contact_dbg["p"],
    # )
    # summary = {
    #     "config": config,
    #     "forward": forward_metrics,
    #     "stationarity": stationarity_metrics,
    #     "magnetic_gradient": mag_grad_metrics,
    #     "contact": contact_metrics,
    #     "jacobian": jacobian_metrics,
    #     "hessian": hessian_metrics,
    #     "benchmark": benchmark_metrics,
    #     # "eps_hess_sweep": eps_hess_rows,
    #     # "fd_step_sweep": fd_step_sweep_rows,
    # }

    # save_json(run_dir / "summary.json", summary)
    # print("Saved paper results to:", run_dir)