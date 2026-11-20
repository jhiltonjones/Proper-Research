import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_point, magnetic_wrench_density_cosserat_profile_segments
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
from pathlib import Path
import pandas as pd
from dataclasses import dataclass

L_tip_full=0.04
from scipy.integrate import solve_ivp
import numpy as np

@dataclass(frozen=True)
class ContactParams:
    r_beam: float = 0.001
    k: float = 1e8
    pen_switch: float = 5e-5
    k_hard: float = 1e10
    smooth: bool = True
    smooth_eps: float = 1e-5
    window: int = 3
def params_from_L(L):
    L_model, wire_len, tip_len = effective_lengths(
        L,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )
    m_local_fun_L = make_m_local_fun_wire_tip(
        wire_len,
        len_tip=tip_len,
        mode="axial",
        alpha_end=0.0,
        eps=1e-3,
    )
    return L_model, wire_len, tip_len, m_local_fun_L
def normalized_rhs_single(
    xi,
    y,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
):
    s = float(L) * float(xi)
    f, _ = cosserat_rhs_single(
        s,
        y,
        m_src=m_src,
        r_src=r_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
    )
    return float(L) * f


def normalized_length_column_fd(
    xi,
    y,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_moment,
    u_star=None,
    eps_L=1e-5,
):
    """
    Computes d/dL of normalized RHS:

        y_xi = L * f(s=L*xi, y, params(L))

    at fixed xi and fixed y.
    """

    Lp = float(L) + eps_L
    Lm = max(float(L) - eps_L, 1e-8)

    _, wire_p, _, m_local_p = params_from_L(Lp)
    _, wire_m, _, m_local_m = params_from_L(Lm)

    gp = normalized_rhs_single(
        xi,
        y,
        L=Lp,
        r_src=r_src,
        m_src=m_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_p,
        m_moment=m_moment,
        wire_len=wire_p,
        u_star=u_star,
    )

    gm = normalized_rhs_single(
        xi,
        y,
        L=Lm,
        r_src=r_src,
        m_src=m_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_m,
        m_moment=m_moment,
        wire_len=wire_m,
        u_star=u_star,
    )

    return (gp - gm) / (Lp - Lm)
def solve_tip_sensitivity_source_dipole_length_shooting(
    sol_nom,
    *,
    L,
    r_src,
    m_src,
    dm_src_dphi=None,
    Kinv_fun,
    m_moment,
    u_star=None,
    n_eval=200,
    rtol=1e-6,
    atol=1e-8,
    method="DOP853",
    eps_L=1e-5,
):
    """
    Fast shooting sensitivity for:

        J = d p_tip / d [r_src, delta_phi, L]

    where m_src is supplied directly in world coordinates.

    Columns:
        0:3 -> source translation
        3:6 -> chosen dipole-rotation perturbation
        6   -> model length L
    """

    L = float(L)
    r_src = np.asarray(r_src, float).reshape(3,)
    m_src = np.asarray(m_src, float).reshape(3,)

    _, wire_len, _, m_local_fun = params_from_L(L)

    if dm_src_dphi is None:
        # World-frame infinitesimal rotation:
        # delta m = delta_phi x m = -[m]_x delta_phi
        dm_src_dphi = -skew(m_src)

    n_state = 13
    n_ctrl = 7
    n_eta = 6

    P0 = np.zeros((n_state, n_ctrl), dtype=float)

    U0 = np.zeros((n_state, n_eta), dtype=float)
    U0[7:13, :] = np.eye(6)

    W0 = np.concatenate([
        P0.reshape(-1),
        U0.reshape(-1),
    ])

    # then keep the rest of your existing sensitivity integration code,
    # but use m_src and dm_src_dphi directly.

    def ode_shoot(xi, W_flat):
        W_flat = np.asarray(W_flat, float)

        P = W_flat[:n_state * n_ctrl].reshape(n_state, n_ctrl)
        U = W_flat[n_state * n_ctrl:].reshape(n_state, n_eta)

        s = L * float(xi)
        y_nom = sol_nom.sol(np.array([s], dtype=float))[:, 0]

        A_s = rhs_state_jacobian_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )

        B_pose_s = rhs_control_jacobian_source_pose_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            dm_src_dphi=dm_src_dphi,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )  # (13,6)

        A_xi = L * A_s

        B_xi = np.zeros((13, 7), dtype=float)
        B_xi[:, 0:6] = L * B_pose_s

        B_xi[:, 6] = normalized_length_column_fd(
            xi,
            y_nom,
            L=L,
            r_src=r_src,
            m_src=m_src,
            Kinv_fun=Kinv_fun,
            m_moment=m_moment,
            u_star=u_star,
            eps_L=eps_L,
        )

        dP = A_xi @ P + B_xi
        dU = A_xi @ U

        return np.concatenate([
            dP.reshape(-1),
            dU.reshape(-1),
        ])

    sol_ivp = solve_ivp(
        ode_shoot,
        t_span=(0.0, 1.0),
        y0=W0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        t_eval=np.linspace(0.0, 1.0, int(n_eval)),
    )

    if not sol_ivp.success:
        raise RuntimeError(f"Pose+length shooting sensitivity failed: {sol_ivp.message}")

    W_1 = sol_ivp.y[:, -1]

    P_1 = W_1[:n_state * n_ctrl].reshape(n_state, n_ctrl)
    U_1 = W_1[n_state * n_ctrl:].reshape(n_state, n_eta)

    C_P = P_1[7:13, :]   # (6,7)
    C_U = U_1[7:13, :]   # (6,6)

    eta = np.linalg.solve(C_U, -C_P)

    Z_1 = P_1 + U_1 @ eta

    J_tip_pose_length = Z_1[0:3, :]  # (3,7)

    diagnostics = {
        "sol_ivp": sol_ivp,
        "P_1": P_1,
        "U_1": U_1,
        "eta": eta,
        "tip_residual_nm": Z_1[7:13, :],
        "terminal_matrix_cond": np.linalg.cond(C_U),
    }

    return J_tip_pose_length, diagnostics
def solve_tip_sensitivity_source_pose_length_shooting(
    sol_nom,
    *,
    L,
    r_src,
    q_src,
    m_body,
    Kinv_fun,
    m_moment,
    u_star=None,
    n_eval=200,
    rtol=1e-6,
    atol=1e-8,
    method="DOP853",
    rotation_convention="world",
    eps_L=1e-5,
):
    """
    Fast shooting sensitivity for:

        J = d p_tip / d [r_src, delta_phi, L]

    Shape:
        J : (3, 7)

    Columns:
        0:3 -> source translation
        3:6 -> source rotation-vector perturbation
        6   -> model length L
    """

    L = float(L)
    r_src = np.asarray(r_src, float).reshape(3,)
    q_src = quat_normalize(np.asarray(q_src, float).reshape(4,))
    m_body = np.asarray(m_body, float).reshape(3,)

    _, wire_len, _, m_local_fun = params_from_L(L)

    m_src = dipole_from_pose(q_src, m_body)

    dm_src_dphi = source_dipole_rotation_jacobian(
        q_src,
        m_body,
        convention=rotation_convention,
    )

    n_state = 13
    n_ctrl = 7
    n_eta = 6

    P0 = np.zeros((n_state, n_ctrl), dtype=float)

    U0 = np.zeros((n_state, n_eta), dtype=float)
    U0[7:13, :] = np.eye(6)

    W0 = np.concatenate([
        P0.reshape(-1),
        U0.reshape(-1),
    ])

    def ode_shoot(xi, W_flat):
        W_flat = np.asarray(W_flat, float)

        P = W_flat[:n_state * n_ctrl].reshape(n_state, n_ctrl)
        U = W_flat[n_state * n_ctrl:].reshape(n_state, n_eta)

        s = L * float(xi)
        y_nom = sol_nom.sol(np.array([s], dtype=float))[:, 0]

        A_s = rhs_state_jacobian_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )

        B_pose_s = rhs_control_jacobian_source_pose_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            dm_src_dphi=dm_src_dphi,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )  # (13,6)

        A_xi = L * A_s

        B_xi = np.zeros((13, 7), dtype=float)
        B_xi[:, 0:6] = L * B_pose_s

        B_xi[:, 6] = normalized_length_column_fd(
            xi,
            y_nom,
            L=L,
            r_src=r_src,
            m_src=m_src,
            Kinv_fun=Kinv_fun,
            m_moment=m_moment,
            u_star=u_star,
            eps_L=eps_L,
        )

        dP = A_xi @ P + B_xi
        dU = A_xi @ U

        return np.concatenate([
            dP.reshape(-1),
            dU.reshape(-1),
        ])

    sol_ivp = solve_ivp(
        ode_shoot,
        t_span=(0.0, 1.0),
        y0=W0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        t_eval=np.linspace(0.0, 1.0, int(n_eval)),
    )

    if not sol_ivp.success:
        raise RuntimeError(f"Pose+length shooting sensitivity failed: {sol_ivp.message}")

    W_1 = sol_ivp.y[:, -1]

    P_1 = W_1[:n_state * n_ctrl].reshape(n_state, n_ctrl)
    U_1 = W_1[n_state * n_ctrl:].reshape(n_state, n_eta)

    C_P = P_1[7:13, :]   # (6,7)
    C_U = U_1[7:13, :]   # (6,6)

    eta = np.linalg.solve(C_U, -C_P)

    Z_1 = P_1 + U_1 @ eta

    J_tip_pose_length = Z_1[0:3, :]  # (3,7)

    diagnostics = {
        "sol_ivp": sol_ivp,
        "P_1": P_1,
        "U_1": U_1,
        "eta": eta,
        "tip_residual_nm": Z_1[7:13, :],
        "terminal_matrix_cond": np.linalg.cond(C_U),
    }

    return J_tip_pose_length, diagnostics

def solve_tip_sensitivity_source_pose_shooting(
    sol_nom,
    *,
    L,
    r_src,
    q_src,
    m_body,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_eval=200,
    rtol=1e-6,
    atol=1e-8,
    method="DOP853",
    rotation_convention="world",
):
    """
    Fast shooting sensitivity for:

        J = d p_tip / d [r_src, delta_phi]

    Shape:
        J : (3, 6)

    Columns:
        0:3 -> source translation perturbations
        3:6 -> source rotation-vector perturbations

    This replaces the 78-state solve_bvp sensitivity solve.
    """

    r_src = np.asarray(r_src, float).reshape(3,)
    q_src = quat_normalize(np.asarray(q_src, float).reshape(4,))
    m_body = np.asarray(m_body, float).reshape(3,)

    m_src = dipole_from_pose(q_src, m_body)

    dm_src_dphi = source_dipole_rotation_jacobian(
        q_src,
        m_body,
        convention=rotation_convention,
    )  # (3,3)

    n_state = 13
    n_ctrl = 6
    n_eta = 6

    # P: particular sensitivity wrt source pose, shape (13,6)
    # U: homogeneous sensitivity wrt unknown initial [n0, m0], shape (13,6)
    #
    # Flatten [P, U] into one IVP vector.
    P0 = np.zeros((n_state, n_ctrl), dtype=float)

    U0 = np.zeros((n_state, n_eta), dtype=float)
    U0[7:13, :] = np.eye(6)  # unknown initial delta n(0), delta m(0)

    W0 = np.concatenate([P0.reshape(-1), U0.reshape(-1)])

    def ode_shoot(s, W_flat):
        W_flat = np.asarray(W_flat, float)

        P = W_flat[:n_state * n_ctrl].reshape(n_state, n_ctrl)
        U = W_flat[n_state * n_ctrl:].reshape(n_state, n_eta)

        y_nom = sol_nom.sol(np.array([s], dtype=float))[:, 0]

        A = rhs_state_jacobian_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )

        B = rhs_control_jacobian_source_pose_analytic(
            s,
            y_nom,
            m_src=m_src,
            r_src=r_src,
            dm_src_dphi=dm_src_dphi,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )  # (13,6)

        dP = A @ P + B
        dU = A @ U

        return np.concatenate([dP.reshape(-1), dU.reshape(-1)])

    sol_ivp = solve_ivp(
        ode_shoot,
        t_span=(0.0, float(L)),
        y0=W0,
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        t_eval=np.linspace(0.0, float(L), int(n_eval)),
    )

    if not sol_ivp.success:
        raise RuntimeError(f"Shooting sensitivity IVP failed: {sol_ivp.message}")

    W_L = sol_ivp.y[:, -1]

    P_L = W_L[:n_state * n_ctrl].reshape(n_state, n_ctrl)
    U_L = W_L[n_state * n_ctrl:].reshape(n_state, n_eta)

    # Tip free-end condition: delta n(L)=0, delta m(L)=0
    C_P = P_L[7:13, :]      # (6,6)
    C_U = U_L[7:13, :]      # (6,6)

    # Solve C_U @ eta = -C_P
    eta = np.linalg.solve(C_U, -C_P)       # (6,6)

    Z_L = P_L + U_L @ eta                  # (13,6)

    J_tip_pose = Z_L[0:3, :]               # (3,6)

    diagnostics = {
        "sol_ivp": sol_ivp,
        "P_L": P_L,
        "U_L": U_L,
        "eta": eta,
        "tip_residual_nm": Z_L[7:13, :],
        "terminal_matrix_cond": np.linalg.cond(C_U),
    }

    return J_tip_pose, diagnostics
def tip_jacobian_source_pose_length_fd_full(
    *,
    model_factory_from_length,
    L,
    r_src,
    q_src,
    m_body,
    eps_pos=1e-5,
    eps_rot=1e-5,
    eps_L=1e-5,
    rotation_convention="world",
):
    model0, wire_len0 = model_factory_from_length(L)

    out0 = model0.forward(
        L=L,
        r_src=r_src,
        q_src=q_src,
        m_body=m_body,
        wire_len=wire_len0,
    )
    p0 = out0["p_tip"]

    J = np.zeros((3, 7), dtype=float)

    # translation columns
    for j in range(3):
        rp = np.asarray(r_src, float).copy()
        rp[j] += eps_pos

        modelp, wire_lenp = model_factory_from_length(L)

        outp = modelp.forward(
            L=L,
            r_src=rp,
            q_src=q_src,
            m_body=m_body,
            wire_len=wire_lenp,
        )

        J[:, j] = (outp["p_tip"] - p0) / eps_pos

    # rotation-vector columns
    for j in range(3):
        dphi = np.zeros(3)
        dphi[j] = eps_rot

        qp = perturb_q_src(q_src, dphi, convention=rotation_convention)

        modelp, wire_lenp = model_factory_from_length(L)

        outp = modelp.forward(
            L=L,
            r_src=r_src,
            q_src=qp,
            m_body=m_body,
            wire_len=wire_lenp,
        )

        J[:, 3 + j] = (outp["p_tip"] - p0) / eps_rot

    # length column
    Lp = float(L) + eps_L
    modelp, wire_lenp = model_factory_from_length(Lp)

    outp = modelp.forward(
        L=Lp,
        r_src=r_src,
        q_src=q_src,
        m_body=m_body,
        wire_len=wire_lenp,
    )

    J[:, 6] = (outp["p_tip"] - p0) / eps_L

    return J
def model_factory_from_length(L_here):
    L_model, wire_len_here, tip_len_here = effective_lengths(
        L_here,
        L_tip_full=0.04,
        L_tip_min=0.01,
    )

    model_here = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kinv_fun,
        m_local_fun=make_m_local_fun_wire_tip(
            wire_len_here,
            len_tip=tip_len_here,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        ),
        m_moment=0.0,
        wire_len=wire_len_here,
        n_nodes=nodes,
        tol=1e-5,
        max_nodes=20000,
    )

    return model_here, wire_len_here
def tip_jacobian_source_pose_fd_full(
    *,
    model_factory,
    L,
    r_src,
    q_src,
    m_body,
    wire_len,
    eps_pos=1e-5,
    eps_rot=1e-5,
    rotation_convention="world",
):
    model0 = model_factory()
    out0 = model0.forward(
        L=L,
        r_src=r_src,
        q_src=q_src,
        m_body=m_body,
        wire_len=wire_len,
    )
    p0 = out0["p_tip"]

    J = np.zeros((3, 6), dtype=float)

    for j in range(3):
        rp = np.asarray(r_src, float).copy()
        rp[j] += eps_pos

        modelp = model_factory()
        outp = modelp.forward(
            L=L,
            r_src=rp,
            q_src=q_src,
            m_body=m_body,
            wire_len=wire_len,
        )

        J[:, j] = (outp["p_tip"] - p0) / eps_pos

    for j in range(3):
        dphi = np.zeros(3)
        dphi[j] = eps_rot

        qp = perturb_q_src(q_src, dphi, convention=rotation_convention)

        modelp = model_factory()
        outp = modelp.forward(
            L=L,
            r_src=r_src,
            q_src=qp,
            m_body=m_body,
            wire_len=wire_len,
        )

        J[:, 3 + j] = (outp["p_tip"] - p0) / eps_rot

    return J
def quat_from_small_rot(delta_phi):
    delta_phi = np.asarray(delta_phi, float).reshape(3,)
    a = np.linalg.norm(delta_phi)

    if a < 1e-14:
        return np.array([1.0, 0.5*delta_phi[0], 0.5*delta_phi[1], 0.5*delta_phi[2]])

    axis = delta_phi / a
    h = 0.5 * a
    return np.array([np.cos(h), *(np.sin(h) * axis)], dtype=float)


def perturb_q_src(q_src, delta_phi, convention="world"):
    dq = quat_from_small_rot(delta_phi)
    q_src = quat_normalize(q_src)

    if convention == "world":
        return quat_normalize(quat_mul(dq, q_src))

    if convention == "body":
        return quat_normalize(quat_mul(q_src, dq))

    raise ValueError("convention must be 'world' or 'body'")
def solve_tip_sensitivity_source_pose(
    sol_nom,
    *,
    L,
    r_src,
    q_src,
    m_body,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_nodes=120,
    tol=1e-5,
    max_nodes=20000,
    rotation_convention="world",
):
    r_src = np.asarray(r_src, float).reshape(3,)
    q_src = quat_normalize(np.asarray(q_src, float).reshape(4,))
    m_body = np.asarray(m_body, float).reshape(3,)

    m_src = dipole_from_pose(q_src, m_body)
    dm_src_dphi = source_dipole_rotation_jacobian(
        q_src,
        m_body,
        convention=rotation_convention,
    )

    s_mesh = np.linspace(0.0, float(L), int(n_nodes))

    def sens_ode(s, Z_flat):
        s = np.asarray(s, float).ravel()
        Z_flat = np.asarray(Z_flat, float)

        ns = s.size
        Z = Z_flat.reshape(13, 6, ns)
        dZ = np.zeros_like(Z)

        Y_nom = sol_nom.sol(s)

        for i in range(ns):
            y_nom = Y_nom[:, i]

            A = rhs_state_jacobian_analytic(
                s[i], y_nom,
                m_src=m_src,
                r_src=r_src,
                Kinv_fun=Kinv_fun,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
                wire_len=wire_len,
                u_star=u_star,
            )

            B = rhs_control_jacobian_source_pose_analytic(
                s[i], y_nom,
                m_src=m_src,
                r_src=r_src,
                dm_src_dphi=dm_src_dphi,
                Kinv_fun=Kinv_fun,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
                wire_len=wire_len,
                u_star=u_star,
            )

            dZ[:, :, i] = A @ Z[:, :, i] + B

        return dZ.reshape(78, ns)

    Z_guess = np.zeros((78, s_mesh.size), dtype=float)

    sol_sens = solve_bvp(
        sens_ode,
        bc_sensitivity_pose_columns,
        s_mesh,
        Z_guess,
        tol=tol,
        max_nodes=max_nodes,
    )

    if not sol_sens.success:
        raise RuntimeError(f"Source-pose sensitivity failed: {sol_sens.message}")

    Z_tip = sol_sens.sol(np.array([L], dtype=float))[:, 0].reshape(13, 6)

    J_tip_pose = Z_tip[0:3, :]

    return J_tip_pose, sol_sens
def bc_sensitivity_pose_columns(Za, Zb):
    Za = np.asarray(Za, float).reshape(13, 6)
    Zb = np.asarray(Zb, float).reshape(13, 6)

    bc = np.zeros((13, 6), dtype=float)
    bc[0:3, :] = Za[0:3, :]
    bc[3:7, :] = Za[3:7, :]
    bc[7:10, :] = Zb[7:10, :]
    bc[10:13, :] = Zb[10:13, :]

    return bc.reshape(-1)
def rhs_control_jacobian_source_pose_analytic(
    s, y, *,
    m_src,
    r_src,
    dm_src_dphi,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
):
    """
    B_pose = d RHS / d [r_src, delta_phi], shape (13,6)

    Columns:
      0:3 -> source translation perturbation
      3:6 -> source rotation perturbation
    """
    y = np.asarray(y, float).reshape(13,)
    p = y[0:3]
    q = quat_normalize(y[3:7])

    R = quat_to_R(q)
    m_local = m_local_fun(np.array([s]), m_moment)[:, 0]
    m_world = R @ m_local

    # -------------------------
    # Translation part
    # -------------------------
    dB_drsrc = dipole_field_jacobian_wrt_rsrc(p, r_src, m_src)
    df_drsrc = force_jacobian_rsrc(p, r_src, m_src, m_world)
    dtau_drsrc = torque_jacobian_rsrc(m_world, dB_drsrc)

    B_r = np.zeros((13, 3), dtype=float)
    B_r[7:10, :] = -df_drsrc
    B_r[10:13, :] = -dtau_drsrc

    # -------------------------
    # Rotation part
    # -------------------------
    dB_dmsrc = dipole_field_jacobian_wrt_msrc(p, r_src)
    df_dmsrc = force_jacobian_msrc(p, r_src, m_src, m_world)
    dtau_dmsrc = torque_jacobian_msrc(m_world, dB_dmsrc)

    B_m = np.zeros((13, 3), dtype=float)
    B_m[7:10, :] = -df_dmsrc
    B_m[10:13, :] = -dtau_dmsrc

    B_phi = B_m @ dm_src_dphi

    return np.hstack([B_r, B_phi])
def dipole_field_jacobian_wrt_msrc(p, r_src, mu0=4*np.pi*1e-7):
    c = mu0 / (4*np.pi)
    R = np.asarray(p, float).reshape(3,) - np.asarray(r_src, float).reshape(3,)

    r2 = np.dot(R, R) + 1e-24
    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r3 * r2

    return c * (3.0 * np.outer(R, R) / r5 - np.eye(3) / r3)


def force_jacobian_msrc(p, r_src, m_src, m_world, mu0=4*np.pi*1e-7):
    """
    df_ext / d m_src, shape (3,3)
    """
    c = mu0 / (4*np.pi)

    R = np.asarray(p, float).reshape(3,) - np.asarray(r_src, float).reshape(3,)
    m = np.asarray(m_world, float).reshape(3,)

    r2 = np.dot(R, R) + 1e-24
    r = np.sqrt(r2)
    r5 = r2 * r2 * r

    a = np.dot(m, R)  # m_world dot R

    return 3.0 * c / r5 * (
        a * np.eye(3)
        + np.outer(m, R)
        + np.outer(R, m)
        - 5.0 * a / r2 * np.outer(R, R)
    )


def torque_jacobian_msrc(m_world, dB_dmsrc):
    """
    d tau_ext / d m_src, where tau = m_world x B.
    """
    return skew(m_world) @ dB_dmsrc
def source_dipole_rotation_jacobian(q_src, m_body, convention="world"):
    """
    Returns dm_src / d(delta_phi), shape (3,3).

    convention="world":
        q_new = delta_q ⊗ q_src
        delta m_src = delta_phi_world x m_src

    convention="body":
        q_new = q_src ⊗ delta_q
        delta m_src = R_src @ (delta_phi_body x m_body)
    """
    R_src = quat_to_R(quat_normalize(q_src))
    m_body = np.asarray(m_body, float).reshape(3,)
    m_src = R_src @ m_body

    if convention == "world":
        return -skew(m_src)

    if convention == "body":
        return -R_src @ skew(m_body)

    raise ValueError("convention must be 'world' or 'body'")
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

def bc_sensitivity(za, zb):
    bc = np.zeros(13, dtype=float)
    bc[0:3] = za[0:3]       # delta p(0) = 0
    bc[3:7] = za[3:7]       # delta q(0) = 0
    bc[7:10] = zb[7:10]     # delta n(L) = 0
    bc[10:13] = zb[10:13]   # delta m(L) = 0
    return bc
def bc_sensitivity_all_columns(Za, Zb):
    """
    Boundary conditions for Z(s) in R^{13x3}, flattened as length-39 vector.

    Base:
        delta p(0) = 0
        delta q(0) = 0

    Tip:
        delta n(L) = 0
        delta m(L) = 0
    for all 3 control directions simultaneously.
    """
    Za = np.asarray(Za, float).reshape(13, 3)
    Zb = np.asarray(Zb, float).reshape(13, 3)

    bc = np.zeros((13, 3), dtype=float)
    bc[0:3, :] = Za[0:3, :]       # delta p(0) = 0
    bc[3:7, :] = Za[3:7, :]       # delta q(0) = 0
    bc[7:10, :] = Zb[7:10, :]     # delta n(L) = 0
    bc[10:13, :] = Zb[10:13, :]   # delta m(L) = 0

    return bc.reshape(-1)
def solve_tip_sensitivity_rsrc_all_columns(
    sol_nom,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_nodes=120,
    tol=1e-5,
    max_nodes=20000,
):
    """
    Compute J = d p_tip / d r_src, shape (3,3),
    by solving ONE linear sensitivity BVP for all 3 control directions at once.

    Z(s) has shape (13,3), satisfying:
        Z'(s) = A(s) Z(s) + B(s)

    where:
        A(s) : (13,13)
        B(s) : (13,3)

    The returned Jacobian is:
        J = Z(L)[0:3, :]
    """
    r_src = np.asarray(r_src, float).reshape(3,)
    m_src = np.asarray(m_src, float).reshape(3,)

    s_mesh = np.linspace(0.0, float(L), int(n_nodes))

    def sens_ode_all(s, Z_flat):
        s = np.asarray(s, float)
        Z_flat = np.asarray(Z_flat, float)

        ns = s.size
        Z = Z_flat.reshape(13, 3, ns)
        dZ = np.zeros_like(Z)

        Y_nom = sol_nom.sol(s)   # shape (13, ns)

        for i in range(ns):
            y_nom = Y_nom[:, i]

            A = rhs_state_jacobian_analytic(
                s[i], y_nom,
                m_src=m_src,
                r_src=r_src,
                Kinv_fun=Kinv_fun,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
                wire_len=wire_len,
                u_star=u_star,
            )

            B = rhs_control_jacobian_rsrc_analytic(
                s[i], y_nom,
                m_src=m_src,
                r_src=r_src,
                Kinv_fun=Kinv_fun,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
                wire_len=wire_len,
                u_star=u_star,
            )

            dZ[:, :, i] = A @ Z[:, :, i] + B

        return dZ.reshape(39, ns)

    Z_guess = np.zeros((39, s_mesh.size), dtype=float)

    sol_sens = solve_bvp(
        sens_ode_all,
        bc_sensitivity_all_columns,
        s_mesh,
        Z_guess,
        tol=tol,
        max_nodes=max_nodes,
    )

    if not sol_sens.success:
        raise RuntimeError(f"All-column sensitivity solve failed: {sol_sens.message}")

    Z_tip = sol_sens.sol(np.array([L], dtype=float))[:, 0].reshape(13, 3)
    J = Z_tip[0:3, :]

    return J, sol_sens
def build_nominal_sensitivity_cache(
    sol_nom,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_nodes=120,
):
    s_mesh = np.linspace(0.0, float(L), int(n_nodes))
    Y_nom = sol_nom.sol(s_mesh)

    N = s_mesh.size
    A_list = np.zeros((N, 13, 13), dtype=float)
    B_list = np.zeros((N, 13, 3), dtype=float)

    for i in range(N):
        y_i = Y_nom[:, i]
        s_i = s_mesh[i]

        A_list[i] = rhs_state_jacobian_analytic(
            s_i, y_i,
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )

        B_list[i] = rhs_control_jacobian_rsrc_analytic(
            s_i, y_i,
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )

    return {
        "s_mesh": s_mesh,
        "Y_nom": Y_nom,
        "A_list": A_list,
        "B_list": B_list,
    }
def solve_tip_sensitivity_rsrc_cached(
    sol_nom,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_nodes=120,
    tol=1e-5,
    max_nodes=20000,
):
    """
    Cached sensitivity solve with interpolated A(s), B(s).
    """
    cache = build_nominal_sensitivity_cache(
        sol_nom,
        L=L,
        r_src=r_src,
        m_src=m_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
        n_nodes=50,
    )

    s_mesh = cache["s_mesh"]
    A_list = cache["A_list"]   # (N,13,13)
    B_list = cache["B_list"]   # (N,13,3)

    J = np.zeros((3, 3), dtype=float)
    sols = []

    for j in range(3):
        def sens_ode(s, Z):
            s = np.asarray(s, float).ravel()
            Z = np.asarray(Z, float)

            dZ = np.zeros_like(Z)

            Aq = interp_matrix_field(s, s_mesh, A_list)   # (ns,13,13)
            Bq = interp_matrix_field(s, s_mesh, B_list)   # (ns,13,3)

            for i in range(s.size):
                dZ[:, i] = Aq[i] @ Z[:, i] + Bq[i][:, j]

            return dZ

        def sens_fun_jac(s, Z):
            """
            Jacobian of sens_ode wrt Z.
            Since dZ/ds = A(s) Z + b(s), this is just A(s).
            solve_bvp expects shape (n, n, m).
            """
            s = np.asarray(s, float).ravel()
            Aq = interp_matrix_field(s, s_mesh, A_list)   # (ns,13,13)

            # transpose to (13,13,ns)
            return np.transpose(Aq, (1, 2, 0))

        Z_guess = np.zeros((13, s_mesh.size), dtype=float)

        sol_sens = solve_bvp(
            sens_ode,
            bc_sensitivity,
            s_mesh,
            Z_guess,
            fun_jac=sens_fun_jac,
            tol=tol,
            max_nodes=max_nodes,
        )

        if not sol_sens.success:
            raise RuntimeError(
                f"Sensitivity solve failed for r_src component {j}: {sol_sens.message}"
            )

        z_tip = sol_sens.sol(np.array([L], dtype=float))[:, 0]
        J[:, j] = z_tip[0:3]
        sols.append(sol_sens)

    return J, sols, cache
def solve_tip_sensitivity_rsrc(
    sol_nom,
    *,
    L,
    r_src,
    m_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    n_nodes=120,
    tol=1e-5,
    max_nodes=20000,
):
    """
    Compute J = d p_tip / d r_src, shape (3,3),
    by solving 3 linear sensitivity BVPs around the nominal solution.
    """
    r_src = np.asarray(r_src, float).reshape(3,)
    m_src = np.asarray(m_src, float).reshape(3,)

    s_mesh = np.linspace(0.0, float(L), int(n_nodes))
    J = np.zeros((3, 3), dtype=float)
    sols = []

    for j in range(3):
        def sens_ode(s, Z):
            s = np.asarray(s, float)
            Z = np.asarray(Z, float)
            dZ = np.zeros_like(Z)

            Y_nom = sol_nom.sol(s)   # shape (13, len(s))

            for i in range(s.size):
                y_nom = Y_nom[:, i]

                A = rhs_state_jacobian_analytic(
                    s[i], y_nom,
                    m_src=m_src,
                    r_src=r_src,
                    Kinv_fun=Kinv_fun,
                    m_local_fun=m_local_fun,
                    m_moment=m_moment,
                    wire_len=wire_len,
                    u_star=u_star,
                )
                B = rhs_control_jacobian_rsrc_analytic(
                    s[i], y_nom,
                    m_src=m_src,
                    r_src=r_src,
                    Kinv_fun=Kinv_fun,
                    m_local_fun=m_local_fun,
                    m_moment=m_moment,
                    wire_len=wire_len,
                    u_star=u_star,
                )

                dZ[:, i] = A @ Z[:, i] + B[:, j]

            return dZ

        Z_guess = np.zeros((13, s_mesh.size), dtype=float)

        sol_sens = solve_bvp(
            sens_ode,
            bc_sensitivity,
            s_mesh,
            Z_guess,
            tol=tol,
            max_nodes=max_nodes,
        )

        if not sol_sens.success:
            raise RuntimeError(
                f"Sensitivity solve failed for r_src component {j}: {sol_sens.message}"
            )

        z_tip = sol_sens.sol(np.array([L]))[:, 0]
        J[:, j] = z_tip[0:3]
        sols.append(sol_sens)

    return J, sols
def interp_matrix_field(s_query, s_mesh, M_list):
    """
    Interpolate matrix-valued field M(s).

    Parameters
    ----------
    s_query : (Nq,)
    s_mesh  : (N,)
    M_list  : (N, a, b)

    Returns
    -------
    Mq : (Nq, a, b)
    """
    s_query = np.asarray(s_query, float).ravel()
    s_mesh = np.asarray(s_mesh, float).ravel()
    M_list = np.asarray(M_list, float)

    Nq = s_query.size
    N, a, b = M_list.shape
    Mq = np.zeros((Nq, a, b), dtype=float)

    for r in range(a):
        for c in range(b):
            Mq[:, r, c] = np.interp(s_query, s_mesh, M_list[:, r, c])

    return Mq
def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3 - 2*x)
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

class LumenQuery:
    def __init__(self, C, R):
        self.C = np.asarray(C, float)
        self.R = np.asarray(R, float)
        self.kdt = cKDTree(self.C)

    def closest(self, p, window=3):
        # p: (3,)
        _, k = self.kdt.query(p)
        i0 = max(0, k-window)
        i1 = min(len(self.C)-2, k+window)
        dmin = np.inf
        best = None
        for i in range(i0, i1+1):
            q, t = closest_point_on_segment(p, self.C[i], self.C[i+1])
            d = np.linalg.norm(p-q)
            if d < dmin:
                dmin = d
                best = (i, t, q)
        i, t, q = best
        Rloc = (1-t)*self.R[i] + t*self.R[i+1]
        return dmin, Rloc, q


def tip_bending_angles_from_tangent_signed(
    sol,
    L,
    beam_tangent_ref,
    beam_y_ref,
    beam_z_ref,
    e1=np.array([-1.0, 0.0, 0.0]),
):
    """
    Signed bending angles in the beam's local transverse directions.

    theta_y_signed:
        signed angular deflection toward beam_y_ref.

    theta_z_signed:
        signed angular deflection toward beam_z_ref.

    theta_total:
        unsigned total bend magnitude away from beam_tangent_ref.
    """

    YL = sol.sol(np.array([L]))
    qL = quat_normalize(YL[3:7, :])
    RL = quat_to_rot(qL)[0]

    tL = RL @ e1
    tL = tL / (np.linalg.norm(tL) + 1e-12)

    beam_tangent_ref = beam_tangent_ref / (np.linalg.norm(beam_tangent_ref) + 1e-12)
    beam_y_ref = beam_y_ref / (np.linalg.norm(beam_y_ref) + 1e-12)
    beam_z_ref = beam_z_ref / (np.linalg.norm(beam_z_ref) + 1e-12)

    forward_component = np.dot(tL, beam_tangent_ref)
    y_component = np.dot(tL, beam_y_ref)
    z_component = np.dot(tL, beam_z_ref)

    theta_y_signed = np.arctan2(y_component, forward_component)
    theta_z_signed = np.arctan2(z_component, forward_component)

    theta_total = np.arccos(
        np.clip(forward_component, -1.0, 1.0)
    )

    return theta_y_signed, theta_z_signed, theta_total
def _as_scalar(x, name="value"):
    x = np.asarray(x)
    if x.ndim == 0:
        return float(x)
    if x.size == 1:
        return float(x.reshape(()))
    raise ValueError(f"{name} must be scalar, got shape={x.shape}")
def dipole_from_pose(q_src, m_body):
    R = quat_to_R(q_src)
    return R @ m_body  # (3,)

import numpy as np

def cosserat_rhs_single(
    s,
    y,
    *,
    m_src,
    r_src,
    Kinv_fun,
    m_local_fun,
    m_moment,
    wire_len,
    u_star=None,
    include_gravity=False,
    beam_params=None,
):
    """
    Pointwise Cosserat rod RHS for one state y at one arclength s.

    Parameters
    ----------
    s : float
        Arclength location.
    y : (13,) array
        State [p(3), q(4), n(3), m(3)].
    m_src : (3,) array
        Source dipole moment in world frame.
    r_src : (3,) array
        Source position in world frame.
    Kinv_fun : callable
        Returns Kinv(s, wire_len) with shape (3,3,1) or (3,3).
    m_local_fun : callable
        Magnetization profile function.
    m_moment : float or other magnetic parameter
        Passed through to magnetic model.
    wire_len : float
        Wire/tip transition parameter.
    u_star : (3,) array or None
        Reference strain.
    include_gravity : bool
        Whether to include gravity density.
    beam_params : object or None
        Must provide f_g if include_gravity=True.

    Returns
    -------
    dy_ds : (13,) array
    aux : dict
        Useful intermediate quantities for debugging / later Jacobian work.
    """
    e1 = np.array([-1.0, 0.0, 0.0], dtype=float)

    y = np.asarray(y, float).reshape(13,)
    p = y[0:3]
    q = y[3:7]
    n = y[7:10]
    m = y[10:13]

    if u_star is None:
        u_star = np.zeros(3, dtype=float)
    else:
        u_star = np.asarray(u_star, float).reshape(3,)

    qn = quat_normalize(q)

    # quat_to_R should return (3,3) for a single quaternion
    R = quat_to_R(qn)

    # position derivative
    p_s = R @ e1

    # body-frame moment
    m_body = R.T @ m

    # constitutive inverse stiffness
    Kinv = Kinv_fun(np.array([s], dtype=float), wire_len)
    if Kinv.ndim == 3:
        Kinv = Kinv[:, :, 0]
    Kinv = np.asarray(Kinv, float).reshape(3, 3)

    # strain
    u = Kinv @ m_body + u_star

    # quaternion derivative
    q_s = quat_derivative_body(qn, u).reshape(4,)

    # ---- pointwise magnetic wrench density ----

    f_ext, tau_ext, B_here, m_world_here = magnetic_wrench_density_cosserat_point(
        s=s,
        p=p,
        q=qn,
        m_ext=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_front_or_overhead=m_moment,
        r_min=1e-6,
    )

    f_ext = np.asarray(f_ext, float).reshape(3,)
    tau_ext = np.asarray(tau_ext, float).reshape(3,)

    if include_gravity:
        if beam_params is None:
            raise ValueError("beam_params must be provided if include_gravity=True")
        fg = np.asarray(beam_params.f_g, float).reshape(3,)
        f_ext = f_ext + fg

    # force balance
    n_s = -f_ext

    # moment balance
    m_s = -np.cross(p_s, n) - tau_ext

    dy_ds = np.zeros(13, dtype=float)
    dy_ds[0:3] = p_s
    dy_ds[3:7] = q_s
    dy_ds[7:10] = n_s
    dy_ds[10:13] = m_s

    aux = dict(
        p=p,
        q=qn,
        R=R,
        p_s=p_s,
        m_body=m_body,
        Kinv=Kinv,
        u=u,
        f_ext=f_ext,
        tau_ext=tau_ext,
        B=B_here,
        m_world=m_world_here,
    )
    return dy_ds, aux
def dnormalize_dq(q, eps=1e-12):
    """
    Jacobian of q_hat = q / ||q|| wrt q.
    Returns shape (4,4).
    """
    q = np.asarray(q, float).reshape(4,)
    n = np.linalg.norm(q)
    if n < eps:
        raise ValueError("Quaternion norm too small in dnormalize_dq")

    qhat = q / n
    I = np.eye(4)
    return (I - np.outer(qhat, qhat)) / n
def make_cosserat_kirchhoff_ode_vectorized(
    m_src, r_src, Kinv_fun, m_local_fun, m_moment, wire_len, u_star=None,
    include_gravity=False, beam_params=None
):
    def ode(s, Y):
        s = np.asarray(s, float)
        Y = np.asarray(Y, float)

        N = s.size
        dY = np.zeros_like(Y)

        for i in range(N):
            dY[:, i], _ = cosserat_rhs_single(
                s[i],
                Y[:, i],
                m_src=m_src,
                r_src=r_src,
                Kinv_fun=Kinv_fun,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
                wire_len=wire_len,
                u_star=u_star,
                include_gravity=include_gravity,
                beam_params=beam_params,
            )
        return dY

    return ode
def dq_s_du_analytic(q):
    """
    Jacobian of quat_derivative_body(q, u) wrt u.
    Returns shape (4,3).
    """
    q = np.asarray(q, float).reshape(4,)
    qw, qx, qy, qz = q

    J = 0.5 * np.array([
        [-qx, -qy, -qz],
        [ qw, -qz,  qy],
        [ qz,  qw, -qx],
        [-qy,  qx,  qw],
    ], dtype=float)

    return J

def dq_s_dm_analytic(q, Kinv, R):
    """
    Jacobian of q_s wrt world-frame internal moment m.
    Returns shape (4,3).

    q    : (4,)
    Kinv : (3,3)
    R    : (3,3)
    """
    J_qu = dq_s_du_analytic(q)   # (4,3)
    J_um = Kinv @ R.T            # (3,3)
    return J_qu @ J_um           # (4,3)
def rhs_state_jacobian_fd(
    s, y, *,
    m_src, r_src, Kinv_fun, m_local_fun, m_moment, wire_len,
    u_star=None,
    eps=1e-7,
):
    """
    Local Jacobian A = df/dy, shape (13,13)
    """
    y = np.asarray(y, float).reshape(13,)
    f0, _ = cosserat_rhs_single(
        s, y,
        m_src=m_src, r_src=r_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
    )

    A = np.zeros((13, 13), dtype=float)
    for j in range(13):
        yp = y.copy()
        yp[j] += eps
        fp, _ = cosserat_rhs_single(
            s, yp,
            m_src=m_src, r_src=r_src,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )
        A[:, j] = (fp - f0) / eps
    return A
def dnormalize_dq(q, eps=1e-12):
    """
    Jacobian of q_hat = q / ||q|| wrt raw q.
    Returns shape (4,4).
    """
    q = np.asarray(q, float).reshape(4,)
    n = np.linalg.norm(q)
    if n < eps:
        raise ValueError("Quaternion norm too small in dnormalize_dq")
    qhat = q / n
    return (np.eye(4) - np.outer(qhat, qhat)) / n
def Rm_dqhat_analytic(qhat, m_local):
    """
    Jacobian of m_world = R(qhat) @ m_local wrt normalized quaternion qhat=[w,x,y,z].
    Returns shape (3,4).
    """
    qhat = np.asarray(qhat, float).reshape(4,)
    qhat = quat_normalize(qhat)
    w, x, y, z = qhat

    mx, my, mz = np.asarray(m_local, float).reshape(3,)

    dm_dw = np.array([
        -2.0*z*my + 2.0*y*mz,
         2.0*z*mx - 2.0*x*mz,
        -2.0*y*mx + 2.0*x*my,
    ], dtype=float)

    dm_dx = np.array([
         2.0*y*my + 2.0*z*mz,
         2.0*y*mx - 4.0*x*my - 2.0*w*mz,
         2.0*z*mx + 2.0*w*my - 4.0*x*mz,
    ], dtype=float)

    dm_dy = np.array([
        -4.0*y*mx + 2.0*x*my + 2.0*w*mz,
         2.0*x*mx + 2.0*z*mz,
        -2.0*w*mx + 2.0*z*my - 4.0*y*mz,
    ], dtype=float)

    dm_dz = np.array([
        -4.0*z*mx - 2.0*w*my + 2.0*x*mz,
         2.0*w*mx - 4.0*z*my + 2.0*y*mz,
         2.0*x*mx + 2.0*y*my,
    ], dtype=float)

    return np.column_stack([dm_dw, dm_dx, dm_dy, dm_dz])
def dn_s_dq_magnetic_analytic(q_raw, p, r_src, m_src, m_local):
    """
    Magnetic contribution to dn_s/dq, shape (3,4).
    """
    J_fm = dforce_dmworld_analytic(p, r_src, m_src)   # (3,3)
    J_mq = Rm_dq_analytic(q_raw, m_local)             # (3,4)
    return - J_fm @ J_mq
def dtau_dq_magnetic_analytic(q_raw, B_here, m_local):
    """
    Magnetic contribution to d tau_ext / d q, shape (3,4).
    """
    J_mq = Rm_dq_analytic(q_raw, m_local)   # (3,4)
    B_here = np.asarray(B_here, float).reshape(3,)

    J = np.zeros((3,4), dtype=float)
    for k in range(4):
        J[:, k] = np.cross(J_mq[:, k], B_here)
    return J
def dm_s_dq_kinematic_analytic(dp_s_dq, n):
    """
    Kinematic contribution to dm_s/dq from -(p_s x n), shape (3,4).
    """
    n = np.asarray(n, float).reshape(3,)
    J = np.zeros((3,4), dtype=float)
    for k in range(4):
        J[:, k] = -np.cross(dp_s_dq[:, k], n)
    return J
def dforce_dmworld_analytic(p, r_src, m_src, mu0=4*np.pi*1e-7):
    """
    Jacobian d f_ext / d m_world, shape (3,3).
    """
    c = mu0 / (4*np.pi)

    R = np.asarray(p, float).reshape(3,) - np.asarray(r_src, float).reshape(3,)
    m0 = np.asarray(m_src, float).reshape(3,)

    r = np.linalg.norm(R) + 1e-12
    r2 = r * r
    r5 = r2 * r2 * r

    m0R = np.dot(m0, R)
    I = np.eye(3)

    J = 3.0 * c / r5 * (
        np.outer(m0, R) +
        m0R * I +
        np.outer(R, m0) -
        5.0 * m0R / r2 * np.outer(R, R)
    )
    return J

def Rm_dq_analytic(q_raw, m_local):
    """
    Jacobian of m_world = R(normalize(q_raw)) @ m_local wrt raw q.
    Returns shape (3,4).
    """
    q_raw = np.asarray(q_raw, float).reshape(4,)
    qhat = quat_normalize(q_raw)
    J_hat = Rm_dqhat_analytic(qhat, m_local)
    J_norm = dnormalize_dq(q_raw)
    return J_hat @ J_norm

def RTm_dqhat_analytic(qhat, m_world):
    """
    Jacobian of b = R(qhat)^T m_world wrt normalized quaternion qhat=[w,x,y,z].
    Returns shape (3,4).
    """
    qhat = np.asarray(qhat, float).reshape(4,)
    qhat = quat_normalize(qhat)
    w, x, y, z = qhat

    mx, my, mz = np.asarray(m_world, float).reshape(3,)

    db_dw = np.array([
        2.0*z*my - 2.0*y*mz,
        -2.0*z*mx + 2.0*x*mz,
        2.0*y*mx - 2.0*x*my,
    ], dtype=float)

    db_dx = np.array([
        2.0*y*my + 2.0*z*mz,
        2.0*y*mx - 4.0*x*my + 2.0*w*mz,
        2.0*z*mx - 2.0*w*my - 4.0*x*mz,
    ], dtype=float)

    db_dy = np.array([
        -4.0*y*mx + 2.0*x*my - 2.0*w*mz,
        2.0*x*mx + 2.0*z*mz,
        2.0*w*mx + 2.0*z*my - 4.0*y*mz,
    ], dtype=float)

    db_dz = np.array([
        -4.0*z*mx + 2.0*w*my + 2.0*x*mz,
        -2.0*w*mx - 4.0*z*my + 2.0*y*mz,
        2.0*x*mx + 2.0*y*my,
    ], dtype=float)

    return np.column_stack([db_dw, db_dx, db_dy, db_dz])




def du_dqhat_analytic(qhat, Kinv, m_world):
    """
    Jacobian of u wrt normalized quaternion qhat.
    Returns shape (3,4).
    """
    return Kinv @ RTm_dqhat_analytic(qhat, m_world)
def dq_s_dqhat_direct_analytic(u):
    """
    Direct Jacobian of quat_derivative_body(qhat, u) wrt normalized qhat,
    holding u fixed.
    Returns shape (4,4).
    """
    u = np.asarray(u, float).reshape(3,)
    ux, uy, uz = u

    J = 0.5 * np.array([
        [0.0,  -ux,   -uy,   -uz],
        [ux,   0.0,    uz,   -uy],
        [uy,   -uz,   0.0,    ux],
        [uz,    uy,   -ux,   0.0],
    ], dtype=float)

    return J
def dq_s_dq_full_analytic(q_raw, u, Kinv, m_world):
    """
    Full analytic Jacobian dq_s/dq_raw.

    q_raw   : raw quaternion state variable (4,)
    u       : nominal strain computed using normalized quaternion
    Kinv    : (3,3)
    m_world : world-frame internal moment (3,)

    Returns
    -------
    J : (4,4)
    """
    q_raw = np.asarray(q_raw, float).reshape(4,)
    qhat = quat_normalize(q_raw)

    J_norm = dnormalize_dq(q_raw)                 # (4,4)

    J_direct_hat = dq_s_dqhat_direct_analytic(u) # (4,4)
    J_qu = dq_s_du_analytic(qhat)                # (4,3)
    J_uqhat = du_dqhat_analytic(qhat, Kinv, m_world)  # (3,4)

    J_hat = J_direct_hat + J_qu @ J_uqhat        # (4,4)

    return J_hat @ J_norm



def rhs_state_jacobian_analytic(
    s, y, *,
    m_src, r_src, Kinv_fun, m_local_fun, m_moment, wire_len,
    u_star=None,
):
    """
    Fully analytic local state Jacobian A = df/dy, shape (13,13).
    """
    y = np.asarray(y, float).reshape(13,)

    _, aux = cosserat_rhs_single(
        s, y,
        m_src=m_src,
        r_src=r_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
    )

    p = y[0:3]
    q_raw = y[3:7]
    n = y[7:10]
    m_world_internal = y[10:13]

    qhat = aux["q"]
    R = aux["R"]
    Kinv = aux["Kinv"]
    u = aux["u"]
    p_s = aux["p_s"]
    B_here = aux["B"]

    m_local = m_local_fun(np.array([s]), m_moment)[:, 0]

    A = np.zeros((13, 13), dtype=float)

    # -------------------------------------------------
    # Row block 1: p_s
    # -------------------------------------------------
    # dp_s/dp = 0
    A[0:3, 0:3] = 0.0

    # dp_s/dq
    dpdq = dp_s_dq_analytic(q_raw)
    A[0:3, 3:7] = dpdq

    # dp_s/dn = 0
    A[0:3, 7:10] = 0.0

    # dp_s/dm = 0
    A[0:3, 10:13] = 0.0

    # -------------------------------------------------
    # Row block 2: q_s
    # -------------------------------------------------
    # dq_s/dp = 0
    A[3:7, 0:3] = 0.0

    # dq_s/dq
    A[3:7, 3:7] = dq_s_dq_full_analytic(q_raw, u, Kinv, m_world_internal)

    # dq_s/dn = 0
    A[3:7, 7:10] = 0.0

    # dq_s/dm
    A[3:7, 10:13] = dq_s_dm_analytic(qhat, Kinv, R)

    # -------------------------------------------------
    # Row block 3: n_s = -f_ext
    # -------------------------------------------------
    # dn_s/dp
    Bctrl = rhs_control_jacobian_rsrc_analytic(
        s, y,
        m_src=m_src,
        r_src=r_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
    )

    A[7:10, 0:3] = -Bctrl[7:10, :]
    A[10:13, 0:3] = -Bctrl[10:13, :]

    # dn_s/dq
    A[7:10, 3:7] = dn_s_dq_magnetic_analytic(
        q_raw, p, r_src, m_src, m_local
    )

    # dn_s/dn = 0
    A[7:10, 7:10] = 0.0

    # dn_s/dm = 0
    A[7:10, 10:13] = 0.0

    # -------------------------------------------------
    # Row block 4: m_s = -(p_s x n) - tau_ext
    # -------------------------------------------------
    # dm_s/dp

    # dm_s/dq
    A[10:13, 3:7] = (
        dm_s_dq_kinematic_analytic(dpdq, n)
        - dtau_dq_magnetic_analytic(q_raw, B_here, m_local)
    )

    # dm_s/dn
    A[10:13, 7:10] = -skew(p_s)

    # dm_s/dm = 0
    A[10:13, 10:13] = 0.0

    return A
def dminus_psxn_dq(dp_dq, n):
    """
    Contribution to dm_s/dq from -(p_s x n), shape (3,4)
    """
    n = np.asarray(n, float).reshape(3,)
    J = np.zeros((3, 4), dtype=float)
    for k in range(4):
        J[:, k] = -np.cross(dp_dq[:, k], n)
    return J
def rhs_control_jacobian_rsrc_fd(
    s, y, *,
    m_src, r_src, Kinv_fun, m_local_fun, m_moment, wire_len,
    u_star=None,
    eps=1e-7,
):
    """
    Local control Jacobian B = df/dr_src, shape (13,3)
    """
    y = np.asarray(y, float).reshape(13,)
    r_src = np.asarray(r_src, float).reshape(3,)

    f0, _ = cosserat_rhs_single(
        s, y,
        m_src=m_src, r_src=r_src,
        Kinv_fun=Kinv_fun,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        u_star=u_star,
    )

    B = np.zeros((13, 3), dtype=float)
    for j in range(3):
        rp = r_src.copy()
        rp[j] += eps
        fp, _ = cosserat_rhs_single(
            s, y,
            m_src=m_src, r_src=rp,
            Kinv_fun=Kinv_fun,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
            wire_len=wire_len,
            u_star=u_star,
        )
        B[:, j] = (fp - f0) / eps
    return B
def bc_cosserat(Ya, Yb, p0, q0):
    pa = Ya[0:3]
    qa = Ya[3:7]
    nb = Yb[7:10]
    mb = Yb[10:13]
    return np.hstack([
        pa - p0,        
        qa - q0,       
        nb,             
        mb              
    ])
def tip_jacobian_rsrc_fd_full(
    *,
    model_factory,
    L,
    r_src,
    q_src,
    m_body,
    wire_len,
    eps=1e-5,
):
    """
    Full nonlinear finite-difference Jacobian of p_tip wrt r_src.
    model_factory: callable returning a fresh CosseratForwardModel
    """
    model0 = model_factory()
    out0 = model0.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body, wire_len=wire_len)
    p0 = out0["p_tip"]

    J = np.zeros((3, 3), dtype=float)
    for j in range(3):
        rp = np.asarray(r_src, float).copy()
        rp[j] += eps

        modelp = model_factory()
        outp = modelp.forward(L=L, r_src=rp, q_src=q_src, m_body=m_body, wire_len=wire_len)
        J[:, j] = (outp["p_tip"] - p0) / eps

    return J
def skew(v):
    v = np.asarray(v, float).reshape(3,)
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2],  0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)
def force_jacobian_rsrc(p, r_src, m_src, m_world, mu0=4*np.pi*1e-7):
    """
    Analytic Jacobian df_ext / d r_src, shape (3,3),
    for dipole-dipole force with:
        R = p - r_src
        m = m_world
        m0 = m_src
    """
    c = mu0 / (4*np.pi)

    R = np.asarray(p, float).reshape(3,) - np.asarray(r_src, float).reshape(3,)
    m = np.asarray(m_world, float).reshape(3,)
    m0 = np.asarray(m_src, float).reshape(3,)

    r2 = np.dot(R, R) + 1e-24
    r = np.sqrt(r2)
    r4 = r2 * r2
    r5 = r4 * r
    r7 = r5 * r2

    a = np.dot(m, R)       # m · R
    b = np.dot(m0, R)      # m0 · R
    mm0 = np.dot(m, m0)    # m · m0

    # inner vector T so that f = 3 c r^-5 T
    T = (
        a * m0
        + b * m
        + mm0 * R
        - 5.0 * a * b / r2 * R
    )

    I = np.eye(3)

    # dT/dR
    dT_dR = (
        np.outer(m0, m)
        + np.outer(m, m0)
        + mm0 * I
        - 5.0 * (
            (a * b / r2) * I
            + np.outer(R, (b * m + a * m0)) / r2
            - 2.0 * a * b * np.outer(R, R) / r4
        )
    )

    # d(r^-5)/dR contribution
    dF_dR = (
        3.0 * c * (
            dT_dR / r5
            - 5.0 * np.outer(T, R) / r7
        )
    )

    # chain rule: R = p - r_src
    return -dF_dR
def torque_jacobian_rsrc(m_world, dB_drsrc):
    """
    d tau / d r_src
    """
    J = np.zeros((3,3))
    for j in range(3):
        J[:, j] = np.cross(m_world, dB_drsrc[:, j])
    return J
def rhs_control_jacobian_rsrc_analytic(
    s, y, *,
    m_src, r_src, Kinv_fun, m_local_fun, m_moment, wire_len,
    u_star=None,
):
    """
    Fully analytic control Jacobian B = d f / d r_src, shape (13,3).
    """
    y = np.asarray(y, float).reshape(13,)
    p = y[0:3]
    q = quat_normalize(y[3:7])

    # world rotation
    R = quat_to_R(q)

    # local magnetisation -> world
    m_local = m_local_fun(np.array([s]), m_moment)[:, 0]
    m_world = R @ m_local

    # analytic field Jacobian
    dB_drsrc = dipole_field_jacobian_wrt_rsrc(p, r_src, m_src)

    # analytic force Jacobian: df_ext / dr_src
    df_drsrc = force_jacobian_rsrc(p, r_src, m_src, m_world)

    # analytic torque Jacobian: dtau_ext / dr_src
    dtau_drsrc = torque_jacobian_rsrc(m_world, dB_drsrc)

    B = np.zeros((13, 3), dtype=float)

    # n_s = -f_ext
    B[7:10, :] = -df_drsrc

    # m_s = -(p_s x n) - tau_ext
    B[10:13, :] = -dtau_drsrc

    return B
def dipole_field_jacobian_wrt_rsrc(p, r_src, m_src, mu0=4*np.pi*1e-7):
    """
    Returns dB / d r_src, shape (3,3)
    """
    c = mu0 / (4*np.pi)

    R = p - r_src
    r = np.linalg.norm(R) + 1e-12
    r2 = r*r
    r5 = r2 * r2 * r
    r7 = r5 * r2

    m = m_src
    Rm = np.dot(R, m)

    I = np.eye(3)

    # dB/dR (3x3)
    term1 = 3 * Rm * I / r5
    term2 = 3 * np.outer(R, m) / r5
    term3 = 3 * np.outer(m, R) / r5
    term4 = -15 * Rm * np.outer(R, R) / r7

    dB_dR = c * (term1 + term2 + term3 + term4)

    # chain rule: d/d r_src = - d/dR
    return -dB_dR

class CosseratForwardModel:
    def __init__(self, *, p0, q0, Kinv_fun, m_local_fun, m_moment, wire_len,
                 n_nodes=120, tol=1e-5, max_nodes=20000):
        self.p0 = np.asarray(p0, float)
        self.q0 = np.asarray(q0, float)
        self.Kinv_fun = Kinv_fun
        self.m_local_fun = m_local_fun
        self.m_moment = m_moment
        self.n_nodes = int(n_nodes)
        self.tol = float(tol)
        self.max_nodes = int(max_nodes)
        self._sol_prev = None
        self.wire_len = wire_len

    def _initial_guess(self, L):
        s = np.linspace(0.0, L, self.n_nodes)
        Y = np.zeros((13, self.n_nodes))
        Y[0, :] = -s   
        Y[3, :] = 1.0        
        return s, Y


    def solve(self, *, L, r_src, m_src, wire_len, try_energy_selection=False):
        L = float(L)
        r_src = np.asarray(r_src, float).reshape(3,)
        m_src = np.asarray(m_src, float).reshape(3,)

        ode = make_cosserat_kirchhoff_ode_vectorized(
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=self.Kinv_fun,
            m_local_fun=self.m_local_fun,
            m_moment=self.m_moment,
            wire_len=wire_len,
        )

        # ------------------------------------------------------------
        # Standard continuation solve
        # ------------------------------------------------------------
        if not try_energy_selection:
            if self._sol_prev is None or not getattr(self._sol_prev, "success", False):
                s_mesh, Y_guess = self._initial_guess(L)
            else:
                s_mesh = np.linspace(0.0, L, self.n_nodes)
                Y_guess = self._sol_prev.sol(s_mesh)

            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s_mesh,
                Y_guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
            )

            if sol.success:
                self._sol_prev = sol

            return sol

        # ------------------------------------------------------------
        # Energy-selection solve with multiple branch guesses
        # ------------------------------------------------------------
        guesses = []

        # Warm-start branch from previous solution
        if self._sol_prev is not None and getattr(self._sol_prev, "success", False):
            s_mesh = np.linspace(0.0, L, self.n_nodes)
            guesses.append((s_mesh, self._sol_prev.sol(s_mesh), "prev"))

        # Branch-seeking guesses
        s0, Y0 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=0,  m_seed=5e-4)
        s1, Y1 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=+1, m_seed=5e-4)
        s2, Y2 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=-1, m_seed=5e-4)
        s3, Y3 = make_initial_guess(L, self.n_nodes, bend_axis="z", bend_sign=+1, m_seed=5e-4)
        s4, Y4 = make_initial_guess(L, self.n_nodes, bend_axis="z", bend_sign=-1, m_seed=5e-4)

        guesses += [
            (s0, Y0, "straight"),
            (s1, Y1, "+y"),
            (s2, Y2, "-y"),
            (s3, Y3, "+z"),
            (s4, Y4, "-z"),
        ]

        candidates = []

        for s_mesh, Y_guess, tag in guesses:
            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s_mesh,
                Y_guess,
                tol=self.tol,
                max_nodes=self.max_nodes,
            )

            if not sol.success:
                continue

            W, parts = compute_total_energy(
                sol,
                L=L,
                r_src=r_src,
                m_src=m_src,
                Kinv_fun=self.Kinv_fun,
                m_local_fun=self.m_local_fun,
                m_moment=self.m_moment,
                wire_len=wire_len,
                u_star=np.zeros(3),
                contact_penalty_fun=None,
                s_out_n=400,
            )

            candidates.append((float(W), sol, tag, parts))

        if len(candidates) == 0:
            # Final fallback to straight guess
            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s0,
                Y0,
                tol=self.tol,
                max_nodes=self.max_nodes,
            )

            if sol.success:
                self._sol_prev = sol

            return sol

        candidates.sort(key=lambda item: item[0])
        W_best, sol_best, tag_best, parts_best = candidates[0]

        self._sol_prev = sol_best

        self._last_energy_choice = {
            "W_best": W_best,
            "tag_best": tag_best,
            "parts_best": parts_best,
            "all": [
                {"W": W, "tag": tag, "parts": parts}
                for W, _, tag, parts in candidates
            ],
        }

        return sol_best

    def forward(self, *, L, r_src, q_src, m_body, wire_len, s_out_n=300):
        """
        Inputs:
          L: beam length
          r_src: magnet position (3,)
          q_src: magnet orientation quaternion (4,) world frame
          m_body: dipole vector in magnet body frame (3,)

        Outputs: dict with tip pose, bending, field at tip, net F/T, profiles (optional)
        """
        m_src = dipole_from_pose(q_src, m_body)  # (3,)
        sol = self.solve(L=L, r_src=r_src, m_src=m_src, wire_len=wire_len)

        # Evaluate along rod for outputs
        s_out = np.linspace(0.0, float(L), int(s_out_n))
        Y = sol.sol(s_out)
        p = Y[0:3, :]
        q = quat_normalize(Y[3:7, :])

        f_mid, tau_mid, B_mid, m_mid_world, s_mid = magnetic_wrench_density_cosserat_profile_segments(
            p, q, s_out, m_src, r_src, self.m_local_fun, self.m_moment, r_min=1e-6
        )
        # f_ext = f_ext*0
        # tau_ext = tau_ext*0
        # f_ext = f_ext*0
        # tau_ext = tau_ext*0
        Fy = np.trapezoid(f_mid[1, :], s_mid)
        Fz = np.trapezoid(f_mid[2, :], s_mid)
        Ty = np.trapezoid(tau_mid[1, :], s_mid)
        Tz = np.trapezoid(tau_mid[2, :], s_mid)

        F_net = np.trapezoid(f_mid, s_mid, axis=1)
        T_net = np.trapezoid(tau_mid, s_mid, axis=1)

        # Tip values
        p_tip = p[:, -1]
        q_tip = q[:, -1]
        B_tip = float(np.linalg.norm(B_mid[:, -1]))

        # Bending angles from tangent at tip
        theta_y, theta_z, theta_tot = tip_bending_angles_from_tangent(sol, L=float(L))

        return dict(
            solved=bool(sol.success),
            message=str(getattr(sol, "message", "")),
            p_tip=p_tip,
            q_tip=q_tip,
            theta_y=theta_y, theta_z=theta_z, theta_total=theta_tot,
            B_tip=B_tip,
            F_net=F_net,
            T_net=T_net,
            profiles=dict(s=s_mid, p=p, q=q, f_ext=f_mid, tau_ext=tau_mid, B=B_mid),
        )


def quat_mul(q1, q2):
    # Hamilton product, q = q1 ⊗ q2, with q = [w,x,y,z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def   compute_total_energy(sol, *, L, r_src, m_src, Kinv_fun, m_local_fun, m_moment, wire_len,
                         u_star=np.zeros(3), contact_penalty_fun=None, s_out_n=400):
    """
    Returns: (W_total, parts_dict)
      W_total = W_s + W_m + W_v(optional)
    """
    s = np.linspace(0.0, float(L), int(s_out_n))
    Y = sol.sol(s)

    p = Y[0:3, :]                          # (3,N)
    q = quat_normalize(Y[3:7, :])          # (4,N)
    n = Y[7:10, :]                         # (3,N)  (not needed)
    m = Y[10:13, :]                        # (3,N)  moment in world frame (your convention)

    # Rotation along rod
    R = quat_to_rot(q)                     # expected (N,3,3)

    # p_s (tangent) if you ever need it:
    # e1 = np.array([-1.0, 0.0, 0.0])
    # p_s = np.einsum('nij,j->ni', R, e1).T   # (3,N)

    # Convert world moment -> body moment: m_body = R^T m
    m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m)  # (3,N)

    # Strain: u = Kinv * m_body + u_star
    Kinv = Kinv_fun(s, wire_len)  # (3,3,N)
    u = np.einsum('ijn,jn->in', Kinv, m_body) + u_star[:, None]    # (3,N)

    # Elastic energy density (Kirchhoff rod bending/torsion):
    # w_s = 0.5 * (u-u_star) · m_body
    du = u - u_star[:, None]
    w_s = 0.5 * np.sum(du * m_body, axis=0)   # (N,)
    W_s = np.trapezoid(w_s, s)

    # Magnetic potential energy density:
    # w_m = - m_world(s) · B_world(s)
    # Your m_local_fun returns body-frame magnetisation distribution (3,N)
    m_local_body = m_local_fun(s, None)       # (3,N)



    # Field along rod from your existing function
    # (it already uses p,q and magnet pose; B is typically in world frame)
    f_mid, tau_mid, B_mid, m_mid_world, s_mid = magnetic_wrench_density_cosserat_profile_segments(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    # w_m = -np.sum(m_local_world * B, axis=0)  # (N,)
    # W_m = np.trapezoid(w_m, s)
    m_dot_B_mid = np.sum(m_mid_world * B_mid, axis=0)     # (N-1,)
    w_m_mid = -m_dot_B_mid                                 # (N-1,)
    ds = np.diff(s)
    # midpoint quadrature
    W_m = np.sum(w_m_mid * ds)
    # Optional vessel/contact penalty
    W_v = 0.0
    if contact_penalty_fun is not None:
        # Should return a scalar density w_v(s) for each s, shape (N,)
        w_v = contact_penalty_fun(p, q, s)    # (N,)
        W_v = np.trapezoid(w_v, s)

    # after computing p (3,N)
    W_g = -np.trapezoid(np.sum(beam_params.f_g[:, None] * p, axis=0), s)   # scalar
    parts = dict(W_s=W_s, W_m=W_m, W_v=W_v, W_g=W_g)
    W_s = _as_scalar(W_s, "W_s")
    W_m = _as_scalar(W_m, "W_m")
    W_v = _as_scalar(W_v, "W_v")
    W_g = _as_scalar(W_g, "W_g")

    W_total = W_s + W_m + W_g + W_v
    W_total = _as_scalar(W_total, "W_total")
    return W_total, parts



def make_initial_guess(
    L,
    n_nodes,
    bend_axis="y",
    bend_sign=0,
    m_seed=5e-4,
):
    """
    Create a simple initial guess for solve_bvp.

    State layout assumed:
        Y[0:3]   = position
        Y[3:7]   = quaternion, wxyz
        Y[7:10]  = internal force-like unknowns
        Y[10:13] = internal moment-like unknowns

    This guess bends the centerline slightly in local/world y or z.
    """

    s = np.linspace(0.0, L, int(n_nodes))
    xi = s / max(L, 1e-12)

    Y = np.zeros((13, len(s)))

    # Straight beam mainly along -x
    Y[0, :] = -s

    if bend_axis == "y":
        Y[1, :] = bend_sign * m_seed * np.sin(np.pi * xi)
    elif bend_axis == "z":
        Y[2, :] = bend_sign * m_seed * np.sin(np.pi * xi)

    # Unit quaternion, wxyz
    Y[3, :] = 1.0

    return s, Y


def dp_s_dqhat_analytic(qhat):
    """
    Jacobian of p_s = R(qhat) @ e1 wrt normalized quaternion qhat=[w,x,y,z].
    Returns shape (3,4).
    """
    qhat = np.asarray(qhat, float).reshape(4,)
    qhat = quat_normalize(qhat)
    w, x, y, z = qhat

    J = np.array([
        [0.0,    0.0,    4.0*y,   4.0*z],
        [-2.0*z, -2.0*y, -2.0*x, -2.0*w],
        [2.0*y, -2.0*z,  2.0*w, -2.0*x],
    ], dtype=float)
    return J

def dp_s_dq_analytic(q):
    """
    Jacobian of p_s = quat_to_R(quat_normalize(q)) @ e1 wrt raw q.
    Returns shape (3,4).
    """
    q = np.asarray(q, float).reshape(4,)
    qhat = quat_normalize(q)

    J_qhat = dp_s_dqhat_analytic(qhat)   # (3,4)
    J_norm = dnormalize_dq(q)            # (4,4)

    return J_qhat @ J_norm               # (3,4)


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



def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
    """
    L_ins      : commanded insertion (what MPC tracks)
    L_tip_full : physical magnetic tip length (4 cm)
    L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

    Returns (L_model, wire_len, tip_len)
    """
    L_ins = float(L_ins)

    # Magnetised tip inside grows with insertion until full tip is inside
    tip_len = min(L_ins, L_tip_full)

    # Wire is everything beyond the physical tip length
    wire_len = max(L_ins - L_tip_full, 0.0)

    # Total model length is the inserted length, but don't go below minimum model length
    L_model = max(L_ins, L_tip_min)

    # If we are below L_tip_min, we still model a minimum rod,
    # but magnetisation should NOT exceed what's actually inserted:
    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len
import time
import statistics as stats
def plot_initial_magnet_geometry(
    *,
    p0_ur,
    p_tip_straight,
    r_src_nom,
    beam_axis,
    m_src_nom,
    run_dir,
    side_axis=None,
    title="Initial magnet placement",
):
    p0_ur = np.asarray(p0_ur, float).reshape(3,)
    p_tip_straight = np.asarray(p_tip_straight, float).reshape(3,)
    r_src_nom = np.asarray(r_src_nom, float).reshape(3,)
    beam_axis = np.asarray(beam_axis, float).reshape(3,)
    beam_axis = beam_axis / (np.linalg.norm(beam_axis) + 1e-12)

    m_dir = np.asarray(m_src_nom, float).reshape(3,)
    m_dir = m_dir / (np.linalg.norm(m_dir) + 1e-12)

    # ---------- 3D view ----------
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # beam line
    ax.plot(
        [p0_ur[0], p_tip_straight[0]],
        [p0_ur[1], p_tip_straight[1]],
        [p0_ur[2], p_tip_straight[2]],
        "k-o",
        linewidth=2,
        label="Straight beam",
    )

    # magnet position
    ax.scatter(
        r_src_nom[0], r_src_nom[1], r_src_nom[2],
        s=80,
        marker="s",
        label="EPM",
    )

    # vector from tip to magnet
    v_tip_mag = r_src_nom - p_tip_straight
    ax.quiver(
        p_tip_straight[0], p_tip_straight[1], p_tip_straight[2],
        v_tip_mag[0], v_tip_mag[1], v_tip_mag[2],
        length=1.0,
        normalize=False,
        linewidth=2,
        label="Tip to magnet",
    )

    # dipole direction at magnet
    arrow_len = 0.03
    ax.quiver(
        r_src_nom[0], r_src_nom[1], r_src_nom[2],
        arrow_len * m_dir[0], arrow_len * m_dir[1], arrow_len * m_dir[2],
        length=1.0,
        normalize=False,
        linewidth=2,
        label="Dipole direction",
    )

    # beam axis direction from tip
    ax.quiver(
        p_tip_straight[0], p_tip_straight[1], p_tip_straight[2],
        0.03 * beam_axis[0], 0.03 * beam_axis[1], 0.03 * beam_axis[2],
        length=1.0,
        normalize=False,
        linewidth=2,
        label="Beam axis",
    )

    if side_axis is not None:
        side_axis = np.asarray(side_axis, float).reshape(3,)
        side_axis = side_axis / (np.linalg.norm(side_axis) + 1e-12)
        ax.quiver(
            p_tip_straight[0], p_tip_straight[1], p_tip_straight[2],
            0.03 * side_axis[0], 0.03 * side_axis[1], 0.03 * side_axis[2],
            length=1.0,
            normalize=False,
            linewidth=2,
            label="Side axis",
        )

    # equal-ish axes
    pts = np.vstack([p0_ur, p_tip_straight, r_src_nom])
    center = np.mean(pts, axis=0)
    span = np.max(np.ptp(pts, axis=0))
    span = max(span, 0.05)

    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_zlim(center[2] - span / 2, center[2] + span / 2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "initial_magnet_geometry_3d.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------- XY top view ----------
    plt.figure(figsize=(7, 7))
    plt.plot(
        [p0_ur[0], p_tip_straight[0]],
        [p0_ur[1], p_tip_straight[1]],
        "k-o",
        linewidth=2,
        label="Straight beam",
    )
    plt.scatter(r_src_nom[0], r_src_nom[1], s=80, marker="s", label="External magnet")

    plt.arrow(
        r_src_nom[0], r_src_nom[1],
        arrow_len * m_dir[0], arrow_len * m_dir[1],
        head_width=0.003,
        length_includes_head=True,
        label="Dipole direction",
    )

    plt.arrow(
        p_tip_straight[0], p_tip_straight[1],
        0.03 * beam_axis[0], 0.03 * beam_axis[1],
        head_width=0.003,
        length_includes_head=True,
    )

    if side_axis is not None:
        plt.arrow(
            p_tip_straight[0], p_tip_straight[1],
            0.03 * side_axis[0], 0.03 * side_axis[1],
            head_width=0.003,
            length_includes_head=True,
        )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title + " — XY view")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "initial_magnet_geometry_xy.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------- XZ side view ----------
    plt.figure(figsize=(7, 5))
    plt.plot(
        [p0_ur[0], p_tip_straight[0]],
        [p0_ur[2], p_tip_straight[2]],
        "k-o",
        linewidth=2,
        label="straight beam",
    )
    plt.scatter(r_src_nom[0], r_src_nom[2], s=80, marker="s", label="external magnet")

    plt.arrow(
        r_src_nom[0], r_src_nom[2],
        arrow_len * m_dir[0], arrow_len * m_dir[2],
        head_width=0.003,
        length_includes_head=True,
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.title(title + " — XZ view")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "initial_magnet_geometry_xz.png", dpi=300, bbox_inches="tight")
    plt.show()
def make_model_factory(
    *,
    p0_ur,
    q0_ur,
    Kinv_fun,
    wire_len,
    tip_len,
    n_nodes=120,
    tol=1e-5,
    max_nodes=20000,
):
    def factory():
        return CosseratForwardModel(
            p0=p0_ur,
            q0=q0_ur,
            Kinv_fun=Kinv_fun,
            m_local_fun=make_m_local_fun_wire_tip(
                wire_len,
                len_tip=tip_len,
                mode="axial",
                alpha_end=0.0,
                eps=1e-3,
            ),
            m_moment=0.0,
            wire_len=wire_len,
            n_nodes=n_nodes,
            tol=tol,
            max_nodes=max_nodes,
        )
    return factory
def benchmark_function(fn, repeats=5, warmup=1, label="function"):
    """
    Benchmark a zero-argument callable.
    Returns dict with timing statistics and last result.
    """
    # warmup
    last_result = None
    for _ in range(warmup):
        last_result = fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "label": label,
        "times_s": times,
        "mean_s": stats.mean(times),
        "std_s": stats.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
        "last_result": last_result,
    }
def magnetic_net_force_torque_from_solution(
    sol,
    *,
    L,
    r_src,
    m_src,
    m_local_fun,
    m_moment=0.0,
    n_eval=300,
):
    s_eval = np.linspace(0.0, float(L), int(n_eval))
    y = sol.sol(s_eval)

    p = y[0:3, :]      # 3 x n
    q = y[3:7, :]      # 4 x n

    F_net = np.zeros(3)
    T_net = np.zeros(3)

    ds = float(L) / max(int(n_eval) - 1, 1)

    for k, s_k in enumerate(s_eval):
        pk = p[:, k]
        qk = quat_normalize(q[:, k])

        Rk = quat_to_R(qk)

        m_local = np.asarray(m_local_fun(s_k), float).reshape(3,)
        m_beam_world = Rk @ m_local

        B = dipole_field_from_source(
            pk.reshape(1, 3),
            r_src,
            m_src,
        )[0]

        # Magnetic torque density: tau = m x B
        tau = np.cross(m_beam_world, B)

        # Magnetic force density: grad_p(m . B)
        # If you already have an analytic force-density helper, use it here.
        # Otherwise use a small finite difference at this point.
        eps = 1e-6
        f = np.zeros(3)

        for a in range(3):
            dp = np.zeros(3)
            dp[a] = eps

            Bp = dipole_field_from_source(
                (pk + dp).reshape(1, 3),
                r_src,
                m_src,
            )[0]

            Bm = dipole_field_from_source(
                (pk - dp).reshape(1, 3),
                r_src,
                m_src,
            )[0]

            wp = np.dot(m_beam_world, Bp)
            wm = np.dot(m_beam_world, Bm)

            f[a] = (wp - wm) / (2.0 * eps)

        weight = ds
        if k == 0 or k == len(s_eval) - 1:
            weight *= 0.5

        F_net += f * weight
        T_net += tau * weight

    return F_net, T_net
def make_run_dir(base="results", name=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = name or f"run_{timestamp}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
def dipole_field_from_source(r_pts, r_src, m_src, r_min=1e-6):

    R = r_pts - r_src[None, :]
    R2 = np.sum(R*R, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, r_min**2))
    Rhat = R / Rnorm[:, None]
    mdot = Rhat @ m_src  
    invR3 = 1.0 / (Rnorm**3)
    MU0_over_4pi = 1e-7
    B = MU0_over_4pi * invR3[:, None] * (3.0 * mdot[:, None] * Rhat - m_src[None, :])
    return B    
def print_benchmark_result(res):
    print(f"\n[{res['label']}]")
    print(f"mean = {res['mean_s']:.6f} s")
    print(f"std  = {res['std_s']:.6f} s")
    print(f"min  = {res['min_s']:.6f} s")
    print(f"max  = {res['max_s']:.6f} s")

if __name__ == "__main__":
    DEBUG = True

    DO_NOMINAL_CHECK = True
    DO_ORIENTATION_MAP = True
    DO_FD_CHECK = False        # keep False for maps; FD over a grid will be very slow
    run_dir = make_run_dir(
        base="results",
        name=f"Comparison Magnetic field",
    )
    L_cmd = 0.025
    nodes = 10
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
    #     0.7981328220229531-L_cmd, -0.70992731669220016, -0.1+0.25,
    #     np.pi, 0.001, 0.001
    # ], float)
    start_point = np.array([
        0.7981328220229531-(L_cmd+0.12), -0.70992731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)
    # start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
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

    def make_m_local():
        return make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )

    def model_factory():
        return CosseratForwardModel(
            p0=p0_ur,
            q0=q0_ur,
            Kinv_fun=Kinv_fun,
            m_local_fun=make_m_local(),
            m_moment=0.0,
            wire_len=wire_len,
            n_nodes=nodes,
            tol=1e-5,
            max_nodes=20000,
        )

    model = model_factory()

    # ------------------------------------------------------------
    # Base insertion direction / lumen setup
    # ------------------------------------------------------------
    Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
    t0 = Rbase @ np.array([-1.0, 0.0, 0.0])
    beam_tangent_ref = Rbase @ np.array([-1.0, 0.0, 0.0])
    beam_y_ref = Rbase @ np.array([0.0, -1.0, 0.0])
    beam_z_ref = Rbase @ np.array([0.0, 0.0, 1.0])

    beam_tangent_ref = beam_tangent_ref / (np.linalg.norm(beam_tangent_ref) + 1e-12)
    beam_y_ref = beam_y_ref / (np.linalg.norm(beam_y_ref) + 1e-12)
    beam_z_ref = beam_z_ref / (np.linalg.norm(beam_z_ref) + 1e-12)

    global INSERTION_DIR_WORLD
    INSERTION_DIR_WORLD = t0 / (np.linalg.norm(t0) + 1e-12)


    # ============================================================
    # Beam axis and straight-tip reference
    # ============================================================
    beam_axis = Rbase @ np.array([-1.0, 0.0, 0.0])
    beam_axis_y = Rbase @ np.array([0.0, -1.0, 0.0])
    beam_axis = beam_axis / (np.linalg.norm(beam_axis) + 1e-12)

    p_tip_straight = p0_ur + L_cmd * beam_axis


    COMPARE_PLACEMENTS = True

    NOMINAL_PLACEMENT_NAME = "side_90"   # used only for nominal/debug geometry
    def rotate_dipole_about_world_z(m0, z_angle):
        Rz = Rot.from_euler("z", z_angle, degrees=False).as_matrix()
        return Rz @ np.asarray(m0, float).reshape(3,)
    # ============================================================
    # Dipole magnitude and nominal dipole alignment
    # ============================================================
    m_src_from_pose = dipole_from_pose(q_src_ur, m_body)
    m_mag = np.linalg.norm(m_src_from_pose)

    # Force nominal source dipole to align with the beam axis.
    # Use +beam_axis or -beam_axis depending on desired magnetic polarity.
    # m_src_nom = -m_mag * beam_axis
    # m_src_zero = m_src_nom.copy()
    m_src_base = m_mag * beam_axis   # or +m_mag * beam_axis depending on polarity

    initial_dipole_zrot_deg = 0.0
    initial_dipole_zrot = np.deg2rad(initial_dipole_zrot_deg)

    m_src_nom = rotate_dipole_about_world_z(m_src_base, initial_dipole_zrot)
    m_src_zero = m_src_nom.copy()


    # ============================================================
    # Placement option A: directly above beam tip
    # ============================================================
    above_distance = 0.2  # 20 cm above tip

    def source_position_above_tip(distance_above_tip):
        return p_tip_straight + np.array([0.0, 0.0, float(distance_above_tip)])

    # ============================================================
    # Placement option B: directly in front of beam tip
    # ============================================================
    front_distance = 0.2 # 12 cm in front of tip, along beam axis

    front_axis = beam_axis.copy()
    front_axis = front_axis / (np.linalg.norm(front_axis) + 1e-12)

    def source_position_in_front(distance_from_tip):
        return p_tip_straight + float(distance_from_tip) * front_axis
    # ============================================================
    # Placement option C: same plane, 90 degrees from beam axis
    # ============================================================
    side_distance = 0.2 # 20 cm to the side of the beam tip

    # Use the beam direction projected into the world xy plane
    beam_axis_xy = beam_axis.copy()
    beam_axis_xy[2] = 0.0
    beam_axis_xy = beam_axis_xy / (np.linalg.norm(beam_axis_xy) + 1e-12)

    # +90 degree rotation about world z gives a side direction in the same xy plane
    Rz_30 = Rot.from_euler("z", np.deg2rad(-30.0), degrees=False).as_matrix()
    side_axis_30 = Rz_30 @ beam_axis_xy
    side_axis_30[2] = 0.0
    side_axis_30 = side_axis_30 / (np.linalg.norm(side_axis_30) + 1e-12)

    def source_position_side_30(distance_from_tip):
        return p_tip_straight + float(distance_from_tip) * side_axis_30
    
    Rz_60 = Rot.from_euler("z", np.deg2rad(-60.0), degrees=False).as_matrix()
    side_axis_60 = Rz_60 @ beam_axis_xy
    side_axis_60[2] = 0.0
    side_axis_60 = side_axis_60 / (np.linalg.norm(side_axis_60) + 1e-12)

    def source_position_side_60(distance_from_tip):
        return p_tip_straight + float(distance_from_tip) * side_axis_60

    # ============================================================
    # Fixed-distance placement definitions
    # ============================================================
    placement_specs = {
        # "above": {
        #     "label": "Above tip",
        #     "distance_m": above_distance,
        #     "source_position_fun": source_position_above_tip,
        #     "m_src_base": -m_mag * beam_axis,
        # },
        "front": {
            "label": "In front",
            "distance_m": front_distance,
            "source_position_fun": source_position_in_front,
            "m_src_base": m_mag * beam_axis,
        },
        "side_90": {
            "label": "Side 30 deg",
            "distance_m": side_distance,
            "source_position_fun": source_position_side_30,
            "m_src_base": m_mag * beam_axis,
        },
        "side_60": {
            "label": "Side 60 deg",
            "distance_m": side_distance,
            "source_position_fun": source_position_side_60,
            "m_src_base": m_mag * beam_axis,
        },
    }

    nominal_spec = placement_specs[NOMINAL_PLACEMENT_NAME]
    dist_nom = nominal_spec["distance_m"]
    r_src_nom = nominal_spec["source_position_fun"](dist_nom)
    # ============================================================
    # Select map distance grid and nominal source position
    # ============================================================
    zrot_grid = np.deg2rad(np.linspace(-180.0, 180.0, 20))

    expected_front = p_tip_straight + front_distance * beam_axis
    actual_vec = r_src_nom - p_tip_straight
    expected_vec = front_distance * beam_axis


    # ============================================================
    # Nominal solve
    # ============================================================
    # plot_initial_magnet_geometry(
    #     p0_ur=p0_ur,
    #     p_tip_straight=p_tip_straight,
    #     r_src_nom=r_src_nom,
    #     beam_axis=beam_axis,
    #     m_src_nom=m_src_nom,
    #     side_axis= None,
    #     run_dir=run_dir,
    #     title="Initial external magnet placement",
    # )
    sol_nom = model.solve(
        L=L_cmd,
        r_src=r_src_nom,
        m_src=m_src_zero,
        wire_len=wire_len,
    )

    p_tip_nom = np.asarray(sol_nom.y[0:3, -1], float)

    B_tip_nom = dipole_field_from_source(
        p_tip_nom.reshape(1, 3),
        r_src_nom,
        m_src_zero,
    )[0]

    print("Nominal tip:", p_tip_nom)
    print("Nominal B_tip norm [T]:", np.linalg.norm(B_tip_nom))
    print("Nominal B_tip norm [mT]:", 1e3 * np.linalg.norm(B_tip_nom))
    print("m_src_zero dot beam_axis:", np.dot(m_src_zero, beam_axis))

    p_tip_nom = np.asarray(sol_nom.y[0:3, -1], float)

    B_tip_nom = dipole_field_from_source(
        p_tip_nom.reshape(1, 3),
        r_src_nom,
        m_src_zero,
    )[0]

    print("Nominal tip:", p_tip_nom)
    print("Nominal B_tip norm [T]:", np.linalg.norm(B_tip_nom))
    print("Nominal B_tip norm [mT]:", 1e3 * np.linalg.norm(B_tip_nom))
    print("m_src_zero dot beam_axis:", np.dot(m_src_zero, beam_axis))

    print("Beam axis:", beam_axis)
    print("Straight tip:", p_tip_straight)
    print("Nominal magnet position:", r_src_nom)
    print("Nominal tip:", p_tip_nom)
    print("Nominal B_tip norm [T]:", np.linalg.norm(B_tip_nom))
    print("Nominal B_tip norm [mT]:", 1e3 * np.linalg.norm(B_tip_nom))
    print("m_src_zero dot beam_axis:", np.dot(m_src_zero, beam_axis))

    # ------------------------------------------------------------
    # Z-rotation / distance map section
    # ------------------------------------------------------------
    if DO_ORIENTATION_MAP:

        def quat_wxyz_to_R(q_wxyz):
            return Rot.from_quat([
                q_wxyz[1],
                q_wxyz[2],
                q_wxyz[3],
                q_wxyz[0],
            ])

        def R_to_quat_wxyz(R):
            q_xyzw = R.as_quat()
            return np.array([
                q_xyzw[3],
                q_xyzw[0],
                q_xyzw[1],
                q_xyzw[2],
            ], dtype=float)

        def rotate_source_quat_world_z(q_src_wxyz, z_angle):
            """
            Apply a world-frame rotation about the global z axis.

            q_new = dq_z_world * q_src
            """
            R_src = quat_wxyz_to_R(q_src_wxyz)
            dR_world = Rot.from_euler("z", z_angle, degrees=False)
            R_new = dR_world * R_src
            return R_to_quat_wxyz(R_new)

        def closest_point_on_straight_beam(p, p0, t):
            """
            Closest point from p to the nominal straight beam line.

            p0 : beam base
            t  : beam tangent direction, unit vector
            """
            p = np.asarray(p, float).reshape(3,)
            p0 = np.asarray(p0, float).reshape(3,)
            t = np.asarray(t, float).reshape(3,)
            t = t / (np.linalg.norm(t) + 1e-12)

            lam = np.dot(p - p0, t)
            return p0 + lam * t

        # ------------------------------------------------------------
        # Define nominal beam-to-magnet radial direction
        # ------------------------------------------------------------
        # Beam tangent direction: pivot -> tip
        # ------------------------------------------------------------
        # Axial magnet placement and aligned dipole convention
        # ------------------------------------------------------------


        def rotate_dipole_about_world_z(m0, z_angle):
            Rz = Rot.from_euler("z", z_angle, degrees=False).as_matrix()
            return Rz @ np.asarray(m0, float).reshape(3,)
        if DO_NOMINAL_CHECK:
            J_pose_L_nom, diag_nom = solve_tip_sensitivity_source_dipole_length_shooting(
                sol_nom,
                L=L_cmd,
                r_src=r_src_nom,
                m_src=m_src_zero,
                dm_src_dphi=-skew(m_src_zero),
                Kinv_fun=Kinv_fun,
                m_moment=model.m_moment,
                u_star=np.zeros(3),
                n_eval=200,
                rtol=1e-6,
                atol=1e-8,
            )

            print("\nNominal J_tip wrt [r_src, delta_phi, L]:")
            print(J_pose_L_nom)
        # ------------------------------------------------------------
        # New grid:
        #   horizontal axis = z rotation angle
        #   vertical axis   = magnet distance from beam
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # New comparison sweep:
        #   horizontal axis = dipole z rotation angle
        #   one curve per magnet placement
        #   distance is fixed for each placement
        # ------------------------------------------------------------
        zrot_grid = np.deg2rad(np.linspace(-180.0, 180.0, 101))
        zrot_deg = np.rad2deg(zrot_grid)

        placement_names = list(placement_specs.keys())
        # placement_name = "side_90"
        # spec = placement_specs[placement_name]

        # placement_label = spec["label"]
        # dist_mag = spec["distance_m"]
        # r_src_i = spec["source_position_fun"](dist_mag)
        # m_src_nom_i = spec["m_src_base"]

        # zrot_grid_forward = np.deg2rad(np.linspace(20.0, 60.0, 161))
        # zrot_grid_backward = np.deg2rad(np.linspace(60.0, 20.0, 161))

        # forward_rows = []
        # backward_rows = []

        # # ============================================================
        # # Forward continuation sweep
        # # ============================================================
        # model_forward = model_factory()

        # for j, zrot in enumerate(zrot_grid_forward):
        #     m_src_rot = rotate_dipole_about_world_z(m_src_nom_i, zrot)

        #     sol_ij = model_forward.solve(
        #         L=L_cmd,
        #         r_src=r_src_i,
        #         m_src=m_src_rot,
        #         wire_len=wire_len,
        #         try_energy_selection=False,
        #     )

        #     if not sol_ij.success:
        #         print(f"[FORWARD] solve failed at zrot={np.rad2deg(zrot):.2f} deg")
        #         continue

        #     theta_y, theta_z, theta_total = tip_bending_angles_from_tangent_signed(
        #         sol_ij,
        #         L=float(L_cmd),
        #         beam_tangent_ref=beam_tangent_ref,
        #         beam_y_ref=beam_y_ref,
        #         beam_z_ref=beam_z_ref,
        #     )

        #     W_ij, parts_ij = compute_total_energy(
        #         sol_ij,
        #         L=L_cmd,
        #         r_src=r_src_i,
        #         m_src=m_src_rot,
        #         Kinv_fun=Kinv_fun,
        #         m_local_fun=model_forward.m_local_fun,
        #         m_moment=model_forward.m_moment,
        #         wire_len=wire_len,
        #         u_star=np.zeros(3),
        #         contact_penalty_fun=None,
        #         s_out_n=400,
        #     )

        #     forward_rows.append({
        #         "direction": "forward",
        #         "zrot_deg": float(np.rad2deg(zrot)),
        #         "theta_y_deg": float(np.rad2deg(theta_y)),
        #         "theta_z_deg": float(np.rad2deg(theta_z)),
        #         "theta_total_deg": float(np.rad2deg(theta_total)),
        #         "energy": float(W_ij),
        #         "success": bool(sol_ij.success),
        #     })


        # # ============================================================
        # # Backward continuation sweep
        # # ============================================================
        # model_backward = model_factory()

        # for j, zrot in enumerate(zrot_grid_backward):
        #     m_src_rot = rotate_dipole_about_world_z(m_src_nom_i, zrot)

        #     sol_ij = model_backward.solve(
        #         L=L_cmd,
        #         r_src=r_src_i,
        #         m_src=m_src_rot,
        #         wire_len=wire_len,
        #         try_energy_selection=False,
        #     )

        #     if not sol_ij.success:
        #         print(f"[BACKWARD] solve failed at zrot={np.rad2deg(zrot):.2f} deg")
        #         continue

        #     theta_y, theta_z, theta_total = tip_bending_angles_from_tangent_signed(
        #         sol_ij,
        #         L=float(L_cmd),
        #         beam_tangent_ref=beam_tangent_ref,
        #         beam_y_ref=beam_y_ref,
        #         beam_z_ref=beam_z_ref,
        #     )

        #     W_ij, parts_ij = compute_total_energy(
        #         sol_ij,
        #         L=L_cmd,
        #         r_src=r_src_i,
        #         m_src=m_src_rot,
        #         Kinv_fun=Kinv_fun,
        #         m_local_fun=model_backward.m_local_fun,
        #         m_moment=model_backward.m_moment,
        #         wire_len=wire_len,
        #         u_star=np.zeros(3),
        #         contact_penalty_fun=None,
        #         s_out_n=400,
        #     )

        #     backward_rows.append({
        #         "direction": "backward",
        #         "zrot_deg": float(np.rad2deg(zrot)),
        #         "theta_y_deg": float(np.rad2deg(theta_y)),
        #         "theta_z_deg": float(np.rad2deg(theta_z)),
        #         "theta_total_deg": float(np.rad2deg(theta_total)),
        #         "energy": float(W_ij),
        #         "success": bool(sol_ij.success),
        #     })


        # # ============================================================
        # # Save and plot once, after both sweeps
        # # ============================================================
        # df_hyst = pd.DataFrame(forward_rows + backward_rows)
        # df_hyst.to_csv(run_dir / "side_hysteresis_forward_backward.csv", index=False)

        # plt.figure(figsize=(8, 5))

        # for direction, group in df_hyst.groupby("direction"):
        #     group_plot = group.sort_values("zrot_deg")

        #     plt.plot(
        #         group_plot["zrot_deg"],
        #         group_plot["theta_y_deg"],
        #         marker="o",
        #         linewidth=2,
        #         label=direction,
        #     )

        # plt.xlabel("External magnet dipole rotation about world z [deg]")
        # plt.ylabel(r"$\theta_y$ [deg]")
        # plt.title("Forward/backward continuation: signed beam bending")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(
        #     run_dir / "hysteresis_theta_y_forward_backward.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # plt.show()


        # plt.figure(figsize=(8, 5))

        # for direction, group in df_hyst.groupby("direction"):
        #     group_plot = group.sort_values("zrot_deg")

        #     plt.plot(
        #         group_plot["zrot_deg"],
        #         group_plot["energy"],
        #         marker="o",
        #         linewidth=2,
        #         label=direction,
        #     )

        # plt.xlabel("External magnet dipole rotation about world z [deg]")
        # plt.ylabel(r"$\Pi$")
        # plt.title("Forward/backward continuation: total potential energy")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(
        #     run_dir / "hysteresis_energy_forward_backward.png",
        #     dpi=300,
        #     bbox_inches="tight",
        # )
        # plt.show()

        n_place = len(placement_names)
        n_rot = len(zrot_grid)

        # Maps are now placement x rotation, not distance x rotation
        Jcond_xyz_map = np.full((n_place, n_rot), np.nan)
        Jcond_xy_map = np.full((n_place, n_rot), np.nan)

        manip_xyz_map = np.full((n_place, n_rot), np.nan)
        manip_xy_map = np.full((n_place, n_rot), np.nan)

        sigma1_xyz_map = np.full((n_place, n_rot), np.nan)
        sigma2_xyz_map = np.full((n_place, n_rot), np.nan)
        sigma3_xyz_map = np.full((n_place, n_rot), np.nan)

        sigma1_xy_map = np.full((n_place, n_rot), np.nan)
        sigma2_xy_map = np.full((n_place, n_rot), np.nan)

        theta_y_map = np.full((n_place, n_rot), np.nan)
        theta_z_map = np.full((n_place, n_rot), np.nan)
        theta_total_map = np.full((n_place, n_rot), np.nan)

        Bnorm_tip_map = np.full((n_place, n_rot), np.nan)
        Fnorm_map = np.full((n_place, n_rot), np.nan)
        Tnorm_map = np.full((n_place, n_rot), np.nan)

        Jnorm_map = np.full((n_place, n_rot), np.nan)
        Jtrans_norm_map = np.full((n_place, n_rot), np.nan)
        Jrot_norm_map = np.full((n_place, n_rot), np.nan)
        Bx_tip_map = np.full((n_place, n_rot), np.nan)
        By_tip_map = np.full((n_place, n_rot), np.nan)
        Bz_tip_map = np.full((n_place, n_rot), np.nan)
        tip_disp_y_map = np.full((n_place, n_rot), np.nan)
        tip_disp_z_map = np.full((n_place, n_rot), np.nan)
        tip_disp_total_map = np.full((n_place, n_rot), np.nan)


        energy_map = np.full((n_place, n_rot), np.nan)
        rows = []

        for i, placement_name in enumerate(placement_names):
            
            spec = placement_specs[placement_name]

            placement_label = spec["label"]
            dist_mag = spec["distance_m"]
            r_src_i = spec["source_position_fun"](dist_mag)
            m_src_nom_i = spec["m_src_base"]
            print(f"\n[PLACEMENT SWEEP] {placement_label}")
            print("Distance [mm]:", 1e3 * dist_mag)
            print("Magnet position:", r_src_i)
            print("Tip -> magnet [mm]:", 1e3 * (r_src_i - p_tip_straight))

            for j, zrot in enumerate(zrot_grid):
                # Rotate only the dipole, not the magnet position
                m_src_rot = rotate_dipole_about_world_z(m_src_nom_i, zrot)

                model_ij = model_factory()

                try:
                    sol_ij = model_ij.solve(
                        L=L_cmd,
                        r_src=r_src_i,
                        m_src=m_src_rot,
                        wire_len=wire_len,
                    )
                    W_ij, parts_ij = compute_total_energy(
                        sol_ij,
                        L=L_cmd,
                        r_src=r_src_i,
                        m_src=m_src_rot,
                        Kinv_fun=Kinv_fun,
                        m_local_fun=model_ij.m_local_fun,
                        m_moment=model_ij.m_moment,
                        wire_len=wire_len,
                        u_star=np.zeros(3),
                        contact_penalty_fun=None,
                        s_out_n=300,
                    )

                    energy_map[i, j] = W_ij
                    theta_y, theta_z, theta_total = tip_bending_angles_from_tangent_signed(
                        sol_ij,
                        L=float(L_cmd),
                        beam_tangent_ref=beam_tangent_ref,
                        beam_y_ref=beam_y_ref,
                        beam_z_ref=beam_z_ref,
                    )

                    theta_y_map[i, j] = np.rad2deg(theta_y)
                    theta_z_map[i, j] = np.rad2deg(theta_z)
                    theta_total_map[i, j] = np.rad2deg(theta_total)

                    p_tip = np.asarray(sol_ij.y[0:3, -1], dtype=float)

                    B_tip = dipole_field_from_source(
                        p_tip.reshape(1, 3),
                        r_src_i,
                        m_src_rot,
                    )[0]
                    Bx_tip_map[i, j] = B_tip[0]
                    By_tip_map[i, j] = B_tip[1]
                    Bz_tip_map[i, j] = B_tip[2]
                    Bnorm_tip_map[i, j] = np.linalg.norm(B_tip)

                    F_net, T_net = magnetic_net_force_torque_from_solution(
                        sol_ij,
                        L=L_cmd,
                        r_src=r_src_i,
                        m_src=m_src_rot,
                        m_local_fun=model_ij.m_local_fun,
                        m_moment=model_ij.m_moment,
                        n_eval=300,
                    )

                    Fnorm_map[i, j] = np.linalg.norm(F_net)
                    Tnorm_map[i, j] = np.linalg.norm(T_net)

                    J_pose_L_ij, diag_ij = solve_tip_sensitivity_source_dipole_length_shooting(
                        sol_ij,
                        L=L_cmd,
                        r_src=r_src_i,
                        m_src=m_src_rot,
                        dm_src_dphi=-skew(m_src_rot),
                        Kinv_fun=Kinv_fun,
                        m_moment=model_ij.m_moment,
                        u_star=np.zeros(3),
                        n_eval=200,
                        rtol=1e-6,
                        atol=1e-8,
                    )

                    J = J_pose_L_ij

                    Jnorm_map[i, j] = np.linalg.norm(J)
                    Jtrans_norm_map[i, j] = np.linalg.norm(J[:, 0:3])
                    Jrot_norm_map[i, j] = np.linalg.norm(J[:, 3:6])

                    # -------------------------------------------------
                    # Full 3D pose Jacobian conditioning
                    # -------------------------------------------------
                    J_pose_xyz = J[:, 0:6]

                    U_xyz, s_xyz, Vt_xyz = np.linalg.svd(
                        J_pose_xyz,
                        full_matrices=False,
                    )

                    Jcond_xyz_map[i, j] = s_xyz[0] / max(s_xyz[-1], 1e-12)
                    manip_xyz_map[i, j] = np.prod(s_xyz)

                    sigma1_xyz_map[i, j] = s_xyz[0]
                    sigma2_xyz_map[i, j] = s_xyz[1]
                    sigma3_xyz_map[i, j] = s_xyz[2]

                    # -------------------------------------------------
                    # Planar xy pose Jacobian conditioning
                    # -------------------------------------------------
                    J_pose_xy = J_pose_xyz[0:2, :]

                    U_xy, s_xy, Vt_xy = np.linalg.svd(
                        J_pose_xy,
                        full_matrices=False,
                    )

                    Jcond_xy_map[i, j] = s_xy[0] / max(s_xy[-1], 1e-12)
                    manip_xy_map[i, j] = np.prod(s_xy)

                    sigma1_xy_map[i, j] = s_xy[0]
                    sigma2_xy_map[i, j] = s_xy[1]

                    rows.append({
                        "placement": placement_name,
                        "placement_label": placement_label,
                        "distance_m": float(dist_mag),
                        "distance_mm": float(1e3 * dist_mag),
                        "zrot_rad": float(zrot),
                        "zrot_deg": float(np.rad2deg(zrot)),

                        "Bnorm_tip_T": float(Bnorm_tip_map[i, j]),
                        "Fnorm_N": float(Fnorm_map[i, j]),
                        "Tnorm_Nm": float(Tnorm_map[i, j]),

                        "J_norm": float(Jnorm_map[i, j]),
                        "Jtrans_norm": float(Jtrans_norm_map[i, j]),
                        "Jrot_norm": float(Jrot_norm_map[i, j]),

                        "cond_xyz": float(Jcond_xyz_map[i, j]),
                        "manip_xyz": float(manip_xyz_map[i, j]),
                        "sigma1_xyz": float(sigma1_xyz_map[i, j]),
                        "sigma2_xyz": float(sigma2_xyz_map[i, j]),
                        "sigma3_xyz": float(sigma3_xyz_map[i, j]),

                        "cond_xy": float(Jcond_xy_map[i, j]),
                        "manip_xy": float(manip_xy_map[i, j]),
                        "sigma1_xy": float(sigma1_xy_map[i, j]),
                        "sigma2_xy": float(sigma2_xy_map[i, j]),

                        "theta_y_deg": float(theta_y_map[i, j]),
                        "theta_z_deg": float(theta_z_map[i, j]),
                        "theta_total_deg": float(theta_total_map[i, j]),
                    })

                except Exception as exc:
                    print(
                        f"Failed at placement={placement_label}, "
                        f"zrot={np.rad2deg(zrot):.1f} deg, "
                        f"distance={dist_mag:.4f} m: {exc}"
                    )
                    continue
        
        
        df_maps = pd.DataFrame(rows)
        df_maps.to_csv(run_dir / "placement_zrot_comparison.csv", index=False)

        np.savez(
            run_dir / "placement_zrot_comparison.npz",
            placement_names=np.array(placement_names),
            placement_labels=np.array([placement_specs[name]["label"] for name in placement_names]),
            zrot_grid=zrot_grid,
            zrot_deg=zrot_deg,

            Jcond_xyz_map=Jcond_xyz_map,
            Jcond_xy_map=Jcond_xy_map,

            manip_xyz_map=manip_xyz_map,
            manip_xy_map=manip_xy_map,

            sigma1_xyz_map=sigma1_xyz_map,
            sigma2_xyz_map=sigma2_xyz_map,
            sigma3_xyz_map=sigma3_xyz_map,

            sigma1_xy_map=sigma1_xy_map,
            sigma2_xy_map=sigma2_xy_map,

            theta_y_map=theta_y_map,
            theta_z_map=theta_z_map,
            theta_total_map=theta_total_map,

            Bnorm_tip_map=Bnorm_tip_map,
            Fnorm_map=Fnorm_map,
            Tnorm_map=Tnorm_map,
        )
        def plot_placement_curves(
            Y,
            title,
            ylabel,
            save_path,
            log10=False,
        ):
            plt.figure(figsize=(8, 5))

            for i, placement_name in enumerate(placement_names):
                label = placement_specs[placement_name]["label"]
                y = np.asarray(Y[i, :], dtype=float)

                if log10:
                    y = np.log10(np.maximum(y, 1e-16))

                plt.plot(
                    zrot_deg,
                    y,
                    marker="o",
                    linewidth=2,
                    label=label,
                )

            plt.xlabel("External magnet dipole rotation about world z [deg]")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()
            
        plot_placement_curves(
            Jcond_xyz_map,
            "3D pose Jacobian conditioning vs dipole rotation",
            r"$\log_{10}\kappa(J_{xyz})$",
            run_dir / "compare_cond_xyz_vs_zrot.png",
            log10=True,
        )

        plot_placement_curves(
            Jcond_xy_map,
            "Planar xy pose Jacobian conditioning vs dipole rotation",
            r"$\log_{10}\kappa(J_{xy})$",
            run_dir / "compare_cond_xy_vs_zrot.png",
            log10=True,
        )

        plot_placement_curves(
            sigma3_xyz_map,
            "Weakest 3D singular value vs dipole rotation",
            r"$\log_{10}\sigma_3(J_{xyz})$",
            run_dir / "compare_sigma3_xyz_vs_zrot.png",
            log10=True,
        )

        plot_placement_curves(
            sigma2_xy_map,
            "Weakest planar xy singular value vs dipole rotation",
            r"$\log_{10}\sigma_2(J_{xy})$",
            run_dir / "compare_sigma2_xy_vs_zrot.png",
            log10=True,
        )

        plot_placement_curves(
            theta_total_map,
            "Total beam tip bending angle vs dipole rotation",
            r"$\theta_{\mathrm{total}}$ [deg]",
            run_dir / "compare_theta_total_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            Bnorm_tip_map,
            "Magnetic field magnitude at beam tip vs dipole rotation",
            r"$\|B_{\mathrm{tip}}\|$ [T]",
            run_dir / "compare_Bnorm_tip_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            Fnorm_map,
            "Net magnetic force magnitude vs dipole rotation",
            r"$\|F_{\mathrm{net}}\|$ [N]",
            run_dir / "compare_Fnorm_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            Tnorm_map,
            "Net magnetic torque magnitude vs dipole rotation",
            r"$\|T_{\mathrm{net}}\|$ [N m]",
            run_dir / "compare_Tnorm_vs_zrot.png",
            log10=False,
        )
        plot_placement_curves(
            Bx_tip_map,
            "Tip magnetic field x-component vs dipole rotation",
            r"$B_x$ [T]",
            run_dir / "compare_Bx_tip_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            By_tip_map,
            "Tip magnetic field y-component vs dipole rotation",
            r"$B_y$ [T]",
            run_dir / "compare_By_tip_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            Bz_tip_map,
            "Tip magnetic field z-component vs dipole rotation",
            r"$B_z$ [T]",
            run_dir / "compare_Bz_tip_vs_zrot.png",
            log10=False,
        )
        plot_placement_curves(
            theta_y_map,
            "Signed beam bending toward local y vs dipole rotation",
            r"$\theta_y$ [deg]",
            run_dir / "compare_theta_y_signed_vs_zrot.png",
            log10=False,
        )

        plot_placement_curves(
            theta_z_map,
            "Signed beam bending toward local z vs dipole rotation",
            r"$\theta_z$ [deg]",
            run_dir / "compare_theta_z_signed_vs_zrot.png",
            log10=False,
        )
        plot_placement_curves(
            energy_map,
            "Total potential energy vs dipole rotation",
            r"$\Pi$",
            run_dir / "compare_energy_vs_zrot.png",
            log10=False,
        )