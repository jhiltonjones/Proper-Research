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

beam_params = default_beam_params()
mag_params = default_magnet_params()
L_tip_full=0.04
def rotate_body_xy(v, yaw_deg):
    """
    Rotate a body-frame vector about the local body z-axis.
    Positive yaw uses right-hand rule.
    """
    a = np.deg2rad(yaw_deg)
    Rz = np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a),  np.cos(a), 0.0],
        [0.0,        0.0,       1.0],
    ], dtype=float)
    return Rz @ np.asarray(v, dtype=float).reshape(3,)
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
def predicted_centerline_robot_to_base_local_mm(
    centerline_robot_m,
    beam_base_point_robot_m,
    pivot_pose6,
):
    pts = np.asarray(centerline_robot_m, dtype=float).reshape(-1, 3)
    base_local = robot_point_to_pivot_local(
        np.asarray(beam_base_point_robot_m, dtype=float),
        np.asarray(pivot_pose6, dtype=float),
    )

    out = []
    for p_robot in pts:
        p_local = robot_point_to_pivot_local(p_robot, pivot_pose6)
        p_base_local = p_local - base_local
        out.append(1e3 * p_base_local)

    return np.asarray(out, dtype=float)
def pivot_rotation_matrix(pivot_pose6: np.ndarray) -> np.ndarray:
    T = ur_pose6_to_T(np.asarray(pivot_pose6, dtype=float).reshape(6,))
    return T[:3, :3]


def robot_point_to_pivot_local(p_robot, pivot_pose6):
    p_robot = np.asarray(p_robot, dtype=float).reshape(3,)
    p_pivot = np.asarray(pivot_pose6[:3], dtype=float).reshape(3,)
    R_pivot = pivot_rotation_matrix(pivot_pose6)
    return R_pivot.T @ (p_robot - p_pivot)
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

# def contact_barrier_energy_and_force_fast(
#     p, lumen_query: LumenQuery, *,
#     Kc=5e3,
#     d_tilde=5e-4,
#     eps=1e-9,
#     penalize_outside=True,
#     k_out=2e3,
#     pen_switch=5e-4,
#     k_hard=2e5,
#     window=3
# ):
#     p = np.asarray(p, float)
#     N = p.shape[1]
#     C = np.zeros(N, float)
#     F = np.zeros((3, N), float)
#     d_arr = np.zeros(N, float)

#     for j in range(N):
#         x = p[:, j]
#         delta, Rloc, q_closest = lumen_query.closest(x, window=window)
#         d = Rloc - delta
#         d_arr[j] = d

#         if delta > eps:
#             n = (x - q_closest) / delta
#         else:
#             n = np.array([1.0, 0.0, 0.0])

#         if (d > 0.0) and (d < d_tilde):
#             log_term = np.log(max(d, eps) / d_tilde)
#             Cj = -Kc * (d - d_tilde)**2 * log_term
#             C[j] = Cj

#             dC_dd = -Kc * (
#                 2.0 * (d - d_tilde) * log_term
#                 + (d - d_tilde)**2 / max(d, eps)
#             )
#             F[:, j] = dC_dd * n

#         elif d >= d_tilde:
#             pass

#         else:
#             if penalize_outside:
#                 pen = -d

#                 if pen <= pen_switch:
#                     C[j] = 0.5 * k_out * pen**2
#                     F[:, j] = -(k_out * pen) * n
#                 else:
#                     dp = pen - pen_switch
#                     C0 = 0.5 * k_out * pen_switch**2
#                     F0 = k_out * pen_switch

#                     C[j] = C0 + F0 * dp + 0.5 * k_hard * dp**2
#                     F[:, j] = -(F0 + k_hard * dp) * n

#     return C, F, d_arr
def contact_barrier_energy_and_force_fast(
    p, lumen_query: LumenQuery, *,
    r_beam=0.0,
    k_contact=3e5,
    pen_switch=5e-4,
    k_hard=3e5,
    eps=1e-12,
    window=3,
    smooth=False,
    smooth_eps=1e-7,
    debug=False,
):
    p = np.asarray(p, float)
    N = p.shape[1]

    C = np.zeros(N, float)
    F = np.zeros((3, N), float)
    gap_arr = np.zeros(N, float)

    for j in range(N):
        x = p[:, j]

        delta, Rloc, q_closest = lumen_query.closest(x, window=window)

        if delta > eps:
            n = (x - q_closest) / delta
        else:
            n = np.array([1.0, 0.0, 0.0])

        gap = Rloc - delta - r_beam
        phi = -gap

        gap_arr[j] = gap

        if smooth:
            phi_pos = 0.5 * (phi + np.sqrt(phi * phi + smooth_eps * smooth_eps))
            dphi_pos_dphi = 0.5 * (1.0 + phi / np.sqrt(phi * phi + smooth_eps * smooth_eps))

            C[j] = 0.5 * k_contact * phi_pos**2
            F[:, j] = -(k_contact * phi_pos * dphi_pos_dphi) * n

        else:
            if phi <= 0.0:
                if debug and gap < 0.0:
                    print("[CONTACT DBG] impossible branch: gap < 0 but phi <= 0")
                continue

            if phi <= pen_switch:
                C[j] = 0.5 * k_contact * phi**2
                F[:, j] = -(k_contact * phi) * n

            else:
                dp = phi - pen_switch
                C0 = 0.5 * k_contact * pen_switch**2
                F0 = k_contact * pen_switch

                C[j] = C0 + F0 * dp + 0.5 * k_hard * dp**2
                F[:, j] = -(F0 + k_hard * dp) * n

        # if gap < 0.0:
        #     print(
        #         "[CONTACT DBG]",
        #         "j=", j,
        #         "delta[mm]=", 1e3 * delta,
        #         "Rloc[mm]=", 1e3 * Rloc,
        #         "r_beam[mm]=", 1e3 * r_beam,
        #         "gap[mm]=", 1e3 * gap,
        #         "phi[mm]=", 1e3 * phi,
        #         "|F|=", np.linalg.norm(F[:, j]),
        #         "n=", n,
        #         "F=", F[:, j],
        #     )
        #     to_center = q_closest - x
        #     to_center_unit = to_center / (np.linalg.norm(to_center) + 1e-12)

        #     print(
        #         "[CONTACT SIGN DBG]",
        #         "j=", j,
        #         "x=", x,
        #         "q_closest=", q_closest,
        #         "n_out=", n,
        #         "to_center_unit=", to_center_unit,
        #         "F_unit=", F[:, j] / (np.linalg.norm(F[:, j]) + 1e-12),
        #         "dot(F, to_center_unit)=",
        #         float(np.dot(F[:, j], to_center_unit)),
        #     )
    return C, F, gap_arr



def lumen_violation_profile(p, lumen_C, lumen_R):
    """
    p: (3,N)
    returns viol: (N,) where viol = d(p(s), centerline) - R_local(s)
    """
    N = p.shape[1]
    viol = np.zeros(N)
    for j in range(N):
        d, i, t, _ = point_to_polyline_distance(p[:, j], lumen_C)
        rloc = interpolate_radius(lumen_R, i, t)
        viol[j] = d - rloc
    return viol

def get_centerline_bvp(sol, s):
    Y = sol.sol(s)
    p = Y[0:3, :]          # (3,N)
    return p


def interpolate_radius(R, i, t):
    # linear interpolation of radius along segment i->i+1
    return (1.0 - t) * R[i] + t * R[i+1]

def tip_bending_angles_from_tangent(sol, L, e1=np.array([-1.0,0.0,0.0])):
    YL = sol.sol(np.array([L]))
    qL = quat_normalize(YL[3:7, :])
    RL = quat_to_rot(qL)[0]
    tL = RL @ e1
    theta_y = np.arctan2(tL[1], tL[0])
    theta_z = np.arctan2(tL[2], tL[0])
    theta_total = np.arccos(np.clip(tL[0] / np.linalg.norm(tL), -1.0, 1.0))
    return theta_y, theta_z, theta_total

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

def make_cosserat_kirchhoff_ode(m_src, r_src, Kinv_fun, m_local_fun,m_moment, wire_len,  u_star=None):
    e1 = np.array([-1.0, 0.0, 0.0])
    if u_star is None:
        u_star = np.zeros(3)

    def ode(s, Y):
        p = Y[0:3]
        q = Y[3:7]
        n = Y[7:10]
        m = Y[10:13]

        qn = quat_normalize(q)
        R = quat_to_rot(qn)

        p_s = np.einsum('nij,j->ni', R, e1).T

        m_body = np.einsum('nij,jn->in', np.transpose(R,(0,2,1)), m)  # (3,N)
        Kinv = Kinv_fun(s, wire_len)                                           # (3,3,N)
        u = np.einsum('ijn,jn->in', Kinv, m_body) + u_star[:,None]    # (3,N)

        q_s = quat_derivative_body(qn, u)

        f_ext, tau_ext, _B,_,_ = magnetic_wrench_density_cosserat_profile_segments(
            p, qn, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
        )

        # Robust shape fix
        if f_ext.shape[1] == p.shape[1] - 1:
            f_ext = mid_to_node_values(f_ext)
        if tau_ext.shape[1] == p.shape[1] - 1:
            tau_ext = mid_to_node_values(tau_ext)
        # f_wall = wall_force_density(p, vessel_centerline, R_vessel, k_wall=k_wall)
        # f_ext = f_ext + f_wall
        include_gravity = False
        if include_gravity == True:
            fg = np.asarray(beam_params.f_g, float)
            if fg.ndim == 1:
                f_ext = f_ext + fg[:, None]   # (3,N) + (3,1)
            else:
                f_ext = f_ext + fg            # if you intentionally supply (3,N)
        # f_ext = f_ext + f_g

        # f_ext = np.zeros_like(p)       # same shape as p (3,N)
        # tau_ext = np.zeros_like(p)     # (3,N)
        n_s = -f_ext
        m_s = -(np.cross(p_s.T, n.T).T) - tau_ext

        dY = np.zeros_like(Y)
        dY[0:3] = p_s
        dY[3:7] = q_s
        dY[7:10] = n_s
        dY[10:13] = m_s
        return dY

    return ode
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

        ode = make_cosserat_kirchhoff_ode(
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=self.Kinv_fun,
            m_local_fun=self.m_local_fun,
            m_moment=self.m_moment,
            wire_len=wire_len,
        )

        if not try_energy_selection:
            if self._sol_prev is None:
                s_mesh, Y_guess = self._initial_guess(L)
            else:
                s_mesh = np.linspace(0.0, L, self.n_nodes)
                Y_guess = self._sol_prev.sol(s_mesh)

            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s_mesh, Y_guess,
                tol=self.tol,
                max_nodes=self.max_nodes
            )
            self._sol_prev = sol
            return sol

        guesses = []

        # Warm-start guess from previous solution (best for continuity)
        if self._sol_prev is not None and self._sol_prev.success:
            s_mesh = np.linspace(0.0, L, self.n_nodes)
            guesses.append((s_mesh, self._sol_prev.sol(s_mesh), "prev"))

        # Add branch-seeking guesses (straight, +y bend, -y bend)
        s0, Y0 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=0,  m_seed=5e-4)
        s1, Y1 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=+1, m_seed=5e-4)
        s2, Y2 = make_initial_guess(L, self.n_nodes, bend_axis="y", bend_sign=-1, m_seed=5e-4)
        guesses += [(s0, Y0, "straight"), (s1, Y1, "+y"), (s2, Y2, "-y")]

        # Optionally also try z-bending branches (useful if your steering plane varies)
        s3, Y3 = make_initial_guess(L, self.n_nodes, bend_axis="z", bend_sign=+1, m_seed=5e-4)
        s4, Y4 = make_initial_guess(L, self.n_nodes, bend_axis="z", bend_sign=-1, m_seed=5e-4)
        guesses += [(s3, Y3, "+z"), (s4, Y4, "-z")]

        candidates = []
        for (s_mesh, Y_guess, tag) in guesses:
            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s_mesh, Y_guess,
                tol=self.tol,
                max_nodes=self.max_nodes
            )
            if not sol.success:
                continue

            W, parts = compute_total_energy(
                sol,
                L=L, r_src=r_src, m_src=m_src,
                Kinv_fun=self.Kinv_fun,
                m_local_fun=self.m_local_fun,
                m_moment=self.m_moment,
                wire_len=wire_len,
                u_star=np.zeros(3),  # adjust if you use u_star
                contact_penalty_fun=None,
                s_out_n=400
            )
            print("\n[ENERGY PARTS DEBUG]")
            print("W_total =", W)
            print("parts =", parts)
            candidates.append((float(W), sol, tag, parts))

        if len(candidates) == 0:
            # Fall back: return last attempted solve (or raise)
            # Here: do one standard solve from straight guess
            sol = solve_bvp(
                lambda s, Y: ode(s, Y),
                lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
                s0, Y0,
                tol=self.tol,
                max_nodes=self.max_nodes
            )
            self._sol_prev = sol
            return sol

        # Choose the minimum-energy solution
        candidates.sort(key=lambda t: t[0])
        W_best, sol_best, tag_best, parts_best = candidates[0]

        # Store for warm-starting next iteration
        self._sol_prev = sol_best

        # Optional: store diagnostics for debugging
        self._last_energy_choice = dict(
            W_best=W_best, tag_best=tag_best, parts_best=parts_best,
            all=[dict(W=W, tag=tag, parts=parts) for (W, _, tag, parts) in candidates]
        )

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

            print(
                f"COSSERAT"
                f"N={N:3d}, success={info['success']}, nit={info['nit']:4d}, "
                f"W0={info['W0']:.6e}, W={info['W']:.6e}, dW={info['dW']:.6e}, "
                f"W_el={parts['W_el']:.6e}, W_m={parts['W_m']:.6e}, W_cf={parts['W_cf']:.6e}, "
                f"||u||={np.linalg.norm(u_opt):.6e}, ||u-u_init||={du_init:.6e}"
            )

            # CRITICAL: carry optimized solution forward
            u0 = u_opt
            N_prev = N

        hist.append(dict(L=L_model, p=p, q=q, info=info, len_wire=len_wire, len_tip=len_tip))
        L += dL

    return hist
def compare_common_metrics(
    s, p_bvp, q_bvp, p_em, q_em, *,
    m_src, r_src, m_local_fun, m_moment
):
    """
    Compare two solutions on the same arclength grid `s` using the same
    post-processing magnetic quadrature and geometric error metrics.

    Parameters
    ----------
    s : (N,) array
        Common node arclength grid.
    p_bvp, p_em : (3,N) arrays
        Centerlines of BVP and energy-min solutions.
    q_bvp, q_em : (4,N) arrays
        Quaternions along rod for BVP and energy-min solutions.
    """
    s = np.asarray(s, float).ravel()
    p_bvp = np.asarray(p_bvp, float)
    p_em  = np.asarray(p_em, float)
    q_bvp = quat_normalize(np.asarray(q_bvp, float))
    q_em  = quat_normalize(np.asarray(q_em, float))

    if p_bvp.shape != p_em.shape:
        raise ValueError(f"p_bvp shape {p_bvp.shape} != p_em shape {p_em.shape}")
    if q_bvp.shape != q_em.shape:
        raise ValueError(f"q_bvp shape {q_bvp.shape} != q_em shape {q_em.shape}")
    if p_bvp.shape[1] != s.size:
        raise ValueError(f"p_bvp has {p_bvp.shape[1]} nodes but s has {s.size}")
    if q_bvp.shape[1] != s.size:
        raise ValueError(f"q_bvp has {q_bvp.shape[1]} nodes but s has {s.size}")

    # Midpoint magnetic quantities, same evaluator for both solutions
    f_b, tau_b, B_b, m_b, s_mid_b = magnetic_wrench_density_cosserat_profile_segments(
        p_bvp, q_bvp, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )
    f_e, tau_e, B_e, m_e, s_mid_e = magnetic_wrench_density_cosserat_profile_segments(
        p_em, q_em, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    if not np.allclose(s_mid_b, s_mid_e):
        raise ValueError("s_mid grids differ unexpectedly")

    ds = np.diff(s)

    # Common midpoint magnetic energy
    Wm_b = float(np.sum(-np.sum(m_b * B_b, axis=0) * ds))
    Wm_e = float(np.sum(-np.sum(m_e * B_e, axis=0) * ds))

    # Common net magnetic wrench
    F_b = np.sum(f_b * ds[None, :], axis=1)
    F_e = np.sum(f_e * ds[None, :], axis=1)
    T_b = np.sum(tau_b * ds[None, :], axis=1)
    T_e = np.sum(tau_e * ds[None, :], axis=1)

    # Geometry errors
    tip_err = np.linalg.norm(p_bvp[:, -1] - p_em[:, -1])
    centerline_err = np.linalg.norm(p_bvp - p_em, axis=0)

    # Tangent error
    dp_b = np.diff(p_bvp, axis=1)
    dp_e = np.diff(p_em, axis=1)
    t_b = dp_b / (np.linalg.norm(dp_b, axis=0, keepdims=True) + 1e-12)
    t_e = dp_e / (np.linalg.norm(dp_e, axis=0, keepdims=True) + 1e-12)
    tan_err = np.linalg.norm(t_b - t_e, axis=0)

    print("\n" + "-" * 80)
    print("COMMON POST-PROCESS COMPARISON")
    print("-" * 80)
    print(f"Wm_bvp            = {Wm_b:.12e}")
    print(f"Wm_energy         = {Wm_e:.12e}")
    print(f"|ΔWm|             = {abs(Wm_b - Wm_e):.12e}")
    print(f"|ΔF_net|          = {np.linalg.norm(F_b - F_e):.12e}")
    print(f"|ΔT_net|          = {np.linalg.norm(T_b - T_e):.12e}")
    print(f"tip error         = {tip_err:.12e}")
    print(f"max centerline    = {np.max(centerline_err):.12e}")
    print(f"mean centerline   = {np.mean(centerline_err):.12e}")
    print(f"max tangent err   = {np.max(tan_err):.12e}")
    print(f"mean tangent err  = {np.mean(tan_err):.12e}")

    return dict(
        Wm_bvp=Wm_b,
        Wm_energy=Wm_e,
        dWm=abs(Wm_b - Wm_e),
        F_bvp=F_b,
        F_energy=F_e,
        T_bvp=T_b,
        T_energy=T_e,
        tip_err=tip_err,
        centerline_err=centerline_err,
        tangent_err=tan_err,
        s_mid=s_mid_b,
        B_bvp=B_b,
        B_energy=B_e,
        m_bvp=m_b,
        m_energy=m_e,
    )
def print_comparison_setup(*, L_model, wire_len, tip_len, p0, q0, r_src, q_src, m_src):
    print("\n" + "=" * 80)
    print("COMPARISON SETUP CHECK")
    print("=" * 80)
    print(f"L_model   = {float(L_model):.12e}")
    print(f"wire_len  = {float(wire_len):.12e}")
    print(f"tip_len   = {float(tip_len):.12e}")
    print(f"p0        = {np.asarray(p0, float)}")
    print(f"q0        = {np.asarray(q0, float)}")
    print(f"r_src     = {np.asarray(r_src, float)}")
    print(f"q_src     = {np.asarray(q_src, float)}")
    print(f"m_src     = {np.asarray(m_src, float)}")
def compare_strains(u_bvp_seg, u_em_seg):
    du = u_bvp_seg - u_em_seg
    print("\n" + "-"*80)
    print("STRAIN COMPARISON")
    print("-"*80)
    print(f"max |Δtwist| = {np.max(np.abs(du[:,0])):.12e}")
    print(f"max |Δbend1| = {np.max(np.abs(du[:,1])):.12e}")
    print(f"max |Δbend2| = {np.max(np.abs(du[:,2])):.12e}")
    print(f"rms strain diff = {np.sqrt(np.mean(du**2)):.12e}")
def recover_segment_strain_from_bvp(sol, *, L, wire_len, Kinv_fun, u_star=np.zeros(3), N=12):
    s = np.linspace(0.0, float(L), int(N))
    Y = sol.sol(s)
    q = quat_normalize(Y[3:7, :])
    m_world = Y[10:13, :]
    R = quat_to_rot(q)
    m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m_world)
    Kinv = Kinv_fun(s, wire_len)
    u_nodes = np.einsum('ijn,jn->in', Kinv, m_body) + np.asarray(u_star)[:, None]
    u_seg = 0.5 * (u_nodes[:, :-1] + u_nodes[:, 1:]).T
    return s, u_seg
def compare_common_metrics(s, p_bvp, q_bvp, p_em, q_em, *, m_src, r_src, m_local_fun, m_moment):
    # midpoint magnetic comparison
    f_b, tau_b, B_b, m_b, s_mid_b = magnetic_wrench_density_cosserat_profile_segments(
        p_bvp, q_bvp, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )
    f_e, tau_e, B_e, m_e, s_mid_e = magnetic_wrench_density_cosserat_profile_segments(
        p_em, q_em, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    ds = np.diff(s)
    Wm_b = float(np.sum(-np.sum(m_b * B_b, axis=0) * ds))
    Wm_e = float(np.sum(-np.sum(m_e * B_e, axis=0) * ds))

    tip_err = np.linalg.norm(p_bvp[:, -1] - p_em[:, -1])
    centerline_err = np.linalg.norm(p_bvp - p_em, axis=0)

    print("\n" + "-"*80)
    print("COMMON POST-PROCESS COMPARISON")
    print("-"*80)
    print(f"Wm_bvp          = {Wm_b:.12e}")
    print(f"Wm_energy       = {Wm_e:.12e}")
    print(f"|ΔWm|           = {abs(Wm_b - Wm_e):.12e}")
    print(f"tip error       = {tip_err:.12e}")
    print(f"max centerline  = {np.max(centerline_err):.12e}")
    print(f"mean centerline = {np.mean(centerline_err):.12e}")
def print_comparison_setup(*, L_model, wire_len, tip_len, p0, q0, r_src, q_src, m_src):
    print("\n" + "="*80)
    print("COMPARISON SETUP CHECK")
    print("="*80)
    print(f"L_model   = {L_model:.12e}")
    print(f"wire_len  = {wire_len:.12e}")
    print(f"tip_len   = {tip_len:.12e}")
    print(f"p0        = {np.asarray(p0)}")
    print(f"q0        = {np.asarray(q0)}")
    print(f"r_src     = {np.asarray(r_src)}")
    print(f"q_src     = {np.asarray(q_src)}")
    print(f"m_src     = {np.asarray(m_src)}")
def format_array_for_paste(name, arr, per_line=6):
    """
    Print an array in a pasteable Python format.
    """
    arr = np.asarray(arr, float).reshape(-1)
    print(f"{name} = np.array([")
    for i in range(0, len(arr), per_line):
        chunk = ", ".join(f"{x:.16e}" for x in arr[i:i+per_line])
        print(f"    {chunk},")
    print("], dtype=float)")
def make_initial_guess(L, n_nodes, *, bend_axis="y", bend_sign=0, m_seed=5e-4):
    """
    bend_sign: 0 (straight), +1, -1
    m_seed: magnitude of seed moment in *world guess coordinates*.
            Start small. Increase if solutions collapse to same branch.
    """
    s = np.linspace(0.0, float(L), int(n_nodes))
    Y = np.zeros((13, s.size))

    # Straight centerline along x
    Y[0, :] = s
    Y[1, :] = 0.0
    Y[2, :] = 0.0

    # Identity quaternion (w=1)
    Y[3, :] = 1.0
    Y[4, :] = 0.0
    Y[5, :] = 0.0
    Y[6, :] = 0.0

    # Seed internal moment guess
    if bend_sign != 0:
        if bend_axis == "y":
            Y[11, :] = bend_sign * m_seed  # m_y guess
        elif bend_axis == "z":
            Y[12, :] = bend_sign * m_seed  # m_z guess
        else:
            raise ValueError("bend_axis must be 'y' or 'z'")

    return s, Y
from scipy.optimize import minimize

def quat_exp_body(u, ds):
    """Quaternion exponential for body strain u over step ds."""
    u = np.asarray(u, float).reshape(3,)
    th = np.linalg.norm(u) * ds
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], float)
    axis = u / (np.linalg.norm(u) + 1e-12)
    h = 0.5 * th
    return np.array([np.cos(h), *(np.sin(h) * axis)], float)

def integrate_pq_from_u(u_flat, *, p0, q0, s, e1=np.array([-1.0,0.0,0.0])):
    """
    u_flat: shape (3*(N-1),)
    returns p: (3,N), q: (4,N), u_seg: (3,N-1)
    """
    s = np.asarray(s, float)
    N = s.size
    ds = np.diff(s)
    u_seg = np.asarray(u_flat, float).reshape(len(s)-1, 3)

    p = np.zeros((3, N), float)
    q = np.zeros((4, N), float)
    p[:, 0] = np.asarray(p0, float).reshape(3,)
    q[:, 0] = quat_normalize(np.asarray(q0, float).reshape(4,))

    for i in range(N-1):
        R = quat_to_R(q[:, i])
        p[:, i+1] = p[:, i] + ds[i] * (R @ e1)

        dq = quat_exp_body(u_seg[i], ds[i])
        q[:, i+1] = quat_normalize(quat_mul(q[:, i], dq))

    return p, q, u_seg
def effective_wire_and_tip(L_ins, *, L_tip_full=0.04, L_tip_stub=0.01):
    L_ins = float(L_ins)
    if L_ins < L_tip_full:
        # dead-zone: only a 1cm tip, no wire
        return 0.0, L_tip_stub, L_tip_stub   # (len_wire, len_tip, L_model)
    else:
        len_wire = L_ins - L_tip_full
        len_tip  = L_tip_full
        L_model  = len_wire + len_tip        # = L_ins
        return len_wire, len_tip, L_model
def u_flat_to_seg(u_flat, N):
    return np.asarray(u_flat, float).reshape(N - 1, 3)

def u_seg_to_flat(u_seg):
    return np.asarray(u_seg, float).reshape(-1)

def interp_u_flat_between_meshes(u_flat_coarse, L, N_from, N_to):
    u_seg_from = u_flat_to_seg(u_flat_coarse, N_from)  # (N_from-1, 3)

    s_from = np.linspace(0.0, float(L), N_from)
    s_from_mid = 0.5 * (s_from[:-1] + s_from[1:])

    s_to = np.linspace(0.0, float(L), N_to)
    s_to_mid = 0.5 * (s_to[:-1] + s_to[1:])

    u_seg_to = np.zeros((N_to - 1, 3), dtype=float)
    for k in range(3):
        u_seg_to[:, k] = np.interp(s_to_mid, s_from_mid, u_seg_from[:, k])

    return u_seg_to_flat(u_seg_to)
def precompute_K_segments(s, Kinv_fun, wire_len):
    """
    Returns K_seg: (N-1, 3, 3) stiffness per segment midpoint.
    You provide Kinv_fun(s, wire_len) -> (3,3,N).
    """
    s = np.asarray(s, float)
    smid = 0.5*(s[:-1] + s[1:])
    Kinv = Kinv_fun(smid, wire_len)  # (3,3,N-1) if your function respects input
    # ensure shape (3,3,N-1)
    if Kinv.shape[-1] != smid.size:
        # if your Kinv_fun always returns (3,3,N) for node-based s, call with nodes then slice midpoints
        raise ValueError(f"Kinv_fun returned shape {Kinv.shape}, expected last dim {smid.size}")

    K_seg = np.zeros((smid.size, 3, 3), float)
    for i in range(smid.size):
        K_seg[i] = np.linalg.inv(Kinv[:, :, i])
    return K_seg

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
    # W_cf = 0.0
    # min_d = np.nan
    # if use_lumen and (lumen_query is not None):
    #     C_nodes, F_nodes, d_nodes = contact_barrier_energy_and_force_fast(
    #         p, lumen_query,
    #         Kc=5,
    #         d_tilde=5e-4,
    #         eps=1e-9,
    #         penalize_outside=True,
    #         k_out=1e5,
    #         pen_switch=5e-4,
    #         k_hard=3e5,
    #         window=3,
    #     )
    #     W_cf = (s[1] - s[0]) * float(np.sum(C_nodes))
    #     min_d = float(np.min(d_nodes))
    # -------------------------
    # Contact / lumen
    # -------------------------
    W_cf = 0.0
    min_gap = np.nan
    n_active_contact = 0

    if use_lumen and (lumen_query is not None):
        C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
            p,
            lumen_query,
            r_beam=beam_params.r,      # or whatever your beam radius variable is
            k_contact=1e8,
            pen_switch=5e-5,
            k_hard=1e10,
            eps=1e-12,
            window=3,
            smooth=True,              # start with nonsmooth pure contact
            smooth_eps=1e-7,
        )

        # Better than assuming uniform spacing
        W_cf = np.trapezoid(C_nodes, s)

        min_gap = float(np.min(gap_nodes))
        n_active_contact = int(np.sum(gap_nodes < 0.0))
        # print("min surface gap [mm] =", 1e3 * np.min(gap_nodes))
        # print("active penetration nodes =", np.sum(gap_nodes < 0.0))
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
        if use_lumen and (lumen_query is not None):
            # print(f"min clearance d={min_d:.6e}")
            print(f"min surface gap={min_gap:.6e}")
            print(f"active contact nodes={n_active_contact}")

    parts = dict(
        W_el=float(W_el),
        W_b=float(W_b),
        W_t=float(W_t),
        W_m=float(W_m),
        W_g=float(W_g),
        W_cf=float(W_cf),

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
        gap_nodes=np.asarray(gap_nodes, float).copy() if use_lumen and lumen_query is not None else None,
        contact_active_nodes=n_active_contact,
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
def contact_barrier_energy_and_force(
    p, lumen_C, lumen_R, *,
    Kc=1e-2,          # stiffness parameter (tune!)
    d_tilde=5e-4,     # barrier distance (m)
    eps=1e-9,         # numerical safety
    penalize_outside=True,
    k_out=5e4         # fallback penalty if d <= 0
):
    """
    Implements paper Eq. (34)-(36): barrier energy based on clearance d = R - delta.
    p: (3,N) node positions.

    Returns:
      C: (N,)   barrier energy per node (sum these, not integrate)
      F: (3,N)  contact force per node (world), from analytic gradient (optional use)
      d: (N,)   clearance d_j
    """
    p = np.asarray(p, float)
    N = p.shape[1]
    C = np.zeros(N, float)
    F = np.zeros((3, N), float)
    d_arr = np.zeros(N, float)

    for j in range(N):
        x = p[:, j]
        delta, i, t, q_closest = point_to_polyline_distance(x, lumen_C)
        Rloc = interpolate_radius(lumen_R, i, t)

        d = Rloc - delta   # clearance to wall
        d_arr[j] = d

        # outward radial unit vector from centerline to point
        if delta > eps:
            n = (x - q_closest) / delta
        else:
            n = np.array([1.0, 0.0, 0.0])

        # --- barrier region: 0 < d < d_tilde ---
        if (d > 0.0) and (d < d_tilde):
            # C(d) = -Kc * (d - d_tilde)^2 * log(d/d_tilde)
            log_term = np.log(max(d, eps) / d_tilde)
            Cj = -Kc * (d - d_tilde)**2 * log_term
            C[j] = Cj

            # dC/dd = -Kc * [ 2(d-dt)*log(d/dt) + (d-dt)^2*(1/d) ]
            dC_dd = -Kc * (2.0*(d - d_tilde)*log_term + (d - d_tilde)**2 * (1.0 / max(d, eps)))

            # Force: F = -∂C/∂x = (dC/dd) * n  because d = R - delta, ∂d/∂x = -n
            F[:, j] = dC_dd * n

        elif d >= d_tilde:
            # no contact energy/force
            pass

        else:
            # d <= 0: outside/penetrating -> barrier undefined; add robust fallback
            if penalize_outside:
                pen = -d  # penetration amount
                C[j] = 0.5 * k_out * pen * pen
                # penalty force pushes inward (toward centerline): -k_out*pen * n
                F[:, j] = -k_out * pen * n

    return C, F, d_arr
def solve_energy_min_3d(
    *, p0, q0, L, wire_len, Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    N=60, u0_flat=None, maxiter=300,
    lumen_C=None, lumen_R=None, use_lumen=True,
    contact_k=3e5, contact_beta=50.0, contact_delta=5e-4,
    contact_mode="tip", contact_s_on=0.0, contact_s_off=0.0
):
    """
    Minimise Π(u) over full segment strain vector u_flat, shape (3*(N-1),).

    This is closer to a DER-style discretisation than the reduced K-control-point version.
    """
    s = np.linspace(0.0, float(L), int(N))
    n_seg = N - 1
    K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

    # -------------------------
    # Initial guess
    # -------------------------
    if u0_flat is None:
        u0_flat = 1e-3 * np.random.randn(3 * n_seg)
    else:
        u0_flat = np.asarray(u0_flat, float).reshape(-1)

        # if warm start is from another discretisation, try to adapt
        if u0_flat.size != 3 * n_seg:
            raise ValueError(
                f"u0_flat has size {u0_flat.size}, expected {3*n_seg} for N={N}"
            )

    # optional scaling to improve conditioning
    u_scale = 30.0
    z0 = u0_flat / u_scale

    lumen_query = None
    if lumen_C is not None and lumen_R is not None:
        lumen_query = LumenQuery(lumen_C, lumen_R)

    # -------------------------
    # Objective in scaled vars
    # -------------------------
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
            lumen_query=lumen_query, use_lumen=use_lumen,
            contact_k=contact_k,
            contact_beta=contact_beta,
            contact_delta=contact_delta,
            contact_mode=contact_mode,
            contact_s_on=contact_s_on,
            contact_s_off=contact_s_off,
            debug_mag=False,
        )
        return W
    W0 = obj(z0)
    res = minimize(
        obj,
        z0,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            ftol=1e-12,
            gtol=1e-10,
            eps=1e-8,
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
        lumen_query=lumen_query, use_lumen=use_lumen,
        contact_k=contact_k,
        contact_beta=contact_beta,
        contact_delta=contact_delta,
        contact_mode=contact_mode,
        contact_s_on=contact_s_on,
        contact_s_off=contact_s_off,
        debug_mag=False,
    )
    W_final = float(W)
    info = dict(
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit),
        W=float(W),
        parts=parts,
        s=s,
        W0=float(W0),
        dW=float(W_final - W0),
        u_flat_opt=u_flat_opt.copy(),
        z_opt=np.asarray(res.x, float).copy(),
        max_bend=float(np.max(np.linalg.norm(u_seg[:, 1:3], axis=1))),
        mean_bend=float(np.mean(np.linalg.norm(u_seg[:, 1:3], axis=1))),
    )

    return p, q, u_seg, info
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
def mid_to_node_values(v_mid):
    """
    Convert segment-midpoint values (3, N-1) to node values (3, N)
    by endpoint copy + interior averaging.
    """
    v_mid = np.asarray(v_mid, float)
    if v_mid.ndim != 2 or v_mid.shape[0] != 3:
        raise ValueError(f"Expected v_mid shape (3, N-1), got {v_mid.shape}")

    Nmid = v_mid.shape[1]
    N = Nmid + 1
    v = np.zeros((3, N), float)

    v[:, 0] = v_mid[:, 0]
    v[:, -1] = v_mid[:, -1]
    if N > 2:
        v[:, 1:-1] = 0.5 * (v_mid[:, :-1] + v_mid[:, 1:])

    return v
def u0_from_bvp(sol, *, L, wire_len, Kinv_fun, N=60, u_star=np.zeros(3)):
    s = np.linspace(0.0, float(L), int(N))
    Y = sol.sol(s)
    q = quat_normalize(Y[3:7, :])
    m_world = Y[10:13, :]                 # (3,N)
    R = quat_to_rot(q)                    # (N,3,3)
    m_body = np.einsum('nij,jn->in', np.transpose(R,(0,2,1)), m_world)  # (3,N)

    Kinv = Kinv_fun(s, wire_len)          # (3,3,N)
    u = np.einsum('ijn,jn->in', Kinv, m_body) + np.asarray(u_star)[:,None]  # (3,N)

    # segment strains: average adjacent nodes
    u_seg = 0.5*(u[:, :-1] + u[:, 1:]).T   # (N-1,3)
    return u_seg_to_flat(u_seg)
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
import pandas as pd
import numpy as np


def validate_cosserat_against_csv(
    csv_path,
    model,
    base_point,
    pivot_point,
    L_cmd,
    wire_len,
    m_body,
    i_idx=0,
    j_values=range(0, 91, 5),
    error_threshold_mm=1.0,
):
    df = pd.read_csv(csv_path)

    results = []

    beam_base_robot_m = np.asarray(model.p0, dtype=float).reshape(3)

    for j in j_values:
        row_match = df[(df["i_idx"] == i_idx) & (df["j_idx"] == j)]

        if len(row_match) == 0:
            print(f"[SKIP] No CSV row for i_idx={i_idx}, j_idx={j}")
            continue

        row = row_match.iloc[0]

        start_point = np.asarray(
            get_point(i_idx, j, base_point, pivot_point),
            dtype=float
        )
        start_point[2] = -0.1

        T_ur_mag = ur_pose6_to_T(start_point)
        r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

        out = model.forward(
            L=L_cmd,
            r_src=r_src_ur,
            q_src=q_src_ur,
            wire_len=wire_len,
            m_body=m_body,
        )

        pred_tip_robot_m = np.asarray(out["p_tip"], dtype=float).reshape(3)

        pred_tip_local_mm = predicted_centerline_robot_to_base_local_mm(
            pred_tip_robot_m.reshape(1, 3),
            beam_base_robot_m,
            pivot_point,
        )[0]
        meas_tip_local_mm = np.array([
            row["meas_base_x_mm"],
            row["meas_base_y_mm"],
            row["meas_base_z_mm"],
        ], dtype=float)

        err_vec_mm = (pred_tip_local_mm - meas_tip_local_mm)
        err_xy_mm = float(np.linalg.norm(err_vec_mm[:2]))
        err_xyz_mm = float(np.linalg.norm(err_vec_mm))

        failed = err_xyz_mm > error_threshold_mm

        results.append({
            "i_idx": i_idx,
            "j_idx": j,

            "pred_x_mm": pred_tip_local_mm[0],
            "pred_y_mm": pred_tip_local_mm[1],
            "pred_z_mm": pred_tip_local_mm[2],

            "meas_x_mm": meas_tip_local_mm[0],
            "meas_y_mm": meas_tip_local_mm[1],
            "meas_z_mm": meas_tip_local_mm[2],

            "err_x_mm": err_vec_mm[0],
            "err_y_mm": err_vec_mm[1],
            "err_z_mm": err_vec_mm[2],
            "err_xy_mm": err_xy_mm,
            "err_xyz_mm": err_xyz_mm,

            "failed": failed,
        })

        status = "FAIL" if failed else "OK"
        print(
            f"[{status}] j={j:3d} | "
            f"err_xyz={err_xyz_mm:.3f} mm | "
            f"err_xy={err_xy_mm:.3f} mm | "
            f"pred_local=({pred_tip_local_mm[0]:.3f}, "
            f"{pred_tip_local_mm[1]:.3f}, "
            f"{pred_tip_local_mm[2]:.3f}) mm | "
            f"meas=({meas_tip_local_mm[0]:.3f}, "
            f"{meas_tip_local_mm[1]:.3f}, "
            f"{meas_tip_local_mm[2]:.3f}) mm"
        )

    results_df = pd.DataFrame(results)

    failed_df = results_df[results_df["failed"]]

    if len(failed_df) > 0:
        print("\nCosserat model validation failed.")
        print(failed_df[[
            "i_idx", "j_idx",
            "err_x_mm", "err_y_mm", "err_z_mm",
            "err_xy_mm", "err_xyz_mm"
        ]])
    else:
        print("\nCosserat model validation passed.")

    return results_df
if __name__ == "__main__":
    DEBUG = True
    L_cmd = 0.0293

    mag_len = beam_params.length_of_mag
    MAG_YAW_CAL_DEG = 12.4  # try -5 first because physically subtracting joint 5 fixed it

    m_body_nominal = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)
    m_body = rotate_body_xy(m_body_nominal, MAG_YAW_CAL_DEG)
    pivot_point = np.array([0.8281328220229531, -0.6812560048066458, -0.1, 3.1374639959012303, 0.13796054585074355, 0.0009377017734513666]
    , float)
    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.17),
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
        E=beam_params.E,
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
    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kinv_fun,
        m_local_fun=make_m_local_fun_wire_tip(wire_len, len_tip=tip_len, mode="axial", alpha_end=0.0),
        m_moment=0.0,
        wire_len=wire_len,
    )
    results_df = validate_cosserat_against_csv(
        csv_path="/home/jack/Proper-Research/results_with_lumen_no_drawing_29/sweep_results.csv",
        model=model,
        base_point=base_point,
        pivot_point=pivot_point,
        L_cmd=L_cmd,
        wire_len=wire_len,
        m_body=m_body,
        i_idx=0,
        j_values=range(0, -91, -5),
        error_threshold_mm=1.0,
    )
    # out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len,m_body=m_body)
    # if hasattr(model, "_last_energy_choice"):
    #     print(model._last_energy_choice)
    # print("tip in UR:", out["p_tip"])
    # print("tip bending y:", np.rad2deg(out["theta_y"]))
    # print("tip bending z:", np.rad2deg(out["theta_z"]))
    # print("Magnet position:", T_ur_mag)
    # print("Magnetic field", (out["B_tip"]))
    # print("Magnetic force this is the gradient force", (out["F_net"]))
    # print("Magnetic torque is the cross product", (out["T_net"]))


    # # base tangent direction
    # q = q0_ur
    # R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
    # global INSERTION_DIR_WORLD
    # INSERTION_DIR_WORLD = t0 / (np.linalg.norm(t0) + 1e-12)
    # # lumen centerline starts at pivot base and bends
    # # s_straight = 0.01
    # Rbase = Rot.from_quat([q0_ur[1], q0_ur[2], q0_ur[3], q0_ur[0]]).as_matrix()
    # t0 = Rbase @ np.array([-1.0, 0.0, 0.0])

    # lumen_C = make_lumen_centerline_turning(
    #     p_start=p0_ur,
    #     t0=t0,
    #     length=0.03,
    #     n_pts=130,
    #     bend_axis=np.array([0.0, 0.0, 1.0]),
    #     bend_angle=np.deg2rad(90.0),
    #     bend_start=0.005,
    #     bend_end=0.03,
    # )
    # lumen_C, _ = resample_polyline(lumen_C, ds_target=1e-3)
    # lumen_R = np.full(len(lumen_C), 0.004)
    # # # lumen_C = make_lumen_centerline_turning(
    # #     p_start=p0_ur,
    # #     t0=t0,
    # #     length=0.08 + s_straight,     
    # #     n_pts=130,                      
    # #     bend_axis=np.array([0.0, 0.0, 1.0]),
    # #     bend_angle=np.deg2rad(-40.0),
    # #     bend_start=0.01 + s_straight,    
    # #     bend_end=0.08 + s_straight       
    # # )

    # # lumen_R = np.full(len(lumen_C), 0.004)  # 4 mm radius
    # print("Base tangent direction (UR) =", t0)
    # p_tip_pred = p0_ur + L_cmd * t0
    # print("Pred straight tip:", p_tip_pred)
    # print("Solved tip:", out["p_tip"])
    # print("Diff:", out["p_tip"] - p_tip_pred, "norm:", np.linalg.norm(out["p_tip"] - p_tip_pred))

    # out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len, m_body=m_body)
    # print("tip in UR:", out["p_tip"])
    # print("tip bending y:", np.rad2deg(out["theta_y"]))
    # print("tip bending z:", np.rad2deg(out["theta_z"]))
    # print("Magnet position:", T_ur_mag)
    # print("Magnetic field", (out["B_tip"]))
    # print("Magnetic force this is the gradient force", (out["F_net"]))
    # print("Magnetic torque is the cross product", (out["T_net"]))


    # q = q0_ur  
    # R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # t0 = R0 @ np.array([-1.0, 0.0, 0.0])
    # print("Base tangent direction (UR) =", t0)
    # p_tip_pred = p0_ur + L_cmd * t0
    # print("Pred straight tip:", p_tip_pred)
    # print("Solved tip:", out["p_tip"])
    # print("Diff:", out["p_tip"] - p_tip_pred, "norm:", np.linalg.norm(out["p_tip"] - p_tip_pred))

    # m_src = dipole_from_pose(q_src_ur, m_body)
    # sol_bvp = model.solve(L=L_cmd, r_src=r_src_ur, m_src=m_src, wire_len=wire_len)

    # u0 = u0_from_bvp(sol_bvp, L=L_cmd, wire_len=wire_len, Kinv_fun=Kinv_fun, N=60)
    # # pE, qE, uE, info = solve_energy_with_wall_continuation(
    # #     p0=p0_ur, q0=q0_ur, L=L_cmd, wire_len=wire_len, Kinv_fun=Kbt_inv_profile,
    # #     u_star=np.zeros(3), r_src=r_src_ur, m_src=m_src,
    # #     m_local_fun=model.m_local_fun, m_moment=0.0,
    # #     lumen_C=lumen_C, lumen_R=lumen_R,
    # #     N=60, u0_flat=u0, maxiter=200
    # # )
    # L_model, wire_len, tip_len = effective_lengths(
    # L_cmd,
    # L_tip_full=0.04,
    # L_tip_min=0.01,
    # )
    # m_local_fun = make_m_local_fun_wire_tip(
    #     wire_len,
    #     len_tip=tip_len,
    #     mode="axial",
    #     alpha_end=0.0,
    #     eps=1e-3
    # )
    # u_init = None
    # n_values = np.linspace(4,200, 5 )
    # # for n in n_values:
    # hist = solve_quasistatic_insertion(
    #     p0=p0_ur, q0=q0_ur,
    #     L0=0.002, Lf=L_cmd, dL=0.002,
    #     wire_len_fun=wire_len_fun,
    #     tip_len_fun=tip_len_fun,
    #     Kinv_fun=Kinv_fun, u_star=np.zeros(3),
    #     r_src=r_src_ur, m_src=m_src,
    #     m_local_fun=m_local_fun, m_moment=0.0,
    #     lumen_C=lumen_C, lumen_R=lumen_R,
    #     N=35, maxiter=30,
    #     use_lumen=False,
    #     u_init=u_init,
    #     debug=True
    # )
    # # take final
    # pE = hist[-1]["p"]
    # qE = hist[-1]["q"]
    # info = hist[-1]["info"]
    # viol = lumen_violation_profile(pE, lumen_C, lumen_R)
    # print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
    # print("mean positive violation [m] =", np.maximum(viol,0).mean())
    # print(info["parts"])
    # print("energy-min tip:", pE[:, -1])
    # # choose a common comparison grid
    # s_cmp = info["s"]                      # energy-min grid
    # p_bvp = get_centerline_bvp(sol_bvp, s_cmp)
    # p_energy = pE

    # # optional straight baseline (same convention as your earlier straight tip)
    # # build straight line from base tangent
    # q = q0_ur
    # R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    # t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
    # p_straight = p0_ur.reshape(3,1) + t0.reshape(3,1) * s_cmp.reshape(1,-1)

    # plot_centerlines_with_lumen_3d(
    #     p_bvp, p_energy,
    #     lumen_C=lumen_C, lumen_R=lumen_R,
    #     p0=p0_ur, p_straight=p_straight,
    #     title="Cosserat vs Energy-min + Lumen constraint"
    # )

    # plot_error_vs_s(s_cmp, p_bvp, p_energy)
    # s_cmp = info["s"]                 # energy-min node grid
    # p_bvp = get_centerline_bvp(sol_bvp, s_cmp)

    # Y_bvp = sol_bvp.sol(s_cmp)
    # q_bvp = quat_normalize(Y_bvp[3:7, :])

    # p_energy = pE
    # q_energy = qE   # from solve_energy_min_3d / hist[-1]["q"]
    # metrics = compare_common_metrics(
    # s_cmp,
    # p_bvp, q_bvp,
    # p_energy, q_energy,
    # m_src=m_src,
    # r_src=r_src_ur,
    # m_local_fun=m_local_fun,
    # m_moment=0.0,
    # )
    # print_comparison_setup(
    #     L_model=L_model,
    #     wire_len=wire_len,
    #     tip_len=tip_len,
    #     p0=p0_ur,
    #     q0=q0_ur,
    #     r_src=r_src_ur,
    #     q_src=q_src_ur,
    #     m_src=m_src,
    # )