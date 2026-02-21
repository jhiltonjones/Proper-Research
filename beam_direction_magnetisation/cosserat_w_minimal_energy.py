import numpy as np
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
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
import matplotlib.pyplot as plt
beam_params = default_beam_params()
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
def contact_penalty_softplus(p, lumen_query, *,
                             k_in=5e4,      # near-wall stiffness
                             k_out=5e5,     # outside stiffness (bigger!)
                             g0=5e-4,       # activation thickness (0.5mm)
                             alpha=80.0,    # softplus sharpness
                             window=3, eps=1e-9):
    """
    Returns:
      C: (N,) penalty energy per node
      F: (3,N) penalty force per node (negative gradient wrt p)
      g: (N,) clearance g = R - delta
    """
    p = np.asarray(p, float)
    N = p.shape[1]
    C = np.zeros(N)
    F = np.zeros((3,N))
    g_arr = np.zeros(N)

    for j in range(N):
        x = p[:,j]
        delta, Rloc, q = lumen_query.closest(x, window=window)
        g = Rloc - delta
        g_arr[j] = g

        # outward normal (from centerline to point)
        if delta > eps:
            n = (x - q) / delta
        else:
            n = np.array([1.0,0.0,0.0])

        # penetration amount (>=0 outside)
        pen = max(-g, 0.0)

        # softplus near-wall barrier (active when g < g0)
        # z = (g0 - g) / g0  (0 deep inside, 1 at wall, >1 outside)
        z = (g0 - g) / (g0 + 1e-12)

        # softplus(z) ~ 0 for z<<0, ~ z for z>>0
        sp = (1.0/alpha) * np.log1p(np.exp(alpha*z))

        # energy:
        # - inside near wall: k_in * sp^2
        # - outside: add strong quadratic on penetration
        Cj = 0.5 * k_in * sp*sp + 0.5 * k_out * pen*pen
        C[j] = Cj

        # derivative wrt g:
        # z = (g0-g)/g0 => dz/dg = -1/g0
        # d(sp)/dz = sigmoid(alpha*z)
        sig = 1.0 / (1.0 + np.exp(-alpha*z))
        dsp_dg = sig * (-1.0/(g0 + 1e-12))

        dC_dg = k_in * sp * dsp_dg  # from 0.5*k_in*sp^2
        if pen > 0.0:
            # pen = -g => d(0.5*k_out*pen^2)/dg = -k_out*pen
            dC_dg += -k_out * pen

        # g = R - delta, delta increases outward:
        # ∂g/∂x = -∂delta/∂x = -n
        # Force = -∂C/∂x = - (dC/dg) * ∂g/∂x = - dC/dg * (-n) = (dC/dg)*n
        F[:,j] = dC_dg * n

    return C, F, g_arr
def contact_barrier_energy_and_force_fast(
    p, lumen_query: LumenQuery, *,
    Kc=5e4,
    d_tilde=1e-3,
    eps=1e-9,
    penalize_outside=True,
    k_out=5e4,
    window=3
):
    p = np.asarray(p, float)
    N = p.shape[1]
    C = np.zeros(N, float)
    F = np.zeros((3, N), float)
    d_arr = np.zeros(N, float)

    for j in range(N):
        x = p[:, j]
        delta, Rloc, q_closest = lumen_query.closest(x, window=window)
        d = Rloc - delta
        d_arr[j] = d

        if delta > eps:
            n = (x - q_closest) / delta
        else:
            n = np.array([1.0, 0.0, 0.0])

        if (d > 0.0) and (d < d_tilde):
            log_term = np.log(max(d, eps) / d_tilde)
            Cj = -Kc * (d - d_tilde)**2 * log_term
            C[j] = Cj

            dC_dd = -Kc * (2.0*(d - d_tilde)*log_term + (d - d_tilde)**2 * (1.0 / max(d, eps)))
            F[:, j] = dC_dd * n

        elif d >= d_tilde:
            pass
        else:
            if penalize_outside:
                pen = -d
                C[j] = 0.5 * k_out * pen * pen
                F[:, j] = -k_out * pen * n

    return C, F, d_arr

def u_flat_from_ctrl(u_ctrl, s):
    """
    u_ctrl: (K,3) control values along s_ctrl
    returns u_flat for (N-1) segments -> shape (3*(N-1),)
    """
    N = len(s)
    s_seg = 0.5*(s[:-1] + s[1:])          # segment midpoints
    K = u_ctrl.shape[0]
    s_ctrl = np.linspace(s_seg[0], s_seg[-1], K)

    u_seg = np.zeros((N-1, 3))
    for k in range(3):
        cs = CubicSpline(s_ctrl, u_ctrl[:, k], bc_type="natural")
        u_seg[:, k] = cs(s_seg)

    return u_seg.reshape(-1)
def point_to_polyline_closest(p, C):
    """
    p: (3,)
    C: (M,3)
    Returns:
      d_min, i, t, q, seg_tangent
    """
    d_min = np.inf
    best = None
    for i in range(len(C) - 1):
        a = C[i]; b = C[i+1]
        q, t = closest_point_on_segment(p, a, b)
        d = np.linalg.norm(p - q)
        if d < d_min:
            d_min = d
            seg_tan = b - a
            seg_tan = seg_tan / (np.linalg.norm(seg_tan) + 1e-12)
            best = (i, t, q, seg_tan)
            d_min = d
    i, t, q, seg_tan = best
    return d_min, i, t, q, seg_tan
def contact_friction_density(p, lumen_C, lumen_R, *,
                             k_n=5e4,          # normal stiffness (outside)
                             g0=2e-4,          # contact shell thickness (0.2 mm)
                             k_shell=0.0,      # optional near-wall attraction; start 0
                             mu=0.05,          # friction coefficient
                             ins_dir=np.array([1.0,0.0,0.0]), # insertion direction (world)
                             eps=1e-6):
    """
    p: (3,N)
    Returns w_contact: (N,) energy density along s.
    NOTE: This is a quasi-static friction *proxy* as an energy term.
    """
    ins_dir = np.asarray(ins_dir, float).reshape(3,)
    ins_dir = ins_dir / (np.linalg.norm(ins_dir) + 1e-12)

    N = p.shape[1]
    w = np.zeros(N, float)

    for j in range(N):
        pj = p[:, j]
        d, i, t, q, seg_tan = point_to_polyline_closest(pj, lumen_C)
        Rloc = interpolate_radius(lumen_R, i, t)

        # gap (positive inside)
        g = Rloc - d

        # outside penetration amount
        pen = max(-g, 0.0)      # = max(d - R, 0)

        # normal penalty (outside)
        phi_out = 0.5 * k_n * pen * pen

        # optional shell term (keeps you near wall when within g0)
        # activates when g < g0
        xi = (g0 - g) / (g0 + 1e-12)
        w_shell = smoothstep01(xi)  # 0 deep inside, 1 at wall/outside
        phi_shell = 0.5 * k_shell * w_shell * (max(g0 - g, 0.0)**2)

        # approximate normal force magnitude (from penalty gradient proxy)
        # N ~ k_n * pen (outside) + k_shell * w_shell * max(g0-g,0)
        Nmag = k_n * pen + k_shell * w_shell * max(g0 - g, 0.0)

        # normal direction (only defined if d>0)
        if d > eps:
            n = (pj - q) / d
        else:
            n = np.array([1.0,0.0,0.0])

        # tangent direction along vessel projected into contact plane
        tv = seg_tan
        tproj = tv - np.dot(tv, n) * n
        tnorm = np.linalg.norm(tproj)
        if tnorm > eps:
            tdir = tproj / tnorm
        else:
            # fallback: pick any orthogonal
            tmp = np.array([1.0,0.0,0.0])
            if abs(np.dot(tmp,n)) > 0.9: tmp = np.array([0.0,1.0,0.0])
            tdir = np.cross(n, tmp)
            tdir = tdir / (np.linalg.norm(tdir) + 1e-12)

        # friction "work" proxy per unit length:
        # resist motion along insertion direction projected onto contact tangent
        slip = abs(np.dot(ins_dir, tdir))
        phi_fric = w_shell * mu * Nmag * slip
        w[j] = phi_out + phi_shell + phi_fric

    return w

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
beam_params = default_beam_params()
mag_params = default_magnet_params()
def tip_bending_angles_from_tangent(sol, L, e1=np.array([1.0,0.0,0.0])):
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

        f_ext, tau_ext, _B = magnetic_wrench_density_cosserat_profile(
            p, qn, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
        )
        # f_wall = wall_force_density(p, vessel_centerline, R_vessel, k_wall=k_wall)
        # f_ext = f_ext + f_wall
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


    def solve(self, *, L, r_src, m_src, wire_len, try_energy_selection=True):
        L = float(L)
        r_src = np.asarray(r_src, float).reshape(3,)
        m_src = np.asarray(m_src, float).reshape(3,)

        ode = make_cosserat_kirchhoff_ode(
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=self.Kinv_fun,
            m_local_fun=self.m_local_fun,
            m_moment=self.m_moment,
            wire_len=self.wire_len,
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

        f_ext, tau_ext, B = magnetic_wrench_density_cosserat_profile(
            p, q, s_out, m_src, r_src, self.m_local_fun, self.m_moment, r_min=1e-6
        )
        # f_ext = f_ext*0
        # tau_ext = tau_ext*0
        # f_ext = f_ext*0
        # tau_ext = tau_ext*0
        Fy = np.trapezoid(f_ext[1, :], s_out)
        Fz = np.trapezoid(f_ext[2, :], s_out)
        Ty = np.trapezoid(tau_ext[1, :], s_out)
        Tz = np.trapezoid(tau_ext[2, :], s_out)

        # print("Net transverse force:", Fy, Fz)
        # print("Net transverse torque:", Ty, Tz)

        # Integrals (net)
        F_net = np.trapezoid(f_ext, s_out, axis=1)
        T_net = np.trapezoid(tau_ext, s_out, axis=1)

        # Tip values
        p_tip = p[:, -1]
        q_tip = q[:, -1]
        B_tip = float(np.linalg.norm(B[:, -1]))

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
            profiles=dict(s=s_out, p=p, q=q, f_ext=f_ext, tau_ext=tau_ext, B=B),
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

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = 0.5 * angle
    return np.array([np.cos(h), *(np.sin(h) * axis)], dtype=float)

# def quat_normalize_np(q):
#     return q / (np.linalg.norm(q) + 1e-12)

def compute_total_energy(sol, *, L, r_src, m_src, Kinv_fun, m_local_fun, m_moment, wire_len,
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

    # Convert to world frame: m_world = R * m_body_local
    m_local_world = np.einsum('nij,jn->in', R, m_local_body)  # (3,N)

    # Field along rod from your existing function
    # (it already uses p,q and magnet pose; B is typically in world frame)
    _, _, B = magnetic_wrench_density_cosserat_profile(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )

    w_m = -np.sum(m_local_world * B, axis=0)  # (N,)
    W_m = np.trapezoid(w_m, s)

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

    W_total = W_s + W_m + W_v + W_g
    W_total = _as_scalar(W_total, "W_total")
    return W_total, parts


def solve_quasistatic_insertion(*,
    p0, q0,
    L0, Lf, dL,
    wire_len_fun,          # function wire_len(L) or constant
    Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    lumen_C, lumen_R,
    N=30, maxiter=200,
    u_init=None,
    use_lumen=True
):
    """
    Returns history list of dicts for each step.
    """
    hist = []
    u0 = u_init

    L = L0
    while L <= Lf + 1e-12:
        wire_len = wire_len_fun(L)

        p, q, u_seg, info = solve_energy_min_3d(
            p0=p0, q0=q0, L=L,
            wire_len=wire_len, Kinv_fun=Kinv_fun, u_star=u_star,
            r_src=r_src, m_src=m_src,
            m_local_fun=m_local_fun, m_moment=m_moment,
            N=N, u0_flat=u0, maxiter=maxiter,
            lumen_C=lumen_C, lumen_R=lumen_R, use_lumen=use_lumen
            # you can keep your continuation on k_n, mu, etc by wrapping solve_energy_min_3d
        )

        hist.append(dict(L=L, p=p, q=q, info=info))
        if not info["success"]:
            break

        u0 = info["u_flat_opt"]        
        L += dL

    return hist

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
    u_seg = u_flat.reshape(N-1, 3)

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
    contact_mode="tip", include_gravity=False,  # <- set default False for safety
    contact_s_on=0.0, contact_s_off=0.0,
    lumen_query=None, use_lumen=True,
    debug_mag=True, debug_every=1, debug_head=8
):
    """
    Computes total potential energy Π(u) in 3D.
    If debug_mag=True: prints per-node magnetic energy density info.
    """
    p, q, u_seg = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)  # p: (3,N), q: (N,4) or (4,N)
    s = np.asarray(s, float).ravel()
    N = s.size

    ds = np.diff(s)
    u_star = np.asarray(u_star, float).reshape(3,)

    # Elastic energy
    W_s = 0.0
    for i in range(len(ds)):
        du = (u_seg[i] - u_star)
        W_s += 0.5 * du @ K_seg[i] @ du * ds[i]

    # Magnetic field at nodes
    f_mag, tau_mag, B = magnetic_wrench_density_cosserat_profile(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )  # B: (3,N)
    if debug_mag:
        fz = f_mag[2, :]
        print("[DBG-MAG] fz min/max:", fz.min(), fz.max())
        print("[DBG-MAG] f at tip:", f_mag[:, -1], "tau at tip:", tau_mag[:, -1])
    # Build m_world at nodes
    R_all = quat_to_rot(quat_normalize(q))     # expect (N,3,3)
    m_body = np.asarray(m_local_fun(s, None), float)  # (3,N)
    m_world = np.einsum('nij,jn->in', R_all, m_body)  # (3,N)

    m_dot_B = np.sum(m_world * B, axis=0)      # (N,)
    w_m = -m_dot_B                              # (N,) magnetic energy density (per unit length-ish)
    W_m = np.trapezoid(w_m, s)

    # Gravity (disable unless you really want it)
    W_g = 0.0
    if include_gravity:
        fg = np.asarray(beam_params.f_g, float).reshape(3,)
        W_g = -np.trapezoid(np.sum(fg[:, None] * p, axis=0), s)

    # Contact / lumen
    W_cf = 0.0
    if use_lumen and (lumen_query is not None):
        C_nodes, F_nodes, d_nodes = contact_barrier_energy_and_force_fast(
            p, lumen_query, Kc=5e4, d_tilde=1e-3, penalize_outside=True, k_out=5e5
        )
        W_cf = (s[1] - s[0]) * float(np.sum(C_nodes))

    W_total = W_s + W_m + W_g + W_cf

    if debug_mag:
        Bnorm = np.linalg.norm(B, axis=0) + 1e-16
        mnorm = np.linalg.norm(m_world, axis=0) + 1e-16

        # alignment cos and angle
        cos_th = np.sum(m_world * B, axis=0) / (Bnorm * mnorm)
        cos_th = np.clip(cos_th, -1.0, 1.0)
        th_deg = np.degrees(np.arccos(cos_th))

        # torque density proxy (direction + magnitude)
        tau = np.cross(m_world.T, B.T).T          # (3,N)
        tau_norm = np.linalg.norm(tau, axis=0)    # (N,)

        # scalar energy density and cumulative
        m_dot_B = np.sum(m_world * B, axis=0)
        w_m = -m_dot_B
        Wm_cum = np.zeros(N, float)
        for i in range(1, N):
            Wm_cum[i] = np.trapezoid(w_m[:i+1], s[:i+1])

        print("\n[DBG-MAG] per-node magnetic + alignment")
        print(" i  s(mm)  x(mm)  y(mm)  z(mm)   |B|     Bx      By      Bz      "
            "m·B     theta(deg)  |m×B|     w_m      Wm_cum")

        head = int(debug_head)
        step = int(debug_every)

        def _row(i):
            return (f"{i:2d} {1e3*s[i]:6.2f} "
                    f"{1e3*p[0,i]:6.2f} {1e3*p[1,i]:6.2f} {1e3*p[2,i]:6.2f} "
                    f"{Bnorm[i]:+7.2e} {B[0,i]:+7.2e} {B[1,i]:+7.2e} {B[2,i]:+7.2e} "
                    f"{m_dot_B[i]:+7.2e} {th_deg[i]:8.2f} {tau_norm[i]:+7.2e} "
                    f"{w_m[i]:+8.2e} {Wm_cum[i]:+8.2e}")

        # head
        count = 0
        for i in range(0, N, step):
            print(_row(i))
            count += 1
            if count >= head:
                break

        # tail
        if N > head:
            print(" ...")
            for i in range(max(0, N - head), N, step):
                print(_row(i))

        print(f"[DBG-MAG] max theta(deg) = {np.max(th_deg):.3f}")
        print(f"[DBG-MAG] max |m×B|      = {np.max(tau_norm):.3e}")

    parts = dict(W_s=float(W_s), W_m=float(W_m), W_g=float(W_g), W_cf=float(W_cf))
    return float(W_total), parts
    # W_total = float(W_s + W_m + W_g + W_c)
    # parts = dict(W_s=float(W_s), W_m=float(W_m), W_g=float(W_g), W_c=float(W_c))

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
def solve_energy_min_3d(*, p0, q0, L, wire_len, Kinv_fun, u_star,
                        r_src, m_src, m_local_fun, m_moment,
                        N=20, u0_flat=None, maxiter=200, lumen_C=None, lumen_R=None,use_lumen=True,
                        contact_k=1e3, contact_beta=50.0, contact_delta=5e-4,
                        contact_mode="tip", contact_s_on=0.0, contact_s_off=0.0, u_ctrl_init=None,
                        K=8):
    """
    Minimises Π(u) using K control points for u(s).
    Returns (p, q, u_seg, info).
    """
    s = np.linspace(0.0, float(L), int(N))
    K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

    # ----- init control points -----
    # default: zeros
    u_ctrl0 = np.zeros((K, 3), float)

    # if you have a warm start in segment space, compress it to control points
    if u0_flat is not None:
        u0_flat = np.asarray(u0_flat, float).reshape(-1)
        # expect segment strains
        if u0_flat.size == 3*(N-1):
            u0_seg = u0_flat.reshape(N-1, 3)
            # sample K points from the segment midpoints
            s_seg = 0.5*(s[:-1] + s[1:])
            s_ctrl = np.linspace(s_seg[0], s_seg[-1], K)
            # simple nearest sampling (good enough); you can spline-fit too
            idx = np.clip(np.searchsorted(s_seg, s_ctrl), 0, len(s_seg)-1)
            u_ctrl0 = u0_seg[idx, :]
        else:
            # if caller already passes control warm start, allow it
            if u0_flat.size == 3*K:
                u_ctrl0 = u0_flat.reshape(K, 3)

    u_ctrl0_flat = u_ctrl0.reshape(-1)
    lumen_query = None
    if lumen_C is not None and lumen_R is not None:
        lumen_query = LumenQuery(lumen_C, lumen_R)
    # ----- objective in control space -----
    u_scale = 30.0   # start with 5; try 10, 20 if needed

    def obj_ctrl(z_flat):
        z = z_flat.reshape(K, 3)
        u_ctrl = u_scale * z
        u_flat = u_flat_from_ctrl(u_ctrl, s)

        W, _ = energy_from_u(
            u_flat,
            p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
            m_src=m_src, r_src=r_src, m_local_fun=m_local_fun, m_moment=m_moment,
            wire_len=wire_len, include_gravity=False,
            lumen_C=lumen_C, lumen_R=lumen_R, lumen_query=lumen_query, use_lumen=use_lumen, debug_mag=False,
        )
        return W
    z0_flat = (u_ctrl0_flat / u_scale)
    # u_test = u_ctrl0_flat.copy()
    # u_test[0] = 5.0  # curvature-ish magnitude (1/m); try also 1.0, 10.0

    # print("W(u0)   =", obj_ctrl(u_ctrl0_flat))
    # print("W(pert) =", obj_ctrl(u_test))
    res = minimize(
        obj_ctrl,
        z0_flat,
        method="L-BFGS-B",
        options=dict(
            maxiter=maxiter,
            ftol=1e-9,
            eps=1e-2,     # try 1e-2 then 1e-1
            maxls=80
        )
    ) 
    z_opt = res.x.reshape(K,3)
    u_ctrl_opt = u_scale * z_opt
    u_flat_opt = u_flat_from_ctrl(u_ctrl_opt, s)# ----- reconstruct final solution in segment space -----
    # u_ctrl_opt = res.x.reshape(K, 3)
    # u_flat_opt = u_flat_from_ctrl(u_ctrl_opt, s)   # shape (3*(N-1),)
    p, q, u_seg = integrate_pq_from_u(u_flat_opt, p0=p0, q0=q0, s=s)

    # recompute energies consistently (IMPORTANT: use u_flat_opt)
    W, parts = energy_from_u(
        u_flat_opt,
        p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
        m_src=m_src, r_src=r_src, m_local_fun=m_local_fun, m_moment=m_moment,
        wire_len=wire_len, include_gravity=False,
        lumen_C=lumen_C, lumen_R=lumen_R,
        contact_k=contact_k, contact_beta=contact_beta, contact_delta=contact_delta,
        contact_mode=contact_mode, contact_s_on=contact_s_on, contact_s_off=contact_s_off, lumen_query=lumen_query, 
        use_lumen=use_lumen, debug_mag=False,
    )
    # print("parts:", parts)

    info = dict(
        success=bool(res.success),
        message=str(res.message),
        nit=int(res.nit),
        W=float(W),
        parts=parts,
        s=s,
        # store BOTH so you can choose what to warm-start with
        u_ctrl_opt=res.x.copy(),        # (3*K,)
        u_flat_opt=u_flat_opt.copy(),   # (3*(N-1),)
    )
    info["z_opt"] = res.x.copy()
    info["u_ctrl_opt"] = u_ctrl_opt.reshape(-1).copy()
    return p, q, u_seg, info
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
    u_seg = 0.5*(u[:, :-1] + u[:, 1:]).T  # (N-1,3)
    return u_seg.reshape(-1)

if __name__ == "__main__":
    DEBUG = True
    L_cmd = 0.089
    mag_len = beam_params.length_of_mag
    m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
    pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)


    start_point = np.array([ 0.681, -0.649,  0.092, -3.054, -0.476 ,
        0.05 ])
    # start_point[2] -=0.25
    wire_len = L_cmd - mag_len
    T_ur_pivot = ur_pose6_to_T(pivot_point)   
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)      
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0),
        m_moment=0.0,
        wire_len=wire_len,
    )

    out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len,m_body=m_body)
    if hasattr(model, "_last_energy_choice"):
        print(model._last_energy_choice)
    print("tip in UR:", out["p_tip"])
    print("tip bending y:", np.rad2deg(out["theta_y"]))
    print("tip bending z:", np.rad2deg(out["theta_z"]))
    print("Magnet position:", T_ur_mag)
    print("Magnetic field", (out["B_tip"]))
    print("Magnetic force this is the gradient force", (out["F_net"]))
    print("Magnetic torque is the cross product", (out["T_net"]))


    # base tangent direction (same convention you already use)
    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
    global INSERTION_DIR_WORLD
    INSERTION_DIR_WORLD = t0 / (np.linalg.norm(t0) + 1e-12)
    # lumen centerline starts at pivot base and bends
    s_straight = 0.01

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.08 + s_straight,     
        n_pts=130,                      
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(-40.0),
        bend_start=0.01 + s_straight,    
        bend_end=0.08 + s_straight       
    )

    lumen_R = np.full(len(lumen_C), 0.004)  # 4 mm radius
    print("Base tangent direction (UR) =", t0)
    p_tip_pred = p0_ur + L_cmd * t0
    print("Pred straight tip:", p_tip_pred)
    print("Solved tip:", out["p_tip"])
    print("Diff:", out["p_tip"] - p_tip_pred, "norm:", np.linalg.norm(out["p_tip"] - p_tip_pred))

    out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len, m_body=m_body)
    print("tip in UR:", out["p_tip"])
    print("tip bending y:", np.rad2deg(out["theta_y"]))
    print("tip bending z:", np.rad2deg(out["theta_z"]))
    print("Magnet position:", T_ur_mag)
    print("Magnetic field", (out["B_tip"]))
    print("Magnetic force this is the gradient force", (out["F_net"]))
    print("Magnetic torque is the cross product", (out["T_net"]))


    q = q0_ur  
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])
    print("Base tangent direction (UR) =", t0)
    p_tip_pred = p0_ur + L_cmd * t0
    print("Pred straight tip:", p_tip_pred)
    print("Solved tip:", out["p_tip"])
    print("Diff:", out["p_tip"] - p_tip_pred, "norm:", np.linalg.norm(out["p_tip"] - p_tip_pred))

    m_src = dipole_from_pose(q_src_ur, m_body)
    sol_bvp = model.solve(L=L_cmd, r_src=r_src_ur, m_src=m_src, wire_len=wire_len)

    u0 = u0_from_bvp(sol_bvp, L=L_cmd, wire_len=wire_len, Kinv_fun=Kbt_inv_profile, N=60)
    # pE, qE, uE, info = solve_energy_with_wall_continuation(
    #     p0=p0_ur, q0=q0_ur, L=L_cmd, wire_len=wire_len, Kinv_fun=Kbt_inv_profile,
    #     u_star=np.zeros(3), r_src=r_src_ur, m_src=m_src,
    #     m_local_fun=model.m_local_fun, m_moment=0.0,
    #     lumen_C=lumen_C, lumen_R=lumen_R,
    #     N=60, u0_flat=u0, maxiter=200
    # )
    hist = solve_quasistatic_insertion(
        p0=p0_ur, q0=q0_ur,
        L0=0.010, Lf=L_cmd, dL=0.002,
        wire_len_fun=lambda L: L - mag_len,
        Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
        r_src=r_src_ur, m_src=m_src,
        m_local_fun=model.m_local_fun, m_moment=0.0,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N=20, maxiter=20, use_lumen = True,
        u_init=u0
    )
    # take final
    pE = hist[-1]["p"]
    info = hist[-1]["info"]
    viol = lumen_violation_profile(pE, lumen_C, lumen_R)
    print("max lumen violation [m] =", viol.max(), "at node", np.argmax(viol))
    print("mean positive violation [m] =", np.maximum(viol,0).mean())
    print(info["parts"])
    print("energy-min tip:", pE[:, -1])
    # choose a common comparison grid
    s_cmp = info["s"]                      # energy-min grid
    p_bvp = get_centerline_bvp(sol_bvp, s_cmp)
    p_energy = pE

    # optional straight baseline (same convention as your earlier straight tip)
    # build straight line from base tangent
    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])   # matches your e1
    p_straight = p0_ur.reshape(3,1) + t0.reshape(3,1) * s_cmp.reshape(1,-1)

    plot_centerlines_with_lumen_3d(
        p_bvp, p_energy,
        lumen_C=lumen_C, lumen_R=lumen_R,
        p0=p0_ur, p_straight=p_straight,
        title="Cosserat vs Energy-min + Lumen constraint"
    )

    plot_error_vs_s(s_cmp, p_bvp, p_energy)