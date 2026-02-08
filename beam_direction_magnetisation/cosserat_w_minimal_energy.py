import numpy as np
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
from beam_direction_magnetisation.magnetism.parameters_cosserat import *
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.magnetism.beam_geometry import smooth_top_hat
from beam_direction_magnetisation.quarternions.shared_rotations import Ry, Rz
from proper_research.parameters import default_magnet_params, default_beam_params
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
def smoothstep01(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3 - 2*x)
def wall_penalty_quadratic(viol, k):
    v = np.maximum(viol, 0.0)
    return 0.5 * k * v * v
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
def contact_weight_profile(s, s_on, s_off):
    """
    Weight = 0 before s_on, ramps to 1 by s_off.
    """
    xi = (s - s_on) / (s_off - s_on + 1e-12)
    return smoothstep01(xi)
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
def solve_energy_with_wall_continuation(
    *,
    p0, q0, L, wire_len, Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    lumen_C, lumen_R,
    N=60,
    u0_flat=None,
    stages=None,
    maxiter=150,
):
    """
    stages: list of dicts, e.g.
      dict(mode="tip", k=1e3, beta=20, delta=5e-4)
    Returns final (p,q,u_seg,info).
    """
    if stages is None:
        stages = [
            dict(contact_mode="tip", k=5e2,  beta=10.0, delta=1e-3, s_on=0.0,   s_off=0.0),
            dict(contact_mode="tip", k=2e3,  beta=20.0, delta=7e-4, s_on=0.0,   s_off=0.0),
            dict(contact_mode="all", k=5e3,  beta=20.0, delta=7e-4, s_on=L-0.005, s_off=L),   # last 5mm
            dict(contact_mode="all", k=1e4,  beta=25.0, delta=5e-4, s_on=L-0.020, s_off=L),   # last 20mm
            dict(contact_mode="all", k=2e4,  beta=30.0, delta=4e-4, s_on=0.0,   s_off=L),     # full length
        ]

    u_init = u0_flat
    last = None

    for st in stages:
        p, q, u_seg, info = solve_energy_min_3d(
            p0=p0, q0=q0, L=L, wire_len=wire_len, Kinv_fun=Kinv_fun,
            u_star=u_star, r_src=r_src, m_src=m_src,
            m_local_fun=m_local_fun, m_moment=m_moment,
            N=N, u0_flat=u_init, maxiter=maxiter,
            lumen_C=lumen_C, lumen_R=lumen_R,
            contact_k=st["k"],
            contact_beta=st["beta"],
            contact_delta=st["delta"],
            contact_mode=st["contact_mode"],
            contact_s_on=st.get("s_on", 0.0),
            contact_s_off=st.get("s_off", 0.0),
        )
        last = (p, q, u_seg, info)
        if not info["success"]:
            # if a stage fails, stop and return the best you have
            return last
        u_init = info["u_opt"]  # warm-start next stage

    return last
def wall_penalty_huber(viol, k, delta):
    v = np.maximum(viol, 0.0)
    return np.where(v <= delta,
                    0.5 * k * (v**2),
                    k * delta * (v - 0.5*delta))
def get_centerline_bvp(sol, s):
    Y = sol.sol(s)
    p = Y[0:3, :]          # (3,N)
    return p

def get_centerline_energy(pE):
    # pE already (3,N)
    return pE
def plot_centerlines_3d(p_bvp, p_energy, p0=None, p_straight=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(p_bvp[0], p_bvp[1], p_bvp[2], label="Cosserat BVP")
    ax.plot(p_energy[0], p_energy[1], p_energy[2], "--", label="Energy-min (3D)")

    if p_straight is not None:
        ax.plot(p_straight[0], p_straight[1], p_straight[2], ":", label="Straight baseline")

    if p0 is not None:
        ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="o", label="Base")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.set_title("Centerline comparison")
    plt.show()
def plot_error_vs_s(s, p_bvp, p_energy):
    err = np.linalg.norm(p_bvp - p_energy, axis=0)  # (N,)
    plt.figure()
    plt.plot(s, err)
    plt.xlabel("s (m)")
    plt.ylabel("||p_bvp(s) - p_energy(s)|| (m)")
    plt.title("Centerline deviation vs arclength")
    plt.grid(True)
    plt.show()
def plot_error_vs_s(s, p_bvp, p_energy):
    err = np.linalg.norm(p_bvp - p_energy, axis=0)  # (N,)
    plt.figure()
    plt.plot(s, err)
    plt.xlabel("s (m)")
    plt.ylabel("||p_bvp(s) - p_energy(s)|| (m)")
    plt.title("Centerline deviation vs arclength")
    plt.grid(True)
    plt.show()
def make_lumen_centerline_turning(
    p_start,
    t0,
    *,
    length=0.12,
    n_pts=60,
    bend_axis=np.array([0.0, 0.0, 1.0]),  # rotate t about this axis
    bend_angle=np.deg2rad(25.0),          # total bend (radians)
    bend_start=0.02,                      # where bending begins along arc (m)
    bend_end=0.09,                        # where bending ends along arc (m)
):
    """
    Returns C: (n_pts,3) polyline points.
    Tangent starts as t0, then smoothly rotates by bend_angle around bend_axis.
    """
    p_start = np.asarray(p_start, float).reshape(3,)
    t0 = np.asarray(t0, float).reshape(3,)
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)

    a = np.asarray(bend_axis, float).reshape(3,)
    a = a / (np.linalg.norm(a) + 1e-12)

    s = np.linspace(0.0, float(length), int(n_pts))
    ds = np.diff(s)

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x*x*(3 - 2*x)

    # bend progress 0->1 between bend_start and bend_end
    xi = (s - bend_start) / (bend_end - bend_start + 1e-12)
    g = smoothstep(xi)  # (n_pts,)
    theta = bend_angle * g

    def rodrigues(v, axis, ang):
        # rotate vector v about axis by ang (scalar)
        v = np.asarray(v, float)
        axis = np.asarray(axis, float)
        return (v*np.cos(ang)
                + np.cross(axis, v)*np.sin(ang)
                + axis*np.dot(axis, v)*(1 - np.cos(ang)))

    # integrate centerline
    C = np.zeros((s.size, 3), float)
    C[0] = p_start
    for i in range(s.size - 1):
        ti = rodrigues(t0, a, theta[i])
        ti = ti / (np.linalg.norm(ti) + 1e-12)
        C[i+1] = C[i] + ds[i] * ti

    return C
def plot_lumen_rings(ax, C, R, n_theta=24, alpha=0.15, linewidth=0.5):
    """
    Draw translucent rings (circles) along a polyline centerline C with radii R.
    Uses a local normal basis per segment.
    """
    C = np.asarray(C, float)
    R = np.asarray(R, float)

    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=True)

    for i in range(len(C)-1):
        a = C[i]
        b = C[i+1]
        t = b - a
        t_norm = np.linalg.norm(t)
        if t_norm < 1e-12:
            continue
        t = t / t_norm

        # pick a reference vector not parallel to t
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, t)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        n1 = np.cross(t, ref)
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        n2 = np.cross(t, n1)

        # radius at segment endpoints
        r0 = R[i]
        r1 = R[i+1]

        # draw ring at start and end of the segment
        for center, rr in [(a, r0), (b, r1)]:
            ring = center[:, None] + rr*(n1[:,None]*np.cos(thetas)[None,:] +
                                         n2[:,None]*np.sin(thetas)[None,:])
            ax.plot(ring[0], ring[1], ring[2], alpha=alpha, linewidth=linewidth)



def plot_centerlines_with_lumen_3d(p_bvp, p_energy, lumen_C=None, lumen_R=None,
                                  p0=None, p_straight=None, title="Centerline + Lumen"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # centerlines
    ax.plot(p_bvp[0], p_bvp[1], p_bvp[2], label="Cosserat BVP")
    ax.plot(p_energy[0], p_energy[1], p_energy[2], "--", label="Energy-min (3D)")

    # baseline / base
    if p_straight is not None:
        ax.plot(p_straight[0], p_straight[1], p_straight[2], ":", label="Straight baseline")
    if p0 is not None:
        ax.scatter([p0[0]], [p0[1]], [p0[2]], marker="o", label="Base")

    # lumen
    if (lumen_C is not None) and (lumen_R is not None):
        C = np.asarray(lumen_C, float)
        ax.plot(C[:,0], C[:,1], C[:,2], label="Lumen centerline")
        plot_lumen_rings(ax, C, np.asarray(lumen_R, float), n_theta=28, alpha=0.2)

        # show closest point on lumen centerline from energy tip
        p_tip = p_energy[:, -1]
        d, i, t, q = point_to_polyline_distance(p_tip, C)
        ax.scatter([p_tip[0]], [p_tip[1]], [p_tip[2]], marker="^", label="Energy tip")
        ax.scatter([q[0]], [q[1]], [q[2]], marker="x", label="Closest lumen point")
        ax.plot([p_tip[0], q[0]], [p_tip[1], q[1]], [p_tip[2], q[2]], linewidth=1.0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.set_title(title)
    plt.show()
def closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    q = a + t * ab
    return q, t

def point_to_polyline_distance(p, C):
    """
    p: (3,)
    C: (M,3) polyline points
    Returns: d_min, seg_idx, t_on_segment, q_closest
    """
    d_min = np.inf
    best = (None, None, None)
    for i in range(len(C) - 1):
        q, t = closest_point_on_segment(p, C[i], C[i+1])
        d = np.linalg.norm(p - q)
        if d < d_min:
            d_min = d
            best = (i, t, q)
    i, t, q = best
    return d_min, i, t, q

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
def softplus(x, beta=50.0):
    z = beta * x
    # stable: log(1+exp(z)) = max(z,0) + log(1+exp(-abs(z)))
    return (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))) / beta

def tip_contact_energy(p_tip, C, R, k=1e3, beta=50.0):
    """
    Penalize if tip is outside lumen: d_tip > r_local
    Returns scalar Wc_tip.
    """
    d, i, t, _ = point_to_polyline_distance(p_tip, C)
    rloc = interpolate_radius(R, i, t)
    viol = d - rloc                      # >0 means outside tube
    v = softplus(viol, beta=beta)        # smooth max(0, viol)
    return 0.5 * k * v * v
def quat_to_R(q):
    qn = quat_normalize(q.reshape(4, 1))
    return quat_to_rot(qn)[0]  # now correct: (1,3,3)[0] -> (3,3)
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
        fg = np.asarray(f_g, float)
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
        Y[0, :] = -s          # x ~ s
        Y[3, :] = 1.0        # quaternion w = 1
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

def m_local_profile_axial(s, eps=1e-3):
    s = np.atleast_1d(s)
    N = s.size
    w = smooth_top_hat(s, s_m, s_m + ell_m, eps)  # only in magnetised region

    m = np.zeros((3, N))
    m[0, :] = mu_line               # axial only
    m[1, :] = 0.0
    m[2, :] = 0.0
    return m * w[None, :]

def make_m_local_fun_axial(eps=1e-3):
    # must accept (s, second_arg) because magnetic_wrench_density_cosserat_profile calls it that way
    def _m_local(s, _unused=None):
        return m_local_profile_axial(s, eps=eps)
    return _m_local
import numpy as np
from beam_direction_magnetisation.magnetism.beam_geometry import smooth_top_hat
from beam_direction_magnetisation.magnetism.parameters_cosserat import s_m, ell_m, mu_line

def m_local_wire_plus_magnetised_tip(
    s,
    *,
    alpha_end=0.0,
    mode="axial",          # "axial" | "constant" | "ramp"
    eps=1e-3,
):
    """
    Returns m_local(s) in BODY frame, shape (3,N).

    Wire region: ~0 magnetisation (outside [s_m, s_m+ell_m])
    Magnetised tip: direction in x-y plane
      - axial    : alpha(s)=0
      - constant : alpha(s)=alpha_end
      - ramp     : alpha(s) = xi*alpha_end, xi in [0,1] over the tip region
    """
    s = np.atleast_1d(s)
    N = s.size

    # Smoothly gate magnetisation into tip region only
    w = smooth_top_hat(s, s_m, s_m + ell_m, eps)  # (N,)

    if mode == "axial":
        alpha_s = np.zeros_like(s)
    elif mode == "constant":
        alpha_s = alpha_end * np.ones_like(s)
    elif mode == "ramp":
        xi = (s - s_m) / (ell_m + 1e-12)
        xi = np.clip(xi, 0.0, 1.0)
        alpha_s = xi * alpha_end
    else:
        raise ValueError(f"Unknown mode='{mode}' (use 'axial','constant','ramp')")

    m = np.zeros((3, N))
    m[0, :] = mu_line * np.cos(alpha_s)
    m[1, :] = mu_line * np.sin(alpha_s)
    m[2, :] = 0.0

    return m * w[None, :]
def make_m_local_fun_wire_tip(*, alpha_end=0.0, mode="axial", eps=1e-3):
    def _m_local(s, _unused=None):
        return m_local_wire_plus_magnetised_tip(
            s, alpha_end=alpha_end, mode=mode, eps=eps
        )
    return _m_local
# def m_local_profile_ramped(s, alpha_end, eps=1e-3):
#     s = np.atleast_1d(s)
#     N = s.size
#     w = smooth_top_hat(s, s_m, s_m + ell_m, eps)
#     xi = np.clip((s - s_m) / (ell_m + 1e-12), 0.0, 1.0)
#     alpha_s = xi * alpha_end
#     m = np.zeros((3, N))
#     m[0, :] = mu_line * np.cos(alpha_s)
#     m[1, :] = mu_line * np.sin(alpha_s)
#     m[2, :] = 0.0
#     return m * w[None, :]

# def make_m_local_fun(alpha_end_fixed, eps=1e-3):
#     def _m_local(s, _m_unused=None):
#         return m_local_profile_ramped(s, alpha_end=alpha_end_fixed, eps=eps)
#     return _m_local

# build model


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

def quat_normalize_np(q):
    return q / (np.linalg.norm(q) + 1e-12)
def epm_pose_orbit_and_spin(
    r_tip,
    rho,
    theta_orbit_z,
    beta_spin,
    *,
    theta_orbit_y=0.0,
    v0_body=np.array([-1.0, 0.0, 0.0]),
):
    """
    Returns:
      r_src: magnet position (3,) in world frame
      q_src: magnet orientation quaternion [w,x,y,z] in world frame

    Semantics:
      - orbit about r_tip with R_orbit = Ry(theta_orbit_y) @ Rz(theta_orbit_z)
      - position: r_tip + R_orbit @ (rho * v0_body)
      - orientation: q_orbit (from R_orbit) then spin about LOCAL z: q_src = q_orbit ⊗ q_spin_local_z
    """

    r_tip = np.asarray(r_tip, float).reshape(3,)
    v0_body = np.asarray(v0_body, float).reshape(3,)

    # Orbit rotation in world
    R_orbit = Ry(theta_orbit_y) @ Rz(theta_orbit_z)

    # Position: pivot about tip
    r_src = r_tip + R_orbit @ (rho * v0_body)

    # Quaternion for orbit rotation.
    # Build it from axis-angles (world y then world z) to match R_orbit = Ry @ Rz.
    q_y = quat_from_axis_angle([0,1,0], theta_orbit_y)
    q_z = quat_from_axis_angle([0,0,1], theta_orbit_z)
    q_orbit = quat_mul(q_y, q_z)  # consistent with Ry @ Rz

    # Spin about magnet's OWN z (local z): post-multiply
    q_spin = quat_from_axis_angle([0,0,1], beta_spin)
    q_src = quat_mul(q_orbit, q_spin)

    return r_src, quat_normalize_np(q_src)

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
    W_g = -np.trapezoid(np.sum(f_g[:, None] * p, axis=0), s)   # scalar
    parts = dict(W_s=W_s, W_m=W_m, W_v=W_v, W_g=W_g)
    W_s = _as_scalar(W_s, "W_s")
    W_m = _as_scalar(W_m, "W_m")
    W_v = _as_scalar(W_v, "W_v")
    W_g = _as_scalar(W_g, "W_g")

    W_total = W_s + W_m + W_v + W_g
    W_total = _as_scalar(W_total, "W_total")
    return W_total, parts
def ur_pose6_to_T(pose6):
    """
    UR RTDE TCP pose6: [x, y, z, rx, ry, rz]
    where [rx,ry,rz] is rotation vector (axis-angle), radians.
    Returns 4x4 transform.
    """
    pose6 = np.asarray(pose6, float).ravel()
    if pose6.size != 6:
        raise ValueError("Expected UR pose6 = [x,y,z,rx,ry,rz]")

    p = pose6[:3]
    rvec = pose6[3:6]
    Rm = Rot.from_rotvec(rvec).as_matrix()

    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T

def solve_quasistatic_insertion(*,
    p0, q0,
    L0, Lf, dL,
    wire_len_fun,          # function wire_len(L) or constant
    Kinv_fun, u_star,
    r_src, m_src, m_local_fun, m_moment,
    lumen_C, lumen_R,
    N=80, maxiter=200,
    u_init=None
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
            lumen_C=lumen_C, lumen_R=lumen_R,
            # you can keep your continuation on k_n, mu, etc by wrapping solve_energy_min_3d
        )

        hist.append(dict(L=L, p=p, q=q, info=info))
        if not info["success"]:
            break

        u0 = info["u_opt"]   # warm start
        L += dL

    return hist
def T_to_p_quat_wxyz(T):
    """
    Convert 4x4 transform -> position (3,) and quaternion [w,x,y,z].
    """
    p = T[:3, 3].copy()
    Rm = T[:3, :3]
    q_xyzw = Rot.from_matrix(Rm).as_quat()  # [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
    q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
    return p, q_wxyz
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
    q[:, 0] = quat_normalize_np(np.asarray(q0, float).reshape(4,))

    for i in range(N-1):
        R = quat_to_R(q[:, i])
        p[:, i+1] = p[:, i] + ds[i] * (R @ e1)

        dq = quat_exp_body(u_seg[i], ds[i])
        q[:, i+1] = quat_normalize_np(quat_mul(q[:, i], dq))

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

def energy_from_u(u_flat, *, p0, q0, s, K_seg, u_star,
                  m_src, r_src, m_local_fun, m_moment, wire_len,
                  lumen_C=None, lumen_R=None,
                  contact_k=1e3, contact_beta=50.0,contact_delta=5e-4,
                  contact_mode="tip", include_gravity=True, contact_s_on=0.0, contact_s_off=0.0):
    """
    Computes total potential energy Π(u) in 3D.
    """
    p, q, u_seg = integrate_pq_from_u(u_flat, p0=p0, q0=q0, s=s)

    ds = np.diff(s)
    u_star = np.asarray(u_star, float).reshape(3,)

    # Elastic energy: sum 0.5 * du^T K du * ds
    W_s = 0.0
    for i in range(len(ds)):
        du = (u_seg[i] - u_star)
        W_s += 0.5 * du @ K_seg[i] @ du * ds[i]

    # Magnetic potential: -∫ m_world · B ds   (node quadrature)
    # Get B at nodes using your existing routine (it already handles full 3D)
    _, _, B = magnetic_wrench_density_cosserat_profile(
        p, q, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
    )  # B: (3,N)

    R_all = quat_to_rot(quat_normalize(q))     # (N,3,3)
    m_body = m_local_fun(s, None)              # (3,N) body-frame magnetisation distribution
    m_world = np.einsum('nij,jn->in', R_all, m_body)
    w_m = -np.sum(m_world * B, axis=0)         # (N,)
    W_m = np.trapezoid(w_m, s)

    # Gravity potential to match your BVP force term (+f_g in f_ext):
    W_g = 0.0
    if include_gravity:
        fg = np.asarray(f_g, float).reshape(3,)
        W_g = -np.trapezoid(np.sum(fg[:, None] * p, axis=0), s)

    W_c = 0.0
    # if (lumen_C is not None) and (lumen_R is not None):
    #     if contact_mode == "tip":
    #         p_tip = p[:, -1]
    #         d, i, t, _ = point_to_polyline_distance(p_tip, lumen_C)
    #         rloc = interpolate_radius(lumen_R, i, t)
    #         viol = d - rloc
    #         W_c = float(wall_penalty_quadratic(viol, contact_k))

    #     elif contact_mode == "all":
    #         # turn contact on only near the tip (continuation will expand this window)
    #         w_s = contact_weight_profile(s, s_on=contact_s_on, s_off=contact_s_off)  # (N,)

    #         w = np.zeros(p.shape[1])
    #         for j in range(p.shape[1]):
    #             d, i, t, _ = point_to_polyline_distance(p[:, j], lumen_C)
    #             rloc = interpolate_radius(lumen_R, i, t)
    #             viol = d - rloc
    #             w[j] = w_s[j] * wall_penalty_quadratic(viol, contact_k)

    #         W_c = float(np.trapezoid(w, s))
    #     else:
    #         raise ValueError("contact_mode must be 'tip' or 'all'")
    W_cf = 0.0
    if (lumen_C is not None) and (lumen_R is not None):
        w_cf = contact_friction_density(
            p, lumen_C, lumen_R,
            k_n=5e4,
            g0=2e-4,
            k_shell=0.0,
            mu=0.05,
            ins_dir=INSERTION_DIR_WORLD
        )
        W_cf = np.trapezoid(w_cf, s)
    W_total = W_s + W_m + W_g + W_cf
    # W_total = float(W_s + W_m + W_g + W_c)
    # parts = dict(W_s=float(W_s), W_m=float(W_m), W_g=float(W_g), W_c=float(W_c))
    parts = dict(W_s=float(W_s), W_m=float(W_m), W_g=float(W_g), W_cf=float(W_cf))
    return W_total, parts

def solve_energy_min_3d(*, p0, q0, L, wire_len, Kinv_fun, u_star,
                        r_src, m_src, m_local_fun, m_moment,
                        N=60, u0_flat=None, maxiter=200, lumen_C=None, 
                        lumen_R=None,
                        contact_k=1e3, contact_beta=50.0,
                        contact_delta=5e-4,
                        contact_mode="tip",
                        contact_s_on=0.0, contact_s_off=0.0):
    """
    Minimises Π(u) and returns (p,q,u_seg, info).
    """
    s = np.linspace(0.0, float(L), int(N))
    K_seg = precompute_K_segments(s, Kinv_fun, wire_len)

    if u0_flat is None:
        u0_flat = np.zeros(3*(N-1), float)

    def obj(u_flat):
        W, _ = energy_from_u(
            u_flat, p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
            m_src=m_src, r_src=r_src, m_local_fun=m_local_fun, m_moment=m_moment,
            wire_len=wire_len, include_gravity=True,
            lumen_C=lumen_C, lumen_R=lumen_R,
            contact_k=contact_k,
            contact_beta=contact_beta,
            contact_delta=contact_delta,
            contact_mode=contact_mode,
            contact_s_on=contact_s_on,
            contact_s_off=contact_s_off
        )
        return W

    res = minimize(obj, u0_flat, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-10))

    # reconstruct final shape
    p, q, u_seg = integrate_pq_from_u(res.x, p0=p0, q0=q0, s=s)

    # recompute energies consistently
    W, parts = energy_from_u(
        res.x, p0=p0, q0=q0, s=s, K_seg=K_seg, u_star=u_star,
        m_src=m_src, r_src=r_src, m_local_fun=m_local_fun, m_moment=m_moment,
        wire_len=wire_len, include_gravity=True,
        lumen_C=lumen_C, lumen_R=lumen_R,
        contact_k=contact_k,
        contact_beta=contact_beta,
        contact_delta=contact_delta,
        contact_mode=contact_mode,
        contact_s_on=contact_s_on,
        contact_s_off=contact_s_off
    )

    info = dict(success=bool(res.success), message=str(res.message),
                nit=int(res.nit), W=float(W), parts=parts, s=s, u_opt=res.x)
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
    L_cmd = 0.049
    mag_len = beam_params.length_of_mag
    m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
    pivot_point = np.array([
        0.7836091530378535, -0.5654053885267907, 0.20700816061967686,
       -3.116988654350607, 0.19059356279735162, 0.028215660130034903
    ])
    start_point = np.array([0.73219777858478, -0.4244994535611557, 0.45711152486163426-.25, -2.4384745401644454, 1.9397464487606495, 0.006339338326166586])
    wire_len = L_cmd - mag_len
    T_ur_pivot = ur_pose6_to_T(pivot_point)   
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)      
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(mode="axial", alpha_end=0.0),
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
    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.05,              # make it longer than rod so distance queries behave well
        n_pts=80,
        bend_axis=np.array([0.0, 0.0, 1.0]),  # bend in x-y plane
        bend_angle=np.deg2rad(120.0),
        bend_start=0.01,
        bend_end=0.08
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
        L0=0.010, Lf=L_cmd, dL=0.001,
        wire_len_fun=lambda L: L - mag_len,
        Kinv_fun=Kbt_inv_profile, u_star=np.zeros(3),
        r_src=r_src_ur, m_src=m_src,
        m_local_fun=model.m_local_fun, m_moment=0.0,
        lumen_C=lumen_C, lumen_R=lumen_R,
        N=60, maxiter=200,
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