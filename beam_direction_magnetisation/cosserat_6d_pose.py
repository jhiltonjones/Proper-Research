import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize
from beam_direction_magnetisation.magnetism.parameters_cosserat import *
from beam_direction_magnetisation.quarternions.rotations import epm_pose_overhead_spin_z, epm_pose_front
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.post_processing.post_processing import compare_magnet_plots, axis3d
from beam_direction_magnetisation.magnetism.beam_geometry import smooth_top_hat
from beam_direction_magnetisation.quarternions.shared_rotations import Rx, Ry, Rz, unpack_pose_euler_L, quat_to_rot_wxyz,rot_to_euler_zyx
from proper_research.parameters import default_magnet_params
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
def quat_to_R(q):
    qn = quat_normalize(q.reshape(4, 1))
    return quat_to_rot(qn)[0]  # now correct: (1,3,3)[0] -> (3,3)

def dipole_from_pose(q_src, m_body):
    R = quat_to_R(q_src)
    return R @ m_body  # (3,)

def make_cosserat_kirchhoff_ode(m_src, r_src, Kinv_fun, m_local_fun,m_moment,  u_star=None):
    e1 = np.array([1.0, 0.0, 0.0])
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
        Kinv = Kinv_fun(s)                                           # (3,3,N)
        u = np.einsum('ijn,jn->in', Kinv, m_body) + u_star[:,None]    # (3,N)

        q_s = quat_derivative_body(qn, u)

        f_ext, tau_ext, _B = magnetic_wrench_density_cosserat_profile(
            p, qn, s, m_src, r_src, m_local_fun, m_moment, r_min=1e-6
        )
        # f_wall = wall_force_density(p, vessel_centerline, R_vessel, k_wall=k_wall)
        # f_ext = f_ext + f_wall
        f_ext = f_ext + f_g

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
    def __init__(self, *, p0, q0, Kinv_fun, m_local_fun, m_moment,
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

    def _initial_guess(self, L):
        s = np.linspace(0.0, L, self.n_nodes)
        Y = np.zeros((13, self.n_nodes))
        Y[0, :] = s          # x ~ s
        Y[3, :] = 1.0        # quaternion w = 1
        return s, Y

    def solve(self, *, L, r_src, m_src):
        L = float(L)
        r_src = np.asarray(r_src, float).reshape(3,)
        m_src = np.asarray(m_src, float).reshape(3,)

        if self._sol_prev is None:
            s_mesh, Y_guess = self._initial_guess(L)
        else:
            s_mesh = np.linspace(0.0, L, self.n_nodes)
            Y_guess = self._sol_prev.sol(s_mesh)

        ode = make_cosserat_kirchhoff_ode(
            m_src=m_src,
            r_src=r_src,
            Kinv_fun=self.Kinv_fun,
            m_local_fun=self.m_local_fun,
            m_moment=self.m_moment,
        )

        sol = solve_bvp(
            lambda s, Y: ode(s, Y),
            lambda Ya, Yb: bc_cosserat(Ya, Yb, self.p0, self.q0),
            s_mesh, Y_guess,
            tol=self.tol,
            max_nodes=self.max_nodes
        )

        self._sol_prev = sol
        return sol

    def forward(self, *, L, r_src, q_src, m_body, s_out_n=300):
        """
        Inputs:
          L: beam length
          r_src: magnet position (3,)
          q_src: magnet orientation quaternion (4,) world frame
          m_body: dipole vector in magnet body frame (3,)

        Outputs: dict with tip pose, bending, field at tip, net F/T, profiles (optional)
        """
        m_src = dipole_from_pose(q_src, m_body)  # (3,)
        sol = self.solve(L=L, r_src=r_src, m_src=m_src)

        # Evaluate along rod for outputs
        s_out = np.linspace(0.0, float(L), int(s_out_n))
        Y = sol.sol(s_out)
        p = Y[0:3, :]
        q = quat_normalize(Y[3:7, :])

        f_ext, tau_ext, B = magnetic_wrench_density_cosserat_profile(
            p, q, s_out, m_src, r_src, self.m_local_fun, self.m_moment, r_min=1e-6
        )
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
    v0_body=np.array([1.0, 0.0, 0.0]),
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


def quat_to_rot(q):
    """
    Quaternion(s) -> rotation matrix/matrices.

    Accepts:
      q shape (4,)         -> returns (1,3,3)
      q shape (4,N)        -> returns (N,3,3)
      q shape (N,4)        -> returns (N,3,3)

    Quaternion convention: [w, x, y, z]
    """
    q = np.asarray(q, dtype=float)

    # Normalize input shape to (4, N)
    if q.ndim == 1:
        if q.size != 4:
            raise ValueError(f"quat_to_rot expected 4 elements, got {q.size}: {q}")
        q = q.reshape(4, 1)
    elif q.ndim == 2:
        if q.shape[0] == 4:
            pass
        elif q.shape[1] == 4:
            q = q.T
        else:
            raise ValueError(f"quat_to_rot expected shape (4,N) or (N,4); got {q.shape}")
    else:
        raise ValueError(f"quat_to_rot expected 1D or 2D array; got ndim={q.ndim}")

    # Normalize each quaternion column
    n = np.linalg.norm(q, axis=0, keepdims=True)
    q = q / (n + 1e-12)

    w, x, y, z = q[0, :], q[1, :], q[2, :], q[3, :]

    # Build rotation matrices, vectorized over N
    R = np.empty((q.shape[1], 3, 3), dtype=float)

    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)

    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)

    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)

    return R
import numpy as np

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi





def pose6_from_r_quat_wxyz(r, q):
    r = np.asarray(r, float).ravel()
    assert r.size == 3
    R = quat_to_rot_wxyz(q)
    roll, pitch, yaw = rot_to_euler_zyx(R)
    # wrap to match MPC-style printing
    roll, pitch, yaw = wrap_pi(roll), wrap_pi(pitch), wrap_pi(yaw)
    return np.array([r[0], r[1], r[2], roll, pitch, yaw], float)

def make_safe_forward_tip_fn(model, m_body, p_min, p_max, penalty=1e3):
    L_lo = float(p_min[6])
    L_hi = float(p_max[6])

    def safe_forward_tip(p):
        p = np.asarray(p, float).ravel()
        if p.size != 7 or (not np.all(np.isfinite(p))):
            return np.array([penalty, penalty, penalty], float)

        # Extract pose
        r_src, q_src, L = unpack_pose_euler_L(p)

        # HARD GUARD: never allow invalid L into solve_bvp
        if (not np.isfinite(L)) or (L <= 1e-9):
            return np.array([penalty, penalty, penalty], float)

        # During optimization, clip L to physical range (trust-constr probes outside constraints)
        L = float(np.clip(L, L_lo, L_hi))

        try:
            out = model.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body)
        except Exception:
            return np.array([penalty, penalty, penalty], float)

        if not out.get("solved", False):
            return np.array([penalty, penalty, penalty], float)

        return np.asarray(out["p_tip"], float).reshape(3,)

    return safe_forward_tip
def forward_cosserat_from_pose_euler_L(p, model, *, m_body, return_mode="tip_xyz"):
    p = np.asarray(p, float).ravel()
    if p.size != 7:
        raise ValueError("Expected [x,y,z,roll,pitch,yaw,L]")
    r_src, q_src, L = unpack_pose_euler_L(p)

    out = model.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body)

    if not out["solved"]:
        # Important for optimizers: return finite penalty or NaNs
        # I suggest a large penalty vector, not NaN.
        if return_mode == "tip_xyz":
            return np.array([1e3, 1e3, 1e3])
        elif return_mode == "tip_xyz_bend":
            return np.array([1e3, 1e3, 1e3, 1e3, 1e3])
        else:
            raise ValueError("unknown return_mode")

    p_tip = out["p_tip"]  # (3,)
    if return_mode == "tip_xyz":
        return p_tip

    if return_mode == "tip_xyz_bend":
        return np.array([p_tip[0], p_tip[1], p_tip[2],
                         float(out["theta_y"]), float(out["theta_z"])])
import numpy as np
from scipy.spatial.transform import Rotation as Rot

def ur_pose6_to_T(pose6):
    """
    UR RTDE TCP pose6: [x, y, z, rx, ry, rz]
    where [rx,ry,rz] is rotation vector (Rodrigues), radians.
    Returns 4x4 transform.
    """
    pose6 = np.asarray(pose6, float).ravel()
    assert pose6.size == 6

    p = pose6[:3]
    rvec = pose6[3:6]

    R = Rot.from_rotvec(rvec).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def T_to_p_quat_wxyz(T):
    """
    Convert 4x4 transform -> position (3,) and quaternion [w,x,y,z].
    """
    p = T[:3, 3].copy()
    R = T[:3, :3]
    q_xyzw = Rot.from_matrix(R).as_quat()  # returns [x,y,z,w]
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], float)
    q_wxyz /= (np.linalg.norm(q_wxyz) + 1e-12)
    return p, q_wxyz   
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# -----------------------------
# UR Pose (RTDE) helpers
# -----------------------------
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


# -----------------------------
# Build SIM frame from pivot+start
# -----------------------------
def build_T_sim_from_ur_using_two_points(pivot_pose6, start_pose6):
    """
    Creates a mapping T_sim_from_ur such that:
      - pivot position maps to (0,0,0)
      - start position maps to (+0.15, 0, 0)
    while preserving right-handedness.

    Orientation convention:
      - sim +X axis points from pivot -> start
      - sim +Z axis is chosen close to UR base +Z
      - sim +Y completes a right-handed frame
    """
    pivot_pose6 = np.asarray(pivot_pose6, float).ravel()
    start_pose6 = np.asarray(start_pose6, float).ravel()

    p_pivot = pivot_pose6[:3]
    p_start = start_pose6[:3]

    d = p_start - p_pivot
    dist = np.linalg.norm(d)
    if dist < 1e-9:
        raise ValueError("pivot_point and start_point positions are identical; cannot define +X axis.")

    # Define sim X axis (in UR base coordinates)
    x_hat = d / dist  # pivot->start direction

    # Choose a reference "up" direction from UR base frame
    up_ref = np.array([0.0, 0.0, 1.0])

    # If x_hat is nearly parallel to up_ref, use Y as fallback
    if abs(np.dot(x_hat, up_ref)) > 0.95:
        up_ref = np.array([0.0, 1.0, 0.0])

    # Build a right-handed basis: z_hat ⟂ x_hat, y_hat = z_hat × x_hat
    z_hat = up_ref - np.dot(up_ref, x_hat) * x_hat
    z_hat /= (np.linalg.norm(z_hat) + 1e-12)

    y_hat = np.cross(z_hat, x_hat)
    y_hat /= (np.linalg.norm(y_hat) + 1e-12)

    # Recompute z_hat to ensure exact orthonormality
    z_hat = np.cross(x_hat, y_hat)
    z_hat /= (np.linalg.norm(z_hat) + 1e-12)

    # Rotation mapping UR->SIM:
    # p_sim = R_map (p_ur - p_pivot)
    R_map = np.vstack([x_hat, y_hat, z_hat])  # rows are sim axes expressed in UR

    t_map = -R_map @ p_pivot

    T_sim_from_ur = np.eye(4)
    T_sim_from_ur[:3, :3] = R_map
    T_sim_from_ur[:3, 3] = t_map

    return T_sim_from_ur, dist
def make_T_from_p_quat_wxyz(p, q_wxyz):
    """Build 4x4 transform from position p (3,) and quaternion [w,x,y,z]."""
    p = np.asarray(p, float).reshape(3,)
    q_wxyz = np.asarray(q_wxyz, float).reshape(4,)
    # scipy expects [x,y,z,w]
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], float)
    Rm = Rot.from_quat(q_xyzw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = p
    return T

def T_to_ur_pose6(T):
    """
    Convert 4x4 transform into UR pose6: [x,y,z, rx,ry,rz]
    where [rx,ry,rz] is rotation vector (axis-angle), radians.
    """
    p = T[:3, 3].copy()
    Rm = T[:3, :3]
    rvec = Rot.from_matrix(Rm).as_rotvec()
    return np.hstack([p, rvec])

if __name__ == "__main__":
    DEBUG = True

    pivot_point = np.array([
        0.7836091530378535, -0.5654053885267907, 0.20700816061967686,
       -2.081278830177124,  2.3523542602060017, 0.040846003961920924
    ])

    start_point = np.array([
        0.6336091530378535, -0.5654053885267907, 0.20700816061967686,
       -2.081278830177124,  2.3523542602060017, 0.040846003961920924
    ])
    # 1) Build mapping UR->SIM once
    T_sim_from_ur, dist = build_T_sim_from_ur_using_two_points(pivot_point, start_point)
    T_ur_from_sim = np.linalg.inv(T_sim_from_ur)

    if DEBUG:
        print("UR distance pivot->start =", dist)

    # 2) Map pivot into SIM -> base BCs
    T_sim_pivot = T_sim_from_ur @ ur_pose6_to_T(pivot_point)
    p0_sim, q0_sim = T_to_p_quat_wxyz(T_sim_pivot)

    if DEBUG:
        print("Pivot mapped to SIM p0_sim =", p0_sim)
        T_sim_start = T_sim_from_ur @ ur_pose6_to_T(start_point)
        p_start_sim, _ = T_to_p_quat_wxyz(T_sim_start)
        print("Start mapped to SIM p =", p_start_sim)

    # 3) Build model
    model = CosseratForwardModel(
        p0=p0_sim,
        q0=q0_sim,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(mode="axial", alpha_end=0.0),
        m_moment=0.0,
        n_nodes=120,
        tol=1e-5
    )

    L_cmd = 0.05
    rho = 0.1
    r_tip_nominal = p0_sim + np.array([L_cmd, 0.0, 0.0])
    r_pivot_sim = p0_sim
    r_src_cmd, q_src_cmd = epm_pose_orbit_and_spin(
        r_pivot_sim, rho=rho,
        theta_orbit_z=np.deg2rad(40.0),
        beta_spin=np.deg2rad(0.0),
        theta_orbit_y=0.0
    )

    # 5) Forward solve
    m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
    out = model.forward(L=L_cmd, r_src=r_src_cmd, q_src=q_src_cmd, m_body=m_body)

    # 6) Convert SIM magnet pose back to UR command pose
    T_sim_mag_cmd = make_T_from_p_quat_wxyz(r_src_cmd, q_src_cmd)

    T_tcp_from_mag = np.eye(4)  # mount correction, identity for now
    T_ur_mag_cmd = T_ur_from_sim @ T_sim_mag_cmd @ T_tcp_from_mag
    ur_mag_pose6_cmd = T_to_ur_pose6(T_ur_mag_cmd)

    if DEBUG:
        print("UR magnet pose6 cmd:", ur_mag_pose6_cmd)