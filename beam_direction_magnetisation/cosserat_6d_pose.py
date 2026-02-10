import numpy as np
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
# from beam_direction_magnetisation.magnetism.parameters_cosserat import *
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile, make_m_local_fun_wire_tip
from beam_direction_magnetisation.quarternions.shared_rotations import Ry, Rz
from proper_research.parameters import default_magnet_params, default_beam_params
from scipy.spatial.transform import Rotation as Rot
mag_params = default_magnet_params()
beam_params = default_beam_params()

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

def make_cosserat_kirchhoff_ode(m_src, r_src, Kinv_fun, m_local_fun,m_moment, wire_len, u_star=None):
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
            f_ext = f_ext + fg            # if you intentionally supply (3,N)        # f_ext = np.zeros_like(p)       # same shape as p (3,N)
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
        Y[0, :] = s          # x ~ s
        Y[3, :] = 1.0        # quaternion w = 1
        return s, Y

    def solve(self, *, L, r_src, m_src, wire_len):
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
            wire_len = wire_len,
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

    def forward(self, *, L, r_src, q_src, m_body, wire_len, s_out_n=300):
        """
        Inputs:
          L: beam length
          r_src: magnet position (3,)
          q_src: magnet orientation quaternion (4,) world frame
          m_body: dipole vector in magnet body frame (3,)

        Outputs: dict with tip pose, bending, field at tip, net F/T, profiles (optional)
        """
        m_src = dipole_from_pose(q_src, m_body)
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
    print(F"DEBUG r_src is: {r_src}")
    # Quaternion for orbit rotation.
    # Build it from axis-angles (world y then world z) to match R_orbit = Ry @ Rz.
    q_y = quat_from_axis_angle([0,1,0], theta_orbit_y)
    q_z = quat_from_axis_angle([0,0,1], theta_orbit_z)
    q_orbit = quat_mul(q_y, q_z)  # consistent with Ry @ Rz

    # Spin about magnet's OWN z (local z): post-multiply
    q_spin = quat_from_axis_angle([0,0,1], beta_spin)
    q_src = quat_mul(q_orbit, q_spin)

    return r_src, quat_normalize_np(q_src)


def  tricepur_pose6_to_T(pose6):
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





if __name__ == "__main__":
    DEBUG = True
    L_cmd = 0.055
    mag_len = beam_params.length_of_mag
    m_body = np.array([mag_params.mag_epm, 0.0, 0.0])
    pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)


    start_point = np.array([
    0.7358772138964855, -0.5598776116145038, 0.1517238977466137, -2.789117819519067, 1.3279582830421093, 0.029877471667822977], float)
    start_point[2] -=0.25
    wire_len = L_cmd - mag_len
    # wire_len = 0.03
    T_ur_pivot = ur_pose6_to_T(pivot_point)   
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)

    T_ur_mag = ur_pose6_to_T(start_point)       # UR TCP pose at magnet
    r_src_ur, q_src_ur = T_to_p_quat_wxyz(T_ur_mag)

    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0),
        m_moment=0.0,
        wire_len = wire_len,
    )

    out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len,m_body=m_body)
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

    out = model.forward(L=L_cmd, r_src=r_src_ur, q_src=q_src_ur, wire_len=wire_len,m_body=m_body)    
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

