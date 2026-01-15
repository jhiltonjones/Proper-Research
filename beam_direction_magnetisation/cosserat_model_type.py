import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
from beam_direction_magnetisation.magnetism.parameters_cosserat import *
from beam_direction_magnetisation.quarternions.rotations import epm_pose_overhead_spin_z, epm_pose_front
from beam_direction_magnetisation.magnetism.beam_geometry import m_local_profile, Kbt_inv_profile
from beam_direction_magnetisation.post_processing.post_processing import compare_magnet_plots, axis3d
# alpha = np.deg2rad(alpha_deg)
# alpha_overhead = np.deg2rad(alpha_deg_overhead)
m_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
m_local_overhead = np.array([mu_line*np.cos(alpha_overhead),0.0, mu_line*np.sin(alpha_overhead)])

def tip_bending_angles_from_tangent(sol, e1=np.array([1.0,0.0,0.0])):
    YL = sol.sol(np.array([L]))
    qL = quat_normalize(YL[3:7, :]) 

    RL = quat_to_rot(qL)[0]          # (3,3) 
    tL = RL @ e1                     # (3,)

    theta_y = np.arctan2(tL[1], tL[0])  
    theta_z = np.arctan2(tL[2], tL[0])   

    theta_total = np.arccos(np.clip(tL[0] / np.linalg.norm(tL), -1.0, 1.0))

    return theta_y, theta_z, theta_total
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
def bc_cosserat_free_twist(Ya, Yb, p0):
    pa = Ya[0:3]
    qa = Ya[3:7]
    ma = Ya[10:13]

    nb = Yb[7:10]
    mb = Yb[10:13]

    qn = qa / np.linalg.norm(qa)
    R0 = quat_to_rot(qn[:, None])[0] 
    e1 = np.array([1.0, 0.0, 0.0])
    t0 = R0 @ e1

    m_body0 = R0.T @ ma

    return np.hstack([
        pa - p0,              # 3: clamp position
        t0[1], t0[2],         # 2: clamp bending angle 
        m_body0[0],           # 1: free twist
        np.linalg.norm(qa) - 1.0,  # 1: quaternion normalization constraint
        nb,                   # 3: free tip force
        mb                    # 3: free tip moment
    ])
Kbt = np.diag([GJ, EI, EI])
Kbt_inv = np.linalg.inv(Kbt)

p0 = np.array([0.0, 0.0, 0.0])
q0 = np.array([1.0, 0.0, 0.0, 0.0]) 

n0 = 120
s_mesh = np.linspace(0, L, n0)

Y_guess = np.zeros((13, n0))
Y_guess[0] = s_mesh    
Y_guess[1] = 0.0     
Y_guess[2] = 0.0          
Y_guess[3] = 1.0           

scales = [1.0]
sol = None
def solve_for_pose(r_src, m_src, Y_guess, s_mesh, m_moment, sol=None):
    ode = make_cosserat_kirchhoff_ode(
        m_src=m_src,
        r_src=r_src,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=m_local_profile,
        m_moment = m_moment,
    )

    soln = solve_bvp(lambda s, Y: ode(s, Y),
                     lambda Ya, Yb: bc_cosserat(Ya, Yb, p0,q0),
                     s_mesh, Y_guess, max_nodes=20000, tol=1e-5)
    return soln
r_tip = np.array([L,0.0,0.0])

angles_deg = np.linspace(-65, 65, 20)

tip_y_front, tip_z_front = [], []
tip_y_over,  tip_z_over  = [], []
bend_y_front, bend_z_front, bend_tot_front = [],[],[]
bend_y_o, bend_z_o, bend_tot_o = [],[],[]
sol_prev_front = None
sol_prev_over  = None
for ang in angles_deg:
    th = np.deg2rad(ang)

    r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=theta_y)

    r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho=rho, phi_z=th)
    # r_src_o, m_src_o = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=theta_y)

    if sol_prev_front is None:
        sol_f = solve_for_pose(r_src_f, m_src_f, Y_guess, s_mesh, m_local)
    else:
        sol_f = solve_for_pose(r_src_f, m_src_f, sol_prev_front.sol(s_mesh), s_mesh, m_local)
    s_out = np.linspace(0, L, 300)
    Yf = sol_f.sol(s_out)
    p_f = Yf[0:3, :]
    q_f = Yf[3:7, :]
    theta_y_fr, theta_z_fr, theta_tot_fr = tip_bending_angles_from_tangent(sol_f)
    print(f"Tip bending angles: theta_y={np.rad2deg(theta_y_fr):+.2f} deg, theta_z={np.rad2deg(theta_z_fr):+.2f} deg, total={np.rad2deg(theta_tot_fr):.2f} deg")
    f_ext, tau_ext, B = magnetic_wrench_density_cosserat_profile(
        p_f, q_f, s_out, m_src_f, r_src_f, m_local_profile, m_local, r_min=1e-6
    )

    B_tip = np.linalg.norm(B[:, -1])

    F_net = np.trapezoid(f_ext, s_out, axis=1)
    T_net = np.trapezoid(tau_ext, s_out, axis=1)

    F_net_mag = np.linalg.norm(F_net)
    T_net_mag = np.linalg.norm(T_net)

    print(f"Front: ang={ang: .1f} deg  |B_tip|={B_tip:.4e} T  |F_net|={F_net_mag:.4e} N  |T_net|={T_net_mag:.4e} N·m/m?")
    if sol_prev_over is None:
        sol_o = solve_for_pose(r_src_o, m_src_o, Y_guess, s_mesh, m_local_overhead)
    else:
        sol_o = solve_for_pose(r_src_o, m_src_o, sol_prev_over.sol(s_mesh), s_mesh, m_local_overhead)
    s_out = np.linspace(0, L, 300)
    Yo = sol_o.sol(s_out)
    p_o = Yo[0:3, :]
    q_o = Yo[3:7, :]

    f_ext_o, tau_ext_o, B_o = magnetic_wrench_density_cosserat_profile(
        p_o, q_o, s_out, m_src_o, r_src_o, m_local_profile, m_local_overhead, r_min=1e-6
    )
    theta_y_o, theta_z_o, theta_tot_o = tip_bending_angles_from_tangent(sol_o)
    print(f"Tip bending angles: theta_y={np.rad2deg(theta_y_o):+.2f} deg, theta_z={np.rad2deg(theta_z_o):+.2f} deg, total={np.rad2deg(theta_tot_o):.2f} deg")
    B_tip_o = np.linalg.norm(B_o[:, -1])

    F_net_o = np.trapezoid(f_ext_o, s_out, axis=1)
    T_net_o = np.trapezoid(tau_ext_o, s_out, axis=1)

    print(f"Overhead: ang={ang: .1f} deg  |B_tip|={B_tip_o:.4e} T  |F_net|={np.linalg.norm(F_net_o):.4e} N  |T_net|={np.linalg.norm(T_net_o):.4e} N·m")
    sol_prev_front, sol_prev_over = sol_f, sol_o

    p_tip_f = sol_f.sol(np.array([L]))[0:3, 0]   
    p_tip_o = sol_o.sol(np.array([L]))[0:3, 0]   

    tip_y_front.append(p_tip_f[1])
    tip_z_front.append(p_tip_f[2])

    tip_y_over.append(p_tip_o[1])
    tip_z_over.append(p_tip_o[2])
    bend_y_front.append(np.rad2deg(theta_y_fr))
    bend_z_front.append(np.rad2deg(theta_z_fr))
    bend_tot_front.append(np.rad2deg(theta_tot_fr))
    bend_y_o.append(np.rad2deg(theta_y_o))
    bend_z_o.append(np.rad2deg(theta_z_o))
    bend_tot_o.append(np.rad2deg(theta_tot_o))
angles_deg = np.asarray(angles_deg)
m_loc = m_local_profile(s_out, m_local)       
print("magnetised points:", np.count_nonzero(np.linalg.norm(m_loc, axis=0) > 0))
print("expected approx:", np.sum((s_out >= s_m) & (s_out <= s_m + ell_m)))
Kinv = Kbt_inv_profile(s_out)
print("Kinv base EI^-1:", Kinv[1,1,0], " tip EI^-1:", Kinv[1,1,-1])
plt.figure()
plt.plot(angles_deg, bend_z_front, marker='o', label="Magnetised at 30 deg bending angle in y-z")
plt.plot(angles_deg, bend_z_o,  marker='o', label="Magnetised at 0 deg and placed overhead bending angle in x-y")
plt.xlabel("Actuation angle [deg]")
plt.ylabel("Tip bending angle θ_z [deg]")
plt.grid(True)
plt.legend()
plt.show()

compare_magnet_plots(angles_deg, tip_y_front, tip_y_over, tip_z_front, tip_z_over)
ang_plot = 60.0  
th_plot = np.deg2rad(ang_plot)

r_tip = np.array([L, 0.0, 0.0])
r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th_plot, theta_y=0)
# r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho, th_plot, m0_dir=np.array([-1.0, 0.0, 0.0]))


sol_f = solve_for_pose(r_src_f, m_src_f, Y_guess, s_mesh, m_local)

s_out = np.linspace(0, L, 300)
Yf = sol_f.sol(s_out)
p_f = Yf[0:3, :]     
q_f = quat_normalize(Yf[3:7, :]) 
m_world = Yf[10:13, :] 
p_pts = p_f.T         
f_pts = f_ext.T    

u = (r_src_f[None, :] - p_pts)                 
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
u_hat = u / np.maximum(u_norm, 1e-12)

f_toward = np.sum(f_pts * u_hat, axis=1)       

print("Local force toward magnet: min/max [N/m] =", f_toward.min(), f_toward.max())
print("Fraction of rod points pulling toward magnet =", np.mean(f_toward > 0.0))
def compute_u_from_solution(s, Y):
    p = Y[0:3, :]
    q = Y[3:7, :]
    m = Y[10:13, :]

    qn = quat_normalize(q)
    R = quat_to_rot(qn)  # (N,3,3)

    # m_body = R^T m
    m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m)  # (3,N)

    Kinv = Kbt_inv_profile(s)  # (3,3,N)
    u = np.einsum('ijn,jn->in', Kinv, m_body)  # (3,N)  
    return u
u = compute_u_from_solution(s_out, Yf)
u1 = u[0, :]  
roll_twist = np.trapezoid(u1, s_out)  # radians
print("Twist-based roll angle:", np.rad2deg(roll_twist), "deg")
R = quat_to_rot(q_f)

m_body = np.einsum('nij,jn->in', np.transpose(R,(0,2,1)), m_world)  

u = (Kbt_inv @ m_body)   

u1 = u[0, :]  
u2 = u[1, :]  
u3 = u[2, :]  

twist_angle_rad = np.trapezoid(u1, s_out) 
twist_angle_deg = np.rad2deg(twist_angle_rad)

print("\nTwist/torsion:")
print("  u1 min/max [rad/m]:", u1.min(), u1.max())
print("  integrated twist angle [deg]:", twist_angle_deg)
x, y, z = p_f[0, :], p_f[1, :], p_f[2, :]

axis3d(x,y,z,r_src_f, m_src_f, s_out, p_f, q_f, ang_plot, f_toward)
