import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat_profile
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
from beam_direction_magnetisation.magnetism.parameters_cosserat import *
from beam_direction_magnetisation.quarternions.rotations import epm_pose_overhead_spin_z, epm_pose_front
from beam_direction_magnetisation.magnetism.beam_geometry import m_local_profile, Kbt_inv_profile
from beam_direction_magnetisation.post_processing.post_processing import compare_magnet_plots, axis3d
from beam_direction_magnetisation.magnetism.beam_geometry import smooth_top_hat
# alpha = np.deg2rad(alpha_deg)
# alpha_overhead = np.deg2rad(alpha_deg_overhead)
# m_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
# m_local = np.array([mu_line*np.cos(alpha),mu_line*np.sin(alpha),0.0])
# m_local = np.array([mu_line*np.sin(alpha),0.0,mu_line*np.cos(alpha)])
m_local_overhead = np.array([mu_line*np.cos(alpha_overhead),0.0, mu_line*np.sin(alpha_overhead)])
def m_local_profile(s, alpha_end, eps=1e-3):
    """
    Magnetisation only in the tip region [s_m, s_m+ell_m], with direction
    ramping linearly from 0 to alpha_end (radians), in the x-z plane.

    Returns: m_local(s) shape (3, N)
    """
    s = np.atleast_1d(s)
    N = s.size

    # Smooth "turn on/off" for magnitude in the tip region
    w = smooth_top_hat(s, s_m, s_m + ell_m, eps)  # (N,)

    # Normalised coordinate within magnetised tip section (0->1)
    xi = (s - s_m) / (ell_m + 1e-12)

    # Clamp xi to [0,1] so angle is well-defined everywhere
    xi = np.clip(xi, 0.0, 1.0)

    # Linear angle ramp: 0 -> alpha_end across the tip section
    alpha_s = xi * alpha_end  # (N,)

    # Magnetisation direction in x-y plane
    m = np.zeros((3, N))
    m[0, :] = mu_line * np.cos(alpha_s)   # x
    m[1, :] = mu_line * np.sin(alpha_s)   # y
    m[2, :] = 0.0                         # z stays 0

    # Apply magnitude mask (outside tip -> ~0)
    return m * w[None, :]
def closest_point_on_segment(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    c = a + t * ab
    return c
def make_curved_centerline(x_corner=0.06, y_up=0.06, r_turn=0.02,
                           n1=50, n_arc=80, n2=50):
    """
    Centerline: straight + quarter-circle arc + straight.
    r_turn = turning radius (smaller = sharper bend).
    """

    # 1) Straight segment: (0,0) -> (x_corner - r_turn, 0)
    x1 = np.linspace(0.0, x_corner - r_turn, n1)
    seg1 = np.stack([x1, 0*x1, 0*x1], axis=1)

    # 2) Quarter-circle arc centered at (x_corner - r_turn, r_turn)
    # angle from -pi/2 (pointing +x) to 0 (pointing +y)
    theta = np.linspace(-np.pi/2, 0.0, n_arc)
    xc, yc = (x_corner - r_turn), r_turn
    arc = np.stack([xc + r_turn*np.cos(theta),
                    yc + r_turn*np.sin(theta),
                    0*theta], axis=1)

    # 3) Straight segment: end of arc -> (x_corner, y_up)
    y2 = np.linspace(r_turn, y_up, n2)
    seg2 = np.stack([x_corner + 0*y2, y2, 0*y2], axis=1)

    return np.vstack([seg1, arc, seg2])
def closest_point_on_polyline(p, poly):
    # poly: (M,3)
    best_c = None
    best_d2 = np.inf
    for i in range(len(poly) - 1):
        c = closest_point_on_segment(p, poly[i], poly[i+1])
        d2 = np.dot(p - c, p - c)
        if d2 < best_d2:
            best_d2 = d2
            best_c = c
    return best_c, np.sqrt(best_d2)
def wall_force_density(p, poly, R, k_wall=2e4, p_exp=3):
    """
    Quadratic (or higher power) penalty on penetration.
    k_wall: stiffness scaling (increase to reduce penetration)
    p_exp: 2 gives quadratic; 3 even stiffer growth
    """
    N = p.shape[1]
    f = np.zeros_like(p)

    for j in range(N):
        pj = p[:, j]
        cstar, d = closest_point_on_polyline(pj, poly)

        pen = d - R
        if pen > 0.0:
            n = (pj - cstar) / (d + 1e-12)
            f[:, j] = -k_wall * (pen**p_exp) * n

    return f
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

angles_deg = np.linspace(-80, 80, 30)

tip_y_front, tip_z_front = [], []
tip_y_over,  tip_z_over  = [], []
bend_y_front, bend_z_front, bend_tot_front = [],[],[]
bend_y_o, bend_z_o, bend_tot_o = [],[],[]
Btip_front_list, Btip_over_list = [], []
Fnet_front_list, Fnet_over_list = [], []
Tnet_front_list, Tnet_over_list = [], []
sol_prev_front = None
sol_prev_over  = None
for ang in angles_deg:
    th = np.deg2rad(ang)

    r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=theta_y)

    # r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho=rho, phi_z=th)
    r_src_o, m_src_o = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=theta_y)

    if sol_prev_front is None:
        sol_f = solve_for_pose(r_src_f, m_src_f, Y_guess, s_mesh, alpha)
    else:
        sol_f = solve_for_pose(r_src_f, m_src_f, sol_prev_front.sol(s_mesh), s_mesh, alpha)
    s_out = np.linspace(0, L, 300)
    Yf = sol_f.sol(s_out)
    p_f = Yf[0:3, :]
    q_f = Yf[3:7, :]
    theta_y_fr, theta_z_fr, theta_tot_fr = tip_bending_angles_from_tangent(sol_f)
    p_tip = sol_f.sol(np.array([L]))[0:3, 0]
    dist = np.linalg.norm(p_tip - r_src_f)
    print(ang, "dist =", dist)
    print(f"Tip bending angles: theta_y={np.rad2deg(theta_y_fr):+.2f} deg, theta_z={np.rad2deg(theta_z_fr):+.2f} deg, total={np.rad2deg(theta_tot_fr):.2f} deg")
    f_ext, tau_ext, B = magnetic_wrench_density_cosserat_profile(
        p_f, q_f, s_out, m_src_f, r_src_f, m_local_profile, alpha, r_min=1e-6
    )

    B_tip = np.linalg.norm(B[:, -1])

    F_net = np.trapezoid(f_ext, s_out, axis=1)
    T_net = np.trapezoid(tau_ext, s_out, axis=1)

    F_net_mag = np.linalg.norm(F_net)
    T_net_mag = np.linalg.norm(T_net)
    Btip_front_list.append(B_tip)
    Fnet_front_list.append(F_net_mag)
    Tnet_front_list.append(T_net_mag)
    print(f"Front: ang={ang: .1f} deg  |B_tip|={B_tip:.4e} T  |F_net|={F_net_mag:.4e} N  |T_net|={T_net_mag:.4e} N·m/m?")
    if sol_prev_over is None:
        sol_o = solve_for_pose(r_src_o, m_src_o, Y_guess, s_mesh, alpha_overhead)
    else:
        sol_o = solve_for_pose(r_src_o, m_src_o, sol_prev_over.sol(s_mesh), s_mesh, alpha_overhead)
    s_out = np.linspace(0, L, 300)
    Yo = sol_o.sol(s_out)
    p_o = Yo[0:3, :]
    q_o = Yo[3:7, :]

    f_ext_o, tau_ext_o, B_o = magnetic_wrench_density_cosserat_profile(
        p_o, q_o, s_out, m_src_o, r_src_o, m_local_profile, alpha_overhead, r_min=1e-6
    )

    theta_y_o, theta_z_o, theta_tot_o = tip_bending_angles_from_tangent(sol_o)
    print(f"Tip bending angles: theta_y={np.rad2deg(theta_y_o):+.2f} deg, theta_z={np.rad2deg(theta_z_o):+.2f} deg, total={np.rad2deg(theta_tot_o):.2f} deg")
    B_tip_o = np.linalg.norm(B_o[:, -1])

    F_net_o = np.trapezoid(f_ext_o, s_out, axis=1)
    T_net_o = np.trapezoid(tau_ext_o, s_out, axis=1)
    Btip_over_list.append(B_tip_o)
    Fnet_over_list.append(np.linalg.norm(F_net_o))
    Tnet_over_list.append(np.linalg.norm(T_net_o))
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
m_loc = m_local_profile(s_out, alpha)      
m_mag = np.linalg.norm(m_loc, axis=0)
mask = m_mag > 1e-9

angle_profile = np.zeros_like(s_out)
angle_profile[mask] = np.arctan2(m_loc[1, mask], m_loc[0, mask])

# plt.figure()
# plt.plot(s_out, np.rad2deg(angle_profile))
# plt.axvline(s_m, color='k', linestyle='--')
# plt.axvline(s_m + ell_m, color='k', linestyle='--')
# plt.xlabel("s [m]")
# plt.ylabel("Magnetisation angle (x-z) [deg]")
# plt.grid(True)
# plt.show() 
print("magnetised points:", np.count_nonzero(np.linalg.norm(m_loc, axis=0) > 0))
print("expected approx:", np.sum((s_out >= s_m) & (s_out <= s_m + ell_m)))
Kinv = Kbt_inv_profile(s_out)
# print("Kinv base EI^-1:", Kinv[1,1,0], " tip EI^-1:", Kinv[1,1,-1])
plt.figure()
plt.plot(angles_deg, bend_y_front, marker='o', label="Magnetised at 30 deg bending angle in y-z")
plt.plot(angles_deg, bend_y_o,  marker='o', label="Magnetised at 0 deg and placed overhead bending angle in x-y")
plt.xlabel("Actuation angle [deg]")
plt.ylabel("Tip bending angle θ_z [deg]")
plt.grid(True)
plt.legend()
plt.show()

compare_magnet_plots(angles_deg, tip_y_front, tip_y_over, tip_z_front, tip_z_over)
Btip_front_arr = np.array(Btip_front_list)
Btip_over_arr  = np.array(Btip_over_list)

Fnet_front_arr = np.array(Fnet_front_list)
Fnet_over_arr  = np.array(Fnet_over_list)

Tnet_front_arr = np.array(Tnet_front_list)
Tnet_over_arr  = np.array(Tnet_over_list)

# # -------------------------
# # Plot |B_tip| vs angle
# # -------------------------
# plt.figure()
# plt.plot(angles_deg, Btip_front_arr, marker="o", label="Front")
# plt.plot(angles_deg, Btip_over_arr,  marker="o", label="Overhead")
# plt.xlabel("Actuation angle [deg]")
# plt.ylabel("Tip magnetic field |B_tip| [T]")
# plt.title("Tip magnetic field magnitude vs actuation angle")
# plt.grid(True)
# plt.legend()
# plt.show()

# # -------------------------
# # Plot |F_net| vs angle
# # -------------------------
# plt.figure()
# plt.plot(angles_deg, Fnet_front_arr, marker="o", label="Front")
# plt.plot(angles_deg, Fnet_over_arr,  marker="o", label="Overhead")
# plt.xlabel("Actuation angle [deg]")
# plt.ylabel("Net magnetic force |F_net| [N]")
# plt.title("Net force magnitude vs actuation angle")
# plt.grid(True)
# plt.legend()
# plt.show()

# # -------------------------
# # Plot |T_net| vs angle
# # -------------------------
# plt.figure()
# plt.plot(angles_deg, Tnet_front_arr, marker="o", label="Front")
# plt.plot(angles_deg, Tnet_over_arr,  marker="o", label="Overhead")
# plt.xlabel("Actuation angle [deg]")
# plt.ylabel("Net magnetic torque |T_net| [N·m]")
# plt.title("Net torque magnitude vs actuation angle")
# plt.grid(True)
# plt.legend()
# plt.show()
ang_plot = 60.0  
k_wall = 1e3
th_plot = np.deg2rad(ang_plot)
vessel_centerline = make_curved_centerline(x_corner=0.06, y_up=0.06, r_turn=0.04)
R_vessel = 0.006 
r_tip = np.array([L, 0.0, 0.0])
r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th_plot, theta_y=theta_y)
# r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho, th_plot, m0_dir=np.array([-1.0, 0.0, 0.0]))


sol_f = solve_for_pose(r_src_f, m_src_f, Y_guess, s_mesh, alpha)

s_out = np.linspace(0, L, 300)
Yf = sol_f.sol(s_out)
print("BVP success:", sol_f.success)
print("BVP status:", sol_f.status, sol_f.message)

p_f = Yf[0:3, :]     
q_f = quat_normalize(Yf[3:7, :]) 
f_ext, tau_ext, B = magnetic_wrench_density_cosserat_profile(
    p_f, q_f, s_out, m_src_f, r_src_f, m_local_profile, alpha, r_min=1e-6
)
f_wall = wall_force_density(p_f, vessel_centerline, R_vessel, k_wall=k_wall)
f_ext_total = f_ext  + f_g
theta_y_sol, theta_z_sol, theta_tot_sol = tip_bending_angles_from_tangent(sol_f)
print("Max |f_wall|:", np.linalg.norm(f_wall, axis=0).max())
print("Mean |f_wall|:", np.linalg.norm(f_wall, axis=0).mean())
m_world = Yf[10:13, :] 
p_pts = p_f.T         
f_pts = f_ext_total.T    

u = (r_src_f[None, :] - p_pts)                 
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
u_hat = u / np.maximum(u_norm, 1e-12)

f_toward = np.sum(f_pts * u_hat, axis=1)       

print("Local force toward magnet: min/max [N/m] =", f_toward.min(), f_toward.max())
print("Fraction of rod points pulling toward magnet =", np.mean(f_toward > 0.0))
print(f"Bending angle in y: {np.rad2deg(theta_y_sol)} and theta angle in z {np.rad2deg(theta_z_sol)}")
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

# axis3d(x,y,z,r_src_f, m_src_f, s_out, p_f, q_f, ang_plot, f_toward,vessel_centerline, R_vessel)
def max_penetration(p, poly, R):
    N = p.shape[1]
    pens = []
    for j in range(N):
        _, d = closest_point_on_polyline(p[:, j], poly)
        pens.append(d - R)
    pens = np.array(pens)
    return pens.max(), pens

max_pen, pen_profile = max_penetration(p_f, vessel_centerline, R_vessel)
print("Max penetration (d - R):", max_pen)
print("Fraction outside:", np.mean(pen_profile > 0.0))

alpha_candidates_deg = np.linspace(-180, 180, 61)  # 6 degree steps
alpha_candidates = np.deg2rad(alpha_candidates_deg)
def tip_y_from_solution(sol):
    p_tip = sol.sol(np.array([L]))[0:3, 0]
    return p_tip[1]  # y component


def find_best_alpha_for_pose(r_src, m_src, s_mesh, Y_guess, alpha_candidates):
    """
    For one fixed magnet pose (r_src, m_src), sweep alpha_end and return
    the alpha that maximises +y tip deflection.
    """
    best_alpha = None
    best_tip_y = -np.inf
    best_sol = None

    sol_prev = None

    for alpha_end in alpha_candidates:
        # Warm-start: reuse last solution to help convergence
        if sol_prev is None:
            guess = Y_guess
        else:
            guess = sol_prev.sol(s_mesh)

        sol = solve_for_pose(r_src, m_src, guess, s_mesh, alpha_end)

        if not sol.success:
            # if the BVP fails, just skip this alpha
            continue

        y_tip = tip_y_from_solution(sol)
        y_tip = abs(y_tip)
        if y_tip > best_tip_y:
            best_tip_y = y_tip
            best_alpha = alpha_end
            best_sol = sol

        sol_prev = sol

    return best_alpha, best_tip_y, best_sol
angles_deg = np.linspace(-90, 90, 20)
alpha_candidates_deg = np.linspace(0, 180, 61)
alpha_candidates = np.deg2rad(alpha_candidates_deg)

best_alpha_per_act2 = []
best_tip_y_per_act2 = []

sol_best_prev = None

for ang in angles_deg:
    th = np.deg2rad(ang)

    r_src, m_src = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=theta_y)

    if sol_best_prev is None:
        guess = Y_guess
    else:
        guess = sol_best_prev.sol(s_mesh)

    best_alpha, best_tip_y, best_sol = find_best_alpha_for_pose(
        r_src, m_src, s_mesh, guess, alpha_candidates
    )

    best_alpha_per_act2.append(best_alpha)
    best_tip_y_per_act2.append(best_tip_y)

    sol_best_prev = best_sol

    print(f"Actuation {ang:+.1f} deg: best alpha_end = {np.rad2deg(best_alpha):+.1f} deg, y_tip = {best_tip_y:+.4e} m")

best_alpha_per_act2 = np.array(best_alpha_per_act2)
best_tip_y_per_act2 = np.array(best_tip_y_per_act2)

plt.figure()
plt.plot(angles_deg, np.rad2deg(best_alpha_per_act2), marker="o")
plt.xlabel("Actuation angle theta_z [deg]")
plt.ylabel("Optimal magnetisation alpha_end [deg]")
plt.grid(True)
plt.title("Optimal alpha_end for max +y tip deflection")
plt.show()

plt.figure()
plt.plot(angles_deg, best_tip_y_per_act2, marker="o")
plt.xlabel("Actuation angle theta_z [deg]")
plt.ylabel("Max tip deflection in +y [m]")
plt.grid(True)
plt.title("Maximum achievable +y tip deflection vs actuation angle")
plt.show()