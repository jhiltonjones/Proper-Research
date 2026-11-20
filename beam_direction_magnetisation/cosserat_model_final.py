import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_wrench_density_cosserat
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
from beam_direction_magnetisation.magnetism.parameters_cosserat import *
def R_y(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def R_z(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


rho = 0.14

alpha_deg =30
alpha_deg_overhead = 0
alpha = np.deg2rad(alpha_deg)
alpha_overhead = np.deg2rad(alpha_deg_overhead)
m_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])
m_local_overhead = np.array([mu_line*np.cos(alpha_overhead),0.0, mu_line*np.sin(alpha_overhead)])

def m_global_from_psi_vec(psi_vec):
    mx, my, mz = m_local
    c = np.cos(psi_vec); s = np.sin(psi_vec)
    return np.column_stack([mx*np.ones_like(psi_vec),
                            my*c - mz*s,
                            my*s + mz*c])
def epm_pose_overhead_spin_z(r_tip, rho, phi_z, m0_dir=np.array([-1.0, 0.0, 0.0])):
    r_src = r_tip + np.array([0.0, 0.0, rho])  
    Rm = R_z(phi_z)                             
    m_src = mag_epm * (Rm @ m0_dir)              
    return r_src, m_src
def epm_pose_front(r_tip, rho, theta_z, theta_y, m0_dir=np.array([1.0,0.0,0.0])):

    v0 = np.array([rho, 0.0, 0.0])      
    Rm = R_y(theta_y) @ R_z(theta_z)       

    r_src = r_tip + Rm @ v0            
    m_src = mag_epm * (Rm @ m0_dir) *1   
    return r_src, m_src


def make_cosserat_kirchhoff_ode(m_src, r_src, Kbt_inv, m_local, u_star=None):
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

        m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m)
        u = (Kbt_inv @ m_body) + u_star[:, None]

        q_s = quat_derivative_body(qn, u)

        f_ext, tau_ext, _B = magnetic_wrench_density_cosserat(
            p, qn, m_src, r_src, m_local, r_min=1e-6
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
Kbt = np.diag([GJ, EI, EI])
Kbt_inv = np.linalg.inv(Kbt)

p0 = np.array([0.0, 0.0, 0.0])
q0 = np.array([1.0, 0.0, 0.0, 0.0]) 

n0 = 400
s_mesh = np.linspace(0, L, n0)

Y_guess = np.zeros((13, n0))
Y_guess[0] = s_mesh    
Y_guess[1] = 0.0     
Y_guess[2] = 0.0          
Y_guess[3] = 1.0           

scales = [1.0]
sol = None
def solve_for_pose(r_src, m_src, Y_guess, s_mesh, m_local, sol=None):
    for lam in scales:
        ode = make_cosserat_kirchhoff_ode(m_src, r_src, Kbt_inv, m_local)

        if sol is None:
            sol = solve_bvp(lambda s, Y: ode(s, Y),
                            lambda Ya, Yb: bc_cosserat(Ya, Yb, p0, q0),
                            s_mesh, Y_guess, max_nodes=20000, tol=3e-3)
        else:
            y_init = sol.sol(s_mesh)
            sol = solve_bvp(lambda s, Y: ode(s, Y),
                            lambda Ya, Yb: bc_cosserat(Ya, Yb, p0, q0),
                            s_mesh, y_init, max_nodes=20000, tol=3e-3)

        print("lambda", lam, "success", sol.success, "nodes", sol.x.size, "msg", sol.message)
        if not sol.success:
            break
    return sol
r_tip = np.array([L,0.0,0.0])

angles_deg = np.linspace(-80, 80, 20)

tip_y_front, tip_z_front = [], []
tip_y_over,  tip_z_over  = [], []

sol_prev_front = None
sol_prev_over  = None
rho = .16
for ang in angles_deg:
    th = np.deg2rad(ang)

    r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=np.deg2rad(-30))

    r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho=rho, phi_z=th)

    if sol_prev_front is None:
        sol_f = solve_for_pose(r_src_f, m_src_f, Y_guess, s_mesh, m_local)
    else:
        sol_f = solve_for_pose(r_src_f, m_src_f, sol_prev_front.sol(s_mesh), s_mesh, m_local)
    s_out = np.linspace(0, L, 300)
    Yf = sol_f.sol(s_out)
    p_f = Yf[0:3, :]
    q_f = Yf[3:7, :]

    f_ext, tau_ext, B = magnetic_wrench_density_cosserat(
        p_f, q_f, m_src_f, r_src_f, m_local, r_min=1e-6
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

    f_ext_o, tau_ext_o, B_o = magnetic_wrench_density_cosserat(
        p_o, q_o, m_src_o, r_src_o, m_local_overhead, r_min=1e-6
    )

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
angles_deg = np.asarray(angles_deg)

plt.figure()
plt.plot(angles_deg, 1e3*np.array(tip_y_front), marker='o', label="Front (rotate magnet)")
plt.plot(angles_deg, 1e3*np.array(tip_y_over),  marker='o', label="Overhead (spin about z)")
plt.xlabel("Actuation angle [deg]")
plt.ylabel("Tip y deflection [mm]")
plt.title("Tip y deflection vs actuation angle")
plt.grid(True)
plt.legend()
plt.show()
plt.figure()
plt.plot(1e3*np.array(tip_y_front), 1e3*np.array(tip_z_front), marker='o', label="Front")
plt.plot(1e3*np.array(tip_y_over),  1e3*np.array(tip_z_over),  marker='o', label="Overhead")
plt.xlabel("Tip y [mm]")
plt.ylabel("Tip z [mm]")
plt.title("Tip trajectory in y–z plane")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
plt.figure()
plt.plot(angles_deg, 1e3*np.array(tip_z_front), marker='o', label="Front (rotate magnet)")
plt.plot(angles_deg, 1e3*np.array(tip_z_over),  marker='o', label="Overhead (spin about z)")
plt.xlabel("Actuation angle [deg]")
plt.ylabel("Tip z deflection [mm]")
plt.title("Tip z deflection vs actuation angle")
plt.grid(True)
plt.legend()
plt.show()
ang_plot = -60.0  
th_plot = np.deg2rad(ang_plot)

r_tip = np.array([L, 0.0, 0.0])
r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th_plot, theta_y=np.deg2rad(-30))
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
print("Fraction of rod points pulling toward magnet =",
      np.mean(f_toward > 0.0))

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
from mpl_toolkits.mplot3d import Axes3D  

fig3d = plt.figure(figsize=(8, 6))
ax = fig3d.add_subplot(111, projection='3d')

ax.plot(x, y, z, linewidth=2, label="rod centerline")
ax.scatter([x[0]], [y[0]], [z[0]], s=40, label="base")
ax.scatter([x[-1]], [y[-1]], [z[-1]], s=40, label="tip")
pts = np.column_stack([x, y, z])          
pts = np.vstack([pts, r_src_f[None, :]]) 

xmin, ymin, zmin = pts.min(axis=0)
xmax, ymax, zmax = pts.max(axis=0)

x_mid = 0.5*(xmin + xmax)
y_mid = 0.5*(ymin + ymax)
z_mid = 0.5*(zmin + zmax)

max_range = max(xmax-xmin, ymax-ymin, zmax-zmin)
half = 0.5*max_range

margin = 0.15 * max_range
half = half + margin

ax.set_xlim(x_mid-half, x_mid+half)
ax.set_ylim(y_mid-half, y_mid+half)
ax.set_zlim(z_mid-half, z_mid+half)
ax.set_box_aspect((1, 1, 1))

ax.scatter([r_src_f[0]], [r_src_f[1]], [r_src_f[2]], s=60, marker='^', label="EPM")

mhat = m_src_f / np.linalg.norm(m_src_f)
arrow_len = 0.02  
ax.quiver(r_src_f[0], r_src_f[1], r_src_f[2],
          mhat[0], mhat[1], mhat[2],
          length=arrow_len, normalize=True)
R = quat_to_rot(q_f)
step = 30             
dlen = 0.005        

for i in range(0, len(s_out), step):
    Ri = R[i, :, :]         
    pi = p_f[:, i]

    d1 = Ri @ np.array([1.0, 0.0, 0.0])
    d2 = Ri @ np.array([0.0, 1.0, 0.0])
    d3 = Ri @ np.array([0.0, 0.0, 1.0])

    ax.quiver(pi[0], pi[1], pi[2], d1[0], d1[1], d1[2], length=dlen, normalize=True)
    ax.quiver(pi[0], pi[1], pi[2], d2[0], d2[1], d2[2], length=dlen, normalize=True)
    ax.quiver(pi[0], pi[1], pi[2], d3[0], d3[1], d3[2], length=dlen, normalize=True)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title(f"Front magnet case: rod shape at angle = {ang_plot:.1f} deg")
ax.legend()
plt.tight_layout()
plt.show()
plt.figure()
plt.plot(s_out, f_toward)
plt.axhline(0.0, linewidth=1)
plt.xlabel("s [m]")
plt.ylabel("f · u_hat [N/m]")
plt.title("Local magnetic force component toward magnet")
plt.grid(True)
plt.show()
