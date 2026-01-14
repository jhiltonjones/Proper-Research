import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from beam_direction_magnetisation.magnetism.magnetic_methods import magnetic_force_analytical, dipole_field_from_source
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_derivative_body, quat_normalize, quat_to_rot
MU0_OVER_4PI = 1e-7

def magnetic_moment(B_r, mu_0, r, p):
    return (B_r / mu_0) * (np.pi * r**2 * p)

def magnetic_wrench_density_cosserat(p, q, m_ext, r_src, m_local, r_min=1e-6):
    """
    p: (3,N) world positions
    q: (4,N) quaternions body->world
    m_local: (3,) magnetic moment per unit length in BODY frame
    returns f_ext (3,N), tau_ext (3,N), B (3,N)
    """
    qn = quat_normalize(q)
    R = quat_to_rot(qn)  # (N,3,3)

    # m_global_i = R_i @ m_local
    m_pts = np.einsum('nij,j->ni', R, m_local)  # (N,3)

    r_pts = p.T  # (N,3)
    B = dipole_field_from_source(r_pts, r_src, m_ext, r_min=r_min)       # (N,3)
    f = magnetic_force_analytical(r_pts, m_pts, r_src, m_ext, r_min=r_min)    # (N,3)
    # f = np.zeros_like(f)
    tau = np.cross(m_pts, B)                                             # (N,3)

    return f.T, tau.T, B.T

mag = 128e3
r = 0.0015
E = 3.5e6
A_cs = np.pi * r**2
I = np.pi * r**4 / 4
L = 0.042
nu = 0.49
G = E / (2*(1+nu))
J = 0.5*np.pi*r**4
EI = E*I
GJ = G*J
mu_line = mag * A_cs  

mu_0 = 4e-7*np.pi
B_r = 1.25
r_epm = 0.03
p_epm = 0.09
mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)
m_ext_full = mag_epm * np.array([1.0, 0.0, 0.0])
rho_mat = 6450          # kg/m^3 (material density)
g = 9.81
w = rho_mat * A_cs * g # N/m
f_g = np.array([0.0, 0.0, -w])[:, None]   # (3,1) for broadcasting

def R_y(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def R_z(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def epm_pose_overhead_spin_z(r_tip, rho, phi_z, m0_dir=np.array([1.0, 0.0, 0.0])):
    """
    Overhead magnet fixed above the tip. Rotate magnet about its OWN z-axis by phi_z (radians).
    Assumes magnet-local axes coincide with world axes at phi_z = 0.
    
    r_tip: (3,) tip reference position (e.g. [L,0,0])
    rho: standoff distance in +z [m]
    phi_z: rotation about magnet-local z [rad]
    m0_dir: dipole direction in magnet-local frame at phi_z=0 (default along +x)
    """
    r_src = r_tip + np.array([0.0, 0.0, rho])   # overhead position (fixed)
    Rm = R_z(phi_z)                              # spin about its own z-axis
    m_src = mag_epm * (Rm @ m0_dir)              # dipole moment in world frame
    return r_src, m_src

alpha_deg =50
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
def magnetic_wrench_density(s, v, w, psi, m_ext, r_src, r_min=1e-6):
    """
    External magnetic wrench per unit length along the rod.
    Returns:
      f_ext: (N,3) force density [N/m]
      tau_ext: (N,3) moment density [N] (i.e., N·m per m)
      B: (N,3) field [T]
    """
    r_pts = np.column_stack((s, v, w))               # your current centerline parameterization
    m_pts = m_global_from_psi_vec(psi)               # (N,3), still your current approximation

    B = dipole_field_from_source(r_pts, r_src, m_ext, r_min=r_min)
    tau = np.cross(m_pts, B)                         # moment per length

    f = magnetic_force_analytical(r_pts, m_pts, r_src, m_ext, r_min=r_min)  # force per length

    return f, tau, B

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
    # Base
    pa = Ya[0:3]
    qa = Ya[3:7]
    # Tip
    nb = Yb[7:10]
    mb = Yb[10:13]
    return np.hstack([
        pa - p0,        # 3
        qa - q0,        # 4
        nb,             # 3
        mb              # 3
    ])
# Bending/torsion stiffness (body axes: x torsion, y/z bending)
Kbt = np.diag([GJ, EI, EI])
Kbt_inv = np.linalg.inv(Kbt)

# Base conditions: straight along +x
p0 = np.array([0.0, 0.0, 0.0])
q0 = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (qw,qx,qy,qz)

n0 = 500
s_mesh = np.linspace(0, L, n0)

# Initial guess: straight rod, zero internal wrench
Y_guess = np.zeros((13, n0))
Y_guess[0] = s_mesh           # p_x(s)=s
Y_guess[1] = 0.0              # p_y
Y_guess[2] = 0.0              # p_z
Y_guess[3] = 1.0              # qw=1 constant

def solve_for_pose(r_src, m_src, Y_guess, s_mesh, m_local):
    ode = make_cosserat_kirchhoff_ode(m_src, r_src, Kbt_inv, m_local)
    sol = solve_bvp(lambda s,Y: ode(s,Y),
                    lambda Ya,Yb: bc_cosserat(Ya, Yb, p0, q0),
                    s_mesh, Y_guess, max_nodes=20000, tol=3e-3)
    return sol
def epm_pose_front(r_tip, rho, theta_z, theta_y, m0_dir=np.array([1.0,0.0,0.0])):
    """
    theta_z, theta_y in radians
    m0_dir is the dipole direction in the magnet's local frame
    """
    v0 = np.array([rho, 0.0, 0.0])          # magnet offset in magnet-local coordinates
    Rm = R_y(theta_y) @ R_z(theta_z)        # magnet orientation in world

    r_src = r_tip + Rm @ v0                 # magnet position in world
    m_src = mag_epm * (Rm @ m0_dir)*-1         # dipole moment direction in world
    return r_src, m_src
r_tip = np.array([L,0.0,0.0])

angles_deg = np.linspace(-90, 90, 20)

tip_y_front, tip_z_front = [], []
tip_y_over,  tip_z_over  = [], []

sol_prev_front = None
sol_prev_over  = None
rho = .15
for ang in angles_deg:
    th = np.deg2rad(ang)

    # Front-of-catheter: rotate around y (you can also set theta_z)
    r_src_f, m_src_f = epm_pose_front(r_tip, rho=rho, theta_z=th, theta_y=np.deg2rad(65))

    # Overhead: twist magnet about x by the same angle
    r_src_o, m_src_o = epm_pose_overhead_spin_z(r_tip, rho=rho, phi_z=th)

    # Use continuation using previous solution as the next guess
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

    # Evaluate at the tip s=L (note: pass an array)
    p_tip_f = sol_f.sol(np.array([L]))[0:3, 0]   # shape (3,)
    p_tip_o = sol_o.sol(np.array([L]))[0:3, 0]   # shape (3,)

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
# s_out = np.linspace(0, L, 300)
# Y = sol.sol(s_out)

# # --- Unpack the Cosserat state ---
# p = Y[0:3, :]       # (3,N)
# q = Y[3:7, :]       # (4,N)
# n_int = Y[7:10, :]  # (3,N)
# m_int = Y[10:13, :] # (3,N)

# # Normalize quaternions for safety (numerical drift)
# q = quat_normalize(q)

# # Rotation matrices along the rod (N,3,3)
# R = quat_to_rot(q)

# # Centerline coordinates
# x = p[0, :]
# y = p[1, :]
# z = p[2, :]

# # Tangent (for Kirchhoff/inextensible this should be R*e1)
# e1 = np.array([1.0, 0.0, 0.0])
# t = np.einsum('nij,j->ni', R, e1).T  # (3,N)

# # --- External magnetic wrench density (WORLD frame) ---
# f_ext, tau_ext, B = magnetic_wrench_density_cosserat(
#     p, q, m_ext_full, r_mag, m_local, r_min=1e-6
# )
# # f_ext, tau_ext, B are (3,N)
# fx, fy, fz = f_ext[0, :], f_ext[1, :], f_ext[2, :]
# tau_x, tau_y, tau_z = tau_ext[0, :], tau_ext[1, :], tau_ext[2, :]

# # --- Tip metrics ---
# tip_pos = p[:, -1]
# tip_tangent = t[:, -1]
# base_tangent = t[:, 0]

# bend_angle_deg = np.rad2deg(np.arccos(np.clip(base_tangent @ tip_tangent, -1.0, 1.0)))

# # You can define "tip slope angles" relative to +x in the lab frame
# theta_y_deg = np.rad2deg(np.arctan2(tip_tangent[1], tip_tangent[0]))
# theta_z_deg = np.rad2deg(np.arctan2(tip_tangent[2], tip_tangent[0]))

# # Optional: compute curvature/twist strain u(s) in body frame from internal moment
# Kbt = np.diag([GJ, EI, EI])
# Kbt_inv = np.linalg.inv(Kbt)
# m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m_int)  # (3,N)
# u = (Kbt_inv @ m_body)  # (3,N), assuming u_star = 0
# kappa_x, kappa_y, kappa_z = u[0, :], u[1, :], u[2, :]

# print("Setup:")
# print("  L =", L, "m")
# print("  EI =", EI, "N·m^2")
# print("  GJ =", GJ, "N·m^2")
# print("  mag_epm =", mag_epm, "A·m^2")
# print("  r_mag =", r_mag)

# print("\nSolution summary (Cosserat):")
# print("  tip position [m]:", tip_pos)
# print("  tip y deflection [m]:", tip_pos[1])
# print("  tip z deflection [m]:", tip_pos[2])
# print("  max |y| [m]:", np.max(np.abs(y)))
# print("  max |z| [m]:", np.max(np.abs(z)))
# print("  bend angle base->tip [deg]:", bend_angle_deg)
# print("  tip slope angle in x–y [deg]:", theta_y_deg)
# print("  tip slope angle in x–z [deg]:", theta_z_deg)

# print("\nExternal magnetic loads (densities):")
# print("  |f_ext| min/max [N/m]:", np.min(np.linalg.norm(f_ext, axis=0)), np.max(np.linalg.norm(f_ext, axis=0)))
# print("  |tau_ext| min/max [N]:", np.min(np.linalg.norm(tau_ext, axis=0)), np.max(np.linalg.norm(tau_ext, axis=0)))
# print("  fx min/max [N/m]:", fx.min(), fx.max())
# print("  fy min/max [N/m]:", fy.min(), fy.max())
# print("  fz min/max [N/m]:", fz.min(), fz.max())

# print("\nInternal wrench:")
# print("  |n| tip [N]:", np.linalg.norm(n_int[:, -1]))
# print("  |m| tip [N·m]:", np.linalg.norm(m_int[:, -1]))

# print("\nStrain (curvature/twist) u in body frame:")
# print("  kappa_x (twist) min/max [1/m]:", kappa_x.min(), kappa_x.max())
# print("  kappa_y min/max [1/m]:", kappa_y.min(), kappa_y.max())
# print("  kappa_z min/max [1/m]:", kappa_z.min(), kappa_z.max())

# # --- Plots ---
# fig = plt.figure(figsize=(10, 10))

# ax1 = fig.add_subplot(411)
# ax1.plot(s_out, y)
# ax1.set_ylabel("y(s) [m]")
# ax1.set_title("Cosserat rod centerline: y(s)")

# ax2 = fig.add_subplot(412)
# ax2.plot(s_out, z)
# ax2.set_ylabel("z(s) [m]")
# ax2.set_title("Cosserat rod centerline: z(s)")

# ax3 = fig.add_subplot(413)
# ax3.plot(s_out, np.rad2deg(np.unwrap(np.arctan2(q[3, :], q[0, :]))))
# ax3.set_ylabel("quaternion phase proxy [deg]")
# ax3.set_title("Orientation proxy (optional)")

# ax4 = fig.add_subplot(414)
# ax4.plot(s_out, fy, label="f_y [N/m]")
# ax4.plot(s_out, fz, label="f_z [N/m]")
# ax4.plot(s_out, tau_x, label="tau_x [N]")
# ax4.set_xlabel("s [m]")
# ax4.set_ylabel("load density")
# ax4.legend(loc="best")
# ax4.set_title("Magnetic force and torque densities")

# fig.tight_layout()

# # --- 3D centerline ---
# fig3d = plt.figure(figsize=(8, 6))
# ax = fig3d.add_subplot(111, projection='3d')
# ax.plot(x, y, z, linewidth=2)
# ax.scatter([x[0]], [y[0]], [z[0]], label="base")
# ax.scatter([x[-1]], [y[-1]], [z[-1]], label="tip")

# # Equal-ish aspect scaling
# x_min, x_max = x.min(), x.max()
# y_min, y_max = y.min(), y.max()
# z_min, z_max = z.min(), z.max()
# x_mid = 0.5*(x_min + x_max)
# y_mid = 0.5*(y_min + y_max)
# z_mid = 0.5*(z_min + z_max)
# max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
# half = 0.5*max_range
# ax.set_xlim(x_mid-half, x_mid+half)
# ax.set_ylim(y_mid-half, y_mid+half)
# ax.set_zlim(z_mid-half, z_mid+half)
# ax.set_box_aspect((1, 1, 1))

# ax.set_xlabel("x [m]")
# ax.set_ylabel("y [m]")
# ax.set_zlabel("z [m]")
# ax.set_title("3D Cosserat rod centerline")
# ax.legend()
# plt.tight_layout()
# plt.show()