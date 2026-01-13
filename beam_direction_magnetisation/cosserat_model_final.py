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

def R_y(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def R_z(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def magnet_pose_about_tip(L, rho, theta_z_deg, theta_y_deg):
    r_tip = np.array([L,0.0,0.0])
    thz = np.deg2rad(theta_z_deg)
    thy = np.deg2rad(theta_y_deg)
    v0 = np.array([rho,0.0,0.0])
    v = R_y(thy) @ (R_z(thz) @ v0)
    return r_tip + v

rho = 0.14
theta_z_deg = 70.0
theta_y_deg = -50.0
r_mag = magnet_pose_about_tip(L, rho, theta_z_deg, theta_y_deg)

# MU0_OVER_4PI = 1e-7  # μ0/(4π)


alpha_deg =-50
alpha = np.deg2rad(alpha_deg)
m_local = np.array([mu_line*np.cos(alpha),0.0, mu_line*np.sin(alpha)])

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
# def make_ode(m_ext_scale):
#     m_ext = m_ext_scale * m_ext_full

#     def ode(s_eval, Y, eps=5e-5):
#         v, vp, w, wp, psi, psip, Vy, Vz, My, Mz = Y
#         f, tau, _B0 = magnetic_wrench_density(s_eval, v, w, psi, m_ext, r_mag, r_min=1e-6)
#         fx, fy, fz = f[:,0], f[:,1], f[:,2]
#         tau_x, tau_y, tau_z = tau[:,0], tau[:,1], tau[:,2]
#         tau_x, tau_y, tau_z = tau[:,0], tau[:,1], tau[:,2]
#         dY = np.zeros_like(Y)
#         dY[0] = vp
#         dY[1] = Mz / EI
#         dY[2] = wp
#         dY[3] = -My / EI
#         dY[4] = psip
#         dY[5] = tau_x / GJ
#         dY[6] = -fy
#         dY[7] = -fz
#         dY[8] = Vz + tau_y
#         dY[9] = Vy + tau_z
#         return dY
#     return ode

# def bc(Ya, Yb):
#     v0, vp0, w0, wp0, psi0, psip0, Vy0, Vz0, My0, Mz0 = Ya
#     vL, vpL, wL, wpL, psiL, psipL, VyL, VzL, MyL, MzL = Yb
#     return np.array([v0, vp0, w0, wp0, psi0,
#                      VyL, VzL, MyL, MzL, psipL])
def make_cosserat_kirchhoff_ode(m_ext_scale, Kbt_inv, m_local, u_star=None):
    m_ext = m_ext_scale * m_ext_full
    e1 = np.array([1.0, 0.0, 0.0])

    if u_star is None:
        u_star = np.zeros(3)  # intrinsic curvature/twist

    def ode(s, Y):
        # Unpack state
        p = Y[0:3]      # (3,N)
        q = Y[3:7]      # (4,N)
        n = Y[7:10]     # (3,N) internal force
        m = Y[10:13]    # (3,N) internal moment

        qn = quat_normalize(q)
        R = quat_to_rot(qn)               # (N,3,3)

        # Kirchhoff kinematics: p' = R e1
        p_s = np.einsum('nij,j->ni', R, e1).T  # (3,N)

        # Constitutive: u (body) = Kbt^{-1} * (R^T m_world) + u_star
        # Compute m_body = R^T m_world
        m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m)  # (3,N)
        u = (Kbt_inv @ m_body) + u_star[:, None]                       # (3,N)

        # Quaternion kinematics
        q_s = quat_derivative_body(qn, u)

        # External magnetic wrench density in WORLD frame
        f_ext, tau_ext, _B = magnetic_wrench_density_cosserat(
            p, qn, m_ext, r_mag, m_local, r_min=1e-6
        )

        # Equilibrium
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

n0 = 120
s_mesh = np.linspace(0, L, n0)

# Initial guess: straight rod, zero internal wrench
Y_guess = np.zeros((13, n0))
Y_guess[0] = s_mesh           # p_x(s)=s
Y_guess[1] = 0.0              # p_y
Y_guess[2] = 0.0              # p_z
Y_guess[3] = 1.0              # qw=1 constant

scales = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
sol = None

for lam in scales:
    ode = make_cosserat_kirchhoff_ode(lam, Kbt_inv, m_local)

    if sol is None:
        sol = solve_bvp(lambda s, Y: ode(s, Y),
                        lambda Ya, Yb: bc_cosserat(Ya, Yb, p0, q0),
                        s_mesh, Y_guess, max_nodes=20000, tol=3e-3)
    else:
        # reuse previous solution as initial guess
        y_init = sol.sol(s_mesh)
        sol = solve_bvp(lambda s, Y: ode(s, Y),
                        lambda Ya, Yb: bc_cosserat(Ya, Yb, p0, q0),
                        s_mesh, y_init, max_nodes=20000, tol=3e-3)

    print("lambda", lam, "success", sol.success, "nodes", sol.x.size, "msg", sol.message)
    if not sol.success:
        break



s_out = np.linspace(0, L, 300)
Y = sol.sol(s_out)

# --- Unpack the Cosserat state ---
p = Y[0:3, :]       # (3,N)
q = Y[3:7, :]       # (4,N)
n_int = Y[7:10, :]  # (3,N)
m_int = Y[10:13, :] # (3,N)

# Normalize quaternions for safety (numerical drift)
q = quat_normalize(q)

# Rotation matrices along the rod (N,3,3)
R = quat_to_rot(q)

# Centerline coordinates
x = p[0, :]
y = p[1, :]
z = p[2, :]

# Tangent (for Kirchhoff/inextensible this should be R*e1)
e1 = np.array([1.0, 0.0, 0.0])
t = np.einsum('nij,j->ni', R, e1).T  # (3,N)

# --- External magnetic wrench density (WORLD frame) ---
f_ext, tau_ext, B = magnetic_wrench_density_cosserat(
    p, q, m_ext_full, r_mag, m_local, r_min=1e-6
)
# f_ext, tau_ext, B are (3,N)
fx, fy, fz = f_ext[0, :], f_ext[1, :], f_ext[2, :]
tau_x, tau_y, tau_z = tau_ext[0, :], tau_ext[1, :], tau_ext[2, :]

# --- Tip metrics ---
tip_pos = p[:, -1]
tip_tangent = t[:, -1]
base_tangent = t[:, 0]

bend_angle_deg = np.rad2deg(np.arccos(np.clip(base_tangent @ tip_tangent, -1.0, 1.0)))

# You can define "tip slope angles" relative to +x in the lab frame
theta_y_deg = np.rad2deg(np.arctan2(tip_tangent[1], tip_tangent[0]))
theta_z_deg = np.rad2deg(np.arctan2(tip_tangent[2], tip_tangent[0]))

# Optional: compute curvature/twist strain u(s) in body frame from internal moment
Kbt = np.diag([GJ, EI, EI])
Kbt_inv = np.linalg.inv(Kbt)
m_body = np.einsum('nij,jn->in', np.transpose(R, (0,2,1)), m_int)  # (3,N)
u = (Kbt_inv @ m_body)  # (3,N), assuming u_star = 0
kappa_x, kappa_y, kappa_z = u[0, :], u[1, :], u[2, :]

print("Setup:")
print("  L =", L, "m")
print("  EI =", EI, "N·m^2")
print("  GJ =", GJ, "N·m^2")
print("  mag_epm =", mag_epm, "A·m^2")
print("  r_mag =", r_mag)

print("\nSolution summary (Cosserat):")
print("  tip position [m]:", tip_pos)
print("  tip y deflection [m]:", tip_pos[1])
print("  tip z deflection [m]:", tip_pos[2])
print("  max |y| [m]:", np.max(np.abs(y)))
print("  max |z| [m]:", np.max(np.abs(z)))
print("  bend angle base->tip [deg]:", bend_angle_deg)
print("  tip slope angle in x–y [deg]:", theta_y_deg)
print("  tip slope angle in x–z [deg]:", theta_z_deg)

print("\nExternal magnetic loads (densities):")
print("  |f_ext| min/max [N/m]:", np.min(np.linalg.norm(f_ext, axis=0)), np.max(np.linalg.norm(f_ext, axis=0)))
print("  |tau_ext| min/max [N]:", np.min(np.linalg.norm(tau_ext, axis=0)), np.max(np.linalg.norm(tau_ext, axis=0)))
print("  fx min/max [N/m]:", fx.min(), fx.max())
print("  fy min/max [N/m]:", fy.min(), fy.max())
print("  fz min/max [N/m]:", fz.min(), fz.max())

print("\nInternal wrench:")
print("  |n| tip [N]:", np.linalg.norm(n_int[:, -1]))
print("  |m| tip [N·m]:", np.linalg.norm(m_int[:, -1]))

print("\nStrain (curvature/twist) u in body frame:")
print("  kappa_x (twist) min/max [1/m]:", kappa_x.min(), kappa_x.max())
print("  kappa_y min/max [1/m]:", kappa_y.min(), kappa_y.max())
print("  kappa_z min/max [1/m]:", kappa_z.min(), kappa_z.max())

# --- Plots ---
fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(411)
ax1.plot(s_out, y)
ax1.set_ylabel("y(s) [m]")
ax1.set_title("Cosserat rod centerline: y(s)")

ax2 = fig.add_subplot(412)
ax2.plot(s_out, z)
ax2.set_ylabel("z(s) [m]")
ax2.set_title("Cosserat rod centerline: z(s)")

ax3 = fig.add_subplot(413)
ax3.plot(s_out, np.rad2deg(np.unwrap(np.arctan2(q[3, :], q[0, :]))))
ax3.set_ylabel("quaternion phase proxy [deg]")
ax3.set_title("Orientation proxy (optional)")

ax4 = fig.add_subplot(414)
ax4.plot(s_out, fy, label="f_y [N/m]")
ax4.plot(s_out, fz, label="f_z [N/m]")
ax4.plot(s_out, tau_x, label="tau_x [N]")
ax4.set_xlabel("s [m]")
ax4.set_ylabel("load density")
ax4.legend(loc="best")
ax4.set_title("Magnetic force and torque densities")

fig.tight_layout()

# --- 3D centerline ---
fig3d = plt.figure(figsize=(8, 6))
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=2)
ax.scatter([x[0]], [y[0]], [z[0]], label="base")
ax.scatter([x[-1]], [y[-1]], [z[-1]], label="tip")

# Equal-ish aspect scaling
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()
x_mid = 0.5*(x_min + x_max)
y_mid = 0.5*(y_min + y_max)
z_mid = 0.5*(z_min + z_max)
max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
half = 0.5*max_range
ax.set_xlim(x_mid-half, x_mid+half)
ax.set_ylim(y_mid-half, y_mid+half)
ax.set_zlim(z_mid-half, z_mid+half)
ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D Cosserat rod centerline")
ax.legend()
plt.tight_layout()
plt.show()