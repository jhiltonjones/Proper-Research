import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

MU0_OVER_4PI = 1e-7

def magnetic_moment(B_r, mu_0, r, p):
    return (B_r / mu_0) * (np.pi * r**2 * p)

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
theta_z_deg = 0.0
theta_y_deg = -70.0
r_mag = magnet_pose_about_tip(L, rho, theta_z_deg, theta_y_deg)

def dipole_field(r_pts, m_ext):
    R = r_pts - r_mag[None,:]
    R2 = np.sum(R**2, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, 1e-18)) 
    Rhat = R / Rnorm[:,None]
    mdot = Rhat @ m_ext
    return MU0_OVER_4PI * (1.0/(Rnorm**3))[:,None] * (3.0*mdot[:,None]*Rhat - m_ext[None,:])

alpha_deg = -50
alpha = np.deg2rad(alpha_deg)
m_local = np.array([mu_line*np.cos(alpha), mu_line*np.sin(alpha), 0.0])

def m_global_from_psi_vec(psi_vec):
    mx, my, mz = m_local
    c = np.cos(psi_vec); s = np.sin(psi_vec)
    return np.column_stack([mx*np.ones_like(psi_vec),
                            my*c - mz*s,
                            my*s + mz*c])
def force_and_torque_density(s, v, w, psi, m_ext, eps=5e-5):
    r_pts = np.column_stack((s, v, w))
    m_pts = m_global_from_psi_vec(psi)
    B0 = dipole_field(r_pts, m_ext)
    tau = np.cross(m_pts, B0)
    dr_y = np.array([0.0, eps, 0.0])
    gp_y = np.sum(m_pts * dipole_field(r_pts + dr_y, m_ext), axis=1)
    gm_y = np.sum(m_pts * dipole_field(r_pts - dr_y, m_ext), axis=1)
    fy = (gp_y - gm_y) / (2*eps)
    dr_z = np.array([0.0, 0.0, eps])
    gp_z = np.sum(m_pts * dipole_field(r_pts + dr_z, m_ext), axis=1)
    gm_z = np.sum(m_pts * dipole_field(r_pts - dr_z, m_ext), axis=1)
    fz = (gp_z - gm_z) / (2*eps)
    return fy, fz, tau, B0
def make_ode(m_ext_scale):
    m_ext = m_ext_scale * m_ext_full

    def ode(s_eval, Y, eps=5e-5):
        v, vp, w, wp, psi, psip, Vy, Vz, My, Mz = Y
        fy, fz, tau, _B0 = force_and_torque_density(s_eval, v, w, psi, m_ext, eps=eps)
        tau_x, tau_y, tau_z = tau[:,0], tau[:,1], tau[:,2]
        dY = np.zeros_like(Y)
        dY[0] = vp
        dY[1] = Mz / EI
        dY[2] = wp
        dY[3] = -My / EI
        dY[4] = psip
        dY[5] = tau_x / GJ
        dY[6] = -fy
        dY[7] = -fz
        dY[8] = Vz + tau_y
        dY[9] = Vy + tau_z
        return dY
    return ode

def bc(Ya, Yb):
    v0, vp0, w0, wp0, psi0, psip0, Vy0, Vz0, My0, Mz0 = Ya
    vL, vpL, wL, wpL, psiL, psipL, VyL, VzL, MyL, MzL = Yb
    return np.array([v0, vp0, w0, wp0, psi0,
                     VyL, VzL, MyL, MzL, psipL])


n0 = 120
s_mesh = np.linspace(0, L, n0)
Y_guess = np.zeros((10, n0))
sol = None
scales = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
for lam in scales:
    ode = make_ode(lam)
    if sol is None:
        sol = solve_bvp(ode, bc, s_mesh, Y_guess, max_nodes=12000, tol=3e-3)
    else:
        y_init = sol.sol(s_mesh)
        sol = solve_bvp(ode, bc, s_mesh, y_init, max_nodes=12000, tol=3e-3)
    print("lambda", lam, "success", sol.success, "nodes", sol.x.size, "msg", sol.message)
    if not sol.success:
        break

sol.success

s_out = np.linspace(0, L, 300)
Y = sol.sol(s_out)
v, vp, w, wp, psi, psip, Vy, Vz, My, Mz = Y

ode_full = make_ode(1.0)

m_ext = m_ext_full

fy, fz, tau, B = force_and_torque_density(s_out, v, w, psi, m_ext_full, eps=5e-5)
tau_x, tau_y, tau_z = tau[:,0], tau[:,1], tau[:,2]

print("Beam/magnet setup:")
print("  L =", L, "m")
print("  EI =", EI, "N·m^2")
print("  GJ =", GJ, "N·m^2")
print("  mu_line =", mu_line, "A·m")
print("  mag_epm =", mag_epm, "A·m^2")
print("  r_mag =", r_mag)

print("\nSolution summary:")
print("  tip deflection v(L) [m]:", v[-1])
print("  tip deflection w(L) [m]:", w[-1])
print("  max |v| [m]:", np.max(np.abs(v)))
print("  max |w| [m]:", np.max(np.abs(w)))
print("  psi range [deg]:", np.rad2deg(np.min(psi)), "to", np.rad2deg(np.max(psi)))

print("\nLoads summary:")
print("  tau_x min/max [N]:", tau_x.min(), tau_x.max())
print("  tau_y min/max [N]:", tau_y.min(), tau_y.max())
print("  tau_z min/max [N]:", tau_z.min(), tau_z.max())
print("  f_y min/max [N/m]:", fy.min(), fy.max())
print("  f_z min/max [N/m]:", fz.min(), fz.max())
print("max |psip|:", np.max(np.abs(psip)))
print("tau_x scale:", np.max(np.abs(tau_x)))
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(411)
ax1.plot(s_out, v)
ax1.set_ylabel("v(s) [m]")
ax1.set_title("Cosserat (Kirchhoff) rod: y-deflection v(s)")

ax2 = fig.add_subplot(412)
ax2.plot(s_out, w)
ax2.set_ylabel("w(s) [m]")
ax2.set_title("z-deflection w(s)")

ax3 = fig.add_subplot(413)
ax3.plot(s_out, np.rad2deg(psi))
ax3.set_ylabel("psi(s) [deg]")
ax3.set_title("Twist angle psi(s)")

ax4 = fig.add_subplot(414)
ax4.plot(s_out, fy, label="f_y")
ax4.plot(s_out, fz, label="f_z")
ax4.plot(s_out, tau_x, label="tau_x")
ax4.set_xlabel("s [m]")
ax4.set_ylabel("load")
ax4.legend(loc="best")
ax4.set_title("Magnetic force densities and torque density")

fig.tight_layout()
# plt.show()

from mpl_toolkits.mplot3d import Axes3D  

x = s_out
y = v
z = w

def unit(vec):
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec

t0 = unit(np.array([1.0, vp[0], wp[0]]))
tL = unit(np.array([1.0, vp[-1], wp[-1]]))

bend_angle_deg = np.rad2deg(np.arccos(np.clip(np.dot(t0, tL), -1.0, 1.0)))
theta_y_deg = np.rad2deg(np.arctan2(vp[-1], 1.0))
theta_z_deg = np.rad2deg(np.arctan2(wp[-1], 1.0))

print("\nGeometry:")
print("  Tip bending angle (t0->tL) [deg]:", bend_angle_deg)
print("  Tip slope angle in x–y plane [deg]:", theta_y_deg)
print("  Tip slope angle in x–z plane [deg]:", theta_z_deg)

fig3d = plt.figure(figsize=(8, 6))
ax = fig3d.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=2)
ax.scatter([x[0]], [y[0]], [z[0]], label="base")
ax.scatter([x[-1]], [y[-1]], [z[-1]], label="tip")
# ... your ax.plot(x, y, z) etc ...

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
# scale = L * 0.2
# ax.quiver(x[0], y[0], z[0], t0[0], t0[1], t0[2], length=scale, normalize=True)
# ax.quiver(x[-1], y[-1], z[-1], tL[0], tL[1], tL[2], length=scale, normalize=True)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_title("3D beam centerline and tangents")
ax.legend()
plt.tight_layout()
plt.show()