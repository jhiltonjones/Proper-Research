import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

MU0_OVER_4PI = 1e-7

# ------------------------
# Your parameters
# ------------------------
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
mu_line = mag * A_cs  # A·m  (interpreted here as magnetic moment per unit length)

mu_0 = 4e-7*np.pi
B_r = 1.25
r_epm = 0.03
p_epm = 0.09
mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)  # A·m^2

# External magnet: position and dipole moment
# rho = 0.14
# r_mag = np.array([L + rho, 0.0, 0.0])  # example: "in front" on +x; change for overhead
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
rho = 0.18
theta_z_deg =40.0
theta_y_deg = 0.0
r_mag = magnet_pose_about_tip(L, rho, theta_z_deg, theta_y_deg)

m_ext_full = mag_epm * np.array([1.0, 0.0, 0.0])     # external dipole moment vector

# Beam magnetization direction in BODY frame (constant)
alpha_deg = -50
alpha = np.deg2rad(alpha_deg)
m_mag_body = np.array([mu_line*np.cos(alpha), 0.0, mu_line*np.sin(alpha)])  # x–z plane

# Stiffness matrix for Kirchhoff rod in BODY frame: [twist, bend2, bend3]
K = np.diag([GJ, EI, EI])
K_inv = np.diag([1.0/GJ, 1.0/EI, 1.0/EI])
u0 = np.zeros(3)  # intrinsic curvature/twist

# ------------------------
# Magnetics: dipole field
# ------------------------
def dipole_field(r_pts, m_ext, r_src):
    """
    r_pts: (N,3), m_ext: (3,), r_src: (3,)
    returns B: (N,3)
    """
    R = r_pts - r_src[None, :]
    R2 = np.sum(R**2, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, 1e-18))
    Rhat = R / Rnorm[:, None]
    mdot = Rhat @ m_ext
    return MU0_OVER_4PI * (1.0/(Rnorm**3))[:, None] * (3.0*mdot[:, None]*Rhat - m_ext[None, :])

# ------------------------
# Quaternion utilities (q = [w,x,y,z], body->world)
# ------------------------
def quat_normalize(q):
    n = np.linalg.norm(q, axis=0)
    n = np.maximum(n, 1e-18)
    return q / n

def quat_mul(q, p):
    """
    Hamilton product for arrays.
    q, p: (4,N) each
    returns (4,N)
    """
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.vstack([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ])

def quat_to_R(q):
    """
    q: (4,N) normalized, body->world
    returns R: (N,3,3)
    """
    qw, qx, qy, qz = q
    # Rotation matrix components
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)

    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)

    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)

    R = np.stack([
        np.stack([R11, R12, R13], axis=-1),
        np.stack([R21, R22, R23], axis=-1),
        np.stack([R31, R32, R33], axis=-1)
    ], axis=-2)  # (N,3,3)
    return R
def quat_to_R_single(q):
    q = q / max(np.linalg.norm(q), 1e-18)
    qw, qx, qy, qz = q
    R11 = 1 - 2*(qy*qy + qz*qz)
    R12 = 2*(qx*qy - qz*qw)
    R13 = 2*(qx*qz + qy*qw)
    R21 = 2*(qx*qy + qz*qw)
    R22 = 1 - 2*(qx*qx + qz*qz)
    R23 = 2*(qy*qz - qx*qw)
    R31 = 2*(qx*qz - qy*qw)
    R32 = 2*(qy*qz + qx*qw)
    R33 = 1 - 2*(qx*qx + qy*qy)
    return np.array([[R11,R12,R13],[R21,R22,R23],[R31,R32,R33]])


# External force/couple density from magnetics
# ------------------------
def force_and_couple_density(r_pts, Rmats, m_ext, r_src, eps=5e-5, include_force=True):
    """
    r_pts: (N,3)
    Rmats: (N,3,3) body->world
    returns:
      f: (N,3)
      l: (N,3)   (couple density)
      B: (N,3)
    """
    # magnetic moment per unit length in world
    m_mag_world = (Rmats @ m_mag_body[None, :, None]).squeeze(-1)  # (N,3)

    B0 = dipole_field(r_pts, m_ext, r_src)                   # (N,3)
    l = np.cross(m_mag_world, B0)                            # (N,3)

    if not include_force:
        f = np.zeros_like(r_pts)
        return f, l, B0

    # Approximate f = grad(m·B) holding m fixed (orientation fixed) over spatial perturbations.
    U0 = np.sum(m_mag_world * B0, axis=1)

    f = np.zeros_like(r_pts)
    for k in range(3):
        dr = np.zeros(3); dr[k] = eps
        Bp = dipole_field(r_pts + dr[None,:], m_ext, r_src)
        Bm = dipole_field(r_pts - dr[None,:], m_ext, r_src)
        Up = np.sum(m_mag_world * Bp, axis=1)
        Um = np.sum(m_mag_world * Bm, axis=1)
        f[:, k] = (Up - Um) / (2*eps)

    return f, l, B0

# ------------------------
# Cosserat/Kirchhoff rod ODE
# State Y = [r(3), q(4), n(3), m(3)] => 13 states
# ------------------------
def make_ode(lam, include_force=True):
    m_ext = lam * m_ext_full
    r_src = r_mag.copy()

    def ode(s, Y):
        # Y shape: (13,N)
        r_xyz = Y[0:3, :]    # (3,N)
        q = Y[3:7, :]        # (4,N)
        n = Y[7:10, :]       # (3,N)
        m = Y[10:13, :]      # (3,N)

        # normalize quaternion to reduce numerical drift
        q = quat_normalize(q)
        Rmats = quat_to_R(q)           # (N,3,3)

        # Kirchhoff constraint: r' = R e1 (first director)
        d1 = Rmats[:, :, 0].T          # (3,N), tangent in world

        # internal moment in BODY frame
        # m: (3,N) world; want m_body: (3,N) body
        m_world = m.T                                 # (N,3)
        m_body_world = (np.transpose(Rmats, (0,2,1)) @ m_world[:, :, None]).squeeze(-1)  # (N,3)
        m_body = m_body_world.T                        # (3,N)

        # strain u in BODY frame from constitutive law: m_body = K (u-u0)
        u = (K_inv @ m_body) + u0[:, None]  # (3,N)

        # quaternion derivative driven by body angular rate u
        # dq/ds = 0.5 * q ⊗ [0, u]
        omega_quat = np.vstack([np.zeros_like(s), u])  # (4,N)
        q_s = 0.5 * quat_mul(q, omega_quat)            # (4,N)

        # external loads
        r_pts = r_xyz.T  # (N,3)
        f, l, _B = force_and_couple_density(r_pts, Rmats, m_ext, r_src,
                                            eps=5e-5, include_force=include_force)
        f = f.T   # (3,N)
        l = l.T   # (3,N)

        # Cosserat equilibrium
        # n' = -f
        n_s = -f

        # m' = - (r' x n + l)
        m_s = -np.cross(d1.T, n.T).T - l

        dY = np.zeros_like(Y)
        dY[0:3, :] = d1
        dY[3:7, :] = q_s
        dY[7:10, :] = n_s
        dY[10:13, :] = m_s
        return dY

    return ode

# ------------------------
# Boundary conditions
# Clamped base: r(0)=0, q(0)=identity
# Free tip: n(L)=0, m(L)=0
# ------------------------
def bc(Ya, Yb):
    r0 = Ya[0:3]
    q0 = Ya[3:7]
    nL = Yb[7:10]
    mL = Yb[10:13]

    q_base = np.array([1.0, 0.0, 0.0, 0.0])
    return np.hstack([
        r0,                 # 3
        q0 - q_base,        # 4 (fix full quaternion at base)
        nL,                 # 3
        mL                  # 3
    ])  # total 13
def bc_clamped(Ya, Yb):
    r0 = Ya[0:3]
    q0 = Ya[3:7]
    nL = Yb[7:10]
    mL = Yb[10:13]
    q_base = np.array([1.0, 0.0, 0.0, 0.0])
    return np.hstack([r0, q0 - q_base, nL, mL])
def bc_clamped_free_torsion(Ya, Yb):
    r0 = Ya[0:3]
    q0 = Ya[3:7]        # [w,x,y,z]
    nL = Yb[7:10]
    mL = Yb[10:13]

    # No bending rotation at base: enforce qy=0, qz=0 (torsion about x still allowed)
    qy0 = q0[2]
    qz0 = q0[3]

    # Enforce quaternion normalization at the base (good conditioning)
    qnorm0 = np.dot(q0, q0) - 1.0

    # Torsion-free hinge: no transmitted moment about the local tangent d1
    R0 = quat_to_R_single(q0)
    d1_0 = R0[:, 0]
    m0 = Ya[10:13]
    m_parallel0 = np.dot(m0, d1_0)

    return np.hstack([
        r0,                    # 3
        [qy0, qz0],            # 2
        [qnorm0],              # 1
        [m_parallel0],         # 1
        nL,                    # 3
        mL                     # 3
    ])  # total 13


n0 = 120
s_mesh = np.linspace(0, L, n0)

Y_guess = np.zeros((13, n0))

Y_guess[0, :] = s_mesh
Y_guess[1, :] = 0.0
Y_guess[2, :] = 0.0

Y_guess[3, :] = 1.0
Y_guess[4, :] = 0.0
Y_guess[5, :] = 0.0
Y_guess[6, :] = 0.0

sol = None
scales = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]

for lam in scales:
    ode = make_ode(lam, include_force=True)  # start without force if you want torque-only
    if sol is None:
        sol = solve_bvp(ode, bc_clamped_free_torsion, s_mesh, Y_guess, max_nodes=15000, tol=3e-3)
    else:
        sol = solve_bvp(ode, bc_clamped_free_torsion, s_mesh, sol.sol(s_mesh), max_nodes=15000, tol=3e-3)

    print("lambda", lam, "success", sol.success, "nodes", sol.x.size, "msg", sol.message)
    if not sol.success:
        break

# ------------------------
# Post-processing: centerline, tip angles
# ------------------------
s_out = np.linspace(0, L, 300)
Y = sol.sol(s_out)
r_xyz = Y[0:3, :].T
q = quat_normalize(Y[3:7, :])
Rmats = quat_to_R(q)
t = Rmats[:, :, 0]      # (N,3)
tL = t[-1]              # tip tangent

theta_xy_tip = np.rad2deg(np.arctan2(tL[1], tL[0]))
theta_xz_tip = np.rad2deg(np.arctan2(tL[2], tL[0]))

print("Tip slope angle in x–y [deg]:", theta_xy_tip)
print("Tip slope angle in x–z [deg]:", theta_xz_tip)
# Choose whether you want forces included in reporting:
include_force_report = True

m_ext = m_ext_full  # or lam*m_ext_full for a particular lam
r_src = r_mag

r_pts = r_xyz  # (N,3)
f, l, B = force_and_couple_density(r_pts, Rmats, m_ext, r_src,
                                  eps=5e-5, include_force=include_force_report)

# f and l are (N,3): columns are x,y,z
fx, fy, fz = f[:,0], f[:,1], f[:,2]
lx, ly, lz = l[:,0], l[:,1], l[:,2]

print("\nDistributed loads (per unit length):")
print("  Force density fx min/max:", fx.min(), fx.max())
print("  Force density fy min/max:", fy.min(), fy.max())
print("  Force density fz min/max:", fz.min(), fz.max())
print("  Couple density lx min/max:", lx.min(), lx.max())
print("  Couple density ly min/max:", ly.min(), ly.max())
print("  Couple density lz min/max:", lz.min(), lz.max())
# internal moment m(s) in world is in Y[10:13,:]
m_world = Y[10:13, :].T  # (N,3)

# world->body: m_body = R^T m_world
m_body = (np.transpose(Rmats,(0,2,1)) @ m_world[:,:,None]).squeeze(-1)  # (N,3)

u = (m_body @ K_inv.T) + u0[None,:]   # careful: K_inv is diagonal
# safer since diagonal:
u = np.column_stack([m_body[:,0]/GJ, m_body[:,1]/EI, m_body[:,2]/EI])

u1 = u[:,0]  # twist rate about local tangent
print("\nTwist/torsion:")
print("  u1 min/max [rad/m]:", u1.min(), u1.max())
print("  approx tip twist angle [deg] (integral u1 ds):",
      np.rad2deg(np.trapezoid(u1, s_out)))
# Optional: plot them
plt.figure()
plt.plot(s_out, fx, label="fx")
plt.plot(s_out, fy, label="fy")
plt.plot(s_out, fz, label="fz")
plt.xlabel("s [m]"); plt.ylabel("Force density"); plt.legend(); plt.grid(True)

plt.figure()
plt.plot(s_out, lx, label="lx")
plt.plot(s_out, ly, label="ly")
plt.plot(s_out, lz, label="lz")
plt.xlabel("s [m]"); plt.ylabel("Couple density"); plt.legend(); plt.grid(True)
plt.show()
tangent = Rmats[:, :, 0]  # d1 in world, (N,3)
t0 = tangent[0]
tL = tangent[-1]

bend_angle_deg = np.rad2deg(np.arccos(np.clip(np.dot(t0, tL), -1.0, 1.0)))

print("\nCosserat rod results:")
print("  tip position:", r_xyz[-1])
print("  tip bending angle (t0->tL) [deg]:", bend_angle_deg)

# Plot 3D centerline
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(r_xyz[:,0], r_xyz[:,1], r_xyz[:,2], linewidth=2)
ax.scatter([r_xyz[0,0]],[r_xyz[0,1]],[r_xyz[0,2]], label="base")
ax.scatter([r_xyz[-1,0]],[r_xyz[-1,1]],[r_xyz[-1,2]], label="tip")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.legend()
plt.show()
