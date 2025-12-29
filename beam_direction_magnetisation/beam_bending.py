import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from proper_research.parameters import default_beam_params, default_magnet_params
MU0_OVER_4PI = 1e-7  # mu0/(4*pi) [T·m/A]

# ----------------------------
# Rotations
# ----------------------------
def R_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def R_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def R_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# ----------------------------
# External dipole field
# ----------------------------
def dipole_field(r, r0, m_vec, mu0_over_4pi=MU0_OVER_4PI):
    """
    Point dipole field:
      B(r) = (mu0/4pi)/|R|^3 * ( 3 (m·Rhat) Rhat - m )
    Inputs:
      r:   (N,3) eval points
      r0:  (3,)  dipole center
      m_vec:(3,) dipole moment in world
    """
    R = r - r0[None, :]
    R2 = np.sum(R**2, axis=1)
    Rnorm = np.sqrt(np.maximum(R2, 1e-18))
    Rhat = R / Rnorm[:, None]
    mdot = Rhat @ m_vec
    B = mu0_over_4pi * (1.0 / (Rnorm**3))[:, None] * (3.0 * mdot[:, None] * Rhat - m_vec[None, :])
    return B

# ----------------------------
# Magnet pose: orbit around tip (Z then Y), then dipole spin about magnet's own z
# ----------------------------
def magnet_pose_about_tip(L, rho, theta_z_deg, theta_y_deg, gamma_deg, m0, dipole_points_outward=True):
    """
    Position:
      r0 = r_tip + [rho,0,0]
      r_mag = r_tip + R_y(theta_y) @ R_z(theta_z) @ [rho,0,0]
    Orientation:
      dipole aligned with +X when gamma=0, then rotate about magnet's own z:
      m_dir_world = R_z(gamma) @ [1,0,0]
    """
    r_tip = np.array([L, 0.0, 0.0])

    thz = np.deg2rad(theta_z_deg)
    thy = np.deg2rad(theta_y_deg)
    gamma = np.deg2rad(gamma_deg)

    v0 = np.array([rho, 0.0, 0.0])
    v = R_y(thy) @ (R_z(thz) @ v0)
    r_mag = r_tip + v

    m_dir_world = R_z(gamma) @ np.array([1.0, 0.0, 0.0])
    if not dipole_points_outward:
        m_dir_world = -m_dir_world
    m_ext = m0 * m_dir_world
    return r_mag, m_ext

# ----------------------------
# Beam magnetization per unit length (world), rotates with psi about x
# ----------------------------
def m_line_world(mu_line, alpha, psi):
    """
    Beam dipole per unit length in world frame for a nominal beam axis along +X.
    Tilt alpha away from x; transverse component rotates about x by psi(s).

    m = mu_line * [cos(alpha), sin(alpha) cos(psi), sin(alpha) sin(psi)]
    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    return mu_line * np.column_stack((
        ca * np.ones_like(psi),
        sa * np.cos(psi),
        sa * np.sin(psi)
    ))

# ----------------------------
# Force density: f = ∇(m·B) via finite differences (robust for any B model)
# ----------------------------
def force_density_grad_mdotB_fd(r_pts, mline_pts, r_mag, m_ext, eps=5e-5):
    """
    r_pts: (N,3) points on deformed beam centerline
    mline_pts: (N,3) dipole-per-length at those points
    Returns:
      f: (N,3) where f = ∇(m·B)
    """
    B0 = dipole_field(r_pts, r_mag, m_ext)
    g0 = np.sum(mline_pts * B0, axis=1)

    f = np.zeros_like(r_pts)
    # for k in range(3):
    #     dr = np.zeros(3)
    #     dr[k] = eps

    #     Bp = dipole_field(r_pts + dr, r_mag, m_ext)
    #     gp = np.sum(mline_pts * Bp, axis=1)

    #     Bm = dipole_field(r_pts - dr, r_mag, m_ext)
    #     gm = np.sum(mline_pts * Bm, axis=1)

    #     f[:, k] = (gp - gm) / (2 * eps)

    return f

# ----------------------------
# 3D bending (v,w) + torsion (psi) BVP solver
# ----------------------------
def solve_bending_torsion_3d(
    s, L, r_mag, m_ext,
    *,
    EI_y=1e-3, EI_z=1e-3,
    GJ=2e-3,
    mu_line=0.06,
    alpha_deg=50.0,
    bc_type="clamped_free_roll_roll",
    eps_grad=5e-5,
    max_nodes=20000
):
    """
    Small-deflection Euler–Bernoulli 3D bending + Saint-Venant torsion.

    Centerline: r(s) = [ s, v(s), w(s) ]
      v: deflection in y
      w: deflection in z

    States (10):
      0 v, 1 v', 2 Mz, 3 Vy, 4 w, 5 w', 6 My, 7 Vz, 8 psi, 9 psi'
    with:
      v''  = Mz / EI_z
      w''  = My / EI_y
      Vy'  = -q_y
      Vz'  = -q_z
      psi''= tau_x / GJ

    Loads:
      tau_x = (m_line x B)·ex
      q = f = ∇(m·B)  (using finite differences)
    """
    alpha = np.deg2rad(alpha_deg)

    def ode(s_eval, Y):
        v, vp, Mz, Vy = Y[0], Y[1], Y[2], Y[3]
        w, wp, My, Vz = Y[4], Y[5], Y[6], Y[7]
        psi, psip     = Y[8], Y[9]

        r_pts = np.column_stack((s_eval, v, w))  # (N,3)

        mline = m_line_world(mu_line, alpha, psi)          # (N,3)
        B     = dipole_field(r_pts, r_mag, m_ext)          # (N,3)

        # Torque density about x
        tau = np.cross(mline, B)                           # (N,3)
        tau_x = tau[:, 0]

        # Force density
        f = force_density_grad_mdotB_fd(r_pts, mline, r_mag, m_ext, eps=eps_grad)
        qy = f[:, 1]
        qz = f[:, 2]

        dY = np.zeros_like(Y)
        # bending in y (v)
        dY[0] = vp
        dY[1] = Mz / EI_z
        dY[2] = Vy
        dY[3] = -qy
        # bending in z (w)
        dY[4] = wp
        dY[5] = My / EI_y
        dY[6] = Vz
        dY[7] = -qz
        # torsion
        dY[8] = psip
        dY[9] = tau_x / GJ

        return dY

    # Boundary conditions you requested:
    # - Base: no translation or bending at base (clamped in bending): v(0)=v'(0)=w(0)=w'(0)=0
    # - Base + tip: free to roll in torsion: psi'(0)=psi'(L)=0
    # - Tip: free in bending: Mz(L)=Vy(L)=My(L)=Vz(L)=0
    if bc_type == "clamped_free_roll_roll":
        def bc(Ya, Yb):
            return np.array([
                Ya[0],  # v(0)=0
                Ya[1],  # v'(0)=0
                Ya[4],  # w(0)=0
                Ya[5],  # w'(0)=0
                Ya[9],  # psi'(0)=0 (roll-free base)

                Yb[2],  # Mz(L)=0
                Yb[3],  # Vy(L)=0
                Yb[6],  # My(L)=0
                Yb[7],  # Vz(L)=0
                Yb[9],  # psi'(L)=0 (roll-free tip)
            ])
    else:
        raise ValueError("Unsupported bc_type. Use 'clamped_free_roll_roll'.")

    # Initial guess (small deflections, small twist)
    N = s.size
    Y_guess = np.zeros((10, N))
    # Give psi a gentle initial shape to help convergence (optional)
    Y_guess[8, :] = 0.0
    Y_guess[9, :] = 0.0

    sol = solve_bvp(ode, bc, s, Y_guess, max_nodes=max_nodes)
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")

    Y = sol.sol(s)
    v, vp, Mz, Vy = Y[0], Y[1], Y[2], Y[3]
    w, wp, My, Vz = Y[4], Y[5], Y[6], Y[7]
    psi, psip     = Y[8], Y[9]

    # Diagnostics: compute B, force, torque along the solved (deformed) centerline
    r_pts = np.column_stack((s, v, w))
    mline = m_line_world(mu_line, alpha, psi)
    B     = dipole_field(r_pts, r_mag, m_ext)
    f     = force_density_grad_mdotB_fd(r_pts, mline, r_mag, m_ext, eps=eps_grad)
    tau   = np.cross(mline, B)
    tau_x = tau[:, 0]

    return {
        "s": s,
        "v": v, "w": w, "psi": psi,
        "Mz": Mz, "Vy": Vy, "My": My, "Vz": Vz, "psip": psip,
        "r_pts": r_pts,
        "B": B,
        "f": f,
        "tau_x": tau_x,
        "sol": sol
    }
def params_from_defaults(beam_params, magnet_params, *, nu=0.49, alpha_deg=50.0):
    # Beam geometry
    r = beam_params.r
    E = beam_params.E
    A_cs = beam_params.A_cs
    I = beam_params.I
    L = 0.08

    # Bending stiffness (small-deflection EB)
    EI_y = E * I
    EI_z = E * I

    # Torsion stiffness (circular rod)
    G = E / (2.0 * (1.0 + nu))
    J = 0.5 * np.pi * r**4
    GJ = G * J

    # Beam magnetic dipole per unit length
    mu_line = beam_params.mag * A_cs  # (A/m)*(m^2) = A·m

    # External magnet dipole magnitude
    m0 = magnet_params.mag_epm *-1
    # Magnet baseline standoff in front of the tip
    rho = 0.05

    return dict(
        L=L,
        EI_y=EI_y, EI_z=EI_z,
        GJ=GJ,
        mu_line=mu_line,
        alpha_deg=alpha_deg,
        m0=m0,
        rho=rho
    )
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_beam_3d(res, *, show_magnet=True, show_field_samples=False, n_field_samples=25,
                 show_projections=True, show_magnet_link=True):
    """
    Visualize the deformed beam centerline in 3D and (optionally) the magnet position,
    plus optional 2D projection plots (XY, XZ, YZ) to clearly show magnet location.
    """
    r = res["r_pts"]
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    # --- Figure layout ---
    if show_projections:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")  # big 3D axis on the left
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 1])
        # We'll make YZ as an inset on the 3D plot to avoid too many panels, OR you can add a 3rd axis.
        # If you prefer 3 panels, see comment below.
        ax_yz = None
    else:
        fig = plt.figure(figsize=(9, 7))
        ax3d = fig.add_subplot(111, projection="3d")
        ax_xy = ax_xz = ax_yz = None

    # --- 3D Beam centerline ---
    ax3d.plot(x, y, z, linewidth=2, label="Beam centerline")
    ax3d.scatter([x[0]], [y[0]], [z[0]], s=40, label="Base")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], s=40, label="Tip")

    # --- Magnet marker and link to beam (optional) ---
    rm = None
    if show_magnet and ("r_mag" in res):
        rm = np.asarray(res["r_mag"]).reshape(3,)
        ax3d.scatter([rm[0]], [rm[1]], [rm[2]], s=100, marker="X", label="Magnet center")

        if show_magnet_link:
            # Find nearest beam point to magnet and draw a dashed connector
            d2 = (x - rm[0])**2 + (y - rm[1])**2 + (z - rm[2])**2
            i = int(np.argmin(d2))
            ax3d.plot([rm[0], x[i]], [rm[1], y[i]], [rm[2], z[i]],
                      linestyle="--", linewidth=1, label="Magnet→nearest beam pt")

    # --- Field quiver samples (optional) ---
    if show_field_samples and ("B" in res):
        B = res["B"]
        N = len(x)
        idx = np.linspace(0, N - 1, min(n_field_samples, N), dtype=int)

        Bdir = B[idx]
        norms = np.linalg.norm(Bdir, axis=1)
        norms = np.maximum(norms, 1e-18)
        Bdir = Bdir / norms[:, None]

        arrow_len = 0.08 * (x.max() - x.min() + 1e-12)
        ax3d.quiver(x[idx], y[idx], z[idx],
                    Bdir[:, 0], Bdir[:, 1], Bdir[:, 2],
                    length=arrow_len, normalize=False, linewidth=1)

    # --- Labels and title ---
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Beam Shape")

    # --- Equal-ish aspect ratio (Matplotlib workaround) ---
    xmid, ymid, zmid = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
    span = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min(), 1e-9)
    half = 0.5 * span
    ax3d.set_xlim(xmid - half, xmid + half)
    ax3d.set_ylim(ymid - half, ymid + half)
    ax3d.set_zlim(zmid - half, zmid + half)

    # --- 2D projection plots (optional) ---
    if show_projections:
        # XY
        ax_xy.plot(x, y, linewidth=2)
        ax_xy.scatter([x[0]], [y[0]], s=30)
        ax_xy.scatter([x[-1]], [y[-1]], s=30)
        if rm is not None:
            ax_xy.scatter([rm[0]], [rm[1]], s=60, marker="X")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title("XY projection")
        ax_xy.axis("equal")

        # XZ
        ax_xz.plot(x, z, linewidth=2)
        ax_xz.scatter([x[0]], [z[0]], s=30)
        ax_xz.scatter([x[-1]], [z[-1]], s=30)
        if rm is not None:
            ax_xz.scatter([rm[0]], [rm[2]], s=60, marker="X")
        ax_xz.set_xlabel("X [m]")
        ax_xz.set_ylabel("Z [m]")
        ax_xz.set_title("XZ projection")
        ax_xz.axis("equal")

        # If you want a dedicated YZ subplot too, replace the gridspec above with a 2x2 where 3D is one cell,
        # and add:
        # ax_yz = fig.add_subplot(gs[1, 1])  # and move xz to gs[0,1], yz to gs[1,1]
        # then:
        # ax_yz.plot(y, z); ax_yz.scatter(rm[1], rm[2], marker="X"); ...

    ax3d.legend(loc="best")
    plt.tight_layout()
    plt.show()

# def plot_beam_3d(res, *, show_magnet=True, show_field_samples=False, n_field_samples=25):
#     """
#     Visualize the deformed beam centerline in 3D and (optionally) the magnet position.

#     Inputs
#     ------
#     res : dict
#         Output dict from solve_bending_torsion_3d(...), expected to contain:
#         - "r_pts": (N,3) deformed centerline points [x, y, z]
#         - "s": (N,)
#         - optionally "B": (N,3) field along centerline (for quiver)
#     show_magnet : bool
#         If True, plots r_mag if present in res (or if you add it).
#     show_field_samples : bool
#         If True, plots a sparse quiver of B vectors along the beam.
#     n_field_samples : int
#         How many quiver samples along the beam (sparse).

#     Notes
#     -----
#     - This uses Matplotlib's 3D axes. It is intended as a lightweight visualization.
#     - For better aspect ratio control, it enforces an equal-ish bounding box.
#     """
#     r = res["r_pts"]
#     x, y, z = r[:, 0], r[:, 1], r[:, 2]

#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection="3d")

#     # Beam centerline
#     ax.plot(x, y, z, linewidth=2, label="Beam centerline")
#     ax.scatter([x[0]], [y[0]], [z[0]], s=40, label="Base")
#     ax.scatter([x[-1]], [y[-1]], [z[-1]], s=40, label="Tip")

#     # Magnet marker (if available)
#     if show_magnet and ("r_mag" in res):
#         rm = res["r_mag"]
#         ax.scatter([rm[0]], [rm[1]], [rm[2]], s=80, marker="X", label="Magnet center")

#     # Field quiver samples (optional)
#     if show_field_samples and ("B" in res):
#         B = res["B"]
#         N = len(x)
#         idx = np.linspace(0, N - 1, min(n_field_samples, N), dtype=int)

#         # Normalize arrows for display (direction only)
#         Bdir = B[idx]
#         norms = np.linalg.norm(Bdir, axis=1)
#         norms = np.maximum(norms, 1e-18)
#         Bdir = Bdir / norms[:, None]

#         # Scale arrows based on beam length for readability
#         arrow_len = 0.08 * (x.max() - x.min() + 1e-12)
#         ax.quiver(x[idx], y[idx], z[idx],
#                   Bdir[:, 0], Bdir[:, 1], Bdir[:, 2],
#                   length=arrow_len, normalize=False, linewidth=1, label="B direction")

#     # Labels
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_zlabel("Z [m]")
#     ax.set_title("3D Beam Shape")

#     # Equal-ish aspect ratio (Matplotlib workaround)
#     xmid, ymid, zmid = (x.max()+x.min())/2, (y.max()+y.min())/2, (z.max()+z.min())/2
#     span = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min(), 1e-9)
#     half = 0.5 * span
#     ax.set_xlim(xmid - half, xmid + half)
#     ax.set_ylim(ymid - half, ymid + half)
#     ax.set_zlim(zmid - half, zmid + half)

#     ax.legend(loc="best")
#     plt.tight_layout()
#     plt.show()

# ----------------------------
# Example main script
# ----------------------------
def main():
    bp = default_beam_params()
    mp = default_magnet_params()

    p = params_from_defaults(bp, mp, nu=0.49, alpha_deg=0.1)

    L = p["L"]
    n = 300
    s = np.linspace(0.0, L, n)

    # Magnet orbit + spin angles (your new pose model)
    theta_z_deg = 50.0
    theta_y_deg = 0.0
    gamma_deg   = 0.0

    r_mag, m_ext = magnet_pose_about_tip(
        L=L, rho=p["rho"],
        theta_z_deg=theta_z_deg,
        theta_y_deg=theta_y_deg,
        gamma_deg=gamma_deg,
        m0=p["m0"],
        dipole_points_outward=True
    )

    res = solve_bending_torsion_3d(
        s, L, r_mag, m_ext,
        EI_y=p["EI_y"], EI_z=p["EI_z"], GJ=p["GJ"],
        mu_line=p["mu_line"], alpha_deg=p["alpha_deg"],
        bc_type="clamped_free_roll_roll",
        eps_grad=5e-5
    )


    # Unpack results
    v, w, psi = res["v"], res["w"], res["psi"]
    B = res["B"]
    f = res["f"]
    tau_x = res["tau_x"]

    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    Bperp = np.sqrt(By**2 + Bz**2)
    beta = np.arctan2(Bz, By)
    print("tau_x min/max:", res["tau_x"].min(), res["tau_x"].max())
    print("By min/max:", res["B"][:,1].min(), res["B"][:,1].max())
    print("Bz min/max:", res["B"][:,2].min(), res["B"][:,2].max())
    print("psi min/max (deg):", np.rad2deg(res["psi"].min()), np.rad2deg(res["psi"].max()))

    res["r_mag"] = r_mag
    plot_beam_3d(res, show_magnet=True, show_field_samples=True, n_field_samples=20)

    # Plot
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(511)
    ax1.plot(s, v)
    ax1.set_ylabel("v(s) [m]")
    ax1.set_title("3D bending + torsion: y-deflection v(s)")

    ax2 = fig.add_subplot(512)
    ax2.plot(s, w)
    ax2.set_ylabel("w(s) [m]")
    ax2.set_title("z-deflection w(s)")

    ax3 = fig.add_subplot(513)
    ax3.plot(s, np.rad2deg(beta), label="beta(s) field azimuth")
    ax3.plot(s, np.rad2deg(psi),  label="psi(s) twist")
    ax3.set_ylabel("Angle [deg]")
    ax3.legend(loc="best")
    ax3.set_title("Twist vs transverse-field azimuth (evaluated on deformed centerline)")

    ax4 = fig.add_subplot(514)
    ax4.plot(s, Bperp)
    ax4.set_ylabel("B_perp [T]")
    ax4.set_title("Transverse field magnitude along deformed centerline")

    ax5 = fig.add_subplot(515)
    ax5.plot(s, tau_x, label="tau_x(s)")
    ax5.plot(s, f[:, 1], label="q_y(s)=f_y")
    ax5.plot(s, f[:, 2], label="q_z(s)=f_z")
    ax5.set_xlabel("s [m]")
    ax5.set_ylabel("Load")
    ax5.legend(loc="best")
    ax5.set_title("Magnetic torque density and force densities")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
