import numpy as np
from proper_research.models.beam_model import theta_angle_solved, dtheta_dB, dtheta_dL, dtheta_dphi, integral_x, integral_y
from beam_direction_magnetisation.magnetism.magnetic_methods import dipole_field_from_source
from proper_research.parameters import default_magnet_params
mag_params = default_magnet_params()

def R2(a):
    """2D rotation matrix."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s],
                     [s,  c]])

def forward_tip_xy_theta_2d(
    gamma_orbit,     # orbit angle of magnet about catheter tip
    beta_spin,       # magnet spin about its own z-axis 
    rho,             # magnet distance to tip
    L,               # catheter length
    *,
    mag, A_cs, E, I, 
    m_mag,           # external magnet dipole magnitude [A·m^2]
    m0_dir_2d=np.array([1.0, 0.0]),  
    r_min=1e-6
):
    """
    Pure 2D model:
      - Beam lies in x-y plane
      - Tip location is (x_tip, y_tip)
      - Magnet orbits around tip in x-y plane
      - Dipole moment is in x-y plane
      - Field is computed using your 3D dipole formula but with z=0

    Returns:
      tip_xy  = [x_tip, y_tip]
      theta_L = signed bending angle at tip [rad]
      dbg     = dictionary of intermediate values
    """


    r_tip0_2d = np.array([L, 0.0])

    r_src_2d = r_tip0_2d + R2(gamma_orbit) @ np.array([rho, 0.0])

    r_tip0_3d = np.array([r_tip0_2d[0], r_tip0_2d[1], 0.0])
    r_src_3d  = np.array([r_src_2d[0],  r_src_2d[1],  0.0])



    # 3) Dipole moment direction (beta relative to gamma)
    #    - gamma rotates the entire magnet frame in the world
    #    - beta spins the dipole inside that rotated magnet frame

    m_dir_2d = R2(gamma_orbit) @ (R2(beta_spin) @ m0_dir_2d)
    m_src_3d = m_mag * np.array([m_dir_2d[0], m_dir_2d[1], 0.0])


    B_vec_3d = dipole_field_from_source(
        r_tip0_3d[None, :], r_src_3d, m_src_3d, r_min=r_min
    )[0]  # (3,)

    B_mag = float(np.linalg.norm(B_vec_3d))
    B_mag = max(B_mag, 1e-12) 


    phi = float(np.arctan2(B_vec_3d[1], B_vec_3d[0]))
    phi_eps = np.deg2rad(0.5) 
    phi_eff = np.sqrt(phi*phi + phi_eps*phi_eps)
    phi_sign = phi / phi_eff                       

    phi_min = np.deg2rad(np.deg2rad(1.0))
    if phi_eff < phi_min:
        tip_xy = np.array([L, 0.0])
        theta_L = 0.0
        dbg = dict(
            r_tip0_2d=r_tip0_2d, r_src_2d=r_src_2d,
            B_vec=B_vec_3d, B_mag=B_mag,
            phi=phi, phi_eff=phi_eff,
            theta_L=theta_L,
            warning="phi too small straight approximation"
        )
        return tip_xy, theta_L, dbg


    theta_abs = theta_angle_solved(B_mag, phi_eff, mag, A_cs, L, E, I)
    theta_L = float(theta_abs * phi_sign)


    scale = np.sqrt((E * I) / (mag * B_mag * A_cs))

    x_tip = scale * integral_x(phi_eff, theta_abs, constant_use=False)
    y_tip = scale * integral_y(phi_eff, theta_abs, constant_use=False) * phi_sign

    tip_xy = np.array([float(x_tip), float(y_tip)])

    r = np.linalg.norm(tip_xy)
    if r > L:
        tip_xy = tip_xy * (L / r)
    dbg = dict(
        r_tip0_2d=r_tip0_2d,
        r_src_2d=r_src_2d,
        m_dir_2d=m_dir_2d,
        B_vec=B_vec_3d,
        B_mag=B_mag,
        phi=phi,
        phi_eff=phi_eff,
        theta_L=theta_L,
        scale=scale
    )
    return tip_xy, theta_L, dbg
mag = 128e3
r = 0.0015
A_cs = np.pi*r**2
E = 3.5e6
I = np.pi*r**4/4
L = 0.05
rho = 0.15

# for deg in [-60, -30, 0, 30, 60]:
#     tip_xy, theta_L, dbg = forward_tip_xy_theta_2d(
#         gamma_orbit=np.deg2rad(0),
#         beta_spin=np.deg2rad(deg),
#         rho=rho,
#         L=L,
#         mag=mag, A_cs=A_cs, E=E, I=I,
#         m_mag=mag_params.mag_epm
#     )
#     print(
#         f"gamma={deg:+4.0f} deg | tip=[{tip_xy[0]:.4f},{tip_xy[1]:.4f}] "
#         f"| theta={np.rad2deg(theta_L):+.2f} deg | B={dbg['B_mag']:.3e} T"
#     )

def numerical_jacobian_tip_2d(p, forward_fn, eps):

    p = np.asarray(p, dtype=float).ravel()
    eps = np.asarray(eps, dtype=float).ravel()

    x0 = forward_fn(p)  # (2,)
    J = np.zeros((2, p.size), dtype=float)

    for i in range(p.size):
        dp = np.zeros_like(p)
        dp[i] = eps[i]

        xp = forward_fn(p + dp)
        xm = forward_fn(p - dp)

        J[:, i] = (xp - xm) / (2.0 * eps[i])

    return J
def make_forward_fn(mag, A_cs, E, I, m_mag):
    def forward(p):
        gamma, beta, rho, L = p
        tip_xy, theta_L, dbg = forward_tip_xy_theta_2d(
            gamma_orbit=gamma,
            beta_spin=beta,
            rho=rho,
            L=L,
            mag=mag, A_cs=A_cs, E=E, I=I,
            m_mag=m_mag
        )
        return tip_xy
    return forward



def damped_least_squares_step(J, e, lam=1e-3):
    J = np.asarray(J, dtype=float)
    e = np.asarray(e, dtype=float).reshape(2,)

    A = J @ J.T + (lam**2) * np.eye(2) 
    dp = J.T @ np.linalg.solve(A, e)    
    return dp

def reach_target_with_jacobian(
    p_init, x_des,
    forward_fn,
    jac_fn,
    *,
    p_min=None,
    p_max=None,
    step_scale=0.5,
    lam=1e-3,
    max_iter=30,
    tol=1e-5,
    verbose=False
):

    p = np.asarray(p_init, dtype=float).copy()
    x_des = np.asarray(x_des, dtype=float).reshape(2,)

    if p_min is not None:
        p_min = np.asarray(p_min, dtype=float)
    if p_max is not None:
        p_max = np.asarray(p_max, dtype=float)

    hist = []

    for k in range(max_iter):
        x = forward_fn(p)
        e = x_des - x
        err = float(np.linalg.norm(e))

        J = jac_fn(p)

     
        dp = damped_least_squares_step(J, e, lam=lam)

  
        p_new = p + step_scale * dp

        if p_min is not None:
            p_new = np.maximum(p_new, p_min)
        if p_max is not None:
            p_new = np.minimum(p_new, p_max)

        hist.append(dict(k=k, p=p.copy(), x=x.copy(), e=e.copy(), err=err, dp=dp.copy(), J=J.copy()))

        if verbose:
            print(f"k={k:02d} | err={err:.4e} m | x=[{x[0]:+.4f},{x[1]:+.4f}] "
                  f"| e=[{e[0]:+.4f},{e[1]:+.4f}]")
            print(f"       dp=[{dp[0]:+.3e},{dp[1]:+.3e},{dp[2]:+.3e},{dp[3]:+.3e}]")

        p = p_new

        if err < tol:
            if verbose:
                print("Converged.")
            break

    return p, hist
def make_jac_fn(forward, eps):
    def jac(p):
        return numerical_jacobian_tip_2d(p, forward, eps)
    return jac
forward = make_forward_fn(mag, A_cs, E, I, mag_params.mag_epm)
eps = np.array([
    np.deg2rad(0.5),
    np.deg2rad(0.5),
    1e-3,
    5e-4
])
jac_fn = make_jac_fn(forward, eps)
p_min = np.array([
    np.deg2rad(-90),   # gamma
    np.deg2rad(-90),   # beta
    0.1,              # rho (m)
    0.03               # L (m)
])

p_max = np.array([
    np.deg2rad(90),
    np.deg2rad(90),
    0.25,
    0.08
])
p_start = np.array([
    np.deg2rad(60.0),   # gamma
    np.deg2rad(2.0),   # beta
    0.15,
    0.05
])

p_true = np.array([
    np.deg2rad(50.0),   # gamma
    np.deg2rad(5.0),  # beta
    0.12,               # rho
    0.06                # L
])
import matplotlib.pyplot as plt

param_names = ["gamma (deg)", "beta (deg)", "rho (m)", "L (m)"]
is_angle = [True, True, False, False]

def sweep_values(p0, i, p_min, p_max, n=200):
    return np.linspace(p_min[i], p_max[i], n)

def plot_jacobian_sweeps(p0, jac_fn, p_min, p_max, n=200):
    p0 = np.asarray(p0, float)

    for i in range(4):
        vals = sweep_values(p0, i, p_min, p_max, n=n)

        dxdpi = np.zeros(n)
        dydpi = np.zeros(n)
        sens  = np.zeros(n)

        for k, v in enumerate(vals):
            p = p0.copy()
            p[i] = v
            J = jac_fn(p)   # shape (2,4)

            dxdpi[k] = J[0, i]
            dydpi[k] = J[1, i]
            sens[k]  = np.sqrt(J[0, i]**2 + J[1, i]**2)

        x = np.rad2deg(vals) if is_angle[i] else vals

        plt.figure()
        plt.plot(x, dxdpi, label=r"$\partial x/\partial p_i$")
        plt.plot(x, dydpi, label=r"$\partial y/\partial p_i$")
        plt.plot(x, sens,  label=r"$\sqrt{(\partial x/\partial p_i)^2+(\partial y/\partial p_i)^2}$")

        plt.xlabel(param_names[i])
        plt.ylabel("Jacobian entry")
        plt.title(f"Jacobian sensitivity vs {param_names[i]}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # plot_jacobian_sweeps(
    #     p0=p_start,
    #     jac_fn=jac_fn,
    #     p_min=p_min,
    #     p_max=p_max,
    #     n=250
    # )
    betas = np.deg2rad(np.linspace(-90, 90, 200))
    ys = []

    p0 = p_start.copy()
    for b in betas:
        p = p0.copy()
        p[1] = b
        tip = forward(p)
        ys.append(tip[1])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.rad2deg(betas), ys)
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel("beta (deg)")
    plt.ylabel("y_tip (m)")
    plt.title("y_tip vs beta")
    plt.grid(True)
    gammas = np.deg2rad(np.linspace(-90, 90, 200))
    ys = []

    p0 = p_start.copy()
    for g in gammas:
        p = p0.copy()
        p[0] = g
        tip = forward(p)
        ys.append(tip[1])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.rad2deg(gammas), ys)
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel("gamma (deg)")
    plt.ylabel("y_tip (m)")
    plt.title("y_tip vs gamma")
    plt.grid(True)
    plt.show()
    lengths =np.linspace(0.05, 0.08, 10)
    ys = []

    p0 = p_start.copy()
    for l in lengths:
        p = p0.copy()
        p[3] = l
        tip = forward(p)
        ys.append(tip[1])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(lengths, ys)
    plt.axhline(0, color='k', linewidth=1)
    plt.xlabel("length (m)")
    plt.ylabel("y_tip (m)")
    plt.title("y_tip vs length")
    plt.grid(True)
    plt.show()