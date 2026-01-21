import numpy as np
from proper_research.models.beam_model import theta_angle_solved, integral_x, integral_y
from beam_direction_magnetisation.magnetism.magnetic_methods import dipole_field_from_source
from proper_research.parameters import default_magnet_params, default_beam_params
mag_params = default_magnet_params()
beam_params = default_beam_params()
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

    phi_min = np.deg2rad(1.0)
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

def solve_params_for_target_xy(
    x_des,                    # desired tip position (2,) in SAME units as forward() returns
    p_init,                   # initial guess [gamma, beta, rho, L]
    forward_fn,
    jac_fn,
    p_min,
    p_max,
    *,
    lam=1e-2,                 # damping (increase if unstable)
    step_scale=0.5,           # reduce if it overshoots
    max_iter=50,
    tol=1e-6,
    verbose=True
):
    p = np.asarray(p_init, dtype=float).copy()
    x_des = np.asarray(x_des, dtype=float).reshape(2,)

    p_min = np.asarray(p_min, dtype=float)
    p_max = np.asarray(p_max, dtype=float)

    for k in range(max_iter):
        x = np.asarray(forward_fn(p), float).reshape(2,)
        e = x_des - x
        err = float(np.linalg.norm(e))

        # --- Jacobian ---
        J = np.asarray(jac_fn(p), float)  # (2,4)

        # Basic Jacobian health
        if not np.all(np.isfinite(J)):
            raise ValueError(f"Jacobian non-finite at iter {k}. p={p}, J={J}")

        # Condition-ish check: singular values of J
        # (If smallest is ~0, the inverse problem is ill-conditioned locally)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        smin, smax = float(np.min(S)), float(np.max(S))
        cond = (smax / smin) if smin > 0 else np.inf

        # --- Damped least squares step: dp = J^T (J J^T + lam^2 I)^-1 e ---
        A = J @ J.T + (lam**2) * np.eye(2)
        dp = J.T @ np.linalg.solve(A, e)

        # Update + clamp
        p_new = p + step_scale * dp
        p_new = np.minimum(np.maximum(p_new, p_min), p_max)

        if verbose:
            print(
                f"[{k:02d}] err={err:.3e} | x=[{x[0]:+.4e},{x[1]:+.4e}] "
                f"| e=[{e[0]:+.4e},{e[1]:+.4e}] "
                f"| dp=[{dp[0]:+.3e},{dp[1]:+.3e},{dp[2]:+.3e},{dp[3]:+.3e}] "
                f"| S={S} cond~{cond:.2e}"
            )

        p = p_new
        if err < tol:
            break

    return p



def numerical_jacobian_tip_2d(
    p,
    forward_fn,
    eps,
    *,
    max_step_halvings=6,
    fail_policy="raise",   # "raise" or "nan" or "zero"
    debug=False,
    name_map=None,         # optional: ["gamma","beta","rho","L"]
):
    """
    Robust finite-difference Jacobian for tip_xy = forward_fn(p).

    - Attempts central difference.
    - If forward(p+dp) or forward(p-dp) fails, falls back to one-sided.
    - If still failing, halves dp and retries.
    - If still failing after retries, uses fail_policy.

    Returns:
      J: (2, n) Jacobian
      dbg: dict with per-column diagnostics (ONLY if debug=True, else not returned)
    """
    p = np.asarray(p, dtype=float).ravel()
    eps = np.asarray(eps, dtype=float).ravel()
    n = p.size

    if name_map is None:
        name_map = [f"p[{i}]" for i in range(n)]

    # Evaluate base point (must succeed)
    x0 = np.asarray(forward_fn(p), dtype=float).reshape(2,)
    J = np.zeros((2, n), dtype=float)

    col_dbg = []

    for i in range(n):
        dp0 = np.zeros_like(p)
        dp0[i] = eps[i]

        mode_used = None
        last_exc = None
        used_step = float(eps[i])

        dp = dp0.copy()

        for h in range(max_step_halvings + 1):
            try:
                xp = np.asarray(forward_fn(p + dp), dtype=float).reshape(2,)
                xp_ok = True
            except Exception as e:
                xp_ok = False
                xp = None
                last_exc = e

            try:
                xm = np.asarray(forward_fn(p - dp), dtype=float).reshape(2,)
                xm_ok = True
            except Exception as e:
                xm_ok = False
                xm = None
                last_exc = e

            if xp_ok and xm_ok:
                # central difference
                J[:, i] = (xp - xm) / (2.0 * dp[i])
                mode_used = "central"
                break

            if xp_ok and (not xm_ok):
                # forward one-sided
                J[:, i] = (xp - x0) / (dp[i])
                mode_used = "forward"
                break

            if xm_ok and (not xp_ok):
                # backward one-sided
                J[:, i] = (x0 - xm) / (dp[i])
                mode_used = "backward"
                break

            # neither side OK -> shrink step and retry
            dp *= 0.5
            used_step = float(dp[i])

        if mode_used is None:
            msg = f"Jacobian column failed for {name_map[i]} after step shrink. p={p.tolist()}, eps0={eps[i]}"
            if debug:
                print("[JAC-FAIL]", msg)
                if last_exc is not None:
                    print("          last exception:", repr(last_exc))

            if fail_policy == "raise":
                raise ValueError(msg) from last_exc
            elif fail_policy == "nan":
                J[:, i] = np.nan
            elif fail_policy == "zero":
                J[:, i] = 0.0
            else:
                raise ValueError(f"Unknown fail_policy='{fail_policy}'")

        if debug:
            col_dbg.append({
                "i": i,
                "name": name_map[i],
                "mode": mode_used,
                "used_step": used_step,
                "col": J[:, i].copy(),
            })

    if debug:
        return J, {"x0": x0, "cols": col_dbg}
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
def make_jac_fn(forward, eps, *, debug=False, fail_policy="raise"):
    name_map = ["gamma", "beta", "rho", "L"]
    def jac(p):
        if debug:
            J, dbg = numerical_jacobian_tip_2d(
                p, forward, eps,
                debug=True,
                fail_policy=fail_policy,
                name_map=name_map
            )
            # You can print dbg here if you want
            return J
        return numerical_jacobian_tip_2d(
            p, forward, eps,
            debug=False,
            fail_policy=fail_policy,
            name_map=name_map
        )
    return jac
import numpy as np

def tangent_test(forward_fn, jac_fn, p0, p_min, p_max, *,
                 n_tests=50,
                 dp_scales=(1.0, 0.5, 0.25, 0.125, 0.0625),
                 base_step=None,
                 seed=0,
                 verbose=True):
    """
    Validate local linearization: forward(p+dp) ~ forward(p) + J(p)dp

    base_step: vector of typical parameter perturbations for dp generation.
               If None, a reasonable default is used.
    """
    rng = np.random.default_rng(seed)

    p0 = np.asarray(p0, float).copy()
    p_min = np.asarray(p_min, float)
    p_max = np.asarray(p_max, float)

    if base_step is None:
        base_step = np.array([
            np.deg2rad(1.0),  # gamma
            np.deg2rad(1.0),  # beta
            2e-3,             # rho (m)
            1e-3              # L (m)
        ], dtype=float)

    x0 = np.asarray(forward_fn(p0), float).reshape(2,)
    J0 = np.asarray(jac_fn(p0), float)
    if not np.all(np.isfinite(J0)):
        raise ValueError(f"Jacobian not finite at p0. J0=\n{J0}")

    # SVD/conditioning information
    U, S, Vt = np.linalg.svd(J0, full_matrices=False)
    cond = (S.max() / S.min()) if S.min() > 0 else np.inf

    if verbose:
        print("=== Tangent test @ p0 ===")
        print("p0 =", p0)
        print("x0 =", x0)
        print("J0 =\n", J0)
        print("singular values:", S, " cond~", cond)

    results = []

    for scale in dp_scales:
        errs = []
        for _ in range(n_tests):
            # random direction in parameter space
            z = rng.normal(size=4)
            z /= (np.linalg.norm(z) + 1e-12)

            dp = scale * base_step * z

            # clamp p1 into bounds (important: MPC will enforce bounds too)
            p1 = np.minimum(np.maximum(p0 + dp, p_min), p_max)

            # effective dp after clamping
            dp_eff = p1 - p0

            x_true = np.asarray(forward_fn(p1), float).reshape(2,)
            x_lin  = x0 + J0 @ dp_eff

            num = np.linalg.norm(x_true - x_lin)
            den = np.linalg.norm(x_true - x0) + 1e-12
            rel = num / den

            errs.append(rel)

        errs = np.asarray(errs)
        results.append((scale, float(np.median(errs)), float(np.percentile(errs, 90)), float(np.max(errs))))

        if verbose:
            print(f"scale={scale:>7} | rel_err: median={np.median(errs):.3e}  p90={np.percentile(errs,90):.3e}  max={np.max(errs):.3e}")

    return results



if __name__ == "__main__":
    eps = np.array([
        np.deg2rad(0.5),
        np.deg2rad(0.5),
        1e-3,
        5e-4
    ])

    p_min = np.array([
        np.deg2rad(-90),
        np.deg2rad(-90),
        0.10,
        0.03
    ])
    p_max = np.array([
        np.deg2rad(90),
        np.deg2rad(90),
        0.25,
        0.08
    ])

    # Initial guess (must be in feasible region!)
    p_init = np.array([
        np.deg2rad(5.0),
        np.deg2rad(5.0),
        0.13,
        0.05
    ])

    forward_fn = make_forward_fn(
        beam_params.mag, beam_params.A_cs, beam_params.E, beam_params.I, mag_params.mag_epm
    )
    jac_fn = make_jac_fn(forward_fn, eps, debug=False, fail_policy="raise")

    # ---- Choose a target tip position (same units as forward_fn output) ----
    # Your forward model outputs tip_xy in meters? (check scale in your integrals)
    # Put something plausible for your L range, e.g. x around ~L and y small-ish.
    x_target = np.array([0.038, 0.031])  # example: 45mm forward, 10mm lateral (if in meters)

    p_sol = solve_params_for_target_xy(
        x_des=x_target,
        p_init=p_init,
        forward_fn=forward_fn,
        jac_fn=jac_fn,
        p_min=p_min,
        p_max=p_max,
        lam=1e-2,
        step_scale=0.5,
        max_iter=80,
        tol=5e-4,
        verbose=True
    )

    print("\n=== Solution ===")
    print("gamma(deg) =", np.rad2deg(p_sol[0]))
    print("beta(deg)  =", np.rad2deg(p_sol[1]))
    print("rho(m)     =", p_sol[2])
    print("L(m)       =", p_sol[3])

    x_check = forward_fn(p_sol)
    print("x_check    =", x_check, " target=", x_target, " err=", np.linalg.norm(x_target - x_check))
    eps = np.array([np.deg2rad(0.5), np.deg2rad(0.5), 1e-3, 5e-4])
    forward_fn = make_forward_fn(beam_params.mag, beam_params.A_cs, beam_params.E, beam_params.I, mag_params.mag_epm)
    jac_fn = make_jac_fn(forward_fn, eps, debug=False, fail_policy="raise")

    p0 = np.array([np.deg2rad(5.0), np.deg2rad(5.0), 0.13, 0.05])
    p_min = np.array([np.deg2rad(-90), np.deg2rad(-90), 0.10, 0.03])
    p_max = np.array([np.deg2rad(90),  np.deg2rad(90),  0.25, 0.08])

    tangent_test(forward_fn, jac_fn, p0, p_min, p_max, n_tests=100)
