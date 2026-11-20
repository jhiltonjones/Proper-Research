"""Numerical study of what separates the controller rungs, and by how much.

This is not a model of your hardware. It is a *minimal plant with the same
structural features* — seven inputs, three outputs, a saturating nonlinearity so
the Jacobian genuinely varies along a path, a state-dependent contact term the
controller's model omits, and the same box/rate/state constraints — built so the
three theoretical predictions can be checked numerically before they are checked
on hardware:

  P1  The A1 (resolved-rate) to A2 (MPC) gap is governed by *constraint
      activity*, and constraint activity is governed by *headroom*.  Under a
      time-optimal plan the headroom is zero by construction, so the gap is
      maximal; dilating the trajectory by gamma restores headroom and the gap
      must close.  Prediction: gap(gamma) is monotone decreasing and reaches
      zero at the gamma where the activity rate alpha reaches zero.

  P2  The A2 (LTI) to A3 (LTV) gap is governed by how far the Jacobian moves
      along the path, measured by the scheduling index S.  There is a hard
      threshold: when the symmetric part of J_true J_model^+ loses positive
      definiteness the LTI loop is no longer contractive and must degrade
      sharply rather than gradually.

  P3  The A3 (offline schedule) to A4 (online relinearisation) gap is governed
      by how far the plant leaves the reference, measured by the linearisation
      index L, which is itself driven by the model mismatch.  Prediction:
      monotone increasing in mismatch, and *zero* when mismatch is zero.

Run:  python3 rung_separation_study.py
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(7)
NX = 7          # configuration dimension
NY = 3          # controlled output dimension
DT = 0.02       # sample period, s


# ===========================================================================
# 1. A QP solver (ADMM, the OSQP algorithm), because none is installed here
# ===========================================================================
def solve_qp(P, q, A, lo, hi, rho=1.0, sigma=1e-6, alpha=1.6, iters=400, tol=1e-9):
    """min 0.5 x'Px + q'x  s.t.  lo <= Ax <= hi, by ADMM with a cached KKT LU.

    This is the OSQP algorithm without the adaptive rho or the polish step.
    Returned solutions therefore satisfy the constraints to solver tolerance
    rather than exactly — the same caveat that applies on real hardware, and the
    reason the violation metrics below use an explicit tolerance.
    """
    n, m = P.shape[0], A.shape[0]
    kkt = np.block([[P + sigma * np.eye(n), A.T],
                    [A, -np.eye(m) / rho]])
    lu = np.linalg.inv(kkt)  # small; explicit inverse is fine and fast here
    x = np.zeros(n)
    z = np.zeros(m)
    y = np.zeros(m)
    for _ in range(iters):
        rhs = np.concatenate([sigma * x - q, z - y / rho])
        sol = lu @ rhs
        xt, nu = sol[:n], sol[n:]
        zt = z + (nu - y) / rho
        x_new = alpha * xt + (1 - alpha) * x
        z_new = np.clip(alpha * zt + (1 - alpha) * z + y / rho, lo, hi)
        y = y + rho * (alpha * zt + (1 - alpha) * z - z_new)
        if np.max(np.abs(x_new - x)) < tol and np.max(np.abs(z_new - z)) < tol:
            x, z = x_new, z_new
            break
        x, z = x_new, z_new
    return x


def _validate_qp():
    """Check the ADMM solver against scipy on random dense problems."""
    from scipy.optimize import minimize
    worst = 0.0
    for _ in range(6):
        n, m = 12, 30
        M = rng.normal(size=(n, n))
        P = M @ M.T + 0.5 * np.eye(n)
        q = rng.normal(size=n)
        A = rng.normal(size=(m, n))
        lo = -np.abs(rng.normal(size=m)) - 0.5
        hi = np.abs(rng.normal(size=m)) + 0.5
        mine = solve_qp(P, q, A, lo, hi, iters=3000)
        ref = minimize(
            lambda x: 0.5 * x @ P @ x + q @ x,
            np.zeros(n), jac=lambda x: P @ x + q, method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda x: hi - A @ x,
                          "jac": lambda x: -A},
                         {"type": "ineq", "fun": lambda x: A @ x - lo,
                          "jac": lambda x: A}],
            options={"maxiter": 400, "ftol": 1e-12},
        ).x
        f = lambda x: 0.5 * x @ P @ x + q @ x
        worst = max(worst, abs(f(mine) - f(ref)) / max(abs(f(ref)), 1e-9))
    return worst


# ===========================================================================
# 2. The plant: a constant-curvature two-bend arc with a lumen wall
# ===========================================================================
K_MAX = 30.0        # curvature saturation, 1/m
A0 = 0.9            # drive scale (sets how hard the tanh saturates)
WALL_R = 0.032      # lumen radius, m
WALL_C = 0.16       # contact stiffening strength


def _drives(chi):
    a = chi[0] + 0.4 * chi[2] + 0.2 * chi[4]
    b = chi[1] + 0.4 * chi[3] + 0.2 * chi[5]
    return a, b


def _arc(kappa, s):
    """Planar constant-curvature arc: lateral offset and axial advance."""
    ks = kappa * s
    small = np.abs(ks) < 1e-6
    lat = np.where(small, 0.5 * kappa * s * s, (1.0 - np.cos(ks)) / np.where(small, 1.0, kappa))
    ax = np.where(small, s, np.sin(ks) / np.where(small, 1.0, kappa))
    return lat, ax


def tip(chi, *, contact=True, curvature_gain=1.0):
    """Forward output map: configuration -> tip position (m)."""
    chi = np.asarray(chi, dtype=float).reshape(NX)
    s = max(chi[6], 1e-4)
    a, b = _drives(chi)
    ka = curvature_gain * K_MAX * np.tanh(a / A0)
    kb = curvature_gain * K_MAX * np.tanh(b / A0)
    if contact:
        # A lumen wall: once the lateral excursion exceeds WALL_R the wall
        # resists further bending. State dependent, absent from the model.
        lat_a, _ = _arc(ka, s)
        lat_b, _ = _arc(kb, s)
        r = np.hypot(lat_a, lat_b)
        if r > WALL_R:
            damp = 1.0 / (1.0 + WALL_C * (r - WALL_R) / WALL_R)
            ka, kb = ka * damp, kb * damp
    lat_a, ax_a = _arc(ka, s)
    lat_b, ax_b = _arc(kb, s)
    return np.array([lat_a, lat_b, 0.5 * (ax_a + ax_b)])


def jacobian(chi, *, contact=True, curvature_gain=1.0, h=1e-6):
    """Central-difference output Jacobian, 3x7."""
    chi = np.asarray(chi, dtype=float).reshape(NX)
    J = np.zeros((NY, NX))
    for i in range(NX):
        d = np.zeros(NX)
        d[i] = h
        J[:, i] = (tip(chi + d, contact=contact, curvature_gain=curvature_gain)
                   - tip(chi - d, contact=contact, curvature_gain=curvature_gain)) / (2 * h)
    return J


# ===========================================================================
# 3. Limits, reference path, and the time-scaling dial
# ===========================================================================
U_MAX = np.array([1.2, 1.2, 1.0, 1.0, 0.8, 0.8, 0.030])      # rate limits
A_MAX = np.array([9.0, 9.0, 8.0, 8.0, 6.0, 6.0, 0.35])       # accel limits
# The first two bounds are set deliberately close to the reference path's own
# extremes. A bound the trajectory never approaches cannot distinguish a
# controller that predicts constraints from one that clips against them, so an
# experiment with generous bounds measures nothing about preview.
# The reference path's own extremes are +1.433 on axis 0 and -1.600 on axis 1.
# The corridor is set 0.04 rad outside them: feasible for the plan, but with so
# little margin that any correction pushing outward will breach it unless the
# controller anticipates the bound. A bound the trajectory never approaches
# cannot distinguish a controller that predicts constraints from one that clips
# against them, so an experiment with generous bounds measures nothing here.
X_MIN = np.array([-2.5, -1.640, -2.5, -2.5, -2.5, -2.5, 0.010])
X_MAX = np.array([1.473, 2.5, 2.5, 2.5, 2.5, 2.5, 0.120])


def geometric_path(n_nodes=161, excursion=1.0):
    """A smooth configuration path. `excursion` scales how far it travels,
    which is the dial that sets the Jacobian-variation index S."""
    t = np.linspace(0.0, 1.0, n_nodes)
    chi = np.zeros((n_nodes, NX))
    chi[:, 0] = excursion * (0.95 * np.sin(2.6 * np.pi * t) + 0.5 * t)
    chi[:, 1] = excursion * (0.80 * np.cos(1.9 * np.pi * t) - 0.80)
    chi[:, 2] = excursion * 0.45 * np.sin(1.3 * np.pi * t)
    chi[:, 3] = excursion * 0.35 * np.cos(1.1 * np.pi * t)
    chi[:, 4] = excursion * 0.25 * np.sin(0.9 * np.pi * t)
    chi[:, 5] = excursion * 0.20 * np.sin(1.7 * np.pi * t)
    chi[:, 6] = 0.030 + 0.055 * t
    return chi


def time_parameterise(chi_nodes, gamma=1.0, dt=DT):
    """Quintic time scaling dilated to feasibility, then dilated again by gamma.

    gamma = 1 is the fastest this profile can go inside the limits, i.e. at
    least one limit is active somewhere. gamma > 1 buys headroom everywhere.
    """
    n = chi_nodes.shape[0]
    sigma = np.linspace(0.0, 1.0, n)
    # quintic profile u(tau) = 10t^3 - 15t^4 + 6t^5
    T = 4.0
    for _ in range(60):
        t = np.arange(0.0, T + dt, dt)
        tau = np.clip(t / T, 0.0, 1.0)
        u = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        du = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        ddu = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
        state = np.stack([np.interp(u, sigma, chi_nodes[:, i]) for i in range(NX)], axis=1)
        dstate = np.stack([np.gradient(state[:, i], t) for i in range(NX)], axis=1)
        ddstate = np.stack([np.gradient(dstate[:, i], t) for i in range(NX)], axis=1)
        rv = np.max(np.abs(dstate) / U_MAX)
        ra = np.max(np.abs(ddstate) / A_MAX)
        need = max(rv, np.sqrt(max(ra, 0.0)))
        if need <= 1.0 + 1e-3:
            break
        T *= need * 1.001
    T *= gamma
    t = np.arange(0.0, T + dt, dt)
    tau = np.clip(t / T, 0.0, 1.0)
    u = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    state = np.stack([np.interp(u, sigma, chi_nodes[:, i]) for i in range(NX)], axis=1)
    dstate = np.stack([np.gradient(state[:, i], t) for i in range(NX)], axis=1)
    ddstate = np.stack([np.gradient(dstate[:, i], t) for i in range(NX)], axis=1)
    return {"t": t, "state": state, "input": dstate, "accel": ddstate, "T": T}


# ===========================================================================
# 4. Controllers
# ===========================================================================
class ResolvedRate:
    """A1: damped least squares, nullspace-regulated, clipped."""
    name = "A1 resolved-rate"

    def __init__(self, ref, model_jac, kp=0.6, lam=2.5e-3, kn=1.0):
        self.ref, self.J = ref, model_jac
        self.kp, self.lam, self.kn = kp, lam, kn

    def solve(self, chi, p_meas, k, u_prev):
        i = min(k + 1, len(self.ref["t"]) - 1)
        J = self.J(chi)
        gram = J @ J.T + self.lam**2 * np.eye(NY)
        Jp = J.T @ np.linalg.solve(gram, np.eye(NY))
        e = self.ref["p_nom"][i] - p_meas
        v_task = Jp @ (self.kp * e / DT)
        Nproj = np.eye(NX) - Jp @ J
        v_null = Nproj @ (self.kn * (self.ref["state"][i] - chi) / DT
                          + self.ref["input"][i])
        v_unc = v_task + v_null
        return _clip(v_unc, chi, u_prev), v_unc


def _clip(v, chi, u_prev):
    v = np.clip(v, -U_MAX, U_MAX)
    v = np.clip(v, u_prev - A_MAX * DT, u_prev + A_MAX * DT)
    v = np.clip(v, (X_MIN - chi) / DT, (X_MAX - chi) / DT)
    return np.clip(v, -U_MAX, U_MAX)


class MPC:
    """A2/A3/A4: the same condensed QP; only `mode` changes where J comes from."""

    # Weights are specified as *normalised* weights divided by the square of a
    # characteristic scale, so that a "unit" of each term means the same thing.
    # Without this the position term (metres^2, ~1e-6) is swamped by the input
    # term (rad^2/s^2, ~1) and the controller simply refuses to move.
    P_SCALE = 0.5e-3      # m    — position error scale
    X_SCALE = 0.05        # rad  — configuration error scale

    def __init__(self, ref, model_jac, mode, N=6, wp=1.0, wx=0.02, wr=0.05):
        self.ref, self.J, self.mode, self.N = ref, model_jac, mode, N
        self.qp = wp / self.P_SCALE**2
        self.qx = wx / self.X_SCALE**2
        self.Rdiag = wr / U_MAX**2
        self.J0 = model_jac(ref["state"][0])
        self.name = {"lti": "A2 MPC-LTI", "ltv": "A3 MPC-LTV",
                     "sqp": "A4 MPC-SQP"}[mode]
        self.relin = 0
        # lower-triangular block integrator
        self.L = np.tril(np.ones((N, N)))

    def _jacs(self, chi, k):
        n = len(self.ref["t"])
        if self.mode == "lti":
            return [self.J0] * self.N
        if self.mode == "ltv":
            return [self.ref["Jsched"][min(k + j, n - 1)] for j in range(self.N)]
        self.relin += 1
        return [self.J(chi)] * self.N

    def solve(self, chi, p_meas, k, u_prev):
        N, n = self.N, len(self.ref["t"])
        Js = self._jacs(chi, k)
        idx = [min(k + 1 + j, n - 1) for j in range(N)]
        # ---------------------------------------------------------------
        # Anchoring. A Taylor expansion has two halves that must match: the
        # point the derivative was taken at, and the constant term. The offline
        # rungs hold J at the REFERENCE, so they must anchor at p_nom and carry
        # an explicit disturbance estimate. The online rung holds J at the
        # MEASURED state, so it must anchor at the measurement — the velocity
        # form — and then the disturbance is absorbed automatically and no
        # estimator is needed. Mixing the two (J at the measurement, constant at
        # the reference) is not a first-order model of anything, and in this
        # simulation it diverges.
        # ---------------------------------------------------------------
        G = np.zeros((NY * N, NX * N))
        c = np.zeros(NY * N)
        for j in range(N):
            for i in range(j + 1):
                G[NY * j:NY * (j + 1), NX * i:NX * (i + 1)] = DT * Js[j]
        if self.mode == "sqp":
            # p_j = p_meas + J(chi_meas) (chi_j - chi_meas)
            for j in range(N):
                c[NY * j:NY * (j + 1)] = p_meas - self.ref["p_nom"][idx[j]]
        else:
            # p_j = p_nom_j + J_j (chi_j - chi*_j) + d,   d = p_meas - model(chi)
            d = p_meas - tip(chi, contact=False, curvature_gain=self.model_gain)
            for j in range(N):
                c[NY * j:NY * (j + 1)] = (
                    Js[j] @ (chi - self.ref["state"][idx[j]]) + d
                )
        # state-tracking and input-tracking references
        Sx = DT * np.kron(self.L, np.eye(NX))
        xfree = np.tile(chi, N) - np.concatenate([self.ref["state"][i] for i in idx])
        uref = np.concatenate([self.ref["input"][i] for i in idx])

        Qp = self.qp * np.eye(NY * N)
        Qp[-NY:, -NY:] *= 20.0                      # terminal weight
        Rbar = np.diag(np.tile(self.Rdiag, N))
        P = 2 * (G.T @ Qp @ G + self.qx * Sx.T @ Sx + Rbar)
        q = 2 * (G.T @ Qp @ c + self.qx * Sx.T @ xfree - Rbar @ uref)
        P = 0.5 * (P + P.T)

        # constraints: input box, rate, state box
        I = np.eye(NX * N)
        D = np.zeros((NX * N, NX * N))
        for j in range(N):
            D[NX * j:NX * (j + 1), NX * j:NX * (j + 1)] = np.eye(NX)
            if j > 0:
                D[NX * j:NX * (j + 1), NX * (j - 1):NX * j] = -np.eye(NX)
        A = np.vstack([I, D, Sx])
        big = 1e6
        lo = np.concatenate([
            np.tile(-U_MAX, N),
            np.concatenate([u_prev - A_MAX * DT] + [-A_MAX * DT] * (N - 1)),
            np.tile(X_MIN, N) - np.tile(chi, N) - big * 0.0,
        ])
        hi = np.concatenate([
            np.tile(U_MAX, N),
            np.concatenate([u_prev + A_MAX * DT] + [A_MAX * DT] * (N - 1)),
            np.tile(X_MAX, N) - np.tile(chi, N),
        ])
        V = solve_qp(P, q, A, lo, hi, rho=8.0, iters=250)
        u = V[:NX]
        return _clip(u, chi, u_prev), u


# ===========================================================================
# 5. Closed-loop run and the diagnostic indices
# ===========================================================================
def run(controller, ref, *, plant_contact=True, plant_gain=1.0, seed=0):
    n = len(ref["t"])
    noise = np.random.default_rng(seed)
    chi = ref["state"][0].copy()
    u_prev = np.zeros(NX)
    err, act, viol, Lidx, mu = [], 0, 0, [], []
    xviol, xviol_mag, cosang = 0, 0.0, []
    for k in range(n - 1):
        p_meas = tip(chi, contact=plant_contact, curvature_gain=plant_gain)
        p_meas = p_meas + noise.normal(scale=1.0e-5, size=NY)
        err.append(np.linalg.norm(p_meas - ref["p_nom"][min(k, n - 1)]))
        u, u_unc = controller.solve(chi, p_meas, k, u_prev)
        # --- diagnostics -------------------------------------------------
        if np.any(np.abs(u_unc) > U_MAX * (1 + 1e-9)) or np.any(
                np.abs(u_unc - u_prev) > A_MAX * DT * (1 + 1e-9)):
            act += 1
        if np.any(np.abs(u) > U_MAX * (1 + 1e-6)) or np.any(
                np.abs(u - u_prev) > A_MAX * DT * (1 + 1e-6)):
            viol += 1
        over = np.maximum(np.maximum(X_MIN - chi, chi - X_MAX), 0.0)
        if np.any(over > 1e-9):
            xviol += 1
            xviol_mag = max(xviol_mag, float(np.max(over)))
        # how far the applied command's tip motion is rotated away from the
        # motion the unconstrained law asked for — the cost of clipping
        Jd = jacobian(chi, contact=plant_contact, curvature_gain=plant_gain)
        a_vec, b_vec = Jd @ u, Jd @ u_unc
        na, nb = np.linalg.norm(a_vec), np.linalg.norm(b_vec)
        if na > 1e-12 and nb > 1e-12:
            cosang.append(float(a_vec @ b_vec / (na * nb)))
        if k % 4:
            chi = chi + DT * u
            u_prev = u
            continue
        Jt = Jd
        at_measurement = isinstance(controller, ResolvedRate) or (
            isinstance(controller, MPC) and controller.mode == "sqp")
        Jm = controller.J(chi) if at_measurement else controller.J(
            ref["state"][min(k, n - 1)])
        Lidx.append(np.linalg.norm(Jt - Jm, 2) / max(np.linalg.norm(Jt, 2), 1e-12))
        Jp = Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-9 * np.eye(NY), np.eye(NY))
        Mx = Jt @ Jp
        mu.append(np.min(np.linalg.eigvalsh(0.5 * (Mx + Mx.T))))
        # -----------------------------------------------------------------
        chi = chi + DT * u          # the state box is NOT enforced here
        u_prev = u
    err = np.array(err)
    return {
        "x_violation": xviol / (n - 1),
        "x_violation_mm_rad": xviol_mag,
        "cos_direction": float(np.mean(cosang)) if cosang else 1.0,
        "rms_mm": 1e3 * float(np.sqrt(np.mean(err**2))),
        "max_mm": 1e3 * float(np.max(err)),
        "final_mm": 1e3 * float(err[-1]),
        "activity": act / (n - 1),
        "violation": viol / (n - 1),
        "L": float(np.mean(Lidx)),
        "mu_min": float(np.min(mu)),
    }


def build_reference(gamma, excursion=1.0, model_gain=1.0):
    traj = time_parameterise(geometric_path(excursion=excursion), gamma=gamma)
    traj["p_nom"] = np.array([tip(c, contact=False, curvature_gain=model_gain)
                              for c in traj["state"]])
    traj["Jsched"] = [jacobian(c, contact=False, curvature_gain=model_gain)
                      for c in traj["state"]]
    # headroom: how much of each limit the plan itself consumes
    traj["headroom"] = float(1.0 - np.max(np.abs(traj["input"]) / U_MAX))
    J0 = traj["Jsched"][0]
    traj["S"] = float(max(np.linalg.norm(J - J0, 2) for J in traj["Jsched"])
                      / np.linalg.norm(J0, 2))
    return traj


def make_controllers(ref, model_gain=1.0, jac_gain=None, kp=0.6, lam=2.5e-3):
    """`model_gain` is the model used for the disturbance estimate; `jac_gain`
    is the model used for the *Jacobian*. Keeping them separate matters: it lets
    an experiment degrade the sensitivity alone, leaving the reference reachable,
    so that the A3-to-A4 comparison measures linearisation point rather than
    reference validity. Perturbing both at once conflates the two and produces a
    result that looks like A4 failing when in fact the target is unreachable."""
    jg = model_gain if jac_gain is None else jac_gain
    mj = lambda c: jacobian(c, contact=False, curvature_gain=jg)
    cs = [ResolvedRate(ref, mj, kp=kp, lam=lam)]
    for mode in ("lti", "ltv", "sqp"):
        m = MPC(ref, mj, mode)
        m.model_gain = model_gain
        cs.append(m)
    return cs


# ===========================================================================
# 6. The three experiments
# ===========================================================================
def experiment_headroom():
    print("\n" + "=" * 78)
    print("E1  HEADROOM SWEEP  —  does the A1->A2 gap close as the plan slows?")
    print("=" * 78)
    print(f"{'gamma':>6}{'headroom':>10}{'alpha(A1)':>11}"
          f"{'A1 rms':>9}{'A2 rms':>9}{'gap %':>8}{'A1 viol':>9}{'A2 viol':>9}")
    rows = []
    for g in (1.0, 1.15, 1.35, 1.6, 2.0, 3.0):
        ref = build_reference(g)
        cs = make_controllers(ref)
        r1 = run(cs[0], ref)
        r2 = run(cs[1], ref)
        gap = 100 * (r1["rms_mm"] - r2["rms_mm"]) / max(r2["rms_mm"], 1e-9)
        rows.append((g, ref["headroom"], r1, r2, gap))
        print(f"{g:>6.2f}{ref['headroom']:>10.3f}{r1['activity']:>11.3f}"
              f"{r1['rms_mm']:>9.3f}{r2['rms_mm']:>9.3f}{gap:>8.1f}"
              f"{r1['violation']:>9.3f}{r2['violation']:>9.3f}")
    return rows


def experiment_scheduling():
    print("\n" + "=" * 78)
    print("E2  JACOBIAN-VARIATION SWEEP  —  when does LTI stop being a model?")
    print("=" * 78)
    print(f"{'excur':>6}{'S':>8}{'mu_min':>9}{'A2 rms':>9}{'A3 rms':>9}{'gap %':>9}")
    rows = []
    for x in (0.25, 0.5, 0.75, 1.0, 1.25):
        ref = build_reference(1.6, excursion=x)
        cs = make_controllers(ref)
        r2 = run(cs[1], ref)
        r3 = run(cs[2], ref)
        gap = 100 * (r2["rms_mm"] - r3["rms_mm"]) / max(r3["rms_mm"], 1e-9)
        rows.append((x, ref["S"], r2, r3, gap))
        print(f"{x:>6.2f}{ref['S']:>8.3f}{r2['mu_min']:>9.3f}"
              f"{r2['rms_mm']:>9.3f}{r3['rms_mm']:>9.3f}{gap:>9.1f}")
    return rows


def experiment_mismatch():
    print("\n" + "=" * 78)
    print("E3  MISMATCH SWEEP  —  when does relinearising at the measurement pay?")
    print("=" * 78)
    print(f"{'plant gain':>11}{'L(A3)':>8}{'A3 rms':>9}{'A4 rms':>9}{'gap %':>9}")
    rows = []
    for pg in (1.0, 0.9, 0.75, 0.6, 0.45):
        ref = build_reference(1.6)
        cs = make_controllers(ref)
        r3 = run(cs[2], ref, plant_gain=pg)
        r4 = run(cs[3], ref, plant_gain=pg)
        gap = 100 * (r3["rms_mm"] - r4["rms_mm"]) / max(r4["rms_mm"], 1e-9)
        rows.append((pg, r3, r4, gap))
        print(f"{pg:>11.2f}{r3['L']:>8.3f}{r3['rms_mm']:>9.3f}"
              f"{r4['rms_mm']:>9.3f}{gap:>9.1f}")
    return rows


def experiment_full_ladder():
    print("\n" + "=" * 78)
    print("E4  THE LADDER, at the aggressive and the relaxed plan")
    print("=" * 78)
    for g, label in ((1.0, "gamma=1.0  (time-optimal, no headroom)"),
                     (2.0, "gamma=2.0  (dilated, headroom everywhere)")):
        ref = build_reference(g)
        cs = make_controllers(ref)
        print(f"\n  {label}   headroom={ref['headroom']:.3f}  S={ref['S']:.3f}")
        print(f"  {'controller':<20}{'rms mm':>9}{'max mm':>9}"
              f"{'activity':>10}{'violation':>11}")
        for c in cs:
            r = run(c, ref, plant_gain=0.75)
            print(f"  {c.name:<20}{r['rms_mm']:>9.3f}{r['max_mm']:>9.3f}"
                  f"{r['activity']:>10.3f}{r['violation']:>11.3f}")


if __name__ == "__main__":
    print(f"[qp] ADMM solver vs scipy SLSQP, worst relative objective gap: "
          f"{_validate_qp():.2e}")
    experiment_headroom()
    experiment_scheduling()
    experiment_mismatch()
    experiment_full_ladder()
