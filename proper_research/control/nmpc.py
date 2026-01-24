import numpy as np
from scipy.optimize import minimize
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.models.forward_model import make_forward_fn, make_jac_fn
from proper_research.parameters import default_magnet_params
mag_params = default_magnet_params()

mag = 128e3
r = 0.0015
A_cs = np.pi*r**2
E = 3.5e6
I = np.pi*r**4/4
L = 0.05
rho = 0.15
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
eps = np.array([
    np.deg2rad(0.5),
    np.deg2rad(0.5),
    1e-3,
    5e-4
])
class nmpc_controller_tipxy_simple:
    """
    NMPC that:
      - optimizes U over horizon directly (nonlinear cost via forward_tip_fn)
      - uses linear constraints for p bounds via cumulative sum matrix S
      - keeps your input bounds as box bounds
      - keeps optional offset-free disturbance d the same way
    """

    def __init__(self, *, forward_tip_fn,
                 dt=0.05, Np=10,
                 w_xy=(50.0, 50.0),
                 w_u=(1e-3, 1e-3, 1e-2, 1e-2),
                 w_du=(0.0, 0.0, 0.0, 0.0),
                 u_max=(np.deg2rad(20), np.deg2rad(20), 0.02, 0.01),
                 p_min=(np.deg2rad(-90), np.deg2rad(-90), 0.10, 0.03),
                 p_max=(np.deg2rad( 90), np.deg2rad( 90), 0.25, 0.08),
                 use_offset_free=True,
                 d_alpha=0.15):

        self.forward_tip_fn = forward_tip_fn
        self.dt = float(dt)
        self.Np = int(Np)

        self.Q  = np.diag(np.asarray(w_xy, dtype=float).ravel())
        self.R  = np.diag(np.asarray(w_u, dtype=float).ravel())
        self.Rd = np.diag(np.asarray(w_du, dtype=float).ravel())

        self.u_max = np.asarray(u_max, dtype=float).ravel()
        self.p_min = np.asarray(p_min, dtype=float).ravel()
        self.p_max = np.asarray(p_max, dtype=float).ravel()

        self.use_offset_free = bool(use_offset_free)
        self.d_alpha = float(d_alpha)
        self.d = np.zeros(2, dtype=float)

        self.p = None
        self.x = None
        self.U_warm = None  # (Np*4,)

        # Precompute S such that p_stack = p0_stack + S @ U_flat
        # where p_stack is (Np*4,) containing p1..pNp stacked.
        self.S = self._build_S()

    def _build_S(self):
        """
        S = kron(tril(ones(Np,Np))*dt, I4)
        so that stacked p1..pNp = p0 replicated + S @ (u0..uNp-1)
        """
        Np = self.Np
        dt = self.dt
        S_np = np.tril(np.ones((Np, Np))) * dt
        return np.kron(S_np, np.eye(4))

    def set_dt(self, dt):
        self.dt = float(dt)
        self.S = self._build_S()

    def set_initial_params(self, gamma0, beta0, rho0, L0):
        self.p = np.array([gamma0, beta0, rho0, L0], dtype=float)
        self.x = np.asarray(self.forward_tip_fn(self.p), dtype=float).reshape(2,)
        self.d = np.zeros(2, dtype=float)
        self.U_warm = None

    def _p_seq_from_U_flat(self, p0, U_flat):
        """
        Compute p1..pNp from linear integrator:
           p_k = p0 + dt * sum_{i<=k-1} u_i
        WITHOUT clamping (constraints enforce bounds).
        """
        Np = self.Np
        p0 = np.asarray(p0, dtype=float).reshape(4,)
        U_flat = np.asarray(U_flat, dtype=float).reshape(Np*4,)

        p0_stack = np.tile(p0, Np)             # (Np*4,)
        p_stack  = p0_stack + self.S @ U_flat  # (Np*4,)
        return p_stack.reshape(Np, 4)

    def _rollout_X_from_U_flat(self, p0, U_flat):
        """
        Nonlinear output rollout using forward_tip_fn(p_k).
        Returns:
          p_seq (Np,4), X (Np,2)
        """
        p_seq = self._p_seq_from_U_flat(p0, U_flat)
        X = np.zeros((self.Np, 2), dtype=float)
        for k in range(self.Np):
            xk = np.asarray(self.forward_tip_fn(p_seq[k]), dtype=float).reshape(2,)
            if self.use_offset_free:
                xk = xk + self.d
            X[k] = xk
        return p_seq, X

    def step(self, xref_seq, x_meas=None, debug=False):
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")
        p_prev = self.p.copy()
        x_prev = self.x.copy()
        # measurement update
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(2,)

        # offset-free disturbance update
        if self.use_offset_free and x_meas is not None:
            x_model = np.asarray(self.forward_tip_fn(self.p), dtype=float).reshape(2,)
            r = self.x - x_model
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        Np = self.Np
        m = 4

        xref_seq = np.asarray(xref_seq, dtype=float).reshape(Np, 2)

        # warm start
        if self.U_warm is None or self.U_warm.size != Np*m:
            U0 = np.zeros(Np*m, dtype=float)
        else:
            U0 = self.U_warm.copy()

        # bounds on U (box)
        bnds = [(-self.u_max[j], self.u_max[j]) for k in range(Np) for j in range(m)]

        # ---- linear inequality constraints for p bounds ----
        # p_stack(U) = p0_stack + S U
        # enforce: p_min <= p_stack <= p_max
        p0_stack = np.tile(self.p, Np)

        # We provide constraints in the SLSQP form g(U) >= 0.
        # Lower:  (p_stack - p_min_stack) >= 0
        # Upper:  (p_max_stack - p_stack) >= 0
        pmin_stack = np.tile(self.p_min, Np)
        pmax_stack = np.tile(self.p_max, Np)

        def g_lower(U_flat):
            p_stack = p0_stack + self.S @ U_flat
            return p_stack - pmin_stack

        def g_upper(U_flat):
            p_stack = p0_stack + self.S @ U_flat
            return pmax_stack - p_stack

        cons = [
            {"type": "ineq", "fun": g_lower},
            {"type": "ineq", "fun": g_upper},
        ]

        # ---- nonlinear objective ----
        Q = self.Q
        R = self.R
        Rd = self.Rd

        def cost(U_flat):
            U_flat = np.asarray(U_flat, dtype=float)
            U_seq = U_flat.reshape(Np, m)

            _, X = self._rollout_X_from_U_flat(self.p, U_flat)

            J = 0.0
            for k in range(Np):
                e = (X[k] - xref_seq[k]).reshape(2, 1)
                u = U_seq[k].reshape(m, 1)
                J += float(e.T @ Q @ e) + float(u.T @ R @ u)

            if Np > 1 and np.any(np.diag(Rd) > 0):
                for k in range(Np - 1):
                    du = (U_seq[k+1] - U_seq[k]).reshape(m, 1)
                    J += float(du.T @ Rd @ du)

            return J

        res = minimize(
            cost,
            U0,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 60, "ftol": 1e-6, "disp": bool(debug)},
        )

        solved = bool(res.success) and np.all(np.isfinite(res.x))
        if not solved:
            U_opt = np.zeros(Np*m, dtype=float)
            status = f"fail:{res.message}"
            infeasible = 1
        else:
            U_opt = np.asarray(res.x, dtype=float)
            status = "solved"
            infeasible = 0

        U_seq = U_opt.reshape(Np, m)
        u0 = U_seq[0].copy()

        # predicted trajectory for logging
        p_seq, X_pred = self._rollout_X_from_U_flat(self.p, U_opt)

        # apply first control (hard clamp at the real controller state update)
        p_next = self.p + self.dt * u0
        p_next = np.minimum(np.maximum(p_next, self.p_min), self.p_max)
        self.p = p_next
        self.x = np.asarray(self.forward_tip_fn(self.p), dtype=float).reshape(2,)

        # warm start shift
        if solved:
            U_shift = np.vstack([U_seq[1:], U_seq[-1:]])  # hold last
            self.U_warm = U_shift.reshape(-1)
        else:
            self.U_warm = None


        info = dict(
            status=status,
            infeasible=int(infeasible),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            d=self.d.copy(),
            X_pred=X_pred.copy(),
            U_seq=U_seq.copy(),
            p_seq=p_seq.copy(),
            solver_iters=int(getattr(res, "nit", -1)),
            solver_cost=float(getattr(res, "fun", np.nan)),

            # NEW (for debug parity with your linear/SQP controller)
            p_prev=p_prev,
            x_prev=x_prev,
        )
        return self.p.copy(), self.x.copy(), info
def debug_step_tipxy_nmpc(k, x_target, xref_seq, p_now, x_now, info, nmpc, print_horizon=4):
    """
    NMPC debug print with the same semantics as your successive-linearization debug:
      - x_prev is x_k (the plant/output BEFORE applying u0)
      - x_next_true is forward_tip(p_k + dt*u0) (with the SAME clamp as in step())
      - x_pred0 is X_pred[0] from the NMPC rollout (which should represent x_{k+1})
    """

    # ---- current tracking error (after the step(), so x_now is x_{k+1} in your loop) ----
    e = x_target - x_now
    err_mm = 1e3 * np.linalg.norm(e)

    print(f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f}] "
          f"||e||={err_mm:.2f}mm status={info['status']}")

    u0 = info["u0"]
    X_pred = info["X_pred"]
    U_seq = info["U_seq"]
    p_seq = info.get("p_seq", None)

    # ---- print p_now (this is p AFTER apply in your current API) ----
    print(f"   p_now: gamma={np.rad2deg(p_now[0]):+.1f}deg, beta={np.rad2deg(p_now[1]):+.1f}deg, "
          f"rho={p_now[2]:.3f}m, L={p_now[3]:.3f}m")

    # ---- u0 ----
    print(f"   u0: gdot={np.rad2deg(u0[0]):+.2f}deg/s, bdot={np.rad2deg(u0[1]):+.2f}deg/s, "
          f"rhodot={u0[2]:+.4f}m/s, Ldot={u0[3]:+.4f}m/s")

    # ---- reconstruct x_prev and x_next_true consistently ----
    # We cannot trust loop variables to mean x_k vs x_{k+1}, so store x_prev in info (recommended),
    # but if you haven't, we reconstruct from the fact that X_pred is based on the pre-step p.
    #
    # Best: modify NMPC.step() to store x_prev, p_prev (see below). If present, use them.
    x_prev = info.get("x_prev", None)
    p_prev = info.get("p_prev", None)

    if x_prev is None or p_prev is None:
        print("   NOTE: x_prev/p_prev not found in info. Add them in step() for exact debugging.")
    else:
        # "true" next state per the controller update law
        p_next_true = p_prev + nmpc.dt * u0
        p_next_true = np.minimum(np.maximum(p_next_true, nmpc.p_min), nmpc.p_max)

        x_next_true = np.asarray(nmpc.forward_tip_fn(p_next_true), dtype=float).reshape(2,)
        if nmpc.use_offset_free:
            x_next_true = x_next_true + nmpc.d

        x_pred0 = X_pred[0].copy()

        pred0_vec_err = (x_next_true - x_pred0).copy()
        pred1_err = float(np.linalg.norm(pred0_vec_err))

        print("x_prev     =", x_prev)
        print("x_next_true=", x_next_true)
        print("x_pred0    =", x_pred0)
        print("err (mm)   =", 1e3 * pred1_err)
        print("err vec(mm)=", 1e3 * pred0_vec_err)

    # ---- horizon print ----
    ph = min(print_horizon, X_pred.shape[0])
    if np.all(np.isfinite(X_pred)):
        for i in range(ph):
            print(f"   pred[{i}]={X_pred[i]}   ref[{i}]={xref_seq[i]}")
    else:
        print("   X_pred NaNs -> infeasible solve")

    # ---- optional: show first few p_seq values if logged ----
    if p_seq is not None and np.all(np.isfinite(p_seq)):
        ph2 = min(print_horizon, p_seq.shape[0])
        for i in range(ph2):
            print(f"   p_seq[{i}] = ["
                  f"g={np.rad2deg(p_seq[i,0]):+.1f}deg, "
                  f"b={np.rad2deg(p_seq[i,1]):+.1f}deg, "
                  f"rho={p_seq[i,2]:.3f}, L={p_seq[i,3]:.3f}]")

    print()    
forward_tip = make_forward_fn(mag, A_cs, E, I, mag_params.mag_epm)
Jxy_fn = make_jac_fn(forward_tip, eps)

nmpc = nmpc_controller_tipxy_simple(
    forward_tip_fn=forward_tip,
    dt=0.1,
    Np=4,
    w_xy=(100.0, 100.0),
    w_u=(1e-4, 1e-4, 1e-2, 1e-2),
    w_du=(1e-3, 1e-3, 1e-2, 1e-2),
    u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
    p_min=p_min,
    p_max=p_max,
    use_offset_free=True,
    d_alpha=0.15,
)
p0 = np.array([np.deg2rad(10), np.deg2rad(5), 0.15, 0.05])
nmpc.set_initial_params(*p0)
print("Initial tip:", nmpc.x)
x_target = forward_tip([np.deg2rad(40), np.deg2rad(0), 0.12, 0.06])
print(f"x_target is {x_target}")
for k in range(10):
    x_now = nmpc.x.copy()
    p_now = nmpc.p.copy()
    xref_seq = np.tile(x_target, (nmpc.Np, 1))

    p_cmd, x_now, info = nmpc.step(xref_seq)

    debug_step_tipxy_nmpc(k, x_target, xref_seq, p_cmd, x_now, info, nmpc, print_horizon=4)
print(f"Final tip position is: [{x_now[0]:+.4f}, {x_now[1]:+.4f}] with target: {x_target}")
theta_y_tip = np.arctan2(x_now[1], x_now[0])
theta_y_target = np.arctan2(x_target[1], x_target[0])
print(f"Angle is: {np.rad2deg(theta_y_tip)}")
print(f"Angle is of target: {np.rad2deg(theta_y_target)}")

