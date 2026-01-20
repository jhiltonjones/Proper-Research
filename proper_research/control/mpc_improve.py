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
def dare_stabilising_K(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
def seq_mat_ltv(A, B_list):
    """
    Build stacked prediction matrices for time-varying B_k (LTV system)
      x_{k+1} = A x_k + B_k u_k

    Returns:
      Mx: (Np*n, n)
      Mc: (Np*n, Np*m)

    B_list: list length Np with each B_k shape (n,m)
            where B_0 corresponds to step from x0 -> x1
    """
    B_list = [np.asarray(B) for B in B_list]
    Np = len(B_list)
    n, m = B_list[0].shape

    Mx = np.zeros((Np*n, n))
    Mc = np.zeros((Np*n, Np*m))

    A_pow = np.eye(n)

    for i in range(Np):
        # x_{i+1} = A^{i+1} x0 + sum_{j=0..i} A^{i-j} B_j u_j
        A_pow = A @ A_pow
        Mx[i*n:(i+1)*n, :] = A_pow

        for j in range(i + 1):
            A_ij = np.linalg.matrix_power(A, i - j)
            Mc[i*n:(i+1)*n, j*m:(j+1)*m] = A_ij @ B_list[j]

    return Mx, Mc

def seq_mat_lti(A, B, N):
    n, m = B.shape
    Mx = np.zeros((N*n, n))
    Mc = np.zeros((N*n, N*m))

    A_pow = np.eye(n)

    A_pow = A @ A_pow
    Mx[0:n, :] = A_pow
    Mc[0:n, 0:m] = B

    for i in range(1, N):
        A_pow = A @ A_pow
        Mx[i*n:(i+1)*n, :] = A_pow
        Mc[i*n:(i+1)*n, 0:i*m] = A @ Mc[(i-1)*n:i*n, 0:i*m]
        Mc[i*n:(i+1)*n, i*m:(i+1)*m] = B

    return Mx, Mc


def solve_qp_osqp(H, f, A, l, u, U_warm=None):
    P = sp.csc_matrix(0.5 * (H + H.T))
    q = f.astype(float)
    A = sp.csc_matrix(A)

    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    if U_warm is not None:
        prob.warm_start(x=U_warm)

    res = prob.solve()
    status = res.info.status
    if status not in ("solved", "solved inaccurate"):
        return None, None, status

    return res.x, res.y, status

class mpc_controller_tipxy_LTI:

    def __init__(self, *, Jxy_fn, forward_tip_fn,
                 dt=0.05, Np=10,
                 w_xy=(50.0, 50.0),
                 w_u=(1e-3, 1e-3, 1e-2, 1e-2),
                 w_du=(0.0, 0.0, 0.0, 0.0),
                 band_xy=0.0,
                 u_max=(np.deg2rad(20), np.deg2rad(20), 0.02, 0.01),
                 p_min=(np.deg2rad(-90), np.deg2rad(-90), 0.10, 0.03),
                 p_max=(np.deg2rad( 90), np.deg2rad( 90), 0.25, 0.08),
                 # --- NEW (SQP + integral action) ---
                 N_sqp=3,
                 use_offset_free=True,
                 d_alpha=0.15):
        """
        N_sqp: number of successive linearization iterations per MPC step (2-4 typical)
        use_offset_free: enable offset-free MPC (disturbance/bias estimation)
        d_alpha: low-pass filter gain for disturbance update (0.05-0.3 typical)
        """

        self.U_warm = None  # OSQP warm-start vector (Np*m,)

        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

        self.dt = float(dt)
        self.Np = int(Np)

        self.A = np.eye(2)

        self.Q = np.diag(np.asarray(w_xy, dtype=float).ravel())  # 2x2
        self.R = np.diag(np.asarray(w_u, dtype=float).ravel())   # 4x4
        self.Rd = np.diag(np.asarray(w_du, dtype=float).ravel()) # 4x4

        self.band_xy = float(band_xy)

        self.u_max = np.asarray(u_max, dtype=float).ravel()
        self.p_min = np.asarray(p_min, dtype=float).ravel()
        self.p_max = np.asarray(p_max, dtype=float).ravel()

        self.p = None
        self.x = None
        self.Qf = None

        self._rebuild_S()
        self.Du = self._build_Du_matrix()

        # --- NEW: successive linearization settings ---
        self.N_sqp = int(N_sqp)

        # --- NEW: offset-free tracking via disturbance state d ---
        self.use_offset_free = bool(use_offset_free)
        self.d_alpha = float(d_alpha)
        self.d = np.zeros(2, dtype=float)   # disturbance/bias on tip dynamics

    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

    def _build_Du_matrix(self):
        Np = self.Np
        m = 4
        if Np <= 1:
            return np.zeros((0, Np*m))

        D1 = np.zeros((Np-1, Np))
        for i in range(Np-1):
            D1[i, i]   = -1.0
            D1[i, i+1] = +1.0

        return np.kron(D1, np.eye(m))

    def set_dt(self, dt):
        self.dt = float(dt)
        self._rebuild_S()
        self.Du = self._build_Du_matrix()

    def set_initial_params(self, gamma0, beta0, rho0, L0):
        self.p = np.array([gamma0, beta0, rho0, L0], dtype=float)
        self.x = self.forward_tip_fn(self.p)
        self.d = np.zeros(2, dtype=float)
        self.U_warm = None

    def _compute_Qtil(self, B0):
        # terminal cost via DARE with B0 (first-step)
        _, P = dare_stabilising_K(self.A, B0, self.Q, self.R)
        self.Qf = P

        n = 2
        Np = self.Np
        Qtil = np.zeros((Np*n, Np*n))
        if Np > 1:
            Qtil[:(Np-1)*n, :(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
        Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf
        return Qtil

    def _disturbance_stack(self, d):
        """
        Build stacked contribution of constant disturbance d across horizon.
        For A = I:
          x1 gets +1*d
          x2 gets +2*d
          ...
        For generic A:
          d_acc_{i+1} = A d_acc_i + d
        """
        n = 2
        Np = self.Np
        d = np.asarray(d, dtype=float).reshape(n,)

        d_stack = np.zeros((Np*n, 1))
        d_acc = np.zeros(n)

        for i in range(Np):
            d_acc = self.A @ d_acc + d
            d_stack[i*n:(i+1)*n, 0] = d_acc

        return d_stack

    def _clamp_p(self, p):
        return np.minimum(np.maximum(p, self.p_min), self.p_max)

    def _p_seq_from_U(self, p0, U_seq):
        """
        Integrate p forward: p_{i+1} = clamp(p_i + dt * u_i)
        Returns list length Np of p_{i+1} values.
        """
        p_running = p0.copy()
        p_list = []
        for i in range(self.Np):
            p_running = p_running + U_seq[i] * self.dt
            p_running = self._clamp_p(p_running)
            p_list.append(p_running.copy())
        return np.array(p_list)

    def _build_ltv_prediction_mats(self, p0, U_guess):
        """
        Build LTV (time-varying) B_i = dt * J(p_i) along horizon based on U_guess.
        """
        if U_guess is None:
            U_guess = np.zeros((self.Np, 4))

        p_seq = self._p_seq_from_U(p0, U_guess)  # (Np,4), these are p1..pNp
        B_list = []
        J_list = []
        for i in range(self.Np):
            Ji = np.asarray(self.Jxy_fn(p_seq[i]), dtype=float)  # (2,4)
            J_list.append(Ji)
            B_list.append(self.dt * Ji)

        Mx, Mc = seq_mat_ltv(self.A, B_list)
        return p_seq, J_list, B_list, Mx, Mc

    def step(self, xref_seq, x_meas=None):
        """
        xref_seq: (Np,2)
        x_meas: measured tip (2,) for offset-free update. If None, uses internal x.
        """

        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        # measurement update
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(2,)

        # --- offset-free disturbance update ---
        # Use model residual to update d (low-pass filtered)
        if self.use_offset_free and x_meas is not None:
            x_model = self.forward_tip_fn(self.p)  # predicted from p
            r = self.x - x_model                   # residual
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        n = 2
        m = 4
        Np = self.Np

        xref_seq = np.asarray(xref_seq, dtype=float).reshape(Np, 2)
        xref_stack = xref_seq.reshape(Np*n, 1)

        # --- SQP / successive linearization loop ---
        # initial guess U: warm-start if available else zeros
        if self.U_warm is not None and self.U_warm.size == Np*m:
            U_opt_vec = self.U_warm.copy()
            U_guess = U_opt_vec.reshape(Np, m)
        else:
            U_guess = np.zeros((Np, m))
            U_opt_vec = None

        status_last = "init"

        for it in range(self.N_sqp):
            # build LTV model based on current U_guess
            p0 = self.p.copy()
            p_seq, J_list, B_list, Mx, Mc = self._build_ltv_prediction_mats(p0, U_guess)

            # stage+terminal cost
            Qtil = self._compute_Qtil(B_list[0])
            Rtil = np.kron(np.eye(Np), self.R)

            # delta-u penalty
            if self.Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np-1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            xk = self.x.reshape(2, 1)

            # baseline predicted state stack (no control)
            X0_stack = (Mx @ xk).reshape(Np*n, 1)

            # add disturbance contribution (offset-free)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)

            # objective
            H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
            f = 2.0 * (Mc.T @ Qtil @ (X0_stack - xref_stack))

            # constraints
            A_list, l_list, u_list = [], [], []

            # (1) input bounds
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np * m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            # (2) parameter bounds across horizon (via integrated controls)
            if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
                A_p = np.kron(self.S_np, np.eye(m))
                p0_stack = np.tile(self.p, Np)

                l_p = np.tile(self.p_min, Np) - p0_stack
                u_p = np.tile(self.p_max, Np) - p0_stack

                A_list.append(A_p)
                l_list.append(l_p)
                u_list.append(u_p)

            # (3) tube constraint around reference (optional)
            if self.band_xy > 0.0:
                band = float(self.band_xy)
                band_stack = band * np.ones((Np*n,))

                rhs_p = band_stack + (xref_stack - X0_stack).reshape(-1)
                rhs_n = band_stack + (X0_stack - xref_stack).reshape(-1)

                A_list.append(Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_p)

                A_list.append(-Mc)
                l_list.append(-np.inf * np.ones(Np*n))
                u_list.append(rhs_n)

            # stack constraints for OSQP
            if A_list:
                A_osqp = np.vstack(A_list)
                l_osqp = np.concatenate(l_list)
                u_osqp = np.concatenate(u_list)
            else:
                A_osqp = np.zeros((0, Np*m))
                l_osqp = np.zeros(0)
                u_osqp = np.zeros(0)

            # solve QP with warm-start (inside SQP loop we warm-start from last iterate)
            U_warm_vec = U_opt_vec if U_opt_vec is not None else self.U_warm
            U_opt_vec, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=U_warm_vec)
            status_last = status

            infeas = (status not in ("solved", "solved inaccurate")) or (U_opt_vec is None)
            if infeas:
                # if infeasible, break SQP loop and apply zero control
                U_guess = np.zeros((Np, m))
                U_opt_vec = None
                break

            # update guess for next SQP iteration
            U_guess = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)

        # --- apply first control action ---
        if U_opt_vec is None:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
            X_pred = np.full((Np, n), np.nan)
            infeas_final = True
        else:
            U_seq = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]
            # build final predicted trajectory using last Mc/Mx
            # note: Mc/Mx correspond to last SQP iteration
            xk = self.x.reshape(2, 1)
            X0_stack = (Mx @ xk).reshape(Np*n, 1)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)
            X_pred_stack = (X0_stack + Mc @ U_opt_vec.reshape(-1, 1))
            X_pred = X_pred_stack.reshape(Np, n)
            infeas_final = False

        # update parameter state
        self.p = self._clamp_p(self.p + u0 * self.dt)

        # update plant state using nonlinear forward model
        self.x = self.forward_tip_fn(self.p)

        # store warm-start for next MPC step
        if not infeas_final:
            self.U_warm = np.asarray(U_opt_vec, dtype=float).copy()
        else:
            self.U_warm = None

        info = dict(
            status=status_last,
            infeasible=int(infeas_final),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            d=self.d.copy(),
            X_pred=X_pred.copy(),
            U_seq=U_seq.copy(),
            N_sqp=self.N_sqp,
        )
        return self.p.copy(), self.x.copy(), info
# class mpc_controller_tipxy_LTI:

#     def __init__(self, *, Jxy_fn, forward_tip_fn,
#                  dt=0.05, Np=10,
#                  w_xy=(50.0, 50.0),
#                  w_u=(1e-3, 1e-3, 1e-2, 1e-2),
#                  w_du=(0.0, 0.0, 0.0, 0.0),
#                  band_xy=0.0,
#                  u_max=(np.deg2rad(20), np.deg2rad(20), 0.02, 0.01),
#                  p_min=(np.deg2rad(-90), np.deg2rad(-90), 0.10, 0.03),
#                  p_max=(np.deg2rad( 90), np.deg2rad( 90), 0.25, 0.08)):
#         self.U_warm = None  # warm-start vector for OSQP (Np*m,)
#         self.Jxy_fn = Jxy_fn
#         self.forward_tip_fn = forward_tip_fn

#         self.dt = float(dt)
#         self.Np = int(Np)

#         # state x = [x_tip, y_tip]
#         self.A = np.eye(2)

#         self.Q = np.diag(np.asarray(w_xy, dtype=float).ravel())  # 2x2
#         self.R = np.diag(np.asarray(w_u, dtype=float).ravel())   # 4x4

#         self.band_xy = float(band_xy)

#         self.u_max = np.asarray(u_max, dtype=float).ravel()  # (4,)
#         self.p_min = np.asarray(p_min, dtype=float).ravel()  # (4,)
#         self.p_max = np.asarray(p_max, dtype=float).ravel()  # (4,)

#         self.p = None      # (4,)
#         self.x = None      # (2,)
#         self.Qf = None
#         self.Rd = np.diag(np.asarray(w_du, dtype=float).ravel())   # 4x4  
#         self._rebuild_S()
#         self.Du = self._build_Du_matrix()
#     def _build_Du_matrix(self):
#         """
#         Creates D such that (D @ U) = [u1-u0, u2-u1, ..., u_{N-1}-u_{N-2}]
#         Shape: ((Np-1)*m, Np*m)
#         """
#         Np = self.Np
#         m = 4
#         if Np <= 1:
#             return np.zeros((0, Np*m))

#         D1 = np.zeros((Np-1, Np))
#         for i in range(Np-1):
#             D1[i, i]   = -1.0
#             D1[i, i+1] = +1.0

#         return np.kron(D1, np.eye(m))
#     def _rebuild_S(self):
#         # lower-triangular integration matrix: p_k+i = p_k + dt * sum_{j<=i} u_j
#         self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

#     def set_dt(self, dt):
#         self.dt = float(dt)
#         self._rebuild_S()

#     def set_initial_params(self, gamma0, beta0, rho0, L0):
#         self.p = np.array([gamma0, beta0, rho0, L0], dtype=float)
#         self.x = self.forward_tip_fn(self.p)  # initialise state from forward model

#     def _build_lti_model(self):
#         if self.p is None:
#             raise ValueError("Call set_initial_params(...) before step().")

#         # Linearise around current p
#         Jxy = np.asarray(self.Jxy_fn(self.p), dtype=float)  # (2,4)
#         if Jxy.shape != (2, 4):
#             raise ValueError(f"Expected Jxy shape (2,4), got {Jxy.shape}")

#         Bmat = self.dt * Jxy  # (2,4)

#         # terminal cost via DARE
#         _, P = dare_stabilising_K(self.A, Bmat, self.Q, self.R)
#         self.Qf = P

#         Mx, Mc = seq_mat_lti(self.A, Bmat, self.Np)
#         return Jxy, Bmat, Mx, Mc

#     def step(self, xref_seq, x_meas=None):
#         """
#         xref_seq: shape (Np,2) reference tip positions
#         x_meas: optional measured tip state (2,) if you have sensor feedback.
#                 If None, we use current stored x.
#         """
#         if x_meas is not None:
#             self.x = np.asarray(x_meas, dtype=float).reshape(2,)

#         Jxy, Bmat, Mx, Mc = self._build_lti_model()

#         n = 2
#         m = 4
#         Np = self.Np

#         # Build Qtil (stage costs + terminal)
#         Qtil = np.zeros((Np*n, Np*n))
#         if Np > 1:
#             Qtil[:(Np-1)*n, :(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
#         Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf


#         xk = self.x.reshape(2, 1)
#         xref_seq = np.asarray(xref_seq, dtype=float).reshape(Np, 2)
#         xref_stack = xref_seq.reshape(Np*n, 1)

#         X0_stack = (Mx @ xk).reshape(Np*n, 1)

#         Rtil = np.kron(np.eye(Np), self.R)

#         # delta-u penalty matrix
#         # penalize differences u_{i+1}-u_i
#         if self.Np > 1 and np.any(np.diag(self.Rd) > 0):
#             Rd_til = np.kron(np.eye(Np-1), self.Rd)  # ((Np-1)*m x (Np-1)*m)
#             H_du = self.Du.T @ Rd_til @ self.Du      # (Np*m x Np*m)
#         else:
#             H_du = 0.0

#         H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
#         f = 2.0 * (Mc.T @ Qtil @ (X0_stack - xref_stack))

#         A_list, l_list, u_list = [], [], []

#         # (1) input bounds: -u_max <= u <= u_max
#         if np.all(np.isfinite(self.u_max)):
#             A_u = np.eye(Np * m)
#             umax_stack = np.tile(self.u_max, Np)
#             A_list.append(A_u)
#             l_list.append(-umax_stack)
#             u_list.append(+umax_stack)

#         # (2) parameter bounds across horizon: p_min <= p0 + S*u <= p_max
#         if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
#             A_p = np.kron(self.S_np, np.eye(m))  # (Np*m, Np*m)
#             p0_stack = np.tile(self.p, Np)       # (Np*m,)

#             l_p = np.tile(self.p_min, Np) - p0_stack
#             u_p = np.tile(self.p_max, Np) - p0_stack

#             A_list.append(A_p)
#             l_list.append(l_p)
#             u_list.append(u_p)

#         # (3) optional tube constraint around xref (component-wise box)
#         if self.band_xy > 0.0:
#             band = float(self.band_xy)
#             band_stack = band * np.ones((Np*n,))

#             # constraint:  X_pred = X0 + Mc U
#             # enforce: -band <= X_pred - xref <= +band
#             # => Mc U <= band + xref - X0
#             # => -Mc U <= band - xref + X0
#             rhs_p = band_stack + (xref_stack - X0_stack).reshape(-1)
#             rhs_n = band_stack + (X0_stack - xref_stack).reshape(-1)

#             A_list.append(Mc)
#             l_list.append(-np.inf * np.ones(Np*n))
#             u_list.append(rhs_p)

#             A_list.append(-Mc)
#             l_list.append(-np.inf * np.ones(Np*n))
#             u_list.append(rhs_n)

#         # Stack constraints
#         if A_list:
#             A_osqp = np.vstack(A_list)
#             l_osqp = np.concatenate(l_list)
#             u_osqp = np.concatenate(u_list)
#         else:
#             A_osqp = np.zeros((0, Np*m))
#             l_osqp = np.zeros(0)
#             u_osqp = np.zeros(0)

#         U_opt, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=self.U_warm)
#         infeas = (status not in ("solved", "solved inaccurate")) or (U_opt is None)
#         if not infeas:
#             self.U_warm = np.asarray(U_opt, dtype=float).copy()
#         else:
#             self.U_warm = None
#         if infeas:
#             u0 = np.zeros(m)
#             U_seq = np.zeros((Np, m))
#             X_pred = np.full((Np, n), np.nan)
#         else:
#             U_seq = np.asarray(U_opt, dtype=float).reshape(Np, m)
#             u0 = U_seq[0, :]
#             X_pred_stack = (X0_stack + Mc @ U_opt.reshape(-1, 1))
#             X_pred = X_pred_stack.reshape(Np, n)

#         self.p = self.p + u0 * self.dt
#         self.p = np.minimum(np.maximum(self.p, self.p_min), self.p_max)

#         self.x = self.forward_tip_fn(self.p)

#         info = dict(
#             status=status,
#             infeasible=int(infeas),
#             u0=u0.copy(),
#             p_now=self.p.copy(),
#             x_now=self.x.copy(),
#             Jxy=Jxy.copy(),
#             Bmat=Bmat.copy(),
#             X_pred=X_pred.copy(),
#             U_seq=U_seq.copy(),
#         )
#         return self.p.copy(), self.x.copy(), info
forward_tip = make_forward_fn(mag, A_cs, E, I, mag_params.mag_epm)
Jxy_fn = make_jac_fn(forward_tip, eps)
# mpc_xy = mpc_controller_tipxy_LTI(
#     Jxy_fn=Jxy_fn,
#     forward_tip_fn=forward_tip,
#     dt=0.1,
#     Np=10,
#     w_xy=(100.0, 100.0),
#     w_u=(1e-4, 1e-4, 1e-2, 1e-2),
#     w_du=(1e-3, 1e-3, 1e-2, 1e-2),     # <--- new
#     band_xy=0.0,
#     u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
#     p_min=p_min,
#     p_max=p_max
# )
mpc_xy = mpc_controller_tipxy_LTI(
    Jxy_fn=Jxy_fn,
    forward_tip_fn=forward_tip,
    dt=0.2,
    Np=10,
    w_xy=(100.0, 100.0),
    w_u=(1e-4, 1e-4, 1e-2, 1e-2),
    w_du=(1e-3, 1e-3, 1e-2, 1e-2),
    u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
    p_min=p_min,
    p_max=p_max,

    # --- NEW ---
    N_sqp=3,               # 2–4 is typical
    use_offset_free=True,  # integral-like disturbance rejection
    d_alpha=0.15           # filter strength for bias estimation
)
def fmt_xy(v, fmt="{:+.4f}"):
    v = np.asarray(v).ravel()
    return f"[{fmt.format(v[0])}, {fmt.format(v[1])}]"

def fmt_p(p, fmt_ang="{:+.1f}", fmt_lin="{:.3f}"):
    p = np.asarray(p).ravel()
    return (f"[gamma={fmt_ang.format(np.rad2deg(p[0]))}deg, "
            f"beta={fmt_ang.format(np.rad2deg(p[1]))}deg, "
            f"rho={fmt_lin.format(p[2])}m, "
            f"L={fmt_lin.format(p[3])}m]")

def fmt_u(u, fmt_ang="{:+.4f}", fmt_lin="{:+.5f}"):
    u = np.asarray(u).ravel()
    return (f"[gdot={fmt_ang.format(np.rad2deg(u[0]))} deg/s, "
            f"bdot={fmt_ang.format(np.rad2deg(u[1]))} deg/s, "
            f"rhodot={fmt_lin.format(u[2])} m/s, "
            f"Ldot={fmt_lin.format(u[3])} m/s]")

def debug_step_tipxy(k, x_target, xref_seq, p_cmd, x_now, info, print_horizon=4):
    e = x_target - x_now
    err_mm = 1e3 * np.linalg.norm(e)

    print(f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f}] "
          f"||e||={err_mm:.2f}mm status={info['status']}")

    u0 = info["u0"]
    Jxy = info["Jxy"]
    X_pred = info["X_pred"]
    U_seq = info["U_seq"]

    print(f"   p_cmd: gamma={np.rad2deg(p_cmd[0]):+.1f}deg, beta={np.rad2deg(p_cmd[1]):+.1f}deg, "
          f"rho={p_cmd[2]:.3f}m, L={p_cmd[3]:.3f}m")

    print(f"   u0: gdot={np.rad2deg(u0[0]):+.2f}deg/s, bdot={np.rad2deg(u0[1]):+.2f}deg/s, "
          f"rhodot={u0[2]:+.4f}m/s, Ldot={u0[3]:+.4f}m/s")

    print("   Jxy:")
    print(Jxy)

    ph = min(print_horizon, X_pred.shape[0])
    if np.all(np.isfinite(X_pred)):
        for i in range(ph):
            print(f"   pred[{i}]={X_pred[i]}   ref[{i}]={xref_seq[i]}")
    else:
        print("   X_pred NaNs -> infeasible QP")

    print()
p0 = np.array([np.deg2rad(10), np.deg2rad(5), 0.15, 0.05])
mpc_xy.set_initial_params(*p0)
print("Initial tip:", mpc_xy.x)
# p_target = np.array([np.deg2rad(5), np.deg2rad(0), 0.12, 0.06])
x_target = forward_tip([np.deg2rad(40), np.deg2rad(0), 0.12, 0.06])
# for k in range(3):
#     x_now = mpc_xy.x.copy()
#     xref_seq = np.tile(x_target, (mpc_xy.Np, 1))

#     p_cmd, x_now, info = mpc_xy.step(xref_seq)

#     debug_step_tipxy(k, x_target, xref_seq, p_cmd, x_now, info, print_horizon=4)
# print(f"Final tip position is: [{x_now[0]:+.4f}, {x_now[1]:+.4f}] with target: {x_target}")
# theta_y_tip = np.arctan2(x_now[1], x_now[0])
# theta_y_target = np.arctan2(x_target[1], x_target[0])
# print(f"Angle is: {np.rad2deg(theta_y_tip)}")
# print(f"Angle is of target: {np.rad2deg(theta_y_target)}")
def path_sinusoid_x(x_start, x_target, M=80, amp=0.02, cycles=1.0):
    """
    Make waypoints where y moves from start->target,
    and x follows a sinusoid around the straight-line baseline.
    """
    x_start = np.asarray(x_start).ravel()
    x_target = np.asarray(x_target).ravel()

    t = np.linspace(0.0, 1.0, M)

    base = (1 - t)[:, None] * x_start[None, :] + t[:, None] * x_target[None, :]

    x_offset = amp * np.sin(2.0 * np.pi * cycles * t)

    path = base.copy()
    path[:, 0] += x_offset
    return path
def closest_point_error(path, x):
    """
    Returns:
      dmin   : minimum distance from x to the waypoint set
      x_star : closest waypoint point
      i_star : index of closest waypoint
    """
    diffs = path - x[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    i_star = int(np.argmin(dists))
    return float(dists[i_star]), path[i_star].copy(), i_star
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Path (sinusoid horizontally)
# ----------------------------
M = 80
x_start = mpc_xy.x.copy()
path = path_sinusoid_x(x_start, x_target, M=M, amp=0.006, cycles=1.0)
idx = 0

# ----------------------------
# helper: closest point on path
# ----------------------------
def closest_point_error(path, x):
    diffs = path - x[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    i_star = int(np.argmin(dists))
    return float(dists[i_star]), path[i_star].copy(), i_star


# ----------------------------
# Post-analysis logger (NEW MPC)
# ----------------------------
log = {
    "k": [],
    "idx": [],

    "x_now_prev": [],
    "x_now": [],
    "x_ref0": [],
    "xref_seq": [],

    "p_cmd": [],
    "u0": [],
    "U_seq": [],

    "X_pred": [],
    "status": [],
    "infeasible": [],

    # offset-free / integral action
    "d_hat": [],

    # tracking metrics
    "track_err_ref0_mm": [],   # ||x_ref0 - x_now|| in mm
    "path_err_mm": [],         # closest distance to entire path in mm
    "i_closest": [],

    # prediction check (1-step horizon sanity)
    "pred1_err_mm": [],        # ||X_pred[0] - x_{k+1}|| (mm)
}

K_sim = 50

for k in range(K_sim):
    x_now_prev = mpc_xy.x.copy()

    # horizon reference window
    xref_seq = path[idx : idx + mpc_xy.Np]
    if xref_seq.shape[0] < mpc_xy.Np:
        last = path[-1]
        pad = np.tile(last, (mpc_xy.Np - xref_seq.shape[0], 1))
        xref_seq = np.vstack([xref_seq, pad])

    # --- MPC step (SQP + offset-free) ---
    p_cmd, x_now, info = mpc_xy.step(xref_seq)

    # closest path error (global)
    dmin, x_closest, i_closest = closest_point_error(path, x_now)

    # tracking error to current reference point (ref[0])
    x_ref0 = xref_seq[0]
    e_ref0 = x_ref0 - x_now

    # horizon 1-step prediction sanity:
    # X_pred[0] is predicted next-step state (for step k+1)
    if info["X_pred"] is not None and np.all(np.isfinite(info["X_pred"])) and info["X_pred"].shape[0] > 0:
        pred1 = info["X_pred"][0]
        pred1_err = np.linalg.norm(pred1 - x_now)  # after update (approx check)
    else:
        pred1_err = np.nan

    # advance waypoint index (same as your logic)
    if np.linalg.norm(x_now - path[idx]) < 1e-3:
        idx = min(idx + 1, M - 1)

    # ----------------------------
    # Log everything
    # ----------------------------
    log["k"].append(k)
    log["idx"].append(idx)

    log["x_now_prev"].append(x_now_prev.copy())
    log["x_now"].append(x_now.copy())
    log["x_ref0"].append(x_ref0.copy())
    log["xref_seq"].append(xref_seq.copy())

    log["p_cmd"].append(p_cmd.copy())
    log["u0"].append(info["u0"].copy())
    log["U_seq"].append(info["U_seq"].copy())

    log["X_pred"].append(info["X_pred"].copy())
    log["status"].append(info["status"])
    log["infeasible"].append(info["infeasible"])

    log["d_hat"].append(info.get("d", np.zeros(2)).copy())

    log["track_err_ref0_mm"].append(1e3 * np.linalg.norm(e_ref0))
    log["path_err_mm"].append(1e3 * dmin)
    log["i_closest"].append(i_closest)

    log["pred1_err_mm"].append(1e3 * pred1_err)

    # debug print: error to current reference waypoint (not end target)
    # debug_step_tipxy(k, x_ref0, xref_seq, p_cmd, x_now, info, print_horizon=6)

print("Final tip:", x_now, "Final ref:", path[-1], "End target:", x_target)
# ----------------------------
# Convert logs to arrays
# ----------------------------
x_now_arr = np.array(log["x_now"])
x_ref0_arr = np.array(log["x_ref0"])

track_err_mm = np.array(log["track_err_ref0_mm"])
path_err_mm  = np.array(log["path_err_mm"])
pred1_err_mm = np.array(log["pred1_err_mm"])

d_hat = np.array(log["d_hat"])  # (T,2)


# ----------------------------
# 1) Trajectory vs path
# ----------------------------
plt.figure()
plt.plot(path[:, 0], path[:, 1], "--", label="path")
plt.plot(x_now_arr[:, 0], x_now_arr[:, 1], "-o", label="actual tip")
plt.plot(x_ref0_arr[:, 0], x_ref0_arr[:, 1], "-x", label="ref[0] each step")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Tip trajectory vs sinusoidal waypoint path")
plt.grid(True)
plt.legend()
plt.tight_layout()


# ----------------------------
# 2) Tracking error to current waypoint (mm)
# ----------------------------
plt.figure()
plt.plot(track_err_mm, "-o")
plt.xlabel("MPC step k")
plt.ylabel("||x_ref0 - x_now|| (mm)")
plt.title("Tracking error to horizon reference point ref[0]")
plt.grid(True)
plt.tight_layout()


# ----------------------------
# 3) Closest distance to entire path (mm)
# ----------------------------
plt.figure()
plt.plot(path_err_mm, "-o")
plt.xlabel("MPC step k")
plt.ylabel("Min distance to path (mm)")
plt.title("Path tracking error (closest waypoint on full path)")
plt.grid(True)
plt.tight_layout()


# ----------------------------
# 4) 1-step prediction sanity check (mm)
# ----------------------------
plt.figure()
plt.plot(pred1_err_mm, "-o")
plt.xlabel("MPC step k")
plt.ylabel("||X_pred[0] - x_now|| (mm)")
plt.title("Horizon prediction sanity (first predicted point vs achieved)")
plt.grid(True)
plt.tight_layout()


# ----------------------------
# 5) Disturbance estimate d_hat (integral action)
# ----------------------------
plt.figure()
plt.plot(1e3 * d_hat[:, 0], "-o", label="d_x (mm)")
plt.plot(1e3 * d_hat[:, 1], "-o", label="d_y (mm)")
plt.xlabel("MPC step k")
plt.ylabel("Estimated bias d (mm)")
plt.title("Offset-free MPC disturbance estimate (integral effect)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()