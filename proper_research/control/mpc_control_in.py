import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.models.forward_model import make_forward_fn, make_jac_fn
from proper_research.parameters import default_magnet_params

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

