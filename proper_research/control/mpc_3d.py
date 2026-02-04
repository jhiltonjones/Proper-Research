import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.parameters import default_magnet_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T, T_to_p_quat_wxyz
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import unpack_pose_ur_rotvec_L
from beam_direction_magnetisation.post_processing.post_processing import plot_mpc_state_3d
from scipy.spatial.transform import Rotation as Rot
mag_params = default_magnet_params()




u_max = np.array([ .05, .05, .05, np.deg2rad(60), np.deg2rad(60), np.deg2rad(60),  0.02])
eps = np.array([
    1e-3, 1e-3, 1e-3,              # x,y,z
    np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  
    5e-4                              # L
], dtype=float)


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
                 w_xy=(50.0, 50.0, 50.0),
                 w_u=None,
                 w_du=None,
                 band_xy=0.0,
                 u_max=None,
                 p_min=None,
                 p_max=None,
                 N_sqp=3,
                 use_offset_free=True,
                 n_out=3,
                 n_u=7,
                 model_mode = "ltv",
                 d_min_tip_mag=0.10,  
                 enable_tip_keepout=True,
                 d_alpha=0.15):

        self.n = int(n_out)
        self.m = int(n_u)

        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

        self.dt = float(dt)
        self.Np = int(Np)

        self.A = np.eye(self.n)

        # tube constraint 
        # tube constraint 
        self.band_xy = float(band_xy)

        self.U_warm = None
        self.model_mode = str(model_mode).lower()
        if self.model_mode not in ("lti", "ltv"):
            raise ValueError("model_mode must be 'lti' or 'ltv'")
        self.model_mode = str(model_mode).lower()
        if self.model_mode not in ("lti", "ltv"):
            raise ValueError("model_mode must be 'lti' or 'ltv'")
        # weights defaults
        if w_u is None:
            w_u = (1e-3,) * self.m
        if w_du is None:
            w_du = (0.0,) * self.m

        w_xy = np.asarray(w_xy, float).ravel()
        w_u  = np.asarray(w_u,  float).ravel()
        w_du = np.asarray(w_du, float).ravel()

        if w_xy.size != self.n:
            raise ValueError(f"w_xy must have length {self.n}, got {w_xy.size}")
        if w_u.size != self.m:
            raise ValueError(f"w_u must have length {self.m}, got {w_u.size}")
        if w_du.size != self.m:
            raise ValueError(f"w_du must have length {self.m}, got {w_du.size}")

        self.Q  = np.diag(w_xy)
        self.R  = np.diag(w_u)
        self.Rd = np.diag(w_du)

        if u_max is None:
            u_max = np.full(self.m, np.inf)
        if p_min is None:
            p_min = -np.full(self.m, np.inf)
        if p_max is None:
            p_max = +np.full(self.m, np.inf)

        self.u_max = np.asarray(u_max, float).ravel()
        self.p_min = np.asarray(p_min, float).ravel()
        self.p_max = np.asarray(p_max, float).ravel()

        if self.u_max.size != self.m: raise ValueError("u_max wrong length")
        if self.p_min.size != self.m: raise ValueError("p_min wrong length")
        if self.p_max.size != self.m: raise ValueError("p_max wrong length")

        self.d  = np.zeros(self.n, float)
        self.p  = None
        self.x  = None
        self.Qf = None

        self.N_sqp = int(N_sqp)
        self.use_offset_free = bool(use_offset_free)
        self.d_alpha = float(d_alpha)

        # matrices
        self._rebuild_S()
        self.Du = self._build_Du_matrix()
        self.d_min_tip_mag = float(d_min_tip_mag)
        self.enable_tip_keepout = bool(enable_tip_keepout)

        self.Sel_pos = np.zeros((3, self.m))
        self.Sel_pos[0, 0] = 1.0
        self.Sel_pos[1, 1] = 1.0
        self.Sel_pos[2, 2] = 1.0
    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt
    def _build_tip_keepout_constraints(self, *, Mc, X0_stack, p0, U_guess_vec):
        """
        Build linearized constraints enforcing ||r_i - x_i|| >= dmin for i=1..Np
        using SQP linearization around current predicted (r0_i, x0_i).

        Returns:
          A_ko: (Np, Np*m)
          l_ko: (Np,)
          u_ko: (Np,)
        """
        if not self.enable_tip_keepout:
            return None, None, None

        dmin = self.d_min_tip_mag
        n, m, Np = self.n, self.m, self.Np

        # Map U -> stacked p along horizon (same as you used for bounds)
        A_p = np.kron(self.S_np, np.eye(m))          # (Np*m, Np*m)
        p0_stack = np.tile(p0, Np)                   # (Np*m,)

        # stacked magnet positions: r_stack = (I kron Sel_pos) (p0_stack + A_p U)
        Spos = np.kron(np.eye(Np), self.Sel_pos)     # (Np*3, Np*m)
        r0_stack_base = Spos @ p0_stack.reshape(-1, 1)   # (Np*3,1)
        Rmap = Spos @ A_p                            # (Np*3, Np*m)

        X_guess = X0_stack + Mc @ U_guess_vec.reshape(-1, 1)

        # predicted p along horizon under current guess:
        p_guess = p0_stack.reshape(-1, 1) + A_p @ U_guess_vec.reshape(-1, 1)  # (Np*m,1)
        r_guess = (Spos @ p_guess).reshape(Np, 3)                              # (Np,3)

        # Extract x_guess per step
        x_guess = X_guess.reshape(Np, n)[:, :3]  
        x_guess = X_guess.reshape(Np, n)[:, :3]  

        # Build one linear constraint per horizon step
        A_rows = []
        l_rows = []
        u_rows = []

        for i in range(Np):
            r0 = r_guess[i, :].reshape(3, 1)
            x0 = x_guess[i, :].reshape(3, 1)
            s0 = (r0 - x0).reshape(3, 1)

            s0n = float(np.linalg.norm(s0))
            # If s0 is extremely small, the linearization direction is ill-defined.
            # In that case, skip (or you can pick a fixed direction).
            if s0n < 1e-8:
                continue

            # Blocks mapping U -> r_i and U -> x_i
            Rmap_i = Rmap[i*3:(i+1)*3, :]                  # (3, Np*m)
            Mc_i   = Mc[i*n:(i+1)*n, :]                    # (n, Np*m)
            Mc_i3  = Mc_i[:3, :]                           # (3, Np*m)

            # r_i - x_i = (r_base_i - X0_i) + (Rmap_i - Mc_i3) U
            X0_i = X0_stack[i*n:(i+1)*n, :][:3, :]         # (3,1)
            r_base_i = r0_stack_base[i*3:(i+1)*3, :]       # (3,1)

            # Linearized constraint: 2 s0^T (r_i - x_i) >= dmin^2 + ||s0||^2
            # => a_i U >= b_i
            a_i = (2.0 * s0.T) @ (Rmap_i - Mc_i3)          # (1, Np*m)
            b_i = (dmin**2 + (s0n**2)) - float((2.0 * s0.T) @ (r_base_i - X0_i))

            A_rows.append(a_i.reshape(1, -1))
            l_rows.append(b_i)
            u_rows.append(np.inf)

        if not A_rows:
            return None, None, None

        A_ko = np.vstack(A_rows)
        l_ko = np.asarray(l_rows, dtype=float)
        u_ko = np.asarray(u_rows, dtype=float)
        return A_ko, l_ko, u_ko
    def _build_Du_matrix(self):
        Np = self.Np
        m = self.m
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

    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.m,)
        self.x = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
        self.d = np.zeros(self.n, float)
        self.U_warm = None

    def _compute_Qtil(self, B0):
        _, P = dare_stabilising_K(self.A, B0, self.Q, self.R)
        self.Qf = P

        n = self.n
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
        n = self.n
        Np = self.Np
        d = np.asarray(d, dtype=float).reshape(n,)

        d_stack = np.zeros((Np*n, 1))
        d_acc = np.zeros(n)

        for i in range(Np):
            d_acc = self.A @ d_acc + d
            d_stack[i*n:(i+1)*n, 0] = d_acc

        return d_stack



    def _clamp_p(self, p):
        p = np.minimum(np.maximum(p, self.p_min), self.p_max)
        # optional: keep a canonical rotvec representation for continuity
        # p[3:6] = canonicalize_rotvec(p[3:6])
        # optional: keep a canonical rotvec representation for continuity
        # p[3:6] = canonicalize_rotvec(p[3:6])
        return p



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

    # def _build_ltv_prediction_mats(self, p0, U_guess):
    #     """
    #     Build LTV (time-varying) B_i = dt * J(p_i) along horizon based on U_guess.
    #     """
    #     if U_guess is None:
    #         U_guess = np.zeros((self.Np, self.m))

    #     p_seq = self._p_seq_from_U(p0, U_guess)  # (Np,4), these are p1..pNp
    #     B_list = []
    #     J_list = []
    #     for i in range(self.Np):
    #         Ji = np.asarray(self.Jxy_fn(p_seq[i]), dtype=float)
    #         if Ji.shape != (self.n, self.m):
    #             raise ValueError(f"Jacobian must be {(self.n, self.m)} but got {Ji.shape}")
    #         J_list.append(Ji)
    #         B_list.append(self.dt * Ji)

    #     Mx, Mc = seq_mat_ltv(self.A, B_list)
    #     return p_seq, J_list, B_list, Mx, Mc
    def _build_prediction_mats(self, p0, U_guess):
        """
        Returns: p_seq, Mx, Mc, B0
        - p_seq: (Np,m) predicted p (only meaningful in LTV; in LTI it's still returned for convenience)
        - Mx, Mc: stacked prediction matrices
        - B0: first-step input matrix (for terminal cost DARE)
        """
        n, m, Np = self.n, self.m, self.Np

        if self.model_mode == "lti":
            # Constant Jacobian at current p0
            J0 = np.asarray(self.Jxy_fn(p0), dtype=float)
            if J0.shape != (n, m):
                raise ValueError(f"Jacobian must be {(n,m)} but got {J0.shape}")
            B0 = self.dt * J0

            Mx, Mc = seq_mat_lti(self.A, B0, Np)

            # p_seq only used for debugging / keepout linearization; compute by integrating guess (optional)
            if U_guess is None:
                U_guess = np.zeros((Np, m))
            p_seq = self._p_seq_from_U(p0, U_guess)

            return p_seq, Mx, Mc, B0

        # --- LTV ---
    # def _build_ltv_prediction_mats(self, p0, U_guess):
    #     """
    #     Build LTV (time-varying) B_i = dt * J(p_i) along horizon based on U_guess.
    #     """
    #     if U_guess is None:
    #         U_guess = np.zeros((self.Np, self.m))

    #     p_seq = self._p_seq_from_U(p0, U_guess)  # (Np,4), these are p1..pNp
    #     B_list = []
    #     J_list = []
    #     for i in range(self.Np):
    #         Ji = np.asarray(self.Jxy_fn(p_seq[i]), dtype=float)
    #         if Ji.shape != (self.n, self.m):
    #             raise ValueError(f"Jacobian must be {(self.n, self.m)} but got {Ji.shape}")
    #         J_list.append(Ji)
    #         B_list.append(self.dt * Ji)

    #     Mx, Mc = seq_mat_ltv(self.A, B_list)
    #     return p_seq, J_list, B_list, Mx, Mc
    def _build_prediction_mats(self, p0, U_guess):
        """
        Returns: p_seq, Mx, Mc, B0
        - p_seq: (Np,m) predicted p (only meaningful in LTV; in LTI it's still returned for convenience)
        - Mx, Mc: stacked prediction matrices
        - B0: first-step input matrix (for terminal cost DARE)
        """
        n, m, Np = self.n, self.m, self.Np

        if self.model_mode == "lti":
            # Constant Jacobian at current p0
            J0 = np.asarray(self.Jxy_fn(p0), dtype=float)
            if J0.shape != (n, m):
                raise ValueError(f"Jacobian must be {(n,m)} but got {J0.shape}")
            B0 = self.dt * J0

            Mx, Mc = seq_mat_lti(self.A, B0, Np)

            # p_seq only used for debugging / keepout linearization; compute by integrating guess (optional)
            if U_guess is None:
                U_guess = np.zeros((Np, m))
            p_seq = self._p_seq_from_U(p0, U_guess)

            return p_seq, Mx, Mc, B0

        # --- LTV ---
        if U_guess is None:
            U_guess = np.zeros((Np, m))

        p_seq = self._p_seq_from_U(p0, U_guess)  # p1..pNp
        B_list = []
        for i in range(Np):
            Ji = np.asarray(self.Jxy_fn(p_seq[i]), dtype=float)
            if Ji.shape != (n, m):
                raise ValueError(f"Jacobian must be {(n,m)} but got {Ji.shape}")
            B_list.append(self.dt * Ji)

        Mx, Mc = seq_mat_ltv(self.A, B_list)
        B0 = B_list[0]
        return p_seq, Mx, Mc, B0


    def step(self, xref_seq, x_meas=None):
        """
        xref_seq: (Np,2)
        x_meas: measured tip (2,) for offset-free update. If None, uses internal x.
        """

        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        p_prev = self.p.copy()
        x_prev = self.x.copy()
        # measurement update
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(self.n,)

        # --- offset-free disturbance update ---
        # Use model residual to update d (low-pass filtered)
        if self.use_offset_free and x_meas is not None:
            x_model = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
            r = self.x - x_model
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        n = self.n
        m = self.m
        Np = self.Np
        Mc_last = None
        X_aff_last = None
        p_seq_last = None
        X_nom_last = None

        xref_seq = np.asarray(xref_seq, float).reshape(Np, self.n)
        xref_stack = xref_seq.reshape(Np*self.n, 1)
        xk = self.x.reshape(self.n, 1)

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
            p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)

            B_first = B0         
            p_lin   = p_prev.copy()
            p_first = p_seq[0].copy()

            Qtil = self._compute_Qtil(B0)
        
            
            Rtil = np.kron(np.eye(Np), self.R)

            # delta-u penalty
            if self.Np > 1 and np.any(np.diag(self.Rd) > 0):
                Rd_til = np.kron(np.eye(Np-1), self.Rd)
                H_du = self.Du.T @ Rd_til @ self.Du
            else:
                H_du = 0.0

            xk = self.x.reshape(self.n, 1)

            # baseline predicted state stack (no control)
            # X0_stack = (Mx @ xk).reshape(Np*n, 1)

            # add disturbance contribution (offset-free)
            if self.use_offset_free:
                X0_stack = X0_stack + self._disturbance_stack(self.d)

            # objective
            U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)

            # Nominal nonlinear rollout along p_seq
            X_nom = np.vstack([self.forward_tip_fn(p_seq[i]) for i in range(Np)]).reshape(Np*n, 1)

            # Optional offset-free disturbance: add it consistently to nominal and prediction
            if self.use_offset_free:
                X_nom = X_nom + self._disturbance_stack(self.d)

            # Affine offset so linear model matches nominal at U_guess
            X_aff = X_nom - Mc @ U_guess_vec
            Mc_last = Mc
            X_aff_last = X_aff
            p_seq_last = p_seq
            X_nom_last = X_nom
            # Now prediction is X_aff + Mc U
            H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
            f = 2.0 * (Mc.T @ Qtil @ (X_aff - xref_stack))

            # --- tip keep-out constraint (magnet must stay >= dmin from tip) ---
            A_list, l_list, u_list = [], [], []
            if self.enable_tip_keepout:
                # Current iterate as vector
                U_guess_vec = U_guess.reshape(-1)

                A_ko, l_ko, u_ko = self._build_tip_keepout_constraints(
                    Mc=Mc,
                    X0_stack=X_aff,
                    p0=self.p.copy(),
                    U_guess_vec=U_guess_vec
                )
                if A_ko is not None:
                    A_list.append(A_ko)
                    l_list.append(l_ko)
                    u_list.append(u_ko)
            # constraints


            # (1) input bounds
            if np.all(np.isfinite(self.u_max)):
                A_u = np.eye(Np * self.m)
                umax_stack = np.tile(self.u_max, Np)
                A_list.append(A_u)
                l_list.append(-umax_stack)
                u_list.append(+umax_stack)

            # (2) parameter bounds across horizon (via integrated controls)
            if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
                A_p = np.kron(self.S_np, np.eye(self.m))
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
                rhs_p = band_stack + (xref_stack - X_aff).reshape(-1)
                rhs_n = band_stack + (X_aff - xref_stack).reshape(-1)


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

        if U_opt_vec is None:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
            X_pred = np.full((Np, n), np.nan)
            infeas_final = True
        else:
            U_seq = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]

            # Use SQP-consistent prediction baseline from the final linearization
            if (Mc_last is None) or (X_aff_last is None):
                X_pred = np.full((Np, n), np.nan)
                infeas_final = True
            else:
                U_vec = U_opt_vec.reshape(-1, 1)                 # (Np*m,1)
                X_pred_stack = X_aff_last + Mc_last @ U_vec      # (Np*n,1)
                X_pred = X_pred_stack.reshape(Np, n)
                infeas_final = False

                p_next_true = self._clamp_p(p_prev + self.dt*u0)
                x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
        p_next_true = self._clamp_p(p_prev + self.dt*u0)
        x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
        if self.use_offset_free:
            x_next_true = x_next_true + self.d 
        # update parameter state
        self.p = self._clamp_p(self.p + u0 * self.dt)

        # update plant state using nonlinear forward model
        self.x = self.forward_tip_fn(self.p)
        print(f"Pose inside mpc step is {self.p}")
        print(f"X inside mpc step is {self.x}")
        print(f"Pose inside mpc step is {self.p}")
        print(f"X inside mpc step is {self.x}")
        # store warm-start for next MPC step
        if not infeas_final:
            self.U_warm = np.asarray(U_opt_vec, dtype=float).copy()
        else:
            self.U_warm = None
        pred1_err = np.linalg.norm(x_next_true - X_pred[0])

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
            pred1_err = pred1_err,
            x_prev = x_prev.copy(),
            x_next_true = x_next_true.copy(),
            X_pred0 = X_pred[0].copy(),
            pred0_vec_err = (x_next_true - X_pred[0]).copy(),
            B_first=B_first.copy(),
            p_lin=p_lin,
            p_first=p_first,
            X_aff_last = X_aff_last.copy() if X_aff_last is not None else None,
            Mc_last    = Mc_last.copy()    if Mc_last is not None else None,
            X_nom_last = X_nom_last.reshape(Np, n).copy() if X_nom_last is not None else None,
            p_seq_last = p_seq_last.copy() if p_seq_last is not None else None,
        )
        return self.p.copy(), self.x.copy(), info


# def forward_cosserat_from_pose_ur_rotvec_L(p, model, *, m_body):
#     r_src, q_src, L = unpack_pose_ur_rotvec_L(p)
def forward_cosserat_from_pose_ur_rotvec_L(p, model, *, m_body):
    r_src, q_src, L = unpack_pose_ur_rotvec_L(p)
    out = model.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body)
    if not out["solved"]:
        return np.array([1e3, 1e3, 1e3], float)
    return np.asarray(out["p_tip"], float).reshape(3,)




def numerical_jacobian_tip_xyz_pose(p, forward_fn, eps):
    p = np.asarray(p, float).ravel()
    eps = np.asarray(eps, float).ravel()
    assert eps.size == p.size

    n_out = 3
    J = np.zeros((n_out, p.size), float)
    for i in range(p.size):
        dp = np.zeros_like(p)
        dp[i] = eps[i]
        xp = np.asarray(forward_fn(p + dp), float).reshape(n_out,)
        xm = np.asarray(forward_fn(p - dp), float).reshape(n_out,)
        J[:, i] = (xp - xm) / (2.0 * eps[i])
    return J

def rollout_open_loop_from_plan(mpc, p_start, U_seq):
    """
    Roll forward the nonlinear plant using the planned control sequence U_seq (Np,m).
    Returns:
      P_nl: (Np, m) parameter trajectory (p1..pNp)
      X_nl: (Np, n) tip trajectory
    """
    p = p_start.copy()
    P_nl = []
    X_nl = []
    for i in range(U_seq.shape[0]):
        p = mpc._clamp_p(p + mpc.dt * U_seq[i])
        x = np.asarray(mpc.forward_tip_fn(p), float).reshape(mpc.n,)
        P_nl.append(p.copy())
        X_nl.append(x.copy())
    return np.vstack(P_nl), np.vstack(X_nl)
def debug_step_pose7(k, x_target, xref_seq, p_now, x_now, info, mpc,
                     print_horizon=4, do_nl_rollout=True):
    e = x_target - x_now
    err_mm = 1e3 * np.linalg.norm(e)

    print(f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f},{x_now[2]:+.4f}] "
          f"||e||={err_mm:.2f}mm status={info['status']} infeas={info['infeasible']}")

    u0 = info["u0"]
    X_pred = info.get("X_pred", None)   # (Np,3)
    U_seq  = info.get("U_seq", None)    # (Np,7)
    pred_err = info.get("pred1_err", np.nan)

    # Pose print
    r_src, q_src, L = unpack_pose_ur_rotvec_L(p_now)
    rvec = p_now[3:6]
    theta_deg = np.rad2deg(np.linalg.norm(rvec))

    print(f"   p_now: x={p_now[0]:+.3f} y={p_now[1]:+.3f} z={p_now[2]:+.3f}  "
          f"rotvec=[{rvec[0]:+.3f},{rvec[1]:+.3f},{rvec[2]:+.3f}] |theta|={theta_deg:.1f}deg  "
          f"L={p_now[6]:.3f}")

    print("   u0:",
          f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
          f"d_rotvec=[{u0[3]:+.4f},{u0[4]:+.4f},{u0[5]:+.4f}]  dL={u0[6]:+.5f}")

    print(f"   pred1_err (one-step tip mismatch): {pred_err}")

    # ---- internal consistency check: should be ~0 ----
    Mc_last = info.get("Mc_last", None)
    X_aff_last = info.get("X_aff_last", None)
    if (Mc_last is not None) and (X_aff_last is not None) and (U_seq is not None):
        n = mpc.n
        U_vec = U_seq.reshape(-1, 1)          # (Np*m,1)
        Mc0 = Mc_last[0:n, :]                 # (n, Np*m)
        x1_from_blocks = (X_aff_last[0:n, :] + Mc0 @ U_vec).reshape(n,)
        if X_pred is not None and np.all(np.isfinite(X_pred)):
            print("   |(X_aff0 + Mc0U) - X_pred[0]| =",
                  np.linalg.norm(x1_from_blocks - X_pred[0]))

    # ---- horizon print: pred vs ref (and optional nominal nonlinear) ----
    xref_seq = np.asarray(xref_seq, float)   # (Np,3)
    Np = xref_seq.shape[0]
    ph = min(print_horizon, Np)

    X_nom_last = info.get("X_nom_last", None)   # optional (Np,3)

    if X_pred is None or not np.all(np.isfinite(X_pred)):
        print("   X_pred NaNs or missing -> infeasible/failed QP")
    else:
        print("   Horizon:  pred (linear)              ref")
        for i in range(ph):
            print(f"   {i:02d}: {X_pred[i]}   {xref_seq[i]}")

        if X_nom_last is not None and np.all(np.isfinite(X_nom_last)):
            print("   Horizon:  nominal (nonlinear @ SQP lin point)")
            for i in range(ph):
                print(f"   {i:02d}: {X_nom_last[i]}")

    # ---- nonlinear open-loop rollout under U_seq ----
    if do_nl_rollout and (U_seq is not None) and np.all(np.isfinite(U_seq)):
        P_nl, X_nl = rollout_open_loop_from_plan(mpc, p_now.copy(), U_seq)
        print("   Horizon:  NL rollout (true f(p) along applied U_seq)")
        for i in range(min(ph, X_nl.shape[0])):
            if X_pred is not None and np.all(np.isfinite(X_pred)):
                print(f"   {i:02d}: {X_nl[i]}   (lin {X_pred[i]})")
            else:
                print(f"   {i:02d}: {X_nl[i]}")

    print("--------------------------------------------------------------------")

def build_xref_from_path(path, k, Np):
    """
    path: (N,3)
    returns xref_seq: (Np,3) with look-ahead
    """
    N = path.shape[0]
    idx = np.clip(np.arange(k, k + Np), 0, N - 1)
    return path[idx]


pivot_point = np.array([
0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)


start_point = np.array([
0.6781328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)
T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)


model = CosseratForwardModel(
    p0=p0_ur,
    q0=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    m_local_fun=make_m_local_fun_wire_tip(mode="axial", alpha_end=0.0),
    m_moment=0.0,
)

# model = CosseratForwardModel(
#     p0=np.array([0.0, 0.0, 0.0]),
#     q0=np.array([1.0, 0.0, 0.0, 0.0]),
#     Kinv_fun=Kbt_inv_profile,
#     m_local_fun=make_m_local_fun_wire_tip(mode="axial", alpha_end=0.0),
#     m_moment=0.0,  # not used by this m_local_fun
#     n_nodes=120,
#     tol=1e-5    
#     )
m_body = np.array([mag_params.mag_epm*2, 0.0, 0.0], dtype=float)
L0 = 0.05      
forward_tip_fn = lambda p: forward_cosserat_from_pose_ur_rotvec_L(p, model, m_body=m_body)
J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_fn, eps)
# p0 = np.array([ 0.63360145, -0.56541852,  0.20705173, -3.116988654350607, 0.19059356279735162, 0.028215660130034903,L0])
start_point_pose6 = start_point  # [x,y,z, rx,ry,rz]
p0 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
               start_point_pose6[3], start_point_pose6[4], start_point_pose6[5],
               L0], float)
p_min = np.array([ 0.2, -1, -0.2,  p0[3]-np.pi*2, p0[4]-np.pi*2, p0[5]-np.pi*2,  0.03])
p_max = np.array([ 0.85,  1,  1.0,  p0[3]+np.pi*2, p0[4]+np.pi*2, p0[5]+np.pi*2,  0.08])
w_u  = np.array([1e-6, 1e-6, 1e-3,   1e-6, 1e-6, 1e-6,   1e-6])
w_du = np.array([1e-6, 1e-6, 1e-6,   1e-3, 1e-3, 1e-3,   1e-6])

mpc = mpc_controller_tipxy_LTI(
    Jxy_fn=J_fn,
    forward_tip_fn=forward_tip_fn,
    dt=0.1,
    Np=4,
    n_out=3,
    n_u=7,
    w_xy=(1, 1, 1),
    w_u=w_u,
    w_du=w_du,
    model_mode="ltv",
    u_max=u_max,
    p_min=p_min,
    p_max=p_max,
    N_sqp=4,
    use_offset_free=False
)


mpc.set_initial_params(p0)
x_target = np.array([ 0.80936455, -0.70098197, -0.10018981])
Np = mpc.Np
xref_seq = np.tile(x_target, (Np, 1))     # (Np,3)
p_test = p0.copy()
J_test = J_fn(p_test)
print("J shape:", J_test.shape)  # must be (3,7)
x_start = mpc.x.copy()
n=5
path = np.linspace(x_start, x_target, n)
for k in range(10):
    p_pre = mpc.p.copy()
    x_pre = mpc.x.copy()

    xref_seq = build_xref_from_path(path, k, mpc.Np)    
    p_post, x_post, info = mpc.step(xref_seq, x_meas=None)

    debug_step_pose7(k, x_target, xref_seq, p_post, x_post, info, mpc,
                     print_horizon=mpc.Np, do_nl_rollout=True)


    error = np.linalg.norm(x_target-x_post)
    if error < 0.001:
        plot_mpc_state_3d(model, m_body, p_post, x_post, title=f"MPC step {k}", dipole_scale=0.05)
        print("UR pose6:", repr(p_post))
        break
print("UR pose6:", repr(p_post))
plot_mpc_state_3d(model, m_body, p_post, x_post, title=f"MPC step {k}", dipole_scale=0.05)

    
    
print("UR pose6:", repr(p_post))

    
    
