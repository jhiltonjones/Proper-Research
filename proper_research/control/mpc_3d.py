import numpy as np
import osqp
import copy
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import unpack_pose_ur_rotvec_L
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
from proper_research.simulation.boundary_forward_model import EnergyMinForwardWithLumen
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
mag_params = default_magnet_params()
beam_params = default_beam_params()
L_MAG = 0.04



def wire_len_from_L(L, L_mag=L_MAG):
    L = float(L)
    wire_len = L - float(L_mag)
    # if wire_len <= 0:
    #     # if beam shorter than magnet segment, clamp or error
    #     raise ValueError(f"L={L:.4f} must be > L_mag={L_mag:.4f}")
    return wire_len

u_max = np.array([ .2, .2, .2, np.deg2rad(60), np.deg2rad(60), np.deg2rad(60),  0.02])
eps = np.array([
    1e-2, 1e-2, 1e-2,              # x,y,z
    np.deg2rad(1), np.deg2rad(1), np.deg2rad(1),  
    5e-4                              # L
], dtype=float)

import numpy as np
from numpy.linalg import LinAlgError


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
                 enable_tip_keepout=False,
                 d_alpha=0.15,
                 ):

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
        self.d_max = np.array([0.03, 0.03, 0.03])  # meters, tune
        self.r_gate = 0.05                         # meters, tune

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
        self.use_adaptive_J = False
        self.G = np.zeros((self.n, self.m))
        self.G_alpha = 0.1          # learning rate (start 0.05–0.3)
        self.G_max = 5.0            # cap each element (units: output per param)
        self.min_dp = 1e-5          # ignore tiny moves
        self.G_beta = 0.2     # how much of G to apply in the MPC model

        # matrices
        self._rebuild_S()
        self.Du = self._build_Du_matrix()
        self.d_min_tip_mag = float(d_min_tip_mag)
        self.enable_tip_keepout = bool(enable_tip_keepout)
        self.p_last_meas = None
        self.x_last_meas = None

        self.Sel_pos = np.zeros((3, self.m))
        self.Sel_pos[0, 0] = 1.0
        self.Sel_pos[1, 1] = 1.0
        self.Sel_pos[2, 2] = 1.0
    def _rebuild_S(self):
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt
    # def _build_tip_keepout_constraints(self, *, Mc, X0_stack, p0, U_guess_vec):
    #     """
    #     Build linearized constraints enforcing ||r_i - x_i|| >= dmin for i=1..Np
    #     using SQP linearization around current predicted (r0_i, x0_i).

    #     Returns:
    #       A_ko: (Np, Np*m)
    #       l_ko: (Np,)
    #       u_ko: (Np,)
    #     """
    #     if not self.enable_tip_keepout:
    #         return None, None, None

    #     dmin = self.d_min_tip_mag
    #     n, m, Np = self.n, self.m, self.Np

    #     # Map U -> stacked p along horizon (same as you used for bounds)
    #     A_p = np.kron(self.S_np, np.eye(m))          # (Np*m, Np*m)
    #     p0_stack = np.tile(p0, Np)                   # (Np*m,)

    #     # stacked magnet positions: r_stack = (I kron Sel_pos) (p0_stack + A_p U)
    #     Spos = np.kron(np.eye(Np), self.Sel_pos)     # (Np*3, Np*m)
    #     r0_stack_base = Spos @ p0_stack.reshape(-1, 1)   # (Np*3,1)
    #     Rmap = Spos @ A_p                            # (Np*3, Np*m)

    #     X_guess = X0_stack + Mc @ U_guess_vec.reshape(-1, 1)

    #     # predicted p along horizon under current guess:
    #     p_guess = p0_stack.reshape(-1, 1) + A_p @ U_guess_vec.reshape(-1, 1)  # (Np*m,1)
    #     r_guess = (Spos @ p_guess).reshape(Np, 3)                              # (Np,3)

    #     x_guess = X_guess.reshape(Np, n)[:, :3]  
    #     x_guess = X_guess.reshape(Np, n)[:, :3]  

    #     A_rows = []
    #     l_rows = []
    #     u_rows = []

    #     for i in range(Np):
    #         r0 = r_guess[i, :].reshape(3, 1)
    #         x0 = x_guess[i, :].reshape(3, 1)
    #         s0 = (r0 - x0).reshape(3, 1)

    #         s0n = float(np.linalg.norm(s0))

    #         if s0n < 1e-8:
    #             continue

    #         # Blocks mapping U -> r_i and U -> x_i
    #         Rmap_i = Rmap[i*3:(i+1)*3, :]                  # (3, Np*m)
    #         Mc_i   = Mc[i*n:(i+1)*n, :]                    # (n, Np*m)
    #         Mc_i3  = Mc_i[:3, :]                           # (3, Np*m)

    #         # r_i - x_i = (r_base_i - X0_i) + (Rmap_i - Mc_i3) U
    #         X0_i = X0_stack[i*n:(i+1)*n, :][:3, :]         # (3,1)
    #         r_base_i = r0_stack_base[i*3:(i+1)*3, :]       # (3,1)

    #         # Linearized constraint: 2 s0^T (r_i - x_i) >= dmin^2 + ||s0||^2
    #         # => a_i U >= b_i
    #         a_i = (2.0 * s0.T) @ (Rmap_i - Mc_i3)          # (1, Np*m)
    #         b_i = (dmin**2 + (s0n**2)) - float((2.0 * s0.T) @ (r_base_i - X0_i))

    #         A_rows.append(a_i.reshape(1, -1))
    #         l_rows.append(b_i)
    #         u_rows.append(np.inf)

    #     if not A_rows:
    #         return None, None, None

    #     A_ko = np.vstack(A_rows)
    #     l_ko = np.asarray(l_rows, dtype=float)
    #     u_ko = np.asarray(u_rows, dtype=float)
    #     return A_ko, l_ko, u_ko
    def   _build_Du_matrix(self):
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

    # def _compute_Qtil(self, B0):
    #     _, P = dare_stabilising_K(self.A, B0, self.Q, self.R)
    #     self.Qf = P

    #     n = self.n
    #     Np = self.Np
    #     Qtil = np.zeros((Np*n, Np*n))
    #     if Np > 1:
    #         Qtil[:(Np-1)*n, :(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
    #     Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf
    #     return Qtil
    def _compute_Qtil(self, B0):
        n = self.n
        Np = self.Np

        # Default/fallback terminal cost
        Qf_fallback = self.Q.copy()   # or 10*self.Q if you want stronger terminal pull

        try:
            # Quick conditioning check: if B0 is tiny or rank-deficient, skip DARE
            s = np.linalg.svd(B0, compute_uv=False)
            # thresholds: tune these to your units
            if s.min() < 1e-8 or s.max() < 1e-8:
                print(f"Fallback")
                P = Qf_fallback
            else:
                
                _, P = dare_stabilising_K(self.A, B0, self.Q, self.R)
        except LinAlgError:
            P = Qf_fallback

        self.Qf = P

        Qtil = np.zeros((Np*n, Np*n))
        if Np > 1:
            Qtil[:(Np-1)*n, :(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
        Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf
        return Qtil
    def _disturbance_stack(self, d):
        n, Np = self.n, self.Np
        d = np.asarray(d, float).reshape(n, 1)         # (n,1)
        return np.tile(d, (Np, 1)) 
 



    def _clamp_p(self, p):
        p = np.minimum(np.maximum(p, self.p_min), self.p_max)

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


    def _build_prediction_mats(self, p0, U_guess):
        """
        Returns: p_seq, Mx, Mc, B0
        - p_seq: (Np,m) predicted p (only meaningful in LTV; in LTI it's still returned for convenience)
        - Mx, Mc: stacked prediction matrices
        - B0: first-step input matrix (for terminal cost DARE)
        """
        n, m, Np = self.n, self.m, self.Np

        if self.model_mode == "lti":
            # J0 = np.asarray(self.Jxy_fn(p0), dtype=float)
            # if J0.shape != (n, m):
            #     raise ValueError(f"Jacobian must be {(n,m)} but got {J0.shape}")
            # B0 = self.dt * J0
            J0 = np.asarray(self.Jxy_fn(p0), float)
            if self.use_adaptive_J:
                J0 = J0 + self.G_beta * self.G
            B0 = self.dt * J0
            print(f"Jacobian is : {np.linalg.norm(J0)}")
            
            Mx, Mc = seq_mat_lti(self.A, B0, Np)

            if U_guess is None:
                U_guess = np.zeros((Np, m))
            p_seq = self._p_seq_from_U(p0, U_guess)

            return p_seq, Mx, Mc, B0

        if U_guess is None:
            U_guess = np.zeros((Np, m))

        p_seq = self._p_seq_from_U(p0, U_guess)  
        B_list = []
        for i in range(Np):
            Ji = np.asarray(self.Jxy_fn(p_seq[i]), dtype=float)
            if Ji.shape != (n, m):
                raise ValueError(f"Jacobian must be {(n,m)} but got {Ji.shape}")
            B_list.append(self.dt * Ji)

        Mx, Mc = seq_mat_ltv(self.A, B_list)
        B0 = B_list[0]
        return p_seq, Mx, Mc, B0
    # def step(self, xref_seq, x_meas=None):
    #     if self.p is None:
    #         raise ValueError("Call set_initial_params(...) before step().")

    #     p_prev = self.p.copy()

    #     if x_meas is not None:
    #         self.x = np.asarray(x_meas, float).reshape(self.n,)

    #     p_now = self.p.copy()
    #     x_now = self.x.copy()

    #     if (self.use_adaptive_J and x_meas is not None
    #         and self.p_last_meas is not None and self.x_last_meas is not None):

    #         dp = (p_now - self.p_last_meas).reshape(self.m,)
    #         dx = (x_now - self.x_last_meas).reshape(self.n,)

    #         if np.linalg.norm(dp) > self.min_dp:
    #             Jm = np.asarray(self.Jxy_fn(self.p_last_meas), float)
    #             dx_hat = Jm @ dp
    #             e = (dx - dx_hat).reshape(self.n, 1)
    #             denom = float(dp @ dp + 1e-9)
    #             self.G = self.G + (self.G_alpha / denom) * (e @ dp.reshape(1, -1))
    #             self.G = np.clip(self.G, -self.G_max, self.G_max)

    #     if x_meas is not None:
    #         self.p_last_meas = p_now.copy()
    #         self.x_last_meas = x_now.copy()

    #     if self.use_offset_free and x_meas is not None:
    #         x_model = np.asarray(self.forward_tip_fn(p_now), float).reshape(self.n,)
    #         r = x_now - (x_model + self.d)
    #         if np.linalg.norm(r) < self.r_gate:
    #             self.d = np.clip(self.d + self.d_alpha * r, -self.d_max, self.d_max)



    #     n = self.n
    #     m = self.m
    #     Np = self.Np
    #     Mc_last = None
    #     X_aff_last = None
    #     p_seq_last = None
    #     X_nom_last = None

    #     xref_seq = np.asarray(xref_seq, float).reshape(Np, self.n)
    #     xref_stack = xref_seq.reshape(Np*self.n, 1)
    #     xk = self.x.reshape(self.n, 1)


    #     if self.U_warm is not None and self.U_warm.size == Np*m:
    #         U_opt_vec = self.U_warm.copy()
    #         U_guess = U_opt_vec.reshape(Np, m)
    #     else:
    #         U_guess = np.zeros((Np, m))
    #         U_opt_vec = None

    #     status_last = "init"

    #     for it in range(self.N_sqp):
    #         p0 = self.p.copy()
    #         p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)

    #         B_first = B0         
    #         p_first = p_seq[0].copy()

    #         Qtil = self._compute_Qtil(B0)
        
            
    #         Rtil = np.kron(np.eye(Np), self.R)

    #         if self.Np > 1 and np.any(np.diag(self.Rd) > 0):
    #             Rd_til = np.kron(np.eye(Np-1), self.Rd)
    #             H_du = self.Du.T @ Rd_til @ self.Du
    #         else:
    #             H_du = 0.0

    #         xk = self.x.reshape(self.n, 1)

    #         # X0_stack = (Mx @ xk).reshape(Np*n, 1)

    #         X0_stack = (Mx @ xk).reshape(Np*n, 1)
    #         if self.use_offset_free:
    #             X0_stack = X0_stack + self._disturbance_stack(self.d)


    #         U_guess_vec = U_guess.reshape(-1, 1)  # (Np*m,1)

    #         X_nom = np.vstack([self.forward_tip_fn(p_seq[i]) for i in range(Np)]).reshape(Np*n, 1)

    #         if self.use_offset_free:
    #             X_nom = X_nom + self._disturbance_stack(self.d)

    #         X_aff = X_nom - Mc @ U_guess_vec
    #         Mc_last = Mc
    #         X_aff_last = X_aff
    #         p_seq_last = p_seq
    #         X_nom_last = X_nom
    #         H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil + H_du)
    #         f = 2.0 * (Mc.T @ Qtil @ (X_aff - xref_stack))

    #         A_list, l_list, u_list = [], [], []
    #         if self.enable_tip_keepout:
    #             U_guess_vec = U_guess.reshape(-1)

    #             A_ko, l_ko, u_ko = self._build_tip_keepout_constraints(
    #                 Mc=Mc,
    #                 X0_stack=X_aff,
    #                 p0=self.p.copy(),
    #                 U_guess_vec=U_guess_vec
    #             )
    #             if A_ko is not None:
    #                 A_list.append(A_ko)
    #                 l_list.append(l_ko)
    #                 u_list.append(u_ko)


    #         if np.all(np.isfinite(self.u_max)):
    #             A_u = np.eye(Np * self.m)
    #             umax_stack = np.tile(self.u_max, Np)
    #             A_list.append(A_u)
    #             l_list.append(-umax_stack)
    #             u_list.append(+umax_stack)

    #         if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
    #             A_p = np.kron(self.S_np, np.eye(self.m))
    #             p0_stack = np.tile(self.p, Np)

    #             l_p = np.tile(self.p_min, Np) - p0_stack
    #             u_p = np.tile(self.p_max, Np) - p0_stack

    #             A_list.append(A_p)
    #             l_list.append(l_p)
    #             u_list.append(u_p)

    #         if self.band_xy > 0.0:
    #             band = float(self.band_xy)
    #             band_stack = band * np.ones((Np*n,))
    #             rhs_p = band_stack + (xref_stack - X_aff).reshape(-1)
    #             rhs_n = band_stack + (X_aff - xref_stack).reshape(-1)


    #             A_list.append(Mc)
    #             l_list.append(-np.inf * np.ones(Np*n))
    #             u_list.append(rhs_p)

    #             A_list.append(-Mc)
    #             l_list.append(-np.inf * np.ones(Np*n))
    #             u_list.append(rhs_n)

    #         if A_list:
    #             A_osqp = np.vstack(A_list)
    #             l_osqp = np.concatenate(l_list)
    #             u_osqp = np.concatenate(u_list)
    #         else:
    #             A_osqp = np.zeros((0, Np*m))
    #             l_osqp = np.zeros(0)
    #             u_osqp = np.zeros(0)

    #         U_warm_vec = U_opt_vec if U_opt_vec is not None else self.U_warm
    #         U_opt_vec, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=U_warm_vec)
    #         status_last = status

    #         infeas = (status not in ("solved", "solved inaccurate")) or (U_opt_vec is None)
    #         if infeas:
    #             U_guess = np.zeros((Np, m))
    #             U_opt_vec = None
    #             break

    #         U_guess = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)

    #     if U_opt_vec is None:
    #         u0 = np.zeros(m)
    #         U_seq = np.zeros((Np, m))
    #         X_pred = np.full((Np, n), np.nan)
    #         infeas_final = True
    #     else:
    #         U_seq = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)
    #         u0 = U_seq[0, :]

    #         # Use SQP-consistent prediction baseline from the final linearization
    #         if (Mc_last is None) or (X_aff_last is None):
    #             X_pred = np.full((Np, n), np.nan)
    #             infeas_final = True
    #         else:
    #             U_vec = U_opt_vec.reshape(-1, 1)                 # (Np*m,1)
    #             X_pred_stack = X_aff_last + Mc_last @ U_vec      # (Np*n,1)
    #             X_pred = X_pred_stack.reshape(Np, n)
    #             infeas_final = False

    #             p_next_true = self._clamp_p(p_prev + self.dt*u0)
    #             x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)

    #     # compute p_next (this is what you will command)
    #     p_next_true = self._clamp_p(p_prev + self.dt * u0)


    #     # OPTIONAL: model prediction of tip at p_next (for logging only)
    #     x_next_model = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
    #     if self.use_offset_free:
    #         x_next_model = x_next_model + self.d

    #     # --- update internal parameter state ONLY ---
    #     self.p = p_next_true.copy()

    #     # --- keep internal x as measurement, not model ---
    #     # If x_meas was provided, keep that; otherwise fall back to model.
    #     if x_meas is not None:
    #         self.x = np.asarray(x_meas, dtype=float).reshape(self.n,)
    #     else:
    #         self.x = x_next_model.copy()

    #     # Warm start stays the same
    #     self.U_warm = np.asarray(U_opt_vec, dtype=float).copy() if (U_opt_vec is not None and not infeas_final) else None

    #     info = dict(
    #         status=status_last,
    #         infeasible=int(infeas_final),
    #         u0=u0.copy(),
    #         p_now=self.p.copy(),
    #         x_now=self.x.copy(),              # <-- now this is measured if provided
    #         x_next_model=x_next_model.copy(), # <-- keep model result for debugging
    #         d=self.d.copy(),
    #         X_pred=X_pred.copy(),
    #         U_seq=U_seq.copy(),
    #         # ... keep your other debug fields ...
    #     )

    #     return self.p.copy(), self.x.copy(), info


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
        B_first = None
        p_lin = None
        p_first = None
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

            X0_stack = (Mx @ xk).reshape(Np*n, 1)
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

        # --- choose control to apply ---
        infeas_final = (U_opt_vec is None)

        if infeas_final:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
        else:
            U_seq = np.asarray(U_opt_vec, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]

        # --- build predicted horizon (linear) if we have final linearization ---
        X_pred = np.full((Np, n), np.nan)
        if (not infeas_final) and (Mc_last is not None) and (X_aff_last is not None):
            U_vec = U_opt_vec.reshape(-1, 1)                 # (Np*m,1)
            X_pred_stack = X_aff_last + Mc_last @ U_vec      # (Np*n,1)
            X_pred = X_pred_stack.reshape(Np, n)

        # --- apply one step to parameters ---
        p_next_true = self._clamp_p(p_prev + self.dt * u0)

        # --- evaluate forward model once ---
        x_next_true = np.asarray(self.forward_tip_fn(p_next_true), float).reshape(self.n,)
        if self.use_offset_free:
            x_next_true = x_next_true + self.d

        # --- commit internal state ONCE ---
        self.p = p_next_true.copy()
        self.x = x_next_true.copy()

        # warm-start
        self.U_warm = np.asarray(U_opt_vec, dtype=float).copy() if not infeas_final else None
        # safe pred1_err
        pred1_err = np.nan
        if (X_pred is not None) and (X_pred.shape[0] > 0) and np.all(np.isfinite(X_pred[0])):
            pred1_err = float(np.linalg.norm(x_next_true - X_pred[0]))

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
            p_prev=p_prev.copy(),
            x_prev = x_prev.copy(),
            x_next_true = x_next_true.copy(),
            X_pred0 = X_pred[0].copy(),
            pred0_vec_err = (x_next_true - X_pred[0]).copy(),
            B_first=B_first.copy() if B_first is not None else None,
            p_lin=p_lin.copy() if isinstance(p_lin, np.ndarray) else p_lin,
            p_first=p_first.copy() if isinstance(p_first, np.ndarray) else p_first,
            X_aff_last = X_aff_last.copy() if X_aff_last is not None else None,
            Mc_last    = Mc_last.copy()    if Mc_last is not None else None,
            X_nom_last = X_nom_last.reshape(Np, n).copy() if X_nom_last is not None else None,
            p_seq_last = p_seq_last.copy() if p_seq_last is not None else None,
        )
        return self.p.copy(), self.x.copy(), info


# def forward_cosserat_from_pose_ur_rotvec_L(p, model, *, m_body):
#     r_src, q_src, L = unpack_pose_ur_rotvec_L(p)



def forward_cosserat_from_pose_ur_rotvec_L(p, model, *, m_body, L_mag=L_MAG):
    r_src, q_src, L = unpack_pose_ur_rotvec_L(p)

    wire_len = wire_len_from_L(L, L_mag)

    # rebuild magnetisation profile for this L
    m_local_fun = make_m_local_fun_wire_tip(wire_len, mode="axial", alpha_end=0.0)

    # run forward with per-call wire_len + per-call m_local_fun
    # easiest: temporarily override model.m_local_fun
    old_fun = model.m_local_fun
    try:
        model.m_local_fun = m_local_fun
        out = model.forward(L=L, r_src=r_src, q_src=q_src, m_body=m_body, wire_len=wire_len)
    finally:
        model.m_local_fun = old_fun

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
def debug_step_pose7(k, x_target, xref_seq, p_now, x_now, info, mpc,i_ref=None,
                     print_horizon=4, do_nl_rollout=True):
    xref_seq = np.asarray(xref_seq, float).reshape(-1, 3)
    e_ref0 = xref_seq[0] - x_now
    err_ref0_mm = 1e3 * np.linalg.norm(e_ref0)

    # optional: also show along-horizon terminal ref error
    e_ref_end = xref_seq[-1] - x_now
    err_ref_end_mm = 1e3 * np.linalg.norm(e_ref_end)

    print(
        f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f},{x_now[2]:+.4f}] "
        f"||e_ref0||={err_ref0_mm:.2f}mm ||e_refN||={err_ref_end_mm:.2f}mm "
        f"status={info['status']} infeas={info['infeasible']}"
    )
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
    if i_ref is not None:
        print(f"   path index i_ref={i_ref}")
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
        p_start = info["p_prev"]
        P_nl, X_nl = rollout_open_loop_from_plan(mpc, p_start, U_seq)
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
def arc_length_param(C):
    """C: (M,3) -> s: (M,) cumulative arc-length."""
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s
def closest_index_on_path(x, path):
    """
    x: (3,)
    path: (M,3)
    returns idx of closest vertex
    """
    d2 = np.sum((path - x.reshape(1,3))**2, axis=1)
    return int(np.argmin(d2))
def build_xref_from_centerline(path, x_tip_now, Np, step_idx=3):
    """
    path: (M,3) centerline points
    x_tip_now: (3,) current tip position
    Np: horizon length
    step_idx: how many path points to advance per horizon step
              (with 1mm spacing, step_idx=3 -> 3mm per step)
    """
    M = path.shape[0]
    i0 = closest_index_on_path(x_tip_now, path)

    idx = i0 + step_idx * np.arange(Np)
    idx = np.clip(idx, 0, M-1)
    return path[idx, :]  # (Np,3)
def resample_polyline(C, ds_target=1e-3):
    """
    Resample polyline C to approximately uniform spacing ds_target.
    Returns C_rs: (Mr,3), s_rs: (Mr,)
    """
    C = np.asarray(C, float)
    s = arc_length_param(C)
    L = s[-1]
    if L < 1e-12:
        return C.copy(), s

    s_rs = np.arange(0.0, L + 0.5*ds_target, ds_target)
    C_rs = np.zeros((len(s_rs), 3), float)

    # piecewise-linear interpolation in arc-length
    for k in range(3):
        C_rs[:, k] = np.interp(s_rs, s, C[:, k])

    return C_rs, s_rs
def advance_cursor_monotone(path, x, i_ref, window=30):
    """
    Find closest index to x, but only search forward from i_ref.
    window: how far ahead you allow matching (in points).
    """
    M = path.shape[0]
    i_lo = int(i_ref)
    i_hi = int(min(M, i_ref + window))
    seg = path[i_lo:i_hi]
    if seg.shape[0] == 0:
        return M - 1
    d2 = np.sum((seg - x.reshape(1, 3))**2, axis=1)
    return i_lo + int(np.argmin(d2))


def build_xref_from_cursor(path, i_ref, Np, step_idx=3):
    M = path.shape[0]
    idx = i_ref + step_idx * np.arange(Np)
    idx = np.clip(idx, 0, M - 1)
    return path[idx], int(idx[0])
def forward_tip_live(p):
    return forward_model(p)  # updates cache normally

def forward_tip_det(p):
    import copy
    last = copy.deepcopy(forward_model._last)
    last_centerline = None if forward_model.last_p_centerline is None else forward_model.last_p_centerline.copy()
    last_tip = None if forward_model.last_tip is None else forward_model.last_tip.copy()

    out = forward_model(p)

    forward_model._last = last
    forward_model.last_p_centerline = last_centerline
    forward_model.last_tip = last_tip
    return out

if __name__ == "__main__":
    pivot_point = np.array([
    0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)


    start_point = np.array([
    0.6681328220229531, -0.7055298925316631, 0.1517853768068757, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
    ], float)
    start_point[2] -=0.25
    T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    L0 = 0.049 
    wire_len0 = wire_len_from_L(L0, L_MAG)
    model = CosseratForwardModel(
        p0=p0_ur,
        q0=q0_ur,
        Kinv_fun=Kbt_inv_profile,
        m_local_fun=make_m_local_fun_wire_tip(wire_len0, mode="axial", alpha_end=0.0),
        m_moment=0.0,
        wire_len = wire_len0,
    )

    m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)
    
    # forward_tip_fn = lambda p: forward_cosserat_from_pose_ur_rotvec_L(p, model, m_body=m_body, L_mag=0.04)
    # J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_fn, eps)
    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])  
    s_straight = 0.005

    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur,
        t0=t0,
        length=0.08 + s_straight,     
        n_pts=130,                       # increase points so resolution stays similar
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(150.0),
        bend_start=0.01 + s_straight,    
        bend_end=0.08 + s_straight       
    )

    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.004)
    lumen_path = lumen_C

    forward_model = EnergyMinForwardWithLumen(
            p0_ur=p0_ur, q0_ur=q0_ur,
            Kinv_fun=Kbt_inv_profile,
            u_star=np.zeros(3),
            mag_len=beam_params.length_of_mag,
            m_body=m_body,
            lumen_C=lumen_C, lumen_R=lumen_R,
            N_nodes=15, maxiter=15,
            L0_init=0.01, dL_internal=0.002
        )

    # forward_tip_fn = lambda p: forward_model(p)
    # J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_fn, eps)


    J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_det, eps)
    # p0 = np.array([ 0.63360145, -0.56541852,  0.20705173, -3.116988654350607, 0.19059356279735162, 0.028215660130034903,L0])
    start_point_pose6 = start_point  # [x,y,z, rx,ry,rz]
    p0 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
                start_point_pose6[3], start_point_pose6[4], start_point_pose6[5],
                L0], float)
    p_min = np.array([ 0.2, -1, -0.2,  p0[3]-np.pi, p0[4]-np.pi, p0[5]-np.pi,  0.03])
    p_max = np.array([ 0.85,  1,  1.0,  p0[3]+np.pi, p0[4]+np.pi, p0[5]+np.pi,  0.12])
    w_u  = np.array([1e-3, 1e-3, 1e-3,  1e-4, 1e-4, 1e-4,  1e-3])
    w_du = np.array([1e-6, 1e-6, 1e-6,   1e-6, 1e-6, 1e-6,   1e-6])

    mpc = mpc_controller_tipxy_LTI(
        Jxy_fn=J_fn,
        forward_tip_fn=forward_tip_live,
        dt=0.05,
        Np=4,
        n_out=3,
        n_u=7,
        w_xy=(1, 1, 1),
        w_u=w_u,
        w_du=w_du,
        model_mode="lti",
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        N_sqp=4,
        use_offset_free=False
    )


    mpc.set_initial_params(p0)
    x_target = np.array([0.80020363, -0.71817534, -0.09914591])
    Np = mpc.Np
    xref_seq = np.tile(x_target, (Np, 1))   
    p_test = p0.copy()
    J_test = J_fn(p_test)
    print("J shape:", J_test.shape)  
    x_start = mpc.x.copy()
    n=15
    # path = np.linspace(x_start, x_target, n)
    # path is lumen_path (already resampled ~1mm)
    M = lumen_path.shape[0]

    i_ref = 0              # start at beginning (or compute once from initial x)
    max_steps = 5000       # safety limit so it doesn't run forever
    window = 80            # search ahead only (50mm if ds=1mm)

    for k in range(max_steps):
        x_now = mpc.x.copy()

        # advance cursor monotonically (never goes backward)
        i_ref = advance_cursor_monotone(lumen_path, x_now, i_ref, window=window)

        # stop when we are at the end
        if i_ref >= M - 1:
            print("Reached end of lumen path.")
            break

        # build reference horizon from cursor
        xref_seq, i_ref0 = build_xref_from_cursor(lumen_path, i_ref, mpc.Np, step_idx=1)

        # one MPC step
        p_post, x_post, info = mpc.step(xref_seq, x_meas=None)

        debug_step_pose7(k, lumen_path[-1], xref_seq, p_post, x_post, info, mpc,
                        print_horizon=mpc.Np, do_nl_rollout=True, i_ref=i_ref)
        if np.linalg.norm(xref_seq[0] - x_post) < 3e-3:   # 1 mm tolerance
            i_ref = min(i_ref + 1, M-1)
        # optional plotting
        if k % 15 == 0:
            cache0 = copy.deepcopy(forward_model._last)
            cl0    = None if forward_model.last_p_centerline is None else forward_model.last_p_centerline.copy()
            tip0   = None if forward_model.last_tip is None else forward_model.last_tip.copy()
            info0  = copy.deepcopy(forward_model.last_info)
            hist0  = forward_model.last_hist

            tip_returned = forward_model(p_post)
            C = forward_model.last_p_centerline.copy()     # <-- capture for plotting
            tip_centerline_end = C[:, -1].copy()

            # t0 = forward_tip_det(p_post)
            # t1 = forward_tip_det(p_post)
            # print("det repeat (mm):", 1e3*np.linalg.norm(t1 - t0))

            # restore EVERYTHING
            forward_model._last = cache0
            forward_model.last_p_centerline = cl0
            forward_model.last_tip = tip0
            forward_model.last_info = info0
            forward_model.last_hist = hist0

            # plot the captured centerline (C), not forward_model.last_p_centerline
            plot_energy_only_3d(
                C,
                lumen_C=lumen_C, lumen_R=lumen_R, p0=p0_ur,
                targets=xref_seq,
                tip=x_post, tip_from_centerline=tip_centerline_end,
                title=f"Energy-min centreline (k={k}, i_ref={i_ref})"
            )
    print("UR pose6:", repr(p_post))
    _ = forward_model(p_post)
    p_energy = forward_model.last_p_centerline
    plot_energy_only_3d(
        p_energy,
        lumen_C=lumen_C, lumen_R=lumen_R,
        p0=p0_ur,
        p_straight=None,
        title=f"Energy-min centreline at final MPC pose"
    )
        
        

    
    
