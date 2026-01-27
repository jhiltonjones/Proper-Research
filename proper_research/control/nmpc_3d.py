import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.models.forward_model import make_forward_fn, make_jac_fn
from proper_research.parameters import default_magnet_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, make_safe_forward_tip_fn
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import Rx, Ry, Rz, unpack_pose_euler_L, quat_to_rot_wxyz, rot_to_euler_zyx
from beam_direction_magnetisation.post_processing.post_processing import plot_mpc_state_3d
from scipy.optimize import NonlinearConstraint, BFGS

mag_params = default_magnet_params()

# p = [x,y,z, roll,pitch,yaw, L]
p_min = np.array([ 0.00, -0.20, -0.20,  -np.pi, -np.pi/2, -np.pi,  0.03])
p_max = np.array([ 0.30,  0.20,  0.20,  +np.pi, +np.pi/2, +np.pi,  0.08])

u_max = np.array([ np.deg2rad(30),np.deg2rad(30), np.deg2rad(30), np.deg2rad(30), np.deg2rad(30), np.deg2rad(45),  0.02])
eps = np.array([
    1e-3, 1e-3, 1e-3,              # x,y,z
    np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  # roll,pitch,yaw
    5e-4                              # L
], dtype=float)
def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi
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

import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class nmpc_controller_tipxyz_pose7:
    def __init__(self, *, forward_tip_fn,
                 dt=0.05, Np=6,
                 w_xyz=(150.0,150.0,150.0),
                 w_u=None, w_du=None,
                 u_max=None, p_min=None, p_max=None,
                 use_offset_free=False, d_alpha=0.15,
                 enable_tip_keepout=True,
                 d_min_tip_mag=0.10,
                 maxiter=60,
                 verbose=False):

        self.forward_tip_fn = forward_tip_fn
        self.dt = float(dt)
        self.Np = int(Np)
        self.n = 3
        self.m = 7

        self.Q  = np.diag(np.asarray(w_xyz, float).ravel())
        if w_u is None:  w_u  = (1e-3,) * self.m
        if w_du is None: w_du = (0.0,)  * self.m
        self.R  = np.diag(np.asarray(w_u,  float).ravel())
        self.Rd = np.diag(np.asarray(w_du, float).ravel())

        self.u_max = np.asarray(u_max, float).ravel() if u_max is not None else np.full(self.m, np.inf)
        self.p_min = np.asarray(p_min, float).ravel() if p_min is not None else -np.full(self.m, np.inf)
        self.p_max = np.asarray(p_max, float).ravel() if p_max is not None else +np.full(self.m, np.inf)

        if self.u_max.size != self.m: raise ValueError("u_max wrong length")
        if self.p_min.size != self.m: raise ValueError("p_min wrong length")
        if self.p_max.size != self.m: raise ValueError("p_max wrong length")

        self.use_offset_free = bool(use_offset_free)
        self.d_alpha = float(d_alpha)
        self.d = np.zeros(self.n)

        self.p = None
        self.x = None
        self.U_warm = None

        self.maxiter = int(maxiter)
        self.verbose = bool(verbose)
        self.enable_tip_keepout = bool(enable_tip_keepout)
        self.d_min_tip_mag = float(d_min_tip_mag)
        # S matrix such that p_stack = p0_stack + S @ U_flat
        self.S = self._build_S()

    def _build_S(self):
        Np, dt, m = self.Np, self.dt, self.m
        S_np = np.tril(np.ones((Np, Np))) * dt
        return np.kron(S_np, np.eye(m))  # (Np*m, Np*m)
    def _keepout_g(self, U_flat):
        """
        g_k(U) = ||r_k - x_k|| for k=1..Np (vector length Np)
        Enforce g_k >= d_min.
        """
        U_flat = np.asarray(U_flat, float)
        p_seq, X = self._rollout_X_from_U_flat(self.p, U_flat)  # p_k, x_k for each step

        r_seq = p_seq[:, 0:3]               # (Np,3) magnet position
        d = np.linalg.norm(r_seq - X, axis=1)  # (Np,)
        return d
    def set_dt(self, dt):
        self.dt = float(dt)
        self.S = self._build_S()

    def _clamp_p_apply(self, p):
        """
        Clamp used when actually updating the live MPC state (safety).
        Wrap roll/yaw to [-pi,pi], clamp pitch to bounds.
        """
        p = np.minimum(np.maximum(p, self.p_min), self.p_max)
        p[3] = wrap_pi(p[3])  # roll
        p[5] = wrap_pi(p[5])  # yaw
        return p

    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.m,)
        self.x = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
        self.d = np.zeros(self.n)
        self.U_warm = None

    def _p_seq_from_U_flat(self, p0, U_flat):
        """
        Compute p1..pNp from linear integrator WITHOUT wrapping.
        Constraints enforce bounds (on unwrapped angles).
        """
        Np, m = self.Np, self.m
        p0 = np.asarray(p0, float).reshape(m,)
        U_flat = np.asarray(U_flat, float).reshape(Np*m,)

        p0_stack = np.tile(p0, Np)           # (Np*m,)
        p_stack  = p0_stack + self.S @ U_flat
        return p_stack.reshape(Np, m)

    def _rollout_X_from_U_flat(self, p0, U_flat):
        """
        Rollout nonlinear outputs X[k]=forward_tip_fn(p_k).
        """
        p_seq = self._p_seq_from_U_flat(p0, U_flat)
        X = np.zeros((self.Np, self.n), float)
        for k in range(self.Np):
            xk = np.asarray(self.forward_tip_fn(p_seq[k]), float).reshape(self.n,)
            if self.use_offset_free: 
                xk = xk + self.d
            X[k] = xk
        return p_seq, X

    def step(self, xref_seq, x_meas=None):
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        p_prev = self.p.copy()
        x_prev = self.x.copy()

        if x_meas is not None:
            self.x = np.asarray(x_meas, float).reshape(self.n,)

        # offset-free update (optional)
        if self.use_offset_free and x_meas is not None:
            x_model = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)
            r = self.x - x_model
            self.d = (1.0 - self.d_alpha) * self.d + self.d_alpha * r

        Np, m = self.Np, self.m
        xref_seq = np.asarray(xref_seq, float).reshape(Np, self.n)

        # warm start
        if self.U_warm is None or self.U_warm.size != Np*m:
            U0 = np.zeros(Np*m, float)
        else:
            U0 = self.U_warm.copy()

        # bounds on U
        lb = np.tile(-self.u_max, Np)
        ub = np.tile(+self.u_max, Np)
        bounds = Bounds(lb, ub)

        # linear constraint for p bounds across horizon:
        # p_stack = p0_stack + S @ U  => S@U in [pmin_stack-p0_stack, pmax_stack-p0_stack]
        p0_stack   = np.tile(self.p, Np)
        pmin_stack = np.tile(self.p_min, Np)
        pmax_stack = np.tile(self.p_max, Np)

        lcon = pmin_stack - p0_stack
        ucon = pmax_stack - p0_stack
        p_constraint = LinearConstraint(self.S, lcon, ucon)

        Q, R, Rd = self.Q, self.R, self.Rd

        def cost(U_flat):
            U_flat = np.asarray(U_flat, float)
            U_seq  = U_flat.reshape(Np, m)
            _, X = self._rollout_X_from_U_flat(self.p, U_flat)

            J = 0.0
            for k in range(Np):
                e = (X[k] - xref_seq[k]).reshape(self.n, 1)
                u = U_seq[k].reshape(m, 1)
                J += (e.T @ Q @ e).item() + (u.T @ R @ u).item()

            if Np > 1 and np.any(np.diag(Rd) > 0):
                for k in range(Np - 1):
                    du = (U_seq[k+1] - U_seq[k]).reshape(m, 1)
                    J += (du.T @ Rd @ du).item()

            return float(J)
        constraints = [p_constraint]

        if self.enable_tip_keepout:
            dmin = self.d_min_tip_mag
            keepout = NonlinearConstraint(
                fun=self._keepout_g,
                lb=dmin * np.ones(self.Np),
                ub=np.inf * np.ones(self.Np),
                jac="2-point"   # keep finite-diff Jacobian
                # no hess here
            )
            constraints.append(keepout)

        res = minimize(
            cost, U0,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            hess=BFGS(),   # <-- add this
            options={
                "verbose": 3 if self.verbose else 0,
                "maxiter": self.maxiter,
                "gtol": 1e-6,
                "xtol": 1e-8,
            }
        )

        solved = bool(res.success) and np.all(np.isfinite(res.x))
        if not solved:
            U_opt = np.zeros(Np*m, float)
            status = f"fail:{res.message}"
            infeasible = 1
        else:
            U_opt = np.asarray(res.x, float)
            status = "solved"
            infeasible = 0

        U_seq = U_opt.reshape(Np, m)
        u0 = U_seq[0].copy()

        # predictions
        p_seq, X_pred = self._rollout_X_from_U_flat(self.p, U_opt)

        # apply first control (with clamp/wrap safety)
        p_next = self.p + self.dt * u0
        p_next = self._clamp_p_apply(p_next)
        self.p = p_next
        self.x = np.asarray(self.forward_tip_fn(self.p), float).reshape(self.n,)

        # warm start shift
        if solved:
            U_shift = np.vstack([U_seq[1:], U_seq[-1:]])
            self.U_warm = U_shift.reshape(-1)
        else:
            self.U_warm = None

        # prediction error diagnostic (one-step)
        x_next_true = self.x.copy()
        x_pred0 = X_pred[0].copy()
        pred0_vec_err = (x_next_true - x_pred0).copy()
        pred1_err = float(np.linalg.norm(pred0_vec_err))

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
            p_prev=p_prev,
            x_prev=x_prev,
            x_next_true=x_next_true,
            X_pred0=x_pred0,
            pred0_vec_err=pred0_vec_err,
            pred1_err=pred1_err,
        )
        return self.p.copy(), self.x.copy(), info
def forward_cosserat_from_pose_euler_L(p, model, *, m_body):
    r_src, q_src, L = unpack_pose_euler_L(p)
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


def debug_step_pose7(k, x_target, xref_seq, p_now, x_now, info, nmpc, print_horizon=4, do_nl_rollout=True):
    e = x_target - x_now
    err_mm = 1e3 * np.linalg.norm(e)

    print(f"k={k:02d} tip=[{x_now[0]:+.4f},{x_now[1]:+.4f},{x_now[2]:+.4f}] "
          f"||e||={err_mm:.2f}mm status={info['status']} infeas={info['infeasible']}")

    u0 = info["u0"]
    X_pred = info["X_pred"]
    U_seq = info["U_seq"]
    pred_err = info.get("pred1_err", np.nan)
    r_src, q_src, L = unpack_pose_euler_L(p_now)
    print(f"Pose is : {r_src} and Q is : {q_src}")
    R = quat_to_rot_wxyz(q_src)
    r2,p2,y2 = rot_to_euler_zyx(R)
    r2,p2,y2 = wrap_pi(r2), wrap_pi(p2), wrap_pi(y2)
    print("   euler-from-q(deg):", np.rad2deg([r2,p2,y2]))
    print("   euler-state(deg) :", np.rad2deg(p_now[3:6]))
    # p_now: [x,y,z, roll,pitch,yaw, L]
    print("   p_now:",
          f"x={p_now[0]:+.3f} y={p_now[1]:+.3f} z={p_now[2]:+.3f}  "
          f"rpy(deg)=[{np.rad2deg(p_now[3]):+.1f},{np.rad2deg(p_now[4]):+.1f},{np.rad2deg(p_now[5]):+.1f}]  "
          f"L={p_now[6]:.3f}")

    print("   u0:",
          f"dx={u0[0]:+.4f} dy={u0[1]:+.4f} dz={u0[2]:+.4f}  "
          f"d_rpy(deg/s)=[{np.rad2deg(u0[3]):+.2f},{np.rad2deg(u0[4]):+.2f},{np.rad2deg(u0[5]):+.2f}]  "
          f"dL={u0[6]:+.5f}")

    print(f"   pred1_err (one-step tip mismatch): {pred_err}")

    # horizon print
    ph = min(print_horizon, X_pred.shape[0]) if X_pred is not None else 0
    if X_pred is not None and np.all(np.isfinite(X_pred)):
        for i in range(ph):
            print(f"   pred[{i}]={X_pred[i]}   ref[{i}]={xref_seq[i]}")
    else:
        print("   X_pred NaNs or missing -> infeasible/failed QP")

def build_xref_from_path(path, k, Np):
    """
    path: (N,3)
    returns xref_seq: (Np,3) with look-ahead
    """
    N = path.shape[0]
    idx = np.clip(np.arange(k, k + Np), 0, N - 1)
    return path[idx]




model = CosseratForwardModel(
    p0=np.array([0.0, 0.0, 0.0]),
    q0=np.array([1.0, 0.0, 0.0, 0.0]),
    Kinv_fun=Kbt_inv_profile,
    m_local_fun=make_m_local_fun_wire_tip(mode="axial", alpha_end=0.0),
    m_moment=0.0,  # not used by this m_local_fun
    n_nodes=120,
    tol=1e-5    
    )
m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)

forward_tip_fn = make_safe_forward_tip_fn(model, m_body, p_min, p_max, penalty=1e3)
J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_fn, eps)



nmpc = nmpc_controller_tipxyz_pose7(
    forward_tip_fn=forward_tip_fn,
    dt=0.05,
    Np=3,
    w_xyz=(150,150,150),
    w_u=(1e-3,)*7,
    w_du=(1e-3,)*7,
    u_max=u_max,
    p_min=p_min,
    p_max=p_max,
    enable_tip_keepout=True,
    d_min_tip_mag=0.10,
    maxiter=100
)

L0 = 0.05                      # 5 cm beam length
d_tip_to_mag = 0.14            # 14 cm from tip

# "Straight beam" nominal tip position/direction assumption:
tip0 = np.array([L0, 0.0, 0.0])
a0   = np.array([1.0, 0.0, 0.0])  # aligned with +x

r_src0 = tip0 + d_tip_to_mag * a0  # magnet center 14 cm in front of tip

roll0 = 0.0
pitch0 = 0.0
yaw0 = 0.0

p0 = np.array([r_src0[0], r_src0[1], r_src0[2], roll0, pitch0, yaw0, L0], dtype=float)
nmpc.set_initial_params(p0)

x_target = np.array([0.05943802, 0.00788534,0])
x_start = nmpc.x.copy()
n=10
path = np.linspace(x_start, x_target, n)
for k in range(n):
    p_pre = nmpc.p.copy()
    x_pre = nmpc.x.copy()

    xref_seq = build_xref_from_path(path, k, nmpc.Np)  
    p_post, x_post, info = nmpc.step(xref_seq, x_meas=None)
    print(f"k={k:02d} tip={x_post}  ||e||={np.linalg.norm(x_target-x_post):.6f}  status={info['status']}  iters={info['solver_iters']}")
    debug_step_pose7(k, x_target, xref_seq, p_post, x_post, info, nmpc,
                     print_horizon=3, do_nl_rollout=True)
    error = np.linalg.norm(x_target-x_post)
    if error < 0.001 or k==n-1:
        plot_mpc_state_3d(model, m_body, p_post, x_post, title=f"MPC step {k}", dipole_scale=0.05)
        break

 