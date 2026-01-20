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
                 band_xy=0.0,
                 u_max=(np.deg2rad(20), np.deg2rad(20), 0.02, 0.01),
                 p_min=(np.deg2rad(-90), np.deg2rad(-90), 0.10, 0.03),
                 p_max=(np.deg2rad( 90), np.deg2rad( 90), 0.25, 0.08)):

        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

        self.dt = float(dt)
        self.Np = int(Np)

        # state x = [x_tip, y_tip]
        self.A = np.eye(2)

        self.Q = np.diag(np.asarray(w_xy, dtype=float).ravel())  # 2x2
        self.R = np.diag(np.asarray(w_u, dtype=float).ravel())   # 4x4

        self.band_xy = float(band_xy)

        self.u_max = np.asarray(u_max, dtype=float).ravel()  # (4,)
        self.p_min = np.asarray(p_min, dtype=float).ravel()  # (4,)
        self.p_max = np.asarray(p_max, dtype=float).ravel()  # (4,)

        self.p = None      # (4,)
        self.x = None      # (2,)
        self.Qf = None

        self._rebuild_S()

    def _rebuild_S(self):
        # lower-triangular integration matrix: p_k+i = p_k + dt * sum_{j<=i} u_j
        self.S_np = np.tril(np.ones((self.Np, self.Np))) * self.dt

    def set_dt(self, dt):
        self.dt = float(dt)
        self._rebuild_S()

    def set_initial_params(self, gamma0, beta0, rho0, L0):
        self.p = np.array([gamma0, beta0, rho0, L0], dtype=float)
        self.x = self.forward_tip_fn(self.p)  # initialise state from forward model

    def _build_lti_model(self):
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        # Linearise around current p
        Jxy = np.asarray(self.Jxy_fn(self.p), dtype=float)  # (2,4)
        if Jxy.shape != (2, 4):
            raise ValueError(f"Expected Jxy shape (2,4), got {Jxy.shape}")

        Bmat = self.dt * Jxy  # (2,4)

        # terminal cost via DARE
        _, P = dare_stabilising_K(self.A, Bmat, self.Q, self.R)
        self.Qf = P

        Mx, Mc = seq_mat_lti(self.A, Bmat, self.Np)
        return Jxy, Bmat, Mx, Mc

    def step(self, xref_seq, x_meas=None):
        """
        xref_seq: shape (Np,2) reference tip positions
        x_meas: optional measured tip state (2,) if you have sensor feedback.
                If None, we use current stored x.
        """
        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(2,)

        Jxy, Bmat, Mx, Mc = self._build_lti_model()

        n = 2
        m = 4
        Np = self.Np

        # Build Qtil (stage costs + terminal)
        Qtil = np.zeros((Np*n, Np*n))
        if Np > 1:
            Qtil[:(Np-1)*n, :(Np-1)*n] = np.kron(np.eye(Np-1), self.Q)
        Qtil[(Np-1)*n:, (Np-1)*n:] = self.Qf

        Rtil = np.kron(np.eye(Np), self.R)

        xk = self.x.reshape(2, 1)
        xref_seq = np.asarray(xref_seq, dtype=float).reshape(Np, 2)
        xref_stack = xref_seq.reshape(Np*n, 1)

        X0_stack = (Mx @ xk).reshape(Np*n, 1)

        # QP objective: 0.5 U^T H U + f^T U
        H = 2.0 * (Mc.T @ Qtil @ Mc + Rtil)
        f = 2.0 * (Mc.T @ Qtil @ (X0_stack - xref_stack))

        A_list, l_list, u_list = [], [], []

        # (1) input bounds: -u_max <= u <= u_max
        if np.all(np.isfinite(self.u_max)):
            A_u = np.eye(Np * m)
            umax_stack = np.tile(self.u_max, Np)
            A_list.append(A_u)
            l_list.append(-umax_stack)
            u_list.append(+umax_stack)

        # (2) parameter bounds across horizon: p_min <= p0 + S*u <= p_max
        if np.all(np.isfinite(self.p_min)) and np.all(np.isfinite(self.p_max)):
            A_p = np.kron(self.S_np, np.eye(m))  # (Np*m, Np*m)
            p0_stack = np.tile(self.p, Np)       # (Np*m,)

            l_p = np.tile(self.p_min, Np) - p0_stack
            u_p = np.tile(self.p_max, Np) - p0_stack

            A_list.append(A_p)
            l_list.append(l_p)
            u_list.append(u_p)

        # (3) optional tube constraint around xref (component-wise box)
        if self.band_xy > 0.0:
            band = float(self.band_xy)
            band_stack = band * np.ones((Np*n,))

            # constraint:  X_pred = X0 + Mc U
            # enforce: -band <= X_pred - xref <= +band
            # => Mc U <= band + xref - X0
            # => -Mc U <= band - xref + X0
            rhs_p = band_stack + (xref_stack - X0_stack).reshape(-1)
            rhs_n = band_stack + (X0_stack - xref_stack).reshape(-1)

            A_list.append(Mc)
            l_list.append(-np.inf * np.ones(Np*n))
            u_list.append(rhs_p)

            A_list.append(-Mc)
            l_list.append(-np.inf * np.ones(Np*n))
            u_list.append(rhs_n)

        # Stack constraints
        if A_list:
            A_osqp = np.vstack(A_list)
            l_osqp = np.concatenate(l_list)
            u_osqp = np.concatenate(u_list)
        else:
            A_osqp = np.zeros((0, Np*m))
            l_osqp = np.zeros(0)
            u_osqp = np.zeros(0)

        U_opt, _, status = solve_qp_osqp(H, f, A_osqp, l_osqp, u_osqp, U_warm=None)
        infeas = (status not in ("solved", "solved inaccurate")) or (U_opt is None)

        if infeas:
            u0 = np.zeros(m)
            U_seq = np.zeros((Np, m))
            X_pred = np.full((Np, n), np.nan)
        else:
            U_seq = np.asarray(U_opt, dtype=float).reshape(Np, m)
            u0 = U_seq[0, :]
            X_pred_stack = (X0_stack + Mc @ U_opt.reshape(-1, 1))
            X_pred = X_pred_stack.reshape(Np, n)

        self.p = self.p + u0 * self.dt
        self.p = np.minimum(np.maximum(self.p, self.p_min), self.p_max)

        self.x = self.forward_tip_fn(self.p)

        info = dict(
            status=status,
            infeasible=int(infeas),
            u0=u0.copy(),
            p_now=self.p.copy(),
            x_now=self.x.copy(),
            Jxy=Jxy.copy(),
            Bmat=Bmat.copy(),
            X_pred=X_pred.copy(),
            U_seq=U_seq.copy(),
        )
        return self.p.copy(), self.x.copy(), info
forward_tip = make_forward_fn(mag, A_cs, E, I, mag_params.mag_epm)
Jxy_fn = make_jac_fn(forward_tip, eps)
mpc_xy = mpc_controller_tipxy_LTI(
    Jxy_fn=Jxy_fn,
    forward_tip_fn=forward_tip,
    dt=2,
    Np=5,
    w_xy=(100.0, 100.0),
    w_u=(1e-4, 1e-4, 1e-2, 1e-2),
    band_xy=0.0,  
    u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
    p_min=p_min,
    p_max=p_max
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
M = 80
x_start = mpc_xy.x.copy()
path = path_sinusoid_x(x_start, x_target, M=40, amp=0.006, cycles=1.0)
idx = 0
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Post-analysis logger
# ----------------------------
log = {
    "k": [],
    "idx": [],
    "x_now": [],          # actual tip after MPC update
    "x_now_prev": [],     # tip before update
    "x_ref0": [],         # first ref point in horizon
    "xref_seq": [],       # full horizon reference
    "p_cmd": [],          # p after update (returned by step)
    "u0": [],             # first control input
    "Jxy": [],            # Jacobian at linearisation point (inside MPC)
    "X_pred": [],         # MPC predicted horizon tip positions (linear)
    "U_seq": [],          # MPC planned horizon inputs
    # linearisation accuracy metrics
    "x_lin_1step": [],    # x_prev + J*(p_cmd - p_prev)
    "x_true_1step": [],   # forward(p_cmd)
    "lin_err_1step": [],  # ||x_true - x_lin||
    "dp": [],             # p_cmd - p_prev
    "dp_norm": [],        # ||dp||
    # horizon accuracy
    "horizon_max_err": [],
    "horizon_mean_err": [],
    "path_err_mm": [],      # ||x_now - x_ref0|| in mm
    "i_closest": [],   # [ex, ey] in mm (component errors)
}

# to compute dp we need previous p (before update)
p_prev = mpc_xy.p.copy()   # this is set after set_initial_params()

for k in range(40):
    x_now_prev = mpc_xy.x.copy()

    # take a window of Np points from the path
    xref_seq = path[idx : idx + mpc_xy.Np]

    # if near the end, pad with the last point
    if xref_seq.shape[0] < mpc_xy.Np:
        last = path[-1]
        pad = np.tile(last, (mpc_xy.Np - xref_seq.shape[0], 1))
        xref_seq = np.vstack([xref_seq, pad])

    # --- MPC step (UNCHANGED) ---
    p_cmd, x_now, info = mpc_xy.step(xref_seq)
    dmin, x_closest, i_closest = closest_point_error(path, x_now)
    log["path_err_mm"].append(1e3 * dmin)
    log["i_closest"].append(i_closest)
    # ----------------------------
    # Linearisation accuracy (1-step)
    # ----------------------------
    # MPC linearised around p_prev (mpc_xy.p before update)
    # so we test: x_lin = x_prev + J(p_prev)*(p_cmd - p_prev)
    dp = (p_cmd - p_prev).reshape(-1)
    J_prev = info["Jxy"]                     # J evaluated at p_prev inside step()
    x_lin_1step = x_now_prev + J_prev @ dp   # predicted tip from linear model
    x_true_1step = forward_tip(p_cmd)        # nonlinear true tip
    lin_err_1step = np.linalg.norm(x_true_1step - x_lin_1step)

    # ----------------------------
    # Horizon linear vs nonlinear accuracy
    # ----------------------------
    # info["X_pred"] is the MPC linear predicted tip horizon from its internal model.
    # We can reconstruct the parameter sequence from U_seq and compare forward_tip.
    horizon_max_err = np.nan
    horizon_mean_err = np.nan

    if info["U_seq"] is not None and np.all(np.isfinite(info["U_seq"])) and info["U_seq"].shape[0] == mpc_xy.Np:
        U_seq = info["U_seq"]  # (Np,4)
        p0 = p_prev.copy()

        # reconstruct parameter sequence: p_i = p0 + dt * sum_{j<=i} u_j
        p_seq = []
        p_running = p0.copy()
        for i in range(mpc_xy.Np):
            p_running = p_running + U_seq[i] * mpc_xy.dt
            # clamp like controller
            p_running = np.minimum(np.maximum(p_running, mpc_xy.p_min), mpc_xy.p_max)
            p_seq.append(p_running.copy())
        p_seq = np.array(p_seq)  # (Np,4)

        # nonlinear predicted horizon
        X_true_h = np.array([forward_tip(p) for p in p_seq])   # (Np,2)

        # linear horizon from MPC
        X_lin_h = info["X_pred"]

        if np.all(np.isfinite(X_lin_h)) and X_lin_h.shape == X_true_h.shape:
            herr = np.linalg.norm(X_true_h - X_lin_h, axis=1)
            horizon_max_err = float(np.max(herr))
            horizon_mean_err = float(np.mean(herr))

    # ----------------------------
    # Advance waypoint index
    # ----------------------------
    if np.linalg.norm(x_now - path[idx]) < 1e-3:  # 1 mm tolerance
        idx = min(idx + 1, M - 1)

    # ----------------------------
    # Log everything
    # ----------------------------
    log["k"].append(k)
    log["idx"].append(idx)
    log["x_now_prev"].append(x_now_prev.copy())
    log["x_now"].append(x_now.copy())
    log["x_ref0"].append(xref_seq[0].copy())
    log["xref_seq"].append(xref_seq.copy())
    log["p_cmd"].append(p_cmd.copy())
    log["u0"].append(info["u0"].copy())
    log["Jxy"].append(info["Jxy"].copy())
    log["X_pred"].append(info["X_pred"].copy())
    log["U_seq"].append(info["U_seq"].copy())

    log["x_lin_1step"].append(x_lin_1step.copy())
    log["x_true_1step"].append(x_true_1step.copy())
    log["lin_err_1step"].append(float(lin_err_1step))
    log["dp"].append(dp.copy())
    log["dp_norm"].append(float(np.linalg.norm(dp)))

    log["horizon_max_err"].append(horizon_max_err)
    log["horizon_mean_err"].append(horizon_mean_err)

    x_ref_now = xref_seq[0]  # first point in the horizon (current waypoint target)
    debug_step_tipxy(k, x_ref_now, xref_seq, p_cmd, x_now, info, print_horizon=6)
    # update p_prev for next iteration
    p_prev = p_cmd.copy()

print("Final tip:", x_now, "Target:", x_target)
# ----------------------------
# Convert logs to arrays
# ----------------------------
x_now_arr = np.array(log["x_now"])           # (T,2)
x_ref0_arr = np.array(log["x_ref0"])         # (T,2)
lin_err_1 = np.array(log["lin_err_1step"])   # (T,)
dp_norm = np.array(log["dp_norm"])           # (T,)

hmax = np.array(log["horizon_max_err"], dtype=float)
hmean = np.array(log["horizon_mean_err"], dtype=float)

# ----------------------------
# 1) Tip trajectory vs path
# ----------------------------
plt.figure()
plt.plot(path[:, 0], path[:, 1], "--", label="path")
plt.plot(x_now_arr[:, 0], x_now_arr[:, 1], "-o", label="actual tip")
plt.plot(x_ref0_arr[:, 0], x_ref0_arr[:, 1], "-x", label="ref[0] each step")
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Tip trajectory vs reference/path")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ----------------------------
# 2) 1-step linearisation error vs time
# ----------------------------
plt.figure()
plt.plot(1e3 * lin_err_1, "-o")
plt.xlabel("MPC step k")
plt.ylabel("||x_true - x_lin|| (mm)")
plt.title("1-step linearisation error (Jacobian accuracy)")
plt.grid(True)
plt.tight_layout()

# ----------------------------
# 3) Linearisation error vs step size ||dp||
# ----------------------------
plt.figure()
plt.plot(dp_norm, 1e3 * lin_err_1, "o")
plt.xlabel("||dp||")
plt.ylabel("linearisation error (mm)")
plt.title("1-step linearisation error vs step size")
plt.grid(True)
plt.tight_layout()
path_err_mm = np.array(log["path_err_mm"])

plt.figure()
plt.plot(path_err_mm, "-o")
plt.xlabel("MPC step k")
plt.ylabel("Min distance to path (mm)")
plt.title("Path tracking error (closest waypoint distance)")
plt.grid(True)
plt.tight_layout()
# ----------------------------
# 4) Horizon linear vs nonlinear mismatch (optional)
# ----------------------------
plt.figure()
plt.plot(hmax, "-o", label="max horizon err")
plt.plot(hmean, "-o", label="mean horizon err")
plt.xlabel("MPC step k")
plt.ylabel("||X_true_h - X_lin_h|| (m)")
plt.title("Horizon linearisation error (linear MPC prediction vs nonlinear forward)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()