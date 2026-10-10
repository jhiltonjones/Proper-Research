"""
Horizon sweep sanity test for MPCControllerTipXY.

Goal:
    Run the same controller code on a known, simple tracking problem.
    Sweep prediction horizon Np and check whether tracking improves.

This script assumes your controller class is importable as:

    from proper_research.simulation_controller.mpc_controller_tip_xy import MPCControllerTipXY

Change that import if your file/module name is different.
"""

import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import types
from proper_research.simulation_controller.controller import MPCControllerTipXY


class DoubleIntegratorForwardTip:
    def start_step(self):
        pass

    def __call__(self, p, commit=False):
        p = np.asarray(p, dtype=float).reshape(8)

        px = p[0]
        py = p[1]
        vx = p[2]
        vy = p[7]

        return np.array([px, py, vx, vy], dtype=float)


class DoubleIntegratorFullStateJacobian:
    def __init__(self, dt):
        self.dt = float(dt)

    def __call__(self, p):
        dt = self.dt

        B = np.zeros((4, 7), dtype=float)

        B[0, 0] = 0.5 * dt**2
        B[1, 1] = 0.5 * dt**2
        B[2, 0] = dt
        B[3, 1] = dt

        return B

    def get_last_diag(self):
        return {}

class DoubleIntegratorJacobian:
    """
    Prediction Jacobian for a 2D double integrator.

    Internal benchmark state:
        p[0] = x position
        p[1] = y position
        p[2] = x velocity
        p[7] = y velocity

    Control:
        u[0] = x acceleration
        u[1] = y acceleration

    Output:
        x_out = [p[0], p[1], 0]
    """

    def __init__(self, dt):
        self.dt = float(dt)

    def __call__(self, p):
        B = np.zeros((3, 7), dtype=float)

        # One-step effect of acceleration on position.
        B[0, 0] = 0.5 * self.dt**2
        B[1, 1] = 0.5 * self.dt**2

        return B

    def get_last_diag(self):
        return {}
class FirstOrderLagJacobian:
    """
    Prediction model for a first-order lag benchmark.

    Plant:
        x[k+1] = a*x[k] + b*u[k]

    We track x = [p0, p1, 0].
    Controls u[0], u[1] drive x/y.
    """

    def __init__(self, dt, tau=0.5, gain=1.0):
        self.dt = float(dt)
        self.tau = float(tau)
        self.gain = float(gain)

        self.a = float(np.exp(-self.dt / self.tau))
        self.b = float(self.gain * (1.0 - self.a))

    def __call__(self, p):
        B = np.zeros((3, 7), dtype=float)
        B[0, 0] = self.b
        B[1, 1] = self.b
        return B

    def get_last_diag(self):
        return {
            "bench_a": self.a,
            "bench_b": self.b,
            "bench_tau": self.tau,
        }
def patch_first_order_lag_rollout(ctrl, *, dt, tau=0.5, gain=1.0):
    """
    Replace only this controller instance's rollout with a simple first-order lag plant.

    This does not modify controller source code.

    Plant:
        x_next = a*x + b*u

    where:
        a = exp(-dt/tau)
        b = gain * (1 - a)
    """

    a = float(np.exp(-dt / tau))
    b = float(gain * (1.0 - a))

    def _apply_control_sequence_first_order_lag(self, U_seq, n_apply):
        U_seq = np.asarray(U_seq, dtype=float).reshape(self.Np, self.m)
        n_apply = int(np.clip(n_apply, 1, self.Np))

        applied = []

        for j in range(n_apply):
            u = U_seq[j].copy()
            applied.append(u)

            x_old = self.x.copy()

            x_new = x_old.copy()
            x_new[0] = a * x_old[0] + b * u[0]
            x_new[1] = a * x_old[1] + b * u[1]
            x_new[2] = 0.0

            self.x = x_new

            # Keep native 8D pose consistent with benchmark output.
            self.p[0] = x_new[0]
            self.p[1] = x_new[1]
            self.p[2] = 0.0

            # Preserve valid identity orientation and positive length.
            self.p[3] = 1.0
            self.p[4] = 0.0
            self.p[5] = 0.0
            self.p[6] = 0.0
            self.p[7] = max(self.p[7], 0.05)

        return {
            "bench_mode": "first_order_lag",
            "bench_a": a,
            "bench_b": b,
            "bench_tau": tau,
            "u_applied": np.asarray(applied),
        }

    ctrl._apply_control_sequence = types.MethodType(
        _apply_control_sequence_first_order_lag,
        ctrl,
    )
def make_smooth_step_reference_full_state(T, dt, t_start=0.5, duration=1.5):
    t = np.arange(T) * dt

    ref = np.zeros((T, 4), dtype=float)

    target_x = 0.05
    target_y = -0.03

    s = np.clip((t - t_start) / duration, 0.0, 1.0)

    # Smoothstep: zero velocity at start and end.
    h = 3.0 * s**2 - 2.0 * s**3

    # Derivative of smoothstep.
    hdot = (6.0 * s - 6.0 * s**2) / duration
    hdot[(t < t_start) | (t > t_start + duration)] = 0.0

    ref[:, 0] = target_x * h
    ref[:, 1] = target_y * h

    ref[:, 2] = target_x * hdot
    ref[:, 3] = target_y * hdot

    return t, ref
def patch_double_integrator_rollout(ctrl, *, dt):
    """
    Replace this controller instance's rollout with a 2D double integrator.

    State:
        p[0] = x position
        p[1] = y position
        p[2] = x velocity
        p[7] = y velocity

    Control:
        u[0] = x acceleration
        u[1] = y acceleration
    """

    dt = float(dt)

    def _apply_control_sequence_double_integrator(self, U_seq, n_apply):
        U_seq = np.asarray(U_seq, dtype=float).reshape(self.Np, self.m)
        n_apply = int(np.clip(n_apply, 1, self.Np))

        applied = []

        for j in range(n_apply):
            u = U_seq[j].copy()
            applied.append(u)

            px = float(self.p[0])
            py = float(self.p[1])
            vx = float(self.p[2])
            vy = float(self.p[7])

            ax = float(u[0])
            ay = float(u[1])

            px_next = px + dt * vx + 0.5 * dt**2 * ax
            py_next = py + dt * vy + 0.5 * dt**2 * ay

            vx_next = vx + dt * ax
            vy_next = vy + dt * ay

            self.p[0] = px_next
            self.p[1] = py_next
            self.p[2] = vx_next
            self.p[7] = vy_next

            # Preserve valid pose orientation.
            self.p[3] = 1.0
            self.p[4] = 0.0
            self.p[5] = 0.0
            self.p[6] = 0.0

            if self.n == 4:
                self.x = np.array([px_next, py_next, vx_next, vy_next], dtype=float)
            else:
                self.x = np.array([px_next, py_next, 0.0], dtype=float)

        return {
            "bench_mode": "double_integrator",
            "u_applied": np.asarray(applied),
        }

    ctrl._apply_control_sequence = types.MethodType(
        _apply_control_sequence_double_integrator,
        ctrl,
    )
# ---------------------------------------------------------------------
# 1. Known benchmark model
# ---------------------------------------------------------------------
# We use a simple 2D single-integrator benchmark:
#
#     p_{k+1} = p_k + dt * u_k
#     x_k     = p_k
#
# where:
#     p = [x_position, y_position]
#     u = [x_velocity_command, y_velocity_command]
#
# This is deliberately simple. A working MPC should track this very well,
# and longer prediction horizons should usually improve tracking for
# time-varying references until performance saturates.
# ---------------------------------------------------------------------

class Pose8ForwardTip:
    def start_step(self):
        pass

    def __call__(self, p, commit=False):
        p = np.asarray(p, dtype=float).reshape(8)
        return np.array([p[0], p[1], 0.0], dtype=float)


class Pose8Jacobian:
    """
    2-output, 7-input Jacobian for the simple benchmark.

    We assume the first two controls approximately move p[0] and p[1]:

        u[0] -> x motion
        u[1] -> y motion

    The remaining controls have no effect in this test.
    """

    def __init__(self, dt):
        self.dt = float(dt)

    def __call__(self, p):
        B = np.zeros((3, 7), dtype=float)
        B[0, 0] = self.dt
        B[1, 1] = self.dt

        return B

    def get_last_diag(self):
        return {}
import numpy as np
from scipy.optimize import minimize


def double_integrator_AB(dt):
    A = np.array(
        [
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    B = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )

    return A, B


def condensed_prediction_matrices(A, B, Np):
    n = A.shape[0]
    m = B.shape[1]

    Sx = np.zeros((Np * n, n))
    Su = np.zeros((Np * n, Np * m))

    for i in range(Np):
        Sx[i*n:(i+1)*n, :] = np.linalg.matrix_power(A, i + 1)

        for j in range(i + 1):
            Su[i*n:(i+1)*n, j*m:(j+1)*m] = (
                np.linalg.matrix_power(A, i - j) @ B
            )

    return Sx, Su


def oracle_double_integrator_u0(
    *,
    x0,
    u_prev=None,
    Np,
    dt,
    q=(100.0, 100.0, 20.0, 20.0),
    r=(1e-1, 1e-1),
    rd=(1e-1, 1e-1),
    u_max=0.5,
):
    """
    Independent condensed LTI MPC oracle.

    Dynamics:
        x[k+1] = A x[k] + B u[k]

    Cost:
        sum x_i.T Q x_i + u_i.T R u_i + du_i.T Rd du_i

    Bounds:
        -u_max <= u_i <= u_max
    """

    x0 = np.asarray(x0, dtype=float).reshape(4)
    u_prev = np.zeros(2) if u_prev is None else np.asarray(u_prev, dtype=float).reshape(2)

    A, B = double_integrator_AB(dt)
    Sx, Su = condensed_prediction_matrices(A, B, Np)

    Qbar = np.kron(np.eye(Np), np.diag(q))
    Rbar = np.kron(np.eye(Np), np.diag(r))
    Rdbar = np.kron(np.eye(Np), np.diag(rd))

    m = 2

    # Difference matrix: du0 = u0 - u_prev, dui = ui - u{i-1}
    D = np.zeros((Np * m, Np * m))
    for i in range(Np):
        D[i*m:(i+1)*m, i*m:(i+1)*m] = np.eye(m)
        if i > 0:
            D[i*m:(i+1)*m, (i-1)*m:i*m] = -np.eye(m)

    E = np.zeros((Np * m, m))
    E[:m, :] = np.eye(m)

    # Objective as 0.5*z.T@H@z + f.T@z
    H = 2.0 * (Su.T @ Qbar @ Su + Rbar + D.T @ Rdbar @ D)
    f = 2.0 * (Su.T @ Qbar @ Sx @ x0 - D.T @ Rdbar @ E @ u_prev)

    def obj(z):
        return 0.5 * z @ H @ z + f @ z

    def grad(z):
        return H @ z + f

    z0 = np.zeros(Np * m)

    res = minimize(
        obj,
        z0,
        jac=grad,
        method="L-BFGS-B",
        bounds=[(-u_max, u_max)] * (Np * m),
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 200},
    )

    if not res.success:
        raise RuntimeError(f"Oracle QP failed: {res.message}")

    return res.x[:2], res.x.reshape(Np, 2)

# ---------------------------------------------------------------------
# 2. Reference generation
# ---------------------------------------------------------------------

def rollout_double_integrator_sequence(x0, U, dt):
    x = np.asarray(x0, dtype=float).reshape(4).copy()
    X = []

    for u in np.asarray(U, dtype=float):
        ax = float(u[0])
        ay = float(u[1])

        px, py, vx, vy = x

        x = np.array(
            [
                px + dt * vx + 0.5 * dt**2 * ax,
                py + dt * vy + 0.5 * dt**2 * ay,
                vx + dt * ax,
                vy + dt * ay,
            ],
            dtype=float,
        )

        X.append(x.copy())

    return np.asarray(X)
def make_reference(T, dt):
    t = np.arange(T) * dt

    # Starts at zero and remains small.
    x_ref = 0.04 * np.sin(2.0 * np.pi * 0.25 * t)
    y_ref = 0.03 * np.sin(2.0 * np.pi * 0.25 * t)
    z_ref = np.zeros_like(t)

    ref = np.column_stack([x_ref, y_ref, z_ref])
    return t, ref
def make_step_reference(T, dt):
    t = np.arange(T) * dt

    ref = np.zeros((T, 3), dtype=float)

    ref[t >= 1.0, 0] = 0.05
    ref[t >= 1.0, 1] = -0.03

    return t, ref
def attach_reference_to_controller(ctrl, ref):
    """
    Attach a simple time-indexed reference trajectory to the controller.

    This bypasses the project-specific ReferenceMixin path logic and makes
    the benchmark use references [k+1, k+2, ..., k+Np] at each MPC step.
    """

    ref = np.asarray(ref, dtype=float)

    if ref.ndim != 2:
        raise ValueError(f"ref must be 2D, got shape {ref.shape}")

    if ref.shape[1] != ctrl.n:
        raise ValueError(
            f"ref has output dimension {ref.shape[1]}, but controller expects {ctrl.n}"
        )

    # Existing aliases.
    ctrl.ref = ref
    ctrl.ref_path = ref
    ctrl.reference = ref
    ctrl.ref_xy = ref
    ctrl.X_ref = ref

    # Required by your QPBuilderMixin._build_reference_stack().
    ctrl.lumen_C = ref

    # Optional useful aliases in case other mixin methods look for them.
    ctrl.lumen = ref
    ctrl.lumen_xy = ref

    ctrl._test_ref_k = 0

    def _select_reference_indices_for_test(self, x0):
        k = int(self._test_ref_k)

        idx = np.arange(k + 1, k + 1 + self.Np, dtype=int)
        idx = np.clip(idx, 0, len(self.lumen_C) - 1)

        self.idx_ref_last = idx.copy()
        self.i_ref_last = int(idx[0])
        self.i_closest_last = int(idx[0])
        self.dist_to_ref_last = float(
            np.linalg.norm(x0 - self.lumen_C[idx[0]])
        )

        return idx

    ctrl._select_reference_indices = types.MethodType(
        _select_reference_indices_for_test,
        ctrl,
    )

# ---------------------------------------------------------------------
# 3. One closed-loop simulation
# ---------------------------------------------------------------------


def run_closed_loop_once(
    *,
    Np,
    ref,
    dt,
    T,
    solver_mode="lti",
    benchmark_mode="single_integrator",
    u_max_value=0.08,
    w_track=None,
    w_u=None,
    w_du=None,
    p0_override=None,
    jacobian_scale=1.0,
):
    """
    Run one closed-loop MPC simulation for a given prediction horizon.

    Supported benchmark modes:
        - "single_integrator"
        - "first_order_lag"
        - "double_integrator"
        - "double_integrator_full_state"
    """

    # ------------------------------------------------------------
    # 1. Select benchmark model dimensions and functions
    # ------------------------------------------------------------

    if benchmark_mode == "single_integrator":
        n_out = 3
        forward_tip_fn = Pose8ForwardTip()
        Jxy_fn = Pose8Jacobian(dt=dt)

        A_bench = np.eye(n_out)

        if w_track is None:
            w_track = (100.0, 100.0, 0.0)

    elif benchmark_mode == "first_order_lag":
        n_out = 3
        forward_tip_fn = Pose8ForwardTip()
        Jxy_fn = FirstOrderLagJacobian(
            dt=dt,
            tau=1.5,
            gain=1.0,
        )

        A_bench = np.eye(n_out)
        A_bench[0, 0] = Jxy_fn.a
        A_bench[1, 1] = Jxy_fn.a
        A_bench[2, 2] = 1.0

        if w_track is None:
            w_track = (100.0, 100.0, 0.0)

    elif benchmark_mode == "double_integrator":
        # Old position-output-only version.
        # This is useful, but velocity is hidden from the MPC.
        n_out = 3
        forward_tip_fn = Pose8ForwardTip()
        Jxy_fn = DoubleIntegratorJacobian(dt=dt)

        A_bench = np.eye(n_out)

        if w_track is None:
            w_track = (100.0, 100.0, 0.0)

    elif benchmark_mode == "double_integrator_full_state":
        # Proper double-integrator MPC benchmark.
        # MPC output/state is [px, py, vx, vy].
        n_out = 4
        forward_tip_fn = DoubleIntegratorForwardTip()
        Jxy_fn = DoubleIntegratorFullStateJacobian(dt=dt)

        A_bench = np.array(
            [
                [1.0, 0.0, dt,  0.0],
                [0.0, 1.0, 0.0, dt ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        if w_track is None:
            # Penalise position strongly, velocity weakly.
            w_track = (100.0, 100.0, 1.0, 1.0)

    else:
        raise ValueError(f"Unknown benchmark_mode: {benchmark_mode!r}")

    if w_u is None:
        w_u = (1e-3,) * 7

    if w_du is None:
        w_du = (1e-3,) * 7

    # ------------------------------------------------------------
    # 2. Bounds
    # ------------------------------------------------------------

    p_min_bench = -1e3 * np.ones(8)
    p_max_bench =  1e3 * np.ones(8)

    # Only impose positive p[7] when p[7] is acting as a fake length.
    # In the full-state double integrator, p[7] is vy, so it must be
    # allowed to be negative.
    if benchmark_mode != "double_integrator_full_state":
        p_min_bench[7] = 1e-4
        p_max_bench[7] = 1.0

    # ------------------------------------------------------------
    # 3. Build controller
    # ------------------------------------------------------------

    ctrl = MPCControllerTipXY(
        Jxy_fn=Jxy_fn,
        forward_tip_fn=forward_tip_fn,
        dt=dt,
        Np=Np,

        n_out=n_out,
        n_p=8,
        n_u=7,

        model_mode="lti",
        solver_mode=solver_mode,

        w_xy=w_track,
        w_u=w_u,
        w_du=w_du,

        u_max=np.array(
            [
                u_max_value,
                u_max_value,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=float,
        ),

        p_min=p_min_bench,
        p_max=p_max_bench,

        enable_hard_epm_tip_clearance=False,
        use_trust_region=False,
        # Make residual beam-specific hard-clearance constraints non-binding.
        epm_tip_hard_min_m=-1e3,

        dL_index=6,
        dL_back_max=1e3,
        dL_fwd_max=1e3,

        qp_reg=1e-9,
    )

    # Set the prediction state matrix.
    ctrl.A = A_bench.copy()
    ctrl.track_all_outputs = True
    ctrl.use_affine_matching = False
    # Patch benchmark-specific closed-loop plant rollout.
    if benchmark_mode == "first_order_lag":
        patch_first_order_lag_rollout(
            ctrl,
            dt=dt,
            tau=1.5,
            gain=1.0,
        )

    elif benchmark_mode in ("double_integrator", "double_integrator_full_state"):
        patch_double_integrator_rollout(ctrl, dt=dt)

    # ------------------------------------------------------------
    # 4. Reference and initial state
    # ------------------------------------------------------------

    attach_reference_to_controller(ctrl, ref)

    p0 = np.zeros(8, dtype=float)

    # Valid identity orientation.
    p0[3] = 1.0

    if benchmark_mode == "double_integrator_full_state":
        # p[2] = vx, p[7] = vy.
        p0[2] = 0.0
        p0[7] = 0.0
    else:
        # Fake positive length state for older benchmark modes.
        p0[7] = 0.05

    if p0_override is not None:
        p0 = np.asarray(p0_override, dtype=float).reshape(8).copy()

    ctrl.set_initial_params(p0.copy())

    if benchmark_mode == "double_integrator_full_state":
        x0_oracle = np.array(
            [ctrl.p[0], ctrl.p[1], ctrl.p[2], ctrl.p[7]],
            dtype=float,
        )

        u0_oracle, U_oracle = oracle_double_integrator_u0(
            x0=x0_oracle,
            u_prev=np.zeros(2),
            Np=Np,
            dt=dt,
            q=w_track,
            r=(w_u[0], w_u[1]),
            rd=(w_du[0], w_du[1]),
            u_max=u_max_value,
        )

        print("oracle u0[:2]:", u0_oracle)

    if benchmark_mode == "double_integrator_full_state":
        ctrl.ref_stage_weights = np.ones(ctrl.Np)
    # ------------------------------------------------------------
    # 5. Closed-loop simulation
    # ------------------------------------------------------------

    xs = []
    us = []
    refs_used = []
    statuses = []
    infeasible_flags = []
    solve_times = []

    u0_oracle = None
    U_oracle = None

    for k in range(T):
        ctrl._test_ref_k = k

        x_meas = ctrl.x.copy()

        if k == 0 and benchmark_mode == "double_integrator_full_state":
            x0_oracle = np.asarray(x_meas, dtype=float).reshape(4)

            print("x0_oracle:", x0_oracle)
            print("ctrl.p:", ctrl.p)
            print("ctrl.x:", ctrl.x)

            u0_oracle, U_oracle = oracle_double_integrator_u0(
                x0=x0_oracle,
                u_prev=np.zeros(2),
                Np=Np,
                dt=dt,
                q=w_track,
                r=(w_u[0], w_u[1]),
                rd=(w_du[0], w_du[1]),
                u_max=u_max_value,
            )

            print("oracle U[:,:2]:")
            print(U_oracle[:, :2])
            print("oracle u0[:2]:", u0_oracle)

        tic = time.perf_counter()
        p_next, x_next, info = ctrl.step(
            x_meas=x_meas,
            rollout_steps=1,
            solver_mode=solver_mode,
        )
        toc = time.perf_counter()
        if k == 0 and benchmark_mode == "double_integrator_full_state":
            U_ctrl = np.asarray(info.get("U_opt", []), dtype=float)

            if U_ctrl.size:
                U_ctrl = U_ctrl.reshape(ctrl.Np, ctrl.m)

                X_oracle = rollout_double_integrator_sequence(x0_oracle, U_oracle[:, :2], dt)
                X_ctrl = rollout_double_integrator_sequence(x0_oracle, U_ctrl[:, :2], dt)

                print("oracle predicted X:")
                print(X_oracle)

                print("controller predicted X from controller U:")
                print(X_ctrl)

                print("controller U[:,:2]:")
                print(U_ctrl[:, :2])
        if k == 0 and benchmark_mode == "double_integrator_full_state":
            u0_controller = np.asarray(info["u0"], dtype=float)[:2]
            u0_error = u0_controller - u0_oracle

            print("controller u0[:2]:", u0_controller)
            print("oracle     u0[:2]:", u0_oracle)
            print("u0 error:", u0_error)
            print("u0 error norm:", np.linalg.norm(u0_error))

            if np.linalg.norm(u0_error) > 1e-4:
                print("WARNING: controller first move does not match oracle")
        if k == 0 and int(info.get("infeasible", 0)):
            print("\nFirst-step infeasible diagnostic:")
            print("status:", info.get("status"))

            for key in [
                "constraint_issue_guess",
                "constraint_max_vio_guess",
                "constraint_num_vio_guess",
                "failed_constraint_issue_guess",
                "failed_constraint_max_vio_guess",
                "failed_constraint_num_vio_guess",
                "valid_H_mpc",
                "lambda_min_H_mpc",
                "num_negative_H_mpc",
                "num_near_zero_H_mpc",
            ]:
                if key in info:
                    print(f"{key}: {info[key]}")

            print("p_now:", info.get("p_now"))
            print("x_now:", info.get("x_now"))
            print("u0:", info.get("u0"))

        xs.append(x_next.copy())
        us.append(info["u0"].copy())
        refs_used.append(ref[min(k + 1, len(ref) - 1)].copy())
        statuses.append(info.get("status", "unknown"))
        infeasible_flags.append(int(info.get("infeasible", 0)))
        solve_times.append(toc - tic)

    xs = np.asarray(xs)
    us = np.asarray(us)
    active_u = us[:, :2]
    sat_tol = 1e-5
    sat = np.abs(active_u) >= (u_max_value - sat_tol)
    sat_fraction = float(np.mean(sat))
    refs_used = np.asarray(refs_used)
    solve_times = np.asarray(solve_times)
    err = xs - refs_used

    if benchmark_mode == "double_integrator_full_state":
        pos_err = err[:, :2]
        vel_err = err[:, 2:4]

        pos_err_norm = np.linalg.norm(pos_err, axis=1)
        vel_err_norm = np.linalg.norm(vel_err, axis=1)

        err_norm = pos_err_norm
    else:
        err_norm = np.linalg.norm(err, axis=1)

    result = {
        "Np": int(Np),
        "benchmark_mode": benchmark_mode,
        "xs": xs,
        "us": us,
        "refs": refs_used,
        "err": err,
        "err_norm": err_norm,
        "rms_error": float(np.sqrt(np.mean(err_norm**2))),
        "mean_error": float(np.mean(err_norm)),
        "max_error": float(np.max(err_norm)),
        "final_error": float(err_norm[-1]),
        "mean_u_norm": float(np.mean(np.linalg.norm(us, axis=1))),
        "max_u_norm": float(np.max(np.linalg.norm(us, axis=1))),
        "infeasible_rate": float(np.mean(infeasible_flags)),
        "mean_solve_time_ms": float(1000.0 * np.mean(solve_times)),
        "saturation_fraction": sat_fraction,
        "statuses": statuses,
    }
    if benchmark_mode == "double_integrator_full_state":
        result["rms_position_error"] = float(np.sqrt(np.mean(pos_err_norm**2)))
        result["max_position_error"] = float(np.max(pos_err_norm))
        result["rms_velocity_error"] = float(np.sqrt(np.mean(vel_err_norm**2)))
        result["max_velocity_error"] = float(np.max(vel_err_norm))
    return result

def debug_first_qp_constraints(ctrl):
    """
    Build the first QP directly and inspect constraint bounds before OSQP.

    This is a test-harness diagnostic only. It does not modify controller code.
    """

    p0 = ctrl.p.copy()
    x0 = ctrl.x.copy()

    if ctrl.U_warm is not None and ctrl.U_warm.size == ctrl.Np * ctrl.m:
        U_init = ctrl.U_warm.reshape(ctrl.Np, ctrl.m).copy()
    else:
        U_init = ctrl._make_initial_U_guess()

    idx_ref = ctrl._select_reference_indices(x0)

    H, f, A, l, u, dbg = ctrl._build_mpc_qp(
        p0=p0,
        x0=x0,
        U_guess=U_init,
        idx_ref=idx_ref,
    )

    l = np.asarray(l, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    print("\nQP preflight diagnostic")
    print("-----------------------")
    print("H shape:", H.shape)
    print("A shape:", A.shape)
    print("l shape:", l.shape)
    print("u shape:", u.shape)

    print("finite H:", np.all(np.isfinite(H.data if hasattr(H, "data") else H)))
    print("finite f:", np.all(np.isfinite(f)))
    print("finite A:", np.all(np.isfinite(A.data if hasattr(A, "data") else A)))
    print("finite l:", np.all(np.isfinite(l)))
    print("finite u:", np.all(np.isfinite(u)))

    bad_bounds = np.where(l > u)[0]
    print("rows with l > u:", bad_bounds.size)

    if bad_bounds.size:
        print("\nFirst bad bound rows:")
        for row in bad_bounds[:20]:
            print(f"row {row}: l={l[row]}, u={u[row]}")

    U_vec = U_init.reshape(-1)

    residual = A @ U_vec
    residual = np.asarray(residual, dtype=float).reshape(-1)

    vio_lower = l - residual
    vio_upper = residual - u
    vio = np.maximum(vio_lower, vio_upper)

    violated = np.where(vio > 1e-8)[0]

    print("initial guess violated rows:", violated.size)

    if violated.size:
        worst = violated[np.argsort(vio[violated])[-20:]][::-1]

        print("\nWorst initial-guess constraint violations:")
        for row in worst:
            print(
                f"row {row}: "
                f"A@U={residual[row]: .6e}, "
                f"l={l[row]: .6e}, "
                f"u={u[row]: .6e}, "
                f"vio={vio[row]: .6e}"
            )

    print("\nScalar dbg values:")
    for key, value in sorted(dbg.items()):
        if np.isscalar(value) or isinstance(value, str):
            print(f"{key}: {value}")

    return H, f, A, l, u, dbg
# ---------------------------------------------------------------------
# 4. Horizon sweep
# ---------------------------------------------------------------------
def make_zero_reference_full_state(T, dt):
    t = np.arange(T) * dt
    ref = np.zeros((T, 4), dtype=float)
    return t, ref

def run_horizon_sweep():
    dt = 0.05
    T = 500

    # t, ref = make_zero_reference_full_state(T + 100, dt)
    t, ref = make_smooth_step_reference_full_state(T + 100, dt, 0.5, 0.25)

    horizons = [1, 2, 3, 5, 10, 20]

    results = []
    p0_reg = np.zeros(8, dtype=float)
    p0_reg[3] = 1.0
    p0_reg[7] = 0.0
    for Np in horizons:
        print(f"\nRunning horizon Np={Np}...")
        result = run_closed_loop_once(
            Np=Np,
            ref=ref,
            dt=dt,
            T=T,
            solver_mode="sqp_full",
            benchmark_mode="double_integrator_full_state",
            u_max_value=0.05,
            w_track=(100.0, 100.0, 20.0, 20.0),
            w_u=(1e-1,) * 7,
            w_du = (1e-1,) * 7,
            p0_override=p0_reg.copy(),
            jacobian_scale=1.0,
        )

        results.append(result)

        print(
            f"Np={Np:>2d} | "
            f"RMS error={result['rms_error']:.6f} | "
            f"max error={result['max_error']:.6f} | "
            f"infeasible={100.0 * result['infeasible_rate']:.1f}% | "
            f"solve={result['mean_solve_time_ms']:.3f} ms"
        )
        print("mean_u_norm:", result["mean_u_norm"])
        print("max_u_norm:", result["max_u_norm"])
        print("saturation_fraction:", result["saturation_fraction"])
        if "rms_velocity_error" in result:
            print("rms_velocity_error:", result["rms_velocity_error"])
        print("final_error:", result["final_error"])
        if "max_velocity_error" in result:
            print("max_velocity_error:", result["max_velocity_error"])
    return t[:T], results


# ---------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------


def plot_results(t, results):
    horizons = [r["Np"] for r in results]
    rms_errors = [r["rms_error"] for r in results]
    max_errors = [r["max_error"] for r in results]
    solve_times = [r["mean_solve_time_ms"] for r in results]

    plt.figure()
    plt.plot(horizons, rms_errors, marker="o")
    plt.xlabel("Prediction horizon Np")
    plt.ylabel("RMS tracking error")
    plt.title("MPC horizon sweep: RMS tracking error")
    plt.grid(True)

    plt.figure()
    plt.plot(horizons, max_errors, marker="o")
    plt.xlabel("Prediction horizon Np")
    plt.ylabel("Maximum tracking error")
    plt.title("MPC horizon sweep: max tracking error")
    plt.grid(True)

    plt.figure()
    plt.plot(horizons, solve_times, marker="o")
    plt.xlabel("Prediction horizon Np")
    plt.ylabel("Mean solve time [ms]")
    plt.title("MPC horizon sweep: solve time")
    plt.grid(True)

    # Plot trajectories for selected horizons.
    selected = [results[0], results[len(results) // 2], results[-1]]

    plt.figure()
    for r in selected:
        xs = r["xs"]
        plt.plot(xs[:, 0], xs[:, 1], label=f"Np={r['Np']}")

    ref = results[0]["refs"]
    plt.plot(ref[:, 0], ref[:, 1], "--", label="reference")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Closed-loop trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.figure()
    for r in selected:
        plt.plot(t, r["err_norm"], label=f"Np={r['Np']}")

    plt.xlabel("time [s]")
    plt.ylabel("tracking error norm")
    plt.title("Tracking error over time")
    plt.grid(True)
    plt.legend()

    plt.show()


# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------


if __name__ == "__main__":
    t, results = run_horizon_sweep()
    plot_results(t, results)