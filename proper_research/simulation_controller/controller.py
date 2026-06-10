import numpy as np

from proper_research.simulation_controller.prediction import seq_mat_lti, seq_mat_ltv, build_Du_matrix
from proper_research.simulation_controller.reference import ReferenceMixin
from proper_research.simulation_controller.rollout import RolloutMixin
from proper_research.simulation_controller.qp_builder import QPBuilderMixin
from proper_research.simulation_controller.osqp_solver import solve_qp_osqp


class MPCControllerTipXY(
    ReferenceMixin,
    RolloutMixin,
    QPBuilderMixin,
):
    """
    Multi-step MPC controller for magnetic beam tip tracking.

    Supported solver modes:
        - "lti": fixed first-stage Jacobian, one QP
        - "ltv_oneshot": LTV rollout, one QP
        - "sqp_full": repeated LTV linearise-solve-update SQP
    """

    def __init__(
        self,
        *,
        Jxy_fn,
        forward_tip_fn,
        dt=0.05,
        Np=10,
        w_xy=(5.0, 5.0, 0.0),
        w_u=None,
        w_du=None,
        u_max=None,
        p_min=None,
        p_max=None,
        N_sqp=3,
        n_out=3,
        n_p=8,
        n_u=7,
        model_mode="ltv",
        solver_mode="sqp_full",
        trust_radius=None,
        epm_tip_hard_min_m=0.011,
        enable_hard_epm_tip_clearance=True,
        dL_index=6,
        dL_back_max=0.002,
        dL_fwd_max=np.inf,
        qp_reg=1e-9,
        
    ):
        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn

        self.dt = float(dt)
        self.Np = int(Np)

        self.np = int(n_p)
        self.n = int(n_out)
        self.m = int(n_u)

        self.A = np.eye(self.n)

        self.model_mode = str(model_mode).lower()
        if self.model_mode not in ("lti", "ltv"):
            raise ValueError("model_mode must be 'lti' or 'ltv'.")

        self.solver_mode = str(solver_mode).lower()
        if self.solver_mode not in ("lti", "ltv_oneshot", "sqp_full"):
            raise ValueError(
                "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
            )

        self.N_sqp = int(N_sqp)
        self.sqp_tol_u = 1e-4
        self.sqp_tol_rel_u = 1e-3

        self.dL_index = int(dL_index)
        self.dL_back_max = float(dL_back_max)
        self.dL_fwd_max = float(dL_fwd_max)

        self.trust_radius = None if trust_radius is None else np.asarray(
            trust_radius,
            float,
        ).reshape(-1)

        self.enable_hard_epm_tip_clearance = bool(enable_hard_epm_tip_clearance)
        self.epm_tip_hard_min_m = float(epm_tip_hard_min_m)

        self.qp_reg = float(qp_reg)

        if w_u is None:
            w_u = (1e-3,) * self.m

        if w_du is None:
            w_du = (0.0,) * self.m

        w_xy = np.asarray(w_xy, float).reshape(-1)
        w_u = np.asarray(w_u, float).reshape(-1)
        w_du = np.asarray(w_du, float).reshape(-1)

        if w_xy.size != self.n:
            raise ValueError(f"w_xy must have length {self.n}, got {w_xy.size}.")

        if w_u.size != self.m:
            raise ValueError(f"w_u must have length {self.m}, got {w_u.size}.")

        if w_du.size != self.m:
            raise ValueError(f"w_du must have length {self.m}, got {w_du.size}.")

        self.Q = np.diag(w_xy)
        self.R = np.diag(w_u)
        self.Rd = np.diag(w_du)

        if u_max is None:
            u_max = np.full(self.m, np.inf)

        if p_min is None:
            p_min = -np.full(self.np, np.inf)

        if p_max is None:
            p_max = np.full(self.np, np.inf)

        self.u_max = np.asarray(u_max, float).reshape(-1)
        self.p_min = np.asarray(p_min, float).reshape(-1)
        self.p_max = np.asarray(p_max, float).reshape(-1)

        if self.u_max.size != self.m:
            raise ValueError(f"u_max must have length {self.m}.")

        if self.p_min.size != self.np:
            raise ValueError(f"p_min must have length {self.np}.")

        if self.p_max.size != self.np:
            raise ValueError(f"p_max must have length {self.np}.")

        self.p = None
        self.x = None
        self.d = np.zeros(self.n, float)

        self.U_warm = None

        self.Du = build_Du_matrix(self.Np, self.m)

        self.ref_stride_pts = 1
        self.ref_stage_weights = np.ones(self.Np)

        self.i_ref_last = 0
        self.idx_ref_last = None
        self.i_closest_last = None
        self.dist_to_ref_last = None
        self.use_affine_matching = True

    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.np,)

        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()
            self.x = np.asarray(
                self.forward_tip_fn(self.p, commit=False),
                float,
            ).reshape(self.n)
        else:
            self.x = np.asarray(
                self.forward_tip_fn(self.p),
                float,
            ).reshape(self.n)

        self.d = np.zeros(self.n, float)
        self.U_warm = None

    def _build_prediction_mats(self, p0, U_guess):
        """
        Returns:
            p_seq: post-control pose sequence [p1, ..., p_Np]
            Mx: stacked state-transition matrix
            Mc: stacked control matrix
            B0: first-stage input Jacobian
        """
        n, m, Np = self.n, self.m, self.Np

        if U_guess is None:
            U_guess = self._make_initial_U_guess()

        U_guess = np.asarray(U_guess, float).reshape(Np, m)

        if self.model_mode == "lti":
            B0 = np.asarray(self.Jxy_fn(p0), float)

            if B0.shape != (n, m):
                raise ValueError(
                    f"Jxy_fn returned {B0.shape}, expected {(n, m)}."
                )

            Mx, Mc = seq_mat_lti(self.A, B0, Np)
            p_seq = self._p_seq_from_U(p0, U_guess)

            return p_seq, Mx, Mc, B0

        if self.model_mode == "ltv":
            p_nodes = self._p_nodes_from_U(p0, U_guess)
            p_seq = p_nodes[1:]

            B_list = []

            for k in range(Np):
                Bk = np.asarray(self.Jxy_fn(p_nodes[k]), float)

                if Bk.shape != (n, m):
                    raise ValueError(
                        f"Jxy_fn at stage {k} returned {Bk.shape}, "
                        f"expected {(n, m)}."
                    )

                B_list.append(Bk)

            Mx, Mc = seq_mat_ltv(self.A, B_list)
            B0 = B_list[0]

            return p_seq, Mx, Mc, B0

        raise ValueError(f"Unknown model_mode: {self.model_mode!r}.")

    def _solve_mpc_sequence(self, *, p0, x0, U_init, idx_ref, solver_mode):
        """
        Solve one MPC problem.

        solver_mode:
            "lti"          -> LTI, one QP
            "ltv_oneshot"  -> LTV, one QP
            "sqp_full"     -> LTV, repeated SQP QPs
        """
        solver_mode = str(solver_mode).lower()

        if solver_mode not in ("lti", "ltv_oneshot", "sqp_full"):
            raise ValueError(
                "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
            )

        old_model_mode = self.model_mode

        if solver_mode == "lti":
            self.model_mode = "lti"
            max_iter = 1
        elif solver_mode == "ltv_oneshot":
            self.model_mode = "ltv"
            max_iter = 1
        else:
            self.model_mode = "ltv"
            max_iter = int(self.N_sqp)

        U_guess = np.asarray(U_init, float).reshape(self.Np, self.m)

        best_U = None
        best_debug = {}
        best_status = "not_solved"
        sqp_hist = []

        try:
            for it in range(max_iter):
                H, f, A, l, u, dbg = self._build_mpc_qp(
                    p0=p0,
                    x0=x0,
                    U_guess=U_guess,
                    idx_ref=idx_ref,
                )

                U_vec, _, status = solve_qp_osqp(
                    H,
                    f,
                    A,
                    l,
                    u,
                    U_warm=U_guess.reshape(-1),
                )

                if status not in ("solved", "solved inaccurate") or U_vec is None:
                    if best_U is not None:
                        return best_U, {
                            "status": f"{status}; using_previous_feasible",
                            "sqp_hist": sqp_hist,
                            **best_debug,
                        }

                    return None, {
                        "status": status,
                        "sqp_hist": sqp_hist,
                        "last_debug": dbg,
                    }

                U_new = np.asarray(U_vec, float).reshape(self.Np, self.m)

                step_norm = float(np.linalg.norm(U_new - U_guess))
                rel_step_norm = float(
                    step_norm / (np.linalg.norm(U_guess) + 1e-12)
                )

                sqp_hist.append(
                    {
                        "it": int(it),
                        "status": status,
                        "step_norm": step_norm,
                        "rel_step_norm": rel_step_norm,
                        "cond_H_mpc": float(dbg.get("cond_H_mpc", np.inf)),
                        "lambda_min_H_mpc": float(dbg.get("lambda_min_H_mpc", np.nan)),
                        "lambda_max_H_mpc": float(dbg.get("lambda_max_H_mpc", np.nan)),
                        "num_negative_H_mpc": int(dbg.get("num_negative_H_mpc", -1)),
                        "num_near_zero_H_mpc": int(dbg.get("num_near_zero_H_mpc", -1)),
                    }
                )

                best_U = U_new.copy()
                best_debug = dict(dbg)
                best_status = status

                U_guess = U_new

                if solver_mode == "sqp_full":
                    if (
                        step_norm < self.sqp_tol_u
                        or rel_step_norm < self.sqp_tol_rel_u
                    ):
                        break

            return best_U, {
                "status": best_status,
                "sqp_hist": sqp_hist,
                **best_debug,
            }

        finally:
            self.model_mode = old_model_mode

    def step(self, x_meas=None, rollout_steps=1, solver_mode=None):
        """
        Multi-step MPC step.

        rollout_steps=1:
            standard receding-horizon MPC.

        rollout_steps>1:
            open-loop execution of several optimised controls before re-solving.
        """
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        if solver_mode is None:
            solver_mode = self.solver_mode

        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()

        if x_meas is not None:
            self.x = np.asarray(x_meas, float).reshape(self.n)

        p0 = self.p.copy()
        x0 = self.x.copy()

        if self.U_warm is not None and self.U_warm.size == self.Np * self.m:
            U_init = self.U_warm.reshape(self.Np, self.m).copy()
        else:
            U_init = self._make_initial_U_guess()

        idx_ref = self._select_reference_indices(x0)

        U_seq, solve_info = self._solve_mpc_sequence(
            p0=p0,
            x0=x0,
            U_init=U_init,
            idx_ref=idx_ref,
            solver_mode=solver_mode,
        )

        infeasible = U_seq is None

        if infeasible:
            U_seq = np.zeros((self.Np, self.m), float)
            self.U_warm = None
        else:
            U_seq = np.asarray(U_seq, float).reshape(self.Np, self.m)

        n_apply = int(np.clip(rollout_steps, 1, self.Np))
        rollout_info = self._apply_control_sequence(U_seq, n_apply)

        if not infeasible:
            self._shift_warm_start(U_seq, n_apply)

        X_pred = None
        if not infeasible and "X_aff" in solve_info and "Mc" in solve_info:
            U_vec = U_seq.reshape(-1, 1)
            X_pred = solve_info["X_aff"] + solve_info["Mc"] @ U_vec
            X_pred = X_pred.reshape(self.Np, self.n)

        info = {
            "status": solve_info.get("status", "unknown"),
            "infeasible": int(infeasible),
            "solver_mode": str(solver_mode),
            "rollout_steps": int(n_apply),
            "u0": U_seq[0].copy(),
            "U_seq": U_seq.copy(),
            "p_now": self.p.copy(),
            "x_now": self.x.copy(),
            "idx_ref": np.asarray(idx_ref, int).copy(),
            "X_pred": X_pred.copy() if X_pred is not None else None,
            "sqp_hist": solve_info.get("sqp_hist", []),
            "trust_radius": (
                None
                if self.trust_radius is None
                else self.trust_radius.copy()
            ),
            **rollout_info,
        }
        for key in (
            "cond_H_mpc",
            "lambda_min_H_mpc",
            "lambda_max_H_mpc",
            "num_negative_H_mpc",
            "num_near_zero_H_mpc",
            "valid_H_mpc",
            "norm_H_mpc",
            "norm_H_track",
            "norm_H_effort",
            "norm_H_smooth",
            "qp_reg",

            # MPC Hessian channel diagnostics.
            "mpc_eig_logged",
            "mpc_eig_failed",
            "mpc_eig_cond",
            "mpc_weak_lambda",
            "mpc_strong_lambda",
            "mpc_H_shape_0",
            "mpc_H_shape_1",
            "mpc_weak_channel_index",
            "mpc_weak_channel_name",
            "mpc_weak_channel_energy",
            "mpc_strong_channel_index",
            "mpc_strong_channel_name",
            "mpc_strong_channel_energy",
            "mpc_weak_energy_vx",
            "mpc_weak_energy_vy",
            "mpc_weak_energy_vz",
            "mpc_weak_energy_wx",
            "mpc_weak_energy_wy",
            "mpc_weak_energy_wz",
            "mpc_weak_energy_dL",
            "mpc_strong_energy_vx",
            "mpc_strong_energy_vy",
            "mpc_strong_energy_vz",
            "mpc_strong_energy_wx",
            "mpc_strong_energy_wy",
            "mpc_strong_energy_wz",
            "mpc_strong_energy_dL",
        ):
            if key in solve_info:
                info[key] = solve_info[key]
        if hasattr(self.Jxy_fn, "get_last_diag"):
            info.update(self.Jxy_fn.get_last_diag())
        return self.p.copy(), self.x.copy(), info