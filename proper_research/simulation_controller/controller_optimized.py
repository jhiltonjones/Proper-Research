from __future__ import annotations

import time
import threading
from typing import Any

import numpy as np

from proper_research.simulation_controller.controller import MPCControllerTipXY

from .persistent_osqp import PersistentOSQPSolver, PersistentOSQPSettings


_QP_DIAGNOSTIC_PATCH_LOCK = threading.RLock()


def _scalar_debug(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only inexpensive scalar/string diagnostics."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if np.isscalar(value) or isinstance(value, str):
            out[key] = value
    return out


class MPCControllerTipXYOptimized(MPCControllerTipXY):
    """
    Lean MPC/SQP controller preserving the existing LTI/LTV/SQP definitions.

    Optimisations:
      - persistent OSQP workspace and warm starts;
      - no controller-console printing in the hot path;
      - optional nonlinear candidate validation rather than mandatory validation;
      - scalar-only per-SQP diagnostics by default;
      - explicit timing and forward/Jacobian call accounting.

    The number and location of Jacobians are unchanged:
      - lti: one Jacobian for the horizon;
      - ltv_oneshot: one Jacobian per prediction stage;
      - sqp_full: one Jacobian per stage per SQP linearisation.
    """

    def __init__(
        self,
        *args,
        qp_settings: PersistentOSQPSettings | None = None,
        validate_nonlinear_candidate: bool = False,
        collect_full_diagnostics: bool = False,
        verbose_controller: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qp_solver = PersistentOSQPSolver(qp_settings)
        self.validate_nonlinear_candidate = bool(validate_nonlinear_candidate)
        self.collect_full_diagnostics = bool(collect_full_diagnostics)
        self.verbose_controller = bool(verbose_controller)

        self.controller_step_count = 0
        self.controller_solve_time_total_s = 0.0
        self.controller_rollout_time_total_s = 0.0

        # Warm start for the optional dimensionless progress-shortfall variable.
        # None means that the QP builder's current nominal shortfall estimate
        # should be used instead. This variable is never part of U_warm because
        # U_warm stores physical controls only.
        self._progress_slack_warm_fraction: float | None = None

    def reset_qp_workspace(self) -> None:
        self.qp_solver.reset()
        self._progress_slack_warm_fraction = None

    def set_initial_params(self, p0):
        super().set_initial_params(p0)
        self._progress_slack_warm_fraction = None

    def set_prediction_horizon(self, Np: int, *, reset_warm: bool = False):
        old_np = int(self.Np)
        super().set_prediction_horizon(Np, reset_warm=reset_warm)
        if int(self.Np) != old_np:
            self.reset_qp_workspace()


    def _build_mpc_qp(self, *, p0, x0, U_guess, idx_ref):
        """
        Build the legacy QP while suppressing diagnostic-only eigendecompositions.

        The objective, prediction model and constraints remain those of the
        original QPBuilderMixin.
        """
        if self.collect_full_diagnostics:
            return super()._build_mpc_qp(
                p0=p0,
                x0=x0,
                U_guess=U_guess,
                idx_ref=idx_ref,
            )

        import proper_research.simulation_controller.qp_builder as qp_module

        def no_hessian_diag(H, *, eps=1.0e-12):
            del eps
            H_arr = np.asarray(H, dtype=float)
            return {
                "valid_H_mpc": True,
                "cond_H_mpc": np.nan,
                "lambda_min_H_mpc": np.nan,
                "lambda_max_H_mpc": np.nan,
                "num_negative_H_mpc": -1,
                "num_near_zero_H_mpc": -1,
                "norm_H_mpc": float(np.linalg.norm(H_arr)),
            }

        def no_channel_diag(*args, **kwargs):
            del args, kwargs
            return {
                "mpc_eig_logged": False,
                "mpc_eig_failed": False,
            }

        with _QP_DIAGNOSTIC_PATCH_LOCK:
            old_hessian = qp_module.symmetric_hessian_diagnostics
            old_channel = qp_module.mpc_hessian_channel_diagnostics
            qp_module.symmetric_hessian_diagnostics = no_hessian_diag
            qp_module.mpc_hessian_channel_diagnostics = no_channel_diag
            try:
                H, f, A, l, u, debug = super()._build_mpc_qp(
                    p0=p0,
                    x0=x0,
                    U_guess=U_guess,
                    idx_ref=idx_ref,
                )
            finally:
                qp_module.symmetric_hessian_diagnostics = old_hessian
                qp_module.mpc_hessian_channel_diagnostics = old_channel

        # Retain the matrices required to reconstruct X_pred and scalar
        # diagnostics. Drop plotting-only arrays from the hot-path result.
        compact = _scalar_debug(debug)
        # X_aff/Mc are required to reconstruct the predicted output. The path
        # reference arrays are only O(Np) and are retained so hardware logging
        # uses the same continuous C(s), T(s), s reference as the objective
        # rather than reconstructing a legacy centreline-node reference.
        for key in (
            "X_aff",
            "Mc",
            "path_ref_points",
            "path_ref_tangents",
            "path_ref_s",
        ):
            if key in debug:
                compact[key] = debug[key]
        return H, f, A, l, u, compact

    @staticmethod
    def _active_scaled_step(
        U_previous: np.ndarray,
        U_new: np.ndarray,
        u_max: np.ndarray,
    ) -> tuple[float, float, float]:
        scale = np.abs(np.asarray(u_max, dtype=float).reshape(-1))
        active = scale > 1.0e-12
        if not np.any(active):
            raise RuntimeError("No active control channels remain.")

        previous_scaled = U_previous[:, active] / scale[active].reshape(1, -1)
        new_scaled = U_new[:, active] / scale[active].reshape(1, -1)
        step_scaled_inf = float(
            np.max(np.abs(new_scaled - previous_scaled))
        )
        iterate_scaled_inf = float(
            max(
                np.max(np.abs(previous_scaled)),
                np.max(np.abs(new_scaled)),
            )
        )
        return step_scaled_inf, iterate_scaled_inf, float(
            step_scaled_inf
            / max(iterate_scaled_inf, 1.0e-12)
        )

    def _solve_mpc_sequence(self, *, p0, x0, U_init, idx_ref, solver_mode):
        solver_mode = str(solver_mode).lower()
        if solver_mode not in {"lti", "ltv_oneshot", "sqp_full"}:
            raise ValueError(
                "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
            )

        old_model_mode = self.model_mode
        if solver_mode == "lti":
            self.model_mode = "lti"
            max_sqp = 1
        elif solver_mode == "ltv_oneshot":
            self.model_mode = "ltv"
            max_sqp = 1
        else:
            self.model_mode = "ltv"
            max_sqp = max(1, int(self.N_sqp))

        U_guess = np.asarray(U_init, dtype=float).reshape(self.Np, self.m)
        Nu = int(self.Np) * int(self.m)
        u_max = np.asarray(self.u_max, dtype=float).reshape(1, self.m)
        U_guess = np.clip(U_guess, -u_max, u_max)
        inactive = np.abs(u_max.reshape(-1)) <= 1.0e-12
        U_guess[:, inactive] = 0.0

        sigma_warm = getattr(self, "_progress_slack_warm_fraction", None)
        if sigma_warm is not None and not np.isfinite(float(sigma_warm)):
            sigma_warm = None

        best_U: np.ndarray | None = None
        best_sigma: float | None = None
        best_debug: dict[str, Any] = {}
        best_status = "not_solved"
        sqp_hist: list[dict[str, Any]] = []

        qp_build_total_s = 0.0
        osqp_total_s = 0.0
        validation_total_s = 0.0

        try:
            for iteration in range(max_sqp):
                build_started = time.perf_counter()
                H, f, A, l, u, dbg = self._build_mpc_qp(
                    p0=p0,
                    x0=x0,
                    U_guess=U_guess,
                    idx_ref=idx_ref,
                )
                qp_build_s = time.perf_counter() - build_started
                qp_build_total_s += qp_build_s

                # The newest QP can append one scalar sigma for soft progress.
                # U_guess/U_warm remain physical-control vectors only, so build a
                # decision-vector warm start explicitly when sigma is present.
                n_decision = int(np.asarray(H).shape[0])
                if n_decision == Nu:
                    z_warm = U_guess.reshape(Nu)
                    has_progress_slack = False
                elif n_decision == Nu + 1:
                    sigma0 = sigma_warm
                    if sigma0 is None:
                        sigma0 = float(
                            dbg.get("progress_slack_guess_fraction", 0.0)
                        )
                    sigma0 = float(np.clip(sigma0, 0.0, 1.0))
                    z_warm = np.concatenate(
                        [U_guess.reshape(Nu), np.array([sigma0], dtype=float)]
                    )
                    has_progress_slack = True
                else:
                    raise ValueError(
                        "Unexpected optimized QP decision dimension: "
                        f"{n_decision}; expected {Nu} controls or {Nu + 1} "
                        "for controls plus progress slack."
                    )

                solve_started = time.perf_counter()
                Z_vec, _, status, osqp_diag = self.qp_solver.solve(
                    H,
                    f,
                    A,
                    l,
                    u,
                    x_warm=z_warm,
                )
                osqp_s = time.perf_counter() - solve_started
                osqp_total_s += osqp_s
                status_text = str(status)
                solved = status_text.lower() in {
                    "solved",
                    "solved inaccurate",
                }

                if not solved or Z_vec is None:
                    failure = {
                        "status": status_text,
                        "infeasible": int("infeasible" in status_text.lower()),
                        "qp_infeasible": int(
                            "infeasible" in status_text.lower()
                        ),
                        "failed_sqp_iter": int(iteration),
                        "sqp_hist": sqp_hist,
                        "qp_build_time_total_s": float(qp_build_total_s),
                        "osqp_time_total_s": float(osqp_total_s),
                        **osqp_diag,
                    }
                    failure.update(
                        dbg if self.collect_full_diagnostics else _scalar_debug(dbg)
                    )
                    if best_U is not None:
                        failure["status"] = (
                            f"{status_text}; using_previous_feasible"
                        )
                        failure["used_previous_feasible"] = True
                        failure.update(best_debug)
                        if best_sigma is not None:
                            self._progress_slack_warm_fraction = float(best_sigma)
                        return best_U, failure
                    return None, failure

                Z_vec = np.asarray(Z_vec, dtype=float).reshape(-1)
                if Z_vec.size != n_decision:
                    raise ValueError(
                        f"OSQP returned {Z_vec.size} decision values; "
                        f"expected {n_decision}."
                    )
                U_new = Z_vec[:Nu].reshape(self.Np, self.m)
                U_new[:, inactive] = 0.0
                sigma_solution = (
                    float(Z_vec[Nu]) if has_progress_slack else None
                )
                if sigma_solution is not None:
                    sigma_warm = float(np.clip(sigma_solution, 0.0, 1.0))

                validation_diag: dict[str, Any] = {}
                if self.validate_nonlinear_candidate:
                    validation_started = time.perf_counter()
                    angle_diag = (
                        self._nonlinear_tip_tangent_constraint_diagnostics(
                            p0=p0,
                            U_candidate=U_new,
                        )
                    )
                    validation_diag.update(angle_diag)

                    if int(
                        angle_diag.get(
                            "tip_tangent_nonlinear_feasible",
                            1,
                        )
                    ) != 1:
                        validation_total_s += (
                            time.perf_counter() - validation_started
                        )
                        return None, {
                            "status": (
                                "nonlinear_tip_tangent_constraint_violation"
                            ),
                            "infeasible": 1,
                            "qp_infeasible": 0,
                            "safety_constraint_infeasible": 1,
                            "failed_sqp_iter": int(iteration),
                            "sqp_hist": sqp_hist,
                            "qp_build_time_total_s": float(qp_build_total_s),
                            "osqp_time_total_s": float(osqp_total_s),
                            "candidate_validation_time_total_s": float(
                                validation_total_s
                            ),
                            **_scalar_debug(validation_diag),
                            **osqp_diag,
                        }

                    if "X_aff" in dbg and "Mc" in dbg:
                        lin_diag = self._linearisation_error_diagnostics(
                            p0=p0,
                            U_candidate=U_new,
                            X_aff=dbg["X_aff"],
                            Mc=dbg["Mc"],
                        )
                        validation_diag.update(lin_diag)
                    validation_total_s += (
                        time.perf_counter() - validation_started
                    )

                step_scaled_inf, iterate_scaled_inf, relative_step = (
                    self._active_scaled_step(
                        U_guess,
                        U_new,
                        self.u_max,
                    )
                )
                threshold = float(
                    self.sqp_tol_abs
                    + self.sqp_tol_rel * iterate_scaled_inf
                )
                converged = bool(step_scaled_inf <= threshold)

                iteration_diag = {
                    "it": int(iteration),
                    "status": status_text,
                    "step_scaled_inf": float(step_scaled_inf),
                    "relative_step": float(relative_step),
                    "step_threshold": float(threshold),
                    "converged": int(converged),
                    "qp_build_s": float(qp_build_s),
                    "osqp_s": float(osqp_s),
                    "osqp_iter": int(osqp_diag.get("osqp_iter", -1)),
                    "osqp_setup_performed": int(
                        osqp_diag.get("osqp_setup_performed", 0)
                    ),
                    "qp_decision_dim": int(n_decision),
                    "progress_slack_solution_fraction": (
                        np.nan
                        if sigma_solution is None
                        else float(sigma_solution)
                    ),
                }
                iteration_diag.update(_scalar_debug(validation_diag))
                sqp_hist.append(iteration_diag)

                best_U = U_new.copy()
                best_sigma = sigma_solution
                best_status = status_text
                if self.collect_full_diagnostics:
                    best_debug = dict(dbg)
                    best_debug.update(validation_diag)
                else:
                    best_debug = _scalar_debug(dbg)
                    best_debug.update(_scalar_debug(validation_diag))
                best_debug.update(osqp_diag)
                best_debug["qp_control_dim"] = int(Nu)
                best_debug["qp_decision_dim"] = int(n_decision)
                best_debug["progress_slack_solution_fraction"] = (
                    np.nan if sigma_solution is None else float(sigma_solution)
                )

                if solver_mode == "sqp_full" and converged:
                    best_debug.update(
                        {
                            "sqp_converged": 1,
                            "sqp_stop_reason": "step_tolerance",
                            "sqp_iterations_used": len(sqp_hist),
                            "sqp_step_scaled_inf": float(step_scaled_inf),
                            "sqp_step_threshold": float(threshold),
                            "sqp_hit_iteration_cap": 0,
                        }
                    )
                    break

                U_guess = U_new

            if best_U is None:
                return None, {
                    "status": "no_qp_feasible_candidate",
                    "infeasible": 0,
                    "qp_infeasible": 0,
                    "solver_failed": 1,
                    "sqp_hist": sqp_hist,
                }

            self._progress_slack_warm_fraction = (
                None if best_sigma is None else float(best_sigma)
            )

            result = {
                "status": best_status,
                "infeasible": 0,
                "qp_infeasible": 0,
                "sqp_hist": sqp_hist,
                "sqp_iterations_used": len(sqp_hist),
                "qp_build_time_total_s": float(qp_build_total_s),
                "osqp_time_total_s": float(osqp_total_s),
                "candidate_validation_time_total_s": float(validation_total_s),
                "persistent_osqp_setup_count": int(
                    self.qp_solver.setup_count
                ),
                "persistent_osqp_update_count": int(
                    self.qp_solver.update_count
                ),
                **best_debug,
            }
            if (
                solver_mode == "sqp_full"
                and not result.get("sqp_converged", 0)
            ):
                result["sqp_stop_reason"] = "iteration_cap"
                result["sqp_hit_iteration_cap"] = 1
            return best_U, result
        finally:
            self.model_mode = old_model_mode

    def step(self, x_meas=None, rollout_steps=1, solver_mode=None):
        """
        Lean controller step.

        The nonlinear plant rollout is still performed because it is the
        simulation state transition. Diagnostic-only forward/Jacobian calls and
        console output from the legacy step are omitted.
        """
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before step().")

        mode = self.solver_mode if solver_mode is None else str(solver_mode)
        self.controller_step_count += 1

        if hasattr(self.forward_tip_fn, "start_step"):
            self.forward_tip_fn.start_step()

        if x_meas is not None:
            self.x = np.asarray(x_meas, dtype=float).reshape(self.n)

        p0 = np.asarray(self.p, dtype=float).copy()
        x0 = np.asarray(self.x, dtype=float).copy()

        if self.U_warm is not None and self.U_warm.size == self.Np * self.m:
            U_init = self.U_warm.reshape(self.Np, self.m).copy()
        else:
            U_init = self._make_initial_U_guess()

        idx_ref = self._select_reference_indices(x0)

        plant_model = getattr(self.forward_tip_fn, "model", None)
        jacobian_model = getattr(self.Jxy_fn, "model", None)
        plant_before = (
            plant_model.performance_counters()
            if hasattr(plant_model, "performance_counters")
            else {}
        )
        jac_before = (
            jacobian_model.performance_counters()
            if hasattr(jacobian_model, "performance_counters")
            else {}
        )

        solve_started = time.perf_counter()
        U_seq, solve_info = self._solve_mpc_sequence(
            p0=p0,
            x0=x0,
            U_init=U_init,
            idx_ref=idx_ref,
            solver_mode=mode,
        )
        solve_wall_s = time.perf_counter() - solve_started
        self.controller_solve_time_total_s += solve_wall_s

        infeasible = U_seq is None
        if infeasible:
            U_seq = np.zeros((self.Np, self.m), dtype=float)
            self.U_warm = None
        else:
            U_seq = np.asarray(U_seq, dtype=float).reshape(self.Np, self.m)

        n_apply = int(np.clip(rollout_steps, 1, self.Np))
        rollout_started = time.perf_counter()
        rollout_info = self._apply_control_sequence(U_seq, n_apply)
        rollout_wall_s = time.perf_counter() - rollout_started
        self.controller_rollout_time_total_s += rollout_wall_s

        U_applied = np.asarray(
            rollout_info.get("U_applied", []),
            dtype=float,
        )
        if U_applied.size:
            U_applied = U_applied.reshape(-1, self.m)
            self.u_prev = U_applied[-1].copy()
        else:
            U_applied = np.empty((0, self.m), dtype=float)
            self.u_prev = U_seq[n_apply - 1].copy()

        if not infeasible:
            self._shift_warm_start(U_seq, n_apply)

        X_pred = None
        if "X_aff" in solve_info and "Mc" in solve_info:
            X_pred = (
                solve_info["X_aff"]
                + solve_info["Mc"] @ U_seq.reshape(-1, 1)
            ).reshape(self.Np, self.n)

        x_rollout = np.asarray(
            rollout_info.get("x_rollout", []),
            dtype=float,
        )
        if x_rollout.size:
            x_rollout = x_rollout.reshape(-1, self.n)
        else:
            x_rollout = np.empty((0, self.n), dtype=float)

        compare_count = min(
            n_apply,
            x_rollout.shape[0],
            0 if X_pred is None else X_pred.shape[0],
        )
        if compare_count > 0:
            error = x_rollout[:compare_count] - X_pred[:compare_count]
            rollout_error_xy = np.linalg.norm(error[:, :2], axis=1)
            rollout_error_xyz = np.linalg.norm(error[:, :3], axis=1)
        else:
            rollout_error_xy = np.empty(0, dtype=float)
            rollout_error_xyz = np.empty(0, dtype=float)

        plant_after = (
            plant_model.performance_counters()
            if hasattr(plant_model, "performance_counters")
            else {}
        )
        jac_after = (
            jacobian_model.performance_counters()
            if hasattr(jacobian_model, "performance_counters")
            else {}
        )

        def delta(after: dict, before: dict, key: str):
            return after.get(key, 0) - before.get(key, 0)

        info: dict[str, Any] = {
            "status": solve_info.get("status", "unknown"),
            "infeasible": int(infeasible),
            "solver_mode": mode,
            "solver_mode_used": mode,
            "rollout_steps": int(n_apply),
            "rollout_steps_used": int(n_apply),
            "u0": U_seq[0].copy(),
            "U_seq": U_seq.copy(),
            "p_now": np.asarray(self.p, dtype=float).copy(),
            "x_now": np.asarray(self.x, dtype=float).copy(),
            "idx_ref": np.asarray(idx_ref, dtype=int).copy(),
            "X_pred": None if X_pred is None else X_pred.copy(),
            "rollout_err_xy": rollout_error_xy.copy(),
            "rollout_err_xyz": rollout_error_xyz.copy(),
            "controller_solve_wall_s": float(solve_wall_s),
            "controller_rollout_wall_s": float(rollout_wall_s),
            "controller_step_wall_s": float(
                solve_wall_s + rollout_wall_s
            ),
            "forward_calls_step": int(
                delta(plant_after, plant_before, "forward_calls")
            ),
            "jacobian_calls_step": int(
                delta(jac_after, jac_before, "jacobian_calls")
            ),
            "forward_time_step_s": float(
                delta(
                    plant_after,
                    plant_before,
                    "forward_time_total_s",
                )
            ),
            "jacobian_time_step_s": float(
                delta(
                    jac_after,
                    jac_before,
                    "jacobian_time_total_s",
                )
            ),
            **rollout_info,
            **solve_info,
        }
        # Always expose the exact frozen continuous reference used by this
        # controller frame. These arrays are small and are required for correct
        # hardware/publication logging even when full diagnostics are disabled.
        active_ref = getattr(self, "active_path_reference", None)
        if isinstance(active_ref, dict):
            try:
                C_ref = np.asarray(active_ref["C_ref"], dtype=float).copy()
                T_ref = np.asarray(active_ref["T_ref"], dtype=float).copy()
                s_ref = np.asarray(active_ref["s_ref"], dtype=float).copy()
            except (KeyError, TypeError, ValueError):
                pass
            else:
                info["path_ref_points"] = C_ref
                info["path_ref_tangents"] = T_ref
                info["path_ref_s"] = s_ref
                info["path_progress_current_m"] = float(
                    active_ref.get(
                        "s_progress",
                        getattr(self, "path_progress_s", np.nan),
                    )
                )

        if hasattr(self.Jxy_fn, "get_last_diag"):
            info.update(self.Jxy_fn.get_last_diag())
        return self.p.copy(), self.x.copy(), info
