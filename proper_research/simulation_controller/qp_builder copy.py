import numpy as np

from proper_research.simulation_controller.constraints import (
    input_bounds,
    trust_region_bounds,
    box_constraint_from_bounds,
    dL_bounds,
    stack_constraints,
)
from .diagnostics import mpc_hessian_channel_diagnostics, diagnose_constraints_at_U, classify_constraint_issue
from .geometry import build_Pm_world

def stack_named_constraints(named_blocks):
    """
    Stack named OSQP constraint blocks.

    Each block is:
        (name, A, l, u)

    Returns:
        A, l, u, slices

    where slices maps:
        name -> slice(row_start, row_end)
    """
    named_blocks = [b for b in named_blocks if b is not None]

    if len(named_blocks) == 0:
        return (
            np.zeros((0, 0), float),
            np.zeros(0, float),
            np.zeros(0, float),
            {},
        )

    A_list = []
    l_list = []
    u_list = []
    slices = {}

    n_cols = None
    row_start = 0

    for name, A, l, u in named_blocks:
        A = np.asarray(A, float)
        l = np.asarray(l, float).reshape(-1)
        u = np.asarray(u, float).reshape(-1)

        if A.ndim != 2:
            raise ValueError(f"{name}: A must be 2D.")

        if A.shape[0] != l.size or A.shape[0] != u.size:
            raise ValueError(
                f"{name}: shape mismatch A={A.shape}, l={l.shape}, u={u.shape}"
            )

        if n_cols is None:
            n_cols = A.shape[1]
        elif A.shape[1] != n_cols:
            raise ValueError(
                f"{name}: column mismatch, got {A.shape[1]}, expected {n_cols}"
            )

        row_end = row_start + A.shape[0]
        slices[name] = slice(row_start, row_end)

        A_list.append(A)
        l_list.append(l)
        u_list.append(u)

        row_start = row_end

    A = np.vstack(A_list)
    l = np.concatenate(l_list)
    u = np.concatenate(u_list)

    return A, l, u, slices
def symmetric_hessian_diagnostics(H, *, eps: float = 1e-12) -> dict:
    """
    Diagnostics for a symmetric Hessian-like matrix.

    Returns condition number, eigenvalue range, and simple definiteness flags.
    This is intended for logging/adaptive rollout, not for modifying the QP.
    """
    H = np.asarray(H, float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return {
            "valid_H_mpc": False,
            "cond_H_mpc": np.inf,
            "lambda_min_H_mpc": np.nan,
            "lambda_max_H_mpc": np.nan,
            "num_negative_H_mpc": -1,
            "num_near_zero_H_mpc": -1,
        }

    Hs = 0.5 * (H + H.T)

    try:
        eigvals = np.linalg.eigvalsh(Hs)
    except np.linalg.LinAlgError:
        return {
            "valid_H_mpc": False,
            "cond_H_mpc": np.inf,
            "lambda_min_H_mpc": np.nan,
            "lambda_max_H_mpc": np.nan,
            "num_negative_H_mpc": -1,
            "num_near_zero_H_mpc": -1,
        }

    abs_eigs = np.abs(eigvals)

    lam_min_abs = float(np.min(abs_eigs))
    lam_max_abs = float(np.max(abs_eigs))

    return {
        "valid_H_mpc": True,
        "cond_H_mpc": float(lam_max_abs / max(lam_min_abs, eps)),
        "lambda_min_H_mpc": float(np.min(eigvals)),
        "lambda_max_H_mpc": float(np.max(eigvals)),
        "num_negative_H_mpc": int(np.sum(eigvals < -eps)),
        "num_near_zero_H_mpc": int(np.sum(abs_eigs < eps)),
    }
class QPBuilderMixin:
    """
    Shared QP builder for the original and optimized MPC controllers.

    The default ``reference_mode='point'`` preserves the legacy objective.
    Set ``reference_mode='contouring'`` to use a continuous path-following
    objective with a strong normal/contouring penalty, a weaker along-path
    lag penalty and configurable progress handling.  For normal safe control,
    ``progress_constraint_mode="soft_slack"`` augments the decision vector with
    one dimensionless progress-shortfall variable sigma.  Hard physical/safety
    constraints remain hard; progress is allowed to relax so the controller can
    safely reposition before advancing.
    """

    # ------------------------------------------------------------------
    # Reference/path helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalised_path_tangents(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError(f"Path tangents must have shape (Np, 3), got {T.shape}.")
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
            raise ValueError("Path tangents must be finite and non-zero.")
        return T / norms

    def _reference_mode(self) -> str:
        mode = str(getattr(self, "reference_mode", "point")).strip().lower()
        aliases = {
            "waypoint": "point",
            "waypoints": "point",
            "path": "contouring",
            "path_following": "contouring",
            "path-following": "contouring",
        }
        mode = aliases.get(mode, mode)
        if mode not in ("point", "contouring"):
            raise ValueError(
                "reference_mode must be 'point' or 'contouring'; "
                f"got {mode!r}."
            )
        return mode

    @staticmethod
    def _state_reference_key(x0: np.ndarray) -> tuple[float, ...]:
        # x0 is fixed throughout one SQP solve, so this key freezes the path
        # reference across all SQP linearisation iterations.
        return tuple(np.asarray(x0, dtype=float).reshape(-1).tolist())

    def _validate_path_reference(self, path_ref: dict) -> dict:
        if not isinstance(path_ref, dict):
            raise TypeError("active_path_reference must be a dictionary.")

        Np = int(self.Np)
        C_ref = np.asarray(path_ref.get("C_ref"), dtype=float)
        T_ref = np.asarray(path_ref.get("T_ref"), dtype=float)
        s_ref = np.asarray(path_ref.get("s_ref"), dtype=float).reshape(-1)

        if C_ref.shape != (Np, 3):
            raise ValueError(
                f"C_ref must have shape {(Np, 3)}, got {C_ref.shape}."
            )
        if T_ref.shape != (Np, 3):
            raise ValueError(
                f"T_ref must have shape {(Np, 3)}, got {T_ref.shape}."
            )
        if s_ref.shape != (Np,):
            raise ValueError(
                f"s_ref must have shape {(Np,)}, got {s_ref.shape}."
            )
        if not (
            np.all(np.isfinite(C_ref))
            and np.all(np.isfinite(T_ref))
            and np.all(np.isfinite(s_ref))
        ):
            raise FloatingPointError("Path reference contains non-finite values.")

        T_ref = self._normalised_path_tangents(T_ref)

        out = dict(path_ref)
        out["C_ref"] = C_ref.copy()
        out["T_ref"] = T_ref.copy()
        out["s_ref"] = s_ref.copy()
        out["s_progress"] = float(
            path_ref.get(
                "s_progress",
                getattr(self, "path_progress_s", s_ref[0]),
            )
        )
        return out

    def _resolve_path_reference(self, x0: np.ndarray) -> dict:
        """
        Return one path reference frozen for the complete SQP solve.

        Preferred integration:
            ReferenceMixin._select_reference_indices(...) sets
            self.active_path_reference before _solve_mpc_sequence(...).

        Compatibility fallback:
            when _build_path_reference exists, build it once for the current
            x0 and reuse it for every subsequent QP in the same SQP solve.
        """
        key = self._state_reference_key(x0)
        cached_key = getattr(self, "_active_path_reference_x0_key", None)
        active = getattr(self, "active_path_reference", None)

        if active is not None and cached_key == key:
            return self._validate_path_reference(active)

        if active is not None and cached_key is None:
            active = self._validate_path_reference(active)
            self.active_path_reference = active
            self._active_path_reference_x0_key = key
            return active

        if hasattr(self, "_build_path_reference"):
            active = self._build_path_reference(x0)
        elif all(
            hasattr(self, name)
            for name in ("ref_points_last", "ref_tangents_last", "ref_s_last")
        ):
            active = {
                "C_ref": np.asarray(self.ref_points_last, dtype=float),
                "T_ref": np.asarray(self.ref_tangents_last, dtype=float),
                "s_ref": np.asarray(self.ref_s_last, dtype=float),
                "s_progress": float(
                    getattr(self, "path_progress_s", self.ref_s_last[0])
                ),
                "projection": {
                    "distance_m": float(
                        getattr(self, "path_projection_distance_last", np.nan)
                    )
                },
            }
        else:
            raise RuntimeError(
                "reference_mode='contouring' requires a continuous path "
                "reference. Install the PathReferenceMixin and either set "
                "self.active_path_reference before solving or provide "
                "_build_path_reference(x_now)."
            )

        active = self._validate_path_reference(active)
        self.active_path_reference = active
        self._active_path_reference_x0_key = key
        return active

    def _build_continuous_reference_stack(self, path_ref: dict) -> np.ndarray:
        n = int(self.n)
        Np = int(self.Np)
        C_ref = np.asarray(path_ref["C_ref"], dtype=float).reshape(Np, 3)
        T_ref = np.asarray(path_ref["T_ref"], dtype=float).reshape(Np, 3)

        X_ref = np.zeros((Np, n), dtype=float)
        X_ref[:, : min(3, n)] = C_ref[:, : min(3, n)]

        # Tangent references are always populated when the output contains
        # them. Whether they are tracked is determined only by the configured
        # tangent weights in self.Q. Zero weights keep tangent out of the
        # objective while still allowing tangent safety constraints.
        if n >= 6:
            X_ref[:, 3:6] = T_ref

        return X_ref.reshape(Np * n, 1)

    def _build_contouring_Qtil(self, path_ref: dict) -> np.ndarray:
        """
        Build the path-following stage weights

            Q_k = q_contour P_perp^T W P_perp
                + q_lag     P_tan^T  W P_tan

        where P_tan = t t^T and P_perp = I - t t^T.
        """
        n = int(self.n)
        Np = int(self.Np)
        T_ref = self._normalised_path_tangents(path_ref["T_ref"])

        stage_weights = np.asarray(
            getattr(self, "ref_stage_weights", np.ones(Np)),
            dtype=float,
        ).reshape(-1)
        if stage_weights.size != Np:
            raise ValueError(
                f"ref_stage_weights must have length {Np}, "
                f"got {stage_weights.size}."
            )
        if np.any(~np.isfinite(stage_weights)) or np.any(stage_weights < 0.0):
            raise ValueError("ref_stage_weights must be finite and non-negative.")

        q_default = 1.0
        try:
            q_diag = np.diag(np.asarray(self.Q, dtype=float)[:3, :3])
            positive = q_diag[np.isfinite(q_diag) & (q_diag > 0.0)]
            if positive.size:
                q_default = float(np.mean(positive))
        except Exception:
            pass

        q_contour = float(getattr(self, "q_contour", q_default))
        q_lag = float(
            getattr(
                self,
                "q_lag",
                float(getattr(self, "contouring_lag_ratio", 0.05)) * q_contour,
            )
        )
        if q_contour < 0.0 or q_lag < 0.0:
            raise ValueError("q_contour and q_lag must be non-negative.")

        axis_weights = np.asarray(
            getattr(self, "contouring_axis_weights", np.ones(3)),
            dtype=float,
        ).reshape(-1)
        if axis_weights.size != 3:
            raise ValueError(
                "contouring_axis_weights must contain three entries."
            )
        if np.any(~np.isfinite(axis_weights)) or np.any(axis_weights < 0.0):
            raise ValueError(
                "contouring_axis_weights must be finite and non-negative."
            )

        W = np.diag(axis_weights)
        I3 = np.eye(3)
        Qtil = np.zeros((Np * n, Np * n), dtype=float)

        for k in range(Np):
            t = T_ref[k]
            P_tan = np.outer(t, t)
            P_perp = I3 - P_tan
            Q_pos = (
                q_contour * (P_perp.T @ W @ P_perp)
                + q_lag * (P_tan.T @ W @ P_tan)
            )
            Q_pos = 0.5 * (Q_pos + Q_pos.T)

            Qk = np.zeros((n, n), dtype=float)
            Qk[:3, :3] = stage_weights[k] * Q_pos

            # Optional tip-tangent tracking. The normal path objective remains
            # position-only. Tangent tracking is a separate ordinary quadratic
            # term with reference T_ref. With weights [0, 0, 0] this block is
            # exactly zero, so the tangent is modelled but not tracked.
            if n >= 6:
                tangent_Q = np.asarray(self.Q, dtype=float)[3:6, 3:6]
                Qk[3:6, 3:6] = stage_weights[k] * tangent_Q

            # Preserve any explicitly configured channels beyond six.
            if n > 6:
                Qk[6:, 6:] = stage_weights[k] * np.asarray(
                    self.Q, dtype=float
                )[6:, 6:]

            rows = slice(k * n, (k + 1) * n)
            Qtil[rows, rows] = Qk

        return Qtil

    @staticmethod
    def _path_error_components(
        positions: np.ndarray,
        C_ref: np.ndarray,
        T_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        C_ref = np.asarray(C_ref, dtype=float).reshape(-1, 3)
        T_ref = np.asarray(T_ref, dtype=float).reshape(-1, 3)
        error = positions - C_ref
        lag = np.sum(error * T_ref, axis=1)
        contour_vec = error - lag[:, None] * T_ref
        contour_norm = np.linalg.norm(contour_vec, axis=1)
        return contour_vec, contour_norm, lag

    def _path_tracking_diagnostics(
        self,
        *,
        X_nom: np.ndarray,
        path_ref: dict,
    ) -> dict:
        n = int(self.n)
        Np = int(self.Np)
        X_nom_stage = np.asarray(X_nom, dtype=float).reshape(Np, n)
        C_ref = np.asarray(path_ref["C_ref"], dtype=float).reshape(Np, 3)
        T_ref = self._normalised_path_tangents(path_ref["T_ref"])
        contour_vec, contour_norm, lag = self._path_error_components(
            X_nom_stage[:, :3], C_ref, T_ref
        )

        projection = path_ref.get("projection", {}) or {}
        projection_distance = float(
            projection.get(
                "distance_m",
                getattr(self, "path_projection_distance_last", np.nan),
            )
        )

        return {
            "reference_mode": "contouring",
            "path_progress_current_m": float(path_ref["s_progress"]),
            "path_projection_distance_m": projection_distance,
            "path_projection_distance_mm": 1.0e3 * projection_distance,
            "contour_error_nominal_rms_m": float(
                np.sqrt(np.mean(contour_norm**2))
            ),
            "contour_error_nominal_max_m": float(np.max(contour_norm)),
            "contour_error_nominal_terminal_m": float(contour_norm[-1]),
            "lag_error_nominal_rms_m": float(np.sqrt(np.mean(lag**2))),
            "lag_error_nominal_terminal_m": float(lag[-1]),
            "contour_error_nominal_m": contour_norm.copy(),
            "lag_error_nominal_m": lag.copy(),
            "contour_error_nominal_vectors_m": contour_vec.copy(),
        }

    def _path_total_length(self) -> float:
        C = np.asarray(self.lumen_C, dtype=float)
        if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] < 3:
            raise ValueError(
                f"lumen_C must have shape (M, >=3), M >= 2; got {C.shape}."
            )
        return float(np.sum(np.linalg.norm(np.diff(C[:, :3], axis=0), axis=1)))

    def _build_terminal_progress_model(
        self,
        *,
        X_aff: np.ndarray,
        Mc: np.ndarray,
        U_guess: np.ndarray,
        path_ref: dict,
    ) -> tuple[np.ndarray, float, dict]:
        """
        Build the affine terminal progress approximation

            s_hat(U) = b_progress + A_progress @ U.

        The local progress coordinate is measured along the terminal reference
        tangent.  It is not an exact-waypoint equality.
        """
        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        Nu = Np * m

        X_aff = np.asarray(X_aff, dtype=float).reshape(Np * n, 1)
        Mc = np.asarray(Mc, dtype=float).reshape(Np * n, Nu)
        U_guess_vec = np.asarray(U_guess, dtype=float).reshape(Nu)

        C_terminal = np.asarray(path_ref["C_ref"], dtype=float).reshape(Np, 3)[-1]
        T_terminal = self._normalised_path_tangents(path_ref["T_ref"])[-1]
        s_anchor = float(np.asarray(path_ref["s_ref"], dtype=float).reshape(Np)[-1])

        rows = slice((Np - 1) * n, (Np - 1) * n + 3)
        X_aff_terminal = X_aff[rows, 0]
        Mc_terminal = Mc[rows, :]

        A_progress = (T_terminal.reshape(1, 3) @ Mc_terminal).reshape(Nu)
        b_progress = float(
            s_anchor
            + T_terminal @ (X_aff_terminal - C_terminal)
        )
        s_nominal = float(b_progress + A_progress @ U_guess_vec)

        s_progress = float(path_ref["s_progress"])
        request = float(getattr(self, "progress_request_m", 0.0))
        if request < 0.0:
            raise ValueError("progress_request_m must be non-negative.")

        path_end = self._path_total_length()
        target = min(path_end, s_progress + request)
        effective_request = max(0.0, target - s_progress)
        shortfall = max(0.0, target - s_nominal)

        debug = {
            "progress_affine_b_m": b_progress,
            "progress_current_m": s_progress,
            "progress_request_m": request,
            "progress_request_effective_m": effective_request,
            "progress_target_m": target,
            "progress_nominal_terminal_m": s_nominal,
            "progress_nominal_shortfall_m": shortfall,
            "progress_nominal_shortfall_mm": 1.0e3 * shortfall,
            "progress_path_end_m": path_end,
            "progress_constraint_row_norm": float(np.linalg.norm(A_progress)),
        }
        return A_progress, b_progress, debug

    def _build_terminal_progress_constraint(
        self,
        *,
        A_progress: np.ndarray,
        b_progress: float,
        progress_debug: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Nu = int(self.Np) * int(self.m)
        A_progress = np.asarray(A_progress, dtype=float).reshape(1, Nu)
        target = float(progress_debug["progress_target_m"])
        lower = np.array([target - float(b_progress)], dtype=float)
        upper = np.array([np.inf], dtype=float)
        return A_progress, lower, upper

    def _progress_constraint_mode(self, *, reference_mode: str) -> str:
        """Resolve progress handling while preserving legacy configuration.

        Modes
        -----
        none:
            No terminal progress constraint.
        hard:
            Legacy hard terminal minimum-progress constraint.
        soft_slack:
            Add one dimensionless slack fraction sigma to the QP.  sigma=0
            means the full requested progress must be achieved; sigma=1 allows
            the terminal progress requirement to relax back to the current path
            station.  Safety constraints are never softened.
        """
        if reference_mode != "contouring":
            return "none"

        configured = getattr(self, "progress_constraint_mode", None)
        if configured is None:
            if bool(getattr(self, "enable_progress_slack", False)):
                return "soft_slack"
            if bool(getattr(self, "enable_hard_progress_constraint", False)):
                return "hard"
            return "none"

        mode = str(configured).strip().lower()
        aliases = {
            "off": "none",
            "disabled": "none",
            "soft": "soft_slack",
            "slack": "soft_slack",
            "soft-progress": "soft_slack",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"none", "hard", "soft_slack"}:
            raise ValueError(
                "progress_constraint_mode must be 'none', 'hard', or "
                f"'soft_slack'; got {mode!r}."
            )
        return mode

    @staticmethod
    def _augment_control_constraint(A: np.ndarray, *, n_control: int, n_decision: int):
        """Append zero columns for non-control decision variables."""
        A = np.asarray(A, dtype=float)
        if A.ndim != 2 or A.shape[1] != n_control:
            raise ValueError(
                f"Control constraint must have {n_control} columns; got {A.shape}."
            )
        if n_decision == n_control:
            return A
        return np.hstack(
            [A, np.zeros((A.shape[0], n_decision - n_control), dtype=float)]
        )

    def _build_soft_progress_constraint(
        self,
        *,
        A_progress: np.ndarray,
        b_progress: float,
        progress_debug: dict,
        n_control: int,
        n_decision: int,
        slack_index: int,
    ):
        """Build soft terminal-progress and sigma-bound rows.

        The terminal progress row is

            A_progress U + ds_request * sigma >= target - b_progress.

        sigma is dimensionless.  With 0 <= sigma <= 1, sigma=1 relaxes the
        requested progress back to zero net forward progress, rather than
        permitting the optimizer to move backwards simply to buy feasibility.
        """
        if n_decision != n_control + 1 or slack_index != n_control:
            raise ValueError("Soft progress currently expects one appended slack variable.")

        effective_request = float(progress_debug["progress_request_effective_m"])
        target = float(progress_debug["progress_target_m"])

        row = np.zeros((1, n_decision), dtype=float)
        if effective_request > 1.0e-12:
            row[0, :n_control] = np.asarray(A_progress, dtype=float).reshape(n_control)
            row[0, slack_index] = effective_request
            l_progress = np.array([target - float(b_progress)], dtype=float)
            u_progress = np.array([np.inf], dtype=float)
        else:
            # Keep a fixed row count for persistent OSQP, but disable the row
            # when there is no remaining progress request (e.g. path end).
            l_progress = np.array([-np.inf], dtype=float)
            u_progress = np.array([np.inf], dtype=float)

        max_fraction = float(getattr(self, "progress_slack_max_fraction", 1.0))
        if (
            not np.isfinite(max_fraction)
            or max_fraction < 0.0
            or max_fraction > 1.0
        ):
            raise ValueError(
                "progress_slack_max_fraction must be finite and lie in [0, 1]."
            )
        if effective_request <= 1.0e-12:
            max_fraction = 0.0

        A_sigma = np.zeros((1, n_decision), dtype=float)
        A_sigma[0, slack_index] = 1.0
        l_sigma = np.array([0.0], dtype=float)
        u_sigma = np.array([max_fraction], dtype=float)

        return (row, l_progress, u_progress), (A_sigma, l_sigma, u_sigma)

    # ------------------------------------------------------------------
    # Main builder
    # ------------------------------------------------------------------
    def _build_mpc_qp(self, *, p0, x0, U_guess, idx_ref):
        """
        Build one local MPC QP around ``U_guess``.

        Point mode preserves the original exact-waypoint tracking objective.
        Contouring mode uses continuous path points and tangents. The physical
        control subvector is unchanged; ``progress_constraint_mode="soft_slack"``
        appends one dimensionless progress-shortfall variable. All physical and
        safety constraints remain hard.
        """
        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        Nu = Np * m

        p0 = np.asarray(p0, float).reshape(self.np)
        x0 = np.asarray(x0, float).reshape(-1)
        U_guess = np.asarray(U_guess, float).reshape(Np, m)
        idx_ref = np.asarray(idx_ref, int).reshape(Np)
        U_guess_vec = U_guess.reshape(Nu, 1)

        # Prediction matrices.
        p_seq, Mx, Mc, B0 = self._build_prediction_mats(p0, U_guess)
        if Mc.shape != (Np * n, Nu):
            raise ValueError(
                f"Mc has shape {Mc.shape}, expected {(Np * n, Nu)}."
            )

        # Nonlinear nominal rollout at the SQP linearisation point.
        Y_nom = [self._eval_forward_tip(p_seq[k]) for k in range(Np)]
        Y_nom = np.asarray(Y_nom, float).reshape(Np, n)
        X_nom = Y_nom.reshape(Np * n, 1)

        use_affine_matching = bool(getattr(self, "use_affine_matching", True))
        if use_affine_matching:
            X_aff = X_nom - Mc @ U_guess_vec
        else:
            X_aff = (Mx @ x0.reshape(-1, 1)).reshape(Np * n, 1)

        # Reference and tracking objective.
        reference_mode = self._reference_mode()
        path_ref = None
        path_diag: dict = {"reference_mode": reference_mode}

        if reference_mode == "point":
            X_ref = self._build_reference_stack(idx_ref)
            Qtil = self._build_Qtil()
        else:
            path_ref = self._resolve_path_reference(x0)
            X_ref = self._build_continuous_reference_stack(path_ref)
            Qtil = self._build_contouring_Qtil(path_ref)
            path_diag = self._path_tracking_diagnostics(
                X_nom=X_nom,
                path_ref=path_ref,
            )

        progress_mode = self._progress_constraint_mode(
            reference_mode=reference_mode
        )
        use_progress_slack = progress_mode == "soft_slack"
        n_decision = Nu + (1 if use_progress_slack else 0)
        progress_slack_index = Nu if use_progress_slack else -1

        H_track = 2.0 * (Mc.T @ Qtil @ Mc)
        f_track = 2.0 * (Mc.T @ Qtil @ (X_aff - X_ref))
        H = H_track.copy()
        f = f_track.copy()

        # Optional progress reward.  In soft-slack mode the requested progress
        # is represented explicitly by sigma, while this reward can still bias
        # the optimizer toward extra forward motion among otherwise similar safe
        # solutions.
        progress_debug: dict = {}
        A_progress = None
        b_progress = None
        if reference_mode == "contouring":
            A_progress, b_progress, progress_debug = (
                self._build_terminal_progress_model(
                    X_aff=X_aff,
                    Mc=Mc,
                    U_guess=U_guess,
                    path_ref=path_ref,
                )
            )
            progress_reward_weight = float(
                getattr(self, "progress_reward_weight", 0.0)
            )
            if progress_reward_weight < 0.0:
                raise ValueError("progress_reward_weight must be non-negative.")
            if progress_reward_weight > 0.0:
                f += -progress_reward_weight * A_progress.reshape(Nu, 1)
            progress_debug["progress_reward_weight"] = progress_reward_weight

        # Input effort.
        Rtil = np.kron(np.eye(Np), self.R)
        H_effort = 2.0 * Rtil
        f_effort = np.zeros((Nu, 1), float)
        H += H_effort
        f += f_effort

        # Input smoothness.
        H_smooth = np.zeros((Nu, Nu), float)
        f_smooth = np.zeros((Nu, 1), float)
        if np.any(np.diag(self.Rd) > 0.0):
            D_full = np.zeros((Np * m, Nu), float)
            for k in range(Np):
                rows = slice(k * m, (k + 1) * m)
                cols = slice(k * m, (k + 1) * m)
                D_full[rows, cols] = np.eye(m)
                if k > 0:
                    prev_cols = slice((k - 1) * m, k * m)
                    D_full[rows, prev_cols] = -np.eye(m)

            u_prev = np.asarray(
                getattr(self, "u_prev", np.zeros(m, float)),
                float,
            ).reshape(m, 1)
            c_prev = np.zeros((Np * m, 1), float)
            c_prev[:m, :] = u_prev
            Rd_til = np.kron(np.eye(Np), self.Rd)
            H_smooth = 2.0 * (D_full.T @ Rd_til @ D_full)
            f_smooth = -2.0 * (D_full.T @ Rd_til @ c_prev)
            H += H_smooth
            f += f_smooth

        qp_reg = float(getattr(self, "qp_reg", 1e-9))
        H += qp_reg * np.eye(Nu)
        H = 0.5 * (H + H.T)
        f = f.reshape(Nu)

        # --------------------------------------------------------------
        # Optional dimensionless progress-shortfall variable sigma.
        # --------------------------------------------------------------
        progress_slack_guess_fraction = 0.0
        progress_slack_quadratic_weight = 0.0
        progress_slack_linear_weight = 0.0
        if use_progress_slack:
            progress_slack_quadratic_weight = float(
                getattr(self, "progress_slack_quadratic_weight", 1.0e3)
            )
            progress_slack_linear_weight = float(
                getattr(self, "progress_slack_linear_weight", 0.0)
            )
            if progress_slack_quadratic_weight < 0.0:
                raise ValueError(
                    "progress_slack_quadratic_weight must be non-negative."
                )
            if progress_slack_linear_weight < 0.0:
                raise ValueError(
                    "progress_slack_linear_weight must be non-negative."
                )

            H_aug = np.zeros((n_decision, n_decision), dtype=float)
            H_aug[:Nu, :Nu] = H
            # OSQP uses 0.5*z.T*H*z, hence 2*w for w*sigma^2.
            H_aug[progress_slack_index, progress_slack_index] = (
                2.0 * progress_slack_quadratic_weight + qp_reg
            )
            f_aug = np.zeros(n_decision, dtype=float)
            f_aug[:Nu] = f
            f_aug[progress_slack_index] = progress_slack_linear_weight
            H, f = H_aug, f_aug

            effective_request = float(
                progress_debug.get("progress_request_effective_m", 0.0)
            )
            nominal_shortfall = float(
                progress_debug.get("progress_nominal_shortfall_m", 0.0)
            )
            max_fraction = float(
                getattr(self, "progress_slack_max_fraction", 1.0)
            )
            if effective_request > 1.0e-12:
                progress_slack_guess_fraction = float(
                    np.clip(
                        nominal_shortfall / effective_request,
                        0.0,
                        max_fraction,
                    )
                )

        hmpc_diag = symmetric_hessian_diagnostics(H)
        hmpc_diag.update({
            "norm_H_mpc": float(np.linalg.norm(H)),
            "norm_H_track": float(np.linalg.norm(H_track)),
            "norm_H_effort": float(np.linalg.norm(H_effort)),
            "norm_H_smooth": float(np.linalg.norm(H_smooth)),
            "qp_reg": qp_reg,
            "qp_control_dim": int(Nu),
            "qp_decision_dim": int(n_decision),
            "progress_constraint_mode": progress_mode,
            "progress_slack_enabled": int(use_progress_slack),
            "progress_slack_index": int(progress_slack_index),
            "progress_slack_guess_fraction": float(progress_slack_guess_fraction),
            "progress_slack_quadratic_weight": float(progress_slack_quadratic_weight),
            "progress_slack_linear_weight": float(progress_slack_linear_weight),
        })
        # Channel diagnostics concern only physical controls, not sigma.
        hmpc_diag.update(
            mpc_hessian_channel_diagnostics(
                H[:Nu, :Nu],
                Np=self.Np,
                m=self.m,
                cond_warn=5e7,
            )
        )

        # Constraints.
        constraint_blocks = []

        def append_control_constraint(name, A_b, l_b, u_b):
            A_aug = self._augment_control_constraint(
                A_b,
                n_control=Nu,
                n_decision=n_decision,
            )
            constraint_blocks.append((name, A_aug, l_b, u_b))

        u_max = np.asarray(self.u_max, float).reshape(self.m)
        if np.all(np.isfinite(u_max)):
            A_b, l_b, u_b = input_bounds(u_max=u_max, Np=Np)
            append_control_constraint("input_bounds", A_b, l_b, u_b)

        if bool(getattr(self, "use_trust_region", True)):
            tr_l, tr_u = trust_region_bounds(
                U_guess,
                getattr(self, "trust_radius", None),
            )
            if np.any(np.isfinite(tr_l)) or np.any(np.isfinite(tr_u)):
                A_b, l_b, u_b = box_constraint_from_bounds(tr_l, tr_u)
                append_control_constraint("trust_region", A_b, l_b, u_b)

        A_b, l_b, u_b = dL_bounds(
            Np=Np,
            m=m,
            dL_index=int(getattr(self, "dL_index", 6)),
            dL_back_max=float(getattr(self, "dL_back_max", 0.1)),
            dL_fwd_max=float(getattr(self, "dL_fwd_max", np.inf)),
        )
        append_control_constraint("dL_bounds", A_b, l_b, u_b)

        if bool(getattr(self, "enable_hard_epm_tip_clearance", True)):
            A_b, l_b, u_b = self._build_epm_lumen_end_clearance_constraint(
                p0=p0,
                U_guess=U_guess,
            )
            append_control_constraint(
                "epm_lumen_end_clearance", A_b, l_b, u_b
            )

        if bool(getattr(self, "enable_hard_tip_tangent_angle", False)):
            A_b, l_b, u_b, angle_dbg = (
                self._build_tip_tangent_angle_constraint(
                    U_guess=U_guess,
                    X_nom=X_nom,
                    Mc=Mc,
                )
            )
            append_control_constraint("tip_tangent_angle", A_b, l_b, u_b)
        else:
            angle_dbg = {}

        if progress_mode == "hard":
            A_b, l_b, u_b = self._build_terminal_progress_constraint(
                A_progress=A_progress,
                b_progress=b_progress,
                progress_debug=progress_debug,
            )
            append_control_constraint("terminal_progress_hard", A_b, l_b, u_b)
        elif progress_mode == "soft_slack":
            progress_block, sigma_block = self._build_soft_progress_constraint(
                A_progress=A_progress,
                b_progress=b_progress,
                progress_debug=progress_debug,
                n_control=Nu,
                n_decision=n_decision,
                slack_index=progress_slack_index,
            )
            constraint_blocks.append(
                ("terminal_progress_soft", *progress_block)
            )
            constraint_blocks.append(
                ("progress_slack_bounds", *sigma_block)
            )

        progress_debug["progress_constraint_enabled"] = int(
            progress_mode in {"hard", "soft_slack"}
        )
        progress_debug["progress_constraint_mode"] = progress_mode
        progress_debug["progress_slack_enabled"] = int(use_progress_slack)
        progress_debug["progress_slack_index"] = int(progress_slack_index)
        progress_debug["progress_slack_guess_fraction"] = float(
            progress_slack_guess_fraction
        )
        progress_debug["progress_slack_quadratic_weight"] = float(
            progress_slack_quadratic_weight
        )
        progress_debug["progress_slack_linear_weight"] = float(
            progress_slack_linear_weight
        )
        A, l, u, constraint_slices = stack_named_constraints(constraint_blocks)
        if A.size == 0:
            A = np.zeros((0, n_decision), float)
            l = np.zeros(0, float)
            u = np.zeros(0, float)
        if A.shape[1] != n_decision:
            raise ValueError(
                f"A has shape {A.shape}; expected second dimension {n_decision}."
            )

        if use_progress_slack:
            Z_guess_vec = np.concatenate(
                [
                    U_guess_vec.reshape(Nu),
                    np.array([progress_slack_guess_fraction], dtype=float),
                ]
            ).reshape(n_decision, 1)
        else:
            Z_guess_vec = U_guess_vec.reshape(Nu, 1)

        constraint_diag = diagnose_constraints_at_U(
            A=A,
            l=l,
            u=u,
            U=Z_guess_vec,
            constraint_slices=constraint_slices,
            tol=1e-8,
        )
        constraint_diag["constraint_issue_guess"] = classify_constraint_issue(
            constraint_diag
        )

        B_sequence_linearisation = np.asarray(
            getattr(
                self,
                "_last_B_sequence_linearisation",
                np.empty((0, n, m)),
            ),
            float,
        )
        B_pose_nodes_linearisation = np.asarray(
            getattr(
                self,
                "_last_B_pose_nodes",
                np.empty((0, self.np)),
            ),
            float,
        )
        if B_sequence_linearisation.shape != (Np, n, m):
            raise ValueError(
                "Stored Jacobian sequence has shape "
                f"{B_sequence_linearisation.shape}; expected {(Np, n, m)}."
            )
        if B_pose_nodes_linearisation.shape != (Np, self.np):
            raise ValueError(
                "Stored Jacobian pose nodes have shape "
                f"{B_pose_nodes_linearisation.shape}; "
                f"expected {(Np, self.np)}."
            )

        debug = {
            "p_seq": np.asarray(p_seq, float).copy(),
            "X_nom": X_nom.copy(),
            "X_aff": X_aff.copy(),
            "X_ref": X_ref.copy(),
            "Mx": Mx.copy(),
            "Mc": Mc.copy(),
            "B0": np.asarray(B0, float).copy(),
            "idx_ref": idx_ref.copy(),
            "H_track": H_track.copy(),
            "f_track": f_track.copy(),
            "H_effort": H_effort.copy(),
            "H_smooth": H_smooth.copy(),
            "f_smooth": f_smooth.copy(),
            "U_guess": U_guess.copy(),
            "Z_guess": Z_guess_vec.copy(),
            "progress_affine_A": (
                np.empty(0, dtype=float)
                if A_progress is None
                else np.asarray(A_progress, dtype=float).copy()
            ),
            "u_prev": np.asarray(
                getattr(self, "u_prev", np.zeros(m, float)),
                float,
            ).copy(),
            "constraint_slices": constraint_slices,
            "constraint_l": l.copy(),
            "constraint_u": u.copy(),
            **constraint_diag,
            **angle_dbg,
            **path_diag,
            **progress_debug,
            "B_sequence_linearisation": B_sequence_linearisation.copy(),
            "B_pose_nodes_linearisation": B_pose_nodes_linearisation.copy(),
            **hmpc_diag,
        }

        if path_ref is not None:
            debug.update({
                "path_ref_points": np.asarray(path_ref["C_ref"], float).copy(),
                "path_ref_tangents": np.asarray(path_ref["T_ref"], float).copy(),
                "path_ref_s": np.asarray(path_ref["s_ref"], float).copy(),
            })

        return H, f, A, l, u, debug

    @staticmethod
    def _normalise_vector(v, *, name, eps=1e-12):
        v = np.asarray(v, float).reshape(3)
        norm = float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm <= eps:
            raise ValueError(
                f"{name} must have a finite non-zero norm, got {v}."
            )
        return v / norm, norm

    def _local_lumen_geometry(self, point):
        """Continuous nearest-segment lumen geometry at ``point``.

        This intentionally uses segment projection rather than nearest centreline
        *node*.  The latter can falsely report low wall clearance for a tip that
        lies exactly on the centreline between two sparsely sampled nodes.
        """
        if not hasattr(self, "lumen_C"):
            raise AttributeError("Lumen geometry requires self.lumen_C.")
        if not hasattr(self, "lumen_R"):
            raise AttributeError("Lumen geometry requires self.lumen_R.")

        C = np.asarray(self.lumen_C, float)
        if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] < 3:
            raise ValueError(
                "lumen_C must have shape (M, >=3) with M >= 2; "
                f"got {C.shape}."
            )
        C = C[:, :3]
        point = np.asarray(point, float).reshape(3)

        A = C[:-1]
        B = C[1:]
        AB = B - A
        denom = np.sum(AB * AB, axis=1)
        if np.any(denom <= 1.0e-24):
            raise ValueError("lumen_C contains duplicate consecutive points.")

        tau = np.sum((point[None, :] - A) * AB, axis=1) / denom
        tau = np.clip(tau, 0.0, 1.0)
        projected = A + tau[:, None] * AB
        distance_sq = np.sum((point[None, :] - projected) ** 2, axis=1)
        seg = int(np.argmin(distance_sq))
        tau_seg = float(tau[seg])
        centre = projected[seg].copy()
        radial_distance = float(np.sqrt(distance_sq[seg]))

        tangent = AB[seg] / np.sqrt(denom[seg])

        R = np.asarray(self.lumen_R, float)
        if R.ndim == 0 or R.size == 1:
            lumen_radius = float(R.reshape(-1)[0])
        elif R.size == C.shape[0]:
            r_nodes = R.reshape(-1)
            lumen_radius = float(
                (1.0 - tau_seg) * r_nodes[seg]
                + tau_seg * r_nodes[seg + 1]
            )
        else:
            raise ValueError(
                "lumen_R must be scalar or contain one radius per lumen node; "
                f"got shape {R.shape} for {C.shape[0]} points."
            )
        if not np.isfinite(lumen_radius) or lumen_radius <= 0.0:
            raise ValueError(
                f"Local lumen radius must be finite and positive; got {lumen_radius}."
            )

        return tangent, centre, seg, tau_seg, lumen_radius, radial_distance

    def _local_lumen_tangent(self, point):
        tangent, _, segment, _, _, _ = self._local_lumen_geometry(point)
        return np.asarray(tangent, float).reshape(3), int(segment)

    def _tip_wall_clearance(self, point):
        """Signed radial tip-surface clearance using continuous segment projection."""
        (
            _,
            _,
            segment,
            _,
            lumen_radius,
            radial_distance,
        ) = self._local_lumen_geometry(point)

        tip_radius = float(getattr(self, "tip_radius_m", 0.0))
        if not np.isfinite(tip_radius) or tip_radius < 0.0:
            raise ValueError(
                f"tip_radius_m must be finite and non-negative; got {tip_radius}."
            )

        clearance = lumen_radius - radial_distance - tip_radius
        return (
            float(clearance),
            int(segment),
            float(lumen_radius),
            float(radial_distance),
        )


    def _build_tip_tangent_angle_constraint(
        self,
        *,
        U_guess,
        X_nom,
        Mc,
    ):
        """
        Build a proximity-gated hard tangent-angle constraint.

        The angle limit is enforced only when the nominal predicted tip is
        close to the vessel wall:

            clearance_nom <= activation_clearance + activation_guard

        Active nonlinear stage constraint:

            g(U) = t_lumen.T normalise(t_tip(U))
                   - cos(alpha_max) >= 0

        The guard distance activates the QP row slightly before the true
        nonlinear safety threshold. The exact nonlinear candidate check uses
        ``activation_clearance`` without the guard.
        """
        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        Nu = Np * m

        if n < 6:
            raise ValueError(
                "Hard tip-tangent angle constraint requires n_out >= 6 "
                "with output [tip_xyz, tip_tangent]."
            )

        U_guess = np.asarray(U_guess, float).reshape(Np, m)
        U_guess_vec = U_guess.reshape(Nu)

        X_nom = np.asarray(X_nom, float).reshape(Np, n)
        Mc = np.asarray(Mc, float).reshape(Np * n, Nu)

        alpha_deg = float(
            getattr(
                self,
                "tip_tangent_max_angle_deg",
                40.0,
            )
        )
        activation_clearance = float(
            getattr(
                self,
                "tip_tangent_activation_clearance_m",
                1.0e-3,
            )
        )
        activation_guard = float(
            getattr(
                self,
                "tip_tangent_activation_guard_m",
                0.5e-3,
            )
        )

        if not (0.0 < alpha_deg < 180.0):
            raise ValueError(
                "tip_tangent_max_angle_deg must lie in (0, 180)."
            )
        if activation_clearance < 0.0:
            raise ValueError(
                "tip_tangent_activation_clearance_m must be non-negative."
            )
        if activation_guard < 0.0:
            raise ValueError(
                "tip_tangent_activation_guard_m must be non-negative."
            )

        cos_limit = float(np.cos(np.deg2rad(alpha_deg)))

        rows = []
        lower = []
        active_stages = []

        nominal_angle_deg = np.full(Np, np.nan, dtype=float)
        nominal_margin = np.full(Np, np.nan, dtype=float)
        nominal_clearance_m = np.full(Np, np.nan, dtype=float)
        nominal_radius_m = np.full(Np, np.nan, dtype=float)
        nominal_radial_distance_m = np.full(Np, np.nan, dtype=float)
        lumen_indices = np.full(Np, -1, dtype=int)
        active_mask = np.zeros(Np, dtype=bool)

        eye3 = np.eye(3)
        qp_activation_threshold = (
            activation_clearance + activation_guard
        )

        for stage in range(Np):
            tip_nom = X_nom[stage, 0:3]
            tangent_raw = X_nom[stage, 3:6]

            tangent_hat, tangent_norm = self._normalise_vector(
                tangent_raw,
                name=f"nominal tip tangent at stage {stage}",
            )
            lumen_tangent, lumen_index = (
                self._local_lumen_tangent(tip_nom)
            )
            (
                clearance_m,
                clearance_index,
                lumen_radius_m,
                radial_distance_m,
            ) = self._tip_wall_clearance(tip_nom)

            # Both helpers use the nearest centreline point. Keep the
            # clearance index because it is directly associated with radius.
            lumen_index = int(clearance_index)

            cosine_nom = float(
                np.clip(
                    np.dot(lumen_tangent, tangent_hat),
                    -1.0,
                    1.0,
                )
            )
            g_nom = cosine_nom - cos_limit

            nominal_angle_deg[stage] = float(
                np.degrees(np.arccos(cosine_nom))
            )
            nominal_margin[stage] = g_nom
            nominal_clearance_m[stage] = clearance_m
            nominal_radius_m[stage] = lumen_radius_m
            nominal_radial_distance_m[stage] = radial_distance_m
            lumen_indices[stage] = lumen_index

            active = bool(
                clearance_m <= qp_activation_threshold
            )
            active_mask[stage] = active

            if not active:
                continue

            dnormalise_dt = (
                eye3 - np.outer(tangent_hat, tangent_hat)
            ) / tangent_norm
            dg_dt = (
                lumen_tangent.reshape(1, 3)
                @ dnormalise_dt
            ).reshape(3)

            tangent_rows = slice(
                stage * n + 3,
                stage * n + 6,
            )
            Mc_tangent = Mc[tangent_rows, :]

            a = (
                dg_dt.reshape(1, 3)
                @ Mc_tangent
            ).reshape(Nu)
            b = g_nom - float(a @ U_guess_vec)

            rows.append(a)
            lower.append(-b)
            active_stages.append(stage)

        # Keep one row per horizon stage so the optimized persistent OSQP
        # workspace does not change dimensions as the proximity gate switches
        # individual stages on and off.  Inactive rows are exactly unconstrained.
        A_angle = np.zeros((Np, Nu), dtype=float)
        l_angle = np.full(Np, -np.inf, dtype=float)
        u_angle = np.full(Np, np.inf, dtype=float)
        for row_vec, lower_value, stage in zip(rows, lower, active_stages):
            A_angle[int(stage), :] = np.asarray(row_vec, dtype=float).reshape(Nu)
            l_angle[int(stage)] = float(lower_value)

        debug = {
            "tip_tangent_angle_limit_deg": alpha_deg,
            "tip_tangent_activation_clearance_m": (
                activation_clearance
            ),
            "tip_tangent_activation_guard_m": activation_guard,
            "tip_tangent_qp_activation_threshold_m": (
                qp_activation_threshold
            ),
            "tip_tangent_angle_nominal_deg": nominal_angle_deg.copy(),
            "tip_tangent_angle_nominal_max_deg": float(
                np.nanmax(nominal_angle_deg)
            ),
            "tip_tangent_angle_nominal_margin": nominal_margin.copy(),
            "tip_tangent_nominal_clearance_m": (
                nominal_clearance_m.copy()
            ),
            "tip_tangent_nominal_clearance_mm": (
                1e3 * nominal_clearance_m
            ),
            "tip_tangent_nominal_lumen_radius_m": (
                nominal_radius_m.copy()
            ),
            "tip_tangent_nominal_radial_distance_m": (
                nominal_radial_distance_m.copy()
            ),
            "tip_tangent_constraint_active_mask": active_mask.copy(),
            "tip_tangent_constraint_active_stages": np.asarray(
                active_stages,
                dtype=int,
            ),
            "tip_tangent_constraint_num_active": int(
                len(active_stages)
            ),
            "tip_tangent_lumen_index": lumen_indices.copy(),
        }

        return A_angle, l_angle, u_angle, debug

    def _build_reference_stack(self, idx_ref):
        n = int(self.n)
        Np = int(self.Np)

        if not hasattr(self, "lumen_C"):
            raise AttributeError(
                "Controller must have self.lumen_C before building QP."
            )

        Cc = np.asarray(self.lumen_C, float)

        if Cc.ndim != 2:
            raise ValueError(f"lumen_C must be 2D, got {Cc.shape}.")

        if Cc.shape[1] < min(3, n):
            raise ValueError(
                f"lumen_C has too few columns for n={n}; got shape {Cc.shape}."
            )

        idx_ref = np.asarray(idx_ref, int).reshape(Np)

        X_ref = np.zeros((Np * n, 1), float)

        n_copy = min(3, n, Cc.shape[1])

        for k in range(Np):
            idx = int(np.clip(idx_ref[k], 0, Cc.shape[0] - 1))
            row0 = k * n
            X_ref[row0:row0 + n_copy, 0] = Cc[idx, :n_copy]

            # When the controller output contains a tip tangent, provide the
            # local lumen tangent as its point-mode reference. Zero tangent
            # weights still mean no tangent tracking.
            if n >= 6:
                tangent, _ = self._local_lumen_tangent(Cc[idx, :3])
                X_ref[row0 + 3:row0 + 6, 0] = tangent

        return X_ref
    def _build_Qtil(self):
        """
        Build stacked tracking matrix Qtil.

        For normal robot tip tracking, you may want position-only tracking.
        For full-state benchmarks, use all output channels.
        """
        n = int(self.n)
        Np = int(self.Np)

        Qtil = np.zeros((Np * n, Np * n), float)

        stage_weights = np.asarray(
            getattr(self, "ref_stage_weights", np.ones(Np)),
            float,
        ).reshape(-1)

        if stage_weights.size != Np:
            raise ValueError(
                f"ref_stage_weights must have length {Np}, got {stage_weights.size}."
            )

        # Weight entries are the primary switch. With n=6 and tangent
        # weights set to zero, only tip position is tracked. Set positive
        # tangent weights to enable tangent tracking explicitly.
        track_all_outputs = bool(getattr(self, "track_all_outputs", True))

        for k in range(Np):
            Qk = np.zeros((n, n), float)

            if track_all_outputs:
                Qk[:, :] = stage_weights[k] * self.Q
            else:
                n_pos = min(3, n)
                Qk[:n_pos, :n_pos] = stage_weights[k] * self.Q[:n_pos, :n_pos]

            Qtil[k * n:(k + 1) * n, k * n:(k + 1) * n] = Qk
        return Qtil
    def _build_Qtil_position_only(self):
        """
        Build stacked tracking matrix Qtil.

        Only position channels 0:3 are penalised by default.

        If self.n == 3:
            this is equivalent to full-output position tracking.

        If self.n > 3:
            channels 3: are not penalised unless you modify this function.
        """
        n = int(self.n)
        Np = int(self.Np)

        Qtil = np.zeros((Np * n, Np * n), float)

        stage_weights = np.asarray(
            getattr(self, "ref_stage_weights", np.ones(Np)),
            float,
        ).reshape(-1)

        if stage_weights.size != Np:
            raise ValueError(
                f"ref_stage_weights must have length {Np}, "
                f"got {stage_weights.size}."
            )

        for k in range(Np):
            Qk = np.zeros((n, n), float)

            # Position-only cost.
            n_pos = min(3, n)
            Qk[:n_pos, :n_pos] = (
                stage_weights[k] * self.Q[:n_pos, :n_pos]
            )

            Qtil[k * n:(k + 1) * n, k * n:(k + 1) * n] = Qk

        return Qtil
    def _build_epm_lumen_end_clearance_constraint(
        self,
        *,
        p0,
        U_guess,
    ):
        """
        Linearised hard minimum distance between external magnet centre and
        the fixed distal end of the lumen.

        Constraint:
            ||r_m(U) - x_end|| >= d_min

        where x_end is independent of the beam output.

        Linearised around U_guess:

            d(U) ≈ b + a @ U

        Enforce:
            b + a @ U >= d_min

        OSQP form:
            l <= A @ U <= u

        so:
            A[k, :] = a_k
            l[k] = d_min - b_k
            u[k] = +inf
        """

        m = int(self.m)
        Np = int(self.Np)
        Nu = Np * m

        p0 = np.asarray(p0, float).reshape(self.np)
        U_guess = np.asarray(U_guess, float).reshape(Np, m)
        U_guess_vec = U_guess.reshape(Nu, 1)

        d_min = float(getattr(self, "epm_lumen_end_hard_min_m", 0.21))
        # print(f"Constraint d_min is : {d_min}")
        if d_min < 0.0:
            raise ValueError("epm_lumen_end_hard_min_m must be non-negative.")

        # Use explicitly supplied lumen end if available, otherwise use final lumen point.
        if hasattr(self, "epm_clearance_point_world"):
            x_end = np.asarray(self.epm_clearance_point_world, float).reshape(3)
        else:
            if not hasattr(self, "lumen_C"):
                raise AttributeError(
                    "Controller must have self.lumen_C or "
                    "self.epm_clearance_point_world for lumen-end clearance."
                )

            Cc = np.asarray(self.lumen_C, float)

            if Cc.ndim != 2 or Cc.shape[1] < 3 or Cc.shape[0] < 1:
                raise ValueError(
                    f"lumen_C must have shape (M, >=3), got {Cc.shape}."
                )

            x_end = Cc[-1, :3].astype(float)

        # Pm maps stacked controls U to stacked magnet positions:
        #     r_m_stack = r0_stack + Pm @ U
        Pm = build_Pm_world(
            dt=float(self.dt),
            Np=Np,
            m=m,
        )

        Pm = np.asarray(Pm, float)

        expected_shape = (3 * Np, Nu)
        if Pm.shape != expected_shape:
            raise ValueError(
                f"build_Pm_world returned {Pm.shape}, expected {expected_shape}."
            )

        r0 = p0[:3].reshape(3, 1)
        r0_stack = np.tile(r0, (Np, 1))
        r_nom_stack = r0_stack + Pm @ U_guess_vec

        A_clear = np.zeros((Np, Nu), float)
        l_clear = np.full(Np, -np.inf, float)
        u_clear = np.full(Np, +np.inf, float)

        eps_dist = 1e-9

        for k in range(Np):
            rows_r = slice(3 * k, 3 * (k + 1))

            Pm_k = Pm[rows_r, :]
            r_nom_k = r_nom_stack[rows_r, 0]

            v_nom = r_nom_k - x_end
            d_nom = float(np.linalg.norm(v_nom))

            if d_nom < eps_dist:
                direction = np.array([1.0, 0.0, 0.0], float)
                d_nom = eps_dist
            else:
                direction = v_nom / d_nom

            # r(U) - x_end = r0 + Pm U - x_end
            #
            # Linearised norm:
            # d(U) ≈ d_nom + direction.T @ Pm_k @ (U - U_guess)
            #
            # Rearranged:
            # d(U) ≈ b + a @ U
            a = (direction.reshape(1, 3) @ Pm_k).reshape(Nu)
            b = d_nom - float(a @ U_guess_vec[:, 0])

            A_clear[k, :] = a
            l_clear[k] = d_min - b
            u_clear[k] = np.inf

        return A_clear, l_clear, u_clear
    def _build_epm_tip_clearance_constraint(
        self,
        *,
        p0,
        U_guess,
        X_aff,
        Mc,
    ):
        """
        Linearised hard minimum distance between external magnet centre and tip.

        Constraint:
            ||r_m(U) - x_tip(U)|| >= d_min

        Linearised around U_guess:

            d(U) ≈ b + a @ U

        Enforce:
            b + a @ U >= d_min

        OSQP form:
            l <= A @ U <= u

        so:
            A[k, :] = a_k
            l[k] = d_min - b_k
            u[k] = +inf
        """
        n = int(self.n)
        m = int(self.m)
        Np = int(self.Np)
        Nu = Np * m

        p0 = np.asarray(p0, float).reshape(self.np)
        U_guess = np.asarray(U_guess, float).reshape(Np, m)
        U_guess_vec = U_guess.reshape(Nu, 1)

        X_aff = np.asarray(X_aff, float).reshape(Np * n, 1)
        Mc = np.asarray(Mc, float).reshape(Np * n, Nu)

        d_min = float(getattr(self, "epm_tip_hard_min_m", 0.005))

        if d_min < 0.0:
            raise ValueError("epm_tip_hard_min_m must be non-negative.")

        # Pm maps stacked controls U to stacked magnet positions:
            # r_m_stack = r0_stack + Pm @ U
        Pm = build_Pm_world(
            dt=float(self.dt),
            Np=Np,
            m=m,
        )

        Pm = np.asarray(Pm, float)

        expected_shape = (3 * Np, Nu)
        if Pm.shape != expected_shape:
            raise ValueError(
                f"build_Pm_world returned {Pm.shape}, "
                f"expected {expected_shape}."
            )

        r0 = p0[:3].reshape(3, 1)
        r0_stack = np.tile(r0, (Np, 1))

        r_nom_stack = r0_stack + Pm @ U_guess_vec

        A_clear = np.zeros((Np, Nu), float)
        l_clear = np.full(Np, -np.inf, float)
        u_clear = np.full(Np, +np.inf, float)

        eps_dist = 1e-9

        for k in range(Np):
            rows_r = slice(3 * k, 3 * (k + 1))
            rows_x = np.array(
                [k * n + 0, k * n + 1, k * n + 2],
                dtype=int,
            )

            Pm_k = Pm[rows_r, :]
            Mc_xk = Mc[rows_x, :]

            r_nom_k = r_nom_stack[rows_r, 0]

            # Because X_aff + Mc @ U_guess = X_nom
            x_nom_k = X_aff[rows_x, 0] + (Mc_xk @ U_guess_vec).reshape(3)

            v_nom = r_nom_k - x_nom_k
            d_nom = float(np.linalg.norm(v_nom))

            if d_nom < eps_dist:
                direction = np.array([1.0, 0.0, 0.0], float)
                d_nom = eps_dist
            else:
                direction = v_nom / d_nom

            # r(U) - x(U) = [r0 + Pm U] - [X_aff_x + Mc_x U]
            #
            # Linearised norm:
            # d(U) ≈ d_nom + direction.T @ (
            #           (Pm_k - Mc_xk) @ (U - U_guess)
            #       )
            #
            # Rearranged as:
            # d(U) ≈ b + a @ U
            a = (direction.reshape(1, 3) @ (Pm_k - Mc_xk)).reshape(Nu)
            b = d_nom - float(a @ U_guess_vec[:, 0])

            A_clear[k, :] = a
            l_clear[k] = d_min - b
            u_clear[k] = np.inf

        return A_clear, l_clear, u_clear