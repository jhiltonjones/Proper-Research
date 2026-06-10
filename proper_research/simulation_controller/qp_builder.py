import numpy as np

from proper_research.simulation_controller.constraints import (
    input_bounds,
    trust_region_bounds,
    box_constraint_from_bounds,
    dL_bounds,
    stack_constraints,
)
from .diagnostics import mpc_hessian_channel_diagnostics
from .geometry import build_Pm_world

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
    def _build_mpc_qp(self, *, p0, x0, U_guess, idx_ref):
        """
        Build one local MPC QP around U_guess.

        Decision variable:
            U = [u_0, u_1, ..., u_{Np-1}]
            shape: (Np*m,)

        Local prediction model:
            X(U) ≈ X_nom + Mc @ (U - U_guess)

        Equivalently:
            X(U) ≈ X_aff + Mc @ U

        where:
            X_aff = X_nom - Mc @ U_guess

        Objective:
            tracking + effort + smoothness

        Constraints:
            input bounds
            trust region around U_guess
            insertion dL bounds
            hard EPM-tip distance constraint

        Returns:
            H, f, A, l, u, debug

        OSQP convention:
            minimize 0.5 * U.T @ H @ U + f.T @ U
            subject to l <= A @ U <= u
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

        # ------------------------------------------------------------
        # Prediction matrices
        # ------------------------------------------------------------
        p_seq, Mx, Mc, B0 = self._build_prediction_mats(
            p0,
            U_guess,
        )

        if Mc.shape != (Np * n, Nu):
            raise ValueError(
                f"Mc has shape {Mc.shape}, expected {(Np * n, Nu)}."
            )

        # ------------------------------------------------------------
        # Nonlinear nominal rollout at the SQP linearisation point
        # ------------------------------------------------------------
        Y_nom = []

        for k in range(Np):
            yk = self._eval_forward_tip(p_seq[k])
            Y_nom.append(yk)

        Y_nom = np.asarray(Y_nom, float).reshape(Np, n)
        X_nom = Y_nom.reshape(Np * n, 1)

        # ------------------------------------------------------------
        # Affine matching
        # ------------------------------------------------------------
        use_affine_matching = bool(getattr(self, "use_affine_matching", True))

        if use_affine_matching:
            # X ≈ X_nom + Mc (U - U_guess)
            #   = X_aff + Mc U
            X_aff = X_nom - Mc @ U_guess_vec
        else:
            # Less accurate for nonlinear implicit model, but useful for testing.
            X_aff = (Mx @ x0.reshape(-1, 1)).reshape(Np * n, 1)

        # ------------------------------------------------------------
        # Reference stack
        # ------------------------------------------------------------
        X_ref = self._build_reference_stack(idx_ref)

        # ------------------------------------------------------------
        # Objective: tracking
        # ------------------------------------------------------------
        Qtil = self._build_Qtil_position_only()

        H_track = 2.0 * (Mc.T @ Qtil @ Mc)
        f_track = 2.0 * (Mc.T @ Qtil @ (X_aff - X_ref))

        H = H_track.copy()
        f = f_track.copy()

        # ------------------------------------------------------------
        # Objective: input effort
        # ------------------------------------------------------------
        Rtil = np.kron(np.eye(Np), self.R)

        H_effort = 2.0 * Rtil
        f_effort = np.zeros((Nu, 1), float)

        H += H_effort
        f += f_effort

        # ------------------------------------------------------------
        # Objective: input smoothness
        # ------------------------------------------------------------
        H_smooth = np.zeros((Nu, Nu), float)

        if Np > 1 and np.any(np.diag(self.Rd) > 0.0):
            Rd_til = np.kron(np.eye(Np - 1), self.Rd)
            H_smooth = 2.0 * (self.Du.T @ Rd_til @ self.Du)
            H += H_smooth

        # ------------------------------------------------------------
        # Numerical regularisation
        # ------------------------------------------------------------
        qp_reg = float(getattr(self, "qp_reg", 1e-9))
        H += qp_reg * np.eye(Nu)

        H = 0.5 * (H + H.T)
        f = f.reshape(Nu)
        # ------------------------------------------------------------
        # MPC Hessian diagnostics
        # ------------------------------------------------------------
        hmpc_diag = symmetric_hessian_diagnostics(H)
        hmpc_diag.update({
            "norm_H_mpc": float(np.linalg.norm(H)),
            "norm_H_track": float(np.linalg.norm(H_track)),
            "norm_H_effort": float(np.linalg.norm(H_effort)),
            "norm_H_smooth": float(np.linalg.norm(H_smooth)),
            "qp_reg": float(qp_reg),
        })
        mpc_channel_diag = mpc_hessian_channel_diagnostics(
            H,
            Np=self.Np,
            m=self.m,
            cond_warn=5e7,
        )

        hmpc_diag.update(mpc_channel_diag)
        # ------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------
        constraint_blocks = []

        # 1. Physical symmetric input bounds: -u_max <= U <= u_max
        u_max = np.asarray(self.u_max, float).reshape(self.m)

        if np.all(np.isfinite(u_max)):
            constraint_blocks.append(
                input_bounds(
                    u_max=u_max,
                    Np=Np,
                )
            )

        # 2. Trust region: U_guess - tr <= U <= U_guess + tr
        tr_l, tr_u = trust_region_bounds(
            U_guess,
            getattr(self, "trust_radius", None),
        )

        if np.any(np.isfinite(tr_l)) or np.any(np.isfinite(tr_u)):
            constraint_blocks.append(
                box_constraint_from_bounds(
                    tr_l,
                    tr_u,
                )
            )

        # 3. Insertion bounds on dL
        constraint_blocks.append(
            dL_bounds(
                Np=Np,
                m=m,
                dL_index=int(getattr(self, "dL_index", 6)),
                dL_back_max=float(getattr(self, "dL_back_max", 0.002)),
                dL_fwd_max=float(getattr(self, "dL_fwd_max", np.inf)),
            )
        )

        # 4. Hard EPM-tip clearance
        if bool(getattr(self, "enable_hard_epm_tip_clearance", True)):
            constraint_blocks.append(
                self._build_epm_tip_clearance_constraint(
                    p0=p0,
                    U_guess=U_guess,
                    X_aff=X_aff,
                    Mc=Mc,
                )
            )

        A, l, u = stack_constraints(constraint_blocks)

        # stack_constraints returns (0, 0) if no blocks exist.
        # For this builder, Nu is known, so enforce correct empty shape.
        if A.size == 0:
            A = np.zeros((0, Nu), float)
            l = np.zeros(0, float)
            u = np.zeros(0, float)

        if A.shape[1] != Nu:
            raise ValueError(
                f"A has shape {A.shape}; expected second dimension {Nu}."
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
            "U_guess": U_guess.copy(),

            # MPC Hessian diagnostics
            **hmpc_diag,
        }

        return H, f, A, l, u, debug

    def _build_reference_stack(self, idx_ref):
        """
        Build stacked reference vector X_ref with shape (Np*n, 1).

        Currently position-only reference:
            x, y, z come from self.lumen_C
            all other output channels are zero

        The cost matrix decides which channels are penalised.
        """
        n = int(self.n)
        Np = int(self.Np)

        if not hasattr(self, "lumen_C"):
            raise AttributeError(
                "Controller must have self.lumen_C before building QP."
            )

        Cc = np.asarray(self.lumen_C, float)

        if Cc.ndim != 2 or Cc.shape[1] < 3:
            raise ValueError(
                f"lumen_C must have shape (M, >=3), got {Cc.shape}."
            )

        idx_ref = np.asarray(idx_ref, int).reshape(Np)

        X_ref = np.zeros((Np * n, 1), float)

        for k in range(Np):
            idx = int(np.clip(idx_ref[k], 0, Cc.shape[0] - 1))

            X_ref[k * n + 0, 0] = Cc[idx, 0]
            X_ref[k * n + 1, 0] = Cc[idx, 1]
            X_ref[k * n + 2, 0] = Cc[idx, 2]

        return X_ref

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

        d_min = float(getattr(self, "epm_tip_hard_min_m", 0.011))

        if d_min < 0.0:
            raise ValueError("epm_tip_hard_min_m must be non-negative.")

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