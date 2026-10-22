import numpy as np

from proper_research.simulation_controller.prediction import seq_mat_lti, seq_mat_ltv, build_Du_matrix
from proper_research.simulation_controller.reference import ReferenceMixin
from proper_research.simulation_controller.rollout import RolloutMixin
from proper_research.simulation_controller.qp_builder import QPBuilderMixin
from proper_research.simulation_controller.osqp_solver import solve_qp_osqp
def _contact_provenance(model):
    if model is None:
        return {
            "available": False,
            "enabled": None,
            "use_in_jacobian": None,
            "lumen_attached": None,
            "model_type": None,
        }

    cfg = getattr(model, "contact_cfg", None)

    return {
        "available": True,
        "enabled": (
            None if cfg is None
            else bool(cfg.enabled)
        ),
        "use_in_jacobian": (
            None if cfg is None
            else bool(cfg.use_in_jacobian)
        ),
        "lumen_attached": (
            getattr(model, "lumen_query", None)
            is not None
        ),
        "model_type": type(model).__name__,
    }
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
        enable_adaptive_trust_region=True,
        use_trust_region = True,
        trust_radius_min=None,
        trust_radius_max=None,
        trust_shrink=0.5,
        trust_expand=1.1,
        trust_linerr_tighten_m=5.0e-3,
        trust_linerr_relax_m=1.0e-4,
        trust_probe_fraction=1.0,
        trust_probe_stages=1,
        epm_tip_hard_min_m=0.011,
        enable_hard_epm_tip_clearance=True,
        dL_index=6,
        dL_back_max=0.002,
        dL_fwd_max=np.inf,
        enable_hard_tip_tangent_angle=False,
        tip_tangent_max_angle_deg=40.0,
        tip_tangent_nonlinear_tol_deg=0.25,
        tip_tangent_activation_clearance_m=1.0e-3,
        tip_tangent_activation_guard_m=0.5e-3,
        tip_radius_m=0.0,
        qp_reg=1e-9,
        
    ):
        self.Jxy_fn = Jxy_fn
        self.forward_tip_fn = forward_tip_fn
        self.use_trust_region = use_trust_region
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
        # self.N_sqp = 20          # safety cap only
        self.sqp_tol_abs = 2e-2  # 0.1% of channel limit
        self.sqp_tol_rel = 5e-2  # 1% relative change

        self.dL_index = int(dL_index)
        self.dL_back_max = float(dL_back_max)
        self.dL_fwd_max = float(dL_fwd_max)

        self.trust_radius = None if trust_radius is None else np.asarray(
            trust_radius,
            float,
        ).reshape(-1)

        # Adaptive trust-region constraint tightening.
        #
        # This does not reject solved QP inputs.  Instead, before each QP solve
        # it probes the local nonlinear-vs-affine prediction mismatch and
        # tightens self.trust_radius, which is then used by _build_mpc_qp to
        # form the trust-region bounds.
        self.enable_adaptive_trust_region = bool(enable_adaptive_trust_region)
        self.trust_radius_nominal = (
            None if self.trust_radius is None else self.trust_radius.copy()
        )
        self.trust_radius_min = (
            None
            if trust_radius_min is None
            else np.asarray(trust_radius_min, float).reshape(-1)
        )
        self.trust_radius_max = (
            None
            if trust_radius_max is None
            else np.asarray(trust_radius_max, float).reshape(-1)
        )
        self.trust_shrink = float(trust_shrink)
        self.trust_expand = float(trust_expand)
        self.trust_linerr_tighten_m = float(trust_linerr_tighten_m)
        self.trust_linerr_relax_m = float(trust_linerr_relax_m)
        self.trust_probe_fraction = float(trust_probe_fraction)
        self.trust_probe_stages = int(trust_probe_stages)
        self.trust_last_action = "fixed"
        self.trust_last_linerr_max_m = np.nan
        self.trust_last_linerr_first_m = np.nan
        self.trust_last_probe_max_err_m = np.nan
        self.trust_last_probe_changed = 0

        if not (0.0 < self.trust_shrink <= 1.0):
            raise ValueError("trust_shrink must be in (0, 1].")

        if self.trust_expand < 1.0:
            raise ValueError("trust_expand must be >= 1.")

        if self.trust_probe_fraction <= 0.0:
            raise ValueError("trust_probe_fraction must be positive.")

        if self.trust_probe_stages <= 0:
            raise ValueError("trust_probe_stages must be positive.")

        self.enable_hard_epm_tip_clearance = bool(enable_hard_epm_tip_clearance)
        self.epm_tip_hard_min_m = float(epm_tip_hard_min_m)

        self.enable_hard_tip_tangent_angle = bool(
            enable_hard_tip_tangent_angle
        )
        self.tip_tangent_max_angle_deg = float(
            tip_tangent_max_angle_deg
        )
        self.tip_tangent_nonlinear_tol_deg = float(
            tip_tangent_nonlinear_tol_deg
        )
        self.tip_tangent_activation_clearance_m = float(
            tip_tangent_activation_clearance_m
        )
        self.tip_tangent_activation_guard_m = float(
            tip_tangent_activation_guard_m
        )
        self.tip_radius_m = float(tip_radius_m)

        if not (
            0.0
            < self.tip_tangent_max_angle_deg
            < 180.0
        ):
            raise ValueError(
                "tip_tangent_max_angle_deg must lie in (0, 180)."
            )

        if self.tip_tangent_nonlinear_tol_deg < 0.0:
            raise ValueError(
                "tip_tangent_nonlinear_tol_deg must be non-negative."
            )
        if self.tip_tangent_activation_clearance_m < 0.0:
            raise ValueError(
                "tip_tangent_activation_clearance_m must be non-negative."
            )
        if self.tip_tangent_activation_guard_m < 0.0:
            raise ValueError(
                "tip_tangent_activation_guard_m must be non-negative."
            )
        if self.tip_radius_m < 0.0:
            raise ValueError(
                "tip_radius_m must be non-negative."
            )

        if (
            self.enable_hard_tip_tangent_angle
            and self.n < 6
        ):
            raise ValueError(
                "Hard tip-tangent angle constraint requires "
                "n_out >= 6 and output "
                "[tip_xyz, tip_tangent]."
            )

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

        if self.enable_adaptive_trust_region and self.trust_radius is None:
            # If adaptation is enabled without an explicit radius, create a
            # conservative initial radius from the finite physical input bounds.
            # Infinite bounds are mapped to 1.0 in their native units.
            u_scale = np.where(
                np.isfinite(self.u_max),
                np.maximum(np.abs(self.u_max), 1e-6),
                1.0,
            )
            self.trust_radius = 0.25 * u_scale
            self.trust_radius_nominal = self.trust_radius.copy()

        if self.enable_adaptive_trust_region and self.trust_radius is not None:
            self.trust_radius = self._clip_trust_radius(self.trust_radius)
            if self.trust_radius_nominal is None:
                self.trust_radius_nominal = self.trust_radius.copy()

        self.p = None
        self.x = None
        self.d = np.zeros(self.n, float)

        self.U_warm = None
        self.u_prev = np.zeros(self.m, float)
        self.Du = build_Du_matrix(self.Np, self.m)
        self.output_bias = np.zeros(self.n)
        self.ref_stride_pts = 1
        self.ref_stage_weights = np.ones(self.Np, dtype=float)
        self.reference_mode = "point"

        self.i_ref_last = 0
        self.idx_ref_last = None
        self.i_closest_last = None
        self.dist_to_ref_last = None
        self.use_affine_matching = True
    def set_output_bias(self, bias):
        bias = np.asarray(bias, dtype=float).reshape(-1)
        if bias.size != self.n:
            raise ValueError(f"output bias size {bias.size} != n={self.n}")
        if not np.all(np.isfinite(bias)):
            raise ValueError(f"non-finite output bias: {bias}")
        self.output_bias = bias.copy()
    def set_initial_params(self, p0):
        self.p = np.asarray(p0, float).reshape(self.np,)
        self.u_prev = np.zeros(self.m)
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
        if hasattr(self, "reset_path_reference_state"):
            self.reset_path_reference_state()
    def set_prediction_horizon(self, Np: int, *, reset_warm: bool = False):
        """
        Safely change the MPC prediction horizon online.

        The controller builds prediction matrices from self.Np each solve, so the
        main horizon-dependent cached objects that must be updated here are:
            - self.Np
            - self.Du
            - self.ref_stage_weights
            - self.U_warm shape

        Parameters
        ----------
        Np:
            New prediction horizon.

        reset_warm:
            If True, discard the warm start. If False, resize it by truncating or
            padding with the last available control.
        """
        Np_new = int(Np)

        if Np_new <= 0:
            raise ValueError(f"Np must be positive, got {Np_new}.")

        Np_old = int(getattr(self, "Np", Np_new))

        if Np_new == Np_old:
            return

        old_U_warm = None
        if getattr(self, "U_warm", None) is not None:
            try:
                old_U_warm = np.asarray(self.U_warm, float).reshape(Np_old, self.m)
            except Exception:
                old_U_warm = None

        self.Np = int(Np_new)
        self.Du = build_Du_matrix(self.Np, self.m)
        # self.ref_stage_weights = np.ones(self.Np)

        if reset_warm or old_U_warm is None:
            self.U_warm = None
            return

        U_new = np.zeros((self.Np, self.m), dtype=float)
        K = min(old_U_warm.shape[0], self.Np)

        if K > 0:
            U_new[:K, :] = old_U_warm[:K, :]

        if self.Np > K and K > 0:
            U_new[K:, :] = old_U_warm[K - 1, :]

        self.U_warm = U_new.reshape(-1)
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
            p_nodes = self._p_nodes_from_U(
                p0,
                U_guess,
            )
            p_seq = p_nodes[1:]

            B0 = np.asarray(
                self.Jxy_fn(p_nodes[0]),
                float,
            ).reshape(n, m)

            B_sequence = np.repeat(
                B0.reshape(1, n, m),
                Np,
                axis=0,
            )

            self._last_B_sequence_linearisation = (
                B_sequence.copy()
            )
            self._last_B_pose_nodes = (
                np.asarray(p_nodes[:-1], float).copy()
            )

            Mx, Mc = seq_mat_lti(
                self.A,
                B0,
                Np,
            )

            return p_seq, Mx, Mc, B0

        if self.model_mode == "ltv":
            p_nodes = self._p_nodes_from_U(
                p0,
                U_guess,
            )
            p_seq = p_nodes[1:]

            B_list = []

            for stage in range(Np):
                B_stage = np.asarray(
                    self.Jxy_fn(p_nodes[stage]),
                    float,
                ).reshape(n, m)

                B_list.append(B_stage)

            B_sequence = np.stack(
                B_list,
                axis=0,
            )

            self._last_B_sequence_linearisation = (
                B_sequence.copy()
            )
            self._last_B_pose_nodes = (
                np.asarray(p_nodes[:-1], float).copy()
            )

            Mx, Mc = seq_mat_ltv(
                self.A,
                B_list,
            )

            B0 = B_sequence[0]

            return p_seq, Mx, Mc, B0

        raise ValueError(f"Unknown model_mode: {self.model_mode!r}.")

    def _trust_radius_to_stage(self, radius=None):
        """
        Broadcast a trust radius to shape (Np, m).

        Accepts the same forms as trust_region_bounds:
            None, scalar, length m, or length Np*m.
        """
        if radius is None:
            radius = self.trust_radius

        if radius is None:
            return None

        r = np.asarray(radius, float).reshape(-1)
        Np = int(self.Np)
        m = int(self.m)

        if r.size == 1:
            return np.full((Np, m), float(r[0]))

        if r.size == m:
            return np.tile(r.reshape(1, m), (Np, 1))

        if r.size == Np * m:
            return r.reshape(Np, m).copy()

        raise ValueError(
            "trust_radius must be None, scalar, length m, or length Np*m."
        )

    def _stage_trust_to_channel_radius(self, tr_stage):
        """Convert a stage-wise trust radius to one radius per input channel."""
        tr_stage = np.asarray(tr_stage, float).reshape(self.Np, self.m)
        return np.max(tr_stage, axis=0)

    def _broadcast_channel_bounds(self, radius_like, *, default):
        """Broadcast min/max trust bounds to shape (m,)."""
        if radius_like is None:
            return np.full(self.m, default, float)

        arr = np.asarray(radius_like, float).reshape(-1)

        if arr.size == 1:
            return np.full(self.m, float(arr[0]))

        if arr.size == self.m:
            return arr.copy()

        if arr.size == self.Np * self.m:
            return np.max(arr.reshape(self.Np, self.m), axis=0)

        raise ValueError(
            "trust_radius_min/max must be scalar, length m, or length Np*m."
        )

    def _clip_trust_radius(self, radius):
        """Clip trust radius against optional min/max bounds."""
        if radius is None:
            return None

        tr_stage = self._trust_radius_to_stage(radius)
        r_ch = self._stage_trust_to_channel_radius(tr_stage)

        r_min = self._broadcast_channel_bounds(
            self.trust_radius_min,
            default=0.0,
        )
        r_max = self._broadcast_channel_bounds(
            self.trust_radius_max,
            default=np.inf,
        )

        r_ch = np.maximum(r_ch, r_min)
        r_ch = np.minimum(r_ch, r_max)
        r_ch = np.maximum(r_ch, 0.0)

        return r_ch.reshape(-1)

    def _scale_trust_radius(self, scale, *, action):
        """
        Scale the active trust radius without rejecting the current input.

        The scaled radius is used the next time _build_mpc_qp constructs the
        trust-region bounds.
        """
        if self.trust_radius is None:
            self.trust_last_action = "no_radius"
            return None

        scale = np.asarray(scale, float)

        if scale.size == 1:
            new_radius = np.asarray(self.trust_radius, float).reshape(-1) * float(scale)
        else:
            tr_stage = self._trust_radius_to_stage(self.trust_radius)
            scale_ch = scale.reshape(self.m)
            new_radius = self._stage_trust_to_channel_radius(tr_stage) * scale_ch

        self.trust_radius = self._clip_trust_radius(new_radius)
        self.trust_last_action = str(action)
        return self.trust_radius.copy()

    def _rollout_tip_stack_for_U(self, p0, U_candidate):
        """Nonlinear rollout of tip outputs for a candidate control sequence."""
        U_candidate = np.asarray(U_candidate, float).reshape(self.Np, self.m)
        p_seq = self._p_seq_from_U(p0, U_candidate)

        Y = []
        for k in range(self.Np):
            pk = np.asarray(p_seq[k], float).reshape(self.np)

            if hasattr(self, "_eval_forward_tip"):
                yk = self._eval_forward_tip(pk)
            else:
                try:
                    yk = self.forward_tip_fn(pk, commit=False)
                except TypeError:
                    yk = self.forward_tip_fn(pk)

            Y.append(np.asarray(yk, float).reshape(self.n))

        return np.asarray(Y, float).reshape(self.Np, self.n)

    def _linearisation_error_diagnostics(
        self,
        *,
        p0,
        U_candidate,
        X_aff,
        Mc,
    ):
        out = {
            "linerr_available": 0,
            "linerr_first_m": np.nan,
            "linerr_max_m": np.nan,
            "linerr_rms_m": np.nan,
        }

        try:
            U_candidate = np.asarray(
                U_candidate,
                float,
            ).reshape(self.Np, self.m)

            U_vec = U_candidate.reshape(
                self.Np * self.m,
                1,
            )

            X_lin = (
                X_aff + Mc @ U_vec
            ).reshape(self.Np, self.n)

            X_nl = self._rollout_tip_stack_for_U(
                p0,
                U_candidate,
            )

            error_vector = X_nl - X_lin
            error_norm = np.linalg.norm(
                error_vector,
                axis=1,
            )

            out.update(
                {
                    "linerr_available": 1,
                    "linerr_first_m": float(
                        error_norm[0]
                    ),
                    "linerr_max_m": float(
                        np.max(error_norm)
                    ),
                    "linerr_rms_m": float(
                        np.sqrt(
                            np.mean(error_norm**2)
                        )
                    ),

                    # New full-stage diagnostics.
                    "X_lin_candidate": X_lin.copy(),
                    "X_nl_candidate": X_nl.copy(),
                    "linerr_stage_vector_m": (
                        error_vector.copy()
                    ),
                    "linerr_stage_norm_m": (
                        error_norm.copy()
                    ),
                }
            )

        except Exception as exc:
            out["linerr_error"] = str(exc)

        return out

    def _candidate_step_diagnostics(self, U_guess, U_candidate):
        """Diagnostics for how far the candidate moved from the linearisation point."""
        U_guess = np.asarray(U_guess, float).reshape(self.Np, self.m)
        U_candidate = np.asarray(U_candidate, float).reshape(self.Np, self.m)
        dU = U_candidate - U_guess

        out = {
            "step_norm": float(np.linalg.norm(dU)),
            "step_inf": float(np.max(np.abs(dU))),
            "first_step_norm": float(np.linalg.norm(dU[0])),
            "first_step_inf": float(np.max(np.abs(dU[0]))),
            "trust_ratio_inf": np.nan,
            "trust_ratio_first_inf": np.nan,
        }

        tr_stage = self._trust_radius_to_stage(self.trust_radius)
        if tr_stage is not None:
            denom = np.maximum(tr_stage, 1e-12)
            ratio = np.abs(dU) / denom
            out.update({
                "trust_ratio_inf": float(np.max(ratio)),
                "trust_ratio_first_inf": float(np.max(ratio[0])),
            })

        return out

    def _tighten_trust_radius_from_linearisation_probe(self, *, p0, U_guess, dbg):
        """
        Pre-solve adaptive trust-region tightening.

        This probes the nonlinear model around the current linearisation point.
        If a perturbation within the current trust region produces too much
        nonlinear-vs-affine mismatch, the trust radius for that input channel is
        tightened before OSQP is called.  There is no post-solve rejection.
        """
        out = {
            "trust_probe_enabled": int(self.enable_adaptive_trust_region),
            "trust_probe_radius_changed": 0,
            "trust_probe_action": "disabled",
            "trust_probe_max_err_m": np.nan,
            "trust_probe_max_err_mm": np.nan,
            "trust_radius_current": (
                np.nan if self.trust_radius is None else np.asarray(self.trust_radius, float).copy()
            ),
        }

        if not self.enable_adaptive_trust_region or self.trust_radius is None:
            return out

        if "X_aff" not in dbg or "Mc" not in dbg:
            out["trust_probe_action"] = "missing_linear_model"
            return out

        tr_stage = self._trust_radius_to_stage(self.trust_radius)
        radius_ch = self._stage_trust_to_channel_radius(tr_stage)

        if not np.any(np.isfinite(radius_ch) & (radius_ch > 0.0)):
            out["trust_probe_action"] = "zero_or_infinite_radius"
            return out

        probe_stages = min(int(self.trust_probe_stages), self.Np)
        probe_fraction = float(self.trust_probe_fraction)
        tighten_threshold = float(self.trust_linerr_tighten_m)
        relax_threshold = float(self.trust_linerr_relax_m)

        max_err_by_channel = np.zeros(self.m, float)
        num_probes = 0

        U_guess = np.asarray(U_guess, float).reshape(self.Np, self.m)

        for k in range(probe_stages):
            for j in range(self.m):
                base_delta = probe_fraction * float(tr_stage[k, j])

                if not np.isfinite(base_delta) or base_delta <= 0.0:
                    continue

                for sign in (-1.0, 1.0):
                    U_probe = U_guess.copy()
                    raw_target = U_probe[k, j] + sign * base_delta

                    # Keep the probe inside finite symmetric physical input bounds
                    # where those bounds exist.  The actual QP still enforces all
                    # original constraints; this is only to avoid meaningless probes.
                    if np.isfinite(self.u_max[j]):
                        raw_target = float(np.clip(raw_target, -self.u_max[j], self.u_max[j]))

                    actual_delta = raw_target - U_probe[k, j]
                    if abs(actual_delta) <= 1e-12:
                        continue

                    U_probe[k, j] = raw_target

                    lin_diag = self._linearisation_error_diagnostics(
                        p0=p0,
                        U_candidate=U_probe,
                        X_aff=dbg["X_aff"],
                        Mc=dbg["Mc"],
                    )

                    if int(lin_diag.get("linerr_available", 0)) != 1:
                        continue

                    err = float(lin_diag.get("linerr_max_m", np.nan))
                    if np.isfinite(err):
                        max_err_by_channel[j] = max(max_err_by_channel[j], err)
                        num_probes += 1

        max_probe_err = float(np.max(max_err_by_channel)) if num_probes > 0 else np.nan

        scale_ch = np.ones(self.m, float)
        action = "keep"

        for j in range(self.m):
            err = float(max_err_by_channel[j])

            if not np.isfinite(err) or err <= 0.0:
                continue

            if err > tighten_threshold:
                # For a local first-order model, the residual is approximately
                # second order in step size, so sqrt(target/error) is a sensible
                # radius scaling law.  trust_shrink caps how aggressively a
                # single MPC call can shrink the radius.
                scale = np.sqrt(max(tighten_threshold, 1e-12) / err)
                scale = min(1.0, max(float(self.trust_shrink), float(scale)))
                scale_ch[j] = min(scale_ch[j], scale)
                action = "tighten_probe_linerr"
            elif err < relax_threshold:
                scale_ch[j] = max(scale_ch[j], float(self.trust_expand))

        if np.any(np.abs(scale_ch - 1.0) > 1e-12):
            old_radius = self.trust_radius.copy()
            new_radius = self._clip_trust_radius(radius_ch * scale_ch)
            changed = bool(np.max(np.abs(new_radius - old_radius)) > 1e-12)

            if changed:
                self.trust_radius = new_radius
                out["trust_probe_radius_changed"] = 1
                out["trust_probe_action"] = action
                self.trust_last_action = action
            else:
                out["trust_probe_action"] = "clip_no_change"
        else:
            out["trust_probe_action"] = "keep"
            self.trust_last_action = "keep"

        self.trust_last_probe_max_err_m = max_probe_err
        self.trust_last_probe_changed = int(out["trust_probe_radius_changed"])

        out.update({
            "trust_probe_num_probes": int(num_probes),
            "trust_probe_max_err_m": max_probe_err,
            "trust_probe_max_err_mm": float(1000.0 * max_probe_err) if np.isfinite(max_probe_err) else np.nan,
            "trust_probe_max_err_vx_mm": float(1000.0 * max_err_by_channel[0]) if self.m > 0 else np.nan,
            "trust_probe_max_err_vy_mm": float(1000.0 * max_err_by_channel[1]) if self.m > 1 else np.nan,
            "trust_probe_max_err_vz_mm": float(1000.0 * max_err_by_channel[2]) if self.m > 2 else np.nan,
            "trust_probe_max_err_wx_mm": float(1000.0 * max_err_by_channel[3]) if self.m > 3 else np.nan,
            "trust_probe_max_err_wy_mm": float(1000.0 * max_err_by_channel[4]) if self.m > 4 else np.nan,
            "trust_probe_max_err_wz_mm": float(1000.0 * max_err_by_channel[5]) if self.m > 5 else np.nan,
            "trust_probe_max_err_dL_mm": float(1000.0 * max_err_by_channel[6]) if self.m > 6 else np.nan,
            "trust_radius_current": self.trust_radius.copy() if self.trust_radius is not None else np.nan,
            "trust_radius_vx": float(self.trust_radius[0]) if self.trust_radius is not None and self.trust_radius.size > 0 else np.nan,
            "trust_radius_vy": float(self.trust_radius[1]) if self.trust_radius is not None and self.trust_radius.size > 1 else np.nan,
            "trust_radius_vz": float(self.trust_radius[2]) if self.trust_radius is not None and self.trust_radius.size > 2 else np.nan,
            "trust_radius_wx": float(self.trust_radius[3]) if self.trust_radius is not None and self.trust_radius.size > 3 else np.nan,
            "trust_radius_wy": float(self.trust_radius[4]) if self.trust_radius is not None and self.trust_radius.size > 4 else np.nan,
            "trust_radius_wz": float(self.trust_radius[5]) if self.trust_radius is not None and self.trust_radius.size > 5 else np.nan,
            "trust_radius_dL": float(self.trust_radius[6]) if self.trust_radius is not None and self.trust_radius.size > 6 else np.nan,
        })

        return out

    def _update_trust_radius_from_solved_candidate(self, *, lin_diag):
        """
        Update the active trust radius for the next SQP/MPC linearisation.

        This is not an accept/reject test.  The candidate has already been
        accepted by the controller; this only changes future QP bounds.
        """
        if not self.enable_adaptive_trust_region or self.trust_radius is None:
            return "fixed"

        if int(lin_diag.get("linerr_available", 0)) != 1:
            self.trust_last_action = "linerr_unavailable"
            return self.trust_last_action

        linerr_max = float(lin_diag.get("linerr_max_m", np.nan))
        linerr_first = float(lin_diag.get("linerr_first_m", np.nan))

        self.trust_last_linerr_max_m = linerr_max
        self.trust_last_linerr_first_m = linerr_first

        if not np.isfinite(linerr_max):
            self.trust_last_action = "linerr_nan"
            return self.trust_last_action

        if linerr_max > self.trust_linerr_tighten_m:
            self._scale_trust_radius(self.trust_shrink, action="tighten_after_linerr")
            return self.trust_last_action

        if linerr_max < self.trust_linerr_relax_m:
            self._scale_trust_radius(self.trust_expand, action="relax_after_low_linerr")
            return self.trust_last_action

        self.trust_last_action = "keep_after_linerr"
        return self.trust_last_action

    def _nonlinear_tip_tangent_constraint_diagnostics(
        self,
        *,
        p0,
        U_candidate,
    ):
        """
        Evaluate the exact nonlinear proximity-gated tangent constraint.

        A stage is hazardous only when both conditions hold:

            tip clearance <= activation clearance
            tip/lumen tangent angle > configured limit + tolerance

        The QP uses a larger activation threshold through its guard distance,
        while this exact validation uses the true activation clearance.
        """
        out = {
            "tip_tangent_nonlinear_available": 0,
            "tip_tangent_nonlinear_feasible": 1,
            "tip_tangent_nonlinear_max_deg": np.nan,
            "tip_tangent_nonlinear_max_near_wall_deg": np.nan,
            "tip_tangent_nonlinear_min_clearance_m": np.nan,
            "tip_tangent_nonlinear_angles_deg": np.full(
                int(self.Np),
                np.nan,
            ),
            "tip_tangent_nonlinear_clearance_m": np.full(
                int(self.Np),
                np.nan,
            ),
            "tip_tangent_nonlinear_near_wall_mask": np.zeros(
                int(self.Np),
                dtype=bool,
            ),
            "tip_tangent_nonlinear_dangerous_mask": np.zeros(
                int(self.Np),
                dtype=bool,
            ),
            "tip_tangent_nonlinear_lumen_indices": np.full(
                int(self.Np),
                -1,
                dtype=int,
            ),
        }

        if not bool(
            getattr(
                self,
                "enable_hard_tip_tangent_angle",
                False,
            )
        ):
            return out

        if int(self.n) < 6:
            raise ValueError(
                "Nonlinear tip-tangent validation requires n_out >= 6."
            )

        U_candidate = np.asarray(
            U_candidate,
            float,
        ).reshape(self.Np, self.m)

        p_seq = self._p_seq_from_U(
            p0,
            U_candidate,
        )

        angles = np.full(self.Np, np.nan)
        clearances = np.full(self.Np, np.nan)
        lumen_indices = np.full(
            self.Np,
            -1,
            dtype=int,
        )

        for stage in range(self.Np):
            y = self._eval_forward_tip(
                p_seq[stage]
            )
            y = np.asarray(y, float).reshape(self.n)

            tip = y[0:3]
            tangent = y[3:6]

            tangent_hat, _ = self._normalise_vector(
                tangent,
                name=(
                    "nonlinear tip tangent "
                    f"at stage {stage}"
                ),
            )
            lumen_tangent, _ = (
                self._local_lumen_tangent(tip)
            )
            (
                clearance_m,
                lumen_index,
                _,
                _,
            ) = self._tip_wall_clearance(tip)

            cosine = float(
                np.clip(
                    np.dot(
                        lumen_tangent,
                        tangent_hat,
                    ),
                    -1.0,
                    1.0,
                )
            )
            angles[stage] = float(
                np.degrees(
                    np.arccos(cosine)
                )
            )
            clearances[stage] = float(clearance_m)
            lumen_indices[stage] = int(lumen_index)

        limit = float(
            self.tip_tangent_max_angle_deg
        )
        tolerance = float(
            self.tip_tangent_nonlinear_tol_deg
        )
        activation_clearance = float(
            self.tip_tangent_activation_clearance_m
        )

        near_wall = (
            clearances <= activation_clearance
        )
        dangerous = (
            near_wall
            & (angles > limit + tolerance)
        )
        feasible = bool(
            not np.any(dangerous)
        )

        finite_angles = angles[np.isfinite(angles)]
        max_angle = (
            float(np.max(finite_angles))
            if finite_angles.size
            else np.nan
        )
        near_wall_angles = angles[
            near_wall & np.isfinite(angles)
        ]
        max_near_wall_angle = (
            float(np.max(near_wall_angles))
            if near_wall_angles.size
            else np.nan
        )
        finite_clearances = clearances[
            np.isfinite(clearances)
        ]
        min_clearance = (
            float(np.min(finite_clearances))
            if finite_clearances.size
            else np.nan
        )

        out.update(
            {
                "tip_tangent_nonlinear_available": 1,
                "tip_tangent_nonlinear_feasible": int(feasible),
                "tip_tangent_nonlinear_max_deg": max_angle,
                "tip_tangent_nonlinear_max_near_wall_deg": (
                    max_near_wall_angle
                ),
                "tip_tangent_nonlinear_min_clearance_m": (
                    min_clearance
                ),
                "tip_tangent_nonlinear_angles_deg": angles.copy(),
                "tip_tangent_nonlinear_clearance_m": (
                    clearances.copy()
                ),
                "tip_tangent_nonlinear_clearance_mm": (
                    1e3 * clearances
                ),
                "tip_tangent_nonlinear_near_wall_mask": (
                    near_wall.copy()
                ),
                "tip_tangent_nonlinear_dangerous_mask": (
                    dangerous.copy()
                ),
                "tip_tangent_nonlinear_lumen_indices": (
                    lumen_indices.copy()
                ),
                "tip_tangent_nonlinear_limit_deg": limit,
                "tip_tangent_nonlinear_tol_deg": tolerance,
                "tip_tangent_nonlinear_activation_clearance_m": (
                    activation_clearance
                ),
            }
        )

        return out

    def _qp_decision_warm_start(self, U_guess, dbg, n_decision):
        """Build a warm start for either U or [U, progress_slack_fraction]."""
        U_vec = np.asarray(U_guess, float).reshape(self.Np * self.m)
        n_decision = int(n_decision)
        if n_decision == U_vec.size:
            return U_vec

        if (
            int(dbg.get("progress_slack_enabled", 0)) == 1
            and n_decision == U_vec.size + 1
        ):
            sigma0 = float(dbg.get("progress_slack_guess_fraction", 0.0))
            sigma_max = float(getattr(self, "progress_slack_max_fraction", 1.0))
            sigma0 = float(np.clip(sigma0, 0.0, max(0.0, sigma_max)))
            return np.concatenate([U_vec, np.array([sigma0], dtype=float)])

        raise ValueError(
            "Unsupported QP decision dimension: "
            f"controls={U_vec.size}, decision={n_decision}."
        )

    def _progress_solution_diagnostics(self, *, U_candidate, sigma, dbg):
        """Classify one solved soft-progress QP without making global claims.

        The result separates ADVANCE/LIMITED_PROGRESS from REPOSITION/BLOCKED.
        A later receding-horizon persistence counter decides whether repeated
        blocking should be escalated to a controller-level 'impossible within
        configured horizon' suggestion.
        """
        if int(dbg.get("progress_slack_enabled", 0)) != 1:
            return {
                "progress_motion_state": "PROGRESS_SLACK_DISABLED",
                "progress_slack_fraction": np.nan,
                "progress_predicted_terminal_m": float(
                    dbg.get("progress_nominal_terminal_m", np.nan)
                ),
                "progress_predicted_safe_m": np.nan,
                "progress_predicted_safe_fraction": np.nan,
            }

        U_candidate = np.asarray(U_candidate, float).reshape(self.Np, self.m)
        sigma = float(sigma)
        sigma_max = float(getattr(self, "progress_slack_max_fraction", 1.0))
        sigma = float(np.clip(sigma, 0.0, max(0.0, sigma_max)))

        request = float(dbg.get("progress_request_effective_m", 0.0))
        current = float(dbg.get("progress_current_m", np.nan))
        target = float(dbg.get("progress_target_m", np.nan))
        b_progress = float(dbg.get("progress_affine_b_m", np.nan))
        A_progress = np.asarray(
            dbg.get("progress_affine_A", np.empty(0)), float
        ).reshape(-1)

        Nu = self.Np * self.m
        if A_progress.size == Nu and np.isfinite(b_progress):
            terminal = float(b_progress + A_progress @ U_candidate.reshape(Nu))
        else:
            # The compact optimized controller should retain progress_affine_A.
            terminal = float(dbg.get("progress_nominal_terminal_m", np.nan))

        if np.isfinite(current) and np.isfinite(terminal):
            achieved = max(0.0, terminal - current)
        else:
            achieved = np.nan
        if np.isfinite(target) and np.isfinite(terminal):
            shortfall = max(0.0, target - terminal)
        else:
            shortfall = np.nan
        if request > 1.0e-12 and np.isfinite(achieved):
            safe_fraction = float(np.clip(achieved / request, 0.0, 1.0))
        elif request <= 1.0e-12:
            safe_fraction = 1.0
        else:
            safe_fraction = np.nan

        u_scale = np.abs(np.asarray(self.u_max, float).reshape(self.m))
        enabled = u_scale > 1.0e-12
        if np.any(enabled):
            normalized = np.zeros_like(U_candidate)
            normalized[:, enabled] = (
                np.abs(U_candidate[:, enabled])
                / u_scale[enabled].reshape(1, -1)
            )
            control_activity = float(np.max(normalized[:, enabled]))
            first_control_activity = float(np.max(normalized[0, enabled]))
        else:
            control_activity = 0.0
            first_control_activity = 0.0

        epm_cols = np.arange(min(6, self.m), dtype=int)
        epm_enabled = epm_cols[u_scale[epm_cols] > 1.0e-12]
        if epm_enabled.size:
            epm_activity = float(
                np.max(
                    np.abs(U_candidate[:, epm_enabled])
                    / u_scale[epm_enabled].reshape(1, -1)
                )
            )
            epm_first_activity = float(
                np.max(
                    np.abs(U_candidate[0, epm_enabled])
                    / u_scale[epm_enabled]
                )
            )
        else:
            epm_activity = 0.0
            epm_first_activity = 0.0

        if self.m > 6 and u_scale[6] > 1.0e-12:
            insertion_activity = float(
                np.max(np.abs(U_candidate[:, 6]) / u_scale[6])
            )
        else:
            insertion_activity = 0.0

        full_tol = float(
            getattr(self, "progress_full_slack_fraction_tol", 0.05)
        )
        blocked_fraction = float(
            getattr(self, "progress_blocked_slack_fraction", 0.95)
        )
        reposition_activity_threshold = float(
            getattr(self, "progress_reposition_activity_fraction", 0.05)
        )
        min_progress_m = float(
            getattr(
                self,
                "progress_min_meaningful_m",
                max(1.0e-5, 0.05 * max(request, 0.0)),
            )
        )
        angle_active = int(dbg.get("tip_tangent_constraint_num_active", 0))

        if request <= 1.0e-12:
            state = "PATH_COMPLETE_OR_NO_REQUEST"
        elif sigma <= full_tol:
            state = "ADVANCE"
        elif np.isfinite(achieved) and achieved >= min_progress_m:
            state = "LIMITED_PROGRESS"
        elif epm_activity >= reposition_activity_threshold:
            state = "REPOSITION"
        else:
            state = "BLOCKED_CURRENT_HORIZON"

        return {
            "progress_motion_state": state,
            "progress_slack_fraction": sigma,
            "progress_slack_fraction_percent": 100.0 * sigma,
            "progress_predicted_terminal_m": terminal,
            "progress_predicted_safe_m": achieved,
            "progress_predicted_safe_mm": 1.0e3 * achieved if np.isfinite(achieved) else np.nan,
            "progress_predicted_safe_fraction": safe_fraction,
            "progress_solution_shortfall_m": shortfall,
            "progress_solution_shortfall_mm": 1.0e3 * shortfall if np.isfinite(shortfall) else np.nan,
            "progress_control_activity_fraction": control_activity,
            "progress_first_control_activity_fraction": first_control_activity,
            "progress_epm_reposition_activity_fraction": epm_activity,
            "progress_epm_first_activity_fraction": epm_first_activity,
            "progress_insertion_activity_fraction": insertion_activity,
            "progress_angle_constraint_active": int(angle_active > 0),
            "progress_safety_limited_candidate": int(
                angle_active > 0 and sigma >= blocked_fraction
            ),
        }

    def _unpack_qp_decision(self, z_vec, dbg):
        """Return physical U and scalar progress diagnostics from OSQP z."""
        z = np.asarray(z_vec, float).reshape(-1)
        Nu = self.Np * self.m
        if z.size < Nu:
            raise ValueError(f"QP solution has length {z.size}; expected at least {Nu}.")
        U_new = z[:Nu].reshape(self.Np, self.m)

        if int(dbg.get("progress_slack_enabled", 0)) == 1:
            slack_index = int(dbg.get("progress_slack_index", Nu))
            if slack_index < 0 or slack_index >= z.size:
                raise ValueError(
                    f"Invalid progress_slack_index={slack_index} for z size {z.size}."
                )
            sigma = float(z[slack_index])
        else:
            sigma = np.nan

        return U_new, self._progress_solution_diagnostics(
            U_candidate=U_new,
            sigma=sigma,
            dbg=dbg,
        )

    def _update_safe_progress_persistence(self, solve_info):
        """Escalate repeated no-progress predictions across controller frames.

        This is an operational controller status, not a proof of global physical
        infeasibility.  It deliberately distinguishes a feasible REPOSITION
        action from persistent safety-limited blocking.
        """
        info = dict(solve_info)

        # With soft progress enabled, OSQP infeasibility is no longer caused by
        # the requested progress itself: sigma can relax that request.  Therefore
        # expose QP infeasibility separately as a hard/local-constraint failure.
        if int(info.get("qp_infeasible", 0)) == 1:
            info.update(
                {
                    "controller_motion_state": "HARD_CONSTRAINT_INFEASIBLE",
                    "hard_constraint_infeasible": 1,
                    "trajectory_impossible_suggested": 0,
                    "trajectory_impossible_scope": "local_qp_only",
                }
            )
            return info

        if int(info.get("safety_constraint_infeasible", 0)) == 1:
            info.update(
                {
                    "controller_motion_state": "NONLINEAR_SAFETY_VIOLATION",
                    "hard_constraint_infeasible": 0,
                    "trajectory_impossible_suggested": 0,
                    "trajectory_impossible_scope": "candidate_validation",
                }
            )
            return info

        info["hard_constraint_infeasible"] = 0
        state = str(info.get("progress_motion_state", "PROGRESS_SLACK_DISABLED"))
        sigma = float(info.get("progress_slack_fraction", np.nan))
        blocked_fraction = float(
            getattr(self, "progress_blocked_slack_fraction", 0.95)
        )
        patience = int(getattr(self, "progress_impossible_patience_steps", 5))
        patience = max(1, patience)
        angle_active = int(info.get("progress_angle_constraint_active", 0)) == 1

        if not hasattr(self, "_safe_progress_blocked_streak"):
            self._safe_progress_blocked_streak = 0
        if not hasattr(self, "_safe_progress_reposition_streak"):
            self._safe_progress_reposition_streak = 0

        strongly_blocked = (
            np.isfinite(sigma)
            and sigma >= blocked_fraction
            and state in {"REPOSITION", "BLOCKED_CURRENT_HORIZON"}
        )
        if strongly_blocked:
            self._safe_progress_blocked_streak += 1
        else:
            self._safe_progress_blocked_streak = 0

        if state == "REPOSITION" and strongly_blocked:
            self._safe_progress_reposition_streak += 1
        else:
            self._safe_progress_reposition_streak = 0

        impossible_suggested = int(
            strongly_blocked
            and angle_active
            and self._safe_progress_blocked_streak >= patience
        )

        if impossible_suggested:
            operational_state = "IMPOSSIBLE_WITHIN_CONFIGURED_HORIZON"
        elif state == "REPOSITION":
            operational_state = "SAFE_REPOSITIONING"
        elif state == "BLOCKED_CURRENT_HORIZON":
            operational_state = "NO_SAFE_PROGRESS_CURRENT_HORIZON"
        else:
            operational_state = state

        info.update(
            {
                "controller_motion_state": operational_state,
                "progress_blocked_streak": int(self._safe_progress_blocked_streak),
                "progress_reposition_streak": int(self._safe_progress_reposition_streak),
                "trajectory_impossible_suggested": impossible_suggested,
                "trajectory_impossible_scope": (
                    "configured_prediction_horizon_and_local_model"
                    if impossible_suggested
                    else ""
                ),
            }
        )
        return info

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

        U_guess = np.asarray(
            U_init,
            float,
        ).reshape(self.Np, self.m)

        u_max = np.asarray(
            self.u_max,
            float,
        ).reshape(1, self.m)

        # The SQP centre must satisfy the new absolute control bounds.
        U_guess = np.clip(
            U_guess,
            -u_max,
            u_max,
        )

        # Enforce exact zeros for disabled channels.
        inactive = u_max.reshape(-1) <= 1e-12
        U_guess[:, inactive] = 0.0
        sqp_hist = []
        best_U = None
        best_debug = {}
        best_status = "not_solved"

        fallback_U = None
        fallback_debug = {}
        fallback_status = "not_available"
        fallback_linerr = np.inf
        try:
            for it in range(max_iter):
                H, f, A, l, u, dbg = self._build_mpc_qp(
                    p0=p0,
                    x0=x0,
                    U_guess=U_guess,
                    idx_ref=idx_ref,
                )

                # Pre-solve trust-region tightening from measured local
                # nonlinear-vs-affine mismatch.  This changes the QP bounds
                # before OSQP is called; it does not reject an OSQP solution.
                # probe_dbg = self._tighten_trust_radius_from_linearisation_probe(
                #     p0=p0,
                #     U_guess=U_guess,
                #     dbg=dbg,
                # )

                # if int(probe_dbg.get("trust_probe_radius_changed", 0)) == 1:
                #     H, f, A, l, u, dbg = self._build_mpc_qp(
                #         p0=p0,
                #         x0=x0,
                #         U_guess=U_guess,
                #         idx_ref=idx_ref,
                #     )

                # dbg.update(probe_dbg)

                z_warm = self._qp_decision_warm_start(
                    U_guess, dbg, H.shape[0]
                )
                z_vec, _, status = solve_qp_osqp(
                    H,
                    f,
                    A,
                    l,
                    u,
                    U_warm=z_warm,
                )

                status_str = str(status)

                # ----------------------------------------------------
                # QP failure: expose current failed QP diagnostics
                # ----------------------------------------------------
                if status not in ("solved", "solved inaccurate") or z_vec is None:
                    failed_scalar_debug = {
                        f"failed_{k}": v
                        for k, v in dbg.items()
                        if np.isscalar(v) or isinstance(v, str)
                    }

                    failure_info = {
                        "status": status_str,
                        "infeasible": int("infeasible" in status_str.lower()),
                        "sqp_hist": sqp_hist,
                        "failed_sqp_iter": int(it),
                        "used_previous_feasible": False,
                        "last_debug": dbg,

                        # Top-level failed QP diagnostics.
                        **dbg,
                    }

                    if best_U is not None:
                        return best_U, {
                            "status": f"{status_str}; using_previous_feasible",
                            "infeasible": int("infeasible" in status_str.lower()),
                            "sqp_hist": sqp_hist,
                            "failed_sqp_iter": int(it),
                            "used_previous_feasible": True,
                            "last_debug": dbg,

                            # Diagnostics for the last feasible QP.
                            **best_debug,

                            # Diagnostics for the failed QP.
                            **failed_scalar_debug,
                        }

                    return None, failure_info

                # ----------------------------------------------------
                # QP solved
                # ----------------------------------------------------
                U_new, progress_solution_diag = self._unpack_qp_decision(
                    z_vec, dbg
                )
                dbg.update(progress_solution_diag)

                angle_nl_diag = (
                    self._nonlinear_tip_tangent_constraint_diagnostics(
                        p0=p0,
                        U_candidate=U_new,
                    )
                )
                dbg.update(angle_nl_diag)

                if (
                    int(
                        angle_nl_diag.get(
                            "tip_tangent_nonlinear_feasible",
                            1,
                        )
                    )
                    != 1
                ):
                    old_radius = (
                        None
                        if self.trust_radius is None
                        else np.asarray(
                            self.trust_radius,
                            float,
                        ).copy()
                    )

                    radius_changed = False
                    if old_radius is not None:
                        new_radius = self._clip_trust_radius(
                            float(self.trust_shrink)
                            * old_radius
                        )
                        radius_changed = not np.allclose(
                            new_radius,
                            old_radius,
                            rtol=0.0,
                            atol=1e-12,
                        )
                        self.trust_radius = new_radius

                    if radius_changed and it + 1 < max_iter:
                        # Re-linearise at the rejected candidate without
                        # applying it. This is important for the proximity gate:
                        # if the previous nominal trajectory was far from the
                        # wall, its QP angle row was inactive. The rejected
                        # candidate may now be near the wall, so centring the
                        # next SQP iteration at U_new activates the angle
                        # constraint and allows the solver to correct the
                        # dangerous approach.
                        U_guess = U_new.copy()
                        continue

                    return None, {
                        "status": (
                            "nonlinear_tip_tangent_constraint_violation"
                        ),
                        "infeasible": 1,
                        "qp_infeasible": 0,
                        "safety_constraint_infeasible": 1,
                        "failed_sqp_iter": int(it),
                        "sqp_hist": sqp_hist,
                        "last_debug": dbg,
                        **dbg,
                    }

                step_diag = self._candidate_step_diagnostics(
                    U_guess,
                    U_new,
                )

                lin_diag = self._linearisation_error_diagnostics(
                    p0=p0,
                    U_candidate=U_new,
                    X_aff=dbg["X_aff"],
                    Mc=dbg["Mc"],
                )

                dbg.update(step_diag)
                dbg.update(lin_diag)

                linerr_first_m = float(
                    lin_diag.get("linerr_first_m", np.inf)
                )

                linerr_limit_m = 0.30e-3  # 0.1 mm

                candidate_accurate = (
                    np.isfinite(linerr_first_m)
                    and linerr_first_m <= linerr_limit_m
                )

                # --------------------------------------------------------
                # Always retain the best QP-feasible candidate.
                #
                # This prevents model-accuracy rejection from being
                # misreported as QP infeasibility.
                # --------------------------------------------------------
                if linerr_first_m < fallback_linerr:
                    fallback_linerr = linerr_first_m
                    fallback_U = U_new.copy()
                    fallback_debug = dict(dbg)
                    fallback_status = status_str

                if not candidate_accurate:
                    old_radius = np.asarray(
                        self.trust_radius,
                        float,
                    ).copy()

                    new_radius = self._clip_trust_radius(
                        1 * old_radius
                    )

                    radius_changed = not np.allclose(
                        new_radius,
                        old_radius,
                        rtol=0.0,
                        atol=1e-12,
                    )

                    self.trust_radius = new_radius

                    fallback_debug.update(
                        {
                            "candidate_accurate": 0,
                            "model_accuracy_warning": 1,
                            "linerr_limit_m": linerr_limit_m,
                            "linerr_first_m": linerr_first_m,
                            "trust_action_after_solve": (
                                "shrink_after_model_mismatch"
                                if radius_changed
                                else "at_minimum_trust_radius"
                            ),
                            "trust_radius_before": old_radius.copy(),
                            "trust_radius_current": new_radius.copy(),
                        }
                    )

                    print(
                        "[TRUST REGION] "
                        f"linearisation error={1e3 * linerr_first_m:.4f} mm "
                        f"> limit={1e3 * linerr_limit_m:.4f} mm; "
                        "retaining candidate and shrinking radius."
                    )

                    # A smaller trust region only affects a newly built QP.
                    if radius_changed and it + 1 < max_iter:
                        # Do not update U_guess here. This rebuilds the QP
                        # around the same point using a smaller radius.
                        continue

                    # No more refinement is possible. Return the best
                    # QP-feasible candidate instead of returning None.
                    return fallback_U, {
                        "status": (
                            f"{fallback_status}; "
                            "using_best_qp_feasible_candidate_with_model_warning"
                        ),
                        "infeasible": 0,
                        "qp_infeasible": 0,
                        "model_rejected": 0,
                        "model_accuracy_warning": 1,
                        "candidate_accurate": 0,
                        "sqp_hist": sqp_hist,
                        "failed_sqp_iter": -1,
                        "used_fallback_candidate": True,
                        "trust_radius_current": self.trust_radius.copy(),
                        **fallback_debug,
                    }

                # --------------------------------------------------------
                # Candidate satisfies the requested model-accuracy limit.
                # --------------------------------------------------------
                best_U = U_new.copy()
                best_debug = dict(dbg)
                best_debug.update(
                    {
                        "candidate_accurate": 1,
                        "model_accuracy_warning": 0,
                        "linerr_limit_m": linerr_limit_m,
                        "used_fallback_candidate": False,
                    }
                )
                best_status = status_str

                sqp_hist.append(
                    {
                        "it": int(it),
                        "status": status_str,
                        "progress_motion_state": str(
                            dbg.get("progress_motion_state", "")
                        ),
                        "progress_slack_fraction": float(
                            dbg.get("progress_slack_fraction", np.nan)
                        ),
                        "progress_predicted_safe_mm": float(
                            dbg.get("progress_predicted_safe_mm", np.nan)
                        ),
                        # "step_norm": step_norm,
                        # "rel_step_norm": rel_step_norm,
                        "step_inf": float(step_diag.get("step_inf", np.nan)),
                        "first_step_norm": float(step_diag.get("first_step_norm", np.nan)),
                        "first_step_inf": float(step_diag.get("first_step_inf", np.nan)),
                        "trust_ratio_inf": float(step_diag.get("trust_ratio_inf", np.nan)),
                        "trust_ratio_first_inf": float(step_diag.get("trust_ratio_first_inf", np.nan)),
                        "linerr_first_mm": float(lin_diag.get("linerr_first_mm", np.nan)),
                        "linerr_max_mm": float(lin_diag.get("linerr_max_mm", np.nan)),
                        "trust_probe_action": str(dbg.get("trust_probe_action", "")),
                        "trust_probe_radius_changed": int(dbg.get("trust_probe_radius_changed", 0)),
                        "trust_probe_max_err_mm": float(dbg.get("trust_probe_max_err_mm", np.nan)),
                        # "trust_action_after_solve": str(trust_action_after_solve),
                        "cond_H_mpc": float(dbg.get("cond_H_mpc", np.inf)),
                        "lambda_min_H_mpc": float(dbg.get("lambda_min_H_mpc", np.nan)),
                        "lambda_max_H_mpc": float(dbg.get("lambda_max_H_mpc", np.nan)),
                        "num_negative_H_mpc": int(dbg.get("num_negative_H_mpc", -1)),
                        "num_near_zero_H_mpc": int(dbg.get("num_near_zero_H_mpc", -1)),

                        # Useful constraint diagnostics per SQP iteration.
                        "constraint_issue_guess": str(
                            dbg.get("constraint_issue_guess", "")
                        ),
                        "constraint_max_vio_guess": float(
                            dbg.get("constraint_max_vio_guess", np.nan)
                        ),
                        "constraint_num_vio_guess": int(
                            dbg.get("constraint_num_vio_guess", -1)
                        ),
                    }
                )

                u_abs = np.abs(
                    np.asarray(
                        self.u_max,
                        float,
                    ).reshape(self.m)
                )

                # Do not normalize disabled zero-width controls.
                active = u_abs > 1e-12

                if not np.any(active):
                    raise RuntimeError(
                        "No active control channels remain."
                    )

                # The iterate used to build and solve the current QP.
                U_previous = np.asarray(
                    U_guess,
                    float,
                ).reshape(self.Np, self.m).copy()

                U_new = np.asarray(U_new, float).reshape(self.Np, self.m).copy()

                # Remove tiny OSQP residuals in exactly locked channels.
                U_previous[:, ~active] = 0.0
                U_new[:, ~active] = 0.0

                control_scale = u_abs[active].reshape(1, -1)

                U_previous_scaled = (
                    U_previous[:, active] / control_scale
                )

                U_new_scaled = (
                    U_new[:, active] / control_scale
                )

                delta_scaled = (
                    U_new_scaled - U_previous_scaled
                )

                step_scaled_inf = float(
                    np.max(np.abs(delta_scaled))
                )

                iterate_scaled_inf = float(
                    max(
                        np.max(np.abs(U_previous_scaled)),
                        np.max(np.abs(U_new_scaled)),
                    )
                )

                sqp_threshold = float(
                    self.sqp_tol_abs
                    + self.sqp_tol_rel * iterate_scaled_inf
                )

                sqp_converged = (
                    step_scaled_inf <= sqp_threshold
                )

                # print(f"sqp_tol_abs       = {self.sqp_tol_abs}")
                # print(f"sqp_tol_rel       = {self.sqp_tol_rel}")
                # print(f"iterate_scaled_inf = {iterate_scaled_inf}")
                # print(f"sqp_threshold      = {sqp_threshold}")
                # print(f"step_scaled_inf    = {step_scaled_inf}")
                # print(f"sqp_converged      = {sqp_converged}")
                best_U = U_new.copy()
                best_debug = dict(dbg)
                best_status = status_str

                if solver_mode == "sqp_full" and sqp_converged:
                    sqp_stop_reason = "step_tolerance"

                    best_debug.update(
                        {
                            "sqp_converged": 1,
                            "sqp_stop_reason": sqp_stop_reason,
                            "sqp_iterations_used": len(sqp_hist),
                            "sqp_step_scaled_inf": step_scaled_inf,
                            "sqp_step_threshold": sqp_threshold,
                            "sqp_hit_iteration_cap": 0,
                        }
                    )

                    print("SQP converged.")
                    break

                # Only now move the linearisation point.
                U_guess = U_new.copy()

            if best_U is not None:
                return best_U, {
                    "status": best_status,
                    "infeasible": 0,
                    "qp_infeasible": 0,
                    "model_accuracy_warning": 0,
                    "used_fallback_candidate": False,
                    "sqp_hist": sqp_hist,
                    "trust_radius_current": self.trust_radius.copy(),
                    **best_debug,
                }

            if fallback_U is not None:
                return fallback_U, {
                    "status": (
                        f"{fallback_status}; "
                        "using_best_qp_feasible_candidate_with_model_warning"
                    ),
                    "infeasible": 0,
                    "qp_infeasible": 0,
                    "model_accuracy_warning": 1,
                    "used_fallback_candidate": True,
                    "sqp_hist": sqp_hist,
                    "trust_radius_current": self.trust_radius.copy(),
                    **fallback_debug,
                }

            return None, {
                "status": "no_qp_feasible_candidate",
                "infeasible": 0,
                "qp_infeasible": 0,
                "solver_failed": 1,
                "sqp_hist": sqp_hist,
            }

        finally:
            self.model_mode = old_model_mode
    def set_measured_params(self, p_meas):
        """
        Hardware compatibility hook.

        The hardware loop estimates the current magnet pose/insertion from the UR
        pose and camera-derived insertion length, then calls this before solving MPC.

        This updates the controller's internal actuator/magnet state but does not
        treat the forward model as plant truth. The measured tip state is still
        supplied separately through step(x_meas=...).
        """
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before set_measured_params(...).")

        p_meas = np.asarray(p_meas, float).reshape(-1)

        if p_meas.size != self.np:
            raise ValueError(
                f"p_meas must have length {self.np}, got shape {p_meas.shape}."
            )

        if not np.all(np.isfinite(p_meas)):
            raise ValueError(f"p_meas contains non-finite values: {p_meas}")

        self.p = p_meas.copy()

        # Do not overwrite self.x here using the forward model.
        # Hardware truth comes from camera x_meas in step(...).


    def apply_open_loop_control(self, u):
        """
        Hardware compatibility hook for buffered rollout.

        Used only for buffered hardware commands after a previous MPC plan.
        It advances the controller's internal actuator/magnet state by one command.
        """
        if self.p is None:
            raise ValueError("Call set_initial_params(...) before apply_open_loop_control(...).")

        u = np.asarray(u, float).reshape(-1)

        if u.size != self.m:
            raise ValueError(f"u must have length {self.m}, got shape {u.shape}.")

        if not np.all(np.isfinite(u)):
            raise ValueError(f"u contains non-finite values: {u}")

        p_next = self._apply_control_to_p(self.p, u)
        self.p = np.asarray(p_next, float).reshape(self.np,)

        return self.p.copy()
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

        # The reference mixin owns both legacy point references and the
        # continuous contouring reference. In contouring mode it also freezes
        # active_path_reference for the complete SQP solve.
        idx_ref = self._select_reference_indices(x0)
        U_seq, solve_info = self._solve_mpc_sequence(
            p0=p0,
            x0=x0,
            U_init=U_init,
            idx_ref=idx_ref,
            solver_mode=solver_mode,
        )

        solve_info = self._update_safe_progress_persistence(solve_info)

        infeasible = U_seq is None

        if infeasible:
            U_seq = np.zeros((self.Np, self.m), float)
            self.U_warm = None
        else:
            U_seq = np.asarray(U_seq, float).reshape(self.Np, self.m)
        plant_model_runtime = getattr(
            self.forward_tip_fn,
            "model",
            None,
        )

        jacobian_model_runtime = getattr(
            self.Jxy_fn,
            "model",
            None,
        )

        plant_provenance = _contact_provenance(
            plant_model_runtime
        )

        jacobian_provenance = _contact_provenance(
            jacobian_model_runtime
        )

        print("\n[PREDICTION MODEL PROVENANCE]")
        print("  nonlinear plant:", plant_provenance)
        print("  Jacobian model: ", jacobian_provenance)
        progress_stop = bool(
            getattr(self, "stop_on_persistent_no_safe_progress", False)
            and int(solve_info.get("trajectory_impossible_suggested", 0)) == 1
        )
        if progress_stop:
            U_seq = np.zeros((self.Np, self.m), dtype=float)
            solve_info["controller_progress_stop_applied"] = 1
            solve_info["status"] = "persistent_no_safe_progress_stop"
        else:
            solve_info["controller_progress_stop_applied"] = 0

        n_apply = int(np.clip(rollout_steps, 1, self.Np))
        rollout_info = self._apply_control_sequence(U_seq, n_apply)
        plant_adapter = getattr(
            self.forward_tip_fn,
            "adapter",
            None,
        )

        plant_last_info = (
            getattr(plant_adapter, "last_info", None)
            if plant_adapter is not None
            else None
        )

        plant_parts = {}

        if isinstance(plant_last_info, dict):
            plant_parts = (
                plant_last_info.get("parts", {})
                or {}
            )

        plant_gap_min = float(
            plant_parts.get("gap_min", np.nan)
        )

        plant_contact_energy = float(
            plant_parts.get("W_cf", np.nan)
        )

        plant_F_nodes = plant_parts.get(
            "F_nodes",
            None,
        )

        plant_contact_force_norm = (
            np.nan
            if plant_F_nodes is None
            else float(
                np.linalg.norm(
                    np.asarray(
                        plant_F_nodes,
                        float,
                    )
                )
            )
        )

        print("\n[NONLINEAR PLANT CONTACT STATE]")
        print(
            "  contact enabled:",
            plant_provenance["enabled"],
        )
        print(
            "  lumen attached:",
            plant_provenance["lumen_attached"],
        )
        print(
            "  minimum gap:",
            1e3 * plant_gap_min,
            "mm",
        )
        print(
            "  contact energy:",
            plant_contact_energy,
        )
        print(
            "  contact force norm:",
            plant_contact_force_norm,
        )
        # Store the last actually applied input for the next MPC solve.
        # This is needed for the next first-move penalty:
        #     du0 = u0 - u_prev
        u_applied = rollout_info.get("U_applied", None)
        if u_applied is not None:
            u_applied_array = np.asarray(
                u_applied,
                float,
            ).reshape(-1, self.m)

            planned_applied = U_seq[
                :u_applied_array.shape[0]
            ]

            applied_control_error_inf = float(
                np.max(
                    np.abs(
                        u_applied_array
                        - planned_applied
                    )
                )
            )
        else:
            u_applied_array = np.empty(
                (0, self.m)
            )
            applied_control_error_inf = np.nan

        print(
            "[CONTROL MATCH] "
            f"max|applied-planned|="
            f"{applied_control_error_inf:.6e}"
        )

        if (
            np.isfinite(applied_control_error_inf)
            and applied_control_error_inf > 1e-8
        ):
            raise RuntimeError(
                "The nonlinear rollout did not apply the same "
                "control sequence used by the affine prediction."
            )
        if u_applied is not None:
            u_applied = np.asarray(u_applied, float).reshape(-1, self.m)

            if u_applied.shape[0] > 0:
                self.u_prev = u_applied[-1].copy()
            else:
                self.u_prev = U_seq[n_apply - 1].copy()
        else:
            self.u_prev = U_seq[n_apply - 1].copy()

        if not infeasible:
            self._shift_warm_start(U_seq, n_apply)
        X_pred = None
        if not infeasible and "X_aff" in solve_info and "Mc" in solve_info:
            U_vec = U_seq.reshape(-1, 1)
            X_pred = solve_info["X_aff"] + solve_info["Mc"] @ U_vec
            X_pred = X_pred.reshape(self.Np, self.n)
        x_rollout = np.asarray(
            rollout_info.get("x_rollout", []),
            float,
        )

        if X_pred is not None and x_rollout.size:
            X_pred_array = np.asarray(
                X_pred,
                float,
            ).reshape(self.Np, self.n)

            x_rollout_array = x_rollout.reshape(
                -1,
                self.n,
            )

            n_compare = min(
                int(n_apply),
                X_pred_array.shape[0],
                x_rollout_array.shape[0],
            )

            X_pred_applied = X_pred_array[:n_compare]
            X_nonlinear_applied = x_rollout_array[:n_compare]

            error_vector = (
                X_nonlinear_applied
                - X_pred_applied
            )

            error_xy = np.linalg.norm(
                error_vector[:, :2],
                axis=1,
            )

            error_xyz = np.linalg.norm(
                error_vector[:, :3],
                axis=1,
            )

            print("\n[PREDICTION/ROLLOUT COMPARISON]")

            for stage in range(n_compare):
                print(
                    f"  stage={stage} "
                    f"pred_tip={X_pred_applied[stage, :3]} "
                    f"nonlinear_tip={X_nonlinear_applied[stage, :3]} "
                    f"error_xyz={1e3*error_xyz[stage]:.6f} mm"
                )
        else:
            X_pred_applied = None
            X_nonlinear_applied = None
            error_xy = np.empty(0)
            error_xyz = np.empty(0)
        info = {
            "status": solve_info.get(
                "status",
                "unknown",
            ),
            "infeasible": int(infeasible),
            "solver_mode": str(solver_mode),
            "rollout_steps": int(n_apply),

            "u0": U_seq[0].copy(),
            "U_seq": U_seq.copy(),

            "p_now": self.p.copy(),
            "x_now": self.x.copy(),

            "idx_ref": np.asarray(
                idx_ref,
                int,
            ).copy(),

            "X_pred": (
                X_pred.copy()
                if X_pred is not None
                else None
            ),

            "sqp_hist": solve_info.get(
                "sqp_hist",
                [],
            ),

            "trust_radius": (
                None
                if self.trust_radius is None
                else self.trust_radius.copy()
            ),

            "trust_radius_current": solve_info.get(
                "trust_radius_current",
                (
                    None
                    if self.trust_radius is None
                    else self.trust_radius.copy()
                ),
            ),

            "trust_last_action": solve_info.get(
                "trust_last_action",
                self.trust_last_action,
            ),

            **rollout_info,
        }
        safe_progress_keys = (
            "controller_motion_state",
            "progress_motion_state",

            "progress_constraint_mode",

            "progress_current_m",
            "progress_target_m",

            "progress_request_m",
            "progress_request_effective_m",

            "progress_slack_fraction",
            "progress_slack_fraction_percent",

            "progress_predicted_terminal_m",
            "progress_predicted_safe_m",
            "progress_predicted_safe_mm",
            "progress_predicted_safe_fraction",

            "progress_solution_shortfall_m",
            "progress_solution_shortfall_mm",

            "progress_epm_reposition_activity_fraction",
            "progress_epm_first_activity_fraction",
            "progress_insertion_activity_fraction",

            "progress_angle_constraint_active",
            "progress_safety_limited_candidate",

            "progress_blocked_streak",
            "progress_reposition_streak",

            "trajectory_impossible_suggested",
            "trajectory_impossible_scope",

            "hard_constraint_infeasible",
            "controller_progress_stop_applied",

            "qp_infeasible",
            "safety_constraint_infeasible",

            "tip_tangent_constraint_num_active",
            "tip_tangent_angle_nominal_max_deg",
            "tip_tangent_nonlinear_max_near_wall_deg",
        )

        for key in safe_progress_keys:
            if key in solve_info:
                info[key] = solve_info[key]
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

            # Linearisation/trust-region diagnostics.
            "linerr_available",
            "linerr_first_m",
            "linerr_max_m",
            "linerr_rms_m",
            "linerr_first_mm",
            "linerr_max_mm",
            "linerr_rms_mm",
            "step_inf",
            "first_step_norm",
            "first_step_inf",
            "trust_ratio_inf",
            "trust_ratio_first_inf",
            "trust_probe_enabled",
            "trust_probe_radius_changed",
            "trust_probe_action",
            "trust_probe_num_probes",
            "trust_probe_max_err_m",
            "trust_probe_max_err_mm",
            "trust_probe_max_err_vx_mm",
            "trust_probe_max_err_vy_mm",
            "trust_probe_max_err_vz_mm",
            "trust_probe_max_err_wx_mm",
            "trust_probe_max_err_wy_mm",
            "trust_probe_max_err_wz_mm",
            "trust_probe_max_err_dL_mm",
            "trust_action_after_solve",
            "trust_radius_vx",
            "trust_radius_vy",
            "trust_radius_vz",
            "trust_radius_wx",
            "trust_radius_wy",
            "trust_radius_wz",
            "trust_radius_dL",

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
            "X_lin_candidate",
            "X_nl_candidate",
            "linerr_stage_vector_m",
            "linerr_stage_norm_m",
            "B_sequence_linearisation",
            "B_pose_nodes_linearisation",
        ):
    
            if key in solve_info:
                info[key] = solve_info[key]
        if hasattr(self.Jxy_fn, "get_last_diag"):
            info.update(self.Jxy_fn.get_last_diag())
        return self.p.copy(), self.x.copy(), info
