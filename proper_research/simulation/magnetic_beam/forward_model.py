from __future__ import annotations

import copy

import numpy as np

from .config import BeamModelConfig, ContactConfig
from .state import ForwardCache, SolveResult, cache_from_result
from .problem import BaseFrameConfig, BeamSolveProblem
from .kinematics import (
    effective_lengths,
    quat_from_rotvec_ur,
    quat_to_R,
    so3_left_jacobian,
)
from .sensitivity import (
    implicit_tip_jacobian,
    SensitivityOptions,
    ThetaModel,
)
from .contact import LumenQuery
from .magnetism import dipole_from_pose
from .solver import solve_quasistatic_insertion

from beam_direction_magnetisation.magnetism.beam_geometry import (
    make_m_local_fun_wire_tip,
)



class MagneticBeamForwardModel:
    """
    Controller-compatible magnetic-beam forward model.

    Public API:
        tip = model(p7)
        J_tip = model.jacobian_tip_pose7(p7)
        fig, ax = model.plot_solution(p7)

    p7 convention:
        [magnet_x, magnet_y, magnet_z, rx, ry, rz, L_inserted]
    """
    def __init__(
        self,
        *,
        base: BaseFrameConfig,
        beam: BeamModelConfig,
        contact: ContactConfig,
        Kinv_fun,
        m_body: np.ndarray,
        lumen_query: LumenQuery | None,
        m_local_factory=None,
    ):
        beam.validate()
        contact.validate()

        self.base = base
        self.beam = beam
        self.contact_cfg = contact
        self.Kinv_fun = Kinv_fun

        # External source-magnet dipole in its body frame.
        self.m_body = np.asarray(
            m_body,
            float,
        ).reshape(3)

        self.lumen_query = lumen_query

        # None preserves the existing behaviour.
        self.m_local_factory = m_local_factory

        if (
            self.contact_cfg.enabled
            and self.lumen_query is None
        ):
            raise ValueError(
                "Contact is enabled, but lumen_query is None."
            )

        self.cache = ForwardCache()

        self.last_J_tip_pose7 = None
        self.last_J_tip_actuation_tangent = None
        self.last_J_output_pose7 = None
        self.last_J_output_actuation_tangent = None
        self.last_sens_info = None
        self.last_jacobian_diag = {}
        self.last_sensitivity_H = None
        self.last_sensitivity_H_p7: np.ndarray | None = None
        self.last_sensitivity_H_eps: float | None = None
        self.last_sensitivity_contact_mask: np.ndarray | None = None
        self._last_tangent_key: tuple | None = None
        self._last_pose7_key: tuple | None = None
    @property
    def last_info(self):
        return self.cache.info

    @property
    def last_hist(self):
        return self.cache.hist

    @property
    def last_tip(self):
        return self.cache.tip

    @property
    def last_p_centerline(self):
        return self.cache.centerline
    @staticmethod
    def _safe_hessian_diagnostics(H, *, eps: float = 1e-12) -> dict:
        if H is None:
            return {
                "valid": False,
                "cond": np.inf,
                "lambda_min": np.nan,
                "lambda_max": np.nan,
                "num_negative": -1,
                "num_near_zero": -1,
            }

        H = np.asarray(H, float)

        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            return {
                "valid": False,
                "cond": np.inf,
                "lambda_min": np.nan,
                "lambda_max": np.nan,
                "num_negative": -1,
                "num_near_zero": -1,
            }

        Hs = 0.5 * (H + H.T)

        try:
            eigvals = np.linalg.eigvalsh(Hs)
        except np.linalg.LinAlgError:
            return {
                "valid": False,
                "cond": np.inf,
                "lambda_min": np.nan,
                "lambda_max": np.nan,
                "num_negative": -1,
                "num_near_zero": -1,
            }

        abs_eigs = np.abs(eigvals)
        lam_min_abs = float(np.min(abs_eigs))
        lam_max_abs = float(np.max(abs_eigs))

        return {
            "valid": True,
            "cond": float(lam_max_abs / max(lam_min_abs, eps)),
            "lambda_min": float(np.min(eigvals)),
            "lambda_max": float(np.max(eigvals)),
            "num_negative": int(np.sum(eigvals < -eps)),
            "num_near_zero": int(np.sum(abs_eigs < eps)),
        }

    @staticmethod
    def _beam_eigen_direction_diagnostics(
        H,
        *,
        cond_warn: float = 5e3,
        eps: float = 1e-12,
        max_dir_components: int = 20,
    ) -> dict:
        out = {
            "beam_eig_logged": False,
            "beam_eig_failed": False,
            "beam_eig_lambda_min": np.nan,
            "beam_eig_lambda_max": np.nan,
            "beam_eig_cond": np.nan,
            "beam_soft_direction_index": -1,
            "beam_soft_direction_absmax": np.nan,
            "beam_stiff_direction_index": -1,
            "beam_stiff_direction_absmax": np.nan,
        }

        for j in range(max_dir_components):
            out[f"beam_soft_dir_{j}"] = np.nan
            out[f"beam_stiff_dir_{j}"] = np.nan

        if H is None:
            return out

        H = np.asarray(H, float)

        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            out["beam_eig_failed"] = True
            return out

        if not np.all(np.isfinite(H)):
            out["beam_eig_failed"] = True
            return out

        Hs = 0.5 * (H + H.T)

        try:
            eigvals, eigvecs = np.linalg.eigh(Hs)
        except np.linalg.LinAlgError:
            out["beam_eig_failed"] = True
            return out

        abs_eigs = np.abs(eigvals)

        if not np.all(np.isfinite(abs_eigs)):
            out["beam_eig_failed"] = True
            return out

        idx_min = int(np.argmin(abs_eigs))
        idx_max = int(np.argmax(abs_eigs))

        lam_min_abs = float(abs_eigs[idx_min])
        lam_max_abs = float(abs_eigs[idx_max])

        cond = float(lam_max_abs / max(lam_min_abs, eps))

        out["beam_eig_lambda_min"] = float(eigvals[idx_min])
        out["beam_eig_lambda_max"] = float(eigvals[idx_max])
        out["beam_eig_cond"] = cond

        # Only log directions when conditioning is high enough to care.
        if cond < cond_warn:
            return out

        v_soft = np.asarray(eigvecs[:, idx_min], float)
        v_stiff = np.asarray(eigvecs[:, idx_max], float)

        v_soft = v_soft / max(np.linalg.norm(v_soft), eps)
        v_stiff = v_stiff / max(np.linalg.norm(v_stiff), eps)

        soft_idx = int(np.argmax(np.abs(v_soft)))
        stiff_idx = int(np.argmax(np.abs(v_stiff)))

        out["beam_eig_logged"] = True

        out["beam_soft_direction_index"] = soft_idx
        out["beam_soft_direction_absmax"] = float(np.abs(v_soft[soft_idx]))

        out["beam_stiff_direction_index"] = stiff_idx
        out["beam_stiff_direction_absmax"] = float(np.abs(v_stiff[stiff_idx]))

        n_log = min(max_dir_components, v_soft.size)

        for j in range(n_log):
            out[f"beam_soft_dir_{j}"] = float(v_soft[j])
            out[f"beam_stiff_dir_{j}"] = float(v_stiff[j])

        return out
    def _make_m_local_fun(
        self,
        *,
        L_model: float,
        wire_len: float,
        tip_len: float,
    ):
        """
        Build the distributed local beam magnetic moment function.

        The returned function must return magnetic dipole moment per
        unit beam length, with shape (3, N), in A m.
        """
        if self.m_local_factory is None:
            # Existing model behaviour.
            return make_m_local_fun_wire_tip(
                wire_len,
                len_tip=tip_len,
                mode="axial",
                alpha_end=0.0,
                eps=1e-3,
            )

        m_local_fun = self.m_local_factory(
            L_model=float(L_model),
            wire_len=float(wire_len),
            tip_len=float(tip_len),
        )

        if not callable(m_local_fun):
            raise TypeError(
                "m_local_factory must return a callable "
                "m_local_fun(s_mid, parameter)."
            )

        # Early shape and finite-value validation.
        s_test = np.array(
            [0.25, 0.75],
            dtype=float,
        ) * float(L_model)

        m_test = np.asarray(
            m_local_fun(
                s_test,
                float(wire_len),
            ),
            float,
        )

        expected_shape = (3, s_test.size)

        if m_test.shape != expected_shape:
            raise ValueError(
                "m_local_fun returned shape "
                f"{m_test.shape}; expected {expected_shape}."
            )

        if not np.all(np.isfinite(m_test)):
            raise ValueError(
                "m_local_fun returned non-finite values."
            )

        return m_local_fun
    def get_last_jacobian_diag(self) -> dict:
        return dict(getattr(self, "last_jacobian_diag", {}) or {})

    def get_last_diag(self) -> dict:
        return self.get_last_jacobian_diag()
    def _build_last_jacobian_diag_from_sensitivity(self, sens) -> tuple[dict, np.ndarray | None]:
        H = getattr(sens, "H", None)
        info = getattr(sens, "info", {}) or {}

        hdiag = self._safe_hessian_diagnostics(H)

        cond_H = info.get("H_condition", hdiag["cond"])
        stationarity_norm = info.get("stationarity_norm", np.nan)

        diag = {
            "stationarity_norm": float(stationarity_norm),
            "cond_H_beam": float(cond_H),
            "lambda_min_H_beam": hdiag["lambda_min"],
            "lambda_max_H_beam": hdiag["lambda_max"],
            "num_negative_H_beam": hdiag["num_negative"],
            "num_near_zero_H_beam": hdiag["num_near_zero"],
            "valid_H_beam": hdiag["valid"],
        }

        optional = {
            "du_dtheta_norm": getattr(sens, "du_dtheta", None),
            "J_implicit_norm": getattr(sens, "J_implicit", None),
            "J_direct_norm": getattr(sens, "J_direct", None),
            "Gtheta_norm": getattr(sens, "Gtheta", None),
            "P_u_norm": getattr(sens, "P_u", None),
        }

        for key, value in optional.items():
            if value is not None:
                diag[key] = float(np.linalg.norm(value))

        for key in (
            "gradient_evaluations",
            "gradient_time_s",
            "sensitivity_time_s",
            "hessian_reused",
            "difference_scheme",
        ):
            if key in info:
                diag[key] = info[key]

        H_copy = None if H is None else np.asarray(H, float).copy()

        beam_eig_diag = self._beam_eigen_direction_diagnostics(
            H_copy,
            cond_warn=1,
            max_dir_components=20,
        )

        diag.update(beam_eig_diag)
        if H is None:
            diag["H_beam_shape_0"] = -1
            diag["H_beam_shape_1"] = -1
        else:
            H_arr = np.asarray(H)
            diag["H_beam_shape_0"] = int(H_arr.shape[0]) if H_arr.ndim >= 1 else -1
            diag["H_beam_shape_1"] = int(H_arr.shape[1]) if H_arr.ndim >= 2 else -1
        return diag, H_copy
    def __call__(self, p7: np.ndarray) -> np.ndarray:
        result = self.solve(p7, commit=True)
        return result.tip.copy()

    def solve(
        self,
        p7: np.ndarray,
        *,
        commit: bool = True,
        reuse_cache: bool = True,
    ) -> SolveResult:
        p7 = self._validate_p7(p7)
        self._validate_source_magnet_clearance(p7)

        if reuse_cache and self._cache_matches(
            p7,
            atol=self.beam.exact_cache_atol,
        ):
            cached = self.cache_as_solution()
            cached.info["cache_hit"] = True
            cached.info["solve_path"] = "exact_cache"
            return cached

        problem = self.build_problem(p7)

        u0 = self.select_warm_start(problem)

        result = solve_quasistatic_insertion(
            problem,
            u0_flat=u0,
            options=self.beam,
        )
        result.info["cache_hit"] = False

        if commit:
            self._invalidate_jacobian_values()
            self.cache = cache_from_result(
                p7=p7,
                result=result,
                L_model=problem.L_model,
            )

        return result

    def jacobian_output_actuation_tangent(
        self,
        p7: np.ndarray,
        *,
        solve_if_needed: bool = False,
        eps_theta: float = 1e-6,
        eps_hess: float = 1e-4,
        debug_jac: bool = False,
        debug_hessian_terms: bool = False,
        mode: str = "fast",
        reuse_cached: bool = True,
    ) -> np.ndarray:
        """Return d[tip_xyz, tip_tangent]/d[source pose tangent, insertion]."""
        p7 = self._validate_p7(p7)
        if mode not in {"fast", "accurate"}:
            raise ValueError("mode must be 'fast' or 'accurate'.")

        tangent_key = self._jacobian_key(
            p7,
            mode=mode,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
            debug_jac=debug_jac,
            debug_hessian_terms=debug_hessian_terms,
        )
        if (
            reuse_cached
            and self.last_J_output_actuation_tangent is not None
            and tangent_key == self._last_tangent_key
        ):
            return self.last_J_output_actuation_tangent.copy()

        if self.cache.u_flat_opt is None:
            if not solve_if_needed:
                raise RuntimeError(
                    "No cached forward solve. Call model(p7) before requesting "
                    "a Jacobian, or pass solve_if_needed=True."
                )
            self.solve(p7, commit=True)
        self._assert_cache_matches(p7)

        problem = self.build_problem(p7)
        solution = self.cache_as_solution()
        theta_model = self.build_theta_model(p7)
        H_override = None
        if mode == "fast" and self._can_reuse_sensitivity_hessian(
            p7,
            eps_hess=eps_hess,
        ):
            H_override = self.last_sensitivity_H

        sens = implicit_tip_jacobian(
            solution=solution,
            problem=problem,
            theta_model=theta_model,
            options=SensitivityOptions(
                eps_theta=eps_theta,
                eps_hess=eps_hess,
                debug_jac=debug_jac,
                debug_hessian_terms=debug_hessian_terms,
                difference_scheme="forward" if mode == "fast" else "central",
            ),
            H_override=H_override,
        )
        J_output = np.asarray(sens.J_output, dtype=float).reshape(6, 7)
        self.last_J_output_actuation_tangent = J_output.copy()
        self.last_J_tip_actuation_tangent = J_output[:3, :].copy()
        self._last_tangent_key = tangent_key
        self.last_J_output_pose7 = None
        self.last_J_tip_pose7 = None
        self._last_pose7_key = None
        self.last_sens_info = sens.info
        diag, H = self._build_last_jacobian_diag_from_sensitivity(sens)
        diag["jacobian_output_dimension"] = 6
        self.last_jacobian_diag = diag
        self.last_sensitivity_H = H
        if not bool(sens.info.get("hessian_reused", False)):
            self.last_sensitivity_H_p7 = p7.copy()
            self.last_sensitivity_H_eps = float(eps_hess)
            self.last_sensitivity_contact_mask = self._contact_active_mask()
        return J_output.copy()

    def jacobian_tip_actuation_tangent(self, p7: np.ndarray, **kwargs) -> np.ndarray:
        """Backwards-compatible position-only 3x7 Jacobian."""
        return self.jacobian_output_actuation_tangent(p7, **kwargs)[:3, :]

    def jacobian_output_pose7(
        self,
        p7: np.ndarray,
        *,
        solve_if_needed: bool = False,
        eps_theta: float = 1e-6,
        eps_hess: float = 1e-4,
        debug_jac: bool = False,
        debug_hessian_terms: bool = False,
        mode: str = "fast",
        reuse_cached: bool = True,
    ) -> np.ndarray:
        """Differentiate [tip_xyz, tip_tangent] w.r.t. public pose7 entries."""
        p7 = self._validate_p7(p7)
        pose_key = self._jacobian_key(
            p7,
            mode=mode,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
            debug_jac=debug_jac,
            debug_hessian_terms=debug_hessian_terms,
        )
        if (
            reuse_cached
            and self.last_J_output_pose7 is not None
            and pose_key == self._last_pose7_key
        ):
            return self.last_J_output_pose7.copy()

        J_output_tangent = self.jacobian_output_actuation_tangent(
            p7,
            solve_if_needed=solve_if_needed,
            eps_theta=eps_theta,
            eps_hess=eps_hess,
            debug_jac=debug_jac,
            debug_hessian_terms=debug_hessian_terms,
            mode=mode,
            reuse_cached=reuse_cached,
        )
        transform = np.eye(7, dtype=float)
        transform[3:6, 3:6] = so3_left_jacobian(p7[3:6])
        J_output_pose7 = J_output_tangent @ transform
        self.last_J_output_pose7 = J_output_pose7.copy()
        self.last_J_tip_pose7 = J_output_pose7[:3, :].copy()
        self._last_pose7_key = pose_key
        return J_output_pose7.copy()

    def jacobian_tip_pose7(self, p7: np.ndarray, **kwargs) -> np.ndarray:
        """Backwards-compatible position-only 3x7 pose-vector Jacobian."""
        return self.jacobian_output_pose7(p7, **kwargs)[:3, :]

    def plot_solution(
        self,
        p7: np.ndarray | None = None,
        *,
        result: SolveResult | None = None,
        solve_if_needed: bool = True,
        show: bool = True,
        save_path=None,
        **plot_kwargs,
    ):
        """
        Plot the source magnet, solved beam, and configured lumen in 3D.

        If ``result`` is omitted, a matching cached solution is reused. When
        no matching solution exists, ``solve_if_needed=True`` performs and
        caches the forward solve before plotting.
        """
        if p7 is None:
            if self.cache.p7_last is None:
                raise RuntimeError(
                    "p7 is required because the model has no cached solution."
                )
            p7 = self.cache.p7_last.copy()
        p7 = self._validate_p7(p7)

        if result is None:
            cache_matches = (
                self.cache.p7_last is not None
                and self.cache.u_flat_opt is not None
                and np.linalg.norm(p7 - self.cache.p7_last) <= 1e-9
            )
            if cache_matches:
                result = self.cache_as_solution()
            elif solve_if_needed:
                result = self.solve(p7, commit=True)
            else:
                raise RuntimeError(
                    "No matching cached forward solution. Call model.solve(p7) "
                    "first or pass solve_if_needed=True."
                )

        from .visualization import plot_magnetic_beam_scene

        return plot_magnetic_beam_scene(
            p7=p7,
            centerline=result.p,
            lumen_query=self.lumen_query,
            magnet_axis_body=self.m_body,
            show=show,
            save_path=save_path,
            **plot_kwargs,
        )

    def build_problem(self, p7: np.ndarray) -> BeamSolveProblem:
        p7 = self._validate_p7(p7)

        r_src = p7[0:3].copy()
        rvec = p7[3:6].copy()
        L_ins = float(p7[6])

        q_src = quat_from_rotvec_ur(rvec)
        m_src = dipole_from_pose(q_src, self.m_body)

        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=self.beam.L_tip_full,
            L_tip_min=self.beam.L_tip_min,
        )

        m_local_fun = self._make_m_local_fun(
            L_model=L_model,
            wire_len=wire_len,
            tip_len=tip_len,
        )

        return BeamSolveProblem(
            p0=self.base.p0_ur.copy(),
            q0=self.base.q0_ur.copy(),
            L_ins=L_ins,
            L_model=L_model,
            wire_len=wire_len,
            tip_len=tip_len,
            Kinv_fun=self.Kinv_fun,
            u_star=self.base.u_star.copy(),
            r_src=r_src,
            q_src=q_src,
            m_src=m_src,
            m_body=self.m_body.copy(),
            m_local_fun=m_local_fun,

            # Keep this because your m_local_fun already contains
            # the physical magnetic-moment magnitude.
            m_moment=0.0,

            lumen_query=(
                self.lumen_query
                if self.contact_cfg.enabled
                else None
            ),
            contact=(
                self.contact_cfg.params
                if self.contact_cfg.enabled
                else None
            ),
            use_contact=self.contact_cfg.enabled,
            use_contact_in_jacobian=(
                self.contact_cfg.use_in_jacobian
            ),
            N_nodes=self.beam.N_nodes,
            L_tip_full=self.beam.L_tip_full,
            L_tip_min=self.beam.L_tip_min,

            # New.
            m_local_factory=self.m_local_factory,
        )
    def set_lumen(self, lumen_C: np.ndarray, lumen_R: np.ndarray, *, reset_cache: bool = True):
        lumen_C = np.asarray(lumen_C, float)

        if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
            raise ValueError(f"lumen_C must have shape (M, >=3), got {lumen_C.shape}.")

        lumen_C = lumen_C[:, :3].copy()
        lumen_R = np.asarray(lumen_R, float).reshape(-1).copy()

        if lumen_R.size != lumen_C.shape[0]:
            raise ValueError(
                f"lumen_R length {lumen_R.size} does not match lumen_C length "
                f"{lumen_C.shape[0]}."
            )

        if self.contact_cfg.enabled:
            self.lumen_query = LumenQuery(lumen_C, lumen_R)
        else:
            self.lumen_query = None

        if reset_cache:
            self.reset_cache()
    def build_theta_model(self, p7: np.ndarray) -> ThetaModel:
        p7 = self._validate_p7(p7)

        r_src = p7[0:3].copy()
        L_ins = float(p7[6])

        theta0 = np.hstack(
            [
                r_src,
                np.zeros(3, dtype=float),
                L_ins,
            ]
        )

        q_src0 = quat_from_rotvec_ur(p7[3:6])

        return ThetaModel(
            theta0=theta0,
            q_src0=q_src0,
            rotation_convention="world",
        )

    def select_warm_start(self, problem: BeamSolveProblem) -> np.ndarray | None:
        if self.cache.u_flat_opt is None:
            return None

        if self.cache.L_model is None:
            return None

        if abs(problem.L_model - self.cache.L_model) > 0.01:
            return None

        return self.cache.u_flat_opt.copy()

    def cache_as_solution(self) -> SolveResult:
        if self.cache.u_flat_opt is None:
            raise RuntimeError("Cannot build solution from empty cache.")

        if self.cache.tip is None or self.cache.centerline is None:
            raise RuntimeError("Cache is missing tip or centerline.")

        return SolveResult(
            p=self.cache.centerline.copy(),
            q=None if self.cache.q is None else self.cache.q.copy(),
            u_seg=None if self.cache.u_seg is None else self.cache.u_seg.copy(),
            tip=self.cache.tip.copy(),
            u_flat_opt=self.cache.u_flat_opt.copy(),
            info=copy.deepcopy(self.cache.info),
            hist=self.cache.hist,
        )

    def get_cache_copy(self) -> ForwardCache:
        return copy.deepcopy(self.cache)

    def set_cache(self, cache: ForwardCache) -> None:
        self._invalidate_jacobian_values()
        self.cache = copy.deepcopy(cache)

    def reset_cache(self) -> None:
        self.cache = ForwardCache()
        self.last_J_tip_pose7 = None
        self.last_J_tip_actuation_tangent = None
        self.last_J_output_pose7 = None
        self.last_J_output_actuation_tangent = None
        self.last_sens_info = None
        self.last_jacobian_diag = {}
        self.last_sensitivity_H = None
        self.last_sensitivity_H_p7 = None
        self.last_sensitivity_H_eps = None
        self.last_sensitivity_contact_mask = None
        self._last_tangent_key = None
        self._last_pose7_key = None

    def _invalidate_jacobian_values(self) -> None:
        """Invalidate pose-specific Jacobians but retain a reusable Hessian."""
        self.last_J_tip_pose7 = None
        self.last_J_tip_actuation_tangent = None
        self.last_J_output_pose7 = None
        self.last_J_output_actuation_tangent = None
        self.last_sens_info = None
        self.last_jacobian_diag = {}
        self._last_tangent_key = None
        self._last_pose7_key = None

    def _cache_matches(self, p7: np.ndarray, *, atol: float) -> bool:
        return bool(
            self.cache.p7_last is not None
            and self.cache.u_flat_opt is not None
            and self.cache.tip is not None
            and self.cache.centerline is not None
            and bool((self.cache.info or {}).get("success", False))
            and np.allclose(
                np.asarray(p7, float),
                self.cache.p7_last,
                rtol=0.0,
                atol=float(atol),
            )
        )

    def _validate_source_magnet_clearance(self, p7: np.ndarray) -> None:
        """
        Reject an undeformed beam that intersects the finite source cylinder.

        The magnetic field is currently evaluated with a point-dipole model.
        It is singular at the source centre and is not physically meaningful
        inside the actual permanent magnet.  This guard catches invalid test
        poses before L-BFGS-B is attracted into that singularity.
        """
        params = getattr(self, "source_magnet_parameters", None)
        if not params:
            return

        diameter = float(params.get("diameter", 0.0))
        length = float(params.get("length", 0.0))
        if diameter <= 0.0 or length <= 0.0:
            return

        q_src = quat_from_rotvec_ur(p7[3:6])
        source_axis = quat_to_R(q_src) @ (
            self.m_body / max(float(np.linalg.norm(self.m_body)), 1e-12)
        )
        source_axis /= max(float(np.linalg.norm(source_axis)), 1e-12)

        L_model, _, _ = effective_lengths(
            float(p7[6]),
            L_tip_full=self.beam.L_tip_full,
            L_tip_min=self.beam.L_tip_min,
        )
        base_tangent = quat_to_R(self.base.q0_ur) @ np.array(
            [-1.0, 0.0, 0.0]
        )
        s_check = np.linspace(0.0, L_model, max(20, self.beam.N_nodes))
        undeformed = (
            self.base.p0_ur[None, :]
            + s_check[:, None] * base_tangent[None, :]
        )

        rel = undeformed - p7[0:3][None, :]
        axial = rel @ source_axis
        radial_vec = rel - axial[:, None] * source_axis[None, :]
        radial = np.linalg.norm(radial_vec, axis=1)

        radial_excess = radial - 0.5 * diameter
        axial_excess = np.abs(axial) - 0.5 * length
        outside = np.hypot(
            np.maximum(radial_excess, 0.0),
            np.maximum(axial_excess, 0.0),
        )
        inside = np.minimum(
            np.maximum(radial_excess, axial_excess),
            0.0,
        )
        cylinder_signed_distance = outside + inside

        beam_radius = 0.0
        composite = getattr(self, "composite_properties", None)
        if composite and float(composite.get("area", 0.0)) > 0.0:
            beam_radius = np.sqrt(float(composite["area"]) / np.pi)
        elif self.contact_cfg.params is not None:
            beam_radius = float(self.contact_cfg.params.r_beam)

        clearance = cylinder_signed_distance - beam_radius
        minimum = float(np.min(clearance))
        if minimum <= 0.0:
            raise ValueError(
                "The undeformed beam intersects the finite source magnet "
                f"(minimum surface clearance {minimum * 1e3:.3f} mm). "
                "The point-dipole field is invalid inside the 68 mm-class "
                "source and becomes singular near its centre. Move or rotate "
                "the source magnet so the complete beam remains outside it, "
                "or replace the point dipole with a finite-cylinder field and "
                "add source-magnet contact."
            )

    @staticmethod
    def _jacobian_key(
        p7: np.ndarray,
        *,
        mode: str,
        eps_theta: float,
        eps_hess: float,
        debug_jac: bool,
        debug_hessian_terms: bool,
    ) -> tuple:
        return (
            tuple(np.asarray(p7, float).tolist()),
            str(mode),
            float(eps_theta),
            float(eps_hess),
            bool(debug_jac),
            bool(debug_hessian_terms),
        )

    def _can_reuse_sensitivity_hessian(
        self,
        p7: np.ndarray,
        *,
        eps_hess: float,
    ) -> bool:
        """
        Reuse H only within a small neighbourhood of its anchor pose.

        G_theta is still recomputed at every changed pose.  This is a
        quasi-Newton approximation intended for controller-rate updates.
        """
        if self.last_sensitivity_H is None or self.last_sensitivity_H_p7 is None:
            return False
        if self.last_sensitivity_H_eps != float(eps_hess):
            return False
        if self.contact_cfg.enabled:
            current_mask = self._contact_active_mask()
            if (
                current_mask is None
                or self.last_sensitivity_contact_mask is None
                or not np.array_equal(
                    current_mask,
                    self.last_sensitivity_contact_mask,
                )
            ):
                return False

        delta = np.asarray(p7, float) - self.last_sensitivity_H_p7
        return bool(
            np.linalg.norm(delta[0:3]) <= 2.0e-3
            and np.linalg.norm(delta[3:6]) <= 5.0e-2
            and abs(float(delta[6])) <= 2.0e-3
        )

    def _contact_active_mask(self) -> np.ndarray | None:
        if not self.contact_cfg.enabled or not self.cache.info:
            return None
        parts = self.cache.info.get("parts", {})
        gaps = parts.get("gap_nodes")
        if gaps is None:
            return None
        smooth_eps = (
            float(self.contact_cfg.params.smooth_eps)
            if self.contact_cfg.params is not None
            else 0.0
        )
        return np.asarray(gaps, float).reshape(-1) <= 3.0 * smooth_eps

    @staticmethod
    def _validate_p7(p7: np.ndarray) -> np.ndarray:
        p7 = np.asarray(p7, float).reshape(-1)

        if p7.size != 7:
            raise ValueError(
                "Expected p7 = [x, y, z, rx, ry, rz, L], shape (7,), "
                f"got shape {p7.shape}."
            )

        if not np.all(np.isfinite(p7)):
            raise ValueError(f"p7 contains non-finite values: {p7}.")

        if p7[6] <= 0:
            raise ValueError(f"Insertion length must be positive, got L={p7[6]}.")

        return p7

    def _assert_cache_matches(self, p7: np.ndarray) -> None:
        if self.cache.p7_last is None:
            raise RuntimeError("Cache has no p7_last. Call model(p7) first.")

        p7 = self._validate_p7(p7)
        err = float(np.linalg.norm(p7 - self.cache.p7_last))

        if err > 1e-9:
            raise RuntimeError(
                "Cached forward solve does not match requested Jacobian pose. "
                f"||p7 - cached_p7|| = {err:.3e}. "
                "Call model(p7) first, then jacobian_tip_pose7(p7)."
            )