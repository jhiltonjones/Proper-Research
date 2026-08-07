from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .forward_model import MagneticBeamForwardModel as LegacyMagneticBeamForwardModel
from .kinematics import so3_left_jacobian
from .sensitivity_optimized import SensitivityOptions, implicit_tip_jacobian
from .solver_optimized import solve_quasistatic_insertion_optimized
from .state import ForwardCache, SolveResult


class MagneticBeamForwardModelOptimized(LegacyMagneticBeamForwardModel):
    """
    Drop-in optimised forward model.

    The physical model and LTI/LTV/SQP research modes are unchanged. This class
    only reduces repeated work inside each forward/Jacobian evaluation.
    """

    def __init__(
        self,
        *,
        base,
        beam,
        contact,
        Kinv_fun,
        m_body: np.ndarray,
        lumen_query,
        m_local_factory=None,
        result_detail: str = "contact",
        store_history: bool = False,
        store_vectors_in_info: bool = False,
        sensitivity_workers: int = 1,
        copy_cached_results: bool = True,
    ):
        super().__init__(
            base=base,
            beam=beam,
            contact=contact,
            Kinv_fun=Kinv_fun,
            m_body=m_body,
            lumen_query=lumen_query,
            m_local_factory=m_local_factory,
        )
        if result_detail not in {"none", "contact", "full"}:
            raise ValueError("result_detail must be 'none', 'contact', or 'full'.")
        if sensitivity_workers < 1:
            raise ValueError("sensitivity_workers must be >= 1.")

        self.result_detail = result_detail
        self.store_history = bool(store_history)
        self.store_vectors_in_info = bool(store_vectors_in_info)
        self.sensitivity_workers = int(sensitivity_workers)
        self.copy_cached_results = bool(copy_cached_results)
        self._m_local_fun_cache: dict[tuple[float, float, float], Any] = {}

        self.forward_call_count = 0
        self.jacobian_call_count = 0
        self.forward_time_total_s = 0.0
        self.jacobian_time_total_s = 0.0

        # Six-output caches are separate from the legacy position-only caches.
        self.last_J_output_actuation_tangent = None
        self.last_J_output_pose7 = None

    @classmethod
    def from_legacy(
        cls,
        model: LegacyMagneticBeamForwardModel,
        **optimized_options,
    ) -> "MagneticBeamForwardModelOptimized":
        """Clone model configuration/metadata without sharing mutable caches."""
        new = cls(
            base=model.base,
            beam=model.beam,
            contact=model.contact_cfg,
            Kinv_fun=model.Kinv_fun,
            m_body=np.asarray(model.m_body, dtype=float).copy(),
            lumen_query=model.lumen_query,
            m_local_factory=model.m_local_factory,
            **optimized_options,
        )
        # Preserve research metadata used by validation/logging.
        excluded = {
            "cache",
            "last_J_tip_pose7",
            "last_J_tip_actuation_tangent",
            "last_J_output_pose7",
            "last_J_output_actuation_tangent",
            "last_sens_info",
            "last_jacobian_diag",
            "last_sensitivity_H",
            "last_sensitivity_H_p7",
            "last_sensitivity_H_eps",
            "last_sensitivity_contact_mask",
            "_last_tangent_key",
            "_last_pose7_key",
            "_m_local_fun_cache",
        }
        for name, value in vars(model).items():
            if name in excluded or hasattr(new, name):
                continue
            try:
                setattr(new, name, copy.deepcopy(value))
            except Exception:
                setattr(new, name, value)
        return new

    def reset_performance_counters(self) -> None:
        self.forward_call_count = 0
        self.jacobian_call_count = 0
        self.forward_time_total_s = 0.0
        self.jacobian_time_total_s = 0.0

    def performance_counters(self) -> dict[str, float | int]:
        return {
            "forward_calls": int(self.forward_call_count),
            "jacobian_calls": int(self.jacobian_call_count),
            "forward_time_total_s": float(self.forward_time_total_s),
            "jacobian_time_total_s": float(self.jacobian_time_total_s),
        }

    def _make_m_local_fun(
        self,
        *,
        L_model: float,
        wire_len: float,
        tip_len: float,
    ):
        # Lengths are deterministic outputs of effective_lengths. Exact float
        # keys therefore safely eliminate repeated factory construction and
        # validation across solve/Jacobian calls at the same insertion length.
        key = (float(L_model), float(wire_len), float(tip_len))
        cached = self._m_local_fun_cache.get(key)
        if cached is not None:
            return cached
        value = super()._make_m_local_fun(
            L_model=L_model,
            wire_len=wire_len,
            tip_len=tip_len,
        )
        self._m_local_fun_cache[key] = value
        return value

    def set_lumen(self, *args, **kwargs):
        out = super().set_lumen(*args, **kwargs)
        self._m_local_fun_cache.clear()
        return out

    def _cache_as_solution_view(self) -> SolveResult:
        """Read-only-by-contract view used internally by sensitivity code."""
        if self.cache.u_flat_opt is None:
            raise RuntimeError("Cannot build solution view from an empty cache.")
        if self.cache.tip is None or self.cache.centerline is None:
            raise RuntimeError("Cache is missing tip or centerline.")
        return SolveResult(
            p=self.cache.centerline,
            q=self.cache.q,
            u_seg=self.cache.u_seg,
            tip=self.cache.tip,
            u_flat_opt=self.cache.u_flat_opt,
            info=self.cache.info or {},
            hist=self.cache.hist,
        )

    def _cache_as_solution_return(self) -> SolveResult:
        if self.copy_cached_results:
            return self.cache_as_solution()
        return self._cache_as_solution_view()

    @staticmethod
    def _copy_minimal_parts(parts: dict | None) -> dict:
        """Copy only small safety/reuse arrays retained in online detail mode."""
        parts = parts or {}
        out: dict[str, Any] = {}
        for key, value in parts.items():
            if isinstance(value, np.ndarray):
                out[key] = value.copy()
            else:
                out[key] = copy.deepcopy(value)
        return out

    def _commit_result(self, p7: np.ndarray, result: SolveResult, L_model: float) -> None:
        info = dict(result.info or {})
        if "parts" in info:
            info["parts"] = self._copy_minimal_parts(info["parts"])
        self.cache = ForwardCache(
            p7_last=np.asarray(p7, dtype=float).reshape(7).copy(),
            L_model=float(L_model),
            u_flat_opt=result.u_flat_opt.copy(),
            tip=result.tip.copy(),
            centerline=result.p.copy(),
            q=None if result.q is None else result.q.copy(),
            u_seg=None if result.u_seg is None else result.u_seg.copy(),
            info=info,
            hist=result.hist if self.store_history else None,
        )

    def _build_last_jacobian_diag_from_sensitivity(self, sens):
        """Scalar-only diagnostics: no eigendecomposition and no Hessian copy."""
        H = getattr(sens, "H", None)
        info = dict(getattr(sens, "info", {}) or {})
        diag: dict[str, Any] = {
            "stationarity_norm": float(info.get("stationarity_norm", np.nan)),
            "cond_H_beam": float(info.get("H_condition", np.nan)),
            "valid_H_beam": bool(H is not None),
            "H_beam_shape_0": -1,
            "H_beam_shape_1": -1,
        }
        if H is not None:
            H_arr = np.asarray(H, dtype=float)
            if H_arr.ndim == 2:
                diag["H_beam_shape_0"] = int(H_arr.shape[0])
                diag["H_beam_shape_1"] = int(H_arr.shape[1])
        for key in (
            "gradient_evaluations",
            "nominal_gradient_time_s",
            "hessian_time_s",
            "gtheta_time_s",
            "kinematics_sensitivity_time_s",
            "linear_solve_time_s",
            "sensitivity_time_s",
            "hessian_reused",
            "difference_scheme",
            "workers",
            "linear_solve_method",
            "regularization",
        ):
            if key in info:
                diag[key] = info[key]
        return diag, None if H is None else np.asarray(H, dtype=float)

    def solve(
        self,
        p7: np.ndarray,
        *,
        commit: bool = True,
        reuse_cache: bool = True,
    ) -> SolveResult:
        started = time.perf_counter()
        self.forward_call_count += 1
        p7 = self._validate_p7(p7)
        self._validate_source_magnet_clearance(p7)

        if reuse_cache and self._cache_matches(
            p7,
            atol=self.beam.exact_cache_atol,
        ):
            result = self._cache_as_solution_return()
            result.info = dict(result.info or {})
            result.info["cache_hit"] = True
            result.info["solve_path"] = "exact_cache"
            elapsed = time.perf_counter() - started
            result.info["forward_wall_s"] = float(elapsed)
            self.forward_time_total_s += elapsed
            return result

        problem = self.build_problem(p7)
        u0 = self.select_warm_start(problem)
        result = solve_quasistatic_insertion_optimized(
            problem,
            u0_flat=u0,
            options=self.beam,
            result_detail=self.result_detail,
            store_history=self.store_history,
            store_vectors_in_info=self.store_vectors_in_info,
        )
        result.info["cache_hit"] = False
        elapsed = time.perf_counter() - started
        result.info["forward_wall_s"] = float(elapsed)
        self.forward_time_total_s += elapsed

        if commit:
            self._invalidate_jacobian_values()
            self._commit_result(p7, result, problem.L_model)
        return result

    def _invalidate_jacobian_values(self) -> None:
        """Invalidate pose-specific output Jacobians while retaining reusable H."""
        super()._invalidate_jacobian_values()
        self.last_J_output_actuation_tangent = None
        self.last_J_output_pose7 = None

    def reset_cache(self) -> None:
        super().reset_cache()
        self.last_J_output_actuation_tangent = None
        self.last_J_output_pose7 = None

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
        """Return d[tip_xyz, tip_tangent]/d[source pose tangent, insertion].

        This reuses one nominal equilibrium, the same H/Gtheta construction and
        the same implicit solve used by the position Jacobian. The extra three
        rows are obtained from analytic quaternion/kinematic sensitivity.
        """
        started = time.perf_counter()
        self.jacobian_call_count += 1
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
            elapsed = time.perf_counter() - started
            self.jacobian_time_total_s += elapsed
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
        solution = self._cache_as_solution_view()
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
                workers=self.sensitivity_workers,
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
        self.last_sens_info = dict(sens.info or {})

        diag, H = self._build_last_jacobian_diag_from_sensitivity(sens)
        elapsed = time.perf_counter() - started
        diag["jacobian_wall_s"] = float(elapsed)
        diag["jacobian_output_dimension"] = 6
        self.last_jacobian_diag = diag
        self.last_sensitivity_H = H
        if not bool(sens.info.get("hessian_reused", False)):
            self.last_sensitivity_H_p7 = p7.copy()
            self.last_sensitivity_H_eps = float(eps_hess)
            self.last_sensitivity_contact_mask = self._contact_active_mask()

        self.jacobian_time_total_s += elapsed
        return J_output.copy()

    def jacobian_tip_actuation_tangent(
        self,
        p7: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Backwards-compatible 3x7 tip-position Jacobian."""
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
        """Backwards-compatible 3x7 pose-vector tip-position Jacobian."""
        return self.jacobian_output_pose7(p7, **kwargs)[:3, :]

# Optional drop-in alias for model-factory imports.
MagneticBeamForwardModel = MagneticBeamForwardModelOptimized
