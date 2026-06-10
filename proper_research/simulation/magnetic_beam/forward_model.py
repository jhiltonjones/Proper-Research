from __future__ import annotations

import copy

import numpy as np

from .config import BeamModelConfig, ContactConfig
from .state import ForwardCache, SolveResult, cache_from_result
from .problem import BaseFrameConfig, BeamSolveProblem
from .kinematics import effective_lengths, quat_from_rotvec_ur
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
    ):
        beam.validate()
        contact.validate()

        self.base = base
        self.beam = beam
        self.contact_cfg = contact
        self.Kinv_fun = Kinv_fun
        self.m_body = np.asarray(m_body, float).reshape(3)
        self.lumen_query = lumen_query

        if self.contact_cfg.enabled and self.lumen_query is None:
            raise ValueError("Contact is enabled, but lumen_query is None.")

        self.cache = ForwardCache()

        self.last_J_tip_pose7: np.ndarray | None = None
        self.last_sens_info: dict | None = None
        self.last_jacobian_diag = {}
        self.last_sensitivity_H = None
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

    def solve(self, p7: np.ndarray, *, commit: bool = True) -> SolveResult:
        p7 = self._validate_p7(p7)
        problem = self.build_problem(p7)

        u0 = self.select_warm_start(problem)

        result = solve_quasistatic_insertion(
            problem,
            u0_flat=u0,
            options=self.beam,
        )

        if commit:
            self.cache = cache_from_result(
                p7=p7,
                result=result,
                L_model=problem.L_model,
            )

        return result

    def jacobian_tip_pose7(
        self,
        p7: np.ndarray,
        *,
        solve_if_needed: bool = False,
        eps_theta: float = 1e-6,
        eps_hess: float = 1e-4,
        debug_jac: bool = False,
        debug_hessian_terms: bool = False,
    ) -> np.ndarray:
        p7 = self._validate_p7(p7)

        if self.cache.u_flat_opt is None:
            if not solve_if_needed:
                raise RuntimeError(
                    "No cached forward solve. Call model(p7) before requesting "
                    "jacobian_tip_pose7(p7), or pass solve_if_needed=True."
                )
            self.solve(p7, commit=True)

        self._assert_cache_matches(p7)

        problem = self.build_problem(p7)
        solution = self.cache_as_solution()
        theta_model = self.build_theta_model(p7)

        sens = implicit_tip_jacobian(
            solution=solution,
            problem=problem,
            theta_model=theta_model,
            options=SensitivityOptions(
                eps_theta=eps_theta,
                eps_hess=eps_hess,
                debug_jac=debug_jac,
                debug_hessian_terms=debug_hessian_terms,
            ),
        )
        self.last_J_tip_pose7 = sens.J_tip.copy()
        self.last_sens_info = sens.info
        diag, H = self._build_last_jacobian_diag_from_sensitivity(sens)

        self.last_jacobian_diag = diag
        self.last_sensitivity_H = H


        return sens.J_tip.copy()

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

        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
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
            m_moment=0.0,
            lumen_query=self.lumen_query if self.contact_cfg.enabled else None,
            contact=self.contact_cfg.params if self.contact_cfg.enabled else None,
            use_contact=self.contact_cfg.enabled,
            use_contact_in_jacobian=self.contact_cfg.use_in_jacobian,
            N_nodes=self.beam.N_nodes,
            L_tip_full=self.beam.L_tip_full,
            L_tip_min=self.beam.L_tip_min,
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
        self.cache = copy.deepcopy(cache)

    def reset_cache(self) -> None:
        self.cache = ForwardCache()
        self.last_J_tip_pose7 = None
        self.last_sens_info = None

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