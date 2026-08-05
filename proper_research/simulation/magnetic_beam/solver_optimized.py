from __future__ import annotations

from dataclasses import replace
from typing import Any
import time

import numpy as np
from scipy.optimize import minimize

from .config import BeamModelConfig
from .energy_optimized import (
    ResultDetail,
    energy_from_state_optimized,
    energy_from_u_optimized,
)
from .gradients import energy_gradient_u
from .kinematics import effective_lengths, integrate_pq_from_u
from .state import SolveResult

from beam_direction_magnetisation.magnetism.beam_geometry import (
    make_m_local_fun_wire_tip,
)


def precompute_K_segments_optimized(s, Kinv_fun, wire_len) -> np.ndarray:
    """Batched inversion of all 3x3 segment compliance matrices."""
    s = np.asarray(s, dtype=float).reshape(-1)
    smid = 0.5 * (s[:-1] + s[1:])
    Kinv = np.asarray(Kinv_fun(smid, wire_len), dtype=float)

    if Kinv.shape == (3, 3, smid.size):
        Kinv_batch = np.moveaxis(Kinv, -1, 0)
    elif Kinv.shape == (smid.size, 3, 3):
        Kinv_batch = Kinv
    else:
        raise ValueError(
            "Kinv_fun returned shape "
            f"{Kinv.shape}; expected (3, 3, {smid.size}) or "
            f"({smid.size}, 3, 3)."
        )

    try:
        return np.linalg.inv(Kinv_batch)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "At least one segment compliance matrix is singular."
        ) from exc


def make_length_schedule(
    *,
    L_start: float,
    L_final: float,
    dL: float,
) -> list[float]:
    L_start = float(L_start)
    L_final = float(L_final)
    dL = float(dL)
    if L_start <= 0.0 or L_final <= 0.0 or dL <= 0.0:
        raise ValueError("L_start, L_final and dL must be positive.")
    if L_start >= L_final:
        return [L_final]

    n_full = int(np.floor((L_final - L_start) / dL))
    values = [L_start + k * dL for k in range(n_full + 1)]
    if values[-1] < L_final - 1e-12:
        values.append(L_final)
    else:
        values[-1] = L_final
    return [float(v) for v in values]


def initialise_u_flat(
    *,
    N_nodes: int,
    u0_flat: np.ndarray | None,
    rng: np.random.Generator | None = None,
    random_scale: float = 0.0,
) -> np.ndarray:
    expected = 3 * (int(N_nodes) - 1)
    if u0_flat is None:
        if random_scale <= 0.0:
            return np.zeros(expected, dtype=float)
        rng = rng or np.random.default_rng(0)
        return float(random_scale) * rng.standard_normal(expected)

    out = np.asarray(u0_flat, dtype=float).reshape(-1)
    if out.size != expected:
        raise ValueError(f"u0_flat has size {out.size}, expected {expected}.")
    return out.copy()


def directional_gradient_check(
    *,
    obj,
    z: np.ndarray,
    analytic_gradient: np.ndarray,
    eps: float,
    n_random: int = 2,
) -> dict[str, Any]:
    z = np.asarray(z, dtype=float).reshape(-1)
    grad = np.asarray(analytic_gradient, dtype=float).reshape(-1)
    directions: list[np.ndarray] = []

    grad_norm = float(np.linalg.norm(grad))
    if grad_norm > 1e-14:
        directions.append(grad / grad_norm)

    rng = np.random.default_rng(0)
    while len(directions) < max(1, int(n_random) + 1):
        direction = rng.standard_normal(z.size)
        direction /= max(float(np.linalg.norm(direction)), 1e-15)
        directions.append(direction)

    records = []
    for direction in directions:
        fp = float(obj(z + eps * direction))
        fm = float(obj(z - eps * direction))
        fd = (fp - fm) / (2.0 * eps)
        analytic = float(np.dot(grad, direction))
        abs_error = abs(fd - analytic)
        rel_error = abs_error / max(1.0, abs(fd), abs(analytic))
        records.append(
            {
                "finite_difference": fd,
                "analytic": analytic,
                "absolute_error": abs_error,
                "relative_error": rel_error,
            }
        )

    return {
        "eps": float(eps),
        "directions": records,
        "max_relative_error": float(
            max(record["relative_error"] for record in records)
        ),
    }


class _LastValueCache:
    """Exact one-entry cache for repeated SciPy callbacks at the same vector."""

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.value = None

    def get(self, x: np.ndarray):
        if self.x is not None and np.array_equal(self.x, x):
            return self.value
        return None

    def put(self, x: np.ndarray, value) -> None:
        self.x = np.asarray(x, dtype=float).copy()
        self.value = value


def solve_energy_min_3d_optimized(
    problem,
    *,
    options: BeamModelConfig,
    u0_flat: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    result_detail: ResultDetail = "contact",
    store_vectors_in_info: bool = False,
) -> SolveResult:
    """
    Optimised fixed-length solve preserving the legacy objective and gradient.

    Optimisations:
      * batched segment-matrix inversion;
      * reuse of the explicit W(z0) evaluation when L-BFGS-B asks for z0;
      * exact one-entry callback caches;
      * no second final beam integration;
      * selectable diagnostic detail;
      * complete wall-clock accounting.
    """
    solve_started = time.perf_counter()
    options.validate()

    N = int(options.N_nodes)
    s = np.linspace(0.0, float(problem.L_model), N)
    K_seg = precompute_K_segments_optimized(s, problem.Kinv_fun, problem.wire_len)
    u0_flat = initialise_u_flat(
        N_nodes=N,
        u0_flat=u0_flat,
        rng=rng,
        random_scale=0.0,
    )

    u_scale = float(options.u_scale)
    energy_scale = float(options.energy_scale)
    z0 = u0_flat / u_scale

    use_contact = bool(problem.use_contact)
    contact = problem.contact
    if contact is not None:
        contact.validate()
    if use_contact and problem.lumen_query is None:
        raise ValueError("problem.use_contact=True requires problem.lumen_query.")

    objective_requests = 0
    objective_calls = 0
    objective_cache_hits = 0
    gradient_requests = 0
    gradient_calls = 0
    gradient_cache_hits = 0
    objective_time_s = 0.0
    gradient_time_s = 0.0
    obj_cache = _LastValueCache()
    grad_cache = _LastValueCache()

    def obj(z: np.ndarray) -> float:
        nonlocal objective_requests, objective_calls, objective_cache_hits
        nonlocal objective_time_s
        objective_requests += 1
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        cached = obj_cache.get(z_arr)
        if cached is not None:
            objective_cache_hits += 1
            return float(cached)

        started = time.perf_counter()
        u_flat = u_scale * z_arr
        W, _ = energy_from_u_optimized(
            u_flat,
            p0=problem.p0,
            q0=problem.q0,
            s=s,
            K_seg=K_seg,
            u_star=problem.u_star,
            m_src=problem.m_src,
            r_src=problem.r_src,
            m_local_fun=problem.m_local_fun,
            m_moment=problem.m_moment,
            wire_len=problem.wire_len,
            lumen_query=problem.lumen_query,
            use_lumen=use_contact,
            contact=contact,
            detail="none",
        )
        value = float(W / energy_scale)
        objective_calls += 1
        objective_time_s += time.perf_counter() - started
        obj_cache.put(z_arr, value)
        return value

    def grad_obj(z: np.ndarray) -> np.ndarray:
        nonlocal gradient_requests, gradient_calls, gradient_cache_hits
        nonlocal gradient_time_s
        gradient_requests += 1
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        cached = grad_cache.get(z_arr)
        if cached is not None:
            gradient_cache_hits += 1
            return np.asarray(cached, dtype=float).copy()

        started = time.perf_counter()
        u_flat = u_scale * z_arr
        grad_u = energy_gradient_u(
            u_flat,
            p0=problem.p0,
            q0=problem.q0,
            s=s,
            K_seg=K_seg,
            u_star=problem.u_star,
            m_src=problem.m_src,
            r_src=problem.r_src,
            m_local_fun=problem.m_local_fun,
            m_moment=problem.m_moment,
            lumen_query=problem.lumen_query,
            use_magnetic=True,
            use_contact=use_contact,
            contact=contact,
        )
        value = (u_scale / energy_scale) * np.asarray(
            grad_u, dtype=float
        ).reshape(-1)
        gradient_calls += 1
        gradient_time_s += time.perf_counter() - started
        grad_cache.put(z_arr, value.copy())
        return value

    # Preserve the legacy W0 definition but cache it for the optimiser's first
    # objective request at exactly z0.
    W0_scaled = obj(z0)

    optimize_started = time.perf_counter()
    res = minimize(
        obj,
        z0,
        jac=grad_obj,
        method="L-BFGS-B",
        options={
            "maxiter": int(options.maxiter),
            "ftol": float(options.optimizer_ftol),
            "gtol": float(options.optimizer_gtol),
            "maxls": int(options.optimizer_maxls),
            "maxfun": int(options.optimizer_maxfun),
            "maxcor": int(options.optimizer_maxcor),
        },
    )
    optimize_time_s = time.perf_counter() - optimize_started

    if not np.all(np.isfinite(res.x)):
        raise FloatingPointError("Optimizer returned a non-finite solution vector.")

    grad_scaled = np.asarray(res.jac, dtype=float).reshape(-1)
    grad_l2_scaled = float(np.linalg.norm(grad_scaled))
    grad_inf_scaled = float(np.linalg.norm(grad_scaled, ord=np.inf))
    accepted_stationary = bool(
        (not res.success)
        and options.accept_stationary_failure
        and np.isfinite(res.fun)
        and np.all(np.isfinite(grad_scaled))
        and grad_inf_scaled <= options.acceptable_grad_inf_scaled
    )
    solve_success = bool(res.success or accepted_stationary)

    gradient_check = None
    if not solve_success and options.diagnose_failed_gradient:
        try:
            gradient_check = directional_gradient_check(
                obj=obj,
                z=np.asarray(res.x, dtype=float),
                analytic_gradient=grad_scaled,
                eps=float(options.gradient_check_eps),
            )
        except Exception as exc:  # diagnostic path must not hide solve result
            gradient_check = {
                "error": f"{type(exc).__name__}: {exc}",
                "max_relative_error": np.nan,
            }

    u_flat_opt = u_scale * np.asarray(res.x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(u_flat_opt)):
        raise FloatingPointError("Optimized u_flat contains non-finite values.")

    # Integrate the converged state once, then evaluate the final energy from
    # that state instead of invoking energy_from_u and integrating again.
    p, q, u_seg = integrate_pq_from_u(
        u_flat_opt,
        p0=problem.p0,
        q0=problem.q0,
        s=s,
    )
    W, parts = energy_from_state_optimized(
        p=p,
        q=q,
        u_seg=u_seg,
        s=s,
        K_seg=K_seg,
        u_star=problem.u_star,
        m_src=problem.m_src,
        r_src=problem.r_src,
        m_local_fun=problem.m_local_fun,
        m_moment=problem.m_moment,
        wire_len=problem.wire_len,
        lumen_query=problem.lumen_query,
        use_lumen=use_contact,
        contact=contact,
        detail=result_detail,
    )

    W0 = float(W0_scaled * energy_scale)
    bend_norm = np.linalg.norm(u_seg[:, 1:3], axis=1)
    fixed_solve_wall_s = time.perf_counter() - solve_started

    info: dict[str, Any] = {
        "success": solve_success,
        "optimizer_success": bool(res.success),
        "accepted_stationary_failure": accepted_stationary,
        "message": (
            f"{res.message} (accepted: projected gradient "
            f"{grad_inf_scaled:.3e} <= "
            f"{options.acceptable_grad_inf_scaled:.3e})"
            if accepted_stationary
            else str(res.message)
        ),
        "optimizer_status": int(res.status),
        "nit": int(res.nit),
        "W": float(W),
        "W0": W0,
        "dW": float(W - W0),
        "parts": parts,
        "grad_norm_scaled": grad_l2_scaled,
        "grad_inf_norm_scaled": grad_inf_scaled,
        "optimizer_gtol": float(options.optimizer_gtol),
        "acceptable_grad_inf_scaled": float(
            options.acceptable_grad_inf_scaled
        ),
        "gradient_check": gradient_check,
        "objective_requests": int(objective_requests),
        "objective_calls": int(objective_calls),
        "objective_cache_hits": int(objective_cache_hits),
        "gradient_requests": int(gradient_requests),
        "gradient_calls": int(gradient_calls),
        "gradient_cache_hits": int(gradient_cache_hits),
        "objective_time_s": float(objective_time_s),
        "gradient_time_s": float(gradient_time_s),
        "optimizer_time_s": float(optimize_time_s),
        "fixed_solve_wall_s": float(fixed_solve_wall_s),
        "nfev": int(getattr(res, "nfev", objective_requests)),
        "njev": int(getattr(res, "njev", gradient_requests)),
        "max_bend": float(np.max(bend_norm)),
        "mean_bend": float(np.mean(bend_norm)),
        "L_model": float(problem.L_model),
        "wire_len": float(problem.wire_len),
        "tip_len": float(problem.tip_len),
        "result_detail": str(result_detail),
    }
    if store_vectors_in_info:
        info.update(
            {
                "s": s.copy(),
                "u_flat_opt": u_flat_opt.copy(),
                "z_opt": np.asarray(res.x, dtype=float).copy(),
            }
        )

    return SolveResult(
        p=p,
        q=q,
        u_seg=u_seg,
        tip=p[:, -1].copy(),
        u_flat_opt=u_flat_opt,
        info=info,
        hist=None,
    )


def problem_for_continuation_length(problem, *, L_model: float):
    L_eff, wire_len, tip_len = effective_lengths(
        float(L_model),
        L_tip_full=problem.L_tip_full,
        L_tip_min=problem.L_tip_min,
    )
    L_model = float(L_eff)
    factory = getattr(problem, "m_local_factory", None)
    if factory is None:
        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )
    else:
        m_local_fun = factory(
            L_model=L_model,
            wire_len=wire_len,
            tip_len=tip_len,
        )

    return replace(
        problem,
        L_ins=L_model,
        L_model=L_model,
        wire_len=wire_len,
        tip_len=tip_len,
        m_local_fun=m_local_fun,
        m_local_factory=factory,
    )


def _work_metrics(info: dict[str, Any] | None) -> dict[str, float | int]:
    info = info or {}
    return {
        "nit": int(info.get("nit", 0)),
        "objective_calls": int(info.get("objective_calls", 0)),
        "gradient_calls": int(info.get("gradient_calls", 0)),
        "objective_time_s": float(info.get("objective_time_s", 0.0)),
        "gradient_time_s": float(info.get("gradient_time_s", 0.0)),
        "optimizer_time_s": float(info.get("optimizer_time_s", 0.0)),
        "wall_s": float(info.get("fixed_solve_wall_s", 0.0)),
    }


def solve_quasistatic_insertion_optimized(
    problem,
    *,
    u0_flat: np.ndarray | None,
    options: BeamModelConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
    result_detail: ResultDetail = "contact",
    store_history: bool = False,
    store_vectors_in_info: bool = False,
) -> SolveResult:
    """Warm-direct/continuation solve with accurate total-work accounting."""
    total_started = time.perf_counter()
    options.validate()
    has_warm_start = u0_flat is not None
    direct_attempt_info: dict[str, Any] | None = None
    direct_attempt_failed = False

    if has_warm_start and options.direct_warm_start:
        direct_result = solve_energy_min_3d_optimized(
            problem,
            options=options,
            u0_flat=u0_flat,
            rng=rng,
            result_detail=result_detail,
            store_vectors_in_info=store_vectors_in_info,
        )
        direct_attempt_info = dict(direct_result.info)
        direct_result.info.update(
            {
                "solve_path": "warm_direct",
                "hist_n": 1,
                "direct_attempted": True,
                "direct_success": bool(direct_result.success),
                "continuation_steps": 0,
            }
        )
        work = _work_metrics(direct_result.info)
        direct_result.info.update(
            {
                "total_nit": work["nit"],
                "total_objective_calls": work["objective_calls"],
                "total_gradient_calls": work["gradient_calls"],
                "total_optimizer_time_s": work["optimizer_time_s"],
                "work_total_nit": work["nit"],
                "work_total_objective_calls": work["objective_calls"],
                "work_total_gradient_calls": work["gradient_calls"],
                "work_total_objective_time_s": work["objective_time_s"],
                "work_total_gradient_time_s": work["gradient_time_s"],
                "work_total_optimizer_time_s": work["optimizer_time_s"],
                "work_total_wall_s": work["wall_s"],
                "direct_attempt_wall_s": work["wall_s"],
                "continuation_wall_s": 0.0,
                "total_wall_s": float(time.perf_counter() - total_started),
            }
        )
        if store_history:
            direct_result.hist = [
                {
                    "L": float(problem.L_model),
                    "p": direct_result.p.copy(),
                    "q": None if direct_result.q is None else direct_result.q.copy(),
                    "u_seg": (
                        None
                        if direct_result.u_seg is None
                        else direct_result.u_seg.copy()
                    ),
                    "tip": direct_result.tip.copy(),
                    "info": direct_result.info,
                    "len_wire": float(problem.wire_len),
                    "len_tip": float(problem.tip_len),
                }
            ]
        if direct_result.success or not options.fallback_to_continuation:
            return direct_result
        direct_attempt_failed = True

    L_start = max(
        float(options.L_tip_min),
        min(float(options.L0_init), float(problem.L_model)),
    )
    schedule = make_length_schedule(
        L_start=L_start,
        L_final=float(problem.L_model),
        dL=float(options.dL_internal),
    )

    history: list[dict[str, Any]] = []
    step_infos: list[dict[str, Any]] = []
    u_current = (
        None
        if u0_flat is None
        else np.asarray(u0_flat, dtype=float).reshape(-1).copy()
    )
    final_result: SolveResult | None = None

    for step_index, L_k in enumerate(schedule):
        problem_k = problem_for_continuation_length(problem, L_model=L_k)
        is_final = step_index == len(schedule) - 1
        result_k = solve_energy_min_3d_optimized(
            problem_k,
            options=options,
            u0_flat=u_current,
            rng=rng,
            result_detail=result_detail if is_final else "none",
            store_vectors_in_info=store_vectors_in_info and is_final,
        )

        u_prev = u_current
        u_current = result_k.u_flat_opt.copy()
        du_init = np.nan
        if u_prev is not None and u_prev.size == u_current.size:
            du_init = float(np.linalg.norm(u_current - u_prev))

        if verbose:
            parts = result_k.parts
            print(
                "[COSSERAT] "
                f"L={L_k:.6f}, N={options.N_nodes:3d}, "
                f"success={result_k.info['success']}, "
                f"nit={result_k.info['nit']:4d}, "
                f"W0={result_k.info['W0']:.6e}, "
                f"W={result_k.info['W']:.6e}, "
                f"dW={result_k.info['dW']:.6e}, "
                f"W_el={parts.get('W_el', np.nan):.6e}, "
                f"W_m={parts.get('W_m', np.nan):.6e}, "
                f"W_cf={parts.get('W_cf', np.nan):.6e}, "
                f"||u-u_init||={du_init:.6e}"
            )

        step_infos.append(result_k.info)
        if store_history:
            history.append(
                {
                    "L": float(L_k),
                    "p": result_k.p.copy(),
                    "q": None if result_k.q is None else result_k.q.copy(),
                    "u_seg": (
                        None if result_k.u_seg is None else result_k.u_seg.copy()
                    ),
                    "tip": result_k.tip.copy(),
                    "info": result_k.info,
                    "len_wire": float(problem_k.wire_len),
                    "len_tip": float(problem_k.tip_len),
                }
            )
        final_result = result_k

    if final_result is None:
        raise RuntimeError("Continuation schedule produced no solve steps.")

    continuation = {
        key: sum(_work_metrics(info)[key] for info in step_infos)
        for key in (
            "nit",
            "objective_calls",
            "gradient_calls",
            "objective_time_s",
            "gradient_time_s",
            "optimizer_time_s",
            "wall_s",
        )
    }
    direct = _work_metrics(direct_attempt_info)

    final_result.hist = history if store_history else None
    final_result.info.update(
        {
            "hist_n": len(schedule),
            "solve_path": (
                "warm_fallback_continuation"
                if direct_attempt_failed
                else "cold_continuation"
            ),
            "direct_attempted": direct_attempt_info is not None,
            "direct_success": bool(
                (direct_attempt_info or {}).get("success", False)
            ),
            "direct_attempt_nit": direct["nit"],
            "direct_attempt_objective_calls": direct["objective_calls"],
            "direct_attempt_gradient_calls": direct["gradient_calls"],
            "direct_attempt_optimizer_time_s": direct["optimizer_time_s"],
            "direct_attempt_wall_s": direct["wall_s"],
            "continuation_steps": len(schedule),
            "continuation_nit": int(continuation["nit"]),
            "continuation_objective_calls": int(
                continuation["objective_calls"]
            ),
            "continuation_gradient_calls": int(
                continuation["gradient_calls"]
            ),
            "continuation_optimizer_time_s": float(
                continuation["optimizer_time_s"]
            ),
            "continuation_wall_s": float(continuation["wall_s"]),
            # Existing totals retain continuation semantics for compatibility.
            "total_nit": int(continuation["nit"]),
            "total_objective_calls": int(continuation["objective_calls"]),
            "total_gradient_calls": int(continuation["gradient_calls"]),
            "total_optimizer_time_s": float(continuation["optimizer_time_s"]),
            # work_total_* includes the failed direct attempt, when present.
            "work_total_nit": int(direct["nit"] + continuation["nit"]),
            "work_total_objective_calls": int(
                direct["objective_calls"] + continuation["objective_calls"]
            ),
            "work_total_gradient_calls": int(
                direct["gradient_calls"] + continuation["gradient_calls"]
            ),
            "work_total_objective_time_s": float(
                direct["objective_time_s"] + continuation["objective_time_s"]
            ),
            "work_total_gradient_time_s": float(
                direct["gradient_time_s"] + continuation["gradient_time_s"]
            ),
            "work_total_optimizer_time_s": float(
                direct["optimizer_time_s"] + continuation["optimizer_time_s"]
            ),
            "work_total_wall_s": float(
                direct["wall_s"] + continuation["wall_s"]
            ),
            "total_wall_s": float(time.perf_counter() - total_started),
        }
    )
    return final_result
