from __future__ import annotations

from dataclasses import replace
from typing import Any
import time

import numpy as np
from scipy.optimize import minimize

from .config import BeamModelConfig
from .contact import ContactParams
from .state import SolveResult
from .energy import energy_from_u
from .gradients import energy_gradient_u
from .kinematics import effective_lengths, integrate_pq_from_u


def precompute_K_segments(s, Kinv_fun, wire_len):
    """
    Returns K_seg: (N-1, 3, 3) stiffness per segment midpoint.
    You provide Kinv_fun(s, wire_len) -> (3,3,N).
    """
    s = np.asarray(s, float)
    smid = 0.5*(s[:-1] + s[1:])
    Kinv = Kinv_fun(smid, wire_len)  # (3,3,N-1) if your function respects input
    # ensure shape (3,3,N-1)
    if Kinv.shape[-1] != smid.size:
        # if your Kinv_fun always returns (3,3,N) for node-based s, call with nodes then slice midpoints
        raise ValueError(f"Kinv_fun returned shape {Kinv.shape}, expected last dim {smid.size}")

    K_seg = np.zeros((smid.size, 3, 3), float)
    for i in range(smid.size):
        K_seg[i] = np.linalg.inv(Kinv[:, :, i])
    return K_seg
def make_length_schedule(
    *,
    L_start: float,
    L_final: float,
    dL: float,
) -> list[float]:
    L_start = float(L_start)
    L_final = float(L_final)
    dL = float(dL)

    if L_start <= 0:
        raise ValueError(f"L_start must be positive, got {L_start}.")
    if L_final <= 0:
        raise ValueError(f"L_final must be positive, got {L_final}.")
    if dL <= 0:
        raise ValueError(f"dL must be positive, got {dL}.")

    if L_start >= L_final:
        return [L_final]

    values = [L_start]
    L = L_start

    while L + dL < L_final:
        L = L + dL
        values.append(float(L))

    if abs(values[-1] - L_final) > 1e-12:
        values.append(L_final)

    return values


def initialise_u_flat(
    *,
    N_nodes: int,
    u0_flat: np.ndarray | None,
    rng: np.random.Generator | None = None,
    random_scale: float = 0.0,
) -> np.ndarray:
    N_nodes = int(N_nodes)
    expected = 3 * (N_nodes - 1)

    if u0_flat is None:
        if random_scale <= 0.0:
            return np.zeros(expected, dtype=float)

        if rng is None:
            rng = np.random.default_rng(0)

        return random_scale * rng.standard_normal(expected)

    u0_flat = np.asarray(u0_flat, float).reshape(-1)

    if u0_flat.size != expected:
        raise ValueError(
            f"u0_flat has size {u0_flat.size}, expected {expected} "
            f"for N_nodes={N_nodes}."
        )

    return u0_flat.copy()


def directional_gradient_check(
    *,
    obj,
    z: np.ndarray,
    analytic_gradient: np.ndarray,
    eps: float,
    n_random: int = 2,
) -> dict[str, Any]:
    """
    Compare directional derivatives of the scalar objective and supplied grad.

    This runs only after an optimizer failure.  A large error indicates that
    increasing ``maxiter`` cannot solve the problem because L-BFGS-B's line
    search is seeing an objective/gradient mismatch.
    """
    z = np.asarray(z, float).reshape(-1)
    grad = np.asarray(analytic_gradient, float).reshape(-1)
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


def solve_energy_min_3d(
    problem,
    *,
    options: BeamModelConfig,
    u0_flat: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> SolveResult:
    """
    Solve one fixed-length 3D energy minimisation problem.

    This is the direct replacement for the old solve_energy_min_3d, but it
    consumes BeamSolveProblem instead of many loose arguments.
    """
    options.validate()

    N = int(options.N_nodes)
    s = np.linspace(0.0, float(problem.L_model), N)
    K_seg = precompute_K_segments(s, problem.Kinv_fun, problem.wire_len)

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
    objective_calls = 0
    gradient_calls = 0
    objective_time_s = 0.0
    gradient_time_s = 0.0

    if contact is not None:
        contact.validate()

    if use_contact and problem.lumen_query is None:
        raise ValueError("problem.use_contact=True requires problem.lumen_query.")

    def obj(z: np.ndarray) -> float:
        nonlocal objective_calls, objective_time_s
        started = time.perf_counter()
        u_flat = u_scale * np.asarray(z, float).reshape(-1)

        W, _ = energy_from_u(
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
            return_parts=False,
        )

        objective_calls += 1
        objective_time_s += time.perf_counter() - started
        return float(W / energy_scale)

    def grad_obj(z: np.ndarray) -> np.ndarray:
        nonlocal gradient_calls, gradient_time_s
        started = time.perf_counter()
        u_flat = u_scale * np.asarray(z, float).reshape(-1)

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

        gradient_calls += 1
        gradient_time_s += time.perf_counter() - started
        return (u_scale / energy_scale) * np.asarray(grad_u, float).reshape(-1)

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
        raise FloatingPointError("Optimizer returned non-finite solution vector.")
    grad_scaled = np.asarray(res.jac, float).reshape(-1)
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
                z=np.asarray(res.x, float),
                analytic_gradient=grad_scaled,
                eps=float(options.gradient_check_eps),
            )
        except Exception as exc:
            gradient_check = {
                "error": f"{type(exc).__name__}: {exc}",
                "max_relative_error": np.nan,
            }

    u_flat_opt = u_scale * np.asarray(res.x, float).reshape(-1)
    if not np.all(np.isfinite(u_flat_opt)):
        raise FloatingPointError("Optimized u_flat contains non-finite values.")
    p, q, u_seg = integrate_pq_from_u(
        u_flat_opt,
        p0=problem.p0,
        q0=problem.q0,
        s=s,
    )

    W, parts = energy_from_u(
        u_flat_opt,
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
    )

    W0 = float(W0_scaled * energy_scale)

    bend_norm = np.linalg.norm(u_seg[:, 1:3], axis=1)

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
        "s": s.copy(),
        "u_flat_opt": u_flat_opt.copy(),
        "z_opt": np.asarray(res.x, float).copy(),
        "grad_norm_scaled": grad_l2_scaled,
        "grad_inf_norm_scaled": grad_inf_scaled,
        "optimizer_gtol": float(options.optimizer_gtol),
        "acceptable_grad_inf_scaled": float(
            options.acceptable_grad_inf_scaled
        ),
        "gradient_check": gradient_check,
        "objective_calls": int(objective_calls),
        "gradient_calls": int(gradient_calls),
        "objective_time_s": float(objective_time_s),
        "gradient_time_s": float(gradient_time_s),
        "optimizer_time_s": float(optimize_time_s),
        "nfev": int(getattr(res, "nfev", objective_calls)),
        "njev": int(getattr(res, "njev", gradient_calls)),
        "max_bend": float(np.max(bend_norm)),
        "mean_bend": float(np.mean(bend_norm)),
        "L_model": float(problem.L_model),
        "wire_len": float(problem.wire_len),
        "tip_len": float(problem.tip_len),
    }

    return SolveResult(
        p=p,
        q=q,
        u_seg=u_seg,
        tip=p[:, -1].copy(),
        u_flat_opt=u_flat_opt,
        info=info,
        hist=None,
    )


def problem_for_continuation_length(
    problem,
    *,
    L_model: float,
):
    L_model = float(L_model)

    L_eff, wire_len, tip_len = effective_lengths(
        L_model,
        L_tip_full=problem.L_tip_full,
        L_tip_min=problem.L_tip_min,
    )

    L_model = float(L_eff)

    factory = getattr(
        problem,
        "m_local_factory",
        None,
    )

    if factory is None:
        # Preserve legacy behaviour.
        m_local_fun = make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )
    else:
        # Rebuild the same physical profile for the
        # current continuation length.
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

        # This is normally already preserved by replace(),
        # but making it explicit is useful.
        m_local_factory=factory,
    )


def solve_quasistatic_insertion(
    problem,
    *,
    u0_flat: np.ndarray | None,
    options: BeamModelConfig,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> SolveResult:
    """
    Continuation solve in model length.

    This replaces the old solve_quasistatic_insertion. It returns only the final
    SolveResult, with continuation history stored in result.hist.

    The previous version returned a list of dicts. If other code still expects
    hist, use result.hist.
    """
    options.validate()
    has_warm_start = u0_flat is not None

    # Controller updates are normally small.  Solving the final problem
    # directly from the last converged strain field avoids replaying every
    # insertion length at every control frame.
    direct_attempt_failed = False
    if has_warm_start and options.direct_warm_start:
        direct_result = solve_energy_min_3d(
            problem,
            options=options,
            u0_flat=u0_flat,
            rng=rng,
        )
        direct_result.info["solve_path"] = "warm_direct"
        direct_result.info["hist_n"] = 1
        direct_result.info["total_nit"] = int(direct_result.info["nit"])
        direct_result.info["total_objective_calls"] = int(
            direct_result.info["objective_calls"]
        )
        direct_result.info["total_gradient_calls"] = int(
            direct_result.info["gradient_calls"]
        )
        direct_result.info["total_optimizer_time_s"] = float(
            direct_result.info["optimizer_time_s"]
        )
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

    hist: list[dict[str, Any]] = []
    u_current = None if u0_flat is None else np.asarray(u0_flat, float).reshape(-1).copy()

    final_result: SolveResult | None = None

    for L_k in schedule:
        problem_k = problem_for_continuation_length(
            problem,
            L_model=L_k,
        )

        result_k = solve_energy_min_3d(
            problem_k,
            options=options,
            u0_flat=u_current,
            rng=rng,
        )

        u_prev = u_current
        u_current = result_k.u_flat_opt.copy()

        du_init = np.nan
        if u_prev is not None and u_prev.size == u_current.size:
            du_init = float(np.linalg.norm(u_current - u_prev))

        parts = result_k.parts

        if verbose:
            print(
                "[COSSERAT] "
                f"L={L_k:.6f}, "
                f"N={options.N_nodes:3d}, "
                f"success={result_k.info['success']}, "
                f"nit={result_k.info['nit']:4d}, "
                f"W0={result_k.info['W0']:.6e}, "
                f"W={result_k.info['W']:.6e}, "
                f"dW={result_k.info['dW']:.6e}, "
                f"W_el={parts.get('W_el', np.nan):.6e}, "
                f"W_m={parts.get('W_m', np.nan):.6e}, "
                f"W_cf={parts.get('W_cf', np.nan):.6e}, "
                f"||u||={np.linalg.norm(u_current):.6e}, "
                f"||u-u_init||={du_init:.6e}"
            )

        hist.append(
            {
                "L": float(L_k),
                "p": result_k.p.copy(),
                "q": None if result_k.q is None else result_k.q.copy(),
                "u_seg": None if result_k.u_seg is None else result_k.u_seg.copy(),
                "tip": result_k.tip.copy(),
                "info": result_k.info,
                "len_wire": float(problem_k.wire_len),
                "len_tip": float(problem_k.tip_len),
            }
        )

        final_result = result_k

    if final_result is None:
        raise RuntimeError("Continuation schedule produced no solve steps.")

    final_result.hist = hist
    final_result.info["hist_n"] = len(hist)
    final_result.info["solve_path"] = (
        "warm_fallback_continuation"
        if direct_attempt_failed
        else "cold_continuation"
    )
    step_infos = [entry["info"] for entry in hist]
    final_result.info["total_nit"] = int(
        sum(int(info.get("nit", 0)) for info in step_infos)
    )
    final_result.info["total_objective_calls"] = int(
        sum(int(info.get("objective_calls", 0)) for info in step_infos)
    )
    final_result.info["total_gradient_calls"] = int(
        sum(int(info.get("gradient_calls", 0)) for info in step_infos)
    )
    final_result.info["total_optimizer_time_s"] = float(
        sum(float(info.get("optimizer_time_s", 0.0)) for info in step_infos)
    )

    return final_result