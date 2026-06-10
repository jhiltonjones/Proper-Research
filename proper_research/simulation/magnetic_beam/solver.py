from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import BeamModelConfig
from .contact import ContactParams
from .state import SolveResult
from .energy import energy_from_u
from .gradients import energy_gradient_u
from .kinematics import effective_lengths, integrate_pq_from_u

from beam_direction_magnetisation.ana_energy import (
    precompute_K_segments,
)
from beam_direction_magnetisation.magnetism.beam_geometry import (
    make_m_local_fun_wire_tip,
)
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

    if contact is not None:
        contact.validate()

    if use_contact and problem.lumen_query is None:
        raise ValueError("problem.use_contact=True requires problem.lumen_query.")

    def obj(z: np.ndarray) -> float:
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
        )

        return float(W / energy_scale)

    def grad_obj(z: np.ndarray) -> np.ndarray:
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

        return (u_scale / energy_scale) * np.asarray(grad_u, float).reshape(-1)

    W0_scaled = obj(z0)

    res = minimize(
        obj,
        z0,
        jac=grad_obj,
        method="L-BFGS-B",
        options={
            "maxiter": int(options.maxiter),
            "ftol": 1e-22,
            "gtol": 1e-14,
            "maxls": 200,
            "maxfun": 100000,
            "maxcor": 50,
        },
    )
    if not np.all(np.isfinite(res.x)):
        raise FloatingPointError("Optimizer returned non-finite solution vector.")
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
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "W": float(W),
        "W0": W0,
        "dW": float(W - W0),
        "parts": parts,
        "s": s.copy(),
        "u_flat_opt": u_flat_opt.copy(),
        "z_opt": np.asarray(res.x, float).copy(),
        "grad_norm_scaled": float(np.linalg.norm(grad_obj(res.x))),
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
    """
    Create a copy of BeamSolveProblem for an intermediate continuation length.

    This regenerates:
      - wire_len
      - tip_len
      - m_local_fun

    The source pose, source dipole, base frame, contact config, and stiffness
    function stay unchanged.
    """
    L_model = float(L_model)

    # In continuation, the loop variable is model length. We reproduce the
    # original behaviour: wire length is based on the current model length.
    L_eff, wire_len, tip_len = effective_lengths(
        L_model,
        L_tip_full=problem.L_tip_full,
        L_tip_min=problem.L_tip_min,
    )

    # Because L_model is already at least L_tip_min in the schedule, L_eff
    # should equal L_model except for pathological inputs.
    L_model = float(L_eff)

    m_local_fun = make_m_local_fun_wire_tip(
        wire_len,
        len_tip=tip_len,
        mode="axial",
        alpha_end=0.0,
        eps=1e-3,
    )

    return replace(
        problem,
        L_ins=L_model,
        L_model=L_model,
        wire_len=wire_len,
        tip_len=tip_len,
        m_local_fun=m_local_fun,
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

    return final_result