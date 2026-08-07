from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import os
import time
from typing import Callable, Iterable, Literal

import numpy as np
from scipy.linalg import LinAlgError, solve

from .gradients import energy_gradient_u
from .kinematics import (
    _normalization_jacobian,
    _quat_exp_body_jacobian_u,
    _quat_left_matrix,
    _quat_right_matrix,
    _rotated_vector_jacobian_q,
    effective_lengths,
    integrate_pq_and_sens_from_u,
    integrate_pq_from_u,
    perturb_quat_world,
    quat_exp_body,
    quat_mul,
    quat_normalize,
    quat_to_R,
)
from .magnetism import dipole_from_pose
from .solver_optimized import precompute_K_segments_optimized

from beam_direction_magnetisation.magnetism.beam_geometry import (
    make_m_local_fun_wire_tip,
)


DifferenceScheme = Literal["forward", "central"]


@dataclass(frozen=True)
class SensitivityOptions:
    """Options compatible with the legacy sensitivity interface."""

    eps_theta: float = 1e-6
    eps_hess: float = 1e-4
    debug_jac: bool = False
    debug_hessian_terms: bool = False
    difference_scheme: DifferenceScheme = "forward"

    # Set MAGBEAM_SENSITIVITY_WORKERS on Iridis, or pass workers explicitly
    # when constructing this dataclass. Keep 1 for deterministic baseline runs.
    workers: int = 0
    symmetrise_hessian: bool = True
    regularization_relative: float = 1e-10
    regularization_attempts: int = 7

    def validate(self) -> None:
        if self.eps_theta <= 0.0:
            raise ValueError("eps_theta must be positive.")
        if self.eps_hess <= 0.0:
            raise ValueError("eps_hess must be positive.")
        if self.difference_scheme not in {"forward", "central"}:
            raise ValueError("difference_scheme must be 'forward' or 'central'.")
        if self.workers < 0:
            raise ValueError("workers must be >= 0.")
        if self.regularization_relative < 0.0:
            raise ValueError("regularization_relative must be non-negative.")
        if self.regularization_attempts < 1:
            raise ValueError("regularization_attempts must be >= 1.")


@dataclass(frozen=True)
class ThetaModel:
    """
    Controller parameterisation used by the implicit sensitivity calculation.

    theta = [source_x, source_y, source_z,
             source_rotation_increment_x/y/z,
             insertion_length]
    """

    theta0: np.ndarray
    q_src0: np.ndarray
    rotation_convention: str = "world"

    def __post_init__(self) -> None:
        theta0 = np.asarray(self.theta0, dtype=float).reshape(7)
        q_src0 = quat_normalize(np.asarray(self.q_src0, dtype=float).reshape(4))
        if self.rotation_convention not in {"world", "body"}:
            raise ValueError("rotation_convention must be 'world' or 'body'.")
        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "q_src0", q_src0)


@dataclass
class SensitivityResult:
    # Backwards-compatible tip-position output.
    J_tip: np.ndarray
    du_dtheta: np.ndarray
    J_implicit: np.ndarray
    J_direct: np.ndarray
    Gtheta: np.ndarray
    P_u: np.ndarray
    H: np.ndarray
    info: dict

    # Optional six-output extension [tip_xyz, tip_tangent]. These fields reuse
    # the same equilibrium, H, Gtheta and implicit solve as J_tip.
    J_tangent: np.ndarray
    J_output: np.ndarray
    J_tangent_implicit: np.ndarray
    J_tangent_direct: np.ndarray
    T_u: np.ndarray
    tip_tangent_base: np.ndarray


@dataclass(frozen=True)
class _PreparedProblem:
    problem: object
    s: np.ndarray
    K_seg: np.ndarray


def _workers(options: SensitivityOptions) -> int:
    if options.workers > 0:
        return int(options.workers)
    raw = os.environ.get("MAGBEAM_SENSITIVITY_WORKERS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _map_ordered(
    fn: Callable[[int], np.ndarray],
    indices: Iterable[int],
    *,
    workers: int,
) -> list[np.ndarray]:
    idx = list(indices)
    if workers <= 1 or len(idx) <= 1:
        return [fn(i) for i in idx]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, idx))


def _rotation_from_theta(theta_model, drot: np.ndarray) -> np.ndarray:
    convention = str(getattr(theta_model, "rotation_convention", "world"))
    q0 = quat_normalize(np.asarray(theta_model.q_src0, dtype=float).reshape(4))
    drot = np.asarray(drot, dtype=float).reshape(3)
    if convention == "world":
        return perturb_quat_world(q0, drot)
    if convention == "body":
        # exp_body expects a rate and a step; using ds=1 applies drot directly.
        return quat_normalize(quat_mul(q0, quat_exp_body(drot, 1.0)))
    raise ValueError(f"Unsupported rotation convention {convention!r}.")


def _m_local_for_lengths(problem, L_model: float, wire_len: float, tip_len: float):
    factory = getattr(problem, "m_local_factory", None)
    if factory is None:
        return make_m_local_fun_wire_tip(
            wire_len,
            len_tip=tip_len,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )
    return factory(
        L_model=float(L_model),
        wire_len=float(wire_len),
        tip_len=float(tip_len),
    )


def _problem_at_theta(problem, theta_model, theta: np.ndarray):
    theta = np.asarray(theta, dtype=float).reshape(7)
    theta0 = np.asarray(theta_model.theta0, dtype=float).reshape(7)
    L_ins = float(theta[6])
    if L_ins <= 0.0:
        raise ValueError("Perturbed insertion length must remain positive.")

    # Most G_theta columns change only source pose. Reuse immutable nominal
    # quantities whenever the corresponding parameter is unchanged.
    if np.array_equal(theta[3:6], theta0[3:6]):
        q_src = np.asarray(problem.q_src, dtype=float)
        m_src = np.asarray(problem.m_src, dtype=float)
    else:
        q_src = _rotation_from_theta(theta_model, theta[3:6])
        m_src = dipole_from_pose(q_src, np.asarray(problem.m_body, dtype=float))

    r_src = (
        np.asarray(problem.r_src, dtype=float)
        if np.array_equal(theta[0:3], theta0[0:3])
        else theta[0:3].copy()
    )

    if float(theta[6]) == float(theta0[6]):
        L_model = float(problem.L_model)
        wire_len = float(problem.wire_len)
        tip_len = float(problem.tip_len)
        m_local_fun = problem.m_local_fun
    else:
        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=problem.L_tip_full,
            L_tip_min=problem.L_tip_min,
        )
        m_local_fun = _m_local_for_lengths(problem, L_model, wire_len, tip_len)

    return replace(
        problem,
        L_ins=L_ins,
        L_model=L_model,
        wire_len=wire_len,
        tip_len=tip_len,
        r_src=r_src,
        q_src=q_src,
        m_src=m_src,
        m_local_fun=m_local_fun,
    )


def _prepare(problem) -> _PreparedProblem:
    s = np.linspace(0.0, float(problem.L_model), int(problem.N_nodes))
    K_seg = precompute_K_segments_optimized(
        s,
        problem.Kinv_fun,
        problem.wire_len,
    )
    return _PreparedProblem(problem=problem, s=s, K_seg=K_seg)


def _gradient(u_flat: np.ndarray, prepared: _PreparedProblem) -> np.ndarray:
    p = prepared.problem
    return np.asarray(
        energy_gradient_u(
            u_flat,
            p0=p.p0,
            q0=p.q0,
            s=prepared.s,
            K_seg=prepared.K_seg,
            u_star=p.u_star,
            m_src=p.m_src,
            r_src=p.r_src,
            m_local_fun=p.m_local_fun,
            m_moment=p.m_moment,
            lumen_query=p.lumen_query,
            use_magnetic=True,
            use_contact=bool(p.use_contact_in_jacobian),
            contact=p.contact,
        ),
        dtype=float,
    ).reshape(-1)


def _finite_difference_hessian(
    *,
    u0: np.ndarray,
    prepared: _PreparedProblem,
    g0: np.ndarray,
    eps: float,
    scheme: DifferenceScheme,
    workers: int,
) -> tuple[np.ndarray, int]:
    n = u0.size

    def column(j: int) -> np.ndarray:
        direction = np.zeros(n, dtype=float)
        direction[j] = eps
        if scheme == "forward":
            return (_gradient(u0 + direction, prepared) - g0) / eps
        return (
            _gradient(u0 + direction, prepared)
            - _gradient(u0 - direction, prepared)
        ) / (2.0 * eps)

    columns = _map_ordered(column, range(n), workers=workers)
    evaluations = n if scheme == "forward" else 2 * n
    return np.column_stack(columns), evaluations


def _finite_difference_Gtheta(
    *,
    u0: np.ndarray,
    prepared0: _PreparedProblem,
    theta_model,
    g0: np.ndarray,
    eps: float,
    scheme: DifferenceScheme,
    workers: int,
) -> tuple[np.ndarray, int]:
    problem = prepared0.problem
    theta0 = np.asarray(theta_model.theta0, dtype=float).reshape(7)

    def gradient_at(theta: np.ndarray) -> np.ndarray:
        perturbed = _problem_at_theta(problem, theta_model, theta)
        # Six of seven theta columns preserve beam length/stiffness. Avoid
        # rebuilding s and inverting all segment matrices for those columns.
        if (
            float(perturbed.L_model) == float(problem.L_model)
            and float(perturbed.wire_len) == float(problem.wire_len)
        ):
            prepared = _PreparedProblem(
                problem=perturbed,
                s=prepared0.s,
                K_seg=prepared0.K_seg,
            )
        else:
            prepared = _prepare(perturbed)
        return _gradient(u0, prepared)

    def column(j: int) -> np.ndarray:
        direction = np.zeros(7, dtype=float)
        direction[j] = eps
        if scheme == "forward":
            return (gradient_at(theta0 + direction) - g0) / eps
        return (
            gradient_at(theta0 + direction)
            - gradient_at(theta0 - direction)
        ) / (2.0 * eps)

    columns = _map_ordered(column, range(7), workers=workers)
    evaluations = 7 if scheme == "forward" else 14
    return np.column_stack(columns), evaluations


def _tip_output_jacobians_u_from_solution(
    *,
    u0: np.ndarray,
    s: np.ndarray,
    q_nodes: np.ndarray | None,
    q0: np.ndarray,
    e1: np.ndarray = np.array([-1.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return position and tangent sensitivities with respect to beam strain.

    Returns
    -------
    P_u:
        d tip_xyz / d u_beam, shape (3, n_u).
    T_u:
        d tip_tangent / d u_beam, shape (3, n_u).
    tip_tangent:
        Nominal unit tangent, shape (3,).

    The recurrence is the same analytic discrete kinematic sensitivity used by
    the legacy code. No equilibrium/BVP solve is performed here.
    """
    u0 = np.asarray(u0, dtype=float).reshape(-1)
    s = np.asarray(s, dtype=float).reshape(-1)
    e1 = np.asarray(e1, dtype=float).reshape(3)
    n_seg = s.size - 1
    n_u = u0.size
    if n_u != 3 * n_seg:
        raise ValueError(f"Expected {3 * n_seg} strain variables, got {n_u}.")

    if q_nodes is None:
        _, q_all, S_p_all, S_q_all = integrate_pq_and_sens_from_u(
            u0,
            p0=np.zeros(3),
            q0=q0,
            s=s,
            e1=e1,
        )
        q_tip = np.asarray(q_all[:, -1], dtype=float)
        P_u = np.asarray(S_p_all[:, -1, :], dtype=float)
        S_q = np.asarray(S_q_all[:, -1, :], dtype=float)
        tip_tangent = quat_to_R(q_tip) @ e1
        T_u = _rotated_vector_jacobian_q(q_tip, e1) @ S_q
        return P_u, T_u, tip_tangent

    q_nodes = np.asarray(q_nodes, dtype=float)
    if q_nodes.shape != (4, s.size):
        raise ValueError(
            f"Cached q has shape {q_nodes.shape}, expected {(4, s.size)}."
        )

    ds = np.diff(s)
    u_seg = u0.reshape(n_seg, 3)
    S_p = np.zeros((3, n_u), dtype=float)
    S_q = np.zeros((4, n_u), dtype=float)

    for i in range(n_seg):
        h = float(ds[i])
        ui = u_seg[i]
        q_i = q_nodes[:, i]
        q_next = q_nodes[:, i + 1]
        dq = quat_exp_body(ui, h)

        S_p += h * _rotated_vector_jacobian_q(q_i, e1) @ S_q

        dqnext_dq = (
            _normalization_jacobian(q_next)
            @ _quat_right_matrix(dq)
            @ _normalization_jacobian(q_i)
        )
        dqnext_du = (
            _normalization_jacobian(q_next)
            @ _quat_left_matrix(q_i)
            @ _quat_exp_body_jacobian_u(ui, h)
        )
        S_q = dqnext_dq @ S_q
        S_q[:, 3 * i : 3 * i + 3] += dqnext_du

    q_tip = q_nodes[:, -1]
    tip_tangent = quat_to_R(q_tip) @ e1
    T_u = _rotated_vector_jacobian_q(q_tip, e1) @ S_q
    return S_p, T_u, np.asarray(tip_tangent, dtype=float)


def _tip_output_direct_theta_derivative(
    *,
    u0: np.ndarray,
    problem,
    theta_model,
    nominal_tip: np.ndarray,
    nominal_tangent: np.ndarray,
    eps: float,
    scheme: DifferenceScheme,
    e1: np.ndarray = np.array([-1.0, 0.0, 0.0]),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fixed-strain direct derivatives of tip position and tip tangent.

    Source translation/rotation affect the output through the equilibrium
    sensitivity du/dtheta. Only insertion length changes the fixed-strain
    kinematic map directly. These evaluations are kinematic integrations, not
    equilibrium optimizations.
    """
    Jp = np.zeros((3, 7), dtype=float)
    Jt = np.zeros((3, 7), dtype=float)
    theta0 = np.asarray(theta_model.theta0, dtype=float).reshape(7)
    e1 = np.asarray(e1, dtype=float).reshape(3)

    def output_at(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_theta = _problem_at_theta(problem, theta_model, theta)
        s_theta = np.linspace(
            0.0,
            float(p_theta.L_model),
            int(p_theta.N_nodes),
        )
        p_nodes, q_nodes, _ = integrate_pq_from_u(
            u0,
            p0=p_theta.p0,
            q0=p_theta.q0,
            s=s_theta,
        )
        tip = np.asarray(p_nodes[:, -1], dtype=float)
        tangent = quat_to_R(np.asarray(q_nodes[:, -1], dtype=float)) @ e1
        return tip, np.asarray(tangent, dtype=float)

    d = np.zeros(7, dtype=float)
    d[6] = eps
    if scheme == "forward":
        tip_plus, tangent_plus = output_at(theta0 + d)
        Jp[:, 6] = (
            tip_plus - np.asarray(nominal_tip, dtype=float).reshape(3)
        ) / eps
        Jt[:, 6] = (
            tangent_plus - np.asarray(nominal_tangent, dtype=float).reshape(3)
        ) / eps
        return Jp, Jt, 1

    tip_plus, tangent_plus = output_at(theta0 + d)
    tip_minus, tangent_minus = output_at(theta0 - d)
    Jp[:, 6] = (tip_plus - tip_minus) / (2.0 * eps)
    Jt[:, 6] = (tangent_plus - tangent_minus) / (2.0 * eps)
    return Jp, Jt, 2


def _solve_implicit_system(
    H: np.ndarray,
    Gtheta: np.ndarray,
    *,
    options: SensitivityOptions,
) -> tuple[np.ndarray, str, float]:
    H = np.asarray(H, dtype=float)
    Gtheta = np.asarray(Gtheta, dtype=float)
    n = H.shape[0]
    eye = np.eye(n, dtype=float)
    scale = max(float(np.linalg.norm(H, ord=np.inf)), 1.0)

    # First attempt is exactly the unregularised implicit system. Regularisation
    # is introduced only if the linear solve fails or returns non-finite values.
    lambdas = [0.0]
    base = float(options.regularization_relative) * scale
    for k in range(max(0, options.regularization_attempts - 1)):
        lambdas.append(base * (10.0**k))

    last_error: Exception | None = None
    for lam in lambdas:
        try:
            H_eff = H if lam == 0.0 else H + lam * eye
            du = solve(
                H_eff,
                -Gtheta,
                assume_a="sym",
                check_finite=False,
            )
            if np.all(np.isfinite(du)):
                return np.asarray(du, dtype=float), "scipy_symmetric_solve", lam
        except (LinAlgError, ValueError) as exc:
            last_error = exc

    # Robust final fallback, still solving all seven right-hand sides together.
    du, *_ = np.linalg.lstsq(H, -Gtheta, rcond=None)
    if not np.all(np.isfinite(du)):
        raise LinAlgError("Implicit sensitivity linear solve failed.") from last_error
    return np.asarray(du, dtype=float), "lstsq", np.nan


def implicit_tip_jacobian(
    *,
    solution,
    problem,
    theta_model,
    options: SensitivityOptions | None = None,
    H_override: np.ndarray | None = None,
) -> SensitivityResult:
    """
    Optimised implicit tip sensitivity.

    The equilibrium condition is g(u, theta)=0. The implementation computes

        du/dtheta = -H^{-1} Gtheta,
        J_tip = P_u du/dtheta + P_theta.

    Performance-sensitive changes relative to a generic finite-difference
    implementation are:
      * P_u is obtained analytically from the discrete Cosserat integrator;
      * all seven right-hand sides are solved in one factorisation;
      * nominal stiffness data are prepared once;
      * H/Gtheta columns may be evaluated concurrently on an HPC CPU node;
      * an externally cached Hessian is accepted without copying/rebuilding.
    """
    started_total = time.perf_counter()
    options = options or SensitivityOptions()
    options.validate()
    workers = _workers(options)

    u0 = np.asarray(solution.u_flat_opt, dtype=float).reshape(-1)
    prepared0 = _prepare(problem)

    started = time.perf_counter()
    g0 = _gradient(u0, prepared0)
    nominal_gradient_time_s = time.perf_counter() - started
    stationarity_norm = float(np.linalg.norm(g0, ord=np.inf))
    gradient_evaluations = 1

    hessian_started = time.perf_counter()
    if H_override is None:
        H, h_evals = _finite_difference_hessian(
            u0=u0,
            prepared=prepared0,
            g0=g0,
            eps=float(options.eps_hess),
            scheme=options.difference_scheme,
            workers=workers,
        )
        gradient_evaluations += h_evals
        hessian_reused = False
        if options.symmetrise_hessian:
            H = 0.5 * (H + H.T)
    else:
        H = np.asarray(H_override, dtype=float)
        if H.shape != (u0.size, u0.size):
            raise ValueError(
                f"H_override has shape {H.shape}, expected {(u0.size, u0.size)}."
            )
        hessian_reused = True
    hessian_time_s = time.perf_counter() - hessian_started

    gtheta_started = time.perf_counter()
    Gtheta, gt_evals = _finite_difference_Gtheta(
        u0=u0,
        prepared0=prepared0,
        theta_model=theta_model,
        g0=g0,
        eps=float(options.eps_theta),
        scheme=options.difference_scheme,
        workers=workers,
    )
    gradient_evaluations += gt_evals
    gtheta_time_s = time.perf_counter() - gtheta_started

    kinematics_started = time.perf_counter()
    P_u, T_u, tip_tangent_base = _tip_output_jacobians_u_from_solution(
        u0=u0,
        s=prepared0.s,
        q_nodes=getattr(solution, "q", None),
        q0=problem.q0,
    )
    J_direct, J_tangent_direct, direct_tip_evaluations = (
        _tip_output_direct_theta_derivative(
            u0=u0,
            problem=problem,
            theta_model=theta_model,
            nominal_tip=solution.tip,
            nominal_tangent=tip_tangent_base,
            eps=float(options.eps_theta),
            scheme=options.difference_scheme,
        )
    )
    kinematics_time_s = time.perf_counter() - kinematics_started

    linear_started = time.perf_counter()
    du_dtheta, linear_solve_method, regularization = _solve_implicit_system(
        H,
        Gtheta,
        options=options,
    )
    linear_solve_time_s = time.perf_counter() - linear_started

    J_implicit = P_u @ du_dtheta
    J_tip = J_implicit + J_direct

    J_tangent_implicit = T_u @ du_dtheta
    J_tangent = J_tangent_implicit + J_tangent_direct
    J_output = np.vstack((J_tip, J_tangent))

    if not np.all(np.isfinite(J_output)):
        raise FloatingPointError(
            "Implicit tip position/tangent Jacobian contains non-finite values."
        )

    sensitivity_time_s = time.perf_counter() - started_total
    info = {
        "stationarity_norm": stationarity_norm,
        "gradient_evaluations": int(gradient_evaluations),
        "nominal_gradient_time_s": float(nominal_gradient_time_s),
        "hessian_time_s": float(hessian_time_s),
        "gtheta_time_s": float(gtheta_time_s),
        "kinematics_sensitivity_time_s": float(kinematics_time_s),
        "linear_solve_time_s": float(linear_solve_time_s),
        "sensitivity_time_s": float(sensitivity_time_s),
        "hessian_reused": bool(hessian_reused),
        "difference_scheme": str(options.difference_scheme),
        "workers": int(workers),
        "linear_solve_method": linear_solve_method,
        "regularization": float(regularization),
        "direct_tip_evaluations": int(direct_tip_evaluations),
        "output_dimension": 6,
        "tip_tangent_norm": float(np.linalg.norm(tip_tangent_base)),
        "H_condition": (
            float(np.linalg.cond(H)) if options.debug_jac else np.nan
        ),
        "debug_hessian_terms_requested": bool(options.debug_hessian_terms),
    }

    return SensitivityResult(
        J_tip=np.asarray(J_tip, dtype=float),
        du_dtheta=np.asarray(du_dtheta, dtype=float),
        J_implicit=np.asarray(J_implicit, dtype=float),
        J_direct=np.asarray(J_direct, dtype=float),
        Gtheta=np.asarray(Gtheta, dtype=float),
        P_u=np.asarray(P_u, dtype=float),
        H=np.asarray(H, dtype=float),
        info=info,
        J_tangent=np.asarray(J_tangent, dtype=float),
        J_output=np.asarray(J_output, dtype=float),
        J_tangent_implicit=np.asarray(J_tangent_implicit, dtype=float),
        J_tangent_direct=np.asarray(J_tangent_direct, dtype=float),
        T_u=np.asarray(T_u, dtype=float),
        tip_tangent_base=np.asarray(tip_tangent_base, dtype=float),
    )
