from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import time

import numpy as np

from .kinematics import (
    perturb_quat_world,
    effective_lengths,
    quat_to_R,
    integrate_pq_from_u,
    integrate_pq_and_sens_from_u,
)
from .gradients import (
    contact_energy_gradient_u_consistent,
    energy_gradient_u,
)
from .magnetism import dipole_from_pose

from beam_direction_magnetisation.ana_energy import (
    precompute_K_segments,
    hessian_diagnostic_for_grad,
    elastic_energy_gradient_u,
    magnetic_energy_gradient_u_virtual_work_analytic,
    check_magnetic_grad_against_energy_fd,
)


@dataclass(frozen=True)
class ThetaModel:
    """
    Parameterisation used by the implicit Jacobian.

    theta convention:
        [source_x, source_y, source_z,
         small_world_rot_x, small_world_rot_y, small_world_rot_z,
         insertion_length]
    """
    theta0: np.ndarray
    q_src0: np.ndarray
    rotation_convention: str = "world"

    def __post_init__(self):
        theta0 = np.asarray(self.theta0, float).reshape(-1)
        q_src0 = np.asarray(self.q_src0, float).reshape(4)

        if theta0.size != 7:
            raise ValueError(f"theta0 must have shape (7,), got {theta0.shape}.")
        if self.rotation_convention != "world":
            raise ValueError(
                "Only rotation_convention='world' is currently implemented."
            )

        q_src0 = q_src0 / (np.linalg.norm(q_src0) + 1e-12)

        object.__setattr__(self, "theta0", theta0)
        object.__setattr__(self, "q_src0", q_src0)


@dataclass(frozen=True)
class SensitivityOptions:
    eps_theta: float = 1e-6
    eps_hess: float = 1e-4
    eps_L_direct: float = 1e-5
    hessian_reg: float = 1e-10
    use_scalar_hessian: bool = False
    symmetrise_hessian: bool = True
    debug_jac: bool = False
    debug_hessian_terms: bool = False
    difference_scheme: str = "forward"

    def validate(self) -> None:
        if self.eps_theta <= 0:
            raise ValueError(f"eps_theta must be positive, got {self.eps_theta}.")
        if self.eps_hess <= 0:
            raise ValueError(f"eps_hess must be positive, got {self.eps_hess}.")
        if self.eps_L_direct <= 0:
            raise ValueError(f"eps_L_direct must be positive, got {self.eps_L_direct}.")
        if self.hessian_reg < 0:
            raise ValueError(f"hessian_reg must be non-negative, got {self.hessian_reg}.")
        if self.difference_scheme not in {"forward", "central"}:
            raise ValueError(
                "difference_scheme must be 'forward' or 'central', got "
                f"{self.difference_scheme!r}."
            )


@dataclass
class SensitivityResult:
    J_tip: np.ndarray
    H: np.ndarray
    Gtheta: np.ndarray
    du_dtheta: np.ndarray
    P_u: np.ndarray
    J_direct: np.ndarray
    J_implicit: np.ndarray
    p_tip_base: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


def ake_energy_grad_fun_for_problem(
    *,
    problem,
    theta_model: ThetaModel,
):
    """
    Build grad_u Π(u, theta) for the current nominal problem.

    This replaces the old nested energy_grad_fun construction. It keeps the same
    theta convention but rebuilds length-dependent quantities consistently when
    theta[-1] changes.
    """
    q_src0 = theta_model.q_src0.copy()

    def grad_fun(u_flat: np.ndarray, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, float).reshape(7)

        r_src = theta[0:3]
        drot_world = theta[3:6]
        L_ins = float(theta[6])

        q_src = perturb_quat_world(q_src0, drot_world)
        m_src = dipole_from_pose(q_src, problem.m_body)

        L_model, wire_len, tip_len = effective_lengths(
            L_ins,
            L_tip_full=problem.L_tip_full,
            L_tip_min=problem.L_tip_min,
        )

        s = np.linspace(0.0, float(L_model), int(problem.N_nodes))
        K_seg = precompute_K_segments(s, problem.Kinv_fun, wire_len)

        m_local_fun = problem.m_local_factory(
            L_model=L_model,
            wire_len=wire_len,
            tip_len=tip_len,
        )

        return energy_gradient_u(
            u_flat,
            p0=problem.p0,
            q0=problem.q0,
            s=s,
            K_seg=K_seg,
            u_star=problem.u_star,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=problem.m_moment,
            lumen_query=problem.lumen_query
            if problem.use_contact_in_jacobian
            else None,
            use_magnetic=True,
            use_contact=problem.use_contact_in_jacobian,
            contact=problem.contact,
        )

    return grad_fun


def finite_difference_jacobian_of_gradient(
    *,
    grad_fun: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    eps: float,
    scheme: str = "central",
    base_value: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute d grad_fun / du by forward or central differences.

    Returns H with columns:
        H[:, k] = d grad / d u_k
    """
    u0 = np.asarray(u0, float).reshape(-1)
    n_u = u0.size

    H = np.empty((n_u, n_u), dtype=float)
    if scheme not in {"forward", "central"}:
        raise ValueError(f"Unknown finite-difference scheme {scheme!r}.")
    if scheme == "forward":
        g0 = (
            np.asarray(grad_fun(u0), float).reshape(-1)
            if base_value is None
            else np.asarray(base_value, float).reshape(-1)
        )

    for k in range(n_u):
        up = u0.copy()
        up[k] += eps
        gp = np.asarray(grad_fun(up), float).reshape(-1)
        if scheme == "forward":
            H[:, k] = (gp - g0) / eps
        else:
            um = u0.copy()
            um[k] -= eps
            gm = np.asarray(grad_fun(um), float).reshape(-1)
            H[:, k] = (gp - gm) / (2.0 * eps)

    return H


def finite_difference_theta_gradient(
    *,
    energy_grad_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    u_opt: np.ndarray,
    theta0: np.ndarray,
    eps: float,
    scheme: str = "central",
    base_value: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute Gtheta = d/dtheta grad_u Π(u_opt, theta).
    """
    u_opt = np.asarray(u_opt, float).reshape(-1)
    theta0 = np.asarray(theta0, float).reshape(-1)

    n_u = u_opt.size
    n_theta = theta0.size

    Gtheta = np.empty((n_u, n_theta), dtype=float)
    if scheme not in {"forward", "central"}:
        raise ValueError(f"Unknown finite-difference scheme {scheme!r}.")
    if scheme == "forward":
        g0 = (
            np.asarray(energy_grad_fun(u_opt, theta0), float).reshape(-1)
            if base_value is None
            else np.asarray(base_value, float).reshape(-1)
        )

    for j in range(n_theta):
        thp = theta0.copy()
        thp[j] += eps
        gp = np.asarray(energy_grad_fun(u_opt, thp), float).reshape(-1)
        if scheme == "forward":
            Gtheta[:, j] = (gp - g0) / eps
        else:
            thm = theta0.copy()
            thm[j] -= eps
            gm = np.asarray(energy_grad_fun(u_opt, thm), float).reshape(-1)
            Gtheta[:, j] = (gp - gm) / (2.0 * eps)

    return Gtheta


def direct_tip_length_jacobian(
    *,
    u_opt: np.ndarray,
    p0: np.ndarray,
    q0: np.ndarray,
    L: float,
    N: int,
    n_theta: int,
    eps_L: float,
) -> np.ndarray:
    P_theta_direct = np.zeros((3, n_theta), dtype=float)

    def tip_from_fixed_u_at_L(L_eval: float) -> np.ndarray:
        p_eval, _, _ = integrate_pq_from_u(
            u_opt,
            p0=p0,
            q0=q0,
            s=np.linspace(0.0, float(L_eval), int(N)),
        )
        return p_eval[:, -1]

    if L - eps_L <= 0.0:
        p_base = tip_from_fixed_u_at_L(L)
        p_plus = tip_from_fixed_u_at_L(L + eps_L)
        P_theta_direct[:, -1] = (p_plus - p_base) / eps_L
    else:
        p_plus = tip_from_fixed_u_at_L(L + eps_L)
        p_minus = tip_from_fixed_u_at_L(L - eps_L)
        P_theta_direct[:, -1] = (p_plus - p_minus) / (2.0 * eps_L)

    return P_theta_direct


def run_hessian_diagnostics(
    *,
    problem,
    u_opt: np.ndarray,
    s: np.ndarray,
    K_seg: np.ndarray,
    energy_grad_fun,
    theta0: np.ndarray,
    eps_hess: float,
) -> None:
    """
    Optional debugging diagnostics copied from the old workflow.
    """
    hessian_diagnostic_for_grad(
        "elastic only",
        lambda u: elastic_energy_gradient_u(
            u,
            s=s,
            K_seg=K_seg,
            u_star=problem.u_star,
        ),
        u_opt,
        eps=eps_hess,
    )

    hessian_diagnostic_for_grad(
        "magnetic only",
        lambda u: magnetic_energy_gradient_u_virtual_work_analytic(
            u,
            p0=problem.p0,
            q0=problem.q0,
            s=s,
            m_src=problem.m_src,
            r_src=problem.r_src,
            m_local_fun=problem.m_local_fun,
            m_moment=problem.m_moment,
        ),
        u_opt,
        eps=eps_hess,
    )

    check_magnetic_grad_against_energy_fd(
        u_opt,
        p0=problem.p0,
        q0=problem.q0,
        s=s,
        m_src=problem.m_src,
        r_src=problem.r_src,
        m_local_fun=problem.m_local_fun,
        m_moment=problem.m_moment,
        wire_len=problem.wire_len,
        eps_list=(1e-3, 3e-4, 1e-4, 3e-5),
        max_cols=12,
    )

    if problem.use_contact_in_jacobian and problem.lumen_query is not None:
        hessian_diagnostic_for_grad(
            "contact only",
            lambda u: contact_energy_gradient_u_consistent(
                u,
                p0=problem.p0,
                q0=problem.q0,
                s=s,
                lumen_query=problem.lumen_query,
                contact=problem.contact,
            ),
            u_opt,
            eps=eps_hess,
        )

    hessian_diagnostic_for_grad(
        "full energy_grad_fun",
        lambda u: energy_grad_fun(u, theta0),
        u_opt,
        eps=eps_hess,
    )


def implicit_tip_jacobian(
    *,
    solution,
    problem,
    theta_model: ThetaModel,
    options: SensitivityOptions | None = None,
    H_override: np.ndarray | None = None,
) -> SensitivityResult:
    """
    Compute d tip_xyz / d theta using implicit differentiation of the
    energy-minimising strain field.

    The stationarity condition is:

        grad_u Π(u*, theta) = 0

    Differentiating gives:

        H du*/dtheta + Gtheta = 0

    therefore:

        du*/dtheta = -H^{-1} Gtheta

    The tip Jacobian is:

        d p_tip / dtheta = P_u du*/dtheta + P_theta_direct
    """
    if options is None:
        options = SensitivityOptions()
    options.validate()

    u_opt = np.asarray(solution.u_flat_opt, float).reshape(-1)
    theta0 = theta_model.theta0.copy()

    n_u = u_opt.size
    n_theta = theta0.size

    s = np.linspace(0.0, float(problem.L_model), int(problem.N_nodes))
    K_seg = precompute_K_segments(s, problem.Kinv_fun, problem.wire_len)

    expected = 3 * (s.size - 1)
    if n_u != expected:
        raise ValueError(
            f"u_opt has size {n_u}, expected {expected} for "
            f"N_nodes={problem.N_nodes}."
        )

    raw_energy_grad_fun = ake_energy_grad_fun_for_problem(
        problem=problem,
        theta_model=theta_model,
    )
    gradient_evaluations = 0
    gradient_time_s = 0.0

    def energy_grad_fun(u, theta):
        nonlocal gradient_evaluations, gradient_time_s
        started = time.perf_counter()
        value = raw_energy_grad_fun(u, theta)
        gradient_evaluations += 1
        gradient_time_s += time.perf_counter() - started
        return value

    sensitivity_started = time.perf_counter()
    g0 = np.asarray(energy_grad_fun(u_opt, theta0), float).reshape(-1)
    contact_gradient_evaluated = bool(
        problem.use_contact_in_jacobian
        and problem.lumen_query is not None
    )

    contact_gradient_norm = 0.0
    contact_gradient_inf = 0.0

    if contact_gradient_evaluated:
        g_contact = contact_energy_gradient_u_consistent(
            u_opt,
            p0=problem.p0,
            q0=problem.q0,
            s=s,
            lumen_query=problem.lumen_query,
            contact=problem.contact,
        )

        g_contact = np.asarray(
            g_contact,
            float,
        ).reshape(-1)

        contact_gradient_norm = float(
            np.linalg.norm(g_contact)
        )

        contact_gradient_inf = float(
            np.max(np.abs(g_contact))
        )
    if options.debug_jac:
        print("\n--- IMPLICIT JAC STATIONARITY ---")
        print("||g0|| =", np.linalg.norm(g0))
        print("max |g0| =", np.max(np.abs(g0)))
        print("mean |g0| =", np.mean(np.abs(g0)))
        print("||u_opt|| =", np.linalg.norm(u_opt))

    if options.debug_hessian_terms:
        run_hessian_diagnostics(
            problem=problem,
            u_opt=u_opt,
            s=s,
            K_seg=K_seg,
            energy_grad_fun=energy_grad_fun,
            theta0=theta0,
            eps_hess=options.eps_hess,
        )

    if options.use_scalar_hessian:
        raise NotImplementedError(
            "Scalar-energy Hessian path is not wired in this refactor yet. "
            "Use finite differences of the analytic gradient."
        )

    hessian_reused = H_override is not None
    if H_override is None:
        H_raw = finite_difference_jacobian_of_gradient(
            grad_fun=lambda u: energy_grad_fun(u, theta0),
            u0=u_opt,
            eps=options.eps_hess,
            scheme=options.difference_scheme,
            base_value=g0,
        )
    else:
        H_raw = np.asarray(H_override, float).copy()
        if H_raw.shape != (n_u, n_u):
            raise ValueError(
                f"H_override must have shape {(n_u, n_u)}, got {H_raw.shape}."
            )
        if not np.all(np.isfinite(H_raw)):
            raise ValueError("H_override contains non-finite values.")

    H = 0.5 * (H_raw + H_raw.T) if options.symmetrise_hessian else H_raw
    H_reg = H + options.hessian_reg * np.eye(n_u)

    if options.debug_jac:
        H_sym_dbg = 0.5 * (H_raw + H_raw.T)
        print("\n--- HESSIAN DIAGNOSTIC ---")
        print("||H_raw|| =", np.linalg.norm(H_raw))
        print("cond(H_reg) =", np.linalg.cond(H_reg))
        print(
            "min/max eig sym(H_raw) =",
            np.min(np.linalg.eigvalsh(H_sym_dbg)),
            np.max(np.linalg.eigvalsh(H_sym_dbg)),
        )
        print(
            "H asymmetry =",
            np.linalg.norm(H_raw - H_raw.T) / (np.linalg.norm(H_raw) + 1e-12),
        )

    Gtheta = finite_difference_theta_gradient(
        energy_grad_fun=energy_grad_fun,
        u_opt=u_opt,
        theta0=theta0,
        eps=options.eps_theta,
        scheme=options.difference_scheme,
        base_value=g0,
    )

    du_dtheta = -np.linalg.solve(H_reg, Gtheta)

    if options.debug_jac:
        print("\n--- DU_DTHETA ---")
        print("||du_dtheta|| =", np.linalg.norm(du_dtheta))
        print("column norms =", np.linalg.norm(du_dtheta, axis=0))
        print("L column norm =", np.linalg.norm(du_dtheta[:, -1]))

    p_base, q_base, S_p_base, S_q_base = integrate_pq_and_sens_from_u(
        u_opt,
        p0=problem.p0,
        q0=problem.q0,
        s=s,
    )

    p_tip_base = p_base[:, -1].copy()
    P_u = S_p_base[:, -1, :]

    if P_u.shape != (3, n_u):
        raise ValueError(f"P_u must have shape (3, {n_u}), got {P_u.shape}.")

    J_direct = direct_tip_length_jacobian(
        u_opt=u_opt,
        p0=problem.p0,
        q0=problem.q0,
        L=problem.L_model,
        N=problem.N_nodes,
        n_theta=n_theta,
        eps_L=options.eps_L_direct,
    )

    J_implicit = P_u @ du_dtheta
    J_tip = J_implicit + J_direct
 
    if options.debug_jac:
        q_tip = q_base[:, -1]
        R_tip = quat_to_R(q_tip)
        t_tip = R_tip @ np.array([-1.0, 0.0, 0.0])

        print("\n--- ANALYTIC J L BREAKDOWN ---")
        print("tangent direct approximation:")
        print(t_tip)
        print("direct L term:")
        print(J_direct[:, -1])
        print("implicit L term:")
        print(J_implicit[:, -1])
        print("total L term:")
        print(J_tip[:, -1])
        print("||Gtheta_L|| =", np.linalg.norm(Gtheta[:, -1]))
        print("||du_dtheta_L|| =", np.linalg.norm(du_dtheta[:, -1]))
        print("||P_u|| =", np.linalg.norm(P_u))

    info = {
        "stationarity_norm": float(np.linalg.norm(g0)),
        "H_raw": H_raw,
        "H_used": H,
        "H_condition": float(np.linalg.cond(H_reg)),
        "theta0": theta0.copy(),
        "s": s.copy(),
        "difference_scheme": options.difference_scheme,
        "hessian_reused": bool(hessian_reused),
        "gradient_evaluations": int(gradient_evaluations),
        "gradient_time_s": float(gradient_time_s),
        "sensitivity_time_s": float(time.perf_counter() - sensitivity_started),
    }

    return SensitivityResult(
        J_tip=np.asarray(J_tip, float).copy(),
        H=np.asarray(H, float).copy(),
        Gtheta=np.asarray(Gtheta, float).copy(),
        du_dtheta=np.asarray(du_dtheta, float).copy(),
        P_u=np.asarray(P_u, float).copy(),
        J_direct=np.asarray(J_direct, float).copy(),
        J_implicit=np.asarray(J_implicit, float).copy(),
        p_tip_base=np.asarray(p_tip_base, float).copy(),
        info=info,
    )
