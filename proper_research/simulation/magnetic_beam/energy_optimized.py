from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .contact import ContactParams, LumenQuery, contact_energy_from_p
from .kinematics import integrate_pq_from_u
from .magnetism import (
    magnetic_energy_quantities_cosserat_profile_segments,
    magnetic_wrench_density_cosserat_profile_segments,
)


ResultDetail = Literal["none", "contact", "full"]


def _validate_detail(detail: str) -> ResultDetail:
    if detail not in {"none", "contact", "full"}:
        raise ValueError("detail must be 'none', 'contact', or 'full'.")
    return detail  # type: ignore[return-value]


def elastic_energy_from_segments_optimized(
    *,
    u_seg: np.ndarray,
    s: np.ndarray,
    K_seg: np.ndarray,
    u_star: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised equivalent of the legacy per-segment elastic-energy loop."""
    s = np.asarray(s, dtype=float).reshape(-1)
    ds = np.diff(s)
    u_seg = np.asarray(u_seg, dtype=float)
    K_seg = np.asarray(K_seg, dtype=float)
    u_star = np.asarray(u_star, dtype=float).reshape(3)

    n_seg = ds.size
    if n_seg < 1:
        raise ValueError("s must contain at least two nodes.")
    if u_seg.shape != (n_seg, 3):
        raise ValueError(f"u_seg must have shape {(n_seg, 3)}, got {u_seg.shape}.")
    if K_seg.shape != (n_seg, 3, 3):
        raise ValueError(
            f"K_seg must have shape {(n_seg, 3, 3)}, got {K_seg.shape}."
        )

    du = u_seg - u_star[None, :]

    # Each expression is exactly the legacy quadratic form, evaluated in one
    # batched NumPy call instead of many tiny Python/NumPy calls.
    W_el_seg = 0.5 * np.einsum(
        "ni,nij,nj->n", du, K_seg, du, optimize=True
    ) * ds
    W_t_seg = 0.5 * (du[:, 0] * K_seg[:, 0, 0] * du[:, 0]) * ds
    W_b_seg = 0.5 * np.einsum(
        "ni,nij,nj->n",
        du[:, 1:3],
        K_seg[:, 1:3, 1:3],
        du[:, 1:3],
        optimize=True,
    ) * ds

    return (
        float(np.sum(W_el_seg)),
        float(np.sum(W_t_seg)),
        float(np.sum(W_b_seg)),
        W_el_seg,
        W_t_seg,
        W_b_seg,
    )


def _magnetic_energy_from_state(
    *,
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    m_src: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_moment: float,
    detail: ResultDetail,
    r_min: float = 1e-6,
) -> tuple[float, dict[str, Any]]:
    ds = np.diff(np.asarray(s, dtype=float).reshape(-1))

    if detail == "full":
        f_mid, tau_mid, B_mid, m_world_mid, s_mid = (
            magnetic_wrench_density_cosserat_profile_segments(
                p,
                q,
                s,
                m_src,
                r_src,
                m_local_fun,
                m_moment,
                r_min=r_min,
            )
        )
    else:
        B_mid, m_world_mid, s_mid = (
            magnetic_energy_quantities_cosserat_profile_segments(
                p,
                q,
                s,
                m_src,
                r_src,
                m_local_fun,
                m_moment,
                r_min=r_min,
            )
        )
        f_mid = tau_mid = None

    m_dot_B = np.einsum("ij,ij->j", m_world_mid, B_mid, optimize=True)
    w_m = -m_dot_B
    if w_m.shape != ds.shape:
        raise ValueError(
            f"Magnetic energy density shape {w_m.shape} does not match {ds.shape}."
        )
    W_m = float(np.dot(w_m, ds))

    if detail != "full":
        return W_m, {}

    Bnorm = np.linalg.norm(B_mid, axis=0)
    mnorm = np.linalg.norm(m_world_mid, axis=0)
    cos_th = m_dot_B / (Bnorm * mnorm + 1e-16)
    angle_deg = np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))
    tau_norm = np.linalg.norm(tau_mid, axis=0)

    return W_m, {
        "B": np.asarray(B_mid, dtype=float).copy(),
        "Bnorm": np.asarray(Bnorm, dtype=float).copy(),
        "m_world": np.asarray(m_world_mid, dtype=float).copy(),
        "mnorm": np.asarray(mnorm, dtype=float).copy(),
        "m_dot_B": np.asarray(m_dot_B, dtype=float).copy(),
        "w_m": np.asarray(w_m, dtype=float).copy(),
        "angle_deg": np.asarray(angle_deg, dtype=float).copy(),
        "tau_norm": np.asarray(tau_norm, dtype=float).copy(),
        "s_mid": np.asarray(s_mid, dtype=float).copy(),
        "f_mid": np.asarray(f_mid, dtype=float).copy(),
        "tau_mid": np.asarray(tau_mid, dtype=float).copy(),
    }


def _contact_energy_from_state(
    *,
    p: np.ndarray,
    s: np.ndarray,
    lumen_query: LumenQuery | None,
    lumen_C: np.ndarray | None,
    lumen_R: np.ndarray | None,
    contact: ContactParams | None,
    detail: ResultDetail,
) -> tuple[float, dict[str, Any]]:
    if lumen_query is None:
        if lumen_C is None or lumen_R is None:
            raise ValueError(
                "Contact energy requires lumen_query or both lumen_C and lumen_R."
            )
        lumen_query = LumenQuery(lumen_C, lumen_R)

    contact = contact or ContactParams()
    contact.validate()

    if detail == "none":
        W_cf = contact_energy_from_p(
            p,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
            return_force=False,
        )
        return float(W_cf), {}

    # Contact detail is retained in both "contact" and "full" modes because it
    # is needed for active-set/Hessian reuse and controller safety checks.
    W_cf, C_nodes, F_nodes, gap_nodes, weights = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )
    return float(W_cf), {
        "gap_min": float(np.min(gap_nodes)),
        "gap_nodes": np.asarray(gap_nodes, dtype=float).copy(),
        "contact_weights": np.asarray(weights, dtype=float).copy(),
        "C_nodes": np.asarray(C_nodes, dtype=float).copy(),
        "F_nodes": np.asarray(F_nodes, dtype=float).copy(),
    }


def energy_from_state_optimized(
    *,
    p: np.ndarray,
    q: np.ndarray,
    u_seg: np.ndarray,
    s: np.ndarray,
    K_seg: np.ndarray,
    u_star: np.ndarray,
    m_src: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_moment: float,
    wire_len: float,
    lumen_C: np.ndarray | None = None,
    lumen_R: np.ndarray | None = None,
    lumen_query: LumenQuery | None = None,
    use_lumen: bool = True,
    contact: ContactParams | None = None,
    gravity_force_density: np.ndarray | None = None,
    detail: ResultDetail = "full",
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate energy from an already integrated beam state.

    The legacy solver integrated the final state, then called ``energy_from_u``
    which integrated it a second time. This routine removes that duplicate
    integration while preserving the same energy definitions.
    """
    detail = _validate_detail(detail)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    u_seg = np.asarray(u_seg, dtype=float)
    s = np.asarray(s, dtype=float).reshape(-1)

    W_el, W_t, W_b, W_el_seg, W_t_seg, W_b_seg = (
        elastic_energy_from_segments_optimized(
            u_seg=u_seg,
            s=s,
            K_seg=K_seg,
            u_star=u_star,
        )
    )

    W_m, mag_parts = _magnetic_energy_from_state(
        p=p,
        q=q,
        s=s,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        detail=detail,
    )

    W_g = 0.0
    if gravity_force_density is not None:
        fg = np.asarray(gravity_force_density, dtype=float).reshape(3)
        W_g = float(-np.trapezoid(np.sum(fg[:, None] * p, axis=0), s))

    W_cf = 0.0
    contact_parts: dict[str, Any] = {}
    if use_lumen:
        W_cf, contact_parts = _contact_energy_from_state(
            p=p,
            s=s,
            lumen_query=lumen_query,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            contact=contact,
            detail=detail,
        )

    W_total = float(W_el + W_m + W_g + W_cf)
    if detail == "none":
        return W_total, {}

    parts: dict[str, Any] = {
        "W_el": W_el,
        "W_b": W_b,
        "W_t": W_t,
        "W_m": W_m,
        "W_g": W_g,
        "W_cf": W_cf,
        "gap_min": np.nan,
    }

    if detail == "full":
        parts.update(
            {
                "s": s.copy(),
                "p": p.copy(),
                "q": q.copy(),
                "W_el_seg": W_el_seg.copy(),
                "W_b_seg": W_b_seg.copy(),
                "W_t_seg": W_t_seg.copy(),
            }
        )

    parts.update(mag_parts)
    parts.update(contact_parts)
    return W_total, parts


def energy_from_u_optimized(
    u_flat: np.ndarray,
    *,
    p0: np.ndarray,
    q0: np.ndarray,
    s: np.ndarray,
    K_seg: np.ndarray,
    u_star: np.ndarray,
    m_src: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_moment: float,
    wire_len: float,
    lumen_C: np.ndarray | None = None,
    lumen_R: np.ndarray | None = None,
    lumen_query: LumenQuery | None = None,
    use_lumen: bool = True,
    contact: ContactParams | None = None,
    gravity_force_density: np.ndarray | None = None,
    detail: ResultDetail = "full",
) -> tuple[float, dict[str, Any]]:
    """Drop-in energy evaluator with vectorised elastic work and detail levels."""
    p, q, u_seg = integrate_pq_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )
    return energy_from_state_optimized(
        p=p,
        q=q,
        u_seg=u_seg,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        wire_len=wire_len,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_query=lumen_query,
        use_lumen=use_lumen,
        contact=contact,
        gravity_force_density=gravity_force_density,
        detail=detail,
    )
