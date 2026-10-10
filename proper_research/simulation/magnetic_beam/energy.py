from __future__ import annotations

from typing import Any

import numpy as np

from .magnetism import (
    magnetic_energy_quantities_cosserat_profile_segments,
    magnetic_wrench_density_cosserat_profile_segments,
)
from .kinematics import integrate_pq_from_u
from .contact import (
    ContactParams,
    LumenQuery,
    contact_energy_from_p,
)


def elastic_energy_from_segments(
    *,
    u_seg: np.ndarray,
    s: np.ndarray,
    K_seg: np.ndarray,
    u_star: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute elastic energy split into total, twist, and bending components.
    """
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)

    u_seg = np.asarray(u_seg, float)
    K_seg = np.asarray(K_seg, float)
    u_star = np.asarray(u_star, float).reshape(3)

    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    if u_seg.shape != (len(ds), 3):
        raise ValueError(f"u_seg must have shape ({len(ds)}, 3), got {u_seg.shape}.")

    if K_seg.shape != (len(ds), 3, 3):
        raise ValueError(f"K_seg must have shape ({len(ds)}, 3, 3), got {K_seg.shape}.")

    W_el_seg = np.empty(len(ds), dtype=float)
    W_t_seg = np.empty(len(ds), dtype=float)
    W_b_seg = np.empty(len(ds), dtype=float)

    for i, dsi in enumerate(ds):
        du = u_seg[i] - u_star
        K = K_seg[i]

        du_t = du[0:1]
        du_b = du[1:3]

        K_tt = K[0:1, 0:1]
        K_bb = K[1:3, 1:3]

        W_el_seg[i] = 0.5 * float(du @ K @ du) * dsi
        W_t_seg[i] = 0.5 * float(du_t @ K_tt @ du_t) * dsi
        W_b_seg[i] = 0.5 * float(du_b @ K_bb @ du_b) * dsi

    W_el = float(np.sum(W_el_seg))
    W_t = float(np.sum(W_t_seg))
    W_b = float(np.sum(W_b_seg))

    return W_el, W_t, W_b, W_el_seg, W_t_seg, W_b_seg


def magnetic_energy_from_centerline(
    *,
    p: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    m_src: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_moment: float,
    r_min: float = 1e-6,
    return_parts: bool = True,
) -> tuple[float, dict[str, np.ndarray]]:
    """
    Compute segment-midpoint distributed magnetic energy.

    Energy convention:
        W_m = integral -m . B ds
    """
    s = np.asarray(s, float).ravel()
    ds = np.diff(s)

    if return_parts:
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

    m_dot_B_mid = np.sum(m_world_mid * B_mid, axis=0)
    w_m_mid = -m_dot_B_mid

    if w_m_mid.shape != ds.shape:
        raise ValueError(
            f"Magnetic energy density shape {w_m_mid.shape} does not match ds shape {ds.shape}."
        )

    W_m = float(np.sum(w_m_mid * ds))

    if not return_parts:
        return W_m, {}

    Bnorm = np.linalg.norm(B_mid, axis=0)
    mnorm = np.linalg.norm(m_world_mid, axis=0)

    cos_th = np.sum(m_world_mid * B_mid, axis=0) / (Bnorm * mnorm + 1e-16)
    cos_th = np.clip(cos_th, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_th))

    tau_norm = np.linalg.norm(tau_mid, axis=0)

    parts = {
        "B": np.asarray(B_mid, float).copy(),
        "Bnorm": np.asarray(Bnorm, float).copy(),
        "m_world": np.asarray(m_world_mid, float).copy(),
        "mnorm": np.asarray(mnorm, float).copy(),
        "m_dot_B": np.asarray(m_dot_B_mid, float).copy(),
        "w_m": np.asarray(w_m_mid, float).copy(),
        "angle_deg": np.asarray(angle_deg, float).copy(),
        "tau_norm": np.asarray(tau_norm, float).copy(),
        "s_mid": np.asarray(s_mid, float).copy(),
        "f_mid": np.asarray(f_mid, float).copy(),
        "tau_mid": np.asarray(tau_mid, float).copy(),
    }

    return W_m, parts


def contact_energy_from_centerline(
    *,
    p: np.ndarray,
    s: np.ndarray,
    lumen_query: LumenQuery | None,
    lumen_C: np.ndarray | None,
    lumen_R: np.ndarray | None,
    contact: ContactParams | None,
    return_parts: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Compute contact/lumen penalty energy from the centerline.
    """
    if lumen_query is None:
        if lumen_C is None or lumen_R is None:
            raise ValueError(
                "Contact energy requires either lumen_query or both lumen_C and lumen_R."
            )
        lumen_query = LumenQuery(lumen_C, lumen_R)

    if contact is None:
        contact = ContactParams()

    contact.validate()

    if not return_parts:
        W_cf = contact_energy_from_p(
            p,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
            return_force=False,
        )
        return float(W_cf), {}

    W_cf, C_nodes, F_nodes, gap_nodes, w_contact = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )

    parts = {
        "gap_min": float(np.min(gap_nodes)),
        "gap_nodes": np.asarray(gap_nodes, float).copy(),
        "contact_weights": np.asarray(w_contact, float).copy(),
        "C_nodes": np.asarray(C_nodes, float).copy(),
        "F_nodes": np.asarray(F_nodes, float).copy(),
    }

    return float(W_cf), parts


def gravity_energy_from_centerline(
    *,
    p: np.ndarray,
    s: np.ndarray,
    gravity_force_density: np.ndarray | None,
    wire_len: float = 0.0,
) -> float:
    """
    Optional gravitational potential contribution.

    If gravity_force_density is None, gravity is disabled.

    ``gravity_force_density`` accepts two forms:
      - a plain (3,) array/tuple: UNIFORM force density along the whole beam
        (the original behaviour -- kept for backward compatibility and for
        any single-material use).
      - a callable ``fn(s, wire_len) -> (3, len(s))`` array: force density
        evaluated AT THE NODES (not segment midpoints -- this integrates via
        trapezoidal rule against ``s`` directly, unlike Kinv_fun/m_local_fun
        which operate on segment midpoints), for a bimaterial (wire/tip)
        beam where the weight-per-length genuinely differs by section. See
        ``make_wire_tip_gravity_force_density_fun`` in
        run_solver_smoke_test.py for the standard wire/tip factory.
    """
    if gravity_force_density is None:
        return 0.0

    s = np.asarray(s, float).ravel()

    if callable(gravity_force_density):
        fg = np.asarray(gravity_force_density(s, wire_len), float)
        if fg.shape != (3, s.size):
            raise ValueError(
                f"gravity_force_density(s, wire_len) must return shape "
                f"(3, {s.size}), got {fg.shape}."
            )
    else:
        fg_uniform = np.asarray(gravity_force_density, float).reshape(3)
        fg = np.repeat(fg_uniform[:, None], s.size, axis=1)

    return float(-np.trapz(np.sum(fg * p, axis=0), s))


def energy_from_u(
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
    debug_mag: bool = False,
    return_parts: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Compute total potential energy Π(u).

    Components:
      - elastic strain energy
      - distributed magnetic energy
      - optional gravity
      - optional lumen/contact penalty
    """
    s = np.asarray(s, float).ravel()
    u_star = np.asarray(u_star, float).reshape(3)

    p, q, u_seg = integrate_pq_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    W_el, W_t, W_b, W_el_seg, W_t_seg, W_b_seg = elastic_energy_from_segments(
        u_seg=u_seg,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
    )

    W_m, mag_parts = magnetic_energy_from_centerline(
        p=p,
        q=q,
        s=s,
        m_src=m_src,
        r_src=r_src,
        m_local_fun=m_local_fun,
        m_moment=m_moment,
        return_parts=return_parts,
    )

    W_g = gravity_energy_from_centerline(
        p=p,
        s=s,
        gravity_force_density=gravity_force_density,
        wire_len=wire_len,
    )

    W_cf = 0.0
    contact_parts: dict[str, Any] = {
        "gap_min": np.nan,
        "gap_nodes": None,
        "contact_weights": None,
        "C_nodes": None,
        "F_nodes": None,
    }

    if use_lumen:
        W_cf, contact_parts = contact_energy_from_centerline(
            p=p,
            s=s,
            lumen_query=lumen_query,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            contact=contact,
            return_parts=return_parts,
        )

    W_total = float(W_el + W_m + W_g + W_cf)

    if not return_parts:
        return W_total, {}

    if debug_mag:
        print("\n[DBG-MAG]")
        print(f"W_el={W_el:.6e}  W_m={W_m:.6e}  W_g={W_g:.6e}  W_cf={W_cf:.6e}")
        print(f"max |B|={np.max(mag_parts['Bnorm']):.6e}")
        print(f"max |m|={np.max(mag_parts['mnorm']):.6e}")
        print(f"max |m x B|={np.max(mag_parts['tau_norm']):.6e}")
        print(f"max angle(m,B) [deg]={np.max(mag_parts['angle_deg']):.3f}")

    parts: dict[str, Any] = {
        "W_el": float(W_el),
        "W_b": float(W_b),
        "W_t": float(W_t),
        "W_m": float(W_m),
        "W_g": float(W_g),
        "W_cf": float(W_cf),
        "s": s.copy(),
        "p": np.asarray(p, float).copy(),
        "q": np.asarray(q, float).copy(),
        "W_el_seg": np.asarray(W_el_seg, float).copy(),
        "W_b_seg": np.asarray(W_b_seg, float).copy(),
        "W_t_seg": np.asarray(W_t_seg, float).copy(),
    }

    parts.update(mag_parts)
    parts.update(contact_parts)

    return W_total, parts
