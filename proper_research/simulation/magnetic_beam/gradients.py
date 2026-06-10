from __future__ import annotations

import numpy as np

from .contact import ContactParams

from beam_direction_magnetisation.ana_energy import (
    elastic_energy_gradient_u,
    magnetic_energy_gradient_u_virtual_work_analytic,
    contact_energy_gradient_u_sens,
)


def energy_gradient_u(
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
    lumen_query=None,
    use_magnetic: bool = True,
    use_contact: bool = True,
    contact: ContactParams | None = None,
) -> np.ndarray:
    """
    Compute dΠ/du for the discretised Cosserat beam.

    Contributions:
      - elastic strain gradient
      - optional magnetic virtual-work gradient
      - optional contact/lumen penalty gradient
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()
    K_seg = np.asarray(K_seg, float)
    u_star = np.asarray(u_star, float).reshape(3)

    if s.size < 2:
        raise ValueError(f"s must contain at least 2 nodes, got {s.size}.")

    expected = 3 * (s.size - 1)

    if u_flat.size != expected:
        raise ValueError(
            f"u_flat has size {u_flat.size}, expected {expected} for "
            f"{s.size} nodes."
        )

    if K_seg.shape != (s.size - 1, 3, 3):
        raise ValueError(
            f"K_seg must have shape ({s.size - 1}, 3, 3), got {K_seg.shape}."
        )

    grad = elastic_energy_gradient_u(
        u_flat,
        s=s,
        K_seg=K_seg,
        u_star=u_star,
    )

    grad = np.asarray(grad, float).reshape(-1)

    if grad.size != expected:
        raise ValueError(
            f"elastic gradient has size {grad.size}, expected {expected}."
        )

    if use_magnetic:
        grad_mag = magnetic_energy_gradient_u_virtual_work_analytic(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            m_src=m_src,
            r_src=r_src,
            m_local_fun=m_local_fun,
            m_moment=m_moment,
        )

        grad_mag = np.asarray(grad_mag, float).reshape(-1)

        if grad_mag.size != expected:
            raise ValueError(
                f"magnetic gradient has size {grad_mag.size}, expected {expected}."
            )

        grad += grad_mag

    if use_contact:
        if lumen_query is None:
            raise ValueError("use_contact=True requires lumen_query.")

        if contact is None:
            contact = ContactParams()

        contact.validate()

        grad_contact = contact_energy_gradient_u_sens(
            u_flat,
            p0=p0,
            q0=q0,
            s=s,
            lumen_query=lumen_query,
            contact=contact,
        )

        grad_contact = np.asarray(grad_contact, float).reshape(-1)

        if grad_contact.size != expected:
            raise ValueError(
                f"contact gradient has size {grad_contact.size}, expected {expected}."
            )

        grad += grad_contact

    if not np.all(np.isfinite(grad)):
        raise FloatingPointError("energy_gradient_u returned non-finite values.")

    return grad