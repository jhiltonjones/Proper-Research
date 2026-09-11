from __future__ import annotations

import numpy as np

from .contact import ContactParams, contact_energy_from_p
from .energy_optimized import _magnetic_energy_from_state
from .kinematics import integrate_pq_and_sens_from_u, integrate_pq_from_u
from .magnetism import _m_local_fun_is_field_dependent

from beam_direction_magnetisation.ana_energy import (
    elastic_energy_gradient_u,
    magnetic_energy_gradient_u_virtual_work_analytic,
)


def magnetic_energy_gradient_u_fd(
    u_flat: np.ndarray,
    *,
    p0: np.ndarray,
    q0: np.ndarray,
    s: np.ndarray,
    m_src: np.ndarray,
    r_src: np.ndarray,
    m_local_fun,
    m_moment: float,
    eps: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference dW_m/du_flat, for INDUCED (field-dependent) magnetisation.

    ``magnetic_energy_gradient_u_virtual_work_analytic`` differentiates p, q
    through the force/torque densities but treats those densities as fixed --
    correct only when m_local does not itself depend on the local field (the
    legacy fixed-moment models).  An induced m_local(s) = f(B(p(s))) adds a
    dm/dB * dB/dp * dp/du term the virtual-work formula does not include.
    Finite-differencing the true energy sidesteps re-deriving that term and is
    correct by construction; it is used automatically (see
    ``_m_local_fun_is_field_dependent``) only for induced-type m_local_fun,
    where the closed-form energy/gradient work is not yet done, so this is
    intentionally the safe-but-slower path, not the production-optimized one.
    """
    u_flat = np.asarray(u_flat, dtype=float).reshape(-1)

    def w_m(u: np.ndarray) -> float:
        p, q, _ = integrate_pq_from_u(u, p0=p0, q0=q0, s=s)
        w, _ = _magnetic_energy_from_state(
            p=p, q=q, s=s, m_src=m_src, r_src=r_src,
            m_local_fun=m_local_fun, m_moment=m_moment, detail="none",
        )
        return w

    grad = np.zeros_like(u_flat)
    for k in range(u_flat.size):
        up = u_flat.copy()
        um = u_flat.copy()
        up[k] += eps
        um[k] -= eps
        grad[k] = (w_m(up) - w_m(um)) / (2.0 * eps)
    return grad


def contact_energy_gradient_u_consistent(
    u_flat: np.ndarray,
    *,
    p0: np.ndarray,
    q0: np.ndarray,
    s: np.ndarray,
    lumen_query,
    contact: ContactParams | None = None,
) -> np.ndarray:
    """
    Differentiate the same discrete contact energy used by ``energy_from_u``.

    ``F_nodes`` is the physical force ``-dC/dp``.  Consequently,

        dW_contact/du = -sum_j w_j S_p[j]^T F_j.

    This includes both closest-point distance and tapered-radius interpolation
    derivatives through ``contact_energy_from_p``.
    """
    u_flat = np.asarray(u_flat, float).reshape(-1)
    s = np.asarray(s, float).ravel()

    p, _, S_p, _ = integrate_pq_and_sens_from_u(
        u_flat,
        p0=p0,
        q0=q0,
        s=s,
    )

    _, _, F_nodes, _, weights = contact_energy_from_p(
        p,
        s=s,
        lumen_query=lumen_query,
        contact=contact,
        return_force=True,
    )

    grad = -np.einsum(
        "j,ajn,aj->n",
        weights,
        S_p,
        F_nodes,
        optimize=True,
    )
    return np.asarray(grad, float).reshape(-1)


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
        if _m_local_fun_is_field_dependent(m_local_fun):
            # Induced/field-following magnetisation: the virtual-work formula
            # is missing a dm/dB*dB/dp*dp/du term for this model, so use the
            # FD gradient of the true energy instead (see its docstring).
            grad_mag = magnetic_energy_gradient_u_fd(
                u_flat,
                p0=p0,
                q0=q0,
                s=s,
                m_src=m_src,
                r_src=r_src,
                m_local_fun=m_local_fun,
                m_moment=m_moment,
            )
        else:
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

        grad_contact = contact_energy_gradient_u_consistent(
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
