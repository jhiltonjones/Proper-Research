from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactParams:
    r_beam: float = 0.001
    k: float = 1e8
    pen_switch: float = 5e-5
    k_hard: float = 1e10
    smooth: bool = True
    smooth_eps: float = 1e-5
    window: int | None = 3

    def validate(self) -> None:
        if self.r_beam <= 0:
            raise ValueError(f"r_beam must be positive, got {self.r_beam}.")
        if self.k < 0:
            raise ValueError(f"k must be non-negative, got {self.k}.")
        if self.pen_switch < 0:
            raise ValueError(f"pen_switch must be non-negative, got {self.pen_switch}.")
        if self.k_hard < 0:
            raise ValueError(f"k_hard must be non-negative, got {self.k_hard}.")
        if self.smooth_eps <= 0:
            raise ValueError(f"smooth_eps must be positive, got {self.smooth_eps}.")
        if self.window is not None and self.window < 1:
            raise ValueError(f"window must be >= 1 or None, got {self.window}.")


class LumenQuery:
    def __init__(self, C: np.ndarray, R: np.ndarray):
        self.C = np.asarray(C, float)
        self.R = np.asarray(R, float).reshape(-1)

        if self.C.ndim != 2 or self.C.shape[1] != 3:
            raise ValueError(f"C must have shape (M, 3), got {self.C.shape}.")
        if self.C.shape[0] < 2:
            raise ValueError("C must contain at least two centreline points.")
        if self.R.size != self.C.shape[0]:
            raise ValueError(
                f"R length {self.R.size} does not match C length {self.C.shape[0]}."
            )
        if np.any(self.R <= 0):
            raise ValueError("All lumen radii must be positive.")

        self.kdt = cKDTree(self.C)

    def closest(self, p: np.ndarray, window: int | None = None):
        p = np.asarray(p, float).reshape(3)

        dmin = np.inf
        best = None

        if window is None:
            i0 = 0
            i1 = len(self.C) - 2
        else:
            _, k = self.kdt.query(p)
            k = int(k)
            i0 = max(0, k - int(window))
            i1 = min(len(self.C) - 2, k + int(window))

        for i in range(i0, i1 + 1):
            q, t = closest_point_on_segment(p, self.C[i], self.C[i + 1])
            d = float(np.linalg.norm(p - q))

            if d < dmin:
                dmin = d
                best = (i, t, q)

        if best is None:
            raise RuntimeError("Failed to find closest lumen segment.")

        i, t, q = best
        Rloc = (1.0 - t) * self.R[i] + t * self.R[i + 1]

        return float(dmin), float(Rloc), q


def closest_point_on_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float]:
    p = np.asarray(p, float).reshape(3)
    a = np.asarray(a, float).reshape(3)
    b = np.asarray(b, float).reshape(3)

    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom < 1e-18:
        return a.copy(), 0.0

    t = float(np.dot(p - a, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    q = a + t * ab

    return q, t


def contact_energy_from_p(
    p: np.ndarray,
    *,
    s: np.ndarray,
    lumen_query: LumenQuery,
    contact: ContactParams | None = None,
    return_force: bool = False,
):
    contact = contact or ContactParams()
    contact.validate()

    p = np.asarray(p, float)
    s = np.asarray(s, float).ravel()

    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"p must have shape (3, N), got {p.shape}.")
    if s.size != p.shape[1]:
        raise ValueError(f"s length {s.size} does not match p nodes {p.shape[1]}.")

    C_nodes, F_nodes, gap_nodes = contact_barrier_energy_and_force_fast(
        p,
        lumen_query,
        r_beam=contact.r_beam,
        k_contact=contact.k,
        pen_switch=contact.pen_switch,
        k_hard=contact.k_hard,
        eps=1e-12,
        window=contact.window,
        smooth=contact.smooth,
        smooth_eps=contact.smooth_eps,
    )

    w = uniform_node_weights_like_current_energy(s)
    W_cf = float(np.dot(w, C_nodes))

    if not return_force:
        return W_cf

    return W_cf, C_nodes, F_nodes, gap_nodes, w


def contact_barrier_energy_and_force_fast(
    p: np.ndarray,
    lumen_query: LumenQuery,
    *,
    r_beam: float = 0.0,
    k_contact: float = 3e5,
    pen_switch: float = 5e-4,
    k_hard: float = 3e5,
    eps: float = 1e-12,
    window: int | None = 3,
    smooth: bool = False,
    smooth_eps: float = 1e-7,
    debug: bool = False,
):
    p = np.asarray(p, float)

    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"p must have shape (3, N), got {p.shape}.")

    N = p.shape[1]

    C = np.zeros(N, dtype=float)
    F = np.zeros((3, N), dtype=float)
    gap_arr = np.zeros(N, dtype=float)

    for j in range(N):
        x = p[:, j]

        delta, Rloc, q_closest = lumen_query.closest(x, window=window)

        if delta > eps:
            n = (x - q_closest) / delta
        else:
            n = np.array([1.0, 0.0, 0.0], dtype=float)

        gap = Rloc - delta - r_beam
        phi = -gap

        gap_arr[j] = gap

        if smooth:
            root = np.sqrt(phi * phi + smooth_eps * smooth_eps)
            phi_pos = 0.5 * (phi + root)
            dphi_pos_dphi = 0.5 * (1.0 + phi / root)

            C[j] = 0.5 * k_contact * phi_pos**2
            F[:, j] = -(k_contact * phi_pos * dphi_pos_dphi) * n
            continue

        if phi <= 0.0:
            if debug and gap < 0.0:
                print("[CONTACT DBG] impossible branch: gap < 0 but phi <= 0")
            continue

        if phi <= pen_switch:
            C[j] = 0.5 * k_contact * phi**2
            F[:, j] = -(k_contact * phi) * n
        else:
            dp = phi - pen_switch
            C0 = 0.5 * k_contact * pen_switch**2
            F0 = k_contact * pen_switch

            C[j] = C0 + F0 * dp + 0.5 * k_hard * dp**2
            F[:, j] = -(F0 + k_hard * dp) * n

    return C, F, gap_arr


def uniform_node_weights_like_current_energy(s: np.ndarray) -> np.ndarray:
    """
    Match your current contact-energy convention:

        W_cf = h * sum(C_nodes)

    where h is the uniform segment spacing.

    Note:
        This is not trapezoidal node quadrature. It intentionally matches your
        existing energy convention.
    """
    s = np.asarray(s, float).ravel()

    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    ds = np.diff(s)

    if not np.all(ds > 0):
        raise ValueError("s must be strictly increasing.")

    if not np.allclose(ds, ds[0], rtol=1e-8, atol=1e-12):
        raise ValueError(
            "uniform_node_weights_like_current_energy assumes uniform s. "
            "Use node_quadrature_weights(s) for nonuniform grids."
        )

    h = float(ds[0])
    return np.full(s.size, h, dtype=float)