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
        """
        Return distance, interpolated radius, and closest centreline point.

        Use :meth:`closest_with_gradients` when differentiating a contact
        energy.  In a tapered lumen the interpolated radius changes as the
        closest point moves along a segment, so the distance normal alone is
        not the complete derivative.
        """
        delta, Rloc, q, _, _ = self.closest_with_gradients(p, window=window)
        return delta, Rloc, q

    def closest_with_gradients(
        self,
        p: np.ndarray,
        window: int | None = None,
    ):
        """
        Return the active-segment closest-point query and its local gradients.

        Returns:
            delta, Rloc, q, grad_delta, grad_Rloc

        The derivatives are exact while the same polyline segment remains
        active.  At an interior projection,

            grad_Rloc = (R[i+1] - R[i]) * (C[i+1] - C[i]) / ||C[i+1]-C[i]||^2.

        At a clamped endpoint the interpolation parameter is locally constant.
        Segment switches and exact centreline points remain nonsmooth, as is
        inherent in a closest-point polyline contact model.
        """
        result = self.closest_many_with_gradients(
            np.asarray(p, float).reshape(1, 3),
            window=window,
        )
        return (
            float(result[0][0]),
            float(result[1][0]),
            result[2][0].copy(),
            result[3][0].copy(),
            result[4][0].copy(),
        )

    def closest_many_with_gradients(
        self,
        points: np.ndarray,
        window: int | None = None,
    ):
        """Vectorized active-segment query for an array of world points."""
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points must have shape (N, 3), got {points.shape}."
            )

        n_points = points.shape[0]
        n_segments = self.C.shape[0] - 1
        if window is None:
            candidates = np.broadcast_to(
                np.arange(n_segments, dtype=int),
                (n_points, n_segments),
            )
        else:
            _, nearest = self.kdt.query(points)
            offsets = np.arange(-int(window), int(window) + 1)
            candidates = np.clip(
                np.asarray(nearest, int)[:, None] + offsets[None, :],
                0,
                n_segments - 1,
            )

        a = self.C[candidates]
        ab = self.C[candidates + 1] - a
        denom = np.einsum("nkj,nkj->nk", ab, ab)
        safe_denom = np.maximum(denom, 1e-18)
        ap = points[:, None, :] - a
        t_raw = np.einsum("nkj,nkj->nk", ap, ab) / safe_denom
        t = np.clip(t_raw, 0.0, 1.0)
        q_all = a + t[:, :, None] * ab
        offsets = points[:, None, :] - q_all
        distance2 = np.einsum("nkj,nkj->nk", offsets, offsets)

        local_choice = np.argmin(distance2, axis=1)
        rows = np.arange(n_points)
        active = candidates[rows, local_choice]
        q = q_all[rows, local_choice]
        offset = points - q
        delta = np.sqrt(distance2[rows, local_choice])
        t_active = t[rows, local_choice]
        t_raw_active = t_raw[rows, local_choice]
        ab_active = ab[rows, local_choice]
        denom_active = denom[rows, local_choice]

        Rloc = (
            (1.0 - t_active) * self.R[active]
            + t_active * self.R[active + 1]
        )
        grad_delta = np.zeros_like(points)
        away = delta > 1e-12
        grad_delta[away] = offset[away] / delta[away, None]

        grad_Rloc = np.zeros_like(points)
        interior = (
            (denom_active >= 1e-18)
            & (t_raw_active > 0.0)
            & (t_raw_active < 1.0)
        )
        radius_delta = self.R[active + 1] - self.R[active]
        grad_Rloc[interior] = (
            radius_delta[interior, None]
            * ab_active[interior]
            / denom_active[interior, None]
        )
        return delta, Rloc, q, grad_delta, grad_Rloc


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

    w = node_quadrature_weights(s)
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

    delta, Rloc, _, grad_delta, grad_Rloc = (
        lumen_query.closest_many_with_gradients(p.T, window=window)
    )
    grad_phi = grad_delta - grad_Rloc
    gap_arr[:] = Rloc - delta - r_beam
    phi = -gap_arr

    if smooth:
        root = np.sqrt(phi * phi + smooth_eps * smooth_eps)
        phi_pos = 0.5 * (phi + root)
        dphi_pos_dphi = 0.5 * (1.0 + phi / root)
        C[:] = 0.5 * k_contact * phi_pos**2
        force_mag = k_contact * phi_pos * dphi_pos_dphi
        F[:, :] = -(force_mag[:, None] * grad_phi).T
        return C, F, gap_arr

    soft = (phi > 0.0) & (phi <= pen_switch)
    hard = phi > pen_switch
    C[soft] = 0.5 * k_contact * phi[soft] ** 2
    F[:, soft] = -(
        (k_contact * phi[soft])[:, None] * grad_phi[soft]
    ).T

    dp = phi[hard] - pen_switch
    C0 = 0.5 * k_contact * pen_switch**2
    F0 = k_contact * pen_switch
    C[hard] = C0 + F0 * dp + 0.5 * k_hard * dp**2
    F[:, hard] = -(
        (F0 + k_hard * dp)[:, None] * grad_phi[hard]
    ).T

    if debug and np.any((gap_arr < 0.0) & (phi <= 0.0)):
        print("[CONTACT DBG] impossible branch: gap < 0 but phi <= 0")

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


def node_quadrature_weights(s: np.ndarray) -> np.ndarray:
    """
    Trapezoidal node weights for an integral sampled at beam nodes.

    The weights sum to ``s[-1] - s[0]`` for both uniform and nonuniform grids.
    """
    s = np.asarray(s, float).ravel()

    if s.size < 2:
        raise ValueError("s must contain at least two nodes.")

    ds = np.diff(s)
    if not np.all(ds > 0):
        raise ValueError("s must be strictly increasing.")

    w = np.empty(s.size, dtype=float)
    w[0] = 0.5 * ds[0]
    w[-1] = 0.5 * ds[-1]

    if s.size > 2:
        w[1:-1] = 0.5 * (ds[:-1] + ds[1:])

    return w
