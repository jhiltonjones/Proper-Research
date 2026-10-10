from __future__ import annotations

import numpy as np


def compute_discrete_path_curvature(path: np.ndarray) -> np.ndarray:
    """
    Estimate curvature along a discrete 3D centreline.

    Parameters
    ----------
    path:
        Array of shape (N, 3).

    Returns
    -------
    curvature:
        Array of shape (N,), units approximately 1/metre if path is in metres.
    """
    P = np.asarray(path, float)

    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError("path must have shape (N, 3) or larger.")

    P = P[:, :3]
    N = P.shape[0]

    if N < 3:
        return np.full(N, np.nan)

    # Segment vectors.
    dP = np.diff(P, axis=0)
    ds = np.linalg.norm(dP, axis=1)

    eps = 1e-12
    valid = ds > eps

    # Unit tangents on segments.
    T = np.zeros_like(dP)
    T[valid] = dP[valid] / ds[valid, None]

    # Change in tangent between adjacent segments.
    dT = np.diff(T, axis=0)

    # Distance between segment centres.
    ds_mid = 0.5 * (ds[:-1] + ds[1:])

    curvature_mid = np.full(N - 2, np.nan)
    valid_mid = ds_mid > eps

    curvature_mid[valid_mid] = (
        np.linalg.norm(dT[valid_mid], axis=1) / ds_mid[valid_mid]
    )

    # Map midpoint curvature back to path nodes.
    curvature = np.full(N, np.nan)
    curvature[1:-1] = curvature_mid

    # Fill endpoints.
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]

    return curvature


def nearest_path_index(path: np.ndarray, point: np.ndarray) -> int:
    """
    Return index of path point nearest to a query point.
    """
    P = np.asarray(path, float)[:, :3]
    q = np.asarray(point, float).reshape(-1)[:3]

    d = np.linalg.norm(P - q[None, :], axis=1)
    return int(np.nanargmin(d))


def curvature_ahead_metrics(
    *,
    path: np.ndarray,
    curvature: np.ndarray,
    tip_pos: np.ndarray,
    lookahead_points: int = 10,
) -> dict:
    """
    Compute local and lookahead curvature metrics near the current tip.

    Returns curvature in 1/m.
    """
    P = np.asarray(path, float)
    kappa = np.asarray(curvature, float).reshape(-1)

    if P.ndim != 2 or P.shape[0] != kappa.size:
        raise ValueError("path and curvature length mismatch.")

    i0 = nearest_path_index(P, tip_pos)
    i1 = min(i0 + int(lookahead_points), P.shape[0] - 1)

    window = kappa[i0 : i1 + 1]

    if window.size == 0:
        local = np.nan
        max_ahead = np.nan
        mean_ahead = np.nan
    else:
        local = float(kappa[i0])
        max_ahead = float(np.nanmax(window))
        mean_ahead = float(np.nanmean(window))

    return {
        "path_idx_nearest": int(i0),
        "curvature_local_1pm": local,
        "curvature_max_ahead_1pm": max_ahead,
        "curvature_mean_ahead_1pm": mean_ahead,
        "curvature_lookahead_points": int(lookahead_points),
    }