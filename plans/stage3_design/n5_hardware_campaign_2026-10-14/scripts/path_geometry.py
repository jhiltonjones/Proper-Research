"""Generalized (contact- and no-contact-capable) path geometry helper
for the three-condition hardware comparison (plan: snug-bouncing-
giraffe.md). Extends the earlier single-plan path_interp_helper.py to
both plan roots, and adds a Cartesian (tip-space) projection alongside
the joint-space one, since physical tip progress is most directly a
tip-position quantity. No top-level execution -- safe to import.
"""
import numpy as np
import pandas as pd

DT = 0.1
VLIM6 = 0.10
INS_RATE_LIM = 2.0e-3
D7 = np.array([VLIM6] * 6 + [INS_RATE_LIM], dtype=float) * DT
D_Q = D7[:6]

_CACHE = {}


def _inverse_csv_path(plan_root):
    return plan_root.rstrip("/") + "/vessel_lumen/offline_inverse_configuration/inverse_configuration_path.csv"


def _resampled_csv_path(plan_root):
    return plan_root.rstrip("/") + "/time_parameterized_configuration_path/time_parameterized_configuration_path.csv"


def load_path_tables(plan_root):
    """Native (301-ish row) geometric path: s_m, q(6), L, tip(3). Cached
    per plan_root since this is read-only reference data reused by
    every tick of every script.
    """
    if plan_root in _CACHE:
        return _CACHE[plan_root]
    df = pd.read_csv(_inverse_csv_path(plan_root))
    tables = dict(
        s_m=df["s_m"].to_numpy(dtype=float),
        q=df[[f"q{i}_rad" for i in range(1, 7)]].to_numpy(dtype=float),
        L=df["insertion_m"].to_numpy(dtype=float),
        tip=df[["tip_x", "tip_y", "tip_z"]].to_numpy(dtype=float),
    )
    tables["s_min"] = float(tables["s_m"].min())
    tables["s_max"] = float(tables["s_m"].max())
    _CACHE[plan_root] = tables
    return tables


def load_resampled_s_of_ref_index(plan_root):
    """The time-resampled reference's own s at each ref_index (0-based,
    sample_index column) -- this is what ref_index actually advances
    through on the live controller, NOT the native 301-row path.
    """
    df = pd.read_csv(_resampled_csv_path(plan_root))
    df = df.sort_values("sample_index")
    return df["path_s_m"].to_numpy(dtype=float)


def _refine_local(dists_fn, s_candidates, lo, hi, n_refine=101):
    s_fine = np.linspace(lo, hi, n_refine)
    dists = dists_fn(s_fine)
    return float(s_fine[int(np.argmin(dists))])


def _global_project(dists_fn, s_min, s_max, n_samples=2001):
    s_candidates = np.linspace(s_min, s_max, n_samples)
    dists = dists_fn(s_candidates)
    i0 = int(np.argmin(dists))
    lo = s_candidates[max(i0 - 1, 0)]
    hi = s_candidates[min(i0 + 1, n_samples - 1)]
    return _refine_local(dists_fn, s_candidates, lo, hi)


def project_s_cartesian(tip_xyz_m, tables, n_samples=2001):
    """Global nearest-point search in tip (x,y,z) space -- the natural
    space for a *physical progress* claim, since it is the tip that
    physically advances. Returns s (m).
    """
    s_m, tip = tables["s_m"], tables["tip"]

    def dists_fn(s_query):
        tip_q = np.stack([np.interp(s_query, s_m, tip[:, j]) for j in range(3)], axis=-1)
        return np.linalg.norm(tip_q - np.asarray(tip_xyz_m)[None, :], axis=1)

    return _global_project(dists_fn, tables["s_min"], tables["s_max"], n_samples=n_samples)


def project_s_q(q_meas_rad, tables, n_samples=2001):
    """Global nearest-point search in D_Q-normalized joint space (same
    convention as contact_jacobian_hybrid.py / path_interp_helper.py).
    """
    s_m, q = tables["s_m"], tables["q"]

    def dists_fn(s_query):
        q_q = np.stack([np.interp(s_query, s_m, q[:, j]) for j in range(6)], axis=-1)
        return np.linalg.norm((q_q - np.asarray(q_meas_rad)[None, :]) / D_Q[None, :], axis=1)

    return _global_project(dists_fn, tables["s_min"], tables["s_max"], n_samples=n_samples)


def q_plan(s, tables):
    s = float(np.clip(s, tables["s_min"], tables["s_max"]))
    return np.array([np.interp(s, tables["s_m"], tables["q"][:, j]) for j in range(6)])


def L_plan(s, tables):
    s = float(np.clip(s, tables["s_min"], tables["s_max"]))
    return float(np.interp(s, tables["s_m"], tables["L"]))


def tip_plan(s, tables):
    s = float(np.clip(s, tables["s_min"], tables["s_max"]))
    return np.array([np.interp(s, tables["s_m"], tables["tip"][:, j]) for j in range(3)])


def tangent(s_q, tables, h=2.0e-4):
    """D_Q-normalized unit tangent at s_q -- same central-difference
    definition as ContactJacobianHybridModel.tangent (contact_jacobian_
    hybrid.py), generalized to any plan's tables so OLD-C/NC can use
    the identical r_q formula NEW's hybrid model already uses."""
    s_lo = max(tables["s_min"], s_q - h)
    s_hi = min(tables["s_max"], s_q + h)
    dq_ds = (q_plan(s_hi, tables) - q_plan(s_lo, tables)) / max(s_hi - s_lo, 1e-9)
    v = dq_ds / D_Q
    return v / max(float(np.linalg.norm(v)), 1e-12)


def r_q(q_meas_rad, s_q, tables):
    """Dimensionless, tangent-projected, D_Q-normalized off-path
    distance -- IDENTICAL formula to ContactJacobianHybridModel.r_q,
    generalized here so it can be computed for OLD-C/NC against their
    own plan's tables, not just NEW's contact-plan-only hybrid model."""
    tau = tangent(s_q, tables)
    raw = (np.asarray(q_meas_rad, dtype=float) - q_plan(s_q, tables)) / D_Q
    delta_q_perp = raw - tau * np.dot(tau, raw)
    return float(np.linalg.norm(delta_q_perp))
