"""Smooth validity-gated J1/J3 contact-Jacobian hybrid for live use.

Offline investigation (2026-10-12..14, see STAGE3_FINAL_REPORT.md and this
session's plan history) established:
  - A trajectory-specific 2D state-conditioned contact-Jacobian map,
    J3(s_q, DeltaL), substantially improves the MPC's local model at
    identified contact-specific transients (ticks 278/366 in the
    floor240 zeroR validation), but its accuracy degrades as the
    measured robot configuration departs from the nominal configuration
    trajectory -- quantified by the dimensionless, tangent-projected
    off-path distance r_q.
  - A supervised search for a single additional configuration coordinate
    did not generalize across runs (mean held-out R^2=0.073) -- no 3D
    map was built.
  - Cheap schedule reindexing (J2) was tested as an alternative fallback
    and found no generalizing benefit (49.4% win rate in the fallback
    zone, no leave-one-run-out gate).
  - The supported practical architecture is therefore: use J3 where
    validated, smoothly blend back to the ORIGINAL SCHEDULED JACOBIAN
    (J1) outside that region, with thresholds r_a=1.79, r_b=11.16
    (dimensionless D_Q-normalized units) and a cubic smoothstep
    transition (27x smaller worst-case per-step Jacobian change than a
    hard switch).

J4, used throughout the offline investigation, refers to the full
contact-model Jacobian recomputed at the measured state -- an OFFLINE
MODEL ORACLE used to validate this hybrid, never a physical measurement.
This module implements J1/J3/blend for live use; it does not recompute
J4 (too expensive at control rate -- that is exactly why this map
exists).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from proper_research.planning.time_parameterized_configuration_path import (
    load_saved_inverse_configuration_path,
)

Array = np.ndarray


@dataclass(frozen=True)
class HybridMapConfig:
    map_dir: str
    manifest_file: str = "contact_jacobian_map_manifest.json"
    thresholds_file: str = "residual_decomp_thresholds.json"
    inverse_output_dir: str | None = None  # defaults to manifest's plan_dir if None


class ContactJacobianHybridModel:
    """Loads the frozen tiles + manifest once; answers per-tick and
    per-horizon (J, nominal_position) queries cheaply (interpolation
    only, no live contact-model solve).
    """

    def __init__(self, config: HybridMapConfig, *, D: Array) -> None:
        self.D = np.asarray(D, dtype=float).reshape(7)
        self.D_Q = self.D[:6]
        map_dir = Path(config.map_dir)
        manifest_path = map_dir / config.manifest_file
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Contact-Jacobian hybrid map manifest not found at {manifest_path}. "
                "This is a hard failure -- there is no silent fallback to the plain "
                "schedule for a map explicitly requested via --jacobian-source hybrid_j1j3."
            )
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self.tiles: list[dict[str, Any]] = []
        for tile_meta in self.manifest["tiles"]:
            npz_path = map_dir / tile_meta["file"]
            if not npz_path.exists():
                raise FileNotFoundError(f"Tile file listed in manifest not found: {npz_path}")
            data = np.load(npz_path)
            if "tip_m" not in data:
                raise FileNotFoundError(
                    f"Tile {tile_meta['name']!r} at {npz_path} has not been augmented with "
                    "nominal tip positions (run augment_map_with_nominal_positions.py first) -- "
                    "refusing to load an incomplete map."
                )
            self.tiles.append(dict(
                name=tile_meta["name"], s_grid_m=data["s_grid_m"], dl_grid_m=data["dl_grid_m"],
                J_grid=data["J_grid"], tip_grid=data["tip_m"],
                max_interp_error=tile_meta.get("max_interp_error"),
            ))

        thresholds_path = map_dir / config.thresholds_file
        with open(thresholds_path) as f:
            thresholds = json.load(f)
        self.r_a = float(thresholds["r_a_conservative"])
        self.r_b = float(thresholds["r_b_conservative"])
        if not (0.0 <= self.r_a < self.r_b):
            raise ValueError(f"Invalid hybrid thresholds r_a={self.r_a}, r_b={self.r_b}.")

        inverse_dir = config.inverse_output_dir
        if inverse_dir is None:
            plan_dir = self.manifest["plan_dir"]
            # plan_dir points at .../<plan_root>/time_parameterized_configuration_path;
            # the inverse-configuration CSV lives at <plan_root>/vessel_lumen/offline_inverse_configuration
            plan_root = os.path.dirname(plan_dir.rstrip("/"))
            inverse_dir = os.path.join(
                "/home/jack/Proper-Research", plan_root, "vessel_lumen", "offline_inverse_configuration"
            )
        path = load_saved_inverse_configuration_path(inverse_dir)
        self.s_m_arr = np.asarray(path.s_m, dtype=float)
        self.states_arr = np.asarray(path.states, dtype=float)  # (N,7): q1..q6, L
        self.s_min, self.s_max = float(self.s_m_arr.min()), float(self.s_m_arr.max())

        # 2026-10-14 speed fix (see root-cause-decomposition-discipline
        # memory / snug-bouncing-giraffe.md): project_s_q's global search
        # used to call q_plan() -- itself 6 scalar np.interp calls -- once
        # per candidate, 2001+101 times per tick (~12.6k np.interp calls,
        # measured at 25-45ms per call in an offline, contention-free,
        # same-machine A/B replay -- the dominant cost by nearly two
        # orders of magnitude over everything else in the hybrid+QP path
        # combined). _q_plan_batch replaces that with 6 VECTORIZED
        # np.interp calls (one per joint, over the whole candidate array
        # at once) -- same two-stage global+local-refine algorithm, same
        # distance metric, same values, just not re-looped in Python. The
        # n_samples=2001 global-stage candidate grid never changes after
        # construction, so it is precomputed once here instead of once
        # per tick.
        self._s_dense_2001 = np.linspace(self.s_min, self.s_max, 2001)
        self._q_dense_2001 = self._q_plan_batch(self._s_dense_2001)

    # ---- continuous path interpolation -----------------------------
    def q_plan(self, s: float) -> Array:
        s = float(np.clip(s, self.s_min, self.s_max))
        return np.array([np.interp(s, self.s_m_arr, self.states_arr[:, j]) for j in range(6)])

    def _q_plan_batch(self, s_array: Array) -> Array:
        """Vectorized q_plan over an array of s values: one np.interp call
        per joint (6 total), not one per (s, joint) pair -- identical
        values to np.stack([self.q_plan(s) for s in s_array], axis=0)."""
        s_clipped = np.clip(np.asarray(s_array, dtype=float), self.s_min, self.s_max)
        return np.stack(
            [np.interp(s_clipped, self.s_m_arr, self.states_arr[:, j]) for j in range(6)], axis=1
        )

    def L_plan(self, s: float) -> float:
        s = float(np.clip(s, self.s_min, self.s_max))
        return float(np.interp(s, self.s_m_arr, self.states_arr[:, 6]))

    def project_s_q(self, q_meas: Array, n_samples: int = 2001) -> float:
        q_meas = np.asarray(q_meas, dtype=float)
        if n_samples == 2001:
            s_candidates, q_candidates = self._s_dense_2001, self._q_dense_2001
        else:
            s_candidates = np.linspace(self.s_min, self.s_max, n_samples)
            q_candidates = self._q_plan_batch(s_candidates)
        dists = np.linalg.norm((q_candidates - q_meas[None, :]) / self.D_Q[None, :], axis=1)
        i0 = int(np.argmin(dists))
        lo = s_candidates[max(i0 - 1, 0)]
        hi = s_candidates[min(i0 + 1, n_samples - 1)]
        s_fine = np.linspace(lo, hi, 101)
        q_fine = self._q_plan_batch(s_fine)
        dists_fine = np.linalg.norm((q_fine - q_meas[None, :]) / self.D_Q[None, :], axis=1)
        return float(s_fine[int(np.argmin(dists_fine))])

    def tangent(self, s_q: float, h: float = 2.0e-4) -> Array:
        s_lo, s_hi = max(self.s_min, s_q - h), min(self.s_max, s_q + h)
        dq_ds = (self.q_plan(s_hi) - self.q_plan(s_lo)) / max(s_hi - s_lo, 1e-9)
        v = dq_ds / self.D_Q
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def r_q(self, q_meas: Array, s_q: float) -> float:
        tau = self.tangent(s_q)
        raw = (q_meas - self.q_plan(s_q)) / self.D_Q
        delta_q_perp = raw - tau * np.dot(tau, raw)
        return float(np.linalg.norm(delta_q_perp))

    # ---- tile lookup --------------------------------------------------
    @staticmethod
    def _bilinear(tile: dict[str, Any], s_q: float, dl: float) -> tuple[Array | None, Array | None]:
        s_grid, dl_grid = tile["s_grid_m"], tile["dl_grid_m"]
        if s_q < s_grid[0] or s_q > s_grid[-1] or dl < dl_grid[0] or dl > dl_grid[-1]:
            return None, None
        i1 = int(np.clip(np.searchsorted(s_grid, s_q, side="right"), 1, len(s_grid) - 1))
        i0 = i1 - 1
        j1 = int(np.clip(np.searchsorted(dl_grid, dl, side="right"), 1, len(dl_grid) - 1))
        j0 = j1 - 1
        s0, s1, d0, d1 = s_grid[i0], s_grid[i1], dl_grid[j0], dl_grid[j1]
        ts = 0.0 if s1 == s0 else (s_q - s0) / (s1 - s0)
        td = 0.0 if d1 == d0 else (dl - d0) / (d1 - d0)
        w00, w01, w10, w11 = (1 - ts) * (1 - td), (1 - ts) * td, ts * (1 - td), ts * td
        J = (w00 * tile["J_grid"][i0, j0] + w01 * tile["J_grid"][i0, j1] +
             w10 * tile["J_grid"][i1, j0] + w11 * tile["J_grid"][i1, j1])
        tip = (w00 * tile["tip_grid"][i0, j0] + w01 * tile["tip_grid"][i0, j1] +
               w10 * tile["tip_grid"][i1, j0] + w11 * tile["tip_grid"][i1, j1])
        return J, tip

    def lookup_j3(self, s_q: float, dl: float) -> tuple[Array | None, Array | None, bool, dict]:
        for tile in self.tiles:
            J, tip = self._bilinear(tile, s_q, dl)
            if J is not None:
                quality = dict(tile=tile["name"], max_interp_error=tile["max_interp_error"])
                return J, tip, True, quality
        return None, None, False, dict(tile=None, max_interp_error=None)

    @staticmethod
    def smoothstep_weight(r_q: float, r_a: float, r_b: float) -> float:
        if r_q <= r_a:
            return 1.0
        if r_q >= r_b:
            return 0.0
        t = (r_q - r_a) / (r_b - r_a)
        return 1.0 - 3.0 * t ** 2 + 2.0 * t ** 3

    # ---- live per-tick / per-horizon API -----------------------------
    def horizon_jacobians_and_nominals(
        self, *, q_meas: Array, L_meas: float,
        sched_J_window: Array, sched_nominal_window: Array,
        s_ref_window: Array,
    ) -> tuple[Array, Array, dict]:
        """Returns (J_horizon, nominal_position_horizon, diagnostics).

        sched_J_window: (H,3,7) -- J1 at each horizon step (the schedule's
            own rows, already what the controller reads today).
        sched_nominal_window: (H,3) -- the schedule's own nominal tip
            position at each horizon step (== desired_position at those
            indices, matching today's implicit default).
        s_ref_window: (H,) -- the reference path's own s at each horizon
            step (s_ref_window[0] is "now").

        Horizon rule (validated offline): hold DeltaL_k and r_q_k fixed
        across the horizon; advance query path-position by the
        reference's own future increments. Returns J1 unchanged
        (bit-identical) whenever the map is invalid or r_q>=r_b -- the
        no-op/backward-compatibility guarantee.
        """
        s_q_k = self.project_s_q(q_meas)
        delta_L_k = float(L_meas) - self.L_plan(s_q_k)
        r_q_k = self.r_q(q_meas, s_q_k)
        w = self.smoothstep_weight(r_q_k, self.r_a, self.r_b)

        H = sched_J_window.shape[0]
        J_horizon = sched_J_window.copy()
        nominal_horizon = sched_nominal_window.copy()
        map_valid_any = False

        if w > 0.0:
            s_ref_k = float(s_ref_window[0])
            for h in range(H):
                s_query_h = float(np.clip(s_q_k + (s_ref_window[h] - s_ref_k), self.s_min, self.s_max))
                J3_h, tip3_h, valid3, _ = self.lookup_j3(s_query_h, delta_L_k)
                if not valid3:
                    continue  # this horizon step stays J1 (already copied above)
                map_valid_any = True
                x0_h = np.concatenate([self.q_plan(s_query_h), [self.L_plan(s_query_h) + delta_L_k]])
                state_ref_h = np.concatenate([self.q_plan(s_ref_window[h]), [self.L_plan(s_ref_window[h])]])
                # nominal_position correction for a linearization point x0_h != state_reference_h:
                # f(x0) + J(x0) @ (state_reference - x0)
                nominal3_h = tip3_h + J3_h @ (state_ref_h - x0_h)
                J_horizon[h] = w * J3_h + (1 - w) * sched_J_window[h]
                nominal_horizon[h] = w * nominal3_h + (1 - w) * sched_nominal_window[h]

        diagnostics = dict(s_q=s_q_k, delta_L=delta_L_k, r_q=r_q_k, weight=w, map_valid_any=map_valid_any)
        return J_horizon, nominal_horizon, diagnostics
