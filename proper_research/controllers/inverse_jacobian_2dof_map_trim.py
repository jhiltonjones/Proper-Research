"""2DOF nominal-feedforward + learned-map feedforward + non-integrating
feedback trim (2026-09-18).

Extends `inverse_jacobian_2dof_trim.TwoDOFTrimController` (frozen kp=0.6,
kn=0 development point) with a third, additive term: a path-indexed
feedforward correction for a repeatable beam-model discrepancy identified
offline on the rectangle open-loop-C data --

    d_k = p_meas,k - [p_nominal,k + J_k(z_meas,k - z_ref,k)]

was found to be highly repeatable across 5 open-loop repeats (repeatability
ratio R_d=0.922, leave-one-out map validation G_map=0.52-0.97 across the 5
held-out runs, mean 0.86) -- i.e. NOT random vibration/noise but a genuine,
predictable configuration-dependent mismatch between the nominal beam model
and the physical beam (gravity sag, magnetic loading, or an omitted state).
See `hardware/online/rectangle_stage_a/README.md` for the full chain
(advancer-excitation ring-down test -> ruled out -> delay/lag analysis ->
repeatability ratio -> leave-one-out map validation -> lambda=1000
curvature-regularized smoothing sweep) and where `dmap_final_lambda1000.npz`
came from.

Architecture (see the parent class for the nominal/trim split this keeps):

    q_nom_{k+1}  = q_nom_k + dt * u_ref,k                    (unchanged)
    dq_map,k+1   = -J_q,k+1^+ d_map,k+1                      (NEW: indexed
                                                                at the NEXT/
                                                                target index,
                                                                since it is
                                                                added to
                                                                q_nom_{k+1})
    e_k          = p_des,k - p_meas,k                         (unchanged --
                                                                the LIVE
                                                                measured
                                                                error; the
                                                                map is never
                                                                subtracted
                                                                from it. The
                                                                robot
                                                                physically
                                                                applies the
                                                                map, so the
                                                                camera
                                                                naturally
                                                                measures
                                                                whatever
                                                                residual is
                                                                left --
                                                                subtracting
                                                                the map here
                                                                too would
                                                                double-count
                                                                it)
    dz_task,k    = J_q,k^+ (kp * e_k)                          (unchanged)
    dz_null,k    = P_DLS,k (kn * (z_ref,k - z_meas,k))         (unchanged)
    dq_trim,k    = sat(dz_task,k + dz_null,k, +-q_trim_max)    (unchanged)

    q*_{k+1}     = q_nom_{k+1} + dq_map,k+1 + dq_trim,k

`dq_map` is recomputed fresh every tick from the CURRENT Jacobian at the
target index (non-integrating, exactly like `dz_task` -- an absolute trim
around the nominal trajectory, never accumulated tick over tick).

Uses the exact same DLS pseudoinverse convention as the parent class
(`Jq @ Jq.T + damping^2 I`, `damping=5e-2` by default) -- this was locked
during the offline replay specifically so the live map term matches what
was validated offline, not a fresh formulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .inverse_jacobian_2dof_trim import TwoDOFTrimConfig, TwoDOFTrimStep

Array = np.ndarray

__all__ = [
    "TwoDOFMapTrimConfig",
    "TwoDOFMapTrimController",
    "build_inv_2dof_map_trim_controller",
]


@dataclass
class TwoDOFMapTrimConfig(TwoDOFTrimConfig):
    pass


class TwoDOFMapTrimController:
    """See module docstring. `dmap` is a (N, 3) array (metres, [x, y, z] --
    z is 0 for this planar beam) indexed by reference sample index, e.g.
    loaded from `dmap_final_lambda1000.npz`'s `dmap` key."""

    name = "inv_2dof_map_trim"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        dmap: Array,
        map_schedule: Array,
        config: TwoDOFMapTrimConfig | None = None,
    ) -> None:
        self.reference = reference
        self.jacobian_provider = jacobian_provider
        self.dmap = np.asarray(dmap, dtype=float).reshape(-1, 3)
        # The map term's J_{q,k+1} MUST be the genuine per-reference-sample
        # LTV schedule (matching exactly what the offline leave-one-out
        # replay/lambda sweep used, schedule[idx][:, :6]) -- NOT a second
        # call to `jacobian_provider(state)`, which is evaluated at the
        # live measured state and is correct for the reactive task term but
        # is a different object from "the schedule's Jacobian at reference
        # sample k+1". Using the live-state Jacobian for the map term would
        # silently replay an untested variant of what was validated offline.
        self.map_schedule = np.asarray(map_schedule, dtype=float).reshape(-1, 3, 7)
        self.config = config or TwoDOFMapTrimConfig()
        self.q_nom: Array | None = None
        self.q_cmd6: Array | None = None
        self.log: dict[str, list] = {
            k: []
            for k in (
                "t_idx", "dq_task_norm", "dq_null_norm", "dq_map_norm",
                "dq_trim_norm_pre", "dq_trim_norm_post", "saturated_any",
                "rate_saturated_any", "q_nom6", "q_cmd6_mirror", "dL_null",
                "u_L_applied", "dq_task_vec", "dq_map_vec",
            )
        }

    def reset(self) -> None:
        self.q_nom = None
        self.q_cmd6 = None

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "kp": self.config.kp,
            "kn": self.config.kn,
            "damping": self.config.damping,
            "q_trim_max": self.config.q_trim_max,
            "dmap_rms_mm": float(1e3 * np.sqrt(np.mean(np.sum(self.dmap[:, :2] ** 2, axis=1)))),
        }

    def _map_at(self, idx: int) -> Array:
        idx = int(np.clip(idx, 0, self.dmap.shape[0] - 1))
        return self.dmap[idx]

    def solve(
        self,
        *,
        measured_state: Any,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> TwoDOFTrimStep:
        cfg = self.config
        idx = int(np.clip(control_index, 0, self.reference.sample_count - 1))
        idx_next = int(np.clip(idx + 1, 0, self.reference.sample_count - 1))
        state = np.asarray(measured_state, dtype=float).reshape(7)
        u_ref = np.asarray(self.reference.input[idx], dtype=float).reshape(7)

        if self.q_nom is None:
            self.q_nom = state.copy()
            self.q_cmd6 = state[:6].copy()

        self.q_nom = self.q_nom + cfg.dt * u_ref  # -> q_nom_{k+1}

        jac = np.asarray(self.jacobian_provider(state), dtype=float).reshape(3, 7)
        Jq = jac[:, :6]
        gram = Jq @ Jq.T + (cfg.damping ** 2) * np.eye(3)
        Jq_pinv = Jq.T @ np.linalg.solve(gram, np.eye(3))
        P = np.eye(6) - Jq_pinv @ Jq

        desired = np.asarray(self.reference.desired_position_m, dtype=float)[idx]
        measured_tip = np.asarray(measured_beam_position, dtype=float).reshape(3)
        e_k = desired - measured_tip           # LIVE measured error, map NOT subtracted
        dz_task = Jq_pinv @ (cfg.kp * e_k)

        z_ref = np.asarray(self.reference.state, dtype=float)[idx, :6]
        dz_null = P @ (cfg.kn * (z_ref - state[:6]))

        dq_trim_pre = dz_task + dz_null
        dq_trim = np.clip(dq_trim_pre, -cfg.q_trim_max, cfg.q_trim_max)
        saturated = bool(np.any(np.abs(dq_trim_pre) > cfg.q_trim_max))

        # Map term: indexed at the TARGET/next sample (added to q_nom_{k+1}),
        # using that sample's own SCHEDULE Jacobian -- not idx's, and not
        # the live-state jacobian_provider (see __init__ note). Non-
        # integrating, recomputed fresh every tick.
        Jq_next = self.map_schedule[idx_next][:, :6]
        gram_next = Jq_next @ Jq_next.T + (cfg.damping ** 2) * np.eye(3)
        Jq_next_pinv = Jq_next.T @ np.linalg.solve(gram_next, np.eye(3))
        d_map_next = self._map_at(idx_next)
        dq_map = -(Jq_next_pinv @ d_map_next)

        q_star6 = self.q_nom[:6] + dq_map + dq_trim
        step6_raw = q_star6 - self.q_cmd6
        step6 = np.clip(step6_raw, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
        rate_saturated = bool(np.any(np.abs(step6_raw) > cfg.max_joint_step_rad))
        self.q_cmd6 = self.q_cmd6 + step6
        command6 = step6 / cfg.dt

        command = np.concatenate([command6, [u_ref[6]]])

        if cfg.enable_logging:
            self.log["t_idx"].append(idx)
            self.log["dq_task_vec"].append(dz_task.copy())
            self.log["dq_map_vec"].append(dq_map.copy())
            self.log["dq_task_norm"].append(float(np.linalg.norm(dz_task)))
            self.log["dq_null_norm"].append(float(np.linalg.norm(dz_null)))
            self.log["dq_map_norm"].append(float(np.linalg.norm(dq_map)))
            self.log["dq_trim_norm_pre"].append(float(np.linalg.norm(dq_trim_pre)))
            self.log["dq_trim_norm_post"].append(float(np.linalg.norm(dq_trim)))
            self.log["saturated_any"].append(saturated)
            self.log["rate_saturated_any"].append(rate_saturated)
            self.log["q_nom6"].append(self.q_nom[:6].copy())
            self.log["q_cmd6_mirror"].append(self.q_cmd6.copy())
            self.log["dL_null"].append(0.0)
            self.log["u_L_applied"].append(float(u_ref[6]))

        return TwoDOFTrimStep(
            command=command,
            planned_input=u_ref.copy(),
            info={
                "status": "inv_2dof_map_trim", "success": True,
                "dq_task_norm": float(np.linalg.norm(dz_task)),
                "dq_map_norm": float(np.linalg.norm(dq_map)),
            },
        )


def build_inv_2dof_map_trim_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    dmap: Array,
    map_schedule: Array,
    kp: float = 0.6,
    kn: float = 0.0,
    damping: float = 5.0e-2,
    q_trim_max: float = 0.03,
    dt: float = 0.1,
    max_joint_step_rad: float = 0.010,
    enable_logging: bool = False,
) -> TwoDOFMapTrimController:
    return TwoDOFMapTrimController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        dmap=dmap,
        map_schedule=map_schedule,
        config=TwoDOFMapTrimConfig(
            kp=kp, kn=kn, damping=damping, q_trim_max=q_trim_max,
            dt=dt, max_joint_step_rad=max_joint_step_rad,
            enable_logging=enable_logging,
        ),
    )
