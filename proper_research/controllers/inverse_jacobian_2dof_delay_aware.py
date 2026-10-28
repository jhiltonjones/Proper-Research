"""2DOF nominal-feedforward + delay-aware feedback trim (2026-09-18).

Companion to `inverse_jacobian_2dof_trim.py` (frozen kp=0.6, kn=0 dev
point), not a replacement -- this is a clean, single-variable ablation:
delay-awareness only, same gain, same architecture otherwise.

Motivation: the execution-delay identification (see
`hardware/online/rectangle_stage_a/README.md` for the full chain) found

    q_meas_k  ~=  q_cmd_{k-2}          (d=2, validated by 5-fold LOO,
                                         G_delay=0.945-0.986 on every fold;
                                         a fractional 2.16-sample model did
                                         NOT generalize -- G_frac was
                                         inconsistent/negative on 3 of 5
                                         held-out folds, so d=2 is frozen)

Combined with a unit-verified indexing fact: a DECISION made at tick k
(which produces q_cmd_{k+1}) leaves q_meas unchanged at k, k+1 AND k+2, and
first shows up at k+3 -- not k+2. (Verified with a synthetic accumulator +
perfect-delay q_meas and a one-tick perturbation to u_k0: q_meas diff is
exactly zero at k0, k0+1, k0+2, and exactly dt*delta at k0+3.) So a
correction computed now should target where the beam will be when q_cmd_k+1
actually becomes physical -- stage k+3 -- not the current stage.

Architecture
------------
Nominal channel is ALSO delay-compensated (previewed 3 samples ahead), not
just the feedback -- otherwise the feedforward trajectory would stay 0.2s
late while only the correction is delay-aware, which would misrepresent
where the "nominal" motion is supposed to be at command-realization time::

    q_nom,D_{k+1}[:6]  = z_ref,k+3[:6]           (direct preview of the
                                                    plan's own joint target
                                                    at the realization
                                                    index -- "either
                                                    directly from q_ref, or
                                                    equivalently via the
                                                    accumulator with a
                                                    previewed u_ref";
                                                    simplest is the former)
    q_nom[6]           = q_nom[6] + dt*u_ref,k[6] (insertion: UNDELAYED,
                                                    same persistent
                                                    accumulator as the base
                                                    controller -- the
                                                    validated d=2 delay is
                                                    specific to the six UR
                                                    joints, not the
                                                    stepper/advancer)

Current model residual (measurement-based, no schedule needed -- uses the
live per-state Jacobian, same source the reactive term already uses)::

    d_k = p_meas,k - [p_nominal,k + J_k(z_meas,k - z_ref,k)]

assumed to persist over the short execution delay: `d_hat_{k+3|k} = d_k`.

Baseline (no-new-correction) physical prediction at the realization index
r=k+3 -- since q_phys_{k+3} = q_cmd_{k+1} (delay d=2), and q_cmd_{k+1} IS
exactly what this tick is about to decide, "without this tick's new
correction" means q_cmd_{k+1} would equal q_nom,D_{k+1} alone::

    z_hat_base_{k+3|k}  = [q_nom,D_{k+1}[:6],  L preview at r]
    p_hat_base_{k+3|k}  = p_nominal,r + J_r(z_hat_base_{k+3|k} - z_ref,r) + d_k

    e_pred,k    = p_des,r - p_hat_base_{k+3|k}
    dq_task,k   = J_q,r^+ (kp * e_pred,k)          (kp=0.6, same DLS as
                                                      the base controller,
                                                      damping=5e-2)

    q*_{k+1}    = q_nom,D_{k+1} + dq_task,k         (non-integrating trim,
                                                      no null term (kn=0
                                                      fixed), no map)

Execution-C accumulator update (q_cmd6) is unchanged from the base
controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .inverse_jacobian_2dof_trim import TwoDOFTrimStep

Array = np.ndarray

__all__ = [
    "TwoDOFDelayAwareConfig",
    "TwoDOFDelayAwareController",
    "build_inv_2dof_delay_aware_controller",
]


@dataclass
class TwoDOFDelayAwareConfig:
    kp: float = 0.6
    kn: float = 0.0                 # kept for interface parity; unused (locked 0)
    damping: float = 5.0e-2
    q_trim_max: float = 0.03
    dt: float = 0.1
    max_joint_step_rad: float = 0.010
    delay_samples: int = 2          # validated d=2 (see module docstring)
    enable_logging: bool = False


@dataclass
class TwoDOFDelayAwareStep(TwoDOFTrimStep):
    pass


class TwoDOFDelayAwareController:
    """See module docstring. `r = control_index + 1 + delay_samples` is the
    realization stage (k+3 at delay_samples=2) -- the ONE line encoding the
    unit-tested causal fact this controller is built around."""

    name = "inv_2dof_delay_aware"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        config: TwoDOFDelayAwareConfig | None = None,
        schedule: Array | None = None,
    ) -> None:
        self.reference = reference
        self.jacobian_provider = jacobian_provider
        # See TwoDOFTrimController.__init__'s docstring note -- same
        # additive path-indexed-schedule option, for the live A/B against
        # it: baseline reads schedule[idx], this controller schedule[idx_r]
        # (idx+3), so the two conditions differ ONLY in which index they
        # read, not also in Jacobian source.
        self.schedule = None if schedule is None else np.asarray(schedule, dtype=float)
        self.config = config or TwoDOFDelayAwareConfig()
        self.q_nom: Array | None = None     # persistent 7D (insertion only actually used from it)
        self.q_cmd6: Array | None = None
        self.log: dict[str, list] = {
            k: []
            for k in (
                "t_idx", "r_idx", "dq_task_norm", "d_k_norm", "e_pred_norm",
                "dq_task_vec", "q_nom6_D", "q_cmd6_mirror", "rate_saturated_any",
                "e_pred_vec",
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
            "delay_samples": self.config.delay_samples,
        }

    def solve(
        self,
        *,
        measured_state: Any,
        measured_beam_position: Any,
        control_index: int,
        previous_input: Any,
    ) -> TwoDOFDelayAwareStep:
        cfg = self.config
        N = self.reference.sample_count
        idx = int(np.clip(control_index, 0, N - 1))
        # realization stage r = k+3 (delay_samples=2): a decision at tick k
        # produces q_cmd_{k+1}, which first becomes physical at k+3 -- see
        # module docstring / the standalone indexing unit test.
        idx_r = int(np.clip(idx + 1 + cfg.delay_samples, 0, N - 1))

        state = np.asarray(measured_state, dtype=float).reshape(7)
        u_ref = np.asarray(self.reference.input[idx], dtype=float).reshape(7)

        if self.q_nom is None:
            self.q_nom = state.copy()
            self.q_cmd6 = state[:6].copy()
        self.q_nom = self.q_nom + cfg.dt * u_ref  # insertion channel: undelayed, unchanged

        z_ref = np.asarray(self.reference.state, dtype=float)
        desired_all = np.asarray(self.reference.desired_position_m, dtype=float)

        # --- current-tick model residual d_k (live-state Jacobian) ---
        if self.schedule is not None:
            jac_k = self.schedule[idx]
        else:
            jac_k = np.asarray(self.jacobian_provider(state), dtype=float).reshape(3, 7)
        p_nominal_k = desired_all[idx]
        measured_tip = np.asarray(measured_beam_position, dtype=float).reshape(3)
        d_k = measured_tip - (p_nominal_k + jac_k @ (state - z_ref[idx]))

        # --- delay-compensated nominal preview: q_nom,D_{k+1} ---
        q_nom_D_next6 = z_ref[idx_r, :6].copy()

        # --- baseline (no new correction) prediction at r=k+3 ---
        z_hat_base = np.concatenate([q_nom_D_next6, [self.q_nom[6]]])
        if self.schedule is not None:
            jac_r = self.schedule[idx_r]
        else:
            jac_r = np.asarray(self.jacobian_provider(z_hat_base), dtype=float).reshape(3, 7)
        p_nominal_r = desired_all[idx_r]
        p_hat_base = p_nominal_r + jac_r @ (z_hat_base - z_ref[idx_r]) + d_k

        p_des_r = p_nominal_r  # reference.desired_position_m IS p_des along this trajectory
        e_pred = p_des_r - p_hat_base

        Jq_r = jac_r[:, :6]
        gram = Jq_r @ Jq_r.T + (cfg.damping ** 2) * np.eye(3)
        Jq_r_pinv = Jq_r.T @ np.linalg.solve(gram, np.eye(3))
        dq_task = Jq_r_pinv @ (cfg.kp * e_pred)

        q_star6 = q_nom_D_next6 + dq_task
        step6_raw = q_star6 - self.q_cmd6
        step6 = np.clip(step6_raw, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
        rate_saturated = bool(np.any(np.abs(step6_raw) > cfg.max_joint_step_rad))
        self.q_cmd6 = self.q_cmd6 + step6
        command6 = step6 / cfg.dt
        command = np.concatenate([command6, [u_ref[6]]])

        if cfg.enable_logging:
            self.log["t_idx"].append(idx)
            self.log["r_idx"].append(idx_r)
            self.log["dq_task_norm"].append(float(np.linalg.norm(dq_task)))
            self.log["dq_task_vec"].append(dq_task.copy())
            self.log["d_k_norm"].append(float(np.linalg.norm(d_k)))
            self.log["e_pred_norm"].append(float(np.linalg.norm(e_pred)))
            self.log["e_pred_vec"].append(e_pred.copy())
            self.log["q_nom6_D"].append(q_nom_D_next6.copy())
            self.log["q_cmd6_mirror"].append(self.q_cmd6.copy())
            self.log["rate_saturated_any"].append(rate_saturated)

        return TwoDOFDelayAwareStep(
            command=command,
            planned_input=u_ref.copy(),
            info={
                "status": "inv_2dof_delay_aware", "success": True,
                "dq_task_norm": float(np.linalg.norm(dq_task)),
                "r_idx": idx_r,
            },
        )


def build_inv_2dof_delay_aware_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    kp: float = 0.6,
    kn: float = 0.0,
    damping: float = 5.0e-2,
    q_trim_max: float = 0.03,
    dt: float = 0.1,
    max_joint_step_rad: float = 0.010,
    delay_samples: int = 2,
    enable_logging: bool = False,
    schedule: Array | None = None,
) -> TwoDOFDelayAwareController:
    return TwoDOFDelayAwareController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        schedule=schedule,
        config=TwoDOFDelayAwareConfig(
            kp=kp, kn=kn, damping=damping, q_trim_max=q_trim_max,
            dt=dt, max_joint_step_rad=max_joint_step_rad,
            delay_samples=delay_samples, enable_logging=enable_logging,
        ),
    )
