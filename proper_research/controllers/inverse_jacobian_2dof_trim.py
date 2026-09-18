"""2DOF nominal-feedforward + non-integrating feedback trim (2026-09-17).

Companion to `inverse_jacobian_controller.py`, not a replacement. Built after
a live windup diagnosis on the rectangle shape: under the "accumulator seam"
execution layer (see `close_loop_path_follow.PathFollowConfig.accumulator_seam`
and this package's `hardware/online/rectangle_stage_a/README.md`), the
existing `InverseJacobianBeamController`'s combined task+null+feedforward
velocity gets integrated into a persistent absolute joint target every tick.
That is correct for the *nominal* planner feedforward (small, persistent
increments should accumulate), but wrong for *feedback* corrections: a large
one-tick error correction should not remain baked into the target forever.
Confirmed live (rectangle, kp=1, kn=1): pending servo error (`q_cmd-q_meas`)
visibly escalated during the harder transit/cornering sections in lockstep
with a high positive `c_wind` correlation between the pending error and the
next feedback correction -- textbook command-layer windup, despite this
controller having no explicit integral term.

Architecture
------------
Persistent nominal channel (integrated, exactly the plan's own feedforward,
seeded once at the true start so it never needs re-anchoring to the
planner)::

    q_nom_{k+1} = q_nom_k + dt * u_ref,k
    q_nom_0     = q_ref,0   (== the measured state once the robot is placed there)

Non-integrating feedback trim (recomputed fresh every tick from the CURRENT
error -- not the k+1 lookahead the plain inverse-Jacobian controller chases,
since the nominal channel already supplies that advance)::

    e_k       = p_des,k - p_meas,k                    (current index, not k+1)
    dz_task,k = J_k^+ (kp * e_k)                       (a position trim; note
                                                         no /dt -- see below)
    dz_null,k = P_DLS,k (kn * (z_ref,k - z_meas,k))    (6 joints only)
    dq_trim,k = sat(dz_task,k + dz_null,k, +-q_trim_max)

    q*_{k+1}  = q_nom_{k+1}[:6] + dq_trim,k

`dz_task` drops the `/dt` the velocity-form inverse-Jacobian controller has
(`u_task = J^+ kp e/dt`) because this is a position trim, not a velocity:
`dt * u_task = dt * J^+ (kp e/dt) = J^+ (kp e)`.

The controller's own internal `P_DLS @ u_ref` feedforward term (see
`InverseJacobianBeamController.feedforward`) is intentionally NOT used here
-- the persistent `q_nom` channel already supplies the full nominal
feedforward, so adding it again would double-count the planner's velocity.

Live-validated on the rectangle shape (10x15mm, execution layer C: 50Hz
interpolated `servoJ` streaming of the accumulator, see the harness README):
a kp/kn ablation with kp in {1.0, 0.6, 0.3, 0.0} at kn=0 found a clean
U-shaped tradeoff, kp=0.6 the minimum on both RMS and max active-region
tracking error, beating the open-loop-FF baseline by ~25%/~30% respectively,
with zero rate-limit saturation and zero beam-error spikes above 4mm.
kp=1.0 alone showed 20% rate-limit saturation. See
`hardware/online/rectangle_stage_a/README.md` for the full numbers and the
kn ablation that motivated dropping the posture/null term to 0 first.

Execution-seam contract
------------------------
This controller does NOT itself talk to the harness's accumulator seam --
it *reproduces* that seam's exact update internally (`self.q_cmd6`, seeded
and updated identically to `PathFollowConfig.accumulator_seam`'s own
`q_cmd`), then returns `command = (q*_{k+1} - q_cmd_k) / dt` as a plain
velocity. Provided `max_joint_step_rad == dt * joint_velocity_limit_rad_s`
(kept consistent by the run scripts), the harness's own
`delta_q = clip(dt*command, +-max_joint_step_rad)` reproduces exactly

    q_cmd_{k+1} = q_cmd_k + clip(q*_{k+1} - q_cmd_k, +-max_joint_step_rad)

so this controller's internal `q_cmd6` mirror and the harness's real `q_cmd`
stay in lockstep by construction (same seed, same deterministic update
applied once per tick by each side). Requires
`PathFollowConfig.accumulator_seam = True` and
`PathFollowConfig.feedforward_joint_trajectory = False` (not anchored to the
planner) -- `build_inv_2dof_trim_controller_kind` in `controller_adapters.py`
does not set these for you; the run script must.

Insertion is nominal-only (`command[6] = u_ref[6]`, no trim) -- handled by
the harness's own accumulator_seam insertion branch, unchanged from every
other accumulator-seam controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

Array = np.ndarray

__all__ = [
    "TwoDOFTrimConfig",
    "TwoDOFTrimStep",
    "TwoDOFTrimController",
    "build_inv_2dof_trim_controller",
]


@dataclass
class TwoDOFTrimConfig:
    kp: float = 0.6
    kn: float = 0.0
    damping: float = 5.0e-2
    q_trim_max: float = 0.03          # rad, position-trim saturation bound
    dt: float = 0.1
    max_joint_step_rad: float = 0.010  # must equal dt * joint_velocity_limit_rad_s
    enable_logging: bool = False


@dataclass
class TwoDOFTrimStep:
    command: Array
    success: bool = True
    status: str = "inv_2dof_trim"
    iterations: int = 0
    solve_time_s: float = 0.0
    planned_input: Array | None = None
    info: dict = field(default_factory=dict)


class TwoDOFTrimController:
    """See module docstring for the full architecture and rationale."""

    name = "inv_2dof_trim"

    def __init__(
        self,
        *,
        reference: Any,
        jacobian_provider: Callable[[Array], Array],
        config: TwoDOFTrimConfig | None = None,
        schedule: Array | None = None,
    ) -> None:
        self.reference = reference
        self.jacobian_provider = jacobian_provider
        # 2026-09-18: optional path-indexed Jacobian schedule (e.g. the
        # genuine LTV schedule). When given, `schedule[idx]` is used instead
        # of `jacobian_provider(state)` -- for a controlled A/B against
        # inverse_jacobian_2dof_delay_aware.py's `schedule[idx+3]`, so the
        # two conditions differ ONLY in which index they read, not also in
        # Jacobian source (live per-state eval vs schedule lookup). None
        # (the default) preserves the original behaviour for every existing
        # caller -- this is additive, not a change to the validated
        # kp=0.6/kn=0 baseline runs that didn't pass it.
        self.schedule = None if schedule is None else np.asarray(schedule, dtype=float)
        self.config = config or TwoDOFTrimConfig()
        self.q_nom: Array | None = None   # 7-vector persistent nominal accumulator
        self.q_cmd6: Array | None = None  # mirrors the harness's accumulator_seam q_cmd[:6]
        self.log: dict[str, list] = {
            k: []
            for k in (
                "t_idx", "dq_task_norm", "dq_null_norm", "dq_trim_norm_pre",
                "dq_trim_norm_post", "saturated_any", "rate_saturated_any",
                "q_nom6", "q_cmd6_mirror", "dL_null", "u_L_applied",
                "dq_task_vec",
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
        }

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
        state = np.asarray(measured_state, dtype=float).reshape(7)
        u_ref = np.asarray(self.reference.input[idx], dtype=float).reshape(7)

        if self.q_nom is None:
            self.q_nom = state.copy()
            self.q_cmd6 = state[:6].copy()

        self.q_nom = self.q_nom + cfg.dt * u_ref

        if self.schedule is not None:
            jac = self.schedule[idx]
        else:
            jac = np.asarray(self.jacobian_provider(state), dtype=float).reshape(3, 7)
        Jq = jac[:, :6]
        gram = Jq @ Jq.T + (cfg.damping ** 2) * np.eye(3)
        Jq_pinv = Jq.T @ np.linalg.solve(gram, np.eye(3))
        P = np.eye(6) - Jq_pinv @ Jq

        desired = np.asarray(self.reference.desired_position_m, dtype=float)[idx]
        measured_tip = np.asarray(measured_beam_position, dtype=float).reshape(3)
        e_k = desired - measured_tip
        dz_task = Jq_pinv @ (cfg.kp * e_k)

        z_ref = np.asarray(self.reference.state, dtype=float)[idx, :6]
        dz_null = P @ (cfg.kn * (z_ref - state[:6]))

        dq_trim_pre = dz_task + dz_null
        dq_trim = np.clip(dq_trim_pre, -cfg.q_trim_max, cfg.q_trim_max)
        saturated = bool(np.any(np.abs(dq_trim_pre) > cfg.q_trim_max))

        q_star6 = self.q_nom[:6] + dq_trim
        step6_raw = q_star6 - self.q_cmd6
        step6 = np.clip(step6_raw, -cfg.max_joint_step_rad, cfg.max_joint_step_rad)
        rate_saturated = bool(np.any(np.abs(step6_raw) > cfg.max_joint_step_rad))
        self.q_cmd6 = self.q_cmd6 + step6
        command6 = step6 / cfg.dt

        command = np.concatenate([command6, [u_ref[6]]])

        if cfg.enable_logging:
            gram7 = jac @ jac.T + (cfg.damping ** 2) * np.eye(3)
            jac_pinv7 = jac.T @ np.linalg.solve(gram7, np.eye(3))
            P7 = np.eye(7) - jac_pinv7 @ jac
            z_ref7 = np.asarray(self.reference.state, dtype=float)[idx]
            dz_null7 = P7 @ (cfg.kn * (z_ref7 - state))
            self.log["t_idx"].append(idx)
            self.log["dq_task_vec"].append(dz_task.copy())
            self.log["dq_task_norm"].append(float(np.linalg.norm(dz_task)))
            self.log["dq_null_norm"].append(float(np.linalg.norm(dz_null)))
            self.log["dq_trim_norm_pre"].append(float(np.linalg.norm(dq_trim_pre)))
            self.log["dq_trim_norm_post"].append(float(np.linalg.norm(dq_trim)))
            self.log["saturated_any"].append(saturated)
            self.log["rate_saturated_any"].append(rate_saturated)
            self.log["q_nom6"].append(self.q_nom[:6].copy())
            self.log["q_cmd6_mirror"].append(self.q_cmd6.copy())
            self.log["dL_null"].append(float(dz_null7[6]))
            self.log["u_L_applied"].append(float(u_ref[6]))

        return TwoDOFTrimStep(
            command=command,
            planned_input=u_ref.copy(),
            info={"status": "inv_2dof_trim", "success": True},
        )


def build_inv_2dof_trim_controller(
    *,
    reference: Any,
    jacobian_provider: Callable[[Array], Array],
    kp: float = 0.6,
    kn: float = 0.0,
    damping: float = 5.0e-2,
    q_trim_max: float = 0.03,
    dt: float = 0.1,
    max_joint_step_rad: float = 0.010,
    enable_logging: bool = False,
    schedule: Array | None = None,
) -> TwoDOFTrimController:
    """Build the 2DOF trim controller with the live-validated defaults.

    ``kp=0.6, kn=0`` is the rectangle Stage-A development point (2026-09-17)
    -- see the module docstring's live-validation numbers. ``schedule``: see
    ``TwoDOFTrimController.__init__``'s docstring note.
    """
    return TwoDOFTrimController(
        reference=reference,
        jacobian_provider=jacobian_provider,
        schedule=schedule,
        config=TwoDOFTrimConfig(
            kp=kp, kn=kn, damping=damping, q_trim_max=q_trim_max,
            dt=dt, max_joint_step_rad=max_joint_step_rad,
            enable_logging=enable_logging,
        ),
    )
