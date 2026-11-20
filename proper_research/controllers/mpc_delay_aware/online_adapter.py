"""Online (hardware) execution-C adapter for `DelayAwareBeamOutputTrackingMPC`.

Deliberately NOT added to `controller_adapters.build_offline_solver`'s
dispatch table -- construction happens here and is injected into
`close_loop_path_follow.py` via the same `build_offline_solver`-monkeypatch
seam every other `rectangle_stage_a` controller variant already uses (see
that file's controller_kind validity check, whose comments say exactly
this for `inv_2dof_trim`/`inv_2dof_map_trim`/`inv_2dof_delay_aware`). This
keeps the shared, hardware-critical adapter file untouched beyond the two
minimal additions it needs regardless (recognizing the "mpc_delay_aware"
name, and building `BeamOutputMPCConfig` for it the same way the three
existing MPC kinds already do).

`DelayAwareOfflineControllerAdapter` differs from
`controller_adapters.OfflineJointControllerAdapter` in exactly one
structural respect: it maintains its own shadow copy of the execution-C
command accumulator (`q_cmd`, `q_cmd_prev`, insertion command) so the
delay-aware MPC's `x_exec` sees the ACTUAL commanded joint history the
accumulator seam is driving toward, not the measured state (`z_meas` is
measured only -- see `delay_aware_mpc.py`'s z_meas/x_exec API-separation
docstring, the whole point of this controller). The shadow update
(`qd = clip(u0[:6], +-vlim); delta_q = clip(qd*dt, +-max_step); q_cmd +=
delta_q`) is a direct copy of `close_loop_path_follow.py`'s own
accumulator-seam block (same clip order and values), so the two stay in
lockstep tick-for-tick: both update exactly once per adapter call, from
the same `command` this adapter returns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.hardware.online.messages import StateEstimate
from proper_research.hardware.online.online_mpc_runner import SolveResult

from .delay_aware_mpc import DelayAwareBeamOutputTrackingMPC

Array = np.ndarray
JointStateGetter = Callable[[StateEstimate], Array]


class DelayAwareOfflineControllerAdapter:
    """`solve(estimate, dt) -> SolveResult` around `DelayAwareBeamOutputTrackingMPC`."""

    def __init__(
        self,
        *,
        controller: DelayAwareBeamOutputTrackingMPC,
        reference: Any,
        joint_velocity_limit_rad_s: float,
        max_joint_step_rad: float,
        config: Optional[OfflineControllerConfig] = None,
        joint_state_getter: Optional[JointStateGetter] = None,
        prediction_log_path: Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.reference = reference
        self.config = config or OfflineControllerConfig()
        self._joint_state_getter = joint_state_getter
        self._vlim = float(joint_velocity_limit_rad_s)
        self._max_step = float(max_joint_step_rad)

        self._t0: Optional[float] = None
        self._counter = 0
        self._last_reference_index = 0
        self._prev_input = np.zeros(7, dtype=float)
        self._q_cmd: Optional[Array] = None
        self._q_cmd_prev: Optional[Array] = None
        self._ins_cmd: Optional[float] = None

        self._log_fh = None
        if prediction_log_path:
            p = Path(prediction_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = p.open("w", encoding="utf-8")
            print(f"[delay_aware_adapter] prediction log -> {p}")

    def reset(self) -> None:
        reset = getattr(self.controller, "reset", None)
        if callable(reset):
            reset()
        self._t0 = None
        self._counter = 0
        self._last_reference_index = 0
        self._prev_input = np.zeros(7, dtype=float)
        self._q_cmd = None
        self._q_cmd_prev = None
        self._ins_cmd = None

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.flush()
            self._log_fh.close()
            self._log_fh = None

    # ------------------------------------------------------------------
    # identical to OfflineJointControllerAdapter's own helpers (kept as a
    # plain copy rather than inheritance -- __call__'s body differs enough,
    # via solve_delay_aware's different signature and the shadow
    # accumulator, that subclassing would mean overriding everything
    # anyway; a flat copy is more honest about that here).
    # ------------------------------------------------------------------
    def _progress_index(self, estimate: StateEstimate) -> int:
        sample_count = int(self.reference.sample_count)
        ceiling = sample_count - 1 + int(self.config.terminal_hold_steps)
        if self.config.progress_mode == "counter":
            index = self._counter
        else:
            if self._t0 is None:
                self._t0 = estimate.t_monotonic
            elapsed = estimate.t_monotonic - self._t0
            index = int(round(elapsed / float(self.reference.sample_period_s)))
        return int(np.clip(index, 0, ceiling))

    def _joint_state(self, estimate: StateEstimate, reference_index: int) -> Array:
        if self._joint_state_getter is not None:
            return np.asarray(self._joint_state_getter(estimate), dtype=float).reshape(7)
        if estimate.robot_joints is None:
            raise RuntimeError(
                "StateEstimate has no robot_joints and no joint_state_getter was given."
            )
        q6 = np.asarray(estimate.robot_joints, dtype=float).reshape(6)
        insertion_m = float(estimate.insertion_length_m)
        if not np.isfinite(insertion_m):
            if not self.config.insertion_from_reference_if_missing:
                raise RuntimeError(
                    "StateEstimate.insertion_length_m is not finite and "
                    "insertion_from_reference_if_missing is False."
                )
            clamped = min(reference_index, int(self.reference.sample_count) - 1)
            insertion_m = float(np.asarray(self.reference.state, dtype=float)[clamped, 6])
        return np.concatenate([q6, [insertion_m]])

    # ------------------------------------------------------------------
    # the seam itself
    # ------------------------------------------------------------------
    def __call__(self, estimate: StateEstimate, dt: float) -> SolveResult:
        sample_count = int(self.reference.sample_count)
        progress_index = self._progress_index(estimate)
        reference_index = min(progress_index, sample_count - 1)
        self._last_reference_index = reference_index

        z_meas = self._joint_state(estimate, reference_index)
        measured_beam_position = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)

        if self._q_cmd is None:
            # seed identically to close_loop_path_follow.py's own
            # `if q_cmd is None: q_cmd = q.copy()` -- starts exactly where
            # the robot is measured to be on the very first tick.
            self._q_cmd = z_meas[:6].copy()
            self._q_cmd_prev = self._q_cmd.copy()
            self._ins_cmd = float(z_meas[6])

        step = self.controller.solve_delay_aware(
            z_meas=z_meas,
            q_cmd=self._q_cmd,
            q_cmd_prev=self._q_cmd_prev,
            insertion_m=self._ins_cmd,
            measured_beam_position=measured_beam_position,
            control_index=reference_index,
            previous_input=self._prev_input,
        )

        command = np.asarray(step.command, dtype=float).reshape(7)
        success = bool(getattr(step, "success", True))
        infeasible = not success
        if infeasible and self.config.zero_command_on_failure:
            command = np.zeros(7, dtype=float)

        if self._log_fh is not None:
            row = {
                "step": self._counter,
                "ref_index": reference_index,
                "z_meas": z_meas.tolist(),
                "q_cmd_k": self._q_cmd.tolist(),
                "q_cmd_km1": self._q_cmd_prev.tolist(),
                "insertion_cmd_m": self._ins_cmd,
                "u_prev": self._prev_input.tolist(),
                "u0": command.tolist(),
                "predicted_beam_positions_m": np.asarray(
                    getattr(step, "predicted_beam_positions"), dtype=float
                ).tolist(),
                "success": success,
            }
            self._log_fh.write(json.dumps(row) + "\n")
            self._log_fh.flush()

        # shadow the accumulator-seam update -- see module docstring.
        qd = np.clip(command[:6], -self._vlim, self._vlim)
        delta_q = np.clip(qd * dt, -self._max_step, self._max_step)
        self._q_cmd_prev = self._q_cmd.copy()
        self._q_cmd = self._q_cmd + delta_q
        self._ins_cmd = float(np.clip(self._ins_cmd + command[6] * dt, 0.005, 0.20))

        self._prev_input = command.copy()
        self._counter += 1

        predicted_inputs = getattr(step, "predicted_inputs", None)
        predicted_states = getattr(step, "predicted_states", None)
        predicted_beam_positions = getattr(step, "predicted_beam_positions", None)
        predicted_beam_errors = getattr(step, "predicted_beam_errors", None)
        planned_input = getattr(step, "planned_input", None)
        info = {
            "controller": getattr(self.controller, "name", None) or "mpc_delay_aware",
            "progress_index": progress_index,
            "reference_index": reference_index,
            "terminal_hold": bool(progress_index >= sample_count - 1),
            "status": str(getattr(step, "status", "")),
            "success": success,
            "iterations": int(getattr(step, "iterations", 0)),
            "solve_time_s": float(getattr(step, "solve_time_s", 0.0)),
            "objective": float(getattr(step, "objective", np.nan)),
            "primal_residual": float(getattr(step, "primal_residual", np.nan)),
            "dual_residual": float(getattr(step, "dual_residual", np.nan)),
            "first_predicted_beam_error_m": float(
                getattr(step, "first_predicted_beam_error_m", np.nan)
            ),
            "planned_input": (
                None if planned_input is None
                else np.asarray(planned_input, dtype=float).reshape(-1).tolist()
            ),
            "predicted_input_0": (
                None if predicted_inputs is None
                else np.asarray(predicted_inputs, dtype=float).reshape(-1, 7)[0].tolist()
            ),
            "predicted_state_0": (
                None if predicted_states is None
                else np.asarray(predicted_states, dtype=float).reshape(-1, 7)[0].tolist()
            ),
            "predicted_beam_position_0_m": (
                None if predicted_beam_positions is None
                else np.asarray(predicted_beam_positions, dtype=float).reshape(-1, 3)[0].tolist()
            ),
            "predicted_beam_error_0_m": (
                None if predicted_beam_errors is None
                else np.asarray(predicted_beam_errors, dtype=float).reshape(-1, 3)[0].tolist()
            ),
            "horizon_len": (
                0 if predicted_inputs is None
                else int(np.asarray(predicted_inputs, dtype=float).reshape(-1, 7).shape[0])
            ),
            "measured_joint_state": z_meas,
        }
        return SolveResult(u0=command, infeasible=infeasible, info=info)


def build_delay_aware_online_adapter(
    *,
    reference: Any,
    mpc_config: Any,
    beam_config: Any,
    schedule: Array,
    joint_velocity_limit_rad_s: float,
    max_joint_step_rad: float,
    delay_samples: int = 2,
    beta_d: float = 1.0,
    adapter_config: Optional[OfflineControllerConfig] = None,
    joint_state_getter: Optional[JointStateGetter] = None,
    prediction_log_path: Optional[str] = None,
) -> DelayAwareOfflineControllerAdapter:
    controller = DelayAwareBeamOutputTrackingMPC(
        reference=reference,
        config=mpc_config,
        beam_config=beam_config,
        reference_position_jacobians=schedule,
        delay_samples=delay_samples,
        beta_d=beta_d,
    )
    return DelayAwareOfflineControllerAdapter(
        controller=controller,
        reference=reference,
        joint_velocity_limit_rad_s=joint_velocity_limit_rad_s,
        max_joint_step_rad=max_joint_step_rad,
        config=adapter_config,
        joint_state_getter=joint_state_getter,
        prediction_log_path=prediction_log_path,
    )
