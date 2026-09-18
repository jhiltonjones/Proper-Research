"""Process-isolated online adapters (2026-09-18) -- see
`proper_research.controllers.mpc_delay_aware.worker_process` for why: a sustained
run of per-tick QP solves inside the SAME process as the RTDE/camera
threads was shown (via `rectangle_stage_a/mpc_stationary_stress_test.py`)
to permanently freeze the RTDE connection. These adapters replace
`DelayAwareOfflineControllerAdapter` / `controller_adapters.
OfflineJointControllerAdapter` for ONLINE (real hardware) use: same public
`solve(estimate, dt) -> SolveResult` seam, same accumulator-seam state
ownership, but the actual QP solve happens in a separate `MPCWorkerHandle`
process instead of a direct in-process method call.

State-ownership rule (per worker_process.py's docstring): THIS
process -- not the worker -- is the single source of truth for
q_cmd/q_cmd_prev/previous_input. On a deadline miss, timeout, stale
sequence number, or worker error, the fallback is u=0 (hold the
execution-C accumulator, don't advance it) -- and the NEXT request's
`previous_input` is built from that actually-applied u=0, never from a
late/rejected solver result.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from proper_research.hardware.online.controller_adapters import OfflineControllerConfig
from proper_research.hardware.online.messages import StateEstimate, now_monotonic
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.hardware.online.online_mpc_runner import SolveResult

Array = np.ndarray
JointStateGetter = Callable[[StateEstimate], Array]


class _BaseProcessIsolatedAdapter:
    """Shared progress/joint-state helpers -- identical to
    OfflineJointControllerAdapter's, duplicated rather than inherited (see
    online_adapter.py's own note on why -- the __call__ bodies differ too
    much to make inheritance the honest choice here)."""

    def __init__(
        self, *, reference: Any, worker: MPCWorkerHandle, deadline_s: float,
        config: Optional[OfflineControllerConfig] = None,
        joint_state_getter: Optional[JointStateGetter] = None,
    ) -> None:
        self.reference = reference
        self._worker = worker
        self._deadline_s = float(deadline_s)
        self.config = config or OfflineControllerConfig()
        self._joint_state_getter = joint_state_getter
        self._t0: Optional[float] = None
        self._counter = 0
        self._prev_input = np.zeros(7, dtype=float)
        self.deadline_misses = 0
        self.consecutive_deadline_misses = 0

    def reset(self) -> None:
        self._t0 = None
        self._counter = 0
        self._prev_input = np.zeros(7, dtype=float)
        self.deadline_misses = 0
        self.consecutive_deadline_misses = 0

    def close(self) -> None:
        self._worker.close()

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


class ProcessIsolatedBaselineAdapter(_BaseProcessIsolatedAdapter):
    """Condition A (undelayed BeamOutputTrackingMPC), solved in the worker."""

    def __init__(self, *, prediction_log_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log_fh = None
        if prediction_log_path:
            p = Path(prediction_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = p.open("w", encoding="utf-8")
            print(f"[process_isolated_adapter] prediction log -> {p}")

    def __call__(self, estimate: StateEstimate, dt: float) -> SolveResult:
        sample_count = int(self.reference.sample_count)
        progress_index = self._progress_index(estimate)
        reference_index = min(progress_index, sample_count - 1)
        z_meas = self._joint_state(estimate, reference_index)
        measured_beam_position = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)

        resp = self._worker.try_solve(
            kind="old", z_meas=z_meas, measured_beam_position=measured_beam_position,
            control_index=reference_index, previous_input=self._prev_input,
            deadline_s=self._deadline_s,
        )

        if resp is not None and resp["status"] == "ok":
            command = np.asarray(resp["command"], dtype=float)
            success = True
            self.consecutive_deadline_misses = 0
        else:
            command = np.zeros(7, dtype=float)
            success = False
            self.deadline_misses += 1
            self.consecutive_deadline_misses += 1

        if self._log_fh is not None:
            row = {
                "step": self._counter, "ref_index": reference_index,
                "z_meas": z_meas.tolist(), "u_prev": self._prev_input.tolist(),
                "u0": command.tolist(),
                "predicted_beam_positions_m": (
                    None if resp is None or resp.get("predicted_beam_positions") is None
                    else np.asarray(resp["predicted_beam_positions"], dtype=float).tolist()
                ),
                "success": success, "deadline_miss": resp is None,
                "t_solve_ms": resp["t_solve_ms"] if resp else None,
            }
            self._log_fh.write(json.dumps(row) + "\n")
            self._log_fh.flush()

        self._prev_input = command.copy()
        self._counter += 1
        info = {
            "controller": "mpc_ltv_offline_process_isolated",
            "progress_index": progress_index, "reference_index": reference_index,
            "terminal_hold": bool(progress_index >= sample_count - 1),
            "success": success,
            "status": "ok" if resp is not None else "deadline_miss",
            "solve_time_s": (resp["t_solve_ms"] / 1e3) if resp else float("nan"),
            "consecutive_deadline_misses": self.consecutive_deadline_misses,
            "predicted_beam_positions_m": (
                None if resp is None or resp.get("predicted_beam_positions") is None
                else np.asarray(resp["predicted_beam_positions"], dtype=float).tolist()
            ),
            "measured_joint_state": z_meas,
        }
        return SolveResult(u0=command, infeasible=not success, info=info)


class ProcessIsolatedDelayAwareAdapter(_BaseProcessIsolatedAdapter):
    """Condition B (DelayAwareBeamOutputTrackingMPC), solved in the worker.

    Owns the execution-C shadow accumulator (q_cmd/q_cmd_prev) exactly as
    `DelayAwareOfflineControllerAdapter` does -- the only structural change
    from that class is WHERE the solve happens and the deadline/fallback
    logic around it. On a miss, u=0 is applied (the accumulator does NOT
    advance), so q_cmd/q_cmd_prev for the NEXT request are unchanged from
    this tick's -- the delay-aware model's command history must reflect
    what was actually dispatched, never a rejected solver result.
    """

    def __init__(
        self, *, joint_velocity_limit_rad_s: float, max_joint_step_rad: float,
        prediction_log_path: Optional[str] = None, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._vlim = float(joint_velocity_limit_rad_s)
        self._max_step = float(max_joint_step_rad)
        self._q_cmd: Optional[Array] = None
        self._q_cmd_prev: Optional[Array] = None
        self._ins_cmd: Optional[float] = None
        self._log_fh = None
        if prediction_log_path:
            p = Path(prediction_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = p.open("w", encoding="utf-8")
            print(f"[process_isolated_adapter] prediction log -> {p}")

    def reset(self) -> None:
        super().reset()
        self._q_cmd = None
        self._q_cmd_prev = None
        self._ins_cmd = None

    def __call__(self, estimate: StateEstimate, dt: float) -> SolveResult:
        sample_count = int(self.reference.sample_count)
        progress_index = self._progress_index(estimate)
        reference_index = min(progress_index, sample_count - 1)
        z_meas = self._joint_state(estimate, reference_index)
        measured_beam_position = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)

        if self._q_cmd is None:
            self._q_cmd = z_meas[:6].copy()
            self._q_cmd_prev = self._q_cmd.copy()
            self._ins_cmd = float(z_meas[6])

        resp = self._worker.try_solve(
            kind="new", z_meas=z_meas, q_cmd=self._q_cmd, q_cmd_prev=self._q_cmd_prev,
            insertion_cmd=self._ins_cmd, measured_beam_position=measured_beam_position,
            control_index=reference_index, previous_input=self._prev_input,
            deadline_s=self._deadline_s,
        )

        if resp is not None and resp["status"] == "ok":
            command = np.asarray(resp["command"], dtype=float)
            success = True
            self.consecutive_deadline_misses = 0
        else:
            command = np.zeros(7, dtype=float)
            success = False
            self.deadline_misses += 1
            self.consecutive_deadline_misses += 1
            if resp is None:
                print(f"[process_isolated_adapter] tick {self._counter}: DEADLINE MISS "
                      f"(seq={seq}, deadline={self._deadline_s*1e3:.0f}ms) -- holding q_cmd")
            else:
                print(f"[process_isolated_adapter] tick {self._counter}: worker status="
                      f"{resp['status']!r} error={resp.get('error')!r} -- holding q_cmd")

        if self._log_fh is not None:
            row = {
                "step": self._counter, "ref_index": reference_index,
                "z_meas": z_meas.tolist(), "q_cmd_k": self._q_cmd.tolist(),
                "q_cmd_km1": self._q_cmd_prev.tolist(), "insertion_cmd_m": self._ins_cmd,
                "u_prev": self._prev_input.tolist(), "u0": command.tolist(),
                "predicted_beam_positions_m": (
                    None if resp is None or resp.get("predicted_beam_positions") is None
                    else np.asarray(resp["predicted_beam_positions"], dtype=float).tolist()
                ),
                "success": success,
                "deadline_miss": resp is None,
                "t_solve_ms": resp["t_solve_ms"] if resp else None,
            }
            self._log_fh.write(json.dumps(row) + "\n")
            self._log_fh.flush()

        # accumulator update from the ACTUALLY APPLIED command (0 on a miss)
        qd = np.clip(command[:6], -self._vlim, self._vlim)
        delta_q = np.clip(qd * dt, -self._max_step, self._max_step)
        self._q_cmd_prev = self._q_cmd.copy()
        self._q_cmd = self._q_cmd + delta_q
        self._ins_cmd = float(np.clip(self._ins_cmd + command[6] * dt, 0.005, 0.20))

        self._prev_input = command.copy()
        self._counter += 1

        info = {
            "controller": "mpc_delay_aware_process_isolated",
            "progress_index": progress_index, "reference_index": reference_index,
            "terminal_hold": bool(progress_index >= sample_count - 1),
            "success": success,
            "status": "ok" if resp is not None else "deadline_miss",
            "solve_time_s": (resp["t_solve_ms"] / 1e3) if resp else float("nan"),
            "consecutive_deadline_misses": self.consecutive_deadline_misses,
            "predicted_beam_positions_m": (
                None if resp is None or resp.get("predicted_beam_positions") is None
                else np.asarray(resp["predicted_beam_positions"], dtype=float).tolist()
            ),
            "measured_joint_state": z_meas,
        }
        return SolveResult(u0=command, infeasible=not success, info=info)
