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
        # NOTE (2026-09-21): the planner(P)<->live-robot(R) frame-mismatch
        # fix does NOT live in this adapter. p_meas, J, and the
        # task-nullspace projector C are all natively robot-frame already
        # (J is built via real robot FK in build_or_load_schedule, labelled
        # "R.x,R.y,R.z" in its own console dump); only p_des/p_nominal
        # (defaulting from reference.desired_position_m) were wrong. The
        # fix is applied ONCE, worker-side, via
        # `MPCWorkerHandle.set_frame_transform(R_fit, t_fit)` -- see
        # worker_process.py. This adapter stays frame-agnostic, exactly as
        # originally designed; an earlier attempt to transform
        # measured_beam_position at this boundary was reverted after an
        # equivalence unit test showed it mixes frames (transforming p_meas
        # without also transforming J/C is not equivalent to the correct
        # fix unless R_fit happens to be near-identity).

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
            "abort_reason": None,
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
        prediction_log_path: Optional[str] = None,
        controller_kind: str = "new",
        controller_label: str = "mpc_delay_aware_process_isolated",
        insertion_offset_abort_m: Optional[float] = None,
        magnet_transform_fn: Optional[Callable[[Array], Array]] = None,
        magnet_z_bounds_m: Optional[tuple] = None,
        magnet_exclusion_lumen_C_m: Optional[Array] = None,
        magnet_exclusion_radius_m: Optional[float] = None,
        **kwargs,
    ) -> None:
        """`insertion_offset_abort_m`: if set, an independent safety monitor
        (2026-09-21) -- NOT part of the optimization, does not clip or alter
        `self._ins_cmd`'s own dynamics -- that abort_reason (via the generic
        info["abort_reason"] hook close_loop_path_follow.py now checks) once
        |insertion_m - L_ref,reference_index| exceeds this threshold. Added
        after a live insertion-position-drift failure mode (workspace abort,
        2026-09-21); see CONTROLLER_JUSTIFICATION.md's insertion-anchor
        section for how the threshold should be chosen from prior runs, not
        invented ad hoc.

        `magnet_transform_fn`/`magnet_z_bounds_m` (2026-09-23): a SECOND,
        independent safety monitor -- computes the source magnet's own
        Cartesian z (via `magnet_transform_fn(q6) -> xyz`, the project's own
        verified forward-kinematics model, NOT the robot controller's own
        `getActualTCPPose()`) from the MEASURED joints every tick, and
        aborts if it leaves `magnet_z_bounds_m=(z_min, z_max)`. Added after
        a live Q_N=0 run (vessel study, 2026-09-23) drove three joints to
        their velocity limit simultaneously for 1+ second, raising the
        magnet 12.6cm in 1.4s -- well past this project's own documented
        ~8cm safe-rise envelope -- while `close_loop_path_follow.py`'s
        generic `tcp_out_of_workspace` check (which reads
        `RobotJointStream.latest_pose()`, i.e. the ROBOT CONTROLLER'S OWN
        configured TCP offset) never fired, for reasons not fully
        root-caused (suspected: the controller's configured TCP offset does
        not match this project's own magnet-mount model, so the two checks
        are silently watching different points in space). This check does
        not depend on that TCP-offset configuration being correct -- it
        recomputes the magnet position itself, independently, from the same
        verified DH+T_F_M model this project's planning/offline work
        already relies on.

        `magnet_exclusion_lumen_C_m`/`magnet_exclusion_radius_m`
        (2026-09-23): a THIRD independent safety monitor -- the offline
        plan's magnet-exclusion-radius constraint (minimum distance from
        the magnet to the vessel lumen centreline, set from an empirically
        validated closest-safe-approach joint state) is baked into the
        offline reference trajectory but was never enforced ONLINE. With
        Q_N=0 removing all posture anchoring, the redundant joint DOF
        (7 actuators for a 3-DOF tip-tracking task) are free to wander
        arbitrarily far from that reference configuration while still
        tracking the tip well -- confirmed live 2026-09-23: both the
        no-contact and contact runs violated the exclusion radius by
        25-33mm mid-run (magnet getting CLOSER to the lumen than the
        validated safe minimum), unrelated to and undetected by either of
        the other two monitors. This check recomputes min-distance(magnet,
        lumen_C) every tick from the measured joints and aborts if it
        drops below `magnet_exclusion_radius_m` -- the magnet is fully
        free to approach right up to that boundary (getting close is the
        physically desired behavior for steering authority), only crossing
        it aborts."""
        super().__init__(**kwargs)
        self._vlim = float(joint_velocity_limit_rad_s)
        self._max_step = float(max_joint_step_rad)
        self._controller_kind = controller_kind
        self._controller_label = controller_label
        self._insertion_offset_abort_m = (
            None if insertion_offset_abort_m is None else float(insertion_offset_abort_m)
        )
        self._magnet_transform_fn = magnet_transform_fn
        self._magnet_z_bounds_m = (
            None if magnet_z_bounds_m is None else
            (float(magnet_z_bounds_m[0]), float(magnet_z_bounds_m[1]))
        )
        if self._magnet_z_bounds_m is not None and self._magnet_transform_fn is None:
            raise ValueError("magnet_z_bounds_m requires magnet_transform_fn.")
        self._magnet_exclusion_lumen_C_m = (
            None if magnet_exclusion_lumen_C_m is None
            else np.asarray(magnet_exclusion_lumen_C_m, dtype=float).reshape(-1, 3)
        )
        self._magnet_exclusion_radius_m = (
            None if magnet_exclusion_radius_m is None else float(magnet_exclusion_radius_m)
        )
        if self._magnet_exclusion_radius_m is not None and (
            self._magnet_transform_fn is None or self._magnet_exclusion_lumen_C_m is None
        ):
            raise ValueError(
                "magnet_exclusion_radius_m requires magnet_transform_fn and "
                "magnet_exclusion_lumen_C_m."
            )
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
        # measured_beam_position: real, robot-frame tip measurement from the
        # camera -- sent to the worker UNCHANGED. The worker's own p_des/
        # p_nominal are now kept frame-consistent via
        # MPCWorkerHandle.set_frame_transform (worker_process.py), not by
        # transforming this measurement -- see this class's __init__ note.
        measured_beam_position = np.asarray(estimate.tip_position_m, dtype=float).reshape(3)

        if self._q_cmd is None:
            self._q_cmd = z_meas[:6].copy()
            self._q_cmd_prev = self._q_cmd.copy()
            self._ins_cmd = float(z_meas[6])

        resp = self._worker.try_solve(
            kind=self._controller_kind, z_meas=z_meas, q_cmd=self._q_cmd, q_cmd_prev=self._q_cmd_prev,
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
                      f"(deadline={self._deadline_s*1e3:.0f}ms) -- holding q_cmd")
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
                "measured_beam_position_m": measured_beam_position.tolist(),
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

        adapter_abort_reason = None
        if self._insertion_offset_abort_m is not None:
            L_ref = float(np.asarray(self.reference.state, dtype=float)[reference_index, 6])
            L_offset_m = self._ins_cmd - L_ref
            if abs(L_offset_m) > self._insertion_offset_abort_m:
                adapter_abort_reason = (
                    f"insertion_offset_exceeded(|L-Lref|={L_offset_m*1e3:.2f}mm > "
                    f"{self._insertion_offset_abort_m*1e3:.1f}mm)"
                )
                print(f"[process_isolated_adapter] tick {self._counter}: SAFETY ABORT -- "
                      f"{adapter_abort_reason}")

        # Computed from the MEASURED joints (z_meas, this tick's actual
        # robot state, not the commanded/predicted one) -- an independent
        # monitor must watch where the magnet actually is, not where the
        # controller intends it to go. Shared between the z-bounds and
        # exclusion-radius checks below (one FK call, not two).
        magnet_xyz = None
        if self._magnet_transform_fn is not None and (
            self._magnet_z_bounds_m is not None or self._magnet_exclusion_radius_m is not None
        ):
            magnet_xyz = np.asarray(self._magnet_transform_fn(z_meas[:6]), dtype=float).reshape(3)

        if adapter_abort_reason is None and self._magnet_z_bounds_m is not None:
            z_min, z_max = self._magnet_z_bounds_m
            if magnet_xyz[2] < z_min or magnet_xyz[2] > z_max:
                adapter_abort_reason = (
                    f"magnet_z_out_of_bounds(z={magnet_xyz[2]*1e3:.1f}mm, "
                    f"bounds=[{z_min*1e3:.1f},{z_max*1e3:.1f}]mm)"
                )
                print(f"[process_isolated_adapter] tick {self._counter}: SAFETY ABORT -- "
                      f"{adapter_abort_reason}")

        if adapter_abort_reason is None and self._magnet_exclusion_radius_m is not None:
            gap_m = float(np.min(np.linalg.norm(
                self._magnet_exclusion_lumen_C_m - magnet_xyz[None, :], axis=1
            )))
            if gap_m < self._magnet_exclusion_radius_m:
                adapter_abort_reason = (
                    f"magnet_exclusion_violated(gap={gap_m*1e3:.1f}mm < "
                    f"{self._magnet_exclusion_radius_m*1e3:.1f}mm)"
                )
                print(f"[process_isolated_adapter] tick {self._counter}: SAFETY ABORT -- "
                      f"{adapter_abort_reason}")

        self._prev_input = command.copy()
        self._counter += 1

        info = {
            "controller": self._controller_label,
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
            "abort_reason": adapter_abort_reason,
        }
        return SolveResult(u0=command, infeasible=not success, info=info)
