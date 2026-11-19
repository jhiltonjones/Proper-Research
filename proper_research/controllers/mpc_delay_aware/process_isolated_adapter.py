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
from proper_research.controllers.mpc_delay_aware.external_safety_check import (
    check_measured_external_safety,
)
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

        # (2026-09-28 instrumentation pass) computed here, BEFORE try_solve,
        # purely so the logging block below can use it too -- moved up from
        # its original spot near the z-bounds/exclusion checks further down.
        # Pure function of z_meas[:6] (already computed above) and instance
        # attributes untouched by anything in between, so moving it earlier
        # cannot change any existing behaviour -- the z-bounds/exclusion
        # block further down now reuses this value instead of recomputing it.
        magnet_xyz = None
        if self._magnet_transform_fn is not None and (
            self._magnet_z_bounds_m is not None or self._magnet_exclusion_radius_m is not None
            or self._log_fh is not None
        ):
            magnet_xyz = np.asarray(self._magnet_transform_fn(z_meas[:6]), dtype=float).reshape(3)

        if self._q_cmd is None:
            self._q_cmd = z_meas[:6].copy()
            self._q_cmd_prev = self._q_cmd.copy()
            self._ins_cmd = float(z_meas[6])

        # 2026-11-19 (stop-path correction, point 3): measured-clearance
        # (and magnet-z-bounds) checks MOVED here, BEFORE try_solve and
        # BEFORE any accumulator advance -- previously these ran AFTER the
        # solve and AFTER the accumulator had already been updated from
        # this tick's command, so a violation was detected only once the
        # worker request had already been sent and the execution-C state
        # already moved. Both checks operate on `magnet_xyz`, computed
        # above from z_meas[:6] -- this tick's own measurement, nothing
        # advanced or solved yet. On a trip here: no worker request is
        # sent, no accumulator advance happens, and the returned command
        # is zero (held), same convention as every other reject case.
        # 2026-11-19 (abort-event record, point 5): gap_m computed
        # unconditionally (when the radius check is configured at all),
        # not only on the tripping branch, so the abort-event record below
        # can report the real measured gap on EVERY tick this check runs,
        # not only the tick it happens to trip -- makes the preceding
        # ticks' approach visible too, not just the trip itself.
        # 2026-11-19: this measured-state check is now a SHARED function
        # (common.check_measured_external_safety) so the inverse-Jacobian
        # comparator can run the identical check without acquiring this
        # adapter's QP -- see that function's docstring for the measured-
        # state-vs-candidate-command distinction. Behaviour here is
        # unchanged: same inputs, same tripped-reason strings.
        early_abort_reason, gap_m = check_measured_external_safety(
            z_meas[:6], self._magnet_transform_fn,
            magnet_z_bounds_m=self._magnet_z_bounds_m,
            magnet_exclusion_radius_m=self._magnet_exclusion_radius_m,
            lumen_C_m=self._magnet_exclusion_lumen_C_m,
        ) if magnet_xyz is not None else (None, None)
        if early_abort_reason is not None:
            print(f"[process_isolated_adapter] tick {self._counter}: SAFETY ABORT "
                  f"(pre-solve) -- {early_abort_reason}")

        if early_abort_reason is not None:
            resp = None
            round_trip_ms = 0.0
            command = np.zeros(7, dtype=float)
            success = False
            self.deadline_misses += 1
            self.consecutive_deadline_misses += 1
        else:
            # (2026-10-26 instrumentation pass) wall-clock the full round trip
            # (parent-side pickle+send, worker recv+deserialize, solve, worker
            # pickle+send, parent poll+recv+deserialize) around try_solve, so
            # round_trip_ms - t_solve_ms isolates IPC/serialization overhead on
            # its own -- previously unmeasured; see A1 loop-overhead localization.
            _ipc_t0 = time.monotonic()
            resp = self._worker.try_solve(
                kind=self._controller_kind, z_meas=z_meas, q_cmd=self._q_cmd, q_cmd_prev=self._q_cmd_prev,
                insertion_cmd=self._ins_cmd, measured_beam_position=measured_beam_position,
                control_index=reference_index, previous_input=self._prev_input,
                deadline_s=self._deadline_s,
                # 2026-11-18: feeds ONLY the jacobian-adaptation estimator's
                # staleness/duplicate-sample gates (see jacobian_adaptation.py);
                # a no-op for every other controller kind/config.
                measurement_frame_index=getattr(estimate, "frame_index", None),
                measurement_age_s=(
                    estimate.robot_joints_age() if hasattr(estimate, "robot_joints_age") else None
                ),
            )
            round_trip_ms = (time.monotonic() - _ipc_t0) * 1e3

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
            # (2026-09-28 instrumentation pass) purely additional fields,
            # read straight off `resp` (already received from the worker
            # above) or `magnet_xyz` (already computed above) -- nothing
            # here feeds back into the control loop.
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
                "round_trip_ms": round_trip_ms,
                "predicted_states": (
                    None if resp is None or resp.get("predicted_states") is None
                    else np.asarray(resp["predicted_states"], dtype=float).tolist()
                ),
                # 2026-11-19 (future-run diagnosability, point 2 of the
                # prior review): the Ec/Sc COMMAND stack -- distinct from
                # predicted_states (Ep/Sp, physical) above. Persisted
                # directly so no offline reconstruction from x_exec/Sp
                # inversion (with its real rank-deficiency limitation,
                # see test_fresh_solve_all_stages_against_qp_and_fk.py) is
                # ever needed again for a FUTURE run.
                "predicted_commands": (
                    None if resp is None or resp.get("predicted_commands") is None
                    else np.asarray(resp["predicted_commands"], dtype=float).tolist()
                ),
                "predicted_inputs": (
                    None if resp is None or resp.get("predicted_inputs") is None
                    else np.asarray(resp["predicted_inputs"], dtype=float).tolist()
                ),
                "magnet_exclusion_mapping": None if resp is None else resp.get("magnet_exclusion_mapping"),
                "jacobian_used": (
                    None if resp is None or resp.get("jacobian_used") is None
                    else np.asarray(resp["jacobian_used"], dtype=float).tolist()
                ),
                "qp_objective": None if resp is None else resp.get("qp_objective"),
                "qp_status": None if resp is None else resp.get("qp_status"),
                "qp_iterations": None if resp is None else resp.get("qp_iterations"),
                "qp_solve_time_s": None if resp is None else resp.get("qp_solve_time_s"),
                "qp_primal_residual": None if resp is None else resp.get("qp_primal_residual"),
                "qp_dual_residual": None if resp is None else resp.get("qp_dual_residual"),
                "magnet_xyz_m": None if magnet_xyz is None else magnet_xyz.tolist(),
                "dual_y": (
                    None if resp is None or resp.get("dual_y") is None
                    else np.asarray(resp["dual_y"], dtype=float).tolist()
                ),
                # 2026-11-19 fix (post-review): same gap as the `info` dict
                # above -- present in `resp`, never persisted here either.
                "jacobian_adaptation_diagnostics": None if resp is None else resp.get("jacobian_adaptation_diagnostics"),
                # 2026-11-19 (corrected): the 0.0 used when resp is None is
                # a PARENT-SIDE HOLD VALUE (no response arrived to read a
                # real gain from) -- it is NEVER an observation of what
                # the worker's own (unobserved, possibly still-running or
                # discarded) solve actually used. `jacobian_scale_is_
                # observed` makes this explicit so a reader never has to
                # infer it from whether resp happened to be None.
                "jacobian_scale_applied": 0.0 if resp is None else resp.get("jacobian_scale_applied", 0.0),
                "jacobian_scale_is_observed": resp is not None,
                # 2026-11-19 (abort-event record, point 5): explicit fields
                # for a pre-solve abort -- all read from values already
                # computed above (estimate.t_monotonic, gap_m, the
                # configured thresholds, self._q_cmd), nothing fabricated.
                # None whenever the corresponding check isn't configured or
                # didn't trip, same as every other optional field here.
                "early_abort_reason": early_abort_reason,
                # 2026-11-19 (point 3 correction): `estimate.t_monotonic` is
                # NOT guaranteed to be the JOINT sample's own timestamp --
                # path_follow.jsonl already logs several DISTINCT ages
                # (joint_sample_age_at_use_s, camera_vision_age_s, ...)
                # precisely because vision and joints can be sampled at
                # different times. The magnet clearance here is computed
                # from JOINTS (via magnet_transform_fn(z_meas[:6])), so its
                # age must be the joint sample's own age -- the same
                # `estimate.robot_joints_age()` accessor try_solve() already
                # uses above for the jacobian-adaptation staleness gate --
                # not a generic `t_monotonic`-based age that may describe a
                # different (e.g. vision) sample.
                "clearance_sample_age_s": (
                    estimate.robot_joints_age() if hasattr(estimate, "robot_joints_age") else None
                ),
                "magnet_exclusion_gap_mm": None if gap_m is None else gap_m * 1e3,
                "magnet_exclusion_radius_mm": (
                    None if self._magnet_exclusion_radius_m is None
                    else self._magnet_exclusion_radius_m * 1e3
                ),
                "magnet_z_bounds_mm": (
                    None if self._magnet_z_bounds_m is None
                    else [self._magnet_z_bounds_m[0] * 1e3, self._magnet_z_bounds_m[1] * 1e3]
                ),
                # 2026-11-19 (point 3 correction): renamed from the earlier
                # "last_dispatched_target_rad" -- that claim was NOT
                # verified against the actual publication record. This
                # field is the execution-C accumulator (self._q_cmd) as of
                # entering THIS tick, i.e. the SAME value as q_cmd_k above,
                # logged before any conditional advance. Whether it equals
                # what close_loop_path_follow.py actually dispatched
                # depends on which of that file's three q_target branches
                # is active (q_target = q_cmd only when feedforward and
                # feedforward_joint_trajectory are both False -- true for
                # this runner's current default config, confirmed against
                # pilotA's own logged q_target_rad, but NOT guaranteed for
                # every config of that file) -- this adapter has no
                # visibility into which branch ran, so it cannot honestly
                # claim to report the dispatch record itself.
                "accumulator_q_cmd_rad": self._q_cmd.tolist(),
            }
            self._log_fh.write(json.dumps(row) + "\n")
            self._log_fh.flush()

        # 2026-11-19 (stop-path correction, point 3): accumulator advance
        # SKIPPED entirely on an early (pre-solve) measured-clearance/
        # z-bounds rejection -- `command` is already zero in that branch,
        # but the point is to never even run this update against a
        # rejected tick's state, not merely to have it compute a no-op.
        if early_abort_reason is None:
            # accumulator update from the ACTUALLY APPLIED command (0 on a miss)
            qd = np.clip(command[:6], -self._vlim, self._vlim)
            delta_q = np.clip(qd * dt, -self._max_step, self._max_step)
            self._q_cmd_prev = self._q_cmd.copy()
            self._q_cmd = self._q_cmd + delta_q
            self._ins_cmd = float(np.clip(self._ins_cmd + command[6] * dt, 0.005, 0.20))

        # adapter_abort_reason: starts as whatever the pre-solve checks
        # above already found (magnet z-bounds / measured-clearance); the
        # insertion-offset check below is INDEPENDENT (depends on
        # self._ins_cmd, which only advances in the branch above) and
        # still runs after the solve, same as before -- only the two
        # magnet checks moved earlier; this one did not.
        adapter_abort_reason = early_abort_reason
        if adapter_abort_reason is None and self._insertion_offset_abort_m is not None:
            L_ref = float(np.asarray(self.reference.state, dtype=float)[reference_index, 6])
            L_offset_m = self._ins_cmd - L_ref
            if abs(L_offset_m) > self._insertion_offset_abort_m:
                adapter_abort_reason = (
                    f"insertion_offset_exceeded(|L-Lref|={L_offset_m*1e3:.2f}mm > "
                    f"{self._insertion_offset_abort_m*1e3:.1f}mm)"
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
            # (2026-09-28 instrumentation pass) fills in slots the caller's
            # own JSONL writer (`close_loop_path_follow.py`) already reads
            # via `info.get(...)` but which were never populated for this
            # controller path -- all values already computed above/in resp,
            # nothing new here changes `command`/`success`/control flow.
            "iterations": None if resp is None else resp.get("qp_iterations"),
            # 2026-10-14 speed-fix investigation: the worker already computes
            # OSQP's own nuanced status (e.g. "run time limit reached" when
            # solver_time_limit_s/accept_time_limit_solution engage) and
            # sends it as qp_status, but nothing downstream of this adapter
            # ever read it -- the top-level "status" above is only the
            # coarse "did the worker reply at all" flag. Logging-only
            # addition: forwards qp_status through so it can finally reach
            # path_follow.jsonl (see close_loop_path_follow.py's row dict).
            # Does not change `command`/`success`/control flow.
            "qp_status": None if resp is None else resp.get("qp_status"),
            # 2026-11-19 fix (post-review): these reached the worker's
            # response correctly but were never forwarded into `info`, so
            # close_loop_path_follow.py's row writer -- the ONLY thing
            # that persists per-tick data to path_follow.jsonl -- never
            # saw them. Present in the IPC response is not the same as
            # saved to disk; this closes that gap. Logging-only, no
            # effect on command/success/control flow.
            "jacobian_adaptation_diagnostics": None if resp is None else resp.get("jacobian_adaptation_diagnostics"),
            # 2026-11-19 (corrected): same hold-value-not-observation
            # distinction as process_isolated_adapter's own log row --
            # see that site for the full comment.
            "jacobian_scale_applied": 0.0 if resp is None else resp.get("jacobian_scale_applied", 0.0),
            "jacobian_scale_is_observed": resp is not None,
            "objective": None if resp is None else resp.get("qp_objective"),
            "primal_residual": None if resp is None else resp.get("qp_primal_residual"),
            "dual_residual": None if resp is None else resp.get("qp_dual_residual"),
            "predicted_input_0": None if resp is None else np.asarray(resp["command"], dtype=float).tolist(),
            "predicted_state_0": (
                None if resp is None or resp.get("predicted_states") is None
                else np.asarray(resp["predicted_states"], dtype=float)[0].tolist()
            ),
            "predicted_beam_position_0_m": (
                None if resp is None or resp.get("predicted_beam_positions") is None
                else np.asarray(resp["predicted_beam_positions"], dtype=float)[0].tolist()
            ),
            "horizon_len": (
                None if resp is None or resp.get("predicted_states") is None
                else int(np.asarray(resp["predicted_states"]).shape[0])
            ),
        }
        return SolveResult(u0=command, infeasible=not success, info=info)
