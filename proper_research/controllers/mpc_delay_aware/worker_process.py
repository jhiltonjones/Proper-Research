"""Process-isolated MPC solve (2026-09-18).

Root cause established via `rectangle_stage_a/mpc_stationary_stress_test.py`
(no-motion, 180s, 1800 ticks): a SUSTAINED run of moderately-heavy per-tick
QP solves (5 consecutive ticks at 55-104ms, one exceeding the full 100ms
control period) directly precedes a PERMANENT freeze of the RTDE receive
connection -- both a value-based check and an independent
`rtde_receive.getTimestamp()`-based check age in exact lockstep with
wall-clock time from that point on, confirming a genuine connection wedge
(matching the "heavy CPU-bound work in the same process" mechanism this
project's rectangle_stage_a/README.md already documents for the LTV
schedule build, just triggered here by a burst of per-tick solves instead
of one long block). `close_loop_path_follow.py`'s existing reconnect-and-
verify logic only runs once, before the control loop starts -- nothing
recovers a mid-loop freeze.

Fix: move the QP solve to a SEPARATE process (not a thread -- the GIL is
exactly what let per-tick solve cost starve the RTDE/camera threads).
Process A (the caller: RTDE + camera + servoJ dispatch + the execution-C
accumulator) owns every hardware handle and is the single source of truth
for q_cmd/q_cmd_prev/u_applied_prev. Process B (this module's
`_worker_main`) owns only the MPC instance -- it never touches RTDE, the
camera, servoJ or the advancer, and never advances q_cmd/u_prev itself;
every solve is against a fresh, immutable snapshot A sends it.

2026-09-18, second pass: a full-pipeline dry-run (worker spawn happening
AFTER camera/robot were already connected, inside `build_offline_solver`)
still hit stale_vision at tick 5. Spawning a `spawn`-context process is
itself real, one-time CPU-bound work in the parent (fork + a fresh Python
interpreter re-importing numpy/scipy/osqp in the child) -- landing that
burst at the same moment the existing solver-build staleness window is
already fragile compounds it. Fix: the LIFECYCLE now hard-orders worker
startup before ANY hardware connection --

    1. spawn the worker
    2. worker imports everything, builds both MPC controllers, and runs
       ONE representative dummy solve (so lazy OSQP/BLAS/numpy init costs
       are paid here, not on the first real tick) -- only THEN sends ready
    3. caller opens RTDE, camera, preflight/freshness checks
    4. control loop starts

`MPCWorkerHandle` records `t_spawn_start`/`t_worker_ready` so a caller can
log the full lifecycle (t_spawn_start, t_worker_ready, t_rtde_connect,
t_camera_connect, t_first_fresh_packet, t_control_start) and tell exactly
which phase a freeze happened in, rather than inferring it from console
output.

This module deliberately lives OUTSIDE `proper_research.hardware.online`
(despite being hardware-adjacent in purpose) -- that package's `__init__.py`
eagerly imports `CameraSource`/`RobotSink`/`AdvancerSink` at import time,
and since a `spawn` child re-imports its target function's module fresh,
importing a worker module that LIVES inside that package would drag the
whole RTDE/camera import graph into the worker process too, even though
`_worker_main` never touches any of it. Keeping this module under
`controllers/mpc_delay_aware/` (whose own package imports -- see
`proper_research/controllers/__init__.py` -- are hardware-free) keeps the
worker's import graph terminating at numpy/scipy/osqp/the MPC controllers.

Protocol (over a `multiprocessing.Pipe`, `spawn` context -- NOT the default
`fork`, since forking a process that already owns RTDE/camera sockets and
background threads is exactly what this is trying to avoid):

    A -> B: {"seq": int, "kind": "old"|"new", "z_meas": (7,),
             "q_cmd": (6,), "q_cmd_prev": (6,), "insertion_cmd": float,
             "measured_beam_position": (3,), "control_index": int,
             "previous_input": (7,)}
    B -> A: {"seq": int, "status": "ok"|"error", "command": (7,) or None,
             "predicted_beam_positions": (N,3) or None,
             "t_solve_ms": float, "error": str or None}

Exactly one request may be outstanding at a time -- A never enqueues a
second problem behind an unanswered one (a QP solved against tick k-3's
state is worse than not solving at all). A response whose `seq` doesn't
match the currently-outstanding request is discarded by the caller, not by
this module (see `MPCWorkerHandle.try_solve`).

On a deadline miss, timeout, stale seq, or worker error, the caller's
convention (not this module's) is: apply u=0 (hold q_cmd, don't advance
the accumulator) for that tick, and feed the NEXT solve's `previous_input`
from what was actually applied (0), never from a late/rejected u0 -- see
`process_isolated_adapter.py`'s adapters.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Optional

import numpy as np

Array = np.ndarray


def _force_single_threaded_blas() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _worker_main(
    conn: "mp.connection.Connection",
    *,
    plan_dir: str,
    schedule_path: str,
    mpc_config_kwargs: dict,
    beam_config_kwargs: dict,
    delay_samples: int,
    beta_d: float,
    single_threaded: bool,
    enable_nullspace_r700: bool = False,
    nullspace_r700_config_kwargs: Optional[dict] = None,
    insertion_state_anchor_weight: float = 0.0,
    enable_exact_qn_zero: bool = False,
    exact_qn_zero_config_kwargs: Optional[dict] = None,
) -> None:
    """Entry point for Process B. Constructs its OWN MPC instances (old +
    new, so one worker can answer requests for either condition -- the
    live A/B alternates between them run-to-run, not tick-to-tick), then
    runs ONE dummy solve of each kind before signaling ready -- pays lazy
    OSQP/BLAS initialization cost here, not on the first real control tick.

    `enable_nullspace_r700`: also build the exact-task-nullspace/gamma=0/
    R700 controller (`StagewiseTaskNullspaceDelayAwareMPC`, the SAME class
    validated offline via `stagewise_task_nullspace_self_test.py`'s
    gamma=1<->M0 equivalence test and the axis-resolved/Q_N-ablation replay
    campaign, 2026-09-20 -- see that package's README). No new projector
    algebra lives here: this worker only constructs the already-tested
    class and, once, verifies the projector properties hold over the FULL
    stored schedule before reporting ready (see the assertions block
    below) -- the live per-tick solve path is untouched from what was
    replayed.

    `enable_exact_qn_zero` (2026-09-23): also build
    `ExactQNZeroTaskNullspaceDelayAwareMPC` -- the vessel-navigation
    model-necessity study's frozen Q_N=0 condition (P_N=P_R=0 exactly at
    every stage, not merely gamma=0, which leaves P_N fully active). Kept
    fully separate from `enable_nullspace_r700`/`null_controller` -- both
    may be enabled simultaneously without interacting; each is dispatched
    by its own `kind` string. Verified offline (before this was wired in)
    to reproduce the study's own `DecoupledProjectorMPC(projector=None)`
    bit-for-bit (max command diff 0.0 across 6 test states).
    """
    if single_threaded:
        _force_single_threaded_blas()

    import math

    from proper_research.controllers.mpc_delay_aware.delay_aware_mpc import (
        DelayAwareBeamOutputTrackingMPC,
    )
    from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
        BeamOutputMPCConfig, BeamOutputTrackingMPC,
    )
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        ConfigurationMPCConfig, load_configuration_reference,
    )

    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(schedule_path)
    mpc_config = ConfigurationMPCConfig(**mpc_config_kwargs)
    beam_config = BeamOutputMPCConfig(**beam_config_kwargs)

    old_controller = BeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule,
    )
    new_controller = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=delay_samples, beta_d=beta_d,
    )

    null_controller = None
    if enable_nullspace_r700:
        from proper_research.controllers.mpc_delay_aware.stagewise_task_nullspace import (
            build_stagewise_projectors,
        )
        from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
            InsertionAnchoredTaskNullspaceMPC,
        )
        from proper_research.controllers.mpc_delay_aware.target_consistent import (
            build_beam_plane_projection,
        )

        null_mpc_config = ConfigurationMPCConfig(**(nullspace_r700_config_kwargs or {}))
        # InsertionAnchoredTaskNullspaceMPC subclasses the plain gamma=0/R700
        # nullspace controller and adds nothing when
        # insertion_state_anchor_weight=0.0 -- verified bit-identical by
        # insertion_anchor_self_test.py (H/f/u0 diff = 0.00e+00). Using it
        # here unconditionally (default weight 0.0) is a strict, regression-
        # tested generalization, not a behavior change for existing callers.
        null_controller = InsertionAnchoredTaskNullspaceMPC(
            gamma=0.0, insertion_state_anchor_weight=insertion_state_anchor_weight,
            reference=reference, config=null_mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=delay_samples, beta_d=beta_d,
        )

        # --- startup assertions (frozen-spec item 13): verify the exact
        # projector properties the gamma=1<->M0 unit test relies on, over
        # the FULL stored schedule, using the SAME beam-plane projection C
        # and numerical-rank rule as the validated replay implementation
        # (build_stagewise_projectors, unchanged, default rank_tol). This
        # is a one-time verification pass, not the live solve path -- the
        # per-tick solve still recomputes P_N,j from the SAME function
        # inside StagewiseTaskNullspaceDelayAwareMPC._dynamic_qp_terms_exec,
        # exactly as replayed. ---
        C = build_beam_plane_projection(axial_axis_R=(-1.0, 0.0, 0.0), normal_axis_R=(0.0, 0.0, -1.0))
        state_scale = np.asarray(null_mpc_config.state_error_scale, dtype=float)
        P_R_full, P_N_full, ranks_full = build_stagewise_projectors(
            J_schedule=schedule, C=C, state_error_scale=state_scale,
        )
        idempotent_err = float(np.max(np.abs(
            np.einsum("nij,njk->nik", P_N_full, P_N_full) - P_N_full
        )))
        symmetric_err = float(np.max(np.abs(P_N_full - np.transpose(P_N_full, (0, 2, 1)))))
        task_annih_err = 0.0
        for j in range(schedule.shape[0]):
            J_s = (C @ schedule[j]) * state_scale[None, :]
            task_annih_err = max(task_annih_err, float(np.max(np.abs(J_s @ P_N_full[j]))))
        eps = 1.0e-6
        assert idempotent_err < eps, f"P_N not idempotent: max err={idempotent_err:.3e}"
        assert symmetric_err < eps, f"P_N not symmetric: max err={symmetric_err:.3e}"
        assert task_annih_err < 1.0e-5, f"J^t S_z P_N not ~0: max err={task_annih_err:.3e}"
        unique_ranks = sorted(set(int(r) for r in ranks_full.tolist()))
        print(f"[worker] nullspace projector startup checks PASS over full schedule "
              f"{schedule.shape}: idempotent_err={idempotent_err:.2e} "
              f"symmetric_err={symmetric_err:.2e} task_annihilation_err={task_annih_err:.2e} "
              f"ranks={unique_ranks}")
        print("[worker] state cost: exact task-nullspace only, gamma=0")
        print(f"[worker] insertion state anchor: w_L={insertion_state_anchor_weight} "
              f"(0.0 = bit-identical to plain R700/gamma=0, regression-tested)")
        print(f"[worker] R multiplier: 700 (input_tracking_weight="
              f"{null_mpc_config.input_tracking_weight})")
        print(f"[worker] Rd: baseline {null_mpc_config.input_increment_weight}, "
              f"increment scale explicitly pinned={null_mpc_config.input_increment_scale}")
        print(f"[worker] delay_samples={delay_samples} beta_d={beta_d} "
              f"horizon={mpc_config.prediction_horizon} terminal_cost=False (V_f=0)")

    exact_qn0_controller = None
    if enable_exact_qn_zero:
        from proper_research.controllers.mpc_delay_aware.exact_qn_zero_mpc import (
            ExactQNZeroTaskNullspaceDelayAwareMPC,
        )

        exact_qn0_mpc_config = ConfigurationMPCConfig(**(exact_qn_zero_config_kwargs or {}))
        exact_qn0_controller = ExactQNZeroTaskNullspaceDelayAwareMPC(
            gamma=0.0, reference=reference, config=exact_qn0_mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=delay_samples, beta_d=beta_d,
        )
        print("[worker] exact Q_N=0 controller built (P_N=P_R=0 at every stage, "
              f"input_tracking_weight={exact_qn0_mpc_config.input_tracking_weight})")

    def _apply_frame_transform(controller, R_fit: np.ndarray, t_fit: np.ndarray) -> None:
        """Planner(P)->live-robot(R) frame-mismatch fix (2026-09-21): p_des
        and p_nominal are the ONLY quantities wrong here (they default from
        reference.desired_position_m, the raw un-registered planner-frame
        reference this worker was necessarily constructed from -- see
        module docstring). measured_beam_position, the reference_position_
        jacobians schedule, and the task-nullspace projector C are already
        natively robot-frame (J is built via real robot FK, independent of
        the planner's own abstract path coordinates) and are NOT touched.
        An earlier attempt to instead transform measured_beam_position at
        the adapter boundary was reverted after an equivalence unit test
        showed that approach mixes frames unless R_fit happens to be
        near-identity -- this is the mathematically exact fix, not an
        approximation.

        Neither p_des nor p_nominal is baked into any constructor-time
        cached quantity (H/the OSQP sparsity pattern depend only on
        Qbar/Rbar/Rdbar/Sp -- cost weights and prediction matrices -- never
        on reference values; the stagewise nullspace projectors depend only
        on the schedule/C/state_error_scale). Both are read fresh from
        `controller.reference`/`controller.nominal_reference_positions_m`
        on every solve, so mutating them here needs no re-warm."""
        from dataclasses import replace as dc_replace

        des_planner = np.asarray(controller.reference.desired_position_m, dtype=float)
        des_R = des_planner @ R_fit.T + t_fit
        replace_kwargs = {"desired_position_m": des_R}
        tan_planner = getattr(controller.reference, "desired_tangent", None)
        if tan_planner is not None:
            tan_R = np.asarray(tan_planner, dtype=float) @ R_fit.T  # rotation only, no translation
            replace_kwargs["desired_tangent"] = tan_R
        controller.reference = dc_replace(controller.reference, **replace_kwargs)
        controller.nominal_reference_positions_m = des_R.copy()

    # warm-up: one real solve of each kind, at the reference's own start
    # state, result discarded -- purely to force one-time lazy costs
    # (OSQP setup, BLAS thread-pool spin-up, etc.) to happen now.
    q0 = np.asarray(reference.state[0, :6], dtype=float)
    l0 = float(reference.state[0, 6])
    p0 = np.asarray(reference.desired_position_m[0], dtype=float)
    zeros7 = np.zeros(7, dtype=float)
    old_controller.solve(
        measured_state=np.concatenate([q0, [l0]]), measured_beam_position=p0,
        control_index=0, previous_input=zeros7,
    )
    new_controller.solve_delay_aware(
        z_meas=np.concatenate([q0, [l0]]), q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
        measured_beam_position=p0, control_index=0, previous_input=zeros7,
    )
    if null_controller is not None:
        null_controller.solve_delay_aware(
            z_meas=np.concatenate([q0, [l0]]), q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
            measured_beam_position=p0, control_index=0, previous_input=zeros7,
        )
    if exact_qn0_controller is not None:
        exact_qn0_controller.solve_delay_aware(
            z_meas=np.concatenate([q0, [l0]]), q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
            measured_beam_position=p0, control_index=0, previous_input=zeros7,
        )

    conn.send({"status": "ready"})

    # frame_ready gates real solves (per set_frame_transform's docstring
    # above): a live run's measured_beam_position is meaningless against
    # this worker's p_des/p_nominal until the live planner->robot
    # registration has been applied. Offline replay callers that never
    # send set_frame_transform (every offline script in this project's
    # history) get frame_ready=True unconditionally the first time ANY
    # solve request arrives with no transform pending -- see the check
    # below: only a LIVE caller that explicitly intends to use this gate
    # sends set_frame_transform first, so gating on "have we ever been
    # asked" rather than an explicit opt-in flag would break every
    # existing offline replay script. Instead: frame_ready starts True by
    # default (preserves all existing offline-replay behavior identically)
    # and is only ever forced back to a wait-state by a caller that
    # explicitly sends {"cmd": "require_frame_transform"} before its first
    # solve -- see MPCWorkerHandle.require_frame_transform().
    frame_ready = True

    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        if req is None:  # sentinel: shut down
            return

        cmd = req.get("cmd")
        if cmd == "require_frame_transform":
            frame_ready = False
            conn.send({"cmd_ack": "require_frame_transform", "status": "ok"})
            continue
        if cmd == "set_frame_transform":
            R_fit = np.asarray(req["R_fit"], dtype=float).reshape(3, 3)
            t_fit = np.asarray(req["t_fit"], dtype=float).reshape(3)
            orth_err = float(np.max(np.abs(R_fit.T @ R_fit - np.eye(3))))
            det = float(np.linalg.det(R_fit))
            if orth_err >= 1e-6 or abs(det - 1.0) >= 1e-6:
                conn.send({"cmd_ack": "set_frame_transform", "status": "error",
                           "error": f"R_fit invalid: orth_err={orth_err:.3e} det={det:.6f}"})
                continue
            _apply_frame_transform(old_controller, R_fit, t_fit)
            _apply_frame_transform(new_controller, R_fit, t_fit)
            if null_controller is not None:
                _apply_frame_transform(null_controller, R_fit, t_fit)
            if exact_qn0_controller is not None:
                _apply_frame_transform(exact_qn0_controller, R_fit, t_fit)
            frame_ready = True
            print(f"[worker] frame transform applied: |t_fit|={np.linalg.norm(t_fit)*1e3:.1f}mm "
                  f"det(R_fit)={det:+.6f} -- p_des/p_nominal now frame-consistent, frame_ready=True")
            conn.send({"cmd_ack": "set_frame_transform", "status": "ok"})
            continue

        if not frame_ready:
            conn.send({
                "seq": req.get("seq"), "status": "error", "command": None,
                "predicted_beam_positions": None, "t_solve_ms": float("nan"),
                "error": "frame transform required but not set -- call "
                         "MPCWorkerHandle.set_frame_transform() before the first live solve",
            })
            continue

        seq = req["seq"]
        try:
            if req.get("worker_sleep_s"):  # diagnostic-only artificial load injection
                time.sleep(float(req["worker_sleep_s"]))
            t0 = time.monotonic()
            if req["kind"] == "old":
                step = old_controller.solve(
                    measured_state=np.asarray(req["z_meas"], dtype=float),
                    measured_beam_position=np.asarray(req["measured_beam_position"], dtype=float),
                    control_index=int(req["control_index"]),
                    previous_input=np.asarray(req["previous_input"], dtype=float),
                )
            elif req["kind"] == "nullspace_r700":
                if null_controller is None:
                    raise RuntimeError(
                        "kind='nullspace_r700' requested but worker was not started with "
                        "enable_nullspace_r700=True"
                    )
                step = null_controller.solve_delay_aware(
                    z_meas=np.asarray(req["z_meas"], dtype=float),
                    q_cmd=np.asarray(req["q_cmd"], dtype=float),
                    q_cmd_prev=np.asarray(req["q_cmd_prev"], dtype=float),
                    insertion_m=float(req["insertion_cmd"]),
                    measured_beam_position=np.asarray(req["measured_beam_position"], dtype=float),
                    control_index=int(req["control_index"]),
                    previous_input=np.asarray(req["previous_input"], dtype=float),
                )
            elif req["kind"] == "exact_qn0":
                if exact_qn0_controller is None:
                    raise RuntimeError(
                        "kind='exact_qn0' requested but worker was not started with "
                        "enable_exact_qn_zero=True"
                    )
                step = exact_qn0_controller.solve_delay_aware(
                    z_meas=np.asarray(req["z_meas"], dtype=float),
                    q_cmd=np.asarray(req["q_cmd"], dtype=float),
                    q_cmd_prev=np.asarray(req["q_cmd_prev"], dtype=float),
                    insertion_m=float(req["insertion_cmd"]),
                    measured_beam_position=np.asarray(req["measured_beam_position"], dtype=float),
                    control_index=int(req["control_index"]),
                    previous_input=np.asarray(req["previous_input"], dtype=float),
                )
            else:
                step = new_controller.solve_delay_aware(
                    z_meas=np.asarray(req["z_meas"], dtype=float),
                    q_cmd=np.asarray(req["q_cmd"], dtype=float),
                    q_cmd_prev=np.asarray(req["q_cmd_prev"], dtype=float),
                    insertion_m=float(req["insertion_cmd"]),
                    measured_beam_position=np.asarray(req["measured_beam_position"], dtype=float),
                    control_index=int(req["control_index"]),
                    previous_input=np.asarray(req["previous_input"], dtype=float),
                )
            t_solve_ms = (time.monotonic() - t0) * 1e3
            conn.send({
                "seq": seq, "status": "ok" if step.success else "infeasible",
                "command": np.asarray(step.command, dtype=float),
                "predicted_beam_positions": np.asarray(step.predicted_beam_positions, dtype=float),
                "t_solve_ms": t_solve_ms, "error": None,
            })
        except Exception as exc:  # noqa: BLE001 -- must never crash silently, always reply
            conn.send({
                "seq": seq, "status": "error", "command": None,
                "predicted_beam_positions": None, "t_solve_ms": float("nan"), "error": repr(exc),
            })


class MPCWorkerHandle:
    """Parent-side handle: spawns the worker, sends one request at a time,
    enforces the deadline, and discards stale/mismatched responses. Owns NO
    hardware state itself -- q_cmd/q_cmd_prev/previous_input bookkeeping
    stays in the caller (`ProcessIsolatedDelayAwareAdapter` /
    `ProcessIsolatedBaselineAdapter`), per this module's docstring.

    Construct this BEFORE opening any RTDE/camera connection -- see the
    module docstring's lifecycle-ordering note. `t_spawn_start`/
    `t_worker_ready` (`time.monotonic()`) are recorded for lifecycle
    logging.
    """

    def __init__(
        self, *, plan_dir: str, schedule_path: str, mpc_config_kwargs: dict,
        beam_config_kwargs: dict, delay_samples: int = 2, beta_d: float = 1.0,
        single_threaded: bool = True, startup_timeout_s: float = 30.0,
        enable_nullspace_r700: bool = False,
        nullspace_r700_config_kwargs: Optional[dict] = None,
        insertion_state_anchor_weight: float = 0.0,
        enable_exact_qn_zero: bool = False,
        exact_qn_zero_config_kwargs: Optional[dict] = None,
    ) -> None:
        self.t_spawn_start = time.monotonic()
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_worker_main, daemon=True,
            kwargs=dict(
                conn=child_conn, plan_dir=plan_dir, schedule_path=schedule_path,
                mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
                delay_samples=delay_samples, beta_d=beta_d, single_threaded=single_threaded,
                enable_nullspace_r700=enable_nullspace_r700,
                nullspace_r700_config_kwargs=nullspace_r700_config_kwargs,
                insertion_state_anchor_weight=insertion_state_anchor_weight,
                enable_exact_qn_zero=enable_exact_qn_zero,
                exact_qn_zero_config_kwargs=exact_qn_zero_config_kwargs,
            ),
        )
        self._proc.start()
        self._seq = 0
        self._pending_seq: Optional[int] = None
        if not self._parent_conn.poll(startup_timeout_s):
            raise RuntimeError("MPC worker process did not start in time")
        ready = self._parent_conn.recv()
        if ready.get("status") != "ready":
            raise RuntimeError(f"MPC worker process failed to start: {ready}")
        self.t_worker_ready = time.monotonic()

    def require_frame_transform(self, *, timeout_s: float = 5.0) -> None:
        """Force the worker to refuse any solve until `set_frame_transform`
        is called -- opt-in, so every existing offline replay caller that
        never calls this (or set_frame_transform) is completely unaffected.
        Call this once, right after spawn/warm-up, for any LIVE run whose
        p_des/p_nominal need the live planner->robot registration applied
        before they're meaningful (see `_apply_frame_transform`'s
        docstring in `_worker_main`)."""
        self._parent_conn.send({"cmd": "require_frame_transform"})
        if not self._parent_conn.poll(timeout_s):
            raise RuntimeError("worker did not ack require_frame_transform in time")
        resp = self._parent_conn.recv()
        if resp.get("status") != "ok":
            raise RuntimeError(f"require_frame_transform failed: {resp}")

    def set_frame_transform(self, R_fit: Array, t_fit: Array, *, timeout_s: float = 5.0) -> None:
        """One-time, blocking: apply the live planner(P)->robot(R)
        registration to every controller's p_des/p_nominal (leaves the
        schedule/J/projectors/warm-up path untouched -- see
        `_apply_frame_transform`). Call this exactly once, after preflight
        computes R_fit/t_fit, before the first real control tick. Must not
        be called concurrently with `try_solve` (this uses the same pipe
        with a direct blocking send/recv, not the one-outstanding-request
        protocol) -- safe because nothing else talks to the worker during
        this one-time startup handshake."""
        R_fit = np.asarray(R_fit, dtype=float).reshape(3, 3)
        t_fit = np.asarray(t_fit, dtype=float).reshape(3)
        self._parent_conn.send({
            "cmd": "set_frame_transform",
            "R_fit": R_fit.tolist(), "t_fit": t_fit.tolist(),
        })
        if not self._parent_conn.poll(timeout_s):
            raise RuntimeError("worker did not ack set_frame_transform in time")
        resp = self._parent_conn.recv()
        if resp.get("status") != "ok":
            raise RuntimeError(f"set_frame_transform failed: {resp}")

    def try_solve(
        self, *, kind: str, z_meas: Array, measured_beam_position: Array,
        control_index: int, previous_input: Array, deadline_s: float,
        q_cmd: Optional[Array] = None, q_cmd_prev: Optional[Array] = None,
        insertion_cmd: Optional[float] = None, worker_sleep_s: float = 0.0,
    ) -> Optional[dict]:
        """Enforces AT MOST ONE outstanding request at a time (a QP solved
        against a stale snapshot is worse than not solving at all -- see
        module docstring). If a PRIOR request's response is still
        outstanding:

          - it may have arrived since the last call -- drain it (it's
            necessarily stale now, since we're past its own deadline, so
            it's discarded either way, never applied) -- then this tick
            proceeds to send its own fresh request;
          - if the worker is STILL working on it, this tick does NOT
            enqueue a second problem behind the first -- it returns None
            immediately (a miss) without sending anything, leaving the
            prior request outstanding for a future tick to drain.

        Returns the matching response dict on a hit within `deadline_s`,
        else None (miss -- timeout, stale seq, or worker still busy on an
        older request). The caller's fallback (apply u=0, hold q_cmd) is
        identical for every None case.
        """
        if self._pending_seq is not None:
            if self._parent_conn.poll(0.0):
                stale = self._parent_conn.recv()
                if stale.get("seq") == self._pending_seq:
                    self._pending_seq = None  # drained -- worker is free again
                # a mismatched seq here would mean two responses queued up,
                # which try_solve's one-outstanding-request invariant should
                # make impossible; left undrained is not silently ignored --
                # the next call's poll(0.0) will surface it.
            if self._pending_seq is not None:
                return None  # worker still busy -- do not enqueue a second problem

        self._seq += 1
        seq = self._seq
        self._pending_seq = seq
        self._parent_conn.send({
            "seq": seq, "kind": kind,
            "z_meas": np.asarray(z_meas, dtype=float),
            "q_cmd": None if q_cmd is None else np.asarray(q_cmd, dtype=float),
            "q_cmd_prev": None if q_cmd_prev is None else np.asarray(q_cmd_prev, dtype=float),
            "insertion_cmd": insertion_cmd,
            "measured_beam_position": np.asarray(measured_beam_position, dtype=float),
            "control_index": int(control_index),
            "previous_input": np.asarray(previous_input, dtype=float),
            "worker_sleep_s": float(worker_sleep_s),
        })

        if not self._parent_conn.poll(deadline_s):
            return None  # miss -- seq stays pending, drained by a future call
        resp = self._parent_conn.recv()
        if resp.get("seq") != seq:
            return None  # shouldn't happen given the invariant above; treat as a miss
        self._pending_seq = None
        return resp

    def close(self) -> None:
        try:
            self._parent_conn.send(None)
        except Exception:
            pass
        self._proc.join(timeout=2.0)
        if self._proc.is_alive():
            self._proc.terminate()
