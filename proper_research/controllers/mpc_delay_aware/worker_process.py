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
) -> None:
    """Entry point for Process B. Constructs its OWN MPC instances (both
    kinds, so one worker can answer requests for either condition -- the
    live A/B alternates between them run-to-run, not tick-to-tick), then
    runs ONE dummy solve of each kind before signaling ready -- pays lazy
    OSQP/BLAS initialization cost here, not on the first real control tick.
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

    conn.send({"status": "ready"})

    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        if req is None:  # sentinel: shut down
            return

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
