"""OnlineMPCRunner: tie the three streaming components to the existing MPC.

This is the orchestrator that replaces ``run_hardware_control_optimized``'s
stop-and-go loop.  It owns nothing about vision, robot or serial protocols - that
lives in :class:`CameraSource`, :class:`RobotSink` and :class:`AdvancerSink` - and
nothing about the controller maths, which stays in the existing MPC / Jacobian
code.  It only:

1. reads the freshest :class:`StateEstimate` from the camera;
2. calls a user-supplied ``solve`` callback (which wraps the existing
   ``mpc.step`` / Jacobian controller) to get the rate command ``u0``;
3. publishes a :class:`ControlCommand` to the robot and feeds ``u0[6]`` to the
   advancer;
4. runs a safety supervisor that stops everything on stale vision, a dead
   worker, a solver fault or an explicit abort.

The solve callback is injected so this file never hard-codes the controller's
signature (which differs between MPC variants):

    def solve(estimate: StateEstimate, dt: float) -> SolveResult | np.ndarray: ...

Return either a 7-vector ``u0`` or a :class:`SolveResult` carrying ``u0`` plus
optional ``p_commanded`` / ``infeasible`` for logging.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np

from .advancer_sink import AdvancerSink
from .camera_source import CameraSource
from .messages import (
    ControlCommand,
    HeartbeatLoop,
    StateEstimate,
    now_monotonic,
)
from .robot_sink import RobotSink

__all__ = ["SolveResult", "OnlineMPCConfig", "OnlineMPCRunner"]


@dataclass
class SolveResult:
    u0: np.ndarray
    p_commanded: Optional[np.ndarray] = None
    infeasible: bool = False
    info: dict = field(default_factory=dict)


SolveFn = Callable[[StateEstimate, float], Union[SolveResult, np.ndarray]]


@dataclass
class OnlineMPCConfig:
    dt: float = 0.1                     # control period fed to the solver / integrators
    max_state_age_s: float = 0.30       # stale vision -> abort
    max_solver_time_s: float = 0.20     # a solve slower than this is logged as overrun
    require_robot_pose: bool = False    # abort if the estimate has no robot pose
    max_steps: Optional[int] = None
    max_duration_s: Optional[float] = None
    zero_command_on_infeasible: bool = True


class OnlineMPCRunner:
    def __init__(
        self,
        *,
        camera: CameraSource,
        robot: RobotSink,
        advancer: AdvancerSink,
        solve: SolveFn,
        config: OnlineMPCConfig,
        on_abort: Optional[Callable[[str], None]] = None,
        on_step: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.camera = camera
        self.robot = robot
        self.advancer = advancer
        self._solve = solve
        self.config = config
        self._on_abort = on_abort
        self._on_step = on_step

        self._loop: Optional[HeartbeatLoop] = None
        self._supervisor: Optional[HeartbeatLoop] = None
        self._done = threading.Event()

        self.aborted = False
        self.abort_reason = ""
        self.steps = 0
        self.solver_overruns = 0
        self.last_u0 = np.zeros(7)
        self._started_at = 0.0
        self._seq = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.camera.start()
        self.robot.start()
        self.advancer.start()
        self._started_at = now_monotonic()
        self._done.clear()

        self._loop = HeartbeatLoop(
            "mpc-control", self._control_body, period_s=self.config.dt,
            on_error=lambda exc: self.abort(f"control_thread_error: {exc!r}"),
        )
        self._supervisor = HeartbeatLoop(
            "mpc-supervisor", self._supervise_body, period_s=0.05,
        )
        self._supervisor.start()
        self._loop.start()

    def stop(self) -> None:
        self._done.set()
        for loop in (self._loop, self._supervisor):
            if loop is not None:
                loop.stop()
        self._loop = None
        self._supervisor = None
        # Stop actuators first, then release devices.
        self.robot.stop()
        self.advancer.stop()
        self.camera.stop()

    def run(self) -> str:
        """Start, block until done/aborted, stop. Returns the stop reason."""
        self.start()
        try:
            self._done.wait()
        except KeyboardInterrupt:
            self.abort("keyboard_interrupt")
        finally:
            self.stop()
        return self.abort_reason or "completed"

    def __enter__(self) -> "OnlineMPCRunner":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def abort(self, reason: str) -> None:
        if not self.aborted:
            self.aborted = True
            self.abort_reason = reason
        self.robot.emergency_stop(reason)
        self.advancer.submit_distance(0.0)
        self._done.set()
        if self._on_abort is not None:
            try:
                self._on_abort(reason)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # control thread
    # ------------------------------------------------------------------
    def _control_body(self, t_start: float) -> None:
        cfg = self.config
        if self._done.is_set():
            return

        estimate, age = self.camera.latest(cfg.max_state_age_s)
        if estimate is None:
            self.abort(f"stale_or_missing_state(age={age:.3f}s)")
            return
        if not estimate.valid:
            self.abort("invalid_state_estimate")
            return
        if cfg.require_robot_pose and estimate.robot_pose6 is None:
            self.abort("state_estimate_without_robot_pose")
            return

        solve_started = now_monotonic()
        result = self._solve(estimate, cfg.dt)
        solve_ms = 1.0e3 * (now_monotonic() - solve_started)
        if solve_ms > 1.0e3 * cfg.max_solver_time_s:
            self.solver_overruns += 1

        if isinstance(result, SolveResult):
            u0 = np.asarray(result.u0, dtype=float).reshape(-1)
            p_commanded = result.p_commanded
            infeasible = bool(result.infeasible)
            info = result.info
        else:
            u0 = np.asarray(result, dtype=float).reshape(-1)
            p_commanded = None
            infeasible = False
            info = {}

        if u0.size < 7:
            u0 = np.pad(u0, (0, 7 - u0.size))
        if not np.all(np.isfinite(u0)):
            self.abort("nonfinite_u0")
            return
        if infeasible and cfg.zero_command_on_infeasible:
            u0 = np.zeros_like(u0)

        self._seq += 1
        command = ControlCommand(
            t_monotonic=now_monotonic(),
            u0=u0,
            dt=cfg.dt,
            p_commanded=None if p_commanded is None
            else np.asarray(p_commanded, dtype=float),
            sequence=self._seq,
            infeasible=infeasible,
        )
        self.robot.submit(command)
        self.advancer.submit_rate(float(u0[6]), cfg.dt)

        self.last_u0 = u0
        self.steps += 1

        if self._on_step is not None:
            try:
                self._on_step(
                    {
                        "step": self.steps,
                        "t_monotonic": now_monotonic(),
                        "state_age_s": age,
                        "solve_ms": solve_ms,
                        "u0": u0.copy(),
                        "infeasible": infeasible,
                        "estimate": estimate,
                        "solver_info": info,
                    }
                )
            except Exception:
                pass

        if cfg.max_steps is not None and self.steps >= cfg.max_steps:
            self.abort_reason = self.abort_reason or "max_steps"
            self._done.set()

    # ------------------------------------------------------------------
    # safety supervisor
    # ------------------------------------------------------------------
    def _supervise_body(self, t_now: float) -> None:
        cfg = self.config
        if self._done.is_set():
            return
        if cfg.max_duration_s is not None and (t_now - self._started_at) > cfg.max_duration_s:
            self.abort_reason = self.abort_reason or "max_duration"
            self._done.set()
            return
        if not self.camera.healthy:
            self.abort(f"camera_unhealthy: {self.camera.last_error or 'thread down'}")
            return
        if not self.robot.healthy:
            self.abort("robot_stream_unhealthy")
            return
        if not self.advancer.healthy:
            self.abort("advancer_unhealthy")
            return
        if self._loop is not None and self._loop.error is not None:
            self.abort(f"control_loop_error: {self._loop.error!r}")
