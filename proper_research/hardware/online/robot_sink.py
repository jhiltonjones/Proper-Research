"""RobotSink: one RTDE session, one streaming ``speedL``/``speedJ`` thread, one watchdog.

The old loop connected, called a blocking ``moveL``, and disconnected — every
step.  ``RobotSink`` keeps a single :class:`URRTDERobot` session for the whole
run and streams the controller's command to the arm:

* a thread runs at ``robot.frequency`` (``initPeriod`` / ``waitPeriod``),
  reads the latest :class:`ControlCommand` from a :class:`LatestSlot`, clamps it
  to configured speed limits, and calls ``speed_l(u0[:6])`` (Cartesian twist,
  the default) or ``speed_j(u0[:6])`` (six joint velocities, when
  ``config.control_mode == "joint"`` — what the joint-space offline
  controllers produce), non-blocking either way;
* if the newest command is older than ``max_command_age_s`` — MPC stalled, or a
  producer died — the **staleness watchdog** calls ``speed_stop`` and keeps the
  arm stopped until fresh commands resume;
* ``dry_run`` computes and records everything but never touches RTDE (the old
  ``send_commands=False`` shadow mode).

The MPC/orchestrator only ever calls :meth:`submit` with a new command; it never
blocks on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from .messages import ControlCommand, HeartbeatLoop, LatestSlot, now_monotonic

if TYPE_CHECKING:  # avoid importing rtde_control/rtde_receive on non-robot machines
    from ..ur_rtde_robot import URRTDERobot

__all__ = ["RobotSinkConfig", "RobotSink"]


@dataclass
class RobotSinkConfig:
    control_frequency_hz: float = 125.0
    acceleration: float = 0.25          # speedL / speedStop accel (m/s^2)
    max_linear_speed_m_s: float = 0.05  # per-axis clamp on u0[:3]
    max_angular_speed_rad_s: float = 0.30  # per-axis clamp on u0[3:6]
    max_command_age_s: float = 0.20     # staleness watchdog threshold
    watchdog_min_frequency_hz: float = 10.0  # RTDE-side backstop
    stop_acceleration: float = 2.0
    require_motion_ready: bool = True
    dry_run: bool = True

    # "cartesian" streams speedL(u0[:6]) as a base-frame twist (the default,
    # magnet-pose-space controllers). "joint" streams speedJ(u0[:6]) as six
    # joint velocities instead — what the joint-space offline controllers
    # (inverse Jacobian, LTI / LTV beam-output MPC) produce. Either way the
    # staleness watchdog and speed_stop() shutdown behave identically.
    control_mode: str = "cartesian"
    max_joint_speed_rad_s: float = 0.6      # per-axis clamp on u0[:6] in joint mode
    joint_acceleration: float = 0.8         # speedJ accel (rad/s^2)

    # Optional Cartesian workspace guard on the *measured* TCP position.
    # Only applied in "cartesian" mode.
    xyz_min: Optional[tuple] = None
    xyz_max: Optional[tuple] = None

    def __post_init__(self) -> None:
        if self.control_mode not in ("cartesian", "joint"):
            raise ValueError(
                f"control_mode must be 'cartesian' or 'joint'; got {self.control_mode!r}"
            )


class RobotSink:
    def __init__(
        self,
        robot: "URRTDERobot",
        config: RobotSinkConfig,
        *,
        command_slot: Optional[LatestSlot[ControlCommand]] = None,
        on_safety_stop: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self.robot = robot
        self.config = config
        self.command_slot: LatestSlot[ControlCommand] = command_slot or LatestSlot()
        self._on_safety_stop = on_safety_stop
        self._on_error = on_error

        self._loop: Optional[HeartbeatLoop] = None
        self._active = False  # True while a non-zero twist is being streamed

        # telemetry
        self.commands_sent = 0
        self.watchdog_trips = 0
        self.last_twist_sent = np.zeros(6)
        self.last_joint_velocity_sent = np.zeros(6)
        self.last_command_seq = -1
        self.last_pose6: Optional[np.ndarray] = None
        self.watchdog_tripped = False
        self.last_stop_reason = ""

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        cfg = self.config
        if not cfg.dry_run:
            if not self.robot.is_connected():
                self.robot.connect()
            if cfg.require_motion_ready:
                self.robot._require_motion_ready_unlocked()  # raises if not ready
            self.robot.set_watchdog(cfg.watchdog_min_frequency_hz)

        period = 1.0 / float(cfg.control_frequency_hz)
        self._loop = HeartbeatLoop(
            "robot-stream", self._stream_body, period_s=period, on_error=self._on_fault
        )
        self._loop.start()

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.stop()
            self._loop = None
        self._safe_speed_stop("shutdown")

    def __enter__(self) -> "RobotSink":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # producer API
    # ------------------------------------------------------------------
    def submit(self, command: ControlCommand) -> None:
        """Publish a new MPC command for the streaming thread to apply."""
        self.command_slot.set(command)

    def emergency_stop(self, reason: str = "external") -> None:
        self._safe_speed_stop(reason)

    def latest_tcp_pose(self) -> Optional[np.ndarray]:
        """Cheap accessor for CameraSource; returns the last streamed-loop read."""
        return None if self.last_pose6 is None else self.last_pose6.copy()

    @property
    def healthy(self) -> bool:
        return self._loop is not None and self._loop.alive and self._loop.error is None

    # ------------------------------------------------------------------
    # streaming thread
    # ------------------------------------------------------------------
    def _stream_body(self, _t_start: float) -> None:
        cfg = self.config
        t_period = None
        if not cfg.dry_run:
            t_period = self.robot.init_period()
            try:
                self.last_pose6 = np.asarray(self.robot.get_tcp_pose(), dtype=float)
            except Exception:
                pass

        command, age = self.command_slot.get(cfg.max_command_age_s)

        if command is None:
            # Stale or absent command -> stop and hold.
            if self._active or not self.watchdog_tripped:
                self._safe_speed_stop(f"stale_command(age={age:.3f}s)")
                self.watchdog_trips += 1
                self.watchdog_tripped = True
            self._wait_period(t_period)
            return

        self.watchdog_tripped = False

        if cfg.control_mode == "joint":
            joint_velocity = self._clamp_joint(command.joint_velocity)

            self.last_joint_velocity_sent = joint_velocity
            self.last_command_seq = command.sequence
            self._active = bool(np.any(np.abs(joint_velocity) > 0.0))

            if not cfg.dry_run:
                self.robot.speed_j(
                    joint_velocity, acceleration=cfg.joint_acceleration, time_s=0.0
                )
            self.commands_sent += 1
            self._wait_period(t_period)
            return

        twist = self._clamp_twist(command.cartesian_twist)

        # Optional workspace guard on the measured TCP position.
        if self.last_pose6 is not None and not self._within_workspace(self.last_pose6):
            self._safe_speed_stop("tcp_out_of_workspace")
            self._wait_period(t_period)
            return

        self.last_twist_sent = twist
        self.last_command_seq = command.sequence
        self._active = bool(np.any(np.abs(twist) > 0.0))

        if not cfg.dry_run:
            self.robot.speed_l(twist, acceleration=cfg.acceleration, time_s=0.0)
        self.commands_sent += 1
        self._wait_period(t_period)

    def _wait_period(self, t_period: Optional[float]) -> None:
        if t_period is not None:
            try:
                self.robot.wait_period(t_period)
            except Exception:
                pass

    def _clamp_twist(self, twist: np.ndarray) -> np.ndarray:
        cfg = self.config
        out = np.asarray(twist, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(out)):
            return np.zeros(6)
        lin = float(cfg.max_linear_speed_m_s)
        ang = float(cfg.max_angular_speed_rad_s)
        out[:3] = np.clip(out[:3], -lin, lin)
        out[3:6] = np.clip(out[3:6], -ang, ang)
        return out

    def _clamp_joint(self, joint_velocity: np.ndarray) -> np.ndarray:
        cfg = self.config
        out = np.asarray(joint_velocity, dtype=float).reshape(6).copy()
        if not np.all(np.isfinite(out)):
            return np.zeros(6)
        return np.clip(out, -float(cfg.max_joint_speed_rad_s), float(cfg.max_joint_speed_rad_s))

    def _within_workspace(self, pose6: np.ndarray) -> bool:
        cfg = self.config
        xyz = np.asarray(pose6, dtype=float).reshape(6)[:3]
        if cfg.xyz_min is not None and np.any(xyz < np.asarray(cfg.xyz_min, float)):
            return False
        if cfg.xyz_max is not None and np.any(xyz > np.asarray(cfg.xyz_max, float)):
            return False
        return True

    def _safe_speed_stop(self, reason: str) -> None:
        self._active = False
        self.last_twist_sent = np.zeros(6)
        self.last_joint_velocity_sent = np.zeros(6)
        self.last_stop_reason = reason
        if not self.config.dry_run:
            try:
                # speed_stop() decelerates whichever of speedL/speedJ is active.
                self.robot.speed_stop(self.config.stop_acceleration)
            except Exception:
                pass
        if self._on_safety_stop is not None:
            try:
                self._on_safety_stop(reason)
            except Exception:
                pass

    def _on_fault(self, exc: BaseException) -> None:
        self._safe_speed_stop(f"stream_thread_error: {exc!r}")
        if self._on_error is not None:
            try:
                self._on_error(exc)
            except Exception:
                pass
