"""Shared plumbing for the online (streaming) hardware stack.

The online stack replaces the old stop-and-go loop with three long-lived
producers/consumers that each own one hardware resource for the whole run:

    CameraSource  -> publishes the latest StateEstimate  (vision in the loop)
    RobotSink     -> consumes the latest ControlCommand   (streaming speedL)
    AdvancerSink  -> consumes an integrated insertion rate (fire-and-forget)

They communicate through :class:`LatestSlot`, a single-slot latest-value-wins
buffer.  Unlike a queue it never backs up: a slow consumer simply misses the
intermediate values and always sees the freshest one, and every write carries a
``time.monotonic()`` stamp so a consumer can reject stale data and a supervisor
can trip a watchdog.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, Optional, Tuple, TypeVar

import numpy as np

__all__ = [
    "now_monotonic",
    "LatestSlot",
    "HeartbeatLoop",
    "StateEstimate",
    "ControlCommand",
    "AdvancerFeedback",
]


def now_monotonic() -> float:
    """Single clock source for every timestamp in the online stack."""
    return time.monotonic()


T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Thread-safe single-slot buffer: one writer, any number of readers.

    ``set`` overwrites whatever is there; ``get`` returns the current value plus
    its age in seconds.  There is no history and no back-pressure.
    """

    __slots__ = ("_lock", "_value", "_stamp", "_seq", "_fresh")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._stamp: Optional[float] = None
        self._seq: int = 0
        self._fresh = threading.Event()

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._stamp = now_monotonic()
            self._seq += 1
        self._fresh.set()

    def get(self, max_age_s: Optional[float] = None) -> Tuple[Optional[T], float]:
        """Return ``(value, age_s)``.

        If the slot is empty, or ``max_age_s`` is given and the value is older
        than that, return ``(None, age_s)`` (age is ``inf`` when empty).
        """
        with self._lock:
            value = self._value
            stamp = self._stamp
        if value is None or stamp is None:
            return None, float("inf")
        age = now_monotonic() - stamp
        if max_age_s is not None and age > float(max_age_s):
            return None, age
        return value, age

    def peek(self) -> Tuple[Optional[T], int]:
        """Return ``(value, sequence)`` ignoring age (sequence increments per set)."""
        with self._lock:
            return self._value, self._seq

    def wait_fresh(self, timeout: Optional[float] = None) -> Optional[T]:
        """Block until the next ``set`` (or ``timeout``); return the value or None.

        Only intended for a single consumer.  Consumers that just want the most
        recent value should poll :meth:`get` instead.
        """
        got = self._fresh.wait(timeout)
        if not got:
            return None
        self._fresh.clear()
        with self._lock:
            return self._value


class HeartbeatLoop:
    """A daemon thread running ``body()`` on a fixed period with a heartbeat.

    ``body`` is called with the loop's start-of-iteration monotonic time.  Any
    exception is captured in :attr:`error` and stops the loop; :attr:`last_beat`
    lets a supervisor detect a hung or dead worker.
    """

    def __init__(
        self,
        name: str,
        body: Callable[[float], None],
        *,
        period_s: float,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        if period_s <= 0.0:
            raise ValueError("period_s must be > 0")
        self.name = str(name)
        self._body = body
        self._period_s = float(period_s)
        self._on_error = on_error
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.error: Optional[BaseException] = None
        self.last_beat: float = 0.0
        self.iterations: int = 0

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"{self.name} already started")
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            t_start = now_monotonic()
            self.last_beat = t_start
            try:
                self._body(t_start)
            except BaseException as exc:  # noqa: BLE001 - surfaced via .error
                self.error = exc
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:
                        pass
                return
            self.iterations += 1
            elapsed = now_monotonic() - t_start
            remaining = self._period_s - elapsed
            if remaining > 0.0:
                self._stop.wait(remaining)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stale_for(self, ref: Optional[float] = None) -> float:
        ref = now_monotonic() if ref is None else ref
        return ref - self.last_beat if self.last_beat else float("inf")


@dataclass(frozen=True)
class StateEstimate:
    """Latest fused estimate of the beam tip / robot state for the controller.

    ``x_meas`` is the raw vision output vector as produced by
    ``vision_result_to_x_meas_robot`` (at least ``[tip_xyz(3), tangent_xyz(3)]``,
    possibly longer).  ``tip_position_m`` / ``tip_tangent`` are the first two
    triples split out for convenience.

    ``robot_joints`` is the concurrent 6-vector of UR joint angles (rad), sampled
    with the frame.  The joint-space offline controllers (inverse Jacobian, LTI /
    LTV beam-output MPC) need it, together with ``insertion_length_m``, to build
    their measured state ``z = [q1..q6, insertion]``.
    """

    t_monotonic: float
    x_meas: np.ndarray
    tip_position_m: np.ndarray
    tip_tangent: np.ndarray
    insertion_length_m: float = float("nan")
    robot_pose6: Optional[np.ndarray] = None
    robot_joints: Optional[np.ndarray] = None
    source: str = "vision"
    frame_index: int = -1
    valid: bool = True
    vision_beam_length_mm: float = float("nan")

    def age(self, ref: Optional[float] = None) -> float:
        return (now_monotonic() if ref is None else ref) - self.t_monotonic


@dataclass(frozen=True)
class ControlCommand:
    """One MPC decision: a base-frame twist plus an insertion rate.

    ``u0`` is the MPC rate vector ``[vx, vy, vz, wx, wy, wz, dL_rate]`` in
    m/s, rad/s and m/s.  ``dt`` is the MPC step it was solved for.

    When the :class:`RobotSink` is configured with ``control_mode="joint"`` (the
    joint-space offline controllers), ``u0[:6]`` instead carries six joint
    velocities in rad/s and is read through :attr:`joint_velocity`.
    """

    t_monotonic: float
    u0: np.ndarray
    dt: float
    p_commanded: Optional[np.ndarray] = None
    sequence: int = -1
    infeasible: bool = False

    @classmethod
    def zero(cls, dt: float = 0.1) -> "ControlCommand":
        return cls(t_monotonic=now_monotonic(), u0=np.zeros(7), dt=float(dt))

    @property
    def cartesian_twist(self) -> np.ndarray:
        """``[vx, vy, vz, wx, wy, wz]`` — the part that goes to ``speedL``."""
        return np.asarray(self.u0, dtype=float).reshape(-1)[:6].copy()

    @property
    def joint_velocity(self) -> np.ndarray:
        """``[qd1..qd6]`` (rad/s) — the part that goes to ``speedJ`` in joint mode."""
        return np.asarray(self.u0, dtype=float).reshape(-1)[:6].copy()

    @property
    def insertion_rate_m_s(self) -> float:
        u = np.asarray(self.u0, dtype=float).reshape(-1)
        return float(u[6]) if u.size > 6 else 0.0


@dataclass(frozen=True)
class AdvancerFeedback:
    """Async status of the linear advancer, published by :class:`AdvancerSink`."""

    t_monotonic: float
    commanded_distance_mm: float = 0.0
    dispatched_distance_mm: float = 0.0
    steps_dispatched_total: int = 0
    residual_mm: float = 0.0
    commands_in_flight: int = 0
    last_status: str = ""
    steps_reported_total: int = 0
    error: str = ""
