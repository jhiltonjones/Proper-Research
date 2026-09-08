"""AdvancerSink: one serial connection, fire-and-forget incremental steps.

The old ``AdvancerUnit`` opened the serial port (with a 2 s Arduino reset), put a
command on a queue, then called ``queue.join()`` — blocking the whole control
loop until the firmware answered ``DONE``.

``AdvancerSink``:

* opens ``serial.Serial`` once and keeps it open for the run;
* takes the MPC insertion **rate** ``u0[6]`` (m/s) each tick via :meth:`submit_rate`
  and integrates it into a commanded distance;
* a writer thread dispatches whole ``mm_per_step`` quanta as ``ON``/``REV``
  commands and returns immediately — no ``DONE`` handshake, no ``queue.join()``;
* a reader thread consumes firmware lines asynchronously, clears the in-flight
  flag on ``DONE``/``INTERRUPTED`` and parses any step-count telemetry;
* the sub-quantum remainder is carried forward so slow insertion still advances;
* ``dry_run`` integrates and publishes feedback but never opens the port.

Feedback is published to :attr:`feedback_slot` as :class:`AdvancerFeedback`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .messages import (
    AdvancerFeedback,
    HeartbeatLoop,
    LatestSlot,
    now_monotonic,
)

__all__ = ["AdvancerSinkConfig", "AdvancerSink"]


@dataclass
class AdvancerSinkConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    serial_timeout_s: float = 0.05
    reset_settle_s: float = 2.0
    mm_per_step: float = 0.166
    min_command_mm: float = 0.166      # smallest travel worth a serial command
    max_command_mm: float = 5.0        # clamp a single dispatched increment
    delay_us: int = 20                 # per-step delay passed to the firmware
    max_rate_m_s: float = 0.01         # clamp on |submitted rate|
    dispatch_period_s: float = 0.02    # writer thread cadence (~50 Hz)
    in_flight_timeout_s: float = 1.0   # assume a lost DONE after this long
    done_tokens: tuple = ("DONE", "INTERRUPTED")
    dry_run: bool = True


class AdvancerSink:
    def __init__(
        self,
        config: AdvancerSinkConfig,
        *,
        feedback_slot: Optional[LatestSlot[AdvancerFeedback]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        self.config = config
        self.feedback_slot: LatestSlot[AdvancerFeedback] = feedback_slot or LatestSlot()
        self._on_error = on_error

        self._serial = None
        self._state_lock = threading.Lock()

        # integrator state (mm, signed; +forward)
        self._commanded_mm = 0.0     # cumulative integral of the rate
        self._dispatched_mm = 0.0    # cumulative travel actually sent as steps
        self._residual_mm = 0.0      # not-yet-dispatched remainder
        self._steps_dispatched = 0

        # async firmware feedback
        self._in_flight = 0
        self._last_dispatch_time = 0.0
        self._last_status = ""
        self._steps_reported = 0
        self._error = ""

        self._writer: Optional[HeartbeatLoop] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        cfg = self.config
        if not cfg.dry_run:
            import serial

            self._serial = serial.Serial(
                cfg.port, baudrate=cfg.baudrate, timeout=cfg.serial_timeout_s
            )
            time.sleep(cfg.reset_settle_s)  # Arduino auto-reset on port open
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            self._reader_stop.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_run, name="advancer-reader", daemon=True
            )
            self._reader_thread.start()

        self._writer = HeartbeatLoop(
            "advancer-writer",
            self._dispatch_body,
            period_s=cfg.dispatch_period_s,
            on_error=self._on_error,
        )
        self._writer.start()
        self._publish_feedback()

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.stop()
            self._writer = None
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            finally:
                self._serial = None

    def __enter__(self) -> "AdvancerSink":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # producer API
    # ------------------------------------------------------------------
    def submit_rate(self, insertion_rate_m_s: float, dt: float) -> None:
        """Integrate one MPC tick of insertion rate into the commanded distance."""
        cfg = self.config
        rate = float(insertion_rate_m_s)
        if rate != rate:  # NaN
            return
        rate = max(-cfg.max_rate_m_s, min(cfg.max_rate_m_s, rate))
        delta_mm = rate * float(dt) * 1000.0
        with self._state_lock:
            self._commanded_mm += delta_mm
            self._residual_mm += delta_mm

    def submit_distance(self, distance_mm: float) -> None:
        """Directly queue a signed travel increment (bypasses rate integration)."""
        d = float(distance_mm)
        with self._state_lock:
            self._commanded_mm += d
            self._residual_mm += d

    def feedback(self) -> AdvancerFeedback:
        with self._state_lock:
            return self._snapshot_locked()

    @property
    def healthy(self) -> bool:
        return self._writer is not None and self._writer.alive and self._writer.error is None

    # ------------------------------------------------------------------
    # writer thread: dispatch whole quanta, fire-and-forget
    # ------------------------------------------------------------------
    def _dispatch_body(self, _t_start: float) -> None:
        cfg = self.config

        with self._state_lock:
            residual = self._residual_mm
            in_flight = self._in_flight
            last_dispatch = self._last_dispatch_time

        # Respect the firmware's one-command-at-a-time nature: wait for the async
        # DONE, but do not wait forever if a reply was dropped.
        if in_flight > 0 and (now_monotonic() - last_dispatch) < cfg.in_flight_timeout_s:
            return

        if abs(residual) < cfg.min_command_mm:
            return

        n_quanta = int(residual / cfg.mm_per_step)  # trunc toward zero
        if n_quanta == 0:
            return
        move_mm = n_quanta * cfg.mm_per_step
        if abs(move_mm) > cfg.max_command_mm:
            n_quanta = int(cfg.max_command_mm / cfg.mm_per_step)
            n_quanta = n_quanta if residual > 0 else -n_quanta
            move_mm = n_quanta * cfg.mm_per_step

        steps = abs(n_quanta)
        token = "ON" if n_quanta > 0 else "REV"
        line = f"{token} {steps} {cfg.delay_us}\n"

        sent = True
        if not cfg.dry_run:
            sent = self._write_line(line)

        if sent:
            with self._state_lock:
                self._residual_mm -= move_mm
                self._dispatched_mm += move_mm
                self._steps_dispatched += steps
                self._last_dispatch_time = now_monotonic()
                # On real hardware wait for the async DONE; in dry-run there is
                # no firmware, so the step is instantly "complete".
                if not cfg.dry_run:
                    self._in_flight += 1
        self._publish_feedback()

    def _write_line(self, line: str) -> bool:
        ser = self._serial
        if ser is None:
            return False
        try:
            ser.write(line.encode("utf-8"))
            return True
        except Exception as exc:  # noqa: BLE001
            with self._state_lock:
                self._error = repr(exc)
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    pass
            return False

    # ------------------------------------------------------------------
    # reader thread: consume firmware lines asynchronously
    # ------------------------------------------------------------------
    def _reader_run(self) -> None:
        cfg = self.config
        ser = self._serial
        while not self._reader_stop.is_set():
            if ser is None:
                break
            try:
                raw = ser.readline()
            except Exception as exc:  # noqa: BLE001
                with self._state_lock:
                    self._error = repr(exc)
                time.sleep(0.05)
                continue
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            self._consume_line(line, cfg.done_tokens)

    def _consume_line(self, line: str, done_tokens) -> None:
        upper = line.upper()
        with self._state_lock:
            self._last_status = line
            if any(tok in upper for tok in done_tokens):
                self._in_flight = max(0, self._in_flight - 1)
            # Optional step-count telemetry: "STEPS <n>" / "POS <n>" / bare int.
            for prefix in ("STEPS", "POS", "COUNT"):
                if upper.startswith(prefix):
                    tail = line[len(prefix):].strip(" :=")
                    if tail.lstrip("-").isdigit():
                        self._steps_reported = int(tail)
                    break
            else:
                if line.lstrip("-").isdigit():
                    self._steps_reported = int(line)
        self._publish_feedback()

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------
    def _snapshot_locked(self) -> AdvancerFeedback:
        return AdvancerFeedback(
            t_monotonic=now_monotonic(),
            commanded_distance_mm=self._commanded_mm,
            dispatched_distance_mm=self._dispatched_mm,
            steps_dispatched_total=self._steps_dispatched,
            residual_mm=self._residual_mm,
            commands_in_flight=self._in_flight,
            last_status=self._last_status,
            steps_reported_total=self._steps_reported,
            error=self._error,
        )

    def _publish_feedback(self) -> None:
        with self._state_lock:
            snap = self._snapshot_locked()
        self.feedback_slot.set(snap)
