"""2026-11-19: non-motion instrumentation for the lab-operator-reviewed
stopping-distance validation test. Defines the event record and the
crossing-to-detection / detection-to-standstill displacement calculation
only -- no motion configuration, no trigger logic, no robot connection.
Built and mock-tested now so the analysis path is ready before the
operator reviews and sets the actual test envelope.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


@dataclass
class StoppingTestEvent:
    """One timestamped event in the sequence. `kind` is one of:
    'boundary_crossing' (the monitored geometric condition first becomes
    true, evaluated on a logged sample -- NOT necessarily when detection
    fires, since detection runs on its own cadence), 'detection' (the
    application's own check observes the crossing and sets an abort
    reason), 'last_dispatch' (the last servo command actually sent before
    shutdown), 'streamer_exit' (BackgroundArmStreamer.stop() returns),
    'stop_request' (robot.servo_stop() is issued), 'standstill' (measured
    joint velocity first drops below the operator-set threshold and stays
    there for the operator-set hold time)."""
    kind: str
    t_monotonic: float
    q_rad: Array | None = None  # joint state at this event, if sampled here


@dataclass
class StoppingTestLog:
    events: list[StoppingTestEvent] = field(default_factory=list)
    joint_samples: list[tuple[float, Array]] = field(default_factory=list)  # (t, q6), continuous through stopping

    def record_event(self, kind: str, t: float, q_rad: Array | None = None) -> None:
        self.events.append(StoppingTestEvent(kind=kind, t_monotonic=t, q_rad=q_rad))

    def record_sample(self, t: float, q6: Array) -> None:
        self.joint_samples.append((t, np.asarray(q6, dtype=float).reshape(6).copy()))

    def get(self, kind: str) -> StoppingTestEvent | None:
        matches = [e for e in self.events if e.kind == kind]
        return matches[0] if matches else None

    def is_complete(self) -> bool:
        """Instrumentation must stay alive (keep sampling joints) until
        this returns True -- i.e. until a 'standstill' event has actually
        been recorded, not merely until 'stop_request' has been issued."""
        return self.get("standstill") is not None


def position_at(log: StoppingTestLog, t: float, fk) -> Array:
    """Nearest-sample joint state at/after `t`, converted to a Cartesian
    point via the supplied `fk(q6) -> xyz` callable (e.g. the magnet
    forward-kinematics function already used elsewhere in this project)."""
    candidates = [(ts, q) for ts, q in log.joint_samples if ts >= t]
    if not candidates:
        raise ValueError(f"no joint sample at or after t={t} -- instrumentation gap")
    ts, q = min(candidates, key=lambda c: c[0])
    return fk(q)


def displacement_segments(log: StoppingTestLog, fk) -> dict:
    """Returns {'crossing_to_detection_m': float, 'detection_to_standstill_m': float},
    kept as two SEPARATE segments, never summed into one "response
    distance" without also reporting the parts."""
    crossing = log.get("boundary_crossing")
    detection = log.get("detection")
    standstill = log.get("standstill")
    if crossing is None or detection is None or standstill is None:
        raise ValueError("log is incomplete -- crossing/detection/standstill events all required")

    p_crossing = position_at(log, crossing.t_monotonic, fk)
    p_detection = position_at(log, detection.t_monotonic, fk)
    p_standstill = position_at(log, standstill.t_monotonic, fk)

    return {
        "crossing_to_detection_m": float(np.linalg.norm(p_detection - p_crossing)),
        "detection_to_standstill_m": float(np.linalg.norm(p_standstill - p_detection)),
        "t_crossing": crossing.t_monotonic,
        "t_detection": detection.t_monotonic,
        "t_last_dispatch": (log.get("last_dispatch").t_monotonic if log.get("last_dispatch") else None),
        "t_streamer_exit": (log.get("streamer_exit").t_monotonic if log.get("streamer_exit") else None),
        "t_stop_request": (log.get("stop_request").t_monotonic if log.get("stop_request") else None),
        "t_standstill": standstill.t_monotonic,
    }
