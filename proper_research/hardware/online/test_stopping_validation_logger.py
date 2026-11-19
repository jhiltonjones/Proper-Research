"""2026-11-19: mock test of the stopping-validation logging/analysis
path, using synthetic joint samples and a trivial FK (identity-like
mapping to a known Cartesian point) -- no robot, no motion configuration.
Proves the event/displacement bookkeeping is correct before any real
test is run.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware.online.stopping_validation_logger import (
    StoppingTestLog, displacement_segments,
)


def _fk(q6: np.ndarray) -> np.ndarray:
    """Synthetic FK: treats q6[0] as a 1D position along a fixed axis,
    scaled -- lets displacement be computed exactly by hand for the
    assertions below."""
    return np.array([float(q6[0]) * 0.1, 0.0, 0.0])


def test_displacement_segments_split_correctly():
    log = StoppingTestLog()
    # joint samples continuously through the sequence
    for i, q0 in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]):
        log.record_sample(t=float(i), q6=np.array([q0, 0, 0, 0, 0, 0]))

    log.record_event("boundary_crossing", t=1.0)   # q0=1.0 -> x=0.1
    log.record_event("detection", t=2.0)             # q0=2.0 -> x=0.2
    log.record_event("last_dispatch", t=2.0)
    log.record_event("streamer_exit", t=3.0)
    log.record_event("stop_request", t=3.0)
    log.record_event("standstill", t=5.0)            # q0=5.0 -> x=0.5

    assert log.is_complete()
    result = displacement_segments(log, _fk)
    assert result["crossing_to_detection_m"] == pytest.approx(0.1)   # |0.2-0.1|
    assert result["detection_to_standstill_m"] == pytest.approx(0.3)  # |0.5-0.2|
    assert result["t_crossing"] == 1.0
    assert result["t_detection"] == 2.0
    assert result["t_last_dispatch"] == 2.0
    assert result["t_streamer_exit"] == 3.0
    assert result["t_stop_request"] == 3.0
    assert result["t_standstill"] == 5.0


def test_incomplete_log_before_standstill_is_not_complete():
    log = StoppingTestLog()
    log.record_sample(t=0.0, q6=np.zeros(6))
    log.record_event("boundary_crossing", t=0.0)
    log.record_event("detection", t=0.1)
    assert not log.is_complete(), (
        "instrumentation must be treated as still required until a "
        "standstill event is actually recorded"
    )
    with pytest.raises(ValueError):
        displacement_segments(log, _fk)


def test_displacement_never_summed_into_one_number_by_the_api():
    log = StoppingTestLog()
    for i, q0 in enumerate([0.0, 1.0, 2.0]):
        log.record_sample(t=float(i), q6=np.array([q0, 0, 0, 0, 0, 0]))
    log.record_event("boundary_crossing", t=0.0)
    log.record_event("detection", t=1.0)
    log.record_event("standstill", t=2.0)
    result = displacement_segments(log, _fk)
    assert "crossing_to_detection_m" in result and "detection_to_standstill_m" in result
    assert "total_displacement_m" not in result  # deliberately not provided as a single figure


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
