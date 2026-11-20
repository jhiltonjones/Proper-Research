"""2026-11-19 (rectangle experiment config): boundary-case tests for the
measured-clearance breach detector's comparison convention at a 230mm
abort radius (the already-reviewed boundary, NOT the unapproved 250mm
proposal). Documents and locks in the EXACT comparison used in
process_isolated_adapter.py: `gap_m < radius_m` -- a STRICT less-than, so
a measured gap exactly AT the radius does NOT trip; only a gap strictly
below it does. No hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.controllers.mpc_delay_aware.test_early_measured_clearance_reject import (
    _WorkerMustNotBeCalled, _estimate, _make_reference,
)

RADIUS_M = 0.230  # the already-reviewed 230mm boundary, not 250mm


class _WorkerCountsCalls:
    def __init__(self):
        self.n_calls = 0

    def try_solve(self, **kwargs):
        self.n_calls += 1
        return None


def _adapter_with_gap(gap_m: float, worker):
    """magnet_transform_fn always returns a point exactly `gap_m` from the
    single lumen point at the origin -- isolates the comparison itself
    from any FK/geometry concern."""
    base_m = np.array([[0.0, 0.0, 0.0]])
    return ProcessIsolatedDelayAwareAdapter(
        reference=_make_reference(), worker=worker, deadline_s=0.07,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="test",
        magnet_transform_fn=lambda q6: np.array([gap_m, 0.0, 0.0]),
        magnet_exclusion_lumen_C_m=base_m, magnet_exclusion_radius_m=RADIUS_M,
    )


def test_just_above_boundary_does_not_trip():
    worker = _WorkerCountsCalls()
    adapter = _adapter_with_gap(RADIUS_M + 1e-3, worker)  # 231mm: 1mm clear
    adapter(_estimate(np.zeros(6)), 0.1)
    assert worker.n_calls == 1, "a gap strictly above the radius must reach try_solve"


def test_exactly_at_boundary_does_not_trip():
    """Documents the convention explicitly: `gap_m < radius_m` is a STRICT
    inequality, so a gap bit-exactly equal to the radius is NOT a trip --
    try_solve DOES get called. This is the actual implemented behaviour,
    not a judgement that it is the right behaviour -- a future reviewer
    changing this comparison should update this test deliberately, not
    discover it broke by accident."""
    worker = _WorkerCountsCalls()
    adapter = _adapter_with_gap(RADIUS_M, worker)
    adapter(_estimate(np.zeros(6)), 0.1)
    assert worker.n_calls == 1, "a gap bit-exactly equal to the radius must still reach try_solve"


def test_just_below_boundary_trips():
    adapter = _adapter_with_gap(RADIUS_M - 1e-3, _WorkerMustNotBeCalled())  # 229mm
    result = adapter(_estimate(np.zeros(6)), 0.1)
    assert result.infeasible
    assert np.allclose(result.u0, 0.0)
    assert "magnet_exclusion_violated" in result.info["abort_reason"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
