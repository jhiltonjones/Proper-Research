"""2026-11-19: verifies OfflineJointControllerAdapter's new optional shared
measured-state external safety check (used to give the inverse-Jacobian
comparator, INV-7, the SAME external monitor as the MPC adapter has,
without adding any predictive constraint handling to the controller
itself -- see controller_adapters.py's __init__ docstring note)."""
from dataclasses import dataclass

import numpy as np
import pytest

from proper_research.hardware.online.controller_adapters import (
    OfflineJointControllerAdapter,
)
from proper_research.hardware.online.messages import StateEstimate


@dataclass
class _Reference:
    sample_count: int = 10
    sample_period_s: float = 0.1


class _StubStep:
    def __init__(self, command):
        self.command = np.asarray(command, dtype=float)
        self.success = True
        self.status = "ok"
        self.iterations = 1
        self.solve_time_s = 0.001


class _StubController:
    name = "stub"

    def __init__(self):
        self.solve_calls = 0

    def solve(self, *, measured_state, measured_beam_position, control_index, previous_input):
        self.solve_calls += 1
        return _StubStep(np.ones(7) * 0.01)


def _estimate(q6, insertion_m=0.03):
    return StateEstimate(
        t_monotonic=0.0,
        x_meas=np.zeros(6),
        tip_position_m=np.zeros(3),
        tip_tangent=np.array([0.0, 0.0, 1.0]),
        insertion_length_m=insertion_m,
        robot_joints=np.asarray(q6, dtype=float),
    )


def _identity_magnet_xyz(q6):
    return np.asarray(q6[:3], dtype=float)


def test_no_external_check_configured_behaves_as_before():
    controller = _StubController()
    adapter = OfflineJointControllerAdapter(controller=controller, reference=_Reference())
    result = adapter(_estimate([0, 0, 0, 0, 0, 0]), dt=0.1)
    assert controller.solve_calls == 1
    assert result.infeasible is False
    assert np.allclose(result.u0, 0.01)
    assert result.info["abort_reason"] is None


def test_external_trip_holds_command_and_never_calls_solve():
    controller = _StubController()
    adapter = OfflineJointControllerAdapter(
        controller=controller, reference=_Reference(),
        magnet_transform_fn=_identity_magnet_xyz,
        magnet_z_bounds_m=(0.0, 0.08),
    )
    # measured_state[:3] = q6[:3] = magnet_xyz; z=0.09 > 0.08 -> trip
    result = adapter(_estimate([0, 0, 0.09, 0, 0, 0]), dt=0.1)
    assert controller.solve_calls == 0, "controller.solve() must never run on an external-check trip"
    assert result.infeasible is True
    assert np.allclose(result.u0, 0.0)
    assert result.info["abort_reason"] is not None
    assert "magnet_z_out_of_bounds" in result.info["abort_reason"]
    assert result.info["status"] == "external_safety_abort"


def test_exclusion_radius_configured_but_not_tripped_reports_gap_and_still_solves():
    controller = _StubController()
    lumen = np.array([[0.0, 0.0, 0.05]])
    adapter = OfflineJointControllerAdapter(
        controller=controller, reference=_Reference(),
        magnet_transform_fn=_identity_magnet_xyz,
        magnet_exclusion_radius_m=0.233, lumen_C_m=lumen,
    )
    result = adapter(_estimate([1.0, 0.0, 0.05, 0, 0, 0]), dt=0.1)
    assert controller.solve_calls == 1
    assert result.infeasible is False
    assert result.info["abort_reason"] is None
    assert result.info["external_measured_clearance_gap_m"] is not None
    assert abs(result.info["external_measured_clearance_gap_m"] - 1.0) < 1e-9
