"""Unit tests for the shared measured-state external safety check
(2026-11-19), extracted from process_isolated_adapter.py so the inverse-
Jacobian comparator can reuse it without acquiring MPC's QP."""
import numpy as np

from proper_research.controllers.mpc_delay_aware.external_safety_check import (
    check_measured_external_safety,
)


def _identity_magnet_xyz(q6):
    # test-only stand-in transform: magnet xyz == first 3 "joint" values
    return np.asarray(q6[:3], dtype=float)


def test_no_configured_checks_never_trips():
    reason, gap_m = check_measured_external_safety(
        np.zeros(6), _identity_magnet_xyz,
        magnet_z_bounds_m=None, magnet_exclusion_radius_m=None, lumen_C_m=None,
    )
    assert reason is None and gap_m is None


def test_none_transform_fn_never_trips():
    reason, gap_m = check_measured_external_safety(
        np.zeros(6), None,
        magnet_z_bounds_m=(0.0, 0.08), magnet_exclusion_radius_m=0.233,
        lumen_C_m=np.zeros((1, 3)),
    )
    assert reason is None and gap_m is None


def test_z_bounds_trip_above_and_below():
    reason, _ = check_measured_external_safety(
        np.array([0.0, 0.0, 0.09, 0, 0, 0]), _identity_magnet_xyz,
        magnet_z_bounds_m=(0.0, 0.08),
    )
    assert reason is not None and "magnet_z_out_of_bounds" in reason

    reason, _ = check_measured_external_safety(
        np.array([0.0, 0.0, -0.01, 0, 0, 0]), _identity_magnet_xyz,
        magnet_z_bounds_m=(0.0, 0.08),
    )
    assert reason is not None and "magnet_z_out_of_bounds" in reason


def test_exclusion_radius_trips_below_boundary_strict_lt():
    lumen = np.array([[0.0, 0.0, 0.05]])
    # magnet at (0.23, 0, 0.05) -> gap=0.23m, radius=0.233m -> gap < radius -> trips
    reason, gap_m = check_measured_external_safety(
        np.array([0.23, 0.0, 0.05, 0, 0, 0]), _identity_magnet_xyz,
        magnet_exclusion_radius_m=0.233, lumen_C_m=lumen,
    )
    assert reason is not None and "magnet_exclusion_violated" in reason
    assert abs(gap_m - 0.23) < 1e-9


def test_exclusion_radius_does_not_trip_above_boundary():
    lumen = np.array([[0.0, 0.0, 0.05]])
    reason, gap_m = check_measured_external_safety(
        np.array([0.30, 0.0, 0.05, 0, 0, 0]), _identity_magnet_xyz,
        magnet_exclusion_radius_m=0.233, lumen_C_m=lumen,
    )
    assert reason is None
    assert abs(gap_m - 0.30) < 1e-9


def test_gap_reported_unconditionally_even_when_not_tripping():
    lumen = np.array([[0.0, 0.0, 0.05]])
    reason, gap_m = check_measured_external_safety(
        np.array([1.0, 0.0, 0.05, 0, 0, 0]), _identity_magnet_xyz,
        magnet_exclusion_radius_m=0.233, lumen_C_m=lumen,
    )
    assert reason is None
    assert gap_m is not None and abs(gap_m - 1.0) < 1e-9


def test_z_bounds_checked_before_exclusion_radius_both_configured():
    # z-bounds trip should win over exclusion-radius trip when both are
    # configured and both would trip -- matches the original adapter's
    # "early_abort_reason is None and ..." short-circuit order.
    lumen = np.array([[0.0, 0.0, 0.05]])
    reason, _ = check_measured_external_safety(
        np.array([0.0, 0.0, 0.09, 0, 0, 0]), _identity_magnet_xyz,
        magnet_z_bounds_m=(0.0, 0.08),
        magnet_exclusion_radius_m=0.233, lumen_C_m=lumen,
    )
    assert "magnet_z_out_of_bounds" in reason
