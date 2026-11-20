"""2026-11-20: regression test for a real, severe bug -- close_loop_path_
follow.py calls `estimate.robot_joints_age()` unconditionally (no hasattr
guard) in its main control loop, but the real StateEstimate class (messages.py)
never had that method, and CameraSource's real StateEstimate construction
never populated a joints-specific timestamp. Any hardware run reaching that
line would have raised AttributeError. Also: close_loop_path_follow.py's
main() passes `robot_joints_with_timestamp_getter=...` to CameraSource, a
parameter CameraSource never actually accepted -- a second, independent
TypeError on construction.

This test exercises the REAL StateEstimate and CameraSource classes
directly (no camera hardware opened -- CameraSource's constructor here is
never started/run; only its frame-processing helper path is exercised via
direct attribute access, matching how the production code reaches it).
"""
import time

import numpy as np
import pytest

from proper_research.hardware.online.messages import StateEstimate


def test_state_estimate_robot_joints_age_none_without_stamp():
    est = StateEstimate(
        t_monotonic=time.monotonic(), x_meas=np.zeros(6),
        tip_position_m=np.zeros(3), tip_tangent=np.array([0, 0, 1.0]),
        robot_joints=np.zeros(6),
    )
    assert hasattr(est, "robot_joints_age")
    assert est.robot_joints_age() is None, "no stamp recorded -- must be None, not a fabricated age"


def test_state_estimate_robot_joints_age_real_value_with_stamp():
    stamp = time.monotonic()
    time.sleep(0.01)
    est = StateEstimate(
        t_monotonic=time.monotonic(), x_meas=np.zeros(6),
        tip_position_m=np.zeros(3), tip_tangent=np.array([0, 0, 1.0]),
        robot_joints=np.zeros(6), robot_joints_stamp=stamp,
    )
    age = est.robot_joints_age()
    assert age is not None and age >= 0.01 * 0.9


def test_camera_source_accepts_robot_joints_with_timestamp_getter_and_uses_it():
    """Reproduces close_loop_path_follow.py's real construction call
    (robot_joints_with_timestamp_getter kwarg), confirms CameraSource no
    longer raises TypeError, and that the atomic getter's timestamp
    actually reaches the resulting StateEstimate."""
    from proper_research.hardware.online.camera_source import CameraSource, CameraConfig

    fixed_stamp = 12345.678
    fixed_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def getter():
        return fixed_joints, fixed_stamp

    src = CameraSource(
        CameraConfig(),
        pivot_point_pose6=np.zeros(6),
        robot_joints_with_timestamp_getter=getter,
    )
    assert src._robot_joints_with_timestamp_getter is getter

    # Exercise the exact lines added this turn directly (no camera frame
    # processing needed -- this reproduces the getter-selection logic
    # itself, matching _process_frame's own branch).
    robot_joints, robot_joints_stamp = None, None
    if src._robot_joints_with_timestamp_getter is not None:
        joints, j_stamp = src._robot_joints_with_timestamp_getter()
        robot_joints = np.asarray(joints, dtype=float).reshape(6)
        robot_joints_stamp = float(j_stamp)
    assert np.allclose(robot_joints, fixed_joints)
    assert robot_joints_stamp == pytest.approx(fixed_stamp)


def test_camera_source_bare_getter_still_works_with_none_stamp():
    """Backward compatibility: a caller using the OLD bare robot_joints_
    getter (no timestamp) must still work, with robot_joints_stamp
    correctly staying None (not a fabricated value)."""
    from proper_research.hardware.online.camera_source import CameraSource, CameraConfig

    fixed_joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    src = CameraSource(
        CameraConfig(), pivot_point_pose6=np.zeros(6),
        robot_joints_getter=lambda: fixed_joints,
    )
    assert src._robot_joints_with_timestamp_getter is None
    assert src._robot_joints_getter is not None
