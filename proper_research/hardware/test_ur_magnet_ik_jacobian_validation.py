"""Offline tests for ur_magnet_ik_jacobian_validation.py.

These tests do not import ur_rtde, connect to a robot, or issue robot commands.
"""

from __future__ import annotations

import copy
import inspect
import math

import numpy as np

import ur_magnet_ik_jacobian_validation as validation


def synthetic_config() -> validation.ValidationConfig:
    cfg = copy.deepcopy(validation.CONFIG)
    cfg.use_live_robot = False
    cfg.robot_model = "ur5e"
    cfg.T_tcp_magnet_pose6 = (0.012, -0.006, 0.038, 0.05, -0.03, 0.02)
    cfg.assume_tcp_is_magnet_frame = False
    cfg.ik_max_iterations = 200
    cfg.ik_position_tolerance_m = 2.0e-8
    cfg.ik_orientation_tolerance_rad = 2.0e-8
    return cfg


def test_pose6_transform_round_trip() -> None:
    pose = np.array([0.3, -0.2, 0.7, 0.4, -0.25, 0.1])
    reconstructed = validation.pose6_to_T(validation.T_to_pose6(validation.pose6_to_T(pose)))
    np.testing.assert_allclose(reconstructed, validation.pose6_to_T(pose), atol=1.0e-12)
    np.testing.assert_allclose(
        validation.inverse_T(reconstructed) @ reconstructed,
        np.eye(4),
        atol=1.0e-12,
    )


def test_target_offsets_do_not_rotate_position_about_origin() -> None:
    T = validation.pose6_to_T((0.4, -0.1, 0.6, 0.1, 0.2, -0.3))
    base_target = validation.IKTargetOffset(
        "base", (2.0, -3.0, 4.0), (0.0, 0.0, 10.0), "robot_base"
    )
    result = validation.apply_target_offset(T, base_target)
    np.testing.assert_allclose(result[:3, 3], T[:3, 3] + [0.002, -0.003, 0.004])

    local_target = validation.IKTargetOffset(
        "local", (2.0, -3.0, 4.0), (0.0, 0.0, 10.0), "magnet_local"
    )
    result_local = validation.apply_target_offset(T, local_target)
    np.testing.assert_allclose(
        result_local[:3, 3],
        T[:3, 3] + T[:3, :3] @ np.array([0.002, -0.003, 0.004]),
    )


def test_nominal_ur5e_parameters_match_expected_values() -> None:
    dh = validation.nominal_ur_dh("ur5e")
    np.testing.assert_allclose(dh.a_m, [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(dh.d_m, [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
    assert math.isclose(dh.alpha_rad[0], math.pi / 2.0)
    assert math.isclose(dh.alpha_rad[4], -math.pi / 2.0)


def test_analytic_geometric_jacobian_matches_central_difference() -> None:
    cfg = synthetic_config()
    dh = validation.corrected_dh_from_config(cfg)
    q = np.array([0.35, -1.05, 1.25, -1.45, -1.2, 0.55])
    T_F_TCP = validation.pose6_to_T((0.0, 0.0, 0.11, 0.0, 0.0, 0.0))
    T_TCP_M = validation.magnet_offset_from_config(cfg)
    T_F_M = T_F_TCP @ T_TCP_M
    analytic = validation.geometric_jacobian(q, dh, T_F_M)
    finite_difference = validation.central_difference_jacobian(
        lambda q_value: validation.forward_kinematics(q_value, dh, T_F_M).T_R_target,
        q,
        1.0e-6,
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=3.0e-7, atol=3.0e-8)
    metrics = validation.jacobian_difference(analytic, finite_difference)
    assert metrics["relative_frobenius_error"] < 1.0e-7


def test_damped_ik_recovers_known_synthetic_pose() -> None:
    cfg = synthetic_config()
    dh = validation.corrected_dh_from_config(cfg)
    T_F_TCP = validation.pose6_to_T((0.0, 0.0, 0.09, 0.0, 0.0, 0.0))
    T_F_M = T_F_TCP @ validation.magnet_offset_from_config(cfg)
    q_seed = np.array([0.2, -1.1, 1.3, -1.6, -1.2, 0.4])
    q_known = q_seed + np.array([0.006, -0.008, 0.005, 0.004, -0.003, 0.007])
    target = validation.forward_kinematics(q_known, dh, T_F_M).T_R_target
    result = validation.inverse_kinematics_dls(target, q_seed, dh, T_F_M, cfg)
    assert result.converged, result.reason
    assert result.final_position_error_m < cfg.ik_position_tolerance_m
    assert result.final_orientation_error_rad < cfg.ik_orientation_tolerance_rad
    solved_pose = validation.forward_kinematics(result.q_rad, dh, T_F_M).T_R_target
    metrics = validation.pose_error_metrics(solved_pose, target)
    assert metrics["position_error_m"] < 2.0e-8
    assert metrics["orientation_error_rad"] < 2.0e-8


def test_joint_step_saturation_applies_axis_then_total_limit() -> None:
    delta = np.array([0.4, -0.4, 0.1, 0.0, 0.0, 0.0])
    limited = validation.saturate_joint_step(delta, 0.12, (0.08,) * 6)
    assert np.linalg.norm(limited) <= 0.12 + 1.0e-14
    assert np.all(np.abs(limited) <= 0.08 + 1.0e-14)
    assert np.sign(limited[0]) == 1.0
    assert np.sign(limited[1]) == -1.0


def test_wrapped_joint_difference_handles_two_pi_equivalence() -> None:
    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    equivalent = q + np.array([2.0 * np.pi, -2.0 * np.pi, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        validation.wrapped_joint_difference(equivalent, q),
        np.zeros(6),
        atol=1.0e-12,
    )


def test_synthetic_oracle_comparison_is_exact_for_known_model() -> None:
    """Mimic the ur_rtde comparison using a known linearized FK oracle."""

    cfg = synthetic_config()
    dh = validation.corrected_dh_from_config(cfg)
    q = np.array([-0.4, -1.0, 1.4, -1.2, -1.1, 0.25])
    T_F_M = validation.pose6_to_T((0.02, 0.01, 0.12, 0.02, -0.01, 0.03))
    own = validation.geometric_jacobian(q, dh, T_F_M)
    known_oracle = validation.central_difference_jacobian(
        lambda q_value: validation.forward_kinematics(q_value, dh, T_F_M).T_R_target,
        q,
        cfg.finite_difference_joint_step_rad,
    )
    difference = validation.jacobian_difference(own, known_oracle)
    assert difference["relative_frobenius_error"] < 1.0e-7


def test_live_adapter_exposes_no_motion_method() -> None:
    forbidden = {
        "moveJ",
        "moveL",
        "moveP",
        "movePath",
        "servoJ",
        "servoL",
        "speedJ",
        "speedL",
        "forceMode",
        "freedriveMode",
        "setTcp",
    }
    exposed = set(dir(validation.ReadOnlyURRTDEOracle))
    assert forbidden.isdisjoint(exposed)
    source = inspect.getsource(validation.ReadOnlyURRTDEOracle)
    for method in forbidden:
        assert f".{method}(" not in source


def run_all_tests() -> None:
    """Run this file without pytest, using only the Python standard runner."""

    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} offline tests passed.")


if __name__ == "__main__":
    run_all_tests()
