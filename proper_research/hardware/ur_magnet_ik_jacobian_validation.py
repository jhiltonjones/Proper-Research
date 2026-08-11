"""Read-only UR magnet-pose, inverse-kinematics, and Jacobian validation.

Edit the configuration section below; this script has no command-line
arguments.  It does not contain or call any robot motion command.

Frame convention
----------------
``T_A_B`` maps coordinates expressed in frame B into frame A::

    p_A = T_A_B @ p_B

The frames used here are:

* R: UR robot base
* F: UR output flange
* TCP: the active UR tool centre point
* M: the source-magnet centre/body frame

The physical chain is therefore::

    T_R_M = T_R_F(q) @ T_F_TCP @ T_TCP_M

``getActualTCPPose()`` supplies ``T_R_TCP``.  Consequently
``T_R_TCP @ T_TCP_M`` is a robot-inferred magnet pose, not an independent
physical measurement.  An optional camera/tracker measurement can be supplied
in ``independent_measured_T_robot_magnet_pose6``.

Validation layers
-----------------
1. Independent standard-DH analytic Jacobian versus central differences.
2. Independent nominal/corrected DH FK/Jacobian versus the UR controller's
   calibrated FK/Jacobian exposed through ur_rtde.
3. Independent damped-least-squares IK versus ur_rtde IK.  Both returned joint
   solutions are evaluated through both forward-kinematics implementations.

Important: constructing ``RTDEControlInterface`` may upload its control script
to the controller even though this file sends no motion command.  Live mode is
disabled by default and requires an exact typed confirmation.  Run it only
with the robot stationary and no production program in progress.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# =============================================================================
# USER CONFIGURATION -- edit here; there are no terminal arguments
# =============================================================================


@dataclass(frozen=True)
class IKTargetOffset:
    """A small target displacement applied at the current magnet centre.

    ``frame`` is either ``"robot_base"`` or ``"magnet_local"``.  Translation
    is expressed in that frame.  Rotation changes the magnet orientation about
    its current centre; it never rotates the magnet position about the origin.
    """

    name: str
    translation_mm: tuple[float, float, float]
    rotation_vector_deg: tuple[float, float, float]
    frame: str = "robot_base"


@dataclass
class ValidationConfig:
    # Live access is deliberately off.  Set the exact robot model, provide
    # T_TCP_M, then enable this only while the robot is stationary.
    use_live_robot: bool = False
    robot_ip: str = "192.168.56.101"
    robot_model: str = "EDIT_ME"  # ur3e, ur5e/ur7e, ur10e/ur12e, ur16e,
                                  # ur3, ur5, or ur10
    live_confirmation_phrase: str = "CONNECT READ ONLY KINEMATICS"

    # T_TCP_M: source-magnet centre/body pose expressed in the ACTIVE TCP.
    # Calibrate all six components.  Do not use a scalar z offset as a second,
    # separate correction.  To assert physical identity explicitly, leave this
    # as None and set assume_tcp_is_magnet_frame=True.
    T_tcp_magnet_pose6: tuple[float, float, float, float, float, float] | None = None
    assume_tcp_is_magnet_frame: bool = False

    # Optional independent camera/tracker measurement T_R_M.  This must already
    # be transformed into the UR base frame.  Without it, the "actual magnet"
    # pose is inferred from the robot TCP and T_TCP_M, not independently sensed.
    independent_measured_T_robot_magnet_pose6: (
        tuple[float, float, float, float, float, float] | None
    ) = None

    # Offline data make it possible to validate the mathematics without a robot.
    # If offline_actual_tcp_pose6 is None, it is generated from independent DH FK.
    offline_actual_q_rad: tuple[float, float, float, float, float, float] = (
        0.0,
        -1.2,
        1.4,
        -1.7,
        -1.57,
        0.0,
    )
    offline_active_T_flange_tcp_pose6: tuple[
        float, float, float, float, float, float
    ] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    offline_actual_tcp_pose6: (
        tuple[float, float, float, float, float, float] | None
    ) = None

    # Optional factory calibration corrections.  The public UR table supplies
    # nominal standard-DH parameters.  If you have controller-specific
    # corrections, enter them here with the same joint ordering.
    dh_delta_a_m: tuple[float, ...] = (0.0,) * 6
    dh_delta_d_m: tuple[float, ...] = (0.0,) * 6
    dh_delta_alpha_rad: tuple[float, ...] = (0.0,) * 6
    dh_delta_theta_rad: tuple[float, ...] = (0.0,) * 6

    # These are algorithm bounds, not a replacement for the UR safety system.
    # Replace them with the limits of the installed robot/configuration.
    joint_lower_bounds_rad: tuple[float, ...] = (-2.0 * math.pi,) * 6
    joint_upper_bounds_rad: tuple[float, ...] = (+2.0 * math.pi,) * 6

    targets: tuple[IKTargetOffset, ...] = field(
        default_factory=lambda: (
            IKTargetOffset("current_pose", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("robot_x_plus_1mm", (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("robot_y_plus_1mm", (0.0, 1.0, 0.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("robot_z_plus_1mm", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("magnet_z_rotation_1deg", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), "magnet_local"),
        )
    )

    # Independent inverse-Jacobian IK.  Orientation rows are multiplied by a
    # characteristic length so position and orientation have compatible units.
    orientation_characteristic_length_m: float = 0.10
    ik_damping_m_per_rad: float = 1.0e-4
    ik_maximum_damping_m_per_rad: float = 1.0e-1
    ik_max_iterations: int = 150
    ik_position_tolerance_m: float = 1.0e-6
    ik_orientation_tolerance_rad: float = 1.0e-5
    ik_maximum_joint_step_norm_rad: float = 0.12
    ik_maximum_per_joint_step_rad: tuple[float, ...] = (0.08,) * 6
    ik_line_search_steps: int = 10

    # Jacobian checks.  Both finite differences perturb the physical joint angle
    # in radians.  ur_rtde is assumed to return [vx, vy, vz, wx, wy, wz] rows;
    # select "angular_linear" only if your installed version documents that.
    finite_difference_joint_step_rad: float = 1.0e-6
    rtde_jacobian_row_order: str = "linear_angular"
    calculate_rtde_fk_finite_difference_jacobian: bool = True
    calculate_jacobians_at_ik_solutions: bool = True

    # Stationary-snapshot and diagnostic thresholds.  strict_validation=False
    # reports failed checks without aborting, which is useful during calibration.
    maximum_snapshot_joint_drift_rad: float = 1.0e-4
    maximum_actual_tcp_vs_rtde_fk_position_error_m: float = 1.0e-3
    maximum_actual_tcp_vs_rtde_fk_orientation_error_rad: float = 5.0e-3
    maximum_own_vs_fd_relative_jacobian_error: float = 1.0e-5
    maximum_own_vs_rtde_relative_jacobian_error: float = 5.0e-2
    strict_validation: bool = False

    output_root: str = "ur_magnet_ik_jacobian_validation"
    run_name: str = "stationary_read_only_check"
    make_plots: bool = True
    show_plots_interactively: bool = False


CONFIG = ValidationConfig()


# =============================================================================
# SE(3) TRANSFORMS
# =============================================================================


def finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {length}; got {value!r}.")
    return vector


def validate_transform(T: Any, name: str = "T") -> np.ndarray:
    matrix = np.asarray(T, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{name} has an invalid homogeneous bottom row.")
    R = matrix[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1.0e-8):
        raise ValueError(f"{name} rotation is not orthogonal.")
    if not math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1.0e-8):
        raise ValueError(f"{name} rotation is not right-handed.")
    return matrix.copy()


def pose6_to_T(pose6: Iterable[float]) -> np.ndarray:
    pose = finite_vector(pose6, 6, "pose6")
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def T_to_pose6(T: Any) -> np.ndarray:
    matrix = validate_transform(T)
    return np.r_[matrix[:3, 3], Rot.from_matrix(matrix[:3, :3]).as_rotvec()]


def inverse_T(T: Any) -> np.ndarray:
    matrix = validate_transform(T)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def pose_error_spatial(T_current: Any, T_target: Any) -> np.ndarray:
    """Return [position, spatial rotation-vector] error in robot-base axes."""

    current = validate_transform(T_current, "T_current")
    target = validate_transform(T_target, "T_target")
    position_error = target[:3, 3] - current[:3, 3]
    rotation_error = Rot.from_matrix(
        target[:3, :3] @ current[:3, :3].T
    ).as_rotvec()
    return np.r_[position_error, rotation_error]


def pose_error_metrics(T_actual: Any, T_target: Any) -> dict[str, float]:
    error = pose_error_spatial(T_actual, T_target)
    return {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "position_error_mm": 1.0e3 * float(np.linalg.norm(error[:3])),
        "orientation_error_rad": float(np.linalg.norm(error[3:])),
        "orientation_error_deg": float(np.degrees(np.linalg.norm(error[3:]))),
    }


def apply_target_offset(T_R_M: Any, target: IKTargetOffset) -> np.ndarray:
    """Displace M about its current centre without rotating its position."""

    current = validate_transform(T_R_M, "T_R_M")
    translation = 1.0e-3 * finite_vector(target.translation_mm, 3, "translation_mm")
    rotation_vector = np.radians(
        finite_vector(target.rotation_vector_deg, 3, "rotation_vector_deg")
    )
    R_delta = Rot.from_rotvec(rotation_vector).as_matrix()
    result = current.copy()
    if target.frame == "robot_base":
        result[:3, 3] = current[:3, 3] + translation
        result[:3, :3] = R_delta @ current[:3, :3]
    elif target.frame == "magnet_local":
        result[:3, 3] = current[:3, 3] + current[:3, :3] @ translation
        result[:3, :3] = current[:3, :3] @ R_delta
    else:
        raise ValueError(f"Unknown target frame {target.frame!r}.")
    return validate_transform(result, "target T_R_M")


def task_scaling(characteristic_length_m: float) -> np.ndarray:
    if not np.isfinite(characteristic_length_m) or characteristic_length_m <= 0.0:
        raise ValueError("orientation_characteristic_length_m must be positive.")
    return np.diag([1.0, 1.0, 1.0] + [float(characteristic_length_m)] * 3)


# =============================================================================
# INDEPENDENT UR STANDARD-DH MODEL
# =============================================================================


@dataclass(frozen=True)
class DHParameters:
    a_m: np.ndarray
    d_m: np.ndarray
    alpha_rad: np.ndarray
    theta_offset_rad: np.ndarray

    def __post_init__(self) -> None:
        for name in ("a_m", "d_m", "alpha_rad", "theta_offset_rad"):
            value = finite_vector(getattr(self, name), 6, name)
            object.__setattr__(self, name, value.copy())


def nominal_ur_dh(robot_model: str) -> DHParameters:
    """Return nominal standard-DH values from Universal Robots' public table."""

    key = robot_model.strip().lower().replace(" ", "")
    aliases = {"ur7e": "ur5e", "ur12e": "ur10e"}
    key = aliases.get(key, key)
    table: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
        "ur3e": (
            (0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0),
            (0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921),
        ),
        "ur5e": (
            (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
            (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
        ),
        "ur10e": (
            (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0),
            (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655),
        ),
        "ur16e": (
            (0.0, -0.4784, -0.36, 0.0, 0.0, 0.0),
            (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655),
        ),
        "ur3": (
            (0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0),
            (0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819),
        ),
        "ur5": (
            (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0),
            (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823),
        ),
        "ur10": (
            (0.0, -0.612, -0.5723, 0.0, 0.0, 0.0),
            (0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922),
        ),
    }
    if key not in table:
        raise ValueError(
            f"robot_model={robot_model!r} is not supported. Choose one of "
            f"{sorted(table)} (ur7e and ur12e are accepted aliases)."
        )
    a_m, d_m = table[key]
    alpha = (math.pi / 2.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0)
    return DHParameters(np.array(a_m), np.array(d_m), np.array(alpha), np.zeros(6))


def corrected_dh_from_config(cfg: ValidationConfig) -> DHParameters:
    nominal = nominal_ur_dh(cfg.robot_model)
    return DHParameters(
        nominal.a_m + finite_vector(cfg.dh_delta_a_m, 6, "dh_delta_a_m"),
        nominal.d_m + finite_vector(cfg.dh_delta_d_m, 6, "dh_delta_d_m"),
        nominal.alpha_rad + finite_vector(cfg.dh_delta_alpha_rad, 6, "dh_delta_alpha_rad"),
        nominal.theta_offset_rad
        + finite_vector(cfg.dh_delta_theta_rad, 6, "dh_delta_theta_rad"),
    )


def standard_dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class ForwardKinematicsResult:
    T_R_F: np.ndarray
    T_R_target: np.ndarray
    joint_origins_R_m: np.ndarray
    joint_axes_R: np.ndarray
    link_points_R_m: np.ndarray


def forward_kinematics(
    q_rad: Iterable[float],
    dh: DHParameters,
    T_F_target: Any | None = None,
) -> ForwardKinematicsResult:
    q = finite_vector(q_rad, 6, "q_rad")
    target_offset = np.eye(4) if T_F_target is None else validate_transform(T_F_target, "T_F_target")
    T = np.eye(4)
    joint_origins: list[np.ndarray] = []
    joint_axes: list[np.ndarray] = []
    link_points = [T[:3, 3].copy()]
    for i in range(6):
        joint_origins.append(T[:3, 3].copy())
        joint_axes.append(T[:3, 2].copy())
        T = T @ standard_dh_transform(
            float(dh.a_m[i]),
            float(dh.d_m[i]),
            float(dh.alpha_rad[i]),
            float(q[i] + dh.theta_offset_rad[i]),
        )
        link_points.append(T[:3, 3].copy())
    T_R_F = validate_transform(T, "T_R_F")
    T_R_target = validate_transform(T_R_F @ target_offset, "T_R_target")
    return ForwardKinematicsResult(
        T_R_F=T_R_F,
        T_R_target=T_R_target,
        joint_origins_R_m=np.asarray(joint_origins),
        joint_axes_R=np.asarray(joint_axes),
        link_points_R_m=np.asarray(link_points),
    )


def geometric_jacobian(
    q_rad: Iterable[float],
    dh: DHParameters,
    T_F_target: Any | None = None,
) -> np.ndarray:
    """Return base-frame geometric J = [linear; angular], shape (6, 6)."""

    fk = forward_kinematics(q_rad, dh, T_F_target)
    p = fk.T_R_target[:3, 3]
    J = np.zeros((6, 6), dtype=float)
    for i, (origin, axis) in enumerate(zip(fk.joint_origins_R_m, fk.joint_axes_R)):
        J[:3, i] = np.cross(axis, p - origin)
        J[3:, i] = axis
    return J


def central_difference_jacobian(
    fk_function: Callable[[np.ndarray], np.ndarray],
    q_rad: Iterable[float],
    step_rad: float,
) -> np.ndarray:
    """Central-difference a pose function into a base-frame geometric Jacobian."""

    q = finite_vector(q_rad, 6, "q_rad")
    if not np.isfinite(step_rad) or step_rad <= 0.0:
        raise ValueError("finite-difference step must be positive.")
    J = np.empty((6, 6), dtype=float)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = step_rad
        T_plus = validate_transform(fk_function(q + dq), "T_plus")
        T_minus = validate_transform(fk_function(q - dq), "T_minus")
        J[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * step_rad)
        delta_R = T_plus[:3, :3] @ T_minus[:3, :3].T
        J[3:, i] = Rot.from_matrix(delta_R).as_rotvec() / (2.0 * step_rad)
    return J


def condition_number(J: Any, characteristic_length_m: float | None = None) -> float:
    matrix = np.asarray(J, dtype=float).reshape(6, 6)
    if not np.all(np.isfinite(matrix)):
        return float("inf")
    if characteristic_length_m is not None:
        matrix = task_scaling(characteristic_length_m) @ matrix
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= np.finfo(float).eps * max(1.0, singular_values[0]):
        return float("inf")
    return float(singular_values[0] / singular_values[-1])


def jacobian_difference(reference: Any, candidate: Any) -> dict[str, Any]:
    A = np.asarray(reference, dtype=float).reshape(6, 6)
    B = np.asarray(candidate, dtype=float).reshape(6, 6)
    error = B - A
    reference_norm = float(np.linalg.norm(A, ord="fro"))
    return {
        "absolute_frobenius_error": float(np.linalg.norm(error, ord="fro")),
        "relative_frobenius_error": float(
            np.linalg.norm(error, ord="fro") / max(reference_norm, 1.0e-15)
        ),
        "maximum_absolute_element_error": float(np.max(np.abs(error))),
        "column_l2_errors": np.linalg.norm(error, axis=0),
        "linear_block_frobenius_error": float(np.linalg.norm(error[:3], ord="fro")),
        "angular_block_frobenius_error": float(np.linalg.norm(error[3:], ord="fro")),
    }


# =============================================================================
# INDEPENDENT DAMPED INVERSE-JACOBIAN IK
# =============================================================================


@dataclass
class IKResult:
    q_rad: np.ndarray
    converged: bool
    iterations: int
    reason: str
    final_position_error_m: float
    final_orientation_error_rad: float
    final_scaled_jacobian_condition: float
    final_damping_m_per_rad: float
    history: list[dict[str, float]]


def saturate_joint_step(
    delta_q: Iterable[float],
    maximum_norm_rad: float,
    per_joint_limits_rad: Iterable[float],
) -> np.ndarray:
    delta = finite_vector(delta_q, 6, "delta_q")
    per_joint = finite_vector(per_joint_limits_rad, 6, "per_joint_limits_rad")
    if maximum_norm_rad <= 0.0 or np.any(per_joint <= 0.0):
        raise ValueError("IK step limits must be positive.")
    delta = np.clip(delta, -per_joint, per_joint)
    norm = float(np.linalg.norm(delta))
    if norm > maximum_norm_rad:
        delta *= maximum_norm_rad / norm
    return delta


def inverse_kinematics_dls(
    T_R_target: Any,
    q_seed_rad: Iterable[float],
    dh: DHParameters,
    T_F_target: Any,
    cfg: ValidationConfig,
) -> IKResult:
    """Solve 6-D pose IK with the independent analytic geometric Jacobian."""

    target = validate_transform(T_R_target, "IK target")
    q_lower = finite_vector(cfg.joint_lower_bounds_rad, 6, "joint_lower_bounds_rad")
    q_upper = finite_vector(cfg.joint_upper_bounds_rad, 6, "joint_upper_bounds_rad")
    if np.any(q_lower >= q_upper):
        raise ValueError("Every joint lower bound must be below its upper bound.")
    q = np.clip(finite_vector(q_seed_rad, 6, "q_seed_rad"), q_lower, q_upper)
    S = task_scaling(cfg.orientation_characteristic_length_m)
    damping = float(cfg.ik_damping_m_per_rad)
    history: list[dict[str, float]] = []
    reason = "maximum_iterations"
    converged = False

    def evaluate(q_value: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        T_current = forward_kinematics(q_value, dh, T_F_target).T_R_target
        error = pose_error_spatial(T_current, target)
        return T_current, error, float(np.linalg.norm(S @ error))

    for iteration in range(cfg.ik_max_iterations + 1):
        _, error, cost = evaluate(q)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:]))
        J = geometric_jacobian(q, dh, T_F_target)
        J_scaled = S @ J
        scaled_condition = condition_number(J, cfg.orientation_characteristic_length_m)
        history.append(
            {
                "iteration": float(iteration),
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "scaled_cost_m": cost,
                "scaled_jacobian_condition": scaled_condition,
                "damping_m_per_rad": damping,
            }
        )
        if (
            position_error <= cfg.ik_position_tolerance_m
            and orientation_error <= cfg.ik_orientation_tolerance_rad
        ):
            converged = True
            reason = "pose_tolerance"
            break
        if iteration >= cfg.ik_max_iterations:
            break

        rhs = S @ error
        regularized = J_scaled @ J_scaled.T + damping**2 * np.eye(6)
        try:
            delta_q = J_scaled.T @ np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            reason = "linear_solve_failed"
            break
        delta_q = saturate_joint_step(
            delta_q,
            cfg.ik_maximum_joint_step_norm_rad,
            cfg.ik_maximum_per_joint_step_rad,
        )

        accepted = False
        for line_index in range(cfg.ik_line_search_steps):
            scale = 0.5**line_index
            trial = np.clip(q + scale * delta_q, q_lower, q_upper)
            _, _, trial_cost = evaluate(trial)
            if trial_cost < cost - 1.0e-14:
                q = trial
                damping = max(cfg.ik_damping_m_per_rad, damping * 0.5)
                accepted = True
                break
        if not accepted:
            damping *= 10.0
            if damping > cfg.ik_maximum_damping_m_per_rad:
                reason = "line_search_stalled"
                break

    final_T = forward_kinematics(q, dh, T_F_target).T_R_target
    final_error = pose_error_spatial(final_T, target)
    return IKResult(
        q_rad=q.copy(),
        converged=converged,
        iterations=len(history) - 1,
        reason=reason,
        final_position_error_m=float(np.linalg.norm(final_error[:3])),
        final_orientation_error_rad=float(np.linalg.norm(final_error[3:])),
        final_scaled_jacobian_condition=condition_number(
            geometric_jacobian(q, dh, T_F_target),
            cfg.orientation_characteristic_length_m,
        ),
        final_damping_m_per_rad=damping,
        history=history,
    )


def wrapped_joint_difference(q_a: Iterable[float], q_b: Iterable[float]) -> np.ndarray:
    """Return q_a - q_b modulo 2*pi, element by element."""

    delta = finite_vector(q_a, 6, "q_a") - finite_vector(q_b, 6, "q_b")
    return np.arctan2(np.sin(delta), np.cos(delta))


# =============================================================================
# READ-ONLY ur_rtde ORACLE
# =============================================================================


@dataclass
class RobotSnapshot:
    q_rad: np.ndarray
    actual_T_R_TCP: np.ndarray
    T_F_TCP: np.ndarray
    q_before_rad: np.ndarray
    q_after_rad: np.ndarray
    actual_T_R_F_reported: np.ndarray | None


class ReadOnlyURRTDEOracle:
    """Expose only state and kinematics queries; intentionally no motion API."""

    def __init__(self, cfg: ValidationConfig):
        print(
            "\nLIVE CONNECTION WARNING:\n"
            "  No motion method exists in this validator.\n"
            "  RTDEControlInterface may still upload a controller script.\n"
            "  Keep the robot stationary and stop other robot programs first.\n"
        )
        confirmation = input(
            f"Type exactly {cfg.live_confirmation_phrase!r} to connect: "
        )
        if confirmation != cfg.live_confirmation_phrase:
            raise RuntimeError("Live connection cancelled: confirmation did not match.")
        try:
            import rtde_control
            import rtde_receive
        except ImportError as exc:
            raise RuntimeError(
                "ur_rtde is not installed in this Python environment."
            ) from exc
        self.receive = rtde_receive.RTDEReceiveInterface(cfg.robot_ip)
        self.control = rtde_control.RTDEControlInterface(cfg.robot_ip)

    def close(self) -> None:
        for interface in (getattr(self, "control", None), getattr(self, "receive", None)):
            if interface is not None and hasattr(interface, "disconnect"):
                try:
                    interface.disconnect()
                except Exception:
                    pass

    def snapshot(self) -> RobotSnapshot:
        q_before = finite_vector(self.receive.getActualQ(), 6, "getActualQ before")
        actual_tcp = pose6_to_T(self.receive.getActualTCPPose())
        q_after = finite_vector(self.receive.getActualQ(), 6, "getActualQ after")
        q = q_before + 0.5 * wrapped_joint_difference(q_after, q_before)
        T_F_TCP = pose6_to_T(self.control.getTCPOffset())
        flange = None
        if hasattr(self.control, "getActualToolFlangePose"):
            try:
                flange = pose6_to_T(self.control.getActualToolFlangePose())
            except Exception:
                flange = None
        return RobotSnapshot(q, actual_tcp, T_F_TCP, q_before, q_after, flange)

    def forward_kinematics(self, q_rad: Iterable[float], T_F_target: Any) -> np.ndarray:
        pose = self.control.getForwardKinematics(
            finite_vector(q_rad, 6, "q_rad").tolist(),
            T_to_pose6(T_F_target).tolist(),
        )
        return pose6_to_T(pose)

    def inverse_kinematics(
        self,
        T_R_TCP_target: Any,
        q_near_rad: Iterable[float],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> tuple[np.ndarray | None, str]:
        pose = T_to_pose6(T_R_TCP_target).tolist()
        q_near = finite_vector(q_near_rad, 6, "q_near_rad").tolist()
        try:
            if hasattr(self.control, "getInverseKinematicsHasSolution"):
                has_solution = self.control.getInverseKinematicsHasSolution(
                    pose,
                    q_near,
                    float(position_tolerance_m),
                    float(orientation_tolerance_rad),
                )
                if not has_solution:
                    return None, "ur_rtde_reported_no_solution"
            result = self.control.getInverseKinematics(
                pose,
                q_near,
                float(position_tolerance_m),
                float(orientation_tolerance_rad),
            )
            return finite_vector(result, 6, "ur_rtde IK result"), "solution"
        except Exception as exc:
            return None, f"ur_rtde_exception: {type(exc).__name__}: {exc}"

    def jacobian(self, q_rad: Iterable[float], T_F_target: Any, row_order: str) -> np.ndarray:
        raw = self.control.getJacobian(
            finite_vector(q_rad, 6, "q_rad").tolist(),
            T_to_pose6(T_F_target).tolist(),
        )
        J = np.asarray(raw, dtype=float).reshape(6, 6)
        if row_order == "linear_angular":
            return J
        if row_order == "angular_linear":
            return np.vstack([J[3:], J[:3]])
        raise ValueError(f"Unknown rtde_jacobian_row_order={row_order!r}.")

    def joints_within_safety_limits(self, q_rad: Iterable[float]) -> bool | None:
        try:
            return bool(self.control.isJointsWithinSafetyLimits(
                finite_vector(q_rad, 6, "q_rad").tolist()
            ))
        except Exception:
            return None


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================


def magnet_offset_from_config(cfg: ValidationConfig) -> np.ndarray:
    if cfg.T_tcp_magnet_pose6 is not None:
        return pose6_to_T(cfg.T_tcp_magnet_pose6)
    if cfg.assume_tcp_is_magnet_frame:
        return np.eye(4)
    raise ValueError(
        "T_tcp_magnet_pose6 is required. Calibrate the magnet centre and body "
        "orientation in the active TCP, or explicitly set "
        "assume_tcp_is_magnet_frame=True when the frames are physically identical."
    )


def validate_config(cfg: ValidationConfig) -> None:
    nominal_ur_dh(cfg.robot_model)
    magnet_offset_from_config(cfg)
    finite_vector(cfg.joint_lower_bounds_rad, 6, "joint_lower_bounds_rad")
    finite_vector(cfg.joint_upper_bounds_rad, 6, "joint_upper_bounds_rad")
    finite_vector(cfg.ik_maximum_per_joint_step_rad, 6, "ik_maximum_per_joint_step_rad")
    if cfg.rtde_jacobian_row_order not in {"linear_angular", "angular_linear"}:
        raise ValueError("rtde_jacobian_row_order must be linear_angular or angular_linear.")
    if not cfg.targets:
        raise ValueError("At least one IK target is required.")
    for target in cfg.targets:
        apply_target_offset(np.eye(4), target)


def offline_snapshot(cfg: ValidationConfig, dh: DHParameters) -> RobotSnapshot:
    q = finite_vector(cfg.offline_actual_q_rad, 6, "offline_actual_q_rad")
    T_F_TCP = pose6_to_T(cfg.offline_active_T_flange_tcp_pose6)
    if cfg.offline_actual_tcp_pose6 is None:
        actual_tcp = forward_kinematics(q, dh, T_F_TCP).T_R_target
    else:
        actual_tcp = pose6_to_T(cfg.offline_actual_tcp_pose6)
    return RobotSnapshot(q, actual_tcp, T_F_TCP, q.copy(), q.copy(), None)


def make_output_directory(cfg: ValidationConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"{stamp}_{cfg.run_name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return to_jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, allow_nan=False) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, list, tuple, dict)):
        return json.dumps(to_jsonable(value), separators=(",", ":"), allow_nan=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})
            handle.flush()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_jacobian_record(
    label: str,
    q_rad: np.ndarray,
    dh: DHParameters,
    T_F_M: np.ndarray,
    cfg: ValidationConfig,
    oracle: ReadOnlyURRTDEOracle | None,
) -> dict[str, Any]:
    own_analytic = geometric_jacobian(q_rad, dh, T_F_M)
    own_fd = central_difference_jacobian(
        lambda q: forward_kinematics(q, dh, T_F_M).T_R_target,
        q_rad,
        cfg.finite_difference_joint_step_rad,
    )
    matrices: dict[str, np.ndarray] = {
        "own_analytic": own_analytic,
        "own_finite_difference": own_fd,
    }
    warnings: list[str] = []
    if oracle is not None:
        try:
            matrices["ur_rtde_analytic"] = oracle.jacobian(
                q_rad, T_F_M, cfg.rtde_jacobian_row_order
            )
        except Exception as exc:
            warnings.append(f"ur_rtde getJacobian unavailable: {type(exc).__name__}: {exc}")
        if cfg.calculate_rtde_fk_finite_difference_jacobian:
            try:
                matrices["ur_rtde_fk_finite_difference"] = central_difference_jacobian(
                    lambda q: oracle.forward_kinematics(q, T_F_M),
                    q_rad,
                    cfg.finite_difference_joint_step_rad,
                )
            except Exception as exc:
                warnings.append(
                    f"ur_rtde FK finite difference unavailable: {type(exc).__name__}: {exc}"
                )

    comparisons: dict[str, Any] = {
        "own_analytic_vs_own_finite_difference": jacobian_difference(
            own_analytic, own_fd
        )
    }
    for name in ("ur_rtde_analytic", "ur_rtde_fk_finite_difference"):
        if name in matrices:
            comparisons[f"own_analytic_vs_{name}"] = jacobian_difference(
                own_analytic, matrices[name]
            )
    if "ur_rtde_analytic" in matrices and "ur_rtde_fk_finite_difference" in matrices:
        comparisons["ur_rtde_analytic_vs_ur_rtde_fk_finite_difference"] = (
            jacobian_difference(
                matrices["ur_rtde_analytic"],
                matrices["ur_rtde_fk_finite_difference"],
            )
        )
    conditions = {
        name: {
            "raw_condition_number": finite_or_none(condition_number(matrix)),
            "scaled_condition_number": finite_or_none(
                condition_number(matrix, cfg.orientation_characteristic_length_m)
            ),
        }
        for name, matrix in matrices.items()
    }
    return {
        "label": label,
        "q_rad": q_rad.copy(),
        "matrices": matrices,
        "conditions": conditions,
        "comparisons": comparisons,
        "warnings": warnings,
    }


def ik_result_row(
    target: IKTargetOffset,
    T_R_M_target: np.ndarray,
    own: IKResult,
    q_actual: np.ndarray,
    q_rtde: np.ndarray | None,
    rtde_status: str,
    dh: DHParameters,
    T_F_M: np.ndarray,
    oracle: ReadOnlyURRTDEOracle | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    own_model_at_own = forward_kinematics(own.q_rad, dh, T_F_M).T_R_target
    detail: dict[str, Any] = {
        "target": target,
        "T_R_M_target": T_R_M_target,
        "own_ik": own,
        "ur_rtde_status": rtde_status,
        "q_ur_rtde_rad": q_rtde,
        "own_model_at_own_solution": pose_error_metrics(own_model_at_own, T_R_M_target),
    }
    row: dict[str, Any] = {
        "target_name": target.name,
        "target_frame": target.frame,
        "target_translation_mm": target.translation_mm,
        "target_rotation_vector_deg": target.rotation_vector_deg,
        "target_T_R_M_pose6": T_to_pose6(T_R_M_target),
        "own_converged": own.converged,
        "own_iterations": own.iterations,
        "own_reason": own.reason,
        "q_actual_rad": q_actual,
        "q_own_rad": own.q_rad,
        "own_delta_from_actual_wrapped_rad": wrapped_joint_difference(own.q_rad, q_actual),
        "own_model_own_solution_position_error_mm": 1.0e3 * own.final_position_error_m,
        "own_model_own_solution_orientation_error_deg": np.degrees(
            own.final_orientation_error_rad
        ),
        "own_model_T_R_M_at_own_solution_pose6": T_to_pose6(own_model_at_own),
        "own_final_scaled_jacobian_condition": finite_or_none(
            own.final_scaled_jacobian_condition
        ),
        "ur_rtde_status": rtde_status,
        "q_ur_rtde_rad": q_rtde,
    }

    if oracle is not None:
        own_solution_rtde_fk = oracle.forward_kinematics(own.q_rad, T_F_M)
        detail["ur_rtde_fk_at_own_solution"] = pose_error_metrics(
            own_solution_rtde_fk, T_R_M_target
        )
        row.update(
            {
                "ur_rtde_T_R_M_at_own_solution_pose6": T_to_pose6(
                    own_solution_rtde_fk
                ),
                "ur_rtde_fk_own_solution_position_error_mm": detail[
                    "ur_rtde_fk_at_own_solution"
                ]["position_error_mm"],
                "ur_rtde_fk_own_solution_orientation_error_deg": detail[
                    "ur_rtde_fk_at_own_solution"
                ]["orientation_error_deg"],
                "own_solution_ur_safety_limits": oracle.joints_within_safety_limits(
                    own.q_rad
                ),
            }
        )

    if q_rtde is not None:
        own_model_at_rtde = forward_kinematics(q_rtde, dh, T_F_M).T_R_target
        detail["own_model_at_ur_rtde_solution"] = pose_error_metrics(
            own_model_at_rtde, T_R_M_target
        )
        row.update(
            {
                "own_model_T_R_M_at_ur_rtde_solution_pose6": T_to_pose6(
                    own_model_at_rtde
                ),
                "ur_rtde_delta_from_actual_wrapped_rad": wrapped_joint_difference(
                    q_rtde, q_actual
                ),
                "own_vs_ur_rtde_solution_wrapped_rad": wrapped_joint_difference(
                    own.q_rad, q_rtde
                ),
                "own_vs_ur_rtde_solution_wrapped_norm_rad": float(
                    np.linalg.norm(wrapped_joint_difference(own.q_rad, q_rtde))
                ),
                "own_model_ur_rtde_solution_position_error_mm": detail[
                    "own_model_at_ur_rtde_solution"
                ]["position_error_mm"],
                "own_model_ur_rtde_solution_orientation_error_deg": detail[
                    "own_model_at_ur_rtde_solution"
                ]["orientation_error_deg"],
            }
        )
        if oracle is not None:
            rtde_fk_at_rtde = oracle.forward_kinematics(q_rtde, T_F_M)
            detail["ur_rtde_fk_at_ur_rtde_solution"] = pose_error_metrics(
                rtde_fk_at_rtde, T_R_M_target
            )
            row.update(
                {
                    "ur_rtde_T_R_M_at_ur_rtde_solution_pose6": T_to_pose6(
                        rtde_fk_at_rtde
                    ),
                    "ur_rtde_fk_ur_rtde_solution_position_error_mm": detail[
                        "ur_rtde_fk_at_ur_rtde_solution"
                    ]["position_error_mm"],
                    "ur_rtde_fk_ur_rtde_solution_orientation_error_deg": detail[
                        "ur_rtde_fk_at_ur_rtde_solution"
                    ]["orientation_error_deg"],
                    "ur_rtde_solution_ur_safety_limits": oracle.joints_within_safety_limits(
                        q_rtde
                    ),
                }
            )
    return row, detail


def jacobian_csv_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    for record in records:
        for source, matrix in record["matrices"].items():
            conditions = record["conditions"][source]
            summary_rows.append(
                {
                    "label": record["label"],
                    "source": source,
                    "q_rad": record["q_rad"],
                    **conditions,
                }
            )
            for row in range(6):
                for column in range(6):
                    element_rows.append(
                        {
                            "label": record["label"],
                            "source": source,
                            "row": row,
                            "row_quantity": (
                                ("vx", "vy", "vz", "wx", "wy", "wz")[row]
                            ),
                            "joint_column": column + 1,
                            "value": float(matrix[row, column]),
                        }
                    )
        for comparison, metrics in record["comparisons"].items():
            summary_rows.append(
                {
                    "label": record["label"],
                    "source": comparison,
                    **metrics,
                }
            )
    return summary_rows, element_rows


def set_equal_3d_axes(axis: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = max(0.05, 0.55 * float(np.max(np.ptp(points, axis=0))))
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)


def plot_results(
    output_dir: Path,
    cfg: ValidationConfig,
    dh: DHParameters,
    snapshot: RobotSnapshot,
    T_R_M_inferred: np.ndarray,
    target_details: list[dict[str, Any]],
    jacobian_records: list[dict[str, Any]],
) -> None:
    if not cfg.make_plots:
        return
    import matplotlib

    if not cfg.show_plots_interactively:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    ax_3d = figure.add_subplot(2, 2, 1, projection="3d")
    actual_fk = forward_kinematics(snapshot.q_rad, dh, np.eye(4))
    links = actual_fk.link_points_R_m
    ax_3d.plot(links[:, 0], links[:, 1], links[:, 2], "o-", label="independent DH arm")
    ax_3d.scatter(*T_R_M_inferred[:3, 3], s=70, marker="*", label="robot-inferred actual M")
    all_points = [*links, T_R_M_inferred[:3, 3]]
    for detail in target_details:
        p = np.asarray(detail["T_R_M_target"])[:3, 3]
        all_points.append(p)
        ax_3d.scatter(*p, s=30, label=f"target: {detail['target'].name}")
    ax_3d.set_title("Robot and magnet targets in UR base frame R")
    ax_3d.set_xlabel("R.x [m]")
    ax_3d.set_ylabel("R.y [m]")
    ax_3d.set_zlabel("R.z [m]")
    ax_3d.legend(fontsize=7)
    set_equal_3d_axes(ax_3d, np.asarray(all_points))

    ax_joint = figure.add_subplot(2, 2, 2)
    names = [detail["target"].name for detail in target_details]
    own_norms = [
        np.linalg.norm(wrapped_joint_difference(detail["own_ik"].q_rad, snapshot.q_rad))
        for detail in target_details
    ]
    rtde_norms = [
        np.nan
        if detail["q_ur_rtde_rad"] is None
        else np.linalg.norm(wrapped_joint_difference(detail["q_ur_rtde_rad"], snapshot.q_rad))
        for detail in target_details
    ]
    x = np.arange(len(names))
    ax_joint.bar(x - 0.2, own_norms, 0.4, label="independent IK")
    ax_joint.bar(x + 0.2, rtde_norms, 0.4, label="ur_rtde IK")
    ax_joint.set_xticks(x, names, rotation=30, ha="right")
    ax_joint.set_ylabel("wrapped ||q - q_actual|| [rad]")
    ax_joint.set_title("IK branch displacement from actual joints")
    ax_joint.legend()

    ax_jac = figure.add_subplot(2, 2, 3)
    actual_record = jacobian_records[0]
    if "ur_rtde_analytic" in actual_record["matrices"]:
        difference = (
            actual_record["matrices"]["own_analytic"]
            - actual_record["matrices"]["ur_rtde_analytic"]
        )
        title = "Own analytic J - ur_rtde J at actual q"
    else:
        difference = (
            actual_record["matrices"]["own_analytic"]
            - actual_record["matrices"]["own_finite_difference"]
        )
        title = "Own analytic J - own finite-difference J"
    image = ax_jac.imshow(difference, cmap="coolwarm", aspect="auto")
    ax_jac.set_title(title)
    ax_jac.set_xlabel("joint column")
    ax_jac.set_ylabel("[vx vy vz wx wy wz] row")
    figure.colorbar(image, ax=ax_jac, shrink=0.85)

    ax_pose = figure.add_subplot(2, 2, 4)
    own_position = [1.0e3 * d["own_ik"].final_position_error_m for d in target_details]
    own_orientation = [np.degrees(d["own_ik"].final_orientation_error_rad) for d in target_details]
    ax_pose.semilogy(x, np.maximum(own_position, 1.0e-12), "o-", label="own IK position [mm]")
    ax_pose.semilogy(x, np.maximum(own_orientation, 1.0e-12), "s-", label="own IK orientation [deg]")
    ax_pose.set_xticks(x, names, rotation=30, ha="right")
    ax_pose.set_title("Independent IK terminal residual")
    ax_pose.grid(True, which="both", alpha=0.3)
    ax_pose.legend()

    figure.savefig(output_dir / "validation_overview.png", dpi=180)
    if cfg.show_plots_interactively:
        plt.show()
    plt.close(figure)


def check_threshold(
    checks: list[dict[str, Any]], name: str, value: float, maximum: float
) -> None:
    checks.append(
        {
            "name": name,
            "value": float(value),
            "maximum": float(maximum),
            "passed": bool(np.isfinite(value) and value <= maximum),
        }
    )


def main() -> None:
    validate_config(CONFIG)
    dh = corrected_dh_from_config(CONFIG)
    T_TCP_M = magnet_offset_from_config(CONFIG)
    output_dir = make_output_directory(CONFIG)
    oracle: ReadOnlyURRTDEOracle | None = None
    try:
        if CONFIG.use_live_robot:
            oracle = ReadOnlyURRTDEOracle(CONFIG)
            snapshot = oracle.snapshot()
        else:
            snapshot = offline_snapshot(CONFIG, dh)

        T_F_M = validate_transform(snapshot.T_F_TCP @ T_TCP_M, "T_F_M")
        T_R_M_inferred = validate_transform(
            snapshot.actual_T_R_TCP @ T_TCP_M, "robot-inferred T_R_M"
        )
        own_at_actual = forward_kinematics(snapshot.q_rad, dh, snapshot.T_F_TCP)
        own_magnet_at_actual = forward_kinematics(snapshot.q_rad, dh, T_F_M).T_R_target
        joint_snapshot_drift = float(
            np.linalg.norm(wrapped_joint_difference(snapshot.q_after_rad, snapshot.q_before_rad))
        )

        actual_state: dict[str, Any] = {
            "q_actual_rad": snapshot.q_rad,
            "q_before_rad": snapshot.q_before_rad,
            "q_after_rad": snapshot.q_after_rad,
            "snapshot_wrapped_joint_drift_norm_rad": joint_snapshot_drift,
            "T_F_TCP": snapshot.T_F_TCP,
            "T_TCP_M": T_TCP_M,
            "T_F_M": T_F_M,
            "T_R_TCP_reported": snapshot.actual_T_R_TCP,
            "T_R_M_robot_inferred": T_R_M_inferred,
            "T_R_F_reported_if_available": snapshot.actual_T_R_F_reported,
            "T_R_TCP_own_DH_at_actual_q": own_at_actual.T_R_target,
            "T_R_M_own_DH_at_actual_q": own_magnet_at_actual,
            "reported_tcp_vs_own_DH": pose_error_metrics(
                own_at_actual.T_R_target, snapshot.actual_T_R_TCP
            ),
        }
        checks: list[dict[str, Any]] = []
        check_threshold(
            checks,
            "stationary snapshot joint drift norm",
            joint_snapshot_drift,
            CONFIG.maximum_snapshot_joint_drift_rad,
        )

        if CONFIG.independent_measured_T_robot_magnet_pose6 is not None:
            independent_magnet = pose6_to_T(
                CONFIG.independent_measured_T_robot_magnet_pose6
            )
            actual_state["T_R_M_independent_measurement"] = independent_magnet
            actual_state["robot_inferred_vs_independent_magnet_measurement"] = (
                pose_error_metrics(T_R_M_inferred, independent_magnet)
            )

        if oracle is not None:
            rtde_tcp_fk = oracle.forward_kinematics(snapshot.q_rad, snapshot.T_F_TCP)
            rtde_magnet_fk = oracle.forward_kinematics(snapshot.q_rad, T_F_M)
            actual_state["T_R_TCP_ur_rtde_calibrated_FK"] = rtde_tcp_fk
            actual_state["T_R_M_ur_rtde_calibrated_FK"] = rtde_magnet_fk
            actual_state["reported_tcp_vs_ur_rtde_calibrated_FK"] = pose_error_metrics(
                rtde_tcp_fk, snapshot.actual_T_R_TCP
            )
            actual_state["own_DH_vs_ur_rtde_calibrated_TCP_FK"] = pose_error_metrics(
                own_at_actual.T_R_target, rtde_tcp_fk
            )
            actual_state["own_DH_vs_ur_rtde_calibrated_magnet_FK"] = pose_error_metrics(
                own_magnet_at_actual, rtde_magnet_fk
            )
            rtde_actual_error = actual_state["reported_tcp_vs_ur_rtde_calibrated_FK"]
            check_threshold(
                checks,
                "reported actual TCP versus ur_rtde calibrated FK position",
                rtde_actual_error["position_error_m"],
                CONFIG.maximum_actual_tcp_vs_rtde_fk_position_error_m,
            )
            check_threshold(
                checks,
                "reported actual TCP versus ur_rtde calibrated FK orientation",
                rtde_actual_error["orientation_error_rad"],
                CONFIG.maximum_actual_tcp_vs_rtde_fk_orientation_error_rad,
            )

        target_rows: list[dict[str, Any]] = []
        target_details: list[dict[str, Any]] = []
        jacobian_records = [
            build_jacobian_record(
                "actual_q", snapshot.q_rad, dh, T_F_M, CONFIG, oracle
            )
        ]

        for target in CONFIG.targets:
            T_R_M_target = apply_target_offset(T_R_M_inferred, target)
            own_ik = inverse_kinematics_dls(
                T_R_M_target,
                snapshot.q_rad,
                dh,
                T_F_M,
                CONFIG,
            )
            q_rtde = None
            rtde_status = "offline_not_requested"
            if oracle is not None:
                # ur_rtde IK operates on the ACTIVE TCP.  Convert the desired
                # magnet pose back through T_TCP_M; do not call setTcp().
                T_R_TCP_target = validate_transform(
                    T_R_M_target @ inverse_T(T_TCP_M), "T_R_TCP_target"
                )
                q_rtde, rtde_status = oracle.inverse_kinematics(
                    T_R_TCP_target,
                    snapshot.q_rad,
                    CONFIG.ik_position_tolerance_m,
                    CONFIG.ik_orientation_tolerance_rad,
                )
            row, detail = ik_result_row(
                target,
                T_R_M_target,
                own_ik,
                snapshot.q_rad,
                q_rtde,
                rtde_status,
                dh,
                T_F_M,
                oracle,
            )
            target_rows.append(row)
            row["robot_inferred_actual_T_R_M_pose6"] = T_to_pose6(T_R_M_inferred)
            if CONFIG.independent_measured_T_robot_magnet_pose6 is not None:
                row["independent_measured_T_R_M_pose6"] = (
                    CONFIG.independent_measured_T_robot_magnet_pose6
                )
            target_details.append(detail)
            if CONFIG.calculate_jacobians_at_ik_solutions:
                jacobian_records.append(
                    build_jacobian_record(
                        f"{target.name}:own_solution",
                        own_ik.q_rad,
                        dh,
                        T_F_M,
                        CONFIG,
                        oracle,
                    )
                )
                if q_rtde is not None:
                    jacobian_records.append(
                        build_jacobian_record(
                            f"{target.name}:ur_rtde_solution",
                            q_rtde,
                            dh,
                            T_F_M,
                            CONFIG,
                            oracle,
                        )
                    )

        own_fd_error = jacobian_records[0]["comparisons"][
            "own_analytic_vs_own_finite_difference"
        ]["relative_frobenius_error"]
        check_threshold(
            checks,
            "own analytic versus own finite-difference Jacobian",
            own_fd_error,
            CONFIG.maximum_own_vs_fd_relative_jacobian_error,
        )
        rtde_comparison_key = "own_analytic_vs_ur_rtde_analytic"
        if rtde_comparison_key in jacobian_records[0]["comparisons"]:
            check_threshold(
                checks,
                "own analytic versus ur_rtde analytic Jacobian",
                jacobian_records[0]["comparisons"][rtde_comparison_key][
                    "relative_frobenius_error"
                ],
                CONFIG.maximum_own_vs_rtde_relative_jacobian_error,
            )

        jac_summary_rows, jac_element_rows = jacobian_csv_rows(jacobian_records)
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "hardware_execution_tested_by_this_generation": False,
            "live_robot_connection_used": CONFIG.use_live_robot,
            "motion_commands_available_in_this_script": False,
            "magnet_pose_classification": (
                "independently measured"
                if CONFIG.independent_measured_T_robot_magnet_pose6 is not None
                else "robot-inferred from actual TCP and calibrated T_TCP_M"
            ),
            "transform_chain": "T_R_M = T_R_F(q) @ T_F_TCP @ T_TCP_M",
            "ur_rtde_ik_chain": "T_R_TCP_target = T_R_M_target @ inverse(T_TCP_M)",
            "jacobian_convention": (
                "6x6 base-frame geometric Jacobian; rows [vx,vy,vz,wx,wy,wz]; "
                "columns UR joints [base,shoulder,elbow,wrist1,wrist2,wrist3]"
            ),
            "units": {
                "translation": "metres",
                "rotation_vectors_and_joints": "radians",
                "jacobian_linear_rows": "metres per radian",
                "jacobian_angular_rows": "radians per radian",
            },
        }
        summary = {
            "metadata": metadata,
            "configuration": CONFIG,
            "dh_parameters_used": dh,
            "actual_state": actual_state,
            "ik_targets": target_details,
            "jacobian_records": jacobian_records,
            "checks": checks,
            "all_checks_passed": bool(all(check["passed"] for check in checks)),
        }
        save_json(output_dir / "summary.json", summary)
        write_csv(output_dir / "ik_comparison.csv", target_rows)
        write_csv(output_dir / "jacobian_summary.csv", jac_summary_rows)
        write_csv(output_dir / "jacobian_elements.csv", jac_element_rows)
        plot_results(
            output_dir,
            CONFIG,
            dh,
            snapshot,
            T_R_M_inferred,
            target_details,
            jacobian_records,
        )

        print(f"\nValidation results: {output_dir.resolve()}")
        print(f"All configured checks passed: {summary['all_checks_passed']}")
        print(
            "Magnet pose used as actual: "
            f"{metadata['magnet_pose_classification']}"
        )
        if CONFIG.strict_validation and not summary["all_checks_passed"]:
            failed = [check["name"] for check in checks if not check["passed"]]
            raise RuntimeError(f"Strict validation failed: {failed}")
    except (KeyboardInterrupt, Exception):
        # There is deliberately no recovery motion.  Closing connections is the
        # only cleanup performed after any failure or keyboard interrupt.
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == "__main__":
    main()
