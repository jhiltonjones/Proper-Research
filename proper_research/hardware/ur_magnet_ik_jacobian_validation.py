"""UR magnet-pose, inverse-kinematics, Jacobian, and motion validation.

Edit the configuration section below; this script has no command-line
arguments.  Live communication and joint motion use the supplied
``URRTDERobot`` class, in the same style as ``testing_robot_class.py``.

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

For the configured axis-aligned 27 mm offset,
``p_R_M = p_R_TCP + R_R_TCP @ [0, 0, +0.044]``.  The local offset is never
added directly to robot-base z.  A requested magnet target is converted back
to the controller's active TCP with
``T_R_TCP_target = T_R_M_target @ inverse(T_TCP_M)`` so rotations occur about
the magnet centre rather than about the TCP origin.

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
4. Optional hardware comparison: send selected joint solutions with ``move_j``
   and record the actual joints/TCP returned by ``get_joints``/``get_tcp_pose``.

Important: live mode connects a control interface and motion mode moves the
robot.  Connection and motion require separate exact typed confirmations.  Run
only after checking the robot model, frames, target offsets, joint bounds,
speed, acceleration, workspace, payload, and safety configuration.
"""

from __future__ import annotations

import csv
import json
import math
import time
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
    # Terminal diagnostics.  Leave debug_enabled=True while bringing the
    # validator up.  IK progress is printed at iteration 0, every N iterations,
    # and at termination.  Matrix dumps are optional because they are verbose.
    debug_enabled: bool = True
    debug_ik_iteration_interval: int = 10
    debug_rtde_call_interval: int = 1
    debug_print_matrices: bool = False
    debug_print_feedback_samples: bool = True

    # This optional controller query is not used by IK/Jacobian validation.
    # Some ur_rtde/controller combinations block inside
    # getActualToolFlangePose(), so keep it disabled unless diagnosing that API.
    query_actual_tool_flange_pose: bool = False

    # Live state, ur_rtde kinematics, and motion all share one URRTDERobot
    # instance.  Keep ur_rtde_robot.py next to this script (or on PYTHONPATH).
    use_live_robot: bool = False
    robot_ip: str = "192.168.56.101"
    rtde_frequency_hz: float = 125.0
    robot_model: str = "ur10e"  # ur3e, ur5e/ur7e, ur10e/ur12e, ur16e,
                                  # ur3, ur5, or ur10
    live_confirmation_phrase: str = "ready"

    # Hardware comparison.  The default sends both solutions for one selected
    # target: custom IK first, then ur_rtde IK.  Set execute_motion=False for a
    # calculation-only run.  Empty motion_target_names means all targets.
    execute_motion: bool = False
    motion_confirmation_phrase: str = "MOVE"
    motion_solution_sources: tuple[str, ...] = ("own", "ur_rtde")
    # Only names listed here are actually sent to the robot.  Use a small
    # magnet-local Y tilt so the 44 mm Z lever arm produces visible TCP motion.
    # A local-Z spin changes orientation, but cannot move a collinear Z offset.
    motion_target_names: tuple[str, ...] = ("magnet_y_tilt_minus_10deg",)
    move_joint_speed_rad_s: float = 0.25
    move_joint_acceleration_rad_s2: float = 0.20
    maximum_commanded_joint_delta_norm_rad: float = 0.35
    maximum_commanded_per_joint_delta_rad: tuple[float, ...] = (0.25,) * 6
    feedback_sample_count: int = 5
    feedback_sample_interval_s: float = 0.05

    # T_TCP_M: magnet-centre/body pose expressed in the LOCAL ACTIVE-TCP frame.
    # The magnet centre is 44 mm along TCP +z and its axes are aligned with the
    # TCP axes.  This is not a robot-base z offset: R_R_TCP rotates this lever
    # arm whenever the tool rotates.  Change the final three rotation-vector
    # values if the magnet body axes are not physically aligned with the TCP.
    # The logged downward reference pose shows that TCP +z points toward the
    # ground, so +0.044 m produces a negative robot-base z displacement.  Do
    # not enter -0.044 merely because "down" is robot-base -z: this translation
    # is expressed in the rotating TCP frame, not the robot-base frame.
    T_tcp_magnet_pose6: tuple[float, float, float, float, float, float] | None = (
        0.0,
        0.0,
        0.044,
        0.0,
        0.0,
        0.0,
    )
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
            IKTargetOffset("robot_x_minus_100mm", (-100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("robot_y_minus_100mm", (0.0, -100.0, 0.0), (0.0, 0.0, 0.0)),
            IKTargetOffset("robot_z_minus_100mm", (0.0, 0.0, -100.0), (0.0, 0.0, 0.0)),
            # Tilting about local X or Y rotates the Z lever arm and therefore
            # requires the TCP to move around the fixed magnet centre.
            IKTargetOffset(
                "magnet_y_tilt_minus_10deg",
                (0.0, 0.0, 0.0),
                (0.0, -10.0, 0.0),
                "magnet_local",
            ),
            # This is a useful control case: it changes magnet orientation but
            # does not translate a TCP-to-magnet offset lying on local Z.
            IKTargetOffset(
                "magnet_z_spin_minus_10deg",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, -10.0),
                "magnet_local",
            ),
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
    run_name: str = "ik_comparison_with_joint_feedback"
    make_plots: bool = True
    show_plots_interactively: bool = False


CONFIG = ValidationConfig()


def debug_print(
    cfg: ValidationConfig,
    stage: str,
    message: str,
    value: Any | None = None,
) -> None:
    """Print one timestamped, immediately flushed diagnostic line."""

    if not cfg.debug_enabled:
        return
    timestamp = datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3]
    prefix = f"[UR-IK DEBUG][{timestamp}][{stage}]"
    if value is None:
        print(f"{prefix} {message}", flush=True)
        return
    if isinstance(value, np.ndarray):
        rendered = np.array2string(
            value,
            precision=8,
            suppress_small=False,
            max_line_width=160,
        )
    else:
        rendered = repr(value)
    print(f"{prefix} {message}: {rendered}", flush=True)


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


def magnet_pose_from_tcp(
    T_A_TCP: Any,
    T_TCP_M: Any,
    name: str = "T_A_M",
) -> np.ndarray:
    """Map the rigid TCP-to-magnet offset into any parent frame A.

    The component equations are::

        p_A_M = p_A_TCP + R_A_TCP @ p_TCP_M
        R_A_M = R_A_TCP @ R_TCP_M

    Thus a TCP rotation rotates the 27 mm lever arm as well as the magnet axes.
    """

    parent_tcp = validate_transform(T_A_TCP, "T_A_TCP")
    tcp_magnet = validate_transform(T_TCP_M, "T_TCP_M")
    return validate_transform(parent_tcp @ tcp_magnet, name)


def tcp_pose_for_magnet_target(
    T_A_M_target: Any,
    T_TCP_M: Any,
    name: str = "T_A_TCP_target",
) -> np.ndarray:
    """Return the TCP pose that realizes a requested magnet-centre pose.

    The robot accepts a TCP target, while this validator defines motion at M::

        T_A_TCP_target = T_A_M_target @ inverse(T_TCP_M)

    For a pure rotation about the magnet centre, the magnet target translation
    stays fixed and the resulting TCP translation moves around that centre.
    """

    magnet_target = validate_transform(T_A_M_target, "T_A_M_target")
    tcp_magnet = validate_transform(T_TCP_M, "T_TCP_M")
    tcp_target = validate_transform(magnet_target @ inverse_T(tcp_magnet), name)
    reconstructed_magnet = magnet_pose_from_tcp(
        tcp_target,
        tcp_magnet,
        "reconstructed magnet target",
    )
    if not np.allclose(reconstructed_magnet, magnet_target, atol=1.0e-10):
        raise RuntimeError("TCP/magnet target conversion failed its round-trip check.")
    return tcp_target


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


def compose_rotation_increment(
    R_current: Any,
    rotation_vector_deg: Any,
    frame: str,
) -> np.ndarray:
    """Compose an axis-angle increment with an existing orientation.

    A UR rotation vector is an axis-angle parameterisation, not three Euler
    angles and not a vector that may be added component-by-component.  Convert
    the requested increment to a rotation matrix, multiply in the requested
    frame, and only convert the final matrix back to a rotation vector when a
    UR pose6 value is required.

    ``robot_base`` uses a spatial/world increment (pre-multiplication), while
    ``magnet_local`` uses a body/local increment (post-multiplication).
    """

    current = np.asarray(R_current, dtype=float)
    if current.shape != (3, 3) or not np.all(np.isfinite(current)):
        raise ValueError("R_current must be a finite 3x3 rotation matrix.")
    current_check = np.eye(4, dtype=float)
    current_check[:3, :3] = current
    current = validate_transform(current_check, "R_current")[:3, :3]

    delta_rotvec_rad = np.radians(
        finite_vector(rotation_vector_deg, 3, "rotation_vector_deg")
    )
    R_delta = Rot.from_rotvec(delta_rotvec_rad).as_matrix()
    if frame == "robot_base":
        R_target = R_delta @ current
    elif frame == "magnet_local":
        R_target = current @ R_delta
    else:
        raise ValueError(f"Unknown target frame {frame!r}.")

    target_check = np.eye(4, dtype=float)
    target_check[:3, :3] = R_target
    return validate_transform(target_check, "composed target rotation")[:3, :3]


def rotation_increment_diagnostics(
    T_current: Any,
    T_target: Any,
    target: IKTargetOffset,
) -> dict[str, Any]:
    """Describe the composed increment using matrices, not raw pose6 subtraction."""

    current = validate_transform(T_current, "rotation diagnostic current")
    result = validate_transform(T_target, "rotation diagnostic target")
    R_current = current[:3, :3]
    R_target = result[:3, :3]

    if target.frame == "robot_base":
        # R_target = R_delta R_current -> R_delta = R_target R_current^T
        R_relative = R_target @ R_current.T
    elif target.frame == "magnet_local":
        # R_target = R_current R_delta -> R_delta = R_current^T R_target
        R_relative = R_current.T @ R_target
    else:
        raise ValueError(f"Unknown target frame {target.frame!r}.")

    achieved_rotvec_rad = Rot.from_matrix(R_relative).as_rotvec()
    achieved_angle_rad = float(np.linalg.norm(achieved_rotvec_rad))
    if achieved_angle_rad > 1.0e-12:
        achieved_axis = achieved_rotvec_rad / achieved_angle_rad
    else:
        achieved_axis = np.zeros(3, dtype=float)

    requested_rotvec_rad = np.radians(
        finite_vector(target.rotation_vector_deg, 3, "rotation_vector_deg")
    )
    R_requested = Rot.from_rotvec(requested_rotvec_rad).as_matrix()
    composition_error_rad = float(
        np.linalg.norm(
            Rot.from_matrix(R_relative @ R_requested.T).as_rotvec()
        )
    )
    return {
        "requested_rotation_vector_deg": np.asarray(
            target.rotation_vector_deg, dtype=float
        ),
        "achieved_relative_rotation_vector_deg": np.degrees(
            achieved_rotvec_rad
        ),
        "achieved_relative_axis": achieved_axis,
        "achieved_relative_angle_deg": float(np.degrees(achieved_angle_rad)),
        "rotation_matrix_composition_error_deg": float(
            np.degrees(composition_error_rad)
        ),
    }


def apply_target_offset(T_R_M: Any, target: IKTargetOffset) -> np.ndarray:
    """Displace M about its current centre without rotating its position."""

    current = validate_transform(T_R_M, "T_R_M")
    translation = 1.0e-3 * finite_vector(target.translation_mm, 3, "translation_mm")
    result = current.copy()
    if target.frame == "robot_base":
        result[:3, 3] = current[:3, 3] + translation
    elif target.frame == "magnet_local":
        result[:3, 3] = current[:3, 3] + current[:3, :3] @ translation
    else:
        raise ValueError(f"Unknown target frame {target.frame!r}.")
    result[:3, :3] = compose_rotation_increment(
        current[:3, :3],
        target.rotation_vector_deg,
        target.frame,
    )
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
    debug_print(cfg, "CUSTOM IK", "seed q [rad]", q)
    debug_print(cfg, "CUSTOM IK", "target pose [m, rotvec rad]", T_to_pose6(target))

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
        if iteration == 0 or iteration % cfg.debug_ik_iteration_interval == 0:
            debug_print(
                cfg,
                "CUSTOM IK",
                (
                    f"iteration={iteration} position_error={1.0e3 * position_error:.6f} mm "
                    f"orientation_error={np.degrees(orientation_error):.6f} deg "
                    f"scaled_cost={cost:.6e} m condition={scaled_condition:.6e} "
                    f"damping={damping:.6e}"
                ),
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
            debug_print(cfg, "CUSTOM IK", "linear solve failed")
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
            debug_print(
                cfg,
                "CUSTOM IK",
                f"line search rejected every step; damping increased to {damping:.6e}",
            )
            if damping > cfg.ik_maximum_damping_m_per_rad:
                reason = "line_search_stalled"
                break

    final_T = forward_kinematics(q, dh, T_F_target).T_R_target
    final_error = pose_error_spatial(final_T, target)
    debug_print(
        cfg,
        "CUSTOM IK",
        (
            f"finished converged={converged} reason={reason} iterations={len(history) - 1} "
            f"position_error={1.0e3 * np.linalg.norm(final_error[:3]):.6f} mm "
            f"orientation_error={np.degrees(np.linalg.norm(final_error[3:])):.6f} deg"
        ),
    )
    debug_print(cfg, "CUSTOM IK", "solution q [rad]", q)
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
# URRTDERobot ADAPTER FOR STATE, KINEMATICS, AND MOTION
# =============================================================================


@dataclass
class RobotSnapshot:
    q_rad: np.ndarray
    actual_T_R_TCP: np.ndarray
    T_F_TCP: np.ndarray
    q_before_rad: np.ndarray
    q_after_rad: np.ndarray
    actual_T_R_F_reported: np.ndarray | None


class URRTDERobotAdapter:
    """Use one supplied ``URRTDERobot`` connection for the entire validation.

    The supplied wrapper intentionally exposes only ordinary state and motion
    methods.  ur_rtde's FK, IK, Jacobian, TCP-offset, and safety-limit queries
    are therefore called on that wrapper's existing control interface.  This
    avoids starting a second RTDE control session or modifying the wrapper.
    """

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg
        self._rtde_call_counts: dict[str, int] = {}
        print(
            "\nLIVE CONNECTION WARNING:\n"
            "  This validator uses URRTDERobot and may move the robot.\n"
            "  Stop other robot programs and verify the complete configuration.\n"
            f"  Configured motion execution: {cfg.execute_motion}.\n"
        )
        confirmation = input(
            f"Type exactly {cfg.live_confirmation_phrase!r} to connect: "
        )
        if confirmation != cfg.live_confirmation_phrase:
            raise RuntimeError("Live connection cancelled: confirmation did not match.")
        debug_print(cfg, "CONNECT", "connection confirmation accepted")
        try:
            debug_print(cfg, "CONNECT", "importing URRTDERobot from ur_rtde_robot")
            from ur_rtde_robot import URRTDERobot
        except ImportError as exc:
            raise RuntimeError(
                "Could not import URRTDERobot. Keep ur_rtde_robot.py beside "
                "this script and install ur_rtde in this Python environment."
            ) from exc
        debug_print(
            cfg,
            "CONNECT",
            f"opening URRTDERobot at {cfg.robot_ip}, frequency={cfg.rtde_frequency_hz} Hz",
        )
        self.robot = URRTDERobot(cfg.robot_ip, frequency=cfg.rtde_frequency_hz)
        debug_print(
            cfg,
            "CONNECT",
            f"URRTDERobot returned; is_connected={self.robot.is_connected()}",
        )
        self.control = getattr(self.robot, "_control", None)
        if self.control is None:
            self.robot.close()
            raise RuntimeError(
                "URRTDERobot connected without an accessible control interface; "
                "controller kinematics cannot be compared without opening a "
                "conflicting second control session."
            )
        debug_print(
            cfg,
            "CONNECT",
            f"control interface ready: {type(self.control).__name__}",
        )
        try:
            debug_print(cfg, "CONNECT", f"robot_mode={self.robot.get_robot_mode()}")
            debug_print(cfg, "CONNECT", f"safety_mode={self.robot.get_safety_mode()}")
            debug_print(
                cfg,
                "CONNECT",
                f"protective_stopped={self.robot.is_protective_stopped()}",
            )
        except Exception as exc:
            debug_print(
                cfg,
                "CONNECT",
                f"state diagnostic query failed: {type(exc).__name__}: {exc}",
            )

    def _trace_rtde_call(self, name: str, message: str) -> int:
        """Trace the first calls and periodic high-volume kinematic calls."""

        count = self._rtde_call_counts.get(name, 0) + 1
        self._rtde_call_counts[name] = count
        if count <= 3 or count % self.cfg.debug_rtde_call_interval == 0:
            debug_print(self.cfg, "RTDE CALL", f"{name} #{count}: {message}")
        return count

    def _trace_rtde_result(self, name: str, count: int, message: str) -> None:
        if count <= 3 or count % self.cfg.debug_rtde_call_interval == 0:
            debug_print(self.cfg, "RTDE RETURN", f"{name} #{count}: {message}")

    def close(self) -> None:
        robot = getattr(self, "robot", None)
        if robot is not None:
            debug_print(self.cfg, "CLOSE", "closing URRTDERobot connection")
            robot.close()
            debug_print(self.cfg, "CLOSE", "URRTDERobot connection closed")

    def snapshot(self) -> RobotSnapshot:
        debug_print(self.cfg, "SNAPSHOT", "requesting first actual joint sample")
        q_before = finite_vector(self.robot.get_joints(), 6, "get_joints before")
        debug_print(self.cfg, "SNAPSHOT", "q_before [rad]", q_before)
        debug_print(self.cfg, "SNAPSHOT", "requesting actual TCP pose")
        actual_tcp = pose6_to_T(self.robot.get_tcp_pose())
        debug_print(
            self.cfg,
            "SNAPSHOT",
            "actual TCP pose [m, rotvec rad]",
            T_to_pose6(actual_tcp),
        )
        debug_print(self.cfg, "SNAPSHOT", "requesting second actual joint sample")
        q_after = finite_vector(self.robot.get_joints(), 6, "get_joints after")
        debug_print(self.cfg, "SNAPSHOT", "q_after [rad]", q_after)
        q = q_before + 0.5 * wrapped_joint_difference(q_after, q_before)
        debug_print(self.cfg, "SNAPSHOT", "requesting active TCP offset")
        T_F_TCP = pose6_to_T(self.control.getTCPOffset())
        debug_print(
            self.cfg,
            "SNAPSHOT",
            "active T_F_TCP pose [m, rotvec rad]",
            T_to_pose6(T_F_TCP),
        )
        flange = None
        if self.cfg.query_actual_tool_flange_pose:
            if hasattr(self.control, "getActualToolFlangePose"):
                debug_print(
                    self.cfg,
                    "SNAPSHOT",
                    "requesting optional actual tool-flange pose",
                )
                try:
                    flange = pose6_to_T(self.control.getActualToolFlangePose())
                    debug_print(
                        self.cfg,
                        "SNAPSHOT",
                        "optional actual tool-flange pose returned",
                        T_to_pose6(flange),
                    )
                except Exception as exc:
                    debug_print(
                        self.cfg,
                        "SNAPSHOT",
                        (
                            "optional tool-flange query failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            else:
                debug_print(
                    self.cfg,
                    "SNAPSHOT",
                    "optional getActualToolFlangePose API is unavailable",
                )
        else:
            debug_print(
                self.cfg,
                "SNAPSHOT",
                "skipping optional getActualToolFlangePose query (disabled)",
            )
        debug_print(self.cfg, "SNAPSHOT", "stationary snapshot complete")
        return RobotSnapshot(q, actual_tcp, T_F_TCP, q_before, q_after, flange)

    def forward_kinematics(self, q_rad: Iterable[float], T_F_target: Any) -> np.ndarray:
        q = finite_vector(q_rad, 6, "q_rad")
        count = self._trace_rtde_call(
            "getForwardKinematics",
            f"q={q.tolist()}",
        )
        pose = self.control.getForwardKinematics(
            q.tolist(),
            T_to_pose6(T_F_target).tolist(),
        )
        result = pose6_to_T(pose)
        self._trace_rtde_result(
            "getForwardKinematics",
            count,
            f"pose={T_to_pose6(result).tolist()}",
        )
        return result

    def inverse_kinematics(
        self,
        T_R_TCP_target: Any,
        q_near_rad: Iterable[float],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> tuple[np.ndarray | None, str]:
        pose = T_to_pose6(T_R_TCP_target).tolist()
        q_near = finite_vector(q_near_rad, 6, "q_near_rad").tolist()
        debug_print(self.cfg, "UR RTDE IK", f"target TCP pose={pose}")
        debug_print(self.cfg, "UR RTDE IK", f"q_near={q_near}")
        try:
            if hasattr(self.control, "getInverseKinematicsHasSolution"):
                debug_print(self.cfg, "UR RTDE IK", "calling has-solution query")
                has_solution = self.control.getInverseKinematicsHasSolution(
                    pose,
                    q_near,
                    float(position_tolerance_m),
                    float(orientation_tolerance_rad),
                )
                debug_print(self.cfg, "UR RTDE IK", f"has_solution={has_solution}")
                if not has_solution:
                    return None, "ur_rtde_reported_no_solution"
            debug_print(self.cfg, "UR RTDE IK", "calling getInverseKinematics")
            result = self.control.getInverseKinematics(
                pose,
                q_near,
                float(position_tolerance_m),
                float(orientation_tolerance_rad),
            )
            q_result = finite_vector(result, 6, "ur_rtde IK result")
            debug_print(self.cfg, "UR RTDE IK", "solution q [rad]", q_result)
            return q_result, "solution"
        except Exception as exc:
            debug_print(
                self.cfg,
                "UR RTDE IK",
                f"exception: {type(exc).__name__}: {exc}",
            )
            return None, f"ur_rtde_exception: {type(exc).__name__}: {exc}"

    def jacobian(self, q_rad: Iterable[float], T_F_target: Any, row_order: str) -> np.ndarray:
        q = finite_vector(q_rad, 6, "q_rad")
        count = self._trace_rtde_call("getJacobian", f"q={q.tolist()}")
        raw = self.control.getJacobian(
            q.tolist(),
            T_to_pose6(T_F_target).tolist(),
        )
        J = np.asarray(raw, dtype=float).reshape(6, 6)
        self._trace_rtde_result(
            "getJacobian",
            count,
            f"shape={J.shape}, finite={bool(np.all(np.isfinite(J)))}",
        )
        if self.cfg.debug_print_matrices:
            debug_print(self.cfg, "UR RTDE JACOBIAN", "matrix", J)
        if row_order == "linear_angular":
            return J
        if row_order == "angular_linear":
            return np.vstack([J[3:], J[:3]])
        raise ValueError(f"Unknown rtde_jacobian_row_order={row_order!r}.")

    def joints_within_safety_limits(self, q_rad: Iterable[float]) -> bool | None:
        try:
            q = finite_vector(q_rad, 6, "q_rad")
            debug_print(self.cfg, "SAFETY", "checking joint target [rad]", q)
            result = bool(self.control.isJointsWithinSafetyLimits(q.tolist()))
            debug_print(self.cfg, "SAFETY", f"controller result={result}")
            return result
        except Exception as exc:
            debug_print(
                self.cfg,
                "SAFETY",
                f"limit query exception: {type(exc).__name__}: {exc}",
            )
            return None

    def move_j(
        self,
        q_rad: Iterable[float],
        speed_rad_s: float,
        acceleration_rad_s2: float,
    ) -> bool:
        """Send one blocking joint command through the supplied robot class."""

        debug_print(self.cfg, "MOVEJ", "entering blocking URRTDERobot.move_j")
        accepted = bool(
            self.robot.move_j(
                finite_vector(q_rad, 6, "q_rad").tolist(),
                speed=float(speed_rad_s),
                acceleration=float(acceleration_rad_s2),
                asynchronous=False,
            )
        )
        debug_print(self.cfg, "MOVEJ", f"blocking move_j returned {accepted}")
        return accepted

    def receive_feedback(self) -> tuple[np.ndarray, np.ndarray]:
        """Receive actual joints and active-TCP pose through public methods."""

        feedback = (
            finite_vector(self.robot.get_joints(), 6, "received joints"),
            finite_vector(self.robot.get_tcp_pose(), 6, "received TCP pose"),
        )
        return feedback


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
    if cfg.debug_ik_iteration_interval < 1:
        raise ValueError("debug_ik_iteration_interval must be at least one.")
    if cfg.debug_rtde_call_interval < 1:
        raise ValueError("debug_rtde_call_interval must be at least one.")
    nominal_ur_dh(cfg.robot_model)
    T_TCP_M = magnet_offset_from_config(cfg)
    # Exercise the composition order with a nontrivial TCP pose.  This catches
    # accidental use of a base-frame scalar offset or reversed multiplication.
    test_T_R_TCP = pose6_to_T((0.3, -0.2, 0.5, 0.4, -0.3, 0.2))
    test_T_R_M = magnet_pose_from_tcp(test_T_R_TCP, T_TCP_M, "test T_R_M")
    reconstructed_T_R_TCP = tcp_pose_for_magnet_target(
        test_T_R_M,
        T_TCP_M,
        "reconstructed test T_R_TCP",
    )
    if not np.allclose(reconstructed_T_R_TCP, test_T_R_TCP, atol=1.0e-10):
        raise ValueError("TCP/magnet transform failed its configuration round-trip test.")
    finite_vector(cfg.joint_lower_bounds_rad, 6, "joint_lower_bounds_rad")
    finite_vector(cfg.joint_upper_bounds_rad, 6, "joint_upper_bounds_rad")
    finite_vector(cfg.ik_maximum_per_joint_step_rad, 6, "ik_maximum_per_joint_step_rad")
    command_joint_limits = finite_vector(
        cfg.maximum_commanded_per_joint_delta_rad,
        6,
        "maximum_commanded_per_joint_delta_rad",
    )
    if (
        cfg.rtde_frequency_hz <= 0.0
        or cfg.rtde_frequency_hz > 500.0
        or not np.isfinite(cfg.rtde_frequency_hz)
    ):
        raise ValueError("rtde_frequency_hz must be positive, finite, and at most 500.")
    if not cfg.live_confirmation_phrase:
        raise ValueError("live_confirmation_phrase must not be empty.")
    if cfg.execute_motion and not cfg.motion_confirmation_phrase:
        raise ValueError("motion_confirmation_phrase must not be empty.")
    if cfg.execute_motion and not cfg.use_live_robot:
        raise ValueError("execute_motion=True requires use_live_robot=True.")
    valid_motion_sources = {"own", "ur_rtde"}
    if not cfg.motion_solution_sources:
        raise ValueError("motion_solution_sources must not be empty.")
    if len(set(cfg.motion_solution_sources)) != len(cfg.motion_solution_sources):
        raise ValueError("motion_solution_sources must not contain duplicates.")
    unknown_sources = set(cfg.motion_solution_sources) - valid_motion_sources
    if unknown_sources:
        raise ValueError(
            f"Unknown motion solution sources: {sorted(unknown_sources)}; "
            f"choose from {sorted(valid_motion_sources)}."
        )
    if (
        not np.isfinite(cfg.move_joint_speed_rad_s)
        or cfg.move_joint_speed_rad_s <= 0.0
        or not np.isfinite(cfg.move_joint_acceleration_rad_s2)
        or cfg.move_joint_acceleration_rad_s2 <= 0.0
    ):
        raise ValueError("moveJ speed and acceleration must be positive and finite.")
    if (
        not np.isfinite(cfg.maximum_commanded_joint_delta_norm_rad)
        or cfg.maximum_commanded_joint_delta_norm_rad <= 0.0
        or np.any(command_joint_limits <= 0.0)
    ):
        raise ValueError("Commanded joint-delta limits must be positive and finite.")
    if cfg.feedback_sample_count < 1:
        raise ValueError("feedback_sample_count must be at least one.")
    if (
        not np.isfinite(cfg.feedback_sample_interval_s)
        or cfg.feedback_sample_interval_s < 0.0
    ):
        raise ValueError("feedback_sample_interval_s must be finite and non-negative.")
    if cfg.rtde_jacobian_row_order not in {"linear_angular", "angular_linear"}:
        raise ValueError("rtde_jacobian_row_order must be linear_angular or angular_linear.")
    if not cfg.targets:
        raise ValueError("At least one IK target is required.")
    target_names = [target.name for target in cfg.targets]
    if len(set(target_names)) != len(target_names):
        raise ValueError("Every IK target name must be unique.")
    unknown_target_names = set(cfg.motion_target_names) - set(target_names)
    if unknown_target_names:
        raise ValueError(
            f"Unknown motion_target_names: {sorted(unknown_target_names)}."
        )
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
    oracle: URRTDERobotAdapter | None,
) -> dict[str, Any]:
    debug_print(cfg, "JACOBIAN", f"{label}: starting")
    debug_print(cfg, "JACOBIAN", f"{label}: q [rad]", q_rad)
    own_analytic = geometric_jacobian(q_rad, dh, T_F_M)
    debug_print(cfg, "JACOBIAN", f"{label}: custom analytic Jacobian complete")
    if cfg.debug_print_matrices:
        debug_print(cfg, "JACOBIAN", f"{label}: custom analytic matrix", own_analytic)
    debug_print(cfg, "JACOBIAN", f"{label}: starting custom finite differences")
    own_fd = central_difference_jacobian(
        lambda q: forward_kinematics(q, dh, T_F_M).T_R_target,
        q_rad,
        cfg.finite_difference_joint_step_rad,
    )
    debug_print(cfg, "JACOBIAN", f"{label}: custom finite differences complete")
    matrices: dict[str, np.ndarray] = {
        "own_analytic": own_analytic,
        "own_finite_difference": own_fd,
    }
    warnings: list[str] = []
    if oracle is not None:
        try:
            debug_print(cfg, "JACOBIAN", f"{label}: requesting ur_rtde Jacobian")
            matrices["ur_rtde_analytic"] = oracle.jacobian(
                q_rad, T_F_M, cfg.rtde_jacobian_row_order
            )
        except Exception as exc:
            warning = f"ur_rtde getJacobian unavailable: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            debug_print(cfg, "JACOBIAN", f"{label}: {warning}")
        if cfg.calculate_rtde_fk_finite_difference_jacobian:
            try:
                debug_print(
                    cfg,
                    "JACOBIAN",
                    f"{label}: starting ur_rtde FK finite differences (12 FK calls)",
                )
                matrices["ur_rtde_fk_finite_difference"] = central_difference_jacobian(
                    lambda q: oracle.forward_kinematics(q, T_F_M),
                    q_rad,
                    cfg.finite_difference_joint_step_rad,
                )
                debug_print(
                    cfg,
                    "JACOBIAN",
                    f"{label}: ur_rtde FK finite differences complete",
                )
            except Exception as exc:
                warning = (
                    "ur_rtde FK finite difference unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                warnings.append(warning)
                debug_print(cfg, "JACOBIAN", f"{label}: {warning}")

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
    for comparison_name, metrics in comparisons.items():
        debug_print(
            cfg,
            "JACOBIAN",
            (
                f"{label}: {comparison_name} relative_error="
                f"{metrics['relative_frobenius_error']:.6e}"
            ),
        )
    debug_print(cfg, "JACOBIAN", f"{label}: complete; sources={list(matrices)}")
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
    oracle: URRTDERobotAdapter | None,
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


def execute_motion_comparison(
    oracle: URRTDERobotAdapter,
    target_details: list[dict[str, Any]],
    T_TCP_M: np.ndarray,
    cfg: ValidationConfig,
) -> list[dict[str, Any]]:
    """Send selected IK joint vectors and receive actual joints/TCP feedback."""

    selected_targets = [
        detail
        for detail in target_details
        if not cfg.motion_target_names
        or detail["target"].name in cfg.motion_target_names
    ]
    planned_count = len(selected_targets) * len(cfg.motion_solution_sources)
    print(
        "\nMOTION CONFIRMATION:\n"
        f"  Planned commands: {planned_count}\n"
        f"  Targets: {[detail['target'].name for detail in selected_targets]}\n"
        f"  Sources/order: {list(cfg.motion_solution_sources)}\n"
        f"  moveJ speed: {cfg.move_joint_speed_rad_s} rad/s\n"
        f"  moveJ acceleration: {cfg.move_joint_acceleration_rad_s2} rad/s^2\n"
        "  Each command is rechecked against controller safety limits and\n"
        "  configured joint-delta limits immediately before it is sent.\n"
    )
    confirmation = input(
        f"Type exactly {cfg.motion_confirmation_phrase!r} to execute motion: "
    )
    if confirmation != cfg.motion_confirmation_phrase:
        print("Motion skipped: confirmation did not match.")
        debug_print(cfg, "MOTION", "motion confirmation did not match; no command sent")
        return [
            {
                "status": "motion_confirmation_not_matched",
                "planned_command_count": planned_count,
            }
        ]

    per_joint_limit = finite_vector(
        cfg.maximum_commanded_per_joint_delta_rad,
        6,
        "maximum_commanded_per_joint_delta_rad",
    )
    records: list[dict[str, Any]] = []
    stop_remaining_commands = False
    for detail in selected_targets:
        target: IKTargetOffset = detail["target"]
        T_R_M_target = validate_transform(detail["T_R_M_target"], "T_R_M_target")
        for source in cfg.motion_solution_sources:
            debug_print(cfg, "MOTION", f"evaluating target={target.name}, source={source}")
            record: dict[str, Any] = {
                "target_name": target.name,
                "solution_source": source,
                "target_T_R_M_pose6": T_to_pose6(T_R_M_target),
            }
            if stop_remaining_commands:
                record["status"] = "skipped_after_previous_motion_failure"
                debug_print(cfg, "MOTION", record["status"])
                records.append(record)
                continue

            if source == "own":
                own_ik: IKResult = detail["own_ik"]
                q_command = own_ik.q_rad
                if not own_ik.converged:
                    record.update(
                        {
                            "status": "skipped_custom_ik_not_converged",
                            "ik_reason": own_ik.reason,
                        }
                    )
                    debug_print(
                        cfg,
                        "MOTION",
                        f"{target.name}/{source}: {record['status']} ({own_ik.reason})",
                    )
                    records.append(record)
                    continue
            else:
                q_command = detail["q_ur_rtde_rad"]
                if q_command is None:
                    record.update(
                        {
                            "status": "skipped_ur_rtde_ik_unavailable",
                            "ik_reason": detail["ur_rtde_status"],
                        }
                    )
                    debug_print(
                        cfg,
                        "MOTION",
                        f"{target.name}/{source}: {record['status']} ({detail['ur_rtde_status']})",
                    )
                    records.append(record)
                    continue

            q_command = finite_vector(q_command, 6, f"{source} q_command")
            debug_print(cfg, "MOTION", "candidate q_command [rad]", q_command)
            debug_print(cfg, "MOTION", "requesting pre-command feedback")
            q_before, tcp_before_pose6 = oracle.receive_feedback()
            debug_print(cfg, "MOTION", "pre-command q [rad]", q_before)
            debug_print(
                cfg,
                "MOTION",
                "pre-command TCP pose [m, rotvec rad]",
                tcp_before_pose6,
            )
            # moveJ receives absolute joint angles.  Apply motion bounds to the
            # raw numerical delta, not a modulo-2*pi comparison: a target one
            # full revolution away must never appear to be a zero-size move.
            command_delta = q_command - q_before
            command_delta_norm = float(np.linalg.norm(command_delta))
            controller_safety_limit = oracle.joints_within_safety_limits(q_command)
            debug_print(cfg, "MOTION", "raw command delta [rad]", command_delta)
            debug_print(
                cfg,
                "MOTION",
                (
                    f"delta_norm={command_delta_norm:.8f} rad, "
                    f"configured_norm_limit={cfg.maximum_commanded_joint_delta_norm_rad:.8f} rad, "
                    f"controller_safety={controller_safety_limit}"
                ),
            )
            record.update(
                {
                    "q_command_rad": q_command,
                    "q_before_rad": q_before,
                    "tcp_before_pose6": tcp_before_pose6,
                    "raw_command_delta_from_received_q_rad": command_delta,
                    "wrapped_command_delta_from_received_q_rad": (
                        wrapped_joint_difference(q_command, q_before)
                    ),
                    "command_delta_norm_rad": command_delta_norm,
                    "controller_joints_within_safety_limits": controller_safety_limit,
                }
            )
            if controller_safety_limit is not True:
                record["status"] = "skipped_controller_safety_limit_not_true"
                debug_print(cfg, "MOTION", f"{target.name}/{source}: {record['status']}")
                records.append(record)
                continue
            if (
                command_delta_norm > cfg.maximum_commanded_joint_delta_norm_rad
                or np.any(np.abs(command_delta) > per_joint_limit)
            ):
                record["status"] = "skipped_configured_joint_delta_limit"
                debug_print(
                    cfg,
                    "MOTION",
                    (
                        f"{target.name}/{source}: {record['status']}; "
                        f"per_joint_limit={per_joint_limit.tolist()}"
                    ),
                )
                records.append(record)
                continue

            print(f"\nSending {source} joints for {target.name}: {q_command.tolist()}")
            try:
                accepted = oracle.move_j(
                    q_command,
                    cfg.move_joint_speed_rad_s,
                    cfg.move_joint_acceleration_rad_s2,
                )
                record["move_j_returned"] = accepted
                if not accepted:
                    record["status"] = "move_j_returned_false"
                    debug_print(cfg, "MOTION", f"{target.name}/{source}: {record['status']}")
                    stop_remaining_commands = True
                    records.append(record)
                    continue

                feedback_samples: list[dict[str, Any]] = []
                feedback_start = time.monotonic()
                for sample_index in range(cfg.feedback_sample_count):
                    if sample_index:
                        time.sleep(cfg.feedback_sample_interval_s)
                    q_received, tcp_received_pose6 = oracle.receive_feedback()
                    feedback_samples.append(
                        {
                            "sample_index": sample_index,
                            "elapsed_s": time.monotonic() - feedback_start,
                            "q_received_rad": q_received,
                            "tcp_received_pose6": tcp_received_pose6,
                            "q_tracking_error_rad": wrapped_joint_difference(
                                q_received, q_command
                            ),
                        }
                    )
                    if cfg.debug_print_feedback_samples:
                        debug_print(
                            cfg,
                            "FEEDBACK",
                            (
                                f"sample={sample_index} elapsed="
                                f"{feedback_samples[-1]['elapsed_s']:.6f}s"
                            ),
                        )
                        debug_print(cfg, "FEEDBACK", "q_received [rad]", q_received)
                        debug_print(
                            cfg,
                            "FEEDBACK",
                            "tcp_received [m, rotvec rad]",
                            tcp_received_pose6,
                        )

                final_feedback = feedback_samples[-1]
                final_q = final_feedback["q_received_rad"]
                final_tcp_pose6 = final_feedback["tcp_received_pose6"]
                T_R_M_received = magnet_pose_from_tcp(
                    pose6_to_T(final_tcp_pose6),
                    T_TCP_M,
                    "received T_R_M",
                )
                record.update(
                    {
                        "status": "completed",
                        "feedback_samples": feedback_samples,
                        "final_q_received_rad": final_q,
                        "final_tcp_received_pose6": final_tcp_pose6,
                        "final_q_tracking_error_rad": wrapped_joint_difference(
                            final_q, q_command
                        ),
                        "final_q_tracking_error_norm_rad": float(
                            np.linalg.norm(
                                wrapped_joint_difference(final_q, q_command)
                            )
                        ),
                        "final_T_R_M_received_pose6": T_to_pose6(T_R_M_received),
                        **{
                            f"received_{key}": value
                            for key, value in pose_error_metrics(
                                T_R_M_received, T_R_M_target
                            ).items()
                        },
                    }
                )
                print(f"Received joints: {final_q.tolist()}")
                print(f"Received TCP pose: {final_tcp_pose6.tolist()}")
                print(
                    "Received magnet target error: "
                    f"{record['received_position_error_mm']:.6f} mm, "
                    f"{record['received_orientation_error_deg']:.6f} deg"
                )
                debug_print(cfg, "MOTION", f"{target.name}/{source}: completed")
            except Exception as exc:
                record.update(
                    {
                        "status": "motion_or_feedback_exception",
                        "exception": f"{type(exc).__name__}: {exc}",
                    }
                )
                stop_remaining_commands = True
                print(f"Motion stopped after exception: {record['exception']}")
            records.append(record)
    return records


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
    actual_tcp_position = snapshot.actual_T_R_TCP[:3, 3]
    actual_magnet_position = T_R_M_inferred[:3, 3]
    ax_3d.scatter(*actual_tcp_position, s=45, marker="o", label="actual TCP")
    ax_3d.scatter(*T_R_M_inferred[:3, 3], s=70, marker="*", label="robot-inferred actual M")
    ax_3d.plot(
        *np.vstack([actual_tcp_position, actual_magnet_position]).T,
        "k--",
        linewidth=1.5,
        label="rigid TCP→M offset",
    )
    all_points = [*links, actual_tcp_position, actual_magnet_position]
    for detail in target_details:
        magnet_target_position = np.asarray(detail["T_R_M_target"])[:3, 3]
        tcp_target_position = np.asarray(detail["T_R_TCP_target"])[:3, 3]
        all_points.extend([magnet_target_position, tcp_target_position])
        ax_3d.scatter(
            *magnet_target_position,
            s=30,
            marker="*",
            label=f"M target: {detail['target'].name}",
        )
        ax_3d.scatter(*tcp_target_position, s=18, marker="o", color="0.45")
        ax_3d.plot(
            *np.vstack([tcp_target_position, magnet_target_position]).T,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
        )
    ax_3d.set_title("TCP and magnet-centre targets in UR base frame R")
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
    debug_print(CONFIG, "START", "validator main() entered")
    debug_print(
        CONFIG,
        "CONFIG",
        (
            f"robot_ip={CONFIG.robot_ip}, robot_model={CONFIG.robot_model}, "
            f"live={CONFIG.use_live_robot}, execute_motion={CONFIG.execute_motion}, "
            f"targets={[target.name for target in CONFIG.targets]}"
        ),
    )
    debug_print(CONFIG, "CONFIG", "validating configuration")
    validate_config(CONFIG)
    debug_print(CONFIG, "CONFIG", "configuration valid")
    debug_print(CONFIG, "MODEL", "building corrected DH parameters")
    dh = corrected_dh_from_config(CONFIG)
    debug_print(CONFIG, "MODEL", "DH a [m]", dh.a_m)
    debug_print(CONFIG, "MODEL", "DH d [m]", dh.d_m)
    debug_print(CONFIG, "MODEL", "DH alpha [rad]", dh.alpha_rad)
    debug_print(CONFIG, "MODEL", "DH theta offsets [rad]", dh.theta_offset_rad)
    T_TCP_M = magnet_offset_from_config(CONFIG)
    debug_print(
        CONFIG,
        "FRAMES",
        "T_TCP_M pose [m, rotvec rad]",
        T_to_pose6(T_TCP_M),
    )
    output_dir = make_output_directory(CONFIG)
    debug_print(CONFIG, "OUTPUT", f"created output directory {output_dir.resolve()}")
    oracle: URRTDERobotAdapter | None = None
    try:
        if CONFIG.use_live_robot:
            debug_print(CONFIG, "CONNECT", "starting live URRTDERobot adapter")
            oracle = URRTDERobotAdapter(CONFIG)
            debug_print(CONFIG, "SNAPSHOT", "starting live snapshot")
            snapshot = oracle.snapshot()
        else:
            debug_print(CONFIG, "SNAPSHOT", "building offline snapshot")
            snapshot = offline_snapshot(CONFIG, dh)
            debug_print(CONFIG, "SNAPSHOT", "offline snapshot complete")

        debug_print(CONFIG, "SNAPSHOT", "snapshot q [rad]", snapshot.q_rad)
        debug_print(
            CONFIG,
            "SNAPSHOT",
            "snapshot actual TCP [m, rotvec rad]",
            T_to_pose6(snapshot.actual_T_R_TCP),
        )

        T_F_M = magnet_pose_from_tcp(snapshot.T_F_TCP, T_TCP_M, "T_F_M")
        T_R_M_inferred = magnet_pose_from_tcp(
            snapshot.actual_T_R_TCP,
            T_TCP_M,
            "robot-inferred T_R_M",
        )
        debug_print(CONFIG, "FRAMES", "T_F_M pose [m, rotvec rad]", T_to_pose6(T_F_M))
        debug_print(
            CONFIG,
            "FRAMES",
            "robot-inferred T_R_M pose [m, rotvec rad]",
            T_to_pose6(T_R_M_inferred),
        )
        tcp_to_magnet_lever_R_m = (
            T_R_M_inferred[:3, 3] - snapshot.actual_T_R_TCP[:3, 3]
        )
        reconstructed_lever_TCP_m = (
            snapshot.actual_T_R_TCP[:3, :3].T @ tcp_to_magnet_lever_R_m
        )
        relative_tcp_magnet_rotvec = Rot.from_matrix(
            snapshot.actual_T_R_TCP[:3, :3].T @ T_R_M_inferred[:3, :3]
        ).as_rotvec()
        debug_print(
            CONFIG,
            "MAGNET OFFSET",
            "configured lever arm in TCP [mm]",
            1.0e3 * T_TCP_M[:3, 3],
        )
        debug_print(
            CONFIG,
            "MAGNET OFFSET",
            "rotated lever arm in robot base [mm]",
            1.0e3 * tcp_to_magnet_lever_R_m,
        )
        debug_print(
            CONFIG,
            "MAGNET OFFSET",
            "base-frame lever transformed back into TCP [mm]",
            1.0e3 * reconstructed_lever_TCP_m,
        )
        debug_print(
            CONFIG,
            "MAGNET OFFSET",
            "magnet orientation relative to TCP [rotvec deg]",
            np.degrees(relative_tcp_magnet_rotvec),
        )
        debug_print(CONFIG, "MODEL", "evaluating custom FK at actual joints")
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
            "configured_tcp_to_magnet_translation_m": T_TCP_M[:3, 3],
            "tcp_to_magnet_lever_arm_in_robot_base_m": tcp_to_magnet_lever_R_m,
            "reconstructed_tcp_to_magnet_translation_m": reconstructed_lever_TCP_m,
            "tcp_to_magnet_relative_rotation_vector_rad": relative_tcp_magnet_rotvec,
            "T_R_F_reported_if_available": snapshot.actual_T_R_F_reported,
            "T_R_TCP_own_DH_at_actual_q": own_at_actual.T_R_target,
            "T_R_M_own_DH_at_actual_q": own_magnet_at_actual,
            "reported_tcp_vs_own_DH": pose_error_metrics(
                own_at_actual.T_R_target, snapshot.actual_T_R_TCP
            ),
        }
        debug_print(
            CONFIG,
            "MODEL",
            f"reported TCP vs custom DH FK={actual_state['reported_tcp_vs_own_DH']}",
        )
        debug_print(
            CONFIG,
            "SNAPSHOT",
            f"wrapped joint drift norm={joint_snapshot_drift:.8e} rad",
        )
        checks: list[dict[str, Any]] = []
        check_threshold(
            checks,
            "stationary snapshot joint drift norm",
            joint_snapshot_drift,
            CONFIG.maximum_snapshot_joint_drift_rad,
        )

        if CONFIG.independent_measured_T_robot_magnet_pose6 is not None:
            debug_print(CONFIG, "FRAMES", "loading independent magnet measurement")
            independent_magnet = pose6_to_T(
                CONFIG.independent_measured_T_robot_magnet_pose6
            )
            actual_state["T_R_M_independent_measurement"] = independent_magnet
            actual_state["robot_inferred_vs_independent_magnet_measurement"] = (
                pose_error_metrics(T_R_M_inferred, independent_magnet)
            )

        if oracle is not None:
            debug_print(CONFIG, "UR RTDE FK", "evaluating calibrated TCP FK")
            rtde_tcp_fk = oracle.forward_kinematics(snapshot.q_rad, snapshot.T_F_TCP)
            debug_print(CONFIG, "UR RTDE FK", "evaluating calibrated magnet FK")
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
            debug_print(
                CONFIG,
                "UR RTDE FK",
                f"reported TCP vs calibrated FK={rtde_actual_error}",
            )
            debug_print(
                CONFIG,
                "UR RTDE FK",
                (
                    "custom DH vs calibrated magnet FK="
                    f"{actual_state['own_DH_vs_ur_rtde_calibrated_magnet_FK']}"
                ),
            )
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
        debug_print(CONFIG, "JACOBIAN", "building actual-joint Jacobian record")
        jacobian_records = [
            build_jacobian_record(
                "actual_q", snapshot.q_rad, dh, T_F_M, CONFIG, oracle
            )
        ]

        for target in CONFIG.targets:
            debug_print(
                CONFIG,
                "TARGET",
                (
                    f"starting {target.name}: frame={target.frame}, "
                    f"translation_mm={target.translation_mm}, "
                    f"rotation_vector_deg={target.rotation_vector_deg}"
                ),
            )
            T_R_M_target = apply_target_offset(T_R_M_inferred, target)
            rotation_debug = rotation_increment_diagnostics(
                T_R_M_inferred,
                T_R_M_target,
                target,
            )
            debug_print(
                CONFIG,
                "ROTATION",
                f"{target.name}: requested rotation vector [deg]",
                rotation_debug["requested_rotation_vector_deg"],
            )
            debug_print(
                CONFIG,
                "ROTATION",
                (
                    f"{target.name}: achieved {target.frame} relative "
                    "rotation vector [deg]"
                ),
                rotation_debug["achieved_relative_rotation_vector_deg"],
            )
            debug_print(
                CONFIG,
                "ROTATION",
                (
                    f"{target.name}: relative angle="
                    f"{rotation_debug['achieved_relative_angle_deg']:.9f} deg, "
                    f"matrix-composition error="
                    f"{rotation_debug['rotation_matrix_composition_error_deg']:.3e} deg"
                ),
            )
            debug_print(
                CONFIG,
                "ROTATION",
                f"{target.name}: achieved relative rotation axis",
                rotation_debug["achieved_relative_axis"],
            )
            debug_print(
                CONFIG,
                "TARGET",
                f"{target.name} T_R_M pose [m, rotvec rad]",
                T_to_pose6(T_R_M_target),
            )
            # Every target is defined at the magnet centre.  Convert it to the
            # active-TCP target only after the magnet translation/orientation
            # has been applied, so rotations are about M rather than TCP.
            T_R_TCP_target = tcp_pose_for_magnet_target(
                T_R_M_target,
                T_TCP_M,
                "T_R_TCP_target",
            )
            debug_print(
                CONFIG,
                "TARGET",
                f"{target.name}: required TCP target pose [m, rotvec rad]",
                T_to_pose6(T_R_TCP_target),
            )
            target_lever_R_m = T_R_M_target[:3, 3] - T_R_TCP_target[:3, 3]
            debug_print(
                CONFIG,
                "MAGNET TARGET",
                f"{target.name}: TCP-to-magnet lever in robot base [mm]",
                1.0e3 * target_lever_R_m,
            )
            tcp_target_displacement_m = (
                T_R_TCP_target[:3, 3] - snapshot.actual_T_R_TCP[:3, 3]
            )
            debug_print(
                CONFIG,
                "MAGNET TARGET",
                (
                    f"{target.name}: required TCP-centre displacement for "
                    "this magnet-centre target [mm]"
                ),
                1.0e3 * tcp_target_displacement_m,
            )
            target_round_trip = magnet_pose_from_tcp(
                T_R_TCP_target,
                T_TCP_M,
                "target round-trip T_R_M",
            )
            debug_print(
                CONFIG,
                "MAGNET TARGET",
                (
                    f"{target.name}: TCP conversion round-trip error="
                    f"{pose_error_metrics(target_round_trip, T_R_M_target)}"
                ),
            )
            debug_print(CONFIG, "TARGET", f"{target.name}: starting custom IK")
            own_ik = inverse_kinematics_dls(
                T_R_M_target,
                snapshot.q_rad,
                dh,
                T_F_M,
                CONFIG,
            )
            debug_print(
                CONFIG,
                "TARGET",
                (
                    f"{target.name}: custom IK converged={own_ik.converged}, "
                    f"reason={own_ik.reason}, iterations={own_ik.iterations}"
                ),
            )
            q_rtde = None
            rtde_status = "offline_not_requested"
            if oracle is not None:
                # ur_rtde IK accepts the active-TCP pose calculated above; do
                # not call setTcp() or apply another scalar offset.
                q_rtde, rtde_status = oracle.inverse_kinematics(
                    T_R_TCP_target,
                    snapshot.q_rad,
                    CONFIG.ik_position_tolerance_m,
                    CONFIG.ik_orientation_tolerance_rad,
                )
                debug_print(
                    CONFIG,
                    "TARGET",
                    f"{target.name}: ur_rtde IK status={rtde_status}",
                )
                if q_rtde is not None:
                    debug_print(
                        CONFIG,
                        "TARGET",
                        f"{target.name}: custom-minus-ur_rtde wrapped q [rad]",
                        wrapped_joint_difference(own_ik.q_rad, q_rtde),
                    )
            debug_print(CONFIG, "TARGET", f"{target.name}: building comparison row")
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
            row["required_T_R_TCP_target_pose6"] = T_to_pose6(T_R_TCP_target)
            row["target_tcp_to_magnet_lever_R_m"] = target_lever_R_m
            row["required_tcp_centre_displacement_mm"] = (
                1.0e3 * tcp_target_displacement_m
            )
            row.update(rotation_debug)
            detail["T_R_TCP_target"] = T_R_TCP_target
            detail["target_tcp_to_magnet_lever_R_m"] = target_lever_R_m
            detail["required_tcp_centre_displacement_m"] = tcp_target_displacement_m
            detail["rotation_increment_diagnostics"] = rotation_debug
            if CONFIG.independent_measured_T_robot_magnet_pose6 is not None:
                row["independent_measured_T_R_M_pose6"] = (
                    CONFIG.independent_measured_T_robot_magnet_pose6
                )
            target_details.append(detail)
            if CONFIG.calculate_jacobians_at_ik_solutions:
                debug_print(
                    CONFIG,
                    "JACOBIAN",
                    f"{target.name}: validating Jacobian at custom IK solution",
                )
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
                    debug_print(
                        CONFIG,
                        "JACOBIAN",
                        f"{target.name}: validating Jacobian at ur_rtde IK solution",
                    )
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
            debug_print(CONFIG, "TARGET", f"{target.name}: complete")

        motion_records: list[dict[str, Any]] = []
        if CONFIG.execute_motion:
            debug_print(CONFIG, "MOTION", "numerical comparison complete; entering motion stage")
            if oracle is None:
                raise RuntimeError("Motion execution requires a live URRTDERobot.")
            # All targets and both IK solutions are computed from the initial
            # snapshot before any motion occurs.  Hardware execution happens
            # only after the numerical comparison is complete.
            motion_records = execute_motion_comparison(
                oracle,
                target_details,
                T_TCP_M,
                CONFIG,
            )
            debug_print(
                CONFIG,
                "MOTION",
                f"motion stage returned statuses={[record['status'] for record in motion_records]}",
            )
            for record in motion_records:
                label = (
                    f"{record.get('target_name', 'motion_plan')}:"
                    f"{record.get('solution_source', 'confirmation')}"
                )
                checks.append(
                    {
                        "name": f"motion execution {label}",
                        "status": record["status"],
                        "passed": record["status"] == "completed",
                    }
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
        for check in checks:
            debug_print(CONFIG, "CHECK", f"{check['name']}: passed={check['passed']}")

        debug_print(CONFIG, "OUTPUT", "flattening Jacobian data for CSV")
        jac_summary_rows, jac_element_rows = jacobian_csv_rows(jacobian_records)
        metadata = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "live_robot_connection_used": CONFIG.use_live_robot,
            "motion_commands_available_in_this_script": True,
            "motion_execution_enabled": CONFIG.execute_motion,
            "hardware_motion_performed_this_run": any(
                record.get("status") == "completed" for record in motion_records
            ),
            "completed_motion_command_count": sum(
                record.get("status") == "completed" for record in motion_records
            ),
            "magnet_pose_classification": (
                "independently measured"
                if CONFIG.independent_measured_T_robot_magnet_pose6 is not None
                else "robot-inferred from actual TCP and calibrated T_TCP_M"
            ),
            "transform_chain": "T_R_M = T_R_F(q) @ T_F_TCP @ T_TCP_M",
            "ur_rtde_ik_chain": "T_R_TCP_target = T_R_M_target @ inverse(T_TCP_M)",
            "magnet_position_equation": (
                "p_R_M = p_R_TCP + R_R_TCP @ p_TCP_M"
            ),
            "magnet_orientation_equation": (
                "R_R_M = R_R_TCP @ R_TCP_M"
            ),
            "configured_T_TCP_M_pose6": T_to_pose6(T_TCP_M),
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
            "motion_records": motion_records,
            "checks": checks,
            "all_checks_passed": bool(all(check["passed"] for check in checks)),
        }
        debug_print(CONFIG, "OUTPUT", "writing summary.json")
        save_json(output_dir / "summary.json", summary)
        debug_print(CONFIG, "OUTPUT", "writing ik_comparison.csv")
        write_csv(output_dir / "ik_comparison.csv", target_rows)
        debug_print(CONFIG, "OUTPUT", "writing jacobian_summary.csv")
        write_csv(output_dir / "jacobian_summary.csv", jac_summary_rows)
        debug_print(CONFIG, "OUTPUT", "writing jacobian_elements.csv")
        write_csv(output_dir / "jacobian_elements.csv", jac_element_rows)
        if motion_records:
            debug_print(CONFIG, "OUTPUT", "writing motion_feedback.csv")
            write_csv(output_dir / "motion_feedback.csv", motion_records)
        debug_print(CONFIG, "PLOT", f"plotting enabled={CONFIG.make_plots}")
        plot_results(
            output_dir,
            CONFIG,
            dh,
            snapshot,
            T_R_M_inferred,
            target_details,
            jacobian_records,
        )
        debug_print(CONFIG, "OUTPUT", "all configured output generation complete")

        print(f"\nValidation results: {output_dir.resolve()}")
        print(f"All configured checks passed: {summary['all_checks_passed']}")
        print(
            "Magnet pose used as actual: "
            f"{metadata['magnet_pose_classification']}"
        )
        if CONFIG.execute_motion:
            print(
                "Completed motion commands: "
                f"{metadata['completed_motion_command_count']}"
            )
        if CONFIG.strict_validation and not summary["all_checks_passed"]:
            failed = [check["name"] for check in checks if not check["passed"]]
            raise RuntimeError(f"Strict validation failed: {failed}")
    except KeyboardInterrupt:
        debug_print(CONFIG, "FATAL", "keyboard interrupt received; no recovery motion sent")
        raise
    except Exception as exc:
        debug_print(
            CONFIG,
            "FATAL",
            f"{type(exc).__name__}: {exc}; no recovery motion sent",
        )
        raise
    finally:
        if oracle is not None:
            oracle.close()


if __name__ == "__main__":
    main()
