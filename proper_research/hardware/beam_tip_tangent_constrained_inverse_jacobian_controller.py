"""Closed-loop beam tip-and-tangent controller in validated robot frames.

This file is intentionally new.  It does not modify the three validation and
legacy experiment files from which it reuses components.

The controller combines:

* camera-to-beam-to-robot measurement from
  ``robotics_frame_measurement_validation``;
* the full rigid offset ``T_R_M = T_R_TCP @ T_TCP_M``;
* the custom UR DH forward kinematics and magnet-centre geometric Jacobian from
  ``ur_magnet_ik_jacobian_validation``;
* the composite magnetic-beam forward model used by the frame validator;
* its implicit analytical tip-position/tip-tangent sensitivity; and
* a constrained proportional inverse-Jacobian step solved with SLSQP.

There are no command-line arguments.  Edit ``CONFIG`` below.  The shipped
default runs mathematical self-tests only.  ``run_mode='preview'`` connects to
the camera and robot and calculates one command without sending it.
``run_mode='live'`` can send motion, but only after two exact confirmations.

Transform notation is ``T_A_B``: it maps coordinates in B into A.  Rotations
are never added as rotation vectors.  They are composed as rotation matrices.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation as Rot


# =============================================================================
# USER CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TipTangentTarget:
    """One desired beam state.

    ``mode='absolute'`` interprets ``tip`` as a position and ``tangent`` as a
    unit direction.  ``mode='offset_from_initial'`` interprets ``tip`` as a
    translation and ``tangent`` as a rotation vector applied to the initial
    tangent.  Values are expressed in ``frame``, either ``'robot'`` or
    ``'beam'``.  All distances are metres and all angles are radians.
    """

    name: str
    mode: str
    frame: str
    tip: tuple[float, float, float]
    tangent: tuple[float, float, float]


@dataclass
class ControllerConfig:
    # Safe execution modes: self_test, preview, live.
    run_mode: str = "self_test"
    robot_ip: str = "192.168.56.101"
    rtde_frequency_hz: float = 125.0
    robot_model: str = "ur10e"
    live_connection_confirmation: str = "ready"
    live_motion_confirmation: str = "MOVE TIP CONTROLLER"

    # The magnet centre/body frame in the rotating active-TCP frame.  The
    # translation is LOCAL TCP +z, not robot-base z.
    T_tcp_magnet_pose6: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.044,
        0.0,
        0.0,
        0.0,
    )
    # None means inherit the validated T_R_B from
    # robotics_frame_measurement_validation.CONFIG.
    T_robot_beam_pose6: tuple[float, float, float, float, float, float] | None = None

    # Empty targets safely hold the first measured tip and tangent.  Example:
    # TipTangentTarget("small_test", "offset_from_initial", "beam",
    #                  (0.001, 0.0, 0.0), (0.0, 0.0, math.radians(2.0)))
    targets: tuple[TipTangentTarget, ...] = field(default_factory=tuple)

    # Vision/model configuration is inherited from the validated frame script.
    capture_new_image_each_iteration: bool = True
    use_measured_beam_length_for_model: bool = False
    maximum_capture_translation_drift_mm: float = 0.25
    maximum_capture_rotation_drift_deg: float = 0.10
    maximum_capture_joint_drift_rad: float = 0.003

    # Proportional task command.  Tangent errors are radians on S^2 and are
    # multiplied by a characteristic length before joining position errors.
    proportional_position_gain: float = 0.35
    proportional_tangent_gain: float = 0.25
    tangent_characteristic_length_m: float = 0.040
    position_row_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tangent_row_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Beam actuation Jacobian.  The default uses the implicit analytical
    # 6x7 derivative returned by
    # raw_model.jacobian_output_actuation_tangent().  The seven analytical
    # columns are [world translation, small world rotation, insertion].  Robot
    # motion uses only columns 0:6 because insertion is fixed.  The final three
    # analytical rows are Cartesian unit-tangent derivatives and are converted
    # to angular S^2 tangent coordinates before composition with the robot
    # Jacobian.  No dt factor is used: this controller solves for delta-q, not
    # joint velocity.
    beam_jacobian_source: str = "analytic"  # "analytic" or "finite_difference"
    analytic_jacobian_mode: str = "fast"  # "fast" or "accurate"
    analytic_eps_theta: float = 1.0e-6
    analytic_eps_hess: float = 1.0e-4
    analytic_reuse_cached: bool = True

    # Independent finite-difference checks.  Keep enabled during preview and
    # commissioning; they add forward-model solves and can be disabled after
    # the analytical convention and operating region have been validated.
    validate_analytic_beam_jacobian_with_fd: bool = True
    maximum_relative_beam_jacobian_disagreement: float = 0.35
    fail_on_beam_jacobian_disagreement: bool = False
    magnet_translation_fd_step_m: float = 2.5e-4
    magnet_rotation_fd_step_rad: float = 2.0e-3
    validate_chain_rule_with_joint_fd: bool = True
    joint_fd_validation_step_rad: float = 5.0e-4
    maximum_relative_jacobian_disagreement: float = 0.35
    fail_on_jacobian_disagreement: bool = False

    # Objective terms.
    damping: float = 0.05
    posture_weight: float = 0.02
    previous_step_weight: float = 0.05
    joint_limit_barrier_weight: float = 0.05
    joint_limit_influence_margin_rad: float = 0.35
    joint_limit_barrier_scale_rad: float = 0.08
    reference_joint_pose_rad: tuple[float, float, float, float, float, float] | None = None

    # IK constraints.  Replace generic bounds with the installed robot's
    # actual configured range before live use.
    joint_lower_bounds_rad: tuple[float, ...] = (-2.0 * math.pi,) * 6
    joint_upper_bounds_rad: tuple[float, ...] = (+2.0 * math.pi,) * 6
    hard_joint_limit_margin_rad: float = 0.05
    maximum_per_joint_step_rad: tuple[float, ...] = (0.030,) * 6
    maximum_step_norm_rad: float = 0.055
    maximum_predicted_tip_step_m: float = 0.0015
    maximum_predicted_tangent_step_rad: float = math.radians(4.0)
    # Optional robot-base bounds for the MAGNET CENTRE.  None disables that
    # side of the box.  Populate these from the validated experimental region.
    magnet_workspace_min_R_m: tuple[float, float, float] | None = None
    magnet_workspace_max_R_m: tuple[float, float, float] | None = None

    # Convergence, plausibility, and motion.
    position_tolerance_m: float = 5.0e-4
    tangent_tolerance_rad: float = math.radians(2.0)
    required_consecutive_converged_frames: int = 2
    maximum_position_error_m: float = 0.040
    maximum_tangent_error_rad: float = math.radians(60.0)
    maximum_iterations_per_target: int = 30
    settle_time_s: float = 0.75
    move_joint_speed_rad_s: float = 0.08
    move_joint_acceleration_rad_s2: float = 0.10

    # Solver and logging.
    solver_max_iterations: int = 200
    solver_ftol: float = 1.0e-11
    constraint_tolerance: float = 1.0e-8
    output_root: str = "beam_tip_tangent_controller"
    debug: bool = True


CONFIG = ControllerConfig()


# =============================================================================
# PURE GEOMETRY AND TASK ERROR
# =============================================================================


def finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values; got {array}.")
    return array


def unit(value: Any, name: str) -> np.ndarray:
    vector = finite_vector(value, 3, name)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"Cannot normalize zero-length {name}.")
    return vector / norm


def skew_matrix(value: Any) -> np.ndarray:
    """Return [v]_x such that [v]_x w = v cross w."""

    x, y, z = finite_vector(value, 3, "skew vector")
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def cartesian_tangent_jacobian_to_angular(
    tangent: Any,
    J_tangent_cartesian: Any,
) -> np.ndarray:
    """Convert d(unit tangent)/da to the minimum angular tangent derivative.

    For a unit direction t, a small Cartesian change satisfies
    ``delta_t = delta_psi cross t``.  Its minimum angular representation is
    therefore ``delta_psi = t cross delta_t``.  This conversion makes the
    analytical model derivative compatible with ``sphere_log(t, t_desired)``.
    """

    t = unit(tangent, "tip tangent for Jacobian conversion")
    J_cart = np.asarray(J_tangent_cartesian, dtype=float)
    if J_cart.ndim != 2 or J_cart.shape[0] != 3 or not np.all(np.isfinite(J_cart)):
        raise ValueError("Cartesian tangent Jacobian must have finite shape (3, N).")
    return skew_matrix(t) @ J_cart


def validate_transform(T: Any, name: str) -> np.ndarray:
    matrix = np.asarray(T, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{name} has an invalid homogeneous last row.")
    R = matrix[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1.0e-8) or not math.isclose(
        float(np.linalg.det(R)), 1.0, abs_tol=1.0e-8
    ):
        raise ValueError(f"{name} does not contain a proper rotation.")
    return matrix.copy()


def pose6_to_T(pose6: Any) -> np.ndarray:
    pose = finite_vector(pose6, 6, "pose6")
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(pose[3:]).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def inverse_T(T: Any) -> np.ndarray:
    matrix = validate_transform(T, "T")
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return result


def sphere_log(t_from: Any, t_to: Any) -> np.ndarray:
    """Shortest tangent-space rotation taking one unit direction to another.

    The returned three-vector is perpendicular to ``t_from`` and has magnitude
    equal to the angular separation in radians.  The antipodal case is rejected
    because infinitely many 180-degree rotation axes are valid.
    """

    a = unit(t_from, "sphere-log start tangent")
    b = unit(t_to, "sphere-log destination tangent")
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1.0e-12:
        if cosine > 0.0:
            return np.zeros(3)
        raise ValueError("Tangent directions are antipodal; the shortest axis is ambiguous.")
    angle = math.atan2(sine, cosine)
    return angle * cross / sine


def tangent_angle(t_a: Any, t_b: Any) -> float:
    return float(np.linalg.norm(sphere_log(t_a, t_b)))


def perturb_magnet_pose(
    T_R_M: Any,
    coordinate: int,
    signed_step: float,
) -> np.ndarray:
    """Perturb magnet position or orientation about its own centre.

    Coordinates 0:3 are robot-base translations.  Coordinates 3:6 are
    robot-base rotation axes, left-composed on the magnet orientation while
    leaving the magnet centre fixed.
    """

    T = validate_transform(T_R_M, "T_R_M")
    result = T.copy()
    axis = np.zeros(3)
    axis[coordinate % 3] = signed_step
    if coordinate < 3:
        result[:3, 3] += axis
    else:
        result[:3, :3] = Rot.from_rotvec(axis).as_matrix() @ T[:3, :3]
    return validate_transform(result, "perturbed T_R_M")


def block_rotate_twist(rotation: Any) -> np.ndarray:
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    result = np.zeros((6, 6))
    result[:3, :3] = R
    result[3:, 3:] = R
    return result


def effective_condition_number(matrix: Any, relative_threshold: float = 1.0e-8) -> tuple[int, float]:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0, float("inf")
    retained = singular[singular > relative_threshold * singular[0]]
    if retained.size == 0:
        return 0, float("inf")
    return int(retained.size), float(retained[0] / retained[-1])


# =============================================================================
# FORWARD-MODEL DIFFERENTIATION
# =============================================================================


@dataclass
class BeamOutput:
    tip_R_m: np.ndarray
    tangent_R: np.ndarray
    raw: np.ndarray


@dataclass
class TaskJacobianResult:
    J_tip_q: np.ndarray
    J_tangent_q: np.ndarray
    J_magnet_q: np.ndarray
    B_tip_magnet: np.ndarray
    B_tangent_magnet: np.ndarray
    registration_T: np.ndarray
    custom_nominal_T_R_M: np.ndarray
    beam_jacobian_source: str
    analytic_J_output_6x7: np.ndarray | None
    analytic_insertion_column: np.ndarray | None
    analytic_tangent_radial_leakage_norm: float | None
    beam_validation_relative_error: float | None
    beam_validation_B_tip_magnet: np.ndarray | None
    beam_validation_B_tangent_magnet: np.ndarray | None
    chain_validation_relative_error: float | None
    chain_validation_J_tip_q: np.ndarray | None
    chain_validation_J_tangent_q: np.ndarray | None


def strict_beam_output(value: Any) -> BeamOutput:
    """Require the true model tip tangent; never substitute a base-tip chord."""

    if hasattr(value, "tip"):
        tip = finite_vector(value.tip, 3, "model tip")
        tangent_value = getattr(value, "tip_tangent", None)
        if tangent_value is None:
            raise RuntimeError("Forward model has no tip_tangent; tangent control is unsafe.")
        tangent = unit(tangent_value, "model tip tangent")
        return BeamOutput(tip, tangent, np.r_[tip, tangent])
    raw = np.asarray(value, dtype=float).reshape(-1)
    if raw.size < 6 or not np.all(np.isfinite(raw[:6])):
        raise RuntimeError(
            "Forward model must return at least [tip_x,tip_y,tip_z,t_x,t_y,t_z]."
        )
    return BeamOutput(raw[:3].copy(), unit(raw[3:6], "model tip tangent"), raw.copy())


def evaluate_beam_at_transform(
    model: Any,
    frames: Any,
    T_R_M: Any,
    insertion_length_m: float,
    *,
    commit: bool,
) -> BeamOutput:
    transform = frames.FrameTransform("R", "M", validate_transform(T_R_M, "T_R_M"))
    p8 = frames.transform_to_p8(transform, insertion_length_m)
    if commit:
        _, output = model.commit_nominal(p8)
    else:
        output = model.forward_tip_fn(p8, commit=False)
    return strict_beam_output(output)


def finite_difference_beam_actuation_jacobian(
    model: Any,
    frames: Any,
    T_R_M: Any,
    insertion_length_m: float,
    cfg: ControllerConfig,
) -> tuple[BeamOutput, np.ndarray, np.ndarray]:
    """FD tip/angular-tangent w.r.t. world-frame magnet-centre motion."""

    T0 = validate_transform(T_R_M, "measured T_R_M")
    nominal = evaluate_beam_at_transform(model, frames, T0, insertion_length_m, commit=True)
    B_tip = np.empty((3, 6))
    B_tangent = np.empty((3, 6))
    for coordinate in range(6):
        step = (
            cfg.magnet_translation_fd_step_m
            if coordinate < 3
            else cfg.magnet_rotation_fd_step_rad
        )
        plus = evaluate_beam_at_transform(
            model,
            frames,
            perturb_magnet_pose(T0, coordinate, +step),
            insertion_length_m,
            commit=False,
        )
        minus = evaluate_beam_at_transform(
            model,
            frames,
            perturb_magnet_pose(T0, coordinate, -step),
            insertion_length_m,
            commit=False,
        )
        B_tip[:, coordinate] = (plus.tip_R_m - minus.tip_R_m) / (2.0 * step)
        # Map both perturbed tangents into the nominal tangent space before
        # differencing.  Direct component subtraction is only first-order valid
        # and obscures the unit-vector geometry.
        B_tangent[:, coordinate] = (
            sphere_log(nominal.tangent_R, plus.tangent_R)
            - sphere_log(nominal.tangent_R, minus.tangent_R)
        ) / (2.0 * step)
    # Some legacy model implementations mutate internal continuation state even
    # for commit=False.  Restore the nominal state explicitly.
    evaluate_beam_at_transform(model, frames, T0, insertion_length_m, commit=True)
    return nominal, B_tip, B_tangent


def analytic_beam_actuation_jacobian(
    model: Any,
    frames: Any,
    T_R_M: Any,
    insertion_length_m: float,
    cfg: ControllerConfig,
) -> tuple[BeamOutput, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the implicit analytical beam Jacobian in controller coordinates.

    The raw model derivative has shape (6, 7):

        d[tip_xyz, Cartesian unit tip tangent] /
        d[world source translation, small world source rotation, insertion].

    This joint controller holds insertion fixed and therefore selects columns
    0:6.  It converts the Cartesian unit-tangent rows to the minimum angular
    coordinates used by ``sphere_log``.  The resulting B matrices both have
    shape (3, 6) and accept the same world-frame magnet-centre spatial motion
    produced by the custom geometric robot Jacobian.
    """

    T0 = validate_transform(T_R_M, "measured T_R_M")
    transform = frames.FrameTransform("R", "M", T0)
    p8 = frames.transform_to_p8(transform, insertion_length_m)
    p7, raw_output = model.commit_nominal(p8)
    nominal = strict_beam_output(raw_output)

    raw_model = getattr(model, "raw_model", None)
    if raw_model is None:
        raise RuntimeError("Analytical beam Jacobian requires model.raw_model.")
    analytic_method = getattr(raw_model, "jacobian_output_actuation_tangent", None)
    if not callable(analytic_method):
        raise RuntimeError(
            "The forward model does not provide "
            "jacobian_output_actuation_tangent(p7, ...)."
        )

    # Refuse an implicit coordinate mismatch.  The robot geometric Jacobian is
    # [world linear; world angular] at the magnet centre, so the analytical
    # source-pose tangent must also be world/spatial.  The old MPC helper
    # integrate_pose8_body() uses body angular velocity and is intentionally
    # not part of this joint-increment controller.
    theta_builder = getattr(raw_model, "build_theta_model", None)
    if callable(theta_builder):
        theta_model = theta_builder(np.asarray(p7, dtype=float).reshape(7))
        convention = getattr(theta_model, "rotation_convention", None)
        if convention != "world":
            raise RuntimeError(
                "Analytical source rotation convention must be 'world' to "
                f"match the robot geometric Jacobian; got {convention!r}."
            )
        theta0 = np.asarray(getattr(theta_model, "theta0", np.empty(0)), dtype=float)
        if theta0.shape != (7,) or not np.allclose(theta0[3:6], 0.0, atol=1.0e-14):
            raise RuntimeError(
                "ThetaModel must represent source rotation as a zero-centred "
                "small world-frame tangent about q_src0."
            )

    J_output = np.asarray(
        analytic_method(
            np.asarray(p7, dtype=float).reshape(7),
            solve_if_needed=False,
            eps_theta=cfg.analytic_eps_theta,
            eps_hess=cfg.analytic_eps_hess,
            mode=cfg.analytic_jacobian_mode,
            reuse_cached=cfg.analytic_reuse_cached,
        ),
        dtype=float,
    ).reshape(6, 7)
    if not np.all(np.isfinite(J_output)):
        raise FloatingPointError("Analytical beam Jacobian contains non-finite values.")

    B_tip = J_output[:3, :6].copy()
    J_tangent_cartesian = J_output[3:6, :6].copy()
    B_tangent = cartesian_tangent_jacobian_to_angular(
        nominal.tangent_R,
        J_tangent_cartesian,
    )
    radial_leakage = float(
        np.linalg.norm(nominal.tangent_R.reshape(1, 3) @ J_tangent_cartesian)
    )
    return nominal, B_tip, B_tangent, J_output, radial_leakage


def select_beam_actuation_jacobian(
    model: Any,
    frames: Any,
    T_R_M: Any,
    insertion_length_m: float,
    cfg: ControllerConfig,
) -> tuple[
    BeamOutput,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    float | None,
    float | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Select analytical/FD B and optionally compare both in identical coordinates."""

    analytic_full = None
    radial_leakage = None
    beam_relative_error = None
    fd_tip = None
    fd_tangent = None

    if cfg.beam_jacobian_source == "analytic":
        nominal, B_tip, B_tangent, analytic_full, radial_leakage = (
            analytic_beam_actuation_jacobian(
                model,
                frames,
                T_R_M,
                insertion_length_m,
                cfg,
            )
        )
        if cfg.validate_analytic_beam_jacobian_with_fd:
            _, fd_tip, fd_tangent = finite_difference_beam_actuation_jacobian(
                model,
                frames,
                T_R_M,
                insertion_length_m,
                cfg,
            )
            characteristic = cfg.tangent_characteristic_length_m
            selected_scaled = np.vstack([B_tip, characteristic * B_tangent])
            fd_scaled = np.vstack([fd_tip, characteristic * fd_tangent])
            beam_relative_error = float(
                np.linalg.norm(selected_scaled - fd_scaled)
                / max(np.linalg.norm(fd_scaled), 1.0e-12)
            )
            if (
                cfg.fail_on_beam_jacobian_disagreement
                and beam_relative_error > cfg.maximum_relative_beam_jacobian_disagreement
            ):
                raise RuntimeError(
                    f"Analytical/FD beam Jacobian disagreement {beam_relative_error:.3f} "
                    f"exceeds {cfg.maximum_relative_beam_jacobian_disagreement:.3f}."
                )
            # FD evaluation restores the nominal model state, but explicitly
            # recommit so the analytical cache and subsequent solve state are
            # both tied to the exact measured operating point.
            evaluate_beam_at_transform(
                model, frames, T_R_M, insertion_length_m, commit=True
            )
        return (
            nominal,
            B_tip,
            B_tangent,
            analytic_full,
            radial_leakage,
            beam_relative_error,
            fd_tip,
            fd_tangent,
        )

    if cfg.beam_jacobian_source == "finite_difference":
        nominal, B_tip, B_tangent = finite_difference_beam_actuation_jacobian(
            model,
            frames,
            T_R_M,
            insertion_length_m,
            cfg,
        )
        return nominal, B_tip, B_tangent, None, None, None, None, None

    raise ValueError(
        "beam_jacobian_source must be 'analytic' or 'finite_difference'."
    )


def registered_custom_robot_jacobian(
    q_rad: Any,
    measured_T_R_M: Any,
    T_F_M: Any,
    dh: Any,
    urik: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locally register custom DH FK to the measured magnet pose.

    C = T_measured inv(T_custom(q)) makes the nominal pose exact.  A constant
    left registration rotates both linear and angular velocity blocks.  This
    avoids injecting the static FK pose bias into the beam-model operating
    point while retaining the custom differential kinematics.
    """

    q = finite_vector(q_rad, 6, "q")
    custom_T = validate_transform(
        urik.forward_kinematics(q, dh, T_F_M).T_R_target,
        "custom T_R_M",
    )
    measured = validate_transform(measured_T_R_M, "measured T_R_M")
    C = measured @ inverse_T(custom_T)
    J_custom = np.asarray(urik.geometric_jacobian(q, dh, T_F_M), dtype=float).reshape(6, 6)
    J_registered = block_rotate_twist(C[:3, :3]) @ J_custom
    return J_registered, validate_transform(C, "registration C"), custom_T


def finite_difference_registered_joint_jacobian(
    *,
    q_rad: Any,
    registration_T: Any,
    T_F_M: Any,
    dh: Any,
    urik: Any,
    model: Any,
    frames: Any,
    insertion_length_m: float,
    nominal_tangent_R: Any,
    step_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = finite_vector(q_rad, 6, "q")
    C = validate_transform(registration_T, "registration_T")
    t0 = unit(nominal_tangent_R, "nominal tangent")
    J_tip = np.empty((3, 6))
    J_tangent = np.empty((3, 6))
    for joint in range(6):
        dq = np.zeros(6)
        dq[joint] = step_rad
        T_plus = C @ urik.forward_kinematics(q + dq, dh, T_F_M).T_R_target
        T_minus = C @ urik.forward_kinematics(q - dq, dh, T_F_M).T_R_target
        plus = evaluate_beam_at_transform(model, frames, T_plus, insertion_length_m, commit=False)
        minus = evaluate_beam_at_transform(model, frames, T_minus, insertion_length_m, commit=False)
        J_tip[:, joint] = (plus.tip_R_m - minus.tip_R_m) / (2.0 * step_rad)
        J_tangent[:, joint] = (
            sphere_log(t0, plus.tangent_R) - sphere_log(t0, minus.tangent_R)
        ) / (2.0 * step_rad)
    return J_tip, J_tangent


def build_task_jacobian(
    *,
    q_rad: Any,
    measured_T_R_M: Any,
    T_F_M: Any,
    dh: Any,
    urik: Any,
    model: Any,
    frames: Any,
    insertion_length_m: float,
    cfg: ControllerConfig,
) -> tuple[BeamOutput, TaskJacobianResult]:
    (
        nominal,
        B_tip,
        B_tangent,
        analytic_full,
        radial_leakage,
        beam_relative_error,
        beam_fd_tip,
        beam_fd_tangent,
    ) = select_beam_actuation_jacobian(
        model, frames, measured_T_R_M, insertion_length_m, cfg
    )
    J_magnet, C, custom_T = registered_custom_robot_jacobian(
        q_rad, measured_T_R_M, T_F_M, dh, urik
    )
    J_tip = B_tip @ J_magnet
    J_tangent = B_tangent @ J_magnet
    fd_tip = None
    fd_tangent = None
    relative = None
    if cfg.validate_chain_rule_with_joint_fd:
        fd_tip, fd_tangent = finite_difference_registered_joint_jacobian(
            q_rad=q_rad,
            registration_T=C,
            T_F_M=T_F_M,
            dh=dh,
            urik=urik,
            model=model,
            frames=frames,
            insertion_length_m=insertion_length_m,
            nominal_tangent_R=nominal.tangent_R,
            step_rad=cfg.joint_fd_validation_step_rad,
        )
        chain = np.vstack([J_tip, cfg.tangent_characteristic_length_m * J_tangent])
        direct = np.vstack([fd_tip, cfg.tangent_characteristic_length_m * fd_tangent])
        relative = float(np.linalg.norm(chain - direct) / max(np.linalg.norm(direct), 1.0e-12))
        if cfg.fail_on_jacobian_disagreement and relative > cfg.maximum_relative_jacobian_disagreement:
            raise RuntimeError(
                f"Chain-rule Jacobian disagreement {relative:.3f} exceeds "
                f"{cfg.maximum_relative_jacobian_disagreement:.3f}."
            )
    evaluate_beam_at_transform(model, frames, measured_T_R_M, insertion_length_m, commit=True)
    return nominal, TaskJacobianResult(
        J_tip_q=J_tip,
        J_tangent_q=J_tangent,
        J_magnet_q=J_magnet,
        B_tip_magnet=B_tip,
        B_tangent_magnet=B_tangent,
        registration_T=C,
        custom_nominal_T_R_M=custom_T,
        beam_jacobian_source=cfg.beam_jacobian_source,
        analytic_J_output_6x7=analytic_full,
        analytic_insertion_column=(
            None if analytic_full is None else analytic_full[:, 6].copy()
        ),
        analytic_tangent_radial_leakage_norm=radial_leakage,
        beam_validation_relative_error=beam_relative_error,
        beam_validation_B_tip_magnet=beam_fd_tip,
        beam_validation_B_tangent_magnet=beam_fd_tangent,
        chain_validation_relative_error=relative,
        chain_validation_J_tip_q=fd_tip,
        chain_validation_J_tangent_q=fd_tangent,
    )


# =============================================================================
# CONSTRAINED PROPORTIONAL INVERSE-JACOBIAN STEP
# =============================================================================


@dataclass
class StepProblem:
    q_rad: np.ndarray
    q_reference_rad: np.ndarray
    previous_delta_q_rad: np.ndarray
    J_tip_q: np.ndarray
    J_tangent_q: np.ndarray
    J_magnet_q: np.ndarray
    magnet_position_R_m: np.ndarray
    position_error_R_m: np.ndarray
    tangent_error_R_rad: np.ndarray


@dataclass
class StepSolution:
    success: bool
    message: str
    delta_q_rad: np.ndarray
    q_command_rad: np.ndarray
    objective: float
    objective_terms: dict[str, float]
    predicted_position_step_R_m: np.ndarray
    predicted_tangent_step_R_rad: np.ndarray
    predicted_magnet_position_R_m: np.ndarray
    constraint_margins: dict[str, float]
    iterations: int


def softplus_and_sigmoid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Stable softplus and logistic sigmoid.
    softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    sigmoid[~positive] = exp_z / (1.0 + exp_z)
    return softplus, sigmoid


def solve_constrained_step(problem: StepProblem, cfg: ControllerConfig) -> StepSolution:
    q = finite_vector(problem.q_rad, 6, "q")
    q_ref = finite_vector(problem.q_reference_rad, 6, "q reference")
    dq_previous = finite_vector(problem.previous_delta_q_rad, 6, "previous delta q")
    Jp = np.asarray(problem.J_tip_q, dtype=float).reshape(3, 6)
    Jt = np.asarray(problem.J_tangent_q, dtype=float).reshape(3, 6)
    Jm = np.asarray(problem.J_magnet_q, dtype=float).reshape(6, 6)
    e_p = finite_vector(problem.position_error_R_m, 3, "position error")
    e_t = finite_vector(problem.tangent_error_R_rad, 3, "tangent error")
    p_m = finite_vector(problem.magnet_position_R_m, 3, "magnet position")

    characteristic = float(cfg.tangent_characteristic_length_m)
    J = np.vstack([Jp, characteristic * Jt])
    request = np.r_[
        cfg.proportional_position_gain * e_p,
        characteristic * cfg.proportional_tangent_gain * e_t,
    ]
    weights = np.r_[cfg.position_row_weights, cfg.tangent_row_weights].astype(float)

    q_lower = finite_vector(cfg.joint_lower_bounds_rad, 6, "joint lower bounds")
    q_upper = finite_vector(cfg.joint_upper_bounds_rad, 6, "joint upper bounds")
    step_limit = finite_vector(cfg.maximum_per_joint_step_rad, 6, "per-joint step")
    hard_lower_q = q_lower + cfg.hard_joint_limit_margin_rad
    hard_upper_q = q_upper - cfg.hard_joint_limit_margin_rad
    lower = np.maximum(-step_limit, hard_lower_q - q)
    upper = np.minimum(+step_limit, hard_upper_q - q)
    if np.any(lower > upper):
        raise RuntimeError("Current joints lie outside configured hard feasible bounds.")

    joint_range = np.maximum(q_upper - q_lower, 1.0e-6)
    barrier_scale = float(cfg.joint_limit_barrier_scale_rad)
    influence = float(cfg.joint_limit_influence_margin_rad)

    def terms_and_gradient(dq: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        residual = weights * (J @ dq - request)
        tracking = 0.5 * float(residual @ residual)
        tracking_grad = J.T @ (weights * residual)

        damping = 0.5 * cfg.damping**2 * float(dq @ dq)
        damping_grad = cfg.damping**2 * dq

        posture_residual = (q + dq - q_ref) / joint_range
        posture = 0.5 * cfg.posture_weight * float(posture_residual @ posture_residual)
        posture_grad = cfg.posture_weight * posture_residual / joint_range

        smooth_residual = dq - dq_previous
        smooth = 0.5 * cfg.previous_step_weight * float(smooth_residual @ smooth_residual)
        smooth_grad = cfg.previous_step_weight * smooth_residual

        q_next = q + dq
        z_low = (influence - (q_next - q_lower)) / barrier_scale
        z_high = (influence - (q_upper - q_next)) / barrier_scale
        sp_low, sig_low = softplus_and_sigmoid(z_low)
        sp_high, sig_high = softplus_and_sigmoid(z_high)
        barrier = 0.5 * cfg.joint_limit_barrier_weight * barrier_scale**2 * float(
            sp_low @ sp_low + sp_high @ sp_high
        )
        barrier_grad = cfg.joint_limit_barrier_weight * barrier_scale * (
            -sp_low * sig_low + sp_high * sig_high
        )

        total_gradient = (
            tracking_grad + damping_grad + posture_grad + smooth_grad + barrier_grad
        )
        return {
            "tracking": tracking,
            "damping": damping,
            "posture": posture,
            "previous_step": smooth,
            "joint_limit_barrier": barrier,
        }, total_gradient

    def objective(dq: np.ndarray) -> float:
        terms, _ = terms_and_gradient(dq)
        return float(sum(terms.values()))

    def gradient(dq: np.ndarray) -> np.ndarray:
        _, grad = terms_and_gradient(dq)
        return grad

    constraints: list[Any] = [
        {
            "type": "ineq",
            "fun": lambda dq: cfg.maximum_step_norm_rad**2 - float(dq @ dq),
            "jac": lambda dq: -2.0 * dq,
        },
        {
            "type": "ineq",
            "fun": lambda dq: cfg.maximum_predicted_tip_step_m**2
            - float((Jp @ dq) @ (Jp @ dq)),
            "jac": lambda dq: -2.0 * Jp.T @ (Jp @ dq),
        },
        {
            "type": "ineq",
            "fun": lambda dq: cfg.maximum_predicted_tangent_step_rad**2
            - float((Jt @ dq) @ (Jt @ dq)),
            "jac": lambda dq: -2.0 * Jt.T @ (Jt @ dq),
        },
    ]
    if cfg.magnet_workspace_min_R_m is not None or cfg.magnet_workspace_max_R_m is not None:
        workspace_lower = (
            np.full(3, -np.inf)
            if cfg.magnet_workspace_min_R_m is None
            else finite_vector(cfg.magnet_workspace_min_R_m, 3, "workspace minimum")
        )
        workspace_upper = (
            np.full(3, +np.inf)
            if cfg.magnet_workspace_max_R_m is None
            else finite_vector(cfg.magnet_workspace_max_R_m, 3, "workspace maximum")
        )
        constraints.append(
            LinearConstraint(Jm[:3], workspace_lower - p_m, workspace_upper - p_m)
        )

    result = minimize(
        objective,
        np.zeros(6),
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"maxiter": cfg.solver_max_iterations, "ftol": cfg.solver_ftol, "disp": False},
    )
    dq = np.asarray(result.x, dtype=float).reshape(6)
    terms, _ = terms_and_gradient(dq)
    predicted_p = Jp @ dq
    predicted_t = Jt @ dq
    predicted_magnet = p_m + Jm[:3] @ dq
    margins = {
        "step_norm_rad": cfg.maximum_step_norm_rad - float(np.linalg.norm(dq)),
        "tip_step_m": cfg.maximum_predicted_tip_step_m - float(np.linalg.norm(predicted_p)),
        "tangent_step_rad": cfg.maximum_predicted_tangent_step_rad
        - float(np.linalg.norm(predicted_t)),
        "joint_lower_rad": float(np.min(q + dq - hard_lower_q)),
        "joint_upper_rad": float(np.min(hard_upper_q - (q + dq))),
    }
    if cfg.magnet_workspace_min_R_m is not None:
        margins["workspace_lower_m"] = float(
            np.min(predicted_magnet - np.asarray(cfg.magnet_workspace_min_R_m))
        )
    if cfg.magnet_workspace_max_R_m is not None:
        margins["workspace_upper_m"] = float(
            np.min(np.asarray(cfg.magnet_workspace_max_R_m) - predicted_magnet)
        )
    feasible = bool(all(value >= -cfg.constraint_tolerance for value in margins.values()))
    success = bool(result.success and feasible and np.all(np.isfinite(dq)))
    return StepSolution(
        success=success,
        message=str(result.message) if result.success else f"SLSQP failed: {result.message}",
        delta_q_rad=dq,
        q_command_rad=q + dq,
        objective=float(result.fun),
        objective_terms=terms,
        predicted_position_step_R_m=predicted_p,
        predicted_tangent_step_R_rad=predicted_t,
        predicted_magnet_position_R_m=predicted_magnet,
        constraint_margins=margins,
        iterations=int(getattr(result, "nit", -1)),
    )


# =============================================================================
# TARGETS, CAMERA MEASUREMENT, LOGGING, AND LIVE LOOP
# =============================================================================


@dataclass
class Measurement:
    q_rad: np.ndarray
    T_R_TCP: np.ndarray
    T_R_M: np.ndarray
    tip_R_m: np.ndarray
    tangent_R: np.ndarray
    length_m: float
    translation_drift_mm: float
    rotation_drift_deg: float
    joint_drift_rad: float
    timestamp_utc: str


def target_in_robot_frame(
    target: TipTangentTarget,
    initial: Measurement,
    T_R_B: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if target.frame not in {"robot", "beam"}:
        raise ValueError(f"Target {target.name}: frame must be 'robot' or 'beam'.")
    if target.mode not in {"absolute", "offset_from_initial"}:
        raise ValueError(f"Target {target.name}: invalid mode {target.mode!r}.")
    vector = finite_vector(target.tip, 3, f"{target.name} tip")
    tangent_data = finite_vector(target.tangent, 3, f"{target.name} tangent")
    R_R_B = T_R_B.rotation
    if target.mode == "absolute":
        if target.frame == "robot":
            return vector, unit(tangent_data, f"{target.name} tangent")
        return T_R_B.apply_points(vector), unit(
            T_R_B.apply_directions(tangent_data), f"{target.name} tangent"
        )

    # Offset targets are relative to the initial measured state.  Rotation
    # vectors are composed through Rotation, never added to a pose rotvec.
    if target.frame == "robot":
        delta_position_R = vector
        delta_rotation_R = Rot.from_rotvec(tangent_data)
    else:
        delta_position_R = R_R_B @ vector
        delta_rotation_R = Rot.from_matrix(R_R_B) * Rot.from_rotvec(tangent_data) * Rot.from_matrix(R_R_B).inv()
    return initial.tip_R_m + delta_position_R, unit(
        delta_rotation_R.apply(initial.tangent_R), f"{target.name} tangent"
    )


def acquire_measurement(
    *,
    robot: Any,
    base: Any,
    frames: Any,
    frame_cfg: Any,
    experiment_cfg: Any,
    calibration: Any,
    T_R_B: Any,
    roi_polygon: Any,
    cfg: ControllerConfig,
) -> Measurement:
    import cv2

    q_before, tcp_before = robot.receive_feedback()
    if cfg.capture_new_image_each_iteration:
        base.new_capture(filename=experiment_cfg.image_filename)
    timestamp = datetime.now(timezone.utc).isoformat()
    q_after, tcp_after = robot.receive_feedback()
    joint_delta = np.asarray(q_after) - np.asarray(q_before)
    joint_delta = (joint_delta + math.pi) % (2.0 * math.pi) - math.pi
    q_mid = np.asarray(q_before) + 0.5 * joint_delta
    tcp_mid = frames.midpoint_pose6(tcp_before, tcp_after)
    translation_drift_mm, rotation_drift_deg = frames.pose_drift(tcp_before, tcp_after)
    joint_drift = float(np.linalg.norm(joint_delta))
    if translation_drift_mm > cfg.maximum_capture_translation_drift_mm:
        raise RuntimeError(f"Robot translated {translation_drift_mm:.3f} mm during image capture.")
    if rotation_drift_deg > cfg.maximum_capture_rotation_drift_deg:
        raise RuntimeError(f"Robot rotated {rotation_drift_deg:.4f} deg during image capture.")
    if joint_drift > cfg.maximum_capture_joint_drift_rad:
        raise RuntimeError(f"Joint drift {joint_drift:.6f} rad exceeded capture limit.")
    image = cv2.imread(experiment_cfg.image_filename)
    if image is None:
        raise FileNotFoundError(experiment_cfg.image_filename)
    measured = frames.measure_beam_from_image(
        image_bgr=image,
        roi_polygon=roi_polygon,
        pivot_hint_px=experiment_cfg.pivot_hint_px,
        calibration=calibration,
        T_R_B=T_R_B,
        cfg=frame_cfg,
        bounds_beam=base.bounds_beam,
    )
    _, _, T_R_M = frames.source_transform_from_tcp(tcp_mid, frame_cfg)
    return Measurement(
        q_rad=finite_vector(q_mid, 6, "captured q"),
        T_R_TCP=pose6_to_T(tcp_mid),
        T_R_M=T_R_M.matrix,
        tip_R_m=finite_vector(measured["tip_R_m"], 3, "measured tip"),
        tangent_R=unit(measured["tangent_R"], "measured tangent"),
        length_m=float(measured["length_m"]),
        translation_drift_mm=float(translation_drift_mm),
        rotation_drift_deg=float(rotation_drift_deg),
        joint_drift_rad=joint_drift,
        timestamp_utc=timestamp,
    )


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}.")


def make_output_directory(cfg: ControllerConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"tip_tangent_control_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=json_default, sort_keys=True) + "\n")


def validate_configuration(cfg: ControllerConfig) -> None:
    if cfg.run_mode not in {"self_test", "preview", "live"}:
        raise ValueError("run_mode must be self_test, preview, or live.")
    if cfg.tangent_characteristic_length_m <= 0.0:
        raise ValueError("tangent_characteristic_length_m must be positive.")
    if cfg.beam_jacobian_source not in {"analytic", "finite_difference"}:
        raise ValueError(
            "beam_jacobian_source must be 'analytic' or 'finite_difference'."
        )
    if cfg.analytic_jacobian_mode not in {"fast", "accurate"}:
        raise ValueError("analytic_jacobian_mode must be 'fast' or 'accurate'.")
    if cfg.analytic_eps_theta <= 0.0 or cfg.analytic_eps_hess <= 0.0:
        raise ValueError("Analytical sensitivity epsilon values must be positive.")
    if cfg.maximum_relative_beam_jacobian_disagreement < 0.0:
        raise ValueError("Beam Jacobian disagreement threshold cannot be negative.")
    if cfg.magnet_translation_fd_step_m <= 0.0 or cfg.magnet_rotation_fd_step_rad <= 0.0:
        raise ValueError("Finite-difference steps must be positive.")
    if cfg.maximum_step_norm_rad <= 0.0:
        raise ValueError("maximum_step_norm_rad must be positive.")
    if cfg.maximum_iterations_per_target < 1:
        raise ValueError("maximum_iterations_per_target must be at least one.")
    if cfg.required_consecutive_converged_frames < 1:
        raise ValueError("required_consecutive_converged_frames must be at least one.")
    if cfg.proportional_position_gain < 0.0 or cfg.proportional_tangent_gain < 0.0:
        raise ValueError("Proportional gains cannot be negative.")
    if np.any(np.asarray(cfg.position_row_weights) <= 0.0) or np.any(
        np.asarray(cfg.tangent_row_weights) <= 0.0
    ):
        raise ValueError("All task row weights must be positive.")
    lower = finite_vector(cfg.joint_lower_bounds_rad, 6, "joint lower bounds")
    upper = finite_vector(cfg.joint_upper_bounds_rad, 6, "joint upper bounds")
    if np.any(lower + 2.0 * cfg.hard_joint_limit_margin_rad >= upper):
        raise ValueError("Joint limits and hard margin leave no feasible interval.")
    if (
        cfg.magnet_workspace_min_R_m is not None
        and cfg.magnet_workspace_max_R_m is not None
        and np.any(
            finite_vector(cfg.magnet_workspace_min_R_m, 3, "workspace minimum")
            >= finite_vector(cfg.magnet_workspace_max_R_m, 3, "workspace maximum")
        )
    ):
        raise ValueError("Each workspace minimum must be below its maximum.")
    pose6_to_T(cfg.T_tcp_magnet_pose6)


def print_iteration(
    target_name: str,
    iteration: int,
    e_p: np.ndarray,
    e_t: np.ndarray,
    step: StepSolution,
    jacobian: TaskJacobianResult,
    cfg: ControllerConfig,
) -> None:
    scaled_J = np.vstack(
        [jacobian.J_tip_q, cfg.tangent_characteristic_length_m * jacobian.J_tangent_q]
    )
    rank, condition = effective_condition_number(scaled_J)
    validation = (
        "disabled"
        if jacobian.chain_validation_relative_error is None
        else f"{jacobian.chain_validation_relative_error:.3e}"
    )
    beam_validation = (
        "disabled"
        if jacobian.beam_validation_relative_error is None
        else f"{jacobian.beam_validation_relative_error:.3e}"
    )
    radial_leakage = (
        "n/a"
        if jacobian.analytic_tangent_radial_leakage_norm is None
        else f"{jacobian.analytic_tangent_radial_leakage_norm:.3e}"
    )
    print(
        f"[CONTROL] target={target_name} iteration={iteration} "
        f"|e_p|={1e3*np.linalg.norm(e_p):.3f} mm "
        f"|e_t|={np.degrees(np.linalg.norm(e_t)):.3f} deg "
        f"|dq|={np.linalg.norm(step.delta_q_rad):.5f} rad "
        f"rank={rank} cond_eff={condition:.3e} "
        f"B={jacobian.beam_jacobian_source} B_fd={beam_validation} "
        f"t_radial={radial_leakage} chain_fd={validation} "
        f"solver={step.success}"
    )


def verify_final_command(
    *,
    robot: Any,
    solution: StepSolution,
    measurement: Measurement,
    T_F_M: np.ndarray,
    cfg: ControllerConfig,
) -> dict[str, Any]:
    """Independent last check; not a substitute for optimizer constraints."""

    if not solution.success:
        raise RuntimeError(f"Refusing command because optimization failed: {solution.message}")
    if any(margin < -cfg.constraint_tolerance for margin in solution.constraint_margins.values()):
        raise RuntimeError(f"Refusing infeasible command: {solution.constraint_margins}")
    if not np.all(np.isfinite(solution.q_command_rad)):
        raise RuntimeError("Refusing non-finite joint command.")
    controller_safe = robot.joints_within_safety_limits(solution.q_command_rad)
    if controller_safe is not True:
        raise RuntimeError(f"UR controller did not affirm joint safety limits: {controller_safe}.")
    calibrated_T_R_M = validate_transform(
        robot.forward_kinematics(solution.q_command_rad, T_F_M),
        "controller FK magnet pose",
    )
    position = calibrated_T_R_M[:3, 3]
    if cfg.magnet_workspace_min_R_m is not None and np.any(
        position < np.asarray(cfg.magnet_workspace_min_R_m) - cfg.constraint_tolerance
    ):
        raise RuntimeError("Controller-FK magnet position violates workspace minimum.")
    if cfg.magnet_workspace_max_R_m is not None and np.any(
        position > np.asarray(cfg.magnet_workspace_max_R_m) + cfg.constraint_tolerance
    ):
        raise RuntimeError("Controller-FK magnet position violates workspace maximum.")
    return {
        "ur_controller_joint_safety": controller_safe,
        "controller_fk_T_R_M": calibrated_T_R_M,
        "controller_fk_magnet_position_R_m": position,
        "capture_drift_mm": measurement.translation_drift_mm,
        "capture_rotation_drift_deg": measurement.rotation_drift_deg,
    }


def run_hardware_controller(cfg: ControllerConfig) -> None:
    # Imports are intentionally lazy: the default self-test mode runs without
    # the research repository or hardware dependencies.
    import proper_research.hardware.robotics_frame_measurement_validation as frames
    import proper_research.hardware.ur_magnet_ik_jacobian_validation as urik

    frame_cfg = copy.deepcopy(frames.CONFIG)
    experiment_cfg = copy.deepcopy(frames.EXPERIMENT_CONFIG)
    experiment_cfg.send_commands = False
    frame_cfg.use_live_hardware = True
    frame_cfg.capture_new_image = cfg.capture_new_image_each_iteration
    frame_cfg.T_tcp_magnet_pose6 = cfg.T_tcp_magnet_pose6
    frame_cfg.assume_tcp_is_magnet_frame = False
    frame_cfg.use_measured_beam_length_for_model = cfg.use_measured_beam_length_for_model
    if cfg.T_robot_beam_pose6 is not None:
        frame_cfg.T_robot_beam_pose6 = cfg.T_robot_beam_pose6

    ur_cfg = copy.deepcopy(urik.CONFIG)
    ur_cfg.use_live_robot = True
    ur_cfg.execute_motion = cfg.run_mode == "live"
    ur_cfg.robot_ip = cfg.robot_ip
    ur_cfg.rtde_frequency_hz = cfg.rtde_frequency_hz
    ur_cfg.robot_model = cfg.robot_model
    ur_cfg.live_confirmation_phrase = cfg.live_connection_confirmation
    ur_cfg.motion_confirmation_phrase = cfg.live_motion_confirmation
    ur_cfg.T_tcp_magnet_pose6 = cfg.T_tcp_magnet_pose6
    ur_cfg.assume_tcp_is_magnet_frame = False
    ur_cfg.joint_lower_bounds_rad = cfg.joint_lower_bounds_rad
    ur_cfg.joint_upper_bounds_rad = cfg.joint_upper_bounds_rad

    frames.validate_configuration(frame_cfg)
    frames.experiment_v2.validate_config(experiment_cfg)
    base = frames.experiment_v2._base_module()
    base.configure_bounds_beam_paths(experiment_cfg)
    manual_frame = base.load_manual_vessel_boundaries_with_frame(experiment_cfg.manual_boundary_path)
    roi_polygon = base.load_polygon(experiment_cfg.roi_polygon_path)
    calibration = frames.load_planar_calibration(base, manual_frame, frame_cfg)
    T_R_B = frames.FrameTransform.from_pose6("R", "B", frame_cfg.T_robot_beam_pose6)
    lumen = frames.build_lumen_in_shared_frames(manual_frame, calibration, T_R_B)
    model, model_metadata = frames.build_forward_model_in_shared_frame(
        experiment_cfg=experiment_cfg,
        T_R_B=T_R_B,
        lumen=lumen,
    )

    output_dir = make_output_directory(cfg)
    with (output_dir / "configuration.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(cfg), stream, indent=2, default=json_default)
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(model_metadata, stream, indent=2, default=json_default)

    robot = urik.URRTDERobotAdapter(ur_cfg)
    motion_confirmed = False
    try:
        snapshot = robot.snapshot()
        T_TCP_M = pose6_to_T(cfg.T_tcp_magnet_pose6)
        T_F_M = validate_transform(snapshot.T_F_TCP @ T_TCP_M, "T_F_M")
        dh = urik.corrected_dh_from_config(ur_cfg)
        initial = acquire_measurement(
            robot=robot,
            base=base,
            frames=frames,
            frame_cfg=frame_cfg,
            experiment_cfg=experiment_cfg,
            calibration=calibration,
            T_R_B=T_R_B,
            roi_polygon=roi_polygon,
            cfg=cfg,
        )
        insertion_length_m = (
            float(
                np.clip(
                    initial.length_m,
                    experiment_cfg.min_vision_length_m,
                    experiment_cfg.max_vision_length_m,
                )
            )
            if cfg.use_measured_beam_length_for_model
            else float(experiment_cfg.initial_beam_length_m)
        )
        targets = cfg.targets or (
            TipTangentTarget(
                "hold_initial_measurement",
                "absolute",
                "robot",
                tuple(initial.tip_R_m),
                tuple(initial.tangent_R),
            ),
        )
        q_reference = (
            initial.q_rad.copy()
            if cfg.reference_joint_pose_rad is None
            else finite_vector(cfg.reference_joint_pose_rad, 6, "reference joint pose")
        )
        previous_delta = np.zeros(6)
        jsonl = output_dir / "controller_iterations.jsonl"

        for target in targets:
            desired_tip, desired_tangent = target_in_robot_frame(target, initial, T_R_B)
            converged_count = 0
            for iteration in range(cfg.maximum_iterations_per_target):
                measurement = initial if (target is targets[0] and iteration == 0) else acquire_measurement(
                    robot=robot,
                    base=base,
                    frames=frames,
                    frame_cfg=frame_cfg,
                    experiment_cfg=experiment_cfg,
                    calibration=calibration,
                    T_R_B=T_R_B,
                    roi_polygon=roi_polygon,
                    cfg=cfg,
                )
                e_p = desired_tip - measurement.tip_R_m
                e_t = sphere_log(measurement.tangent_R, desired_tangent)
                if np.linalg.norm(e_p) > cfg.maximum_position_error_m:
                    raise RuntimeError(f"Position error {np.linalg.norm(e_p):.4f} m exceeds plausibility limit.")
                if np.linalg.norm(e_t) > cfg.maximum_tangent_error_rad:
                    raise RuntimeError(
                        f"Tangent error {np.degrees(np.linalg.norm(e_t)):.2f} deg exceeds plausibility limit."
                    )
                within_tolerance = (
                    np.linalg.norm(e_p) <= cfg.position_tolerance_m
                    and np.linalg.norm(e_t) <= cfg.tangent_tolerance_rad
                )
                if within_tolerance:
                    converged_count += 1
                    append_jsonl(
                        jsonl,
                        {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "target": asdict(target),
                            "iteration": iteration,
                            "desired_tip_R_m": desired_tip,
                            "desired_tangent_R": desired_tangent,
                            "measurement": asdict(measurement),
                            "position_error_R_m": e_p,
                            "tangent_error_R_rad": e_t,
                            "within_tolerance": True,
                            "consecutive_converged_frames": converged_count,
                            "optimization_skipped": True,
                            "motion_sent": False,
                        },
                    )
                    print(
                        f"[CONTROL] target={target.name} iteration={iteration} "
                        f"inside tolerance ({converged_count}/"
                        f"{cfg.required_consecutive_converged_frames}); no motion"
                    )
                    if converged_count >= cfg.required_consecutive_converged_frames:
                        print(f"[CONTROL] target {target.name!r} converged.")
                        break
                    continue
                converged_count = 0
                nominal, jacobian = build_task_jacobian(
                    q_rad=measurement.q_rad,
                    measured_T_R_M=measurement.T_R_M,
                    T_F_M=T_F_M,
                    dh=dh,
                    urik=urik,
                    model=model,
                    frames=frames,
                    insertion_length_m=insertion_length_m,
                    cfg=cfg,
                )
                problem = StepProblem(
                    measurement.q_rad,
                    q_reference,
                    previous_delta,
                    jacobian.J_tip_q,
                    jacobian.J_tangent_q,
                    jacobian.J_magnet_q,
                    measurement.T_R_M[:3, 3],
                    e_p,
                    e_t,
                )
                solution = solve_constrained_step(problem, cfg)
                print_iteration(target.name, iteration, e_p, e_t, solution, jacobian, cfg)
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "target": asdict(target),
                    "iteration": iteration,
                    "desired_tip_R_m": desired_tip,
                    "desired_tangent_R": desired_tangent,
                    "measurement": asdict(measurement),
                    "position_error_R_m": e_p,
                    "tangent_error_R_rad": e_t,
                    "model_nominal_tip_R_m": nominal.tip_R_m,
                    "model_nominal_tangent_R": nominal.tangent_R,
                    "jacobian": asdict(jacobian),
                    "solution": asdict(solution),
                    "motion_sent": False,
                }
                append_jsonl(jsonl, record)
                if not solution.success:
                    raise RuntimeError(solution.message)

                if cfg.run_mode == "preview":
                    print("[PREVIEW] Command calculated but not sent. Preview stops after one iteration.")
                    return
                if cfg.run_mode != "live":
                    raise RuntimeError("Internal mode error: hardware loop requires preview or live.")
                if not motion_confirmed:
                    print(
                        "\nLIVE MOTION WARNING:\n"
                        "  The next step sends joint motion from a vision/model controller.\n"
                        "  Verify the target, frames, workspace, joint limits, speed, and clear cell.\n"
                    )
                    phrase = input(f"Type exactly {cfg.live_motion_confirmation!r} to enable motion: ")
                    if phrase != cfg.live_motion_confirmation:
                        raise RuntimeError("Live motion cancelled: confirmation did not match.")
                    motion_confirmed = True
                final_checks = verify_final_command(
                    robot=robot,
                    solution=solution,
                    measurement=measurement,
                    T_F_M=T_F_M,
                    cfg=cfg,
                )
                accepted = robot.move_j(
                    solution.q_command_rad,
                    cfg.move_joint_speed_rad_s,
                    cfg.move_joint_acceleration_rad_s2,
                )
                if not accepted:
                    raise RuntimeError("URRTDERobot.move_j did not accept the command.")
                append_jsonl(
                    jsonl,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "target_name": target.name,
                        "iteration": iteration,
                        "motion_sent": True,
                        "q_command_rad": solution.q_command_rad,
                        "final_checks": final_checks,
                    },
                )
                previous_delta = solution.delta_q_rad.copy()
                time.sleep(cfg.settle_time_s)
            else:
                raise RuntimeError(f"Target {target.name!r} did not converge within the iteration limit.")
    finally:
        robot.close()
    print(f"Controller output: {output_dir}")


# =============================================================================
# PURE SELF-TESTS
# =============================================================================


def run_self_tests() -> None:
    print("[SELF TEST] starting")
    np.testing.assert_allclose(
        sphere_log([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        [0.0, 0.0, math.pi / 2.0],
        atol=1.0e-12,
    )
    T = np.eye(4)
    T[:3, 3] = [0.2, -0.1, 0.4]
    rotated = perturb_magnet_pose(T, 4, math.pi / 2.0)
    np.testing.assert_allclose(rotated[:3, 3], T[:3, 3], atol=1.0e-14)
    np.testing.assert_allclose(
        rotated[:3, :3], Rot.from_rotvec([0.0, math.pi / 2.0, 0.0]).as_matrix(), atol=1.0e-12
    )
    tangent = np.array([1.0, 0.0, 0.0])
    # For a world angular perturbation psi, delta_t = psi cross t.
    J_tangent_cartesian = -skew_matrix(tangent)
    J_tangent_angular = cartesian_tangent_jacobian_to_angular(
        tangent,
        J_tangent_cartesian,
    )
    np.testing.assert_allclose(
        J_tangent_angular,
        np.diag([0.0, 1.0, 1.0]),
        atol=1.0e-12,
    )

    test_cfg = copy.deepcopy(CONFIG)
    test_cfg.run_mode = "self_test"
    test_cfg.maximum_predicted_tip_step_m = 0.01
    test_cfg.maximum_predicted_tangent_step_rad = 0.2
    Jp = np.zeros((3, 6))
    Jp[:, :3] = np.eye(3) * 0.1
    Jt = np.zeros((3, 6))
    Jt[:, 3:] = np.eye(3)
    Jm = np.eye(6)
    problem = StepProblem(
        q_rad=np.zeros(6),
        q_reference_rad=np.zeros(6),
        previous_delta_q_rad=np.zeros(6),
        J_tip_q=Jp,
        J_tangent_q=Jt,
        J_magnet_q=Jm,
        magnet_position_R_m=np.zeros(3),
        position_error_R_m=np.array([0.002, -0.001, 0.0005]),
        tangent_error_R_rad=np.array([0.02, -0.01, 0.0]),
    )
    solution = solve_constrained_step(problem, test_cfg)
    if not solution.success:
        raise AssertionError(solution.message)
    if not np.all(np.isfinite(solution.delta_q_rad)):
        raise AssertionError("Optimizer produced a non-finite step.")
    if np.linalg.norm(solution.delta_q_rad) > test_cfg.maximum_step_norm_rad + 1.0e-9:
        raise AssertionError("Trust-region constraint failed.")
    initial_tracking = np.linalg.norm(
        np.r_[
            test_cfg.proportional_position_gain * problem.position_error_R_m,
            test_cfg.tangent_characteristic_length_m
            * test_cfg.proportional_tangent_gain
            * problem.tangent_error_R_rad,
        ]
    )
    final_tracking = np.linalg.norm(
        np.r_[Jp @ solution.delta_q_rad, test_cfg.tangent_characteristic_length_m * Jt @ solution.delta_q_rad]
        - np.r_[
            test_cfg.proportional_position_gain * problem.position_error_R_m,
            test_cfg.tangent_characteristic_length_m
            * test_cfg.proportional_tangent_gain
            * problem.tangent_error_R_rad,
        ]
    )
    if final_tracking >= initial_tracking:
        raise AssertionError("Constrained inverse step did not reduce the linearized task residual.")

    near_limit_q = np.zeros(6)
    near_limit_q[0] = test_cfg.joint_upper_bounds_rad[0] - test_cfg.hard_joint_limit_margin_rad - 5.0e-4
    near_limit_problem = StepProblem(
        q_rad=near_limit_q,
        q_reference_rad=np.zeros(6),
        previous_delta_q_rad=np.zeros(6),
        J_tip_q=Jp,
        J_tangent_q=Jt,
        J_magnet_q=Jm,
        magnet_position_R_m=np.zeros(3),
        position_error_R_m=np.array([0.002, 0.0, 0.0]),
        tangent_error_R_rad=np.zeros(3),
    )
    near_limit_solution = solve_constrained_step(near_limit_problem, test_cfg)
    if not near_limit_solution.success:
        raise AssertionError(near_limit_solution.message)
    allowed_upper = (
        test_cfg.joint_upper_bounds_rad[0] - test_cfg.hard_joint_limit_margin_rad
    )
    if near_limit_solution.q_command_rad[0] > allowed_upper + 1.0e-9:
        raise AssertionError("Joint-position constraint failed near an upper limit.")
    print("[SELF TEST] sphere geometry: PASS")
    print("[SELF TEST] magnet-centre rotation: PASS")
    print("[SELF TEST] analytical Cartesian-to-angular tangent Jacobian: PASS")
    print("[SELF TEST] constrained optimizer: PASS")
    print("[SELF TEST] joint-limit objective and hard constraint: PASS")
    print("[SELF TEST] all tests passed; no hardware was imported or commanded")


def main() -> None:
    validate_configuration(CONFIG)
    if CONFIG.run_mode == "self_test":
        run_self_tests()
        return
    run_hardware_controller(CONFIG)


if __name__ == "__main__":
    main()
