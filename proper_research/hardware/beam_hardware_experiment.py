"""Incremental hardware experiments for a magnetically actuated beam.

This file intentionally has no command-line arguments.  Edit the CONFIG block
below, then run the file from the root of the Proper-Research repository.

Recommended order:

1. ``mode="arc_validation"`` and ``send_commands=False``: check imports,
   camera calibration, the planned arc and model predictions without motion.
2. Keep ``mode="arc_validation"``, set a very small arc, then set
   ``send_commands=True``.  The script records measured versus predicted beam
   tip position and tangent angle at every arc point.
3. ``mode="point_control"`` with ``send_commands=False``: inspect the first
   inverse-Jacobian step.
4. Only after the first three checks, set ``send_commands=True`` for slow
   closed-loop point control.

The live path uses the same ``LiveHardwareController.send_step`` convention as
``hardware_runner_optimized.py``: ``p_now`` is the commanded p8 state and
``u0`` is the corresponding seven-component control rate.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
)
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T

from proper_research.control.lab_ready_mpc import (
    build_initial_lumen_from_vision,
    build_measured_p8_from_pose6_and_length,
    rotate_body_xy,
    vision_result_to_x_meas_robot,
)
from proper_research.hardware.hardware_model_factory_optimized import (
    build_hardware_model_bundle_optimized,
)
from proper_research.parameters import default_beam_params, default_magnet_params
from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.simulation.magnetic_beam.contact import ContactParams
from proper_research.simulation.minimal_energy import rod_section_stiffness
from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
)
from proper_research.simulation.simulations.controller_factory_optimized import (
    build_controller_optimized,
)
from proper_research.simulation_controller.persistent_osqp import (
    PersistentOSQPSettings,
)
from proper_research.simulation_controller.sim_mpc_control import (
    make_Kbt_inv_profile,
)
from proper_research.vision import bounds_beam
from proper_research.vision.bounds_beam import load_polygon
from proper_research.vision.detect_blue import (
    load_manual_vessel_boundaries_with_frame,
)
from proper_research.vision.measure_length import new_capture


# =============================================================================
# USER CONFIGURATION -- edit values here; there are no terminal arguments
# =============================================================================


@dataclass
class ExperimentConfig:
    # Select "arc_validation" or "point_control".
    mode: str = "arc_validation"

    # SAFETY: False performs camera/model checks but never calls send_step().
    send_commands: bool = False
    live_confirmation_phrase: str = "MOVE MAGNET"

    # Output files are placed in output_root/run_name_<UTC timestamp>/.
    output_root: str = "beam_hardware_experiments"
    run_name: str = "beam_arc_then_inverse_jacobian"
    save_vision_overlays: bool = True

    # Robot and source-magnet hardware.
    robot_ip: str = "192.168.56.101"
    z_offset_m: float = 0.27
    robot_speed_m_s: float = 0.03
    robot_acceleration_m_s2: float = 0.10
    workspace_xyz_min_m: tuple[float, float, float] = (0.20, -1.50, -0.30)
    workspace_xyz_max_m: tuple[float, float, float] = (1.20, +1.50, +1.50)
    controller_max_translation_m: float = 0.010
    controller_max_rotation_rad: float = 0.20
    advancer_port: str = "/dev/ttyACM0"
    advancer_baud: int = 115200
    advancer_delay_us: int = 20
    advancer_min_command_mm: float = 0.166

    # Camera/vision paths.  These replace the hard-coded calibration and manual
    # boundary paths used inside the uploaded bounds_beam.py at run time.
    image_filename: str = "focused_image.jpg"
    roi_polygon_path: str = "/home/jack/Proper-Research/custom_area.json"
    manual_boundary_path: str = (
        "/home/jack/Proper-Research/manual_vessel_boundaries.json"
    )
    calibration_points_path: str = (
        "/home/jack/Proper-Research/calibration_points.json"
    )
    blue_roi_path: str = "blue_roi_box.json"
    green_roi_path: str = "green_roi_box.json"
    pivot_hint_px: tuple[int, int] = (309, 330)
    vision_samples_per_point: int = 1
    settle_time_s: float = 1.0

    # Fixed beam/model frame and initial insertion length.
    pivot_pose6: tuple[float, float, float, float, float, float] = (
        0.84813282,
        -0.68127317,
        -0.10000000,
        float(np.pi),
        1.0e-3,
        1.0e-3,
    )
    initial_beam_length_m: float = 0.044
    use_vision_length_in_model: bool = False
    min_vision_length_m: float = 0.010
    max_vision_length_m: float = 0.050
    magnet_yaw_calibration_deg: float = -10.0
    jacobian_variant: str = "contact"  # "contact" or "no_contact"
    sensitivity_workers: int = 1

    # Model contact parameters (copied from the current optimized script).
    beam_contact_radius_m: float = 0.001
    contact_k: float = 1.0e5
    contact_penetration_switch_m: float = 5.0e-5
    contact_k_hard: float = 1.0e10
    contact_smooth_epsilon_m: float = 1.0e-5
    contact_window: int = 3

    # Model/controller sample time.  A commanded translation is delta/dt.
    dt_s: float = 0.05

    # Compare tangent angles in these robot-frame axes.  (0, 1) means XY.
    angle_plane_axes: tuple[int, int] = (0, 1)

    # -------------------------------------------------------------------------
    # ARC VALIDATION
    # -------------------------------------------------------------------------
    # The arc center is in the same robot/model frame as p8[:3].  Confirm this
    # point carefully before enabling motion.  The default is the pivot xyz.
    arc_center_robot_m: tuple[float, float, float] = (
        0.84813282,
        -0.68127317,
        -0.10000000,
    )
    arc_plane_axes: tuple[int, int] = (0, 1)  # XY arc; the third axis is fixed
    arc_radius_mm: float | None = None  # None: derive radius from current pose
    arc_sweep_deg: float = 10.0  # signed: negative reverses direction
    arc_number_of_points: int = 21  # includes the current/start point
    return_along_arc_to_start: bool = True
    maximum_allowed_arc_radius_mm: float = 250.0

    # Independent per-command safety gate.  Keep this small for initial tests.
    maximum_magnet_step_mm: float = 1.0

    # -------------------------------------------------------------------------
    # DAMPED INVERSE-JACOBIAN + PROPORTIONAL POINT CONTROL
    # -------------------------------------------------------------------------
    # "beam_local_mm": points are transformed by pivot_pose6.
    # "robot_m": points are already robot-frame metres.
    target_point_frame: str = "beam_local_mm"
    target_points: tuple[tuple[float, float, float], ...] = (
        (-25.0, 3.0, 0.0),
    )

    # First-stage controller deliberately uses translation only.  Indices 0 and
    # 1 refer to magnet p8 x and y.  Outputs 0 and 1 are beam-tip robot x and y.
    controlled_tip_axes: tuple[int, ...] = (0, 1)
    active_magnet_translation_axes: tuple[int, ...] = (0, 1)
    proportional_gain: float = 0.35
    damped_inverse_lambda: float = 0.05
    finite_difference_step_mm: float = 0.25
    target_tolerance_mm: float = 0.50
    required_consecutive_hits: int = 2
    maximum_iterations_per_target: int = 30
    stall_window_iterations: int = 6
    minimum_progress_over_stall_window_mm: float = 0.10


CONFIG = ExperimentConfig()


# =============================================================================
# IMPLEMENTATION
# =============================================================================


@dataclass(frozen=True)
class HardwareControllerConfig:
    """Small duck-typed run configuration required by the controller factory."""

    Np: int = 2
    N_sqp: int = 1
    solver_mode: str = "lti"
    controller_kind: str = "mpc"

    def validate(self) -> None:
        if self.Np <= 0 or self.N_sqp <= 0:
            raise ValueError("Np and N_sqp must be positive.")
        if self.solver_mode not in {"lti", "ltv_oneshot", "sqp_full"}:
            raise ValueError(f"Unsupported solver_mode: {self.solver_mode}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite numbers; got {array}.")
    return array


def _wrap_degrees(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _plane_angle_deg(vector: np.ndarray, axes: tuple[int, int]) -> float:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    i, j = axes
    if max(i, j) >= vector.size:
        return float("nan")
    if np.linalg.norm(vector[[i, j]]) < 1.0e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(vector[j], vector[i])))


def _safe_stop(hw: Any) -> None:
    try:
        hw.stop()
    except Exception:
        pass


def validate_config(cfg: ExperimentConfig) -> None:
    if cfg.mode not in {"arc_validation", "point_control"}:
        raise ValueError("mode must be 'arc_validation' or 'point_control'.")
    if cfg.jacobian_variant not in {"contact", "no_contact"}:
        raise ValueError("jacobian_variant must be 'contact' or 'no_contact'.")
    if cfg.target_point_frame not in {"beam_local_mm", "robot_m"}:
        raise ValueError("target_point_frame must be 'beam_local_mm' or 'robot_m'.")
    if cfg.dt_s <= 0.0 or cfg.settle_time_s < 0.0:
        raise ValueError("dt_s must be positive and settle_time_s non-negative.")
    if cfg.vision_samples_per_point < 1:
        raise ValueError("vision_samples_per_point must be at least 1.")
    if cfg.arc_number_of_points < 2:
        raise ValueError("arc_number_of_points must be at least 2.")
    if cfg.maximum_magnet_step_mm <= 0.0:
        raise ValueError("maximum_magnet_step_mm must be positive.")
    if cfg.finite_difference_step_mm <= 0.0:
        raise ValueError("finite_difference_step_mm must be positive.")
    if not (0.0 < cfg.proportional_gain <= 1.0):
        raise ValueError("proportional_gain must lie in (0, 1] for this script.")
    if cfg.damped_inverse_lambda < 0.0:
        raise ValueError("damped_inverse_lambda cannot be negative.")
    if len(cfg.controlled_tip_axes) != len(cfg.active_magnet_translation_axes):
        raise ValueError(
            "Use the same number of controlled tip axes and active magnet axes "
            "for this first-stage controller."
        )
    for axis in cfg.controlled_tip_axes + cfg.active_magnet_translation_axes:
        if axis not in (0, 1, 2):
            raise ValueError("The first-stage inverse Jacobian supports xyz axes only.")

    xyz_min = _finite_vector(cfg.workspace_xyz_min_m, 3, "workspace_xyz_min_m")
    xyz_max = _finite_vector(cfg.workspace_xyz_max_m, 3, "workspace_xyz_max_m")
    if np.any(xyz_min >= xyz_max):
        raise ValueError("Every workspace minimum must be below its maximum.")


def build_physical_stiffness_profile(beam_params: Any) -> Any:
    wire = rod_section_stiffness(r=200e-6, E=50e6, nu=0.4)
    tip = rod_section_stiffness(
        r=float(beam_params.r),
        E=float(beam_params.E),
        nu=0.49,
    )
    return make_Kbt_inv_profile(
        EI_wire=wire["EI"],
        EI_tip=tip["EI"],
        GJ_wire=wire["GJ"],
        GJ_tip=tip["GJ"],
        bend_soft=1.0,
        tors_soft=1.0,
    )


def configure_bounds_beam_paths(cfg: ExperimentConfig) -> None:
    """Make paths hard-coded in the uploaded vision function configurable."""

    bounds_beam.MANUAL_VESSEL_BOUNDARY_FILE = cfg.manual_boundary_path
    original_loader = getattr(
        bounds_beam,
        "_beam_experiment_original_calibration_loader",
        bounds_beam.get_saved_2_point_calibration,
    )
    bounds_beam._beam_experiment_original_calibration_loader = original_loader

    def configured_loader(_ignored_path: str = "calibration_points.json") -> dict:
        return original_loader(cfg.calibration_points_path)

    bounds_beam.get_saved_2_point_calibration = configured_loader


def connect_hardware(cfg: ExperimentConfig) -> LiveHardwareController:
    # use_advancer is deliberately False: these experiments do not insert beam.
    return LiveHardwareController(
        robot_ip=cfg.robot_ip,
        dry_run=False,
        use_advancer=False,
        advancer_port=cfg.advancer_port,
        advancer_baud=cfg.advancer_baud,
        advancer_delay_us=cfg.advancer_delay_us,
        advancer_min_cmd_mm=cfg.advancer_min_command_mm,
        xyz_min=cfg.workspace_xyz_min_m,
        xyz_max=cfg.workspace_xyz_max_m,
        max_trans_m=cfg.controller_max_translation_m,
        max_rot_rad=cfg.controller_max_rotation_rad,
        z_offset=cfg.z_offset_m,
        use_moveL_params=False,
        v=cfg.robot_speed_m_s,
        a=cfg.robot_acceleration_m_s2,
    )


def build_model_and_controller(
    cfg: ExperimentConfig,
    robot_pose6: np.ndarray,
) -> tuple[Any, Any]:
    """Build the existing forward model and expose its forward/Jacobian wrappers."""

    # Refresh the image used by the initial lumen reconstruction.
    new_capture(filename=cfg.image_filename)
    pivot_pose6 = _finite_vector(cfg.pivot_pose6, 6, "pivot_pose6")
    lumen_C, lumen_R, _ = build_initial_lumen_from_vision(
        pivot_point=pivot_pose6,
        image_filename=cfg.image_filename,
        roi_polygon_path=cfg.roi_polygon_path,
        blue_roi_path=cfg.blue_roi_path,
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint_px,
        show=False,
    )

    magnet_params = default_magnet_params()
    beam_params = default_beam_params()
    pivot_transform = ur_pose6_to_T(pivot_pose6)
    pivot_position, pivot_quaternion = T_to_p_quat_wxyz(pivot_transform)
    nominal_magnetisation = np.array(
        [-float(magnet_params.mag_epm), 0.0, 0.0], dtype=float
    )
    magnetisation = rotate_body_xy(
        nominal_magnetisation,
        cfg.magnet_yaw_calibration_deg,
    )
    stiffness_profile = build_physical_stiffness_profile(beam_params)
    contact_params = ContactParams(
        r_beam=cfg.beam_contact_radius_m,
        k=cfg.contact_k,
        pen_switch=cfg.contact_penetration_switch_m,
        k_hard=cfg.contact_k_hard,
        smooth=True,
        smooth_eps=cfg.contact_smooth_epsilon_m,
        window=cfg.contact_window,
    )

    model_bundle = build_hardware_model_bundle_optimized(
        p0_ur=pivot_position,
        q0_ur=pivot_quaternion,
        Kinv_fun=stiffness_profile,
        m_body=magnetisation,
        lumen_C_robot_m=lumen_C,
        lumen_R_robot_m=lumen_R,
        contact_params=contact_params,
        jacobian_variant=cfg.jacobian_variant,
        sensitivity_workers=cfg.sensitivity_workers,
        result_detail="contact",
        store_history=False,
        store_vectors_in_info=False,
        copy_cached_results=False,
    )

    run_cfg = HardwareControllerConfig()
    run_cfg.validate()
    design_cfg = ControllerDesignConfig(
        n_out=6,
        w_tracking=(1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
        reference_mode="point",
        ref_lookahead_m=0.5e-3,
        ref_stride_m=0.5e-3,
        ref_search_backward_m=0.5e-3,
        ref_search_forward_m=5.0e-3,
        ref_max_progress_per_step_m=1.0e-3,
        ref_weight_start=1.0,
        ref_weight_end=1.0,
        q_contour=0.0,
        q_lag=0.0,
        contouring_axis_weights=(1.0, 1.0, 1.0),
        progress_constraint_mode="none",
        enable_progress_slack=False,
        enable_hard_progress_constraint=False,
        progress_request_m=0.0,
        progress_reward_weight=0.0,
        progress_slack_quadratic_weight=0.0,
        progress_slack_linear_weight=0.0,
        progress_slack_max_fraction=1.0,
        enable_hard_tip_tangent_angle=False,
        tip_tangent_max_angle_deg=40.0,
        tip_tangent_activation_clearance_m=1.0e-3,
    )
    design_cfg.validate()
    qp_settings = PersistentOSQPSettings(
        eps_abs=1.0e-5,
        eps_rel=1.0e-5,
        max_iter=100000,
        polishing=False,
        verbose=False,
    )
    controller_pack = build_controller_optimized(
        start_point=np.asarray(robot_pose6, dtype=float).reshape(6),
        L0=cfg.initial_beam_length_m,
        dt=cfg.dt_s,
        plant_model=model_bundle.prediction_model,
        jacobian_model=model_bundle.jacobian_model,
        lumen_C=model_bundle.lumen_C,
        lumen_R=model_bundle.lumen_R,
        run_cfg=run_cfg,
        design_cfg=design_cfg,
        copy_models=False,
        validate_nonlinear_candidate=False,
        collect_full_diagnostics=False,
        verbose_controller=False,
        qp_settings=qp_settings,
    )
    return controller_pack["mpc"], model_bundle


def _forward_output(mpc: Any, p8: np.ndarray) -> np.ndarray:
    try:
        output = mpc.forward_tip_fn(p8, commit=False)
    except TypeError:
        output = mpc.forward_tip_fn(p8)
    output = np.asarray(output, dtype=float).reshape(-1)
    if output.size < 3 or not np.all(np.isfinite(output)):
        raise RuntimeError(f"Forward model returned invalid output: {output}")
    return output


def _output_parts(
    output: np.ndarray,
    pivot_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(output, dtype=float).reshape(-1)
    tip = output[:3].copy()
    if output.size >= 6 and np.linalg.norm(output[3:6]) > 1.0e-12:
        tangent = output[3:6].copy()
    else:
        tangent = tip - np.asarray(pivot_xyz, dtype=float).reshape(3)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1.0e-12:
        tangent = tangent / tangent_norm
    return tip, tangent


def _model_length_from_vision(cfg: ExperimentConfig, length_mm: float) -> float:
    if not cfg.use_vision_length_in_model or not np.isfinite(length_mm):
        return cfg.initial_beam_length_m
    return float(
        np.clip(
            length_mm / 1.0e3,
            cfg.min_vision_length_m,
            cfg.max_vision_length_m,
        )
    )


def measure_state(
    *,
    cfg: ExperimentConfig,
    hw: LiveHardwareController,
    mpc: Any,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    overlay_path: Path | None,
) -> dict[str, Any]:
    """Capture one or more frames and return median measured/model state."""

    measurements: list[np.ndarray] = []
    raw_angles: list[float] = []
    raw_tip_x: list[float] = []
    raw_tip_y: list[float] = []
    beam_lengths_mm: list[float] = []

    for sample_index in range(cfg.vision_samples_per_point):
        new_capture(filename=cfg.image_filename)
        save_path = None
        if overlay_path is not None and sample_index == cfg.vision_samples_per_point - 1:
            save_path = str(overlay_path)
        result = bounds_beam.reconstruct_beam_within_vessel(
            image_filename=cfg.image_filename,
            red_roi_path=cfg.roi_polygon_path,
            red_roi_polygon=roi_polygon,
            blue_roi_path=cfg.blue_roi_path,
            green_roi_path=cfg.green_roi_path,
            pivot_hint=cfg.pivot_hint_px,
            show=False,
            save_overlay_path=save_path,
            base_px_ref=manual_frame["base_px"],
            ex_ref=manual_frame["ex_img"],
            ey_ref=manual_frame["ey_img"],
        )
        result["base_px_ref"] = manual_frame["base_px"]
        result["ex_ref"] = manual_frame["ex_img"]
        result["ey_ref"] = manual_frame["ey_img"]
        measured = np.asarray(
            vision_result_to_x_meas_robot(
                result,
                pivot_point_pose6=np.asarray(cfg.pivot_pose6, dtype=float),
            ),
            dtype=float,
        ).reshape(-1)
        if measured.size < 3 or not np.all(np.isfinite(measured)):
            raise RuntimeError(f"Vision returned invalid measurement: {measured}")
        measurements.append(measured)

        tip_result = result.get("tip_result", {})
        raw_angles.append(float(tip_result.get("tip_tangent_angle_deg", np.nan)))
        raw_tip = tip_result.get("tip_image_px", (np.nan, np.nan))
        raw_tip_x.append(float(raw_tip[0]))
        raw_tip_y.append(float(raw_tip[1]))
        beam_lengths_mm.append(float(result.get("beam_length_mm", np.nan)))

    x_measured = np.nanmedian(np.vstack(measurements), axis=0)
    if x_measured.size >= 6 and np.linalg.norm(x_measured[3:6]) > 1.0e-12:
        x_measured[3:6] /= np.linalg.norm(x_measured[3:6])

    robot_pose6 = hw.get_robot_pose_once()
    if robot_pose6 is None:
        raise RuntimeError("Could not read the current UR robot pose.")
    robot_pose6 = _finite_vector(robot_pose6, 6, "robot_pose6")
    beam_length_mm = float(np.nanmedian(beam_lengths_mm))
    p8 = np.asarray(
        build_measured_p8_from_pose6_and_length(
            robot_pose6,
            _model_length_from_vision(cfg, beam_length_mm),
            z_offset=cfg.z_offset_m,
        ),
        dtype=float,
    ).reshape(-1)
    if p8.size != int(mpc.np):
        raise RuntimeError(f"Expected p8 size {mpc.np}, got {p8.size}.")
    mpc.set_measured_params(p8)

    model_output = _forward_output(mpc, p8)
    pivot_xyz = np.asarray(cfg.pivot_pose6[:3], dtype=float)
    measured_tip, measured_tangent = _output_parts(x_measured, pivot_xyz)
    model_tip, model_tangent = _output_parts(model_output, pivot_xyz)

    return {
        "timestamp_utc": _utc_now(),
        "robot_pose6": robot_pose6,
        "p8": p8,
        "x_measured": x_measured,
        "model_output": model_output,
        "measured_tip": measured_tip,
        "measured_tangent": measured_tangent,
        "model_tip": model_tip,
        "model_tangent": model_tangent,
        "measured_angle_deg": _plane_angle_deg(
            measured_tangent, cfg.angle_plane_axes
        ),
        "model_angle_deg": _plane_angle_deg(model_tangent, cfg.angle_plane_axes),
        "raw_vision_tangent_angle_deg": float(np.nanmedian(raw_angles)),
        "raw_vision_tip_px": np.array(
            [np.nanmedian(raw_tip_x), np.nanmedian(raw_tip_y)], dtype=float
        ),
        "beam_length_mm": beam_length_mm,
    }


def calculate_bias(initial_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_m": initial_state["measured_tip"] - initial_state["model_tip"],
        "angle_deg": _wrap_degrees(
            initial_state["measured_angle_deg"] - initial_state["model_angle_deg"]
        ),
    }


def _workspace_check(cfg: ExperimentConfig, xyz: np.ndarray, label: str) -> None:
    xyz = _finite_vector(xyz, 3, label)
    xyz_min = np.asarray(cfg.workspace_xyz_min_m, dtype=float)
    xyz_max = np.asarray(cfg.workspace_xyz_max_m, dtype=float)
    if np.any(xyz < xyz_min) or np.any(xyz > xyz_max):
        raise RuntimeError(
            f"{label} {xyz} is outside workspace [{xyz_min}, {xyz_max}]."
        )


def _command_translation(
    *,
    cfg: ExperimentConfig,
    hw: LiveHardwareController,
    current_state: dict[str, Any],
    desired_p8: np.ndarray,
) -> np.ndarray:
    """Send one small translation while preserving robot orientation/insertion."""

    desired_p8 = np.asarray(desired_p8, dtype=float).reshape(-1)
    current_p8 = np.asarray(current_state["p8"], dtype=float).reshape(-1)
    delta_xyz = desired_p8[:3] - current_p8[:3]
    step_mm = 1.0e3 * float(np.linalg.norm(delta_xyz))
    if step_mm > cfg.maximum_magnet_step_mm + 1.0e-9:
        raise RuntimeError(
            f"Rejected {step_mm:.3f} mm command; limit is "
            f"{cfg.maximum_magnet_step_mm:.3f} mm."
        )
    _workspace_check(cfg, desired_p8[:3], "commanded magnet xyz")

    target_pose6 = np.asarray(current_state["robot_pose6"], dtype=float).copy()
    target_pose6[:3] += delta_xyz
    command_p8 = np.asarray(
        build_measured_p8_from_pose6_and_length(
            target_pose6,
            float(current_p8[7]),
            z_offset=cfg.z_offset_m,
        ),
        dtype=float,
    ).reshape(-1)

    # Preserve the exact desired model xyz.  Conversion should already agree,
    # but this explicit assignment avoids accumulating z-offset roundoff.
    command_p8[:3] = desired_p8[:3]
    u0 = np.zeros(7, dtype=float)
    u0[:3] = delta_xyz / cfg.dt_s
    hw.send_step(p_now=command_p8, u0=u0, dt=cfg.dt_s)
    return u0


def make_arc_points(cfg: ExperimentConfig, start_xyz: np.ndarray) -> np.ndarray:
    start_xyz = _finite_vector(start_xyz, 3, "start_xyz")
    center = _finite_vector(cfg.arc_center_robot_m, 3, "arc_center_robot_m")
    axis_a, axis_b = cfg.arc_plane_axes
    if axis_a == axis_b or axis_a not in (0, 1, 2) or axis_b not in (0, 1, 2):
        raise ValueError("arc_plane_axes must contain two different xyz axes.")

    relative = start_xyz - center
    start_angle = math.atan2(relative[axis_b], relative[axis_a])
    derived_radius = float(np.linalg.norm(relative[[axis_a, axis_b]]))
    radius = (
        derived_radius
        if cfg.arc_radius_mm is None
        else float(cfg.arc_radius_mm) / 1.0e3
    )
    if radius <= 1.0e-6:
        raise ValueError(
            "Arc radius is nearly zero. Check arc_center_robot_m or set arc_radius_mm."
        )
    if 1.0e3 * radius > cfg.maximum_allowed_arc_radius_mm:
        raise ValueError(
            f"Arc radius {1.0e3 * radius:.1f} mm exceeds the configured maximum."
        )

    # If a radius is explicitly supplied, use the point at the current angle.
    # The first step is independently checked by maximum_magnet_step_mm.
    angles = start_angle + np.deg2rad(
        np.linspace(0.0, cfg.arc_sweep_deg, cfg.arc_number_of_points)
    )
    points = np.repeat(start_xyz[None, :], len(angles), axis=0)
    points[:, axis_a] = center[axis_a] + radius * np.cos(angles)
    points[:, axis_b] = center[axis_b] + radius * np.sin(angles)

    for index, point in enumerate(points):
        _workspace_check(cfg, point, f"arc point {index}")
    step_sizes_mm = 1.0e3 * np.linalg.norm(np.diff(points, axis=0), axis=1)
    if step_sizes_mm.size and float(np.max(step_sizes_mm)) > cfg.maximum_magnet_step_mm:
        raise ValueError(
            f"Largest planned arc step is {np.max(step_sizes_mm):.3f} mm, above "
            f"maximum_magnet_step_mm={cfg.maximum_magnet_step_mm:.3f}. "
            "Increase arc_number_of_points or reduce the sweep/radius."
        )
    return points


def point_targets_robot_m(cfg: ExperimentConfig) -> list[np.ndarray]:
    if not cfg.target_points:
        raise ValueError("target_points cannot be empty in point_control mode.")
    points = [np.asarray(point, dtype=float).reshape(3) for point in cfg.target_points]
    if cfg.target_point_frame == "robot_m":
        return points

    transform = np.asarray(ur_pose6_to_T(np.asarray(cfg.pivot_pose6)), dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return [translation + rotation @ (point / 1.0e3) for point in points]


def finite_difference_translation_jacobian(
    *,
    cfg: ExperimentConfig,
    mpc: Any,
    p8: np.ndarray,
) -> np.ndarray:
    """Return d(selected tip position)/d(selected magnet translation).

    Central finite differences make the units explicit: both numerator and
    denominator are metres, so the Jacobian is dimensionless.  This avoids any
    ambiguity over whether the optimized MPC Jacobian has already been scaled
    by dt.  It still differentiates the same nonlinear forward beam model.
    """

    p8 = np.asarray(p8, dtype=float).reshape(-1)
    h = cfg.finite_difference_step_mm / 1.0e3
    rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    columns: list[np.ndarray] = []
    for magnet_axis in cfg.active_magnet_translation_axes:
        plus = p8.copy()
        minus = p8.copy()
        plus[magnet_axis] += h
        minus[magnet_axis] -= h
        y_plus = _forward_output(mpc, plus)[:3]
        y_minus = _forward_output(mpc, minus)[:3]
        columns.append((y_plus[rows] - y_minus[rows]) / (2.0 * h))
    jacobian = np.column_stack(columns)
    if not np.all(np.isfinite(jacobian)):
        raise RuntimeError(f"Non-finite finite-difference Jacobian: {jacobian}")
    return jacobian


def damped_inverse_step(
    *,
    cfg: ExperimentConfig,
    jacobian: np.ndarray,
    error_m: np.ndarray,
) -> np.ndarray:
    """Compute Kp J^T (J J^T + lambda^2 I)^-1 error."""

    jacobian = np.asarray(jacobian, dtype=float)
    error_m = np.asarray(error_m, dtype=float).reshape(jacobian.shape[0])
    damping_sq = float(cfg.damped_inverse_lambda) ** 2
    regularized = jacobian @ jacobian.T + damping_sq * np.eye(jacobian.shape[0])
    try:
        solved = np.linalg.solve(regularized, error_m)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Damped inverse-Jacobian solve failed.") from exc
    active_step = cfg.proportional_gain * (jacobian.T @ solved)

    max_step_m = cfg.maximum_magnet_step_mm / 1.0e3
    norm = float(np.linalg.norm(active_step))
    if norm > max_step_m:
        active_step *= max_step_m / norm
    return active_step


ROW_FIELDS = [
    "sample",
    "timestamp_utc",
    "mode",
    "phase",
    "target_index",
    "iteration",
    "command_sent",
    "magnet_command_x_m",
    "magnet_command_y_m",
    "magnet_command_z_m",
    "magnet_actual_x_m",
    "magnet_actual_y_m",
    "magnet_actual_z_m",
    "target_tip_x_m",
    "target_tip_y_m",
    "target_tip_z_m",
    "measured_tip_x_m",
    "measured_tip_y_m",
    "measured_tip_z_m",
    "model_tip_x_m",
    "model_tip_y_m",
    "model_tip_z_m",
    "bias_corrected_model_tip_x_m",
    "bias_corrected_model_tip_y_m",
    "bias_corrected_model_tip_z_m",
    "model_at_command_tip_x_m",
    "model_at_command_tip_y_m",
    "model_at_command_tip_z_m",
    "linear_next_tip_x_m",
    "linear_next_tip_y_m",
    "linear_next_tip_z_m",
    "measured_angle_deg",
    "model_angle_deg",
    "bias_corrected_model_angle_deg",
    "model_at_command_angle_deg",
    "raw_vision_tangent_angle_deg",
    "raw_vision_tip_x_px",
    "raw_vision_tip_y_px",
    "beam_length_mm",
    "model_position_error_mm",
    "bias_corrected_model_position_error_mm",
    "model_angle_error_deg",
    "bias_corrected_model_angle_error_deg",
    "target_error_controlled_axes_mm",
    "jacobian_condition_number",
    "jacobian_damping",
    "control_dx_m",
    "control_dy_m",
    "control_dz_m",
    "control_step_norm_mm",
]


class CsvRunLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=ROW_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def append(self, row: dict[str, Any]) -> None:
        self._writer.writerow({field: row.get(field, np.nan) for field in ROW_FIELDS})
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def make_row(
    *,
    sample: int,
    cfg: ExperimentConfig,
    phase: str,
    target_index: int,
    iteration: int,
    state: dict[str, Any] | None,
    bias: dict[str, Any],
    target_tip: np.ndarray | None = None,
    magnet_command: np.ndarray | None = None,
    model_at_command: np.ndarray | None = None,
    linear_next_tip: np.ndarray | None = None,
    jacobian: np.ndarray | None = None,
    control_step_xyz: np.ndarray | None = None,
    command_sent: bool = False,
) -> dict[str, Any]:
    nan3 = np.full(3, np.nan)
    measured_tip = nan3 if state is None else state["measured_tip"]
    model_tip = nan3 if state is None else state["model_tip"]
    actual_magnet = nan3 if state is None else state["p8"][:3]
    measured_angle = np.nan if state is None else state["measured_angle_deg"]
    model_angle = np.nan if state is None else state["model_angle_deg"]

    target_tip = nan3 if target_tip is None else np.asarray(target_tip, dtype=float)
    magnet_command = nan3 if magnet_command is None else np.asarray(magnet_command)[:3]
    linear_next_tip = nan3 if linear_next_tip is None else np.asarray(linear_next_tip)[:3]
    control_step_xyz = (
        nan3 if control_step_xyz is None else np.asarray(control_step_xyz, dtype=float)
    )

    if model_at_command is None:
        command_tip = nan3
        command_angle = np.nan
    else:
        command_tip, command_tangent = _output_parts(
            model_at_command,
            np.asarray(cfg.pivot_pose6[:3], dtype=float),
        )
        command_angle = _plane_angle_deg(command_tangent, cfg.angle_plane_axes)

    # A shadow/preview row has no new camera state, so use the nonlinear model
    # evaluated at the planned command as its model series value.
    if state is None and model_at_command is not None:
        model_tip = command_tip.copy()
        model_angle = command_angle
    corrected_tip = model_tip + np.asarray(bias["position_m"], dtype=float)
    corrected_angle = _wrap_degrees(model_angle + bias["angle_deg"])

    model_position_error_mm = (
        1.0e3 * float(np.linalg.norm(measured_tip - model_tip))
        if np.all(np.isfinite(measured_tip))
        else np.nan
    )
    corrected_position_error_mm = (
        1.0e3 * float(np.linalg.norm(measured_tip - corrected_tip))
        if np.all(np.isfinite(measured_tip))
        else np.nan
    )
    controlled_rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    target_error_mm = (
        1.0e3
        * float(np.linalg.norm(target_tip[controlled_rows] - measured_tip[controlled_rows]))
        if np.all(np.isfinite(target_tip)) and np.all(np.isfinite(measured_tip))
        else np.nan
    )
    jacobian_condition = (
        float(np.linalg.cond(jacobian)) if jacobian is not None else np.nan
    )

    raw_tip_px = nan3[:2] if state is None else state["raw_vision_tip_px"]
    row = {
        "sample": sample,
        "timestamp_utc": _utc_now() if state is None else state["timestamp_utc"],
        "mode": cfg.mode,
        "phase": phase,
        "target_index": target_index,
        "iteration": iteration,
        "command_sent": int(command_sent),
        "model_position_error_mm": model_position_error_mm,
        "bias_corrected_model_position_error_mm": corrected_position_error_mm,
        "model_angle_error_deg": _wrap_degrees(measured_angle - model_angle),
        "bias_corrected_model_angle_error_deg": _wrap_degrees(
            measured_angle - corrected_angle
        ),
        "target_error_controlled_axes_mm": target_error_mm,
        "jacobian_condition_number": jacobian_condition,
        "jacobian_damping": cfg.damped_inverse_lambda,
        "control_step_norm_mm": 1.0e3 * float(np.linalg.norm(control_step_xyz)),
        "measured_angle_deg": measured_angle,
        "model_angle_deg": model_angle,
        "bias_corrected_model_angle_deg": corrected_angle,
        "model_at_command_angle_deg": command_angle,
        "raw_vision_tangent_angle_deg": (
            np.nan if state is None else state["raw_vision_tangent_angle_deg"]
        ),
        "raw_vision_tip_x_px": raw_tip_px[0],
        "raw_vision_tip_y_px": raw_tip_px[1],
        "beam_length_mm": np.nan if state is None else state["beam_length_mm"],
    }
    for prefix, vector in (
        ("magnet_command", magnet_command),
        ("magnet_actual", actual_magnet),
        ("target_tip", target_tip),
        ("measured_tip", measured_tip),
        ("model_tip", model_tip),
        ("bias_corrected_model_tip", corrected_tip),
        ("model_at_command_tip", command_tip),
        ("linear_next_tip", linear_next_tip),
    ):
        row[f"{prefix}_x_m"] = float(vector[0])
        row[f"{prefix}_y_m"] = float(vector[1])
        row[f"{prefix}_z_m"] = float(vector[2])
    row["control_dx_m"] = float(control_step_xyz[0])
    row["control_dy_m"] = float(control_step_xyz[1])
    row["control_dz_m"] = float(control_step_xyz[2])
    return row


def _overlay_path(output_dir: Path, sample: int, cfg: ExperimentConfig) -> Path | None:
    if not cfg.save_vision_overlays:
        return None
    overlay_dir = output_dir / "vision_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    return overlay_dir / f"sample_{sample:05d}.png"


def run_arc_validation(
    *,
    cfg: ExperimentConfig,
    hw: LiveHardwareController,
    mpc: Any,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    output_dir: Path,
    logger: CsvRunLogger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = 0
    initial = measure_state(
        cfg=cfg,
        hw=hw,
        mpc=mpc,
        manual_frame=manual_frame,
        roi_polygon=roi_polygon,
        overlay_path=_overlay_path(output_dir, sample, cfg),
    )
    bias = calculate_bias(initial)
    arc_points = make_arc_points(cfg, initial["p8"][:3])

    first_row = make_row(
        sample=sample,
        cfg=cfg,
        phase="outbound",
        target_index=0,
        iteration=0,
        state=initial,
        bias=bias,
        magnet_command=arc_points[0],
        model_at_command=initial["model_output"],
    )
    logger.append(first_row)
    rows.append(first_row)

    print(f"Arc start p8 xyz: {initial['p8'][:3]}")
    print(f"Arc end xyz:      {arc_points[-1]}")
    print(
        "Largest arc step: "
        f"{1.0e3 * np.max(np.linalg.norm(np.diff(arc_points, axis=0), axis=1)):.3f} mm"
    )

    if not cfg.send_commands:
        print("Preview only: model predictions are being logged; the robot will not move.")
        for index, xyz in enumerate(arc_points[1:], start=1):
            p8_command = initial["p8"].copy()
            p8_command[:3] = xyz
            model_command = _forward_output(mpc, p8_command)
            sample += 1
            row = make_row(
                sample=sample,
                cfg=cfg,
                phase="preview",
                target_index=index,
                iteration=0,
                state=None,
                bias=bias,
                magnet_command=xyz,
                model_at_command=model_command,
            )
            logger.append(row)
            rows.append(row)
        return rows, bias

    path: list[tuple[str, int, np.ndarray]] = [
        ("outbound", index, xyz)
        for index, xyz in enumerate(arc_points[1:], start=1)
    ]
    if cfg.return_along_arc_to_start:
        path.extend(
            ("return", index, xyz)
            for index, xyz in reversed(list(enumerate(arc_points[:-1])))
        )

    state = initial
    for phase, index, xyz in path:
        p8_command = state["p8"].copy()
        p8_command[:3] = xyz
        model_command = _forward_output(mpc, p8_command)
        control_rate = _command_translation(
            cfg=cfg,
            hw=hw,
            current_state=state,
            desired_p8=p8_command,
        )
        time.sleep(cfg.settle_time_s)
        sample += 1
        state = measure_state(
            cfg=cfg,
            hw=hw,
            mpc=mpc,
            manual_frame=manual_frame,
            roi_polygon=roi_polygon,
            overlay_path=_overlay_path(output_dir, sample, cfg),
        )
        control_delta = control_rate[:3] * cfg.dt_s
        row = make_row(
            sample=sample,
            cfg=cfg,
            phase=phase,
            target_index=index,
            iteration=0,
            state=state,
            bias=bias,
            magnet_command=xyz,
            model_at_command=model_command,
            control_step_xyz=control_delta,
            command_sent=True,
        )
        logger.append(row)
        rows.append(row)
        print(
            f"[{phase:8s} {index:03d}] "
            f"measured={state['measured_angle_deg']:+7.2f} deg, "
            f"model={state['model_angle_deg']:+7.2f} deg, "
            f"bias-corrected error={row['bias_corrected_model_angle_error_deg']:+6.2f} deg"
        )
    return rows, bias


def run_point_control(
    *,
    cfg: ExperimentConfig,
    hw: LiveHardwareController,
    mpc: Any,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    output_dir: Path,
    logger: CsvRunLogger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = 0
    state = measure_state(
        cfg=cfg,
        hw=hw,
        mpc=mpc,
        manual_frame=manual_frame,
        roi_polygon=roi_polygon,
        overlay_path=_overlay_path(output_dir, sample, cfg),
    )
    bias = calculate_bias(state)
    targets = point_targets_robot_m(cfg)
    controlled_rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    active_axes = np.asarray(cfg.active_magnet_translation_axes, dtype=int)

    for target_index, target in enumerate(targets):
        hits = 0
        error_history_mm: list[float] = []
        for iteration in range(cfg.maximum_iterations_per_target):
            error = target[controlled_rows] - state["measured_tip"][controlled_rows]
            error_mm = 1.0e3 * float(np.linalg.norm(error))
            error_history_mm.append(error_mm)
            if error_mm <= cfg.target_tolerance_mm:
                hits += 1
            else:
                hits = 0

            if hits >= cfg.required_consecutive_hits:
                row = make_row(
                    sample=sample,
                    cfg=cfg,
                    phase="target_reached",
                    target_index=target_index,
                    iteration=iteration,
                    state=state,
                    bias=bias,
                    target_tip=target,
                )
                logger.append(row)
                rows.append(row)
                print(f"Target {target_index} reached: {error_mm:.3f} mm error.")
                break

            # Once inside tolerance, verify with another independent camera
            # frame instead of commanding a needless sub-tolerance movement.
            if hits > 0:
                row = make_row(
                    sample=sample,
                    cfg=cfg,
                    phase="tolerance_check",
                    target_index=target_index,
                    iteration=iteration,
                    state=state,
                    bias=bias,
                    target_tip=target,
                )
                logger.append(row)
                rows.append(row)
                time.sleep(cfg.settle_time_s)
                sample += 1
                state = measure_state(
                    cfg=cfg,
                    hw=hw,
                    mpc=mpc,
                    manual_frame=manual_frame,
                    roi_polygon=roi_polygon,
                    overlay_path=_overlay_path(output_dir, sample, cfg),
                )
                continue

            jacobian = finite_difference_translation_jacobian(
                cfg=cfg,
                mpc=mpc,
                p8=state["p8"],
            )
            active_step = damped_inverse_step(
                cfg=cfg,
                jacobian=jacobian,
                error_m=error,
            )
            step_xyz = np.zeros(3, dtype=float)
            step_xyz[active_axes] = active_step
            p8_command = state["p8"].copy()
            p8_command[:3] += step_xyz
            _workspace_check(cfg, p8_command[:3], "inverse-Jacobian command")
            model_command = _forward_output(mpc, p8_command)

            # Linear prediction in all three tip coordinates.  Only controlled
            # axes have a calculated increment; the other axes remain NaN.
            linear_tip = np.full(3, np.nan)
            linear_tip[controlled_rows] = (
                state["model_tip"][controlled_rows] + jacobian @ active_step
            )

            if not cfg.send_commands:
                row = make_row(
                    sample=sample,
                    cfg=cfg,
                    phase="preview",
                    target_index=target_index,
                    iteration=iteration,
                    state=state,
                    bias=bias,
                    target_tip=target,
                    magnet_command=p8_command[:3],
                    model_at_command=model_command,
                    linear_next_tip=linear_tip,
                    jacobian=jacobian,
                    control_step_xyz=step_xyz,
                    command_sent=False,
                )
                logger.append(row)
                rows.append(row)
                print(
                    f"Preview target {target_index}: error={error_mm:.3f} mm, "
                    f"proposed magnet step={1.0e3*np.linalg.norm(step_xyz):.3f} mm, "
                    f"cond(J)={np.linalg.cond(jacobian):.2f}"
                )
                break

            _command_translation(
                cfg=cfg,
                hw=hw,
                current_state=state,
                desired_p8=p8_command,
            )
            time.sleep(cfg.settle_time_s)
            sample += 1
            state = measure_state(
                cfg=cfg,
                hw=hw,
                mpc=mpc,
                manual_frame=manual_frame,
                roi_polygon=roi_polygon,
                overlay_path=_overlay_path(output_dir, sample, cfg),
            )
            row = make_row(
                sample=sample,
                cfg=cfg,
                phase="control",
                target_index=target_index,
                iteration=iteration,
                state=state,
                bias=bias,
                target_tip=target,
                magnet_command=p8_command[:3],
                model_at_command=model_command,
                linear_next_tip=linear_tip,
                jacobian=jacobian,
                control_step_xyz=step_xyz,
                command_sent=True,
            )
            logger.append(row)
            rows.append(row)
            new_error_mm = row["target_error_controlled_axes_mm"]
            print(
                f"[target {target_index:02d}, iter {iteration:02d}] "
                f"error={new_error_mm:.3f} mm, "
                f"step={1.0e3*np.linalg.norm(step_xyz):.3f} mm, "
                f"cond(J)={np.linalg.cond(jacobian):.2f}"
            )

            window = cfg.stall_window_iterations
            if len(error_history_mm) >= window:
                progress = error_history_mm[-window] - error_history_mm[-1]
                if progress < cfg.minimum_progress_over_stall_window_mm:
                    raise RuntimeError(
                        f"Point controller stalled at target {target_index}: "
                        f"only {progress:.3f} mm progress over {window} iterations."
                    )
        else:
            raise RuntimeError(
                f"Target {target_index} was not reached within "
                f"{cfg.maximum_iterations_per_target} iterations."
            )
    return rows, bias


def _finite_column(rows: Iterable[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([float(row.get(name, np.nan)) for row in rows], dtype=float)


def save_plot(rows: list[dict[str, Any]], path: Path, cfg: ExperimentConfig) -> None:
    if not rows:
        return
    samples = _finite_column(rows, "sample")
    measured_angle = _finite_column(rows, "measured_angle_deg")
    model_angle = _finite_column(rows, "model_angle_deg")
    corrected_angle = _finite_column(rows, "bias_corrected_model_angle_deg")
    command_angle = _finite_column(rows, "model_at_command_angle_deg")
    model_angle = np.where(np.isfinite(model_angle), model_angle, command_angle)
    corrected_angle = np.where(
        np.isfinite(corrected_angle),
        corrected_angle,
        np.asarray(
            [_wrap_degrees(value + 0.0) for value in model_angle], dtype=float
        ),
    )
    raw_error = _finite_column(rows, "model_position_error_mm")
    corrected_error = _finite_column(rows, "bias_corrected_model_position_error_mm")

    measured_x = _finite_column(rows, "measured_tip_x_m") * 1.0e3
    measured_y = _finite_column(rows, "measured_tip_y_m") * 1.0e3
    model_x = _finite_column(rows, "model_tip_x_m") * 1.0e3
    model_y = _finite_column(rows, "model_tip_y_m") * 1.0e3
    corrected_x = _finite_column(rows, "bias_corrected_model_tip_x_m") * 1.0e3
    corrected_y = _finite_column(rows, "bias_corrected_model_tip_y_m") * 1.0e3
    command_x = _finite_column(rows, "model_at_command_tip_x_m") * 1.0e3
    command_y = _finite_column(rows, "model_at_command_tip_y_m") * 1.0e3
    model_x = np.where(np.isfinite(model_x), model_x, command_x)
    model_y = np.where(np.isfinite(model_y), model_y, command_y)
    target_x = _finite_column(rows, "target_tip_x_m") * 1.0e3
    target_y = _finite_column(rows, "target_tip_y_m") * 1.0e3

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(samples, measured_angle, "o-", label="measured")
    axes[0].plot(samples, model_angle, "s--", label="model")
    axes[0].plot(samples, corrected_angle, ".-", label="model + initial bias")
    axes[0].set_xlabel("sample")
    axes[0].set_ylabel("beam tangent angle (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(measured_x, measured_y, "o-", label="measured tip")
    axes[1].plot(model_x, model_y, "s--", label="model tip")
    axes[1].plot(corrected_x, corrected_y, ".-", label="model + initial bias")
    if np.any(np.isfinite(target_x)):
        axes[1].scatter(target_x, target_y, marker="x", s=80, label="target")
    axes[1].set_xlabel("robot x (mm)")
    axes[1].set_ylabel("robot y (mm)")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(samples, raw_error, "o-", label="raw model")
    axes[2].plot(samples, corrected_error, ".-", label="model + initial bias")
    if cfg.mode == "point_control":
        target_error = _finite_column(rows, "target_error_controlled_axes_mm")
        axes[2].plot(samples, target_error, "x-", label="target error")
    axes[2].set_xlabel("sample")
    axes[2].set_ylabel("position error (mm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle(f"{cfg.mode}: measured versus predicted beam state")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def save_summary(
    rows: list[dict[str, Any]],
    bias: dict[str, Any],
    path: Path,
    stop_reason: str,
) -> None:
    summary = {
        "stop_reason": stop_reason,
        "samples_logged": len(rows),
        "initial_position_bias_m": bias.get("position_m"),
        "initial_angle_bias_deg": bias.get("angle_deg"),
        "raw_position_error_rmse_mm": _rmse(
            _finite_column(rows, "model_position_error_mm")
        ),
        "bias_corrected_position_error_rmse_mm": _rmse(
            _finite_column(rows, "bias_corrected_model_position_error_mm")
        ),
        "raw_angle_error_rmse_deg": _rmse(
            _finite_column(rows, "model_angle_error_deg")
        ),
        "bias_corrected_angle_error_rmse_deg": _rmse(
            _finite_column(rows, "bias_corrected_model_angle_error_deg")
        ),
    }
    path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def make_output_directory(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def request_live_confirmation(cfg: ExperimentConfig, robot_pose6: np.ndarray) -> None:
    if not cfg.send_commands:
        return
    print("\nLIVE ROBOT MOTION IS ENABLED")
    print(f"Current robot pose6: {robot_pose6}")
    print(f"Per-step translation limit: {cfg.maximum_magnet_step_mm:.3f} mm")
    print("The advancer is disabled; only the source magnet robot will move.")
    response = input(
        f"Type exactly {cfg.live_confirmation_phrase!r} to continue: "
    ).strip()
    if response != cfg.live_confirmation_phrase:
        raise RuntimeError("Live run cancelled: confirmation phrase did not match.")


def main() -> None:
    cfg = CONFIG
    validate_config(cfg)
    output_dir = make_output_directory(cfg)
    logger = CsvRunLogger(output_dir / "measurements.csv")
    rows: list[dict[str, Any]] = []
    bias: dict[str, Any] = {}
    stop_reason = "completed"
    hw: LiveHardwareController | None = None

    print(f"Mode: {cfg.mode}")
    print(f"Commands enabled: {cfg.send_commands}")
    print(f"Output: {output_dir}")

    try:
        configure_bounds_beam_paths(cfg)
        hw = connect_hardware(cfg)
        robot_pose6 = hw.get_robot_pose_once()
        if robot_pose6 is None:
            raise RuntimeError("Could not read the initial UR robot pose.")
        robot_pose6 = _finite_vector(robot_pose6, 6, "initial robot pose6")
        request_live_confirmation(cfg, robot_pose6)

        mpc, _model_bundle = build_model_and_controller(cfg, robot_pose6)
        manual_frame = load_manual_vessel_boundaries_with_frame(
            cfg.manual_boundary_path
        )
        roi_polygon = load_polygon(cfg.roi_polygon_path)

        if cfg.mode == "arc_validation":
            rows, bias = run_arc_validation(
                cfg=cfg,
                hw=hw,
                mpc=mpc,
                manual_frame=manual_frame,
                roi_polygon=roi_polygon,
                output_dir=output_dir,
                logger=logger,
            )
        else:
            rows, bias = run_point_control(
                cfg=cfg,
                hw=hw,
                mpc=mpc,
                manual_frame=manual_frame,
                roi_polygon=roi_polygon,
                output_dir=output_dir,
                logger=logger,
            )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if hw is not None:
            _safe_stop(hw)
        print("\nStopped by user. No automatic recovery motion was attempted.")
    except Exception as exc:
        stop_reason = f"exception:{type(exc).__name__}: {exc}"
        if hw is not None:
            _safe_stop(hw)
        print(f"Stopped safely: {stop_reason}")
        raise
    finally:
        logger.close()
        if hw is not None:
            _safe_stop(hw)
        save_plot(rows, output_dir / "comparison.png", cfg)
        save_summary(rows, bias, output_dir / "summary.json", stop_reason)
        print(f"CSV:     {output_dir / 'measurements.csv'}")
        print(f"Plot:    {output_dir / 'comparison.png'}")
        print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
