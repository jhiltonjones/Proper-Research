"""Composite-beam hardware experiments with analytic/FD Jacobian selection.

This is the second-stage experiment driver.  It deliberately leaves
``beam_hardware_experiment.py`` unchanged and reuses its camera and hardware
helpers at run time.  There are no command-line arguments: edit only the
configuration block below.

State convention
----------------
The existing hardware runner establishes
``build_measured_p8_from_pose6_and_length(..., z_offset=hw.z_offset)`` as the
authoritative conversion from the UR pose to model coordinates.  This script
uses that exact helper.  ``z_offset_m`` is applied once, inside that helper.
The resulting state is

    p8 = [source_xyz, source_quaternion_wxyz, insertion_length]

and is converted without another translation offset to

    p7 = [source_xyz, source_rotation_vector, insertion_length].

For the initial controller only translation columns 0 and 1 are enabled, so
the analytic actuation-tangent and public-pose Jacobians have identical selected
columns.  Rotation/insertion columns remain structurally supported by the
analytic path but are rejected by the live first-stage safety validation.

Hardware execution has not been tested by this file's author.  Begin with
``send_commands=False`` and verify the logged raw/model poses before live use.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# =============================================================================
# USER CONFIGURATION -- all editable experiment values live in this section
# =============================================================================


@dataclass
class CompositeBeamConfig:
    """Physical and magnetic inputs for the specified composite beam."""

    beam_diameter_m: float = 2.0e-3
    particle_mass_fraction: float = 0.8
    particle_density_kg_m3: float = 7450.0
    particle_specific_moment_A_m2_kg: float = 80.6
    silicone_density_kg_m3: float = 1070.0
    remanence_fraction: float = 1.0
    # Calibrated 2026-09-10 against the live arc + dipole + insertion sweeps,
    # then refined against the CORNER sweep (calibration_2026-09-10/
    # corner_sweep_data.json): magnet in B.y at the triangle-corner insertion
    # (38 mm).  cam_v = 0.108*mag_y + 0.71 mm ; the model's lateral gain was
    # ~6% steep -> 3.0 MPa flattens it (slope ratio 1.008), and the +0.71 mm
    # real-beam pre-curl is matched by magnet_yaw_calibration_deg = -6 (was
    # -10, which over-biased the model +0.9 mm).  Corner RMS 0.96 -> 0.25 mm.
    effective_youngs_modulus_pa: float = 3.0e6
    poisson_ratio: float = 0.49
    inner_diameter_m: float = 0.0
    # -local x = the rod growth direction = world +B.x = along the axial field
    # from the coaxial source magnet -> aligned -> straight beam at the reference.
    magnetisation_axis_local: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    # 2026-09-11: selectable INDUCED (field-following) magnetisation, added to
    # test whether the beam's real magnetic response (2 mm dia. PDMS + 80%
    # iron particles -- a soft, not permanent, magnet) explains the ~3x
    # under-sensitivity to source-DIPOLE ROTATION found in the corner/arc/
    # dipole-rotation-vs-radius sweeps.  "fixed_axial" (default) is the
    # ORIGINAL, still-in-use model: a fixed moment_per_length along the rod,
    # independent of the local field.  "induced_axial_transverse" instead uses
    # proper_research.simulation.magnetic_beam.run_solver_smoke_test.
    # make_induced_axial_transverse_m_local_factory (chi_axial/chi_transverse
    # susceptibilities against the LOCAL field, tanh-saturated at
    # induced_m_sat_A_m).
    #
    # STATUS: implemented and gradient-verified (FD-vs-analytic on the total
    # energy, rel. error ~1e-6 including a chain-rule term the old virtual-work
    # formula does not have -- see gradients.py:magnetic_energy_gradient_u_fd),
    # but a joint fit against 66 live poses did NOT beat the fixed_axial model
    # (RMS 3.49 vs 3.13 mm; the dipole-rotation subset got WORSE, 4.42 vs
    # 4.15 mm) and drove chi_transverse -> 0 -- i.e. this simple linear,
    # shape-anisotropic susceptibility form does not explain the rotation gap
    # either.  Left available (NOT the default) for further investigation
    # rather than treated as the fix; see calibration_2026-09-10/fit_induced*.
    magnetisation_model: str = "fixed_axial"  # or "induced_axial_transverse"
    induced_chi_axial: float = 8.0
    induced_chi_transverse: float = 3.0
    induced_m_sat_A_m: float | None = 8.0


@dataclass
class ExperimentConfig:
    # Select "arc_validation" or "point_control".
    mode: str = "arc_validation"

    # False performs camera/model/plan checks and never calls send_step().
    send_commands: bool = False
    live_confirmation_phrase: str = "MOVE MAGNET"

    output_root: str = "beam_hardware_experiments"
    run_name: str = "beam_composite_analytic_jacobian"
    save_vision_overlays: bool = True

    # Robot and external source magnet.
    robot_ip: str = "192.168.56.101"
    z_offset_m: float = 0.0
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

    # Camera and fixed vision-frame files.
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
    pivot_hint_px: tuple[int, int] = (318, 292)
    vision_samples_per_point: int = 1
    settle_time_s: float = 1.0

    # Beam base/model frame and fixed insertion length for these experiments.
    pivot_pose6: tuple[float, float, float, float, float, float] = (
        0.5261705543740489+0.3,
        -0.6702,
        0,
        float(np.pi),
        1.0e-3,
        1.0e-3,
    )
    initial_beam_length_m: float = 0.01
    use_vision_length_in_model: bool = True
    min_vision_length_m: float = 0.010
    max_vision_length_m: float = 0.050
    # 2026-09-10 corner-sweep calibration: -6 deg reproduces the real beam's
    # +0.71 mm pre-curl toward +B.y; -10 over-biased the model by ~+0.9 mm.
    magnet_yaw_calibration_deg: float = -6.0
    jacobian_variant: str = "contact"  # "contact" or "no_contact"
    sensitivity_workers: int = 1

    composite: CompositeBeamConfig = field(default_factory=CompositeBeamConfig)

    # Smooth contact parameters used by the selected forward model.
    beam_contact_radius_m: float = 0.001
    contact_k: float = 1.0e5
    contact_penetration_switch_m: float = 5.0e-5
    contact_k_hard: float = 1.0e10
    contact_smooth_epsilon_m: float = 1.0e-5
    contact_window: int = 3

    # Jacobian source and numerical settings.
    jacobian_source: str = "analytic"
    # Allowed: "analytic", "finite_difference", "compare".
    analytic_jacobian_mode: str = "fast"  # "fast" or "accurate"
    analytic_eps_theta: float = 1.0e-6
    analytic_eps_hess: float = 1.0e-4
    finite_difference_step_mm: float = 0.25
    controlled_output: str = "tip_position"
    controlled_tip_axes: tuple[int, ...] = (0, 1)
    active_control_axes: tuple[int, ...] = (0, 1)

    # Optional compare-mode command gate.
    enable_jacobian_consistency_gate: bool = True
    maximum_analytic_condition_number: float = 1.0e4
    maximum_relative_jacobian_disagreement: float = 0.35
    maximum_predicted_tip_displacement_mm: float = 2.0

    # Damped inverse-Jacobian proportional control and saturation.
    proportional_gain: float = 0.35
    damped_inverse_lambda: float = 0.05
    maximum_magnet_step_mm: float = 1.0
    maximum_axis_steps_mm: tuple[float, ...] = (0.75, 0.75)
    target_tolerance_mm: float = 0.50
    required_consecutive_hits: int = 2
    maximum_iterations_per_target: int = 30
    stall_window_iterations: int = 6
    minimum_progress_over_stall_window_mm: float = 0.10
    dt_s: float = 0.05

    # Angle comparison plane in robot/model coordinates; (0, 1) means XY.
    angle_plane_axes: tuple[int, int] = (0, 1)

    # Arc motion around arc_center_robot_m in arc_plane_axes.
    arc_center_robot_m: tuple[float, float, float] = (
        0.84813282,
        -0.68127317,
        -0.10000000,
    )
    arc_plane_axes: tuple[int, int] = (0, 1)
    arc_radius_mm: float | None = None
    arc_sweep_deg: float = 10.0
    arc_number_of_points: int = 21
    return_along_arc_to_start: bool = True
    maximum_allowed_arc_radius_mm: float = 250.0

    # Point targets: beam-local millimetres or robot-frame metres.
    target_point_frame: str = "beam_local_mm"
    target_points: tuple[tuple[float, float, float], ...] = (
        (-25.0, 3.0, 0.0),
    )


CONFIG = ExperimentConfig()


# =============================================================================
# PURE STATE, COMPOSITE AND JACOBIAN HELPERS
# =============================================================================


def _base_module():
    """Load v1 only when hardware/vision functionality is actually needed."""

    import proper_research.hardware.beam_hardware_experiment as base

    return base


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
        raise ValueError(f"{name} must contain {size} finite values; got {array}.")
    return array


def _wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _output_parts(output: Any, pivot_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(output, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array)):
        raise RuntimeError(f"Forward model returned invalid output: {array}")
    tip = array[:3].copy()
    if array.size >= 6 and np.linalg.norm(array[3:6]) > 1.0e-12:
        tangent = array[3:6].copy()
    else:
        # Position-only legacy fallback: this is the base-to-tip chord angle,
        # not the local distal tangent. It is explicitly logged as an assumption.
        tangent = tip - np.asarray(pivot_xyz, dtype=float).reshape(3)
    norm = float(np.linalg.norm(tangent))
    if norm > 1.0e-12:
        tangent /= norm
    return tip, tangent


def _plane_angle_deg(vector: np.ndarray, axes: tuple[int, int]) -> float:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    i, j = axes
    if max(i, j) >= vector.size or np.linalg.norm(vector[[i, j]]) < 1.0e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(vector[j], vector[i])))


def p8_to_p7(p8: np.ndarray) -> np.ndarray:
    """Convert [xyz, quaternion_wxyz, length] to [xyz, rotvec, length]."""

    p8 = _finite_vector(p8, 8, "p8")
    quaternion_wxyz = p8[3:7]
    norm = float(np.linalg.norm(quaternion_wxyz))
    if norm < 1.0e-12:
        raise ValueError("p8 contains a zero source quaternion.")
    quaternion_wxyz = quaternion_wxyz / norm
    quaternion_xyzw = np.r_[quaternion_wxyz[1:4], quaternion_wxyz[0]]
    rotvec = Rot.from_quat(quaternion_xyzw).as_rotvec()
    return np.r_[p8[:3], rotvec, p8[7]]


def robot_pose6_to_p8(
    robot_pose6: np.ndarray,
    insertion_length_m: float,
    z_offset_m: float,
) -> np.ndarray:
    """Apply z_offset exactly once through the existing hardware helper."""

    from proper_research.control.lab_ready_mpc import (
        build_measured_p8_from_pose6_and_length,
    )

    return _finite_vector(
        build_measured_p8_from_pose6_and_length(
            _finite_vector(robot_pose6, 6, "robot_pose6"),
            float(insertion_length_m),
            z_offset=float(z_offset_m),
        ),
        8,
        "converted p8",
    )


def robot_pose6_to_p7(
    robot_pose6: np.ndarray,
    insertion_length_m: float,
    z_offset_m: float,
) -> np.ndarray:
    return p8_to_p7(robot_pose6_to_p8(robot_pose6, insertion_length_m, z_offset_m))


def validate_composite_config(composite: CompositeBeamConfig) -> None:
    if composite.beam_diameter_m <= 0.0:
        raise ValueError("beam_diameter_m must be positive.")
    if not 0.0 <= composite.particle_mass_fraction <= 1.0:
        raise ValueError("particle_mass_fraction must lie in [0, 1].")
    if composite.particle_density_kg_m3 <= 0.0:
        raise ValueError("particle_density_kg_m3 must be positive.")
    if composite.particle_specific_moment_A_m2_kg < 0.0:
        raise ValueError("particle_specific_moment_A_m2_kg cannot be negative.")
    if composite.silicone_density_kg_m3 <= 0.0:
        raise ValueError("silicone_density_kg_m3 must be positive.")
    if not 0.0 <= composite.remanence_fraction <= 1.0:
        raise ValueError("remanence_fraction must lie in [0, 1].")
    if composite.effective_youngs_modulus_pa <= 0.0:
        raise ValueError("effective_youngs_modulus_pa must be positive.")
    if not -1.0 < composite.poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1, 0.5).")
    if not 0.0 <= composite.inner_diameter_m < composite.beam_diameter_m:
        raise ValueError("inner_diameter_m must be non-negative and below diameter.")
    axis = _finite_vector(composite.magnetisation_axis_local, 3, "magnetisation axis")
    if np.linalg.norm(axis) < 1.0e-12:
        raise ValueError("magnetisation_axis_local cannot be zero.")


def validate_config(cfg: ExperimentConfig) -> None:
    validate_composite_config(cfg.composite)
    if cfg.mode not in {"arc_validation", "point_control"}:
        raise ValueError("mode must be 'arc_validation' or 'point_control'.")
    if cfg.jacobian_source not in {"analytic", "finite_difference", "compare"}:
        raise ValueError(
            "jacobian_source must be 'analytic', 'finite_difference', or 'compare'."
        )
    if cfg.analytic_jacobian_mode not in {"fast", "accurate"}:
        raise ValueError("analytic_jacobian_mode must be 'fast' or 'accurate'.")
    if cfg.controlled_output != "tip_position":
        raise ValueError("This first-stage script supports controlled_output='tip_position'.")
    if cfg.jacobian_variant not in {"contact", "no_contact"}:
        raise ValueError("jacobian_variant must be 'contact' or 'no_contact'.")
    if cfg.target_point_frame not in {"beam_local_mm", "robot_m"}:
        raise ValueError("target_point_frame must be 'beam_local_mm' or 'robot_m'.")
    if len(cfg.controlled_tip_axes) != len(cfg.active_control_axes):
        raise ValueError("controlled_tip_axes and active_control_axes lengths must match.")
    if len(cfg.maximum_axis_steps_mm) != len(cfg.active_control_axes):
        raise ValueError("maximum_axis_steps_mm must match active_control_axes length.")
    if any(axis not in (0, 1, 2) for axis in cfg.controlled_tip_axes):
        raise ValueError("controlled_tip_axes must select tip XYZ rows 0, 1, or 2.")
    if any(axis not in range(7) for axis in cfg.active_control_axes):
        raise ValueError("active_control_axes must select actuation columns 0..6.")
    if cfg.send_commands and any(axis not in (0, 1, 2) for axis in cfg.active_control_axes):
        raise ValueError(
            "Live v2 motion initially supports source translations only; "
            "rotation and insertion must remain disabled."
        )
    if cfg.finite_difference_step_mm <= 0.0:
        raise ValueError("finite_difference_step_mm must be positive.")
    if cfg.analytic_eps_theta <= 0.0 or cfg.analytic_eps_hess <= 0.0:
        raise ValueError("Analytic Jacobian eps values must be positive.")
    if not 0.0 < cfg.proportional_gain <= 1.0:
        raise ValueError("proportional_gain must lie in (0, 1].")
    if cfg.damped_inverse_lambda < 0.0:
        raise ValueError("damped_inverse_lambda cannot be negative.")
    if cfg.maximum_magnet_step_mm <= 0.0:
        raise ValueError("maximum_magnet_step_mm must be positive.")
    if any(limit <= 0.0 for limit in cfg.maximum_axis_steps_mm):
        raise ValueError("All maximum_axis_steps_mm entries must be positive.")
    if cfg.dt_s <= 0.0 or cfg.settle_time_s < 0.0:
        raise ValueError("dt_s must be positive and settle_time_s non-negative.")
    if cfg.vision_samples_per_point < 1:
        raise ValueError("vision_samples_per_point must be at least 1.")
    if cfg.arc_number_of_points < 2:
        raise ValueError("arc_number_of_points must be at least 2.")
    xyz_min = _finite_vector(cfg.workspace_xyz_min_m, 3, "workspace minimum")
    xyz_max = _finite_vector(cfg.workspace_xyz_max_m, 3, "workspace maximum")
    if np.any(xyz_min >= xyz_max):
        raise ValueError("Every workspace minimum must be below its maximum.")


def calculate_composite_components(
    composite_cfg: CompositeBeamConfig,
) -> tuple[dict[str, Any], Any, Any]:
    """Return calculated properties, distributed magnetisation and stiffness."""

    validate_composite_config(composite_cfg)
    from proper_research.simulation.magnetic_beam.run_solver_smoke_test import (
        calculate_composite_beam_properties,
        make_Kinv_fun,
        make_induced_axial_transverse_m_local_factory,
        make_uniform_axial_m_local_factory,
    )

    calculated = calculate_composite_beam_properties(
        beam_diameter=composite_cfg.beam_diameter_m,
        particle_mass_fraction=composite_cfg.particle_mass_fraction,
        particle_density=composite_cfg.particle_density_kg_m3,
        particle_specific_moment=(
            composite_cfg.particle_specific_moment_A_m2_kg
        ),
        silicone_density=composite_cfg.silicone_density_kg_m3,
        remanence_fraction=composite_cfg.remanence_fraction,
    )
    axis = np.asarray(composite_cfg.magnetisation_axis_local, dtype=float)
    axis /= np.linalg.norm(axis)
    model = getattr(composite_cfg, "magnetisation_model", "fixed_axial")
    if model == "induced_axial_transverse":
        m_local_factory = make_induced_axial_transverse_m_local_factory(
            chi_axial=composite_cfg.induced_chi_axial,
            chi_transverse=composite_cfg.induced_chi_transverse,
            local_axis=tuple(axis.tolist()),
            m_sat_A_m=composite_cfg.induced_m_sat_A_m,
        )
    elif model == "fixed_axial":
        m_local_factory = make_uniform_axial_m_local_factory(
            moment_per_length=float(calculated["moment_per_length"]),
            local_axis=tuple(axis.tolist()),
        )
    else:
        raise ValueError(
            f"magnetisation_model must be 'fixed_axial' or "
            f"'induced_axial_transverse', got {model!r}."
        )
    if not callable(m_local_factory):
        raise TypeError("Composite m_local_factory must be callable.")
    Kinv_fun = make_Kinv_fun(
        youngs_modulus=composite_cfg.effective_youngs_modulus_pa,
        poisson_ratio=composite_cfg.poisson_ratio,
        outer_diameter=composite_cfg.beam_diameter_m,
        inner_diameter=composite_cfg.inner_diameter_m,
    )
    if not callable(Kinv_fun):
        raise TypeError("Composite Kinv_fun must be callable.")
    return dict(calculated), m_local_factory, Kinv_fun


class DirectForwardModelAdapter:
    """Expose a p8-compatible facade while retaining the raw p7 model."""

    np = 8

    def __init__(self, raw_model: Any, pivot_xyz: Iterable[float]):
        self.raw_model = raw_model
        self.pivot_xyz = _finite_vector(pivot_xyz, 3, "pivot_xyz")
        self.p = np.full(8, np.nan)

    def set_measured_params(self, p8: np.ndarray) -> None:
        self.p = _finite_vector(p8, 8, "measured p8").copy()

    def _coerce_output(self, result: Any) -> np.ndarray:
        try:
            array = np.asarray(result, dtype=float).reshape(-1)
            if array.size >= 3 and np.all(np.isfinite(array)):
                return array
        except (TypeError, ValueError):
            pass
        if hasattr(result, "tip"):
            tip = _finite_vector(result.tip, 3, "solve result tip")
            tangent = getattr(result, "tip_tangent", None)
            if tangent is not None:
                return np.r_[tip, _finite_vector(tangent, 3, "tip tangent")]
            return tip
        raise RuntimeError(
            "Could not extract a tip output from the forward model result. "
            "Expected an array or an object with .tip."
        )

    def evaluate_p7(self, p7: np.ndarray, *, commit: bool) -> np.ndarray:
        p7 = _finite_vector(p7, 7, "p7")
        try:
            result = self.raw_model(p7, commit=commit)
        except TypeError:
            if hasattr(self.raw_model, "solve"):
                result = self.raw_model.solve(p7, commit=commit)
            else:
                # Legacy callable models may always commit. The FD path restores
                # the nominal state immediately after all perturbations.
                result = self.raw_model(p7)
        return self._coerce_output(result)

    def forward_tip_fn(self, p8: np.ndarray, commit: bool = False) -> np.ndarray:
        return self.evaluate_p7(p8_to_p7(p8), commit=bool(commit))

    def commit_nominal(self, p8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p8 = _finite_vector(p8, 8, "nominal p8")
        p7 = p8_to_p7(p8)
        output = self.evaluate_p7(p7, commit=True)
        self.p = p8.copy()
        return p7, output


@dataclass
class JacobianEvaluation:
    selected: np.ndarray
    analytic: np.ndarray | None
    finite_difference: np.ndarray | None
    analytic_time_s: float
    finite_difference_time_s: float
    analytic_condition: float
    finite_difference_condition: float
    absolute_frobenius_difference: float
    relative_frobenius_difference: float
    maximum_absolute_element_error: float
    column_error_norms: np.ndarray
    hessian_reused: bool
    diagnostics: dict[str, Any]


def _condition_number(matrix: np.ndarray | None) -> float:
    if matrix is None or not np.all(np.isfinite(matrix)):
        return float("nan")
    return float(np.linalg.cond(matrix))


def analytic_selected_jacobian(
    adapter: DirectForwardModelAdapter,
    p8: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    """Commit nominal equilibrium, then request the matching implicit Jacobian."""

    started = time.perf_counter()
    p7, nominal_output = adapter.commit_nominal(p8)
    full = adapter.raw_model.jacobian_tip_actuation_tangent(
        p7,
        solve_if_needed=False,
        eps_theta=cfg.analytic_eps_theta,
        eps_hess=cfg.analytic_eps_hess,
        mode=cfg.analytic_jacobian_mode,
        reuse_cached=True,
    )
    full = np.asarray(full, dtype=float).reshape(3, 7)
    selected = full[np.ix_(cfg.controlled_tip_axes, cfg.active_control_axes)]
    diagnostics = dict(getattr(adapter.raw_model, "last_jacobian_diag", {}) or {})
    elapsed = time.perf_counter() - started
    return selected, nominal_output, diagnostics, elapsed


def finite_difference_selected_jacobian(
    adapter: DirectForwardModelAdapter,
    p8: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Central FD in the same selected p7 variables, restoring nominal cache."""

    started = time.perf_counter()
    p8 = _finite_vector(p8, 8, "p8")
    p7, nominal_output = adapter.commit_nominal(p8)
    rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    columns: list[np.ndarray] = []
    h_translation = cfg.finite_difference_step_mm / 1.0e3

    for control_axis in cfg.active_control_axes:
        if control_axis in (0, 1, 2):
            h = h_translation
        elif control_axis == 6:
            h = h_translation
        else:
            raise NotImplementedError(
                "Finite-difference rotation columns require SO(3) tangent "
                "composition. Keep active_control_axes translational in v2."
            )
        plus = p7.copy()
        minus = p7.copy()
        plus[control_axis] += h
        minus[control_axis] -= h
        y_plus = adapter.evaluate_p7(plus, commit=False)[:3]
        y_minus = adapter.evaluate_p7(minus, commit=False)[:3]
        columns.append((y_plus[rows] - y_minus[rows]) / (2.0 * h))

    # Mandatory restore: even a legacy callable that ignored commit=False now
    # has the exact nominal equilibrium cached for the next analytic request.
    nominal_output = adapter.evaluate_p7(p7, commit=True)
    adapter.p = p8.copy()
    selected = np.column_stack(columns)
    return selected, nominal_output, time.perf_counter() - started


def evaluate_jacobians(
    adapter: DirectForwardModelAdapter,
    p8: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[JacobianEvaluation, np.ndarray]:
    analytic = None
    finite_difference = None
    analytic_time = 0.0
    fd_time = 0.0
    diagnostics: dict[str, Any] = {}
    nominal_output = adapter.forward_tip_fn(p8, commit=True)

    if cfg.jacobian_source in {"analytic", "compare"}:
        analytic, nominal_output, diagnostics, analytic_time = (
            analytic_selected_jacobian(adapter, p8, cfg)
        )
    if cfg.jacobian_source in {"finite_difference", "compare"}:
        finite_difference, nominal_output, fd_time = (
            finite_difference_selected_jacobian(adapter, p8, cfg)
        )

    selected = analytic if analytic is not None else finite_difference
    if selected is None:
        raise RuntimeError("No Jacobian was calculated.")
    if analytic is not None and finite_difference is not None:
        difference = analytic - finite_difference
        absolute = float(np.linalg.norm(difference, ord="fro"))
        denominator = max(float(np.linalg.norm(finite_difference, ord="fro")), 1e-12)
        relative = absolute / denominator
        max_abs = float(np.max(np.abs(difference)))
        column_errors = np.linalg.norm(difference, axis=0)
    else:
        absolute = relative = max_abs = float("nan")
        column_errors = np.full(len(cfg.active_control_axes), np.nan)

    evaluation = JacobianEvaluation(
        selected=np.asarray(selected, dtype=float),
        analytic=None if analytic is None else np.asarray(analytic, dtype=float),
        finite_difference=(
            None
            if finite_difference is None
            else np.asarray(finite_difference, dtype=float)
        ),
        analytic_time_s=float(analytic_time),
        finite_difference_time_s=float(fd_time),
        analytic_condition=_condition_number(analytic),
        finite_difference_condition=_condition_number(finite_difference),
        absolute_frobenius_difference=absolute,
        relative_frobenius_difference=relative,
        maximum_absolute_element_error=max_abs,
        column_error_norms=column_errors,
        hessian_reused=bool(diagnostics.get("hessian_reused", False)),
        diagnostics=diagnostics,
    )
    enforce_jacobian_gate(evaluation, cfg)
    return evaluation, nominal_output


def enforce_jacobian_gate(
    evaluation: JacobianEvaluation,
    cfg: ExperimentConfig,
) -> None:
    if not np.all(np.isfinite(evaluation.selected)):
        raise RuntimeError("Jacobian safety gate: selected Jacobian is non-finite.")
    if cfg.jacobian_source != "compare" or not cfg.enable_jacobian_consistency_gate:
        return
    if evaluation.analytic is None or evaluation.finite_difference is None:
        raise RuntimeError("Jacobian safety gate: compare mode is missing a matrix.")
    if not np.all(np.isfinite(evaluation.analytic)) or not np.all(
        np.isfinite(evaluation.finite_difference)
    ):
        raise RuntimeError("Jacobian safety gate: compared Jacobian is non-finite.")
    if evaluation.analytic_condition > cfg.maximum_analytic_condition_number:
        raise RuntimeError(
            "Jacobian safety gate: analytic condition number "
            f"{evaluation.analytic_condition:.3g} exceeds "
            f"{cfg.maximum_analytic_condition_number:.3g}."
        )
    if (
        evaluation.relative_frobenius_difference
        > cfg.maximum_relative_jacobian_disagreement
    ):
        raise RuntimeError(
            "Jacobian safety gate: relative analytic/FD disagreement "
            f"{evaluation.relative_frobenius_difference:.3f} exceeds "
            f"{cfg.maximum_relative_jacobian_disagreement:.3f}."
        )


def damped_inverse_step(
    jacobian: np.ndarray,
    error_m: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """Calculate Kp J.T solve(J J.T + lambda^2 I, error), then saturate."""

    jacobian = np.asarray(jacobian, dtype=float)
    error_m = np.asarray(error_m, dtype=float).reshape(jacobian.shape[0])
    regularized = (
        jacobian @ jacobian.T
        + float(cfg.damped_inverse_lambda) ** 2 * np.eye(jacobian.shape[0])
    )
    try:
        step = cfg.proportional_gain * (
            jacobian.T @ np.linalg.solve(regularized, error_m)
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Damped inverse-Jacobian solve failed.") from exc

    axis_limits_m = np.asarray(cfg.maximum_axis_steps_mm, dtype=float) / 1.0e3
    step = np.clip(step, -axis_limits_m, axis_limits_m)
    total_limit_m = cfg.maximum_magnet_step_mm / 1.0e3
    norm = float(np.linalg.norm(step))
    if norm > total_limit_m:
        step *= total_limit_m / norm
    return step


def enforce_predicted_displacement_gate(
    evaluation: JacobianEvaluation,
    active_step: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    predicted = evaluation.selected @ np.asarray(active_step, dtype=float)
    predicted_norm_mm = 1.0e3 * float(np.linalg.norm(predicted))
    if predicted_norm_mm > cfg.maximum_predicted_tip_displacement_mm:
        raise RuntimeError(
            "Jacobian safety gate: predicted tip displacement "
            f"{predicted_norm_mm:.3f} mm exceeds "
            f"{cfg.maximum_predicted_tip_displacement_mm:.3f} mm."
        )
    return predicted


def _workspace_check(cfg: ExperimentConfig, xyz: np.ndarray, label: str) -> None:
    xyz = _finite_vector(xyz, 3, label)
    low = np.asarray(cfg.workspace_xyz_min_m, dtype=float)
    high = np.asarray(cfg.workspace_xyz_max_m, dtype=float)
    if np.any(xyz < low) or np.any(xyz > high):
        raise RuntimeError(f"{label} {xyz} lies outside [{low}, {high}].")


def make_arc_points(cfg: ExperimentConfig, start_xyz: np.ndarray) -> np.ndarray:
    start_xyz = _finite_vector(start_xyz, 3, "arc start")
    center = _finite_vector(cfg.arc_center_robot_m, 3, "arc center")
    axis_a, axis_b = cfg.arc_plane_axes
    if axis_a == axis_b or axis_a not in (0, 1, 2) or axis_b not in (0, 1, 2):
        raise ValueError("arc_plane_axes must contain two different XYZ axes.")
    relative = start_xyz - center
    start_angle = math.atan2(relative[axis_b], relative[axis_a])
    radius = (
        float(np.linalg.norm(relative[[axis_a, axis_b]]))
        if cfg.arc_radius_mm is None
        else cfg.arc_radius_mm / 1.0e3
    )
    if radius <= 1.0e-6:
        raise ValueError("Arc radius is nearly zero; check the configured center.")
    if radius * 1.0e3 > cfg.maximum_allowed_arc_radius_mm:
        raise ValueError("Arc radius exceeds maximum_allowed_arc_radius_mm.")
    angles = start_angle + np.deg2rad(
        np.linspace(0.0, cfg.arc_sweep_deg, cfg.arc_number_of_points)
    )
    points = np.repeat(start_xyz[None, :], angles.size, axis=0)
    points[:, axis_a] = center[axis_a] + radius * np.cos(angles)
    points[:, axis_b] = center[axis_b] + radius * np.sin(angles)
    for index, point in enumerate(points):
        _workspace_check(cfg, point, f"arc point {index}")
    steps_mm = 1.0e3 * np.linalg.norm(np.diff(points, axis=0), axis=1)
    if steps_mm.size and np.max(steps_mm) > cfg.maximum_magnet_step_mm:
        raise ValueError(
            f"Largest arc step {np.max(steps_mm):.3f} mm exceeds "
            f"maximum_magnet_step_mm={cfg.maximum_magnet_step_mm:.3f}."
        )
    return points


def point_targets_robot_m(cfg: ExperimentConfig) -> list[np.ndarray]:
    if not cfg.target_points:
        raise ValueError("target_points cannot be empty.")
    points = [np.asarray(point, dtype=float).reshape(3) for point in cfg.target_points]
    if cfg.target_point_frame == "robot_m":
        return points
    pivot = np.asarray(cfg.pivot_pose6, dtype=float).reshape(6)
    rotation = Rot.from_rotvec(pivot[3:6]).as_matrix()
    return [pivot[:3] + rotation @ (point / 1.0e3) for point in points]


# =============================================================================
# DIRECT COMPOSITE FORWARD MODEL CONSTRUCTION
# =============================================================================


def build_direct_model(
    cfg: ExperimentConfig,
    robot_pose6: np.ndarray,
) -> tuple[DirectForwardModelAdapter, dict[str, Any]]:
    """Build the raw optimized model with explicit composite components."""

    base = _base_module()
    from beam_direction_magnetisation.quarternions.quarternions_functions import (
        T_to_p_quat_wxyz,
    )
    from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
    from proper_research.parameters import default_magnet_params
    from proper_research.simulation.magnetic_beam.contact import ContactParams
    from proper_research.simulation.simulations.model_factory_optimized import (
        build_forward_model,
    )

    base.new_capture(filename=cfg.image_filename)
    pivot_pose6 = np.asarray(cfg.pivot_pose6, dtype=float)
    lumen_C, lumen_R, _ = base.build_initial_lumen_from_vision(
        pivot_point=pivot_pose6,
        image_filename=cfg.image_filename,
        roi_polygon_path=cfg.roi_polygon_path,
        blue_roi_path=cfg.blue_roi_path,
        green_roi_path=cfg.green_roi_path,
        pivot_hint=cfg.pivot_hint_px,
        show=False,
    )
    pivot_transform = ur_pose6_to_T(pivot_pose6)
    p0_ur, q0_ur = T_to_p_quat_wxyz(pivot_transform)

    calculated, m_local_factory, Kinv_fun = calculate_composite_components(
        cfg.composite
    )
    # Composite experiments require a callable distributed magnetisation.
    # There is intentionally no None/legacy fallback in this driver.
    if not callable(m_local_factory):
        raise TypeError("A callable composite m_local_factory is required.")

    magnet_params = default_magnet_params()
    nominal_m_body = np.array([-float(magnet_params.mag_epm), 0.0, 0.0])
    m_body = base.rotate_body_xy(nominal_m_body, cfg.magnet_yaw_calibration_deg)
    contact = ContactParams(
        r_beam=cfg.beam_contact_radius_m,
        k=cfg.contact_k,
        pen_switch=cfg.contact_penetration_switch_m,
        k_hard=cfg.contact_k_hard,
        smooth=True,
        smooth_eps=cfg.contact_smooth_epsilon_m,
        window=cfg.contact_window,
    )
    use_contact = cfg.jacobian_variant == "contact"
    raw_model = build_forward_model(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        contact_enabled=use_contact,
        use_lumen_jac=use_contact,
        contact_params=contact if use_contact else None,
        m_local_factory=m_local_factory,
        result_detail="contact",
        store_history=False,
        store_vectors_in_info=False,
        sensitivity_workers=cfg.sensitivity_workers,
        copy_cached_results=False,
    )
    raw_model.composite_properties = dict(calculated)
    raw_model.composite_inputs = asdict(cfg.composite)
    raw_model.beam_magnetisation_profile = "uniform_axial_composite"
    raw_model.beam_magnetisation_axis_local = np.asarray(
        cfg.composite.magnetisation_axis_local, dtype=float
    )

    adapter = DirectForwardModelAdapter(raw_model, pivot_pose6[:3])
    initial_p8 = robot_pose6_to_p8(
        robot_pose6,
        cfg.initial_beam_length_m,
        cfg.z_offset_m,
    )
    adapter.set_measured_params(initial_p8)
    metadata = {
        "composite_inputs": asdict(cfg.composite),
        "composite_calculated": calculated,
        "external_source_magnet_body_moment": np.asarray(m_body, dtype=float),
        "model_type": type(raw_model).__name__,
        "model_contact_enabled": use_contact,
        "p7_mapping": (
            "UR pose6 -> build_measured_p8_from_pose6_and_length(z_offset once) "
            "-> quaternion_wxyz_to_rotvec -> p7"
        ),
    }
    return adapter, metadata


# =============================================================================
# LOGGING
# =============================================================================


ROW_FIELDS = [
    "sample",
    "timestamp_utc",
    "mode",
    "phase",
    "target_index",
    "iteration",
    "command_sent",
    "stop_detail",
    "jacobian_source",
    "controlled_tip_axes_json",
    "active_control_axes_json",
    "pre_magnet_p8_json",
    "commanded_magnet_p8_json",
    "post_magnet_p8_json",
    "target_tip_m_json",
    "pre_measured_tip_m_json",
    "post_measured_tip_m_json",
    "model_at_pre_measured_pose_m_json",
    "model_at_commanded_pose_m_json",
    "model_at_post_measured_pose_m_json",
    "bias_corrected_model_post_tip_m_json",
    "one_step_jacobian_prediction_m_json",
    "pre_measured_angle_deg",
    "post_measured_angle_deg",
    "model_pre_angle_deg",
    "model_command_angle_deg",
    "model_post_angle_deg",
    "bias_corrected_model_post_angle_deg",
    "raw_model_post_position_error_mm",
    "bias_corrected_model_post_position_error_mm",
    "one_step_prediction_error_mm",
    "target_error_before_mm",
    "target_error_after_mm",
    "active_control_step_m_json",
    "source_translation_step_m_json",
    "control_step_norm_mm",
    "predicted_tip_displacement_mm",
    "selected_jacobian_json",
    "analytic_jacobian_json",
    "finite_difference_jacobian_json",
    "analytic_condition_number",
    "finite_difference_condition_number",
    "jacobian_absolute_frobenius_difference",
    "jacobian_relative_frobenius_difference",
    "jacobian_maximum_absolute_element_error",
    "jacobian_column_error_norms_json",
    "analytic_jacobian_time_ms",
    "finite_difference_jacobian_time_ms",
    "hessian_reused",
    "forward_model_jacobian_diagnostics_json",
    "composite_inputs_json",
    "composite_calculated_json",
]


def _json_cell(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=_json_default)


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


def _state_tip_angle(
    state: dict[str, Any] | None,
    key: str,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, float]:
    if state is None:
        return np.full(3, np.nan), float("nan")
    if key == "measured":
        tip = np.asarray(state["measured_tip"], dtype=float)
        tangent = np.asarray(state["measured_tangent"], dtype=float)
    else:
        tip = np.asarray(state["model_tip"], dtype=float)
        tangent = np.asarray(state["model_tangent"], dtype=float)
    return tip, _plane_angle_deg(tangent, cfg.angle_plane_axes)


def _output_tip_angle(output: Any, cfg: ExperimentConfig) -> tuple[np.ndarray, float]:
    tip, tangent = _output_parts(output, np.asarray(cfg.pivot_pose6[:3], dtype=float))
    return tip, _plane_angle_deg(tangent, cfg.angle_plane_axes)


def make_transition_row(
    *,
    sample: int,
    cfg: ExperimentConfig,
    phase: str,
    target_index: int,
    iteration: int,
    pre_state: dict[str, Any] | None,
    post_state: dict[str, Any] | None,
    commanded_p8: np.ndarray | None,
    target_tip: np.ndarray | None,
    model_at_command: np.ndarray | None,
    one_step_tip: np.ndarray | None,
    evaluation: JacobianEvaluation | None,
    active_step: np.ndarray | None,
    translation_step: np.ndarray | None,
    bias: dict[str, Any],
    composite_metadata: dict[str, Any],
    command_sent: bool,
    stop_detail: str = "",
) -> dict[str, Any]:
    nan3 = np.full(3, np.nan)
    pre_measured_tip, pre_measured_angle = _state_tip_angle(
        pre_state, "measured", cfg
    )
    pre_model_tip, pre_model_angle = _state_tip_angle(pre_state, "model", cfg)
    post_measured_tip, post_measured_angle = _state_tip_angle(
        post_state, "measured", cfg
    )
    post_model_tip, post_model_angle = _state_tip_angle(post_state, "model", cfg)
    if model_at_command is None:
        command_tip, command_angle = nan3, float("nan")
    else:
        command_tip, command_angle = _output_tip_angle(model_at_command, cfg)
    target_tip = nan3 if target_tip is None else np.asarray(target_tip, dtype=float)
    one_step_tip = nan3 if one_step_tip is None else np.asarray(one_step_tip, dtype=float)
    active_step = (
        np.full(len(cfg.active_control_axes), np.nan)
        if active_step is None
        else np.asarray(active_step, dtype=float)
    )
    translation_step = nan3 if translation_step is None else np.asarray(translation_step)
    corrected_model_tip = post_model_tip + np.asarray(bias["position_m"], dtype=float)
    corrected_model_angle = _wrap_degrees(post_model_angle + bias["angle_deg"])
    rows = np.asarray(cfg.controlled_tip_axes, dtype=int)

    def norm_mm(a: np.ndarray, b: np.ndarray, selected: bool = False) -> float:
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return float("nan")
        if selected:
            a, b = a[rows], b[rows]
        return 1.0e3 * float(np.linalg.norm(a - b))

    if evaluation is None:
        selected_jacobian = analytic_jacobian = fd_jacobian = None
        analytic_cond = fd_cond = abs_diff = rel_diff = max_error = np.nan
        column_errors = None
        analytic_ms = fd_ms = np.nan
        hessian_reused = False
        diagnostics = {}
        predicted_displacement_mm = np.nan
    else:
        selected_jacobian = evaluation.selected
        analytic_jacobian = evaluation.analytic
        fd_jacobian = evaluation.finite_difference
        analytic_cond = evaluation.analytic_condition
        fd_cond = evaluation.finite_difference_condition
        abs_diff = evaluation.absolute_frobenius_difference
        rel_diff = evaluation.relative_frobenius_difference
        max_error = evaluation.maximum_absolute_element_error
        column_errors = evaluation.column_error_norms
        analytic_ms = 1.0e3 * evaluation.analytic_time_s
        fd_ms = 1.0e3 * evaluation.finite_difference_time_s
        hessian_reused = evaluation.hessian_reused
        diagnostics = evaluation.diagnostics
        predicted_displacement_mm = (
            1.0e3 * float(np.linalg.norm(evaluation.selected @ active_step))
            if np.all(np.isfinite(active_step))
            else np.nan
        )

    return {
        "sample": int(sample),
        "timestamp_utc": _utc_now(),
        "mode": cfg.mode,
        "phase": phase,
        "target_index": int(target_index),
        "iteration": int(iteration),
        "command_sent": int(command_sent),
        "stop_detail": stop_detail,
        "jacobian_source": cfg.jacobian_source,
        "controlled_tip_axes_json": _json_cell(cfg.controlled_tip_axes),
        "active_control_axes_json": _json_cell(cfg.active_control_axes),
        "pre_magnet_p8_json": _json_cell(
            None if pre_state is None else pre_state["p8"]
        ),
        "commanded_magnet_p8_json": _json_cell(commanded_p8),
        "post_magnet_p8_json": _json_cell(
            None if post_state is None else post_state["p8"]
        ),
        "target_tip_m_json": _json_cell(target_tip),
        "pre_measured_tip_m_json": _json_cell(pre_measured_tip),
        "post_measured_tip_m_json": _json_cell(post_measured_tip),
        "model_at_pre_measured_pose_m_json": _json_cell(pre_model_tip),
        "model_at_commanded_pose_m_json": _json_cell(command_tip),
        "model_at_post_measured_pose_m_json": _json_cell(post_model_tip),
        "bias_corrected_model_post_tip_m_json": _json_cell(corrected_model_tip),
        "one_step_jacobian_prediction_m_json": _json_cell(one_step_tip),
        "pre_measured_angle_deg": pre_measured_angle,
        "post_measured_angle_deg": post_measured_angle,
        "model_pre_angle_deg": pre_model_angle,
        "model_command_angle_deg": command_angle,
        "model_post_angle_deg": post_model_angle,
        "bias_corrected_model_post_angle_deg": corrected_model_angle,
        "raw_model_post_position_error_mm": norm_mm(post_measured_tip, post_model_tip),
        "bias_corrected_model_post_position_error_mm": norm_mm(
            post_measured_tip, corrected_model_tip
        ),
        "one_step_prediction_error_mm": norm_mm(post_measured_tip, one_step_tip),
        "target_error_before_mm": norm_mm(
            pre_measured_tip, target_tip, selected=True
        ),
        "target_error_after_mm": norm_mm(
            post_measured_tip, target_tip, selected=True
        ),
        "active_control_step_m_json": _json_cell(active_step),
        "source_translation_step_m_json": _json_cell(translation_step),
        "control_step_norm_mm": 1.0e3 * float(np.linalg.norm(translation_step)),
        "predicted_tip_displacement_mm": predicted_displacement_mm,
        "selected_jacobian_json": _json_cell(selected_jacobian),
        "analytic_jacobian_json": _json_cell(analytic_jacobian),
        "finite_difference_jacobian_json": _json_cell(fd_jacobian),
        "analytic_condition_number": analytic_cond,
        "finite_difference_condition_number": fd_cond,
        "jacobian_absolute_frobenius_difference": abs_diff,
        "jacobian_relative_frobenius_difference": rel_diff,
        "jacobian_maximum_absolute_element_error": max_error,
        "jacobian_column_error_norms_json": _json_cell(column_errors),
        "analytic_jacobian_time_ms": analytic_ms,
        "finite_difference_jacobian_time_ms": fd_ms,
        "hessian_reused": int(hessian_reused),
        "forward_model_jacobian_diagnostics_json": _json_cell(diagnostics),
        "composite_inputs_json": _json_cell(
            composite_metadata["composite_inputs"]
        ),
        "composite_calculated_json": _json_cell(
            composite_metadata["composite_calculated"]
        ),
    }


# =============================================================================
# HARDWARE EXPERIMENT LOOPS
# =============================================================================


def _measure_state(
    cfg: ExperimentConfig,
    hw: Any,
    adapter: DirectForwardModelAdapter,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    overlay_path: Path | None,
) -> dict[str, Any]:
    return _base_module().measure_state(
        cfg=cfg,
        hw=hw,
        mpc=adapter,
        manual_frame=manual_frame,
        roi_polygon=roi_polygon,
        overlay_path=overlay_path,
    )


def _overlay_path(output_dir: Path, sample: int, cfg: ExperimentConfig) -> Path | None:
    if not cfg.save_vision_overlays:
        return None
    directory = output_dir / "vision_overlays"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"sample_{sample:05d}.png"


def _calculate_bias(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "position_m": state["measured_tip"] - state["model_tip"],
        "angle_deg": _wrap_degrees(
            state["measured_angle_deg"] - state["model_angle_deg"]
        ),
    }


def _command_translation(
    cfg: ExperimentConfig,
    hw: Any,
    state: dict[str, Any],
    command_p8: np.ndarray,
) -> None:
    _base_module()._command_translation(
        cfg=cfg,
        hw=hw,
        current_state=state,
        desired_p8=command_p8,
    )


def _translation_and_active_step(
    cfg: ExperimentConfig,
    pre_p8: np.ndarray,
    command_p8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(command_p8[:3] - pre_p8[:3], dtype=float)
    active = []
    for axis in cfg.active_control_axes:
        if axis not in (0, 1, 2):
            raise RuntimeError("Live first-stage commands must remain translational.")
        active.append(translation[axis])
    return translation, np.asarray(active, dtype=float)


def _one_step_tip(
    state: dict[str, Any],
    evaluation: JacobianEvaluation,
    active_step: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    predicted = np.asarray(state["model_tip"], dtype=float).copy()
    rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    predicted[rows] += evaluation.selected @ active_step
    return predicted


def _shadow_state(
    pre_state: dict[str, Any],
    command_p8: np.ndarray,
    model_output: np.ndarray,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    state = dict(pre_state)
    tip, tangent = _output_parts(model_output, np.asarray(cfg.pivot_pose6[:3]))
    state["p8"] = np.asarray(command_p8, dtype=float).copy()
    state["model_output"] = np.asarray(model_output, dtype=float).copy()
    state["model_tip"] = tip
    state["model_tangent"] = tangent
    # Shadow state has no new real measurement.
    state["measured_tip"] = np.full(3, np.nan)
    state["measured_tangent"] = np.full(3, np.nan)
    state["measured_angle_deg"] = np.nan
    state["model_angle_deg"] = _plane_angle_deg(tangent, cfg.angle_plane_axes)
    return state


def run_arc_validation(
    *,
    cfg: ExperimentConfig,
    hw: Any,
    adapter: DirectForwardModelAdapter,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    output_dir: Path,
    logger: CsvRunLogger,
    composite_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = 0
    state = _measure_state(
        cfg,
        hw,
        adapter,
        manual_frame,
        roi_polygon,
        _overlay_path(output_dir, sample, cfg),
    )
    bias = _calculate_bias(state)
    points = make_arc_points(cfg, state["p8"][:3])
    path = [("outbound", i, p) for i, p in enumerate(points[1:], start=1)]
    if cfg.return_along_arc_to_start and cfg.send_commands:
        path.extend(
            ("return", i, p)
            for i, p in reversed(list(enumerate(points[:-1])))
        )

    baseline = make_transition_row(
        sample=sample,
        cfg=cfg,
        phase="initial",
        target_index=0,
        iteration=0,
        pre_state=state,
        post_state=state,
        commanded_p8=state["p8"],
        target_tip=None,
        model_at_command=state["model_output"],
        one_step_tip=state["model_tip"],
        evaluation=None,
        active_step=None,
        translation_step=np.zeros(3),
        bias=bias,
        composite_metadata=composite_metadata,
        command_sent=False,
    )
    logger.append(baseline)
    rows.append(baseline)

    for phase, target_index, xyz in path:
        pre_state = state
        command_p8 = np.asarray(pre_state["p8"], dtype=float).copy()
        command_p8[:3] = xyz
        _workspace_check(cfg, xyz, "arc command")
        translation_step, active_step = _translation_and_active_step(
            cfg, pre_state["p8"], command_p8
        )
        if 1.0e3 * np.linalg.norm(translation_step) > cfg.maximum_magnet_step_mm + 1e-9:
            raise RuntimeError("Actual-to-command arc step exceeds the hard limit.")

        evaluation, nominal_output = evaluate_jacobians(
            adapter, pre_state["p8"], cfg
        )
        pre_state["model_output"] = nominal_output
        pre_tip, pre_tangent = _output_parts(
            nominal_output, np.asarray(cfg.pivot_pose6[:3])
        )
        pre_state["model_tip"] = pre_tip
        pre_state["model_tangent"] = pre_tangent
        predicted_delta = enforce_predicted_displacement_gate(
            evaluation, active_step, cfg
        )
        one_step = pre_tip.copy()
        one_step[np.asarray(cfg.controlled_tip_axes)] += predicted_delta
        model_at_command = adapter.forward_tip_fn(command_p8, commit=False)

        if cfg.send_commands:
            _command_translation(cfg, hw, pre_state, command_p8)
            time.sleep(cfg.settle_time_s)
            sample += 1
            post_state = _measure_state(
                cfg,
                hw,
                adapter,
                manual_frame,
                roi_polygon,
                _overlay_path(output_dir, sample, cfg),
            )
            state = post_state
        else:
            sample += 1
            post_state = None
            state = _shadow_state(pre_state, command_p8, model_at_command, cfg)

        row = make_transition_row(
            sample=sample,
            cfg=cfg,
            phase=phase if cfg.send_commands else "preview",
            target_index=target_index,
            iteration=0,
            pre_state=pre_state,
            post_state=post_state,
            commanded_p8=command_p8,
            target_tip=None,
            model_at_command=model_at_command,
            one_step_tip=one_step,
            evaluation=evaluation,
            active_step=active_step,
            translation_step=translation_step,
            bias=bias,
            composite_metadata=composite_metadata,
            command_sent=cfg.send_commands,
        )
        logger.append(row)
        rows.append(row)
        print(
            f"[{row['phase']} {target_index:03d}] "
            f"step={row['control_step_norm_mm']:.3f} mm, "
            f"cond(Ja)={row['analytic_condition_number']:.3g}, "
            f"J disagreement={row['jacobian_relative_frobenius_difference']:.3g}"
        )
    return rows, bias


def run_point_control(
    *,
    cfg: ExperimentConfig,
    hw: Any,
    adapter: DirectForwardModelAdapter,
    manual_frame: dict[str, Any],
    roi_polygon: Any,
    output_dir: Path,
    logger: CsvRunLogger,
    composite_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample = 0
    state = _measure_state(
        cfg,
        hw,
        adapter,
        manual_frame,
        roi_polygon,
        _overlay_path(output_dir, sample, cfg),
    )
    bias = _calculate_bias(state)
    targets = point_targets_robot_m(cfg)
    controlled_rows = np.asarray(cfg.controlled_tip_axes, dtype=int)
    error_history: list[float] = []

    for target_index, target in enumerate(targets):
        hits = 0
        error_history.clear()
        for iteration in range(cfg.maximum_iterations_per_target):
            error = target[controlled_rows] - state["measured_tip"][controlled_rows]
            error_mm = 1.0e3 * float(np.linalg.norm(error))
            error_history.append(error_mm)
            hits = hits + 1 if error_mm <= cfg.target_tolerance_mm else 0
            if hits >= cfg.required_consecutive_hits:
                print(f"Target {target_index} reached at {error_mm:.3f} mm.")
                break
            if hits > 0:
                time.sleep(cfg.settle_time_s)
                sample += 1
                state = _measure_state(
                    cfg,
                    hw,
                    adapter,
                    manual_frame,
                    roi_polygon,
                    _overlay_path(output_dir, sample, cfg),
                )
                continue

            pre_state = state
            evaluation, nominal_output = evaluate_jacobians(
                adapter, pre_state["p8"], cfg
            )
            pre_tip, pre_tangent = _output_parts(
                nominal_output, np.asarray(cfg.pivot_pose6[:3])
            )
            pre_state["model_output"] = nominal_output
            pre_state["model_tip"] = pre_tip
            pre_state["model_tangent"] = pre_tangent
            active_step = damped_inverse_step(evaluation.selected, error, cfg)
            predicted_delta = enforce_predicted_displacement_gate(
                evaluation, active_step, cfg
            )

            translation_step = np.zeros(3)
            for column_index, control_axis in enumerate(cfg.active_control_axes):
                if control_axis not in (0, 1, 2):
                    raise RuntimeError("Live v2 control is translation-only.")
                translation_step[control_axis] = active_step[column_index]
            command_p8 = np.asarray(pre_state["p8"], dtype=float).copy()
            command_p8[:3] += translation_step
            _workspace_check(cfg, command_p8[:3], "inverse-Jacobian command")
            model_at_command = adapter.forward_tip_fn(command_p8, commit=False)
            one_step = pre_tip.copy()
            one_step[controlled_rows] += predicted_delta

            if cfg.send_commands:
                _command_translation(cfg, hw, pre_state, command_p8)
                time.sleep(cfg.settle_time_s)
                sample += 1
                post_state = _measure_state(
                    cfg,
                    hw,
                    adapter,
                    manual_frame,
                    roi_polygon,
                    _overlay_path(output_dir, sample, cfg),
                )
                state = post_state
            else:
                post_state = None

            row = make_transition_row(
                sample=sample,
                cfg=cfg,
                phase="control" if cfg.send_commands else "preview",
                target_index=target_index,
                iteration=iteration,
                pre_state=pre_state,
                post_state=post_state,
                commanded_p8=command_p8,
                target_tip=target,
                model_at_command=model_at_command,
                one_step_tip=one_step,
                evaluation=evaluation,
                active_step=active_step,
                translation_step=translation_step,
                bias=bias,
                composite_metadata=composite_metadata,
                command_sent=cfg.send_commands,
            )
            logger.append(row)
            rows.append(row)
            print(
                f"[target {target_index:02d}, iter {iteration:02d}] "
                f"error={error_mm:.3f} mm, "
                f"step={row['control_step_norm_mm']:.3f} mm"
            )
            if not cfg.send_commands:
                return rows, bias

            window = cfg.stall_window_iterations
            if len(error_history) >= window:
                progress = error_history[-window] - error_history[-1]
                if progress < cfg.minimum_progress_over_stall_window_mm:
                    raise RuntimeError(
                        f"Controller stalled: {progress:.3f} mm progress over "
                        f"{window} iterations."
                    )
        else:
            raise RuntimeError(
                f"Target {target_index} not reached within the iteration limit."
            )
    return rows, bias


# =============================================================================
# OUTPUTS AND ENTRY POINT
# =============================================================================


def _finite_column(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)


def save_plot(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = _finite_column(rows, "sample")
    target_before = _finite_column(rows, "target_error_before_mm")
    target_after = _finite_column(rows, "target_error_after_mm")
    model_error = _finite_column(rows, "raw_model_post_position_error_mm")
    prediction_error = _finite_column(rows, "one_step_prediction_error_mm")
    analytic_cond = _finite_column(rows, "analytic_condition_number")
    fd_cond = _finite_column(rows, "finite_difference_condition_number")
    disagreement = _finite_column(rows, "jacobian_relative_frobenius_difference")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(samples, target_before, "o--", label="target before")
    axes[0].plot(samples, target_after, "o-", label="target after")
    axes[0].plot(samples, model_error, ".-", label="model error")
    axes[0].plot(samples, prediction_error, ".-", label="one-step error")
    axes[0].set_ylabel("error (mm)")
    axes[0].set_xlabel("sample")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(samples, analytic_cond, "o-", label="analytic")
    axes[1].semilogy(samples, fd_cond, ".-", label="finite difference")
    axes[1].set_ylabel("Jacobian condition number")
    axes[1].set_xlabel("sample")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(samples, disagreement, "o-")
    axes[2].set_ylabel("relative analytic/FD difference")
    axes[2].set_xlabel("sample")
    axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def save_summary(
    rows: list[dict[str, Any]],
    path: Path,
    cfg: ExperimentConfig,
    composite_metadata: dict[str, Any],
    bias: dict[str, Any],
    stop_reason: str,
) -> None:
    summary = {
        "stop_reason": stop_reason,
        "samples_logged": len(rows),
        "jacobian_source": cfg.jacobian_source,
        "p7_mapping": composite_metadata.get("p7_mapping"),
        "composite_inputs": composite_metadata.get("composite_inputs"),
        "composite_calculated": composite_metadata.get("composite_calculated"),
        "initial_position_bias_m": bias.get("position_m"),
        "initial_angle_bias_deg": bias.get("angle_deg"),
        "one_step_prediction_rmse_mm": _rmse(
            _finite_column(rows, "one_step_prediction_error_mm")
        ),
        "model_position_rmse_mm": _rmse(
            _finite_column(rows, "raw_model_post_position_error_mm")
        ),
        "jacobian_relative_difference_rmse": _rmse(
            _finite_column(rows, "jacobian_relative_frobenius_difference")
        ),
        "physical_verification_assumptions": [
            "The sign and magnitude of z_offset_m in the existing p8 helper match the lab setup.",
            "The raw forward model accepts p7=[xyz, rotation-vector, insertion].",
            "The model callable returns tip XYZ and preferably tip tangent XYZ.",
            "Source translation columns 0/1 correspond to the live robot XY command.",
            "The configured composite properties and magnetisation-axis sign match the specimen.",
            "LiveHardwareController.send_step uses the same p8 convention as the optimized runner.",
        ],
    }
    path.write_text(
        json.dumps(summary, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def make_output_directory(
    cfg: ExperimentConfig,
    composite_metadata: dict[str, Any] | None = None,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(cfg.output_root) / f"{cfg.run_name}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    payload = {
        "experiment": asdict(cfg),
        "composite_inputs": asdict(cfg.composite),
        "composite_calculated": (
            None
            if composite_metadata is None
            else composite_metadata.get("composite_calculated")
        ),
    }
    (path / "config.json").write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    cfg = CONFIG
    validate_config(cfg)
    base = _base_module()
    base.configure_bounds_beam_paths(cfg)
    hw = None
    logger = None
    output_dir: Path | None = None
    rows: list[dict[str, Any]] = []
    bias: dict[str, Any] = {}
    composite_metadata: dict[str, Any] = {
        "composite_inputs": asdict(cfg.composite),
        "composite_calculated": {},
    }
    stop_reason = "completed"

    try:
        hw = base.connect_hardware(cfg)
        robot_pose6 = hw.get_robot_pose_once()
        if robot_pose6 is None:
            raise RuntimeError("Could not read the initial UR robot pose.")
        robot_pose6 = _finite_vector(robot_pose6, 6, "initial robot pose6")
        base.request_live_confirmation(cfg, robot_pose6)

        adapter, composite_metadata = build_direct_model(cfg, robot_pose6)
        output_dir = make_output_directory(cfg, composite_metadata)
        logger = CsvRunLogger(output_dir / "measurements.csv")
        manual_frame = base.load_manual_vessel_boundaries_with_frame(
            cfg.manual_boundary_path
        )
        roi_polygon = base.load_polygon(cfg.roi_polygon_path)

        if cfg.mode == "arc_validation":
            rows, bias = run_arc_validation(
                cfg=cfg,
                hw=hw,
                adapter=adapter,
                manual_frame=manual_frame,
                roi_polygon=roi_polygon,
                output_dir=output_dir,
                logger=logger,
                composite_metadata=composite_metadata,
            )
        else:
            rows, bias = run_point_control(
                cfg=cfg,
                hw=hw,
                adapter=adapter,
                manual_frame=manual_frame,
                roi_polygon=roi_polygon,
                output_dir=output_dir,
                logger=logger,
                composite_metadata=composite_metadata,
            )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        if hw is not None:
            base._safe_stop(hw)
        print("Stopped by user; no automatic recovery movement was attempted.")
    except Exception as exc:
        stop_reason = f"exception:{type(exc).__name__}: {exc}"
        if hw is not None:
            base._safe_stop(hw)
        print(f"Stopped safely: {stop_reason}")
        raise
    finally:
        if logger is not None:
            logger.close()
        if hw is not None:
            base._safe_stop(hw)
        if output_dir is not None:
            save_plot(rows, output_dir / "comparison.png")
            save_summary(
                rows,
                output_dir / "summary.json",
                cfg,
                composite_metadata,
                bias,
                stop_reason,
            )
            print(f"CSV:     {output_dir / 'measurements.csv'}")
            print(f"Plot:    {output_dir / 'comparison.png'}")
            print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
