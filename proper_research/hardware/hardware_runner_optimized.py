from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from proper_research.control.lab_ready_mpc import (
    build_measured_p8_from_pose6_and_length,
    dipole_dir_from_p8,
    plot_reference_debug,
    vision_result_to_x_meas_robot,
)
from proper_research.simulation.simulations.curvature_utils import (
    compute_discrete_path_curvature,
    curvature_ahead_metrics,
)
from proper_research.simulation.simulations.logging_utils import (
    compute_centreline_metrics,
)
from proper_research.vision.bounds_beam import load_polygon, reconstruct_beam_within_vessel
from proper_research.vision.detect_blue import load_manual_vessel_boundaries_with_frame
from proper_research.vision.measure_length import new_capture

from .publication_metrics import (
    PreviousHardwarePlan,
    PublicationRunLogger,
    compute_hardware_transition_metrics,
    current_tracking_metrics,
    jacobian_matrix_metrics,
    model_linearisation_metrics,
    vector_components,
)


@dataclass(frozen=True)
class HardwareRunConfig:
    """Fixed-mode hardware MPC configuration with publication logging."""

    max_steps: int = 100
    solver_mode: str = "lti"
    rollout_steps: int = 1
    send_commands: bool = False

    image_filename: str = "focused_image.jpg"
    roi_polygon_path: str = "/home/jack/Proper-Research/custom_area.json"
    manual_boundary_path: str = "/home/jack/Proper-Research/manual_vessel_boundaries.json"
    blue_roi_path: str = "blue_roi_box.json"
    green_roi_path: str = "green_roi_box.json"
    pivot_hint: tuple[int, int] = (309, 330)
    show_vision: bool = False
    save_vision_overlays: bool = False
    save_controller_plots: bool = False

    output_dir: str = "hardware_publication_runs"
    run_name: str = "hardware_optimized"
    fsync_log_each_step: bool = True
    verbose: bool = True

    # Vision-derived insertion-length correction.
    min_valid_length_m: float = 0.005
    max_valid_length_m: float = 0.080
    min_estimated_length_m: float = 0.010
    max_estimated_length_m: float = 0.050
    max_length_correction_per_frame_m: float = 0.0003
    length_correction_alpha: float = 1.0

    # Hardware command design limits. Controls are rates; these are per-frame steps.
    max_translation_step_mm: float = 4.0
    max_rotation_step_rad: float = 0.04
    max_insertion_step_mm: float = 1.2
    disable_z_translation: bool = True
    disable_x_rotation: bool = True
    disable_y_rotation: bool = True

    # Independent final safety gate.
    hard_max_translation_step_mm: float = 6.0
    hard_max_rotation_step_rad: float = 0.06
    hard_max_insertion_step_mm: float = 1.8
    clearance_stop_mm: float = -0.5
    enable_tip_angle_stop: bool = True
    tip_angle_stop_deg: float = 40.0
    tip_angle_activation_clearance_mm: float = 1.0
    z_bound_margin_m: float = 0.003

    use_output_bias: bool = True
    collect_model_bias: bool = True
    curvature_lookahead_points: int = 10
    stop_on_infeasible: bool = True

    def validate(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if self.solver_mode not in {"lti", "ltv_oneshot", "sqp_full"}:
            raise ValueError(
                "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
            )
        if int(self.rollout_steps) != 1:
            raise ValueError(
                "The hardware runner requires rollout_steps=1 so every command "
                "is followed by a new camera measurement and publication metric."
            )
        if self.send_commands and self.hard_max_translation_step_mm <= 0.0:
            raise ValueError("Hard command limits must be positive in live mode.")
        if self.max_length_correction_per_frame_m <= 0.0:
            raise ValueError("max_length_correction_per_frame_m must be positive.")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _safe_array(value: Any, *, size: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if size is not None and array.size != size:
        return None
    return array


def _first_stage_matrix(mpc, info: dict[str, Any]) -> np.ndarray | None:
    direct = info.get("B_first")
    if direct is not None:
        B = np.asarray(direct, dtype=float)
        if B.ndim == 2:
            return B.copy()
    sequence = getattr(mpc, "_last_B_sequence_linearisation", None)
    if sequence is None:
        return None
    array = np.asarray(sequence, dtype=float)
    if array.ndim != 3 or array.shape[0] == 0:
        return None
    return array[0].copy()


def _first_prediction(info: dict[str, Any], n_out: int) -> np.ndarray | None:
    value = info.get("X_pred")
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size < n_out:
        return None
    return array.reshape(-1, n_out)[0].copy()


def _first_nonlinear_rollout(info: dict[str, Any], n_out: int) -> np.ndarray | None:
    value = info.get("x_rollout")
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size < n_out:
        return None
    return array.reshape(-1, n_out)[0].copy()


def _reference_from_info(mpc, info: dict[str, Any]) -> dict[str, Any]:
    """Return the exact first reference used by the controller objective.

    In contouring mode the first reference is generally between sampled lumen
    nodes, so reconstructing it from ``idx_ref`` is incorrect. Prefer the frozen
    continuous C_ref/T_ref/s_ref stored in ``info`` and fall back to the active
    controller reference. ``idx_ref`` is retained only as a legacy/debug index.
    """
    indices = np.asarray(info.get("idx_ref", []), dtype=int).reshape(-1)
    index = int(indices[0]) if indices.size else -1
    mode = str(
        info.get("reference_mode", getattr(mpc, "reference_mode", "point"))
    ).strip().lower()

    point = None
    tangent = None
    s0 = np.nan
    progress = _safe_float(
        info.get(
            "path_progress_current_m",
            info.get(
                "progress_current_m",
                getattr(mpc, "path_progress_s", np.nan),
            ),
        )
    )

    C_ref = info.get("path_ref_points")
    T_ref = info.get("path_ref_tangents")
    s_ref = info.get("path_ref_s")

    if C_ref is not None:
        array = np.asarray(C_ref, dtype=float)
        if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] >= 3:
            point = array[0, :3].copy()
    if T_ref is not None:
        array = np.asarray(T_ref, dtype=float)
        if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] >= 3:
            tangent = array[0, :3].copy()
    if s_ref is not None:
        array = np.asarray(s_ref, dtype=float).reshape(-1)
        if array.size:
            s0 = _safe_float(array[0])

    active = getattr(mpc, "active_path_reference", None)
    if isinstance(active, dict):
        if point is None and active.get("C_ref") is not None:
            array = np.asarray(active["C_ref"], dtype=float)
            if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] >= 3:
                point = array[0, :3].copy()
        if tangent is None and active.get("T_ref") is not None:
            array = np.asarray(active["T_ref"], dtype=float)
            if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] >= 3:
                tangent = array[0, :3].copy()
        if not np.isfinite(s0) and active.get("s_ref") is not None:
            array = np.asarray(active["s_ref"], dtype=float).reshape(-1)
            if array.size:
                s0 = _safe_float(array[0])
        if not np.isfinite(progress):
            progress = _safe_float(active.get("s_progress"))

    # Legacy point-mode fallback. Never replace a valid continuous reference.
    if point is None and index >= 0:
        C = np.asarray(mpc.lumen_C, dtype=float)
        if C.ndim == 2 and index < C.shape[0] and C.shape[1] >= 3:
            point = C[index, :3].copy()

    return {
        "mode": mode,
        "index": index,
        "point": point,
        "tangent": tangent,
        "s0_m": float(s0),
        "progress_m": float(progress),
    }


def _extract_vision_metrics(vision_result: dict[str, Any]) -> dict[str, Any]:
    distance = dict(vision_result.get("tip_distance_info_mm", {}) or {})
    angle = dict(vision_result.get("tip_wall_angle_info", {}) or {})
    left_tangent = _safe_array(
        angle.get("left_wall_tangent_vec_cartesian"), size=2
    )
    right_tangent = _safe_array(
        angle.get("right_wall_tangent_vec_cartesian"), size=2
    )
    if left_tangent is None:
        left_tangent = np.full(2, np.nan)
    if right_tangent is None:
        right_tangent = np.full(2, np.nan)
    return {
        "vision_beam_length_mm": _safe_float(vision_result.get("beam_length_mm")),
        "vision_dist_left_center_mm": _safe_float(distance.get("dist_left_center_mm")),
        "vision_dist_right_center_mm": _safe_float(distance.get("dist_right_center_mm")),
        "vision_dist_left_edge_mm": _safe_float(distance.get("dist_left_edge_mm")),
        "vision_dist_right_edge_mm": _safe_float(distance.get("dist_right_edge_mm")),
        "vision_closest_distance_mm": _safe_float(distance.get("closest_distance_mm")),
        "vision_beam_left_wall_angle_deg": _safe_float(
            angle.get("beam_left_wall_tangent_angle_deg")
        ),
        "vision_beam_right_wall_angle_deg": _safe_float(
            angle.get("beam_right_wall_tangent_angle_deg")
        ),
        "vision_left_wall_tangent_x": float(left_tangent[0]),
        "vision_left_wall_tangent_y": float(left_tangent[1]),
        "vision_right_wall_tangent_x": float(right_tangent[0]),
        "vision_right_wall_tangent_y": float(right_tangent[1]),
    }


def _estimate_insertion_length(
    *,
    model_length_m: float,
    vision_length_mm: Any,
    config: HardwareRunConfig,
) -> tuple[float, float, str]:
    raw = _safe_float(vision_length_mm) / 1.0e3
    if (
        not np.isfinite(raw)
        or raw < config.min_valid_length_m
        or raw > config.max_valid_length_m
    ):
        return float(model_length_m), raw, "model_only_invalid_vision"
    error = raw - float(model_length_m)
    correction = float(
        np.clip(
            error,
            -config.max_length_correction_per_frame_m,
            config.max_length_correction_per_frame_m,
        )
    )
    estimated = float(model_length_m) + config.length_correction_alpha * correction
    estimated = float(
        np.clip(
            estimated,
            config.min_estimated_length_m,
            config.max_estimated_length_m,
        )
    )
    return estimated, raw, "model_plus_limited_vision"


def _set_live_control_limits(mpc, config: HardwareRunConfig) -> np.ndarray:
    dt = float(mpc.dt)
    if dt <= 0.0:
        raise ValueError("mpc.dt must be positive.")
    limits = np.array(
        [
            config.max_translation_step_mm / 1.0e3 / dt,
            config.max_translation_step_mm / 1.0e3 / dt,
            config.max_translation_step_mm / 1.0e3 / dt,
            config.max_rotation_step_rad / dt,
            config.max_rotation_step_rad / dt,
            config.max_rotation_step_rad / dt,
            config.max_insertion_step_mm / 1.0e3 / dt,
        ],
        dtype=float,
    )
    if config.disable_z_translation:
        limits[2] = 0.0
    if config.disable_x_rotation:
        limits[3] = 0.0
    if config.disable_y_rotation:
        limits[4] = 0.0
    mpc.u_max = limits.copy()
    return limits


def _update_fixed_lumen_once(mpc, lumen_C: np.ndarray, lumen_R: np.ndarray) -> None:
    """
    Set fixed lumen geometry once. Unlike the old runner, this does not reset
    forward/Jacobian caches on every camera frame.
    """
    mpc.lumen_C = np.asarray(lumen_C, dtype=float)[:, :3].copy()
    mpc.lumen_R = np.asarray(lumen_R, dtype=float).reshape(-1).copy()
    if hasattr(mpc, "reset_path_reference_state"):
        mpc.reset_path_reference_state()
    seen: set[int] = set()
    for wrapper in (getattr(mpc, "forward_tip_fn", None), getattr(mpc, "Jxy_fn", None)):
        if wrapper is None:
            continue
        model = getattr(wrapper, "model", getattr(wrapper, "fwd", wrapper))
        if id(model) in seen:
            continue
        seen.add(id(model))
        if hasattr(model, "set_lumen"):
            model.set_lumen(mpc.lumen_C, mpc.lumen_R, reset_cache=True)


def _safe_stop(hw) -> None:
    if hw is None:
        return
    try:
        hw.stop()
    except Exception:
        pass


def _controller_performance_fields(mpc, info: dict[str, Any]) -> dict[str, Any]:
    sqp_hist = list(info.get("sqp_hist", []) or [])
    last_sqp = sqp_hist[-1] if sqp_hist else {}
    plant_model = getattr(getattr(mpc, "forward_tip_fn", None), "model", None)
    jacobian_model = getattr(getattr(mpc, "Jxy_fn", None), "model", None)
    plant_info = dict(getattr(plant_model, "last_info", {}) or {})
    jac_diag = (
        dict(jacobian_model.get_last_diag())
        if jacobian_model is not None and hasattr(jacobian_model, "get_last_diag")
        else {}
    )
    return {
        "status": str(info.get("status", "unknown")),
        "infeasible": int(info.get("infeasible", 0)),
        "solver_mode_used": str(info.get("solver_mode_used", info.get("solver_mode", ""))),
        "Np": int(getattr(mpc, "Np", -1)),
        "N_sqp": int(getattr(mpc, "N_sqp", -1)),
        "sqp_iterations": len(sqp_hist),
        "sqp_converged": int(info.get("sqp_converged", 0)),
        "sqp_stop_reason": str(info.get("sqp_stop_reason", "")),
        "sqp_step_scaled_inf": _safe_float(info.get("sqp_step_scaled_inf")),
        "sqp_step_threshold": _safe_float(info.get("sqp_step_threshold")),
        "controller_solve_ms": 1.0e3 * _safe_float(info.get("controller_solve_wall_s"), 0.0),
        "controller_rollout_ms": 1.0e3 * _safe_float(info.get("controller_rollout_wall_s"), 0.0),
        "controller_reported_step_ms": 1.0e3 * _safe_float(info.get("controller_step_wall_s"), 0.0),
        "qp_build_ms": 1.0e3 * _safe_float(info.get("qp_build_time_total_s"), 0.0),
        "osqp_ms": 1.0e3 * _safe_float(info.get("osqp_time_total_s"), 0.0),
        "candidate_validation_ms": 1.0e3 * _safe_float(
            info.get("candidate_validation_time_total_s"), 0.0
        ),
        "forward_calls": int(info.get("forward_calls_step", 0)),
        "jacobian_calls": int(info.get("jacobian_calls_step", 0)),
        "forward_time_ms": 1.0e3 * _safe_float(info.get("forward_time_step_s"), 0.0),
        "jacobian_time_ms": 1.0e3 * _safe_float(info.get("jacobian_time_step_s"), 0.0),
        "osqp_iterations": int(last_sqp.get("osqp_iter", -1)),
        "osqp_setup_performed": int(last_sqp.get("osqp_setup_performed", 0)),
        "persistent_osqp_setup_count": int(info.get("persistent_osqp_setup_count", -1)),
        "persistent_osqp_update_count": int(info.get("persistent_osqp_update_count", -1)),
        "cond_H_beam": _safe_float(info.get("cond_H_beam", jac_diag.get("cond_H_beam"))),
        "hessian_reused": int(bool(info.get("hessian_reused", jac_diag.get("hessian_reused", False)))),
        "sensitivity_workers": int(jac_diag.get("workers", getattr(jacobian_model, "sensitivity_workers", 1))),
        "sensitivity_ms": 1.0e3 * _safe_float(jac_diag.get("sensitivity_time_s"), 0.0),
        "hessian_ms": 1.0e3 * _safe_float(jac_diag.get("hessian_time_s"), 0.0),
        "gtheta_ms": 1.0e3 * _safe_float(jac_diag.get("gtheta_time_s"), 0.0),
        "kinematics_sensitivity_ms": 1.0e3 * _safe_float(
            jac_diag.get("kinematics_sensitivity_time_s"), 0.0
        ),
        "linear_solve_ms": 1.0e3 * _safe_float(jac_diag.get("linear_solve_time_s"), 0.0),
        "reference_mode": str(
            info.get("reference_mode", getattr(mpc, "reference_mode", "point"))
        ),
        "path_progress_current_m": _safe_float(
            info.get(
                "path_progress_current_m",
                info.get("progress_current_m", getattr(mpc, "path_progress_s", np.nan)),
            )
        ),
        "path_projection_distance_mm": 1.0e3 * _safe_float(
            info.get("path_projection_distance_m")
        ),
        "progress_constraint_mode": str(
            info.get(
                "progress_constraint_mode",
                getattr(mpc, "progress_constraint_mode", "none"),
            )
        ),
        "progress_request_m": _safe_float(
            info.get("progress_request_m", getattr(mpc, "progress_request_m", np.nan))
        ),
        "progress_target_m": _safe_float(info.get("progress_target_m")),
        "progress_nominal_terminal_m": _safe_float(
            info.get("progress_nominal_terminal_m")
        ),
        "progress_nominal_shortfall_mm": _safe_float(
            info.get("progress_nominal_shortfall_mm")
        ),
        "progress_slack_solution_fraction": _safe_float(
            info.get("progress_slack_solution_fraction")
        ),
        "qp_control_dim": int(info.get("qp_control_dim", int(mpc.Np) * int(mpc.m))),
        "qp_decision_dim": int(info.get("qp_decision_dim", int(mpc.Np) * int(mpc.m))),
        "forward_solve_path": str(plant_info.get("solve_path", "")),
        "forward_cache_hit": int(bool(plant_info.get("cache_hit", False))),
        "forward_iterations": int(plant_info.get("work_total_nit", plant_info.get("total_nit", plant_info.get("nit", 0)))),
        "forward_objective_calls": int(
            plant_info.get("work_total_objective_calls", plant_info.get("total_objective_calls", plant_info.get("objective_calls", 0)))
        ),
        "forward_gradient_calls": int(
            plant_info.get("work_total_gradient_calls", plant_info.get("total_gradient_calls", plant_info.get("gradient_calls", 0)))
        ),
    }


def run_hardware_control_optimized(
    *,
    mpc,
    pivot_point: np.ndarray,
    lumen_C_robot_m: np.ndarray,
    lumen_R_robot_m: np.ndarray,
    config: HardwareRunConfig,
    hw=None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run fixed-mode optimized MPC against camera/robot measurements.

    No hierarchical controller, adaptive solver switching, buffered open-loop
    commands or dynamic node upgrades are used. Each controller command is
    followed by a new camera measurement, which permits one-step Jacobian and
    prediction validation.
    """
    config.validate()
    if config.send_commands and hw is None:
        raise ValueError("Live command mode requires a hardware controller object.")

    pivot_point = np.asarray(pivot_point, dtype=float).reshape(6)
    lumen_C = np.asarray(lumen_C_robot_m, dtype=float)
    lumen_R = np.asarray(lumen_R_robot_m, dtype=float).reshape(-1)
    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(f"lumen_C_robot_m has invalid shape {lumen_C.shape}.")
    lumen_C = lumen_C[:, :3].copy()
    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError("Lumen centreline and radius lengths do not match.")

    output_dir = Path(config.output_dir)
    overlay_dir = output_dir / f"{config.run_name}_vision_overlays"
    plot_dir = output_dir / f"{config.run_name}_controller_plots"
    if config.save_vision_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if config.save_controller_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "runner": "run_hardware_control_optimized",
        "config": asdict(config),
        "pivot_point": pivot_point,
        "lumen_points": int(lumen_C.shape[0]),
        "lumen_radius_min_mm": 1.0e3 * float(np.min(lumen_R)),
        "lumen_radius_max_mm": 1.0e3 * float(np.max(lumen_R)),
        "controller_type": type(mpc).__name__,
        "forward_model_type": type(getattr(getattr(mpc, "forward_tip_fn", None), "model", None)).__name__,
        "jacobian_model_type": type(getattr(getattr(mpc, "Jxy_fn", None), "model", None)).__name__,
        "controller_reference_and_objective": {
            "reference_mode": str(getattr(mpc, "reference_mode", "point")),
            "ref_lookahead_m": _safe_float(getattr(mpc, "ref_lookahead_m", np.nan)),
            "ref_stride_m": _safe_float(getattr(mpc, "ref_stride_m", np.nan)),
            "ref_search_backward_m": _safe_float(getattr(mpc, "ref_search_backward_m", np.nan)),
            "ref_search_forward_m": _safe_float(getattr(mpc, "ref_search_forward_m", np.nan)),
            "ref_max_progress_per_step_m": _safe_float(
                getattr(mpc, "ref_max_progress_per_step_m", np.nan)
            ),
            "q_contour": _safe_float(getattr(mpc, "q_contour", np.nan)),
            "q_lag": _safe_float(getattr(mpc, "q_lag", np.nan)),
            "contouring_axis_weights": np.asarray(
                getattr(mpc, "contouring_axis_weights", np.ones(3)), dtype=float
            ),
            "ref_stage_weights": np.asarray(
                getattr(mpc, "ref_stage_weights", np.ones(int(mpc.Np))), dtype=float
            ),
            "progress_constraint_mode": str(
                getattr(mpc, "progress_constraint_mode", None)
            ),
            "progress_request_m": _safe_float(
                getattr(mpc, "progress_request_m", np.nan)
            ),
            "progress_reward_weight": _safe_float(
                getattr(mpc, "progress_reward_weight", np.nan)
            ),
            "progress_slack_quadratic_weight": _safe_float(
                getattr(mpc, "progress_slack_quadratic_weight", np.nan)
            ),
            "progress_slack_linear_weight": _safe_float(
                getattr(mpc, "progress_slack_linear_weight", np.nan)
            ),
            "progress_slack_max_fraction": _safe_float(
                getattr(mpc, "progress_slack_max_fraction", np.nan)
            ),
        },
    }
    if run_metadata:
        metadata.update(run_metadata)
    logger = PublicationRunLogger(
        output_dir,
        run_name=config.run_name,
        metadata=metadata,
        fsync_each_step=config.fsync_log_each_step,
    )

    manual = load_manual_vessel_boundaries_with_frame(config.manual_boundary_path)
    roi_polygon = load_polygon(config.roi_polygon_path)
    path_curvature = compute_discrete_path_curvature(lumen_C)
    _update_fixed_lumen_once(mpc, lumen_C, lumen_R)

    previous_plan: PreviousHardwarePlan | None = None
    history: list[dict[str, Any]] = []
    stop_reason = "max_steps"
    run_started = time.perf_counter()

    try:
        for step in range(int(config.max_steps)):
            loop_started = time.perf_counter()
            utc_now = datetime.now(timezone.utc).isoformat()

            capture_started = time.perf_counter()
            new_capture()
            capture_ms = 1.0e3 * (time.perf_counter() - capture_started)

            overlay_path = (
                str(overlay_dir / f"step_{step:05d}.png")
                if config.save_vision_overlays
                else None
            )
            vision_started = time.perf_counter()
            vision_result = reconstruct_beam_within_vessel(
                image_filename=config.image_filename,
                red_roi_polygon=roi_polygon,
                blue_roi_path=config.blue_roi_path,
                green_roi_path=config.green_roi_path,
                pivot_hint=tuple(config.pivot_hint),
                show=bool(config.show_vision),
                save_overlay_path=overlay_path,
                base_px_ref=manual["base_px"],
                ex_ref=manual["ex_img"],
                ey_ref=manual["ey_img"],
            )
            vision_ms = 1.0e3 * (time.perf_counter() - vision_started)
            vision_result["base_px_ref"] = manual["base_px"]
            vision_result["ex_ref"] = manual["ex_img"]
            vision_result["ey_ref"] = manual["ey_img"]

            x_meas = np.asarray(
                vision_result_to_x_meas_robot(
                    vision_result,
                    pivot_point_pose6=pivot_point,
                ),
                dtype=float,
            ).reshape(-1)
            if x_meas.size < int(mpc.n):
                raise ValueError(
                    f"Vision measurement has {x_meas.size} values; controller requires {mpc.n}."
                )
            x_mpc = x_meas[: int(mpc.n)].copy()
            measured_tip = x_meas[:3].copy()
            measured_tangent = x_meas[3:6].copy() if x_meas.size >= 6 else None

            robot_read_started = time.perf_counter()
            robot_pose6 = None
            if hw is not None:
                robot_pose6 = hw.get_robot_pose_once()
                if robot_pose6 is None:
                    raise RuntimeError("Hardware returned no robot pose.")
                robot_pose6 = np.asarray(robot_pose6, dtype=float).reshape(6)
            robot_read_ms = 1.0e3 * (time.perf_counter() - robot_read_started)
            measurement_monotonic_s = time.perf_counter()

            transition_metrics = compute_hardware_transition_metrics(
                previous_plan,
                x_meas_now=x_meas,
                robot_pose6_now=robot_pose6,
                monotonic_now_s=measurement_monotonic_s,
            )

            model_length = float(np.asarray(mpc.p, dtype=float).reshape(-1)[7])
            length_estimate, length_vision_raw, length_source = _estimate_insertion_length(
                model_length_m=model_length,
                vision_length_mm=vision_result.get("beam_length_mm"),
                config=config,
            )
            if robot_pose6 is not None:
                p_measured = build_measured_p8_from_pose6_and_length(
                    robot_pose6,
                    length_estimate,
                    z_offset=float(getattr(hw, "z_offset", 0.0)),
                )
                p_measured = np.asarray(p_measured, dtype=float).reshape(int(mpc.np))
                mpc.set_measured_params(p_measured)
            else:
                p_measured = np.asarray(mpc.p, dtype=float).reshape(int(mpc.np)).copy()

            if hasattr(mpc, "p_min") and hasattr(mpc, "p_max"):
                mpc.p_min = np.asarray(mpc.p_min, dtype=float).reshape(int(mpc.np))
                mpc.p_max = np.asarray(mpc.p_max, dtype=float).reshape(int(mpc.np))
                mpc.p_min[2] = float(p_measured[2]) - config.z_bound_margin_m
                mpc.p_max[2] = float(p_measured[2]) + config.z_bound_margin_m
            live_u_max = _set_live_control_limits(mpc, config)

            model_eval_started = time.perf_counter()
            x_model0 = None
            if config.collect_model_bias or config.use_output_bias:
                try:
                    x_model0 = np.asarray(
                        mpc.forward_tip_fn(p_measured, commit=False),
                        dtype=float,
                    ).reshape(-1)[: int(mpc.n)]
                except TypeError:
                    x_model0 = np.asarray(
                        mpc.forward_tip_fn(p_measured), dtype=float
                    ).reshape(-1)[: int(mpc.n)]
            model_bias_eval_ms = 1.0e3 * (time.perf_counter() - model_eval_started)
            if config.use_output_bias and x_model0 is not None and hasattr(mpc, "set_output_bias"):
                mpc.set_output_bias(x_mpc - x_model0)

            centreline_metrics = compute_centreline_metrics(
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                tip_pos=measured_tip,
                tip_tan=measured_tangent,
            )
            curvature_metrics = curvature_ahead_metrics(
                path=lumen_C,
                curvature=path_curvature,
                tip_pos=measured_tip,
                lookahead_points=int(config.curvature_lookahead_points),
            )

            controller_started = time.perf_counter()
            p_commanded, x_model_post, info = mpc.step(
                x_meas=x_mpc,
                rollout_steps=1,
                solver_mode=config.solver_mode,
            )
            controller_measured_ms = 1.0e3 * (
                time.perf_counter() - controller_started
            )
            info = dict(info or {})
            p_commanded = np.asarray(p_commanded, dtype=float).reshape(int(mpc.np))
            u0 = np.asarray(
                info.get("u0", np.full(int(mpc.m), np.nan)), dtype=float
            ).reshape(int(mpc.m))
            B_first = _first_stage_matrix(mpc, info)
            x_pred0 = _first_prediction(info, int(mpc.n))
            x_nonlinear_next = _first_nonlinear_rollout(info, int(mpc.n))
            reference = _reference_from_info(mpc, info)
            reference_index = int(reference["index"])
            reference0 = reference["point"]
            reference_tangent0 = reference["tangent"]
            reference_s0_m = float(reference["s0_m"])
            path_progress_m = float(reference["progress_m"])

            tracking_metrics = current_tracking_metrics(
                x_meas=x_mpc,
                reference0=reference0,
                x_pred0=x_pred0,
                reference_tangent0=reference_tangent0,
            )
            matrix_metrics = jacobian_matrix_metrics(B_first)
            model_metrics = model_linearisation_metrics(
                x_meas=x_mpc,
                x_model0=x_model0,
                x_nonlinear_next=x_nonlinear_next,
                x_pred0=x_pred0,
                B_first=B_first,
                u0=u0,
            )
            performance = _controller_performance_fields(mpc, info)

            dt = float(mpc.dt)
            translation_step_mm = 1.0e3 * float(np.linalg.norm(u0[:3] * dt))
            rotation_step_rad = float(np.linalg.norm(u0[3:6] * dt))
            insertion_step_mm = 1.0e3 * float(u0[6] * dt) if u0.size > 6 else np.nan

            hardware_stop = False
            hardware_stop_reason = ""
            if not np.all(np.isfinite(u0)) or not np.all(np.isfinite(p_commanded)):
                hardware_stop = True
                hardware_stop_reason = "nonfinite_command"
            elif config.stop_on_infeasible and int(info.get("infeasible", 0)):
                hardware_stop = True
                hardware_stop_reason = "controller_infeasible"
            elif translation_step_mm > config.hard_max_translation_step_mm:
                hardware_stop = True
                hardware_stop_reason = "translation_step_too_large"
            elif rotation_step_rad > config.hard_max_rotation_step_rad:
                hardware_stop = True
                hardware_stop_reason = "rotation_step_too_large"
            elif np.isfinite(insertion_step_mm) and abs(insertion_step_mm) > config.hard_max_insertion_step_mm:
                hardware_stop = True
                hardware_stop_reason = "insertion_step_too_large"

            clearance_mm = _safe_float(centreline_metrics.get("clearance_mm"))
            tip_angle_deg = _safe_float(
                centreline_metrics.get("tip_vessel_angle_deg")
            )
            if (
                not hardware_stop
                and np.isfinite(clearance_mm)
                and clearance_mm < config.clearance_stop_mm
            ):
                hardware_stop = True
                hardware_stop_reason = "clearance_stop"
            if (
                not hardware_stop
                and config.enable_tip_angle_stop
                and np.isfinite(clearance_mm)
                and clearance_mm <= config.tip_angle_activation_clearance_mm
                and (
                    not np.isfinite(tip_angle_deg)
                    or tip_angle_deg > config.tip_angle_stop_deg
                )
            ):
                hardware_stop = True
                hardware_stop_reason = "tip_angle_stop"

            send_started = time.perf_counter()
            command_executed = False
            if config.send_commands and not hardware_stop:
                hw.send_step(p_now=p_commanded, u0=u0, dt=dt)
                command_executed = True
            command_monotonic_s = time.perf_counter()
            send_ms = 1.0e3 * (command_monotonic_s - send_started)
            if hardware_stop:
                _safe_stop(hw)
                stop_reason = hardware_stop_reason

            plot_ms = 0.0
            if config.save_controller_plots:
                plot_started = time.perf_counter()
                plot_reference_debug(
                    mpc,
                    x_meas,
                    info=info,
                    n_ref=3,
                    show_boundaries=True,
                    save_path=str(plot_dir / f"step_{step:05d}.png"),
                    mag_pos=np.asarray(p_measured[:3], dtype=float),
                    mag_dir=np.asarray(dipole_dir_from_p8(p_measured), dtype=float),
                    mag_pos_next=np.asarray(p_commanded[:3], dtype=float),
                    mag_dir_next=np.asarray(dipole_dir_from_p8(p_commanded), dtype=float),
                )
                plot_ms = 1.0e3 * (time.perf_counter() - plot_started)

            loop_wall_ms = 1.0e3 * (time.perf_counter() - loop_started)
            vision_metrics = _extract_vision_metrics(vision_result)

            row: dict[str, Any] = {
                "step": int(step),
                "timestamp_utc": utc_now,
                "elapsed_run_s": float(time.perf_counter() - run_started),
                "mode": "live" if config.send_commands else "shadow",
                "command_executed": int(command_executed),
                "hardware_stop": int(hardware_stop),
                "hardware_stop_reason": hardware_stop_reason,
                "reference_index": int(reference_index),
                "reference_mode": str(reference["mode"]),
                "path_reference_s0_m": reference_s0_m,
                "path_progress_current_m": path_progress_m,
                "capture_ms": float(capture_ms),
                "vision_ms": float(vision_ms),
                "robot_read_ms": float(robot_read_ms),
                "model_bias_eval_ms": float(model_bias_eval_ms),
                "controller_step_ms": float(controller_measured_ms),
                "send_ms": float(send_ms),
                "plot_ms": float(plot_ms),
                "loop_wall_ms": float(loop_wall_ms),
                "length_model_mm": 1.0e3 * model_length,
                "length_vision_mm": 1.0e3 * length_vision_raw if np.isfinite(length_vision_raw) else np.nan,
                "length_estimated_mm": 1.0e3 * length_estimate,
                "length_source": length_source,
                "translation_step_mm": translation_step_mm,
                "rotation_step_rad": rotation_step_rad,
                "insertion_step_mm": insertion_step_mm,
                "clearance_mm": clearance_mm,
                "tip_vessel_angle_deg": tip_angle_deg,
                "nearest_centreline_distance_mm": _safe_float(
                    centreline_metrics.get("nearest_centerline_distance_mm", centreline_metrics.get("distance_to_centerline_mm"))
                ),
                "curvature_nearest_1pm": _safe_float(curvature_metrics.get("curvature_nearest_1pm")),
                "curvature_mean_ahead_1pm": _safe_float(curvature_metrics.get("curvature_mean_ahead_1pm")),
                "curvature_max_ahead_1pm": _safe_float(curvature_metrics.get("curvature_max_ahead_1pm")),
                "u_limit_vx": float(live_u_max[0]),
                "u_limit_vy": float(live_u_max[1]),
                "u_limit_vz": float(live_u_max[2]),
                "u_limit_wx": float(live_u_max[3]),
                "u_limit_wy": float(live_u_max[4]),
                "u_limit_wz": float(live_u_max[5]),
                "u_limit_dL": float(live_u_max[6]),
                "u0_vx": float(u0[0]),
                "u0_vy": float(u0[1]),
                "u0_vz": float(u0[2]),
                "u0_wx": float(u0[3]),
                "u0_wy": float(u0[4]),
                "u0_wz": float(u0[5]),
                "u0_dL": float(u0[6]),
            }
            row.update(vector_components("measured_tip_m", measured_tip))
            row.update(vector_components("measured_tangent", measured_tangent if measured_tangent is not None else np.full(3, np.nan)))
            row.update(vector_components("magnet_measured_position_m", p_measured[:3]))
            row.update(vector_components("magnet_commanded_position_m", p_commanded[:3]))
            row.update(vector_components("reference_m", reference0 if reference0 is not None else np.full(3, np.nan)))
            row.update(
                vector_components(
                    "reference_tangent",
                    reference_tangent0
                    if reference_tangent0 is not None
                    else np.full(3, np.nan),
                )
            )
            row.update(vector_components("mpc_prediction0_m", x_pred0 if x_pred0 is not None else np.full(3, np.nan)))
            row.update(vector_components("model_current_m", x_model0 if x_model0 is not None else np.full(3, np.nan)))
            row.update(vector_components("model_nonlinear_next_m", x_nonlinear_next if x_nonlinear_next is not None else np.full(3, np.nan)))
            row.update(vision_metrics)
            row.update(transition_metrics)
            row.update(tracking_metrics)
            row.update(matrix_metrics)
            row.update(model_metrics)
            row.update(performance)
            logger.append(row)

            history.append(
                {
                    "step": int(step),
                    "x_meas": x_meas.copy(),
                    "p_measured": p_measured.copy(),
                    "p_commanded": p_commanded.copy(),
                    "u0": u0.copy(),
                    "B_first": None if B_first is None else B_first.copy(),
                    "x_pred0": None if x_pred0 is None else x_pred0.copy(),
                    "reference0": None if reference0 is None else reference0.copy(),
                    "reference_tangent0": (
                        None
                        if reference_tangent0 is None
                        else reference_tangent0.copy()
                    ),
                    "reference_s0_m": reference_s0_m,
                    "path_progress_current_m": path_progress_m,
                    "command_executed": bool(command_executed),
                    "info": info,
                }
            )

            previous_plan = PreviousHardwarePlan(
                step=step,
                x_meas=x_meas.copy(),
                u0=u0.copy(),
                B_first=None if B_first is None else B_first.copy(),
                x_pred0=None if x_pred0 is None else x_pred0.copy(),
                reference0=None if reference0 is None else reference0.copy(),
                robot_pose6_before=(
                    None if robot_pose6 is None else robot_pose6.copy()
                ),
                command_executed=bool(command_executed),
                monotonic_s=command_monotonic_s,
                reference_tangent0=(
                    None
                    if reference_tangent0 is None
                    else reference_tangent0.copy()
                ),
            )

            if config.verbose:
                print(
                    f"[{config.solver_mode}] step={step:04d} "
                    f"status={performance['status']} "
                    f"contour={tracking_metrics['tracking_contour_error_mm']:.3f} mm "
                    f"jac_err={transition_metrics['jac_hw_err_norm_xy_mm']:.3f} mm "
                    f"gain={transition_metrics['jac_hw_gain_xy']:.3f} "
                    f"controller={controller_measured_ms:.1f} ms "
                    f"loop={loop_wall_ms:.1f} ms"
                )

            if hardware_stop:
                break

    except Exception as exc:
        stop_reason = f"exception:{type(exc).__name__}"
        _safe_stop(hw)
        logger.finalize(
            stop_reason=stop_reason,
            extra={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "run_wall_s": float(time.perf_counter() - run_started),
            },
        )
        raise

    summary = logger.finalize(
        stop_reason=stop_reason,
        extra={
            "run_wall_s": float(time.perf_counter() - run_started),
            "steps_completed": len(history),
        },
    )
    return {
        "history": history,
        "summary": summary,
        "csv_path": logger.csv_path,
        "metadata_path": logger.metadata_path,
        "summary_path": logger.summary_path,
        "stop_reason": stop_reason,
    }


# Compatibility alias replacing the old hierarchical run_control function.
run_control = run_hardware_control_optimized
