"""Vessel Stage-A rig configuration over the shared Stage-A preflight/
health-check infrastructure in `proper_research.hardware.online.
stage_a_common`.

This rig's beam base IS physically raised: vision reconstruction and the
insertion-length chord check both need `T_robot_beam_pose6`'s z to be the
recalibrated height (`build_vessel_plan.BEAM_BASE_PIVOT_Z`), not the
DEFAULT `StateStreamConfig`/`PIVOT_XYZ` z used elsewhere in this codebase
-- those are deliberately NOT changed globally, since every other shape's
hardware runner (rectangle/triangle/U-shape/S-curve) still depends on
them being unraised. Found live 2026-09-23: the vision pipeline
reconstructs 3D tip position by projecting the 2D camera image onto a
plane at `T_robot_beam_pose6`'s z -- without matching this to the real
rig height, the reported tip z stays pinned at the WRONG height, which is
harmless for `run_open_loop_c.py`'s own error metric (it explicitly
projects out the z-component before reporting error_mm) but would feed an
uncorrected phantom z-residual into the closed-loop MPC's disturbance
estimator (`process_isolated_adapter.py` sends the RAW 3-vector
`measured_beam_position`, no z-projection) and into `Qpbar`'s equally-
weighted xyz cost -- silently corrupting every closed-loop tick.

2026-10-10: this module used to carry its own full copy of every function
below (byte-identical to rectangle_stage_a/common.py for the plan-
agnostic ones, independently diverged for `check_camera_healthy`/
`preflight`). Both packages now share one implementation in
`stage_a_common`, parameterized by a `PivotConfig` -- see that module's
docstring. This file supplies vessel's own raised `PivotConfig` and
magnet-exclusion-aware preflight wiring; it owns no infrastructure logic
of its own anymore.
"""
from __future__ import annotations

import numpy as np

from proper_research.hardware.online.stage_a_common import (
    ROBOT_IP,
    SERVO_STREAM_HZ,
    CONTROL_HZ,
    JOINT_VELOCITY_LIMIT_RAD_S,
    MAX_JOINT_STEP_RAD,
    JOINT_ACCELERATION_LIMIT_RAD_S2,
    PivotConfig,
    load_plan_initial_state,
    reset_to_plan_initial,
    reset_to_plan_initial_safe,
    check_robot_safe,
    raised_state_stream_config_factory,
    patch_state_stream_config_for_recalibrated_rig,
    resolve_start_pose_at_radius,
    _validate_reset_path_safe,
)
from proper_research.hardware.online import stage_a_common as _shared

__all__ = [
    "ROBOT_IP", "SERVO_STREAM_HZ", "CONTROL_HZ", "JOINT_VELOCITY_LIMIT_RAD_S",
    "MAX_JOINT_STEP_RAD", "JOINT_ACCELERATION_LIMIT_RAD_S2", "Z_RAISE_M",
    "load_plan_initial_state", "reset_to_plan_initial", "reset_to_plan_initial_safe",
    "check_robot_safe", "raised_state_stream_config_factory",
    "patch_state_stream_config_for_recalibrated_rig", "resolve_start_pose_at_radius",
    "check_camera_healthy", "preflight",
]

Z_RAISE_M = 0.03  # STALE -- do not use for the recalibrated setup, see _raised_pivot_config


def _raised_pivot_config() -> PivotConfig:
    """This rig's own `PivotConfig`: the raised `StateStreamConfig` (z ==
    the recalibrated `BEAM_BASE_PIVOT_Z`), with the chord-check pivot taken
    straight from that same config -- so the camera health check and the
    real run can never disagree about the beam-base height. Built from the
    same `raised_state_stream_config_factory` the live runners use."""
    from proper_research.hardware.online.state_stream import StateStreamConfig

    cfg_cls = raised_state_stream_config_factory(StateStreamConfig)
    return PivotConfig(
        build_scfg=lambda: cfg_cls(exposure=29.0),
        chord_pivot_xyz=lambda scfg: np.asarray(scfg.T_robot_beam_pose6[:3], dtype=float),
        image_filename="/dev/shm/vessel_stage_a_camera_check.png",
    )


def _raised_stream_stream_config():
    """A StateStreamConfig instance with T_robot_beam_pose6's z set to the
    recalibrated beam-base height -- everything else (calibration, axis
    conventions, ROI paths) taken from the real, validated default. Used
    directly (not just via `check_camera_healthy`) by several scripts in
    this package that need a raised `StateStreamConfig` of their own
    (e.g. `approach_vessel_contact.py`, `measure_jcam_*.py`,
    `fixed_dipole_arc_*.py`, `sweep_free_space_arc_dipole.py`)."""
    return _raised_pivot_config().build_scfg()


def check_camera_healthy(
    min_valid_fraction: float = 0.8, n_frames: int = 20,
    *, expected_insertion_m: float | None = None, insertion_tol_mm: float = 3.0,
) -> None:
    """Raise if the camera/vision pipeline can't reliably find the tip.

    See `stage_a_common.check_camera_healthy` for the full behavior; this
    just supplies this rig's own raised `PivotConfig`.
    """
    _shared.check_camera_healthy(
        min_valid_fraction, n_frames,
        expected_insertion_m=expected_insertion_m, insertion_tol_mm=insertion_tol_mm,
        pivot_config=_raised_pivot_config(),
    )


def preflight(
    plan_dir: str, *, robot_ip: str = ROBOT_IP, insertion_tol_mm: float = 3.0,
    magnet_transform_fn=None,
    magnet_exclusion_lumen_C_m: np.ndarray | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
    reset_target_q0: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan.

    See `stage_a_common.preflight` for the full behavior (magnet-exclusion
    reset-path safety, `reset_target_q0` override); this just supplies
    this rig's own raised `PivotConfig`.
    """
    return _shared.preflight(
        plan_dir, robot_ip=robot_ip, insertion_tol_mm=insertion_tol_mm,
        pivot_config=_raised_pivot_config(),
        magnet_transform_fn=magnet_transform_fn,
        magnet_exclusion_lumen_C_m=magnet_exclusion_lumen_C_m,
        magnet_exclusion_radius_m=magnet_exclusion_radius_m,
        magnet_z_bounds_m=magnet_z_bounds_m,
        reset_target_q0=reset_target_q0,
    )
