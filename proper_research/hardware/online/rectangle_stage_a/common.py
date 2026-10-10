"""Rectangle Stage-A rig configuration over the shared Stage-A preflight/
health-check infrastructure in `proper_research.hardware.online.
stage_a_common`.

This rig's beam base is NOT raised -- `check_camera_healthy`/`preflight`
below use the plain default `StateStreamConfig` and the chord-pivot
convention `advancer_excitation.measure_l0.PIVOT_XYZ` already uses
(||tip_position_m - PIVOT_XYZ||, the same already-validated base-to-tip
chord measurement that tool uses to seed --l0-mm), NOT `StateEstimate.
vision_beam_length_mm`, which this project's live pipeline (NewFrameTip
Mapper's frame_processor fast path) never actually populates (confirmed
empirically: 0/20 valid readings despite 20/20 valid tip detections in
the same frames). Compare `vessel_stage_a/common.py`, whose rig IS
raised and which supplies a different `PivotConfig` for exactly that
reason -- see `stage_a_common.PivotConfig`'s docstring for why this is a
parameter rather than two independently-hand-maintained copies of
`check_camera_healthy`.

Every run script in this package does the same three things before it
touches `close_loop_path_follow.main()`:

1. Load the plan and read its own initial state (joints + insertion) --
   never hardcode a shape's start pose; different plans start differently.
2. Move the robot there on its own short-lived connection, verify it landed
   (`servo_stop()` BEFORE `move_j()` -- see `reset_to_plan_initial`'s
   docstring for why this specific ordering matters), then close that
   connection and settle before the harness opens its own.
3. Refuse to proceed if the reset didn't land or the robot isn't in a
   normal safety state.

Skipping step 1/2 and re-using wherever the robot happened to be left by a
previous run is exactly the mistake that produced a `tcp_out_of_workspace`
abort on 2026-09-17 -- the accumulator seeds itself from the CURRENT
measured joints, silently, with no warning if that isn't the plan's start.
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
    check_robot_safe,
)
from proper_research.hardware.online import stage_a_common as _shared

__all__ = [
    "ROBOT_IP", "SERVO_STREAM_HZ", "CONTROL_HZ", "JOINT_VELOCITY_LIMIT_RAD_S",
    "MAX_JOINT_STEP_RAD", "JOINT_ACCELERATION_LIMIT_RAD_S2",
    "load_plan_initial_state", "reset_to_plan_initial", "check_robot_safe",
    "check_camera_healthy", "preflight",
]


def _rectangle_pivot_config() -> PivotConfig:
    from proper_research.hardware.online.state_stream import StateStreamConfig
    from proper_research.hardware.online.advancer_excitation.measure_l0 import PIVOT_XYZ

    return PivotConfig(
        build_scfg=lambda: StateStreamConfig(exposure=29.0, marker_min_count=2),
        chord_pivot_xyz=lambda _scfg: PIVOT_XYZ,
        image_filename="/dev/shm/stage_a_camera_check.png",
    )


def check_camera_healthy(
    min_valid_fraction: float = 0.8, n_frames: int = 20,
    *, expected_insertion_m: float | None = None, insertion_tol_mm: float = 3.0,
) -> None:
    """Raise if the camera/vision pipeline can't reliably find the tip.

    See `stage_a_common.check_camera_healthy` for the full behavior; this
    just supplies this rig's own (unraised) `PivotConfig`.
    """
    _shared.check_camera_healthy(
        min_valid_fraction, n_frames,
        expected_insertion_m=expected_insertion_m, insertion_tol_mm=insertion_tol_mm,
        pivot_config=_rectangle_pivot_config(),
    )


def preflight(
    plan_dir: str, *, robot_ip: str = ROBOT_IP, insertion_tol_mm: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan."""
    return _shared.preflight(
        plan_dir, robot_ip=robot_ip, insertion_tol_mm=insertion_tol_mm,
        pivot_config=_rectangle_pivot_config(),
    )
