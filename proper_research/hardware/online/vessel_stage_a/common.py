"""Shared helpers for the vessel-navigation Stage-A run scripts: pre-flight
reset, robot/camera health checks, and execution-layer-C defaults.

Identical to `rectangle_stage_a/common.py` (same three-step preflight
sequence, same execution-C defaults, same safety checks -- see that
module's docstring for the full rationale) with ONE change: the beam
base's z-height used for vision reconstruction and insertion-length
verification is raised +30mm to match the physically raised rig used
for the vessel study (`Z_RAISE_M`). The DEFAULT `StateStreamConfig`/
`PIVOT_XYZ` values elsewhere in this codebase are the pre-raise values
and are deliberately NOT changed globally -- every other shape's
hardware runner (rectangle/triangle/U-shape/S-curve) still depends on
them being un-raised. Found live 2026-09-23: the vision pipeline
reconstructs 3D tip position by projecting the 2D camera image onto a
plane at `T_robot_beam_pose6`'s z -- without this raise, the reported
tip z stays pinned at the OLD height regardless of the real rig height,
which is harmless for `run_open_loop_c.py`'s own error metric (it
explicitly projects out the z-component before reporting error_mm --
see `close_loop_path_follow.py:1593`) but would feed an uncorrected
~30mm phantom z-residual into the closed-loop MPC's disturbance
estimator (`process_isolated_adapter.py` sends the RAW 3-vector
`measured_beam_position`, no z-projection) and into `Qpbar`'s equally-
weighted xyz cost -- silently corrupting every closed-loop tick.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROBOT_IP = "192.168.56.101"

# Same execution-layer-C defaults as rectangle_stage_a/common.py -- see
# that module's docstring for the validation history (accumulator seam +
# 50Hz servoJ streaming).
SERVO_STREAM_HZ = 50.0
CONTROL_HZ = 10.0
JOINT_VELOCITY_LIMIT_RAD_S = 0.10
MAX_JOINT_STEP_RAD = 0.010
JOINT_ACCELERATION_LIMIT_RAD_S2 = 0.40

Z_RAISE_M = 0.03


def load_plan_initial_state(plan_dir: str) -> tuple[np.ndarray, float]:
    """Return (initial 6 joints, initial insertion_m) from a plan's own state[0]."""
    npz_path = Path(plan_dir) / (Path(plan_dir).name + ".npz")
    if not npz_path.exists():
        candidates = list(Path(plan_dir).glob("*.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not find a unique plan .npz under {plan_dir!r} "
                f"(looked for {npz_path.name}, found {[c.name for c in candidates]})."
            )
        npz_path = candidates[0]
    data = np.load(npz_path)
    state = data["state_reference"] if "state_reference" in data else data["state"]
    return np.asarray(state[0, :6], dtype=float), float(state[0, 6])


def reset_to_plan_initial(
    plan_initial_q: np.ndarray,
    *,
    robot_ip: str = ROBOT_IP,
    tol: float = 0.005,
    speed: float = 0.2,
    acceleration: float = 0.2,
) -> None:
    """Move the robot to the plan's start joints on its own short connection.
    Identical to rectangle_stage_a's version -- see that module's docstring
    for why servo_stop() before move_j() and the 3s settle matter."""
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    plan_initial_q = np.asarray(plan_initial_q, dtype=float).reshape(6)
    robot = URRTDERobot(robot_ip, frequency=125.0)
    robot.connect()
    try:
        robot.servo_stop()
        ok = robot.move_j(list(plan_initial_q), speed=speed, acceleration=acceleration)
        q = np.array(robot.get_joints())
        offset = float(np.max(np.abs(q - plan_initial_q)))
        print(f"[reset] move_j returned {ok}; max offset from plan-initial: {offset:.5f} rad")
        if offset >= tol:
            raise RuntimeError(
                f"refusing to start: not at plan-initial joints after reset "
                f"(offset {offset:.5f} rad >= tol {tol} rad)"
            )
    finally:
        robot.close()
    print("[reset] settling 3s before the harness opens its own connection...")
    time.sleep(3.0)


def _validate_reset_path_safe(
    current_q: np.ndarray,
    plan_initial_q: np.ndarray,
    *,
    magnet_transform_fn: Callable[[np.ndarray], np.ndarray],
    magnet_exclusion_lumen_C_m: np.ndarray | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
    n_samples: int = 200,
) -> None:
    """Raise if the straight-line JOINT-SPACE path `move_j` will take from
    `current_q` to `plan_initial_q` would violate the magnet-exclusion
    radius or magnet-z bounds anywhere along the way.

    `move_j` has zero real-time monitoring or interruption capability once
    started, and a joint-space-linear interpolation between two safe
    endpoints is not guaranteed to stay safe in between -- both real
    closed-loop incidents on 2026-09-23 showed the redundant DOF can move
    the magnet a long way for a small joint change. This must be checked
    BEFORE the blocking motion starts.
    """
    if magnet_exclusion_radius_m is None and magnet_z_bounds_m is None:
        return
    current_q = np.asarray(current_q, dtype=float).reshape(6)
    plan_initial_q = np.asarray(plan_initial_q, dtype=float).reshape(6)
    worst_gap_m = math.inf
    violations: list[str] = []
    for i in range(n_samples):
        t = i / (n_samples - 1)
        q = current_q + t * (plan_initial_q - current_q)
        xyz = np.asarray(magnet_transform_fn(q), dtype=float).reshape(3)
        if magnet_exclusion_radius_m is not None:
            gap_m = float(np.min(np.linalg.norm(
                np.asarray(magnet_exclusion_lumen_C_m, dtype=float) - xyz[None, :], axis=1
            )))
            worst_gap_m = min(worst_gap_m, gap_m)
            if gap_m < magnet_exclusion_radius_m:
                violations.append(
                    f"t={t:.3f}: exclusion gap={gap_m*1e3:.1f}mm < {magnet_exclusion_radius_m*1e3:.1f}mm"
                )
        if magnet_z_bounds_m is not None:
            z_min, z_max = magnet_z_bounds_m
            if xyz[2] < z_min or xyz[2] > z_max:
                violations.append(
                    f"t={t:.3f}: magnet_z={xyz[2]*1e3:.1f}mm outside "
                    f"[{z_min*1e3:.1f},{z_max*1e3:.1f}]mm"
                )
    if violations:
        raise RuntimeError(
            f"refusing reset: straight-line joint-space path from current joints to "
            f"plan-initial joints would violate a magnet safety constraint at "
            f"{len(violations)}/{n_samples} sampled points along the path -- move_j "
            f"cannot be interrupted mid-motion once started, so this reset is refused "
            f"before any motion begins. First violations: " + "; ".join(violations[:5])
        )
    msg = f"[reset-safety] path from current joints to plan-initial verified safe ({n_samples} samples)"
    if magnet_exclusion_radius_m is not None:
        msg += f"; worst exclusion gap along path = {worst_gap_m*1e3:.1f}mm"
    print(msg)


def reset_to_plan_initial_safe(
    plan_initial_q: np.ndarray,
    *,
    robot_ip: str = ROBOT_IP,
    tol: float = 0.005,
    speed: float = 0.2,
    acceleration: float = 0.2,
    magnet_transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    magnet_exclusion_lumen_C_m: np.ndarray | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
) -> None:
    """Like `reset_to_plan_initial`, but first reads the robot's current
    joints on a short read-only connection and validates that the
    straight-line joint-space path `move_j` will take stays clear of the
    magnet-exclusion radius and magnet-z bounds the whole way -- see
    `_validate_reset_path_safe`. Refuses (raises, no motion) if not.

    If no exclusion/z-bounds info is passed, this behaves exactly like
    plain `reset_to_plan_initial` (no-op check).
    """
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    plan_initial_q = np.asarray(plan_initial_q, dtype=float).reshape(6)

    if magnet_transform_fn is not None and (
        magnet_exclusion_radius_m is not None or magnet_z_bounds_m is not None
    ):
        robot = URRTDERobot(robot_ip, frequency=125.0)
        robot.connect()
        try:
            current_q = np.array(robot.get_joints())
        finally:
            robot.close()
        print(f"[reset-safety] current joints: {np.round(current_q, 4).tolist()}")
        _validate_reset_path_safe(
            current_q, plan_initial_q,
            magnet_transform_fn=magnet_transform_fn,
            magnet_exclusion_lumen_C_m=magnet_exclusion_lumen_C_m,
            magnet_exclusion_radius_m=magnet_exclusion_radius_m,
            magnet_z_bounds_m=magnet_z_bounds_m,
        )

    reset_to_plan_initial(plan_initial_q, robot_ip=robot_ip, tol=tol, speed=speed, acceleration=acceleration)


def check_robot_safe(robot_ip: str = ROBOT_IP) -> None:
    """Raise if the robot isn't connected and in a normal safety state."""
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    robot = URRTDERobot(robot_ip, frequency=125.0)
    robot.connect()
    try:
        safety_mode = robot.get_safety_mode()
        protective_stopped = robot.is_protective_stopped()
        print(f"[health] robot safety_mode={safety_mode} protective_stopped={protective_stopped}")
        if protective_stopped or safety_mode not in (1, 2):
            raise RuntimeError(
                f"robot not in a safe/normal state (safety_mode={safety_mode}, "
                f"protective_stopped={protective_stopped})"
            )
    finally:
        robot.close()


def _raised_stream_stream_config():
    """A StateStreamConfig instance with T_robot_beam_pose6's z raised by
    Z_RAISE_M -- everything else (calibration, axis conventions, ROI
    paths) taken from the real, validated default."""
    from proper_research.hardware.online.state_stream import StateStreamConfig

    base = StateStreamConfig(exposure=29.0, marker_min_count=2)
    pose6 = list(base.T_robot_beam_pose6)
    pose6[2] += Z_RAISE_M
    return dataclasses.replace(base, T_robot_beam_pose6=tuple(pose6))


def check_camera_healthy(
    min_valid_fraction: float = 0.8, n_frames: int = 20,
    *, expected_insertion_m: float | None = None, insertion_tol_mm: float = 3.0,
) -> None:
    """Raise if the camera/vision pipeline can't reliably find the tip.

    Identical to rectangle_stage_a's version except the pivot pose (both
    for vision reconstruction and for the insertion-length chord
    measurement) is raised +30mm in z to match the physically raised rig
    -- see this module's docstring. Without this, the insertion-length
    chord ||tip - pivot|| would pick up a spurious ~30mm vertical
    component (tip.z correctly raised, pivot.z stale), inflating the
    measured length by roughly 10-15mm and spuriously failing this check
    even when the advancer is correctly positioned.
    """
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource

    scfg = _raised_stream_stream_config()
    pivot_xyz_raised = np.asarray(scfg.T_robot_beam_pose6[:3], dtype=float)
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/vessel_stage_a_camera_check.png",
            roi_polygon_path=scfg.roi_polygon_path, manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=lambda: None, robot_pose_getter=lambda: None,
        insertion_length_getter=lambda: 0.02, frame_processor=mapper,
    )
    camera.start()
    try:
        time.sleep(1.0)
        found = 0
        lengths_mm: list[float] = []
        for _ in range(n_frames):
            est, _age = camera.latest(0.5)
            if est is not None:
                found += 1
                tip = np.asarray(est.tip_position_m, dtype=float)
                if np.all(np.isfinite(tip)):
                    lengths_mm.append(float(np.linalg.norm(tip - pivot_xyz_raised)) * 1e3)
            time.sleep(0.1)
        frac = found / n_frames
        print(f"[health] camera: {found}/{n_frames} frames had a valid tip estimate "
              f"(raised pivot z={pivot_xyz_raised[2]*1e3:.1f}mm)")
        if frac < min_valid_fraction:
            raise RuntimeError(
                f"camera unhealthy: only {found}/{n_frames} frames found the tip "
                f"(need >= {min_valid_fraction:.0%})"
            )
        if expected_insertion_m is not None:
            expected_mm = expected_insertion_m * 1000.0
            if len(lengths_mm) < int(min_valid_fraction * n_frames):
                raise RuntimeError(
                    f"insertion-length check: only {len(lengths_mm)}/{n_frames} frames had a "
                    f"valid reading -- cannot verify physical insertion against the plan's "
                    f"expected L0={expected_mm:.1f}mm. Refusing to proceed."
                )
            measured_mm = float(np.median(lengths_mm))
            diff_mm = abs(measured_mm - expected_mm)
            print(f"[health] insertion length: measured={measured_mm:.1f}mm "
                  f"(median of {len(lengths_mm)} frames) expected={expected_mm:.1f}mm "
                  f"diff={diff_mm:.1f}mm")
            if diff_mm > insertion_tol_mm:
                raise RuntimeError(
                    f"insertion mismatch: camera-measured physical length {measured_mm:.1f}mm "
                    f"differs from the plan's expected L0={expected_mm:.1f}mm by {diff_mm:.1f}mm "
                    f"(tolerance {insertion_tol_mm:.1f}mm). The advancer likely retained an "
                    f"extension from a previous run -- physically retract/re-home it to the "
                    f"plan's start before proceeding."
                )
    finally:
        camera.stop()


def preflight(
    plan_dir: str, *, robot_ip: str = ROBOT_IP, insertion_tol_mm: float = 3.0,
    magnet_transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    magnet_exclusion_lumen_C_m: np.ndarray | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan.

    When `magnet_transform_fn` and at least one of the exclusion/z-bounds
    args are given, the reset motion's straight-line joint-space path is
    validated safe before it moves -- see `reset_to_plan_initial_safe`.
    """
    q0, l0 = load_plan_initial_state(plan_dir)
    print(f"[preflight] plan initial state: q0={np.round(q0, 4).tolist()} L0={l0*1000:.2f}mm")
    check_camera_healthy(expected_insertion_m=l0, insertion_tol_mm=insertion_tol_mm)
    reset_to_plan_initial_safe(
        q0, robot_ip=robot_ip,
        magnet_transform_fn=magnet_transform_fn,
        magnet_exclusion_lumen_C_m=magnet_exclusion_lumen_C_m,
        magnet_exclusion_radius_m=magnet_exclusion_radius_m,
        magnet_z_bounds_m=magnet_z_bounds_m,
    )
    check_robot_safe(robot_ip=robot_ip)
    return q0, l0
