"""Shared helpers for the Stage-A execution-layer-C run scripts in this
package: pre-flight reset, robot/camera health checks, and the execution-
layer-C defaults validated live on the rectangle shape (2026-09-17).

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

import time
from pathlib import Path
from typing import Any

import numpy as np

ROBOT_IP = "192.168.56.101"

# Execution layer "C" (2026-09-17 A/B/C identification test): accumulator
# seam (controller-owned q_cmd, not reset from q_meas or anchored to
# q_ref+trim) + 50Hz interpolated servoJ streaming between outer 10Hz ticks.
# G (final realization gain) went 0.199 (direct-apply seam) -> 1.000
# (accumulator) -> mean/max command-tracking error additionally dropped
# ~4x (11.8->2.7mrad / 36.2->8.7mrad) from 10Hz->50Hz streaming on top of
# that. See this package's README.md for the full numbers.
SERVO_STREAM_HZ = 50.0
CONTROL_HZ = 10.0
JOINT_VELOCITY_LIMIT_RAD_S = 0.10   # >= every plan's own peak |u_ref| tested so far
MAX_JOINT_STEP_RAD = 0.010          # kept consistent: == dt * JOINT_VELOCITY_LIMIT_RAD_S
JOINT_ACCELERATION_LIMIT_RAD_S2 = 0.40


def load_plan_initial_state(plan_dir: str) -> tuple[np.ndarray, float]:
    """Return (initial 6 joints, initial insertion_m) from a plan's own state[0].

    Never hand-copy a shape's start pose into a script -- read it from the
    plan, so the same run script works for the rectangle, triangle, U-shape,
    S-curve, or anything else `run_time_parameterization.py` produced.
    """
    npz_path = Path(plan_dir) / (Path(plan_dir).name + ".npz")
    if not npz_path.exists():
        # plan_dir may already point at the file's parent with a
        # differently-named npz -- fall back to the only *.npz in the dir.
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

    `servo_stop()` before `move_j()` matters: issuing a blocking `moveJ`
    right after a `servoJ` streaming sequence (from a *previous* run in the
    same session) without first calling `servoStop()` reliably fails --
    confirmed live 2026-09-17 (`move_j` returned `False`, then the process
    crashed with repeated "Robot is disconnected, reconnecting..." on the
    next attempt). Calling `servo_stop()` unconditionally, even when nothing
    was streaming, is a no-op and cheap insurance.

    Raises RuntimeError if the reset doesn't land within `tol` (refusing to
    start a run from an unverified position) rather than silently
    continuing -- see this package's README for what happened the one time
    a run script skipped this check.
    """
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
    # Let the controller fully release this connection before the harness
    # opens its own -- skipping this produced 4 consecutive connection
    # failures in a row on 2026-09-17 (2 silent stale-joint aborts, 1 crash).
    print("[reset] settling 3s before the harness opens its own connection...")
    time.sleep(3.0)


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


def check_camera_healthy(min_valid_fraction: float = 0.8, n_frames: int = 20) -> None:
    """Raise if the camera/vision pipeline can't reliably find the tip.

    Does not touch the robot connection at all -- safe to run before or
    after any robot health check, in either order.
    """
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource

    scfg = StateStreamConfig(exposure=29.0, marker_min_count=2)
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename="/dev/shm/stage_a_camera_check.png",
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
        for _ in range(n_frames):
            est, _age = camera.latest(0.5)
            if est is not None:
                found += 1
            time.sleep(0.1)
        frac = found / n_frames
        print(f"[health] camera: {found}/{n_frames} frames had a valid tip estimate")
        if frac < min_valid_fraction:
            raise RuntimeError(
                f"camera unhealthy: only {found}/{n_frames} frames found the tip "
                f"(need >= {min_valid_fraction:.0%})"
            )
    finally:
        camera.stop()


def preflight(plan_dir: str, *, robot_ip: str = ROBOT_IP) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan."""
    q0, l0 = load_plan_initial_state(plan_dir)
    print(f"[preflight] plan initial state: q0={np.round(q0, 4).tolist()} L0={l0*1000:.2f}mm")
    check_camera_healthy()
    reset_to_plan_initial(q0, robot_ip=robot_ip)
    check_robot_safe(robot_ip=robot_ip)
    return q0, l0
