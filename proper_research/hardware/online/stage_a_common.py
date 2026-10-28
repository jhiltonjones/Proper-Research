"""Single shared implementation of the Stage-A preflight/health-check
infrastructure used by every shape's run-script package (currently
`rectangle_stage_a` and `vessel_stage_a`).

2026-10-10 DRY pass: `vessel_stage_a/common.py` and `rectangle_stage_a/
common.py` used to each carry their own copy of `load_plan_initial_state`,
`reset_to_plan_initial`, `check_robot_safe`, `check_camera_healthy` and
`preflight` -- byte-identical for the first three, but `check_camera_
healthy`/`preflight` had quietly diverged (vessel's rig is physically
raised, so it needs a different beam-base pivot height and a magnet-
exclusion-aware reset path; rectangle's doesn't). That divergence was
expressed as two independent hand-maintained copies rather than as a
parameter, which is exactly the kind of drift this module exists to make
structurally impossible: every "what pivot/frame does this rig use"
decision is now a single `PivotConfig` value passed in by the caller,
not a fact baked into a function body. `vessel_stage_a/common.py` and
`rectangle_stage_a/common.py` are now thin per-rig shims over this
module -- see their own module docstrings for which `PivotConfig` each
one supplies. No behavior changed for any existing caller of either
shim; this is a pure extract-and-parameterize refactor.

A new shape's run-script package should do the same: define its own
`PivotConfig` (and, if its rig needs it, its own `magnet_transform_fn`
and exclusion/z-bounds for `reset_to_plan_initial_safe`/`preflight`)
rather than re-implementing any of the functions below.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROBOT_IP = "192.168.56.101"

# Execution layer "C" (2026-09-17 A/B/C identification test): accumulator
# seam (controller-owned q_cmd, not reset from q_meas or anchored to
# q_ref+trim) + 50Hz interpolated servoJ streaming between outer 10Hz ticks.
# Shared by every shape's run scripts -- see rectangle_stage_a's README for
# the full validation numbers.
SERVO_STREAM_HZ = 50.0
CONTROL_HZ = 10.0
JOINT_VELOCITY_LIMIT_RAD_S = 0.10   # >= every plan's own peak |u_ref| tested so far
MAX_JOINT_STEP_RAD = 0.010          # kept consistent: == dt * JOINT_VELOCITY_LIMIT_RAD_S
JOINT_ACCELERATION_LIMIT_RAD_S2 = 0.40


def load_plan_initial_state(plan_dir: str) -> tuple[np.ndarray, float]:
    """Return (initial 6 joints, initial insertion_m) from a plan's own state[0].

    Never hand-copy a shape's start pose into a script -- read it from the
    plan, so the same run script works for any shape `run_time_
    parameterization.py` produced.
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
    continuing.
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
    plain `reset_to_plan_initial` (no-op check) -- any rig that has no
    magnet-exclusion zone (e.g. rectangle_stage_a) can call this with its
    defaults and get its original behavior unchanged.
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


def raised_state_stream_config_factory(real_config_cls):
    """Return a drop-in replacement callable for `real_config_cls`
    (StateStreamConfig) whose T_robot_beam_pose6 z defaults to the
    recalibrated beam-base height (BEAM_BASE_PIVOT_Z) and whose
    marker_min_count/marker_max_count default to 2 -- the physical rig's
    middle marker was permanently removed 2026-09-29; tip-position tracking
    is unaffected, only the now-unused chord tangent is. Only the DEFAULT
    changes -- an explicit kwarg the caller passes still wins.

    Generic in the class it patches and the height it asks for; a specific
    rig (e.g. vessel_stage_a) supplies its own `BEAM_BASE_PIVOT_Z`-style
    constant when it calls this.
    """
    from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import BEAM_BASE_PIVOT_Z

    def _patched_state_stream_config(**kwargs):
        kwargs.setdefault("marker_min_count", 2)
        kwargs.setdefault("marker_max_count", 2)
        cfg = real_config_cls(**kwargs)
        if "T_robot_beam_pose6" not in kwargs:
            pose6 = list(cfg.T_robot_beam_pose6)
            pose6[2] = BEAM_BASE_PIVOT_Z
            cfg = dataclasses.replace(cfg, T_robot_beam_pose6=tuple(pose6))
        return cfg

    return _patched_state_stream_config


def patch_state_stream_config_for_recalibrated_rig(pf_module) -> None:
    """Monkeypatch `pf_module.StateStreamConfig` (close_loop_path_follow's
    own module-level reference, read by its `StateStreamConfig(...)` call
    site) with `raised_state_stream_config_factory`'s callable, built from
    whatever class `pf_module.StateStreamConfig` currently is. Must be
    called before anything else reassigns that attribute, since the
    "real" class is captured from it at call time, not re-resolved later."""
    pf_module.StateStreamConfig = raised_state_stream_config_factory(pf_module.StateStreamConfig)


@dataclasses.dataclass(frozen=True)
class PivotConfig:
    """Everything `check_camera_healthy` needs to know about a rig's own
    beam-base frame, and nothing else -- the single knob that varies
    between shapes/rigs. Swap this to change frame/height; no function
    below needs to know which rig it's running for.

    `build_scfg`: zero-arg callable returning this rig's own
        `StateStreamConfig` instance (calibration, axis conventions, ROI
        paths, and the beam-base pivot pose all live inside it).
    `chord_pivot_xyz`: given that `StateStreamConfig` instance, return the
        3-vector pivot used for the insertion-length chord check
        (||tip - pivot||). Kept as a separate hook from `build_scfg`
        because at least one existing rig (rectangle_stage_a) sources this
        from a different constant than its own `T_robot_beam_pose6`.
    `image_filename`: where the health-check snapshot is written (kept
        per-rig only so concurrent checks for different rigs can't clobber
        each other's `/dev/shm` file).
    """

    build_scfg: Callable[[], Any]
    chord_pivot_xyz: Callable[[Any], np.ndarray]
    image_filename: str


def check_camera_healthy(
    min_valid_fraction: float = 0.8, n_frames: int = 20,
    *, expected_insertion_m: float | None = None, insertion_tol_mm: float = 3.0,
    pivot_config: PivotConfig,
) -> None:
    """Raise if the camera/vision pipeline can't reliably find the tip.

    Does not touch the robot connection at all -- safe to run before or
    after any robot health check, in either order.

    `expected_insertion_m`, if given, ALSO verifies the PHYSICAL insertion
    length via the camera (2026-09-21) -- the advancer (/dev/ttyACM0) has
    no position encoder; the software-side `insertion_m` accumulator
    resets to the plan's L0 at the start of every run regardless of where
    the physical catheter/wire actually is, since nothing else ever
    re-homes or verifies it. A prior run that stopped mid-path (e.g.
    tcp_out_of_workspace, stale_vision) can leave the physical advancer
    extended well past the next run's assumed starting length, with no
    warning -- this is the only automatic check for that gap.

    `pivot_config` supplies the rig-specific `StateStreamConfig` and chord
    pivot -- see `PivotConfig`'s own docstring. This is the one place
    "what height/frame is this rig's beam base at" enters the check.
    """
    from proper_research.hardware.online.state_stream import NewFrameTipMapper
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource

    scfg = pivot_config.build_scfg()
    pivot_xyz = np.asarray(pivot_config.chord_pivot_xyz(scfg), dtype=float)
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=29.0, image_filename=pivot_config.image_filename,
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
                    lengths_mm.append(float(np.linalg.norm(tip - pivot_xyz)) * 1e3)
            time.sleep(0.1)
        frac = found / n_frames
        print(f"[health] camera: {found}/{n_frames} frames had a valid tip estimate "
              f"(pivot z={pivot_xyz[2]*1e3:.1f}mm)")
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
                    f"expected L0={expected_mm:.1f}mm. Refusing to proceed rather than assume "
                    f"the software accumulator's value is correct."
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
    pivot_config: PivotConfig,
    magnet_transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    magnet_exclusion_lumen_C_m: np.ndarray | None = None,
    magnet_exclusion_radius_m: float | None = None,
    magnet_z_bounds_m: tuple[float, float] | None = None,
    reset_target_q0: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan.

    When `magnet_transform_fn` and at least one of the exclusion/z-bounds
    args are given, the reset motion's straight-line joint-space path is
    validated safe before it moves -- see `reset_to_plan_initial_safe`. A
    rig with no magnet-exclusion zone (e.g. rectangle_stage_a) simply never
    passes these, and gets the original plain-reset behavior.

    `reset_target_q0`: opt-in, None by default (unchanged behaviour for
    every existing caller -- resets to the plan's own q0, same as before).
    When given, the robot resets to THIS joint target instead -- lets a run
    start the magnet at a different distance from the beam base than the
    plan's own designed start position, WITHOUT re-running the offline
    configuration-path/Jacobian-schedule pipeline at all. The function
    still RETURNS the plan's own (q0, l0) -- used for cfg.initial_
    insertion_m and this function's own insertion-length camera check,
    both unaffected by this override -- only the reset motion's actual
    target changes.
    """
    q0, l0 = load_plan_initial_state(plan_dir)
    print(f"[preflight] plan initial state: q0={np.round(q0, 4).tolist()} L0={l0*1000:.2f}mm")
    reset_q0 = q0 if reset_target_q0 is None else np.asarray(reset_target_q0, dtype=float).reshape(6)
    if reset_target_q0 is not None:
        print(f"[preflight] reset target OVERRIDDEN to {np.round(reset_q0, 4).tolist()} "
              f"(plan's own trajectory/schedule unaffected)")
    check_camera_healthy(
        expected_insertion_m=l0, insertion_tol_mm=insertion_tol_mm, pivot_config=pivot_config,
    )
    reset_to_plan_initial_safe(
        reset_q0, robot_ip=robot_ip,
        magnet_transform_fn=magnet_transform_fn,
        magnet_exclusion_lumen_C_m=magnet_exclusion_lumen_C_m,
        magnet_exclusion_radius_m=magnet_exclusion_radius_m,
        magnet_z_bounds_m=magnet_z_bounds_m,
    )
    check_robot_safe(robot_ip=robot_ip)
    return q0, l0


def resolve_start_pose_at_radius(
    q0_seed: np.ndarray, radius_mm: float, *, robot_kin,
) -> np.ndarray:
    """Return a new joint target that moves the magnet to `radius_mm` from
    the beam base, along the SAME direction `q0_seed`'s own magnet pose
    already sits at, with the SAME orientation -- i.e. a radial move, not
    a new arc position.

    Used to test a --beam-base-exclusion-floor-mm (or --magnet-exclusion-
    soft-radius-mm) wider than a plan's own built-in start margin would
    otherwise allow through the preflight reset-path check, WITHOUT
    re-running the offline configuration-path/Jacobian-schedule pipeline
    (a multi-hour operation) -- see preflight's own `reset_target_q0`
    parameter, which this is meant to feed.

    Orientation is reused UNCHANGED, not re-solved: this project's
    reference_orientation_matrix (sweep_free_space_arc_dipole.py, used by
    design_stage3_tracked_path.py to build every plan's own magnet_pose6_R)
    depends only on the DIRECTION to the beam base, never on distance --
    confirmed by reading its own implementation (both the calibration and
    target directions are normalized before use) -- so a purely radial
    move needs no new orientation solve.

    Raises RuntimeError if the IK for the new pose does not converge.
    """
    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
    from proper_research.rig_calibration import BEAM_BASE_XYZ_M

    q0_seed = np.asarray(q0_seed, dtype=float).reshape(6)
    seed_T = urik.forward_kinematics(q0_seed, robot_kin.dh, robot_kin.T_F_M).T_R_target
    seed_xyz = seed_T[:3, 3].copy()
    direction = seed_xyz - BEAM_BASE_XYZ_M
    direction_norm_m = float(np.linalg.norm(direction))
    direction = direction / direction_norm_m
    new_xyz = BEAM_BASE_XYZ_M + direction * (float(radius_mm) * 1e-3)

    T_new = seed_T.copy()
    T_new[:3, 3] = new_xyz
    ik = urik.inverse_kinematics_dls(
        T_R_target=T_new, q_seed_rad=q0_seed, dh=robot_kin.dh,
        T_F_target=robot_kin.T_F_M, cfg=robot_kin.ik_cfg,
    )
    if not ik.converged:
        raise RuntimeError(
            f"resolve_start_pose_at_radius({radius_mm}): IK did not converge for the "
            f"radial move from {direction_norm_m*1e3:.1f}mm to {radius_mm:.1f}mm along "
            f"the seed pose's own direction from the beam base."
        )
    print(f"[start-radius-override] reset target moved from {direction_norm_m*1e3:.1f}mm "
          f"to {radius_mm:.1f}mm from the beam base (same direction/orientation, "
          f"IK position_error={ik.final_position_error_m*1e3:.4f}mm)")
    return np.asarray(ik.q_rad, dtype=float).reshape(6)
