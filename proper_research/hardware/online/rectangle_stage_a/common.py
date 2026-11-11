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

import glob
import json
import os
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

# Free-space TCP workspace box (2026-11-04). TEMPORARY single source for
# this package: both run scripts explicitly set close_loop_path_follow.
# CONFIG.workspace_xyz_min_m/max_m to THESE values (instead of silently
# inheriting PathFollowConfig's own class default, which used to be the
# ONLY place this box lived) and both build their new in-QP TCP-workspace
# constraint (see run_mpc_delay_aware_qn_zero.py / run_mpc_delay_aware_
# insertion_anchor.py's --tcp-workspace-robust-margin-mm) from the SAME
# two tuples below -- one post-hoc monitor check and one QP constraint
# reading one definition, not two independently-maintained copies. To be
# absorbed into the single rig-configuration file once the broader
# rig-config refactor lands; until then, edit the box HERE only.
#
# Widened 2026-11-04 (author decision, R1): the rebuilt (corrected-pivot)
# triangle plans violate the original box (min=(0.15,-1.20,0.20),
# max=(1.10,-0.483,0.70)) by 28-125mm. Both y faces and the negative-x
# face widened outward by 0.05m; z and the positive-x face unchanged.
# SUPERSEDED 2026-11-05 (Round 27): the above was an ad-hoc guess, never
# grounded in measurement. The author jogged the real arm to its six
# physical workspace limits and reported the raw TCP extremes
# (robot-base frame): x in [0.22469, 0.60971], y in [-0.91361,
# -0.47576], z in [0.34544, 0.48496]. Reduced here by a 20mm safety
# margin on every face -- that margin must NOT be re-applied anywhere
# downstream (the L1 tcp_box_margin_m / QP
# --tcp-workspace-robust-margin-mm terms are a separate, additional
# requirement on top of this already-safe box, not a second
# application of this 20mm).
#
# 2026-11-10: z_min was briefly loosened by 1.3mm (0.36544 -> 0.36414m)
# to make the shared, hardware-fixed start pose clear the 25mm gate.
# REVERTED same day (Round 33, AF1): the box is the rig's measured
# physical limit, not something sized to a plan's requirement -- the
# loosening was never grounded in a new measurement, it was sized only
# to pass the gate, which is exactly the practice this box exists to
# rule out. z_min is back at the original 0.36544.
#
# DOCUMENTED EXCEPTION, not a box change: node 0 -- the real,
# hardware-fixed start pose shared by every shape plan (rectangle,
# triangle-35, and any future shape using the same start) -- clears
# this original z_min by 24.7336mm against the general 25mm QP/
# tracking clearance requirement (0.2664mm short). This node is
# fixed/committed in L1 and is never re-solved by any residual, so no
# planner weight or threshold can move it closer to or further from
# this face. The exception is accepted for this one node only: 24.7336
# mm at the real start pose, vs the 25mm gate that applies to every
# other (solved) node. See BOX_ADJUSTMENT_NOTE.txt (plan dirs and
# here) for the recomputation and the revert.
WORKSPACE_XYZ_MIN_M: tuple[float, float, float] = (0.24469, -0.89361, 0.36544)
WORKSPACE_XYZ_MAX_M: tuple[float, float, float] = (0.58971, -0.49576, 0.46496)


def _calibrated_default_initial_poses_for_guard() -> np.ndarray:
    from proper_research.planning.planning_context import (
        _calibrated_default_initial_poses,
    )
    pivot, _start, _L0, _dt = _calibrated_default_initial_poses()
    return pivot[:3]


def build_or_load_schedule(
    plan_dir: str,
    cache_path: str | None,
    allow_unverified: bool = False,
    label: str = "schedule",
) -> np.ndarray:
    """The ONE rectangle-stage-A Jacobian-schedule builder (Z4 consolidation,
    2026-11-10): this used to be copy-pasted, independently, across nine
    other run_*.py scripts in this package (run_mpc_delay_aware.py,
    run_mpc_delay_aware_insertion_anchor.py, run_mpc_delay_aware_nullq_r700.py,
    run_mpc_delay_baseline.py, run_mpc_ltv.py, run_inv_2dof_trim.py,
    run_inv_2dof_map_trim.py, run_inv_2dof_delay_aware.py,
    run_inv_7dof_delay_aware.py), eight of which never received the
    Round-27/AD1 beam-base guard or the accurate-Jacobian-mode fix below --
    only run_mpc_delay_aware_qn_zero.py had both, and is the version this
    function is lifted from verbatim. Every script now imports this one
    instead of defining its own.

    `label` is just the log-line prefix (kept per-caller for readable
    console output); it changes no behaviour.
    """
    # 2026-11-05 (Round 27): build_planning_context()'s pivot now matches
    # the calibrated beam base by default (previously a THIRD stale-
    # pivot reader, separate from the two already fixed elsewhere).
    # Verify the plan this schedule is FOR actually recorded the same
    # beam base its own L1 solve used, and refuse rather than silently
    # building/loading a schedule for the wrong geometry.
    current_beam_base = _calibrated_default_initial_poses_for_guard()
    # 2026-11-10 (AD1): plan_dir is the time_parameterized_configuration_path
    # directory; plan_shape_summary.json is NOT one level up from there (the
    # plan root) -- it's written one level further IN, under the shape-
    # specific shape_{rectangle,triangle}/ subdirectory (plan_shape_path.py's
    # `shape_dir`). Glob for the real file instead of assuming a fixed
    # relative path.
    _plan_root = os.path.dirname(plan_dir.rstrip("/"))
    _candidates = sorted(glob.glob(os.path.join(_plan_root, "shape_*", "plan_shape_summary.json")))
    if len(_candidates) > 1:
        raise RuntimeError(
            f"[{label}] found {len(_candidates)} plan_shape_summary.json files under "
            f"{_plan_root}/shape_*/ ({_candidates}) -- ambiguous, refusing to guess which "
            f"one describes this plan."
        )
    plan_summary_path = _candidates[0] if _candidates else os.path.join(_plan_root, "plan_shape_summary.json")
    # 2026-11-09: hard-fail by default on EITHER a missing field or a real
    # mismatch. The ONLY way around either check is the explicit
    # allow_unverified=True (plumbed in from each caller's own
    # --allow-unverified-plan flag) -- never to be used for a real
    # campaign run.
    _plan_beam_base = None
    if os.path.exists(plan_summary_path):
        with open(plan_summary_path) as _f:
            _plan_beam_base = json.load(_f).get("beam_base_xyz_m")
    if _plan_beam_base is None:
        _msg = (
            f"[{label}] REFUSING to build/load a schedule: {plan_summary_path} "
            f"does not exist or has no beam_base_xyz_m field, so this plan's "
            f"own L1 beam base cannot be verified against the current "
            f"build_planning_context() default ({current_beam_base.tolist()}). "
            f"Pass --allow-unverified-plan to bypass this (never for a real "
            f"campaign run)."
        )
        if allow_unverified:
            print(f"[{label}] WARNING (bypassed via --allow-unverified-plan): {_msg}")
        else:
            raise RuntimeError(_msg)
    else:
        _diff = float(np.max(np.abs(np.asarray(_plan_beam_base) - current_beam_base)))
        if _diff > 1.0e-6:
            _msg = (
                f"[{label}] REFUSING to build/load a schedule: plan's own "
                f"recorded beam base {_plan_beam_base} differs from the "
                f"current build_planning_context() default {current_beam_base.tolist()} "
                f"by {_diff*1000:.3f}mm (> 1e-6m). This plan and this schedule "
                f"build would be using different geometry. Pass "
                f"--allow-unverified-plan to bypass this (never for a real "
                f"campaign run)."
            )
            if allow_unverified:
                print(f"[{label}] WARNING (bypassed via --allow-unverified-plan): {_msg}")
            else:
                raise RuntimeError(_msg)
        else:
            print(
                f"[{label}] beam-base check PASSED: plan={plan_summary_path}, "
                f"value={_plan_beam_base}, ref={current_beam_base.tolist()}"
            )
    meta_path = (cache_path + ".meta.json") if cache_path else None
    if cache_path and os.path.exists(cache_path) and meta_path and os.path.exists(meta_path):
        with open(meta_path) as _f:
            _cached_beam_base = np.asarray(json.load(_f)["beam_base_xyz_m"], dtype=float)
        if float(np.max(np.abs(_cached_beam_base - current_beam_base))) <= 1.0e-6:
            print(f"[{label}] loading cached genuine LTV schedule from {cache_path} "
                  f"(beam base verified matching)")
            return np.load(cache_path)
        print(f"[{label}] cache at {cache_path} was built with a different beam base "
              f"({_cached_beam_base.tolist()} vs current {current_beam_base.tolist()}) "
              f"-- rebuilding, not reusing.")
    elif cache_path and os.path.exists(cache_path):
        print(f"[{label}] cache at {cache_path} has no beam-base metadata (pre-Round-27) "
              f"-- rebuilding, not reusing.")
    print(f"[{label}] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    from proper_research.controllers import mpc_variants
    from proper_research.controllers.beam_jacobian_providers import from_model_bundle
    from proper_research.planning.planning_context import build_planning_context
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    # jacobian_mode="accurate" for every caller now (2026-11-10 Z4
    # consolidation): checked 2026-09-24 on the raised plan, worst-case
    # relative Frobenius error between fast/accurate is 5.3% -- not huge,
    # but there is no reason to accept even that once it's this cheap to
    # remove, and nine of the ten pre-consolidation copies of this function
    # were still silently using "fast".
    jac_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate",
    )
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        if meta_path:
            with open(meta_path, "w") as _f:
                json.dump({"beam_base_xyz_m": current_beam_base.tolist()}, _f)
        print(f"[{label}] schedule cached -> {cache_path}")
    return schedule


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


def check_camera_healthy(
    min_valid_fraction: float = 0.8, n_frames: int = 20,
    *, expected_insertion_m: float | None = None, insertion_tol_mm: float = 3.0,
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

    Reuses `advancer_excitation.measure_l0`'s own convention exactly
    (||tip_position_m - PIVOT_XYZ||, the same already-validated
    base-to-tip chord measurement that tool uses to seed --l0-mm) rather
    than `StateEstimate.vision_beam_length_mm`, which this project's
    live pipeline (NewFrameTipMapper's frame_processor fast path) never
    actually populates -- confirmed empirically (0/20 valid readings
    despite 20/20 valid tip detections in the same frames).
    """
    from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig
    from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
    from proper_research.hardware.online.advancer_excitation.measure_l0 import PIVOT_XYZ

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
        lengths_mm: list[float] = []
        for _ in range(n_frames):
            est, _age = camera.latest(0.5)
            if est is not None:
                found += 1
                tip = np.asarray(est.tip_position_m, dtype=float)
                if np.all(np.isfinite(tip)):
                    lengths_mm.append(float(np.linalg.norm(tip - PIVOT_XYZ)) * 1e3)
            time.sleep(0.1)
        frac = found / n_frames
        print(f"[health] camera: {found}/{n_frames} frames had a valid tip estimate")
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
                    f"valid vision_beam_length_mm reading -- cannot verify physical insertion "
                    f"against the plan's expected L0={expected_mm:.1f}mm. Refusing to proceed "
                    f"rather than assume the software accumulator's value is correct."
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
) -> tuple[np.ndarray, float]:
    """Full pre-run sequence: health checks + reset. Returns (q0, L0) from the plan."""
    q0, l0 = load_plan_initial_state(plan_dir)
    print(f"[preflight] plan initial state: q0={np.round(q0, 4).tolist()} L0={l0*1000:.2f}mm")
    check_camera_healthy(expected_insertion_m=l0, insertion_tol_mm=insertion_tol_mm)
    reset_to_plan_initial(q0, robot_ip=robot_ip)
    check_robot_safe(robot_ip=robot_ip)
    return q0, l0


def patch_state_stream_config_two_markers(pf_module) -> None:
    """Monkeypatch `pf_module.StateStreamConfig` so its marker_min_count/
    marker_max_count default to 2 instead of the module's stale 3-4
    default -- the physical rig's middle marker was permanently removed
    2026-09-29 (see vessel_stage_a/common.py's `raised_state_stream_config_
    factory`, the original home of this fix for the vessel runners;
    rectangle_stage_a's own scripts never got the equivalent patch, so any
    of them doing real live-tracking warmup -- not just the open-loop
    scripts' simpler tip estimator -- fails with "expected 3-4 red markers,
    found 2" against the current rig). Deliberately does NOT touch
    T_robot_beam_pose6 (unlike the vessel version): rectangle_stage_a's own
    default pivot height is independently validated by this package's own
    open-loop runs and must not be overwritten by the vessel rig's
    recalibrated z. Only the DEFAULT changes -- an explicit kwarg at the
    call site still wins. Must be called before anything else reassigns
    `pf_module.StateStreamConfig`, since the "real" class is captured from
    it at call time, not re-resolved later."""
    import dataclasses

    real_config_cls = pf_module.StateStreamConfig

    def _patched_state_stream_config(**kwargs):
        kwargs.setdefault("marker_min_count", 2)
        kwargs.setdefault("marker_max_count", 2)
        return real_config_cls(**kwargs)

    pf_module.StateStreamConfig = _patched_state_stream_config
