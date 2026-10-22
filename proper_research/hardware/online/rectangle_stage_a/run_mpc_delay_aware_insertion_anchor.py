#!/usr/bin/env python3
"""Live validation of gamma=0/R700 with the planner<->live FRAME-MISMATCH
FIX, w_L=0 (2026-09-21).

Root-cause finding (2026-09-21, from the two nullq_r700 live runs): the
process-isolated MPC worker is spawned+warmed BEFORE preflight/the live
planner->robot frame registration exists (by design -- see
worker_process.py), so it was permanently solving against the RAW,
un-registered planner-frame reference (p_des^P, p_nominal^P), while the
harness's own live tip measurement and logged `desired_mm` are in the
live-registered robot frame R. Direct verification: for every affected
live run, E_chosen^QP (=||p_hat - p_des^P||, the QP's OWN internal task
error) was ~0.06-0.45mm -- i.e. the optimizer was tracking its own target
almost perfectly -- while E_chosen^harness/e_actual (measured against the
CORRECT, frame-registered target) matched Delta_p_target = p_des^P -
p_des^harness to within 0.02-0.04mm at every tested run. The "insertion
authority" pathology investigated all day (MT rejection, R700 selection,
the w_L insertion anchor) was, to first order, the optimizer successfully
cancelling a rigid coordinate-frame translation it had no way of knowing
about, not a genuine control-design failure -- see
CONTROLLER_JUSTIFICATION.md's frame-mismatch section for the full
diagnostic chain.

Fix, corrected design (2026-09-21, superseding an earlier measurement-
transform attempt at the adapter boundary -- reverted after an equivalence
unit test showed it mixes frames): p_des and p_nominal (both default from
reference.desired_position_m, the raw un-registered array) are the ONLY
things wrong. The schedule (J), the task-nullspace projector C, z_ref,
u_ref, and the measured beam position are already natively robot-frame --
J is built from real robot FK in build_or_load_schedule (labelled
"R.x,R.y,R.z" in its own console dump), independent of the planner's own
abstract path coordinates -- and must NOT be touched. The fix is applied
ONCE, worker-side, via `MPCWorkerHandle.set_frame_transform(R_fit, t_fit)`
(see worker_process.py's `_apply_frame_transform`): p_des^R = R_fit@p_des^P
+ t_fit, p_nominal^R = R_fit@p_nominal^P + t_fit. Neither is baked into any
constructor-time cached quantity (H/the OSQP sparsity pattern depend only
on cost weights and prediction matrices; the nullspace projectors depend
only on the schedule/C/state scale), so no re-warm is needed. The worker
refuses any real solve between `require_frame_transform()` and
`set_frame_transform()` (see main(), called right after spawn/warm-up and
right after the live registration is computed, respectively) -- this
adapter itself stays completely frame-agnostic.

This is the FIRST CORRECTED validation, run at w_L=0 (gamma=0, R700 as
originally selected) -- the insertion-anchor tuning from earlier today is
DISCARDED as contaminated by the frame bug, not carried into this run. If
this performs well live, most of today's "insertion authority instability"
story was the frame bug, not a control-design problem.

Everything else is frozen identically to nullq_r700: gamma_q=0, R=700*R0,
original Qp/Rd/u_ref, d=2, beta_d=1, N=15, V_f=0, insertion authority
(s_u,L=5mm/s, |u_L|<=2mm/s unchanged), execution-C, process isolation, same
plan/schedule.

Independent, non-optimization safety monitor kept from the insertion-anchor
attempt (not silent clipping): if the live insertion accumulator ever
drifts more than `--insertion-offset-abort-mm` from L_ref, the run
aborts/holds via the generic adapter_abort_reason hook in
close_loop_path_follow.py. RESTORED to the original conservative 5mm
default -- the earlier 10mm relaxation was itself contaminated by the
frame bug (the large excursions it was sized against were largely
frame-induced); there is no reason to grant that much runway now.
Threshold rationale, from the two nullq_r700 live runs: the completed
run's insertion offset stayed under ~2.8mm throughout; the aborted run
crossed 5mm by roughly tick 14 -- long before its eventual
tcp_out_of_workspace abort near tick 240 -- so 5mm sits outside normal
operation but well inside the early-warning zone of a runaway trajectory.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \\
        --plan-dir plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online import controller_adapters
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.planning.planning_context import build_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import common

_real_build_offline_solver = controller_adapters.build_offline_solver
_real_make_output_dir = pf._make_output_dir
_LAST_OUTPUT_DIR: Path | None = None
_METADATA_WRITTEN = False

# Frozen controller spec (see module docstring) -- module-level constants,
# not CLI flags, so this run cannot silently drift from the offline-
# validated formulation.
_GAMMA = 0.0
_INPUT_TRACKING_WEIGHT = 7.0          # 700 * default 1.0e-2 = R700
_INPUT_INCREMENT_WEIGHT = 1.0e-3      # unchanged from M0/nullq_r700
_INSERTION_RATE_LIMIT_M_S = 2.0e-3    # unchanged -- s_u,L stays 5mm/s via defaults
_INSERTION_STATE_ANCHOR_WEIGHT = 0.0  # RESTORED to 0 -- w_L=0.1 tuning discarded as
# contaminated by the frame-mismatch bug (see module docstring); this run
# isolates the frame fix alone, on the originally-selected gamma=0/R700.
_INSERTION_STATE_ANCHOR_SCALE = 0.25e-3  # s_L, matches state_error_scale[6]


def _wrapped_make_output_dir(cfg):
    global _LAST_OUTPUT_DIR
    out = _real_make_output_dir(cfg)
    _LAST_OUTPUT_DIR = out
    return out


pf._make_output_dir = _wrapped_make_output_dir


_DEADLINE_MS = 70.0
_INSERTION_OFFSET_ABORT_M = 0.005  # 5mm, see module docstring for rationale
_WORKER: MPCWorkerHandle | None = None


def spawn_and_warm_worker(
    *, plan_dir: str, schedule_cache: str, control_hz: float, prediction_horizon: int,
    joint_velocity_limit_rad_s: float, insertion_rate_limit_m_s: float,
    joint_acceleration_limit_rad_s2: float, position_error_scale_mm: float,
    position_tracking_weight: float,
) -> MPCWorkerHandle:
    """Spawn + warm the MPC worker BEFORE any RTDE/camera connection opens --
    identical lifecycle ordering to nullq_r700's runner."""
    dt = 1.0 / control_hz
    mpc_config_kwargs = dict(
        sample_period_s=dt, prediction_horizon=int(prediction_horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]), state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [insertion_rate_limit_m_s]),
        acceleration_limit=tuple([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s]),
    )
    increment_scale = tuple(
        dt * a for a in ([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s])
    )
    nullspace_r700_config_kwargs = dict(
        mpc_config_kwargs,
        input_tracking_weight=_INPUT_TRACKING_WEIGHT,
        input_increment_weight=_INPUT_INCREMENT_WEIGHT,
        input_increment_scale=increment_scale,
    )
    s = float(position_error_scale_mm) * 1.0e-3
    beam_config_kwargs = dict(
        position_error_scale_m=(s, s, s), position_tracking_weight=float(position_tracking_weight),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    print("[insertion_anchor] spawning + warming process-isolated MPC worker "
          "(BEFORE any hardware connection); building M0 + nullspace/R700/w_L...")
    worker = MPCWorkerHandle(
        plan_dir=plan_dir, schedule_path=schedule_cache,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True,
        nullspace_r700_config_kwargs=nullspace_r700_config_kwargs,
        insertion_state_anchor_weight=_INSERTION_STATE_ANCHOR_WEIGHT,
    )
    spawn_ms = (worker.t_worker_ready - worker.t_spawn_start) * 1e3
    print(f"[insertion_anchor] worker ready (spawn+import+construct+warm-up+checks = {spawn_ms:.0f}ms)")
    print(f"[insertion_anchor] CONTROLLER CONFIRMED: gamma={_GAMMA}, R700 "
          f"(input_tracking_weight={_INPUT_TRACKING_WEIGHT}), "
          f"insertion anchor w_L={_INSERTION_STATE_ANCHOR_WEIGHT} s_L={_INSERTION_STATE_ANCHOR_SCALE*1e3}mm, "
          f"safety abort at |L-Lref|>{_INSERTION_OFFSET_ABORT_M*1e3:.1f}mm")
    # Gate real solves until the live planner->robot registration is
    # applied (see module docstring's frame-mismatch fix) -- offline
    # replay callers never call this, so this is purely additive for the
    # live path.
    worker.require_frame_transform()
    print("[insertion_anchor] worker will refuse solves until set_frame_transform() is called "
          "(happens once preflight computes the live registration)")
    return worker


def _wrapped_build_offline_solver(kind, **kwargs):
    global _METADATA_WRITTEN
    if kind != "mpc_delay_aware":
        return _real_build_offline_solver(kind, **kwargs)
    assert _LAST_OUTPUT_DIR is not None, "output dir not yet created"
    assert _WORKER is not None, "worker must be spawned+warmed BEFORE pf.main() -- see main()"
    cfg = pf.CONFIG

    transform = pf._PLANNER_TO_LIVE_TRANSFORM
    if transform is None:
        raise RuntimeError(
            "pf._PLANNER_TO_LIVE_TRANSFORM is not set -- _load_plan_reference() must run "
            "(reference_source='plan_dir') and compute the live planner->R fit BEFORE "
            "build_offline_solver() is called, or the worker (which solves entirely in "
            "planner frame) would be paired with an unregistered live measurement."
        )
    r_fit, t_fit = transform

    if not _METADATA_WRITTEN:
        meta = {
            "controller_variant": "nullspace_Q_R700_frame_fix",
            "gamma": _GAMMA,
            "input_tracking_weight": _INPUT_TRACKING_WEIGHT,
            "input_increment_weight": _INPUT_INCREMENT_WEIGHT,
            "insertion_state_anchor_weight": _INSERTION_STATE_ANCHOR_WEIGHT,
            "insertion_state_anchor_scale_m": _INSERTION_STATE_ANCHOR_SCALE,
            "insertion_offset_abort_m": _INSERTION_OFFSET_ABORT_M,
            "state_scale_q_rad": math.radians(0.5),
            "state_scale_L_m": 0.25e-3,
            "input_scale_q_rad_s": 0.05,
            "input_scale_L_m_s": 5.0e-3,
            "delay_samples": 2,
            "beta_d": 1.0,
            "horizon": int(cfg.mpc_prediction_horizon),
            "terminal_cost": False,
            "insertion_rate_limit_m_s": _INSERTION_RATE_LIMIT_M_S,
            "plan_dir": str(cfg.plan_dir),
            "planner_to_live_R_fit": r_fit.tolist(),
            "planner_to_live_t_fit_m": t_fit.tolist(),
            "planner_to_live_t_fit_norm_m": float(np.linalg.norm(t_fit)),
        }
        (_LAST_OUTPUT_DIR / "controller_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[insertion_anchor] wrote controller_metadata.json -> {_LAST_OUTPUT_DIR} "
              f"(includes planner->live R_fit/t_fit, |t_fit|={np.linalg.norm(t_fit)*1e3:.1f}mm)")
        _METADATA_WRITTEN = True

    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=kwargs["reference"], worker=_WORKER, deadline_s=_DEADLINE_MS / 1e3,
        joint_velocity_limit_rad_s=float(cfg.joint_velocity_limit_rad_s),
        max_joint_step_rad=float(cfg.max_joint_step_rad),
        config=kwargs.get("adapter_config"),
        prediction_log_path=str(_LAST_OUTPUT_DIR / "predicted_beam_positions.jsonl"),
        controller_kind="nullspace_r700",
        controller_label="mpc_delay_aware_nullspace_r700_frame_fix_process_isolated",
        insertion_offset_abort_m=_INSERTION_OFFSET_ABORT_M,
    )
    _WORKER.set_frame_transform(r_fit, t_fit)
    print(f"[insertion_anchor] worker frame transform applied: "
          f"|t_fit|={np.linalg.norm(t_fit)*1e3:.1f}mm det(R_fit)={np.linalg.det(r_fit):+.3f}")

    # Regression check #4 (2026-09-21): verify the worker/harness target
    # discrepancy is now essentially zero, not just trust the fix silently.
    # p_des^P: the SAME raw planner-frame reference the worker was built
    # from (load_configuration_reference, no frame fit -- mirrors
    # _worker_main exactly). p_des^harness: the live, frame-fit-corrected
    # target the harness itself uses (kwargs["reference"]).
    p_des_P = np.asarray(
        load_configuration_reference(cfg.plan_dir, require_planned_beam_feasible=False).desired_position_m,
        dtype=float,
    )
    p_des_R = p_des_P @ r_fit.T + t_fit
    p_des_harness = np.asarray(kwargs["reference"].desired_position_m, dtype=float)
    discrepancy = float(np.max(np.linalg.norm(p_des_R - p_des_harness, axis=1)))
    print(f"[insertion_anchor] frame-fix check: max_i||R_fit@p_des_i^P+t_fit - p_des_i^harness|| "
          f"= {discrepancy*1e3:.6f}mm (should be numerical noise)")
    assert discrepancy < 1e-6, (
        f"worker/harness target discrepancy after frame fix is {discrepancy*1e3:.3f}mm, "
        f"not numerical noise -- refusing to proceed live"
    )
    return adapter


pf.build_offline_solver = _wrapped_build_offline_solver


def build_or_load_schedule(plan_dir: str, cache_path: str) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[insertion_anchor] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[insertion_anchor] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[insertion_anchor] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    global _DEADLINE_MS, _WORKER, _INSERTION_OFFSET_ABORT_M
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="mpc_delay_aware_insertion_anchor_accumC")
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--deadline-ms", type=float, default=70.0,
                    help="process-isolated worker solve deadline; miss -> hold q_cmd (u=0) for that tick")
    p.add_argument("--insertion-offset-abort-mm", type=float, default=5.0,
                    help="independent safety monitor: abort/hold if |L-L_ref| exceeds this (mm); "
                         "see module docstring for how the default was chosen")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0,
                    help="preflight refuses to proceed unless the camera-measured physical "
                         "insertion is within this of the plan's expected L0 (see common.py's "
                         "check_camera_healthy)")
    p.add_argument("--dry-run", action="store_true",
                    help="real camera+robot connection, real solves, but no servoJ motion "
                         "-- integration smoke test before committing real robot time")
    args = p.parse_args()

    _DEADLINE_MS = args.deadline_ms
    _INSERTION_OFFSET_ABORT_M = args.insertion_offset_abort_mm * 1e-3

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[insertion_anchor] schedule: {schedule.shape} "
          f"(nullq_r700 -> +insertion anchor w_L={_INSERTION_STATE_ANCHOR_WEIGHT}, "
          f"d=2, beta_d=1.0, V_f=0, N={args.horizon}, process-isolated, "
          f"deadline={args.deadline_ms:.0f}ms, insertion-offset-abort={args.insertion_offset_abort_mm:.1f}mm)")

    _WORKER = spawn_and_warm_worker(
        plan_dir=args.plan_dir, schedule_cache=args.schedule_cache,
        control_hz=common.CONTROL_HZ, prediction_horizon=args.horizon,
        joint_velocity_limit_rad_s=args.joint_velocity_limit_rad_s,
        insertion_rate_limit_m_s=_INSERTION_RATE_LIMIT_M_S,
        joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
    )

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir, insertion_tol_mm=args.insertion_tol_mm)

    cfg = pf.CONFIG
    cfg.controller_kind = "mpc_delay_aware"  # gate string this script's own
    # _wrapped_build_offline_solver intercepts -- unchanged, no edits to
    # close_loop_path_follow.py's validity gate needed.
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"
    cfg.mpc_prediction_horizon = args.horizon
    cfg.mpc_use_dare_terminal_cost = False  # V_f=0 -- required

    cfg.feedforward_joint_trajectory = False
    cfg.accumulator_seam = True
    cfg.servo_stream_hz = args.servo_stream_hz
    cfg.force_mpc_feedforward = False

    cfg.joint_velocity_limit_rad_s = args.joint_velocity_limit_rad_s
    cfg.max_joint_step_rad = args.max_joint_step_rad
    cfg.joint_acceleration_limit_rad_s2 = common.JOINT_ACCELERATION_LIMIT_RAD_S2
    cfg.control_insertion = True
    cfg.control_hz = common.CONTROL_HZ
    cfg.max_control_steps = args.max_control_steps
    cfg.initial_insertion_m = l0

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name
    cfg.dry_run = args.dry_run

    print(
        f"[insertion_anchor] plan={args.plan_dir}\n"
        f"[insertion_anchor] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[insertion_anchor] horizon={cfg.mpc_prediction_horizon} d=2 beta_d=1.0 V_f=0 "
        f"servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}\n"
        f"[insertion_anchor] CONTROLLER = gamma=0 exact task-nullspace Q_N, R700, "
        f"insertion anchor w_L={_INSERTION_STATE_ANCHOR_WEIGHT}, "
        f"safety abort |L-Lref|>{args.insertion_offset_abort_mm:.1f}mm"
    )
    try:
        pf.main()
    finally:
        _WORKER.close()


if __name__ == "__main__":
    main()
