#!/usr/bin/env python3
"""Live validation of the frozen exact-task-nullspace / gamma=0 / R700
delay-aware MPC (2026-09-20), against the same M0 infrastructure as
`run_mpc_delay_aware.py`.

This is a THIN WRAPPER, not a new implementation: the actual QP
formulation is `StagewiseTaskNullspaceDelayAwareMPC(gamma=0.0, ...)`
(`proper_research/controllers/mpc_delay_aware/stagewise_task_nullspace.py`),
the exact class that:
  - passed the gamma=1<->M0 bit-for-bit equivalence unit test
    (`stagewise_task_nullspace_self_test.py`),
  - was validated in the offline replay campaign against the 268-tick
    "good run" log (axis-resolved E_chosen improvement, physical
    insertion/joint task-direction decomposition, and the Q_N-ablation
    showing Q_N is NOT redundant against full u_ref).

The ONE structural change from `run_mpc_delay_aware.py` (M0) is:
    Q_full -> Q_N only (gamma=0, exact task-nullspace projector)
    R -> 700 * R0 (input_tracking_weight = 7.0)
Everything else is frozen identically to M0: Qp, Rd (input_increment_weight
= 1e-3, with input_increment_scale EXPLICITLY pinned -- never left to
auto-derive from a relaxed acceleration_limit, see
`ConfigurationMPCConfig.effective_input_increment_scale` and the bug this
project hit doing that during offline FREE-BOTH diagnostics), d=2,
beta_d=1.0, V_f=0, N=15, insertion authority (s_u,L=5mm/s, |u_L|<=2mm/s
unchanged -- do NOT reintroduce either insertion ablation), execution-C
accumulator seam, process-isolated worker architecture, and the same
rectangle_10x15mm_skipglobal_2026-09-12 plan/schedule used for the
successful M0 and kp=1 inverse runs.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_nullq_r700 \\
        --plan-dir plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --schedule-cache /path/to/skipglobal_genuine_ltv_schedule.npy
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

# Frozen controller spec (see module docstring) -- kept as module-level
# constants, not CLI flags, so this run cannot silently drift from the
# offline-validated formulation.
_GAMMA = 0.0
_INPUT_TRACKING_WEIGHT = 7.0          # 700 * default 1.0e-2 = R700
_INPUT_INCREMENT_WEIGHT = 1.0e-3      # unchanged from M0
_INSERTION_RATE_LIMIT_M_S = 2.0e-3    # unchanged -- do NOT rescale (s_u,L stays 5mm/s via defaults)


def _wrapped_make_output_dir(cfg):
    global _LAST_OUTPUT_DIR
    out = _real_make_output_dir(cfg)
    _LAST_OUTPUT_DIR = out
    return out


pf._make_output_dir = _wrapped_make_output_dir


_DEADLINE_MS = 70.0
_WORKER: MPCWorkerHandle | None = None


def spawn_and_warm_worker(
    *, plan_dir: str, schedule_cache: str, control_hz: float, prediction_horizon: int,
    joint_velocity_limit_rad_s: float, insertion_rate_limit_m_s: float,
    joint_acceleration_limit_rad_s2: float, position_error_scale_mm: float,
    position_tracking_weight: float,
) -> MPCWorkerHandle:
    """Spawn + warm the MPC worker BEFORE any RTDE/camera connection opens
    -- identical lifecycle ordering to `run_mpc_delay_aware.py`. Builds the
    worker's M0 (old/new) controllers with UNCHANGED config, plus the
    nullspace/R700 controller with ONLY input_tracking_weight raised to
    7.0 -- the input_increment_scale is explicitly pinned here (not left
    to auto-derive) so Rd's normalization is guaranteed identical to what
    the offline replay used."""
    dt = 1.0 / control_hz
    mpc_config_kwargs = dict(
        sample_period_s=dt, prediction_horizon=int(prediction_horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]), state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [insertion_rate_limit_m_s]),
        acceleration_limit=tuple([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s]),
    )
    # Same physical bounds as M0's config, EXPLICITLY pinning
    # input_increment_scale to dt*acceleration_limit -- the value the
    # offline replay used -- rather than relying on
    # effective_input_increment_scale's auto-derivation coincidentally
    # matching it (see this module's docstring / the offline FREE-BOTH bug
    # this project hit doing exactly that with a relaxed acceleration_limit).
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
    print("[nullq_r700] spawning + warming process-isolated MPC worker "
          "(BEFORE any hardware connection); building M0 + nullspace/R700...")
    worker = MPCWorkerHandle(
        plan_dir=plan_dir, schedule_path=schedule_cache,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True,
        nullspace_r700_config_kwargs=nullspace_r700_config_kwargs,
    )
    spawn_ms = (worker.t_worker_ready - worker.t_spawn_start) * 1e3
    print(f"[nullq_r700] worker ready (spawn+import+construct+warm-up+projector-checks = {spawn_ms:.0f}ms)")
    print(f"[nullq_r700] CONTROLLER CONFIRMED: state cost = exact task-nullspace only (gamma={_GAMMA}), "
          f"R multiplier = 700 (input_tracking_weight={_INPUT_TRACKING_WEIGHT}), "
          f"input_increment_weight={_INPUT_INCREMENT_WEIGHT}, "
          f"input_increment_scale(pinned)={increment_scale}")
    return worker


def _wrapped_build_offline_solver(kind, **kwargs):
    global _METADATA_WRITTEN
    if kind != "mpc_delay_aware":
        return _real_build_offline_solver(kind, **kwargs)
    assert _LAST_OUTPUT_DIR is not None, "output dir not yet created"
    assert _WORKER is not None, "worker must be spawned+warmed BEFORE pf.main() -- see main()"
    cfg = pf.CONFIG
    if not _METADATA_WRITTEN:
        meta = {
            "controller_variant": "nullspace_Q_R700",
            "gamma": _GAMMA,
            "input_tracking_weight": _INPUT_TRACKING_WEIGHT,
            "input_increment_weight": _INPUT_INCREMENT_WEIGHT,
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
        }
        (_LAST_OUTPUT_DIR / "controller_metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"[nullq_r700] wrote controller_metadata.json -> {_LAST_OUTPUT_DIR}")
        _METADATA_WRITTEN = True
    return ProcessIsolatedDelayAwareAdapter(
        reference=kwargs["reference"], worker=_WORKER, deadline_s=_DEADLINE_MS / 1e3,
        joint_velocity_limit_rad_s=float(cfg.joint_velocity_limit_rad_s),
        max_joint_step_rad=float(cfg.max_joint_step_rad),
        config=kwargs.get("adapter_config"),
        prediction_log_path=str(_LAST_OUTPUT_DIR / "predicted_beam_positions.jsonl"),
        controller_kind="nullspace_r700",
        controller_label="mpc_delay_aware_nullspace_r700_process_isolated",
    )


pf.build_offline_solver = _wrapped_build_offline_solver


def build_or_load_schedule(plan_dir: str, cache_path: str) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[nullq_r700] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[nullq_r700] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[nullq_r700] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    global _DEADLINE_MS, _WORKER
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="mpc_delay_aware_nullq_r700_accumC")
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--deadline-ms", type=float, default=70.0,
                    help="process-isolated worker solve deadline; miss -> hold q_cmd (u=0) for that tick")
    p.add_argument("--dry-run", action="store_true",
                    help="real camera+robot connection, real solves, but no servoJ motion "
                         "-- integration smoke test before committing real robot time")
    args = p.parse_args()

    _DEADLINE_MS = args.deadline_ms

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[nullq_r700] schedule: {schedule.shape} "
          f"(M0->Q_N+gamma0+R700, d=2, beta_d=1.0, V_f=0, N={args.horizon}, process-isolated, "
          f"deadline={args.deadline_ms:.0f}ms)")

    # Spawn + warm the worker BEFORE any RTDE/camera connection (preflight
    # included -- it opens both).
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
        q0, l0 = common.preflight(args.plan_dir)

    cfg = pf.CONFIG
    cfg.controller_kind = "mpc_delay_aware"  # gate string this script's own
    # _wrapped_build_offline_solver intercepts -- unchanged from M0's runner
    # so no edits to close_loop_path_follow.py's validity gate are needed;
    # the nullspace/R700 controller is selected purely by which run script
    # is invoked, not by a new controller_kind value.
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
        f"[nullq_r700] plan={args.plan_dir}\n"
        f"[nullq_r700] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[nullq_r700] horizon={cfg.mpc_prediction_horizon} d=2 beta_d=1.0 V_f=0 "
        f"servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}\n"
        f"[nullq_r700] CONTROLLER = gamma=0 exact task-nullspace Q_N, R700 "
        f"(input_tracking_weight={_INPUT_TRACKING_WEIGHT}), insertion s_u,L=5mm/s unchanged"
    )
    try:
        pf.main()
    finally:
        _WORKER.close()


if __name__ == "__main__":
    main()
