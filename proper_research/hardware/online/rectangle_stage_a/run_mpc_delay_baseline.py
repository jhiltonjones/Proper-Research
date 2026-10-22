#!/usr/bin/env python3
"""Condition A for the live old-MPC-vs-delay-aware-MPC development A/B
(2026-09-18): `mpc_ltv_offline` (undelayed BeamOutputTrackingMPC) with
`use_dare_terminal_cost=False`, `prediction_horizon=15` -- EXACTLY the "old"
baseline the offline delay-aware ablation used (mpc_delay_ablation_v2.py's
`old_c`), through execution layer C, genuine per-sample LTV schedule. NOT
q_ablation.py's / run_mpc_ltv.py's historical default (DARE on,
horizon=12) -- V_f=0 here is deliberate, so this is a fair apples-to-apples
match against run_mpc_delay_aware.py (condition B), which structurally
cannot use DARE (see delay_aware_mpc.py).

Also logs the full predicted-beam-position horizon stack per tick (side
file `predicted_beam_positions.jsonl` in the run's output dir) via a thin
wrapper around the adapter `build_offline_solver` returns -- needed to
compute E_pred,+j = RMS||p_meas,k+j - p_hat_k+j|k|| post-hoc, the same
diagnostic condition B logs natively through
`mpc_delay_aware/online_adapter.py`.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_baseline \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/my_run --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedBaselineAdapter,
)
from proper_research.hardware.online import controller_adapters
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.planning.planning_context import build_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import common

# 2026-09-18: moved from a direct in-process BeamOutputTrackingMPC.solve()
# call to a process-isolated worker -- see worker_process.py's docstring.
# The stationary stress test (mpc_stationary_stress_test.py) showed a
# sustained run of per-tick QP solves in the SAME process as the RTDE/
# camera threads permanently freezes the RTDE connection; condition A's
# undelayed MPC has the same per-tick QP cost as condition B's delay-aware
# MPC (both OSQP solves of comparable size) and failed identically in the
# first live batch, so it needs the identical fix, not just condition B.
_real_build_offline_solver = controller_adapters.build_offline_solver
_real_make_output_dir = pf._make_output_dir
_LAST_OUTPUT_DIR: Path | None = None
_DEADLINE_MS = 70.0
_WORKER: MPCWorkerHandle | None = None


def _wrapped_make_output_dir(cfg):
    global _LAST_OUTPUT_DIR
    out = _real_make_output_dir(cfg)
    _LAST_OUTPUT_DIR = out
    return out


pf._make_output_dir = _wrapped_make_output_dir


def spawn_and_warm_worker(
    *, plan_dir: str, schedule_cache: str, control_hz: float, prediction_horizon: int,
    joint_velocity_limit_rad_s: float, insertion_rate_limit_m_s: float,
    joint_acceleration_limit_rad_s2: float, position_error_scale_mm: float,
    position_tracking_weight: float,
) -> MPCWorkerHandle:
    """Spawn + warm the MPC worker BEFORE any RTDE/camera connection opens
    -- see worker_process.py's module docstring on why the lifecycle order
    matters (spawning + warming is itself real, one-time CPU-bound work)."""
    mpc_config_kwargs = dict(
        sample_period_s=1.0 / control_hz, prediction_horizon=int(prediction_horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [-0.05]), state_max=tuple([2.0 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [insertion_rate_limit_m_s]),
        acceleration_limit=tuple([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s]),
    )
    s = float(position_error_scale_mm) * 1.0e-3
    beam_config_kwargs = dict(
        position_error_scale_m=(s, s, s), position_tracking_weight=float(position_tracking_weight),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    print("[mpc_delay_baseline] spawning + warming process-isolated MPC worker "
          "(BEFORE any hardware connection)...")
    worker = MPCWorkerHandle(
        plan_dir=plan_dir, schedule_path=schedule_cache,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
    )
    spawn_ms = (worker.t_worker_ready - worker.t_spawn_start) * 1e3
    print(f"[mpc_delay_baseline] worker ready (spawn+import+construct+warm-up solve = {spawn_ms:.0f}ms)")
    return worker


def _wrapped_build_offline_solver(kind, **kwargs):
    if kind != "mpc_ltv_offline":
        return _real_build_offline_solver(kind, **kwargs)
    assert _LAST_OUTPUT_DIR is not None, "output dir not yet created"
    assert _WORKER is not None, "worker must be spawned+warmed BEFORE pf.main() -- see main()"
    return ProcessIsolatedBaselineAdapter(
        reference=kwargs["reference"], worker=_WORKER, deadline_s=_DEADLINE_MS / 1e3,
        config=kwargs.get("adapter_config"),
        prediction_log_path=str(_LAST_OUTPUT_DIR / "predicted_beam_positions.jsonl"),
    )


pf.build_offline_solver = _wrapped_build_offline_solver


def build_or_load_schedule(plan_dir: str, cache_path: str) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[mpc_delay_baseline] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[mpc_delay_baseline] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[mpc_delay_baseline] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="mpc_delay_baseline_accumC")
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
                    help="real camera+robot connection, real solves, but no servoJ motion")
    args = p.parse_args()

    global _DEADLINE_MS, _WORKER
    _DEADLINE_MS = args.deadline_ms

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[mpc_delay_baseline] schedule: {schedule.shape} (paired A/B, V_f=0, N={args.horizon}, "
          f"process-isolated, deadline={args.deadline_ms:.0f}ms)")

    # Spawn + warm the worker BEFORE any RTDE/camera connection (preflight
    # included -- it opens both) -- see spawn_and_warm_worker's docstring.
    _WORKER = spawn_and_warm_worker(
        plan_dir=args.plan_dir, schedule_cache=args.schedule_cache,
        control_hz=common.CONTROL_HZ, prediction_horizon=args.horizon,
        joint_velocity_limit_rad_s=args.joint_velocity_limit_rad_s,
        insertion_rate_limit_m_s=2.0e-3,
        joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
    )

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir)

    cfg = pf.CONFIG
    cfg.controller_kind = "mpc_ltv_offline"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"
    cfg.mpc_prediction_horizon = args.horizon
    cfg.mpc_use_dare_terminal_cost = False  # V_f=0 -- matches condition B exactly
    # Deliberately NOT touched: mpc_position_error_scale_mm (0.5mm default),
    # mpc_position_tracking_weight (1.0 default), mpc_directional_damping
    # (0.0 default), mpc_wall_avoidance_gain (0.0 default) -- all already
    # equal to mpc_delay_ablation_v2.py's build_configs() convention.

    cfg.feedforward_joint_trajectory = False   # not anchored to the planner
    cfg.accumulator_seam = True                # execution layer C
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
        f"[mpc_delay_baseline] plan={args.plan_dir}\n"
        f"[mpc_delay_baseline] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[mpc_delay_baseline] horizon={cfg.mpc_prediction_horizon} V_f=0 "
        f"servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}"
    )
    try:
        pf.main()
    finally:
        _WORKER.close()


if __name__ == "__main__":
    main()
