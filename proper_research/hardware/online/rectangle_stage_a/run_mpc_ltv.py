#!/usr/bin/env python3
"""mpc_ltv_offline (default Q/Qp/R/Rd/N -- not tuned for this comparison)
through execution layer C, with a GENUINE per-sample-relinearised Jacobian
schedule.

Bug this works around: with `PathFollowConfig.jacobian_source` left at its
default ("analytical_beam"), the harness builds `AnalyticalBeamJacobianProvider`,
whose `__call__` ignores its `state` argument entirely (frozen at
construction -- confirmed directly in `close_loop_tip_control.py`: `return
self._jac.copy()`). `mpc_variants.precompute_schedule` calls that once per
reference sample, so the "LTV" schedule built from it is really N identical
copies of one frozen Jacobian -- mathematically `mpc_lti` under an
`mpc_ltv_offline` label, not real per-sample relinearisation. Confirmed live
2026-09-17 by the schedule-build time alone: the frozen provider's "build"
took ~7s (no real work), the genuine one ~90-150s (one real beam solve per
sample) -- and by the resulting Jacobian condition numbers being constant
(~62700 throughout) vs genuinely varying (37k -> 5.5k -> 45k across the
rectangle).

Fix: build the schedule from `beam_jacobian_providers.from_model_bundle`
(the same state-dependent source `naive_inverse_jacobian_ltv` is documented
to use for a fair Jacobian-source match) and inject it via the harness's
existing `_SCHEDULE_OVERRIDE` module global -- the mechanism
`close_loop_path_follow.py` already supports for exactly this (see its
`_schedule_applies` check).

2026-09-17 rectangle result (genuine schedule): E_RMS(active)=1.871mm,
E_max(active)=5.860mm, servo_error_RMS=16.95mrad, P(rate sat)=23.3% --
still worse than both open-loop-C (1.785mm) and tuned INV (kp=0.6,kn=0:
1.342mm), and the Jacobian fix barely moved these numbers versus the frozen-
schedule run (1.979mm/17.73mrad/29.5%) -- so the frozen-Jacobian bug was
real but NOT the primary driver of MPC's saturation/servo-lag signature.
See this package's README for the Q-ablation that followed (falsified the
"state/posture tracking is the cause" hypothesis).

The schedule build is slow (~90-150s, one real beam solve per reference
sample) -- this script caches it to `--schedule-cache` so repeat runs on the
same plan skip straight to the robot.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_mpc_ltv \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/my_run \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
import argparse
import os

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.planning.planning_context import build_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import common


def build_or_load_schedule(plan_dir: str, cache_path: str | None) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[mpc_ltv] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[mpc_ltv] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[mpc_ltv] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="mpc_ltv_accumC")
    p.add_argument("--schedule-cache", default=None,
                   help="path to save/load the genuine LTV Jacobian schedule (skips "
                        "the ~90-150s rebuild on repeat runs against the same plan)")
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[mpc_ltv] schedule: {schedule.shape}, "
          f"cond(J[0][:, :6])={np.linalg.cond(schedule[0][:, :6]):.1f}, "
          f"cond(J[-1][:, :6])={np.linalg.cond(schedule[-1][:, :6]):.1f}")

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir)

    cfg = pf.CONFIG
    cfg.controller_kind = "mpc_ltv_offline"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"
    cfg.mpc_prediction_horizon = args.horizon
    # Deliberately NOT touched: mpc_position_error_scale_mm,
    # mpc_position_tracking_weight, mpc_use_dare_terminal_cost,
    # mpc_*_weight_override, mpc_directional_damping, mpc_conditioning_fix --
    # all left at their dataclass defaults (Q/Qp/R/Rd unchanged; see the Q-
    # ablation script if you want to test removing state/posture tracking).

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

    print(
        f"[mpc_ltv] plan={args.plan_dir}\n"
        f"[mpc_ltv] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[mpc_ltv] horizon={cfg.mpc_prediction_horizon} servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}"
    )
    pf.main()


if __name__ == "__main__":
    main()
