#!/usr/bin/env python3
"""INV 2DOF nominal + non-integrating trim controller, through execution
layer C. See `proper_research/controllers/inverse_jacobian_2dof_trim.py`
for the architecture and the windup diagnosis that motivated it.

2026-09-17 rectangle kp/kn ablation (kn=0 fixed first, from the kn ablation
below; then kp swept at kn=0):

    kn: 1.0 -> 0.5 -> 0.25 -> 0.0   (kp=1.0 fixed)
      E_RMS(active)   1.98 -> 1.59 -> 1.65 -> 1.55 mm
      E_servo_RMS     31.1 -> 15.3 -> 14.7 -> 11.8 mrad
      P(rate sat)      n/a -> 35.3 -> 29.3 -> 20.5 %

    kp: 0.0 -> 0.3 -> 0.6 -> 1.0    (kn=0 fixed)
      E_RMS(active)   1.785 -> 1.498 -> 1.342 -> 1.552 mm
      E_max(active)   3.865 -> 3.119 -> 2.717 -> 4.657 mm
      P(rate sat)     0.0%  -> 0.0%  -> 0.0%  -> 20.5%

kp=0.6, kn=0 is the clean U-shaped optimum: beats open-loop-C (1.785mm) by
~25% RMS / ~30% max, zero rate-limit saturation, zero beam-error spikes
above 4mm. That is the default here and the frozen Stage-A development
point as of 2026-09-17 -- see this package's README for the full context
and why kp=1 alone is NOT recommended (windup: pending servo error escalated
in lockstep with a high c_wind correlation during the harder transit
sections, confirmed via an offline replay of the logged run).

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_inv_2dof_trim \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/my_run --kp 0.6 --kn 0.0

Pass --schedule-cache to use the genuine path-indexed LTV schedule (J_k)
instead of the default live per-state jacobian_provider -- for a paired A/B
against run_inv_2dof_delay_aware.py (J_{k+3}), so the two conditions differ
ONLY in which index they read, not also in Jacobian source. Omit it for the
original (2026-09-17-validated) behaviour, unchanged.
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


def build_or_load_schedule(plan_dir: str, cache_path: str) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        print(f"[inv_2dof_trim] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[inv_2dof_trim] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[inv_2dof_trim] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default=None,
                   help="default: inv_2dof_trim_kp<kp>_kn<kn>")
    p.add_argument("--kp", type=float, default=0.6)
    p.add_argument("--kn", type=float, default=0.0)
    p.add_argument("--damping", type=float, default=5.0e-2)
    p.add_argument("--q-trim-max", type=float, default=0.03)
    p.add_argument("--enable-decomposition-logging", action="store_true",
                   help="log ||dq_task||, ||dq_null||, ||dq_trim||, |dL_null| etc "
                        "per tick -- retrieve via close_loop_path_follow.LAST_SOLVER"
                        ".controller.log after this script returns")
    p.add_argument("--schedule-cache", default=None,
                   help="use the genuine path-indexed LTV schedule (J_k) instead of the "
                        "default live per-state jacobian_provider -- see module docstring")
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    if args.schedule_cache:
        schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
        pf._SCHEDULE_OVERRIDE = schedule
        print(f"[inv_2dof_trim] schedule: {schedule.shape} (paired-A/B mode: J_k)")

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir)

    cfg = pf.CONFIG
    cfg.controller_kind = "inv_2dof_trim"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"

    cfg.trim_kp = args.kp
    cfg.trim_kn = args.kn
    cfg.trim_damping = args.damping
    cfg.trim_q_max = args.q_trim_max
    cfg.trim_enable_logging = args.enable_decomposition_logging

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
    cfg.run_name = args.run_name or f"inv_2dof_trim_kp{args.kp:g}_kn{args.kn:g}"

    print(
        f"[inv_2dof_trim] plan={args.plan_dir}\n"
        f"[inv_2dof_trim] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[inv_2dof_trim] kp={args.kp} kn={args.kn} damping={args.damping} q_trim_max={args.q_trim_max}\n"
        f"[inv_2dof_trim] servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}"
    )
    pf.main()


if __name__ == "__main__":
    main()
