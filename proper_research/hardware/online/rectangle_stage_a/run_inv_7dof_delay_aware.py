#!/usr/bin/env python3
"""Matched-authority 7DOF inverse, through execution layer C (2026-09-21).

See `proper_research/controllers/inverse_jacobian_7dof_delay_aware.py` for
the architecture: byte-for-byte the SAME delay-aware preview as
run_inv_2dof_delay_aware.py (d_k, z_hat_base, e_pred, realization stage
r=k+3), the ONLY difference is that the feedback-allocation step is a
normalized-DLS pinv over all 7 actuators (joints + insertion) instead of a
6-column joint-only pinv -- so `u_L = u_ref,L + delta_u_L` instead of
`u_L = u_ref,L` fixed. Built to remove the "MPC has 7 feedback actuators,
inverse only has 6" confound from the inv-vs-MPC comparison. Validated
offline (unit tests + counterfactual replay against real live states) before
this script existed -- see close_loop_logs/myrun/mechanism_analysis_2026-09-21/
for that offline validation.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware \\
        --plan-dir plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/myrun --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
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
        print(f"[inv_7dof_delay_aware] loading cached genuine LTV schedule from {cache_path}")
        return np.load(cache_path)
    print("[inv_7dof_delay_aware] building genuine from_model_bundle Jacobian schedule "
          "(one beam solve per reference sample, ~90-150s)...")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(bundle=bundle, controller_pack=controller_pack, contact=False)
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
        print(f"[inv_7dof_delay_aware] schedule cached -> {cache_path}")
    return schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default=None, help="default: inv_7dof_delay_aware_kp<kp>")
    p.add_argument("--schedule-cache", required=True,
                    help="genuine schedule for SJ, or /tmp/rectangle_frozen_J0_schedule.npy for FJ")
    p.add_argument("--kp", type=float, default=1.0)
    p.add_argument("--damping", type=float, default=5.0e-2,
                    help="INV-6-equivalent damping in unnormalized (rad) units; internally "
                         "rescaled by su-joint so insertion-disabled behaviour matches INV-6 "
                         "exactly -- see the controller module docstring")
    p.add_argument("--su-joint", type=float, default=0.05, help="rad/s, input_error_scale convention")
    p.add_argument("--su-insertion", type=float, default=5.0e-3, help="m/s, input_error_scale convention")
    p.add_argument("--max-l-step-m", type=float, default=2.0e-4,
                    help="dt * insertion_velocity_limit_m_s, matched to MPC's |u_L|<=2mm/s")
    p.add_argument("--enable-decomposition-logging", action="store_true")
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0,
                    help="preflight refuses to proceed unless the camera-measured physical "
                         "insertion is within this of the plan's expected L0")
    p.add_argument("--dry-run", action="store_true",
                    help="camera/robot connect and every tick solves normally, but no servoJ/"
                         "advancer motion commands are sent -- for validating the wiring before "
                         "the first live pilot")
    args = p.parse_args()

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[inv_7dof_delay_aware] schedule: {schedule.shape} (paired-A/B mode: J_{{k+3}})")

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir, insertion_tol_mm=args.insertion_tol_mm)

    cfg = pf.CONFIG
    cfg.controller_kind = "inv_7dof_delay_aware"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"

    cfg.trim_kp = args.kp
    cfg.trim_kn = 0.0
    cfg.trim_damping = args.damping
    cfg.trim_su_joint = args.su_joint
    cfg.trim_su_insertion = args.su_insertion
    cfg.trim_max_l_step_m = args.max_l_step_m
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
    cfg.run_name = args.run_name or f"inv_7dof_delay_aware_kp{args.kp:g}"
    cfg.dry_run = args.dry_run

    print(
        f"[inv_7dof_delay_aware] plan={args.plan_dir}\n"
        f"[inv_7dof_delay_aware] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[inv_7dof_delay_aware] kp={args.kp} damping={args.damping} "
        f"su_joint={args.su_joint} su_insertion={args.su_insertion} max_l_step_m={args.max_l_step_m}\n"
        f"[inv_7dof_delay_aware] servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}"
    )
    pf.main()


if __name__ == "__main__":
    main()
