#!/usr/bin/env python3
"""OPEN-LOOP-FF through execution layer C: u_k = u_ref,k exactly, streamed
through the accumulator seam + 50Hz interpolation, no correction of any
kind. This is the acceptance test to run FIRST on any new plan/shape --
before trusting any closed-loop controller run through the same execution
layer, confirm the layer itself faithfully reproduces the planned motion.

2026-09-17 rectangle results (two separate runs, neither with every later
fix applied at once -- see the README's "known gaps in today's data" note
before citing either as THE number):
  - insertion-init bug present (cfg.initial_insertion_m left at its 30mm
    dataclass default against the plan's true 25mm start), joint-reader
    healthy: rms=2.38mm, servo_error_rms=4.4mrad (clean, trustworthy --
    confirms the execution layer's own fidelity).
  - insertion fix applied (this script now passes L0 read from the plan,
    so this specific bug cannot recur), but the joint-reader froze for the
    WHOLE run (a since-fixed RobotJointStream bug, see state_stream.py's
    RobotJointStream.reconnect_receive): rms=1.60mm on tip-tracking (a
    real, camera-derived number, unaffected by the frozen joint reads) but
    servo_error is meaningless for that run specifically.
  Both beat the historical absolute-anchored-playback baseline (1.79mm).
  Nobody has re-run this with BOTH fixes present at once -- do that before
  trusting a servo-error number for open-loop-C specifically.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_open_loop_c \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/my_run --run-name openloopff_accumC
"""
import argparse

import proper_research.hardware.online.close_loop_path_follow as pf

from . import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True,
                   help="time_parameterized_configuration_path directory from run_time_parameterization.py")
    p.add_argument("--out-dir", required=True, help="cfg.output_root")
    p.add_argument("--run-name", default="openloopff_accumC")
    p.add_argument("--joint-velocity-limit-rad-s", type=float, default=common.JOINT_VELOCITY_LIMIT_RAD_S)
    p.add_argument("--max-joint-step-rad", type=float, default=common.MAX_JOINT_STEP_RAD)
    p.add_argument("--servo-stream-hz", type=float, default=common.SERVO_STREAM_HZ)
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true",
                   help="skip the health checks + reset (only if you just ran one)")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0,
                   help="preflight refuses to proceed unless the camera-measured physical "
                        "insertion is within this of the plan's expected L0 (see common.py's "
                        "check_camera_healthy) -- matches the same flag on "
                        "run_mpc_delay_aware_insertion_anchor.py / run_inv_7dof_delay_aware.py, "
                        "added here for consistency (this script previously had no override).")
    args = p.parse_args()

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir, insertion_tol_mm=args.insertion_tol_mm)

    cfg = pf.CONFIG
    cfg.controller_kind = "open_loop_ff"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"

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
    cfg.initial_insertion_m = l0   # from the plan, NOT the 30mm dataclass default

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name

    print(
        f"[open_loop_c] plan={args.plan_dir}\n"
        f"[open_loop_c] q0={q0.tolist()} L0={l0*1000:.2f}mm\n"
        f"[open_loop_c] servo_stream_hz={cfg.servo_stream_hz} "
        f"joint_velocity_limit_rad_s={cfg.joint_velocity_limit_rad_s} "
        f"max_joint_step_rad={cfg.max_joint_step_rad}"
    )
    pf.main()


if __name__ == "__main__":
    main()
