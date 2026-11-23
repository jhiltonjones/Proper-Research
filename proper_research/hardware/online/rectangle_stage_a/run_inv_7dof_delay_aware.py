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
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _resolve_robot_kinematics,
)
from proper_research.planning.planning_context import make_robot_config

from . import common

# 2026-11-19: SAME robot-kinematics resolution the MPC rectangle runner
# uses (run_mpc_delay_aware_insertion_anchor.py), needed here only to
# build the shared external measured-state safety check's magnet FK --
# see main()'s own comment at the pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE
# assignment below.
_robot_kin = _resolve_robot_kinematics(make_robot_config())

# 2026-10-23: physical rig's middle marker is permanently gone (removed
# 2026-09-29, see common.py's patch_state_stream_config_two_markers) --
# this script's own live-tracking warmup needs 3-4 markers by default and
# fails outright against the current rig without this patch. Must run
# before anything else reassigns pf.StateStreamConfig.
common.patch_state_stream_config_two_markers(pf)


def build_or_load_schedule(plan_dir: str, cache_path: str) -> np.ndarray:
    # Z4 (2026-11-10): consolidated into common.py -- see its docstring.
    return common.build_or_load_schedule(plan_dir, cache_path, label="inv_7dof_delay_aware")


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
    p.add_argument(
        "--servo-lookahead-s", type=float, default=None,
        help="2026-11-24 servo pilot (ported from run_mpc_delay_aware_insertion_anchor.py "
             "for a like-for-like A-E comparison): override close_loop_path_follow.py's "
             "PathFollowConfig.servo_lookahead_s (production default 0.20s, unchanged if "
             "omitted). UR's documented valid range is [0.03, 0.2]s; out-of-range values "
             "are refused, never silently clamped or sent to the real controller.",
    )
    p.add_argument(
        "--servo-gain", type=int, default=None,
        help="2026-11-24 servo pilot (ported for parity with the MPC runner): override "
             "PathFollowConfig.servo_gain (production default 200, unchanged if omitted). "
             "UR's documented valid range is [100, 2000]; out-of-range values are refused.",
    )
    p.add_argument("--max-control-steps", type=int, default=800)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--insertion-tol-mm", type=float, default=3.0,
                    help="preflight refuses to proceed unless the camera-measured physical "
                         "insertion is within this of the plan's expected L0")
    p.add_argument("--max-tracking-error-mm", type=float, default=0.0,
                    help="Z4b step 2 (2026-11-11): ported from the vessel runner -- stop the "
                         "run and report failure if the measured beam-tip tracking error ever "
                         "exceeds this. Default 0.0 means 'let close_loop_path_follow.main() "
                         "fill in the controller-aware default' (15mm for this INV-7 path); "
                         "pass a nonzero value to override.")
    p.add_argument("--dry-run", action="store_true",
                    help="camera/robot connect and every tick solves normally, but no servoJ/"
                         "advancer motion commands are sent -- for validating the wiring before "
                         "the first live pilot")
    p.add_argument("--insertion-command-min-mm", type=float, default=5.0,
                    help="2026-11-19: SAME shared online insertion-command clip bound as the "
                         "MPC rectangle runner's own flag of this name -- see its help text. "
                         "Default (5mm) matches the prior hardcoded value; pass 30.0 to "
                         "enforce this plan's intended envelope.")
    p.add_argument("--insertion-command-max-mm", type=float, default=200.0,
                    help="see --insertion-command-min-mm. Default (200mm); pass 50.0 to "
                         "enforce this plan's intended envelope.")
    p.add_argument("--measured-clearance-abort-mm", type=float,
                    default=common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM,
                    help="2026-11-19: SAME shared measured-state external safety check the MPC "
                         "rectangle runner uses (run_mpc_delay_aware_insertion_anchor.py), now "
                         "wired to this reactive controller via OfflineJointControllerAdapter's "
                         "pre-solve hold-and-abort -- no predictive constraint is added to the "
                         "controller itself. Default is the already-reviewed "
                         "MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM (230mm), same as the MPC runner.")
    p.add_argument("--protective-config-reviewed", action="store_true",
                    help="2026-11-20: ported from run_mpc_delay_aware_insertion_anchor.py -- "
                         "this script shares the SAME --measured-clearance-abort-mm external "
                         "safety check (see that flag's own docstring above) but, until now, "
                         "had no gate requiring a human to have reviewed it before a real start. "
                         "REQUIRED to start a real (non-dry-run) run. States that a human "
                         "reviewed --measured-clearance-abort-mm's ACTUAL value for this run "
                         "(printed at startup) before starting. Without this flag, the runner "
                         "REFUSES to start rather than silently running protected by an "
                         "unreviewed value or running with no protection at all.")
    p.add_argument("--experimental-virtual-boundary-hold-radius-mm", type=float, default=None,
                    help="2026-11-20, constraint study (demonstration of what a non-"
                         "predictive controller does near a virtual boundary it has no "
                         "optimizer-level awareness of): ADDITIVE, non-destructive. "
                         "Default None -- behavior is byte-identical to not passing this "
                         "flag. When set, any tick whose measured magnet-to-beam-base "
                         "clearance drops below this radius gets its command ZEROED for "
                         "that tick only -- held, not aborted: no abort_reason is set, the "
                         "run is NOT stopped, and the controller resumes normal reactive "
                         "commands the next tick the clearance recovers above this radius. "
                         "The REAL --measured-clearance-abort-mm check is completely "
                         "unchanged and always takes precedence if both would apply on the "
                         "same tick. This does not give the controller any predictive/"
                         "optimizer-level awareness of the boundary -- it is a reactive, "
                         "after-the-fact freeze, demonstrating the ABSENCE of predictive "
                         "handling, not a substitute for it.")
    args = p.parse_args()

    # 2026-11-20: ported verbatim from run_mpc_delay_aware_insertion_anchor.py's
    # own identical check -- see that file's own comment for why this must
    # be placed immediately after args = p.parse_args() (args must exist)
    # and before any schedule/worker setup (fail before anything else
    # happens, not partway through).
    if not args.dry_run and not args.protective_config_reviewed:
        raise SystemExit(
            "[inv_7dof] REFUSING to start: the measured-clearance breach detector's abort "
            f"radius for THIS run ({args.measured_clearance_abort_mm:.1f}mm, from "
            "--measured-clearance-abort-mm) has not been confirmed reviewed. Pass "
            "--protective-config-reviewed only after a human has reviewed this run's ACTUAL "
            "value (printed above/below) -- this flag is never implied by any other argument."
        )

    schedule = build_or_load_schedule(args.plan_dir, args.schedule_cache)
    pf._SCHEDULE_OVERRIDE = schedule

    # 2026-11-19: shared external measured-state safety check (see
    # proper_research.controllers.mpc_delay_aware.external_safety_check
    # and OfflineJointControllerAdapter/close_loop_path_follow.py's
    # _TRIM_KINDS wiring, both added this session). Uses the SAME FK and
    # beam-base point the MPC rectangle runner uses, so the two
    # conditions see an identical geometric monitor -- only the control
    # law differs, matching the inv-vs-MPC comparison's intent.
    _beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(1, 3)

    def _magnet_position_fn(state7: np.ndarray) -> np.ndarray:
        # convention matches run_inverse_jacobian_online_vessel.py's own
        # magnet_position_fn(state7) docstring: the magnet never depends
        # on insertion length, only the 6 joints.
        q6 = np.asarray(state7, dtype=float)[:6]
        fk = urik.forward_kinematics(q6.reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
        return np.asarray(fk.T_R_target[:3, 3], dtype=float)

    pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = dict(
        magnet_position_fn=_magnet_position_fn,
        magnet_exclusion_lumen_C_m=_beam_base_xyz_r,
        magnet_exclusion_radius_m=args.measured_clearance_abort_mm * 1e-3,
        virtual_boundary_hold_radius_m=(
            args.experimental_virtual_boundary_hold_radius_mm * 1e-3
            if args.experimental_virtual_boundary_hold_radius_mm is not None else None
        ),
    )
    if args.experimental_virtual_boundary_hold_radius_mm is not None:
        print(f"[inv_7dof_delay_aware] CONSTRAINT STUDY virtual-boundary HOLD (not abort) "
              f"ENABLED at {args.experimental_virtual_boundary_hold_radius_mm:.1f}mm -- "
              f"command zeroed only while violated (and the real "
              f"{args.measured_clearance_abort_mm:.1f}mm abort radius is not also "
              f"violated); run continues. No predictive handling is added to the "
              f"controller -- this is a reactive, after-the-fact freeze only.")
    print(f"[inv_7dof_delay_aware] shared external measured-state safety check: "
          f"abort_radius={args.measured_clearance_abort_mm:.1f}mm (reactive controller -- "
          f"no predictive constraint added; a trip holds the command and enters the shared "
          f"main loop's abort/shutdown path, same as the MPC runner's own monitor)")
    print(f"[inv_7dof_delay_aware] schedule: {schedule.shape} (paired-A/B mode: J_{{k+3}})")

    if args.skip_preflight:
        q0, l0 = common.load_plan_initial_state(args.plan_dir)
    else:
        q0, l0 = common.preflight(args.plan_dir, insertion_tol_mm=args.insertion_tol_mm)

    cfg = pf.CONFIG
    cfg.controller_kind = "inv_7dof_delay_aware"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"
    cfg.max_tracking_error_m = args.max_tracking_error_mm / 1000.0
    cfg.insertion_command_min_m = args.insertion_command_min_mm / 1000.0
    cfg.insertion_command_max_m = args.insertion_command_max_mm / 1000.0

    # 2026-11-24 servo pilot (ported from run_mpc_delay_aware_insertion_anchor.py for
    # a like-for-like A-E comparison): additive overrides, refused outside UR's own
    # documented servoj() ranges, never silently clamped before reaching the controller.
    if args.servo_lookahead_s is not None:
        if not (0.03 <= args.servo_lookahead_s <= 0.2):
            raise SystemExit(
                f"[inv_7dof_delay_aware] --servo-lookahead-s {args.servo_lookahead_s} is "
                f"outside UR's documented servoj() lookahead_time range [0.03, 0.2]s -- refusing."
            )
        cfg.servo_lookahead_s = float(args.servo_lookahead_s)
        print(f"[inv_7dof_delay_aware] servo_lookahead_s OVERRIDDEN to {cfg.servo_lookahead_s}s "
              f"(production default 0.20s)")
    if args.servo_gain is not None:
        if not (100 <= args.servo_gain <= 2000):
            raise SystemExit(
                f"[inv_7dof_delay_aware] --servo-gain {args.servo_gain} is outside UR's "
                f"documented servoj() gain range [100, 2000] -- refusing."
            )
        cfg.servo_gain = int(args.servo_gain)
        print(f"[inv_7dof_delay_aware] servo_gain OVERRIDDEN to {cfg.servo_gain} "
              f"(production default 200)")

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
