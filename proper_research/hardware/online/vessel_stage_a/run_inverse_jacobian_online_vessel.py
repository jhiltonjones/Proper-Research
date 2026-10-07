#!/usr/bin/env python3
"""Damped inverse-Jacobian closed-loop run for a vessel plan -- the naive-
controller comparator against run_mpc_delay_aware_vessel.py's MPC, built on
the SAME precomputed (S,3,7) Jacobian schedule MPC uses (--schedule-cache),
not a live per-tick evaluation.

2026-10-07 history: this script originally built a genuinely state-dependent
live jacobian_provider (from_model_bundle, "accurate" mode, re-evaluated
every control tick from the measured state). That was found live to be far
too slow for closed-loop control: "accurate" mode is a full quasistatic +
implicit-sensitivity solve, ~1-3s/call, versus the 100ms/tick budget at
control_hz=10 -- the reference (which advances on wall-clock time,
independent of whether the robot actually kept up) outran the robot by 19
samples in two real control ticks, tripping max_tracking_error_m almost
immediately. Switched to the SAME offline schedule mechanism MPC uses
instead (controller_kind="naive_inverse_jacobian_ltv",
close_loop_path_follow._SCHEDULE_OVERRIDE) -- this also means the Jacobian
SOURCE is now provably identical between the MPC and naive-controller
comparison runs, isolating the control law (one-step resolved-rate vs
finite-horizon QP) as the only variable, exactly why
InverseJacobianBeamController's own module docstring already describes
"inverse-LTV" as MPC-LTV's matched baseline. `run_inverse_jacobian_vessel.py`
already does this for the no-contact-only case; this script adds
--contact/--no-contact (which that one lacks) plus the magnet-exclusion/
z-workspace clip-vs-hold comparison below.

Purpose (explicit hypothesis under test): does the MPC's in-QP constraint
formulation genuinely respect hard physical limits (magnet-exclusion-radius,
magnet-z-workspace) in a way a naive resolved-rate controller does not?
Before this script, the naive inverse-Jacobian controller had NO protection
against either constraint beyond the toothless +-2pi joint box and the
generic velocity/accel clip every controller kind shares. `--magnet-
protection` selects between THREE genuinely different ways of answering
that question, not just on/off:
  - "clip" (default): InverseJacobianBeamController's own anticipatory
    half-space projection (2026-10-07,
    InverseJacobianBeamController._clip_magnet_constraints) -- the SAME
    linearized constraints delay_aware_mpc.py's `_configure_magnet_
    exclusion`/`_configure_magnet_workspace` add to the MPC's QP, but
    enforced as a closed-form projection instead of a QP inequality. The
    controller "handles" the constraint and keeps moving -- useful to show
    THIS specific fix closes the gap, but it makes the naive controller
    look smarter than it actually is about the constraint itself.
  - "hold": the controller's own math stays COMPLETELY naive -- no clip at
    all, it keeps trying to command motion into the excluded region every
    tick it wants to -- but close_loop_path_follow._COMMAND_SAFETY_GATE
    (make_magnet_exclusion_hold_gate, 2026-10-07) sits between the
    controller's raw output and the robot and HOLDS position (zero
    command) on any tick that raw output would violate the constraint.
    This is the cleaner demonstration of "the naive controller does not
    understand this constraint at all, unlike MPC" -- the robot visibly
    stalls against the boundary instead of being quietly corrected.
  - "off": the raw, unaware command goes straight to the robot. NOT a safe
    hardware baseline for this constraint specifically -- use "hold" to
    see the naive controller's true behaviour without the physical risk.

Mirrors run_mpc_delay_aware_vessel.py's vessel-specific wiring (preflight,
frame-registration bypass, StateStreamConfig patch, workspace box, magnet
z-bounds derivation) and run_inverse_jacobian_vessel.py's controller
plumbing (close_loop_path_follow._SCHEDULE_OVERRIDE for the schedule,
plus this project's sibling overrides added 2026-10-07:
_MAGNET_CONSTRAINT_CLIP_OVERRIDE for the clip kwargs, _COMMAND_SAFETY_GATE
for the hold gate) -- reuses all three rather than re-deriving any of them.

What this does NOT have (known, documented limitation -- not silently
missing): no per-tick REACTIVE monitor for insertion-offset drift
(ProcessIsolatedDelayAwareAdapter's insertion_offset_abort_m is MPC-only
machinery this controller never instantiates) and no reactive magnet-
exclusion/z-bounds monitor from MEASURED joints after a command is applied
(the MPC gets this as a second line of defense; here "clip"/"hold" are each
the ONLY line of defense against those two, same as the LTV script before
it -- "hold"'s gate IS itself the line of defense, not an additional one on
top of a clip). The generic close_loop_path_follow.py tcp_out_of_workspace
box and the opt-in max_tracking_error_m check both still apply regardless
of --magnet-protection, same as every other vessel script.

Usage
-----
    python -m proper_research.hardware.online.vessel_stage_a.run_inverse_jacobian_online_vessel \\
        --plan-dir plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06/time_parameterized_configuration_path \\
        --lumen-file vessel_lumen_robot_frame_2026-10-06_zcorrected.json \\
        --insertion-max-mm 110 --contact --damping 0.05 \\
        --schedule-cache plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy \\
        --magnet-protection hold \\
        --out-dir close_loop_logs/myrun --run-name vessel_invjac_online_contact
"""
from __future__ import annotations

import argparse

import numpy as np

import proper_research.hardware.online.close_loop_path_follow as pf
from proper_research.hardware.online.vessel_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.planning.planning_context import make_robot_config
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.controllers.inverse_jacobian_controller import make_magnet_exclusion_hold_gate
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import BEAM_BASE_PIVOT_Z
from proper_research.simulation.simulations.initial_conditions import make_initial_poses
from pathlib import Path

# --- fix 5 (same bug documented in run_mpc_delay_aware_vessel.py and
# run_open_loop_vessel.py): the generic planner<->robot frame-registration
# step silently picks up an unrelated stale shape_centreline.npz. The
# vessel reference is already in robot frame, so no registration is needed.
def _identity_fit_planner_to_robot(npz_path, start_tip_R, T_R_B):
    return np.eye(3), np.zeros(3)


pf._fit_planner_to_robot = _identity_fit_planner_to_robot
pf._find_shape_npz = lambda plan_dir: Path(__file__)

# DRY: the single shared StateStreamConfig patch (recalibrated beam-base z,
# marker_min/max_count=2) run_open_loop_vessel.py and run_mpc_delay_aware_
# vessel.py both already use -- see common.py's own docstring.
common.patch_state_stream_config_for_recalibrated_rig(pf)

_robot_kin = _resolve_robot_kinematics(make_robot_config())


def _magnet_transform_fn(q6: np.ndarray) -> np.ndarray:
    T = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return np.asarray(T.T_R_target[:3, 3], dtype=float)


def _magnet_position_fn(state7: np.ndarray) -> np.ndarray:
    """Magnet xyz from the controller's own 7-vector state [q1..q6,L] --
    same convention as InverseJacobianBeamController's new magnet_position_fn
    parameter: takes the measured state, returns the magnet position (the
    magnet never depends on L, only the 6 joints)."""
    return _magnet_transform_fn(np.asarray(state7, dtype=float)[:6])


def _magnet_position_jacobian_fn(state7: np.ndarray) -> np.ndarray:
    """(3,7) magnet position Jacobian w.r.t. [q1..q6,L] -- insertion column
    always zero, same convention as delay_aware_mpc.py's magnet_position_
    jacobians and JointSpaceBeamMPCAdapter.magnet_position_jacobian."""
    q6 = np.asarray(state7, dtype=float)[:6]
    J = np.zeros((3, 7), dtype=float)
    J[:, :6] = urik.geometric_jacobian(q6, _robot_kin.dh, _robot_kin.T_F_M)[:3, :]
    return J


_BEAM_BASE_PIVOT_XY = np.array([BEAM_BASE_XYZ_M[0], BEAM_BASE_XYZ_M[1]])


def _recalibrated_initial_poses():
    """Same fix run_mpc_delay_aware_vessel.py's _recalibrated_make_initial_
    poses applies: make_initial_poses()'s own DEFAULT pivot/start-point z is
    the STALE pre-recalibration value (-0.016567) -- build_vessel_planning_
    context silently falls back to this unless initial_poses is passed
    explicitly (the "initial_poses construction bug" this session already
    found and fixed in several diagnostic scripts: an isolated re-solve at
    node 0 was 28mm off the recorded trajectory because of exactly this).
    Reimplemented directly here (not imported from run_mpc_delay_aware_
    vessel.py) to avoid that module's own import-time side effects (IK
    resolution, StateStreamConfig patching) running a second, redundant
    time in this script."""
    p, s, L, dt = make_initial_poses()
    p = np.array(p, dtype=float).copy()
    s = np.array(s, dtype=float).copy()
    p[2] = BEAM_BASE_PIVOT_Z
    s[2] = BEAM_BASE_PIVOT_Z
    return p, s, L, dt


def compute_magnet_exclusion_point(exclusion_floor_m: float, beam_base_pivot_z: float):
    """Fixed single-point magnet-exclusion (same convention as
    run_mpc_delay_aware_vessel.py's compute_magnet_exclusion, reproduced
    here rather than imported -- that module has heavy import-time side
    effects (robot kinematics resolution, StateStreamConfig monkeypatch)
    this script does not want to trigger a second time)."""
    point = np.array([[_BEAM_BASE_PIVOT_XY[0], _BEAM_BASE_PIVOT_XY[1], beam_base_pivot_z]])
    return point, float(exclusion_floor_m)


def _safe_reset_with_retreat(
    plan_dir: str, insertion_tol_mm: float, exclusion_floor_m: float,
    beam_base_pivot_xyz: np.ndarray, magnet_z_bounds_m: tuple[float, float] | None,
    reset_target_q0: np.ndarray | None = None,
):
    try:
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_floor_m,
            magnet_z_bounds_m=magnet_z_bounds_m,
            reset_target_q0=reset_target_q0,
        )
    except RuntimeError as exc:
        print(f"[reset] direct path unsafe ({exc}); routing via a retreat waypoint")
        from proper_research.hardware.ur_rtde_robot import URRTDERobot
        from scipy.spatial.transform import Rotation as Rot

        robot = URRTDERobot(common.ROBOT_IP, frequency=125.0)
        robot.connect()
        q_now = np.array(robot.get_joints())
        robot.close()
        m_now = _magnet_transform_fn(q_now)
        base = beam_base_pivot_xyz[0]
        d_now = np.linalg.norm(m_now - base)
        direction = (m_now - base) / d_now
        retreat_xyz = base + direction * (d_now + 0.030)
        T_now = urik.forward_kinematics(q_now, _robot_kin.dh, _robot_kin.T_F_M)
        rotvec_now = Rot.from_matrix(T_now.T_R_target[:3, :3]).as_rotvec()
        T_retreat = urik.pose6_to_T(np.r_[retreat_xyz, rotvec_now])
        ik = urik.inverse_kinematics_dls(
            T_R_target=T_retreat, q_seed_rad=q_now, dh=_robot_kin.dh,
            T_F_target=_robot_kin.T_F_M, cfg=_robot_kin.ik_cfg,
        )
        if not ik.converged:
            raise RuntimeError("retreat-waypoint IK did not converge") from exc
        common.reset_to_plan_initial_safe(
            ik.q_rad, magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz, magnet_exclusion_radius_m=exclusion_floor_m,
        )
        return common.preflight(
            plan_dir, insertion_tol_mm=insertion_tol_mm,
            magnet_transform_fn=_magnet_transform_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_floor_m,
            magnet_z_bounds_m=magnet_z_bounds_m,
            reset_target_q0=reset_target_q0,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="vessel_invjac_online")
    p.add_argument("--lumen-file", required=True)
    p.add_argument("--insertion-max-mm", type=float, required=True)
    contact_group = p.add_mutually_exclusive_group(required=True)
    contact_group.add_argument("--contact", dest="contact", action="store_true")
    contact_group.add_argument("--no-contact", dest="contact", action="store_false")
    p.add_argument("--schedule-cache", required=True,
                    help="2026-10-07: switched from a LIVE per-tick Jacobian "
                         "(from_model_bundle, 'accurate' mode) to a pre-built (S,3,7) "
                         "schedule .npy -- confirmed live that 'accurate' mode costs "
                         "~1-3s/call, far beyond the 10Hz/100ms control budget; the "
                         "reference (wall-clock-driven) outran the robot by 19 samples "
                         "in two real control ticks, tripping max_tracking_error_m almost "
                         "immediately. This now uses the EXACT SAME precomputed schedule "
                         "mechanism run_mpc_delay_aware_vessel.py uses -- e.g. "
                         "plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_"
                         "newwall_2026-10-06.npy (contact) or the _nc_ sibling -- so the "
                         "Jacobian SOURCE is identical between the MPC and naive-controller "
                         "comparison runs, isolating the control law as the only variable, "
                         "same principle naive_inverse_jacobian_ltv's own module docstring "
                         "documents for why 'inverse-LTV' exists as MPC-LTV's matched "
                         "baseline. --contact/--no-contact must match whichever schedule "
                         "you pass (checked below) -- it no longer builds a live provider.")

    p.add_argument("--damping", type=float, default=0.05,
                    help="Levenberg parameter in J^T(JJ^T+damping^2 I)^-1 -- raise if "
                         "the run shows large commands where the Jacobian's condition "
                         "number spikes (e.g. the first ~0.1mm of insertion on this "
                         "vessel plan, cond up to ~5000 -- see mpc_schedule_scan.csv). "
                         "Default 0.05 matches this project's established PathFollow "
                         "damping/trim_damping convention, not the controller class's "
                         "own much smaller internal default (1e-3), which assumes a "
                         "better-conditioned system than this one.")
    p.add_argument("--position-gain", type=float, default=0.6)
    p.add_argument("--nullspace-gain", type=float, default=0.0)
    p.add_argument("--selective-damping-gain", type=float, default=0.0)

    p.add_argument("--insertion-tol-mm", type=float, default=3.0)
    p.add_argument("--beam-base-exclusion-floor-mm", type=float, default=210.0,
                    help="magnet-to-beam-base exclusion floor -- both the anticipatory "
                         "per-tick clip AND the one-time preflight reset-path check use "
                         "this SAME value (unlike the MPC script, this controller has no "
                         "separate post-hoc measured-joint monitor, so there is only one "
                         "number to set, not a floor+tolerance pair). Default 210.0mm "
                         "matches plans/stage3_design/phi30_L30_left1mm_newwall_design."
                         "json's own exclusion_floor_mm -- NOT 97.0mm, a stale default "
                         "found live 2026-10-07 in run_mpc_delay_aware_vessel.py (correct "
                         "for a different, older campaign's start position, wrong for "
                         "this one -- see that script's own 2026-10-07 fix comment).")
    p.add_argument("--start-radius-override-mm", type=float, default=None,
                    help="2026-10-07: same flag as run_mpc_delay_aware_vessel.py's -- reset "
                         "the robot to a DIFFERENT magnet-to-beam-base distance than the "
                         "plan's own designed start position, WITHOUT touching the offline "
                         "configuration-path/schedule pipeline, so a wider "
                         "--beam-base-exclusion-floor-mm than the plan's own built-in start "
                         "margin can still pass the preflight reset-path check. Moves "
                         "radially along the plan's own start direction, reusing its "
                         "orientation unchanged (common.resolve_start_pose_at_radius). "
                         "The reference trajectory and Jacobian schedule this controller "
                         "tracks are completely unaffected. None (default) = reset to the "
                         "plan's own start position, unchanged behaviour.")
    p.add_argument("--magnet-rise-limit-mm", type=float, default=40.0)
    p.add_argument("--magnet-floor-margin-mm", type=float, default=40.0)
    p.add_argument("--magnet-protection", choices=["clip", "hold", "off"], default="clip",
                    help="how the magnet-exclusion-radius/z-workspace constraints are "
                         "enforced (2026-10-07, three genuinely different experimental "
                         "conditions, not just on/off): "
                         "'clip' (default) -- InverseJacobianBeamController's own "
                         "anticipatory clip projects the command to the boundary and the "
                         "robot keeps moving; the controller 'handles' the constraint. "
                         "'hold' -- the controller's own math stays COMPLETELY naive/"
                         "unaware of the constraint (no clip at all), but an external "
                         "gate (close_loop_path_follow._COMMAND_SAFETY_GATE) sits between "
                         "the controller's raw output and the robot and HOLDS position "
                         "(zero command) on any tick the raw command would violate -- the "
                         "'does the naive controller actually understand this constraint, "
                         "unlike MPC's in-QP formulation' comparison. "
                         "'off' -- neither: the controller's raw, unaware command is sent "
                         "to the robot unmodified. NOT SAFE as a live hardware baseline "
                         "for this constraint specifically -- only the generic velocity/"
                         "accel/toothless-state-box clip and the tcp_out_of_workspace box "
                         "remain. Use 'hold' instead if you want to see the naive "
                         "controller's true (unaware) behaviour without risking the magnet "
                         "actually reaching the beam base.")

    p.add_argument("--max-control-steps", type=int, default=1000)
    p.add_argument("--max-tracking-error-mm", type=float, default=5.0,
                    help="stop the run and report failure if the measured beam-tip "
                         "tracking error ever exceeds this -- same PathFollowConfig."
                         "max_tracking_error_m field every other vessel script uses. "
                         "0 disables.")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    exclusion_floor_m = args.beam_base_exclusion_floor_mm / 1000.0

    schedule = np.load(args.schedule_cache)
    ref_npz = np.load(f"{args.plan_dir}/time_parameterized_configuration_path.npz")
    expected_n = int(ref_npz["state_reference"].shape[0])
    if schedule.shape[0] != expected_n:
        raise ValueError(
            f"--schedule-cache {args.schedule_cache!r} has {schedule.shape[0]} samples "
            f"but --plan-dir {args.plan_dir!r}'s own reference has {expected_n} -- these "
            f"don't match the same plan. Pass the schedule built for THIS plan_dir (e.g. "
            f"by run_mpc_delay_aware_vessel.py's own build_or_load_schedule for the same "
            f"--contact/--no-contact), not a schedule from a different plan."
        )
    pf._SCHEDULE_OVERRIDE = schedule
    print(f"[inv-jac-online] loaded Jacobian schedule {schedule.shape} from "
          f"{args.schedule_cache} ({'contact' if args.contact else 'no_contact'} -- "
          f"verify this matches the schedule file you actually built)")

    q0, l0 = common.load_plan_initial_state(args.plan_dir)
    magnet_z_start = float(_magnet_transform_fn(q0)[2])
    magnet_z_bounds_m = (
        magnet_z_start - args.magnet_floor_margin_mm * 1e-3,
        magnet_z_start + args.magnet_rise_limit_mm * 1e-3,
    )
    beam_base_pivot_xyz, exclusion_radius_m = compute_magnet_exclusion_point(
        exclusion_floor_m, float(BEAM_BASE_XYZ_M[2]),
    )
    print(f"[inv-jac-online] magnet exclusion: point={beam_base_pivot_xyz[0].tolist()} "
          f"radius={exclusion_radius_m*1e3:.1f}mm")
    print(f"[inv-jac-online] magnet z-bounds: start_z={magnet_z_start*1e3:.1f}mm "
          f"bounds=[{magnet_z_bounds_m[0]*1e3:.1f},{magnet_z_bounds_m[1]*1e3:.1f}]mm")

    pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = None
    pf._COMMAND_SAFETY_GATE = None
    if args.magnet_protection == "clip":
        pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = dict(
            magnet_position_fn=_magnet_position_fn,
            magnet_position_jacobian_fn=_magnet_position_jacobian_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_radius_m,
            magnet_z_bounds_m=magnet_z_bounds_m,
        )
        print("[inv-jac-online] magnet protection: CLIP (controller's own anticipatory "
              "projection -- it 'handles' the constraint and keeps moving)")
    elif args.magnet_protection == "hold":
        pf._COMMAND_SAFETY_GATE = make_magnet_exclusion_hold_gate(
            magnet_position_fn=_magnet_position_fn,
            magnet_position_jacobian_fn=_magnet_position_jacobian_fn,
            dt=1.0 / common.CONTROL_HZ,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=exclusion_radius_m,
            magnet_z_bounds_m=magnet_z_bounds_m,
        )
        print("[inv-jac-online] magnet protection: HOLD (controller stays completely "
              "naive/unaware of the constraint; an external gate holds position -- zero "
              "command -- on any tick its raw output would violate it)")
    else:
        print("[inv-jac-online] magnet protection: OFF -- the controller's raw, "
              "unaware command goes straight to the robot. NOT a safe baseline for "
              "this constraint specifically; only the generic velocity/accel/"
              "toothless-state-box clip and tcp_out_of_workspace remain. Use "
              "--magnet-protection hold if you want this comparison without risking "
              "the magnet actually reaching the beam base.")

    reset_target_q0 = None
    if args.start_radius_override_mm is not None:
        reset_target_q0 = common.resolve_start_pose_at_radius(
            q0, args.start_radius_override_mm, robot_kin=_robot_kin,
        )
        print(f"[inv-jac-online] --start-radius-override-mm {args.start_radius_override_mm:.1f}: "
              f"plan's own trajectory/schedule UNCHANGED -- only the physical reset pose moves\n"
              f"[inv-jac-online]   plan q0={np.round(q0,4).tolist()}\n"
              f"[inv-jac-online]   reset q0={np.round(reset_target_q0,4).tolist()}")

    if args.skip_preflight:
        pass
    else:
        q0, l0 = _safe_reset_with_retreat(
            args.plan_dir, args.insertion_tol_mm, exclusion_floor_m,
            beam_base_pivot_xyz, magnet_z_bounds_m,
            reset_target_q0=reset_target_q0,
        )

    cfg = pf.CONFIG
    cfg.controller_kind = "naive_inverse_jacobian_ltv"
    cfg.plan_dir = args.plan_dir
    cfg.reference_source = "plan_dir"

    cfg.damping = args.damping
    cfg.position_gain = args.position_gain
    cfg.nullspace_gain = args.nullspace_gain
    cfg.inv_selective_damping_gain = args.selective_damping_gain
    # 2026-10-07: feedforward=False (was True) for a fair comparison against
    # the MPC's Q_N=0 condition. nullspace_gain=0 alone was not enough --
    # InverseJacobianBeamController.solve() still adds
    # `projector @ reference_input` whenever feedforward=True, which injects
    # the offline plan's own recorded velocity into the null space EVEN
    # WHEN nullspace_gain=0 (the posture-pull term and the feedforward term
    # both use the same null-space projector but are otherwise independent
    # -- see that method's own command-assembly order). This is the exact
    # same class of effect run_mpc_delay_aware_vessel.py's
    # --zero-input-reference-in-r flag exists to remove from the MPC's
    # R-term. feedforward is already a plain on/off weight on the existing
    # InverseJacobianBeamController -- no new capability needed, just using
    # the setting that was already there.
    cfg.feedforward = False
    cfg.feedforward_full = False

    cfg.feedforward_joint_trajectory = False
    cfg.accumulator_seam = True
    cfg.servo_stream_hz = common.SERVO_STREAM_HZ
    cfg.force_mpc_feedforward = False

    cfg.joint_velocity_limit_rad_s = common.JOINT_VELOCITY_LIMIT_RAD_S
    cfg.max_joint_step_rad = common.MAX_JOINT_STEP_RAD
    cfg.joint_acceleration_limit_rad_s2 = common.JOINT_ACCELERATION_LIMIT_RAD_S2
    cfg.control_insertion = True
    cfg.control_hz = common.CONTROL_HZ
    cfg.max_control_steps = args.max_control_steps
    cfg.initial_insertion_m = l0
    cfg.max_tracking_error_m = args.max_tracking_error_mm / 1000.0

    # Same validated TCP workspace box run_mpc_delay_aware_vessel.py derived
    # and fixed for this exact plan (2026-10-07: planned z 0.3867-0.3908m,
    # live MPC correction observed climbing toward the old 0.393m ceiling --
    # see that script's own 2026-10-07 comment for the full derivation).
    # Reused here, not re-derived -- this is the SAME vessel plan family.
    cfg.workspace_xyz_min_m = (0.277, -0.832, 0.170)
    cfg.workspace_xyz_max_m = (0.656, -0.534, 0.433)

    cfg.output_root = args.out_dir
    cfg.run_name = args.run_name

    print(f"[inv-jac-online] plan={args.plan_dir}\n"
          f"[inv-jac-online] q0={q0.tolist()} L0={l0 * 1000:.2f}mm\n"
          f"[inv-jac-online] contact={args.contact} damping={args.damping} "
          f"position_gain={args.position_gain} nullspace_gain={args.nullspace_gain}\n"
          f"[inv-jac-online] max_tracking_error_mm={args.max_tracking_error_mm} "
          f"(0=disabled)")
    pf.main()


if __name__ == "__main__":
    main()
