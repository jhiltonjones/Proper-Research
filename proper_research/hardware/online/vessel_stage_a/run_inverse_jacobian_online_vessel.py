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
from proper_research.controllers.inverse_jacobian_controller import make_magnet_exclusion_selective_clip_gate
from proper_research.controllers.inverse_jacobian_controller import make_magnet_independent_safety_monitor
from proper_research.rig_calibration import BEAM_BASE_XYZ_M
from proper_research.hardware.online.vessel_stage_a.build_vessel_plan import BEAM_BASE_PIVOT_Z
from proper_research.simulation.simulations.initial_conditions import make_initial_poses
from pathlib import Path

# 2026-10-15: TCP workspace box now sourced live from run_mpc_delay_aware_vessel.py's
# own module-level constants instead of a hardcoded duplicate here -- that script is
# the single source of truth (the user edits it directly when tuning the box), and a
# second hardcoded copy here silently went stale the moment that happened. Importing
# the module (not just its constants) is safe: it is guarded by `if __name__ ==
# "__main__"`, so no hardware connection or argparse is triggered just by importing it.
from proper_research.hardware.online.vessel_stage_a import run_mpc_delay_aware_vessel as _mpc_runner

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


# 2026-10-15: TCP-workspace clip helpers -- SEPARATE frame/DH from the magnet
# ones above, same distinction run_mpc_delay_aware_vessel.py's
# _configure_tcp_workspace documents: this is the literal UR robot-base frame
# "R" that getActualTCPPose() reports in (urik.corrected_dh_from_config(urik.
# CONFIG) + the flange-to-TCP offset), not the beam-centric _robot_kin used
# for the magnet above.
_tcp_dh = urik.corrected_dh_from_config(urik.CONFIG)
_tcp_T_F_TCP = urik.pose6_to_T(urik.CONFIG.offline_active_T_flange_tcp_pose6)


def _tcp_position_fn(state7: np.ndarray) -> np.ndarray:
    """TCP xyz (robot-base frame R) from the controller's own 7-vector state
    [q1..q6,L] -- same convention as InverseJacobianBeamController's new
    tcp_position_fn parameter (the TCP never depends on L, only the 6
    joints)."""
    q6 = np.asarray(state7, dtype=float)[:6]
    return np.asarray(urik.forward_kinematics(q6, _tcp_dh, _tcp_T_F_TCP).T_R_target[:3, 3], dtype=float)


def _tcp_position_jacobian_fn(state7: np.ndarray) -> np.ndarray:
    """(3,7) TCP position Jacobian w.r.t. [q1..q6,L] -- insertion column
    always zero, same convention as run_mpc_delay_aware_vessel.py's
    tcp_workspace_kwargs builder."""
    q6 = np.asarray(state7, dtype=float)[:6]
    J = np.zeros((3, 7), dtype=float)
    J[:, :6] = urik.geometric_jacobian(q6, _tcp_dh, _tcp_T_F_TCP)[:3, :]
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

    p.add_argument("--magnetic-force-preference-schedule", type=str, default=None,
                    help="path to the .npz written by compute_magnetic_force_preference_"
                         "schedule.py for THIS plan. None (default) = disabled, exact "
                         "existing behaviour. When given, also requires --force-preference-"
                         "gain > 0 to actually move the command -- see build_magnetic_force_"
                         "preference_gradient_fn's docstring: resolved into the SAME task "
                         "nullspace the posture-regulation pull already uses, never fighting "
                         "the tip-tracking task term.")
    p.add_argument("--force-desired-n", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("FX", "FY", "FZ"),
                    help="desired resultant magnetic force on the beam (Newtons, robot frame "
                         "R) the soft preference pulls toward.")
    p.add_argument("--torque-desired-nm", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    metavar=("TX", "TY", "TZ"),
                    help="desired resultant magnetic torque on the beam (N*m, robot frame R).")
    p.add_argument("--lambda-force", type=float, default=0.0,
                    help="weight on the force term of the soft preference (0.0 disables it).")
    p.add_argument("--lambda-torque", type=float, default=0.0,
                    help="weight on the torque term of the soft preference (0.0 disables it).")
    p.add_argument("--force-preference-gain", type=float, default=0.0,
                    help="nullspace gradient-descent step size for the force/torque "
                         "preference (0.0 = disabled regardless of the lambdas above).")

    p.add_argument("--dipole-alignment-weight", type=float, default=0.0,
                    help="2026-10-19: enable the dipole-to-point alignment nullspace term -- "
                         "see controllers.dipole_alignment's module docstring for why this is "
                         "preferred over the torque-based preference above for 'point the "
                         "magnet somewhere': minimizing torque can be satisfied by distance "
                         "alone, this term cannot (its residual is a difference of two unit "
                         "vectors). Purely kinematic, computed fresh every tick -- no schedule "
                         "file needed, unlike --magnetic-force-preference-schedule. Mutually "
                         "exclusive with the --magnetic-force-preference-schedule/-lambda-* "
                         "flags above (both use the SAME nullspace-descent slot in the "
                         "controller; only one preference can be active per run). 0.0 "
                         "(default) disables it exactly.")
    p.add_argument("--dipole-alignment-target-point-m", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="defaults to BEAM_BASE_XYZ_M if unset.")
    p.add_argument("--dipole-alignment-sign", type=float, default=1.0, choices=(1.0, -1.0),
                    help="+1.0 (default): dipole points TOWARD the target. -1.0: away from it.")

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
    p.add_argument("--magnet-protection-margin-mm", type=float, default=3.0,
                    help="2026-10-08: safety margin ADDED to the exclusion radius (and "
                         "SUBTRACTED from the z-workspace bounds) that the PREDICTIVE "
                         "gate (clip/hold/selective -- whichever --magnet-protection "
                         "chose) enforces, so it starts reacting before the TRUE floor "
                         "rather than exactly at it. Does NOT change the TRUE floor "
                         "itself, which the independent measured-state monitor "
                         "(make_magnet_independent_safety_monitor, always active) keeps "
                         "checking unchanged -- this only buys the predictive gate some "
                         "slack. Exists because the gate's one-tick-ahead check is a "
                         "first-order linear prediction assuming the commanded velocity "
                         "is achieved instantly and exactly; real execution (servoJ's own "
                         "--servo-lookahead-s smoothing, vision/FK measurement noise) can "
                         "differ from that by a small amount each tick, and a controller "
                         "riding the floor with nullspace_gain=0 has nothing pulling it "
                         "back -- confirmed live 2026-10-08: a 19-tick, ~0.1-0.3mm/tick "
                         "smooth creep through a 255mm floor, final breach only 0.097mm, "
                         "with the gate never once predicting a violation in that window. "
                         "0 disables (restores the original exactly-at-the-floor gate "
                         "behaviour).")
    p.add_argument("--magnet-protection", choices=["clip", "hold", "selective", "off"], default="clip",
                    help="how the magnet-exclusion-radius/z-workspace constraints are "
                         "enforced (2026-10-07/2026-10-08, four genuinely different "
                         "experimental conditions, not just on/off): "
                         "'clip' (default) -- InverseJacobianBeamController's own "
                         "anticipatory clip: a minimum-norm projection of the WHOLE "
                         "6-vector onto the constraint boundary (every joint touched a "
                         "little); the robot keeps moving; the controller 'handles' the "
                         "constraint. "
                         "'hold' -- the controller's own math stays COMPLETELY naive/"
                         "unaware of the constraint (no clip at all), but an external "
                         "gate (close_loop_path_follow._COMMAND_SAFETY_GATE) sits between "
                         "the controller's raw output and the robot and HOLDS position "
                         "(zeroes the WHOLE 7-vector) on any tick the raw command would "
                         "violate -- the 'does the naive controller actually understand "
                         "this constraint, unlike MPC's in-QP formulation' comparison. "
                         "'selective' -- also an external gate (controller stays naive), "
                         "but instead of zeroing everything it zeroes ONLY the individual "
                         "joint-velocity components whose own sign is pushing the "
                         "predicted margin further into violation (g_i*command_i<0), and "
                         "passes every other component -- including the insertion-rate "
                         "channel, which this constraint never touches -- through exactly "
                         "as the naive controller commanded it. Not guaranteed to fully "
                         "resolve the violation (logged as magnet_gate_still_violating "
                         "when it doesn't); see make_magnet_exclusion_selective_clip_gate. "
                         "'off' -- neither: the controller's raw, unaware command is sent "
                         "to the robot unmodified. NOT SAFE as a live hardware baseline "
                         "for this constraint specifically -- only the generic velocity/"
                         "accel/toothless-state-box clip and the tcp_out_of_workspace box "
                         "remain. Use 'hold' or 'selective' instead if you want to see the "
                         "naive controller's true (unaware) behaviour without risking the "
                         "magnet actually reaching the beam base.")

    p.add_argument("--max-control-steps", type=int, default=1000)
    p.add_argument("--max-tracking-error-mm", type=float, default=5.0,
                    help="stop the run and report failure if the measured beam-tip "
                         "tracking error ever exceeds this -- same PathFollowConfig."
                         "max_tracking_error_m field every other vessel script uses. "
                         "0 disables.")
    p.add_argument("--tcp-protection-margin-mm", type=float, default=15.0,
                    help="2026-10-15: same margin concept as --magnet-protection-margin-mm, "
                         "for the TCP-workspace clip. Default is larger (15mm vs magnet's "
                         "3mm) because the observed drift rate here is faster: confirmed "
                         "live, mpc_closedloop_invjac_contact_floor240_v1 (zero margin) "
                         "crossed the true y-boundary within 7 ticks at ~3-6mm/tick near "
                         "the end (nullspace_gain=0, nothing pulling the redundant DOF "
                         "back), overshooting the true floor by ~2mm before the post-hoc "
                         "tcp_out_of_workspace monitor caught it one tick later -- the "
                         "clip's one-tick-ahead linear prediction only starts resisting "
                         "once already within one tick's displacement of the TRUE "
                         "boundary, which at this drift rate is too late. Does NOT change "
                         "the TRUE box itself (tcp_out_of_workspace keeps checking the "
                         "real one, unchanged) -- only shrinks what the clip targets.")
    p.add_argument("--tcp-true-box-relax-mm", type=float, default=0.0,
                    help="2026-10-15: widens the ACTUAL TCP workspace box (both what "
                         "tcp_out_of_workspace checks AND what the clip's own margin is "
                         "computed relative to) by this amount on every side, LOCAL to "
                         "this script only -- does NOT modify run_mpc_delay_aware_vessel."
                         "py's _WORKSPACE_XYZ_MIN_M/_MAX_M, so MPC runs are unaffected. "
                         "Exists because nullspace_gain=0 gives this controller's "
                         "redundant DOF no restoring force, so it can persistently drift "
                         "toward a boundary rather than just momentarily overshoot one -- "
                         "confirmed live across TCP-y, magnet-z and magnet-exclusion-"
                         "radius each in turn. --tcp-protection-margin-mm alone cannot fix "
                         "this (it only changes when the clip starts reacting, not how "
                         "much room the drift has to work with); this flag gives the "
                         "drift more room directly. 0.0 (default) = unchanged, exactly "
                         "the real box used elsewhere today.")
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

    pf._FORCE_PREFERENCE_OVERRIDE = None
    if args.magnetic_force_preference_schedule is not None:
        if args.force_preference_gain <= 0.0 or (args.lambda_force <= 0.0 and args.lambda_torque <= 0.0):
            raise SystemExit(
                "--magnetic-force-preference-schedule was given but --force-preference-gain "
                "and/or both lambdas are 0.0 -- the preference would have no effect; pass "
                "positive --force-preference-gain and at least one positive lambda, or drop "
                "the schedule flag."
            )
        from proper_research.controllers.inverse_jacobian_controller import (
            build_magnetic_force_preference_gradient_fn,
        )

        _force_pref_npz = np.load(args.magnetic_force_preference_schedule)
        _force_pref_reference_state = np.asarray(ref_npz["state_reference"], dtype=float)
        if _force_pref_npz["force_jacobians"].shape[0] != _force_pref_reference_state.shape[0]:
            raise ValueError(
                f"--magnetic-force-preference-schedule {args.magnetic_force_preference_schedule!r} "
                f"has {_force_pref_npz['force_jacobians'].shape[0]} samples but --plan-dir "
                f"{args.plan_dir!r}'s reference has {_force_pref_reference_state.shape[0]} -- "
                "these don't match the same plan."
            )
        grad_fn = build_magnetic_force_preference_gradient_fn(
            reference_state=_force_pref_reference_state,
            force_jacobians=np.asarray(_force_pref_npz["force_jacobians"], dtype=float),
            torque_jacobians=np.asarray(_force_pref_npz["torque_jacobians"], dtype=float),
            nominal_force_N=np.asarray(_force_pref_npz["nominal_force_N"], dtype=float),
            nominal_torque_Nm=np.asarray(_force_pref_npz["nominal_torque_Nm"], dtype=float),
            force_desired_N=np.asarray(args.force_desired_n, dtype=float),
            torque_desired_Nm=np.asarray(args.torque_desired_nm, dtype=float),
            lambda_force=float(args.lambda_force), lambda_torque=float(args.lambda_torque),
        )
        pf._FORCE_PREFERENCE_OVERRIDE = dict(
            force_preference_gradient_fn=grad_fn,
            force_preference_gain=float(args.force_preference_gain),
        )
        print(f"[inv-jac-online] magnetic force/torque PREFERENCE requested "
              f"(schedule={args.magnetic_force_preference_schedule}, "
              f"F_des={args.force_desired_n}N, tau_des={args.torque_desired_nm}Nm, "
              f"lambda_force={args.lambda_force:.3g}, lambda_torque={args.lambda_torque:.3g}, "
              f"force_preference_gain={args.force_preference_gain:.3g}) -- soft nullspace "
              f"tie-breaker only")
    elif args.dipole_alignment_weight > 0.0:
        from proper_research.controllers.dipole_alignment import build_gradient_fn_live
        from proper_research.planning.vessel_context import build_vessel_planning_context

        _dipole_target = (
            np.asarray(BEAM_BASE_XYZ_M, dtype=float) if args.dipole_alignment_target_point_m is None
            else np.asarray(args.dipole_alignment_target_point_m, dtype=float)
        )
        print("[inv-jac-online] building model bundle once to read the calibrated "
              "source-magnet dipole moment (m_body) for --dipole-alignment-weight ...")
        _, _dipole_bundle, _dipole_controller_pack, _, _, _, _ = build_vessel_planning_context(
            lumen_file=args.lumen_file, insertion_max_m=args.insertion_max_mm * 1.0e-3,
            jacobian_mode="fast", plant_contact=args.contact,
        )
        _dipole_m_body = np.asarray(
            _dipole_controller_pack["plant_diagnostic_joint_adapter"].model.m_body, dtype=float,
        ).reshape(3)
        grad_fn = build_gradient_fn_live(
            dh=_robot_kin.dh, T_F_M=_robot_kin.T_F_M, m_body=_dipole_m_body,
            target_point_m=_dipole_target, sign=float(args.dipole_alignment_sign),
        )
        pf._FORCE_PREFERENCE_OVERRIDE = dict(
            force_preference_gradient_fn=grad_fn,
            force_preference_gain=float(args.dipole_alignment_weight),
        )
        print(f"[inv-jac-online] dipole-to-point alignment PREFERENCE requested "
              f"(target_point_m={_dipole_target.tolist()}, weight={args.dipole_alignment_weight:.3g}, "
              f"sign={args.dipole_alignment_sign:+.0f}, m_body={_dipole_m_body.tolist()}) -- "
              f"computed fresh every tick, no schedule file, soft nullspace term only")

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

    # The PREDICTIVE gate (clip/hold/selective) enforces the TRUE floor plus
    # this margin, so it starts reacting before the true boundary rather than
    # exactly at it -- see --magnet-protection-margin-mm's help text. The
    # independent measured-state monitor below is NOT given this margin: it
    # keeps checking the TRUE floor unchanged, as the final backstop.
    protection_margin_m = args.magnet_protection_margin_mm * 1e-3
    gate_exclusion_radius_m = exclusion_radius_m + protection_margin_m
    gate_z_bounds_m = (
        magnet_z_bounds_m[0] + protection_margin_m,
        magnet_z_bounds_m[1] - protection_margin_m,
    )
    if protection_margin_m > 0:
        print(f"[inv-jac-online] predictive-gate safety margin: {args.magnet_protection_margin_mm:.1f}mm "
              f"-> gate radius={gate_exclusion_radius_m*1e3:.1f}mm "
              f"(true floor {exclusion_radius_m*1e3:.1f}mm unchanged for the independent monitor), "
              f"gate z-bounds=[{gate_z_bounds_m[0]*1e3:.1f},{gate_z_bounds_m[1]*1e3:.1f}]mm "
              f"(true [{magnet_z_bounds_m[0]*1e3:.1f},{magnet_z_bounds_m[1]*1e3:.1f}]mm unchanged)")

    # 2026-10-15: TCP-workspace anticipatory clip -- ALWAYS enabled, independent
    # of --magnet-protection (a genuinely separate constraint from the magnet-
    # exclusion/z-workspace ones that flag selects between). Previously this
    # controller had NO protection for the general TCP box at all (confirmed
    # live: mpc_closedloop_invjac_contact_floor240_v1 left the box within 7
    # ticks). Sourced live from run_mpc_delay_aware_vessel.py's own
    # _WORKSPACE_XYZ_MIN_M/_MAX_M constants -- see this script's import of
    # that module above.
    _tcp_relax_m = args.tcp_true_box_relax_mm * 1.0e-3
    _tcp_true_xyz_min = np.asarray(_mpc_runner._WORKSPACE_XYZ_MIN_M, dtype=float) - _tcp_relax_m
    _tcp_true_xyz_max = np.asarray(_mpc_runner._WORKSPACE_XYZ_MAX_M, dtype=float) + _tcp_relax_m
    if args.tcp_true_box_relax_mm > 0:
        print(f"[inv-jac-online] TCP true box RELAXED by {args.tcp_true_box_relax_mm:.1f}mm/side "
              f"(LOCAL to this script -- run_mpc_delay_aware_vessel.py's own box is unchanged): "
              f"min={_tcp_true_xyz_min.tolist()} max={_tcp_true_xyz_max.tolist()} "
              f"(real box was min={_mpc_runner._WORKSPACE_XYZ_MIN_M} max={_mpc_runner._WORKSPACE_XYZ_MAX_M})")

    _tcp_margin_m = args.tcp_protection_margin_mm * 1.0e-3
    _tcp_gate_xyz_min = _tcp_true_xyz_min + _tcp_margin_m
    _tcp_gate_xyz_max = _tcp_true_xyz_max - _tcp_margin_m
    if np.any(_tcp_gate_xyz_min >= _tcp_gate_xyz_max):
        raise ValueError(
            f"--tcp-protection-margin-mm {args.tcp_protection_margin_mm:.1f}mm is too large: "
            f"it would invert the clip's own TCP-workspace bounds (box after relax: "
            f"min={_tcp_true_xyz_min.tolist()}, max={_tcp_true_xyz_max.tolist()})."
        )
    _tcp_clip_kwargs = dict(
        tcp_position_fn=_tcp_position_fn,
        tcp_position_jacobian_fn=_tcp_position_jacobian_fn,
        tcp_workspace_xyz_min_m=_tcp_gate_xyz_min,
        tcp_workspace_xyz_max_m=_tcp_gate_xyz_max,
    )
    print(f"[inv-jac-online] TCP-workspace clip: ALWAYS ACTIVE (independent of "
          f"--magnet-protection), gate bounds min={_tcp_gate_xyz_min.tolist()} "
          f"max={_tcp_gate_xyz_max.tolist()} (true bounds after relax min={_tcp_true_xyz_min.tolist()} "
          f"max={_tcp_true_xyz_max.tolist()}, --tcp-protection-margin-mm={args.tcp_protection_margin_mm:.1f}mm)")

    pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = dict(_tcp_clip_kwargs)
    pf._COMMAND_SAFETY_GATE = None
    if args.magnet_protection == "clip":
        pf._MAGNET_CONSTRAINT_CLIP_OVERRIDE = dict(
            _tcp_clip_kwargs,
            magnet_position_fn=_magnet_position_fn,
            magnet_position_jacobian_fn=_magnet_position_jacobian_fn,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=gate_exclusion_radius_m,
            magnet_z_bounds_m=gate_z_bounds_m,
        )
        print("[inv-jac-online] magnet protection: CLIP (controller's own anticipatory "
              "projection -- it 'handles' the constraint and keeps moving)")
    elif args.magnet_protection == "hold":
        pf._COMMAND_SAFETY_GATE = make_magnet_exclusion_hold_gate(
            magnet_position_fn=_magnet_position_fn,
            magnet_position_jacobian_fn=_magnet_position_jacobian_fn,
            dt=1.0 / common.CONTROL_HZ,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=gate_exclusion_radius_m,
            magnet_z_bounds_m=gate_z_bounds_m,
        )
        print("[inv-jac-online] magnet protection: HOLD (controller stays completely "
              "naive/unaware of the constraint; an external gate holds position -- zero "
              "command -- on any tick its raw output would violate it)")
    elif args.magnet_protection == "selective":
        pf._COMMAND_SAFETY_GATE = make_magnet_exclusion_selective_clip_gate(
            magnet_position_fn=_magnet_position_fn,
            magnet_position_jacobian_fn=_magnet_position_jacobian_fn,
            dt=1.0 / common.CONTROL_HZ,
            magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
            magnet_exclusion_radius_m=gate_exclusion_radius_m,
            magnet_z_bounds_m=gate_z_bounds_m,
        )
        print("[inv-jac-online] magnet protection: SELECTIVE (controller stays completely "
              "naive/unaware of the constraint; an external gate zeroes ONLY the "
              "individual joint-velocity components pushing toward violation on any tick "
              "its raw output would violate it, passing the rest -- including insertion -- "
              "through unmodified)")
    else:
        print("[inv-jac-online] magnet protection: OFF -- the controller's raw, "
              "unaware command goes straight to the robot. NOT a safe baseline for "
              "this constraint specifically; only the generic velocity/accel/"
              "toothless-state-box clip and tcp_out_of_workspace remain. Use "
              "--magnet-protection hold if you want this comparison without risking "
              "the magnet actually reaching the beam base.")

    # DRY port (2026-10-08) of run_mpc_delay_aware_vessel.py's "fix 4"/"fix 6"
    # independent, measured-state safety monitors -- always installed,
    # composing with whatever --magnet-protection chose above (including
    # None for "off"), exactly mirroring that these monitors run
    # unconditionally in the MPC script too, regardless of the QP's own
    # separately-validated in-solve constraints. See
    # make_magnet_independent_safety_monitor's docstring for what this does
    # and does NOT catch (notably: NOT tcp_out_of_workspace excursions --
    # that remains the only thing checking the robot's own TCP frame against
    # the generic Cartesian box, since redundant joints can swing the arm
    # there even while the magnet itself stays within these bounds).
    pf._COMMAND_SAFETY_GATE = make_magnet_independent_safety_monitor(
        magnet_position_fn=_magnet_position_fn,
        magnet_exclusion_lumen_C_m=beam_base_pivot_xyz,
        magnet_exclusion_radius_m=exclusion_radius_m,
        magnet_z_bounds_m=magnet_z_bounds_m,
        inner_gate=pf._COMMAND_SAFETY_GATE,
    )
    print("[inv-jac-online] independent magnet-z/exclusion safety monitor: ACTIVE "
          "(measured-state backstop, same mechanism as run_mpc_delay_aware_vessel.py's "
          "fix 4/fix 6 -- aborts if the magnet has ALREADY left bounds, independent of "
          "the --magnet-protection mode above)")

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
    # (2026-10-07: planned z 0.3867-0.3908m, live MPC correction observed
    # climbing toward the old 0.393m ceiling -- see that script's own
    # 2026-10-07 comment for the full derivation). Read LIVE from that
    # script's own module-level constants (2026-10-15) instead of a second
    # hardcoded copy here, so this box can never silently drift out of sync
    # with the one the MPC runner actually uses/the user tunes directly.
    # Uses _tcp_true_xyz_min/max (the --tcp-true-box-relax-mm-widened box
    # computed above, identical to the real box when relax=0.0) so this
    # post-hoc monitor stays consistent with what the clip itself targets.
    # dataclasses.asdict(cfg) (used when writing summary.json) does not
    # convert numpy arrays -- must be plain tuples, matching workspace_xyz_
    # min_m/max_m's own `tuple[float, float, float]` field type, or the run
    # crashes at the very end on json.dump(TypeError: ndarray not JSON
    # serializable), discovered live (2026-10-20) only after a full run
    # otherwise completed cleanly.
    cfg.workspace_xyz_min_m = tuple(float(v) for v in _tcp_true_xyz_min)
    cfg.workspace_xyz_max_m = tuple(float(v) for v in _tcp_true_xyz_max)

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
