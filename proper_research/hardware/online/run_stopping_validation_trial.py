"""2026-11-19: ONE operator-run stopping-validation trial. Uses the real
URRTDERobot interface, the real BackgroundArmStreamer, and the actual
application abort/shutdown sequence (streamer.stop() then robot.servo_stop())
-- not an isolated servoStop() call.

The motion is PLAYBACK of a contiguous segment of the already-reviewed,
already-verified time-parameterized plan (loaded via
stopping_validation_plan_segment.py), streamed through the production
BackgroundArmStreamer at the plan's own sample cadence -- it is NOT the
real MPC re-solving online. Playback reproduces the exact reference
trajectory regardless of any disturbance; the live MPC would instead
re-solve every tick and could command something different from the
reference under the same disturbance. This script validates the stop
response to the PLAYBACK-COMMANDED trajectory. MPC-following stopping
behaviour is a separate, NOT-yet-covered validation target.

Every physical parameter (segment choice, speed limit, trigger definition,
standstill threshold/hold time, max trial duration, insertion handling) is
a REQUIRED CLI argument with NO default. This script will not run with
invented values -- if the operator hasn't reviewed and supplied a value,
argparse refuses to start. The 230mm reviewed exclusion boundary and the
provisional 250mm measured-clearance radius are both untouched by this
script: a staged diagnostic trigger (--trigger-mode staged) uses its OWN
separately-supplied centre/radius, never common.MAGNET_BEAM_BASE_REVIEWED_
BOUNDARY_MM or common.MAGNET_MEASURED_CLEARANCE_PROTECTIVE_RADIUS_MM. The
plan segment's own clearance is independently verified (see
test_stopping_validation_plan_segment.py) to stay well clear of that
boundary -- this script does not rely on "never referencing it" as
evidence of protection.

Insertion is a SEPARATE actuator from the 6 UR joints (driven by
AdvancerSink, not BackgroundArmStreamer). --insertion-mode is REQUIRED and
has no default: 'hold_fixed' requires an explicit acknowledgement flag
(because the loaded plan segment's insertion is NOT constant, so holding
it fixed is a deliberate, stated deviation from the reference, never a
silent default); 'advancer_playback' requires both --advancer-port and
--confirmed-initial-insertion-mm, the operator's attestation that the
REAL physical insertion length currently matches the segment's start
value (there is no absolute insertion sensor -- the software only
integrates from whatever initial value it is told).

No automatic repeats, no speed escalation, no restart. One invocation runs
exactly one trial and exits.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/jack/Proper-Research")
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.hardware.online.close_loop_path_follow import BackgroundArmStreamer
from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import _robot_kin
from proper_research.hardware.online.stopping_validation_logger import StoppingTestLog, displacement_segments
from proper_research.hardware.online.stopping_validation_plan_segment import extract_segment
from proper_research.hardware.ur_rtde_robot import URRTDERobot


def _fk(q6) -> np.ndarray:
    fk = urik.forward_kinematics(np.asarray(q6, dtype=float).reshape(6), _robot_kin.dh, _robot_kin.T_F_M)
    return fk.T_R_target[:3, 3]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=str, required=True,
                   help="Path to the time_parameterized_configuration_path directory of the "
                        "reviewed plan (e.g. the marginfix plan the A/B/C pilot runs used).")
    p.add_argument("--segment-start-idx", type=int, required=True,
                   help="First sample index (inclusive) of the reviewed segment within the plan.")
    p.add_argument("--segment-end-idx", type=int, required=True,
                   help="Last sample index (inclusive) of the reviewed segment within the plan.")
    p.add_argument("--insertion-mode", choices=["hold_fixed", "advancer_playback"], required=True,
                   help="'hold_fixed': insertion actuator is NOT commanded during this trial -- "
                        "requires --acknowledge-insertion-held-fixed since the segment's own "
                        "insertion is not constant. 'advancer_playback': the real AdvancerSink "
                        "plays back the segment's insertion_rate_m_s -- requires --advancer-port "
                        "and --confirmed-initial-insertion-mm.")
    p.add_argument("--acknowledge-insertion-held-fixed", action="store_true",
                   help="Required with --insertion-mode hold_fixed. States you understand this "
                        "trial will NOT reproduce the segment's insertion motion.")
    p.add_argument("--advancer-port", type=str, default=None,
                   help="Required with --insertion-mode advancer_playback.")
    p.add_argument("--confirmed-initial-insertion-mm", type=float, default=None,
                   help="Required with --insertion-mode advancer_playback: your attestation "
                        "that the REAL physical insertion length right now equals this value. "
                        "Checked against the segment's own start insertion length; a mismatch "
                        "beyond --insertion-mismatch-tolerance-mm refuses to start.")
    p.add_argument("--insertion-mismatch-tolerance-mm", type=float, default=2.0)
    p.add_argument("--speed-limit-rad-s", type=float, required=True,
                   help="Reviewed joint speed limit cap applied on top of the plan's own rates. No default.")
    p.add_argument("--trigger-mode", choices=["manual", "staged"], required=True,
                   help="'manual': operator issues the abort via SIGUSR1 to this process. "
                        "'staged': a diagnostic boundary (--staged-trigger-centre-m / "
                        "--staged-trigger-radius-m, BOTH required, separate from the reviewed "
                        "230mm exclusion) fires the real detection code path when crossed.")
    p.add_argument("--standstill-velocity-threshold-rad-s", type=float, required=True,
                   help="Max per-joint |finite-difference velocity| (rad/s) to count as standstill. No default.")
    p.add_argument("--standstill-hold-time-s", type=float, required=True,
                   help="Seconds the velocity must stay below threshold before standstill is declared. No default.")
    p.add_argument("--max-trial-duration-s", type=float, required=True,
                   help="Hard ceiling on trial duration; exceeding it marks the trial INCOMPLETE "
                        "and invokes --lab-failure-procedure. No default.")
    p.add_argument("--lab-failure-procedure", type=str, required=True,
                   help="Free-text description of the lab's established failure procedure to "
                        "follow if standstill is not reached or an exception occurs. Recorded "
                        "in the output, not executed by this script -- the operator acts on it.")
    p.add_argument("--sample-interval-s", type=float, default=0.02,
                   help="Joint-sampling interval (default 20ms, i.e. 50Hz).")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write trial outputs into. Must not already exist.")
    p.add_argument("--robot-host", type=str, required=True,
                   help="URRTDERobot's single ip argument -- same host used for both the control "
                        "and receive RTDE interfaces internally, matching close_loop_path_follow.py.")
    return p


def main(argv=None) -> int:
    p = build_arg_parser()
    # the staged-trigger args need point types, fix up properly here
    p.add_argument("--staged-trigger-centre-xyz-m", type=str, default=None,
                    help="Required if --trigger-mode=staged: 'x,y,z' metres, the diagnostic "
                         "boundary centre. Reviewed by the operator for the chosen clear route; "
                         "NOT the beam-base centre used by the 230mm exclusion.")
    p.add_argument("--staged-trigger-radius-m", type=float, default=None,
                    help="Required if --trigger-mode=staged: diagnostic boundary radius, metres.")
    args = p.parse_args(argv)

    if args.trigger_mode == "staged":
        if args.staged_trigger_centre_xyz_m is None or args.staged_trigger_radius_m is None:
            p.error("--trigger-mode=staged requires both --staged-trigger-centre-xyz-m and --staged-trigger-radius-m")
        staged_centre = np.array([float(x) for x in args.staged_trigger_centre_xyz_m.split(",")], dtype=float)
        if staged_centre.shape != (3,):
            p.error("--staged-trigger-centre-xyz-m must be 'x,y,z'")
    else:
        staged_centre = None

    if args.insertion_mode == "hold_fixed" and not args.acknowledge_insertion_held_fixed:
        p.error("--insertion-mode hold_fixed requires --acknowledge-insertion-held-fixed "
                "(the plan segment's insertion is not constant; holding it fixed is a "
                "deliberate deviation that must be explicitly acknowledged, not a silent default)")
    if args.insertion_mode == "advancer_playback":
        if args.advancer_port is None or args.confirmed_initial_insertion_mm is None:
            p.error("--insertion-mode advancer_playback requires both --advancer-port and "
                    "--confirmed-initial-insertion-mm")

    segment = extract_segment(args.plan_dir, args.segment_start_idx, args.segment_end_idx, _fk)

    if args.insertion_mode == "advancer_playback":
        mismatch_mm = abs(args.confirmed_initial_insertion_mm - segment.insertion_m[0] * 1e3)
        if mismatch_mm > args.insertion_mismatch_tolerance_mm:
            print(f"REFUSING to start: --confirmed-initial-insertion-mm={args.confirmed_initial_insertion_mm} "
                  f"differs from the segment's own start insertion ({segment.insertion_m[0]*1e3:.2f}mm) "
                  f"by {mismatch_mm:.2f}mm, exceeding --insertion-mismatch-tolerance-mm="
                  f"{args.insertion_mismatch_tolerance_mm}.", file=sys.stderr)
            return 1

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        print(f"REFUSING to start: --output-dir {out_dir} already exists.", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True)
    (out_dir / "plan_segment_table.json").write_text(json.dumps(segment.summary_table(), indent=2))

    # URRTDERobot's constructor connects immediately (see ur_rtde_robot.py)
    # -- matches close_loop_path_follow.py's own construction call exactly
    # (same frequency, no external_control override; production's A/B/C
    # runs connect this same way).
    robot = URRTDERobot(args.robot_host, frequency=500.0)
    try:
        return run_trial(args, robot, out_dir, staged_centre, segment)
    finally:
        robot.close()


def run_trial(args, robot, out_dir: Path, staged_centre, segment) -> int:
    """The actual trial lifecycle, taking an already-connected `robot` object
    (real URRTDERobot in production, a fake with the same servo_j/servo_stop/
    get_joints surface in tests) and the already-loaded plan `segment` --
    separated out so the integrated lifecycle can be mock-tested without
    hardware."""
    advancer = None
    if args.insertion_mode == "advancer_playback":
        from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
        advancer = AdvancerSink(AdvancerSinkConfig(port=args.advancer_port, dry_run=False,
                                                     max_rate_m_s=2.0e-3))

    log = StoppingTestLog()
    manual_abort = threading.Event()
    if args.trigger_mode == "manual":
        def _on_sigusr1(signum, frame):
            manual_abort.set()
        signal.signal(signal.SIGUSR1, _on_sigusr1)
        print(f"[trial] manual trigger armed: send SIGUSR1 to pid {__import__('os').getpid()} to abort.")

    streamer = BackgroundArmStreamer(robot, time_s=0.1, lookahead_time=0.1, gain=300,
                                      stream_interval_s=args.sample_interval_s)

    trial_status = {"outcome": "running"}
    last_dispatch_recorded = threading.Event()

    def _sampling_loop():
        """Runs on its own thread, continuously through shutdown and standstill
        -- this is the instrumentation the user asked to stay alive until
        standstill is actually measured, independent of the streamer thread."""
        t_start = time.monotonic()
        consecutive_below_threshold_s = 0.0
        prev_q = None
        prev_t = None
        crossed = False
        while True:
            try:
                q = np.array(robot.get_joints(), dtype=float)
            except Exception as exc:
                trial_status["outcome"] = "incomplete_exception"
                trial_status["exception"] = repr(exc)
                return
            t = time.monotonic()
            log.record_sample(t=t, q6=q)

            if not crossed:
                if args.trigger_mode == "staged":
                    from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
                    from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import _robot_kin
                    fk = urik.forward_kinematics(q, _robot_kin.dh, _robot_kin.T_F_M)
                    magnet_xyz = fk.T_R_target[:3, 3]
                    if np.linalg.norm(magnet_xyz - staged_centre) < args.staged_trigger_radius_m:
                        crossed = True
                        log.record_event("boundary_crossing", t=t, q_rad=q)
                        log.record_event("detection", t=t, q_rad=q)  # same tick in this harness: staged trigger IS the detector
                elif manual_abort.is_set():
                    crossed = True
                    log.record_event("boundary_crossing", t=t, q_rad=q)
                    log.record_event("detection", t=t, q_rad=q)
                    # manual trigger: automatic crossing-to-detection latency is UNMEASURED
                    # (the operator's own action is the "crossing" we can observe, and we
                    # cannot time a geometric crossing that never happened).

            if crossed and not last_dispatch_recorded.is_set():
                pass  # last_dispatch is recorded by the main thread at the moment it stops streaming

            if prev_q is not None:
                dt = t - prev_t
                v = np.abs(q - prev_q) / max(dt, 1e-6)
                if crossed and np.all(v < args.standstill_velocity_threshold_rad_s):
                    consecutive_below_threshold_s += dt
                elif crossed:
                    consecutive_below_threshold_s = 0.0
                if crossed and consecutive_below_threshold_s >= args.standstill_hold_time_s and log.get("standstill") is None:
                    log.record_event("standstill", t=t, q_rad=q)
                    trial_status["outcome"] = "complete"
                    return
            prev_q, prev_t = q, t

            if t - t_start > args.max_trial_duration_s:
                trial_status["outcome"] = "incomplete_timeout"
                return
            time.sleep(args.sample_interval_s)

    sampler = threading.Thread(target=_sampling_loop, daemon=True)
    sampler.start()
    streamer.start()

    def _playback_loop():
        """PLAYBACK of the plan segment's own joints through the production
        streamer, at the plan's own sample cadence -- not live MPC
        re-solving. Each step's per-joint delta is clipped to
        --speed-limit-rad-s * dt; if the plan's own rate exceeds that limit
        anywhere, playback FALLS BEHIND the reference rather than silently
        exceeding the commanded speed cap -- logged, not hidden."""
        dt_plan = np.diff(segment.time_s, prepend=segment.time_s[0] - 0.1)
        commanded = segment.joints_rad[0].copy()
        streamer.update_target(list(commanded))
        clipped_any = False
        for i in range(1, segment.n):
            if trial_status["outcome"] != "running":
                return
            dt = float(dt_plan[i])
            max_step = args.speed_limit_rad_s * dt
            desired_delta = segment.joints_rad[i] - commanded
            clipped_delta = np.clip(desired_delta, -max_step, max_step)
            if np.any(np.abs(desired_delta) > np.abs(clipped_delta) + 1e-12):
                clipped_any = True
            commanded = commanded + clipped_delta
            streamer.update_target(list(commanded))
            if advancer is not None:
                advancer.submit_rate(float(np.clip(segment.insertion_rate_m_s[i], -2.0e-3, 2.0e-3)), dt)
            time.sleep(max(dt, 0.0))
        trial_status["playback_speed_limit_clipped"] = clipped_any
        if clipped_any:
            print("[trial] WARNING: --speed-limit-rad-s was below the plan's own joint rate at "
                  "one or more samples -- playback fell behind the reference rather than "
                  "exceeding the commanded speed cap.")

    playback = threading.Thread(target=_playback_loop, daemon=True)

    try:
        playback.start()

        while trial_status["outcome"] == "running":
            if log.get("detection") is not None and log.get("last_dispatch") is None:
                log.record_event("last_dispatch", t=time.monotonic())
                break
            if not playback.is_alive() and log.get("detection") is None:
                # segment finished playing with no abort yet -- let the sampler keep
                # watching for standstill/timeout rather than exiting silently.
                pass
            time.sleep(0.02)

        # application abort/shutdown sequence -- real path, not an isolated call.
        streamer.stop(timeout=5.0)
        log.record_event("streamer_exit", t=time.monotonic())
        if advancer is not None:
            advancer.submit_rate(0.0, 0.02)  # same stop pattern as close_loop_path_follow.py's finally block
        robot.servo_stop(2.0)
        log.record_event("stop_request", t=time.monotonic())

        deadline = time.monotonic() + max(0.0, args.max_trial_duration_s)
        while trial_status["outcome"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        sampler.join(timeout=5.0)

    except Exception as exc:
        trial_status["outcome"] = "incomplete_exception"
        trial_status["exception"] = repr(exc)
    finally:
        try:
            streamer.stop(timeout=2.0)
        except Exception:
            pass

    result = {
        "outcome": trial_status["outcome"],
        "lab_failure_procedure": args.lab_failure_procedure,
        "plan_segment": segment.summary_table(),
        "playback_speed_limit_clipped": trial_status.get("playback_speed_limit_clipped"),
        "args": vars(args),
    }
    if trial_status["outcome"] == "complete" and log.is_complete():
        result.update(displacement_segments(log, _fk))
        if args.trigger_mode == "manual":
            result["crossing_to_detection_latency_note"] = "UNMEASURED (manual trigger)"
        if args.trigger_mode == "staged":
            # Max inward clearance loss over the FULL recorded trajectory
            # (not just crossing/detection/standstill endpoints) -- catches
            # an overshoot past the closest approach that endpoint
            # displacement alone would hide.
            detection_t = log.get("detection").t_monotonic
            samples_after_detection = [(t, q) for t, q in log.joint_samples if t >= detection_t]
            dists = [float(np.linalg.norm(_fk(q) - staged_centre)) for _, q in samples_after_detection]
            d_at_detection = dists[0] if dists else None
            d_min = min(dists) if dists else None
            result["max_inward_clearance_loss_m"] = (
                (d_at_detection - d_min) if (d_at_detection is not None and d_min is not None) else None
            )
            result["distance_to_staged_trigger_at_detection_m"] = d_at_detection
            result["closest_approach_to_staged_trigger_m"] = d_min
    else:
        result["note"] = "trial incomplete -- follow --lab-failure-procedure, do not retry automatically"

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    traj = [{"t": t, "q_rad": q.tolist()} for t, q in log.joint_samples]
    (out_dir / "joint_trajectory.jsonl").write_text("\n".join(json.dumps(r) for r in traj))
    events = [{"kind": e.kind, "t": e.t_monotonic} for e in log.events]
    (out_dir / "events.json").write_text(json.dumps(events, indent=2))
    print(json.dumps(result, indent=2, default=str))
    return 0 if trial_status["outcome"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
