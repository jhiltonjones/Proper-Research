#!/usr/bin/env python3
"""LIVE acquisition for the advancer-excitation identification test.

Question this answers: does smooth commanded insertion produce real beam-tip
oscillation that a feedback controller would later chase? The robot is never
commanded by this script -- fix it (and the magnet) in one safe pose before
running, or pass --move-to-plan-initial once to place it there and release
the connection. Only the linear advancer moves.

Per length L_i in {L_min, L_min+dL/3, L_min+2dL/3, L_max} (read from
--plan-dir, or pass --l-min-mm/--l-max-mm directly):

  1. re-center the (software-integral) insertion estimate to L_i
  2. one ~5s stationary baseline (phase="baseline") -- the camera/noise floor
     at this beam length
  3. --repeats (default 5) repeats of:
       hold_pre (2s, phase="hold_pre")
       advance --advance-mm at --advance-rate-mm-s (phase="advance")
       hold_post (3s, phase="hold_post") <- the part analyze.py cares about
       retract back to L_i (phase="reposition", not analyzed)

Camera frames are logged AS THEY ARRIVE (native reconstruction rate, no
10Hz-tick downsampling) -- see the tight polling loop in `_run_phase`.

Nothing moves for real unless --live is passed; the default is a dry run
(advancer integrates + the sequence runs at full speed, but no serial bytes
go out and no beam motion can occur) so the sequencing/logging can be
verified on a laptop with no hardware attached.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.acquire \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --l0-mm 25.0 --out-dir advancer_excitation_logs/rectangle_2026-09-18 \\
        --live
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
from proper_research.hardware.online.camera_source import CameraConfig, CameraSource
from proper_research.hardware.online.messages import now_monotonic
from proper_research.hardware.online.state_stream import NewFrameTipMapper, StateStreamConfig

from . import common


@dataclass
class AcquireConfig:
    out_dir: str = "advancer_excitation_logs"
    run_name: str = "advancer_excitation"

    l0_mm: float = 0.0            # required: the beam's ACTUAL current insertion (no encoder)
    l_min_mm: float = 0.0
    l_max_mm: float = 0.0

    repeats: int = 5
    advance_mm: float = 1.5
    advance_rate_mm_s: float = 1.3
    hold_pre_s: float = 2.0
    hold_post_s: float = 3.0
    baseline_s: float = 5.0
    reposition_rate_mm_s: float = 1.5   # faster/slower moves between lengths & retract
    settle_after_reposition_s: float = 2.0

    live: bool = False
    advancer_port: str = "/dev/ttyACM0"
    robot_ip: str = common.ROBOT_IP
    read_robot_joints: bool = True

    camera_reconstruct_period_s: float = 0.005   # target; actual capped by vision compute cost
    poll_period_s: float = 0.001                  # busy-poll interval for "new frame arrived"


def _run_phase(
    *,
    camera: CameraSource,
    advancer: Optional[AdvancerSink],
    robot,
    rows: list,
    seconds: float,
    phase: str,
    L_index: int,
    L_target_mm: float,
    repeat_index: int,
    rate_m_s: float = 0.0,
    poll_period_s: float = 0.001,
) -> int:
    """Busy-poll for new camera frames for `seconds`, logging every one (no
    downsampling). If rate_m_s != 0, keeps submitting it to the advancer at
    the polling cadence (dt = actual elapsed since the previous submission),
    exactly like a real controller submitting one rate command per tick."""
    n_logged = 0
    deadline = now_monotonic() + seconds
    last_seq = -1
    last_submit_t = now_monotonic()
    while True:
        now = now_monotonic()
        if now >= deadline:
            break
        if advancer is not None and rate_m_s != 0.0:
            dt = now - last_submit_t
            advancer.submit_rate(rate_m_s, dt)
            last_submit_t = now
        est, seq = camera.state_slot.peek()
        if est is not None and seq != last_seq:
            last_seq = seq
            fb = advancer.feedback() if advancer is not None else None
            joints = None
            if robot is not None:
                q = robot.latest_joints(0.5)
                joints = None if q is None else [round(float(v), 6) for v in q]
            rows.append({
                "t_monotonic": now,
                "L_index": L_index,
                "L_target_mm": L_target_mm,
                "repeat_index": repeat_index,
                "phase": phase,
                "frame_index": int(est.frame_index),
                "tip_position_m": [round(float(v), 6) for v in est.tip_position_m],
                "tip_tangent": [round(float(v), 6) for v in est.tip_tangent],
                "insertion_est_m": round(float(est.insertion_length_m), 6),
                "advancer_commanded_mm": None if fb is None else round(fb.commanded_distance_mm, 4),
                "advancer_dispatched_mm": None if fb is None else round(fb.dispatched_distance_mm, 4),
                "advancer_residual_mm": None if fb is None else round(fb.residual_mm, 4),
                "advancer_in_flight": None if fb is None else fb.commands_in_flight,
                "steps_dispatched_total": None if fb is None else fb.steps_dispatched_total,
                "steps_reported_total": None if fb is None else fb.steps_reported_total,
                "cmd_rate_m_s": rate_m_s,
                "robot_joints": joints,
            })
            n_logged += 1
        time.sleep(poll_period_s)
    if advancer is not None and rate_m_s != 0.0:
        advancer.submit_rate(0.0, 0.0)
    return n_logged


def _wait_advancer_settled(advancer: AdvancerSink, *, timeout_s: float = 15.0) -> None:
    deadline = now_monotonic() + timeout_s
    while now_monotonic() < deadline:
        fb = advancer.feedback()
        if abs(fb.residual_mm) < advancer.config.min_command_mm and fb.commands_in_flight == 0:
            return
        time.sleep(0.05)
    print("[acquire] WARNING: advancer did not report settled within timeout; continuing anyway")


def _go_to_length(
    *, advancer: Optional[AdvancerSink], camera: CameraSource, robot, rows: list,
    current_mm: float, target_mm: float, rate_mm_s: float, settle_s: float,
    L_index: int, poll_period_s: float,
) -> float:
    delta_mm = target_mm - current_mm
    if abs(delta_mm) < 1e-6:
        return current_mm
    duration_s = abs(delta_mm) / max(rate_mm_s, 1e-6)
    rate_m_s = np.sign(delta_mm) * rate_mm_s * 1e-3
    print(f"[acquire] reposition {current_mm:.3f}mm -> {target_mm:.3f}mm "
          f"(delta={delta_mm:+.3f}mm, {duration_s:.2f}s @ {rate_mm_s:.2f}mm/s)")
    _run_phase(
        camera=camera, advancer=advancer, robot=robot, rows=rows,
        seconds=duration_s, phase="reposition", L_index=L_index, L_target_mm=target_mm,
        repeat_index=-1, rate_m_s=rate_m_s, poll_period_s=poll_period_s,
    )
    if advancer is not None:
        _wait_advancer_settled(advancer)
    time.sleep(settle_s)
    return target_mm


def build_camera(cfg: AcquireConfig, insertion_length_getter) -> tuple[CameraSource, NewFrameTipMapper]:
    scfg = StateStreamConfig(
        exposure=29.0, marker_min_count=2,
        reconstruct_period_s=cfg.camera_reconstruct_period_s,
    )
    mapper = NewFrameTipMapper(scfg)
    camera = CameraSource(
        CameraConfig(
            cam_index=0, exposure=scfg.exposure,
            warmup_frames=scfg.warmup_frames, grab_period_s=scfg.grab_period_s,
            reconstruct_period_s=scfg.reconstruct_period_s,
            image_filename="/dev/shm/advancer_excitation_frame.png",
            roi_polygon_path=scfg.roi_polygon_path,
            manual_boundary_path=scfg.manual_boundary_path,
            pivot_hint=tuple(scfg.pivot_hint_px),
        ),
        pivot_point_pose6=np.asarray(scfg.T_robot_beam_pose6),
        robot_joints_getter=None, robot_pose_getter=None,
        insertion_length_getter=insertion_length_getter,
        frame_processor=mapper,
    )
    return camera, mapper


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", default=None,
                    help="reads L_min/L_max/representative insertion rate from the plan; "
                         "omit and pass --l-min-mm/--l-max-mm/--advance-rate-mm-s directly instead")
    p.add_argument("--l0-mm", type=float, required=True,
                    help="the beam's ACTUAL current insertion length right now (no encoder -- "
                         "you must know this, e.g. from the same calibration used to seed "
                         "close_loop_path_follow's initial_insertion_m)")
    p.add_argument("--l-min-mm", type=float, default=None)
    p.add_argument("--l-max-mm", type=float, default=None)
    p.add_argument("--advance-rate-mm-s", type=float, default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="advancer_excitation")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--advance-mm", type=float, default=1.5)
    p.add_argument("--hold-pre-s", type=float, default=2.0)
    p.add_argument("--hold-post-s", type=float, default=3.0)
    p.add_argument("--baseline-s", type=float, default=5.0)
    p.add_argument("--reposition-rate-mm-s", type=float, default=1.5)
    p.add_argument("--settle-after-reposition-s", type=float, default=2.0)
    p.add_argument("--advancer-port", default="/dev/ttyACM0")
    p.add_argument("--robot-ip", default=common.ROBOT_IP)
    p.add_argument("--no-robot-joints", action="store_true",
                    help="skip the (best-effort, non-blocking) robot-fixed sanity log")
    p.add_argument("--move-to-plan-initial", action="store_true",
                    help="one-shot moveJ to --plan-dir's initial joints before starting, then "
                         "releases the connection -- this script issues no further robot commands")
    p.add_argument("--live", action="store_true",
                    help="actually open the serial port and move the advancer. Omit for a dry "
                         "run (sequencing + camera + logging exercised, nothing moves).")
    args = p.parse_args()

    if args.plan_dir is not None:
        lp = common.length_plan_from_plan_dir(args.plan_dir)
        l_min_mm = args.l_min_mm if args.l_min_mm is not None else lp.L_min_m * 1e3
        l_max_mm = args.l_max_mm if args.l_max_mm is not None else lp.L_max_m * 1e3
        rate_mm_s = args.advance_rate_mm_s if args.advance_rate_mm_s is not None else lp.representative_rate_m_s * 1e3
    else:
        if args.l_min_mm is None or args.l_max_mm is None or args.advance_rate_mm_s is None:
            raise SystemExit("without --plan-dir you must pass --l-min-mm, --l-max-mm and --advance-rate-mm-s")
        l_min_mm, l_max_mm, rate_mm_s = args.l_min_mm, args.l_max_mm, args.advance_rate_mm_s

    cfg = AcquireConfig(
        out_dir=args.out_dir, run_name=args.run_name, l0_mm=args.l0_mm,
        l_min_mm=l_min_mm, l_max_mm=l_max_mm, repeats=args.repeats,
        advance_mm=args.advance_mm, advance_rate_mm_s=rate_mm_s,
        hold_pre_s=args.hold_pre_s, hold_post_s=args.hold_post_s, baseline_s=args.baseline_s,
        reposition_rate_mm_s=args.reposition_rate_mm_s,
        settle_after_reposition_s=args.settle_after_reposition_s,
        live=args.live, advancer_port=args.advancer_port, robot_ip=args.robot_ip,
        read_robot_joints=not args.no_robot_joints,
    )
    d = l_max_mm - l_min_mm
    test_lengths_mm = [l_min_mm, l_min_mm + d / 3.0, l_min_mm + 2.0 * d / 3.0, l_max_mm]
    print(f"[acquire] {'LIVE' if cfg.live else 'DRY-RUN'} -- test lengths (mm): "
          f"{[round(v, 2) for v in test_lengths_mm]}, advance={cfg.advance_mm}mm "
          f"@ {cfg.advance_rate_mm_s:.3f}mm/s, repeats={cfg.repeats}")

    if args.move_to_plan_initial:
        if args.plan_dir is None:
            raise SystemExit("--move-to-plan-initial requires --plan-dir")
        from proper_research.hardware.online.rectangle_stage_a import common as rsa_common
        q0, _l0 = rsa_common.load_plan_initial_state(args.plan_dir)
        rsa_common.reset_to_plan_initial(q0, robot_ip=cfg.robot_ip)
        print("[acquire] robot placed at plan-initial and connection released; "
              "this script will not command it again")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"{cfg.run_name}_{stamp}.jsonl"

    robot = None
    if cfg.read_robot_joints:
        try:
            from proper_research.hardware.online.state_stream import RobotJointStream
            robot = RobotJointStream(cfg.robot_ip, poll_hz=125.0)
            robot.start()
            time.sleep(0.3)
            print("[acquire] robot-fixed sanity stream connected (receive-only, no commands issued)")
        except Exception as exc:  # noqa: BLE001
            print(f"[acquire] WARNING: could not open robot-fixed sanity stream ({exc!r}); "
                  f"continuing without it -- this does not block the test")
            robot = None

    # Advancer built (and its InsertionEstimate wired into the camera) BEFORE
    # the camera starts -- otherwise every StateEstimate's insertion_length_m
    # is a placeholder forever, not the actually-tracked (software-integral)
    # insertion, since CameraSource captures its getter once at construction.
    advancer = AdvancerSink(AdvancerSinkConfig(port=cfg.advancer_port, dry_run=not cfg.live))
    advancer.start()
    from proper_research.hardware.online.state_stream import InsertionEstimate
    insertion_est = InsertionEstimate(cfg.l0_mm * 1e-3, advancer)

    camera, mapper = build_camera(cfg, insertion_est.get)
    camera.start()
    print("[acquire] camera started, waiting for first tip detection...")
    t_deadline = now_monotonic() + 20.0
    first = None
    while now_monotonic() < t_deadline:
        first, _ = camera.latest()
        if first is not None:
            break
        if not camera.healthy:
            raise RuntimeError(f"camera unhealthy during warmup: {camera.last_error!r}")
        time.sleep(0.05)
    if first is None:
        raise RuntimeError(f"no tip estimate within warmup; detection_failures={mapper.detection_failures}, "
                            f"last_error={mapper.last_error!r}")
    print("[acquire] first tip detected, streaming")

    rows: list = []
    current_mm = cfg.l0_mm
    try:
        for L_index, L_target_mm in enumerate(test_lengths_mm, start=1):
            current_mm = _go_to_length(
                advancer=advancer, camera=camera, robot=robot, rows=rows,
                current_mm=current_mm, target_mm=L_target_mm,
                rate_mm_s=cfg.reposition_rate_mm_s, settle_s=cfg.settle_after_reposition_s,
                L_index=L_index, poll_period_s=cfg.poll_period_s,
            )
            print(f"[acquire] L{L_index} = {L_target_mm:.3f}mm: baseline ({cfg.baseline_s}s, stationary)")
            _run_phase(
                camera=camera, advancer=advancer, robot=robot, rows=rows,
                seconds=cfg.baseline_s, phase="baseline", L_index=L_index,
                L_target_mm=L_target_mm, repeat_index=0, rate_m_s=0.0,
                poll_period_s=cfg.poll_period_s,
            )

            for rep in range(1, cfg.repeats + 1):
                print(f"[acquire] L{L_index} repeat {rep}/{cfg.repeats}: hold_pre")
                _run_phase(
                    camera=camera, advancer=advancer, robot=robot, rows=rows,
                    seconds=cfg.hold_pre_s, phase="hold_pre", L_index=L_index,
                    L_target_mm=L_target_mm, repeat_index=rep, rate_m_s=0.0,
                    poll_period_s=cfg.poll_period_s,
                )
                duration_s = cfg.advance_mm / cfg.advance_rate_mm_s
                print(f"[acquire] L{L_index} repeat {rep}/{cfg.repeats}: advance "
                      f"{cfg.advance_mm}mm over {duration_s:.2f}s")
                _run_phase(
                    camera=camera, advancer=advancer, robot=robot, rows=rows,
                    seconds=duration_s, phase="advance", L_index=L_index,
                    L_target_mm=L_target_mm, repeat_index=rep,
                    rate_m_s=cfg.advance_rate_mm_s * 1e-3, poll_period_s=cfg.poll_period_s,
                )
                current_mm += cfg.advance_mm
                print(f"[acquire] L{L_index} repeat {rep}/{cfg.repeats}: hold_post (ring-down)")
                _run_phase(
                    camera=camera, advancer=advancer, robot=robot, rows=rows,
                    seconds=cfg.hold_post_s, phase="hold_post", L_index=L_index,
                    L_target_mm=L_target_mm, repeat_index=rep, rate_m_s=0.0,
                    poll_period_s=cfg.poll_period_s,
                )
                current_mm = _go_to_length(
                    advancer=advancer, camera=camera, robot=robot, rows=rows,
                    current_mm=current_mm, target_mm=L_target_mm,
                    rate_mm_s=cfg.reposition_rate_mm_s, settle_s=cfg.settle_after_reposition_s,
                    L_index=L_index, poll_period_s=cfg.poll_period_s,
                )
    finally:
        advancer.submit_rate(0.0, 0.0)
        time.sleep(0.2)
        advancer.stop()
        camera.stop()
        if robot is not None:
            robot.stop()

    meta = {
        "_meta": True, "config": asdict(cfg), "test_lengths_mm": test_lengths_mm,
        "n_rows": len(rows), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\n[acquire] done: {len(rows)} rows -> {log_path}")
    for L_index, L_target_mm in enumerate(test_lengths_mm, start=1):
        n = sum(1 for r in rows if r["L_index"] == L_index and r["phase"] == "hold_post")
        print(f"  L{L_index}={L_target_mm:.2f}mm: {n} hold_post samples across {cfg.repeats} repeats "
              f"({n / max(cfg.repeats, 1):.0f}/repeat)")


if __name__ == "__main__":
    main()
