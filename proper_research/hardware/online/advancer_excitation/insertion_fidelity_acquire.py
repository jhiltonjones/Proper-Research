#!/usr/bin/env python3
"""LIVE acquisition for the load-dependent insertion-fidelity test.

Different hypothesis than acquire.py's free-beam ring-down test (which came
back weak/ambiguous and does NOT explain the closed-loop jaggedness -- see
this package's README): magnetic bending/load changes how much the advancer
actually moves the beam (roller slip, stick-slip, insufficient torque under
transverse load), so L_actual != L_controller during the real trajectory
even though the free (near-unloaded) mechanism looks fine. This is a much
more serious hidden-state error than a small ring-down.

The robot moves ONLY to place the magnet at a fixed bending pose (picked
from real, feasible plan waypoints via `common.select_bend_poses` -- never
an invented pose), then holds still for that pose's 5 repeats. Per repeat:
hold_pre -> command a small ΔL (2mm default) at the plan's own insertion
rate -> hold_post -> retract. L is measured TWO ways every frame: the
software-integral L_cmd (what the controller believes) and the vision-
measured L_meas = ||tip - pivot|| (ground truth, same technique
measure_l0.py validated).

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.insertion_fidelity_acquire \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir advancer_excitation_logs/rectangle_2026-09-18_load \\
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
from proper_research.hardware.online.messages import now_monotonic
from proper_research.hardware.online.state_stream import InsertionEstimate

from . import common
from .acquire import AcquireConfig, _run_phase, _wait_advancer_settled, build_camera
from .measure_l0 import PIVOT_XYZ, build_measurement_camera, measure_l_mm


@dataclass
class LoadFidelityConfig:
    out_dir: str = "advancer_excitation_logs_load"
    run_name: str = "insertion_fidelity"
    repeats: int = 5
    advance_mm: float = 2.0
    advance_rate_mm_s: float = 1.3
    hold_pre_s: float = 4.5
    hold_post_s: float = 4.5
    settle_after_move_s: float = 3.0
    live: bool = False
    advancer_port: str = "/dev/ttyACM0"
    robot_ip: str = common.ROBOT_IP
    move_speed: float = 0.2
    move_acceleration: float = 0.2


def move_robot_to(q6: np.ndarray, *, robot_ip: str, speed: float, acceleration: float, tol: float = 0.01) -> None:
    """One-shot moveJ to an arbitrary joint target, own short-lived
    connection -- mirrors rectangle_stage_a.common.reset_to_plan_initial but
    for an arbitrary plan waypoint, not just index 0."""
    from proper_research.hardware.ur_rtde_robot import URRTDERobot

    q6 = np.asarray(q6, dtype=float).reshape(6)
    robot = URRTDERobot(robot_ip, frequency=125.0)
    robot.connect()
    try:
        robot.servo_stop()
        ok = robot.move_j(list(q6), speed=speed, acceleration=acceleration)
        # The move itself can succeed while the immediately-following read
        # hits a transient RTDE hiccup (confirmed live 2026-09-18: moveJ
        # landed exactly on target, but the verification get_joints() raised
        # a disconnect right after) -- retry the READ, don't let a transient
        # receive-side glitch on an already-completed motion abort the run.
        q = None
        last_exc = None
        for attempt in range(4):
            try:
                q = np.array(robot.get_joints())
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"[insertion_fidelity] verification read attempt {attempt+1} failed "
                      f"({exc!r}), retrying...")
                time.sleep(1.0)
                try:
                    robot.reconnect()
                except Exception:
                    pass
        if q is None:
            raise RuntimeError(f"could not verify robot position after move_j "
                                f"(last error: {last_exc!r})")
        offset = float(np.max(np.abs(q - q6)))
        print(f"[insertion_fidelity] move_j returned {ok}; max offset from target: {offset:.5f} rad")
        if offset >= tol:
            raise RuntimeError(f"refusing to proceed: not at target joints after move "
                                f"(offset {offset:.5f} rad >= tol {tol} rad)")
    finally:
        robot.close()
    print("[insertion_fidelity] settling 3s before the harness opens its own connection...")
    time.sleep(3.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True, help="used both for pose selection and the "
                    "representative insertion rate")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-name", default="insertion_fidelity")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--advance-mm", type=float, default=2.0)
    p.add_argument("--advance-rate-mm-s", type=float, default=None,
                    help="default: the plan's own representative insertion rate")
    p.add_argument("--hold-pre-s", type=float, default=4.5)
    p.add_argument("--hold-post-s", type=float, default=4.5)
    p.add_argument("--settle-after-move-s", type=float, default=3.0)
    p.add_argument("--advancer-port", default="/dev/ttyACM0")
    p.add_argument("--robot-ip", default=common.ROBOT_IP)
    p.add_argument("--return-to-plan-initial", action="store_true", default=True)
    p.add_argument("--no-return-to-plan-initial", dest="return_to_plan_initial", action="store_false")
    p.add_argument("--live", action="store_true")
    args = p.parse_args()

    poses = common.select_bend_poses(args.plan_dir)
    lp = common.length_plan_from_plan_dir(args.plan_dir)
    rate_mm_s = args.advance_rate_mm_s if args.advance_rate_mm_s is not None else lp.representative_rate_m_s * 1e3

    cfg = LoadFidelityConfig(
        out_dir=args.out_dir, run_name=args.run_name, repeats=args.repeats,
        advance_mm=args.advance_mm, advance_rate_mm_s=rate_mm_s,
        hold_pre_s=args.hold_pre_s, hold_post_s=args.hold_post_s,
        settle_after_move_s=args.settle_after_move_s, live=args.live,
        advancer_port=args.advancer_port, robot_ip=args.robot_ip,
    )
    print(f"[insertion_fidelity] {'LIVE' if cfg.live else 'DRY-RUN'} -- poses:")
    for pose in poses:
        print(f"  {pose.label}: idx={pose.sample_index} b={pose.b_mm:.2f}mm "
              f"L_plan={pose.L_at_index_mm:.2f}mm q_deg={np.round(np.degrees(pose.q6), 1).tolist()}")
    print(f"[insertion_fidelity] advance={cfg.advance_mm}mm @ {cfg.advance_rate_mm_s:.3f}mm/s, "
          f"repeats={cfg.repeats}, hold_pre={cfg.hold_pre_s}s hold_post={cfg.hold_post_s}s")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"{cfg.run_name}_{stamp}.jsonl"

    move_robot_to(poses[0].q6, robot_ip=cfg.robot_ip, speed=cfg.move_speed, acceleration=cfg.move_acceleration)

    # Advancer + InsertionEstimate constructed first (same reasoning as
    # acquire.py: CameraSource captures the getter once at construction).
    advancer = AdvancerSink(AdvancerSinkConfig(port=cfg.advancer_port, dry_run=not cfg.live))
    advancer.start()
    print("[insertion_fidelity] measuring current insertion length by vision before seeding L_cmd...")
    probe_camera, probe_mapper = build_measurement_camera()
    probe_camera.start()
    try:
        L0_samples = measure_l_mm(probe_camera, probe_mapper, n=15, timeout_s=20.0)
    finally:
        probe_camera.stop()
    l0_mm = float(L0_samples.mean())
    print(f"[insertion_fidelity] measured L0 = {l0_mm:.3f}mm (std={L0_samples.std():.3f}mm)")
    insertion_est = InsertionEstimate(l0_mm * 1e-3, advancer)

    camera, mapper = build_camera(AcquireConfig(), insertion_est.get)
    camera.start()
    print("[insertion_fidelity] camera started, waiting for first tip detection...")
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
        raise RuntimeError("no tip estimate within warmup")
    print("[insertion_fidelity] first tip detected, streaming")

    rows: list = []
    current_mm = l0_mm
    try:
        for load_index, pose in enumerate(poses, start=1):
            if load_index > 1:
                move_robot_to(pose.q6, robot_ip=cfg.robot_ip, speed=cfg.move_speed, acceleration=cfg.move_acceleration)
            print(f"[insertion_fidelity] load {load_index} ({pose.label}, b={pose.b_mm:.2f}mm): "
                  f"settling {cfg.settle_after_move_s}s at fixed pose")
            time.sleep(cfg.settle_after_move_s)

            for rep in range(1, cfg.repeats + 1):
                print(f"[insertion_fidelity] {pose.label} repeat {rep}/{cfg.repeats}: hold_pre")
                _run_phase(
                    camera=camera, advancer=advancer, robot=None, rows=rows,
                    seconds=cfg.hold_pre_s, phase="hold_pre", L_index=load_index,
                    L_target_mm=current_mm, repeat_index=rep, rate_m_s=0.0, poll_period_s=0.001,
                )
                duration_s = cfg.advance_mm / cfg.advance_rate_mm_s
                print(f"[insertion_fidelity] {pose.label} repeat {rep}/{cfg.repeats}: advance "
                      f"{cfg.advance_mm}mm over {duration_s:.2f}s")
                _run_phase(
                    camera=camera, advancer=advancer, robot=None, rows=rows,
                    seconds=duration_s, phase="advance", L_index=load_index,
                    L_target_mm=current_mm, repeat_index=rep,
                    rate_m_s=cfg.advance_rate_mm_s * 1e-3, poll_period_s=0.001,
                )
                current_mm += cfg.advance_mm
                print(f"[insertion_fidelity] {pose.label} repeat {rep}/{cfg.repeats}: hold_post")
                _run_phase(
                    camera=camera, advancer=advancer, robot=None, rows=rows,
                    seconds=cfg.hold_post_s, phase="hold_post", L_index=load_index,
                    L_target_mm=current_mm, repeat_index=rep, rate_m_s=0.0, poll_period_s=0.001,
                )
                print(f"[insertion_fidelity] {pose.label} repeat {rep}/{cfg.repeats}: retract")
                duration_back_s = cfg.advance_mm / cfg.advance_rate_mm_s
                _run_phase(
                    camera=camera, advancer=advancer, robot=None, rows=rows,
                    seconds=duration_back_s, phase="retract", L_index=load_index,
                    L_target_mm=current_mm, repeat_index=rep,
                    rate_m_s=-cfg.advance_rate_mm_s * 1e-3, poll_period_s=0.001,
                )
                if advancer is not None:
                    _wait_advancer_settled(advancer)
                current_mm -= cfg.advance_mm
                time.sleep(1.0)
    finally:
        advancer.submit_rate(0.0, 0.0)
        time.sleep(0.2)
        advancer.stop()
        camera.stop()

    if args.return_to_plan_initial:
        print("[insertion_fidelity] returning robot to plan-initial (poses[0])...")
        move_robot_to(poses[0].q6, robot_ip=cfg.robot_ip, speed=cfg.move_speed, acceleration=cfg.move_acceleration)

    meta = {
        "_meta": True, "config": asdict(cfg),
        "poses": [asdict(pose) if not isinstance(pose.q6, np.ndarray) else
                  {**asdict(pose), "q6": pose.q6.tolist()} for pose in poses],
        "pivot_xyz": PIVOT_XYZ.tolist(), "l0_mm": l0_mm,
        "n_rows": len(rows), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\n[insertion_fidelity] done: {len(rows)} rows -> {log_path}")


if __name__ == "__main__":
    main()
