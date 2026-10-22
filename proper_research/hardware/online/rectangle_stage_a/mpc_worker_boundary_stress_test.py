#!/usr/bin/env python3
"""Validation for the process-isolated MPC architecture (2026-09-18), BEFORE
any live motion: the same no-motion sustained-load protocol as
`mpc_stationary_stress_test.py`, but routed through `MPCWorkerHandle`
(separate process) instead of a direct in-process controller call, with
DELIBERATE artificial solve-time injection (50/100/200ms sleeps in the
worker) to stress the deadline/fallback path harder than real OSQP timing
alone would.

Acceptance criteria (per the diagnosis that led here -- see
mpc_worker_process.py's docstring):
    1. rtde_ts_age_ms never grows unbounded -- the RTDE timestamp keeps
       advancing for the ENTIRE test, regardless of how slow the worker's
       solves are (this is the whole point: Process A's RTDE thread must
       never be starved by Process B's compute, however slow B gets).
    2. cam_age_ms stays bounded (< max_state_age_s) throughout.
    3. Every injected-sleep tick produces a clean deadline-miss fallback
       (command=0, logged), not a crash, hang, or unbounded age growth.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.mpc_worker_boundary_stress_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy \\
        --duration-s 120 --out /tmp/stress_boundary.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time

ROBOT_IP = "192.168.56.101"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--robot-ip", default=ROBOT_IP)
    p.add_argument("--deadline-ms", type=float, default=70.0)
    args = p.parse_args()

    import numpy as np
    import psutil
    import rtde_receive

    from proper_research.hardware.online.messages import HeartbeatLoop, now_monotonic
    from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
    from proper_research.hardware.online.state_stream import RobotJointStream
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        load_configuration_reference,
    )

    reference = load_configuration_reference(args.plan_dir, require_planned_beam_feasible=False)
    N, DT = 15, 1.0 / args.control_hz

    mpc_config_kwargs = dict(
        sample_period_s=0.1, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [-0.05]), state_max=tuple([2 * math.pi] * 6 + [0.20]),
        velocity_limit=tuple([0.10] * 6 + [2.0e-3]), acceleration_limit=tuple([0.40] * 6 + [0.02]),
    )
    beam_config_kwargs = dict(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )

    print("[boundary_stress] spawning MPC worker process (spawn context, single-threaded BLAS)...")
    worker = MPCWorkerHandle(
        plan_dir=args.plan_dir, schedule_path=args.schedule_cache,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
    )
    print("[boundary_stress] worker ready")

    print("[boundary_stress] opening camera (real hardware, robot stays still)...")
    from proper_research.hardware.online.advancer_excitation.measure_l0 import build_measurement_camera
    camera, _mapper = build_measurement_camera()
    camera.start()

    print(f"[boundary_stress] opening RobotJointStream({args.robot_ip}, poll_hz=125, receive_only=True)...")
    joint_stream = RobotJointStream(args.robot_ip, poll_hz=125.0, receive_only=True)
    joint_stream.start()

    rtde_ts = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    ts_state = {"last_ts": None, "last_change_wall": now_monotonic()}

    def _poll_timestamp(_t: float) -> None:
        ts = rtde_ts.getTimestamp()
        if ts_state["last_ts"] is None or ts != ts_state["last_ts"]:
            ts_state["last_ts"] = ts
            ts_state["last_change_wall"] = now_monotonic()

    ts_loop = HeartbeatLoop("rtde-timestamp-poll", _poll_timestamp, period_s=1.0 / 50.0)
    ts_loop.start()

    proc = psutil.Process()
    proc.cpu_percent(interval=None)

    print("[boundary_stress] warming up 3s...")
    time.sleep(3.0)

    q0 = np.asarray(reference.state[0, :6], dtype=float)
    l0 = float(reference.state[0, 6])
    p0 = np.asarray(reference.desired_position_m[0], dtype=float)
    prev_input = np.zeros(7, dtype=float)
    q_cmd = q0.copy()
    q_cmd_prev = q0.copy()

    n_ticks = int(round(args.duration_s * args.control_hz))
    injected_sleep_schedule = {}  # tick -> sleep_s, deliberately stress the deadline path
    for k in range(20, n_ticks, 40):
        injected_sleep_schedule[k] = [0.05, 0.10, 0.20][(k // 40) % 3]

    rows = []
    misses = 0
    t_scheduled = now_monotonic()
    print(f"[boundary_stress] running {n_ticks} ticks @ {args.control_hz:.1f}Hz "
          f"({args.duration_s:.0f}s), deadline={args.deadline_ms:.0f}ms, "
          f"{len(injected_sleep_schedule)} ticks with injected worker sleep...")
    for k in range(n_ticks):
        now = now_monotonic()
        if t_scheduled > now:
            time.sleep(t_scheduled - now)
        lateness_ms = (now_monotonic() - t_scheduled) * 1e3
        t_scheduled += DT

        joints_val, joints_age = joint_stream.joints_slot.get(None)
        cam_val, cam_age = camera.latest(None)
        rtde_ts_age = now_monotonic() - ts_state["last_change_wall"]

        ref_idx = k % int(reference.sample_count)
        sleep_s = injected_sleep_schedule.get(k, 0.0)

        t0 = now_monotonic()
        resp = worker.try_solve(
            kind="new", z_meas=np.concatenate([q0, [l0]]), q_cmd=q_cmd, q_cmd_prev=q_cmd_prev,
            insertion_cmd=l0, measured_beam_position=p0, control_index=ref_idx,
            previous_input=prev_input, worker_sleep_s=sleep_s,
            deadline_s=args.deadline_ms / 1e3,
        )
        t_roundtrip_ms = (now_monotonic() - t0) * 1e3

        if resp is not None and resp["status"] == "ok":
            command = np.asarray(resp["command"], dtype=float)
            hit = True
        else:
            command = np.zeros(7, dtype=float)
            hit = False
            misses += 1

        qd = np.clip(command[:6], -0.10, 0.10)
        delta_q = np.clip(qd * DT, -0.01, 0.01)
        q_cmd_prev = q_cmd.copy()
        q_cmd = q_cmd + delta_q
        prev_input = command.copy()

        rows.append({
            "k": k, "injected_sleep_ms": sleep_s * 1e3, "hit_deadline": hit,
            "t_roundtrip_ms": round(t_roundtrip_ms, 3),
            "t_solve_ms": resp["t_solve_ms"] if resp else None,
            "lateness_ms": round(lateness_ms, 3),
            "rtde_age_ms": round(joints_age * 1e3, 3) if math.isfinite(joints_age) else None,
            "rtde_ts_age_ms": round(rtde_ts_age * 1e3, 3),
            "cam_age_ms": round(cam_age * 1e3, 3) if math.isfinite(cam_age) else None,
        })

        if k % 100 == 0 or sleep_s > 0:
            print(f"[boundary_stress] k={k}/{n_ticks} inj_sleep={sleep_s*1e3:.0f}ms "
                  f"hit={hit} t_rt={t_roundtrip_ms:.1f}ms rtde_ts_age={rows[-1]['rtde_ts_age_ms']}ms "
                  f"cam_age={rows[-1]['cam_age_ms']}ms cpu%={proc.cpu_percent(interval=None):.0f}")

    print(f"\n[boundary_stress] total deadline misses: {misses}/{n_ticks} "
          f"(expected ~= {len(injected_sleep_schedule)} injected-sleep ticks)")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[boundary_stress] wrote {len(rows)} rows -> {args.out}")

    max_rtde_ts_age = max(r["rtde_ts_age_ms"] for r in rows)
    max_cam_age = max(r["cam_age_ms"] for r in rows if r["cam_age_ms"] is not None)
    print(f"\n=== ACCEPTANCE CHECK ===")
    print(f"max rtde_ts_age_ms over the whole run: {max_rtde_ts_age:.1f}ms "
          f"({'PASS' if max_rtde_ts_age < 500 else 'FAIL'} -- should never approach robot_max_age_s-scale growth)")
    print(f"max cam_age_ms over the whole run: {max_cam_age:.1f}ms "
          f"({'PASS' if max_cam_age < 500 else 'FAIL'} -- should stay well under max_state_age_s=0.5s)")

    ts_loop.stop()
    joint_stream.stop()
    camera.stop()
    worker.close()


if __name__ == "__main__":
    main()
