#!/usr/bin/env python3
"""No-motion sustained-load diagnostic for the delay-aware MPC's live
staleness failures (2026-09-18). Reproduces the runtime conditions --
real camera thread, real RTDE robot-reader thread, real MPC QP solve at
10Hz -- WITHOUT spending a beam/insertion cycle: no servoJ, no advancer
motion, robot stays exactly where it is.

Logs, every tick, time-aligned:
    t_solve_ms         -- wall-clock duration of the real MPC solve call
    lateness_ms         -- actual tick time minus the FIXED absolute
                            schedule (t_scheduled_k = t_scheduled_0 + k*dt),
                            not solve-then-sleep(dt) (which silently makes
                            the real period ~dt + t_solve)
    rtde_age_ms          -- age of RobotJointStream.joints_slot's last
                            update (this already distinguishes a frozen
                            reader from a genuinely-stationary-but-noisy
                            robot -- RobotJointStream only refreshes the
                            slot's timestamp when the raw reading actually
                            CHANGES, and a real robot's joints are never
                            bit-identical two polls running, see
                            state_stream.py's own docstring on this)
    rtde_ts_age_ms        -- age since the RTDE CONTROLLER's own packet
                            timestamp (rtde_receive.getTimestamp(), a
                            SEPARATE connection from RobotJointStream's, so
                            no shared-object contention) last changed --
                            independent cross-check of rtde_age_ms using
                            the controller's own clock, not just value
                            inequality
    cam_age_ms          -- age of CameraSource's last published estimate
    cam_seq             -- CameraSource's internal LatestSlot sequence
                            counter (monotonically increasing iff the
                            camera thread is genuinely publishing)

Run twice to test the thread-oversubscription hypothesis: once with the
process's default BLAS threading, once with OMP_NUM_THREADS=OPENBLAS_NUM_
THREADS=MKL_NUM_THREADS=1 (set via --single-threaded, which re-execs this
script with those env vars set BEFORE numpy is ever imported -- setting
them after import has no effect on an already-initialized BLAS backend).

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.mpc_stationary_stress_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy \\
        --duration-s 180 --out /tmp/stress_default.jsonl

    python -m proper_research.hardware.online.rectangle_stage_a.mpc_stationary_stress_test \\
        --plan-dir ... --schedule-cache ... --duration-s 180 \\
        --out /tmp/stress_singlethread.jsonl --single-threaded
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

ROBOT_IP = "192.168.56.101"


def _reexec_single_threaded() -> None:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    argv = [a for a in sys.argv if a != "--single-threaded"]
    os.execve(sys.executable, [sys.executable] + argv, env)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--duration-s", type=float, default=180.0)
    p.add_argument("--control-hz", type=float, default=10.0)
    p.add_argument("--robot-ip", default=ROBOT_IP)
    p.add_argument("--single-threaded", action="store_true",
                    help="re-exec with BLAS thread count forced to 1 before numpy import")
    args = p.parse_args()

    if args.single_threaded and os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        _reexec_single_threaded()

    # numpy/scipy imported only AFTER any re-exec above, so the thread-count
    # env vars (if set) are in effect before the BLAS backend initializes.
    import numpy as np
    import psutil
    import rtde_receive

    from proper_research.controllers.mpc_delay_aware.delay_aware_mpc import (
        DelayAwareBeamOutputTrackingMPC,
    )
    from proper_research.hardware.online.messages import HeartbeatLoop, now_monotonic
    from proper_research.hardware.online.state_stream import RobotJointStream
    from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
        BeamOutputMPCConfig,
    )
    from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
        ConfigurationMPCConfig, load_configuration_reference,
    )

    print(f"[stress] single_threaded={args.single_threaded} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
          f"nproc={os.cpu_count()}")

    reference = load_configuration_reference(args.plan_dir, require_planned_beam_feasible=False)
    schedule = np.load(args.schedule_cache)
    N, DT = 15, 1.0 / args.control_hz
    vlim = tuple([0.10] * 6 + [2.0e-3])
    alim = tuple([0.40] * 6 + [10.0 * 2.0e-3])
    mpc_config = ConfigurationMPCConfig(
        sample_period_s=0.1, prediction_horizon=N,
        state_min=tuple([-2 * math.pi] * 6 + [-0.05]), state_max=tuple([2 * math.pi] * 6 + [0.20]),
        velocity_limit=vlim, acceleration_limit=alim,
    )
    beam_config = BeamOutputMPCConfig(
        position_error_scale_m=(5.0e-4, 5.0e-4, 5.0e-4), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    controller = DelayAwareBeamOutputTrackingMPC(
        reference=reference, config=mpc_config, beam_config=beam_config,
        reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
    )
    print("[stress] delay-aware MPC controller built")

    print("[stress] opening camera (real hardware, robot stays still)...")
    # Reuse the same production tip-only pipeline factory acquire.py /
    # reset_insertion.py use, rather than hand-rolling CameraConfig here, so
    # this diagnostic opens the camera identically to a real run.
    from proper_research.hardware.online.advancer_excitation.measure_l0 import build_measurement_camera
    camera, _mapper = build_measurement_camera()
    camera.start()

    print(f"[stress] opening RobotJointStream({args.robot_ip}, poll_hz=125, receive_only=True)...")
    joint_stream = RobotJointStream(args.robot_ip, poll_hz=125.0, receive_only=True)
    joint_stream.start()

    print("[stress] opening a SEPARATE rtde_receive connection for getTimestamp() cross-check...")
    rtde_ts = rtde_receive.RTDEReceiveInterface(args.robot_ip)
    ts_lock = threading.Lock()
    ts_state = {"last_ts": None, "last_change_wall": now_monotonic()}

    def _poll_timestamp(_t: float) -> None:
        ts = rtde_ts.getTimestamp()
        with ts_lock:
            if ts_state["last_ts"] is None or ts != ts_state["last_ts"]:
                ts_state["last_ts"] = ts
                ts_state["last_change_wall"] = now_monotonic()

    ts_loop = HeartbeatLoop("rtde-timestamp-poll", _poll_timestamp, period_s=1.0 / 50.0)
    ts_loop.start()

    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime the internal counter

    print("[stress] waiting 3s for both streams to warm up...")
    time.sleep(3.0)

    q0 = np.asarray(reference.state[0, :6], dtype=float)
    l0 = float(reference.state[0, 6])
    p0 = np.asarray(reference.desired_position_m[0], dtype=float)
    prev_input = np.zeros(7, dtype=float)

    n_ticks = int(round(args.duration_s * args.control_hz))
    rows = []
    t_scheduled = now_monotonic()
    print(f"[stress] running {n_ticks} ticks @ {args.control_hz:.1f}Hz "
          f"({args.duration_s:.0f}s), NO motion commands...")
    for k in range(n_ticks):
        now = now_monotonic()
        if t_scheduled > now:
            time.sleep(t_scheduled - now)
        lateness_ms = (now_monotonic() - t_scheduled) * 1e3
        t_scheduled += DT  # fixed absolute schedule, not solve-then-sleep(DT)

        joints_val, joints_age = joint_stream.joints_slot.get(None)
        cam_val, cam_age = camera.latest(None)
        cam_seq = camera.state_slot._seq  # noqa: SLF001 -- diagnostic-only introspection
        with ts_lock:
            rtde_ts_age = now_monotonic() - ts_state["last_change_wall"]

        control_index = k % int(reference.sample_count)
        ref_idx = min(control_index, int(reference.sample_count) - 1)

        t0 = now_monotonic()
        step = controller.solve_delay_aware(
            z_meas=np.concatenate([q0, [l0]]), q_cmd=q0, q_cmd_prev=q0, insertion_m=l0,
            measured_beam_position=p0, control_index=ref_idx, previous_input=prev_input,
        )
        t_solve_ms = (now_monotonic() - t0) * 1e3
        prev_input = np.asarray(step.command, dtype=float)

        rows.append({
            "k": k,
            "t_solve_ms": round(t_solve_ms, 3),
            "lateness_ms": round(lateness_ms, 3),
            "rtde_age_ms": round(joints_age * 1e3, 3) if math.isfinite(joints_age) else None,
            "rtde_ts_age_ms": round(rtde_ts_age * 1e3, 3),
            "cam_age_ms": round(cam_age * 1e3, 3) if math.isfinite(cam_age) else None,
            "cam_seq": int(cam_seq),
            "solver_success": bool(step.success),
        })

        if k % 100 == 0:
            print(f"[stress] k={k}/{n_ticks} t_solve={t_solve_ms:.1f}ms "
                  f"rtde_age={rows[-1]['rtde_age_ms']}ms rtde_ts_age={rows[-1]['rtde_ts_age_ms']}ms "
                  f"cam_age={rows[-1]['cam_age_ms']}ms lateness={lateness_ms:.1f}ms "
                  f"cpu%={proc.cpu_percent(interval=None):.0f} threads={proc.num_threads()}")

    joint_stream_stats = dict(
        reads=joint_stream.reads, read_failures=joint_stream.read_failures,
        stale_repeats=joint_stream.stale_repeats, last_error=joint_stream.last_error,
    )
    print(f"[stress] RobotJointStream stats: {joint_stream_stats}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[stress] wrote {len(rows)} rows -> {args.out}")

    ts_loop.stop()
    joint_stream.stop()
    camera.stop()


if __name__ == "__main__":
    main()
