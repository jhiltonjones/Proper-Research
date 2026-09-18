#!/usr/bin/env python3
"""Stage 2 validation (2026-09-18): repeat the baseline preflight
reliability check (see `preflight_reliability_test.py`) but with the MPC
worker PRE-SPAWNED and already READY (warm-up solve done, idle in its IPC
loop) BEFORE the first preflight cycle -- the worker never receives a
solve request here, it just exists in the background the whole time. If
this is also clean, the worker's mere presence (a second live process,
daemon thread bookkeeping, etc.) isn't disturbing RTDE/camera -- isolating
phenomenon B (one-time worker startup load) from A (baseline reliability,
already checked separately) and C (sustained per-tick compute, checked by
mpc_worker_boundary_stress_test.py / mpc_stationary_stress_test.py).

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.preflight_with_idle_worker_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy --n 10
"""
from __future__ import annotations

import argparse
import time

from .run_mpc_delay_aware import spawn_and_warm_worker
from . import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--pause-s", type=float, default=2.0)
    args = p.parse_args()

    worker = spawn_and_warm_worker(
        plan_dir=args.plan_dir, schedule_cache=args.schedule_cache,
        control_hz=common.CONTROL_HZ, prediction_horizon=15,
        joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
        insertion_rate_limit_m_s=2.0e-3,
        joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
    )
    print(f"[preflight_idle_worker] worker idle in background, PID alive={worker._proc.is_alive()}")

    results = []
    for i in range(args.n):
        print(f"\n[preflight_idle_worker] cycle {i + 1}/{args.n}...")
        t0 = time.monotonic()
        try:
            q0, l0 = common.preflight(args.plan_dir)
            dt = time.monotonic() - t0
            worker_alive = worker._proc.is_alive()
            print(f"[preflight_idle_worker] cycle {i + 1}: OK in {dt:.2f}s "
                  f"(worker still alive={worker_alive})")
            results.append((True, dt, None))
        except Exception as exc:  # noqa: BLE001
            dt = time.monotonic() - t0
            print(f"[preflight_idle_worker] cycle {i + 1}: FAILED after {dt:.2f}s -- {exc!r}")
            results.append((False, dt, repr(exc)))
        if i < args.n - 1:
            time.sleep(args.pause_s)

    worker.close()

    n_ok = sum(1 for ok, _, _ in results if ok)
    print(f"\n=== SUMMARY: {n_ok}/{args.n} clean preflight cycles WITH an idle worker present ===")
    for i, (ok, dt, err) in enumerate(results):
        status = "OK" if ok else f"FAIL ({err})"
        print(f"  cycle {i + 1}: {status}  ({dt:.2f}s)")


if __name__ == "__main__":
    main()
