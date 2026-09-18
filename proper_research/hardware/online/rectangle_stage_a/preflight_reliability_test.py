#!/usr/bin/env python3
"""Baseline RTDE/network reliability check (2026-09-18) -- NO MPC worker,
no camera-beyond-preflight's-own-check, no control loop. Just
`common.preflight()` (robot-joint reset via a fresh RTDE connection +
camera health check) repeated N times in a row, each with a full
connect/read/disconnect cycle, timed.

This isolates phenomenon A (baseline RTDE/network reliability) from B
(worker startup load) and C (sustained per-tick MPC computation) -- see
the investigation that led here. If this alone isn't clean today, that's
an infrastructure problem upstream of anything in mpc_worker_process.py /
process_isolated_adapter.py, and no amount of architecture work downstream
will fix it -- stop and address this first.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.preflight_reliability_test \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --n 10
"""
from __future__ import annotations

import argparse
import time

from . import common


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--pause-s", type=float, default=2.0, help="pause between cycles")
    args = p.parse_args()

    results = []
    for i in range(args.n):
        print(f"\n[preflight_reliability] cycle {i + 1}/{args.n}...")
        t0 = time.monotonic()
        try:
            q0, l0 = common.preflight(args.plan_dir)
            dt = time.monotonic() - t0
            print(f"[preflight_reliability] cycle {i + 1}: OK in {dt:.2f}s "
                  f"(q0={q0.round(4).tolist()}, L0={l0*1000:.2f}mm)")
            results.append((True, dt, None))
        except Exception as exc:  # noqa: BLE001 -- want every failure, not a stack trace abort
            dt = time.monotonic() - t0
            print(f"[preflight_reliability] cycle {i + 1}: FAILED after {dt:.2f}s -- {exc!r}")
            results.append((False, dt, repr(exc)))
        if i < args.n - 1:
            time.sleep(args.pause_s)

    n_ok = sum(1 for ok, _, _ in results if ok)
    print(f"\n=== SUMMARY: {n_ok}/{args.n} clean preflight cycles ===")
    for i, (ok, dt, err) in enumerate(results):
        status = "OK" if ok else f"FAIL ({err})"
        print(f"  cycle {i + 1}: {status}  ({dt:.2f}s)")

    if n_ok < args.n:
        print(f"\n[preflight_reliability] NOT clean -- {args.n - n_ok}/{args.n} cycles failed. "
              f"This is a baseline infrastructure problem, independent of the MPC worker "
              f"architecture. Do not proceed to worker-related validation until this passes.")
    else:
        print(f"\n[preflight_reliability] CLEAN -- {n_ok}/{args.n}. Baseline RTDE/network "
              f"reliability looks fine right now; safe to proceed to the next validation stage.")


if __name__ == "__main__":
    main()
