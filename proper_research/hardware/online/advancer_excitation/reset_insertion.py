#!/usr/bin/env python3
"""Close the loop on vision-measured L to actually put the advancer back at
a target insertion length -- not just tell the software what L0 "should"
be (which is all `--initial-insertion-m`/`cfg.initial_insertion_m` do; the
advancer has no encoder, so nothing physically moves unless you command it).

Built after discovering live (2026-09-18) that running open-loop-C rectangle
repeats back-to-back with no insertion reset between them silently drifted
the physical insertion by +18.85mm over 2 repeats (each repeat's net
insertion motion carries over, uncorrected) -- by repeat 3 marker detection
failed outright ("Expected at least 3 red markers, found 2") because the
beam was far outside the plan's expected range. `run_open_loop_c.py`'s own
preflight only resets robot joints, not insertion -- this script is the
missing piece, meant to run BEFORE each repeat's `run_open_loop_c.py` call.

Usage
-----
    python -m proper_research.hardware.online.advancer_excitation.reset_insertion \\
        --target-mm 25.0 --live
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from proper_research.hardware.online.advancer_sink import AdvancerSink, AdvancerSinkConfig
from proper_research.hardware.online.messages import now_monotonic

from .measure_l0 import build_measurement_camera, measure_l_mm


def reset_insertion(
    target_mm: float, *, live: bool, advancer_port: str = "/dev/ttyACM0",
    rate_mm_s: float = 1.5, tolerance_mm: float = 0.3, max_iters: int = 5,
) -> float:
    camera, mapper = build_measurement_camera()
    camera.start()
    advancer = AdvancerSink(AdvancerSinkConfig(port=advancer_port, dry_run=not live))
    advancer.start()
    try:
        current = float(measure_l_mm(camera, mapper, n=10, timeout_s=20.0).mean())
        print(f"[reset_insertion] start: L={current:.3f}mm target={target_mm:.3f}mm")
        for it in range(max_iters):
            delta = target_mm - current
            if abs(delta) <= tolerance_mm:
                print(f"[reset_insertion] within tolerance ({abs(delta):.3f}mm <= {tolerance_mm}mm) after {it} correction(s)")
                break
            duration_s = abs(delta) / rate_mm_s
            rate_m_s = np.sign(delta) * rate_mm_s * 1e-3
            print(f"[reset_insertion] iter {it+1}: delta={delta:+.3f}mm, driving {duration_s:.2f}s @ {rate_mm_s:.2f}mm/s")
            t_last = now_monotonic()
            deadline = t_last + duration_s
            while now_monotonic() < deadline:
                now = now_monotonic()
                advancer.submit_rate(rate_m_s, now - t_last)
                t_last = now
                time.sleep(0.02)
            advancer.submit_rate(0.0, 0.0)
            deadline_settle = now_monotonic() + 5.0
            while now_monotonic() < deadline_settle:
                fb = advancer.feedback()
                if abs(fb.residual_mm) < advancer.config.min_command_mm and fb.commands_in_flight == 0:
                    break
                time.sleep(0.05)
            time.sleep(1.0)  # let ring-down settle before re-measuring
            current = float(measure_l_mm(camera, mapper, n=10, timeout_s=20.0).mean())
            print(f"[reset_insertion] iter {it+1} result: L={current:.3f}mm (target {target_mm:.3f}mm)")
        else:
            print(f"[reset_insertion] WARNING: did not converge within {max_iters} iterations "
                  f"(final L={current:.3f}mm, target={target_mm:.3f}mm)")
        return current
    finally:
        advancer.submit_rate(0.0, 0.0)
        time.sleep(0.2)
        advancer.stop()
        camera.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-mm", type=float, required=True)
    p.add_argument("--advancer-port", default="/dev/ttyACM0")
    p.add_argument("--rate-mm-s", type=float, default=1.5)
    p.add_argument("--tolerance-mm", type=float, default=0.3)
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--live", action="store_true")
    args = p.parse_args()
    final = reset_insertion(
        args.target_mm, live=args.live, advancer_port=args.advancer_port,
        rate_mm_s=args.rate_mm_s, tolerance_mm=args.tolerance_mm, max_iters=args.max_iters,
    )
    print(f"\n[reset_insertion] final L = {final:.3f}mm")


if __name__ == "__main__":
    main()
