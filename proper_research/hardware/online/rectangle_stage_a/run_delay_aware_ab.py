#!/usr/bin/env python3
"""Live old-MPC-vs-delay-aware-MPC development A/B (2026-09-18): 3 paired
blocks / 6 runs, randomized order within each block (e.g. AB, BA, AB).

    A = condition A: run_mpc_delay_baseline (mpc_ltv_offline, V_f=0, N=15)
    B = condition B: run_mpc_delay_aware (d=2, beta_d=1.0, V_f=0)

Before EVERY run: `common.preflight()` (robot-joint reset + camera/safety
checks) AND `advancer_excitation.reset_insertion.reset_insertion()`
(physical insertion reset via vision-closed-loop advancer control --
REQUIRED, not optional: without it, insertion silently drifts run-to-run
and vision marker detection eventually fails outright -- see
`reset_insertion.py`'s docstring for the 2026-09-18 incident that
motivated it, discovered on this exact rectangle plan).

Each run is a separate subprocess (matches `run_all.py`'s isolation
rationale: `close_loop_path_follow.CONFIG` is module-level state that must
not leak between stages/conditions).

Predeclared outcome rule: a controller-triggered workspace/safety abort
counts as a genuine outcome for that schedule slot -- this script does NOT
distinguish that from an infrastructure failure automatically (it can't,
from a bare return code alone) and does NOT auto-rerun anything. On any
nonzero exit it stops and reports, so a human applies the predeclared rule
(reschedule that slot only for an infrastructure failure) and re-invokes
with --resume-from to continue the same schedule/seed.

Usage
-----
    python -m proper_research.hardware.online.rectangle_stage_a.run_delay_aware_ab \\
        --plan-dir plans/rectangle_10x15mm_bimaterial/time_parameterized_configuration_path \\
        --out-dir close_loop_logs/rectangle_delay_aware_ab_2026-09-18 \\
        --schedule-cache /tmp/rectangle_genuine_ltv_schedule.npy
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from proper_research.hardware.online.advancer_excitation.reset_insertion import reset_insertion

from . import common

CONDITION_SCRIPTS = {
    "A": "run_mpc_delay_baseline",
    "B": "run_mpc_delay_aware",
}


def build_schedule(n_blocks: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    order: list[str] = []
    for _ in range(n_blocks):
        pair = ["A", "B"]
        rng.shuffle(pair)
        order.extend(pair)
    return order


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--schedule-cache", required=True)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--seed", type=int, default=None, help="omit for a fresh random draw, printed either way")
    p.add_argument("--resume-from", type=int, default=0, help="0-indexed run to start at (for continuing after a stop)")
    p.add_argument("--advancer-port", default="/dev/ttyACM0")
    p.add_argument("--insertion-tolerance-mm", type=float, default=0.3)
    p.add_argument("--dry-run-insertion-reset", action="store_true",
                    help="use reset_insertion(live=False) -- prints intended moves, no advancer "
                         "motion. For rehearsing the schedule only, NOT for real A/B data.")
    args = p.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    schedule = build_schedule(args.n_blocks, seed)
    print(f"[delay_aware_ab] seed={seed} schedule={schedule} (resuming from index {args.resume_from})")

    log_path = Path(args.out_dir) / f"ab_schedule_seed{seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    pkg = "proper_research.hardware.online.rectangle_stage_a"

    for i in range(args.resume_from, len(schedule)):
        cond = schedule[i]
        print(f"\n{'=' * 88}\n[delay_aware_ab] run {i + 1}/{len(schedule)}: condition {cond} "
              f"({CONDITION_SCRIPTS[cond]})\n{'=' * 88}")

        # 2026-09-18: run 4/6 hit URConnectionError here mid-schedule -- the
        # previous run's subprocess had just exited and the UR controller's
        # RTDE interface needed a moment to release that connection before
        # accepting a new one. A short settle avoids re-hitting the race.
        time.sleep(3.0)
        print("[delay_aware_ab] preflight: robot-joint reset + camera/safety checks...")
        q0, l0 = common.preflight(args.plan_dir)

        print(f"[delay_aware_ab] resetting physical insertion to {l0 * 1000:.2f}mm "
              f"({'LIVE' if not args.dry_run_insertion_reset else 'dry-run'})...")
        final_l_mm = reset_insertion(
            l0 * 1000.0,
            live=not args.dry_run_insertion_reset,
            advancer_port=args.advancer_port,
            tolerance_mm=args.insertion_tolerance_mm,
        )
        print(f"[delay_aware_ab] insertion reset done: L={final_l_mm:.3f}mm (target {l0 * 1000:.2f}mm)")

        # 2026-09-18: the first live batch failed ALL 6/6 runs at stale_vision
        # after only 5-6 ticks -- reset_insertion() just closed its own
        # cam_index=0 CameraSource session, and the very next thing that
        # happened was the run subprocess opening a fresh CameraSource on
        # the same physical device. A device handle doesn't necessarily
        # release instantly; this settle gives it room to before the next
        # session grabs it. (See this package's README for a related,
        # already-fixed first-tick staleness bug after a robot-reader
        # reconnect -- this is a different symptom: staleness developing a
        # few ticks IN, only in this reset_insertion-adjacent code path.)
        settle_s = 8.0
        print(f"[delay_aware_ab] settling {settle_s:.0f}s for the camera device to fully "
              f"release before the run's own camera session opens...")
        time.sleep(settle_s)

        cmd = [
            py, "-m", f"{pkg}.{CONDITION_SCRIPTS[cond]}",
            "--plan-dir", args.plan_dir, "--out-dir", args.out_dir,
            "--schedule-cache", args.schedule_cache, "--skip-preflight",
        ]
        print(f"[delay_aware_ab] $ {' '.join(cmd)}")
        result = subprocess.run(cmd)

        row = {
            "run_index": i, "condition": cond, "seed": seed,
            "return_code": result.returncode,
            "insertion_target_mm": l0 * 1000.0, "insertion_final_mm": final_l_mm,
            "t_unix": time.time(),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        if result.returncode != 0:
            print(
                f"\n[delay_aware_ab] run {i + 1} (condition {cond}) exited with code "
                f"{result.returncode} -- STOPPING. Per the predeclared rule: if this was a "
                f"controller-triggered workspace/safety abort, it counts as a genuine outcome "
                f"for slot {i} (do not rerun it, just continue with "
                f"--resume-from {i + 1} --seed {seed}). If this was an infrastructure failure "
                f"(RTDE disconnect etc.) before meaningful control started, rerun this exact "
                f"slot with --resume-from {i} --seed {seed}.\n"
                f"[delay_aware_ab] schedule log so far: {log_path}"
            )
            return

    print(f"\n[delay_aware_ab] all {len(schedule)} runs completed. schedule log: {log_path}")


if __name__ == "__main__":
    main()
