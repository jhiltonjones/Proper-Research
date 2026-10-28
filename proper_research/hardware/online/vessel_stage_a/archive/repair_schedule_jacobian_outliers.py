"""Patch isolated numerical-artifact samples in a precomputed per-s MPC
Jacobian schedule (the cached .npy that `build_or_load_schedule` in
run_mpc_delay_aware_vessel.py loads verbatim on every run -- see
plans/stage3_design/closedloop_hw_analysis/README.md's schedule-
conditioning section for how these were found).

This is the same failure mode already called out in
run_mpc_delay_aware_vessel.py's own comment on `jacobian_mode="accurate"`:
a single-tick spurious singular-value spike (true tip output changes
smoothly there) -- that comment is about the "fast" estimator; this
schedule was already built with "accurate" and still has at least one
such sample (k=505, s=72.10mm, condition number 665 vs ~4.1 at both
immediate neighbours, sigma_min collapsing 150x on that one knot only).
A real kinematic trough shows up gradually over several neighbouring
samples; an isolated single-knot collapse does not, so it is treated as
an artifact of the per-sample Jacobian estimator (e.g. the continuation
contact-solve landing on a different contact-node state at that one knot)
rather than real physics, and is replaced -- NOT by just damping/zeroing
rows, but by linearly interpolating the entire 3x7 Jacobian (all 21
entries, in path-progress s) between the nearest *clean* neighbouring
samples on either side. That keeps every genuine, gradual feature of the
schedule intact -- including the real, sustained 2-4x conditioning gap
that starts right at the contact onset -- and only overwrites the
isolated bad knot(s).

Detection: robust, not a hand-picked index list, so this also works if
the schedule is rebuilt and the artifact lands on a different sample.
For each k, compare its condition number against the MEDIAN condition
number of a local window (+/-7 samples, excluding a +/-1 guard band
around k so the point itself and an adjacent partner can't pull the
local baseline toward it). Flag k if cond[k] exceeds both an absolute
floor and a multiple of that local median -- two independent conditions
so an already-elevated-but-legitimate region (e.g. the post-contact-onset
plateau at cond~7-10) isn't swept up by the ratio test alone. Defaults
(floor=50, ratio=5x) are deliberately conservative: they catch the two
unambiguous single-knot collapses (s=0mm, s=72.10mm) and deliberately
leave the milder s=45-50mm bump (cond 7->9->17, a <2.3x local deviation)
untouched, since that one sits inside an already-elevated plateau and
looks more like genuine recurring contact-transition behaviour than a
pure numerical artifact -- see --ratio/--floor to repair it too.

Usage:
  python repair_schedule_jacobian_outliers.py \
      --schedule plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06.npy \
      --plan-dir plans/vessel_phi30_L30_left1mm_newwall_tol0p5_2026-10-06 \
      --out plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def condition_numbers(J_sched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = J_sched.shape[0]
    cond = np.zeros(n)
    sigma_min = np.zeros(n)
    for k in range(n):
        sv = np.linalg.svd(J_sched[k], compute_uv=False)
        sigma_min[k] = sv[-1]
        cond[k] = sv[0] / max(sv[-1], 1e-12)
    return cond, sigma_min


def detect_outliers(cond: np.ndarray, window: int = 7, guard: int = 1,
                     floor: float = 50.0, ratio: float = 5.0) -> np.ndarray:
    n = len(cond)
    flagged = np.zeros(n, dtype=bool)
    log_cond = np.log(cond)
    for k in range(n):
        lo, hi = max(0, k - window), min(n, k + window + 1)
        idx = np.array([i for i in range(lo, hi) if abs(i - k) > guard])
        if len(idx) < 3:
            continue
        local_median_log = np.median(log_cond[idx])
        local_median = np.exp(local_median_log)
        if cond[k] > floor and cond[k] > ratio * local_median:
            flagged[k] = True
    return flagged


def flagged_runs(flagged: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, end] (inclusive) index runs of flagged samples."""
    runs = []
    k = 0
    n = len(flagged)
    while k < n:
        if flagged[k]:
            j = k
            while j + 1 < n and flagged[j + 1]:
                j += 1
            runs.append((k, j))
            k = j + 1
        else:
            k += 1
    return runs


def repair(J_sched: np.ndarray, s_mm: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    repaired = J_sched.copy()
    n = len(flagged)
    for start, end in flagged_runs(flagged):
        left = start - 1
        right = end + 1
        if left < 0 or right >= n:
            # Can't interpolate an edge run (e.g. the very first/last
            # sample) between two clean neighbours; fall back to the
            # single nearest clean sample (flat extrapolation) rather
            # than guessing.
            src = right if left < 0 else left
            for k in range(start, end + 1):
                repaired[k] = J_sched[src]
            continue
        s_left, s_right = s_mm[left], s_mm[right]
        J_left, J_right = J_sched[left], J_sched[right]
        for k in range(start, end + 1):
            t = (s_mm[k] - s_left) / max(s_right - s_left, 1e-12)
            repaired[k] = (1.0 - t) * J_left + t * J_right
    return repaired


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--schedule", required=True)
    p.add_argument("--plan-dir", required=True,
                   help="plan directory whose time_parameterized_configuration_path.npz "
                        "gives this schedule's own path_s_m (for s-based interpolation "
                        "and the diagnostic plot's x-axis).")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=7)
    p.add_argument("--guard", type=int, default=1)
    p.add_argument("--floor", type=float, default=50.0)
    p.add_argument("--ratio", type=float, default=5.0)
    p.add_argument("--fig-out", default=None,
                   help="optional before/after condition-number comparison plot path")
    args = p.parse_args()

    J_sched = np.load(args.schedule)
    npz = np.load(f"{args.plan_dir}/time_parameterized_configuration_path/time_parameterized_configuration_path.npz")
    s_mm = npz["path_s_m"] * 1e3
    assert len(s_mm) == J_sched.shape[0], (len(s_mm), J_sched.shape)

    cond_before, smin_before = condition_numbers(J_sched)
    flagged = detect_outliers(cond_before, window=args.window, guard=args.guard,
                               floor=args.floor, ratio=args.ratio)
    runs = flagged_runs(flagged)

    print(f"schedule: {args.schedule}  ({J_sched.shape[0]} samples)")
    print(f"detector: window=+/-{args.window} guard=+/-{args.guard} floor={args.floor} ratio={args.ratio}x")
    if not runs:
        print("no outliers flagged -- nothing to repair.")
        return
    print(f"flagged {flagged.sum()} sample(s) in {len(runs)} run(s):")
    for start, end in runs:
        for k in range(start, end + 1):
            print(f"  k={k:4d} s={s_mm[k]:7.3f}mm cond={cond_before[k]:10.2f} sigma_min={smin_before[k]:.6f}")

    repaired = repair(J_sched, s_mm, flagged)
    cond_after, smin_after = condition_numbers(repaired)

    print("\nafter repair, at the same indices:")
    for start, end in runs:
        for k in range(start, end + 1):
            print(f"  k={k:4d} s={s_mm[k]:7.3f}mm cond={cond_after[k]:10.2f} sigma_min={smin_after[k]:.6f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, repaired)
    print(f"\nsaved repaired schedule -> {args.out}")

    if args.fig_out:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(s_mm, cond_before, color="#b3331d", lw=1.1, label="before repair")
        ax.plot(s_mm, cond_after, color="#1b7f3b", lw=1.1, label="after repair")
        for start, end in runs:
            ax.axvspan(s_mm[start] - 0.05, s_mm[end] + 0.05, color="gray", alpha=0.25)
        ax.set_yscale("log")
        ax.set_xlabel("path progress s (mm)")
        ax.set_ylabel("condition number (sigma_max/sigma_min)")
        ax.set_title(f"Schedule repair: {Path(args.schedule).name}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(args.fig_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_out, dpi=160)
        plt.close(fig)
        print(f"saved comparison figure -> {args.fig_out}")


if __name__ == "__main__":
    main()
