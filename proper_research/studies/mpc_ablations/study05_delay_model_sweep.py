#!/usr/bin/env python3
"""Study 5 -- delay model d (joint command-to-effect delay, in control
ticks), recreating (a SIMPLIFIED, single-log version of)
CONTROLLER_JUSTIFICATION.md sec 2's finding that d=2 beats d=0 substantially
on prediction accuracy. NOTE: the original evidence was a proper 5-fold
leave-one-out cross-validation across several logs (G=0.973-0.992); this
script replays d in {0,1,2} on ONE real log and compares E_chosen -- useful
for seeing the qualitative effect and for extending to your own multi-log
LOO study, not a reproduction of the original statistical claim.

`delay_samples` is a constructor argument of DelayAwareBeamOutputTrackingMPC
(and everything that subclasses it); d=0 collapses the delay-compensation
machinery to a regression-test-only case (see delay_aware_mpc.py's
docstring) -- undelayed joints, matching the "old" BeamOutputTrackingMPC
baseline in effect.

Usage
-----
    python -m proper_research.studies.mpc_ablations.study05_delay_model_sweep \\
        --log-dir close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z \\
        --delay-samples 0,1,2
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proper_research.controllers.mpc_delay_aware.stagewise_task_nullspace import (
    StagewiseTaskNullspaceDelayAwareMPC,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

from . import replay_common as rc

HORIZON_CHECKPOINTS = (3, 5, 10)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=rc.DEFAULT_LOG_DIR)
    p.add_argument("--schedule-cache", default=rc.DEFAULT_SCHEDULE_CACHE)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--input-tracking-weight", type=float, default=7.0)
    p.add_argument("--delay-samples", default="0,1,2")
    args = p.parse_args()
    delays = [int(x) for x in args.delay_samples.split(",")]

    print(f"[delay_sweep] real log: {args.log_dir}")
    rows, meta = rc.load_real_log(args.log_dir)
    plan_dir = meta.get("plan_dir", "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    dt = float(reference.sample_period_s)
    schedule = rc.build_schedule(plan_dir, args.schedule_cache)
    mpc_config = rc.build_mpc_config(dt=dt, horizon=args.horizon, input_tracking_weight=args.input_tracking_weight)
    beam_config = rc.build_beam_config()

    controllers = {
        d: StagewiseTaskNullspaceDelayAwareMPC(
            gamma=args.gamma, reference=reference, config=mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=d, beta_d=1.0,
        )
        for d in delays
    }

    ticks = rc.pick_tick_indices(len(rows))
    print(f"[delay_sweep] replaying {len(ticks)} ticks under delay_samples in {delays}...")

    e_chosen = {d: {h: [] for h in HORIZON_CHECKPOINTS} for d in delays}
    for i in ticks:
        inputs = rc.reconstruct_solve_inputs(rows, i)
        for d in delays:
            step = controllers[d].solve_delay_aware(**inputs)
            for h in HORIZON_CHECKPOINTS:
                stage = h - 1
                if stage < step.predicted_beam_errors.shape[0]:
                    e_chosen[d][h].append(float(np.linalg.norm(step.predicted_beam_errors[stage])) * 1e3)

    print("\n[delay_sweep] E_chosen (mm), mean over replayed ticks:")
    for d in delays:
        vals = "  ".join(f"+{h}={np.mean(e_chosen[d][h]):.3f}" for h in HORIZON_CHECKPOINTS)
        print(f"  d={d}: {vals}")

    out = rc.ensure_outdir("study05_delay_model_sweep")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for h in HORIZON_CHECKPOINTS:
        ax.plot(delays, [np.mean(e_chosen[d][h]) for d in delays], "o-", label=f"E_chosen +{h}")
    ax.set_xlabel("delay_samples (d)"); ax.set_ylabel("E_chosen (mm)")
    ax.set_xticks(delays)
    ax.set_title("Task tracking vs joint command-to-effect delay model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "delay_model_sweep.png", dpi=150)
    print(f"\n[delay_sweep] plot -> {out / 'delay_model_sweep.png'}")


if __name__ == "__main__":
    main()
