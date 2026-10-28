#!/usr/bin/env python3
"""Study 4 -- disturbance-persistence sensitivity: beta_d, the constant that
scales how much of the estimated output residual (disturbance) is carried
across the WHOLE prediction horizon (see delay_aware_mpc.py's docstring;
CONTROLLER_JUSTIFICATION.md sec 3). beta_d=1 (full persistence) is frozen;
this recreates the sensitivity check that beta_d=0.82 shifts commands and
predictions non-trivially (so the choice isn't inert) without claiming
superiority either way (that would need a closed-loop live comparison, not
this open-loop replay).

Replays several beta_d values against a REAL logged closed-loop run and
reports/plots, RELATIVE TO beta_d=1.0:
  - command RMS shift (%), on the chosen u0
  - horizon-averaged predicted-beam-position RMS shift (mm)

Usage
-----
    python -m proper_research.studies.mpc_ablations.study04_disturbance_beta_sweep \\
        --log-dir close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z \\
        --beta-ds 0.5,0.82,1.0
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=rc.DEFAULT_LOG_DIR)
    p.add_argument("--schedule-cache", default=rc.DEFAULT_SCHEDULE_CACHE)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--input-tracking-weight", type=float, default=7.0)
    p.add_argument("--beta-ds", default="0.5,0.82,1.0")
    args = p.parse_args()
    beta_ds = [float(x) for x in args.beta_ds.split(",")]
    if 1.0 not in beta_ds:
        beta_ds.append(1.0)

    print(f"[beta_sweep] real log: {args.log_dir}")
    rows, meta = rc.load_real_log(args.log_dir)
    plan_dir = meta.get("plan_dir", "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    dt = float(reference.sample_period_s)
    schedule = rc.build_schedule(plan_dir, args.schedule_cache)
    mpc_config = rc.build_mpc_config(dt=dt, horizon=args.horizon, input_tracking_weight=args.input_tracking_weight)
    beam_config = rc.build_beam_config()

    controllers = {
        b: StagewiseTaskNullspaceDelayAwareMPC(
            gamma=args.gamma, reference=reference, config=mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=2, beta_d=b,
        )
        for b in beta_ds
    }

    ticks = rc.pick_tick_indices(len(rows))
    print(f"[beta_sweep] replaying {len(ticks)} ticks under beta_d in {beta_ds}...")

    commands = {b: [] for b in beta_ds}
    predicted_beams = {b: [] for b in beta_ds}

    for i in ticks:
        inputs = rc.reconstruct_solve_inputs(rows, i)
        for b in beta_ds:
            step = controllers[b].solve_delay_aware(**inputs)
            commands[b].append(np.asarray(step.command, dtype=float).copy())
            predicted_beams[b].append(np.asarray(step.predicted_beam_positions, dtype=float).copy())

    baseline_cmd = np.asarray(commands[1.0])
    baseline_pred = np.asarray(predicted_beams[1.0])

    print("\n[beta_sweep] shift RELATIVE TO beta_d=1.0, mean over replayed ticks:")
    cmd_shift_pct = {}
    pred_shift_mm = {}
    for b in beta_ds:
        cmd_b = np.asarray(commands[b])
        pred_b = np.asarray(predicted_beams[b])
        cmd_rms_shift = float(np.sqrt(np.mean((cmd_b - baseline_cmd) ** 2)))
        cmd_rms_base = float(np.sqrt(np.mean(baseline_cmd ** 2))) + 1e-12
        cmd_shift_pct[b] = 100.0 * cmd_rms_shift / cmd_rms_base
        pred_shift_mm[b] = float(np.sqrt(np.mean((pred_b - baseline_pred) ** 2))) * 1e3
        print(f"  beta_d={b:<5}: command RMS shift={cmd_shift_pct[b]:.1f}%   "
              f"predicted-beam RMS shift={pred_shift_mm[b]:.3f}mm")

    out = rc.ensure_outdir("study04_disturbance_beta_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(beta_ds, [cmd_shift_pct[b] for b in beta_ds], "o-")
    axes[0].set_xlabel("beta_d"); axes[0].set_ylabel("command RMS shift vs beta_d=1.0 (%)")
    axes[0].set_title("Command sensitivity to disturbance persistence")

    axes[1].plot(beta_ds, [pred_shift_mm[b] for b in beta_ds], "o-", color="tab:orange")
    axes[1].set_xlabel("beta_d"); axes[1].set_ylabel("predicted-beam RMS shift vs beta_d=1.0 (mm)")
    axes[1].set_title("Prediction sensitivity to disturbance persistence")
    fig.suptitle("Disturbance persistence (beta_d) sensitivity, replayed on real log")
    fig.tight_layout()
    fig.savefig(out / "beta_d_sweep.png", dpi=150)
    print(f"\n[beta_sweep] plot -> {out / 'beta_d_sweep.png'}")


if __name__ == "__main__":
    main()
