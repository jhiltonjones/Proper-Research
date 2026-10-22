#!/usr/bin/env python3
"""Study 3 -- the R (input-tracking weight) sweep at gamma=0, recreating
CONTROLLER_JUSTIFICATION.md sec 6.3/7's log-scale sweep and refinement:
R x {1, 10, 100, 300, 500, 700, 1000} x R0, R0 = 1e-2 diag(1/s_u^2).

`input_tracking_weight` in ConfigurationMPCConfig IS this multiplier
directly (input_tracking_weight=7.0 means R700). Replays each condition
against a REAL logged closed-loop run and reports/plots E_chosen vs R and
the joint-channel active-rate-limit fraction vs R -- the original sweep's
finding was that R700 and R1000 give essentially identical joint-channel
activity, so R700 is not unnecessarily conservative relative to R1000.

Usage
-----
    python -m proper_research.studies.mpc_ablations.study03_r_sweep \\
        --log-dir close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z \\
        --r-multipliers 1,10,100,300,500,700,1000
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
SATURATION_TOL = 0.97
R0 = 1.0e-2  # base scale; input_tracking_weight = multiplier * R0 factor baked into ConfigurationMPCConfig's own convention


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=rc.DEFAULT_LOG_DIR)
    p.add_argument("--schedule-cache", default=rc.DEFAULT_SCHEDULE_CACHE)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--r-multipliers", default="1,10,100,300,500,700,1000")
    args = p.parse_args()
    multipliers = [float(x) for x in args.r_multipliers.split(",")]

    print(f"[r_sweep] real log: {args.log_dir}")
    rows, meta = rc.load_real_log(args.log_dir)
    plan_dir = meta.get("plan_dir", "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    dt = float(reference.sample_period_s)
    schedule = rc.build_schedule(plan_dir, args.schedule_cache)
    beam_config = rc.build_beam_config()

    controllers = {}
    velocity_limits = {}
    for mult in multipliers:
        mpc_config = rc.build_mpc_config(dt=dt, horizon=args.horizon, input_tracking_weight=mult * R0)
        controllers[mult] = StagewiseTaskNullspaceDelayAwareMPC(
            gamma=args.gamma, reference=reference, config=mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        )
        velocity_limits[mult] = np.asarray(mpc_config.velocity_limit, dtype=float)

    ticks = rc.pick_tick_indices(len(rows))
    print(f"[r_sweep] replaying {len(ticks)} ticks under R multiplier in {multipliers} (gamma={args.gamma})...")

    e_chosen = {mult: {h: [] for h in HORIZON_CHECKPOINTS} for mult in multipliers}
    joint_saturated = {mult: [] for mult in multipliers}
    insertion_saturated = {mult: [] for mult in multipliers}

    for i in ticks:
        inputs = rc.reconstruct_solve_inputs(rows, i)
        for mult in multipliers:
            step = controllers[mult].solve_delay_aware(**inputs)
            for h in HORIZON_CHECKPOINTS:
                stage = h - 1
                if stage < step.predicted_beam_errors.shape[0]:
                    e_chosen[mult][h].append(float(np.linalg.norm(step.predicted_beam_errors[stage])) * 1e3)
            cmd = np.asarray(step.command)
            vlim = velocity_limits[mult]
            joint_saturated[mult].append(float(np.mean(np.abs(cmd[:6]) >= SATURATION_TOL * vlim[:6])))
            insertion_saturated[mult].append(float(abs(cmd[6]) >= SATURATION_TOL * vlim[6]))

    print("\n[r_sweep] E_chosen (mm) and saturation, mean over replayed ticks:")
    for mult in multipliers:
        vals = "  ".join(f"+{h}={np.mean(e_chosen[mult][h]):.3f}" for h in HORIZON_CHECKPOINTS)
        print(f"  R={mult:>5.0f}xR0: {vals}   joint_sat={np.mean(joint_saturated[mult]):.3f}  "
              f"insertion_sat={np.mean(insertion_saturated[mult]):.3f}")

    out = rc.ensure_outdir("study03_r_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for h in HORIZON_CHECKPOINTS:
        axes[0].semilogx(multipliers, [np.mean(e_chosen[m][h]) for m in multipliers], "o-", label=f"E_chosen +{h}")
    axes[0].set_xlabel("R multiplier (x R0)"); axes[0].set_ylabel("E_chosen (mm)")
    axes[0].set_title(f"Task tracking vs R (gamma={args.gamma})"); axes[0].legend()

    axes[1].semilogx(multipliers, [np.mean(joint_saturated[m]) for m in multipliers], "o-", label="joint channels")
    axes[1].semilogx(multipliers, [np.mean(insertion_saturated[m]) for m in multipliers], "s-", label="insertion channel")
    axes[1].set_xlabel("R multiplier (x R0)"); axes[1].set_ylabel("fraction of ticks near rate limit")
    axes[1].set_title("Actuator saturation vs R, by channel"); axes[1].legend()
    fig.suptitle(f"R sweep at gamma={args.gamma}, replayed on real log")
    fig.tight_layout()
    fig.savefig(out / "r_sweep.png", dpi=150)
    print(f"\n[r_sweep] plot -> {out / 'r_sweep.png'}")


if __name__ == "__main__":
    main()
