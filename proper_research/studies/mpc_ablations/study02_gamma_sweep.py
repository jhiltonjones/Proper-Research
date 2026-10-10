#!/usr/bin/env python3
"""Study 2 -- the gamma sweep: from isotropic state cost (gamma=1, "M0") to
exact task-nullspace-only (gamma=0, the frozen live formulation), recreating
CONTROLLER_JUSTIFICATION.md sec 6.2/6.3's sweep.

    l_Q,j = z~_j^T (P_N,j + gamma * P_R,j) z~_j,   z~ = S_z^-1 (z - z_ref)

gamma=1 reproduces the isotropic Q exactly (P_N+P_R=I); gamma=0 removes the
row-space (task-producing-direction) weight entirely, so a configuration
offset the task cost has already justified is no longer fought by Q.
StagewiseTaskNullspaceDelayAwareMPC already takes `gamma` as a constructor
argument, so this sweep is just that ONE class instantiated at several
gamma values -- no separate controller class needed.

Replays each gamma condition against a REAL logged closed-loop run
(single-tick counterfactual, see replay_common.py's docstring) and reports/
plots E_chosen (task tracking) vs actuator saturation (fraction of ticks
where a joint or the insertion channel sits at its rate limit) across gamma
-- the Pareto tradeoff the original sweep was run to see.

Usage
-----
    python -m proper_research.studies.mpc_ablations.study02_gamma_sweep \\
        --log-dir close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z \\
        --gammas 0.0,0.10,0.25,1.0
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
SATURATION_TOL = 0.97  # fraction of the rate limit counted as "saturated"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=rc.DEFAULT_LOG_DIR)
    p.add_argument("--schedule-cache", default=rc.DEFAULT_SCHEDULE_CACHE)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--input-tracking-weight", type=float, default=7.0)
    p.add_argument("--gammas", default="0.0,0.10,0.25,1.0")
    args = p.parse_args()
    gammas = [float(g) for g in args.gammas.split(",")]

    print(f"[gamma_sweep] real log: {args.log_dir}")
    rows, meta = rc.load_real_log(args.log_dir)
    plan_dir = meta.get("plan_dir", "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path")
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    dt = float(reference.sample_period_s)
    schedule = rc.build_schedule(plan_dir, args.schedule_cache)
    mpc_config = rc.build_mpc_config(dt=dt, horizon=args.horizon, input_tracking_weight=args.input_tracking_weight)
    beam_config = rc.build_beam_config()
    velocity_limit = np.asarray(mpc_config.velocity_limit, dtype=float)

    controllers = {
        g: StagewiseTaskNullspaceDelayAwareMPC(
            gamma=g, reference=reference, config=mpc_config, beam_config=beam_config,
            reference_position_jacobians=schedule, delay_samples=2, beta_d=1.0,
        )
        for g in gammas
    }

    ticks = rc.pick_tick_indices(len(rows))
    print(f"[gamma_sweep] replaying {len(ticks)} ticks under gamma in {gammas}...")

    e_chosen = {g: {h: [] for h in HORIZON_CHECKPOINTS} for g in gammas}
    saturated_frac = {g: [] for g in gammas}

    for i in ticks:
        inputs = rc.reconstruct_solve_inputs(rows, i)
        for g in gammas:
            step = controllers[g].solve_delay_aware(**inputs)
            for h in HORIZON_CHECKPOINTS:
                stage = h - 1
                if stage < step.predicted_beam_errors.shape[0]:
                    e_chosen[g][h].append(float(np.linalg.norm(step.predicted_beam_errors[stage])) * 1e3)
            near_limit = np.abs(step.command) >= SATURATION_TOL * velocity_limit
            saturated_frac[g].append(float(np.mean(near_limit)))

    print("\n[gamma_sweep] E_chosen (mm), mean over replayed ticks:")
    for g in gammas:
        vals = "  ".join(f"+{h}={np.mean(e_chosen[g][h]):.3f}" for h in HORIZON_CHECKPOINTS)
        print(f"  gamma={g:<5}: {vals}   saturated_frac={np.mean(saturated_frac[g]):.3f}")

    out = rc.ensure_outdir("study02_gamma_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for h in HORIZON_CHECKPOINTS:
        axes[0].plot(gammas, [np.mean(e_chosen[g][h]) for g in gammas], "o-", label=f"E_chosen +{h}")
    axes[0].set_xlabel("gamma"); axes[0].set_ylabel("E_chosen (mm)")
    axes[0].set_title("Task tracking vs gamma"); axes[0].legend()

    axes[1].plot(gammas, [np.mean(saturated_frac[g]) for g in gammas], "o-", color="tab:red")
    axes[1].set_xlabel("gamma"); axes[1].set_ylabel("fraction of ticks near a rate limit")
    axes[1].set_title("Actuator saturation vs gamma")
    fig.suptitle(f"Gamma sweep (state-cost row-space weight), replayed on real log")
    fig.tight_layout()
    fig.savefig(out / "gamma_sweep.png", dpi=150)
    print(f"\n[gamma_sweep] plot -> {out / 'gamma_sweep.png'}")


if __name__ == "__main__":
    main()
