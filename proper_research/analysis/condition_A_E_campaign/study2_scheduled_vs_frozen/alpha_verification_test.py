"""Algebraic + empirical verification of the alpha=0 bias-equivalence
claim made in STUDY2_PROPOSAL.md's section 5b (extended_replay_multirun.py).

ALGEBRAIC ARGUMENT (full derivation in JACOBIAN_STUDY_HARDWARE_PROPOSAL.md
section 1a; summarized here as the basis for this script):

  filtered_t(J) = alpha*filtered_{t-1}(J) + (1-alpha)*instantaneous_t(J)
  instantaneous_t(J) = measured_beam_position_t
      - nominal_ref[idx_t] - J[idx_t] @ (z_meas_t - ref_state[idx_t])

  Two predictors (scheduled, using J=schedule[.]; frozen, using J=J0)
  differ in instantaneous_t only through
      (J_sched[idx_t] - J0) @ (z_meas_t - ref_state[idx_t]),
  i.e. the Jacobian-difference term evaluated at the CURRENT tracking
  error (z_meas_t - ref_state[idx_t]), which this campaign's own tracking
  performance keeps small (sub-degree joint deviations; active-motion RMSE
  0.5-0.7mm Cartesian per condition_summary.csv) -- this per-tick term is
  therefore itself small, REGARDLESS of alpha.

  What alpha controls is not the SIZE of each tick's contribution but
  whether it ACCUMULATES: with alpha=0, filtered_t=instantaneous_t has no
  memory, so each tick's (necessarily small) Jacobian-difference
  contribution is discarded immediately -- the two predictors' bias
  states can never drift apart by more than one tick's worth of this
  small quantity. With alpha>0, filtered_t is an exponential moving
  average with time constant 1/(1-alpha); even a small PERSISTENT
  per-tick difference between instantaneous_sched_t and
  instantaneous_frozen_t integrates over many ticks, so the two
  predictors' filtered bias states can diverge by much more than any
  single tick's contribution as the run progresses. This predicts:
  (a) vs (b) divergence should GROW with alpha, not stay pinned near 0,
  and should shrink back toward 0 as alpha -> 0.

EMPIRICAL TEST: this script implements a PROPERLY STATEFUL, causal
recursive filter (unlike extended_replay_multirun.py's (b), which -- note,
confirmed by reading that script directly -- never actually carries a
filtered_{t-1} term forward in time at all; it recomputes an
"instantaneous-only" residual fresh every tick, which is only a correct
implementation of "(b) separate causal bias" because alpha=0 makes
filtered_t==instantaneous_t exactly, with no memory to omit). Here, TWO
independent recursive filters (one per predictor) are run in real tick
order for a chosen alpha, confirming the prior pass's near-exact (a)~=(b)
agreement is a real consequence of alpha=0 and not an implementation
artifact that would hold regardless of alpha.

No hardware; reads only already-saved logs from one real run. Standalone
script -- does NOT modify delay_aware_mpc.py / worker_process.py; the
filter update below is a copy of the real equation (confirmed against
simulate_time_parameterized_beam_output_mpc.py's _estimate_output_residual),
parametrized by an injected alpha for this test only.
"""
import json
import os
import sys

import numpy as np

REPO = "/home/jack/Proper-Research"
os.chdir(REPO)
sys.path.insert(0, REPO)

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
N_HORIZON = 15
DT_S = 0.1
BETA_D = 1.0
RUN_DIR = "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z"  # same run as the original single-run replay
ALPHAS_TESTED = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    ref_state = np.asarray(reference.state, dtype=float)
    S = ref_state.shape[0]
    nominal_ref = np.asarray(reference.desired_position_m, dtype=float)
    schedule = np.load(SCHEDULE_PATH)
    J0 = schedule[0].copy()
    J0_full = np.broadcast_to(J0, schedule.shape)

    pred_rows = load_jsonl(f"{RUN_DIR}/predicted_beam_positions.jsonl")
    pf_rows = load_jsonl(f"{RUN_DIR}/path_follow.jsonl")
    pf_by_step = {r["step"]: r for r in pf_rows}
    t_by_step = {r["step"]: r["t_s"] for r in pf_rows}
    tip_by_step = {
        r["step"]: np.asarray(pf_by_step[r["step"]]["tip_mm"], dtype=float) / 1e3
        for r in pf_rows if pf_by_step[r["step"]].get("tip_mm") is not None
    }

    # Ticks in time order, with the primitives needed; pred_rows are
    # already step-ordered in the log (confirmed elsewhere this session).
    ticks = []
    for r in pred_rows:
        if not r.get("success") or r.get("predicted_states") is None or r.get("measured_beam_position_m") is None:
            continue
        k_step = r["step"]
        t_k = t_by_step.get(k_step)
        if t_k is None:
            continue
        ticks.append(r)

    results = {}
    for alpha in ALPHAS_TESTED:
        filtered_sched_prev = None
        filtered_frozen_prev = None
        diffs_mm = []  # per-tick |forecast_b_other - forecast_a_other| at lead stage 14, mm
        for r in ticks:
            ref_index = int(np.clip(r["ref_index"], 0, S - 1))
            z_meas = np.asarray(r["z_meas"], dtype=float)
            meas_beam = np.asarray(r["measured_beam_position_m"], dtype=float)
            z_pred = np.asarray(r["predicted_states"], dtype=float)
            if z_pred.shape[0] != N_HORIZON:
                continue
            idx_window = np.clip(ref_index + 1 + np.arange(N_HORIZON), 0, S - 1)

            # Instantaneous residual, each predictor's own Jacobian at the
            # CURRENT ref_index (the real _estimate_output_residual formula).
            local_nom_sched = nominal_ref[ref_index] + schedule[ref_index] @ (z_meas - ref_state[ref_index])
            local_nom_frozen = nominal_ref[ref_index] + J0 @ (z_meas - ref_state[ref_index])
            inst_sched = meas_beam - local_nom_sched
            inst_frozen = meas_beam - local_nom_frozen

            # Stateful recursive filter, properly carried tick-to-tick --
            # the real equation, parametrized by the injected alpha.
            if filtered_sched_prev is None:
                filtered_sched = inst_sched.copy()
                filtered_frozen = inst_frozen.copy()
            else:
                filtered_sched = alpha * filtered_sched_prev + (1.0 - alpha) * inst_sched
                filtered_frozen = alpha * filtered_frozen_prev + (1.0 - alpha) * inst_frozen
            filtered_sched_prev = filtered_sched
            filtered_frozen_prev = filtered_frozen

            # (b) separate causal bias: each predictor's full horizon
            # forecast built with its OWN filtered bias.
            k = N_HORIZON - 1  # lead stage 14, matches the headline comparison elsewhere in this study
            dz = z_pred[k] - ref_state[idx_window[k]]
            forecast_sched_b = nominal_ref[idx_window[k]] + schedule[idx_window[k]] @ dz + BETA_D * filtered_sched
            forecast_frozen_b = nominal_ref[idx_window[k]] + J0 @ dz + BETA_D * filtered_frozen

            # (a) fixed-bias substitution: take the "own" (scheduled, since
            # this is an A run) predictor's filtered bias and reuse it
            # UNCHANGED to score the OTHER (frozen) Jacobian's horizon term.
            forecast_frozen_a = forecast_sched_b - (schedule[idx_window[k]] - J0) @ dz

            diff_mm = float(np.linalg.norm(forecast_frozen_b - forecast_frozen_a) * 1e3)
            diffs_mm.append(diff_mm)

        results[str(alpha)] = {
            "alpha": alpha,
            "n_ticks": len(diffs_mm),
            "mean_abs_diff_a_vs_b_mm_at_lead14": float(np.mean(diffs_mm)),
            "max_abs_diff_a_vs_b_mm_at_lead14": float(np.max(diffs_mm)),
            "p95_abs_diff_a_vs_b_mm_at_lead14": float(np.percentile(diffs_mm, 95)),
        }

    out = {
        "run": RUN_DIR,
        "description": (
            "Divergence between (a) fixed-bias substitution and (b) separate "
            "causal bias at lead stage 14, as a function of an ARTIFICIALLY "
            "INJECTED filter alpha (not the real campaign's own alpha=0 "
            "default) -- a properly stateful recursive filter is run "
            "separately for the scheduled and frozen predictor at each "
            "tested alpha. If divergence grows with alpha and shrinks "
            "toward 0 as alpha->0, this confirms the prior pass's exact "
            "(a)~=(b) agreement under the real alpha=0 is a genuine "
            "consequence of zero filter memory, not an implementation "
            "artifact that would hold regardless of alpha."
        ),
        "by_alpha": results,
    }

    with open("proper_research/analysis/condition_A_E_campaign/study2_scheduled_vs_frozen/alpha_verification_test_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Run: {RUN_DIR}")
    print(f"{'alpha':>6} {'mean_diff_mm':>14} {'max_diff_mm':>14} {'p95_diff_mm':>14}")
    for alpha in ALPHAS_TESTED:
        res = results[str(alpha)]
        print(f"{alpha:6.1f} {res['mean_abs_diff_a_vs_b_mm_at_lead14']:14.6f} "
              f"{res['max_abs_diff_a_vs_b_mm_at_lead14']:14.6f} {res['p95_abs_diff_a_vs_b_mm_at_lead14']:14.6f}")


if __name__ == "__main__":
    main()
