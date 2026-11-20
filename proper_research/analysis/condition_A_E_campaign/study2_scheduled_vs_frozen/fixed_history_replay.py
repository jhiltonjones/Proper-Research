"""Study 2, task 3-4: fixed-history prediction replay (not closed-loop
evidence) and comparison against the empirical run-to-run noise floor.

Method: take ONE completed condition-A run's actual recorded execution
history. At each tick k, A's own saved predicted_states[k] (the delayed
physical state stack actually computed at that tick, using only
information available up to that tick -- no future leakage) and
predicted_beam_positions_m[k] (A's actual logged forecast, built with the
scheduled Jacobian window) are read directly from the log. The FROZEN
predictor's forecast for the identical history is NOT a new QP solve (that
would require re-running the real worker, which is a mechanistic ablation
better suited to Study 1's with/without pattern, and was not required for
this task) -- it is a LINEAR REPLAY using the exact decomposition
confirmed in decompose_jacobian_contribution.py:

    predicted_beam_position[k] = nominal_ref(idx_k) + J[k] @ (z_pred[k]-z_ref[k]) + bias[k]

Since nominal_ref and bias are identical for scheduled and frozen (see that
script's source-confirmed mechanism), the frozen forecast for the SAME
z_pred[k] (same executed command history, no re-optimization) is:

    frozen_forecast[k] = scheduled_forecast[k] - (J_sched[k]-J0) @ (z_pred[k]-z_ref[k])

This is a linear substitution on an already-recorded history, explicitly
labeled PREDICTION REPLAY, not closed-loop evidence -- it does not capture
how the frozen controller would have ACTED differently (which could only
be observed from a real frozen-Jacobian closed-loop run, i.e. condition B
itself), only how its FORECAST would have differed for the SAME already-
executed inputs.

Both forecasts are then scored against the SAME run's actual future
measured tip (tip_mm) at matched lead times, using the same nearest-later-
tick-within-gap matching rule the main campcampaign's forecast comparison
already used (no change to that matching logic, reused here for a fair,
identical method on identical data).
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
REPLAY_RUN = "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z"
N_HORIZON = 15
DT_S = 0.1  # confirmed control rate this session (10Hz) -- matches t_s spacing


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    ref_state = np.asarray(reference.state, dtype=float)
    S = ref_state.shape[0]
    schedule = np.load(SCHEDULE_PATH)
    J0 = schedule[0].copy()

    pred_rows = load_jsonl(f"{REPLAY_RUN}/predicted_beam_positions.jsonl")
    pf_rows = load_jsonl(f"{REPLAY_RUN}/path_follow.jsonl")
    pf_by_step = {r["step"]: r for r in pf_rows}
    t_by_step = {r["step"]: r["t_s"] for r in pf_rows}
    tip_by_step = {
        r["step"]: np.asarray(pf_by_step[r["step"]]["tip_mm"], dtype=float) / 1e3
        for r in pf_rows if pf_by_step[r["step"]].get("tip_mm") is not None
    }

    results_scheduled = []  # (lead_j, abs_err_m)
    results_frozen = []
    max_gap_s = 0.5 * DT_S
    n_scored = 0

    for r in pred_rows:
        if not r.get("success") or r.get("predicted_states") is None or r.get("predicted_beam_positions_m") is None:
            continue
        k_step = r["step"]
        t_k = t_by_step.get(k_step)
        if t_k is None:
            continue
        ref_index = int(r["ref_index"])
        z_pred = np.asarray(r["predicted_states"], dtype=float)
        scheduled_forecast = np.asarray(r["predicted_beam_positions_m"], dtype=float)  # (N,3), A's real logged forecast
        if z_pred.shape[0] != N_HORIZON or scheduled_forecast.shape[0] != N_HORIZON:
            continue
        idx_window = np.clip(ref_index + 1 + np.arange(N_HORIZON), 0, S - 1)
        z_ref_window = ref_state[idx_window]
        J_sched_window = schedule[idx_window]

        for j in range(N_HORIZON):
            dz = z_pred[j] - z_ref_window[j]
            dJ = J_sched_window[j] - J0
            term = dJ @ dz  # (3,)
            frozen_forecast_j = scheduled_forecast[j] - term

            target_t = t_k + j * DT_S
            candidates = [(abs(t_by_step[s] - target_t), s) for s in t_by_step if t_by_step[s] >= t_k]
            if not candidates:
                continue
            gap, best_s = min(candidates)
            if gap > max_gap_s or best_s not in tip_by_step:
                continue
            meas = tip_by_step[best_s]
            if np.any(np.isnan(meas)):
                continue
            results_scheduled.append((j, float(np.linalg.norm(scheduled_forecast[j] - meas))))
            results_frozen.append((j, float(np.linalg.norm(frozen_forecast_j - meas))))
        n_scored += 1

    by_lead_sched = {}
    by_lead_frozen = {}
    for j, e in results_scheduled:
        by_lead_sched.setdefault(j, []).append(e)
    for j, e in results_frozen:
        by_lead_frozen.setdefault(j, []).append(e)

    leads = sorted(by_lead_sched)
    out = {
        "replay_run": REPLAY_RUN,
        "label": "PREDICTION REPLAY on one fixed recorded A execution history -- NOT closed-loop evidence; frozen forecast is a linear substitution (scheduled_forecast - (J_sched-J0)@(z_pred-z_ref)), not a re-solved QP, so it cannot capture how a real frozen-Jacobian controller would have acted differently",
        "n_ticks_scored": n_scored,
        "by_lead_stage_mm": {
            str(j): {
                "scheduled_mean_mm": float(np.mean(by_lead_sched[j]) * 1e3),
                "frozen_mean_mm": float(np.mean(by_lead_frozen[j]) * 1e3),
                "diff_frozen_minus_scheduled_mm": float((np.mean(by_lead_frozen[j]) - np.mean(by_lead_sched[j])) * 1e3),
                "n": len(by_lead_sched[j]),
            }
            for j in leads
        },
    }
    out_path = "proper_research/analysis/condition_A_E_campaign/study2_scheduled_vs_frozen/fixed_history_replay.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"n_ticks_scored={n_scored}")
    for j in leads:
        d = out["by_lead_stage_mm"][str(j)]
        print(f"lead {j:2d}: scheduled={d['scheduled_mean_mm']:.4f}mm frozen={d['frozen_mean_mm']:.4f}mm diff(frozen-sched)={d['diff_frozen_minus_scheduled_mm']:+.4f}mm n={d['n']}")


if __name__ == "__main__":
    main()
