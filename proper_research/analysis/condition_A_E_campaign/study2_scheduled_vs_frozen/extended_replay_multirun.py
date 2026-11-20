"""Study 2 bounded-check extension (2026-11-20, second pass, v2 -- fixes a
labeling bug found while running v1: A's own logged forecast is a
SCHEDULED-native forecast, but B's own logged forecast is a FROZEN-native
forecast (B's worker actually solved with J0 throughout) -- v1 treated
every run's logged "predicted_beam_positions_m" as if it were scheduled-
native, which silently mislabeled B's real data as a "scheduled" column.
Fixed here: every quantity is recomputed from primitives (z_meas, z_pred,
measured_beam_position_m, ref_index) independently for BOTH the scheduled
and the frozen predictor, on every run, with no run-conditional meaning
baked into a shared variable name. The run's own logged forecast is used
only as a SANITY CHECK (does the from-scratch reconstruction match the
logged value, for whichever Jacobian that run's own worker actually used)
-- not as an input baked asymmetrically into one labeled column.

Implements, from source (confirmed, not assumed), the exact replay
equations; extends the prior single-run fixed-bias replay to all 4
completed A runs, all 4 completed B runs, and B's one aborted run (with an
explicit, stated scoring window). No hardware, no source/log modification.

EXACT EQUATIONS, confirmed by reading delay_aware_mpc.py and
worker_process.py directly:

  Registration (applied once at startup, worker_process.py's
  set_frame_transform -- NOT per-tick):
      des_R = des_planner @ R_fit.T + t_fit
      controller.reference.desired_position_m <- des_R
      controller.nominal_reference_positions_m <- des_R.copy()
  After registration these two are the SAME array. This campaign's logs
  show R_fit ~= I, t_fit ~= 0mm (near-identity), so the numeric
  contribution here is negligible, but it is a real, named pipeline step,
  reported as such rather than skipped.

  Per-tick residual (delay_aware_mpc.py's _estimate_output_residual,
  confirmed in simulate_time_parameterized_beam_output_mpc.py):
      index = clip(control_index, 0, S-1)              # == ref_index here
      local_nominal(J) = nominal_ref[index] + J[index] @ (z_meas - ref_state[index])
      instantaneous(J) = measured_beam_position - local_nominal(J)
      filtered_t = alpha*filtered_{t-1} + (1-alpha)*instantaneous_t   (t>0)
      filtered_0 = instantaneous_0
  disturbance_filter_alpha's confirmed default AND this campaign's actual
  (unoverridden, per controller_metadata.json) value is 0.0 -> filtered_t
  == instantaneous_t every tick, no time-dependent state to propagate.

  Per-horizon-stage prediction (delay_aware_mpc.py's solve_delay_aware /
  _dynamic_qp_terms_exec, confirmed):
      idx_window[k] = clip(ref_index + 1 + k, 0, S-1),  k=0..N-1
      predicted_beam_position(J)[k] = nominal_ref[idx_window[k]]
          + J[idx_window[k]] @ (z_pred[k] - ref_state[idx_window[k]])
          + beta_d * residual_t(J)       (SAME value for every k -- source
                                           tiles one residual across the
                                           whole horizon, not per-stage)
  where J is schedule[.] for the scheduled predictor or J0=schedule[0] for
  the frozen predictor, EVERYWHERE J appears above, including inside the
  residual -- this is what makes "separate causal bias" (b) different
  from "fixed-bias substitution" (a) below.

  (a) FIXED-BIAS substitution: take the run's OWN actually-used residual
      (embedded in ITS OWN logged forecast -- scheduled-native for A,
      frozen-native for B) and reuse it UNCHANGED for the OTHER Jacobian's
      horizon term:
          other_forecast_a[k] = own_logged_forecast[k]
                                 - (J_own[idx_window[k]] - J_other[idx_window[k]])
                                   @ (z_pred[k]-ref_state[idx_window[k]])
      Does not recompute the residual with J_other -- isolates the
      horizon-term alone, not a faithful stand-alone predictor.

  (b) SEPARATE CAUSAL BIAS: recompute the residual with J_other at the
      current tick too, then build J_other's forecast end-to-end with its
      own bias:
          residual_other_t = measured_beam_position
              - (nominal_ref[ref_index] + J_other[ref_index] @ (z_meas-ref_state[ref_index]))
          other_forecast_b[k] = nominal_ref[idx_window[k]]
              + J_other[idx_window[k]] @ (z_pred[k]-ref_state[idx_window[k]])
              + beta_d * residual_other_t
      A faithful stand-alone predictor for the SAME already-executed
      history (z_pred/z_meas/measured_beam_position all real/recorded;
      only the predictor's internal Jacobian choice swaps, consistently,
      at both the bias step and the horizon-prediction step).

Both (a) and (b) are PREDICTION REPLAY on an already-recorded history --
NOT closed-loop evidence, and do not capture how a real controller using
the other Jacobian would have ACTED differently (only condition B's own
real closed-loop runs show that for the frozen case).
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
MAX_GAP_S = 0.5 * DT_S  # same convention as plot_pilot_run.py's score_forecasts() and the prior Study 2 pass
BETA_D = 1.0

A_RUNS = [
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T130429Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T131743Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T132839Z",
]
B_RUNS_COMPLETED = [
    "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_20261120T124729Z",
    "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_20261120T130657Z",
    "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_20261120T131921Z",
    "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_20261120T133227Z",
]
B_RUN_ABORTED = "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_20261120T133013Z"
B_ABORT_LAST_VALID_STEP = 116  # stop_reason tracking_error_exceeded at tick 117/235 -> last fully executed tick is 116


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def forecast_horizon(J_full, nominal_ref, ref_state, idx_window, z_pred, residual_t):
    """predicted_beam_position(J)[k] for k=0..N-1, per the confirmed formula."""
    out = np.zeros((N_HORIZON, 3))
    for k in range(N_HORIZON):
        dz = z_pred[k] - ref_state[idx_window[k]]
        out[k] = nominal_ref[idx_window[k]] + J_full[idx_window[k]] @ dz + BETA_D * residual_t
    return out


def residual_at_tick(J_full, nominal_ref, ref_state, ref_index, z_meas, meas_beam):
    local_nominal = nominal_ref[ref_index] + J_full[ref_index] @ (z_meas - ref_state[ref_index])
    return meas_beam - local_nominal


def replay_one_run(run_dir, is_frozen_native, ref_state, nominal_ref, schedule, J0, S, max_step=None):
    """is_frozen_native: True for a B run (its own worker solved with J0
    throughout, so its OWN logged forecast is frozen-native); False for an
    A run (scheduled-native)."""
    pred_rows = load_jsonl(f"{run_dir}/predicted_beam_positions.jsonl")
    pf_rows = load_jsonl(f"{run_dir}/path_follow.jsonl")
    if max_step is not None:
        pred_rows = [r for r in pred_rows if r["step"] <= max_step]
        pf_rows = [r for r in pf_rows if r["step"] <= max_step]
    pf_by_step = {r["step"]: r for r in pf_rows}
    t_by_step = {r["step"]: r["t_s"] for r in pf_rows}
    tip_by_step = {
        r["step"]: np.asarray(pf_by_step[r["step"]]["tip_mm"], dtype=float) / 1e3
        for r in pf_rows if pf_by_step[r["step"]].get("tip_mm") is not None
    }

    out = {"scheduled": [], "frozen_a": [], "frozen_b": []}
    sanity_max_abs_diff_mm = 0.0
    n_sanity_checked = 0
    n_excluded_no_pred = 0
    n_excluded_future = 0
    n_scored_ticks = 0

    for r in pred_rows:
        if not r.get("success") or r.get("predicted_states") is None or r.get("predicted_beam_positions_m") is None:
            n_excluded_no_pred += 1
            continue
        k_step = r["step"]
        t_k = t_by_step.get(k_step)
        if t_k is None:
            n_excluded_no_pred += 1
            continue
        ref_index = int(np.clip(r["ref_index"], 0, S - 1))
        z_meas = np.asarray(r["z_meas"], dtype=float)
        meas_beam = np.asarray(r.get("measured_beam_position_m"), dtype=float) if r.get("measured_beam_position_m") is not None else None
        z_pred = np.asarray(r["predicted_states"], dtype=float)
        own_logged_forecast = np.asarray(r["predicted_beam_positions_m"], dtype=float)
        if z_pred.shape[0] != N_HORIZON or own_logged_forecast.shape[0] != N_HORIZON or meas_beam is None:
            n_excluded_no_pred += 1
            continue

        idx_window = np.clip(ref_index + 1 + np.arange(N_HORIZON), 0, S - 1)

        # Recompute BOTH predictors from scratch, independent of which one
        # this run's own worker actually used.
        residual_sched = residual_at_tick(schedule, nominal_ref, ref_state, ref_index, z_meas, meas_beam)
        residual_frozen = residual_at_tick(
            np.broadcast_to(J0, schedule.shape), nominal_ref, ref_state, ref_index, z_meas, meas_beam
        )
        forecast_sched = forecast_horizon(schedule, nominal_ref, ref_state, idx_window, z_pred, residual_sched)
        forecast_frozen_b = forecast_horizon(
            np.broadcast_to(J0, schedule.shape), nominal_ref, ref_state, idx_window, z_pred, residual_frozen
        )

        # Sanity check: the from-scratch reconstruction of THIS RUN's own
        # native predictor must match its own logged forecast closely.
        own_reconstruction = forecast_frozen_b if is_frozen_native else forecast_sched
        diff_mm = float(np.max(np.abs(own_reconstruction - own_logged_forecast)) * 1e3)
        sanity_max_abs_diff_mm = max(sanity_max_abs_diff_mm, diff_mm)
        n_sanity_checked += 1

        # Fixed-bias substitution (a): reuse THIS RUN's own native residual
        # (as embedded in its own logged forecast) for the OTHER Jacobian's
        # horizon term only.
        if is_frozen_native:
            # own native = frozen; "other" fixed-bias forecast = scheduled,
            # built by ADDING the (J_sched-J0)@dz delta onto the logged
            # (frozen-native) forecast.
            other_fixed_bias = own_logged_forecast + (forecast_sched - forecast_frozen_b)
            sched_col, frozen_a_col, frozen_b_col = other_fixed_bias, own_logged_forecast, forecast_frozen_b
        else:
            # own native = scheduled; "other" fixed-bias forecast = frozen,
            # built by SUBTRACTING the (J_sched-J0)@dz delta from the
            # logged (scheduled-native) forecast -- this is the prior
            # pass's original method, unchanged for A runs.
            other_fixed_bias = own_logged_forecast - (forecast_sched - forecast_frozen_b)
            sched_col, frozen_a_col, frozen_b_col = own_logged_forecast, other_fixed_bias, forecast_frozen_b

        target_ts = [t_k + j * DT_S for j in range(N_HORIZON)]
        any_scored_this_tick = False
        for j in range(N_HORIZON):
            candidates = [(abs(t_by_step[s] - target_ts[j]), s) for s in t_by_step if t_by_step[s] >= t_k]
            if not candidates:
                n_excluded_future += 1
                continue
            gap, best_s = min(candidates)
            if gap > MAX_GAP_S or best_s not in tip_by_step:
                n_excluded_future += 1
                continue
            meas = tip_by_step[best_s]
            if np.any(np.isnan(meas)):
                n_excluded_future += 1
                continue

            out["scheduled"].append((j, float(np.linalg.norm(sched_col[j] - meas))))
            out["frozen_a"].append((j, float(np.linalg.norm(frozen_a_col[j] - meas))))
            out["frozen_b"].append((j, float(np.linalg.norm(frozen_b_col[j] - meas))))
            any_scored_this_tick = True
        if any_scored_this_tick:
            n_scored_ticks += 1

    return out, {
        "n_scored_ticks": n_scored_ticks,
        "n_excluded_no_prediction": n_excluded_no_pred,
        "n_excluded_future_measurement_unavailable_or_gap": n_excluded_future,
        "n_pred_rows_total": len(pred_rows),
        "own_native_reconstruction_sanity_max_abs_diff_mm": sanity_max_abs_diff_mm,
        "n_sanity_checked": n_sanity_checked,
    }


def summarize_by_lead(pairs):
    by_lead = {}
    for j, e in pairs:
        by_lead.setdefault(j, []).append(e)
    return by_lead


def main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    ref_state = np.asarray(reference.state, dtype=float)
    S = ref_state.shape[0]
    nominal_ref = np.asarray(reference.desired_position_m, dtype=float)  # post-registration in the real run
    schedule = np.load(SCHEDULE_PATH)
    J0 = schedule[0].copy()

    all_runs = (
        [("A", r, False, None) for r in A_RUNS]
        + [("B_completed", r, True, None) for r in B_RUNS_COMPLETED]
        + [("B_aborted", B_RUN_ABORTED, True, B_ABORT_LAST_VALID_STEP)]
    )

    per_run_results = {}
    for cond_label, run_dir, is_frozen_native, max_step in all_runs:
        out, bookkeeping = replay_one_run(run_dir, is_frozen_native, ref_state, nominal_ref, schedule, J0, S, max_step=max_step)
        by_lead_sched = summarize_by_lead(out["scheduled"])
        by_lead_a = summarize_by_lead(out["frozen_a"])
        by_lead_b = summarize_by_lead(out["frozen_b"])
        leads = sorted(by_lead_sched)
        per_run_results[run_dir] = {
            "condition_label": cond_label,
            "own_native_predictor": "frozen" if is_frozen_native else "scheduled",
            "scoring_window": ("ticks 0-116 only (abort at tick 117); ticks beyond 116 excluded, "
                                "not scored, not treated as zero" if max_step is not None else "full run"),
            "bookkeeping": bookkeeping,
            "by_lead_stage_mm": {
                str(j): {
                    "scheduled_mean_mm": float(np.mean(by_lead_sched[j]) * 1e3),
                    "frozen_a_mean_mm": float(np.mean(by_lead_a[j]) * 1e3),
                    "frozen_b_mean_mm": float(np.mean(by_lead_b[j]) * 1e3),
                    "diff_a_minus_sched_mm": float((np.mean(by_lead_a[j]) - np.mean(by_lead_sched[j])) * 1e3),
                    "diff_b_minus_sched_mm": float((np.mean(by_lead_b[j]) - np.mean(by_lead_sched[j])) * 1e3),
                    "diff_b_minus_a_mm": float((np.mean(by_lead_b[j]) - np.mean(by_lead_a[j])) * 1e3),
                    "n": len(by_lead_sched[j]),
                }
                for j in leads
            },
        }

    j_rep = 14
    a_minus_sched_at_14 = []
    b_minus_sched_at_14 = []
    for run_dir, res in per_run_results.items():
        d = res["by_lead_stage_mm"].get(str(j_rep))
        if d is not None:
            a_minus_sched_at_14.append(d["diff_a_minus_sched_mm"])
            b_minus_sched_at_14.append(d["diff_b_minus_sched_mm"])

    cross_run_spread = {
        "lead_stage": j_rep,
        "note": ("SPREAD, ACROSS THE 9 AVAILABLE RUNS, of THIS REPLAY COMPUTATION'S "
                 "OWN frozen-minus-scheduled difference at lead stage 14 -- genuine "
                 "repeated-measurement variability of the replay result itself. This "
                 "is explicitly NOT the main campaign's across-run tracking-RMSE "
                 "standard deviation (a different question: closed-loop tracking "
                 "variability across repeated hardware runs) and must not be called "
                 "a measurement-noise floor -- it reflects run-to-run differences in "
                 "the recorded execution history (initial state, registration, camera "
                 "noise in z_meas/measured_beam_position, timing jitter) feeding an "
                 "otherwise-identical replay computation."),
        "frozen_a_minus_scheduled": {
            "n_runs": len(a_minus_sched_at_14), "values_mm": a_minus_sched_at_14,
            "mean_mm": float(np.mean(a_minus_sched_at_14)),
            "sd_mm": float(np.std(a_minus_sched_at_14, ddof=1)) if len(a_minus_sched_at_14) > 1 else None,
        },
        "frozen_b_minus_scheduled": {
            "n_runs": len(b_minus_sched_at_14), "values_mm": b_minus_sched_at_14,
            "mean_mm": float(np.mean(b_minus_sched_at_14)),
            "sd_mm": float(np.std(b_minus_sched_at_14, ddof=1)) if len(b_minus_sched_at_14) > 1 else None,
        },
    }

    result = {
        "label": "Extended multi-run prediction replay (fixed-bias 'a' and separate-causal-bias 'b'), NOT closed-loop evidence. v2: fixed a labeling bug where B's own frozen-native logged forecast was previously mislabeled as scheduled.",
        "disturbance_filter_alpha_confirmed": 0.0,
        "alpha_zero_implication": "filtered residual == instantaneous residual every tick; no time-dependent filter state to propagate",
        "per_run_results": per_run_results,
        "cross_run_spread_at_lead_14": cross_run_spread,
    }
    out_path = "proper_research/analysis/condition_A_E_campaign/study2_scheduled_vs_frozen/extended_replay_multirun.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
    for run_dir, res in per_run_results.items():
        d14 = res["by_lead_stage_mm"].get("14")
        bk = res["bookkeeping"]
        print(f"{res['condition_label']:14s} native={res['own_native_predictor']:10s} {run_dir.split('/')[-1][:40]:40s} "
              f"n_scored={bk['n_scored_ticks']:4d} sanity_maxdiff={bk['own_native_reconstruction_sanity_max_abs_diff_mm']:.6f}mm "
              f"diff_a@14={d14['diff_a_minus_sched_mm']:+.4f}mm diff_b@14={d14['diff_b_minus_sched_mm']:+.4f}mm")
    print("\ncross-run spread at lead 14 (frozen_a - scheduled):", cross_run_spread["frozen_a_minus_scheduled"])
    print("cross-run spread at lead 14 (frozen_b - scheduled):", cross_run_spread["frozen_b_minus_scheduled"])


if __name__ == "__main__":
    main()
