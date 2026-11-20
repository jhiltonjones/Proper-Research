"""Study 2, tasks 1-2: confirm the real prediction-construction mechanism
from source, then compute (J_sched[k]-J0) @ (z_pred[k]-z_ref[k]) by horizon
stage k and planar axis, using real logged data from the 4 completed
condition-A hardware runs. No hardware, no source/log modification -- reads
existing close_loop_logs/ and plans/ files only.

Mechanism, confirmed from source (not assumed):
  - delay_aware_mpc.py's solve_delay_aware(): predicted_beam_vector =
    desired_position (nominal, from nominal_reference_positions_m, i.e.
    reference.desired_position_m -- IDENTICAL between A and B, never
    touched by --freeze-jacobian) + G @ predicted_inputs (the QP's own
    Jacobian-weighted term) + constant_error (a same-tick OUTPUT-BIAS
    correction built from _estimate_output_residual()'s filtered residual
    between measured and nominal-linearized beam position -- NOT logged
    per-tick as a standalone numeric field, since it depends on the
    controller's own running low-pass filter state, which the saved JSON
    rows do not expose). "jacobian_used" (logged) is ONLY stage-0's (3,n)
    Jacobian block, not the full horizon schedule window.
  - The horizon window of reference/schedule indices a tick at
    control_index=ref_index actually used is _reference_indices(ref_index,
    future=True) = clip(ref_index + 1 + arange(N), 0, S-1) (confirmed in
    simulate_time_parameterized_beam_output_mpc.py). This is how J_sched[k]
    for k=0..N-1 is reconstructed here: schedule[clip(ref_index+1+k, 0, S-1)].
  - J0 (B's frozen Jacobian) = schedule[0] for A's own schedule file (the
    same file B's run_mpc_delay_aware_insertion_anchor.py --freeze-jacobian
    logic froze at index 0 by default) -- confirmed directly against B's own
    saved frozen schedule file below, not assumed.
  - z_ref[k] = reference.state[clip(ref_index+1+k, 0, S-1)] (the schedule's
    own linearization point, 7-dim: 6 joints + insertion).
  - z_pred[k] = predicted_states[k] from the logged row (delayed physical
    state stack, (N,7), already saved per tick in predicted_beam_positions.jsonl).

This isolates the Jacobian-CHOICE term from the (identical-for-A-and-B)
nominal baseline and the (not independently recoverable from these logs)
output-bias filter state -- exactly the "separate... effects" the user's
brief asked for, with the bias-term limitation stated explicitly rather
than approximated.
"""
import json
import os

import numpy as np

REPO = "/home/jack/Proper-Research"
os.chdir(REPO)
import sys
sys.path.insert(0, REPO)

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
B_FROZEN_SCHEDULE_PATH = "close_loop_logs/condition_B_frozen_jacobian_2026-11-20_frozen_jacobian_schedule.npy"

A_RUNS = [
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T130429Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T131743Z",
    "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T132839Z",
]

N_HORIZON = 15


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    ref_state = np.asarray(reference.state, dtype=float)  # (S, 7)
    S = ref_state.shape[0]

    schedule = np.load(SCHEDULE_PATH)  # (S, 3, 7)
    assert schedule.shape[0] == S, f"schedule rows {schedule.shape[0]} != reference samples {S}"

    b_frozen = np.load(B_FROZEN_SCHEDULE_PATH)
    assert b_frozen.shape == schedule.shape
    # Confirm, not assume: B's frozen file really is schedule[0] tiled.
    assert np.allclose(b_frozen[0], schedule[0])
    for row in b_frozen:
        assert np.array_equal(row, b_frozen[0])
    J0 = schedule[0].copy()  # (3, 7) -- confirmed identical to B's actually-used frozen Jacobian

    per_stage_terms = {k: {"x": [], "y": [], "z": []} for k in range(N_HORIZON)}
    n_ticks_used = 0
    n_ticks_skipped_no_pred = 0

    for run_dir in A_RUNS:
        pred_rows = load_jsonl(f"{run_dir}/predicted_beam_positions.jsonl")
        for r in pred_rows:
            if not r.get("success") or r.get("predicted_states") is None:
                n_ticks_skipped_no_pred += 1
                continue
            ref_index = int(r["ref_index"])
            z_pred = np.asarray(r["predicted_states"], dtype=float)  # (N,7)
            if z_pred.shape[0] != N_HORIZON:
                continue
            idx_window = np.clip(ref_index + 1 + np.arange(N_HORIZON), 0, S - 1)
            z_ref_window = ref_state[idx_window]  # (N,7)
            J_sched_window = schedule[idx_window]  # (N,3,7)

            for k in range(N_HORIZON):
                dz = z_pred[k] - z_ref_window[k]  # (7,)
                dJ = J_sched_window[k] - J0  # (3,7)
                term = dJ @ dz  # (3,)
                per_stage_terms[k]["x"].append(term[0])
                per_stage_terms[k]["y"].append(term[1])
                per_stage_terms[k]["z"].append(term[2])
            n_ticks_used += 1

    # Also record ||J_sched[k]-J0|| by stage, unconditional on dz, as a
    # separate, more primitive sensitivity measure requested implicitly by
    # the user's own caution ("do not infer that increasing ||J[k]-J0||
    # alone proves causation") -- report both, kept distinct.
    dJ_norms_by_stage = {}
    for k in range(N_HORIZON):
        # evaluate at ref_index=0 start-of-path window as one fixed
        # reference point for a path-position-independent sensitivity figure
        idx_window0 = np.clip(0 + 1 + np.arange(N_HORIZON), 0, S - 1)
        dJ_norms_by_stage[k] = float(np.linalg.norm(schedule[idx_window0[k]] - J0))

    result = {
        "mechanism_confirmed_from_source": {
            "jacobian_used_field_is_stage0_only": True,
            "nominal_reference_positions_m_identical_A_and_B": True,
            "horizon_window_formula": "clip(ref_index + 1 + arange(N), 0, S-1)",
            "J0_equals_schedule_row0_confirmed_against_B_frozen_file": True,
            "output_bias_constant_error_term_not_independently_recoverable_from_logs": True,
        },
        "n_ticks_used": n_ticks_used,
        "n_ticks_skipped_no_prediction": n_ticks_skipped_no_pred,
        "term_mm_by_stage_and_axis": {
            str(k): {
                axis: {
                    "mean_abs_mm": float(np.mean(np.abs(vals)) * 1e3),
                    "rms_mm": float(np.sqrt(np.mean(np.square(vals))) * 1e3),
                    "max_abs_mm": float(np.max(np.abs(vals)) * 1e3),
                    "n": len(vals),
                }
                for axis, vals in per_stage_terms[k].items()
            }
            for k in range(N_HORIZON)
        },
        "dJ_norm_by_stage_from_path_start_window": dJ_norms_by_stage,
    }

    out_path = "proper_research/analysis/condition_A_E_campaign/study2_scheduled_vs_frozen/jacobian_contribution_term.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"n_ticks_used={n_ticks_used} n_ticks_skipped={n_ticks_skipped_no_pred}")
    for k in (0, 1, 2, 5, 10, 14):
        d = result["term_mm_by_stage_and_axis"][str(k)]
        print(f"stage {k:2d}: x_rms={d['x']['rms_mm']:.5f}mm y_rms={d['y']['rms_mm']:.5f}mm z_rms={d['z']['rms_mm']:.5f}mm (n={d['x']['n']})")


if __name__ == "__main__":
    main()
