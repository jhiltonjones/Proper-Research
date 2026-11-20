"""Section 2+3: per-run metrics table for every INCLUDED run (from
build_manifest_and_metrics.py's manifest.json). Read-only.

Verified facts baked into this script (checked directly against real
log rows before writing this, not assumed):
  - tip_mm/desired_mm's z-component is CONSTANT (-39.627mm) and
    IDENTICAL between tip and desired on every single row checked --
    the vision pipeline for this planar rectangle plan does not
    independently resolve Z at all, so error_mm's z-component is
    always exactly 0.0 and error_norm_mm is, in practice, a pure
    planar (xy) quantity for this campaign. Reported as such below,
    per the instruction to separate camera-derived Z from planar error.
  - "deadline_miss" = whole-tick loop-period miss (read-state+solve+
    compute vs the ~100ms tick budget); "solver_deadline_missed" =
    the WORKER's own sub-deadline miss (resp is None within its
    70ms budget). Confirmed from close_loop_path_follow.py's own
    inline comments at the fields' construction site, not inferred
    from field names.
  - reference_samples for this plan = 214 (the schedule's own row
    count, confirmed from every run's saved frozen_jacobian.json /
    this session's own worker-spawn logs this entire day).
  - C (inv_7dof_delay_aware) has NO predicted_beam_positions.jsonl at
    all -- it is a reactive controller with no horizon forecast to
    log. Every "predicted"/"QP"-derived metric is reported as None
    for C, never inferred or substituted.
"""
import csv
import json
import os
import statistics

import numpy as np

ROOT = "/home/jack/Proper-Research"
LOGS = os.path.join(ROOT, "close_loop_logs")
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

REFERENCE_SAMPLES = 214
MAGNET_EXCLUSION_RADIUS_MM = 233.0  # 230 (floor) + 3 (robust margin), the in-QP boundary for A/B/D/E
MEASURED_CLEARANCE_ABORT_MM = 230.0  # shared external check, all 5 conditions


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _longest_true_streak(bool_list):
    best = cur = 0
    for b in bool_list:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def _pctl(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_metrics(condition, run_dir):
    d = os.path.join(LOGS, run_dir)
    pf = _load_jsonl(os.path.join(d, "path_follow.jsonl"))
    pred = _load_jsonl(os.path.join(d, "predicted_beam_positions.jsonl"))  # None for C
    summary = json.load(open(os.path.join(d, "summary.json")))

    stop_reason = summary.get("stop_reason")
    control_steps = summary.get("control_steps")
    completed = stop_reason == "path_complete"

    # progress -- use the LAST row's ref_index/progress_index as the
    # furthest point reached, over the known reference_samples denominator
    last_ref_index = pf[-1].get("ref_index") if pf else None
    completed_path_fraction = (
        None if last_ref_index is None else min(1.0, (last_ref_index + 1) / REFERENCE_SAMPLES)
    )

    active_rows = [r for r in pf if not r.get("terminal_hold")]
    hold_rows = [r for r in pf if r.get("terminal_hold")]

    active_err = [r["error_norm_mm"] for r in active_rows if r.get("error_norm_mm") is not None]
    hold_err = [r["error_norm_mm"] for r in hold_rows if r.get("error_norm_mm") is not None]

    active_err_z = [r["error_mm"][2] for r in active_rows if r.get("error_mm") is not None]
    z_error_always_zero = all(abs(v) < 1e-9 for v in active_err_z) if active_err_z else None

    def _rmse(vals):
        return None if not vals else float(np.sqrt(np.mean(np.square(vals))))

    final_error_mm = summary.get("final_error_mm")
    max_error_mm_summary = summary.get("max_error_mm")

    # timing
    t_all = [r["t_s"] for r in pf]
    t_active = [r["t_s"] for r in active_rows]
    run_duration_s = (max(t_all) - min(t_all)) if len(t_all) > 1 else None
    active_motion_duration_s = (max(t_active) - min(t_active)) if len(t_active) > 1 else None
    dt_samples = np.diff(sorted(t_all)) if len(t_all) > 1 else np.array([])
    dt_median = float(np.median(dt_samples)) if len(dt_samples) else None

    # deadline misses -- keep the two kinds distinct
    solver_miss = [bool(r.get("solver_deadline_missed")) for r in pf]
    whole_tick_miss = [bool(r.get("deadline_miss")) for r in pf]
    n_solver_miss = sum(solver_miss)
    n_whole_tick_miss = sum(whole_tick_miss)
    longest_solver_miss_streak = _longest_true_streak(solver_miss)
    longest_whole_tick_miss_streak = _longest_true_streak(whole_tick_miss)

    # measured clearance (shared field across all 5 conditions)
    clearance_vals_mm = [
        r["external_measured_clearance_gap_m"] * 1e3
        for r in pf if r.get("external_measured_clearance_gap_m") is not None
    ]
    min_measured_clearance_mm = min(clearance_vals_mm) if clearance_vals_mm else None
    clearance_missing_rows = len(pf) - len(clearance_vals_mm)

    # insertion command extrema
    ins_vals_mm = [r["insertion_length_m"] * 1e3 for r in pf if r.get("insertion_length_m") is not None]
    ins_min_mm = min(ins_vals_mm) if ins_vals_mm else None
    ins_max_mm = max(ins_vals_mm) if ins_vals_mm else None
    # "limiting events": ticks within 0.5mm of either bound (20/70mm envelope)
    n_at_lower_bound = sum(1 for v in ins_vals_mm if v <= 20.5)
    n_at_upper_bound = sum(1 for v in ins_vals_mm if v >= 69.5)

    # command smoothness: RMS of q_target_rad sample-to-sample delta,
    # normalized by the ACTUAL sample interval (not assumed 0.1s)
    q_targets = [r.get("q_target_rad") for r in pf if r.get("q_target_rad") is not None]
    smoothness_rms_rad_s = None
    if len(q_targets) > 1 and dt_median:
        diffs = np.diff(np.asarray(q_targets, dtype=float), axis=0)
        rates = diffs / dt_median
        smoothness_rms_rad_s = float(np.sqrt(np.mean(np.square(rates))))

    # external interventions / aborts
    abort_reasons = sorted(set(r.get("adapter_abort_reason") for r in pf if r.get("adapter_abort_reason")))

    # logging completeness
    pf_n = len(pf)
    pred_n = len(pred) if pred is not None else None
    logging_note = ""
    if pred is None:
        logging_note = "no predicted_beam_positions.jsonl (expected for inv_7dof_delay_aware -- no horizon forecast exists to log)"
    elif pred_n != pf_n:
        logging_note = f"pf_rows({pf_n}) != pred_rows({pred_n}) -- step-ID join is NOT exact-set-equal for this run, verify before joining"
    else:
        pf_steps = set(r["step"] for r in pf)
        pred_steps = set(r["step"] for r in pred)
        logging_note = "exact-set-equal step IDs" if pf_steps == pred_steps else "same row COUNT but DIFFERENT step ID sets -- do not assume alignment"

    # QP-side metrics (A/B/D/E only; None for C)
    min_qp_measured_gap_mm = None
    n_qp_success = None
    if pred is not None:
        gaps = [r["magnet_exclusion_gap_mm"] for r in pred if r.get("magnet_exclusion_gap_mm") is not None]
        min_qp_measured_gap_mm = min(gaps) if gaps else None
        n_qp_success = sum(1 for r in pred if r.get("success"))

    return {
        "condition": condition,
        "run_dir": run_dir,
        "stop_reason": stop_reason,
        "completed": completed,
        "control_steps": control_steps,
        "completed_path_fraction": completed_path_fraction,
        "active_motion_n_ticks": len(active_rows),
        "terminal_hold_n_ticks": len(hold_rows),
        "active_tracking_rmse_mm": _rmse(active_err),
        "active_tracking_mean_mm": (statistics.mean(active_err) if active_err else None),
        "active_tracking_p95_mm": _pctl(active_err, 95),
        "active_tracking_max_mm": (max(active_err) if active_err else None),
        "z_error_always_zero_in_active_motion": z_error_always_zero,
        "terminal_hold_mean_mm": (statistics.mean(hold_err) if hold_err else None),
        "terminal_hold_max_mm": (max(hold_err) if hold_err else None),
        "final_error_mm": final_error_mm,
        "max_error_mm_whole_run": max_error_mm_summary,
        "run_duration_s": run_duration_s,
        "active_motion_duration_s": active_motion_duration_s,
        "dt_median_s": dt_median,
        "n_solver_deadline_miss": n_solver_miss,
        "n_whole_tick_deadline_miss": n_whole_tick_miss,
        "longest_solver_miss_streak": longest_solver_miss_streak,
        "longest_whole_tick_miss_streak": longest_whole_tick_miss_streak,
        "min_measured_clearance_mm": min_measured_clearance_mm,
        "clearance_field_missing_rows": clearance_missing_rows,
        "insertion_min_mm": ins_min_mm,
        "insertion_max_mm": ins_max_mm,
        "n_ticks_at_lower_bound_20mm": n_at_lower_bound,
        "n_ticks_at_upper_bound_70mm": n_at_upper_bound,
        "command_smoothness_rms_rad_s": smoothness_rms_rad_s,
        "abort_reasons_seen": ";".join(abort_reasons) if abort_reasons else None,
        "pf_row_count": pf_n,
        "pred_row_count": pred_n,
        "logging_completeness_note": logging_note,
        "min_qp_measured_gap_mm": min_qp_measured_gap_mm,
        "n_qp_solve_success": n_qp_success,
    }


def main():
    manifest = json.load(open(os.path.join(OUT_DIR, "manifest.json")))
    rows = []
    for m in manifest:
        if not m["include"]:
            continue
        rows.append(compute_metrics(m["condition"], m["run_dir"]))

    fieldnames = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "per_run_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(OUT_DIR, "per_run_metrics.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"Wrote {len(rows)} run rows to per_run_metrics.csv/.json")
    for r in rows:
        print(f"  {r['condition']} {r['run_dir']}: completed={r['completed']} "
              f"active_rmse={r['active_tracking_rmse_mm']:.3f}mm "
              f"final_err={r['final_error_mm']}")
    return rows


if __name__ == "__main__":
    main()
