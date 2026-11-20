"""Verify (don't assume) that B actually used the frozen schedule, E
actually applied adaptation mode='active' on every real tick, and C's
logs show the shared external-clearance check wired in despite having
no in-QP geometric constraint. Read-only.
"""
import json
import os

import numpy as np

LOGS = "/home/jack/Proper-Research/close_loop_logs"

B_RUNS = [
    "condition_B_frozen_jacobian_2026-11-20_20261120T124729Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T130657Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T131921Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T133013Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T133227Z",
]
E_RUNS = [
    "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T130244Z",
    "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T131539Z",
    "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T132522Z",
    "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T133739Z",
]
C_RUNS = [
    "condition_C_inv7_2026-11-20_20261120T125338Z",
    "condition_C_inv7_2026-11-20_20261120T130824Z",
    "condition_C_inv7_2026-11-20_20261120T132057Z",
    "condition_C_inv7_2026-11-20_20261120T133412Z",
]

print("=== B: frozen schedule verification ===")
for name in B_RUNS:
    d = os.path.join(LOGS, name)
    cm = json.load(open(os.path.join(d, "controller_metadata.json")))
    sched_path = cm["schedule_cache_path"]
    is_frozen_path = "frozen_jacobian_schedule.npy" in sched_path
    full_path = os.path.join("/home/jack/Proper-Research", sched_path)
    rows_identical = None
    if os.path.exists(full_path):
        arr = np.load(full_path)
        rows_identical = all(np.array_equal(arr[i], arr[0]) for i in range(arr.shape[0]))
    print(f"  {name}: schedule_cache_path={sched_path!r} "
          f"is_frozen_filename={is_frozen_path} all_rows_identical={rows_identical} "
          f"shape={arr.shape if os.path.exists(full_path) else 'FILE MISSING'}")

print()
print("=== E: adaptation mode verification (per-tick, real logs) ===")
for name in E_RUNS:
    d = os.path.join(LOGS, name)
    pred_path = os.path.join(d, "predicted_beam_positions.jsonl")
    rows = [json.loads(l) for l in open(pred_path)]
    modes = set()
    n_active = 0
    n_accepted_final = None
    n_rejected_final = None
    for r in rows:
        diag = r.get("jacobian_adaptation_diagnostics")
        if diag is None:
            modes.add(None)
            continue
        modes.add(diag.get("mode"))
        if diag.get("mode") == "active":
            n_active += 1
            n_accepted_final = diag.get("n_accepted")
            n_rejected_final = diag.get("n_rejected")
    print(f"  {name}: n_rows={len(rows)} modes_seen={modes} n_active_mode_rows={n_active} "
          f"final_n_accepted={n_accepted_final} final_n_rejected={n_rejected_final}")

print()
print("=== C: shared external-clearance check wiring verification ===")
for name in C_RUNS:
    d = os.path.join(LOGS, name)
    pf_path = os.path.join(d, "path_follow.jsonl")
    rows = [json.loads(l) for l in open(pf_path)]
    has_field = all("external_measured_clearance_gap_m" in r for r in rows)
    non_none = sum(1 for r in rows if r.get("external_measured_clearance_gap_m") is not None)
    vals = [r["external_measured_clearance_gap_m"] for r in rows if r.get("external_measured_clearance_gap_m") is not None]
    min_gap_mm = (min(vals) * 1e3) if vals else None
    print(f"  {name}: field_present_every_row={has_field} non_none={non_none}/{len(rows)} "
          f"min_measured_clearance_mm={min_gap_mm}")
    has_pred_file = os.path.exists(os.path.join(d, "predicted_beam_positions.jsonl"))
    print(f"    predicted_beam_positions.jsonl exists: {has_pred_file} "
          f"(expected False -- INV-7 has no predictive log; no in-QP geometric constraint exists for it)")
