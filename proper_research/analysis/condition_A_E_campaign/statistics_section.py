"""Section 7: statistical interpretation. n=4 runs per condition is the
independent unit (NOT ticks). Real order structure discovered directly
from each run's own summary.json timestamp: 4 repeated blocks, each
block running A->B->C->D->E in strict sequence within roughly one hour
(block timestamps ~12:29, ~13:05-13:18, ~13:18-13:29, ~13:29-13:38 UTC
on 2026-11-20). Condition is therefore CONFOUNDED with within-block
position (A always first, E always last) -- reported as a limitation,
not something this analysis can correct for.

B's block-4 slot is the completed RETRY (133227Z) for paired purposes;
the aborted attempt (133013Z) is the same block's PRECEDING, non-paired
extra attempt and is reported separately, never silently dropped.

Pairing is used ONLY because this genuine block structure exists --
not because runs were sorted/numbered after the fact.
"""
import json
import os
import statistics

import numpy as np

ROOT = "/home/jack/Proper-Research"
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

# block -> {condition: run_dir}, built from verified timestamps
BLOCKS = [
    {
        "A": "baseline_A_pilot_2026-11-20_retry1_20261120T122825Z",
        "B": "condition_B_frozen_jacobian_2026-11-20_20261120T124729Z",
        "C": "condition_C_inv7_2026-11-20_20261120T125338Z",
        "D": "condition_D_mpc_N3_2026-11-20_20261120T125551Z",
        "E": "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T130244Z",
    },
    {
        "A": "baseline_A_pilot_2026-11-20_retry1_20261120T130429Z",
        "B": "condition_B_frozen_jacobian_2026-11-20_20261120T130657Z",
        "C": "condition_C_inv7_2026-11-20_20261120T130824Z",
        "D": "condition_D_mpc_N3_2026-11-20_20261120T131021Z",
        "E": "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T131539Z",
    },
    {
        "A": "baseline_A_pilot_2026-11-20_retry1_20261120T131743Z",
        "B": "condition_B_frozen_jacobian_2026-11-20_20261120T131921Z",
        "C": "condition_C_inv7_2026-11-20_20261120T132057Z",
        "D": "condition_D_mpc_N3_2026-11-20_20261120T132340Z",
        "E": "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T132522Z",
    },
    {
        "A": "baseline_A_pilot_2026-11-20_retry1_20261120T132839Z",
        "B": "condition_B_frozen_jacobian_2026-11-20_20261120T133227Z",  # retry, not the abort
        "C": "condition_C_inv7_2026-11-20_20261120T133412Z",
        "D": "condition_D_mpc_N3_2026-11-20_20261120T133608Z",
        "E": "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T133739Z",
    },
]
B_EXTRA_ABORT = "condition_B_frozen_jacobian_2026-11-20_20261120T133013Z"  # block 4, preceded the retry


def _by_dir(rows):
    return {r["run_dir"]: r for r in rows}


def paired_diff(by_dir, cond_a, cond_b, field):
    diffs = []
    for block in BLOCKS:
        va = by_dir[block[cond_a]][field]
        vb = by_dir[block[cond_b]][field]
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    return diffs


def cohen_d_paired(diffs):
    if len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    return (statistics.mean(diffs) / sd) if sd > 0 else None


def main():
    rows = json.load(open(os.path.join(OUT_DIR, "per_run_metrics.json")))
    by_dir = _by_dir(rows)

    report = {}
    for (ca, cb, label) in [
        ("A", "C", "A_minus_C (H1: MPC vs reactive)"),
        ("A", "B", "A_minus_B (H2: scheduled vs frozen, 4 paired completions)"),
        ("A", "D", "A_minus_D (N=15 vs N=3)"),
        ("A", "E", "A_minus_E (adaptation off vs on)"),
    ]:
        entry = {}
        for field in ("active_tracking_rmse_mm", "final_error_mm", "completed_path_fraction"):
            diffs = paired_diff(by_dir, ca, cb, field)
            entry[field] = {
                "n_pairs": len(diffs),
                "paired_diffs": diffs,
                "mean_diff": statistics.mean(diffs) if diffs else None,
                "sd_diff": statistics.stdev(diffs) if len(diffs) > 1 else None,
                "cohen_d_paired": cohen_d_paired(diffs),
            }
        report[label] = entry

    with open(os.path.join(OUT_DIR, "paired_statistics.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("=== Paired (block-matched) differences, n=4 pairs each ===")
    print("NOTE: n=4 is far too small for a formal significance test; no")
    print("p-values are reported. Effect sizes (Cohen's d, paired) and the")
    print("raw 4 paired differences are reported instead, per instruction.")
    print()
    for label, entry in report.items():
        print(f"--- {label} ---")
        for field, stats in entry.items():
            print(f"  {field}: pairs={stats['paired_diffs']} mean_diff={stats['mean_diff']} "
                  f"d={stats['cohen_d_paired']}")
    print()
    print(f"B's extra, non-paired aborted attempt (block 4, preceded the retry "
          f"used for pairing): {B_EXTRA_ABORT}")
    print(f"  active_tracking_rmse_mm={by_dir[B_EXTRA_ABORT]['active_tracking_rmse_mm']} "
          f"final_error_mm={by_dir[B_EXTRA_ABORT]['final_error_mm']} "
          f"stop_reason={by_dir[B_EXTRA_ABORT]['stop_reason']}")


if __name__ == "__main__":
    main()
