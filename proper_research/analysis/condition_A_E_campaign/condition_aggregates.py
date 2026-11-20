"""Section 3 (tail): per-condition aggregation across included runs.
B's aborted run is reported separately (its executed-portion metrics
are already in per_run_metrics.csv) and EXCLUDED from B's "4 completed"
aggregate columns, with a second aggregate row ("B_all_5_incl_abort")
shown alongside so the abort cannot silently vanish from the summary.
n=4 (or n=5 for B_all) is the run; this is NOT a tick-level sample.
"""
import csv
import json
import os
import statistics

ROOT = "/home/jack/Proper-Research"
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

NUMERIC_FIELDS = [
    "active_tracking_rmse_mm", "active_tracking_mean_mm", "active_tracking_p95_mm",
    "active_tracking_max_mm", "final_error_mm", "run_duration_s",
    "active_motion_duration_s", "min_measured_clearance_mm", "insertion_min_mm",
    "insertion_max_mm", "command_smoothness_rms_rad_s", "n_solver_deadline_miss",
    "n_whole_tick_deadline_miss", "completed_path_fraction",
]


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def main():
    rows = json.load(open(os.path.join(OUT_DIR, "per_run_metrics.json")))

    groups = {}
    for r in rows:
        cond = r["condition"]
        if cond == "B" and not r["completed"]:
            groups.setdefault("B_aborted_only", []).append(r)
            continue
        groups.setdefault(cond, []).append(r)
    # B_all_5: every B run including the abort, for transparency
    groups["B_all_5_incl_abort"] = [r for r in rows if r["condition"] == "B"]

    out_rows = []
    for cond, rs in sorted(groups.items()):
        entry = {"condition": cond, "n_runs": len(rs), "run_dirs": ";".join(r["run_dir"] for r in rs)}
        for field in NUMERIC_FIELDS:
            s = _stats([r.get(field) for r in rs])
            entry[f"{field}_n"] = s["n"]
            entry[f"{field}_mean"] = s["mean"]
            entry[f"{field}_sd"] = s["sd"]
            entry[f"{field}_median"] = s["median"]
            entry[f"{field}_min"] = s["min"]
            entry[f"{field}_max"] = s["max"]
        out_rows.append(entry)

    fieldnames = list(out_rows[0].keys())
    with open(os.path.join(OUT_DIR, "condition_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print("=== Condition aggregates (n=run, not tick) ===")
    for e in out_rows:
        print(f"{e['condition']} (n={e['n_runs']}): "
              f"active_rmse mean={e['active_tracking_rmse_mm_mean']:.3f} sd={e['active_tracking_rmse_mm_sd']:.3f} "
              f"median={e['active_tracking_rmse_mm_median']:.3f} "
              f"final_err mean={e['final_error_mm_mean']:.3f} "
              f"min_clearance_mm min={e['min_measured_clearance_mm_min']:.1f}")
    return out_rows


if __name__ == "__main__":
    main()
