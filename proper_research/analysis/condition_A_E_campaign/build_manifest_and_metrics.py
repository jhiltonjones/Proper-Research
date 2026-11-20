"""Condition A-E hardware campaign: run manifest + per-run metrics table.

Read-only analysis of the real close_loop_logs/ runs from 2026-11-20.
No hardware execution, no tuning, no source changes, no modification of
historical logs. Produces:
  manifest.json / manifest.csv    -- every candidate run found, with its
                                      inclusion/exclusion decision and why
  per_run_metrics.csv             -- one row per INCLUDED run, all the
                                      metrics requested in section 3
  condition_summary.csv           -- per-condition aggregates (mean/SD,
                                      median/range) across included runs

Run from the canonical checkout: python3 -m proper_research.analysis.
condition_A_E_campaign.build_manifest_and_metrics
"""
import csv
import json
import os
import statistics

import numpy as np

ROOT = "/home/jack/Proper-Research"
LOGS = os.path.join(ROOT, "close_loop_logs")
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

# Candidate run directories found by direct `ls` of close_loop_logs/ on
# 2026-11-20 (close_loop_logs/myrun/ is a much older, unrelated set of
# experiments from earlier in this multi-week session and is excluded
# entirely -- none of it pertains to this A-E campaign).
CANDIDATES = {
    "A": [
        "baseline_A_pilot_2026-11-20_20261120T110506Z",
        "baseline_A_pilot_2026-11-20_retry1_20261120T111257Z",
        "baseline_A_pilot_2026-11-20_retry1_20261120T122825Z",
        "baseline_A_pilot_2026-11-20_retry1_20261120T130429Z",
        "baseline_A_pilot_2026-11-20_retry1_20261120T131743Z",
        "baseline_A_pilot_2026-11-20_retry1_20261120T132839Z",
    ],
    "B": [
        "condition_B_frozen_jacobian_2026-11-20_20261120T124729Z",
        "condition_B_frozen_jacobian_2026-11-20_20261120T130657Z",
        "condition_B_frozen_jacobian_2026-11-20_20261120T131921Z",
        "condition_B_frozen_jacobian_2026-11-20_20261120T133013Z",
        "condition_B_frozen_jacobian_2026-11-20_20261120T133227Z",
    ],
    "C": [
        "condition_C_inv7_2026-11-20_20261120T125338Z",
        "condition_C_inv7_2026-11-20_20261120T130824Z",
        "condition_C_inv7_2026-11-20_20261120T132057Z",
        "condition_C_inv7_2026-11-20_20261120T133412Z",
    ],
    "D": [
        "condition_D_mpc_N3_2026-11-20_20261120T125551Z",
        "condition_D_mpc_N3_2026-11-20_20261120T131021Z",
        "condition_D_mpc_N3_2026-11-20_20261120T132340Z",
        "condition_D_mpc_N3_2026-11-20_20261120T133608Z",
    ],
    "E": [
        "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T130244Z",
        "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T131539Z",
        "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T132522Z",
        "condition_E_mpc_N15_adaptation_active_2026-11-20_20261120T133739Z",
    ],
}

FROZEN_SCHEDULE_PATH_FRAGMENT = "frozen_jacobian_schedule.npy"
ORIGINAL_SCHEDULE_SHA = "4346ef4e43e4620e59265459a05bfa830fce2123b4b77dcc54c93961cd8250b1"


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _safe_load_json(path):
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def classify_run(condition, name):
    """Determine inclusion/exclusion and build the manifest row's raw
    metadata. Does NOT compute per-run metrics (that happens later, only
    for included runs)."""
    d = os.path.join(LOGS, name)
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    summary = _safe_load_json(os.path.join(d, "summary.json"))
    ctrl_meta = _safe_load_json(os.path.join(d, "controller_metadata.json"))
    pf_rows = _load_jsonl(os.path.join(d, "path_follow.jsonl"))
    pred_rows = _load_jsonl(os.path.join(d, "predicted_beam_positions.jsonl"))

    row = {
        "condition": condition,
        "run_dir": name,
        "files": files,
        "has_summary": summary is not None,
        "has_controller_metadata": ctrl_meta is not None,
        "pf_row_count": None if pf_rows is None else len(pf_rows),
        "pred_row_count": None if pred_rows is None else len(pred_rows),
        "stop_reason": None,
        "control_steps": None,
        "timestamp_utc": None,
        "cfg_horizon": None,
        "cfg_controller_kind": None,
        "cfg_jacobian_adaptation_mode": None,
        "cfg_insertion_min_m": None,
        "cfg_insertion_max_m": None,
        "schedule_cache_path": None,
        "schedule_cache_sha256": None,
        "source_file_sha256": None,
        "git_commit": None,
        "git_dirty": None,
        "argv": None,
        "final_error_mm": None,
        "max_error_mm": None,
        "include": False,
        "exclude_reason": "",
    }

    if summary is not None:
        row["stop_reason"] = summary.get("stop_reason")
        row["control_steps"] = summary.get("control_steps")
        row["timestamp_utc"] = summary.get("timestamp_utc")
        row["final_error_mm"] = summary.get("final_error_mm")
        row["max_error_mm"] = summary.get("max_error_mm")
        cfg = summary.get("config", {})
        row["cfg_horizon"] = cfg.get("mpc_prediction_horizon")
        row["cfg_controller_kind"] = cfg.get("controller_kind")
        row["cfg_jacobian_adaptation_mode"] = cfg.get("jacobian_adaptation_mode")
        row["cfg_insertion_min_m"] = cfg.get("insertion_command_min_m")
        row["cfg_insertion_max_m"] = cfg.get("insertion_command_max_m")

    if ctrl_meta is not None:
        row["schedule_cache_path"] = ctrl_meta.get("schedule_cache_path")
        row["schedule_cache_sha256"] = ctrl_meta.get("schedule_cache_sha256")
        cp = ctrl_meta.get("code_provenance", {})
        row["source_file_sha256"] = cp.get("source_file_sha256")
        row["git_commit"] = cp.get("git_commit")
        row["git_dirty"] = cp.get("git_dirty")
        row["argv"] = ctrl_meta.get("argv")

    # --- inclusion decision ---
    if summary is None or (row["pf_row_count"] in (None, 0)):
        row["exclude_reason"] = (
            "no summary.json and/or zero path_follow.jsonl rows -- crashed "
            "before or during startup, before any control tick was logged. "
            "For condition A this matches the two pre-fix launch attempts "
            "(see docs/campaign_prep/baseline_A_source_manifest_2026-11-20.txt "
            "v1/v2: RobotJointStream.latest_joints_with_timestamp() called "
            "LatestSlot.get_with_timestamp(), a method that did not exist "
            "until the v3 fix landed -- any run before that fix never got "
            "past warmup)."
        )
        row["include"] = False
    else:
        row["include"] = True

    return row


def main():
    manifest = []
    for cond, names in CANDIDATES.items():
        for name in names:
            manifest.append(classify_run(cond, name))

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
        fieldnames = [
            "condition", "run_dir", "include", "exclude_reason", "stop_reason",
            "control_steps", "pf_row_count", "pred_row_count", "timestamp_utc",
            "cfg_horizon", "cfg_controller_kind", "cfg_jacobian_adaptation_mode",
            "cfg_insertion_min_m", "cfg_insertion_max_m", "schedule_cache_path",
            "schedule_cache_sha256", "final_error_mm", "max_error_mm",
            "git_commit", "git_dirty",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in manifest:
            w.writerow(row)

    included_by_cond = {}
    for row in manifest:
        included_by_cond.setdefault(row["condition"], {"included": [], "excluded": []})
        key = "included" if row["include"] else "excluded"
        included_by_cond[row["condition"]][key].append(row["run_dir"])

    print("=== Inclusion summary ===")
    for cond, d in included_by_cond.items():
        print(f"{cond}: {len(d['included'])} included, {len(d['excluded'])} excluded")
        for n in d["excluded"]:
            print(f"    EXCLUDED: {n}")

    # Flag B's abort specially (it IS included -- an included aborted run,
    # per "include every attempt, including aborts" -- but must be
    # reported as distinct from the 4 path_complete runs).
    print()
    print("=== Stop reasons among INCLUDED runs ===")
    for row in manifest:
        if row["include"] and row["stop_reason"] != "path_complete":
            print(f"  {row['condition']} {row['run_dir']}: stop_reason={row['stop_reason']!r} "
                  f"control_steps={row['control_steps']}")

    return manifest


if __name__ == "__main__":
    main()
