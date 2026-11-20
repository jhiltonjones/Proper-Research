"""Section 5: causal forecast-accuracy comparison, scheduled (A) vs
frozen (B), by lead time. Uses each tick's full-horizon
predicted_beam_positions_m, saved BEFORE the corresponding future
measurement arrived, matched against tip_mm at the nearest later tick
within half a control period. This is prediction-accuracy evidence,
not a claim about which controller tracked better in closed loop
(that is reported separately in per_run_metrics.csv).

Caveat applied per instruction: only lead stages where the matched
measurement's own tick is "reasonably soon" are counted; later-lead
forecasts are influenced by RE-OPTIMIZED inputs issued on the ticks in
between, not pure model error, and this is stated in the output, not
silently absorbed into one number.
"""
import json
import os

import numpy as np

ROOT = "/home/jack/Proper-Research"
LOGS = os.path.join(ROOT, "close_loop_logs")
OUT_DIR = os.path.join(ROOT, "proper_research/analysis/condition_A_E_campaign")

A_RUNS = [
    "baseline_A_pilot_2026-11-20_retry1_20261120T122825Z",
    "baseline_A_pilot_2026-11-20_retry1_20261120T130429Z",
    "baseline_A_pilot_2026-11-20_retry1_20261120T131743Z",
    "baseline_A_pilot_2026-11-20_retry1_20261120T132839Z",
]
B_RUNS = [
    "condition_B_frozen_jacobian_2026-11-20_20261120T124729Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T130657Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T131921Z",
    "condition_B_frozen_jacobian_2026-11-20_20261120T133227Z",  # the completed retry, not the abort
]


def score_run(run_dir):
    d = os.path.join(LOGS, run_dir)
    pred = [json.loads(l) for l in open(os.path.join(d, "predicted_beam_positions.jsonl"))]
    pf = [json.loads(l) for l in open(os.path.join(d, "path_follow.jsonl"))]
    pf_by_step = {r["step"]: r for r in pf}
    t_by_step = {r["step"]: r["t_s"] for r in pf}
    tip_by_step = {
        r["step"]: np.asarray(r["tip_mm"], dtype=float) / 1e3
        for r in pf if r.get("tip_mm") is not None
    }
    t_all = sorted(t_by_step.values())
    dt = float(np.median(np.diff(t_all))) if len(t_all) > 1 else 0.1
    max_gap_s = 0.5 * dt

    by_lead = {}
    excluded = 0
    for r in pred:
        if not r.get("success") or r.get("predicted_beam_positions_m") is None:
            continue
        k = r["step"]
        t_k = t_by_step.get(k)
        if t_k is None:
            continue
        stages = np.asarray(r["predicted_beam_positions_m"], dtype=float)
        for j in range(stages.shape[0]):
            target_t = t_k + j * dt
            candidates = [(abs(t_by_step[s] - target_t), s) for s in t_by_step if t_by_step[s] >= t_k]
            if not candidates:
                excluded += 1
                continue
            gap, best_s = min(candidates)
            if gap > max_gap_s:
                excluded += 1
                continue
            meas = tip_by_step.get(best_s)
            if meas is None or np.any(np.isnan(meas)):
                excluded += 1
                continue
            err_mm = float(np.linalg.norm(stages[j] - meas)) * 1e3
            by_lead.setdefault(j, []).append(err_mm)
    return by_lead, excluded, dt


def aggregate(run_list, label):
    per_lead_all = {}
    total_excluded = 0
    dts = []
    for run_dir in run_list:
        by_lead, excluded, dt = score_run(run_dir)
        dts.append(dt)
        total_excluded += excluded
        for j, errs in by_lead.items():
            per_lead_all.setdefault(j, []).extend(errs)
    leads = sorted(per_lead_all)
    means = {j: float(np.mean(per_lead_all[j])) for j in leads}
    counts = {j: len(per_lead_all[j]) for j in leads}
    print(f"{label}: dt~{np.median(dts):.3f}s, excluded={total_excluded}, leads={leads[:3]}...{leads[-3:] if len(leads)>3 else ''}")
    return {"leads": leads, "mean_err_mm": means, "n": counts}


def main():
    a = aggregate(A_RUNS, "A (scheduled)")
    b = aggregate(B_RUNS, "B (frozen, 4 completed runs; abort's partial ticks excluded from this forecast comparison)")

    out = {"A_scheduled": a, "B_frozen": b,
           "note": ("Early lead stages (small j) are dominated by already-issued joint "
                    "commands; later lead stages reflect newly-optimized inputs from "
                    "intervening re-solves, not pure model error. B's aborted run "
                    "(133013Z) is excluded from this forecast table -- its early "
                    "ticks before the abort are a separate, much smaller sample and "
                    "are not pooled in here to avoid quietly diluting the comparison "
                    "with a different failure regime.")}
    with open(os.path.join(OUT_DIR, "forecast_scheduled_vs_frozen.json"), "w") as f:
        json.dump(out, f, indent=2)

    print()
    print("lead_j | A mean_err_mm (n) | B mean_err_mm (n)")
    common_leads = sorted(set(a["leads"]) & set(b["leads"]))
    for j in common_leads:
        print(f"{j:6d} | {a['mean_err_mm'][j]:.4f} ({a['n'][j]:5d}) | {b['mean_err_mm'][j]:.4f} ({b['n'][j]:5d})")


if __name__ == "__main__":
    main()
