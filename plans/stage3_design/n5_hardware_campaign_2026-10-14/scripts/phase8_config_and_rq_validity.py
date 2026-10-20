"""New script (n=5 round) -- plan: snug-bouncing-giraffe.md.

Compares L(s), joint-space departure, and r_q(s) for ALL THREE
conditions (not just NEW) -- OLD-C/NC get their own r_q computed the
SAME way NEW's hybrid model does (confirmed bit-exact against the
production ContactJacobianHybridModel.r_q, not re-derived), via the
generalized path_geometry.r_q/tangent added for this script.

Tests the proposed virtuous-cycle hypothesis (better J -> better
command -> less configuration departure -> map stays valid longer):
does r_q^NEW(s) < r_q^OLD-C(s) hold in region C? Reported here as a
descriptive finding ONLY -- its INTERPRETATION as supporting the
hypothesis is explicitly deferred to whether Script 6's chronology
check (E_J,D down -> theta_u down -> E_future down) also holds; this
script does not assert the mechanism on its own.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import run_registry as reg
import path_geometry as pg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

R_A = (0.012, 0.017)
R_C_PRIMARY = (0.055, 0.065)


def per_run_config_series(condition, rep, run_path):
    rows = reg.load_path_follow(run_path)
    plan_root = reg.plan_root_for(condition)
    tables = pg.load_path_tables(plan_root)

    n = len(rows)
    s_q = np.zeros(n)
    r_q_arr = np.zeros(n)
    delta_L_mm = np.zeros(n)
    q_departure_rad = np.zeros(n)

    for k, r in enumerate(rows):
        q_meas = np.asarray(r["q_meas_rad"], dtype=float)
        L_meas = float(r["insertion_length_m"])
        s_q_k = pg.project_s_q(q_meas, tables)
        s_q[k] = s_q_k
        r_q_arr[k] = pg.r_q(q_meas, s_q_k, tables)
        delta_L_mm[k] = (L_meas - pg.L_plan(s_q_k, tables)) * 1000.0
        q_departure_rad[k] = float(np.linalg.norm(q_meas - pg.q_plan(s_q_k, tables)))

    return dict(s_q_m=s_q, r_q=r_q_arr, delta_L_mm=delta_L_mm, q_departure_rad=q_departure_rad)


def region_mean(s_q, values, lo, hi):
    mask = (s_q >= lo) & (s_q <= hi)
    return float(np.mean(values[mask])) if mask.any() else float("nan")


def main():
    all_series = {}
    for cond, rep, path in reg.all_runs():
        all_series[(cond, rep)] = per_run_config_series(cond, rep, path)
        print(f"[{cond}_rep{rep}] done", flush=True)

    print()
    print("=" * 100)
    print("r_q(s) BY REGION, ALL THREE CONDITIONS (dimensionless, D_Q-normalized)")
    print("=" * 100)
    region_table = {}
    for region_label, (lo, hi) in (("A", R_A), ("C", R_C_PRIMARY)):
        print(f"--- region {region_label} = [{lo*1000:.0f},{hi*1000:.0f}]mm ---")
        for cond in ("NEW", "OLD-C", "NC"):
            vals = []
            for rep, _ in reg.RUNS[cond]:
                series = all_series[(cond, rep)]
                v = region_mean(series["s_q_m"], series["r_q"], lo, hi)
                vals.append(v)
                region_table[f"{cond}_rep{rep}_region{region_label}"] = v
            vals_clean = [v for v in vals if not np.isnan(v)]
            print(f"  {cond:6s}: per-rep={[f'{v:.3f}' for v in vals]}  "
                  f"median={np.median(vals_clean):.3f}  range=[{min(vals_clean):.3f},{max(vals_clean):.3f}]")

    new_c = [region_table[f"NEW_rep{r}_regionC"] for r, _ in reg.RUNS["NEW"]]
    old_c = [region_table[f"OLD-C_rep{r}_regionC"] for r, _ in reg.RUNS["OLD-C"]]
    virtuous_cycle_finding = bool(np.median(new_c) < np.median(old_c))
    print()
    print(f"DESCRIPTIVE FINDING: median(r_q^NEW, region C) = {np.median(new_c):.3f}  "
          f"median(r_q^OLD-C, region C) = {np.median(old_c):.3f}  "
          f"r_q^NEW < r_q^OLD-C: {virtuous_cycle_finding}")
    print("INTERPRETATION STATUS: this finding alone does NOT establish the virtuous-cycle mechanism -- "
          "see Script 6's chronology check (E_J,D down -> theta_u down -> E_future down) before "
          "attributing causality. Reported here as a descriptive input to that later synthesis only.")

    # L(s) and q_departure(s) summary, all conditions, whole-run means
    print()
    print("=" * 100)
    print("CONFIGURATION SUMMARY (whole-run means): |Delta_L| and joint departure")
    print("=" * 100)
    for cond in ("NEW", "OLD-C", "NC"):
        dl_vals, qd_vals = [], []
        for rep, _ in reg.RUNS[cond]:
            series = all_series[(cond, rep)]
            dl_vals.append(float(np.mean(np.abs(series["delta_L_mm"]))))
            qd_vals.append(float(np.mean(series["q_departure_rad"])))
        print(f"  {cond:6s}: mean|Delta_L|_mm per-rep={[f'{v:.3f}' for v in dl_vals]}  "
              f"median={np.median(dl_vals):.3f}")
        print(f"           mean q_departure_rad per-rep={[f'{v:.4f}' for v in qd_vals]}  "
              f"median={np.median(qd_vals):.4f}")

    # Figure: r_q(s) binned, condition-mean overlay
    s_bins = np.arange(0.0, 0.0761, 0.005)
    s_centers_mm = (s_bins[:-1] + s_bins[1:]) / 2 * 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"NEW": "tab:blue", "OLD-C": "tab:orange", "NC": "tab:green"}
    for cond in ("NEW", "OLD-C", "NC"):
        curves = []
        for rep, _ in reg.RUNS[cond]:
            series = all_series[(cond, rep)]
            idx = np.digitize(series["s_q_m"], s_bins) - 1
            idx = np.clip(idx, 0, len(s_centers_mm) - 1)
            binned = np.full(len(s_centers_mm), np.nan)
            for b in range(len(s_centers_mm)):
                mask = idx == b
                if mask.any():
                    binned[b] = np.mean(series["r_q"][mask])
            curves.append(binned)
        mean_curve = np.nanmean(np.stack(curves), axis=0)
        ax.plot(s_centers_mm, mean_curve, color=colors[cond], label=cond, linewidth=2)
    ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color="gray", alpha=0.15, label="region C")
    ax.axvspan(R_A[0] * 1000, R_A[1] * 1000, color="gray", alpha=0.3, label="region A")
    ax.set_xlabel("s_q (mm)")
    ax.set_ylabel("r_q (dimensionless, mean across 5 reps)")
    ax.set_title("Configuration departure from plan (r_q) vs progress, all three conditions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase8_rq_vs_progress.png"), dpi=130)

    out = dict(region_rq=region_table, virtuous_cycle_descriptive_finding=virtuous_cycle_finding)
    with open(os.path.join(OUT_DIR, "phase8_config_rq_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/phase8_config_rq_results.json and phase8_rq_vs_progress.png")


if __name__ == "__main__":
    main()
