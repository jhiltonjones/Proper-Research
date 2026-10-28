"""Script 6 -- plan: snug-bouncing-giraffe.md.

Reuses Scripts 2/4/5's already-computed series (does not resolve
anything). Two parts:

1. Condition-level, progress-aligned curves (5mm s-bins, 0-75mm):
   mean tip error e(s) [Script 2], mean E_J,D(s) [Script 4, sampled
   ticks], mean theta_u_own_vs_J4(s) [Script 5, sampled anchors] --
   averaged ACROSS a condition's 3 reps at each bin, never by
   subtracting same-numbered reps. Delta_e, Delta_E_J, Delta_theta_u
   are then the bin-wise differences between condition-mean curves
   (NEW-OLD-C, OLD-C-NC, NEW-NC), reported descriptively (no
   inferential correlation claim).

2. E_future(k) = (1/H) * sum_{h=d}^{d+H-1} ||e_{k+h}|| (H samples,
   using the REAL tick delay d=delay_samples and horizon H, from this
   run's own controller_metadata -- both are 2 and 15 for every run
   here per Script 1's manifest), computed WITHIN each run using
   Script 2's own err_mm series. This is compared, per run, against
   Script 4's sampled E_J,D(k) at the SAME tick k -- a descriptive
   scatter/correlation only, never used to link a command's model
   fidelity to anything other than its own run's own later physical
   consequence.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import run_registry as reg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")

R_C_PRIMARY = (0.055, 0.065)
S_BIN_EDGES_M = np.arange(0.0, 0.0761, 0.005)
S_BIN_CENTERS_MM = (S_BIN_EDGES_M[:-1] + S_BIN_EDGES_M[1:]) / 2 * 1000
DELAY_SAMPLES = reg.DELAY_SAMPLES
HORIZON = reg.HORIZON


def load_series_npz():
    data = np.load(os.path.join(OUT_DIR, "phase1_2_series.npz"))
    out = {}
    for cond, rep, _ in reg.all_runs():
        key = f"{cond}_rep{rep}"
        out[key] = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith(key + "__")}
    return out


def bin_by_s(s_m, values):
    bin_idx = np.digitize(s_m, S_BIN_EDGES_M) - 1
    bin_idx = np.clip(bin_idx, 0, len(S_BIN_CENTERS_MM) - 1)
    sums = np.zeros(len(S_BIN_CENTERS_MM))
    counts = np.zeros(len(S_BIN_CENTERS_MM))
    for i, v in zip(bin_idx, values):
        if np.isfinite(v):
            sums[i] += v
            counts[i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    means[counts == 0] = np.nan
    return means


def main():
    series = load_series_npz()
    with open(os.path.join(OUT_DIR, "phase4_e_jd_results.json")) as f:
        e_jd_results = json.load(f)
    with open(os.path.join(OUT_DIR, "phase5_anchor_results.json")) as f:
        anchor_results = json.load(f)

    # --- Part 1: condition-level s-binned curves ---
    curves = {}  # cond -> metric -> array over bins (mean across reps)
    for cond in ("NEW", "OLD-C", "NC"):
        e_bins_per_rep, ejd_bins_per_rep, theta_bins_per_rep = [], [], []
        for rep, _ in reg.RUNS[cond]:
            key = f"{cond}_rep{rep}"
            s_tip = series[key]["s_tip_m"]
            err_mm = series[key]["err_mm"]
            e_bins_per_rep.append(bin_by_s(s_tip, err_mm))

            d = e_jd_results[key]
            idxs = d["sampled_idx"]
            s_sampled = np.asarray(d["s_tip_m"])[idxs]
            ejd_sampled = np.asarray(d["E_JD"])[idxs]
            ejd_bins_per_rep.append(bin_by_s(s_sampled, ejd_sampled))

            recs = anchor_results[key]
            if recs:
                s_anchor = np.array([r["s_tip_mm"] / 1000.0 for r in recs])
                theta_anchor = np.array([r.get("theta_u_own_vs_J4_deg", np.nan) for r in recs])
                theta_bins_per_rep.append(bin_by_s(s_anchor, theta_anchor))
            else:
                theta_bins_per_rep.append(np.full(len(S_BIN_CENTERS_MM), np.nan))

        curves[cond] = dict(
            e_mm=np.nanmean(np.stack(e_bins_per_rep), axis=0),
            E_JD=np.nanmean(np.stack(ejd_bins_per_rep), axis=0),
            theta_u_deg=np.nanmean(np.stack(theta_bins_per_rep), axis=0),
        )

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colors = {"NEW": "tab:blue", "OLD-C": "tab:orange", "NC": "tab:green"}
    for cond in ("NEW", "OLD-C", "NC"):
        axes[0].plot(S_BIN_CENTERS_MM, curves[cond]["e_mm"], color=colors[cond], label=cond)
        axes[1].plot(S_BIN_CENTERS_MM, curves[cond]["E_JD"], color=colors[cond], label=cond)
        axes[2].plot(S_BIN_CENTERS_MM, curves[cond]["theta_u_deg"], color=colors[cond], label=cond)
    for ax, ylabel in zip(axes, ["tip error e(s) (mm)", "E_J,D(s)", "theta_u own-vs-J4 (deg)"]):
        ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color="gray", alpha=0.15)
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[-1].set_xlabel("physical progress s (mm)")
    fig.suptitle("Condition-level, progress-aligned curves (5mm bins, mean across 3 reps)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase6_condition_curves.png"), dpi=130)

    print("=" * 100)
    print("CONDITION-LEVEL DELTAS (NEW-OLD-C, OLD-C-NC, NEW-NC), region-C bins only")
    print("=" * 100)
    region_c_mask = (S_BIN_CENTERS_MM >= R_C_PRIMARY[0] * 1000) & (S_BIN_CENTERS_MM <= R_C_PRIMARY[1] * 1000)
    out_deltas = {}
    for metric in ("e_mm", "E_JD", "theta_u_deg"):
        d_new_old = curves["NEW"][metric] - curves["OLD-C"][metric]
        d_old_nc = curves["OLD-C"][metric] - curves["NC"][metric]
        d_new_nc = curves["NEW"][metric] - curves["NC"][metric]
        print(f"  {metric:14s} region-C mean Delta(NEW-OLD-C)={np.nanmean(d_new_old[region_c_mask]):+.4f}  "
              f"Delta(OLD-C-NC)={np.nanmean(d_old_nc[region_c_mask]):+.4f}  "
              f"Delta(NEW-NC)={np.nanmean(d_new_nc[region_c_mask]):+.4f}")
        out_deltas[metric] = dict(
            new_minus_oldc=d_new_old.tolist(), oldc_minus_nc=d_old_nc.tolist(), new_minus_nc=d_new_nc.tolist())

    # --- Part 2: E_future(k) within-run, descriptive correlation with E_J,D(k) ---
    print()
    print("=" * 100)
    print(f"E_future(k) within-run (d={DELAY_SAMPLES}, H={HORIZON}) vs Script 4's sampled E_J,D(k)")
    print("=" * 100)
    e_future_report = {}
    e_future_bins_by_cond = {cond: [] for cond in ("NEW", "OLD-C", "NC")}
    for cond, rep, _ in reg.all_runs():
        key = f"{cond}_rep{rep}"
        err_mm = series[key]["err_mm"]
        s_tip = series[key]["s_tip_m"]
        N = len(err_mm)
        d_jd = e_jd_results[key]
        idxs = np.asarray(d_jd["sampled_idx"])
        ejd = np.asarray(d_jd["E_JD"])[idxs]

        e_future = np.full(len(idxs), np.nan)
        for i, k in enumerate(idxs):
            lo, hi = k + DELAY_SAMPLES, k + DELAY_SAMPLES + HORIZON  # H samples, h=d..d+H-1
            if hi <= N:
                e_future[i] = float(np.mean(err_mm[lo:hi]))
        valid = np.isfinite(e_future) & np.isfinite(ejd)
        if valid.sum() > 5:
            r = float(np.corrcoef(ejd[valid], e_future[valid])[0, 1])
        else:
            r = float("nan")
        e_future_report[key] = dict(pearson_r=r, n=int(valid.sum()))
        print(f"  {key:12s} pearson_r(E_J,D, E_future)={r:+.3f}  n={int(valid.sum())}")

        e_future_bins_by_cond[cond].append(bin_by_s(s_tip[idxs], e_future))

    e_future_curve = {cond: np.nanmean(np.stack(v), axis=0) for cond, v in e_future_bins_by_cond.items()}

    # --- Chronology-consistency check (condition-aggregate, region C): does
    # E_J,D down -> theta_u down -> E_future down hold IN THAT ORDER for
    # NEW vs OLD-C? Descriptive only (signs/order), not an inferential claim
    # -- temporal autocorrelation means tick-level correlation alone would
    # not be strong evidence either way.
    print()
    print("=" * 100)
    print("CHRONOLOGY-CONSISTENCY CHECK (condition-aggregate, region C, NEW vs OLD-C): "
          "does E_J,D down -> theta_u down -> E_future down hold in order?")
    print("=" * 100)
    d_ejd = float(np.nanmean(curves["NEW"]["E_JD"][region_c_mask]) - np.nanmean(curves["OLD-C"]["E_JD"][region_c_mask]))
    d_theta = float(np.nanmean(curves["NEW"]["theta_u_deg"][region_c_mask]) - np.nanmean(curves["OLD-C"]["theta_u_deg"][region_c_mask]))
    d_efuture = float(np.nanmean(e_future_curve["NEW"][region_c_mask]) - np.nanmean(e_future_curve["OLD-C"][region_c_mask]))
    print(f"  Delta(E_J,D)   NEW-OLD-C = {d_ejd:+.5f}  ({'down' if d_ejd < 0 else 'NOT down'})")
    print(f"  Delta(theta_u) NEW-OLD-C = {d_theta:+.4f}deg  ({'down' if d_theta < 0 else 'NOT down'})")
    print(f"  Delta(E_future) NEW-OLD-C = {d_efuture:+.5f}mm  ({'down' if d_efuture < 0 else 'NOT down'})")
    chronology_holds = d_ejd < 0 and d_theta < 0 and d_efuture < 0
    print(f"  CHRONOLOGY HOLDS IN FULL (E_J,D down -> theta_u down -> E_future down): "
          f"{'YES' if chronology_holds else 'NO'}")
    chronology_result = dict(delta_ejd=d_ejd, delta_theta_u=d_theta, delta_e_future=d_efuture,
                               chronology_holds=bool(chronology_holds))

    with open(os.path.join(OUT_DIR, "phase6_results.json"), "w") as f:
        json.dump(dict(condition_curves={c: {m: v.tolist() for m, v in curves[c].items()} for c in curves},
                        e_future_curve={c: v.tolist() for c, v in e_future_curve.items()},
                        s_bin_centers_mm=S_BIN_CENTERS_MM.tolist(),
                        deltas=out_deltas, e_future_correlation=e_future_report,
                        chronology_check=chronology_result), f, indent=2)
    print(f"\nWrote {OUT_DIR}/phase6_results.json and phase6_condition_curves.png")


if __name__ == "__main__":
    main()
