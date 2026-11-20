"""Script 2 -- plan: snug-bouncing-giraffe.md, n=5 round.

Per-run AND per-condition primary tracking outcomes vs physical path
progress s (same 0-75.32mm path for all 15 runs -- contact and
no-contact plans share identical path length, confirmed read-only).
Region boundaries A/B/C are FIXED FROM THE PRIOR INVESTIGATION (this
session's own history, not fit to these runs): R_A is the early
low-authority zone at ticks 81/87 on the SAME 527-sample reference,
which map to s=13.90mm and s=14.80mm respectively here -> R_A=[12,17]mm.
R_C=[55,65]mm is the already-established late contact-sensitive region
(contains tick 366 -> s=58.01mm). R_B is everything else (not A, not C).
A +/-2-3mm boundary-sensitivity check on R_C is reported alongside, not
instead of, the primary boundary.

Per-rep 5x3 (condition x rep) tables are reported for every metric/
region -- NOT mean+-SD only -- with an explicit caution against assuming
rep-number pairing (rep1-NEW is not assumed to correspond to rep1-OLD-C
unless genuinely matched, which they were not here). The primary effect
is the UNPAIRED median difference `Delta_C = median_r(RMSE^NEW_C,r) -
median_r(RMSE^OLD-C_C,r)`, with a run-level bootstrap CI (resampling
the 5 runs, never ticks); full-range separation
(`max_r(RMSE^NEW_C,r) < min_r(RMSE^OLD-C_C,r)`) is reported as a bonus
strong-evidence marker, not a requirement.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import run_registry as reg
import path_geometry as pg

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

R_A = (0.012, 0.017)   # m; from ticks 81/87 on the shared 527-sample reference
R_C_PRIMARY = (0.055, 0.065)   # m; established late contact-sensitive region (tick 366)
R_C_SENSITIVITY = [(0.053, 0.067), (0.052, 0.068)]   # +/-2mm, +/-3mm


def region_of(s_m):
    if R_A[0] <= s_m <= R_A[1]:
        return "A"
    if R_C_PRIMARY[0] <= s_m <= R_C_PRIMARY[1]:
        return "C"
    return "B"


def per_tick_series(condition, rep, run_path):
    rows = reg.load_path_follow(run_path)
    plan_root = reg.plan_root_for(condition)
    tables = pg.load_path_tables(plan_root)
    s_of_ref_index = pg.load_resampled_s_of_ref_index(plan_root)
    resampled_df = _resampled_tangents(plan_root)

    n = len(rows)
    s_ref = np.zeros(n)
    s_tip = np.zeros(n)
    err_mm = np.zeros(n)
    lag_mm = np.zeros(n)
    crosstrack_mm = np.zeros(n)

    for k, r in enumerate(rows):
        ref_idx = int(np.clip(r["ref_index"], 0, len(s_of_ref_index) - 1))
        s_ref[k] = s_of_ref_index[ref_idx]
        tip_m = np.asarray(r["tip_mm"], dtype=float) / 1000.0
        s_tip[k] = pg.project_s_cartesian(tip_m, tables)
        err_vec_mm = np.asarray(r["error_mm"], dtype=float)  # desired - tip, from the live controller
        err_mm[k] = float(np.linalg.norm(err_vec_mm))
        lag_mm[k] = (s_ref[k] - s_tip[k]) * 1000.0
        tangent = resampled_df[ref_idx]
        tangent_hat = tangent / max(np.linalg.norm(tangent), 1e-12)
        long_comp = np.dot(err_vec_mm, tangent_hat)
        crosstrack_mm[k] = float(np.linalg.norm(err_vec_mm - long_comp * tangent_hat))

    region = np.array([region_of(s) for s in s_tip])
    return dict(s_ref_m=s_ref, s_tip_m=s_tip, err_mm=err_mm, lag_mm=lag_mm,
                crosstrack_mm=crosstrack_mm, region=region, n_ticks=n)


_RESAMPLED_TANGENT_CACHE = {}


def _resampled_tangents(plan_root):
    if plan_root in _RESAMPLED_TANGENT_CACHE:
        return _RESAMPLED_TANGENT_CACHE[plan_root]
    import pandas as pd
    df = pd.read_csv(plan_root.rstrip("/") + "/time_parameterized_configuration_path/time_parameterized_configuration_path.csv")
    df = df.sort_values("sample_index")
    tang = df[["desired_tangent_x", "desired_tangent_y", "desired_tangent_z"]].to_numpy(dtype=float)
    _RESAMPLED_TANGENT_CACHE[plan_root] = tang
    return tang


def region_metrics(series, region_mask_fn):
    mask = region_mask_fn(series["s_tip_m"])
    if not np.any(mask):
        return None
    e = series["err_mm"][mask]
    lag = series["lag_mm"][mask]
    ct = series["crosstrack_mm"][mask]
    return dict(
        n=int(mask.sum()),
        rmse_mm=float(np.sqrt(np.mean(e ** 2))),
        median_mm=float(np.median(e)),
        p90_mm=float(np.percentile(e, 90)),
        p95_mm=float(np.percentile(e, 95)),
        max_mm=float(np.max(e)),
        mean_lag_mm=float(np.mean(lag)),
        median_lag_mm=float(np.median(lag)),
        max_lag_mm=float(np.max(np.abs(lag))),
        mean_crosstrack_mm=float(np.mean(ct)),
        max_crosstrack_mm=float(np.max(ct)),
    )


def whole_run_metrics(series):
    return region_metrics(series, lambda s: np.ones_like(s, dtype=bool))


def common_progress_metrics(series, s_min_common):
    return region_metrics(series, lambda s: s <= s_min_common)


N_REPS = 5
REP_RANGE = tuple(range(1, N_REPS + 1))


def print_3x3_table(title, results, metric_key):
    print(f"--- {title} ({metric_key}) ---")
    conds = ["NEW", "OLD-C", "NC"]
    header = f"{'rep':6s}" + "".join(f"{c:>12s}" for c in conds)
    print(header)
    for rep in REP_RANGE:
        row = f"rep{rep:<3d}"
        for c in conds:
            v = results.get((c, rep))
            row += f"{v[metric_key] if v is not None else float('nan'):>12.4f}"
        print(row)
    for c in conds:
        vals = [results[(c, r)][metric_key] for r in REP_RANGE if results.get((c, r)) is not None]
        if vals:
            print(f"  {c:6s} mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  "
                  f"range=[{min(vals):.4f},{max(vals):.4f}]")


def main():
    all_series = {}
    for cond, rep, path in reg.all_runs():
        all_series[(cond, rep)] = per_tick_series(cond, rep, path)

    final_s_tip = {k: v["s_tip_m"][-1] for k, v in all_series.items()}
    s_min_common = min(final_s_tip.values())
    print("=" * 100)
    print("FINAL PROGRESS (s_tip, mm) per run -- completion already confirmed path_complete by Script 1")
    print("=" * 100)
    for (c, r), s in sorted(final_s_tip.items()):
        print(f"  {c:6s} rep{r}: final s_tip={s*1000:.3f}mm")
    print(f"s_min_common across all 15 runs = {s_min_common*1000:.3f}mm "
          f"(used to clip the matched common-progress analysis below)")

    region_names = {"A": lambda s: (R_A[0] <= s) & (s <= R_A[1]),
                     "C_primary": lambda s: (R_C_PRIMARY[0] <= s) & (s <= R_C_PRIMARY[1]),
                     "B": lambda s: ~(((R_A[0] <= s) & (s <= R_A[1])) | ((R_C_PRIMARY[0] <= s) & (s <= R_C_PRIMARY[1])))}

    out = dict(s_min_common_m=s_min_common, region_boundaries_m=dict(R_A=R_A, R_C=R_C_PRIMARY),
                per_run={})

    print()
    print("=" * 100)
    print("WHOLE-RUN METRICS (every tick, no progress clipping)")
    print("=" * 100)
    whole_run_results = {}
    for (c, r), series in all_series.items():
        m = whole_run_metrics(series)
        whole_run_results[(c, r)] = m
        out["per_run"].setdefault(f"{c}_rep{r}", {})["whole_run"] = m
    for metric in ("rmse_mm", "median_mm", "p90_mm", "max_mm", "mean_lag_mm", "mean_crosstrack_mm"):
        print_3x3_table("WHOLE RUN", whole_run_results, metric)
        print()

    print("=" * 100)
    print(f"COMMON-PROGRESS METRICS (clipped to s<={s_min_common*1000:.2f}mm, matched comparison)")
    print("=" * 100)
    common_results = {}
    for (c, r), series in all_series.items():
        m = common_progress_metrics(series, s_min_common)
        common_results[(c, r)] = m
        out["per_run"][f"{c}_rep{r}"]["common_progress"] = m
    for metric in ("rmse_mm", "median_mm", "p90_mm", "max_mm"):
        print_3x3_table("COMMON PROGRESS", common_results, metric)
        print()

    for region_label, region_fn in region_names.items():
        print("=" * 100)
        print(f"REGION {region_label} (primary boundary): "
              f"{'R_A=[12,17]mm' if region_label=='A' else 'R_C=[55,65]mm' if region_label=='C_primary' else 'everything else'}")
        print("=" * 100)
        region_results = {}
        for (c, r), series in all_series.items():
            m = region_metrics(series, region_fn)
            region_results[(c, r)] = m
            out["per_run"][f"{c}_rep{r}"][f"region_{region_label}"] = m
        for metric in ("n", "rmse_mm", "median_mm", "max_mm", "mean_lag_mm", "max_lag_mm", "mean_crosstrack_mm"):
            print_3x3_table(f"REGION {region_label}", region_results, metric)
            print()

    print("=" * 100)
    print("REGION-C BOUNDARY SENSITIVITY (+/-2mm, +/-3mm around the primary [55,65]mm boundary)")
    print("=" * 100)
    sensitivity_out = {}
    for (lo, hi) in [R_C_PRIMARY] + R_C_SENSITIVITY:
        label = f"[{lo*1000:.0f},{hi*1000:.0f}]mm"
        results = {}
        for (c, r), series in all_series.items():
            m = region_metrics(series, lambda s, lo=lo, hi=hi: (lo <= s) & (s <= hi))
            results[(c, r)] = m
        sensitivity_out[label] = {f"{c}_rep{r}": results[(c, r)] for (c, r) in results}
        print(f"--- R_C boundary = {label} ---")
        print_3x3_table(f"R_C={label}", results, "rmse_mm")
        print()
    out["region_c_sensitivity"] = sensitivity_out

    # Primary effect: UNPAIRED median difference, with a run-level bootstrap CI.
    region_c_new = [out["per_run"][f"NEW_rep{r}"]["region_C_primary"]["rmse_mm"]
                     if out["per_run"][f"NEW_rep{r}"]["region_C_primary"] else None for r in REP_RANGE]
    region_c_old = [out["per_run"][f"OLD-C_rep{r}"]["region_C_primary"]["rmse_mm"]
                     if out["per_run"][f"OLD-C_rep{r}"]["region_C_primary"] else None for r in REP_RANGE]
    region_c_new = [v for v in region_c_new if v is not None]
    region_c_old = [v for v in region_c_old if v is not None]

    print("=" * 100)
    print("PRIMARY HYPOTHESIS CHECK (region C -- NOT same-numbered-rep-paired)")
    print("=" * 100)
    print(f"NEW region-C RMSE across its {len(region_c_new)} reps:   {[f'{v:.3f}' for v in region_c_new]}  "
          f"median={np.median(region_c_new):.3f}  range=[{min(region_c_new):.3f},{max(region_c_new):.3f}]mm")
    print(f"OLD-C region-C RMSE across its {len(region_c_old)} reps:  {[f'{v:.3f}' for v in region_c_old]}  "
          f"median={np.median(region_c_old):.3f}  range=[{min(region_c_old):.3f},{max(region_c_old):.3f}]mm")

    delta_c_median = float(np.median(region_c_new) - np.median(region_c_old))
    delta_c_mean = float(np.mean(region_c_new) - np.mean(region_c_old))
    print(f"\nDelta_C (median difference, primary effect) = {delta_c_median:+.4f}mm  "
          f"(mean-difference form: {delta_c_mean:+.4f}mm)")

    rng = np.random.default_rng(12345)  # fixed seed: reproducible, chosen before inspecting these results
    n_boot = 10000
    boot_deltas = np.empty(n_boot)
    new_arr, old_arr = np.asarray(region_c_new), np.asarray(region_c_old)
    for i in range(n_boot):
        boot_new = rng.choice(new_arr, size=len(new_arr), replace=True)
        boot_old = rng.choice(old_arr, size=len(old_arr), replace=True)
        boot_deltas[i] = np.median(boot_new) - np.median(boot_old)
    ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])
    print(f"Run-level bootstrap 95% CI on Delta_C (resampling the {len(new_arr)} vs {len(old_arr)} "
          f"RUNS, never ticks, n_boot={n_boot}): [{ci_lo:+.4f}, {ci_hi:+.4f}]mm")

    primary_pass = delta_c_median < 0.0
    range_separation = max(region_c_new) < min(region_c_old)
    print(f"\nPRIMARY (Delta_C < 0): {'PASS' if primary_pass else 'FAIL'}")
    print(f"BONUS strong-evidence marker (max_r RMSE_C^NEW < min_r RMSE_C^OLD-C, full range "
          f"separation -- NOT required): {'YES' if range_separation else 'no'}")

    out["primary_hypothesis_region_c"] = dict(
        new_rmse=region_c_new, old_c_rmse=region_c_old,
        delta_c_median=delta_c_median, delta_c_mean=delta_c_mean,
        bootstrap_ci_95=[float(ci_lo), float(ci_hi)], n_boot=n_boot,
        primary_pass=bool(primary_pass), range_separation_bonus=bool(range_separation),
    )

    with open(os.path.join(OUT_DIR, "phase1_2_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print()
    print(f"Wrote {OUT_DIR}/phase1_2_results.json")

    # persist per-tick series for later scripts (avoids recomputing s_tip projections)
    np.savez(os.path.join(OUT_DIR, "phase1_2_series.npz"),
              **{f"{c}_rep{r}__{k}": v for (c, r), series in all_series.items()
                 for k, v in series.items() if k != "n_ticks"})


if __name__ == "__main__":
    main()
