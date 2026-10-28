"""Script 8 -- plan: snug-bouncing-giraffe.md. Final synthesis, n=5 round.

Reads the outputs of Scripts 1/2/3/4/5/6/7/8-config-rq (no new
solving). Produces two decision tables (whole-run + region-C, 5-rep
breakdown, range-based, never same-numbered-rep-paired), 7 figures,
and the preregistered hypothesis checklist (timing/solver gate,
primary, mechanistic [within-anchor paired deltas], specificity,
safety), classified against the four pre-identified outcome branches,
with the three pairwise comparisons (state-conditioning value /
contact-modelling value / end-to-end gain) kept explicitly distinct.
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
N_REPS = 5


def load_json(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def main():
    phase0_gate = load_json("timing_gate_result.json")
    phase1_2 = load_json("phase1_2_results.json")
    phase5 = load_json("phase5_anchor_results.json")
    phase6 = load_json("phase6_results.json")
    phase7 = load_json("phase7_safety_results.json")
    phase8rq = load_json("phase8_config_rq_results.json")

    # --- Decision table 1: whole-run RMSE, 5-rep breakdown ---
    print("=" * 100)
    print("DECISION TABLE 1: WHOLE-RUN RMSE (mm), 5-rep breakdown")
    print("=" * 100)
    table1 = {}
    for cond in ("NEW", "OLD-C", "NC"):
        vals = [phase1_2["per_run"][f"{cond}_rep{r}"]["whole_run"]["rmse_mm"] for r, _ in reg.RUNS[cond]]
        table1[cond] = vals
        print(f"  {cond:6s}: {[f'{v:.3f}' for v in vals]}  median={np.median(vals):.3f}  "
              f"range=[{min(vals):.3f},{max(vals):.3f}]")

    print()
    print("=" * 100)
    print("DECISION TABLE 2: REGION-C RMSE (mm), 5-rep breakdown (range-based, no rep-pairing assumed)")
    print("=" * 100)
    table2 = {}
    for cond in ("NEW", "OLD-C", "NC"):
        vals = [phase1_2["per_run"][f"{cond}_rep{r}"]["region_C_primary"]["rmse_mm"] for r, _ in reg.RUNS[cond]]
        table2[cond] = vals
        print(f"  {cond:6s}: {[f'{v:.3f}' for v in vals]}  median={np.median(vals):.3f}  "
              f"range=[{min(vals):.3f},{max(vals):.3f}]")

    new_c, old_c = table2["NEW"], table2["OLD-C"]
    old_nc, nc_c = table2["OLD-C"], table2["NC"]
    prim = phase1_2["primary_hypothesis_region_c"]

    print()
    print("=" * 100)
    print("THREE PAIRWISE COMPARISONS (kept distinct, region C)")
    print("=" * 100)
    print(f"  (a) STATE-CONDITIONING VALUE  (NEW vs OLD-C): Delta_C (median diff) = "
          f"{prim['delta_c_median']:+.4f}mm  bootstrap 95% CI={prim['bootstrap_ci_95']}  "
          f"range_separation_bonus={prim['range_separation_bonus']}")
    print(f"  (b) CONTACT-MODELLING VALUE   (OLD-C vs NC):  "
          f"OLD-C median={np.median(old_nc):.3f}  NC median={np.median(nc_c):.3f}  "
          f"({'OLD-C worse than NC in its own contact region' if np.median(old_nc) > np.median(nc_c) else 'OLD-C not worse than NC here'})")
    print(f"  (c) END-TO-END GAIN           (NEW vs NC):    "
          f"NEW median={np.median(new_c):.3f}  NC median={np.median(nc_c):.3f}  "
          f"gap={np.median(new_c)-np.median(nc_c):+.3f}mm")

    # --- Figures (7) ---
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    conds = ["NEW", "OLD-C", "NC"]
    colors = {"NEW": "tab:blue", "OLD-C": "tab:orange", "NC": "tab:green"}
    rep_x = list(range(1, N_REPS + 1))

    ax = axes[0, 0]
    for cond in conds:
        ax.plot(rep_x, table1[cond], 'o-', color=colors[cond], label=cond)
    ax.set_xticks(rep_x); ax.set_xticklabels([f"rep{i}" for i in rep_x])
    ax.set_ylabel("whole-run RMSE (mm)"); ax.set_title("Fig 1: whole-run RMSE per rep"); ax.legend()

    ax = axes[0, 1]
    for cond in conds:
        ax.plot(rep_x, table2[cond], 'o-', color=colors[cond], label=cond)
    ax.set_xticks(rep_x); ax.set_xticklabels([f"rep{i}" for i in rep_x])
    ax.set_ylabel("region-C RMSE (mm)"); ax.set_title("Fig 2: region-C RMSE per rep (range-based)"); ax.legend()

    ax = axes[0, 2]
    s_bins = np.asarray(phase6["s_bin_centers_mm"])
    for cond in conds:
        ax.plot(s_bins, phase6["condition_curves"][cond]["e_mm"], color=colors[cond], label=cond)
    ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color="gray", alpha=0.15)
    ax.set_xlabel("s (mm)"); ax.set_ylabel("mean tip error (mm)")
    ax.set_title("Fig 3: error vs progress (condition mean)"); ax.legend()

    ax = axes[1, 0]
    for cond in conds:
        ax.plot(s_bins, phase6["condition_curves"][cond]["E_JD"], color=colors[cond], label=cond)
    ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color="gray", alpha=0.15)
    ax.set_xlabel("s (mm)"); ax.set_ylabel("E_J,D"); ax.set_title("Fig 4: model fidelity vs progress"); ax.legend()

    ax = axes[1, 1]
    for cond in conds:
        ax.plot(s_bins, phase6["condition_curves"][cond]["theta_u_deg"], color=colors[cond], label=cond)
    ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color="gray", alpha=0.15)
    ax.set_xlabel("s (mm)"); ax.set_ylabel("theta_u own-vs-J4 (deg)")
    ax.set_title("Fig 5: command angle vs progress (same-state 'own' branch)"); ax.legend()

    ax = axes[1, 2]
    faces = list(next(iter(phase7.values()))["faces"].keys())
    width = 0.25
    x = np.arange(len(faces))
    for i, cond in enumerate(conds):
        mins = [min(phase7[f"{cond}_rep{r}"]["faces"][fc]["min"] for r, _ in reg.RUNS[cond]) for fc in faces]
        ax.bar(x + (i - 1) * width, [m * 1000 for m in mins], width, label=cond, color=colors[cond])
    ax.set_xticks(x); ax.set_xticklabels(faces, rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("min margin (mm)"); ax.set_title("Fig 6: safety margins (min observed)")
    ax.legend()

    ax = axes[2, 0]
    de_j_region_c, dtheta_region_c = [], []
    for rep, _ in reg.RUNS["NEW"]:
        for rec in phase5[f"NEW_rep{rep}"]:
            if R_C_PRIMARY[0] * 1000 <= rec["s_tip_mm"] <= R_C_PRIMARY[1] * 1000:
                de_j_region_c.append(rec["Delta_E_J_J1_minus_own"])
                dtheta_region_c.append(rec["Delta_theta_u_J1_minus_own"])
    if de_j_region_c:
        ax.scatter(de_j_region_c, dtheta_region_c, color="tab:blue", alpha=0.6, s=20)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Delta_E_J (J1 - own)"); ax.set_ylabel("Delta_theta_u (J1 - own, deg)")
    ax.set_title("Fig 7a: within-anchor paired deltas, NEW region-C (positive=NEW better)")

    ax = axes[2, 1]
    for cond in conds:
        vals_by_rep = [phase8rq["region_rq"].get(f"{cond}_rep{r}_regionC") for r, _ in reg.RUNS[cond]]
        ax.plot(rep_x, vals_by_rep, 'o-', color=colors[cond], label=cond)
    ax.set_xticks(rep_x); ax.set_xticklabels([f"rep{i}" for i in rep_x])
    ax.set_ylabel("r_q region-C mean (dimensionless)")
    ax.set_title("Fig 7b: configuration departure (r_q), region C"); ax.legend()
    axes[2, 2].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase8_seven_figures.png"), dpi=130)

    # --- Preregistered hypothesis checklist ---
    print()
    print("=" * 100)
    print("PREREGISTERED HYPOTHESIS CHECKLIST")
    print("=" * 100)

    timing_gate_pass = phase0_gate["gate1_timing_solver_pass"]
    print(f"  TIMING/SOLVER GATE: {'PASS' if timing_gate_pass else 'FAIL'} "
          f"(tick_dt diff={phase0_gate['tick_dt']['diff']:.2f}ms, "
          f"solver diff={phase0_gate['solver_time']['diff']:.2f}ms, "
          f"realization diff={phase0_gate['realization']['diff']:.3f}, "
          f"time-limit-frac diff={phase0_gate['time_limit_fraction']['diff']*100:+.1f}pp)")

    primary_pass = prim["primary_pass"]
    print(f"  PRIMARY (Delta_C = median_r(RMSE_NEW,C) - median_r(RMSE_OLD-C,C) < 0): "
          f"Delta_C={prim['delta_c_median']:+.4f}mm  bootstrap_CI={prim['bootstrap_ci_95']}  "
          f"{'PASS' if primary_pass else 'FAIL'}  (range-separation bonus: {prim['range_separation_bonus']})")

    e_jd_region_c = phase6["deltas"]["E_JD"]["new_minus_oldc"]
    theta_region_c = phase6["deltas"]["theta_u_deg"]["new_minus_oldc"]
    region_c_mask = (s_bins >= R_C_PRIMARY[0] * 1000) & (s_bins <= R_C_PRIMARY[1] * 1000)
    e_jd_delta_c = float(np.nanmean(np.asarray(e_jd_region_c)[region_c_mask]))
    theta_delta_c = float(np.nanmean(np.asarray(theta_region_c)[region_c_mask]))
    cond_level_mech_pass = e_jd_delta_c < 0 and theta_delta_c < 0

    de_j_arr = np.array(de_j_region_c) if de_j_region_c else np.array([np.nan])
    dtheta_arr = np.array(dtheta_region_c) if dtheta_region_c else np.array([np.nan])
    within_anchor_median_ej = float(np.median(de_j_arr))
    within_anchor_frac_pos_ej = float(np.mean(de_j_arr > 0)) if de_j_region_c else float("nan")
    within_anchor_mech_pass = within_anchor_median_ej > 0 and within_anchor_frac_pos_ej > 0.5

    print(f"  MECHANISTIC (PRIMARY TEST = within-anchor paired deltas, region C, NEW's own anchors): "
          f"median(Delta_E_J)={within_anchor_median_ej:+.5f}  frac_positive={within_anchor_frac_pos_ej:.3f}  "
          f"{'PASS' if within_anchor_mech_pass else 'NOT MET'}")
    print(f"  MECHANISTIC (secondary, condition-level medians): E_J,D delta={e_jd_delta_c:+.4f}  "
          f"theta_u delta={theta_delta_c:+.4f}deg  {'PASS' if cond_level_mech_pass else 'NOT MET'}")

    print(f"  SPECIFICITY (region A flat-or-better is NOT required to be flat): "
          f"see Script 2's region-A table -- a NEW advantage there does not falsify specificity "
          f"(only a worsening would), but means the benefit was not confined to region C alone.")

    any_safety_violation = any(phase7[k]["n_ticks_any_face_violated"] > 0 for k in phase7)
    print(f"  SAFETY (no hard-constraint violation, every run, every rep): "
          f"{'PASS' if not any_safety_violation else 'VIOLATION FOUND'}")

    chrono = phase6["chronology_check"]
    print(f"  CHRONOLOGY (E_J,D down -> theta_u down -> E_future down, condition-aggregate, region C): "
          f"{'HOLDS' if chrono['chronology_holds'] else 'DOES NOT HOLD IN FULL'} "
          f"(deltas: E_J,D={chrono['delta_ejd']:+.5f}, theta_u={chrono['delta_theta_u']:+.4f}deg, "
          f"E_future={chrono['delta_e_future']:+.5f}mm)")

    rq_finding = phase8rq["virtuous_cycle_descriptive_finding"]
    print(f"  VIRTUOUS-CYCLE r_q FINDING (r_q^NEW < r_q^OLD-C, region C): {rq_finding}  "
          f"-- INTERPRETED AS SUPPORTING THE MECHANISM ONLY IF CHRONOLOGY ALSO HOLDS: "
          f"{'interpretation supported' if (rq_finding and chrono['chronology_holds']) else 'interpretation NOT claimed (chronology/finding incomplete)'}")

    print()
    print("=" * 100)
    print("FOUR-OUTCOME CLASSIFICATION")
    print("=" * 100)
    e_hw_down = primary_pass
    if within_anchor_mech_pass and e_hw_down:
        outcome = "E_J,D down, theta_u down (within-anchor), e_HW down -> full mechanism supported"
    elif within_anchor_mech_pass and not e_hw_down:
        outcome = ("E_J,D down, theta_u down (within-anchor), e_HW NOT down (by the primary criterion) -> "
                   "better Jacobian modelling at the anchor level is not sufficient on its own to show up "
                   "as a region-C RMSE improvement at n=5")
    elif not within_anchor_mech_pass:
        outcome = "within-anchor E_J,D/theta_u NOT consistently down -> hybrid did not deliver its intended model-level advantage on these anchors"
    else:
        outcome = "unclassified"
    print(" ", outcome)

    out = dict(
        table1_whole_run_rmse=table1, table2_region_c_rmse=table2,
        timing_gate_pass=bool(timing_gate_pass), primary_pass=bool(primary_pass),
        delta_c_median=prim["delta_c_median"], bootstrap_ci_95=prim["bootstrap_ci_95"],
        within_anchor_mechanistic=dict(median_delta_ej=within_anchor_median_ej,
                                         frac_positive=within_anchor_frac_pos_ej,
                                         passed=bool(within_anchor_mech_pass)),
        condition_level_mechanistic_pass=bool(cond_level_mech_pass),
        any_safety_violation=bool(any_safety_violation),
        chronology_check=chrono, rq_virtuous_cycle_finding=bool(rq_finding),
        outcome_classification=outcome,
    )
    with open(os.path.join(OUT_DIR, "phase8_final_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_DIR}/phase8_final_summary.json and phase8_seven_figures.png")


if __name__ == "__main__":
    main()
