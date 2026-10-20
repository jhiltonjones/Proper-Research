"""Script 4 -- plan: snug-bouncing-giraffe.md.

For common sampled ticks (stride-5 + full region-C coverage) across all
9 runs, computes:
  - J4(k): the full contact-model Jacobian oracle, recomputed via the
    canonical corrected bundle (phase0_helper.build_corrected_jacobian_
    provider(), same construction as build_c_schedule_floor240.py),
    evaluated AT THE ACTUAL MEASURED STATE (q_meas_rad, insertion_
    length_m) from hardware -- not at a path-reprojected nominal state.
    Warm-start continuation: reset to a fixed baseline snapshot, pre-
    solve at the nominal path point nearest the measured tip (s_tip,
    already computed by Script 2), then solve at the true measured
    state -- avoids the warm-start/branch-aliasing bug found earlier
    in this investigation.
  - J_used(k): what the live controller actually used at that tick.
      NEW:   w*J3(s_q,dL) + (1-w)*J1_schedule[ref_index]  (exact
             reconstruction of the production hybrid blend, using
             Script 3's per-tick s_q/dL/w diagnostics).
      OLD-C: J1_schedule[ref_index]  (the plain scheduled contact
             Jacobian, unmodified).
      NC:    J_NC_schedule[ref_index]  (the no-contact schedule's own
             row at NC's own ref_index -- NC never needs cross-
             condition index matching for J_used itself).
  - E_J,D(k) = || (J4(k) - J_used(k)) @ diag(D) ||_F , the authority-
    normalized (D = [VLIM6*DT]*6 + [INS_RATE_LIM*DT]) Jacobian-mismatch
    norm from this investigation's established SVD/modal methodology.

E_J,D is plotted against each run's OWN physical progress s_tip (from
Script 2), not raw tick/sample index -- this is what makes the 527-
sample contact schedule and 415-sample no-contact schedule comparable
on one axis.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/contact-mechanism-analysis")
import run_registry as reg
import path_geometry as pg

from phase0_helper import build_corrected_jacobian_provider
from proper_research.controllers.beam_jacobian_providers import position_jacobian_from_output
from proper_research.controllers.contact_jacobian_hybrid import ContactJacobianHybridModel, HybridMapConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "comparison_out")
os.makedirs(OUT_DIR, exist_ok=True)

DT = reg.DT
VLIM6, INS_RATE_LIM = 0.10, 2.0e-3
D7 = np.array([VLIM6] * 6 + [INS_RATE_LIM], dtype=float) * DT

SCHED_C = np.load("/home/jack/Proper-Research/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_floor240_2026-10-12.npy")
SCHED_NC = np.load("/home/jack/Proper-Research/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_floor240_2026-10-12.npy")

R_C_PRIMARY = (0.055, 0.065)
STRIDE = 5


def sample_tick_indices(n_ticks, s_tip_arr):
    strided = set(range(0, n_ticks, STRIDE))
    region_c = set(int(i) for i in np.where((s_tip_arr >= R_C_PRIMARY[0]) & (s_tip_arr <= R_C_PRIMARY[1]))[0])
    return sorted(strided | region_c)


def load_series_npz():
    data = np.load(os.path.join(OUT_DIR, "phase1_2_series.npz"))
    out = {}
    for cond, rep, _ in reg.all_runs():
        key = f"{cond}_rep{rep}"
        out[key] = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith(key + "__")}
    return out


def main():
    t0 = time.time()
    series = load_series_npz()

    jac_provider = build_corrected_jacobian_provider()
    adapter = jac_provider.adapter
    baseline_snapshot = adapter.beam_output_fn.capture_cache_state()
    contact_tables = pg.load_path_tables(reg.PLAN_ROOT_CONTACT)

    def solve_J4(state7):
        adapter.forward_output(state7, commit=True)
        return position_jacobian_from_output(adapter.continuous_output_jacobian(state7))

    def J4_at_measured(q_meas, L_meas, s_tip_anchor):
        adapter.beam_output_fn.restore_cache_state(baseline_snapshot)
        q_anchor = pg.q_plan(s_tip_anchor, contact_tables)
        L_anchor = pg.L_plan(s_tip_anchor, contact_tables)
        _ = solve_J4(np.concatenate([q_anchor, [L_anchor]]))
        J = solve_J4(np.concatenate([np.asarray(q_meas, dtype=float), [float(L_meas)]]))
        adapter.beam_output_fn.restore_cache_state(baseline_snapshot)
        return J

    hybrid_model = ContactJacobianHybridModel(HybridMapConfig(map_dir=reg.HYBRID_MAP_DIR), D=D7)

    results = {}
    n_total_solves = 0
    for cond, rep, run_path in reg.all_runs():
        key = f"{cond}_rep{rep}"
        rows = reg.load_path_follow(run_path)
        s_tip_arr = series[key]["s_tip_m"]
        idxs = sample_tick_indices(len(rows), s_tip_arr)
        print(f"[{key}] sampling {len(idxs)}/{len(rows)} ticks", flush=True)

        E_JD = np.full(len(rows), np.nan)
        for k in idxs:
            r = rows[k]
            q_meas = np.asarray(r["q_meas_rad"], dtype=float)
            L_meas = float(r["insertion_length_m"])
            ref_idx = int(r["ref_index"])

            J4 = J4_at_measured(q_meas, L_meas, float(s_tip_arr[k]))
            n_total_solves += 1

            if cond == "NEW":
                sched_idx = min(ref_idx, SCHED_C.shape[0] - 1)
                J1 = SCHED_C[sched_idx]
                s_q = hybrid_model.project_s_q(q_meas)
                dL = L_meas - hybrid_model.L_plan(s_q)
                r_q = hybrid_model.r_q(q_meas, s_q)
                w = hybrid_model.smoothstep_weight(r_q, hybrid_model.r_a, hybrid_model.r_b)
                if w > 0.0:
                    J3, _, valid3, _ = hybrid_model.lookup_j3(s_q, dL)
                else:
                    J3, valid3 = None, False
                J_used = (w * J3 + (1 - w) * J1) if (w > 0.0 and valid3) else J1
            elif cond == "OLD-C":
                sched_idx = min(ref_idx, SCHED_C.shape[0] - 1)
                J_used = SCHED_C[sched_idx]
            else:  # NC
                sched_idx = min(ref_idx, SCHED_NC.shape[0] - 1)
                J_used = SCHED_NC[sched_idx]

            E_JD[k] = float(np.linalg.norm((J4 - J_used) @ np.diag(D7)))

        results[key] = dict(sampled_idx=idxs, E_JD=E_JD.tolist(), s_tip_m=s_tip_arr.tolist())
        print(f"[{key}] done, elapsed={time.time()-t0:.1f}s total_solves={n_total_solves}", flush=True)

    with open(os.path.join(OUT_DIR, "phase4_e_jd_results.json"), "w") as f:
        json.dump(results, f)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = dict(NEW="tab:blue", **{"OLD-C": "tab:orange"}, NC="tab:green")
    for cond, rep, _ in reg.all_runs():
        key = f"{cond}_rep{rep}"
        d = results[key]
        s_mm = np.asarray(d["s_tip_m"])[d["sampled_idx"]] * 1000
        e = np.asarray(d["E_JD"])[d["sampled_idx"]]
        ax.plot(s_mm, e, '.', ms=3, color=colors[cond], alpha=0.6,
                 label=f"{cond}" if rep == 1 else None)
    ax.axvspan(R_C_PRIMARY[0] * 1000, R_C_PRIMARY[1] * 1000, color='gray', alpha=0.15, label="region C")
    ax.set_xlabel("physical progress s_tip (mm)")
    ax.set_ylabel("E_J,D  (D-normalized Jacobian-mismatch norm)")
    ax.set_title("Model fidelity vs physical progress, all 15 runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase4_e_jd_vs_progress.png"), dpi=130)

    print("\nSummary (mean E_J,D by condition, sampled ticks only):")
    for cond in ("NEW", "OLD-C", "NC"):
        vals = []
        for rep, _ in reg.RUNS[cond]:
            d = results[f"{cond}_rep{rep}"]
            e = np.asarray(d["E_JD"])[d["sampled_idx"]]
            vals.extend(e[~np.isnan(e)].tolist())
        print(f"  {cond:6s}: mean={np.mean(vals):.5f}  median={np.median(vals):.5f}  n={len(vals)}")

    print(f"\nTotal J4 solves: {n_total_solves}, wall time: {time.time()-t0:.1f}s")
    print(f"Wrote {OUT_DIR}/phase4_e_jd_results.json and phase4_e_jd_vs_progress.png")


if __name__ == "__main__":
    main()
