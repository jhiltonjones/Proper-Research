"""Script 1 (hard gate) -- plan: snug-bouncing-giraffe.md, n=5 round.

Matched-comparison manifest across the 15 floor240 hardware runs
(5 NEW hybrid_j1j3, 5 OLD-C schedule_compare, 5 NC no-contact), plus a
VERIFICATION gate (not a diagnosis -- the earlier n=3 round already
diagnosed and fixed the timing regression) that:
  (a) NEW's tick_dt/solver_time/realization-ratio are now within a
      PREDEFINED practical-equivalence tolerance of OLD-C's (chosen
      here, before looking at these 15 runs' numbers);
  (b) qp_status (solved / solved inaccurate / run time limit reached)
      is comparable by condition and repetition -- zero deadline_miss
      alone is necessary but not sufficient;
  (c) the new in-QP TCP-workspace constraint was configured
      identically across all three conditions.

Writes manifest.json, control_step_characterization.json, and
timing_gate_result.json. HARD GATE: later scripts should not be
trusted causally until this script's gate passes.
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

# ---- predefined practical-equivalence tolerances (chosen now, before
# inspecting these 15 runs' results) -----------------------------------
DELTA_T_TICK_DT_MS = 10.0        # |median(T_tick^NEW) - median(T_tick^OLD-C)| must be < this
DELTA_T_SOLVER_MS = 10.0         # same form, solver time
DELTA_T_REALIZATION = 0.05       # same form, velocity-realization ratio (absolute, e.g. 0.93 vs 0.97)
QP_TIME_LIMIT_FRACTION_MAX_EXCESS = 0.05  # NEW's time-limited-QP fraction must not exceed OLD-C's by more than this (absolute)


def build_manifest_entry(condition, rep, run_path):
    summary = reg.load_summary(run_path)
    meta = reg.load_controller_metadata(run_path)
    rows = reg.load_path_follow(run_path)
    cfg = summary["config"]

    entry = dict(
        condition=condition, rep=rep, run_dir=os.path.basename(run_path),
        plan_dir=cfg["plan_dir"],
        stop_reason=summary["stop_reason"],
        control_steps=summary["control_steps"],
        reference_samples=summary["reference_samples"],
        final_error_mm=summary["final_error_mm"],
        max_error_mm=summary["max_error_mm"],
        jacobian_condition_at_start=summary.get("jacobian_condition"),
        input_tracking_weight=meta["input_tracking_weight"],
        input_increment_weight=meta["input_increment_weight"],
        horizon=meta["horizon"],
        delay_samples=meta["delay_samples"],
        beta_d=meta["beta_d"],
        dt_s=reg.DT,
        q0_rad=rows[0]["q_meas_rad"],
        L0_m=rows[0]["insertion_length_m"],
        n_path_follow_rows=len(rows),
    )
    if condition == "NEW":
        entry["jacobian_source"] = "hybrid_j1j3"
        entry["map_dir"] = reg.HYBRID_MAP_DIR
        entry["r_a"] = reg.R_A
        entry["r_b"] = reg.R_B
    elif condition == "OLD-C":
        entry["jacobian_source"] = "schedule_j1_only (plain scheduled contact Jacobian, no live hybrid)"
    else:
        entry["jacobian_source"] = "schedule_nc (no-contact schedule)"
    return entry


def qp_status_breakdown(rows):
    counts = {}
    for r in rows:
        st = r.get("qp_status")
        counts[st] = counts.get(st, 0) + 1
    n = len(rows)
    return {k: v / n for k, v in counts.items()}, counts


def characterize_run(condition, rep, run_path):
    rows = reg.load_path_follow(run_path)
    summary = reg.load_summary(run_path)
    plan_root = reg.plan_root_for(condition)
    tables = pg.load_path_tables(plan_root)
    s_of_ref_index = pg.load_resampled_s_of_ref_index(plan_root)

    t_s = np.array([r["t_s"] for r in rows], dtype=float)
    ref_index = np.array([r["ref_index"] for r in rows], dtype=int)
    tip_m = np.array([r["tip_mm"] for r in rows], dtype=float) / 1000.0
    solver_time_s = np.array([r["solver_time_s"] if r["solver_time_s"] is not None else np.nan for r in rows])
    solver_iters = np.array([r["solver_iterations"] if r["solver_iterations"] is not None else np.nan for r in rows])
    q = np.array([r["q_meas_rad"] for r in rows], dtype=float)
    u0 = np.array([r["u0"] for r in rows], dtype=float)
    servo_ms = np.array([r["servo_ms"] for r in rows], dtype=float)

    T_run = float(t_s[-1] - t_s[0])
    tick_dt = np.diff(t_s)

    s_ref = s_of_ref_index[np.clip(ref_index, 0, len(s_of_ref_index) - 1)]
    dt = np.diff(t_s)
    ds_ref = np.diff(s_ref)
    valid = dt > 1e-6
    mean_ds_ref_dt = float(np.mean(ds_ref[valid] / dt[valid]))

    s_tip = np.array([pg.project_s_cartesian(tip_m[k], tables) for k in range(len(rows))])
    ds_tip = np.diff(s_tip)
    mean_ds_tip_dt = float(np.mean(ds_tip[valid] / dt[valid]))

    # velocity-realization ratio: achieved/commanded joint velocity, meaningful ticks only
    dq_real = np.diff(q, axis=0)
    v_real = np.linalg.norm(dq_real, axis=1) / dt
    v_cmd = np.linalg.norm(u0[:-1, :6], axis=1)
    meaningful = v_cmd > 0.01
    realization_ratio = float(np.mean(v_real[meaningful] / v_cmd[meaningful])) if meaningful.any() else float("nan")

    qp_frac, qp_counts = qp_status_breakdown(rows)
    delta_ref_index = np.diff(ref_index)

    return dict(
        condition=condition, rep=rep,
        T_run_s=T_run, n_ticks=len(rows),
        mean_tick_dt_ms=float(np.mean(tick_dt) * 1000.0),
        median_tick_dt_ms=float(np.median(tick_dt) * 1000.0),
        mean_solver_time_ms=float(np.nanmean(solver_time_s) * 1000.0),
        median_solver_time_ms=float(np.nanmedian(solver_time_s) * 1000.0),
        mean_solver_iterations=float(np.nanmean(solver_iters)),
        mean_servo_ms=float(np.mean(servo_ms)),
        mean_ds_ref_dt_m_s=mean_ds_ref_dt, mean_ds_tip_dt_m_s=mean_ds_tip_dt,
        realization_ratio=realization_ratio,
        qp_status_fraction=qp_frac, qp_status_counts=qp_counts,
        n_deadline_miss=int(sum(1 for r in rows if r.get("solver_status") == "deadline_miss")),
        delta_ref_index_mean=float(np.mean(delta_ref_index)),
        control_steps=summary["control_steps"], stop_reason=summary["stop_reason"],
    )


def check_tcp_workspace_identical():
    """The TCP-workspace constraint's bounds/margin/live_relinearize are
    NOT recorded in controller_metadata.json (confirmed by inspection --
    a gap in the original logging addition, not fixed retroactively
    here per this plan's "state the limitation, don't invent a number"
    rule). We therefore verify this indirectly, from what IS knowable:
    all 15 of these runs' hardware commands (as given to/confirmed by
    the user) omitted --disable-tcp-workspace-in-qp, --tcp-workspace-
    live-relinearize, and any --tcp-workspace-robust-margin-mm
    override, and the box itself (_WORKSPACE_XYZ_MIN_M/MAX_M in
    run_mpc_delay_aware_vessel.py) is a shared module-level constant,
    not CLI-overridable per condition. This is a real limitation, not
    a verified-from-artifacts pass -- reported as such.
    """
    return dict(
        verified_from_artifacts=False,
        reason="controller_metadata.json does not record tcp_workspace bounds/margin/live_relinearize "
               "-- confirmed by inspection, a gap in the original logging addition, not fixed here.",
        indirect_evidence="all 15 commands omitted --disable-tcp-workspace-in-qp, --tcp-workspace-"
               "live-relinearize, and any --tcp-workspace-robust-margin-mm override; the box is a "
               "shared module-level constant (_WORKSPACE_XYZ_MIN_M/MAX_M), not per-condition-settable.",
        limitation="this check is NOT a verified-from-artifacts pass -- it is the best available "
               "indirect evidence. A follow-up should add tcp_workspace bounds/margin to "
               "controller_metadata.json so future rounds can verify this directly.",
    )


def main():
    manifest = [build_manifest_entry(cond, rep, path) for cond, rep, path in reg.all_runs()]
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    all_char = [characterize_run(c, r, p) for c, r, p in reg.all_runs()]
    with open(os.path.join(OUT_DIR, "control_step_characterization.json"), "w") as f:
        json.dump(all_char, f, indent=2)

    print("=" * 110)
    print("MANIFEST (15 runs)")
    print("=" * 110)
    for e in manifest:
        print(f"{e['condition']:6s} rep{e['rep']} {e['run_dir']:70s} "
              f"stop={e['stop_reason']:14s} steps={e['control_steps']:4d} ref_n={e['reference_samples']:4d}")

    print()
    print("=" * 110)
    print("TIMING / SOLVER-STATUS CHARACTERIZATION (all 15 runs)")
    print("=" * 110)
    print(f"{'cond':6s} {'rep':4s} {'steps':6s} {'tick_dt_ms':11s} {'solver_ms':10s} "
          f"{'realization':11s} {'%solved':9s} {'%inaccurate':12s} {'%time_limit':12s} {'n_dl_miss':10s}")
    for c in all_char:
        qp = c["qp_status_fraction"]
        print(f"{c['condition']:6s} {c['rep']:<4d} {c['control_steps']:<6d} "
              f"{c['mean_tick_dt_ms']:<11.1f} {c['mean_solver_time_ms']:<10.2f} "
              f"{c['realization_ratio']:<11.3f} "
              f"{qp.get('solved', 0)*100:<9.1f} {qp.get('solved inaccurate', 0)*100:<12.1f} "
              f"{qp.get('run time limit reached', 0)*100:<12.1f} {c['n_deadline_miss']:<10d}")

    def cond_vals(cond, key):
        return [c[key] for c in all_char if c["condition"] == cond]

    print()
    print("=" * 110)
    print("HARD GATE 1: TIMING / SOLVER-STATUS PRACTICAL EQUIVALENCE (predefined tolerances, not fit to results)")
    print("=" * 110)
    new_tick = np.median(cond_vals("NEW", "mean_tick_dt_ms"))
    old_tick = np.median(cond_vals("OLD-C", "mean_tick_dt_ms"))
    new_solver = np.median(cond_vals("NEW", "mean_solver_time_ms"))
    old_solver = np.median(cond_vals("OLD-C", "mean_solver_time_ms"))
    new_real = np.median(cond_vals("NEW", "realization_ratio"))
    old_real = np.median(cond_vals("OLD-C", "realization_ratio"))
    new_tl_frac = np.median([c["qp_status_fraction"].get("run time limit reached", 0.0) for c in all_char if c["condition"] == "NEW"])
    old_tl_frac = np.median([c["qp_status_fraction"].get("run time limit reached", 0.0) for c in all_char if c["condition"] == "OLD-C"])

    diff_tick = abs(new_tick - old_tick)
    diff_solver = abs(new_solver - old_solver)
    diff_real = abs(new_real - old_real)
    diff_tl = new_tl_frac - old_tl_frac

    pass_tick = diff_tick < DELTA_T_TICK_DT_MS
    pass_solver = diff_solver < DELTA_T_SOLVER_MS
    pass_real = diff_real < DELTA_T_REALIZATION
    pass_tl = diff_tl < QP_TIME_LIMIT_FRACTION_MAX_EXCESS

    print(f"median tick_dt:       NEW={new_tick:.2f}ms  OLD-C={old_tick:.2f}ms  "
          f"|diff|={diff_tick:.2f}ms  tolerance={DELTA_T_TICK_DT_MS}ms  {'PASS' if pass_tick else 'FAIL'}")
    print(f"median solver_time:   NEW={new_solver:.2f}ms  OLD-C={old_solver:.2f}ms  "
          f"|diff|={diff_solver:.2f}ms  tolerance={DELTA_T_SOLVER_MS}ms  {'PASS' if pass_solver else 'FAIL'}")
    print(f"median realization:   NEW={new_real:.3f}  OLD-C={old_real:.3f}  "
          f"|diff|={diff_real:.3f}  tolerance={DELTA_T_REALIZATION}  {'PASS' if pass_real else 'FAIL'}")
    print(f"median %time-limit:   NEW={new_tl_frac*100:.1f}%  OLD-C={old_tl_frac*100:.1f}%  "
          f"diff={diff_tl*100:+.1f}pp  max_excess={QP_TIME_LIMIT_FRACTION_MAX_EXCESS*100:.0f}pp  "
          f"{'PASS' if pass_tl else 'FAIL'}")

    gate1_pass = pass_tick and pass_solver and pass_real and pass_tl
    print(f"\nGATE 1 (timing/solver equivalence): {'PASS' if gate1_pass else 'FAIL'}")

    print()
    print("=" * 110)
    print("HARD GATE 2: TCP-WORKSPACE CONSTRAINT CONFIGURED IDENTICALLY ACROSS CONDITIONS")
    print("=" * 110)
    tcp_check = check_tcp_workspace_identical()
    print(f"verified_from_artifacts: {tcp_check['verified_from_artifacts']}")
    print(f"reason: {tcp_check['reason']}")
    print(f"indirect_evidence: {tcp_check['indirect_evidence']}")
    print(f"LIMITATION: {tcp_check['limitation']}")

    print()
    print("=" * 110)
    print("ZERO-DEADLINE-MISS CONFIRMATION (necessary, not sufficient on its own)")
    print("=" * 110)
    total_misses = sum(c["n_deadline_miss"] for c in all_char)
    print(f"total deadline_miss across all 15 runs: {total_misses}")

    gate_result = dict(
        gate1_timing_solver_pass=bool(gate1_pass),
        tick_dt=dict(new=float(new_tick), old_c=float(old_tick), diff=float(diff_tick), tol=DELTA_T_TICK_DT_MS, passed=bool(pass_tick)),
        solver_time=dict(new=float(new_solver), old_c=float(old_solver), diff=float(diff_solver), tol=DELTA_T_SOLVER_MS, passed=bool(pass_solver)),
        realization=dict(new=float(new_real), old_c=float(old_real), diff=float(diff_real), tol=DELTA_T_REALIZATION, passed=bool(pass_real)),
        time_limit_fraction=dict(new=float(new_tl_frac), old_c=float(old_tl_frac), diff=float(diff_tl), max_excess=QP_TIME_LIMIT_FRACTION_MAX_EXCESS, passed=bool(pass_tl)),
        tcp_workspace_check=tcp_check,
        total_deadline_misses=total_misses,
    )
    with open(os.path.join(OUT_DIR, "timing_gate_result.json"), "w") as f:
        json.dump(gate_result, f, indent=2)

    print()
    if gate1_pass and total_misses == 0:
        print("OVERALL GATE: PASS (with the stated TCP-workspace-config limitation) -- "
              "proceeding to causal interpretation in later scripts is justified.")
    else:
        print("OVERALL GATE: FAIL -- DO NOT proceed to causal NEW-vs-OLD-C claims until this is resolved.")

    print(f"\nWrote {OUT_DIR}/manifest.json, control_step_characterization.json, timing_gate_result.json")


if __name__ == "__main__":
    main()
