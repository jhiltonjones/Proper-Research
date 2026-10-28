"""Final closure check on the s~57-61mm contact-aware MPC transient
(STAGE3_FINAL_REPORT_REVISED.md Sec 9).

The contact-state recomputation (1D) in transient_57mm_part2.py used a coarse
stride (N_1D_SAMPLES_PER_REP=8 over the s=53-61mm window) and did not always
land exactly on the true per-rep peak tick. transient_peak_refine.py already
closed that specific gap -- it recomputes gap_min_m/contact_force_norm/W_cf
at the EXACT peak tick (and its immediate +/-3-tick neighbours) for all 3
reps; see tables/transient_peak_refine.csv, is_peak==True rows. This script
adds the one thing that table does NOT have: a same-state MPC replay at
those exact 3 peak ticks, swapping the scheduled Jacobian J_C^sched (the
real baseline -- no override) for J_C^state(x_k) (the contact model
recomputed at that exact measured state) and comparing the resulting
command. If the command barely changes, a schedule/state mismatch specific
to this exact tick is definitively ruled out as a contributor.

Read-only with respect to proper_research/** and all hardware logs; writes
only tables/transient_exact_peak_closure.csv.
"""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis/scripts")
import numpy as np
import pandas as pd

import manifest
import live_jac
import mpc_same_state_replay as msr

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"

REP_TO_DIRNAME = {
    2: "mpc_closedloop_contact_floor210_baseline_v2_20261007T180726Z",
    3: "mpc_closedloop_contact_floor210_baseline_v3_20261007T181008Z",
    4: "mpc_closedloop_contact_floor210_baseline_v4_20261008T120734Z",
}

peak_df = pd.read_csv(f"{OUT}/tables/transient_peak_refine.csv")
peaks = peak_df[peak_df.is_peak == True]
print(f"[closure] {len(peaks)} exact peak ticks: "
      f"{list(zip(peaks.rep, peaks.step, peaks.s_mm))}")

rows = []
for _, prow in peaks.iterrows():
    rep = int(prow["rep"])
    dirname = REP_TO_DIRNAME[rep]
    controller, sched_path, tick_log = msr.build_controller(dirname)

    # tick_log is indexed by control step starting at its own tick 0; the
    # peak table's "step" field comes from loader.enrich_run's out["step"],
    # confirmed elsewhere in this investigation to be the same 1-based/0-based
    # control-tick counter predicted_beam_positions.jsonl uses -- verify by
    # matching s_mm directly rather than assuming the index convention.
    target_s_mm = float(prow["s_mm"])
    import loader
    rm = next(r for r in manifest.RUNS if r["dirname"] == dirname)
    out, meta = loader.enrich_run(rm)
    s_arr = out["s_ref_mm"]
    # out["step"] and tick_log/predicted_beam_positions.jsonl are both
    # 0-indexed per-tick logs written by the same control loop; verify the
    # direct index match holds (rather than assume it) before trusting it.
    k_direct = int(prow["step"])
    if k_direct < len(tick_log) and abs(float(s_arr[k_direct]) - target_s_mm) < 0.01:
        k = k_direct
    else:
        k = int(np.argmin(np.abs(s_arr - target_s_mm)))
    print(f"  rep{rep}: using tick_log index k={k} (s_mm target={target_s_mm:.3f}, "
          f"resolved out[s_ref_mm][k]={float(s_arr[k]):.3f}, direct-index-matched={k==k_direct})")

    msr.replay_residual_state(controller, tick_log, k)

    step_sched = msr.solve_same_state(controller, tick_log, k, J_override=None)
    u0_sched = np.asarray(step_sched.command, dtype=float)
    u0_logged = np.asarray(tick_log[k]["u0"], dtype=float)
    sanity_err = float(np.linalg.norm(u0_sched - u0_logged) / max(np.linalg.norm(u0_logged), 1e-9))

    q = np.asarray(tick_log[k]["q_cmd_k"], dtype=float)
    L = float(tick_log[k]["insertion_cmd_m"])
    state7 = np.concatenate([q, [L]])
    J_C_state = live_jac.live_jacobian(state7, True)

    step_state = msr.solve_same_state(controller, tick_log, k, J_override=J_C_state)
    u0_state = np.asarray(step_state.command, dtype=float)

    cmd_diff_norm = float(np.linalg.norm(u0_sched - u0_state))
    cmd_diff_rel = float(cmd_diff_norm / max(np.linalg.norm(u0_sched), 1e-9))

    peak_row = peak_df[(peak_df.rep == rep) & (peak_df.step == int(prow["step"]))].iloc[0]

    rows.append(dict(
        rep=rep, step=int(prow["step"]), s_mm=target_s_mm,
        e_track_mm=float(prow["e_track_mm"]),
        gap_min_m=float(peak_row["gap_min_m"]), contact_force_norm=float(peak_row["contact_force_norm"]),
        W_cf=float(peak_row["W_cf"]),
        tick_log_k_used=k, sanity_rel_err_u0_sched_vs_logged=sanity_err,
        u0_sched_norm=float(np.linalg.norm(u0_sched)), u0_state_norm=float(np.linalg.norm(u0_state)),
        cmd_diff_norm=cmd_diff_norm, cmd_diff_rel=cmd_diff_rel,
    ))
    print(f"  rep{rep} s={target_s_mm:.2f}mm: sanity_rel_err={sanity_err:.4%}  "
          f"||u0_sched||={np.linalg.norm(u0_sched):.5f}  ||u0_state||={np.linalg.norm(u0_state):.5f}  "
          f"||diff||={cmd_diff_norm:.3e}  rel_diff={cmd_diff_rel:.3%}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/tables/transient_exact_peak_closure.csv", index=False)
print(f"\n[closure] wrote {OUT}/tables/transient_exact_peak_closure.csv")
print(df.to_string(index=False))
