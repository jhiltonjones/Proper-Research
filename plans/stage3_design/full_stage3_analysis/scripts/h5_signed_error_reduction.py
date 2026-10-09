"""H5 fix (review issue #4): replace the sign-discarded useful-correction
metric c_parallel = e_hat^T J Dchi (e_hat = unit vector toward the CURRENT
tip error) with a signed one-step predicted error-reduction metric that
cannot be misread as "more is always better" regardless of sign:

    Delta_e = ||e_before|| - ||e_after||
    e_before = p_ref_next - p_tip_k
    e_after  = p_ref_next - (p_tip_k + J_C^state @ Dchi * dt)

using p_ref_next = the NEXT reference sample (ref_index+1), not the current
one -- the point the system will actually be asked to track one control tick
after executing this command, which is a better (though still not fully
delay/horizon-aware) target for "what is this command trying to achieve"
than the current-tick reference. This is an explicit, honestly-labelled
one-step-ahead approximation: it does NOT propagate the solved trajectory
through the MPC's own delay buffer/horizon to get the true predicted-cost
change (Delta_V_track) the reviewer also floated as the ideal fix -- that
would require re-deriving the controller's internal prediction machinery,
which is out of scope for this pass. Positive Delta_e = genuine improvement
(error norm shrinks); negative = the command actually moves the tip further
from where it needs to be next.

Reuses the IDENTICAL 11 states, 3 commands (idx100-actual, scheduled-
counterfactual, idx0-counterfactual) and same-state replay machinery as
h5_same_state_counterfactual.py -- see that script's docstring for the
full experimental design (why the idx100 run's own states, not the
scheduled baseline's). Also recomputes the OLD c_parallel metric alongside
Delta_e so the two can be compared directly, tick by tick.

Read-only with respect to proper_research/** and all run logs; writes only
tables/h5_signed_error_reduction.csv and figures/h5_signed_error_reduction.png.
"""
import sys, time, json
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis/scripts")
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest, loader
import mpc_same_state_replay as rep

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
DT = 0.1
MARKS_MM = [20, 30, 40, 50, 55, 58, 60, 62, 64, 67, 70]
IDX100_RUN = "mpc_closedloop_contact_frozen_precontact_idx100_20261008T123031Z"
SCHED_REF_RUN = "mpc_closedloop_contact_floor210_baseline_v2_20261007T180726Z"
IDX0, IDX100 = 0, 100


def nearest_tick(s_ref_mm, target, max_n):
    idx = int(np.argmin(np.abs(s_ref_mm[:max_n] - target)))
    if abs(s_ref_mm[idx] - target) > 3.0:
        return None
    return idx


def old_c_parallel(J, dchi, e_hat):
    dp = J @ dchi
    return float(e_hat @ dp)


def signed_delta_e(J, dchi, p_tip, p_ref_next):
    e_before = p_ref_next - p_tip
    p_tip_after = p_tip + J @ dchi
    e_after = p_ref_next - p_tip_after
    return float(np.linalg.norm(e_before) - np.linalg.norm(e_after))


def main():
    rm = next(r for r in manifest.RUNS if r["dirname"] == IDX100_RUN)
    out, meta = loader.enrich_run(rm)
    with open(f"{manifest.BASE}/{IDX100_RUN}/controller_metadata.json") as f:
        cm = json.load(f)
    s_ref_mm = out["s_ref_mm"]
    n = len(s_ref_mm)
    print(f"[{IDX100_RUN}] n_ticks={n} (stop at s_end={s_ref_mm[-1]:.1f}mm)")

    _, full_sched_path, _ = rep.build_controller(SCHED_REF_RUN)
    full_schedule = np.load(full_sched_path)
    J_idx0 = full_schedule[IDX0]
    J_idx100 = full_schedule[IDX100]

    frozen_schedule = np.tile(J_idx100[None, :, :], (full_schedule.shape[0], 1, 1))
    controller, _, tick_log = rep.build_controller(IDX100_RUN, force_schedule=frozen_schedule)

    import live_jac

    ref = loader.get_reference(cm["plan_dir"])
    desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)
    n_ref = desired_position_m.shape[0]

    rows = []
    for s_mark in MARKS_MM:
        k = nearest_tick(s_ref_mm, s_mark, n - 1)
        if k is None or k >= len(tick_log) - 1:
            rows.append(dict(s_mark_mm=s_mark, observed=False))
            continue

        z_meas = np.asarray(tick_log[k]["z_meas"], dtype=float)
        J_C_state = live_jac.live_jacobian(z_meas, True)

        rep.replay_residual_state(controller, tick_log, k)
        step_idx100 = rep.solve_same_state(controller, tick_log, k, J_override=None)
        u0_logged = np.asarray(tick_log[k]["u0"], dtype=float)
        sanity_rel = float(np.linalg.norm(step_idx100.command - u0_logged)) / max(float(np.linalg.norm(u0_logged)), 1e-9)

        ref_idx = tick_log[k]["ref_index"]
        ref_idx_next = min(ref_idx + 1, n_ref - 1)
        J_sched_here = full_schedule[min(ref_idx + 1, full_schedule.shape[0] - 1)]
        rep.replay_residual_state(controller, tick_log, k)
        step_sched = rep.solve_same_state(controller, tick_log, k, J_override=J_sched_here)
        rep.replay_residual_state(controller, tick_log, k)
        step_idx0 = rep.solve_same_state(controller, tick_log, k, J_override=J_idx0)

        u_idx100, u_sched, u_idx0 = step_idx100.command, step_sched.command, step_idx0.command
        dchi_idx100, dchi_sched, dchi_idx0 = u_idx100 * DT, u_sched * DT, u_idx0 * DT

        p_tip = np.asarray(tick_log[k]["measured_beam_position_m"], dtype=float)
        p_ref_cur = desired_position_m[ref_idx]
        p_ref_next = desired_position_m[ref_idx_next]
        e_vec_cur = p_ref_cur - p_tip
        e_hat_cur = e_vec_cur / max(float(np.linalg.norm(e_vec_cur)), 1e-12)

        # OLD metric (sign-discarded in the original report text) -- recomputed
        # here, unmodified, for direct side-by-side comparison.
        cpar_100_old = old_c_parallel(J_C_state, dchi_idx100, e_hat_cur)
        cpar_s_old = old_c_parallel(J_C_state, dchi_sched, e_hat_cur)
        cpar_0_old = old_c_parallel(J_C_state, dchi_idx0, e_hat_cur)

        # NEW signed one-step error-reduction metric.
        de_100 = signed_delta_e(J_C_state, dchi_idx100, p_tip, p_ref_next)
        de_s = signed_delta_e(J_C_state, dchi_sched, p_tip, p_ref_next)
        de_0 = signed_delta_e(J_C_state, dchi_idx0, p_tip, p_ref_next)

        e_before_norm = float(np.linalg.norm(p_ref_next - p_tip))

        rows.append(dict(
            s_mark_mm=s_mark, observed=True, tick=k,
            e_norm_cur_mm=float(np.linalg.norm(e_vec_cur)) * 1e3,
            e_before_next_mm=e_before_norm * 1e3,
            c_parallel_idx100_mm_OLD=cpar_100_old * 1e3,
            c_parallel_sched_mm_OLD=cpar_s_old * 1e3,
            c_parallel_idx0_mm_OLD=cpar_0_old * 1e3,
            delta_e_idx100_mm_NEW=de_100 * 1e3,
            delta_e_sched_mm_NEW=de_s * 1e3,
            delta_e_idx0_mm_NEW=de_0 * 1e3,
            sanity_rel_err=sanity_rel,
        ))
        print(f"  s={s_mark}mm k={k}: OLD c_par idx100/sched/idx0 = "
              f"{cpar_100_old*1e3:+.4f}/{cpar_s_old*1e3:+.4f}/{cpar_0_old*1e3:+.4f} mm | "
              f"NEW Delta_e idx100/sched/idx0 = "
              f"{de_100*1e3:+.4f}/{de_s*1e3:+.4f}/{de_0*1e3:+.4f} mm "
              f"(sanity_rel_err={sanity_rel:.3%})")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/tables/h5_signed_error_reduction.csv", index=False)
    print(f"\nwrote {OUT}/tables/h5_signed_error_reduction.csv ({len(df)} rows)")

    obs = df[df.observed].copy()
    pd.set_option("display.width", 220)
    print("\nFull comparison table (OLD c_parallel vs NEW signed Delta_e):")
    print(obs[["s_mark_mm", "tick", "e_before_next_mm",
               "c_parallel_idx100_mm_OLD", "delta_e_idx100_mm_NEW",
               "c_parallel_sched_mm_OLD", "delta_e_sched_mm_NEW",
               "c_parallel_idx0_mm_OLD", "delta_e_idx0_mm_NEW"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(obs.s_mark_mm, obs.delta_e_idx100_mm_NEW, "^-", color="#1b4f8f", label="idx100 (actual)")
    ax.plot(obs.s_mark_mm, obs.delta_e_sched_mm_NEW, "o-", color="#1b7f3b", label="scheduled (counterfactual)")
    ax.plot(obs.s_mark_mm, obs.delta_e_idx0_mm_NEW, "s-", color="#b3331d", label="idx0 (counterfactual)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.fill_between(obs.s_mark_mm, -10, 0, color="red", alpha=0.05)
    ax.set_ylabel(r"signed one-step error reduction $\Delta e$ (mm), positive = improvement")
    ax.legend(fontsize=8)
    ax.axvspan(58, 68, color="gray", alpha=0.12)
    ax.set_title("NEW metric: $\\Delta e = \\|e_{before}\\| - \\|e_{after}\\|$")

    ax2 = axes[1]
    ax2.plot(obs.s_mark_mm, obs.c_parallel_idx100_mm_OLD, "^--", color="#1b4f8f", alpha=0.7, label="idx100 (actual)")
    ax2.plot(obs.s_mark_mm, obs.c_parallel_sched_mm_OLD, "o--", color="#1b7f3b", alpha=0.7, label="scheduled (counterfactual)")
    ax2.plot(obs.s_mark_mm, obs.c_parallel_idx0_mm_OLD, "s--", color="#b3331d", alpha=0.7, label="idx0 (counterfactual)")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel(r"OLD $c_\parallel$ (mm) -- sign was discarded in the original report text")
    ax2.legend(fontsize=8)
    ax2.axvspan(58, 68, color="gray", alpha=0.12)
    ax2.set_title("OLD metric (superseded): $c_\\parallel=\\hat e_k^T J\\Delta\\chi$")

    for a in axes:
        a.set_xlabel("path progress s (mm)")
        a.grid(alpha=0.3)
    fig.suptitle("H5: signed one-step error-reduction metric vs the superseded sign-discarded projection\n"
                 "(shaded: idx100's reported failure-onset window, report Sec 7.3)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fpath = f"{OUT}/figures/h5_signed_error_reduction.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"saved {fpath}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
