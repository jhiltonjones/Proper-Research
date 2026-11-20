"""Task 6 (H5): same-state scheduled vs frozen-idx0 vs frozen-idx100
counterfactual.

States are taken from the FROZEN-IDX100 run's own hardware log (not the
scheduled baseline) -- deliberately, after an earlier pass using the
scheduled baseline's own (small, noisily-signed) tracking error showed
single-tick useful-correction sign flips too often to give a clean read (the
scheduled controller's own tracking error has no sustained direction to
resolve against). The idx100 run genuinely fails late in the path with a
sustained, growing tracking error -- exactly the regime the paradox
("well-conditioned but eventually fails") needs explaining -- so its own
states give a non-degenerate e_hat (correction direction) to project onto.

At matched ticks spanning s = {20,30,40,50,55,60,65,70}mm of
mpc_closedloop_contact_frozen_precontact_idx100_20261008T123031Z (same floor,
plan, and controller variant as the scheduled baseline and the idx0
ablation -- only the Jacobian differs), solve the exact same MPC three times,
swapping only the Jacobian the QP's horizon window is built from:

  u_0^idx100 = no override (the real frozen-idx100 run, cross-checked
               against its own logged u0)
  u_0^sched  = schedule[ref_index+1] (what the trajectory-varying schedule
               would have supplied at this exact tick)
  u_0^idx0   = schedule[0] (the other frozen ablation's matrix)

All three commands are evaluated through the SAME J_C^state(x_k) (the live
contact model recomputed at the exact measured state), decomposed into
useful correction along e_hat=(p_ref-p_tip)/||p_ref-p_tip|| and
transverse/wasted correction.

Read-only; writes h5_same_state_counterfactual.csv and
h5_same_state_frozen_counterfactual.png only.
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
SCHED_REF_RUN = "mpc_closedloop_contact_floor210_baseline_v2_20261007T180726Z"  # for the full schedule array only
IDX0, IDX100 = 0, 100


def nearest_tick(s_ref_mm, target, max_n):
    idx = int(np.argmin(np.abs(s_ref_mm[:max_n] - target)))
    if abs(s_ref_mm[idx] - target) > 3.0:
        return None
    return idx


def eval_through(J, dchi, e_hat):
    dp = J @ dchi
    c_par = float(e_hat @ dp)
    c_perp = float(np.linalg.norm(dp - c_par * e_hat))
    eta = c_par / max(float(np.linalg.norm(dchi)), 1e-12)
    return dp, c_par, c_perp, eta


def main():
    rm = next(r for r in manifest.RUNS if r["dirname"] == IDX100_RUN)
    out, meta = loader.enrich_run(rm)
    with open(f"{manifest.BASE}/{IDX100_RUN}/controller_metadata.json") as f:
        cm = json.load(f)
    s_ref_mm = out["s_ref_mm"]
    n = len(s_ref_mm)
    print(f"[{IDX100_RUN}] n_ticks={n} (stop at s_end={s_ref_mm[-1]:.1f}mm)")

    # full trajectory-varying schedule, for the "what would the scheduled
    # controller have used right here" counterfactual
    _, full_sched_path, _ = rep.build_controller(SCHED_REF_RUN)
    full_schedule = np.load(full_sched_path)
    J_idx0 = full_schedule[IDX0]
    J_idx100 = full_schedule[IDX100]

    frozen_schedule = np.tile(J_idx100[None, :, :], (full_schedule.shape[0], 1, 1))
    controller, _, tick_log = rep.build_controller(IDX100_RUN, force_schedule=frozen_schedule)

    import live_jac

    rows = []
    sanity_rows = []
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
        sanity_err = float(np.linalg.norm(step_idx100.command - u0_logged))
        sanity_rel = sanity_err / max(float(np.linalg.norm(u0_logged)), 1e-9)
        sanity_rows.append(dict(k=k, s_mark_mm=s_mark, sanity_rel_err=sanity_rel))

        ref_idx = tick_log[k]["ref_index"]
        J_sched_here = full_schedule[min(ref_idx + 1, full_schedule.shape[0] - 1)]
        rep.replay_residual_state(controller, tick_log, k)
        step_sched = rep.solve_same_state(controller, tick_log, k, J_override=J_sched_here)
        rep.replay_residual_state(controller, tick_log, k)
        step_idx0 = rep.solve_same_state(controller, tick_log, k, J_override=J_idx0)

        u_idx100, u_sched, u_idx0 = step_idx100.command, step_sched.command, step_idx0.command
        dchi_idx100, dchi_sched, dchi_idx0 = u_idx100 * DT, u_sched * DT, u_idx0 * DT

        p_tip = np.asarray(tick_log[k]["measured_beam_position_m"], dtype=float)
        ref = loader.get_reference(cm["plan_dir"])
        p_ref = np.asarray(ref["desired_position_m"][ref_idx], dtype=float)
        e_vec = p_ref - p_tip
        e_norm = np.linalg.norm(e_vec)
        e_hat = e_vec / max(e_norm, 1e-12)

        _, cpar_100, cperp_100, eta_100 = eval_through(J_C_state, dchi_idx100, e_hat)
        _, cpar_s, cperp_s, eta_s = eval_through(J_C_state, dchi_sched, e_hat)
        _, cpar_0, cperp_0, eta_0 = eval_through(J_C_state, dchi_idx0, e_hat)

        rows.append(dict(
            s_mark_mm=s_mark, observed=True, tick=k, e_norm_mm=e_norm * 1e3,
            c_parallel_idx100_mm=cpar_100 * 1e3, c_parallel_sched_mm=cpar_s * 1e3, c_parallel_idx0_mm=cpar_0 * 1e3,
            c_perp_idx100_mm=cperp_100 * 1e3, c_perp_sched_mm=cperp_s * 1e3, c_perp_idx0_mm=cperp_0 * 1e3,
            eta_idx100_mm=eta_100 * 1e3, eta_sched_mm=eta_s * 1e3, eta_idx0_mm=eta_0 * 1e3,
            sanity_rel_err=sanity_rel,
        ))
        print(f"  s={s_mark}mm k={k} e={e_norm*1e3:.3f}mm: "
              f"c_par idx100/sched/idx0 = {cpar_100*1e3:.4f}/{cpar_s*1e3:.4f}/{cpar_0*1e3:.4f} mm, "
              f"sanity_rel_err={sanity_rel:.3%}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/tables/h5_same_state_counterfactual.csv", index=False)
    pd.DataFrame(sanity_rows).to_csv(f"{OUT}/tables/h5_same_state_sanity_check.csv", index=False)
    print(f"\nwrote {OUT}/tables/h5_same_state_counterfactual.csv ({len(df)} rows)")

    obs = df[df.observed]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(obs.s_mark_mm, obs.c_parallel_idx100_mm, "^-", color="#1b4f8f", label="idx100 (actual)")
    axes[0].plot(obs.s_mark_mm, obs.c_parallel_sched_mm, "o-", color="#1b7f3b", label="scheduled (counterfactual)")
    axes[0].plot(obs.s_mark_mm, obs.c_parallel_idx0_mm, "s-", color="#b3331d", label="idx0 (counterfactual)")
    axes[0].axhline(0, color="gray", lw=0.6, ls="--")
    axes[0].set_ylabel(r"useful correction $c_\parallel$ (mm), through $J_C^{state}$")
    axes[0].legend(fontsize=8)
    axes[1].plot(obs.s_mark_mm, obs.c_perp_idx100_mm, "^-", color="#1b4f8f")
    axes[1].plot(obs.s_mark_mm, obs.c_perp_sched_mm, "o-", color="#1b7f3b")
    axes[1].plot(obs.s_mark_mm, obs.c_perp_idx0_mm, "s-", color="#b3331d")
    axes[1].set_ylabel("transverse (wasted) correction, $c_\\perp$ (mm)")
    axes[2].plot(obs.s_mark_mm, obs.e_norm_mm, "d-", color="#444")
    axes[2].set_ylabel("actual tracking error at this tick (mm)")
    for ax in axes:
        ax.set_xlabel("path progress s (mm)")
        ax.grid(alpha=0.3)
        ax.axvspan(58, 68, color="gray", alpha=0.12)
    fig.suptitle("H5 same-state counterfactual, evaluated at the FROZEN-idx100 run's own states\n"
                 "(shaded: idx100's reported failure-onset window, report Sec 7.3)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fpath = f"{OUT}/figures/h5_same_state_frozen_counterfactual.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"saved {fpath}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
