"""H2 fix (review issues #2 and #3).

Issue #2: the existing h2_same_state_counterfactual.py's "J_C" counterfactual
substitutes J_C^state(x_k) -- a SINGLE matrix, the contact model recomputed
once at the exact measured state -- for every horizon-stage slot
(confirmed by reading mpc_same_state_replay.solve_same_state: J_override is
broadcast-assigned into controller.reference_position_jacobians[future_idx],
and future_idx has multiple entries for horizon>1, so a (3,7) J_override is
the same matrix repeated across the whole horizon window). This is a
"state-recomputed model diagnostic", NOT the literal contact-schedule window
the real contact-aware MPC would have used at this exact state (whose
per-horizon-step entries vary across the window). This script adds that
second, genuinely "controller-realistic" counterfactual: the literal
window sched_C[future_idx] (an array, not a single matrix) gathered from the
real repaired contact schedule, substituted the same way.

Issue #3: the old c_parallel = e_hat . J_C^state @ dchi metric (e_hat from
the CURRENT tip error p_ref,k - p_tip,k) silently discards sign in exactly
the way the review flagged for H5 -- it is kept here ONLY for side-by-side
comparison, not as the reported result. The new metric is a SIGNED one-step
predicted error reduction using the NEXT reference sample (closer to what a
delay-aware controller's command is actually trying to achieve one tick from
now, though still not a full delay/horizon-propagated cost -- that
remains a known limitation, not resolved by this script):

    delta_e = ||e_before|| - ||e_after||
    e_before = p_ref[k+1] - p_tip[k]
    e_after  = p_ref[k+1] - (p_tip[k] + J_C^state(x_k) @ dchi * dt)

evaluated through J_C^state in every case (actual J_NC command, J_C^state-
repeat counterfactual, J_C^sched-window counterfactual), so the three are
exactly comparable. delta_e > 0 means genuine improvement; delta_e < 0 means
the command would have made it worse.

Read-only against hardware logs and proper_research/** source; writes only
h2_signed_and_schedule_counterfactual.csv and the companion figure.
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

import manifest, loader, live_jac
import mpc_same_state_replay as rep

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
DT = 0.1
MARKS_MM = [45, 50, 55, 58, 60, 62, 64]

RUNS = [
    "mpc_closedloop_nocontact_floor255_20261007T183458Z",
    "mpc_closedloop_nocontact_floor255_v2_20261007T183732Z",
    "mpc_closedloop_nocontact_floor255_v3_20261007T183945Z",
]

SCHED_C_PATH = rep.SCHED_CANDIDATES[True][0]  # repaired contact schedule, same file solve_same_state would use
SCHED_C = np.load(SCHED_C_PATH)


def nearest_tick(s_ref_mm, target, max_n):
    idx = int(np.argmin(np.abs(s_ref_mm[:max_n] - target)))
    if abs(s_ref_mm[idx] - target) > 3.0:
        return None
    return idx


def delta_e(p_ref_next, p_tip, J, dchi):
    e_before = p_ref_next - p_tip
    e_after = p_ref_next - (p_tip + J @ dchi)
    return float(np.linalg.norm(e_before) - np.linalg.norm(e_after))


def main():
    rows = []
    for dirname in RUNS:
        rm = next(r for r in manifest.RUNS if r["dirname"] == dirname)
        out, meta = loader.enrich_run(rm)
        with open(f"{manifest.BASE}/{dirname}/controller_metadata.json") as f:
            cm = json.load(f)
        s_ref_mm = out["s_ref_mm"]
        n = len(s_ref_mm)

        controller, sched_path, tick_log = rep.build_controller(dirname)
        ref = loader.get_reference(cm["plan_dir"])
        desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)
        n_ref = desired_position_m.shape[0]
        print(f"[{dirname}] n_ticks={n} schedule={sched_path.split('/')[-1]}")

        for s_mark in MARKS_MM:
            k = nearest_tick(s_ref_mm, s_mark, n - 1)
            if k is None or k >= len(tick_log) - 1:
                rows.append(dict(dirname=dirname, rep=rm["rep"], s_mark_mm=s_mark, observed=False))
                continue
            row = tick_log[k]
            z_meas = np.asarray(row["z_meas"], dtype=float)
            control_index = int(row["ref_index"])
            J_C_state = live_jac.live_jacobian(z_meas, True)

            # --- (1) actual J_NC command, no override ---
            rep.replay_residual_state(controller, tick_log, k)
            step_NC = rep.solve_same_state(controller, tick_log, k, J_override=None)
            u0_logged = np.asarray(tick_log[k]["u0"], dtype=float)
            sanity_rel = float(np.linalg.norm(step_NC.command - u0_logged)) / max(float(np.linalg.norm(u0_logged)), 1e-9)

            # --- (2) J_C^state repeated across the whole horizon (existing "diagnostic" c/f) ---
            rep.replay_residual_state(controller, tick_log, k)
            step_Cstate = rep.solve_same_state(controller, tick_log, k, J_override=J_C_state)

            # --- (3) NEW: literal contact-schedule window (the "controller-realistic" c/f) ---
            future_idx = controller._reference_indices(control_index, future=True)
            J_sched_window = SCHED_C[np.clip(future_idx, 0, SCHED_C.shape[0] - 1)]
            rep.replay_residual_state(controller, tick_log, k)
            step_Csched = rep.solve_same_state(controller, tick_log, k, J_override=J_sched_window)

            u_NC, u_Cstate, u_Csched = step_NC.command.copy(), step_Cstate.command.copy(), step_Csched.command.copy()
            dchi_NC, dchi_Cstate, dchi_Csched = u_NC * DT, u_Cstate * DT, u_Csched * DT

            cmd_diff_Cstate_vs_Csched = float(np.linalg.norm(u_Cstate - u_Csched))
            cmd_diff_NC_vs_Cstate = float(np.linalg.norm(u_NC - u_Cstate))
            cmd_diff_NC_vs_Csched = float(np.linalg.norm(u_NC - u_Csched))

            p_tip = np.asarray(tick_log[k]["measured_beam_position_m"], dtype=float)
            p_ref_now = desired_position_m[min(control_index, n_ref - 1)]
            p_ref_next = desired_position_m[min(control_index + 1, n_ref - 1)]
            e_vec_now = p_ref_now - p_tip
            e_now_norm = float(np.linalg.norm(e_vec_now))
            e_hat_now = e_vec_now / max(e_now_norm, 1e-12)

            # old (sign-discarded-in-text) metric, kept ONLY for side-by-side comparison
            c_par_old_NC = float(e_hat_now @ (J_C_state @ dchi_NC)) * 1e3
            c_par_old_Cstate = float(e_hat_now @ (J_C_state @ dchi_Cstate)) * 1e3
            c_par_old_Csched = float(e_hat_now @ (J_C_state @ dchi_Csched)) * 1e3

            # new signed one-step error-reduction metric, all evaluated through J_C^state, next-reference target
            de_NC = delta_e(p_ref_next, p_tip, J_C_state, dchi_NC) * 1e3
            de_Cstate = delta_e(p_ref_next, p_tip, J_C_state, dchi_Cstate) * 1e3
            de_Csched = delta_e(p_ref_next, p_tip, J_C_state, dchi_Csched) * 1e3

            rows.append(dict(
                dirname=dirname, rep=rm["rep"], s_mark_mm=s_mark, observed=True, tick=k,
                e_norm_now_mm=e_now_norm * 1e3, sanity_rel_err_NC=sanity_rel,
                cmd_diff_NC_vs_Cstate=cmd_diff_NC_vs_Cstate, cmd_diff_NC_vs_Csched=cmd_diff_NC_vs_Csched,
                cmd_diff_Cstate_vs_Csched=cmd_diff_Cstate_vs_Csched,
                c_par_old_NC_mm=c_par_old_NC, c_par_old_Cstate_mm=c_par_old_Cstate, c_par_old_Csched_mm=c_par_old_Csched,
                delta_e_NC_mm=de_NC, delta_e_Cstate_mm=de_Cstate, delta_e_Csched_mm=de_Csched,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/tables/h2_signed_and_schedule_counterfactual.csv", index=False)
    print(f"\nwrote {OUT}/tables/h2_signed_and_schedule_counterfactual.csv ({len(df)} rows)")

    obs = df[df.observed]
    pd.set_option("display.width", 220)
    print("\n--- full per-row table ---")
    print(obs[["dirname", "rep", "s_mark_mm", "e_norm_now_mm",
               "cmd_diff_Cstate_vs_Csched", "c_par_old_NC_mm", "c_par_old_Cstate_mm", "c_par_old_Csched_mm",
               "delta_e_NC_mm", "delta_e_Cstate_mm", "delta_e_Csched_mm"]].to_string(index=False))

    print("\n--- per-mark mean over reps ---")
    summary = obs.groupby("s_mark_mm")[["cmd_diff_Cstate_vs_Csched", "cmd_diff_NC_vs_Cstate", "cmd_diff_NC_vs_Csched",
                                          "delta_e_NC_mm", "delta_e_Cstate_mm", "delta_e_Csched_mm"]].mean()
    print(summary.to_string())

    n_NC_negative = int((obs.delta_e_NC_mm < 0).sum())
    n_Cstate_negative = int((obs.delta_e_Cstate_mm < 0).sum())
    n_Csched_negative = int((obs.delta_e_Csched_mm < 0).sum())
    print(f"\nnegative delta_e counts (out of {len(obs)}): "
          f"NC-actual={n_NC_negative}, C-state-repeat c/f={n_Cstate_negative}, C-sched-window c/f={n_Csched_negative}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"NC": "#b3331d", "Cstate": "#1b7f3b", "Csched": "#1b4f8f"}
    for rname, g in obs.groupby("dirname"):
        g = g.sort_values("s_mark_mm")
        lab = rname == RUNS[0]
        axes[0].plot(g.s_mark_mm, g.delta_e_NC_mm, "o-", color=colors["NC"], alpha=0.7, label="actual $J_{NC}$" if lab else None)
        axes[0].plot(g.s_mark_mm, g.delta_e_Cstate_mm, "s-", color=colors["Cstate"], alpha=0.7, label="$J_C^{state}$-repeat c/f" if lab else None)
        axes[0].plot(g.s_mark_mm, g.delta_e_Csched_mm, "^-", color=colors["Csched"], alpha=0.7, label="$J_C^{sched}$-window c/f" if lab else None)
        axes[1].plot(g.s_mark_mm, g.cmd_diff_Cstate_vs_Csched, "d-", color="#444", alpha=0.7)
        axes[2].plot(g.s_mark_mm, g.c_par_old_NC_mm, "o--", color=colors["NC"], alpha=0.4)
        axes[2].plot(g.s_mark_mm, g.delta_e_NC_mm, "o-", color=colors["NC"], alpha=0.9,
                     label="actual $J_{NC}$: old $c_\\parallel$ (dashed) vs new $\\Delta e$ (solid)" if True else None)
    axes[0].axhline(0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel(r"$\Delta e$ = signed one-step error reduction (mm)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel(r"$\|u_0^{C,state\text{-}repeat}-u_0^{C,sched\text{-}window}\|$")
    axes[1].set_title("does the literal-schedule-window c/f\ndiffer from the state-repeated c/f?", fontsize=9)
    axes[2].axhline(0, color="gray", lw=0.8, ls="--")
    axes[2].set_ylabel("mm")
    axes[2].set_title("old $c_\\parallel$ metric (dashed) vs\nnew signed $\\Delta e$ (solid), actual $J_{NC}$ command", fontsize=9)
    for ax in axes:
        ax.set_xlabel("path progress s (mm)")
        ax.grid(alpha=0.3)
    fig.suptitle("H2 fix: literal-schedule-window counterfactual + signed one-step error-reduction metric", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fpath = f"{OUT}/figures/h2_signed_and_schedule_counterfactual.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"saved {fpath}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
