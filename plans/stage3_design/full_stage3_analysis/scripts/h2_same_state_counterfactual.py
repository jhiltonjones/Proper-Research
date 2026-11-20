"""Task 2B-2E (H2): same-state MPC counterfactual, J_NC (as actually run) vs
J_C^state (the contact model recomputed at the measured state), at matched
path-progress marks on the tight-floor (255mm enforced 250mm) MPC-J_NC runs.

For each of the 3 primary MPC-J_NC@255mm reps and each mark
s in {45,50,55,58,60,62,64}mm (skipped if the run had already terminated
before reaching it -- all 3 reps abort near s~64mm on the 5mm tracking-error
criterion, confirmed directly from summary.json, not a workspace abort):

  u_0^NC = solve_delay_aware at that tick's exact logged state, with NO
           Jacobian override -- i.e. literally what the real run computed
           (cross-checked against the logged u0 as an inline sanity check).
  u_0^C  = the SAME solve, with the schedule's horizon window replaced by
           J_C^state(x_k) -- the live contact model recomputed at the exact
           measured state (x_k = z_meas, the same quantity H4 uses as its one
           physically-grounded reference). This is the only tractable
           "what if the optimizer had the correct model right now" question,
           since no pre-built contact SCHEDULE exists off the reference
           manifold.

Both resulting commands are then evaluated through the SAME J_C^state(x_k)
(never through the schedule either command came from), decomposed into the
useful-correction component along e_hat = (p_ref-p_tip)/||p_ref-p_tip|| and
the transverse/wasted component, plus the resulting one-step change in the
magnet-exclusion and z-workspace margins (via a finite difference: command
integrated over one tick, FK'd to a new joint vector, re-evaluated).

Read-only; writes h2_same_state_counterfactual.csv and
h2_same_state_counterfactual.png only.
"""
import sys, time
sys.path.insert(0, "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis/scripts")
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import json
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


def nearest_tick(s_ref_mm, target, max_n):
    idx = int(np.argmin(np.abs(s_ref_mm[:max_n] - target)))
    if abs(s_ref_mm[idx] - target) > 3.0:
        return None
    return idx


def main():
    rows = []
    sanity_rows = []
    for dirname in RUNS:
        rm = next(r for r in manifest.RUNS if r["dirname"] == dirname)
        out, meta = loader.enrich_run(rm)
        with open(f"{manifest.BASE}/{dirname}/controller_metadata.json") as f:
            cm = json.load(f)
        s_ref_mm = out["s_ref_mm"]
        n = len(s_ref_mm)

        controller, sched_path, tick_log = rep.build_controller(dirname)
        print(f"[{dirname}] n_ticks={n} schedule={sched_path.split('/')[-1]}")

        for s_mark in MARKS_MM:
            k = nearest_tick(s_ref_mm, s_mark, n - 1)
            if k is None or k >= len(tick_log) - 1:
                rows.append(dict(dirname=dirname, rep=rm["rep"], s_mark_mm=s_mark, observed=False))
                continue

            rep.replay_residual_state(controller, tick_log, k)
            step_NC = rep.solve_same_state(controller, tick_log, k, J_override=None)
            u0_logged = np.asarray(tick_log[k]["u0"], dtype=float)
            sanity_err = float(np.linalg.norm(step_NC.command - u0_logged))
            sanity_rel = sanity_err / max(float(np.linalg.norm(u0_logged)), 1e-9)
            sanity_rows.append(dict(dirname=dirname, k=k, s_mark_mm=s_mark,
                                     sanity_abs_err=sanity_err, sanity_rel_err=sanity_rel))

            z_meas = np.asarray(tick_log[k]["z_meas"], dtype=float)  # x_k, the measured state
            J_C_state = live_jac.live_jacobian(z_meas, True)

            rep.replay_residual_state(controller, tick_log, k)  # fresh controller state for 2nd solve
            step_C = rep.solve_same_state(controller, tick_log, k, J_override=J_C_state)

            u_NC = step_NC.command.copy()
            u_C = step_C.command.copy()
            dchi_NC = u_NC * DT
            dchi_C = u_C * DT

            dp_NC = J_C_state @ dchi_NC
            dp_C = J_C_state @ dchi_C

            p_tip = np.asarray(tick_log[k]["measured_beam_position_m"], dtype=float)
            ref_idx = tick_log[k]["ref_index"]
            ref = loader.get_reference(cm["plan_dir"])
            p_ref = np.asarray(ref["desired_position_m"][ref_idx], dtype=float)
            e_vec = p_ref - p_tip
            e_norm = np.linalg.norm(e_vec)
            e_hat = e_vec / max(e_norm, 1e-12)

            c_par_NC = float(e_hat @ dp_NC)
            c_par_C = float(e_hat @ dp_C)
            c_perp_NC = float(np.linalg.norm(dp_NC - c_par_NC * e_hat))
            c_perp_C = float(np.linalg.norm(dp_C - c_par_C * e_hat))
            eta_NC = c_par_NC / max(float(np.linalg.norm(dchi_NC)), 1e-12)
            eta_C = c_par_C / max(float(np.linalg.norm(dchi_C)), 1e-12)

            q_now = z_meas[:6]
            magnet_now = loader.magnet_xyz_batch(q_now.reshape(1, 6))[0]
            margins_now = rep.workspace_margins(magnet_now, cm)
            q_next_NC = q_now + dchi_NC[:6]
            q_next_C = q_now + dchi_C[:6]
            magnet_next_NC = loader.magnet_xyz_batch(q_next_NC.reshape(1, 6))[0]
            magnet_next_C = loader.magnet_xyz_batch(q_next_C.reshape(1, 6))[0]
            margins_next_NC = rep.workspace_margins(magnet_next_NC, cm)
            margins_next_C = rep.workspace_margins(magnet_next_C, cm)

            rows.append(dict(
                dirname=dirname, rep=rm["rep"], s_mark_mm=s_mark, observed=True, tick=k,
                e_norm_mm=e_norm * 1e3,
                u_cmd_diff_norm=float(np.linalg.norm(u_NC - u_C)),
                u_cmd_diff_per_joint=json.dumps([float(x) for x in (u_NC - u_C)[:6]]),
                u_cmd_diff_insertion=float((u_NC - u_C)[6]),
                c_parallel_NC_mm=c_par_NC * 1e3, c_parallel_C_mm=c_par_C * 1e3,
                c_perp_NC_mm=c_perp_NC * 1e3, c_perp_C_mm=c_perp_C * 1e3,
                eta_NC_mm=eta_NC * 1e3, eta_C_mm=eta_C * 1e3,
                d_exclusion_margin_NC_mm=margins_next_NC["exclusion_margin_mm"] - margins_now["exclusion_margin_mm"],
                d_exclusion_margin_C_mm=margins_next_C["exclusion_margin_mm"] - margins_now["exclusion_margin_mm"],
                exclusion_margin_now_mm=margins_now["exclusion_margin_mm"],
            ))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/tables/h2_same_state_counterfactual.csv", index=False)
    sdf = pd.DataFrame(sanity_rows)
    sdf.to_csv(f"{OUT}/tables/h2_same_state_sanity_check.csv", index=False)
    print(f"\nwrote {OUT}/tables/h2_same_state_counterfactual.csv ({len(df)} rows)")
    print(f"sanity check: max rel_err={sdf.sanity_rel_err.max():.3%}, "
          f"median rel_err={sdf.sanity_rel_err.median():.3%}")

    obs = df[df.observed]
    print("\n--- per-mark summary (mean over reps) ---")
    summary = obs.groupby("s_mark_mm")[["eta_NC_mm", "eta_C_mm", "c_perp_NC_mm", "c_perp_C_mm",
                                          "u_cmd_diff_norm", "d_exclusion_margin_NC_mm",
                                          "d_exclusion_margin_C_mm", "e_norm_mm"]].mean()
    print(summary.to_string())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for rname, g in obs.groupby("dirname"):
        axes[0, 0].plot(g.s_mark_mm, g.eta_NC_mm, "o-", color="#b3331d", alpha=0.6,
                         label="useful correction, actual $J_{NC}$ command" if rname == RUNS[0] else None)
        axes[0, 0].plot(g.s_mark_mm, g.eta_C_mm, "s-", color="#1b7f3b", alpha=0.6,
                         label="useful correction, counterfactual $J_C^{state}$ command" if rname == RUNS[0] else None)
        axes[0, 1].plot(g.s_mark_mm, g.u_cmd_diff_norm, "d-", color="#444", alpha=0.6)
        axes[1, 0].plot(g.s_mark_mm, g.e_norm_mm, "^-", color="#1b4f8f", alpha=0.6)
        axes[1, 1].plot(g.s_mark_mm, g.d_exclusion_margin_NC_mm, "o-", color="#b3331d", alpha=0.6,
                         label="$\\Delta h_{excl}$ under actual $J_{NC}$ command" if rname == RUNS[0] else None)
        axes[1, 1].plot(g.s_mark_mm, g.d_exclusion_margin_C_mm, "s-", color="#1b7f3b", alpha=0.6,
                         label="$\\Delta h_{excl}$ under counterfactual $J_C^{state}$ command" if rname == RUNS[0] else None)
    axes[0, 0].set_ylabel(r"$\eta$ = useful correction / $\|\Delta\chi\|$ (mm)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_ylabel(r"$\|u_0^{NC}-u_0^{C}\|$ (command norm)")
    axes[1, 0].set_ylabel("tracking error at this tick (mm)")
    axes[1, 1].set_ylabel("one-tick change in exclusion margin (mm)")
    axes[1, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel("path progress s (mm)")
        ax.grid(alpha=0.3)
    fig.suptitle("H2 same-state counterfactual: tight-floor (255mm) MPC-$J_{NC}$,\n"
                 "actual command vs counterfactual $J_C^{state}$-informed command, evaluated through $J_C^{state}$")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fpath = f"{OUT}/figures/h2_same_state_counterfactual.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"saved {fpath}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
