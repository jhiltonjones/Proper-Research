"""MPC (Q_N on) vs INV-7 mechanism/effort/efficiency report, generalized
from close_loop_logs/myrun/zshift_analysis/{mechanism_figure_zraise,
effort_and_constraint_activity,useful_lateral_and_efficiency}.py.

Compares the SJ (genuine-schedule) runs of both controllers -- this is a
controller-vs-controller mechanism comparison, orthogonal to the SJ-vs-FJ
divergence report (plot_grid_report.py). Auto-discovers MPC-SJ and INV-SJ
run directories the same way plot_grid_report.py does.

Produces 3 figures in --report-dir:
  mechanism_figure: (1) error norm vs path index mean+/-SD with detected
    corners, (2) axis-resolved mean error, (3) common-model +3 prediction
    vs realization (controller-agnostic, works for INV-7 despite it having
    no internal multi-step prediction), (4) MPC's own actuator allocation
    (joint fb vs insertion fb, x/y task-space contribution).
  effort_and_constraint_activity: (A) combined normalized feedback effort,
    (B) effort by channel, (C) max-over-joints joint-increment-vs-limit
    ratio, (D) P(joint increment active).
  useful_lateral_and_efficiency: (A) sign-aligned useful lateral
    correction, (B) cumulative net correction efficiency vs path progress.

Usage:
    python3 -m proper_research.hardware.online.zshift_grid_toolkit.mechanism_and_effort_report \\
        --plan-dir plans/<name>/time_parameterized_configuration_path \\
        --sj-schedule /tmp/<name>_schedule_SJ.npy \\
        --run-dir close_loop_logs/myrun --run-name-prefix <name> \\
        --shape-label "<Human-readable shape name>" --zraise-mm 30 \\
        --report-dir close_loop_logs/myrun/<name>_analysis
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

MAX_JOINT_STEP_RAD = 0.010
S_U_Q = 0.05
S_U_L = 0.005
ACTIVE_THRESHOLD = 0.9
DT = 0.1
HORIZON = 3


def load_rows(rundir):
    return [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]


def by_ref_index(rows):
    out = {}
    for r in rows:
        ri = r["ref_index"]
        if ri not in out:
            out[ri] = r
    return out


def by_step(rundir):
    return {r["step"]: r for r in load_rows(rundir)}


def discover(run_dir, prefix, controller, jacobian):
    pattern = os.path.join(run_dir, f"{prefix}_{controller}_{jacobian.upper()}_rep*_*")
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--sj-schedule", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-name-prefix", required=True)
    p.add_argument("--shape-label", required=True)
    p.add_argument("--zraise-mm", type=float, default=0.0)
    p.add_argument("--report-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    mpc_runs = discover(args.run_dir, args.run_name_prefix, "mpc", "sj")
    inv_runs = discover(args.run_dir, args.run_name_prefix, "inv7", "sj")
    print(f"MPC-SJ: {len(mpc_runs)} reps: {[os.path.basename(d) for d in mpc_runs]}")
    print(f"INV-SJ: {len(inv_runs)} reps: {[os.path.basename(d) for d in inv_runs]}")
    if not mpc_runs or not inv_runs:
        raise SystemExit("no MPC-SJ or INV-SJ runs discovered -- check --run-dir/--run-name-prefix")

    plan = np.load(glob.glob(args.plan_dir + "/*.npz")[0])
    desired_tangent = plan["desired_tangent"]
    input_reference = plan["input_reference"]
    N_REF = plan["state_reference"].shape[0]
    schedule = np.load(args.sj_schedule)

    dot = np.clip(np.sum(desired_tangent[:-1] * desired_tangent[1:], axis=1), -1, 1)
    ang = np.degrees(np.arccos(dot))
    corner_idx, _ = find_peaks(ang, height=3, distance=10)
    print("detected corner indices:", corner_idx.tolist())

    # ===================================================================
    # MECHANISM FIGURE
    # ===================================================================
    def panel12_data(runs):
        per_run = [by_ref_index(load_rows(rd)) for rd in runs]
        idxs = sorted(set().union(*[set(d.keys()) for d in per_run]))
        mean_norm, sd_norm, mean_ex, mean_ey = [], [], [], []
        for ri in idxs:
            vals_norm, vals_ex, vals_ey = [], [], []
            for d in per_run:
                r = d.get(ri)
                if r is None:
                    continue
                vals_norm.append(r["error_norm_mm"])
                vals_ex.append(r["error_mm"][0])
                vals_ey.append(r["error_mm"][1])
            if vals_norm:
                mean_norm.append(np.mean(vals_norm)); sd_norm.append(np.std(vals_norm))
                mean_ex.append(np.mean(vals_ex)); mean_ey.append(np.mean(vals_ey))
            else:
                mean_norm.append(np.nan); sd_norm.append(np.nan)
                mean_ex.append(np.nan); mean_ey.append(np.nan)
        return (np.array(idxs), np.array(mean_norm), np.array(sd_norm),
                np.array(mean_ex), np.array(mean_ey))

    mpc_idx, mpc_mean, mpc_sd, mpc_ex, mpc_ey = panel12_data(mpc_runs)
    inv_idx, inv_mean, inv_sd, inv_ex, inv_ey = panel12_data(inv_runs)

    def panel3_data(runs, horizon=HORIZON):
        all_dchosen, all_dactual = [], []
        for rd in runs:
            rows = load_rows(rd)
            bs = {r["step"]: r for r in rows}
            for r in rows:
                future = bs.get(r["step"] + horizon)
                if future is None:
                    continue
                ref_now = min(r["ref_index"], N_REF - 1)
                q_now = np.asarray(r["q_meas_rad"], dtype=float)
                L_now = float(r["insertion_length_m"])
                q_future = np.asarray(future["q_meas_rad"], dtype=float)
                L_future = float(future["insertion_length_m"])
                dz = np.concatenate([q_future - q_now, [L_future - L_now]])
                dp_model_mm = (schedule[ref_now] @ dz) * 1000.0
                p_pred_future_mm = np.asarray(r["tip_mm"], dtype=float) + dp_model_mm
                desired_future_mm = np.asarray(future["desired_mm"], dtype=float)
                e_chosen_future = float(np.linalg.norm(desired_future_mm - p_pred_future_mm))
                all_dchosen.append(r["error_norm_mm"] - e_chosen_future)
                all_dactual.append(r["error_norm_mm"] - future["error_norm_mm"])
        return np.array(all_dchosen), np.array(all_dactual)

    mpc_dchosen, mpc_dactual = panel3_data(mpc_runs)
    inv_dchosen, inv_dactual = panel3_data(inv_runs)
    r_mpc = np.corrcoef(mpc_dchosen, mpc_dactual)[0, 1] if len(mpc_dchosen) > 2 else np.nan
    r_inv = np.corrcoef(inv_dchosen, inv_dactual)[0, 1] if len(inv_dchosen) > 2 else np.nan
    print(f"Panel 3: MPC common-model r={r_mpc:.3f} (n={len(mpc_dchosen)}), "
          f"INV-7 common-model r={r_inv:.3f} (n={len(inv_dchosen)})")

    def panel4_data(runs):
        per_run_alloc = []
        for rd in runs:
            by_ri = {}
            for r in load_rows(rd):
                ri = min(r["ref_index"], N_REF - 1)
                if ri in by_ri:
                    continue
                u0 = np.asarray(r["u0"], dtype=float)
                du_q = u0[:6] - input_reference[ri][:6]
                du_L = u0[6] - input_reference[ri][6]
                J = schedule[ri]
                dp_joint_mm = (J[:, :6] @ du_q) * DT * 1000.0
                dp_ins_mm = (J[:, 6] * du_L) * DT * 1000.0
                by_ri[ri] = (dp_joint_mm[0], dp_joint_mm[1], dp_ins_mm[0], dp_ins_mm[1])
            per_run_alloc.append(by_ri)
        idxs = sorted(set().union(*[set(d.keys()) for d in per_run_alloc]))
        jx, jy, ix, iy = [], [], [], []
        for ri in idxs:
            vals = [d[ri] for d in per_run_alloc if ri in d]
            if not vals:
                jx.append(np.nan); jy.append(np.nan); ix.append(np.nan); iy.append(np.nan)
                continue
            vals = np.array(vals)
            jx.append(vals[:, 0].mean()); jy.append(vals[:, 1].mean())
            ix.append(vals[:, 2].mean()); iy.append(vals[:, 3].mean())
        return np.array(idxs), np.array(jx), np.array(jy), np.array(ix), np.array(iy)

    p4_idx, p4_jx, p4_jy, p4_ix, p4_iy = panel4_data(mpc_runs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Mechanism figure ({args.shape_label}): MPC (Q_N on) vs INV-7 (kp=1.0), "
                 f"SJ (n={len(mpc_runs)}/{len(inv_runs)} reps)", fontsize=12)

    ax = axes[0, 0]
    ax.plot(mpc_idx, mpc_mean, color="tab:blue", label="MPC (Q_N on)")
    ax.fill_between(mpc_idx, mpc_mean - mpc_sd, mpc_mean + mpc_sd, color="tab:blue", alpha=0.2)
    ax.plot(inv_idx, inv_mean, color="tab:orange", label="INV-7 (kp=1.0)")
    ax.fill_between(inv_idx, inv_mean - inv_sd, inv_mean + inv_sd, color="tab:orange", alpha=0.2)
    for ci in corner_idx:
        ax.axvline(ci, color="gray", alpha=0.4)
    ax.set_title("Panel 1: error norm vs path index (mean+/-SD), gray=corners", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("|e| (mm)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(mpc_idx, mpc_ex, color="tab:blue", linestyle="-", label="MPC e_x")
    ax.plot(mpc_idx, mpc_ey, color="tab:blue", linestyle="--", label="MPC e_y")
    ax.plot(inv_idx, inv_ex, color="tab:orange", linestyle="-", label="INV-7 e_x")
    ax.plot(inv_idx, inv_ey, color="tab:orange", linestyle="--", label="INV-7 e_y")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Panel 2: axis-resolved mean error", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("mean e_x/e_y (mm)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(mpc_dchosen, mpc_dactual, s=6, alpha=0.3, color="tab:blue", label=f"MPC (r={r_mpc:.2f})")
    ax.scatter(inv_dchosen, inv_dactual, s=6, alpha=0.3, color="tab:orange", label=f"INV-7 (r={r_inv:.2f})")
    lim = max(2.0, np.nanpercentile(np.abs(np.concatenate([mpc_dchosen, inv_dchosen])), 95))
    ax.plot([-lim, lim], [-lim, lim], "k:", alpha=0.6)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title("Panel 3: common-model +3 prediction vs realization", fontsize=10)
    ax.set_xlabel("Delta E_chosen (predicted reduction, mm)")
    ax.set_ylabel("Delta E_actual (realized reduction, mm)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(p4_idx, p4_jx, color="tab:green", linestyle="-", label="MPC joint fb, x")
    ax.plot(p4_idx, p4_jy, color="tab:green", linestyle="--", label="MPC joint fb, y")
    ax.plot(p4_idx, p4_ix, color="tab:red", linestyle="-", label="MPC insertion fb, x")
    ax.plot(p4_idx, p4_iy, color="tab:red", linestyle="--", label="MPC insertion fb, y")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Panel 4: actuator allocation (MPC Q_N on)", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("task-space feedback contribution (mm)")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{args.report_dir}/mechanism_figure.png", dpi=150)
    plt.close(fig)
    print(f"saved -> {args.report_dir}/mechanism_figure.png")

    # ===================================================================
    # EFFORT + CONSTRAINT ACTIVITY
    # ===================================================================
    def effort_and_activity(runs):
        per_run_combined, per_run_joint, per_run_ins, per_run_maxjoint = [], [], [], []
        all_joint_ratios = []
        for rd in runs:
            by_ri = by_ref_index(load_rows(rd))
            combined, joint_eff, ins_eff, max_joint_ratio = {}, {}, {}, {}
            for ri, r in by_ri.items():
                ref_idx = min(ri, N_REF - 1)
                u0 = np.asarray(r["u0"], dtype=float)
                u_ref = input_reference[ref_idx]
                du_q = (u0[:6] - u_ref[:6]) / S_U_Q
                du_L = (u0[6] - u_ref[6]) / S_U_L
                combined[ri] = float(np.sqrt(np.sum(du_q ** 2) + du_L ** 2))
                joint_eff[ri] = float(np.linalg.norm(du_q))
                ins_eff[ri] = float(abs(du_L))
                dq = np.asarray(r["q_target_delta_rad"], dtype=float)
                ratios = np.abs(dq) / MAX_JOINT_STEP_RAD
                max_joint_ratio[ri] = float(ratios.max())
                all_joint_ratios.extend(ratios.tolist())
            per_run_combined.append(combined); per_run_joint.append(joint_eff)
            per_run_ins.append(ins_eff); per_run_maxjoint.append(max_joint_ratio)
        idxs = sorted(set().union(*[set(d.keys()) for d in per_run_combined]))

        def agg(dicts):
            means, sds = [], []
            for ri in idxs:
                vals = [d[ri] for d in dicts if ri in d]
                means.append(np.mean(vals) if vals else np.nan)
                sds.append(np.std(vals) if vals else np.nan)
            return np.array(means), np.array(sds)

        combined_mean, combined_sd = agg(per_run_combined)
        joint_mean, joint_sd = agg(per_run_joint)
        ins_mean, ins_sd = agg(per_run_ins)
        maxjoint_mean, maxjoint_sd = agg(per_run_maxjoint)
        p_active = float(np.mean(np.asarray(all_joint_ratios) >= ACTIVE_THRESHOLD))
        return dict(idx=np.array(idxs), combined_mean=combined_mean, combined_sd=combined_sd,
                    joint_mean=joint_mean, joint_sd=joint_sd, ins_mean=ins_mean, ins_sd=ins_sd,
                    maxjoint_mean=maxjoint_mean, maxjoint_sd=maxjoint_sd, p_active=p_active,
                    n_pairs=len(all_joint_ratios))

    mpc_eff = effort_and_activity(mpc_runs)
    inv_eff = effort_and_activity(inv_runs)
    print(f"P(joint increment active, >={ACTIVE_THRESHOLD}): MPC={mpc_eff['p_active']:.4f} "
          f"(n={mpc_eff['n_pairs']}), INV-7={inv_eff['p_active']:.4f} (n={inv_eff['n_pairs']})")
    print(f"Normalized feedback effort (combined, mean): MPC={np.nanmean(mpc_eff['combined_mean']):.3f}, "
          f"INV-7={np.nanmean(inv_eff['combined_mean']):.3f}")

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f"Feedback effort & joint-increment constraint activity ({args.shape_label}): "
                  f"MPC (Q_N on) vs INV-7 (kp=1.0)", fontsize=12)

    ax = axes2[0, 0]
    ax.plot(mpc_eff["idx"], mpc_eff["combined_mean"], color="tab:blue", label="MPC (Q_N on)")
    ax.fill_between(mpc_eff["idx"], mpc_eff["combined_mean"] - mpc_eff["combined_sd"],
                     mpc_eff["combined_mean"] + mpc_eff["combined_sd"], color="tab:blue", alpha=0.2)
    ax.plot(inv_eff["idx"], inv_eff["combined_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
    ax.fill_between(inv_eff["idx"], inv_eff["combined_mean"] - inv_eff["combined_sd"],
                     inv_eff["combined_mean"] + inv_eff["combined_sd"], color="tab:orange", alpha=0.2)
    ax.set_title("Panel A: normalized feedback effort (combined)", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel(r"$\|S_u^{-1}(u_0-u_{ref})\|$")
    ax.legend()

    ax = axes2[0, 1]
    ax.plot(mpc_eff["idx"], mpc_eff["joint_mean"], color="tab:blue", linestyle="-", label="MPC joint")
    ax.plot(mpc_eff["idx"], mpc_eff["ins_mean"], color="tab:blue", linestyle="--", label="MPC insertion")
    ax.plot(inv_eff["idx"], inv_eff["joint_mean"], color="tab:orange", linestyle="-", label="INV-7 joint")
    ax.plot(inv_eff["idx"], inv_eff["ins_mean"], color="tab:orange", linestyle="--", label="INV-7 insertion")
    ax.set_title("Panel B: normalized feedback effort, by channel", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("normalized effort")
    ax.legend(fontsize=8)

    ax = axes2[1, 0]
    ax.plot(mpc_eff["idx"], mpc_eff["maxjoint_mean"], color="tab:blue", label="MPC (Q_N on)")
    ax.fill_between(mpc_eff["idx"], mpc_eff["maxjoint_mean"] - mpc_eff["maxjoint_sd"],
                     mpc_eff["maxjoint_mean"] + mpc_eff["maxjoint_sd"], color="tab:blue", alpha=0.2)
    ax.plot(inv_eff["idx"], inv_eff["maxjoint_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
    ax.fill_between(inv_eff["idx"], inv_eff["maxjoint_mean"] - inv_eff["maxjoint_sd"],
                     inv_eff["maxjoint_mean"] + inv_eff["maxjoint_sd"], color="tab:orange", alpha=0.2)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.6, label="hard limit")
    ax.axhline(ACTIVE_THRESHOLD, color="gray", linestyle=":", alpha=0.6, label=f"active >= {ACTIVE_THRESHOLD}")
    ax.set_title("Panel C: max-over-joints |dq_i|/max_joint_step_rad", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("fraction of hard limit")
    ax.legend(fontsize=8)

    ax = axes2[1, 1]
    labels = ["MPC\n(Q_N on)", "INV-7\n(kp=1.0)"]
    vals = [mpc_eff["p_active"], inv_eff["p_active"]]
    ax.bar(labels, vals, color=["tab:blue", "tab:orange"])
    ax.set_title(f"Panel D: P(joint increment active >= {ACTIVE_THRESHOLD})\n"
                 f"MPC={vals[0]:.4f}  INV-7={vals[1]:.4f}", fontsize=10)
    ax.set_ylabel("fraction of (tick, joint) pairs")

    fig2.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.08, hspace=0.35, wspace=0.25)
    fig2.savefig(f"{args.report_dir}/effort_and_constraint_activity.png", dpi=150)
    plt.close(fig2)
    print(f"saved -> {args.report_dir}/effort_and_constraint_activity.png")

    # ===================================================================
    # USEFUL LATERAL CORRECTION + NET EFFICIENCY
    # ===================================================================
    def per_run_series(rundir):
        by_step_map = by_step(rundir)
        steps = sorted(by_step_map.keys())
        useful_lat, effort, dE_actual, ref_idx = [], [], [], []
        for k in steps:
            r = by_step_map[k]
            ref_now = min(r["ref_index"], N_REF - 1)
            u0 = np.asarray(r["u0"], dtype=float)
            du = u0 - input_reference[ref_now]
            J = schedule[ref_now]
            dp_fb_mm = (J @ du) * DT * 1000.0
            e_y = r["error_mm"][1]
            useful_lat.append(float(np.sign(e_y) * dp_fb_mm[1]) if e_y != 0 else 0.0)
            du_norm_q = du[:6] / S_U_Q
            du_norm_L = du[6] / S_U_L
            effort.append(float(np.sqrt(np.sum(du_norm_q ** 2) + du_norm_L ** 2)))
            future = by_step_map.get(k + HORIZON)
            dE_actual.append(r["error_norm_mm"] - future["error_norm_mm"] if future is not None else np.nan)
            ref_idx.append(ref_now)
        return dict(step=np.array(steps), ref_idx=np.array(ref_idx),
                    useful_lat=np.array(useful_lat), effort=np.array(effort),
                    dE_actual=np.array(dE_actual))

    def aggregate(runs):
        per_run = [per_run_series(rd) for rd in runs]
        idxs = sorted(set().union(*[set(d["ref_idx"].tolist()) for d in per_run]))
        ul_by_ri = {ri: [] for ri in idxs}
        for d in per_run:
            for ri, v in zip(d["ref_idx"], d["useful_lat"]):
                ul_by_ri[ri].append(v)
        ul_mean = np.array([np.mean(ul_by_ri[ri]) for ri in idxs])
        ul_sd = np.array([np.std(ul_by_ri[ri]) for ri in idxs])
        ul_frac_useful = float(np.mean([v > 0 for vals in ul_by_ri.values() for v in vals]))

        n_grid = 100
        BURN_IN = 5
        grid = np.linspace(0, 1, n_grid)
        eff_curves = []
        for d in per_run:
            valid = ~np.isnan(d["dE_actual"])
            cum_reduction = np.cumsum(np.where(valid, d["dE_actual"], 0.0))
            cum_effort = np.cumsum(d["effort"])
            eff = np.full_like(cum_reduction, np.nan)
            eff[BURN_IN:] = cum_reduction[BURN_IN:] / cum_effort[BURN_IN:]
            progress = np.linspace(0, 1, len(eff))
            mask = ~np.isnan(eff)
            eff_curves.append(np.interp(grid, progress[mask], eff[mask]))
        eff_curves = np.array(eff_curves)
        eff_mean = eff_curves.mean(axis=0)
        eff_sd = eff_curves.std(axis=0)
        final_eff = eff_curves[:, -1]
        return dict(idx=np.array(idxs), ul_mean=ul_mean, ul_sd=ul_sd, ul_frac_useful=ul_frac_useful,
                    grid=grid, eff_mean=eff_mean, eff_sd=eff_sd,
                    final_eff_mean=float(final_eff.mean()), final_eff_sd=float(final_eff.std()))

    mpc_agg = aggregate(mpc_runs)
    inv_agg = aggregate(inv_runs)
    print(f"Useful lateral correction: MPC mean={np.nanmean(mpc_agg['ul_mean']):+.4f}mm "
          f"frac_useful={mpc_agg['ul_frac_useful']:.3f}; "
          f"INV-7 mean={np.nanmean(inv_agg['ul_mean']):+.4f}mm frac_useful={inv_agg['ul_frac_useful']:.3f}")
    print(f"Net correction efficiency (final): MPC={mpc_agg['final_eff_mean']:+.4f}+/-{mpc_agg['final_eff_sd']:.4f}, "
          f"INV-7={inv_agg['final_eff_mean']:+.4f}+/-{inv_agg['final_eff_sd']:.4f}")

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5.5))
    fig3.suptitle(f"Useful lateral correction & net correction efficiency ({args.shape_label}): "
                  f"MPC (Q_N on) vs INV-7 (kp=1.0)", fontsize=12)

    ax = axes3[0]
    ax.plot(mpc_agg["idx"], mpc_agg["ul_mean"], color="tab:blue", label="MPC (Q_N on)")
    ax.fill_between(mpc_agg["idx"], mpc_agg["ul_mean"] - mpc_agg["ul_sd"],
                     mpc_agg["ul_mean"] + mpc_agg["ul_sd"], color="tab:blue", alpha=0.2)
    ax.plot(inv_agg["idx"], inv_agg["ul_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
    ax.fill_between(inv_agg["idx"], inv_agg["ul_mean"] - inv_agg["ul_sd"],
                     inv_agg["ul_mean"] + inv_agg["ul_sd"], color="tab:orange", alpha=0.2)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Panel A: useful lateral correction\n(+=correcting, -=worsening)", fontsize=10)
    ax.set_xlabel("reference index"); ax.set_ylabel("mm per tick")
    ax.legend()

    ax = axes3[1]
    ax.plot(mpc_agg["grid"] * 100, mpc_agg["eff_mean"], color="tab:blue", label="MPC (Q_N on)")
    ax.fill_between(mpc_agg["grid"] * 100, mpc_agg["eff_mean"] - mpc_agg["eff_sd"],
                     mpc_agg["eff_mean"] + mpc_agg["eff_sd"], color="tab:blue", alpha=0.2)
    ax.plot(inv_agg["grid"] * 100, inv_agg["eff_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
    ax.fill_between(inv_agg["grid"] * 100, inv_agg["eff_mean"] - inv_agg["eff_sd"],
                     inv_agg["eff_mean"] + inv_agg["eff_sd"], color="tab:orange", alpha=0.2)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Panel B: net correction efficiency (cumulative)\nmm realized +3 error reduction / unit effort", fontsize=10)
    ax.set_xlabel("path progress (%)"); ax.set_ylabel("cumulative efficiency")
    ax.legend()

    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    fig3.savefig(f"{args.report_dir}/useful_lateral_and_efficiency.png", dpi=150)
    plt.close(fig3)
    print(f"saved -> {args.report_dir}/useful_lateral_and_efficiency.png")

    print("\nDONE.")


if __name__ == "__main__":
    main()
