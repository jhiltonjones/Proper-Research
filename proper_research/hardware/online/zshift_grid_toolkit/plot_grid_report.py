"""Generalized 4-figure + tables report for the 2x2 {SJ,FJ} x {MPC,INV-7}
grid, for any shape/plan. Auto-discovers run directories by the
run_name_prefix used by run_2x2_grid.sh (i.e. it does not need an explicit
list of run directories or which reps aborted -- both are read from each
run's own summary.json).

Produces the same 4 figures + tables as the original raised-rectangle
study (close_loop_logs/myrun/zshift_analysis/fj_full_report.py), which this
script generalizes:
  Fig 1: ||e_k|| vs ref_index, every rep individually, aborts marked
  Fig 2: Jacobian staleness r_J(k) + weak-axis principal angle, aligned
         above tracking error (staleness-precedes-divergence check)
  Fig 3: predicted-vs-actual motion direction (c_k) + "confidently wrong"
         ticks, for a representative failing FJ rep vs its SJ counterpart
  Fig 4: why MPC diverges while INV-7 degrades -- effort, joint-increment
         constraint activity, effective feedback gain K_eff(k)
  Table A: completion/aborts/RMS(completed, flagged if partial)/common-
           path RMS/max err/constraint saturation, per arm
  Table A2: common-path RMS by progress regime (0-33/33-66/66-100%)
  Table B: per-replicate MPC-FJ breakdown
  Table C: mechanism table (solver health, r_J, c_k, wrong-descent %, first
           saturation) -- MPC-SJ vs MPC-FJ

Usage:
    python3 -m proper_research.hardware.online.zshift_grid_toolkit.plot_grid_report \\
        --plan-dir plans/ushape_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path \\
        --sj-schedule /tmp/ushape_zraise30mm_schedule_SJ.npy \\
        --fj-schedule /tmp/ushape_zraise30mm_schedule_FJ.npy \\
        --run-dir close_loop_logs/myrun --run-name-prefix ushape_zraise30mm \\
        --shape-label "U-shape (+30mm)" --zraise-mm 30 \\
        --report-dir close_loop_logs/myrun/ushape_zraise30mm_analysis
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELAY_SAMPLES = 2
MAX_JOINT_STEP_RAD = 0.010
S_U_Q = 0.05
S_U_L = 0.005
TITLE_KW = dict(fontsize=10)


def discover_runs(run_dir, prefix, controller, jacobian):
    pattern = os.path.join(run_dir, f"{prefix}_{controller}_{jacobian.upper()}_rep*_*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    runs = []
    for d in dirs:
        summary_path = os.path.join(d, "summary.json")
        stop_reason = None
        if os.path.exists(summary_path):
            summary = json.load(open(summary_path))
            sr = summary.get("stop_reason", "")
            if "path_complete" not in sr:
                stop_reason = sr
        rep_label = os.path.basename(d).split("_rep")[-1].split("_")[0]
        runs.append((f"rep{rep_label}", d, stop_reason))
    return runs


def load_rows(rundir):
    rows = [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]
    rows.sort(key=lambda r: r["step"])
    return rows


def load_pb(rundir):
    path = f"{rundir}/predicted_beam_positions.jsonl"
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path)]
    return {r["step"]: r for r in rows}


def principal_angles_deg(A, B):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def diag_c_dV(rows, pb, J_model_fn):
    out = dict(ref_idx=[], c_k=[], dV_pred=[], dV_real=[])
    by_step = {r["step"]: r for r in rows}
    steps = sorted(by_step.keys())
    for i in range(len(steps) - 1):
        s, s1 = steps[i], steps[i + 1]
        if s1 != s + 1 or pb is None or s not in pb or s1 not in pb:
            continue
        z_k = np.asarray(pb[s]["z_meas"], dtype=float)
        z_k1 = np.asarray(pb[s1]["z_meas"], dtype=float)
        dz = z_k1 - z_k
        p_a = np.asarray(pb[s]["measured_beam_position_m"], dtype=float)
        p_b = np.asarray(pb[s1]["measured_beam_position_m"], dtype=float)
        dp_meas = p_b - p_a
        J = J_model_fn(by_step[s]["ref_index"])
        dp_model = J @ dz
        nfj, nmeas = np.linalg.norm(dp_model), np.linalg.norm(dp_meas)
        if nfj < 1e-9 or nmeas < 1e-9:
            continue
        c_k = float(np.dot(dp_model, dp_meas) / (nfj * nmeas))
        e_k = -np.asarray(by_step[s]["error_mm"], dtype=float) / 1000.0
        e_k1 = -np.asarray(by_step[s1]["error_mm"], dtype=float) / 1000.0
        dV_pred = float(2.0 * e_k @ dp_model)
        dV_real = float(e_k1 @ e_k1 - e_k @ e_k)
        out["ref_idx"].append(by_step[s]["ref_index"])
        out["c_k"].append(c_k)
        out["dV_pred"].append(dV_pred)
        out["dV_real"].append(dV_real)
    return {k: np.array(v) for k, v in out.items()}


def rms_over_window(rows, lo, hi):
    vals = [r["error_norm_mm"] for r in rows if lo <= r["ref_index"] <= hi]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def full_rms(rows):
    vals = [r["error_norm_mm"] for r in rows]
    return float(np.sqrt(np.mean(np.square(vals))))


def sat_p_active(rows, threshold=0.9):
    pairs = []
    for r in rows:
        dq = np.asarray(r["q_target_delta_rad"], dtype=float)
        pairs.extend((np.abs(dq) / MAX_JOINT_STEP_RAD).tolist())
    pairs = np.array(pairs)
    return float(np.mean(pairs >= threshold)) if len(pairs) else float("nan")


def solver_health(rows, pb):
    n = len(rows)
    ok = sum(1 for r in rows if r["solver_success"])
    dm = sum(1 for r in pb.values() if r.get("deadline_miss")) if pb else None
    delay_errs = []
    if pb:
        for r in rows:
            s_prev = r["step"] - DELAY_SAMPLES
            if s_prev in pb and r["step"] in pb:
                q_now = np.asarray(pb[r["step"]]["z_meas"][:6])
                q_then = np.asarray(pb[s_prev]["q_cmd_k"][:6])
                delay_errs.append(np.max(np.abs(q_now - q_then)))
    return ok, n, dm, (np.mean(delay_errs) if delay_errs else float("nan"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--sj-schedule", required=True)
    p.add_argument("--fj-schedule", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--run-name-prefix", required=True)
    p.add_argument("--shape-label", required=True)
    p.add_argument("--zraise-mm", type=float, default=0.0)
    p.add_argument("--report-dir", required=True)
    p.add_argument("--relinearize", action="store_true",
                    help="also relinearize J(z_k) online at a few representative ticks "
                         "(needs the beam model, ~10-20s; skip for a quick text-only pass)")
    args = p.parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    from proper_research.hardware.online.zshift_grid_toolkit import zraise_patch
    zraise_patch.apply(args.zraise_mm)

    J_SJ_schedule = np.load(args.sj_schedule)
    J_FJ_schedule = np.load(args.fj_schedule)
    J_FJ = J_FJ_schedule[0]
    N_REF = J_SJ_schedule.shape[0]

    mpc_sj = discover_runs(args.run_dir, args.run_name_prefix, "mpc", "sj")
    mpc_fj = discover_runs(args.run_dir, args.run_name_prefix, "mpc", "fj")
    inv_sj = discover_runs(args.run_dir, args.run_name_prefix, "inv7", "sj")
    inv_fj = discover_runs(args.run_dir, args.run_name_prefix, "inv7", "fj")
    for name, runs in [("MPC-SJ", mpc_sj), ("MPC-FJ", mpc_fj), ("INV-SJ", inv_sj), ("INV-FJ", inv_fj)]:
        print(f"[discover] {name}: {len(runs)} reps -> "
              f"{[(lbl, os.path.basename(d), reason) for lbl, d, reason in runs]}")
    if not (mpc_sj and mpc_fj and inv_sj and inv_fj):
        raise SystemExit("one or more of the 4 arms has zero discovered runs -- "
                          "check --run-dir/--run-name-prefix match run_2x2_grid.sh's naming")

    # ===================================================================
    # FIGURE 1
    # ===================================================================
    fig1, axes1 = plt.subplots(2, 1, figsize=(13, 11), sharex=True)
    fig1.suptitle(f"Figure 1 -- tracking error vs path progress, every replicate individually\n"
                  f"({args.shape_label}, MPC Q_N-on vs INV-7 kp=1.0, SJ vs FJ)", fontsize=12)
    colors_fj = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]

    for ax, sj_runs, fj_runs, title in [
        (axes1[0], mpc_sj, mpc_fj, "Panel A: MPC (Q_N on)"),
        (axes1[1], inv_sj, inv_fj, "Panel B: INV-7 (kp=1.0)"),
    ]:
        for _, d, _ in sj_runs:
            rows = load_rows(d)
            ax.plot([r["ref_index"] for r in rows], [r["error_norm_mm"] for r in rows],
                    color="tab:blue", alpha=0.5, lw=1.2)
        ax.plot([], [], color="tab:blue", alpha=0.5, lw=1.2, label=f"SJ (n={len(sj_runs)} reps)")
        for (label, d, abort_reason), c in zip(fj_runs, colors_fj):
            rows = load_rows(d)
            xs = [r["ref_index"] for r in rows]
            ys = [r["error_norm_mm"] for r in rows]
            tag = f"FJ {label}" + (f" [ABORT: {abort_reason}]" if abort_reason else " [completed]")
            ax.plot(xs, ys, color=c, lw=1.6, label=tag)
            if abort_reason:
                ax.scatter([xs[-1]], [ys[-1]], color=c, marker="x", s=140, zorder=6, linewidths=3)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("||e_k|| (mm)")
        ax.legend(fontsize=8, ncol=2)
    axes1[1].set_xlabel("ref_index")
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    fig1.savefig(f"{args.report_dir}/fig1_error_vs_progress_per_rep.png", dpi=150)
    print(f"saved -> {args.report_dir}/fig1_error_vs_progress_per_rep.png")

    # representative reps for figs 2-4: first ABORTED MPC-FJ rep if any,
    # else its rep1; first MPC-SJ / INV-FJ / INV-SJ rep as counterparts
    mpc_fj_aborted = [r for r in mpc_fj if r[2] is not None]
    rep_mpc_fj = mpc_fj_aborted[0] if mpc_fj_aborted else mpc_fj[0]
    rep_mpc_sj = mpc_sj[0]
    rep_inv_fj = inv_fj[0]
    mpc_fj_rows = load_rows(rep_mpc_fj[1])
    mpc_fj_pb = load_pb(rep_mpc_fj[1])
    mpc_sj_rows = load_rows(rep_mpc_sj[1])
    mpc_sj_pb = load_pb(rep_mpc_sj[1])
    inv_fj_rows = load_rows(rep_inv_fj[1])
    print(f"\n[representative reps] MPC-FJ={rep_mpc_fj[0]} ({'aborted: '+rep_mpc_fj[2] if rep_mpc_fj[2] else 'completed'}), "
          f"MPC-SJ={rep_mpc_sj[0]}, INV-FJ={rep_inv_fj[0]}")

    # ===================================================================
    # FIGURE 2
    # ===================================================================
    r_J_curve = np.array([
        np.linalg.norm(J_SJ_schedule[ri] - J_FJ, "fro") / np.linalg.norm(J_SJ_schedule[ri], "fro")
        for ri in range(N_REF)
    ])
    angle_curve = np.array([principal_angles_deg(J_FJ, J_SJ_schedule[ri]).max() for ri in range(N_REF)])

    relin_mpc, relin_inv = [], []
    if args.relinearize and mpc_fj_pb is not None:
        from proper_research.controllers.beam_jacobian_providers import from_model_bundle
        from proper_research.planning.planning_context import build_planning_context
        _, bundle, controller_pack, _ = build_planning_context()
        jac_provider_accurate = from_model_bundle(
            bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate"
        )
        steps_avail = sorted(mpc_fj_pb.keys())
        pick_ticks = steps_avail[len(steps_avail) // 2::max(1, len(steps_avail) // 12)][:6] if steps_avail else []
        for t in pick_ticks:
            z_k = np.asarray(mpc_fj_pb[t]["z_meas"], dtype=float)
            J_live = np.asarray(jac_provider_accurate(z_k), dtype=float).reshape(3, 7)
            r = float(np.linalg.norm(J_live - J_FJ, "fro") / np.linalg.norm(J_live, "fro"))
            relin_mpc.append((mpc_fj_pb[t]["ref_index"], r))
            print(f"[fig2] MPC-FJ {rep_mpc_fj[0]} relinearized tick={t} ref_idx={mpc_fj_pb[t]['ref_index']} r_J={r:.4f}")
        inv_by_ri = {r["ref_index"]: r for r in inv_fj_rows}
        for target_ri, _ in relin_mpc:
            if not inv_by_ri:
                break
            closest = min(inv_by_ri.keys(), key=lambda ri: abs(ri - target_ri))
            row = inv_by_ri[closest]
            z_k = np.array(row["q_meas_rad"] + [row["insertion_length_m"]], dtype=float)
            J_live = np.asarray(jac_provider_accurate(z_k), dtype=float).reshape(3, 7)
            r = float(np.linalg.norm(J_live - J_FJ, "fro") / np.linalg.norm(J_live, "fro"))
            relin_inv.append((closest, r))

    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    fig2.suptitle(f"Figure 2 -- Jacobian staleness precedes tracking divergence\n({args.shape_label})", fontsize=12)
    ax = axes2[0]
    ax.plot(range(N_REF), r_J_curve, color="tab:purple", label="r_J(ref_idx) vs genuine SJ schedule")
    if relin_mpc:
        ax.scatter(*zip(*relin_mpc), color="black", zorder=5, label=f"relinearized at MPC-FJ {rep_mpc_fj[0]}'s measured z_k")
    if relin_inv:
        ax.scatter(*zip(*relin_inv), color="darkgreen", marker="^", zorder=5, label=f"relinearized at INV-FJ {rep_inv_fj[0]}'s measured z_k")
    ax.set_ylabel("relative Frobenius error r_J")
    ax.set_title("Panel A: Jacobian staleness r_J(k)", **TITLE_KW)
    ax.legend(fontsize=8)

    ax = axes2[1]
    ax.plot(range(N_REF), angle_curve, color="tab:purple")
    ax.set_ylabel("degrees")
    ax.set_title("Panel B: weak-direction principal angle\nrow(J_FJ) vs row(J_sched(ref_idx))", **TITLE_KW)

    ax = axes2[2]
    ax.plot([r["ref_index"] for r in mpc_fj_rows], [r["error_norm_mm"] for r in mpc_fj_rows],
            color="tab:red", label=f"MPC-FJ {rep_mpc_fj[0]}" + (" (failing)" if rep_mpc_fj[2] else ""))
    ax.plot([r["ref_index"] for r in mpc_sj_rows], [r["error_norm_mm"] for r in mpc_sj_rows],
            color="tab:blue", alpha=0.7, label=f"MPC-SJ {rep_mpc_sj[0]} (clean, for reference)")
    ax.set_yscale("log")
    ax.set_xlabel("ref_index"); ax.set_ylabel("||e_k|| (mm)")
    ax.set_title("Panel C: tracking error, same x-axis\n(staleness leads, divergence follows)", **TITLE_KW)
    ax.legend(fontsize=8)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig(f"{args.report_dir}/fig2_jacobian_staleness.png", dpi=150)
    print(f"saved -> {args.report_dir}/fig2_jacobian_staleness.png")

    # ===================================================================
    # FIGURE 3
    # ===================================================================
    tail_lo = 0.6 * max(r["ref_index"] for r in mpc_fj_rows)
    d3_fj = diag_c_dV(mpc_fj_rows, mpc_fj_pb, lambda ri: J_FJ)
    d3_sj = diag_c_dV(mpc_sj_rows, mpc_sj_pb, lambda ri: J_SJ_schedule[min(ri, N_REF - 1)])
    wrong_fj = (d3_fj["dV_pred"] < 0) & (d3_fj["dV_real"] > 0)
    wrong_sj = (d3_sj["dV_pred"] < 0) & (d3_sj["dV_real"] > 0)
    tail_fj = d3_fj["ref_idx"] >= tail_lo
    tail_sj = d3_sj["ref_idx"] >= tail_lo
    pct_fj = 100 * wrong_fj[tail_fj].mean() if tail_fj.any() else float("nan")
    pct_sj = 100 * wrong_sj[tail_sj].mean() if tail_sj.any() else float("nan")
    print(f"[fig3] MPC-FJ {rep_mpc_fj[0]} tail(ref_idx>={tail_lo:.0f}) c_k mean={d3_fj['c_k'][tail_fj].mean():.3f}, wrong-descent%={pct_fj:.1f}%")
    print(f"[fig3] MPC-SJ {rep_mpc_sj[0]} tail(ref_idx>={tail_lo:.0f}) c_k mean={d3_sj['c_k'][tail_sj].mean():.3f}, wrong-descent%={pct_sj:.1f}%")

    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True)
    fig3.suptitle(f"Figure 3 -- predicted vs actual motion direction (MPC-FJ {rep_mpc_fj[0]} vs MPC-SJ {rep_mpc_sj[0]})", fontsize=12)
    ax = axes3[0]
    ax.plot(d3_fj["ref_idx"], d3_fj["c_k"], color="tab:red", label="FJ (model=J_FJ)")
    ax.plot(d3_sj["ref_idx"], d3_sj["c_k"], color="tab:blue", label="SJ (model=J_SJ,k)")
    ax.axhline(0, color="black", lw=0.7, linestyle=":")
    xmax = max(d3_fj["ref_idx"].max(), d3_sj["ref_idx"].max())
    ax.axvspan(tail_lo, xmax, color="gray", alpha=0.1, label=f"tail (ref_idx>={tail_lo:.0f})")
    ax.set_ylabel("c_k = cos<J*dz, dp_meas>")
    ax.set_title(f"Panel A: directional agreement\n(FJ tail mean={d3_fj['c_k'][tail_fj].mean():+.2f}, SJ tail mean={d3_sj['c_k'][tail_sj].mean():+.2f})", **TITLE_KW)
    ax.legend(fontsize=8)

    ax = axes3[1]
    ax.scatter(d3_fj["ref_idx"][wrong_fj], np.ones(wrong_fj.sum()), color="tab:red", marker="|", s=200,
               label=f"FJ: dV_pred<0 & dV_real>0 (tail {pct_fj:.0f}%)")
    ax.scatter(d3_sj["ref_idx"][wrong_sj], np.zeros(wrong_sj.sum()), color="tab:blue", marker="|", s=200,
               label=f"SJ: dV_pred<0 & dV_real>0 (tail {pct_sj:.0f}%)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["SJ", "FJ"]); ax.set_ylim(-0.5, 1.5)
    ax.axvspan(tail_lo, xmax, color="gray", alpha=0.1)
    ax.set_xlabel("ref_index")
    ax.set_title('Panel B: "confidently wrong" ticks\n(controller predicted error reduction, error actually grew)', **TITLE_KW)
    ax.legend(fontsize=8, loc="center left")
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig(f"{args.report_dir}/fig3_predicted_vs_actual_motion.png", dpi=150)
    print(f"saved -> {args.report_dir}/fig3_predicted_vs_actual_motion.png")

    # ===================================================================
    # FIGURE 4
    # ===================================================================
    plan = np.load(glob.glob(args.plan_dir + "/*.npz")[0])
    input_reference = plan["input_reference"]

    def effort_series(rows):
        out = dict(ref_idx=[], du_norm=[], max_joint_ratio=[], k_eff=[])
        for r in rows:
            ri = min(r["ref_index"], N_REF - 1)
            u0 = np.asarray(r["u0"], dtype=float)
            du = u0 - input_reference[ri]
            du_norm = float(np.sqrt(np.sum((du[:6] / S_U_Q) ** 2) + (du[6] / S_U_L) ** 2))
            dq = np.asarray(r["q_target_delta_rad"], dtype=float)
            max_ratio = float(np.max(np.abs(dq)) / MAX_JOINT_STEP_RAD)
            k_eff = du_norm / (r["error_norm_mm"] / 1000.0 + 1e-6)
            out["ref_idx"].append(r["ref_index"]); out["du_norm"].append(du_norm)
            out["max_joint_ratio"].append(max_ratio); out["k_eff"].append(k_eff)
        return {k: np.array(v) for k, v in out.items()}

    eff_mpc_fj = effort_series(mpc_fj_rows)
    eff_inv_fj = effort_series(inv_fj_rows)
    eff_mpc_sj = effort_series(mpc_sj_rows)
    eff_inv_sj = effort_series(load_rows(inv_sj[0][1]))

    sat_idx = np.where(eff_mpc_fj["max_joint_ratio"] >= 0.99)[0]
    first_sat_step = mpc_fj_rows[sat_idx[0]]["step"] if len(sat_idx) else None
    first_sat_ri = mpc_fj_rows[sat_idx[0]]["ref_index"] if len(sat_idx) else None
    print(f"[fig4] MPC-FJ {rep_mpc_fj[0]} first joint-step saturation at step={first_sat_step}")

    fig4, axes4 = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig4.suptitle(f"Figure 4 -- why MPC diverges while INV-7 merely degrades (both under the same stale J_FJ)\n({args.shape_label})", fontsize=12)
    ax = axes4[0]
    ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["du_norm"], color="tab:red", label=f"MPC-FJ {rep_mpc_fj[0]}")
    ax.plot(eff_mpc_sj["ref_idx"], eff_mpc_sj["du_norm"], color="tab:red", alpha=0.4, linestyle="--", label=f"MPC-SJ {rep_mpc_sj[0]} (clean)")
    ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["du_norm"], color="tab:orange", label=f"INV-FJ {rep_inv_fj[0]}")
    ax.plot(eff_inv_sj["ref_idx"], eff_inv_sj["du_norm"], color="tab:orange", alpha=0.4, linestyle="--", label="INV-SJ (clean)")
    ax.set_ylabel(r"$\|\delta u_k\| = \|S_u^{-1}(u_0-u_{ref})\|$")
    ax.set_title("Panel A: normalized commanded correction magnitude", **TITLE_KW)
    ax.legend(fontsize=8)

    ax = axes4[1]
    ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["max_joint_ratio"], color="tab:red", label=f"MPC-FJ {rep_mpc_fj[0]}")
    ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["max_joint_ratio"], color="tab:orange", label=f"INV-FJ {rep_inv_fj[0]}")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.7, label="hard limit")
    if first_sat_ri is not None:
        ax.axvline(first_sat_ri, color="black", linestyle="-.", alpha=0.6, label=f"first MPC-FJ saturation (ref_idx={first_sat_ri})")
    ax.set_ylabel("max|dq_i|/max_step")
    ax.set_title("Panel B: joint-increment constraint activity\n(saturation is a consequence, not the trigger)", **TITLE_KW)
    ax.legend(fontsize=8)

    ax = axes4[2]
    ax.plot(eff_mpc_fj["ref_idx"], eff_mpc_fj["k_eff"], color="tab:red", label=f"MPC-FJ {rep_mpc_fj[0]}")
    ax.plot(eff_inv_fj["ref_idx"], eff_inv_fj["k_eff"], color="tab:orange", label=f"INV-FJ {rep_inv_fj[0]}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$K_{eff}(k)=\|\delta u_k\|/(\|e_k\|+\epsilon)$")
    ax.set_xlabel("ref_index")
    ax.set_title("Panel C: effective feedback gain K_eff(k)\n(does correction strength grow with error, or stay bounded?)", **TITLE_KW)
    ax.legend(fontsize=8)
    fig4.tight_layout(rect=[0, 0, 1, 0.93])
    fig4.savefig(f"{args.report_dir}/fig4_why_mpc_diverges.png", dpi=150)
    print(f"saved -> {args.report_dir}/fig4_why_mpc_diverges.png")

    # ===================================================================
    # TABLES
    # ===================================================================
    mpc_fj_last_ref = {lbl: load_rows(d)[-1]["ref_index"] for lbl, d, _ in mpc_fj}
    K_MIN = min(mpc_fj_last_ref.values())
    print(f"\n[common-path] k_min = {K_MIN} (limited by {min(mpc_fj_last_ref, key=mpc_fj_last_ref.get)})")

    def summarize_arm(name, jac, runs):
        completed = sum(1 for _, _, r in runs if r is None)
        rms_completed_vals = [full_rms(load_rows(d)) for _, d, r in runs if r is None]
        common_rms_vals = [rms_over_window(load_rows(d), 0, K_MIN) for _, d, _ in runs]
        max_err_vals = [max(row["error_norm_mm"] for row in load_rows(d)) for _, d, _ in runs]
        p_sat_vals = [sat_p_active(load_rows(d)) for _, d, _ in runs]
        n = len(runs)
        flag = "*" if jac == "FJ" and completed < n else ""
        rms_c = f"{np.mean(rms_completed_vals):.3f}{flag}" if rms_completed_vals else "n/a"
        print(f"{name:10s} {jac:4s} {completed}/{n:<8d} {n-completed}/{n:<5d} {rms_c:16s} "
              f"{np.mean(common_rms_vals):.3f} +/- {np.std(common_rms_vals):.3f}      "
              f"{max(max_err_vals):.2f}                {np.mean(p_sat_vals):.4f}")
        return dict(completed=completed, n=n, common_rms=common_rms_vals)

    print("\n" + "=" * 100)
    print(f"TABLE A: primary live-performance results ({args.shape_label})")
    print("=" * 100)
    print(f"{'Controller':10s} {'Jac':4s} {'Completed':10s} {'Aborts':7s} {'RMS(completed)':16s} "
          f"{'CommonPathRMS(0-'+str(K_MIN)+')':24s} {'Max/final err (mm)':20s} {'P(joint sat)':12s}")
    res_mpc_sj = summarize_arm("MPC", "SJ", mpc_sj)
    res_mpc_fj = summarize_arm("MPC", "FJ", mpc_fj)
    res_inv_sj = summarize_arm("INV-7", "SJ", inv_sj)
    res_inv_fj = summarize_arm("INV-7", "FJ", inv_fj)
    if res_mpc_fj["completed"] < res_mpc_fj["n"]:
        print(f"\n* MPC-FJ RMS(completed) is only from {res_mpc_fj['completed']}/{res_mpc_fj['n']} completed "
              f"replicate(s) and must not be read as overall FJ performance.")
        print(f"  MPC-FJ's primary result is completion rate: {res_mpc_fj['completed']}/{res_mpc_fj['n']} "
              f"= {100*res_mpc_fj['completed']/res_mpc_fj['n']:.0f}%.")

    print("\n" + "=" * 100)
    print(f"TABLE A2: common-path RMS by progress regime (window = ref_index 0-{K_MIN})")
    print("=" * 100)
    regimes = [(0, K_MIN * 0.33, "0-33%"), (K_MIN * 0.33, K_MIN * 0.66, "33-66%"), (K_MIN * 0.66, K_MIN, "66-100%")]
    print(f"{'Controller':10s} {'Jac':4s} " + " ".join(f"{name:>10s}" for _, _, name in regimes))
    for name, jac, runs in [("MPC", "SJ", mpc_sj), ("MPC", "FJ", mpc_fj), ("INV-7", "SJ", inv_sj), ("INV-7", "FJ", inv_fj)]:
        per_regime = [np.nanmean([rms_over_window(load_rows(d), lo, hi) for _, d, _ in runs]) for lo, hi, _ in regimes]
        print(f"{name:10s} {jac:4s} " + " ".join(f"{v:10.3f}" for v in per_regime))

    print("\n" + "=" * 100)
    print("TABLE B: MPC-FJ per-replicate breakdown")
    print("=" * 100)
    print(f"{'rep':6s} {'outcome':11s} {'termination':30s} {'final err (mm)':16s} {'max err (mm)':14s} {'RMS (mm)':10s}")
    for lbl, d, reason in mpc_fj:
        rows = load_rows(d)
        outcome = "aborted" if reason else "completed"
        print(f"{lbl:6s} {outcome:11s} {str(reason):30s} {rows[-1]['error_norm_mm']:16.2f} "
              f"{max(r['error_norm_mm'] for r in rows):14.2f} {full_rms(rows):10.3f}")

    print("\n" + "=" * 100)
    print("TABLE C: mechanism table -- MPC-SJ vs MPC-FJ (representative reps)")
    print("=" * 100)
    ok_fj, n_fj, dm_fj, delay_fj = solver_health(mpc_fj_rows, mpc_fj_pb)
    ok_sj, n_sj, dm_sj, delay_sj = solver_health(mpc_sj_rows, mpc_sj_pb)
    print(f"{'metric':32s} {'MPC-SJ ('+rep_mpc_sj[0]+')':20s} {'MPC-FJ ('+rep_mpc_fj[0]+')':20s}")
    print(f"{'solver success':32s} {f'{ok_sj}/{n_sj}':20s} {f'{ok_fj}/{n_fj}':20s}")
    print(f"{'deadline misses':32s} {str(dm_sj):20s} {str(dm_fj):20s}")
    print(f"{'delay-consistency mean (rad)':32s} {delay_sj:<20.5f} {delay_fj:<20.5f}")
    if relin_mpc:
        r_J_late = np.mean([r for _, r in relin_mpc[-2:]])
        angle_late = angle_curve[[ri for ri, _ in relin_mpc][-1]]
        print(f"{'r_J late (relinearized)':32s} {'--':20s} {r_J_late:<20.3f}")
        print(f"{'weak-axis principal angle (deg)':32s} {'--':20s} {angle_late:<20.1f}")
    print(f"{'tail c_k (ref_idx>='+f'{tail_lo:.0f})':32s} {d3_sj['c_k'][tail_sj].mean():<20.3f} {d3_fj['c_k'][tail_fj].mean():<20.3f}")
    print(f"{'wrong-descent ticks, tail %':32s} {pct_sj:<20.1f} {pct_fj:<20.1f}")
    print(f"{'first joint-step saturation':32s} {'--':20s} {str(first_sat_step) + ' (step)':20s}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
