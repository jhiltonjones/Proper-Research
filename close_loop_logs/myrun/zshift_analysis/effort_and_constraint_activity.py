"""Normalized feedback effort + joint command increment constraint activity,
MPC (Q_N on) vs INV-7 (kp=1.0), raised (+30mm) rectangle workspace.
Companion to mechanism_figure_zraise.py (same rep lists, same alignment-
by-ref_index convention).

Normalized feedback effort: delta_u = u0 - u_ref (u_ref = the plan's own
input_reference at that reference index -- both controllers run with
feedforward_joint_trajectory=False, so u0 IS the full commanded input,
not a residual on top of an applied feedforward; u_ref here is the
COST-reference/schedule input, not the executed feedforward), normalized
by this project's own established input scales (input_scale_q_rad_s=0.05,
input_scale_L_m_s=0.005 -- same convention as controller_metadata.json
throughout this project). Reported both combined (all 7 channels) and
split by channel (joint vs insertion).

Joint command increment constraint activity: |q_target_delta_rad_i| /
max_joint_step_rad (the actual hard constraint enforced every live tick,
common.py's MAX_JOINT_STEP_RAD=0.010 rad -- q_target_delta_rad is already
logged directly, the realized per-tick joint position step). Reported as
the max-over-joints ratio per tick (how close the MOST-active joint gets
to saturating) and as P(active), the fraction of (tick, joint) pairs
above a 0.9 threshold -- matching this project's own "constraint
activity" convention from CONTROLLER_JUSTIFICATION.md.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
MAX_JOINT_STEP_RAD = 0.010
S_U_Q = 0.05      # rad/s, input_scale_q_rad_s
S_U_L = 0.005     # m/s, input_scale_L_m_s
ACTIVE_THRESHOLD = 0.9

MPC_RUNS = [
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep4_20260924T113802Z",
    "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep5_20260924T114043Z",
]
INV7_RUNS = [
    "close_loop_logs/myrun/inv_7dof_delay_aware_zraise30mm_20260924T103744Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep2_20260924T111308Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep3_20260924T111434Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep4_20260924T111548Z",
    "close_loop_logs/myrun/inv_7dof_delay_aware_kp1_zraise30mm_rep5_20260924T111702Z",
]

plan = np.load(glob.glob(PLAN_DIR + "/*.npz")[0])
input_reference = plan["input_reference"]
N_REF = plan["state_reference"].shape[0]


def load_rows(rundir):
    return [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]


def by_ref_index(rows):
    out = {}
    for r in rows:
        ri = r["ref_index"]
        if ri not in out:
            out[ri] = r
    return out


def effort_and_activity(runs):
    per_run_combined, per_run_joint, per_run_ins, per_run_maxjoint = [], [], [], []
    all_joint_ratios = []  # pooled (tick,joint) pairs, for P(active)
    for rd in runs:
        rows = load_rows(rd)
        by_ri = by_ref_index(rows)
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
        per_run_combined.append(combined)
        per_run_joint.append(joint_eff)
        per_run_ins.append(ins_eff)
        per_run_maxjoint.append(max_joint_ratio)

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

    return dict(
        idx=np.array(idxs), combined_mean=combined_mean, combined_sd=combined_sd,
        joint_mean=joint_mean, joint_sd=joint_sd, ins_mean=ins_mean, ins_sd=ins_sd,
        maxjoint_mean=maxjoint_mean, maxjoint_sd=maxjoint_sd, p_active=p_active,
        n_pairs=len(all_joint_ratios),
    )


mpc = effort_and_activity(MPC_RUNS)
inv = effort_and_activity(INV7_RUNS)

print(f"P(joint increment active, |dq_i|/max_step >= {ACTIVE_THRESHOLD}):")
print(f"  MPC (Q_N on):  {mpc['p_active']:.4f}  (n={mpc['n_pairs']} tick-joint pairs)")
print(f"  INV-7 (kp=1):  {inv['p_active']:.4f}  (n={inv['n_pairs']} tick-joint pairs)")
print()
print("Normalized feedback effort (combined, mean over whole run):")
print(f"  MPC (Q_N on):  {np.nanmean(mpc['combined_mean']):.3f}")
print(f"  INV-7 (kp=1):  {np.nanmean(inv['combined_mean']):.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feedback effort & joint-increment constraint activity, raised (+30mm) workspace: "
             "MPC (Q_N on) vs INV-7 (kp=1.0)", fontsize=13)

ax = axes[0, 0]
ax.plot(mpc["idx"], mpc["combined_mean"], color="tab:blue", label="MPC (Q_N on)")
ax.fill_between(mpc["idx"], mpc["combined_mean"] - mpc["combined_sd"],
                 mpc["combined_mean"] + mpc["combined_sd"], color="tab:blue", alpha=0.2)
ax.plot(inv["idx"], inv["combined_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
ax.fill_between(inv["idx"], inv["combined_mean"] - inv["combined_sd"],
                 inv["combined_mean"] + inv["combined_sd"], color="tab:orange", alpha=0.2)
ax.set_title("Panel A: normalized feedback effort (combined, all channels)")
ax.set_xlabel("reference index"); ax.set_ylabel(r"$\|S_u^{-1}(u_0-u_{ref})\|$")
ax.legend()

ax = axes[0, 1]
ax.plot(mpc["idx"], mpc["joint_mean"], color="tab:blue", linestyle="-", label="MPC joint")
ax.plot(mpc["idx"], mpc["ins_mean"], color="tab:blue", linestyle="--", label="MPC insertion")
ax.plot(inv["idx"], inv["joint_mean"], color="tab:orange", linestyle="-", label="INV-7 joint")
ax.plot(inv["idx"], inv["ins_mean"], color="tab:orange", linestyle="--", label="INV-7 insertion")
ax.set_title("Panel B: normalized feedback effort, by channel")
ax.set_xlabel("reference index"); ax.set_ylabel("normalized effort")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(mpc["idx"], mpc["maxjoint_mean"], color="tab:blue", label="MPC (Q_N on)")
ax.fill_between(mpc["idx"], mpc["maxjoint_mean"] - mpc["maxjoint_sd"],
                 mpc["maxjoint_mean"] + mpc["maxjoint_sd"], color="tab:blue", alpha=0.2)
ax.plot(inv["idx"], inv["maxjoint_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
ax.fill_between(inv["idx"], inv["maxjoint_mean"] - inv["maxjoint_sd"],
                 inv["maxjoint_mean"] + inv["maxjoint_sd"], color="tab:orange", alpha=0.2)
ax.axhline(1.0, color="black", linestyle=":", alpha=0.6, label="hard limit")
ax.axhline(ACTIVE_THRESHOLD, color="gray", linestyle=":", alpha=0.6, label=f"active >= {ACTIVE_THRESHOLD}")
ax.set_title("Panel C: max-over-joints |dq_i| / max_joint_step_rad")
ax.set_xlabel("reference index"); ax.set_ylabel("fraction of hard limit (0.010 rad)")
ax.legend(fontsize=8)

ax = axes[1, 1]
labels = ["MPC\n(Q_N on)", "INV-7\n(kp=1.0)"]
vals = [mpc["p_active"], inv["p_active"]]
ax.bar(labels, vals, color=["tab:blue", "tab:orange"])
ax.set_title(f"Panel D: P(joint increment active), threshold={ACTIVE_THRESHOLD}")
ax.set_ylabel("fraction of (tick, joint) pairs")
for i, v in enumerate(vals):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center")

fig.tight_layout()
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/effort_and_constraint_activity.png"
fig.savefig(OUT, dpi=150)
print(f"\nsaved -> {OUT}")
