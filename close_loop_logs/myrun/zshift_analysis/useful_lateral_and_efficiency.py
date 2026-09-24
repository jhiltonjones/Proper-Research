"""Useful lateral correction + net correction efficiency, MPC (Q_N on) vs
INV-7 (kp=1.0), raised (+30mm) rectangle workspace. Third companion to
mechanism_figure_zraise.py / effort_and_constraint_activity.py (same rep
lists, same schedule, same alignment-by-ref_index convention).

Useful lateral correction: at each tick, decompose the TOTAL commanded
feedback (delta_u = u0 - u_ref, both joint and insertion channels
combined) through the schedule Jacobian into its task-space contribution
Delta_p_fb = J @ delta_u * dt, then take the y (lateral/steering)
component and project it onto the CURRENT error direction:
    useful_lateral_k = sign(e_y,k) * Delta_p_fb_y,k
error convention: error = desired - achieved, so a positive value means
the commanded lateral motion is aimed correctly (closing the gap); a
negative value means it is aimed the wrong way (widening it) -- this is
a signed, per-tick, DIRECTIONAL correctness measure, not just magnitude.

Net correction efficiency: how much REAL tracking improvement is bought
per unit of normalized feedback effort spent. Per-tick ratios are noisy
whenever effort is near zero, so this uses a cumulative running ratio
instead:
    efficiency(k) = [sum_{i<=k} Delta_E_actual,i(+3)] / [sum_{i<=k} effort_i]
where Delta_E_actual,i(+3) is the REALIZED error-norm reduction over the
next 3 ticks (same definition as mechanism_figure_zraise.py's Panel 3,
E_now - E_actual_at_i+3) and effort_i is the same combined normalized
feedback effort from effort_and_constraint_activity.py. This converges to
a stable "mm of net tracking improvement per unit of normalized effort"
as the run progresses, rather than a single noisy per-tick number.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_accurate.npy"
S_U_Q = 0.05
S_U_L = 0.005
DT = 0.1
HORIZON = 3

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
schedule = np.load(SCHEDULE_PATH)


def load_rows(rundir):
    return [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]


def per_run_series(rundir):
    """Per-tick (indexed by step, in log order) useful_lateral, effort,
    and delta_E_actual(+3) -- plus the ref_index for alignment."""
    rows = load_rows(rundir)
    by_step = {r["step"]: r for r in rows}
    steps = sorted(by_step.keys())
    useful_lat, effort, dE_actual, ref_idx = [], [], [], []
    for k in steps:
        r = by_step[k]
        ref_now = min(r["ref_index"], N_REF - 1)
        u0 = np.asarray(r["u0"], dtype=float)
        u_ref = input_reference[ref_now]
        du = u0 - u_ref
        J = schedule[ref_now]
        dp_fb_mm = (J @ du) * DT * 1000.0
        e_y = r["error_mm"][1]
        useful_lat.append(float(np.sign(e_y) * dp_fb_mm[1]) if e_y != 0 else 0.0)

        du_norm_q = du[:6] / S_U_Q
        du_norm_L = du[6] / S_U_L
        effort.append(float(np.sqrt(np.sum(du_norm_q ** 2) + du_norm_L ** 2)))

        future = by_step.get(k + HORIZON)
        if future is not None:
            dE_actual.append(r["error_norm_mm"] - future["error_norm_mm"])
        else:
            dE_actual.append(np.nan)
        ref_idx.append(ref_now)
    return dict(step=np.array(steps), ref_idx=np.array(ref_idx),
                useful_lat=np.array(useful_lat), effort=np.array(effort),
                dE_actual=np.array(dE_actual))


def aggregate(runs):
    per_run = [per_run_series(rd) for rd in runs]

    # --- useful lateral, aligned by ref_index, mean+/-SD across reps ---
    idxs = sorted(set().union(*[set(d["ref_idx"].tolist()) for d in per_run]))
    ul_by_ri = {ri: [] for ri in idxs}
    for d in per_run:
        for ri, v in zip(d["ref_idx"], d["useful_lat"]):
            ul_by_ri[ri].append(v)
    ul_mean = np.array([np.mean(ul_by_ri[ri]) for ri in idxs])
    ul_sd = np.array([np.std(ul_by_ri[ri]) for ri in idxs])
    ul_frac_useful = float(np.mean([v > 0 for vals in ul_by_ri.values() for v in vals]))

    # --- cumulative efficiency, per run (own tick order), then averaged
    # by FRACTION OF PATH PROGRESS (0..1) so runs of slightly different
    # length align sensibly ---
    n_grid = 100
    BURN_IN = 5  # skip the first few ticks -- cumulative effort is near
    # zero there, so the ratio is dominated by division-noise, not signal
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

    return dict(idx=np.array(idxs), ul_mean=ul_mean, ul_sd=ul_sd,
                ul_frac_useful=ul_frac_useful, grid=grid, eff_mean=eff_mean,
                eff_sd=eff_sd, final_eff_mean=float(final_eff.mean()),
                final_eff_sd=float(final_eff.std()))


mpc = aggregate(MPC_RUNS)
inv = aggregate(INV7_RUNS)

print("Useful lateral correction (sign-aligned with current error, mm per tick):")
print(f"  MPC (Q_N on):  mean={np.nanmean(mpc['ul_mean']):+.4f}mm  "
      f"frac_ticks_useful={mpc['ul_frac_useful']:.3f}")
print(f"  INV-7 (kp=1):  mean={np.nanmean(inv['ul_mean']):+.4f}mm  "
      f"frac_ticks_useful={inv['ul_frac_useful']:.3f}")
print()
print("Net correction efficiency (cumulative mm of realized +3 error reduction "
      "per unit normalized effort), final value at path completion:")
print(f"  MPC (Q_N on):  {mpc['final_eff_mean']:+.4f} +/- {mpc['final_eff_sd']:.4f}")
print(f"  INV-7 (kp=1):  {inv['final_eff_mean']:+.4f} +/- {inv['final_eff_sd']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Useful lateral correction & net correction efficiency, raised (+30mm) workspace: "
             "MPC (Q_N on) vs INV-7 (kp=1.0)", fontsize=12)

ax = axes[0]
ax.plot(mpc["idx"], mpc["ul_mean"], color="tab:blue", label="MPC (Q_N on)")
ax.fill_between(mpc["idx"], mpc["ul_mean"] - mpc["ul_sd"], mpc["ul_mean"] + mpc["ul_sd"],
                 color="tab:blue", alpha=0.2)
ax.plot(inv["idx"], inv["ul_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
ax.fill_between(inv["idx"], inv["ul_mean"] - inv["ul_sd"], inv["ul_mean"] + inv["ul_sd"],
                 color="tab:orange", alpha=0.2)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("Panel A: useful lateral correction\n(sign-aligned with current error, +=correcting, -=worsening)")
ax.set_xlabel("reference index"); ax.set_ylabel("mm per tick")
ax.legend()

ax = axes[1]
ax.plot(mpc["grid"] * 100, mpc["eff_mean"], color="tab:blue", label="MPC (Q_N on)")
ax.fill_between(mpc["grid"] * 100, mpc["eff_mean"] - mpc["eff_sd"], mpc["eff_mean"] + mpc["eff_sd"],
                 color="tab:blue", alpha=0.2)
ax.plot(inv["grid"] * 100, inv["eff_mean"], color="tab:orange", label="INV-7 (kp=1.0)")
ax.fill_between(inv["grid"] * 100, inv["eff_mean"] - inv["eff_sd"], inv["eff_mean"] + inv["eff_sd"],
                 color="tab:orange", alpha=0.2)
ax.axhline(0, color="black", lw=0.7)
ax.set_title("Panel B: net correction efficiency (cumulative)\nmm realized +3 error reduction per unit normalized effort")
ax.set_xlabel("path progress (%)"); ax.set_ylabel("cumulative efficiency")
ax.legend()

fig.tight_layout()
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/useful_lateral_and_efficiency.png"
fig.savefig(OUT, dpi=150)
print(f"\nsaved -> {OUT}")
