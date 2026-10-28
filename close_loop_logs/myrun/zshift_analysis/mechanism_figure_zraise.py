"""Mechanism figure, raised (+30mm) workspace: MPC (Q_N on) vs INV-7 (kp=1.0),
reconstructed to match the style/methodology of the earlier
close_loop_logs/myrun/mechanism_analysis_2026-09-21/mechanism_figure.png
(MPC-SJ vs INV-SJ, original workspace). That original script was not
preserved (interactive/scratch work from an earlier session), so this is a
careful reconstruction from the reference figure's own panel titles/axis
labels, not a byte-for-byte reproduction.

Panel 1: |e| (error norm, mm) vs reference index, mean +/- SD band across
  reps, both controllers overlaid. Gray vertical lines mark detected
  "corners" (local maxima of desired_tangent angular change).
Panel 2: axis-resolved MEAN error (not norm) vs reference index -- e_x
  solid, e_y dashed, both controllers, averaged across reps at each index.
Panel 3: "common-model" +3 prediction vs realization. Uses ONE shared,
  controller-agnostic linear model (the genuine schedule Jacobian, the
  SAME one both controllers' own schedules were built from) to predict
  what the REALIZED joint/insertion motion over the next 3 ticks should
  do to the beam tip -- independent of whether the controller that
  produced that motion has its own internal multi-step prediction. This
  is what makes the comparison possible at all for INV-7 (which has no
  internal prediction horizon of its own). Delta_E_chosen = predicted
  error REDUCTION over 3 ticks (E_now - E_common-model-predicted,3);
  Delta_E_actual = realized error reduction (E_now - E_actual,3).
Panel 4: actuator allocation for MPC (Q_N on) only, matching the
  reference figure's own scope -- decomposes the controller's OWN chosen
  feedback correction (u0 - u_ref) into joint-channel and insertion-
  channel contributions, each mapped through the same schedule Jacobian
  into x/y task-space displacement, averaged across MPC's clean reps.
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "/tmp/rectangle_zraise30mm_schedule_accurate.npy"

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
desired_tangent = plan["desired_tangent"]
input_reference = plan["input_reference"]
N_REF = plan["state_reference"].shape[0]
schedule = np.load(SCHEDULE_PATH)  # (209,3,7)
DT = 0.1

# --- corner detection ---
dot = np.clip(np.sum(desired_tangent[:-1] * desired_tangent[1:], axis=1), -1, 1)
ang = np.degrees(np.arccos(dot))
corner_idx, _ = find_peaks(ang, height=3, distance=10)
print("detected corner indices:", corner_idx.tolist())


def load_rows(rundir):
    return [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]


def by_ref_index(rows):
    """Collapse to one row per ref_index (last occurrence -- terminal-hold
    repeats all share the final ref_index, keep the first arrival)."""
    out = {}
    for r in rows:
        ri = r["ref_index"]
        if ri not in out:
            out[ri] = r
    return out


# ---------------------------------------------------------------------
# Panel 1 & 2 data: per-controller, per-ref-index mean/SD across reps
# ---------------------------------------------------------------------
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
            mean_norm.append(np.mean(vals_norm))
            sd_norm.append(np.std(vals_norm))
            mean_ex.append(np.mean(vals_ex))
            mean_ey.append(np.mean(vals_ey))
        else:
            mean_norm.append(np.nan); sd_norm.append(np.nan)
            mean_ex.append(np.nan); mean_ey.append(np.nan)
    return (np.array(idxs), np.array(mean_norm), np.array(sd_norm),
            np.array(mean_ex), np.array(mean_ey))


mpc_idx, mpc_mean, mpc_sd, mpc_ex, mpc_ey = panel12_data(MPC_RUNS)
inv_idx, inv_mean, inv_sd, inv_ex, inv_ey = panel12_data(INV7_RUNS)

# ---------------------------------------------------------------------
# Panel 3 data: common-model +3 prediction vs realization, pooled ticks
# ---------------------------------------------------------------------
def panel3_data(runs, horizon=3):
    all_dchosen, all_dactual = [], []
    for rd in runs:
        rows = load_rows(rd)
        by_step = {r["step"]: r for r in rows}
        for r in rows:
            k = r["step"]
            future = by_step.get(k + horizon)
            if future is None:
                continue
            ref_now = min(r["ref_index"], N_REF - 1)
            q_now = np.asarray(r["q_meas_rad"], dtype=float)
            L_now = float(r["insertion_length_m"])
            q_future = np.asarray(future["q_meas_rad"], dtype=float)
            L_future = float(future["insertion_length_m"])
            dz = np.concatenate([q_future - q_now, [L_future - L_now]])
            J = schedule[ref_now]  # (3,7)
            dp_model_mm = (J @ dz) * 1000.0
            p_now_mm = np.asarray(r["tip_mm"], dtype=float)
            p_pred_future_mm = p_now_mm + dp_model_mm
            desired_future_mm = np.asarray(future["desired_mm"], dtype=float)
            e_now = r["error_norm_mm"]
            e_chosen_future = float(np.linalg.norm(desired_future_mm - p_pred_future_mm))
            e_actual_future = future["error_norm_mm"]
            all_dchosen.append(e_now - e_chosen_future)
            all_dactual.append(e_now - e_actual_future)
    return np.array(all_dchosen), np.array(all_dactual)


mpc_dchosen, mpc_dactual = panel3_data(MPC_RUNS)
inv_dchosen, inv_dactual = panel3_data(INV7_RUNS)

r_mpc = np.corrcoef(mpc_dchosen, mpc_dactual)[0, 1] if len(mpc_dchosen) > 2 else np.nan
r_inv = np.corrcoef(inv_dchosen, inv_dactual)[0, 1] if len(inv_dchosen) > 2 else np.nan
print(f"Panel 3: MPC common-model r={r_mpc:.3f} (n={len(mpc_dchosen)}), "
      f"INV-7 common-model r={r_inv:.3f} (n={len(inv_dchosen)})")

# ---------------------------------------------------------------------
# Panel 4 data: MPC-only actuator allocation, averaged across reps by ref_index
# ---------------------------------------------------------------------
def panel4_data(runs):
    per_run_alloc = []
    for rd in runs:
        rows = load_rows(rd)
        by_ri = {}
        for r in rows:
            ri = min(r["ref_index"], N_REF - 1)
            if ri in by_ri:
                continue
            u0 = np.asarray(r["u0"], dtype=float)
            u_ref = input_reference[ri]
            du_q = u0[:6] - u_ref[:6]
            du_L = u0[6] - u_ref[6]
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


p4_idx, p4_jx, p4_jy, p4_ix, p4_iy = panel4_data(MPC_RUNS)

# ---------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mechanism figure (raised +30mm workspace): MPC (Q_N on) vs INV-7 (kp=1.0), "
             "rectangle path (n=4/5 reps)", fontsize=13)

ax = axes[0, 0]
ax.plot(mpc_idx, mpc_mean, color="tab:blue", label="MPC (Q_N on)")
ax.fill_between(mpc_idx, mpc_mean - mpc_sd, mpc_mean + mpc_sd, color="tab:blue", alpha=0.2)
ax.plot(inv_idx, inv_mean, color="tab:orange", label="INV-7 (kp=1.0)")
ax.fill_between(inv_idx, inv_mean - inv_sd, inv_mean + inv_sd, color="tab:orange", alpha=0.2)
for ci in corner_idx:
    ax.axvline(ci, color="gray", alpha=0.4)
ax.set_title("Panel 1: error norm vs path index (mean±SD), gray=corners")
ax.set_xlabel("reference index"); ax.set_ylabel("|e| (mm)")
ax.legend()

ax = axes[0, 1]
ax.plot(mpc_idx, mpc_ex, color="tab:blue", linestyle="-", label="MPC e_x")
ax.plot(mpc_idx, mpc_ey, color="tab:blue", linestyle="--", label="MPC e_y")
ax.plot(inv_idx, inv_ex, color="tab:orange", linestyle="-", label="INV-7 e_x")
ax.plot(inv_idx, inv_ey, color="tab:orange", linestyle="--", label="INV-7 e_y")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Panel 2: axis-resolved mean error")
ax.set_xlabel("reference index"); ax.set_ylabel("mean e_x/e_y (mm)")
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.scatter(mpc_dchosen, mpc_dactual, s=6, alpha=0.3, color="tab:blue", label=f"MPC (r={r_mpc:.2f})")
ax.scatter(inv_dchosen, inv_dactual, s=6, alpha=0.3, color="tab:orange", label=f"INV-7 (r={r_inv:.2f})")
lim = 2.0
ax.plot([-lim, lim], [-lim, lim], "k:", alpha=0.6)
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_title("Panel 3: common-model +3 prediction vs realization")
ax.set_xlabel("Delta E_chosen (predicted reduction, mm)")
ax.set_ylabel("Delta E_actual (realized reduction, mm)")
ax.legend()

ax = axes[1, 1]
ax.plot(p4_idx, p4_jx, color="tab:green", linestyle="-", label="MPC joint fb, x")
ax.plot(p4_idx, p4_jy, color="tab:green", linestyle="--", label="MPC joint fb, y")
ax.plot(p4_idx, p4_ix, color="tab:red", linestyle="-", label="MPC insertion fb, x")
ax.plot(p4_idx, p4_iy, color="tab:red", linestyle="--", label="MPC insertion fb, y")
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Panel 4: actuator allocation (MPC Q_N on) -- insertion→x, joints→y")
ax.set_xlabel("reference index"); ax.set_ylabel("task-space feedback contribution (mm)")
ax.legend(fontsize=8)

fig.tight_layout()
OUT = "/home/jack/.claude/jobs/3710eca5/tmp/mechanism_figure_zraise.png"
fig.savefig(OUT, dpi=150)
print(f"saved -> {OUT}")
