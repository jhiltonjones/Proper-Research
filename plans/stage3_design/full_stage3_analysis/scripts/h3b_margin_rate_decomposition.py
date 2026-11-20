"""§5.4 extension: instantaneous workspace-margin-rate decomposition and a
same-state Kp=0.6 vs Kp=1.0 counterfactual, for the contact-Jacobian
selective-gate runs (255mm) only.

Builds the causal chain requested directly, rather than only via onset
chronology: e_lag(k), ||u_raw(k)||, ||delta_chi_actual(k)||, and the
INSTANTANEOUS, linearized workspace-margin RATE

    delta_h_xmax,k ~= grad_chi(h_xmax)^T @ delta_chi_actual_k

decomposed per DOF (delta_h_i = dh/dchi_i * delta_chi_i), using the exact
same margin-gradient machinery as h3b_workspace_and_nullspace.py (flange FK,
finite-difference gradient -- cheap, no live contact-model calls needed).
The predicted delta_h is sanity-checked against the REAL observed
h_xmax(k+1)-h_xmax(k) before being used as evidence.

Then, at a sample of late (pre-failure) states, a same-state counterfactual:
recompute the controller's raw command with Kp=1.0 instead of its own
logged Kp=0.6, holding the measured state, the desired/reference position,
and the REAL logged recursive previous-command history fixed (exactly the
same "freeze everything else, vary one input" convention used by every
other same-state counterfactual in this report). Since task_velocity =
pseudo @ (Kp * e / dt) and neither `pseudo` nor `e` depends on Kp, the
PRE-CLIP command scales exactly linearly in Kp; the velocity/accel/
state-box clip is then reapplied (not linear), using h3_replay.py's own
clip_command, to get a physically realistic post-clip counterfactual
command. Both the real (Kp=0.6) and counterfactual (Kp=1.0) commands are
then run through the identical margin-gradient decomposition to compare
delta_h_xmax under each gain, at the same states.

Read-only against existing hardware logs and controller source; writes
only into plans/stage3_design/full_stage3_analysis/{tables,figures}/.
"""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest
import loader
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
DT = 0.1
WS_MIN = np.array([0.277, -0.832, 0.170])
WS_MAX = np.array([0.656, -0.200, 0.433])
I4 = np.eye(4)

SCHED_C = f"{loader.REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy"

VEL_LIMIT = np.array([0.1] * 6 + [0.002])
ACC_LIMIT = np.array([0.4] * 6 + [10.0 * 0.002])
STATE_MIN = np.array([-2 * np.pi] * 6 + [-0.05])
STATE_MAX = np.array([2 * np.pi] * 6 + [0.20])


def clip_command(command, state, previous):
    command = np.clip(command, -VEL_LIMIT, VEL_LIMIT)
    command = np.clip(command, previous - ACC_LIMIT * DT, previous + ACC_LIMIT * DT)
    command = np.clip(command, (STATE_MIN - state) / DT, (STATE_MAX - state) / DT)
    return np.clip(command, -VEL_LIMIT, VEL_LIMIT)


def flange_xyz(q6):
    T = urik.forward_kinematics(q6, loader._ROBOT_KIN.dh, I4)
    return T.T_R_target[:3, 3]


def h_xmax(q6):
    """Signed margin (m) to the x_max TCP-box face -- positive = inside."""
    return WS_MAX[0] - flange_xyz(q6)[0]


def h_xmax_gradient(q6, eps=1e-6):
    grad = np.zeros(7)  # 7th (insertion) is always exactly 0: flange pose doesn't depend on L
    h0 = h_xmax(q6)
    for i in range(6):
        qp = q6.copy()
        qp[i] += eps
        grad[i] = (h_xmax(qp) - h0) / eps
    return grad


def raw_command_preclip(position_gain, damping, J_sched, desired, measured):
    gram = J_sched @ J_sched.T + (damping ** 2) * np.eye(3)
    pseudo = J_sched.T @ np.linalg.solve(gram, np.eye(3))
    return pseudo @ (position_gain * (desired - measured) / DT)  # nullspace_gain=0, feedforward=False


sched = np.load(SCHED_C)
runs = sorted([rm for rm in manifest.RUNS if rm["group"] == "invjac_selective"], key=lambda r: r["rep"])

chrono_rows = []
kp_rows = []

for rm in runs:
    out, meta = loader.enrich_run(rm)
    ref = loader.get_reference(meta["plan_dir"])
    desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)
    N = len(out["step"])
    q6 = out["q_meas_rad"]
    L = out["insertion_length_m"]
    tip_m = out["tip_mm"] / 1000.0
    ref_idx = out["ref_index"]
    s = out["s_ref_mm"]
    e_lag = out["e_lag_mm"]
    position_gain = float(meta["position_gain"])
    damping = float(meta["damping"])

    margins = np.array([h_xmax(q6[k]) for k in range(N)])
    executed = out["u0"]  # post-gate, what actually happened

    prev_raw = np.zeros(7)
    for k in range(N - 1):
        index = int(np.clip(ref_idx[k] + 1, 0, sched.shape[0] - 1))
        J_sched = sched[index]
        desired = desired_position_m[index]
        measured = tip_m[k]
        state7 = np.concatenate([q6[k], [L[k]]])

        u_raw_preclip = raw_command_preclip(position_gain, damping, J_sched, desired, measured)
        u_raw = clip_command(u_raw_preclip, state7, prev_raw)
        prev_raw = u_raw.copy()

        delta_chi_actual = executed[k] * DT  # realized configuration change this tick
        grad = h_xmax_gradient(q6[k])
        delta_h_pred = float(grad @ delta_chi_actual)
        delta_h_actual = float(margins[k + 1] - margins[k])
        per_dof = grad[:6] * delta_chi_actual[:6]

        chrono_rows.append(dict(
            rep=rm["rep"], tick=int(out["step"][k]), s_mm=float(s[k]),
            e_lag_mm=float(e_lag[k]), h_xmax_mm=float(margins[k] * 1000.0),
            norm_u_raw=float(np.linalg.norm(u_raw)),
            norm_delta_chi_actual=float(np.linalg.norm(delta_chi_actual)),
            delta_h_xmax_pred_mm=delta_h_pred * 1000.0,
            delta_h_xmax_actual_mm=delta_h_actual * 1000.0,
            **{f"delta_h_q{i+1}_mm": float(per_dof[i] * 1000.0) for i in range(6)},
        ))

        # --- Kp=0.6 vs Kp=1.0 same-state counterfactual, late window only ---
        if s[k] > 45.0:
            u_raw_preclip_kp1 = u_raw_preclip * (1.0 / position_gain)  # exact linear rescale
            u_kp1 = clip_command(u_raw_preclip_kp1, state7, prev_raw)  # same real previous-command history
            delta_chi_kp1 = u_kp1 * DT
            delta_h_kp06 = float(grad @ delta_chi_actual) * 1000.0  # same as delta_h_pred above, Kp=0.6 reconstruction
            delta_h_kp10 = float(grad @ delta_chi_kp1) * 1000.0
            kp_rows.append(dict(
                rep=rm["rep"], tick=int(out["step"][k]), s_mm=float(s[k]),
                h_xmax_mm=float(margins[k] * 1000.0),
                norm_u_kp06=float(np.linalg.norm(u_raw)), norm_u_kp10=float(np.linalg.norm(u_kp1)),
                delta_h_xmax_kp06_mm=delta_h_kp06, delta_h_xmax_kp10_mm=delta_h_kp10,
            ))

    print(f"rep{rm['rep']}: {N} ticks done, final h_xmax={margins[-1]*1000:.2f}mm, stop={meta['stop_reason']}")

chrono_df = pd.DataFrame(chrono_rows)
chrono_df.to_csv(f"{OUT}/tables/h3b_margin_rate_decomposition.csv", index=False)
print(f"\nwrote {OUT}/tables/h3b_margin_rate_decomposition.csv ({len(chrono_df)} rows)")

# sanity check: predicted vs actual one-tick margin change
corr = np.corrcoef(chrono_df.delta_h_xmax_pred_mm, chrono_df.delta_h_xmax_actual_mm)[0, 1]
mae = float((chrono_df.delta_h_xmax_pred_mm - chrono_df.delta_h_xmax_actual_mm).abs().mean())
print(f"sanity check: predicted-vs-actual one-tick delta_h_xmax correlation={corr:.3f}, "
      f"mean abs error={mae:.4f}mm")

kp_df = pd.DataFrame(kp_rows)
kp_df.to_csv(f"{OUT}/tables/h3b_kp_counterfactual_margin_rate.csv", index=False)
print(f"wrote {OUT}/tables/h3b_kp_counterfactual_margin_rate.csv ({len(kp_df)} rows)")
print(f"\nKp=0.6 vs Kp=1.0, s>45mm (n={len(kp_df)}):")
print(f"  mean delta_h_xmax @ Kp=0.6: {kp_df.delta_h_xmax_kp06_mm.mean():.4f}mm")
print(f"  mean delta_h_xmax @ Kp=1.0: {kp_df.delta_h_xmax_kp10_mm.mean():.4f}mm")
print(f"  fraction of ticks where Kp=1.0 is MORE negative than Kp=0.6: "
      f"{(kp_df.delta_h_xmax_kp10_mm < kp_df.delta_h_xmax_kp06_mm).mean():.3f}")
print(f"  mean |delta_h| @ Kp=0.6: {kp_df.delta_h_xmax_kp06_mm.abs().mean():.4f}mm, "
      f"@ Kp=1.0: {kp_df.delta_h_xmax_kp10_mm.abs().mean():.4f}mm")

# --- figure 1: chronology ---
fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
colors = {1: "#1b7f3b", 2: "#2f8fd1", 3: "#b3331d"}
panels = [
    ("e_lag_mm", r"$e_{lag}$ (mm)"),
    ("norm_u_raw", r"$\|u_{raw}\|$"),
    ("norm_delta_chi_actual", r"$\|\Delta\chi_{actual}\|$"),
    ("delta_h_xmax_pred_mm", r"$\Delta h_{x_{max}}$ per tick (mm, predicted)"),
    ("h_xmax_mm", r"$h_{x_{max}}$ margin (mm)"),
]
for ax, (col, ylabel) in zip(axes, panels):
    for rep, d in chrono_df.groupby("rep"):
        d = d.sort_values("s_mm")
        ax.plot(d.s_mm, d[col], color=colors[rep], lw=1.0, alpha=0.8, label=f"rep{rep}" if col == "e_lag_mm" else None)
    if col == "delta_h_xmax_pred_mm":
        ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
axes[0].set_title("Contact-Jacobian selective-gate runs (255mm): lag -> command -> margin-rate -> margin chronology", fontsize=11)
axes[-1].set_xlabel("path progress $s$ (mm)")
fig.tight_layout()
f1 = f"{OUT}/figures/h3b_margin_rate_chronology.png"
fig.savefig(f1, dpi=160)
plt.close(fig)
print(f"\nsaved {f1}")

# --- figure 2: Kp counterfactual ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
ax = axes[0]
for rep, d in kp_df.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d.s_mm, d.delta_h_xmax_kp06_mm, color="#1b7f3b", lw=1.0, alpha=0.7,
            label="$K_p=0.6$ (actual)" if rep == kp_df.rep.iloc[0] else None)
    ax.plot(d.s_mm, d.delta_h_xmax_kp10_mm, color="#b3331d", lw=1.0, alpha=0.7,
            label="$K_p=1.0$ (counterfactual)" if rep == kp_df.rep.iloc[0] else None)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("path progress $s$ (mm)")
ax.set_ylabel(r"$\Delta h_{x_{max}}$ per tick (mm)")
ax.set_title("Same-state counterfactual:\nper-tick margin change, $K_p=0.6$ vs $1.0$", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(kp_df.delta_h_xmax_kp06_mm, kp_df.delta_h_xmax_kp10_mm, s=10, alpha=0.5, color="#333333")
lims = [min(kp_df.delta_h_xmax_kp06_mm.min(), kp_df.delta_h_xmax_kp10_mm.min()),
        max(kp_df.delta_h_xmax_kp06_mm.max(), kp_df.delta_h_xmax_kp10_mm.max())]
ax.plot(lims, lims, color="gray", lw=1.0, ls="--", label="$y=x$ (no change)")
ax.set_xlabel(r"$\Delta h_{x_{max}}$ @ $K_p=0.6$ (mm)")
ax.set_ylabel(r"$\Delta h_{x_{max}}$ @ $K_p=1.0$ (mm)")
ax.set_title("Points below $y=x$:\nhigher gain makes this tick's margin step more negative", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
f2 = f"{OUT}/figures/h3b_kp_counterfactual_margin_rate.png"
fig.savefig(f2, dpi=160)
plt.close(fig)
print(f"saved {f2}")
