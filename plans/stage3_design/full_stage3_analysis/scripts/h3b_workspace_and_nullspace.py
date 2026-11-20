"""Task 3/4 (H3b): exact workspace face, per-DOF divergence, margin-sensitivity
decomposition, and null-space projection for the 6 selective-gate
inverse-Jacobian runs (255mm floor) that all stop on `tcp_out_of_workspace`.

Read-only against existing hardware logs and model/controller code; writes
only into plans/stage3_design/full_stage3_analysis/{scripts,tables,figures}/.

TCP/flange frame: confirmed from close_loop_path_follow.py (pose = reader.
latest_pose(); checked directly against cfg.workspace_xyz_min_m/max_m) and
ur_magnet_ik_jacobian_validation.ValidationConfig.offline_active_T_flange_tcp_pose6
== (0,0,0,0,0,0) -- i.e. the robot's configured TCP IS the bare flange frame
(identity offset), not the magnet point. So TCP xyz here = forward_kinematics
(q, dh, I_4) -- NOT loader.magnet_xyz_batch (which uses T_F_M, the magnet
offset).
"""
import sys, time
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest, loader, live_jac
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"

WS_MIN = np.array([0.277, -0.832, 0.170])
WS_MAX = np.array([0.656, -0.200, 0.433])
FACE_NAMES = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
I4 = np.eye(4)
DOF_NAMES = ["q1", "q2", "q3", "q4", "q5", "q6", "L"]


def flange_xyz_batch(q6_batch):
    out = np.zeros((q6_batch.shape[0], 3))
    for i in range(q6_batch.shape[0]):
        T = urik.forward_kinematics(q6_batch[i], loader._ROBOT_KIN.dh, I4)
        out[i] = T.T_R_target[:3, 3]
    return out


def face_margins(xyz):
    """xyz: (N,3) -> (N,6) signed margins (positive = inside), order FACE_NAMES."""
    N = xyz.shape[0]
    m = np.zeros((N, 6))
    m[:, 0] = xyz[:, 0] - WS_MIN[0]   # x_min margin
    m[:, 1] = WS_MAX[0] - xyz[:, 0]   # x_max margin
    m[:, 2] = xyz[:, 1] - WS_MIN[1]
    m[:, 3] = WS_MAX[1] - xyz[:, 1]
    m[:, 4] = xyz[:, 2] - WS_MIN[2]
    m[:, 5] = WS_MAX[2] - xyz[:, 2]
    return m


def flange_margin_gradient(q6, face_idx):
    """Finite-difference d(margin_face)/d(q_i) for i=0..5, margin w.r.t.
    insertion L is exactly 0 (flange pose depends only on the 6 arm joints)."""
    eps = 1e-6
    grad = np.zeros(7)
    xyz0 = flange_xyz_batch(q6.reshape(1, 6))[0]
    m0 = face_margins(xyz0.reshape(1, 3))[0, face_idx]
    for i in range(6):
        qp = q6.copy(); qp[i] += eps
        xyzp = flange_xyz_batch(qp.reshape(1, 6))[0]
        mp = face_margins(xyzp.reshape(1, 3))[0, face_idx]
        grad[i] = (mp - m0) / eps
    return grad  # grad[6] (insertion) stays 0


# --- selective-gate runs + matched MPC comparators ---
sel_c = [rm for rm in manifest.RUNS if rm["group"] == "invjac_selective"]
sel_nc = [rm for rm in manifest.RUNS if rm["group"] == "invjac_clip_nc"]
mpc_c_255 = [rm for rm in manifest.RUNS if rm["group"] == "closedloop" and rm["condition"] == "mpc_C" and rm["radius_intended_mm"] == 255]
mpc_nc_255 = [rm for rm in manifest.RUNS if rm["group"] == "closedloop" and rm["condition"] == "mpc_NC" and rm["radius_intended_mm"] == 255]
mpc_c_255 = sorted(mpc_c_255, key=lambda r: r["rep"])
mpc_nc_255 = sorted(mpc_nc_255, key=lambda r: r["rep"])

print(f"sel_c={len(sel_c)} sel_nc={len(sel_nc)} mpc_c_255={len(mpc_c_255)} mpc_nc_255={len(mpc_nc_255)}")

PAIRINGS = [("selective_contact", sel_c, mpc_c_255), ("selective_nocontact", sel_nc, mpc_nc_255)]

decomp_rows = []
summary_rows = []
per_run_cache = {}

for pairing_name, runs, mpc_runs in PAIRINGS:
    for ridx, rm in enumerate(runs):
        out, meta = loader.enrich_run(rm)
        N = len(out["step"])
        q6 = out["q_meas_rad"]
        L = out["insertion_length_m"]
        s = out["s_ref_mm"]
        xyz = flange_xyz_batch(q6)
        margins = face_margins(xyz)
        min_margin = margins.min(axis=1)
        binding_face_idx = np.argmin(margins[-1])  # face at the actual stop tick
        binding_face = FACE_NAMES[binding_face_idx]
        # s at which the eventually-binding face's margin first drops under 5mm
        face_series = margins[:, binding_face_idx]
        under5 = np.where(face_series < 5.0)[0]
        s_bind_5mm = float(s[under5[0]]) if len(under5) else np.nan

        # matched MPC comparator (same rep index if available, else rep0)
        mpc_rm = mpc_runs[ridx] if ridx < len(mpc_runs) else mpc_runs[0]
        mpc_out, mpc_meta = loader.enrich_run(mpc_rm)
        mpc_s = mpc_out["s_ref_mm"]
        mpc_chi = np.concatenate([mpc_out["q_meas_rad"], mpc_out["insertion_length_m"][:, None]], axis=1)

        inv_chi = np.concatenate([q6, L[:, None]], axis=1)
        # interpolate MPC chi onto invjac's own s grid (clip to MPC's own s range)
        mpc_s_sorted_idx = np.argsort(mpc_s)
        mpc_s_sorted = mpc_s[mpc_s_sorted_idx]
        mpc_chi_sorted = mpc_chi[mpc_s_sorted_idx]
        mpc_chi_interp = np.zeros_like(inv_chi)
        for d in range(7):
            mpc_chi_interp[:, d] = np.interp(s, mpc_s_sorted, mpc_chi_sorted[:, d],
                                              left=mpc_chi_sorted[0, d], right=mpc_chi_sorted[-1, d])
        delta_chi = inv_chi - mpc_chi_interp

        # per-DOF divergence onset: early-region (s<20mm) noise floor -> 3x threshold
        early_mask = s < 20.0
        noise = np.std(delta_chi[early_mask], axis=0) if early_mask.sum() > 3 else np.full(7, 1e-6)
        noise = np.maximum(noise, 1e-6)
        onset_s = np.full(7, np.nan)
        for d in range(7):
            exceed = np.where(np.abs(delta_chi[:, d]) > 3 * noise[d])[0]
            exceed = exceed[s[exceed] > 20.0]  # ignore early-region false positives
            if len(exceed):
                onset_s[d] = float(s[exceed[0]])
        first_dof_idx = int(np.nanargmin(onset_s)) if np.any(~np.isnan(onset_s)) else -1
        first_dof = DOF_NAMES[first_dof_idx] if first_dof_idx >= 0 else "none"

        # margin-sensitivity decomposition near the failure window (last 15 ticks before stop)
        tail = slice(max(0, N - 15), N)
        dh_dchi_rows = []
        for k in range(tail.start, tail.stop):
            grad = flange_margin_gradient(q6[k], binding_face_idx)
            dh_i = grad * delta_chi[k]
            dh_dchi_rows.append(dh_i)
        dh_dchi_mean = np.mean(dh_dchi_rows, axis=0) if dh_dchi_rows else np.zeros(7)
        dominant_dof_idx = int(np.argmax(np.abs(dh_dchi_mean[:6])))  # exclude L (always 0 for this face)
        dominant_dof = DOF_NAMES[dominant_dof_idx]

        row = dict(
            pairing=pairing_name, dirname=meta["dirname"], rep=rm["rep"],
            stop_reason=meta["stop_reason"], final_s_mm=float(s[-1]),
            binding_face=binding_face, s_bind_under5mm_mm=s_bind_5mm,
            min_margin_at_stop_mm=float(min_margin[-1] * 1000.0),
            first_diverging_dof=first_dof, first_diverging_dof_onset_s_mm=float(onset_s[first_dof_idx]) if first_dof_idx >= 0 else np.nan,
            dominant_margin_sensitivity_dof=dominant_dof,
        )
        for d in range(7):
            row[f"onset_s_mm_{DOF_NAMES[d]}"] = float(onset_s[d]) if not np.isnan(onset_s[d]) else np.nan
            row[f"dh_dchi_mean_{DOF_NAMES[d]}"] = float(dh_dchi_mean[d])
        summary_rows.append(row)

        for k in range(N):
            decomp_rows.append(dict(
                pairing=pairing_name, dirname=meta["dirname"], rep=rm["rep"], tick=int(out["step"][k]),
                s_mm=float(s[k]), tcp_x=float(xyz[k, 0]), tcp_y=float(xyz[k, 1]), tcp_z=float(xyz[k, 2]),
                margin_x_min=float(margins[k, 0]), margin_x_max=float(margins[k, 1]),
                margin_y_min=float(margins[k, 2]), margin_y_max=float(margins[k, 3]),
                margin_z_min=float(margins[k, 4]), margin_z_max=float(margins[k, 5]),
                binding_face=binding_face, binding_face_margin_mm=float(face_series[k] * 1000.0),
                error_norm_mm=float(out["error_norm_mm"][k]),
                **{f"delta_chi_{DOF_NAMES[d]}": float(delta_chi[k, d]) for d in range(7)},
            ))

        per_run_cache[rm["dirname"]] = dict(out=out, s=s, xyz=xyz, margins=margins, binding_face_idx=binding_face_idx,
                                             delta_chi=delta_chi, mpc_dirname=mpc_meta["dirname"], mpc_s=mpc_s,
                                             mpc_out=mpc_out)
        print(f"[4A-4C] {meta['dirname']}: stop={meta['stop_reason']} final_s={s[-1]:.1f}mm "
              f"binding_face={binding_face} first_diverging_dof={first_dof}@{onset_s[first_dof_idx] if first_dof_idx>=0 else np.nan:.1f}mm "
              f"dominant_margin_sensitivity_dof={dominant_dof}")

decomp_df = pd.DataFrame(decomp_rows)
decomp_df.to_csv(f"{OUT}/tables/h3b_workspace_failure_decomposition.csv", index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/tables/h3b_workspace_failure_summary.csv", index=False)
print(f"\nwrote {OUT}/tables/h3b_workspace_failure_decomposition.csv ({len(decomp_df)} rows)")
print(f"wrote {OUT}/tables/h3b_workspace_failure_summary.csv ({len(summary_df)} rows)")
print(summary_df[["pairing", "dirname", "stop_reason", "final_s_mm", "binding_face",
                   "first_diverging_dof", "first_diverging_dof_onset_s_mm",
                   "dominant_margin_sensitivity_dof"]].to_string(index=False))

# =====================================================================
# 4D -- null-space projection (expensive: needs live_jacobian). Guided
# sample per run: early (s<30), onset region, pre-failure tail.
# =====================================================================
t0 = time.time()
live_jac.get_context()
print(f"\n[4D] live context ready at {time.time()-t0:.0f}s", flush=True)

proj_rows = []
for pairing_name, runs, mpc_runs in PAIRINGS:
    contact_model = (runs[0]["jacobian"] == "contact")
    for ridx, rm in enumerate(runs):
        cache = per_run_cache[rm["dirname"]]
        out, s = cache["out"], cache["s"]
        N = len(out["step"])
        early_idx = np.where(s < 30.0)[0]
        onset_s = summary_df[summary_df.dirname == rm["dirname"]]["first_diverging_dof_onset_s_mm"].iloc[0]
        onset_idx = np.where((s >= onset_s - 3) & (s <= onset_s + 3))[0] if not np.isnan(onset_s) else np.array([], dtype=int)
        tail_idx = np.arange(max(0, N - 12), N)
        pick = sorted(set(list(early_idx[::max(1, len(early_idx)//8)]) + list(onset_idx[::max(1, len(onset_idx)//5 or 1)]) + list(tail_idx)))

        q6 = out["q_meas_rad"]; L = out["insertion_length_m"]; u = out["u0"]
        for k in pick:
            state7 = np.concatenate([q6[k], [L[k]]])
            Jp = live_jac.live_jacobian(state7, contact=contact_model)  # (3,7)
            Jp_pinv = np.linalg.pinv(Jp)
            P_N = np.eye(7) - Jp_pinv @ Jp
            uk = u[k]
            u_N = P_N @ uk
            u_R = uk - u_N
            E_N = float(np.dot(u_N, u_N) / max(np.dot(uk, uk), 1e-18))
            proj_rows.append(dict(
                pairing=pairing_name, dirname=rm["dirname"], rep=rm["rep"], tick=int(out["step"][k]),
                s_mm=float(s[k]), region=("early" if k in early_idx else ("onset" if k in onset_idx else "pre_failure")),
                E_N=E_N, norm_uN=float(np.linalg.norm(u_N)), norm_uR=float(np.linalg.norm(u_R)),
                norm_u=float(np.linalg.norm(uk)),
            ))
        print(f"[4D] {rm['dirname']}: {len(pick)} samples done ({time.time()-t0:.0f}s elapsed)", flush=True)

proj_df = pd.DataFrame(proj_rows)
proj_df.to_csv(f"{OUT}/tables/h3b_null_redundant_projection.csv", index=False)
print(f"\nwrote {OUT}/tables/h3b_null_redundant_projection.csv ({len(proj_df)} rows)")
print(proj_df.groupby(["pairing", "region"])["E_N"].describe().to_string())

# =====================================================================
# Figures
# =====================================================================
fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex="col")
for col, (pairing_name, runs, mpc_runs) in enumerate(PAIRINGS):
    rm = runs[0]
    cache = per_run_cache[rm["dirname"]]
    s = cache["s"]; out = cache["out"]; margins = cache["margins"]
    binding_idx = cache["binding_face_idx"]
    delta_chi = cache["delta_chi"]
    row0 = summary_df[summary_df.dirname == rm["dirname"]].iloc[0]

    ax = axes[0, col]
    ax.plot(s, margins[:, binding_idx] * 1000.0, color="#b3331d", lw=1.3)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{pairing_name}: {rm['dirname'][:40]}...\nbinding face={row0.binding_face}", fontsize=8)
    ax.set_ylabel("binding-face\nmargin (mm)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, col]
    for d in range(6):
        ax.plot(s, delta_chi[:, d], lw=1.0, label=DOF_NAMES[d])
    ax.set_ylabel(r"$\delta\chi_q$ (rad)", fontsize=8)
    ax.legend(fontsize=6, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[2, col]
    ax.plot(s, delta_chi[:, 6] * 1000.0, color="#333333", lw=1.2)
    ax.set_ylabel(r"$\delta\chi_L$ (mm)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3, col]
    ax.plot(s, out["error_norm_mm"], color="#1d4fb3", lw=1.2)
    ax.set_ylabel("tip error (mm)", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[4, col]
    tcp = cache["xyz"]
    ax.plot(s, tcp[:, 0], label="x", lw=1.0)
    ax.plot(s, tcp[:, 1], label="y", lw=1.0)
    ax.plot(s, tcp[:, 2], label="z", lw=1.0)
    ax.axvline(row0.s_bind_under5mm_mm, color="red", ls=":", lw=1.0, label="margin<5mm")
    ax.set_ylabel("TCP xyz (m)", fontsize=8)
    ax.set_xlabel("path progress s (mm)", fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)

fig.suptitle("H3b workspace-drift mechanism: representative selective-gate run per pairing", fontsize=11)
fig.tight_layout()
f = f"{OUT}/figures/h3b_workspace_drift_mechanism.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print(f"\nsaved {f}")

fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
colors = {"selective_contact": "#1b7f3b", "selective_nocontact": "#b3331d"}
markers = {"early": "o", "onset": "s", "pre_failure": "^"}
for pairing_name in colors:
    sub = proj_df[proj_df.pairing == pairing_name]
    for region, mk in markers.items():
        rsub = sub[sub.region == region]
        ax.scatter(rsub.s_mm, rsub.E_N, color=colors[pairing_name], marker=mk, s=25, alpha=0.7,
                   label=f"{pairing_name} ({region})")
ax.set_xlabel("path progress s (mm)")
ax.set_ylabel(r"$E_N = \|u_N\|^2/\|u\|^2$  (fraction of command energy in the task null space)")
ax.set_title("H3b: null/redundant command-energy fraction vs path progress")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()
f = f"{OUT}/figures/h3b_redundant_projection.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print(f"saved {f}")
