"""Lag-chronology analysis for the 6 selective-gate inverse-Jacobian runs
(255mm floor): tests whether the proposed causal chain

    e_lag up -> ||e|| up -> ||u_raw|| up -> E_N up -> q2/q1 drift -> h_xmax down

actually holds in that order, and whether J_C vs J_NC selective pairings
develop substantially different lag (the key H3b-common-cause question,
since BOTH pairings fail 0/3 on the same tcp_out_of_workspace/x_max face).

Reuses: loader.enrich_run (e_lag_mm, cross_track_mm, error_norm_mm,
q_meas_rad already computed there); h3_replay.py's raw-command damped-
least-squares math (cheap, no live model) for ||u_raw||; live_jac.py's
live_jacobian (expensive contact-model call) for E_N, on a moderate stride
across the FULL run (not just two narrow regions, unlike the existing
h3b_pipeline_stage_nullspace.csv); the same flange-TCP-margin convention as
h3b_workspace_and_nullspace.py for h_xmax.

Read-only against hardware logs/model/controller code; writes only new
table/figure files.
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
REPO = "/home/jack/Proper-Research"
SCHED_C = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_2026-10-06_repaired.npy"
SCHED_NC = f"{REPO}/plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_2026-10-06.npy"
DT = 0.1
CONTACT_ONSET_S_MM = 28.5
I4 = np.eye(4)
WS_MIN = np.array([0.277, -0.832, 0.170])
WS_MAX = np.array([0.656, -0.200, 0.433])
EN_STRIDE = 4  # every 4th tick, full s-range, for the expensive live-model call


def flange_xyz_batch(q6_batch):
    out = np.zeros((q6_batch.shape[0], 3))
    for i in range(q6_batch.shape[0]):
        T = urik.forward_kinematics(q6_batch[i], loader._ROBOT_KIN.dh, I4)
        out[i] = T.T_R_target[:3, 3]
    return out


def margin_x_max(xyz):
    return WS_MAX[0] - xyz[:, 0]


def replay_u_raw(out, meta, schedule, desired_position_m):
    """Damped-least-squares task-space solution, BEFORE any clip -- exact
    math from h3_replay.py's replay_raw_commands up to command_preclip."""
    N = len(out["step"])
    position_gain = float(meta["position_gain"])
    damping = float(meta["damping"])
    n_sched = schedule.shape[0]
    n_ref = desired_position_m.shape[0]
    q = out["q_meas_rad"]; L = out["insertion_length_m"]; tip_m = out["tip_mm"] / 1000.0
    ref_idx = out["ref_index"]
    u_raw = np.zeros((N, 7))
    for k in range(N):
        measured = tip_m[k]
        index = int(np.clip(ref_idx[k] + 1, 0, min(n_sched, n_ref) - 1))
        J = schedule[index]
        desired = desired_position_m[index]
        gram = J @ J.T + (damping ** 2) * np.eye(3)
        pseudo = J.T @ np.linalg.solve(gram, np.eye(3))
        u_raw[k] = pseudo @ (position_gain * (desired - measured) / DT)
    return u_raw


def onset_s(s, series, early_cut=20.0, k_sigma=3.0, min_persist=5):
    """First s (beyond early_cut) at which |series - early_mean| exceeds
    k_sigma * early_std for >= min_persist consecutive available samples."""
    early = s < early_cut
    if early.sum() < 4:
        return np.nan
    mu, sd = np.mean(series[early]), max(np.std(series[early]), 1e-9)
    dev = np.abs(series - mu) > k_sigma * sd
    dev = dev & (s > early_cut)
    N = len(dev)
    for i in range(N - min_persist + 1):
        if dev[i:i + min_persist].all():
            return float(s[i])
    return np.nan


sel_c = sorted([rm for rm in manifest.RUNS if rm["group"] == "invjac_selective"], key=lambda r: r["rep"])
sel_nc = sorted([rm for rm in manifest.RUNS if rm["group"] == "invjac_clip_nc"], key=lambda r: r["rep"])
PAIRINGS = [("selective_contact", sel_c, True, SCHED_C), ("selective_nocontact", sel_nc, False, SCHED_NC)]

wf_summary = pd.read_csv(f"{OUT}/tables/h3b_workspace_failure_summary.csv")

per_tick_rows = []
en_rows = []
onset_rows = []
cache = {}

t0 = time.time()
for pairing_name, runs, is_contact, sched_path in PAIRINGS:
    schedule = np.load(sched_path)
    for rm in runs:
        out, meta = loader.enrich_run(rm)
        ref = loader.get_reference(meta["plan_dir"])
        desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)

        N = len(out["step"])
        s = out["s_ref_mm"]
        e_lag = out["e_lag_mm"]
        e_perp = out["cross_track_mm"]
        err = out["error_norm_mm"]
        q1 = out["q_meas_rad"][:, 0]
        q2 = out["q_meas_rad"][:, 1]
        xyz = flange_xyz_batch(out["q_meas_rad"])
        hxmax = margin_x_max(xyz) * 1000.0  # mm

        u_raw = replay_u_raw(out, meta, schedule, desired_position_m)
        u_raw_norm = np.linalg.norm(u_raw, axis=1)

        for k in range(N):
            per_tick_rows.append(dict(
                dirname=meta["dirname"], pairing=pairing_name, rep=rm["rep"], tick=int(out["step"][k]),
                s_mm=float(s[k]), e_lag_mm=float(e_lag[k]), e_perp_mm=float(e_perp[k]),
                error_norm_mm=float(err[k]), u_raw_norm=float(u_raw_norm[k]),
                q1_rad=float(q1[k]), q2_rad=float(q2[k]), margin_x_max_mm=float(hxmax[k]),
            ))

        # onsets on full-resolution series
        o_lag = onset_s(s, e_lag)
        o_err = onset_s(s, err)
        o_uraw = onset_s(s, u_raw_norm)
        o_q1 = onset_s(s, q1)
        o_q2 = onset_s(s, q2)
        under20 = np.where(hxmax < 20.0)[0]
        s_under20 = float(s[under20[0]]) if len(under20) else np.nan
        under0 = np.where(hxmax < 0.0)[0]
        s_exit = float(s[under0[0]]) if len(under0) else float(s[-1])

        cache[rm["dirname"]] = dict(out=out, s=s, e_lag=e_lag, e_perp=e_perp, err=err,
                                     u_raw=u_raw, u_raw_norm=u_raw_norm, q1=q1, q2=q2,
                                     hxmax=hxmax, is_contact=is_contact)

        # strided E_N using live contact model, on u_raw, full s-range
        idxs = np.arange(0, N, EN_STRIDE)
        for k in idxs:
            state7 = np.concatenate([out["q_meas_rad"][k], [out["insertion_length_m"][k]]])
            Jp = live_jac.live_jacobian(state7, contact=is_contact)
            Jp_pinv = np.linalg.pinv(Jp)
            P_N = np.eye(7) - Jp_pinv @ Jp
            uk = u_raw[k]
            u_N = P_N @ uk
            E_N = float(np.dot(u_N, u_N) / max(np.dot(uk, uk), 1e-18))
            en_rows.append(dict(dirname=meta["dirname"], pairing=pairing_name, rep=rm["rep"],
                                 tick=int(out["step"][k]), s_mm=float(s[k]), E_N=E_N,
                                 norm_uN=float(np.linalg.norm(u_N)), norm_uraw=float(np.linalg.norm(uk))))

        onset_rows.append(dict(
            dirname=meta["dirname"], pairing=pairing_name, rep=rm["rep"],
            onset_s_e_lag=o_lag, onset_s_err=o_err, onset_s_u_raw=o_uraw,
            onset_s_q1=o_q1, onset_s_q2=o_q2,
            s_hxmax_under20mm=s_under20, s_hxmax_exit=s_exit,
            mean_e_lag_mm=float(np.mean(e_lag)), median_e_lag_mm=float(np.median(e_lag)),
            final_e_lag_mm=float(e_lag[-1]), max_e_lag_mm=float(np.max(e_lag)),
            mean_e_perp_mm=float(np.mean(e_perp)), median_e_perp_mm=float(np.median(e_perp)),
            final_e_perp_mm=float(e_perp[-1]), max_e_perp_mm=float(np.max(e_perp)),
        ))
        print(f"[{pairing_name}] {meta['dirname']}: N={N} onset(lag={o_lag},err={o_err},"
              f"u_raw={o_uraw},q1={o_q1},q2={o_q2}) hxmax<20mm@{s_under20} exit@{s_exit} "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)

per_tick_df = pd.DataFrame(per_tick_rows)
per_tick_df.to_csv(f"{OUT}/tables/lag_chronology_per_tick.csv", index=False)
en_df = pd.DataFrame(en_rows)
en_df.to_csv(f"{OUT}/tables/lag_chronology_EN_strided.csv", index=False)
onset_df = pd.DataFrame(onset_rows)

# E_N onset (on its own stride) appended to onset_df
en_onset = []
for dirname, g in en_df.groupby("dirname"):
    g = g.sort_values("s_mm")
    o = onset_s(g.s_mm.to_numpy(), g.E_N.to_numpy())
    en_onset.append(dict(dirname=dirname, onset_s_E_N=o))
en_onset_df = pd.DataFrame(en_onset)
onset_df = onset_df.merge(en_onset_df, on="dirname", how="left")
onset_df = onset_df.merge(wf_summary[["dirname", "final_s_mm", "binding_face"]], on="dirname", how="left")
onset_df.to_csv(f"{OUT}/tables/lag_chronology_onset_summary.csv", index=False)
print(f"\nwrote {OUT}/tables/lag_chronology_per_tick.csv ({len(per_tick_df)} rows)")
print(f"wrote {OUT}/tables/lag_chronology_EN_strided.csv ({len(en_df)} rows)")
print(f"wrote {OUT}/tables/lag_chronology_onset_summary.csv ({len(onset_df)} rows)")

pd.set_option("display.width", 250)
print("\n=== onset summary ===")
print(onset_df[["pairing", "rep", "onset_s_e_lag", "onset_s_err", "onset_s_u_raw", "onset_s_E_N",
                 "onset_s_q2", "onset_s_q1", "s_hxmax_under20mm", "s_hxmax_exit"]].to_string(index=False))

print("\n=== mean onset per pairing ===")
print(onset_df.groupby("pairing")[["onset_s_e_lag", "onset_s_err", "onset_s_u_raw", "onset_s_E_N",
                                     "onset_s_q2", "onset_s_q1", "s_hxmax_under20mm", "s_hxmax_exit"]].mean().to_string())

print("\n=== lag/cross-track comparison, J_C vs J_NC ===")
print(onset_df.groupby("pairing")[["mean_e_lag_mm", "median_e_lag_mm", "final_e_lag_mm", "max_e_lag_mm",
                                     "mean_e_perp_mm", "median_e_perp_mm", "final_e_perp_mm", "max_e_perp_mm"]].mean().to_string())

# =====================================================================
# combined figure
# =====================================================================
COLORS = {"selective_contact": "#1b7f3b", "selective_nocontact": "#b3331d"}
fig, axes = plt.subplots(8, 1, figsize=(11, 22), sharex=True)
panels = [
    ("e_lag", r"$e_{lag}$ (mm)"),
    ("e_perp", r"$e_\perp$ (mm)"),
    ("err", r"$\|e\|$ (mm)"),
    ("u_raw_norm", r"$\|u_{raw}\|$"),
    ("EN", r"$E_N$"),
    ("q1", r"$q_1$ (rad)"),
    ("q2", r"$q_2$ (rad)"),
    ("hxmax", r"$h_{x_{max}}$ (mm)"),
]
for ax, (key, ylabel) in zip(axes, panels):
    for pairing_name, runs, is_contact, sched_path in PAIRINGS:
        color = COLORS[pairing_name]
        for i, rm in enumerate(runs):
            c = cache[rm["dirname"]]
            lab = pairing_name if i == 0 else None
            if key == "EN":
                sub = en_df[en_df.dirname == rm["dirname"]].sort_values("s_mm")
                ax.plot(sub.s_mm, sub.E_N, color=color, lw=1.0, alpha=0.8, marker="o", ms=2, label=lab)
            else:
                ax.plot(c["s"], c[key], color=color, lw=1.0, alpha=0.8, label=lab)
        exit_s = wf_summary[wf_summary.dirname.isin([r["dirname"] for r in runs])]["final_s_mm"]
        for es in exit_s:
            ax.axvline(es, color=color, lw=0.6, ls=":", alpha=0.6)
    ax.axvline(CONTACT_ONSET_S_MM, color="k", lw=0.8, ls="--", alpha=0.6)
    if key == "hxmax":
        ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
axes[0].set_title("Lag chronology: selective-gate inverse-Jacobian, contact vs no-contact (dashed=contact onset s=28.5mm, dotted=each rep's own workspace exit)", fontsize=9)
axes[-1].set_xlabel("path progress s (mm)")
fig.tight_layout()
f = f"{OUT}/figures/lag_chronology_combined.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print(f"\nsaved {f}")
