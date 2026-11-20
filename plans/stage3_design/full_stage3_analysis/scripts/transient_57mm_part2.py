"""Follow-up diagnostic on the s~57mm contact-aware MPC tracking-error
transient (STAGE3_FINAL_REPORT.md Sec 1.4 / s57_targeted.py).

The prior targeted pass (s57_targeted.py) checked 8 candidates in s=54-60mm
and ruled all out: tracking error itself, one-step prediction error, all
three scheduled-Jacobian singular values, commanded-velocity norm,
tick-to-tick commanded-velocity change, the live exclusion-constraint
margin, commanded insertion rate, and scheduled-vs-live Jacobian mismatch.

This script checks 4 NEW categories, in a slightly wider s=53-61mm window,
across all 3 (non-excluded) reps of the scheduled contact-aware MPC
condition at 210mm:

  1A. Vision/measurement: raw vs filtered tip position, per-axis error,
      measurement age, tick timing.
  1B. Reference-path geometry: curvature, first/second derivatives of the
      desired position w.r.t. path progress, reference-index stepping.
  1C. Commanded vs executed robot motion: per-tick delta-chi command vs
      delta-chi reconstructed from consecutive measured states.
  1D. Contact-state evolution: beam centreline, per-node gap (penetration),
      contact force norm, and contact energy, recomputed directly from the
      contact model at each measured state (subsampled within the window --
      each full equilibrium re-solve costs ~5s).

Read-only with respect to all proper_research/** source and all hardware
logs. Writes only under plans/stage3_design/full_stage3_analysis/{tables,figures}/.
"""
import sys
import time
import contextlib
import io

sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest
import loader
import live_jac
from proper_research.simulation.simulations.controller_factory_joint_space import (
    _pose7_from_transform,
)

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
S_LO, S_HI = 53.0, 61.0
N_1D_SAMPLES_PER_REP = 8  # subsample for the expensive contact re-solve

runs = sorted(
    [rm for rm in manifest.RUNS if rm["group"] == "closedloop"
     and rm["condition"] == "mpc_C" and rm["radius_intended_mm"] == 210],
    key=lambda r: r["rep"],
)
print(f"[transient2] {len(runs)} reps: {[r['dirname'] for r in runs]}", flush=True)


def path_curvature_and_derivs(desired_position_m, path_s_m):
    """Per-sample ||dp/ds||, ||d2p/ds2||, and local curvature kappa, from the
    offline reference polyline (finite differences in arclength)."""
    p = np.asarray(desired_position_m, dtype=float)
    s = np.asarray(path_s_m, dtype=float)
    n = p.shape[0]
    dp_ds = np.gradient(p, s, axis=0)
    d2p_ds2 = np.gradient(dp_ds, s, axis=0)
    speed = np.linalg.norm(dp_ds, axis=1)
    # curvature kappa = |p' x p''| / |p'|^3 (3D curve curvature)
    cross = np.cross(dp_ds, d2p_ds2)
    kappa = np.linalg.norm(cross, axis=1) / np.maximum(speed ** 3, 1e-12)
    return dp_ds, d2p_ds2, np.linalg.norm(dp_ds, axis=1), np.linalg.norm(d2p_ds2, axis=1), kappa


def tangent_rotation_per_tick(tangent, idx):
    """Angle (deg) between consecutive reference-tangent samples at idx."""
    out = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        if i + 1 >= tangent.shape[0]:
            continue
        a, b = tangent[i], tangent[i + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        out[j] = np.degrees(np.arccos(c))
    return out


rows_1abc = []
rows_1d = []

t_start = time.time()
for rm in runs:
    out, meta = loader.enrich_run(rm)
    raw_rows = loader.load_path_follow(rm["path"])
    assert len(raw_rows) == len(out["step"]), "raw rows / enriched arrays length mismatch"

    s = out["s_ref_mm"]
    window = (s >= S_LO) & (s <= S_HI)
    idxs = np.where(window)[0]
    idxs = idxs[idxs < len(out["step"]) - 1]
    print(f"  rep{rm['rep']}: {len(idxs)} ticks in [{S_LO},{S_HI}]mm", flush=True)

    plan_dir = meta["plan_dir"]
    ref = loader.get_reference(plan_dir)
    desired_position_m = np.asarray(ref["desired_position_m"], dtype=float)
    path_s_m = np.asarray(ref["path_s_m"], dtype=float)
    desired_tangent = np.asarray(ref["desired_tangent"], dtype=float)
    dp_ds, d2p_ds2, speed_ref, accel_ref, kappa_ref = path_curvature_and_derivs(
        desired_position_m, path_s_m
    )

    for k in idxs:
        r = raw_rows[k]
        r1 = raw_rows[k + 1]
        tip_raw = np.asarray(r["tip_raw_mm"], dtype=float)
        tip_filt = np.asarray(r["tip_mm"], dtype=float)
        desired = np.asarray(r["desired_mm"], dtype=float)
        err_xyz = np.asarray(r["error_mm"], dtype=float)
        ref_idx = int(np.clip(r["ref_index"], 0, len(path_s_m) - 1))
        ref_idx_next = int(np.clip(r.get("target_index", ref_idx + 1), 0, len(path_s_m) - 1))

        # --- 1A: vision/measurement ---
        raw_minus_filt = tip_raw - tip_filt
        t_now = float(r["t_s"])
        t_prev = float(raw_rows[k - 1]["t_s"]) if k > 0 else np.nan
        dt_actual = t_now - t_prev if k > 0 else np.nan

        # --- 1B: reference-path geometry at this tick's reference index ---
        tan_rot_deg = np.nan
        if ref_idx + 1 < desired_tangent.shape[0]:
            a, b = desired_tangent[ref_idx], desired_tangent[ref_idx + 1]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
                tan_rot_deg = float(np.degrees(np.arccos(c)))
        ref_index_step = int(r.get("target_index", ref_idx) - r.get("ref_index", ref_idx))

        # --- 1C: commanded vs executed configuration motion ---
        dt_cmd = float(r.get("servo_ms", 100.0)) / 1000.0
        u0 = np.asarray(r["u0"], dtype=float)
        dchi_cmd = u0 * dt_cmd
        q_now = np.asarray(r["q_meas_rad"], dtype=float)
        L_now = float(r["insertion_length_m"])
        q_next = np.asarray(r1["q_meas_rad"], dtype=float)
        L_next = float(r1["insertion_length_m"])
        chi_now = np.concatenate([q_now, [L_now]])
        chi_next = np.concatenate([q_next, [L_next]])
        dchi_actual = chi_next - chi_now
        e_chi_vec = dchi_cmd - dchi_actual
        e_chi_norm = float(np.linalg.norm(e_chi_vec))

        row = dict(
            rep=rm["rep"], step=int(r["step"]), s_mm=float(s[k]),
            e_track_mm=float(out["error_norm_mm"][k]),
            # 1A
            tip_raw_x=tip_raw[0], tip_raw_y=tip_raw[1], tip_raw_z=tip_raw[2],
            tip_filt_x=tip_filt[0], tip_filt_y=tip_filt[1], tip_filt_z=tip_filt[2],
            raw_minus_filt_norm_mm=float(np.linalg.norm(raw_minus_filt)),
            desired_x=desired[0], desired_y=desired[1], desired_z=desired[2],
            e_x_mm=err_xyz[0], e_y_mm=err_xyz[1], e_z_mm=err_xyz[2],
            state_age_s=float(r.get("state_age_s", np.nan)),
            t_s=t_now, dt_actual_s=dt_actual, servo_ms=float(r.get("servo_ms", np.nan)),
            # 1B
            ref_index=ref_idx, target_index=int(r.get("target_index", ref_idx)),
            ref_index_step=ref_index_step,
            path_speed_ref_m_s=float(speed_ref[ref_idx]) if ref_idx < len(speed_ref) else np.nan,
            path_accel_ref_m_s2=float(accel_ref[ref_idx]) if ref_idx < len(accel_ref) else np.nan,
            kappa_ref_1_per_m=float(kappa_ref[ref_idx]) if ref_idx < len(kappa_ref) else np.nan,
            tangent_rotation_deg=tan_rot_deg,
            # 1C
            dchi_cmd_norm=float(np.linalg.norm(dchi_cmd)),
            dchi_actual_norm=float(np.linalg.norm(dchi_actual)),
            e_chi_norm=e_chi_norm,
            e_chi_q1=e_chi_vec[0], e_chi_q2=e_chi_vec[1], e_chi_q3=e_chi_vec[2],
            e_chi_q4=e_chi_vec[3], e_chi_q5=e_chi_vec[4], e_chi_q6=e_chi_vec[5],
            e_chi_L=e_chi_vec[6],
        )
        rows_1abc.append(row)
    print(f"    rep{rm['rep']} 1A/1B/1C done ({time.time()-t_start:.0f}s)", flush=True)

df_abc = pd.DataFrame(rows_1abc)
df_abc.to_csv(f"{OUT}/tables/transient_57mm_diagnostic_summary.csv", index=False)
print(f"[transient2] saved 1A/1B/1C table: {len(df_abc)} rows", flush=True)

# ---------------------------------------------------------------------
# 1D: contact-state re-solve (expensive) -- subsampled within the window
# ---------------------------------------------------------------------
print("[transient2] building live-model context for 1D...", flush=True)
live_jac.get_context()
prov = live_jac.get_provider(True)
model = prov.model

for rm in runs:
    out, meta = loader.enrich_run(rm)
    s = out["s_ref_mm"]
    window = (s >= S_LO) & (s <= S_HI)
    idxs_all = np.where(window)[0]
    if len(idxs_all) == 0:
        continue
    sel = np.unique(np.linspace(0, len(idxs_all) - 1, N_1D_SAMPLES_PER_REP).astype(int))
    idxs = idxs_all[sel]
    for k in idxs:
        state7 = np.concatenate([out["q_meas_rad"][k], [out["insertion_length_m"][k]]])
        t0 = time.time()
        T_R_M = prov.adapter.magnet_transform(state7)
        p7 = _pose7_from_transform(T_R_M, float(state7[6]))
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.solve(p7, commit=False, reuse_cache=False)
        parts = (result.info or {}).get("parts", {})
        gap_nodes = parts.get("gap_nodes")
        F_nodes = parts.get("F_nodes")
        gap_min = float(np.min(gap_nodes)) if gap_nodes is not None else np.nan
        n_penetrating = int(np.count_nonzero(np.asarray(gap_nodes) <= 0.0)) if gap_nodes is not None else -1
        contact_force_norm = float(np.linalg.norm(F_nodes)) if F_nodes is not None else np.nan
        row = dict(
            rep=rm["rep"], step=int(out["step"][k]), s_mm=float(s[k]),
            e_track_mm=float(out["error_norm_mm"][k]),
            gap_min_m=gap_min, n_penetrating_nodes=n_penetrating,
            contact_force_norm=contact_force_norm,
            W_cf=float(parts.get("W_cf", np.nan)),
            beam_n_nodes=int(result.p.shape[1]) if result.p is not None else -1,
            solve_wall_s=time.time() - t0,
        )
        rows_1d.append(row)
        print(f"    1D rep{rm['rep']} s={s[k]:.2f}mm gap_min={gap_min*1e3:.3f}mm "
              f"n_pen={n_penetrating} F={contact_force_norm:.3e} ({time.time()-t_start:.0f}s)", flush=True)
        pd.DataFrame(rows_1d).to_csv(f"{OUT}/tables/transient_57mm_1D_contact_state.csv", index=False)

df_1d = pd.DataFrame(rows_1d)
print(f"[transient2] saved 1D table: {len(df_1d)} rows", flush=True)

# ---------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------
colors = {2: "#1b7f3b", 3: "#2f8fd1", 4: "#b3331d"}
fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

ax = axes[0]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_track_mm"], color=colors[rep], lw=1.3, label=f"rep{rep}")
ax.set_ylabel("$e_{track}$ (mm)", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title("1A-1D: s=53-61mm deep diagnostic, scheduled MPC-$J_C$@210mm, all 3 reps", fontsize=11)

ax = axes[1]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_x_mm"], color=colors[rep], lw=1.0, ls="-", label=f"rep{rep} $e_x$" if rep == 2 else None)
    ax.plot(d["s_mm"], d["e_y_mm"], color=colors[rep], lw=1.0, ls="--", alpha=0.7)
    ax.plot(d["s_mm"], d["e_z_mm"], color=colors[rep], lw=1.0, ls=":", alpha=0.7)
ax.set_ylabel("1A: $e_x,e_y,e_z$ (mm)\n(solid/dash/dot)", fontsize=8)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[2]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["raw_minus_filt_norm_mm"], color=colors[rep], lw=1.2, marker="o", ms=3)
ax.set_ylabel("1A: |raw-filt tip| (mm)", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[3]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["kappa_ref_1_per_m"], color=colors[rep], lw=1.0, ls="-", label="$\\kappa$" if rep == 2 else None)
ax.set_ylabel("1B: ref curvature $\\kappa$ (1/m)", fontsize=8)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

ax = axes[4]
for rep, d in df_abc.groupby("rep"):
    d = d.sort_values("s_mm")
    ax.plot(d["s_mm"], d["e_chi_norm"], color=colors[rep], lw=1.2, marker="o", ms=3)
ax.set_ylabel("1C: $\\|\\Delta\\chi_{cmd}-\\Delta\\chi_{actual}\\|$", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[5]
if len(df_1d):
    for rep, d in df_1d.groupby("rep"):
        d = d.sort_values("s_mm")
        ax.plot(d["s_mm"], np.asarray(d["gap_min_m"]) * 1e3, color=colors[rep], lw=1.2, marker="s", ms=4,
                label=f"rep{rep} gap_min" if rep == df_1d["rep"].min() else None)
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.legend(fontsize=7)
ax.set_ylabel("1D: min gap (mm)\n(<=0 = penetrating)", fontsize=8)
ax.set_xlabel("path progress s (mm)")
ax.grid(alpha=0.3)

fig.tight_layout()
f = f"{OUT}/figures/transient_57mm_deep_diagnostic.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"[transient2] saved figure: {f}")
print(f"[transient2] DONE in {time.time()-t_start:.0f}s", flush=True)
