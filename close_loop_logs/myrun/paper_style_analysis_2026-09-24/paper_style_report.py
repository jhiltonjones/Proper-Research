"""Paper-style figures for the raised (+30mm) rectangle and wide triangle
z-shift study, reconstructed in the style of the existing paper package
figures (paper_package_2026-09-21/00_main/Figure1_spatial_path_tracking.png,
Figure3_FJ_SJ_robustness.png, Figure4_rectangle_mechanism.png) -- NOT
byte-for-byte reproductions (that generating script wasn't preserved, same
situation as mechanism_figure_zraise.py earlier this session), but built to
match each reference figure's own panel layout/axis conventions, using
TODAY's raised-workspace data only (U-shape excluded -- its FJ/SJ grid
was not completed today).

Produces, into this directory:
  fig1_spatial_path_tracking.png   -- task-plane x/y trajectory, SJ runs,
                                       Desired vs INV-7 vs MPC, per shape
  fig3_fj_sj_robustness.png        -- RMS error, Frozen-J vs Scheduled-J,
                                       per controller, per shape
  fig4_mechanism_rectangle.png     -- 6-panel mechanism figure (rectangle)
  fig4_mechanism_triangle.png      -- 6-panel mechanism figure (triangle)

Metric definitions that are NOT a direct byte-for-byte match to the
original (reconstructed from the reference figures' own axis labels):
  - Fig4 Panel C "Jacobian-row rotation": angle between J_FJ's lateral (y)
    ROW (all 7 columns) and the genuine schedule's y-row at that ref_index
    -- a single-vector angle, not the whole-matrix principal angle used
    elsewhere in this session's diagnostics. Chosen because Panel C is
    specifically about LATERAL tracking, and the y-row is what lateral
    tracking actually draws on.
  - Fig4 Panel C "lateral-error gap": mean_FJ(|e_y| INV-7) - mean_FJ(|e_y|
    MPC) at each ref_index, i.e. how much worse INV-7's lateral tracking
    is than MPC's under the frozen model, at each point along the path.
  - Fig4 D1 "useful lateral corrections": fraction of ticks where the
    commanded correction's task-space y-contribution (via the schedule
    Jacobian) is sign-aligned with reducing the current e_y (matches this
    session's useful_lateral_and_efficiency.py per-tick criterion,
    collapsed to a single summary fraction per condition).
  - Fig4 D2 "net correction efficiency": final cumulative value of
    (realized |e_y| reduction at +3 ticks) / (normalized combined
    feedback effort), same definition as useful_lateral_and_efficiency.py
    Panel B, reported as a single per-rep summary number per condition.

Self-contained: run with
    python3 close_loop_logs/myrun/paper_style_analysis_2026-09-24/paper_style_report.py
from the repo root.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

OUTDIR = os.path.dirname(os.path.abspath(__file__))
DT = 0.1
HORIZON = 3
S_U_Q = 0.05
S_U_L = 0.005

RECT_DIR = "close_loop_logs/myrun"
TRI_DIR = "close_loop_logs/myrun"

SHAPES = {
    "Rectangle": dict(
        plan_dir="plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path",
        sj_schedule="/tmp/rectangle_zraise30mm_schedule_accurate.npy",
        fj_schedule="/tmp/rectangle_zraise30mm_schedule_FJ.npy",
        mpc_sj=[
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep2_20260924T113512Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep4_20260924T113802Z",
            f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_rep5_20260924T114043Z",
        ],
        mpc_fj=[
            (f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep1_20260924T134324Z", "tcp_out_of_workspace"),
            (f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep2_20260924T140357Z", "tcp_out_of_workspace"),
            (f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep3_20260924T140528Z", None),
            (f"{RECT_DIR}/mpc_delay_aware_insertion_anchor_zraise30mm_FJ_rep4_20260924T140647Z", "insertion_offset_exceeded"),
        ],
        inv_sj=[
            f"{RECT_DIR}/inv_7dof_delay_aware_zraise30mm_20260924T103744Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep2_20260924T111308Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep3_20260924T111434Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep4_20260924T111548Z",
            f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_rep5_20260924T111702Z",
        ],
        inv_fj=[
            (f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep1_20260924T140105Z", None),
            (f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep2_20260924T140829Z", None),
            (f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep3_20260924T140945Z", None),
            (f"{RECT_DIR}/inv_7dof_delay_aware_kp1_zraise30mm_FJ_rep4_20260924T141059Z", None),
        ],
    ),
    "Triangle (25 mm)": dict(
        plan_dir="plans/triangle_base25mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path",
        sj_schedule="/tmp/triangle_wide_zraise30mm_schedule_SJ.npy",
        fj_schedule="/tmp/triangle_wide_zraise30mm_schedule_FJ.npy",
        mpc_sj=[
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep1_20260924T162512Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep2_20260924T162637Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep3_20260924T162805Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_SJ_rep4_20260924T162939Z",
        ],
        mpc_fj=[
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_FJ_rep1_20260924T163100Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_FJ_rep2_20260924T163224Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_FJ_rep3_20260924T164720Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_mpc_FJ_rep4_20260924T163401Z", None),
        ],
        inv_sj=[
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep1_20260924T164854Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep2_20260924T163610Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep3_20260924T163730Z",
            f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_SJ_rep4_20260924T163855Z",
        ],
        inv_fj=[
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_FJ_rep1_20260924T164025Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_FJ_rep2_20260924T164151Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_FJ_rep3_20260924T164313Z", None),
            (f"{TRI_DIR}/triangle_wide_zraise30mm_closed_inv7_FJ_rep4_20260924T164433Z", None),
        ],
    ),
}


def load_rows(rundir):
    rows = [json.loads(l) for l in open(f"{rundir}/path_follow.jsonl")]
    rows.sort(key=lambda r: r["step"])
    return rows


def by_ref_index(rows):
    out = {}
    for r in rows:
        ri = r["ref_index"]
        if ri not in out:
            out[ri] = r
    return out


def full_rms(rows):
    vals = [r["error_norm_mm"] for r in rows]
    return float(np.sqrt(np.mean(np.square(vals))))


def principal_angles_deg(A, B):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    _, s, _ = np.linalg.svd(Qa.T @ Qb)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


# =======================================================================
# FIGURE 1: spatial path tracking (task-plane x/y), SJ runs
# =======================================================================
fig1, axes1 = plt.subplots(1, len(SHAPES), figsize=(7 * len(SHAPES), 5.5))
if len(SHAPES) == 1:
    axes1 = [axes1]

for ax, (shape_name, cfg) in zip(axes1, SHAPES.items()):
    all_mpc, all_inv = [], []
    desired_ref = None
    for label, runs, color in [("INV-7", cfg["inv_sj"], "tab:green"), ("MPC", cfg["mpc_sj"], "tab:blue")]:
        per_rep_xy = []
        for rd in runs:
            rows = load_rows(rd)
            by_ri = by_ref_index(rows)
            idxs = sorted(by_ri.keys())
            x0, y0 = by_ri[idxs[0]]["desired_mm"][0], by_ri[idxs[0]]["desired_mm"][1]
            xs = np.array([by_ri[ri]["tip_mm"][0] - x0 for ri in idxs])
            ys = np.array([by_ri[ri]["tip_mm"][1] - y0 for ri in idxs])
            ax.plot(xs, ys, color=color, alpha=0.25, lw=0.8)
            per_rep_xy.append((idxs, xs, ys))
            if desired_ref is None or len(idxs) > len(desired_ref[0]):
                dxs = np.array([by_ri[ri]["desired_mm"][0] - x0 for ri in idxs])
                dys = np.array([by_ri[ri]["desired_mm"][1] - y0 for ri in idxs])
                desired_ref = (idxs, dxs, dys)
        # mean trajectory across reps, aligned by ref_index
        common_idx = sorted(set.intersection(*[set(i) for i, _, _ in per_rep_xy]))
        mean_x = np.array([np.mean([xs[list(idxs).index(ri)] for idxs, xs, ys in per_rep_xy]) for ri in common_idx])
        mean_y = np.array([np.mean([ys[list(idxs).index(ri)] for idxs, xs, ys in per_rep_xy]) for ri in common_idx])
        ax.plot(mean_x, mean_y, color=color, lw=2.0, label=label)
        (all_mpc if label == "MPC" else all_inv).append((mean_x, mean_y))

    didx, dxs, dys = desired_ref
    ax.plot(dxs, dys, "k--", lw=1.5, label="Desired")
    ax.scatter([dxs[0]], [dys[0]], color="black", marker="o", s=70, zorder=6, label="Start")
    ax.scatter([dxs[-1]], [dys[-1]], color="black", marker="s", s=70, zorder=6, label="End")
    ax.set_title(shape_name)
    ax.set_xlabel("Task-plane x (mm)")
    ax.set_ylabel("Task-plane y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
fig1.suptitle("Task-plane spatial path tracking (SJ), raised +30mm workspace, today's data", fontsize=13)
fig1.tight_layout(rect=[0, 0.04, 1, 0.95])
fig1.savefig(f"{OUTDIR}/fig1_spatial_path_tracking.png", dpi=150)
plt.close(fig1)
print(f"saved -> {OUTDIR}/fig1_spatial_path_tracking.png")

# =======================================================================
# FIGURE 3: FJ vs SJ RMS robustness
# =======================================================================
fig3, axes3 = plt.subplots(1, len(SHAPES), figsize=(7 * len(SHAPES), 5.5))
if len(SHAPES) == 1:
    axes3 = [axes3]

fj_sj_summary = {}
for ax, (shape_name, cfg) in zip(axes3, SHAPES.items()):
    shape_summary = {}
    for label, sj_runs, fj_runs, color in [
        ("INV-7", cfg["inv_sj"], cfg["inv_fj"], "tab:green"),
        ("MPC", cfg["mpc_sj"], cfg["mpc_fj"], "tab:blue"),
    ]:
        sj_rms = [full_rms(load_rows(rd)) for rd in sj_runs]
        fj_rms = [full_rms(load_rows(rd)) for rd, _ in fj_runs]
        n_abort = sum(1 for _, reason in fj_runs if reason is not None)
        shape_summary[label] = dict(sj_rms=sj_rms, fj_rms=fj_rms, n_abort=n_abort, n_fj=len(fj_runs))
        means = [np.mean(fj_rms), np.mean(sj_rms)]
        sds = [np.std(fj_rms), np.std(sj_rms)]
        ax.errorbar([0, 1], means, yerr=sds, marker="o", markersize=10, capsize=5,
                    color=color, label=label, lw=2.0)
        ax.scatter(np.zeros(len(fj_rms)) + np.random.uniform(-0.03, 0.03, len(fj_rms)), fj_rms,
                   color=color, alpha=0.4, s=25, zorder=5)
        ax.scatter(np.ones(len(sj_rms)) + np.random.uniform(-0.03, 0.03, len(sj_rms)), sj_rms,
                   color=color, alpha=0.4, s=25, zorder=5)
        if n_abort > 0:
            ax.annotate(f"{label}-FJ: {n_abort}/{len(fj_runs)}\nsafety-terminated",
                        xy=(0, means[0]), xytext=(0.15, means[0] * 1.15 if label == "MPC" else means[0] * 0.7),
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.9))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Frozen J", "Scheduled J"])
    ax.set_ylabel("RMS error (mm)")
    ax.set_title(shape_name)
    ax.legend()
    ax.grid(alpha=0.3)
    fj_sj_summary[shape_name] = shape_summary

fig3.suptitle("Robustness to Jacobian staleness: Frozen-J vs Scheduled-J RMS, raised +30mm workspace", fontsize=13)
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(f"{OUTDIR}/fig3_fj_sj_robustness.png", dpi=150)
plt.close(fig3)
print(f"saved -> {OUTDIR}/fig3_fj_sj_robustness.png")

for shape_name, s in fj_sj_summary.items():
    print(f"\n{shape_name}:")
    for label, d in s.items():
        print(f"  {label}: FJ mean={np.mean(d['fj_rms']):.3f}+/-{np.std(d['fj_rms']):.3f}mm "
              f"(n={d['n_fj']}, {d['n_abort']} aborted)  SJ mean={np.mean(d['sj_rms']):.3f}+/-{np.std(d['sj_rms']):.3f}mm")

# =======================================================================
# FIGURE 4: 6-panel mechanism figure, per shape
# =======================================================================
def progress_pct(rows, n_ref):
    return np.array([min(r["ref_index"], n_ref - 1) / (n_ref - 1) * 100.0 for r in rows])


def mean_sd_by_progress(runs, n_ref, n_grid=150):
    grid = np.linspace(0, 100, n_grid)
    curves = []
    for rd in runs:
        rows = load_rows(rd)
        by_ri = by_ref_index(rows)
        idxs = sorted(by_ri.keys())
        pct = np.array([min(ri, n_ref - 1) / (n_ref - 1) * 100.0 for ri in idxs])
        err = np.array([by_ri[ri]["error_norm_mm"] for ri in idxs])
        curves.append(np.interp(grid, pct, err))
    curves = np.array(curves)
    return grid, curves.mean(axis=0), curves.std(axis=0)


def ey_mean_sd_by_progress(runs, n_ref, n_grid=150):
    grid = np.linspace(0, 100, n_grid)
    curves = []
    for rd, _ in runs:
        rows = load_rows(rd)
        by_ri = by_ref_index(rows)
        idxs = sorted(by_ri.keys())
        pct = np.array([min(ri, n_ref - 1) / (n_ref - 1) * 100.0 for ri in idxs])
        ey = np.array([abs(by_ri[ri]["error_mm"][1]) for ri in idxs])
        curves.append(np.interp(grid, pct, ey))
    curves = np.array(curves)
    return grid, curves.mean(axis=0)


def useful_lateral_and_efficiency(runs, schedule, input_reference, n_ref, is_fj_list):
    """Returns per-rep (useful_frac, final_efficiency) lists."""
    useful_fracs, final_effs = [], []
    for item in runs:
        rd = item[0] if is_fj_list else item
        rows = load_rows(rd)
        by_step_map = {r["step"]: r for r in rows}
        steps = sorted(by_step_map.keys())
        useful_flags, effort, dE_actual = [], [], []
        for k in steps:
            r = by_step_map[k]
            ref_now = min(r["ref_index"], n_ref - 1)
            u0 = np.asarray(r["u0"], dtype=float)
            du = u0 - input_reference[ref_now]
            J = schedule[ref_now]
            dp_fb_mm = (J @ du) * DT * 1000.0
            e_y = r["error_mm"][1]
            useful_flags.append(1.0 if (e_y != 0 and np.sign(e_y) * dp_fb_mm[1] > 0) else 0.0)
            du_norm_q = du[:6] / S_U_Q
            du_norm_L = du[6] / S_U_L
            effort.append(float(np.sqrt(np.sum(du_norm_q ** 2) + du_norm_L ** 2)))
            future = by_step_map.get(k + HORIZON)
            dE_actual.append(r["error_norm_mm"] - future["error_norm_mm"] if future is not None else np.nan)
        useful_flags = np.array(useful_flags)
        useful_fracs.append(float(useful_flags.mean()))
        effort = np.array(effort)
        dE_actual = np.array(dE_actual)
        valid = ~np.isnan(dE_actual)
        cum_reduction = np.cumsum(np.where(valid, dE_actual, 0.0))
        cum_effort = np.cumsum(effort)
        BURN_IN = 5
        final_eff = cum_reduction[-1] / cum_effort[-1] if len(cum_effort) > BURN_IN else np.nan
        final_effs.append(float(final_eff))
    return useful_fracs, final_effs


for shape_name, cfg in SHAPES.items():
    plan = np.load(glob.glob(cfg["plan_dir"] + "/*.npz")[0])
    desired_tangent = plan["desired_tangent"]
    input_reference = plan["input_reference"]
    N_REF = plan["state_reference"].shape[0]
    J_SJ_schedule = np.load(cfg["sj_schedule"])
    J_FJ = np.load(cfg["fj_schedule"])[0]

    dot = np.clip(np.sum(desired_tangent[:-1] * desired_tangent[1:], axis=1), -1, 1)
    ang = np.degrees(np.arccos(dot))
    corner_idx, _ = find_peaks(ang, height=3, distance=10)
    corner_pct = corner_idx / (N_REF - 1) * 100.0

    fig4, axes4 = plt.subplots(3, 2, figsize=(14, 13))
    fig4.suptitle(f"Mechanism figure ({shape_name}), raised +30mm workspace, today's data", fontsize=13)

    # Panel A: Scheduled-J tracking
    ax = axes4[0, 0]
    for label, runs, color in [("MPC", cfg["mpc_sj"], "tab:blue"), ("INV-7", cfg["inv_sj"], "tab:green")]:
        grid, mean, sd = mean_sd_by_progress(runs, N_REF)
        ax.plot(grid, mean, color=color, label=label)
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.2)
    for cp in corner_pct:
        ax.axvline(cp, color="gray", alpha=0.4)
    ax.set_title("A. Scheduled-J tracking", fontsize=10)
    ax.set_xlabel("path progress (%)"); ax.set_ylabel("|e| (mm)")
    ax.legend()

    # Panel B: Frozen-J tracking
    ax = axes4[0, 1]
    for label, runs, color in [("MPC", cfg["mpc_fj"], "tab:blue"), ("INV-7", cfg["inv_fj"], "tab:green")]:
        grid, mean, sd = mean_sd_by_progress([rd for rd, _ in runs], N_REF)
        ax.plot(grid, mean, color=color, label=label)
        ax.fill_between(grid, mean - sd, mean + sd, color=color, alpha=0.2)
    for cp in corner_pct:
        ax.axvline(cp, color="gray", alpha=0.4)
    ax.set_title("B. Frozen-J tracking", fontsize=10)
    ax.set_xlabel("path progress (%)"); ax.set_ylabel("|e| (mm)")
    ax.legend()

    # Panel C: Frozen-J lateral Jacobian rotation + lateral-error gap
    ax = axes4[1, 0]
    row_rot = np.array([
        np.degrees(np.arccos(np.clip(
            np.dot(J_FJ[1, :], J_SJ_schedule[ri][1, :]) /
            (np.linalg.norm(J_FJ[1, :]) * np.linalg.norm(J_SJ_schedule[ri][1, :]) + 1e-12), -1, 1)))
        for ri in range(N_REF)
    ])
    row_rot_pct = np.linspace(0, 100, N_REF)
    ax.plot(row_rot_pct, row_rot, color="tab:red", label="Jacobian-row rotation")
    ax.set_xlabel("path progress (%)")
    ax.set_ylabel("joint-to-y row rotation (deg)", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax.twinx()
    grid_mpc, ey_mpc = ey_mean_sd_by_progress(cfg["mpc_fj"], N_REF)
    grid_inv, ey_inv = ey_mean_sd_by_progress(cfg["inv_fj"], N_REF)
    gap = ey_inv - ey_mpc
    ax2.plot(grid_mpc, gap, color="gray", linestyle=":", label="lateral-error gap (INV-7 - MPC)")
    ax2.set_ylabel("lateral-error gap (mm)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title("C. Frozen-J lateral Jacobian rotation", fontsize=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # D1 / D2: useful lateral corrections + net efficiency, 4 conditions
    conditions = [
        ("INV-7\nSJ", cfg["inv_sj"], False, "tab:green"),
        ("MPC\nSJ", cfg["mpc_sj"], False, "tab:blue"),
        ("INV-7\nFJ", cfg["inv_fj"], True, "tab:green"),
        ("MPC\nFJ", cfg["mpc_fj"], True, "tab:blue"),
    ]
    d1_data, d2_data = [], []
    for name, runs, is_fj, color in conditions:
        schedule = J_FJ[None, :, :].repeat(N_REF, axis=0) if is_fj else J_SJ_schedule
        useful_fracs, final_effs = useful_lateral_and_efficiency(runs, schedule, input_reference, N_REF, is_fj)
        d1_data.append((name, useful_fracs, color))
        d2_data.append((name, final_effs, color))

    ax = axes4[1, 1]
    for i, (name, vals, color) in enumerate(d1_data):
        vals_pct = np.array(vals) * 100.0
        ax.scatter([i] * len(vals_pct), vals_pct, color=color, alpha=0.4, s=25)
        ax.errorbar([i], [np.mean(vals_pct)], yerr=[np.std(vals_pct)], marker="D", markersize=10,
                   color=color, capsize=5)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _, _ in d1_data], fontsize=8)
    ax.set_ylabel("useful lateral correction ticks (%)")
    ax.set_title("D1. Useful lateral corrections", fontsize=10)

    ax = axes4[2, 0]
    for i, (name, vals, color) in enumerate(d2_data):
        ax.scatter([i] * len(vals), vals, color=color, alpha=0.4, s=25)
        ax.errorbar([i], [np.nanmean(vals)], yerr=[np.nanstd(vals)], marker="D", markersize=10,
                   color=color, capsize=5)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _, _ in d2_data], fontsize=8)
    ax.set_ylabel("net |e_y| reduction / normalized effort")
    ax.set_title("D2. Net correction efficiency", fontsize=10)

    axes4[2, 1].axis("off")

    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f"fig4_mechanism_{shape_name.split()[0].lower()}.png"
    fig4.savefig(f"{OUTDIR}/{fname}", dpi=150)
    plt.close(fig4)
    print(f"saved -> {OUTDIR}/{fname}")

print("\nDONE.")
