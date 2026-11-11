"""Master data pipeline + publication figures for the full vessel-control
story: open-loop -> closed-loop -> MPC vs inverse-Jacobian -> scheduled vs
frozen Jacobian -> horizon N=15 vs N=1 -> dipole-align vs non-alignment.
All on the floor240_dipolealign_recal_matchedtarget plan family (dipole-
aligned and no-dipole variants), n=10 reps per condition throughout.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

BASE = "/home/jack/Proper-Research/close_loop_logs/myrun"

GROUPS = {
"OL_dipole": [
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_20261020T120855Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v2_20261021T175900Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v3_20261021T180332Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v4_20261021T180730Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v5_20261021T181112Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v6_20261021T181432Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v7_20261021T181750Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v8_20261021T182114Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v9_20261021T182800Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_dipolealigned_gain1p0_v10_20261021T183113Z",
],
"OL_nodipole": [
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v1_20261021T171920Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v2_20261021T172234Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v3_20261021T172554Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v4_20261021T173037Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v5_20261021T173334Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v6_20261021T173719Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v7_20261021T174017Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v8_20261021T174403Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v9_20261021T174652Z",
    "openloop_contact_floor240_dipolealign_recal_matchedtarget_nodipole_v10_20261021T174953Z",
],
"MPC_N15_dipole": [
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T141102Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v2_20261020T142028Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v3_20261020T143016Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v4_20261020T155708Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v5_20261020T160027Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v6_20261021T095348Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v7_20261021T104705Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v8_20261021T163704Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v9_20261021T164053Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v10_20261021T164409Z",
],
"MPC_N1_dipole": [
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v1_20261021T103706Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v2_20261021T104238Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v3_20261021T105103Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v4_20261021T105419Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v5_20261021T110156Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v6_20261021T110600Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v7_20261021T161355Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v8_20261021T161741Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v9_20261021T162921Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_horizon1_v10_20261021T163248Z",
],
"MPC_N15_nodipole": [
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v1_20261021T095022Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v2_20261021T095742Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v3_20261021T100030Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v4_20261021T100329Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v5_20261021T101031Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v6_20261021T101348Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v7_20261021T135705Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v8_20261021T135954Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v9_20261021T140250Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v10_20261021T140548Z",
],
"MPC_N1_nodipole": [
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v1_20261021T101752Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v2_20261021T102053Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v3_20261021T102353Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v4_20261021T102702Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v5_20261021T103003Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v6_20261021T103327Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v7_20261021T143036Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v8_20261021T143335Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v9_20261021T143624Z",
    "mpc_closedloop_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_horizon1_v10_20261021T143919Z",
],
"Frozen_dipole": [
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T144802Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v2_20261020T145158Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v3_20261020T145505Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v4_20261021T113518Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v5_20261021T112954Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v6_20261021T113225Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v7_20261021T134633Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v8_20261021T134941Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v9_20261021T135148Z",
    "mpc_closedloop_frozen_floor240_dipolealign_recal_matchedtarget_dipolealigned_v10_20261021T135414Z",
],
"InvJac_dipole": [
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_20261020T153443Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v2_20261020T154108Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v3_20261020T154654Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v4_20261021T115621Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v5_20261021T115801Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v6_20261021T115950Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v7_20261021T164853Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v8_20261021T165221Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v9_20261021T165545Z",
    "invjac_hybrid_contactq_nocontactL_floor240_dipolealign_recal_matchedtarget_dipolealigned_v10_20261021T165939Z",
],
"InvJac_nodipole": [
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v1_20261021T132341Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v2_20261021T132709Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v3_20261021T132855Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v4_20261021T133036Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v5_20261021T170647Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v6_20261021T133553Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v7_20261021T133747Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v8_20261021T133954Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v9_20261021T170847Z",
    "invjac_hybrid_contactq_nocontactL_floor240_recal_matchedtarget_nodipole_v10_20261021T134418Z",
],
}


def load_metrics(run_name):
    d = BASE + "/" + run_name
    s = json.load(open(d + "/summary.json"))
    errs = []
    with open(d + "/path_follow.jsonl") as f:
        for line in f:
            r = json.loads(line)
            e = r.get("error_norm_mm")
            if e is not None:
                errs.append(e)
    errs = np.array(errs)
    rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else np.nan
    return dict(
        rmse=rmse,
        stop_reason=s.get("stop_reason"),
        complete=(s.get("stop_reason") == "path_complete"),
        control_steps=s.get("control_steps"),
        reference_samples=s.get("reference_samples"),
        final_error_mm=s.get("final_error_mm"),
    )


print("=== loading all 9 groups x 10 reps ===")
DATA = {}
for gname, runs in GROUPS.items():
    rows = [load_metrics(r) for r in runs]
    rmse_vals = np.array([r["rmse"] for r in rows])
    n_complete = sum(r["complete"] for r in rows)
    DATA[gname] = dict(rmse=rmse_vals, rows=rows, n_complete=n_complete)
    print(f"  {gname:20s} n={len(rmse_vals)}  median_rmse={np.median(rmse_vals):.3f}  "
          f"range=[{rmse_vals.min():.3f},{rmse_vals.max():.3f}]  complete={n_complete}/10")

np.savez("/home/jack/.claude/jobs/043c0b8b/tmp/figures/all_rmse_data.npz",
         **{k: v["rmse"] for k, v in DATA.items()})

# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

COLOR_A = "#2b6cb0"   # blue -- "better" condition in each panel
COLOR_B = "#c0392b"   # red -- "worse" / baseline condition
COLOR_OL = "#7f8c8d"  # grey -- open loop


def strip_box(ax, data_list, labels, colors, title, ylabel="RMSE (mm)", log=False,
              annotate_p=True, incomplete_flags=None):
    positions = np.arange(1, len(data_list) + 1)
    bp = ax.boxplot(data_list, positions=positions, widths=0.45, showfliers=False,
                     patch_artist=True, medianprops=dict(color="black", linewidth=1.8),
                     whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip(data_list, colors)):
        jitter = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(positions[i] + jitter, d, color=c, s=38, alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=12.5)
    ax.set_title(title, fontsize=13, pad=62, weight="bold", wrap=True)
    if log:
        ax.set_yscale("log")
    ax.set_xlim(0.4, len(data_list) + 0.6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    # n= labels: fixed axes-fraction y position, immune to data range/log-scale issues
    for i, d in enumerate(data_list):
        ax.annotate(f"n={len(d)}", xy=(positions[i], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -52), textcoords="offset points", ha="center", fontsize=9.5, color="#666666")
    if incomplete_flags is not None:
        for i, flag in enumerate(incomplete_flags):
            if flag:
                ax.annotate("did not\ncomplete", xy=(positions[i] + 0.38, 1), xycoords=("data", "axes fraction"),
                            xytext=(0, 34), textcoords="offset points", ha="center", va="bottom",
                            fontsize=9, color="#888888", style="italic")
    if annotate_p and len(data_list) == 2:
        u, p = sstats.mannwhitneyu(data_list[0], data_list[1], alternative="two-sided")
        p_text = f"p = {p:.4f}" if p >= 0.0001 else f"p = {p:.1e}"
        y_bracket = 1.14
        ax.annotate("", xy=(1, y_bracket), xycoords=("data", "axes fraction"),
                    xytext=(2, y_bracket), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#333333", linewidth=1.0,
                                     connectionstyle="bar,fraction=0.08"))
        ax.annotate(p_text, xy=(1.5, y_bracket), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10.5)
        ax.set_ylim(top=ax.get_ylim()[1] * (1.35 if log else 1.0))
        if not log:
            top = ax.get_ylim()[1]
            ax.set_ylim(top=top * 1.12)


FIGDIR = "/home/jack/.claude/jobs/043c0b8b/tmp/figures"

# ---- Figure 1: Open-loop vs closed-loop ----
fig, ax = plt.subplots(figsize=(7.2, 5.2))
strip_box(ax, [DATA["OL_dipole"]["rmse"], DATA["MPC_N15_dipole"]["rmse"]],
          ["Open-loop\n(feedforward only)", "Closed-loop MPC\n(scheduled Jacobian, N=15)"],
          [COLOR_OL, COLOR_A],
          "Closed-loop feedback dramatically\nreduces tracking error", log=True)
fig.subplots_adjust(left=0.14, right=0.97, top=0.62, bottom=0.17)
fig.savefig(f"{FIGDIR}/fig1_openloop_vs_closedloop.png", dpi=220)
print("saved fig1")

# ---- Figure 2: MPC vs inverse-Jacobian ----
fig, ax = plt.subplots(figsize=(7.2, 5.2))
strip_box(ax, [DATA["MPC_N15_dipole"]["rmse"], DATA["InvJac_dipole"]["rmse"]],
          ["MPC\n(scheduled Jacobian)", "Inverse-Jacobian\ncontroller"],
          [COLOR_A, COLOR_B],
          "MPC's in-QP constraint handling outperforms\nthe inverse-Jacobian controller", log=True,
          incomplete_flags=[False, DATA["InvJac_dipole"]["n_complete"] < 10])
fig.subplots_adjust(left=0.14, right=0.97, top=0.60, bottom=0.17)
fig.savefig(f"{FIGDIR}/fig2_mpc_vs_invjac.png", dpi=220)
print("saved fig2")

# ---- Figure 3: Scheduled vs frozen Jacobian ----
fig, ax = plt.subplots(figsize=(7.2, 5.2))
strip_box(ax, [DATA["MPC_N15_dipole"]["rmse"], DATA["Frozen_dipole"]["rmse"]],
          ["Scheduled Jacobian\n(re-linearized every tick)", "Frozen Jacobian\n(fixed at t=0)"],
          [COLOR_A, COLOR_B],
          "A continuously-updated Jacobian schedule\noutperforms a single frozen linearization",
          incomplete_flags=[False, DATA["Frozen_dipole"]["n_complete"] < 10])
fig.subplots_adjust(left=0.14, right=0.97, top=0.60, bottom=0.17)
fig.savefig(f"{FIGDIR}/fig3_scheduled_vs_frozen.png", dpi=220)
print("saved fig3")

# ---- Figure 4: Horizon N=15 vs N=1 ----
fig, ax = plt.subplots(figsize=(7.2, 5.2))
strip_box(ax, [DATA["MPC_N15_dipole"]["rmse"], DATA["MPC_N1_dipole"]["rmse"]],
          ["N = 15", "N = 1"],
          [COLOR_A, COLOR_B],
          "Longer prediction horizon yields\nsubstantially lower tracking error")
fig.subplots_adjust(left=0.14, right=0.97, top=0.62, bottom=0.17)
fig.savefig(f"{FIGDIR}/fig4_horizon_n15_vs_n1.png", dpi=220)
print("saved fig4")

# ---- Figure 5: Dipole-align vs non-alignment, grouped by horizon ----
fig, ax = plt.subplots(figsize=(9.0, 6.4))
groups_5 = [DATA["MPC_N15_dipole"]["rmse"], DATA["MPC_N15_nodipole"]["rmse"],
            DATA["MPC_N1_dipole"]["rmse"], DATA["MPC_N1_nodipole"]["rmse"]]
labels_5 = ["Dipole-\naligned", "No-dipole", "Dipole-\naligned", "No-dipole"]
colors_5 = [COLOR_A, "#5ba3d9", COLOR_B, "#e07b6b"]
positions = np.array([1, 1.8, 3.2, 4.0])
bp = ax.boxplot(groups_5, positions=positions, widths=0.55, showfliers=False,
                 patch_artist=True, medianprops=dict(color="black", linewidth=1.8),
                 whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
for patch, c in zip(bp["boxes"], colors_5):
    patch.set_facecolor(c); patch.set_alpha(0.28); patch.set_edgecolor(c); patch.set_linewidth(1.5)
rng = np.random.default_rng(1)
for i, (d, c) in enumerate(zip(groups_5, colors_5)):
    jitter = rng.uniform(-0.15, 0.15, size=len(d))
    ax.scatter(positions[i] + jitter, d, color=c, s=36, alpha=0.85, edgecolor="white",
               linewidth=0.6, zorder=5)
ax.set_xticks(positions)
ax.set_xticklabels(labels_5, fontsize=11)
ax.set_ylabel("RMSE (mm)", fontsize=12.5)
ax.set_title("Dipole-alignment cost is horizon-dependent:\nnegligible at N=15, real at N=1",
             fontsize=13.5, pad=22, weight="bold")
ax.set_xlim(0.3, 4.7)
ax.axvline(2.5, color="#cccccc", linewidth=1.0, linestyle="--")
ax.annotate("N = 15", xy=(1.4, 1), xycoords=("data", "axes fraction"), xytext=(0, -22),
            textcoords="offset points", ha="center", fontsize=11.5, color="#555555", weight="bold")
ax.annotate("N = 1", xy=(3.6, 1), xycoords=("data", "axes fraction"), xytext=(0, -22),
            textcoords="offset points", ha="center", fontsize=11.5, color="#555555", weight="bold")
ax.grid(axis="y", alpha=0.25, linewidth=0.8)
for i, d in enumerate(groups_5):
    ax.annotate(f"n={len(d)}", xy=(positions[i], 0), xycoords=("data", "axes fraction"),
                xytext=(0, -50), textcoords="offset points", ha="center", fontsize=9.5, color="#666666")
# p-values for each horizon pair, placed just above each pair's own max (data coords, safe since linear scale)
for (i1, i2, xc) in [(0, 1, 1.4), (2, 3, 3.6)]:
    u, p = sstats.mannwhitneyu(groups_5[i1], groups_5[i2], alternative="two-sided")
    y_top = max(np.max(groups_5[i1]), np.max(groups_5[i2])) * 1.08
    ax.plot([positions[i1], positions[i1], positions[i2], positions[i2]],
            [y_top*0.97, y_top, y_top, y_top*0.97], color="#333333", linewidth=1.0)
    p_text = f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}"
    ax.text(xc, y_top*1.05, p_text, ha="center", fontsize=10)
ax.set_ylim(top=ax.get_ylim()[1] * 1.28)
fig.subplots_adjust(left=0.11, right=0.97, top=0.72, bottom=0.24)
fig.savefig(f"{FIGDIR}/fig5_dipole_vs_nodipole.png", dpi=220)
print("saved fig5")

print("\nAll figures saved to", FIGDIR)
