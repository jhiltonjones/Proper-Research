"""Task 2A (H2): workspace/feasibility timeline for tight-floor (255mm)
MPC-J_C vs MPC-J_NC.

Computes, vs path progress s, for every rep of both conditions:
  - exclusion-radius margin to the beam base (source-magnet xy distance minus
    the condition's own live-enforced floor, 250mm for both at this radius --
    verified identical via controller_metadata.json, see report Sec 1.2/3.1);
  - magnet vertical workspace margin (z_min, z_max, from the same formula
    run_mpc_delay_aware_vessel.py / h3_replay.py use: z0 +/- margins; default
    floor_margin=40mm, rise_limit=40mm, i.e. the SAME magnet_z_bounds
    convention used throughout this investigation's H3 analysis);
  - generic magnet/TCP workspace-box margins to all 6 faces of the vessel
    stage's actual configured box, cfg.workspace_xyz_min_m=(0.277,-0.832,0.170),
    cfg.workspace_xyz_max_m=(0.656,-0.2,0.433) (verified directly from
    run_mpc_delay_aware_vessel.py's own source, ~line 1018-1019);
  - the "toothless" joint/insertion state-box margins enforced by the
    controller's own clip (STATE_MIN=[-2pi]*6+[-0.05], STATE_MAX=[2pi]*6+
    [0.20], from h3_replay.py/inverse_jacobian_controller's own clip_command
    -- the MPC's QP enforces the analogous box directly);
  - source-magnet xyz, magnet-to-base distance, insertion L, all 6 joints.

Marks s={45,50,55,58,60,62,64}mm explicitly and reports, at each mark,
whether MPC-J_NC is already closer to a workspace/joint boundary than
MPC-J_C, i.e. whether it is "approaching infeasibility" ahead of its actual
s~64mm tracking-error abort (confirmed from summary.json: all 3 reps stop on
stop_reason=tracking_error_exceeded(5mm), NOT a workspace check -- so this
script answers "approaching" vs "crossing", not which check fired).

Read-only against run data and source; writes only new table/figure.
"""
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
sys.path.insert(0, "/home/jack/Proper-Research")
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest
import loader

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"

BOX_MIN = np.array([0.277, -0.832, 0.170])
BOX_MAX = np.array([0.656, -0.2, 0.433])
R_LIVE_255_M = 0.250  # enforced (live) exclusion floor, both MPC-JC(matched) and MPC-JNC @255mm (report Sec 3.1)
STATE_MIN6 = np.full(6, -2 * np.pi)
STATE_MAX6 = np.full(6, 2 * np.pi)
L_MIN, L_MAX = -0.05, 0.20
MARKS = [45, 50, 55, 58, 60, 62, 64]


def magnet_z_bounds(plan_dir, floor_margin_m=0.040, rise_limit_m=0.040):
    ref = loader.get_reference(plan_dir)
    q0 = np.asarray(ref["state_reference"][0, :6], dtype=float)
    z0 = float(loader.magnet_xyz_batch(q0.reshape(1, 6))[0, 2])
    return z0 - floor_margin_m, z0 + rise_limit_m


def collect(rm):
    out, meta = loader.enrich_run(rm)
    q = out["q_meas_rad"]
    p_mag = loader.magnet_xyz_batch(q)
    z_min, z_max = magnet_z_bounds(meta["plan_dir"])

    s = out["s_ref_mm"]
    dist_mm = out["magnet_xy_dist_mm"]
    excl_margin_mm = dist_mm - R_LIVE_255_M * 1000.0
    z_margin_min_mm = (p_mag[:, 2] - z_min) * 1000.0
    z_margin_max_mm = (z_max - p_mag[:, 2]) * 1000.0

    box_margin_mm = {}
    for i, name in enumerate(["x", "y", "z"]):
        box_margin_mm[f"box_{name}_min_mm"] = (p_mag[:, i] - BOX_MIN[i]) * 1000.0
        box_margin_mm[f"box_{name}_max_mm"] = (BOX_MAX[i] - p_mag[:, i]) * 1000.0

    joint_margin_min = (q - STATE_MIN6[None, :])
    joint_margin_max = (STATE_MAX6[None, :] - q)
    L_margin_min_mm = (out["insertion_length_m"] - L_MIN) * 1000.0
    L_margin_max_mm = (L_MAX - out["insertion_length_m"]) * 1000.0

    rows = []
    for k in range(len(s)):
        row = dict(
            dirname=meta["dirname"], condition=rm["condition"], rep=rm["rep"],
            s_mm=float(s[k]), insertion_mm=float(out["insertion_length_m"][k] * 1000),
            magnet_x_m=float(p_mag[k, 0]), magnet_y_m=float(p_mag[k, 1]), magnet_z_m=float(p_mag[k, 2]),
            magnet_base_dist_mm=float(dist_mm[k]),
            exclusion_margin_mm=float(excl_margin_mm[k]),
            z_margin_min_mm=float(z_margin_min_mm[k]), z_margin_max_mm=float(z_margin_max_mm[k]),
            L_margin_min_mm=float(L_margin_min_mm[k]), L_margin_max_mm=float(L_margin_max_mm[k]),
            error_norm_mm=float(out["error_norm_mm"][k]),
        )
        for name, arr in box_margin_mm.items():
            row[name] = float(arr[k])
        for j in range(6):
            row[f"q{j+1}_margin_min_rad"] = float(joint_margin_min[k, j])
            row[f"q{j+1}_margin_max_rad"] = float(joint_margin_max[k, j])
            row[f"q{j+1}_rad"] = float(q[k, j])
        rows.append(row)
    return pd.DataFrame(rows), out["s_tip_mm"] if "s_tip_mm" in out else None


all_rows = []
cache = {}
for cond in ("mpc_C", "mpc_NC"):
    for rm in manifest.RUNS:
        if rm["group"] == "closedloop" and rm["radius_intended_mm"] == 255 and rm["condition"] == cond:
            pass
    # mpc_C @255 primary headline = matched255 group; mpc_NC @255 = closedloop group
    group_for = {"mpc_C": "matched255", "mpc_NC": "closedloop"}[cond]
    runs = [rm for rm in manifest.RUNS if rm["group"] == group_for and rm["condition"] == cond
            and rm["radius_intended_mm"] == 255]
    for rm in runs:
        df, _ = collect(rm)
        all_rows.append(df)
        cache[rm["dirname"]] = df

full = pd.concat(all_rows, ignore_index=True)
full.to_csv(f"{OUT}/tables/h2_workspace_margin_timeline.csv", index=False)
print(f"wrote {OUT}/tables/h2_workspace_margin_timeline.csv ({len(full)} rows, "
      f"{full.dirname.nunique()} runs)")

# --- marks table: nearest-tick values at s in MARKS, per run ---
mark_rows = []
for dirname, df in cache.items():
    cond = df["condition"].iloc[0]
    rep = df["rep"].iloc[0]
    for s_mark in MARKS:
        idx = (df["s_mm"] - s_mark).abs().idxmin()
        row = df.loc[idx].to_dict()
        row["s_mark_mm"] = s_mark
        mark_rows.append(row)
marks_df = pd.DataFrame(mark_rows)
marks_df.to_csv(f"{OUT}/tables/h2_workspace_margin_at_marks.csv", index=False)
print(f"wrote {OUT}/tables/h2_workspace_margin_at_marks.csv")

summary_cols = ["s_mark_mm", "condition", "rep", "exclusion_margin_mm", "z_margin_min_mm", "z_margin_max_mm",
                "box_y_max_mm", "box_x_max_mm", "magnet_base_dist_mm", "insertion_mm", "error_norm_mm"]
print(marks_df[summary_cols].sort_values(["s_mark_mm", "condition", "rep"]).to_string(index=False))

# --- figure ---
fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
colors = {"mpc_C": "#1b7f3b", "mpc_NC": "#b3331d"}
panels = [
    ("exclusion_margin_mm", "exclusion-radius margin (mm)"),
    ("z_margin_min_mm", "magnet z-floor margin (mm)"),
    ("box_x_max_mm", "box x_max margin (mm)"),
    ("box_y_max_mm", "box y_max margin (mm)"),
    ("magnet_base_dist_mm", "magnet-to-base distance (mm)"),
    ("error_norm_mm", "tracking error (mm)"),
]
for ax, (col, ylabel) in zip(axes.flat, panels):
    for dirname, df in cache.items():
        cond = df["condition"].iloc[0]
        d = df.sort_values("s_mm")
        ax.plot(d["s_mm"], d[col], color=colors[cond], lw=1.1, alpha=0.85,
                label=cond if dirname == next(k for k, v in cache.items() if v["condition"].iloc[0] == cond) else None)
    if col == "exclusion_margin_mm":
        ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    for s_mark in MARKS:
        ax.axvline(s_mark, color="black", lw=0.4, alpha=0.3)
axes.flat[0].legend(fontsize=8)
axes[-1, 0].set_xlabel("path progress s (mm)")
axes[-1, 1].set_xlabel("path progress s (mm)")
fig.suptitle("H2 workspace/feasibility timeline, 255mm (enforced 250mm): MPC-$J_C$ vs MPC-$J_{NC}$", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
f = f"{OUT}/figures/h2_workspace_margin_timeline.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"\nsaved {f}")
