"""Beam tip trajectory, MPC-J_C vs MPC-J_NC at the 255mm (tight) floor,
plotted against the vessel's own geometry -- the lumen centreline and its
wall boundaries -- rather than only against tracking-error statistics.

The vessel lumen is stored as 60 (centre point, local radius) samples along
its own centreline (vessel_lumen_robot_frame_*.json, robot frame R). Both
the lumen centreline and the beam tip are planar in this rig (z is constant
to the mm for both, confirmed directly from the data), so this is rendered
as a clean 2D x-y plot in the robot frame, in mm, with an equal aspect
ratio -- no projection distortion.

Wall boundaries are reconstructed from the centreline samples by offsetting
each sample by +/- its own local radius along the local in-plane normal
(estimated by finite-differencing the centreline sequence itself).

Runs shown: the headline, radius-matched 255mm comparison this report uses
throughout H2 -- MPC-J_C (group 'matched255', 3 reps, all path_complete) vs
MPC-J_NC (group 'closedloop', condition 'mpc_NC', 3 reps, all
tracking_error_exceeded(5mm)). Each condition's own offline reference path
is also shown, lightly, since the two conditions are driven by two
different offline plans (contact-aware vs no-contact), not only two
different online Jacobians.

Read-only against existing run data, the lumen geometry file, and the plan
reference files; writes only the one new figure + its underlying table.
"""
import json
import sys
sys.path.insert(0, "/home/jack/.claude/jobs/ea647104/tmp/stage3_prior")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import manifest
import loader

OUT = "/home/jack/Proper-Research/.claude/worktrees/stage3-final-rewrite/plans/stage3_design/full_stage3_analysis"
LUMEN_FILE = "/home/jack/Proper-Research/vessel_lumen_robot_frame_2026-10-06_zcorrected.json"

COND_COLOR = {"mpc_C": "#1b7f3b", "mpc_NC": "#b3331d"}
COND_LABEL = {"mpc_C": "MPC, contact Jacobian $J_C$ (completes, 3/3)",
              "mpc_NC": "MPC, no-contact Jacobian $J_{NC}$ (fails, 0/3)"}
COND_GROUP = {"mpc_C": "matched255", "mpc_NC": "closedloop"}

# --- vessel geometry: centreline + wall boundaries, in mm ---
lumen = json.load(open(LUMEN_FILE))
assert lumen["frame"] == "R", "lumen geometry must be in robot frame R to match tip_mm/desired_mm"
C = np.asarray(lumen["lumen_C_m"], dtype=float) * 1000.0  # (60,3) mm
R = np.asarray(lumen["lumen_R_m"], dtype=float) * 1000.0  # (60,) mm
z_vals = C[:, 2]
assert np.ptp(z_vals) < 0.01, f"lumen is not planar in z (range {np.ptp(z_vals):.4f}mm) -- 2D plot would be invalid"

Cxy = C[:, :2]
tangent = np.gradient(Cxy, axis=0)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # in-plane perpendicular
wall_left = Cxy + R[:, None] * normal
wall_right = Cxy - R[:, None] * normal

geom_df = pd.DataFrame(dict(
    centreline_x_mm=Cxy[:, 0], centreline_y_mm=Cxy[:, 1], radius_mm=R,
    wall_left_x_mm=wall_left[:, 0], wall_left_y_mm=wall_left[:, 1],
    wall_right_x_mm=wall_right[:, 0], wall_right_y_mm=wall_right[:, 1],
))
geom_df.to_csv(f"{OUT}/tables/vessel_geometry_xy_mm.csv", index=False)
print(f"wrote {OUT}/tables/vessel_geometry_xy_mm.csv ({len(geom_df)} rows)")

# --- figure ---
fig, ax = plt.subplots(figsize=(9, 11))

ax.fill(np.concatenate([wall_left[:, 0], wall_right[::-1, 0]]),
        np.concatenate([wall_left[:, 1], wall_right[::-1, 1]]),
        color="#d8e6f3", alpha=0.6, zorder=0, label="vessel lumen (wall-to-wall)")
ax.plot(wall_left[:, 0], wall_left[:, 1], color="#5b7fa6", lw=1.3, zorder=1)
ax.plot(wall_right[:, 0], wall_right[:, 1], color="#5b7fa6", lw=1.3, zorder=1, label="vessel wall")
ax.plot(Cxy[:, 0], Cxy[:, 1], color="#5b7fa6", lw=1.0, ls=":", alpha=0.9, zorder=1,
        label="vessel centreline")

tip_rows = []
plotted_ref = set()
for cond in ("mpc_C", "mpc_NC"):
    group = COND_GROUP[cond]
    runs = [rm for rm in manifest.RUNS
            if rm["group"] == group and rm["condition"] == cond and rm["radius_intended_mm"] == 255]
    runs = sorted(runs, key=lambda r: r["rep"])
    for i, rm in enumerate(runs):
        out, meta = loader.enrich_run(rm)
        tip = out["tip_mm"]
        ax.plot(tip[:, 0], tip[:, 1], color=COND_COLOR[cond], lw=1.4, alpha=0.85, zorder=3,
                label=COND_LABEL[cond] if i == 0 else None)
        if meta["stop_reason"] != "path_complete":
            ax.plot(tip[-1, 0], tip[-1, 1], marker="x", ms=10, mew=2.5,
                     color=COND_COLOR[cond], zorder=4,
                     label=f"{cond} abort point" if i == 0 else None)
        if meta["plan_dir"] not in plotted_ref:
            desired = out["desired_mm"]
            ax.plot(desired[:, 0], desired[:, 1], color=COND_COLOR[cond], lw=0.9, ls="--",
                     alpha=0.45, zorder=2,
                     label=f"{cond}'s own offline reference" if cond not in plotted_ref else None)
            plotted_ref.add(meta["plan_dir"])
        for k in range(len(tip)):
            tip_rows.append(dict(
                condition=cond, rep=rm["rep"], tick=int(out["step"][k]),
                s_ref_mm=float(out["s_ref_mm"][k]),
                tip_x_mm=float(tip[k, 0]), tip_y_mm=float(tip[k, 1]),
                stop_reason=meta["stop_reason"],
            ))

ax.plot(C[0, 0], C[0, 1], marker="*", ms=16, color="black", zorder=5, label="vessel entry (beam base end)")

ax.set_xlabel("robot-frame $x$ (mm)", fontsize=11)
ax.set_ylabel("robot-frame $y$ (mm)", fontsize=11)
ax.set_title("Beam tip trajectory vs vessel centreline/walls, 255mm (tight) floor\n"
             "MPC-$J_C$ (completes) vs MPC-$J_{NC}$ (fails) -- H2's headline comparison", fontsize=12)
ax.set_aspect("equal", adjustable="datalim")
ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25)
fig.tight_layout()

tip_df = pd.DataFrame(tip_rows)
tip_df.to_csv(f"{OUT}/tables/tip_vs_vessel_geometry_255mm.csv", index=False)
print(f"wrote {OUT}/tables/tip_vs_vessel_geometry_255mm.csv ({len(tip_df)} rows)")

f = f"{OUT}/figures/tip_vs_vessel_geometry_255mm.png"
fig.savefig(f, dpi=160)
plt.close(fig)
print(f"saved {f}")
