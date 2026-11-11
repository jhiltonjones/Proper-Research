"""Tip trajectory plot (world XY, mm) for open-loop vs closed-loop MPC,
overlaid on the actual vessel lumen walls and the planned target path.
Picks the median-RMSE rep from each group for a representative single trace.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/jack/.claude/jobs/043c0b8b/tmp/figures")
from publication_figures import GROUPS, BASE, COLOR_A, COLOR_OL

LUMEN_NPZ = ("/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_"
             "floor240_dipolealign_recal_matchedtarget_dipolealigned_2026-10-20/vessel_lumen/"
             "vessel_centreline.npz")
FIGDIR = "/home/jack/.claude/jobs/043c0b8b/tmp/figures"


def median_run(gname):
    """Pick the rep whose RMSE is closest to the group median."""
    rows = []
    for r in GROUPS[gname]:
        errs = []
        with open(BASE + "/" + r + "/path_follow.jsonl") as f:
            for line in f:
                e = json.loads(line).get("error_norm_mm")
                if e is not None:
                    errs.append(e)
        rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
        rows.append((r, rmse))
    med = np.median([x[1] for x in rows])
    rows.sort(key=lambda x: abs(x[1] - med))
    return rows[0][0], rows[0][1]


def load_tip_xy_mm(run_name):
    path = BASE + "/" + run_name + "/tip_trajectory.csv"
    data = np.genfromtxt(path, delimiter=",", names=True)
    x = data["tip_x_m"] * 1000.0
    y = data["tip_y_m"] * 1000.0
    dx = data["des_x_m"] * 1000.0
    dy = data["des_y_m"] * 1000.0
    return x, y, dx, dy


# ---- vessel wall geometry ----
d = np.load(LUMEN_NPZ)
C = d["lumen_C"][:, :2] * 1000.0  # (N,2) mm, XY only (Z constant)
R = d["lumen_R"] * 1000.0  # mm

tangents = np.gradient(C, axis=0)
tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
wall_left = C + normals * R[:, None]
wall_right = C - normals * R[:, None]

# ---- representative reps ----
ol_run, ol_rmse = median_run("OL_dipole")
cl_run, cl_rmse = median_run("MPC_N15_dipole")
print(f"open-loop median-RMSE rep: {ol_run} (RMSE={ol_rmse:.3f}mm)")
print(f"closed-loop median-RMSE rep: {cl_run} (RMSE={cl_rmse:.3f}mm)")

ol_x, ol_y, dx, dy = load_tip_xy_mm(ol_run)
cl_x, cl_y, _, _ = load_tip_xy_mm(cl_run)

# ---- plot ----
plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, ax = plt.subplots(figsize=(9.5, 7.5))

# vessel wall band
ax.fill(np.concatenate([wall_left[:, 0], wall_right[::-1, 0]]),
        np.concatenate([wall_left[:, 1], wall_right[::-1, 1]]),
        color="#f0d9c0", alpha=0.6, zorder=0, label="Vessel lumen")
ax.plot(wall_left[:, 0], wall_left[:, 1], color="#b08968", linewidth=1.3, zorder=1)
ax.plot(wall_right[:, 0], wall_right[:, 1], color="#b08968", linewidth=1.3, zorder=1)

# target path
ax.plot(dx, dy, color="#333333", linewidth=1.4, linestyle="--", zorder=2, label="Planned target path")

# tip trajectories
ax.plot(ol_x, ol_y, color=COLOR_OL, linewidth=2.2, zorder=4,
        label=f"Open-loop tip path (RMSE={ol_rmse:.2f}mm)")
ax.plot(cl_x, cl_y, color=COLOR_A, linewidth=2.2, zorder=5,
        label=f"Closed-loop MPC tip path (RMSE={cl_rmse:.2f}mm)")

ax.scatter([dx[0]], [dy[0]], color="#333333", s=60, zorder=6, marker="o")
ax.annotate("start", xy=(dx[0], dy[0]), xytext=(6, 6), textcoords="offset points", fontsize=10)

ax.set_xlabel("World X (mm)", fontsize=12.5)
ax.set_ylabel("World Y (mm)", fontsize=12.5)
ax.set_title("Beam tip path through the vessel lumen:\nopen-loop vs closed-loop MPC", fontsize=14, weight="bold", pad=14)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
ax.grid(alpha=0.2, linewidth=0.7)

fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.10)
fig.savefig(f"{FIGDIR}/fig6_trajectory_vs_wall.png", dpi=220)
print("saved fig6")
