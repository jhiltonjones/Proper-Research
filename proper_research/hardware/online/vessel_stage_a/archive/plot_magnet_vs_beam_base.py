"""Plot the source magnet's position relative to the fixed beam base over a
completed inverse-configuration path (from its own saved CSV, which already
records magnet_pose_{x,y,z} per node -- no FK recomputation needed).

Usage: python plot_magnet_vs_beam_base.py <plan_output_root>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proper_research.rig_calibration import BEAM_BASE_XYZ_M

plan_root = Path(sys.argv[1])
csv_path = plan_root / "vessel_lumen" / "offline_inverse_configuration" / "inverse_configuration_path.csv"
df = pd.read_csv(csv_path)

s_mm = df["s_m"].to_numpy() * 1e3
mx, my, mz = df["magnet_pose_x"].to_numpy(), df["magnet_pose_y"].to_numpy(), df["magnet_pose_z"].to_numpy()
base = BEAM_BASE_XYZ_M

d3 = np.sqrt((mx - base[0])**2 + (my - base[1])**2 + (mz - base[2])**2) * 1e3
dxy = np.sqrt((mx - base[0])**2 + (my - base[1])**2) * 1e3
dz_mm = (mz - base[2]) * 1e3
floor_mm = df["source_magnet_lumen_exclusion_radius_m"].to_numpy() * 1e3 \
    if "source_magnet_lumen_exclusion_radius_m" in df.columns else None
# magnet_beam_base_exclusion_radius_m isn't itself a per-node CSV column
# (it's a scalar config value) -- pull it from the summary JSON instead.
import json
summary = json.loads((plan_root / "vessel_lumen" / "offline_inverse_configuration" / "inverse_configuration_summary.json").read_text())
base_floor_mm = summary["configuration"].get("magnet_beam_base_exclusion_radius_m")
base_floor_mm = base_floor_mm * 1e3 if base_floor_mm is not None else None

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(s_mm, d3, "-", color="tab:blue", lw=2, label="3D distance, magnet-to-beam-base")
ax.plot(s_mm, dxy, "--", color="tab:orange", lw=1.5, label="XY-only (in-plane) distance")
if base_floor_mm is not None:
    ax.axhline(base_floor_mm, color="red", ls=":", lw=1.5, label=f"hard exclusion floor ({base_floor_mm:.1f}mm)")
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("distance [mm]")
ax.set_title("Magnet-to-beam-base distance")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(s_mm, dz_mm, "-", color="tab:purple", lw=2)
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("magnet Z - beam-base Z [mm]")
ax.set_title("Magnet height relative to beam base")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(mx, my, "-", color="tab:green", lw=2, label="magnet XY path")
ax.plot(base[0], base[1], "ks", markersize=10, label="beam base")
ax.plot(mx[0], my[0], "o", color="tab:green", markersize=8, label="start")
ax.plot(mx[-1], my[-1], "^", color="tab:green", markersize=8, label="end")
ax.set_xlabel("world X [m]")
ax.set_ylabel("world Y [m]")
ax.set_title("Top-down: magnet position vs beam base")
ax.set_aspect("equal")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(s_mm, mz * 1e3, "-", color="tab:purple", lw=2, label="magnet Z (world)")
ax.axhline(base[2] * 1e3, color="k", ls="--", lw=1.5, label=f"beam-base Z ({base[2]*1e3:.1f}mm)")
ax.set_xlabel("path coordinate s [mm]")
ax.set_ylabel("Z [mm]")
ax.set_title("Absolute magnet Z vs beam-base Z")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

fig.suptitle(f"Source magnet vs beam base: {plan_root.name}", fontsize=13)
fig.tight_layout()
out_png = plan_root / "magnet_vs_beam_base.png"
fig.savefig(out_png, dpi=150)
print(f"3D distance range: {d3.min():.2f} - {d3.max():.2f} mm")
print(f"XY-only distance range: {dxy.min():.2f} - {dxy.max():.2f} mm")
print(f"magnet Z - beam-base Z range: {dz_mm.min():.2f} - {dz_mm.max():.2f} mm")
if base_floor_mm is not None:
    print(f"hard exclusion floor: {base_floor_mm:.1f} mm (min margin: {d3.min()-base_floor_mm:.2f}mm)")
print(f"saved {out_png}")
