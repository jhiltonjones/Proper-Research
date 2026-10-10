import numpy as np, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

bp = np.array([0.525575, -0.670028, -0.016567]); u = np.array([-1., 0, 0]); v = np.array([0, 1., 0])
runs = [
    ("3 mm square (insertion peaks 37 mm)", "uprgrade_configuration/bends_p30_m70_jac_contact_plant_contact_1_ctrl_mpc_sqp_full_rollout10_Np15_sqp50_bends_30_-70/time_parameterized_configuration_path/time_parameterized_configuration_path.npz"),
    ("2 mm square (all limits respected)", "/home/jack/.claude/jobs/3710eca5/tmp/plan_sq2mm/time_parameterized_configuration_path/time_parameterized_configuration_path.npz"),
]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for name, npz in runs:
    d = np.load(npz)
    P = np.asarray(d["desired_position_m"]); S = np.asarray(d["state"]); t = np.asarray(d["time_s"])
    uv = np.c_[(P - bp) @ u, (P - bp) @ v] * 1e3
    ax[0].plot(uv[:, 1], uv[:, 0], "o-", ms=3, label=name)
    ax[1].plot(t, S[:, 6] * 1e3, "o-", ms=3, label=name)
    ax[2].plot(t, 1e3 * np.asarray(d["position_error_m"]), "o-", ms=3, label=name)
ax[0].set_xlabel("tip B.y  (in-plane, mm)"); ax[0].set_ylabel("tip B.x  (axial / insertion, mm)")
ax[0].set_title("planned tip path (bend plane)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8); ax[0].set_aspect("equal")
ax[1].axhline(36, color="r", ls="--", lw=1, label="0.036 m cap")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("insertion [mm]"); ax[1].set_title("insertion vs time"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("plan tip position error [mm]"); ax[2].set_title("model tracking of the centreline"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig("/home/jack/.claude/jobs/3710eca5/tmp/planned_squares.png", dpi=110)
print("saved planned_squares.png")
