"""Compare inverse-Jacobian vs MPC-LTI on the triangle, from tip_trajectory.csv."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

dirs = json.load(open("/home/jack/.claude/jobs/3710eca5/tmp/compare_dirs.json"))
bp = np.array([0.525575, -0.670028, -0.016567]); u = np.array([-1., 0, 0]); v = np.array([0, 1., 0])

LABEL = {"naive_inverse_jacobian": "inverse Jacobian", "mpc_lti": "MPC-LTI"}
COL = {"naive_inverse_jacobian": "tab:blue", "mpc_lti": "tab:red"}

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
rows = []
for kind, d in dirs.items():
    csv = f"{d}/tip_trajectory.csv"
    a = np.genfromtxt(csv, delimiter=",", names=True)
    tip = np.c_[a["tip_x_m"], a["tip_y_m"], a["tip_z_m"]]
    des = np.c_[a["des_x_m"], a["des_y_m"], a["des_z_m"]]
    err = a["err_norm_mm"]
    t = a["t_s"]
    ins = a["insertion_m"] * 1e3
    hold = a["terminal_hold"].astype(bool)
    tracking = ~hold
    tip_uv = np.c_[(tip - bp) @ u, (tip - bp) @ v] * 1e3
    des_uv = np.c_[(des - bp) @ u, (des - bp) @ v] * 1e3
    c = COL[kind]; lab = LABEL[kind]
    ax[0].plot(des_uv[:, 1], des_uv[:, 0], "k--", lw=1, alpha=0.5)
    ax[0].plot(tip_uv[:, 1], tip_uv[:, 0], "-", color=c, label=lab)
    ax[1].plot(t, err, "-", color=c, label=lab)
    ax[2].plot(t, ins, "-", color=c, label=lab)
    rows.append((lab,
                 float(np.sqrt(np.mean(err[tracking] ** 2))),
                 float(np.max(err[tracking])),
                 float(np.mean(err[hold])) if hold.any() else float("nan"),
                 float(err[-1])))

ax[0].plot([], [], "k--", label="desired")
ax[0].set_xlabel("tip B.y  (in-plane, mm)"); ax[0].set_ylabel("tip B.x  (axial, mm)")
ax[0].set_title("tip path"); ax[0].set_aspect("equal"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("tracking error [mm]"); ax[1].set_title("position error vs time")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("insertion [mm]"); ax[2].set_title("insertion vs time")
ax[2].grid(alpha=0.3); ax[2].legend(fontsize=9)
fig.tight_layout()
fig.savefig("/home/jack/.claude/jobs/3710eca5/tmp/compare_controllers.png", dpi=110)

print(f"{'controller':20s} {'RMS(track)':>12s} {'max(track)':>12s} {'mean(hold)':>12s} {'final':>10s}   [mm]")
for r in rows:
    print(f"{r[0]:20s} {r[1]:12.2f} {r[2]:12.2f} {r[3]:12.2f} {r[4]:10.2f}")
json.dump({r[0]: {"rms_track_mm": r[1], "max_track_mm": r[2], "mean_hold_mm": r[3], "final_mm": r[4]} for r in rows},
          open("/home/jack/.claude/jobs/3710eca5/tmp/compare_metrics.json", "w"), indent=1)
print("\nsaved compare_controllers.png")
