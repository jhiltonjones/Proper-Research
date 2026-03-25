import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/jack/Proper-Research/results_sweep_forward_validation2/sweep_results.csv")

# 1) Predicted vs measured XY tip path
plt.figure(figsize=(7, 6))
plt.plot(df["pred_base_x_mm"], df["pred_base_y_mm"], "o-", label="Predicted")
plt.plot(df["meas_base_x_mm"], df["meas_base_y_mm"], "s-", label="Measured")
plt.xlabel("Base-local x [mm]")
plt.ylabel("Base-local y [mm]")
plt.title("Predicted vs measured tip path")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2) X vs j_idx
plt.figure(figsize=(8, 5))
plt.plot(df["j_idx"], df["pred_base_x_mm"], "o-", label="Predicted x")
plt.plot(df["j_idx"], df["meas_base_x_mm"], "s-", label="Measured x")
plt.xlabel("j_idx")
plt.ylabel("Tip x [mm]")
plt.title("Tip x vs j_idx")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# 3) Y vs j_idx
plt.figure(figsize=(8, 5))
plt.plot(df["j_idx"], df["pred_base_y_mm"], "o-", label="Predicted y")
plt.plot(df["j_idx"], df["meas_base_y_mm"], "s-", label="Measured y")
plt.xlabel("j_idx")
plt.ylabel("Tip y [mm]")
plt.title("Tip y vs j_idx")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# 4) Angle vs j_idx
plt.figure(figsize=(8, 5))
plt.plot(df["j_idx"], df["pred_tip_angle_deg"], "o-", label="Predicted angle")
plt.plot(df["j_idx"], df["meas_tip_angle_deg"], "s-", label="Measured angle")
plt.xlabel("j_idx")
plt.ylabel("Tip angle [deg]")
plt.title("Predicted vs measured tip angle")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# 5) Error vs j_idx
plt.figure(figsize=(8, 5))
plt.plot(df["j_idx"], df["err_x_mm"], "o-", label="err x")
plt.plot(df["j_idx"], df["err_y_mm"], "s-", label="err y")
plt.plot(df["j_idx"], df["err_xy_mm"], "^-", label="err xy")
plt.xlabel("j_idx")
plt.ylabel("Error [mm]")
plt.title("Prediction error vs j_idx")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()
import json
import numpy as np
import matplotlib.pyplot as plt

with open("results_sweep_forward_validation2/sweep_results.json", "r") as f:
    data = json.load(f)

row = next(r for r in data if r["j_idx"] == 30)

pred = np.asarray(row["pred_beam_local_mm"], dtype=float)
meas = np.asarray(row["meas_beam_local_mm"], dtype=float)

# measured beam is tip -> base, so reverse it
meas = meas[::-1]

plt.figure(figsize=(7, 7))
plt.plot(pred[:, 0], pred[:, 1], "-o", markersize=2, label="Predicted")
plt.plot(meas[:, 0], meas[:, 1], "-o", markersize=2, label="Measured (reversed)")
plt.axis("equal")
plt.grid(True)
plt.xlabel("Local x [mm]")
plt.ylabel("Local y [mm]")
plt.title(f"Beam shape comparison, j_idx={row['j_idx']}")
plt.legend()
plt.tight_layout()
plt.show()