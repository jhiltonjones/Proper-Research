import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load run 5 (CSV)
# ----------------------------
df5 = pd.read_csv("/home/jack/Proper-Research/results_sweep_forward_wlumen6/sweep_results.csv")

# ----------------------------
# Load run 6 (JSON)
# ----------------------------
with open("/home/jack/Proper-Research/results_sweep_forward_wo_lumen/sweep_results.json", "r") as f:
    data6 = json.load(f)

# Extract run 6 tip data
pred_x6 = []
pred_y6 = []
meas_x6 = []
meas_y6 = []

for r in data6:
    pred = np.asarray(r["pred_tip_base_local_m"], float) * 1e3
    meas = np.asarray(r["meas_tip_base_local_m"], float) * 1e3

    pred_x6.append(pred[0])
    pred_y6.append(pred[1])
    meas_x6.append(meas[0])
    meas_y6.append(meas[1])

# ----------------------------
# Plot everything together
# ----------------------------
plt.figure(figsize=(7, 7))

# Run 5
plt.plot(
    df5["pred_base_x_mm"],
    df5["pred_base_y_mm"],
    "o-",
    label="Predicted (run 5)"
)
plt.plot(
    df5["meas_base_x_mm"],
    df5["meas_base_y_mm"],
    "s-",
    label="Measured (run 5)"
)

# Run 6
# plt.plot(
#     pred_x6,
#     pred_y6,
#     "o--",
#     label="Predicted (run 6)"
# )
# plt.plot(
#     meas_x6,
#     meas_y6,
#     "s--",
#     label="Measured (run 6)"
# )

# Pivot origin
plt.plot(0, 0, "k+", markersize=10, label="Pivot")

plt.xlabel("Base-local x [mm]")
plt.ylabel("Base-local y [mm]")
plt.title("Predicted vs Measured Tip (Run 5 vs Run 6)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()