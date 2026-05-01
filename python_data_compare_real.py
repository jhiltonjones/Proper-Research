import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "/home/jack/Proper-Research/control_run_log_test_run_centreline_track_day2wboc90_inline2_try2_no_boundary.csv"  # <-- change this


df = pd.read_csv(CSV_FILE)

required_columns = [
    "meas_ref_err_xy",
    "raw_model_err_xy",
    "meas_ref_idx",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required CSV columns: {missing}")

if "step" in df.columns:
    x = df["step"]
    x_label = "step"
else:
    x = df.index
    x_label = "row index"


# Graph 1: error values
plt.figure(figsize=(12, 6))
plt.plot(x, df["meas_ref_err_xy"], label="meas_ref_err_xy")
plt.plot(x, df["raw_model_err_xy"], label="raw_model_err_xy")

plt.xlabel(x_label)
plt.ylabel("Error XY")
plt.title("Error Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Graph 2: meas_ref_idx
plt.figure(figsize=(12, 4))
plt.plot(x, df["meas_ref_idx"], label="meas_ref_idx")

plt.xlabel(x_label)
plt.ylabel("meas_ref_idx")
plt.title("Measurement Reference Index")
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.show()