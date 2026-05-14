import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from pathlib import Path

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir = Path("mpc_boundary_condition_figures")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# CSV file paths
# ------------------------------------------------------------
with_bc_path = Path("/home/jack/Proper-Research/control_run_log_test_run_testing copy 6.csv")
without_bc_path = Path("/home/jack/Proper-Research/control_run_log_test_run_testing.csv")

# ------------------------------------------------------------
# Load data from files
# ------------------------------------------------------------
with_bc = pd.read_csv(with_bc_path)
without_bc = pd.read_csv(without_bc_path)

# Clean column names in case there are hidden spaces
with_bc.columns = with_bc.columns.str.strip()
without_bc.columns = without_bc.columns.str.strip()

print("With-BC columns:")
print(with_bc.columns.tolist())

print("\nWithout-BC columns:")
print(without_bc.columns.tolist())

with_bc["condition"] = "With boundary conditions"
without_bc["condition"] = "Without boundary conditions"

df = pd.concat([with_bc, without_bc], ignore_index=True)

# ------------------------------------------------------------
# Ensure numeric columns are numeric
# ------------------------------------------------------------
numeric_cols = [
    "meas_ref_err_xy",
    "raw_model_err_xy",
    "one_step_pred_err_xy",
    "pred_meas_err_xy",
    "pred_ref_err_xy",
    "closest_distance_mm",
    "jac_gain_real_over_pred",
    "jac_cos_xy",
    "jac_err_xy_mm",
    "i_ref",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------
def summarize_condition(data, condition_name):
    d = data[data["condition"] == condition_name].copy()

    summary = {
        "Condition": condition_name,

        "Steps": len(d),

        "Mean measured-reference error [mm]": d["meas_ref_err_xy"].mean(),
        "Max measured-reference error [mm]": d["meas_ref_err_xy"].max(),

        "Mean predicted-measured error [mm]": d["pred_meas_err_xy"].mean(),
        "Max predicted-measured error [mm]": d["pred_meas_err_xy"].max(),

        "Mean one-step prediction error [mm]": d["one_step_pred_err_xy"].mean(),
        "Max one-step prediction error [mm]": d["one_step_pred_err_xy"].max(),

        "Mean raw model error [mm]": d["raw_model_err_xy"].mean(),
        "Max raw model error [mm]": d["raw_model_err_xy"].max(),

        "Mean prediction-reference error [mm]": d["pred_ref_err_xy"].mean(),
        "Max prediction-reference error [mm]": d["pred_ref_err_xy"].max(),

        "Mean wall clearance [mm]": d["closest_distance_mm"].mean(),
        "Minimum wall clearance [mm]": d["closest_distance_mm"].min(),

        "Mean Jacobian gain": d["jac_gain_real_over_pred"].mean(),
        "Mean Jacobian directional cosine": d["jac_cos_xy"].mean(),
        "Mean Jacobian XY error [mm]": d["jac_err_xy_mm"].mean(),
        "Max Jacobian XY error [mm]": d["jac_err_xy_mm"].max(),

        "Final reference index": d["i_ref"].dropna().iloc[-1],
    }

    return summary

summary = pd.DataFrame(
    [
        summarize_condition(df, "With boundary conditions"),
        summarize_condition(df, "Without boundary conditions"),
    ]
)

# ------------------------------------------------------------
# Convert to long comparison table
# ------------------------------------------------------------
comparison_rows = []

metrics_to_compare = [
    "Mean measured-reference error [mm]",
    "Max measured-reference error [mm]",
    "Mean predicted-measured error [mm]",
    "Max predicted-measured error [mm]",
    "Mean one-step prediction error [mm]",
    "Max one-step prediction error [mm]",
    "Mean raw model error [mm]",
    "Max raw model error [mm]",
    "Mean prediction-reference error [mm]",
    "Max prediction-reference error [mm]",
    "Mean wall clearance [mm]",
    "Minimum wall clearance [mm]",
    "Mean Jacobian gain",
    "Mean Jacobian directional cosine",
    "Mean Jacobian XY error [mm]",
    "Max Jacobian XY error [mm]",
]

with_row = summary[summary["Condition"] == "With boundary conditions"].iloc[0]
without_row = summary[summary["Condition"] == "Without boundary conditions"].iloc[0]

for metric in metrics_to_compare:
    with_val = with_row[metric]
    without_val = without_row[metric]

    if "clearance" in metric.lower():
        change = with_val - without_val
        change_text = f"{change:+.3f} mm"
    elif "gain" in metric.lower():
        change = abs(1.0 - without_val) - abs(1.0 - with_val)
        change_text = "closer to 1" if change > 0 else "not closer to 1"
    elif "cosine" in metric.lower():
        change = with_val - without_val
        change_text = f"{change:+.3f}"
    else:
        change = without_val / with_val if with_val != 0 else np.nan
        change_text = f"{change:.2f}x lower" if change > 1 else f"{1/change:.2f}x higher"

    comparison_rows.append(
        {
            "Metric": metric,
            "With BC": with_val,
            "Without BC": without_val,
            "Change": change_text,
        }
    )

comparison = pd.DataFrame(comparison_rows)

# Print summary tables
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 140)
pd.set_option("display.precision", 4)

print("\nSummary:")
print(summary.T)

print("\nComparison:")
print(comparison)

# Save CSV versions
summary.to_csv(out_dir / "h1_60deg_summary.csv", index=False)
comparison.to_csv(out_dir / "h1_60deg_comparison.csv", index=False)

# ------------------------------------------------------------
# Figure 1: representative prediction/model errors
# ------------------------------------------------------------
error_metrics = pd.DataFrame(
    [
        {
            "metric": "Mean predicted-measured",
            "with_bc": with_row["Mean predicted-measured error [mm]"],
            "without_bc": without_row["Mean predicted-measured error [mm]"],
        },
        {
            "metric": "Mean one-step",
            "with_bc": with_row["Mean one-step prediction error [mm]"],
            "without_bc": without_row["Mean one-step prediction error [mm]"],
        },
        {
            "metric": "Mean raw model",
            "with_bc": with_row["Mean raw model error [mm]"],
            "without_bc": without_row["Mean raw model error [mm]"],
        },
        {
            "metric": "Max raw model",
            "with_bc": with_row["Max raw model error [mm]"],
            "without_bc": without_row["Max raw model error [mm]"],
        },
    ]
)

x = np.arange(len(error_metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(7.2, 4.2))

ax.bar(
    x - width / 2,
    error_metrics["with_bc"],
    width,
    label="With boundary conditions",
)

ax.bar(
    x + width / 2,
    error_metrics["without_bc"],
    width,
    label="Without boundary conditions",
)

ax.set_title("One-step MPC prediction error in 60-degree vessel")
ax.set_ylabel("Error [mm]")
ax.set_xticks(x)
ax.set_xticklabels(error_metrics["metric"], rotation=20, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "h1_60deg_prediction_error_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "h1_60deg_prediction_error_comparison.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Figure 2: wall clearance and Jacobian validity
# ------------------------------------------------------------
safety_metrics = pd.DataFrame(
    [
        {
            "metric": "Mean wall clearance",
            "with_bc": with_row["Mean wall clearance [mm]"],
            "without_bc": without_row["Mean wall clearance [mm]"],
            "ylabel": "Clearance [mm]",
        },
        {
            "metric": "Minimum wall clearance",
            "with_bc": with_row["Minimum wall clearance [mm]"],
            "without_bc": without_row["Minimum wall clearance [mm]"],
            "ylabel": "Clearance [mm]",
        },
    ]
)

x = np.arange(len(safety_metrics))
width = 0.36

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.bar(
    x - width / 2,
    safety_metrics["with_bc"],
    width,
    label="With boundary conditions",
)

ax.bar(
    x + width / 2,
    safety_metrics["without_bc"],
    width,
    label="Without boundary conditions",
)

ax.set_title("One-step MPC wall clearance in 60-degree vessel")
ax.set_ylabel("Closest wall/centre distance [mm]")
ax.set_xticks(x)
ax.set_xticklabels(safety_metrics["metric"], rotation=15, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "h1_60deg_wall_clearance_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "h1_60deg_wall_clearance_comparison.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Figure 3: time-series prediction error
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["pred_meas_err_xy"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.set_title("Predicted-measured error over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Predicted-measured error [mm]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "h1_60deg_pred_meas_error_timeseries.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "h1_60deg_pred_meas_error_timeseries.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------
# Figure 4: raw model error over steps
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["raw_model_err_xy"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.set_title("Raw model error over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Raw model error [mm]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(out_dir / "h1_60deg_raw_model_error_timeseries.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "h1_60deg_raw_model_error_timeseries.pdf", bbox_inches="tight")
plt.close(fig)

print("\nSaved figures and tables in:", out_dir.resolve())