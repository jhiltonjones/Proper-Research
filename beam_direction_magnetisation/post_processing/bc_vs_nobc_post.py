
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import StringIO

# ------------------------------------------------------------
# Output folder
# ------------------------------------------------------------
out_dir = Path("mpc_boundary_condition_figures_run2")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Option A: load from file paths
# ------------------------------------------------------------
use_file_paths = True

with_bc_path = Path("/home/jack/Proper-Research/control_run_log_test_run_testing copy 12.csv")
without_bc_path = Path("/home/jack/Proper-Research/control_run_log_test_run_testing copy 9.csv")

# ------------------------------------------------------------
# Option B: paste CSV text directly
# ------------------------------------------------------------
with_bc_csv = r"""
PASTE_WITH_BOUNDARY_CONDITIONS_CSV_HERE
"""

without_bc_csv = r"""
PASTE_WITHOUT_BOUNDARY_CONDITIONS_CSV_HERE
"""

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
if use_file_paths:
    with_bc = pd.read_csv(with_bc_path)
    without_bc = pd.read_csv(without_bc_path)
else:
    with_bc = pd.read_csv(StringIO(with_bc_csv.strip()))
    without_bc = pd.read_csv(StringIO(without_bc_csv.strip()))

# Clean column names
with_bc.columns = with_bc.columns.str.strip()
without_bc.columns = without_bc.columns.str.strip()

with_bc["condition"] = "With boundary conditions"
without_bc["condition"] = "Without boundary conditions"

# ------------------------------------------------------------
# Numeric conversion
# ------------------------------------------------------------
numeric_cols = [
    "step",
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

for d in [with_bc, without_bc]:
    for col in numeric_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

df = pd.concat([with_bc, without_bc], ignore_index=True)

# ------------------------------------------------------------
# Check columns
# ------------------------------------------------------------
required_cols = [
    "step",
    "pred_meas_err_xy",
    "one_step_pred_err_xy",
    "raw_model_err_xy",
    "closest_distance_mm",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nAvailable columns:\n{df.columns.tolist()}")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def safe_mean(d, col):
    return pd.to_numeric(d[col], errors="coerce").mean() if col in d.columns else np.nan

def safe_median(d, col):
    return pd.to_numeric(d[col], errors="coerce").median() if col in d.columns else np.nan

def safe_max(d, col):
    return pd.to_numeric(d[col], errors="coerce").max() if col in d.columns else np.nan

def safe_min(d, col):
    return pd.to_numeric(d[col], errors="coerce").min() if col in d.columns else np.nan

def safe_last(d, col):
    if col not in d.columns:
        return np.nan
    s = pd.to_numeric(d[col], errors="coerce").dropna()
    return s.iloc[-1] if len(s) else np.nan

def summarize(d, label):
    return {
        "Condition": label,
        "Steps": len(d),
        "Step min": safe_min(d, "step"),
        "Step max": safe_max(d, "step"),

        "Mean measured-reference error [mm]": safe_mean(d, "meas_ref_err_xy"),
        "Max measured-reference error [mm]": safe_max(d, "meas_ref_err_xy"),

        "Mean predicted-measured error [mm]": safe_mean(d, "pred_meas_err_xy"),
        "Median predicted-measured error [mm]": safe_median(d, "pred_meas_err_xy"),
        "Max predicted-measured error [mm]": safe_max(d, "pred_meas_err_xy"),

        "Mean one-step prediction error [mm]": safe_mean(d, "one_step_pred_err_xy"),
        "Max one-step prediction error [mm]": safe_max(d, "one_step_pred_err_xy"),

        "Mean raw model error [mm]": safe_mean(d, "raw_model_err_xy"),
        "Median raw model error [mm]": safe_median(d, "raw_model_err_xy"),
        "Max raw model error [mm]": safe_max(d, "raw_model_err_xy"),

        "Mean prediction-reference error [mm]": safe_mean(d, "pred_ref_err_xy"),
        "Max prediction-reference error [mm]": safe_max(d, "pred_ref_err_xy"),

        "Mean wall clearance [mm]": safe_mean(d, "closest_distance_mm"),
        "Minimum wall clearance [mm]": safe_min(d, "closest_distance_mm"),

        "Mean Jacobian gain": safe_mean(d, "jac_gain_real_over_pred"),
        "Median Jacobian gain": safe_median(d, "jac_gain_real_over_pred"),
        "Mean Jacobian directional cosine": safe_mean(d, "jac_cos_xy"),
        "Mean Jacobian XY error [mm]": safe_mean(d, "jac_err_xy_mm"),
        "Max Jacobian XY error [mm]": safe_max(d, "jac_err_xy_mm"),

        "Final reference index": safe_last(d, "i_ref"),
    }

# ------------------------------------------------------------
# Full-run summary
# ------------------------------------------------------------
summary_full = pd.DataFrame(
    [
        summarize(with_bc, "With boundary conditions"),
        summarize(without_bc, "Without boundary conditions"),
    ]
)

# ------------------------------------------------------------
# Common-step summary
# This is the fairest comparison when the runs have different lengths.
# ------------------------------------------------------------
max_common_step = min(
    safe_max(with_bc, "step"),
    safe_max(without_bc, "step"),
)

with_bc_common = with_bc[with_bc["step"] <= max_common_step].copy()
without_bc_common = without_bc[without_bc["step"] <= max_common_step].copy()

summary_common = pd.DataFrame(
    [
        summarize(with_bc_common, "With boundary conditions"),
        summarize(without_bc_common, "Without boundary conditions"),
    ]
)

summary_full.to_csv(out_dir / "run2_summary_full.csv", index=False)
summary_common.to_csv(out_dir / "run2_summary_common_steps.csv", index=False)

print("\nFull-run summary:")
print(summary_full.T)

print("\nCommon-step summary:")
print(summary_common.T)

# ------------------------------------------------------------
# Long comparison table from common-step summary
# ------------------------------------------------------------
metrics_to_compare = [
    "Mean predicted-measured error [mm]",
    "Median predicted-measured error [mm]",
    "Max predicted-measured error [mm]",
    "Mean one-step prediction error [mm]",
    "Max one-step prediction error [mm]",
    "Mean raw model error [mm]",
    "Median raw model error [mm]",
    "Max raw model error [mm]",
    "Mean wall clearance [mm]",
    "Minimum wall clearance [mm]",
    "Mean Jacobian gain",
    "Median Jacobian gain",
    "Mean Jacobian directional cosine",
    "Mean Jacobian XY error [mm]",
    "Max Jacobian XY error [mm]",
]

with_row = summary_common[summary_common["Condition"] == "With boundary conditions"].iloc[0]
without_row = summary_common[summary_common["Condition"] == "Without boundary conditions"].iloc[0]

comparison_rows = []

for metric in metrics_to_compare:
    with_val = with_row[metric]
    without_val = without_row[metric]

    if pd.isna(with_val) or pd.isna(without_val):
        change_text = "n/a"
    elif "clearance" in metric.lower():
        change = with_val - without_val
        change_text = f"{change:+.3f} mm"
    elif "gain" in metric.lower():
        with_dist = abs(1.0 - with_val)
        without_dist = abs(1.0 - without_val)
        change_text = "closer to 1" if with_dist < without_dist else "farther from 1"
    elif "cosine" in metric.lower():
        change = with_val - without_val
        change_text = f"{change:+.3f}"
    else:
        if with_val == 0:
            change_text = "n/a"
        else:
            ratio = without_val / with_val
            change_text = f"{ratio:.2f}x lower" if ratio > 1 else f"{1/ratio:.2f}x higher"

    comparison_rows.append(
        {
            "Metric": metric,
            "With BC": with_val,
            "Without BC": without_val,
            "Change": change_text,
        }
    )

comparison_common = pd.DataFrame(comparison_rows)
comparison_common.to_csv(out_dir / "run2_comparison_common_steps.csv", index=False)

print("\nCommon-step comparison:")
print(comparison_common)

# ------------------------------------------------------------
# Plot style helper
# ------------------------------------------------------------
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Figure 1: selected prediction/model errors, common-step comparison
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

fig, ax = plt.subplots(figsize=(7.4, 4.2))

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

ax.set_title("Prediction/model error comparison, common steps")
ax.set_ylabel("Error [mm]")
ax.set_xticks(x)
ax.set_xticklabels(error_metrics["metric"], rotation=20, ha="right")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

savefig(fig, "run2_prediction_model_error_common_steps")

# ------------------------------------------------------------
# Figure 2: raw model error time-series
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["raw_model_err_xy"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.axvline(max_common_step, linestyle="--", linewidth=1, label="End of common range")
ax.set_title("Raw model error over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Raw model error [mm]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

savefig(fig, "run2_raw_model_error_timeseries")

# ------------------------------------------------------------
# Figure 3: predicted-measured error time-series
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["pred_meas_err_xy"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.axvline(max_common_step, linestyle="--", linewidth=1, label="End of common range")
ax.set_title("Predicted-measured error over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Predicted-measured error [mm]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

savefig(fig, "run2_pred_meas_error_timeseries")

# ------------------------------------------------------------
# Figure 4: wall clearance time-series
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["closest_distance_mm"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.axvline(max_common_step, linestyle="--", linewidth=1, label="End of common range")
ax.set_title("Wall clearance over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Closest wall/centre distance [mm]")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

savefig(fig, "run2_wall_clearance_timeseries")

# ------------------------------------------------------------
# Figure 5: Jacobian gain time-series
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))

for condition, d in df.groupby("condition"):
    ax.plot(
        d["step"],
        d["jac_gain_real_over_pred"],
        marker="o",
        linewidth=1.5,
        label=condition,
    )

ax.axhline(1.0, linestyle="--", linewidth=1, label="Ideal gain")
ax.axvline(max_common_step, linestyle="--", linewidth=1, label="End of common range")
ax.set_title("Realised Jacobian gain over MPC steps")
ax.set_xlabel("Step")
ax.set_ylabel("Realised/predicted displacement gain")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)

savefig(fig, "run2_jacobian_gain_timeseries")

print("\nSaved all outputs in:", out_dir.resolve())