import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load CSV
# --------------------------------------------------
csv_path = "/Users/jackhilton-jones/Proper-Research/comparison_outputs/model_comparison_angle_sweep_25_node.csv"   # change if needed
df = pd.read_csv(csv_path)

# Sort by angle so plots are clean
df = df.sort_values("angle_deg").reset_index(drop=True)

# --------------------------------------------------
# Basic summary
# --------------------------------------------------
print("=" * 80)
print("DATA SUMMARY")
print("=" * 80)
print(f"Number of samples: {len(df)}")
print(f"Angle range: {df['angle_deg'].min()} to {df['angle_deg'].max()} deg")
print()

for col in ["cos_wrapper_success", "der_wrapper_success", "cos_bvp_direct_success"]:
    if col in df.columns:
        success_count = df[col].fillna(False).astype(bool).sum()
        print(f"{col:28s}: {success_count}/{len(df)} successful")

print()
print("Mean pairwise tip differences:")
for col in [
    "tipdiff_cos_wrapper_vs_der_wrapper",
    "tipdiff_cos_wrapper_vs_cos_bvp_direct",
    "tipdiff_der_wrapper_vs_cos_bvp_direct",
]:
    if col in df.columns:
        print(f"{col:35s}: mean={df[col].mean():.6e}, max={df[col].max():.6e}")

# --------------------------------------------------
# Helper: unwrap angle columns and convert to degrees
# --------------------------------------------------
def unwrap_to_deg(rad_series):
    arr = rad_series.to_numpy(dtype=float)
    return np.rad2deg(np.unwrap(arr))

# --------------------------------------------------
# Plot 1: solver success vs angle
# --------------------------------------------------
plt.figure(figsize=(8, 4))
for col, label in [
    ("cos_wrapper_success", "Cosserat wrapper"),
    ("der_wrapper_success", "DER wrapper"),
    ("cos_bvp_direct_success", "Cosserat BVP direct"),
]:
    if col in df.columns:
        y = df[col].fillna(False).astype(int)
        plt.plot(df["angle_deg"], y, marker="o", label=label)

plt.xlabel("Magnet angle [deg]")
plt.ylabel("Success (1=True, 0=False)")
plt.title("Solver success vs magnet angle")
plt.yticks([0, 1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 2: tip y-position comparison
# Often the clearest coordinate for comparing bending trend
# --------------------------------------------------
plt.figure(figsize=(8, 5))
for col, label in [
    ("cos_wrapper_tip_y", "Cosserat wrapper"),
    ("der_wrapper_tip_y", "DER wrapper"),
    ("cos_bvp_direct_tip_y", "Cosserat BVP direct"),
]:
    if col in df.columns:
        plt.plot(df["angle_deg"], df[col], marker="o", label=label)

plt.xlabel("Magnet angle [deg]")
plt.ylabel("Tip y-position [m]")
plt.title("Tip y-position vs magnet angle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 3: tip x/y/z for the two most reliable models
# --------------------------------------------------
fig = plt.figure(figsize=(9, 10))

coords = ["x", "y", "z"]
for i, coord in enumerate(coords, start=1):
    plt.subplot(3, 1, i)

    cw = f"cos_wrapper_tip_{coord}"
    cb = f"cos_bvp_direct_tip_{coord}"

    if cw in df.columns:
        plt.plot(df["angle_deg"], df[cw], marker="o", label=f"cos_wrapper {coord}")
    if cb in df.columns:
        plt.plot(df["angle_deg"], df[cb], marker="o", label=f"cos_bvp_direct {coord}")

    plt.xlabel("Magnet angle [deg]")
    plt.ylabel(f"Tip {coord} [m]")
    plt.grid(True)
    plt.legend()

plt.suptitle("Tip coordinates vs magnet angle")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 4: pairwise tip differences
# --------------------------------------------------
plt.figure(figsize=(8, 5))
for col, label in [
    ("tipdiff_cos_wrapper_vs_der_wrapper", "cos_wrapper vs der_wrapper"),
    ("tipdiff_cos_wrapper_vs_cos_bvp_direct", "cos_wrapper vs cos_bvp_direct"),
    ("tipdiff_der_wrapper_vs_cos_bvp_direct", "der_wrapper vs cos_bvp_direct"),
]:
    if col in df.columns:
        plt.plot(df["angle_deg"], df[col], marker="o", label=label)

plt.xlabel("Magnet angle [deg]")
plt.ylabel("Tip difference norm [m]")
plt.title("Pairwise model disagreement vs magnet angle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 5: Cosserat BVP magnetic field at tip
# --------------------------------------------------
if "cos_bvp_direct_B_tip" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(df["angle_deg"], df["cos_bvp_direct_B_tip"], marker="o")
    plt.xlabel("Magnet angle [deg]")
    plt.ylabel("B_tip [T]")
    plt.title("Cosserat BVP tip magnetic field vs angle")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Plot 6: Cosserat BVP angles
# theta_y is wrapped, so unwrap it first
# --------------------------------------------------
if "cos_bvp_direct_theta_y" in df.columns:
    theta_y_deg = unwrap_to_deg(df["cos_bvp_direct_theta_y"])

    plt.figure(figsize=(8, 5))
    plt.plot(df["angle_deg"], theta_y_deg, marker="o")
    plt.xlabel("Magnet angle [deg]")
    plt.ylabel("Unwrapped theta_y [deg]")
    plt.title("Cosserat BVP theta_y vs angle")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if "cos_bvp_direct_theta_total" in df.columns:
    theta_total_deg = np.rad2deg(df["cos_bvp_direct_theta_total"].to_numpy(dtype=float))

    plt.figure(figsize=(8, 5))
    plt.plot(df["angle_deg"], theta_total_deg, marker="o")
    plt.xlabel("Magnet angle [deg]")
    plt.ylabel("theta_total [deg]")
    plt.title("Cosserat BVP total bending angle vs magnet angle")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Optional: save a compact interpreted summary CSV
# --------------------------------------------------
summary_cols = [
    "angle_deg",
    "cos_wrapper_success",
    "der_wrapper_success",
    "cos_bvp_direct_success",
    "cos_wrapper_tip_x", "cos_wrapper_tip_y", "cos_wrapper_tip_z",
    "der_wrapper_tip_x", "der_wrapper_tip_y", "der_wrapper_tip_z",
    "cos_bvp_direct_tip_x", "cos_bvp_direct_tip_y", "cos_bvp_direct_tip_z",
    "tipdiff_cos_wrapper_vs_der_wrapper",
    "tipdiff_cos_wrapper_vs_cos_bvp_direct",
    "tipdiff_der_wrapper_vs_cos_bvp_direct",
    "cos_bvp_direct_B_tip",
]
summary_cols = [c for c in summary_cols if c in df.columns]

df_summary = df[summary_cols].copy()
df_summary.to_csv("model_comparison_summary.csv", index=False)
print("\nSaved compact summary to model_comparison_summary.csv")