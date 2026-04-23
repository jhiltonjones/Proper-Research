import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/home/jack/Proper-Research/results_sweep_beam_theory/current_results.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Sort by sweep index if present
    if "j_idx" in df.columns:
        df = df.sort_values(["i_idx", "j_idx"]).reset_index(drop=True)

    print("\nColumns:")
    print(list(df.columns))

    # Basic numeric cleanup
    numeric_cols = [
        "j_idx",
        "pred_tip_angle_deg",
        "meas_tip_angle_deg",
        "err_x_mm",
        "err_y_mm",
        "err_z_mm",
        "err_xy_mm",
        "err_xyz_mm",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\nLowest 10 err_xy_mm rows:")
    cols_to_show = [
        c for c in [
            "i_idx", "j_idx",
            "pred_tip_angle_deg", "meas_tip_angle_deg",
            "err_x_mm", "err_y_mm", "err_xy_mm", "err_xyz_mm"
        ] if c in df.columns
    ]
    print(df.nsmallest(10, "err_xy_mm")[cols_to_show].to_string(index=False))

    # 1) measured angle vs xy error
    plt.figure(figsize=(8, 5))
    plt.plot(df["meas_tip_angle_deg"], df["err_xy_mm"], "o-")
    plt.xlabel("Measured tip angle [deg]")
    plt.ylabel("XY error [mm]")
    plt.title("Measured tip angle vs XY error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 2) predicted angle vs xy error
    plt.figure(figsize=(8, 5))
    plt.plot(df["pred_tip_angle_deg"], df["err_xy_mm"], "o-")
    plt.xlabel("Predicted tip angle [deg]")
    plt.ylabel("XY error [mm]")
    plt.title("Predicted tip angle vs XY error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 3) sweep index vs error
    if "j_idx" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["j_idx"], df["err_xy_mm"], "o-", label="err_xy_mm")
        if "err_xyz_mm" in df.columns:
            plt.plot(df["j_idx"], df["err_xyz_mm"], "s-", label="err_xyz_mm")
        plt.xlabel("j_idx")
        plt.ylabel("Error [mm]")
        plt.title("Sweep index vs error")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # 4) measured vs predicted angle
    plt.figure(figsize=(8, 5))
    plt.plot(df["j_idx"], df["meas_tip_angle_deg"], "o-", label="Measured angle")
    plt.plot(df["j_idx"], df["pred_tip_angle_deg"], "s-", label="Predicted angle")
    plt.xlabel("j_idx")
    plt.ylabel("Tip angle [deg]")
    plt.title("Measured vs predicted tip angle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()