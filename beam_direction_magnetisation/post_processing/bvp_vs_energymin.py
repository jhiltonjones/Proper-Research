import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sweep_heatmap(
    df,
    *,
    value_col,
    title,
    cbar_label,
    save_path=None,
    show=True,
):
    if value_col not in df.columns:
        raise KeyError(
            f"Column '{value_col}' not found. Available columns are:\n"
            f"{list(df.columns)}"
        )

    df = df.copy()

    if "L_mm" not in df.columns:
        df["L_mm"] = 1e3 * df["L_cmd"]

    pivot = (
        df.pivot(
            index="L_mm",
            columns="angle_deg",
            values=value_col,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    L_vals = pivot.index.values
    angle_vals = pivot.columns.values
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[
            angle_vals[0],
            angle_vals[-1],
            L_vals[0],
            L_vals[-1],
        ],
    )

    ax.set_xlabel("Magnet arc angle [deg]")
    ax.set_ylabel("Insertion length [mm]")
    ax.set_title(title)

    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_line_by_length(
    df,
    *,
    value_col,
    ylabel,
    title,
    save_path=None,
    show=True,
):
    if value_col not in df.columns:
        raise KeyError(
            f"Column '{value_col}' not found. Available columns are:\n"
            f"{list(df.columns)}"
        )

    df = df.copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    for L_cmd, group in df.groupby("L_cmd"):
        group = group.sort_values("angle_deg")
        ax.plot(
            group["angle_deg"],
            group[value_col],
            marker="o",
            label=f"L = {1e3 * L_cmd:.0f} mm",
        )

    ax.set_xlabel("Magnet arc angle [deg]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_summary_by_length(df):
    df = df.copy()

    if "L_mm" not in df.columns:
        df["L_mm"] = 1e3 * df["L_cmd"]

    summary = (
        df.groupby("L_mm")
        .agg(
            max_energy_deflection_mm=("energy_deflection_mm", "max"),
            max_bvp_deflection_mm=("bvp_deflection_mm", "max"),
            max_tip_error_mm=("tip_error_bvp_energy_mm", "max"),
            max_rms_centerline_error_mm=("rms_centerline_error_mm", "max"),
            max_centerline_error_mm=("max_centerline_error_mm", "max"),
            max_W_el=("W_el", "max"),
            min_W_m=("W_m", "min"),
        )
        .reset_index()
    )

    return summary


def load_sweep_csv(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Remove failed rows if present.
    if "failed" in df.columns:
        df = df[df["failed"] != True].copy()

    required = [
        "L_cmd",
        "angle_deg",
        "energy_deflection_mm",
        "bvp_deflection_mm",
        "tip_error_bvp_energy_mm",
        "rms_centerline_error_mm",
        "W_m",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"The CSV is missing required columns: {missing}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    df["L_mm"] = 1e3 * df["L_cmd"]

    return df


def plot_all(csv_path, output_dir="beam_sweep_graphs", show=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_sweep_csv(csv_path)

    # Save a cleaned copy.
    df.to_csv(output_dir / "parsed_beam_sweep_data.csv", index=False)

    # Heatmaps.
    plot_sweep_heatmap(
        df,
        value_col="energy_deflection_mm",
        title="Energy-min tip deflection from straight baseline",
        cbar_label="Deflection [mm]",
        save_path=output_dir / "heatmap_energy_deflection_mm.png",
        show=show,
    )

    plot_sweep_heatmap(
        df,
        value_col="bvp_deflection_mm",
        title="Cosserat-rod  tip deflection from straight baseline",
        cbar_label="Deflection [mm]",
        save_path=output_dir / "heatmap_bvp_deflection_mm.png",
        show=show,
    )

    plot_sweep_heatmap(
        df,
        value_col="tip_error_bvp_energy_mm",
        title="Cosserat-rod  vs energy-min tip error",
        cbar_label="Tip error [mm]",
        save_path=output_dir / "heatmap_bvp_energy_tip_error_mm.png",
        show=show,
    )

    plot_sweep_heatmap(
        df,
        value_col="rms_centerline_error_mm",
        title="Cosserat-rod  vs energy-min RMS centerline error",
        cbar_label="RMS error [mm]",
        save_path=output_dir / "heatmap_rms_centerline_error_mm.png",
        show=show,
    )

    plot_sweep_heatmap(
        df,
        value_col="W_m",
        title="Magnetic energy term",
        cbar_label=r"$W_m$ [J]",
        save_path=output_dir / "heatmap_magnetic_energy.png",
        show=show,
    )

    # Optional additional heatmap if present.
    if "max_centerline_error_mm" in df.columns:
        plot_sweep_heatmap(
            df,
            value_col="max_centerline_error_mm",
            title="Cosserat-rod  vs energy-min max centerline error",
            cbar_label="Max error [mm]",
            save_path=output_dir / "heatmap_max_centerline_error_mm.png",
            show=show,
        )

    # Line plots.
    plot_line_by_length(
        df,
        value_col="energy_deflection_mm",
        ylabel="Energy-min deflection [mm]",
        title="Energy-min deflection versus magnet angle",
        save_path=output_dir / "line_energy_deflection_vs_angle.png",
        show=show,
    )

    plot_line_by_length(
        df,
        value_col="bvp_deflection_mm",
        ylabel="Cosserat-rod deflection [mm]",
        title="Cosserat-rod deflection versus magnet angle",
        save_path=output_dir / "line_bvp_deflection_vs_angle.png",
        show=show,
    )

    plot_line_by_length(
        df,
        value_col="tip_error_bvp_energy_mm",
        ylabel="Tip error [mm]",
        title="Tip error of Cosserat-rod model and Energy Minimisation",
        save_path=output_dir / "line_tip_error_vs_angle.png",
        show=show,
    )

    plot_line_by_length(
        df,
        value_col="rms_centerline_error_mm",
        ylabel="RMS centerline error [mm]",
        title="Cosserat-rod and Energy Minimisation RMS centerline error versus magnet angle",
        save_path=output_dir / "line_rms_centerline_error_vs_angle.png",
        show=show,
    )

    # Summary.
    summary = make_summary_by_length(df)
    summary.to_csv(output_dir / "summary_by_length.csv", index=False)

    print("\nSaved plots and tables to:")
    print(output_dir.resolve())

    print("\nSummary by length:")
    print(summary.to_string(index=False))

    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot length-angle sweep results from CSV."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to length_angle_sweep_results.csv",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="beam_sweep_graphs",
        help="Output directory for plots and parsed CSV.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save plots without opening interactive windows.",
    )

    args = parser.parse_args()

    plot_all(
        csv_path=args.csv,
        output_dir=args.out,
        show=not args.no_show,
    )


# ============================================================
# Set paths here
# ============================================================
if __name__ == "__main__":

    CSV_PATH = Path(
        "/Users/jackhilton-jones/Proper-Research/results/no_contact_length_angle_sweep_N35/length_angle_sweep_results.csv"
    )

    OUTPUT_DIR = Path(
        "results/no_contact_length_angle_sweep_N35/plots"
    )

    SHOW_PLOTS = True

    df, summary = plot_all(
        csv_path=CSV_PATH,
        output_dir=OUTPUT_DIR,
        show=SHOW_PLOTS,
    )