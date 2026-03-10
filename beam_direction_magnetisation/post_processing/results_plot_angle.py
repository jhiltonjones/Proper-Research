import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# EDIT THESE PATHS
# =========================
CSV_FILE_1 = r"mpc_run_90_low_tangent/log.csv"
CSV_FILE_2 = r"mpc_run_90_low_tangent_3step_ltv/log.csv"

# Choose x-axis column. If not found, row index will be used instead.
X_AXIS_COLUMN = "k"

# Set to None to show plots in a window instead of saving
SAVE_DIR = None
# Example:
# SAVE_DIR = r"/full/path/to/output_plots"
# =========================


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required_cols: list[str], csv_path: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in '{csv_path}': {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def get_label(path: str) -> str:
    return Path(path).stem


def prepare_x_axis(df1: pd.DataFrame, df2: pd.DataFrame, x_col: str):
    if x_col in df1.columns and x_col in df2.columns:
        return df1, df2, x_col

    print(f"Warning: column '{x_col}' not found in both files. Using row index instead.")
    df1 = df1.copy()
    df2 = df2.copy()
    df1["row_index"] = range(len(df1))
    df2["row_index"] = range(len(df2))
    return df1, df2, "row_index"


def plot_metric(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_col: str,
    y_col: str,
    label1: str,
    label2: str,
    title: str,
    ylabel: str,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df1[x_col], df1[y_col], label=label1, linewidth=1.5)
    plt.plot(df2[x_col], df2[y_col], label=label2, linewidth=1.5)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()


def main() -> None:
    required_columns = ["clearance", "vessel_tip alignment"]

    if not os.path.exists(CSV_FILE_1):
        raise FileNotFoundError(f"CSV_FILE_1 does not exist: {CSV_FILE_1}")
    if not os.path.exists(CSV_FILE_2):
        raise FileNotFoundError(f"CSV_FILE_2 does not exist: {CSV_FILE_2}")

    df1 = load_csv(CSV_FILE_1)
    df2 = load_csv(CSV_FILE_2)

    validate_columns(df1, required_columns, CSV_FILE_1)
    validate_columns(df2, required_columns, CSV_FILE_2)

    df1, df2, x_col = prepare_x_axis(df1, df2, X_AXIS_COLUMN)

    df1 = df1[(df1[x_col] >= 1) & (df1[x_col] <=24)].copy()
    df2 = df2[(df2[x_col] >= 1) & (df2[x_col] <= 24)].copy()

    label1="1 step",
    label2="3 step",

    combined_save_path = None
    if SAVE_DIR is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        combined_save_path = os.path.join(SAVE_DIR, "clearance_alignment_combined.png")

    plot_clearance_and_alignment_together(
        df1=df1,
        df2=df2,
        x_col=x_col,
        label1=label1,
        label2=label2,
        save_path=combined_save_path,
    )
    phase_save_path = None
    if SAVE_DIR is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        phase_save_path = os.path.join(SAVE_DIR, "alignment_vs_clearance.png")

    plot_alignment_vs_clearance(
        df1=df1,
        df2=df2,
        label1=label1,
        label2=label2,
        save_path=phase_save_path,
    )
def plot_alignment_vs_clearance(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str,
    label2: str,
    save_path: str | None = None,
    ) -> None:
    plt.figure(figsize=(7, 6))

    plt.plot(
        df1["clearance"],
        df1["vessel_tip alignment"],
        label=label1,
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.9,
    )
    plt.plot(
        df2["clearance"],
        df2["vessel_tip alignment"],
        label=label2,
        linewidth=1.8,
        marker="o",
        markersize=3,
        alpha=0.9,
    )

    plt.xlabel("Clearance")
    plt.ylabel("Vessel Tip Alignment")
    plt.title("Alignment vs Clearance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()
def plot_clearance_and_alignment_together(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_col: str,
    label1: str,
    label2: str,
    save_path: str | None = None,
    ) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left y-axis: clearance
    l1, = ax1.plot(df1[x_col], df1["clearance"], label=f"{label1} clearance", linewidth=1.5)
    l2, = ax1.plot(df2[x_col], df2["clearance"], label=f"{label2} clearance", linewidth=1.5)
    ax1.set_xlabel(x_col)
    ax1.set_ylabel("Clearance")
    ax1.grid(True, alpha=0.3)

    # Right y-axis: alignment / angle
    ax2 = ax1.twinx()
    l3, = ax2.plot(df1[x_col], df1["vessel_tip alignment"], label=f"{label1} alignment", linewidth=1.5, linestyle="--")
    l4, = ax2.plot(df2[x_col], df2["vessel_tip alignment"], label=f"{label2} alignment", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Vessel Tip Alignment")

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Clearance and Vessel Tip Alignment")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
