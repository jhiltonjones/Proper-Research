import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel, load_polygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from proper_research.vision.bounds_beam import (
    reconstruct_beam_within_vessel,
    load_polygon,
)

from proper_research.experiments.charactisations import (
    transform_local_points_to_robot,
    robot_points_to_base_local,
)
PIVOT_POSE6 = np.array([
    0.8281328220229531, -0.6812731669220016, -0.1, np.pi, 0.001, 0.001
], dtype=float)

L0_M = 0.035

BEAM_BASE_POINT_ROBOT_M = np.array([
    PIVOT_POSE6[0] - (L0_M + 0.14),
    PIVOT_POSE6[1],
    -0.1,
], dtype=float)
# =========================
# Configuration
# =========================
ONE_STEP_FILE = "/home/jack/Proper-Research/control_run_log_test_opti_mid_low_18_mag_opti.csv"
THREE_STEP_FILE = "/home/jack/Proper-Research/control_run_log_test_opti_3step.csv"

SAVE_FIGURES = False
FIGURE_DIR = Path("mpc_comparison_figures")
FIG_DPI = 150
def lumen_result_to_base_local(result, pivot_pose6, beam_base_point_robot_m):
    """
    Convert reconstructed lumen from image-local frame into robot frame,
    then into base-local frame.
    """
    lumen_C_robot_m = transform_local_points_to_robot(
        result["lumen_C_m"],
        pivot_pose6,
        flip_y=False,
    )

    lumen_R_m = np.asarray(result["lumen_R_m"], dtype=float)

    lumen_C_base_local_m = robot_points_to_base_local(
        lumen_C_robot_m,
        pivot_pose6=pivot_pose6,
        beam_base_point_robot_m=beam_base_point_robot_m,
    )

    return lumen_C_robot_m, lumen_C_base_local_m, lumen_R_m
# =========================
# Helpers
# =========================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df
def print_requested_averages(df: pd.DataFrame, label: str) -> None:
    avg_wall_clearance = df["closest_distance_mm"].mean()
    avg_tracking_error_nominal = df["err_nom_xy"].mean()
    avg_right_wall_angle = df["beam_right_wall_angle_deg"].mean()

    print(f"\n=== Requested averages: {label} ===")
    print(f"Average wall clearance (closest_distance_mm): {avg_wall_clearance:.6f} mm")
    print(f"Average nominal tracking error (err_nom_xy): {avg_tracking_error_nominal:.6f}")
    print(f"Average angle to right wall (beam_right_wall_angle_deg): {avg_right_wall_angle:.6f} deg")

def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tracking error in x/y/z between measured and nominal
    out["track_err_x"] = out["meas_x"] - out["nom_x"]
    out["track_err_y"] = out["meas_y"] - out["nom_y"]
    out["track_err_z"] = out["meas_z"] - out["nom_z"]

    out["track_err_xy_calc"] = np.sqrt(out["track_err_x"]**2 + out["track_err_y"]**2)
    out["track_err_xyz_calc"] = np.sqrt(
        out["track_err_x"]**2 + out["track_err_y"]**2 + out["track_err_z"]**2
    )

    # Prediction error: predicted vs measured
    out["pred_err_x"] = out["meas_x"] - out["pred_x"]
    out["pred_err_y"] = out["meas_y"] - out["pred_y"]
    out["pred_err_z"] = out["meas_z"] - out["pred_z"]

    out["pred_err_xy"] = np.sqrt(out["pred_err_x"]**2 + out["pred_err_y"]**2)
    out["pred_err_xyz"] = np.sqrt(
        out["pred_err_x"]**2 + out["pred_err_y"]**2 + out["pred_err_z"]**2
    )

    # Magnet command magnitude in XY and XYZ
    out["mag_xy"] = np.sqrt(out["mag_x"]**2 + out["mag_y"]**2)
    out["mag_xyz"] = np.sqrt(out["mag_x"]**2 + out["mag_y"]**2 + out["mag_z"]**2)

    # Beam speed estimate in XY between samples
    dx = out["meas_x"].diff()
    dy = out["meas_y"].diff()
    out["beam_step_xy"] = np.sqrt(dx**2 + dy**2)

    # Wall clearance asymmetry: positive means closer to right wall? depends on convention
    out["clearance_balance_mm"] = out["dist_left_edge_mm"] - out["dist_right_edge_mm"]

    return out


def summarize(df: pd.DataFrame, label: str) -> pd.Series:
    summary = {
        "controller": label,
        "n_steps": len(df),
        "mean_err_xy_reported": df["err_xy"].mean(),
        "max_err_xy_reported": df["err_xy"].max(),
        "rmse_err_xy_reported": np.sqrt(np.mean(df["err_xy"]**2)),
        "mean_err_nom_xy": df["err_nom_xy"].mean(),
        "max_err_nom_xy": df["err_nom_xy"].max(),
        "rmse_err_nom_xy": np.sqrt(np.mean(df["err_nom_xy"]**2)),
        "mean_track_err_xy_calc": df["track_err_xy_calc"].mean(),
        "max_track_err_xy_calc": df["track_err_xy_calc"].max(),
        "rmse_track_err_xy_calc": np.sqrt(np.mean(df["track_err_xy_calc"]**2)),
        "mean_pred_err_xy": df["pred_err_xy"].mean(),
        "max_pred_err_xy": df["pred_err_xy"].max(),
        "rmse_pred_err_xy": np.sqrt(np.mean(df["pred_err_xy"]**2)),
        "min_closest_distance_mm": df["closest_distance_mm"].min(),
        "mean_closest_distance_mm": df["closest_distance_mm"].mean(),
        "time_below_3mm_clearance": (df["closest_distance_mm"] < 3.0).sum(),
        "time_below_2mm_clearance": (df["closest_distance_mm"] < 2.0).sum(),
        "mean_i_ref": df["i_ref"].mean(),
        "max_i_ref": df["i_ref"].max(),
        "mean_mag_xy": df["mag_xy"].mean(),
        "max_mag_xy": df["mag_xy"].max(),
        "mean_beam_step_xy": df["beam_step_xy"].mean(skipna=True),
        "max_beam_step_xy": df["beam_step_xy"].max(skipna=True),
        "mean_left_wall_angle_deg": df["beam_left_wall_angle_deg"].mean(),
        "mean_right_wall_angle_deg": df["beam_right_wall_angle_deg"].mean(),
    }
    return pd.Series(summary)


def print_summary_table(one_df: pd.DataFrame, three_df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            summarize(one_df, "low field MPC"),
            summarize(three_df, "3-step MPC"),
        ]
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n=== Summary comparison ===")
    print(summary.round(6).to_string(index=False))


def save_fig(name: str):
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(exist_ok=True, parents=True)
        plt.savefig(FIGURE_DIR / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")


# =========================
# Plotting
# =========================
def plot_xy_trajectory(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(8, 6))
    plt.plot(one_df["nom_x"], one_df["nom_y"], label="Nominal path", linewidth=2)
    plt.plot(one_df["meas_x"], one_df["meas_y"], "o-", markersize=3, label="Measured low field")
    plt.plot(three_df["meas_x"], three_df["meas_y"], "o-", markersize=3, label="Measured 3-step")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Beam trajectory in XY plane")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("xy_trajectory")


def plot_tracking_error(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["err_nom_xy"], label="low field reported nominal XY error")
    plt.plot(three_df["step"], three_df["err_nom_xy"], label="3-step reported nominal XY error")
    plt.plot(one_df["step"], one_df["track_err_xy_calc"], "--", label="low field calc measured-nominal XY")
    plt.plot(three_df["step"], three_df["track_err_xy_calc"], "--", label="3-step calc measured-nominal XY")
    plt.xlabel("Step")
    plt.ylabel("Error")
    plt.title("Tracking error vs nominal path")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("tracking_error")


def plot_prediction_error(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["pred_err_xy"], label="low field prediction error XY")
    plt.plot(three_df["step"], three_df["pred_err_xy"], label="3-step prediction error XY")
    plt.xlabel("Step")
    plt.ylabel("Prediction error")
    plt.title("Prediction accuracy: measured vs predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("prediction_error")


def plot_clearance(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["closest_distance_mm"], label="low field closest distance")
    plt.plot(three_df["step"], three_df["closest_distance_mm"], label="3-step closest distance")
    plt.axhline(3.0, linestyle="--", label="3 mm threshold")
    plt.axhline(2.0, linestyle="--", label="2 mm threshold")
    plt.xlabel("Step")
    plt.ylabel("Closest wall distance [mm]")
    plt.title("Safety margin to vessel wall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("clearance")


def plot_wall_balance(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["clearance_balance_mm"], label="low field left-right clearance balance")
    plt.plot(three_df["step"], three_df["clearance_balance_mm"], label="3-step left-right clearance balance")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("Left edge dist - Right edge dist [mm]")
    plt.title("Wall clearance balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("wall_balance")


def plot_control_effort(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["i_ref"], label="low field i_ref")
    plt.plot(three_df["step"], three_df["i_ref"], label="3-step i_ref")
    plt.xlabel("Step")
    plt.ylabel("i_ref")
    plt.title("Control effort / reference current")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("control_effort")


def plot_magnet_commands(one_df: pd.DataFrame, three_df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(one_df["step"], one_df["mag_x"], label="low field")
    axes[0].plot(three_df["step"], three_df["mag_x"], label="3-step")
    axes[0].set_ylabel("mag_x")
    axes[0].set_title("Magnet commands")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(one_df["step"], one_df["mag_y"], label="low field")
    axes[1].plot(three_df["step"], three_df["mag_y"], label="3-step")
    axes[1].set_ylabel("mag_y")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(one_df["step"], one_df["mag_z"], label="low field")
    axes[2].plot(three_df["step"], three_df["mag_z"], label="3-step")
    axes[2].set_ylabel("mag_z")
    axes[2].set_xlabel("Step")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("magnet_commands")


def plot_heading_angles(one_df: pd.DataFrame, three_df: pd.DataFrame):
    plt.figure(figsize=(10, 5))
    plt.plot(one_df["step"], one_df["beam_left_wall_angle_deg"], label="low field left-wall angle")
    plt.plot(one_df["step"], one_df["beam_right_wall_angle_deg"], label="low field right-wall angle")
    plt.plot(three_df["step"], three_df["beam_left_wall_angle_deg"], "--", label="3-step left-wall angle")
    plt.plot(three_df["step"], three_df["beam_right_wall_angle_deg"], "--", label="3-step right-wall angle")
    plt.xlabel("Step")
    plt.ylabel("Angle [deg]")
    plt.title("Beam angle relative to vessel walls")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("heading_angles")


def plot_histograms(one_df: pd.DataFrame, three_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(one_df["err_nom_xy"], bins=12, alpha=0.7, label="low field")
    axes[0].hist(three_df["err_nom_xy"], bins=12, alpha=0.7, label="3-step")
    axes[0].set_title("Distribution of nominal XY error")
    axes[0].set_xlabel("err_nom_xy")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(one_df["closest_distance_mm"], bins=12, alpha=0.7, label="low field")
    axes[1].hist(three_df["closest_distance_mm"], bins=12, alpha=0.7, label="3-step")
    axes[1].set_title("Distribution of closest wall distance")
    axes[1].set_xlabel("closest_distance_mm")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("histograms")


# =========================
# Interpretation
# =========================
def print_interpretation(one_df: pd.DataFrame, three_df: pd.DataFrame) -> None:
    one_rmse = np.sqrt(np.mean(one_df["err_nom_xy"] ** 2))
    three_rmse = np.sqrt(np.mean(three_df["err_nom_xy"] ** 2))

    one_clear_min = one_df["closest_distance_mm"].min()
    three_clear_min = three_df["closest_distance_mm"].min()

    one_below_3 = int((one_df["closest_distance_mm"] < 3.0).sum())
    three_below_3 = int((three_df["closest_distance_mm"] < 3.0).sum())

    one_i = one_df["i_ref"].mean()
    three_i = three_df["i_ref"].mean()

    print("\n=== Interpretation ===")

    if three_rmse < one_rmse:
        pct = 100.0 * (one_rmse - three_rmse) / one_rmse
        print(f"- 3-step MPC has lower nominal XY tracking RMSE by about {pct:.1f}%.")
    else:
        pct = 100.0 * (three_rmse - one_rmse) / three_rmse
        print(f"- low field MPC has lower nominal XY tracking RMSE by about {pct:.1f}%.")

    if three_clear_min > one_clear_min:
        print("- 3-step MPC maintains a larger worst-case wall clearance.")
    else:
        print("- low field MPC maintains a larger worst-case wall clearance.")

    if three_below_3 < one_below_3:
        print("- 3-step MPC spends fewer steps below the 3 mm safety margin.")
    elif three_below_3 > one_below_3:
        print("- low field MPC spends fewer steps below the 3 mm safety margin.")
    else:
        print("- Both controllers spend the same number of steps below the 3 mm safety margin.")

    if three_i < one_i:
        print("- 3-step MPC uses lower average control effort (mean i_ref).")
    else:
        print("- low field MPC uses lower average control effort (mean i_ref).")

    print("- Check the XY trajectory plot first for path-following shape.")
    print("- Check the tracking error and prediction error plots for model/controller quality.")
    print("- Check the clearance plot for safety-critical differences near the vessel wall.")
    print("- Check control effort and magnet command plots for smoothness vs aggressiveness.")
def plot_csv_trajectory_with_lumen_base_local(
    csv_base_local_m,
    lumen_C_base_local_m,
    lumen_R_m=None,
    title="CSV trajectory over lumen in base-local frame",
    step_labels=True,
):
    import numpy as np
    import matplotlib.pyplot as plt

    traj = np.asarray(csv_base_local_m, dtype=float)
    lumen = np.asarray(lumen_C_base_local_m, dtype=float)

    plt.figure(figsize=(8, 8))

    # lumen centerline
    plt.plot(
        1e3 * lumen[:, 0],
        1e3 * lumen[:, 1],
        "-",
        linewidth=2.0,
        label="Lumen centerline",
    )

    # lumen walls
    if lumen_R_m is not None and len(lumen_R_m) == len(lumen):
        lumen_R_m = np.asarray(lumen_R_m, dtype=float).reshape(-1)

        tangents = np.zeros_like(lumen)
        tangents[1:-1] = lumen[2:] - lumen[:-2]
        tangents[0] = lumen[1] - lumen[0]
        tangents[-1] = lumen[-1] - lumen[-2]

        normals = np.zeros_like(tangents)
        for i, t in enumerate(tangents):
            tx, ty = t[0], t[1]
            n = np.array([-ty, tx, 0.0], dtype=float)
            nn = np.linalg.norm(n[:2])
            if nn > 1e-12:
                normals[i] = n / nn

        upper = lumen + normals * lumen_R_m[:, None]
        lower = lumen - normals * lumen_R_m[:, None]

        plt.plot(1e3 * upper[:, 0], 1e3 * upper[:, 1], "--", alpha=0.6, label="Lumen wall")
        plt.plot(1e3 * lower[:, 0], 1e3 * lower[:, 1], "--", alpha=0.6)

    # csv trajectory
    plt.plot(
        1e3 * traj[:, 0],
        1e3 * traj[:, 1],
        "ro-",
        markersize=4,
        label="CSV measured trajectory",
    )

    if step_labels:
        for i, p in enumerate(traj):
            plt.text(1e3 * p[0], 1e3 * p[1], str(i), fontsize=8)

    plt.xlabel("Base-local x [mm]")
    plt.ylabel("Base-local y [mm]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
def csv_robot_xy_to_base_local(csv_path, pivot_pose6, beam_base_point_robot_m,

                               x_col="meas_x", y_col="meas_y", z_value=-0.1):

    import pandas as pd

    import numpy as np

    df = pd.read_csv(csv_path)

    P_robot = np.column_stack([

        df[x_col].to_numpy(dtype=float),

        df[y_col].to_numpy(dtype=float),

        np.full(len(df), float(z_value)),

    ])

    P_base_local = robot_points_to_base_local(

        P_robot,

        pivot_pose6=pivot_pose6,

        beam_base_point_robot_m=beam_base_point_robot_m,

    )

    return df, P_base_local
# =========================
# Main
# =========================
# def main():
#     one_df = load_csv(ONE_STEP_FILE)
#     three_df = load_csv(THREE_STEP_FILE)

#     one_df = add_derived_metrics(one_df)
#     three_df = add_derived_metrics(three_df)

#     print_summary_table(one_df, three_df)

#     print_requested_averages(one_df, "1-step MPC")
#     print_requested_averages(three_df, "3-step MPC")

#     print_interpretation(one_df, three_df)

#     # plot_xy_trajectory(one_df, three_df)
#     # plot_tracking_error(one_df, three_df)
#     # plot_prediction_error(one_df, three_df)
#     # plot_clearance(one_df, three_df)
#     # plot_wall_balance(one_df, three_df)
#     # plot_control_effort(one_df, three_df)
#     # plot_magnet_commands(one_df, three_df)
#     # plot_heading_angles(one_df, three_df)
#     # plot_histograms(one_df, three_df)
#     pivot_hint = (300, 391)
#     roi_polygon = load_polygon("/home/jack/Proper-Research/custom_area.json")
def plot_two_csv_paths_with_lumen_base_local(
    lumen_C_base_local_m,
    lumen_R_m,
    one_meas_base_local_m,
    three_meas_base_local_m,
    one_nom_base_local_m=None,
    three_nom_base_local_m=None,
    title="low field vs 3-step trajectories over reconstructed lumen",
    step_labels=False,
):
    lumen = np.asarray(lumen_C_base_local_m, dtype=float)
    one_meas = np.asarray(one_meas_base_local_m, dtype=float)
    three_meas = np.asarray(three_meas_base_local_m, dtype=float)

    plt.figure(figsize=(8, 8))

    # lumen centerline
    plt.plot(
        1e3 * lumen[:, 0],
        1e3 * lumen[:, 1],
        "k-",
        linewidth=2,
        label="Lumen centerline",
    )

    # lumen walls
    if lumen_R_m is not None and len(lumen_R_m) == len(lumen):
        lumen_R_m = np.asarray(lumen_R_m, dtype=float).reshape(-1)

        tangents = np.zeros_like(lumen)
        tangents[1:-1] = lumen[2:] - lumen[:-2]
        tangents[0] = lumen[1] - lumen[0]
        tangents[-1] = lumen[-1] - lumen[-2]

        normals = np.zeros_like(tangents)
        for i, t in enumerate(tangents):
            tx, ty = t[0], t[1]
            n = np.array([-ty, tx, 0.0], dtype=float)
            nn = np.linalg.norm(n[:2])
            if nn > 1e-12:
                normals[i] = n / nn

        upper = lumen + normals * lumen_R_m[:, None]
        lower = lumen - normals * lumen_R_m[:, None]

        plt.plot(1e3 * upper[:, 0], 1e3 * upper[:, 1], "k--", alpha=0.5, label="Lumen wall")
        plt.plot(1e3 * lower[:, 0], 1e3 * lower[:, 1], "k--", alpha=0.5)

    # measured paths
    plt.plot(
        1e3 * one_meas[:, 0],
        1e3 * one_meas[:, 1],
        "ro-",
        markersize=4,
        linewidth=1.8,
        label="low field measured",
    )

    plt.plot(
        1e3 * three_meas[:, 0],
        1e3 * three_meas[:, 1],
        "bo-",
        markersize=4,
        linewidth=1.8,
        label="3-step measured",
    )

    # optional nominal paths
    # if one_nom_base_local_m is not None:
    #     one_nom = np.asarray(one_nom_base_local_m, dtype=float)
    #     plt.plot(
    #         1e3 * one_nom[:, 0],
    #         1e3 * one_nom[:, 1],
    #         "r--",
    #         linewidth=1.5,
    #         alpha=0.9,
    #         label="1-step nominal",
    #     )

    # if three_nom_base_local_m is not None:
    #     three_nom = np.asarray(three_nom_base_local_m, dtype=float)
    #     plt.plot(
    #         1e3 * three_nom[:, 0],
    #         1e3 * three_nom[:, 1],
    #         "b--",
    #         linewidth=1.5,
    #         alpha=0.9,
    #         label="3-step nominal",
    #     )

    if step_labels:
        for i, p in enumerate(one_meas):
            plt.text(1e3 * p[0], 1e3 * p[1], f"1:{i}", fontsize=7, color="red")
        for i, p in enumerate(three_meas):
            plt.text(1e3 * p[0], 1e3 * p[1], f"3:{i}", fontsize=7, color="blue")

    plt.xlabel("Base-local x [mm]")
    plt.ylabel("Base-local y [mm]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
def main():
    one_df = load_csv(ONE_STEP_FILE)
    three_df = load_csv(THREE_STEP_FILE)

    one_df = add_derived_metrics(one_df)
    three_df = add_derived_metrics(three_df)

    print_summary_table(one_df, three_df)

    print_requested_averages(one_df, "low field MPC")
    print_requested_averages(three_df, "3-step MPC")

    print_interpretation(one_df, three_df)

    pivot_hint = (300, 391)
    roi_polygon = load_polygon("/home/jack/Proper-Research/custom_area.json")

    result = reconstruct_beam_within_vessel(
        image_filename="focused_image.jpg",
        red_roi_polygon=roi_polygon,
        blue_roi_path="blue_roi_box.json",
        pivot_hint=pivot_hint,
        show=True,
        save_overlay_path="debug_outputs/reconstruction_overlay.png",
    )

    # Convert lumen into base-local frame
    lumen_C_robot_m, lumen_C_base_local_m, lumen_R_m = lumen_result_to_base_local(
        result,
        pivot_pose6=PIVOT_POSE6,
        beam_base_point_robot_m=BEAM_BASE_POINT_ROBOT_M,
    )

    # 1-step measured
    _, one_meas_base_local_m = csv_robot_xy_to_base_local(
        csv_path=ONE_STEP_FILE,
        pivot_pose6=PIVOT_POSE6,
        beam_base_point_robot_m=BEAM_BASE_POINT_ROBOT_M,
        x_col="meas_x",
        y_col="meas_y",
        z_value=-0.1,
    )

    # 3-step measured
    _, three_meas_base_local_m = csv_robot_xy_to_base_local(
        csv_path=THREE_STEP_FILE,
        pivot_pose6=PIVOT_POSE6,
        beam_base_point_robot_m=BEAM_BASE_POINT_ROBOT_M,
        x_col="meas_x",
        y_col="meas_y",
        z_value=-0.1,
    )

    # Optional nominal paths
    _, one_nom_base_local_m = csv_robot_xy_to_base_local(
        csv_path=ONE_STEP_FILE,
        pivot_pose6=PIVOT_POSE6,
        beam_base_point_robot_m=BEAM_BASE_POINT_ROBOT_M,
        x_col="nom_x",
        y_col="nom_y",
        z_value=-0.1,
    )

    _, three_nom_base_local_m = csv_robot_xy_to_base_local(
        csv_path=THREE_STEP_FILE,
        pivot_pose6=PIVOT_POSE6,
        beam_base_point_robot_m=BEAM_BASE_POINT_ROBOT_M,
        x_col="nom_x",
        y_col="nom_y",
        z_value=-0.1,
    )

    # Plot both measured paths on the same lumen
    plot_two_csv_paths_with_lumen_base_local(
        lumen_C_base_local_m=lumen_C_base_local_m,
        lumen_R_m=lumen_R_m,
        one_meas_base_local_m=one_meas_base_local_m,
        three_meas_base_local_m=three_meas_base_local_m,
        one_nom_base_local_m=one_nom_base_local_m,
        three_nom_base_local_m=three_nom_base_local_m,
        title="low field vs 3-step trajectories over reconstructed lumen",
        step_labels=False,
    )


if __name__ == "__main__":
    main()