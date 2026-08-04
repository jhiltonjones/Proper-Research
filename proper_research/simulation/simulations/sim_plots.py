from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from proper_research.simulation_controller.diagnostic_channel_visibility import (
    plot_command_visibility_split,
    plot_actual_jacobian_authority_split,
    plot_controller_jacobian_authority_split,
    plot_selected_channel_contributions_split,
)

DEFAULT_CHANNEL_NAMES = (
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "dL",
)


def _safe_array(
    info: dict,
    *keys: str,
    ndim: int | None = None,
) -> np.ndarray | None:
    """
    Return the first valid non-empty array found in `keys`.
    """
    for key in keys:
        value = info.get(key, None)

        if value is None:
            continue

        try:
            array = np.asarray(value, float)
        except (TypeError, ValueError):
            continue

        if array.size == 0:
            continue

        if ndim is not None and array.ndim != ndim:
            continue

        return array.copy()

    return None


def _plot_title(
    *,
    k: int,
    info: dict,
    description: str,
) -> str:
    controller_type = str(
        info.get("controller_type", "mpc")
    )

    if controller_type == "inverse_jacobian":
        mode = str(
            info.get(
                "inverse_sequence_mode",
                "inverse_jacobian",
            )
        )
    else:
        mode = str(
            info.get(
                "solver_mode_used",
                info.get("solver_mode", "mpc"),
            )
        )

    Np = int(
        info.get(
            "Np_used",
            np.asarray(
                info.get("U_seq", [])
            ).shape[0]
            if np.asarray(
                info.get("U_seq", [])
            ).ndim == 2
            else -1,
        )
    )

    Nr = int(
        info.get(
            "rollout_steps_used",
            info.get("rollout_steps", -1),
        )
    )

    return (
        f"{description}\n"
        f"k={k}, controller={controller_type}, "
        f"mode={mode}, Np={Np}, Nr={Nr}"
    )


def _save_figure(
    *,
    fig,
    output_root: Path,
    category: str,
    k: int,
    dpi: int,
) -> Path:
    category_dir = output_root / category
    category_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        category_dir
        / f"{category}_{k:05d}.png"
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def plot_rollout_trajectory_xy(
    *,
    k: int,
    info: dict,
    output_root: Path,
    dpi: int = 160,
) -> Path | None:
    """
    Compare:
        - reference horizon,
        - controller linear prediction,
        - nonlinear planned rollout,
        - actually applied nonlinear rollout.
    """
    X_ref = _safe_array(
        info,
        "X_ref_plan",
        ndim=2,
    )

    X_linear = _safe_array(
        info,
        "X_pred",
        "X_lin_candidate",
        ndim=2,
    )

    X_nonlinear = _safe_array(
        info,
        "X_nl_plan",
        "X_nl_candidate",
        ndim=2,
    )

    X_applied = _safe_array(
        info,
        "x_rollout",
        ndim=2,
    )

    x_start = _safe_array(
        info,
        "x_start",
    )

    if (
        X_ref is None
        and X_linear is None
        and X_nonlinear is None
        and X_applied is None
    ):
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    if X_ref is not None and X_ref.shape[1] >= 2:
        ax.plot(
            1e3 * X_ref[:, 0],
            1e3 * X_ref[:, 1],
            marker="o",
            linestyle="--",
            label="Reference horizon",
        )

    def add_trajectory(
        X: np.ndarray | None,
        label: str,
    ) -> None:
        if X is None or X.shape[1] < 2:
            return

        X_plot = X

        if x_start is not None and x_start.size >= 2:
            start_row = np.asarray(
                x_start[: X.shape[1]],
                float,
            ).reshape(1, -1)

            X_plot = np.vstack(
                [start_row, X]
            )

        ax.plot(
            1e3 * X_plot[:, 0],
            1e3 * X_plot[:, 1],
            marker="o",
            label=label,
        )

    add_trajectory(
        X_linear,
        "Controller linear prediction",
    )

    add_trajectory(
        X_nonlinear,
        "Nonlinear planned rollout",
    )

    add_trajectory(
        X_applied,
        "Applied nonlinear rollout",
    )

    if x_start is not None and x_start.size >= 2:
        ax.scatter(
            [1e3 * x_start[0]],
            [1e3 * x_start[1]],
            marker="x",
            s=80,
            label="Start of feedback frame",
        )

    ax.set_xlabel("Tip x [mm]")
    ax.set_ylabel("Tip y [mm]")
    ax.set_title(
        _plot_title(
            k=k,
            info=info,
            description=(
                "Reference and predicted tip trajectories"
            ),
        )
    )
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return _save_figure(
        fig=fig,
        output_root=output_root,
        category="rollout_trajectory_xy",
        k=k,
        dpi=dpi,
    )


def plot_prediction_errors(
    *,
    k: int,
    info: dict,
    output_root: Path,
    dpi: int = 160,
) -> Path | None:
    """
    Plot stagewise differences between:
        linear prediction,
        nonlinear plan,
        applied plant rollout.
    """
    X_linear = _safe_array(
        info,
        "X_pred",
        "X_lin_candidate",
        ndim=2,
    )

    X_nonlinear = _safe_array(
        info,
        "X_nl_plan",
        "X_nl_candidate",
        ndim=2,
    )

    X_applied = _safe_array(
        info,
        "x_rollout",
        ndim=2,
    )

    if X_linear is None:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plotted = False

    if X_nonlinear is not None:
        K = min(
            X_linear.shape[0],
            X_nonlinear.shape[0],
        )

        error = np.linalg.norm(
            X_nonlinear[:K, :3]
            - X_linear[:K, :3],
            axis=1,
        )

        ax.plot(
            np.arange(1, K + 1),
            1e3 * error,
            marker="o",
            label="Nonlinear plan − linear prediction",
        )

        plotted = True

    if X_applied is not None:
        K = min(
            X_linear.shape[0],
            X_applied.shape[0],
        )

        error = np.linalg.norm(
            X_applied[:K, :3]
            - X_linear[:K, :3],
            axis=1,
        )

        ax.plot(
            np.arange(1, K + 1),
            1e3 * error,
            marker="o",
            label="Applied rollout − linear prediction",
        )

        plotted = True

    if (
        X_applied is not None
        and X_nonlinear is not None
    ):
        K = min(
            X_applied.shape[0],
            X_nonlinear.shape[0],
        )

        error = np.linalg.norm(
            X_applied[:K, :3]
            - X_nonlinear[:K, :3],
            axis=1,
        )

        ax.plot(
            np.arange(1, K + 1),
            1e3 * error,
            marker="o",
            label="Applied rollout − nonlinear plan",
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Prediction stage")
    ax.set_ylabel("Position difference [mm]")
    ax.set_title(
        _plot_title(
            k=k,
            info=info,
            description=(
                "Linear, nonlinear and executed prediction errors"
            ),
        )
    )
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    return _save_figure(
        fig=fig,
        output_root=output_root,
        category="prediction_errors",
        k=k,
        dpi=dpi,
    )


def _plot_channel_heatmap(
    *,
    k: int,
    info: dict,
    values: np.ndarray,
    output_root: Path,
    category: str,
    description: str,
    colorbar_label: str,
    channel_names: tuple[str, ...],
    value_scale: float = 1.0,
    dpi: int = 160,
) -> Path | None:
    values = np.asarray(values, float)

    if values.ndim != 2 or values.size == 0:
        return None

    Np, number_of_channels = values.shape

    labels = list(
        channel_names[:number_of_channels]
    )

    if len(labels) < number_of_channels:
        labels.extend(
            [
                f"u{j}"
                for j in range(
                    len(labels),
                    number_of_channels,
                )
            ]
        )

    fig, ax = plt.subplots(
        figsize=(8.0, 5.0)
    )

    image = ax.imshow(
        (value_scale * values).T,
        origin="lower",
        aspect="auto",
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )
    colorbar.set_label(colorbar_label)

    ax.set_yticks(
        np.arange(number_of_channels)
    )
    ax.set_yticklabels(labels)

    ax.set_xticks(np.arange(Np))
    ax.set_xticklabels(
        np.arange(1, Np + 1)
    )

    ax.set_xlabel("Prediction stage")
    ax.set_ylabel("Control channel")
    ax.set_title(
        _plot_title(
            k=k,
            info=info,
            description=description,
        )
    )

    fig.tight_layout()

    return _save_figure(
        fig=fig,
        output_root=output_root,
        category=category,
        k=k,
        dpi=dpi,
    )


def plot_actual_jacobian_authority(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...],
    dpi: int = 160,
) -> Path | None:
    """
    Physical channel authority evaluated along the selected
    actuator trajectory.
    """
    authority = _safe_array(
        info,
        "jac_solution_scaled_authority",
        ndim=2,
    )

    if authority is None:
        return None

    return _plot_channel_heatmap(
        k=k,
        info=info,
        values=authority,
        output_root=output_root,
        category="jacobian_authority_actual",
        description=(
            "Actual Jacobian authority along selected trajectory"
        ),
        colorbar_label=(
            "Tip displacement at characteristic input [mm]"
        ),
        channel_names=channel_names,
        value_scale=1e3,
        dpi=dpi,
    )


def plot_controller_jacobian_authority(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...],
    dpi: int = 160,
) -> Path | None:
    """
    Authority represented by the Jacobians used to construct
    the controller optimisation or inverse policy.
    """
    authority = _safe_array(
        info,
        "jac_linearisation_scaled_authority",
        ndim=2,
    )

    if authority is None:
        return None

    return _plot_channel_heatmap(
        k=k,
        info=info,
        values=authority,
        output_root=output_root,
        category="jacobian_authority_controller",
        description=(
            "Jacobian authority assumed by controller"
        ),
        colorbar_label=(
            "Tip displacement at characteristic input [mm]"
        ),
        channel_names=channel_names,
        value_scale=1e3,
        dpi=dpi,
    )


def plot_selected_channel_contributions(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...],
    dpi: int = 160,
) -> Path | None:
    """
    Plot ||B[:,j] u_j|| for every horizon stage and input.
    """
    contribution = _safe_array(
        info,
        "jac_solution_contribution_norm",
        ndim=2,
    )

    if contribution is None:
        return None

    return _plot_channel_heatmap(
        k=k,
        info=info,
        values=contribution,
        output_root=output_root,
        category="selected_channel_contribution",
        description=(
            "Tip-motion contribution selected for each channel"
        ),
        colorbar_label=(
            "Selected tip displacement contribution [mm]"
        ),
        channel_names=channel_names,
        value_scale=1e3,
        dpi=dpi,
    )


def plot_jacobian_drift(
    *,
    k: int,
    info: dict,
    output_root: Path,
    dpi: int = 160,
) -> Path | None:
    """
    Compare the Jacobian used by the controller against the
    stagewise Jacobian along the selected trajectory.
    """
    drift = _safe_array(
        info,
        "B_drift_relative_stage",
    )

    if drift is None:
        return None

    drift = drift.reshape(-1)

    fig, ax = plt.subplots(
        figsize=(7.5, 4.5)
    )

    ax.plot(
        np.arange(1, drift.size + 1),
        drift,
        marker="o",
    )

    ax.set_xlabel("Prediction stage")
    ax.set_ylabel(
        r"$\|B_i^{actual}-B_i^{controller}\|_F"
        r"/\|B_i^{controller}\|_F$"
    )
    ax.set_title(
        _plot_title(
            k=k,
            info=info,
            description=(
                "Jacobian drift over the selected trajectory"
            ),
        )
    )
    ax.grid(True)
    fig.tight_layout()

    return _save_figure(
        fig=fig,
        output_root=output_root,
        category="jacobian_drift",
        k=k,
        dpi=dpi,
    )


def save_step_diagnostic_plots(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...] = (
        DEFAULT_CHANNEL_NAMES
    ),
    dpi: int = 160,
) -> list[Path]:
    """
    Generate every available controller diagnostic plot for
    one feedback frame.

    Missing diagnostics are skipped rather than causing the
    complete experiment to fail.
    """
    output_root = Path(output_root)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_functions = (
        lambda: plot_rollout_trajectory_xy(
            k=k,
            info=info,
            output_root=output_root,
            dpi=dpi,
        ),

        lambda: plot_prediction_errors(
            k=k,
            info=info,
            output_root=output_root,
            dpi=dpi,
        ),

        # --------------------------------------------------------
        # New command plot:
        # six source-pose channels together, dL separately.
        # --------------------------------------------------------
        lambda: plot_command_visibility_split(
            k=k,
            info=info,
            output_root=output_root,
            channel_names=channel_names,
            dpi=dpi,
        ),

        # --------------------------------------------------------
        # New actual Jacobian authority plot.
        # --------------------------------------------------------
        lambda: plot_actual_jacobian_authority_split(
            k=k,
            info=info,
            output_root=output_root,
            channel_names=channel_names,
            dpi=dpi,
        ),

        # --------------------------------------------------------
        # New controller-assumed Jacobian authority plot.
        # --------------------------------------------------------
        lambda: plot_controller_jacobian_authority_split(
            k=k,
            info=info,
            output_root=output_root,
            channel_names=channel_names,
            dpi=dpi,
        ),

        # --------------------------------------------------------
        # New B[:,j] u_j contribution plot.
        # --------------------------------------------------------
        lambda: plot_selected_channel_contributions_split(
            k=k,
            info=info,
            output_root=output_root,
            channel_names=channel_names,
            dpi=dpi,
        ),

        lambda: plot_jacobian_drift(
            k=k,
            info=info,
            output_root=output_root,
            dpi=dpi,
        ),
    )

    saved_paths: list[Path] = []

    for plot_function in plot_functions:
        try:
            path = plot_function()

            if path is not None:
                saved_paths.append(path)

        except Exception as exc:
            print(
                "[WARN] Controller diagnostic plot "
                f"failed at k={k}: {exc}"
            )

    return saved_paths