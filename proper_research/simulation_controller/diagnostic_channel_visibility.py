from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHANNEL_NAMES = (
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
    "dL",
)

SOURCE_CHANNEL_COUNT = 6

# Fixed, colour-blind-friendly colours for the six source-magnet DOFs.
# dL is deliberately black and is always plotted on a separate axis.
CHANNEL_COLOURS = {
    "vx": "#0072B2",
    "vy": "#E69F00",
    "vz": "#009E73",
    "wx": "#D55E00",
    "wy": "#CC79A7",
    "wz": "#56B4E9",
    "dL": "#222222",
}


def _as_float_array(
    info: dict,
    keys: Iterable[str],
    *,
    ndim: int | None = None,
) -> np.ndarray | None:
    for key in keys:
        value = info.get(key, None)
        if value is None:
            continue

        try:
            array = np.asarray(value, float)
        except Exception:
            continue

        if array.size == 0:
            continue

        if ndim is not None and array.ndim != ndim:
            continue

        return array.copy()

    return None


def _channel_scale(
    info: dict,
    number_of_channels: int,
) -> np.ndarray:
    scale = _as_float_array(
        info,
        (
            "jacobian_diag_input_scale",
            "input_scale",
        ),
    )

    if scale is None:
        return np.ones(number_of_channels, dtype=float)

    scale = np.asarray(scale, float).reshape(-1)

    if scale.size == 1:
        scale = np.full(
            number_of_channels,
            float(scale[0]),
        )

    if scale.size < number_of_channels:
        scale = np.pad(
            scale,
            (0, number_of_channels - scale.size),
            constant_values=1.0,
        )

    scale = scale[:number_of_channels]
    scale = np.where(
        np.isfinite(scale) & (np.abs(scale) > 1e-12),
        np.abs(scale),
        1.0,
    )

    return scale


def _stage_axis(number_of_stages: int) -> np.ndarray:
    return np.arange(1, number_of_stages + 1)


def _save(
    fig: plt.Figure,
    output_root: Path,
    folder: str,
    k: int,
    dpi: int,
) -> Path:
    output_dir = Path(output_root) / folder
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"frame_{int(k):05d}.png"
    )

    fig.savefig(
        output_path,
        dpi=int(dpi),
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


def _source_labels(
    channel_names: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        channel_names[:SOURCE_CHANNEL_COUNT]
    )


def _colour(
    channel_name: str,
    fallback_index: int,
) -> str:
    fallback = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
    )

    return CHANNEL_COLOURS.get(
        channel_name,
        fallback[
            fallback_index % len(fallback)
        ],
    )


def plot_command_visibility_split(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
    dpi: int = 160,
) -> Path | None:
    """
    Plot normalized source-magnet commands separately from insertion.

    Top:
        Six source-magnet channels:
            [vx, vy, vz, wx, wy, wz]

        Each command is divided by its characteristic input scale.

    Bottom:
        dL on a separate axis, so insertion cannot compress the
        source-channel display.
    """
    U = _as_float_array(
        info,
        ("U_seq",),
        ndim=2,
    )

    if U is None or U.shape[0] == 0:
        return None

    number_of_channels = min(
        U.shape[1],
        len(channel_names),
    )

    if number_of_channels < 1:
        return None

    U = U[:, :number_of_channels]
    scale = _channel_scale(
        info,
        number_of_channels,
    )

    normalized = U / scale.reshape(1, -1)
    stages = _stage_axis(U.shape[0])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 7.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.0, 1.15],
        },
    )

    ax_source, ax_dL = axes

    number_of_source_channels = min(
        SOURCE_CHANNEL_COUNT,
        number_of_channels,
    )

    for channel in range(number_of_source_channels):
        name = channel_names[channel]

        ax_source.plot(
            stages,
            normalized[:, channel],
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=name,
            color=_colour(name, channel),
        )

    ax_source.axhline(
        0.0,
        linewidth=0.8,
        color="0.35",
    )
    ax_source.axhline(
        1.0,
        linestyle="--",
        linewidth=0.9,
        color="0.55",
    )
    ax_source.axhline(
        -1.0,
        linestyle="--",
        linewidth=0.9,
        color="0.55",
    )

    ax_source.set_ylabel(
        "Source command / characteristic scale"
    )
    ax_source.set_title(
        f"Frame {k}: normalised source-magnet commands"
    )
    ax_source.grid(True, alpha=0.3)
    ax_source.legend(
        ncol=3,
        loc="best",
    )

    if number_of_channels > 6:
        dL_name = channel_names[6]

        ax_dL.plot(
            stages,
            normalized[:, 6],
            marker="s",
            linewidth=2.2,
            markersize=5.5,
            label=dL_name,
            color=CHANNEL_COLOURS["dL"],
        )

        ax_dL.axhline(
            0.0,
            linewidth=0.8,
            color="0.35",
        )
        ax_dL.axhline(
            1.0,
            linestyle="--",
            linewidth=0.9,
            color="0.55",
            label="+ characteristic limit",
        )
        ax_dL.axhline(
            -1.0,
            linestyle="--",
            linewidth=0.9,
            color="0.55",
        )

        ax_dL.legend(
            loc="best",
        )

    ax_dL.set_xlabel(
        "Prediction stage"
    )
    ax_dL.set_ylabel(
        "dL / scale"
    )
    ax_dL.grid(True, alpha=0.3)

    fig.tight_layout()

    return _save(
        fig,
        output_root,
        "command_visibility_split",
        k,
        dpi,
    )


def _authority_from_B(
    B_sequence: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Return scaled one-step authority:

        ||B[:, j] scale_j||

    Shape:
        (Np, number_of_channels)
    """
    B_sequence = np.asarray(
        B_sequence,
        float,
    )

    number_of_channels = min(
        B_sequence.shape[2],
        scale.size,
    )

    scaled_columns = (
        B_sequence[:, :, :number_of_channels]
        * scale[
            None,
            None,
            :number_of_channels,
        ]
    )

    return np.linalg.norm(
        scaled_columns,
        axis=1,
    )


def _contribution_from_B_and_U(
    B_sequence: np.ndarray,
    U_sequence: np.ndarray,
) -> np.ndarray:
    """
    Return selected channel contribution magnitudes:

        ||B[:, j] u_j||

    Shape:
        (Np, number_of_channels)
    """
    B_sequence = np.asarray(
        B_sequence,
        float,
    )
    U_sequence = np.asarray(
        U_sequence,
        float,
    )

    number_of_stages = min(
        B_sequence.shape[0],
        U_sequence.shape[0],
    )

    number_of_channels = min(
        B_sequence.shape[2],
        U_sequence.shape[1],
    )

    vectors = (
        B_sequence[
            :number_of_stages,
            :,
            :number_of_channels,
        ]
        * U_sequence[
            :number_of_stages,
            None,
            :number_of_channels,
        ]
    )

    return np.linalg.norm(
        vectors,
        axis=1,
    )


def _plot_magnitude_visibility(
    *,
    values: np.ndarray,
    k: int,
    output_root: Path,
    folder: str,
    title: str,
    absolute_ylabel: str,
    channel_names: tuple[str, ...],
    dpi: int,
    value_multiplier: float,
    show_normalized_panel: bool = False,
) -> Path | None:
    """
    Plot absolute channel magnitudes.

    Default:
        Panel 1:
            Absolute source-channel magnitudes.

        Panel 2:
            Absolute dL magnitude on a separate axis.

    Optional:
        A normalized source-channel panel can be enabled with
        show_normalized_panel=True. Each source channel is divided
        by its own maximum magnitude across the prediction horizon.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 2 or values.shape[0] == 0:
        return None

    number_of_channels = min(
        values.shape[1],
        len(channel_names),
    )

    if number_of_channels < 1:
        return None

    values = values[:, :number_of_channels]
    stages = _stage_axis(values.shape[0])

    source_count = min(
        SOURCE_CHANNEL_COUNT,
        number_of_channels,
    )

    # Absolute source-channel magnitudes.
    source_values = np.abs(
        value_multiplier
        * values[:, :source_count]
    )

    # ------------------------------------------------------------
    # Create either:
    #   2 panels: absolute source + absolute dL
    #   3 panels: absolute source + normalized source + absolute dL
    # ------------------------------------------------------------
    if show_normalized_panel:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(9.0, 9.5),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2.6, 2.2, 1.2],
            },
        )

        ax_absolute, ax_normalized, ax_dL = axes

    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9.0, 7.0),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2.6, 1.2],
            },
        )

        ax_absolute, ax_dL = axes
        ax_normalized = None

    # ------------------------------------------------------------
    # Source channels
    # ------------------------------------------------------------
    for channel in range(source_count):
        name = channel_names[channel]
        colour = _colour(name, channel)

        ax_absolute.plot(
            stages,
            source_values[:, channel],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=name,
            color=colour,
        )

        # Optional normalized visibility plot.
        if ax_normalized is not None:
            channel_max = float(
                np.nanmax(
                    source_values[:, channel]
                )
            )

            if (
                np.isfinite(channel_max)
                and channel_max > 1e-15
            ):
                normalized = (
                    source_values[:, channel]
                    / channel_max
                )
            else:
                normalized = np.zeros(
                    source_values.shape[0],
                    dtype=float,
                )

            ax_normalized.plot(
                stages,
                normalized,
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                label=(
                    f"{name} "
                    f"(maximum = {channel_max:.3g})"
                ),
                color=colour,
            )

    # ------------------------------------------------------------
    # Absolute source-channel plot
    # ------------------------------------------------------------
    ax_absolute.set_ylabel(
        absolute_ylabel
    )
    ax_absolute.set_title(
        f"Frame {k}: {title}"
    )
    ax_absolute.grid(
        True,
        alpha=0.3,
    )
    ax_absolute.legend(
        ncol=3,
        loc="best",
    )

    # ------------------------------------------------------------
    # Optional normalized source-channel plot
    # ------------------------------------------------------------
    if ax_normalized is not None:
        ax_normalized.set_ylim(
            -0.03,
            1.08,
        )
        ax_normalized.set_ylabel(
            "Normalized magnitude"
        )
        ax_normalized.set_title(
            "Normalized source-channel magnitudes "
            "(each channel divided by its maximum)"
        )
        ax_normalized.grid(
            True,
            alpha=0.3,
        )
        ax_normalized.legend(
            ncol=2,
            loc="best",
            fontsize=8,
        )

    # ------------------------------------------------------------
    # Absolute dL magnitude
    # ------------------------------------------------------------
    if number_of_channels > 6:
        dL_values = np.abs(
            value_multiplier
            * values[:, 6]
        )

        ax_dL.plot(
            stages,
            dL_values,
            marker="s",
            linewidth=2.2,
            markersize=5.0,
            label=channel_names[6],
            color=CHANNEL_COLOURS["dL"],
        )

        ax_dL.legend(
            loc="best",
        )

    ax_dL.set_xlabel(
        "Prediction stage"
    )
    ax_dL.set_ylabel(
        absolute_ylabel
    )
    ax_dL.set_title(
        "Absolute insertion-length magnitude"
    )
    ax_dL.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    return _save(
        fig,
        output_root,
        folder,
        k,
        dpi,
    )


def plot_actual_jacobian_authority_split(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
    dpi: int = 160,
) -> Path | None:
    """
    Plot stagewise Jacobian authority evaluated along the selected
    control sequence.

    Preferred stored key:
        jac_solution_scaled_authority

    Fallback:
        compute from B_sequence_solution.
    """
    values = _as_float_array(
        info,
        (
            "jac_solution_scaled_authority",
            "jac_solution_weighted_scaled_authority",
        ),
        ndim=2,
    )

    if values is None:
        B = _as_float_array(
            info,
            ("B_sequence_solution",),
            ndim=3,
        )

        if B is None:
            return None

        scale = _channel_scale(
            info,
            B.shape[2],
        )

        values = _authority_from_B(
            B,
            scale,
        )

    return _plot_magnitude_visibility(
        values=values,
        k=k,
        output_root=output_root,
        folder=(
            "actual_jacobian_authority_split"
        ),
        title=(
            "Actual stagewise Jacobian authority "

        ),
        absolute_ylabel=(
            "Scaled authority [mm]"
        ),
        channel_names=channel_names,
        dpi=dpi,
        value_multiplier=1e3,
    )


def plot_controller_jacobian_authority_split(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
    dpi: int = 160,
) -> Path | None:
    """
    Plot the authority assumed by the controller linearization.
    """
    values = _as_float_array(
        info,
        (
            "jac_linearisation_scaled_authority",
            "jac_linearisation_weighted_scaled_authority",
        ),
        ndim=2,
    )

    if values is None:
        B = _as_float_array(
            info,
            (
                "B_sequence_linearisation",
                "B_sequence",
            ),
            ndim=3,
        )

        if B is None:
            return None

        scale = _channel_scale(
            info,
            B.shape[2],
        )

        values = _authority_from_B(
            B,
            scale,
        )

    return _plot_magnitude_visibility(
        values=values,
        k=k,
        output_root=output_root,
        folder=(
            "controller_jacobian_authority_split"
        ),
        title=(
            "Controller-assumed Jacobian authority"
        ),
        absolute_ylabel=(
            "Scaled authority [mm]"
        ),
        channel_names=channel_names,
        dpi=dpi,
        value_multiplier=1e3,
    )


def plot_selected_channel_contributions_split(
    *,
    k: int,
    info: dict,
    output_root: Path,
    channel_names: tuple[str, ...] = DEFAULT_CHANNEL_NAMES,
    dpi: int = 160,
) -> Path | None:
    """
    Plot the actual selected channel contribution magnitudes:

        ||B[:,j] u_j||

    dL is shown separately. The middle panel exposes the shape of
    weak source-channel contributions without altering the absolute
    panel.
    """
    values = _as_float_array(
        info,
        (
            "jac_solution_contribution_norm",
        ),
        ndim=2,
    )

    if values is None:
        B = _as_float_array(
            info,
            (
                "B_sequence_solution",
                "B_sequence_linearisation",
                "B_sequence",
            ),
            ndim=3,
        )

        U = _as_float_array(
            info,
            ("U_seq",),
            ndim=2,
        )

        if B is None or U is None:
            return None

        values = _contribution_from_B_and_U(
            B,
            U,
        )

    return _plot_magnitude_visibility(
        values=values,
        k=k,
        output_root=output_root,
        folder=(
            "selected_channel_contributions_split"
        ),
        title=(
            "Selected per-channel tip-motion contribution"
        ),
        absolute_ylabel=(
            "||B[:,j] u_j|| [mm]"
        ),
        channel_names=channel_names,
        dpi=dpi,
        value_multiplier=1e3,
    )
