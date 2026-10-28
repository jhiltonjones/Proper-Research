"""Plotting helpers for placement-versus-angle sweep results."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_placement_curves(
    values: np.ndarray,
    *,
    angles_deg: np.ndarray,
    placement_names: list[str],
    placement_labels: Mapping[str, str],
    title: str,
    ylabel: str,
    save_path: str | Path,
    logarithmic: bool = False,
    show: bool = False,
) -> None:
    """Plot one curve per placement and save it."""
    values = np.asarray(values, dtype=float)
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    for row, placement_name in enumerate(placement_names):
        curve = values[row, :]
        if logarithmic:
            curve = np.log10(np.maximum(curve, 1e-16))
        axis.plot(
            angles_deg,
            curve,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=placement_labels[placement_name],
        )
    axis.set_xlabel("External magnet dipole rotation about world z [deg]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_standard_sweep_plots(result, *, output_directory: str | Path, show: bool = False) -> None:
    """Save the standard set of comparison curves."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "angles_deg": result.angles_deg,
        "placement_names": result.placement_names,
        "placement_labels": result.placement_labels,
        "show": show,
    }
    plots = [
        ("condition_3d", "3D pose Jacobian conditioning", r"$\log_{10}\kappa(J_{xyz})$", True),
        ("condition_xy", "Planar xy pose Jacobian conditioning", r"$\log_{10}\kappa(J_{xy})$", True),
        ("sigma3_3d", "Weakest 3D singular value", r"$\log_{10}\sigma_3(J_{xyz})$", True),
        ("sigma2_xy", "Weakest planar singular value", r"$\log_{10}\sigma_2(J_{xy})$", True),
        ("theta_total_deg", "Total beam tip bending angle", r"$\theta_{total}$ [deg]", False),
        ("theta_y_deg", "Signed beam bending toward local y", r"$\theta_y$ [deg]", False),
        ("theta_z_deg", "Signed beam bending toward local z", r"$\theta_z$ [deg]", False),
        ("field_norm_t", "Magnetic field magnitude at beam tip", r"$\|B_{tip}\|$ [T]", False),
        ("field_x_t", "Tip magnetic field x-component", r"$B_x$ [T]", False),
        ("field_y_t", "Tip magnetic field y-component", r"$B_y$ [T]", False),
        ("field_z_t", "Tip magnetic field z-component", r"$B_z$ [T]", False),
        ("force_norm_n", "Net magnetic force magnitude", r"$\|F_{net}\|$ [N]", False),
        ("torque_norm_nm", "Net magnetic torque magnitude", r"$\|T_{net}\|$ [N m]", False),
        ("energy_j", "Total potential energy", r"$\Pi$ [J]", False),
    ]
    for key, title, ylabel, logarithmic in plots:
        if key not in result.maps:
            continue
        plot_placement_curves(
            result.maps[key],
            title=title,
            ylabel=ylabel,
            logarithmic=logarithmic,
            save_path=output / f"{key}.png",
            **common,
        )
