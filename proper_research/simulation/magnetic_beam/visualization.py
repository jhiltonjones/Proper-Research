from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as Rot


def _points_n_by_3(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, float)
    if points.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got {points.shape}.")
    if points.shape[1] == 3:
        return points.copy()
    if points.shape[0] == 3:
        return points.T.copy()
    raise ValueError(f"{name} must have shape (N, 3) or (3, N), got {points.shape}.")


def _unit(v: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(3)
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        return v / norm
    fallback = np.asarray(fallback, float).reshape(3)
    return fallback / max(float(np.linalg.norm(fallback)), 1e-12)


def _tube_surface(
    centreline: np.ndarray,
    radius: np.ndarray,
    *,
    n_theta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a circular tube around a sampled 3D centreline."""
    C = _points_n_by_3(centreline, name="lumen centreline")
    radius = np.asarray(radius, float).reshape(-1)

    if C.shape[0] < 2:
        raise ValueError("Lumen centreline must contain at least two points.")
    if radius.size != C.shape[0]:
        raise ValueError(
            f"Lumen radius has {radius.size} entries for {C.shape[0]} points."
        )
    if np.any(radius <= 0):
        raise ValueError("All lumen radii must be positive.")
    if n_theta < 8:
        raise ValueError("n_theta must be at least 8.")

    tangents = np.gradient(C, axis=0)
    tangents = np.vstack(
        [_unit(tangent, fallback=np.array([1.0, 0.0, 0.0])) for tangent in tangents]
    )

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    # Choose the global axis least aligned with the first tangent.
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ tangents[0])))]
    normals[0] = _unit(
        np.cross(tangents[0], reference),
        fallback=np.array([0.0, 1.0, 0.0]),
    )
    binormals[0] = _unit(
        np.cross(tangents[0], normals[0]),
        fallback=np.array([0.0, 0.0, 1.0]),
    )

    # Rotation-minimising-style transport: project the previous normal onto
    # each new normal plane. This avoids arbitrary tube twisting.
    for i in range(1, C.shape[0]):
        projected = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        if np.linalg.norm(projected) < 1e-10:
            reference = axes[int(np.argmin(np.abs(axes @ tangents[i])))]
            projected = np.cross(tangents[i], reference)
        normals[i] = _unit(projected, fallback=normals[i - 1])
        binormals[i] = _unit(
            np.cross(tangents[i], normals[i]),
            fallback=binormals[i - 1],
        )

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta) + 1)
    radial = (
        np.cos(theta)[None, :, None] * normals[:, None, :]
        + np.sin(theta)[None, :, None] * binormals[:, None, :]
    )
    surface = C[:, None, :] + radius[:, None, None] * radial
    return surface[:, :, 0], surface[:, :, 1], surface[:, :, 2]


def _draw_source_magnet(
    ax,
    *,
    position: np.ndarray,
    axis_world: np.ndarray,
    length: float,
    radius: float,
    n_theta: int = 32,
) -> np.ndarray:
    """Draw a two-pole cylindrical source magnet and return its end points."""
    position = np.asarray(position, float).reshape(3)
    axis_world = _unit(axis_world, fallback=np.array([0.0, 0.0, 1.0]))

    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ axis_world)))]
    normal = _unit(
        np.cross(axis_world, reference),
        fallback=np.array([1.0, 0.0, 0.0]),
    )
    binormal = _unit(
        np.cross(axis_world, normal),
        fallback=np.array([0.0, 1.0, 0.0]),
    )

    theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta) + 1)
    circle = (
        np.cos(theta)[:, None] * normal[None, :]
        + np.sin(theta)[:, None] * binormal[None, :]
    )

    pole_colours = ("tab:blue", "tab:red")
    axial_halves = ((-0.5 * length, 0.0), (0.0, 0.5 * length))
    for (z0, z1), colour in zip(axial_halves, pole_colours, strict=True):
        axial = np.array([z0, z1], dtype=float)
        surface = (
            position[None, None, :]
            + axial[:, None, None] * axis_world[None, None, :]
            + radius * circle[None, :, :]
        )
        ax.plot_surface(
            surface[:, :, 0],
            surface[:, :, 1],
            surface[:, :, 2],
            color=colour,
            alpha=0.85,
            linewidth=0.0,
            shade=True,
        )

    south = position - 0.5 * length * axis_world
    north = position + 0.5 * length * axis_world
    ax.scatter(*position, color="tab:orange", marker="o", s=32, depthshade=False)
    ax.quiver(
        *position,
        *(0.75 * length * axis_world),
        color="tab:red",
        arrow_length_ratio=0.22,
        linewidth=2.0,
    )
    ax.text(*south, " S", color="tab:blue")
    ax.text(*north, " N", color="tab:red")
    return np.vstack([south, north])


def _set_equal_3d_limits(ax, points: np.ndarray, *, padding_fraction: float = 0.08):
    points = _points_n_by_3(points, name="scene points")
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    centre = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1e-6)
    half = 0.5 * span * (1.0 + 2.0 * float(padding_fraction))

    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_magnetic_beam_scene(
    *,
    p7: np.ndarray,
    centerline: np.ndarray,
    lumen_query=None,
    magnet_axis_body: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
    magnet_length: float | None = None,
    magnet_radius: float | None = None,
    n_lumen_theta: int = 28,
    show_lumen_surface: bool = True,
    ax=None,
    title: str | None = "Magnetic-beam forward solution",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """
    Plot source magnet, solved beam centreline, and circular lumen geometry.

    ``p7`` follows the model convention
    ``[x, y, z, rx, ry, rz, L_inserted]``. Rotation is interpreted as a
    SciPy/UR rotation vector. The returned ``(fig, ax)`` can be further edited
    by a caller or used in a non-interactive test.
    """
    p7 = np.asarray(p7, float).reshape(-1)
    if p7.size != 7:
        raise ValueError(f"p7 must have seven entries, got {p7.shape}.")

    beam = _points_n_by_3(centerline, name="beam centerline")
    if beam.shape[0] < 2:
        raise ValueError("Beam centerline must contain at least two nodes.")

    if ax is None:
        fig = plt.figure(figsize=(9.0, 7.0), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    scene_points = [beam, p7[None, 0:3]]

    if lumen_query is not None:
        lumen_C = _points_n_by_3(lumen_query.C, name="lumen centreline")
        lumen_R = np.asarray(lumen_query.R, float).reshape(-1)
        scene_points.append(lumen_C)

        if show_lumen_surface:
            X, Y, Z = _tube_surface(
                lumen_C,
                lumen_R,
                n_theta=n_lumen_theta,
            )
            ax.plot_surface(
                X,
                Y,
                Z,
                color="tab:cyan",
                alpha=0.16,
                linewidth=0.0,
                shade=False,
            )
            scene_points.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        ax.plot(
            lumen_C[:, 0],
            lumen_C[:, 1],
            lumen_C[:, 2],
            color="tab:cyan",
            linestyle="--",
            linewidth=1.2,
            label="Lumen centreline",
        )

    ax.plot(
        beam[:, 0],
        beam[:, 1],
        beam[:, 2],
        color="tab:green",
        marker="o",
        markersize=3.0,
        linewidth=2.8,
        label="Solved beam",
    )
    ax.scatter(*beam[0], color="tab:purple", marker="s", s=46, label="Beam base")
    ax.scatter(*beam[-1], color="tab:green", marker="*", s=110, label="Beam tip")

    extent_points = np.vstack(scene_points)
    characteristic_length = max(
        float(np.max(np.ptp(extent_points, axis=0))),
        float(abs(p7[6])),
        1e-3,
    )
    if magnet_length is None:
        magnet_length = 0.12 * characteristic_length
    if magnet_radius is None:
        magnet_radius = 0.28 * float(magnet_length)
    if magnet_length <= 0 or magnet_radius <= 0:
        raise ValueError("magnet_length and magnet_radius must be positive.")

    axis_world = Rot.from_rotvec(p7[3:6]).apply(
        _unit(magnet_axis_body, fallback=np.array([0.0, 0.0, 1.0]))
    )
    magnet_ends = _draw_source_magnet(
        ax,
        position=p7[0:3],
        axis_world=axis_world,
        length=float(magnet_length),
        radius=float(magnet_radius),
    )
    scene_points.append(magnet_ends)

    magnet_legend = Line2D(
        [0],
        [0],
        color="tab:red",
        linewidth=5,
        label="Source magnet / dipole axis",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(magnet_legend)
    labels.append(magnet_legend.get_label())
    ax.legend(handles, labels, loc="best")

    _set_equal_3d_limits(ax, np.vstack(scene_points))
    ax.set_xlabel("World x [m]")
    ax.set_ylabel("World y [m]")
    ax.set_zlabel("World z [m]")
    if title:
        ax.set_title(title)
    ax.view_init(elev=24.0, azim=-55.0)

    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
