"""Visualize an inverse or global magnetic-beam configuration path.

The figure connects the four representations used by the planner:

    lumen arc length s
        -> robot/insertion state [q1..q6, L]
        -> source-magnet pose T_R_M
        -> beam centreline, tip and tangent

For inverse results, the script reads ``inverse_configuration_summary.json``.
For global results, it reads ``global_configuration_summary.json``.  When beam
replay is enabled, it rebuilds the planning context and evaluates the states in
path order with ``commit=True`` so that the equilibrium continuation branch is
preserved.  Only a small number of replayed beam centrelines are drawn, while
all states are still evaluated sequentially to reach those snapshots safely.

Examples
--------
python -u -m proper_research.planning.visualize_configuration_path

python -u -m proper_research.planning.visualize_configuration_path \
    --path-dir joint_space_smoke_test_60/.../global_configuration_full_debug \
    --kind global --snapshots 7
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.planning.planning_context import build_planning_context


Array = np.ndarray


@dataclass(frozen=True)
class PathData:
    kind: str
    source_directory: Path
    s_m: Array
    states: Array
    magnet_pose6: Array
    desired_position_m: Array
    achieved_position_m: Array
    desired_tangent: Array
    achieved_tangent: Array
    position_error_m: Array
    tangent_error_rad: Array
    feasible: Array
    position_tolerance_m: float
    tangent_tolerance_rad: float


def _finite_matrix(value: Any, columns: int, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}); got {result.shape}.")
    if result.shape[0] == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values and at least one row.")
    return result


def _unit_rows(value: Any, name: str) -> Array:
    result = _finite_matrix(value, 3, name)
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= 1.0e-12):
        raise ValueError(f"{name} contains a near-zero vector.")
    return result / norms[:, None]


def _infer_kind(directory: Path) -> str:
    inverse = directory / "inverse_configuration_summary.json"
    global_path = directory / "global_configuration_summary.json"
    if inverse.exists() and global_path.exists():
        raise ValueError(
            f"Both inverse and global summaries exist in {directory}; pass --kind."
        )
    if inverse.exists():
        return "inverse"
    if global_path.exists():
        return "global"
    raise FileNotFoundError(
        "Could not find inverse_configuration_summary.json or "
        f"global_configuration_summary.json in {directory}."
    )


def _load_path(directory: Path, kind: str) -> PathData:
    directory = Path(directory)
    resolved_kind = _infer_kind(directory) if kind == "auto" else kind
    filename = (
        "inverse_configuration_summary.json"
        if resolved_kind == "inverse"
        else "global_configuration_summary.json"
    )
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])
    if not nodes:
        raise ValueError(f"{path} contains no nodes.")

    s_m = np.asarray([node["s_m"] for node in nodes], dtype=float)
    q_rad = _finite_matrix([node["q_rad"] for node in nodes], 6, "joint states")
    insertion_m = np.asarray([node["insertion_m"] for node in nodes], dtype=float)
    states = np.column_stack((q_rad, insertion_m))
    magnet_pose6 = _finite_matrix(
        [node["magnet_pose6"] for node in nodes], 6, "magnet poses"
    )

    if resolved_kind == "inverse":
        achieved_key = "tip_position_m"
        tangent_key = "tip_tangent"
    else:
        achieved_key = "achieved_position_m"
        tangent_key = "achieved_tangent"

    desired_position = _finite_matrix(
        [node["desired_position_m"] for node in nodes], 3, "desired positions"
    )
    achieved_position = _finite_matrix(
        [node[achieved_key] for node in nodes], 3, "achieved positions"
    )
    desired_tangent = _unit_rows(
        [node["desired_tangent"] for node in nodes], "desired tangents"
    )
    achieved_tangent = _unit_rows(
        [node[tangent_key] for node in nodes], "achieved tangents"
    )
    position_error = np.asarray(
        [node["position_error_m"] for node in nodes], dtype=float
    )
    tangent_error = np.asarray(
        [node["tangent_error_rad"] for node in nodes], dtype=float
    )
    feasible = np.asarray([bool(node["feasible"]) for node in nodes], dtype=bool)
    configuration = payload.get("configuration", {})
    position_tolerance = float(configuration["position_tolerance_m"])
    tangent_tolerance = float(configuration["tangent_tolerance_rad"])

    if np.any(np.diff(s_m) <= 0.0):
        raise ValueError("Path coordinates must be strictly increasing.")
    return PathData(
        kind=resolved_kind,
        source_directory=directory,
        s_m=s_m,
        states=states,
        magnet_pose6=magnet_pose6,
        desired_position_m=desired_position,
        achieved_position_m=achieved_position,
        desired_tangent=desired_tangent,
        achieved_tangent=achieved_tangent,
        position_error_m=position_error,
        tangent_error_rad=tangent_error,
        feasible=feasible,
        position_tolerance_m=position_tolerance,
        tangent_tolerance_rad=tangent_tolerance,
    )


def _snapshot_indices(data: PathData, count: int) -> Array:
    count = min(max(2, int(count)), data.s_m.size)
    evenly_spaced = np.rint(np.linspace(0, data.s_m.size - 1, count)).astype(int)
    worst_position = int(np.argmax(data.position_error_m))
    failures = np.flatnonzero(~data.feasible)
    indices = set(int(index) for index in evenly_spaced)
    indices.add(worst_position)
    indices.update(int(index) for index in failures[:3])
    return np.asarray(sorted(indices), dtype=int)


def _replay_beam_shapes(
    *,
    data: PathData,
    controller_pack: dict[str, Any],
    selected_indices: Array,
) -> dict[int, Array]:
    adapter = controller_pack["plant_diagnostic_joint_adapter"]
    pose_adapter = controller_pack.get("plant_forward_pose8_adapter")
    if pose_adapter is None:
        raise KeyError("controller_pack has no plant_forward_pose8_adapter.")

    capture = getattr(pose_adapter, "capture_cache_state", None)
    restore = getattr(pose_adapter, "restore_cache_state", None)
    snapshot = capture() if callable(capture) else None
    reset_to_initial = getattr(pose_adapter, "reset_to_initial_baseline", None)
    selected = set(int(index) for index in selected_indices)
    shapes: dict[int, Array] = {}
    try:
        if callable(reset_to_initial):
            reset_to_initial()
        else:
            reset = getattr(pose_adapter, "reset", None)
            if callable(reset):
                reset()
        for index, state in enumerate(data.states):
            adapter.forward_output(state, commit=True)
            if index not in selected:
                continue
            centerline = getattr(pose_adapter, "last_p_centerline", None)
            if centerline is None:
                raise RuntimeError(
                    "Beam replay produced no last_p_centerline at "
                    f"node {index}."
                )
            shapes[index] = _finite_matrix(
                centerline, 3, f"beam centreline at node {index}"
            ).copy()
    finally:
        if snapshot is not None and callable(restore):
            restore(snapshot)
    return shapes


def _set_equal_3d(axis, points: Array) -> None:
    points = _finite_matrix(points, 3, "3D scene points")
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(1.0e-4, 0.55 * float(np.max(upper - lower)))
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _draw_magnet_frame(axis, pose6: Array, colour: Any, scale: float) -> None:
    position = pose6[:3]
    rotation = Rot.from_rotvec(pose6[3:6]).as_matrix()
    axis.scatter(*position, marker="s", s=32, color=colour, depthshade=False)
    # Draw the magnet's local z direction prominently.  This is an orientation
    # marker, not a physical line connecting the magnet to the beam.
    direction = rotation[:, 2]
    axis.quiver(
        *position,
        *(scale * direction),
        color=colour,
        linewidth=1.5,
        arrow_length_ratio=0.25,
    )


def _make_figure(
    *,
    data: PathData,
    lumen_C: Array,
    beam_shapes: dict[int, Array],
    selected_indices: Array,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(17, 14), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(1.45, 1.0, 1.0))
    axis_scene = figure.add_subplot(grid[0, 0], projection="3d")
    axis_close = figure.add_subplot(grid[0, 1], projection="3d")
    axis_joints = figure.add_subplot(grid[1, 0])
    axis_magnet = figure.add_subplot(grid[1, 1])
    axis_insertion = figure.add_subplot(grid[2, 0])
    axis_error = figure.add_subplot(grid[2, 1])

    lumen = _finite_matrix(np.asarray(lumen_C, dtype=float)[:, :3], 3, "lumen")
    axis_scene.plot(*lumen.T, "k--", linewidth=1.3, label="lumen centreline")
    axis_scene.plot(
        *data.desired_position_m.T,
        color="0.35",
        linestyle=":",
        linewidth=1.8,
        label="desired tip path",
    )
    axis_scene.plot(
        *data.achieved_position_m.T,
        color="tab:blue",
        linewidth=2.2,
        label="achieved beam-tip path",
    )
    axis_scene.plot(
        *data.magnet_pose6[:, :3].T,
        color="tab:red",
        linewidth=1.7,
        label="source-magnet path",
    )
    axis_close.plot(*lumen.T, "k--", linewidth=1.3, label="lumen centreline")
    axis_close.plot(
        *data.desired_position_m.T,
        color="0.35",
        linestyle=":",
        linewidth=1.8,
        label="desired tip path",
    )
    axis_close.plot(
        *data.achieved_position_m.T,
        color="tab:blue",
        linewidth=2.2,
        label="achieved beam-tip path",
    )

    colours = plt.cm.viridis(np.linspace(0.08, 0.92, selected_indices.size))
    all_scene_points = [
        lumen,
        data.desired_position_m,
        data.achieved_position_m,
        data.magnet_pose6[:, :3],
    ]
    span_points = np.vstack(all_scene_points)
    orientation_scale = max(2.0e-3, 0.035 * float(np.max(np.ptp(span_points, axis=0))))
    tangent_scale = orientation_scale
    for colour, index in zip(colours, selected_indices):
        index = int(index)
        shape = beam_shapes.get(index)
        if shape is not None:
            axis_scene.plot(
                *shape.T,
                color=colour,
                linewidth=2.1,
                alpha=0.95,
                label="beam snapshots" if index == int(selected_indices[0]) else None,
            )
            axis_close.plot(
                *shape.T,
                color=colour,
                linewidth=2.1,
                alpha=0.95,
                label="beam snapshots" if index == int(selected_indices[0]) else None,
            )
            all_scene_points.append(shape)
        _draw_magnet_frame(
            axis_scene,
            data.magnet_pose6[index],
            colour,
            orientation_scale,
        )
        tip = data.achieved_position_m[index]
        tangent = data.achieved_tangent[index]
        axis_scene.scatter(*tip, s=28, color=colour, depthshade=False)
        axis_scene.quiver(
            *tip,
            *(tangent_scale * tangent),
            color=colour,
            linewidth=1.3,
            arrow_length_ratio=0.25,
        )
        axis_close.scatter(*tip, s=28, color=colour, depthshade=False)
        axis_close.quiver(
            *tip,
            *(tangent_scale * tangent),
            color=colour,
            linewidth=1.3,
            arrow_length_ratio=0.25,
        )
        axis_scene.text(
            *tip,
            f"  {index} (s={1.0e3 * data.s_m[index]:.2f} mm)",
            color=colour,
            fontsize=8,
        )

    _set_equal_3d(axis_scene, np.vstack(all_scene_points))
    axis_scene.set_xlabel("world x [m]")
    axis_scene.set_ylabel("world y [m]")
    axis_scene.set_zlabel("world z [m]")
    axis_scene.set_title(
        "Robot state -> source magnet -> beam equilibrium -> lumen reference"
    )
    axis_scene.legend(loc="upper left", fontsize=8)

    close_points = np.vstack(
        [lumen, data.desired_position_m, data.achieved_position_m]
        + list(beam_shapes.values())
    )
    _set_equal_3d(axis_close, close_points)
    axis_close.set_xlabel("world x [m]")
    axis_close.set_ylabel("world y [m]")
    axis_close.set_zlabel("world z [m]")
    axis_close.set_title("Beam/lumen close-up")
    axis_close.legend(loc="upper left", fontsize=8)

    s_mm = 1.0e3 * data.s_m
    for joint in range(6):
        axis_joints.plot(s_mm, np.degrees(data.states[:, joint]), label=f"q{joint + 1}")
    axis_joints.set_title("Robotic-arm configuration path")
    axis_joints.set_xlabel("lumen arc length s [mm]")
    axis_joints.set_ylabel("joint angle [deg]")
    axis_joints.grid(alpha=0.25)
    axis_joints.legend(ncol=3, fontsize=8)

    for coordinate, label in enumerate(("x", "y", "z")):
        axis_magnet.plot(
            s_mm,
            1.0e3 * data.magnet_pose6[:, coordinate],
            label=f"magnet {label}",
        )
    axis_magnet.set_title("Source-magnet translation caused by q1..q6")
    axis_magnet.set_xlabel("lumen arc length s [mm]")
    axis_magnet.set_ylabel("world position [mm]")
    axis_magnet.grid(alpha=0.25)
    axis_magnet.legend(fontsize=8)

    axis_insertion.plot(
        s_mm,
        1.0e3 * data.states[:, 6],
        color="tab:purple",
        linewidth=2.0,
    )
    axis_insertion.set_title("Beam insertion state")
    axis_insertion.set_xlabel("lumen arc length s [mm]")
    axis_insertion.set_ylabel("insertion L [mm]")
    axis_insertion.grid(alpha=0.25)

    axis_error.plot(
        s_mm,
        1.0e3 * data.position_error_m,
        color="tab:blue",
        label="position error [mm]",
    )
    axis_error.axhline(
        1.0e3 * data.position_tolerance_m,
        color="tab:blue",
        linestyle="--",
        alpha=0.75,
        label="position tolerance",
    )
    axis_angle = axis_error.twinx()
    axis_angle.plot(
        s_mm,
        np.degrees(data.tangent_error_rad),
        color="tab:orange",
        label="tangent error [deg]",
    )
    axis_angle.axhline(
        np.degrees(data.tangent_tolerance_rad),
        color="tab:orange",
        linestyle="--",
        alpha=0.75,
        label="tangent tolerance",
    )
    failures = np.flatnonzero(~data.feasible)
    if failures.size:
        axis_error.scatter(
            s_mm[failures],
            1.0e3 * data.position_error_m[failures],
            marker="x",
            color="red",
            label="infeasible node",
            zorder=5,
        )
    axis_error.set_title("Physical tracking constraints")
    axis_error.set_xlabel("lumen arc length s [mm]")
    axis_error.set_ylabel("position error [mm]", color="tab:blue")
    axis_angle.set_ylabel("tangent error [deg]", color="tab:orange")
    axis_error.grid(alpha=0.25)
    handles_a, labels_a = axis_error.get_legend_handles_labels()
    handles_b, labels_b = axis_angle.get_legend_handles_labels()
    axis_error.legend(handles_a + handles_b, labels_a + labels_b, fontsize=7)

    figure.suptitle(
        f"{data.kind.capitalize()} configuration path: "
        f"{data.s_m.size} nodes, {int(np.count_nonzero(data.feasible))} feasible",
        fontsize=16,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize robot, source-magnet and magnetic-beam path data."
    )
    parser.add_argument(
        "--path-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing an inverse or global summary. The default is "
            "the same-process inverse directory under the planning context."
        ),
    )
    parser.add_argument(
        "--kind", choices=("auto", "inverse", "global"), default="auto"
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=6,
        help="Number of evenly spaced beam/magnet snapshots (default: 6).",
    )
    parser.add_argument(
        "--skip-beam-replay",
        action="store_true",
        help="Plot saved tip/magnet paths without recomputing full beam shapes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: PATH_DIR/configuration_path_scene.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the saved image after generation when a GUI backend is available.",
    )
    arguments = parser.parse_args()
    if arguments.snapshots < 2:
        parser.error("--snapshots must be at least 2")
    return arguments


def main() -> None:
    arguments = _arguments()
    _, bundle, controller_pack, out_root = build_planning_context()
    path_directory = arguments.path_dir or (
        out_root / "offline_inverse_configuration_60"
    )
    data = _load_path(path_directory, arguments.kind)
    selected_indices = _snapshot_indices(data, arguments.snapshots)

    beam_shapes: dict[int, Array] = {}
    if not arguments.skip_beam_replay:
        try:
            beam_shapes = _replay_beam_shapes(
                data=data,
                controller_pack=controller_pack,
                selected_indices=selected_indices,
            )
        except Exception as exc:
            warnings.warn(
                "Full beam-shape replay failed; plotting saved tip/magnet data "
                f"only. Reason: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )

    output_path = arguments.output or (
        path_directory / "configuration_path_scene.png"
    )
    _make_figure(
        data=data,
        lumen_C=bundle.lumen_C,
        beam_shapes=beam_shapes,
        selected_indices=selected_indices,
        output_path=output_path,
    )
    print(f"Saved path visualization: {output_path}")
    print(f"Selected snapshot nodes: {selected_indices.tolist()}")
    if beam_shapes:
        print(f"Replayed full beam shapes: {sorted(beam_shapes)}")
    else:
        print("Replayed full beam shapes: none")

    if arguments.show:
        try:
            from PIL import Image

            Image.open(output_path).show()
        except Exception as exc:
            warnings.warn(f"Could not open image viewer: {exc}", RuntimeWarning)


if __name__ == "__main__":
    main()
