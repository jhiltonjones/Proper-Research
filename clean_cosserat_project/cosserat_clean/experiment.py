"""High-level nominal and placement/orientation sweep workflows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .analysis import (
    compute_total_energy,
    jacobian_metrics,
    magnetic_net_force_torque,
    tip_bending_angles_signed,
)
from .config import ExperimentConfig
from .external import load_external_dependencies
from .magnetics import dipole_field_from_source
from .mechanics import (
    effective_lengths,
    make_inverse_bending_torsion_profile,
    rod_section_stiffness,
)
from .plotting import save_standard_sweep_plots
from .quaternions import dipole_from_pose, skew
from .rod import CosseratForwardModel
from .sensitivity import solve_tip_sensitivity_source_dipole_and_length


@dataclass(frozen=True)
class Placement:
    name: str
    label: str
    distance_m: float
    source_position: np.ndarray
    source_dipole_base: np.ndarray


@dataclass
class SimulationContext:
    config: ExperimentConfig
    beam_parameters: object
    magnet_parameters: object
    base_position: np.ndarray
    base_quaternion: np.ndarray
    beam_axis: np.ndarray
    local_y_axis: np.ndarray
    local_z_axis: np.ndarray
    straight_tip_position: np.ndarray
    source_dipole_body: np.ndarray
    source_dipole_magnitude: float
    model_length_m: float
    wire_length_m: float
    tip_length_m: float
    inverse_stiffness: Callable
    gravity_density: np.ndarray | None
    placements: dict[str, Placement]

    def make_model(self) -> CosseratForwardModel:
        ext = load_external_dependencies()
        profile = ext.make_m_local_fun_wire_tip(
            self.wire_length_m,
            len_tip=self.tip_length_m,
            mode="axial",
            alpha_end=0.0,
            eps=1e-3,
        )
        return CosseratForwardModel(
            base_position=self.base_position,
            base_quaternion=self.base_quaternion,
            inverse_stiffness=self.inverse_stiffness,
            magnetisation_profile=profile,
            magnetic_parameter=0.0,
            wire_length=self.wire_length_m,
            n_nodes=self.config.solver.n_nodes,
            tolerance=self.config.solver.tolerance,
            max_nodes=self.config.solver.max_nodes,
            gravity_density=None,
        )


@dataclass
class SweepResult:
    output_directory: Path
    angles_deg: np.ndarray
    placement_names: list[str]
    placement_labels: dict[str, str]
    maps: dict[str, np.ndarray]
    table: pd.DataFrame


def build_context(config: ExperimentConfig) -> SimulationContext:
    """Resolve external parameters and construct all deterministic model inputs."""
    ext = load_external_dependencies()
    beam_parameters = ext.default_beam_params()
    magnet_parameters = ext.default_magnet_params()

    pivot_pose = np.asarray(config.poses.pivot_pose6, dtype=float)
    pivot_transform = ext.ur_pose6_to_T(pivot_pose)
    base_position, base_quaternion = ext.T_to_p_quat_wxyz(pivot_transform)
    base_position = np.asarray(base_position, dtype=float).reshape(3)
    base_quaternion = np.asarray(base_quaternion, dtype=float).reshape(4)

    rotation = Rotation.from_quat(
        [base_quaternion[1], base_quaternion[2], base_quaternion[3], base_quaternion[0]]
    ).as_matrix()
    beam_axis = _unit(rotation @ np.array([-1.0, 0.0, 0.0]))
    local_y_axis = _unit(rotation @ np.array([0.0, -1.0, 0.0]))
    local_z_axis = _unit(rotation @ np.array([0.0, 0.0, 1.0]))

    inserted = config.insertion.commanded_length_m
    model_length, wire_length, tip_length = effective_lengths(
        inserted,
        full_tip_length_m=config.insertion.full_tip_length_m,
        minimum_model_length_m=config.insertion.minimum_model_length_m,
    )
    straight_tip = base_position + inserted * beam_axis

    wire = rod_section_stiffness(
        config.material.wire_radius_m,
        config.material.wire_youngs_modulus_pa,
        config.material.wire_poisson_ratio,
    )
    tip = rod_section_stiffness(
        float(beam_parameters.r),
        config.material.tip_youngs_modulus_pa,
        config.material.tip_poisson_ratio,
    )
    inverse_stiffness = make_inverse_bending_torsion_profile(
        wire_bending_stiffness=wire.bending_stiffness_nm2,
        tip_bending_stiffness=tip.bending_stiffness_nm2,
        wire_torsional_stiffness=wire.torsional_stiffness_nm2,
        tip_torsional_stiffness=tip.torsional_stiffness_nm2,
        bend_softening=config.material.bend_softening,
        torsion_softening=config.material.torsion_softening,
    )

    source_dipole_body = np.array([-float(magnet_parameters.mag_epm), 0.0, 0.0])
    start_pose = np.asarray(config.poses.magnet_start_pose6, dtype=float)
    source_transform = ext.ur_pose6_to_T(start_pose)
    _, source_quaternion = ext.T_to_p_quat_wxyz(source_transform)
    source_dipole_magnitude = float(
        np.linalg.norm(dipole_from_pose(source_quaternion, source_dipole_body))
    )
    placements = _make_placements(
        config=config,
        straight_tip=straight_tip,
        beam_axis=beam_axis,
        source_dipole_magnitude=source_dipole_magnitude,
    )
    gravity_density = getattr(beam_parameters, "f_g", None)
    if gravity_density is not None:
        gravity_density = np.asarray(gravity_density, dtype=float).reshape(3)

    return SimulationContext(
        config=config,
        beam_parameters=beam_parameters,
        magnet_parameters=magnet_parameters,
        base_position=base_position,
        base_quaternion=base_quaternion,
        beam_axis=beam_axis,
        local_y_axis=local_y_axis,
        local_z_axis=local_z_axis,
        straight_tip_position=straight_tip,
        source_dipole_body=source_dipole_body,
        source_dipole_magnitude=source_dipole_magnitude,
        model_length_m=model_length,
        wire_length_m=wire_length,
        tip_length_m=tip_length,
        inverse_stiffness=inverse_stiffness,
        gravity_density=gravity_density,
        placements=placements,
    )


def _make_placements(
    *,
    config: ExperimentConfig,
    straight_tip: np.ndarray,
    beam_axis: np.ndarray,
    source_dipole_magnitude: float,
) -> dict[str, Placement]:
    base_dipole = source_dipole_magnitude * beam_axis
    placements: dict[str, Placement] = {}
    front_distance = config.sweep.front_distance_m
    placements["front"] = Placement(
        name="front",
        label="In front",
        distance_m=front_distance,
        source_position=straight_tip + front_distance * beam_axis,
        source_dipole_base=base_dipole,
    )

    beam_axis_xy = beam_axis.copy()
    beam_axis_xy[2] = 0.0
    beam_axis_xy = _unit(beam_axis_xy)
    for angle_deg in config.sweep.side_angles_deg:
        side_axis = Rotation.from_euler("z", np.deg2rad(angle_deg)).as_matrix() @ beam_axis_xy
        side_axis[2] = 0.0
        side_axis = _unit(side_axis)
        absolute_angle = int(round(abs(angle_deg)))
        name = f"side_{absolute_angle}"
        placements[name] = Placement(
            name=name,
            label=f"Side {absolute_angle} deg",
            distance_m=config.sweep.side_distance_m,
            source_position=straight_tip + config.sweep.side_distance_m * side_axis,
            source_dipole_base=base_dipole,
        )
    if config.sweep.nominal_placement not in placements:
        raise ValueError(
            f"nominal placement '{config.sweep.nominal_placement}' is not one of {list(placements)}"
        )
    return placements


def run_nominal(
    context: SimulationContext,
    *,
    compute_sensitivity: bool = True,
) -> dict[str, object]:
    """Run the configured nominal placement at zero world-z dipole rotation."""
    placement = context.placements[context.config.sweep.nominal_placement]
    source_dipole = placement.source_dipole_base.copy()
    model = context.make_model()
    solution = model.solve(
        length_m=context.model_length_m,
        source_position=placement.source_position,
        source_dipole=source_dipole,
    )
    metrics = _solution_metrics(
        context,
        model,
        solution,
        placement=placement,
        source_dipole=source_dipole,
        compute_sensitivity=compute_sensitivity,
    )
    metrics.update(
        {
            "placement": placement.name,
            "placement_label": placement.label,
            "source_position": placement.source_position,
            "source_dipole": source_dipole,
            "solution": solution,
        }
    )
    return metrics


def _solution_metrics(
    context: SimulationContext,
    model: CosseratForwardModel,
    solution,
    *,
    placement: Placement,
    source_dipole: np.ndarray,
    compute_sensitivity: bool,
) -> dict[str, object]:
    tip_position = np.asarray(solution.sol(np.array([context.model_length_m]))[0:3, 0], dtype=float)
    field_tip = dipole_field_from_source(
        tip_position.reshape(1, 3), placement.source_position, source_dipole
    )[0]
    angle_y, angle_z, angle_total = tip_bending_angles_signed(
        solution,
        length_m=context.model_length_m,
        tangent_reference=context.beam_axis,
        local_y_reference=context.local_y_axis,
        local_z_reference=context.local_z_axis,
    )
    net_force, net_torque = magnetic_net_force_torque(
        solution,
        length_m=context.model_length_m,
        source_position=placement.source_position,
        source_dipole=source_dipole,
        magnetisation_profile=model.magnetisation_profile,
        magnetic_parameter=model.magnetic_parameter,
        evaluation_points=300,
    )
    energy, energy_parts = compute_total_energy(
        solution,
        length_m=context.model_length_m,
        source_position=placement.source_position,
        source_dipole=source_dipole,
        inverse_stiffness=context.inverse_stiffness,
        magnetisation_profile=model.magnetisation_profile,
        magnetic_parameter=model.magnetic_parameter,
        wire_length=context.wire_length_m,
        gravity_density=context.gravity_density,
        evaluation_points=300,
    )

    result: dict[str, object] = {
        "tip_position": tip_position,
        "field_tip": field_tip,
        "field_norm_t": float(np.linalg.norm(field_tip)),
        "theta_y_deg": float(np.rad2deg(angle_y)),
        "theta_z_deg": float(np.rad2deg(angle_z)),
        "theta_total_deg": float(np.rad2deg(angle_total)),
        "net_force": net_force,
        "force_norm_n": float(np.linalg.norm(net_force)),
        "net_torque": net_torque,
        "torque_norm_nm": float(np.linalg.norm(net_torque)),
        "energy_j": float(energy),
        "energy_parts": energy_parts,
    }
    if compute_sensitivity:
        jacobian, diagnostics = solve_tip_sensitivity_source_dipole_and_length(
            solution,
            length_m=context.model_length_m,
            source_position=placement.source_position,
            source_dipole=source_dipole,
            source_dipole_rotation_jacobian=-skew(source_dipole),
            inverse_stiffness=context.inverse_stiffness,
            magnetic_parameter=model.magnetic_parameter,
            reference_strain=np.zeros(3),
            full_tip_length_m=context.config.insertion.full_tip_length_m,
            minimum_model_length_m=context.config.insertion.minimum_model_length_m,
            evaluation_points=context.config.solver.sensitivity_points,
            relative_tolerance=context.config.solver.sensitivity_rtol,
            absolute_tolerance=context.config.solver.sensitivity_atol,
        )
        result["tip_jacobian"] = jacobian
        result["jacobian_metrics"] = jacobian_metrics(jacobian)
        result["sensitivity_diagnostics"] = diagnostics
    return result


def run_orientation_sweep(
    context: SimulationContext,
    *,
    output_directory: str | Path | None = None,
    show_plots: bool = False,
) -> SweepResult:
    """Run the placement-versus-world-z dipole rotation comparison."""
    config = context.config
    angles_deg = np.linspace(
        config.sweep.start_angle_deg,
        config.sweep.stop_angle_deg,
        config.sweep.angle_count,
    )
    placement_names = list(context.placements)
    placement_labels = {
        name: context.placements[name].label for name in placement_names
    }
    shape = (len(placement_names), angles_deg.size)
    map_names = [
        "condition_3d",
        "condition_xy",
        "manipulability_3d",
        "manipulability_xy",
        "sigma1_3d",
        "sigma2_3d",
        "sigma3_3d",
        "sigma1_xy",
        "sigma2_xy",
        "jacobian_norm",
        "translation_norm",
        "rotation_norm",
        "length_norm",
        "theta_y_deg",
        "theta_z_deg",
        "theta_total_deg",
        "field_norm_t",
        "field_x_t",
        "field_y_t",
        "field_z_t",
        "force_norm_n",
        "torque_norm_nm",
        "energy_j",
    ]
    maps = {name: np.full(shape, np.nan, dtype=float) for name in map_names}
    rows: list[dict[str, object]] = []

    for placement_index, placement_name in enumerate(placement_names):
        placement = context.placements[placement_name]
        for angle_index, angle_deg in enumerate(angles_deg):
            source_dipole = _rotate_vector_about_world_z(
                placement.source_dipole_base, np.deg2rad(angle_deg)
            )
            model = context.make_model()
            row: dict[str, object] = {
                "placement": placement.name,
                "placement_label": placement.label,
                "distance_m": placement.distance_m,
                "distance_mm": 1000.0 * placement.distance_m,
                "zrot_deg": float(angle_deg),
                "zrot_rad": float(np.deg2rad(angle_deg)),
                "status": "ok",
                "error": "",
            }
            try:
                solution = model.solve(
                    length_m=context.model_length_m,
                    source_position=placement.source_position,
                    source_dipole=source_dipole,
                )
                metrics = _solution_metrics(
                    context,
                    model,
                    solution,
                    placement=placement,
                    source_dipole=source_dipole,
                    compute_sensitivity=config.sweep.compute_sensitivity,
                )
                field_tip = np.asarray(metrics["field_tip"], dtype=float)
                scalar_values: dict[str, float] = {
                    "theta_y_deg": float(metrics["theta_y_deg"]),
                    "theta_z_deg": float(metrics["theta_z_deg"]),
                    "theta_total_deg": float(metrics["theta_total_deg"]),
                    "field_norm_t": float(metrics["field_norm_t"]),
                    "field_x_t": float(field_tip[0]),
                    "field_y_t": float(field_tip[1]),
                    "field_z_t": float(field_tip[2]),
                    "force_norm_n": float(metrics["force_norm_n"]),
                    "torque_norm_nm": float(metrics["torque_norm_nm"]),
                    "energy_j": float(metrics["energy_j"]),
                }
                if "jacobian_metrics" in metrics:
                    scalar_values.update(metrics["jacobian_metrics"])
                for name, value in scalar_values.items():
                    if name in maps:
                        maps[name][placement_index, angle_index] = value
                    row[name] = value
            except Exception as exc:  # preserve sweep progress when one point fails
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    table = pd.DataFrame(rows)
    output = Path(output_directory) if output_directory is not None else make_run_directory(
        config.output_root, "placement_orientation_comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "placement_orientation_comparison.csv", index=False)
    np.savez(
        output / "placement_orientation_comparison.npz",
        angles_deg=angles_deg,
        placement_names=np.asarray(placement_names),
        placement_labels=np.asarray([placement_labels[name] for name in placement_names]),
        **maps,
    )
    _save_config(config, output / "config.json")
    result = SweepResult(
        output_directory=output,
        angles_deg=angles_deg,
        placement_names=placement_names,
        placement_labels=placement_labels,
        maps=maps,
        table=table,
    )
    if config.sweep.make_plots:
        save_standard_sweep_plots(result, output_directory=output, show=show_plots)
    return result


def make_run_directory(base: str | Path, name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    directory = Path(base) / f"{name}_{timestamp}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def quick_config(config: ExperimentConfig) -> ExperimentConfig:
    """Return a low-cost smoke-test sweep configuration."""
    return replace(
        config,
        solver=replace(config.solver, n_nodes=max(10, min(config.solver.n_nodes, 30))),
        sweep=replace(config.sweep, angle_count=9),
    )


def _rotate_vector_about_world_z(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    return Rotation.from_euler("z", float(angle_rad)).as_matrix() @ np.asarray(
        vector, dtype=float
    ).reshape(3)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    return vector / (np.linalg.norm(vector) + 1e-12)


def _save_config(config: ExperimentConfig, path: Path) -> None:
    data = asdict(config)
    data["output_root"] = str(config.output_root)
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
