"""Configuration dataclasses for reproducible simulations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    n_nodes: int = 120
    tolerance: float = 1e-5
    max_nodes: int = 20_000
    sensitivity_points: int = 200
    sensitivity_rtol: float = 1e-6
    sensitivity_atol: float = 1e-8

    def __post_init__(self) -> None:
        if self.n_nodes < 5:
            raise ValueError("n_nodes must be at least 5")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_nodes < self.n_nodes:
            raise ValueError("max_nodes must be at least n_nodes")


@dataclass(frozen=True)
class MaterialConfig:
    wire_radius_m: float = 200e-6
    wire_youngs_modulus_pa: float = 50e6
    wire_poisson_ratio: float = 0.4
    tip_youngs_modulus_pa: float = 1e6
    tip_poisson_ratio: float = 0.49
    bend_softening: float = 1.0
    torsion_softening: float = 1.0


@dataclass(frozen=True)
class InsertionConfig:
    commanded_length_m: float = 0.025
    full_tip_length_m: float = 0.04
    minimum_model_length_m: float = 0.01

    def __post_init__(self) -> None:
        if self.commanded_length_m <= 0.0:
            raise ValueError("commanded_length_m must be positive")
        if self.full_tip_length_m <= 0.0:
            raise ValueError("full_tip_length_m must be positive")
        if self.minimum_model_length_m <= 0.0:
            raise ValueError("minimum_model_length_m must be positive")


@dataclass(frozen=True)
class PoseConfig:
    pivot_pose6: tuple[float, ...] = (
        0.7981328220229531,
        -0.70992731669220016,
        -0.1,
        np.pi,
        0.001,
        0.001,
    )
    magnet_start_pose6: tuple[float, ...] = (
        0.6531328220229531,
        -0.70992731669220016,
        -0.1,
        np.pi,
        0.001,
        0.001,
    )


@dataclass(frozen=True)
class SweepConfig:
    start_angle_deg: float = -180.0
    stop_angle_deg: float = 180.0
    angle_count: int = 101
    front_distance_m: float = 0.2
    side_distance_m: float = 0.2
    side_angles_deg: tuple[float, ...] = (-30.0, -60.0)
    nominal_placement: str = "side_30"
    compute_sensitivity: bool = True
    compute_energy: bool = True
    make_plots: bool = True

    def __post_init__(self) -> None:
        if self.angle_count < 2:
            raise ValueError("angle_count must be at least 2")
        if self.front_distance_m <= 0.0 or self.side_distance_m <= 0.0:
            raise ValueError("placement distances must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    poses: PoseConfig = field(default_factory=PoseConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    output_root: Path = Path("results")

    def with_nodes(self, n_nodes: int) -> "ExperimentConfig":
        return replace(self, solver=replace(self.solver, n_nodes=n_nodes))

    def with_output_root(self, output_root: str | Path) -> "ExperimentConfig":
        return replace(self, output_root=Path(output_root))
