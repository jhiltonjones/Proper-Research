"""Cosserat rod equations and the nonlinear forward model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_bvp

from .external import load_external_dependencies
from .mechanics import make_initial_guess
from .quaternions import dipole_from_pose

StateArray = np.ndarray
KInverseFunction = Callable[[np.ndarray, float], np.ndarray]
MagnetisationFunction = Callable[..., np.ndarray]


def cosserat_rhs_single(
    s: float,
    state: StateArray,
    *,
    source_dipole: np.ndarray,
    source_position: np.ndarray,
    inverse_stiffness: KInverseFunction,
    magnetisation_profile: MagnetisationFunction,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
    gravity_density: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the 13-state Kirchhoff/Cosserat equations at one arclength."""
    ext = load_external_dependencies()
    state = np.asarray(state, dtype=float).reshape(13)
    position = state[0:3]
    quaternion_raw = state[3:7]
    internal_force = state[7:10]
    internal_moment_world = state[10:13]

    quaternion = ext.quat_normalize(quaternion_raw)
    rotation = ext.quat_to_R(quaternion)
    material_axis = np.array([-1.0, 0.0, 0.0], dtype=float)
    position_derivative = rotation @ material_axis

    internal_moment_body = rotation.T @ internal_moment_world
    stiffness_inverse = inverse_stiffness(np.array([s], dtype=float), wire_length)
    if stiffness_inverse.ndim == 3:
        stiffness_inverse = stiffness_inverse[:, :, 0]
    stiffness_inverse = np.asarray(stiffness_inverse, dtype=float).reshape(3, 3)

    reference = (
        np.zeros(3, dtype=float)
        if reference_strain is None
        else np.asarray(reference_strain, dtype=float).reshape(3)
    )
    strain = stiffness_inverse @ internal_moment_body + reference
    quaternion_derivative = ext.quat_derivative_body(quaternion, strain).reshape(4)

    force_density, torque_density, field, beam_dipole_world = (
        ext.magnetic_wrench_density_cosserat_point(
            s=s,
            p=position,
            q=quaternion,
            m_ext=np.asarray(source_dipole, dtype=float).reshape(3),
            r_src=np.asarray(source_position, dtype=float).reshape(3),
            m_local_fun=magnetisation_profile,
            m_front_or_overhead=magnetic_parameter,
            r_min=1e-6,
        )
    )
    force_density = np.asarray(force_density, dtype=float).reshape(3)
    torque_density = np.asarray(torque_density, dtype=float).reshape(3)
    if gravity_density is not None:
        force_density = force_density + np.asarray(gravity_density, dtype=float).reshape(3)

    derivative = np.zeros(13, dtype=float)
    derivative[0:3] = position_derivative
    derivative[3:7] = quaternion_derivative
    derivative[7:10] = -force_density
    derivative[10:13] = -np.cross(position_derivative, internal_force) - torque_density

    return derivative, {
        "position": position,
        "quaternion": quaternion,
        "rotation": rotation,
        "position_derivative": position_derivative,
        "internal_moment_body": internal_moment_body,
        "inverse_stiffness": stiffness_inverse,
        "strain": strain,
        "force_density": force_density,
        "torque_density": torque_density,
        "field": np.asarray(field, dtype=float).reshape(3),
        "beam_dipole_world": np.asarray(beam_dipole_world, dtype=float).reshape(3),
    }


def make_vectorized_rod_ode(
    *,
    source_dipole: np.ndarray,
    source_position: np.ndarray,
    inverse_stiffness: KInverseFunction,
    magnetisation_profile: MagnetisationFunction,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
    gravity_density: np.ndarray | None = None,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Wrap the pointwise RHS in the vectorized signature required by solve_bvp."""

    def ode(arclength: np.ndarray, states: np.ndarray) -> np.ndarray:
        arclength = np.asarray(arclength, dtype=float).ravel()
        states = np.asarray(states, dtype=float)
        derivatives = np.zeros_like(states)
        for index, s_value in enumerate(arclength):
            derivatives[:, index], _ = cosserat_rhs_single(
                float(s_value),
                states[:, index],
                source_dipole=source_dipole,
                source_position=source_position,
                inverse_stiffness=inverse_stiffness,
                magnetisation_profile=magnetisation_profile,
                magnetic_parameter=magnetic_parameter,
                wire_length=wire_length,
                reference_strain=reference_strain,
                gravity_density=gravity_density,
            )
        return derivatives

    return ode


def clamped_free_boundary_conditions(
    state_at_base: np.ndarray,
    state_at_tip: np.ndarray,
    *,
    base_position: np.ndarray,
    base_quaternion: np.ndarray,
) -> np.ndarray:
    """Clamp base position/orientation and leave tip force/moment free."""
    return np.hstack(
        [
            state_at_base[0:3] - np.asarray(base_position, dtype=float).reshape(3),
            state_at_base[3:7] - np.asarray(base_quaternion, dtype=float).reshape(4),
            state_at_tip[7:10],
            state_at_tip[10:13],
        ]
    )


@dataclass
class CosseratForwardModel:
    """Stateful nonlinear BVP model with continuation warm starts."""

    base_position: np.ndarray
    base_quaternion: np.ndarray
    inverse_stiffness: KInverseFunction
    magnetisation_profile: MagnetisationFunction
    magnetic_parameter: float
    wire_length: float
    n_nodes: int = 120
    tolerance: float = 1e-5
    max_nodes: int = 20_000
    gravity_density: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.base_position = np.asarray(self.base_position, dtype=float).reshape(3)
        self.base_quaternion = np.asarray(self.base_quaternion, dtype=float).reshape(4)
        self.wire_length = float(self.wire_length)
        self._previous_solution = None

    def _straight_initial_guess(self, length_m: float) -> tuple[np.ndarray, np.ndarray]:
        ext = load_external_dependencies()
        arclength = np.linspace(0.0, float(length_m), int(self.n_nodes))
        states = np.zeros((13, arclength.size), dtype=float)
        base_quaternion = ext.quat_normalize(self.base_quaternion)
        tangent = ext.quat_to_R(base_quaternion) @ np.array([-1.0, 0.0, 0.0])
        states[0:3, :] = self.base_position[:, None] + tangent[:, None] * arclength[None, :]
        states[3:7, :] = base_quaternion[:, None]
        return arclength, states

    def solve(
        self,
        *,
        length_m: float,
        source_position: np.ndarray,
        source_dipole: np.ndarray,
        select_lowest_energy_branch: bool = False,
    ):
        """Solve the nonlinear rod BVP."""
        length_m = float(length_m)
        ode = make_vectorized_rod_ode(
            source_dipole=source_dipole,
            source_position=source_position,
            inverse_stiffness=self.inverse_stiffness,
            magnetisation_profile=self.magnetisation_profile,
            magnetic_parameter=self.magnetic_parameter,
            wire_length=self.wire_length,
            gravity_density=self.gravity_density,
        )

        guesses: list[tuple[np.ndarray, np.ndarray]] = []
        if self._previous_solution is not None and getattr(self._previous_solution, "success", False):
            mesh = np.linspace(0.0, length_m, int(self.n_nodes))
            guesses.append((mesh, self._previous_solution.sol(mesh)))

        if select_lowest_energy_branch:
            for axis, sign in (("y", 0), ("y", 1), ("y", -1), ("z", 1), ("z", -1)):
                guesses.append(make_initial_guess(length_m, self.n_nodes, bend_axis=axis, bend_sign=sign))
        elif not guesses:
            guesses.append(self._straight_initial_guess(length_m))

        best_solution = None
        for mesh, initial_state in guesses:
            solution = solve_bvp(
                ode,
                lambda ya, yb: clamped_free_boundary_conditions(
                    ya,
                    yb,
                    base_position=self.base_position,
                    base_quaternion=self.base_quaternion,
                ),
                mesh,
                initial_state,
                tol=float(self.tolerance),
                max_nodes=int(self.max_nodes),
            )
            if solution.success:
                best_solution = solution
                break
            if best_solution is None:
                best_solution = solution

        if best_solution is None:
            raise RuntimeError("No BVP solve was attempted")
        if not best_solution.success:
            raise RuntimeError(f"Cosserat BVP failed: {best_solution.message}")
        self._previous_solution = best_solution
        return best_solution

    def forward(
        self,
        *,
        length_m: float,
        source_position: np.ndarray,
        source_quaternion: np.ndarray,
        source_dipole_body: np.ndarray,
        output_points: int = 300,
    ) -> dict[str, object]:
        """Solve from a source pose and return tip and distributed outputs."""
        ext = load_external_dependencies()
        source_dipole = dipole_from_pose(source_quaternion, source_dipole_body)
        solution = self.solve(
            length_m=length_m,
            source_position=source_position,
            source_dipole=source_dipole,
        )
        arclength = np.linspace(0.0, float(length_m), int(output_points))
        states = solution.sol(arclength)
        positions = states[0:3, :]
        quaternions = ext.quat_normalize(states[3:7, :])
        force, torque, field, beam_dipole, segment_s = (
            ext.magnetic_wrench_density_cosserat_profile_segments(
                positions,
                quaternions,
                arclength,
                source_dipole,
                np.asarray(source_position, dtype=float).reshape(3),
                self.magnetisation_profile,
                self.magnetic_parameter,
                r_min=1e-6,
            )
        )
        return {
            "solution": solution,
            "source_dipole": source_dipole,
            "tip_position": positions[:, -1],
            "tip_quaternion": quaternions[:, -1],
            "net_force": np.trapezoid(force, segment_s, axis=1),
            "net_torque": np.trapezoid(torque, segment_s, axis=1),
            "profiles": {
                "arclength": arclength,
                "segment_arclength": segment_s,
                "position": positions,
                "quaternion": quaternions,
                "force_density": force,
                "torque_density": torque,
                "field": field,
                "beam_dipole_world": beam_dipole,
            },
        }
