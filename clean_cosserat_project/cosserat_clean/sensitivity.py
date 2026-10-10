"""Shooting-based tip sensitivities around a nominal BVP solution."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .external import load_external_dependencies
from .jacobians import source_pose_control_jacobian, state_jacobian
from .mechanics import effective_lengths
from .quaternions import dipole_from_pose, source_dipole_rotation_jacobian
from .rod import cosserat_rhs_single


def _length_dependent_magnetisation(
    length_m: float,
    *,
    full_tip_length_m: float,
    minimum_model_length_m: float,
):
    ext = load_external_dependencies()
    model_length, wire_length, tip_length = effective_lengths(
        length_m,
        full_tip_length_m=full_tip_length_m,
        minimum_model_length_m=minimum_model_length_m,
    )
    profile = ext.make_m_local_fun_wire_tip(
        wire_length,
        len_tip=tip_length,
        mode="axial",
        alpha_end=0.0,
        eps=1e-3,
    )
    return model_length, wire_length, tip_length, profile


def _normalized_rhs(
    xi: float,
    state: np.ndarray,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_dipole: np.ndarray,
    inverse_stiffness,
    magnetisation_profile,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None,
) -> np.ndarray:
    arclength = float(length_m) * float(xi)
    derivative, _ = cosserat_rhs_single(
        arclength,
        state,
        source_dipole=source_dipole,
        source_position=source_position,
        inverse_stiffness=inverse_stiffness,
        magnetisation_profile=magnetisation_profile,
        magnetic_parameter=magnetic_parameter,
        wire_length=wire_length,
        reference_strain=reference_strain,
    )
    return float(length_m) * derivative


def _normalized_length_column_finite_difference(
    xi: float,
    state: np.ndarray,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_dipole: np.ndarray,
    inverse_stiffness,
    magnetic_parameter: float,
    reference_strain: np.ndarray | None,
    full_tip_length_m: float,
    minimum_model_length_m: float,
    epsilon_length_m: float,
) -> np.ndarray:
    length_plus = float(length_m) + epsilon_length_m
    length_minus = max(float(length_m) - epsilon_length_m, 1e-8)
    _, wire_plus, _, profile_plus = _length_dependent_magnetisation(
        length_plus,
        full_tip_length_m=full_tip_length_m,
        minimum_model_length_m=minimum_model_length_m,
    )
    _, wire_minus, _, profile_minus = _length_dependent_magnetisation(
        length_minus,
        full_tip_length_m=full_tip_length_m,
        minimum_model_length_m=minimum_model_length_m,
    )
    rhs_plus = _normalized_rhs(
        xi,
        state,
        length_m=length_plus,
        source_position=source_position,
        source_dipole=source_dipole,
        inverse_stiffness=inverse_stiffness,
        magnetisation_profile=profile_plus,
        magnetic_parameter=magnetic_parameter,
        wire_length=wire_plus,
        reference_strain=reference_strain,
    )
    rhs_minus = _normalized_rhs(
        xi,
        state,
        length_m=length_minus,
        source_position=source_position,
        source_dipole=source_dipole,
        inverse_stiffness=inverse_stiffness,
        magnetisation_profile=profile_minus,
        magnetic_parameter=magnetic_parameter,
        wire_length=wire_minus,
        reference_strain=reference_strain,
    )
    return (rhs_plus - rhs_minus) / (length_plus - length_minus)


def solve_tip_sensitivity_source_dipole_and_length(
    nominal_solution,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_dipole: np.ndarray,
    source_dipole_rotation_jacobian: np.ndarray | None,
    inverse_stiffness,
    magnetic_parameter: float,
    reference_strain: np.ndarray | None = None,
    full_tip_length_m: float = 0.04,
    minimum_model_length_m: float = 0.01,
    evaluation_points: int = 200,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-8,
    method: str = "DOP853",
    epsilon_length_m: float = 1e-5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return d(tip position)/d[source translation, rotation, length], shape (3, 7)."""
    length_m = float(length_m)
    source_position = np.asarray(source_position, dtype=float).reshape(3)
    source_dipole = np.asarray(source_dipole, dtype=float).reshape(3)
    if source_dipole_rotation_jacobian is None:
        from .quaternions import skew

        source_dipole_rotation_jacobian = -skew(source_dipole)
    source_dipole_rotation_jacobian = np.asarray(
        source_dipole_rotation_jacobian, dtype=float
    ).reshape(3, 3)

    _, wire_length, _, magnetisation_profile = _length_dependent_magnetisation(
        length_m,
        full_tip_length_m=full_tip_length_m,
        minimum_model_length_m=minimum_model_length_m,
    )

    state_count = 13
    control_count = 7
    shooting_unknown_count = 6
    particular_initial = np.zeros((state_count, control_count), dtype=float)
    homogeneous_initial = np.zeros((state_count, shooting_unknown_count), dtype=float)
    homogeneous_initial[7:13, :] = np.eye(6)
    augmented_initial = np.concatenate(
        [particular_initial.ravel(), homogeneous_initial.ravel()]
    )

    def augmented_ode(xi: float, augmented_flat: np.ndarray) -> np.ndarray:
        particular = augmented_flat[: state_count * control_count].reshape(
            state_count, control_count
        )
        homogeneous = augmented_flat[state_count * control_count :].reshape(
            state_count, shooting_unknown_count
        )
        arclength = length_m * float(xi)
        nominal_state = nominal_solution.sol(np.array([arclength], dtype=float))[:, 0]
        local_state_jacobian = state_jacobian(
            arclength,
            nominal_state,
            source_dipole=source_dipole,
            source_position=source_position,
            inverse_stiffness=inverse_stiffness,
            magnetisation_profile=magnetisation_profile,
            magnetic_parameter=magnetic_parameter,
            wire_length=wire_length,
            reference_strain=reference_strain,
        )
        pose_control_jacobian = source_pose_control_jacobian(
            arclength,
            nominal_state,
            source_dipole=source_dipole,
            source_position=source_position,
            source_dipole_rotation_jacobian=source_dipole_rotation_jacobian,
            inverse_stiffness=inverse_stiffness,
            magnetisation_profile=magnetisation_profile,
            magnetic_parameter=magnetic_parameter,
            wire_length=wire_length,
            reference_strain=reference_strain,
        )
        state_xi_jacobian = length_m * local_state_jacobian
        control_xi_jacobian = np.zeros((13, 7), dtype=float)
        control_xi_jacobian[:, 0:6] = length_m * pose_control_jacobian
        control_xi_jacobian[:, 6] = _normalized_length_column_finite_difference(
            xi,
            nominal_state,
            length_m=length_m,
            source_position=source_position,
            source_dipole=source_dipole,
            inverse_stiffness=inverse_stiffness,
            magnetic_parameter=magnetic_parameter,
            reference_strain=reference_strain,
            full_tip_length_m=full_tip_length_m,
            minimum_model_length_m=minimum_model_length_m,
            epsilon_length_m=epsilon_length_m,
        )
        particular_derivative = state_xi_jacobian @ particular + control_xi_jacobian
        homogeneous_derivative = state_xi_jacobian @ homogeneous 
        return np.concatenate([particular_derivative.ravel(), homogeneous_derivative.ravel()])

    integration = solve_ivp(
        augmented_ode,
        t_span=(0.0, 1.0),
        y0=augmented_initial,
        method=method,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        dense_output=True,
        t_eval=np.linspace(0.0, 1.0, int(evaluation_points)),
    )
    if not integration.success:
        raise RuntimeError(f"Shooting sensitivity failed: {integration.message}")

    terminal = integration.y[:, -1]
    particular_terminal = terminal[: state_count * control_count].reshape(
        state_count, control_count
    )
    homogeneous_terminal = terminal[state_count * control_count :].reshape(
        state_count, shooting_unknown_count
    )
    boundary_particular = particular_terminal[7:13, :]
    boundary_homogeneous = homogeneous_terminal[7:13, :]
    shooting_correction = np.linalg.solve(boundary_homogeneous, -boundary_particular)
    corrected_terminal = particular_terminal + homogeneous_terminal @ shooting_correction
    return corrected_terminal[0:3, :], {
        "integration": integration,
        "particular_terminal": particular_terminal,
        "homogeneous_terminal": homogeneous_terminal,
        "shooting_correction": shooting_correction,
        "tip_force_moment_residual": corrected_terminal[7:13, :],
        "terminal_condition_number": float(np.linalg.cond(boundary_homogeneous)),
    }


def solve_tip_sensitivity_source_pose_and_length(
    nominal_solution,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_quaternion: np.ndarray,
    source_dipole_body: np.ndarray,
    inverse_stiffness,
    magnetic_parameter: float,
    rotation_convention: str = "world",
    **kwargs,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pose-based wrapper for the seven-column shooting sensitivity."""
    source_dipole = dipole_from_pose(source_quaternion, source_dipole_body)
    dipole_rotation_jacobian = source_dipole_rotation_jacobian(
        source_quaternion,
        source_dipole_body,
        convention=rotation_convention,
    )
    return solve_tip_sensitivity_source_dipole_and_length(
        nominal_solution,
        length_m=length_m,
        source_position=source_position,
        source_dipole=source_dipole,
        source_dipole_rotation_jacobian=dipole_rotation_jacobian,
        inverse_stiffness=inverse_stiffness,
        magnetic_parameter=magnetic_parameter,
        **kwargs,
    )
