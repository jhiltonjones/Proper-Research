"""Analytic state and source-pose Jacobians for the Cosserat ODE."""

from __future__ import annotations

import numpy as np

from .external import load_external_dependencies
from .magnetics import (
    dipole_field_jacobian_wrt_source_dipole,
    dipole_field_jacobian_wrt_source_position,
    force_jacobian_wrt_beam_dipole,
    force_jacobian_wrt_source_dipole,
    force_jacobian_wrt_source_position,
    torque_jacobian_from_field_jacobian,
)
from .quaternions import dnormalize_dq, skew
from .rod import cosserat_rhs_single


def quaternion_rate_jacobian_wrt_strain(quaternion: np.ndarray) -> np.ndarray:
    """Return d(q_s)/d(strain), shape (4, 3)."""
    w, x, y, z = np.asarray(quaternion, dtype=float).reshape(4)
    return 0.5 * np.array(
        [
            [-x, -y, -z],
            [w, -z, y],
            [z, w, -x],
            [-y, x, w],
        ],
        dtype=float,
    )


def quaternion_rate_jacobian_wrt_world_moment(
    quaternion: np.ndarray,
    inverse_stiffness: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    return quaternion_rate_jacobian_wrt_strain(quaternion) @ inverse_stiffness @ rotation.T


def rotated_vector_jacobian_wrt_normalized_quaternion(
    quaternion: np.ndarray,
    vector_body: np.ndarray,
) -> np.ndarray:
    """Return d(R(q) v)/dq for normalized q=[w,x,y,z]."""
    ext = load_external_dependencies()
    w, x, y, z = ext.quat_normalize(np.asarray(quaternion, dtype=float).reshape(4))
    vx, vy, vz = np.asarray(vector_body, dtype=float).reshape(3)
    derivative_w = np.array(
        [-2.0 * z * vy + 2.0 * y * vz, 2.0 * z * vx - 2.0 * x * vz, -2.0 * y * vx + 2.0 * x * vy]
    )
    derivative_x = np.array(
        [2.0 * y * vy + 2.0 * z * vz, 2.0 * y * vx - 4.0 * x * vy - 2.0 * w * vz, 2.0 * z * vx + 2.0 * w * vy - 4.0 * x * vz]
    )
    derivative_y = np.array(
        [-4.0 * y * vx + 2.0 * x * vy + 2.0 * w * vz, 2.0 * x * vx + 2.0 * z * vz, -2.0 * w * vx + 2.0 * z * vy - 4.0 * y * vz]
    )
    derivative_z = np.array(
        [-4.0 * z * vx - 2.0 * w * vy + 2.0 * x * vz, 2.0 * w * vx - 4.0 * z * vy + 2.0 * y * vz, 2.0 * x * vx + 2.0 * y * vy]
    )
    return np.column_stack([derivative_w, derivative_x, derivative_y, derivative_z])


def rotated_vector_jacobian_wrt_raw_quaternion(
    quaternion_raw: np.ndarray,
    vector_body: np.ndarray,
) -> np.ndarray:
    ext = load_external_dependencies()
    quaternion_raw = np.asarray(quaternion_raw, dtype=float).reshape(4)
    quaternion = ext.quat_normalize(quaternion_raw)
    return rotated_vector_jacobian_wrt_normalized_quaternion(quaternion, vector_body) @ dnormalize_dq(quaternion_raw)


def inverse_rotated_vector_jacobian_wrt_normalized_quaternion(
    quaternion: np.ndarray,
    vector_world: np.ndarray,
) -> np.ndarray:
    """Return d(R(q)^T v)/dq for normalized q=[w,x,y,z]."""
    ext = load_external_dependencies()
    w, x, y, z = ext.quat_normalize(np.asarray(quaternion, dtype=float).reshape(4))
    vx, vy, vz = np.asarray(vector_world, dtype=float).reshape(3)
    derivative_w = np.array([2.0 * z * vy - 2.0 * y * vz, -2.0 * z * vx + 2.0 * x * vz, 2.0 * y * vx - 2.0 * x * vy])
    derivative_x = np.array([2.0 * y * vy + 2.0 * z * vz, 2.0 * y * vx - 4.0 * x * vy + 2.0 * w * vz, 2.0 * z * vx - 2.0 * w * vy - 4.0 * x * vz])
    derivative_y = np.array([-4.0 * y * vx + 2.0 * x * vy - 2.0 * w * vz, 2.0 * x * vx + 2.0 * z * vz, 2.0 * w * vx + 2.0 * z * vy - 4.0 * y * vz])
    derivative_z = np.array([-4.0 * z * vx + 2.0 * w * vy + 2.0 * x * vz, -2.0 * w * vx - 4.0 * z * vy + 2.0 * y * vz, 2.0 * x * vx + 2.0 * y * vy])
    return np.column_stack([derivative_w, derivative_x, derivative_y, derivative_z])


def position_rate_jacobian_wrt_normalized_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return d(R(q)[-1,0,0])/dq for normalized q."""
    ext = load_external_dependencies()
    w, x, y, z = ext.quat_normalize(np.asarray(quaternion, dtype=float).reshape(4))
    return np.array(
        [
            [0.0, 0.0, 4.0 * y, 4.0 * z],
            [-2.0 * z, -2.0 * y, -2.0 * x, -2.0 * w],
            [2.0 * y, -2.0 * z, 2.0 * w, -2.0 * x],
        ],
        dtype=float,
    )


def position_rate_jacobian_wrt_raw_quaternion(quaternion_raw: np.ndarray) -> np.ndarray:
    ext = load_external_dependencies()
    quaternion_raw = np.asarray(quaternion_raw, dtype=float).reshape(4)
    quaternion = ext.quat_normalize(quaternion_raw)
    return position_rate_jacobian_wrt_normalized_quaternion(quaternion) @ dnormalize_dq(quaternion_raw)


def quaternion_rate_direct_jacobian_wrt_normalized_quaternion(strain: np.ndarray) -> np.ndarray:
    ux, uy, uz = np.asarray(strain, dtype=float).reshape(3)
    return 0.5 * np.array(
        [
            [0.0, -ux, -uy, -uz],
            [ux, 0.0, uz, -uy],
            [uy, -uz, 0.0, ux],
            [uz, uy, -ux, 0.0],
        ],
        dtype=float,
    )


def quaternion_rate_jacobian_wrt_raw_quaternion(
    quaternion_raw: np.ndarray,
    strain: np.ndarray,
    inverse_stiffness: np.ndarray,
    internal_moment_world: np.ndarray,
) -> np.ndarray:
    ext = load_external_dependencies()
    quaternion_raw = np.asarray(quaternion_raw, dtype=float).reshape(4)
    quaternion = ext.quat_normalize(quaternion_raw)
    strain_jacobian = inverse_stiffness @ inverse_rotated_vector_jacobian_wrt_normalized_quaternion(
        quaternion, internal_moment_world
    )
    normalized_jacobian = quaternion_rate_direct_jacobian_wrt_normalized_quaternion(strain) + (
        quaternion_rate_jacobian_wrt_strain(quaternion) @ strain_jacobian
    )
    return normalized_jacobian @ dnormalize_dq(quaternion_raw)


def source_position_control_jacobian(
    s: float,
    state: np.ndarray,
    *,
    source_dipole: np.ndarray,
    source_position: np.ndarray,
    inverse_stiffness,
    magnetisation_profile,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
) -> np.ndarray:
    """Return d(RHS)/d(source position), shape (13, 3)."""
    ext = load_external_dependencies()
    state = np.asarray(state, dtype=float).reshape(13)
    position = state[0:3]
    quaternion = ext.quat_normalize(state[3:7])
    rotation = ext.quat_to_R(quaternion)
    local_dipole = magnetisation_profile(np.array([s], dtype=float), magnetic_parameter)[:, 0]
    beam_dipole_world = rotation @ local_dipole
    field_jacobian = dipole_field_jacobian_wrt_source_position(position, source_position, source_dipole)
    force_jacobian = force_jacobian_wrt_source_position(position, source_position, source_dipole, beam_dipole_world)
    torque_jacobian = torque_jacobian_from_field_jacobian(beam_dipole_world, field_jacobian)
    result = np.zeros((13, 3), dtype=float)
    result[7:10, :] = -force_jacobian
    result[10:13, :] = -torque_jacobian
    return result


def source_pose_control_jacobian(
    s: float,
    state: np.ndarray,
    *,
    source_dipole: np.ndarray,
    source_position: np.ndarray,
    source_dipole_rotation_jacobian: np.ndarray,
    inverse_stiffness,
    magnetisation_profile,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
) -> np.ndarray:
    """Return d(RHS)/d[source position, source rotation], shape (13, 6)."""
    del inverse_stiffness, reference_strain, wire_length
    ext = load_external_dependencies()
    state = np.asarray(state, dtype=float).reshape(13)
    position = state[0:3]
    quaternion = ext.quat_normalize(state[3:7])
    rotation = ext.quat_to_R(quaternion)
    local_dipole = magnetisation_profile(np.array([s], dtype=float), magnetic_parameter)[:, 0]
    beam_dipole_world = rotation @ local_dipole

    field_position_jacobian = dipole_field_jacobian_wrt_source_position(position, source_position, source_dipole)
    force_position_jacobian = force_jacobian_wrt_source_position(position, source_position, source_dipole, beam_dipole_world)
    torque_position_jacobian = torque_jacobian_from_field_jacobian(beam_dipole_world, field_position_jacobian)
    position_block = np.zeros((13, 3), dtype=float)
    position_block[7:10, :] = -force_position_jacobian
    position_block[10:13, :] = -torque_position_jacobian

    field_dipole_jacobian = dipole_field_jacobian_wrt_source_dipole(position, source_position)
    force_dipole_jacobian = force_jacobian_wrt_source_dipole(position, source_position, source_dipole, beam_dipole_world)
    torque_dipole_jacobian = torque_jacobian_from_field_jacobian(beam_dipole_world, field_dipole_jacobian)
    dipole_block = np.zeros((13, 3), dtype=float)
    dipole_block[7:10, :] = -force_dipole_jacobian
    dipole_block[10:13, :] = -torque_dipole_jacobian
    rotation_block = dipole_block @ np.asarray(source_dipole_rotation_jacobian, dtype=float).reshape(3, 3)
    return np.hstack([position_block, rotation_block])


def state_jacobian(
    s: float,
    state: np.ndarray,
    *,
    source_dipole: np.ndarray,
    source_position: np.ndarray,
    inverse_stiffness,
    magnetisation_profile,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
) -> np.ndarray:
    """Return the full analytic local state Jacobian d(RHS)/d(state)."""
    state = np.asarray(state, dtype=float).reshape(13)
    _, aux = cosserat_rhs_single(
        s,
        state,
        source_dipole=source_dipole,
        source_position=source_position,
        inverse_stiffness=inverse_stiffness,
        magnetisation_profile=magnetisation_profile,
        magnetic_parameter=magnetic_parameter,
        wire_length=wire_length,
        reference_strain=reference_strain,
    )
    position = state[0:3]
    quaternion_raw = state[3:7]
    internal_force = state[7:10]
    internal_moment_world = state[10:13]
    quaternion = aux["quaternion"]
    rotation = aux["rotation"]
    stiffness_inverse = aux["inverse_stiffness"]
    strain = aux["strain"]
    position_derivative = aux["position_derivative"]
    field = aux["field"]
    local_dipole = magnetisation_profile(np.array([s], dtype=float), magnetic_parameter)[:, 0]

    jacobian = np.zeros((13, 13), dtype=float)
    dp_dq = position_rate_jacobian_wrt_raw_quaternion(quaternion_raw)
    jacobian[0:3, 3:7] = dp_dq
    jacobian[3:7, 3:7] = quaternion_rate_jacobian_wrt_raw_quaternion(
        quaternion_raw, strain, stiffness_inverse, internal_moment_world
    )
    jacobian[3:7, 10:13] = quaternion_rate_jacobian_wrt_world_moment(
        quaternion, stiffness_inverse, rotation
    )

    position_control = source_position_control_jacobian(
        s,
        state,
        source_dipole=source_dipole,
        source_position=source_position,
        inverse_stiffness=inverse_stiffness,
        magnetisation_profile=magnetisation_profile,
        magnetic_parameter=magnetic_parameter,
        wire_length=wire_length,
        reference_strain=reference_strain,
    )
    # Translation invariance: d/d(point position) = -d/d(source position).
    jacobian[7:10, 0:3] = -position_control[7:10, :]
    jacobian[10:13, 0:3] = -position_control[10:13, :]

    beam_dipole_q_jacobian = rotated_vector_jacobian_wrt_raw_quaternion(quaternion_raw, local_dipole)
    jacobian[7:10, 3:7] = -force_jacobian_wrt_beam_dipole(
        position, source_position, source_dipole
    ) @ beam_dipole_q_jacobian

    torque_q_jacobian = np.zeros((3, 4), dtype=float)
    for column in range(4):
        torque_q_jacobian[:, column] = np.cross(beam_dipole_q_jacobian[:, column], field)
    kinematic_q_jacobian = np.zeros((3, 4), dtype=float)
    for column in range(4):
        kinematic_q_jacobian[:, column] = -np.cross(dp_dq[:, column], internal_force)
    jacobian[10:13, 3:7] = kinematic_q_jacobian - torque_q_jacobian
    jacobian[10:13, 7:10] = -skew(position_derivative)
    return jacobian


def state_jacobian_finite_difference(
    s: float,
    state: np.ndarray,
    *,
    epsilon: float = 1e-7,
    **rhs_kwargs,
) -> np.ndarray:
    """Reference finite-difference state Jacobian for validation."""
    state = np.asarray(state, dtype=float).reshape(13)
    baseline, _ = cosserat_rhs_single(s, state, **rhs_kwargs)
    result = np.zeros((13, 13), dtype=float)
    for column in range(13):
        perturbed = state.copy()
        perturbed[column] += epsilon
        value, _ = cosserat_rhs_single(s, perturbed, **rhs_kwargs)
        result[:, column] = (value - baseline) / epsilon
    return result
