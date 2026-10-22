"""Post-processing for rod solutions and sensitivity matrices."""

from __future__ import annotations

import numpy as np

from .external import load_external_dependencies


def tip_bending_angles_signed(
    solution,
    *,
    length_m: float,
    tangent_reference: np.ndarray,
    local_y_reference: np.ndarray,
    local_z_reference: np.ndarray,
    material_axis: np.ndarray = np.array([-1.0, 0.0, 0.0]),
) -> tuple[float, float, float]:
    """Return signed y/z bend angles and total bend angle in radians."""
    ext = load_external_dependencies()
    terminal_state = solution.sol(np.array([float(length_m)], dtype=float))[:, 0]
    terminal_quaternion = ext.quat_normalize(terminal_state[3:7])
    terminal_tangent = ext.quat_to_R(terminal_quaternion) @ np.asarray(
        material_axis, dtype=float
    ).reshape(3)
    terminal_tangent /= np.linalg.norm(terminal_tangent) + 1e-12

    tangent_reference = _unit(tangent_reference)
    local_y_reference = _unit(local_y_reference)
    local_z_reference = _unit(local_z_reference)
    forward = float(np.dot(terminal_tangent, tangent_reference))
    component_y = float(np.dot(terminal_tangent, local_y_reference))
    component_z = float(np.dot(terminal_tangent, local_z_reference))
    angle_y = float(np.arctan2(component_y, forward))
    angle_z = float(np.arctan2(component_z, forward))
    total = float(np.arccos(np.clip(forward, -1.0, 1.0)))
    return angle_y, angle_z, total


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    return vector / (np.linalg.norm(vector) + 1e-12)


def jacobian_metrics(jacobian_pose_length: np.ndarray) -> dict[str, float]:
    """Compute norms, singular values, conditioning, and manipulability."""
    jacobian = np.asarray(jacobian_pose_length, dtype=float)
    if jacobian.shape != (3, 7):
        raise ValueError(f"expected a (3, 7) Jacobian, got {jacobian.shape}")
    pose_3d = jacobian[:, 0:6]
    singular_3d = np.linalg.svd(pose_3d, compute_uv=False)
    pose_xy = pose_3d[0:2, :]
    singular_xy = np.linalg.svd(pose_xy, compute_uv=False)
    return {
        "jacobian_norm": float(np.linalg.norm(jacobian)),
        "translation_norm": float(np.linalg.norm(jacobian[:, 0:3])),
        "rotation_norm": float(np.linalg.norm(jacobian[:, 3:6])),
        "length_norm": float(np.linalg.norm(jacobian[:, 6])),
        "condition_3d": float(singular_3d[0] / max(singular_3d[-1], 1e-12)),
        "manipulability_3d": float(np.prod(singular_3d)),
        "sigma1_3d": float(singular_3d[0]),
        "sigma2_3d": float(singular_3d[1]),
        "sigma3_3d": float(singular_3d[2]),
        "condition_xy": float(singular_xy[0] / max(singular_xy[-1], 1e-12)),
        "manipulability_xy": float(np.prod(singular_xy)),
        "sigma1_xy": float(singular_xy[0]),
        "sigma2_xy": float(singular_xy[1]),
    }


def magnetic_net_force_torque(
    solution,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_dipole: np.ndarray,
    magnetisation_profile,
    magnetic_parameter: float,
    evaluation_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate magnetic force and torque density along a solved rod."""
    ext = load_external_dependencies()
    arclength = np.linspace(0.0, float(length_m), int(evaluation_points))
    state = solution.sol(arclength)
    position = state[0:3, :]
    quaternion = ext.quat_normalize(state[3:7, :])
    force, torque, _, _, segment_s = ext.magnetic_wrench_density_cosserat_profile_segments(
        position,
        quaternion,
        arclength,
        np.asarray(source_dipole, dtype=float).reshape(3),
        np.asarray(source_position, dtype=float).reshape(3),
        magnetisation_profile,
        magnetic_parameter,
        r_min=1e-6,
    )
    return (
        np.trapezoid(np.asarray(force, dtype=float), segment_s, axis=1),
        np.trapezoid(np.asarray(torque, dtype=float), segment_s, axis=1),
    )


def compute_total_energy(
    solution,
    *,
    length_m: float,
    source_position: np.ndarray,
    source_dipole: np.ndarray,
    inverse_stiffness,
    magnetisation_profile,
    magnetic_parameter: float,
    wire_length: float,
    reference_strain: np.ndarray | None = None,
    gravity_density: np.ndarray | None = None,
    contact_energy_density=None,
    evaluation_points: int = 400,
) -> tuple[float, dict[str, float]]:
    """Compute elastic, magnetic, gravity, and optional contact energy."""
    ext = load_external_dependencies()
    arclength = np.linspace(0.0, float(length_m), int(evaluation_points))
    state = solution.sol(arclength)
    position = state[0:3, :]
    quaternion = ext.quat_normalize(state[3:7, :])
    internal_moment_world = state[10:13, :]
    rotations = ext.quat_to_rot(quaternion)
    internal_moment_body = np.einsum(
        "nij,jn->in", np.transpose(rotations, (0, 2, 1)), internal_moment_world
    )
    stiffness_inverse = inverse_stiffness(arclength, wire_length)
    reference = (
        np.zeros(3, dtype=float)
        if reference_strain is None
        else np.asarray(reference_strain, dtype=float).reshape(3)
    )
    strain = np.einsum("ijn,jn->in", stiffness_inverse, internal_moment_body) + reference[:, None]
    elastic_density = 0.5 * np.sum((strain - reference[:, None]) * internal_moment_body, axis=0)
    elastic = float(np.trapezoid(elastic_density, arclength))

    _, _, field_mid, beam_dipole_mid, _ = ext.magnetic_wrench_density_cosserat_profile_segments(
        position,
        quaternion,
        arclength,
        np.asarray(source_dipole, dtype=float).reshape(3),
        np.asarray(source_position, dtype=float).reshape(3),
        magnetisation_profile,
        magnetic_parameter,
        r_min=1e-6,
    )
    magnetic_density_mid = -np.sum(
        np.asarray(beam_dipole_mid, dtype=float) * np.asarray(field_mid, dtype=float), axis=0
    )
    magnetic = float(np.sum(magnetic_density_mid * np.diff(arclength)))

    gravity = 0.0
    if gravity_density is not None:
        gravity_vector = np.asarray(gravity_density, dtype=float).reshape(3)
        gravity = float(-np.trapezoid(np.sum(gravity_vector[:, None] * position, axis=0), arclength))

    contact = 0.0
    if contact_energy_density is not None:
        density = np.asarray(contact_energy_density(position, quaternion, arclength), dtype=float)
        contact = float(np.trapezoid(density, arclength))

    total = elastic + magnetic + gravity + contact
    return total, {
        "elastic": elastic,
        "magnetic": magnetic,
        "gravity": gravity,
        "contact": contact,
        "total": total,
    }
