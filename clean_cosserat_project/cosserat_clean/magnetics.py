"""Point-dipole magnetic field and analytic derivative helpers."""

from __future__ import annotations

import numpy as np

from .quaternions import skew

MU0 = 4.0 * np.pi * 1e-7


def dipole_field_from_source(
    points_m: np.ndarray,
    source_position_m: np.ndarray,
    source_dipole_am2: np.ndarray,
    *,
    minimum_radius_m: float = 1e-6,
) -> np.ndarray:
    """Evaluate a point-dipole field at one or more world-frame points."""
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    source = np.asarray(source_position_m, dtype=float).reshape(3)
    dipole = np.asarray(source_dipole_am2, dtype=float).reshape(3)
    displacement = points - source[None, :]
    radius_sq = np.sum(displacement * displacement, axis=1)
    radius = np.sqrt(np.maximum(radius_sq, minimum_radius_m**2))
    direction = displacement / radius[:, None]
    projection = direction @ dipole
    inverse_radius_cubed = radius**-3
    return 1e-7 * inverse_radius_cubed[:, None] * (
        3.0 * projection[:, None] * direction - dipole[None, :]
    )


def dipole_field_jacobian_wrt_source_position(
    point_m: np.ndarray,
    source_position_m: np.ndarray,
    source_dipole_am2: np.ndarray,
    *,
    mu0: float = MU0,
) -> np.ndarray:
    """Return dB/d(source position), shape (3, 3)."""
    coefficient = mu0 / (4.0 * np.pi)
    displacement = np.asarray(point_m, dtype=float).reshape(3) - np.asarray(
        source_position_m, dtype=float
    ).reshape(3)
    radius = float(np.linalg.norm(displacement) + 1e-12)
    radius_sq = radius * radius
    radius_5 = radius_sq * radius_sq * radius
    radius_7 = radius_5 * radius_sq
    dipole = np.asarray(source_dipole_am2, dtype=float).reshape(3)
    projection = float(np.dot(displacement, dipole))
    identity = np.eye(3)
    derivative_wrt_displacement = coefficient * (
        3.0 * projection * identity / radius_5
        + 3.0 * np.outer(displacement, dipole) / radius_5
        + 3.0 * np.outer(dipole, displacement) / radius_5
        - 15.0 * projection * np.outer(displacement, displacement) / radius_7
    )
    return -derivative_wrt_displacement


def dipole_field_jacobian_wrt_source_dipole(
    point_m: np.ndarray,
    source_position_m: np.ndarray,
    *,
    mu0: float = MU0,
) -> np.ndarray:
    """Return dB/d(source dipole), shape (3, 3)."""
    coefficient = mu0 / (4.0 * np.pi)
    displacement = np.asarray(point_m, dtype=float).reshape(3) - np.asarray(
        source_position_m, dtype=float
    ).reshape(3)
    radius_sq = float(np.dot(displacement, displacement) + 1e-24)
    radius = np.sqrt(radius_sq)
    radius_3 = radius_sq * radius
    radius_5 = radius_3 * radius_sq
    return coefficient * (
        3.0 * np.outer(displacement, displacement) / radius_5 - np.eye(3) / radius_3
    )


def force_jacobian_wrt_source_position(
    point_m: np.ndarray,
    source_position_m: np.ndarray,
    source_dipole_am2: np.ndarray,
    beam_dipole_am2: np.ndarray,
    *,
    mu0: float = MU0,
) -> np.ndarray:
    """Return d(force density)/d(source position), shape (3, 3)."""
    coefficient = mu0 / (4.0 * np.pi)
    displacement = np.asarray(point_m, dtype=float).reshape(3) - np.asarray(
        source_position_m, dtype=float
    ).reshape(3)
    beam_dipole = np.asarray(beam_dipole_am2, dtype=float).reshape(3)
    source_dipole = np.asarray(source_dipole_am2, dtype=float).reshape(3)

    radius_sq = float(np.dot(displacement, displacement) + 1e-24)
    radius = np.sqrt(radius_sq)
    radius_4 = radius_sq * radius_sq
    radius_5 = radius_4 * radius
    radius_7 = radius_5 * radius_sq

    a = float(np.dot(beam_dipole, displacement))
    b = float(np.dot(source_dipole, displacement))
    dipole_dot = float(np.dot(beam_dipole, source_dipole))
    inner = (
        a * source_dipole
        + b * beam_dipole
        + dipole_dot * displacement
        - 5.0 * a * b / radius_sq * displacement
    )
    identity = np.eye(3)
    inner_jacobian = (
        np.outer(source_dipole, beam_dipole)
        + np.outer(beam_dipole, source_dipole)
        + dipole_dot * identity
        - 5.0
        * (
            (a * b / radius_sq) * identity
            + np.outer(displacement, b * beam_dipole + a * source_dipole) / radius_sq
            - 2.0 * a * b * np.outer(displacement, displacement) / radius_4
        )
    )
    force_jacobian_wrt_displacement = 3.0 * coefficient * (
        inner_jacobian / radius_5 - 5.0 * np.outer(inner, displacement) / radius_7
    )
    return -force_jacobian_wrt_displacement


def force_jacobian_wrt_source_dipole(
    point_m: np.ndarray,
    source_position_m: np.ndarray,
    source_dipole_am2: np.ndarray,
    beam_dipole_am2: np.ndarray,
    *,
    mu0: float = MU0,
) -> np.ndarray:
    """Return d(force density)/d(source dipole), shape (3, 3)."""
    del source_dipole_am2  # derivative is linear and does not depend on its nominal value
    coefficient = mu0 / (4.0 * np.pi)
    displacement = np.asarray(point_m, dtype=float).reshape(3) - np.asarray(
        source_position_m, dtype=float
    ).reshape(3)
    beam_dipole = np.asarray(beam_dipole_am2, dtype=float).reshape(3)
    radius_sq = float(np.dot(displacement, displacement) + 1e-24)
    radius = np.sqrt(radius_sq)
    radius_5 = radius_sq * radius_sq * radius
    projection = float(np.dot(beam_dipole, displacement))
    return 3.0 * coefficient / radius_5 * (
        projection * np.eye(3)
        + np.outer(beam_dipole, displacement)
        + np.outer(displacement, beam_dipole)
        - 5.0 * projection / radius_sq * np.outer(displacement, displacement)
    )


def force_jacobian_wrt_beam_dipole(
    point_m: np.ndarray,
    source_position_m: np.ndarray,
    source_dipole_am2: np.ndarray,
    *,
    mu0: float = MU0,
) -> np.ndarray:
    """Return d(force density)/d(beam dipole), shape (3, 3)."""
    coefficient = mu0 / (4.0 * np.pi)
    displacement = np.asarray(point_m, dtype=float).reshape(3) - np.asarray(
        source_position_m, dtype=float
    ).reshape(3)
    source_dipole = np.asarray(source_dipole_am2, dtype=float).reshape(3)
    radius = float(np.linalg.norm(displacement) + 1e-12)
    radius_sq = radius * radius
    radius_5 = radius_sq * radius_sq * radius
    source_dot_r = float(np.dot(source_dipole, displacement))
    return 3.0 * coefficient / radius_5 * (
        np.outer(source_dipole, displacement)
        + source_dot_r * np.eye(3)
        + np.outer(displacement, source_dipole)
        - 5.0 * source_dot_r / radius_sq * np.outer(displacement, displacement)
    )


def torque_jacobian_from_field_jacobian(
    beam_dipole_am2: np.ndarray,
    field_jacobian: np.ndarray,
) -> np.ndarray:
    """For tau=m x B, return d(tau)/d(control) from dB/d(control)."""
    return skew(beam_dipole_am2) @ np.asarray(field_jacobian, dtype=float)
