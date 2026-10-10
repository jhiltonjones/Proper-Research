"""Beam geometry, stiffness, and insertion-length helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SectionStiffness:
    radius_m: float
    youngs_modulus_pa: float
    poisson_ratio: float
    area_m2: float
    second_moment_m4: float
    polar_moment_m4: float
    shear_modulus_pa: float
    axial_stiffness_n: float
    bending_stiffness_nm2: float
    torsional_stiffness_nm2: float


def rod_section_stiffness(radius_m: float, youngs_modulus_pa: float, poisson_ratio: float) -> SectionStiffness:
    """Calculate circular-section stiffness properties."""
    if radius_m <= 0.0 or youngs_modulus_pa <= 0.0:
        raise ValueError("radius and Young's modulus must be positive")
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("Poisson ratio must lie between -1 and 0.5")

    area = np.pi * radius_m**2
    second_moment = np.pi * radius_m**4 / 4.0
    polar_moment = 0.5 * np.pi * radius_m**4
    shear_modulus = youngs_modulus_pa / (2.0 * (1.0 + poisson_ratio))
    return SectionStiffness(
        radius_m=radius_m,
        youngs_modulus_pa=youngs_modulus_pa,
        poisson_ratio=poisson_ratio,
        area_m2=area,
        second_moment_m4=second_moment,
        polar_moment_m4=polar_moment,
        shear_modulus_pa=shear_modulus,
        axial_stiffness_n=youngs_modulus_pa * area,
        bending_stiffness_nm2=youngs_modulus_pa * second_moment,
        torsional_stiffness_nm2=shear_modulus * polar_moment,
    )


def make_inverse_bending_torsion_profile(
    *,
    wire_bending_stiffness: float,
    tip_bending_stiffness: float,
    wire_torsional_stiffness: float,
    tip_torsional_stiffness: float,
    bend_softening: float = 1.0,
    torsion_softening: float = 1.0,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return K_bt^-1(s), with wire properties before the wire/tip transition."""
    values = (
        wire_bending_stiffness,
        tip_bending_stiffness,
        wire_torsional_stiffness,
        tip_torsional_stiffness,
        bend_softening,
        torsion_softening,
    )
    if any(value <= 0.0 for value in values):
        raise ValueError("all stiffness and softening values must be positive")

    def profile(s: np.ndarray, wire_length_m: float) -> np.ndarray:
        s = np.atleast_1d(np.asarray(s, dtype=float))
        is_tip = s >= float(wire_length_m)
        bending = np.where(is_tip, tip_bending_stiffness, wire_bending_stiffness)
        torsion = np.where(is_tip, tip_torsional_stiffness, wire_torsional_stiffness)
        inverse = np.zeros((3, 3, s.size), dtype=float)
        inverse[0, 0, :] = torsion_softening / torsion
        inverse[1, 1, :] = bend_softening / bending
        inverse[2, 2, :] = bend_softening / bending
        return inverse

    return profile


def effective_lengths(
    inserted_length_m: float,
    *,
    full_tip_length_m: float = 0.04,
    minimum_model_length_m: float = 0.01,
) -> tuple[float, float, float]:
    """Return model length, inserted wire length, and magnetised tip length."""
    inserted = float(inserted_length_m)
    if inserted < 0.0:
        raise ValueError("inserted length cannot be negative")
    tip_length = min(inserted, full_tip_length_m)
    wire_length = max(inserted - full_tip_length_m, 0.0)
    model_length = max(inserted, minimum_model_length_m)
    tip_length = min(tip_length, model_length)
    return model_length, wire_length, tip_length


def make_initial_guess(
    length_m: float,
    n_nodes: int,
    *,
    bend_axis: str = "y",
    bend_sign: int = 0,
    seed_amplitude_m: float = 5e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a straight or weakly bent initial guess for the BVP solver."""
    s = np.linspace(0.0, float(length_m), int(n_nodes))
    xi = s / max(float(length_m), 1e-12)
    state = np.zeros((13, s.size), dtype=float)
    state[0, :] = -s
    if bend_axis == "y":
        state[1, :] = bend_sign * seed_amplitude_m * np.sin(np.pi * xi)
    elif bend_axis == "z":
        state[2, :] = bend_sign * seed_amplitude_m * np.sin(np.pi * xi)
    else:
        raise ValueError("bend_axis must be 'y' or 'z'")
    state[3, :] = 1.0
    return s, state
