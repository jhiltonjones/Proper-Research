"""Lazy access to the original project's local/private dependencies.

Keeping these imports in one module makes the rest of the package importable and
testable even when the research repositories are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


class MissingExternalDependency(RuntimeError):
    """Raised when the original research packages are not installed."""


@dataclass(frozen=True)
class ExternalDependencies:
    magnetic_wrench_density_cosserat_point: Callable[..., Any]
    magnetic_wrench_density_cosserat_profile_segments: Callable[..., Any]
    make_m_local_fun_wire_tip: Callable[..., Any]
    quat_derivative_body: Callable[..., Any]
    quat_normalize: Callable[..., Any]
    quat_to_rot: Callable[..., Any]
    quat_to_R: Callable[..., Any]
    T_to_p_quat_wxyz: Callable[..., Any]
    ur_pose6_to_T: Callable[..., Any]
    default_magnet_params: Callable[..., Any]
    default_beam_params: Callable[..., Any]


@lru_cache(maxsize=1)
def load_external_dependencies() -> ExternalDependencies:
    """Import and return all local/private functions used by the simulation."""
    try:
        from beam_direction_magnetisation.magnetism.magnetic_methods import (
            magnetic_wrench_density_cosserat_point,
            magnetic_wrench_density_cosserat_profile_segments,
        )
        from beam_direction_magnetisation.magnetism.beam_geometry import (
            make_m_local_fun_wire_tip,
        )
        from beam_direction_magnetisation.quarternions.quarternions_functions import (
            T_to_p_quat_wxyz,
            quat_derivative_body,
            quat_normalize,
            quat_to_R,
            quat_to_rot,
        )
        from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
        from proper_research.parameters import default_beam_params, default_magnet_params
    except ModuleNotFoundError as exc:
        missing = exc.name or "an unknown package"
        raise MissingExternalDependency(
            "The physical simulation requires the original local research packages. "
            f"Python could not import '{missing}'. Install both "
            "'beam_direction_magnetisation' and 'proper_research' in this environment."
        ) from exc

    return ExternalDependencies(
        magnetic_wrench_density_cosserat_point=magnetic_wrench_density_cosserat_point,
        magnetic_wrench_density_cosserat_profile_segments=(
            magnetic_wrench_density_cosserat_profile_segments
        ),
        make_m_local_fun_wire_tip=make_m_local_fun_wire_tip,
        quat_derivative_body=quat_derivative_body,
        quat_normalize=quat_normalize,
        quat_to_rot=quat_to_rot,
        quat_to_R=quat_to_R,
        T_to_p_quat_wxyz=T_to_p_quat_wxyz,
        ur_pose6_to_T=ur_pose6_to_T,
        default_magnet_params=default_magnet_params,
        default_beam_params=default_beam_params,
    )


def dependency_status() -> dict[str, str]:
    """Return a human-readable dependency report without raising."""
    status: dict[str, str] = {}
    for package in (
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "beam_direction_magnetisation",
        "proper_research",
    ):
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "installed")
            status[package] = str(version)
        except ModuleNotFoundError:
            status[package] = "MISSING"
    return status
