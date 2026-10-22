from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from proper_research.simulation.magnetic_beam.contact import ContactParams
from proper_research.simulation.magnetic_beam.forward_model_optimized import (
    MagneticBeamForwardModelOptimized as MagneticBeamForwardModel,
)
from proper_research.simulation.simulations.model_factory_optimized import (
    build_forward_model,
)


@dataclass(frozen=True)
class HardwareModelBundleOptimized:
    """
    Nominal prediction models used by the hardware MPC.

    The real plant remains the robot, catheter and camera measurement. The
    `prediction_model` is the contact-aware nominal model used for rollouts.
    `jacobian_model` can be contact-aware or no-contact for the research
    comparison.
    """

    prediction_model: MagneticBeamForwardModel
    jacobian_model: MagneticBeamForwardModel
    contact_model: MagneticBeamForwardModel
    no_contact_model: MagneticBeamForwardModel
    lumen_C: np.ndarray
    lumen_R: np.ndarray
    p0_ur: np.ndarray
    q0_ur: np.ndarray
    sensitivity_workers: int
    jacobian_variant: str


# Compatibility name used by the old hardware script.
HardwareModelBundle = HardwareModelBundleOptimized


def _validate_lumen(
    lumen_C_robot_m: np.ndarray,
    lumen_R_robot_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lumen_C = np.asarray(lumen_C_robot_m, dtype=float)
    lumen_R = np.asarray(lumen_R_robot_m, dtype=float).reshape(-1)
    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(
            "lumen_C_robot_m must have shape (M, >=3), "
            f"got {lumen_C.shape}."
        )
    lumen_C = lumen_C[:, :3].copy()
    if lumen_C.shape[0] < 2:
        raise ValueError("The lumen must contain at least two centreline points.")
    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError(
            f"lumen_R_robot_m length {lumen_R.size} does not match "
            f"lumen_C_robot_m length {lumen_C.shape[0]}."
        )
    if not np.all(np.isfinite(lumen_C)):
        raise ValueError("lumen_C_robot_m contains non-finite values.")
    if not np.all(np.isfinite(lumen_R)) or np.any(lumen_R <= 0.0):
        raise ValueError("lumen_R_robot_m must contain finite positive radii.")
    return lumen_C, lumen_R.copy()


def build_hardware_model_bundle_optimized(
    *,
    p0_ur: np.ndarray,
    q0_ur: np.ndarray,
    Kinv_fun,
    m_body: np.ndarray,
    lumen_C_robot_m: np.ndarray,
    lumen_R_robot_m: np.ndarray,
    contact_params: ContactParams,
    jacobian_variant: str = "contact",
    sensitivity_workers: int = 1,
    result_detail: str = "contact",
    store_history: bool = False,
    store_vectors_in_info: bool = False,
    copy_cached_results: bool = False,
    m_local_factory=None,
) -> HardwareModelBundleOptimized:
    """Build optimized contact and no-contact models for hardware MPC."""
    variant = str(jacobian_variant).lower()
    if variant not in {"contact", "no_contact"}:
        raise ValueError(
            "jacobian_variant must be 'contact' or 'no_contact', "
            f"got {jacobian_variant!r}."
        )
    workers = int(sensitivity_workers)
    if workers < 1:
        raise ValueError("sensitivity_workers must be >= 1.")

    lumen_C, lumen_R = _validate_lumen(
        lumen_C_robot_m,
        lumen_R_robot_m,
    )
    common: dict[str, Any] = {
        "p0_ur": np.asarray(p0_ur, dtype=float).reshape(3),
        "q0_ur": np.asarray(q0_ur, dtype=float).reshape(4),
        "Kinv_fun": Kinv_fun,
        "m_body": np.asarray(m_body, dtype=float).reshape(3),
        "lumen_C": lumen_C,
        "lumen_R": lumen_R,
        "m_local_factory": m_local_factory,
        "result_detail": result_detail,
        "store_history": bool(store_history),
        "store_vectors_in_info": bool(store_vectors_in_info),
        "sensitivity_workers": workers,
        "copy_cached_results": bool(copy_cached_results),
    }

    contact_model = build_forward_model(
        **common,
        contact_enabled=True,
        use_lumen_jac=True,
        contact_params=contact_params,
    )
    no_contact_model = build_forward_model(
        **common,
        contact_enabled=False,
        use_lumen_jac=False,
        contact_params=None,
    )

    for name, model in (
        ("contact_model", contact_model),
        ("no_contact_model", no_contact_model),
    ):
        if not isinstance(model, MagneticBeamForwardModel):
            raise TypeError(
                f"{name} was built as {type(model).__name__}, expected "
                "MagneticBeamForwardModelOptimized. Check that "
                "model_factory_optimized.py imports forward_model_optimized."
            )

    jacobian_model = contact_model if variant == "contact" else no_contact_model
    return HardwareModelBundleOptimized(
        prediction_model=contact_model,
        jacobian_model=jacobian_model,
        contact_model=contact_model,
        no_contact_model=no_contact_model,
        lumen_C=lumen_C.copy(),
        lumen_R=lumen_R.copy(),
        p0_ur=np.asarray(p0_ur, dtype=float).reshape(3).copy(),
        q0_ur=np.asarray(q0_ur, dtype=float).reshape(4).copy(),
        sensitivity_workers=workers,
        jacobian_variant=variant,
    )


# Compatibility aliases for the previous hardware code.
def build_hardware_model_bundle(**kwargs) -> HardwareModelBundleOptimized:
    return build_hardware_model_bundle_optimized(**kwargs)
