from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from proper_research.simulation.boundary_forward_model import ContactParams
from proper_research.simulation.simulations.model_factory import build_forward_model


@dataclass
class HardwareModelBundle:
    plant_model: object
    jacobian_model: object
    lumen_C: np.ndarray
    lumen_R: np.ndarray
    p0_ur: np.ndarray
    q0_ur: np.ndarray


def build_hardware_model_bundle(
    *,
    p0_ur: np.ndarray,
    q0_ur: np.ndarray,
    Kinv_fun,
    m_body: np.ndarray,
    lumen_C_robot_m: np.ndarray,
    lumen_R_robot_m: np.ndarray,
    contact_params: ContactParams,
    jacobian_variant: str = "contact",
) -> HardwareModelBundle:
    """
    Build the same magnetic-beam forward models used in simulation, but with
    the lumen geometry reconstructed from the hardware camera.

    On hardware:
        plant_model is not the real plant.
        It is the nominal prediction model used inside MPC.

    Real plant:
        robot + catheter + camera measurement.
    """

    lumen_C_robot_m = np.asarray(lumen_C_robot_m, float)

    if lumen_C_robot_m.ndim != 2 or lumen_C_robot_m.shape[1] < 3:
        raise ValueError(
            f"lumen_C_robot_m must have shape (M, >=3), got {lumen_C_robot_m.shape}."
        )

    lumen_C_robot_m = lumen_C_robot_m[:, :3].copy()
    lumen_R_robot_m = np.asarray(lumen_R_robot_m, float).reshape(-1).copy()

    if lumen_R_robot_m.size != lumen_C_robot_m.shape[0]:
        raise ValueError(
            f"lumen_R_robot_m length {lumen_R_robot_m.size} does not match "
            f"lumen_C_robot_m length {lumen_C_robot_m.shape[0]}."
        )

    contact_model = build_forward_model(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C=lumen_C_robot_m,
        lumen_R=lumen_R_robot_m,
        contact_enabled=True,
        use_lumen_jac=True,
        contact_params=contact_params,
    )

    no_contact_model = build_forward_model(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C=lumen_C_robot_m,
        lumen_R=lumen_R_robot_m,
        contact_enabled=False,
        use_lumen_jac=False,
        contact_params=None,
    )

    if jacobian_variant == "contact":
        jacobian_model = contact_model
    elif jacobian_variant == "no_contact":
        jacobian_model = no_contact_model
    else:
        raise ValueError(
            f"Unknown jacobian_variant={jacobian_variant!r}. "
            "Use 'contact' or 'no_contact'."
        )
    print("[MODEL CONFIG] plant_model =", type(contact_model))
    print("[MODEL CONFIG] jacobian_model =", type(jacobian_model))
    print("[MODEL CONFIG] N_nodes =", getattr(contact_model, "N_nodes", None))
    print("[MODEL CONFIG] L_tip_full =", getattr(contact_model, "L_tip_full", None))
    print("[MODEL CONFIG] L_tip_min =", getattr(contact_model, "L_tip_min", None))
    print("[MODEL CONFIG] dL_internal =", getattr(contact_model, "dL_internal", None))
    print("[MODEL CONFIG] maxiter =", getattr(contact_model, "maxiter", None))
    print("[MODEL CONFIG] use_lumen_jac =", getattr(contact_model, "use_lumen_jac", None))
    print("[MODEL CONFIG] use_fast_contact_grad =", getattr(contact_model, "use_fast_contact_grad", None))
    return HardwareModelBundle(
        plant_model=contact_model,
        jacobian_model=jacobian_model,
        lumen_C=lumen_C_robot_m,
        lumen_R=lumen_R_robot_m,
        p0_ur=np.asarray(p0_ur, float).reshape(3).copy(),
        q0_ur=np.asarray(q0_ur, float).reshape(4).copy(),
    )