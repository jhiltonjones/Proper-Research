import numpy as np
from proper_research.simulation_controller.geometry import pose8_quat_to_pose7_rotvec, quat_to_R
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize
def J_full_from_robot_reduced_tip_tangent(J_red, n_out_full=5):
    """
    J_red shape: (5,4)
        columns = [x, y, yaw_z, L]

    Returns full robot Jacobian in 7 controls:
        [vx, vy, vz, wx, wy, wz, dL]
    """
    J_full = np.zeros((n_out_full, 7), float)
    J_full[:, 0] = J_red[:, 0]   # x translation
    J_full[:, 1] = J_red[:, 1]   # y translation
    J_full[:, 5] = J_red[:, 2]   # yaw about z
    J_full[:, 6] = J_red[:, 3]   # insertion
    return J_full

def analytic_J_robot_xy_yaw_dL(
    p8,
    forward_model,
    n_out=6,
):
    p8 = np.asarray(p8, float).reshape(8,)

    if hasattr(forward_model, "fwd"):
        wrapper = forward_model
        fm = forward_model.fwd
    else:
        wrapper = None
        fm = forward_model

    p7 = pose8_quat_to_pose7_rotvec(p8)

    # One nominal solve only
    if wrapper is not None:
        _ = wrapper(p8, commit=True)
    else:
        _ = fm(p7)

    # No second solve here
    J_tip_7 = fm.jacobian_tip_pose7_from_cache(p7)

    J = np.zeros((n_out, 4), dtype=float)

    J[0:3, 0] = J_tip_7[:, 0]   # robot x
    J[0:3, 1] = J_tip_7[:, 1]   # robot y
    J[0:3, 2] = J_tip_7[:, 5]   # yaw ≈ world delta_phi_z
    J[0:3, 3] = J_tip_7[:, 6]   # length

    return J


def analytic_J_robot_full_with_diag(
    p8,
    forward_model,
    n_out: int = 3,
):
    """
    Full control-to-tip-position Jacobian plus beam diagnostics.

    Pose
    ----
    p8:
        [x, y, z, qw, qx, qy, qz, L]

    Controller input
    ----------------
    u:
        [
            vx_world,
            vy_world,
            vz_world,
            wx_body,
            wy_body,
            wz_body,
            dL_rate,
        ]

    Returns
    -------
    J_control_state:
        Shape (3, 7). This is the instantaneous sensitivity

            d p_tip / d [
                r_src_world,
                phi_src_body_increment,
                L,
            ]

        It has not yet been multiplied by dt.

    diag:
        Beam-Hessian and sensitivity diagnostics.
    """
    p8 = np.asarray(
        p8,
        float,
    ).reshape(8)

    if n_out != 3:
        raise ValueError(
            "analytic_J_robot_full_with_diag currently "
            "provides the 3D tip-position Jacobian only, "
            f"so n_out must be 3; got {n_out}."
        )

    # ------------------------------------------------------------
    # Resolve adapter versus raw forward model
    # ------------------------------------------------------------
    if hasattr(forward_model, "model"):
        wrapper = forward_model
        fm = forward_model.model
    else:
        wrapper = None
        fm = forward_model

    p7 = pose8_quat_to_pose7_rotvec(p8)

    # ------------------------------------------------------------
    # Commit the nominal equilibrium solution
    # ------------------------------------------------------------
    if wrapper is not None:
        _ = wrapper(
            p8,
            commit=True,
        )
    else:
        _ = fm.solve(
            p7,
            commit=True,
        )

    # ------------------------------------------------------------
    # Complete beam sensitivity
    #
    # Column convention:
    #   0:3 -> source translation in world coordinates
    #   3:6 -> incremental source rotation in world coordinates
    #   6   -> insertion length
    # ------------------------------------------------------------
    J_tip_theta_world = np.asarray(
        fm.jacobian_tip_pose7(p7),
        float,
    )

    if J_tip_theta_world.shape != (3, 7):
        raise ValueError(
            "Expected the beam model to return a full "
            "tip Jacobian with shape (3, 7), but got "
            f"{J_tip_theta_world.shape}."
        )

    J_translation_world = (
        J_tip_theta_world[:, 0:3]
    )

    J_rotation_world = (
        J_tip_theta_world[:, 3:6]
    )

    J_length = (
        J_tip_theta_world[:, 6:7]
    )

    # ------------------------------------------------------------
    # Convert controller angular velocity convention
    #
    # Controller:
    #   q_next = q ⊗ dq(dt * omega_body)
    #
    # Beam sensitivity:
    #   q_perturbed = dq(delta_phi_world) ⊗ q
    #
    # First-order relationship:
    #   delta_phi_world = R_src delta_phi_body
    # ------------------------------------------------------------
    q_src = quat_wxyz_normalize(
        p8[3:7]
    )

    R_src = quat_to_R(q_src)

    J_rotation_body = (
        J_rotation_world @ R_src
    )

    # ------------------------------------------------------------
    # Full controller-coordinate state Jacobian
    # ------------------------------------------------------------
    J_control_state = np.concatenate(
        (
            J_translation_world,
            J_rotation_body,
            J_length,
        ),
        axis=1,
    )

    if J_control_state.shape != (3, 7):
        raise RuntimeError(
            "Internal full-Jacobian construction produced "
            f"shape {J_control_state.shape}; expected (3, 7)."
        )

    if not np.all(
        np.isfinite(J_control_state)
    ):
        raise FloatingPointError(
            "Full tip Jacobian contains non-finite values."
        )

    diag = {}

    if hasattr(
        fm,
        "get_last_jacobian_diag",
    ):
        diag.update(
            fm.get_last_jacobian_diag()
        )

    return J_control_state, diag