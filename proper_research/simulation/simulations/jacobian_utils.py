import numpy as np
from proper_research.simulation_controller.geometry import pose8_quat_to_pose7_rotvec
from proper_research.simulation_controller.diagnostics import symmetric_matrix_diagnostics
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
def analytic_J_robot_xy_yaw_dL_with_diag(
    p8,
    forward_model,
    n_out=3,
):
    """
    Reduced Jacobian for MPC plus beam Hessian diagnostics.

    Input p8:
        [x, y, z, qw, qx, qy, qz, L]

    Reduced control columns:
        [vx, vy, yaw_rate, dL_rate]

    Returns:
        J:    shape (n_out, 4)
        diag: dict, including cond_H_beam etc.
    """
    p8 = np.asarray(p8, float).reshape(8)

    # Adapter case: ControllerForwardAdapter has .model.
    if hasattr(forward_model, "model"):
        wrapper = forward_model
        fm = forward_model.model
    else:
        wrapper = None
        fm = forward_model

    p7 = pose8_quat_to_pose7_rotvec(p8)

    # Commit nominal solve so jacobian_tip_pose7 can use cache.
    if wrapper is not None:
        _ = wrapper(p8, commit=True)
    else:
        _ = fm.solve(p7, commit=True)

    J_tip_7 = fm.jacobian_tip_pose7(p7)
    J_tip_7 = np.asarray(J_tip_7, float)

    if J_tip_7.shape != (3, 7):
        raise ValueError(
            f"Expected J_tip_7 shape (3, 7), got {J_tip_7.shape}."
        )

    if n_out < 3:
        raise ValueError("n_out must be at least 3 for tip position tracking.")

    J = np.zeros((n_out, 4), dtype=float)

    J[:3, 0] = J_tip_7[:, 0]   # source x translation
    J[:3, 1] = J_tip_7[:, 1]   # source y translation
    J[:3, 2] = J_tip_7[:, 5]   # source yaw / rz
    J[:3, 3] = J_tip_7[:, 6]   # insertion length

    diag = {}
    if hasattr(fm, "get_last_jacobian_diag"):
        diag.update(fm.get_last_jacobian_diag())

    return J, diag
