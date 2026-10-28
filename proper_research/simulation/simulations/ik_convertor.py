from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation as R

from proper_research.hardware import (
    ur_magnet_ik_jacobian_validation as ur_ik
)


def find_initial_joint_state(
    start_point: np.ndarray,
    L_cmd: float,
    q_seed: np.ndarray,
    *,
    active_tcp_pose6: np.ndarray | None = None,
    tcp_to_magnet_pose6: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    """
    Find robot joints whose magnet frame matches start_point.

    Parameters
    ----------
    start_point:
        Desired magnet pose in the robot base frame:
        [x, y, z, rx, ry, rz].

    L_cmd:
        Initial beam insertion length.

    q_seed:
        Initial joint estimate in radians. Prefer the measured joints
        from a nearby safe robot configuration.

    active_tcp_pose6:
        Flange-to-active-TCP transformation.

    tcp_to_magnet_pose6:
        Active-TCP-to-magnet fixed transformation.

    Returns
    -------
    state0:
        Initial MPC state [q1, ..., q6, L].

    result:
        Complete custom IK result and diagnostics.
    """
    start_point = np.asarray(start_point, dtype=float).reshape(6)
    q_seed = np.asarray(q_seed, dtype=float).reshape(6)

    if active_tcp_pose6 is None:
        active_tcp_pose6 = np.zeros(6, dtype=float)

    if tcp_to_magnet_pose6 is None:
        # Use this sign only if the magnet is -27 mm along the
        # local TCP z-axis.
        tcp_to_magnet_pose6 = np.array(
            [0.0, 0.0, -0.044, 0.0, 0.0, 0.0],
            dtype=float,
        )

    # Desired robot-base-to-magnet transformation.
    T_R_M_desired = ur_ik.pose6_to_T(start_point)

    # Fixed end-effector transformations.
    T_F_TCP = ur_ik.pose6_to_T(active_tcp_pose6)
    T_TCP_M = ur_ik.pose6_to_T(tcp_to_magnet_pose6)

    # Fixed flange-to-magnet transformation.
    T_F_M = T_F_TCP @ T_TCP_M

    # Use the same robot model, DH corrections and joint limits as
    # the validation script.
    cfg = replace(
        ur_ik.CONFIG,
        debug_enabled=True,
    )
    dh = ur_ik.corrected_dh_from_config(cfg)

    result = ur_ik.inverse_kinematics_dls(
        T_R_target=T_R_M_desired,
        q_seed_rad=q_seed,
        dh=dh,
        T_F_target=T_F_M,
        cfg=cfg,
    )

    print("\nINITIAL MAGNET IK")
    print(f"converged              : {result.converged}")
    print(f"termination reason     : {result.reason}")
    print(f"iterations             : {result.iterations}")
    print(f"position error [m]     : "
          f"{result.final_position_error_m:.6e}")
    print(f"orientation error [rad]: "
          f"{result.final_orientation_error_rad:.6e}")
    print(f"Jacobian condition     : "
          f"{result.final_scaled_jacobian_condition:.6e}")
    print(f"q0 [rad]               : {result.q_rad}")
    print(f"q0 [deg]               : {np.rad2deg(result.q_rad)}")

    if not result.converged:
        raise RuntimeError(
            "Could not find an initial joint configuration. "
            "Inspect the IK errors and try another nearby q_seed."
        )

    q0 = np.asarray(result.q_rad, dtype=float).reshape(6)

    # Independent forward-kinematics verification.
    fk = ur_ik.forward_kinematics(
        q_rad=q0,
        dh=dh,
        T_F_target=T_F_M,
    )
    T_R_M_actual = fk.T_R_target

    position_error = (
        T_R_M_desired[:3, 3] - T_R_M_actual[:3, 3]
    )

    R_error = (
        T_R_M_desired[:3, :3]
        @ T_R_M_actual[:3, :3].T
    )
    orientation_error = R.from_matrix(R_error).as_rotvec()

    print("\nFORWARD-KINEMATICS CHECK")
    print(f"desired magnet pose : "
          f"{ur_ik.T_to_pose6(T_R_M_desired)}")
    print(f"actual magnet pose  : "
          f"{ur_ik.T_to_pose6(T_R_M_actual)}")
    print(f"position error [m]  : {position_error}")
    print(f"|position error| [m]: "
          f"{np.linalg.norm(position_error):.6e}")
    print(f"rotation error [rad]: {orientation_error}")
    print(f"|rotation error|    : "
          f"{np.linalg.norm(orientation_error):.6e}")

    # Joint-space MPC state.
    state0 = np.concatenate([q0, [float(L_cmd)]])

    return state0, result