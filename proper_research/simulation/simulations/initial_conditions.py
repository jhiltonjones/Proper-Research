import numpy as np

from proper_research.robot.transformations import get_point
from scipy.spatial.transform import Rotation as R


# =====================================================================
# LIVE HARDWARE STATE  (read from the robot on 2026-09-09 via robot_class)
# ---------------------------------------------------------------------
#   joints (rad) = [-0.62780553, -2.06211915, -1.61096144,
#                   -1.05756475,  1.56630862, -1.76096565]
#   TCP pose6 (flange == TCP; the installed TCP offset is zero)
#              = [0.62616089, -0.67019182, 0.31343135,
#                -3.06532747,  0.68084844,  0.03120102]
#   Source magnet: 30 mm below the TCP along TCP.z
#              (T_tcp_magnet_pose6 = (0, 0, 0.03, 0, 0, 0), matches
#               robotics_frame_measurement_validation.CONFIG).
#
# start_point below is the magnet pose6 in the robot base frame R,
# recomputed from the offline nominal-DH forward kinematics at the live
# joints so that the offline IK (planning_context.make_robot_config,
# q_seed = the live joints) returns those exact six joint angles.
#
# NOTE: the offline nominal-DH FK differs from the robot's own calibrated
# FK by ~1.3 mm at this pose (urik.CONFIG has zero dh_delta_*).  The
# commanded joint trajectory is unaffected; the beam-tip predictions
# carry that ~1.3 mm systematic offset until the DH is calibrated.
# =====================================================================

LIVE_JOINTS_RAD = (
    -0.62780553,
    -2.06211915,
    -1.61096144,
    -1.05756475,
    1.56630862,
    -1.76096565,
)


def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build the fixed initial robot/lumen experiment pose, matched to the
    live hardware state above.

    Returns
    -------
    pivot_point:
        UR-style pose6 for the beam base / pivot: [x, y, z, rx, ry, rz].
        The rotation matters: ``build_model_bundle`` grows the Cosserat rod
        along ``R0 @ [-1, 0, 0]``, so ``R0`` must map ``[-1, 0, 0]`` onto the
        true beam axial direction.  The hardware beam stands straight up
        (+R.z), and ``Ry(+90 deg) @ [-1, 0, 0] = [0, 0, 1]``.

    start_point:
        UR-style pose6 for the initial source-magnet pose in R.

    L_cmd:
        Initial insertion length [m].

    dt:
        Controller timestep [s].
    """
    L_cmd = 0.037
    dt = 0.01

    # Beam base in R (robotics_frame_measurement_validation.CONFIG
    # T_robot_beam_pose6 position), with the rotation chosen so the model
    # rod grows along +R.z.
    pivot_point = np.array(
        [0.525575, -0.670028, -0.016567, 0.0, np.pi / 2.0, 0.0],
        dtype=float,
    )

    # Source-magnet pose6 in R, consistent with the offline FK at the live
    # joints (see the header note).  IK(start_point, q_seed=LIVE_JOINTS_RAD)
    # -> LIVE_JOINTS_RAD exactly.
    start_point = np.array(
        [
            0.62484299,
            -0.66947913,
            0.28254041,
            -3.06349506,
            0.68119035,
            0.02908557,
        ],
        dtype=float,
    )

    print(f"[INIT] start_point (magnet in R) = {start_point}")
    print(f"[INIT] pivot_point  (beam base)  = {pivot_point}")
    print(f"[INIT] L0={L_cmd:.4f}, dt={dt:.4f}")

    return pivot_point, start_point, L_cmd, dt
