from proper_research.robot.live_hardware_control import (
    LiveHardwareController as ImportedLiveHardwareController,
    p8_to_ur_pose6_and_L as imported_p8_to_ur_pose6_and_L,
)
from proper_research.robot.transformations import get_point
import numpy as np
from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz

def make_initial_poses_single_use(hw):
    pivot_point = np.array([
        0.9081328220229531, -0.7112731669220016, -0.1, np.pi, 0.001, 0.001
    ], float)

    L0 = 0.055

    pose6 = np.asarray(get_point(0, 0), dtype=float)
    pose6[2] = -0.1
    print("POSE6 is", pose6)

    T = ur_pose6_to_T(pose6)
    p, q_wxyz = T_to_p_quat_wxyz(T)

    p_now = np.concatenate([p, q_wxyz, [L0]])
    ur_pose6_next, _ = imported_p8_to_ur_pose6_and_L(p_now)

    print("original pose6:", pose6)
    print("recovered pose6:", ur_pose6_next)
    print("difference:", ur_pose6_next - pose6)

    final_pose = np.array([
        ur_pose6_next[0],
        ur_pose6_next[1],
        ur_pose6_next[2] + hw.z_offset,
        ur_pose6_next[3],
        ur_pose6_next[4],
        ur_pose6_next[5],
    ])
    print("final sent pose:", final_pose)
    hw.open_robot()
    print("robot_mode:", hw.robo.robot_mode())
    print("safety_mode:", hw.robo.safety_mode())
    print("protective_stopped:", hw.robo.protective_stopped())
    print("pose after open:", hw.robo.get_pose())
    hw.close_robot()
    u0 = np.zeros(7, dtype=float)
    hw.send_step(p_now=p_now, u0=u0, dt=0.01)

    robot_pose6 = hw.get_robot_pose_once()
    robot_pose6[2] = -0.1
    print("START POINT:", robot_pose6)

    return pivot_point, robot_pose6, L0, 0.01
if __name__ == "__main__":
    hw = ImportedLiveHardwareController(
        robot_ip="192.168.56.101",
        dry_run=False,                 # True first
        use_advancer=True,
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=0.28,
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )
    make_initial_poses_single_use(hw)