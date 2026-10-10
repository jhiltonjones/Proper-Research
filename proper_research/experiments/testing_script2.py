from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.parameters import ROBOT_IP
from proper_research.vision.camera import measure_beam_angle_deg

import numpy as np 
import os
import pickle
import time


robo = URRtde(ROBOT_IP)


try:
    robo.get_pose()
    base_point = get_point(0,0)
    robo.moveL(base_point)
    angle_deg =90
    target = 0
    tol = 3
    tol_2 = 1

    kp,ki,kd = 0.005,0.005,0.001
    dt = 0.1
    integral_sum = 0
    last_error = 0
    while abs(angle_deg - target) > tol:
        current_pose = robo.get_pose()
        angle_deg, (base_px, mag_start_px, tip_px), roi_box = measure_beam_angle_deg(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        show_debug_markers=False,
        use_segment="base_to_tip",
        )

        error = angle_deg - target
        p_out = kp*error
        integral_sum +=error*dt
        i_out = integral_sum*ki
        derivative = (error-last_error)/dt
        d_out = derivative*kd
        u = p_out + i_out + d_out
        max_u = 0.02
        u = np.clip(u,-max_u, max_u)
        last_error = error
        current_pose[1] -= u
        robo.moveL(current_pose)
        time.sleep(dt)
    while abs(angle_deg - target) > tol_2:
        current_joints = robo.get_joints()
        angle_deg, (base_px, mag_start_px, tip_px), roi_box = measure_beam_angle_deg(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        show_debug_markers=False,
        use_segment="base_to_tip",
        )

        error = angle_deg - target
        p_out = kp*error
        integral_sum +=error*dt
        i_out = integral_sum*ki
        derivative = (error-last_error)/dt
        d_out = derivative*kd
        u = p_out + i_out + d_out
        max_u = 0.02
        u = np.clip(u,-max_u, max_u)
        last_error = error
        current_joints[5] += u
        robo.moveJ(current_joints)
        time.sleep(dt)
finally:
    robo.shutdown()