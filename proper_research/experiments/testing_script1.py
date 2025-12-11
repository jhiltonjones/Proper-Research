from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.parameters import ROBOT_IP, default_beam_params, default_magnet_params, default_alpha_controller_params
from proper_research.vision.camera import new_capture, measure_theta_from_camera, detect_red_points_and_angle
from proper_research.control.jacobian_controller import jacobian_controller
from proper_research.control.alpha_controller import alpha_controller_measured
import numpy as np 
import os
import pickle
import time

def move_function(alpha_abs):
    joints = robo.get_joints()
    joints[5] = alpha_zero + alpha_abs 
    robo.moveJ(joints)


robo = URRtde(ROBOT_IP)
mag_params = default_magnet_params()
beam_params = default_beam_params()
alpha_params = default_alpha_controller_params()
beam_params.L_init = 0.032
beam_params.B_init = 0.03
alpha_params.Kp = 7.0
alpha_params.Ki = 0.02
alpha_params.kd = 0.7
measure_fn = measure_theta_from_camera

danger_file = "/home/jack/Proper-Research/data/alpha_danger_map.pkl"
if os.path.exists(danger_file):
    with open(danger_file, "rb") as f:
        saved = pickle.load(f)
    phi_values = saved["phi_values"]
    alphas = saved["alphas"]
    danger_intervals_by_phi = saved["danger_intervals_by_phi"]
    print("Loaded precomputed data")
else:
    print("Could not load the danger file")

try:
    theta_des = np.deg2rad(15)
    base_point = get_point(0,0)
    robo.moveL(base_point)
    B_cmd, phi_cmd, L_cmd, theta_model_deg = jacobian_controller(theta_des, beam_params.B_init, beam_params.phi_init, beam_params.L_init, beam_params.mag, beam_params.A_cs, beam_params.E
                                                                 , beam_params.I, L_min= beam_params.L_init, L_max=beam_params.L_init,
                                                                 B_min= beam_params.B_init, B_max=beam_params.B_init)
    print("Jacobian controller result (feedforward):")
    print("  B   =", B_cmd)
    print("  phi =", phi_cmd, "(rad)", f"= {np.rad2deg(phi_cmd):.2f} deg")
    print("  L   =", L_cmd, "(m)")
    print("  model-predicted angle =", theta_model_deg, "deg")
    current_phi = phi_cmd
    new_pose = get_point(0, np.rad2deg(current_phi))
    robo.moveL(new_pose)
    time.sleep(0.1)
    # input("Continue?")
    zero_joints = robo.get_joints()
    alpha_zero = zero_joints[5]
    alpha_sol, theta_final, B_final, status = alpha_controller_measured(
        theta_des=theta_des,
        phi_fixed=phi_cmd,
        alpha_init=alpha_params.alpha_init,
        R=0.1,
        m0=mag_params.mag_epm,
        mu0=mag_params.mu_0,
        mag=beam_params.mag,
        A_cs=beam_params.A_cs,
        L=beam_params.L_init,
        E=beam_params.E,
        I=beam_params.I,
        danger_intervals_by_phi=danger_intervals_by_phi,
        phi_values=phi_values,
        measure_theta_fn=measure_fn,
        move_fn=move_function,
        params=alpha_params, 
        use_online=True,
        )
    img_file = new_capture(filename="focused_image.jpg")
    pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
        img_file,
        show=True,
        use_roi=True
    )
    robo.moveL(base_point)
finally:
    robo.shutdown()