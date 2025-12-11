import numpy as np 
from robot_class_trans import URRtde
from robot_class_trans import URRtde
from transformations import get_point
from camera import detect_red_points_and_angle, new_capture
from jac_controller import jacobian_controller
import numpy as np
from magnetic_magnitude import magnetic_moment
import time
import pickle
from modelling_alpha import alpha_controller, alpha_controller_measured
import os


def move_function(alpha_abs):
    joints = robo.get_joints()
    joints[5] = alpha_zero + alpha_abs 
    robo.moveJ(joints)



def measure_theta_from_camera():
    img_file = new_capture(filename="focused_image.jpg")
    pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
        img_file,
        show=False,
        use_roi=True
    )
    return np.deg2rad(angle_deg)  




theta_des = np.deg2rad(36)      

mag = 128e3
r = 0.0015
E = 3e6
A_cs = np.pi * r**2
I = np.pi * r**4 / 4
R = 0.11

B_init = 0.025
L_init = 0.033

mu_0 = 4e-7 * np.pi
B_r = 1.25
r_epm = 0.03
p_epm = 0.09
m_hat = np.array([1, 0, 0])
x_init = 0.09
mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)
phi_init = np.deg2rad(60)
alpha_init = 0.0
# ---------------------------------------------------------------------
# VISION-BASED PID SETTINGS (outer loop)
# ---------------------------------------------------------------------
Kp_vis = 0.3          # proportional gain on visual angle error (deg -> deg)
Ki_vis = 0.0          # integral gain (often start with 0)
Kd_vis = 0.05         # derivative gain
dt_vis = 0.2          # approximate loop time [s]
max_vis_iters = 20
visual_tol_deg = 1  # acceptable error in degrees

# phi limits (like your Jacobian controller)
phi_min = np.deg2rad(-90)
phi_max = np.deg2rad(90)
danger_file = "alpha_danger_map.pkl"
if os.path.exists(danger_file):
    with open(danger_file, "rb") as f:
        saved = pickle.load(f)
    phi_values = saved["phi_values"]
    alphas = saved["alphas"]
    danger_intervals_by_phi = saved["danger_intervals_by_phi"]
    print("Loaded precomputed data")
else:
    print("Could not load the danger file")
ROBOT_IP = "192.168.56.101"
robo = URRtde(ROBOT_IP)

try:
    base_pose = get_point(0, 0)
    robo.moveL(base_pose)
    print(f"finished base movement")
    B_cmd, phi_cmd, L_cmd, theta_model_deg = jacobian_controller(
        theta_des,
        B_init, phi_init, L_init,
        mag, A_cs, E, I,
        L_max=L_init)
    print("Jacobian controller result (feedforward):")
    print("  B   =", B_cmd)
    print("  phi =", phi_cmd, "(rad)", f"= {np.rad2deg(phi_cmd):.2f} deg")
    print("  L   =", L_cmd, "(m)")
    print("  model-predicted angle =", theta_model_deg, "deg")

    # Move robot to the model-predicted configuration (using phi_cmd)
    current_phi = phi_cmd
    new_pose = get_point(0, np.rad2deg(current_phi))
    robo.moveL(new_pose)
    zero_joints = robo.get_joints()
    alpha_zero = zero_joints[5]  
    alpha_init = 0.0

    measure_fn = measure_theta_from_camera  

    alpha_final, theta_meas_final, B_final, status = alpha_controller_measured(
        theta_des,
        phi_init,
        alpha_init,
        R, mag_epm, mu_0,
        mag, A_cs, L_init, E, I,
        danger_intervals_by_phi,
        phi_values,
        measure_theta_fn=measure_fn,
        move_fn=move_function,
        max_iter=10,
        Kp=7.0, Ki=0.01, Kd=0.5
    )
    img_file = new_capture(filename="focused_image.jpg")
    pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
        img_file,
        show=True,
        use_roi=True
    )
    robo.moveL(base_pose)
finally:
    robo.shutdown()