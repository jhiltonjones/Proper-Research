from robot_class_trans import URRtde
from transformations import get_point
from camera import detect_red_points_and_angle, new_capture
from jac_controller import jacobian_controller
import numpy as np
from magnetic_magnitude import magnetic_moment
import time

ROBOT_IP = "192.168.56.101"

def measure_beam_angle_deg(image_filename="focused_image.jpg", use_roi=True, show=False):
    """
    Captures an image (or uses an existing file), detects the two red markers,
    and returns the measured beam angle in degrees.
    """
    img_file = new_capture(filename=image_filename)
    pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(
        img_file,
        show=show,
        use_roi=use_roi
    )
    return angle_deg, (pt1, pt2), roi_box

# ---------------------------------------------------------------------
# DESIRED ANGLE & MODEL PARAMETERS
# ---------------------------------------------------------------------
theta_des = np.deg2rad(50)      # desired beam angle in radians

mag = 128e3
r = 0.0015
E = 3.5e6
A_cs = np.pi * r**2
I = np.pi * r**4 / 4

# initial guesses for Jacobian controller
B_init = 0.01
phi_init = np.deg2rad(25)
L_init = 0.042

mu_0 = 4e-7 * np.pi
B_r = 1.25
r_epm = 0.03
p_epm = 0.09
m_hat = np.array([1, 0, 0])
x_init = 0.09
mag_epm = magnetic_moment(B_r, mu_0, r_epm, p_epm)

# ---------------------------------------------------------------------
# VISION-BASED PID SETTINGS (outer loop)
# ---------------------------------------------------------------------
Kp_vis = 0.3          # proportional gain on visual angle error (deg -> deg)
Ki_vis = 0.0          # integral gain (often start with 0)
Kd_vis = 0.05         # derivative gain
dt_vis = 0.2          # approximate loop time [s]
max_vis_iters = 20
visual_tol_deg = 0.5  # acceptable error in degrees

# phi limits (like your Jacobian controller)
phi_min = np.deg2rad(-90)
phi_max = np.deg2rad(90)

robo = URRtde(ROBOT_IP)

try:
    # -----------------------------------------------------------------
    # 0) Move to base pose
    # -----------------------------------------------------------------
    base_pose = get_point(0, 0)
    robo.moveL(base_pose)

    # -----------------------------------------------------------------
    # 1) FEEDFORWARD: one-shot Jacobian controller
    # -----------------------------------------------------------------
    B_cmd, phi_cmd, L_cmd, theta_model_deg = jacobian_controller(
        theta_des,
        B_init, phi_init, L_init,
        mag, A_cs, E, I,
        L_max=0.04
    )
    print("Jacobian controller result (feedforward):")
    print("  B   =", B_cmd)
    print("  phi =", phi_cmd, "(rad)", f"= {np.rad2deg(phi_cmd):.2f} deg")
    print("  L   =", L_cmd, "(m)")
    print("  model-predicted angle =", theta_model_deg, "deg")

    # Move robot to the model-predicted configuration (using phi_cmd)
    current_phi = phi_cmd
    new_pose = get_point(0, np.rad2deg(current_phi))
    robo.moveL(new_pose)

    # -----------------------------------------------------------------
    # 2) FEEDBACK: visual PID loop on phi to correct angle
    # -----------------------------------------------------------------
    theta_des_deg = np.rad2deg(theta_des)

    e_int = 0.0
    e_prev = 0.0

    for k in range(max_vis_iters):
        print(f"\n[Visual PID iteration {k}]")

        # 2a) Measure current angle from camera
        theta_meas_deg, (pt1, pt2), roi_box = measure_beam_angle_deg(
            image_filename="focused_image.jpg",
            use_roi=True,
            show=False
        )

        # NOTE: your previous code used error = | -theta_des_deg - theta_meas_deg |
        # If your sign convention is such that beam angles appear negated, keep:
        #   e = -theta_des_deg - theta_meas_deg
        # Otherwise use:
        #   e = theta_des_deg - theta_meas_deg
        # I keep your convention here:
        e = -theta_des_deg - theta_meas_deg  # error in degrees

        print("  Measured beam angle from camera:", theta_meas_deg, "deg")
        print("  Error (desired - measured, with sign convention):", e, "deg")

        if abs(e) <= visual_tol_deg:
            print("  Visual error within tolerance; PID loop converged.")
            break

        # 2b) PID on angle error
        e_int += e * dt_vis
        e_dot = (e - e_prev) / dt_vis

        dphi_deg = Kp_vis * e + Ki_vis * e_int + Kd_vis * e_dot  # change in phi (deg)
        e_prev = e

        # 2c) Update phi command
        phi_deg = np.rad2deg(current_phi) - dphi_deg
        # enforce limits
        phi_deg = max(min(phi_deg, np.rad2deg(phi_max)), np.rad2deg(phi_min))
        current_phi = np.deg2rad(phi_deg)

        print("  PID dphi:", dphi_deg, "deg  -> new phi:", phi_deg, "deg")

        # 2d) Move robot to new phi
        new_pose = get_point(0, phi_deg)
        robo.moveL(new_pose)

        # give some time between iterations (approximate dt_vis)
        time.sleep(dt_vis)

    print("\nFinal configuration after feedforward + visual PID:")
    print("  B   =", B_cmd, "(unchanged in visual loop)")
    print("  phi =", current_phi, "(rad)", f"= {np.rad2deg(current_phi):.2f} deg")
    print("  L   =", L_cmd, "(m)")

finally:
    robo.shutdown()


