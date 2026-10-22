import csv
import matplotlib.pyplot as plt
import numpy as np
import time
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.parameters import ROBOT_IP, default_beam_params, default_magnet_params, default_alpha_controller_params
from proper_research.vision.camera import new_capture, measure_theta_from_camera, detect_red_points_and_angle
from proper_research.control.jacobian_controller import jacobian_controller, solve_mag_pose
from proper_research.control.alpha_controller import alpha_controller_measured
from proper_research.advancer_unit.advancer_control import advancer_go
from proper_research.vision.measure_length import measure_beam_length_mm_with_checkerboard
from proper_research.vision.click_target import (detect_tip_px_using_reference_base, image_points_to_mm, 
                                                plot_all_targets_and_tips_on_image, compute_beam_targets_from_clicked_points)
import numpy as np 
import os
import cv2
import pickle
import time

def move_function(alpha_abs):
    joints = robo.get_joints()
    joints[5] = alpha_zero +alpha_abs 
    robo.moveJ(joints)

start_point = np.array([0.6372552555949702, -0.5765906755688711, 0.4569897127437801, 1.8596684013618174, -2.529358846151256, -0.040608351291805976])
pivot_point = np.array([0.7772552555949702, -0.5765906755688711, 0.2069897127437801, 1.8596684013618174, -2.529358846151256, -0.040608351291805976])
robo = URRtde(ROBOT_IP)
mag_params = default_magnet_params()
beam_params = default_beam_params()
alpha_params = default_alpha_controller_params()
beam_params.L_init = 0.018
beam_params.B_init = 0.02
alpha_params.Kp = 10.0
alpha_params.Ki = 0.02
alpha_params.kd = 0.7
measure_fn = measure_theta_from_camera
base_point_pose = pivot_point[0]
start_pivot_point = pivot_point

start_point_pose = start_point
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

def run_one_target(target_id, length_des_mm, theta_des_deg):
    """
    Executes one (length, theta) command and returns a dict with final measured tip angle.
    """
    theta_des = np.deg2rad(theta_des_deg)
    theta_des_abs = abs(theta_des)

    if theta_des_abs >= np.deg2rad(50) or length_des_mm >= 80:
        return {
            "target_id": target_id,
            "length_des_mm": float(length_des_mm),
            "theta_des_deg": float(theta_des_deg),
            "status": "not_achievable",
        }

    advancer_go(length_des_mm)
    result_len = measure_beam_length_mm_with_checkerboard(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=False,
    )
    length_curr_mm = float(result_len["length_mm"])
    beam_params.L_init = length_curr_mm / 1000.0  

    if length_curr_mm < 40:
        base_point = get_point(0, 0)
        robo.moveL(base_point)
    else:
        diff = length_curr_mm - 40
        temp_pivot_point = start_pivot_point
        temp_pivot_point[0]-= (diff/1000)
        base_point = get_point(0,0,pivot_point=temp_pivot_point)

    B_cmd, phi_cmd, L_cmd, theta_model_deg = jacobian_controller(
        theta_des_abs,
        beam_params.B_init,
        beam_params.phi_init,
        beam_params.L_init,
        beam_params.mag,
        beam_params.A_cs,
        beam_params.E,
        beam_params.I,
        L_min=beam_params.L_init,
        L_max=beam_params.L_init,
        B_min=beam_params.B_init,
        B_max=beam_params.B_init,
    )
    print(f"B is: {B_cmd}")
    print(f"Phi is: {phi_cmd}")
    print(f"Estimated theta: {theta_model_deg}")
    mag_pose, _ = solve_mag_pose(
        B_cmd, 0.1, mag_params.mu_0, mag_params.mag_epm, mu_hat=np.array([1, 0, 0])
    )
    if length_curr_mm < 40:
        start_point = base_point_pose - mag_pose
        start_point_pose[0] = start_point
        current_phi = phi_cmd

        if theta_des > 0:
            new_pose = get_point(0, np.rad2deg(current_phi), start_point_pose)
        else:
            new_pose = get_point(0, np.rad2deg(-current_phi), start_point_pose)

        new_base_pose = get_point(0, 0, start_point_pose)
    else:
        start_point = temp_pivot_point[0] - mag_pose
        start_point_pose[0] = start_point
        current_phi = phi_cmd

        if theta_des > 0:
            new_pose = get_point(0, np.rad2deg(current_phi), start_point_pose)
        else:
            new_pose = get_point(0, np.rad2deg(-current_phi), start_point_pose)

        new_base_pose = get_point(0, 0, start_point_pose)

    if start_point <= 0.3:
        img_file = new_capture(filename=f"focused_image_target_{target_id}.jpg")

        _, _, tip_angle_deg, _ = detect_red_points_and_angle(img_file, show=False, use_roi=True)

        robo.moveL(base_point)

        return {
            "target_id": target_id,
            "final_image": img_file,
            "tip_angle_deg_measured": float(-tip_angle_deg),
            "length_des_mm": float(length_des_mm),
            "theta_des_deg": float(theta_des_deg),
            "length_meas_mm": float(length_curr_mm),
            "status": "too_close",
            "timestamp": time.time(),
        }



    robo.moveL(new_base_pose)
    robo.moveL(new_pose)

    zero_joints = robo.get_joints()
    global alpha_zero
    alpha_zero = zero_joints[5]

    alpha_sol, theta_final_rad, B_final, status = alpha_controller_measured(
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
    _, _, tip_angle_deg, _ = detect_red_points_and_angle(
        img_file, show=False, use_roi=True
    )

    # robo.moveL(base_point)

    return {
        "target_id": target_id,
        "final_image": img_file,
        "length_des_mm": float(length_des_mm),
        "theta_des_deg": float(theta_des_deg),
        "length_meas_mm": float(length_curr_mm),
        "phi_cmd_deg": float(np.rad2deg(phi_cmd)),
        "theta_model_deg": float(theta_model_deg),
        "alpha_sol": float(alpha_sol),
        "theta_final_rad": float(theta_final_rad),
        "tip_angle_deg_measured": float(-tip_angle_deg),
        "status": str(status),
        "timestamp": time.time(),
    }
results = []

try:
    robo.get_pose()
    base_point = get_point(0, 0)
    robo.moveL(base_point)

    rect = compute_beam_targets_from_clicked_points(
        image_filename="focused_image.jpg",
        use_roi=True,
        show=True,
        n_points=None,                 
        csv_path="clicked_targets.csv",
    )


    reference_image = rect["image_file"]         
    H_img_to_mm = rect["H_img_to_mm"]
    depth_scale = rect["depth_scale"]

    base_px_ref = rect["base_px"]

    targets_px = [tuple(t["target_px"]) for t in rect["targets"]]
    final_tips_px = []
    errors_mm = []

    for t in rect["targets"]:
        length_des_mm = t["length_mm"]
        theta_des_deg = t["theta_target_deg"]

        r = run_one_target(t["target_index"], length_des_mm, theta_des_deg)
        print("Result:", r)
        results.append(r)

        if "final_image" not in r:
            final_tips_px.append((np.nan, np.nan))
            errors_mm.append(None)
            continue

        final_image_filename = r["final_image"]
        img_bgr = cv2.imread(final_image_filename)
        if img_bgr is None:
            raise FileNotFoundError(final_image_filename)

        tip_px_final, _ = detect_tip_px_using_reference_base(
            img_bgr, base_px_ref=base_px_ref, use_roi=True
        )
        final_tips_px.append(tuple(tip_px_final))

        (target_mm_board,) = image_points_to_mm([t["target_px"]], H_img_to_mm)
        (tip_mm_board,)    = image_points_to_mm([tip_px_final], H_img_to_mm)

        target_mm = np.array(target_mm_board, dtype=np.float32) * depth_scale
        tip_mm    = np.array(tip_mm_board,    dtype=np.float32) * depth_scale

        err = float(np.linalg.norm(tip_mm - target_mm))
        errors_mm.append(err)
    # advancer_go(length_des_mm+15)

    plot_all_targets_and_tips_on_image(
        image_filename=reference_image,
        targets_px=targets_px,
        tips_px=final_tips_px,
        errors_mm=errors_mm,
        title="Rectangle targets (yellow) vs final tips (red)",
        draw_lines=True,
        annotate=True,
    )


finally:
    robo.shutdown()
