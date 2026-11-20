from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.parameters import ROBOT_IP, default_beam_params, default_magnet_params, default_alpha_controller_params
from proper_research.vision.camera import measure_beam_angle_deg
import numpy as np 
import matplotlib.pyplot as plt 
import time

robo = URRtde(ROBOT_IP)
mag_params = default_magnet_params()
beam_params = default_beam_params()
alpha_params = default_alpha_controller_params()
beam_params.L_init = 0.018
beam_params.B_init = 0.03
alpha_params.Kp = 10.0
alpha_params.Ki = 0.02
alpha_params.kd = 0.7

try:
    robo.get_pose()
    base_point = get_point(0,0)
    robo.moveL(base_point)


    angles_path = np.linspace(0,80,3)
    for ang in angles_path:
        pose = get_point(0,ang)
        robo.moveL(pose)
    
    angles = np.linspace(80, -80, 20)

    com_angles = []
    prev_deg = None

    max_retries = 5
    jump_thresh = 20.0

    for i, tes in enumerate(angles):
        next_pos = get_point(0, tes)
        robo.moveL(next_pos)
        time.sleep(0.2)

        # take a measurement (with retries if it jumps too much)
        angle_deg, pts, roi_box = measure_beam_angle_deg(
            image_filename="focused_image.jpg",
            use_roi=True,
            show=False,
            show_debug_markers=False,
            use_segment="base_to_tip",
        )

        retries = 0
        while prev_deg is not None and abs(angle_deg - prev_deg) > jump_thresh and retries < max_retries:
            retries += 1
            time.sleep(0.1)  # let camera settle a bit
            angle_deg, pts, roi_box = measure_beam_angle_deg(
                image_filename="focused_image.jpg",
                use_roi=True,
                show=False,
                show_debug_markers=False,
                use_segment="base_to_tip",
            )

        if prev_deg is not None and abs(angle_deg - prev_deg) > jump_thresh:
            # still bad after retries: choose a policy
            # Policy A (skip): don't update prev, store NaN
            print(f"[WARN] Outlier persists at tes={tes:.1f}: {angle_deg:.2f} vs prev {prev_deg:.2f} after {max_retries} retries")
            com_angles.append(np.nan)
            continue

            # Policy B (clamp): accept but clamp to prev +/- jump_thresh
            # angle_deg = prev_deg + np.sign(angle_deg - prev_deg) * jump_thresh

        print(f"Angle (deg) {angle_deg:.2f} with rotation: {tes:.1f} (retries={retries})")

        prev_deg = angle_deg
        com_angles.append(angle_deg)

    



    current_pose = robo.get_pose()
    current_pose[0] -=0.2
    robo.moveL(current_pose)
    robo.moveL(base_point)




finally:
    robo.shutdown()

    plt.figure()
    plt.plot(angles, com_angles, label="Bending in x-y plane")
    plt.xlabel("Commanded angle (deg)")
    plt.ylabel("Measured beam angle (deg)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("bending_curve.png", dpi=200)   # <- saves to current working directory
    print("Saved plot to bending_curve.png")

    plt.show()
