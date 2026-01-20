    
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
from proper_research.control.mpc_control_in import mpc_controller_tipxy_LTI
from proper_research.models.forward_model import make_forward_fn, make_jac_fn
from proper_research.parameters import default_magnet_params, BeamParams
mag_params = default_magnet_params()
beam_params = default_beam_params()
eps = np.array([
    np.deg2rad(0.5),
    np.deg2rad(0.5),
    1e-3,
    5e-4
])
p_min = np.array([
    np.deg2rad(-90),   # gamma
    np.deg2rad(-90),   # beta
    0.1,              # rho (m)
    0.03               # L (m)
])
p_max = np.array([
    np.deg2rad(90),
    np.deg2rad(90),
    0.25,
    0.08
])
# forward_tip = make_forward_fn(beam_params.mag, beam_params.A_cs, beam_params.E, beam_params.I, mag_params.mag_epm)
# Jxy_fn = make_jac_fn(forward_tip, eps)

# mpc_xy = mpc_controller_tipxy_LTI(
#     Jxy_fn=Jxy_fn,
#     forward_tip_fn=forward_tip,
#     dt=0.2,
#     Np=10,
#     w_xy=(100.0, 100.0),
#     w_u=(1e-4, 1e-4, 1e-2, 1e-2),
#     w_du=(1e-3, 1e-3, 1e-2, 1e-2),
#     u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
#     p_min=p_min,
#     p_max=p_max,

#     N_sqp=3,      
#     use_offset_free=True, 
#     d_alpha=0.15          
# )
# rect = compute_beam_targets_from_clicked_points(
#         image_filename="focused_image.jpg",
#         use_roi=True,
#         show=True,
#         n_points=None,                 
#         csv_path="clicked_targets.csv",
#     )


# reference_image = rect["image_file"]         
# H_img_to_mm = rect["H_img_to_mm"]
# depth_scale = rect["depth_scale"]
# tip_pos = rect["tip_px"]
# base_px_ref = rect["base_px"]

# targets_px = [tuple(t["target_px"]) for t in rect["targets"]]
# print("hello")
# print(tip_pos)
# print(targets_px[0])
# import numpy as np

import numpy as np
import cv2
import time

def apply_homography(H, pt_xy):
    x, y = pt_xy
    p = np.array([x, y, 1.0], dtype=float)
    q = H @ p
    q /= q[2]
    return np.array([q[0], q[1]], dtype=float)  # mm on checkerboard plane

def px_to_beam_mm(pt_px, H_img_to_mm, depth_scale):
    pt_mm_board = apply_homography(H_img_to_mm, pt_px)
    return pt_mm_board * float(depth_scale)  # mm on beam plane

def cam_beam_mm_to_mpc_m(pt_mm_beam_rel, swap_xy=True, flip_x=False, flip_y=False):
    Xmm, Ymm = float(pt_mm_beam_rel[0]), float(pt_mm_beam_rel[1])
    if swap_xy:
        x_mpc_mm, y_mpc_mm = Ymm, Xmm
    else:
        x_mpc_mm, y_mpc_mm = Xmm, Ymm
    if flip_x:
        x_mpc_mm = -x_mpc_mm
    if flip_y:
        y_mpc_mm = -y_mpc_mm
    return np.array([x_mpc_mm, y_mpc_mm], dtype=float) * 1e-3  # meters

def build_xref_seq(target_mpc, Np):
    return np.tile(np.asarray(target_mpc, dtype=float).reshape(1, 2), (Np, 1))
def apply_robot_command_from_p(robo, p_next):
    """
    p_next: [gamma(rad), beta(rad), rho(m), L(m)]
    Adjust this mapping to match your robot kinematics and conventions.
    """
    gamma_rad, beta_rad, rho_m, L_m = map(float, p_next)

    # 1) Advance
    advancer_go(L_m * 1000.0)  # meters -> mm

    # 2) Set gamma (your get_point expects degrees in your example)
    gamma_deg = np.rad2deg(gamma_rad)
    new_pose = get_point(0, gamma_deg)
    robo.moveL(new_pose)

    # 3) Set beta (your example uses joint 5 in radians)
    joint_now = robo.get_joints()
    joint_now[5] -= beta_rad
    robo.moveJ(joint_now)
import time
import numpy as np
import cv2

def run_closed_loop_to_target(
    *,
    mpc_xy,
    target_px,
    H_img_to_mm,
    depth_scale,
    swap_xy=True,
    flip_x=False,
    flip_y=False,
    tol_m=1.5e-3,            # 1.5 mm tolerance in MPC meters
    max_iters=3,
    sleep_s=0.1,
    capture_filename="focused_image.jpg",
    show_debug=False,

    # --- NEW: plotting/debug outputs ---
    reference_image=None,    # image to plot on (e.g., rect["image_file"])
    targets_px=None,         # list of all targets for plotting
    plot_at_end=True,
    annotate_plot=True,
    draw_lines=True,
):
    """
    Closed-loop:
      capture -> detect base/tip -> compute tip/target in MPC coords -> MPC step -> robot command
    Adds:
      - rich debug prints
      - returns a debug dict
      - plots all targets and final tips at the end if enabled
    """

    robo = URRtde(ROBOT_IP)

    # Collect history for debugging/plotting
    hist = {
        "img_files": [],
        "tip_px": [],
        "base_px": [],
        "tip_mm_rel": [],
        "tgt_mm_rel": [],
        "tip_mpc": [],
        "tgt_mpc": [],
        "err_m": [],
        "err_norm_m": [],
        "angle_deg": [],
        "roi_box": [],
        "status": [],
        "infeasible": [],
        "u0": [],
        "p_next": [],
        "p_now": [],
        "d": [],
    }

    last_tip_px = None  # for plotting "final" tips; in a multi-target run you would collect per target

    try:
        for k in range(max_iters):
            t0 = time.time()

            # ---- 1) capture ----
            img_file = new_capture(filename=capture_filename)
            hist["img_files"].append(img_file)

            image = cv2.imread(img_file)
            if image is None:
                print(f"[WARN] Iter {k}: Failed to read captured image '{img_file}'. Skipping.")
                time.sleep(sleep_s)
                continue

            # ---- 2) detect base/tip ----
            pt1, pt2, angle_deg, roi_box = detect_red_points_and_angle(img_file, show=False, use_roi=True)

            # Convention in your detector: sorted by y, so pt1 is higher, pt2 is lower.
            tip_px = tuple(pt1)
            base_px = tuple(pt2)

            hist["tip_px"].append(tip_px)
            hist["base_px"].append(base_px)
            hist["angle_deg"].append(float(angle_deg))
            hist["roi_box"].append(roi_box)

            last_tip_px = tip_px

            # Basic detection sanity checks
            if any(v is None for v in tip_px) or any(v is None for v in base_px):
                print(f"[WARN] Iter {k}: tip/base invalid: tip={tip_px}, base={base_px}")
                time.sleep(sleep_s)
                continue

            # If base and tip are extremely close, segmentation is probably wrong
            px_sep = float(np.linalg.norm(np.array(tip_px, float) - np.array(base_px, float)))
            if px_sep < 5:
                print(f"[WARN] Iter {k}: tip/base too close in pixels ({px_sep:.2f}px). Likely bad detection.")
                time.sleep(sleep_s)
                continue

            # ---- 3) pixels -> beam-plane mm ----
            base_mm = px_to_beam_mm(base_px, H_img_to_mm, depth_scale)
            tip_mm  = px_to_beam_mm(tip_px,  H_img_to_mm, depth_scale)
            tgt_mm  = px_to_beam_mm(target_px, H_img_to_mm, depth_scale)

            # base-relative (base becomes 0,0)
            tip_mm_rel = tip_mm - base_mm
            tgt_mm_rel = tgt_mm - base_mm

            hist["tip_mm_rel"].append(tip_mm_rel.copy())
            hist["tgt_mm_rel"].append(tgt_mm_rel.copy())

            # ---- 4) map to MPC meters ----
            tip_mpc    = cam_beam_mm_to_mpc_m(tip_mm_rel, swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)
            target_mpc = cam_beam_mm_to_mpc_m(tgt_mm_rel, swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)

            hist["tip_mpc"].append(tip_mpc.copy())
            hist["tgt_mpc"].append(target_mpc.copy())

            # Validate numeric health (NaNs kill MPC)
            if not np.all(np.isfinite(tip_mpc)) or not np.all(np.isfinite(target_mpc)):
                print(f"[WARN] Iter {k}: Non-finite MPC coords. tip_mpc={tip_mpc}, target_mpc={target_mpc}")
                time.sleep(sleep_s)
                continue

            # ---- 5) stop condition ----
            err = target_mpc - tip_mpc
            err_norm = float(np.linalg.norm(err))
            hist["err_m"].append(err.copy())
            hist["err_norm_m"].append(err_norm)

            if show_debug:
                # Also report error in mm in beam-plane coordinates (base-relative, before axis swap)
                err_mm_rel = (tgt_mm_rel - tip_mm_rel)
                err_norm_mm = float(np.linalg.norm(err_mm_rel))
                print(
                    f"[{k}] px_sep={px_sep:.1f}px angle={angle_deg:.2f}deg "
                    f"tip_px={tip_px} base_px={base_px} target_px={target_px}\n"
                    f"    tip_mm_rel={tip_mm_rel} tgt_mm_rel={tgt_mm_rel} err_mm_rel={err_mm_rel} |err_mm|={err_norm_mm:.2f}\n"
                    f"    tip_mpc={tip_mpc} target_mpc={target_mpc} err_m={err} |err_m|={err_norm:.6f}"
                )

            if err_norm <= tol_m:
                print(f"[DONE] Reached target within {tol_m*1000:.1f} mm at iter {k}.")
                break

            # ---- 6) MPC step ----
            xref_seq = build_xref_seq(target_mpc, mpc_xy.Np)

            # Extra debug: check internal parameter bounds before solve
            if show_debug:
                p_now = mpc_xy.p.copy()
                inside = np.all(p_now >= mpc_xy.p_min - 1e-12) and np.all(p_now <= mpc_xy.p_max + 1e-12)
                print(f"    p_now(rad)={p_now} inside_bounds={inside}")

            p_next, x_next, info = mpc_xy.step(xref_seq, x_meas=tip_mpc)

            hist["status"].append(info.get("status", ""))
            hist["infeasible"].append(int(info.get("infeasible", 1)))
            hist["u0"].append(np.asarray(info.get("u0", np.zeros(4)), float).copy())
            hist["p_next"].append(np.asarray(p_next, float).copy())
            hist["p_now"].append(np.asarray(info.get("p_now", mpc_xy.p), float).copy())
            hist["d"].append(np.asarray(info.get("d", np.zeros(2)), float).copy())

            if info["infeasible"] == 1:
                print(f"[WARN] Iter {k}: MPC infeasible, status={info['status']}. No motion applied.")
                time.sleep(sleep_s)
                continue

            if show_debug:
                p_disp = p_next.copy()
                p_disp[:2] = np.rad2deg(p_disp[:2])
                print(f"    u0={info['u0']}")
                print(f"    p_next(deg,deg,m,m)={p_disp}  d={info.get('d')}")

            # ---- 7) apply to robot ----
            apply_robot_command_from_p(robo, p_next)

            # ---- 8) timing ----
            elapsed = time.time() - t0
            if show_debug:
                print(f"    iter_time={elapsed:.3f}s\n")

            time.sleep(sleep_s)

        else:
            print(f"[STOP] Max iterations reached ({max_iters}).")

    finally:
        robo.shutdown()

    # ---- END-OF-RUN: optional plotting ----
    # For a single-target run, plot "all targets" and the final detected tip only.
    if plot_at_end and reference_image is not None and targets_px is not None and len(targets_px) > 0:
        final_tips_px = [last_tip_px] * len(targets_px) if last_tip_px is not None else [(np.nan, np.nan)] * len(targets_px)

        # Compute an error per target in mm using the last detected tip/base frame
        errors_mm = None
        if last_tip_px is not None and len(hist["tip_mm_rel"]) > 0 and len(hist["tgt_mm_rel"]) > 0:
            # Use the last computed rel coordinates
            tip_mm_rel_last = hist["tip_mm_rel"][-1]
            base_px_last = hist["base_px"][-1]
            base_mm_last = px_to_beam_mm(base_px_last, H_img_to_mm, depth_scale)

            errors_mm = []
            for tpx in targets_px:
                t_mm = px_to_beam_mm(tuple(tpx), H_img_to_mm, depth_scale)
                t_mm_rel = t_mm - base_mm_last
                e_mm = float(np.linalg.norm(t_mm_rel - tip_mm_rel_last))
                errors_mm.append(e_mm)

        plot_all_targets_and_tips_on_image(
            image_filename=reference_image,
            targets_px=targets_px,
            tips_px=final_tips_px,
            errors_mm=errors_mm,
            title="Rectangle targets (yellow) vs final tip (red)",
            draw_lines=draw_lines,
            annotate=annotate_plot,
        )

    return hist

# --- build MPC as you already do ---
forward_tip = make_forward_fn(beam_params.mag, beam_params.A_cs, beam_params.E, beam_params.I, mag_params.mag_epm)
Jxy_fn = make_jac_fn(forward_tip, eps)

mpc_xy = mpc_controller_tipxy_LTI(
    Jxy_fn=Jxy_fn,
    forward_tip_fn=forward_tip,
    dt=0.2,
    Np=10,
    w_xy=(100.0, 100.0),
    w_u=(1e-4, 1e-4, 1e-2, 1e-2),
    w_du=(1e-3, 1e-3, 1e-2, 1e-2),
    u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
    p_min=p_min,
    p_max=p_max,
    N_sqp=3,
    use_offset_free=True,
    d_alpha=0.15,
)
mpc_xy.set_initial_params(gamma0=0.0, beta0=0.0, rho0=0.11, L0=0.055)

# --- get target and calibration once ---
rect = compute_beam_targets_from_clicked_points(
    image_filename="focused_image.jpg",
    use_roi=True,
    show=True,
    n_points=None,
    csv_path="clicked_targets.csv",
)

H_img_to_mm = rect["H_img_to_mm"]
depth_scale = rect["depth_scale"]
targets_px = [tuple(t["target_px"]) for t in rect["targets"]]
target_px = targets_px[0]

reference_image = rect["image_file"]
targets_px = [tuple(t["target_px"]) for t in rect["targets"]]
target_px = targets_px[0]

hist = run_closed_loop_to_target(
    mpc_xy=mpc_xy,
    target_px=target_px,
    H_img_to_mm=H_img_to_mm,
    depth_scale=depth_scale,
    swap_xy=True,
    flip_x=False,
    flip_y=False,
    tol_m=1.5e-3,
    max_iters=3,
    sleep_s=0.15,
    show_debug=True,
    reference_image=reference_image,
    targets_px=targets_px,
    plot_at_end=True,
)

