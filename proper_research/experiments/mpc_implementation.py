    
import matplotlib.pyplot as plt
import numpy as np
import time
import traceback
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from proper_research.parameters import ROBOT_IP, default_beam_params, default_magnet_params
from proper_research.vision.camera import new_capture, compute_signed_angle, measure_beam_angle_deg
from proper_research.vision.measure_length import detect_red_markers_in_roi
from proper_research.advancer_unit.advancer_control import advancer_go
from proper_research.vision.click_target import (plot_all_targets_and_tips_on_image, compute_beam_targets_from_clicked_points)
import cv2
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

def angle_from_points_deg(base_px, p2_px):
    base = np.array(base_px, dtype=np.float32)
    p2   = np.array(p2_px,   dtype=np.float32)
    v = p2 - base
    reference = np.array([0.0, 1.0], dtype=np.float32)  # "down" in image coords
    return compute_signed_angle(reference, v)


def _fmt(v, nd=4):
    v = np.asarray(v, dtype=float).ravel()
    return "[" + ", ".join(f"{x:.{nd}f}" for x in v) + "]"

def _assert_ok(cond, msg):
    if not cond:
        raise RuntimeError(msg)

def run_closed_loop_to_target(
    *,
    mpc_xy,
    target_px,
    H_img_to_mm,
    depth_scale,
    swap_xy=True,
    flip_x=False,
    flip_y=False,
    tol_m=1.5e-3,
    max_iters=3,
    sleep_s=0.1,
    capture_filename="focused_image.jpg",
    show_debug=False,
    reference_image=None,
    targets_px=None,
    plot_at_end=True,
    annotate_plot=True,
    draw_lines=True,

    # --- NEW debug knobs ---
    debug_dir=".",              # where to write debug images
    write_debug_images=True,
    raise_on_exception=False,   # set True if you want it to stop at first error
):
    """
    Closed-loop:
      capture -> detect base/mag_start/tip -> compute tip/target in MPC coords
      -> MPC step -> robot command

    Adds strong debugging to localize failures.
    """

    # -------------------------
    # PRE-FLIGHT CHECKS
    # -------------------------
    _assert_ok(mpc_xy is not None, "mpc_xy is None. Pass an MPC instance.")
    _assert_ok(getattr(mpc_xy, "p", None) is not None,
               "mpc_xy.p is None. You must call mpc_xy.set_initial_params(...) before running.")
    _assert_ok(np.asarray(H_img_to_mm).shape == (3, 3),
               f"H_img_to_mm must be 3x3. Got shape {np.asarray(H_img_to_mm).shape}")
    _assert_ok(np.isfinite(depth_scale) and float(depth_scale) > 0.0,
               f"depth_scale invalid: {depth_scale}")

    # Confirm target shape
    target_px = tuple(target_px)
    _assert_ok(len(target_px) == 2, f"target_px must be (x,y). Got {target_px}")

    if show_debug:
        print("[PRE] depth_scale:", depth_scale)
        print("[PRE] H_img_to_mm:\n", np.asarray(H_img_to_mm))
        print("[PRE] MPC p_init:", _fmt(mpc_xy.p, nd=6))
        print("[PRE] swap_xy/flip_x/flip_y:", swap_xy, flip_x, flip_y)
        print("[PRE] target_px:", target_px)

    robo = URRtde(ROBOT_IP)

    hist = {
        "img_files": [],
        "base_px": [],
        "mag_start_px": [],
        "tip_px": [],
        "roi_box": [],
        "angle_deg": [],
        "base_mm": [],
        "mag_mm": [],
        "tip_mm": [],
        "tgt_mm": [],
        "tip_mm_rel": [],
        "tgt_mm_rel": [],
        "tip_mpc": [],
        "tgt_mpc": [],
        "err_m": [],
        "err_norm_m": [],
        "status": [],
        "infeasible": [],
        "u0": [],
        "p_now": [],
        "p_next": [],
        "d": [],
        "exceptions": [],
    }

    last_tip_px = None

    try:
        for k in range(max_iters):
            t0 = time.time()
            if show_debug:
                print("\n" + "=" * 72)
                print(f"[ITER {k}] start")

            try:
                # ---- 1) capture ----
                img_file = new_capture(filename=capture_filename)
                hist["img_files"].append(img_file)

                image = cv2.imread(img_file)
                if image is None:
                    msg = f"[WARN] Iter {k}: Failed to read captured image '{img_file}'."
                    print(msg)
                    hist["exceptions"].append(msg)
                    time.sleep(sleep_s)
                    continue

                h, w = image.shape[:2]
                if show_debug:
                    print(f"[ITER {k}] captured {img_file} (H,W)=({h},{w})")

                # ---- 2) detect base/mag_start/tip ----
                # IMPORTANT: detector returns base, mag_start, tip
                base_px, mag_start_px, tip_px, roi_box = detect_red_markers_in_roi(
                    image, use_roi=True, expected_markers=3, show_debug=False
                )

                base_px = tuple(base_px)
                mag_start_px = tuple(mag_start_px)
                tip_px = tuple(tip_px)

                hist["base_px"].append(base_px)
                hist["mag_start_px"].append(mag_start_px)
                hist["tip_px"].append(tip_px)
                hist["roi_box"].append(roi_box)
                beam_angle_deg   = angle_from_points_deg(base_px, tip_px)*-1
                target_angle_deg = angle_from_points_deg(base_px, target_px)*-1

                angle_err_deg = target_angle_deg - beam_angle_deg
                # wrap to [-90, 90] if you want consistent
                if angle_err_deg > 90: angle_err_deg -= 180
                if angle_err_deg < -90: angle_err_deg += 180

                print(f"[DBG] beam_angle={beam_angle_deg:.2f}deg, target_angle={target_angle_deg:.2f}deg, angle_err={angle_err_deg:.2f}deg")

                last_tip_px = tip_px

                # Sanity checks (ordering + separation)
                px_sep_bt = float(np.linalg.norm(np.array(tip_px) - np.array(base_px)))
                px_sep_bm = float(np.linalg.norm(np.array(mag_start_px) - np.array(base_px)))
                px_sep_mt = float(np.linalg.norm(np.array(tip_px) - np.array(mag_start_px)))

                if show_debug:
                    print(f"[ITER {k}] base_px={base_px} mag_px={mag_start_px} tip_px={tip_px}")
                    print(f"[ITER {k}] px distances: |B-T|={px_sep_bt:.2f}, |B-M|={px_sep_bm:.2f}, |M-T|={px_sep_mt:.2f}")

                if px_sep_bt < 5 or px_sep_bm < 5 or px_sep_mt < 5:
                    msg = f"[WARN] Iter {k}: markers too close; detection likely bad."
                    print(msg)
                    hist["exceptions"].append(msg)
                    time.sleep(sleep_s)
                    continue

                # Optional: write overlay image
                if show_debug and write_debug_images:
                    dbg = image.copy()
                    if roi_box is not None:
                        x, y, rw, rh = roi_box
                        cv2.rectangle(dbg, (x, y), (x+rw, y+rh), (0, 255, 255), 2)

                    for (pt, col, lab) in [
                        (base_px,      (255, 0, 0),   "B"),
                        (mag_start_px, (0, 255, 255), "M"),
                        (tip_px,       (0, 0, 255),   "T"),
                        (target_px,    (0, 255, 0),   "X"),
                    ]:
                        cv2.circle(dbg, (int(pt[0]), int(pt[1])), 7, col, -1)
                        cv2.putText(dbg, lab, (int(pt[0])+8, int(pt[1])+8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                    out_path = f"{debug_dir}/debug_iter_{k}.png"
                    cv2.imwrite(out_path, dbg)
                    print(f"[ITER {k}] wrote {out_path}")

                # ---- 3) pixels -> beam-plane mm ----
                base_mm = px_to_beam_mm(base_px, H_img_to_mm, depth_scale)
                mag_mm  = px_to_beam_mm(mag_start_px, H_img_to_mm, depth_scale)
                tip_mm  = px_to_beam_mm(tip_px,  H_img_to_mm, depth_scale)
                tgt_mm  = px_to_beam_mm(target_px, H_img_to_mm, depth_scale)

                hist["base_mm"].append(base_mm.copy())
                hist["mag_mm"].append(mag_mm.copy())
                hist["tip_mm"].append(tip_mm.copy())
                hist["tgt_mm"].append(tgt_mm.copy())

                tip_mm_rel = tip_mm - base_mm
                tgt_mm_rel = tgt_mm - base_mm

                hist["tip_mm_rel"].append(tip_mm_rel.copy())
                hist["tgt_mm_rel"].append(tgt_mm_rel.copy())

                if show_debug:
                    print(f"[ITER {k}] base_mm={_fmt(base_mm,3)} tip_mm={_fmt(tip_mm,3)} tgt_mm={_fmt(tgt_mm,3)}")
                    print(f"[ITER {k}] tip_mm_rel={_fmt(tip_mm_rel,3)} tgt_mm_rel={_fmt(tgt_mm_rel,3)}")

                if not (np.all(np.isfinite(tip_mm_rel)) and np.all(np.isfinite(tgt_mm_rel))):
                    msg = f"[WARN] Iter {k}: non-finite mm coords."
                    print(msg)
                    hist["exceptions"].append(msg)
                    time.sleep(sleep_s)
                    continue

                # ---- 4) map to MPC meters ----
                tip_mpc    = cam_beam_mm_to_mpc_m(tip_mm_rel, swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)
                target_mpc = cam_beam_mm_to_mpc_m(tgt_mm_rel, swap_xy=swap_xy, flip_x=flip_x, flip_y=flip_y)

                hist["tip_mpc"].append(tip_mpc.copy())
                hist["tgt_mpc"].append(target_mpc.copy())

                if show_debug:
                    print(f"[ITER {k}] tip_mpc={_fmt(tip_mpc,6)} target_mpc={_fmt(target_mpc,6)}")

                if not np.all(np.isfinite(tip_mpc)) or not np.all(np.isfinite(target_mpc)):
                    msg = f"[WARN] Iter {k}: Non-finite MPC coords."
                    print(msg)
                    hist["exceptions"].append(msg)
                    time.sleep(sleep_s)
                    continue

                # ---- 5) stop condition ----
                err = target_mpc - tip_mpc
                err_norm = float(np.linalg.norm(err))
                hist["err_m"].append(err.copy())
                hist["err_norm_m"].append(err_norm)

                if show_debug:
                    print(f"[ITER {k}] err_m={_fmt(err,6)} |err|={err_norm*1000.0:.3f} mm")

                if err_norm <= tol_m:
                    print(f"[DONE] Reached target within {tol_m*1000:.1f} mm at iter {k}.")
                    break

                # ---- 6) MPC step ----
                xref_seq = build_xref_seq(target_mpc, mpc_xy.Np)

                # Log controller internal state BEFORE solve
                if show_debug:
                    p_now = np.asarray(mpc_xy.p, float).copy()
                    inside = np.all(p_now >= mpc_xy.p_min - 1e-12) and np.all(p_now <= mpc_xy.p_max + 1e-12)
                    print(f"[ITER {k}] p_now(rad)={_fmt(p_now,6)} inside_bounds={inside}")

                p_next, x_next, info = mpc_xy.step(xref_seq, x_meas=tip_mpc)

                hist["status"].append(info.get("status", ""))
                hist["infeasible"].append(int(info.get("infeasible", 1)))
                hist["u0"].append(np.asarray(info.get("u0", np.zeros(4)), float).copy())
                hist["p_now"].append(np.asarray(info.get("p_now", mpc_xy.p), float).copy())
                hist["p_next"].append(np.asarray(p_next, float).copy())
                hist["d"].append(np.asarray(info.get("d", np.zeros(2)), float).copy())

                if show_debug:
                    p_disp = np.asarray(p_next, float).copy()
                    p_disp[:2] = np.rad2deg(p_disp[:2])
                    print(f"[ITER {k}] MPC status={info.get('status')} infeasible={info.get('infeasible')}")
                    print(f"[ITER {k}] u0={_fmt(info.get('u0', np.zeros(4)),6)}")
                    print(f"[ITER {k}] p_next(deg,deg,m,m)={_fmt(p_disp,6)} d={_fmt(info.get('d', np.zeros(2)),6)}")

                if int(info.get("infeasible", 1)) == 1:
                    msg = f"[WARN] Iter {k}: MPC infeasible, status={info.get('status')}."
                    print(msg)
                    hist["exceptions"].append(msg)
                    time.sleep(sleep_s)
                    continue

                # ---- 7) apply to robot ----
                apply_robot_command_from_p(robo, p_next)

                # ---- 8) timing ----
                elapsed = time.time() - t0
                if show_debug:
                    print(f"[ITER {k}] iter_time={elapsed:.3f}s")
                time.sleep(sleep_s)

            except Exception as e:
                msg = f"[EXC] Iter {k}: {repr(e)}"
                print(msg)
                if show_debug:
                    print(traceback.format_exc())
                hist["exceptions"].append(msg)
                if raise_on_exception:
                    raise
                time.sleep(sleep_s)
                continue

        else:
            print(f"[STOP] Max iterations reached ({max_iters}).")

    finally:
        robo.shutdown()

    # ---- END-OF-RUN plot ----
    if plot_at_end and reference_image is not None and targets_px is not None and len(targets_px) > 0:
        final_tips_px = [last_tip_px] * len(targets_px) if last_tip_px is not None else [(np.nan, np.nan)] * len(targets_px)

        errors_mm = None
        if last_tip_px is not None and len(hist["tip_mm_rel"]) > 0:
            tip_mm_rel_last = hist["tip_mm_rel"][-1]
            base_px_last = hist["base_px"][-1]
            base_mm_last = px_to_beam_mm(base_px_last, H_img_to_mm, depth_scale)

            errors_mm = []
            for tpx in targets_px:
                t_mm = px_to_beam_mm(tuple(tpx), H_img_to_mm, depth_scale)
                t_mm_rel = t_mm - base_mm_last
                errors_mm.append(float(np.linalg.norm(t_mm_rel - tip_mm_rel_last)))

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

def angle_from_points_deg(p0_px, p1_px):
    p0 = np.array(p0_px, dtype=np.float32)
    p1 = np.array(p1_px, dtype=np.float32)
    v = p1 - p0
    reference = np.array([0.0, 1.0], dtype=np.float32)  # down
    return compute_signed_angle(reference, v)

# --- build MPC as you already do ---
forward_tip = make_forward_fn(beam_params.mag, beam_params.A_cs, beam_params.E, beam_params.I, mag_params.mag_epm)
Jxy_fn = make_jac_fn(forward_tip, eps, debug=False, fail_policy="nan")

mpc_xy = mpc_controller_tipxy_LTI(
    Jxy_fn=Jxy_fn,
    forward_tip_fn=forward_tip,
    dt=0.2,
    Np=10,
    w_xy=(100.0, 100.0),
    w_u=(1e-4, 1e-1, 1e-2, 1e-2),
    w_du=(1e-3, 1e-3, 1e-2, 1e-2),
    u_max=(np.deg2rad(40), np.deg2rad(40), 0.03, 0.01),
    p_min=p_min,
    p_max=p_max,
    N_sqp=3,
    use_offset_free=True,
    d_alpha=0.15,
)
mpc_xy.set_initial_params(gamma0=np.deg2rad(5.0), beta0=np.deg2rad(5.0), rho0=0.13, L0=0.05)

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

