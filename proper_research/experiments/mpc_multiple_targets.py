    
import matplotlib.pyplot as plt
import numpy as np
import time
from proper_research.robot.ur_rtde import URRtde
from proper_research.robot.transformations import get_point
from beam_direction_magnetisation.post_processing.post_processing import plot_target_and_final_tip_on_image
from proper_research.vision.vision_w_arco import pick_multiple_target_points_world ,measure_lengths_mm
from proper_research.parameters import ROBOT_IP, default_beam_params, default_magnet_params
from proper_research.advancer_unit.advancer_control import advancer_go
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile

import cv2
from proper_research.control.mpc_3d import mpc_controller_tipxy_LTI, forward_cosserat_from_pose_ur_rotvec_L, numerical_jacobian_tip_xyz_pose, wire_len_from_L
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T, T_to_p_quat_wxyz
import time
import traceback

RTDE_EOF_MARKERS = (
    "End of file",
    "asio.misc:2",
    "RTDEReceiveInterface",
)

def is_rtde_disconnect(err: Exception) -> bool:
    s = str(err)
    return any(k in s for k in RTDE_EOF_MARKERS)

def safe_shutdown(robo):
    if robo is None:
        return
    try:
        robo.shutdown()
    except Exception:
        pass

def connect_robo(robot_ip, *, tries=3, delay_s=0.3):
    """Create a fresh URRtde and sanity-check it."""
    last_err = None
    for i in range(tries):
        try:
            r = URRtde(robot_ip)
            # sanity check: forces RTDE handshake + data
            _ = r.get_joints()
            return r
        except Exception as e:
            last_err = e
            safe_shutdown(r if "r" in locals() else None)
            time.sleep(delay_s * (i + 1))
    raise RuntimeError(f"Failed to connect to robot after {tries} tries: {last_err}")

def with_robot_retry(
    robo,
    robot_ip,
    fn,
    *,
    label="op",
    retries=2,
    delay_s=0.2,
    fatal=False,
):
    """
    Runs fn(robo). If RTDE EOF occurs:
      - logs it
      - shutdown + reconnect
      - retries fn
    Returns: (result, robo)  where robo may be a new connection
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn(robo), robo
        except Exception as e:
            last_err = e
            if not is_rtde_disconnect(e):
                # not the EOF error; re-raise
                raise

            print(f"[RTDE] disconnect during {label} (attempt {attempt+1}/{retries+1}): {e}")
            # optional: print stack once
            # traceback.print_exc()

            safe_shutdown(robo)
            time.sleep(delay_s * (attempt + 1))
            robo = connect_robo(robot_ip, tries=3, delay_s=0.3)

    # ran out of retries
    if fatal:
        raise RuntimeError(f"[RTDE] failed {label} after retries") from last_err
    print(f"[RTDE] giving up on {label} for now; continuing.")
    return None, robo

mag_params = default_magnet_params()
beam_params = default_beam_params()
def world_xy_to_pixel(target_world_xy_m, dbg):
    """
    target_world_xy_m: (2,) in meters in your world frame
    dbg: dict returned by measure_lengths_mm (must include base_mm, rvec,tvec,K,dist, A_2x2, BASE_WORLD_M, plane_z_mm)
    Returns (u,v) pixel tuple (float,float)
    """
    target_world_xy_m = np.asarray(target_world_xy_m, dtype=np.float64).reshape(2,)

    base_mm = np.asarray(dbg["base_mm"], dtype=np.float64).reshape(2,)
    A = np.asarray(dbg["A_2x2"], dtype=np.float64).reshape(2,2)
    BASE_WORLD_M = np.asarray(dbg["BASE_WORLD_M"], dtype=np.float64).reshape(2,)
    plane_z_mm = float(dbg["plane_z_mm"])

    # invert mapping: p_mm = base_mm + 1000 * A^{-1} * (p_world - BASE_WORLD)
    d_world = (target_world_xy_m - BASE_WORLD_M).reshape(2,1)
    d_board_m = np.linalg.solve(A, d_world).reshape(2,)     # A^{-1} * d_world
    target_mm = base_mm + 1000.0 * d_board_m                # meters -> mm

    # 3D point in board coordinates (mm)
    Pw = np.array([[target_mm[0], target_mm[1], plane_z_mm]], dtype=np.float64)  # shape (1,3)

    K = np.asarray(dbg["K"], dtype=np.float64)
    dist = np.asarray(dbg["dist"], dtype=np.float64).reshape(-1,1)
    rvec = np.asarray(dbg["rvec"], dtype=np.float64).reshape(3,1)
    tvec = np.asarray(dbg["tvec"], dtype=np.float64).reshape(3,1)

    # project board point -> pixel
    uv, _ = cv2.projectPoints(Pw, rvec, tvec, K, dist)
    u, v = uv.reshape(2,)
    return (float(u), float(v))
def make_get_tip_xyz_meas_from_camera(*, cam_index=0, use_roi=True, z_target=-0.1, show=False, target_world_xyz=None):
    target_world_xyz = None if target_world_xyz is None else np.asarray(target_world_xyz, float).reshape(3,)

    def _get_tip_xyz_meas():
        res = measure_lengths_mm(
            image_filename="focused_image.jpg",
            use_roi=use_roi,
            show=show,
            cam_index=cam_index
        )

        tip_xy = np.array(res["tip_world_m"], dtype=float).reshape(2,)
        tip_xyz = np.array([tip_xy[0], tip_xy[1], float(z_target)], dtype=float)

        # compute target_px if target provided
        if target_world_xyz is not None:
            res["target_px"] = world_xy_to_pixel(target_world_xyz[:2], res)

        return tip_xyz, res

    return _get_tip_xyz_meas



def make_straight_path_xyz(x_start, x_target, n_points=20):
    x_start = np.asarray(x_start, float).reshape(3,)
    x_target = np.asarray(x_target, float).reshape(3,)
    return np.linspace(x_start, x_target, int(n_points))  # (n_points, 3)

def build_xref_from_path_xyz(path_xyz, idx, Np):
    """
    path_xyz: (N,3)
    idx: current waypoint index
    returns (Np,3) by lookahead with clamping at end
    """
    path_xyz = np.asarray(path_xyz, float)
    N = path_xyz.shape[0]
    idxs = np.clip(np.arange(idx, idx + Np), 0, N - 1)
    return path_xyz[idxs]

def nearest_path_index(path_xyz, tip_xyz, last_idx=0, window=10):
    """
    Finds the nearest waypoint index to tip_xyz, searching forward from last_idx.
    window limits how far ahead we search (prevents jumping backwards).
    """
    path_xyz = np.asarray(path_xyz, float)
    tip_xyz = np.asarray(tip_xyz, float).reshape(3,)

    i0 = int(last_idx)
    i1 = min(i0 + int(window), path_xyz.shape[0]-1)
    seg = path_xyz[i0:i1+1]
    d2 = np.sum((seg - tip_xyz[None, :])**2, axis=1)
    return i0 + int(np.argmin(d2))

def apply_robot_command_from_p_pose7(robo, p_next, *, speed=None, accel=None):
    """
    p_next: [x,y,z, rx,ry,rz, L]
    - move robot TCP to pose6 = p_next[:6]
    - set advancer to L in meters
    """
    p_next = np.asarray(p_next, float).ravel()
    assert p_next.size == 7

    pose6 = p_next[:6].copy()
    L_m   = float(p_next[6])
    pose6[2]+=0.25
    # 1) set insertion length (meters -> mm)
    advancer_go(L_m * 1000.0)

    # 2) command UR to magnet pose
    # adjust to your URRtde API: moveL(pose, speed=?, accel=?)
    if speed is None and accel is None:
        robo.moveL(pose6)
    else:
        robo.moveL(pose6, speed=speed, accel=accel)
def preview_square_world(center_xy, square_xy, title="Square preview (world XY)"):
    c = np.asarray(center_xy, float).reshape(2,)
    pts = np.asarray(square_xy, float).reshape(-1, 2)

    # close the loop for plotting edges
    pts_closed = np.vstack([pts, pts[0]])

    plt.figure()
    plt.plot(pts_closed[:,0], pts_closed[:,1], marker="o")
    plt.scatter([c[0]], [c[1]], marker="x")
    for i, p in enumerate(pts):
        plt.text(p[0], p[1], f"{i}")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.show()

# usage
import numpy as np

def rect_targets_from_center_xy(center_xy, width_m=0.10, height_m=0.05, yaw_rad=0.0):
    """
    center_xy: (2,) world meters
    width_m, height_m: rectangle side lengths in meters
    yaw_rad: rotation of the rectangle in world frame (0 => axis-aligned)
    returns: list of 4 (2,) points (corners), in order around the perimeter
    """
    c = np.asarray(center_xy, float).reshape(2,)
    hx = 0.5 * float(width_m)
    hy = 0.5 * float(height_m)

    # axis-aligned rectangle corners around origin
    corners = np.array([
        [-hx, -hy],
        [ hx, -hy],
        [ hx,  hy],
        [-hx,  hy],
    ], dtype=float)

    # rotate by yaw about center
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[cy, -sy],
                  [sy,  cy]], dtype=float)

    corners_rot = (corners @ R.T) + c[None, :]
    return [corners_rot[i].copy() for i in range(4)]

def square_targets_from_center_xy(center_xy, side_m=0.10, yaw_rad=0.0):
    """
    center_xy: (2,) world meters
    side_m: square side length in meters
    yaw_rad: rotation of the square in world frame (0 => axis-aligned)
    returns: list of 4 (2,) points, corners in order
    """
    c = np.asarray(center_xy, float).reshape(2,)
    a = 0.5 * float(side_m)

    # axis-aligned corners around origin
    corners = np.array([
        [-a, -a],
        [ a, -a],
        [ a,  a],
        [-a,  a],
    ], dtype=float)

    # rotate by yaw about center
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[cy, -sy],
                  [sy,  cy]], dtype=float)

    corners_rot = (corners @ R.T) + c[None, :]
    return [corners_rot[i].copy() for i in range(4)]


def _fmt(v, nd=4):
    v = np.asarray(v, dtype=float).ravel()
    return "[" + ", ".join(f"{x:.{nd}f}" for x in v) + "]"
def print_mpc_rollout(info, target_mpc, *, k=None, max_rows=None):
    """
    info: dict returned by mpc.step(...)
    target_mpc: (2,) target in MPC meters
    """
    X = np.asarray(info.get("X_pred", []), dtype=float)   # (Np,2)
    U = np.asarray(info.get("U_seq", []), dtype=float)    # (Np,4)
    if X.size == 0 or U.size == 0:
        print("[MPC] No rollout available in info (infeasible or not returned).")
        return

    Np = X.shape[0]
    target = np.asarray(target_mpc, dtype=float).reshape(1, 2)

    if max_rows is None:
        max_rows = Np

    print("-" * 72)
    hdr = f"[MPC rollout]{'' if k is None else f' iter={k}'} status={info.get('status')} infeasible={info.get('infeasible')}"
    print(hdr)
    print(f"  u0 = {_fmt(info.get('u0', np.zeros(4)), 6)}")
    print(f"  d  = {_fmt(info.get('d', np.zeros(2)), 6)}")
    print("  step |   x_pred(m)        y_pred(m)      | err_norm(mm) |  u(g,b,rho,L) ")
    print("  -----+-------------------------------+-------------+------------------------")

    for i in range(min(Np, max_rows)):
        xi = X[i, :]
        ui = U[i, :]
        if np.any(~np.isfinite(xi)):
            err_mm = np.nan
        else:
            err_mm = 1e3 * float(np.linalg.norm(xi - target.reshape(2,)))

        print(f"  {i:>4d} | {xi[0]:>10.6f} {xi[1]:>10.6f} | {err_mm:>11.3f} | {ui[0]:>8.5f} {ui[1]:>8.5f} {ui[2]:>8.5f} {ui[3]:>8.5f}")

    if Np > max_rows:
        print(f"  ... ({Np-max_rows} more steps)")
    print("-" * 72)
def run_target_list_sequence(
    *,
    mpc,
    robo,
    targets_xyz,               # list of (3,)
    cam_index=0,
    use_roi=True,
    show=False,
    tol_m=1.5e-3,
    max_iters_per_target=3,
    sleep_s=0.15,
    n_path_points=2,
    lookahead_mode="index",
    nearest_window=5,
):
    all_hists = []
    final_hits = []

    for i, tgt in enumerate(targets_xyz):
        print("\n" + "#"*80)
        print(f"[LIST] Target {i}/{len(targets_xyz)-1}: {tgt}")

        seg_hist, seg_last = run_segment_to_target(
            mpc=mpc,
            robo=robo,
            target_xyz=tgt,
            cam_index=cam_index,
            use_roi=use_roi,
            show=show,
            tol_m=tol_m,
            max_iters=max_iters_per_target,
            sleep_s=sleep_s,
            n_path_points=n_path_points,
            lookahead_mode=lookahead_mode,
            nearest_window=nearest_window,
        )

        all_hists.append(seg_hist)
        final_hits.append(dict(
            target_xyz=tgt,
            tip_xyz=seg_last.get("tip", None),
            err_mm=seg_last.get("err_mm", None),
            target_px=seg_last.get("target_px", None),
            tip_px=seg_last.get("tip_px", None),
            image_file=seg_last.get("dbg", {}).get("image_file", None),
            dbg=seg_last.get("dbg", None),
        ))

    return all_hists, final_hits


def run_closed_loop_pose7_to_target_with_path(
    *,
    mpc_xyz,
    get_tip_xyz_meas,          # callable -> (tip_xyz_ur: (3,), debug_dict)
    target_xyz_ur,             # (3,)
    robo,
    tol_m=1.5e-3,
    max_iters=10,
    sleep_s=0.15,
    show_debug=True,
    speed=None,
    accel=None,

    # path knobs
    n_path_points=10,          # how many sub-targets
    lookahead_mode="index",    # "index" or "nearest"
    nearest_window=5,         # search window if lookahead_mode="nearest"
):
    """
    Builds a straight-line waypoint path ONCE (from initial measured tip to target).
    Each iteration feeds MPC a horizon of Np waypoints ahead.
    """
    hist = []
    prev_xpred0 = None
    # robo = ensure_robo(robo)
    # --- measure once to define the path start ---
    tip0_xyz, dbg0 = get_tip_xyz_meas()
    tip0_xyz = np.asarray(tip0_xyz, float).reshape(3,)
    target_xyz_ur = np.asarray(target_xyz_ur, float).reshape(3,)

    path_xyz = make_straight_path_xyz(tip0_xyz, target_xyz_ur, n_points=n_path_points)
    path_idx = 0

    if show_debug:
        print(f"[PATH] built {path_xyz.shape[0]} waypoints from tip0 -> target")

    for k in range(max_iters):
        # --- 1) measure ---
        tip_xyz, dbg = get_tip_xyz_meas()
        tip_xyz = np.asarray(tip_xyz, float).reshape(3,)
        pose6_actual, robo = with_robot_retry(
            robo, ROBOT_IP,
            lambda r: r.get_pose(),
            label=f"get_pose(post) iter={k}",
            retries=3,
            delay_s=0.2,
        )
        if pose6_actual is None:
            time.sleep(sleep_s)
            continue
        pose6_model = pose6_actual.copy()
        pose6_model[2] -= 0.25
        mpc_xyz.p = np.r_[pose6_model, float(mpc_xyz.p[6])]
        # --- 2) stop check vs final target ---
        err = target_xyz_ur - tip_xyz
        err_norm = float(np.linalg.norm(err))
        if show_debug:
            print("\n" + "="*72)
            print(f"[ITER {k}] tip={tip_xyz} target={target_xyz_ur} |err|={err_norm*1000:.2f} mm")

        if err_norm <= tol_m:
            print(f"[DONE] reached final target within {tol_m*1000:.1f} mm")
            break

        # optional predicted-vs-measured cross-check
        if prev_xpred0 is not None and np.all(np.isfinite(prev_xpred0)):
            pred_err = tip_xyz - prev_xpred0
            if show_debug:
                print(f"[PREDCHK] ||meas - prev X_pred[0]|| = {np.linalg.norm(pred_err)*1000:.2f} mm")

        # --- 3) advance along path ---
        if lookahead_mode == "index":
            # simple: one waypoint per iteration
            # (you can also advance faster when close; see note below)
            path_idx = min(path_idx + 1, path_xyz.shape[0] - 1)
        elif lookahead_mode == "nearest":
            # robust: snap to closest waypoint ahead of current idx
            path_idx = nearest_path_index(path_xyz, tip_xyz, last_idx=path_idx, window=nearest_window)
        else:
            raise ValueError("lookahead_mode must be 'index' or 'nearest'")

        # --- 4) build horizon from waypoints ---
        xref_seq = build_xref_from_path_xyz(path_xyz, path_idx, mpc_xyz.Np)

        if show_debug:
            print(f"[ITER {k}] path_idx={path_idx}/{path_xyz.shape[0]-1} xref0={xref_seq[0]} xref_last={xref_seq[-1]}")
        # robo = ensure_robo(robo)
        # --- 5) MPC step ---
        p_now = np.asarray(mpc_xyz.p, float).copy()
        p_next, x_used, info = mpc_xyz.step(xref_seq, x_meas=tip_xyz)
        # robo = ensure_robo(robo)
        if show_debug:
            print(f"[ITER {k}] meas tip     = {tip_xyz}")
            print(f"[ITER {k}] MPC x_used   = {x_used}   (should match meas)")
            print(f"[ITER {k}] model x_next = {info.get('x_next_model')}")

        # store one-step prediction for next iteration
        X_pred = info.get("X_pred", None)
        if X_pred is not None:
            X_pred = np.asarray(X_pred, float)
            ok = (X_pred.ndim == 2 and X_pred.shape[1] == 3 and X_pred.shape[0] >= 1)
            prev_xpred0 = X_pred[0].copy() if ok else None
        else:
            prev_xpred0 = None

        infeas = int(info.get("infeasible", 1))
        if show_debug:
            print(f"[ITER {k}] MPC status={info.get('status')} infeasible={infeas}")
            print(f"[ITER {k}] u0={info.get('u0')}")
            print(f"[ITER {k}] p_next={p_next}")

        if infeas:
            print(f"[WARN] infeasible at iter {k}, skipping command")
            time.sleep(sleep_s)
            continue
        # robo = ensure_robo(robo)
        # --- 6) apply robot command ---
        pose6_actual, robo = with_robot_retry(
            robo, ROBOT_IP,
            lambda r: r.get_pose(),
            label=f"get_pose iter={k}",
            retries=3,
            delay_s=0.2,
        )

        # If pose read failed even after retries, skip this iteration safely
        if pose6_actual is None:
            time.sleep(sleep_s)
            continue

        # Command (wrap moveL inside apply_robot_command_from_p_pose7 or wrap the call here)
        def _do_cmd(r):
            apply_robot_command_from_p_pose7(r, p_next)
            return True

        ok, robo = with_robot_retry(
            robo, ROBOT_IP,
            _do_cmd,
            label=f"apply_cmd iter={k}",
            retries=2,
            delay_s=0.3,
        )

        if not ok:
            time.sleep(sleep_s)
            continue
        # robo = ensure_robo(robo)
        pose6_actual, robo = with_robot_retry(
            robo, ROBOT_IP,
            lambda r: r.get_pose(),
            label=f"get_pose(pre) iter={k}",
            retries=3,
            delay_s=0.2,
        )

        if pose6_actual is None:
            time.sleep(sleep_s)
            continue

        pose6_model = pose6_actual.copy()
        pose6_model[2] -= 0.25
        mpc_xyz.p = np.r_[pose6_model, float(mpc_xyz.p[6])]

        print("UR pose actual:", pose6_actual)
        print("MPC pose model:", pose6_model)
        print("MPC p used    :", mpc_xyz.p)
        tip_px = dbg.get("tip_px", None)
        target_px = dbg.get("target_px", None)
        err_mm = 1000.0 * float(np.linalg.norm(target_xyz_ur - tip_xyz))

        hist.append(dict(
            k=k,
            tip=tip_xyz,
            err=err,
            err_mm=err_mm,
            tip_px=tip_px,
            target_px=target_px,
            dbg=dbg,
            p_now=p_now,
            p_next=p_next,
            xref_seq=xref_seq,
            path_idx=path_idx,
            info=info,
        ))


        time.sleep(sleep_s)

    return hist
def run_segment_to_target(
    *,
    mpc,
    robo,
    target_xyz,          # (3,)
    cam_index=0,
    use_roi=True,
    show=False,
    tol_m=1.5e-3,
    max_iters=15,
    sleep_s=0.15,
    n_path_points=10,
    lookahead_mode="index",
    nearest_window=5,
):
    Z_TARGET = float(target_xyz[2])

    get_tip_xyz_meas = make_get_tip_xyz_meas_from_camera(
        cam_index=cam_index,
        use_roi=use_roi,
        z_target=Z_TARGET,
        show=show,
        target_world_xyz=target_xyz,
    )

    hist = run_closed_loop_pose7_to_target_with_path(
        mpc_xyz=mpc,
        get_tip_xyz_meas=get_tip_xyz_meas,
        target_xyz_ur=target_xyz,
        robo=robo,
        tol_m=tol_m,
        max_iters=max_iters,
        sleep_s=sleep_s,
        show_debug=True,
        n_path_points=n_path_points,
        lookahead_mode=lookahead_mode,
        nearest_window=nearest_window,
    )

    # Robustly pick last entry with pixels; if none, fall back to last dbg
    last_valid = None
    for h in reversed(hist):
        if h.get("tip_px") is not None and h.get("target_px") is not None:
            last_valid = h
            break

    if last_valid is None:
        # If your loop ended early before appending, do a final measurement right here:
        tip_xyz, dbg = get_tip_xyz_meas()
        tip_px = dbg.get("tip_px", None)
        target_px = dbg.get("target_px", None)
        err_mm = 1000.0 * float(np.linalg.norm(np.asarray(target_xyz) - np.asarray(tip_xyz)))
        last_valid = dict(
            tip=tip_xyz,
            err_mm=err_mm,
            tip_px=tip_px,
            target_px=target_px,
            dbg=dbg,
        )

    return hist, last_valid
def run_multi_target_sequence(
    *,
    mpc,
    robo,
    cam_index=0,
    use_roi=True,
    show=False,
    z_target=-0.1,
    tol_m=1.5e-3,
    max_iters_per_target=15,
    sleep_s=0.15,
    n_path_points=10,
    lookahead_mode="index",
    nearest_window=5,
):
    # 1) user clicks multiple targets (world XY)
    targets_xy_world_m, dbg_click = pick_multiple_target_points_world(
        n_points=None,
        cam_index=cam_index,
        use_roi=use_roi,
        show_debug=True
    )

    if len(targets_xy_world_m) == 0:
        raise RuntimeError("No targets selected.")

    # Promote to XYZ
    targets_xyz = [np.array([xy[0], xy[1], float(z_target)], dtype=float) for xy in targets_xy_world_m]
    print("picked targets:", len(targets_xy_world_m), targets_xy_world_m)

    all_hists = []
    final_hits = []   # one per target

    for i, tgt in enumerate(targets_xyz):
        print("\n" + "#"*80)
        print(f"[MULTI] Target {i}/{len(targets_xyz)-1}: {tgt}")

        seg_hist, seg_last = run_segment_to_target(
            mpc=mpc,
            robo=robo,
            target_xyz=tgt,
            cam_index=cam_index,
            use_roi=use_roi,
            show=show,
            tol_m=tol_m,
            max_iters=max_iters_per_target,
            sleep_s=sleep_s,
            n_path_points=n_path_points,
            lookahead_mode=lookahead_mode,
            nearest_window=nearest_window,
        )

        all_hists.append(seg_hist)
        final_hits.append(dict(
            target_xyz=tgt,
            tip_xyz=seg_last.get("tip", None),
            err_mm=seg_last.get("err_mm", None),
            target_px=seg_last.get("target_px", None),
            tip_px=seg_last.get("tip_px", None),
            image_file=seg_last.get("dbg", {}).get("image_file", None),
            dbg=seg_last.get("dbg", None),
        ))

    return targets_xyz, all_hists, final_hits, dbg_click
def plot_all_targets_and_tips_on_image(
    image_filename,
    targets_px,
    tips_px,
    errors_mm=None,
    title="All targets and achieved tips",
    draw_lines=True,
    annotate=True,
):
    img_bgr = cv2.imread(image_filename)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_filename}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(7, 7))
    plt.imshow(img_rgb)

    tx = [p[0] for p in targets_px]
    ty = [p[1] for p in targets_px]
    plt.scatter(tx, ty, s=70, marker="x", color="yellow", label="Targets", zorder=3)

    sx = [p[0] for p in tips_px]
    sy = [p[1] for p in tips_px]
    plt.scatter(sx, sy, s=50, marker="o", color="red", label="Achieved tips", zorder=3)

    for i, (tgt, tip) in enumerate(zip(targets_px, tips_px)):
        if draw_lines:
            plt.plot([tgt[0], tip[0]], [tgt[1], tip[1]], linestyle="--", linewidth=1, color="white", zorder=2)

        if annotate:
            label = f"{i}"
            if errors_mm is not None and i < len(errors_mm) and errors_mm[i] is not None:
                label = f"{i} ({errors_mm[i]:.1f}mm)"
            plt.text(tip[0] + 3, tip[1] + 3, label, color="cyan", fontsize=9, zorder=4)

    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.show()
def preview_square_on_image(image_filename, square_xy_world, dbg):
    img_bgr = cv2.imread(image_filename)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image '{image_filename}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # world XY -> pixel
    pts_px = [world_xy_to_pixel(np.array(xy, float), dbg) for xy in square_xy_world]
    pts_px = np.asarray(pts_px, dtype=float)

    # close loop
    pts_closed = np.vstack([pts_px, pts_px[0]])

    plt.figure(figsize=(7,7))
    plt.imshow(img_rgb)
    plt.plot(pts_closed[:,0], pts_closed[:,1], marker="o")
    for i, (u, v) in enumerate(pts_px):
        plt.text(u + 3, v + 3, f"{i}", color="cyan", fontsize=10)
    plt.title("Square preview (pixel overlay)")
    plt.axis("off")
    plt.show()

pivot_point = np.array([
0.8581328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)


start_point = np.array([
0.6781328220229531, -0.7055298925316631, -0.1, -3.10153453698904, 0.024928591141737892, 0.06094868352765547
], float)
L0 = 0.05
start_point_pose6 = start_point  # [x,y,z, rx,ry,rz]
p0 = np.array([start_point_pose6[0], start_point_pose6[1], start_point_pose6[2],
               start_point_pose6[3], start_point_pose6[4], start_point_pose6[5],
               L0], float)

eps = np.array([
    np.deg2rad(0.5),
    np.deg2rad(0.5),
    1e-3,
    5e-4
])
p_min = np.array([ 0.2, -1, -0.2,  p0[3]-np.pi*2, p0[4]-np.pi*2, p0[5]-np.pi*2,  0.03])
p_max = np.array([ 0.85,  1,  1.0,  p0[3]+np.pi*2, p0[4]+np.pi*2, p0[5]+np.pi*2,  0.08])
w_u  = np.array([1e-8, 1e-8, 1e-3,   1e-5, 1e-5, 1e-3,   1e-4])
w_du = np.array([1e-6, 1e-6, 1e-6,   1e-5, 1e-5, 1e-6,   1e-6])
u_max = np.array([ .05, .05, .05, np.deg2rad(60), np.deg2rad(60), np.deg2rad(60),  0.02])
eps = np.array([
    1e-3, 1e-3, 1e-3,              # x,y,z
    np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(0.5),  
    5e-4                              # L
], dtype=float)

T_ur_pivot = ur_pose6_to_T(pivot_point)     # UR TCP pose at catheter base
p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
m_body = np.array([mag_params.mag_epm, 0.0, 0.0], dtype=float)
L0 = 0.05   
L_MAG = beam_params.length_of_mag
wire_len0 = wire_len_from_L(L0, L_MAG)
model = CosseratForwardModel(
    p0=p0_ur,
    q0=q0_ur,
    Kinv_fun=Kbt_inv_profile,
    m_local_fun=make_m_local_fun_wire_tip(wire_len0, mode="axial", alpha_end=0.0),
    m_moment=0.0,
    wire_len = wire_len0,
)

forward_tip_fn = lambda p: forward_cosserat_from_pose_ur_rotvec_L(p, model, m_body=m_body, L_mag=L_MAG)
J_fn = lambda p: numerical_jacobian_tip_xyz_pose(p, forward_tip_fn, eps)

mpc = mpc_controller_tipxy_LTI(
    Jxy_fn=J_fn,
    forward_tip_fn=forward_tip_fn,
    dt=0.5,
    Np=2,
    n_out=3,
    n_u=7,
    w_xy=(1, 1, 1),
    w_u=w_u,
    w_du=w_du,
    model_mode="ltv",
    u_max=u_max,
    p_min=p_min,
    p_max=p_max,
    N_sqp=4,
    use_offset_free=True
)


mpc.set_initial_params(p0)

def ensure_robo(robo):
    try:
        robo.get_joints()
        return robo
    except Exception:
        try:
            robo.shutdown()
        except Exception:
            pass
        return URRtde(ROBOT_IP)
# --- choose target (world frame) ---
x_target = np.array([0.8122726201998716, -0.7424378888535957, -0.1], float)
Z_TARGET = float(x_target[2])   

get_tip_xyz_meas = make_get_tip_xyz_meas_from_camera(
    cam_index=0,
    use_roi=True,
    z_target=Z_TARGET,
    show=False,
    target_world_xyz=x_target,   # <-- add this
)


# --- click ONE point (world XY) ---
clicked_xy_world_m, dbg_click = pick_multiple_target_points_world(
    n_points=1,     # only works if your picker supports it
    cam_index=0,
    use_roi=True,
    show_debug=True
)

# If your picker ignores n_points, you can just take the first click:
if len(clicked_xy_world_m) < 1:
    raise RuntimeError("Expected at least 1 clicked point.")

center_xy = np.asarray(clicked_xy_world_m[0], float).reshape(2,)

# --- build square in world XY ---
SIDE_M = 0.040   # 10 mm
YAW_RAD = 0.0
# square_xy = square_targets_from_center_xy(center_xy, side_m=SIDE_M, yaw_rad=YAW_RAD)
rect_xy = rect_targets_from_center_xy(center_xy, width_m=0.015, height_m=0.040, yaw_rad=np.deg2rad(5))
# targets_xyz = [np.array([xy[0], xy[1], z_target], float) for xy in rect_xy]
# Take one snapshot to get dbg (camera model + image_file)
res_preview = measure_lengths_mm(
    image_filename="focused_image.jpg",
    use_roi=True,
    show=False,
    cam_index=0
)

img_file = res_preview.get("image_file", "focused_image.jpg")
preview_square_on_image(img_file, rect_xy, res_preview)
# promote to XYZ
z_target = -0.1
targets_xyz = [np.array([xy[0], xy[1], z_target], float) for xy in rect_xy]
print("Square targets:", targets_xyz)
robo = connect_robo(ROBOT_IP)

# --- run those 4 points as your target sequence ---
all_hists, final_hits = run_target_list_sequence(
    mpc=mpc,
    robo=robo,
    targets_xyz=targets_xyz,
    cam_index=0,
    use_roi=True,
    show=False,
    tol_m=1.5e-3,
    max_iters_per_target=5,
    sleep_s=0.15,
    n_path_points=3,
    lookahead_mode="index",
    nearest_window=5,
)

# plotting (unchanged)
plot_image = None
for h in reversed(final_hits):
    if h["image_file"] is not None:
        plot_image = h["image_file"]
        break
if plot_image is None:
    raise RuntimeError("No image_file captured in final_hits; check measure_lengths_mm returns image_file.")

targets_px = [h["target_px"] for h in final_hits]
tips_px    = [h["tip_px"] for h in final_hits]
errors_mm  = [h["err_mm"] for h in final_hits]

keep = [i for i,(tp,sp) in enumerate(zip(targets_px, tips_px)) if tp is not None and sp is not None]
targets_px = [targets_px[i] for i in keep]
tips_px    = [tips_px[i] for i in keep]
errors_mm  = [errors_mm[i] for i in keep]

plot_all_targets_and_tips_on_image(
    image_filename=plot_image,
    targets_px=targets_px,
    tips_px=tips_px,
    errors_mm=errors_mm,
    title="Targets vs achieved tips (square run)"
)

robo.shutdown()
