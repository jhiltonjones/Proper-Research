import numpy as np
import copy
import json
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from proper_research.robot.transformations import get_point
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.cosserat_6d_pose import CosseratForwardModel, make_m_local_fun_wire_tip, ur_pose6_to_T
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile
from beam_direction_magnetisation.quarternions.shared_rotations import unpack_pose_ur_rotvec_L
from beam_direction_magnetisation.post_processing.post_processing import plot_energy_only_3d, quat_wxyz_to_R, make_lumen_centerline_double_turn
from beam_direction_magnetisation.post_processing.results_sim_paper import analyze_run 
from scipy.spatial.transform import Rotation as Rot
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
from proper_research.simulation.boundary_forward_model import ContactParams,EnergyMinForwardWithAnalyticJac ,EnergyMinForwardWithLumen, WarmForwardP8TipTangent, DeterministicForward6D
from beam_direction_magnetisation.cosserat_w_minimal_energy import make_lumen_centerline_turning
from scipy.stats import skew
from beam_direction_magnetisation.quarternions.quarternions_functions import quat_wxyz_normalize, quat_wxyz_mul, rotvec_to_quat_wxyz, quat_wxyz_to_rotvec, small_rot_quat_wxyz, unit, T_to_p_quat_wxyz
from pathlib import Path
from proper_research.simulation.simulations.ik_convertor import find_initial_joint_state
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from proper_research.simulation_controller.controller import MPCControllerTipXY
from proper_research.simulation_controller.diagnostics import (
    rollout_error_metrics,
    jacobian_svd_diagnostics,
)
mag_params = default_magnet_params()
beam_params = default_beam_params()
L_MAG = 0.04

from dataclasses import dataclass

import numpy as np

def arc_length_param(C):
    C = np.asarray(C, float)
    ds = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.zeros(len(C))
    s[1:] = np.cumsum(ds)
    return s

def unit(v, eps=1e-12):
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + eps)

def closest_point_polyline(C, x):
    """
    Return closest point on polyline C to point x.
    Outputs:
      i_seg: segment index (0..M-2)
      u:     segment parameter in [0,1]
      c:     closest point (3,)
      d2:    squared distance
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3,)
    V = C[1:] - C[:-1]                 # (M-1,3)
    W = x.reshape(1,3) - C[:-1]        # (M-1,3)
    VV = np.sum(V*V, axis=1) + 1e-15
    u = np.sum(W*V, axis=1) / VV
    u = np.clip(u, 0.0, 1.0)
    P = C[:-1] + u.reshape(-1,1)*V
    d2 = np.sum((P - x.reshape(1,3))**2, axis=1)
    i_seg = int(np.argmin(d2))
    return i_seg, float(u[i_seg]), P[i_seg], float(d2[i_seg])
def make_Kbt_inv_profile(EI_wire, EI_tip, GJ_wire, GJ_tip, bend_soft=1.0, tors_soft=1.0):
    def Kbt_inv_profile(s, len_wire):
        s = np.asarray(s, float)
        mask_tip = (s >= len_wire)

        EI_s = np.where(mask_tip, EI_tip, EI_wire)
        GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)

        Kinv = np.zeros((3, 3, s.size), float)
        Kinv[0, 0, :] = tors_soft / GJ_s
        Kinv[1, 1, :] = bend_soft / EI_s
        Kinv[2, 2, :] = bend_soft / EI_s
        return Kinv

    return Kbt_inv_profile
def rod_section_stiffness(r, E, nu):
    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    EA = E * A
    EI = E * I
    GJ = G * J

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": EA,
        "EI": EI,
        "GJ": GJ,
    }

def _angle_deg(u, v, eps=1e-12):
    u = np.asarray(u, float).reshape(3,)
    v = np.asarray(v, float).reshape(3,)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    if un < eps or vn < eps:
        return np.nan
    c = float(np.clip(np.dot(u/un, v/vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def _closest_centerline_tangent(Cc, x):
    i_seg, u_seg, c_closest, _ = closest_point_polyline(Cc[:, :3], x)
    i_seg = int(np.clip(i_seg, 0, Cc.shape[0]-2))
    t = Cc[i_seg+1, :3] - Cc[i_seg, :3]     # segment tangent at closest point
    t /= (np.linalg.norm(t) + 1e-12)
    return i_seg, float(u_seg), c_closest.reshape(3,), t


def pose7_rotvec_to_pose8_quat(p7):
    p7 = np.asarray(p7, float).ravel()
    t = p7[0:3]
    rvec = p7[3:6]
    L = p7[6]
    q = rotvec_to_quat_wxyz(rvec)
    return np.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3], L], float)

def pose8_quat_to_pose7_rotvec(p8):
    p8 = np.asarray(p8, float).ravel()
    t = p8[0:3]
    q = p8[3:7]
    L = p8[7]
    rvec = quat_wxyz_to_rotvec(q)
    return np.array([t[0], t[1], t[2], rvec[0], rvec[1], rvec[2], L], float)


def resample_polyline(C, ds_target=1e-3):
    """
    Resample polyline C to approximately uniform spacing ds_target.
    Returns C_rs: (Mr,3), s_rs: (Mr,)
    """
    C = np.asarray(C, float)
    s = arc_length_param(C)
    L = s[-1]
    if L < 1e-12:
        return C.copy(), s

    s_rs = np.arange(0.0, L + 0.5*ds_target, ds_target)
    C_rs = np.zeros((len(s_rs), 3), float)

    # piecewise-linear interpolation in arc-length
    for k in range(3):
        C_rs[:, k] = np.interp(s_rs, s, C[:, k])

    return C_rs, s_rs



def save_step_artifacts(
    *,
    k: int,
    frames_dir,
    log_csv_path,
    u0,
    p_now,
    y_now,
    i_ref: int,
    info: dict,
    centerline_tip,
    lumen_C,
    lumen_R,
    p0_ur,
    tip_pos,
    tip_tan,
    fixed_limits,
    tip_from_centerline=None,
):
    """
    Save one simulation step.

    This version matches the new MPCControllerTipXY info dictionary.

    Expected new-controller info keys:
        status
        infeasible
        solver_mode
        rollout_steps
        u0
        U_seq
        p_now
        x_now
        idx_ref
        X_pred
        sqp_hist
        trust_radius
        U_applied
        p_rollout
        x_rollout
    """
    import csv
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    frames_dir = Path(frames_dir)
    log_csv_path = Path(log_csv_path)

    frames_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Robust array formatting
    # ------------------------------------------------------------------
    def _pad(v, n):
        v = np.asarray(v, float).reshape(-1)

        if v.size < n:
            v = np.pad(v, (0, n - v.size), constant_values=np.nan)

        return v[:n]

    u0 = _pad(u0, 7)
    p_now = _pad(p_now, 8)
    y_now = _pad(y_now, 6)

    Cc = np.asarray(lumen_C, float)

    if Cc.ndim != 2 or Cc.shape[1] < 3:
        raise ValueError(f"lumen_C must have shape (M, >=3), got {Cc.shape}.")

    tip_pos = _pad(tip_pos, 3)
    tip_tan = _pad(tip_tan, 3)

    # ------------------------------------------------------------------
    # Reference index
    # ------------------------------------------------------------------
    idx_ref = np.asarray(info.get("idx_ref", []), int).reshape(-1)
    i_ref_mpc = int(idx_ref[0]) if idx_ref.size else int(i_ref)

    # ------------------------------------------------------------------
    # Centreline / vessel-contact diagnostics
    # ------------------------------------------------------------------
    i_seg, u_seg, c_cl, t_v = _closest_centerline_tangent(Cc, tip_pos)

    tip_vessel_angle_deg = _angle_deg(tip_tan, t_v)

    r = tip_pos - c_cl
    axial_component = float(np.dot(r, t_v))
    r_perp = r - axial_component * t_v
    rho = float(np.linalg.norm(r_perp))

    clearance = np.nan

    if lumen_R is not None:
        Rr = np.asarray(lumen_R, float).reshape(-1)

        if Rr.size > 0:
            idx_v = int(np.clip(i_seg + int(u_seg >= 0.5), 0, Rr.size - 1))
            clearance = float(Rr[idx_v] - rho)

    # ------------------------------------------------------------------
    # Save current matplotlib figure, if one exists
    # ------------------------------------------------------------------
    fig = plt.gcf()

    if fig is not None and len(fig.axes) > 0:
        fig_path = frames_dir / f"frame_{k:06d}.png"
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        fig_path = ""

    # ------------------------------------------------------------------
    # New-controller rollout diagnostics
    # ------------------------------------------------------------------
    U_applied = np.asarray(info.get("U_applied", []), float)
    x_rollout = np.asarray(info.get("x_rollout", []), float)
    p_rollout = np.asarray(info.get("p_rollout", []), float)
    X_pred = np.asarray(info.get("X_pred", []), float)
    U_seq = np.asarray(info.get("U_seq", []), float)

    if U_applied.ndim != 2:
        U_applied = np.empty((0, 7), float)

    if x_rollout.ndim != 2:
        x_rollout = np.empty((0, 6), float)

    if p_rollout.ndim != 2:
        p_rollout = np.empty((0, 8), float)

    if X_pred.ndim != 2:
        X_pred = np.empty((0, 6), float)

    if U_seq.ndim != 2:
        U_seq = np.empty((0, 7), float)

    rollout_steps_logged = int(info.get("rollout_steps", 0))

    K_roll = min(
        rollout_steps_logged,
        U_applied.shape[0],
        x_rollout.shape[0],
        X_pred.shape[0],
    )

    # Compute rollout prediction errors here, rather than requiring info
    # to already contain them.
    rollout_err_xy_seq = np.full(K_roll, np.nan, float)
    rollout_err_xyz_seq = np.full(K_roll, np.nan, float)

    for j in range(K_roll):
        rollout_err_xy_seq[j] = float(
            np.linalg.norm(x_rollout[j, :2] - X_pred[j, :2])
        )
        rollout_err_xyz_seq[j] = float(
            np.linalg.norm(x_rollout[j, :3] - X_pred[j, :3])
        )

    # ------------------------------------------------------------------
    # SQP diagnostics from new sqp_hist format
    # ------------------------------------------------------------------
    sqp_hist = info.get("sqp_hist", [])

    if sqp_hist is None:
        sqp_hist = []

    sqp_iters_done = len(sqp_hist)

    if sqp_iters_done > 0:
        sqp_du_final = float(sqp_hist[-1].get("step_norm", np.nan))
        sqp_du_rel_final = float(sqp_hist[-1].get("rel_step_norm", np.nan))
    else:
        sqp_du_final = np.nan
        sqp_du_rel_final = np.nan

    # ------------------------------------------------------------------
    # Basic first-stage prediction diagnostics
    # ------------------------------------------------------------------
    if X_pred.shape[0] > 0:
        pred_tip_0 = _pad(X_pred[0], 6)
        pred1_err_xy = float(np.linalg.norm(pred_tip_0[:2] - y_now[:2]))
        pred1_err_xyz = float(np.linalg.norm(pred_tip_0[:3] - y_now[:3]))
    else:
        pred1_err_xy = np.nan
        pred1_err_xyz = np.nan

    # ------------------------------------------------------------------
    # Main CSV row
    # ------------------------------------------------------------------
    row = {
        "k": int(k),

        "status": str(info.get("status", "")),
        "infeasible": int(info.get("infeasible", -1)),
        "solver_mode": str(info.get("solver_mode", "")),
        "rollout_steps": int(info.get("rollout_steps", 0)),

        "i_ref": int(i_ref),
        "i_ref_mpc": int(i_ref_mpc),
        "idx_ref_0": int(idx_ref[0]) if idx_ref.size > 0 else -1,
        "idx_ref_terminal": int(idx_ref[-1]) if idx_ref.size > 0 else -1,

        "u0_vx": float(u0[0]),
        "u0_vy": float(u0[1]),
        "u0_vz": float(u0[2]),
        "u0_wx": float(u0[3]),
        "u0_wy": float(u0[4]),
        "u0_wz": float(u0[5]),
        "u0_dL": float(u0[6]),

        "p_now_x": float(p_now[0]),
        "p_now_y": float(p_now[1]),
        "p_now_z": float(p_now[2]),
        "p_now_qw": float(p_now[3]),
        "p_now_qx": float(p_now[4]),
        "p_now_qy": float(p_now[5]),
        "p_now_qz": float(p_now[6]),
        "p_now_L": float(p_now[7]),

        "tip_x": float(y_now[0]),
        "tip_y": float(y_now[1]),
        "tip_z": float(y_now[2]),
        "tip_tx": float(y_now[3]),
        "tip_ty": float(y_now[4]),
        "tip_tz": float(y_now[5]),

        "pred1_err_xy_m": float(pred1_err_xy),
        "pred1_err_xy_mm": 1e3 * float(pred1_err_xy),
        "pred1_err_xyz_m": float(pred1_err_xyz),
        "pred1_err_xyz_mm": 1e3 * float(pred1_err_xyz),

        "tip_vessel_angle_deg": float(tip_vessel_angle_deg),
        "i_seg": int(i_seg),
        "u_seg": float(u_seg),
        "rho_m": float(rho),
        "rho_mm": 1e3 * float(rho),
        "clearance_m": float(clearance),
        "clearance_mm": 1e3 * float(clearance),

        "sqp_iters_done": int(sqp_iters_done),
        "sqp_du_final": float(sqp_du_final),
        "sqp_du_rel_final": float(sqp_du_rel_final),

        "frame_path": str(fig_path),
    }

    # ------------------------------------------------------------------
    # Trust-radius logging
    # ------------------------------------------------------------------
    trust_radius = info.get("trust_radius", None)

    if trust_radius is None:
        trust_radius = np.full(7, np.nan)
    else:
        trust_radius = np.asarray(trust_radius, float).reshape(-1)

    if trust_radius.size == 1:
        trust_radius = np.full(7, float(trust_radius[0]))
    elif trust_radius.size < 7:
        trust_radius = np.pad(
            trust_radius,
            (0, 7 - trust_radius.size),
            constant_values=np.nan,
        )

    trust_radius = trust_radius[:7]

    row.update(
        {
            "trust_vx": float(trust_radius[0]),
            "trust_vy": float(trust_radius[1]),
            "trust_vz": float(trust_radius[2]),
            "trust_wx": float(trust_radius[3]),
            "trust_wy": float(trust_radius[4]),
            "trust_wz": float(trust_radius[5]),
            "trust_dL": float(trust_radius[6]),
        }
    )

    # ------------------------------------------------------------------
    # Per-SQP iteration logging
    # ------------------------------------------------------------------
    for j, h in enumerate(sqp_hist):
        row[f"sqp_status_{j}"] = str(h.get("status", ""))
        row[f"sqp_du_{j}"] = float(h.get("step_norm", np.nan))
        row[f"sqp_du_rel_{j}"] = float(h.get("rel_step_norm", np.nan))

    # ------------------------------------------------------------------
    # Applied rollout controls and predicted/applied states
    # ------------------------------------------------------------------
    for j in range(K_roll):
        uj = _pad(U_applied[j], 7)
        xj = _pad(x_rollout[j], 6)
        xpj = _pad(X_pred[j], 6)

        row.update(
            {
                f"u_applied_{j}_vx": float(uj[0]),
                f"u_applied_{j}_vy": float(uj[1]),
                f"u_applied_{j}_vz": float(uj[2]),
                f"u_applied_{j}_wx": float(uj[3]),
                f"u_applied_{j}_wy": float(uj[4]),
                f"u_applied_{j}_wz": float(uj[5]),
                f"u_applied_{j}_dL": float(uj[6]),

                f"x_rollout_{j}_x": float(xj[0]),
                f"x_rollout_{j}_y": float(xj[1]),
                f"x_rollout_{j}_z": float(xj[2]),
                f"x_rollout_{j}_tx": float(xj[3]),
                f"x_rollout_{j}_ty": float(xj[4]),
                f"x_rollout_{j}_tz": float(xj[5]),

                f"x_pred_{j}_x": float(xpj[0]),
                f"x_pred_{j}_y": float(xpj[1]),
                f"x_pred_{j}_z": float(xpj[2]),
                f"x_pred_{j}_tx": float(xpj[3]),
                f"x_pred_{j}_ty": float(xpj[4]),
                f"x_pred_{j}_tz": float(xpj[5]),

                f"rollout_err_xy_{j}_m": float(rollout_err_xy_seq[j]),
                f"rollout_err_xy_{j}_mm": 1e3 * float(rollout_err_xy_seq[j]),
                f"rollout_err_xyz_{j}_m": float(rollout_err_xyz_seq[j]),
                f"rollout_err_xyz_{j}_mm": 1e3 * float(rollout_err_xyz_seq[j]),
            }
        )

    # ------------------------------------------------------------------
    # Full planned sequence logging, useful for debugging MPC solutions
    # ------------------------------------------------------------------
    K_plan = min(U_seq.shape[0], 20)

    for j in range(K_plan):
        uj = _pad(U_seq[j], 7)

        row.update(
            {
                f"u_plan_{j}_vx": float(uj[0]),
                f"u_plan_{j}_vy": float(uj[1]),
                f"u_plan_{j}_vz": float(uj[2]),
                f"u_plan_{j}_wx": float(uj[3]),
                f"u_plan_{j}_wy": float(uj[4]),
                f"u_plan_{j}_wz": float(uj[5]),
                f"u_plan_{j}_dL": float(uj[6]),
            }
        )

    # ------------------------------------------------------------------
    # Append to CSV
    # ------------------------------------------------------------------
    file_exists = log_csv_path.exists()

    with open(log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists or log_csv_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def snapshot_forward(forward6d, p8, *, commit=False):
    """
    Returns a self-consistent snapshot from ONE forward call.
    """
    y = np.asarray(forward6d(p8, commit=commit), float).reshape(6,)
    C = getattr(forward6d, "last_p_centerline", None)
    C = None if C is None else np.asarray(C, float).copy()

    tip = y[:3].copy()
    tan = y[3:6].copy()
    return tip, tan, C, y

def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    L_cmd = 0.016

    pivot_point = np.array([
        0.7981328220229531, -0.70992731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)

    base_point = np.array([
        pivot_point[0] - (L_cmd + 0.12),
        pivot_point[1],
        -0.1,
        np.pi, 0.001, 0.001
    ], float)
    # start_point = np.array([
    #     0.7981328220229531-(L_cmd), -0.70992731669220016, -0.1+0.2,
    #     np.pi, 0.001, 0.001
    # ], float)
    start_point = np.asarray(get_point(0, 0, base_point, pivot_point), dtype=float)
    print(f"Start point {start_point}")
    dt = 0.02
    return pivot_point, start_point, L_cmd, dt

def effective_lengths(L_ins, *, L_tip_full=0.04, L_tip_min=0.01):
    """
    L_ins      : commanded insertion (what MPC tracks)
    L_tip_full : physical magnetic tip length (4 cm)
    L_tip_min  : minimum model length so solver has something to solve (e.g. 1 cm)

    Returns (L_model, wire_len, tip_len)
    """
    L_ins = float(L_ins)

    # Magnetised tip inside grows with insertion until full tip is inside
    tip_len = min(L_ins, L_tip_full)

    # Wire is everything beyond the physical tip length
    wire_len = max(L_ins - L_tip_full, 0.0)

    # Total model length is the inserted length, but don't go below minimum model length
    L_model = max(L_ins, L_tip_min)

    # If we are below L_tip_min, we still model a minimum rod,
    # but magnetisation should NOT exceed what's actually inserted:
    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len
def build_lumen_and_forward_models(pivot_point: np.ndarray, L0: float):
    T_ur_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    pivot_point_lumen = np.array([
        0.7981328220229531, -0.7112731669220016, -0.1,
        np.pi, 0.001, 0.001
    ], float)
    T_ur_pivot_lumen = ur_pose6_to_T(pivot_point_lumen)   
    p0_ur_lumen, q0_ur_lumen = T_to_p_quat_wxyz(T_ur_pivot_lumen)   
    L_model, wire_len_model, tip_len_model = effective_lengths(L0)
    print(
        f"[INIT] L_ins={L0:.3f} -> "
        f"L_model={L_model:.3f}, wire_len={wire_len_model:.3f}, tip_len={tip_len_model:.3f}"
    )

    m_body = np.array([-mag_params.mag_epm, 0.0, 0.0], dtype=float)

    q = q0_ur
    R0 = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    t0 = R0 @ np.array([-1.0, 0.0, 0.0])


    lumen_C = make_lumen_centerline_turning(
        p_start=p0_ur_lumen,
        t0=t0,
        length=0.045,
        n_pts=130,
        bend_axis=np.array([0.0, 0.0, 1.0]),
        bend_angle=np.deg2rad(-30.0),
        bend_start=0.015,
        bend_end=0.025,
    )
    lumen_C, s_path = resample_polyline(lumen_C, ds_target=1e-3)
    lumen_R = np.full(len(lumen_C), 0.004)
    lumen_path = lumen_C
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )

    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )
    EA_wire = wire["EA"]
    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]

    EA_tip = tip["EA"]
    EI_tip = tip["EI"]
    GJ_tip = tip["GJ"]
    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )
    contact = ContactParams(
        r_beam=0.001,
        k=1e5,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=None,
    )
    forward_model = EnergyMinForwardWithAnalyticJac(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, float),
        lumen_R=np.asarray(lumen_R, float),
        N_nodes=10,
        maxiter=40,
        L0_init=0.01,
        dL_internal=0.04,
        use_lumen_jac=True,
        L_tip_full=0.04,
        L_tip_min=0.01,
        contact_params=contact,
        use_fast_contact_grad=False,
    )
    forward_model_wrong = EnergyMinForwardWithAnalyticJac(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        u_star=np.zeros(3),
        m_body=m_body,
        lumen_C=np.asarray(lumen_C, float),
        lumen_R=np.asarray(lumen_R, float),
        N_nodes=10,
        maxiter=40,
        L0_init=0.01,
        dL_internal=0.04,
        use_lumen_jac=False,
        L_tip_full=0.04,
        L_tip_min=0.01,
        contact_params=None,
        use_fast_contact_grad=False,
    )
    return p0_ur, q0_ur, lumen_C, lumen_R, lumen_path, s_path, forward_model, forward_model_wrong
def J_full_from_robot_reduced_tip_tangent(J_red, n_out_full=5):
    """
    J_red shape: (5,4)
        columns = [x, y, yaw_z, L]

    Returns full robot Jacobian in 7 controls:
        [vx, vy, vz, wx, wy, wz, dL]
    """
    J_full = np.zeros((n_out_full, 7), float)
    J_full[:, 0] = J_red[:, 0]   # x translation
    J_full[:, 1] = J_red[:, 1]   # y translation
    J_full[:, 5] = J_red[:, 2]   # yaw about z
    J_full[:, 6] = J_red[:, 3]   # insertion
    return J_full

def analytic_J_robot_xy_yaw_dL(
    p8,
    forward_model,
    n_out=6,
):
    p8 = np.asarray(p8, float).reshape(8,)

    if hasattr(forward_model, "fwd"):
        wrapper = forward_model
        fm = forward_model.fwd
    else:
        wrapper = None
        fm = forward_model

    p7 = pose8_quat_to_pose7_rotvec(p8)

    # One nominal solve only
    if wrapper is not None:
        _ = wrapper(p8, commit=True)
    else:
        _ = fm(p7)

    # No second solve here
    J_tip_7 = fm.jacobian_tip_pose7_from_cache(p7)

    J = np.zeros((n_out, 4), dtype=float)

    J[0:3, 0] = J_tip_7[:, 0]   # robot x
    J[0:3, 1] = J_tip_7[:, 1]   # robot y
    J[0:3, 2] = J_tip_7[:, 5]   # yaw ≈ world delta_phi_z
    J[0:3, 3] = J_tip_7[:, 6]   # length

    return J
def build_controller(
    start_point: np.ndarray,
    L0: float,
    dt: float,
    forward_model,
    forward_model_wrong,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    p0_ur: np.ndarray,
):
    start_point_pose6 = np.asarray(start_point, float)

    p0_pose7 = np.array(
        [
            start_point_pose6[0],
            start_point_pose6[1],
            start_point_pose6[2],
            start_point_pose6[3],
            start_point_pose6[4],
            start_point_pose6[5],
            L0,
        ],
        dtype=float,
    )

    p0 = pose7_rotvec_to_pose8_quat(p0_pose7)

    p_min = np.array(
        [
            0.2,
            -1.0,
            start_point[2],
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            0.01,
        ],
        dtype=float,
    )

    p_max = np.array(
        [
            0.8,
            1.5,
            start_point[2],
            +np.inf,
            +np.inf,
            +np.inf,
            +np.inf,
            0.05,
        ],
        dtype=float,
    )

    u_max = np.array(
        [
            1.0,
            1.0,
            1.0,
            np.deg2rad(60.0),
            np.deg2rad(60.0),
            np.deg2rad(360.0),
            0.01,
        ],
        dtype=float,
    )

    # Start conservative. You can loosen these later.
    trust_radius = np.array(
        [
            5e-3,
            5e-3,
            1e-4,
            np.deg2rad(5.0),
            np.deg2rad(5.0),
            np.deg2rad(10.0),
            1e-3,
        ],
        dtype=float,
    )

    w_u = np.array(
        [
            1e-3,
            1e-3,
            1e-2,
            1e-2,
            1e-2,
            1e-3,
            1e-3,
        ],
        dtype=float,
    )

    w_du = np.array(
        [
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
        ],
        dtype=float,
    )

    forward6d = WarmForwardP8TipTangent(copy.deepcopy(forward_model))
    forward6d_wrong = WarmForwardP8TipTangent(copy.deepcopy(forward_model_wrong))

    def J_fn(p8):
        """
        Return discrete-time input-output Jacobian B.

        Your analytic Jacobian gives sensitivity wrt reduced robot state:
            [x, y, yaw_z, L]

        Then we multiply by dt to convert velocity controls to one-step increments.
        """
        forward6d_jac = WarmForwardP8TipTangent(copy.deepcopy(forward_model))

        Jred_state = analytic_J_robot_xy_yaw_dL(
            p8,
            forward6d_jac,
            n_out=6,
        )

        # Controls are velocities/rates, so one-step state increment is dt*u.
        B_reduced = Jred_state @ np.diag([dt, dt, dt, dt])

        return J_full_from_robot_reduced_tip_tangent(
            B_reduced,
            n_out_full=6,
        )

    mpc = MPCControllerTipXY(
        Jxy_fn=J_fn,
        forward_tip_fn=forward6d,
        dt=dt,
        Np=12,
        n_out=6,
        n_u=7,
        n_p=8,
        w_xy=(1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
        w_u=w_u,
        w_du=w_du,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        model_mode="ltv",
        solver_mode="sqp_full",
        N_sqp=5,
        trust_radius=trust_radius,
        enable_hard_epm_tip_clearance=True,
        epm_tip_hard_min_m=0.11,
        dL_index=6,
        dL_back_max=0.002,
        dL_fwd_max=0.01,
        qp_reg=1e-9,
    )

    mpc.lumen_C = np.asarray(lumen_C, float)
    mpc.lumen_R = np.asarray(lumen_R, float)

    mpc.use_xy_ref_distance = True
    mpc.ref_lookahead_pts = 0
    mpc.allow_ref_backward = False
    mpc.ref_stride_pts = 1

    # Must have length Np.
    mpc.ref_stage_weights = np.linspace(2.0, 0.5, mpc.Np)

    # Seed insertion for LTV nominal rollout.
    mpc.dL_guess = 0.002

    mpc.set_initial_params(p0)

    return mpc, p0_ur, p0, p_min, p_max, u_max, forward6d, forward6d_wrong
def setup_output_dirs(out_root: Path, lumen_C: np.ndarray, lumen_R: np.ndarray, mpc, u_max, p_min, p_max):
    frames_dir = out_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_csv_path = out_root / "log.csv"
    log_meta_path = out_root / "meta.json"

    np.save(out_root / "lumen_C.npy", lumen_C)
    np.save(out_root / "lumen_R.npy", lumen_R)

    meta = dict(
        dt=float(mpc.dt),
        Np=int(mpc.Np),
        u_max=u_max.tolist(),
        p_min=p_min.tolist(),
        p_max=p_max.tolist(),
    )
    with open(log_meta_path, "w") as f:
        json.dump(meta, f, indent=2)


    return frames_dir, log_csv_path


def maybe_plot_summary(k_hist, pred1_hist, svd_S_hist, svd_cond_hist):
    K = np.asarray(k_hist)
    pred1 = np.asarray(pred1_hist)
    Smat = np.asarray(svd_S_hist)
    cond = np.asarray(svd_cond_hist)

    plt.figure()
    plt.plot(K, 1e3 * pred1)
    plt.xlabel("k")
    plt.ylabel("pred1_err (mm)")
    plt.title("One-step prediction error vs step")
    plt.grid(True)
    # plt.show()

    plt.figure()
    if Smat.ndim == 2 and Smat.shape[0] == K.size:
        for i in range(Smat.shape[1]):
            plt.plot(K, Smat[:, i], label=f"σ{i+1}")
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("singular values of B (log)")
    plt.title("Jacobian singular values vs step")
    plt.grid(True)
    plt.legend()
    # plt.show()

    plt.figure()
    plt.plot(K, cond)
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("cond(B) (log)")
    plt.title("Jacobian conditioning vs step")
    plt.grid(True)
    # plt.show()

    fig, ax1 = plt.subplots()
    l1, = ax1.plot(K, 1e3 * pred1, label="pred1_err (mm)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("pred1_err (mm)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    l2, = ax2.plot(K, cond, label="cond(B)")
    ax2.set_yscale("log")
    ax2.set_ylabel("cond(B) (log)")

    ax1.legend(handles=[l1, l2], loc="best")
    plt.title("pred1_err vs Jacobian conditioning")
    # plt.show()


def run_simulation(
    mpc,
    forward6d,
    p0_ur,
    p0,
    lumen_C,
    lumen_R,
    lumen_path,
    s_path,
    frames_dir: Path,
    log_csv_path: Path,
    *,
    max_steps=200,
    rollout_steps=1,
    solver_mode="sqp_full",
    save_plots=True,
):
    fixed_limits = None

    hist = {
        "k": [],
        "status": [],
        "infeasible": [],
        "i_ref": [],
        "tip": [],
        "p": [],
        "u0": [],
        "U_applied": [],
        "rollout_err_xyz": [],
        "rollout_err_xy": [],
        "sqp_iters": [],
        "sqp_final_step_norm": [],
        "jac_singular_values": [],
        "jac_cond": [],
    }

    for k in range(max_steps):
        p_pre = mpc.p.copy()
        x_pre = mpc.x.copy()
        tip_pre = x_pre[:3].copy()

        # Optional: local Jacobian diagnostic before solve.
        try:
            _, _, _, B0 = mpc._build_prediction_mats(p_pre, U_guess=None)
            jac_diag = jacobian_svd_diagnostics(B0)
            S = jac_diag["singular_values"]
            cond = jac_diag["condition_number"]
        except Exception:
            S = np.full(min(mpc.n, mpc.m), np.nan)
            cond = np.nan

        p_post, x_post, info = mpc.step(
            x_meas=None,
            rollout_steps=rollout_steps,
            solver_mode=solver_mode,
        )

        tip_post = x_post[:3].copy()

        X_pred = info.get("X_pred", None)
        x_rollout = info.get("x_rollout", None)

        if X_pred is not None and x_rollout is not None:
            try:
                roll_diag = rollout_error_metrics(
                    x_rollout=np.asarray(x_rollout, float),
                    X_pred=np.asarray(X_pred, float),
                )
                rollout_err_xyz = roll_diag["stage_position_error"]
                rollout_err_xy = np.linalg.norm(
                    np.asarray(x_rollout)[:, :2]
                    - np.asarray(X_pred)[: np.asarray(x_rollout).shape[0], :2],
                    axis=1,
                )
            except Exception:
                rollout_err_xyz = np.full(0, np.nan)
                rollout_err_xy = np.full(0, np.nan)
        else:
            rollout_err_xyz = np.full(0, np.nan)
            rollout_err_xy = np.full(0, np.nan)

        sqp_hist = info.get("sqp_hist", [])
        if len(sqp_hist) > 0:
            sqp_final_step = float(sqp_hist[-1].get("step_norm", np.nan))
        else:
            sqp_final_step = np.nan

        idx_ref = np.asarray(info.get("idx_ref", []), int)
        i_ref = int(idx_ref[0]) if idx_ref.size else int(getattr(mpc, "i_ref_last", 0))

        hist["k"].append(k)
        hist["status"].append(info.get("status", "unknown"))
        hist["infeasible"].append(int(info.get("infeasible", 0)))
        hist["i_ref"].append(i_ref)
        hist["tip"].append(tip_post.copy())
        hist["p"].append(p_post.copy())
        hist["u0"].append(info["u0"].copy())
        hist["U_applied"].append(np.asarray(info["U_applied"], float).copy())
        hist["rollout_err_xyz"].append(np.asarray(rollout_err_xyz, float).copy())
        hist["rollout_err_xy"].append(np.asarray(rollout_err_xy, float).copy())
        hist["sqp_iters"].append(len(sqp_hist))
        hist["sqp_final_step_norm"].append(sqp_final_step)
        hist["jac_singular_values"].append(np.asarray(S, float).copy())
        hist["jac_cond"].append(float(cond))

        print(
            f"[{solver_mode}] k={k:04d} "
            f"status={info.get('status', 'unknown')} "
            f"infeas={int(info.get('infeasible', 0))} "
            f"i_ref={i_ref:4d} "
            f"rollout={rollout_steps} "
            f"Δtip={1e3*np.linalg.norm(tip_post - tip_pre):7.3f} mm "
            f"sqp_it={len(sqp_hist)} "
            f"cond(B0)={cond:.2e}"
        )

        if save_plots:
            try:
                tip_pre_vis, tan_pre, C_pre, y_pre = snapshot_forward(
                    forward6d,
                    p_pre,
                    commit=False,
                )

                fixed_limits = plot_energy_only_3d(
                    C_pre,
                    lumen_C=lumen_C,
                    lumen_R=lumen_R,
                    p0=p0_ur,
                    tip=tip_pre,
                    p_mag=p_post,
                    show=False,
                    fixed_limits=fixed_limits,
                    zoom_out=1.5,
                )

                save_step_artifacts(
                    k=k,
                    frames_dir=frames_dir,
                    log_csv_path=log_csv_path,
                    u0=info["u0"],
                    p_now=p_post,
                    y_now=x_post,
                    i_ref=i_ref,
                    info=info,
                    centerline_tip=C_pre,
                    lumen_C=lumen_C,
                    lumen_R=lumen_R,
                    p0_ur=p0_ur,
                    tip_pos=tip_pre,
                    tip_tan=tan_pre,
                    fixed_limits=fixed_limits,
                    tip_from_centerline=(
                        C_pre[:, -1]
                        if (
                            C_pre is not None
                            and getattr(C_pre, "shape", None) is not None
                            and C_pre.shape[0] == 3
                        )
                        else (C_pre[-1] if C_pre is not None else None)
                    ),
                )
            except Exception as e:
                print(f"[WARN] plotting/logging failed at k={k}: {e}")

        if i_ref >= lumen_path.shape[0] - 6:
            print("[DONE] reached final reference region")
            break

        if int(info.get("infeasible", 0)):
            print("[STOP] MPC infeasible")
            break

    return hist

def run_experiment(
    *,
    solver_mode,
    rollout_steps,
    run_name,
):
    out_root = Path("runs") / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    pivot_point, start_point, L_cmd, dt = make_initial_poses()

    # Best choice: actual joints measured from a nearby safe pose.
    q_seed = np.array(
        [
            -0.4124,
            -1.5810,
            -1.9740,
            -1.1403,
            1.5828,
            -0.2107,
        ],
        dtype=float,
    )

    state0, ik_result = find_initial_joint_state(
        start_point=start_point,
        L_cmd=L_cmd,
        q_seed=q_seed,
        active_tcp_pose6=np.zeros(6),
        tcp_to_magnet_pose6=np.array(
            [0.0, 0.0, -0.044, 0.0, 0.0, 0.0]
        ),
    )

    q0 = state0[:6]
    L0 = state0[6]

    print(f"MPC initial state: {state0}")

    (
        p0_ur,
        q0_ur,
        lumen_C,
        lumen_R,
        lumen_path,
        s_path,
        forward_model,
        forward_model_wrong,
    ) = build_lumen_and_forward_models(pivot_point, L0)

    (
        mpc,
        p0_ur,
        p0,
        p_min,
        p_max,
        u_max,
        forward6d,
        forward6d_wrong,
    ) = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        forward_model=forward_model,
        forward_model_wrong=forward_model_wrong,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        p0_ur=p0_ur,
    )

    frames_dir, log_csv_path = setup_output_dirs(
        out_root=out_root,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        mpc=mpc,
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
    )

    if log_csv_path.exists():
        log_csv_path.unlink()

    stats = run_simulation(
        mpc=mpc,
        forward6d=forward6d,
        p0_ur=p0_ur,
        p0=p0,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_path=lumen_path,
        s_path=s_path,
        frames_dir=frames_dir,
        log_csv_path=log_csv_path,
        max_steps=200,
        rollout_steps=rollout_steps,
        solver_mode=solver_mode,
        save_plots=True,
    )

    return stats
if __name__ == "__main__":
    experiments = [
        # ("lti", 1, "lti_rollout1"),
        # ("ltv_oneshot", 1, "ltv_oneshot_rollout1"),
        ("sqp_full", 1, "sqp_full_rollout1"),
        # ("sqp_full", 3, "sqp_full_rollout3"),
        # ("sqp_full", 5, "sqp_full_rollout5"),
    ]

    all_stats = {}

    for solver_mode, rollout_steps, run_name in experiments:
        print("\n" + "=" * 80)
        print(f"Running {run_name}")
        print("=" * 80)

        all_stats[run_name] = run_experiment(
            solver_mode=solver_mode,
            rollout_steps=rollout_steps,
            run_name=run_name,
        )