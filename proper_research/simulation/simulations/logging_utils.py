import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from proper_research.simulation.simulations.rollout_diagnostics import (
    build_rollout_stage_rows,
    append_rollout_stage_csv,
    save_rollout_npz,
)
from proper_research.simulation_controller.safe_progress_logging import (
    safe_progress_log_row,
)

def save_controller_diagnostics_npz(
    *,
    output_dir: Path,
    k: int,
    info: dict,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    keys = (
        "p_start",
        "x_start",
        "U_seq",
        "idx_ref",
        "X_ref_plan",
        "reference_s_plan_m",
        "X_pred",
        "X_lin_candidate",
        "X_nl_candidate",
        "X_nl_plan",
        "x_rollout",
        "u_applied",

        "linerr_stage_vector_m",
        "linerr_stage_norm_m",

        "B_sequence_linearisation",
        "B_pose_nodes_linearisation",
        "B_sequence_solution",
        "B_pose_nodes_solution",
        "B_drift_relative_stage",

        "jacobian_diag_input_scale",

        "jac_solution_column_norm_xyz",
        "jac_solution_column_norm_xy",
        "jac_solution_scaled_authority",
        "jac_solution_weighted_scaled_authority",
        "jac_solution_contribution_vectors",
        "jac_solution_contribution_norm",
        "jac_solution_predicted_increment",
        "jac_solution_matrix_change_relative",
        "jac_solution_column_gain_ratio",
        "jac_solution_column_direction_change_deg",

        "jac_linearisation_column_norm_xyz",
        "jac_linearisation_column_norm_xy",
        "jac_linearisation_scaled_authority",
        "jac_linearisation_weighted_scaled_authority",
        "jac_linearisation_contribution_vectors",
        "jac_linearisation_contribution_norm",
        "jac_linearisation_predicted_increment",
        "jac_linearisation_matrix_change_relative",
        "jac_linearisation_column_gain_ratio",
        "jac_linearisation_column_direction_change_deg",
    )

    payload = {
        "k": np.asarray(int(k)),
    }

    for key in keys:
        value = info.get(key, None)

        if value is None:
            continue

        try:
            arr = np.asarray(value)

            if arr.size > 0:
                payload[key] = arr.copy()

        except Exception:
            continue

    path = (
        output_dir
        / f"controller_diag_{k:05d}.npz"
    )

    np.savez_compressed(
        path,
        **payload,
    )

    return path
def setup_output_dirs(
    *,
    out_root: Path,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    mpc,
    u_max: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    save_plots: bool = True,
):
    """
    Create output folders and save basic run metadata.

    Returns
    -------
    frames_dir:
        Directory where frame images are saved.

    log_csv_path:
        Per-step CSV log path.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if save_plots:
        frames_dir = out_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        frames_dir = None

    log_csv_path = out_root / "log.csv"
    log_meta_path = out_root / "meta.json"

    np.save(out_root / "lumen_C.npy", np.asarray(lumen_C, float))
    np.save(out_root / "lumen_R.npy", np.asarray(lumen_R, float))

    meta = {
        "dt": float(mpc.dt),
        "Np": int(mpc.Np),
        "n": int(mpc.n),
        "m": int(mpc.m),
        "np": int(mpc.np),
        "u_max": np.asarray(u_max, float).reshape(-1).tolist(),
        "p_min": np.asarray(p_min, float).reshape(-1).tolist(),
        "p_max": np.asarray(p_max, float).reshape(-1).tolist(),
    }

    with open(log_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return frames_dir, log_csv_path


def append_row_to_csv(log_csv_path: Path, row: dict):
    """
    Append one row to a CSV.

    Uses the row keys as the header if the file does not exist yet.
    """
    log_csv_path = Path(log_csv_path)
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_csv_path.exists()

    with open(log_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists or log_csv_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)


def save_current_figure(
    *,
    frames_dir: Path,
    k: int,
    dpi: int = 160,
):
    """
    Save the current matplotlib figure if one exists.

    Returns
    -------
    fig_path:
        Path to saved frame, or empty string if no figure was present.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.gcf()

    if fig is not None and len(fig.axes) > 0:
        fig_path = frames_dir / f"frame_{k:06d}.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    return ""


def _pad(v, n: int):
    v = np.asarray(v, float).reshape(-1)

    if v.size < n:
        v = np.pad(v, (0, n - v.size), constant_values=np.nan)

    return v[:n]


def _angle_deg(u, v, eps=1e-12):
    u = np.asarray(u, float).reshape(3)
    v = np.asarray(v, float).reshape(3)

    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)

    if un < eps or vn < eps:
        return np.nan

    c = float(np.clip(np.dot(u / un, v / vn), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def closest_point_polyline(C: np.ndarray, x: np.ndarray):
    """
    Closest point on polyline C to point x.

    Returns
    -------
    i_seg:
        Segment index.

    u:
        Segment interpolation parameter in [0, 1].

    c:
        Closest point, shape (3,).

    d2:
        Squared distance.
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3)

    if C.ndim != 2 or C.shape[1] < 3:
        raise ValueError(f"C must have shape (M, >=3), got {C.shape}.")

    if C.shape[0] < 2:
        raise ValueError("Polyline must contain at least two points.")

    C3 = C[:, :3]

    V = C3[1:] - C3[:-1]
    W = x.reshape(1, 3) - C3[:-1]

    VV = np.sum(V * V, axis=1) + 1e-15

    u = np.sum(W * V, axis=1) / VV
    u = np.clip(u, 0.0, 1.0)

    P = C3[:-1] + u.reshape(-1, 1) * V

    d2 = np.sum((P - x.reshape(1, 3)) ** 2, axis=1)

    i_seg = int(np.argmin(d2))

    return i_seg, float(u[i_seg]), P[i_seg], float(d2[i_seg])


def closest_centerline_tangent(C: np.ndarray, x: np.ndarray):
    """
    Return closest centreline point and local tangent.
    """
    C = np.asarray(C, float)
    x = np.asarray(x, float).reshape(3)

    i_seg, u_seg, c_closest, _ = closest_point_polyline(C[:, :3], x)

    i_seg = int(np.clip(i_seg, 0, C.shape[0] - 2))

    t = C[i_seg + 1, :3] - C[i_seg, :3]
    t = t / (np.linalg.norm(t) + 1e-12)

    return i_seg, float(u_seg), c_closest.reshape(3), t.reshape(3)


def compute_centreline_metrics(
    *,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray | None,
    tip_pos: np.ndarray,
    tip_tan: np.ndarray,
):
    """
    Compute vessel-relative diagnostic quantities.
    """
    Cc = np.asarray(lumen_C, float)

    if Cc.ndim != 2 or Cc.shape[1] < 3:
        raise ValueError(f"lumen_C must have shape (M, >=3), got {Cc.shape}.")

    tip_pos = _pad(tip_pos, 3)
    tip_tan = _pad(tip_tan, 3)

    i_seg, u_seg, c_closest, t_vessel = closest_centerline_tangent(Cc, tip_pos)

    tip_vessel_angle_deg = _angle_deg(tip_tan, t_vessel)

    r = tip_pos - c_closest
    axial_component = float(np.dot(r, t_vessel))
    r_perp = r - axial_component * t_vessel

    rho = float(np.linalg.norm(r_perp))

    clearance = np.nan

    if lumen_R is not None:
        Rr = np.asarray(lumen_R, float).reshape(-1)

        if Rr.size > 0:
            idx_v = int(np.clip(i_seg + int(u_seg >= 0.5), 0, Rr.size - 1))
            clearance = float(Rr[idx_v] - rho)

    return {
        "i_seg": int(i_seg),
        "u_seg": float(u_seg),
        "rho_m": float(rho),
        "rho_mm": 1e3 * float(rho),
        "clearance_m": float(clearance),
        "clearance_mm": 1e3 * float(clearance),
        "tip_vessel_angle_deg": float(tip_vessel_angle_deg),
    }

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
    lumen_C,
    lumen_R,
    tip_pos,
    tip_tan,
    save_plots: bool = True,
):
    """
    Save one simulation step.

    This version supports:
        - fixed rollout
        - existing adaptive rollout
        - hierarchical MPC
        - hybrid adaptive-rollout + hierarchical solver-fidelity scheduling
    """
    frames_dir = Path(frames_dir) if frames_dir is not None else None
    log_csv_path = Path(log_csv_path)

    u0 = _pad(u0, 7)
    p_now = _pad(p_now, 8)
    y_now = _pad(y_now, 6)

    idx_ref = np.asarray(info.get("idx_ref", []), int).reshape(-1)
    i_ref_mpc = int(idx_ref[0]) if idx_ref.size else int(i_ref)

    centreline_metrics = compute_centreline_metrics(
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        tip_pos=tip_pos,
        tip_tan=tip_tan,
    )

    if save_plots and frames_dir is not None:
        fig_path = save_current_figure(
            frames_dir=frames_dir,
            k=k,
        )
    else:
        fig_path = ""

    U_applied = np.asarray(info.get("U_applied", []), float)
    x_rollout = np.asarray(info.get("x_rollout", []), float)
    X_pred = np.asarray(info.get("X_pred", []), float)
    U_seq = np.asarray(info.get("U_seq", []), float)

    if U_applied.ndim != 2:
        U_applied = np.empty((0, 7), float)

    if x_rollout.ndim != 2:
        x_rollout = np.empty((0, 6), float)

    if X_pred.ndim != 2:
        X_pred = np.empty((0, 6), float)

    if U_seq.ndim != 2:
        U_seq = np.empty((0, 7), float)

    # Important:
    # Prefer rollout_steps_used because this is the actual number applied
    # by fixed, adaptive, or hierarchical modes.
    rollout_steps_logged = int(
        info.get("rollout_steps_used", info.get("rollout_steps", 0))
    )

    K_roll = min(
        rollout_steps_logged,
        U_applied.shape[0],
        x_rollout.shape[0],
        X_pred.shape[0],
    )

    sqp_hist = info.get("sqp_hist", []) or []
    sqp_iters_done = len(sqp_hist)

    if sqp_iters_done > 0:
        sqp_du_final = float(sqp_hist[-1].get("step_norm", np.nan))
        sqp_du_rel_final = float(sqp_hist[-1].get("rel_step_norm", np.nan))
    else:
        sqp_du_final = np.nan
        sqp_du_rel_final = np.nan

    # Prefer rollout-aligned first-step error when available.
    if (
        X_pred.ndim == 2
        and x_rollout.ndim == 2
        and X_pred.shape[0] > 0
        and x_rollout.shape[0] > 0
    ):
        pred_tip_0 = _pad(X_pred[0], 6)
        roll_tip_0 = _pad(x_rollout[0], 6)

        pred1_err_xy = float(np.linalg.norm(pred_tip_0[:2] - roll_tip_0[:2]))
        pred1_err_xyz = float(np.linalg.norm(pred_tip_0[:3] - roll_tip_0[:3]))

    elif X_pred.shape[0] > 0:
        pred_tip_0 = _pad(X_pred[0], 6)

        pred1_err_xy = float(np.linalg.norm(pred_tip_0[:2] - y_now[:2]))
        pred1_err_xyz = float(np.linalg.norm(pred_tip_0[:3] - y_now[:3]))

    else:
        pred1_err_xy = np.nan
        pred1_err_xyz = np.nan

    row = {
        "k": int(k),

        # Solver status.
        "status": str(info.get("status", "")),
        "infeasible": int(info.get("infeasible", -1)),

        # Old/general fields.
        "solver_mode": str(
            info.get("solver_mode_used", info.get("solver_mode", ""))
        ),
        "rollout_steps": int(rollout_steps_logged),

        # Explicit active-mode fields.
        "solver_mode_used": str(info.get("solver_mode_used", "")),
        "N_sqp_used": info.get("N_sqp_used", ""),
        "rollout_steps_used": info.get("rollout_steps_used", rollout_steps_logged),
        "rollout_steps_next": info.get("rollout_steps_next", ""),
        "rollout_adaptation_reason": info.get("rollout_adaptation_reason", ""),

        # Hierarchical MPC fields.
        "hierarchy_enabled": info.get("hierarchy_enabled", ""),
        "hierarchy_solver_mode": info.get("hierarchy_solver_mode", ""),
        "hierarchy_rollout_steps": info.get("hierarchy_rollout_steps", ""),
        "hierarchy_N_sqp": info.get("hierarchy_N_sqp", ""),
        "hierarchy_confidence": info.get("hierarchy_confidence", ""),
        "hierarchy_risk": info.get("hierarchy_risk", ""),
        "hierarchy_level": info.get("hierarchy_level", ""),
        "hierarchy_reason": info.get("hierarchy_reason", ""),

        # Reference.
        "i_ref": int(i_ref),
        "i_ref_mpc": int(i_ref_mpc),
        "idx_ref_0": int(idx_ref[0]) if idx_ref.size > 0 else -1,
        "idx_ref_terminal": int(idx_ref[-1]) if idx_ref.size > 0 else -1,

        # First applied control.
        "u0_vx": float(u0[0]),
        "u0_vy": float(u0[1]),
        "u0_vz": float(u0[2]),
        "u0_wx": float(u0[3]),
        "u0_wy": float(u0[4]),
        "u0_wz": float(u0[5]),
        "u0_dL": float(u0[6]),

        # Magnet/source state.
        "p_now_x": float(p_now[0]),
        "p_now_y": float(p_now[1]),
        "p_now_z": float(p_now[2]),
        "p_now_qw": float(p_now[3]),
        "p_now_qx": float(p_now[4]),
        "p_now_qy": float(p_now[5]),
        "p_now_qz": float(p_now[6]),
        "p_now_L": float(p_now[7]),

        # Tip state.
        "tip_x": float(y_now[0]),
        "tip_y": float(y_now[1]),
        "tip_z": float(y_now[2]),
        "tip_tx": float(y_now[3]),
        "tip_ty": float(y_now[4]),
        "tip_tz": float(y_now[5]),

        # First-step prediction error.
        "pred1_err_xy_m": float(pred1_err_xy),
        "pred1_err_xy_mm": 1e3 * float(pred1_err_xy),
        "pred1_err_xyz_m": float(pred1_err_xyz),
        "pred1_err_xyz_mm": 1e3 * float(pred1_err_xyz),

        # SQP.
        "sqp_iters_done": int(sqp_iters_done),
        "sqp_du_final": float(sqp_du_final),
        "sqp_du_rel_final": float(sqp_du_rel_final),

        "frame_path": str(fig_path),
    }
    row.update(
        safe_progress_log_row(info)
    )
    # Current geometry metrics.
    row.update(centreline_metrics)

    # ------------------------------------------------------------
    # Trust radius
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Adaptive rollout and Hessian diagnostics
    # ------------------------------------------------------------
    row.update(
        {
            # Exact values used by update_rollout_steps(...)
            "adapt_pred_err_xy_mm": info.get("adapt_pred_err_xy_mm", ""),
            "adapt_pred_err_xyz_mm": info.get("adapt_pred_err_xyz_mm", ""),
            "adapt_clearance_mm": info.get("adapt_clearance_mm", ""),
            "adapt_tip_vessel_angle_deg": info.get(
                "adapt_tip_vessel_angle_deg", ""
            ),

            # Beam Hessian.
            "cond_H_beam": info.get("cond_H_beam", ""),
            "lambda_min_H_beam": info.get("lambda_min_H_beam", ""),
            "lambda_max_H_beam": info.get("lambda_max_H_beam", ""),
            "num_negative_H_beam": info.get("num_negative_H_beam", ""),
            "num_near_zero_H_beam": info.get("num_near_zero_H_beam", ""),
            "stationarity_norm": info.get("stationarity_norm", ""),

            # MPC Hessian.
            "cond_H_mpc": info.get("cond_H_mpc", ""),
            "lambda_min_H_mpc": info.get("lambda_min_H_mpc", ""),
            "lambda_max_H_mpc": info.get("lambda_max_H_mpc", ""),
            "num_negative_H_mpc": info.get("num_negative_H_mpc", ""),
            "num_near_zero_H_mpc": info.get("num_near_zero_H_mpc", ""),
            "valid_H_mpc": info.get("valid_H_mpc", ""),

            "norm_H_mpc": info.get("norm_H_mpc", ""),
            "norm_H_track": info.get("norm_H_track", ""),
            "norm_H_effort": info.get("norm_H_effort", ""),
            "norm_H_smooth": info.get("norm_H_smooth", ""),
            "qp_reg": info.get("qp_reg", ""),
        }
    )
    row.update(
        {
            "path_idx_nearest": info.get("path_idx_nearest", ""),
            "curvature_local_1pm": info.get("curvature_local_1pm", ""),
            "curvature_max_ahead_1pm": info.get("curvature_max_ahead_1pm", ""),
            "curvature_mean_ahead_1pm": info.get("curvature_mean_ahead_1pm", ""),
            "curvature_lookahead_points": info.get("curvature_lookahead_points", ""),
        }
    )


    # ------------------------------------------------------------
    # SQP iteration diagnostics
    # ------------------------------------------------------------
    K_SQP_MAX_LOG = int(info.get("sqp_max_log", 5))

    for j in range(K_SQP_MAX_LOG):
        if j < len(sqp_hist):
            h = sqp_hist[j]
            row[f"sqp_status_{j}"] = str(h.get("status", ""))
            row[f"sqp_du_{j}"] = float(h.get("step_norm", np.nan))
            row[f"sqp_du_rel_{j}"] = float(h.get("rel_step_norm", np.nan))
        else:
            row[f"sqp_status_{j}"] = ""
            row[f"sqp_du_{j}"] = np.nan
            row[f"sqp_du_rel_{j}"] = np.nan

    # ------------------------------------------------------------
    # Applied rollout traces
    # ------------------------------------------------------------
    K_ROLL_MAX_LOG = int(info.get("rollout_max_log", 5))

    for j in range(K_ROLL_MAX_LOG):
        if j < K_roll:
            uj = _pad(U_applied[j], 7)
            xj = _pad(x_rollout[j], 6)
            xpj = _pad(X_pred[j], 6)

            err_xy = float(np.linalg.norm(xj[:2] - xpj[:2]))
            err_xyz = float(np.linalg.norm(xj[:3] - xpj[:3]))
        else:
            uj = np.full(7, np.nan)
            xj = np.full(6, np.nan)
            xpj = np.full(6, np.nan)
            err_xy = np.nan
            err_xyz = np.nan

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

                f"rollout_err_xy_{j}_m": float(err_xy),
                f"rollout_err_xy_{j}_mm": 1e3 * float(err_xy),
                f"rollout_err_xyz_{j}_m": float(err_xyz),
                f"rollout_err_xyz_{j}_mm": 1e3 * float(err_xyz),
            }
        )

    # ------------------------------------------------------------
    # Planned control sequence
    # ------------------------------------------------------------
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
    row.update(
        {
            "mpc_eig_logged": info.get("mpc_eig_logged", ""),
            "mpc_eig_failed": info.get("mpc_eig_failed", ""),
            "mpc_eig_cond": info.get("mpc_eig_cond", ""),
            "mpc_weak_lambda": info.get("mpc_weak_lambda", ""),
            "mpc_strong_lambda": info.get("mpc_strong_lambda", ""),
            "mpc_H_shape_0": info.get("mpc_H_shape_0", ""),
            "mpc_H_shape_1": info.get("mpc_H_shape_1", ""),

            "mpc_weak_channel_index": info.get("mpc_weak_channel_index", ""),
            "mpc_weak_channel_name": info.get("mpc_weak_channel_name", ""),
            "mpc_weak_channel_energy": info.get("mpc_weak_channel_energy", ""),

            "mpc_strong_channel_index": info.get("mpc_strong_channel_index", ""),
            "mpc_strong_channel_name": info.get("mpc_strong_channel_name", ""),
            "mpc_strong_channel_energy": info.get("mpc_strong_channel_energy", ""),

            "mpc_weak_energy_vx": info.get("mpc_weak_energy_vx", ""),
            "mpc_weak_energy_vy": info.get("mpc_weak_energy_vy", ""),
            "mpc_weak_energy_vz": info.get("mpc_weak_energy_vz", ""),
            "mpc_weak_energy_wx": info.get("mpc_weak_energy_wx", ""),
            "mpc_weak_energy_wy": info.get("mpc_weak_energy_wy", ""),
            "mpc_weak_energy_wz": info.get("mpc_weak_energy_wz", ""),
            "mpc_weak_energy_dL": info.get("mpc_weak_energy_dL", ""),

            "mpc_strong_energy_vx": info.get("mpc_strong_energy_vx", ""),
            "mpc_strong_energy_vy": info.get("mpc_strong_energy_vy", ""),
            "mpc_strong_energy_vz": info.get("mpc_strong_energy_vz", ""),
            "mpc_strong_energy_wx": info.get("mpc_strong_energy_wx", ""),
            "mpc_strong_energy_wy": info.get("mpc_strong_energy_wy", ""),
            "mpc_strong_energy_wz": info.get("mpc_strong_energy_wz", ""),
            "mpc_strong_energy_dL": info.get("mpc_strong_energy_dL", ""),
        }
    )
    K_BEAM_DIR_LOG = 20

    for j in range(K_BEAM_DIR_LOG):
        row[f"beam_soft_dir_{j}"] = info.get(f"beam_soft_dir_{j}", "")
    row.update(
        {
            "beam_stiff_direction_index": info.get("beam_stiff_direction_index", ""),
            "beam_stiff_direction_absmax": info.get("beam_stiff_direction_absmax", ""),
        }
    )

    K_BEAM_DIR_LOG = 20
    stage_rows = build_rollout_stage_rows(
        k=k,
        info=info,
        lumen_C=lumen_C,
        controller_name=str(
            info.get(
                "controller_type",
                info.get("solver_mode", ""),
            )
        ),
    )

    stage_csv_path = (
        log_csv_path.parent
        / "rollout_stages.csv"
    )

    append_rollout_stage_csv(
        stage_csv_path,
        stage_rows,
    )

    rollout_frame_dir = (
        log_csv_path.parent
        / "rollout_frames"
    )
    for key in (
        "U_seq",
        "B_sequence",
        "B_sequence_linearisation",
        "B_sequence_solution",
        "B_drift_relative_stage",
    ):
        value = info.get(key, None)

        print(
            f"[BEFORE NPZ] {key:32s}: "
            f"{None if value is None else np.asarray(value).shape}"
        )
    save_rollout_npz(
        rollout_frame_dir
        / f"rollout_{k:04d}.npz",
        k=k,
        info=info,
        lumen_C=lumen_C,
    )
    for j in range(K_BEAM_DIR_LOG):
        row[f"beam_soft_dir_{j}"] = info.get(f"beam_soft_dir_{j}", "")
        row[f"beam_stiff_dir_{j}"] = info.get(f"beam_stiff_dir_{j}", "")
    append_row_to_csv(log_csv_path, row)