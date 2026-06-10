from pathlib import Path

import numpy as np
from .adaptive_rollout import update_rollout_steps
from beam_direction_magnetisation.post_processing.post_processing import (
    plot_energy_only_3d,
)
from proper_research.hardware.hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
    HierarchicalMPCState,
    choose_hierarchical_mpc_action,
)
from proper_research.simulation_controller.diagnostics import (
    rollout_error_metrics,
    jacobian_svd_diagnostics,
)
from proper_research.simulation.simulations.curvature_utils import (
    compute_discrete_path_curvature,
    curvature_ahead_metrics,
)
from proper_research.simulation.simulations.logging_utils import save_step_artifacts, compute_centreline_metrics


def snapshot_forward(forward6d, p8, *, commit=False):
    """
    Get a consistent forward-model snapshot from a single forward call.

    Returns
    -------
    tip:
        Tip position, shape (3,)

    tan:
        Tip tangent, shape (3,)

    C:
        Centreline/shape from the forward model, if available.

    y:
        Full output, usually [tip_x, tip_y, tip_z, tx, ty, tz].
    """
    y = np.asarray(forward6d(p8, commit=commit), float).reshape(-1)

    if y.size < 6:
        raise ValueError(f"forward6d returned {y.size} values, expected at least 6.")

    y = y[:6]

    C = getattr(forward6d, "last_p_centerline", None)

    if C is not None:
        C = np.asarray(C, float).copy()

    tip = y[:3].copy()
    tan = y[3:6].copy()

    return tip, tan, C, y


def _compute_rollout_errors(X_pred, x_rollout):
    """
    Compute rollout-vs-prediction errors.
    """
    X_pred = np.asarray(X_pred, float)
    x_rollout = np.asarray(x_rollout, float)

    if X_pred.ndim != 2 or x_rollout.ndim != 2:
        return np.full(0, np.nan), np.full(0, np.nan)

    K = min(X_pred.shape[0], x_rollout.shape[0])

    if K <= 0:
        return np.full(0, np.nan), np.full(0, np.nan)

    err_xy = np.linalg.norm(
        x_rollout[:K, :2] - X_pred[:K, :2],
        axis=1,
    )

    err_xyz = np.linalg.norm(
        x_rollout[:K, :3] - X_pred[:K, :3],
        axis=1,
    )

    return err_xy, err_xyz
def beam_eigen_direction_diagnostics(
    H_beam,
    *,
    cond_warn: float = 5e3,
    prefix: str = "beam",
) -> dict:
    out = {
        f"{prefix}_eig_logged": False,
        f"{prefix}_eig_failed": False,
        f"{prefix}_eig_lambda_min": np.nan,
        f"{prefix}_eig_lambda_max": np.nan,
        f"{prefix}_eig_cond": np.nan,
        f"{prefix}_soft_direction_index": -1,
        f"{prefix}_soft_direction_absmax": np.nan,
    }

    if H_beam is None:
        return out

    H = np.asarray(H_beam, float)

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        return out

    H = 0.5 * (H + H.T)

    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        out[f"{prefix}_eig_failed"] = True
        return out

    abs_eigs = np.abs(eigvals)

    if not np.all(np.isfinite(abs_eigs)):
        return out

    idx_min = int(np.argmin(abs_eigs))
    idx_max = int(np.argmax(abs_eigs))

    lam_min = float(eigvals[idx_min])
    lam_max = float(eigvals[idx_max])

    cond = float(abs_eigs[idx_max] / max(abs_eigs[idx_min], 1e-12))

    out[f"{prefix}_eig_lambda_min"] = lam_min
    out[f"{prefix}_eig_lambda_max"] = lam_max
    out[f"{prefix}_eig_cond"] = cond

    if cond < cond_warn:
        return out

    v = np.asarray(eigvecs[:, idx_min], float)
    v = v / max(np.linalg.norm(v), 1e-12)

    dominant_index = int(np.argmax(np.abs(v)))

    out[f"{prefix}_eig_logged"] = True
    out[f"{prefix}_soft_direction_index"] = dominant_index
    out[f"{prefix}_soft_direction_absmax"] = float(np.abs(v[dominant_index]))

    for j, value in enumerate(v):
        out[f"{prefix}_soft_dir_{j}"] = float(value)

    return out

def _safe_jacobian_diagnostics(mpc, p_pre):
    """
    Compute local prediction-matrix/Jacobian diagnostics.

    Returns NaNs if the diagnostic call fails.
    """
    try:
        _, _, _, B0 = mpc._build_prediction_mats(p_pre, U_guess=None)

        jac_diag = jacobian_svd_diagnostics(B0)

        S = np.asarray(jac_diag["singular_values"], float)
        cond = float(jac_diag["condition_number"])
        rank = int(jac_diag["rank"])

        return S, cond, rank

    except Exception:
        S = np.full(min(mpc.n, mpc.m), np.nan)
        cond = np.nan
        rank = -1

        return S, cond, rank


def run_simulation(
    *,
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
    max_steps: int = 200,
    rollout_steps: int = 1,
    solver_mode: str = "sqp_full",
    save_plots: bool = True,
    adaptive_rollout_enabled: bool = False,
    hierarchical_mpc_enabled: bool = False,
    hierarchical_policy_config: HierarchicalMPCPolicyConfig | None = None,
):
    """
    Run one closed-loop MPC simulation.

    Supports four operating modes:

        1. Fixed rollout + fixed solver
        2. Existing adaptive rollout + fixed solver
        3. Hierarchical MPC controlling rollout + solver fidelity
        4. Hybrid: existing adaptive rollout controls rollout,
           hierarchy controls solver fidelity and N_sqp

    If both adaptive_rollout_enabled and hierarchical_mpc_enabled are True,
    the existing adaptive rollout controller is preserved and owns the rollout
    update. The hierarchy only chooses solver_mode_current and N_sqp_current.
    """
    frames_dir = Path(frames_dir) if frames_dir is not None else None
    log_csv_path = Path(log_csv_path)

    max_steps = int(max_steps)
    rollout_steps_max = int(rollout_steps)

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    if rollout_steps_max <= 0:
        raise ValueError("rollout_steps must be positive.")

    fixed_limits = None

    # ------------------------------------------------------------
    # Rollout state
    # ------------------------------------------------------------
    if adaptive_rollout_enabled:
        rollout_steps_current = 1
    elif hierarchical_mpc_enabled:
        rollout_steps_current = 1
    else:
        rollout_steps_current = rollout_steps_max

    rollout_steps_current = int(np.clip(
        rollout_steps_current,
        1,
        rollout_steps_max,
    ))

    # ------------------------------------------------------------
    # Hierarchy state
    # ------------------------------------------------------------
    if hierarchical_policy_config is None:
        hierarchical_policy_config = HierarchicalMPCPolicyConfig(
            enabled=hierarchical_mpc_enabled,
            rollout_max=rollout_steps_max,
            sqp_full_iters=int(getattr(mpc, "N_sqp", 5)),
        )

    hierarchy_state = HierarchicalMPCState(
        rollout_steps=rollout_steps_current,
        previous_level=0,
        downgrade_counter=0,
    )

    # This is deliberately updated at the END of each loop, after all
    # diagnostics have been attached to info.
    previous_info = {
        "status": "solved",
        "infeasible": False,
    }

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
        "jac_rank": [],
        "solver_mode_used": [],
        "N_sqp_used": [],
        "rollout_steps_used": [],
        "hierarchy_level": [],
        "hierarchy_confidence": [],
        "hierarchy_risk": [],
    }
    path_for_curvature = np.asarray(lumen_path, float)

    if path_for_curvature.ndim != 2 or path_for_curvature.shape[1] < 3:
        path_for_curvature = np.asarray(lumen_C, float)

    path_curvature = compute_discrete_path_curvature(path_for_curvature)
    for k in range(max_steps):
        # ------------------------------------------------------------
        # Pre-step state snapshot
        # ------------------------------------------------------------
        p_pre = np.asarray(mpc.p, float).copy()
        x_pre = np.asarray(mpc.x, float).copy()
        tip_pre = x_pre[:3].copy()

        tip_pre_vis, tan_pre, C_pre, y_pre = snapshot_forward(
            forward6d,
            p_pre,
            commit=False,
        )

        centreline_metrics_pre = compute_centreline_metrics(
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip_pre_vis,
            tip_tan=tan_pre,
        )

        # ------------------------------------------------------------
        # Preventive hierarchy input
        # ------------------------------------------------------------
        curvature_metrics_pre = curvature_ahead_metrics(
            path=path_for_curvature,
            curvature=path_curvature,
            tip_pos=tip_pre_vis,
            lookahead_points=10,
        )

        info_pre = dict(previous_info)
        info_pre.update(centreline_metrics_pre)
        info_pre.update(curvature_metrics_pre)

        info_pre["clearance_mm"] = float(info_pre.get("clearance_mm", np.nan))
        info_pre["tip_vessel_angle_deg"] = float(
            info_pre.get("tip_vessel_angle_deg", np.nan)
        )
        # ------------------------------------------------------------
        # Choose solver fidelity and rollout
        # ------------------------------------------------------------
        decision = None

        if hierarchical_mpc_enabled:
            decision, hierarchy_state = choose_hierarchical_mpc_action(
                info_pre=info_pre,
                state=hierarchy_state,
                cfg=hierarchical_policy_config,
            )

            solver_mode_current = decision.solver_mode
            N_sqp_current = int(decision.N_sqp)

            if adaptive_rollout_enabled:
                # Hybrid mode:
                # Existing adaptive rollout owns rollout.
                # Hierarchy owns solver fidelity and N_sqp.
                rollout_steps_current = int(np.clip(
                    rollout_steps_current,
                    1,
                    rollout_steps_max,
                ))
            else:
                # Pure hierarchy mode:
                # Hierarchy owns rollout as well.
                rollout_steps_current = int(np.clip(
                    decision.rollout_steps,
                    1,
                    rollout_steps_max,
                ))

        else:
            solver_mode_current = solver_mode
            N_sqp_current = int(getattr(mpc, "N_sqp", 1))

            # Important:
            # Do NOT reset rollout_steps_current here.
            # If adaptive rollout is enabled, it was updated at the end
            # of the previous loop.
            rollout_steps_current = int(np.clip(
                rollout_steps_current,
                1,
                rollout_steps_max,
            ))

        # ------------------------------------------------------------
        # Local Jacobian diagnostics before solve
        # ------------------------------------------------------------
        S, cond, rank = _safe_jacobian_diagnostics(mpc, p_pre)

        # ------------------------------------------------------------
        # Solve/apply MPC
        # ------------------------------------------------------------
        old_N_sqp = int(getattr(mpc, "N_sqp", N_sqp_current))
        mpc.N_sqp = int(N_sqp_current)

        try:
            p_post, x_post, info = mpc.step(
                x_meas=None,
                rollout_steps=rollout_steps_current,
                solver_mode=solver_mode_current,
            )
        finally:
            mpc.N_sqp = old_N_sqp
        # print("[MPC EIG DEBUG]", {
        #     "cond_H_mpc": info.get("cond_H_mpc", None),
        #     "mpc_eig_cond": info.get("mpc_eig_cond", None),
        #     "mpc_eig_logged": info.get("mpc_eig_logged", None),
        #     "mpc_eig_failed": info.get("mpc_eig_failed", None),
        #     "H_shape": (
        #         info.get("mpc_H_shape_0", None),
        #         info.get("mpc_H_shape_1", None),
        #     ),
        #     "weak_channel": info.get("mpc_weak_channel_name", None),
        #     "weak_energy": info.get("mpc_weak_channel_energy", None),
        #     "strong_channel": info.get("mpc_strong_channel_name", None),
        #     "strong_energy": info.get("mpc_strong_channel_energy", None),
        # })

        p_post = np.asarray(p_post, float)
        x_post = np.asarray(x_post, float)
        tip_post = x_post[:3].copy()

        # ------------------------------------------------------------
        # Attach active-mode metadata
        # ------------------------------------------------------------
        info["solver_mode_used"] = str(solver_mode_current)
        info["N_sqp_used"] = int(N_sqp_current)
        info["rollout_steps_used"] = int(rollout_steps_current)

        if hierarchical_mpc_enabled and decision is not None:
            info["hierarchy_enabled"] = True
            info["hierarchy_solver_mode"] = decision.solver_mode
            info["hierarchy_rollout_steps"] = int(decision.rollout_steps)
            info["hierarchy_N_sqp"] = int(decision.N_sqp)
            info["hierarchy_confidence"] = float(decision.confidence)
            info["hierarchy_risk"] = float(decision.risk)
            info["hierarchy_level"] = int(decision.level)
            info["hierarchy_reason"] = str(decision.reason)
        else:
            info["hierarchy_enabled"] = False
            info["hierarchy_solver_mode"] = ""
            info["hierarchy_rollout_steps"] = ""
            info["hierarchy_N_sqp"] = ""
            info["hierarchy_confidence"] = np.nan
            info["hierarchy_risk"] = np.nan
            info["hierarchy_level"] = ""
            info["hierarchy_reason"] = ""

        # ------------------------------------------------------------
        # Prediction/rollout error diagnostics
        # ------------------------------------------------------------
        X_pred = info.get("X_pred", None)
        x_rollout = info.get("x_rollout", None)

        rollout_err_xy, rollout_err_xyz = _compute_rollout_errors(
            X_pred,
            x_rollout,
        )

        if rollout_err_xy.size > 0:
            rollout_err_xy_mm = 1e3 * np.asarray(rollout_err_xy, float)

            info["rollout_err_xy_0_mm"] = float(rollout_err_xy_mm[0])
            info["pred1_err_xy_mm"] = float(rollout_err_xy_mm[0])
            info["adapt_pred_err_xy_mm"] = float(np.nanmax(rollout_err_xy_mm))
        else:
            info["rollout_err_xy_0_mm"] = np.nan
            info["pred1_err_xy_mm"] = np.nan
            info["adapt_pred_err_xy_mm"] = np.nan

        if rollout_err_xyz.size > 0:
            rollout_err_xyz_mm = 1e3 * np.asarray(rollout_err_xyz, float)

            info["rollout_err_xyz_0_mm"] = float(rollout_err_xyz_mm[0])
            info["pred1_err_xyz_mm"] = float(rollout_err_xyz_mm[0])
            info["adapt_pred_err_xyz_mm"] = float(np.nanmax(rollout_err_xyz_mm))
        else:
            info["rollout_err_xyz_0_mm"] = np.nan
            info["pred1_err_xyz_mm"] = np.nan
            info["adapt_pred_err_xyz_mm"] = np.nan

        # Existing adaptive rollout reads pred1_err_xy_mm.
        # Deliberately point it at the worst applied-rollout error.
        info["pred1_err_xy_mm"] = float(
            info.get("adapt_pred_err_xy_mm", np.nan)
        )

        # ------------------------------------------------------------
        # SQP diagnostics
        # ------------------------------------------------------------
        sqp_hist = info.get("sqp_hist", []) or []

        if len(sqp_hist) > 0:
            sqp_final_step = float(sqp_hist[-1].get("step_norm", np.nan))
        else:
            sqp_final_step = np.nan
        if len(sqp_hist) > 0:
            sqp_final_step = float(sqp_hist[-1].get("step_norm", np.nan))
            sqp_final_rel_step = float(sqp_hist[-1].get("rel_step_norm", np.nan))
        else:
            sqp_final_step = np.nan
            sqp_final_rel_step = np.nan

        info["sqp_du_final"] = float(sqp_final_step)
        info["sqp_du_rel_final"] = float(sqp_final_rel_step)
        # ------------------------------------------------------------
        # Reference index
        # ------------------------------------------------------------
        idx_ref = np.asarray(info.get("idx_ref", []), int).reshape(-1)

        if idx_ref.size > 0:
            i_ref = int(idx_ref[0])
        else:
            i_ref = int(getattr(mpc, "i_ref_last", 0))

        # ------------------------------------------------------------
        # Post-step geometry diagnostics
        # ------------------------------------------------------------
        tip_post_vis, tan_post, C_post, y_post = snapshot_forward(
            forward6d,
            p_post,
            commit=False,
        )

        centreline_metrics = compute_centreline_metrics(
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip_post_vis,
            tip_tan=tan_post,
        )

        info.update(centreline_metrics)
        info.update(curvature_metrics_pre)

        info["adapt_clearance_mm"] = float(info.get("clearance_mm", np.nan))
        info["adapt_tip_vessel_angle_deg"] = float(
            info.get("tip_vessel_angle_deg", np.nan)
        )

        info["adapt_clearance_mm"] = float(
            info.get("clearance_mm", np.nan)
        )
        info["adapt_tip_vessel_angle_deg"] = float(
            info.get("tip_vessel_angle_deg", np.nan)
        )

        # ------------------------------------------------------------
        # Existing adaptive rollout update
        # ------------------------------------------------------------
        rollout_steps_next, rollout_reason = update_rollout_steps(
            current_rollout_steps=rollout_steps_current,
            info=info,
            rollout_min=1,
            rollout_max=rollout_steps_max,
        )

        info["rollout_steps_next"] = int(rollout_steps_next)
        info["rollout_adaptation_reason"] = str(rollout_reason)

        # Ownership rule:
        # - adaptive rollout enabled: old adaptive controller owns next rollout
        # - hierarchy enabled only: hierarchy owns next rollout
        # - neither: fixed rollout
        if adaptive_rollout_enabled:
            rollout_steps_current = int(np.clip(
                rollout_steps_next,
                1,
                rollout_steps_max,
            ))
        elif hierarchical_mpc_enabled and decision is not None:
            rollout_steps_current = int(np.clip(
                decision.rollout_steps,
                1,
                rollout_steps_max,
            ))
        else:
            rollout_steps_current = int(rollout_steps_max)

        # Keep hierarchy's internal rollout state consistent in hybrid mode.
        if hierarchical_mpc_enabled:
            hierarchy_state.rollout_steps = int(rollout_steps_current)

        # ------------------------------------------------------------
        # History
        # ------------------------------------------------------------
        hist["k"].append(k)
        hist["status"].append(str(info.get("status", "unknown")))
        hist["infeasible"].append(int(info.get("infeasible", 0)))
        hist["i_ref"].append(i_ref)
        hist["tip"].append(tip_post.copy())
        hist["p"].append(p_post.copy())
        hist["u0"].append(
            np.asarray(info.get("u0", np.full(mpc.m, np.nan)), float).copy()
        )
        hist["U_applied"].append(
            np.asarray(info.get("U_applied", []), float).copy()
        )
        hist["rollout_err_xyz"].append(
            np.asarray(rollout_err_xyz, float).copy()
        )
        hist["rollout_err_xy"].append(
            np.asarray(rollout_err_xy, float).copy()
        )
        hist["sqp_iters"].append(len(sqp_hist))
        hist["sqp_final_step_norm"].append(float(sqp_final_step))
        hist["jac_singular_values"].append(np.asarray(S, float).copy())
        hist["jac_cond"].append(float(cond))
        hist["jac_rank"].append(int(rank))
        hist["solver_mode_used"].append(str(solver_mode_current))
        hist["N_sqp_used"].append(int(N_sqp_current))
        hist["rollout_steps_used"].append(
            int(info.get("rollout_steps_used", np.nan))
        )
        hist["hierarchy_level"].append(info.get("hierarchy_level", ""))
        hist["hierarchy_confidence"].append(
            float(info.get("hierarchy_confidence", np.nan))
        )
        hist["hierarchy_risk"].append(
            float(info.get("hierarchy_risk", np.nan))
        )

        # ------------------------------------------------------------
        # Terminal output
        # ------------------------------------------------------------
        print(
            f"[{solver_mode_current}] "
            f"k={k:04d} "
            f"status={info.get('status', 'unknown')} "
            f"infeas={int(info.get('infeasible', 0))} "
            f"i_ref={i_ref:4d} "
            f"rollout={info.get('rollout_steps_used', '')} "
            f"next={info.get('rollout_steps_next', '')} "
            f"reason={info.get('rollout_adaptation_reason', '')} "
            f"hier={info.get('hierarchy_level', '')} "
            f"conf={float(info.get('hierarchy_confidence', np.nan)):.2f} "
            f"risk={float(info.get('hierarchy_risk', np.nan)):.2f} "
            f"Δtip={1e3 * np.linalg.norm(tip_post - tip_pre):7.3f} mm "
            f"pred={float(info.get('adapt_pred_err_xy_mm', np.nan)):.3f} mm "
            f"clear={float(info.get('adapt_clearance_mm', np.nan)):.3f} mm "
            f"angle={float(info.get('adapt_tip_vessel_angle_deg', np.nan)):.2f} deg "
            f"condHb={float(info.get('cond_H_beam', np.nan)):.2e} "
            f"condHm={float(info.get('cond_H_mpc', np.nan)):.2e} "
            f"sqp_it={len(sqp_hist)} "
            f"cond(B0)={cond:.2e}"
        )
        u0_arr = np.asarray(info.get("u0", np.full(mpc.m, np.nan)), float).reshape(-1)

        if u0_arr.size >= 7:
            info["u0_vx"] = float(u0_arr[0])
            info["u0_vy"] = float(u0_arr[1])
            info["u0_vz"] = float(u0_arr[2])
            info["u0_wx"] = float(u0_arr[3])
            info["u0_wy"] = float(u0_arr[4])
            info["u0_wz"] = float(u0_arr[5])
            info["u0_dL"] = float(u0_arr[6])
        # ------------------------------------------------------------
        # Plotting and CSV logging
        # ------------------------------------------------------------
        try:
            if save_plots:
                fixed_limits = plot_energy_only_3d(
                    C_pre,
                    lumen_C=lumen_C,
                    lumen_R=lumen_R,
                    p0=p0_ur,
                    tip=tip_pre,
                    p_mag=p_post[:3],
                    show=False,
                    fixed_limits=fixed_limits,
                    zoom_out=1.5,
                )

            save_step_artifacts(
                k=k,
                frames_dir=frames_dir,
                log_csv_path=log_csv_path,
                u0=info.get("u0", np.full(mpc.m, np.nan)),
                p_now=p_post,
                y_now=y_post,
                i_ref=i_ref,
                info=info,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                tip_pos=tip_post_vis,
                tip_tan=tan_post,
                save_plots=save_plots,
            )

        except Exception as e:
            print(f"[WARN] logging failed at k={k}: {e}")

        # ------------------------------------------------------------
        # Update previous info for next preventive hierarchy decision
        # ------------------------------------------------------------
        previous_info = dict(info)

        # ------------------------------------------------------------
        # Stop conditions
        # ------------------------------------------------------------
        if i_ref >= np.asarray(lumen_path).shape[0] - 6:
            print("[DONE] reached final reference region")
            break

        if int(info.get("infeasible", 0)):
            print("[STOP] MPC infeasible")
            break

    return hist