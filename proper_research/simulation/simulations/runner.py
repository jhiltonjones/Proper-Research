from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .adaptive_rollout import update_rollout_steps
from beam_direction_magnetisation.post_processing.post_processing import (
    plot_energy_only_3d,
)
from proper_research.simulation_controller.safe_progress_logging import (
    safe_progress_log_row,
)
from dataclasses import replace
from proper_research.hardware.evidence_based_hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
    HierarchicalMPCState,
    HierarchicalMPCDecision,
    choose_hierarchical_mpc_action,
    evaluate_selected_policy_tolerance,
    update_hierarchical_state_after_solve,
    attach_policy_metadata,
    tolerance_for_zone,
    sqp_cap_for_zone,
)
from proper_research.simulation_controller.diagnostics import (
    rollout_error_metrics,
    jacobian_svd_diagnostics,
)
from proper_research.simulation.simulations.curvature_utils import (
    compute_discrete_path_curvature,
    curvature_ahead_metrics,
)
from proper_research.simulation.simulations.logging_utils import save_step_artifacts, compute_centreline_metrics, save_controller_diagnostics_npz
from proper_research.simulation.simulations.sim_plots import (
    save_step_diagnostic_plots,
)


def _mark_magnet_and_expand_axes(ax, p_mag) -> None:
    """Draw a visible magnet-centre marker and keep it inside all 3-D limits."""
    point = np.asarray(p_mag, dtype=float).reshape(3)
    if not np.all(np.isfinite(point)) or ax is None:
        return

    required = (
        "scatter", "get_xlim3d", "get_ylim3d", "get_zlim3d",
        "set_xlim3d", "set_ylim3d", "set_zlim3d",
    )
    if not all(hasattr(ax, name) for name in required):
        return

    ax.scatter(
        [point[0]], [point[1]], [point[2]],
        s=90,
        marker="D",
        c="crimson",
        edgecolors="black",
        linewidths=0.8,
        depthshade=False,
        label="source magnet centre",
        zorder=20,
    )
    if hasattr(ax, "text"):
        ax.text(point[0], point[1], point[2], "  M", color="crimson")

    axis_specs = (
        (point[0], ax.get_xlim3d, ax.set_xlim3d),
        (point[1], ax.get_ylim3d, ax.set_ylim3d),
        (point[2], ax.get_zlim3d, ax.set_zlim3d),
    )
    for coordinate, getter, setter in axis_specs:
        low, high = map(float, getter())
        new_low = min(low, float(coordinate))
        new_high = max(high, float(coordinate))
        span = max(new_high - new_low, 1.0e-6)
        pad = 0.04 * span
        setter(new_low - pad, new_high + pad)

    if hasattr(ax, "legend"):
        ax.legend(loc="best")


def save_geometry_frames(
    *,
    k: int,
    frames_dir,
    C_beam,
    lumen_C,
    lumen_R,
    p0,
    tip,
    p_mag,
    dpi: int = 180,
    limits=None,  
):
    frames_dir = Path(frames_dir)

    with_mag_dir = frames_dir / "with_source_magnet"
    without_mag_dir = frames_dir / "without_source_magnet"

    with_mag_dir.mkdir(parents=True, exist_ok=True)
    without_mag_dir.mkdir(parents=True, exist_ok=True)

    # Complete-scene limits.
    fig_mag, ax_mag, limits = plot_energy_only_3d(
        C_beam,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        p0=p0,
        tip=tip,
        p_mag=p_mag,
        plot_mag=True,
        title=f"With source magnet, k={k}",
        show=False,
        fixed_limits=limits, 
        zoom_out=1.2,
    )
    # The source magnet can be roughly 0.2 m away from a 0.04 m beam.  Some
    # plot helpers compute limits only from the beam/lumen, so explicitly mark
    # the centre and expand this figure's axes to include it.
    _mark_magnet_and_expand_axes(ax_mag, p_mag)

    fig_mag.savefig(
        with_mag_dir / f"frame_{k:05d}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig_mag)

    fig_no_mag, _, _ = plot_energy_only_3d(
        C_beam,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        p0=p0,
        tip=tip,
        p_mag=p_mag,
        plot_mag=False,
        title=f"Without source magnet, k={k}",
        show=False,
        fixed_limits=None,
        zoom_out=0.65,
    )

    fig_no_mag.savefig(
        without_mag_dir / f"frame_{k:05d}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig_no_mag)

    return {
        "with_magnet": (
            with_mag_dir / f"frame_{k:05d}.png"
        ),
        "without_magnet": (
            without_mag_dir / f"frame_{k:05d}.png"
        ),
        "fixed_limits": limits,
    }, limits
def snapshot_forward(forward6d, state, *, commit=False):
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
    # ``state`` is [q1..q6, insertion] in the joint-space simulator.  The
    # forward adapter performs robot FK and then calls the unchanged beam model.
    y = np.asarray(forward6d(state, commit=commit), float).reshape(-1)

    if y.size < 6:
        raise ValueError(f"forward6d returned {y.size} values, expected at least 6.")

    y = y[:6]

    C = getattr(forward6d, "last_p_centerline", None)

    if C is not None:
        C = np.asarray(C, float).copy()

    tip = y[:3].copy()
    tan = y[3:6].copy()

    return tip, tan, C, y



def _fallback_tip_vessel_angle_deg(
    *,
    lumen_C,
    tip_pos,
    tip_tan,
    eps: float = 1.0e-12,
) -> tuple[float, int]:
    """
    Compute the directed angle between the current beam-tip tangent and the
    local lumen-centreline tangent at the nearest centreline point.

    Returns the angle in [0, 180] degrees and the nearest centreline index.
    A reversed tangent is approximately 180 degrees and is unsafe.
    """
    C = np.asarray(lumen_C, float)
    tip = np.asarray(tip_pos, float).reshape(-1)
    tangent = np.asarray(tip_tan, float).reshape(-1)

    if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] < 3:
        return np.nan, -1
    if tip.size < 3 or tangent.size < 3:
        return np.nan, -1

    C = C[:, :3]
    tip = tip[:3]
    tangent = tangent[:3]

    if not (
        np.all(np.isfinite(C))
        and np.all(np.isfinite(tip))
        and np.all(np.isfinite(tangent))
    ):
        return np.nan, -1

    nearest = int(np.argmin(np.linalg.norm(C - tip[None, :], axis=1)))

    if nearest == 0:
        lumen_tangent = C[1] - C[0]
    elif nearest == C.shape[0] - 1:
        lumen_tangent = C[-1] - C[-2]
    else:
        lumen_tangent = C[nearest + 1] - C[nearest - 1]

    tip_norm = float(np.linalg.norm(tangent))
    lumen_norm = float(np.linalg.norm(lumen_tangent))

    if tip_norm <= eps or lumen_norm <= eps:
        return np.nan, nearest

    tip_unit = tangent / tip_norm
    lumen_unit = lumen_tangent / lumen_norm
    cosine = float(np.clip(np.dot(tip_unit, lumen_unit), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosine)))

    return angle_deg, nearest



def _tip_wall_clearance_m(
    *,
    centreline_metrics,
    lumen_C,
    lumen_R,
    tip_pos,
    tip_radius_m: float = 0.0,
) -> tuple[float, int, str]:
    """
    Return signed tip-to-wall clearance in metres.

    Prefer the existing centreline metric ``clearance_mm`` so the runner uses
    the same geometry convention as the rest of the simulation. Fall back to
    nearest-centreline radial clearance when that metric is unavailable.
    """
    metrics = dict(centreline_metrics or {})

    try:
        clearance_mm = float(
            metrics.get(
                "clearance_mm",
                np.nan,
            )
        )
    except Exception:
        clearance_mm = np.nan

    try:
        index = int(
            metrics.get(
                "centreline_index",
                -1,
            )
        )
    except Exception:
        index = -1

    if np.isfinite(clearance_mm):
        return (
            1e-3 * clearance_mm,
            index,
            "compute_centreline_metrics",
        )

    C = np.asarray(lumen_C, float)
    tip = np.asarray(tip_pos, float).reshape(-1)
    R = np.asarray(lumen_R, float)

    if (
        C.ndim != 2
        or C.shape[0] < 1
        or C.shape[1] < 3
        or tip.size < 3
    ):
        return np.nan, -1, "runner_fallback_invalid"

    C_xyz = C[:, :3]
    tip = tip[:3]
    distances = np.linalg.norm(
        C_xyz - tip.reshape(1, 3),
        axis=1,
    )
    index = int(np.argmin(distances))

    if R.ndim == 0 or R.size == 1:
        radius = float(R.reshape(-1)[0])
    elif R.size == C_xyz.shape[0]:
        radius = float(R.reshape(-1)[index])
    else:
        return np.nan, index, "runner_fallback_invalid_radius"

    tip_radius_m = float(tip_radius_m)
    if (
        not np.isfinite(radius)
        or radius <= 0.0
        or not np.isfinite(tip_radius_m)
        or tip_radius_m < 0.0
    ):
        return np.nan, index, "runner_fallback_invalid_radius"

    clearance = (
        radius
        - float(distances[index])
        - tip_radius_m
    )

    return float(clearance), index, "runner_fallback"

def _tip_tangent_safety_status(
    *,
    centreline_metrics,
    lumen_C,
    lumen_R,
    tip_pos,
    tip_tan,
    angle_limit_deg: float,
    activation_clearance_m: float,
    tip_radius_m: float,
    invalid_is_stop: bool,
) -> dict:
    """
    Evaluate a proximity-gated current-state safety rule.

    Stop only when:

        clearance <= activation_clearance
        and
        tip/lumen tangent angle > angle_limit

    If clearance itself is invalid and fail-closed behaviour is enabled, stop.
    If the tip is known to be far from the wall, an invalid angle does not
    trigger the proximity-gated stop.
    """
    metrics = dict(centreline_metrics or {})

    try:
        angle_deg = float(
            metrics.get(
                "tip_vessel_angle_deg",
                np.nan,
            )
        )
    except Exception:
        angle_deg = np.nan

    try:
        lumen_index = int(
            metrics.get(
                "centreline_index",
                -1,
            )
        )
    except Exception:
        lumen_index = -1

    angle_source = "compute_centreline_metrics"

    if not np.isfinite(angle_deg):
        angle_deg, lumen_index_fallback = (
            _fallback_tip_vessel_angle_deg(
                lumen_C=lumen_C,
                tip_pos=tip_pos,
                tip_tan=tip_tan,
            )
        )
        angle_source = "runner_fallback"
        if lumen_index < 0:
            lumen_index = int(
                lumen_index_fallback
            )

    (
        clearance_m,
        clearance_index,
        clearance_source,
    ) = _tip_wall_clearance_m(
        centreline_metrics=centreline_metrics,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        tip_pos=tip_pos,
        tip_radius_m=tip_radius_m,
    )
    if lumen_index < 0:
        lumen_index = int(clearance_index)

    angle_valid = bool(
        np.isfinite(angle_deg)
    )
    clearance_valid = bool(
        np.isfinite(clearance_m)
    )
    near_wall = bool(
        clearance_valid
        and clearance_m
        <= float(activation_clearance_m)
    )

    violated = bool(
        angle_valid
        and near_wall
        and angle_deg > float(angle_limit_deg)
    )

    invalid_stop = bool(
        invalid_is_stop
        and (
            not clearance_valid
            or (
                near_wall
                and not angle_valid
            )
        )
    )
    should_stop = bool(
        violated or invalid_stop
    )

    if violated:
        reason = (
            "near_wall_tip_tangent_angle_exceeded"
        )
    elif invalid_stop and not clearance_valid:
        reason = "tip_wall_clearance_invalid"
    elif invalid_stop:
        reason = (
            "near_wall_tip_tangent_angle_invalid"
        )
    else:
        reason = ""

    margin_deg = (
        float(angle_limit_deg - angle_deg)
        if angle_valid
        else np.nan
    )
    clearance_margin_m = (
        float(
            clearance_m
            - activation_clearance_m
        )
        if clearance_valid
        else np.nan
    )

    return {
        "should_stop": should_stop,
        "violated": violated,
        "valid": bool(
            angle_valid and clearance_valid
        ),
        "angle_valid": angle_valid,
        "clearance_valid": clearance_valid,
        "near_wall": near_wall,
        "reason": reason,
        "angle_deg": float(angle_deg),
        "limit_deg": float(angle_limit_deg),
        "margin_deg": float(margin_deg),
        "clearance_m": float(clearance_m),
        "clearance_mm": float(
            1e3 * clearance_m
        ),
        "activation_clearance_m": float(
            activation_clearance_m
        ),
        "activation_clearance_mm": float(
            1e3 * activation_clearance_m
        ),
        "clearance_margin_m": float(
            clearance_margin_m
        ),
        "lumen_index": int(lumen_index),
        "angle_source": angle_source,
        "clearance_source": clearance_source,
    }


def _write_tip_tangent_stop_report(
    *,
    log_csv_path,
    k: int,
    phase: str,
    safety: dict,
    tip_pos,
    tip_tan,
    p_state,
) -> Path:
    """Write a human-readable record of a runner-level safety stop."""
    output_dir = Path(log_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tip_tangent_safety_stop.txt"

    tip = np.asarray(tip_pos, float).reshape(-1)
    tangent = np.asarray(tip_tan, float).reshape(-1)
    actuator_state = np.asarray(p_state, float).reshape(-1)

    lines = [
        "Runner tip-tangent safety stop",
        f"feedback_frame={int(k)}",
        f"phase={phase}",
        f"reason={safety.get('reason', '')}",
        f"angle_deg={float(safety.get('angle_deg', np.nan)):.12g}",
        f"limit_deg={float(safety.get('limit_deg', np.nan)):.12g}",
        f"margin_deg={float(safety.get('margin_deg', np.nan)):.12g}",
        f"angle_valid={bool(safety.get('valid', False))}",
        f"angle_source={safety.get('angle_source', '')}",
        f"clearance_m={float(safety.get('clearance_m', np.nan)):.12g}",
        f"clearance_mm={float(safety.get('clearance_mm', np.nan)):.12g}",
        f"activation_clearance_m={float(safety.get('activation_clearance_m', np.nan)):.12g}",
        f"near_wall={bool(safety.get('near_wall', False))}",
        f"clearance_valid={bool(safety.get('clearance_valid', False))}",
        f"clearance_source={safety.get('clearance_source', '')}",
        f"lumen_index={int(safety.get('lumen_index', -1))}",
        "tip_xyz=" + np.array2string(tip[:3], precision=12, separator=","),
        "tip_tangent=" + np.array2string(
            tangent[:3], precision=12, separator=","),
        "actuator_state_q_and_insertion=" + np.array2string(
            actuator_state, precision=12, separator=","),
    ]

    report_path.write_text("\n".join(lines) + "\n")
    return report_path

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

def _snapshot_mpc_state_for_policy_retry(mpc):
    """
    Snapshot internal MPC state before a candidate solve.

    Needed because mpc.step(...) advances the simulated plant/controller state.
    If tolerance rejects the candidate plan, we restore and re-solve with a
    shorter rollout.
    """
    snap = {
        "p": np.asarray(mpc.p, float).copy(),
        "x": np.asarray(mpc.x, float).copy(),
    }

    # Optional warm-start/reference attributes if present.
    for name in (
        "U_guess",
        "u_guess",
        "X_guess",
        "i_ref_last",
    ):
        if hasattr(mpc, name):
            value = getattr(mpc, name)
            try:
                if isinstance(value, (int, float, str, bool)):
                    snap[name] = value
                elif value is None:
                    snap[name] = None
                else:
                    snap[name] = np.asarray(value).copy()
            except Exception:
                pass

    return snap


def _restore_mpc_state_for_policy_retry(mpc, snap):
    mpc.p = np.asarray(snap["p"], float).copy()
    mpc.x = np.asarray(snap["x"], float).copy()

    for name, value in snap.items():
        if name in {"p", "x"}:
            continue
        if hasattr(mpc, name):
            try:
                if isinstance(value, np.ndarray):
                    setattr(mpc, name, value.copy())
                else:
                    setattr(mpc, name, value)
            except Exception:
                pass


def _attach_sqp_update_diagnostics(info):
    """
    Normalise SQP diagnostic keys expected by the evidence-based policy.
    """
    sqp_hist = info.get("sqp_hist", []) or []

    if len(sqp_hist) > 0:
        first = sqp_hist[0]
        last = sqp_hist[-1]

        info["sqp_du_0"] = float(first.get("step_norm", np.nan))
        info["sqp_du_rel_0"] = float(first.get("rel_step_norm", np.nan))

        info["sqp_du_final"] = float(last.get("step_norm", np.nan))
        info["sqp_du_rel_final"] = float(last.get("rel_step_norm", np.nan))

        info["sqp_iters_done"] = int(len(sqp_hist))
    else:
        info["sqp_du_0"] = np.nan
        info["sqp_du_rel_0"] = np.nan
        info["sqp_du_final"] = np.nan
        info["sqp_du_rel_final"] = np.nan
        info["sqp_iters_done"] = 0

    return info


def _attach_beam_hessian_risk(info, cfg):
    """
    Attach risk_low_lambda_min_H_beam if possible.

    The policy can use either:
        risk_low_lambda_min_H_beam
    or:
        beam_eig_lambda_min / lambda_min_H_beam

    This helper computes the risk if the minimum beam eigenvalue is available.
    """
    # If already present, leave it.
    try:
        existing = float(info.get("risk_low_lambda_min_H_beam", np.nan))
        if np.isfinite(existing):
            return info
    except Exception:
        pass

    # If mpc.step already returned a beam Hessian matrix, compute eigen diagnostics.
    H_beam = None
    for key in ("H_beam", "beam_H", "H_beam_raw", "H_beam_mpc"):
        if key in info:
            H_beam = info.get(key)
            break

    if H_beam is not None:
        try:
            eig_diag = beam_eigen_direction_diagnostics(
                H_beam,
                cond_warn=0.0,
                prefix="beam",
            )
            info.update(eig_diag)
        except Exception:
            pass

    # Try all known lambda-min key variants.
    lam = np.nan
    for key in (
        "lambda_min_H_beam",
        "beam_lambda_min_H_beam",
        "beam_eig_lambda_min",
        "H_beam_lambda_min",
        "eig_lambda_min_H_beam",
    ):
        try:
            value = float(info.get(key, np.nan))
            if np.isfinite(value):
                lam = value
                break
        except Exception:
            pass

    if np.isfinite(lam):
        risk = -np.log10(max(abs(lam), float(cfg.eigenvalue_floor)))
        info["lambda_min_H_beam"] = float(lam)
        info["risk_low_lambda_min_H_beam"] = float(risk)

    return info

def _solve_mpc_once_with_policy_metadata(
    *,
    mpc,
    x_meas,
    rollout_steps_current,
    solver_mode_current,
    N_sqp_current,
    decision,
    cfg,
):
    """
    Solve one controller update.

    Prediction horizon:
        - changed by the hierarchical policy only when a decision
          explicitly exists;
        - otherwise retains the controller's configured Np.

    Execution horizon:
        - determined independently by rollout_steps_current.
    """

    if decision is not None:
        # Hierarchical policy may explicitly change Np.
        Np_current = int(
            getattr(
                decision,
                "Np",
                mpc.Np,
            )
        )

        Np_current = max(
            1,
            Np_current,
        )

        if int(mpc.Np) != Np_current:
            mpc.set_prediction_horizon(
                Np_current,
                reset_warm=False,
            )

    else:
        # No hierarchy decision:
        # preserve the configured prediction horizon.
        Np_current = int(mpc.Np)

    if rollout_steps_current > Np_current:
        raise ValueError(
            "Execution horizon cannot exceed prediction "
            f"horizon: Nr={rollout_steps_current}, "
            f"Np={Np_current}."
        )

    old_N_sqp = int(
        getattr(
            mpc,
            "N_sqp",
            N_sqp_current,
        )
    )

    mpc.N_sqp = int(N_sqp_current)

    try:
        p_post, x_post, info = mpc.step(
            x_meas=x_meas,
            rollout_steps=int(
                rollout_steps_current
            ),
            solver_mode=str(
                solver_mode_current
            ),
        )
    finally:
        mpc.N_sqp = old_N_sqp

    info["solver_mode_used"] = str(
        solver_mode_current
    )
    info["N_sqp_used"] = int(
        N_sqp_current
    )
    info["rollout_steps_used"] = int(
        rollout_steps_current
    )
    info["Np_used"] = int(
        Np_current
    )

    # Keep the rest of your existing policy-metadata code.
    info = _attach_sqp_update_diagnostics(info)
    info = _attach_beam_hessian_risk(
        info,
        cfg,
    )

    if decision is not None:
        evaluation = (
            evaluate_selected_policy_tolerance(
                info=info,
                decision=decision,
                cfg=cfg,
            )
        )

        info = attach_policy_metadata(
            info,
            decision=decision,
            evaluation=evaluation,
        )

        info["Np_used"] = int(
            Np_current
        )
        info["hierarchy_Np"] = int(
            Np_current
        )

    else:
        evaluation = None

    return (
        p_post,
        x_post,
        info,
        evaluation,
    )
def evaluate_jacobian_sequence_along_solution(
    *,
    controller,
    p0,
    U_seq,
    use_ltv: bool,
):
    U_seq = np.asarray(
        U_seq,
        float,
    ).reshape(controller.Np, controller.m)

    p_nodes = controller._p_nodes_from_U(
        p0,
        U_seq,
    )

    def discrete_B(state):
        # Use the same conversion as the optimizer.  The joint-space adapter
        # returns continuous J = dy/d[qd,dL], while the prediction model uses
        # B_k = dt*J_k.  Calling Jxy_fn directly here used to omit dt and made
        # the sequence diagnostics inconsistent with the QP.
        stage_builder = getattr(controller, "_stage_input_matrix", None)
        if callable(stage_builder):
            return np.asarray(
                stage_builder(state), float
            ).reshape(controller.n, controller.m)

        matrix = np.asarray(
            controller.Jxy_fn(state), float
        ).reshape(controller.n, controller.m)
        if bool(getattr(controller, "jacobian_returns_continuous", False)):
            matrix = float(controller.dt) * matrix
        return matrix

    B0 = discrete_B(p_nodes[0])

    if use_ltv:
        B_sequence = np.stack(
            [
                discrete_B(p_nodes[stage])
                for stage in range(
                    controller.Np
                )
            ],
            axis=0,
        )
    else:
        B_sequence = np.repeat(
            B0.reshape(
                1,
                controller.n,
                controller.m,
            ),
            controller.Np,
            axis=0,
        )

    return (
        B_sequence,
        np.asarray(p_nodes[:-1], float),
    )
def jacobian_channel_metrics(
    *,
    B_sequence,
    U_sequence,
    Q=None,
    input_scale=None,
):
    B_sequence = np.asarray(
        B_sequence,
        float,
    )

    U_sequence = np.asarray(
        U_sequence,
        float,
    )

    Np, n, m = B_sequence.shape
    U_sequence = U_sequence.reshape(Np, m)

    if input_scale is None:
        input_scale = np.ones(m, float)

    input_scale = np.asarray(
        input_scale,
        float,
    ).reshape(m)

    if Q is None:
        Q = np.eye(n)

    Q = np.asarray(Q, float).reshape(n, n)

    # Raw sensitivity of each channel.
    column_norm_xyz = np.linalg.norm(
        B_sequence,
        axis=1,
    )

    column_norm_xy = np.linalg.norm(
        B_sequence[:, :2, :],
        axis=1,
    )

    # Sensitivity scaled by a meaningful command size.
    B_scaled = (
        B_sequence
        * input_scale.reshape(1, 1, m)
    )

    scaled_authority = np.linalg.norm(
        B_scaled,
        axis=1,
    )

    weighted_scaled_authority = np.sqrt(
        np.maximum(
            np.einsum(
                "kim,ij,kjm->km",
                B_scaled,
                Q,
                B_scaled,
            ),
            0.0,
        )
    )

    # Contribution actually used by controller.
    contribution_vectors = (
        B_sequence
        * U_sequence[:, None, :]
    )

    contribution_norm = np.linalg.norm(
        contribution_vectors,
        axis=1,
    )

    predicted_increment = np.einsum(
        "knm,km->kn",
        B_sequence,
        U_sequence,
    )

    # Change from first predicted-stage Jacobian.
    B0 = B_sequence[0]

    matrix_change_relative = (
        np.linalg.norm(
            B_sequence-B0[None, :, :],
            axis=(1, 2),
        )
        /
        max(np.linalg.norm(B0), 1e-12)
    )

    norm0 = np.linalg.norm(
        B0,
        axis=0,
    )

    column_gain_ratio = (
        column_norm_xyz
        /
        np.maximum(norm0[None, :], 1e-12)
    )

    dot = np.sum(
        B_sequence*B0[None, :, :],
        axis=1,
    )

    denominator = (
        column_norm_xyz
        * np.maximum(
            norm0[None, :],
            1e-12,
        )
    )

    cosine = np.clip(
        dot/np.maximum(denominator, 1e-12),
        -1.0,
        1.0,
    )

    column_direction_change_deg = (
        np.rad2deg(np.arccos(cosine))
    )

    return {
        "column_norm_xyz": column_norm_xyz,
        "column_norm_xy": column_norm_xy,
        "scaled_authority": scaled_authority,
        "weighted_scaled_authority": (
            weighted_scaled_authority
        ),
        "contribution_vectors": (
            contribution_vectors
        ),
        "contribution_norm": contribution_norm,
        "predicted_increment": (
            predicted_increment
        ),
        "matrix_change_relative": (
            matrix_change_relative
        ),
        "column_gain_ratio": (
            column_gain_ratio
        ),
        "column_direction_change_deg": (
            column_direction_change_deg
        ),
    }
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

    # New options.
    diagnostic_plots_enabled: bool = True,
    diagnostic_plot_stride: int = 1,
    diagnostic_plot_dpi: int = 160,

    # Runner-level current-state safety guard.
    enable_tip_tangent_stop: bool = True,
    tip_tangent_stop_deg: float = 40.0,
    tip_tangent_stop_clearance_m: float | None = None,
    tip_radius_m: float | None = None,
    tip_tangent_invalid_is_stop: bool = True,

    adaptive_rollout_enabled: bool = False,
    hierarchical_mpc_enabled: bool = False,
    hierarchical_policy_config=None,
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

    tip_tangent_stop_deg = float(tip_tangent_stop_deg)
    if not (0.0 < tip_tangent_stop_deg < 180.0):
        raise ValueError(
            "tip_tangent_stop_deg must be strictly between 0 and 180 degrees."
        )

    enable_tip_tangent_stop = bool(enable_tip_tangent_stop)

    if tip_tangent_stop_clearance_m is None:
        tip_tangent_stop_clearance_m = float(
            getattr(
                mpc,
                "tip_tangent_activation_clearance_m",
                1.0e-3,
            )
        )
    else:
        tip_tangent_stop_clearance_m = float(
            tip_tangent_stop_clearance_m
        )

    if tip_radius_m is None:
        tip_radius_m = float(
            getattr(
                mpc,
                "tip_radius_m",
                0.0,
            )
        )
    else:
        tip_radius_m = float(tip_radius_m)

    if tip_tangent_stop_clearance_m < 0.0:
        raise ValueError(
            "tip_tangent_stop_clearance_m must be non-negative."
        )
    if tip_radius_m < 0.0:
        raise ValueError(
            "tip_radius_m must be non-negative."
        )
    tip_tangent_invalid_is_stop = bool(tip_tangent_invalid_is_stop)
    runner_safety_stop = None

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
            enabled=bool(hierarchical_mpc_enabled),

            # The policy chooses 4, 2, or 1.
            rollout_min=1,
            rollout_max=min(4, rollout_steps_max),

            # Strict evidence-matched version:
            # low risk -> SQP rollout 4
            # medium risk -> SQP rollout 2
            # high risk -> SQP rollout 1
            use_lti_for_low_risk=False,

            # Conservative first step if beam-Hessian risk is not available yet.
            missing_beam_risk_bin="high",

            # Empirical beam-Hessian thresholds.
            beam_risk_low_max=8.8665,
            beam_risk_medium_max=8.9859,

            low_risk_rollout=4,
            medium_risk_rollout=2,
            high_risk_rollout=1,

            # SQP caps.
            sqp_cap_low=4,
            sqp_cap_medium=4,
            sqp_cap_high=3,

            # Tolerances.
            low_abs_tol=2.4,
            low_rel_guard=2.0,
            medium_abs_tol=2.4,
            medium_rel_guard=2.0,
            high_rel_tol=0.60,
            high_rel_strict_tol=0.30,
            high_abs_guard=7.5,

            # Probe/recovery thresholds.
            probe_abs_moderate=3.35,
            probe_abs_high=5.89,
            probe_rel_warn=1.0,
            probe_rel_high=2.0,

            # Safety guards.
            clearance_stop_mm=0.8,
            clearance_bad_mm=1.0,
            clearance_warn_mm=1.5,
            mpc_cond_extreme=1e8,

            downgrade_patience=2,
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
        "hierarchy_risk_bin": [],
        "hierarchy_beam_risk_value": [],
        "hierarchy_tolerance_accepted": [],
        "hierarchy_tolerance_reason": [],
        "hierarchy_rel_tol": [],
        "hierarchy_abs_tol": [],
        "Np_used": [],
        "hierarchy_Np": [],
    }
    path_for_curvature = np.asarray(lumen_path, float)

    if path_for_curvature.ndim != 2 or path_for_curvature.shape[1] < 3:
        path_for_curvature = np.asarray(lumen_C, float)

    path_curvature = compute_discrete_path_curvature(path_for_curvature)


    diagnostic_plot_root = (
        Path(log_csv_path).parent
        / "diagnostic_frames"
    )

    if diagnostic_plots_enabled:
        diagnostic_plot_root.mkdir(
            parents=True,
            exist_ok=True,
        )
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
        # Runner-level hard current-state tip-tangent safety check.
        # This runs before mpc.step(), so an already-unsafe state cannot
        # generate or apply another command.
        # ------------------------------------------------------------
        pre_tip_tangent_safety = _tip_tangent_safety_status(
            centreline_metrics=centreline_metrics_pre,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip_pre_vis,
            tip_tan=tan_pre,
            angle_limit_deg=tip_tangent_stop_deg,
            activation_clearance_m=(
                tip_tangent_stop_clearance_m
            ),
            tip_radius_m=tip_radius_m,
            invalid_is_stop=tip_tangent_invalid_is_stop,
        )

        if (
            enable_tip_tangent_stop
            and pre_tip_tangent_safety["should_stop"]
        ):
            runner_safety_stop = {
                **pre_tip_tangent_safety,
                "k": int(k),
                "phase": "pre_step",
            }
            report_path = _write_tip_tangent_stop_report(
                log_csv_path=log_csv_path,
                k=k,
                phase="pre_step",
                safety=pre_tip_tangent_safety,
                tip_pos=tip_pre_vis,
                tip_tan=tan_pre,
                p_state=p_pre,
            )
            runner_safety_stop["report_path"] = str(report_path)
            setattr(mpc, "last_runner_safety_stop", dict(runner_safety_stop))

            angle_text = (
                f"{pre_tip_tangent_safety['angle_deg']:.3f} deg"
                if pre_tip_tangent_safety["valid"]
                else "invalid/non-finite"
            )
            print(
                "[SAFETY STOP] Current tip is within "
                f"{pre_tip_tangent_safety['clearance_mm']:.3f} mm "
                "signed wall clearance and the beam-tip/lumen-tangent "
                f"angle is {angle_text}; hard angle limit is "
                f"{tip_tangent_stop_deg:.3f} deg when clearance is <= "
                f"{1e3 * tip_tangent_stop_clearance_m:.3f} mm. "
                "No new controller command was applied. "
                f"Report: {report_path}"
            )
            break

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
        # ------------------------------------------------------------
        # Solve/apply MPC with evidence-based tolerance acceptance.
        # If the selected rollout is rejected by tolerance, restore and
        # re-solve once with a shorter rollout.
        # ------------------------------------------------------------
        mpc_snapshot = _snapshot_mpc_state_for_policy_retry(mpc)

        p_post, x_post, info, policy_eval = _solve_mpc_once_with_policy_metadata(
            mpc=mpc,
            x_meas=None,
            rollout_steps_current=rollout_steps_current,
            solver_mode_current=solver_mode_current,
            N_sqp_current=N_sqp_current,
            decision=decision,
            cfg=hierarchical_policy_config,
        )

        # If the policy rejects a rollout > 1, restore the pre-solve state and
        # immediately retry with a shorter SQP-full rollout.
        if (
            hierarchical_mpc_enabled
            and decision is not None
            and policy_eval is not None
            and not policy_eval.accepted
            and policy_eval.should_reduce_rollout
        ):
            reduced_rollout = max(
                int(hierarchical_policy_config.rollout_min),
                min(int(decision.rollout_steps) - 1, int(decision.rollout_steps) // 2),
            )

            reduced_rollout = int(np.clip(
                reduced_rollout,
                hierarchical_policy_config.rollout_min,
                hierarchical_policy_config.rollout_max,
            ))

            reduced_rel_tol, reduced_abs_tol, reduced_strict_rel_tol, reduced_tol_mode = (
                tolerance_for_zone(
                    decision.risk_bin,
                    reduced_rollout,
                    hierarchical_policy_config,
                )
            )

            reduced_N_sqp = sqp_cap_for_zone(
                decision.risk_bin,
                reduced_rollout,
                hierarchical_policy_config,
            )

            reduced_decision = replace(
                decision,
                solver_mode="sqp_full",
                rollout_steps=int(reduced_rollout),
                Np=int(reduced_rollout),
                N_sqp=int(reduced_N_sqp),
                rel_tol=float(reduced_rel_tol),
                abs_tol=float(reduced_abs_tol),
                strict_rel_tol=float(reduced_strict_rel_tol),
                tolerance_mode=reduced_tol_mode,
                reason=(
                    str(decision.reason)
                    + "|retry_after_tolerance_reject"
                    + f"|reduced_rollout={reduced_rollout}"
                    + f"|reduced_Np={reduced_rollout}"
                ),
            )

            _restore_mpc_state_for_policy_retry(mpc, mpc_snapshot)

            solver_mode_current = "sqp_full"
            rollout_steps_current = int(reduced_rollout)
            N_sqp_current = int(reduced_N_sqp)
            decision = reduced_decision

            p_post, x_post, info, policy_eval = _solve_mpc_once_with_policy_metadata(
                mpc=mpc,
                x_meas=None,
                rollout_steps_current=rollout_steps_current,
                solver_mode_current=solver_mode_current,
                N_sqp_current=N_sqp_current,
                decision=decision,
                cfg=hierarchical_policy_config,
            )

        U_seq = np.asarray(
            info.get("U_seq", []),
            float,
        )

        if U_seq.size == mpc.Np * mpc.m:
            U_seq = U_seq.reshape(
                mpc.Np,
                mpc.m,
            )

            controller_type = str(
                info.get("controller_type", "mpc")
            )

            if controller_type == "inverse_jacobian":
                use_ltv_jacobian = (
                    str(
                        info.get(
                            "inverse_sequence_mode",
                            "",
                        )
                    )
                    == "rollout_ltv"
                )
            else:
                use_ltv_jacobian = (
                    str(solver_mode_current) != "lti"
                )

            B_solution, B_solution_nodes = (
                evaluate_jacobian_sequence_along_solution(
                    controller=mpc,
                    p0=p_pre,
                    U_seq=U_seq,
                    use_ltv=use_ltv_jacobian,
                )
            )

            info["B_sequence_solution"] = (
                B_solution.copy()
            )
            info["B_state_nodes_solution"] = (
                B_solution_nodes.copy()
            )
            # Temporary compatibility alias for existing log readers.
            info["B_pose_nodes_solution"] = info[
                "B_state_nodes_solution"
            ].copy()
            info["controller_dt_s"] = float(mpc.dt)
            if B_solution.shape[2] >= 7:
                B0_insertion = B_solution[0, :, 6].copy()
                info["B0_insertion_column"] = B0_insertion
                info["J0_insertion_column"] = (
                    B0_insertion / float(mpc.dt)
                )

            B_linearisation = np.asarray(
                info.get(
                    "B_sequence_linearisation",
                    [],
                ),
                float,
            )

            if B_linearisation.shape == B_solution.shape:
                denominator = np.maximum(
                    np.linalg.norm(
                        B_linearisation,
                        axis=(1, 2),
                    ),
                    1e-12,
                )

                info["B_drift_relative_stage"] = (
                    np.linalg.norm(
                        B_solution-B_linearisation,
                        axis=(1, 2),
                    )
                    / denominator
                )
        channel_names = np.asarray([
            "qd1", "qd2", "qd3",
            "qd4", "qd5", "qd6",
            "dL",
        ])

        # Use a fixed scale across controllers and experiments.
        # Do not use the current adaptive trust radius here because that
        # would mix changes in B with changes in the trust-region radius.
        input_scale = np.asarray(
            getattr(
                mpc,
                "jacobian_diag_input_scale",
                np.ones(mpc.m),
            ),
            float,
        ).reshape(mpc.m)

        info["jacobian_channel_names"] = channel_names.copy()
        info["jacobian_diag_input_scale"] = input_scale.copy()

        solution_metrics = jacobian_channel_metrics(
            B_sequence=B_solution,
            U_sequence=U_seq,
            Q=mpc.Q,
            input_scale=input_scale,
        )

        for name, value in solution_metrics.items():
            info[f"jac_solution_{name}"] = (
                np.asarray(value).copy()
            )

        if B_linearisation.shape == B_solution.shape:
            linearisation_metrics = jacobian_channel_metrics(
                B_sequence=B_linearisation,
                U_sequence=U_seq,
                Q=mpc.Q,
                input_scale=input_scale,
            )

            for name, value in linearisation_metrics.items():
                info[f"jac_linearisation_{name}"] = (
                    np.asarray(value).copy()
                )
        info["jac_solution_column_norm_xyz"]
        info["jac_solution_column_norm_xy"]
        info["jac_solution_scaled_authority"]
        info["jac_solution_weighted_scaled_authority"]
        info["jac_solution_contribution_norm"]
        info["jac_solution_predicted_increment"]
        info["jac_solution_matrix_change_relative"]
        info["jac_solution_column_gain_ratio"]
        info["jac_solution_column_direction_change_deg"]
        p_post = np.asarray(p_post, float)
        x_post = np.asarray(x_post, float)
        tip_post = x_post[:3].copy()

        # ------------------------------------------------------------
        # Attach active-mode metadata
        # ------------------------------------------------------------
        info["solver_mode_used"] = str(solver_mode_current)
        info["N_sqp_used"] = int(N_sqp_current)
        info["rollout_steps_used"] = int(rollout_steps_current)
        info["p_start"] = p_pre.copy()
        info["x_start"] = x_pre.copy()
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

        if idx_ref.size > 0:
            info["X_ref_plan"] = np.asarray(
                lumen_C[idx_ref, :mpc.n],
                float,
            ).copy()

            # Physical arc-length coordinates.
            lumen_xyz = np.asarray(
                lumen_C[:, :3],
                float,
            )

            lumen_ds = np.linalg.norm(
                np.diff(lumen_xyz, axis=0),
                axis=1,
            )

            lumen_s = np.concatenate(
                ([0.0], np.cumsum(lumen_ds))
            )

            info["reference_s_plan_m"] = (
                lumen_s[idx_ref].copy()
            )

            info["reference_horizon_span_m"] = float(
                lumen_s[idx_ref[-1]]
                -
                lumen_s[idx_ref[0]]
            )
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
        # Post-step current-state safety check. The unsafe final frame is
        # logged below, then the simulation stops before the next command.
        # ------------------------------------------------------------
        post_tip_tangent_safety = _tip_tangent_safety_status(
            centreline_metrics=centreline_metrics,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip_post_vis,
            tip_tan=tan_post,
            angle_limit_deg=tip_tangent_stop_deg,
            activation_clearance_m=(
                tip_tangent_stop_clearance_m
            ),
            tip_radius_m=tip_radius_m,
            invalid_is_stop=tip_tangent_invalid_is_stop,
        )
        post_tip_tangent_stop = bool(
            enable_tip_tangent_stop
            and post_tip_tangent_safety["should_stop"]
        )

        info["runner_tip_tangent_safety_enabled"] = bool(
            enable_tip_tangent_stop
        )
        info["runner_tip_tangent_angle_deg"] = float(
            post_tip_tangent_safety["angle_deg"]
        )
        info["runner_tip_tangent_limit_deg"] = float(
            tip_tangent_stop_deg
        )
        info["runner_tip_tangent_margin_deg"] = float(
            post_tip_tangent_safety["margin_deg"]
        )
        info["runner_tip_tangent_angle_valid"] = bool(
            post_tip_tangent_safety["valid"]
        )
        info["runner_tip_tangent_lumen_index"] = int(
            post_tip_tangent_safety["lumen_index"]
        )
        info["runner_tip_tangent_angle_source"] = str(
            post_tip_tangent_safety["angle_source"]
        )
        info["runner_tip_wall_clearance_m"] = float(
            post_tip_tangent_safety["clearance_m"]
        )
        info["runner_tip_wall_clearance_mm"] = float(
            post_tip_tangent_safety["clearance_mm"]
        )
        info["runner_tip_tangent_activation_clearance_m"] = float(
            tip_tangent_stop_clearance_m
        )
        info["runner_tip_tangent_near_wall"] = bool(
            post_tip_tangent_safety["near_wall"]
        )
        info["runner_tip_wall_clearance_valid"] = bool(
            post_tip_tangent_safety["clearance_valid"]
        )
        info["runner_tip_wall_clearance_source"] = str(
            post_tip_tangent_safety["clearance_source"]
        )
        info["runner_safety_stop"] = bool(post_tip_tangent_stop)
        info["runner_safety_stop_reason"] = (
            str(post_tip_tangent_safety["reason"])
            if post_tip_tangent_stop
            else ""
        )
        info["runner_safety_stop_phase"] = (
            "post_step" if post_tip_tangent_stop else ""
        )

        if post_tip_tangent_stop:
            runner_safety_stop = {
                **post_tip_tangent_safety,
                "k": int(k),
                "phase": "post_step",
            }
            report_path = _write_tip_tangent_stop_report(
                log_csv_path=log_csv_path,
                k=k,
                phase="post_step",
                safety=post_tip_tangent_safety,
                tip_pos=tip_post_vis,
                tip_tan=tan_post,
                p_state=p_post,
            )
            runner_safety_stop["report_path"] = str(report_path)
            info["runner_safety_stop_report"] = str(report_path)
            setattr(mpc, "last_runner_safety_stop", dict(runner_safety_stop))

        info["X_nl_plan"] = info.get(
            "X_nl_candidate",
            None,
        )
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
        hist["hierarchy_risk_bin"].append(info.get("hierarchy_risk_bin", ""))
        hist["hierarchy_beam_risk_value"].append(
            float(info.get("hierarchy_beam_risk_value", np.nan))
        )
        hist["hierarchy_tolerance_accepted"].append(
            bool(info.get("hierarchy_tolerance_accepted", False))
        )
        hist["hierarchy_tolerance_reason"].append(
            str(info.get("hierarchy_tolerance_reason", ""))
        )
        hist["hierarchy_rel_tol"].append(
            float(info.get("hierarchy_rel_tol", np.nan))
        )
        hist["hierarchy_abs_tol"].append(
            float(info.get("hierarchy_abs_tol", np.nan))
        )
        hist["Np_used"].append(int(info.get("Np_used", getattr(mpc, "Np", np.nan))))
        hist["hierarchy_Np"].append(int(info.get("hierarchy_Np", info.get("Np_used", -1))))
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
            f"risk_bin={info.get('hierarchy_risk_bin', '')} "
            f"beamRisk={float(info.get('hierarchy_beam_risk_value', np.nan)):.3f} "
            f"tolOK={info.get('hierarchy_tolerance_accepted', '')} "
            f"tolReason={info.get('hierarchy_tolerance_reason', '')} "
            f"hierReason={info.get('hierarchy_reason', '')} "
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
            f"risk_bin={info.get('hierarchy_risk_bin', '')} "
            f"beamRisk={float(info.get('hierarchy_beam_risk_value', np.nan)):.3f} "
            f"tolOK={info.get('hierarchy_tolerance_accepted', '')} "
            f"tol={info.get('hierarchy_tolerance_mode', '')} "
            f"absTol={float(info.get('hierarchy_abs_tol', np.nan)):.3g} "
            f"relTol={float(info.get('hierarchy_rel_tol', np.nan)):.3g} "
            f"Np={info.get('Np_used', '')} "
            f"risk_bin={info.get('hierarchy_risk_bin', '')} "
            f"beamRisk={float(info.get('hierarchy_beam_risk_value', np.nan)):.3f} "
            f"tolOK={info.get('hierarchy_tolerance_accepted', '')} "
            f"tol={info.get('hierarchy_tolerance_mode', '')} "
        )
        u0_arr = np.asarray(info.get("u0", np.full(mpc.m, np.nan)), float).reshape(-1)

        if u0_arr.size >= 7:
            info["u0_qd1"] = float(u0_arr[0])
            info["u0_qd2"] = float(u0_arr[1])
            info["u0_qd3"] = float(u0_arr[2])
            info["u0_qd4"] = float(u0_arr[3])
            info["u0_qd5"] = float(u0_arr[4])
            info["u0_qd6"] = float(u0_arr[5])
            info["u0_dL"] = float(u0_arr[6])
        # ------------------------------------------------------------
        # Plotting and CSV logging
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # Controller diagnostic figures
        # ------------------------------------------------------------
        if (
            diagnostic_plots_enabled
            and k % diagnostic_plot_stride == 0
        ):
            saved_diagnostic_paths = (
                save_step_diagnostic_plots(
                    k=k,
                    info=info,
                    output_root=diagnostic_plot_root,
                    channel_names=(
                    "qd1", "qd2", "qd3",
                    "qd4", "qd5", "qd6",
                    "dL",
                    ),
                    dpi=diagnostic_plot_dpi,
                )
            )

            info["num_diagnostic_plots_saved"] = int(
                len(saved_diagnostic_paths)
            )



        # Always attach the physical magnet position to logs, even when plots
        # are disabled.
        p_mag_post = mpc._eval_magnet_position(p_post)
        info["magnet_position_R_m"] = np.asarray(
            p_mag_post, float
        ).reshape(3).copy()
        info["insertion_now_m"] = float(np.asarray(p_post, float)[6])

        if save_plots:
            if 'limits' not in locals():
                limits = None
            geometry_paths, limits = save_geometry_frames(
                k=k,
                frames_dir=frames_dir,
                C_beam=C_post if C_post is not None else C_pre,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                p0=p0_ur,
                tip=tip_post_vis,
                p_mag=p_mag_post,
                limits=limits,  
            )

            info["frame_with_magnet"] = str(
                geometry_paths["with_magnet"]
            )

            info["frame_without_magnet"] = str(
                geometry_paths["without_magnet"]
            )

        save_step_artifacts(
            k=k,
            frames_dir=frames_dir,
            log_csv_path=log_csv_path,
            u0=info.get(
                "u0",
                np.full(mpc.m, np.nan),
            ),
            p_now=p_post,
            y_now=y_post,
            i_ref=i_ref,
            info=info,
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip_post_vis,
            tip_tan=tan_post,

            # Set False here if the two figures above replace the
            # original single-frame figure.
            save_plots=False,
        )


        # ------------------------------------------------------------
        # Update previous info for next preventive hierarchy decision
        # ------------------------------------------------------------
        previous_info = dict(info)

        # ------------------------------------------------------------
        # Stop conditions
        # ------------------------------------------------------------
        if post_tip_tangent_stop:
            angle_text = (
                f"{post_tip_tangent_safety['angle_deg']:.3f} deg"
                if post_tip_tangent_safety["valid"]
                else "invalid/non-finite"
            )
            print(
                "[SAFETY STOP] Applied rollout ended within "
                f"{post_tip_tangent_safety['clearance_mm']:.3f} mm "
                "signed wall clearance and with a beam-tip/lumen-tangent "
                f"angle of {angle_text}; hard angle limit is "
                f"{tip_tangent_stop_deg:.3f} deg when clearance is <= "
                f"{1e3 * tip_tangent_stop_clearance_m:.3f} mm. "
                "The final frame was saved and no further command will be applied."
            )
            break

        if i_ref >= np.asarray(lumen_C).shape[0] - 2:
            print("[DONE] reached final reference region")
            break

        if int(info.get("infeasible", 0)):
            print("[STOP] MPC infeasible")
            break

    if not hasattr(mpc, "last_runner_safety_stop"):
        setattr(mpc, "last_runner_safety_stop", runner_safety_stop)

    return hist
