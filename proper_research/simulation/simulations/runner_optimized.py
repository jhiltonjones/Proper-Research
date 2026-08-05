from __future__ import annotations

import csv
from pathlib import Path
import time
from typing import Any

import numpy as np

from proper_research.simulation.simulations.logging_utils import (
    compute_centreline_metrics,
    save_step_artifacts,
)


def snapshot_forward(forward6d, p8, *, commit: bool = False):
    """Return tip, tangent, centreline and six-dimensional output."""
    y = np.asarray(forward6d(p8, commit=commit), dtype=float).reshape(-1)
    if y.size < 6:
        raise ValueError(
            f"forward6d returned {y.size} values; expected at least 6."
        )
    y = y[:6]
    centreline = getattr(forward6d, "last_p_centerline", None)
    if centreline is not None:
        centreline = np.asarray(centreline, dtype=float).copy()
    return y[:3].copy(), y[3:6].copy(), centreline, y.copy()


_LIGHT_COLUMNS = (
    "k",
    "status",
    "infeasible",
    "solver_mode",
    "Np",
    "N_sqp",
    "rollout_steps",
    "i_ref",
    "tip_x",
    "tip_y",
    "tip_z",
    "clearance_mm",
    "tip_vessel_angle_deg",
    "u0_vx",
    "u0_vy",
    "u0_vz",
    "u0_wx",
    "u0_wy",
    "u0_wz",
    "u0_dL",
    "controller_solve_ms",
    "controller_rollout_ms",
    "controller_step_ms",
    "qp_build_ms",
    "osqp_ms",
    "candidate_validation_ms",
    "forward_calls",
    "jacobian_calls",
    "forward_time_ms",
    "jacobian_time_ms",
    "sqp_iterations",
    "osqp_iterations",
    "osqp_setup_performed",
    "hessian_reused",
)


def _append_light_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(_LIGHT_COLUMNS),
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_simulation_optimized(
    *,
    mpc,
    forward6d,
    p0_ur,
    p0,
    lumen_C,
    lumen_R,
    lumen_path=None,
    s_path=None,
    frames_dir: Path | None = None,
    log_csv_path: Path = Path("optimized_log.csv"),
    max_steps: int = 200,
    rollout_steps: int = 1,
    solver_mode: str = "sqp_full",
    save_plots: bool = False,
    full_logging: bool = False,
    verbose: bool = True,
    enable_tip_tangent_stop: bool = True,
    tip_tangent_stop_deg: float = 40.0,
    tip_tangent_stop_clearance_m: float = 1.0e-3,
    tip_radius_m: float = 0.0,
    stop_reference_margin: int = 5,
):
    """
    Fixed-mode simulation runner with no hierarchical controller.

    `solver_mode` is held fixed for the complete run, preserving direct
    LTI/LTV/SQP comparisons. `rollout_steps` is also fixed.
    """
    del lumen_path, s_path  # retained only for call-site compatibility

    max_steps = int(max_steps)
    rollout_steps = int(rollout_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive.")
    if solver_mode not in {"lti", "ltv_oneshot", "sqp_full"}:
        raise ValueError(
            "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
        )

    log_csv_path = Path(log_csv_path)
    frames_dir = None if frames_dir is None else Path(frames_dir)
    lumen_C = np.asarray(lumen_C, dtype=float)
    lumen_R = np.asarray(lumen_R, dtype=float).reshape(-1)

    history: dict[str, list] = {
        "k": [],
        "status": [],
        "infeasible": [],
        "i_ref": [],
        "tip": [],
        "p": [],
        "u0": [],
        "controller_step_s": [],
        "forward_calls": [],
        "jacobian_calls": [],
        "sqp_iterations": [],
    }
    stop_reason = "max_steps"
    run_started = time.perf_counter()

    for k in range(max_steps):
        step_started = time.perf_counter()
        p_post, x_post, info = mpc.step(
            x_meas=None,
            rollout_steps=rollout_steps,
            solver_mode=solver_mode,
        )
        measured_step_s = time.perf_counter() - step_started

        tip, tangent, _, y_post = snapshot_forward(
            forward6d,
            p_post,
            commit=False,
        )
        metrics = compute_centreline_metrics(
            lumen_C=lumen_C,
            lumen_R=lumen_R,
            tip_pos=tip,
            tip_tan=tangent,
        )
        info.update(metrics)

        idx_ref = np.asarray(info.get("idx_ref", []), dtype=int).reshape(-1)
        i_ref = int(idx_ref[0]) if idx_ref.size else -1
        U0 = np.asarray(
            info.get("u0", np.full(mpc.m, np.nan)),
            dtype=float,
        ).reshape(-1)
        if U0.size < 7:
            U0 = np.pad(U0, (0, 7 - U0.size), constant_values=np.nan)

        sqp_hist = info.get("sqp_hist", []) or []
        osqp_iter = -1
        osqp_setup = 0
        if sqp_hist:
            osqp_iter = int(sqp_hist[-1].get("osqp_iter", -1))
            osqp_setup = int(
                sqp_hist[-1].get("osqp_setup_performed", 0)
            )

        row = {
            "k": int(k),
            "status": str(info.get("status", "unknown")),
            "infeasible": int(info.get("infeasible", 0)),
            "solver_mode": solver_mode,
            "Np": int(getattr(mpc, "Np", -1)),
            "N_sqp": int(getattr(mpc, "N_sqp", -1)),
            "rollout_steps": int(
                info.get("rollout_steps_used", rollout_steps)
            ),
            "i_ref": i_ref,
            "tip_x": float(tip[0]),
            "tip_y": float(tip[1]),
            "tip_z": float(tip[2]),
            "clearance_mm": float(metrics.get("clearance_mm", np.nan)),
            "tip_vessel_angle_deg": float(
                metrics.get("tip_vessel_angle_deg", np.nan)
            ),
            "u0_vx": float(U0[0]),
            "u0_vy": float(U0[1]),
            "u0_vz": float(U0[2]),
            "u0_wx": float(U0[3]),
            "u0_wy": float(U0[4]),
            "u0_wz": float(U0[5]),
            "u0_dL": float(U0[6]),
            "controller_solve_ms": 1.0e3 * float(
                info.get("controller_solve_wall_s", np.nan)
            ),
            "controller_rollout_ms": 1.0e3 * float(
                info.get("controller_rollout_wall_s", np.nan)
            ),
            "controller_step_ms": 1.0e3 * float(
                info.get("controller_step_wall_s", measured_step_s)
            ),
            "qp_build_ms": 1.0e3 * float(
                info.get("qp_build_time_total_s", np.nan)
            ),
            "osqp_ms": 1.0e3 * float(
                info.get("osqp_time_total_s", np.nan)
            ),
            "candidate_validation_ms": 1.0e3 * float(
                info.get("candidate_validation_time_total_s", 0.0)
            ),
            "forward_calls": int(info.get("forward_calls_step", 0)),
            "jacobian_calls": int(info.get("jacobian_calls_step", 0)),
            "forward_time_ms": 1.0e3 * float(
                info.get("forward_time_step_s", 0.0)
            ),
            "jacobian_time_ms": 1.0e3 * float(
                info.get("jacobian_time_step_s", 0.0)
            ),
            "sqp_iterations": len(sqp_hist),
            "osqp_iterations": osqp_iter,
            "osqp_setup_performed": osqp_setup,
            "hessian_reused": int(bool(info.get("hessian_reused", False))),
        }
        _append_light_row(log_csv_path, row)

        if full_logging:
            save_step_artifacts(
                k=k,
                frames_dir=frames_dir,
                log_csv_path=log_csv_path.with_name(
                    f"{log_csv_path.stem}_full.csv"
                ),
                u0=U0,
                p_now=p_post,
                y_now=y_post,
                i_ref=i_ref,
                info=info,
                lumen_C=lumen_C,
                lumen_R=lumen_R,
                tip_pos=tip,
                tip_tan=tangent,
                save_plots=bool(save_plots),
            )

        history["k"].append(int(k))
        history["status"].append(str(info.get("status", "unknown")))
        history["infeasible"].append(int(info.get("infeasible", 0)))
        history["i_ref"].append(i_ref)
        history["tip"].append(tip.copy())
        history["p"].append(np.asarray(p_post, dtype=float).copy())
        history["u0"].append(U0.copy())
        history["controller_step_s"].append(
            float(info.get("controller_step_wall_s", measured_step_s))
        )
        history["forward_calls"].append(int(info.get("forward_calls_step", 0)))
        history["jacobian_calls"].append(
            int(info.get("jacobian_calls_step", 0))
        )
        history["sqp_iterations"].append(len(sqp_hist))

        if verbose:
            print(
                f"[{solver_mode}] k={k:04d} "
                f"status={row['status']} "
                f"step={row['controller_step_ms']:.2f} ms "
                f"forward={row['forward_time_ms']:.2f} ms/"
                f"{row['forward_calls']} "
                f"jac={row['jacobian_time_ms']:.2f} ms/"
                f"{row['jacobian_calls']} "
                f"qp={row['osqp_ms']:.2f} ms "
                f"clear={row['clearance_mm']:.3f} mm"
            )

        unsafe_angle = (
            np.isfinite(row["clearance_mm"])
            and row["clearance_mm"]
            <= 1.0e3 * (tip_tangent_stop_clearance_m - tip_radius_m)
            and (
                not np.isfinite(row["tip_vessel_angle_deg"])
                or row["tip_vessel_angle_deg"] > tip_tangent_stop_deg
            )
        )
        if enable_tip_tangent_stop and unsafe_angle:
            stop_reason = "tip_tangent_safety"
            break
        if int(info.get("infeasible", 0)):
            stop_reason = "controller_infeasible"
            break
        if i_ref >= lumen_C.shape[0] - int(stop_reference_margin):
            stop_reason = "reference_complete"
            break

    history["stop_reason"] = stop_reason
    history["run_wall_s"] = time.perf_counter() - run_started
    history["steps_completed"] = len(history["k"])
    return history


# Compatibility alias for a new experiment script.
run_simulation = run_simulation_optimized
