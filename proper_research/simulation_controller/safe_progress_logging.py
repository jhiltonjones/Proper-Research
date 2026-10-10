"""Small logging helpers for the safe-progress MPC status fields."""
from __future__ import annotations

from typing import Any
import numpy as np


SAFE_PROGRESS_LOG_FIELDS = (
    "status",
    "controller_motion_state",
    "progress_motion_state",
    "progress_constraint_mode",
    "progress_request_m",
    "progress_request_effective_m",
    "progress_current_m",
    "progress_target_m",
    "progress_slack_fraction",
    "progress_slack_fraction_percent",
    "progress_predicted_terminal_m",
    "progress_predicted_safe_m",
    "progress_predicted_safe_mm",
    "progress_predicted_safe_fraction",
    "progress_solution_shortfall_m",
    "progress_solution_shortfall_mm",
    "progress_epm_reposition_activity_fraction",
    "progress_epm_first_activity_fraction",
    "progress_insertion_activity_fraction",
    "progress_angle_constraint_active",
    "progress_safety_limited_candidate",
    "progress_blocked_streak",
    "progress_reposition_streak",
    "trajectory_impossible_suggested",
    "trajectory_impossible_scope",
    "hard_constraint_infeasible",
    "controller_progress_stop_applied",
    "tip_tangent_constraint_num_active",
    "tip_tangent_angle_nominal_max_deg",
    "tip_tangent_nonlinear_max_near_wall_deg",
    "qp_infeasible",
    "safety_constraint_infeasible",
    "sqp_converged",
    "sqp_iterations_used",
    "linerr_first_mm",
    "osqp_status",
    "osqp_setup_performed",
    "osqp_workspace_reused",
    "osqp_primal_dim",
    "osqp_constraint_dim",
)


def safe_progress_log_row(info: dict[str, Any]) -> dict[str, Any]:
    """Return CSV-friendly scalar fields from one controller ``info`` dict."""
    out: dict[str, Any] = {}
    for key in SAFE_PROGRESS_LOG_FIELDS:
        value = info.get(key, np.nan)
        if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
            out[key] = value
        else:
            out[key] = str(value)
    return out
