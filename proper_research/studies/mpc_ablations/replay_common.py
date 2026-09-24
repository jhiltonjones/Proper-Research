"""Shared infrastructure for every study in this package: load a REAL logged
closed-loop rectangle run, load/build its Jacobian schedule, and re-solve any
delay-aware MPC controller class at each real historical tick using the
ACTUAL recorded measured state / previous command / previous input --
"single-tick counterfactual replay" (same technique used throughout this
project's own ablation history, e.g. the manipulability_ablation package's
A/B/C replay). Because each tick is re-solved independently against the REAL
recorded state (not the counterfactual controller's own rollout), trajectories
never diverge/compound -- this isolates each formulation's one-step decision
cleanly, which is what you want when comparing controller VARIANTS, not
re-running full closed-loop control.

Every study script in this directory imports from here rather than
duplicating this logic. Nothing here touches hardware.
"""
from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.planning.planning_context import build_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)

Array = np.ndarray

# The default real closed-loop log this package's studies replay against:
# the validated frozen rectangle controller (gamma=0, R700, d=2, beta_d=1,
# N=15, V_f=0 -- see CONTROLLER_JUSTIFICATION.md), a complete 274-tick run.
# Any other close_loop_logs/... run with the same plan_dir works too -- pass
# --log-dir to any script here to use a different one.
DEFAULT_LOG_DIR = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_accumC_20260921T103319Z"
DEFAULT_SCHEDULE_CACHE = "/tmp/rectangle_study_schedule.npy"


def load_real_log(log_dir: str) -> tuple[list[dict], dict]:
    """Return (path_follow rows, controller_metadata) for a real logged run."""
    rows = [json.loads(line) for line in open(f"{log_dir}/path_follow.jsonl")]
    meta_path = Path(log_dir) / "controller_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return rows, meta


def build_schedule(plan_dir: str, cache_path: str | None = DEFAULT_SCHEDULE_CACHE,
                    contact: bool = False) -> Array:
    """Genuine from_model_bundle Jacobian schedule for `plan_dir` -- cached
    after the first (slow, ~90-150s, one nonlinear beam solve per reference
    sample) build. `contact=False` matches the validated rectangle
    controller (free-space, no wall)."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    reference = load_configuration_reference(plan_dir, require_planned_beam_feasible=False)
    _, bundle, controller_pack, _ = build_planning_context()
    jac_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack, contact=contact, jacobian_mode="accurate",
    )
    schedule = mpc_variants.precompute_schedule(
        reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    )
    if cache_path:
        np.save(cache_path, schedule)
    return schedule


def build_mpc_config(
    *, dt: float, horizon: int, input_tracking_weight: float, input_increment_weight: float = 1.0e-3,
    joint_velocity_limit_rad_s: float = 0.10, insertion_rate_limit_m_s: float = 2.0e-3,
    joint_acceleration_limit_rad_s2: float = 0.5, insertion_min_m: float = -0.05,
    insertion_max_m: float = 0.20,
) -> ConfigurationMPCConfig:
    """The frozen hard-constraint / scale convention used everywhere in this
    project's MPC work (see CONTROLLER_JUSTIFICATION.md secs 7-10) -- only
    `input_tracking_weight` (R) is meant to vary between study conditions."""
    increment_scale = tuple(
        dt * a for a in ([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s])
    )
    return ConfigurationMPCConfig(
        sample_period_s=dt, prediction_horizon=int(horizon),
        state_min=tuple([-2.0 * math.pi] * 6 + [insertion_min_m]),
        state_max=tuple([2.0 * math.pi] * 6 + [insertion_max_m]),
        velocity_limit=tuple([joint_velocity_limit_rad_s] * 6 + [insertion_rate_limit_m_s]),
        acceleration_limit=tuple([joint_acceleration_limit_rad_s2] * 6 + [10.0 * insertion_rate_limit_m_s]),
        input_tracking_weight=float(input_tracking_weight),
        input_increment_weight=float(input_increment_weight),
        input_increment_scale=increment_scale,
    )


def build_beam_config(position_error_scale_mm: float = 0.5, position_tracking_weight: float = 1.0) -> BeamOutputMPCConfig:
    s = float(position_error_scale_mm) * 1.0e-3
    return BeamOutputMPCConfig(
        position_error_scale_m=(s, s, s), position_tracking_weight=float(position_tracking_weight),
        use_dare_terminal_cost=False, directional_damping=0.0,
    )


def reconstruct_solve_inputs(rows: list[dict], i: int) -> dict[str, Any]:
    """Rebuild the exact solve_delay_aware(...) kwargs for tick i from the
    logged path_follow.jsonl rows -- q_cmd/q_cmd_prev are the accumulator's
    OWN tracked commanded joints (the previous tick's q_target_rad, NOT
    q_meas_rad -- see process_isolated_adapter.py's __call__), previous_input
    is the previous tick's actually-issued u0."""
    row = rows[i]
    q_meas = np.asarray(row["q_meas_rad"], dtype=float)
    L_meas = float(row["insertion_length_m"])
    z_meas = np.concatenate([q_meas, [L_meas]])
    measured_beam_position = np.asarray(row["tip_mm"], dtype=float) / 1000.0
    control_index = int(row["ref_index"])
    if i == 0:
        q_cmd = q_meas.copy()
        q_cmd_prev = q_meas.copy()
        ins_cmd = L_meas
        previous_input = np.zeros(7)
    else:
        prev_row = rows[i - 1]
        q_cmd = np.asarray(prev_row["q_target_rad"], dtype=float)
        ins_cmd = L_meas
        q_cmd_prev = (
            q_meas.copy() if i == 1 else np.asarray(rows[i - 2]["q_target_rad"], dtype=float)
        )
        previous_input = np.asarray(prev_row["u0"], dtype=float)
    return dict(
        z_meas=z_meas, q_cmd=q_cmd, q_cmd_prev=q_cmd_prev, insertion_m=ins_cmd,
        measured_beam_position=measured_beam_position, control_index=control_index,
        previous_input=previous_input,
    )


def pick_tick_indices(n: int, n_ticks: int = 40) -> list[int]:
    """Evenly-spaced tick indices spanning a run, for sweeps where re-solving
    EVERY tick under every condition would be needlessly slow."""
    if n <= n_ticks:
        return list(range(n))
    return sorted(set(np.linspace(2, n - 1, n_ticks).astype(int).tolist()))


def ensure_outdir(name: str) -> Path:
    out = Path(__file__).parent / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    return out
