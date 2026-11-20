"""Regression test for a real bug: `t_worker_ready` was set inside
`set_late_response_log_path` (a LATER, optional call) instead of `__init__`
(right after the real startup handshake). A caller computing
`(worker.t_worker_ready - worker.t_spawn_start)` immediately after
construction -- exactly what `spawn_and_warm_worker` in
run_mpc_delay_aware_insertion_anchor.py does, before setting the late-
response log path -- hit AttributeError. Fixed by setting t_worker_ready
right after the startup handshake in __init__ instead.

No hardware; genuine subprocess + Pipe + OSQP, synthetic schedule (same
pattern as test_offline_worker_subprocess_ipc.py).
"""
import os

import numpy as np

from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCRATCH_DIR = os.environ.get("CLAUDE_JOB_TMP", "/tmp")
SCHEDULE_PATH = os.path.join(SCRATCH_DIR, "test_synthetic_schedule_spawn_lifecycle.npy")


def _build_synthetic_schedule() -> str:
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    sched = np.tile(J0[None, :, :], (ref.sample_count, 1, 1))
    np.save(SCHEDULE_PATH, sched)
    return SCHEDULE_PATH


def test_t_worker_ready_set_immediately_after_construction_before_log_path_set():
    schedule_path = _build_synthetic_schedule()
    worker = MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=schedule_path,
        mpc_config_kwargs=dict(
            sample_period_s=0.1, prediction_horizon=10,
            state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
            velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        ),
        beam_config_kwargs=dict(
            position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
            use_dare_terminal_cost=False, directional_damping=0.0,
        ),
        delay_samples=2, beta_d=1.0, single_threaded=True,
    )
    try:
        # exactly spawn_and_warm_worker's own line, run BEFORE
        # set_late_response_log_path is ever called (matching the real
        # runner's lifecycle order: spawn worker, THEN create the output
        # dir, THEN set the log path).
        spawn_ms = (worker.t_worker_ready - worker.t_spawn_start) * 1e3
        assert spawn_ms >= 0.0
        assert worker.t_spawn_start <= worker.t_worker_ready
    finally:
        worker.close()
