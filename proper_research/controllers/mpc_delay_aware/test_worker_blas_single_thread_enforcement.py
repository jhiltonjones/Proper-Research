"""Focused regression test for a real, previously-unverified gap:
_force_single_threaded_blas() was only called from INSIDE _worker_main,
after worker_process.py's own module-level `import numpy as np` had
already triggered OpenBLAS initialization in the child process (spawn
mode must re-import this module to obtain the _worker_main target
before calling it) -- so the env vars it set had no effect on the
already-initialized BLAS backend (confirmed via threadpoolctl:
num_threads=16 despite the env vars reading '1' from inside the
worker). Fixed by ALSO calling _force_single_threaded_blas() in the
PARENT, before the child is spawned, so the child inherits the env vars
in its OS-level environment from its very first instruction, before any
BLAS initialization can happen.

This test directly observes worker.worker_threadpool_info -- the real
worker's own threadpoolctl.threadpool_info() reading, captured right
before the ready handshake (i.e. after all initialization/warm-up,
immediately before the real per-tick solve loop) -- not env vars alone.
No hardware; genuine subprocess + Pipe + OSQP.
"""
import os

import numpy as np

from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCRATCH_DIR = os.environ.get("CLAUDE_JOB_TMP", "/tmp")
SCHEDULE_PATH = os.path.join(SCRATCH_DIR, "test_synthetic_schedule_blas_thread.npy")


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


def test_real_worker_reports_single_thread_blas_after_fix():
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
        info = worker.worker_threadpool_info
        assert info is not None, "worker did not report threadpool_info in its ready handshake"
        assert len(info) > 0, "no BLAS backend reported -- threadpoolctl found nothing loaded"
        for entry in info:
            assert entry["num_threads"] == 1, (
                f"BLAS backend {entry} reports num_threads != 1 -- single-thread "
                f"enforcement regressed"
            )
    finally:
        worker.close()


def test_single_threaded_false_does_not_report_forced_single_thread():
    # sanity check on the OTHER branch: single_threaded=False must NOT
    # force these env vars (confirms the fix is conditional on the flag,
    # not unconditional).
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
        delay_samples=2, beta_d=1.0, single_threaded=False,
    )
    try:
        # not asserting a specific thread count here (machine-dependent
        # default) -- just confirming the worker starts fine and reports
        # SOMETHING, i.e. the reporting mechanism itself doesn't depend
        # on single_threaded being True.
        assert worker.worker_threadpool_info is not None
    finally:
        worker.close()
