"""Part D, check 3/8 (subprocess level): the REAL `MPCWorkerHandle`
(genuine `multiprocessing.Process`, genuine pickling over a `Pipe`, genuine
OSQP backend) with `jacobian_adaptation_kwargs` actually set, solving
against the REAL rectangle plan reference (`load_configuration_reference`
loads the real, already-validated
`plans/rectangle_10x15mm_skipglobal_2026-09-12/...` plan). The 3x7
Jacobian SCHEDULE is a cheap synthetic stand-in (shape/sample-count valid,
not the real expensive beam-model solve) -- this test's purpose is to
confirm the worker subprocess/IPC/OSQP plumbing itself carries the new
kwargs end-to-end, not to validate beam-model numerics (already covered
in-process by test_offline_jacobian_adaptation_and_input_penalty.py).

No hardware, no RTDE, no camera import anywhere in this file. Run:
    python3 -m proper_research.controllers.mpc_delay_aware.test_offline_worker_subprocess_ipc
"""
from __future__ import annotations

import os
import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_2026-09-12/time_parameterized_configuration_path"
SCRATCH_DIR = os.environ.get("CLAUDE_JOB_TMP", "/tmp")
SCHEDULE_PATH = os.path.join(SCRATCH_DIR, "test_synthetic_schedule_jacadapt_ipc.npy")


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


def main() -> None:
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
        enable_nullspace_r700=True,
        nullspace_r700_config_kwargs=dict(
            sample_period_s=0.1, prediction_horizon=10,
            state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
            velocity_limit=tuple([0.3] * 6 + [0.01]), acceleration_limit=tuple([1.0] * 6 + [0.05]),
        ),
        jacobian_adaptation_kwargs=dict(mode="active", excitation_floor_m=1e-6, forgetting_lambda=1.0),
    )
    try:
        ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
        q0 = np.asarray(ref.state[0, :6], dtype=float)
        L0 = float(ref.state[0, 6])
        bp0 = np.asarray(ref.desired_position_m[0], dtype=float)
        prev_input = np.zeros(7)
        results = []
        for k in range(5):
            q_off = q0 + 0.01 * (k + 1)
            resp = worker.try_solve(
                kind="nullspace_r700", z_meas=np.concatenate([q_off, [L0]]),
                q_cmd=q_off, q_cmd_prev=q_off, insertion_cmd=L0,
                measured_beam_position=bp0, control_index=k, previous_input=prev_input,
                deadline_s=2.0, measurement_frame_index=k, measurement_age_s=0.01,
            )
            assert resp is not None, f"tick {k}: no response within deadline (real subprocess)"
            assert resp["status"] in ("ok", "infeasible"), resp
            results.append(resp)
            print(f"tick {k}: status={resp['status']} t_solve_ms={resp['t_solve_ms']:.2f} "
                  f"jacobian_scale_applied={resp['jacobian_scale_applied']:.4f} "
                  f"jac_adapt_diag={resp['jacobian_adaptation_diagnostics']}")
        assert any(r["jacobian_adaptation_diagnostics"] is not None for r in results), (
            "jacobian_adaptation_diagnostics never came back through the real IPC channel"
        )
        print("PASS: real MPCWorkerHandle subprocess + real OSQP + jacobian_adaptation_kwargs "
              "round-tripped successfully over real multiprocessing IPC")
    finally:
        worker.close()
        if os.path.exists(SCHEDULE_PATH):
            os.remove(SCHEDULE_PATH)


if __name__ == "__main__":
    main()
