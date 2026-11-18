"""2026-11-19 (second review round): investigate the condition-B deadline
exceedances seen in part_e_real_schedule_comparison.py. That script used a
DELIBERATELY GENEROUS try_solve deadline (0.5s, so the real 70ms miss/hold
logic was never exercised -- it only flagged, post-hoc, which ticks'
SOLVER time alone exceeded 70ms). This script instead drives the REAL
production adapter (`ProcessIsolatedDelayAwareAdapter`, the exact class
`run_mpc_delay_aware_insertion_anchor.py`'s online loop uses) against the
REAL worker with the REAL, UNMODIFIED 70ms deadline, so a genuine miss
actually engages the real hold/accumulator/recovery policy -- not a
post-hoc flag.

Run: python3 -m proper_research.controllers.mpc_delay_aware.investigate_condition_b_deadline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle
from proper_research.controllers.mpc_delay_aware.process_isolated_adapter import (
    ProcessIsolatedDelayAwareAdapter,
)
from proper_research.hardware.online.messages import StateEstimate

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/mpc_schedules/jacobian_schedule_nullspace_r700_genuine.npy"
LOG_PATH = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z/path_follow.jsonl"
HORIZON = 15
CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ
REAL_DEADLINE_S = 0.070  # the unmodified, real deadline -- NOT increased
N_REPETITIONS = 5


def _load_rows() -> list[dict]:
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _spawn_condition_b_worker() -> MPCWorkerHandle:
    mpc_config_kwargs = dict(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.1] * 6 + [0.002]), acceleration_limit=tuple([0.4] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=False,  # condition B: planned input, matching Part E
    )
    beam_config_kwargs = dict(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    jac_kwargs = dict(mode="active", excitation_floor_m=5e-4, forgetting_lambda=0.98)  # condition B: adaptation active
    return MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=SCHEDULE_PATH,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True, nullspace_r700_config_kwargs=mpc_config_kwargs,
        jacobian_adaptation_kwargs=jac_kwargs,
    )


def _run_one_repetition(rep: int, ref, rows: list[dict]) -> dict:
    worker = _spawn_condition_b_worker()
    adapter = ProcessIsolatedDelayAwareAdapter(
        reference=ref, worker=worker, deadline_s=REAL_DEADLINE_S,
        joint_velocity_limit_rad_s=0.1, max_joint_step_rad=0.01,
        controller_kind="nullspace_r700", controller_label="investigate_condition_b",
    )
    misses = []
    consecutive_trace = []
    abort_reasons = []
    q_cmd_before_miss = {}
    tick_start = time.monotonic()
    try:
        for i, row in enumerate(rows):
            q_meas = np.asarray(row["q_meas_rad"], dtype=float)
            L_meas = float(row["insertion_length_m"])
            tip_m = np.asarray(row["tip_mm"], dtype=float) * 1.0e-3
            estimate = StateEstimate(
                t_monotonic=time.monotonic(), x_meas=np.concatenate([tip_m, np.zeros(3)]),
                tip_position_m=tip_m, tip_tangent=np.array([1.0, 0.0, 0.0]),
                insertion_length_m=L_meas, robot_joints=q_meas, frame_index=i,
            )
            q_cmd_pre = adapter._q_cmd.copy() if adapter._q_cmd is not None else None
            result = adapter(estimate, DT)
            info = result.info
            consecutive_trace.append(info["consecutive_deadline_misses"])
            if info["status"] == "deadline_miss":
                misses.append({
                    "tick": i, "solve_time_s": info["solve_time_s"],
                    "consecutive_deadline_misses": info["consecutive_deadline_misses"],
                    "q_cmd_before_miss": q_cmd_pre.tolist() if q_cmd_pre is not None else None,
                    "q_cmd_after_miss": adapter._q_cmd.tolist(),
                })
            if info["abort_reason"]:
                abort_reasons.append((i, info["abort_reason"]))
            # Real ~100ms pacing.
            elapsed = time.monotonic() - tick_start
            target = (i + 1) * DT
            if target > elapsed:
                time.sleep(target - elapsed)
    finally:
        worker.close()

    # Verify the hold policy directly: for every recorded miss, q_cmd must
    # be UNCHANGED (held), not advanced -- the real safety property this
    # investigation is actually checking.
    hold_policy_correct = all(
        m["q_cmd_before_miss"] is not None and
        np.allclose(np.asarray(m["q_cmd_before_miss"]), np.asarray(m["q_cmd_after_miss"]))
        for m in misses
    )
    return {
        "repetition": rep, "n_ticks": len(rows), "n_misses": len(misses),
        "misses": misses, "max_consecutive_deadline_misses": max(consecutive_trace) if consecutive_trace else 0,
        "would_have_aborted(>=5 consecutive)": max(consecutive_trace, default=0) >= 5,
        "abort_reasons": abort_reasons,
        "hold_policy_correct_for_every_miss": hold_policy_correct,
    }


def main() -> None:
    rows = _load_rows()
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    print(f"[investigate_b] {len(rows)} real recorded ticks, real deadline={REAL_DEADLINE_S*1e3:.0f}ms "
          f"(UNCHANGED, not increased), {N_REPETITIONS} repetitions")

    all_reps = []
    for rep in range(1, N_REPETITIONS + 1):
        print(f"\n[investigate_b] --- repetition {rep}/{N_REPETITIONS} ---")
        r = _run_one_repetition(rep, ref, rows)
        all_reps.append(r)
        print(f"  misses: {r['n_misses']}/{r['n_ticks']}  "
              f"max_consecutive={r['max_consecutive_deadline_misses']}  "
              f"would_abort={r['would_have_aborted(>=5 consecutive)']}  "
              f"hold_policy_correct={r['hold_policy_correct_for_every_miss']}")
        for m in r["misses"]:
            print(f"    MISS at tick {m['tick']}: real solve_time_s={m['solve_time_s']}, "
                  f"consecutive={m['consecutive_deadline_misses']}, "
                  f"q_cmd held: {np.allclose(np.asarray(m['q_cmd_before_miss']), np.asarray(m['q_cmd_after_miss']))}")
        for i, reason in r["abort_reasons"]:
            print(f"    ABORT at tick {i}: {reason}")

    out_path = Path("close_loop_logs/myrun/investigate_condition_b_deadline_result.json")
    out_path.write_text(json.dumps(all_reps, indent=2))
    total_misses = sum(r["n_misses"] for r in all_reps)
    print(f"\n[investigate_b] TOTAL across {N_REPETITIONS} repetitions: {total_misses} real deadline misses "
          f"(real 70ms deadline, real hold/accumulator policy, NOT increased). Full detail -> {out_path}")


if __name__ == "__main__":
    main()
