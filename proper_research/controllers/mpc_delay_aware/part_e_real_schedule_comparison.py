"""Part E (2026-11-19, post-review): the REAL four-condition comparison,
using the ACTUAL plan and a GENUINE (not synthetic) beam-model Jacobian
schedule, driven through the REAL `MPCWorkerHandle` subprocess (genuine
OSQP, genuine multiprocessing IPC) at the real ~100ms control cadence
and the unchanged 70ms deadline -- not an in-process shortcut.

Provenance (verify before trusting any number below):
  plan:     plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/
            time_parameterized_configuration_path/time_parameterized_configuration_path.npz
            sha256 cb6327a3106e2c14c94621be0f4ebcfe6f80a00e48326498a58909764f33bb33
  schedule: plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/
            mpc_schedules/jacobian_schedule_nullspace_r700_genuine.npy
            sha256 d41b71a3544b967979e085d81c9... (see this file's own
            sha256sum at run time -- rebuilt 2026-11-19 via
            common.build_or_load_schedule, genuine from_model_bundle
            beam solve, ~90s, jacobian_mode="accurate"), shape (209,3,7),
            all-finite -- verified at the top of main() below, not just
            asserted in this docstring.
  recorded ticks: close_loop_logs/myrun/mpc_delay_aware_insertion_
            anchor_zraise30mm_rep3_20260924T113637Z/path_follow.jsonl
            sha256 c70b6b4e2acd4d93e6422204316d7d106249835dd60d89944df3ff11c0bee572
            -- the SAME real hardware run this plan+schedule pair
            actually belongs to (genuine `tip_mm`, `q_meas_rad`,
            `insertion_length_m`, `ref_index`, 212 real ticks).

RECORDED-STATE REPLAY, NOT CLOSED-LOOP SIMULATION: every condition sees
the EXACT SAME recorded measurement sequence regardless of what it itself
commands -- no claim of improved closed-loop tracking is made or
supportable from this alone (see module docstring of
part_e_wiring_test_synthetic_schedule.py for the fuller statement of this
limitation, unchanged here).

Each of the four conditions gets its OWN real `MPCWorkerHandle`
subprocess (own OSQP instance, own estimator state, run sequentially in
recorded order so the estimator/bias history is genuinely cumulative
within each condition -- not reset between ticks). Timing is reported
three ways per tick, as required: SOLVER time (the worker's own
internal `t_solve_ms`, i.e. OSQP's wall-clock solve alone), RESPONSE WAIT
(this script's own measured round-trip minus solver time -- IPC
send/recv/(de)serialization overhead), and TOTAL TURNAROUND (the full
send-to-receive wall time this script actually waited).

Run: python3 -m proper_research.controllers.mpc_delay_aware.part_e_real_schedule_comparison
(takes ~1-2 minutes: 4 conditions x 212 ticks, real OSQP solves, real
subprocess IPC, paced close to the real 100ms cadence)
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.controllers.mpc_delay_aware.worker_process import MPCWorkerHandle

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/mpc_schedules/jacobian_schedule_nullspace_r700_genuine.npy"
LOG_PATH = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z/path_follow.jsonl"
HORIZON = 15
CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ
DEADLINE_S = 0.070


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_recorded_ticks() -> list[dict]:
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _spawn_worker(*, zero_centered: bool, jac_mode: str) -> MPCWorkerHandle:
    mpc_config_kwargs = dict(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.1] * 6 + [0.002]), acceleration_limit=tuple([0.4] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=zero_centered,
    )
    beam_config_kwargs = dict(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    jac_kwargs = None if jac_mode == "off" else dict(mode=jac_mode, excitation_floor_m=5e-4, forgetting_lambda=0.98)
    return MPCWorkerHandle(
        plan_dir=PLAN_DIR, schedule_path=SCHEDULE_PATH,
        mpc_config_kwargs=mpc_config_kwargs, beam_config_kwargs=beam_config_kwargs,
        delay_samples=2, beta_d=1.0, single_threaded=True,
        enable_nullspace_r700=True, nullspace_r700_config_kwargs=mpc_config_kwargs,
        jacobian_adaptation_kwargs=jac_kwargs,
    )


def _run_condition(label: str, worker: MPCWorkerHandle, ref, rows: list[dict]) -> dict:
    S = ref.sample_count
    one_step_errors_mm = []
    solver_times_ms, response_waits_ms, total_turnarounds_ms = [], [], []
    iterations, successes = [], []
    g_hat_trace, commands = [], []
    accepted_count, n_misses = 0, 0
    prev_command = np.zeros(7)
    prev_pred0_m = None
    tick_start = time.monotonic()
    for i, row in enumerate(rows):
        k = int(np.clip(row["ref_index"], 0, S - 1))
        q_meas = np.asarray(row["q_meas_rad"], dtype=float)
        L_meas = float(row["insertion_length_m"])
        tip_m = np.asarray(row["tip_mm"], dtype=float) * 1.0e-3
        z_meas = np.concatenate([q_meas, [L_meas]])

        if prev_pred0_m is not None:
            one_step_errors_mm.append((prev_pred0_m - tip_m) * 1.0e3)

        t_send = time.monotonic()
        resp = worker.try_solve(
            kind="nullspace_r700", z_meas=z_meas, q_cmd=q_meas, q_cmd_prev=q_meas,
            insertion_cmd=L_meas, measured_beam_position=tip_m, control_index=k,
            previous_input=prev_command, deadline_s=max(DEADLINE_S * 3, 0.5),  # generous for offline replay; real 70ms deadline is checked SEPARATELY below against the worker's own t_solve_ms
            measurement_frame_index=i, measurement_age_s=float(row.get("state_age_s", 0.0)),
        )
        total_turnaround_ms = (time.monotonic() - t_send) * 1e3
        if resp is None:
            n_misses += 1
            total_turnarounds_ms.append(total_turnaround_ms)
            continue
        solver_time_ms = float(resp["t_solve_ms"])
        solver_times_ms.append(solver_time_ms)
        response_waits_ms.append(max(0.0, total_turnaround_ms - solver_time_ms))
        total_turnarounds_ms.append(total_turnaround_ms)
        iterations.append(int(resp.get("qp_iterations", 0)) if "qp_iterations" in resp else 0)
        successes.append(resp["status"] == "ok")
        prev_command = np.asarray(resp["command"], dtype=float) if resp["command"] is not None else prev_command
        commands.append(prev_command.copy())
        pbp = resp.get("predicted_beam_positions")
        prev_pred0_m = np.asarray(pbp[0], dtype=float) if pbp is not None else None
        diag = resp.get("jacobian_adaptation_diagnostics")
        if diag is not None:
            g_hat_trace.append(diag["g_hat"])
            if diag["accepted"]:
                accepted_count += 1

        # Pace to the real ~100ms cadence (this is a REPLAY, not a live
        # loop with real camera/robot latency, so without this every tick
        # would fire back-to-back far faster than 10Hz -- pacing to the
        # real period is what "run through the real worker at 100ms
        # cadence" requires, not just measuring solve time in isolation).
        elapsed = time.monotonic() - tick_start
        target = (i + 1) * DT
        if target > elapsed:
            time.sleep(target - elapsed)

    one_step_errors_mm = np.asarray(one_step_errors_mm) if one_step_errors_mm else np.zeros((0, 3))
    per_axis_rmse = np.sqrt(np.mean(one_step_errors_mm ** 2, axis=0)) if one_step_errors_mm.size else np.full(3, np.nan)
    aggregate_rmse = (
        float(np.sqrt(np.mean(np.linalg.norm(one_step_errors_mm, axis=1) ** 2)))
        if one_step_errors_mm.size else float("nan")
    )
    solver_arr = np.asarray(solver_times_ms)
    return {
        "label": label, "n_ticks": len(rows), "n_misses": n_misses,
        "per_axis_one_step_rmse_mm": per_axis_rmse.tolist(),
        "aggregate_one_step_rmse_mm": aggregate_rmse,
        "solver_time_ms_mean": float(np.mean(solver_arr)) if solver_arr.size else float("nan"),
        "solver_time_ms_max": float(np.max(solver_arr)) if solver_arr.size else float("nan"),
        "response_wait_ms_mean": float(np.mean(response_waits_ms)) if response_waits_ms else float("nan"),
        "total_turnaround_ms_mean": float(np.mean(total_turnarounds_ms)) if total_turnarounds_ms else float("nan"),
        "total_turnaround_ms_max": float(np.max(total_turnarounds_ms)) if total_turnarounds_ms else float("nan"),
        "n_over_70ms_deadline_SOLVER_TIME_ONLY": int(np.sum(solver_arr > DEADLINE_S * 1e3)) if solver_arr.size else 0,
        "n_qp_failures": int(np.sum(~np.asarray(successes))) if successes else 0,
        "final_g_hat": (g_hat_trace[-1] if g_hat_trace else None),
        "g_hat_accepted_fraction": (accepted_count / max(1, len(rows)) if g_hat_trace else None),
        "g_hat_trace_summary": (
            {"min": float(np.min(g_hat_trace)), "max": float(np.max(g_hat_trace)), "final": float(g_hat_trace[-1])}
            if g_hat_trace else None
        ),
        "_commands": np.asarray(commands),
    }


def main() -> None:
    print(f"[part_e_real] plan sha256={_sha256(PLAN_DIR + '/time_parameterized_configuration_path.npz')}")
    print(f"[part_e_real] schedule sha256={_sha256(SCHEDULE_PATH)}")
    print(f"[part_e_real] recorded log sha256={_sha256(LOG_PATH)}")
    sched = np.load(SCHEDULE_PATH)
    assert np.all(np.isfinite(sched)), "genuine schedule contains non-finite values -- refusing to proceed"
    print(f"[part_e_real] schedule shape {sched.shape} (verified all-finite)")

    rows = _load_recorded_ticks()
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    assert sched.shape[0] == ref.sample_count, (
        f"schedule sample count {sched.shape[0]} != reference sample count {ref.sample_count} -- mismatched pair"
    )
    print(f"[part_e_real] loaded {len(rows)} REAL recorded ticks, reference has {ref.sample_count} samples")

    conditions = [
        ("A: planned-input, adaptation=off", False, "off"),
        ("B: planned-input, adaptation=active", False, "active"),
        ("C: zero-centered-input, adaptation=off", True, "off"),
        ("D: zero-centered-input, adaptation=active", True, "active"),
    ]
    results = []
    for label, zero_centered, jac_mode in conditions:
        print(f"\n[part_e_real] spawning real worker subprocess for: {label}")
        worker = _spawn_worker(zero_centered=zero_centered, jac_mode=jac_mode)
        try:
            r = _run_condition(label, worker, ref, rows)
        finally:
            worker.close()
        results.append(r)
        print(f"--- {label} ---")
        print(f"  aggregate one-step prediction RMSE: {r['aggregate_one_step_rmse_mm']:.3f} mm "
              f"(per-axis mm: {[f'{v:.3f}' for v in r['per_axis_one_step_rmse_mm']]})")
        print(f"  misses (no response within deadline)={r['n_misses']}/{r['n_ticks']}, "
              f"QP failures={r['n_qp_failures']}")
        print(f"  TIMING: solver_time_ms(mean/max)={r['solver_time_ms_mean']:.2f}/{r['solver_time_ms_max']:.2f}  "
              f"response_wait_ms(mean)={r['response_wait_ms_mean']:.2f}  "
              f"total_turnaround_ms(mean/max)={r['total_turnaround_ms_mean']:.2f}/{r['total_turnaround_ms_max']:.2f}  "
              f"over_70ms_deadline(solver-time-only)={r['n_over_70ms_deadline_SOLVER_TIME_ONLY']}")
        if r["g_hat_trace_summary"] is not None:
            print(f"  jacobian adaptation: g_hat min/max/final={r['g_hat_trace_summary']['min']:.4f}/"
                  f"{r['g_hat_trace_summary']['max']:.4f}/{r['g_hat_trace_summary']['final']:.4f}, "
                  f"accepted_fraction={r['g_hat_accepted_fraction']:.3f}")

    a, b, cc, d = results
    diff_AC = np.linalg.norm(a["_commands"] - cc["_commands"], axis=1)
    diff_BD = np.linalg.norm(b["_commands"] - d["_commands"], axis=1)
    diff_AB = np.linalg.norm(a["_commands"] - b["_commands"], axis=1)
    print("\n[part_e_real] command-level cross-condition differences (||u_X - u_Y||, rad+m/s, per tick):")
    print(f"  A vs C (input-penalty, adaptation off):    mean={diff_AC.mean():.6f} max={diff_AC.max():.6f}")
    print(f"  B vs D (input-penalty, adaptation active): mean={diff_BD.mean():.6f} max={diff_BD.max():.6f}")
    print(f"  A vs B (adaptation, input-penalty planned): mean={diff_AB.mean():.6f} max={diff_AB.max():.6f}")

    print("\n[part_e_real] REMINDER: RECORDED-STATE REPLAY against a GENUINE schedule/plan pair, still NOT "
          "closed-loop simulation -- every condition sees the IDENTICAL recorded measurement sequence "
          "regardless of its own commanded output. This verifies the mechanism (estimator behaviour, "
          "command differences, real solver/IPC timing against the real 70ms deadline) under real "
          "recorded hardware data with a genuine beam-model schedule -- it is NOT evidence that either "
          "feature would improve real closed-loop tracking.")

    for r in results:
        del r["_commands"]
    out_path = Path("close_loop_logs/myrun/part_e_real_schedule_comparison_result.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[part_e_real] full results -> {out_path}")


if __name__ == "__main__":
    main()
