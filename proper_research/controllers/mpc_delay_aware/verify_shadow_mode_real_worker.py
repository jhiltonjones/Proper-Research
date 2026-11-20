"""2026-11-19 (second review round), point 4 prerequisite: verify, through
the REAL worker (real OSQP, real subprocess IPC) and the EXACT plan +
genuine schedule prepared for the shadow-mode hardware command below,
that shadow mode cannot alter predictions, QP terms, or commands -- and
measure its timing overhead relative to adaptation fully off.

Run: python3 -m proper_research.controllers.mpc_delay_aware.verify_shadow_mode_real_worker
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

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/mpc_schedules/jacobian_schedule_nullspace_r700_genuine.npy"
LOG_PATH = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z/path_follow.jsonl"
HORIZON = 15
DT = 0.1


def _load_rows() -> list[dict]:
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f]


def _spawn(jac_mode: str) -> MPCWorkerHandle:
    mpc_config_kwargs = dict(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.1] * 6 + [0.002]), acceleration_limit=tuple([0.4] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=False,  # planned-input penalty, as required for this run
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


def main() -> None:
    rows = _load_rows()
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    print(f"[verify_shadow] plan sample_count={ref.sample_count}, {len(rows)} real recorded ticks")

    results = {}
    for mode in ("off", "shadow"):
        worker = _spawn(mode)
        commands, predictions, jacobians_used, solve_times_ms, diags = [], [], [], [], []
        prev_command = np.zeros(7)
        t0_all = time.monotonic()
        for i, row in enumerate(rows):
            k = int(np.clip(row["ref_index"], 0, ref.sample_count - 1))
            q_meas = np.asarray(row["q_meas_rad"], dtype=float)
            L_meas = float(row["insertion_length_m"])
            tip_m = np.asarray(row["tip_mm"], dtype=float) * 1.0e-3
            z_meas = np.concatenate([q_meas, [L_meas]])
            resp = worker.try_solve(
                kind="nullspace_r700", z_meas=z_meas, q_cmd=q_meas, q_cmd_prev=q_meas,
                insertion_cmd=L_meas, measured_beam_position=tip_m, control_index=k,
                previous_input=prev_command, deadline_s=1.0,  # generous: this is an offline equivalence check, not the live deadline (see investigate_condition_b_deadline.py for that)
                measurement_frame_index=i, measurement_age_s=float(row.get("state_age_s", 0.0)),
            )
            assert resp is not None and resp["status"] == "ok", f"tick {i} failed under mode={mode}: {resp}"
            prev_command = np.asarray(resp["command"], dtype=float)
            commands.append(prev_command.copy())
            predictions.append(np.asarray(resp["predicted_beam_positions"], dtype=float).copy())
            jacobians_used.append(np.asarray(resp["jacobian_used"], dtype=float).copy())
            solve_times_ms.append(float(resp["t_solve_ms"]))
            diags.append(resp.get("jacobian_adaptation_diagnostics"))
        worker.close()
        results[mode] = dict(
            commands=np.asarray(commands), predictions=np.asarray(predictions),
            jacobians_used=np.asarray(jacobians_used), solve_times_ms=np.asarray(solve_times_ms),
            diags=diags, wall_s=time.monotonic() - t0_all,
        )
        print(f"  mode={mode}: {len(rows)} ticks, mean solve_time_ms={results[mode]['solve_times_ms'].mean():.3f}, "
              f"wall_time_s={results[mode]['wall_s']:.2f}")

    off, shadow = results["off"], results["shadow"]
    commands_identical = np.array_equal(off["commands"], shadow["commands"])
    predictions_identical = np.array_equal(off["predictions"], shadow["predictions"])
    jacobians_identical = np.array_equal(off["jacobians_used"], shadow["jacobians_used"])
    n_shadow_accepted = sum(1 for d in shadow["diags"] if d is not None and d.get("accepted"))
    n_shadow_with_diag = sum(1 for d in shadow["diags"] if d is not None)

    print(f"\n[verify_shadow] commands bit-identical (off vs shadow): {commands_identical}")
    print(f"[verify_shadow] predicted_beam_positions bit-identical: {predictions_identical}")
    print(f"[verify_shadow] jacobian_used bit-identical: {jacobians_identical}")
    print(f"[verify_shadow] shadow estimator DID run: {n_shadow_with_diag}/{len(rows)} ticks had diagnostics, "
          f"{n_shadow_accepted} accepted (proves shadow is not simply disabled internally, "
          f"only prevented from reaching the QP)")
    assert all(d is None or d["g_used"] == 0.0 for d in shadow["diags"]), (
        "shadow mode g_used was nonzero on some tick -- this would be a real bug"
    )
    print("[verify_shadow] g_used == 0.0 on every shadow-mode tick: CONFIRMED")

    overhead_mean_ms = float(shadow["solve_times_ms"].mean() - off["solve_times_ms"].mean())
    overhead_max_ms = float(shadow["solve_times_ms"].max() - off["solve_times_ms"].max())
    print(f"\n[verify_shadow] TIMING OVERHEAD (shadow vs off, same real tick sequence, real worker):")
    print(f"  mean solve_time_ms: off={off['solve_times_ms'].mean():.3f}  shadow={shadow['solve_times_ms'].mean():.3f}  "
          f"delta={overhead_mean_ms:+.3f}ms")
    print(f"  max  solve_time_ms: off={off['solve_times_ms'].max():.3f}  shadow={shadow['solve_times_ms'].max():.3f}  "
          f"delta={overhead_max_ms:+.3f}ms")

    assert commands_identical and predictions_identical and jacobians_identical, (
        "SHADOW MODE IS NOT A NO-OP -- do not proceed to any hardware preparation with this result"
    )
    print("\n[verify_shadow] RESULT: shadow mode confirmed bit-identical to adaptation=off "
          "through the REAL worker, real OSQP, the EXACT plan+schedule prepared for the "
          "shadow-mode hardware command.")


if __name__ == "__main__":
    main()
