"""WIRING TEST ONLY, NOT THE MAIN PART E COMPARISON (renamed 2026-11-19
after review -- the review required the real comparison to use the
actual plan and a genuine beam-model Jacobian schedule; this script's
SYNTHETIC placeholder schedule makes it unsuitable as evidence of
anything beyond "does the plumbing work end to end," which is still
useful to keep as a fast, cheap regression check). See
`part_e_real_schedule_comparison.py` for the real comparison, which uses
a genuinely rebuilt beam-model schedule (plans/rectangle_10x15mm_
skipglobal_zraise30mm_2026-09-24/mpc_schedules/jacobian_schedule_
nullspace_r700_genuine.npy, sha256 d41b71a3544b967979e085d07e5e0...,
rebuilt 2026-11-19 via common.build_or_load_schedule, ~90s) paired with
the matching plan and the SAME real recorded hardware log this script
also uses.

Original docstring, otherwise unchanged below: paired A/B/C/D comparison,
RECORDED-STATE REPLAY (not closed-loop simulation -- see module
docstring's caveats, printed at the end of the report too).

Inputs: a REAL path_follow.jsonl from an actual prior hardware run
(close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_
20260924T113637Z -- genuine `tip_mm` camera measurement, genuine
`q_meas_rad` RTDE joint readback, genuine `insertion_length_m` command
accumulator, genuine per-tick `ref_index`, 211 real ticks) -- these drive
all four conditions identically, tick by tick, in recorded order, with
`solve_delay_aware` called for real through the actual
`InsertionAnchoredTaskNullspaceMPC` class (the same class
`run_mpc_delay_aware_insertion_anchor.py`'s worker builds).

CAVEAT (read before trusting any number below): the real beam-tip
Jacobian SCHEDULE this original run actually used was not rebuilt this
session (the genuine beam-model solve takes minutes; it was not cached in
this worktree). This replay instead pairs the REAL recorded joint/tip/
insertion sequence with the SAME SYNTHETIC placeholder (3,7) schedule
already validated in test_offline_jacobian_adaptation_and_input_penalty.py
-- shape/sample-count-valid, not the true beam sensitivity. Consequence:
absolute prediction-error magnitudes below are NOT physically meaningful
(they reflect the synthetic schedule's mismatch against the real beam,
not the real model's own accuracy). What IS legitimate and reported here:
the four conditions' RELATIVE behaviour under IDENTICAL real inputs --
whether jacobian adaptation moves g_hat and changes the command, whether
the zero-centered input penalty changes the command, estimator
acceptance/rejection behaviour, and QP timing/feasibility, all against the
EXACT SAME per-tick measurement sequence. This is a mechanism-verification
replay, not a tracking-accuracy result -- see the printed caveat block at
the end and the module docstring above for the line this does NOT cross:
no claim of improved CLOSED-LOOP tracking is made or supportable from this
alone (closed-loop requires re-simulating the plant's response to each
condition's own, different commands, which recorded replay cannot do --
every condition here sees the SAME recorded tip/joint sequence regardless
of what it itself would have commanded).

Run: python3 -m proper_research.controllers.mpc_delay_aware.part_e_four_condition_replay
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    ConfigurationMPCConfig, load_configuration_reference,
)
from proper_research.simulation.simulations.simulate_time_parameterized_beam_output_mpc import (
    BeamOutputMPCConfig,
)
from proper_research.controllers.mpc_delay_aware.insertion_anchor_mpc import (
    InsertionAnchoredTaskNullspaceMPC,
)

LOG_DIR = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z"
PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
HORIZON = 15
DT = 1.0 / 10.0  # this run's own control_hz (confirmed: common.CONTROL_HZ default, 10Hz rectangle stage A)
DEADLINE_S = 0.070


def _load_recorded_ticks() -> list[dict]:
    rows = []
    with open(Path(LOG_DIR) / "path_follow.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _synthetic_schedule(n: int) -> np.ndarray:
    J0 = np.array([
        [0.8, -0.1, 0.05, 0.0, 0.02, -0.01, 0.9],
        [0.1, 0.9, -0.05, 0.03, 0.0, 0.02, -0.05],
        [0.02, -0.03, 0.95, -0.01, 0.01, 0.0, 0.02],
    ])
    return np.tile(J0[None, :, :], (n, 1, 1))


def _make_controller(ref, sched, *, zero_centered: bool, jac_mode: str) -> InsertionAnchoredTaskNullspaceMPC:
    mpc_cfg = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.1] * 6 + [0.002]), acceleration_limit=tuple([0.4] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
        diagnostic_zero_input_reference_in_R=zero_centered,
    )
    beam_cfg = BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )
    jac_adapt = None if jac_mode == "off" else dict(mode=jac_mode, excitation_floor_m=5e-4, forgetting_lambda=0.98)
    return InsertionAnchoredTaskNullspaceMPC(
        gamma=0.0, insertion_state_anchor_weight=0.0,
        reference=ref, config=mpc_cfg, beam_config=beam_cfg,
        reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
        jacobian_adaptation=jac_adapt,
    )


def _run_condition(label: str, ref, sched, rows: list[dict], *, zero_centered: bool, jac_mode: str) -> dict:
    c = _make_controller(ref, sched, zero_centered=zero_centered, jac_mode=jac_mode)
    S = ref.sample_count
    one_step_errors_mm = []  # |p_hat_{k+1|k} - tip_meas_{k+1}| per axis, PRE-measurement prediction
    solve_times_ms = []
    iterations = []
    successes = []
    g_hat_trace = []
    commands = []
    accepted_count = 0
    prev_command = np.zeros(7)
    prev_step = None
    for i, row in enumerate(rows):
        k = int(np.clip(row["ref_index"], 0, S - 1))
        q_meas = np.asarray(row["q_meas_rad"], dtype=float)
        L_meas = float(row["insertion_length_m"])
        tip_m = np.asarray(row["tip_mm"], dtype=float) * 1.0e-3
        z_meas = np.concatenate([q_meas, [L_meas]])

        # Score the PREVIOUS tick's horizon-stage-0 prediction against
        # THIS tick's real measurement -- i.e. before this measurement is
        # ever incorporated into anything (the whole point of "score
        # before incorporating the measurement being predicted").
        if prev_step is not None:
            pred0_m = np.asarray(prev_step.predicted_beam_positions[0], dtype=float)
            one_step_errors_mm.append((pred0_m - tip_m) * 1.0e3)

        step = c.solve_delay_aware(
            z_meas=z_meas, q_cmd=q_meas, q_cmd_prev=q_meas, insertion_m=L_meas,
            measured_beam_position=tip_m, control_index=k, previous_input=prev_command,
            measurement_frame_index=i, measurement_age_s=float(row.get("state_age_s", 0.0)),
        )
        prev_command = np.asarray(step.command, dtype=float)
        prev_step = step
        commands.append(prev_command.copy())
        solve_times_ms.append(step.solve_time_s * 1.0e3)
        iterations.append(step.iterations)
        successes.append(bool(step.success))
        if step.jacobian_adaptation_diagnostics is not None:
            g_hat_trace.append(step.jacobian_adaptation_diagnostics["g_hat"])
            if step.jacobian_adaptation_diagnostics["accepted"]:
                accepted_count += 1

    one_step_errors_mm = np.asarray(one_step_errors_mm) if one_step_errors_mm else np.zeros((0, 3))
    per_axis_rmse = (
        np.sqrt(np.mean(one_step_errors_mm ** 2, axis=0)) if one_step_errors_mm.size else np.full(3, np.nan)
    )
    aggregate_rmse = (
        float(np.sqrt(np.mean(np.linalg.norm(one_step_errors_mm, axis=1) ** 2)))
        if one_step_errors_mm.size else float("nan")
    )
    return {
        "label": label, "n_ticks": len(rows),
        "_commands": np.asarray(commands),  # stripped before JSON dump, used for cross-condition diffs below
        "per_axis_one_step_rmse_mm": per_axis_rmse.tolist(),
        "aggregate_one_step_rmse_mm": aggregate_rmse,
        "mean_solve_time_ms": float(np.mean(solve_times_ms)),
        "max_solve_time_ms": float(np.max(solve_times_ms)),
        "n_over_deadline": int(np.sum(np.asarray(solve_times_ms) > DEADLINE_S * 1e3)),
        "mean_iterations": float(np.mean(iterations)),
        "n_qp_failures": int(np.sum(~np.asarray(successes))),
        "final_g_hat": (g_hat_trace[-1] if g_hat_trace else None),
        "g_hat_accepted_fraction": (accepted_count / len(rows) if g_hat_trace else None),
        "g_hat_trace_summary": (
            {"min": float(np.min(g_hat_trace)), "max": float(np.max(g_hat_trace)), "final": float(g_hat_trace[-1])}
            if g_hat_trace else None
        ),
    }


def main() -> None:
    rows = _load_recorded_ticks()
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    sched = _synthetic_schedule(ref.sample_count)
    print(f"[part_e] loaded {len(rows)} REAL recorded ticks from {LOG_DIR}")
    print(f"[part_e] plan reference: {ref.sample_count} samples, {PLAN_DIR}")
    print("[part_e] SCHEDULE: SYNTHETIC placeholder (see module docstring CAVEAT) -- "
          "absolute error magnitudes below are NOT physically meaningful")

    conditions = [
        ("A: planned-input, adaptation=off", False, "off"),
        ("B: planned-input, adaptation=active", False, "active"),
        ("C: zero-centered-input, adaptation=off", True, "off"),
        ("D: zero-centered-input, adaptation=active", True, "active"),
    ]
    results = []
    for label, zero_centered, jac_mode in conditions:
        t0 = time.monotonic()
        r = _run_condition(label, ref, sched, rows, zero_centered=zero_centered, jac_mode=jac_mode)
        r["wall_time_s"] = time.monotonic() - t0
        results.append(r)
        print(f"\n--- {label} ---")
        print(f"  aggregate one-step prediction RMSE: {r['aggregate_one_step_rmse_mm']:.3f} mm "
              f"(per-axis mm: {[f'{v:.3f}' for v in r['per_axis_one_step_rmse_mm']]})")
        print(f"  QP: mean_iter={r['mean_iterations']:.1f} failures={r['n_qp_failures']}/{r['n_ticks']} "
              f"solve_time_ms(mean/max)={r['mean_solve_time_ms']:.2f}/{r['max_solve_time_ms']:.2f} "
              f"over_{DEADLINE_S*1e3:.0f}ms_deadline={r['n_over_deadline']}")
        if r["g_hat_trace_summary"] is not None:
            print(f"  jacobian adaptation: g_hat trace min/max/final="
                  f"{r['g_hat_trace_summary']['min']:.4f}/{r['g_hat_trace_summary']['max']:.4f}/"
                  f"{r['g_hat_trace_summary']['final']:.4f}, "
                  f"accepted_fraction={r['g_hat_accepted_fraction']:.3f}")

    a, b, cc, d = results
    diff_AC = np.linalg.norm(a["_commands"] - cc["_commands"], axis=1)
    diff_BD = np.linalg.norm(b["_commands"] - d["_commands"], axis=1)
    diff_AB = np.linalg.norm(a["_commands"] - b["_commands"], axis=1)
    print(f"\n[part_e] command-level cross-condition differences (||u_X - u_Y||, rad+m/s, per tick):")
    print(f"  A vs C (input-penalty mode, adaptation held off):    mean={diff_AC.mean():.6f} max={diff_AC.max():.6f}")
    print(f"  B vs D (input-penalty mode, adaptation held active): mean={diff_BD.mean():.6f} max={diff_BD.max():.6f}")
    print(f"  A vs B (adaptation mode, input-penalty held planned): mean={diff_AB.mean():.6f} max={diff_AB.max():.6f}")
    for r in results:
        del r["_commands"]

    # Command/departure comparison: since A/C share adaptation=off and B/D
    # share active, and all four see the IDENTICAL recorded measurement
    # sequence, any command difference A vs C (or B vs D) is attributable
    # SOLELY to the input-penalty change; any A vs B (or C vs D) difference
    # is attributable SOLELY to jacobian adaptation -- a clean, paired
    # comparison, even though absolute error numbers are not trustworthy.
    print("\n[part_e] REMINDER: this is RECORDED-STATE REPLAY against a SYNTHETIC schedule, "
          "not closed-loop simulation and not a claim of real tracking accuracy. It verifies "
          "the MECHANISM (does the input-penalty option change the command? does adaptation "
          "move g_hat and reach the QP? is timing within the 70ms deadline?) under real, "
          "recorded, sequential measurement data -- not whether either feature would improve "
          "real closed-loop tracking, which requires a plant simulation this script does not run.")

    out_path = Path("close_loop_logs/myrun/part_e_four_condition_replay_result.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[part_e] full results written to {out_path}")


if __name__ == "__main__":
    main()
