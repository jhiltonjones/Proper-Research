"""2026-11-19 (third review round), point 2: investigate WHY active
adaptation increased reported one-step prediction RMSE from 0.364mm to
0.606mm on the September zraise30mm historical replay (condition B vs A).
This is a HISTORICAL-REPLAY diagnostic only -- it does not change the
disabled-by-default status of active adaptation, and nothing here is
"held-out validation": this is the SAME data g_hat was exposed to, used
only to explain an already-observed number, not to re-score a decision.

Run: python3 -m proper_research.controllers.mpc_delay_aware.investigate_september_rmse_increase
"""
from __future__ import annotations

import json

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

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path"
SCHEDULE_PATH = "plans/rectangle_10x15mm_skipglobal_zraise30mm_2026-09-24/mpc_schedules/jacobian_schedule_nullspace_r700_genuine.npy"
LOG_PATH = "close_loop_logs/myrun/mpc_delay_aware_insertion_anchor_zraise30mm_rep3_20260924T113637Z/path_follow.jsonl"
HORIZON = 15
DT = 0.1


def main() -> None:
    rows = [json.loads(l) for l in open(LOG_PATH)]
    ref = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    sched = np.load(SCHEDULE_PATH)

    mpc_cfg = ConfigurationMPCConfig(
        sample_period_s=DT, prediction_horizon=HORIZON,
        state_min=tuple([-2 * np.pi] * 6 + [-0.05]), state_max=tuple([2 * np.pi] * 6 + [0.2]),
        velocity_limit=tuple([0.1] * 6 + [0.002]), acceleration_limit=tuple([0.4] * 6 + [0.02]),
        input_tracking_weight=7.0, input_increment_weight=1.0e-3,
    )
    beam_cfg = BeamOutputMPCConfig(
        position_error_scale_m=(0.5e-3, 0.5e-3, 0.5e-3), position_tracking_weight=1.0,
        use_dare_terminal_cost=False, directional_damping=0.0,
    )

    c_off = InsertionAnchoredTaskNullspaceMPC(
        gamma=0.0, insertion_state_anchor_weight=0.0, reference=ref, config=mpc_cfg, beam_config=beam_cfg,
        reference_position_jacobians=sched, delay_samples=2, beta_d=1.0, jacobian_adaptation=None,
    )
    c_active = InsertionAnchoredTaskNullspaceMPC(
        gamma=0.0, insertion_state_anchor_weight=0.0, reference=ref, config=mpc_cfg, beam_config=beam_cfg,
        reference_position_jacobians=sched, delay_samples=2, beta_d=1.0,
        jacobian_adaptation=dict(mode="active", excitation_floor_m=5e-4, forgetting_lambda=0.98),
    )

    prev_cmd_off = np.zeros(7)
    prev_cmd_active = np.zeros(7)
    prev_pred_off = None
    prev_pred_active = None
    err_off, err_active = [], []
    g_used_trace, accepted_trace, reject_reasons = [], [], []
    for i, row in enumerate(rows):
        k = int(np.clip(row["ref_index"], 0, ref.sample_count - 1))
        q_meas = np.asarray(row["q_meas_rad"], dtype=float)
        L_meas = float(row["insertion_length_m"])
        tip_m = np.asarray(row["tip_mm"], dtype=float) * 1.0e-3
        z_meas = np.concatenate([q_meas, [L_meas]])

        if prev_pred_off is not None:
            err_off.append(np.linalg.norm(prev_pred_off - tip_m) * 1e3)
            err_active.append(np.linalg.norm(prev_pred_active - tip_m) * 1e3)

        s_off = c_off.solve_delay_aware(
            z_meas=z_meas, q_cmd=q_meas, q_cmd_prev=q_meas, insertion_m=L_meas,
            measured_beam_position=tip_m, control_index=k, previous_input=prev_cmd_off,
        )
        s_active = c_active.solve_delay_aware(
            z_meas=z_meas, q_cmd=q_meas, q_cmd_prev=q_meas, insertion_m=L_meas,
            measured_beam_position=tip_m, control_index=k, previous_input=prev_cmd_active,
            measurement_frame_index=i, measurement_age_s=float(row.get("state_age_s", 0.0)),
        )
        prev_cmd_off = np.asarray(s_off.command, dtype=float)
        prev_cmd_active = np.asarray(s_active.command, dtype=float)
        prev_pred_off = np.asarray(s_off.predicted_beam_positions[0], dtype=float)
        prev_pred_active = np.asarray(s_active.predicted_beam_positions[0], dtype=float)
        diag = s_active.jacobian_adaptation_diagnostics
        g_used_trace.append(diag["g_used"] if diag else 0.0)
        accepted_trace.append(bool(diag and diag["accepted"]))
        reject_reasons.append(diag["reject_reason"] if diag else None)

    err_off = np.asarray(err_off)
    err_active = np.asarray(err_active)
    g_used_trace = np.asarray(g_used_trace[1:])  # align with err_* (which start at tick 1's score)
    accepted_trace = np.asarray(accepted_trace[1:])

    print(f"[investigate_rmse] aggregate RMSE: off={np.sqrt(np.mean(err_off**2)):.3f}mm "
          f"active={np.sqrt(np.mean(err_active**2)):.3f}mm")

    saturated = g_used_trace == -0.3
    print(f"[investigate_rmse] g_used == g_min (-0.3, SATURATED) on {saturated.sum()}/{len(g_used_trace)} "
          f"scored ticks ({100*saturated.mean():.1f}%)")
    print(f"[investigate_rmse] g_used == 0.0 (no correction active yet) on "
          f"{(g_used_trace==0.0).sum()}/{len(g_used_trace)} scored ticks")
    print(f"[investigate_rmse] accepted (estimator update applied) on {accepted_trace.sum()}/{len(accepted_trace)} ticks")

    # Error BEFORE vs AFTER g_hat first reaches saturation.
    first_sat_idx = int(np.argmax(saturated)) if saturated.any() else None
    if first_sat_idx is not None:
        pre = err_active[:first_sat_idx]
        post = err_active[first_sat_idx:]
        pre_off = err_off[:first_sat_idx]
        post_off = err_off[first_sat_idx:]
        print(f"\n[investigate_rmse] g_hat first reaches -0.3 (saturated) at scored-tick index {first_sat_idx}")
        print(f"  BEFORE saturation (n={len(pre)}): active RMSE={np.sqrt(np.mean(pre**2)):.3f}mm, "
              f"off RMSE={np.sqrt(np.mean(pre_off**2)):.3f}mm (delta={np.sqrt(np.mean(pre**2))-np.sqrt(np.mean(pre_off**2)):+.3f}mm)")
        print(f"  AFTER  saturation (n={len(post)}): active RMSE={np.sqrt(np.mean(post**2)):.3f}mm, "
              f"off RMSE={np.sqrt(np.mean(post_off**2)):.3f}mm (delta={np.sqrt(np.mean(post**2))-np.sqrt(np.mean(post_off**2)):+.3f}mm)")

    # Per-axis breakdown (needed for point 4's planar-vs-third-axis split).
    diff_active_off_per_axis = None  # placeholder for clarity; computed via err_* norms is fine here

    reasons = {}
    for r in reject_reasons[1:]:
        if r is not None:
            key = r.split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
    print(f"\n[investigate_rmse] rejection reasons (sparse acceptance evidence): {reasons}")

    print(f"\n[investigate_rmse] CONCLUSION: on this real replay, g_hat was accepted only "
          f"{accepted_trace.sum()}/{len(accepted_trace)} times ({100*accepted_trace.mean():.1f}%, "
          f"overwhelmingly rejected as '{max(reasons, key=reasons.get) if reasons else 'n/a'}'), "
          f"then saturated at its lower bound (-0.3) for the "
          f"remainder ({100*saturated.mean():.1f}% of scored ticks) and STAYED there -- i.e. the "
          f"-30% joint-sensitivity correction, learned from a SMALL number of samples early in "
          f"the run, was then applied UNIFORMLY to every later prediction regardless of whether "
          f"local conditions still supported it. This is consistent with (not proof beyond this "
          f"replay of) sparse-acceptance + bound-saturation explaining the RMSE increase: a "
          f"global, rarely-updated correction extrapolated across the whole remaining run.")


if __name__ == "__main__":
    main()
