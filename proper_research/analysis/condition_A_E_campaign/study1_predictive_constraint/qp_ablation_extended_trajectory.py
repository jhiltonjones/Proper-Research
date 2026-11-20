"""Study 1 extension (per user's bounded-check follow-up, after the
original 5-state ablation): extends the same-state with/without (233mm
production vs 274mm virtual) QP ablation across EVERY logged tick of the
4 completed A runs, all 5 B runs (4 completed + the 1 aborted run, included
for coverage -- its executed 118 ticks are real logged states too), using
the REAL worker's try_solve() -- no hardware, no simulation of anything not
already executed.

This is OFFLINE REPLAY against already-executed states. It demonstrates
what the QP would have commanded from that real state under a tighter
virtual radius; it is NOT proof that a real closed-loop run under a 274mm
constraint would actually reach, avoid, or recover from those states --
reoptimization at the next tick, under feedback, was not simulated here.

Joint and insertion command deltas are reported SEPARATELY (predicted_commands
has 7 channels: indices 0-5 are the 6 joint rates/positions, index 6 is
insertion -- confirmed from delay_aware_mpc.py's own predicted_commands
construction and magnet_workspace_self_test.py's own
`step.predicted_commands[0, :6]` joint-channel convention).

For the states nearest the 274mm boundary, also computes NONLINEAR FK
clearance (real urik.forward_kinematics, not the QP's own linearized
constraint model) along the full predicted command horizon for both radii,
to check whether the QP's linearized model and the true nonlinear clearance
agree or diverge at later horizon stages.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/home/jack/Proper-Research")
os.chdir("/home/jack/Proper-Research")

import numpy as np

from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    spawn_and_warm_worker, build_magnet_exclusion_schedule, load_configuration_reference,
)
from proper_research.hardware.online.rectangle_stage_a import common
from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.planning.planning_context import make_robot_config

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
PRODUCTION_RADIUS_M = (common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + 3.0) / 1e3  # 233mm
CANDIDATE_VIRTUAL_RADIUS_M = 274.0 / 1e3  # exploratory-data-selected; see STUDY1_PROPOSAL.md's explicit selection rule

RUN_DIRS = sorted(glob.glob("close_loop_logs/baseline_A_pilot_2026-11-20_retry1_*")) + \
           sorted(glob.glob("close_loop_logs/condition_B_frozen_jacobian_2026-11-20_*"))
RUN_DIRS = [d for d in RUN_DIRS if os.path.isdir(d)]

OUT_PATH = os.path.join(
    os.path.dirname(__file__), "qp_ablation_extended_trajectory_results.json"
)
OUT_SUMMARY_PATH = os.path.join(
    os.path.dirname(__file__), "qp_ablation_extended_trajectory_summary.json"
)


def _main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(3)
    robot_kin = _resolve_robot_kinematics(make_robot_config())

    # Load all states from all 9 run directories (4 completed A + 5 B,
    # including B's 1 aborted run, included explicitly for coverage).
    all_states = []
    for d in RUN_DIRS:
        p = f"{d}/predicted_beam_positions.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        for r in rows:
            all_states.append((d, r))
    print(f"[extended ablation] {len(all_states)} total logged states across {len(RUN_DIRS)} runs")

    workers = {}
    for radius_label, radius_m in (("production_233mm", PRODUCTION_RADIUS_M),
                                    ("virtual_274mm", CANDIDATE_VIRTUAL_RADIUS_M)):
        magnet_exclusion_kwargs = dict(
            position_jacobians=jacobians, nominal_positions_m=positions,
            lumen_C_m=beam_base_xyz_r.reshape(1, 3), radius_m=radius_m,
        )
        w = spawn_and_warm_worker(
            plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
            prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
            insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
            position_error_scale_mm=0.5, position_tracking_weight=1.0,
            magnet_exclusion_kwargs=magnet_exclusion_kwargs,
            insertion_command_min_m=0.020, insertion_command_max_m=0.070,
        )
        w.set_frame_transform(np.eye(3), np.zeros(3))
        workers[radius_label] = w

    t0 = time.time()
    results = []
    failures_274 = []
    try:
        for i, (d, r) in enumerate(all_states):
            q_cmd = np.asarray(r["q_cmd_k"], dtype=float)
            z_meas = np.asarray(r["z_meas"], dtype=float)
            per_radius = {}
            for radius_label, w in workers.items():
                resp = w.try_solve(
                    kind="nullspace_r700",
                    z_meas=z_meas,
                    q_cmd=q_cmd, q_cmd_prev=np.asarray(r["q_cmd_km1"], dtype=float),
                    insertion_cmd=float(r["insertion_cmd_m"]),
                    measured_beam_position=np.asarray(r["measured_beam_position_m"], dtype=float),
                    control_index=r["ref_index"], previous_input=np.asarray(r["u_prev"], dtype=float),
                    deadline_s=2.0,
                )
                per_radius[radius_label] = dict(
                    qp_status=resp.get("qp_status"),
                    predicted_commands=(
                        np.asarray(resp["predicted_commands"], dtype=float).tolist()
                        if resp.get("predicted_commands") is not None else None
                    ),
                )
            status_274 = per_radius["virtual_274mm"]["qp_status"]
            if status_274 != "solved":
                failures_274.append(dict(
                    run_dir=d, step=r["step"], ref_index=r["ref_index"],
                    qp_status_274=status_274, logged_gap_mm=r["magnet_exclusion_gap_mm"],
                ))

            entry = dict(
                run_dir=d, step=r["step"], ref_index=r["ref_index"],
                logged_gap_mm=r["magnet_exclusion_gap_mm"],
                qp_status_233=per_radius["production_233mm"]["qp_status"],
                qp_status_274=status_274,
            )
            pc233 = per_radius["production_233mm"]["predicted_commands"]
            pc274 = per_radius["virtual_274mm"]["predicted_commands"]
            if pc233 is not None and pc274 is not None:
                u0_233 = np.asarray(pc233[0], dtype=float)
                u0_274 = np.asarray(pc274[0], dtype=float)
                joint_delta = float(np.linalg.norm(u0_233[:6] - u0_274[:6]))
                insertion_delta = float(abs(u0_233[6] - u0_274[6]))
                entry["joint_delta_rad"] = joint_delta
                entry["insertion_delta_m"] = insertion_delta
            else:
                entry["joint_delta_rad"] = None
                entry["insertion_delta_m"] = None
            results.append(entry)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"[extended ablation] {i+1}/{len(all_states)} states, {elapsed:.1f}s elapsed")
    finally:
        for w in workers.values():
            w.close()

    elapsed = time.time() - t0
    print(f"[extended ablation] done: {len(results)} states in {elapsed:.1f}s")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f)
    print(f"Saved: {OUT_PATH}")

    # --- nonlinear FK clearance along the predicted horizon, for the
    # states nearest the 274mm boundary (both radii, every solved state
    # among the tightest 20 by logged clearance) ---
    solved_both = [r for r in results if r["joint_delta_rad"] is not None]
    tightest20 = sorted(solved_both, key=lambda r: r["logged_gap_mm"])[:20]

    print("[extended ablation] computing nonlinear-FK horizon clearance for the 20 tightest solved-both states "
          "(re-solving both radii once more per state to get full-horizon predicted_commands)...")
    workers2 = {}
    for radius_label, radius_m in (("production_233mm", PRODUCTION_RADIUS_M),
                                    ("virtual_274mm", CANDIDATE_VIRTUAL_RADIUS_M)):
        magnet_exclusion_kwargs = dict(
            position_jacobians=jacobians, nominal_positions_m=positions,
            lumen_C_m=beam_base_xyz_r.reshape(1, 3), radius_m=radius_m,
        )
        w = spawn_and_warm_worker(
            plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
            prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
            insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
            position_error_scale_mm=0.5, position_tracking_weight=1.0,
            magnet_exclusion_kwargs=magnet_exclusion_kwargs,
            insertion_command_min_m=0.020, insertion_command_max_m=0.070,
        )
        w.set_frame_transform(np.eye(3), np.zeros(3))
        workers2[radius_label] = w

    fk_horizon_out = []
    try:
        for entry in tightest20:
            d, step = entry["run_dir"], entry["step"]
            raw_row = None
            for dd, r in all_states:
                if dd == d and r["step"] == step:
                    raw_row = r
                    break
            q_cmd = np.asarray(raw_row["q_cmd_k"], dtype=float)
            z_meas = np.asarray(raw_row["z_meas"], dtype=float)
            per_radius_fk = {}
            for radius_label, w in workers2.items():
                resp = w.try_solve(
                    kind="nullspace_r700", z_meas=z_meas,
                    q_cmd=q_cmd, q_cmd_prev=np.asarray(raw_row["q_cmd_km1"], dtype=float),
                    insertion_cmd=float(raw_row["insertion_cmd_m"]),
                    measured_beam_position=np.asarray(raw_row["measured_beam_position_m"], dtype=float),
                    control_index=raw_row["ref_index"], previous_input=np.asarray(raw_row["u_prev"], dtype=float),
                    deadline_s=2.0,
                )
                pc = resp.get("predicted_commands")
                gaps_mm = []
                if pc is not None:
                    pc = np.asarray(pc, dtype=float)
                    for stage in range(pc.shape[0]):
                        q_stage = pc[stage, :6]
                        fk = urik.forward_kinematics(q_stage, robot_kin.dh, robot_kin.T_F_M)
                        magnet_xyz = np.asarray(fk.T_R_target[:3, 3], dtype=float)
                        gap_mm = float(np.linalg.norm(magnet_xyz - beam_base_xyz_r)) * 1e3
                        gaps_mm.append(gap_mm)
                per_radius_fk[radius_label] = dict(
                    qp_status=resp.get("qp_status"), nonlinear_fk_gap_mm_by_stage=gaps_mm,
                )
            fk_horizon_out.append(dict(
                run_dir=d, step=step, logged_gap_mm=entry["logged_gap_mm"],
                production_233mm=per_radius_fk["production_233mm"],
                virtual_274mm=per_radius_fk["virtual_274mm"],
            ))
    finally:
        for w in workers2.values():
            w.close()

    with open(os.path.join(os.path.dirname(__file__), "qp_ablation_nonlinear_fk_horizon_results.json"), "w") as f:
        json.dump(fk_horizon_out, f, indent=2)
    print(f"Saved: {os.path.join(os.path.dirname(__file__), 'qp_ablation_nonlinear_fk_horizon_results.json')}")

    # --- summary stats ---
    n_total = len(results)
    n_solved_both = len(solved_both)
    joint_deltas = np.array([r["joint_delta_rad"] for r in solved_both])
    insertion_deltas = np.array([r["insertion_delta_m"] for r in solved_both])
    n_nonzero_joint = int(np.sum(joint_deltas > 1e-9))
    n_nonzero_insertion = int(np.sum(insertion_deltas > 1e-9))

    # nonlinear-FK: max gap shrink (233 -> 274 should, if constraint active,
    # push the horizon trajectory further from the base at the tightest
    # stages) vs stage-0 of each.
    min_nonlinear_gap_233 = min(
        (min(e["production_233mm"]["nonlinear_fk_gap_mm_by_stage"], default=float("inf")) for e in fk_horizon_out),
        default=None,
    )
    min_nonlinear_gap_274 = min(
        (min(e["virtual_274mm"]["nonlinear_fk_gap_mm_by_stage"], default=float("inf")) for e in fk_horizon_out),
        default=None,
    )

    summary = dict(
        n_total_states_checked=n_total,
        n_runs=len(RUN_DIRS),
        n_solved_both_radii=n_solved_both,
        n_274mm_infeasible_or_non_solved=len(failures_274),
        failures_274_detail=failures_274,
        n_states_with_nonzero_joint_delta=n_nonzero_joint,
        n_states_with_nonzero_insertion_delta=n_nonzero_insertion,
        joint_delta_rad_stats=dict(
            mean=float(np.mean(joint_deltas)) if len(joint_deltas) else None,
            max=float(np.max(joint_deltas)) if len(joint_deltas) else None,
            p95=float(np.percentile(joint_deltas, 95)) if len(joint_deltas) else None,
        ),
        insertion_delta_m_stats=dict(
            mean=float(np.mean(insertion_deltas)) if len(insertion_deltas) else None,
            max=float(np.max(insertion_deltas)) if len(insertion_deltas) else None,
            p95=float(np.percentile(insertion_deltas, 95)) if len(insertion_deltas) else None,
        ),
        n_fk_horizon_states_checked=len(fk_horizon_out),
        min_nonlinear_fk_gap_mm_233=min_nonlinear_gap_233,
        min_nonlinear_fk_gap_mm_274=min_nonlinear_gap_274,
        elapsed_s=elapsed,
    )
    with open(OUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_SUMMARY_PATH}")
    print(json.dumps(summary, indent=2)[:3000])


if __name__ == "__main__":
    _main()
