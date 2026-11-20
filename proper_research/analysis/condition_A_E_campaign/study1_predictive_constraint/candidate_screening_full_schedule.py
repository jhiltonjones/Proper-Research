"""Constraint-study hardware proposal, candidate screening.

Predefined candidate set and selection rule (stated BEFORE running this
script, see CANDIDATE_SET_MM / SELECTION_RULE below; reproduced in
CONSTRAINT_STUDY_HARDWARE_PROPOSAL.md):

  CANDIDATE_SET_MM = [274, 276, 278, 280]
  SELECTION_RULE: among candidates with ZERO infeasible solves across all
  214 of the plan's own NOMINAL schedule samples (full-schedule feasible
  task continuation), select the one with the LONGEST run of CONSECUTIVE
  reference indices showing a nonzero joint-channel predicted-command
  difference (production 233mm vs the candidate), using the plan's own
  nominal trajectory as the probed state at each reference index (not
  historical logged states -- this checks the WHOLE path, not just where
  real runs happened to pass close to the beam base).

This rule is fixed before results are inspected. INV is not run or
consulted anywhere in this script -- the selection depends only on MPC
QP feasibility and command-difference coverage.

Offline only: real worker (spawn_and_warm_worker), real schedule, no
hardware. This is replay against the plan's own nominal trajectory, not
proof of future closed-loop behavior under any candidate.
"""
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
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.planning.planning_context import make_robot_config

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
PRODUCTION_RADIUS_MM = common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + common.MAGNET_EXCLUSION_ROBUST_MARGIN_MM  # 233.0
CANDIDATE_SET_MM = [274.0, 276.0, 278.0, 280.0]

OUT_DIR = os.path.dirname(__file__)


def _spawn(radius_m, jacobians, positions, beam_base_xyz_r):
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
    return w


def _longest_consecutive_run(flags):
    best = cur = 0
    best_range = cur_start = None
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > best:
                best = cur
                best_range = (cur_start, i)
        else:
            cur = 0
    return best, best_range


def _main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(3)
    n_ref = reference.state.shape[0]
    print(f"[screening] {n_ref} nominal reference samples; production radius {PRODUCTION_RADIUS_MM}mm; "
          f"candidates {CANDIDATE_SET_MM}mm")

    w_prod = _spawn(PRODUCTION_RADIUS_MM / 1e3, jacobians, positions, beam_base_xyz_r)

    per_candidate = {}
    try:
        for radius_mm in CANDIDATE_SET_MM:
            w_cand = _spawn(radius_mm / 1e3, jacobians, positions, beam_base_xyz_r)
            t0 = time.time()
            rows = []
            n_infeasible = 0
            try:
                for i in range(n_ref):
                    q_cmd = reference.state[i, :6].astype(float)
                    insertion_cmd = float(reference.state[i, 6])
                    q_cmd_km1 = reference.state[max(i - 1, 0), :6].astype(float)
                    tip = reference.desired_position_m[i].astype(float)
                    z_meas = np.concatenate([q_cmd, [insertion_cmd]])
                    u_prev = np.zeros(7)

                    r_prod = w_prod.try_solve(
                        kind="nullspace_r700", z_meas=z_meas, q_cmd=q_cmd, q_cmd_prev=q_cmd_km1,
                        insertion_cmd=insertion_cmd, measured_beam_position=tip,
                        control_index=i, previous_input=u_prev, deadline_s=2.0,
                    )
                    r_cand = w_cand.try_solve(
                        kind="nullspace_r700", z_meas=z_meas, q_cmd=q_cmd, q_cmd_prev=q_cmd_km1,
                        insertion_cmd=insertion_cmd, measured_beam_position=tip,
                        control_index=i, previous_input=u_prev, deadline_s=2.0,
                    )
                    status_cand = r_cand.get("qp_status")
                    if status_cand != "solved":
                        n_infeasible += 1
                    joint_delta = insertion_delta = None
                    if r_prod.get("predicted_commands") is not None and r_cand.get("predicted_commands") is not None:
                        u0_prod = np.asarray(r_prod["predicted_commands"], dtype=float)[0]
                        u0_cand = np.asarray(r_cand["predicted_commands"], dtype=float)[0]
                        joint_delta = float(np.linalg.norm(u0_prod[:6] - u0_cand[:6]))
                        insertion_delta = float(abs(u0_prod[6] - u0_cand[6]))
                    rows.append(dict(
                        ref_index=i,
                        qp_status_production=r_prod.get("qp_status"),
                        qp_status_candidate=status_cand,
                        joint_delta_rad=joint_delta,
                        insertion_delta_m=insertion_delta,
                    ))
            finally:
                w_cand.close()
            elapsed = time.time() - t0

            nonzero_joint_flags = [bool(r["joint_delta_rad"] and r["joint_delta_rad"] > 1e-9) for r in rows]
            longest_run, longest_range = _longest_consecutive_run(nonzero_joint_flags)
            n_nonzero_joint = sum(nonzero_joint_flags)
            n_nonzero_insertion = sum(1 for r in rows if r["insertion_delta_m"] and r["insertion_delta_m"] > 1e-9)

            per_candidate[radius_mm] = dict(
                radius_mm=radius_mm,
                n_ref_samples=n_ref,
                n_infeasible=n_infeasible,
                full_schedule_feasible=(n_infeasible == 0),
                n_states_nonzero_joint_delta=n_nonzero_joint,
                n_states_nonzero_insertion_delta=n_nonzero_insertion,
                longest_consecutive_nonzero_joint_run=longest_run,
                longest_consecutive_nonzero_joint_range=longest_range,
                elapsed_s=elapsed,
                rows=rows,
            )
            print(f"[screening] radius={radius_mm}mm: infeasible={n_infeasible}/{n_ref}, "
                  f"nonzero-joint-states={n_nonzero_joint}/{n_ref}, "
                  f"longest consecutive run={longest_run} (range={longest_range}), {elapsed:.1f}s")
    finally:
        w_prod.close()

    with open(os.path.join(OUT_DIR, "candidate_screening_full_schedule_results.json"), "w") as f:
        json.dump(per_candidate, f)

    # Apply the pre-stated selection rule.
    feasible_candidates = {r: d for r, d in per_candidate.items() if d["full_schedule_feasible"]}
    if not feasible_candidates:
        selected = None
        print("[screening] SELECTION: no candidate was fully feasible across the whole schedule")
    else:
        selected = max(feasible_candidates.items(), key=lambda kv: kv[1]["longest_consecutive_nonzero_joint_run"])
        print(f"[screening] SELECTION: {selected[0]}mm "
              f"(longest consecutive run={selected[1]['longest_consecutive_nonzero_joint_run']}, "
              f"range={selected[1]['longest_consecutive_nonzero_joint_range']})")

    summary = dict(
        candidate_set_mm=CANDIDATE_SET_MM,
        production_radius_mm=PRODUCTION_RADIUS_MM,
        n_ref_samples=n_ref,
        per_candidate_summary={
            r: {k: v for k, v in d.items() if k != "rows"} for r, d in per_candidate.items()
        },
        selected_radius_mm=(selected[0] if selected else None),
    )
    with open(os.path.join(OUT_DIR, "candidate_screening_full_schedule_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: candidate_screening_full_schedule_results.json / _summary.json")


if __name__ == "__main__":
    _main()
