"""Cross-check the full-schedule-selected candidate (280mm) against the
REAL 1995 historical logged states (not the idealized nominal trajectory)
-- real executed states deviate from nominal and may approach the beam
base more closely than the nominal path does, so full-schedule nominal
feasibility alone is not sufficient evidence; this checks whether the
candidate also holds up against real execution history.
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

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
PRODUCTION_RADIUS_MM = common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + common.MAGNET_EXCLUSION_ROBUST_MARGIN_MM
CANDIDATE_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
OUT_SUFFIX = str(CANDIDATE_MM).rstrip("0").rstrip(".")

RUN_DIRS = sorted(glob.glob("close_loop_logs/baseline_A_pilot_2026-11-20_retry1_*")) + \
           sorted(glob.glob("close_loop_logs/condition_B_frozen_jacobian_2026-11-20_*"))
RUN_DIRS = [d for d in RUN_DIRS if os.path.isdir(d)]
OUT_DIR = os.path.dirname(__file__)


def _main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(3)

    all_states = []
    for d in RUN_DIRS:
        p = f"{d}/predicted_beam_positions.jsonl"
        if not os.path.exists(p):
            continue
        for r in [json.loads(l) for l in open(p)]:
            all_states.append((d, r))
    print(f"[280mm historical check] {len(all_states)} real logged states")

    def _spawn(radius_mm):
        kw = dict(position_jacobians=jacobians, nominal_positions_m=positions,
                   lumen_C_m=beam_base_xyz_r.reshape(1, 3), radius_m=radius_mm / 1e3)
        w = spawn_and_warm_worker(
            plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
            prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
            insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
            position_error_scale_mm=0.5, position_tracking_weight=1.0,
            magnet_exclusion_kwargs=kw, insertion_command_min_m=0.020, insertion_command_max_m=0.070,
        )
        w.set_frame_transform(np.eye(3), np.zeros(3))
        return w

    w_prod = _spawn(PRODUCTION_RADIUS_MM)
    w_cand = _spawn(CANDIDATE_MM)

    t0 = time.time()
    results = []
    failures = []
    try:
        for i, (d, r) in enumerate(all_states):
            q_cmd = np.asarray(r["q_cmd_k"], dtype=float)
            z_meas = np.asarray(r["z_meas"], dtype=float)
            kwargs = dict(
                kind="nullspace_r700", z_meas=z_meas, q_cmd=q_cmd,
                q_cmd_prev=np.asarray(r["q_cmd_km1"], dtype=float),
                insertion_cmd=float(r["insertion_cmd_m"]),
                measured_beam_position=np.asarray(r["measured_beam_position_m"], dtype=float),
                control_index=r["ref_index"], previous_input=np.asarray(r["u_prev"], dtype=float),
                deadline_s=2.0,
            )
            r_prod = w_prod.try_solve(**kwargs)
            r_cand = w_cand.try_solve(**kwargs)
            status_cand = r_cand.get("qp_status")
            if status_cand != "solved":
                failures.append(dict(run_dir=d, step=r["step"], ref_index=r["ref_index"],
                                      qp_status=status_cand, logged_gap_mm=r["magnet_exclusion_gap_mm"]))
            joint_delta = insertion_delta = None
            if r_prod.get("predicted_commands") is not None and r_cand.get("predicted_commands") is not None:
                u0p = np.asarray(r_prod["predicted_commands"], dtype=float)[0]
                u0c = np.asarray(r_cand["predicted_commands"], dtype=float)[0]
                joint_delta = float(np.linalg.norm(u0p[:6] - u0c[:6]))
                insertion_delta = float(abs(u0p[6] - u0c[6]))
            results.append(dict(run_dir=d, step=r["step"], ref_index=r["ref_index"],
                                 logged_gap_mm=r["magnet_exclusion_gap_mm"],
                                 qp_status_production=r_prod.get("qp_status"), qp_status_candidate=status_cand,
                                 joint_delta_rad=joint_delta, insertion_delta_m=insertion_delta))
            if (i + 1) % 400 == 0:
                print(f"  {i+1}/{len(all_states)}, {time.time()-t0:.1f}s")
    finally:
        w_prod.close()
        w_cand.close()

    n_nonzero_joint = sum(1 for r in results if r["joint_delta_rad"] and r["joint_delta_rad"] > 1e-9)
    n_nonzero_insertion = sum(1 for r in results if r["insertion_delta_m"] and r["insertion_delta_m"] > 1e-9)
    summary = dict(
        candidate_mm=CANDIDATE_MM, n_states=len(results), n_failures=len(failures),
        failures=failures, n_nonzero_joint=n_nonzero_joint, n_nonzero_insertion=n_nonzero_insertion,
        joint_delta_max=max((r["joint_delta_rad"] or 0.0) for r in results),
        insertion_delta_max=max((r["insertion_delta_m"] or 0.0) for r in results),
        min_logged_gap_mm=min(r["logged_gap_mm"] for r in results),
        elapsed_s=time.time() - t0,
    )
    with open(os.path.join(OUT_DIR, f"candidate_{OUT_SUFFIX}mm_historical_check_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUT_DIR, f"candidate_{OUT_SUFFIX}mm_historical_check_results.json"), "w") as f:
        json.dump(results, f)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    _main()
