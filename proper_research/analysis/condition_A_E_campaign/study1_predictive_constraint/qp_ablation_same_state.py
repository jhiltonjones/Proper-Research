"""Study 1 offline screening, part 2: real same-state QP ablation.
Calls the REAL worker's try_solve() twice from each of several real
logged states (no hardware), once with the production 233mm in-QP
magnet-exclusion radius, once with a candidate tighter "virtual" radius
added on top -- to see whether predicted_commands actually differ.
This is mechanistic offline evidence, not a simulated hardware outcome.
"""
import sys
sys.path.insert(0, "/home/jack/Proper-Research")
import os
os.chdir("/home/jack/Proper-Research")
import json

import numpy as np

from proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor import (
    spawn_and_warm_worker, build_magnet_exclusion_schedule, load_configuration_reference,
)
from proper_research.hardware.online.rectangle_stage_a import common

PLAN_DIR = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED = "plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
PRODUCTION_RADIUS_M = (common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + 3.0) / 1e3  # 233mm
CANDIDATE_VIRTUAL_RADIUS_M = 274.0 / 1e3  # see STUDY1_PROPOSAL.md for justification -- 260/265/270mm were screened first and produced zero command difference (constraint not yet binding); 274mm is the smallest round number above the campaign-wide observed minimum gap (272.98mm) that still leaves every probed real state qp_status==solved.


def _main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(1, 3)

    d = "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z"
    rows = [json.loads(l) for l in open(f"{d}/predicted_beam_positions.jsonl")]
    probe_steps = [109, 110, 111, 112, 113]

    results = []
    for radius_label, radius_m in (("production_233mm", PRODUCTION_RADIUS_M),
                                    ("virtual_274mm", CANDIDATE_VIRTUAL_RADIUS_M)):
        magnet_exclusion_kwargs = dict(
            position_jacobians=jacobians, nominal_positions_m=positions,
            lumen_C_m=beam_base_xyz_r, radius_m=radius_m,
        )
        worker = spawn_and_warm_worker(
            plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
            prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
            insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
            position_error_scale_mm=0.5, position_tracking_weight=1.0,
            magnet_exclusion_kwargs=magnet_exclusion_kwargs,
            insertion_command_min_m=0.020, insertion_command_max_m=0.070,
        )
        try:
            worker.set_frame_transform(np.eye(3), np.zeros(3))
            for step_idx in probe_steps:
                r = rows[step_idx]
                q_cmd = np.asarray(r["q_cmd_k"], dtype=float)
                z_meas = np.asarray(r["z_meas"], dtype=float)
                resp = worker.try_solve(
                    kind="nullspace_r700",
                    z_meas=z_meas,
                    q_cmd=q_cmd, q_cmd_prev=np.asarray(r["q_cmd_km1"], dtype=float),
                    insertion_cmd=float(r["insertion_cmd_m"]),
                    measured_beam_position=np.asarray(r["measured_beam_position_m"], dtype=float),
                    control_index=r["ref_index"], previous_input=np.asarray(r["u_prev"], dtype=float),
                    deadline_s=2.0,
                )
                results.append(dict(
                    radius_label=radius_label, radius_m=radius_m, step=r["step"],
                    ref_index=r["ref_index"], qp_status=resp.get("qp_status"),
                    predicted_commands=(
                        np.asarray(resp["predicted_commands"], dtype=float).tolist()
                        if resp.get("predicted_commands") is not None else None
                    ),
                    logged_gap_mm=r["magnet_exclusion_gap_mm"],
                ))
                print(f"[{radius_label}] step={r['step']} logged_gap_mm={r['magnet_exclusion_gap_mm']:.2f} "
                      f"qp_status={resp.get('qp_status')}")
        finally:
            worker.close()

    out_path = "/home/jack/.claude/jobs/043c0b8b/tmp/study1_qp_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    # Compare predicted_commands between the two radii, per probed step.
    by_step = {}
    for r in results:
        by_step.setdefault(r["step"], {})[r["radius_label"]] = r
    print("\n--- command comparison (stage-0 predicted_commands, first 7 values) ---")
    for step, pair in sorted(by_step.items()):
        a = pair.get("production_233mm", {}).get("predicted_commands")
        b = pair.get("virtual_274mm", {}).get("predicted_commands")
        if a is None or b is None:
            print(f"step={step}: one or both solves failed to produce predicted_commands "
                  f"(production qp_status={pair.get('production_233mm',{}).get('qp_status')}, "
                  f"virtual qp_status={pair.get('virtual_260mm',{}).get('qp_status')})")
            continue
        a0 = np.asarray(a[0], dtype=float)
        b0 = np.asarray(b[0], dtype=float)
        diff = np.linalg.norm(a0 - b0)
        print(f"step={step}: ||u0_production - u0_virtual|| = {diff:.6e}  "
              f"(production qp_status={pair['production_233mm']['qp_status']}, "
              f"virtual qp_status={pair['virtual_260mm']['qp_status']})")


if __name__ == "__main__":
    _main()
