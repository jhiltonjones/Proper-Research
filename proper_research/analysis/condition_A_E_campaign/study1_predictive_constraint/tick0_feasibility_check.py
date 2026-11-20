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


def _main():
    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(1, 3)

    d = "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z"
    rows = [json.loads(l) for l in open(f"{d}/predicted_beam_positions.jsonl")]
    r0 = rows[0]

    worker = spawn_and_warm_worker(
        plan_dir=PLAN_DIR, schedule_cache=SCHED, control_hz=common.CONTROL_HZ,
        prediction_horizon=15, joint_velocity_limit_rad_s=common.JOINT_VELOCITY_LIMIT_RAD_S,
        insertion_rate_limit_m_s=0.002, joint_acceleration_limit_rad_s2=common.JOINT_ACCELERATION_LIMIT_RAD_S2,
        position_error_scale_mm=0.5, position_tracking_weight=1.0,
        magnet_exclusion_kwargs=dict(
            position_jacobians=jacobians, nominal_positions_m=positions,
            lumen_C_m=beam_base_xyz_r, radius_m=0.274,
        ),
        insertion_command_min_m=0.020, insertion_command_max_m=0.070,
    )
    try:
        worker.set_frame_transform(np.eye(3), np.zeros(3))
        resp = worker.try_solve(
            kind="nullspace_r700",
            z_meas=np.asarray(r0["z_meas"], dtype=float),
            q_cmd=np.asarray(r0["q_cmd_k"], dtype=float), q_cmd_prev=np.asarray(r0["q_cmd_km1"], dtype=float),
            insertion_cmd=float(r0["insertion_cmd_m"]),
            measured_beam_position=np.asarray(r0["measured_beam_position_m"], dtype=float),
            control_index=0, previous_input=np.asarray(r0["u_prev"], dtype=float),
            deadline_s=2.0,
        )
        print(f"tick0 (start, logged_gap_mm={r0['magnet_exclusion_gap_mm']:.2f}) at virtual radius 274mm: "
              f"qp_status={resp.get('qp_status')}")
    finally:
        worker.close()


if __name__ == "__main__":
    _main()
