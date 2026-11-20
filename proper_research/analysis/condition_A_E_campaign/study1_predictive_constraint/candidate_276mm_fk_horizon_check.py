"""Nonlinear FK clearance (real urik.forward_kinematics, not the QP's own
linearized model) along the full predicted command horizon, for the 276mm
selected candidate, at every real logged state in its effect window
(ref_index 105-112 across the runs where it was found to produce a
nonzero joint-channel command difference)."""
import json
import os
import sys

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
PRODUCTION_RADIUS_MM = common.MAGNET_BEAM_BASE_REVIEWED_BOUNDARY_MM + common.MAGNET_EXCLUSION_ROBUST_MARGIN_MM
CANDIDATE_MM = 276.0
OUT_DIR = os.path.dirname(__file__)


def _main():
    results = json.load(open(os.path.join(OUT_DIR, "candidate_276mm_historical_check_results.json")))
    tight = [r for r in results if r["joint_delta_rad"] and r["joint_delta_rad"] > 1e-9]
    print(f"[FK horizon] {len(tight)} states in the 276mm effect window")

    reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
    jacobians, positions = build_magnet_exclusion_schedule(reference)
    beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(3)
    robot_kin = _resolve_robot_kinematics(make_robot_config())

    # Rebuild the raw states dict (run_dir, step) -> raw predicted_beam_positions row
    import glob
    run_dirs = sorted(set(r["run_dir"] for r in tight))
    raw_by_key = {}
    for d in run_dirs:
        for r in [json.loads(l) for l in open(f"{d}/predicted_beam_positions.jsonl")]:
            raw_by_key[(d, r["step"])] = r

    workers = {}
    for label, radius_mm in (("production_233mm", PRODUCTION_RADIUS_MM), ("virtual_276mm", CANDIDATE_MM)):
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
        workers[label] = w

    out = []
    try:
        for r in tight:
            raw = raw_by_key[(r["run_dir"], r["step"])]
            q_cmd = np.asarray(raw["q_cmd_k"], dtype=float)
            z_meas = np.asarray(raw["z_meas"], dtype=float)
            entry = dict(run_dir=r["run_dir"], step=r["step"], logged_gap_mm=r["logged_gap_mm"])
            for label, w in workers.items():
                resp = w.try_solve(
                    kind="nullspace_r700", z_meas=z_meas, q_cmd=q_cmd,
                    q_cmd_prev=np.asarray(raw["q_cmd_km1"], dtype=float),
                    insertion_cmd=float(raw["insertion_cmd_m"]),
                    measured_beam_position=np.asarray(raw["measured_beam_position_m"], dtype=float),
                    control_index=raw["ref_index"], previous_input=np.asarray(raw["u_prev"], dtype=float),
                    deadline_s=2.0,
                )
                pc = resp.get("predicted_commands")
                gaps = []
                if pc is not None:
                    pc = np.asarray(pc, dtype=float)
                    for stage in range(pc.shape[0]):
                        fk = urik.forward_kinematics(pc[stage, :6], robot_kin.dh, robot_kin.T_F_M)
                        magnet_xyz = np.asarray(fk.T_R_target[:3, 3], dtype=float)
                        gaps.append(float(np.linalg.norm(magnet_xyz - beam_base_xyz_r)) * 1e3)
                entry[label] = dict(qp_status=resp.get("qp_status"), nonlinear_fk_gap_mm_by_stage=gaps)
            out.append(entry)
    finally:
        for w in workers.values():
            w.close()

    min_233 = min(min(e["production_233mm"]["nonlinear_fk_gap_mm_by_stage"]) for e in out if e["production_233mm"]["nonlinear_fk_gap_mm_by_stage"])
    min_276 = min(min(e["virtual_276mm"]["nonlinear_fk_gap_mm_by_stage"]) for e in out if e["virtual_276mm"]["nonlinear_fk_gap_mm_by_stage"])
    summary = dict(n_states=len(out), min_nonlinear_fk_gap_mm_233=min_233, min_nonlinear_fk_gap_mm_276=min_276)
    with open(os.path.join(OUT_DIR, "candidate_276mm_fk_horizon_results.json"), "w") as f:
        json.dump(out, f)
    with open(os.path.join(OUT_DIR, "candidate_276mm_fk_horizon_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
