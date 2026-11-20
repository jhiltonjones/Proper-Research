import sys
sys.path.insert(0, "/home/jack/Proper-Research")
import os
os.chdir("/home/jack/Proper-Research")
import json
import numpy as np

from proper_research.hardware import ur_magnet_ik_jacobian_validation as urik
from proper_research.simulation.simulations.controller_factory_joint_space import _resolve_robot_kinematics
from proper_research.planning.planning_context import make_robot_config
from proper_research.hardware.online.rectangle_stage_a import common

robot_kin = _resolve_robot_kinematics(make_robot_config())
beam_base_xyz_r = common._calibrated_default_initial_poses_for_guard().reshape(3)
print("beam_base_xyz_r:", beam_base_xyz_r)

d = "close_loop_logs/baseline_A_pilot_2026-11-20_retry1_20261120T122825Z"
rows = [json.loads(l) for l in open(f"{d}/predicted_beam_positions.jsonl")]

for idx in (109, 110, 111, 112, 113):
    r = rows[idx]
    q = np.asarray(r["q_cmd_k"], dtype=float)
    fk = urik.forward_kinematics(q, robot_kin.dh, robot_kin.T_F_M)
    magnet_xyz_fk = np.asarray(fk.T_R_target[:3, 3], dtype=float)
    magnet_xyz_logged = np.asarray(r["magnet_xyz_m"], dtype=float)
    gap_fk_mm = float(np.linalg.norm(magnet_xyz_fk - beam_base_xyz_r)) * 1e3
    gap_logged_mm = r["magnet_exclusion_gap_mm"]
    diff_mm = float(np.linalg.norm(magnet_xyz_fk - magnet_xyz_logged)) * 1e3
    print(f"step={r['step']} fk_pos={magnet_xyz_fk.round(5).tolist()} "
          f"logged_pos={magnet_xyz_logged.round(5).tolist()} pos_diff_mm={diff_mm:.4f} "
          f"gap_fk_mm={gap_fk_mm:.3f} gap_logged_mm={gap_logged_mm:.3f} "
          f"gap_diff_mm={abs(gap_fk_mm-gap_logged_mm):.4f}")
