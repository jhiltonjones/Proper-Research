"""Phase 0 (plan: snug-bouncing-giraffe.md): shared corrected model-bundle
helper, built EXACTLY like build_c_schedule_floor240.py (same
initial_poses, same LUMEN_FILE, same from_model_bundle+jacobian_mode=
"accurate" call). No top-level execution -- safe to import from any
script without re-running anything.
"""
import json

import numpy as np

from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.rig_calibration import beam_base_pose6, CURRENT_LUMEN_FILE

PLAN_DIR = "/home/jack/Proper-Research/plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_2026-10-12/time_parameterized_configuration_path"
START_POSITION_JSON = "/home/jack/Proper-Research/vessel_magnet_initial_position_2026-10-12_floor240.json"
SCHED_C = "/home/jack/Proper-Research/plans/stage3_design/mpc_schedules/vessel_c_schedule_phi30_L30_newwall_floor240_2026-10-12.npy"
INSERTION_MAX_MM = 110.0
INSERTION_START_MM = 30.44
INIT_DT = 0.01


def build_corrected_jacobian_provider():
    with open(START_POSITION_JSON) as f:
        start_ref = json.load(f)
    start_point = np.array(start_ref["magnet_pose6_R"], dtype=float)
    pivot_point = beam_base_pose6()
    initial_poses = (pivot_point, start_point, INSERTION_START_MM / 1000.0, INIT_DT)

    _, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
        lumen_file=CURRENT_LUMEN_FILE, insertion_max_m=INSERTION_MAX_MM * 1.0e-3,
        initial_poses=initial_poses,
    )
    jac_provider = from_model_bundle(
        bundle=bundle, controller_pack=controller_pack, contact=True, jacobian_mode="accurate",
    )
    return jac_provider
