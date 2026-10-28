"""Builds the (S,3,7) NO-CONTACT-model Jacobian schedule for the floor240
offline plan (2026-10-12). See build_c_schedule_floor240.py's docstring --
same approach, contact=False and pointed at the no-contact plan's own
time-parameterized reference.
"""
import json
import os
import time

import numpy as np

from proper_research.controllers import mpc_variants
from proper_research.controllers.beam_jacobian_providers import from_model_bundle
from proper_research.planning.vessel_context import build_vessel_planning_context
from proper_research.simulation.simulations.simulate_time_parameterized_configuration_mpc import (
    load_configuration_reference,
)
from proper_research.rig_calibration import beam_base_pose6

PLAN_DIR = "plans/vessel_phi30_L30_left1mm_newwall_tol0p5_floor240_nocontact_2026-10-12/time_parameterized_configuration_path"
LUMEN_FILE = "vessel_lumen_robot_frame_2026-10-06_zcorrected.json"
START_POSITION_JSON = "vessel_magnet_initial_position_2026-10-12_floor240.json"
INSERTION_MAX_MM = 110.0
INSERTION_START_MM = 30.44
DT = 0.01
OUT = "plans/stage3_design/mpc_schedules/vessel_nc_schedule_phi30_L30_newwall_floor240_2026-10-12.npy"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(START_POSITION_JSON) as f:
    start_ref = json.load(f)
start_point = np.array(start_ref["magnet_pose6_R"], dtype=float)
pivot_point = beam_base_pose6()
initial_poses = (pivot_point, start_point, INSERTION_START_MM / 1000.0, DT)

t0 = time.time()
reference = load_configuration_reference(PLAN_DIR, require_planned_beam_feasible=False)
print(f"[build-nc-schedule] reference loaded ({time.time()-t0:.1f}s), {reference.state.shape[0]} samples")

_, bundle, controller_pack, _, _, _, _ = build_vessel_planning_context(
    lumen_file=LUMEN_FILE, insertion_max_m=INSERTION_MAX_MM * 1.0e-3,
    initial_poses=initial_poses,
)
print(f"[build-nc-schedule] bundle built ({time.time()-t0:.1f}s)")

jac_provider = from_model_bundle(
    bundle=bundle, controller_pack=controller_pack, contact=False, jacobian_mode="accurate",
)
print(f"[build-nc-schedule] jacobian provider ready, starting precompute ({time.time()-t0:.1f}s)")

schedule = mpc_variants.precompute_schedule(
    reference=reference, jacobian_provider=jac_provider, allow_undeclared_jacobian=True,
    progress_every=25,
)
np.save("/tmp/vessel_nc_schedule_floor240_BACKUP.npy", schedule)
np.save(OUT, schedule)
print(f"[build-nc-schedule] DONE: {schedule.shape} saved -> {OUT} (total {time.time()-t0:.1f}s)")
