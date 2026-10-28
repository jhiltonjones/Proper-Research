#!/usr/bin/env bash
# Walkthrough: the three-way Jacobian-schedule MPC comparison on the vessel
# plan (contact-aware scheduled, no-contact scheduled, frozen-at-start).
#
# This is a REFERENCE script, not something to blindly `bash` end to end --
# read each block, run it, watch the live output, confirm the run finished
# the way you expect (check `stop_reason` / `solver_status` in
# path_follow.jsonl, see HOWTO_CLOSED_LOOP_MPC.md section 3), THEN move to
# the next block. Insertion drifts between runs (the advancer has no
# encoder), so every block resets it first -- do not skip that step even if
# you just ran one.
#
# Prerequisites (see HOWTO_CLOSED_LOOP_MPC.md section 1):
#   - robot in Remote Control mode on the teach pendant
#   - camera connected, vessel/markers in view
#   - the plan's own open-loop check has already passed
#
# Edit these four variables for your own plan/vessel before running anything:
set -euo pipefail
cd /home/jack/Proper-Research

PLAN_DIR="plans/vessel_live_trimmed6mm_2026-09-28/time_parameterized_configuration_path"
LUMEN_FILE="vessel_lumen_robot_frame_raised3cm_trimmed6mm_2026-09-28.json"
MAGNET_EXCLUSION_REF="vessel_magnet_exclusion_reference_joints_2026-09-24.json"
INSERTION_MAX_MM=65
OUT_DIR="close_loop_logs/myrun"

# L0 read from the plan itself -- do not hardcode this by hand, it must
# match the plan's own saved start state or the insertion-tolerance
# preflight check will (correctly) refuse to proceed.
L0_MM=$(python3 -c "
from proper_research.hardware.online.vessel_stage_a import common
q0, l0 = common.load_plan_initial_state('${PLAN_DIR}')
print(f'{l0*1000:.4f}')
")
echo "[experiments] plan L0 = ${L0_MM}mm"

reset_insertion() {
    # The advancer has no encoder -- insertion never moves unless
    # explicitly commanded, and it drifts between runs. Call this before
    # EVERY closed-loop run below, not just the first.
    python3 -c "
from proper_research.hardware.online.vessel_stage_a.checkpoint_beam_shape_campaign import (
    reset_insertion_to_target,
)
from pathlib import Path
reset_insertion_to_target(${L0_MM}, live=True, out_dir=Path('/tmp'))
"
}

# ---------------------------------------------------------------------------
# Run 1: scheduled Jacobian, CONTACT-AWARE model
# ---------------------------------------------------------------------------
# The schedule is a genuine from-model Jacobian at every reference sample
# (contact model) -- if the cache file doesn't exist yet this build takes
# ~90-150s (one real beam solve per sample); after that it's an instant
# load. Build it once, reuse the cache for every future contact run on this
# plan.
SCHEDULE_CONTACT="/tmp/vessel_live_trimmed6mm_schedule_contact.npy"

reset_insertion
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir "${PLAN_DIR}" \
    --out-dir "${OUT_DIR}" \
    --run-name vessel_live_trimmed6mm_mpc_contact \
    --schedule-cache "${SCHEDULE_CONTACT}" \
    --lumen-file "${LUMEN_FILE}" \
    --insertion-max-mm "${INSERTION_MAX_MM}" \
    --magnet-exclusion-reference-joints "${MAGNET_EXCLUSION_REF}" \
    --contact

# ---------------------------------------------------------------------------
# Run 2: scheduled Jacobian, NO-CONTACT model
# ---------------------------------------------------------------------------
# Same plan, same reference trajectory -- ONLY the Jacobian model differs
# (no-contact beam physics instead of contact-aware). This MUST use a
# DIFFERENT cache file from run 1: the contact and no-contact schedules are
# numerically different arrays, and reusing the same cache path would
# silently load the wrong one.
SCHEDULE_NOCONTACT="/tmp/vessel_live_trimmed6mm_schedule_nocontact.npy"

reset_insertion
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir "${PLAN_DIR}" \
    --out-dir "${OUT_DIR}" \
    --run-name vessel_live_trimmed6mm_mpc_nocontact \
    --schedule-cache "${SCHEDULE_NOCONTACT}" \
    --lumen-file "${LUMEN_FILE}" \
    --insertion-max-mm "${INSERTION_MAX_MM}" \
    --magnet-exclusion-reference-joints "${MAGNET_EXCLUSION_REF}" \
    --no-contact

# ---------------------------------------------------------------------------
# Run 3: FROZEN Jacobian (contact model, frozen at the start-of-path value)
# ---------------------------------------------------------------------------
# A "frozen" schedule repeats the FIRST row of the scheduled contact
# Jacobian across every horizon sample -- i.e. the controller keeps using
# the linearization from the very start of the path the whole way through,
# instead of re-linearizing as the arm moves. This needs run 1's contact
# schedule to already exist (it's built from it, not from scratch), so run
# block 1 before this one.
SCHEDULE_FROZEN="/tmp/vessel_live_trimmed6mm_schedule_frozen.npy"

if [[ ! -f "${SCHEDULE_FROZEN}" ]]; then
    python3 -c "
import numpy as np
sj = np.load('${SCHEDULE_CONTACT}')
fj = np.repeat(sj[0:1], sj.shape[0], axis=0)
np.save('${SCHEDULE_FROZEN}', fj)
print(f'[experiments] froze schedule: {sj.shape} -> {fj.shape}, all rows == row 0')
"
fi

reset_insertion
python -m proper_research.hardware.online.vessel_stage_a.run_mpc_delay_aware_vessel \
    --plan-dir "${PLAN_DIR}" \
    --out-dir "${OUT_DIR}" \
    --run-name vessel_live_trimmed6mm_mpc_frozen \
    --schedule-cache "${SCHEDULE_FROZEN}" \
    --lumen-file "${LUMEN_FILE}" \
    --insertion-max-mm "${INSERTION_MAX_MM}" \
    --magnet-exclusion-reference-joints "${MAGNET_EXCLUSION_REF}" \
    --contact
    # NOTE: --contact here only matters if SCHEDULE_FROZEN were a cache
    # miss (it won't be, since the block above always creates it first) --
    # the frozen .npy file itself, not this flag, is what actually
    # determines the Jacobian values used.

# ---------------------------------------------------------------------------
# After all three: compare
# ---------------------------------------------------------------------------
# Each run's directory under ${OUT_DIR} has its own controller_metadata.json
# / path_follow.jsonl / predicted_beam_positions.jsonl (see
# HOWTO_CLOSED_LOOP_MPC.md section 3). At minimum, compare final/max
# tracking error and stop_reason across the three run directories --
# `grep -l stop_reason ${OUT_DIR}/*/path_follow.jsonl` or read each run's
# console summary.
