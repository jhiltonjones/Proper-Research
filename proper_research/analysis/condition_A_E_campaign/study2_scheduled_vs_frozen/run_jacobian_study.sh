#!/usr/bin/env bash
# Jacobian study: concrete hardware run script.
# Implements the FROZEN design in JACOBIAN_STUDY_HARDWARE_PROPOSAL.md
# exactly: 4 scheduled-Jacobian MPC repetitions, 4 frozen-Jacobian MPC
# repetitions (same mechanism as condition B: --freeze-jacobian
# --freeze-jacobian-index 0), adaptation off throughout, alternating
# order across repetition pairs so neither condition is always first.
#
# This script performs REAL HARDWARE MOTION when run -- it is not
# executed by the agent. Run it yourself, one repetition at a time, after
# reviewing each run's own printed settings.
#
# Usage:
#   bash run_jacobian_study.sh 1   # repetition 1: scheduled, then frozen
#   bash run_jacobian_study.sh 2   # repetition 2: frozen, then scheduled
#   bash run_jacobian_study.sh 3   # repetition 3: scheduled, then frozen
#   bash run_jacobian_study.sh 4   # repetition 4: frozen, then scheduled
# Run one repetition at a time; review both runs' output before starting
# the next repetition. Do not change the per-run settings below after
# seeing results -- this design is frozen.

set -euo pipefail
cd /home/jack/Proper-Research

PLAN_DIR="plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED="plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_scheduled () {
  local rep="$1"
  python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
    --plan-dir "$PLAN_DIR" --schedule-cache "$SCHED" \
    --out-dir close_loop_logs --run-name "jacobian_study_rep${rep}_scheduled_${STAMP}" \
    --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
    --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
    --measured-clearance-abort-mm 230.0 \
    --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
    --protective-config-reviewed
}

run_frozen () {
  local rep="$1"
  python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
    --plan-dir "$PLAN_DIR" --schedule-cache "$SCHED" \
    --out-dir close_loop_logs --run-name "jacobian_study_rep${rep}_frozen_${STAMP}" \
    --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
    --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
    --measured-clearance-abort-mm 230.0 \
    --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
    --freeze-jacobian --freeze-jacobian-index 0 \
    --protective-config-reviewed
}

rep="${1:?Usage: bash run_jacobian_study.sh <1|2|3|4>}"

case "$rep" in
  1)
    echo "=== Repetition 1: scheduled -> frozen ==="
    run_scheduled 1
    run_frozen 1
    ;;
  2)
    echo "=== Repetition 2: frozen -> scheduled ==="
    run_frozen 2
    run_scheduled 2
    ;;
  3)
    echo "=== Repetition 3: scheduled -> frozen ==="
    run_scheduled 3
    run_frozen 3
    ;;
  4)
    echo "=== Repetition 4: frozen -> scheduled ==="
    run_frozen 4
    run_scheduled 4
    ;;
  *)
    echo "repetition must be 1, 2, 3, or 4 (the four-pair balanced design is complete at 4)" >&2
    exit 1
    ;;
esac
