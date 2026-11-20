#!/usr/bin/env bash
# Constraint study: concrete hardware run script.
# Implements the FROZEN design in CONSTRAINT_STUDY_HARDWARE_PROPOSAL.md
# exactly: three conditions (constrained MPC @ 276mm, unconstrained MPC
# [= condition A], INV), three blocks, each a different permutation of a
# 3x3 Latin square so no condition is always first/last (the A-E
# campaign's own documented confound).
#
# --experimental-magnet-exclusion-radius-mm is implemented, additive,
# default-unchanged, and offline-verified this session (6/6 new+existing
# e2e smoke tests passing, full 165-test production regression suite
# unaffected). This script performs REAL HARDWARE MOTION when run -- it
# is not executed by the agent. Run it yourself, block by block, after
# reviewing each run's own printed settings and confirming the robot is
# ready, same as every other hardware launch this session.
#
# Usage:
#   bash run_constraint_study.sh 1   # run block 1 (3 runs)
#   bash run_constraint_study.sh 2   # run block 2 (3 runs)
#   bash run_constraint_study.sh 3   # run block 3 (3 runs)
# Run one block at a time; review each run's own output before starting
# the next. Do not change the per-run settings below after seeing results
# -- this design is frozen.

set -euo pipefail
cd /home/jack/Proper-Research

PLAN_DIR="plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/time_parameterized_configuration_path"
SCHED="plans/rectangle_10x15mm_skipglobal_L30to50mm_2026-11-19_noDipole/jacobian_schedule_insertion_anchor.npy"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_constrained_mpc () {
  local block="$1"
  python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
    --plan-dir "$PLAN_DIR" --schedule-cache "$SCHED" \
    --out-dir close_loop_logs --run-name "constraint_study_block${block}_constrained_mpc_${STAMP}" \
    --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
    --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
    --measured-clearance-abort-mm 230.0 \
    --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
    --experimental-magnet-exclusion-radius-mm 276.0 \
    --protective-config-reviewed
}

run_unconstrained_mpc () {
  local block="$1"
  python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
    --plan-dir "$PLAN_DIR" --schedule-cache "$SCHED" \
    --out-dir close_loop_logs --run-name "constraint_study_block${block}_unconstrained_mpc_${STAMP}" \
    --horizon 15 --jacobian-adaptation-mode off --input-penalty planned --dipole-alignment-weight 0.0 \
    --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
    --measured-clearance-abort-mm 230.0 \
    --beam-base-exclusion-floor-mm 230.0 --magnet-exclusion-robust-margin-mm 3.0 \
    --protective-config-reviewed
}

run_inv () {
  local block="$1"
  python3 -m proper_research.hardware.online.rectangle_stage_a.run_inv_7dof_delay_aware \
    --plan-dir "$PLAN_DIR" --schedule-cache "$SCHED" \
    --out-dir close_loop_logs --run-name "constraint_study_block${block}_inv_${STAMP}" \
    --insertion-command-min-mm 20.0 --insertion-command-max-mm 70.0 \
    --measured-clearance-abort-mm 230.0 \
    --protective-config-reviewed
}

block="${1:?Usage: bash run_constraint_study.sh <1|2|3>}"

case "$block" in
  1)
    echo "=== Block 1: constrained-MPC -> unconstrained-MPC -> INV ==="
    run_constrained_mpc 1
    run_unconstrained_mpc 1
    run_inv 1
    ;;
  2)
    echo "=== Block 2: unconstrained-MPC -> INV -> constrained-MPC ==="
    run_unconstrained_mpc 2
    run_inv 2
    run_constrained_mpc 2
    ;;
  3)
    echo "=== Block 3: INV -> constrained-MPC -> unconstrained-MPC ==="
    run_inv 3
    run_constrained_mpc 3
    run_unconstrained_mpc 3
    ;;
  *)
    echo "block must be 1, 2, or 3 (one 3x3 Latin square -- repeat 1-3 again for blocks 4-6 if more are later wanted)" >&2
    exit 1
    ;;
esac
