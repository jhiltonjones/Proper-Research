#!/bin/bash
# Runs the full 2x2 {SJ,FJ} x {MPC,INV-7} closed-loop grid, N reps per cell,
# resetting insertion before every single rep. Continues past a failed/
# aborted rep (divergence is an expected, informative outcome for the FJ
# cells) instead of stopping the whole grid -- every rep's outcome is
# printed in the final summary so nothing is silently skipped.
#
# Usage:
#   ./run_2x2_grid.sh <plan_dir> <sj_schedule> <fj_schedule> <out_dir> \
#       <run_name_prefix> <n_reps> <zraise_mm> [insertion_tol_mm]
#
# Example (U-shape, +30mm):
#   ./run_2x2_grid.sh \
#     plans/ushape_10x15mm_skipglobal_zraise30mm_2026-09-24/time_parameterized_configuration_path \
#     /tmp/ushape_zraise30mm_schedule_SJ.npy /tmp/ushape_zraise30mm_schedule_FJ.npy \
#     close_loop_logs/myrun ushape_zraise30mm 4 30 0.3
set -u
cd /home/jack/Proper-Research

PLAN_DIR="$1"
SJ_SCHEDULE="$2"
FJ_SCHEDULE="$3"
OUT_DIR="$4"
RUN_NAME_PREFIX="$5"
N_REPS="$6"
ZRAISE_MM="$7"
INSERTION_TOL_MM="${8:-0.3}"

declare -a OUTCOMES

for CONTROLLER in mpc inv7; do
  for JACOBIAN in sj fj; do
    for i in $(seq 1 "$N_REPS"); do
      CELL="${CONTROLLER}_${JACOBIAN}"
      echo "=== [$CELL] rep $i/$N_REPS: resetting insertion ==="
      python3 -m proper_research.hardware.online.advancer_excitation.reset_insertion \
        --target-mm 25.0 --live 2>&1 | grep -v "^\[DEBUG\]" | tail -6

      echo "=== [$CELL] rep $i/$N_REPS: running ==="
      LOGFILE="/tmp/grid_${RUN_NAME_PREFIX}_${CELL}_rep${i}.log"
      python3 -m proper_research.hardware.online.zshift_grid_toolkit.run_controller_rep \
        --controller "$CONTROLLER" --jacobian "$JACOBIAN" \
        --plan-dir "$PLAN_DIR" --sj-schedule "$SJ_SCHEDULE" --fj-schedule "$FJ_SCHEDULE" \
        --out-dir "$OUT_DIR" --run-name-prefix "$RUN_NAME_PREFIX" \
        --rep "$i" --zraise-mm "$ZRAISE_MM" --insertion-tol-mm "$INSERTION_TOL_MM" \
        > "$LOGFILE" 2>&1

      if grep -qE "stop_reason\s*:\s*path_complete" "$LOGFILE"; then
        echo "=== [$CELL] rep $i/$N_REPS: path_complete OK ==="
        OUTCOMES+=("$CELL rep$i: path_complete")
      else
        STOP_REASON=$(grep -oE "stop_reason\s*:\s*.*" "$LOGFILE" | head -1)
        echo "=== [$CELL] rep $i/$N_REPS: DID NOT COMPLETE CLEANLY (${STOP_REASON:-no stop_reason found}) ==="
        tail -15 "$LOGFILE"
        OUTCOMES+=("$CELL rep$i: ${STOP_REASON:-UNKNOWN/ERROR, see $LOGFILE}")
      fi
    done
  done
done

echo ""
echo "=== 2x2 grid ($RUN_NAME_PREFIX) all reps attempted -- summary ==="
for o in "${OUTCOMES[@]}"; do
  echo "  $o"
done
