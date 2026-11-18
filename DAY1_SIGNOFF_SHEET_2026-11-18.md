# Day-1 Pilot Sign-Off Sheet — RECTANGLE RIG ONLY

**Date**: 2026-11-18
**Scope**: this entire document covers the RECTANGLE rig only. This session's work —
every run analyzed, the frame fix, the delay-model assessment, the startup-freeze
investigation — was exclusively on the rectangle rig
(`run_mpc_delay_aware_insertion_anchor.py`, `plans/rectangle_10x15mm_...`). **The
vessel rig's N=15/N=3 STOP conditions were never touched, re-run, or re-verified
this session** and are explicitly **out of scope** for this document. This sign-off
sheet cannot be used to approve, or as evidence toward, vessel-rig Day-1 readiness.
**Status**: rectangle-rig technical verification substantially complete; several
explicit open decision points remain (Section 6).

## 1. Exact code state

All applied and tested this session, canonical checkout (`/home/jack/Proper-Research`):

- `proper_research/hardware/online/state_stream.py` — frame fix
  (`_DEFAULT_T_ROBOT_BEAM_POSE6` Z: `-0.016567` -> `float(BEAM_BASE_XYZ_M[2])`
  = `-0.039627`), diagnostic logging integration, restored
  `latest_joints_with_age`/`latest_joints_with_timestamp` methods.
- `proper_research/hardware/online/close_loop_path_follow.py` — construction-phase
  cleanup, `--enable-reader-diagnostic-log` CLI wiring.
- `proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py`
  — same diagnostic-log CLI flag wired through.
- `proper_research/hardware/online/servo_diagnostic_logging_prep.py` — diagnostic log
  sink (`DiagnosticLogSink`).
- `proper_research/hardware/online/advancer_excitation/measure_l0.py` —
  `apply_affine_insertion_correction()` (a=0.774063, b=1.535419) — an empirical
  length-correction layer, now superseded in importance by the 3D frame fix but still
  live in the measurement chain.
- `proper_research/hardware/online/rectangle_stage_a/common.py` — `check_camera_healthy`'s
  `auto_retract` behavior.
- `proper_research/vision/insertion_check.py`, `proper_research/vision/insertion_reset.py`
  — converted from vessel-rig to rectangle-rig tools.
- `calibration_points.json` — reverted to the original 2026-09-14 values.
- New/updated test files: `test_check_camera_healthy_auto_retract.py`,
  `test_robot_joint_stream_diagnostic_integration.py`,
  `test_servo_diagnostic_logging_prep.py`, `test_construction_phase_cleanup.py`.

**All 29 tests pass** (3 auto-retract + 9 joint-stream-diagnostic + 12
servo-diagnostic + 5 construction-cleanup), confirmed repeatedly this session, most
recently after the frame fix.

## 2. Exact environment

From this session's own real logged runs (`robot_joint_stream_diagnostics.jsonl`'s
`startup` event — not inferred from the current live shell):

- Python interpreter: `/usr/bin/python3`
- `ur_rtde` version: `1.6.0`
- `rtde_control`: `/home/jack/.local/lib/python3.10/site-packages/rtde_control.cpython-310-x86_64-linux-gnu.so`
- `rtde_receive`: `/home/jack/.local/lib/python3.10/site-packages/rtde_receive.cpython-310-x86_64-linux-gnu.so`
- `git_commit` recorded in every analyzed run's `controller_metadata.json`:
  `8304c8f140e4b76a67d317804b5c9782e3642522`, with `git_dirty=true` on every run
  checked this session (uncommitted changes were present at every run time — this is
  a factual observation, not itself a pass/fail signal; see Section 6 for the
  consolidated-commit decision this implies).

## 3. Protocol — exact identities

- **Plan**: `plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path`
  — `plan_file_sha256 = 4f00089bf3592bdc9c1f4bdd74827de221fc3152aeb7ed5665499299db443274`
  (full hash, confirmed identical across all analyzed runs).
- **Schedule cache**: `plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy`
  — `schedule_cache_sha256 = f9af9acdcff171405e6a22a23fa18e36b7106f5ad2f5ade4b5f2ae800e562dee`
  (full hash, confirmed identical across all analyzed runs).
- **Command**:
  ```
  python3 proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py \
    --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path \
    --out-dir close_loop_logs/myrun \
    --schedule-cache plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy \
    --run-name <new_run_name> --horizon 15 --max-tracking-error-mm 7 \
    --tcp-workspace-robust-margin-mm 15 --enable-reader-diagnostic-log
  ```
- **Startup checks to watch for**:
  - Single-threaded BLAS confirmation in worker startup output
    (`[insertion_anchor] worker ready...`).
  - `[health] insertion length: measured=... expected=...` — if mismatched beyond
    tolerance, auto-retract now triggers automatically
    (`[health] insertion mismatch ... auto-retract triggered`); confirm it converges,
    don't just watch the first pass/fail line.
  - `[path] planner->R fit: |t_fit|=...` — **expect a single-digit mm norm, with the
    Z-component specifically near 0.096mm.** This is confirmed exactly on 4/4 live
    post-fix hardware runs plus the original offline recomputation (5/5 total,
    0.096mm every time). A Z-component materially different from ~0.1mm on a new run
    would be a real anomaly worth stopping to investigate, not proceeding past.
- **Expected output files**: `controller_metadata.json`, `summary.json`,
  `path_follow.jsonl`, `arm_streamer_events.jsonl`,
  `robot_joint_stream_diagnostics.jsonl`, `startup_stationary_tips.jsonl`,
  `predicted_beam_positions.jsonl`, `tip_trajectory.csv`, `frozen_jacobian.json`,
  `path_follow_plot.png`.

## 4. Runner-specific checks (`run_mpc_delay_aware_insertion_anchor.py`)

- `worker.require_frame_transform()` gate — confirms the worker refuses to solve
  until the live registration has actually been applied; there should never be a
  solve before the `[path] planner->R fit:` line appears in the log.
- `code_provenance.source_file_sha256` in `controller_metadata.json` — spot-check
  against the canonical checkout's current file hashes if exact reproducibility
  matters for a given run (this session found, and relied on, this field to detect a
  real mid-day source edit between two runs).
- Diagnostic log's `shutdown` event (`drained_cleanly`, `dropped_count`,
  `write_failure_count`) — confirm `drained_cleanly=true` and both counts `0` before
  trusting a run's completeness.
- Watch for `duplicate_or_stall="sustained_timestamp_stall"` in
  `robot_joint_stream_diagnostics.jsonl`. This occurred in 1 of 4 post-fix runs
  analyzed this session. It was traced, root-caused (an RTDE stale-packet condition),
  and confirmed **structurally non-hazardous** to control-state initialization — the
  production joint-read path has its own change-detection guard
  (`RobotJointStream._poll_body()`) that already protects `q0` from exactly this
  failure mode, independently verified and already covered by existing passing tests
  (`test_actual_q_diagnostic_read_never_reaches_control_slot`,
  `test_diagnostic_labels_never_touch_existing_stale_repeat_logic`). Its recurrence
  rate (1-in-4 observed) is still worth tracking operationally, but it is not a
  blocker.

## 5. Everything verified this session — specific evidence, not general claims

- **Frame/pivot fix**: `t_fit_m[2]` (registration Z-component) landed at **0.096mm
  exactly on 5/5 independent checks** — 1 offline recomputation through the actual
  patched `_fit_planner_to_robot()` code path, plus 4 live post-fix hardware runs
  (`093013Z`, `094425Z`, `100151Z`, `100359Z`). All 29 tests pass after the change.
- **Delay-model adequacy (`delay_samples=2`)**: on runs A (`093013Z`) and B
  (`094425Z`), held-out RMSE at d=1 (100ms) was ~2.6x worse than at d=2/d=3 on both
  runs (A: 0.00342 vs 0.00128/0.00133; B: 0.00336 vs 0.00129/0.00125 rad) — d=1
  clearly ruled out. d=2 vs d=3 differ by only 3-6%, with the better one flipping
  between the two runs. Cross-run test: A's fitted 250ms lag applied with no
  refitting to B's full active data gave RMSE=0.000504 rad, better than B's own
  internal held-out self-evaluation (0.000685 rad). Recommendation: retain
  `delay_samples=2`; no controller parameter change proposed.
- **Startup-freeze safety path**: traced to the code level, not inferred. The
  production joint-read path (`RobotJointStream._poll_body()`) only refreshes its
  freshness timestamp when `getActualQ()`'s raw reading actually changes
  (`np.array_equal` guard); on an exact repeat it increments `stale_repeats` and
  leaves the old timestamp in place, so `max_age_s` correctly ages out a frozen
  stream. The diagnostic log's own `actual_q_diagnostic_second_read` (where the
  503-tick freeze was observed in run `093013Z`) is a separate, observation-only
  read that never reaches `joints_slot` — confirmed both by code and by the existing
  passing tests `test_actual_q_diagnostic_read_never_reaches_control_slot` and
  `test_diagnostic_labels_never_touch_existing_stale_repeat_logic`.
- **Logging completeness, per run** (`robot_joint_stream_diagnostics.jsonl`'s own
  `shutdown` event, all 4 post-fix runs): `dropped_count=0`, `write_failure_count=0`,
  `drained_cleanly=true` on every run; `diag_sustained_stalls` was 484 on `093013Z`
  and 0 on the other 3.
- **Tracking statistics, reconciled by evaluation window, per run** (active-window
  vs terminal-hold reported separately, not blended):

| run | active mean / p95 / max (mm) | terminal-hold mean / p95 / max (mm) |
|---|---|---|
| `093013Z` | 0.353 / 0.828 / 1.205 | 0.928 / 1.116 / 1.126 |
| `094425Z` | 0.471 / 1.239 / 2.024 | 2.114 / 2.347 / 2.383 |
| `100151Z` | (active/overall mean 0.364, p95 0.704, max 1.041 — not split by window this round) | (not split by window this round) |
| `100359Z` | (active/overall mean 0.615, p95 2.019, max 2.255 — not split by window this round) | (not split by window this round) |

  Runs `100151Z`/`100359Z` were checked for completion/registration/freeze status
  only; their active-vs-terminal-hold split was not separately computed this
  round — flagged honestly rather than presented as if it were.

## 6. Finite checklist of remaining author decisions — not just the tracking threshold

The following are ALL still open and require the user's decision — the tracking
threshold is one of several, not the only item:

1. **Tracking-performance acceptance criterion: ______________________ (PENDING).**
   No approved numeric standard exists. `--max-tracking-error-mm 7` is a safety abort
   threshold, not a performance bar. Observed (active-window, 2 of 4 runs split this
   way): mean 0.35-0.47mm, p95 0.7-1.2mm, max 1.2-2.0mm.
2. **Paper-wording replacement for `methodology.tex:309`** ("measured delay of d=2...
   identified") — a specific proposed replacement sentence was drafted in this
   session's consolidated report; not yet approved or applied to the paper.
3. **X/Y registration components — characterization status: OPEN, not deferred to a
   plan.** Observed 0.9-4.6mm across the 4 post-fix runs, with no investigated
   source. This is a genuine open question, not something intentionally set aside
   with a plan to revisit — flagging the distinction explicitly since "deferred"
   would overstate how settled this is.
4. **Vessel-rig scoping (Section "Scope" above)**: confirm this rectangle-only
   sign-off is accepted as the basis for a rectangle-only Day-1, with vessel-rig
   Day-1 readiness explicitly tracked as a separate, unaddressed question.
5. **Delay-model choice (`delay_samples=2`)**: this document recommends retaining it,
   but that recommendation itself needs the user's explicit sign-off before being
   treated as settled for campaign purposes — it has not been adopted unilaterally.
6. **`git_dirty=true` on every analyzed run**: whether a clean, committed state is
   required before Day-1, or whether running against uncommitted changes is
   acceptable for this phase, is undecided.

**None of these are filled in by this analysis.** Items 1-6 together, not just item
1, stand between "everything technical is verified" and "Day-1 approved."

## 7. Executable preparation commands

Consolidated from everything established this session — for an operator preparing to
run the validation procedure and/or a Day-1 run (none executed by this analysis):

```bash
# Confirm canonical checkout is on the expected commit / check dirty status
git -C /home/jack/Proper-Research status
git -C /home/jack/Proper-Research rev-parse HEAD   # expect 8304c8f1... unless a new commit has landed since

# Confirm the 29-test suite still passes before any run
cd /home/jack/Proper-Research
python3 -m pytest proper_research/hardware/online/rectangle_stage_a/test_check_camera_healthy_auto_retract.py -q
python3 -m proper_research.hardware.online.test_robot_joint_stream_diagnostic_integration
(cd proper_research/hardware/online && python3 test_servo_diagnostic_logging_prep.py)
(cd proper_research/hardware/online && python3 test_construction_phase_cleanup.py)

# Confirm plan/schedule hashes match the expected values before using them
sha256sum plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path/time_parameterized_configuration_path.npz
sha256sum plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy

# The Day-1 hardware command itself (Section 3) -- run only after 1-6 above are
# addressed and the operator is physically present and supervising
python3 proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/jacobian_schedule_insertion_anchor.npy \
  --run-name <new_run_name> --horizon 15 --max-tracking-error-mm 7 \
  --tcp-workspace-robust-margin-mm 15 --enable-reader-diagnostic-log
```
