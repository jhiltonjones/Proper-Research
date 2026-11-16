# K2 instrumented diagnostic repeat — operator procedure (2026-11-16, consolidated)

## Authorization and status — read first

- **`protocol_campaign2.yaml` remains DRAFT / unsigned.** Nothing in this investigation changed that.
- **Preparing this one instrumented N=15 diagnostic repeat (the command, pre-flight steps, and
  expected outputs below) is NOT approval to run it, and is absolutely not approval of the broader
  campaign.** It is preparation only.
- **Author (user) approval is still required before this repeat is actually executed on hardware.**
  Nothing in this document substitutes for that approval.
- **Both original K2 STOP results remain the official record**, unchanged by anything here:
  N=15 — 4/5 folds vote d=2 but `G_2` never exceeds 0.9 on any fold; N=3 — unanimous d=1.
  `delay_samples=2` and the stop rule are unchanged throughout this entire investigation.
- **This document supersedes all earlier handoff drafts.** It is the one current, standalone
  procedure — no amendment history is mixed into the instructions below. A revision history appendix
  is at the very end, for traceability only; nothing in it is operative.

---

## 1. What this repeat is for

A single supervised, instrumented repeat of the original N=15 K2 run, using newly added logging that
the original two runs did not have: per-cycle streamer events (real `servo_j()` timing, not the
configured interval), an atomically-paired joint-sample timestamp (so sample-age claims are provably
about the exact sample the controller used, not a separately-fetched one), and real code-commit
provenance. It collects diagnostic evidence only.

## 2. Exact command

Identical settings to the original N=15 K2 run (verified against its own recorded `argv`); only
`--run-name` differs.

```
python3 -m proper_research.hardware.online.rectangle_stage_a.run_mpc_delay_aware_insertion_anchor \
  --plan-dir plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/time_parameterized_configuration_path \
  --out-dir close_loop_logs/myrun \
  --schedule-cache plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/jacobian_schedule_insertion_anchor.npy \
  --run-name k2_instrumented_repeat_$(date -u +%Y%m%dT%H%M%SZ) \
  --horizon 15 \
  --max-tracking-error-mm 7 \
  --tcp-workspace-robust-margin-mm 15
```

**Plan/schedule identity:** these paths point to `plans/rectangle_10x15mm_skipglobal_z4b_2026-11-11/`
— the same plan the two original K2 runs used, **not** the separate
`plans/rectangle_10x15mm_skipglobal_z4b_marginfix_2026-11-11/` plan (built earlier in this session for
an unrelated live-arm acceleration-margin problem; that plan is not used anywhere in this repeat).
Content-verified for the plan file: live SHA-256 of `time_parameterized_configuration_path.npz` is
`62cf493b0dc48518985e19fb09c97021de8434b7ca2a92a3a7a4229602453613`, an exact match to
`protocol_campaign2.yaml`'s own recorded `rectangle_mpc_sj_n15.plan_sha256`. This confirms the file
hasn't changed since the protocol was built; it does **not** confirm identity against either original
K2 run's actual input, since neither original run recorded a hash (they predate this investigation's
provenance work). No historical hash exists anywhere for the schedule file
(`jacobian_schedule_insertion_anchor.npy`) to compare against — stated plainly, not assumed.

## 3. Required setup and pre-flight

1. Manual teach-pendant pose check, operator-supervised — no scripted route to extreme poses exists in
   this codebase.
2. Pre-motion workspace-box confirmation — read `WORKSPACE_XYZ_MIN_M`/`MAX_M` from `common.py` and
   visually confirm against the physical rig before any motion.
3. Pre-flight config-mismatch test against `protocol_campaign2.yaml` (already built and verified:
   12/12 deliberately-wrong values correctly refused).
4. Same plan, schedule, horizon=15, and limits as the original N=15 run (above) — this is a diagnostic
   repeat, not a new condition.

## 4. Expected startup confirmation (check within the first few seconds)

Console lines that should appear, confirming the new instrumentation is actually active (not just that
the run started):
- `[path] background arm-streaming thread started (event log -> <out_dir>/arm_streamer_events.jsonl)`
- `[insertion_anchor] code provenance: commit=<hash or None> dirty=<True/False/None> error=<...>` — if
  you've committed the staged changes (Section 7) before running, expect a real commit hash and
  `dirty=False`; if run before committing, expect the current HEAD hash with `dirty=True`.
- `[insertion_anchor] wrote controller_metadata.json -> ...`
- `[insertion_anchor] beam-base check PASSED: ...`

**Verified (not assumed) that this actually fires for this exact command:** `run_mpc_delay_aware_
insertion_anchor.py` line 335 unconditionally monkeypatches `pf.build_offline_solver =
_wrapped_build_offline_solver` at module load — the function containing the `code_provenance`/
`plan_file_sha256`/`schedule_cache_sha256` writes is the one `pf.main()` actually calls for this
controller (`controller_kind="mpc_delay_aware"`, confirmed from the original runs' own
`summary.json`), not a dead or unused code path.

**First logged row check** — within seconds of the run starting:
```
python3 -c "import json; print(json.loads(open('<out_dir>/path_follow.jsonl').readline()).keys())"
```
Confirm `camera_vision_age_s`, `joint_sample_age_at_use_s`, `joint_sample_age_at_log_s`, and
`robot_acquisition_age_s` (the last should be `null`) are all present.

## 5. Preserving the run directory after shutdown

Expected files: `controller_metadata.json`, `summary.json`, `path_follow.jsonl`,
`path_follow_plot.png`, `predicted_beam_positions.jsonl`, `startup_stationary_tips.jsonl`,
`tip_trajectory.csv`, `frozen_jacobian.json`, and `arm_streamer_events.jsonl` (new).

`BackgroundArmStreamer.stop()` blocks until its writer thread has fully drained the event queue and
closed the file (confirmed by a real regression test, Section 6) — the run directory is safe to copy
immediately after a **normal** shutdown (the run reaching its own `stop()` call). This guarantee does
**not** apply if the process is killed abruptly instead of via its own shutdown path.

## 6. Verify the work yourself — exact test commands

Run these directly; no hardware, no robot, no camera required.

```
python3 proper_research/hardware/online/test_close_loop_path_follow_streamer.py
```
Expected: 3 checks pass — fast-servo_j rate (~51Hz vs. 50Hz configured), slow-servo_j degraded rate
(~25Hz realized, correctly below the 50Hz configured rate), and orderly shutdown (50/50 events survive
`stop()`). Ends with `=== ALL ASSERTIONS PASSED ===`.

```
python3 proper_research/hardware/online/test_atomic_joint_snapshot.py
```
Expected: 3 checks pass — `LatestSlot` atomicity under a simulated concurrent update; the old
two-call race reproduced and shown NOT to occur in the new atomic path; age correctly growing
(0 → 0.037s → 0.100s) from one held timestamp. Ends with `=== ALL ASSERTIONS PASSED ===`.

## 7. Files changed, complete list, and staging

**Every file touched across this entire investigation** (11 files):
```
proper_research/studies/mpc_ablations/k2_delay_and_step_analysis.py
proper_research/hardware/online/close_loop_path_follow.py
proper_research/hardware/online/state_stream.py
proper_research/hardware/online/messages.py
proper_research/hardware/online/camera_source.py
proper_research/hardware/online/rectangle_stage_a/common.py
proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py
proper_research/hardware/online/test_close_loop_path_follow_streamer.py
proper_research/hardware/online/test_atomic_joint_snapshot.py
proper_research/hardware/online/K2_REVISED_PROCEDURE_DRAFT.md
proper_research/hardware/online/K2_INSTRUMENTED_REPEAT_HANDOFF.md
```

**This list is derived entirely from my own edit history. I cannot run `git status` or `git diff`
against the canonical checkout to verify it — every git command targeting
`/home/jack/Proper-Research` is refused outright from this session, confirmed repeatedly throughout
this investigation. Treat this as my best reconstruction, not a git-verified fact.**

Staging and commit (run yourself):
```
git -C /home/jack/Proper-Research status --short
git -C /home/jack/Proper-Research add \
  proper_research/studies/mpc_ablations/k2_delay_and_step_analysis.py \
  proper_research/hardware/online/close_loop_path_follow.py \
  proper_research/hardware/online/state_stream.py \
  proper_research/hardware/online/messages.py \
  proper_research/hardware/online/camera_source.py \
  proper_research/hardware/online/rectangle_stage_a/common.py \
  proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py \
  proper_research/hardware/online/test_close_loop_path_follow_streamer.py \
  proper_research/hardware/online/test_atomic_joint_snapshot.py \
  proper_research/hardware/online/K2_REVISED_PROCEDURE_DRAFT.md \
  proper_research/hardware/online/K2_INSTRUMENTED_REPEAT_HANDOFF.md
git -C /home/jack/Proper-Research commit -F - <<'EOF'
Add K2 timing/delay diagnostic instrumentation and repair provenance gaps

- BackgroundArmStreamer: log per-cycle publish/pickup/servo_j events via a
  queue-drained writer thread (new arm_streamer_events.jsonl), closing the
  gap where last_servo_ms/calls were computed and silently discarded.
  target_update_ms's original meaning is preserved (event-log overhead
  measured and confirmed excluded, not just asserted).
- Atomic joint-sample snapshot: LatestSlot.get_with_timestamp() and
  RobotJointStream.latest_joints_with_timestamp() return (value, stamp)
  from one lock acquisition; StateEstimate carries robot_joints_stamp
  paired with robot_joints via a new opt-in CameraSource parameter
  (legacy robot_joints_getter unchanged for its other 3 callers). Fixes a
  real atomicity gap where a joint-sample age was computed from a
  separate, later stream read than the one that produced q_meas_rad.
  Logged as joint_sample_age_at_use_s / joint_sample_age_at_log_s.
  robot_acquisition_age_s stays None -- not fabricated.
- common.py: add capture_code_provenance() (commit hash + dirty flag +
  source-file hash fallback), wired into controller_metadata.json along
  with plan/schedule file hashes -- neither existing K2 run recorded a
  commit hash at all; this closes that gap for future runs (the two
  existing runs remain permanently without it).
- k2_delay_and_step_analysis.py: one reproducible script for the 5-fold
  delay check and the step-realization ratio, using the exact formula
  documented in HOWTO_CAMPAIGN2.md (same-tick pairing,
  max(|commanded|, 1e-12) denominator) -- reproduces the official N=15
  numbers exactly.
- No change to delay_samples or the stop rule. Both existing K2 runs
  (N=15, N=3) remain official STOP results. protocol_campaign2.yaml
  remains DRAFT/unsigned.
EOF
```

## 8. No-drift confirmation

- N=15 command: unchanged from Section 2 above, same as every prior pass.
- Controller settings/weights/limits: unchanged.
- `delay_samples`: unchanged, still `2`.
- Stop rule (`G_2 > 0.9` on every fold, `>=4/5` folds voting d=2): unchanged.

---

## Revision history (traceability only — not operative)

- 2026-11-11 to 2026-11-14: K2 branch-confusion correction (streamer timing analysis, step-realization
  grid-search method tried and explicitly rejected), initial instrumentation (streamer events,
  provenance repair, `k2_delay_and_step_analysis.py`).
- 2026-11-15: exact documented step-realization formula implemented (reproduces N=15 exactly);
  `joint_sample_host_age_s` added (later found non-atomic); `target_update_ms` contamination found and
  fixed, measured.
- 2026-11-16: atomic joint-sample snapshot implemented (`messages.py`, `camera_source.py`,
  `state_stream.py`), replacing the non-atomic `joint_sample_host_age_s` with the paired
  `joint_sample_age_at_use_s`/`joint_sample_age_at_log_s`; plan/schedule identity for the repeat
  resolved; this document consolidated into one standalone procedure.
